// Gemma 4 device pieces (docs/gemma4-plan.md 3.4, 3.5; task M1-15). Same plain C ABI as kernels.h:
// device pointers and the stream as int64_t, shapes/strides in ELEMENTS, bf16 in/out with fp32
// math. Implemented in src/kernels/src/gemma_kernels.hip (and r4dx_gelu_tanh_mul_trellis_bf16 in
// trellis_transform.hip, next to its silu twin), built with exactly r4dx_kernels.hip's flags. Every
// entry point counts into r4dx_kernel_launch_counter (after its rows <= 0 early return).
//
// The per-element conventions are Gemma 4's HF modeling code (docs/gemma4-plan.md 2 / 3.1); the
// ones the plan lists as UNVERIFIED (v_norm placement, embed scale rounding, layer_scalar site) are
// parameters of the callers here, not baked in, and each is pinned by a rung-3 golden.
#pragma once

#include <cstdint>

extern "C" {

// ---- proportional / partial rope ---------------------------------------------------------------
// In-place NeoX rotate-half rope over pairs (i, i + head_dim/2) for i in [0, n_pairs); dims of a head
// that are not in a rotated pair (pairs n_pairs..head_dim/2-1 and their partners) pass through
// unchanged. Pair i rotates by angle pos * theta^(-2i / freq_dim). Same expression order as
// r4dx_rope_neox_bf16 (exp2f/log2f, one multiply, accurate sincosf; fp32 math, one rounding to
// bf16), so n_pairs = head_dim/2 with freq_dim = head_dim is r4dx_rope_neox_bf16 bit for bit.
//   Gemma 4 full layer   (rope "proportional"): head_dim 512, n_pairs 64, freq_dim 512, theta 1e6 --
//     HF's inv_freq is 1/theta^(2i/512) for i < 64 and ZERO for 64..255, and a zero frequency is
//     the identity rotation, which is exactly "do not touch the dim".
//   Gemma 4 sliding layer (rope "default"):      head_dim 256, n_pairs 128, freq_dim 256, theta 1e4.
// q: [tokens, heads_q, head_dim] bf16, k: [tokens, heads_k, head_dim] bf16, both in place; pass q = 0
// with heads_q = 0 (or k = 0 with heads_k = 0) to rotate one of them. pos: [tokens] int32 DEVICE.
// Preconditions (throw): head_dim even and <= 1024, 1 <= n_pairs <= head_dim/2, freq_dim >= 2 and even.
void r4dx_rope_proportional_bf16(int64_t q, int64_t k, int64_t pos, int tokens, int heads_q,
                                  int heads_k, int head_dim, int n_pairs, int freq_dim, float theta,
                                  int64_t stream);

// ---- rmsnorm without a weight ------------------------------------------------------------------
// out[row,:] = bf16(x[row,:] * rsqrt(mean(x[row,:]^2) + eps)), fp32 math (Gemma4RMSNorm with
// with_scale=False: v_norm). Rows are the caller's: for v_norm over [T, kv_heads, head_dim], rows =
// T * kv_heads and hidden = head_dim. In place (out == x) is supported (the reduction's barrier sits
// between every read and every write of a row). x, out: [rows, hidden] bf16.
void r4dx_rmsnorm_noscale_bf16(int64_t x, int64_t out, int64_t rows, int64_t hidden, float eps,
                                int64_t stream);

// ---- GeGLU ---------------------------------------------------------------------------------------
// The fused mlp.gate_up layout of r4dx_silu_mul_bf16 (gate_up[r, 0..I) = gate, [I..2I) = up, row
// stride in_row_stride >= 2I), with the tanh GELU of `gelu_pytorch_tanh`:
//   out[r,i] = bf16( float(bf16(gelu_tanh(gate))) * up )
// The activation is rounded to bf16 BEFORE the multiply -- the HF op sequence (`act_fn(gate) * up`,
// two bf16 tensor ops) -- unlike silu_mul's single rounding. out: [rows, I] bf16, contiguous.
// r4dx_gelu_tanh_mul_bf16: one workgroup per row; _wide: grid (rows, ceil(I / 2048)), bit-identical.
void r4dx_gelu_tanh_mul_bf16(int64_t gate_up, int64_t out, int64_t rows, int64_t intermediate,
                              int64_t in_row_stride, int64_t stream);
void r4dx_gelu_tanh_mul_wide_bf16(int64_t gate_up, int64_t out, int64_t rows, int64_t intermediate,
                                   int64_t in_row_stride, int64_t stream);
// The trellis producer (r4dx_silu_mul_trellis_bf16's twin, docs/trellis-kernel.md 4.8): the bf16
// value above fed straight into the trellis input transform in one launch, BYTE-IDENTICAL to the v1
// pair (r4dx_gelu_tanh_mul_wide_bf16 then r4dx_trellis_input_bf16 with nout = 1). suh fp32
// [intermediate]; out f16 [rows, intermediate]. Same preconditions as r4dx_silu_mul_trellis_bf16.
void r4dx_gelu_tanh_mul_trellis_bf16(int64_t gate_up, int64_t rows, int64_t intermediate,
                                      int64_t in_row_stride, int64_t suh, int64_t out,
                                      int prescale_log2, int64_t stream);

// ---- scaled embedding gather ----------------------------------------------------------------------
// out[i,:] = bf16(float(table[ids[i],:]) * scale), the device counterpart of Gemma4's
// ScaledWordEmbedding (`embedding(ids) * embed_scale.to(weight.dtype)`): HF multiplies the bf16 row by
// a bf16 scalar, so for parity the CALLER passes scale already rounded to bf16 (sqrt(3840) = 61.968
// becomes 62.0). Same bounds guard as r4dx_embedding_gather_bf16 (an id outside [0, vocab) reads
// row 0). table: [vocab, hidden] bf16. ids: [n] int32 DEVICE. out: [n, hidden] bf16.
void r4dx_embedding_gather_scaled_bf16(int64_t table, int64_t ids, int64_t out, int64_t n,
                                        int64_t hidden, int64_t vocab, float scale, int64_t stream);

// ---- sandwich post-norm + residual (+ next pre-norm) ---------------------------------------------------
// One Gemma 4 sublayer's tail, fused:
//   normed  = bf16( y * rsqrt(mean(y^2) + eps) * w_post )          (plain weight, NOT 1 + w)
//   sum     = bf16( res + normed )
//   out_res = bf16( sum * scalar )                                  (skipped when scalar == 1.0f)
//   out_normed = bf16( out_res * rsqrt(mean(out_res^2) + eps) * w_next )   (only when out_normed != 0)
// i.e. each step is rounded to bf16 exactly where the HF module chain rounds. `scalar` is the
// layer's `layer_scalar` and applies ONCE per layer, after the MLP residual add: the attention-half
// call MUST pass 1.0f, the MLP-half call passes layer_scalar (a bf16-representable value, read to the
// host at load). w_next / out_normed are the NEXT layer's input_layernorm (or the final norm) so the
// following sublayer's input is produced in the same launch; pass 0 for both to skip it.
// res and out_res may alias (in-place residual update); y must not alias them.
// res, y, out_res, out_normed: [rows, hidden] bf16. w_post, w_next: [hidden] bf16.
// One workgroup per row; scalar loops (a perf pass is M1-34).
void r4dx_gemma_postnorm_residual_rmsnorm_bf16(int64_t res, int64_t y, int64_t w_post,
                                                int64_t w_next, float eps, int64_t out_res,
                                                int64_t out_normed, float scalar, int64_t rows,
                                                int64_t hidden, int64_t stream);

// ==== fp32-residual variants (GemmaModel's default residual dtype; R4DX_GEMMA_RESID=bf16 uses the
// kernels above) ============================================================================================
// HF-bf16 is not a stable yardstick for Gemma 4 (the residual reaches 100-300 and bf16 rounding is
// amplified), so r4dx keeps the residual stream in fp32 and only the GEMM inputs (norm outputs) are bf16.
// The residual is [rows, hidden] fp32; sublayer outputs y stay bf16 (GEMM outputs). Under TP the later
// all-reduce operates on those bf16 sublayer outputs, never on the residual.

// out fp32 = float(table[ids[i],:]) * scale (scale = bf16(sqrt(hidden)), 62.0 for 3840; the product is exact).
void r4dx_embedding_gather_scaled_f32(int64_t table, int64_t ids, int64_t out, int64_t n, int64_t hidden,
                                       int64_t vocab, float scale, int64_t stream);

// out bf16 = bf16(x * rsqrt(mean(x^2) + eps) * w), x fp32 [rows, hidden], w bf16 [hidden]. The pre-norms and
// the final norm. Not in place.
void r4dx_rmsnorm_plain_f32in_bf16(int64_t x, int64_t weight, int64_t out, int64_t rows, int64_t hidden,
                                    float eps, int64_t stream);

// r4dx_gemma_postnorm_residual_rmsnorm_bf16 with an fp32 residual, all in fp32 with no intermediate bf16
// rounding:  out_res = (res + y * rsqrt(mean(y^2)+eps) * w_post) * scalar   (scalar == 1 skips the multiply)
//            out_normed (bf16, optional) = bf16(rms(out_res) * w_next)
// res, out_res fp32 (may alias); y bf16; out_normed bf16.
void r4dx_gemma_postnorm_residual_rmsnorm_f32res(int64_t res, int64_t y, int64_t w_post, int64_t w_next,
                                                  float eps, int64_t out_res, int64_t out_normed, float scalar,
                                                  int64_t rows, int64_t hidden, int64_t stream);

// out[i] = bf16(x[i]) for n elements (RNE). Used for the drafter feature capture.
void r4dx_f32_to_bf16(int64_t x, int64_t out, int64_t n, int64_t stream);

}  // extern "C"
