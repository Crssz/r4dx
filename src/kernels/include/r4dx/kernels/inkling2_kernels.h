// Inkling2 (inkling2_loop) device pieces (docs/inkling2-plan.md 3.3, 5). Same plain C ABI as lfm2_kernels.h:
// device pointers and the stream as int64_t, shapes / strides in ELEMENTS, fp32 math. Implemented in
// src/kernels/src/inkling2_kernels.hip (its own hipcc object, r4dx_kernels.hip's flags, -ffp-contract=off). The
// SPEC of every entry point is the CPU reference r4dx/kernels/inkling2_ref.hpp (HF semantics); the kernels evaluate
// the same expressions, with block-reduction sum orders (the GPU tests compare with a tolerance). Every entry point
// counts one launch into r4dx_kernel_launch_counter (after its early return for empty work) and throws
// std::runtime_error on a violated precondition (callers: /EHc- or the model's own checks).
//
// REUSED, no new kernel (the math matches HF exactly):
//   embedding + embed_norm      r4dx_embedding_gather_bf16 + r4dx_rmsnorm_plain_bf16 (out_fp32 = 1): h fp32
//   attn_norm / q,k,v_input_norm / mlp_norm / final norm on the fp32 residual
//                               r4dx_rmsnorm_plain_f32in_bf16 (bf16 out = the GEMM input, plain weight w)
//   fp32 -> bf16                r4dx_f32_to_bf16
//   MoE sort / gather / combine r4dx_lfm2_moe_sort, r4dx_lfm2_moe_gather_bf16, r4dx_lfm2_moe_combine_bf16 (fp32
//                               combine in slot order, one bf16 rounding; <= 256 experts)
// NEW (below): the attention residual, the static and dynamic short convolutions (with the fp32 residual add fused),
// the per-head clip / rms (optionally writing into the paged bf16 KV cache), the relative-bias bank mix, the sigmoid
// output gate, the message gate, the sigmoid top-k router (Inkling's 1e-20 and up to 256 experts: LFM2's router has
// 1e-6 and <= 64), the softcapped SwiGLU, the final logit softcap, and the reference attention with bias + sink.
#pragma once

#include <cstdint>

extern "C" {

// ---- attention residual (inkling2_ref::AttnRes) --------------------------------------------------------------------
// out[q][t][:] = sum_i softmax_i((queries[q] . rms(S_i[t])) / sqrt(H)) * rms(S_i[t]),  S_i = states[i] (fp32 [T, H],
// row stride H). `states`: a HOST array of n_states device pointers (1..6), in stack order (blocks..., partial).
// queries: fp32 [nq, H], nq 1..4. out: fp32 [nq, T, H]. eps: 1e-6. One workgroup per token. H <= 8192.
void r4dx_inkling2_attnres_f32(const int64_t* states, int n_states, int64_t queries, int nq, int64_t out, int T,
                               int H, float eps, int64_t stream);

// ---- static short convolution (inkling2_ref::SconvStatic) ---------------------------------------------------------
// out_t = [residual_t +] (w0 x_{t-3} + w1 x_{t-2} + w2 x_{t-1} + w3 x_t + x_t) / sqrt(5), per channel, fp32.
// x: [T, C] bf16 (x_bf16 != 0) or fp32, row stride C. weight: fp32 [C, 4] (tap 3 = current). state: fp32 [3, C]
// (x_{t-3}, x_{t-2}, x_{t-1} before the chunk), updated IN PLACE to the chunk's last three inputs (a T < 3 chunk
// shifts). residual: fp32 [T, C] or 0. out: fp32 [T, C]; may alias residual (in place h += conv(o)), not x.
// Chunked prefill == decode, bit for bit. One thread per channel walks the chunk.
void r4dx_inkling2_sconv_static_f32(int64_t x, int x_bf16, int64_t weight, int64_t state, int64_t residual,
                                    int64_t out, int T, int C, int64_t stream);

// ---- dynamic short convolution (inkling2_ref::SconvDynamic) -------------------------------------------------------
// tap[t][c][lag] = static_w[lag][c] + dyn[t][head(c) * 4 + lag] (lag 0 = current, head(c) = c / (C / heads));
// out_t = (x_t tap0 + x_{t-1} tap1 + x_{t-2} tap2 + x_{t-3} tap3 + x_t) / 3. x: [T, C] bf16 or fp32. dyn:
// [T, heads * 4] bf16 (dyn_bf16 != 0) or fp32 (= z @ weight_proj^T). static_w: fp32 [4, C]. state, out as above.
// C % heads == 0.
void r4dx_inkling2_sconv_dynamic_f32(int64_t x, int x_bf16, int64_t dyn, int dyn_bf16, int64_t static_w,
                                     int64_t state, int64_t out, int T, int C, int heads, int64_t stream);

// ---- per-head ops (inkling2_ref::HeadClip / HeadRms) ---------------------------------------------------------------
// Where the head vectors go. kind 0: dense, element out[t * out_row_stride + h * d + i] (out_row_stride = heads * d
// when contiguous), bf16 or fp32. kind 1: the paged bf16 KV cache of the reference attention (attn_ref.h's
// layout): element (block_table[p / block_size] * kv_block_stride + h * kv_head_stride + (p % block_size) * 2 * d +
// col_offset + i), p = pos0 + t (col_offset 0 = K, d = V). kind 1 writes bf16 only.
struct R4dxInkling2HeadOut {
  int kind;               // 0 dense, 1 paged KV
  int out_bf16;           // dense only: 1 bf16, 0 fp32
  int64_t out;            // dense: the output; paged: the cache base (uint16)
  int64_t out_row_stride; // dense: elements per token
  int64_t block_table;    // paged: int32
  int block_size;         // paged
  int col_offset;         // paged: 0 (K) or d (V)
  int64_t kv_block_stride, kv_head_stride;  // paged, elements
  int64_t pos_dev;        // optional device int32* holding pos0 (0 = use host pos0); for HIP graph capture
};
// v = x * scale[h] (scale fp32 [heads, d] or 0 = none); out = v * min(1, threshold / (||v|| + 1e-6)), times
// 1 + log_alpha * ln(max((pos0 + t + 1) / log_floor, 1)) when log_alpha != 0. x: [T, heads * d] bf16 or fp32 with row
// stride x_row_stride. d % 4 == 0, d <= 512.
void r4dx_inkling2_head_clip(int64_t x, int x_bf16, int64_t x_row_stride, int64_t scale, float threshold,
                             int64_t pos0, float log_alpha, float log_floor, const R4dxInkling2HeadOut* out, int T,
                             int heads, int d, int64_t stream);
// out = x * (1 / sqrt(mean over the head of x^2 + eps)) (the value norm), same addressing.
void r4dx_inkling2_head_rms(int64_t x, int x_bf16, int64_t x_row_stride, float eps, int64_t pos0,
                            const R4dxInkling2HeadOut* out, int T, int heads, int d, int64_t stream);

// ---- relative-bias bank mix (inkling2_ref::RelLogits) --------------------------------------------------------------
// rel[row][e] = sum_j r[row][j] * proj[j][e] for rows = T * heads. r fp32 [rows, d_rel], proj fp32 [d_rel, extent],
// rel fp32 [rows, extent]. d_rel <= 64.
void r4dx_inkling2_rel_logits_f32(int64_t r, int64_t proj, int64_t rel, int rows, int d_rel, int extent,
                                  int64_t stream);

// ---- sigmoid output gate -------------------------------------------------------------------------------------------
// out[i] = bf16(float(o[i]) * sigmoid(float(g[i]))), n elements; o, g, out bf16 (out may alias o).
void r4dx_inkling2_sigmoid_gate_mul_bf16(int64_t o, int64_t g, int64_t out, int64_t n, int64_t stream);

// ---- message gate (inkling2_ref::MsgGatePrep / MsgGateApply) -------------------------------------------------------
// prep: out[t] = bf16([rms(x[t]) | rms(m[t])]), x, m fp32 [T, H]; out bf16 [T, 2H] (the input of the down GEMM).
void r4dx_inkling2_msg_gate_prep_bf16(int64_t x, int64_t m, int64_t out, int T, int H, float eps, int64_t stream);
// apply: out = (1 - tanh(g @ forget^T)) * x + tanh(g @ gain^T) * m. g: [T, R] bf16 (g_bf16) or fp32; forget, gain
// fp32 [H, R]; x, m, out fp32 [T, H] (out may alias x or m). R <= 256.
void r4dx_inkling2_msg_gate_apply_f32(int64_t x, int64_t m, int64_t g, int g_bf16, int64_t forget, int64_t gain,
                                      int64_t out, int T, int H, int R, int64_t stream);

// ---- router (inkling2_ref::Router) ---------------------------------------------------------------------------------
// logits[e] = x . gate_w[e] (fp32 over bf16 operands); s = sigmoid; top_k of s + bias descending, ties -> lower index;
// w[slot] = s / (sum + 1e-20) * route_scale. x bf16 [T, H] (row stride x_row_stride), gate_w bf16 [E, H], bias fp32
// [E] (or 0), ids int32 [T, top_k], w fp32 [T, top_k], logits_out fp32 [T, E] or 0. 1 <= E <= 256,
// 1 <= top_k <= min(16, E), H <= 8192.
void r4dx_inkling2_router_f32(int64_t x, int64_t gate_w, int64_t bias, int64_t ids, int64_t w, int T, int H, int E,
                              int top_k, float route_scale, int64_t x_row_stride, int64_t logits_out, int64_t stream);

// ---- softcapped SwiGLU (inkling2_ref::SwigluSoftcap) ---------------------------------------------------------------
// out[r, i] = bf16((4 tanh(g/4) * sigmoid(g)) * 25 tanh(u/25)), g = gate_up[r, i], u = gate_up[r, intermediate + i]
// (row stride in_row_stride >= 2 * intermediate); out [rows, intermediate] contiguous. r4dx_silu_mul_bf16's layout.
void r4dx_inkling2_swiglu_softcap_bf16(int64_t gate_up, int64_t out, int64_t rows, int64_t intermediate,
                                       int64_t in_row_stride, int64_t stream);

// ---- final logit softcap -------------------------------------------------------------------------------------------
// out[r, i] = tanh(x[r, i] / cap) * cap for i < n; x bf16 (x_bf16) or fp32 with row stride x_row_stride, out fp32 with
// row stride out_row_stride (may alias an fp32 x).
void r4dx_inkling2_logit_softcap_f32(int64_t x, int x_bf16, int64_t x_row_stride, int64_t out, int64_t out_row_stride,
                                     int rows, int n, float cap, int64_t stream);

// ---- small dense GEMM (no tiling: one wave per output element) ------------------------------------------------------
// out[m][n] = sum_k x[m][k] * w[n][k] (fp32, k in lane-strided order then a wave reduction), x bf16 [M, K] (row stride
// x_row_stride), w bf16 [N, K] contiguous, out fp32 (out_bf16 == 0) or bf16 (RNE) with row stride out_row_stride. Any
// N, K >= 1 (the shapes the tiled GEMMs do not take: the sink [heads, H], the dynamic-conv weight_proj [kv*4, H], the
// message-gate down [rank, 2H], and tiny-fixture linears with K % 64 != 0). M <= 65535.
void r4dx_inkling2_dense_bf16(int64_t x, int64_t x_row_stride, int64_t w, int64_t out, int out_bf16,
                              int64_t out_row_stride, int M, int N, int K, int64_t stream);

// ---- reference attention with relative bias and sinks (inkling2_ref::Attention) -----------------------------------
// The stage-0 oracle, derived from r4dx_gemma_attn_ref_bf16kv (attn_ref.h; that entry point is unchanged): one
// workgroup per (query row, q head), fp32 two-pass softmax, bf16 paged KV in attn_ref.h's layout (num_blocks,
// kv_heads, block_size, 2 * head_dim; K then V; key p in block block_table[p / block_size], slot p % block_size, so a
// ring is just its block table). Query row t is at qpos = ctx - q_len + t and sees keys klow..qpos,
// klow = window > 0 ? max(0, qpos - window + 1) : 0. Score s_j = (q . k_j) * scale + bias, bias =
// rel[t][h][qpos - j] when rel != 0 and qpos - j < rel_extent, else 0. With sink != 0, one more softmax column
// sink[t][h] whose value row is zero. out = softmax . V, one bf16 rounding.
struct R4dxInkling2AttnRefArgs {
  int64_t q;            // [q_len, q_heads, head_dim] bf16
  int64_t kv;           // paged bf16 cache
  int64_t block_table;  // int32
  int64_t out;          // [q_len, q_heads, head_dim] bf16
  int64_t rel;          // fp32 [q_len, q_heads, rel_extent] or 0
  int64_t sink;         // fp32 [q_len, q_heads] or 0
  int q_len, q_heads, kv_heads, head_dim, block_size;
  int ctx;
  int window;           // 0 = full causal
  int rel_extent;       // > 0 when rel != 0
  float scale;          // head_dim^-0.5
  int64_t kv_block_stride;
  int64_t kv_head_stride;
};
// 0 on success; negative for a shape it does not serve (head_dim not in [8, 512] or not a multiple of 8,
// q_heads % kv_heads, non-positive sizes, window < 0, ctx < q_len, rel without rel_extent > 0, null pointers).
int r4dx_inkling2_attn_ref_bf16kv(const R4dxInkling2AttnRefArgs* a, int64_t stream);

// ---- fused h128 attention with on-the-fly relative bias, sink, window, paged bf16 KV (S3) --------------------------
// Computes the same fp32 attention as r4dx_inkling2_attn_ref_bf16kv, without materializing rel[T, H, extent]:
// bias(t, h, dist) = sum_{j=0..d_rel-1} r[t, h, j] * rel_proj[j, dist] for 0 <= dist < rel_extent.
// Decode (q_len == 1) uses split-K over context (needs workspace of q_heads * max_splits * (head_dim + 2) floats
// when ctx > 128); prefill (q_len > 1) tiles K/V in LDS with online softmax across 4 query rows x GQA heads.
// Supports head_dim == 128, q_heads % kv_heads == 0, 1 <= q_heads/kv_heads <= 8, d_rel <= 32. Returns < 0 otherwise.
struct R4dxInkling2AttnFusedArgs {
  int64_t q;            // [q_len, q_heads, 128] bf16
  int64_t kv;           // paged bf16 cache
  int64_t block_table;  // int32
  int64_t out;          // [q_len, q_heads, 128] bf16
  int64_t r;            // fp32 [q_len, q_heads, d_rel] or 0
  int64_t rel_proj;     // fp32 [d_rel, rel_extent] or 0
  int64_t sink;         // fp32 [q_len, q_heads] or 0
  int64_t workspace;    // fp32 [q_heads * max_splits * (head_dim + 2)] (for q_len == 1 split-K)
  int q_len, q_heads, kv_heads, head_dim, block_size;
  int ctx;
  int window;           // 0 = full causal
  int d_rel;            // <= 32 (16 for Inkling2)
  int rel_extent;       // > 0 when r != 0
  int max_splits;       // >= 1 (e.g. 32)
  float scale;          // head_dim^-0.5
  int64_t kv_block_stride;
  int64_t kv_head_stride;
  int64_t pos_dev;      // optional device int32* holding qpos (q_len == 1 only; 0 = use host ctx - 1)
};
int r4dx_inkling2_attn_fused_bf16kv(const R4dxInkling2AttnFusedArgs* a, int64_t stream);

}  // extern "C"
