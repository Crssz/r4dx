// quant2 online rotations (docs/quant2.md section 3 "Q2a" and section 4 "Q2b"): the residual-stream
// rotation Q / Q^T at the 64-layer stack's entry and exit, and the in-place blockwise Hadamard on
// gdn.out_proj's input (Qwen), plus, for Gemma 4, the fused post-norm + rotate + residual add.
// Same plain C ABI as kernels.h (device pointers and stream as int64_t, shapes in ELEMENTS),
// implemented in src/kernels/src/rotate_residual.hip, linked into r4dx_kernels. Every launch counts
// toward r4dx_kernel_launch_counter_get() (kernels.h).
//
// The other two quant2 online Hadamards are fused into existing producers and declared next to
// them: r4dx_silu_mul_hadamard_bf16 (mlp.down's input, kernels.h) and
// r4dx_model_attn_gate_mul_hadamard_bf16 (attn.o's input, src/model/attention/.../attn_kernels.h).
//
// Every kernel here: bf16 in/out, fp32 math in LDS, one bf16 rounding per output element,
// deterministic and row-independent (a row's result does not depend on `rows` or on grid shape).
// The signs / mix5 arguments are the container's fp32 rotation tensors, uploaded as-is; nothing
// here generates or caches them.
#pragma once

#include <cstdint>

extern "C" {

// ---- residual rotation Q (quant2 contract, row-vector convention) ------------------------------
// In place over x: [rows, hidden] bf16, row-major, contiguous. The geometry follows from `hidden`:
// block = the largest power of two dividing hidden, capped at 1024 (the converter's
// ChooseRotationBlock; the loader refuses a container whose metadata says otherwise), nblk = hidden /
// block: 5120 = 5 x 1024 (Qwen), 3840 = 15 x 256 (Gemma 4), ... Throws for block < 2, nblk > 32 or a
// row that does not fit 48 KB of LDS (hidden + nblk^2 + 512 floats). nblk 5 and 15 are specialized
// (unrolled); every other nblk runs the same arithmetic through a runtime-nblk instantiation.
//   inverse == 0:  x <- x Q
//       y = x * d                                          (d = signs, fp32[hidden] of +-1)
//       y[b*block : (b+1)*block] = FWHT(y[block b]) / sqrt(block)   (natural/Sylvester order)
//       z[b*block + i] = sum_c y[c*block + i] * R[c][b]     (R = mix, fp32[nblk][nblk] row-major,
//                                                            mix[c*nblk + b] = R[c][b])
//   inverse != 0:  x <- x Q^T (the exact inverse)
//       y[c*block + i] = sum_b x[b*block + i] * R[c][b];  FWHT/sqrt(block) per block;  * d
// signs: fp32 [hidden] (rotation.signs). mix: fp32 [nblk * nblk] (rotation.mix5 for 5120,
// rotation.mix otherwise). One workgroup per row (about 4 * (hidden + nblk^2 + threads) bytes of LDS).
void r4dx_rotate_residual_bf16(int64_t x, int64_t rows, int64_t hidden, int64_t signs,
                                int64_t mix5, int inverse, int64_t stream);

// ---- fused post-norm + rotate + residual add (Gemma 4 option A, docs/gemma4-plan.md 4.4) --------
//   resid <- (resid + Q( rmsnorm_plain(y, weight) )) * layer_scale
// resid: [rows, hidden] bf16, the residual stream, ALREADY in the rotated basis (resid = r Q), updated
// in place. y: [rows, hidden] bf16, the o_proj / down_proj output in the ORIGINAL basis (those
// weights carry only W Hb, never Q^T). weight: [hidden] bf16, the post_attention / post_feedforward
// layernorm weight, PLAIN (x * rsqrt(mean(x^2) + eps) * w, no 1 + w). Q, signs, mix and the hidden
// geometry are exactly r4dx_rotate_residual_bf16's. layer_scale: Gemma's per-layer `layer_scalar`
// (1.0f for the attention sublayer, the layer's scalar for the MLP sublayer); it commutes with Q.
// All arithmetic is fp32 in LDS, the normed row is never written out, and the only bf16 rounding is
// the final store. The norm's sum of squares is a fixed-order reduction, so a row's result does not
// depend on `rows`. resid and y must not alias. Throws for the hidden limits above and null pointers.
// One workgroup per row. Counts one launch.
void r4dx_post_rmsnorm_rotate_add_bf16(int64_t resid, int64_t y, int64_t weight, int64_t signs,
                                        int64_t mix, int64_t rows, int64_t hidden, float eps,
                                        float layer_scale, int64_t stream);
// ---- blockwise Hadamard Hb, in place -----------------------------------------------------------
// x <- x Hb on each row: h Hb := (h * s) then FWHT / sqrt(block) on each contiguous block of
// `block` elements. x: [rows, K] bf16, row-major, contiguous. signs: fp32 [K] (+-1), indexed by the
// row-local column, so a TP rank passes its own K-slice of the sign vector together with its own K.
// Preconditions (throw): block a power of two in [2, 1024], K % block == 0.
// Used for gdn.out_proj's input (block 128 = one GDN value head, signs = rotation.had_gdn_out_signs)
// after the gated norm, on both the prefill and the decode/verify path. One workgroup per
// (row, block): grid = rows x (K / block).
void r4dx_hadamard_inplace_bf16(int64_t x, int64_t rows, int64_t K, int64_t signs, int block,
                                 int64_t stream);

}  // extern "C"
