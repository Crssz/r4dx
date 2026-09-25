// quant2 online rotations (docs/quant2.md section 3 "Q2a" and section 4 "Q2b"): the residual-stream
// rotation Q / Q^T at the 64-layer stack's entry and exit, and the in-place blockwise Hadamard on
// gdn.out_proj's input. Same plain C ABI as kernels.h (device pointers and stream as int64_t,
// shapes in ELEMENTS), implemented in src/kernels/src/rotate_residual.hip, linked into
// r4dx_kernels. Both launches count toward r4dx_kernel_launch_counter_get() (kernels.h).
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
// In place over x: [rows, hidden] bf16, row-major, contiguous. hidden MUST be 5120 (= 5 x 1024;
// throws otherwise -- the kernel is specialized for it).
//   inverse == 0:  x <- x Q
//       y = x * d                                          (d = signs, fp32[5120] of +-1)
//       y[b*1024 : (b+1)*1024] = FWHT(y[block b]) / 32     (b = 0..4, natural/Sylvester order)
//       z[b*1024 + i] = sum_c y[c*1024 + i] * R[c][b]       (R = mix5, fp32[5][5] row-major,
//                                                            mix5[c*5 + b] = R[c][b])
//   inverse != 0:  x <- x Q^T (the exact inverse)
//       y[c*1024 + i] = sum_b x[b*1024 + i] * R[c][b];  FWHT/32 per block;  * d
// signs: fp32 [5120] (rotation.signs). mix5: fp32 [25] (rotation.mix5).
// One workgroup per row (20 KB of LDS).
void r4dx_rotate_residual_bf16(int64_t x, int64_t rows, int64_t hidden, int64_t signs,
                                int64_t mix5, int inverse, int64_t stream);

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
