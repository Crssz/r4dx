// r4dx-model-attention-owned HIP kernels: the two small elementwise ops the fused q_proj+gate
// layout and the attn_output_gate math need that neither r4d.h nor src/kernels provides, plus the
// gate multiply's quant2 Hadamard variant. Same C
// ABI convention as third_party/libr4d/r4d.h and src/kernels/include/r4dx/kernels/kernels.h:
// device pointers and stream as int64_t, shapes/strides in ELEMENTS. Implemented in
// src/model/attention/src/attn_kernels.hip, compiled by hipcc the same way
// src/kernels/CMakeLists.txt builds r4dx_kernels (CMake's HIP language fights clang-cl on this
// toolchain; docs/build-windows.md), linked into the r4dx_model_attention static library.
#pragma once

#include <cstdint>

extern "C" {

// Splits the fused q_proj+output-gate GEMM output into separate contiguous q and gate buffers.
// docs/container-format.md "attn.qg": per-head-interleaved rows, NOT query-block-then-gate-block
// -- for head h, columns [h*2*head_dim, h*2*head_dim+head_dim) of the GEMM output are query,
// [+head_dim, +2*head_dim) are the gate.
// qg: [T, num_heads, 2*head_dim] bf16, row-major (the raw r4d_gemm_bf16_nt_m64 output).
// q, gate: [T, num_heads, head_dim] bf16, row-major, contiguous.
void r4dx_model_attn_split_qg_bf16(int64_t qg, int64_t q, int64_t gate, int T, int num_heads,
                                    int head_dim, int64_t stream);

// out[i] = bf16(fp32(attn_out[i]) * sigmoid(fp32(gate[i]))), elementwise over `n` elements -- the
// attn_output_gate epilogue (modeling_qwen3_5.py Qwen3_5Attention.forward: `attn_output *
// sigmoid(gate)`, docs/architecture.md "Attention layer"). Not r4dx_silu_mul_bf16: that kernel
// computes silu(gate)*up from the MLP's fused gate_up layout, a different activation (silu vs
// sigmoid) over a different fusion (gate_up vs qg) -- reusing it here would silently apply the
// wrong nonlinearity.
void r4dx_model_attn_gate_mul_bf16(int64_t attn_out, int64_t gate, int64_t out, int64_t n,
                                    int64_t stream);

// quant2 Q2b (docs/quant2.md section 4): the same gate multiply, then attn.o's online blockwise
// Hadamard on each row of `row_len` elements: out[r, :] = (attn_out[r, :] * sigmoid(gate[r, :])) Hb,
// h Hb := (h * s) then FWHT / sqrt(block) on each contiguous block of `block` (natural/Sylvester
// order; see src/kernels/include/r4dx/kernels/rotate_residual.h). The product stays fp32 into the
// transform, so each output is rounded to bf16 once.
// attn_out, gate, out: [n / row_len, row_len] bf16 contiguous -- the [T, H, D] buffers above, with
// row_len = H * D (the o-proj K, whose column order is head * D + d). With block = D = 256 each Hb
// block is exactly one head. signs: fp32 [row_len] (+-1, rotation.had_o_signs), indexed by the
// row-local column, so a TP rank passes its own K-slice with its own H * D (12 x 256 at TP=2).
// Preconditions (throw): block a power of two in [2, 1024], row_len % block == 0,
// n % row_len == 0. `row_len` is the one argument r4dx_model_attn_gate_mul_bf16 does not have: the
// flat elementwise op never needed to know where a row ends, this one indexes `signs` by it.
// One workgroup per (row, block). Like the plain variant, not counted by
// r4dx_kernel_launch_counter (r4dx_model_attention's launches never were), so swapping one for the
// other leaves the profiled launch count unchanged.
void r4dx_model_attn_gate_mul_hadamard_bf16(int64_t attn_out, int64_t gate, int64_t out, int64_t n,
                                             int64_t stream, int64_t signs, int block,
                                             int64_t row_len);

}  // extern "C"
