// r4dx-model-attention-owned HIP kernels: the two small elementwise ops the fused q_proj+gate
// layout and the attn_output_gate math need that neither r4d.h nor src/kernels provides. Same C
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

}  // extern "C"
