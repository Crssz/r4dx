// MXFP8 weight-only GEMM / GEMV (docs/inkling2-plan.md 4.1): OCP e4m3fn weights with one E8M0 scale per
// (row, 32-K group), bf16 activations in memory, f16 WMMA operands, f32 accumulation, bf16 output. A copy-adapt
// of r4dx_gemm_w8a16_nt_m64 (w8a16_gemm.h, docs/w8a16.md); same plain C ABI as the other r4dx_* kernels: device
// pointers and the stream as int64_t, shapes in ELEMENTS. Implemented in src/kernels/src/mxfp8_gemm.hip (its own
// hipcc object, r4dx_kernels.hip's flags, so -ffp-contract=off). The on-disk format and its CPU codec are fixed in
// src/convert/include/r4dx_convert/mxfp8_format.hpp; the kernel-exact CPU reference is tests/kernels/mxfp8_ref.hpp.
//
// Problem: C[M,N] bf16 = A[M,K] bf16 @ dequant(W)[N,K]^T, 1 <= M <= 256 (decode GEMV M <= 16 is the same kernel at
// one row tile; prefill launches up to 256 rows at once, m64-tiled: up to 4 row tiles per block).
//
// Byte order (mxfp8_format.hpp; w8a16's packed fragment order, docs/w8a16.md 3.2):
//   wq dword ((t*(K/32) + kb)*32 + l)*4 + j, t = row tile (N/16), kb = 32-K block = scale group, l = lane 0..31,
//      j = 0..3; byte i (0..3) holds the e4m3fn code of (row = 16t + (l&15), k = 32kb + 16(j>>1) + 8(j&1) + 4(l>>4) + i)
//   ws byte ((t*(K/32) + g)*16 + r) = the E8M0 exponent of row 16t + r, group g (value 2^(byte - 127))
// A and C are dense row-major (row stride K and N elements).
//
// Numerics (per output element, row m, column n):
//   a16[k]  = f16 of the bf16 A value (RTZ, as w8a16: exact for |a| in [2^-14, 65504], saturates above);
//   w16[k]  = the e4m3 value as an f16, EXACT (every e4m3 value, subnormals included, is a normal f16): the code's
//             low 7 bits shifted into an f16 (value * 2^-8, an f16 subnormal for an e4m3 subnormal), the sign
//             copied, then one exact f16 multiply by 2^8;
//   per group g (32 K = 2 WMMA k steps): gacc_g = sum over the group's k of a16[k]*w16[k] in f32 (the WMMA; every
//             product is exact, the sum order is the hardware's);
//   per split: acc = fmaf(gacc_g, 2^(ws_g - 127), acc) over the split's groups in order, from 0 (the scale is an
//             exact power of two, 2^-127 an f32 subnormal: f32 denormals must be on, HIP's default);
//   v = sum over the SK splits in split order (fixed, from 0.0f), then bf16 RNE. Scales are applied once per group
//   to the f32 partial sums because 2^e can leave the f16 range.
// Bit properties: a row's result depends on SK only (not on WV, MB, NPW, NT or the other rows of the launch), and
// runs are deterministic. No scratch, no tickets: two launches may overlap freely. Codes 0x7F / 0xFF (NaN) and scale
// byte 255 are never written by the converter and are not supported inputs.
#pragma once

#include <cstdint>

extern "C" {

// Launch geometry as r4dx_gemm_w8a16_nt_m64: a block holds WV column-tile groups x SK K-splits waves (wave32), each
// wave NPW column tiles of 16 and MB row tiles of 16; NT = non-temporal weight loads. Throws std::runtime_error on
// an illegal launch: M not in 1..256, N % 16, K % (SK * 32), WV < 1 or SK < 1, WV*SK*32 > 1024, MB not in 1..4,
// NPW not in {1, 4}, LDS WV*NPW*SK KiB > 64, a null pointer.
void r4dx_gemm_mxfp8_nt(int64_t a_bf16, int64_t wq, int64_t ws, int64_t c, int M, int K, int N, int WV, int SK,
                        int MB, int NPW, int NT, int64_t stream);

int r4dx_gemm_mxfp8_group(void);  // 32 (K per scale; the loader asserts the container's quant.mxfp8.group)
int r4dx_gemm_mxfp8_max_m(void);  // 256

// Device-indexed grouped MoE decode GEMM (docs/inkling2-speed-plan.md S1; copy-adapt of r4dx_gemm_w8a16_moe_decode):
// runs M = 1, MB = 1 across n_pairs = T * top_k routed (token, slot) pairs in one launch (grid.z = p in 0..n_pairs-1).
// For pair p:
//   expert e = ids[p] (int32 on device; a pair with e outside [0, n_experts) returns without writing C);
//   weights  = wq_ptrs[e], ws_ptrs[e] (uint64 device pointer tables of length n_experts);
//   input  A = a_bf16 + (p / a_row_group) * K (a_row_group = top_k for gate_up, 1 for down);
//   output C = c + p * N ([1, N] bf16).
// Bit-for-bit identical to calling r4dx_gemm_mxfp8_nt(A_p, wq_e, ws_e, C_p, 1, K, N, WV, SK, 1, NPW, NT, stream) per
// pair. Counts 1 launch in r4dx_kernel_launch_counter.
// Preconditions (throw): 1 <= n_pairs <= 16384, 1 <= a_row_group <= n_pairs, n_pairs % a_row_group == 0,
// 1 <= n_experts <= 256, and r4dx_gemm_mxfp8_nt's rules at M = 1, MB = 1.
void r4dx_gemm_mxfp8_moe_decode(int64_t a_bf16, int64_t wq_ptrs, int64_t ws_ptrs, int64_t ids, int64_t c,
                                int n_pairs, int a_row_group, int n_experts, int K, int N, int WV, int SK,
                                int NPW, int NT, int64_t stream);

// Device-indexed grouped MoE prefill GEMM (docs/inkling2-speed-plan.md S2; copy-adapt of r4dx_gemm_w8a16_moe_prefill):
// runs over the expert-sorted rows A[n_pairs, K] -> C[n_pairs, N], driven by device `expert_offsets[n_experts + 1]`
// with zero host readback. Each 16-row tile of each non-empty expert e runs the exact MT = 1 WMMA + LDS split-K
// reduction of r4dx_gemm_mxfp8_nt on wq_ptrs[e], ws_ptrs[e], producing output bytes identical to calling
// r4dx_gemm_mxfp8_nt per non-empty expert at the same SK. Counts 1 launch in r4dx_kernel_launch_counter.
void r4dx_gemm_mxfp8_moe_prefill(int64_t a_bf16, int64_t wq_ptrs, int64_t ws_ptrs, int64_t expert_offsets,
                                 int64_t c, int n_pairs, int n_experts, int K, int N, int WV, int SK,
                                 int NPW, int NT, int64_t stream);

// Native NVFP4 (FP4 E2M1 packed + F8_E4M3 per-16 scales + per-expert fp32 factor) grouped MoE decode/prefill GEMMs.
// Half the weight bandwidth of MXFP8; exact match to Inkling-2-Mini's native NVFP4 checkpoint experts.
void r4dx_gemm_nvfp4_moe_decode(int64_t a_bf16, int64_t wq_ptrs, int64_t ws_ptrs, int64_t factors_f32, int64_t ids,
                                int64_t c, int n_pairs, int a_row_group, int n_experts, int K, int N, int WV, int SK,
                                int NPW, int NT, int64_t stream);
void r4dx_gemm_nvfp4_moe_prefill(int64_t a_bf16, int64_t wq_ptrs, int64_t ws_ptrs, int64_t factors_f32,
                                 int64_t expert_offsets, int64_t c, int n_pairs, int n_experts, int K, int N, int WV,
                                 int SK, int NPW, int NT, int64_t stream);

}  // extern "C"
