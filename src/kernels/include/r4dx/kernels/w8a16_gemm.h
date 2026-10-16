// W8A16 weight-only GEMM (docs/w8a16.md sections 3-4): int8 weights with f16 group scales, bf16
// activations in memory, f16 WMMA operands, f32 accumulation, bf16 output. Same plain C ABI as the
// other r4dx_* kernels: device pointers and the stream as int64_t, shapes in ELEMENTS. Implemented in
// src/kernels/src/w8a16_gemm.hip (its own hipcc object, exactly r4dx_kernels.hip's flags, so
// -ffp-contract=off). CPU reference with the kernel's bit-exact semantics: tests/kernels/w8a16_ref.hpp.
//
// Problem: C[M,N] bf16 = A[M,K] bf16 @ dequant(W)[N,K]^T, 1 <= M <= 64 (a prefill caller chunks M into
// 64-row slices), group G in {64, 128} along K, symmetric int8 codes -127..127 stored offset-binary
// (byte = q + 128), one f16 scale per (row, group) stored as the f16 bits of s * 2^12
// (r4dx_gemm_w8a16_scale_log2_bias()).
//
// Byte order (docs/w8a16.md 3.2; the converter's PackW8Bytes / PackW8a16Scales):
//   wq dword ((t*(K/32) + kb)*32 + l)*4 + j, t = row tile (N/16), kb = 32-K block, l = lane 0..31,
//      j = 0..3; byte i (0..3) holds code(row = 16t + (l&15), k = 32kb + 16(j>>1) + 8(j&1) + 4(l>>4) + i)
//   ws u16 ((t*(K/G) + g)*16 + r) = f16 bits of s*2^12 for row 16t + r, group g
// A and C are dense row-major (row stride K and N elements).
//
// Numerics (per output element, row m, column n):
//   a16[k]  = f16 of the bf16 A value: exact for |a| in [2^-14, 65504]; RTZ, so |a| > 65504 saturates to
//             +-65504 (bf16 inf/nan are not supported inputs) and |a| < 2^-14 becomes an f16 subnormal
//             truncated toward zero;
//   q[k]    = byte - 128 (exact);
//   per group g: gacc_g = sum over the group's k of a16[k]*q[k] in f32 (the WMMA; every product is exact,
//             the sum order is the hardware's);
//   per split: acc = fmaf(gacc_g, float(s16_g), acc) over the split's groups in order, from 0;
//   v = sum over the SK splits in split order (fixed, from 0.0f), then v * 2^-12 (exact), then bf16 RNE.
// Bit properties: a row's result depends on SK only (not on WV, MB, NPW, NT or the other rows of the
// launch), and runs are deterministic. No scratch, no tickets: two launches may overlap freely.
#pragma once

#include <cstdint>

extern "C" {

// Launch geometry, as r4d_gemm_w4a16_nt_m64: a block holds WV column-tile groups x SK K-splits waves
// (wave32), each wave NPW column tiles of 16 and MB row tiles of 16; NT = non-temporal weight loads.
// Throws std::runtime_error on an illegal launch: M not in 1..64, N % 16, K % (SK * group),
// group not in {64, 128}, WV < 1 or SK < 1, WV*SK*32 > 1024, MB not in 1..4, NPW not in {1, 4},
// LDS WV*NPW*SK KiB > 64, a null pointer.
void r4dx_gemm_w8a16_nt_m64(int64_t a_bf16, int64_t wq, int64_t ws, int64_t c, int M, int K, int N,
                            int group, int WV, int SK, int MB, int NPW, int NT, int64_t stream);

int r4dx_gemm_w8a16_has_group(int group);   // 1 for 64 and 128, else 0
int r4dx_gemm_w8a16_scale_log2_bias(void);  // 12 (the converter and the loader assert it)
int r4dx_gemm_w8a16_max_m(void);            // 64

}  // extern "C"
