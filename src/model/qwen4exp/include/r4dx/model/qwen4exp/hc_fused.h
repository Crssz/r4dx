// Fused decode (M = 1) kernels of the gated residual's Prep (hc_conn.h): they replace
//     down GEMM -> silu_scaled          and      inject GEMM -> inject gate         by q4x_hc_prep_gemm1_bf16
//     up GEMM -> hc mix                                                              by q4x_hc_prep_up_mix_bf16
// Bit-identical to the chain they replace: each output tile is the same WMMA accumulation over the same K slice as
// r4d_gemm_bf16_nt_m64 (down: SK 4 over 2560, inject: SK 16 over 640, up: SK 4 over 80), the splits are summed in the same
// ascending order from 0.f, and every bf16 rounding point of the unfused kernels (GEMM output, silu input, sigmoid, product,
// mean) is kept. Only the launch structure differs: more blocks for the GEMMs (the down GEMM used 5, the inject GEMM 1) and no
// intermediate tensor round trips.
//
// Shapes: hc == 4, hidden == 2560, lowrank == 320 (the qwen4_exp config); anything else returns hipErrorInvalidValue and the
// caller keeps the unfused chain (q4x_hc_fused_supported).
#pragma once

#include <cstdint>

extern "C" {

// 1 when q4x_hc_prep_gemm1_bf16 / q4x_hc_prep_up_mix_bf16 accept these dimensions.
int q4x_hc_fused_supported(int hc, int hidden, int lowrank);

// n: bf16 [hc * hidden] (the normed residual row). w_down: bf16 [lowrank, hc * hidden]; mid: bf16 [lowrank] =
// silu(Rb(down(n) * silu_scale)). w_inject (may be null, then inj is not written): bf16 [hc, hc * hidden]; inj: bf16 [hc] =
// 2 * sigmoid(inject(n) / hc).
int q4x_hc_prep_gemm1_bf16(const void* n, const void* w_down, const void* w_inject, void* mid, void* inj, int hc, int hidden,
                           int lowrank, float silu_scale, void* stream);

// x: bf16 [hidden] = (1/hc) sum_g Rb(Rb(sigmoid(up(mid)[g, d])) * n[g, d]); w_up: bf16 [hc * hidden, lowrank].
int q4x_hc_prep_up_mix_bf16(const void* mid, const void* w_up, const void* n, void* x, int hc, int hidden, int lowrank,
                            void* stream);

}  // extern "C"
