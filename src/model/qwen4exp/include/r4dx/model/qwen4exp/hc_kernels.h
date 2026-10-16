// qwen4_exp gated-residual ("hyper-connection", HF Qwen4ExpTextGatedResidual, modular_qwen4_exp.py:531-559) elementwise
// kernels, plus the grouped (1 + w) RMSNorm they share with the PLE layer. docs/qwen4exp-plan.md 2.2, 4.5.
//
// The residual h is [rows, hc * hidden] bf16 (hc = 4 streams of hidden = 2560). The two GEMMs and the inject
// projection between these kernels are issued by HyperConn (hc_conn.h). Plain C ABI like third_party/libr4d/r4d.h:
// device pointers are `const void*` / `void*`, `stream` a hipStream_t, the return value a hipError_t (0 on
// success); the C++ wrappers in hc_conn.h / hc_bf16_gemm.h throw on a non-zero one.
//
// Rounding: every kernel rounds to bf16 exactly where the HF bf16 graph does (the Rb points are listed per kernel)
// and computes the rest in fp32.
#pragma once

#include <cstdint>

extern "C" {

// Grouped RMSNorm with (1 + w) centring, Qwen4ExpTextRMSNorm(group_size) (modular_qwen4_exp.py:299-310, forward in
// qwen3_5 Qwen3_5RMSNorm:849-854): for each of `groups` consecutive groups of `group_size` channels of a row,
//     out = bf16( x * rsqrt(mean(x^2 over the group) + eps) * (1 + w) )          fp32 math, one rounding.
// x, out: bf16 [rows, groups * group_size] (out may alias x); w: bf16 [groups * group_size], the RAW checkpoint
// weight (the 1 + is applied here). group_size <= 4096. Used by the gated residual (hc_norm), and by the PLE layer
// (norm_key, norm_query, norm_conv).
int q4x_grouped_rmsnorm_bf16(const void* x, const void* w, void* out, int64_t rows, int groups, int group_size,
                              float eps, void* stream);

// out[i] = bf16( silu( Rb(in[i] * scale) ) ), i in [0, n)  -- HF `F.silu(down(n) / hc_count)` (line 550) with
// scale = 1 / hc_count (a power of two, so the division is exact in bf16 as well). in / out: bf16 (may alias).
int q4x_hc_silu_scaled_bf16(const void* in, void* out, int64_t n, float scale, void* stream);

// x[r, d] = bf16( (1/hc) * sum_{g<hc} Rb( Rb(sigmoid(up[r, g*hidden + d])) * n[r, g*hidden + d] ) )  -- HF lines
// 551-555: sigmoid(up(.)) is a bf16 tensor, the product with the normed input is a bf16 tensor, and the mean over the
// stream axis accumulates in fp32 (g ascending) and rounds once. n, up: bf16 [rows, hc*hidden]; x: bf16 [rows, hidden].
int q4x_hc_mix_bf16(const void* n, const void* up, void* x, int64_t rows, int hc, int hidden, void* stream);

// inj[i] = bf16( 2 * Rb(sigmoid(Rb(v[i] / hc))) ) -- HF line 558 (`2 * sigmoid(block_inject(n) / hc_count)`). The
// factor 2 is exact. v (the block_inject GEMM output) and inj: bf16 [n] (rows * hc), may alias.
int q4x_hc_inject_gate_bf16(const void* v, void* inj, int64_t n, int hc, void* stream);

// h_out[r, g*hidden + d] = bf16( h[r, g*hidden + d] + Rb(block_out[r, d] * inj[r, g]) ) -- the injection of the
// decoder layer (modular_qwen4_exp.py:830-831, 836-837). h, h_out: bf16 [rows, hc*hidden] (h_out may alias h);
// block_out: bf16 [rows, hidden]; inj: bf16 [rows, hc].
int q4x_hc_inject_bf16(const void* h, const void* block_out, const void* inj, void* h_out, int64_t rows, int hc,
                        int hidden, void* stream);

// out[r, g*hidden + d] = table[ids[r], d] for g < repeat: the embedding gather with the `repeat(1, 1, hc_count)` of
// the text model (modular_qwen4_exp.py:1014) fused. table: bf16 [vocab, hidden]; ids: int32 [rows] (device); out: bf16
// [rows, repeat * hidden]. An id outside [0, vocab) fills that row with quiet NaNs (a device kernel cannot throw; NaN
// is loud downstream, never a silent wrong embedding).
int q4x_embed_gather_repeat_bf16(const void* table, const int32_t* ids, void* out, int64_t rows, int64_t vocab,
                                  int hidden, int repeat, void* stream);

}  // extern "C"
