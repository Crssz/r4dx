// r4d_gdn_chunk_scan_stable_k128_v128_c64_bf16.hip -- the span-safe gated-delta-net chunk scan: the shipped kernel's source
// (r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip) compiled with R4D_GDN_SCAN_STABLE = 1, under its own kernel and entry names.
//
// A unit of its own, as the `_wq` units of the trellis GEMMs are, so the shipped scan's code cannot change: the shipped unit is
// compiled with R4D_GDN_SCAN_STABLE undefined (0), where every changed line keeps its original text, and this unit carries the
// variant. The Qwen3.5 / Gemma models call the shipped entry point; qwen4_exp's GDN layers call this one.
//
// Why: the shipped scan splits the pair decay e^{G_i - G_j} (G = the chunk's gate cumsum) as e^{G_i - c} e^{c - G_j} around the chunk
// midpoint c and clamps each half at e^80, which is exact only while the chunk's gate span stays under ~160 and attenuates the
// near-diagonal weights beyond. A model whose gate A = e^A_log reaches 158 (Qwen3.8-Flash-Next layer 0: per-chunk spans up to 6505,
// 14 of 48 heads of the first chunk over 160) loses them: layer 0's output was wrong by 0.34 of its rms on the real T70 prompt. This
// variant applies e^{G_i - G_j} per element in the score step and weights the K rows of the state update by e^{G_last - G_t}; every
// exponent is <= 0, so nothing can overflow at any span. It costs 16 __expf per lane per chunk in the score step plus one
// LDS pass over the K tile; the shipped kernel's own header says why it did not take that road (speed, and spans "up to 64 of the 88
// fp32 allows" on the models it was written for).
//
// Geometry, layouts, arguments and the 64-row chunk contract are the shipped kernel's (r4d.h documents the entry point). The kkt_solve
// kernel that forms A is already span-safe (per-element e^{G_i - G_j}); conv_prep's cumsum is fp32.
#define R4D_GDN_SCAN_STABLE 1
#define r4d_gdn_chunk_scan_kernel r4d_gdn_chunk_scan_stable_kernel
#define r4d_gdn_chunk_scan_k128_v128_c64_bf16 r4d_gdn_chunk_scan_stable_k128_v128_c64_bf16
#define r4d_gdn_dims r4d_gdn_stable_dims
#include "r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip"
