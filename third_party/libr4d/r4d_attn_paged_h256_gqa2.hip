// R4D sliding-window attention dispatch: head_dim 256, 2 queries per KV head, paged block 16, fp8-e4m3
// KV, bf16 query -- the sliding layers of Gemma 4 (16 q heads / 8 KV heads), windowed to W keys.
//
// This is the gqa6 family's own kernel code (r4d_attn_prefill_h256_gqa6.hip and
// r4d_attn_decode_h256_gqa6.hip, included below) instantiated with WIN = 1: the geometry (GQA, NWARPS)
// is a template argument, and the window and the klimit_ext bidirectional extension are the WIN / EXT
// arguments those files document. The Qwen instantiations in r4d_attn_paged_h256_gqa6.hip are the
// WIN = EXT = 0 ones and compile to the same code as before this file existed (the build checks:
// third_party/check_qwen_attn_isa.cmake).
//
// The KV ring is nothing here: the kernels address keys as block_table[kpos / 16], and a sliding
// layer's block table is r4d_attn_window.h's T[i] = i % RB. The cache MUST be zero-initialised -- a
// tile can start up to TILE-1 keys below the window and stage slots that alias newer keys.
//
// Entry points: r4d_attn_{prefill,decode}_h256_gqa2_fp8kv (R4DArgsW, r4d.h) and
// r4d_attn_decode_h256_gqa2_scratch_bytes.
#include "r4d.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

#define CHK() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

#define W_HEAD_DIM   256
#define W_GQA          2
#define W_BLOCK_SIZE  16
#define W_MAX_DECODE_ROWS 64        // q_len * gqa a single decode workgroup can hold -> q_len <= 32

// ---- prefill ---------------------------------------------------------------------------------
// 8 warps x 16 rows / gqa 2 = 64 query rows per workgroup; LDS at head_dim 256, fp8 KV, TILE 48:
// sK = 48*264*2 = 25344 B, sV = 256*56*2 = 28672 B. The same measured option set as the gqa6 prefill
// (P_OPT, from the included file's family).
// (Overridable on the command line, -DW_P_OPT=..., for ISA experiments.)
#ifndef W_P_NWARPS
#define W_P_NWARPS 8
#endif
#ifndef W_P_TILE
#define W_P_TILE   48
#endif
#ifndef W_P_OPT
#define W_P_OPT    3955211
#endif

extern "C" int r4d_attn_prefill_h256_gqa2_fp8kv(const R4DArgsW* a, hipStream_t s) {
    if (a->head_dim != W_HEAD_DIM || a->block_size != W_BLOCK_SIZE) return -1;
    if (a->kv_heads <= 0 || a->q_heads / a->kv_heads != W_GQA || a->q_heads % a->kv_heads != 0) return -2;
    if (a->window < 1) return -6;
    constexpr int BQ = (W_P_NWARPS * 16) / W_GQA;
    dim3 grid((a->q_len + BQ - 1) / BQ, a->kv_heads, a->num_seqs);
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<W_P_NWARPS, W_P_TILE, W_HEAD_DIM, W_GQA, W_BLOCK_SIZE,
                                                0, W_P_OPT, 0, 1, 1, 1>),
                       grid, dim3(W_P_NWARPS * 32), 0, s, *a, 1);
    CHK();
    return 0;
}

// ---- decode ----------------------------------------------------------------------------------
#define W_D_TILE 16
#ifndef W_D_OPT
#define W_D_OPT  3430971
#endif

// The gqa6 split law (r4d_attn_paged_h256_gqa6.hip's decode_splits) fed the WINDOW's context bound:
// a sliding layer's work is the window, however long the sequence, so the segment count is a function
// of min(max_ctx, W + q_len + 16) (r4d_win_decode_max_ctx) and q_len, never of the full context.
static int decode_splits_win(const R4DArgsW* a) {
    if (a->splits > 0) return a->splits;
    const int ctx_eff = r4d_win_decode_max_ctx(a->max_ctx, a->window, a->q_len);
    const int tiles = (ctx_eff + W_D_TILE - 1) / W_D_TILE;
    const int units = max(1, a->kv_heads * a->num_seqs);
    const int raw   = 192 / units;              // CU-fill target over the non-segment grid units
    int s = 128;
    while (s > 16 && s > raw) s >>= 1;
    const int cap = tiles / 2;                  // >= 2 tiles per segment
    while (s > 16 && s > cap) s >>= 1;
    if (s > tiles) s = tiles;
    if (s < 1) s = 1;
    return s;
}

extern "C" int64_t r4d_attn_decode_h256_gqa2_scratch_bytes(const R4DArgsW* a) {
    const int splits = decode_splits_win(a);
    const int64_t rows = (int64_t)a->num_seqs * a->q_len * a->q_heads;
    return (W_D_OPT & 32) ? rows * splits * (int64_t)(a->head_dim * 2 + 8)       // f16 O/l + m + l
                          : rows * splits * (int64_t)(a->head_dim + 2) * 4L;     // f32 O partials + m + l
}

extern "C" int r4d_attn_decode_h256_gqa2_fp8kv(const R4DArgsW* a, hipStream_t s) {
    if (a->head_dim != W_HEAD_DIM || a->block_size != W_BLOCK_SIZE) return -1;
    if (a->kv_heads <= 0 || a->q_heads / a->kv_heads != W_GQA || a->q_heads % a->kv_heads != 0) return -2;
    if (a->window < 1) return -6;
    const int rows = a->q_len * W_GQA;
    if (rows > W_MAX_DECODE_ROWS) return -4;    // verify width <= 32 at gqa 2
    if (a->scratch == nullptr) return -5;
    const int nw = (rows + 15) / 16;            // gqa-aligned height; no dead m-tiles issued
    const int splits = decode_splits_win(a);

#define D_LAUNCH(NW)                                                                          \
    hipLaunchKernelGGL((r4d_attn_decode_kernel<NW, W_D_TILE, W_HEAD_DIM, W_GQA, W_BLOCK_SIZE,  \
                                               0, W_D_OPT, 1>),                               \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, s, *a, splits)
    if      (nw == 1) D_LAUNCH(1);
    else if (nw == 2) D_LAUNCH(2);
    else if (nw == 3) D_LAUNCH(3);
    else              D_LAUNCH(4);
#undef D_LAUNCH
    CHK();

    const int jobs = a->num_seqs * a->q_len * a->q_heads;
    if (W_D_OPT & 32)
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<W_HEAD_DIM, 4, 1, 1>), dim3(jobs),
                           dim3(256), splits * sizeof(float), s, *a, splits, W_D_TILE);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<W_HEAD_DIM, 4, 0, 1>), dim3(jobs),
                           dim3(256), splits * sizeof(float), s, *a, splits, W_D_TILE);
    CHK();
    return 0;
}
