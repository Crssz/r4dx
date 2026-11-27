// R4D attention dispatch: head_dim 128, 8 queries per KV head, paged block 16, bf16 or fp8-e4m3 KV, bf16 query --
// the attention layers of the hy_v3 family (Hy-MT3-30B-A3B: 32 q heads / 4 KV heads).
//
// Instantiates r4d_attn_prefill_h256_gqa6.hip and r4d_attn_decode_h256_gqa6.hip at HEAD_DIM=128, GQA=8.
// Prefill: 16 warps x 16 rows / gqa 8 = 32 query rows per workgroup, 64-key tiles.
// Decode: at most 64 query rows (q_len <= 8).
#include "r4d.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

#define CHK() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

#define H_HEAD_DIM  128
#define H_GQA         8
#define H_BLOCK_SIZE 16
#define H_MAX_DECODE_ROWS 64        // q_len * gqa <= 64 -> q_len <= 8

static int h128g8_check(const R4DArgs* a) {
    if (a->head_dim != H_HEAD_DIM || a->block_size != H_BLOCK_SIZE) return -1;
    if (a->kv_heads <= 0 || a->q_heads % a->kv_heads != 0 || a->q_heads / a->kv_heads != H_GQA) return -2;
    return 0;
}

#ifndef H_P_NWARPS
#define H_P_NWARPS 16
#endif
#ifndef H_P_TILE
#define H_P_TILE   64
#endif
#ifndef H_P_OPT
#define H_P_OPT    3430923
#endif
static_assert((H_P_NWARPS * 16) % H_GQA == 0, "a prefill workgroup must hold whole queries (rows = NWARPS*16)");

template <int KVP, int SPLIT>
static void prefill_launch_h128g8(const R4DArgs* a, int splits, hipStream_t s) {
    constexpr int BQ = (H_P_NWARPS * 16) / H_GQA;
    dim3 grid(((a->q_len + BQ - 1) / BQ) * splits, a->kv_heads, a->num_seqs);
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<H_P_NWARPS, H_P_TILE, H_HEAD_DIM, H_GQA, H_BLOCK_SIZE,
                                                KVP, H_P_OPT, SPLIT>),
                       grid, dim3(H_P_NWARPS * 32), 0, s, *a, splits);
}

template <int KVP>
static int prefill_h128g8(const R4DArgs* a, hipStream_t s) {
    if (const int rc = h128g8_check(a)) return rc;
    prefill_launch_h128g8<KVP, 0>(a, 1, s);
    CHK();
    return 0;
}

extern "C" int r4d_attn_prefill_h128_gqa8_bf16kv(const R4DArgs* a, hipStream_t s) { return prefill_h128g8<1>(a, s); }
extern "C" int r4d_attn_prefill_h128_gqa8_fp8kv(const R4DArgs* a, hipStream_t s) { return prefill_h128g8<0>(a, s); }

#define H_P_MAX_SPLITS 64
static int prefill_splits_h128g8(const R4DArgs* a) {
    return a->splits < 1 ? 1 : (a->splits > H_P_MAX_SPLITS ? H_P_MAX_SPLITS : a->splits);
}

extern "C" int64_t r4d_attn_prefill_splitkv_h128_gqa8_scratch_bytes(const R4DArgs* a) {
    const int splits = prefill_splits_h128g8(a);
    if (splits <= 1) return 0;
    const int64_t rows = (int64_t)a->num_seqs * a->q_len * a->q_heads;
    return rows * splits * (int64_t)(a->head_dim + 2) * 4;
}

template <int KVP>
static int prefill_splitkv_h128g8(const R4DArgs* a, hipStream_t s) {
    if (const int rc = h128g8_check(a)) return rc;
    const int splits = prefill_splits_h128g8(a);
    if (splits <= 1) return prefill_h128g8<KVP>(a, s);
    if (a->scratch == nullptr) return -5;
    prefill_launch_h128g8<KVP, 1>(a, splits, s);
    CHK();
    const int jobs = a->num_seqs * a->q_len * a->q_heads;
    hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<H_HEAD_DIM, 4, 0>), dim3(jobs), dim3(H_HEAD_DIM),
                       splits * sizeof(float), s, *a, splits, H_P_TILE);
    CHK();
    return 0;
}

extern "C" int r4d_attn_prefill_splitkv_h128_gqa8_bf16kv(const R4DArgs* a, hipStream_t s) {
    return prefill_splitkv_h128g8<1>(a, s);
}
extern "C" int r4d_attn_prefill_splitkv_h128_gqa8_fp8kv(const R4DArgs* a, hipStream_t s) {
    return prefill_splitkv_h128g8<0>(a, s);
}

#ifndef H_D_TILE
#define H_D_TILE 32
#endif
#ifndef H_D_OPT
#define H_D_OPT  3430971
#endif
#ifndef H_D_TARGET_WGS
#define H_D_TARGET_WGS 1024
#endif
#define H_D_MAX_SPLITS 256

static int decode_splits_h128g8(const R4DArgs* a) {
    if (a->splits > 0) return a->splits > H_D_MAX_SPLITS ? H_D_MAX_SPLITS : a->splits;
    const int tiles = (a->max_ctx + H_D_TILE - 1) / H_D_TILE;
    const int units = max(1, a->kv_heads * a->num_seqs);
    const int raw = H_D_TARGET_WGS / units;
    int s = H_D_MAX_SPLITS;
    while (s > 1 && s > raw) s >>= 1;
    const int cap = tiles / 2;
    while (s > 1 && s > cap) s >>= 1;
    if (s < 1) s = 1;
    return s;
}

extern "C" int64_t r4d_attn_decode_h128_gqa8_scratch_bytes(const R4DArgs* a) {
    const int splits = decode_splits_h128g8(a);
    const int64_t rows = (int64_t)a->num_seqs * a->q_len * a->q_heads;
    return (H_D_OPT & 32) ? rows * splits * (int64_t)(a->head_dim * 2 + 8)
                          : rows * splits * (int64_t)(a->head_dim + 2) * 4L;
}

template <int KVP>
static int decode_h128g8(const R4DArgs* a, hipStream_t s) {
    if (const int rc = h128g8_check(a)) return rc;
    const int rows = a->q_len * H_GQA;
    if (a->q_len < 1 || rows > H_MAX_DECODE_ROWS) return -4;
    if (a->scratch == nullptr) return -5;
    const int nw = (rows + 15) / 16;
    const int splits = decode_splits_h128g8(a);

#define D_LAUNCH(NW)                                                                          \
    hipLaunchKernelGGL((r4d_attn_decode_kernel<NW, H_D_TILE, H_HEAD_DIM, H_GQA, H_BLOCK_SIZE,  \
                                               KVP, H_D_OPT>),                                \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, s, *a, splits)
    if      (nw == 1) D_LAUNCH(1);
    else if (nw == 2) D_LAUNCH(2);
    else if (nw == 3) D_LAUNCH(3);
    else              D_LAUNCH(4);
#undef D_LAUNCH
    CHK();

    const int jobs = a->num_seqs * a->q_len * a->q_heads;
    if (H_D_OPT & 32)
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<H_HEAD_DIM, 4, 1>), dim3(jobs),
                           dim3(H_HEAD_DIM), splits * sizeof(float), s, *a, splits, H_D_TILE);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<H_HEAD_DIM, 4, 0>), dim3(jobs),
                           dim3(H_HEAD_DIM), splits * sizeof(float), s, *a, splits, H_D_TILE);
    CHK();
    return 0;
}

extern "C" int r4d_attn_decode_h128_gqa8_bf16kv(const R4DArgs* a, hipStream_t s) { return decode_h128g8<1>(a, s); }
extern "C" int r4d_attn_decode_h128_gqa8_fp8kv(const R4DArgs* a, hipStream_t s) { return decode_h128g8<0>(a, s); }
