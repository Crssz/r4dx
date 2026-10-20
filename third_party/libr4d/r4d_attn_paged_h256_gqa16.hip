// R4D attention dispatch: head_dim 256, 16 queries per KV head, paged block 16, fp8-e4m3 KV, bf16 query -- the full-attention
// layers of the Qwen3.5 MoE family (Ornith-1.5-122B: 32 q heads / 2 KV heads; docs/qwen35moe-plan.md, task M3).
//
// This is the gqa6 family's own kernel code (r4d_attn_prefill_h256_gqa6.hip and r4d_attn_decode_h256_gqa6.hip, included
// below) instantiated at another GQA: HEAD_DIM, GQA and the KV dtype are template arguments of those kernels, so nothing in
// them changes and the Qwen / Gemma / LFM2 instantiations compile to the same code (third_party/check_qwen_attn_isa.cmake).
// A row is (query position, head) = (row / GQA, row % GQA); at GQA 16 a 16-row MMA band is exactly one query position.
//
// Option sets: the gqa6 ones (r4d_attn_paged_h256_gqa6.hip) unchanged. Prefill keeps its 24 warps (the wide K fetch needs
// 24 * 32 threads to cover a 48-key tile in whole passes), i.e. 24 query positions per workgroup (24 warps x 16 rows / 16);
// decode serves q_len <= 4 (q_len * gqa <= 64). There is no exact-wide variant here (its q-block arithmetic assumes the
// 64-row block of gqa 6): the exact-wide request maps to the plain prefill launch, which is the same bits by definition.
//
// Entry points (r4d.h): r4d_attn_{prefill,prefill_splitkv,decode}_h256_gqa16_fp8kv and the two scratch_bytes queries.
#include "r4d.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

#define CHK() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

#define G_HEAD_DIM   256
#define G_GQA         16
#define G_BLOCK_SIZE  16
#define G_MAX_DECODE_ROWS 64        // q_len * gqa a single decode workgroup can hold -> q_len <= 4

static int g16_check(const R4DArgs* a) {
    if (a->head_dim != G_HEAD_DIM || a->block_size != G_BLOCK_SIZE) return -1;
    if (a->kv_heads <= 0 || a->q_heads % a->kv_heads != 0 || a->q_heads / a->kv_heads != G_GQA) return -2;
    return 0;
}

// ---- prefill ---------------------------------------------------------------------------------
// LDS at head_dim 256, fp8 KV, TILE 48: sK = 48*264*2 = 25344 B, sV = 256*56*2 = 28672 B (the gqa6 prefill's).
#define G_P_NWARPS 24
#define G_P_TILE   48
#define G_P_OPT    3955211

template <int SPLIT>
static void prefill_launch_g16(const R4DArgs* a, int splits, hipStream_t s) {
    constexpr int BQ = (G_P_NWARPS * 16) / G_GQA;
    dim3 grid(((a->q_len + BQ - 1) / BQ) * splits, a->kv_heads, a->num_seqs);
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<G_P_NWARPS, G_P_TILE, G_HEAD_DIM, G_GQA, G_BLOCK_SIZE, 0, G_P_OPT, SPLIT>),
                       grid, dim3(G_P_NWARPS * 32), 0, s, *a, splits);
}

extern "C" int r4d_attn_prefill_h256_gqa16_fp8kv(const R4DArgs* a, hipStream_t s) {
    if (const int rc = g16_check(a)) return rc;
    prefill_launch_g16<0>(a, 1, s);
    CHK();
    return 0;
}

// Split-KV prefill: the gqa6 contract (caller's segment count, clamped to 64; <= 1 IS the plain launch, same bits, no
// scratch; fp32 partials, fixed-order merge).
#define G_P_MAX_SPLITS 64
static int prefill_splits_g16(const R4DArgs* a) {
    return a->splits < 1 ? 1 : (a->splits > G_P_MAX_SPLITS ? G_P_MAX_SPLITS : a->splits);
}

extern "C" int64_t r4d_attn_prefill_splitkv_h256_gqa16_scratch_bytes(const R4DArgs* a) {
    const int splits = prefill_splits_g16(a);
    if (splits <= 1) return 0;
    const int64_t rows = (int64_t)a->num_seqs * a->q_len * a->q_heads;
    return rows * splits * (int64_t)(a->head_dim + 2) * 4;    // f32 O partials + m + l
}

extern "C" int r4d_attn_prefill_splitkv_h256_gqa16_fp8kv(const R4DArgs* a, hipStream_t s) {
    if (const int rc = g16_check(a)) return rc;
    const int splits = prefill_splits_g16(a);
    if (splits <= 1) return r4d_attn_prefill_h256_gqa16_fp8kv(a, s);
    if (a->scratch == nullptr) return -5;
    prefill_launch_g16<1>(a, splits, s);
    CHK();
    const int jobs = a->num_seqs * a->q_len * a->q_heads;
    hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<G_HEAD_DIM, 4, 0>), dim3(jobs), dim3(256), splits * sizeof(float), s,
                       *a, splits, G_P_TILE);
    CHK();
    return 0;
}

// ---- decode ----------------------------------------------------------------------------------
#define G_D_TILE 16
#define G_D_OPT  3430971

// The gqa6 split law (r4d_attn_paged_h256_gqa6.hip's decode_splits): a machine-fill target divided by the grid units that
// are not segments, snapped down to a power of two, then clamped so segments never get thinner than 2 tiles.
static int decode_splits_g16(const R4DArgs* a) {
    if (a->splits > 0) return a->splits;
    const int tiles = (a->max_ctx + G_D_TILE - 1) / G_D_TILE;
    const int units = max(1, a->kv_heads * a->num_seqs);
    const int raw   = 192 / units;
    int s = 128;
    while (s > 16 && s > raw) s >>= 1;
    const int cap = tiles / 2;
    while (s > 16 && s > cap) s >>= 1;
    if (s > tiles) s = tiles;
    if (s < 1) s = 1;
    return s;
}

extern "C" int64_t r4d_attn_decode_h256_gqa16_scratch_bytes(const R4DArgs* a) {
    const int splits = decode_splits_g16(a);
    const int64_t rows = (int64_t)a->num_seqs * a->q_len * a->q_heads;
    return (G_D_OPT & 32) ? rows * splits * (int64_t)(a->head_dim * 2 + 8)       // f16 O/l + m + l
                          : rows * splits * (int64_t)(a->head_dim + 2) * 4L;     // f32 O partials + m + l
}

extern "C" int r4d_attn_decode_h256_gqa16_fp8kv(const R4DArgs* a, hipStream_t s) {
    if (const int rc = g16_check(a)) return rc;
    const int rows = a->q_len * G_GQA;
    if (a->q_len < 1 || rows > G_MAX_DECODE_ROWS) return -4;    // verify width <= 4 at gqa 16
    if (a->scratch == nullptr) return -5;
    const int nw = (rows + 15) / 16;            // gqa-aligned height; no dead m-tiles issued
    const int splits = decode_splits_g16(a);

#define D_LAUNCH(NW)                                                                              \
    hipLaunchKernelGGL((r4d_attn_decode_kernel<NW, G_D_TILE, G_HEAD_DIM, G_GQA, G_BLOCK_SIZE,      \
                                               0, G_D_OPT>),                                      \
                       dim3(splits, a->kv_heads, a->num_seqs), dim3(NW * 32), 0, s, *a, splits)
    if      (nw == 1) D_LAUNCH(1);
    else if (nw == 2) D_LAUNCH(2);
    else if (nw == 3) D_LAUNCH(3);
    else              D_LAUNCH(4);
#undef D_LAUNCH
    CHK();

    const int jobs = a->num_seqs * a->q_len * a->q_heads;
    if (G_D_OPT & 32)
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<G_HEAD_DIM, 4, 1>), dim3(jobs), dim3(256),
                           splits * sizeof(float), s, *a, splits, G_D_TILE);
    else
        hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<G_HEAD_DIM, 4, 0>), dim3(jobs), dim3(256),
                           splits * sizeof(float), s, *a, splits, G_D_TILE);
    CHK();
    return 0;
}
