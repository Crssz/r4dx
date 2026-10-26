// R4D attention dispatch: head_dim 128, 7 queries per KV head, paged block 16, bf16 or fp8-e4m3 KV, bf16 query --
// the attention layers of the dense Llama family (Nanbeige4.3-33B: 56 q heads / 8 KV heads, 28 / 4 per
// rank at --tp 2; docs/nanbeige-plan.md 3.3, task N5-1).
//
// This is the gqa6 family's own kernel code (r4d_attn_prefill_h256_gqa6.hip and
// r4d_attn_decode_h256_gqa6.hip, included below) instantiated at another geometry: HEAD_DIM, GQA and
// the KV dtype (KVP = 1 bf16, KVP = 0 fp8 with per-(seq, head) descales, task N5-2) are template arguments of those kernels, so nothing in them changes and
// the Qwen / Gemma / LFM2 instantiations compile to the same code (third_party/check_qwen_attn_isa.cmake).
//
// GQA 7 and the prefill row map: a prefill workgroup holds NWARPS x 16 query-head rows, row r serving
// query r / GQA, head r % GQA, and the kernel requires the rows to be whole queries (its static_assert:
// a partial query's rows would be stored by two workgroups, one of them short of its last key). 14 warps
// = 224 rows = 32 queries of 7 heads. 224 threads do not cover the wide K fetch's 512 passes of a
// 64-key tile in whole rounds, so the prefill drops O_KWIDE (as the h64 unit does). The decode kernel maps
// q_len x GQA rows onto 16-row bands and leaves the rows past q_len x GQA dead, so it needs nothing.
//
// Option sets: the gqa6 ones (r4d_attn_paged_h256_gqa6.hip's header lists every bit) minus O_KWIDE for
// the prefill, as h64_gqa4. Entry points (r4d.h): r4d_attn_{prefill,prefill_splitkv,decode}_h128_gqa7_{bf16kv,fp8kv}
// and the two scratch_bytes queries (one geometry, so shared). bf16: k_descale / v_descale may be null (1.0); fp8: they
// are the cache's per-(seq, kv head) dequant scales.
#include "r4d.h"
#include "r4d_common.h"
#include <cmath>
#include <algorithm>
using std::max;
#include "r4d_attn_prefill_h256_gqa6.hip"
#include "r4d_attn_decode_h256_gqa6.hip"

#define CHK() do { hipError_t e_ = hipGetLastError(); if (e_ != hipSuccess) return -100 - (int)e_; } while (0)

#define H_HEAD_DIM  128
#define H_GQA         7
#define H_BLOCK_SIZE 16
#define H_MAX_DECODE_ROWS 64        // q_len * gqa a single decode workgroup can hold -> q_len <= 9

static int h128_check(const R4DArgs* a) {
    if (a->head_dim != H_HEAD_DIM || a->block_size != H_BLOCK_SIZE) return -1;
    if (a->kv_heads <= 0 || a->q_heads % a->kv_heads != 0 || a->q_heads / a->kv_heads != H_GQA) return -2;
    return 0;
}

// ---- prefill ---------------------------------------------------------------------------------
// 14 warps x 16 rows / gqa 7 = 32 query rows per workgroup. LDS at head_dim 128, bf16 KV, TILE 64:
// sK = 64*136*2 = 17408 B, sV = 128*72*2 = 18432 B. (Overridable on the command line for ISA experiments.)
#ifndef H_P_NWARPS
#define H_P_NWARPS 14
#endif
#ifndef H_P_TILE
#define H_P_TILE   64
#endif
#ifndef H_P_OPT
#define H_P_OPT    3430923          // the gqa6 prefill set (3955211) minus O_KWIDE (524288)
#endif
static_assert((H_P_NWARPS * 16) % H_GQA == 0, "a prefill workgroup must hold whole queries (rows = NWARPS*16)");

template <int KVP, int SPLIT>
static void prefill_launch_h128(const R4DArgs* a, int splits, hipStream_t s) {
    constexpr int BQ = (H_P_NWARPS * 16) / H_GQA;
    dim3 grid(((a->q_len + BQ - 1) / BQ) * splits, a->kv_heads, a->num_seqs);
    hipLaunchKernelGGL((r4d_attn_prefill_kernel<H_P_NWARPS, H_P_TILE, H_HEAD_DIM, H_GQA, H_BLOCK_SIZE,
                                                KVP, H_P_OPT, SPLIT>),
                       grid, dim3(H_P_NWARPS * 32), 0, s, *a, splits);
}


template <int KVP>
static int prefill_h128(const R4DArgs* a, hipStream_t s) {
    if (const int rc = h128_check(a)) return rc;
    prefill_launch_h128<KVP, 0>(a, 1, s);
    CHK();
    return 0;
}

extern "C" int r4d_attn_prefill_h128_gqa7_bf16kv(const R4DArgs* a, hipStream_t s) { return prefill_h128<1>(a, s); }
extern "C" int r4d_attn_prefill_h128_gqa7_fp8kv(const R4DArgs* a, hipStream_t s) { return prefill_h128<0>(a, s); }

// Split-KV prefill: the gqa6 contract (caller's segment count, clamped to 64; <= 1 IS the plain launch,
// same bits, no scratch; fp32 partials, fixed-order merge).
#define H_P_MAX_SPLITS 64
static int prefill_splits_h128(const R4DArgs* a) {
    return a->splits < 1 ? 1 : (a->splits > H_P_MAX_SPLITS ? H_P_MAX_SPLITS : a->splits);
}

extern "C" int64_t r4d_attn_prefill_splitkv_h128_gqa7_scratch_bytes(const R4DArgs* a) {
    const int splits = prefill_splits_h128(a);
    if (splits <= 1) return 0;
    const int64_t rows = (int64_t)a->num_seqs * a->q_len * a->q_heads;
    return rows * splits * (int64_t)(a->head_dim + 2) * 4;    // f32 O partials + m + l
}

template <int KVP>
static int prefill_splitkv_h128(const R4DArgs* a, hipStream_t s) {
    if (const int rc = h128_check(a)) return rc;
    const int splits = prefill_splits_h128(a);
    if (splits <= 1) return prefill_h128<KVP>(a, s);
    if (a->scratch == nullptr) return -5;
    prefill_launch_h128<KVP, 1>(a, splits, s);
    CHK();
    const int jobs = a->num_seqs * a->q_len * a->q_heads;
    // The merge runs one thread per output column: HEAD_DIM threads (its reduction buffer is red[HEAD_DIM]).
    hipLaunchKernelGGL((r4d_attn_splitkv_combine_kernel<H_HEAD_DIM, 4, 0>), dim3(jobs), dim3(H_HEAD_DIM),
                       splits * sizeof(float), s, *a, splits, H_P_TILE);
    CHK();
    return 0;
}

extern "C" int r4d_attn_prefill_splitkv_h128_gqa7_bf16kv(const R4DArgs* a, hipStream_t s) {
    return prefill_splitkv_h128<1>(a, s);
}
extern "C" int r4d_attn_prefill_splitkv_h128_gqa7_fp8kv(const R4DArgs* a, hipStream_t s) {
    return prefill_splitkv_h128<0>(a, s);
}

// ---- decode ----------------------------------------------------------------------------------
#ifndef H_D_TILE
#define H_D_TILE 32
#endif
#ifndef H_D_OPT
#define H_D_OPT  3430971            // the gqa6 decode set
#endif
// Workgroups the split law aims for over the non-segment grid units (kv_heads x num_seqs): as h64_gqa4 (a
// decode workgroup is ONE wave for q_len 1 x gqa 7 rows, streaming 8 KiB of KV per 32-key tile).
#ifndef H_D_TARGET_WGS
#define H_D_TARGET_WGS 1024
#endif
#define H_D_MAX_SPLITS 256

// Split law: CU-fill target over the non-segment grid units, snapped down to a power of two, then
// clamped so segments never get thinner than 2 tiles. A function of (max_ctx, kv_heads, num_seqs) only.
static int decode_splits_h128(const R4DArgs* a) {
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

extern "C" int64_t r4d_attn_decode_h128_gqa7_scratch_bytes(const R4DArgs* a) {
    const int splits = decode_splits_h128(a);
    const int64_t rows = (int64_t)a->num_seqs * a->q_len * a->q_heads;
    return (H_D_OPT & 32) ? rows * splits * (int64_t)(a->head_dim * 2 + 8)       // f16 O/l + m + l
                          : rows * splits * (int64_t)(a->head_dim + 2) * 4L;     // f32 O partials + m + l
}

template <int KVP>
static int decode_h128(const R4DArgs* a, hipStream_t s) {
    if (const int rc = h128_check(a)) return rc;
    const int rows = a->q_len * H_GQA;
    if (a->q_len < 1 || rows > H_MAX_DECODE_ROWS) return -4;
    if (a->scratch == nullptr) return -5;
    const int nw = (rows + 15) / 16;
    const int splits = decode_splits_h128(a);

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

extern "C" int r4d_attn_decode_h128_gqa7_bf16kv(const R4DArgs* a, hipStream_t s) { return decode_h128<1>(a, s); }
extern "C" int r4d_attn_decode_h128_gqa7_fp8kv(const R4DArgs* a, hipStream_t s) { return decode_h128<0>(a, s); }