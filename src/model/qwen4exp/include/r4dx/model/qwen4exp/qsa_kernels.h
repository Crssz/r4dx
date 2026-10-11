// qwen4_exp QSA (Qwen Sparse Attention) attention core, one full-attention layer: C ABI of the HIP kernels
// (src/model/qwen4exp/kernels/qsa_*.hip). The host class that sequences them is QsaLayerCore (qsa.h).
// HF source of truth: modeling_qwen4_exp.py `Qwen4ExpTextQSAIndexer` (:672-778) and `Qwen4ExpTextAttention`
// (:818-900); docs/qwen4exp-plan.md 2.4, 4.6, 4.11 item 7.
//
// Plain C ABI like third_party/libr4d/r4d.h and hc_kernels.h: device pointers are typed raw pointers, `stream`
// a hipStream_t, the return value a hipError_t (0 on success; the C++ wrapper throws on a non-zero one).
// bf16 tensors are uint16_t. Every position argument is a device int32 array `positions[T]` with
// positions[t] = first_pos + t (consecutive; the kernels read positions[0] and positions[t]).
//
// Rounding points (HF's bf16 graph; every other op is fp32). Rb = round to nearest even bf16:
//   - per-head RMSNorm (q_norm, k_norm, indexer q_layernorm / k_layernorm):
//         out = Rb( (x * rsqrt(sum(x^2) / D + eps)) * (1 + w) ), fp32 sum in a wave-shuffle order
//   - rope (partial neox half rotation, cos / sin bf16 as HF's `cos.to(x.dtype)`):
//         o[i]     = Rb( Rb(x[i] * c[i])      + Rb(-x[i + h] * s[i]) )
//         o[i + h] = Rb( Rb(x[i + h] * c[i]) + Rb( x[i]     * s[i]) )
//     c, s = Rb(cosf(pos * inv_freq[i])), Rb(sinf(...)), angle in fp32 (HF: freqs = inv_freq @ pos in fp32)
//   - indexer block key: Rb( ((k0 + k1) + k2 + k3) * 0.25 )  (fp32 mean of the 4 raw bf16 keys, cast to bf16),
//     then k_layernorm, then rope at the block's FIRST position
//   - indexer score: fp32 dot of the bf16 q and block key (128 terms, fp32 accumulation), relu per head,
//     summed over the 4 heads in order, divided by 11.3137085f (= sqrt(128), fp32 division)
//   - attention: scores = fp32 dot(q bf16, fp8 K) * (k_descale * 0.0625), fp32 online softmax, fp32 P.V with
//     fp8 V, times v_descale / l; Rb(attn); gate = Rb(sigmoid(gate_bf16)); out = Rb(attn * gate)
//     (HF rounds the scores and the probabilities to bf16 before the P.V matmul; this kernel keeps them fp32)
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

extern "C" {

// ---- 1. projections' epilogue: norm + rope + cache write ---------------------------------------------------
struct Q4QsaPrepArgs {
  // inputs (device bf16): the GEMM outputs of the layer
  const uint16_t* qg;     // [T, heads_q, 2 * head_dim] per-head interleaved (q | gate)
  const uint16_t* k;      // [T, heads_kv, head_dim]
  const uint16_t* v;      // [T, heads_kv, head_dim]
  const uint16_t* idxqk;  // [T, (index_heads + 1) * index_dim]: index_heads query heads then one key head
  // weights
  const uint16_t* q_norm;      // bf16 [head_dim], raw checkpoint weight (1 + w applied here)
  const uint16_t* k_norm;      // bf16 [head_dim]
  const uint16_t* idx_q_norm;  // bf16 [index_dim]
  const float* k_descale;      // fp32 [heads_kv]
  const float* v_descale;      // fp32 [heads_kv]
  const float* inv_freq;       // fp32 [rotary_dim / 2]
  const int32_t* positions;    // int32 [T]
  // outputs
  uint16_t* q_out;      // [T, heads_q, head_dim]   normed, rope'd q
  uint16_t* idx_q_out;  // [T, index_heads, index_dim]  q_layernorm'ed, rope'd indexer queries
  uint16_t* raw_keys;   // [max_ctx, index_dim]  raw indexer key of position p stored at row p
  uint8_t* kv_cache;    // PagedKvCache layout (blocks, kv_heads, block_size, 2 * head_dim)
  int T, heads_q, heads_kv, head_dim, rotary_dim, index_heads, index_dim, block_size;
  float eps;
  int64_t kv_block_stride, kv_head_stride;  // elements (bytes)
};
// grid (T, heads_q + heads_kv + index_heads + 1), one wave per row. block_size is the PagedKvCache block (16).
hipError_t r4dx_q4_qsa_prep(const Q4QsaPrepArgs* a, hipStream_t stream);

// ---- 2. indexer block keys ---------------------------------------------------------------------------------
struct Q4QsaBlockKeyArgs {
  const uint16_t* raw_keys;    // [max_ctx, index_dim]
  const uint16_t* idx_k_norm;  // bf16 [index_dim]
  const float* inv_freq;       // fp32 [rotary_dim / 2]
  const int32_t* positions;    // int32 [T] of the chunk
  uint16_t* block_keys;        // [max_ctx / ratio, index_dim]
  int T, ratio, index_dim, rotary_dim;
  float eps;
};
// Completes every block b with ratio*b + ratio - 1 in [positions[0], positions[T-1]]: pool, k_layernorm, rope at
// ratio*b, store block_keys[b]. Block b = positions[0]/ratio + blockIdx.x; grid ceil(T / ratio) + 1, extra
// blocks return. Needs the raw keys of the chunk written by r4dx_q4_qsa_prep on the same stream.
hipError_t r4dx_q4_qsa_block_keys(const Q4QsaBlockKeyArgs* a, hipStream_t stream);

// ---- 3. scoring + top-k ---------------------------------------------------------------------------------------
struct Q4QsaScoreArgs {
  const uint16_t* idx_q;       // [T, index_heads(4), index_dim(128)] from r4dx_q4_qsa_prep
  const uint16_t* block_keys;  // [blocks, 128]
  const int32_t* positions;    // [T]
  float* scores;               // [rows, score_stride] fp32; row r scores blocks [0, nb(query row0 + r))
  int row0;                    // first chunk row of this sub-batch
  int rows;                    // rows in this sub-batch (<= the scratch rows)
  int blocks_max;              // max over the sub-batch of the complete block count (grid.x sizing)
  int score_stride;
  int ratio, topk;             // rows with nb <= topk are dense and skipped
};
// Rows with more complete blocks nb = (p + 1) / ratio than topk get scores[r][b], b < nb. grid
// (ceil(blocks_max / 256), ceil(rows / rpt)), rpt = 1 or 4 queries per thread (the launcher picks 4 when rows >= 4).
hipError_t r4dx_q4_qsa_scores(const Q4QsaScoreArgs* a, hipStream_t stream);

struct Q4QsaSelectArgs {
  const float* scores;         // same scratch as the score kernel
  const int32_t* positions;    // [T]
  int32_t* sel_tokens;         // [T, sel_stride] selected token indices, ascending, tail last
  int32_t* sel_count;          // [T] token count of the row, -1 = dense row (attends to [0, p])
  int row0, rows, score_stride, sel_stride, ratio, topk;
};
// One workgroup per row: exact top-`topk` blocks by (score descending, block index descending) -- see
// qsa_select.hip for the radix-select algorithm -- written as ascending token lists plus the incomplete tail.
hipError_t r4dx_q4_qsa_select(const Q4QsaSelectArgs* a, hipStream_t stream);

// ---- 4. sparse attention + output gate -------------------------------------------------------------------------
struct Q4QsaAttnArgs {
  const uint16_t* q;             // [T, heads_q, head_dim] from r4dx_q4_qsa_prep
  const uint16_t* qg;            // [T, heads_q, 2 * head_dim]: the gate half is read
  const uint8_t* kv_cache;       // PagedKvCache layout, block_size 16
  const int32_t* block_table;    // [max_blocks] logical block -> physical block
  const int32_t* positions;      // [T]
  const int32_t* sel_tokens;     // [T, sel_stride]
  const int32_t* sel_count;      // [T]
  const float* k_descale;        // [heads_kv]
  const float* v_descale;        // [heads_kv]
  float* part_acc;               // [T, heads_kv, nsplit, group, head_dim] fp32
  float* part_ml;                // [T, heads_kv, nsplit, group, 2] fp32 (running max, sum)
  uint16_t* out;                 // [T, heads_q * head_dim] bf16, gated, ready for o_proj
  int T, heads_q, heads_kv, sel_stride, ratio, topk;
  int tiles_per_split;           // 256-token tiles per workgroup
  int nsplit;                    // grid.x of the main kernel; ceil(max tiles / tiles_per_split)
  int64_t kv_block_stride, kv_head_stride;
};
// heads_q / heads_kv == 12, head_dim == 256, block_size == 16 only. Stage 1: grid (nsplit, heads_kv, T), 256
// threads. Stage 2 (merge + gate): grid (heads_q, T).
hipError_t r4dx_q4_qsa_attn(const Q4QsaAttnArgs* a, hipStream_t stream);

}  // extern "C"
