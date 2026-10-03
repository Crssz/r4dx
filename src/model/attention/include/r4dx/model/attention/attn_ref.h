// Stage-0 reference attention for Gemma 4 (docs/gemma4-plan.md 3.3 "Kernel plan, Stage 0"): the
// correctness oracle that unblocks the rung-3 goldens and the KL gate independently of the libr4d
// perf kernels. r4dx-owned, generic over head_dim <= 512, GQA ratio, sliding window, softmax scale,
// causal + per-row bidirectional extension (`klimit_ext`), fp8 or bf16 paged KV, and the ring block
// table. One workgroup per (query row, q head), no WMMA, fp32 scores with a numerically stable
// two-pass softmax (max, then exp-sum and P.V recomputing the scores): slow at long context (every
// q head of a GQA group re-reads the KV; 8 full layers x 262k keys is minutes), exact in the sense a
// golden needs.
//
// One sequence per call (num_seqs = 1), the shape the per-layer goldens and the KL harness use. KV
// layout is R4DArgs.kv's: (num_blocks, kv_heads, block_size, 2*head_dim), K at columns [0, head_dim)
// and V at [head_dim, 2*head_dim) of each slot; key kpos lives in block block_table[kpos / block_size]
// at slot kpos % block_size. Strides are in ELEMENTS of the cache dtype (uint8 fp8 / uint16 bf16).
//
// Visibility of query row t (absolute position qpos = ctx - q_len + t):
//     klow  = window > 0 ? max(0, qpos - window + 1) : 0
//     khigh = max(qpos, klimit_ext[t])  (klimit_ext < 0 or null: qpos), clamped to ctx - 1
//     keys klow..khigh. The mask is the same r4d_attn_window.h's (the kernels' contract).
// Scores: s = (q . k) * scale (* k_descale[kvh] for fp8); out = softmax(s) . v (* v_descale[kvh]).
// k_descale / v_descale: fp32 [kv_heads], null = 1.
#pragma once

#include <cstdint>

extern "C" {

struct R4dxGemmaAttnRefArgs {
  int64_t q;            // [q_len, q_heads, head_dim] bf16
  int64_t kv;           // paged cache, see above
  int64_t block_table;  // int32 [>= ceil(ctx / block_size)] (contiguous or the ring's shared table)
  int64_t out;          // [q_len, q_heads, head_dim] bf16
  int64_t k_descale;    // fp32 [kv_heads] or 0
  int64_t v_descale;    // fp32 [kv_heads] or 0
  int64_t klimit_ext;   // int32 [q_len] absolute key limit per row, or 0
  int q_len, q_heads, kv_heads, head_dim, block_size;
  int ctx;              // keys 0..ctx-1 exist; the q_len queries are the last q_len positions
  int window;           // sliding window W (> 0), or 0 = full causal
  float scale;          // 1.0 for Gemma 4
  int64_t kv_block_stride;  // elements between consecutive blocks
  int64_t kv_head_stride;   // elements between kv heads inside a block
};

// Return 0 on success; negative for a shape it does not serve (head_dim not in [8, 512] or not a
// multiple of 8, q_heads % kv_heads != 0, non-positive sizes, window < 0, ctx < q_len).
int r4dx_gemma_attn_ref_fp8kv(const R4dxGemmaAttnRefArgs* a, int64_t stream);
int r4dx_gemma_attn_ref_bf16kv(const R4dxGemmaAttnRefArgs* a, int64_t stream);

}  // extern "C"
