// QsaLayerCore: the attention core of one qwen4_exp full-attention (QSA) layer -- everything between the
// q/k/v/index projections and o_proj. docs/qwen4exp-plan.md 2.4, 4.6, 4.11 item 7. Kernel contracts and every
// rounding point: qsa_kernels.h.
//
//   caller:   qg = attn.qg GEMM, k = attn.k GEMM, v = attn.v GEMM, idxqk = attn.indexer.qk GEMM   (bf16)
//   Forward:  q/k RMSNorm + partial rope + fp8 KV write; indexer q norm + rope; raw index keys; block keys of the
//             blocks completing in the chunk; per-query top-(budget / ratio) block selection (skipped while the
//             query sees at most budget / ratio complete blocks: then attention is dense over [0, p]); sparse
//             attention over the selected tokens + the incomplete tail; output gate.
//   returns:  attn_out [T, heads_q * head_dim] bf16 (gated), the input of o_proj (done by the caller, ApplyLinear).
//
// State owned by the object (per layer, device): raw index keys [max_ctx, 128] (row p = key of position p),
// block keys [max_ctx / ratio, 128] (row b = pooled, normed, roped key of positions ratio*b .. ratio*b + ratio - 1,
// valid once those positions are written), and the selection scratch. The fp8 KV lives in the caller's PagedKvCache.
//
// Position bookkeeping: ValidPositions() is the number of leading positions whose state is trusted. Forward
// requires first_pos <= ValidPositions() (no holes), rewrites the raw keys / KV rows of [first_pos, first_pos + T)
// and every block key completing in the chunk, and afterwards ValidPositions() == first_pos + T: state beyond is
// overwritten before it is read, never trusted. TruncateTo(pos) rewinds (prefix reuse); the first Forward after it
// may start anywhere <= pos, including in the middle of a block.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/model/attention/paged_kv_cache.hpp"

namespace r4dx::model::qwen4exp {

struct QsaDims {
  int heads_q = 24;
  int heads_kv = 2;
  int head_dim = 256;
  int rotary_dim = 64;       // partial rotary 0.25 * head_dim, plain neox half rotation (text-only mrope)
  double rope_theta = 1.0e7;
  float rms_eps = 1e-6f;
  int index_heads = 4;
  int index_dim = 128;
  int compress_ratio = 4;    // tokens per block
  int budget = 2048;         // selected tokens (indexer_budget); top budget / compress_ratio blocks
  int max_ctx = 262144;      // positions the layer can hold; raw / block key buffers are sized from it
};

// Device pointers of the layer's weights (bf16 raw checkpoint norm weights, fp32 descales).
struct QsaWeights {
  const uint16_t* q_norm = nullptr;      // [head_dim]  attn.q_norm
  const uint16_t* k_norm = nullptr;      // [head_dim]  attn.k_norm
  const uint16_t* idx_q_norm = nullptr;  // [index_dim] attn.indexer.q_norm
  const uint16_t* idx_k_norm = nullptr;  // [index_dim] attn.indexer.k_norm
  const float* k_descale = nullptr;      // [heads_kv]  attn.k_descale
  const float* v_descale = nullptr;      // [heads_kv]  attn.v_descale
};

// The four projection outputs of the chunk (device bf16, row-major, row t = position first_pos + t).
struct QsaInputs {
  const uint16_t* qg = nullptr;     // [T, heads_q * 2 * head_dim]  per-head interleaved q | gate
  const uint16_t* k = nullptr;      // [T, heads_kv * head_dim]
  const uint16_t* v = nullptr;      // [T, heads_kv * head_dim]
  const uint16_t* idxqk = nullptr;  // [T, (index_heads + 1) * index_dim]
};

class QsaLayerCore {
 public:
  static constexpr int kMaxChunk = 256;  // queries per Forward

  // Throws std::invalid_argument for a geometry the kernels do not serve (heads_q / heads_kv == 12, head_dim 256,
  // index_dim 128, even rotary_dim <= 64, budget a positive multiple of compress_ratio, max_ctx > 0).
  explicit QsaLayerCore(const QsaDims& dims);

  const QsaDims& Dims() const { return dims_; }

  // Arena bytes one Forward(T) allocates (q, indexer q, attn_out, split partials, alignment padding).
  size_t ArenaBytes(int T) const;

  // Forget everything: ValidPositions() == 0.
  void Reset() { valid_pos_ = 0; }
  // Rewind to `pos` leading positions (pos >= 0): ValidPositions() = min(ValidPositions(), pos).
  void TruncateTo(int pos);
  int ValidPositions() const { return valid_pos_; }

  // Runs the chunk [first_pos, first_pos + T), 1 <= T <= kMaxChunk, and returns attn_out [T, heads_q * head_dim]
  // bf16 allocated from `arena` (valid until the caller resets the arena). positions: device int32 [T] with
  // positions[t] == first_pos + t (used by the kernels for rope, cache slots and the per-query block counts).
  // kv: this layer's cache (kv_heads, head_dim, block_size 16, capacity >= first_pos + T). Throws
  // std::runtime_error / std::out_of_range on a violated precondition or a failed launch.
  uint16_t* Forward(hipStream_t stream, core::Arena& arena, const QsaInputs& in, const QsaWeights& w,
                    attention::PagedKvCache& kv, int first_pos, int T, const int32_t* positions);

  // ---- diagnostics / tests ----
  // Selection of the LAST Forward, rows = chunk rows: sel_tokens [kMaxChunk, SelectStride()] int32 (ascending
  // token indices, the incomplete tail last) and sel_count [kMaxChunk]: the token count, valid for the rows whose
  // query saw more than budget / ratio complete blocks (the others attend densely to [0, p]; their entries are
  // unspecified).
  const int32_t* SelectedTokens() const { return sel_tokens_.data(); }
  const int32_t* SelectedCounts() const { return sel_count_.data(); }
  int SelectStride() const { return sel_stride_; }
  const uint16_t* RawKeys() const { return raw_keys_.data(); }
  const uint16_t* BlockKeys() const { return block_keys_.data(); }
  int TopkBlocks() const { return topk_; }
  int ScoreRows() const { return score_rows_; }
  // The fp32 rope inverse frequencies, as uploaded (host copy).
  const float* InvFreqHost() const { return inv_freq_host_; }
  // The indexer queries (q_layernorm'ed, rope'd, bf16 [T, index_heads, index_dim]) of the last Forward: points into
  // the arena passed to it, valid until the caller resets that arena.
  const uint16_t* LastIndexQueries() const { return last_idx_q_; }

 private:
  QsaDims dims_;
  int topk_ = 0;          // budget / ratio
  int max_blocks_ = 0;    // max_ctx / ratio
  int sel_stride_ = 0;    // budget + ratio - 1
  int score_rows_ = 0;    // query rows scored per pass (score scratch rows)
  int ntiles_max_ = 0;    // ceil((budget + ratio - 1) / 256)
  int valid_pos_ = 0;
  float inv_freq_host_[32] = {};
  const uint16_t* last_idx_q_ = nullptr;
  core::DeviceBuffer<float> inv_freq_;
  core::DeviceBuffer<uint16_t> raw_keys_;
  core::DeviceBuffer<uint16_t> block_keys_;
  core::DeviceBuffer<float> scores_;
  core::DeviceBuffer<int32_t> sel_tokens_;
  core::DeviceBuffer<int32_t> sel_count_;
};

}  // namespace r4dx::model::qwen4exp
