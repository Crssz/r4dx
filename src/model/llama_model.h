// r4dx::model::LlamaModel -- the single-device dense Llama (`llama`, Nanbeige4.3-33B) text engine (docs/nanbeige-plan.md
// 3.1-3.3, task N1-6), and one tensor-parallel rank of it (N1-7; LlamaTpModel drives two). A separate class from Model,
// GemmaModel and Lfm2Model on purpose: every layer is full GQA attention + a SwiGLU MLP, one sequence, no speculation,
// no vision. Its public surface is Lfm2Model's minus the ShortConv / MoE debug hooks, so LlamaTpModel is a copy-adapt
// of Lfm2TpModel.
//
// Per decoder layer (bf16 residual stream, docs/nanbeige-plan.md 3.2):
//     x = input_layernorm(h)                        r4dx_rmsnorm_plain_bf16 (plain weight, fp32, one rounding)
//     m = LlamaAttnLayer(x)                         q/k/v, rope, bf16 KV, reference attention, o_proj
//     [TP] all-reduce(m);  h = h + m;  x = post_attention_layernorm(h)    (fused residual + norm)
//     f = down(silu(gate(x)) * up(x))               mlp.gate_up gate rows first
//     [TP] all-reduce(f);  h = h + f
// then the final norm on the last row, the untied lm_head (the rank's vocab slice under TP), fp32 logits. Columns >=
// RealVocab() are -inf when a mask is set (off by default: HF does not mask, and the KL gate compares against HF).
//
// State: KV (bf16, contiguous, one cache per layer) is positional, so SaveCheckpoint / RestoreCheckpoint only move the
// position (prompt_checkpoint): nothing is copied.
#pragma once

#include <cstdint>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "llama_container.h"
#include "model_types.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/summary_sampler.hpp"
#include "r4dx/model/attention/llama_attention_layer.hpp"
#include "tp/tp_submit.h"  // tp::SubmitBounder (inert at TP=1)
#include "tp/tp_vocab.h"   // tp::ArgmaxPair, the vocab-split merges

namespace r4dx::model {

class SpanAccumulator;  // profile_span.h

struct LlamaModelOptions {
  std::string container_path;
  Layout layout = Layout::kBf16;
  int64_t layer_limit = -1;     // tiny fixtures, nanbeige-l8
  int64_t max_ctx = 0;          // <= 0: LlamaConfig::ResolveMaxCtx's default (min(32768, max_position_embeddings))
  int prefill_chunk = 256;      // rows per prefill chunk, [1, 256]
  bool prompt_checkpoint = false;  // SaveCheckpoint / RestoreCheckpoint allowed (positional: no copy)
  // Logit columns >= real_vocab are -inf. 0 (default): no mask; > 0: that count. R4DX_LLAMA_REAL_VOCAB overrides.
  int64_t real_vocab = 0;
  // Tensor parallel (LlamaTpModel fills these per rank; nothing else should): rank `tp_rank` of `tp_world` (1 or 2), the
  // row-parallel sublayer outputs (attn.o, mlp.down) all-reduced through `tp_comm`.
  int tp_world = 1;
  int tp_rank = 0;
  core::TpComm* tp_comm = nullptr;
  int tp_submit_layers = 0;
  int tp_max_inflight_units = 0;
};

// What DebugLayerForward reports besides the layer output (host copies; device synchronized).
struct LlamaLayerDebug {
  std::vector<uint16_t> input_norm_out;  // [T, hidden]
  std::vector<uint16_t> attn_out;        // [T, hidden] (after the TP all-reduce)
  std::vector<uint16_t> mid_residual;    // [T, hidden]
  std::vector<uint16_t> post_norm_out;   // [T, hidden]
  std::vector<uint16_t> mlp_out;         // [T, hidden]
};

class LlamaModel {
 public:
  static LlamaModel Load(const LlamaModelOptions& opts);

  LlamaModel(LlamaModel&&) = default;
  LlamaModel& operator=(LlamaModel&&) = default;

  const LlamaConfig& Config() const { return container_.Config(); }  // rank-local under TP (vocab_size global)
  const LlamaContainer& GetContainer() const { return container_; }
  int64_t MaxCtx() const { return max_ctx_; }
  int64_t PositionCount() const { return pos_; }
  int64_t SampledFallbackRows() const { return sampled_fallback_rows_; }
  int64_t KvBytes() const { return kv_bytes_; }
  int64_t RealVocab() const { return real_vocab_; }  // 0 = no mask
  int TpWorld() const { return opts_.tp_world; }
  int TpRank() const { return opts_.tp_rank; }
  int64_t VocabLocal() const { return vocab_local_; }
  tp::SubmitBounder::Stats TpSubmitStats() const { return submit_.GetStats(); }

  // TP only: the load-time warm-up (docs/tp.md 2.9 step 9): one full prefill chunk, a 64-row prefill, a greedy, a sampled
  // and a full-logits step. Ends with Reset(). A lockstep collective.
  void TpWarmup();
  static int64_t WarmupPositions(const LlamaModelOptions& o);

  void Reset();
  void SaveCheckpoint();     // needs prompt_checkpoint: records the position (KV is positional)
  void RestoreCheckpoint();
  int64_t CheckpointPosition() const { return ckpt_pos_; }

  // Feeds `token_ids` in chunks of <= prefill_chunk rows; returns the fp32 logits[vocab] of the last row.
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids);
  std::vector<float> DecodeStep(int32_t token_id);
  int32_t DecodeStepGreedy(int32_t token_id);
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng);
  StepProfile DecodeStepProfiled(int32_t token_id);
  StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids);

  // ---- rung-3 test hook (tests/model/test_llama_layers.cpp; TP=1 only) ---------------------------------------------
  // Decoder layer `layer` alone on `x_rows` ([T, hidden] bf16 host, T <= prefill_chunk) at absolute position
  // `start_pos`: the layer's own KV continues from earlier calls. Returns the layer output. Does not touch
  // PositionCount().
  std::vector<uint16_t> DebugLayerForward(int64_t layer, const std::vector<uint16_t>& x_rows, int64_t T, int64_t start_pos,
                                          LlamaLayerDebug* dbg = nullptr);

 private:
  LlamaModel() = default;
  struct SummaryRequest {
    float inv_temperature = 1.0f;
    kernels::RowSummary* out = nullptr;
  };
  std::vector<float> RunChunk(const std::vector<int32_t>& token_ids, bool want_logits, int32_t* greedy_out,
                              const SummaryRequest* summary_out, SpanAccumulator* prof, bool is_prefill = false);
  void RunLayer(int64_t i, uint16_t* h, int64_t T, int64_t start_pos, SpanAccumulator* prof,
                LlamaLayerDebug* dbg = nullptr);
  void UploadPositions(int64_t start_pos, int64_t T);
  void MaskPaddingLogits(float* logits_local, int64_t rows, int64_t row_stride);
  int32_t MergeGreedyPair();
  void MergeSummary(kernels::RowSummary* out);
  void GatherVocabRow(std::vector<float>* full);
  void FetchRowSummary(float inv_temperature, kernels::RowSummary* out);
  void RequireNotTp(const char* what) const;

  LlamaModelOptions opts_;
  LlamaContainer container_;
  core::Stream stream_;
  core::Arena arena_;
  int64_t max_ctx_ = 0;
  int max_chunk_ = 256;
  int64_t pos_ = 0;
  bool started_ = false;
  int64_t sampled_fallback_rows_ = 0;
  int64_t kv_bytes_ = 0;
  int64_t real_vocab_ = 0;

  core::TpComm* comm_ = nullptr;
  int64_t vocab_local_ = 0, vocab_offset_ = 0;
  tp::SubmitBounder submit_;
  int submit_layers_ = 0;
  core::DeviceBuffer<int32_t> argmax_pair_dev_;
  core::DeviceBuffer<uint8_t> topk_lse_ws_;
  std::vector<float> gather_shard_host_;

  std::vector<attention::LlamaAttnLayer> attn_;   // per layer
  std::vector<attention::GemmaKvCache> kv_;       // per layer
  int64_t ckpt_pos_ = -1;

  core::PinnedBuffer<int32_t> ids_host_;
  core::DeviceBuffer<int32_t> ids_dev_, positions_dev_;
  core::DeviceBuffer<uint16_t> buf_h_;  // bf16 residual [max_chunk, hidden]
  core::DeviceBuffer<float> logits_dev_;
  core::DeviceBuffer<int32_t> argmax_dev_, summary_ids_dev_;
  core::DeviceBuffer<float> summary_vals_dev_, summary_lse_dev_;
  std::vector<float> sampled_row_scratch_;
};

}  // namespace r4dx::model
