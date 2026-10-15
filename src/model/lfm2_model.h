// r4dx::model::Lfm2Model -- the single-device LFM2 (lfm2_moe, LFM2.5-24B-A2B) text engine (docs/lfm2-plan.md 3.1-3.5,
// task L1-8), and one tensor-parallel rank of it (L1-10; Lfm2TpModel drives two). A separate class from Model and
// GemmaModel on purpose: 40 layers of {ShortConv | attention} mixers + {dense SwiGLU | 64-expert MoE} FFNs, one
// sequence, no speculation, no vision. Its public surface is GemmaModel's minus vision / audio / features / DFlash, so
// Lfm2TpModel is a copy-adapt of GemmaTpModel.
//
// Per decoder layer (docs/lfm2-semantics.md 1; bf16 residual stream, docs/lfm2-plan.md 3.2):
//     x = operator_norm(h)                          r4dx_rmsnorm_plain_bf16 (plain weight, eps 1e-5, fp32, one rounding)
//     m = conv layers: Lfm2ConvLayer(x)             in_proj -> fused ShortConv (+ the layer's [2, C] state) -> out_proj
//         attention:   Lfm2AttnLayer(x)             q/k/v, per-head q/k norm, rope theta 1e7, bf16 KV, reference attn
//     [TP] all-reduce(m);  h = h + m                r4dx_residual_add_bf16
//     x = ffn_norm(h)
//     f = dense (i < 2): Lfm2DenseMlp(x)            gate_up -> silu_mul -> down
//         MoE:           Lfm2MoeFfn(x)              fp32 router, staged experts (lfm2_layers.h), fp32 combine
//     [TP] all-reduce(f);  h = h + f
// then embedding_norm (the FINAL norm) on the last row, the tied lm_head (the rank's vocab slice under TP), fp32 logits,
// no softcap, columns >= RealVocab() set to -inf (the padding rows of the embedding table: 125017..127999 for the real
// checkpoint, docs/lfm2-plan.md 11 Q10 decision).
//
// State: KV (bf16, contiguous, one cache per attention layer) is positional; the ShortConv state ([n_conv][2][C] bf16)
// is recurrent: Reset zeroes it, SaveCheckpoint / RestoreCheckpoint D2D-copy it (prompt_checkpoint), the KV needs no
// copy (docs/lfm2-plan.md 3.3).
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "lfm2_container.h"
#include "lfm2_layers.h"
#include "model_types.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/summary_sampler.hpp"
#include "r4dx/model/attention/lfm2_attention_layer.hpp"
#include "tp/tp_submit.h"  // tp::SubmitBounder (inert at TP=1)
#include "tp/tp_vocab.h"   // tp::ArgmaxPair, the vocab-split merges

namespace r4dx::model {

class SpanAccumulator;  // profile_span.h

// The real LFM2.5 tokenizer's id count (max id 124906 + the added tokens up to 125016): logits at or above it are padding
// rows of the 128000-row embedding table and are masked (decision 2026-10-14).
constexpr int64_t kLfm2RealVocab = 125017;

struct Lfm2ModelOptions {
  std::string container_path;
  Layout layout = Layout::kBf16;
  int64_t layer_limit = -1;     // tiny fixtures, lfm2-l6
  int64_t max_ctx = 0;          // <= 0: the config's max_position_embeddings (131072)
  int prefill_chunk = 256;      // rows per prefill chunk, [1, 256]
  bool prompt_checkpoint = false;  // a spare copy of the ShortConv state for SaveCheckpoint/RestoreCheckpoint
  // Logit columns >= real_vocab are -inf. < 0 (default): kLfm2RealVocab when the config's vocab_size is the real
  // checkpoint's 128000, else no mask (tiny fixtures); 0: no mask; > 0: that count. R4DX_LFM2_REAL_VOCAB overrides.
  int64_t real_vocab = -1;
  // Tensor parallel (Lfm2TpModel fills these per rank; nothing else should): rank `tp_rank` of `tp_world` (1 or 2), the
  // row-parallel sublayer outputs (attn.o, conv.out_proj, mlp.down, the MoE combine) all-reduced through `tp_comm`.
  int tp_world = 1;
  int tp_rank = 0;
  core::TpComm* tp_comm = nullptr;
  int tp_submit_layers = 0;
  int tp_max_inflight_units = 0;
};

// What DebugLayerForward reports besides the layer output (host copies; device synchronized).
struct Lfm2LayerDebug {
  std::vector<uint16_t> operator_norm_out;  // [T, hidden]
  std::vector<uint16_t> mixer_out;          // [T, hidden] (after the TP all-reduce)
  std::vector<uint16_t> mid_residual;       // [T, hidden]
  std::vector<uint16_t> ffn_norm_out;       // [T, hidden]
  std::vector<uint16_t> ffn_out;            // [T, hidden]
  Lfm2MoeDebug moe;                         // MoE layers: router ids / weights
};

class Lfm2Model {
 public:
  static Lfm2Model Load(const Lfm2ModelOptions& opts);

  Lfm2Model(Lfm2Model&&) = default;
  Lfm2Model& operator=(Lfm2Model&&) = default;

  const Lfm2Config& Config() const { return container_.Config(); }  // rank-local under TP (vocab_size global)
  const Lfm2Container& GetContainer() const { return container_; }
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
  // and a full-logits step, and one MoE pass with forced routing that sends rows to EVERY expert of the first MoE layer
  // (the sorted path's per-expert GEMM shapes at several M). Ends with Reset(). A lockstep collective.
  void TpWarmup();
  static int64_t WarmupPositions(const Lfm2ModelOptions& o);

  void Reset();
  void SaveCheckpoint();     // needs prompt_checkpoint: D2D copy of the ShortConv state
  void RestoreCheckpoint();
  int64_t CheckpointPosition() const { return ckpt_pos_; }

  // Feeds `token_ids` in chunks of <= prefill_chunk rows; returns the fp32 logits[vocab] of the last row.
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids);
  std::vector<float> DecodeStep(int32_t token_id);
  int32_t DecodeStepGreedy(int32_t token_id);
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng);
  StepProfile DecodeStepProfiled(int32_t token_id);
  StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids);

  // ---- rung-3 test hooks (tests/model/test_lfm2_layers.cpp; TP=1 only) ---------------------------------------------
  // Decoder layer `layer` alone on `x_rows` ([T, hidden] bf16 host, T <= prefill_chunk) at absolute position
  // `start_pos`: the layer's own KV / ShortConv state continue from earlier calls; start_pos == 0 zeroes that layer's
  // ShortConv state first. Returns the layer output. Does not touch PositionCount(). `moe_path` picks the MoE path.
  std::vector<uint16_t> DebugLayerForward(int64_t layer, const std::vector<uint16_t>& x_rows, int64_t T, int64_t start_pos,
                                          Lfm2LayerDebug* dbg = nullptr, Lfm2MoePath moe_path = Lfm2MoePath::kAuto);
  // The FFN of `layer` alone on `x_normed_rows` ([T, hidden], already ffn_norm'ed). Returns the FFN output.
  std::vector<uint16_t> DebugFfnForward(int64_t layer, const std::vector<uint16_t>& x_normed_rows, int64_t T,
                                        Lfm2MoePath moe_path = Lfm2MoePath::kAuto, Lfm2MoeDebug* dbg = nullptr);
  // The ShortConv state of conv layer `layer` (host bf16 [2, conv_dim]).
  std::vector<uint16_t> DebugConvState(int64_t layer);

 private:
  Lfm2Model() = default;
  struct SummaryRequest {
    float inv_temperature = 1.0f;
    kernels::RowSummary* out = nullptr;
  };
  std::vector<float> RunChunk(const std::vector<int32_t>& token_ids, bool want_logits, int32_t* greedy_out,
                              const SummaryRequest* summary_out, SpanAccumulator* prof, bool is_prefill = false);
  void RunLayer(int64_t i, uint16_t* h, int64_t T, int64_t start_pos, SpanAccumulator* prof,
                Lfm2LayerDebug* dbg = nullptr, Lfm2MoePath moe_path = Lfm2MoePath::kAuto);
  void RunFfn(int64_t i, const uint16_t* x_normed, uint16_t* out, int64_t T, SpanAccumulator* prof, Lfm2MoePath path,
              Lfm2MoeDebug* dbg, const std::vector<int32_t>* forced_ids = nullptr,
              const std::vector<float>* forced_w = nullptr);
  void UploadPositions(int64_t start_pos, int64_t T);
  uint16_t* ConvState(int64_t layer);  // device [2, conv_dim] of conv layer `layer`
  void MaskPaddingLogits(float* logits_local, int64_t rows, int64_t row_stride);
  int32_t MergeGreedyPair();
  void MergeSummary(kernels::RowSummary* out);
  void GatherVocabRow(std::vector<float>* full);
  void FetchRowSummary(float inv_temperature, kernels::RowSummary* out);
  void RequireNotTp(const char* what) const;

  Lfm2ModelOptions opts_;
  Lfm2Container container_;
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

  std::vector<std::optional<attention::Lfm2AttnLayer>> attn_;  // per layer (attention layers only)
  std::vector<std::optional<attention::GemmaKvCache>> kv_;     // per layer (attention layers only)
  std::vector<int64_t> conv_slot_;                             // per layer: index into the conv state, -1 = attention
  int64_t n_conv_ = 0;
  core::DeviceBuffer<uint16_t> conv_state_, conv_state_ckpt_;  // [n_conv][2][conv_dim] bf16
  int64_t ckpt_pos_ = -1;
  Lfm2MoeHost moe_host_;

  core::PinnedBuffer<int32_t> ids_host_;
  core::DeviceBuffer<int32_t> ids_dev_, positions_dev_;
  core::DeviceBuffer<uint16_t> buf_h_;  // bf16 residual [max_chunk, hidden]
  core::DeviceBuffer<float> logits_dev_;
  core::DeviceBuffer<int32_t> argmax_dev_, summary_ids_dev_;
  core::DeviceBuffer<float> summary_vals_dev_, summary_lse_dev_;
  std::vector<float> sampled_row_scratch_;
};

}  // namespace r4dx::model
