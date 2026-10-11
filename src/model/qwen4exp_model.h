// r4dx::model::Qwen4ExpModel -- the qwen4_exp (Qwen3.8-Flash-Next) forward pass over one or two devices
// (docs/qwen4exp-plan.md 3.3, 4.5, 4.11 item 4, tasks M1-16 .. M1-18).
//
// Two Qwen4ExpStage objects (qwen4exp_stage.h): stage 0 holds layers [0, k) -- the host embedding gather, the n-gram rows
// and the PLE layer are driven from here -- and stage 1 layers [k, L), the final mixer and the lm_head. k comes from the
// loader's ChooseSplit (bytes balanced per card) unless asked for. The two are driven in sequence on the calling thread:
// stage 0 runs a chunk, its residual [rows, 4 * hidden] bf16 goes through pinned host memory (D2H on stage 0's
// stream, synchronize, H2D on stage 1's), stage 1 runs it. When both stages sit on one device and no split was asked for
// there is a single stage with everything. The hand-off copies bytes, so a two-stage run equals the one-stage run bit for
// bit.
//
// A chunk is up to 256 rows (QSA's limit; ApplyLinear slices M internally): a prompt is prefilled in chunks of 256, a
// decode step is a chunk of one row. Per chunk: the n-gram ids of the tokens (host, NgramIds) -> asynchronous NVMe row
// reads into pinned memory (NgramReader) that overlap the layers below the PLE layer -> the embedding gather ->
// the stages. Logits come from the LAST row of the last chunk only (the mixer and the lm_head are row-independent).
//
// State: GDN recurrent + conv state and the PLE conv state per layer, the paged fp8 KV and QSA block-key buffers of the
// attention layers, the 2-token n-gram history, and the position. Reset() clears them; SaveCheckpoint /
// RestoreCheckpoint keep Qwen3.5's contract (ModelOptions::prompt_checkpoint, one slot, valid right after a Prefill):
// the checkpoint copies the GDN + PLE state and the n-gram history, and a restore rewinds the position and the QSA
// bookkeeping (KV and block keys beyond the position are overwritten before they are read).
//
// Not thread-safe; callers must serialize. Both stages are driven in sequence on the calling thread. After an exception
// out of Prefill / Decode*, call Reset().
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "model_config.h"
#include "qwen4exp_config.h"
#include "qwen4exp_container.h"
#include "text_model.h"  // VramReport
#include "qwen4exp_stage.h"
#include "quant_linear.h"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/kernels/sampler.hpp"
#include "r4dx/model/qwen4exp/ngram_ids.h"
#include "r4dx/model/qwen4exp/ngram_reader.h"
#include "r4dx/model/qwen4exp/ngram_sidecar.h"
namespace r4dx::model {

struct Qwen4ExpModelOptions {
  std::string container_path;
  // Body layout (GDN / QSA / expert linears): the container's own -- trellis for a trellis container, bf16 or w4a16 for
  // the tiny ones (Qwen4ExpContainer::Load refuses a mismatch). lm_head: empty = the container's.
  Layout layout = Layout::kBf16;
  std::optional<Layout> lm_head_layout;
  int64_t max_ctx = 262144;      // KV / QSA capacity in tokens (<= the checkpoint's max_position_embeddings)
  bool prompt_checkpoint = false;
  // HIP ordinals {stage 0, stage 1}. Empty = R4DX_Q4_DEVICES ("a,b"), else {0, 1} when two devices are visible, else {0, 0}.
  std::vector<int> devices;
  // The layer boundary k of a two-stage run. 0 = R4DX_Q4_SPLIT, else ChooseSplit when the devices differ, else no split
  // (one stage). A non-zero split with equal devices runs two stages on one device (tests).
  int64_t split = 0;
  int64_t weight_cap_bytes = kQwen4ExpDefaultWeightCapBytes;  // ChooseSplit's per-card cap
  // GDN layers prefill through the recurrent decode kernels, one token at a time, instead of libr4d's chunked scan: the slow
  // reference path (see Qwen4ExpGdnCall::prefill_recurrent). Also on when R4DX_Q4_GDN_PREFILL=recurrent.
  bool gdn_prefill_recurrent = false;
};

class Qwen4ExpModel {
 public:
  static constexpr int64_t kChunkRows = 256;

  // Loads the container (one range per stage), builds the stages and the n-gram path, zeroes the state. Prints one
  // line per stage to stderr (layers, device, bytes).
  static std::unique_ptr<Qwen4ExpModel> Load(const Qwen4ExpModelOptions& options);

  ~Qwen4ExpModel();
  Qwen4ExpModel(const Qwen4ExpModel&) = delete;
  Qwen4ExpModel& operator=(const Qwen4ExpModel&) = delete;

  const Qwen4ExpConfig& Config() const { return cfg_; }
  const ModelConfig& GenericConfig() const { return generic_config_; }
  const std::string& ModelId() const { return model_id_; }
  int NumStages() const { return static_cast<int>(stages_.size()); }
  int64_t SplitLayer() const { return split_; }  // k; == num_hidden_layers for a single stage
  int StageDevice(int stage) const { return stages_.at(static_cast<size_t>(stage))->Device(); }
  int64_t StageFirstLayer(int stage) const { return stages_.at(static_cast<size_t>(stage))->FirstLayer(); }
  int64_t StageEndLayer(int stage) const { return stages_.at(static_cast<size_t>(stage))->EndLayer(); }
  int64_t NumLoadedLayers() const { return cfg_.num_hidden_layers; }
  int64_t StageWeightBytes(int stage) const { return containers_.at(static_cast<size_t>(stage))->DeviceBytes(); }
  int64_t PositionCount() const { return pos_; }
  int64_t MaxCtx() const { return opt_.max_ctx; }
  int64_t SampledFallbackRows() const { return sampled_fallback_rows_; }

  // ---- sequence state -----------------------------------------------------------------------------------------------
  void Reset();
  void SaveCheckpoint();
  void RestoreCheckpoint();

  // ---- forward --------------------------------------------------------------------------------------------------------
  // Feeds `token_ids` (>= 1) in chunks of 256 and returns the fp32 logits [vocab] of the last one (the lm_head's bf16
  // output widened). Continues the sequence at PositionCount().
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids);
  std::vector<float> DecodeStep(int32_t token_id);
  int32_t DecodeStepGreedy(int32_t token_id);
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng);
  // ---- diagnostics / tests --------------------------------------------------------------------------------------------
  // Installs `fn` on every stage (see Qwen4ExpCapture). With a capture installed every chunk of a Prefill also runs the
  // head so `mixer_out` / `logits` cover all rows. An empty function removes it.
  void SetCapture(Qwen4ExpCaptureFn fn);

  // Layer-isolated forward (tests, docs/qwen4exp-plan.md 4.11): Reset()s the sequence, then feeds `token_ids` in chunks of 256
  // through ONLY layer `layer` (global index): each chunk's residual input is the caller's `residual_in` rows
  // [T, hc * hidden] bf16 (HOST; e.g. the HF golden's `layer{L}.residual_in`), not the previous layer's output, so the layer
  // is judged on its own error. The layer's own state (GDN recurrent / conv state, PLE conv state, KV, QSA block keys) carries
  // from chunk to chunk exactly as in a prefill; the tokens only feed the n-gram rows of a PLE layer. No logits. Captures
  // (SetCapture) see that layer's points; call Reset() before the next normal Prefill.
  void RunLayerIsolated(const std::vector<int32_t>& token_ids, int64_t layer, const uint16_t* residual_in);

  struct Timings {
    double total_ms = 0;       // the whole call
    double embed_ms = 0;       // n-gram ids + submit + embedding gather (host work before the first stage)
    // RunLayers per stage, host wall. A prefill chunk synchronizes inside RunLayers (full time); a decode step only enqueues,
    // so there it is the host time to issue the stage's layers (+ the PLE rows wait) and the GPU time shows up in
    // handoff_ms (stage 0: the wait for its stream) and head_ms (the last stage).
    double stage_ms[2] = {0, 0};
    double ple_wait_ms = 0;    // time the PLE layer waited for the n-gram reader
    double handoff_ms = 0;     // host wall of the residual hand-off between the stages (includes the wait for stage 0's GPU)
    double head_ms = 0;        // final mixer + lm_head + readback (host wall, includes the wait for the last stage's GPU)
    double gpu_ms[2] = {0, 0}; // R4DX_Q4_PROFILE=1 only: stream time of each stage's RunLayers (kernels + launch gaps)
    double head_gpu_ms = 0;    // R4DX_Q4_PROFILE=1 only: stream time of the argmax head (greedy decode steps)
    int64_t rows = 0;          // rows of the LAST chunk
    int64_t chunks = 0;
  };
  // Accumulated over the chunks of the last Prefill / Decode* call (`rows` is the last chunk's).
  const Timings& LastTimings() const { return last_; }

  // VRAM of each stage's device (hipMemGetInfo is device-wide) next to this process's own DeviceBuffer bytes.
  std::vector<VramReport> Vram() const;

 private:
  Qwen4ExpModel() = default;

  struct HeadOut {
    std::vector<float> logits;
    int32_t token = -1;
    kernels::RowSummary summary;
  };
  // One chunk end to end. `head` null: no logits wanted. `iso` (RunLayerIsolated): skip the embedding and the other layers.
  struct IsolatedLayer {
    int64_t layer;
    const uint16_t* residual_in;  // this chunk's rows [rows, hc * hidden]
  };
  void RunChunk(const int32_t* tokens, int64_t rows, bool is_prefill, const Qwen4ExpHeadRequest* head, HeadOut* out,
                const IsolatedLayer* iso = nullptr);
  const uint16_t* WaitNgramRows();  // the PLE rows callback
  void DrainNgram() noexcept;
  // R4DX_Q4_PROFILE=1: every decode step's Timings is summed; the mean is printed to stderr by Reset() and the destructor.
  void ProfileDecodeStep();
  void ReportDecodeProfile() noexcept;

  Qwen4ExpModelOptions opt_;
  Qwen4ExpConfig cfg_;
  ModelConfig generic_config_;
  std::string model_id_;
  int64_t split_ = 0;

  // Declaration order is destruction order reversed: the stages (which borrow the containers' weights) go first.
  std::vector<std::unique_ptr<Qwen4ExpContainer>> containers_;
  std::vector<std::unique_ptr<Qwen4ExpStage>> stages_;
  // n-gram path
  bool ple_enabled_ = false;
  bool gdn_recurrent_ = false;
  std::optional<qwen4exp::NgramSidecarHeader> ngram_header_;
  std::unique_ptr<qwen4exp::NgramIds> ngram_ids_;
  std::unique_ptr<qwen4exp::NgramReader> ngram_reader_;
  core::PinnedBuffer<int64_t> ngram_ids_host_;    // [kChunkRows, 16]
  core::PinnedBuffer<uint16_t> ngram_rows_host_;  // [kChunkRows, ple_embed_dim]
  std::optional<qwen4exp::NgramReader::Handle> ngram_pending_;
  double ngram_wait_ms_ = 0;

  core::PinnedBuffer<uint16_t> embed_rows_host_;  // [kChunkRows, hidden]
  core::PinnedBuffer<uint16_t> handoff_host_;     // [kChunkRows, hc * hidden]

  int64_t pos_ = 0;
  bool at_prefill_end_ = false;
  bool prompt_checkpoint_ = false;
  int64_t ckpt_pos_ = -1;
  qwen4exp::NgramIds::State ckpt_ngram_;
  int64_t sampled_fallback_rows_ = 0;
  bool capture_active_ = false;
  Timings last_;
  bool profile_ = false;  // R4DX_Q4_PROFILE=1 at Load
  int64_t profile_steps_ = 0;
  Timings profile_sum_;
};

}  // namespace r4dx::model
