// r4dx::model::Qwen35MoeModel -- the qwen3_5_moe (Ornith-1.5-122B) forward pass over one or two devices
// (docs/qwen35moe-plan.md M3), organised exactly like Qwen4ExpModel (qwen4exp_model.h).
//
// Two Qwen35MoeStage objects (qwen35moe_stage.h): stage 0 holds layers [0, k) -- the host embedding gather is driven from
// here -- and stage 1 layers [k, L), the final norm and the lm_head. k comes from the loader's ChooseSplit (bytes balanced
// per card) unless asked for (R4DX_Q35_SPLIT). The two are driven in sequence on the calling thread: stage 0 runs a chunk,
// its residual [rows, hidden] bf16 goes through pinned host memory (D2H on stage 0's stream, synchronize, H2D on stage 1's),
// stage 1 runs it. When both stages sit on one device and no split was asked for there is a single stage with everything.
// The hand-off copies bytes, so a two-stage run equals the one-stage run bit for bit.
//
// A chunk is up to 256 rows, either <= 64 or a multiple of 64 (AttentionLayer's sliced path): a prompt is prefilled in chunks
// of 256, then one chunk of floor(rest / 64) * 64 rows and one of the remaining < 64; a decode step is a chunk of one row.
// Logits come from the LAST row of the last chunk only.
//
// State: GDN recurrent + conv state per layer, the paged fp8 KV of the attention layers, and the position. Reset() clears
// them; SaveCheckpoint / RestoreCheckpoint keep Qwen3.5's contract (ModelOptions::prompt_checkpoint, one slot, valid right
// after a Prefill).
//
// Not thread-safe; callers must serialize. After an exception out of Prefill / Decode*, call Reset().
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "model_config.h"
#include "qwen35moe_config.h"
#include "qwen35moe_container.h"
#include "qwen35moe_stage.h"
#include "quant_linear.h"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/kernels/sampler.hpp"
#include "text_model.h"  // VramReport

namespace r4dx::model {

struct Qwen35MoeModelOptions {
  std::string container_path;
  // Body layout (GDN / attention / expert linears): empty = the container's own (trellis for a trellis container, bf16
  // otherwise; Qwen35MoeContainer::Load refuses a mismatch). lm_head: empty = the container's.
  std::optional<Layout> layout;
  std::optional<Layout> lm_head_layout;
  int64_t max_ctx = 262144;      // KV capacity in tokens (<= the checkpoint's max_position_embeddings)
  bool prompt_checkpoint = false;
  // HIP ordinals {stage 0, stage 1}. Empty = R4DX_Q35_DEVICES ("a,b"), else {0, 1} when two devices are visible, else {0, 0}.
  std::vector<int> devices;
  // The layer boundary k of a two-stage run. 0 = R4DX_Q35_SPLIT, else ChooseSplit when the devices differ, else no split
  // (one stage). A non-zero split with equal devices runs two stages on one device (tests).
  int64_t split = 0;
  int64_t weight_cap_bytes = kQwen4ExpDefaultWeightCapBytes;  // ChooseSplit's per-card cap
  // GDN layers prefill through the recurrent decode kernels, one token at a time, instead of the chunked scan (the slow
  // reference path). Also on when R4DX_Q35_GDN_PREFILL=recurrent.
  bool gdn_prefill_recurrent = false;
};

class Qwen35MoeModel {
 public:
  static constexpr int64_t kChunkRows = 256;

  // Loads the container (one range per stage), builds the stages, zeroes the state. Prints one line per stage to stderr.
  static std::unique_ptr<Qwen35MoeModel> Load(const Qwen35MoeModelOptions& options);

  ~Qwen35MoeModel();
  Qwen35MoeModel(const Qwen35MoeModel&) = delete;
  Qwen35MoeModel& operator=(const Qwen35MoeModel&) = delete;

  const Qwen35MoeConfig& Config() const { return cfg_; }
  const ModelConfig& GenericConfig() const { return generic_config_; }
  const std::string& ModelId() const { return model_id_; }
  int NumStages() const { return static_cast<int>(stages_.size()); }
  int64_t SplitLayer() const { return split_; }  // k; == the layer count for a single stage
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
  // Feeds `token_ids` (>= 1) in chunks and returns the fp32 logits [vocab] of the last one. Continues the sequence at
  // PositionCount().
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids);
  std::vector<float> DecodeStep(int32_t token_id);
  int32_t DecodeStepGreedy(int32_t token_id);
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng);

  // ---- diagnostics / tests --------------------------------------------------------------------------------------------
  // Installs `fn` on every stage (see Qwen35MoeCapture). With a capture installed every chunk of a Prefill also runs the
  // head so `norm_out` / `logits` cover all rows. An empty function removes it.
  void SetCapture(Qwen35MoeCaptureFn fn);

  // Layer-isolated forward (tests): Reset()s the sequence, then feeds `token_ids` in chunks through ONLY layer `layer`
  // (global index): each chunk's residual input is the caller's `residual_in` rows [T, hidden] bf16 (HOST; e.g. the HF
  // golden's `layer{L}.residual_in`), not the previous layer's output, so the layer is judged on its own error. The
  // layer's own state carries from chunk to chunk exactly as in a prefill. No logits. Call Reset() before the next normal
  // Prefill.
  void RunLayerIsolated(const std::vector<int32_t>& token_ids, int64_t layer, const uint16_t* residual_in);

  struct Timings {
    double total_ms = 0;       // the whole call
    double embed_ms = 0;       // host embedding gather + upload
    double stage_ms[2] = {0, 0};
    double handoff_ms = 0;     // host wall of the residual hand-off between the stages
    double head_ms = 0;        // final norm + lm_head + readback
    int64_t rows = 0;          // rows of the LAST chunk
    int64_t chunks = 0;
  };
  const Timings& LastTimings() const { return last_; }

  // VRAM of each stage's device (hipMemGetInfo is device-wide) next to this process's own DeviceBuffer bytes.
  std::vector<VramReport> Vram() const;

  // The chunk sizes a `rows`-row call is split into (the rule in the file comment). Public for the unit test.
  static std::vector<int64_t> PlanChunks(int64_t rows);

 private:
  Qwen35MoeModel() = default;

  struct HeadOut {
    std::vector<float> logits;
    int32_t token = -1;
    kernels::RowSummary summary;
  };
  struct IsolatedLayer {
    int64_t layer;
    const uint16_t* residual_in;  // this chunk's rows [rows, hidden]
  };
  void RunChunk(const int32_t* tokens, int64_t rows, bool is_prefill, const Qwen35MoeHeadRequest* head, HeadOut* out,
                const IsolatedLayer* iso = nullptr);

  Qwen35MoeModelOptions opt_;
  Qwen35MoeConfig cfg_;
  ModelConfig generic_config_;
  std::string model_id_;
  int64_t split_ = 0;

  // Declaration order is destruction order reversed: the stages (which borrow the containers' weights) go first.
  std::vector<std::unique_ptr<Qwen35MoeContainer>> containers_;
  std::vector<std::unique_ptr<Qwen35MoeStage>> stages_;
  bool gdn_recurrent_ = false;

  core::PinnedBuffer<uint16_t> embed_rows_host_;  // [kChunkRows, hidden]
  core::PinnedBuffer<uint16_t> handoff_host_;     // [kChunkRows, hidden]

  int64_t pos_ = 0;
  bool at_prefill_end_ = false;
  bool prompt_checkpoint_ = false;
  int64_t ckpt_pos_ = -1;
  int64_t sampled_fallback_rows_ = 0;
  bool capture_active_ = false;
  Timings last_;
};

}  // namespace r4dx::model
