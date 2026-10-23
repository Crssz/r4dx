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
// Speculative decoding (docs/qwen35moe-plan.md section 6, docs/dflash2.md section 14): with Qwen35MoeModelOptions::spec_window > 1 (or a
// DFlash2 drafter) VerifyWindow runs [anchor, drafts...] as one chunk that commits nothing and CommitVerifiedWindow(n) commits its first n rows;
// the GDN layers keep the 27B's write-once window state so rejected rows cost nothing to roll back. A verify row is bit-identical to the
// plain decode step of its position (tool_q35moe_dflash --audit-only), so a DFlash2 run equals plain greedy decode token for token.
// With a drafter the model injects the target features of every committed row into it (Prefill, DecodeStep*, accepted verify rows).
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

#include "dflash_draft.h"
#include "model_config.h"
#include "qwen35moe_config.h"
#include "qwen35moe_container.h"
#include "qwen35moe_stage.h"
#include "quant_linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_comm.hpp"
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
  // The on-device M = 1 MoE of a decode step (Qwen35GroupedMoeDecode): -1 = R4DX_Q35_GROUPED (default on, 0 = the host-routed MoE), 0 / 1 force.
  int grouped_decode = -1;
  // Speculative verify window: the most rows VerifyWindow takes (1 = none). The GDN state is then the 27B's write-once window
  // state. A drafter (below) raises it to at least dflash_draft_k + 1.
  int64_t spec_window = 1;
  // DFlash2 drafter (docs/dflash2.md): a dflash2_draft container ("" = none) and the longest draft it is asked for (1 ..
  // block_size - 1). The drafter lives on the LAST stage's device next to the lm_head it shares; the residual stream entering its
  // `target_layers` is captured by the stages into pinned host memory and injected into the drafter after each committed chunk.
  std::string dflash_container;
  int64_t dflash_draft_k = 0;
  // Test hook: read these layer inputs instead of the drafter's own target_layers (same count), for a container with fewer layers
  // than the drafter was trained against. Empty = the drafter's.
  std::vector<int64_t> dflash_target_layers;
  // Test hook: a target whose vocabulary is SMALLER than the drafter's (the tiny test container, 4096 vs 248320): the drafter's mask
  // token becomes vocab - 1 and only the first `vocab` rows of its codebooks are read.
  bool dflash_small_vocab = false;

  // ---- tensor parallelism (--tp 2; qwen35moe_tp_model.h drives one of these per rank) --------------------------------------
  // tp_world > 1: this model is RANK `tp_rank` of the group: ONE stage holding all layers on `tp_device` (no hand-off), loaded sharded
  // (Qwen35MoeContainer::LoadShard), all-reducing through `tp_comm` (an endpoint of the same world and rank), its head a vocab slice merged
  // with the other rank. Config() is then the RANK-local config (heads, experts halved; vocab global). `shared_embed_host`: the process's one pinned
  // copy of text.embed_tokens (never duplicated per rank). A drafter / speculative window is not supported under TP yet.
  int tp_world = 1;
  int tp_rank = 0;
  int tp_device = -1;
  core::TpComm* tp_comm = nullptr;
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> shared_embed_host;
  int tp_submit_layers = 0;        // Qwen35MoeStageOptions::tp_submit_layers / tp_max_inflight_units (bounded prefill submission on the desktop card)
  int tp_max_inflight_units = 0;
};

class Qwen35MoeModel {
 public:
  static constexpr int64_t kChunkRows = 256;

  // Loads the container (one range per stage), builds the stages, zeroes the state. Prints one line per stage to stderr.
  static std::unique_ptr<Qwen35MoeModel> Load(const Qwen35MoeModelOptions& options);

  ~Qwen35MoeModel();
  Qwen35MoeModel(const Qwen35MoeModel&) = delete;
  Qwen35MoeModel& operator=(const Qwen35MoeModel&) = delete;

  const Qwen35MoeConfig& Config() const { return cfg_; }  // the rank-local config under TP
  const Qwen35MoeConfig& GlobalConfig() const { return global_cfg_; }
  bool IsTp() const { return opt_.tp_world > 1; }
  const Qwen35MoeContainer& StageContainer(int stage) const { return *containers_.at(static_cast<size_t>(stage)); }
  // The pinned host embedding this model reads (the one the facade shares with the other rank).
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> EmbedHostShared() const { return containers_.front()->EmbedTokensHostShared(); }
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

  // ---- speculative decoding (Qwen35MoeModelOptions::spec_window > 1) -------------------------------------------------------
  int64_t SpecWindow() const { return opt_.spec_window; }
  // Runs `candidates` (1 .. SpecWindow(): the last committed token, then drafts) as ONE chunk at PositionCount() and returns, for
  // every row, the argmax of the target's next token. Commits nothing: the KV rows and the GDN window rows it wrote stay
  // uncommitted until CommitVerifiedWindow(n) says how many leading rows are real (n >= 1: row 0, the anchor, always is), and a
  // rejected row's KV slot is simply overwritten by the next write at that position.
  std::vector<int32_t> VerifyWindow(const std::vector<int32_t>& candidates);
  void CommitVerifiedWindow(int64_t n);
  // Test hook: the fp32 logits [rows, vocab] behind the last VerifyWindow's argmax (D2H).
  std::vector<float> LastVerifyLogits() const;

  // DFlash2. DecodeStepDflashGreedy: the round (draft k tokens after `token_id`, verify, accept the longest prefix the target's own
  // argmax confirms, inject the accepted rows' features, commit) returns the accepted drafts and the bonus token (1..k+1
  // tokens); the output equals what that many greedy DecodeStepGreedy calls would return, token for token, up to the batched
  // verify GEMMs' rounding (docs/dflash2.md section 7a item 4).
  bool DflashEnabled() const { return dflash_.has_value(); }
  int64_t DflashDraftK() const { return dflash_k_; }
  int64_t DflashInjectedCount() const;
  void SetDflashInjectionEnabled(bool enabled);
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                              int64_t* walk_len_out = nullptr);
  // Test hook: replaces the drafter's tokens of every round (anchor, k) -> drafts (at most k) after the real draft round has run, so a
  // test drives every partial-accept length through the round's own accept / inject / commit code. Empty function = off.
  void SetDraftOverride(std::function<std::vector<int32_t>(int32_t anchor, int64_t k)> fn) { draft_override_ = std::move(fn); }
  // Wall-clock breakdown of the DecodeStepDflashGreedy rounds since load (printed to stderr by the destructor when R4DX_Q35_STATS=1).
  struct RoundStats {
    int64_t rounds = 0, tokens = 0;
    double total_ms = 0, draft_ms = 0, verify_ms = 0, inject_ms = 0, stage_ms[2] = {0, 0}, handoff_ms = 0, head_ms = 0;
  };
  const RoundStats& GetRoundStats() const { return round_stats_; }
  std::string RoundStatsLine() const;
  // The drafts the last DecodeStepDflashGreedy verified (tests / stats).
  const std::vector<int32_t>& LastDrafts() const { return last_drafts_; }
  // Pinned [rows, FeatureCols] features captured by the last chunk / verify window (valid until the next one).
  const uint16_t* FeatureRowsHost() const { return feat_host_.data(); }

  // ---- diagnostics / tests --------------------------------------------------------------------------------------------
  // Installs `fn` on every stage (see Qwen35MoeCapture). With a capture installed every chunk of a Prefill also runs the
  // head so `norm_out` / `logits` cover all rows. An empty function removes it.
  void SetCapture(Qwen35MoeCaptureFn fn);
  // Test hook: the grouped decode MoE on / off at run time; GroupedDecodeLayers() = how many layers (all stages) take it right now.
  void SetGroupedDecode(bool on, bool with_capture = false);
  int64_t GroupedDecodeLayers() const;

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
    double draft_ms = 0;       // DecodeStepDflashGreedy: the draft round (host wall)
    double inject_ms = 0;      // ... and the injection of the committed rows' features
    int64_t rows = 0;          // rows of the LAST chunk
    int64_t chunks = 0;
  };
  const Timings& LastTimings() const { return last_; }

  // ---- tensor parallelism -------------------------------------------------------------------------------------------------
  // The first use of every path on this rank, run by the facade as ONE collective command at load (a lockstep sequence on both ranks): a full 256-row
  // prefill chunk (the sliced all-reduces, the per-expert GEMM shapes), a 64-row chunk, then a greedy, a sampled (row-summary merge) and a full-logits
  // decode step, so no first-use launch or merge lands inside a request. Leaves the model Reset().
  void TpWarmup();
  static int64_t WarmupPositions() { return kChunkRows + 64 + 3; }
  tp::SubmitBounder::Stats TpSubmitStats() const { return stages_.front()->SubmitStats(); }

  // GPU time of the last chunk, hipEvent spans on each stage's stream (the host clock of Timings only sees the enqueue of a decode step):
  // every stage's layers and the head. EnableStepProfile(true) first; read after the call returned (the head synchronized).
  void EnableStepProfile(bool on);
  struct StepGpu {
    double layers_ms[2] = {0, 0};  // stage 0, stage 1 (a single stage fills [0])
    double head_ms = 0;
    double Total() const { return layers_ms[0] + layers_ms[1] + head_ms; }
  };
  StepGpu LastStepGpu() const;

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
  // verify_out != null: a verify window (spec_window rows max, !is_prefill, pos_ is not advanced), the per-row argmax goes there.
  void RunChunk(const int32_t* tokens, int64_t rows, bool is_prefill, const Qwen35MoeHeadRequest* head, HeadOut* out,
                const IsolatedLayer* iso = nullptr, std::vector<int32_t>* verify_out = nullptr);
  // Feeds the first `rows` captured feature rows (positions start_pos ...) to the drafter's ring.
  // Rows at positions below `min_pos` are skipped (a prompt's rows older than the drafter's window).
  void InjectCaptured(int64_t rows, int64_t start_pos, int64_t min_pos = 0);

  Qwen35MoeModelOptions opt_;
  Qwen35MoeConfig cfg_;
  Qwen35MoeConfig global_cfg_;
  ModelConfig generic_config_;
  std::string model_id_;
  int64_t split_ = 0;

  // Declaration order is destruction order reversed: the stages (which borrow the containers' weights) go first.
  std::vector<std::unique_ptr<Qwen35MoeContainer>> containers_;
  std::vector<std::unique_ptr<Qwen35MoeStage>> stages_;
  bool gdn_recurrent_ = false;

  core::PinnedBuffer<uint16_t> embed_rows_host_;  // [kChunkRows, hidden]
  core::PinnedBuffer<uint16_t> handoff_host_;     // [kChunkRows, hidden]

  // ---- DFlash2 (all on the last stage's device) ----
  std::optional<DflashDraft> dflash_;
  std::unique_ptr<core::Stream> dflash_stream_;
  core::Arena dflash_arena_;
  core::DeviceBuffer<uint16_t> dflash_feat_dev_;  // [kChunkRows, FeatureCols] injection staging
  core::PinnedBuffer<uint16_t> feat_host_;        // [kChunkRows, FeatureCols] captured by the stages
  int64_t feat_cols_ = 0;
  int64_t dflash_k_ = 0;
  bool dflash_injection_ = true;
  std::vector<int32_t> last_drafts_;
  std::function<std::vector<int32_t>(int32_t, int64_t)> draft_override_;
  int64_t verified_rows_ = 0;
  RoundStats round_stats_;

  int64_t pos_ = 0;
  bool at_prefill_end_ = false;
  bool prompt_checkpoint_ = false;
  int64_t ckpt_pos_ = -1;
  int64_t sampled_fallback_rows_ = 0;
  bool capture_active_ = false;
  Timings last_;
};

}  // namespace r4dx::model
