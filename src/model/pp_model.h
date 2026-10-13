// r4dx::model::PpModel -- the pipeline-parallel-prefill TextModel (docs/pp-prefill.md, Phase 2; `--pp 2`).
//
// Two GPUs, two Models, one conversation. Stage B is the ordinary full decode Model on the facade's own thread and the
// headless card (the facade thread is bound to it with hipSetDevice at every entry; by default the last visible HIP ordinal,
// or --pp-devices B,A / R4DX_PP_DEVICES -- pp::ResolvePlacement, never HIP_VISIBLE_DEVICES ordering -- decode is
// byte-for-byte today's, it is the same Model on the same device on the same thread). Stage A is a half-weight Model on its
// own rank thread and the other card (the desktop one): layers [0, k) + the embedding (it is loaded with layer_limit = k + 1, the last layer
// only supplying the input_layernorm weight that layer k - 1's Mlp fuses). A Prefill / PrefillMultimodal call of at
// least `min_rows` rows runs on both at once: A's chunk c + 1 overlaps B's chunk c, the carry (residual stream, the
// fused-norm pair, DFlash feature columns, the KV rows A's attention layers wrote) crossing through a ring of pinned host
// slots (pp::StageChannel; no P2P copy, no spin kernel), and at the end of the call A's GDN live state follows
// (compactly, with the conv-line pitch conversion), overlapped with B's last chunk. Shorter calls, and everything that
// is not a prompt prefill, run on B alone.
//
// A's copy of the sequence goes stale whenever B moves alone (decode, speculative rounds, a restore, a B-only prefill);
// pp::MirrorTracker (pp_sync.h) tracks exactly how stale and the next pipelined call first copies B's GDN state and the
// missing KV rows to A ("sync-back"). R4DX_PP_VERIFY=1 (PpOptions::verify) digests the live state of both stages after
// every hand-off and fails the call on a difference.
//
// Errors: a failure on either stage poisons the channel (the other stage's waits return), the facade joins stage A, and
// the model is kNeedsRecovery: every device-work call throws until Reset() (which resets both Models and the channel)
// heals it, as TpModel's state machine does (docs/tp.md 2.4). A stage A that makes no progress for 60 s is kFatal.
//
// Orchestration: the per-call work is the thread-agnostic stage runners of pp_stage_runner.h (RunStageA on stage A's worker, RunStageB on
// the facade, StageBExportSyncBack ...); PpModel owns the threads, the channel, the mirror tracker and the error state. Every pinned buffer
// (pp::StageBuffers) is allocated by the thread that owns its producing device -- stage A's worker for the slots and the GDN hand-off, the
// facade (stage B's thread) for the sync-back buffers -- and none lazily (docs/pp-tp2-hybrid.md 7).
//
// Thread safety: like Model, one facade thread issues every call.
#pragma once

#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "model.h"
#include "pp_channel.h"
#include "pp_stage_buffers.h"
#include "pp_stage_runner.h"
#include "pp_sync.h"
#include "text_model.h"
#include "tp/tp_rank_worker.h"

namespace r4dx::model {

// ModelOptions::pp resolved: 0 = off, > 0 = on, -1 = R4DX_PP (unset / 0 / off = off, 1 / on / 2 = on). Throws
// std::invalid_argument for an unrecognised R4DX_PP value.
bool ResolvePpSwitch(int option);

class PpModel final : public TextModel {
 public:
  enum class State { kReady, kNeedsRecovery, kFatal };

  // `opts` is the ordinary single-model option set (opts.tp must be default, opts.pp is not consulted here). Refuses, by
  // name, fewer than two visible devices, a malformed or out-of-range --pp-devices / R4DX_PP_DEVICES, two ordinals on one physical GPU, GPUs of
  // different architectures, a rotated (quant2) container, a split outside [1, layers - 1].
  static std::unique_ptr<PpModel> Load(const ModelOptions& opts, const PpOptions& pp);
  ~PpModel() override;
  PpModel(const PpModel&) = delete;
  PpModel& operator=(const PpModel&) = delete;

  // ---- identity / capabilities: cached at load ------------------------------------------------------
  const ModelConfig& Config() const override { return b_->Config(); }
  const std::string& ModelId() const override { return b_->GetContainer().ModelId(); }
  int64_t ImageTokenId() const override { return b_->GetContainer().ImageTokenId(); }
  int64_t VisionMergeSize() const override;
  bool HasVision() const override { return b_->HasVision(); }
  bool MtpEnabled() const override { return b_->MtpEnabled(); }
  bool MtpUsingReducedVocabDraft() const override { return b_->MtpUsingReducedVocabDraft(); }
  bool DflashEnabled() const override { return b_->DflashEnabled(); }
  int64_t SampledFallbackRows() const override { return b_->SampledFallbackRows(); }
  int64_t PositionCount() const override { return b_->PositionCount(); }
  int64_t NumLoadedLayers() const override { return b_->GetContainer().NumLoadedLayers(); }
  int TpWorld() const override { return 1; }
  // One entry per stage: rank 0 = the decode card (stage B), rank 1 = the desktop card (stage A); .device is the HIP ordinal. Takes a command on
  // stage A's thread (hipMemGetInfo there).
  std::vector<VramReport> Vram() const override;

  // ---- sequence state -------------------------------------------------------------------------------
  // kReady: Model::Reset() on both stages. kNeedsRecovery: the same, which also heals the channel and returns to kReady.
  void Reset() override;
  void SetDflashInjectionEnabled(bool enabled) override;
  void SaveCheckpoint() override;
  void RestoreCheckpoint() override;

  void EncodeImages(const float* pixel_values, int64_t total_patches, const std::vector<vision::GridThw>& grids,
                    ImageRows* out, vision::VisionEncodeStats* stats = nullptr) override;

  // ---- forward ----------------------------------------------------------------------------------------
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids) override;
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                       const std::vector<ImageSpan>& images) override;
  std::vector<float> DecodeStep(int32_t token_id) override;
  int32_t DecodeStepGreedy(int32_t token_id) override;
  int32_t DecodeStepGreedyOverlap(int32_t token_id, const std::function<void()>& while_busy) override;
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng) override;
  std::vector<int32_t> DecodeStepMtpGreedy(int32_t token_id, int64_t k) override;
  std::vector<int32_t> DecodeStepMtpSampled(int32_t token_id, int64_t k, const kernels::SampleParams& params,
                                            std::mt19937_64& rng) override;
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                              int64_t* walk_len_out = nullptr) override;
  std::vector<int32_t> DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                               const kernels::SampleParams& params, std::mt19937_64& rng,
                                               int64_t* walk_len_out = nullptr) override;

  // Batched decode (docs/batch-decode.md): stage B's, the decode Model's. A pipelined prefill leaves the single-sequence state in B
  // (that is the hand-off), BatchImport copies it into a batch slot there, and DecodeBatch never touches stage A -- the mirror
  // tracker does not see a batch step, because the single-sequence state it mirrors does not move.
  int BatchSlots() const override { return b_->BatchSlots(); }
  int64_t BatchSlotCtx() const override { return b_->BatchSlotCtx(); }
  void BatchImport(int slot) override;
  void BatchRelease(int slot) override;
  std::vector<int32_t> DecodeBatch(const std::vector<BatchDecodeRow>& rows) override;

  StepProfile DecodeStepProfiled(int32_t token_id) override;
  StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids) override;

  // ---- diagnostics (tests, tools, --stats) ---------------------------------------------------------------
  State GetState() const { return state_; }
  const PpOptions& Options() const { return pp_; }
  int64_t Split() const { return split_; }
  int StageBDevice() const { return dev_b_; }  // process-visible HIP ordinals the stages run on
  int StageADevice() const { return dev_a_; }
  const pp::MirrorTracker& Mirror() const { return tracker_; }
  struct Stats {
    int64_t pipelined_calls = 0;   // Prefill / PrefillMultimodal calls that ran on both stages
    int64_t b_only_calls = 0;      // ... that ran on the decode Model alone (shorter than min_rows, or no room to sync)
    int64_t sync_gdn = 0;          // sync-backs that copied the GDN state
    int64_t sync_kv_rows = 0;      // KV rows copied back to stage A, summed
    int64_t early_gdn_imports = 0; // calls whose GDN import overlapped the last chunk (the rest imported after it)
    int64_t chunks = 0;            // chunks stage B took from the channel
    double last_total_ms = 0, last_sync_ms = 0, last_tail_ms = 0;  // the latest pipelined call
    double last_gdn_export_ms = 0;  // ... stage A's GDN export (D2H of the live state, after its last chunk)
    double last_gdn_wait_ms = 0;    // ... stage B's wait for the GDN state after its last chunk (0 when it overlapped)
    double stage_a_wait_ms = 0;    // stage A blocked on a full ring since load (back-pressure: B is the slower stage)
    double stage_b_wait_ms = 0;    // stage B blocked on an empty ring (starvation: the fill, or A is the slower stage)
  };
  Stats GetStats() const;
  // The `--stats` line (without the "[stats] " prefix).
  std::string StatsLine() const;
  // Tests and tools. SetSplit moves the split layer k (up to PpOptions::reserve_split, which stage A was loaded for) and
  // resets both stages; SetMinRows changes the pipelining threshold; AttachDflashFeatureCapture attaches a feature capture on
  // the decode Model (like Model::AttachDflashFeatureCapture) and the below-the-split prefix of it on stage A.
  // Negative controls (tests/model/test_pp_real_identity.cpp requires each to change an observable): do not copy the sync-back
  // to stage A (a stale mirror), do not import stage A's GDN state into stage B.
  enum class TestFault { kNone, kSkipSyncBack, kSkipGdnImport };
  void SetTestFault(TestFault f) {
    fault_ = f;
    run_.skip_gdn_import = f == TestFault::kSkipGdnImport;
  }
  void SetSplit(int64_t k);
  void SetMinRows(int64_t min_rows);
  void AttachDflashFeatureCapture(std::vector<int64_t> target_layers);
  // The decode Model, for tests and tools (the facade thread only).
  Model& DecodeModel() { return *b_; }
  // Runs fn(stage-A Model&) on stage A's thread as one command and waits (tests: digests, state pokes).
  void RunOnStageA(const std::function<void(Model&)>& fn);

 private:
  PpModel() = default;

  // hipSetDevice(stage B's ordinal) on the calling thread. The decode Model's device is not necessarily the process default
  // (ordinal 0) and the thread that calls into the PpModel is not necessarily the one that loaded it (the server loads on
  // main, runs on the engine worker), so every entry point binds first (RequireReady, Reset, ...).
  void BindStageB() const;
  void RequireReady() const;
  void NoteBOnly() { tracker_.BOnly(b_->PositionCount()); }
  std::vector<float> PipelinedPrefill(const std::vector<int32_t>& ids, const std::vector<ImageSpan>* images);
  // Waits until stage A is idle (the progress watchdog: 60 s without a chunk is kFatal). Returns stage A's error.
  std::exception_ptr JoinStageA(std::chrono::milliseconds stall);
  void RunA(const std::function<void(Model&)>& fn, std::chrono::milliseconds stall);
  [[noreturn]] void Fail(std::exception_ptr cause);
  Model::PpStageSetup StageSetup(Model::PpRole role);
  std::vector<int64_t> TargetsBelow(int64_t split) const;
  void UpdateStageACapture();

  ModelOptions opts_;
  PpOptions pp_;
  State state_ = State::kReady;
  int dev_b_ = 0;                       // process-visible HIP ordinals of the stages (pp::ResolvePlacement)
  int dev_a_ = 1;
  int64_t split_ = 0;
  int64_t reserve_split_ = 0;           // stage A is loaded for splits up to this
  int64_t hidden_ = 0;
  std::vector<int64_t> targets_;        // the DFlash target layers B captures (empty without a drafter / capture)
  int64_t min_rows_ = 1024;
  bool verify_ = false;
  bool dflash_injection_ = true;
  int64_t call_id_ = 0;
  TestFault fault_ = TestFault::kNone;
  pp::StageRunOptions run_;      // what the stage runners are told (split_ mirrored into run_.split, the GDN-import fault)
  pp::StageBCallState b_state_;  // stage B's per-call state (the call in flight, whether its GDN import started early)
  pp::MirrorTracker tracker_;
  Stats stats_;

  // Declared in reverse order of destruction: the worker (joins stage A's thread) goes first; stage A's Model was already
  // destroyed on that thread by ~PpModel (which also released the stage-A-produced buffers there); then stage B; then the channel
  // and the pinned memory it points into (the stage-B-produced buffers were released by ~PpModel on the facade).
  std::unique_ptr<pp::StageBuffers> bufs_;  // slots + GDN hand-off (A-produced), GDN / KV sync-back (B-produced)
  std::unique_ptr<pp::StageChannel> channel_;
  std::optional<Model> b_;
  std::optional<Model> a_;  // built, used and destroyed ONLY on stage A's thread
  tp::CompletionGroup done_;
  tp::WorkerTiming timing_;
  std::unique_ptr<tp::RankWorker> worker_;
};

}  // namespace r4dx::model
