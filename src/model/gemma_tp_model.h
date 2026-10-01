// r4dx::model::GemmaTpModel -- the tensor-parallel (`--tp 2`) TextModel of a gemma4_unified container
// (docs/gemma4-plan.md 3.2, 3.7, task M1b-1). The facade of docs/tp.md 2.1-2.9, specialised to GemmaModel: one rank
// thread per rank, each owning that rank's GemmaModel, TpComm endpoint and every device buffer they allocate; the
// facade never computes and never calls HIP after load -- every call is broadcast to the rank threads as one command,
// the results are compared, and every failure goes through TpModel's error state machine:
//
//   kReady          -- normal operation.
//   kNeedsRecovery  -- a command failed on any rank (or the ranks' results diverged). Device-work calls throw
//                      core::TpStateError("... call Reset() first"); Reset() runs recovery (docs/tp.md 2.5).
//   kFatal          -- recovery failed, or a rank made no progress for 60 s. Everything throws.
//
// The failing call rethrows the ROOT CAUSE with its original type (the lowest rank's exception that is not a
// core::TpAbortedError), and every rank's message is logged to stderr as "[r4dx-tp] rank <r> (dev <d>): <what>".
//
// Modes (TpOptions::Mode, docs/tp.md 9.1): kReal -- one rank per GPU, HostMailboxComm through the pinned host mailbox;
// kEmulate -- two ranks on ONE device, EmulatedComm; kNoop -- ONE rank's shard with a no-op all-reduce (timing only,
// tokens meaningless). Prefill submissions are bounded in every mode (TpOptions::submit_layers / max_inflight_units).
//
// What is sharded (GemmaConfig::Shard, tp::RuleFor's Gemma table, GemmaContainer::Load):
//   q heads 16 -> 8 per rank; sliding kv heads 8 -> 4 per rank (GQA stays 2); the full layers' ONE kv head (k_eq_v) is
//   REPLICATED -- every rank loads k_proj and keeps the whole 1-head KV, its per-rank GQA is 8;
//   o_proj and down_proj are row-parallel (K split) and their bf16 outputs are all-reduced BEFORE the fp32 post-norm and
//   residual add, which stay replicated on every rank; gate_up is column-parallel (gate | up each 15360 -> 7680);
//   the tied head is a vocab slice (rank r holds rows [r * 131072, (r + 1) * 131072)) merged on the host (docs/tp.md 7);
//   the embedding table is replicated on every rank's device (the scaled gather runs locally); rotation tensors replicate
//   except the Hadamard sign vectors, which split with their linear's K.
//
// Not supported here (throws, naming the feature): MTP, DFlash (the drafter track), vision (M2), profiling.
//
// This class deliberately duplicates the TpModel state machine instead of templating it (docs/gemma4-plan.md 3.8's
// listed risk): TpModel (tp_model.h) is wired to Model and is untouched, so Qwen behaviour cannot change.
//
// Thread safety: like TpModel, one facade thread issues every call; the cached accessors are plain reads of values fixed
// at load.
#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <exception>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "gemma_model.h"
#include "model_config.h"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_alloc_guard.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "text_model.h"
#include "tp/tp_group.h"
#include "tp/tp_rank_worker.h"

namespace r4dx::model {

struct ModelOptions;  // model.h

class GemmaTpModel final : public TextModel, public TpDiagnostics {
 public:
  enum class State { kReady, kNeedsRecovery, kFatal };

  // `opts` is the ordinary single-model option set (opts.tp must be default: the per-rank options are filled in here);
  // `tp.world` must be 2.
  static std::unique_ptr<GemmaTpModel> Load(const ModelOptions& opts, const TpOptions& tp);
  // Every rank's model and endpoint are destroyed on that rank's thread; a rank that does not finish within 30 s ends the
  // process (quick_exit(3)) rather than free memory a kernel may still touch.
  ~GemmaTpModel() override;
  GemmaTpModel(const GemmaTpModel&) = delete;
  GemmaTpModel& operator=(const GemmaTpModel&) = delete;

  // ---- identity / capabilities: cached at load, legal in every state ------------------------------
  const ModelConfig& Config() const override { return global_config_; }
  const std::string& ModelId() const override { return model_id_; }
  int64_t ImageTokenId() const override { return image_token_id_; }
  int64_t VisionMergeSize() const override { return 2; }  // no vision until M2
  bool HasVision() const override { return false; }
  bool MtpEnabled() const override { return false; }
  bool MtpUsingReducedVocabDraft() const override { return false; }
  bool DflashEnabled() const override { return false; }
  int64_t SampledFallbackRows() const override { return cached_fallback_rows_; }
  int64_t PositionCount() const override { return cached_position_; }
  int64_t NumLoadedLayers() const override { return num_loaded_layers_; }
  int TpWorld() const override { return 2; }
  std::vector<VramReport> Vram() const override;

  // ---- sequence state -------------------------------------------------------------------------
  void Reset() override;
  void SetDflashInjectionEnabled(bool) override {}
  void SaveCheckpoint() override;     // each rank D2D-copies its own sliding rings (needs ModelOptions::prompt_checkpoint)
  void RestoreCheckpoint() override;

  void EncodeImages(const float* pixel_values, int64_t total_patches, const std::vector<vision::GridThw>& grids,
                    ImageRows* out, vision::VisionEncodeStats* stats = nullptr) override;

  // ---- forward: one collective command each; results compared across ranks ------------------------
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids) override;
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                       const std::vector<ImageSpan>& images) override;
  std::vector<float> DecodeStep(int32_t token_id) override;
  int32_t DecodeStepGreedy(int32_t token_id) override;
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng) override;
  std::vector<int32_t> DecodeStepMtpGreedy(int32_t token_id, int64_t k) override;
  std::vector<int32_t> DecodeStepMtpSampled(int32_t token_id, int64_t k, const kernels::SampleParams& params,
                                            std::mt19937_64& rng) override;
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                              int64_t* walk_len_out = nullptr) override;
  std::vector<int32_t> DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                               const kernels::SampleParams& params, std::mt19937_64& rng,
                                               int64_t* walk_len_out = nullptr) override;

  // Profiling is not supported under tensor parallelism: core::TpUnsupportedError.
  StepProfile DecodeStepProfiled(int32_t token_id) override;
  StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids) override;

  // TpDiagnostics (text_model.h): what r4dx-cli --stats and the server's group-state handling reach through a
  // dynamic_cast (State's enumerators have the same order as TpDiagnostics::Health's).
  Health GroupHealth() const override { return static_cast<Health>(static_cast<int>(state_)); }
  const TpOptions& GroupOptions() const override { return tp_; }
  std::string GroupStatsLine() override { return StatsLine(); }

  // ---- tensor-parallel diagnostics (tests, tools, --stats) --------------------------------------
  State GetState() const { return state_; }
  const TpOptions& Options() const { return tp_; }
  int NumRankThreads() const { return static_cast<int>(ranks_.size()); }  // 2 (emulate, real) or 1 (noop)
  std::vector<std::array<uint64_t, 2>> CallCounts();
  std::vector<core::TpCommStats> CommStats();
  std::vector<tp::SubmitBounder::Stats> SubmitStats();
  // The `--stats` line (without the "[stats] " prefix), same fields as TpModel::StatsLine.
  std::string StatsLine();
  // Test-only: arms fault injection on `rank`'s endpoint now (counted from this call). Legal in kReady.
  void ArmFaultInjection(int rank, int64_t at_allreduce, int kind);
  // Test-only: runs fn(model, rank) on every rank as ONE collective command (same state check, allocation guard,
  // abort-on-error and root-cause rethrow as a forward call, no result comparison). The caller must issue the same
  // collective sequence on every rank.
  void RunCollectiveForTest(const std::function<void(GemmaModel&, int)>& fn);

 private:
  struct RankSlot {
    int index = 0;   // position in ranks_ (== rank, except noop's single slot)
    int rank = 0;    // TP rank
    int device = 0;  // HIP ordinal (process-visible)
    std::unique_ptr<tp::RankWorker> worker;
    std::unique_ptr<tp::TpEndpoint> endpoint;  // emulate / real
    std::unique_ptr<core::TpComm> noop;        // noop
    std::optional<core::Stream> aux_stream;    // warm-up latency sample; lives as long as the endpoint
    std::optional<GemmaModel> model;
    // The facade's own copy of Comm(), written and read only on the facade thread, so the facade can poison the group while
    // this rank is still inside a command (docs/tp.md 2.6 step 1; Appendix B N53).
    core::TpComm* facade_comm = nullptr;
    core::TpComm* Comm() const { return endpoint ? static_cast<core::TpComm*>(endpoint.get()) : noop.get(); }
  };
  enum class CmdKind {
    kCollective,  // may use TpComm: CheckHealthy after the body, Abort(kAbortHost) on a failure
    kPlain,       // no TpComm (load helpers, Reset, Vram, counters)
  };

  GemmaTpModel() = default;

  std::vector<int> AllSlots() const;
  void Run(const std::vector<int>& slots, const std::function<void(RankSlot&)>& body, CmdKind kind,
           std::chrono::milliseconds stall);
  void Run(const std::vector<int>& slots, const std::function<void(RankSlot&)>& body, CmdKind kind) {
    Run(slots, body, kind, stall_limit_);
  }
  void RunGuarded(const std::vector<int>& slots, const std::function<void(RankSlot&)>& body, CmdKind kind,
                  std::chrono::milliseconds stall);
  void RunGuarded(const std::vector<int>& slots, const std::function<void(RankSlot&)>& body, CmdKind kind) {
    RunGuarded(slots, body, kind, stall_limit_);
  }
  void RequireReady() const;
  void AbortAllFromFacade(uint32_t code, const char* why) noexcept;
  void Recover();
  [[noreturn]] void Diverged(const std::string& what);
  template <class T>
  void RequireAllEqual(const std::vector<T>& v, const char* what);
  void RequireRngsEqual(const std::vector<std::mt19937_64>& rngs, const char* what);
  std::vector<VramReport> VramImpl();

  // What one command's rank closures share (docs/tp.md Appendix B N53): the caller's fn, its per-rank result slots and
  // rank 0's re-cached counters live on the HEAP, co-owned by every posted closure, so a rank still running after a
  // watchdog stall touches only this block. `fn` must itself own what it reads.
  template <class Fn>
  struct CmdState {
    using R = std::invoke_result_t<std::decay_t<Fn>&, GemmaModel&, int>;
    using Slot = std::optional<std::conditional_t<std::is_void_v<R>, char, R>>;
    CmdState(Fn&& f, size_t n) : fn(std::forward<Fn>(f)), out(n) {}
    std::decay_t<Fn> fn;
    std::vector<Slot> out;
    int64_t pos0 = 0, fallback0 = 0;
    auto Take() {
      std::vector<std::conditional_t<std::is_void_v<R>, char, R>> v;
      v.reserve(out.size());
      for (auto& o : out) v.push_back(std::move(*o));
      return v;
    }
  };

  // The forward-command wrapper: state check, then on every rank one collective command running fn(model, slot index)
  // inside a core::TpCollectiveScope; PositionCount / SampledFallbackRows are re-cached from rank 0 on success.
  template <class Fn>
  auto RunCollective(Fn&& fn)
      -> std::conditional_t<std::is_void_v<std::invoke_result_t<std::decay_t<Fn>&, GemmaModel&, int>>, void,
                            std::vector<std::invoke_result_t<std::decay_t<Fn>&, GemmaModel&, int>>> {
    using St = CmdState<Fn>;
    RequireReady();
    auto st = std::make_shared<St>(std::forward<Fn>(fn), ranks_.size());
    RunGuarded(AllSlots(),
               [st](RankSlot& s) {
                 core::TpCollectiveScope scope;
                 if constexpr (std::is_void_v<typename St::R>) {
                   st->fn(*s.model, s.index);
                 } else {
                   st->out[static_cast<size_t>(s.index)].emplace(st->fn(*s.model, s.index));
                 }
                 if (s.index == 0) {
                   st->pos0 = s.model->PositionCount();
                   st->fallback0 = s.model->SampledFallbackRows();
                 }
               },
               CmdKind::kCollective);
    cached_position_ = st->pos0;
    cached_fallback_rows_ = st->fallback0;
    if constexpr (std::is_void_v<typename St::R>) {
      return;
    } else {
      return st->Take();
    }
  }

  // fn(model, rank) on every rank thread as ONE plain command (no TpComm use, no state change); same heap ownership.
  template <class Fn>
  auto RunAll(Fn&& fn)
      -> std::conditional_t<std::is_void_v<std::invoke_result_t<std::decay_t<Fn>&, GemmaModel&, int>>, void,
                            std::vector<std::invoke_result_t<std::decay_t<Fn>&, GemmaModel&, int>>> {
    using St = CmdState<Fn>;
    auto st = std::make_shared<St>(std::forward<Fn>(fn), ranks_.size());
    Run(AllSlots(),
        [st](RankSlot& s) {
          if constexpr (std::is_void_v<typename St::R>) {
            st->fn(*s.model, s.rank);
          } else {
            st->out[static_cast<size_t>(s.index)].emplace(st->fn(*s.model, s.rank));
          }
        },
        CmdKind::kPlain);
    if constexpr (std::is_void_v<typename St::R>) {
      return;
    } else {
      return st->Take();
    }
  }

  GemmaModelOptions gopts_;  // the options every rank was loaded with (tp_* filled per rank)
  ModelConfig global_config_;
  TpOptions tp_;
  State state_ = State::kReady;
  tp::CompletionGroup done_;
  tp::WorkerTiming timing_;
  std::chrono::milliseconds stall_limit_{60000};
  std::unique_ptr<tp::TpGroup> group_;           // emulate / real
  std::vector<std::unique_ptr<RankSlot>> ranks_;  // stable addresses (closures hold RankSlot*)

  std::string model_id_;
  int64_t image_token_id_ = -1;
  int64_t num_loaded_layers_ = 0;
  int64_t cached_position_ = 0;
  int64_t cached_fallback_rows_ = 0;
  mutable std::vector<VramReport> cached_vram_;
};

}  // namespace r4dx::model
