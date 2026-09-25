// r4dx::model::TpModel -- the tensor-parallel TextModel (docs/tp.md 2.1-2.9): the facade that owns
// one rank thread per rank, each of which owns that rank's Model, TpComm endpoint and every device
// buffer they allocate. The facade never computes and never calls HIP after load: every call is
// broadcast to the rank threads (RankWorker, tp/tp_rank_worker.h) as one command, the results are
// compared (2.3), and every failure goes through one error state machine (2.4):
//
//   kReady          -- normal operation.
//   kNeedsRecovery  -- a command failed (on any rank, for any reason, including a result or rng
//                      divergence). Device-work calls throw core::TpStateError("... call Reset()
//                      first"); host-only calls (the cached accessors, SetDflashInjectionEnabled,
//                      Vram) keep working; Reset() runs recovery (2.5) and returns to kReady.
//   kFatal          -- recovery failed, or a rank made no progress for 60 s (2.2 step 6). Every
//                      device-work call and Reset() throw core::TpStateError("tp: fatal, restart
//                      the process").
//
// The failing call itself rethrows the ROOT CAUSE with its original type: the lowest rank's
// exception that is not a core::TpAbortedError, else the lowest rank's TpAbortedError; every rank's
// message is logged to stderr as "[r4dx-tp] rank <r> (dev <d>): <what>".
//
// Modes (TpOptions::Mode, docs/tp.md 9.1): kReal -- one rank per GPU, HostMailboxComm through the
// pinned host mailbox (6.3), after the device validation and wall-clock check of 2.9 steps 3-4;
// kEmulate -- two ranks on ONE device, EmulatedComm (6.5); kNoop -- ONE rank's shard with a no-op
// all-reduce (timing only, tokens meaningless). Every mode bounds prefill submissions
// (TpOptions::submit_layers / max_inflight_units, docs/tp.md Appendix B N57). MTP, DFlash2 and
// vision run under every mode (docs/tp.md 8.1-8.3): the vision tower lives on rank 0, whose solo
// EncodeImages command returns HOST rows that every rank splices; the DFlash2 codebooks are read
// once, on this thread (CPU only), and shared by both ranks' drafters. R4DX_TP_FAULT
// ("<rank>:<n>:<kind>", docs/tp.md 9.1) arms TpOptions::fault_* from the environment.
//
// Thread safety: like Model, one facade thread issues every call. HasVision() and the other cached
// accessors are plain reads of values fixed at load and are safe from any thread.
#pragma once

#include <array>
#include <atomic>
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

#include "model.h"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_alloc_guard.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "text_model.h"
#include "tp/tp_group.h"
#include "tp/tp_rank_worker.h"

namespace r4dx::model {

class TpModel final : public TextModel {
 public:
  enum class State { kReady, kNeedsRecovery, kFatal };

  // docs/tp.md 2.9. `opts` is the ordinary single-model option set (opts.tp must be default: the
  // per-rank TpRankOptions are filled in here); `tp.world` must be 2.
  static std::unique_ptr<TpModel> Load(const ModelOptions& opts, const TpOptions& tp);
  // docs/tp.md 2.6: every rank's Model and endpoint are destroyed on that rank's thread; a rank that
  // does not finish within 30 s ends the process (quick_exit(3)) rather than free memory a kernel
  // may still touch.
  ~TpModel() override;
  TpModel(const TpModel&) = delete;
  TpModel& operator=(const TpModel&) = delete;

  // ---- identity / capabilities: cached at load, legal in every state ------------------------------
  const ModelConfig& Config() const override { return global_config_; }
  const std::string& ModelId() const override { return model_id_; }
  int64_t ImageTokenId() const override { return image_token_id_; }
  int64_t VisionMergeSize() const override { return vision_merge_size_; }
  bool HasVision() const override { return has_vision_; }
  bool MtpEnabled() const override { return mtp_enabled_; }
  bool MtpUsingReducedVocabDraft() const override { return mtp_reduced_vocab_; }
  bool DflashEnabled() const override { return dflash_enabled_; }
  int64_t SampledFallbackRows() const override { return cached_fallback_rows_; }
  int64_t PositionCount() const override { return cached_position_; }
  int64_t NumLoadedLayers() const override { return num_loaded_layers_; }
  int TpWorld() const override { return 2; }
  // One hipMemGetInfo per rank, on the rank threads (no TpComm): legal in kNeedsRecovery; in kFatal
  // it returns the last report it produced.
  std::vector<VramReport> Vram() const override;

  // ---- sequence state -------------------------------------------------------------------------
  // kReady: Model::Reset() + the cached injection policy on every rank. kNeedsRecovery: recovery
  // (docs/tp.md 2.5) first; kReady on success, kFatal + rethrow on failure. kFatal: TpStateError.
  void Reset() override;
  // Host-only (docs/tp.md 2.4): stores the policy on the facade; every forward command applies it
  // on each rank before it runs, and Reset() applies it after recovery. Legal in every state.
  void SetDflashInjectionEnabled(bool enabled) override { dflash_injection_ = enabled; }

  // docs/tp.md 8.3: a solo command on rank 0 (the tower's rank); `out` gets pageable host rows
  // (ImageRows::host, on_host() true) that PrefillMultimodal splices on every rank. A device-work
  // call: TpStateError unless kReady, and a failure makes the group kNeedsRecovery.
  void EncodeImages(const float* pixel_values, int64_t total_patches,
                    const std::vector<vision::GridThw>& grids, ImageRows* out,
                    vision::VisionEncodeStats* stats = nullptr) override;

  // ---- forward: one collective command each; results compared across ranks (docs/tp.md 2.3) ----
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids) override;
  // Every span must be host-resident (ImageSpan::embeds_on_host, EncodeImages' rows); the rows are
  // copied once into memory the command owns (Appendix B N53) and spliced by every rank.
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                       const std::vector<ImageSpan>& images) override;
  std::vector<float> DecodeStep(int32_t token_id) override;
  int32_t DecodeStepGreedy(int32_t token_id) override;
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params,
                            std::mt19937_64& rng) override;
  std::vector<int32_t> DecodeStepMtpGreedy(int32_t token_id, int64_t k) override;
  std::vector<int32_t> DecodeStepMtpSampled(int32_t token_id, int64_t k, const kernels::SampleParams& params,
                                            std::mt19937_64& rng) override;
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                              int64_t* walk_len_out = nullptr) override;
  std::vector<int32_t> DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                               const kernels::SampleParams& params, std::mt19937_64& rng,
                                               int64_t* walk_len_out = nullptr) override;

  // Profiling is not supported under tensor parallelism (docs/tp.md 1.2): core::TpUnsupportedError.
  StepProfile DecodeStepProfiled(int32_t token_id) override;
  StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids) override;

  // ---- tensor-parallel diagnostics (tests, tools, --stats) --------------------------------------
  State GetState() const { return state_; }
  const TpOptions& Options() const { return tp_; }
  // Rank threads: 2 (emulate, real) or 1 (noop).
  int NumRankThreads() const { return static_cast<int>(ranks_.size()); }
  // Per rank thread, in rank order: the endpoint's CallCounts() / Stats(). Host-side counters, read
  // on the rank threads; legal in kReady and kNeedsRecovery.
  std::vector<std::array<uint64_t, 2>> CallCounts();
  std::vector<core::TpCommStats> CommStats();
  // Per rank thread: the prefill submission bounding's counters (Model::TpSubmitStats). Same rules.
  std::vector<tp::SubmitBounder::Stats> SubmitStats();
  // The `--stats` line of docs/tp.md 9.1 (without the "[stats] " prefix): mode, devices, rank 0's
  // per-channel all-reduce calls, host exchanges, the max exchange wait and aborts over ranks, and
  // the bounding (submit_layers/max_inflight_units, units, cap waits, max cap wait). The comm
  // counters restart at every recovery; aborts and the bounding's counters count from load.
  std::string StatsLine();
  // Test-only: arms fault injection on `rank`'s endpoint now (counted from this call, docs/tp.md
  // TpOptions::fault_*). Legal in kReady.
  void ArmFaultInjection(int rank, int64_t at_allreduce, int kind);
  // Test-only: runs fn(model, rank) on every rank as ONE collective command -- the same state
  // check, allocation guard, CheckHealthy, abort-on-error and root-cause rethrow as a forward call,
  // no result comparison. The caller must issue the same collective sequence on every rank.
  void RunCollectiveForTest(const std::function<void(Model&, int)>& fn);

 private:
  struct RankSlot {
    int index = 0;   // position in ranks_ (== rank, except noop's single slot)
    int rank = 0;    // TP rank
    int device = 0;  // HIP ordinal (process-visible)
    std::unique_ptr<tp::RankWorker> worker;
    std::unique_ptr<tp::TpEndpoint> endpoint;  // emulate / real
    std::unique_ptr<core::TpComm> noop;        // noop
    std::optional<core::Stream> aux_stream;    // warm-up latency sample; lives as long as the endpoint
    std::optional<Model> model;
    // The facade's own copy of Comm(), written and read only on the facade thread (set once the
    // endpoint-creation command has returned, cleared before teardown), so the facade can poison the
    // group while this rank is still inside a command without reading `endpoint`/`noop` as the rank
    // writes them (docs/tp.md 2.6 step 1; the watchdog path of Appendix B N53).
    core::TpComm* facade_comm = nullptr;
    core::TpComm* Comm() const { return endpoint ? static_cast<core::TpComm*>(endpoint.get()) : noop.get(); }
  };
  enum class CmdKind {
    kCollective,  // may use TpComm: CheckHealthy after the body, Abort(kAbortHost) on a failure
    kPlain,       // no TpComm (load helpers, Reset, Vram, counters)
  };

  TpModel() = default;

  std::vector<int> AllSlots() const;
  // Runs body(slot) on each listed slot's rank thread as one command, waits with the progress
  // watchdog (`stall`), and rethrows the root cause of any failure (after logging every rank's).
  // Changes no state except kFatal on a stall -- which also poisons the whole group first (N53), so
  // a rank that later returns from the stuck HIP call stops at its next comm operation.
  void Run(const std::vector<int>& slots, const std::function<void(RankSlot&)>& body, CmdKind kind,
           std::chrono::milliseconds stall);
  void Run(const std::vector<int>& slots, const std::function<void(RankSlot&)>& body, CmdKind kind) {
    Run(slots, body, kind, stall_limit_);
  }
  // Run, and any failure makes the group kNeedsRecovery (unless it is already kFatal).
  void RunGuarded(const std::vector<int>& slots, const std::function<void(RankSlot&)>& body, CmdKind kind,
                  std::chrono::milliseconds stall);
  void RunGuarded(const std::vector<int>& slots, const std::function<void(RankSlot&)>& body, CmdKind kind) {
    RunGuarded(slots, body, kind, stall_limit_);
  }
  // Throws TpStateError unless kReady (the device-work precondition, docs/tp.md 2.4).
  void RequireReady() const;
  // Host-side poison of every endpoint the facade knows (RankSlot::facade_comm), from the facade
  // thread; legal while ranks are busy (docs/tp.md 2.6 step 1).
  void AbortAllFromFacade(uint32_t code, const char* why) noexcept;
  // docs/tp.md 2.5 (TpGroup::Recover for emulate/real; the counters only for noop).
  void Recover();
  // A result or rng mismatch: kNeedsRecovery + core::TpDivergenceError.
  [[noreturn]] void Diverged(const std::string& what);
  template <class T>
  void RequireAllEqual(const std::vector<T>& v, const char* what);
  void RequireRngsEqual(const std::vector<std::mt19937_64>& rngs, const char* what);
  std::vector<VramReport> VramImpl();

  // What one command's rank closures share (docs/tp.md Appendix B N53): the caller's fn, its
  // per-rank result slots and rank 0's re-cached counters live on the HEAP, co-owned by every posted
  // closure. If the progress watchdog fires, the facade throws while a stalled rank may still be
  // running its closure; that rank then touches only this block (never the facade's stack), and the
  // last owner frees it. `fn` must itself own what it reads (capture arguments by value or through a
  // shared_ptr): the forward methods below do.
  template <class Fn>
  struct CmdState {
    using R = std::invoke_result_t<std::decay_t<Fn>&, Model&, int>;
    using Slot = std::optional<std::conditional_t<std::is_void_v<R>, char, R>>;
    CmdState(Fn&& f, size_t n) : fn(std::forward<Fn>(f)), out(n) {}
    std::decay_t<Fn> fn;
    std::vector<Slot> out;
    bool inject = true;
    int64_t pos0 = 0, fallback0 = 0;
    auto Take() {
      std::vector<std::conditional_t<std::is_void_v<R>, char, R>> v;
      v.reserve(out.size());
      for (auto& o : out) v.push_back(std::move(*o));
      return v;
    }
  };

  // The forward-command wrapper (docs/tp.md 2.2 RunCollective): state check, then on every rank
  // one collective command running fn(model, slot index) inside a core::TpCollectiveScope, after
  // applying the cached injection policy; PositionCount/SampledFallbackRows are re-cached from
  // rank 0 on success. Returns every slot's result (or nothing for a void fn).
  template <class Fn>
  auto RunCollective(Fn&& fn)
      -> std::conditional_t<std::is_void_v<std::invoke_result_t<std::decay_t<Fn>&, Model&, int>>, void,
                            std::vector<std::invoke_result_t<std::decay_t<Fn>&, Model&, int>>> {
    using St = CmdState<Fn>;
    RequireReady();
    auto st = std::make_shared<St>(std::forward<Fn>(fn), ranks_.size());
    st->inject = dflash_injection_;
    RunGuarded(AllSlots(),
               [st](RankSlot& s) {
                 core::TpCollectiveScope scope;
                 s.model->SetDflashInjectionEnabled(st->inject);
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

  // docs/tp.md 2.2: fn(model, rank) on every rank thread as ONE plain command (no TpComm use, no
  // state change -- the caller decides what a failure means); results in rank-thread order. Same
  // heap ownership as RunCollective (N53).
  template <class Fn>
  auto RunAll(Fn&& fn)
      -> std::conditional_t<std::is_void_v<std::invoke_result_t<std::decay_t<Fn>&, Model&, int>>, void,
                            std::vector<std::invoke_result_t<std::decay_t<Fn>&, Model&, int>>> {
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
  // docs/tp.md 2.2: fn(model, rank) on ONE rank's thread (a solo command: must not use TpComm).
  template <class Fn>
  auto RunOne(int rank, Fn&& fn) -> std::invoke_result_t<std::decay_t<Fn>&, Model&, int> {
    using St = CmdState<Fn>;
    for (const auto& s : ranks_) {
      if (s->rank != rank) continue;
      auto st = std::make_shared<St>(std::forward<Fn>(fn), 1);
      Run({s->index},
          [st](RankSlot& x) {
            if constexpr (std::is_void_v<typename St::R>) {
              st->fn(*x.model, x.rank);
            } else {
              st->out[0].emplace(st->fn(*x.model, x.rank));
            }
          },
          CmdKind::kPlain);
      if constexpr (std::is_void_v<typename St::R>) {
        return;
      } else {
        return std::move(*st->out[0]);
      }
    }
    throw std::invalid_argument("TpModel::RunOne: no rank " + std::to_string(rank) + " in this group");
  }

  ModelOptions opts_;
  TpOptions tp_;
  State state_ = State::kReady;
  tp::CompletionGroup done_;
  tp::WorkerTiming timing_;
  std::chrono::milliseconds stall_limit_{60000};
  std::unique_ptr<tp::TpGroup> group_;           // emulate / real
  std::vector<std::unique_ptr<RankSlot>> ranks_;  // stable addresses (closures hold RankSlot*)

  // docs/tp.md 2.9 step 10 / 2.4: host-only state.
  ModelConfig global_config_;
  std::string model_id_;
  int64_t image_token_id_ = -1;
  int64_t vision_merge_size_ = 2;
  int64_t vision_patch_dim_ = 0;  // floats per pixel_values row EncodeImages copies (0: no vision)
  bool has_vision_ = false;
  bool mtp_enabled_ = false;
  bool mtp_reduced_vocab_ = false;
  bool dflash_enabled_ = false;
  int64_t num_loaded_layers_ = 0;
  int64_t cached_position_ = 0;
  int64_t cached_fallback_rows_ = 0;
  bool dflash_injection_ = true;  // the Model default (model.h's SetDflashInjectionEnabled)
  mutable std::vector<VramReport> cached_vram_;
};

}  // namespace r4dx::model
