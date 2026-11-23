// r4dx::model::Inkling2TpModel -- the tensor-parallel (`--tp 2`) TextModel of an inkling2_loop container
// (docs/inkling2-plan.md 5). A copy-adapt of Lfm2TpModel (lfm2_tp_model.h, itself GemmaTpModel's; LFM2 code untouched)
// specialised to Inkling2Model: one rank thread per rank, each owning that rank's Inkling2Model, TpComm endpoint and every
// device buffer they allocate; the facade never computes and never calls HIP after load -- every call is broadcast to the
// rank threads as one command, the results are compared, and every failure goes through the same error state machine:
//
//   kReady          -- normal operation.
//   kNeedsRecovery  -- a command failed on any rank (or the ranks' results diverged). Device-work calls throw
//                      core::TpStateError("... call Reset() first"); Reset() runs recovery (docs/tp.md 2.5).
//   kFatal          -- recovery failed, or a rank made no progress for 60 s. Everything throws.
//
// Modes (TpOptions::Mode): kReal -- one rank per GPU (HostMailboxComm); kEmulate -- two ranks on ONE device
// (EmulatedComm); kNoop -- ONE rank's shard with a no-op all-reduce (timing only).
//
// What is sharded (Inkling2Config::Shard, Inkling2Container::Load): q heads 24 -> 12 and kv heads 8 -> 4 per rank (GQA 3):
// wq / gate_attn / sink / q_scale / rel_scale by q head, wk / wv and the K/V short convs (taps, static_w columns,
// weight_proj rows) by kv head, wo row-parallel; EVERY expert TP-split (gate|up halves 1152 -> 576, down K 576); wr, the
// norms, convs, attention residuals, message gates, router and bias replicated (identical bits on both ranks); the
// embedding replicated; lm_head's 200064 stored rows split 100032 / 100032 (pad columns -inf), merged on the host
// (tp_vocab.h). Two all-reduces per visit (attention o, MoE combine) on bf16 [T, 2304]: 96 per token.
//
// Thread safety: like TpModel, one facade thread issues every call.
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

#include "inkling2_model.h"
#include "model_config.h"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_alloc_guard.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "text_model.h"
#include "tp/tp_group.h"
#include "tp/tp_rank_worker.h"

namespace r4dx::model {

struct ModelOptions;  // model.h

class Inkling2TpModel final : public TextModel, public TpDiagnostics {
 public:
  enum class State { kReady, kNeedsRecovery, kFatal };

  // `opts` is the ordinary single-model option set (opts.tp must be default); `tp.world` must be 2.
  static std::unique_ptr<Inkling2TpModel> Load(const ModelOptions& opts, const TpOptions& tp);
  ~Inkling2TpModel() override;
  Inkling2TpModel(const Inkling2TpModel&) = delete;
  Inkling2TpModel& operator=(const Inkling2TpModel&) = delete;

  // ---- identity / capabilities: cached at load, legal in every state ------------------------------
  const ModelConfig& Config() const override { return global_config_; }
  const std::string& ModelId() const override { return model_id_; }
  int64_t ImageTokenId() const override { return has_vision_ ? 200054 : -1; }
  int64_t VisionMergeSize() const override { return 2; }
  bool HasVision() const override { return has_vision_; }
  bool MtpEnabled() const override { return false; }
  bool MtpUsingReducedVocabDraft() const override { return false; }
  bool DflashEnabled() const override { return !lopts_.dspark_path.empty(); }
  int64_t SampledFallbackRows() const override { return cached_fallback_rows_; }
  int64_t PositionCount() const override { return cached_position_; }
  int64_t NumLoadedLayers() const override { return num_loaded_layers_; }
  int TpWorld() const override { return 2; }
  std::vector<VramReport> Vram() const override;

  // ---- sequence state -------------------------------------------------------------------------
  void Reset() override;
  void SetDflashInjectionEnabled(bool) override {}  // no drafter
  void SaveCheckpoint() override;     // each rank D2D-copies its own ShortConv state (needs ModelOptions::prompt_checkpoint)
  void RestoreCheckpoint() override;

  void EncodeImages(const float*, int64_t, const std::vector<vision::GridThw>&, ImageRows*,
                    vision::VisionEncodeStats* = nullptr) override;

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

  // TpDiagnostics (text_model.h).
  Health GroupHealth() const override { return static_cast<Health>(static_cast<int>(state_)); }
  const TpOptions& GroupOptions() const override { return tp_; }
  std::string GroupStatsLine() override { return StatsLine(); }

  // ---- tensor-parallel diagnostics (tests, tools, --stats) --------------------------------------
  State GetState() const { return state_; }
  const TpOptions& Options() const { return tp_; }
  int NumRankThreads() const { return static_cast<int>(ranks_.size()); }
  std::vector<std::array<uint64_t, 2>> CallCounts();
  std::vector<core::TpCommStats> CommStats();
  std::string StatsLine();
  // Test-only: arms fault injection on `rank`'s endpoint now (counted from this call). Legal in kReady.
  void ArmFaultInjection(int rank, int64_t at_allreduce, int kind);
  // Test-only: runs fn(model, rank) on every rank as ONE collective command.
  void RunCollectiveForTest(const std::function<void(Inkling2Model&, int)>& fn);

 private:
  struct RankSlot {
    int index = 0;
    int rank = 0;
    int device = 0;
    std::unique_ptr<tp::RankWorker> worker;
    std::unique_ptr<tp::TpEndpoint> endpoint;  // emulate / real
    std::unique_ptr<core::TpComm> noop;        // noop
    std::optional<core::Stream> aux_stream;
    std::optional<Inkling2Model> model;
    core::TpComm* facade_comm = nullptr;
    core::TpComm* Comm() const { return endpoint ? static_cast<core::TpComm*>(endpoint.get()) : noop.get(); }
  };
  enum class CmdKind { kCollective, kPlain };

  Inkling2TpModel() = default;

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

  template <class Fn>
  struct CmdState {
    using R = std::invoke_result_t<std::decay_t<Fn>&, Inkling2Model&, int>;
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

  template <class Fn>
  auto RunCollective(Fn&& fn)
      -> std::conditional_t<std::is_void_v<std::invoke_result_t<std::decay_t<Fn>&, Inkling2Model&, int>>, void,
                            std::vector<std::invoke_result_t<std::decay_t<Fn>&, Inkling2Model&, int>>> {
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

  template <class Fn>
  auto RunAll(Fn&& fn)
      -> std::conditional_t<std::is_void_v<std::invoke_result_t<std::decay_t<Fn>&, Inkling2Model&, int>>, void,
                            std::vector<std::invoke_result_t<std::decay_t<Fn>&, Inkling2Model&, int>>> {
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

  Inkling2ModelOptions lopts_;
  ModelConfig global_config_;
  TpOptions tp_;
  State state_ = State::kReady;
  tp::CompletionGroup done_;
  tp::WorkerTiming timing_;
  std::chrono::milliseconds stall_limit_{60000};
  std::unique_ptr<tp::TpGroup> group_;
  std::vector<std::unique_ptr<RankSlot>> ranks_;

  std::string model_id_;
  int64_t num_loaded_layers_ = 0;
  bool has_vision_ = false;
  int64_t cached_position_ = 0;
  int64_t cached_fallback_rows_ = 0;
  mutable std::vector<VramReport> cached_vram_;
};

}  // namespace r4dx::model
