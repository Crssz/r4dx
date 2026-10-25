// r4dx::model::Qwen35MoeTpModel -- the tensor-parallel (`--tp 2`) TextModel of a qwen3_5_moe (Ornith-1.5-122B) container
// (docs/qwen35moe-tp-design.md P3). A copy-adapt of Lfm2TpModel (lfm2_tp_model.h; docs/lfm2-plan.md open question 3: copied, the other
// facades untouched) over per-rank Qwen35MoeModels: one rank thread per rank, each owning that rank's Qwen35MoeModel (ONE stage with all 48
// layers, no hand-off), TpComm endpoint and every device buffer they allocate; the facade never computes and never calls HIP after load --
// every call is broadcast to the rank threads as one command, the results are compared, and every failure goes through the same error
// state machine:
//
//   kReady          -- normal operation.
//   kNeedsRecovery  -- a command failed on any rank (or the ranks' results diverged). Device-work calls throw
//                      core::TpStateError("... call Reset() first"); Reset() runs recovery (docs/tp.md 2.5).
//   kFatal          -- recovery failed, or a rank made no progress for 60 s. Everything throws.
//
// Modes (TpOptions::Mode): kReal -- one rank per GPU (HostMailboxComm); kEmulate -- two ranks on ONE device (EmulatedComm); kNoop -- ONE
// rank's shard with a no-op all-reduce (timing only).
//
// What is sharded (Qwen35MoeConfig::Shard, tp::RuleFor's qwen3_5_moe table, Qwen35MoeContainer::LoadShard): q heads 32 -> 16 and the kv head
// 2 -> 1 (GQA 16 kept); GDN key / value heads 16 / 64 -> 8 / 32 (conv channels 12288 -> 6144); qkv / z / qg / k / v column-parallel,
// out_proj / o row-parallel; EVERY expert TP-split (gate | up halves 1024 -> 512, down K 512, trellis slices cut from the one container by byte
// runs) and the bf16 shared expert split the same way; router, shared_gate and norms replicated; the embedding replicated (ONE pinned copy shared
// by both ranks), lm_head a vocab slice merged on the host (greedy pair, row summary, full row). Three all-reduces per layer pair of sublayers
// (GDN/attention out_proj, MoE combine) on the bf16 [T, 3072] outputs; the bf16 residual stays replicated.
//
// MoE under TP: both ranks compute the router from identical bits, so they pick identical experts; each rank's readback of the ids (prefill)
// is its own, the decode step runs the on-device grouped kernels (expert width 512), and both end in the one all-reduce of the combined partial.
//
// DFlash2 under TP is supported (P6; --dflash-k <= 7 is enforced by the CLI / server parsers). Not supported: MTP, vision, profiling. `--tp 2 --pp 2` is refused by
// LoadTextModel. Thread safety: like TpModel, one facade thread issues every call.#pragma once

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

#include "qwen35moe_model.h"
#include "model_config.h"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_alloc_guard.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "text_model.h"
#include "tp/tp_group.h"
#include "tp/tp_rank_worker.h"

namespace r4dx::model {

struct ModelOptions;  // model.h

class Qwen35MoeTpModel final : public TextModel, public TpDiagnostics {
 public:
  enum class State { kReady, kNeedsRecovery, kFatal };

  // `opts` is the ordinary single-model option set (opts.tp must be default); `tp.world` must be 2.
  static std::unique_ptr<Qwen35MoeTpModel> Load(const ModelOptions& opts, const TpOptions& tp);
  ~Qwen35MoeTpModel() override;
  Qwen35MoeTpModel(const Qwen35MoeTpModel&) = delete;
  Qwen35MoeTpModel& operator=(const Qwen35MoeTpModel&) = delete;

  // ---- identity / capabilities: cached at load, legal in every state ------------------------------
  const ModelConfig& Config() const override { return global_config_; }
  const std::string& ModelId() const override { return model_id_; }
  int64_t ImageTokenId() const override { return -1; }
  int64_t VisionMergeSize() const override { return 2; }
  bool HasVision() const override { return false; }
  bool MtpEnabled() const override { return false; }
  bool MtpUsingReducedVocabDraft() const override { return false; }
  bool DflashEnabled() const override { return dflash_enabled_; }
  int64_t SampledFallbackRows() const override { return cached_fallback_rows_; }
  int64_t PositionCount() const override { return cached_position_; }
  int64_t NumLoadedLayers() const override { return num_loaded_layers_; }
  int TpWorld() const override { return 2; }
  std::vector<VramReport> Vram() const override;

  // ---- sequence state -------------------------------------------------------------------------
  void Reset() override;
  void SetDflashInjectionEnabled(bool enabled) override;  // every rank's drafter (a no-op without --dflash)
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
  std::vector<tp::SubmitBounder::Stats> SubmitStats();
  std::string StatsLine();
  // Test-only: arms fault injection on `rank`'s endpoint now (counted from this call). Legal in kReady.
  void ArmFaultInjection(int rank, int64_t at_allreduce, int kind);
  // Test-only: runs fn(model, rank) on every rank as ONE collective command.
  void RunCollectiveForTest(const std::function<void(Qwen35MoeModel&, int)>& fn);

 private:
  struct RankSlot {
    int index = 0;
    int rank = 0;
    int device = 0;
    std::unique_ptr<tp::RankWorker> worker;
    std::unique_ptr<tp::TpEndpoint> endpoint;  // emulate / real
    std::unique_ptr<core::TpComm> noop;        // noop
    std::optional<core::Stream> aux_stream;
    std::unique_ptr<Qwen35MoeModel> model;
    core::TpComm* facade_comm = nullptr;
    core::TpComm* Comm() const { return endpoint ? static_cast<core::TpComm*>(endpoint.get()) : noop.get(); }
  };
  enum class CmdKind { kCollective, kPlain };

  Qwen35MoeTpModel() = default;

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
    using R = std::invoke_result_t<std::decay_t<Fn>&, Qwen35MoeModel&, int>;
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
      -> std::conditional_t<std::is_void_v<std::invoke_result_t<std::decay_t<Fn>&, Qwen35MoeModel&, int>>, void,
                            std::vector<std::invoke_result_t<std::decay_t<Fn>&, Qwen35MoeModel&, int>>> {
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
      -> std::conditional_t<std::is_void_v<std::invoke_result_t<std::decay_t<Fn>&, Qwen35MoeModel&, int>>, void,
                            std::vector<std::invoke_result_t<std::decay_t<Fn>&, Qwen35MoeModel&, int>>> {
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

  Qwen35MoeModelOptions lopts_;
  ModelConfig global_config_;
  TpOptions tp_;
  State state_ = State::kReady;
  tp::CompletionGroup done_;
  tp::WorkerTiming timing_;
  std::chrono::milliseconds stall_limit_{60000};
  std::unique_ptr<tp::TpGroup> group_;
  std::vector<std::unique_ptr<RankSlot>> ranks_;

  bool dflash_enabled_ = false;
  std::string model_id_;
  int64_t num_loaded_layers_ = 0;
  int64_t cached_position_ = 0;
  int64_t cached_fallback_rows_ = 0;
  mutable std::vector<VramReport> cached_vram_;
};

}  // namespace r4dx::model
