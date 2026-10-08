// TpModel's hybrid mode (docs/pp-tp2-hybrid.md 1-7, 15; P3 step 2): the PP-2 prefill on stage Models that live next to the TP=2 rank Models on
// the same two cards, with one bulk reshard into the ranks at the end of a call. The members here are TpModel's (they need RankSlot, Run and the
// state machine); tp_hybrid.h has the types, hybrid_dispatch.h / hybrid_ring.h / reshard_*.h / hybrid_sync.h the HIP-free policy they run.
//
// One pipelined call = ONE plain command on both rank workers (no TpComm, not inside a TpCollectiveScope), after a small read command:
//   read     both ranks collapse their speculative windows (GDN state back to window 0); rank 0's scalars + MTP seed and both drafters'
//            frontiers come back to the facade, which settles the dispatch rule (hybrid_dispatch.h) with them;
//   command  each rank thread, in its own Tp2TuningScope per Model:
//     1  StageSetSyncState: the scalars + seed into this card's stage;
//     2  warm gather (only when the TpMasterTracker says the stage is stale): the TP shards' state into the stages, the same-card half
//        device-to-device, the other half through the pinned host ring, both directions concurrently (RunLockstep);
//     3  the pipelined stage prefill: rank 1 / card X runs RunStageA, rank 0 / card Y RunStageB (+ the DFlash tail capture), the carry
//        through the StageChannel; Y publishes its end state, logits and tail (the end gate);
//     4  the bulk reshard: each stage's live state into BOTH ranks, same-card half D2D, the rest through the ring, concurrently;
//     5  TpAdoptPrefill on both ranks from Y's end state, then the DFlash tail into both drafters.
// Any error on either card poisons the channel, the ring pipes and the gate, which frees the other thread at once; the facade joins both
// (the 60 s no-progress watchdog is Run's: kFatal), rethrows the root cause (Run ranks a poison consequence last) and the group is
// kNeedsRecovery until Reset(), which resets both stage Models, the channel and the tracker next to the ranks.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>

#include "dflash_tail_plan.h"
#include "linear.h"
#include "pp_model.h"
#include "pp_plan.h"
#include "r4dx/core/error.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "tp_hybrid.h"

namespace r4dx::model {

namespace {

using Clock = std::chrono::steady_clock;
double Ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }
constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
double ToGibD(size_t bytes) { return static_cast<double>(bytes) / kGiB; }
double ToMib(uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0); }

std::string GetEnvVar(const char* name) {
  char* v = nullptr;
  size_t len = 0;
  std::string out;
  if (_dupenv_s(&v, &len, name) == 0 && v != nullptr) out = v;
  std::free(v);
  return out;
}

// What the load reads off a rank Model, on its thread.
struct RankProbe {
  bool rotated = false;
  int64_t block = 0, conv_pitch = 0;
  std::vector<int64_t> targets;
  size_t free_b = 0, total_b = 0;
};
// ... and off its stage Model.
struct StageProbe {
  int64_t chunk = 0, block = 0, hidden = 0, conv_pitch = 0, layers = 0;
  size_t stride = 0;
  bool i8 = false;
  size_t free_b = 0, total_b = 0;
};

}  // namespace

// ---- Load: the request -------------------------------------------------------------------------------------------------------------

bool ResolveHybridRequest(const ModelOptions& opts, const TpOptions& tp, const PpOptions& pp, const std::vector<int>& devices) {
  if (!ResolvePpSwitch(opts.pp)) return false;  // no `--pp 2`: plain --tp 2
  int sw = pp.hybrid;
  if (sw < 0) {
    const std::string env = GetEnvVar("R4DX_HYBRID");
    sw = hybrid::ParseHybridSwitch(env.c_str());
    if (sw < 0) throw std::invalid_argument("R4DX_HYBRID='" + env + "' is not 0 / off or 1 / on");
  }
  if (sw == 0) {
    std::fprintf(stderr, "[r4dx-hybrid] --hybrid off / R4DX_HYBRID=0: --pp 2 is ignored, plain --tp 2\n");
    return false;
  }
  if (tp.mode != TpOptions::Mode::kReal) {
    throw std::invalid_argument("TpModel::Load: the hybrid mode (--tp 2 --pp 2) needs --tp-mode real (two GPUs); add --hybrid off to run --tp-mode "
                                "emulate / noop without the pipelined prefill");
  }
  if (opts.pp_emulate_split > 0) {
    throw std::invalid_argument("TpModel::Load: --pp 2 and the PP-emulate mode (R4DX_PP_EMULATE) are mutually exclusive");
  }
  if (pp.split < 0) throw std::invalid_argument("TpModel::Load: --pp-split must be >= 0 (0 = auto)");
  if (pp.min_rows_given && pp.min_rows < 1) throw std::invalid_argument("TpModel::Load: --pp-min-rows must be >= 1");
  if (pp.slots < 2 || pp.slots > 8) throw std::invalid_argument("TpModel::Load: the hybrid's stage ring needs 2..8 slots");
  if (pp.timeout_ms < 100) throw std::invalid_argument("TpModel::Load: the pipeline timeout must be >= 100 ms");
  if (pp.submit_layers < 0 || pp.submit_layers > 64 || pp.max_inflight < 0 || pp.max_inflight > 64) {
    throw std::invalid_argument("TpModel::Load: --pp-submit-layers and --pp-max-inflight must be in [0, 64]");
  }
  if (pp.verify) std::fprintf(stderr, "[r4dx-hybrid] --pp-verify is not available in the hybrid mode (the stages' state moves to the ranks, not to a peer mirror): ignored\n");
  if (pp.hybrid_ctx < 0) throw std::invalid_argument("TpModel::Load: --hybrid-ctx must be 'auto' or a token count");
  if (pp.hybrid_reserve_gib > 24.0) throw std::invalid_argument("TpModel::Load: --hybrid-reserve-gib is above the card");
  // Placement: stage B (decode) is TP rank 0, stage A rank 1 -- the two defaults already agree (docs/tp.md 9.2, docs/pp-prefill.md 4.2), so
  // an explicit --pp-devices / R4DX_PP_DEVICES must name the same cards in the same roles.
  std::vector<int> requested = pp.devices;
  const char* src = "--pp-devices";
  if (requested.empty()) {
    const std::string env = GetEnvVar("R4DX_PP_DEVICES");
    if (const std::string why = pp::ParseDevicePair(env, &requested); !why.empty()) {
      throw std::invalid_argument("TpModel::Load: R4DX_PP_DEVICES " + why);
    }
    src = "R4DX_PP_DEVICES";
  }
  if (!requested.empty() && (requested.size() != 2 || devices.size() != 2 || requested[0] != devices[0] || requested[1] != devices[1])) {
    throw std::invalid_argument(std::string("TpModel::Load: the hybrid mode runs the decode stage on TP rank 0's card and the front stage on rank 1's; ") +
                                src + " must equal --tp-devices (rank 0, rank 1) = " +
                                (devices.size() == 2 ? std::to_string(devices[0]) + "," + std::to_string(devices[1]) : std::string("?")));
  }
  return true;
}

// ---- Load: the stages --------------------------------------------------------------------------------------------------------------

bool TpModel::HybridLoad(const ModelOptions& opts, const PpOptions& pp, const std::shared_ptr<const core::PinnedBuffer<uint16_t>>& embed) {
  const auto kNoStall = tp::ProgressWatchdog::kNoStallLimit;
  const auto t_load0 = Clock::now();
  const int64_t layers = num_loaded_layers_;
  const int64_t split = pp.split > 0 ? pp.split : pp::DefaultSplit(dflash_enabled_, layers);
  if (!pp::ValidSplit(split, layers)) {
    throw std::invalid_argument("TpModel::Load: --pp-split " + std::to_string(split) + " must leave at least one of the " + std::to_string(layers) +
                                " loaded layers on each side");
  }
  int slot_of[2] = {-1, -1};  // ranks_ index of stage A (rank 1) / stage B (rank 0)
  for (const auto& s : ranks_) slot_of[s->rank == 1 ? hybrid::kStageA : hybrid::kStageB] = s->index;
  if (slot_of[0] < 0 || slot_of[1] < 0) throw std::logic_error("TpModel::HybridLoad: the group has no rank 0 / rank 1");

  hy_ = std::make_unique<Hybrid>();
  Hybrid& h = *hy_;
  try {
    // ---- 1. what the ranks hold: rotation, KV block, conv pitch, the drafter's target layers, free VRAM (after the warm-up)
    std::vector<RankProbe> rp(ranks_.size());
    Run(AllSlots(),
        [&rp](RankSlot& s) {
          Model& m = *s.model;
          RankProbe& p = rp[static_cast<size_t>(s.index)];
          p.rotated = m.GetContainer().HasRotation();
          p.block = m.PpKvBlockSize();
          p.conv_pitch = m.ReshardConvPitch();
          p.targets = m.DflashTargetLayers();
          R4DX_HIP_CHECK(hipMemGetInfo(&p.free_b, &p.total_b));
        },
        CmdKind::kPlain, kNoStall);
    for (const RankProbe& p : rp) {
      if (p.rotated) {
        throw std::invalid_argument("the hybrid mode is not available on a rotated (quant2) container: the stack-entry and -exit rotations straddle the "
                                    "stages (docs/pp-prefill.md 4); add --hybrid off");
      }
    }
    if (rp[0].block != rp[1].block || rp[0].conv_pitch != rp[1].conv_pitch || rp[0].targets != rp[1].targets) {
      throw std::logic_error("the two ranks disagree on the KV block, the conv line pitch or the drafter's target layers");
    }
    if (dflash_enabled_ && rp[0].targets.empty()) throw std::logic_error("the ranks have a drafter but no feature-capture target layers");
    const int64_t block = rp[0].block;
    h.split = split;
    h.block = block;
    h.layers = layers;
    h.mtp = mtp_enabled_;
    h.dflash = dflash_enabled_;
    h.targets = rp[0].targets;
    for (const int64_t t : h.targets) h.dfl_cols_x += t < split ? 1 : 0;
    h.hidden = global_config_.hidden_size;
    h.min_rows = pp.min_rows_given ? pp.min_rows : hybrid::kHybridMinRowsDefault;
    h.step_timeout = std::chrono::milliseconds(pp.timeout_ms);
    for (int s = 0; s < 2; ++s) h.slot_of_stage[s] = slot_of[s];

    // ---- 2. the geometry and the plan of the stage-KV capacity S, from the MEASURED free VRAM (it already holds the desktop)
    ModelConfig gc = global_config_;
    gc.num_hidden_layers = layers;
    gc.layer_types.resize(static_cast<size_t>(layers));
    h.geo = hybrid::StateGeometry::FromRules(gc, block);
    const hybrid::StageLayers la = hybrid::LayersOfStage(h.geo, split, hybrid::kStageA), lb = hybrid::LayersOfStage(h.geo, split, hybrid::kStageB);
    const int64_t bpt_x = hybrid::StageKvBytesPerToken(static_cast<int64_t>(la.attn.size()), h.geo.kv_heads_full, h.geo.head_dim);
    const int64_t bpt_y = hybrid::StageKvBytesPerToken(static_cast<int64_t>(lb.attn.size()), h.geo.kv_heads_full, h.geo.head_dim);
    for (int s = 0; s < 2; ++s) {
      const hybrid::StageLayers& sl = s == hybrid::kStageA ? la : lb;
      h.gather_bytes_per_row[s] = hybrid::GatherBytesPerRow(static_cast<int64_t>(sl.attn.size()), h.geo.ranks[static_cast<size_t>(s)].kv_heads, h.geo.head_dim);
    }
    const double scale = hybrid::LayerWeightScale(opts.layout == Layout::kBf16);
    const double reserve_x = pp.hybrid_reserve_gib >= 0 ? pp.hybrid_reserve_gib : hybrid::kReserveXGib;
    const double reserve_y = pp.hybrid_reserve_gib >= 0 ? std::min(pp.hybrid_reserve_gib, hybrid::kReserveYGib) : hybrid::kReserveYGib;
    const size_t free_x = rp[static_cast<size_t>(slot_of[hybrid::kStageA])].free_b, free_y = rp[static_cast<size_t>(slot_of[hybrid::kStageB])].free_b;
    const hybrid::CardBudget bx = hybrid::BudgetX(static_cast<int64_t>(free_x), hybrid::StageFixedX(split, scale), reserve_x);
    const hybrid::CardBudget by = hybrid::BudgetY(static_cast<int64_t>(free_y), hybrid::StageFixedY(layers, split, mtp_enabled_, scale), reserve_y);
    const hybrid::StageCtxPlan plan = hybrid::PlanStageCtxPerCard(opts.max_ctx, pp.hybrid_ctx, bx, by, bpt_x, bpt_y);
    const int64_t S = hybrid::AlignStageCtx(plan.s, block);
    std::fprintf(stderr,
                 "[r4dx-hybrid] split k=%lld (stage X: %zu attention layers, stage Y: %zu), stage-KV capacity S=%lld tokens (card X holds %lld, card Y %lld; "
                 "--max-ctx %lld); planned free after the stage load: X %.2f GiB (reserve %.2f), Y %.2f GiB (reserve %.2f); free now: X %.2f, Y %.2f GiB\n",
                 static_cast<long long>(split), la.attn.size(), lb.attn.size(), static_cast<long long>(S), static_cast<long long>(plan.s_x),
                 static_cast<long long>(plan.s_y), static_cast<long long>(opts.max_ctx), hybrid::ToGib(plan.x_free_after), reserve_x,
                 hybrid::ToGib(plan.y_free_after), reserve_y, ToGibD(free_x), ToGibD(free_y));
    if (!plan.engaged) {
      HybridDrop(plan.refusal);
      return false;
    }
    if (const std::string why = hybrid::CheckStageCtx(S, opts.max_ctx, block); !why.empty()) {
      HybridDrop(why);
      return false;
    }
    h.stage_ctx = S;

    // ---- 3. the stage Models, on the rank threads (the rank Model first on each: the lease of its embedding mirror is borrowed)
    std::vector<StageProbe> sp(ranks_.size());
    Run(AllSlots(),
        [&](RankSlot& s) {
          const bool front = s.rank == 1;
          ModelOptions so = opts;
          so.tp = TpRankOptions{};
          so.max_ctx = S;
          so.pp = 0;
          so.pp_emulate_split = 0;
          so.prompt_checkpoint = false;
          so.dflash_container.clear();
          so.dflash_draft_k = 0;
          so.stage_only.role = front ? stage::Role::kFront : stage::Role::kBack;
          so.stage_only.split = split;
          so.stage_only.shared_embed_host = embed;
          const std::shared_ptr<const EmbedMirrorLease> lease = s.model->GetContainer().EmbedMirrorLeaseHandle();
          so.stage_only.borrowed_embed_mirror = lease;
          so.stage_only.embed_device_resident_decided = lease ? -1 : 0;
          if (front) {
            so.layer_limit = split + 1;
            so.mtp_draft_k = 0;
          }
          s.stage.emplace(Model::Load(so));
          Model& st = *s.stage;
          st.CheckStageKv(S);
          StageProbe& p = sp[static_cast<size_t>(s.index)];
          p.chunk = st.PrefillChunkRows();
          p.i8 = st.PrefillInt8Enabled();
          p.block = st.PpKvBlockSize();
          p.stride = st.PpKvBlockStrideBytes();
          p.hidden = st.Config().hidden_size;
          p.conv_pitch = st.ReshardConvPitch();
          p.layers = st.GetContainer().NumLoadedLayers();
        },
        CmdKind::kPlain, kNoStall);
    const StageProbe& px = sp[static_cast<size_t>(slot_of[hybrid::kStageA])];
    const StageProbe& py = sp[static_cast<size_t>(slot_of[hybrid::kStageB])];
    if (px.chunk != py.chunk || px.i8 != py.i8) {
      throw std::runtime_error("the stages chose different prefill chunk sizes / int8 prefill (X " + std::to_string(px.chunk) + "/" + std::to_string(px.i8) +
                               ", Y " + std::to_string(py.chunk) + "/" + std::to_string(py.i8) + "): the bits would differ");
    }
    if (px.layers != layers || py.layers != layers || px.hidden != h.hidden || py.hidden != h.hidden || px.block != block || py.block != block) {
      throw std::runtime_error("the stage Models disagree with the ranks on the layer count / hidden size / KV block");
    }
    h.chunk_rows = px.chunk;

    // ---- 4. the pinned buffers, allocated by the thread that owns the producing card, none lazily
    pp::StageBufferGeometry bg;
    bg.slots = pp.slots;
    bg.chunk_rows = h.chunk_rows;
    bg.hidden = h.hidden;
    bg.dfl_cols_max = h.dfl_cols_x;
    bg.attn_layers_max = 0;  // the hybrid's slots carry no KV payload
    bg.block_size = block;
    bg.block_stride_bytes = px.stride;
    pp::StageBufferSpec spec = pp::MakeStageBufferSpec(bg, /*carry_kv=*/false);
    spec.gdn_bytes = 0;  // no GDN hand-off, no sync-back: the reshard moves each stage's state to the ranks directly
    spec.kv_sync_bytes = 0;
    h.run.split = split;
    h.run.timeout_ms = pp.timeout_ms;
    h.run.carry_kv = false;
    h.run.gdn_handoff = false;
    h.bufs = std::make_unique<pp::StageBuffers>(spec, pp::PinnedAllocator());
    const int64_t dfl_cols = static_cast<int64_t>(h.targets.size());
    pp::StageBuffers* const bufs = h.bufs.get();
    const bool want_tail = h.dflash;
    const int64_t hidden = h.hidden;
    Run({slot_of[hybrid::kStageA]},
        [bufs](RankSlot& s) {
          bufs->AllocateAProduced();  // the slot ring: stage X's D2H lands in it
          s.hy_ring = core::PinnedBuffer<uint8_t>(static_cast<size_t>(hybrid::kHybridRingBytes), hipHostMallocPortable);
        },
        CmdKind::kPlain, kNoStall);
    Run({slot_of[hybrid::kStageB]},
        [bufs, want_tail, dfl_cols, hidden](RankSlot& s) {
          bufs->AllocateBProduced();
          s.hy_ring = core::PinnedBuffer<uint8_t>(static_cast<size_t>(hybrid::kHybridRingBytes), hipHostMallocPortable);
          if (want_tail) {
            s.hy_tail = core::PinnedBuffer<uint16_t>(static_cast<size_t>(hybrid::kTailCapacityRows * dfl_cols * hidden), hipHostMallocPortable);
          }
        },
        CmdKind::kPlain, kNoStall);
    h.channel = std::make_unique<pp::StageChannel>(h.bufs->SlotPointers(), spec.slot_bytes);

    // ---- 5. attach the stages, set up the reshard executor, the DFlash feature capture
    const std::vector<int64_t> targets = h.targets;
    pp::StageChannel* const channel = h.channel.get();
    const pp::StageRunOptions run = h.run;
    const int bound_layers = pp.submit_layers, bound_inflight = pp.max_inflight;
    Run(AllSlots(),
        [&](RankSlot& s) {
          const bool front = s.rank == 1;
          Model& st = *s.stage;
          Model::PpStageSetup setup = pp::MakeStageSetup(front ? Model::PpRole::kStageA : Model::PpRole::kStageB, run, channel);
          std::atomic<uint64_t>* const beat = &s.worker->Heartbeat();
          setup.on_chunk = [beat] { beat->fetch_add(1, std::memory_order_relaxed); };  // the progress watchdog's input
          st.PpAttach(setup);
          // Stage X drives the desktop card, which cannot preempt compute: its chunks submit in bounded units, as the PP stage A's do.
          if (front) st.PpEnableBounding(bound_layers, bound_inflight);
          st.ReshardInit();
          s.model->ReshardInit();
          if (!targets.empty()) {
            std::vector<int64_t> mine;
            for (const int64_t t : targets) {
              if (!front || t < split) mine.push_back(t);  // stage X captures the targets below k, stage Y all of them
            }
            if (!mine.empty()) st.AttachDflashFeatureCapture(mine);
          }
        },
        CmdKind::kPlain, kNoStall);

    // ---- 6. the plan parameters (the conv line pitches are the Models' own), the tracker
    h.plan_params = hybrid::PlanParams::ForSplit(h.geo, split, rp[0].conv_pitch, mtp_enabled_);
    h.plan_params.stage_conv_pitch = px.conv_pitch;
    h.plan_params.stage_b_conv_pitch = py.conv_pitch;
    h.tracker = std::make_unique<hybrid::TpMasterTracker>(block, mtp_enabled_);

    // ---- 7. what is left: a stage load that leaves less than the hard floor is not kept
    std::vector<RankProbe> fr(ranks_.size());
    Run(AllSlots(), [&fr](RankSlot& s) { R4DX_HIP_CHECK(hipMemGetInfo(&fr[static_cast<size_t>(s.index)].free_b, &fr[static_cast<size_t>(s.index)].total_b)); },
        CmdKind::kPlain, kNoStall);
    const double fx = ToGibD(fr[static_cast<size_t>(slot_of[hybrid::kStageA])].free_b), fy = ToGibD(fr[static_cast<size_t>(slot_of[hybrid::kStageB])].free_b);
    std::fprintf(stderr,
                 "[r4dx-hybrid] stage Models loaded: free VRAM now X %.2f GiB (planned %.2f), Y %.2f GiB (planned %.2f); pinned: stage ring %d x %.1f MiB, "
                 "2 x %.0f MiB reshard rings, DFlash tail %.0f MiB\n",
                 fx, hybrid::ToGib(plan.x_free_after), fy, hybrid::ToGib(plan.y_free_after), pp.slots, ToMib(spec.slot_bytes), ToMib(hybrid::kHybridRingBytes),
                 want_tail ? ToMib(static_cast<uint64_t>(hybrid::kTailCapacityRows * dfl_cols * hidden) * 2) : 0.0);
    if (fx < hybrid::kHardFreeFloorGib || fy < hybrid::kHardFreeFloorGib) {
      HybridDrop("the stage load left only " + std::to_string(std::min(fx, fy)) + " GiB free on a card (floor " + std::to_string(hybrid::kHardFreeFloorGib) +
                 " GiB): lower --hybrid-ctx or --max-ctx");
      return false;
    }

    // ---- 8. one pipelined warm-up through every path (kernel modules, the ring, the gather, the tail), then everything is reset
    HybridWarmup();
  } catch (const std::invalid_argument&) {
    if (state_ != State::kFatal) HybridDrop("configuration error");
    throw;  // a refused configuration (rotated container, ...) is the caller's to see
  } catch (const std::exception& e) {
    if (state_ == State::kFatal) throw;
    HybridDrop(std::string("loading the stages failed: ") + e.what());
    return false;
  }
  std::fprintf(stderr, "[r4dx-hybrid] engaged in %.1f s: pipelined prefill for calls of >= %lld rows (S = %lld tokens), TP=2 decode and short prefills unchanged\n",
               Ms(t_load0, Clock::now()) / 1000.0, static_cast<long long>(h.min_rows), static_cast<long long>(h.stage_ctx));
  return true;
}

void TpModel::HybridDrop(const std::string& why) {
  std::fprintf(stderr, "[r4dx-hybrid] hybrid mode OFF, plain --tp 2: %s\n", why.c_str());
  if (!hy_) return;
  if (state_ == State::kFatal) {
    // A rank is stuck inside a HIP call: what it may still touch must not be freed. Leak it (the process is going down anyway).
    (void)hy_.release();
    return;
  }
  pp::StageBuffers* const bufs = hy_->bufs.get();
  try {
    Run(AllSlots(),
        [bufs](RankSlot& s) {
          s.stage.reset();
          s.hy_ring = core::PinnedBuffer<uint8_t>();
          s.hy_tail = core::PinnedBuffer<uint16_t>();
          if (bufs != nullptr) {
            if (s.rank == 1) {
              bufs->ReleaseAProduced();
            } else {
              bufs->ReleaseBProduced();
            }
          }
        },
        CmdKind::kPlain, tp::ProgressWatchdog::kNoStallLimit);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[r4dx-hybrid] freeing the stages failed (%s)\n", e.what());
    if (state_ == State::kFatal) {
      (void)hy_.release();
      return;
    }
  }
  hy_.reset();  // the channel, the (already released) buffers
  // The ranks may have been touched by a failed warm-up: back to a fresh sequence.
  try {
    RunAll([](Model& m, int) {
      m.Reset();
      m.SetDflashInjectionEnabled(true);
    });
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[r4dx-hybrid] resetting the ranks failed (%s)\n", e.what());
  }
  cached_position_ = 0;
  cached_fallback_rows_ = 0;
}

void TpModel::HybridWarmup() {
  Hybrid& h = *hy_;
  const auto t0 = Clock::now();
  const auto ids_of = [](int64_t n, int salt) {
    std::vector<int32_t> ids(static_cast<size_t>(n));
    for (size_t i = 0; i < ids.size(); ++i) ids[i] = static_cast<int32_t>(300 + (i * 37 + static_cast<size_t>(salt) * 977) % 5000);
    return ids;
  };
  const auto call = [&](const std::vector<int32_t>& ids) {
    int64_t injected[2] = {-1, -1};
    const StageSyncState sync = HybridReadSync(injected);
    const int64_t p0 = sync.pos;
    const hybrid::TailPlan tail = hybrid::PlanTail(h.dflash, true, p0, p0 + static_cast<int64_t>(ids.size()), injected[0]);
    if (!tail.problem.empty()) throw std::runtime_error("warm-up: " + tail.problem);
    (void)HybridPrefillRun(ids, nullptr, nullptr, sync, tail.use, tail.start, tail.rows);
  };
  // Two super-chunks and a tail (the cold path of every stage kernel, the ring, the scatter, the DFlash tail) ...
  call(ids_of(520, 0));
  // ... then a warm call that has to gather everything (as after a decode): the GDN state, the seed, the MTP block, the stale KV rows.
  h.tracker->TpOnly(0);
  call(ids_of(264, 1));
  RunAll([](Model& m, int) {
    m.Reset();
    m.SetDflashInjectionEnabled(true);
  });
  HybridResetStages();
  cached_position_ = 0;
  cached_fallback_rows_ = 0;
  h.stats = HybridStats{};
  h.channel->ResetStats();
  std::fprintf(stderr, "[r4dx-hybrid] warm-up (520 + 264 rows, full gather on the second): %.0f ms\n", Ms(t0, Clock::now()));
}

// ---- sequence state ------------------------------------------------------------------------------------------------------------------------

void TpModel::HybridResetStages() {
  if (!hy_) return;
  Hybrid& h = *hy_;
  Run(AllSlots(),
      [](RankSlot& s) {
        if (!s.stage.has_value()) return;
        Model& st = *s.stage;
        Tp2TuningScope scope(st.Tp2TuningForModel());
        if (st.StageDflashTailArmed()) {  // a failed call left the capture armed: it holds the observer slot
          try {
            (void)st.StageDisarmDflashTail();
          } catch (...) {
          }
        }
        st.Reset();
      },
      CmdKind::kPlain);
  h.channel->Reset();  // both threads are out of it (Run returned)
  h.tracker->Reset();
}

void TpModel::NoteTpMoved(int64_t from_pos) {
  if (hy_) hy_->tracker->TpOnly(from_pos);
}

// ---- dispatch ------------------------------------------------------------------------------------------------------------------------------

std::optional<std::vector<float>> TpModel::TryHybridPrefill(const std::vector<int32_t>& ids, const std::vector<ImageSpan>* spans,
                                                            const std::shared_ptr<std::vector<uint16_t>>& rows) {
  Hybrid& h = *hy_;
  RequireReady();  // the TP path's own first check (RunCollective): the same TpStateError
  const int64_t n = static_cast<int64_t>(ids.size());
  const auto decline = [&h](hybrid::Why why) -> std::optional<std::vector<float>> {
    ++h.stats.tp_prefill_calls;
    ++h.stats.declined[static_cast<int>(why)];
    return std::nullopt;
  };
  hybrid::DispatchIn in;
  in.engaged = true;
  in.rows = n;
  in.p0 = cached_position_;
  in.min_rows = h.min_rows;
  in.stage_ctx = h.stage_ctx;
  in.gather_bytes_per_row[0] = h.gather_bytes_per_row[0];
  in.gather_bytes_per_row[1] = h.gather_bytes_per_row[1];
  hybrid::StaleRows(h.tracker->PlanSync(in.p0), in.stale_rows);
  if (const hybrid::Dispatch d = hybrid::Decide(in); d.route != hybrid::Route::kHybrid) return decline(d.why);

  // The rest needs the ranks (the drafters' frontier): one small command, which also collapses their speculative windows.
  int64_t injected[2] = {-1, -1};
  const StageSyncState sync = HybridReadSync(injected);
  if (sync.pos != cached_position_) {
    Diverged("hybrid: rank 0 is at position " + std::to_string(sync.pos) + ", the facade cached " + std::to_string(cached_position_));
  }
  hybrid::TailPlan tail = hybrid::PlanTail(h.dflash, dflash_injection_, sync.pos, sync.pos + n, injected[0]);
  if (tail.use && injected[0] != injected[1]) tail.problem = "the two drafters' frontiers differ (" + std::to_string(injected[0]) + " vs " + std::to_string(injected[1]) + ")";
  in.dflash_problem = tail.problem;
  if (const hybrid::Dispatch d = hybrid::Decide(in); d.route != hybrid::Route::kHybrid) return decline(d.why);

  return HybridPrefillRun(ids, spans, rows, sync, tail.use, tail.start, tail.rows);
}

StageSyncState TpModel::HybridReadSync(int64_t injected[2]) {
  struct Read {
    StageSyncState state;
    int64_t injected[2] = {-1, -1};
  };
  auto rd = std::make_shared<Read>();
  const bool dflash = hy_->dflash;
  RunGuarded(AllSlots(),
             [rd, dflash](RankSlot& s) {
               Model& m = *s.model;
               m.ReshardCollapse();
               m.ReshardSync();
               if (dflash) rd->injected[s.index] = m.DflashInjectedCount();
               if (s.rank == 0) rd->state = m.StageGetSyncState();
             },
             CmdKind::kPlain);
  injected[0] = rd->injected[0];
  injected[1] = rd->injected[1];
  return std::move(rd->state);
}

std::vector<float> TpModel::HybridPrefillRun(const std::vector<int32_t>& ids, const std::vector<ImageSpan>* spans,
                                             const std::shared_ptr<std::vector<uint16_t>>& rows, const StageSyncState& sync, bool tail_use,
                                             int64_t tail_start, int64_t tail_rows) {
  Hybrid& h = *hy_;
  const auto t_call0 = Clock::now();
  const int64_t n = static_cast<int64_t>(ids.size());
  const int64_t p0 = sync.pos;
  auto c = std::make_shared<HyCall>();
  c->ids = ids;
  if (spans != nullptr) {
    c->spans = *spans;
    c->rows = rows;  // the spans point into it
    c->multimodal = true;
  }
  c->p0 = p0;
  c->n = n;
  c->call_id = ++h.call_id;
  c->sync = sync;
  c->inject = dflash_injection_;
  const hybrid::TpMasterTracker::SyncPlan sync_plan = h.tracker->PlanSync(p0);
  c->gather = hybrid::GatherPlan(h.geo, h.plan_params, sync_plan);
  c->scatter = hybrid::ScatterPlan(h.geo, h.plan_params, p0, p0 + n);
  if (h.neg_control == HybridNegControl::kSkipGather) c->gather.ops.clear();  // test-only: the stale mirror
  if (h.neg_control == HybridNegControl::kWrongRankGdn) {                     // test-only: rank 0's first recurrent op reads rank 1's v-heads
    for (hybrid::CopyOp& op : c->scatter.ops) {
      if (op.kind != hybrid::StateKind::kGdnRecurrent || op.rank != 0) continue;
      for (const hybrid::CopyOp& other : c->scatter.ops) {
        if (other.kind == op.kind && other.layer == op.layer && other.stage == op.stage && other.rank != op.rank) {
          op.runs[0].full_off = other.runs[0].full_off;
          break;
        }
      }
      break;
    }
  }
  for (int s = 0; s < 2; ++s) {
    c->gather_batches[s] = hybrid::PackCrossOps(c->gather, s, hybrid::kHybridRingPieceBytes, hybrid::kHybridRingSlots);
    c->scatter_batches[s] = hybrid::PackCrossOps(c->scatter, s, hybrid::kHybridRingPieceBytes, hybrid::kHybridRingSlots);
  }
  c->tail.use = tail_use;
  c->tail.start = tail_start;
  c->tail.rows = tail_rows;
  c->tail_host = tail_use ? ranks_[static_cast<size_t>(h.slot_of_stage[hybrid::kStageB])]->hy_tail.data() : nullptr;
  c->channel = h.channel.get();
  c->timeout = h.step_timeout;
  c->xcall.ids = ids;
  c->xcall.multimodal = spans != nullptr;
  if (spans != nullptr) c->xcall.a_spans = c->spans;
  c->xcall.call_id = c->call_id;
  c->ystate.call_id = c->call_id;
  h.channel->BeginCall(c->call_id);

  struct Disarm {  // an armed test fault fires once, in the call it was armed for
    Hybrid& hy;
    ~Disarm() {
      hy.fault_rank = -1;
      hy.fault_phase = HybridFaultPhase::kNone;
    }
  } disarm{h};
  TpModel* const self = this;
  RunGuarded(AllSlots(), [self, c](RankSlot& s) { self->HybridMain(s, *c); }, CmdKind::kPlain);

  // ---- bookkeeping: both shards and both stages agree through the call's end
  h.tracker->AfterPipelined(p0 + n);
  cached_position_ = c->pos_after[hybrid::kStageB];
  cached_fallback_rows_ = c->fallback_after[hybrid::kStageB];
  HybridStats& st = h.stats;
  ++st.pipelined_calls;
  st.last_p0 = p0;
  st.last_rows = n;
  const auto slower = [&c](const double (&v)[2]) { return std::max(v[0], v[1]); };
  st.last_gather_ms = slower(c->gather_ms);
  st.last_prefill_ms = slower(c->prefill_ms);
  st.last_reshard_ms = slower(c->reshard_ms);
  st.last_tail_ms = slower(c->tail_ms);
  st.last_total_ms = Ms(t_call0, Clock::now());
  const hybrid::PlanTotals tg = hybrid::Totals(c->gather), ts = hybrid::Totals(c->scatter);
  for (int s = 0; s < 2; ++s) {
    st.last_gather_bytes[s] = static_cast<int64_t>(tg.CrossFrom(s));
    st.last_reshard_bytes[s] = static_cast<int64_t>(ts.CrossFrom(s));
    st.gather_bytes[s] += tg.CrossFrom(s);
    st.reshard_bytes[s] += ts.CrossFrom(s);
  }
  st.local_bytes += tg.Local() + ts.Local();
  if (tail_use) st.tail_rows += static_cast<uint64_t>(tail_rows);
  if (c->logits.empty()) throw std::runtime_error("TpModel: the hybrid prefill produced no logits");
  return std::move(c->logits);
}

// ---- one rank thread's share of a pipelined call ---------------------------------------------------------------------------------------------

void TpModel::HybridMain(RankSlot& s, HyCall& c) {
  Hybrid& h = *hy_;
  const bool front = s.rank == 1;  // rank 1 = the desktop card = stage X = the PP stage A
  const int sidx = front ? hybrid::kStageA : hybrid::kStageB;
  Model& rank = *s.model;
  Model& stg = *s.stage;
  const auto t_call = Clock::now();
  // Test-only fault injection (ArmHybridFault): this rank's thread throws when it reaches the armed phase.
  const auto fault = [&h, &s](HybridFaultPhase phase) {
    if (h.fault_rank == s.rank && h.fault_phase == phase) throw std::runtime_error("injected hybrid fault (rank " + std::to_string(s.rank) + ")");
  };
  try {
    // 1. the scalars (pos, mrope) and, for stage Y, TP rank 0's MTP seed: the stages start from the shards' state. The stage Models run under
    //    the TP=1 GEMM tuning flag, the rank Models under their own (Tp2TuningScope per Model).
    {
      Tp2TuningScope scope(stg.Tp2TuningForModel());
      if (!front && h.neg_control == HybridNegControl::kStaleSeed) {  // test-only: stage Y keeps its own (stale) MTP seed
        StageSyncState stale = c.sync;
        const StageSyncState own = stg.StageGetSyncState();
        stale.mtp_seed_valid = own.mtp_seed_valid;
        stale.mtp_seed = own.mtp_seed;
        stg.StageSetSyncState(stale);
      } else {
        stg.StageSetSyncState(c.sync);
      }
    }
    rank.SetDflashInjectionEnabled(c.inject);

    // 2. the warm gather
    auto t = Clock::now();
    fault(HybridFaultPhase::kGather);
    if (!c.gather.ops.empty()) HybridExec(s, stg, rank, c, /*scatter=*/false);
    c.gather_ms[sidx] = Ms(t, Clock::now());

    // 3. the pipelined stage prefill
    t = Clock::now();
    fault(HybridFaultPhase::kStagePrefill);
    {
      Tp2TuningScope scope(stg.Tp2TuningForModel());
      if (front) {
        pp::RunStageA(stg, *c.channel, *h.bufs, h.run, c.xcall);  // poisons the channel and rethrows on failure
      } else {
        const int64_t end_pos = c.p0 + c.n;
        if (c.tail.use) stg.StageArmDflashTail(c.tail_host, hybrid::kTailCapacityRows, c.tail.start, end_pos);
        pp::StageBResult res = pp::RunStageB(stg, *c.channel, *h.bufs, h.run, c.ystate, c.ids, c.multimodal ? &c.spans : nullptr,
                                             c.multimodal && c.tail.use ? &c.rope_rows : nullptr);
        if (c.tail.use) {
          try {
            (void)stg.StageDisarmDflashTail();  // always: an armed capture holds the observer slot
          } catch (...) {
            if (!res.error) res.error = std::current_exception();
          }
        }
        if (res.error) std::rethrow_exception(res.error);  // RunStageB poisoned the channel
        c.logits = std::move(res.logits);
        c.end = stg.StageGetSyncState();
        // The tail's temporal rope row: the call's own rows while an image's mrope is active (PrefillMultimodal's block for a multimodal call,
        // `position + delta` for a plain one -- what RunChunk would have fed InjectFeatures); none otherwise (the drafter's delta shortcut).
        c.tail_rope.clear();
        if (c.tail.use) {
          c.tail_rope = hybrid::TailRopeRow(c.end.mrope_active, c.multimodal, c.rope_rows, c.n, c.p0, c.tail.start, c.tail.rows, c.end.mrope_delta);
        }
        c.end_gate.Open();  // the end state, the logits, the tail features and their rope row are published
      }
    }
    c.prefill_ms[sidx] = Ms(t, Clock::now());

    // 4. the bulk reshard: each stage's live state into both ranks
    t = Clock::now();
    fault(HybridFaultPhase::kReshard);
    HybridExec(s, stg, rank, c, /*scatter=*/true);
    c.reshard_ms[sidx] = Ms(t, Clock::now());

    // 5. the rank adopts the call's end state (stage Y's), then the drafter gets the tail
    c.end_gate.Wait(c.timeout);
    fault(HybridFaultPhase::kAdopt);
    {
      Tp2TuningScope scope(rank.Tp2TuningForModel());
      rank.TpAdoptPrefill(c.end);
      if (c.tail.use && h.neg_control != HybridNegControl::kSkipDflashTail) {
        t = Clock::now();
        rank.TpInjectDflashTail(c.tail_host, c.tail.rows, c.tail.start, c.tail_rope.empty() ? nullptr : c.tail_rope.data());
        c.tail_ms[sidx] = Ms(t, Clock::now());
      }
    }
    c.pos_after[sidx] = rank.PositionCount();
    c.fallback_after[sidx] = rank.SampledFallbackRows();
    c.total_ms[sidx] = Ms(t_call, Clock::now());
  } catch (...) {
    const std::exception_ptr e = std::current_exception();
    c.PoisonAll(pp::WhatOf(e));  // frees the other thread wherever it waits
    // Nothing this thread enqueued (a reshard copy into the ring, a stage chunk) may outlive the command: the recovery reuses the ring and
    // resets the Models, and the peer may read a slot only after a publish that will never come.
    try {
      stg.ReshardSync();
      rank.ReshardSync();
    } catch (...) {
    }
    if (hybrid::IsPoisonConsequence(e)) {
      // Only the other card's failure told this one: rank it last, so Run reports the root cause (docs/tp.md 2.4).
      throw core::TpAbortedError("hybrid: aborted by the other card's failure: " + pp::WhatOf(e));
    }
    throw;
  }
}

void TpModel::HybridExec(RankSlot& s, Model& stage, Model& rank, HyCall& c, bool scatter) {
  Hybrid& h = *hy_;
  const int sidx = s.rank == 1 ? hybrid::kStageA : hybrid::kStageB;
  const int peer = 1 - sidx;
  const hybrid::ReshardPlan& plan = scatter ? c.scatter : c.gather;
  const std::vector<hybrid::RingBatch>* const batches = scatter ? c.scatter_batches : c.gather_batches;
  RankSlot& peer_slot = *ranks_[static_cast<size_t>(h.slot_of_stage[peer])];
  uint8_t* const ring_out = s.hy_ring.data();
  const uint8_t* const ring_in = peer_slot.hy_ring.data();
  std::atomic<uint64_t>& beat = s.worker->Heartbeat();

  // The same-card halves: one D2D copy each, enqueued on the destination Model's stream (synced with the rest below).
  for (const hybrid::CopyOp& op : plan.ops) {
    if (!op.local || op.stage != sidx) continue;
    for (const hybrid::OpSlice& sl : hybrid::WholeOp(op)) Model::ReshardCopyLocal(stage, rank, op, sl, plan.dir);
  }

  // The halves that change cards, through the ring, in lockstep with the peer's thread. A scatter exports from this card's STAGE into the
  // other card's rank (kFull -> kRank), a gather from this card's RANK into the other card's stage; the imports are the peer's exports.
  Model& exp_model = scatter ? stage : rank;
  Model& imp_model = scatter ? rank : stage;
  const hybrid::Side exp_side = scatter ? hybrid::Side::kFull : hybrid::Side::kRank;
  const hybrid::Side imp_side = scatter ? hybrid::Side::kRank : hybrid::Side::kFull;
  const std::vector<hybrid::RingBatch>& out_b = batches[sidx];
  const std::vector<hybrid::RingBatch>& in_b = batches[peer];
  // The pipes live for the whole call and keep counting: the scatter's first batch is numbered after the gather's, so a slot is never refilled
  // while the peer may still be draining it from the previous phase (the threads have no barrier between the phases).
  const int64_t out_base = scatter ? static_cast<int64_t>(c.gather_batches[sidx].size()) : 0;
  const int64_t in_base = scatter ? static_cast<int64_t>(c.gather_batches[peer].size()) : 0;
  const auto slot_of = [](int64_t base, int64_t i) { return static_cast<uint64_t>((base + i) % hybrid::kHybridRingSlots) * hybrid::kHybridRingPieceBytes; };

  hybrid::LockstepCallbacks cb;
  cb.export_enqueue = [&](int64_t i) {
    uint8_t* p = ring_out + slot_of(out_base, i);
    for (const hybrid::SliceRef& ref : out_b[static_cast<size_t>(i)].slices) {
      const hybrid::CopyOp& op = plan.ops[ref.op];
      exp_model.ReshardExport(op, ref.slice, exp_side, p);
      p += hybrid::SliceBytes(op, ref.slice);
    }
  };
  cb.export_sync = [&](int64_t) { exp_model.ReshardSync(); };
  cb.import_enqueue = [&](int64_t i) {
    const uint8_t* p = ring_in + slot_of(in_base, i);
    for (const hybrid::SliceRef& ref : in_b[static_cast<size_t>(i)].slices) {
      const hybrid::CopyOp& op = plan.ops[ref.op];
      imp_model.ReshardImport(op, ref.slice, imp_side, p);
      p += hybrid::SliceBytes(op, ref.slice);
    }
  };
  cb.import_sync = [&](int64_t) { imp_model.ReshardSync(); };
  cb.on_step = [&beat] { beat.fetch_add(1, std::memory_order_relaxed); };
  hybrid::RunLockstep(c.pipe[sidx], c.pipe[peer], static_cast<int64_t>(out_b.size()), static_cast<int64_t>(in_b.size()), out_base, in_base, cb, c.timeout);
  stage.ReshardSync();  // the same-card copies, and anything else queued on either stream
  rank.ReshardSync();
}

// ---- diagnostics ---------------------------------------------------------------------------------------------------------------------------

TpModel::HybridStats TpModel::GetHybridStats() const { return hy_ ? hy_->stats : HybridStats{}; }
int64_t TpModel::HybridSplit() const { return hy_ ? hy_->split : 0; }
int64_t TpModel::HybridStageCtx() const { return hy_ ? hy_->stage_ctx : 0; }
int64_t TpModel::HybridMinRows() const { return hy_ ? hy_->min_rows : 0; }
void TpModel::SetHybridMinRows(int64_t min_rows) {
  if (min_rows < 1) throw std::invalid_argument("TpModel::SetHybridMinRows: min_rows must be >= 1");
  if (hy_) hy_->min_rows = min_rows;
}
void TpModel::ArmHybridFault(int rank, HybridFaultPhase phase) {
  RequireReady();
  if (!hy_) throw std::logic_error("TpModel::ArmHybridFault: the hybrid mode is not engaged");
  if (rank != 0 && rank != 1) throw std::invalid_argument("TpModel::ArmHybridFault: rank must be 0 or 1");
  hy_->fault_rank = phase == HybridFaultPhase::kNone ? -1 : rank;
  hy_->fault_phase = phase;
}
void TpModel::SetHybridNegControl(HybridNegControl nc) {
  RequireReady();
  if (!hy_) throw std::logic_error("TpModel::SetHybridNegControl: the hybrid mode is not engaged");
  hy_->neg_control = nc;
}

std::string TpModel::HybridStatsLine() {
  if (!hy_) return "";
  const Hybrid& h = *hy_;
  const HybridStats& s = h.stats;
  const pp::StageChannel::Stats cs = h.channel->GetStats();
  const auto why = [&s](hybrid::Why w) { return static_cast<long long>(s.declined[static_cast<int>(w)]); };
  char buf[1400];
  std::snprintf(buf, sizeof buf,
                "hybrid: split k=%lld, S=%lld tokens, min rows %lld; %lld pipelined / %lld TP-prefill calls (below min rows %lld, over S %lld, gather budget "
                "%lld, DFlash frontier %lld, not engaged %lld); last pipelined call %lld rows at position %lld: %.1f ms (gather %.1f, stage prefill %.1f, "
                "reshard %.1f, DFlash tail %.1f); host bytes X->Y / Y->X: gather %.1f / %.1f MiB (last %.1f / %.1f), reshard %.1f / %.1f MiB (last %.1f / %.1f), "
                "same-card %.1f MiB, tail rows %llu; stage X blocked %.0f ms on a full ring, stage Y %.0f ms on an empty one",
                static_cast<long long>(h.split), static_cast<long long>(h.stage_ctx), static_cast<long long>(h.min_rows),
                static_cast<long long>(s.pipelined_calls), static_cast<long long>(s.tp_prefill_calls), why(hybrid::Why::kBelowMinRows),
                why(hybrid::Why::kOverStageCtx), why(hybrid::Why::kGatherBudget), why(hybrid::Why::kDflashFrontier), why(hybrid::Why::kNotEngaged),
                static_cast<long long>(s.last_rows), static_cast<long long>(s.last_p0), s.last_total_ms, s.last_gather_ms, s.last_prefill_ms,
                s.last_reshard_ms, s.last_tail_ms, ToMib(s.gather_bytes[0]), ToMib(s.gather_bytes[1]), ToMib(static_cast<uint64_t>(s.last_gather_bytes[0])),
                ToMib(static_cast<uint64_t>(s.last_gather_bytes[1])), ToMib(s.reshard_bytes[0]), ToMib(s.reshard_bytes[1]),
                ToMib(static_cast<uint64_t>(s.last_reshard_bytes[0])), ToMib(static_cast<uint64_t>(s.last_reshard_bytes[1])), ToMib(s.local_bytes),
                static_cast<unsigned long long>(s.tail_rows), cs.producer_wait_ms, cs.consumer_wait_ms);
  return buf;
}

}  // namespace r4dx::model
