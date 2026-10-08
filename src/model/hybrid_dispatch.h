// HIP-free policy of TpModel's hybrid mode (docs/pp-tp2-hybrid.md 2, 5, 7, 11; P3 step 2): the kill switch, the stage-VRAM planning
// that turns the MEASURED free VRAM after the TP ranks into the stage-KV capacity S (per card, because the two stages hold different
// numbers of attention layers), and the dispatch rule of Prefill / PrefillMultimodal -- pipelined (stage X || stage Y + reshard) or the
// ordinary TP=2 prefill -- with every fallback named. Header-only so tests/model/test_hybrid_dispatch_cpu.cpp reproduces the design's
// tables and walks the rule, no device.
//
// The rule (Decide), first match wins; everything but kHybrid means "the ordinary TP prefill, unchanged":
//   kNotEngaged     the hybrid is not loaded (kill switch, refused at load, torn down after an error at load)
//   kBelowMinRows   rows < min_rows (hybrid default 1024: below it the pipeline's fill + drain and the fixed tail do not pay)
//   kOverStageCtx   p0 + rows > S: the stage caches cannot hold the conversation
//   kGatherBudget   a warm call whose stale rows (decode / TP-prefill rows the stages have not seen) would move more than 512 MiB
//                   per direction through the host before the pipeline could start
//   kDflashFrontier a drafter whose frontier the tail injection cannot continue from (hybrid::CheckTailInject): the TP prefill
//                   feeds every row and has no such constraint
#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "dflash_tail_plan.h"
#include "hybrid_budget.h"
#include "hybrid_sync.h"
#include "pp_sync.h"

namespace r4dx::model::hybrid {

// ---- the kill switch ----------------------------------------------------------------------------------------------------------
// `--hybrid {on|off}` / R4DX_HYBRID: unset, empty, "1", "on", "true", "auto" = on; "0", "off", "false" = off; anything else = -1
// (the caller refuses it). Whether the hybrid is wanted AT ALL is `--tp 2 --pp 2`; this only lets a user turn it back off.
inline int ParseHybridSwitch(const char* e) {
  if (e == nullptr || *e == '\0') return 1;
  const std::string s = e;
  if (s == "1" || s == "on" || s == "true" || s == "auto") return 1;
  if (s == "0" || s == "off" || s == "false") return 0;
  return -1;
}

// ---- the pinned ring ----------------------------------------------------------------------------------------------------------
// A ring piece is the unit that crosses the host in one go; a smaller piece than the design's 128 MiB lets the D2H of one batch
// overlap the H2D of the previous one even for a short prompt (8k crosses ~105 MiB per direction cold: 4 batches, not 1).
constexpr uint64_t kHybridRingPieceBytes = 32ull << 20;
constexpr int kHybridRingSlots = 3;
constexpr uint64_t kHybridRingBytes = kHybridRingPieceBytes * kHybridRingSlots;  // per card

// ---- stage VRAM planning ------------------------------------------------------------------------------------------------------
// Planning figures of docs/pp-tp2-hybrid.md 2 (mix4.5m trellis; P-1 / P0 replace them with the load log's numbers): one body layer's
// weights, and what a stage holds besides its layers. The check after the stage load (TpModel) compares the plan with the measured
// free VRAM, so a figure that is too low is caught, not trusted.
constexpr double kPlanLayerGib = 0.2033;
constexpr double kPlanXExtraGib = 1.5;       // activations + arena + GDN state + the int8 prefill tables' share + slack (design: X total 8.6 at k = 35)
constexpr double kPlanYExtraGib = 0.9;       // ... the same pieces for stage Y (design 0.55 + a 0.35 share of the int8 tables)
constexpr double kPlanLmHeadGib = 0.67;      // stage Y's FULL lm_head
constexpr double kPlanMtpHeadGib = 0.31;     // stage Y's FULL MTP head (when the ranks have MTP)
constexpr double kHardFreeFloorGib = 1.0;    // measured free VRAM after the stage load below which the hybrid is torn down again

// Body-layer weight size relative to the trellis container (bf16 is ~3.5x; w4a16 and trellis are within a few percent).
inline double LayerWeightScale(bool bf16) { return bf16 ? 3.5 : 1.0; }

inline int64_t StageFixedX(int64_t split, double weight_scale) { return Gib(static_cast<double>(split) * kPlanLayerGib * weight_scale + kPlanXExtraGib); }
inline int64_t StageFixedY(int64_t layers, int64_t split, bool mtp, double weight_scale) {
  return Gib(static_cast<double>(layers - split) * kPlanLayerGib * weight_scale + kPlanYExtraGib + kPlanLmHeadGib + (mtp ? kPlanMtpHeadGib : 0.0));
}

// PlanStageCtx (hybrid_budget.h) with ONE bytes-per-token for both cards; the two stages hold different numbers of attention layers
// (k = 29: X 7, Y 9), so each card gets its own.
inline StageCtxPlan PlanStageCtxPerCard(int64_t max_ctx, int64_t requested, const CardBudget& x, const CardBudget& y, int64_t bytes_per_token_x,
                                        int64_t bytes_per_token_y) {
  StageCtxPlan p;
  p.s_x = StageCapacityTokens(x, bytes_per_token_x);
  p.s_y = StageCapacityTokens(y, bytes_per_token_y);
  const int64_t cap = std::min(p.s_x, p.s_y);
  if (requested > 0) {
    p.s = std::min(requested, max_ctx);
    if (p.s > cap) {
      p.refusal = "--hybrid-ctx " + std::to_string(requested) + " does not fit: card " + (p.s_x <= p.s_y ? "X" : "Y") + " holds at most " +
                  std::to_string(cap) + " tokens of stage KV after its reserve";
      p.s = cap;
    }
  } else {
    p.s = std::min(max_ctx, cap);
  }
  if (p.refusal.empty() && p.s < kMinStageCtx) {
    p.refusal = "stage KV capacity " + std::to_string(p.s) + " tokens < " + std::to_string(kMinStageCtx);
  }
  p.engaged = p.refusal.empty();
  p.x_free_after = FreeAfterStageLoad(x, p.s, bytes_per_token_x);
  p.y_free_after = FreeAfterStageLoad(y, p.s, bytes_per_token_y);
  return p;
}

// ---- the DFlash tail of one call ------------------------------------------------------------------------------------------------
struct TailPlan {
  bool use = false;      // the call injects a tail (a drafter, injection on)
  int64_t start = 0;     // first position of the tail (on the call's 64-row grid)
  int64_t rows = 0;      // tail rows (<= window + slice - 1)
  std::string problem;   // non-empty: the tail injection would be refused (CheckTailInject): take the TP prefill
};
// `injected`: the drafter's frontier on both ranks; the call covers positions [p0, n_end).
inline TailPlan PlanTail(bool drafter, bool injection_on, int64_t p0, int64_t n_end, int64_t injected) {
  TailPlan t;
  if (!drafter || !injection_on) return t;  // no drafter / injection off: the TP prefill feeds the ring nothing either
  t.use = true;
  t.start = DflashTailStart(p0, n_end);
  t.rows = n_end - t.start;
  t.problem = CheckTailInject(injected, n_end, t.start, t.rows);
  return t;
}
// The tail capture buffer's capacity in rows: a tail never exceeds the window plus the part of the last slice before it.
constexpr int64_t kTailCapacityRows = kDflashWindowRows + kDflashSliceRows;
// The temporal rope row of the tail's positions [tail_start, tail_start + tail_rows) of a call over [p0, p0 + n), as Model::RunChunk would
// have fed InjectFeatures: while mrope is active, the call's own block row for a multimodal call (`rope3` = PrefillMultimodal's
// rope_rows_out, [3, n], row 0 = temporal) and `position + delta` for a plain one (RopePositionsHost's rule outside a block); empty
// when mrope is not active (the drafter's delta shortcut, nullptr to InjectFeatures). Throws what TemporalRopeRows throws.
inline std::vector<int32_t> TailRopeRow(bool mrope_active, bool multimodal, const std::vector<int32_t>& rope3, int64_t n, int64_t p0,
                                        int64_t tail_start, int64_t tail_rows, int64_t mrope_delta) {
  std::vector<int32_t> out;
  if (!mrope_active) return out;
  if (multimodal) return TemporalRopeRows(rope3, n, tail_start - p0, tail_rows);
  out.resize(static_cast<size_t>(tail_rows));
  for (int64_t i = 0; i < tail_rows; ++i) out[static_cast<size_t>(i)] = static_cast<int32_t>(tail_start + i + mrope_delta);
  return out;
}

// ---- the dispatch rule --------------------------------------------------------------------------------------------------------------
enum class Route { kHybrid, kTpPrefill };
enum class Why { kHybrid, kNotEngaged, kBelowMinRows, kOverStageCtx, kGatherBudget, kDflashFrontier };
constexpr int kWhyCount = 6;
inline const char* WhyName(Why w) {
  switch (w) {
    case Why::kHybrid: return "pipelined";
    case Why::kNotEngaged: return "not engaged";
    case Why::kBelowMinRows: return "below min rows";
    case Why::kOverStageCtx: return "over the stage-KV capacity";
    case Why::kGatherBudget: return "warm-gather budget";
    case Why::kDflashFrontier: return "DFlash frontier";
  }
  return "?";
}

struct DispatchIn {
  bool engaged = false;
  int64_t rows = 0;          // rows of this call
  int64_t p0 = 0;            // the conversation's position now
  int64_t min_rows = kHybridMinRowsDefault;
  int64_t stage_ctx = 0;     // S
  // What the warm gather would move (TpMasterTracker::PlanSync): stale KV rows per stage, and the remote half's bytes per row of that stage.
  int64_t stale_rows[2] = {0, 0};
  int64_t gather_bytes_per_row[2] = {0, 0};
  // The DFlash tail's problem (PlanTail), "" when fine or not asked yet.
  std::string dflash_problem;
};
struct Dispatch {
  Route route = Route::kTpPrefill;
  Why why = Why::kNotEngaged;
};
inline Dispatch Decide(const DispatchIn& in) {
  Dispatch d;
  const auto tp = [&](Why w) {
    d.route = Route::kTpPrefill;
    d.why = w;
    return d;
  };
  if (!in.engaged) return tp(Why::kNotEngaged);
  if (!pp::ShouldPipeline(in.rows, in.min_rows)) return tp(Why::kBelowMinRows);
  if (!StageFits(in.p0 + in.rows, in.stage_ctx)) return tp(Why::kOverStageCtx);
  for (int s = 0; s < 2; ++s) {
    if (in.stale_rows[s] > 0 && !GatherWithinBudget(in.stale_rows[s], in.gather_bytes_per_row[s])) return tp(Why::kGatherBudget);
  }
  if (!in.dflash_problem.empty()) return tp(Why::kDflashFrontier);
  d.route = Route::kHybrid;
  d.why = Why::kHybrid;
  return d;
}

// Stale rows per stage of a TpMasterTracker::SyncPlan.
inline void StaleRows(const TpMasterTracker::SyncPlan& plan, int64_t out[2]) {
  for (int s = 0; s < 2; ++s) out[s] = std::max<int64_t>(0, plan.stage[s].kv_row1 - plan.stage[s].kv_row0);
}

}  // namespace r4dx::model::hybrid
