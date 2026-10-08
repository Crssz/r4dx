// HIP-free budget arithmetic of the hybrid serving mode (docs/pp-tp2-hybrid.md 2 and 5): how many tokens of stage KV each
// card can afford once the TP ranks are loaded (the stage-KV capacity S, `--hybrid-ctx`), the free VRAM that leaves, the
// rows threshold below which the TP prefill beats the pipeline (`--pp-min-rows` in hybrid mode) and the warm-gather budget.
// Header-only so a CPU test (tests/model/test_hybrid_budget_cpu.cpp) reproduces the design's tables.
//
// Method (design 2): S comes from the MEASURED free VRAM after the TP ranks loaded -- that number already contains the
// desktop and every other tenant of the card -- not from totals. The planning pieces (FreeAfterRanksPlanning) exist to
// reproduce the design's table before any measurement, and are replaced by the load log's numbers.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>

namespace r4dx::model::hybrid {

constexpr int64_t kGiB = int64_t{1} << 30;
constexpr int64_t kMiB = int64_t{1} << 20;
inline int64_t Gib(double g) { return static_cast<int64_t>(std::llround(g * static_cast<double>(kGiB))); }
inline double ToGib(int64_t bytes) { return static_cast<double>(bytes) / static_cast<double>(kGiB); }

// Reserves kept free after the stage load (design 2): X is the desktop card (the desktop grows with open windows, allocator
// slack, lazily allocated arenas); for scale TP=1 production leaves 3.5 GiB free on the headless card.
constexpr double kReserveXGib = 3.0;
constexpr double kReserveYGib = 1.5;
// Hybrid is refused (plain --tp 2, one log line) below this stage-KV capacity.
constexpr int64_t kMinStageCtx = 16384;

// Stage KV bytes per token on one card: the stage's attention layers x ALL kv heads x (K + V fp8 rows). 8 layers x 4 heads x
// 512 B = 16 KiB for both k = 32 and k = 35 (layer k itself gets no cache).
inline int64_t StageKvBytesPerToken(int64_t stage_attn_layers, int64_t kv_heads_full, int64_t head_dim) {
  return stage_attn_layers * kv_heads_full * 2 * head_dim;
}
// The TP rank's KV per token: all 16 layers x its 2 heads, sized by --max-ctx.
inline int64_t RankKvBytesPerToken(int64_t attn_layers, int64_t kv_heads_rank, int64_t head_dim) {
  return attn_layers * kv_heads_rank * 2 * head_dim;
}

struct CardBudget {
  int64_t free_after_ranks = 0;  // hipMemGetInfo free, measured after the TP ranks loaded (desktop included)
  int64_t stage_fixed = 0;       // the stage's weights + activations + arena + GDN state (everything but its KV)
  int64_t reserve = 0;           // kept free after the stage load
};
inline CardBudget BudgetX(int64_t free_after_ranks, int64_t stage_fixed, double reserve_gib = kReserveXGib) {
  return {free_after_ranks, stage_fixed, Gib(reserve_gib)};
}
inline CardBudget BudgetY(int64_t free_after_ranks, int64_t stage_fixed, double reserve_gib = kReserveYGib) {
  return {free_after_ranks, stage_fixed, Gib(reserve_gib)};
}

// Planning stand-in for the measured number: card total minus the desktop (X only), the rank's process buffers, the vision
// tower (Y only) and the rank's KV.
inline int64_t FreeAfterRanksPlanning(double total_gib, double desktop_gib, double rank_buffers_gib, double vision_gib,
                                      int64_t rank_kv_bytes) {
  return Gib(total_gib - desktop_gib - rank_buffers_gib - vision_gib) - rank_kv_bytes;
}

// S = floor((free_after_ranks - stage_fixed - reserve) / bytes_per_token), 0 when nothing is left. A stage with no attention layer
// has 0 KV bytes per token: its KV never limits S (kUnboundedTokens) as long as its fixed part fits.
constexpr int64_t kUnboundedTokens = INT64_MAX;
inline int64_t StageCapacityTokens(const CardBudget& b, int64_t bytes_per_token) {
  const int64_t avail = b.free_after_ranks - b.stage_fixed - b.reserve;
  if (avail <= 0 || bytes_per_token < 0) return 0;
  return bytes_per_token == 0 ? kUnboundedTokens : avail / bytes_per_token;
}
inline int64_t FreeAfterStageLoad(const CardBudget& b, int64_t stage_ctx, int64_t bytes_per_token) {
  return b.free_after_ranks - b.stage_fixed - stage_ctx * bytes_per_token;
}

struct StageCtxPlan {
  int64_t s_x = 0, s_y = 0;  // each card's capacity (uncapped by max_ctx)
  int64_t s = 0;             // the stage-KV context actually allocated
  bool engaged = false;      // hybrid on; else `refusal` says why plain --tp 2 runs
  std::string refusal;
  int64_t x_free_after = 0, y_free_after = 0;  // free VRAM left after the stage load at `s`
};
// `requested`: --hybrid-ctx N (N > 0) or auto (<= 0). Auto: S = min(max_ctx, S_x, S_y). Explicit: min(N, max_ctx), refused when
// a card cannot hold it. Either way refused when S < kMinStageCtx.
inline StageCtxPlan PlanStageCtx(int64_t max_ctx, int64_t requested, const CardBudget& x, const CardBudget& y,
                                 int64_t bytes_per_token) {
  StageCtxPlan p;
  p.s_x = StageCapacityTokens(x, bytes_per_token);
  p.s_y = StageCapacityTokens(y, bytes_per_token);
  const int64_t cap = std::min(p.s_x, p.s_y);
  if (requested > 0) {
    p.s = std::min(requested, max_ctx);
    if (p.s > cap) {
      p.refusal = "--hybrid-ctx " + std::to_string(requested) + " does not fit: card " + (p.s_x <= p.s_y ? "X" : "Y") +
                  " holds at most " + std::to_string(cap) + " tokens of stage KV after its reserve";
      p.s = cap;
    }
  } else {
    p.s = std::min(max_ctx, cap);
  }
  if (p.refusal.empty() && p.s < kMinStageCtx) {
    p.refusal = "stage KV capacity " + std::to_string(p.s) + " tokens < " + std::to_string(kMinStageCtx);
  }
  p.engaged = p.refusal.empty();
  p.x_free_after = FreeAfterStageLoad(x, p.s, bytes_per_token);
  p.y_free_after = FreeAfterStageLoad(y, p.s, bytes_per_token);
  return p;
}

// ---- the stage-KV capacity S as a load argument (design 9 P0, "stage-KV max_ctx = S check") ---------------------------------
// The stage Models are loaded with ModelOptions::max_ctx = S, so S has to be something a PagedKvCache can honour exactly and
// something the ranks' own caches cover: whole blocks (the cache rounds UP to a block, which would let a prompt past S be
// accepted by the stage while the budget never paid for it), at least kMinStageCtx, at most the ranks' --max-ctx (a prompt
// longer than the rank cache cannot be decoded anyway). AlignStageCtx rounds a planned S down to a block; CheckStageCtx is the
// load-time check ("" = fine); Model::CheckStageKv is the device-side half (the caches really have that capacity, and no layer the
// stage does not run has one). A prompt whose total context exceeds S takes the ordinary TP prefill (StageFits).
inline int64_t AlignStageCtx(int64_t s, int64_t kv_block) { return kv_block > 0 && s > 0 ? s / kv_block * kv_block : 0; }
inline std::string CheckStageCtx(int64_t s, int64_t rank_max_ctx, int64_t kv_block) {
  if (kv_block <= 0) return "the KV block size must be positive";
  if (s < kMinStageCtx) return "stage KV capacity " + std::to_string(s) + " tokens < " + std::to_string(kMinStageCtx);
  if (s % kv_block != 0) {
    return "stage KV capacity " + std::to_string(s) + " is not a whole number of " + std::to_string(kv_block) + "-token blocks";
  }
  if (s > rank_max_ctx) {
    return "stage KV capacity " + std::to_string(s) + " exceeds the TP ranks' --max-ctx " + std::to_string(rank_max_ctx);
  }
  return "";
}
inline bool StageFits(int64_t n_total, int64_t s) { return n_total >= 0 && n_total <= s; }

// ---- the min-rows model (design 5, "Break-even") ----------------------------------------------------------------------
// A pipelined call of c chunks costs (c + 1) stage-chunk times (fill and drain) plus a fixed tail (warm gather, reshard,
// DFlash window injection); the TP prefill costs c TP-chunk times. Provisional until P-1 re-measures the PP figures with the
// corrected placement (8k: 1.657 s / 32 chunks stage, 2.88 s / 32 TP).
struct MinRowsModel {
  double stage_chunk_ms = 1657.0 / 32.0;
  double tp_chunk_ms = 2880.0 / 32.0;
  double tail_ms = 45.0;
  int64_t chunk_rows = 256;
};
inline int64_t ChunksOf(int64_t rows, int64_t chunk_rows) { return (rows + chunk_rows - 1) / chunk_rows; }
inline double PipelinedMs(int64_t rows, const MinRowsModel& m) {
  return static_cast<double>(ChunksOf(rows, m.chunk_rows) + 1) * m.stage_chunk_ms + m.tail_ms;
}
inline double TpPrefillMs(int64_t rows, const MinRowsModel& m) {
  return static_cast<double>(ChunksOf(rows, m.chunk_rows)) * m.tp_chunk_ms;
}
// The chunk count where the two are equal: (stage + tail) / (tp - stage); infinite when the pipeline never wins.
inline double BreakEvenChunks(const MinRowsModel& m) {
  return m.tp_chunk_ms > m.stage_chunk_ms ? (m.stage_chunk_ms + m.tail_ms) / (m.tp_chunk_ms - m.stage_chunk_ms) : INFINITY;
}
inline int64_t BreakEvenRows(const MinRowsModel& m) {
  const double c = BreakEvenChunks(m);
  return std::isfinite(c) ? static_cast<int64_t>(std::ceil(c * static_cast<double>(m.chunk_rows))) : INT64_MAX;
}
// The smallest whole-chunk row count at which the pipeline is at least `gain` times as fast as TP (gain 1.0 = break-even,
// 0.9 = a clear win); INT64_MAX when never.
inline int64_t MinRowsForGain(const MinRowsModel& m, double gain) {
  for (int64_t c = 1; c <= 4096; ++c) {
    const int64_t rows = c * m.chunk_rows;
    if (PipelinedMs(rows, m) <= gain * TpPrefillMs(rows, m)) return rows;
  }
  return INT64_MAX;
}
// `--pp-min-rows` defaults: 512 for plain --pp 2 (its baseline is TP=1), 1024 in hybrid mode (its baseline is TP=2 prefill,
// 90 ms per chunk against the pipeline's 52).
constexpr int64_t kPpMinRowsDefault = 512;
constexpr int64_t kHybridMinRowsDefault = 1024;
inline int64_t DefaultMinRows(bool hybrid) { return hybrid ? kHybridMinRowsDefault : kPpMinRowsDefault; }

// ---- the warm-gather budget (design 5) ---------------------------------------------------------------------------------
// Stale rows behind a warm call cost, per direction, the REMOTE half of the stage's KV: layers x remote heads x 512 B per row
// (8 KiB for 8 layers x 2 heads). Past kGatherBudgetBytes (~64k rows behind) the TP prefill path runs instead.
constexpr int64_t kGatherBudgetBytes = 512 * kMiB;
inline int64_t GatherBytesPerRow(int64_t stage_attn_layers, int64_t remote_kv_heads, int64_t head_dim) {
  return stage_attn_layers * remote_kv_heads * 2 * head_dim;
}
inline bool GatherWithinBudget(int64_t stale_rows, int64_t bytes_per_row) {
  return stale_rows * bytes_per_row <= kGatherBudgetBytes;
}

}  // namespace r4dx::model::hybrid
