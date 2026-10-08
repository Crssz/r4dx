// tests/model/test_hybrid_budget_cpu.cpp -- CPU-only checks of src/model/hybrid_budget.h against the numbers of
// docs/pp-tp2-hybrid.md: the stage-KV capacity S of section 2 (X free after the TP ranks = 31.86 - 4.6 - 11.12 - rank KV, stage
// fixed 8.6, reserve 3.0 on X / 1.5 on Y, 16 KiB of stage KV per token) reproduces the table of S and of the free VRAM
// left on both cards at --max-ctx 32768 .. 262144; the min-rows model of section 5 reproduces the ~650-row break-even, the
// 512 / 768 / 1024 comparison and the 1024 default; the warm-gather budget reproduces "512 MiB = ~64k rows behind".
// No HIP call; always runs.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <utility>

#include "hybrid_budget.h"

using namespace r4dx::model::hybrid;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}
bool Near(double got, double want, double tol) { return std::fabs(got - want) <= tol; }

// The design's planning pieces (section 2), GiB.
constexpr double kTotal = 31.86, kDesktop = 4.6, kRankBuffers = 11.12, kVision = 0.92, kStageFixedX = 8.6, kStageFixedY = 7.4;
constexpr int64_t kHeadDim = 256, kKvHeads = 4, kRankKvHeads = 2, kStageAttn = 8, kRankAttn = 16;

void CapacityTable() {
  const int64_t bpt = StageKvBytesPerToken(kStageAttn, kKvHeads, kHeadDim);
  Check(bpt == 16384, "stage KV: 8 layers x 4 heads x 512 B = 16 KiB per token");
  Check(RankKvBytesPerToken(kRankAttn, kRankKvHeads, kHeadDim) == 16384, "rank KV: 16 layers x 2 heads x 512 B = 16 KiB per token");

  struct Row {
    int64_t max_ctx;
    double rank_kv_gib, free_after_ranks, x_free_after;
    int64_t s_lo, s_hi;  // the table's S (exact max_ctx, or "~100000" / "~35000")
  };
  // | --max-ctx | rank KV | free after ranks | S (auto) | X free after stage load |
  const Row rows[] = {
      {32768, 0.5, 15.6, 6.5, 32768, 32768},
      {65536, 1.0, 15.1, 5.5, 65536, 65536},
      {131072, 2.0, 14.1, 3.5, 131072, 131072},
      {196608, 3.0, 13.1, 3.0, 98000, 103000},   // ~100000, S-limited
      {262144, 4.0, 12.1, 3.0, 34000, 37000},    // ~35000, S-limited
  };
  for (const Row& r : rows) {
    const int64_t rank_kv = r.max_ctx * RankKvBytesPerToken(kRankAttn, kRankKvHeads, kHeadDim);
    Check(Near(ToGib(rank_kv), r.rank_kv_gib, 0.01), "rank KV column");
    const int64_t fx = FreeAfterRanksPlanning(kTotal, kDesktop, kRankBuffers, 0.0, rank_kv);
    const int64_t fy = FreeAfterRanksPlanning(kTotal, 0.0, kRankBuffers, kVision, rank_kv);
    Check(Near(ToGib(fx), r.free_after_ranks, 0.05), "X free after the ranks column");
    const StageCtxPlan p = PlanStageCtx(r.max_ctx, /*auto*/ 0, BudgetX(fx, Gib(kStageFixedX)), BudgetY(fy, Gib(kStageFixedY)), bpt);
    Check(p.engaged, "hybrid engages at every table row");
    Check(p.s >= r.s_lo && p.s <= r.s_hi, "S (auto) column");
    Check(Near(ToGib(p.x_free_after), r.x_free_after, 0.05), "X free after the stage load column");
    std::fprintf(stderr, "  max-ctx %6lld: free after ranks X %.2f  S_x %lld  S_y %lld  -> S %lld;  X free after %.2f GiB, Y free after %.2f GiB\n",
                 static_cast<long long>(r.max_ctx), ToGib(fx), static_cast<long long>(p.s_x), static_cast<long long>(p.s_y), static_cast<long long>(p.s),
                 ToGib(p.x_free_after), ToGib(p.y_free_after));
  }
  // "Y at the same settings is not tight": S = 131072 / max-ctx 131072 -> 8.4 free; 262144 / S = 262144 -> 4.4 free. (The table's Y rows
  // use S = max_ctx; with the real auto S at 262144, X limits it, so evaluate Y's free VRAM at the stated S.)
  for (const auto& y : {std::pair<int64_t, double>{131072, 8.4}, std::pair<int64_t, double>{262144, 4.4}}) {
    const int64_t rank_kv = y.first * RankKvBytesPerToken(kRankAttn, kRankKvHeads, kHeadDim);
    const CardBudget by = BudgetY(FreeAfterRanksPlanning(kTotal, 0.0, kRankBuffers, kVision, rank_kv), Gib(kStageFixedY));
    Check(Near(ToGib(FreeAfterStageLoad(by, y.first, bpt)), y.second, 0.05), "Y free after the stage load at S = max-ctx (8.4 / 4.4 GiB)");
  }
  // X is the tight card: at 131072 / 131072 its free VRAM is below Y's.
  {
    const int64_t rank_kv = 131072LL * 16384;
    const int64_t fx = FreeAfterRanksPlanning(kTotal, kDesktop, kRankBuffers, 0.0, rank_kv);
    const int64_t fy = FreeAfterRanksPlanning(kTotal, 0.0, kRankBuffers, kVision, rank_kv);
    Check(FreeAfterStageLoad(BudgetX(fx, Gib(kStageFixedX)), 131072, bpt) < FreeAfterStageLoad(BudgetY(fy, Gib(kStageFixedY)), 131072, bpt),
          "X (the desktop card) is the tight card");
  }
  // "with the computed 7.9 instead of the adopted 8.6": 4.2 free at 131072
  {
    const int64_t rank_kv = 131072LL * 16384;
    const int64_t fx = FreeAfterRanksPlanning(kTotal, kDesktop, kRankBuffers, 0.0, rank_kv);
    Check(Near(ToGib(FreeAfterStageLoad(BudgetX(fx, Gib(7.9)), 131072, bpt)), 4.2, 0.05), "with the computed X stage figure 7.9: 4.2 GiB free at 131072");
  }
}

void Refusals() {
  const int64_t bpt = 16384;
  // Hybrid is refused when S < 16384.
  const int64_t rank_kv = 262144LL * 16384;
  const CardBudget x = BudgetX(FreeAfterRanksPlanning(kTotal, kDesktop, kRankBuffers, 0.0, rank_kv), Gib(kStageFixedX));
  const CardBudget y = BudgetY(FreeAfterRanksPlanning(kTotal, 0.0, kRankBuffers, kVision, rank_kv), Gib(kStageFixedY));
  Check(PlanStageCtx(262144, 0, x, y, bpt).engaged, "262144: S ~ 35000 >= 16384, hybrid engages for short prompts");
  const CardBudget tight = BudgetX(Gib(8.6 + 3.0 + 0.2), Gib(8.6));  // 0.2 GiB for stage KV = 13107 tokens
  const StageCtxPlan p = PlanStageCtx(262144, 0, tight, y, bpt);
  Check(!p.engaged && p.refusal.find("16384") != std::string::npos && p.s > 13000 && p.s < 13200, "S < 16384 (~13100 here) refuses hybrid with a reason");
  const CardBudget none = BudgetX(Gib(10.0), Gib(8.6));  // reserve alone exceeds what is left
  Check(StageCapacityTokens(none, bpt) == 0 && !PlanStageCtx(131072, 0, none, y, bpt).engaged, "nothing left after the reserve: capacity 0, refused");
  // an explicit --hybrid-ctx
  const int64_t rank_kv2 = 131072LL * 16384;
  const CardBudget x2 = BudgetX(FreeAfterRanksPlanning(kTotal, kDesktop, kRankBuffers, 0.0, rank_kv2), Gib(kStageFixedX));
  const CardBudget y2 = BudgetY(FreeAfterRanksPlanning(kTotal, 0.0, kRankBuffers, kVision, rank_kv2), Gib(kStageFixedY));
  StageCtxPlan e = PlanStageCtx(131072, 65536, x2, y2, bpt);
  Check(e.engaged && e.s == 65536, "--hybrid-ctx 65536 within capacity: S = 65536");
  e = PlanStageCtx(262144, 500000, x2, y2, bpt);  // clamped to max-ctx 262144, which card X (S_x ~166k here) cannot hold
  Check(!e.engaged && !e.refusal.empty() && e.refusal.find("does not fit") != std::string::npos && e.refusal.find("card X") != std::string::npos,
        "--hybrid-ctx beyond the tight card's capacity is refused and names the card");
  e = PlanStageCtx(40000, 100000, x2, y2, bpt);
  Check(e.engaged && e.s == 40000, "--hybrid-ctx above --max-ctx is clamped to --max-ctx");
  // the reserves are overridable (--hybrid-reserve-gib)
  const CardBudget x3 = BudgetX(x2.free_after_ranks, x2.stage_fixed, /*reserve_gib=*/1.0);
  Check(StageCapacityTokens(x3, bpt) > StageCapacityTokens(x2, bpt), "a smaller reserve buys more stage KV");
  Check(Near(kReserveXGib, 3.0, 0) && Near(kReserveYGib, 1.5, 0) && kMinStageCtx == 16384, "the design's reserves and the minimum S");
}

void MinRows() {
  const MinRowsModel m;
  Check(Near(m.stage_chunk_ms, 51.78, 0.01) && Near(m.tp_chunk_ms, 90.0, 0.001), "1.657 s / 32 and 2.88 s / 32");
  const double c = BreakEvenChunks(m);
  Check(c > 2.5 && c < 2.6, "break-even at ~2.55 chunks (38c = 97)");
  const int64_t be = BreakEvenRows(m);
  Check(be >= 640 && be <= 660, "break-even at ~650 rows");
  // "at 512 rows (2 chunks) PP ~175-200 ms vs TP2 180 ms (no gain), at 768 about equal, clear gain from 1024 (~305 vs 360 ms)"
  Check(PipelinedMs(512, m) >= 175 && PipelinedMs(512, m) <= 201 && Near(TpPrefillMs(512, m), 180.0, 0.01), "512 rows: pipeline 175-200 ms vs TP 180 ms");
  Check(PipelinedMs(512, m) > TpPrefillMs(512, m) * 0.95, "... no gain");
  Check(Near(PipelinedMs(768, m), TpPrefillMs(768, m), 25.0), "768 rows: about equal");
  Check(Near(PipelinedMs(1024, m), 305.0, 3.0) && Near(TpPrefillMs(1024, m), 360.0, 0.01), "1024 rows: ~305 vs 360 ms");
  Check(MinRowsForGain(m, 1.0) == 768, "the first whole-chunk row count where the pipeline wins at all: 768");
  Check(MinRowsForGain(m, 0.9) == 1024, "the first with a clear (10 %) win: 1024 = the hybrid default");
  Check(DefaultMinRows(true) == 1024 && DefaultMinRows(false) == 512 && kHybridMinRowsDefault == MinRowsForGain(m, 0.9),
        "--pp-min-rows defaults: 1024 in hybrid mode, 512 for plain --pp 2; the hybrid default follows from the model");
  // a pipeline that never wins
  MinRowsModel slow = m;
  slow.stage_chunk_ms = 95.0;
  Check(!std::isfinite(BreakEvenChunks(slow)) && BreakEvenRows(slow) == INT64_MAX && MinRowsForGain(slow, 1.0) == INT64_MAX,
        "a stage slower than a TP chunk never breaks even");
  // monotone: a longer prompt never favours TP more
  bool mono = true;
  double prev = -1e9;
  for (int64_t rows = 256; rows <= 65536; rows += 256) {
    const double adv = TpPrefillMs(rows, m) - PipelinedMs(rows, m);
    if (adv < prev) mono = false;
    prev = adv;
  }
  Check(mono, "the pipeline's advantage grows with the prompt length");
  Check(ChunksOf(1, 256) == 1 && ChunksOf(256, 256) == 1 && ChunksOf(257, 256) == 2, "chunk count");
}

void GatherBudget() {
  const int64_t per_row = GatherBytesPerRow(kStageAttn, kRankKvHeads, kHeadDim);
  Check(per_row == 8192, "8 KiB of remote KV per stale row per direction (8 layers x 2 heads x 512 B)");
  Check(GatherWithinBudget(65536, per_row) && !GatherWithinBudget(65537, per_row), "512 MiB = exactly 64k rows behind");
  Check(kGatherBudgetBytes == 512 * kMiB, "512 MiB");
}

}  // namespace

int main() {
  CapacityTable();
  Refusals();
  MinRows();
  GatherBudget();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_hybrid_budget_cpu: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_hybrid_budget_cpu: PASS\n");
  return 0;
}
