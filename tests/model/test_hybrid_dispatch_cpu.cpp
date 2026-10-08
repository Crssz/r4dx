// tests/model/test_hybrid_dispatch_cpu.cpp -- CPU-only checks of src/model/hybrid_dispatch.h, the HIP-free policy of TpModel's hybrid
// mode (docs/pp-tp2-hybrid.md 2, 5, 7): the kill-switch parse, the stage-VRAM planning (the design's per-card figures and the per-card
// stage-KV capacity S), the DFlash tail plan of one call and the dispatch rule of Prefill / PrefillMultimodal with every fallback named
// -- including a randomized walk of the rule against an independently written oracle, driven through the real TpMasterTracker. No HIP
// call; always runs.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

#include "hybrid_dispatch.h"

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

void KillSwitch() {
  Check(ParseHybridSwitch(nullptr) == 1, "unset = on");
  Check(ParseHybridSwitch("") == 1, "empty = on");
  for (const char* on : {"1", "on", "true", "auto"}) Check(ParseHybridSwitch(on) == 1, "on spellings");
  for (const char* off : {"0", "off", "false"}) Check(ParseHybridSwitch(off) == 0, "off spellings");
  for (const char* bad : {"2", "yes", "OFF", "no", " 0"}) Check(ParseHybridSwitch(bad) == -1, "anything else is refused");
}

void RingConstants() {
  Check(kHybridRingBytes == 3 * (32ull << 20), "ring = 3 x 32 MiB per card");
  Check(kHybridRingPieceBytes >= (3ull << 19), "a piece holds the widest row of any op (the 1.5 MiB recurrent run)");
  Check(kTailCapacityRows == 2048 + 64, "tail capture room = window + one slice");
}

void StagePlanning() {
  // Design 2, k = 35 with a drafter: stage X total 8.6, stage Y 7.4 (the planning figures); Y here is a little more conservative (+0.35 for the
  // int8 tables' share).
  Check(Near(ToGib(StageFixedX(35, 1.0)), 8.6, 0.05), "stage X fixed at k = 35 reproduces the design's 8.6 GiB");
  Check(Near(ToGib(StageFixedY(64, 35, true, 1.0)), 7.78, 0.05), "stage Y fixed at k = 35 (+ MTP head) is the design's 7.4 plus the int8 share");
  Check(StageFixedY(64, 35, false, 1.0) < StageFixedY(64, 35, true, 1.0), "no MTP head, less on Y");
  Check(StageFixedX(35, 3.5) > StageFixedX(35, 1.0), "a bf16 body weighs more");

  // Equal per-token bytes: identical to hybrid_budget.h's PlanStageCtx at every table row.
  const int64_t bpt = StageKvBytesPerToken(8, 4, 256);
  for (const int64_t max_ctx : {32768, 65536, 131072, 196608, 262144}) {
    const int64_t rank_kv = max_ctx * RankKvBytesPerToken(16, 2, 256);
    const CardBudget x = BudgetX(FreeAfterRanksPlanning(31.86, 4.6, 11.12, 0.0, rank_kv), StageFixedX(35, 1.0));
    const CardBudget y = BudgetY(FreeAfterRanksPlanning(31.86, 0.0, 11.12, 0.92, rank_kv), StageFixedY(64, 35, true, 1.0));
    const StageCtxPlan a = PlanStageCtx(max_ctx, 0, x, y, bpt);
    const StageCtxPlan b = PlanStageCtxPerCard(max_ctx, 0, x, y, bpt, bpt);
    Check(a.s == b.s && a.engaged == b.engaged && a.s_x == b.s_x && a.s_y == b.s_y && a.x_free_after == b.x_free_after && a.y_free_after == b.y_free_after,
          "PlanStageCtxPerCard with equal bytes per token == PlanStageCtx");
  }

  // k = 29 (the measured default without a drafter): stage X holds 7 attention layers, stage Y 9 -> 14 vs 18 KiB per token. Each card gets its own.
  const int64_t bx = StageKvBytesPerToken(7, 4, 256), by = StageKvBytesPerToken(9, 4, 256);
  Check(bx == 14 * 1024 && by == 18 * 1024, "k = 29: 14 KiB per token on X, 18 KiB on Y");
  {
    const int64_t max_ctx = 131072;
    const int64_t rank_kv = max_ctx * RankKvBytesPerToken(16, 2, 256);
    const CardBudget x = BudgetX(FreeAfterRanksPlanning(31.86, 4.6, 9.63, 0.0, rank_kv), StageFixedX(29, 1.0));
    const CardBudget y = BudgetY(FreeAfterRanksPlanning(31.86, 0.0, 9.63, 0.92, rank_kv), StageFixedY(64, 29, false, 1.0));
    const StageCtxPlan p = PlanStageCtxPerCard(max_ctx, 0, x, y, bx, by);
    Check(p.engaged && p.s == max_ctx, "k = 29, no drafter, --max-ctx 131072: the stage holds the whole context");
    Check(p.x_free_after > Gib(kReserveXGib) && p.y_free_after > Gib(kReserveYGib), "... and both cards keep their reserve");
    // The same card budgets with ONE (the larger) per-token figure would be more pessimistic on X.
    const StageCtxPlan q = PlanStageCtxPerCard(max_ctx, 0, x, y, by, by);
    Check(p.s_x > q.s_x, "per-card bytes give the lighter stage the larger capacity");
  }
  {  // a tight X: capacity limited by X, explicit ask refused, tiny leftover refused
    const CardBudget x = BudgetX(Gib(12.0), Gib(8.6));  // 12 - 8.6 - 3.0 = 0.4 GiB -> ~26k tokens at 16 KiB
    const CardBudget y = BudgetY(Gib(20.0), Gib(7.8));
    const StageCtxPlan p = PlanStageCtxPerCard(262144, 0, x, y, 16384, 16384);
    Check(p.s_x < 30000 && p.s_x > 24000 && p.s == p.s_x && p.engaged, "auto: S = min(max_ctx, S_x, S_y), here X's ~26k");
    const StageCtxPlan e = PlanStageCtxPerCard(262144, 100000, x, y, 16384, 16384);
    Check(!e.engaged && !e.refusal.empty() && e.s == e.s_x, "--hybrid-ctx 100000 does not fit X: refused, with the capacity named");
    const StageCtxPlan t = PlanStageCtxPerCard(262144, 0, BudgetX(Gib(11.7), Gib(8.6)), y, 16384, 16384);
    Check(!t.engaged && t.refusal.find("16384") != std::string::npos, "S below kMinStageCtx is refused");
    const StageCtxPlan z = PlanStageCtxPerCard(262144, 0, BudgetX(Gib(5.0), Gib(8.6)), y, 16384, 16384);
    Check(!z.engaged && z.s == 0, "nothing left on X: S = 0, refused");
    const StageCtxPlan small_max = PlanStageCtxPerCard(8192, 0, BudgetX(Gib(20.0), Gib(8.6)), y, 16384, 16384);
    Check(!small_max.engaged, "--max-ctx below the minimum stage capacity: refused");
  }
}

void TailPlans() {
  Check(!PlanTail(false, true, 0, 3000, 0).use, "no drafter: no tail");
  Check(!PlanTail(true, false, 0, 3000, 0).use, "injection off (a sampled turn): the TP prefill feeds nothing, neither does the hybrid");
  {  // a long cold call: the tail is the last >= 2048 rows on the call's 64-row grid, the gap before it is fine (a whole window follows)
    const TailPlan t = PlanTail(true, true, 0, 3000, 0);
    Check(t.use && t.start == 896 && t.rows == 3000 - 896 && t.rows >= kDflashWindowRows && t.rows <= kTailCapacityRows && t.problem.empty(),
          "cold 3000 rows: tail starts at 896 (3000 - 2048 = 952, rounded down to the grid)");
  }
  {  // a short cold call: everything is the tail
    const TailPlan t = PlanTail(true, true, 0, 1100, 0);
    Check(t.use && t.start == 0 && t.rows == 1100 && t.problem.empty(), "cold 1100 rows: the whole call is the tail");
  }
  {  // a warm call: the grid is anchored at p0
    const TailPlan t = PlanTail(true, true, 700, 5000, 700);
    Check(t.start == 700 + (5000 - 2048 - 700) / 64 * 64 && (t.start - 700) % 64 == 0 && t.problem.empty(), "warm call: the tail start is on p0 + 64 j");
    Check(t.rows >= kDflashWindowRows && t.rows < kDflashWindowRows + kDflashSliceRows, "... and the tail is a window plus less than one slice");
  }
  Check(!PlanTail(true, true, 700, 1500, 500).problem.empty(), "frontier behind p0 with a short call: refused (the TP prefill feeds every row)");
  Check(PlanTail(true, true, 700, 5000, 500).problem.empty(), "frontier behind p0 with a long call: the gap is a whole window, fine");
  Check(!PlanTail(true, true, 700, 1500, 900).problem.empty(), "frontier past the tail start: injection is monotonic, refused");
  for (int64_t p0 : {0, 1, 63, 64, 1000, 4096}) {
    for (int64_t n : {1, 64, 1024, 2047, 2048, 2049, 4000, 9000}) {
      const TailPlan t = PlanTail(true, true, p0, p0 + n, p0);
      Check(t.start >= p0 && t.start + t.rows == p0 + n && t.rows <= kTailCapacityRows && t.rows >= 1 && (t.start - p0) % kDflashSliceRows == 0,
            "tail invariants: on the grid, ends at the call's end, fits the capture buffer");
      Check(t.problem.empty(), "frontier == p0 is always continuable");
    }
  }
}

// Model::RopePositionsHost's rule, restated: a position inside the call's block takes the block's row, any other `position + delta`.
std::vector<int32_t> RunChunkTemporalRow(const std::vector<int32_t>& block3, int64_t block_n, int64_t block_base, int64_t delta, int64_t start, int64_t count) {
  std::vector<int32_t> out(static_cast<size_t>(count));
  for (int64_t t = 0; t < count; ++t) {
    const int64_t s = start + t, in_block = s - block_base;
    out[static_cast<size_t>(t)] = block_n > 0 && in_block >= 0 && in_block < block_n ? block3[static_cast<size_t>(in_block)] : static_cast<int32_t>(s + delta);
  }
  return out;
}

void TailRope() {
  const int64_t p0 = 300, n = 2600;
  // an image conversation: the call's block [3, n] has an image in the middle (positions compress after it), delta from earlier turns
  std::vector<int32_t> rope3(static_cast<size_t>(3 * n));
  for (int64_t i = 0; i < n; ++i) {
    const int32_t t = static_cast<int32_t>(p0 + (i < 900 ? i : i < 1156 ? 900 + (i - 900) / 16 : i - 240) + 7);
    rope3[static_cast<size_t>(i)] = t;
    rope3[static_cast<size_t>(n + i)] = t + 1;
    rope3[static_cast<size_t>(2 * n + i)] = t + 2;
  }
  const TailPlan tail = PlanTail(true, true, p0, p0 + n, p0);
  const std::vector<int32_t> want = RunChunkTemporalRow(std::vector<int32_t>(rope3.begin(), rope3.begin() + n), n, p0, 123, tail.start, tail.rows);
  const std::vector<int32_t> got = TailRopeRow(true, true, rope3, n, p0, tail.start, tail.rows, 123);
  Check(got == want && static_cast<int64_t>(got.size()) == tail.rows, "multimodal call: the tail's temporal row is the block's row 0 over the tail");
  // a plain call on an image conversation: position + delta
  const std::vector<int32_t> plain = TailRopeRow(true, false, {}, n, p0, tail.start, tail.rows, -77);
  const std::vector<int32_t> want_plain = RunChunkTemporalRow({}, 0, 0, -77, tail.start, tail.rows);
  Check(plain == want_plain, "plain call while mrope is active: position + delta");
  Check(TailRopeRow(false, true, rope3, n, p0, tail.start, tail.rows, 5).empty() && TailRopeRow(false, false, {}, n, p0, tail.start, tail.rows, 5).empty(),
        "mrope not active: no row (the drafter's delta shortcut)");
  bool threw = false;
  try {
    (void)TailRopeRow(true, true, std::vector<int32_t>(7), n, p0, tail.start, tail.rows, 0);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  Check(threw, "a rope block that is not [3, n] is refused, not misread");
}

DispatchIn Base() {
  DispatchIn in;
  in.engaged = true;
  in.rows = 4096;
  in.p0 = 0;
  in.min_rows = 1024;
  in.stage_ctx = 131072;
  in.gather_bytes_per_row[0] = GatherBytesPerRow(7, 2, 256);
  in.gather_bytes_per_row[1] = GatherBytesPerRow(9, 2, 256);
  return in;
}

void Rule() {
  Check(GatherBytesPerRow(8, 2, 256) == 8192, "8 layers x 2 remote heads x 512 B = 8 KiB per row");
  {
    DispatchIn in = Base();
    const Dispatch d = Decide(in);
    Check(d.route == Route::kHybrid && d.why == Why::kHybrid, "a long cold prompt is pipelined");
  }
  {
    DispatchIn in = Base();
    in.engaged = false;
    Check(Decide(in).why == Why::kNotEngaged && Decide(in).route == Route::kTpPrefill, "not engaged (kill switch / refused at load): TP prefill");
  }
  {
    DispatchIn in = Base();
    in.rows = 1023;
    Check(Decide(in).why == Why::kBelowMinRows, "1023 rows < the hybrid default 1024: TP prefill");
    in.rows = 1024;
    Check(Decide(in).route == Route::kHybrid, "1024 rows pipelines");
    in.min_rows = 512;
    in.rows = 600;
    Check(Decide(in).route == Route::kHybrid, "--pp-min-rows 512 pipelines 600 rows");
    in.min_rows = 0;
    in.rows = 1;
    Check(Decide(in).route == Route::kHybrid, "min rows <= 1 pipelines every call (tests)");
    in.rows = 0;
    Check(Decide(in).route == Route::kTpPrefill, "an empty call never pipelines (the ranks reject it)");
  }
  {
    DispatchIn in = Base();
    in.rows = 4000;
    in.p0 = in.stage_ctx - 4000;
    Check(Decide(in).route == Route::kHybrid, "p0 + rows == S fits");
    in.p0 += 1;
    Check(Decide(in).why == Why::kOverStageCtx, "p0 + rows == S + 1: TP prefill (over the stage-KV capacity)");
    in.p0 = 0;
    in.rows = in.stage_ctx + 1;
    Check(Decide(in).why == Why::kOverStageCtx, "a prompt longer than S: TP prefill");
  }
  {
    DispatchIn in = Base();
    in.p0 = 70000;
    in.stage_ctx = 262144;
    in.stale_rows[0] = 65536;  // X: 65536 x 7 layers x 1 KiB = 448 MiB
    in.stale_rows[1] = 65536;  // Y: 65536 x 9 KiB = 576 MiB -> over
    Check(Decide(in).why == Why::kGatherBudget, "stage Y 65536 rows behind (576 MiB): over the gather budget");
    const int64_t y_rows_at_budget = kGatherBudgetBytes / in.gather_bytes_per_row[1];
    in.stale_rows[1] = y_rows_at_budget;
    Check(Decide(in).route == Route::kHybrid, "exactly at the budget: pipelined");
    in.stale_rows[1] = y_rows_at_budget + 1;
    Check(Decide(in).why == Why::kGatherBudget, "one row over: TP prefill");
    in.stale_rows[1] = 0;
    in.stale_rows[0] = kGatherBudgetBytes / in.gather_bytes_per_row[0] + 1;
    Check(Decide(in).why == Why::kGatherBudget, "stage X's own bytes per row count too");
  }
  {
    DispatchIn in = Base();
    in.dflash_problem = "start 5 is below the drafter's frontier 9";
    Check(Decide(in).why == Why::kDflashFrontier, "a frontier the tail cannot continue from: TP prefill");
  }
  {  // first match wins
    DispatchIn in = Base();
    in.engaged = false;
    in.rows = 10;
    in.dflash_problem = "x";
    Check(Decide(in).why == Why::kNotEngaged, "not engaged beats everything");
    in.engaged = true;
    Check(Decide(in).why == Why::kBelowMinRows, "min rows beats the rest");
    in.rows = 5000;
    in.p0 = in.stage_ctx;
    in.stale_rows[0] = 1 << 30;
    Check(Decide(in).why == Why::kOverStageCtx, "the stage capacity beats the gather budget");
    in.p0 = 0;
    Check(Decide(in).why == Why::kGatherBudget, "the gather budget beats the DFlash check");
  }
  for (int w = 0; w < kWhyCount; ++w) Check(std::string(WhyName(static_cast<Why>(w))) != "?", "every reason has a name");
}

// A randomized walk: the real tracker supplies the stale rows, an independently written oracle supplies the verdict.
void RandomWalk() {
  std::mt19937_64 rng(0xD15BA7C4ull);
  const int64_t block = 16;
  TpMasterTracker tracker(block, true);
  int64_t pos = 0;
  int hybrid_calls = 0, tp_calls = 0;
  int reasons[kWhyCount] = {};
  const int64_t S = 100000, min_rows = 1024;
  for (int step = 0; step < 20000; ++step) {
    const int op = static_cast<int>(rng() % 10);
    if (op < 4) {  // a prefill call
      const int64_t rows = 1 + static_cast<int64_t>(rng() % 6000) * (rng() % 4 == 0 ? 16 : 1);
      DispatchIn in;
      in.engaged = true;
      in.rows = rows;
      in.p0 = pos;
      in.min_rows = min_rows;
      in.stage_ctx = S;
      in.gather_bytes_per_row[0] = GatherBytesPerRow(7, 2, 256);
      in.gather_bytes_per_row[1] = GatherBytesPerRow(9, 2, 256);
      StaleRows(tracker.PlanSync(pos), in.stale_rows);
      const Dispatch d = Decide(in);
      // oracle
      bool want_hybrid = true;
      Why want = Why::kHybrid;
      if (rows < min_rows) {
        want_hybrid = false;
        want = Why::kBelowMinRows;
      } else if (pos + rows > S) {
        want_hybrid = false;
        want = Why::kOverStageCtx;
      } else if (in.stale_rows[0] * in.gather_bytes_per_row[0] > kGatherBudgetBytes || in.stale_rows[1] * in.gather_bytes_per_row[1] > kGatherBudgetBytes) {
        want_hybrid = false;
        want = Why::kGatherBudget;
      }
      Check((d.route == Route::kHybrid) == want_hybrid && d.why == want, "random walk: dispatch == oracle");
      ++reasons[static_cast<int>(d.why)];
      if (pos + rows > 120000) {  // keep the walk inside the interesting range
        tracker.Reset();
        pos = 0;
        continue;
      }
      if (d.route == Route::kHybrid) {
        ++hybrid_calls;
        const TpMasterTracker::SyncPlan plan = tracker.PlanSync(pos);
        tracker.AfterSync(pos);
        (void)plan;
        pos += rows;
        tracker.AfterPipelined(pos);
      } else {
        ++tp_calls;
        tracker.TpOnly(pos);
        pos += rows;
      }
    } else if (op < 8) {  // TP decode (now and then a long stretch of it: the stages fall far behind)
      const bool stretch = rng() % 25 == 0 && pos < 45000;
      const int reps = stretch ? 25 : 1;
      for (int r = 0; r < reps; ++r) {
        const int64_t n = stretch ? 3000 : 1 + static_cast<int64_t>(rng() % 3000);
        tracker.TpOnly(pos);
        pos += n;
      }
    } else if (op == 8) {
      tracker.TpRestored();
      pos = pos > 100 ? pos - static_cast<int64_t>(rng() % 100) : pos;
    } else {
      tracker.Reset();
      pos = 0;
    }
  }
  std::fprintf(stderr, "  random walk: %d pipelined, %d TP prefill; reasons:", hybrid_calls, tp_calls);
  for (int w = 0; w < kWhyCount; ++w) std::fprintf(stderr, " %s=%d", WhyName(static_cast<Why>(w)), reasons[w]);
  std::fprintf(stderr, "\n");
  Check(hybrid_calls > 100 && tp_calls > 100, "the walk exercises both routes");
  Check(reasons[static_cast<int>(Why::kBelowMinRows)] > 0 && reasons[static_cast<int>(Why::kOverStageCtx)] > 0 &&
            reasons[static_cast<int>(Why::kGatherBudget)] > 0,
        "... and every fallback the walk can reach");
}

}  // namespace

int main() {
  KillSwitch();
  RingConstants();
  StagePlanning();
  TailPlans();
  TailRope();
  Rule();
  RandomWalk();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_hybrid_dispatch_cpu: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_hybrid_dispatch_cpu: PASS\n");
  return 0;
}
