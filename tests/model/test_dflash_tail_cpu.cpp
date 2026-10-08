// tests/model/test_dflash_tail_cpu.cpp -- CPU-only checks of src/model/dflash_tail_plan.h, the rules of the hybrid mode's DFlash tail
// (docs/pp-tp2-hybrid.md 3, 6): after a pipelined prefill each TP rank's replicated drafter ring is filled from the last >= 2048 rows
// only, in the 64-row slices a full run injects.
//   * DflashTailStart: aligned to the call's own 64-row grid, never below p0, always leaving >= window rows (a table + a fuzz);
//   * TailCover (the stage-side capture's bookkeeping) takes exactly the rows of [start, end) from the slices a prefill call drains
//     (256-row super-chunks as four 64-row slices, then 64-row chunks, the last one short) and refuses gaps, overlaps and a call that
//     ends early;
//   * a SIMULATED drafter ring (a row's content depends on its features, its rope row and the size of the slice it was injected in,
//     which stands for the GEMM's M) shows the tail-fed ring has the same visible window as the fully fed one -- cold calls, warm
//     calls, images (rope rows != positions) -- and the CheckTailInject rule accepts exactly the safe tails;
//   * NEGATIVE CONTROLS: a tail starting after n - window, off the call's grid, with dropped rope rows or a skipped slice each leave a
//     different window.
// No HIP call, no container; always runs.
#include <cstdint>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "dflash_tail_plan.h"

using namespace r4dx::model::hybrid;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}
template <class F>
bool Throws(F&& f) {
  try {
    f();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

// ---- the start rule -----------------------------------------------------------------------------------------------------------------
void Start() {
  struct Row {
    int64_t p0, n, want;
  };
  const Row table[] = {
      {0, 1, 0},       {0, 1000, 0},    {0, 2048, 0},    {0, 2049, 0},    {0, 2111, 0},     {0, 2112, 64},   {0, 2113, 64},   {0, 8192, 6144},
      {0, 8193, 6144}, {100, 600, 100}, {100, 2148, 100}, {100, 2212, 164}, {100, 3000, 932}, {5000, 5100, 5000}, {5000, 9000, 6920},
  };
  for (const Row& r : table) {
    const int64_t s = DflashTailStart(r.p0, r.n);
    if (s != r.want) std::fprintf(stderr, "  DflashTailStart(%lld, %lld) = %lld, want %lld\n", static_cast<long long>(r.p0), static_cast<long long>(r.n), static_cast<long long>(s), static_cast<long long>(r.want));
    Check(s == r.want, "DflashTailStart: the table (aligned to p0 + 64 j, never below p0)");
  }
  std::mt19937_64 rng(77);
  bool ok = true;
  for (int i = 0; i < 20000; ++i) {
    const int64_t p0 = static_cast<int64_t>(rng() % 20000), n = p0 + 1 + static_cast<int64_t>(rng() % 40000);
    const int64_t s = DflashTailStart(p0, n);
    ok = ok && s >= p0 && (s - p0) % 64 == 0 && s < n;
    ok = ok && (s == p0 || s <= n - 2048);        // a gap only when the tail still covers the whole window
    ok = ok && (s == p0 || n - s < 2048 + 64);    // ... and no more rows than necessary (one slice of slack)
    ok = ok && n - s >= std::min<int64_t>(n - p0, 2048);
  }
  Check(ok, "DflashTailStart fuzz: on the call's grid, >= p0, leaves >= min(n - p0, 2048) rows and at most one slice more");
  Check(Throws([] { (void)DflashTailStart(5, 5); }) && Throws([] { (void)DflashTailStart(-1, 5); }) && Throws([] { (void)DflashTailStart(0, 5, 0); }),
        "DflashTailStart: empty / negative calls and a zero window are refused");
}

void SlicesAndRope() {
  const std::vector<TailSlice> s = TailSlices(2112 + 17);
  int64_t sum = 0;
  bool ok = true;
  for (size_t i = 0; i < s.size(); ++i) {
    ok = ok && s[i].row0 == sum && s[i].rows >= 1 && s[i].rows <= 64 && (i + 1 == s.size() || s[i].rows == 64);
    sum += s[i].rows;
  }
  Check(ok && sum == 2112 + 17 && s.size() == 34, "TailSlices: 64-row slices, the last one short, contiguous");
  Check(TailSlices(0).empty() && TailSlices(64).size() == 1 && TailSlices(65).size() == 2, "TailSlices: edge sizes");

  std::vector<int32_t> rope(3 * 10);
  for (int i = 0; i < 10; ++i) {
    rope[static_cast<size_t>(i)] = 100 + i;          // t
    rope[static_cast<size_t>(10 + i)] = 200 + i;     // h
    rope[static_cast<size_t>(20 + i)] = 300 + i;     // w
  }
  const std::vector<int32_t> t = TemporalRopeRows(rope, 10, 3, 4);
  Check(t == std::vector<int32_t>({103, 104, 105, 106}), "TemporalRopeRows: the temporal row of rows [3, 7)");
  Check(TemporalRopeRows(rope, 10, 0, 10).size() == 10 && TemporalRopeRows(rope, 10, 10, 0).empty(), "TemporalRopeRows: whole block / empty tail");
  Check(Throws([&] { (void)TemporalRopeRows(rope, 10, 8, 4); }) && Throws([&] { (void)TemporalRopeRows(rope, 9, 0, 1); }), "TemporalRopeRows: rows past the block and a block that is not [3, n] are refused");
}

// ---- the capture bookkeeping ---------------------------------------------------------------------------------------------------
// The slices (start, rows) a prefill call over [p0, n) drains: super-chunks of 256 rows while >= 256 remain (four slices each), then
// 64-row chunks, the last one short.
std::vector<TailSlice> DrainedSlices(int64_t p0, int64_t n) {
  std::vector<TailSlice> out;  // row0 holds the ABSOLUTE start
  int64_t pos = p0;
  while (pos < n) {
    const int64_t chunk = (n - pos >= 256) ? 256 : std::min<int64_t>(64, n - pos);
    for (int64_t r = 0; r < chunk; r += 64) out.push_back({pos + r, std::min<int64_t>(64, chunk - r)});
    pos += chunk;
  }
  return out;
}

void Cover() {
  std::mt19937_64 rng(5);
  bool ok = true;
  for (int i = 0; i < 400; ++i) {
    const int64_t p0 = static_cast<int64_t>(rng() % 3000), n = p0 + 1 + static_cast<int64_t>(rng() % 6000);
    const int64_t s = DflashTailStart(p0, n);
    TailCover cover(s, n);
    std::vector<int64_t> got(static_cast<size_t>(n - s), -1);
    for (const TailSlice& sl : DrainedSlices(p0, n)) {
      const TailCover::Take t = cover.Offer(sl.row0, sl.rows);
      for (int64_t k = 0; k < t.rows; ++k) {
        int64_t& cell = got[static_cast<size_t>(t.dst_row0 + k)];
        ok = ok && cell == -1;
        cell = sl.row0 + t.src_row0 + k;  // the absolute position that landed here
      }
    }
    ok = ok && cover.Complete() && cover.Taken() == n - s;
    for (int64_t k = 0; k < n - s && ok; ++k) ok = ok && got[static_cast<size_t>(k)] == s + k;
  }
  Check(ok, "TailCover: over the slices of random calls it takes exactly rows [start, n), each once, onto consecutive buffer rows");

  // a straddling first slice takes its upper part
  TailCover c(10, 30);
  Check(c.Offer(0, 8).rows == 0, "TailCover: a slice entirely below the tail is skipped");
  const TailCover::Take t = c.Offer(8, 8);
  Check(t.rows == 6 && t.src_row0 == 2 && t.dst_row0 == 0, "TailCover: a slice straddling the start takes its upper rows");
  const TailCover::Take u = c.Offer(16, 14);
  Check(u.rows == 14 && u.src_row0 == 0 && u.dst_row0 == 6 && c.Complete(), "TailCover: the next slice continues and completes the tail");
  Check(c.Offer(30, 8).rows == 0, "TailCover: slices after a complete tail are ignored");

  TailCover gap(10, 100);
  Check(Throws([&] { (void)gap.Offer(16, 16); }), "TailCover: a first slice that starts after the tail start is a gap");
  TailCover hole(0, 100);
  hole.Offer(0, 32);
  Check(Throws([&] { (void)hole.Offer(64, 32); }), "TailCover: a missing slice in the middle is a gap");
  TailCover overlap(0, 100);
  overlap.Offer(0, 32);
  Check(Throws([&] { (void)overlap.Offer(16, 32); }), "TailCover: an overlapping slice is refused");
  TailCover early(0, 100);
  early.Offer(0, 64);
  Check(!early.Complete() && Throws([&] { (void)early.Offer(200, 64); }), "TailCover: a call that moved past the end without completing the tail is refused");
  Check(Throws([] { TailCover bad(5, 5); }), "TailCover: an empty tail is refused");
}

// ---- the simulated drafter ring -----------------------------------------------------------------------------------------------
uint64_t Mix(uint64_t a, uint64_t b) { return (a ^ (b + 0x9E3779B97F4A7C15ull + (a << 6) + (a >> 2))) * 1099511628211ull; }

struct SimDraft {
  static constexpr int64_t kSlots = 2048;
  std::vector<uint64_t> ring = std::vector<uint64_t>(kSlots, 0);
  int64_t injected = 0, valid_from = 0;
  // InjectFeatures: strictly monotonic, a gap moves valid_from; a row's content depends on its features, its rope row and the size of
  // the slice (the GEMM's M).
  void InjectSlice(int64_t start, int64_t rows, const std::vector<uint64_t>& feat, const std::vector<int32_t>* rope) {
    if (start < injected) throw std::runtime_error("sim: injection is monotonic");
    if (start > injected) {
      valid_from = start;
      injected = start;
    }
    for (int64_t t = 0; t < rows; ++t) {
      const int64_t pos = start + t;
      const uint64_t r = rope != nullptr ? static_cast<uint64_t>((*rope)[static_cast<size_t>(pos)]) : static_cast<uint64_t>(pos);
      ring[static_cast<size_t>(pos % kSlots)] = Mix(Mix(feat[static_cast<size_t>(pos)], r), static_cast<uint64_t>(rows));
    }
    injected = start + rows;
  }
  void InjectRange(int64_t start, int64_t rows, const std::vector<uint64_t>& feat, const std::vector<int32_t>* rope) {
    for (const TailSlice& s : TailSlices(rows)) InjectSlice(start + s.row0, s.rows, feat, rope);
  }
  // The visible store: positions [lo, injected) with their ring contents.
  std::vector<uint64_t> Visible(int64_t* lo_out) const {
    const int64_t hi = injected;
    int64_t lo = std::max<int64_t>(valid_from, hi - kSlots);
    lo = std::max<int64_t>(lo, 0);
    *lo_out = lo;
    std::vector<uint64_t> v;
    for (int64_t p = lo; p < hi; ++p) v.push_back(ring[static_cast<size_t>(p % kSlots)]);
    return v;
  }
};

struct Scenario {
  int64_t p0, n;
  std::vector<uint64_t> feat;
  std::vector<int32_t> rope;  // temporal rope row per position (an "image" run shares one value)
  bool use_rope = false;
  SimDraft before;            // the drafter before the call (injected == p0 from earlier decode / prefill)
};
Scenario Make(std::mt19937_64& rng, int64_t p0, int64_t n, bool images) {
  Scenario s;
  s.p0 = p0;
  s.n = n;
  s.use_rope = images;
  s.feat.resize(static_cast<size_t>(n));
  s.rope.resize(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    s.feat[static_cast<size_t>(i)] = rng();
    s.rope[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  }
  if (images) {  // two image runs: every position of a run ropes at one temporal value (they straddle slices)
    for (int k = 0; k < 2; ++k) {
      const int64_t a = p0 + static_cast<int64_t>(rng() % static_cast<uint64_t>(n - p0)), len = 50 + static_cast<int64_t>(rng() % 400);
      for (int64_t i = a; i < std::min(n, a + len); ++i) s.rope[static_cast<size_t>(i)] = static_cast<int32_t>(a);
    }
  }
  // earlier history: whatever was injected below p0 (any slicing), then the call starts at p0
  if (p0 > 0) s.before.InjectRange(0, p0, s.feat, nullptr);
  return s;
}

void RingEquivalence() {
  std::mt19937_64 rng(2026);
  bool equal = true, rule_ok = true, sizes_ok = true;
  int gaps = 0;
  for (int i = 0; i < 500; ++i) {
    const bool warm = (rng() & 1) != 0, images = (rng() % 3) == 0;
    const int64_t p0 = warm ? static_cast<int64_t>(rng() % 6000) : 0;
    const int64_t n = p0 + 1 + static_cast<int64_t>(rng() % 9000);
    Scenario sc = Make(rng, p0, n, images);
    const std::vector<int32_t>* rope = sc.use_rope ? &sc.rope : nullptr;
    // the fully fed ring: the call's every slice
    SimDraft full = sc.before;
    full.InjectRange(p0, n - p0, sc.feat, rope);
    // the tail-fed ring: only [s, n), exactly what the hybrid does
    const int64_t s = DflashTailStart(p0, n);
    SimDraft tail = sc.before;
    rule_ok = rule_ok && CheckTailInject(tail.injected, n, s, n - s).empty();
    tail.InjectRange(s, n - s, sc.feat, rope);
    gaps += s > p0 ? 1 : 0;
    int64_t lo_f = 0, lo_t = 0;
    const std::vector<uint64_t> vf = full.Visible(&lo_f), vt = tail.Visible(&lo_t);
    equal = equal && vf == vt && lo_f == lo_t && full.injected == tail.injected;
    sizes_ok = sizes_ok && static_cast<int64_t>(vt.size()) == std::min<int64_t>(n, 2048);
  }
  Check(equal, "ring equivalence: a drafter fed only the tail (start on the call's grid) has the same visible window, bounds and frontier as one fed every row (cold, warm, images)");
  Check(rule_ok, "CheckTailInject accepts every tail DflashTailStart produces");
  Check(sizes_ok && gaps > 100, "the visible window is min(n, 2048) rows, and the fuzz exercised many gapped tails");

  // The stage-side capture feeds the tail: slices drained by a prefill call -> TailCover -> a host buffer -> the rank's injection
  bool via_cover = true;
  for (int i = 0; i < 100; ++i) {
    const int64_t p0 = (rng() & 1) ? static_cast<int64_t>(rng() % 4000) : 0, n = p0 + 1 + static_cast<int64_t>(rng() % 7000);
    Scenario sc = Make(rng, p0, n, (rng() & 1) != 0);
    const std::vector<int32_t>* rope = sc.use_rope ? &sc.rope : nullptr;
    const int64_t s = DflashTailStart(p0, n);
    TailCover cover(s, n);
    std::vector<uint64_t> buf_feat(static_cast<size_t>(n - s), 0);
    std::vector<int32_t> buf_rope(static_cast<size_t>(n - s), 0);
    for (const TailSlice& sl : DrainedSlices(p0, n)) {
      const TailCover::Take t = cover.Offer(sl.row0, sl.rows);
      for (int64_t k = 0; k < t.rows; ++k) {
        const int64_t pos = sl.row0 + t.src_row0 + k;
        buf_feat[static_cast<size_t>(t.dst_row0 + k)] = sc.feat[static_cast<size_t>(pos)];
        buf_rope[static_cast<size_t>(t.dst_row0 + k)] = sc.rope[static_cast<size_t>(pos)];
      }
    }
    // the rank injects from the buffer: the sim's arrays are indexed by position, so lay the buffer back at its positions
    std::vector<uint64_t> feat(static_cast<size_t>(n), 0);
    std::vector<int32_t> rope_all(static_cast<size_t>(n), 0);
    for (int64_t k = 0; k < n - s; ++k) {
      feat[static_cast<size_t>(s + k)] = buf_feat[static_cast<size_t>(k)];
      rope_all[static_cast<size_t>(s + k)] = buf_rope[static_cast<size_t>(k)];
    }
    SimDraft tail = sc.before, full = sc.before;
    tail.InjectRange(s, n - s, feat, sc.use_rope ? &rope_all : nullptr);
    full.InjectRange(p0, n - p0, sc.feat, rope);
    int64_t a = 0, b = 0;
    via_cover = via_cover && cover.Complete() && tail.Visible(&a) == full.Visible(&b) && a == b;
  }
  Check(via_cover, "capture -> TailCover -> host buffer -> rank injection reproduces the fully fed ring");
}

void NegativeControls() {
  std::mt19937_64 rng(11);
  int differ_late = 0, differ_rope = 0, differ_grid = 0, differ_skip = 0, trials = 0;
  for (int i = 0; i < 100; ++i) {
    const int64_t p0 = (rng() & 1) ? static_cast<int64_t>(rng() % 3000) : 0, n = p0 + 2300 + static_cast<int64_t>(rng() % 5000);
    Scenario sc = Make(rng, p0, n, true);
    const int64_t s = DflashTailStart(p0, n);
    // an image run inside the tail window for sure (so dropping the rope rows must show): 100 positions sharing one temporal value
    for (int64_t k = n - 1500; k < n - 1400; ++k) sc.rope[static_cast<size_t>(k)] = static_cast<int32_t>(n - 1500);
    SimDraft full = sc.before;
    full.InjectRange(p0, n - p0, sc.feat, &sc.rope);
    int64_t lo_f = 0;
    const std::vector<uint64_t> vf = full.Visible(&lo_f);
    const auto window_of = [&](const SimDraft& d) {
      int64_t lo = 0;
      const std::vector<uint64_t> v = d.Visible(&lo);
      return std::make_pair(v, lo);
    };
    ++trials;
    {  // a tail that starts one slice too late
      SimDraft d = sc.before;
      d.InjectRange(s + 64, n - s - 64, sc.feat, &sc.rope);
      const auto w = window_of(d);
      differ_late += (w.first != vf || w.second != lo_f) ? 1 : 0;
    }
    {  // the image rope rows dropped
      SimDraft d = sc.before;
      d.InjectRange(s, n - s, sc.feat, nullptr);
      differ_rope += window_of(d).first != vf ? 1 : 0;
    }
    {  // a start off the call's grid (13 rows after a grid point): the slices are cut elsewhere, so some rows see another M
      const int64_t off = s + 13 <= n - 2048 ? s + 13 : s - 13;
      if (off >= p0 && (off - p0) % 64 != 0 && (n - off) % 64 != (n - p0) % 64) {
        SimDraft d = sc.before;
        d.InjectRange(off, n - off, sc.feat, &sc.rope);
        differ_grid += window_of(d).first != vf ? 1 : 0;
      } else {
        ++differ_grid;  // not applicable to this draw: do not count against the control
      }
    }
    {  // one slice skipped in the middle of the tail
      SimDraft d = sc.before;
      const std::vector<TailSlice> sl = TailSlices(n - s);
      for (size_t k = 0; k < sl.size(); ++k) {
        if (k == sl.size() / 2) continue;
        d.InjectSlice(s + sl[k].row0, sl[k].rows, sc.feat, &sc.rope);
      }
      differ_skip += window_of(d).first != vf ? 1 : 0;
    }
  }
  Check(differ_late == trials, "NEGATIVE CONTROL: a tail starting one slice after n - 2048 changes the visible window");
  Check(differ_rope == trials, "NEGATIVE CONTROL: dropping the image rope rows changes the ring");
  Check(differ_grid == trials, "NEGATIVE CONTROL: a tail starting off the call's grid changes some rows (they were injected in slices of another size)");
  Check(differ_skip == trials, "NEGATIVE CONTROL: a skipped slice changes the ring");
}

void InjectRule() {
  Check(CheckTailInject(0, 5000, 2900, 2100).empty(), "CheckTailInject: a cold call's gapped tail of >= 2048 rows");
  Check(CheckTailInject(1000, 1500, 1000, 500).empty(), "CheckTailInject: an append at the frontier needs no full window");
  Check(!CheckTailInject(0, 5000, 4000, 1000).empty(), "CheckTailInject: a gap with fewer than 2048 rows is refused (the visible store would be short)");
  Check(!CheckTailInject(3000, 5000, 2900, 2100).empty(), "CheckTailInject: a start below the drafter's frontier is refused (injection is monotonic)");
  Check(!CheckTailInject(0, 5000, 2900, 2000).empty(), "CheckTailInject: a tail that does not end at the Model's position is refused");
  Check(!CheckTailInject(0, 5000, 5000, 0).empty() && !CheckTailInject(0, 5000, -1, 5001).empty(), "CheckTailInject: an empty tail and a negative start are refused");
}

}  // namespace

int main() {
  Start();
  SlicesAndRope();
  Cover();
  RingEquivalence();
  NegativeControls();
  InjectRule();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_dflash_tail_cpu: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_dflash_tail_cpu: PASS\n");
  return 0;
}
