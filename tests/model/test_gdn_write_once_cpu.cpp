// tests/model/test_gdn_write_once_cpu.cpp -- pure CPU. The host side of the write-once GDN state
// (docs/gdn-write-once.md, src/model/gdn_write_once.h):
//
//   1. R4DX_GDN_WRITE_ONCE's parser, the decode-legacy `gdnwo` token and DecideGdnWriteOnce's table;
//   2. the slot arithmetic (GdnSlotPlan), the ping-pong ring and the log sizing against libr4d's own
//      r4d_gdn_wo_log_bytes (a host function: no device is touched);
//   3. the ALGEBRA and the BOOKKEEPING, on a small fp32 emulation of the kernel's arithmetic: the same
//      scripted rounds run through the window-slot path (every row stores its own state, num_accepted picks
//      the seed) and through the write-once path (one state B, a per-row log, the accepted prefix replayed in
//      the next launch's seed, GdnPendingBook deciding when) must leave bitwise the same outputs and the
//      same logical state after EVERY event: verify (T = 1, 2, 5, 8) + commit n for every n, plain decode,
//      a collapse (the prefill / digest flush), chains of all of them from random scripts;
//   4. the latent bug the design found, in the GDN recurrent state alone (the emulation has no conv history):
//      two bare VerifyWindow calls with no commit between, after a round that committed n > 1. The window-slot
//      path re-reads a slot its first call overwrote and gives different rows; the write-once path seeds both
//      calls from the committed state (this test pins both). At MODEL level the second call's rows still differ
//      from the first's in both modes -- the conv history is a rolling buffer each call rewrites -- so
//      test_gdn_write_once_model checks the committed recurrent state, not the second call's logits;
//   5. the arithmetic contract: the replay's two roundings per element are the main loop's -- a replay with an
//      fma (the design's first draft, __fmaf_rn) is NOT bit-identical (negative control), which is why
//      libr4d's unit is built with -ffp-contract=off and pins it again in the write-once section.
//
// The emulation is the kernel's per-element arithmetic (decay, dot, correction, update, output) in the same
// order, one head, V = 6 rows, K = 16; the kernel's pair-split dot order is irrelevant here (both paths use
// whatever order they like, as long as it is the same function). The GPU side is tests/model/
// test_gdn_write_once.cpp (kernel level) and test_gdn_write_once_model.cpp (model level).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "gdn_write_once.h"
#include "r4dx/core/decode_legacy.hpp"
#include "r4dx/core/r4d.hpp"

// Every product and every sum below is its own rounding, as the GPU unit's -ffp-contract=off compiles them.
#pragma clang fp contract(off)

namespace {

using namespace r4dx::model;

int g_fail = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    ++g_fail;
    std::fprintf(stderr, "FAIL: %s\n", what);
  }
}
#define CHECKF(cond, ...)                                      \
  do {                                                         \
    if (!(cond)) {                                             \
      ++g_fail;                                                \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      std::fprintf(stderr, __VA_ARGS__);                       \
      std::fprintf(stderr, "\n");                              \
    }                                                          \
  } while (0)

// ---- 1. parsers and the decision ---------------------------------------------------------------------

void TestPolicy() {
  Check(ParseGdnWriteOnce(nullptr, false) == false && ParseGdnWriteOnce(nullptr, true) == true, "unset = the default");
  Check(ParseGdnWriteOnce("", false) == false && ParseGdnWriteOnce("", true) == true, "empty = the default");
  Check(ParseGdnWriteOnce("1") && ParseGdnWriteOnce("on"), "1 / on");
  Check(!ParseGdnWriteOnce("0", true) && !ParseGdnWriteOnce("off", true), "0 / off");
  Check(ParseGdnWriteOnce("bogus", true, false) == true && ParseGdnWriteOnce("bogus", false, false) == false,
        "an unreadable value is the default");
  Check(ParseGdnWriteOnce("ON", false, false) == false, "case matters like the other R4DX_* switches (unreadable)");
  Check(kGdnWriteOnceDefault == true, "the default is on since the GPU gates passed (2026-10-08)");
  Check(ValidGdnWriteOnceOption(-1) && ValidGdnWriteOnceOption(0) && ValidGdnWriteOnceOption(1) &&
            !ValidGdnWriteOnceOption(2) && !ValidGdnWriteOnceOption(-2),
        "option range");

  using r4dx::core::DecodeItem;
  using r4dx::core::ParseDecodeLegacy;
  const unsigned wo = static_cast<unsigned>(DecodeItem::kGdnWo);
  Check(ParseDecodeLegacy("gdnwo") == wo, "decode-legacy token gdnwo");
  Check(ParseDecodeLegacy("ab,gdnwo") == (static_cast<unsigned>(DecodeItem::kAb) | wo), "gdnwo among others");
  Check((ParseDecodeLegacy("all") & wo) != 0, "all includes gdnwo (all = every legacy path)");

  GdnWriteOnceInputs in;
  in.window = 8;
  const char* why = nullptr;
  in.env_request = false;
  Check(!DecideGdnWriteOnce(in, &why) && why != nullptr, "env off, option follows: window-slot path, with a reason");
  in.env_request = true;
  Check(DecideGdnWriteOnce(in), "env on: write-once");
  in.decode_legacy_gdnwo = true;
  Check(!DecideGdnWriteOnce(in, &why), "the gdnwo kill switch beats the env request");
  in.option = 1;
  Check(DecideGdnWriteOnce(in), "option 1 forces on whatever the environment and the kill switch say");
  in.option = 0;
  in.decode_legacy_gdnwo = false;
  Check(!DecideGdnWriteOnce(in, &why), "option 0 forces off whatever the environment says");
  in = GdnWriteOnceInputs{};
  in.option = 1;
  in.env_request = true;
  in.window = 1;
  Check(!DecideGdnWriteOnce(in, &why), "no speculative window: nothing to save, even when forced on");
}

// ---- 2. slots, ring, log size --------------------------------------------------------------------------

void TestPlanAndLog() {
  GdnSlotPlan legacy{1, 8, false};
  Check(legacy.SlotsPerSeq() == 8 && legacy.RecurrentSlots() == 9 && legacy.SlotForSeq(0) == 1, "window-slot plan: 8 + the null slot");
  GdnSlotPlan wo{1, 8, true};
  Check(wo.SlotsPerSeq() == 1 && wo.RecurrentSlots() == 2 && wo.SlotForSeq(0) == 1, "write-once plan: B + the null slot");
  Check(wo.ValidSlot(1) && !wo.ValidSlot(0) && !wo.ValidSlot(2), "write-once: only slot 1 exists (slot 0 is NULL_BLOCK_ID)");
  GdnSlotPlan wo2{2, 8, true};
  Check(wo2.SlotForSeq(0) == 1 && wo2.SlotForSeq(1) == 2 && wo2.RecurrentSlots() == 3, "two sequences: one slot each");
  GdnSlotPlan one{1, 1, false};
  Check(one.RecurrentSlots() == 2 && one.SlotForSeq(0) == 1, "window 1: unchanged");
  // VRAM: legacy (1x8+1) slots of H*V*K fp32 vs (1+1) slots + two logs, the design's numbers at TP=1 / TP=2
  const double slot_mib = 48.0 * 128 * 128 * 4 / 1048576.0;
  Check(slot_mib == 3.0, "one layer's state is 3 MiB");
  const double legacy_mib = 9 * slot_mib, wo_mib = 2 * slot_mib + 2.0 * GdnLogFloats(48, 16, 128, 128, 8) * 4 / 1048576.0;
  CHECKF(legacy_mib - wo_mib > 20.4 && legacy_mib - wo_mib < 20.6 && 48 * (legacy_mib - wo_mib) > 980.0,
         "per-layer saving %.2f MiB (x48 layers = %.0f MiB, the design's 984)", legacy_mib - wo_mib,
         48 * (legacy_mib - wo_mib));

  GdnLogRing ring;
  Check(ring.In() == 0 && ring.Out() == 1, "ring starts at In 0 / Out 1");
  ring.Flip();
  Check(ring.In() == 1 && ring.Out() == 0, "after one launch the written half is the next input");
  ring.Flip();
  Check(ring.In() == 0 && ring.Out() == 1, "and back");

  // the log layout contract: libr4d's own answer equals the host formula (floats * 4) for both ranks' shapes
  for (int depth : {2, 4, 8}) {
    for (auto hh : {std::pair<int, int>{48, 16}, std::pair<int, int>{24, 8}}) {
      const int64_t bytes = r4dx::core::r4d::GdnWoLogBytes(hh.first, hh.second, depth);
      CHECKF(bytes == GdnLogFloats(hh.first, hh.second, 128, 128, depth) * 4, "log bytes H %d Hg %d depth %d: kernel %lld host %lld",
             hh.first, hh.second, depth, static_cast<long long>(bytes),
             static_cast<long long>(GdnLogFloats(hh.first, hh.second, 128, 128, depth) * 4));
    }
  }
  Check(r4dx::core::r4d::GdnWoLogBytes(48, 16, 8) == 263680 && r4dx::core::r4d::GdnWoLogBytes(48, 16, 8) % 256 == 0,
        "the design's 263,680 B per buffer per layer at W = 8, 256-byte padded");
  Check(r4dx::core::r4d::GdnWoLogBytes(0, 16, 8) < 0, "a bad shape is refused");
}

// ---- 3. the emulation ----------------------------------------------------------------------------------

constexpr int kV = 6, kK = 16, kW = 8;  // v rows, k elements, the window

struct Row {  // one token's inputs to one head, already normalised the way the kernel hands them to the loop
  float eg, bt;
  std::vector<float> kk, qq, v;
};

Row RandomRow(std::mt19937& rng) {
  std::uniform_real_distribution<float> d(-1.0f, 1.0f), p(0.2f, 0.98f);
  Row r;
  r.eg = p(rng);
  r.bt = p(rng);
  r.kk.resize(kK);
  r.qq.resize(kK);
  r.v.resize(kV);
  for (auto& x : r.kk) x = d(rng) * 0.4f;
  for (auto& x : r.qq) x = d(rng) * 0.4f;
  for (auto& x : r.v) x = d(rng);
  return r;
}

using State = std::vector<float>;  // [kV][kK]

// The kernel's main loop for one token, per v row: decay, dot, correction, update, output. Returns the
// output row and (optionally) the logged correction u per v row.
std::vector<float> Step(State& h, const Row& r, std::vector<float>* u_out) {
  std::vector<float> o(kV);
  for (int row = 0; row < kV; ++row) {
    float* hr = &h[static_cast<size_t>(row) * kK];
    float dot = 0.0f;
    for (int j = 0; j < kK; ++j) {
      hr[j] = hr[j] * r.eg;
      dot = dot + hr[j] * r.kk[static_cast<size_t>(j)];
    }
    const float u = (r.v[static_cast<size_t>(row)] - dot) * r.bt;
    float on = 0.0f;
    for (int j = 0; j < kK; ++j) {
      hr[j] = hr[j] + u * r.kk[static_cast<size_t>(j)];
      on = on + hr[j] * r.qq[static_cast<size_t>(j)];
    }
    o[static_cast<size_t>(row)] = on;
    if (u_out != nullptr) (*u_out)[static_cast<size_t>(row)] = u;
  }
  return o;
}

// One logged row: exactly the three things the replay needs.
struct LogRow {
  float eg;
  std::vector<float> u;   // [kV]
  std::vector<float> kk;  // [kK]
};
using Log = std::vector<LogRow>;  // depth kW

// The replay's arithmetic: decay then update, the main loop's two roundings per element. `fused` is the
// negative control (a single-rounding fma in the update).
void Replay(State& h, const Log& log, int n, bool fused = false) {
  for (int t = 0; t < n; ++t) {
    for (int row = 0; row < kV; ++row) {
      float* hr = &h[static_cast<size_t>(row) * kK];
      const float u = log[static_cast<size_t>(t)].u[static_cast<size_t>(row)];
      for (int j = 0; j < kK; ++j) {
        hr[j] = hr[j] * log[static_cast<size_t>(t)].eg;
        const float kj = log[static_cast<size_t>(t)].kk[static_cast<size_t>(j)];
        hr[j] = fused ? std::fma(u, kj, hr[j]) : hr[j] + u * kj;
      }
    }
  }
}

bool Same(const std::vector<float>& a, const std::vector<float>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

// The window-slot path: kW slots, naccept picks the seed (naccept == 0: not valid, window 0).
struct LegacyPath {
  std::vector<State> slot = std::vector<State>(kW, State(kV * kK, 0.0f));
  int naccept = 0;

  std::vector<std::vector<float>> Verify(const std::vector<Row>& rows) {
    State h = slot[static_cast<size_t>(naccept > 0 ? naccept - 1 : 0)];
    std::vector<std::vector<float>> outs;
    for (size_t t = 0; t < rows.size(); ++t) {
      outs.push_back(Step(h, rows[t], nullptr));
      slot[t] = h;  // every candidate row stores its own state
    }
    return outs;
  }
  void Commit(int n) { naccept = n; }
  void Collapse() {  // CollapseWindow, then the thread is dropped
    if (naccept > 1) slot[0] = slot[static_cast<size_t>(naccept - 1)];
    naccept = 0;
  }
  State Logical() const { return slot[static_cast<size_t>(naccept > 0 ? naccept - 1 : 0)]; }
};

// The write-once path, with the real GdnPendingBook and GdnLogRing deciding.
struct WoPath {
  State B = State(kV * kK, 0.0f);
  Log logs[2] = {Log(kW), Log(kW)};
  GdnLogRing ring;
  GdnPendingBook book;

  std::vector<std::vector<float>> Verify(const std::vector<Row>& rows) {
    const int T = static_cast<int>(rows.size());
    State h = B;
    const bool pend = book.Pending();
    if (pend) Replay(h, logs[ring.In()], static_cast<int>(book.Count()));
    std::vector<std::vector<float>> outs;
    if (T == 1) {
      outs.push_back(Step(h, rows[0], nullptr));
      B = h;  // direct / legacy-in-place: the final state into B
      book.OnVerifyLaunched(1);
      return outs;
    }
    if (pend) B = h;  // the replay's write-back
    Log& out_log = logs[ring.Out()];
    for (int t = 0; t < T; ++t) {
      std::vector<float> u(kV);
      outs.push_back(Step(h, rows[static_cast<size_t>(t)], &u));
      out_log[static_cast<size_t>(t)] = LogRow{rows[static_cast<size_t>(t)].eg, u, rows[static_cast<size_t>(t)].kk};
    }
    ring.Flip();
    book.OnVerifyLaunched(T);
    return outs;
  }
  // A plain decode step is a T == 1 launch whose commit is implicit.
  std::vector<float> Plain(const Row& r) {
    auto outs = Verify({r});
    book.OnPlainDecode();
    return outs[0];
  }
  bool Commit(int n) { return book.OnCommit(n); }
  void Collapse() {  // Model::FlushGdnPending
    if (!book.Pending()) return;
    Replay(B, logs[ring.In()], static_cast<int>(book.Count()));
    book.Clear();
  }
  State Logical() const {  // the live state without mutating anything
    State h = B;
    if (book.Pending()) Replay(h, logs[ring.In()], static_cast<int>(book.Count()));
    return h;
  }
};

void SeedBoth(LegacyPath& a, WoPath& b, std::mt19937& rng) {
  std::uniform_real_distribution<float> d(-0.3f, 0.3f);
  for (auto& x : a.slot[0]) x = d(rng);
  b.B = a.slot[0];
}

std::vector<Row> Rows(std::mt19937& rng, int T) {
  std::vector<Row> r;
  for (int i = 0; i < T; ++i) r.push_back(RandomRow(rng));
  return r;
}

void CheckEqualOuts(const std::vector<std::vector<float>>& a, const std::vector<std::vector<float>>& b, const char* what, int step) {
  bool same = a.size() == b.size();
  for (size_t i = 0; same && i < a.size(); ++i) same = Same(a[i], b[i]);
  CHECKF(same, "%s (step %d): row outputs differ between the window-slot and the write-once path", what, step);
}

// every (T, n) with n <= T, from a fresh seed, one verify then the commit, then the state, then a second round
void TestEveryCommit() {
  std::mt19937 rng(7);
  int cases = 0;
  for (int T : {1, 2, 5, 8}) {
    for (int n = 1; n <= T; ++n) {
      LegacyPath lg;
      WoPath wo;
      SeedBoth(lg, wo, rng);
      const std::vector<Row> r1 = Rows(rng, T), r2 = Rows(rng, 8), r3 = Rows(rng, 3);
      CheckEqualOuts(lg.Verify(r1), wo.Verify(r1), "round 1", 0);
      lg.Commit(n);
      CHECKF(wo.Commit(n), "commit %d of %d rejected", n, T);
      CHECKF(Same(lg.Logical(), wo.Logical()), "T %d n %d: logical state after the commit differs", T, n);
      CHECKF(wo.book.Pending() == (T > 1), "T %d: pending = %d", T, wo.book.Pending());
      CheckEqualOuts(lg.Verify(r2), wo.Verify(r2), "round 2 (seeded from the commit)", 1);
      lg.Commit(8);
      wo.Commit(8);
      CHECKF(Same(lg.Logical(), wo.Logical()), "T %d n %d: logical state after round 2 differs", T, n);
      CheckEqualOuts(lg.Verify(r3), wo.Verify(r3), "round 3 (after a full-window commit)", 2);
      lg.Commit(1);
      wo.Commit(1);
      CHECKF(Same(lg.Logical(), wo.Logical()), "T %d n %d: logical state after commit 1 differs", T, n);
      ++cases;
    }
  }
  std::printf("[ok] every (T, n) for T in {1,2,5,8}: %d cases, outputs and logical state bitwise equal\n", cases);
}

// Random scripts of rounds, plain decodes and collapses (a prefill, a digest flush).
void TestScripts() {
  int events = 0;
  for (uint32_t seed = 1; seed <= 400; ++seed) {
    std::mt19937 rng(seed * 2654435761u);
    LegacyPath lg;
    WoPath wo;
    SeedBoth(lg, wo, rng);
    for (int step = 0; step < 24; ++step) {
      const int kind = static_cast<int>(rng() % 10);
      if (kind < 6) {  // a verify round + commit
        const int T = 1 + static_cast<int>(rng() % kW);
        const int n = 1 + static_cast<int>(rng() % static_cast<unsigned>(T));
        const std::vector<Row> rows = Rows(rng, T);
        CheckEqualOuts(lg.Verify(rows), wo.Verify(rows), "round", step);
        lg.Commit(n);
        CHECKF(wo.Commit(n), "seed %u step %d: commit rejected", seed, step);
      } else if (kind < 8) {  // a plain decode step (threads naccept = 1 on the legacy side)
        const Row r = RandomRow(rng);
        const auto lo = lg.Verify({r});
        lg.Commit(1);
        const auto wout = wo.Plain(r);
        CHECKF(lo.size() == 1 && Same(lo[0], wout), "seed %u step %d: plain decode row differs", seed, step);
      } else {  // a prefill-style collapse / a digest flush; the legacy path also drops its thread
        lg.Collapse();
        wo.Collapse();
        CHECKF(!wo.book.Pending(), "seed %u step %d: pending after a collapse", seed, step);
      }
      CHECKF(Same(lg.Logical(), wo.Logical()), "seed %u step %d (kind %d): logical state differs", seed, step, kind);
      ++events;
    }
    // a flush at any point is invisible: the continuing run is unchanged
    WoPath twin = wo;
    twin.Collapse();
    const std::vector<Row> rows = Rows(rng, 8);
    WoPath a = wo, b = twin;
    CheckEqualOuts(a.Verify(rows), b.Verify(rows), "flush invisibility", 99);
    a.Commit(5);
    b.Commit(5);
    CHECKF(Same(a.Logical(), b.Logical()), "seed %u: a flush before a round changed the continuing state", seed);
  }
  std::printf("[ok] 400 random scripts, %d events: outputs and logical state bitwise equal; flushes invisible\n", events);
}

// The commit count is bounded by the rows the last verify call logged.
void TestCommitBounds() {
  GdnPendingBook b;
  Check(!b.OnCommit(2), "a commit before any verify: only n = 1 (nothing was logged)");
  b.OnVerifyLaunched(3);
  Check(!b.OnCommit(4) && !b.OnCommit(0) && b.OnCommit(3) && b.Pending() && b.Count() == 3, "commit n <= T");
  b.OnVerifyLaunched(1);
  Check(b.OnCommit(1) && !b.Pending(), "a T = 1 verify leaves nothing pending (the kernel stored the state)");
  b.OnVerifyLaunched(8);
  b.OnCommit(2);
  b.OnPlainDecode();
  Check(!b.Pending() && b.LastVerifyT() == 1, "a plain decode consumes the pending prefix");
  b.OnVerifyLaunched(8);
  b.OnCommit(8);
  b.Clear();
  Check(!b.Pending() && !b.OnCommit(2), "Clear (prefill / reset / restore / flush)");
}

// 4. the latent double-VerifyWindow bug
void TestBareVerify() {
  std::mt19937 rng(99);
  LegacyPath lg;
  WoPath wo;
  SeedBoth(lg, wo, rng);
  // a round that committed n = 3 of 8
  const std::vector<Row> r1 = Rows(rng, 8);
  (void)lg.Verify(r1);
  (void)wo.Verify(r1);
  lg.Commit(3);
  wo.Commit(3);
  // two bare verify calls of the same window, no commit between
  const std::vector<Row> w = Rows(rng, 8);
  const auto l1 = lg.Verify(w), l2 = lg.Verify(w);
  const auto w1 = wo.Verify(w), w2 = wo.Verify(w);
  bool legacy_same = true, wo_same = true;
  for (size_t i = 0; i < l1.size(); ++i) {
    legacy_same = legacy_same && Same(l1[i], l2[i]);
    wo_same = wo_same && Same(w1[i], w2[i]);
  }
  Check(!legacy_same, "the window-slot path's second bare verify re-reads a slot its first call overwrote (the latent bug)");
  Check(wo_same, "the write-once path's second bare verify seeds from the committed state: same rows as the first (recurrent state only)");
  // and the committed state survived both bare calls
  CheckEqualOuts(w1, wo.Verify(w), "third bare verify", 2);
  std::printf("[ok] bare double verify: window-slot path differs between calls (bug pinned), write-once path idempotent\n");
}

// 5. the arithmetic contract
void TestFmaNegativeControl() {
  std::mt19937 rng(5);
  int differing = 0, total = 0;
  for (int trial = 0; trial < 200; ++trial) {
    State a(kV * kK), b;
    std::uniform_real_distribution<float> d(-0.3f, 0.3f);
    for (auto& x : a) x = d(rng);
    b = a;
    Log log(kW);
    State scratch = a;
    for (int t = 0; t < 4; ++t) {
      std::vector<float> u(kV);
      const Row r = RandomRow(rng);
      (void)Step(scratch, r, &u);
      log[static_cast<size_t>(t)] = LogRow{r.eg, u, r.kk};
    }
    State plain = a, fused = a;
    Replay(plain, log, 4);
    Replay(fused, log, 4, /*fused=*/true);
    Check(Same(plain, scratch), "the unfused replay reproduces the main loop's state bit for bit");
    for (size_t i = 0; i < plain.size(); ++i) {
      ++total;
      differing += std::memcmp(&plain[i], &fused[i], 4) != 0;
    }
  }
  CHECKF(differing > 0, "negative control: an fma replay never differed from the two-rounding loop");
  std::printf("[ok] negative control: an fma replay differs from the main loop in %d of %d elements (so the unit pins "
              "two roundings)\n", differing, total);
}

}  // namespace

int main() {
  TestPolicy();
  TestPlanAndLog();
  TestEveryCommit();
  TestScripts();
  TestCommitBounds();
  TestBareVerify();
  TestFmaNegativeControl();
  if (g_fail != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_fail);
    return 1;
  }
  std::printf("test_gdn_write_once_cpu: OK\n");
  return 0;
}
