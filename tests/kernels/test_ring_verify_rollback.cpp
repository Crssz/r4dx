// tests/kernels/test_ring_verify_rollback.cpp -- pure CPU (no HIP). docs/gemma4-plan.md D-6: the speculative
// verify window on Gemma's sliding KV ring needs NO explicit rollback of rejected rows, and this test is the
// proof by simulation of the exact lifecycle GemmaModel::DecodeStepDflashGreedy runs:
//
//   round: frontier F (= PositionCount()); a candidate window [anchor, d1..dk] (W = k + 1 <= 16 rows) is
//   written into the ring at slots (F + t) % R for ALL W rows (rejected rows included), every row's
//   attention reads its window keys [q - 1023, q] back through slot(kpos) = kpos % R, then only a + 1 rows
//   are committed (F += a + 1); the W - (a + 1) rejected rows' bytes stay in the ring, never undone.
//
// The ring is simulated as slot -> (position, token) labels, with the SlidingRingGeometry the model really
// uses (Gemma 4: window 1024, 288 chunk rows, 1536 slots). Invariants checked on every round of a long
// randomized run (wraps the ring ~100 times, W in [1, 16], any accept count 0..W-1, plus an adversarial
// all-reject and an all-accept pattern):
//   (1) every key a query row reads is exactly the key it should see: the COMMITTED token for a position
//       below the round's frontier, the CANDIDATE token for a position inside the window (earlier rows of the
//       same chunk) -- never a stale rejected row from an earlier round, never a wrapped older position;
//   (2) the full (non-ring) layers' contiguous cache obeys the same rule (positions are written before they
//       are read and a rejected row's position is rewritten by the next round's anchor);
//   (3) the bound is tight: a ring with no headroom breaks (1), so the 1536-slot ring (window + 288 -> 512
//       rows of headroom) is real margin for a 16-row verify window.
// Why it holds, in one line: a verify row writes position p into slot p % R, clobbering position p - R;
// p - R <= F + W - 1 - R < F - 1023 whenever R >= 1024 + W - 1, and every later query is at a position >= F,
// whose window starts at F - 1023 or later -- so a clobbered key is outside every future window.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "r4dx/model/attention/sliding_ring.hpp"

using r4dx::model::attention::SlidingRingGeometry;

namespace {

int g_fail = 0;
void Fail(const std::string& what) {
  if (++g_fail <= 20) std::printf("FAIL: %s\n", what.c_str());
}
#define CHECK(cond, msg)                                    \
  do {                                                      \
    if (!(cond)) Fail(std::string(msg) + "  [" #cond "]");  \
  } while (0)

struct Slot {
  int64_t pos = -1;
  int32_t tok = -1;
};

// Runs `rounds` verify rounds on a ring of `ring_tokens` slots with window `window`; returns the number of
// wrong reads (0 == the ring is rollback-safe at this size). `mode`: 0 random accept, 1 all reject, 2 all accept.
int64_t Simulate(int ring_tokens, int window, int rounds, int mode, uint64_t seed, int max_w = 16) {
  std::mt19937_64 rng(seed);
  std::vector<Slot> ring(static_cast<size_t>(ring_tokens));
  std::vector<int32_t> committed;  // committed[p] = the committed token at position p
  std::vector<Slot> full;          // the full layers' contiguous cache (never wraps)
  int64_t wrong = 0;

  // Prefill 3000 tokens (wraps the ring about twice).
  for (int64_t p = 0; p < 3000; ++p) {
    const int32_t t = static_cast<int32_t>(rng() % 50000);
    committed.push_back(t);
    ring[static_cast<size_t>(p % ring_tokens)] = {p, t};
    full.push_back({p, t});
  }
  for (int r = 0; r < rounds; ++r) {
    const int64_t F = static_cast<int64_t>(committed.size());
    const int W = 1 + static_cast<int>(rng() % static_cast<uint64_t>(max_w));
    // candidate window: row 0 = the anchor (committed at position F whatever happens), rows 1.. = drafts.
    std::vector<int32_t> cand(static_cast<size_t>(W));
    for (auto& c : cand) c = static_cast<int32_t>(rng() % 50000);
    int accepted;  // drafts accepted, 0..W-1
    if (mode == 1) accepted = 0;
    else if (mode == 2) accepted = W - 1;
    else accepted = static_cast<int>(rng() % static_cast<uint64_t>(W));

    // write ALL W rows first (the chunk write precedes the attention), then every row reads its window.
    for (int t = 0; t < W; ++t) {
      ring[static_cast<size_t>((F + t) % ring_tokens)] = {F + t, cand[static_cast<size_t>(t)]};
      if (static_cast<int64_t>(full.size()) <= F + t) full.resize(static_cast<size_t>(F + t + 1));
      full[static_cast<size_t>(F + t)] = {F + t, cand[static_cast<size_t>(t)]};
    }
    for (int t = 0; t < W; ++t) {
      const int64_t q = F + t;
      for (int64_t kpos = std::max<int64_t>(0, q - window + 1); kpos <= q; ++kpos) {
        const int32_t expect = kpos < F ? committed[static_cast<size_t>(kpos)] : cand[static_cast<size_t>(kpos - F)];
        const Slot& s = ring[static_cast<size_t>(kpos % ring_tokens)];
        if (s.pos != kpos || s.tok != expect) ++wrong;
        const Slot& f = full[static_cast<size_t>(kpos)];
        if (f.pos != kpos || f.tok != expect) ++wrong;
      }
    }
    // commit the anchor + the accepted drafts; the rejected rows stay in both caches as stale bytes. They
    // sit at positions >= the new frontier: later queries only read kpos <= qpos, and every such position
    // is rewritten (by its own round's chunk write) before it can be read.
    for (int t = 0; t <= accepted; ++t) committed.push_back(cand[static_cast<size_t>(t)]);
  }
  return wrong;
}

void TestGemmaRing() {
  const SlidingRingGeometry g(1024, 288, 16, 131072);
  CHECK(g.RingTokens() == 1536 && g.ChunkFits(16), "Gemma ring: 1536 slots, a 16-row window fits");
  for (int mode = 0; mode < 3; ++mode) {
    const int64_t wrong = Simulate(g.RingTokens(), 1024, 20000, mode, 0xD1F1A5ull + static_cast<uint64_t>(mode));
    CHECK(wrong == 0, "1536-slot ring, W<=16, mode " + std::to_string(mode) + ": every window key correct (wrong=" +
                          std::to_string(wrong) + ")");
  }
}

void TestBoundIsTight() {
  // exactly window + W_max slots (a whole number of 16-token blocks): safe.
  CHECK(Simulate(1024 + 16, 1024, 4000, 0, 11) == 0, "ring = window + 16: safe for a 16-row window");
  // no headroom: a 16-row verify window clobbers a key a row still needs.
  CHECK(Simulate(1024, 1024, 4000, 0, 12) > 0, "ring = window (no headroom): the simulation detects the clobber");
  const SlidingRingGeometry tight(1024, 16, 16, 131072, /*ring_tokens=*/1024 + 16);
  CHECK(tight.ChunkFits(16) && !tight.ChunkFits(17), "geometry refuses a chunk beyond its headroom");
}

void TestOtherWindows() {
  // the rule is ring >= window + W - 1 for any window (the drafter's own 2048 ring included).
  CHECK(Simulate(256 + 16, 256, 3000, 0, 21) == 0, "window 256 / ring 272 / W<=16: safe");
  CHECK(Simulate(2048 + 16, 2048, 1500, 0, 22) == 0, "window 2048 / ring 2064 / W<=16: safe");
}

}  // namespace

int main() {
  TestGemmaRing();
  TestBoundIsTight();
  TestOtherWindows();
  if (g_fail != 0) {
    std::printf("%d failure(s)\n", g_fail);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
