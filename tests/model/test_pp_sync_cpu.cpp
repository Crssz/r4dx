// tests/model/test_pp_sync_cpu.cpp -- CPU-only checks of src/model/pp_sync.h, the stale-state rules of the real
// pipeline-parallel prefill (docs/pp-prefill.md Phase 2, section 3.3): the MirrorTracker's two numbers against an
// abstract model of both devices' contents under a few hundred thousand random operation sequences (pipelined
// prefill, B-only prefill, decode, speculative round, checkpoint save / restore, reset), the switches and the
// device-placement rule (--pp-devices / R4DX_PP_DEVICES / auto). A stale mirror silently produces wrong tokens, so this is the property the design leans on:
// whenever the tracker says "no copy needed", the mirror really does equal the decode Model on everything the next
// pipelined call reads. No HIP call, no container, always runs.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "pp_sync.h"

using namespace r4dx::model::pp;

namespace {

int g_fails = 0;
void Check(bool ok, const char* what) {
  if (!ok) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_fails;
  }
}

// The abstract contents of a device: a version stamp per KV row (what was written there last) and one stamp for the
// whole GDN state (what history it was last advanced by). Two devices agree on a row / the state iff the stamps match.
struct Dev {
  std::vector<int64_t> kv = std::vector<int64_t>(4096, -1);
  int64_t gdn = 0;
  int64_t pos = 0;
};

class World {
 public:
  void Reset() {
    b = Dev{};
    a = Dev{};
    t.Reset();
    ckpt_valid = false;
    stamp = 0;
  }
  // The tracker's plan applied: copy exactly what it names from B to A (the real code copies whole blocks, which only
  // ever copies more, never less, so the exact rows are the strictest model).
  void Sync() {
    const MirrorTracker::SyncPlan p = t.PlanSync(b.pos);
    if (p.gdn) a.gdn = b.gdn;
    for (int64_t r = p.kv_row0; r < p.kv_row1; ++r) a.kv[static_cast<size_t>(r)] = b.kv[static_cast<size_t>(r)];
    t.AfterSync(b.pos);
    a.pos = b.pos;
  }
  // The invariant the next pipelined call depends on: right after a sync, A equals B on rows [0, pos) and on the GDN
  // state; and the tracker's claims are never stronger than the truth at any time.
  bool Consistent() const {
    for (int64_t r = 0; r < t.KvValidTo(); ++r) {
      if (a.kv[static_cast<size_t>(r)] != b.kv[static_cast<size_t>(r)]) return false;
    }
    if (t.GdnPos() >= 0 && t.GdnPos() == b.pos && a.gdn != b.gdn) return false;
    return true;
  }
  bool EqualAfterSync() const {
    for (int64_t r = 0; r < b.pos; ++r) {
      if (a.kv[static_cast<size_t>(r)] != b.kv[static_cast<size_t>(r)]) return false;
    }
    return a.gdn == b.gdn && a.pos == b.pos;
  }

  void PipelinedPrefill(int64_t n) {
    Sync();
    if (!EqualAfterSync()) ++sync_failures;
    const int64_t p0 = b.pos;
    ++stamp;
    for (int64_t r = p0; r < p0 + n; ++r) {
      a.kv[static_cast<size_t>(r)] = b.kv[static_cast<size_t>(r)] = stamp * 10000 + r;
    }
    a.gdn = b.gdn = ++gdn_counter;
    a.pos = b.pos = p0 + n;
    t.AfterPipelined(b.pos);
  }
  void BOnlyAdvance(int64_t n) {  // decode / speculative round / B-only prefill: B moves, A does not
    const int64_t p0 = b.pos;
    ++stamp;
    for (int64_t r = p0; r < p0 + n; ++r) b.kv[static_cast<size_t>(r)] = stamp * 10000 + r + 5000000;
    b.gdn = ++gdn_counter;
    b.pos = p0 + n;
    t.BOnly(p0);
  }
  void Save() {  // SaveCheckpoint: B's state at its position (the engine saves right after a prefill)
    ckpt = b;
    ckpt_valid = true;
  }
  void Restore() {  // RestoreCheckpoint: B's GDN state and position come back; KV rows keep whatever they hold
    if (!ckpt_valid) return;
    b.gdn = ckpt.gdn;
    b.pos = ckpt.pos;
    t.BRestored();
  }

  Dev a, b, ckpt;
  MirrorTracker t;
  bool ckpt_valid = false;
  int64_t stamp = 0, gdn_counter = 0;
  int64_t sync_failures = 0;
};

void RandomSequences() {
  std::mt19937_64 rng(20261008);
  int64_t syncs = 0, bad = 0, claims_violated = 0;
  for (int seq = 0; seq < 20000; ++seq) {
    World w;
    w.Reset();
    for (int step = 0; step < 14; ++step) {
      const int op = static_cast<int>(rng() % 10);
      const int64_t n = 1 + static_cast<int64_t>(rng() % 200);
      if (w.b.pos + n + 300 >= 4000) {
        w.Reset();
        continue;
      }
      switch (op) {
        case 0:
          w.Reset();
          break;
        case 1:
        case 2:
        case 3:
          w.PipelinedPrefill(n);
          ++syncs;
          if (!w.EqualAfterSync()) ++bad;
          break;
        case 4:
        case 5:
          w.BOnlyAdvance(std::min<int64_t>(n, 12));  // decode / a speculative round
          break;
        case 6:
          w.BOnlyAdvance(n);  // a short B-only prefill
          break;
        case 7:
          w.Save();
          break;
        case 8:
          w.Restore();
          break;
        default:
          w.BOnlyAdvance(1);
          break;
      }
      if (!w.Consistent()) ++claims_violated;
    }
  }
  Check(syncs > 40000,"the random walk exercised many pipelined prefills");
  Check(bad == 0, "after the planned sync, the mirror equals the decode Model on every row and on the GDN state");
  Check(claims_violated == 0, "the tracker never claims more than the mirror holds");
}

void Rules() {
  MirrorTracker t;
  t.Reset();
  MirrorTracker::SyncPlan p = t.PlanSync(0);
  Check(!p.Any(), "a fresh pair needs no sync for a cold prompt");
  t.AfterPipelined(1000);
  p = t.PlanSync(1000);
  Check(!p.Any(), "right after a pipelined call, the next call at the same position needs nothing");
  t.BOnly(1000);  // decode from 1000
  p = t.PlanSync(1030);
  Check(p.gdn && p.kv_row0 == 1000 && p.kv_row1 == 1030, "after 30 decoded tokens: the GDN state and rows [1000, 1030)");
  t.AfterSync(1030);
  Check(!t.PlanSync(1030).Any(), "synced");
  t.BOnly(1030);
  t.BRestored();
  p = t.PlanSync(1000);  // restore to a checkpoint at 1000
  Check(p.gdn, "a restore always forces the GDN copy");
  Check(p.kv_row1 <= p.kv_row0 || p.kv_row0 == 1000, "no KV rows below the restored position are copied twice");
  // rewinding below the valid prefix never copies rows
  t.AfterPipelined(500);
  p = t.PlanSync(300);
  Check(p.kv_row0 == 300 && p.kv_row1 == 300, "a rewind copies no KV rows");
  // B-only writes invalidate exactly from their start
  t.AfterPipelined(2000);
  t.BOnly(1500);
  Check(t.KvValidTo() == 1500 && t.GdnPos() == -1, "a B-only write at 1500 shortens the valid prefix to 1500");
  t.BOnly(1800);
  Check(t.KvValidTo() == 1500, "a later B-only write does not lengthen it");
}

void Switches() {
  Check(ParsePpEnable(nullptr) == 0 && ParsePpEnable("") == 0 && ParsePpEnable("0") == 0 && ParsePpEnable("off") == 0 &&
            ParsePpEnable("false") == 0,
        "R4DX_PP unset / 0 / off / false = off (the default)");
  Check(ParsePpEnable("1") == 1 && ParsePpEnable("on") == 1 && ParsePpEnable("true") == 1 && ParsePpEnable("2") == 1,
        "R4DX_PP 1 / on / true / 2 = on");
  Check(ParsePpEnable("3") == -1 && ParsePpEnable("yes") == -1 && ParsePpEnable("-1") == -1, "anything else is refused");
  Check(DefaultSplit(false, 64) == 32 && DefaultSplit(true, 64) == 35, "default split 32, 35 with a drafter");
  Check(DefaultSplit(false, 4) == 3 && DefaultSplit(true, 4) == 3 && DefaultSplit(false, 2) == 1,
        "the default split clamps into [1, layers - 1] for a short container");
  Check(ShouldPipeline(1024, 1024) && !ShouldPipeline(1023, 1024) && ShouldPipeline(1, 1) && ShouldPipeline(1, 0),
        "the pipeline engages from min_rows (1 pipelines every call)");
}

void PlacementRules() {
  // --pp-devices / R4DX_PP_DEVICES text.
  std::vector<int> v{7};
  Check(ParseDevicePair("", &v).empty() && v.empty() && ParseDevicePair("auto", &v).empty() && v.empty(),
        "'' and 'auto' are the auto placement");
  Check(ParseDevicePair("1,0", &v).empty() && v == std::vector<int>({1, 0}), "'1,0' = stage B on ordinal 1, stage A on 0");
  Check(ParseDevicePair(" 0 , 1 ", &v).empty() && v == std::vector<int>({0, 1}), "whitespace is tolerated");
  for (const char* bad : {"1", "1,0,2", "1,1", "a,b", "1,", ",1", "-1,0", "1;0", "1.5,0", "99999,0"}) {
    v.assign({5, 5});
    Check(!ParseDevicePair(bad, &v).empty() && v.empty(), (std::string("'") + bad + "' is refused").c_str());
  }
  Check(ParseDevicePair("2,2", &v).find("different") != std::string::npos, "the same ordinal twice names the problem");

  // Auto: stage B on the LAST visible ordinal (physical device 1 = the headless card with HIP_VISIBLE_DEVICES unset), A before it.
  Placement p = ResolvePlacement({}, 2, "");
  Check(p.Ok() && p.stage_b == 1 && p.stage_a == 0, "auto with two visible cards: decode on ordinal 1, stage A on 0");
  p = ResolvePlacement({}, 2, "1,0");
  Check(p.Ok() && p.stage_b == 1 && p.stage_a == 0, "HIP_VISIBLE_DEVICES=1,0 is accepted and does not change the auto pick");
  p = ResolvePlacement({}, 3, "");
  Check(p.Ok() && p.stage_b == 2 && p.stage_a == 1, "auto with three visible: the last two");
  // Fewer than two visible: refused, naming the variable and the count.
  p = ResolvePlacement({}, 1, "1");
  Check(!p.Ok() && p.error.find("HIP_VISIBLE_DEVICES=1") != std::string::npos && p.error.find("exposes 1") != std::string::npos,
        "HIP_VISIBLE_DEVICES=1 (one card) is refused with a clear message");
  p = ResolvePlacement({}, 0, "");
  Check(!p.Ok() && p.error.find("<unset>") != std::string::npos && p.error.find("exposes 0") != std::string::npos,
        "no device: refused, names <unset>");
  p = ResolvePlacement({1, 0}, 1, "1");
  Check(!p.Ok(), "an explicit pair does not rescue a one-card process");
  // Explicit.
  p = ResolvePlacement({1, 0}, 2, "");
  Check(p.Ok() && p.stage_b == 1 && p.stage_a == 0, "explicit 1,0");
  p = ResolvePlacement({0, 1}, 2, "");
  Check(p.Ok() && p.stage_b == 0 && p.stage_a == 1, "explicit 0,1 (decode on the other card)");
  Check(!ResolvePlacement({2, 0}, 2, "").Ok() && !ResolvePlacement({0, 2}, 2, "").Ok() && !ResolvePlacement({-1, 0}, 2, "").Ok(),
        "an ordinal outside the visible range is refused");
  Check(!ResolvePlacement({1, 1}, 2, "").Ok() && !ResolvePlacement({1}, 2, "").Ok() && !ResolvePlacement({0, 1, 2}, 3, "").Ok(),
        "the same ordinal twice, or not exactly two entries, is refused");
}

}  // namespace

int main() {
  RandomSequences();
  Rules();
  Switches();
  PlacementRules();
  if (g_fails != 0) {
    std::fprintf(stderr, "test_pp_sync_cpu: %d FAILED\n", g_fails);
    return 1;
  }
  std::fprintf(stderr, "test_pp_sync_cpu: PASS\n");
  return 0;
}
