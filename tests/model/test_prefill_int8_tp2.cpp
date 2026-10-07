// tests/model/test_prefill_int8_tp2.cpp -- R4DX_PREFILL_INT8_TP2 (docs/int8-prefill.md "Tensor parallel"): the int8 prefill GEMM on the
// shards of a TP = 2 rank. Built, and NOT RUN by the session that wrote it (CPU only): device 1, the machine idle, both ranks emulated on
// that one device (TpOptions::Mode::kEmulate, ~21 GiB), the real 64-layer trellis container loaded up to four times, one after the other.
//
//   A  TpModel, prefill_int8 = 1, prefill_int8_tp2 = 0 (the default): TP = 2 keeps the f16 kernels. Both ranks report
//      PrefillInt8Enabled() == false and run no int8 chunk. This is the TP = 2 f16 reference of everything below.
//   B  TpModel, prefill_int8 = 1, prefill_int8_tp2 = 1. What it must do depends on the TP = 2 tuning tables
//      (gemm_tuning_table_trellis_i8_tp2.inc, the bench's --tp 2 --emit-rows fills them):
//        * the tables are EMPTY (no shard shape has a row): the Model refuses int8 at load ("no TP rank-shard linear has an int8 plan"),
//          both ranks run f16, and EVERY observable (the last row's logits of every call, the greedy tokens after them) is byte-identical
//          to A's -- the fallback is correct, not silent and not different. The test stops there (nothing int8 to measure yet);
//        * rows exist (a shard shape's linears take the int8 GEMM): both ranks agree (TpModel::Load refuses a split decision), each rank
//          runs exactly floor(rows / 256) int8 chunks per call (the chunk grid) and the same number of them, scenarios that never reach
//          a 256-row super-chunk are byte-identical to A's (no int8 launch can have happened), the others are deterministic (a rerun gives
//          the same bytes), DIFFER from A's (negative control: the logits see the int8 rows) yet stay close (KL(A || B) bounded, the top-1
//          token the same unless the f16 top two were within 0.25 logits).
//   C, D  (only when B ran int8) the TP = 1 Model with prefill_int8 = 0 and 1: TP = 2 int8 against TP = 1 int8 (D) is as close as TP = 2 f16
//      against TP = 1 f16 (C vs A) is -- the per-(row, 128 k) A scales and per-(column, 128 k) weight scales of a shard are exactly the
//      TP = 1 scales of its blocks (docs/int8-prefill.md "Tensor parallel"; test_tp_loader checks the tables bit for bit), so the two differ
//      only in the order of the fp32 sums and the bf16 all-reduce. Printed as numbers; failing at KL(D || B) >= 0.1, where a wrong shard
//      scale would sit.
// SKIPs (77) for a missing container; an exception from a present one is a FAIL.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "model.h"
#include "test_common.h"
#include "tp_model.h"

using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx::model::TpModel;
using r4dx::model::TpOptions;

namespace {

int32_t Argmax(const std::vector<float>& v) {
  size_t best = 0;
  for (size_t i = 1; i < v.size(); ++i) {
    if (v[i] > v[best]) best = i;
  }
  return static_cast<int32_t>(best);
}
std::vector<int32_t> Tokens(int n, int salt) {
  std::vector<int32_t> ids(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) ids[static_cast<size_t>(i)] = 200 + (i * 53 + salt * 17 + (i / 7) * 3) % 5000;
  return ids;
}
// KL(softmax(p) || softmax(q)) in double, nats
double Kl(const std::vector<float>& p, const std::vector<float>& q) {
  double mp = -1e300, mq = -1e300;
  for (float x : p) mp = std::max(mp, static_cast<double>(x));
  for (float x : q) mq = std::max(mq, static_cast<double>(x));
  double zp = 0, zq = 0;
  for (float x : p) zp += std::exp(static_cast<double>(x) - mp);
  for (float x : q) zq += std::exp(static_cast<double>(x) - mq);
  double kl = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    const double lp = static_cast<double>(p[i]) - mp - std::log(zp), lq = static_cast<double>(q[i]) - mq - std::log(zq);
    kl += std::exp(lp) * (lp - lq);
  }
  return kl;
}
double Top2Margin(const std::vector<float>& v) {
  double a = -1e300, b = -1e300;
  for (float x : v) {
    if (x > a) {
      b = a;
      a = x;
    } else if (x > b) {
      b = x;
    }
  }
  return a - b;
}

struct Scenario {
  std::string name;
  std::vector<int> calls;  // Prefill call lengths, in order (each continues the previous)
};
const std::vector<Scenario>& Scenarios() {
  static const std::vector<Scenario> s = {{"len255", {255}}, {"len256", {256}}, {"len600", {600}}, {"split300+333", {300, 333}}, {"len1024", {1024}}};
  return s;
}
int64_t SuperChunks(const Scenario& sc) {
  int64_t n = 0;
  for (int c : sc.calls) n += c / 256;
  return n;
}

struct Obs {
  std::vector<float> logits;     // the last row's
  std::vector<int32_t> decode;   // two greedy tokens after it
  int64_t i8[2] = {0, 0};        // int8 chunks each rank ran during the scenario
  int64_t wide[2] = {0, 0};
};
struct Side {
  bool enabled[2] = {false, false};
  std::vector<Obs> obs;
  bool rerun_same = true;        // the last scenario with a super-chunk, run twice: the same bytes
};

struct Counts {
  bool enabled[2] = {false, false};
  int64_t i8[2] = {0, 0}, wide[2] = {0, 0};
};
Counts ReadCounts(TpModel& m) {
  Counts c;
  m.RunCollectiveForTest([&c](Model& r, int rank) {   // each rank thread writes its own slot; no device work, no comm call
    c.enabled[rank] = r.PrefillInt8Enabled();
    c.i8[rank] = r.PrefillInt8ChunksRun();
    c.wide[rank] = r.PrefillWideChunksRun();
  });
  return c;
}
Counts ReadCounts(Model& r) {
  Counts c;
  c.enabled[0] = r.PrefillInt8Enabled();
  c.i8[0] = r.PrefillInt8ChunksRun();
  c.wide[0] = r.PrefillWideChunksRun();
  return c;
}

template <class M>
Obs RunScenario(M& m, const Scenario& sc) {
  Obs o;
  const Counts before = ReadCounts(m);
  m.Reset();
  int total = 0;
  for (int c : sc.calls) total += c;
  const std::vector<int32_t> ids = Tokens(total, /*salt=*/1);
  size_t off = 0;
  for (int c : sc.calls) {
    const std::vector<int32_t> part(ids.begin() + static_cast<ptrdiff_t>(off), ids.begin() + static_cast<ptrdiff_t>(off + static_cast<size_t>(c)));
    o.logits = m.Prefill(part);
    off += static_cast<size_t>(c);
  }
  int32_t tok = Argmax(o.logits);
  for (int i = 0; i < 2; ++i) {
    o.decode.push_back(m.DecodeStepGreedy(tok));
    tok = o.decode.back();
  }
  const Counts after = ReadCounts(m);
  for (int r = 0; r < 2; ++r) {
    o.i8[r] = after.i8[r] - before.i8[r];
    o.wide[r] = after.wide[r] - before.wide[r];
  }
  return o;
}

bool SameBytes(const Obs& a, const Obs& b) {
  return a.logits.size() == b.logits.size() && std::memcmp(a.logits.data(), b.logits.data(), a.logits.size() * sizeof(float)) == 0 && a.decode == b.decode;
}

template <class M>
Side RunAll(M& m) {
  Side s;
  const Counts c = ReadCounts(m);
  s.enabled[0] = c.enabled[0];
  s.enabled[1] = c.enabled[1];
  for (const Scenario& sc : Scenarios()) {
    s.obs.push_back(RunScenario(m, sc));
    std::fprintf(stderr, "[prefill-int8-tp2]   %s: int8 chunks rank0 %lld rank1 %lld, super-chunks rank0 %lld\n", sc.name.c_str(),
                 static_cast<long long>(s.obs.back().i8[0]), static_cast<long long>(s.obs.back().i8[1]), static_cast<long long>(s.obs.back().wide[0]));
  }
  // determinism: a scenario with a super-chunk, again, on the same loaded Model
  for (size_t i = Scenarios().size(); i-- > 0;) {
    if (SuperChunks(Scenarios()[i]) == 0) continue;
    s.rerun_same = SameBytes(s.obs[i], RunScenario(m, Scenarios()[i]));
    break;
  }
  return s;
}

TpOptions EmulateOptions() {
  TpOptions t;
  t.world = 2;
  t.mode = TpOptions::Mode::kEmulate;
  return t;
}

Side RunTp(const ModelOptions& base, int tp2_option) {
  ModelOptions o = base;
  o.prefill_chunk = 256;
  o.prefill_int8 = 1;
  o.prefill_int8_tp2 = tp2_option;
  std::unique_ptr<TpModel> m = TpModel::Load(o, EmulateOptions());
  return RunAll(*m);
}
Side RunTp1(const ModelOptions& base, int int8) {
  ModelOptions o = base;
  o.prefill_chunk = 256;
  o.prefill_int8 = int8;
  Model m = Model::Load(o);
  return RunAll(m);
}

int Run(const char* path) {
  ModelOptions base;
  base.container_path = path;
  base.layout = r4dx::model::LayoutFromName(r4dx_test::ProductionLayoutName());
  base.max_ctx = 2048;
  base.vision = ModelOptions::VisionMode::kOff;
  const std::vector<Scenario>& scs = Scenarios();
  int fails = 0;
  std::fprintf(stderr, "[prefill-int8-tp2] real container %s: TP = 2 emulated, %zu scenarios per load\n", path, scs.size());

  const Side a = RunTp(base, /*tp2_option=*/0);
  if (a.enabled[0] || a.enabled[1]) {
    std::fprintf(stderr, "FAIL: prefill_int8_tp2 = 0 loaded a TP = 2 rank with the int8 path on (rank0 %d, rank1 %d)\n", a.enabled[0], a.enabled[1]);
    ++fails;
  }
  for (size_t i = 0; i < scs.size(); ++i) {
    if (a.obs[i].i8[0] != 0 || a.obs[i].i8[1] != 0) {
      std::fprintf(stderr, "FAIL A/%s: int8 chunks ran with the TP switch off\n", scs[i].name.c_str());
      ++fails;
    }
  }

  const Side b = RunTp(base, /*tp2_option=*/1);
  if (b.enabled[0] != b.enabled[1]) {
    std::fprintf(stderr, "FAIL B: the ranks disagree on the int8 decision (rank0 %d, rank1 %d)\n", b.enabled[0], b.enabled[1]);
    return fails + 1;
  }
  if (!b.enabled[0]) {
    // empty tables (or no row for any shard shape): f16 everywhere, byte-identical to A
    int bad = 0;
    for (size_t i = 0; i < scs.size(); ++i) {
      if (!SameBytes(a.obs[i], b.obs[i]) || b.obs[i].i8[0] != 0 || b.obs[i].i8[1] != 0) {
        std::fprintf(stderr, "FAIL B/%s: the TP int8 switch is on but no shard has a plan: the bytes must equal the f16 run's\n", scs[i].name.c_str());
        ++bad;
      }
    }
    fails += bad;
    if (bad == 0)
      std::fprintf(stderr,
                   "[PASS] tp2 fallback: R4DX_PREFILL_INT8_TP2 on, no shard shape has an int8 tuning row (the TP = 2 tables are empty until "
                   "tool_int8_gemm_proto --tp 2 --emit-rows has run): both ranks stay f16, every observable of %zu scenarios is byte-identical to the "
                   "switch-off run. Fill the tables, rebuild and run this test again for the int8 part.\n",
                   scs.size());
    if (fails != 0) return fails;
    return 0;
  }

  // int8 is on at both ranks
  for (size_t i = 0; i < scs.size(); ++i) {
    const int64_t expect = SuperChunks(scs[i]);
    const std::string cfg = "tp2/" + scs[i].name;
    bool ok = true;
    if (b.obs[i].i8[0] != expect || b.obs[i].i8[1] != expect || b.obs[i].wide[0] != expect) {
      std::fprintf(stderr, "FAIL %s: int8 chunks rank0 %lld rank1 %lld, super-chunks %lld, expected %lld\n", cfg.c_str(),
                   static_cast<long long>(b.obs[i].i8[0]), static_cast<long long>(b.obs[i].i8[1]), static_cast<long long>(b.obs[i].wide[0]),
                   static_cast<long long>(expect));
      ok = false;
    }
    if (expect == 0) {
      if (!SameBytes(a.obs[i], b.obs[i])) {
        std::fprintf(stderr, "FAIL %s: no super-chunk, yet the int8 run differs from the f16 run\n", cfg.c_str());
        ok = false;
      }
      if (ok) std::fprintf(stderr, "[PASS] %s: no super-chunk, byte-identical to f16\n", cfg.c_str());
    } else {
      const double kl = Kl(a.obs[i].logits, b.obs[i].logits), margin = Top2Margin(a.obs[i].logits);
      const bool top1 = Argmax(a.obs[i].logits) == Argmax(b.obs[i].logits);
      if (SameBytes(a.obs[i], b.obs[i])) {
        std::fprintf(stderr, "FAIL %s: the int8 run equals the f16 run byte for byte (the int8 chunks ran but changed nothing?)\n", cfg.c_str());
        ok = false;
      }
      if (!(kl >= 0.0 && kl < 0.1)) {
        std::fprintf(stderr, "FAIL %s: KL(f16 || int8) of the last row's logits is %.5f (bound 0.1)\n", cfg.c_str(), kl);
        ok = false;
      }
      if (!top1 && margin >= 0.25) {
        std::fprintf(stderr, "FAIL %s: the top-1 token differs although the f16 top-2 margin is %.3f logits\n", cfg.c_str(), margin);
        ok = false;
      }
      if (ok)
        std::fprintf(stderr, "[PASS] %s: %lld int8 chunks on each rank, KL(tp2 f16 || tp2 int8) %.5f, top-1 %s (f16 margin %.3f)\n", cfg.c_str(),
                     static_cast<long long>(expect), kl, top1 ? "same" : "flipped (near tie)", margin);
    }
    if (!ok) ++fails;
  }
  if (!b.rerun_same) {
    std::fprintf(stderr, "FAIL tp2: a rerun of a super-chunk scenario on the same Model gives different bytes\n");
    ++fails;
  } else {
    std::fprintf(stderr, "[PASS] tp2: a rerun of a super-chunk scenario is byte-identical (deterministic)\n");
  }

  // TP = 1, f16 and int8: how far each TP = 2 run sits from its own TP = 1 twin
  const Side c = RunTp1(base, 0);
  const Side d = RunTp1(base, 1);
  if (c.enabled[0] || !d.enabled[0]) {
    std::fprintf(stderr, "FAIL: the TP = 1 Models are not f16 / int8 as asked (c %d, d %d)\n", c.enabled[0], d.enabled[0]);
    ++fails;
  }
  for (size_t i = 0; i < scs.size(); ++i) {
    if (SuperChunks(scs[i]) == 0) continue;
    const double kl_f16 = Kl(c.obs[i].logits, a.obs[i].logits);   // TP = 1 f16 || TP = 2 f16
    const double kl_i8 = Kl(d.obs[i].logits, b.obs[i].logits);    // TP = 1 int8 || TP = 2 int8
    const double kl_cross = Kl(d.obs[i].logits, a.obs[i].logits);  // TP = 1 int8 || TP = 2 f16 (the int8 cost, seen from the other side)
    std::fprintf(stderr, "[prefill-int8-tp2] %s: KL(tp1 f16 || tp2 f16) %.6f   KL(tp1 int8 || tp2 int8) %.6f   KL(tp1 int8 || tp2 f16) %.6f   (top-1 tp1 int8 vs tp2 int8: %s)\n",
                 scs[i].name.c_str(), kl_f16, kl_i8, kl_cross, Argmax(d.obs[i].logits) == Argmax(b.obs[i].logits) ? "same" : "differs");
    if (!(kl_i8 >= 0.0 && kl_i8 < 0.1)) {
      std::fprintf(stderr, "FAIL tp2/%s: KL(TP=1 int8 || TP=2 int8) is %.5f (bound 0.1: a wrong shard scale would sit far above the TP=2 f16 distance %.6f)\n",
                   scs[i].name.c_str(), kl_i8, kl_f16);
      ++fails;
    }
  }
  return fails;
}

}  // namespace

static int RunTest() {
  const char* real = r4dx_test::ProductionTargetPath();
  if (!r4dx_test::FileExists(real)) return r4dx_test::SkipMissing(real);
  const int fails = Run(real);
  if (fails != 0) {
    std::fprintf(stderr, "test_prefill_int8_tp2: %d FAILED\n", fails);
    return 1;
  }
  std::fprintf(stderr, "test_prefill_int8_tp2: PASS\n");
  return 0;
}

int main() { return r4dx_test::RunGuardedMain("test_prefill_int8_tp2", RunTest); }
