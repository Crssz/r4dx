// tests/model/test_prefill_int8.cpp -- R4DX_PREFILL_INT8 (docs/int8-prefill.md "Production path") on a Model, against the
// f16 path. Built, and NOT RUN by the session that wrote it (CPU only): device 1, the machine idle, ~10 minutes
// (three loads of the 64-layer trellis container, one after the other).
//
// The real trellis production container, loaded three times with prefill_chunk = 256 and ModelOptions::prefill_int8
// 0 (off), 1 (on) and 1 again (on, for determinism); the same scenarios run on each (Prefill calls, then plain decode
// steps), and every observable is kept: the last row's logits, the greedy tokens that follow and a digest of ALL the
// per-sequence state (every KV cache, every GDN state: Model::DebugStateDigest, R4DX_TP_TESTING).
//   * T of 1, 63, 64, 65 and 255 (and a prefix-split pair that never reaches a super-chunk): no 256-row
//     super-chunk exists, so no int8 launch happens -- on == off in EVERY observable, bit for bit, and the Model counts
//     no int8 chunk;
//   * T of 256, 257, 511 and 1024 (and split calls whose parts reach one): the int8 chunks run exactly where the
//     chunk grid puts super-chunks (PrefillInt8ChunksRun == the sum over calls of floor(rows / 256), and equals the
//     off Model's PrefillWideChunksRun), the result is deterministic (the second on-load gives the same bytes), the
//     per-sequence state DIFFERS from off (negative control: the digest sees the int8 rows -- if it did not, the
//     comparisons above would prove nothing), the last row's logits stay close (KL(off || on) bounded, the top-1 token
//     the same unless the f16 top two were within 0.25 of each other);
//   * PrefillMultimodal (a text-only call through it) is f16 on a Model that has the switch on: the bytes equal off's and
//     no int8 chunk is counted; a Prefill call after it runs int8 again (the per-call flag does not stick);
//   * the 4-layer w4a16 and bf16 containers (no trellis linears) with the switch on: the Model refuses it, says so
//     (PrefillInt8Enabled() == false) and every byte equals off's.
// SKIPs (77) for a container that is missing; an exception from a present one is a FAIL.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "model.h"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {

const char* kL4Container = r4dx_test::ContainerPath("r4dx/qwen38-27b-l4-allmtp.r4dx");

using Trace = std::vector<std::pair<std::string, uint64_t>>;

uint64_t Fnv(const void* p, size_t n) {
  const auto* b = static_cast<const uint8_t*>(p);
  uint64_t h = 1469598103934665603ull;
  for (size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}
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

struct Scenario {
  std::string name;
  std::vector<int> calls;  // Prefill call lengths, in order (each continues the previous)
  bool multimodal = false;  // the first call goes through PrefillMultimodal (text only), the rest through Prefill
};

// the number of 256-row super-chunks the chunk grid makes for the calls (the grid is anchored at each call's start)
int64_t ExpectedSuperChunks(const Scenario& sc) {
  int64_t n = 0;
  for (size_t i = 0; i < sc.calls.size(); ++i) {
    if (sc.multimodal && i == 0) continue;   // PrefillMultimodal: super-chunks, but never int8
    n += sc.calls[i] / 256;
  }
  return n;
}
int64_t ExpectedWideChunks(const Scenario& sc) {
  int64_t n = 0;
  for (int c : sc.calls) n += c / 256;
  return n;
}

struct Obs {
  Trace trace;                  // logits hash, decode tokens and the state digest of the scenario, named
  std::vector<float> logits;    // the last row's logits
  std::vector<int32_t> decode;
  int64_t i8_chunks = 0, wide_chunks = 0;
};

Obs RunScenario(Model& m, const Scenario& sc) {
  Obs o;
  m.Reset();
  int total = 0;
  for (int c : sc.calls) total += c;
  const std::vector<int32_t> ids = Tokens(total, /*salt=*/1);
  const int64_t i8_before = m.PrefillInt8ChunksRun(), wide_before = m.PrefillWideChunksRun();
  size_t off = 0;
  int k = 0;
  for (int c : sc.calls) {
    const std::vector<int32_t> part(ids.begin() + static_cast<ptrdiff_t>(off), ids.begin() + static_cast<ptrdiff_t>(off + static_cast<size_t>(c)));
    o.logits = (sc.multimodal && k == 0) ? m.PrefillMultimodal(part, {}) : m.Prefill(part);
    off += static_cast<size_t>(c);
    o.trace.emplace_back("call" + std::to_string(k) + "/logits", Fnv(o.logits.data(), o.logits.size() * sizeof(float)));
    for (const auto& kv : m.DebugStateDigest()) o.trace.emplace_back("call" + std::to_string(k) + "/" + kv.first, kv.second);
    ++k;
  }
  int32_t tok = Argmax(o.logits);
  for (int i = 0; i < 2; ++i) {
    o.decode.push_back(m.DecodeStepGreedy(tok));
    tok = o.decode.back();
  }
  o.trace.emplace_back("decode/tokens", Fnv(o.decode.data(), o.decode.size() * sizeof(int32_t)));
  for (const auto& kv : m.DebugStateDigest()) o.trace.emplace_back("end/" + kv.first, kv.second);
  o.i8_chunks = m.PrefillInt8ChunksRun() - i8_before;
  o.wide_chunks = m.PrefillWideChunksRun() - wide_before;
  return o;
}

struct Side {
  std::vector<Obs> obs;     // per scenario
  bool enabled = false;
};

Side RunSide(const ModelOptions& base, int prefill_int8, const std::vector<Scenario>& scs) {
  ModelOptions o = base;
  o.prefill_chunk = 256;
  o.prefill_int8 = prefill_int8;
  Model m = Model::Load(o);
  Side s;
  s.enabled = m.PrefillInt8Enabled();
  for (const Scenario& sc : scs) {
    s.obs.push_back(RunScenario(m, sc));
    std::fprintf(stderr, "[prefill-int8]   prefill_int8 = %d: %s: %lld int8 / %lld super-chunks\n", prefill_int8, sc.name.c_str(),
                 static_cast<long long>(s.obs.back().i8_chunks), static_cast<long long>(s.obs.back().wide_chunks));
  }
  return s;
}

size_t CountDiff(const Trace& a, const Trace& b, const char* what, const std::string& cfg, bool print) {
  size_t bad = 0;
  if (a.size() != b.size()) {
    std::fprintf(stderr, "FAIL %s: %s: trace sizes differ (%zu vs %zu)\n", cfg.c_str(), what, a.size(), b.size());
    return a.size() + b.size();
  }
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      if (print && bad < 6) {
        std::fprintf(stderr, "FAIL %s: %s: %s differs: %016llx vs %016llx\n", cfg.c_str(), what, a[i].first.c_str(),
                     static_cast<unsigned long long>(a[i].second), static_cast<unsigned long long>(b[i].second));
      }
      ++bad;
    }
  }
  return bad;
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

std::vector<Scenario> Scenarios() {
  std::vector<Scenario> s;
  for (int n : {1, 63, 64, 65, 255}) s.push_back({"len" + std::to_string(n), {n}, false});
  s.push_back({"split64+191", {64, 191}, false});        // two calls, neither reaches a super-chunk
  for (int n : {256, 257, 511, 1024}) s.push_back({"len" + std::to_string(n), {n}, false});
  s.push_back({"split300+333", {300, 333}, false});      // a suffix call anchors its own grid: two super-chunks
  s.push_back({"split257+1", {257, 1}, false});
  s.push_back({"split1+255+257", {1, 255, 257}, false});
  s.push_back({"mm600", {600}, true});                   // text-only PrefillMultimodal: f16 even with the switch on
  s.push_back({"mm300+Prefill300", {300, 300}, true});   // ... and the next Prefill call runs int8 again
  return s;
}

int RunReal(const char* path) {
  ModelOptions base;
  base.container_path = path;
  base.layout = r4dx::model::LayoutFromName(r4dx_test::ProductionLayoutName());
  base.max_ctx = 2048;
  const std::vector<Scenario> scs = Scenarios();
  std::fprintf(stderr, "[prefill-int8] real container %s: three loads, %zu scenarios each\n", path, scs.size());
  const Side off = RunSide(base, 0, scs);
  const Side on = RunSide(base, 1, scs);
  const Side on2 = RunSide(base, 1, scs);
  int fails = 0;
  if (off.enabled) {
    std::fprintf(stderr, "FAIL: prefill_int8 = 0 loaded a Model that has the switch on\n");
    ++fails;
  }
  if (!on.enabled || !on2.enabled) {
    std::fprintf(stderr, "FAIL: prefill_int8 = 1 on the trellis production container did not enable the int8 path (see the load line)\n");
    return fails + 1;
  }
  for (size_t i = 0; i < scs.size(); ++i) {
    const Scenario& sc = scs[i];
    const std::string cfg = "real/" + sc.name;
    const int64_t expect_i8 = ExpectedSuperChunks(sc), expect_wide = ExpectedWideChunks(sc);
    bool ok = true;
    // the counters
    if (off.obs[i].i8_chunks != 0) {
      std::fprintf(stderr, "FAIL %s: the off Model ran %lld int8 chunks\n", cfg.c_str(), static_cast<long long>(off.obs[i].i8_chunks));
      ok = false;
    }
    if (off.obs[i].wide_chunks != expect_wide || on.obs[i].wide_chunks != expect_wide) {
      std::fprintf(stderr, "FAIL %s: super-chunks run: off %lld, on %lld, expected %lld\n", cfg.c_str(),
                   static_cast<long long>(off.obs[i].wide_chunks), static_cast<long long>(on.obs[i].wide_chunks),
                   static_cast<long long>(expect_wide));
      ok = false;
    }
    if (on.obs[i].i8_chunks != expect_i8 || on2.obs[i].i8_chunks != expect_i8) {
      std::fprintf(stderr, "FAIL %s: int8 chunks run: %lld and %lld, expected %lld\n", cfg.c_str(), static_cast<long long>(on.obs[i].i8_chunks),
                   static_cast<long long>(on2.obs[i].i8_chunks), static_cast<long long>(expect_i8));
      ok = false;
    }
    if (expect_i8 == 0) {
      // no int8 launch can have happened: every observable of on equals off's
      const size_t bad = CountDiff(off.obs[i].trace, on.obs[i].trace, "on vs off", cfg, true);
      if (bad != 0) {
        std::fprintf(stderr, "FAIL %s: %zu of %zu observables differ between on and off, but this scenario has no int8 chunk\n", cfg.c_str(),
                     bad, off.obs[i].trace.size());
        ok = false;
      }
      if (ok) std::fprintf(stderr, "[PASS] %s: no int8 chunk, %zu observables bit-identical on vs off\n", cfg.c_str(), off.obs[i].trace.size());
    } else {
      // determinism: the second on-load gives the same bytes
      const size_t bad_det = CountDiff(on.obs[i].trace, on2.obs[i].trace, "on vs on (second load)", cfg, true);
      if (bad_det != 0) {
        std::fprintf(stderr, "FAIL %s: the int8 path is not deterministic across loads: %zu observables differ\n", cfg.c_str(), bad_det);
        ok = false;
      }
      // the negative control: the state digest must see the int8 rows (the KV of the super-chunk differs from off's) --
      // except when every super-chunk of the scenario was PrefillMultimodal's (expect_i8 > 0 here, so it did run)
      size_t kv_diff = 0;
      for (size_t t = 0; t < off.obs[i].trace.size() && t < on.obs[i].trace.size(); ++t)
        if (off.obs[i].trace[t].first.find("/kv.") != std::string::npos && off.obs[i].trace[t] != on.obs[i].trace[t]) ++kv_diff;
      if (kv_diff == 0) {
        std::fprintf(stderr, "FAIL %s: the KV digest of the int8 run equals the f16 run's: either no int8 launch ran or the digest cannot see it\n",
                     cfg.c_str());
        ok = false;
      }
      const double kl = Kl(off.obs[i].logits, on.obs[i].logits);
      const double margin = Top2Margin(off.obs[i].logits);
      const bool top1 = Argmax(off.obs[i].logits) == Argmax(on.obs[i].logits);
      if (!(kl >= 0.0 && kl < 0.1)) {
        std::fprintf(stderr, "FAIL %s: KL(off || on) of the last row's logits is %.5f (bound 0.1)\n", cfg.c_str(), kl);
        ok = false;
      }
      if (!top1 && margin >= 0.25) {
        std::fprintf(stderr, "FAIL %s: the top-1 token differs although the f16 top-2 margin is %.3f logits\n", cfg.c_str(), margin);
        ok = false;
      }
      if (ok)
        std::fprintf(stderr, "[PASS] %s: %lld int8 chunks, deterministic, KV differs from f16 in %zu digests, KL(off||on) %.5f, top-1 %s (f16 margin %.3f)\n",
                     cfg.c_str(), static_cast<long long>(expect_i8), kv_diff, kl, top1 ? "same" : "flipped (near tie)", margin);
    }
    if (!ok) ++fails;
  }
  // PrefillMultimodal is f16: its scenarios' bytes with the switch on equal off's (the second call of mm300+Prefill300 is
  // an int8 Prefill call and is covered above)
  {
    const size_t i = static_cast<size_t>(std::find_if(scs.begin(), scs.end(), [](const Scenario& s) { return s.name == "mm600"; }) - scs.begin());
    const size_t bad = CountDiff(off.obs[i].trace, on.obs[i].trace, "on vs off", "real/mm600", true);
    if (bad != 0 || on.obs[i].i8_chunks != 0) {
      std::fprintf(stderr, "FAIL real/mm600: PrefillMultimodal with the switch on differs from f16 in %zu observables (%lld int8 chunks)\n", bad,
                   static_cast<long long>(on.obs[i].i8_chunks));
      ++fails;
    } else {
      std::fprintf(stderr, "[PASS] real/mm600: PrefillMultimodal is f16 with the switch on (bytes equal, no int8 chunk)\n");
    }
  }
  return fails;
}

int RunL4(Layout layout, const char* name) {
  ModelOptions base;
  base.container_path = kL4Container;
  base.layout = layout;
  base.max_ctx = 1536;
  base.layer_limit = 4;
  const std::vector<Scenario> scs = {{"len300", {300}, false}, {"len600", {600}, false}, {"split300+333", {300, 333}, false}};
  const Side off = RunSide(base, 0, scs);
  const Side on = RunSide(base, 1, scs);
  int fails = 0;
  if (on.enabled) {
    std::fprintf(stderr, "FAIL l4/%s: a container without trellis linears enabled the int8 path\n", name);
    ++fails;
  }
  for (size_t i = 0; i < scs.size(); ++i) {
    const size_t bad = CountDiff(off.obs[i].trace, on.obs[i].trace, "on vs off", std::string("l4/") + name + "/" + scs[i].name, true);
    if (bad != 0 || on.obs[i].i8_chunks != 0) {
      std::fprintf(stderr, "FAIL l4/%s/%s: %zu observables differ, %lld int8 chunks\n", name, scs[i].name.c_str(), bad,
                   static_cast<long long>(on.obs[i].i8_chunks));
      ++fails;
    }
  }
  if (fails == 0) std::fprintf(stderr, "[PASS] l4/%s: the switch is refused (no trellis linears) and every byte equals off's\n", name);
  return fails;
}

}  // namespace

static int RunTest() {
  int fails = 0, ran = 0;
  if (FileExists(kL4Container)) {
    const std::pair<Layout, const char*> kLayouts[] = {{Layout::kBf16, "bf16"}, {Layout::kW4a16, "w4a16"}};
    for (const auto& lp : kLayouts) {
      ++ran;
      fails += RunL4(lp.first, lp.second);
    }
  } else {
    std::fprintf(stderr, "[SKIP] %s not found -- the 4-layer part did not run\n", kL4Container);
  }
  const char* real = r4dx_test::ProductionTargetPath();
  if (FileExists(real)) {
    ++ran;
    fails += RunReal(real);
  } else {
    std::fprintf(stderr, "[SKIP] %s not found -- the real-container part did not run\n", real);
  }
  if (ran == 0) return SkipMissing(kL4Container);
  if (fails != 0) {
    std::fprintf(stderr, "test_prefill_int8: %d FAILED\n", fails);
    return 1;
  }
  std::fprintf(stderr, "test_prefill_int8: PASS\n");
  return 0;
}

int main() { return r4dx_test::RunGuardedMain("test_prefill_int8", RunTest); }
