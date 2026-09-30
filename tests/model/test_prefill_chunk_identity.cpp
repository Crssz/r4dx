// tests/model/test_prefill_chunk_identity.cpp -- the 256-row prefill chunk (docs/prefill.md) leaves
// exactly the bits the 64-row chunk leaves. Two Models are loaded one after the other, ModelOptions::
// prefill_chunk = 64 (the kill switch's engine) and 256 (the default), fed the same prompts, and every
// observable is compared bit for bit: the last row's logits, the greedy tokens that follow, a digest of
// ALL the per-sequence state (every attention KV cache, every GDN recurrent and conv state, the MTP head's
// KV cache, the DFlash2 drafter's K / V ring: Model::DebugStateDigest, R4DX_TP_TESTING), the speculative
// rounds' tokens and the DFlash feature capture drained through Prefill's per-chunk callback.
//
// What is prefilled (the tail cases of the chunk grid, docs/prefill.md): one call of 1, 63, 64, 65, 255,
// 256, 257, 511 and 8145 tokens (8145 on the 4-layer container only), and prefix-reuse shapes -- a second
// Prefill call that continues the first (the chunk grid is anchored at each call's start), a suffix
// shorter than a chunk, a one-token suffix -- each followed by plain decode steps and, where the
// configuration has a drafter, speculative rounds.
//
// Two container sets:
//   * the 4-layer test container (qwen38-27b-l4-allmtp, layouts bf16 and w4a16), fast: plain, --mtp 3, and
//     a target feature capture drained through the on_chunk_captured callback. Its linears are not
//     trellis, so ApplyLinear slices a 256-row call into four 64-row launches: this covers the chunked
//     GDN, attention, MTP priming, DFlash capture and grid logic;
//   * the real 64-layer trellis production container (SKIPs without it): plain, --mtp 3 and --dflash with
//     its real drafter, where the M = 256 trellis GEMM itself is under test (the KL harness and the CLI
//     runs of docs/prefill.md cover the long prompts).
// SKIPs (CTest SKIPPED) for a container that is missing; an exception from a present one is a FAIL.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <array>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "model.h"
#include "r4dx/kernels/kernels.h"
#include "tp_model.h"
#include "test_common.h"

using namespace r4dx_test;
using r4dx::model::Layout;
using r4dx::model::Model;
using r4dx::model::ModelOptions;

namespace {

const char* kL4Container = r4dx_test::ContainerPath("D:/models/r4dx/qwen38-27b-l4-allmtp.r4dx");

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
uint64_t HashLogits(const std::vector<float>& x) { return Fnv(x.data(), x.size() * sizeof(float)); }
uint64_t HashTokens(const std::vector<int32_t>& x) { return Fnv(x.data(), x.size() * sizeof(int32_t)); }

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
};

enum class Mode { kPlain, kMtp, kDflash, kCapture };

struct Config {
  std::string name;
  ModelOptions opts;
  Mode mode;
  int64_t spec_k = 0;
  bool expect_wide;  // this configuration must load 256-row (the decision table, prefill_chunk.h)
  std::vector<Scenario> scenarios;
};

// One scenario on `m`: reset, the Prefill calls, then 2 plain decode steps and (drafter configurations) 3
// speculative rounds; every observable is appended to `tr` under `<scenario>/<what>`.
void RunScenario(Model& m, const Config& cfg, const Scenario& sc, Trace* tr) {
  m.Reset();
  int total = 0;
  for (int c : sc.calls) total += c;
  const std::vector<int32_t> ids = Tokens(total, /*salt=*/1);
  const auto add = [&](const std::string& what, uint64_t v) { tr->emplace_back(sc.name + "/" + what, v); };
  const auto add_state = [&](const std::string& tag) {
    for (const auto& kv : m.DebugStateDigest()) add(tag + "/" + kv.first, kv.second);
  };
  // Feature capture drained per chunk (kCapture / kDflash use the same hook): rows and content hash.
  uint64_t feature_chain = 1469598103934665603ull;
  int64_t feature_rows = 0, feature_calls = 0;
  const auto drain = [&] {
    const int64_t rows = m.DflashFeatureRows();
    const int64_t elems = rows * m.DflashFeatureCols();
    ++feature_calls;
    feature_rows += rows;
    if (elems > 0) {
      std::vector<uint16_t> host(static_cast<size_t>(elems));
      (void)hipMemcpy(host.data(), m.DflashFeatureBuffer(), host.size() * 2, hipMemcpyDeviceToHost);
      const uint64_t h = Fnv(host.data(), host.size() * 2);
      feature_chain = (feature_chain ^ h) * 1099511628211ull;
    }
  };
  const std::function<void()> cb = cfg.mode == Mode::kCapture ? std::function<void()>(drain) : nullptr;

  std::vector<float> logits;
  size_t off = 0;
  int k = 0;
  for (int c : sc.calls) {
    const std::vector<int32_t> part(ids.begin() + static_cast<ptrdiff_t>(off),
                                    ids.begin() + static_cast<ptrdiff_t>(off + static_cast<size_t>(c)));
    logits = m.Prefill(part, cb);
    off += static_cast<size_t>(c);
    add("call" + std::to_string(k) + "/logits", HashLogits(logits));
    add_state("call" + std::to_string(k));
    ++k;
  }
  if (cfg.mode == Mode::kCapture) {
    add("capture/rows", static_cast<uint64_t>(feature_rows));
    add("capture/callbacks", static_cast<uint64_t>(feature_calls));
    add("capture/content", feature_chain);
  }
  int32_t tok = Argmax(logits);
  std::vector<int32_t> plain;
  for (int i = 0; i < 2; ++i) {
    plain.push_back(m.DecodeStepGreedy(tok));
    tok = plain.back();
  }
  add("decode/tokens", HashTokens(plain));
  std::vector<int32_t> spec;
  for (int i = 0; i < 3 && cfg.mode != Mode::kPlain && cfg.mode != Mode::kCapture; ++i) {
    const std::vector<int32_t> round = cfg.mode == Mode::kMtp
                                           ? m.DecodeStepMtpGreedy(tok, cfg.spec_k)
                                           : m.DecodeStepDflashGreedy(tok, cfg.spec_k, /*p_min=*/0.0f, /*n_min=*/0);
    spec.insert(spec.end(), round.begin(), round.end());
    tok = round.back();
  }
  if (!spec.empty()) add("spec/tokens", HashTokens(spec));
  add_state("end");
}

// Loads `cfg` with ModelOptions::prefill_chunk = chunk, runs every scenario, returns the trace. Also fills
// `launches` with the kernel launch count of the prefill in `count_scenario` (the last scenario's first
// call) when non-null.
bool RunConfigSide(const Config& cfg, int chunk, Trace* tr, std::vector<int64_t>* launches, std::vector<int64_t>* wide_chunks,
                   bool* wide_out) {
  ModelOptions o = cfg.opts;
  o.prefill_chunk = chunk;
  Model m = Model::Load(o);
  *wide_out = m.PrefillChunkRows() == 256;
  if (cfg.mode == Mode::kCapture) m.AttachDflashFeatureCapture({0, 1, 2, 3});
  for (const Scenario& sc : cfg.scenarios) {
    r4dx_kernel_launch_counter_reset();
    const int64_t before = m.PrefillWideChunksRun();
    RunScenario(m, cfg, sc, tr);
    if (launches != nullptr) launches->push_back(r4dx_kernel_launch_counter_get());
    if (wide_chunks != nullptr) wide_chunks->push_back(m.PrefillWideChunksRun() - before);
  }
  return true;
}

bool RunConfig(const Config& cfg) {
  // R4DX_TEST_ONLY=<substring>: run only the configurations whose name contains it (a debugging aid).
  if (const char* only = std::getenv("R4DX_TEST_ONLY")) {
    if (*only != '\0' && cfg.name.find(only) == std::string::npos) return true;
  }
  std::fprintf(stderr, "[chunk-identity] %s: %zu scenarios\n", cfg.name.c_str(), cfg.scenarios.size());
  Trace narrow, wide;
  std::vector<int64_t> narrow_launches, wide_launches, narrow_chunks, wide_chunks;
  bool narrow_wide = false, wide_wide = false;
  RunConfigSide(cfg, 64, &narrow, &narrow_launches, &narrow_chunks, &narrow_wide);
  RunConfigSide(cfg, 256, &wide, &wide_launches, &wide_chunks, &wide_wide);
  bool ok = true;
  // The wide path really ran, in every configuration: super-chunks were run exactly where a call has
  // 256 rows or more (the grid: floor(rows / 256) per call), never by the 64-row Model.
  for (size_t s = 0; s < cfg.scenarios.size(); ++s) {
    int64_t expect = 0;
    for (int c : cfg.scenarios[s].calls) expect += c / 256;
    if (narrow_chunks[s] != 0) {
      std::fprintf(stderr, "FAIL %s: %s: the 64-row Model ran %lld super-chunks\n", cfg.name.c_str(),
                   cfg.scenarios[s].name.c_str(), static_cast<long long>(narrow_chunks[s]));
      ok = false;
    }
    if (wide_chunks[s] != (cfg.expect_wide ? expect : 0)) {
      std::fprintf(stderr, "FAIL %s: %s: the 256-row Model ran %lld super-chunks, expected %lld\n", cfg.name.c_str(),
                   cfg.scenarios[s].name.c_str(), static_cast<long long>(wide_chunks[s]),
                   static_cast<long long>(cfg.expect_wide ? expect : 0));
      ok = false;
    }
  }
  if (narrow_wide) {
    std::fprintf(stderr, "FAIL %s: prefill_chunk = 64 loaded a 256-row Model (the kill switch)\n", cfg.name.c_str());
    ok = false;
  }
  if (wide_wide != cfg.expect_wide) {
    std::fprintf(stderr, "FAIL %s: prefill_chunk = 256 loaded %d-row, expected %d-row\n", cfg.name.c_str(),
                 wide_wide ? 256 : 64, cfg.expect_wide ? 256 : 64);
    ok = false;
  }
  if (narrow.size() != wide.size()) {
    std::fprintf(stderr, "FAIL %s: trace sizes differ (%zu vs %zu)\n", cfg.name.c_str(), narrow.size(), wide.size());
    return false;
  }
  size_t bad = 0;
  for (size_t i = 0; i < narrow.size(); ++i) {
    if (narrow[i].first != wide[i].first || narrow[i].second != wide[i].second) {
      if (bad++ < 12) {
        std::fprintf(stderr, "FAIL %s: %s differs: 64-row %016llx, 256-row %016llx\n", cfg.name.c_str(),
                     narrow[i].first.c_str(), static_cast<unsigned long long>(narrow[i].second),
                     static_cast<unsigned long long>(wide[i].second));
      }
    }
  }
  if (bad != 0) {
    std::fprintf(stderr, "FAIL %s: %zu of %zu observables differ\n", cfg.name.c_str(), bad, narrow.size());
    ok = false;
  }
  // The wide path really ran (the capture configuration loads wide but its callback still falls back at\n  // the call level until the callback is served, so the launch check is for the plain one): a scenario that prefills at least 256 rows in one call launches fewer
  // kernels wide than narrow (embedding, norms, rope, ... are one launch per 256 rows instead of four).
  if (cfg.expect_wide && cfg.mode == Mode::kPlain) {
    for (size_t s = 0; s < cfg.scenarios.size(); ++s) {
      const int64_t first = cfg.scenarios[s].calls[0];
      if (first >= 256 && !(wide_launches[s] < narrow_launches[s])) {
        std::fprintf(stderr, "FAIL %s: %s launched %lld kernels wide vs %lld narrow (the wide path did not run)\n",
                     cfg.name.c_str(), cfg.scenarios[s].name.c_str(), static_cast<long long>(wide_launches[s]),
                     static_cast<long long>(narrow_launches[s]));
        ok = false;
      }
      if (first < 256 && cfg.scenarios[s].calls.size() == 1 && wide_launches[s] != narrow_launches[s]) {
        std::fprintf(stderr, "FAIL %s: %s (%lld rows) launched %lld kernels wide vs %lld narrow (expected equal)\n",
                     cfg.name.c_str(), cfg.scenarios[s].name.c_str(), static_cast<long long>(first),
                     static_cast<long long>(wide_launches[s]), static_cast<long long>(narrow_launches[s]));
        ok = false;
      }
    }
  }
  if (ok) {
    std::fprintf(stderr, "[PASS] %s: %zu observables bit-identical, 64-row vs 256-row%s\n", cfg.name.c_str(),
                 narrow.size(), cfg.expect_wide ? "" : " (256 falls back to 64 here)");
  }
  return ok;
}


// Negative control: the digest DOES see a moved chunk grid. The same 633 tokens fed as 300 + 333 and as
// 333 + 300 (two Prefill calls, so different chunk boundaries) leave different GDN state, hence different
// digests; if they compared equal, "the 256-row state equals the 64-row state" would prove nothing.
bool CheckDigestSensitivity(const Config& cfg) {
  ModelOptions o = cfg.opts;
  o.prefill_chunk = 64;
  Model m = Model::Load(o);
  Trace a, b;
  Config plain = cfg;
  plain.mode = Mode::kPlain;
  RunScenario(m, plain, {"grid", {300, 333}}, &a);
  RunScenario(m, plain, {"grid", {333, 300}}, &b);
  size_t differing = 0;
  for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
    if (a[i].first.find("end/gdn.rec.") != std::string::npos && a[i].second != b[i].second) ++differing;
  }
  if (differing == 0) {
    std::fprintf(stderr, "FAIL %s: the state digest did not change when the chunk grid moved (no sensitivity)\n",
                 cfg.name.c_str());
    return false;
  }
  std::fprintf(stderr, "[PASS] %s: negative control -- a moved chunk grid changes %zu GDN state digests\n",
               cfg.name.c_str(), differing);
  return true;
}
// ---- tensor parallel (both ranks on device 1, --tp-mode emulate) ---------------------------------------
// The same comparison through r4dx::model::TpModel: a 256-row super-chunk's all-reduces are four 64-row
// calls (TpComm::AllReduceSumBf16Rows), so the logits and the tokens must be bit-identical to the 64-row
// TP model's, and the number of all-reduces each rank makes for a scenario must be EQUAL (the same calls,
// not just the same sums). The real two-GPU comparison is tools/prefill (docs/prefill.md "TP = 2").
bool RunTpEmulate(const std::string& name, const ModelOptions& base, Mode mode,
                   const std::vector<Scenario>& scenarios) {
  if (const char* only = std::getenv("R4DX_TEST_ONLY")) {
    if (*only != '\0' && name.find(only) == std::string::npos) return true;
  }
  std::fprintf(stderr, "[chunk-identity] %s\n", name.c_str());
  Trace side[2];
  for (int s = 0; s < 2; ++s) {
    ModelOptions o = base;
    o.prefill_chunk = s == 0 ? 64 : 256;
    r4dx::model::TpOptions t;
    t.world = 2;
    t.mode = r4dx::model::TpOptions::Mode::kEmulate;
    std::unique_ptr<r4dx::model::TpModel> tpm = r4dx::model::TpModel::Load(o, t);
    std::vector<uint64_t> per_scenario_calls;
    for (const Scenario& sc : scenarios) {
      tpm->Reset();
      const auto before = tpm->CallCounts();
      const auto wide_before = tpm->PrefillWideChunksRun();
      int total = 0;
      for (int c : sc.calls) total += c;
      const std::vector<int32_t> ids = Tokens(total, /*salt=*/1);
      size_t off = 0;
      std::vector<float> logits;
      int k = 0;
      for (int c : sc.calls) {
        const std::vector<int32_t> part(ids.begin() + static_cast<ptrdiff_t>(off),
                                        ids.begin() + static_cast<ptrdiff_t>(off + static_cast<size_t>(c)));
        logits = tpm->Prefill(part);
        off += static_cast<size_t>(c);
        side[s].emplace_back(sc.name + "/call" + std::to_string(k++) + "/logits", HashLogits(logits));
      }
      int32_t tok = Argmax(logits);
      std::vector<int32_t> toks;
      for (int i = 0; i < 3; ++i) {
        toks.push_back(tpm->DecodeStepGreedy(tok));
        tok = toks.back();
      }
      if (mode == Mode::kMtp) {
        for (int i = 0; i < 2; ++i) {
          const std::vector<int32_t> round = tpm->DecodeStepMtpGreedy(tok, 3);
          toks.insert(toks.end(), round.begin(), round.end());
          tok = round.back();
        }
      }
      side[s].emplace_back(sc.name + "/tokens", HashTokens(toks));
      const auto after = tpm->CallCounts();
      const auto wide_after = tpm->PrefillWideChunksRun();
      int64_t expect_wide = 0;
      for (int c : sc.calls) expect_wide += c / 256;
      for (size_t r = 0; r < wide_after.size(); ++r) {
        const int64_t got = wide_after[r] - wide_before[r];
        if (got != (s == 1 ? expect_wide : 0)) {
          std::fprintf(stderr, "FAIL %s: %s: TP rank %zu ran %lld super-chunks with prefill_chunk = %d, expected %lld\n",
                       name.c_str(), sc.name.c_str(), r, static_cast<long long>(got), s == 1 ? 256 : 64,
                       static_cast<long long>(s == 1 ? expect_wide : 0));
          side[s].emplace_back(sc.name + "/wide-chunks-wrong", 1);
        }
      }
      for (size_t r = 0; r < after.size(); ++r) {
        for (int ch = 0; ch < 2; ++ch) {
          side[s].emplace_back(sc.name + "/allreduce_calls/rank" + std::to_string(r) + "/ch" + std::to_string(ch),
                               after[r][static_cast<size_t>(ch)] - before[r][static_cast<size_t>(ch)]);
        }
      }
    }
  }
  bool ok = side[0].size() == side[1].size();
  size_t bad = 0;
  for (size_t i = 0; ok && i < side[0].size(); ++i) {
    if (side[0][i] != side[1][i]) {
      if (bad++ < 12) {
        std::fprintf(stderr, "FAIL %s: %s differs: 64-row %016llx, 256-row %016llx\n", name.c_str(),
                     side[0][i].first.c_str(), static_cast<unsigned long long>(side[0][i].second),
                     static_cast<unsigned long long>(side[1][i].second));
      }
    }
  }
  if (bad != 0 || !ok) {
    std::fprintf(stderr, "FAIL %s: %zu of %zu observables differ\n", name.c_str(), bad, side[0].size());
    return false;
  }
  std::fprintf(stderr, "[PASS] %s: %zu observables (logits, tokens, all-reduce call counts) identical, TP=2 64-row vs "
                       "256-row\n", name.c_str(), side[0].size());
  return true;
}

std::vector<Scenario> TailScenarios(bool with_8k) {
  std::vector<Scenario> s;
  for (int n : {1, 63, 64, 65, 255, 256, 257, 511}) s.push_back({"len" + std::to_string(n), {n}});
  if (with_8k) s.push_back({"len8145", {8145}});
  s.push_back({"split300+333", {300, 333}});     // a prefix, then a suffix over one super-chunk
  s.push_back({"split64+511", {64, 511}});       // grid anchored at the second call, not at row 0
  s.push_back({"split257+1", {257, 1}});         // a one-token suffix after a wide + tail call
  s.push_back({"split1+255+257", {1, 255, 257}});  // three calls
  return s;
}

}  // namespace

static int RunTest() {
  int fails = 0;
  int ran = 0;
  if (FileExists(kL4Container)) {
    for (Layout layout : {Layout::kBf16, Layout::kW4a16}) {
      const std::string ln = layout == Layout::kBf16 ? "bf16" : "w4a16";
      ModelOptions base;
      base.container_path = kL4Container;
      base.layout = layout;
      base.max_ctx = 8448;
      base.layer_limit = 4;
      const std::vector<Scenario> tails = TailScenarios(/*with_8k=*/true);

      Config plain{"l4/" + ln + "/plain", base, Mode::kPlain, 0, /*expect_wide=*/true, tails};
      Config capture{"l4/" + ln + "/capture", base, Mode::kCapture, 0, /*expect_wide=*/true, TailScenarios(false)};
      ModelOptions mo = base;
      mo.mtp_draft_k = 3;
      Config mtp{"l4/" + ln + "/mtp3", mo, Mode::kMtp, 3, /*expect_wide=*/true, TailScenarios(false)};
      for (const Config* c : {&plain, &capture, &mtp}) {
        ++ran;
        if (!RunConfig(*c)) ++fails;
      }
      if (layout == Layout::kBf16 && !CheckDigestSensitivity(plain)) ++fails;
    }
  } else {
    std::fprintf(stderr, "[SKIP] %s not found -- the 4-layer part did not run\n", kL4Container);
  }

  if (FileExists(kL4Container)) {
    // TP = 2, both ranks on device 1 (--tp-mode emulate): the 4-layer container, plain and --mtp 3.
    ModelOptions t;
    t.container_path = kL4Container;
    t.layout = Layout::kW4a16;
    t.max_ctx = 1024;
    t.layer_limit = 4;
    t.vision = ModelOptions::VisionMode::kOff;
    const std::vector<Scenario> tp_sc = {{"len64", {64}},   {"len255", {255}}, {"len256", {256}},
                                         {"len257", {257}}, {"len511", {511}}, {"len600", {600}},
                                         {"split300+333", {300, 333}}, {"split257+1", {257, 1}}};
    ++ran;
    if (!RunTpEmulate("tp2-emulate/l4/w4a16/plain", t, Mode::kPlain, tp_sc)) ++fails;
    t.mtp_draft_k = 3;
    ++ran;
    if (!RunTpEmulate("tp2-emulate/l4/w4a16/mtp3", t, Mode::kMtp, tp_sc)) ++fails;
  }

  const char* real = r4dx_test::ProductionTargetPath();
  const char* drafter = r4dx_test::ProductionDrafterPath();
  if (FileExists(real)) {
    ModelOptions base;
    base.container_path = real;
    base.layout = r4dx::model::LayoutFromName(r4dx_test::ProductionLayoutName());
    base.max_ctx = 2048;
    std::vector<Scenario> sc;
    for (int n : {63, 64, 255, 256, 257, 511, 1100}) sc.push_back({"len" + std::to_string(n), {n}});
    sc.push_back({"split300+333", {300, 333}});
    sc.push_back({"split257+1", {257, 1}});
    Config plain{"real/plain", base, Mode::kPlain, 0, /*expect_wide=*/true, sc};
    ++ran;
    if (!RunConfig(plain)) ++fails;

    std::vector<Scenario> spec_sc = {{"len257", {257}}, {"len511", {511}}, {"split300+333", {300, 333}}};
    ModelOptions mo = base;
    mo.mtp_draft_k = 3;
    Config mtp{"real/mtp3", mo, Mode::kMtp, 3, /*expect_wide=*/true, spec_sc};
    ++ran;
    if (!RunConfig(mtp)) ++fails;

    {
      // TP = 2 emulated on one device, real trellis container: the M = 256 GEMM at the rank-shard shapes.
      ModelOptions t = base;
      t.vision = ModelOptions::VisionMode::kOff;
      const std::vector<Scenario> tp_sc = {{"len257", {257}}, {"len511", {511}}, {"split300+333", {300, 333}}};
      ++ran;
      if (!RunTpEmulate("tp2-emulate/real/plain", t, Mode::kPlain, tp_sc)) ++fails;
    }
    if (FileExists(drafter)) {
      ModelOptions dof = base;
      dof.dflash_container = drafter;
      dof.dflash_draft_k = 7;
      Config dflash{"real/dflash7", dof, Mode::kDflash, 7, /*expect_wide=*/true, spec_sc};
      ++ran;
      if (!RunConfig(dflash)) ++fails;
    } else {
      std::fprintf(stderr, "[SKIP] %s not found -- the DFlash part did not run\n", drafter);
    }
  } else {
    std::fprintf(stderr, "[SKIP] %s not found -- the real-container part did not run\n", real);
  }
  if (ran == 0) return SkipMissing(kL4Container);
  if (fails != 0) {
    std::fprintf(stderr, "test_prefill_chunk_identity: %d of %d configurations FAILED\n", fails, ran);
    return 1;
  }
  std::fprintf(stderr, "test_prefill_chunk_identity: PASS (%d configurations)\n", ran);
  return 0;
}

int main() { return r4dx_test::RunGuardedMain("test_prefill_chunk_identity", RunTest); }
