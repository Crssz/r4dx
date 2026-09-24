// tests/model/test_tp_emulation.cpp -- tensor parallel P2b (docs/tp.md 10.1): the TP=2 facade
// (r4dx::model::TpModel) in --tp-mode emulate -- both ranks on ONE device, EmulatedComm's
// host-synchronized exact add at every all-reduce -- against the TP=1 Model, on the 4-layer
// container qwen38-27b-l4-allmtp.r4dx, all four layouts:
//
//   1. numerics: prefill 40 and 70 tokens (one and two chunks) + 16 teacher-forced DecodeStep rows,
//      per-row relative L2 of the logits vs TP=1 <= 1e-2 (bf16, w4a16) / 5e-2 (mxfp4); w4a8 against
//      the TP=1 bf16 yardstick instead (see TestLayout, docs/tp.md Appendix B N50);
//   2. DecodeStepGreedy == the argmax of the DecodeStep row it replaces, exactly;
//   3. DecodeStepSampled vs DecodeStep + the full-vocab canonical sampler on the SAME TpModel, three
//      filter configs x three seeds (plus one sub-0.01 temperature), trajectories exactly equal --
//      the merged device row summaries of docs/tp.md 7.4, end to end;
//   4. Reset() then the same script again == the first run, byte for byte.
//
// On w4a16 (the production layout) additionally:
//   5. VerifyWindow under TP (docs/tp.md 7.6, the H4/H5 merges): both ranks return identical
//      results; every merged greedy pred == the argmax of its gathered full row; every merged row
//      summary == r4dx_topk_lse_f32 over the gathered full row (ids / vals exact, lse <= 1e-4);
//      ReadVerifyLogitsRow == the gathered row; CommitVerifiedWindow keeps the ranks in step;
//   6. fault injection (docs/tp.md 2.4, 2.5): rank 1 throws at all-reduce #37 (armed through
//      TpOptions, counted from the end of warm-up) -- the injected exception reaches the caller, the
//      next forward call throws TpStateError, SetDflashInjectionEnabled and every cached accessor
//      still work in kNeedsRecovery, Reset() recovers, both endpoints report equal CallCounts(), the
//      injection policy set in kNeedsRecovery is the one the ranks run with, and the rerun equals a
//      fresh TpModel's run byte for byte; then the same with an asymmetric fault (kind 1: rank 1
//      stalls 700 ms, rank 0's all-reduce times out); then a lockstep divergence (the ranks feed
//      different tokens: TpDivergenceError, kNeedsRecovery, Reset() recovers, rerun == fresh); last,
//      a recovery that itself fails (fault armed into recovery's SelfTest): kFatal, and every
//      device-work call and Reset() then throw TpStateError while host-only calls keep working.
//
// Every case ends with core::g_tp_collective_allocs == 0 (docs/tp.md 2.7: no device allocation or
// free inside a collective command after warm-up).
//
// Device 1 (HIP_VISIBLE_DEVICES=1, from CMake), SKIP 77 without the container.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "local_text_model.h"
#include "model.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_alloc_guard.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/kernels/sampler.hpp"
#include "r4dx/kernels/summary_sampler.hpp"
#include "test_common.h"
#include "tp_model.h"

namespace {

using r4dx::model::LocalTextModel;
using r4dx::model::Model;
using r4dx::model::ModelOptions;
using r4dx::model::TextModel;
using r4dx::model::TpModel;
using r4dx::model::TpOptions;
namespace core = r4dx::core;
namespace kernels = r4dx::kernels;

const char* kContainerPath = r4dx_test::ContainerPath("D:/models/r4dx/qwen38-27b-l4-allmtp.r4dx");
constexpr int64_t kLayers = 4;
constexpr int kDecodeRows = 16;

int g_failures = 0;

#define CHECK(cond, ...)                                              \
  do {                                                                \
    if (!(cond)) {                                                    \
      std::fprintf(stderr, "FAIL %s:%d: %s -- ", __FILE__, __LINE__, #cond); \
      std::fprintf(stderr, __VA_ARGS__);                              \
      std::fprintf(stderr, "\n");                                     \
      ++g_failures;                                                   \
    }                                                                 \
  } while (0)

// Deterministic in-vocabulary token ids (the test asserts engine agreement, not meaning).
std::vector<int32_t> Tokens(size_t n, uint32_t seed) {
  std::vector<int32_t> v(n);
  uint32_t x = seed;
  for (size_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    v[i] = static_cast<int32_t>(1000u + (x >> 8) % 150000u);
  }
  return v;
}

ModelOptions BaseOptions(const std::string& layout) {
  ModelOptions o;
  o.container_path = kContainerPath;
  o.layout = r4dx::model::LayoutFromName(layout);
  o.max_ctx = 512;
  o.layer_limit = kLayers;
  o.vision = ModelOptions::VisionMode::kOff;
  return o;
}

TpOptions EmulateOptions() {
  TpOptions t;
  t.world = 2;
  t.mode = TpOptions::Mode::kEmulate;
  return t;
}

using Rows = std::vector<std::vector<float>>;

// Reset; Prefill(prompt) -> row 0; DecodeStep(forced[i]) -> row i+1.
Rows Script(TextModel& m, const std::vector<int32_t>& prompt, const std::vector<int32_t>& forced) {
  Rows rows;
  m.Reset();
  rows.push_back(m.Prefill(prompt));
  for (int i = 0; i < kDecodeRows; ++i) rows.push_back(m.DecodeStep(forced[static_cast<size_t>(i)]));
  return rows;
}

double RelL2(const std::vector<float>& ref, const std::vector<float>& got) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = static_cast<double>(got[i]) - static_cast<double>(ref[i]);
    num += d * d;
    den += static_cast<double>(ref[i]) * static_cast<double>(ref[i]);
  }
  return den > 0 ? std::sqrt(num / den) : std::sqrt(num);
}

bool SameBytes(const Rows& a, const Rows& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i].size() != b[i].size()) return false;
    if (std::memcmp(a[i].data(), b[i].data(), a[i].size() * sizeof(float)) != 0) return false;
  }
  return true;
}

bool SameSummary(const kernels::RowSummary& a, const kernels::RowSummary& b) {
  if (a.k != b.k || a.vocab != b.vocab || std::memcmp(&a.inv_temperature, &b.inv_temperature, 4) != 0 ||
      std::memcmp(&a.lse, &b.lse, 4) != 0) {
    return false;
  }
  return std::memcmp(a.ids, b.ids, sizeof(int32_t) * static_cast<size_t>(a.k)) == 0 &&
         std::memcmp(a.vals, b.vals, sizeof(float) * static_cast<size_t>(a.k)) == 0;
}

void CheckNoCollectiveAllocs(const char* where) {
  const uint64_t n = core::g_tp_collective_allocs.load();
  CHECK(n == 0, "%s: %llu device allocation(s)/free(s) inside collective commands", where,
        static_cast<unsigned long long>(n));
}

struct SampleConfig {
  const char* name;
  float temperature;
  int top_k;
  float top_p;
  float min_p;
};

// Plain sampled decode two ways on one TpModel: DecodeStepSampled (merged device summaries) vs
// DecodeStep + kernels::Sample (full-vocab canonical sampler on the gathered row). One draw per
// token either way, so the trajectories must be identical.
void CheckSampled(TpModel& tpm, const std::string& layout, const std::vector<int32_t>& prompt) {
  const SampleConfig configs[] = {
      {"T0.7/top_k20/top_p0.8", 0.7f, 20, 0.8f, 0.0f},
      {"T1.0 pure temperature", 1.0f, 0, 1.0f, 0.0f},
      {"T0.9/top_p0.95/min_p0.02", 0.9f, 0, 0.95f, 0.02f},
  };
  const uint64_t seeds[] = {1, 7, 12345};
  constexpr int kSteps = 24;
  const int64_t V = tpm.Config().vocab_size;
  const auto run = [&](const kernels::SampleParams& sp, uint64_t seed, bool summaries) {
    std::vector<int32_t> out;
    std::mt19937_64 rng = kernels::MakeRng(seed);
    tpm.Reset();
    std::vector<float> logits = tpm.Prefill(prompt);
    int32_t next = kernels::Sample(logits.data(), V, sp, rng);
    for (int s = 0; s < kSteps; ++s) {
      out.push_back(next);
      if (summaries) {
        next = tpm.DecodeStepSampled(next, sp, rng);
      } else {
        logits = tpm.DecodeStep(next);
        next = kernels::Sample(logits.data(), V, sp, rng);
      }
    }
    out.push_back(next);
    return out;
  };
  for (const SampleConfig& c : configs) {
    for (uint64_t seed : seeds) {
      kernels::SampleParams sp;
      sp.temperature = c.temperature;
      sp.top_k = c.top_k;
      sp.top_p = c.top_p;
      sp.min_p = c.min_p;
      sp.seed = seed;
      const int64_t fb0 = tpm.SampledFallbackRows();
      const std::vector<int32_t> a = run(sp, seed, /*summaries=*/true);
      const int64_t fb = tpm.SampledFallbackRows() - fb0;
      const std::vector<int32_t> b = run(sp, seed, /*summaries=*/false);
      size_t first = 0;
      while (first < a.size() && a[first] == b[first]) ++first;
      CHECK(a == b, "[%s] sampled %s seed %llu: DecodeStepSampled diverges from DecodeStep+Sample at token %zu",
            layout.c_str(), c.name, static_cast<unsigned long long>(seed), first);
      std::printf("[%s] sampled %-26s seed %5llu: %zu tokens equal, %lld fallback row(s)\n", layout.c_str(), c.name,
                  static_cast<unsigned long long>(seed), a.size(), static_cast<long long>(fb));
    }
  }
  // Below kMinSummaryTemperature (0.01) the device summary is skipped: DecodeStep's gathered row +
  // the canonical sampler, still one draw per token.
  kernels::SampleParams tiny;
  tiny.temperature = 0.005f;
  const std::vector<int32_t> a = run(tiny, 3, true), b = run(tiny, 3, false);
  CHECK(a == b, "[%s] sampled T=0.005: DecodeStepSampled diverges from DecodeStep+Sample", layout.c_str());
}

// The TP=1 bf16 runs of both scripts (bf16 is the first layout tested): the common yardstick for how
// far each quantized layout sits from exact arithmetic, at TP=1 and under TP.
Rows g_bf16_ref[2];

// Numerics vs TP=1, greedy == argmax, sampled == canonical, Reset rerun byte identity. Returns the
// TP run of the 40-token script (the fault test's fresh-run reference for w4a16).
//
// The numerics gate is docs/tp.md 10.1's per-row relative L2 vs TP=1: <= 1e-2 for bf16 and w4a16,
// <= 5e-2 for mxfp4. w4a8 is gated differently (docs/tp.md Appendix B N50): its int8 activation
// scales are taken per row over the rank's LOCAL K on the row-parallel layers (docs/tp.md 4.3), and
// on this container that alone moves the logits ~8% from TP=1 -- while moving them CLOSER to the
// bf16 run, not further. So for w4a8 the TP run must (a) stay within 1e-1 of TP=1 per row and (b)
// be no further from the TP=1 bf16 run than TP=1 w4a8 itself is (max over the script's rows; ratio
// <= 1.0, measured 0.86 / 0.82 -- an independent TP-only error of ~5e-2 rel L2 on top of the
// measured one already fails it, docs/tp.md Appendix B N50/N53).
Rows TestLayout(const std::string& layout) {
  std::printf("==== layout %s ====\n", layout.c_str());
  const std::vector<int32_t> p40 = Tokens(40, 11), p70 = Tokens(70, 23), forced = Tokens(kDecodeRows, 37);
  const bool yardstick_gate = layout == "w4a8";
  const double tol = (layout == "bf16" || layout == "w4a16") ? 1e-2 : (yardstick_gate ? 1e-1 : 5e-2);

  Rows ref[2];
  {
    LocalTextModel m(Model::Load(BaseOptions(layout)));
    ref[0] = Script(m, p40, forced);
    ref[1] = Script(m, p70, forced);
  }
  if (layout == "bf16") {
    g_bf16_ref[0] = ref[0];
    g_bf16_ref[1] = ref[1];
  }

  std::unique_ptr<TpModel> tpm = TpModel::Load(BaseOptions(layout), EmulateOptions());
  CHECK(tpm->Config().vocab_size == 248320, "[%s] TpModel::Config() must be the GLOBAL config", layout.c_str());
  Rows tp[2];
  tp[0] = Script(*tpm, p40, forced);
  tp[1] = Script(*tpm, p70, forced);

  // 1. numerics vs TP=1 (and, printed for every layout, each side's distance from TP=1 bf16)
  for (int which = 0; which < 2; ++which) {
    const Rows& r = ref[which];
    const Rows& t = tp[which];
    const int plen = which == 0 ? 40 : 70;
    double worst = 0.0, tp1_vs_bf16 = 0.0, tp_vs_bf16 = 0.0;
    for (size_t i = 0; i < r.size(); ++i) {
      CHECK(t[i].size() == r[i].size(), "[%s] row %zu: %zu logits vs %zu", layout.c_str(), i, t[i].size(),
            r[i].size());
      if (t[i].size() != r[i].size()) continue;
      const double e = RelL2(r[i], t[i]);
      worst = std::max(worst, e);
      CHECK(e <= tol, "[%s] prompt %d row %zu: rel L2 %.3e > %.0e", layout.c_str(), plen, i, e, tol);
      if (!g_bf16_ref[which].empty()) {
        tp1_vs_bf16 = std::max(tp1_vs_bf16, RelL2(g_bf16_ref[which][i], r[i]));
        tp_vs_bf16 = std::max(tp_vs_bf16, RelL2(g_bf16_ref[which][i], t[i]));
      }
    }
    std::printf("[%s] prompt %d (+%d decode rows): max rel L2 vs TP=1 = %.3e (tol %.0e); vs TP=1 bf16: TP=1 %.3e, "
                "TP=2 emulated %.3e (ratio %.3f)\n",
                layout.c_str(), plen, kDecodeRows, worst, tol, tp1_vs_bf16, tp_vs_bf16,
                tp1_vs_bf16 > 0 ? tp_vs_bf16 / tp1_vs_bf16 : 0.0);
    if (yardstick_gate) {
      CHECK(!g_bf16_ref[which].empty() && tp_vs_bf16 <= tp1_vs_bf16,
            "[%s] prompt %d: TP=2 sits %.3e from TP=1 bf16, TP=1 %s only %.3e (limit x1.0)", layout.c_str(), plen,
            tp_vs_bf16, layout.c_str(), tp1_vs_bf16);
    }
  }
  const Rows& tp40 = tp[0];
  const Rows& tp70 = tp[1];

  // 4. Reset() then rerun == the first run, byte for byte
  const Rows tp70b = Script(*tpm, p70, forced);
  CHECK(SameBytes(tp70, tp70b), "[%s] Reset() + rerun is not byte-identical to the first run", layout.c_str());

  // 2. DecodeStepGreedy == argmax of the DecodeStep row
  {
    tpm->Reset();
    (void)tpm->Prefill(p40);
    int mismatches = 0;
    for (int i = 0; i < kDecodeRows; ++i) {
      const int32_t g = tpm->DecodeStepGreedy(forced[static_cast<size_t>(i)]);
      const std::vector<float>& row = tp40[static_cast<size_t>(i + 1)];
      const int32_t want = kernels::Argmax(row.data(), static_cast<int64_t>(row.size()));
      if (g != want) ++mismatches;
    }
    CHECK(mismatches == 0, "[%s] DecodeStepGreedy != argmax(DecodeStep) on %d of %d rows", layout.c_str(),
          mismatches, kDecodeRows);
    std::printf("[%s] DecodeStepGreedy == argmax(DecodeStep): %d/%d rows\n", layout.c_str(), kDecodeRows - mismatches,
                kDecodeRows);
  }

  // 3. sampled
  CheckSampled(*tpm, layout, p40);

  CHECK(tpm->GetState() == TpModel::State::kReady, "[%s] group not ready at the end", layout.c_str());
  CheckNoCollectiveAllocs(layout.c_str());
  return tp40;
}

// 5. VerifyWindow / ReadVerifyLogitsRow / CommitVerifiedWindow under TP (docs/tp.md 7.6).
void TestVerifyWindow() {
  std::printf("==== VerifyWindow under TP (w4a16) ====\n");
  ModelOptions o = BaseOptions("w4a16");
  o.dflash_draft_k = 7;  // sizes an 8-row verify window; no drafter is loaded
  std::unique_ptr<TpModel> tpm = TpModel::Load(o, EmulateOptions());
  const std::vector<int32_t> p40 = Tokens(40, 11), forced = Tokens(kDecodeRows, 37);
  const int64_t V = tpm->Config().vocab_size;

  tpm->Reset();
  const std::vector<float> lg = tpm->Prefill(p40);
  std::vector<int32_t> cands = {kernels::Argmax(lg.data(), V)};
  for (int i = 0; i < 7; ++i) cands.push_back(forced[static_cast<size_t>(i)]);
  const int64_t T = static_cast<int64_t>(cands.size());
  const float inv_t = 1.0f / 0.7f;

  struct RankOut {
    std::vector<int32_t> preds;
    std::vector<float> logits;
    std::vector<kernels::RowSummary> sums;
    std::vector<float> row3;
  };
  std::vector<RankOut> outs(2);
  tpm->RunCollectiveForTest([&](Model& m, int rank) {
    RankOut& ro = outs[static_cast<size_t>(rank)];
    ro.preds = m.VerifyWindow(cands, &ro.logits, &ro.sums, inv_t);
    m.ReadVerifyLogitsRow(3, ro.row3);
  });
  // Both ranks hold identical merged results.
  CHECK(outs[0].preds == outs[1].preds, "VerifyWindow preds differ between ranks");
  CHECK(outs[0].logits.size() == static_cast<size_t>(T * V) && outs[1].logits.size() == outs[0].logits.size() &&
            std::memcmp(outs[0].logits.data(), outs[1].logits.data(), outs[0].logits.size() * 4) == 0,
        "VerifyWindow logits_out differs between ranks (or has the wrong size)");
  CHECK(outs[0].sums.size() == static_cast<size_t>(T) && outs[1].sums.size() == static_cast<size_t>(T),
        "VerifyWindow summaries_out has the wrong size");
  for (int64_t t = 0; t < T && outs[0].sums.size() == static_cast<size_t>(T); ++t) {
    CHECK(SameSummary(outs[0].sums[static_cast<size_t>(t)], outs[1].sums[static_cast<size_t>(t)]),
          "row %lld summary differs between ranks", static_cast<long long>(t));
  }
  const RankOut& r = outs[0];
  // Merged greedy pred == argmax of the gathered full row (lowest index on ties).
  for (int64_t t = 0; t < T; ++t) {
    const int32_t want = kernels::Argmax(r.logits.data() + t * V, V);
    CHECK(r.preds[static_cast<size_t>(t)] == want, "row %lld: merged pred %d, argmax of the gathered row %d",
          static_cast<long long>(t), r.preds[static_cast<size_t>(t)], want);
  }
  // ReadVerifyLogitsRow == the gathered row.
  CHECK(r.row3.size() == static_cast<size_t>(V) &&
            std::memcmp(r.row3.data(), r.logits.data() + 3 * V, static_cast<size_t>(V) * 4) == 0,
        "ReadVerifyLogitsRow(3) differs from VerifyWindow's gathered row 3");
  // Merged summary == r4dx_topk_lse_f32 over the gathered full row (on this thread's device, with the
  // kernels' module scratch: every rank is idle between commands).
  {
    core::Stream st;
    core::DeviceBuffer<float> dl(static_cast<size_t>(T * V));
    dl.CopyFromHost(r.logits.data(), r.logits.size());
    core::DeviceBuffer<int32_t> ids(static_cast<size_t>(T) * R4DX_TOPK_LSE_K);
    core::DeviceBuffer<float> vals(static_cast<size_t>(T) * R4DX_TOPK_LSE_K), lse(static_cast<size_t>(T));
    r4dx_topk_lse_f32(reinterpret_cast<int64_t>(dl.data()), reinterpret_cast<int64_t>(ids.data()),
                      reinterpret_cast<int64_t>(vals.data()), reinterpret_cast<int64_t>(lse.data()),
                      static_cast<int>(T), V, inv_t, reinterpret_cast<int64_t>(st.get()));
    st.Synchronize();
    const std::vector<int32_t> hid = ids.CopyToHost();
    const std::vector<float> hval = vals.CopyToHost(), hlse = lse.CopyToHost();
    double worst_lse = 0.0;
    for (int64_t t = 0; t < T; ++t) {
      const kernels::RowSummary& s = r.sums[static_cast<size_t>(t)];
      CHECK(s.k == R4DX_TOPK_LSE_K && s.vocab == V, "row %lld: merged summary k %d vocab %lld",
            static_cast<long long>(t), s.k, static_cast<long long>(s.vocab));
      bool exact = true;
      for (int j = 0; j < R4DX_TOPK_LSE_K; ++j) {
        const size_t at = static_cast<size_t>(t) * R4DX_TOPK_LSE_K + static_cast<size_t>(j);
        if (s.ids[j] != hid[at] || std::memcmp(&s.vals[j], &hval[at], 4) != 0) exact = false;
      }
      CHECK(exact, "row %lld: merged top-64 differs from r4dx_topk_lse_f32 over the gathered row",
            static_cast<long long>(t));
      const double e = std::fabs(static_cast<double>(s.lse) - static_cast<double>(hlse[static_cast<size_t>(t)]));
      worst_lse = std::max(worst_lse, e);
      CHECK(e <= 1e-4, "row %lld: merged lse %.7f vs full-row %.7f", static_cast<long long>(t), s.lse,
            hlse[static_cast<size_t>(t)]);
    }
    std::printf("[verify] %lld rows: preds == argmax(gathered), top-64 exact, max |lse delta| %.3e\n",
                static_cast<long long>(T), worst_lse);
  }
  // A greedy verify + commit keeps both ranks in step (the facade and H1/H2 would catch a drift).
  {
    std::vector<std::vector<int32_t>> preds(2);
    tpm->RunCollectiveForTest([&](Model& m, int rank) {
      preds[static_cast<size_t>(rank)] = m.VerifyWindow(cands);
      m.CommitVerifiedWindow(3);
    });
    CHECK(preds[0] == preds[1], "greedy VerifyWindow preds differ between ranks");
    CHECK(tpm->PositionCount() == 43, "PositionCount after CommitVerifiedWindow(3): %lld",
          static_cast<long long>(tpm->PositionCount()));
    const std::vector<float> next = tpm->DecodeStep(cands[3]);
    CHECK(static_cast<int64_t>(next.size()) == V && tpm->PositionCount() == 44, "decode after a verified commit");
    kernels::SampleParams sp;
    sp.temperature = 0.8f;
    sp.top_k = 40;
    std::mt19937_64 rng = kernels::MakeRng(5);
    (void)tpm->DecodeStepSampled(cands[4], sp, rng);  // sampled decode on a draft-window-8 rank
  }
  CHECK(tpm->GetState() == TpModel::State::kReady, "group not ready after the VerifyWindow checks");
  CheckNoCollectiveAllocs("verify");
}

// Runs the 40-token script, expecting the armed fault to break it; checks the state machine and
// recovery; returns true if the post-recovery rerun equals `fresh` byte for byte.
void CheckFaultCycle(TpModel& tpm, const Rows& fresh, const char* label, bool expect_timeout) {
  const std::vector<int32_t> p40 = Tokens(40, 11), forced = Tokens(kDecodeRows, 37);
  bool threw = false;
  std::string what;
  try {
    (void)Script(tpm, p40, forced);
  } catch (const core::TpStateError& e) {
    threw = true;
    what = std::string("TpStateError: ") + e.what();
  } catch (const std::exception& e) {
    threw = true;
    what = e.what();
  }
  CHECK(threw, "[%s] the injected fault did not reach the caller", label);
  std::printf("[%s] caller saw: %s\n", label, what.c_str());
  if (expect_timeout) {
    CHECK(what.find("timeout") != std::string::npos, "[%s] expected the peer's all-reduce timeout, got: %s", label,
          what.c_str());
  } else {
    CHECK(what.find("tp fault injection") != std::string::npos, "[%s] expected the injected exception, got: %s",
          label, what.c_str());
  }
  CHECK(tpm.GetState() == TpModel::State::kNeedsRecovery, "[%s] state after the fault is not kNeedsRecovery", label);

  // The next device-work call throws TpStateError.
  bool state_error = false;
  try {
    (void)tpm.DecodeStep(forced[0]);
  } catch (const core::TpStateError&) {
    state_error = true;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "[%s] unexpected: %s\n", label, e.what());
  }
  CHECK(state_error, "[%s] a forward call in kNeedsRecovery must throw TpStateError", label);

  // Host-only calls keep working in kNeedsRecovery (docs/tp.md 2.4). The injection policy is left
  // OFF here: after Reset() every rank must run with it (checked below).
  try {
    tpm.SetDflashInjectionEnabled(true);
    tpm.SetDflashInjectionEnabled(false);
    CHECK(tpm.Config().vocab_size == 248320, "[%s] Config() in kNeedsRecovery", label);
    CHECK(!tpm.ModelId().empty(), "[%s] ModelId() in kNeedsRecovery", label);
    CHECK(tpm.ImageTokenId() > 0, "[%s] ImageTokenId() in kNeedsRecovery", label);
    CHECK(tpm.VisionMergeSize() > 0, "[%s] VisionMergeSize() in kNeedsRecovery", label);
    CHECK(!tpm.HasVision() && !tpm.MtpEnabled() && !tpm.MtpUsingReducedVocabDraft() && !tpm.DflashEnabled(),
          "[%s] capability accessors in kNeedsRecovery", label);
    CHECK(tpm.NumLoadedLayers() == kLayers && tpm.TpWorld() == 2, "[%s] NumLoadedLayers/TpWorld", label);
    (void)tpm.PositionCount();
    (void)tpm.SampledFallbackRows();
    const auto vram = tpm.Vram();
    CHECK(vram.size() == 2 && vram[0].total_gib > 0, "[%s] Vram() in kNeedsRecovery", label);
  } catch (const std::exception& e) {
    CHECK(false, "[%s] a host-only call threw in kNeedsRecovery: %s", label, e.what());
  }
  CHECK(tpm.GetState() == TpModel::State::kNeedsRecovery, "[%s] host-only calls changed the state", label);

  // Reset() recovers.
  try {
    tpm.Reset();
  } catch (const std::exception& e) {
    CHECK(false, "[%s] Reset() did not recover: %s", label, e.what());
    return;
  }
  CHECK(tpm.GetState() == TpModel::State::kReady, "[%s] state after Reset() is not kReady", label);
  const auto counts = tpm.CallCounts();
  CHECK(counts.size() == 2 && counts[0] == counts[1], "[%s] CallCounts differ after recovery: {%llu,%llu} vs {%llu,%llu}",
        label, static_cast<unsigned long long>(counts[0][0]), static_cast<unsigned long long>(counts[0][1]),
        static_cast<unsigned long long>(counts[1][0]), static_cast<unsigned long long>(counts[1][1]));
  // The policy stored in kNeedsRecovery survived recovery and is the one every rank runs with.
  {
    int inj[2] = {-1, -1};
    tpm.RunCollectiveForTest([&](Model& m, int rank) { inj[rank] = m.DflashInjectionEnabled() ? 1 : 0; });
    CHECK(inj[0] == 0 && inj[1] == 0, "[%s] SetDflashInjectionEnabled(false) in kNeedsRecovery: ranks run with %d/%d",
          label, inj[0], inj[1]);
    tpm.SetDflashInjectionEnabled(true);
  }
  const Rows rerun = Script(tpm, p40, forced);
  CHECK(SameBytes(rerun, fresh), "[%s] the rerun after recovery is not byte-identical to a fresh run", label);
  std::printf("[%s] recovered; CallCounts {%llu,%llu} on both ranks; rerun == fresh run: %s\n", label,
              static_cast<unsigned long long>(counts[0][0]), static_cast<unsigned long long>(counts[0][1]),
              SameBytes(rerun, fresh) ? "yes" : "NO");
}

// 6. fault injection and recovery (w4a16).
void TestFaults(const Rows& fresh40) {
  std::printf("==== fault injection + recovery (w4a16) ====\n");
  TpOptions t = EmulateOptions();
  t.fault_rank = 1;
  t.fault_at_allreduce = 37;
  t.fault_kind = 0;  // throw
  std::unique_ptr<TpModel> tpm = TpModel::Load(BaseOptions("w4a16"), t);
  CheckFaultCycle(*tpm, fresh40, "fault kind 0 (rank 1 throws at AR #37)", /*expect_timeout=*/false);
  // Asymmetric: rank 1 stalls 700 ms before its all-reduce, rank 0 times out at 500 ms.
  tpm->ArmFaultInjection(1, 37, 1);
  CheckFaultCycle(*tpm, fresh40, "fault kind 1 (rank 1 stalls at AR #37)", /*expect_timeout=*/true);

  const std::vector<int32_t> p40 = Tokens(40, 11), forced = Tokens(kDecodeRows, 37);
  CHECK(forced[0] != forced[1], "the divergence case needs two different tokens");
  // The ranks feed different tokens to one DecodeStep: H1's token hash differs, both ranks throw
  // TpDivergenceError before any all-reduce (docs/tp.md 2.3, 2.4, 6.2).
  const auto diverge = [&](const char* label) {
    bool div = false;
    std::string what;
    try {
      tpm->RunCollectiveForTest(
          [&](Model& m, int rank) { (void)m.DecodeStep(forced[static_cast<size_t>(rank)]); });
    } catch (const core::TpDivergenceError& e) {
      div = true;
      what = e.what();
    } catch (const std::exception& e) {
      what = std::string("WRONG TYPE: ") + e.what();
    }
    CHECK(div, "[%s] expected TpDivergenceError, got: %s", label, what.c_str());
    CHECK(tpm->GetState() == TpModel::State::kNeedsRecovery, "[%s] state after a divergence is not kNeedsRecovery",
          label);
    std::printf("[%s] caller saw TpDivergenceError: %s\n", label, what.c_str());
  };
  {
    tpm->Reset();
    (void)tpm->Prefill(p40);
    diverge("divergence");
    try {
      tpm->Reset();
    } catch (const std::exception& e) {
      CHECK(false, "[divergence] Reset() did not recover: %s", e.what());
    }
    CHECK(tpm->GetState() == TpModel::State::kReady, "[divergence] state after Reset() is not kReady");
    if (tpm->GetState() == TpModel::State::kReady) {
      const Rows rerun = Script(*tpm, p40, forced);
      CHECK(SameBytes(rerun, fresh40), "[divergence] the rerun after recovery is not byte-identical to a fresh run");
      std::printf("[divergence] recovered; rerun == fresh run: %s\n", SameBytes(rerun, fresh40) ? "yes" : "NO");
    }
  }
  CheckNoCollectiveAllocs("faults");

  // A recovery that fails makes the group kFatal (docs/tp.md 2.4). Rank 1 is armed to throw at its
  // 5th all-reduce counted from now; the divergence issues none (H1 precedes the layers), and
  // recovery restarts the fault counter (2.5 step 6) before its SelfTest -- whose 5th all-reduce
  // then throws on rank 1 (rank 0 times out in the same call).
  if (tpm->GetState() == TpModel::State::kReady) {
    tpm->Reset();
    (void)tpm->Prefill(p40);
    tpm->ArmFaultInjection(1, 5, 0);
    diverge("fatal: divergence");
    bool reset_threw = false;
    std::string what;
    try {
      tpm->Reset();
    } catch (const std::exception& e) {
      reset_threw = true;
      what = e.what();
    }
    CHECK(reset_threw && what.find("tp fault injection") != std::string::npos,
          "[fatal] Reset() must rethrow the failed recovery's root cause, got: %s", what.c_str());
    CHECK(tpm->GetState() == TpModel::State::kFatal, "[fatal] state after a failed recovery is not kFatal");
    const auto state_error = [&](const char* call, const std::function<void()>& f) {
      bool se = false;
      try {
        f();
      } catch (const core::TpStateError& e) {
        se = std::string(e.what()).find("fatal") != std::string::npos;
      } catch (const std::exception& e) {
        std::fprintf(stderr, "[fatal] %s: unexpected %s\n", call, e.what());
      }
      CHECK(se, "[fatal] %s in kFatal must throw TpStateError(\"tp: fatal, ...\")", call);
    };
    state_error("Reset()", [&] { tpm->Reset(); });
    state_error("DecodeStep()", [&] { (void)tpm->DecodeStep(forced[0]); });
    state_error("CallCounts()", [&] { (void)tpm->CallCounts(); });
    try {
      tpm->SetDflashInjectionEnabled(true);
      CHECK(tpm->Config().vocab_size == 248320 && tpm->TpWorld() == 2, "[fatal] cached accessors in kFatal");
      CHECK(tpm->Vram().size() == 2, "[fatal] Vram() in kFatal returns the last report");
    } catch (const std::exception& e) {
      CHECK(false, "[fatal] a host-only call threw in kFatal: %s", e.what());
    }
    std::printf("[fatal] failed recovery -> kFatal (%s); device-work calls and Reset() refuse, host-only calls work\n",
                what.c_str());
  }
  // ~TpModel in kFatal: the group is poisoned, then every rank drains and tears down normally.
}

int RunTest() {
  if (!r4dx_test::FileExists(kContainerPath)) return r4dx_test::SkipMissing(kContainerPath);
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep stdout in order with the facade's stderr lines
  Rows fresh40_w4a16;
  for (const char* layout : {"bf16", "w4a16", "w4a8", "mxfp4"}) {
    Rows r = TestLayout(layout);
    if (std::string(layout) == "w4a16") fresh40_w4a16 = std::move(r);
  }
  TestVerifyWindow();
  TestFaults(fresh40_w4a16);
  CheckNoCollectiveAllocs("end");
  if (g_failures != 0) {
    std::fprintf(stderr, "test_tp_emulation: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("[PASS] test_tp_emulation\n");
  return 0;
}

}  // namespace

int main() { return r4dx_test::RunGuardedMain("test_tp_emulation", [] { return RunTest(); }); }
