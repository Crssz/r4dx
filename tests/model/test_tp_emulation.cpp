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
//      ReadVerifyLogitsRow == the gathered row; CommitVerifiedWindow keeps the ranks in step; and
//      (P5, docs/tp.md Appendix B N80) a script of four 8-row windows committing 3, 8, 1 and 5 plus a
//      decode step on TP and on the TP=1 Model, every row within 1e-2 rel L2 of TP=1's;
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
// P5 (docs/tp.md 10.1 "P5 additions", 8.1, 8.2; G12's emulation half) -- the speculative rounds
// under TP, through the R4DX_TP_TESTING hooks (this test links r4dx_model_tptest):
//   7. MTP K=3 on l4-allmtp (w4a16) and on the real v6 container (which actually accepts drafts):
//      greedy rounds -- H6 exactness (every draft step's merged token == the argmax of that step's
//      gathered full row), accepted drafts == the drafted tokens, every emitted token == the argmax
//      of its gathered verify row (lossless by construction), committed positions == emitted tokens
//      after every round, both ranks' probes identical; seeded sampled rounds (three configs) and
//      T = 0.005 rounds (the logits_out path, H4) -- every emitted token == SampleCanonical over
//      its gathered verify row with the replayed draw, and the generator advanced by exactly one
//      draw per emitted token. The reduced-vocab head on l4-mtp-draftvocab (replicated: with the
//      capture on, no step merges, and every draft is a member of the head's vocabulary subset).
//   8. DFlash2 k=7 on v6 + ProductionDrafterPath() (the drafter's target layers need the 64-layer
//      target; docs/tp.md 10.1 names l4-allmtp, whose 4 layers cannot host them), greedy and seeded
//      sampled, 3 prompts: H7 exactness (the merged cand/unary of every round == the CPU top-16 of
//      the drafter's gathered [8, 248320] logits under r4dx_topk16_f32's total order), lockstep
//      bookkeeping (DflashInjectedCount() == PositionCount() on both ranks after every round), the
//      same per-row losslessness checks as MTP, and a Reset() rerun equal to the first run.
//   Every MTP / DFlash2 trajectory but the reruns is then replayed on the TP=1 Model -- each round's
//   VerifyWindow + CommitVerifiedWindow, the target's whole view of a round -- and the decode row
//   after the last round must sit within 1e-2 (4 layers) / 5e-2 (v6) rel L2 of TP's (docs/tp.md
//   Appendix B N80): the per-round checks alone compare each round with its own rows only.
//   9. vision on v6 (the 4-layer containers have no tower): two images of different grids through
//      one TpModel::EncodeImages (rank 0, host rows) and PrefillMultimodal + 3 decode steps against
//      the TP=1 Model -- encoder rows within 1e-2 rel L2, logits within 5e-2 and within 0.5x of how
//      far swapping one image moves TP=1's; the TP=1 host-row splice == its device-row splice byte
//      for byte; a device span refused by TpModel.
// The v6 cases SKIP (with a line saying so) when the production container (pair) is absent.
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

#ifndef R4DX_TP_TESTING
#error "test_tp_emulation needs the R4DX_TP_TESTING hooks: link r4dx_model_tptest (src/model/CMakeLists.txt)"
#endif

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
const char* kDraftVocabPath = r4dx_test::ContainerPath("D:/models/r4dx/qwen38-27b-l4-mtp-draftvocab.r4dx");
constexpr int64_t kLayers = 4;
constexpr int kDecodeRows = 16;
// The P5 comparisons with TP=1 (the verify / commit script, the speculative replays, vision; docs/tp.md
// Appendix B N80), per-row rel L2 of the logits: on the 4-layer container the P2b w4a16 tolerance of
// TestLayout; on the 64-layer v6 the TP=1 distance is larger with depth (measured up to 2.2e-2 over
// ~200 positions), so 5e-2 there -- a TP-only commit error (rank 1 seeding its GDN heads one window
// index early, tried as a mutation) put the v6 replays at 1.05e-1 .. 3.1e-1.
constexpr double kVsTp1Tol = 1e-2;
constexpr double kVsTp1TolV6 = 5e-2;

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
  {
    // core::DeviceBufferBytes (docs/tp.md Appendix B N64): both emulated ranks share one device, so
    // they report the same live-buffer figure, which the device-wide "used" must cover.
    const std::vector<r4dx::model::VramReport> v = tpm->Vram();
    CHECK(v.size() == 2 && v[0].buffers_gib == v[1].buffers_gib && v[0].buffers_gib > 0.5 &&
              v[0].buffers_gib <= v[0].used_gib,
          "[%s] Vram(): buffers %.3f / %.3f GiB, device-wide used %.3f GiB", layout.c_str(),
          v.empty() ? 0.0 : v[0].buffers_gib, v.size() < 2 ? 0.0 : v[1].buffers_gib, v.empty() ? 0.0 : v[0].used_gib);
    if (!v.empty()) {
      std::printf("[%s] Vram(): this process's buffers %.3f GiB, device-wide used %.3f GiB\n", layout.c_str(),
                  v[0].buffers_gib, v[0].used_gib);
    }
  }

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
  // The verify rows and the committed state against TP=1 (docs/tp.md Appendix B N80). Every check
  // above compares a merged result with rows gathered from the SAME TP run, so a TP-only error in a
  // verify row, or in what a commit leaves behind -- say a GDN window-bank seed read at the wrong
  // index for the rank-local heads -- would pass them all. So the same script on the TP=1 Model:
  // after Prefill(p40), four 8-row windows committing 3, 8, 1 and 5 (seeds from four window indices,
  // the whole window, the anchor alone), then one decode step; every window row and the decode row
  // must sit within the P2b tolerance of TP=1's.
  {
    const std::vector<int64_t> commits = {3, 8, 1, 5};
    const int32_t last = Tokens(1, 71)[0];
    const auto window = [](size_t w) { return Tokens(8, 101 + static_cast<uint32_t>(w)); };
    tpm->Reset();
    (void)tpm->Prefill(p40);
    auto tp_windows = std::make_shared<Rows>();  // rank 0's gathered rows (both ranks' are equal)
    tpm->RunCollectiveForTest([tp_windows, commits, window](Model& m, int rank) {
      for (size_t w = 0; w < commits.size(); ++w) {
        std::vector<float> logits;
        (void)m.VerifyWindow(window(w), &logits);
        m.CommitVerifiedWindow(commits[w]);
        if (rank == 0) tp_windows->push_back(std::move(logits));
      }
    });
    const std::vector<float> tp_last = tpm->DecodeStep(last);
    const int64_t tp_pos = tpm->PositionCount();
    Rows ref_windows;
    std::vector<float> ref_last;
    int64_t ref_pos = 0;
    {
      Model m1 = Model::Load(o);  // the same options at TP=1
      (void)m1.Prefill(p40);
      for (size_t w = 0; w < commits.size(); ++w) {
        std::vector<float> logits;
        (void)m1.VerifyWindow(window(w), &logits);
        m1.CommitVerifiedWindow(commits[w]);
        ref_windows.push_back(std::move(logits));
      }
      ref_last = m1.DecodeStep(last);
      ref_pos = m1.PositionCount();
    }
    CHECK(tp_pos == ref_pos && tp_pos == 40 + 3 + 8 + 1 + 5 + 1, "positions after the commit script: TP %lld, TP=1 %lld",
          static_cast<long long>(tp_pos), static_cast<long long>(ref_pos));
    double worst_window = 0.0;
    for (size_t w = 0; w < commits.size() && w < tp_windows->size(); ++w) {
      const std::vector<float>& a = ref_windows[w];
      const std::vector<float>& b = (*tp_windows)[w];
      CHECK(a.size() == b.size() && a.size() == static_cast<size_t>(8 * V), "window %zu: %zu vs %zu logits", w, b.size(),
            a.size());
      if (a.size() != b.size() || a.size() != static_cast<size_t>(8 * V)) continue;
      for (int64_t t = 0; t < 8; ++t) {
        const std::vector<float> ra(a.begin() + t * V, a.begin() + (t + 1) * V), rb(b.begin() + t * V, b.begin() + (t + 1) * V);
        const double e = RelL2(ra, rb);
        worst_window = std::max(worst_window, e);
        CHECK(e <= kVsTp1Tol, "window %zu (commit %lld) row %lld: rel L2 vs TP=1 %.3e > %.0e", w,
              static_cast<long long>(commits[w]), static_cast<long long>(t), e, kVsTp1Tol);
      }
    }
    const double e_last = tp_last.size() == ref_last.size() ? RelL2(ref_last, tp_last) : 1.0;
    CHECK(e_last <= kVsTp1Tol, "the decode row after the commit script: rel L2 vs TP=1 %.3e > %.0e", e_last, kVsTp1Tol);
    std::printf("[verify] commit script 3/8/1/5 vs TP=1: max rel L2 %.3e over the 32 window rows, %.3e on the decode "
                "row after it (tol %.0e)\n", worst_window, e_last, kVsTp1Tol);
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

// ---- P5: the speculative rounds under TP (docs/tp.md 8.1, 8.2, 10.1 "P5 additions") -------------

// What each rank reports right after one round, through RunCollectiveForTest: the round's verify
// rows gathered into full rows (ReadVerifyLogitsRow -- collective), its position bookkeeping and the
// drafter's debug capture (R4DX_TP_TESTING hooks).
struct RoundProbe {
  std::vector<float> verify_rows;    // [emitted rows][V]
  int64_t position = 0;
  int64_t injected = -1;             // DFlash only: DflashInjectedCount()
  std::vector<float> draft_rows;     // MTP: [steps][V] per-step draft rows; DFlash: [8][V] logits
  std::vector<int32_t> draft_tokens;  // MTP: the merged per-step tokens (empty when nothing merged)
  std::vector<int32_t> drafts;       // MTP: what the round's Draft() returned, on every path
  std::vector<int32_t> cand;         // DFlash: the merged top-16 the walk consumed, [8][16]
  std::vector<float> unary;
};

enum class Drafter { kMtp, kDflash };

std::vector<RoundProbe> ProbeRound(TpModel& tpm, size_t rows, Drafter drafter) {
  auto out = std::make_shared<std::vector<RoundProbe>>(2);
  tpm.RunCollectiveForTest([out, rows, drafter](Model& m, int rank) {
    RoundProbe& p = (*out)[static_cast<size_t>(rank)];
    const int64_t V = m.Config().vocab_size;
    std::vector<float> row;
    p.verify_rows.resize(rows * static_cast<size_t>(V));
    for (size_t i = 0; i < rows; ++i) {
      m.ReadVerifyLogitsRow(static_cast<int64_t>(i), row);
      std::memcpy(p.verify_rows.data() + i * static_cast<size_t>(V), row.data(), static_cast<size_t>(V) * 4);
    }
    p.position = m.PositionCount();
    if (drafter == Drafter::kMtp) {
      m.MtpDebugLastDraft(&p.draft_rows, &p.draft_tokens, &p.drafts);
    } else {
      p.injected = m.DflashInjectedCount();
      m.DflashDebugLastTop16(&p.cand, &p.unary);
      m.DflashDebugGatherDraftLogits(&p.draft_rows);
    }
  });
  return *out;
}

bool SameProbe(const RoundProbe& a, const RoundProbe& b) {
  const auto same_f = [](const std::vector<float>& x, const std::vector<float>& y) {
    return x.size() == y.size() && (x.empty() || std::memcmp(x.data(), y.data(), x.size() * 4) == 0);
  };
  return same_f(a.verify_rows, b.verify_rows) && a.position == b.position && a.injected == b.injected &&
         same_f(a.draft_rows, b.draft_rows) && a.draft_tokens == b.draft_tokens && a.drafts == b.drafts &&
         a.cand == b.cand && same_f(a.unary, b.unary);
}

// The CPU top-16 of one row under r4dx_topk16_f32's total order: value descending, ties to the
// LOWER id.
void CpuTop16(const float* row, int64_t V, int32_t* ids, float* vals) {
  std::vector<int32_t> idx(static_cast<size_t>(V));
  for (int64_t i = 0; i < V; ++i) idx[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::partial_sort(idx.begin(), idx.begin() + 16, idx.end(), [row](int32_t a, int32_t b) {
    return row[a] > row[b] || (row[a] == row[b] && a < b);
  });
  for (int j = 0; j < 16; ++j) {
    ids[j] = idx[static_cast<size_t>(j)];
    vals[j] = row[idx[static_cast<size_t>(j)]];
  }
}

// Running totals of one speculative case, for its summary line.
struct SpecTally {
  int rounds = 0, emitted = 0, max_round = 0, draft_steps = 0, top16_rows = 0;
};

// One speculative trajectory as the target model saw it, for the TP=1 replay (docs/tp.md Appendix B
// N80). Every per-round check compares a round with its OWN gathered rows, so a TP-only error in
// what a round commits -- a GDN window-bank seed, a KV position -- would pass them all: the later
// rounds would verify on the wrong state, identically on both ranks. A round's effect on the
// target is exactly VerifyWindow(window) + CommitVerifiedWindow(n) (Model::DecodeStepMtpImpl /
// DecodeStepDflashImpl), so ReplayAtTp1 runs those on the TP=1 Model and compares the decode row
// after the last round with TP's. The committed state depends only on each window's first n
// candidates (causal attention, a row-independent GEMM, the GDN bank's state after candidate n-1),
// so a DFlash2 window's rejected drafts -- which the round does not report -- are filled with the
// round's last token; its width (1 + walk_len) is kept, because the width picks the GEMM tiling.
struct Trajectory {
  std::string label;
  std::vector<int32_t> prompt;
  std::vector<std::vector<int32_t>> windows;  // each round's verify candidates
  std::vector<int64_t> commits;               // each round's committed count (== its emitted tokens)
  int32_t frontier = 0;                       // the last round's last token, not yet fed
  std::vector<float> tp_row;                  // TP's DecodeStep(frontier) after the rounds
};

// Replays `trajs` on a TP=1 Model loaded like `tp_opts` but with no drafter (the same verify window,
// dflash_draft_k sizing it): every trajectory's decode row after its rounds must sit within `tol`
// of TP's.
void ReplayAtTp1(const ModelOptions& tp_opts, const std::vector<Trajectory>& trajs, const std::string& name,
                 double tol) {
  if (trajs.empty()) return;
  ModelOptions o = tp_opts;
  o.dflash_draft_k = std::max(o.mtp_draft_k, o.dflash_draft_k);
  o.mtp_draft_k = 0;
  o.mtp_draft_reduced_vocab = false;
  o.dflash_container.clear();
  Model m = Model::Load(o);
  double worst = 0.0;
  size_t rounds = 0, tokens = 0;
  for (const Trajectory& t : trajs) {
    m.Reset();
    (void)m.Prefill(t.prompt);
    for (size_t r = 0; r < t.windows.size(); ++r) {
      (void)m.VerifyWindow(t.windows[r]);
      m.CommitVerifiedWindow(t.commits[r]);
      tokens += static_cast<size_t>(t.commits[r]);
    }
    rounds += t.windows.size();
    const std::vector<float> row = m.DecodeStep(t.frontier);
    const double e = row.size() == t.tp_row.size() ? RelL2(row, t.tp_row) : 1.0;
    worst = std::max(worst, e);
    CHECK(e <= tol, "[%s] after %zu rounds, TP's next decode row is %.3e (rel L2) from the TP=1 replay's (tol %.0e)",
          t.label.c_str(), t.windows.size(), e, tol);
  }
  std::printf("[%s] TP=1 replay of %zu trajectories (%zu rounds, %zu committed tokens): max rel L2 of the next decode "
              "row vs TP %.3e (tol %.0e)\n", name.c_str(), trajs.size(), rounds, tokens, worst, tol);
}

// The checks every round gets, greedy or sampled. `params` null = greedy (emitted token == argmax
// of its gathered verify row), else sampled: emitted token i == SampleCanonical(gathered verify row
// i, u_i) with u_i replayed from `rng_before`, and the caller's generator (`rng_after`) advanced by
// exactly round.size() draws. MTP: H6; DFlash: H7 and the injected-count bookkeeping.
void CheckRound(TpModel& tpm, const char* label, Drafter drafter, int64_t k, const std::vector<int32_t>& round,
                int64_t pos_before, const kernels::SampleParams* params, const std::mt19937_64* rng_before,
                const std::mt19937_64* rng_after, SpecTally* tally, std::vector<int32_t>* drafts_out) {
  const int64_t V = tpm.Config().vocab_size;
  CHECK(!round.empty() && static_cast<int64_t>(round.size()) <= k + 1, "[%s] round of %zu tokens (k %lld)", label,
        round.size(), static_cast<long long>(k));
  if (round.empty()) return;
  const std::vector<RoundProbe> pr = ProbeRound(tpm, round.size(), drafter);
  CHECK(SameProbe(pr[0], pr[1]), "[%s] the ranks' round probes differ", label);
  const RoundProbe& p = pr[0];
  *drafts_out = p.drafts;
  // Bookkeeping lockstep: the round committed exactly the tokens it emitted.
  CHECK(p.position == pos_before + static_cast<int64_t>(round.size()) && tpm.PositionCount() == p.position,
        "[%s] position %lld after a %zu-token round from %lld (facade %lld)", label, static_cast<long long>(p.position),
        round.size(), static_cast<long long>(pos_before), static_cast<long long>(tpm.PositionCount()));
  // Lossless by construction: every emitted token is the target's own choice on its verify row.
  std::mt19937_64 replay = rng_before != nullptr ? *rng_before : std::mt19937_64();
  for (size_t i = 0; i < round.size(); ++i) {
    const float* row = p.verify_rows.data() + i * static_cast<size_t>(V);
    const int32_t want = params == nullptr
                             ? kernels::Argmax(row, V)
                             : kernels::SampleCanonical(row, V, *params, kernels::DrawUniform01(replay));
    CHECK(round[i] == want, "[%s] emitted token %zu is %d, its gathered verify row says %d", label, i, round[i], want);
  }
  if (params != nullptr) CHECK(replay == *rng_after, "[%s] the round did not take exactly one draw per token", label);
  if (drafter == Drafter::kMtp) {
    // H6: every draft step's merged token is the full row's argmax (lowest index on ties).
    const size_t steps = p.draft_tokens.size();
    CHECK(static_cast<int64_t>(steps) == k && p.draft_rows.size() == steps * static_cast<size_t>(V),
          "[%s] MTP capture: %zu steps, %zu floats (k %lld)", label, steps, p.draft_rows.size(), static_cast<long long>(k));
    for (size_t s = 0; s < steps && p.draft_rows.size() == steps * static_cast<size_t>(V); ++s) {
      const int32_t want = kernels::Argmax(p.draft_rows.data() + s * static_cast<size_t>(V), V);
      CHECK(p.draft_tokens[s] == want, "[%s] H6: draft step %zu merged %d, the gathered row's argmax is %d", label, s,
            p.draft_tokens[s], want);
    }
    CHECK(p.drafts == p.draft_tokens, "[%s] the round verified drafts other than the merged tokens", label);
    for (size_t j = 0; j + 1 < round.size() && j < steps; ++j) {
      CHECK(round[j] == p.draft_tokens[j], "[%s] accepted token %zu (%d) is not draft %zu (%d)", label, j, round[j], j,
            p.draft_tokens[j]);
    }
    tally->draft_steps += static_cast<int>(steps);
  } else {
    // H7: the merged top-16 of every block row == the CPU top-16 of the gathered row, ids and value
    // bits; and the drafter's frontier is the model's position (injection kept in step).
    CHECK(p.injected == p.position, "[%s] DflashInjectedCount %lld != PositionCount %lld", label,
          static_cast<long long>(p.injected), static_cast<long long>(p.position));
    const size_t B = p.cand.size() / 16;
    CHECK(B > 0 && p.cand.size() == B * 16 && p.unary.size() == B * 16 &&
              p.draft_rows.size() == B * static_cast<size_t>(V),
          "[%s] DFlash capture sizes: cand %zu unary %zu logits %zu", label, p.cand.size(), p.unary.size(),
          p.draft_rows.size());
    int32_t ids[16];
    float vals[16];
    for (size_t t = 0; t < B && p.draft_rows.size() == B * static_cast<size_t>(V); ++t) {
      CpuTop16(p.draft_rows.data() + t * static_cast<size_t>(V), V, ids, vals);
      const bool exact = std::memcmp(ids, p.cand.data() + t * 16, sizeof ids) == 0 &&
                         std::memcmp(vals, p.unary.data() + t * 16, sizeof vals) == 0;
      CHECK(exact, "[%s] H7: block row %zu's merged top-16 differs from the gathered row's (first id %d vs %d)", label,
            t, p.cand[t * 16], ids[0]);
      ++tally->top16_rows;
    }
  }
  ++tally->rounds;
  tally->emitted += static_cast<int>(round.size());
  tally->max_round = std::max(tally->max_round, static_cast<int>(round.size()));
}

// One speculative trajectory: Reset, Prefill(prompt), then `rounds` rounds from the prefill's
// greedy / sampled first token, every round checked. With `record`, the rounds' windows and commits
// and the decode row after them are appended for ReplayAtTp1. Returns the emitted tokens.
std::vector<int32_t> RunSpec(TpModel& tpm, const char* label, Drafter drafter, int64_t k,
                             const std::vector<int32_t>& prompt, int rounds, const kernels::SampleParams* params,
                             uint64_t seed, SpecTally* tally, std::vector<Trajectory>* record = nullptr) {
  const int64_t V = tpm.Config().vocab_size;
  std::mt19937_64 rng = kernels::MakeRng(seed);
  tpm.Reset();
  if (drafter == Drafter::kMtp) {
    tpm.RunCollectiveForTest([](Model& m, int) { m.MtpDebugSetCapture(true); });
  }
  const std::vector<float> lg = tpm.Prefill(prompt);
  int32_t next = params == nullptr ? kernels::Argmax(lg.data(), V)
                                   : kernels::SampleCanonical(lg.data(), V, *params, kernels::DrawUniform01(rng));
  Trajectory traj;
  traj.label = label;
  traj.prompt = prompt;
  std::vector<int32_t> emitted;
  for (int r = 0; r < rounds; ++r) {
    const int64_t pos = tpm.PositionCount();
    const std::mt19937_64 rng_before = rng;
    int64_t walk = -1;
    std::vector<int32_t> round;
    if (drafter == Drafter::kMtp) {
      round = params == nullptr ? tpm.DecodeStepMtpGreedy(next, k) : tpm.DecodeStepMtpSampled(next, k, *params, rng);
    } else {
      round = params == nullptr ? tpm.DecodeStepDflashGreedy(next, k, 0.0f, 0, &walk)
                                : tpm.DecodeStepDflashSampled(next, k, 0.0f, 0, *params, rng, &walk);
      CHECK(walk >= static_cast<int64_t>(round.size()) - 1 && walk <= k, "[%s] walk_len %lld for a %zu-token round",
            label, static_cast<long long>(walk), round.size());
    }
    std::vector<int32_t> drafts;
    CheckRound(tpm, label, drafter, k, round, pos, params, params ? &rng_before : nullptr, params ? &rng : nullptr,
               tally, &drafts);
    if (round.empty()) break;
    // The target's view of the round (Trajectory): [anchor, drafts] -- MTP's k drafts as drafted; a
    // DFlash2 walk's accepted drafts, then its rejected ones' slots filled -- and the commit.
    std::vector<int32_t> window = {next};
    if (drafter == Drafter::kMtp) {
      window.insert(window.end(), drafts.begin(), drafts.end());
    } else {
      window.insert(window.end(), round.begin(), round.end() - 1);
      window.resize(static_cast<size_t>(1 + walk), round.back());
    }
    traj.windows.push_back(std::move(window));
    traj.commits.push_back(static_cast<int64_t>(round.size()));
    emitted.insert(emitted.end(), round.begin(), round.end());
    next = round.back();
  }
  if (drafter == Drafter::kMtp) {
    tpm.RunCollectiveForTest([](Model& m, int) { m.MtpDebugSetCapture(false); });
  }
  if (record != nullptr && !emitted.empty()) {
    traj.frontier = next;
    traj.tp_row = tpm.DecodeStep(next);
    record->push_back(std::move(traj));
  }
  return emitted;
}

// A prompt a real model continues predictably (a random run, repeated), so drafts get accepted and
// rounds grow past one token.
std::vector<int32_t> RepeatedPrompt(size_t unit, int copies, uint32_t seed) {
  const std::vector<int32_t> u = Tokens(unit, seed);
  std::vector<int32_t> p;
  for (int c = 0; c < copies; ++c) p.insert(p.end(), u.begin(), u.end());
  return p;
}

std::vector<kernels::SampleParams> SpecSampleConfigs() {
  std::vector<kernels::SampleParams> v(3);
  v[0].temperature = 0.7f;
  v[0].top_k = 20;
  v[0].top_p = 0.8f;
  v[1].temperature = 1.0f;
  v[2].temperature = 0.8f;
  v[2].min_p = 0.05f;
  return v;
}

// docs/tp.md 9.2's pre-flight for the cases that put BOTH v6 ranks on this one device: a
// shortfall is a failure naming the likely cause, not a skip (WDDM would page an over-committed
// load over PCIe instead of failing it).
bool EnoughVram(double need_gib, const char* what) {
  size_t free_b = 0, total_b = 0;
  const bool ok = hipMemGetInfo(&free_b, &total_b) == hipSuccess;
  const double free_gib = static_cast<double>(free_b) / (1024.0 * 1024.0 * 1024.0);
  CHECK(ok && free_gib >= need_gib, "%s: need %.1f GiB free on this HIP device, have %.2f GiB -- is the production "
        "server running?", what, need_gib, free_gib);
  return ok && free_gib >= need_gib;
}
// Measured (Appendix B N73): both emulated v6 w4a16 ranks use 20.29 GiB with the MTP head and 22.91
// GiB with the DFlash2 drafters at --max-ctx 2048.
constexpr double kNeedGiBV6Mtp = 22.0;
constexpr double kNeedGiBV6Dflash = 24.0;

void PrintTally(const char* label, const SpecTally& t) {
  std::printf("[%s] %d rounds checked, %d tokens emitted (longest round %d), %d H6 draft steps, %d H7 block rows\n",
              label, t.rounds, t.emitted, t.max_round, t.draft_steps, t.top16_rows);
}

// Greedy, three sampled configs and T = 0.005 through one drafter on one TpModel; every trajectory
// but the rerun is recorded for ReplayAtTp1.
void SpecSuite(TpModel& tpm, const std::string& name, Drafter drafter, int64_t k,
               const std::vector<std::vector<int32_t>>& prompts, int rounds, std::vector<Trajectory>* record) {
  SpecTally greedy, sampled, tiny;
  std::vector<int32_t> first;
  for (size_t i = 0; i < prompts.size(); ++i) {
    const std::string label = name + " greedy prompt " + std::to_string(i);
    std::vector<int32_t> e = RunSpec(tpm, label.c_str(), drafter, k, prompts[i], rounds, nullptr, 0, &greedy, record);
    if (i == 0) first = std::move(e);
  }
  // Reset + rerun == the first run (the rounds are deterministic under TP).
  SpecTally again;
  const std::vector<int32_t> rerun =
      RunSpec(tpm, (name + " greedy rerun").c_str(), drafter, k, prompts[0], rounds, nullptr, 0, &again);
  CHECK(rerun == first, "[%s] Reset() + rerun of the greedy rounds differs from the first run", name.c_str());
  PrintTally((name + " greedy").c_str(), greedy);
  const std::vector<kernels::SampleParams> configs = SpecSampleConfigs();
  for (size_t c = 0; c < configs.size(); ++c) {
    const std::string label = name + " sampled config " + std::to_string(c);
    (void)RunSpec(tpm, label.c_str(), drafter, k, prompts[c % prompts.size()], rounds, &configs[c], 11 + c, &sampled,
                  record);
  }
  PrintTally((name + " sampled").c_str(), sampled);
  kernels::SampleParams t005;
  t005.temperature = 0.005f;  // below kMinSummaryTemperature: the verify window's logits_out path (H4)
  (void)RunSpec(tpm, (name + " T=0.005").c_str(), drafter, k, prompts[0], rounds, &t005, 3, &tiny, record);
  PrintTally((name + " T=0.005").c_str(), tiny);
  CHECK(tpm.GetState() == TpModel::State::kReady, "[%s] group not ready after the speculative cases", name.c_str());
}

// 7. MTP under TP.
void TestMtp() {
  std::printf("==== MTP K=3 under TP ====\n");
  constexpr int64_t kK = 3;
  {
    ModelOptions o = BaseOptions("w4a16");
    o.mtp_draft_k = kK;
    std::vector<Trajectory> trajs;
    {
      std::unique_ptr<TpModel> tpm = TpModel::Load(o, EmulateOptions());
      CHECK(tpm->MtpEnabled() && !tpm->MtpUsingReducedVocabDraft(), "l4-allmtp: MTP enabled, full-vocab head");
      SpecSuite(*tpm, "mtp l4-allmtp", Drafter::kMtp, kK, {Tokens(40, 11), Tokens(70, 23)}, 6, &trajs);
    }
    ReplayAtTp1(o, trajs, "mtp l4-allmtp", kVsTp1Tol);
  }
  if (r4dx_test::FileExists(kDraftVocabPath)) {
    // The reduced-vocab draft head is replicated: its argmax and id mapping stay on the device, no
    // merge. With the capture ON, a merge would record its tokens, so an empty H6 capture beside k
    // drafts that are all members of the head's vocabulary subset shows the round drafted through
    // the replicated head's own id mapping; the rounds still get every per-row check.
    ModelOptions o = BaseOptions("w4a16");
    o.container_path = kDraftVocabPath;
    o.mtp_draft_k = kK;
    o.mtp_draft_reduced_vocab = true;
    std::unique_ptr<TpModel> tpm = TpModel::Load(o, EmulateOptions());
    CHECK(tpm->MtpUsingReducedVocabDraft(), "l4-mtp-draftvocab: the reduced-vocab head is in use");
    std::vector<int32_t> subset;
    tpm->RunCollectiveForTest([&subset](Model& m, int rank) {
      if (rank == 0) subset = m.GetContainer().Mtp().draft_vocab_ids.CopyToHost();
    });
    std::sort(subset.begin(), subset.end());
    CHECK(!subset.empty(), "l4-mtp-draftvocab: no draft_vocab_ids");
    SpecTally t;
    for (int r = 0; r < 2; ++r) {
      const std::string label = "mtp reduced-vocab prompt " + std::to_string(r);
      tpm->Reset();
      tpm->RunCollectiveForTest([](Model& m, int) { m.MtpDebugSetCapture(true); });
      const std::vector<float> lg = tpm->Prefill(Tokens(40, 11 + static_cast<uint32_t>(r)));
      int32_t next = kernels::Argmax(lg.data(), tpm->Config().vocab_size);
      for (int i = 0; i < 6; ++i) {
        const int64_t pos = tpm->PositionCount();
        const std::vector<int32_t> round = tpm->DecodeStepMtpGreedy(next, kK);
        const std::vector<RoundProbe> pr = ProbeRound(*tpm, round.size(), Drafter::kMtp);
        CHECK(SameProbe(pr[0], pr[1]), "[%s] the ranks' round probes differ", label.c_str());
        CHECK(pr[0].draft_tokens.empty(), "[%s] the replicated reduced head merged %zu step(s) across ranks",
              label.c_str(), pr[0].draft_tokens.size());
        CHECK(static_cast<int64_t>(pr[0].drafts.size()) == kK, "[%s] %zu drafts, want %lld", label.c_str(),
              pr[0].drafts.size(), static_cast<long long>(kK));
        for (int32_t d : pr[0].drafts) {
          CHECK(std::binary_search(subset.begin(), subset.end(), d),
                "[%s] draft %d is not in the reduced head's vocabulary subset", label.c_str(), d);
        }
        for (size_t j = 0; j + 1 < round.size() && j < pr[0].drafts.size(); ++j) {
          CHECK(round[j] == pr[0].drafts[j], "[%s] accepted token %zu (%d) is not draft %zu (%d)", label.c_str(), j,
                round[j], j, pr[0].drafts[j]);
        }
        CHECK(pr[0].position == pos + static_cast<int64_t>(round.size()), "[%s] position bookkeeping", label.c_str());
        for (size_t j = 0; j < round.size(); ++j) {
          const int64_t V = tpm->Config().vocab_size;
          CHECK(round[j] == kernels::Argmax(pr[0].verify_rows.data() + j * static_cast<size_t>(V), V),
                "[%s] emitted token %zu is not its verify row's argmax", label.c_str(), j);
        }
        ++t.rounds;
        t.emitted += static_cast<int>(round.size());
        t.max_round = std::max(t.max_round, static_cast<int>(round.size()));
        next = round.back();
      }
      tpm->RunCollectiveForTest([](Model& m, int) { m.MtpDebugSetCapture(false); });
    }
    PrintTally("mtp reduced-vocab greedy", t);
  } else {
    std::printf("SKIP (not a failure): %s missing -- no reduced-vocab MTP case\n", kDraftVocabPath);
  }
  // The real container, whose MTP head does accept (multi-token rounds, the reseed after an accept).
  const char* target = r4dx_test::ProductionTargetPath();
  if (r4dx_test::FileExists(target) && EnoughVram(kNeedGiBV6Mtp, "mtp v6")) {
    ModelOptions o;
    o.container_path = target;
    o.layout = r4dx::model::Layout::kW4a16;
    o.max_ctx = 1024;
    o.vision = ModelOptions::VisionMode::kOff;
    o.mtp_draft_k = kK;
    std::vector<Trajectory> trajs;
    {
      std::unique_ptr<TpModel> tpm = TpModel::Load(o, EmulateOptions());
      SpecSuite(*tpm, "mtp v6", Drafter::kMtp, kK, {RepeatedPrompt(24, 3, 5), RepeatedPrompt(32, 2, 9)}, 6, &trajs);
    }
    ReplayAtTp1(o, trajs, "mtp v6", kVsTp1TolV6);
  } else if (!r4dx_test::FileExists(target)) {
    std::printf("SKIP (not a failure): %s missing -- no MTP case on the real container\n", target);
  }
  CheckNoCollectiveAllocs("mtp");
}

// 8. DFlash2 under TP.
void TestDflash() {
  std::printf("==== DFlash2 k=7 under TP ====\n");
  const char* target = r4dx_test::ProductionTargetPath();
  const char* drafter = r4dx_test::ProductionDrafterPath();
  if (!r4dx_test::FileExists(target) || !r4dx_test::FileExists(drafter)) {
    std::printf("SKIP (not a failure): %s or %s missing -- no DFlash2 case\n", target, drafter);
    return;
  }
  if (!EnoughVram(kNeedGiBV6Dflash, "dflash v6")) return;
  constexpr int64_t kK = 7;
  ModelOptions o;
  o.container_path = target;
  o.layout = r4dx::model::Layout::kW4a16;
  o.max_ctx = 1024;
  o.vision = ModelOptions::VisionMode::kOff;
  o.dflash_container = drafter;
  o.dflash_draft_k = kK;
  std::vector<Trajectory> trajs;
  {
    std::unique_ptr<TpModel> tpm = TpModel::Load(o, EmulateOptions());
    CHECK(tpm->DflashEnabled(), "DFlash2 enabled");
    SpecSuite(*tpm, "dflash v6", Drafter::kDflash, kK,
              {RepeatedPrompt(24, 3, 5), RepeatedPrompt(32, 2, 9), Tokens(48, 77)}, 6, &trajs);
  }
  ReplayAtTp1(o, trajs, "dflash v6", kVsTp1TolV6);
  CheckNoCollectiveAllocs("dflash");
}

// ---- vision under TP (docs/tp.md 8.3; Appendix B N68, N69, N80) --------------------------------

// A deterministic synthetic RGB image: two ramps and a noise channel (the test asserts engine
// agreement, not what the model makes of it).
r4dx::vision::DecodedImage SyntheticImage(int w, int h, uint32_t seed) {
  r4dx::vision::DecodedImage im;
  im.width = w;
  im.height = h;
  im.rgb.resize(static_cast<size_t>(w) * static_cast<size_t>(h) * 3);
  uint32_t x = seed;
  for (int y = 0; y < h; ++y) {
    for (int c = 0; c < w; ++c) {
      const size_t at = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(c)) * 3;
      x = x * 1664525u + 1013904223u;
      im.rgb[at] = static_cast<uint8_t>(c * 255 / w);
      im.rgb[at + 1] = static_cast<uint8_t>(y * 255 / h);
      im.rgb[at + 2] = static_cast<uint8_t>(x >> 24);
    }
  }
  return im;
}

// Measured (Appendix B N73): both emulated v6 w4a16 ranks with the tower use 20.89 GiB at --max-ctx
// 2048; the TP=1 load before them needs less.
constexpr double kNeedGiBV6Vision = 22.0;

// 9. Vision under TP: rank 0's solo EncodeImages (host rows), TpModel::PrefillMultimodal's copy and
//    span rebasing, the host-to-device splice on both ranks, rank 1's merge size and 3-axis rope, and
//    the rope delta the decode steps after the image use -- against the TP=1 Model on the real
//    container (the 4-layer test containers carry no vision.* tensors). Two images with different
//    grids go through ONE EncodeImages call (a multi-image turn), so both spans must land on the
//    right rows. The TP-vs-TP=1 distance of the logits must stay within the v6 tolerance AND at most
//    half of how far the TP=1 logits move when image 1 is swapped for another image of the same
//    grid -- a TP error that dropped or misplaced an image's rows would be about that large.
void TestVision() {
  std::printf("==== vision under TP (v6 w4a16, two images) ====\n");
  namespace vision = r4dx::vision;
  using r4dx::model::ImageRows;
  using r4dx::model::ImageSpan;
  const char* target = r4dx_test::ProductionTargetPath();
  if (!r4dx_test::FileExists(target)) {
    std::printf("SKIP (not a failure): %s missing -- no vision case\n", target);
    return;
  }
  if (!EnoughVram(kNeedGiBV6Vision, "vision v6")) return;
  const vision::ImageProcessorConfig pcfg;
  const vision::PreprocessedImages pre =
      vision::PreprocessImages({SyntheticImage(256, 256, 3), SyntheticImage(320, 256, 7)}, pcfg);
  // The yardstick's pair: image 1 replaced by another image of the same size (so the same grid).
  const vision::PreprocessedImages alt =
      vision::PreprocessImages({SyntheticImage(256, 256, 3), SyntheticImage(320, 256, 99)}, pcfg);
  ModelOptions o;
  o.container_path = target;
  o.layout = r4dx::model::Layout::kW4a16;
  o.max_ctx = 1024;
  o.vision = ModelOptions::VisionMode::kOn;
  const std::vector<int32_t> forced = Tokens(3, 47);

  // text, image 0's placeholder run, text, image 1's run, text; spans in prompt order.
  const auto build = [&](int64_t image_token, int64_t merge, std::vector<ImageSpan>* spans) {
    std::vector<int32_t> ids = Tokens(10, 41);
    for (size_t i = 0; i < pre.grid_thw.size(); ++i) {
      ImageSpan sp;
      sp.offset = static_cast<int64_t>(ids.size());
      sp.grid = pre.grid_thw[i];
      sp.tokens = sp.grid.MergedTokenCount(merge);
      ids.insert(ids.end(), static_cast<size_t>(sp.tokens), static_cast<int32_t>(image_token));
      spans->push_back(sp);
      const std::vector<int32_t> gap = Tokens(6, 43 + static_cast<uint32_t>(i));
      ids.insert(ids.end(), gap.begin(), gap.end());
    }
    return ids;
  };
  // Points the spans at consecutive [tokens, hidden] slices of one EncodeImages output.
  const auto point = [](std::vector<ImageSpan>& spans, const uint16_t* rows, bool on_host, int64_t hidden) {
    int64_t row = 0;
    for (ImageSpan& sp : spans) {
      sp.embeds = rows + row * hidden;
      sp.embeds_on_host = on_host;
      row += sp.tokens;
    }
  };
  // Reset; PrefillMultimodal -> row 0; DecodeStep(forced[i]) -> row i+1 (the rope delta path).
  const auto script = [&forced](TextModel& m, const std::vector<int32_t>& ids, const std::vector<ImageSpan>& spans) {
    Rows rows;
    m.Reset();
    rows.push_back(m.PrefillMultimodal(ids, spans));
    for (int32_t t : forced) rows.push_back(m.DecodeStep(t));
    return rows;
  };

  std::vector<int32_t> ids;
  std::vector<ImageSpan> spans;
  std::vector<uint16_t> rows1;  // TP=1's encoder rows, copied to the host
  int64_t image_token = 0, merge = 0, hidden = 0;
  Rows ref, ref_alt;
  {
    LocalTextModel m(Model::Load(o));
    CHECK(m.HasVision(), "TP=1: the tower is loaded");
    image_token = m.ImageTokenId();
    merge = m.VisionMergeSize();
    hidden = m.Config().hidden_size;
    ids = build(image_token, merge, &spans);
    ImageRows er;
    m.EncodeImages(pre.pixel_values.data(), pre.TotalPatches(), pre.grid_thw, &er);
    CHECK(!er.on_host() && er.rows() == spans[0].tokens + spans[1].tokens, "TP=1 EncodeImages: %lld device rows",
          static_cast<long long>(er.rows()));
    rows1 = er.dev.CopyToHost();
    point(spans, er.data(), /*on_host=*/false, hidden);
    ref = script(m, ids, spans);
    // The same rows spliced from the HOST (the H2D copy every TP rank uses): byte for byte the D2D run.
    point(spans, rows1.data(), /*on_host=*/true, hidden);
    CHECK(SameBytes(ref, script(m, ids, spans)), "TP=1: host-row splice differs from the device-row splice");
    ImageRows er_alt;
    m.EncodeImages(alt.pixel_values.data(), alt.TotalPatches(), alt.grid_thw, &er_alt);
    point(spans, er_alt.data(), /*on_host=*/false, hidden);
    ref_alt = script(m, ids, spans);
  }

  std::unique_ptr<TpModel> tpm = TpModel::Load(o, EmulateOptions());
  CHECK(tpm->HasVision() && tpm->ImageTokenId() == image_token && tpm->VisionMergeSize() == merge,
        "TP: vision capability / image token / merge size differ from TP=1");
  ImageRows er;
  vision::VisionEncodeStats stats;
  tpm->EncodeImages(pre.pixel_values.data(), pre.TotalPatches(), pre.grid_thw, &er, &stats);
  CHECK(er.on_host() && er.rows() == spans[0].tokens + spans[1].tokens && er.host.size() == rows1.size(),
        "TP EncodeImages: on_host %d, %lld rows, %zu values (TP=1 %zu)", static_cast<int>(er.on_host()),
        static_cast<long long>(er.rows()), er.host.size(), rows1.size());
  const bool rows_exact =
      er.host.size() == rows1.size() && std::memcmp(er.host.data(), rows1.data(), rows1.size() * 2) == 0;
  const double rows_rel =
      er.host.size() == rows1.size() ? RelL2(r4dx_test::WidenBf16(rows1), r4dx_test::WidenBf16(er.host)) : 1.0;
  CHECK(rows_rel <= kVsTp1Tol, "TP encoder rows are %.3e (rel L2) from TP=1's", rows_rel);
  point(spans, er.data(), /*on_host=*/true, hidden);
  const Rows tp = script(*tpm, ids, spans);
  double worst = 0.0, yard = 0.0;
  for (size_t i = 0; i < ref.size() && i < tp.size(); ++i) {
    const double e = tp[i].size() == ref[i].size() ? RelL2(ref[i], tp[i]) : 1.0;
    worst = std::max(worst, e);
    yard = std::max(yard, RelL2(ref[i], ref_alt[i]));
    CHECK(e <= kVsTp1TolV6, "vision row %zu: rel L2 vs TP=1 %.3e > %.0e", i, e, kVsTp1TolV6);
  }
  CHECK(worst <= 0.5 * yard, "vision: TP sits %.3e from TP=1, a swapped image moves TP=1 only %.3e (limit x0.5)", worst,
        yard);
  CHECK(SameBytes(tp, script(*tpm, ids, spans)), "vision: Reset() + rerun differs from the first TP run");
  // A device span names memory on one device: refused before any command, the group stays ready.
  {
    std::vector<ImageSpan> dev_spans = spans;
    dev_spans[1].embeds_on_host = false;
    bool refused = false;
    try {
      (void)tpm->PrefillMultimodal(ids, dev_spans);
    } catch (const std::invalid_argument&) {
      refused = true;
    }
    CHECK(refused && tpm->GetState() == TpModel::State::kReady, "a device-resident span was not refused by name");
  }
  std::printf("[vision] 2 images (%lld + %lld tokens) encoded on rank 0 in %.1f ms: rows %s TP=1's (rel L2 %.3e); "
              "a %zu-token prompt + %zu decode steps: max rel L2 vs TP=1 %.3e (tol %.0e), a swapped image moves TP=1 "
              "%.3e (ratio %.2f, limit 0.5)\n",
              static_cast<long long>(spans[0].tokens), static_cast<long long>(spans[1].tokens), stats.encode_ms,
              rows_exact ? "byte-identical to" : "differ from", rows_rel, ids.size(), forced.size(), worst, kVsTp1TolV6,
              yard, yard > 0 ? worst / yard : 0.0);
  CHECK(tpm->GetState() == TpModel::State::kReady, "group not ready after the vision checks");
  CheckNoCollectiveAllocs("vision");
}

// tp::UnitLayersForContext (docs/tp.md Appendix B N64): the prefill unit shrinks with context.
// Host arithmetic only.
void TestUnitLayers() {
  namespace tp = r4dx::model::tp;
  CHECK(tp::UnitLayersForContext(0, 262144) == 0, "submit_layers 0 stays off");
  CHECK(tp::UnitLayersForContext(32, 64) == 32 && tp::UnitLayersForContext(32, 16384) == 32, "<= 16k: submit_layers");
  CHECK(tp::UnitLayersForContext(32, 16385) == 16 && tp::UnitLayersForContext(32, 65536) == 16, "<= 64k: 16");
  CHECK(tp::UnitLayersForContext(32, 65537) == 8 && tp::UnitLayersForContext(32, 131072) == 8, "<= 128k: 8");
  CHECK(tp::UnitLayersForContext(32, 131073) == 4 && tp::UnitLayersForContext(32, 262144) == 4, "> 128k: 4");
  CHECK(tp::UnitLayersForContext(1, 262144) == 1 && tp::UnitLayersForContext(6, 20000) == 6,
        "a finer setting is never coarsened");
}

int RunTest() {
  TestUnitLayers();
  if (!r4dx_test::FileExists(kContainerPath)) return r4dx_test::SkipMissing(kContainerPath);
  std::setvbuf(stdout, nullptr, _IONBF, 0);  // keep stdout in order with the facade's stderr lines
  Rows fresh40_w4a16;
  for (const char* layout : {"bf16", "w4a16", "w4a8", "mxfp4"}) {
    Rows r = TestLayout(layout);
    if (std::string(layout) == "w4a16") fresh40_w4a16 = std::move(r);
  }
  TestVerifyWindow();
  TestFaults(fresh40_w4a16);
  TestMtp();
  TestDflash();
  TestVision();
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
