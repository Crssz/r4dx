// Qwen35MoeModel on the tiny container against the HF goldens (docs/qwen35moe-plan.md M3). Links r4dx_model.
//
// For each golden prompt (T 1, 7, 70, 300), on both GDN prefill paths (libr4d's chunked scan = production, the recurrent
// decode kernels = the slow reference): Prefill with a capture installed and compare EVERY capture point the golden also has
// (residual in / attn_in / attn_out (GDN layers) / residual after the mixer / MoE input / router top-8 ids / MoE output /
// residual out of each layer, the final norm, the logits of every row) with the model's. The last row's argmax must equal the
// golden's, and a 24-token greedy continuation (DecodeStepGreedy) is compared with the HF free run (the report says how many
// leading tokens agree; bf16 noise may flip a near-tie later on, so only a minimum prefix is checked).
// Tolerances are noise bounds (see GoldenTol): the model computes in bf16 with a different accumulation order than torch's CPU
// matmuls, so elementwise equality is not expected; bounds are on max|err| / rms(ref) and mean|err| / rms(ref) per tensor.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "q35moe_model_util.h"

using namespace q35m;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

struct Tol {
  double mean_norm;   // mean|err| / rms(ref)
  double atol_rel, rtol, max_bad;  // at most `max_bad` of the elements may exceed atol_rel * rms(ref) + rtol * |ref|
};

// The bound of one capture point against the HF golden, as max|err| / rms(ref) and mean|err| / rms(ref) over all rows of the
// prompt. First-run values go here once observed; until then generous bounds that still exclude garbage (a wrong kernel
// reads as errors of several rms).
Tol GoldenTol(const std::string& name) {
  const auto has = [&](const char* s) { return name.find(s) != std::string::npos; };
  // Whole model against the fp32 golden: the noise of four layers of a noise-amplifying random network (the tiny net's HF bf16 run
  // is itself at mean/rms 0.045-0.09 from its fp32 run at layer 3), so these bounds only exclude garbage; the per-layer check is
  // the layer-isolated one below.
  if (has("moe_out")) return {0.25, 0.25, 0.15, 0.05};
  return {0.15, 0.25, 0.10, 0.02};
}

// One layer alone over the golden's own residual input: bf16 noise of that layer, not the amplified noise of those before it.
Tol IsolatedTol(const std::string& name) {
  const auto has = [&](const char* s) { return name.find(s) != std::string::npos; };
  if (has("moe_out")) return {0.06, 0.08, 0.05, 0.01};
  return {0.03, 0.05, 0.03, 0.004};
}

// Compares every capture the golden also has.
void CompareToGolden(const CaptureStore& store, const Golden& g, const std::string& tag, Tol (*tol)(const std::string&) = GoldenTol,
                     const std::string& prefix = "", bool strict_experts = false) {
  PrintStatsHeader();
  for (const auto& [name, cap] : store.all()) {
    if (!g.Has(name) || name.rfind(prefix, 0) != 0) continue;
    if (cap.dtype == Qwen35MoeCapture::Dtype::kI32) {
      // top-8 of 16 router logits that are bf16 numbers: a row whose 8th / 9th logits are within bf16 noise of each other may
      // legitimately pick the other one. The golden's own router logits tell which rows those are.
      int64_t n = 0, nl = 0;
      const int64_t* ref = g.I64(name, &n);
      const int rows = static_cast<int>(cap.rows), k = static_cast<int>(cap.cols);
      const std::string rl_name = name.substr(0, name.find('.')) + ".router_logits";
      const uint16_t* rl = g.Has(rl_name) ? g.Bf16(rl_name, &nl) : nullptr;
      const int E = rl != nullptr ? static_cast<int>(nl / rows) : 0;
      int order_rows = 0, set_rows = 0, unexplained = 0;
      double worst = 0;
      for (int r = 0; r < rows; ++r) {
        const int32_t* mine = cap.i32.data() + static_cast<size_t>(r) * k;
        const int64_t* hf = ref + static_cast<int64_t>(r) * k;
        bool same_order = true;
        std::vector<int> only_mine, only_hf;
        for (int j = 0; j < k; ++j) {
          same_order &= mine[j] == hf[j];
          bool in_hf = false, in_mine = false;
          for (int q = 0; q < k; ++q) {
            in_hf |= mine[j] == hf[q];
            in_mine |= hf[j] == mine[q];
          }
          if (!in_hf) only_mine.push_back(mine[j]);
          if (!in_mine) only_hf.push_back(static_cast<int>(hf[j]));
        }
        order_rows += same_order ? 0 : 1;
        if (only_mine.empty() && only_hf.empty()) continue;
        ++set_rows;
        bool explained = rl != nullptr;
        if (rl != nullptr) {
          std::vector<float> l(static_cast<size_t>(E));
          for (int e = 0; e < E; ++e) l[static_cast<size_t>(e)] = Bf(rl[static_cast<int64_t>(r) * E + e]);
          std::vector<float> sorted = l;
          std::sort(sorted.begin(), sorted.end(), std::greater<float>());
          const float cut = sorted[static_cast<size_t>(k - 1)];
          double sq = 0;
          for (float v : l) sq += static_cast<double>(v) * v;
          const double row_rms = std::sqrt(sq / static_cast<double>(E));
          for (const std::vector<int>* side : {&only_mine, &only_hf}) {
            for (int e : *side) {
              const double d = std::fabs(l[static_cast<size_t>(e)] - cut) / (row_rms > 0 ? row_rms : 1.0);
              worst = std::max(worst, d);
              explained &= d <= 0.15;
            }
          }
        }
        unexplained += explained ? 0 : 1;
      }
      std::printf("  %-26s rows %d: order differs in %d rows, expert SET differs in %d rows (%d not explained by a router near-tie; worst %.4f of the row's logit rms from the cut)\n",
                  name.c_str(), rows, order_rows, set_rows, unexplained, worst);
      // Whole-model runs only REPORT the expert-set differences (upstream noise moves a router logit across the cut anywhere in a
      // noise-amplifying network); the layer-isolated runs, which start from the golden's own residual, enforce them.
      if (strict_experts) Check(unexplained == 0, tag + " " + name + ": expert sets differ from the golden beyond router near-ties");
      continue;
    }
    int64_t n = 0;
    const uint16_t* ref = g.Bf16(name, &n);
    const int64_t have = static_cast<int64_t>(cap.bf16.size());
    if (have != n) {
      std::fprintf(stderr, "FAIL: %s %s has %lld elements, golden %lld\n", tag.c_str(), name.c_str(),
                   static_cast<long long>(have), static_cast<long long>(n));
      ++g_failures;
      continue;
    }
    const Stats s = CompareBf16(cap.bf16.data(), ref, n);
    PrintStats(name, s);
    const Tol t = tol(name);
    Check(!s.nan, tag + " " + name + ": NaN / Inf");
    const double bad = BadFraction(cap.bf16.data(), ref, n, t.atol_rel, t.rtol);
    std::printf("      %-26s mean/rms %.5f (<= %.3f), elements outside %.2f rms + %.0f%%: %.4f%% (<= %.2f%%)\n", name.c_str(), s.mean_norm,
                t.mean_norm, t.atol_rel, 100.0 * t.rtol, 100.0 * bad, 100.0 * t.max_bad);
    Check(s.mean_norm <= t.mean_norm && bad <= t.max_bad,
          tag + " " + name + ": error above its bound (mean/rms " + std::to_string(s.mean_norm) + " > " + std::to_string(t.mean_norm) +
              " or " + std::to_string(100.0 * bad) + "% of the elements outside " + std::to_string(t.atol_rel) + " rms + " +
              std::to_string(100.0 * t.rtol) + "% (allowed " + std::to_string(100.0 * t.max_bad) + "%))");
  }
}

}  // namespace

int RunTest();
int main() {
  try {
    return RunTest();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
  }
}

int RunTest() {
  Unbuffer();
  if (!Exists(ContainerPath("bf16"))) {
    std::printf("SKIP: %s missing\n", ContainerPath("bf16").c_str());
    return kSkip;
  }
  const int prompts[] = {1, 7, 70, 300};
  for (int T : prompts) {
    if (!Exists(GoldenPath(T)) || !Exists(FreePath(T))) {
      std::printf("SKIP: %s (or its free run) missing\n", GoldenPath(T).c_str());
      return kSkip;
    }
  }
  if (VisibleDevices() < 1) {
    std::printf("SKIP: no HIP device\n");
    return kSkip;
  }

  struct Config {
    bool recurrent;
    const char* what;
  };
  const Config configs[] = {
      {true, "GDN prefill through the recurrent kernels (the slow reference path)"},
      {false, "GDN prefill through libr4d's span-safe chunked scan (the production path)"},
  };
  for (const Config& cfgc : configs) {
    std::printf("\n==== bf16, %s (device %d) ====\n", cfgc.what, TestDevice());
    auto model = LoadTiny("bf16", false, {}, 0, 4096, cfgc.recurrent);
    for (int T : prompts) {
      const Golden g(T, GoldenPath(T));
      CaptureStore store;
      model->Reset();
      model->SetCapture(store.Fn());
      const std::vector<float> logits = model->Prefill(g.ids());
      model->SetCapture(nullptr);
      const auto& tm = model->LastTimings();
      std::printf("\n-- prompt T=%d: prefill %.1f ms (stage0 %.1f, embed %.2f, head %.1f; %lld chunk(s))\n", T, tm.total_ms,
                  tm.stage_ms[0], tm.embed_ms, tm.head_ms, static_cast<long long>(tm.chunks));
      CompareToGolden(store, g, std::string(cfgc.recurrent ? "rec" : "chunk") + " T" + std::to_string(T));

      // last-row logits
      int64_t n = 0;
      const uint16_t* gl = g.Bf16("logits", &n);
      const int64_t vocab = n / T;
      const std::vector<float> ref_last = WidenBf16(gl + (T - 1) * vocab, vocab);
      const FStats fs = CompareF32(logits.data(), ref_last.data(), vocab);
      const int64_t mine = ArgMax(logits.data(), vocab), theirs = ArgMax(ref_last.data(), vocab);
      std::printf(" last-row logits: max|err| %.4g mean|err| %.4g rms(ref) %.4g (max/rms %.4f); argmax ours %lld golden %lld\n",
                  fs.max_abs, fs.mean_abs, fs.rms_ref, fs.max_norm, static_cast<long long>(mine), static_cast<long long>(theirs));
      std::vector<float> sorted = ref_last;
      std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<float>());
      std::printf("   golden top-1 - top-2 logit margin %.4g\n", sorted[0] - sorted[1]);
      Check(mine == theirs, "T" + std::to_string(T) + ": last-row argmax differs from the golden");

      // free-running greedy continuation against HF's
      const std::vector<int32_t> want = FreeRun(T);
      std::vector<int32_t> got;
      int32_t tok = static_cast<int32_t>(mine);
      got.push_back(tok);
      for (size_t i = 1; i < want.size(); ++i) {
        tok = model->DecodeStepGreedy(tok);
        got.push_back(tok);
      }
      size_t agree = 0;
      while (agree < want.size() && got[agree] == want[agree]) ++agree;
      std::printf(" free run: %zu of %zu leading tokens agree with HF%s\n", agree, want.size(),
                  agree == want.size() ? "" : " (first difference at a bf16 near-tie is expected)");
      Check(agree >= 2, "T" + std::to_string(T) + ": the free run diverges from HF within the first 2 tokens");
    }
  }
  // ---- layer-isolated: each layer alone over the golden's own residual input ---------------------------------------------
  // The whole-model comparison above carries the noise the earlier layers add (a random-weight network amplifies it: HF's own
  // bf16 run differs from its fp32 run by ~7% (mean/rms) at layer 3's attention output, tools/reference/qwen35moe/noise_floor.py).
  // Here every layer starts from the golden's `layer{L}.residual_in` (so its input error is the bf16 rounding of the golden
  // alone) and is judged on its own outputs: a defect in a mixer, the router or an expert shows up at O(1), bf16 noise stays
  // within a few percent.
  {
    std::printf("\n==== layer-isolated (golden residual in, GDN prefill through libr4d's chunked scan) ====\n");
    auto model = LoadTiny("bf16", false, {}, 0, 4096, /*gdn_recurrent=*/false);
    for (int T : {7, 70, 300}) {
      const Golden g(T, GoldenPath(T));
      const Golden giso(T, IsoGoldenPath(T));  // the reference the isolated outputs are judged against
      for (int layer = 0; layer < 4; ++layer) {
        const std::string pfx = "layer" + std::to_string(layer) + ".";
        CaptureStore store;
        model->SetCapture(store.Fn());
        model->RunLayerIsolated(g.ids(), layer, g.Bf16(pfx + "residual_in"));
        model->SetCapture(nullptr);
        std::printf("\n-- isolated layer %d, T=%d\n", layer, T);
        CompareToGolden(store, giso, "iso L" + std::to_string(layer) + " T" + std::to_string(T), IsolatedTol, pfx, /*strict_experts=*/true);
      }
    }
  }

  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}




