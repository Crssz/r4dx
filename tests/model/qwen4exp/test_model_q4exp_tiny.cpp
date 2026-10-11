// Qwen4ExpModel on the tiny container against the HF goldens (docs/qwen4exp-plan.md M1-16). Links r4dx_model.
//
//   Part A  bf16 container, one device: for each golden prompt (T 1, 7, 70, 300) Prefill with a capture installed and
//           compare EVERY capture point of the golden (residual in / after PLE / after attention / after MoE of each layer,
//           the hyper-connection mixes and injection weights, the GDN / QSA block outputs, the QSA o_proj input, the
//           router's top-10 ids, the MoE outputs, the final mixer, the logits of every row) with the model's. The last
//           row's argmax must equal the golden's.
//   Part B  w4a16 container: its own quantization error against the golden is reported, and the checked bound is against
//           the bf16-container run of the same code (the layout differs only in the weights' precision).
// Tolerances are set from the noise this run observes (see the comments at kTol): the model computes in bf16 with a
// different accumulation order than torch's CPU matmuls, so elementwise equality is not expected; bounds are on
// max|err| / rms(ref) and mean|err| / rms(ref) of each tensor.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "q4exp_model_util.h"

using namespace q4m;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

struct Tol {
  double max_norm, mean_norm;
};

// The bound of one capture point against the HF golden (bf16 container), as max|err| / rms(ref) and mean|err| / rms(ref) of the
// whole tensor (all rows of the prompt). Set from the noise of the first run: T = 1 and T = 7 (both GDN prefill paths agree there)
// showed, at the worst point of any of the four layers, max/rms 0.05 (inject), 0.25 (mixes, residuals), 0.18 (block
// outputs), 0.38 (QSA o_proj input), 0.29 (moe_out), 0.31 (mixer), 0.18 (logits) and mean/rms <= 0.057; every bound below is about
// twice that. They are noise bounds, not ulp bounds: the model is bf16 end to end with different accumulation orders than torch's
// CPU matmuls (a bf16 ulp is 0.8% of a value, a random-weight network's residual accumulates a few of them per layer, and the
// MoE / mix outputs are ~0.01-0.5 against a layer residual an order larger).
Tol GoldenTol(const std::string& name) {
  const auto has = [&](const char* s) { return name.find(s) != std::string::npos; };
  if (has("inject")) return {0.15, 0.04};
  if (has("moe_out")) return {1.0, 0.11};  // layer 3 at T = 70 showed 0.70: an expert swapped at a router near-tie moves a row by ~its weight
  if (has("attn_pre_o")) return {0.8, 0.07};
  if (has("attn_out")) return {0.4, 0.07};
  if (has("logits") || has("mixer_out")) return {0.6, 0.07};
  return {0.5, 0.07};  // residuals, mixes
}
// ... and of the w4a16 container against the bf16 container (same code and GDN path, 4-bit weights): the quantization error of
// RTN int4 / group 64 on random normal weights is ~10% per weight, so a layer's linears move their outputs by 10-25% and four layers
// compound it (observed at T = 1: layer 0 attn_out mean/rms 0.23, logits 0.39). The bounds only exclude garbage (a wrong layout or
// scale reads as errors of several rms).
Tol QuantTol(const std::string& name) {
  if (name.find("layer0.") == 0) return {3.0, 0.45};
  return {12.0, 1.0};
}
// w4a16 against the golden: reported, not bounded (its own quantization error; the bound is against the bf16 container).
Tol ReportOnlyTol(const std::string&) {
  return {1e9, 1e9};
}

// The tiny indexer keeps the top 16 blocks (indexer_budget 64 / ratio 4) plus the incomplete tail: a query that sees at most 67
// tokens attends densely, so the QSA layer's output is exact there whatever the block scores are. From the 68th token on, the
// selection cuts blocks whose scores (sums of relu dot products of a random-weight network) are near-ties, and bf16 noise in the
// indexer's inputs flips some of them: the rows after that point of the QSA layer and of everything downstream (the layer's MoE,
// the head) are reported but not bounded; the dense rows [0, 67) are checked like every other tensor.
constexpr int kDenseRows = 67;
bool QsaDependent(const std::string& name) {
  if (name == "mixer_out" || name == "logits") return true;
  if (name.rfind("layer", 0) != 0) return false;
  const int layer = std::atoi(name.c_str() + 5);
  if (layer % 4 != 3) return false;
  for (const char* s : {"attn_mix", "attn_inject", "residual_in", "residual_ple"}) {
    if (name.size() > std::strlen(s) && name.compare(name.size() - std::strlen(s), std::strlen(s), s) == 0) return false;
  }
  return true;
}

// Compares every capture the golden also has. Returns the per-name worst stats (max over prompts is the caller's job).
void CompareToGolden(const CaptureStore& store, const Golden& g, const std::string& tag, Tol (*tol)(const std::string&),
                     bool strict) {
  PrintStatsHeader();
  for (const auto& [name, cap] : store.all()) {
    if (!g.Has(name)) continue;
    if (cap.dtype == Qwen4ExpCapture::Dtype::kI32) {
      // top-10 of 16 router logits that are bf16 numbers: a row whose 10th / 11th logits are within bf16 noise of each
      // other may legitimately pick the other one. The golden's own router logits tell which rows those are.
      int64_t n = 0, nl = 0;
      const int64_t* ref = g.I64(name, &n);
      const int rows = static_cast<int>(cap.rows), k = static_cast<int>(cap.cols);
      const std::string rl_name = name.substr(0, name.find('.')) + ".router_logits";
      const uint16_t* rl = g.Has(rl_name) ? g.Bf16(rl_name, &nl) : nullptr;
      const int E = rl != nullptr ? static_cast<int>(nl / rows) : 0;
      const int rows_checked = (QsaDependent(name) && rows > kDenseRows) ? kDenseRows : rows;
      int order_rows = 0, set_rows = 0, unexplained = 0;
      double worst_ulps = 0;
      for (int r = 0; r < rows_checked; ++r) {
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
              worst_ulps = std::max(worst_ulps, d);
              explained &= d <= 0.15;  // layer 3's router sees the QSA output (mlp_mix max/rms 0.15 at T = 70): 0.13 of the logit rms observed
            }
          }
        }
        unexplained += explained ? 0 : 1;
      }
      std::printf("  %-26s rows %d: order differs in %d rows, expert SET differs in %d rows (%d not explained by a router near-tie; worst %.4f of the row's logit rms from the cut)\n",
                  name.c_str(), rows, order_rows, set_rows, unexplained, worst_ulps);
      if (strict) Check(unexplained == 0, tag + " " + name + ": expert sets differ from the golden beyond router near-ties");
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
    const int64_t cols = cap.cols;
    const int64_t rows_all = cols > 0 ? n / cols : 0;
    if (QsaDependent(name) && rows_all > kDenseRows) {
      const Stats head = CompareBf16(cap.bf16.data(), ref, kDenseRows * cols);
      const Stats tail = CompareBf16(cap.bf16.data() + kDenseRows * cols, ref + kDenseRows * cols, n - kDenseRows * cols);
      PrintStats(name + " [0,67)", head);
      PrintStats(name + " [67,T) (unbounded)", tail);
      const Tol t = tol(name);
      Check(!head.nan && !tail.nan, tag + " " + name + ": NaN / Inf");
      Check(head.max_norm <= t.max_norm && head.mean_norm <= t.mean_norm,
            tag + " " + name + " dense rows: error above its bound (max/rms " + std::to_string(head.max_norm) + " > " +
                std::to_string(t.max_norm) + " or mean/rms " + std::to_string(head.mean_norm) + " > " + std::to_string(t.mean_norm) + ")");
      continue;
    }
    const Stats s = CompareBf16(cap.bf16.data(), ref, n);
    PrintStats(name, s);
    const Tol t = tol(name);
    Check(!s.nan, tag + " " + name + ": NaN / Inf");
    Check(s.max_norm <= t.max_norm && s.mean_norm <= t.mean_norm,
          tag + " " + name + ": error above its bound (max/rms " + std::to_string(s.max_norm) + " > " +
              std::to_string(t.max_norm) + " or mean/rms " + std::to_string(s.mean_norm) + " > " + std::to_string(t.mean_norm) + ")");
  }
}

void CompareStores(const CaptureStore& got, const CaptureStore& ref, const std::string& tag, Tol (*tol)(const std::string&)) {
  PrintStatsHeader();
  for (const auto& [name, cap] : got.all()) {
    if (cap.dtype != Qwen4ExpCapture::Dtype::kBf16 || !ref.Has(name)) continue;
    const Captured& r = ref.Get(name);
    if (r.bf16.size() != cap.bf16.size()) continue;
    const Stats s = CompareBf16(cap.bf16.data(), r.bf16.data(), static_cast<int64_t>(cap.bf16.size()));
    PrintStats(name, s);
    const Tol t = tol(name);
    Check(!s.nan, tag + " " + name + ": NaN / Inf");
    Check(s.max_norm <= t.max_norm && s.mean_norm <= t.mean_norm, tag + " " + name + ": error above its bound");
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
  for (const char* l : {"bf16", "w4a16"}) {
    if (!Exists(ContainerPath(l))) {
      std::printf("SKIP: %s missing\n", ContainerPath(l).c_str());
      return kSkip;
    }
  }
  const int prompts[] = {1, 7, 70, 300};
  for (int T : prompts) {
    if (!Exists(GoldenPath(T))) {
      std::printf("SKIP: %s missing\n", GoldenPath(T).c_str());
      return kSkip;
    }
  }
  if (VisibleDevices() < 1) {
    std::printf("SKIP: no HIP device\n");
    return kSkip;
  }

  // (layout, GDN prefill path, the longest prompt whose golden comparison is a hard check)
  struct Config {
    const char* layout;
    bool recurrent;
    int strict_up_to;
    const char* what;
  };
  const Config configs[] = {
      {"bf16", true, 300, "bf16, GDN prefill through the recurrent kernels (the slow reference path)"},
      {"bf16", false, 300, "bf16, GDN prefill through libr4d's span-safe chunked scan (the production path)"},
      {"w4a16", true, 0, "w4a16, GDN prefill through the recurrent kernels"},
  };
  std::map<int, CaptureStore> bf16_runs;  // the reference run per prompt (config 0), the baseline of the w4a16 comparison
  for (const Config& cfgc : configs) {
    const std::string layout = cfgc.layout;
    const bool is_bf16 = layout == "bf16";
    std::printf("\n==== %s (device %d) ====\n", cfgc.what, TestDevice());
    auto model = LoadTiny(layout, false, {}, 0, 4096, cfgc.recurrent);
    for (int T : prompts) {
      const Golden g(T, GoldenPath(T));
      const bool strict = T <= cfgc.strict_up_to;
      CaptureStore local_store;
      CaptureStore& store = (is_bf16 && cfgc.recurrent) ? bf16_runs[T] : local_store;
      store.Clear();
      model->Reset();
      model->SetCapture(store.Fn());
      const std::vector<float> logits = model->Prefill(g.ids());
      model->SetCapture(nullptr);
      const auto& tm = model->LastTimings();
      std::printf("\n-- prompt T=%d: prefill %.1f ms (stage0 %.1f, ple wait %.2f, embed %.2f, head %.1f; %lld chunk(s))%s\n", T,
                  tm.total_ms, tm.stage_ms[0], tm.ple_wait_ms, tm.embed_ms, tm.head_ms, static_cast<long long>(tm.chunks),
                  strict ? "" : "  [reported, not checked]");

      std::printf(" vs HF golden (recent):\n");
      CompareToGolden(store, g, layout + (cfgc.recurrent ? " rec" : " chunk") + " T" + std::to_string(T), strict ? GoldenTol : ReportOnlyTol, strict);
      if (!is_bf16) {
        std::printf(" vs the bf16 container (same GDN path):\n");
        CompareStores(store, bf16_runs[T], std::string("w4a16-vs-bf16 T") + std::to_string(T), QuantTol);
      }

      // last-row logits
      int64_t n = 0;
      const uint16_t* gl = g.Bf16("logits", &n);
      const int64_t vocab = n / T;
      const std::vector<float> ref_last = WidenBf16(gl + (T - 1) * vocab, vocab);
      const FStats fs = CompareF32(logits.data(), ref_last.data(), vocab);
      std::printf(" last-row logits: max|err| %.4g mean|err| %.4g rms(ref) %.4g (max/rms %.4f); argmax ours %lld golden %lld\n",
                  fs.max_abs, fs.mean_abs, fs.rms_ref, fs.max_norm, static_cast<long long>(ArgMax(logits.data(), vocab)),
                  static_cast<long long>(ArgMax(ref_last.data(), vocab)));
      // the margin between the golden's top-2 (what an argmax check can tolerate)
      std::vector<float> sorted = ref_last;
      std::partial_sort(sorted.begin(), sorted.begin() + 2, sorted.end(), std::greater<float>());
      std::printf("   golden top-1 - top-2 logit margin %.4g\n", sorted[0] - sorted[1]);
      if (strict) {
        Check(ArgMax(logits.data(), vocab) == ArgMax(ref_last.data(), vocab), "T" + std::to_string(T) + ": last-row argmax differs from the golden");
      }
    }
  }
  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
