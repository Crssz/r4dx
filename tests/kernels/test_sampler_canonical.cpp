// tests/kernels/test_sampler_canonical.cpp -- CPU-only (no GPU).
//
// Milestone 6 S1 item 2 (docs/sampling.md "Canonical sampling"): the canonical rewrite
// (r4dx::kernels::SampleCanonical, which maps ONE uniform draw u to a token by walking the
// post-filter CDF in canonical order) must have exactly the DISTRIBUTION the pre-M6 sampler had.
// Only the map from a seed to a token is allowed to change.
//
// The test therefore checks BOTH samplers against the SAME third thing -- the exact post-filter
// distribution computed independently in fp64 from the filter semantics sampler.hpp documents --
// rather than checking the new one against the old one. Checking new-vs-old would pass if both had
// drifted together; checking each against the exact distribution catches drift in either.
//
// Per filter combination:
//   * 400000 draws from SampleCanonical, chi-square against the exact distribution;
//   * 400000 draws from SampleLegacy (the pre-M6 function, kept verbatim in sampler.hpp),
//     chi-square against the same exact distribution;
//   * the candidate SET (tokens with non-zero post-filter probability) must be identical between
//     the two samplers and equal to the exact distribution's support.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "r4dx/kernels/sampler.hpp"

using namespace r4dx::kernels;

namespace {

constexpr int64_t kVocab = 2000;
constexpr int kDraws = 400000;

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

// A Zipf-like decode row plus a deliberate NEAR-tie cluster: 24 tokens in the middle of the
// candidate range whose logits differ by 1e-4, which is ~400 fp32 ulps at these magnitudes -- far
// enough apart that dividing by any tested temperature keeps them distinct floats (so the
// candidate set is unambiguous and the two samplers must agree on it exactly), yet close enough
// that their probabilities are equal to 5 significant figures and any order-dependent bias in the
// CDF walk would show up as a chi-square failure.
std::vector<float> MakeLogits() {
  std::vector<float> logits(static_cast<size_t>(kVocab));
  std::mt19937 rng(424242u);
  std::uniform_real_distribution<float> jitter(-0.35f, 0.35f);
  for (int64_t i = 0; i < kVocab; ++i) {
    logits[static_cast<size_t>(i)] =
        static_cast<float>(8.0 - 1.4 * std::log(static_cast<double>(i + 1))) + jitter(rng);
  }
  for (int j = 0; j < 24; ++j) {
    logits[static_cast<size_t>(10 + j)] = 5.0f + static_cast<float>(j) * 1e-4f;
  }
  return logits;
}

struct ExactDist {
  std::vector<int32_t> ids;  // support, canonical order
  std::vector<double> p;     // normalised, same indexing as `ids`
};

// The filter pipeline of sampler.hpp, recomputed in fp64: temperature -> top_k -> min_p (relative
// to the UNTRUNCATED max) -> top_p (nucleus over the canonical-order prefix) -> renormalise.
ExactDist Exact(const std::vector<float>& logits, const SampleParams& params) {
  const double t = static_cast<double>(params.temperature);
  std::vector<int32_t> order(static_cast<size_t>(kVocab));
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
  });
  double max_scaled = -std::numeric_limits<double>::infinity();
  for (int64_t i = 0; i < kVocab; ++i) {
    max_scaled = std::max(max_scaled, static_cast<double>(logits[i]) / t);
  }
  const auto w = [&](int32_t id) {
    return std::exp(static_cast<double>(logits[id]) / t - max_scaled);
  };

  std::vector<int32_t> cand = order;
  if (params.top_k > 0 && params.top_k < static_cast<int>(kVocab)) {
    cand.resize(static_cast<size_t>(params.top_k));
  }
  if (params.min_p > 0.0f) {
    size_t keep = 0;
    while (keep < cand.size() && w(cand[keep]) >= static_cast<double>(params.min_p)) ++keep;
    cand.resize(keep);
  }
  if (params.top_p < 1.0f) {
    double sum = 0.0;
    for (int32_t id : cand) sum += w(id);
    const double threshold = static_cast<double>(params.top_p) * sum;
    double cum = 0.0;
    size_t keep = cand.size();
    for (size_t j = 0; j < cand.size(); ++j) {
      cum += w(cand[j]);
      if (cum >= threshold) {
        keep = j + 1;
        break;
      }
    }
    cand.resize(keep);
  }

  ExactDist out;
  out.ids = cand;
  double sum = 0.0;
  for (int32_t id : cand) sum += w(id);
  out.p.reserve(cand.size());
  for (int32_t id : cand) out.p.push_back(w(id) / sum);
  return out;
}

// Upper 1-alpha quantile of chi-square with `df` degrees of freedom, Wilson-Hilferty cube-root
// approximation (accurate to well under 2% for df >= 10, which is every case here). alpha = 1e-6:
// with 12 chi-square tests in this file the family-wise false-alarm probability is ~1.2e-5, so a
// green run is not luck and a red run is not noise.
double Chi2Threshold(int df) {
  const double z = 4.753424309; /* Phi^-1(1 - 1e-6) */
  const double a = 2.0 / (9.0 * df);
  const double h = 1.0 - a + z * std::sqrt(a);
  return df * h * h * h;
}

struct Chi2Result {
  double stat = 0.0;
  double threshold = 0.0;
  int df = 0;
  int pooled_bins = 0;
};

// Pearson chi-square of `counts` (indexed like dist.ids) against `dist`, pooling every cell whose
// expected count is below 5 into a single tail cell -- the standard rule, and necessary here
// because a 2000-token pure-temperature support has a long tail of cells with E << 1 that would
// otherwise dominate the statistic with pure discreteness noise.
Chi2Result Chi2(const std::vector<int64_t>& counts, const ExactDist& dist, int64_t n) {
  Chi2Result r;
  double pooled_e = 0.0;
  int64_t pooled_o = 0;
  double stat = 0.0;
  int bins = 0;
  for (size_t j = 0; j < dist.ids.size(); ++j) {
    const double e = static_cast<double>(n) * dist.p[j];
    if (e >= 5.0) {
      const double d = static_cast<double>(counts[j]) - e;
      stat += d * d / e;
      ++bins;
    } else {
      pooled_e += e;
      pooled_o += counts[j];
      ++r.pooled_bins;
    }
  }
  if (pooled_e > 0.0) {
    const double d = static_cast<double>(pooled_o) - pooled_e;
    stat += d * d / pooled_e;
    ++bins;
  }
  r.stat = stat;
  r.df = bins - 1;
  r.threshold = Chi2Threshold(r.df);
  return r;
}

struct Combo {
  const char* name;
  SampleParams p;
};

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  const std::vector<float> logits = MakeLogits();

  const Combo combos[] = {
      {"T=1.0 (pure temperature)", [] { SampleParams p; p.temperature = 1.0f; return p; }()},
      {"T=0.7 top_k=20 top_p=0.8",
       [] { SampleParams p; p.temperature = 0.7f; p.top_k = 20; p.top_p = 0.8f; return p; }()},
      {"T=0.6 top_k=20 top_p=0.95",
       [] { SampleParams p; p.temperature = 0.6f; p.top_k = 20; p.top_p = 0.95f; return p; }()},
      {"T=1.0 top_p=0.9", [] { SampleParams p; p.temperature = 1.0f; p.top_p = 0.9f; return p; }()},
      {"T=0.8 min_p=0.05",
       [] { SampleParams p; p.temperature = 0.8f; p.min_p = 0.05f; return p; }()},
      {"T=0.7 top_k=50 min_p=0.02 top_p=0.9",
       [] {
         SampleParams p;
         p.temperature = 0.7f;
         p.top_k = 50;
         p.min_p = 0.02f;
         p.top_p = 0.9f;
         return p;
       }()},
  };

  for (const Combo& c : combos) {
    std::printf("\n=== %s ===\n", c.name);
    const ExactDist dist = Exact(logits, c.p);
    std::vector<int32_t> slot(static_cast<size_t>(kVocab), -1);
    for (size_t j = 0; j < dist.ids.size(); ++j) slot[static_cast<size_t>(dist.ids[j])] = static_cast<int32_t>(j);
    std::printf("  exact support: %d tokens, p_max=%.6g p_min=%.6g\n",
                static_cast<int>(dist.ids.size()), dist.p.front(), dist.p.back());

    // ---- candidate-set equality (new vs old vs the exact support) -------------------------------
    {
      std::mt19937_64 r1(1), r2(1);
      std::vector<int32_t> cand_new, cand_old;
      SampleCanonical(logits.data(), kVocab, c.p, DrawUniform01(r1), &cand_new);
      SampleLegacy(logits.data(), kVocab, c.p, r2, &cand_old);
      std::sort(cand_new.begin(), cand_new.end());
      std::sort(cand_old.begin(), cand_old.end());
      std::vector<int32_t> cand_exact = dist.ids;
      std::sort(cand_exact.begin(), cand_exact.end());
      Check(cand_new == cand_old,
            std::string(c.name) + ": candidate set identical between SampleCanonical and "
                                  "SampleLegacy (" + std::to_string(cand_new.size()) + " vs " +
                std::to_string(cand_old.size()) + " tokens)");
      Check(cand_new == cand_exact,
            std::string(c.name) + ": candidate set equals the fp64 exact support");
    }

    // ---- 400000 draws from each sampler, chi-square against the exact distribution --------------
    std::vector<int64_t> counts_new(dist.ids.size(), 0), counts_old(dist.ids.size(), 0);
    int64_t out_of_support_new = 0, out_of_support_old = 0;
    {
      std::mt19937_64 rng(0xC0FFEEu);
      for (int i = 0; i < kDraws; ++i) {
        const int32_t tok = SampleCanonical(logits.data(), kVocab, c.p, DrawUniform01(rng));
        const int32_t s = (tok >= 0 && tok < kVocab) ? slot[static_cast<size_t>(tok)] : -1;
        if (s < 0) {
          ++out_of_support_new;
        } else {
          ++counts_new[static_cast<size_t>(s)];
        }
      }
    }
    {
      std::mt19937_64 rng(0xBADC0DEu);
      for (int i = 0; i < kDraws; ++i) {
        const int32_t tok = SampleLegacy(logits.data(), kVocab, c.p, rng);
        const int32_t s = (tok >= 0 && tok < kVocab) ? slot[static_cast<size_t>(tok)] : -1;
        if (s < 0) {
          ++out_of_support_old;
        } else {
          ++counts_old[static_cast<size_t>(s)];
        }
      }
    }
    Check(out_of_support_new == 0,
          std::string(c.name) + ": SampleCanonical never draws outside the support");
    Check(out_of_support_old == 0,
          std::string(c.name) + ": SampleLegacy never draws outside the support");

    const Chi2Result cn = Chi2(counts_new, dist, kDraws);
    const Chi2Result co = Chi2(counts_old, dist, kDraws);
    std::printf("  SampleCanonical: chi2=%.2f df=%d threshold(alpha=1e-6)=%.2f (%d cells pooled)\n",
                cn.stat, cn.df, cn.threshold, cn.pooled_bins);
    std::printf("  SampleLegacy   : chi2=%.2f df=%d threshold(alpha=1e-6)=%.2f (%d cells pooled)\n",
                co.stat, co.df, co.threshold, co.pooled_bins);
    Check(cn.stat <= cn.threshold,
          std::string(c.name) + ": SampleCanonical matches the exact distribution");
    Check(co.stat <= co.threshold,
          std::string(c.name) + ": SampleLegacy matches the exact distribution");
  }

  // ---- The properties the pre-M6 test pinned, re-checked on the canonical sampler ---------------
  std::printf("\n=== invariants ===\n");
  {
    const std::vector<float> small = {0.1f, 5.0f, -2.0f, 4.9f, 3.0f};
    SampleParams greedy;
    greedy.temperature = 0.0f;
    std::mt19937_64 r = MakeRng(0);
    const std::mt19937_64 before = r;
    Check(Sample(small.data(), small.size(), greedy, r) == 1, "temperature<=0 is greedy argmax");
    Check(r == before, "the greedy path consumes no rng draw");

    SampleParams top1;
    top1.temperature = 1.0f;
    top1.top_k = 1;
    bool all_argmax = true;
    for (int i = 0; i < 1000; ++i) {
      all_argmax &= (SampleCanonical(small.data(), small.size(), top1,
                                     static_cast<double>(i) / 1000.0) == 1);
    }
    Check(all_argmax, "top_k=1 always returns the argmax, for every u");

    // Exactly one rng draw per emitted token.
    std::mt19937_64 a = MakeRng(99), b = MakeRng(99);
    SampleParams plain;
    for (int i = 0; i < 50; ++i) Sample(small.data(), small.size(), plain, a);
    b.discard(50);
    Check(a == b, "Sample() consumes exactly one rng draw per token");

    // u == 0 must pick the canonical first token; u -> 1 the canonical last one.
    SampleParams k3;
    k3.temperature = 1.0f;
    k3.top_k = 3;
    Check(SampleCanonical(small.data(), small.size(), k3, 0.0) == 1, "u=0 picks the canonical head");
    Check(SampleCanonical(small.data(), small.size(), k3, 0.9999999999) == 4,
          "u->1 picks the canonical tail");

    // Ties resolve toward the LOWER id, deterministically, for every u.
    const std::vector<float> tied = {2.0f, 2.0f, 2.0f, 2.0f};
    SampleParams tie_k1;
    tie_k1.temperature = 1.0f;
    tie_k1.top_k = 1;
    bool tie_ok = true;
    for (int i = 0; i < 100; ++i) {
      tie_ok &= (SampleCanonical(tied.data(), tied.size(), tie_k1,
                                 static_cast<double>(i) / 100.0) == 0);
    }
    Check(tie_ok, "an exact tie resolves toward the lowest id");
  }

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    std::printf("FAIL\n");
    return 1;
  }
  std::printf("\nALL PASS\n");
  return 0;
}
