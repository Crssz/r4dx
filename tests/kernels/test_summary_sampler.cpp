// tests/kernels/test_summary_sampler.cpp -- CPU-only (no GPU).
//
// Milestone 6 S1 item 3 (docs/sampling.md "When the summary is enough"):
// r4dx::kernels::SampleFromSummary must be EXACT or SILENT. Whenever it returns a token, that
// token must equal what the full-vocab canonical sampler would have returned for the same u --
// exactly, zero mismatches tolerated -- and whenever it cannot prove the answer from the top-K
// summary it must say so rather than guess.
//
// Three logit shapes, chosen so the summary is provably enough in one, provably NOT enough in the
// others, and marginal at the boundary:
//   * "peaked"      -- a realistic Zipf-like decode row; every filter closes the set inside K.
//   * "flat"        -- near-uniform over the whole vocabulary; a bare top_p nucleus and a
//                      pure-temperature walk both reach far past K.
//   * "wide nucleus"-- shaped so top_p=0.9 needs a few hundred tokens: inside K for the tighter
//                      nucleus, outside it for the looser one.
// Plus the explicit boundary cases: u landing exactly on a CDF step, top_k == K, top_k > K, and an
// lse error placed exactly on the nucleus threshold (which must produce a fallback, not a guess).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "r4dx/kernels/summary_sampler.hpp"

using namespace r4dx::kernels;

namespace {

constexpr int64_t kVocab = 2000;
constexpr int kK = kRowSummaryMaxK;
constexpr int kUs = 100000;  // 50000 on a uniform grid + 50000 random, per (shape, combination)

int g_failures = 0;

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

// The CPU stand-in for r4dx_topk_lse_f32: same canonical order (value descending, ties toward the
// lower id) and the same stable logsumexp, computed in fp64 and rounded to the fp32 the kernel
// writes. tests/kernels/test_topk_lse.cpp is what pins the DEVICE kernel to this contract; this
// file is about what the host does with the summary.
RowSummary MakeSummary(const std::vector<float>& logits, int64_t vocab, float inv_t) {
  RowSummary s;
  s.k = kK;
  s.vocab = vocab;
  s.inv_temperature = inv_t;
  std::vector<int32_t> order(static_cast<size_t>(vocab));
  std::iota(order.begin(), order.end(), 0);
  std::partial_sort(order.begin(), order.begin() + kK, order.end(), [&](int32_t a, int32_t b) {
    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
  });
  for (int j = 0; j < kK; ++j) {
    s.ids[j] = order[j];
    s.vals[j] = logits[static_cast<size_t>(order[j])];
  }
  const double m = static_cast<double>(s.vals[0]);
  double sum = 0.0, comp = 0.0;
  for (int64_t i = 0; i < vocab; ++i) {
    const double term = std::exp((static_cast<double>(logits[static_cast<size_t>(i)]) - m) *
                                 static_cast<double>(inv_t));
    const double y = term - comp;
    const double t = sum + y;
    comp = (t - sum) - y;
    sum = t;
  }
  s.lse = static_cast<float>(m * static_cast<double>(inv_t) + std::log(sum));
  return s;
}

std::vector<float> MakePeaked() {
  std::vector<float> l(static_cast<size_t>(kVocab));
  std::mt19937 rng(11u);
  std::uniform_real_distribution<float> jitter(-0.3f, 0.3f);
  for (int64_t i = 0; i < kVocab; ++i) {
    l[static_cast<size_t>(i)] =
        static_cast<float>(10.0 - 2.4 * std::log(static_cast<double>(i + 1))) + jitter(rng);
  }
  return l;
}

std::vector<float> MakeFlat() {
  std::vector<float> l(static_cast<size_t>(kVocab));
  std::mt19937 rng(12u);
  std::uniform_real_distribution<float> jitter(-0.02f, 0.02f);
  for (auto& v : l) v = 1.0f + jitter(rng);
  return l;
}

std::vector<float> MakeWideNucleus() {
  // Slow decay: ~300 tokens carry 90% of the mass at T=1, so top_p=0.9 reaches well past K while
  // top_k=20 and min_p=0.05 still close inside it.
  std::vector<float> l(static_cast<size_t>(kVocab));
  std::mt19937 rng(13u);
  std::uniform_real_distribution<float> jitter(-0.05f, 0.05f);
  for (int64_t i = 0; i < kVocab; ++i) {
    l[static_cast<size_t>(i)] =
        static_cast<float>(4.0 - 0.45 * std::log(static_cast<double>(i + 1))) + jitter(rng);
  }
  return l;
}

struct Combo {
  const char* name;
  SampleParams p;
  bool must_always_resolve;  // the provable region: a fallback here is a performance defect
};

// A deterministic mix of grid and random u values in [0,1).
std::vector<double> MakeUs() {
  std::vector<double> us;
  us.reserve(kUs);
  for (int i = 0; i < kUs / 2; ++i) us.push_back(static_cast<double>(i) / (kUs / 2));
  std::mt19937_64 rng(777);
  for (int i = 0; i < kUs / 2; ++i) us.push_back(DrawUniform01(rng));
  return us;
}

struct SweepResult {
  int64_t resolved = 0;
  int64_t fallback = 0;
  int64_t mismatch = 0;
  int32_t first_bad_u_index = -1;
};

SweepResult Sweep(const std::vector<float>& logits, const RowSummary& s, const SampleParams& p,
                  const std::vector<double>& us) {
  SweepResult r;
  for (size_t i = 0; i < us.size(); ++i) {
    const SummarySampleResult got = SampleFromSummary(s, p, us[i]);
    if (!got.resolved) {
      ++r.fallback;
      continue;
    }
    ++r.resolved;
    const int32_t want = SampleCanonical(logits.data(), s.vocab, p, us[i]);
    if (got.token != want) {
      if (r.first_bad_u_index < 0) r.first_bad_u_index = static_cast<int32_t>(i);
      ++r.mismatch;
    }
  }
  return r;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);

  const Combo combos[] = {
      {"T=1.0 (pure temperature)", [] { SampleParams p; p.temperature = 1.0f; return p; }(), false},
      {"T=0.7 top_k=20 top_p=0.8",
       [] { SampleParams p; p.temperature = 0.7f; p.top_k = 20; p.top_p = 0.8f; return p; }(), true},
      {"T=0.6 top_k=20 top_p=0.95",
       [] { SampleParams p; p.temperature = 0.6f; p.top_k = 20; p.top_p = 0.95f; return p; }(),
       true},
      {"T=1.0 top_p=0.9", [] { SampleParams p; p.temperature = 1.0f; p.top_p = 0.9f; return p; }(),
       false},
      {"T=0.8 min_p=0.05",
       [] { SampleParams p; p.temperature = 0.8f; p.min_p = 0.05f; return p; }(), false},
      {"T=0.7 top_k=50 min_p=0.02 top_p=0.9",
       [] {
         SampleParams p;
         p.temperature = 0.7f;
         p.top_k = 50;
         p.min_p = 0.02f;
         p.top_p = 0.9f;
         return p;
       }(),
       true},
  };

  const std::vector<double> us = MakeUs();
  struct Shape {
    const char* name;
    std::vector<float> logits;
  };
  const Shape shapes[] = {
      {"peaked", MakePeaked()},
      {"flat", MakeFlat()},
      {"wide nucleus", MakeWideNucleus()},
  };

  std::printf("%d values of u per cell (%d grid + %d random), V=%lld, K=%d\n\n", kUs, kUs / 2,
              kUs / 2, static_cast<long long>(kVocab), kK);
  std::printf("%-14s %-38s %10s %10s %10s\n", "shape", "filters", "resolved", "fallback%",
              "mismatch");

  int64_t total_mismatch = 0;
  for (const Shape& sh : shapes) {
    for (const Combo& c : combos) {
      const RowSummary summary = MakeSummary(sh.logits, kVocab, 1.0f / c.p.temperature);
      const SweepResult r = Sweep(sh.logits, summary, c.p, us);
      total_mismatch += r.mismatch;
      std::printf("%-14s %-38s %10lld %9.2f%% %10lld\n", sh.name, c.name,
                  static_cast<long long>(r.resolved),
                  100.0 * static_cast<double>(r.fallback) / static_cast<double>(us.size()),
                  static_cast<long long>(r.mismatch));
      if (r.mismatch != 0) {
        const double u = us[static_cast<size_t>(r.first_bad_u_index)];
        std::fprintf(stderr, "    first mismatch at u=%.17g: summary=%d full=%d\n", u,
                     SampleFromSummary(summary, c.p, u).token,
                     SampleCanonical(sh.logits.data(), kVocab, c.p, u));
      }
      // The provable region: a filter that closes the candidate set inside the top-K must NEVER
      // need the full row, whatever the logits look like.
      if (c.must_always_resolve) {
        Check(r.fallback == 0, std::string(sh.name) + " / " + c.name +
                                   ": closes inside the top-K, so no fallback is ever needed");
      }
    }
  }
  Check(total_mismatch == 0,
        "every resolved summary token equals the full-vocab canonical token (exact)");

  // Fallback rate on a realistic peaked row, for the record (a performance property, not a
  // contract): reported above per cell; the peaked row's pure-temperature and top_p cells are the
  // interesting ones.

  // ---- Boundary cases ---------------------------------------------------------------------------
  std::printf("\n=== boundary cases ===\n");
  const std::vector<float> peaked = MakePeaked();

  // (a) u exactly ON a CDF step. The walk rule is `target < cum`, so u == cum_j/S must select slot
  //     j+1, not j -- and the summary path and the full path must agree on which.
  {
    SampleParams p;
    p.temperature = 0.7f;
    p.top_k = 20;
    const RowSummary s = MakeSummary(peaked, kVocab, 1.0f / p.temperature);
    const float max_s = s.vals[0] / p.temperature;
    double sum = 0.0;
    std::vector<double> w(20);
    for (int j = 0; j < 20; ++j) {
      w[j] = static_cast<double>(std::exp(s.vals[j] / p.temperature - max_s));
      sum += w[j];
    }
    bool ok = true;
    double cum = 0.0;
    for (int j = 0; j < 19; ++j) {
      cum += w[j];
      const double u = cum / sum;  // exactly a step (up to the division's own rounding)
      const SummarySampleResult got = SampleFromSummary(s, p, u);
      const int32_t want = SampleCanonical(peaked.data(), kVocab, p, u);
      ok = ok && got.resolved && got.token == want;
      // ... and the two neighbours of that u.
      for (double du : {-1e-12, 1e-12}) {
        const double u2 = u + du;
        if (u2 < 0.0 || u2 >= 1.0) continue;
        const SummarySampleResult g2 = SampleFromSummary(s, p, u2);
        ok = ok && g2.resolved && g2.token == SampleCanonical(peaked.data(), kVocab, p, u2);
      }
    }
    Check(ok, "u exactly on a CDF step (and either side of it) agrees with the full path");
  }

  // (b) top_k == K resolves; top_k == K+1 with no other filter must ask for the fallback; the same
  //     top_k > K DOES resolve once another filter closes the set inside the top-K.
  {
    const RowSummary s = MakeSummary(peaked, kVocab, 1.0f / 0.8f);
    SampleParams at_k;
    at_k.temperature = 0.8f;
    at_k.top_k = kK;
    SampleParams past_k;
    past_k.temperature = 0.8f;
    past_k.top_k = kK + 1;
    SampleParams past_k_closed = past_k;
    past_k_closed.min_p = 0.05f;  // bites well inside the top-K on this peaked row

    bool at_ok = true, past_ok = true, closed_ok = true;
    for (double u : us) {
      const SummarySampleResult a = SampleFromSummary(s, at_k, u);
      at_ok = at_ok && a.resolved &&
              a.token == SampleCanonical(peaked.data(), kVocab, at_k, u);
      past_ok = past_ok && !SampleFromSummary(s, past_k, u).resolved;
      const SummarySampleResult c = SampleFromSummary(s, past_k_closed, u);
      closed_ok = closed_ok && c.resolved &&
                  c.token == SampleCanonical(peaked.data(), kVocab, past_k_closed, u);
    }
    Check(at_ok, "top_k == K always resolves and matches the full path");
    Check(past_ok, "top_k == K+1 with no other filter always asks for the fallback");
    Check(closed_ok, "top_k > K still resolves when min_p closes the set inside the top-K");
  }

  // (c) An lse error that puts the top-K's mass exactly ON the top_p threshold must produce a
  //     FALLBACK, not a guess. Construct the summary's lse so that top_p * S_full lands exactly on
  //     the cumulative mass of the nucleus boundary; then widen it well past the band and check the
  //     answer comes back.
  {
    SampleParams p;
    p.temperature = 1.0f;
    p.top_p = 0.9f;
    RowSummary s = MakeSummary(peaked, kVocab, 1.0f / p.temperature);
    const float max_s = s.vals[0] / p.temperature;
    std::vector<double> w(kK);
    for (int j = 0; j < kK; ++j) w[j] = std::exp(s.vals[j] / p.temperature - max_s);

    // The true nucleus boundary under the honest summary.
    const double s_true = std::exp(static_cast<double>(s.lse) -
                                   static_cast<double>(s.vals[0]) *
                                       static_cast<double>(s.inv_temperature));
    double cum = 0.0;
    int n = -1;
    for (int j = 0; j < kK; ++j) {
      cum += w[j];
      if (cum >= static_cast<double>(p.top_p) * s_true) {
        n = j + 1;
        break;
      }
    }
    Check(n > 0, "the peaked row's top_p=0.9 nucleus is inside the top-K (test setup)");
    if (n > 0) {
      Check(SampleFromSummary(s, p, 0.5).resolved,
            "an honest lse resolves the top_p=0.9 nucleus on the peaked row");
      // Poison lse so that top_p * S_full == cum_n exactly: S_full = cum_n / top_p.
      const double poisoned_s_full = cum / static_cast<double>(p.top_p);
      RowSummary bad = s;
      bad.lse = static_cast<float>(std::log(poisoned_s_full) +
                                   static_cast<double>(s.vals[0]) *
                                       static_cast<double>(s.inv_temperature));
      bool all_fallback = true;
      for (double u : us) all_fallback = all_fallback && !SampleFromSummary(bad, p, u).resolved;
      Check(all_fallback,
            "an lse that puts the nucleus mass exactly on the threshold always falls back");
      // A 10x-the-band error in the SAFE direction (the top-K provably overshoots the nucleus)
      // must still resolve -- the band must be conservative, not paralysing.
      RowSummary loose = s;
      loose.lse = static_cast<float>(std::log(poisoned_s_full * (1.0 - 50.0 * kRowSummaryLseRelTol)) +
                                     static_cast<double>(s.vals[0]) *
                                         static_cast<double>(s.inv_temperature));
      Check(SampleFromSummary(loose, p, 0.5).resolved,
            "an lse comfortably clear of the threshold still resolves");
    }
  }

  // (d) A pure-temperature walk whose u lands in the tail beyond the top-K must fall back, and one
  //     that lands in the head must not.
  {
    SampleParams p;
    p.temperature = 1.0f;
    const RowSummary s = MakeSummary(peaked, kVocab, 1.0f / p.temperature);
    Check(SampleFromSummary(s, p, 0.0).resolved, "pure temperature, u=0 resolves from the summary");
    Check(!SampleFromSummary(s, p, 0.9999999).resolved,
          "pure temperature, u deep in the tail asks for the fallback");
    const RowSummary flat_s = MakeSummary(MakeFlat(), kVocab, 1.0f / p.temperature);
    int64_t flat_resolved = 0;
    for (double u : us) flat_resolved += SampleFromSummary(flat_s, p, u).resolved ? 1 : 0;
    std::printf("  flat row, pure temperature: %lld / %lld resolved from the summary\n",
                static_cast<long long>(flat_resolved), static_cast<long long>(us.size()));
    Check(flat_resolved < static_cast<int64_t>(us.size()) / 10,
          "a flat row's pure-temperature walk is almost never provable from 64 entries");
  }

  // (e) A summary that covers the whole row (vocab <= K) is always closed.
  {
    std::vector<float> tiny(40);
    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-3.0f, 3.0f);
    for (auto& v : tiny) v = d(rng);
    RowSummary s;
    s.k = 40;
    s.vocab = 40;
    s.inv_temperature = 1.0f;
    std::vector<int32_t> order(40);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
      return tiny[a] > tiny[b] || (tiny[a] == tiny[b] && a < b);
    });
    double sum = 0.0;
    for (int j = 0; j < 40; ++j) {
      s.ids[j] = order[j];
      s.vals[j] = tiny[static_cast<size_t>(order[j])];
    }
    for (int j = 0; j < 40; ++j) sum += std::exp(static_cast<double>(s.vals[j] - s.vals[0]));
    s.lse = static_cast<float>(static_cast<double>(s.vals[0]) + std::log(sum));
    SampleParams p;
    p.temperature = 1.0f;
    bool ok = true;
    for (int i = 0; i < 10000; ++i) {
      const double u = static_cast<double>(i) / 10000.0;
      const SummarySampleResult got = SampleFromSummary(s, p, u);
      ok = ok && got.resolved && got.token == SampleCanonical(tiny.data(), 40, p, u);
    }
    Check(ok, "a summary covering the whole row resolves every u with no lse dependence");
  }

  // (f) Greedy (temperature <= 0) reads straight off slot 0.
  {
    const RowSummary s = MakeSummary(peaked, kVocab, 1.0f);
    SampleParams g;
    g.temperature = 0.0f;
    const SummarySampleResult got = SampleFromSummary(s, g, 0.5);
    Check(got.resolved && got.token == SampleCanonical(peaked.data(), kVocab, g, 0.5),
          "temperature <= 0 resolves to the argmax from slot 0");
  }

  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    std::printf("FAIL\n");
    return 1;
  }
  std::printf("\nALL PASS\n");
  return 0;
}
