// r4dx::kernels -- exact sampling from a DEVICE ROW SUMMARY (docs/sampling.md, Milestone 6 S1).
//
// The summary is what r4dx_topk_lse_f32 (src/kernels/include/r4dx/kernels/kernels.h) writes per
// row: the top-K raw logits with their ids in canonical order, plus the row's logsumexp of the
// temperature-scaled logits. That is K*8 + 4 = 516 B per row (K=64) instead of ~993 KB.
//
// SampleFromSummary returns the SAME token SampleCanonical (sampler.hpp) would return for the same
// logits and the same u -- or says "I cannot prove it" and asks the caller to fall back to the
// full-vocab path for that row only. It never guesses: every case below either reconstructs the
// post-filter candidate set EXACTLY (bit-identical softmax numerators, bit-identical partial sums,
// bit-identical comparisons) or reports `resolved == false`.
//
// Header-only and HIP-free on purpose: the server/CLI sampling path includes it without pulling in
// any device dependency, and tests can drive it with synthetic summaries.
#pragma once

#include <cmath>
#include <cstdint>

#include "r4dx/kernels/sampler.hpp"

namespace r4dx::kernels {

// Must match kernels.h's R4DX_TOPK_LSE_K. Kept as its own constant so this header stays HIP-free
// (kernels.h declares device entry points); the two are tied together by a static_assert wherever
// both are visible (tests/kernels/test_topk_lse.cpp).
inline constexpr int kRowSummaryMaxK = 64;

// One decode row's device summary. `vals`/`ids` are the top-`k` entries of the row in CANONICAL
// order (raw logit descending, ties toward the lower id -- detail::CanonicalOrder in sampler.hpp);
// `lse` is log(sum_i exp(logit_i * inv_temperature)) over the FULL row.
struct RowSummary {
  int k = 0;                      // valid entries in ids/vals, <= kRowSummaryMaxK
  int64_t vocab = 0;              // the full row width the summary was taken from
  float inv_temperature = 1.0f;   // the scale `lse` was accumulated at
  float lse = 0.0f;
  int32_t ids[kRowSummaryMaxK] = {};
  float vals[kRowSummaryMaxK] = {};
};

struct SummarySampleResult {
  bool resolved = false;  // false => the caller must run the full-vocab path for this row
  int32_t token = -1;
};

// Relative uncertainty band applied to the full-row partition function recovered from `lse`.
//
// Everything SampleFromSummary computes from the top-K itself is bit-exact; the ONLY approximate
// quantity is S_full = exp(lse - max_scaled), and it is only needed by the two cases whose
// threshold depends on the mass OUTSIDE the top-K (a bare top_p nucleus, and the plain
// pure-temperature walk). Sources of error, all bounded well inside this band:
//   * the kernel's own lse error, <= 1e-4 ABSOLUTE (tests/kernels/test_topk_lse.cpp), i.e. <= 1e-4
//     relative once exponentiated;
//   * the kernel scaling by `inv_temperature` where the host divides by `temperature`
//     (x*(1/T) vs x/T differ by <= |x/T| * 2^-24, i.e. ~1e-5 relative at |x/T| ~ 200);
//   * fp32 expf differing between device and host, and the two paths summing in different orders
//     (~1e-7 relative).
// 1e-3 leaves roughly 5x headroom over the sum of those -- CONDITIONAL on |max_logit/temperature|
// staying under ~1e4 (the second bullet's ~1e-5-relative term grows with |x/T|, measured 1.7e-6 at
// T=1.0 but 1.5e-3 at T=0.001 with sigma-12 logits, i.e. it can exceed this band below roughly
// T~0.002 on this model's logit scale). Model::SummaryInvTemperature screens temperatures below
// kMinSummaryTemperature (model.cpp) to a full-logits fallback for exactly this reason, so a caller
// that goes through Model never exercises this header below the temperature this band was measured
// to hold at; a caller that constructs a RowSummary directly at an extreme temperature is not
// covered by that screen and should apply an equivalent one itself. Widening the band costs only a
// slightly higher fallback rate; narrowing it risks returning a token the full-vocab path would not
// have.
inline constexpr double kRowSummaryLseRelTol = 1e-3;

// token == SampleCanonical(full row, params, u), or resolved == false.
//
// Provability, case by case (docs/sampling.md "When the summary is enough"):
//   * top_k in [1, K]                  -- always exact: the candidate set is a prefix of the
//                                         top-K, so its mass is known exactly.
//   * min_p > 0 with w_K < min_p       -- exact: min_p cuts inside the top-K, so the surviving set
//                                         (a prefix, since w is non-increasing) is fully visible.
//   * top_p < 1, no other filter       -- exact iff the nucleus boundary is provably inside the
//                                         top-K under the S_full band.
//   * pure temperature                 -- exact iff u*S_full provably lands on a step inside the
//                                         top-K, i.e. u is not in the tail.
//   * top_k > K, or min_p that does not bite inside the top-K, and nothing else closes the set
//                                      -- NOT provable, fall back.
inline SummarySampleResult SampleFromSummary(const RowSummary& s, const SampleParams& p, double u) {
  SummarySampleResult r;
  if (s.k <= 0 || s.vocab <= 0 || s.k > kRowSummaryMaxK) return r;  // unusable summary
  if (p.temperature <= 0.0f) {                                      // greedy: the argmax is slot 0
    r.resolved = true;
    r.token = s.ids[0];
    return r;
  }

  const int K = s.k;
  // Exactly SampleCanonical's arithmetic, restricted to the rows the summary carries. vals[0] is
  // the row's raw max, so max_s is the same float SampleCanonical computes over the full row and
  // w[0] is exactly 1.0.
  const float max_s = s.vals[0] / p.temperature;
  float w[kRowSummaryMaxK];
  for (int j = 0; j < K; ++j) w[j] = std::exp(s.vals[j] / p.temperature - max_s);

  const bool need_topk = p.top_k > 0 && p.top_k < static_cast<int>(s.vocab);
  const bool need_topp = p.top_p < 1.0f;
  const bool need_minp = p.min_p > 0.0f;

  int limit = K;                          // entries of the top-K still in play
  bool closed = (static_cast<int64_t>(K) >= s.vocab);  // the summary already IS the whole row

  if (need_topk) {
    if (p.top_k <= K) {
      limit = p.top_k;
      closed = true;
    }
    // top_k > K: the set reaches past what we can see -- `closed` stays as it was.
  }

  if (need_minp) {
    int m = 0;
    while (m < limit && w[m] >= p.min_p) ++m;
    if (m < limit) {  // the filter bit INSIDE the visible prefix, so the set is now fully visible
      limit = m;
      closed = true;
    }
  }

  // Sum of the first `n` weights, accumulated exactly as SampleCanonical's own loop does (float
  // weights into a double, canonical order, from 0.0) so the two agree bit for bit.
  const auto prefix_sum = [&](int n) {
    double c = 0.0;
    for (int j = 0; j < n; ++j) c += static_cast<double>(w[j]);
    return c;
  };

  // The full row's partition function in the same unnormalised units as w, with its uncertainty
  // band. Only consulted by the two cases below that genuinely need the mass outside the top-K.
  const double s_full = std::exp(static_cast<double>(s.lse) -
                                 static_cast<double>(s.vals[0]) *
                                     static_cast<double>(s.inv_temperature));
  const double s_lo = s_full * (1.0 - kRowSummaryLseRelTol);
  const double s_hi = s_full * (1.0 + kRowSummaryLseRelTol);

  if (need_topp) {
    if (closed) {
      const double threshold = static_cast<double>(p.top_p) * prefix_sum(limit);
      double cum = 0.0;
      int keep = limit;
      for (int j = 0; j < limit; ++j) {
        cum += static_cast<double>(w[j]);
        if (cum >= threshold) {
          keep = j + 1;
          break;
        }
      }
      limit = keep;
    } else {
      // Not closed. The pre-top_p set can then only be the WHOLE row: a top_k > K set, or a min_p
      // that did not bite, leaves a set whose own mass is unknown, and top_p's threshold is a
      // fraction of that mass.
      if (need_topk || need_minp) return r;
      const double thr_lo = static_cast<double>(p.top_p) * s_lo;
      const double thr_hi = static_cast<double>(p.top_p) * s_hi;
      double cum = 0.0;
      int keep = -1;
      for (int j = 0; j < limit; ++j) {
        cum += static_cast<double>(w[j]);
        if (cum >= thr_hi) {  // reached the nucleus for EVERY S_full in the band
          keep = j + 1;
          break;
        }
        if (cum >= thr_lo) return r;  // straddles the band: conservative, do not guess
      }
      if (keep < 0) return r;  // the nucleus extends past the top-K
      limit = keep;
      closed = true;
    }
  }

  if (!closed) {
    // Nothing narrowed the set to something visible. Either a filter is active whose own set
    // reaches past the top-K (unprovable), or there is no filter at all and the walk runs over the
    // full row with S_full as its normaliser.
    if (need_topk || need_minp || need_topp) return r;
    const double t_lo = u * s_lo;
    const double t_hi = u * s_hi;
    double cum = 0.0;
    for (int j = 0; j < limit; ++j) {
      cum += static_cast<double>(w[j]);
      // Reaching step j means every earlier cum was <= t_lo <= target, so `target >= cum_{j-1}`
      // already holds; `t_hi < cum` then pins `target < cum_j` for every S_full in the band.
      if (t_hi < cum) {
        r.resolved = true;
        r.token = s.ids[j];
        return r;
      }
      if (t_lo < cum) return r;  // straddles the band
    }
    return r;  // u lands in the tail beyond the top-K
  }

  if (limit <= 0) {  // min_p > 1 emptied the set; the full path falls back to Argmax
    r.resolved = true;
    r.token = s.ids[0];
    return r;
  }

  const double target = u * prefix_sum(limit);
  double cum = 0.0;
  for (int j = 0; j < limit; ++j) {
    cum += static_cast<double>(w[j]);
    if (target < cum) {
      r.resolved = true;
      r.token = s.ids[j];
      return r;
    }
  }
  r.resolved = true;
  r.token = s.ids[limit - 1];
  return r;
}

}  // namespace r4dx::kernels
