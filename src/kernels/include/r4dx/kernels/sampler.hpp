// r4dx::kernels sampler -- CPU sampling over fp32 logits (docs/sampling.md).
//
// Two entry points share ONE definition of "the sampled token" (docs/sampling.md "Canonical
// sampling"):
//   * SampleCanonical (here) -- the FULL-VOCAB path: it sees every logit of the row.
//   * SampleFromSummary (summary_sampler.hpp) -- the SUMMARY path: it sees only the device
//     kernel's top-K + logsumexp row summary (r4dx_topk_lse_f32), and says "I cannot prove it"
//     instead of guessing whenever the answer is not determined by that summary.
// Both map the SAME uniform draw u in [0,1) to the SAME token, so a caller may use whichever is
// cheaper without changing a single emitted token.
//
// vocab = 248320 floats (~1MB) per row, which is why the summary path exists at all: the
// full-vocab path costs a 1 MB D2H plus an O(vocab) host pass per token.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

namespace r4dx::kernels {

inline int32_t Argmax(const float* logits, int64_t vocab) {
  if (vocab <= 0) return -1;  // no candidates -- caller error, not a logits[0] OOB read
  int64_t best = 0;
  float best_v = logits[0];
  for (int64_t i = 1; i < vocab; ++i) {
    if (logits[i] > best_v) {
      best_v = logits[i];
      best = i;
    }
  }
  return static_cast<int32_t>(best);
}

struct SampleParams {
  float temperature = 1.0f;  // <= 0 means greedy (Argmax), skipping the rest of the pipeline
  int top_k = 0;              // 0 = disabled
  float top_p = 1.0f;         // 1.0 = disabled (nucleus sampling)
  float min_p = 0.0f;         // 0.0 = disabled (drop tokens below min_p * max_prob)
  uint64_t seed = 0;
};

namespace detail {

// The CANONICAL order (docs/sampling.md): RAW logit DESCENDING, ties broken toward the LOWER token
// id. Deliberately keyed on the RAW logit rather than on logits[i]/temperature: dividing by a
// positive temperature is monotone but not injective in fp32, so two distinct raw logits can
// collapse to the same scaled float. Ordering on the raw value keeps this a strict TOTAL order
// that the device summary kernel -- r4dx_topk_lse_f32, which returns raw values and never applies
// the temperature -- produces identically, which is what makes the two sampling paths agree token
// for token.
struct CanonicalOrder {
  const float* logits;
  bool operator()(int32_t a, int32_t b) const {
    return logits[a] > logits[b] || (logits[a] == logits[b] && a < b);
  }
};

// Fills `out` with the canonical-order prefix of length min(m, vocab). O(vocab) selection plus an
// O(m log m) sort -- the full-vocab paths below start at m = 64 and grow by 8x until the CDF walk
// resolves, so a peaked row never pays for sorting 248320 elements.
inline void CanonicalPrefix(const float* logits, int64_t vocab, int64_t m,
                             std::vector<int32_t>& out) {
  const CanonicalOrder order{logits};
  if (m > vocab) m = vocab;
  out.resize(static_cast<size_t>(vocab));
  std::iota(out.begin(), out.end(), 0);
  if (m < vocab) {
    std::nth_element(out.begin(), out.begin() + static_cast<ptrdiff_t>(m), out.end(), order);
    out.resize(static_cast<size_t>(m));
  }
  std::sort(out.begin(), out.end(), order);
}

}  // namespace detail

// One uniform draw u in [0,1) per emitted token: the top 53 bits of one mt19937_64 output, which
// is exactly one rng() call (std::generate_canonical<double,53> is NOT specified to make exactly
// one call, and libstdc++/MSVC disagree on it -- this is spelled out so a given seed produces the
// same token stream on any standard library).
inline double DrawUniform01(std::mt19937_64& rng) {
  return static_cast<double>(rng() >> 11) * (1.0 / 9007199254740992.0);  // 2^-53
}

// ---- the canonical full-vocab sampler ---------------------------------------------------------
// token = InvCDF(u) over the post-filter distribution p, walked in CANONICAL ORDER.
//
// Filters, in the standard vLLM/HF order and with the exact semantics the pre-M6 Sample() had
// (SampleLegacy below is that function, kept verbatim as the drift reference):
//   temperature -> top_k (rank) -> min_p (mass, relative to the UNTRUNCATED distribution's max)
//   -> top_p (nucleus over the canonical-order prefix) -> renormalise -> InvCDF(u).
// `u` must be in [0,1); a caller with an rng uses Sample() below.
//
// Ties: the pre-M6 Sample() left equal-logit ordering to std::sort/std::nth_element, i.e.
// unspecified. This one is deterministic (lower id first). The post-filter DISTRIBUTION is
// unchanged -- only the map from a draw to a token differs, which is what
// tests/kernels/test_sampler_canonical.cpp pins statistically against SampleLegacy.
//
// `out_candidates` (optional) receives the SET of tokens with non-zero post-filter probability;
// order is unspecified (it is a set, and materialising the pure-temperature case in canonical
// order would mean sorting the whole vocabulary for nothing).
inline int32_t SampleCanonical(const float* logits, int64_t vocab, const SampleParams& params,
                                double u, std::vector<int32_t>* out_candidates = nullptr) {
  if (vocab <= 0) return -1;
  if (params.temperature <= 0.0f) {
    const int32_t g = Argmax(logits, vocab);
    if (out_candidates) out_candidates->assign(1, g);
    return g;
  }

  const float temperature = params.temperature;
  const detail::CanonicalOrder order{logits};

  // The global max of the SCALED logits. Division by a positive temperature is monotone
  // non-decreasing, so this is exactly (max raw logit)/temperature -- no second pass needed.
  float max_raw = logits[0];
  for (int64_t i = 1; i < vocab; ++i) {
    if (logits[i] > max_raw) max_raw = logits[i];
  }
  const float max_logit = max_raw / temperature;
  // Unnormalised softmax numerator. Because max_logit is the UNTRUNCATED global max, the global
  // argmax's own weight is exactly 1.0 whatever subset is in play -- which is what makes min_p's
  // threshold plainly `min_p` in these units (same reasoning as the pre-M6 implementation).
  const auto weight = [&](int64_t id) {
    return std::exp(logits[id] / temperature - max_logit);
  };

  const bool need_topk = params.top_k > 0 && params.top_k < static_cast<int>(vocab);
  const bool need_topp = params.top_p < 1.0f;
  const bool need_minp = params.min_p > 0.0f;

  std::vector<int32_t> cand;
  bool complete = false;  // `cand` already holds the entire pre-top_p candidate set

  if (need_topk) {
    std::vector<int32_t> idx(static_cast<size_t>(vocab));
    std::iota(idx.begin(), idx.end(), 0);
    std::nth_element(idx.begin(), idx.begin() + params.top_k, idx.end(), order);
    idx.resize(static_cast<size_t>(params.top_k));
    std::sort(idx.begin(), idx.end(), order);
    cand.swap(idx);
    if (need_minp) {
      // min_p runs AFTER top_k. `cand` is in canonical order and weight() is non-increasing along
      // it, so the surviving set is a prefix.
      size_t keep = 0;
      while (keep < cand.size() && weight(cand[keep]) >= params.min_p) ++keep;
      cand.resize(keep);
    }
    complete = true;
  } else if (need_minp) {
    for (int64_t i = 0; i < vocab; ++i) {
      if (weight(i) >= params.min_p) cand.push_back(static_cast<int32_t>(i));
    }
    std::sort(cand.begin(), cand.end(), order);
    complete = true;
  }

  if (!complete) {
    // No rank/mass filter has narrowed anything: the pre-top_p candidate set is the whole row, so
    // its partition function is needed either as top_p's nucleus threshold or as the multinomial
    // walk's own normaliser. Accumulated in id order in double (O(vocab), no sort).
    double s_full = 0.0;
    for (int64_t i = 0; i < vocab; ++i) s_full += static_cast<double>(weight(i));
    const double threshold = need_topp ? static_cast<double>(params.top_p) * s_full : u * s_full;

    std::vector<int32_t> prefix;
    ptrdiff_t found = -1;
    for (int64_t m = 64;; m = (m * 8 < vocab) ? m * 8 : vocab) {
      detail::CanonicalPrefix(logits, vocab, m, prefix);
      double cum = 0.0;
      for (size_t j = 0; j < prefix.size(); ++j) {
        cum += static_cast<double>(weight(prefix[j]));
        // top_p: the shortest prefix whose mass REACHES the nucleus threshold (>=, as the pre-M6
        // implementation had it). Plain walk: the first step the target falls strictly short of.
        if (need_topp ? (cum >= threshold) : (threshold < cum)) {
          found = static_cast<ptrdiff_t>(j);
          break;
        }
      }
      if (found >= 0 || m >= vocab) break;
    }
    if (found < 0) found = static_cast<ptrdiff_t>(prefix.size()) - 1;  // fp rounding only

    if (!need_topp) {
      if (out_candidates) {
        out_candidates->resize(static_cast<size_t>(vocab));
        std::iota(out_candidates->begin(), out_candidates->end(), 0);
      }
      return prefix[static_cast<size_t>(found)];
    }
    cand.assign(prefix.begin(), prefix.begin() + found + 1);
  } else if (need_topp && !cand.empty()) {
    double sum = 0.0;
    for (int32_t id : cand) sum += static_cast<double>(weight(id));
    const double threshold = static_cast<double>(params.top_p) * sum;
    double cum = 0.0;
    size_t keep = cand.size();
    for (size_t j = 0; j < cand.size(); ++j) {
      cum += static_cast<double>(weight(cand[j]));
      if (cum >= threshold) {
        keep = j + 1;
        break;
      }
    }
    cand.resize(keep);
  }

  if (cand.empty()) {  // defensive: only reachable via min_p > 1, same as the pre-M6 path
    const int32_t g = Argmax(logits, vocab);
    if (out_candidates) out_candidates->assign(1, g);
    return g;
  }
  if (out_candidates) *out_candidates = cand;

  double sum = 0.0;
  for (int32_t id : cand) sum += static_cast<double>(weight(id));
  const double target = u * sum;
  double cum = 0.0;
  for (int32_t id : cand) {
    cum += static_cast<double>(weight(id));
    if (target < cum) return id;
  }
  return cand.back();
}

// Thin rng-taking wrapper: exactly ONE draw per emitted token, and no draw at all on the greedy
// path (temperature <= 0), so a greedy request's generator state is untouched -- the pre-M6
// behaviour every existing call site (src/cli/main.cpp, src/server/engine.cpp) relies on.
inline int32_t Sample(const float* logits, int64_t vocab, const SampleParams& params,
                       std::mt19937_64& rng) {
  if (params.temperature <= 0.0f || vocab <= 0) return Argmax(logits, vocab);
  return SampleCanonical(logits, vocab, params, DrawUniform01(rng));
}

// ---- the pre-M6 sampler, kept verbatim --------------------------------------------------------
// This is exactly the function Sample() was before Milestone 6, preserved so
// tests/kernels/test_sampler_canonical.cpp can check BOTH it and SampleCanonical against the same
// fp64 exact post-filter distribution -- i.e. so any semantic drift introduced by the canonical
// rewrite is caught rather than assumed away. Not called by any production path.
//
// Its filter semantics are the specification SampleCanonical implements; the only intended
// differences are (a) equal-logit ordering, unspecified here and lower-id-first there, and (b) the
// walk order in the no-filter fast path, id order here and canonical order there. Neither changes
// the distribution.
//
// `out_candidates` (optional, added for that test) receives the set of tokens with non-zero
// post-filter probability; it does not affect the returned token.
inline int32_t SampleLegacy(const float* logits, int64_t vocab, const SampleParams& params,
                             std::mt19937_64& rng, std::vector<int32_t>* out_candidates = nullptr) {
  if (params.temperature <= 0.0f || vocab <= 0) {
    if (out_candidates) out_candidates->assign(1, Argmax(logits, vocab));
    return Argmax(logits, vocab);
  }

  std::vector<float> scaled(static_cast<size_t>(vocab));
  float max_logit = -std::numeric_limits<float>::infinity();
  for (int64_t i = 0; i < vocab; ++i) {
    scaled[i] = logits[i] / params.temperature;
    max_logit = std::max(max_logit, scaled[i]);
  }

  const bool need_topk = params.top_k > 0 && params.top_k < static_cast<int>(vocab);
  const bool need_topp = params.top_p < 1.0f;
  const bool need_minp = params.min_p > 0.0f;

  std::uniform_real_distribution<float> dist(0.0f, 1.0f);

  if (!need_topk && !need_topp && !need_minp) {
    // Fast path: no candidate-set narrowing requested at all.
    double sum = 0.0;
    std::vector<float> probs(static_cast<size_t>(vocab));
    for (int64_t i = 0; i < vocab; ++i) {
      float p = std::exp(scaled[i] - max_logit);
      probs[i] = p;
      sum += p;
    }
    if (out_candidates) {
      out_candidates->resize(static_cast<size_t>(vocab));
      std::iota(out_candidates->begin(), out_candidates->end(), 0);
    }
    float r = dist(rng) * static_cast<float>(sum);
    double cum = 0.0;
    for (int64_t i = 0; i < vocab; ++i) {
      cum += probs[i];
      if (r <= cum) return static_cast<int32_t>(i);
    }
    return static_cast<int32_t>(vocab - 1);
  }

  std::vector<int32_t> idx(static_cast<size_t>(vocab));
  std::iota(idx.begin(), idx.end(), 0);

  if (need_topk) {
    std::nth_element(idx.begin(), idx.begin() + params.top_k, idx.end(),
                      [&](int32_t a, int32_t b) { return scaled[a] > scaled[b]; });
    idx.resize(params.top_k);
  }
  if (need_topk || need_topp) {
    std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return scaled[a] > scaled[b]; });
  }

  std::vector<float> probs(idx.size());
  double sum = 0.0;
  for (size_t i = 0; i < idx.size(); ++i) {
    float p = std::exp(scaled[idx[i]] - max_logit);
    probs[i] = p;
    sum += p;
  }

  if (need_minp) {
    size_t keep = 0;
    for (size_t i = 0; i < idx.size(); ++i) {
      if (probs[i] >= params.min_p) {
        idx[keep] = idx[i];
        probs[keep] = probs[i];
        ++keep;
      }
    }
    idx.resize(keep);
    probs.resize(keep);
    sum = std::accumulate(probs.begin(), probs.end(), 0.0);
  }

  if (need_topp) {
    double cum = 0.0;
    size_t keep = probs.size();
    for (size_t i = 0; i < probs.size(); ++i) {
      cum += probs[i];
      if (cum >= static_cast<double>(params.top_p) * sum) {
        keep = i + 1;
        break;
      }
    }
    idx.resize(keep);
    probs.resize(keep);
    sum = std::accumulate(probs.begin(), probs.end(), 0.0);
  }

  if (idx.empty()) {  // defensive: filters should never empty the set
    if (out_candidates) out_candidates->assign(1, Argmax(logits, vocab));
    return Argmax(logits, vocab);
  }
  if (out_candidates) *out_candidates = idx;

  float r = dist(rng) * static_cast<float>(sum);
  double cum = 0.0;
  for (size_t i = 0; i < probs.size(); ++i) {
    cum += probs[i];
    if (r <= cum) return idx[i];
  }
  return idx.back();
}

inline std::mt19937_64 MakeRng(uint64_t seed) { return std::mt19937_64(seed); }

}  // namespace r4dx::kernels
