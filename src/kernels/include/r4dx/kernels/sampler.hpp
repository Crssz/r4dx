// r4dx::kernels sampler -- CPU sampling over fp32 logits copied back from the device
// (docs/architecture.md "argmax_sample": "greedy argmax now; top-k/top-p later" -- this
// implements the full CPU path the task calls for; a GPU sampling kernel is later work).
// vocab = 248320 floats (~1MB), well within "copy back and do it on the CPU" territory for a
// single-token decode step.
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

// Temperature scale -> optional top-k -> optional min-p -> optional top-p (nucleus) -> softmax ->
// multinomial draw, in that order (the standard vLLM/HF filter order: top-k first narrows the
// candidate set before the probability-mass-based filters see it). Deterministic for a fixed
// seed. `rng` is caller-owned so a server can keep one generator per sequence across decode
// steps.
//
// Performance: at vocab=248320 and a ~10ms/token decode budget, an unconditional O(vocab log
// vocab) sort (the previous version of this function, always run even when no rank/mass filter
// was requested) costs several ms of pure CPU sort time plus three vocab-sized allocations. This
// version only pays for what `params` actually asks for: the common pure-temperature case (no
// top_k/top_p/min_p) does one softmax pass and one multinomial walk with no sort and no index
// vector at all; top_k uses std::nth_element (O(vocab) average) instead of a full sort and only
// sorts the surviving k candidates (needed for top_p's cumulative-mass walk); min_p alone needs
// no sort either, since its threshold is order-independent (see below).
inline int32_t Sample(const float* logits, int64_t vocab, const SampleParams& params,
                       std::mt19937_64& rng) {
  if (params.temperature <= 0.0f || vocab <= 0) return Argmax(logits, vocab);

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
    // top_p's cumulative-mass walk needs descending order; std::nth_element above only
    // partitions the top-k set, it does not sort it.
    std::sort(idx.begin(), idx.end(), [&](int32_t a, int32_t b) { return scaled[a] > scaled[b]; });
  }

  // Unnormalized softmax numerators (exp(scaled - max_logit)) over the current candidate set.
  // `max_logit` above is always the GLOBAL max over the full vocab (computed before any
  // narrowing), so the global argmax's unnormalized value is exactly 1.0 regardless of which
  // subset `idx` currently holds.
  std::vector<float> probs(idx.size());
  double sum = 0.0;
  for (size_t i = 0; i < idx.size(); ++i) {
    float p = std::exp(scaled[idx[i]] - max_logit);
    probs[i] = p;
    sum += p;
  }

  if (need_minp) {
    // min_p's threshold is `min_p * (full-distribution max prob)`. Because probs[i] here is
    // exp(scaled[idx[i]] - max_logit) and max_logit is the untruncated global max, the global
    // argmax's own unnormalized prob is exactly 1.0 -- so the threshold in these unnormalized
    // units is simply `min_p`, with no need to separately track a full-distribution max or
    // renormalize first. This also makes the filter correct when combined with top_k: the
    // previous version compared against the max of the (already top-k-truncated) softmax, which
    // is stricter/looser than vLLM/HF's "relative to the untruncated distribution" semantics.
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
    // Requires idx/probs sorted descending, guaranteed above whenever need_topp is set.
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

  if (idx.empty()) return Argmax(logits, vocab);  // defensive: filters should never empty the set

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
