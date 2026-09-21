// tests/model/sampled_equality.hpp -- shared pieces of the Milestone 6 stage S2 "a sampled
// speculative round emits exactly what plain sampled decode emits" tests (docs/sampling.md section
// 9), used by tests/model/test_mtp.cpp and tests/model/test_dflash_e2e.cpp.
//
// The contract under test, restated: every sampling path consumes exactly ONE uniform draw per
// EMITTED token, and maps that draw to a token by the same canonical rule (docs/sampling.md section
// 2). So for a fixed seed, `DecodeStepSampled` called N times and `DecodeStepMtpSampled` /
// `DecodeStepDflashSampled` rounds totalling N tokens must produce the SAME N tokens, in the same
// order -- not merely the same distribution. That is this milestone's losslessness gate, and it is
// a far stronger handle than "the acceptance rate looks plausible".
//
// Header-only, HIP-free apart from what model.h itself pulls in; everything below is host
// arithmetic over logits a caller already has.
#pragma once

#include <cinttypes>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <random>
#include <string>
#include <vector>

#include "model.h"
#include "r4dx/kernels/summary_sampler.hpp"

namespace r4dx_test {

// The three sampling configurations the stage's own equality matrix names: a typical chat request,
// pure temperature (no candidate-set filter at all -- the case whose CDF walk depends on the mass
// OUTSIDE the summary's top-64, so it is the one that actually exercises the fallback), and a
// min_p request.
struct SampledConfig {
  const char* name;
  r4dx::kernels::SampleParams params;
};

inline std::vector<SampledConfig> SampledConfigs() {
  std::vector<SampledConfig> out;
  {
    r4dx::kernels::SampleParams p;
    p.temperature = 0.7f;
    p.top_k = 20;
    p.top_p = 0.8f;
    out.push_back({"T0.7-topk20-topp0.8", p});
  }
  {
    r4dx::kernels::SampleParams p;
    p.temperature = 1.0f;
    out.push_back({"T1.0-pure", p});
  }
  {
    r4dx::kernels::SampleParams p;
    p.temperature = 0.8f;
    p.min_p = 0.05f;
    out.push_back({"T0.8-minp0.05", p});
  }
  return out;
}

// The first token of a sampled run: Prefill() hands back the full fp32 logits row, so the canonical
// full-vocab sampler produces it directly -- with ONE draw, exactly like every token after it, so
// the two runs being compared stay in lockstep on their generators from the very first token.
inline int32_t SampleFirstToken(const std::vector<float>& prefill_logits, int64_t vocab,
                                 const r4dx::kernels::SampleParams& params, std::mt19937_64& rng) {
  return r4dx::kernels::SampleCanonical(prefill_logits.data(), vocab, params,
                                        r4dx::kernels::DrawUniform01(rng));
}

// The draws a run with this seed consumes, in order -- draw i is the one that produced emitted
// token i (the one-draw-per-emitted-token invariant). Only used to report WHICH u a divergence
// straddled; never used to drive a run.
inline std::vector<double> ReplayDraws(uint64_t seed, size_t n) {
  std::mt19937_64 rng = r4dx::kernels::MakeRng(seed);
  std::vector<double> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = r4dx::kernels::DrawUniform01(rng);
  return out;
}

// ---- divergence forensics ----------------------------------------------------------------------
// Only ever run on a MISMATCH. The question it answers is the one docs/perf.md's "known batched-
// verify divergence" class and tools/validate_dflash.ps1 exist for: a speculative round verifies
// several rows in ONE GEMM whose tiling/reduction order differs from single-row decode's, so two
// mathematically-equal logits rows can differ in their last fp32 bits. That can only change an
// EMITTED token when the draw `u` lands within that noise of a CDF boundary -- i.e. when two
// canonical-order-adjacent candidates are near-tied AND u sits essentially exactly between them.
// Anything else (tokens far apart in the CDF, a comfortable margin, a non-adjacent pair) is a
// bookkeeping bug, and the test must fail.
struct NearTieReport {
  bool both_in_candidate_set = false;
  bool adjacent = false;      // adjacent in canonical order within the post-filter candidate set
  int32_t rank_a = -1, rank_b = -1;
  double p_a = 0.0, p_b = 0.0;        // post-filter probabilities
  double boundary = 0.0;              // the CDF value separating the two, in [0,1]
  double margin = 0.0;                // |u - boundary|, in probability units
  double margin_rel = 0.0;            // margin / min(p_a, p_b)
  double logit_gap = 0.0;             // |raw logit difference|
};

// `logits` is the full fp32 row a PLAIN single-row decode computed for the diverging position.
inline NearTieReport AnalyzeNearTie(const std::vector<float>& logits, int64_t vocab,
                                     const r4dx::kernels::SampleParams& params, double u,
                                     int32_t token_a, int32_t token_b) {
  NearTieReport r;
  std::vector<int32_t> cand;
  r4dx::kernels::SampleCanonical(logits.data(), vocab, params, u, &cand);
  // SampleCanonical documents `cand` as a SET (unordered for the pure-temperature case) -- put it
  // back into canonical order here so the CDF walk below is the real one.
  std::sort(cand.begin(), cand.end(), [&](int32_t x, int32_t y) {
    return logits[static_cast<size_t>(x)] > logits[static_cast<size_t>(y)] ||
           (logits[static_cast<size_t>(x)] == logits[static_cast<size_t>(y)] && x < y);
  });

  const float temperature = params.temperature;
  float max_raw = logits[0];
  for (int64_t i = 1; i < vocab; ++i) max_raw = std::max(max_raw, logits[static_cast<size_t>(i)]);
  const float max_scaled = max_raw / temperature;
  const auto weight = [&](int32_t id) {
    return std::exp(logits[static_cast<size_t>(id)] / temperature - max_scaled);
  };

  double sum = 0.0;
  for (int32_t id : cand) sum += static_cast<double>(weight(id));
  ptrdiff_t ia = -1, ib = -1;
  for (size_t i = 0; i < cand.size(); ++i) {
    if (cand[i] == token_a) ia = static_cast<ptrdiff_t>(i);
    if (cand[i] == token_b) ib = static_cast<ptrdiff_t>(i);
  }
  r.both_in_candidate_set = (ia >= 0 && ib >= 0);
  r.rank_a = static_cast<int32_t>(ia);
  r.rank_b = static_cast<int32_t>(ib);
  r.logit_gap = std::fabs(static_cast<double>(logits[static_cast<size_t>(token_a)]) -
                          static_cast<double>(logits[static_cast<size_t>(token_b)]));
  if (!r.both_in_candidate_set || sum <= 0.0) return r;

  r.p_a = static_cast<double>(weight(token_a)) / sum;
  r.p_b = static_cast<double>(weight(token_b)) / sum;
  r.adjacent = (std::abs(ia - ib) == 1);
  // The boundary u would have to cross to swap these two: the cumulative mass up to and including
  // the EARLIER of the two in canonical order.
  const ptrdiff_t first = std::min(ia, ib);
  double cum = 0.0;
  for (ptrdiff_t i = 0; i <= first; ++i) cum += static_cast<double>(weight(cand[static_cast<size_t>(i)]));
  r.boundary = cum / sum;
  r.margin = std::fabs(u - r.boundary);
  const double p_min = std::min(r.p_a, r.p_b);
  r.margin_rel = p_min > 0.0 ? r.margin / p_min : 0.0;
  return r;
}

inline void PrintNearTie(const char* tag, const NearTieReport& r, double u, int32_t token_a,
                          int32_t token_b) {
  std::fprintf(stderr,
               "%s near-tie analysis: plain=%d (rank %d, p=%.6e) spec=%d (rank %d, p=%.6e) "
               "adjacent=%s u=%.12f boundary=%.12f margin=%.3e margin_rel=%.3e logit_gap=%.6e\n",
               tag, token_a, r.rank_a, r.p_a, token_b, r.rank_b, r.p_b,
               r.adjacent ? "yes" : "no", u, r.boundary, r.margin, r.margin_rel, r.logit_gap);
  // Descriptive only -- the verdict belongs to the caller's own classifier, which decides on the
  // two EXACT reproductions (does each row, sampled canonically at this very u, give back the token
  // its own run emitted?). These notes just say how far apart the two tokens sit in the row that
  // was measured.
  if (!r.both_in_candidate_set) {
    std::fprintf(stderr,
                 "%s   -> at least one of the two tokens is outside the post-filter candidate set "
                 "of THIS row, so the two rows disagree about the candidate set itself, not merely "
                 "about a boundary inside it\n",
                 tag);
  } else if (!r.adjacent) {
    std::fprintf(stderr,
                 "%s   -> the two tokens are %d ranks apart in THIS row's canonical order, so the "
                 "other row must also have permuted the candidates between them (expected where "
                 "several candidates sit within the rows' own difference of each other, e.g. in the "
                 "flat tail of a pure-temperature distribution)\n",
                 tag, std::abs(r.rank_a - r.rank_b));
  }
}

// Reports a mismatch between two token sequences, with whatever forensic detail the caller could
// gather. `plain_row_at_divergence` may be empty when the caller could not recompute it.
inline void ReportSequenceMismatch(const char* tag, const std::vector<int32_t>& plain,
                                    const std::vector<int32_t>& spec, uint64_t seed,
                                    const r4dx::kernels::SampleParams& params, int64_t vocab,
                                    const std::vector<float>& plain_row_at_divergence) {
  size_t j = 0;
  while (j < plain.size() && j < spec.size() && plain[j] == spec[j]) ++j;
  std::fprintf(stderr,
               "FAIL %s: sampled speculative sequence diverges from plain sampled decode at index "
               "%zu (plain=%d spec=%d); %zu of %zu tokens matched before it\n",
               tag, j, j < plain.size() ? plain[j] : -1, j < spec.size() ? spec[j] : -1, j,
               std::min(plain.size(), spec.size()));
  if (j >= plain.size() || j >= spec.size()) return;
  const std::vector<double> draws = ReplayDraws(seed, j + 1);
  const double u = draws[j];
  std::fprintf(stderr, "%s   the draw that token consumed: u=%.17g\n", tag, u);
  if (!plain_row_at_divergence.empty()) {
    const NearTieReport r =
        AnalyzeNearTie(plain_row_at_divergence, vocab, params, u, plain[j], spec[j]);
    PrintNearTie(tag, r, u, plain[j], spec[j]);
  } else {
    std::fprintf(stderr,
                 "%s   (no full logits row for that position was recoverable, so no near-tie "
                 "analysis -- rerun the plain trajectory capturing DecodeStep's logits at that "
                 "index to classify this)\n",
                 tag);
  }
}

// ---- an oracle-drafted SAMPLED round, driven through the public API ----------------------------
// Deliberately an INDEPENDENT re-implementation of Model::DecodeStepMtpSampled's own walk, built
// only out of public methods (VerifyWindow with row summaries + CommitVerifiedWindow), for two
// jobs: (a) a drafter for containers whose real MTP head never accepts anything, so the
// all-drafts-accepted path and the mid-round-stop scenario can be exercised at all, and (b) a
// second opinion on the walk itself -- if Model's own loop drifted from this one, the oracle
// variants of the equality checks would stop matching.
//
// It also cross-checks, for every row it resolves, that SampleFromSummary's answer equals the
// full-vocab canonical sampler's answer on that row's own real logits -- which is stage S1's
// exactness claim re-tested on REAL model rows rather than synthetic ones. `fallbacks_out` counts
// rows the summary could not prove.
struct OracleRoundStats {
  int64_t rows = 0;
  int64_t fallbacks = 0;
  bool summary_mismatch = false;  // a resolved summary disagreed with the full-vocab sampler
};

inline std::vector<int32_t> RunSampledOracleRound(r4dx::model::Model& m, int32_t anchor,
                                                   const std::vector<int32_t>& drafts,
                                                   const r4dx::kernels::SampleParams& params,
                                                   std::mt19937_64& rng, OracleRoundStats* stats,
                                                   std::vector<float>* window_logits_out = nullptr) {
  const int64_t vocab = m.Config().vocab_size;
  const float inv_t = 1.0f / params.temperature;
  std::vector<int32_t> candidates{anchor};
  candidates.insert(candidates.end(), drafts.begin(), drafts.end());

  std::vector<float> logits;
  std::vector<r4dx::kernels::RowSummary> summaries;
  m.VerifyWindow(candidates, &logits, &summaries, inv_t);
  if (window_logits_out) *window_logits_out = logits;  // forensics: the EXACT rows this round used

  std::vector<int32_t> round;
  const int64_t mm = static_cast<int64_t>(drafts.size());
  for (int64_t i = 0;; ++i) {
    const double u = r4dx::kernels::DrawUniform01(rng);
    const r4dx::kernels::SummarySampleResult sr =
        r4dx::kernels::SampleFromSummary(summaries[static_cast<size_t>(i)], params, u);
    const int32_t full = r4dx::kernels::SampleCanonical(logits.data() + i * vocab, vocab, params, u);
    if (stats) {
      ++stats->rows;
      if (!sr.resolved) ++stats->fallbacks;
      if (sr.resolved && sr.token != full) {
        stats->summary_mismatch = true;
        std::fprintf(stderr,
                     "FAIL: SampleFromSummary resolved row %lld to %d but the full-vocab canonical "
                     "sampler on that row's own logits says %d (u=%.17g)\n",
                     static_cast<long long>(i), sr.token, full, u);
      }
    }
    const int32_t tok = sr.resolved ? sr.token : full;
    round.push_back(tok);
    if (i < mm && tok == drafts[static_cast<size_t>(i)]) continue;
    m.CommitVerifiedWindow(i + 1);
    break;
  }
  return round;
}

}  // namespace r4dx_test
