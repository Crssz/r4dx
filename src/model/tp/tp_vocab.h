// r4dx::model::tp -- merges of per-rank results over a VOCAB-SPLIT lm_head (docs/tp.md 7.2-7.4,
// 8.2). Under TP=2 rank r holds lm_head rows [r * V/2, (r+1) * V/2), so every per-row result the
// device computes (argmax, row summary, top-16) covers one shard; the host all-gathers the per-rank
// results and these functions turn them into exactly the full-row result.
//
// Every merge is EXACT under the total order the device kernels already use -- "(v, i) beats
// (v', i') iff v > v' || (v == v' && i < i')" (r4dx_argmax_f32's lowest-index tie-break,
// r4dx_topk16_f32, r4dx_topk_lse_f32 / detail::CanonicalOrder): every member of the global top-k
// is in the top-k of its own shard, and rank r's ids are all below rank r+1's, so a merge in rank
// order reproduces the full row's result bit for bit. The one approximate quantity, a row summary's
// logsumexp, is combined in double (LogAddExp) and stays inside the kRowSummaryLseRelTol band the
// summary sampler already carries (docs/tp.md 7.4).
//
// Header-only and HIP-free (tests/model/test_tp_vocab_merge.cpp drives it on the CPU). Every input
// id is already GLOBAL: the caller adds rank * VocabShardSize() to the device's local ids first.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

#include "r4dx/kernels/summary_sampler.hpp"

namespace r4dx::model::tp {

struct ArgmaxPair {
  int32_t idx;  // GLOBAL id of the rank's lowest-index maximum
  float val;
};

// The full row's argmax with the lowest-index tie-break: rank r wins only with a STRICTLY larger
// value than every lower rank (whose ids are all smaller), so a tie resolves to the lower global
// id, exactly as r4dx_argmax_f32 / kernels::Argmax over the full row. A row that is all -inf
// returns rank 0's id (its index 0). NaN is undefined, as it is for the full-row kernels.
inline int32_t MergeArgmax(const ArgmaxPair* per_rank, int world) {
  int best = 0;
  for (int r = 1; r < world; ++r) {
    if (per_rank[r].val > per_rank[best].val) best = r;
  }
  return per_rank[best].idx;
}

// log(exp(a) + exp(b)) in double, stable, with -inf as the identity. Callers pass the per-rank
// values in rank order so every rank computes the same bits.
inline double LogAddExp(double a, double b) {
  const double m = std::max(a, b);
  if (m == -std::numeric_limits<double>::infinity()) return m;
  return m + std::log(std::exp(a - m) + std::exp(b - m));
}

namespace detail {
inline bool Beats(float va, int32_t ia, float vb, int32_t ib) {
  return va > vb || (va == vb && ia < ib);
}
}  // namespace detail

// The full row's summary from the per-rank summaries (each taken over its own shard, ids already
// global): ids/vals = the first kRowSummaryMaxK of the rank-order merge of the sorted per-rank
// lists under the canonical order; lse = float(fold of LogAddExp over the ranks, in rank order);
// vocab = global_vocab; inv_temperature unchanged (every rank must have used the same one --
// std::invalid_argument otherwise).
inline kernels::RowSummary MergeRowSummaries(const kernels::RowSummary* per_rank, int world,
                                             int64_t global_vocab) {
  constexpr int kMaxWorld = 8;
  if (world < 1 || world > kMaxWorld) {
    throw std::invalid_argument("tp::MergeRowSummaries: world " + std::to_string(world) +
                                " outside [1, 8]");
  }
  int total = 0;
  for (int r = 0; r < world; ++r) {
    if (per_rank[r].k < 0 || per_rank[r].k > kernels::kRowSummaryMaxK) {
      throw std::invalid_argument("tp::MergeRowSummaries: rank " + std::to_string(r) + " has k " +
                                  std::to_string(per_rank[r].k));
    }
    if (per_rank[r].inv_temperature != per_rank[0].inv_temperature) {
      throw std::invalid_argument(
          "tp::MergeRowSummaries: ranks summarized at different inv_temperature");
    }
    total += per_rank[r].k;
  }

  kernels::RowSummary out;
  out.vocab = global_vocab;
  out.inv_temperature = per_rank[0].inv_temperature;
  out.k = std::min(total, kernels::kRowSummaryMaxK);
  int pos[kMaxWorld] = {};
  for (int j = 0; j < out.k; ++j) {
    int best = -1;
    for (int r = 0; r < world; ++r) {
      if (pos[r] >= per_rank[r].k) continue;
      if (best < 0 || detail::Beats(per_rank[r].vals[pos[r]], per_rank[r].ids[pos[r]],
                                    per_rank[best].vals[pos[best]],
                                    per_rank[best].ids[pos[best]])) {
        best = r;
      }
    }
    out.ids[j] = per_rank[best].ids[pos[best]];
    out.vals[j] = per_rank[best].vals[pos[best]];
    ++pos[best];
  }

  double lse = static_cast<double>(per_rank[0].lse);
  for (int r = 1; r < world; ++r) lse = LogAddExp(lse, static_cast<double>(per_rank[r].lse));
  out.lse = static_cast<float>(lse);
  return out;
}

// Per-row top-16 of the full row from the per-rank top-16s (docs/tp.md 8.2, H7). Inputs are laid
// out [world][rows][16], each rank's row sorted descending under the total order above (exactly
// r4dx_topk16_f32's output, ids made global); outputs are [rows][16].
inline void MergeTop16(const int32_t* ids, const float* vals, int world, int rows,
                       int32_t* out_ids, float* out_vals) {
  constexpr int kTop = 16;
  constexpr int kMaxWorld = 8;
  if (world < 1 || world > kMaxWorld) {
    throw std::invalid_argument("tp::MergeTop16: world " + std::to_string(world) +
                                " outside [1, 8]");
  }
  const size_t rank_stride = static_cast<size_t>(rows) * kTop;
  for (int row = 0; row < rows; ++row) {
    int pos[kMaxWorld] = {};
    for (int j = 0; j < kTop; ++j) {
      int best = -1;
      size_t best_at = 0;
      for (int r = 0; r < world; ++r) {
        if (pos[r] >= kTop) continue;
        const size_t at = r * rank_stride + static_cast<size_t>(row) * kTop + pos[r];
        if (best < 0 || detail::Beats(vals[at], ids[at], vals[best_at], ids[best_at])) {
          best = r;
          best_at = at;
        }
      }
      out_ids[static_cast<size_t>(row) * kTop + j] = ids[best_at];
      out_vals[static_cast<size_t>(row) * kTop + j] = vals[best_at];
      ++pos[best];
    }
  }
}

}  // namespace r4dx::model::tp
