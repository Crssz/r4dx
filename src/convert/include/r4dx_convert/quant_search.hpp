// r4dx_convert::quant_search -- error-minimizing (optionally importance-weighted) variant of the
// w4a16 quantizer in quant_int4.hpp.
//
// THE BYTE LAYOUT IS UNCHANGED. Everything here produces exactly the same `q` / `scale` / `zero`
// triples the round-to-nearest quantizer produces, in the same containers, for the same packers
// (PackW4Nibbles, PackW4A16Scales). Only the *values chosen* differ: instead of one fixed min/max grid
// plus round-to-nearest, each (row, group) picks the (scale, zero) pair that minimizes
//
//     E(scale, zero) = sum_k  wt_k * (x_k - scale * (q_k - zero))^2,
//     q_k = clamp(round_half_away_from_zero(x_k / scale) + zero, 0, 15)
//
// over a small candidate grid modelled on llama.cpp's `make_qkx2_quants`, followed by one weighted
// least-squares refit of the float scale. `wt_k` is the Stage-1 importance matrix entry for input
// channel k (tools/reference/imatrix_capture.py: mean over calibration tokens of x_k^2) when one
// was supplied, else 1 (plain unweighted MSE).
//
// ---- Why the RTN candidate is evaluated FIRST and kept on ties ------------------------------
// The very first candidate scored is the exact round-to-nearest grid the old quantizer would have
// produced, and every later candidate must beat the incumbent *strictly* to replace it. That makes
// "search error <= RTN error, always" a structural property rather than something that happens to
// hold on the test data: it survives an all-zero importance vector, a degenerate group, and any
// float tie. tests/convert/test_quant_search.cpp gates it on random matrices.
//
// ---- Fixed evaluation order (the Python reference depends on it) ---------------------------
// tools/convert_ref/w4_ref.py reimplements this search and
// tools/convert_ref/selftest_compare.py diffs the two byte-for-byte, so every arithmetic step
// below is pinned:
//   * w4a16 works in float32 end to end -- scale, reconstruction, residual and the error
//     ACCUMULATOR (a float64 accumulator here and a float32 one in numpy would disagree on which
//     candidate wins for near-ties).
//   * the error sum runs sequentially over k = 0 .. group-1 (numpy's pairwise `np.sum` is a
//     different order, so the reference vectorizes across GROUPS and loops over k instead).
//   * `err = err + wt_k * (d * d)` is written with explicit parentheses and fp-contraction is
//     disabled below, so no `fma` folds two roundings into one on one side only.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/threadpool.hpp"

#if defined(__clang__)
#define R4DX_NO_FP_CONTRACT _Pragma("clang fp contract(off)")
#else
#define R4DX_NO_FP_CONTRACT
#endif

namespace r4dx_convert {

// 21 candidate scale multipliers spanning [0.85, 1.15] of the min/max grid, `mult(i) = 0.85f +
// 0.3f * (i / 20.0f)`. 21 (not 20) steps because i == 10 then lands on EXACTLY 1.0f in binary32
// (0.85f + 0.15f rounds to 1.0f; 0.3f * 0.5f is exact), i.e. the RTN grid is a genuine member of
// the candidate set rather than merely near one.
inline constexpr int kW4SearchSteps = 21;

inline float W4SearchScaleMult(int step) {
  return 0.85f + 0.3f * (static_cast<float>(step) / 20.0f);
}

// Importance ("imatrix") weights for one linear: `data` is nullptr for plain unweighted MSE, or a
// float32 vector of length K, one entry per INPUT channel, indexed by the absolute k within the
// row (so group g's element j uses data[g * group + j]).
struct ImportanceVector {
  const float* data = nullptr;
  int64_t size = 0;

  float at(int64_t k) const { return data ? data[k] : 1.0f; }
  bool empty() const { return data == nullptr; }
};

// ---- w4a16: asymmetric, free integer zero 0..15 ----------------------------------------------
//
// One (row, group): `wg` is the group's `group` weights, `wt` its importance weights (nullptr = 1),
// `rq` / `best_q` caller-owned scratch of `group` entries. Writes the chosen scale/zero and the
// codes (into `qg`). `refit` = false skips the final least-squares scale refit: LDLQ
// (quant_ldlq.hpp) picks the grid here but then re-rounds every column after error feedback, so a
// refit fitted to THESE codes would be fitted to codes it never stores.
//
// Degenerate groups (a group whose max == min) bypass the search entirely and take the identical
// branch quant_int4.hpp takes, so an all-zero or constant group still encodes exactly the way it
// does today -- searching a zero-width range would only pick an arbitrary member of a set of
// equally-perfect candidates.
inline void SearchInt4AsymGroup(const float* wg, const float* wt, int group, bool refit,
                                int32_t* rq, uint8_t* best_q, uint8_t* qg, float* out_scale,
                                int* out_zero) {
  R4DX_NO_FP_CONTRACT
  float wmin = wg[0], wmax = wg[0];
  for (int k = 1; k < group; ++k) {
    wmin = std::min(wmin, wg[k]);
    wmax = std::max(wmax, wg[k]);
  }
  const float range = wmax - wmin;

  // --- degenerate group: byte-identical to QuantizeInt4Asymmetric's special case ---------------
  if (range <= 0.0f) {
    float sc;
    int zp;
    bool degenerate_zero = false;
    if (wmax == 0.0f) {
      sc = 0.0f;
      zp = 0;
      degenerate_zero = true;
    } else {
      sc = std::fabs(wmax);
      zp = (wmax >= 0.0f) ? 0 : 15;
    }
    *out_scale = sc;
    *out_zero = zp;
    for (int k = 0; k < group; ++k) {
      qg[k] = static_cast<uint8_t>(
          degenerate_zero ? 0 : ClampInt(RoundHalfAwayFromZero(wg[k] / sc) + zp, 0, 15));
    }
    return;
  }

  const float s0 = std::max(range, 1e-12f) / 15.0f;

  // --- candidate 0: the RTN grid itself (incumbent; later candidates must beat it) -------------
  float best_sc = s0;
  int best_z = ClampInt(RoundHalfAwayFromZero(-wmin / s0), 0, 15);
  float best_err = 0.0f;
  for (int k = 0; k < group; ++k) {
    const int qi = ClampInt(RoundHalfAwayFromZero(wg[k] / s0) + best_z, 0, 15);
    best_q[k] = static_cast<uint8_t>(qi);
    const float d = wg[k] - s0 * (static_cast<float>(qi) - static_cast<float>(best_z));
    best_err = best_err + (wt ? wt[k] : 1.0f) * (d * d);
  }

  // --- the grid: 21 scales x 3 integer zeros around round(-min/scale) --------------------------
  for (int step = 0; step < kW4SearchSteps; ++step) {
    const float sc = s0 * W4SearchScaleMult(step);
    if (!(sc > 0.0f)) continue;
    for (int k = 0; k < group; ++k) rq[k] = RoundHalfAwayFromZero(wg[k] / sc);
    const int zc = RoundHalfAwayFromZero(-wmin / sc);
    for (int dz = -1; dz <= 1; ++dz) {
      const int z = ClampInt(zc + dz, 0, 15);
      const float zf = static_cast<float>(z);
      float err = 0.0f;
      for (int k = 0; k < group; ++k) {
        const int qi = ClampInt(rq[k] + z, 0, 15);
        const float d = wg[k] - sc * (static_cast<float>(qi) - zf);
        err = err + (wt ? wt[k] : 1.0f) * (d * d);
      }
      if (err < best_err) {
        best_err = err;
        best_sc = sc;
        best_z = z;
        for (int k = 0; k < group; ++k) best_q[k] = static_cast<uint8_t>(ClampInt(rq[k] + z, 0, 15));
      }
    }
  }

  // --- one weighted least-squares refit of the float scale, q and zero held fixed --------------
  // scale* = sum_k wt_k x_k (q_k - z) / sum_k wt_k (q_k - z)^2, then re-score with the SAME
  // codes (the stored q bytes are best_q, so the refit must not silently imply a re-rounding).
  if (refit) {
    const float zf = static_cast<float>(best_z);
    float num = 0.0f, den = 0.0f;
    for (int k = 0; k < group; ++k) {
      const float t = static_cast<float>(best_q[k]) - zf;
      const float w_k = wt ? wt[k] : 1.0f;
      num = num + (w_k * wg[k]) * t;
      den = den + w_k * (t * t);
    }
    if (den > 0.0f) {
      const float sc2 = num / den;
      if (sc2 > 0.0f) {
        float err2 = 0.0f;
        for (int k = 0; k < group; ++k) {
          const float d = wg[k] - sc2 * (static_cast<float>(best_q[k]) - zf);
          err2 = err2 + (wt ? wt[k] : 1.0f) * (d * d);
        }
        if (err2 < best_err) {
          best_err = err2;
          best_sc = sc2;
        }
      }
    }
  }

  *out_scale = best_sc;
  *out_zero = best_z;
  for (int k = 0; k < group; ++k) qg[k] = best_q[k];
}

// Same outputs/argument order as QuantizeInt4Asymmetric, plus `imp`.
inline void QuantizeInt4AsymmetricSearch(const float* w, int N, int K, int group,
                                          const ImportanceVector& imp, int nthreads,
                                          std::vector<uint8_t>& q, std::vector<float>& scale,
                                          std::vector<uint8_t>& zero) {
  RequireDivisible(K, group, "K", "QuantizeInt4AsymmetricSearch");
  const int groups_per_row = K / group;
  q.assign(static_cast<size_t>(N) * K, 0);
  scale.assign(static_cast<size_t>(N) * groups_per_row, 0.0f);
  zero.assign(static_cast<size_t>(N) * groups_per_row, 0);

  ParallelFor(0, N, nthreads, [&](int64_t r0, int64_t r1) {
    std::vector<int32_t> rq(static_cast<size_t>(group));  // round(x/sc), hoisted per candidate scale
    std::vector<uint8_t> best_q(static_cast<size_t>(group));
    for (int64_t row = r0; row < r1; ++row) {
      const float* wr = w + row * K;
      for (int g = 0; g < groups_per_row; ++g) {
        const int64_t kbase = static_cast<int64_t>(g) * group;
        const size_t gidx = static_cast<size_t>(row) * groups_per_row + g;
        int zp = 0;
        SearchInt4AsymGroup(wr + kbase, imp.empty() ? nullptr : imp.data + kbase, group,
                            /*refit=*/true, rq.data(), best_q.data(),
                            q.data() + row * static_cast<int64_t>(K) + kbase, &scale[gidx], &zp);
        zero[gidx] = static_cast<uint8_t>(zp);
      }
    }
  });
}

}  // namespace r4dx_convert
