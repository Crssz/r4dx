// r4dx_convert::quant_ldlq -- LDLQ / GPTQ error-feedback rounding for the w4a16 / w4a8 / mxfp4
// layouts (docs/quant2.md section 2.2, "Q1").
//
// THE BYTE LAYOUT IS UNCHANGED. Like quant_search.hpp, everything here produces exactly the `q` /
// `scale` / `zero` (and mxfp4 `packed` / `escale` / `wref`) vectors QuantizeInt4Asymmetric /
// QuantizeInt4SymmetricPinned8 / QuantizeMxfp4 produce, for the same packers. Only the values
// chosen differ: instead of rounding each weight to its own nearest grid point, the rounding error
// of column i is pushed onto the not-yet-quantized columns j > i in the direction that the layer's
// input statistics say cancels it, minimizing the proxy loss
//
//     L(Wq) = tr((W - Wq) H (W - Wq)^T),    H = E_tokens[x x^T]   (tools/reference/hessian_capture.py)
//
// i.e. the expected squared error of the layer OUTPUT, not of the weights.
//
// ---- The algorithm (GPTQ's lazy-batch form; LDLQ is the same update written with U) -----------
// With U upper triangular and H^-1 = U^T U (the upper Cholesky factor of H^-1), process columns in
// order; for every row, after rounding column i to wq_i:
//
//     err_i   = (w_i - wq_i) / U[i][i]
//     w_j    -= err_i * U[i][j]              for every j > i
//
// This is exactly optimal-brain-surgeon for "fix w_i, re-optimize w_{>i} under L" (Frantar et al.,
// GPTQ, 2022; Chee et al., QuIP's LDLQ, 2023, prove the two are the same rule). Applied naively the
// update is a rank-1 write to the whole rest of the row per column; the lazy form keeps it inside a
// 128-column block and applies the block's accumulated errors E (R x 128) to everything right of the
// block as one GEMM, W[:, b1:] -= E . U[b0:b1, b1:] (linalg::SubMatMulSerial, the AVX-512
// microkernel), which is where all the flops are (N * K^2 / 2 multiply-adds per linear).
//
// Rows are independent (U is shared, read-only), so the matrix is cut into tiles of R = 128 rows,
// each quantized start to finish by one thread on its own mutable copy of those rows.
//
// ---- U = chol(H^-1) upper, without ever forming H^-1 (the reversal trick) ---------------------
// Let J be the K x K reversal permutation (J = J^T = J^-1). Factor the reversed matrix
// J H J = L L^T (ordinary lower Cholesky). Then
//
//     H = J L L^T J = (J L J)(J L J)^T = R R^T,    R = J L J  upper triangular (J flips both axes),
//     H^-1 = R^-T R^-1 = (R^-1)^T (R^-1),
//
// and R^-1 = J L^-1 J is upper triangular with a positive diagonal -- so it IS the unique upper
// Cholesky factor U of H^-1. Cost: one Cholesky + one triangular inverse (each ~K^3/3), against
// Cholesky + full inverse + Cholesky for the textbook `cholesky(inverse(H))`, and never an
// explicitly inverted SPD matrix (whose rounding error the second Cholesky would then amplify).
// On a row-major K x K array J M J is just the FLAT reversal of the K*K elements (element (i, j)
// at flat p = i*K + j lands at K*K-1-p = (K-1-i)*K + (K-1-j)), so both flips are a std::reverse.
//
// ---- Damping and retry ---------------------------------------------------------------------------
// H = X^T X / rows is only positive SEMI-definite (dead input channels, fewer tokens than K, fp32
// rounding of a near-singular matrix), so FactorHessian factors H + damp * mean(diag H) * I, `damp`
// relative (--ldlq-damp, default 0.01). If the Cholesky still meets a non-positive pivot, or the
// resulting U is not finite with a positive diagonal, it retries from the ORIGINAL H at 10x and then
// 100x damp, logging each retry to stderr; a third failure throws naming K and all three damps.
// `damp_used` / `retries` in the returned factor say what actually happened (the CLI logs them per
// linear). damp == 0 is allowed (tests use it with H = I to prove LDLQ degenerates to the plain
// no-refit search); its retries use 1e-3 and 1e-2, since 10 x 0 would retry the same failure.
//
// ---- Per-group parameters: chosen late, from the updated weights, WITHOUT the refit ------------
// Every (row, group) still carries one (scale, zero) / scale / E8M0 exponent. They are chosen at the
// group's FIRST column, from the group's CURRENT weights (all error feedback from earlier columns
// already applied -- choosing from the original weights would fit a grid to values the loop no
// longer rounds), by the measured Milestone 10 grid search in quant_search.hpp weighted by the
// UNDAMPED diag(H) (the imatrix is exactly diag(H), so this is the same importance --quant search
// uses). The search's final least-squares scale refit is skipped (refit = false): it fits the scale
// to the codes the search just picked with the group's weights held fixed, but LDLQ then re-rounds
// every column of the group after that column has absorbed more error feedback, so the refit would
// be fitted to codes that are never stored. The group never straddles a 128-column block
// (128 % group == 0 is required), so its columns are all in the block being processed.
//
// Per-column rounding, given the group's parameters (identical formulas to the RTN quantizers):
//   w4a16: q = clamp(round_half_away(w / sc) + z, 0, 15), dequant sc * (q - z);
//          a degenerate all-zero group (sc == 0) stores q = z and dequantizes to 0.
//   w4a8:  q = clamp(round_half_away(w / sc), -8, 7) + 8, dequant sc * (q - 8).
//   mxfp4: code = EncodeE2M1(w / 2^(raw-127)), dequant sign * |e2m1| * 2^(raw-127) (exact in fp32);
//          wref = max raw over the row's groups, as QuantizeMxfp4Search.
// The dequantized value uses the fp32 scale, not the f16 value PackW4A16Scales / PackW4A8Scales
// store (a relative 2^-11 difference, far below the 4-bit step), like the search's error model.
//
// ---- Determinism -----------------------------------------------------------------------------
// Output bytes do not depend on nthreads. Tile boundaries are fixed (128 rows), tiles share nothing
// mutable, and every operation inside a tile runs in a fixed order: the in-block update is one
// fused multiply-add per element, y = fma(-err, U[i][j], y) (the AVX-512 path's _mm512_fmadd_ps with
// a negated broadcast and the scalar path's std::fma produce the same bits), and the block update is
// linalg::SubMatMulSerial, whose accumulation order is fixed (dense_linalg.hpp). FactorHessian's
// Cholesky / inverse are likewise thread-count independent. fp contraction is disabled in the
// scalar arithmetic here so no compiler-chosen FMA changes a rounding on one build only.
// (One sub-bit nit, harmless: fma(-err, 0, -0.0f) can yield +0.0f when err < 0, so with an exactly
// diagonal H a weight that is exactly -0.0 may encode as mxfp4 code +0 instead of -0. Both decode
// to 0; no other layout distinguishes the sign of zero.)
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx_convert/dense_linalg.hpp"
#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/quant_mxfp4.hpp"
#include "r4dx_convert/quant_search.hpp"
#include "r4dx_convert/threadpool.hpp"

namespace r4dx_convert {

struct LdlqFactor {
  int64_t K = 0;
  std::vector<float> U;       // K*K row-major, upper triangular, U = chol(H_damped^-1): H^-1 = U^T U
  std::vector<float> diag_h;  // K, the UNDAMPED diag(H) (importance weights for the group search)
  float damp_used = 0.0f;     // relative damping actually applied
  int retries = 0;
};

namespace ldlq_detail {

inline constexpr int kTileRows = 128;   // R: rows per independent tile (one thread each)
inline constexpr int kBlockCols = 128;  // B: columns per lazy-update block
inline constexpr float kZeroDampRetryBase = 1e-4f;  // damp == 0 retries at 10x / 100x of this

// y[0..n) = fma(-a, x[j], y[j]) -- the in-block error-feedback row update.
R4DX_TARGET_AVX512 inline void AxpyNegAvx512(float a, const float* x, float* y, int64_t n) {
  const __m512 na = _mm512_set1_ps(-a);
  int64_t j = 0;
  for (; j + 16 <= n; j += 16) {
    _mm512_storeu_ps(y + j,
                     _mm512_fmadd_ps(na, _mm512_loadu_ps(x + j), _mm512_loadu_ps(y + j)));
  }
  if (j < n) {
    const __mmask16 m = static_cast<__mmask16>((1u << static_cast<unsigned>(n - j)) - 1u);
    _mm512_mask_storeu_ps(
        y + j, m,
        _mm512_fmadd_ps(na, _mm512_maskz_loadu_ps(m, x + j), _mm512_maskz_loadu_ps(m, y + j)));
  }
}

inline void AxpyNeg(bool avx512, float a, const float* x, float* y, int64_t n) {
  if (n <= 0) return;
  if (avx512) {
    AxpyNegAvx512(a, x, y, n);
    return;
  }
  for (int64_t j = 0; j < n; ++j) y[j] = std::fma(-a, x[j], y[j]);
}

inline void CheckArgs(const char* who, int N, int K, int group, const LdlqFactor& f) {
  const std::string w(who);
  if (N < 0 || K <= 0) throw std::runtime_error(w + ": bad shape N=" + std::to_string(N) +
                                                " K=" + std::to_string(K));
  if (K % kBlockCols != 0)
    throw std::runtime_error(w + ": K=" + std::to_string(K) + " is not a multiple of the " +
                             std::to_string(kBlockCols) + "-column LDLQ block");
  if (group <= 0 || kBlockCols % group != 0)
    throw std::runtime_error(w + ": group=" + std::to_string(group) + " does not divide the " +
                             std::to_string(kBlockCols) + "-column LDLQ block");
  if (f.K != K || f.U.size() != static_cast<size_t>(K) * static_cast<size_t>(K) ||
      f.diag_h.size() != static_cast<size_t>(K))
    throw std::runtime_error(w + ": LDLQ factor is for K=" + std::to_string(f.K) +
                             " but the weight has K=" + std::to_string(K));
}

// The shared LDLQ loop. `Policy` (copied once per worker, so its scratch is thread-private) has
//   void  Choose(int64_t row, int r, int g, const float* wg, const float* wt);
//         -- pick row `row`'s (tile-local r) group-g parameters from the current weights wg[0..group)
//            with importance wt, and store them in the outputs;
//   float Quant(int64_t row, int r, int64_t k, float x);
//         -- round element (row, k) with the row's current group parameters, store the code, return
//            the dequantized value.
template <class Policy>
void Run(const float* w, int N, int K, int group, const LdlqFactor& f, int nthreads,
         const Policy& proto) {
  const int64_t Kl = K;
  const int64_t ntiles = (static_cast<int64_t>(N) + kTileRows - 1) / kTileRows;
  const float* U = f.U.data();
  const float* diag_h = f.diag_h.data();

  ParallelFor(0, ntiles, nthreads, [&](int64_t t0, int64_t t1) {
    R4DX_NO_FP_CONTRACT
    Policy pol = proto;
    const bool avx512 = linalg::UseAvx512();
    std::vector<float> Wt(static_cast<size_t>(kTileRows) * static_cast<size_t>(Kl));
    std::vector<float> E(static_cast<size_t>(kTileRows) * kBlockCols);
    for (int64_t tile = t0; tile < t1; ++tile) {
      const int64_t row0 = tile * kTileRows;
      const int rows = static_cast<int>(std::min<int64_t>(kTileRows, N - row0));
      std::memcpy(Wt.data(), w + row0 * Kl, sizeof(float) * static_cast<size_t>(rows * Kl));

      for (int64_t b0 = 0; b0 < Kl; b0 += kBlockCols) {
        const int64_t b1 = b0 + kBlockCols;
        for (int64_t i = b0; i < b1; ++i) {
          if ((i - b0) % group == 0) {
            const int g = static_cast<int>(i / group);
            for (int r = 0; r < rows; ++r)
              pol.Choose(row0 + r, r, g, Wt.data() + r * Kl + i, diag_h + i);
          }
          const float uii = U[i * Kl + i];
          const float* urow = U + i * Kl + (i + 1);
          const int64_t n_rest = b1 - (i + 1);
          for (int r = 0; r < rows; ++r) {
            float* wr = Wt.data() + r * Kl;
            const float x = wr[i];
            const float dq = pol.Quant(row0 + r, r, i, x);
            const float err = (x - dq) / uii;
            E[static_cast<size_t>(r) * kBlockCols + static_cast<size_t>(i - b0)] = err;
            AxpyNeg(avx512, err, urow, wr + i + 1, n_rest);
          }
        }
        // W[:, b1:] -= E . U[b0:b1, b1:]
        if (b1 < Kl) {
          linalg::SubMatMulSerial(rows, Kl - b1, kBlockCols, E.data(), kBlockCols,
                                  U + b0 * Kl + b1, Kl, Wt.data() + b1, Kl);
        }
      }
    }
  });
}

struct AsymPolicy {
  int64_t K = 0;
  int group = 0, groups_per_row = 0;
  uint8_t* q = nullptr;
  float* scale = nullptr;
  uint8_t* zero = nullptr;
  std::vector<int32_t> rq;
  std::vector<uint8_t> best_q, qg;
  float sc[kTileRows] = {};
  int zp[kTileRows] = {};

  void Choose(int64_t row, int r, int g, const float* wg, const float* wt) {
    SearchInt4AsymGroup(wg, wt, group, /*refit=*/false, rq.data(), best_q.data(), qg.data(),
                        &sc[r], &zp[r]);
    const size_t gi = static_cast<size_t>(row) * groups_per_row + g;
    scale[gi] = sc[r];
    zero[gi] = static_cast<uint8_t>(zp[r]);
  }
  float Quant(int64_t row, int r, int64_t k, float x) {
    R4DX_NO_FP_CONTRACT
    const float s = sc[r];
    const int z = zp[r];
    uint8_t& out = q[row * K + k];
    if (s == 0.0f) {  // degenerate all-zero group at choice time: code z, value 0
      out = static_cast<uint8_t>(z);
      return 0.0f;
    }
    const int qi = ClampInt(RoundHalfAwayFromZero(x / s) + z, 0, 15);
    out = static_cast<uint8_t>(qi);
    return s * (static_cast<float>(qi) - static_cast<float>(z));
  }
};

struct Pinned8Policy {
  int64_t K = 0;
  int group = 0, groups_per_row = 0;
  uint8_t* q = nullptr;
  float* scale = nullptr;
  std::vector<uint8_t> best_q, qg;
  float sc[kTileRows] = {};

  void Choose(int64_t row, int r, int g, const float* wg, const float* wt) {
    SearchInt4Pinned8Group(wg, wt, group, /*refit=*/false, best_q.data(), qg.data(), &sc[r]);
    scale[static_cast<size_t>(row) * groups_per_row + g] = sc[r];
  }
  float Quant(int64_t row, int r, int64_t k, float x) {
    R4DX_NO_FP_CONTRACT
    const float s = sc[r];
    const int qs = ClampInt(RoundHalfAwayFromZero(x / s), -8, 7);
    q[row * K + k] = static_cast<uint8_t>(qs + 8);
    return s * static_cast<float>(qs);
  }
};

struct Mxfp4Policy {
  int64_t K = 0;
  int group = 0, groups_per_row = 0;
  uint8_t* packed = nullptr;
  uint8_t* escale = nullptr;
  std::vector<uint8_t> codes, best_codes;
  float sf[kTileRows] = {};

  void Choose(int64_t row, int r, int g, const float* wg, const float* wt) {
    int raw = 0;
    SearchMxfp4Group(wg, wt, group, codes.data(), best_codes.data(), &raw);
    escale[static_cast<size_t>(row) * groups_per_row + g] = static_cast<uint8_t>(raw);
    sf[r] = std::ldexp(1.0f, raw - 127);
  }
  float Quant(int64_t row, int r, int64_t k, float x) {
    R4DX_NO_FP_CONTRACT
    const float s = sf[r];
    const uint8_t code = EncodeE2M1(x / s);
    uint8_t& byte = packed[row * (K / 2) + k / 2];
    if (k % 2 == 0) {
      byte = static_cast<uint8_t>((byte & 0xF0u) | (code & 0x0Fu));
    } else {
      byte = static_cast<uint8_t>((byte & 0x0Fu) | ((code & 0x0Fu) << 4));
    }
    const float mag = kE2M1Magnitude[code & 0x7];
    return ((code & 0x8) ? -mag : mag) * s;  // exact: <= 3 significant bits times a power of two
  }
};

// Flat reversal of a K*K array (== J M J), parallel over mirrored row pairs.
inline void ReverseSquareInPlace(std::vector<float>& a, int64_t K, int nthreads) {
  const int64_t n = K * K;
  const int64_t half = n / 2;
  const int64_t chunk = std::max<int64_t>(1, K);
  const int64_t chunks = (half + chunk - 1) / chunk;
  ParallelFor(0, chunks, nthreads, [&](int64_t c0, int64_t c1) {
    for (int64_t c = c0; c < c1; ++c) {
      const int64_t p0 = c * chunk, p1 = std::min(half, p0 + chunk);
      for (int64_t p = p0; p < p1; ++p) std::swap(a[p], a[n - 1 - p]);
    }
  });
}

}  // namespace ldlq_detail

// H (full symmetric K x K, row-major; consumed) -> U = chol((H + damp*mean(diag)*I)^-1), upper.
// See the header comment for the reversal trick and the retry schedule.
inline LdlqFactor FactorHessian(std::vector<float> H, int64_t K, float damp, int nthreads) {
  if (K <= 0 || H.size() != static_cast<size_t>(K) * static_cast<size_t>(K)) {
    throw std::runtime_error("FactorHessian: H has " + std::to_string(H.size()) +
                             " elements, expected K*K with K=" + std::to_string(K));
  }
  if (!(damp >= 0.0f) || !std::isfinite(damp)) {
    throw std::runtime_error("FactorHessian: damp must be finite and >= 0 (got " +
                             std::to_string(damp) + ")");
  }
  LdlqFactor f;
  f.K = K;
  f.diag_h.resize(static_cast<size_t>(K));
  double dsum = 0.0;
  for (int64_t i = 0; i < K; ++i) {
    const float d = H[static_cast<size_t>(i * K + i)];
    f.diag_h[static_cast<size_t>(i)] = d;
    dsum += static_cast<double>(d);
  }
  const double mean_diag = dsum / static_cast<double>(K);
  if (!(mean_diag > 0.0) || !std::isfinite(mean_diag)) {
    throw std::runtime_error("FactorHessian: mean(diag(H)) = " + std::to_string(mean_diag) +
                             " for K=" + std::to_string(K) +
                             " -- H is zero or non-finite, no damping can make it positive definite");
  }

  const float retry_base = damp > 0.0f ? damp : ldlq_detail::kZeroDampRetryBase;
  const float damps[3] = {damp, retry_base * 10.0f, retry_base * 100.0f};
  std::vector<float> A;
  for (int attempt = 0; attempt < 3; ++attempt) {
    const float d_rel = damps[attempt];
    const float add = static_cast<float>(static_cast<double>(d_rel) * mean_diag);
    // A = J (H + add*I) J: flat reversal of H, then the (reversal-invariant) diagonal shift.
    A.resize(H.size());
    ParallelFor(0, K, nthreads, [&](int64_t r0, int64_t r1) {
      for (int64_t i = r0; i < r1; ++i) {
        const float* src = H.data() + (K - 1 - i) * K;
        float* dst = A.data() + i * K;
        for (int64_t j = 0; j < K; ++j) dst[j] = src[K - 1 - j];
        dst[i] = dst[i] + add;
      }
    });

    bool ok = linalg::CholeskyLower(A.data(), K, nthreads);
    if (ok) {
      linalg::ZeroStrictUpper(A.data(), K);
      linalg::InvertLower(A.data(), K, nthreads);
      ldlq_detail::ReverseSquareInPlace(A, K, nthreads);  // U = J L^-1 J
      // A tiny pivot can overflow the inverse; a non-finite U would silently wreck every row.
      std::atomic<bool> bad{false};
      ParallelFor(0, K, nthreads, [&](int64_t r0, int64_t r1) {
        for (int64_t i = r0; i < r1 && !bad.load(std::memory_order_relaxed); ++i) {
          const float* u = A.data() + i * K;
          if (!(u[i] > 0.0f) || !std::isfinite(u[i])) {
            bad = true;
            return;
          }
          for (int64_t j = i + 1; j < K; ++j) {
            if (!std::isfinite(u[j])) {
              bad = true;
              return;
            }
          }
        }
      });
      ok = !bad.load();
    }
    if (ok) {
      f.U = std::move(A);
      f.damp_used = d_rel;
      f.retries = attempt;
      return f;
    }
    if (attempt < 2) {
      std::fprintf(stderr,
                   "[ldlq] FactorHessian: K=%lld not positive definite at damp=%g (mean diag %g); "
                   "retrying at damp=%g\n",
                   static_cast<long long>(K), static_cast<double>(d_rel), mean_diag,
                   static_cast<double>(damps[attempt + 1]));
    }
  }
  throw std::runtime_error("FactorHessian: Cholesky of H (K=" + std::to_string(K) +
                           ") failed at damp=" + std::to_string(damps[0]) + ", " +
                           std::to_string(damps[1]) + " and " + std::to_string(damps[2]));
}

// ---- the three layouts: same outputs/argument order as the RTN quantizers, plus the factor ----

inline void QuantizeInt4AsymmetricLdlq(const float* w, int N, int K, int group,
                                       const LdlqFactor& f, int nthreads, std::vector<uint8_t>& q,
                                       std::vector<float>& scale, std::vector<uint8_t>& zero) {
  ldlq_detail::CheckArgs("QuantizeInt4AsymmetricLdlq", N, K, group, f);
  const int groups_per_row = K / group;
  q.assign(static_cast<size_t>(N) * K, 0);
  scale.assign(static_cast<size_t>(N) * groups_per_row, 0.0f);
  zero.assign(static_cast<size_t>(N) * groups_per_row, 0);

  ldlq_detail::AsymPolicy p;
  p.K = K;
  p.group = group;
  p.groups_per_row = groups_per_row;
  p.q = q.data();
  p.scale = scale.data();
  p.zero = zero.data();
  p.rq.resize(static_cast<size_t>(group));
  p.best_q.resize(static_cast<size_t>(group));
  p.qg.resize(static_cast<size_t>(group));
  ldlq_detail::Run(w, N, K, group, f, nthreads, p);
}

inline void QuantizeInt4Pinned8Ldlq(const float* w, int N, int K, int group, const LdlqFactor& f,
                                    int nthreads, std::vector<uint8_t>& q,
                                    std::vector<float>& scale) {
  ldlq_detail::CheckArgs("QuantizeInt4Pinned8Ldlq", N, K, group, f);
  const int groups_per_row = K / group;
  q.assign(static_cast<size_t>(N) * K, 0);
  scale.assign(static_cast<size_t>(N) * groups_per_row, 0.0f);

  ldlq_detail::Pinned8Policy p;
  p.K = K;
  p.group = group;
  p.groups_per_row = groups_per_row;
  p.q = q.data();
  p.scale = scale.data();
  p.best_q.resize(static_cast<size_t>(group));
  p.qg.resize(static_cast<size_t>(group));
  ldlq_detail::Run(w, N, K, group, f, nthreads, p);
}

inline Mxfp4Quantized QuantizeMxfp4Ldlq(const float* w, int N, int K, int group,
                                        const LdlqFactor& f, int nthreads) {
  ldlq_detail::CheckArgs("QuantizeMxfp4Ldlq", N, K, group, f);
  const int groups_per_row = K / group;
  Mxfp4Quantized out;
  out.packed.assign(static_cast<size_t>(N) * (K / 2), 0);
  out.escale.assign(static_cast<size_t>(N) * groups_per_row, 0);
  out.wref.assign(static_cast<size_t>(N), 0);

  ldlq_detail::Mxfp4Policy p;
  p.K = K;
  p.group = group;
  p.groups_per_row = groups_per_row;
  p.packed = out.packed.data();
  p.escale = out.escale.data();
  p.codes.resize(static_cast<size_t>(group));
  p.best_codes.resize(static_cast<size_t>(group));
  ldlq_detail::Run(w, N, K, group, f, nthreads, p);

  for (int64_t row = 0; row < N; ++row) {
    const uint8_t* e = out.escale.data() + row * groups_per_row;
    out.wref[static_cast<size_t>(row)] = *std::max_element(e, e + groups_per_row);
  }
  return out;
}

}  // namespace r4dx_convert
