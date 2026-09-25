// r4dx_convert::linalg -- the dense float32 kernels LDLQ needs (docs/quant2.md section 2.2):
// C -= A.B, a blocked lower Cholesky, and a blocked lower-triangular inverse. Row-major throughout.
//
// ---- Determinism -------------------------------------------------------------------------------
// Every output element is produced by the same sequence of IEEE operations no matter how many
// threads run or where the element falls inside a register tile: `acc` starts at +0, is updated by
// one fused multiply-add per k in ascending k, and is then subtracted from C once. The AVX-512
// microkernel, its masked column tail and the scalar fallback (std::fma) all follow that sequence,
// and threads only ever split OUTPUT rows. So a converted container's bytes do not depend on
// --threads (tests/convert/test_quant_ldlq.cpp gates it).
//
// ---- Why hand-written and not a BLAS ------------------------------------------------------------
// The repo vendors header-only libraries only, and r4dx-convert is the single reproducible producer
// of containers. The FMA path is compiled per function (`target("avx512f")`) and dispatched at run
// time, so the rest of the converter keeps its default code generation -- in particular no FMA
// contraction reaches the RTN / search quantizers, whose bytes are pinned to the Python references.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(_MSC_VER)
#include <intrin.h>
#endif
#include <immintrin.h>

#include "r4dx_convert/threadpool.hpp"

#if defined(__clang__)
#define R4DX_TARGET_AVX512 __attribute__((target("avx512f")))
#else
#define R4DX_TARGET_AVX512
#endif

namespace r4dx_convert {
namespace linalg {

inline bool CpuHasAvx512f() {
#if defined(_M_X64) || defined(__x86_64__)
  static const bool has = []() {
    int r[4];
#if defined(_MSC_VER)
    __cpuid(r, 0);
    if (r[0] < 7) return false;
    __cpuid(r, 1);
    const bool osxsave = (r[2] >> 27) & 1;
    if (!osxsave) return false;
    const unsigned long long xcr0 = _xgetbv(0);
    // XMM (1), YMM (2), opmask (5), ZMM_Hi256 (6), Hi16_ZMM (7) state all OS-enabled.
    if ((xcr0 & 0xE6) != 0xE6) return false;
    __cpuidex(r, 7, 0);
    return ((r[1] >> 16) & 1) != 0;  // EBX bit 16 = AVX512F
#else
    return __builtin_cpu_supports("avx512f");
#endif
  }();
  return has;
#else
  return false;
#endif
}

// Force the scalar path (tests use it to prove the two paths agree bit for bit).
inline bool& ForceScalar() {
  static bool force = false;
  return force;
}

inline bool UseAvx512() { return !ForceScalar() && CpuHasAvx512f(); }

// ---- C[M,N] -= A[M,K] . B[K,N], serial -------------------------------------------------------

inline void SubMatMulScalar(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda,
                            const float* B, int64_t ldb, float* C, int64_t ldc) {
  for (int64_t i = 0; i < M; ++i) {
    const float* a = A + i * lda;
    float* c = C + i * ldc;
    for (int64_t j = 0; j < N; ++j) {
      float acc = 0.0f;
      for (int64_t k = 0; k < K; ++k) acc = std::fma(a[k], B[k * ldb + j], acc);
      c[j] = c[j] - acc;
    }
  }
}

// 8 rows x 32 columns per register tile (16 zmm accumulators), masked on the column tail and
// row-count-generic on the row tail. Loop order: column panel outermost so a K x 32 panel of B
// stays in L1 while every row group of A streams past it.
R4DX_TARGET_AVX512 inline void SubMatMulAvx512(int64_t M, int64_t N, int64_t K, const float* A,
                                               int64_t lda, const float* B, int64_t ldb, float* C,
                                               int64_t ldc) {
  constexpr int kRows = 8;
  for (int64_t j0 = 0; j0 < N; j0 += 32) {
    const int64_t nc = std::min<int64_t>(32, N - j0);
    const __mmask16 m0 = nc >= 16 ? static_cast<__mmask16>(0xFFFF)
                                  : static_cast<__mmask16>((1u << nc) - 1u);
    const __mmask16 m1 = nc >= 32   ? static_cast<__mmask16>(0xFFFF)
                         : nc <= 16 ? static_cast<__mmask16>(0)
                                    : static_cast<__mmask16>((1u << (nc - 16)) - 1u);
    for (int64_t i0 = 0; i0 < M; i0 += kRows) {
      const int rows = static_cast<int>(std::min<int64_t>(kRows, M - i0));
      __m512 acc0[kRows], acc1[kRows];
      for (int r = 0; r < kRows; ++r) {
        acc0[r] = _mm512_setzero_ps();
        acc1[r] = _mm512_setzero_ps();
      }
      const float* bk = B + j0;
      if (rows == kRows) {
        const float* a0 = A + i0 * lda;
        for (int64_t k = 0; k < K; ++k, bk += ldb) {
          const __m512 b0 = _mm512_maskz_loadu_ps(m0, bk);
          const __m512 b1 = _mm512_maskz_loadu_ps(m1, bk + 16);
          for (int r = 0; r < kRows; ++r) {
            const __m512 a = _mm512_set1_ps(a0[r * lda + k]);
            acc0[r] = _mm512_fmadd_ps(a, b0, acc0[r]);
            acc1[r] = _mm512_fmadd_ps(a, b1, acc1[r]);
          }
        }
      } else {
        const float* a0 = A + i0 * lda;
        for (int64_t k = 0; k < K; ++k, bk += ldb) {
          const __m512 b0 = _mm512_maskz_loadu_ps(m0, bk);
          const __m512 b1 = _mm512_maskz_loadu_ps(m1, bk + 16);
          for (int r = 0; r < rows; ++r) {
            const __m512 a = _mm512_set1_ps(a0[r * lda + k]);
            acc0[r] = _mm512_fmadd_ps(a, b0, acc0[r]);
            acc1[r] = _mm512_fmadd_ps(a, b1, acc1[r]);
          }
        }
      }
      for (int r = 0; r < rows; ++r) {
        float* c = C + (i0 + r) * ldc + j0;
        _mm512_mask_storeu_ps(c, m0, _mm512_sub_ps(_mm512_maskz_loadu_ps(m0, c), acc0[r]));
        _mm512_mask_storeu_ps(c + 16, m1,
                              _mm512_sub_ps(_mm512_maskz_loadu_ps(m1, c + 16), acc1[r]));
      }
    }
  }
}

inline void SubMatMulSerial(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda,
                            const float* B, int64_t ldb, float* C, int64_t ldc) {
  if (M <= 0 || N <= 0) return;
  if (K <= 0) return;  // acc stays +0; c - 0 == c for every finite and infinite c
  if (UseAvx512())
    SubMatMulAvx512(M, N, K, A, lda, B, ldb, C, ldc);
  else
    SubMatMulScalar(M, N, K, A, lda, B, ldb, C, ldc);
}

// Parallel over row groups of 8, dealt cyclically so a triangular caller balances.
inline void SubMatMul(int64_t M, int64_t N, int64_t K, const float* A, int64_t lda, const float* B,
                      int64_t ldb, float* C, int64_t ldc, int nthreads) {
  const int64_t groups = (M + 7) / 8;
  const int t = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nthreads, groups)));
  ParallelFor(0, t, t, [&](int64_t t0, int64_t t1) {
    for (int64_t tid = t0; tid < t1; ++tid) {
      for (int64_t g = tid; g < groups; g += t) {
        const int64_t i0 = g * 8;
        const int64_t rows = std::min<int64_t>(8, M - i0);
        SubMatMulSerial(rows, N, K, A + i0 * lda, lda, B, ldb, C + i0 * ldc, ldc);
      }
    }
  });
}

// ---- blocked lower Cholesky, in place ---------------------------------------------------------
//
// A (n x n, row-major, symmetric; only the lower triangle is read) -> L with A = L L^T in the lower
// triangle. The strict upper triangle is NOT preserved: the trailing update writes whole 8 x 8
// squares on the diagonal (see step 3), so it holds partial-update garbage there afterwards --
// callers call ZeroStrictUpper before using L as a triangular matrix, and must keep their own copy
// of A if they need it again (FactorHessian retries from its original H). Returns false on a
// non-positive or non-finite pivot (the caller retries with more damping). Diagonal blocks and the
// panel solve run in double; the trailing update -- where all the flops are -- is SubMatMul.
inline bool CholeskyLower(float* A, int64_t n, int nthreads, int64_t nb = 128) {
  std::vector<double> d(static_cast<size_t>(nb * nb));
  std::vector<float> pt;  // panel transposed: [nb, n - k1]
  for (int64_t k0 = 0; k0 < n; k0 += nb) {
    const int64_t k1 = std::min(n, k0 + nb);
    const int64_t b = k1 - k0;
    // 1) factor the diagonal block (unblocked, double).
    for (int64_t i = 0; i < b; ++i)
      for (int64_t j = 0; j <= i; ++j) d[i * nb + j] = A[(k0 + i) * n + (k0 + j)];
    for (int64_t j = 0; j < b; ++j) {
      double s = d[j * nb + j];
      for (int64_t t = 0; t < j; ++t) s -= d[j * nb + t] * d[j * nb + t];
      if (!(s > 0.0) || !std::isfinite(s)) return false;
      const double ljj = std::sqrt(s);
      d[j * nb + j] = ljj;
      for (int64_t i = j + 1; i < b; ++i) {
        double v = d[i * nb + j];
        for (int64_t t = 0; t < j; ++t) v -= d[i * nb + t] * d[j * nb + t];
        d[i * nb + j] = v / ljj;
      }
    }
    for (int64_t i = 0; i < b; ++i)
      for (int64_t j = 0; j <= i; ++j) A[(k0 + i) * n + (k0 + j)] = static_cast<float>(d[i * nb + j]);
    if (k1 >= n) break;
    // 2) panel: rows k1..n-1, columns k0..k1-1:  X L11^T = A21  (forward substitution per row).
    ParallelFor(k1, n, nthreads, [&](int64_t r0, int64_t r1) {
      std::vector<double> x(static_cast<size_t>(nb));
      for (int64_t r = r0; r < r1; ++r) {
        float* row = A + r * n + k0;
        for (int64_t j = 0; j < b; ++j) {
          double v = row[j];
          for (int64_t t = 0; t < j; ++t) v -= x[t] * d[j * nb + t];
          x[j] = v / d[j * nb + j];
        }
        for (int64_t j = 0; j < b; ++j) row[j] = static_cast<float>(x[j]);
      }
    });
    // 3) trailing lower update: A22 -= P P^T, P = the panel just solved. Row group g of A22 only
    //    needs columns k1 .. (its own last row), so each call is sized to the triangle (and so also
    //    updates the strict-upper half of its own 8 x 8 diagonal square, which nothing reads).
    const int64_t m = n - k1;
    pt.assign(static_cast<size_t>(b * m), 0.0f);
    for (int64_t r = 0; r < m; ++r)
      for (int64_t j = 0; j < b; ++j) pt[j * m + r] = A[(k1 + r) * n + k0 + j];
    const int64_t groups = (m + 7) / 8;
    const int t = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(nthreads, groups)));
    ParallelFor(0, t, t, [&](int64_t t0, int64_t t1) {
      for (int64_t tid = t0; tid < t1; ++tid) {
        for (int64_t g = tid; g < groups; g += t) {
          const int64_t i0 = g * 8;
          const int64_t rows = std::min<int64_t>(8, m - i0);
          const int64_t cols = i0 + rows;  // lower triangle incl. this group's diagonal square
          SubMatMulSerial(rows, cols, b, A + (k1 + i0) * n + k0, n, pt.data(), m,
                          A + (k1 + i0) * n + k1, n);
        }
      }
    });
  }
  return true;
}

// Zero the strict upper triangle (InvertLower's products rely on it).
inline void ZeroStrictUpper(float* A, int64_t n) {
  for (int64_t i = 0; i < n; ++i) std::fill(A + i * n + i + 1, A + (i + 1) * n, 0.0f);
}

// ---- blocked lower-triangular inverse, in place -------------------------------------------------
//
// L (n x n lower, strict upper already zero) -> L^-1. Row block I, in increasing order:
//   T = L[I, 0:i0] . M[0:i0, 0:i0]      (rows above I are already inverted; M lower, upper zeros)
//   M[I, I] = inv(L[I, I])              (double, unblocked)
//   M[I, 0:i0] = -M[I, I] . T
inline void InvertLower(float* L, int64_t n, int nthreads, int64_t nb = 128) {
  std::vector<double> d(static_cast<size_t>(nb * nb)), inv(static_cast<size_t>(nb * nb));
  std::vector<float> invf(static_cast<size_t>(nb * nb));
  std::vector<float> T;
  for (int64_t i0 = 0; i0 < n; i0 += nb) {
    const int64_t i1 = std::min(n, i0 + nb);
    const int64_t b = i1 - i0;
    // T = L[I, 0:i0] . M[0:i0, 0:i0]  (computed as 0 - (-(...)) via SubMatMul on a zero C, then
    // negated below together with the M_II product -- written out to keep one kernel).
    if (i0 > 0) {
      T.assign(static_cast<size_t>(b * i0), 0.0f);
      SubMatMul(b, i0, i0, L + i0 * n, n, L, n, T.data(), i0, nthreads);  // T = -L_I0 . M
    }
    // inv(L_II) in double.
    for (int64_t i = 0; i < b; ++i)
      for (int64_t j = 0; j < b; ++j)
        d[i * nb + j] = j <= i ? static_cast<double>(L[(i0 + i) * n + i0 + j]) : 0.0;
    for (int64_t j = 0; j < b; ++j) {
      for (int64_t i = 0; i < b; ++i) inv[i * nb + j] = 0.0;
      inv[j * nb + j] = 1.0 / d[j * nb + j];
      for (int64_t i = j + 1; i < b; ++i) {
        double s = 0.0;
        for (int64_t t = j; t < i; ++t) s += d[i * nb + t] * inv[t * nb + j];
        // (0 - s), not -s: identical for every nonzero s, but +0.0 rather than -0.0 when s == 0, so
        // the structural zeros of inv(I) (and of any block-diagonal L) are +0.0 byte for byte.
        inv[i * nb + j] = (0.0 - s) / d[i * nb + i];
      }
    }
    for (int64_t i = 0; i < b; ++i)
      for (int64_t j = 0; j < b; ++j) {
        const float v = static_cast<float>(inv[i * nb + j]);
        invf[i * nb + j] = v;
        L[(i0 + i) * n + i0 + j] = v;
      }
    if (i0 > 0) {
      // M[I, 0:i0] = -inv . L_I0 . M = inv . T  (T already holds -L_I0 . M).  C = 0 - (-inv) . T.
      for (int64_t i = 0; i < b; ++i) std::fill(L + (i0 + i) * n, L + (i0 + i) * n + i0, 0.0f);
      for (auto& v : invf) v = -v;
      SubMatMul(b, i0, b, invf.data(), nb, T.data(), i0, L + i0 * n, n, nthreads);
    }
  }
}

}  // namespace linalg
}  // namespace r4dx_convert
