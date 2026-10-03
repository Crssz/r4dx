// tests/kernels/rotation_ref.hpp -- fp64 CPU reference for the quant2 rotation kernels
// (docs/quant2.md sections 3-4; src/kernels/include/r4dx/kernels/rotate_residual.h), written from
// the rotation contract's definitions and deliberately NOT sharing code with src/convert or with
// the kernels: a transcription error in either shows up as a mismatch here instead of cancelling.
// Header-only and host-only, used by test_rotation_ref_cpu (CPU checks of this reference itself),
// test_rotate_residual and test_attn_gate_mul_hadamard.
//
// Contract, row-vector convention (x a row):
//   x Q:   y = x * d;  y[block b] = FWHT(y[block b]) / 32 (5 blocks of 1024, natural order);
//          z[b*1024 + i] = sum_c y[c*1024 + i] * R[c][b]            (R row-major, R[c*5 + b])
//   x Q^T: y[c*1024 + i] = sum_b x[b*1024 + i] * R[c][b];  FWHT / 32 per block;  * d
//   h Hb:  (h * s) then FWHT / sqrt(B) on each contiguous block of B
#pragma once

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

namespace rotation_ref {

constexpr int64_t kHidden = 5120;
constexpr int64_t kBlock = 1024;
constexpr int kBlocks = 5;

// Unnormalized fast Walsh-Hadamard transform in natural (Sylvester) order, in place over n (a power
// of two) values: out[j] = sum_i in[i] * (-1)^popcount(i & j).
inline void Fwht(double* v, int64_t n) {
  for (int64_t h = 1; h < n; h *= 2) {
    for (int64_t i = 0; i < n; i += 2 * h) {
      for (int64_t j = i; j < i + h; ++j) {
        const double a = v[j];
        const double b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
      }
    }
  }
}

// h Hb over one row of K = h.size() values; s holds K signs (the row-local slice).
inline std::vector<double> ApplyHb(const std::vector<double>& h, const float* s, int64_t B) {
  const int64_t K = static_cast<int64_t>(h.size());
  std::vector<double> out(h.size());
  for (int64_t k = 0; k < K; ++k) out[k] = h[k] * static_cast<double>(s[k]);
  const double scale = 1.0 / std::sqrt(static_cast<double>(B));
  for (int64_t k0 = 0; k0 < K; k0 += B) {
    Fwht(out.data() + k0, B);
    for (int64_t k = k0; k < k0 + B; ++k) out[k] *= scale;
  }
  return out;
}

// h Hb^T = the inverse of ApplyHb: FWHT / sqrt(B) per block, then * s.
inline std::vector<double> ApplyHbT(const std::vector<double>& h, const float* s, int64_t B) {
  const int64_t K = static_cast<int64_t>(h.size());
  std::vector<double> out = h;
  const double scale = 1.0 / std::sqrt(static_cast<double>(B));
  for (int64_t k0 = 0; k0 < K; k0 += B) {
    Fwht(out.data() + k0, B);
    for (int64_t k = k0; k < k0 + B; ++k) out[k] *= scale;
  }
  for (int64_t k = 0; k < K; ++k) out[k] *= static_cast<double>(s[k]);
  return out;
}

// x Q (inverse == false) or x Q^T (inverse == true) over one row of kHidden values.
// d: kHidden signs. R: 25 floats, R[c*5 + b].
inline std::vector<double> ApplyQ(const std::vector<double>& x, const float* d, const float* R,
                                  bool inverse) {
  std::vector<double> y(kHidden);
  if (!inverse) {
    for (int64_t k = 0; k < kHidden; ++k) y[k] = x[k] * static_cast<double>(d[k]);
    for (int b = 0; b < kBlocks; ++b) {
      Fwht(y.data() + b * kBlock, kBlock);
      for (int64_t i = 0; i < kBlock; ++i) y[b * kBlock + i] /= 32.0;
    }
    std::vector<double> z(kHidden);
    for (int b = 0; b < kBlocks; ++b) {
      for (int64_t i = 0; i < kBlock; ++i) {
        double acc = 0.0;
        for (int c = 0; c < kBlocks; ++c) acc += y[c * kBlock + i] * static_cast<double>(R[c * 5 + b]);
        z[b * kBlock + i] = acc;
      }
    }
    return z;
  }
  for (int c = 0; c < kBlocks; ++c) {
    for (int64_t i = 0; i < kBlock; ++i) {
      double acc = 0.0;
      for (int b = 0; b < kBlocks; ++b) acc += x[b * kBlock + i] * static_cast<double>(R[c * 5 + b]);
      y[c * kBlock + i] = acc;
    }
  }
  for (int c = 0; c < kBlocks; ++c) {
    Fwht(y.data() + c * kBlock, kBlock);
    for (int64_t i = 0; i < kBlock; ++i) y[c * kBlock + i] /= 32.0;
  }
  for (int64_t k = 0; k < kHidden; ++k) y[k] *= static_cast<double>(d[k]);
  return y;
}

// ---- any hidden = nblk * block (Gemma 4: 3840 = 15 x 256) ------------------------------------------
// The converter's block rule: the largest power of two dividing hidden, capped at 1024.
inline int64_t ChooseBlock(int64_t hidden) {
  const int64_t low = hidden & -hidden;
  return low < 1024 ? low : 1024;
}

// x Q / x Q^T for any hidden = nblk * block, written from the same contract (R: nblk*nblk floats,
// R[c*nblk + b]); equals ApplyQ at (5120, 1024).
inline std::vector<double> ApplyQGeneral(const std::vector<double>& x, int64_t block, const float* d,
                                         const float* R, bool inverse) {
  const int64_t hidden = static_cast<int64_t>(x.size());
  const int64_t nblk = hidden / block;
  const double scale = 1.0 / std::sqrt(static_cast<double>(block));
  std::vector<double> y(static_cast<size_t>(hidden));
  if (!inverse) {
    for (int64_t k = 0; k < hidden; ++k) y[k] = x[k] * static_cast<double>(d[k]);
    for (int64_t b = 0; b < nblk; ++b) {
      Fwht(y.data() + b * block, block);
      for (int64_t i = 0; i < block; ++i) y[b * block + i] *= scale;
    }
    std::vector<double> z(static_cast<size_t>(hidden));
    for (int64_t b = 0; b < nblk; ++b) {
      for (int64_t i = 0; i < block; ++i) {
        double acc = 0.0;
        for (int64_t c = 0; c < nblk; ++c) acc += y[c * block + i] * static_cast<double>(R[c * nblk + b]);
        z[b * block + i] = acc;
      }
    }
    return z;
  }
  for (int64_t c = 0; c < nblk; ++c) {
    for (int64_t i = 0; i < block; ++i) {
      double acc = 0.0;
      for (int64_t b = 0; b < nblk; ++b) acc += x[b * block + i] * static_cast<double>(R[c * nblk + b]);
      y[c * block + i] = acc;
    }
  }
  for (int64_t c = 0; c < nblk; ++c) {
    Fwht(y.data() + c * block, block);
    for (int64_t i = 0; i < block; ++i) y[c * block + i] *= scale;
  }
  for (int64_t k = 0; k < hidden; ++k) y[k] *= static_cast<double>(d[k]);
  return y;
}

// resid + Q(rmsnorm_plain(y, w)), times layer_scale, for one row: the contract of
// r4dx_post_rmsnorm_rotate_add_bf16 (rmsnorm_plain = y * rsqrt(mean(y^2) + eps) * w, no 1 + w).
inline std::vector<double> PostNormRotateAddRef(const std::vector<double>& resid,
                                                const std::vector<double>& y,
                                                const std::vector<double>& w, int64_t block,
                                                const float* d, const float* R, double eps,
                                                double layer_scale) {
  const size_t n = y.size();
  double ss = 0.0;
  for (double v : y) ss += v * v;
  const double rstd = 1.0 / std::sqrt(ss / static_cast<double>(n) + eps);
  std::vector<double> normed(n);
  for (size_t k = 0; k < n; ++k) normed[k] = y[k] * rstd * w[k];
  std::vector<double> rot = ApplyQGeneral(normed, block, d, R, false);
  for (size_t k = 0; k < n; ++k) rot[k] = (resid[k] + rot[k]) * layer_scale;
  return rot;
}

// A random n x n orthogonal matrix, row-major (Gram-Schmidt of a Gaussian matrix, rows in order,
// twice). RandomOrthogonal5 is the n = 5 original; this one has no symmetry either.
inline std::vector<float> RandomOrthogonal(std::mt19937_64& rng, int n) {
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> m(static_cast<size_t>(n) * n);
  for (double& v : m) v = nd(rng);
  for (int r = 0; r < n; ++r) {
    for (int pass = 0; pass < 2; ++pass) {
      for (int p = 0; p < r; ++p) {
        double dot = 0.0;
        for (int c = 0; c < n; ++c) dot += m[r * n + c] * m[p * n + c];
        for (int c = 0; c < n; ++c) m[r * n + c] -= dot * m[p * n + c];
      }
      double nrm = 0.0;
      for (int c = 0; c < n; ++c) nrm += m[r * n + c] * m[r * n + c];
      nrm = std::sqrt(nrm);
      for (int c = 0; c < n; ++c) m[r * n + c] /= nrm;
    }
  }
  std::vector<float> R(m.size());
  for (size_t i = 0; i < m.size(); ++i) R[i] = static_cast<float>(m[i]);
  return R;
}

// n random +-1 signs.
inline std::vector<float> RandomSigns(std::mt19937_64& rng, int64_t n) {
  std::vector<float> s(static_cast<size_t>(n));
  for (auto& v : s) v = (rng() >> 63) ? -1.0f : 1.0f;
  return s;
}

// A random 5x5 orthogonal matrix, row-major: Gram-Schmidt of a Gaussian matrix, rows orthonormalized
// in order (the converter's recipe, reimplemented here independently). Not symmetric, so a kernel
// that reads R[b][c] where the contract says R[c][b] fails the comparison.
inline std::vector<float> RandomOrthogonal5(std::mt19937_64& rng) {
  std::normal_distribution<double> nd(0.0, 1.0);
  double m[5][5];
  for (auto& row : m) {
    for (double& v : row) v = nd(rng);
  }
  for (int r = 0; r < 5; ++r) {
    for (int p = 0; p < r; ++p) {
      double dot = 0.0;
      for (int c = 0; c < 5; ++c) dot += m[r][c] * m[p][c];
      for (int c = 0; c < 5; ++c) m[r][c] -= dot * m[p][c];
    }
    double nrm = 0.0;
    for (int c = 0; c < 5; ++c) nrm += m[r][c] * m[r][c];
    nrm = std::sqrt(nrm);
    for (int c = 0; c < 5; ++c) m[r][c] /= nrm;
  }
  std::vector<float> R(25);
  for (int r = 0; r < 5; ++r) {
    for (int c = 0; c < 5; ++c) R[r * 5 + c] = static_cast<float>(m[r][c]);
  }
  return R;
}

// Tolerance for a kernel output rounded once to bf16 from an fp32 computation of `ref`:
//   |got - ref| <= 2^-8 * |ref| + 1e-4 * rms(ref row)
// The first term is RTNE's half ulp (bf16 has 8 significand bits, so half an ulp is at most 2^-8
// of the value); the second absorbs the fp32 arithmetic's own error (~1e-7 relative to the row's
// scale for these transform sizes) on elements whose exact value is near zero, where a relative
// bound alone would be meaningless. Returns the number of elements outside the bound and reports
// the worst err/bound ratio through *worst.
inline int64_t CountOutOfTolerance(const std::vector<float>& got, const std::vector<double>& ref,
                                   double* worst) {
  double ss = 0.0;
  for (double v : ref) ss += v * v;
  const double rms = std::sqrt(ss / static_cast<double>(ref.size()));
  const double floor_abs = 1e-4 * rms + 1e-30;
  int64_t bad = 0;
  double w = 0.0;
  for (size_t k = 0; k < ref.size(); ++k) {
    const double err = std::fabs(static_cast<double>(got[k]) - ref[k]);
    const double bound = std::ldexp(std::fabs(ref[k]), -8) + floor_abs;
    if (err > bound) ++bad;
    if (err / bound > w) w = err / bound;
  }
  if (worst != nullptr) *worst = w;
  return bad;
}

}  // namespace rotation_ref
