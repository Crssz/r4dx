// tests/kernels/trellis_ref.hpp -- CPU reference decode of the trellis (EXL3 / QTIP "mul1") weight
// format, for the libr4d trellis kernels' tests (docs/trellis-kernel.md 2.1, 4.2, 6).
//
// A C++ transcription of tools/reference/trellis_quant.py's decode, function for function:
//   Mul1CodebookF16   codebook_np("mul1"): x = s * 0x83DCD12D, f16((1024 + bytesum(x)) * f16(0x1EEE)
//                     + f16(0xC931)) with one rounding -- the product and the sum are exact in
//                     float (11-bit by 11-bit, and |result| * 2^19 < 2^21), so FloatToF16 of the
//                     float result IS the single f16 rounding;
//   TrellisState      unpack_states: state(p) = the 16 ring bits ending at bit KB*(p+1), tail-biting;
//   TensorCorePerm    tensor_core_perm: position 8t + j -> row-major tile element (r*16 + c,
//                     r = k inside the tile, c = n);
//   DecodeOracle      decode_words: oracle-layout words [K/16][N/16][8KB] -> Q[K][N] f16 bits;
//   ToPairGrid        trellis_golden.py's to_pair_grid: word w of tile (tn, tk) at
//                     (((tn>>1)*(K/16) + tk)*2 + (tn&1))*8KB + w;
//   DecodePairGrid    the same decode, read from the pair grid.
// and the linear around the decode (trellis_golden.py's transform_input / linear_ref /
// linear_ref_from_a, docs/trellis-kernel.md 4.5 and 4.8):
//   FwhtLdsF32        r4dx's FwhtLds (hadamard_device.h) on one 128-block, the same fp32 ops in the
//                     same order -- so it is the bits of both GPU butterflies;
//   TransformInput    r4dx_trellis_input_bf16's arithmetic, bit for bit: f16_rn(FWHT128_fp32(
//                     fp32(x) * suh) * float(2^s / sqrt(128)));
//   LinearFromA       fp64 2^-s / sqrt(128) * svh[n] * FWHT128_n(A_p(n) @ Q): the per-element
//                     reference of r4d_gemm_trellis_nt_m64 (from the f16 A it was given);
//   LinearExact       fp64 x @ W_hat from the exact x (the Frobenius reference);
//   Bf16Ulp           the bf16 ulp at |v|.
// test_trellis_decode checks the decode against the committed goldens (tests/kernels/golden/trellis,
// from the Python) bit for bit before trusting it on shapes the goldens do not cover;
// test_trellis_input and test_trellis_gemm do the same for the transform and the linear.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "r4dx/core/dtype.hpp"

namespace trellis_ref {

inline const std::vector<uint16_t>& Mul1CodebookF16() {
  static const std::vector<uint16_t> cb = [] {
    const double kinv = r4dx::core::F16ToFloat(0x1EEE), kbias = r4dx::core::F16ToFloat(0xC931);
    std::vector<uint16_t> out(1u << 16);
    for (uint32_t s = 0; s < (1u << 16); ++s) {
      const uint32_t x = s * 0x83DCD12Du;
      const uint32_t bs = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
      const double v = (1024.0 + bs) * kinv + kbias;   // exact (see the header comment)
      out[s] = r4dx::core::FloatToF16(static_cast<float>(v));
    }
    return out;
  }();
  return cb;
}

// state(p) of one tile's ring `w` (8*KB words), integer KB.
inline uint32_t TrellisState(const uint32_t* w, int KB, int p) {
  const int R = 256 * KB, nw = 8 * KB;
  const int start = ((KB * (p + 1) - 16) % R + R) % R;
  const int i0 = start / 32, off = start % 32, i1 = (i0 + 1) % nw;
  const uint64_t w0 = w[i0], w1 = w[i1];
  return static_cast<uint32_t>((((w0 << off) & 0xFFFFFFFFull) >> 16) | (w1 >> (48 - off))) & 0xFFFFu;
}

// Position p -> row-major tile element r*16 + c.
inline int TensorCorePerm(int p) {
  const int t = p / 8, j = p % 8;
  const int r0 = (t % 4) * 2, c0 = t / 4;
  const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
  return j < 4 ? rows[j] * 16 + c0 : rows[j - 4] * 16 + c0 + 8;
}

// One tile's ring -> its 256 f16 values at (r, c) of out[(k0 + r) * N + n0 + c].
inline void DecodeTile(const uint32_t* w, int KB, uint16_t* out, int64_t N, int64_t k0, int64_t n0) {
  const std::vector<uint16_t>& cb = Mul1CodebookF16();
  for (int p = 0; p < 256; ++p) {
    const int e = TensorCorePerm(p);
    out[(k0 + e / 16) * N + n0 + e % 16] = cb[TrellisState(w, KB, p)];
  }
}

inline size_t PairGridIndex(int64_t K, int KB, int64_t tn, int64_t tk) {
  return static_cast<size_t>((((tn >> 1) * (K / 16) + tk) * 2 + (tn & 1)) * 8 * KB);
}

// Oracle layout [K/16][N/16][8KB] -> Q[K][N] f16 bits.
inline std::vector<uint16_t> DecodeOracle(const std::vector<uint32_t>& words, int64_t K, int64_t N,
                                          int KB) {
  std::vector<uint16_t> q(static_cast<size_t>(K * N));
  const int nw = 8 * KB;
  for (int64_t tk = 0; tk < K / 16; ++tk)
    for (int64_t tn = 0; tn < N / 16; ++tn)
      DecodeTile(&words[static_cast<size_t>((tk * (N / 16) + tn) * nw)], KB, q.data(), N, 16 * tk,
                 16 * tn);
  return q;
}

inline std::vector<uint32_t> ToPairGrid(const std::vector<uint32_t>& words, int64_t K, int64_t N,
                                        int KB) {
  std::vector<uint32_t> grid(words.size());
  const int nw = 8 * KB;
  for (int64_t tk = 0; tk < K / 16; ++tk)
    for (int64_t tn = 0; tn < N / 16; ++tn)
      for (int w = 0; w < nw; ++w)
        grid[PairGridIndex(K, KB, tn, tk) + w] = words[static_cast<size_t>((tk * (N / 16) + tn) * nw + w)];
  return grid;
}

inline std::vector<uint16_t> DecodePairGrid(const std::vector<uint32_t>& grid, int64_t K, int64_t N,
                                            int KB) {
  std::vector<uint16_t> q(static_cast<size_t>(K * N));
  for (int64_t tk = 0; tk < K / 16; ++tk)
    for (int64_t tn = 0; tn < N / 16; ++tn)
      DecodeTile(&grid[PairGridIndex(K, KB, tn, tk)], KB, q.data(), N, 16 * tk, 16 * tn);
  return q;
}

// ---- the linear around the decode ---------------------------------------------------------------

// FwhtLds on one 128-float block: stages lg = 0..6, pair (i, i + 2^lg) -> (a + b, a - b), fp32.
inline void FwhtLdsF32(float* v) {
#pragma clang fp contract(off)
  for (int lg = 0; lg < 7; ++lg) {
    const int h = 1 << lg;
    for (int p = 0; p < 64; ++p) {
      const int i = ((p >> lg) << (lg + 1)) | (p & (h - 1));
      const float a = v[i], b = v[i + h];
      v[i] = a + b;
      v[i + h] = a - b;
    }
  }
}

// The same transform in fp64 (any order: the references only need it exact to ~1e-16).
inline void Fwht128F64(double* v) {
  for (int h = 1; h < 128; h <<= 1)
    for (int i = 0; i < 128; i += 2 * h)
      for (int j = i; j < i + h; ++j) {
        const double a = v[j], b = v[j + h];
        v[j] = a + b;
        v[j + h] = a - b;
      }
}

// float(2^s / sqrt(128)), rounded once from double (trellis_golden.py's transform_scale).
inline float TransformScale(int prescale_log2) {
  return static_cast<float>(std::ldexp(1.0, prescale_log2) / std::sqrt(128.0));
}

// x: bf16 bits [M][K]; suh: fp32 [K] (fp16 values widened) -> A f16 bits [M][K].
inline std::vector<uint16_t> TransformInput(const std::vector<uint16_t>& x, int64_t M, int64_t K,
                                            const float* suh, int prescale_log2) {
#pragma clang fp contract(off)
  const float scale = TransformScale(prescale_log2);
  std::vector<uint16_t> a(static_cast<size_t>(M * K));
  float v[128];
  for (int64_t m = 0; m < M; ++m)
    for (int64_t k0 = 0; k0 < K; k0 += 128) {
      for (int i = 0; i < 128; ++i) {
        const float xv = r4dx::core::Bf16ToFloat(x[static_cast<size_t>(m * K + k0 + i)]);
        v[i] = xv * suh[k0 + i];
      }
      FwhtLdsF32(v);
      for (int i = 0; i < 128; ++i) {
        const float s = v[i] * scale;
        a[static_cast<size_t>(m * K + k0 + i)] = r4dx::core::FloatToF16(s);
      }
    }
  return a;
}

// Column n reads part p(n) = (n >= n_split). a[p]: f16 bits [M][K]; q: f16 bits [K][N]; svh: fp32
// [N]. Returns fp64 [M][N] = 2^-s / sqrt(128) * svh[n] * FWHT128_n(sum_k A_p(n)[m][k] Q[k][n]).
inline std::vector<double> LinearFromA(const std::vector<const uint16_t*>& a, int64_t M, int64_t K,
                                       int64_t N, int64_t n_split, const std::vector<uint16_t>& q,
                                       const float* svh, int prescale_log2) {
  std::vector<float> qf(q.size());
  for (size_t i = 0; i < q.size(); ++i) qf[i] = r4dx::core::F16ToFloat(q[i]);
  std::vector<double> y(static_cast<size_t>(M * N), 0.0);
  std::vector<double> row(static_cast<size_t>(N));
  for (int64_t m = 0; m < M; ++m) {
    std::fill(row.begin(), row.end(), 0.0);
    for (size_t p = 0; p < a.size(); ++p) {
      const int64_t c0 = p == 0 ? 0 : n_split, c1 = (p == 0 && a.size() > 1) ? n_split : N;
      for (int64_t k = 0; k < K; ++k) {
        const double av = r4dx::core::F16ToFloat(a[p][m * K + k]);
        if (av == 0.0) continue;
        const float* qr = &qf[static_cast<size_t>(k * N)];
        for (int64_t n = c0; n < c1; ++n) row[n] += av * qr[n];
      }
    }
    const double out_scale = std::ldexp(1.0, -prescale_log2) / std::sqrt(128.0);
    for (int64_t n0 = 0; n0 < N; n0 += 128) Fwht128F64(&row[n0]);
    for (int64_t n = 0; n < N; ++n) y[m * N + n] = row[n] * out_scale * svh[n];
  }
  return y;
}

// x: bf16 bits [M][K]; suh[p]: fp32 [K]. fp64 x @ W_hat, W_hat = diag(suh_p) H Q H diag(svh) with H
// the normalized 128-point Hadamard (the linear the weights stand for, before any f16 rounding of A).
inline std::vector<double> LinearExact(const std::vector<uint16_t>& x, int64_t M, int64_t K, int64_t N,
                                       int64_t n_split, const std::vector<const float*>& suh,
                                       const std::vector<uint16_t>& q, const float* svh) {
  const double r = 1.0 / std::sqrt(128.0);
  std::vector<float> qf(q.size());
  for (size_t i = 0; i < q.size(); ++i) qf[i] = r4dx::core::F16ToFloat(q[i]);
  std::vector<double> y(static_cast<size_t>(M * N), 0.0);
  std::vector<double> t(static_cast<size_t>(K)), row(static_cast<size_t>(N));
  for (int64_t m = 0; m < M; ++m) {
    std::fill(row.begin(), row.end(), 0.0);
    for (size_t p = 0; p < suh.size(); ++p) {
      const int64_t c0 = p == 0 ? 0 : n_split, c1 = (p == 0 && suh.size() > 1) ? n_split : N;
      for (int64_t k = 0; k < K; ++k)
        t[k] = static_cast<double>(r4dx::core::Bf16ToFloat(x[m * K + k])) * suh[p][k];
      for (int64_t k0 = 0; k0 < K; k0 += 128) Fwht128F64(&t[k0]);
      for (int64_t k = 0; k < K; ++k) {
        const double tv = t[k] * r;
        if (tv == 0.0) continue;
        const float* qr = &qf[static_cast<size_t>(k * N)];
        for (int64_t n = c0; n < c1; ++n) row[n] += tv * qr[n];
      }
    }
    for (int64_t n0 = 0; n0 < N; n0 += 128) Fwht128F64(&row[n0]);
    for (int64_t n = 0; n < N; ++n) y[m * N + n] = row[n] * r * svh[n];
  }
  return y;
}

// r4d_gemm_trellis_nt_m64's epilogue applied to the fp32 sums s [M][N] that its _raw entry returns:
// per row and 128-column group FwhtLdsF32, then bf16_rn((y * svh[n]) * out_scale) -- two fp32
// products in that order, never contracted, and one bf16 rounding (docs/trellis-kernel.md 4.5).
inline std::vector<uint16_t> EpilogueFromRaw(const std::vector<float>& s, int64_t M, int64_t N,
                                             const float* svh, float out_scale) {
#pragma clang fp contract(off)
  std::vector<uint16_t> c(static_cast<size_t>(M * N));
  float v[128];
  for (int64_t m = 0; m < M; ++m)
    for (int64_t n0 = 0; n0 < N; n0 += 128) {
      for (int i = 0; i < 128; ++i) v[i] = s[static_cast<size_t>(m * N + n0 + i)];
      FwhtLdsF32(v);
      for (int i = 0; i < 128; ++i) {
        const float y = v[i] * svh[n0 + i];
        const float z = y * out_scale;
        c[static_cast<size_t>(m * N + n0 + i)] = r4dx::core::FloatToBf16(z);
      }
    }
  return c;
}

inline double Bf16Ulp(double v) {
  const double a = std::fabs(v) > std::ldexp(1.0, -126) ? std::fabs(v) : std::ldexp(1.0, -126);
  return std::ldexp(1.0, static_cast<int>(std::floor(std::log2(a))) - 7);
}

}  // namespace trellis_ref
