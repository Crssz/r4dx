// tests/kernels/w8a16_ref.hpp -- header-only CPU reference of r4dx_gemm_w8a16_nt_m64 (docs/w8a16.md 2-4,
// src/kernels/include/r4dx/kernels/w8a16_gemm.h). No HIP, no r4dx library: the CPU test
// (test_w8a16_ref_cpu) and the GPU test (test_w8a16_gemm) both include it.
//
// What it provides:
//   * the f16 / bf16 conversions the kernel performs (f16 RNE for the stored scale, f16 RTZ for the
//     activation, bf16 RNE for the output);
//   * the quantizer of docs/w8a16.md 2 (RTN, round half away from zero, against the STORED biased f16 scale);
//   * the byte packer and the unpacker of docs/w8a16.md 3.2. Both are written from the KERNEL's forward
//     index map -- (t, kb, lane, j, i) -> (row, k) -- not one from the inverse of the other;
//   * Gemm(): the kernel's result, in the kernel's rounding order:
//       1. a16[k]  = f16_rtz(bf16 A[m,k])                        (saturates to +-65504, subnormals truncated)
//       2. q[k]    = byte - 128
//       3. per k step of 16 (in k order), per group: c = f32( c + sum of the 16 exact products a16*q )
//          -- the WMMA. c starts at 0 on the group's first k step. ASSUMPTION: one rounding per WMMA (the
//          16 products and c summed exactly, then rounded to f32 RNE). This is the ONLY step whose order the
//          hardware does not pin down; everything below is bit-exact. The GPU test therefore gates on an
//          error bound and only REPORTS how many outputs match this reference bit for bit.
//       4. per K split (Ks = K/SK, in group order): acc = fmaf(g_acc, float(s16), acc), acc from 0
//       5. v = ((0 + acc_split0) + acc_split1) + ... in split order (f32)
//       6. v = v * 2^-12 (exact unless the result is f32-subnormal)
//       7. out = bf16_rne(v)
//   * GemmNaive(): fp64 dot of a16 with the dequantized weight float(s16) * 2^-12 * q -- the "true" answer
//     the error bound is measured against.
//
// FORMAT CONSTANTS. Lane W1 owns them in src/convert/include/r4dx_convert/w8a16_format.hpp. When that header
// is on the include path it is used and the local copies below are static_assert-ed against it; until then
// (this branch, before the W1 merge) the local block is the definition. MERGE NOTE: once w8a16_format.hpp is
// merged, the local block can be deleted in favour of the include.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// ---- LOCAL FORMAT CONSTANTS (docs/w8a16.md 2, 3.2; mirror of W1's w8a16_format.hpp) ---------------------
namespace w8a16_ref {
inline constexpr int kGroup64 = 64;
inline constexpr int kGroup128 = 128;
inline constexpr int kScaleLog2Bias = 12;
inline constexpr int kCodeMax = 127;
inline constexpr int kCodeOffset = 128;
inline constexpr int kTileRows = 16;
inline constexpr int kKBlock = 32;
inline constexpr int kMaxM = 64;
inline constexpr bool IsGroupSupported(int g) { return g == kGroup64 || g == kGroup128; }
}  // namespace w8a16_ref
#if defined(__has_include)
#if __has_include("r4dx_convert/w8a16_format.hpp")
#include "r4dx_convert/w8a16_format.hpp"
static_assert(w8a16_ref::kScaleLog2Bias == r4dx_convert::w8a16::kScaleLog2Bias, "w8a16 scale bias mismatch");
static_assert(w8a16_ref::kCodeMax == r4dx_convert::w8a16::kCodeMax, "w8a16 code max mismatch");
static_assert(w8a16_ref::kCodeOffset == r4dx_convert::w8a16::kCodeOffset, "w8a16 code offset mismatch");
static_assert(w8a16_ref::kTileRows == r4dx_convert::w8a16::kTileRows, "w8a16 tile rows mismatch");
static_assert(w8a16_ref::kKBlock == r4dx_convert::w8a16::kKBlock, "w8a16 K block mismatch");
static_assert(w8a16_ref::kGroup64 == r4dx_convert::w8a16::kGroup64 &&
                  w8a16_ref::kGroup128 == r4dx_convert::w8a16::kGroup128,
              "w8a16 group set mismatch");
#endif
#endif
// ---- end of the local block ------------------------------------------------------------------------------

namespace w8a16_ref {

inline uint32_t FBits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
}
inline float FFromBits(uint32_t u) {
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// ---- bf16 ----
inline float Bf16ToFloat(uint16_t b) { return FFromBits(static_cast<uint32_t>(b) << 16); }
inline uint16_t FloatToBf16Rne(float f) {
  const uint32_t u = FBits(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40u);  // quiet NaN
  return static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
inline float Bf16Round(float f) { return Bf16ToFloat(FloatToBf16Rne(f)); }

// ---- f16 ----
inline float F16ToFloat(uint16_t h) {
  const uint32_t s = (h >> 15) & 1u, e = (h >> 10) & 0x1fu, m = h & 0x3ffu;
  float v;
  if (e == 0)
    v = std::ldexp(static_cast<float>(m), -24);
  else if (e == 31)
    v = m ? NAN : INFINITY;
  else
    v = std::ldexp(static_cast<float>(m | 0x400u), static_cast<int>(e) - 25);
  return s ? -v : v;
}

// f32 -> f16 with round-to-nearest-even (rtz = false) or round-toward-zero (rtz = true). RTZ maps a finite
// overflow to the largest finite (0x7bff), the v_cvt_pk_rtz_f16_f32 behaviour; RNE maps it to inf.
inline uint16_t FloatToF16(float f, bool rtz) {
  const uint32_t u = FBits(f);
  const uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000u);
  const uint32_t a = u & 0x7fffffffu;
  if (a > 0x7f800000u) return static_cast<uint16_t>(sign | 0x7e00u);  // NaN
  if (a == 0x7f800000u) return static_cast<uint16_t>(sign | 0x7c00u);
  const double x = static_cast<double>(FFromBits(a));  // |f|, exact
  // Express |f| in units of the f16 quantum at its binade: subnormal quantum 2^-24, else 2^(e-10).
  int e2;
  (void)std::frexp(x, &e2);  // x = frac * 2^e2, frac in [0.5, 1)
  int qexp = e2 - 11;        // quantum exponent for a normal f16 (11-bit significand)
  if (qexp < -24) qexp = -24;
  const double scaled = std::ldexp(x, -qexp);  // exact (power-of-two scaling of a double)
  double r;
  if (rtz) {
    r = std::floor(scaled);
  } else {
    r = std::floor(scaled);
    const double frac = scaled - r;
    if (frac > 0.5 || (frac == 0.5 && std::fmod(r, 2.0) != 0.0)) r += 1.0;
  }
  const double v = std::ldexp(r, qexp);  // rounded magnitude
  if (v > 65504.0) return static_cast<uint16_t>(sign | (rtz ? 0x7bffu : 0x7c00u));
  if (v == 0.0) return sign;
  if (v < std::ldexp(1.0, -14))  // subnormal: mantissa = v / 2^-24
    return static_cast<uint16_t>(sign | static_cast<uint16_t>(std::ldexp(v, 24)));
  int ve;
  const double vf = std::frexp(v, &ve);  // v = vf * 2^ve, vf in [0.5, 1)
  const uint32_t exp16 = static_cast<uint32_t>(ve - 1 + 15);
  const uint32_t mant = static_cast<uint32_t>(std::ldexp(vf, 11)) - 0x400u;
  return static_cast<uint16_t>(sign | (exp16 << 10) | mant);
}
inline uint16_t FloatToF16Rne(float f) { return FloatToF16(f, false); }
inline uint16_t FloatToF16Rtz(float f) { return FloatToF16(f, true); }

// The kernel's activation conversion: bf16 bits -> f32 (exact) -> f16 RTZ -> value.
inline float ActivationF16(uint16_t bf16_bits) { return F16ToFloat(FloatToF16Rtz(Bf16ToFloat(bf16_bits))); }

// ---- quantizer (docs/w8a16.md 2) ----
inline float RoundHalfAwayFromZero(float x) { return std::round(x); }

struct Quantized {
  std::vector<int8_t> q;        // [N*K] codes -127..127, row-major
  std::vector<uint16_t> s16;    // [N*(K/G)] f16 bits of s*2^12, row-major (row, group)
  int64_t zero_groups = 0;
};

inline Quantized Quantize(const std::vector<float>& w, int N, int K, int G) {
  if (!IsGroupSupported(G) || K % G || N < 1 || w.size() != static_cast<size_t>(N) * K)
    throw std::runtime_error("w8a16_ref::Quantize: bad shape/group");
  Quantized out;
  const int ng = K / G;
  out.q.assign(static_cast<size_t>(N) * K, 0);
  out.s16.assign(static_cast<size_t>(N) * ng, 0);
  const float bias = std::ldexp(1.0f, kScaleLog2Bias), unbias = std::ldexp(1.0f, -kScaleLog2Bias);
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < ng; ++g) {
      const float* src = &w[static_cast<size_t>(n) * K + static_cast<size_t>(g) * G];
      float amax = 0.0f;
      for (int i = 0; i < G; ++i) amax = std::fmax(amax, std::fabs(src[i]));
      const float s = amax / 127.0f;
      const uint16_t s16 = FloatToF16Rne(s * bias);
      if ((s16 & 0x7c00u) == 0x7c00u)
        throw std::runtime_error("w8a16_ref::Quantize: scale overflows f16 (amax " + std::to_string(amax) + ")");
      out.s16[static_cast<size_t>(n) * ng + g] = s16;
      if ((s16 & 0x7fffu) == 0) {
        ++out.zero_groups;
        continue;  // codes stay 0
      }
      const float s_use = F16ToFloat(s16) * unbias;
      for (int i = 0; i < G; ++i) {
        float r = RoundHalfAwayFromZero(src[i] / s_use);
        r = std::fmin(std::fmax(r, -127.0f), 127.0f);
        out.q[static_cast<size_t>(n) * K + static_cast<size_t>(g) * G + i] = static_cast<int8_t>(r);
      }
    }
  return out;
}

// ---- the kernel's index map (docs/w8a16.md 3.2) ----
// Byte offset in wq of (t, kb, lane, j, i), and the (row, k) the kernel reads there.
inline size_t WqByteOffset(int K, int t, int kb, int lane, int j, int i) {
  return ((((static_cast<size_t>(t) * (K / kKBlock) + kb) * 32 + lane) * 4 + j) * 4) + i;
}
inline int WqRow(int t, int lane) { return kTileRows * t + (lane & 15); }
inline int WqK(int kb, int lane, int j, int i) { return 32 * kb + 16 * (j >> 1) + 8 * (j & 1) + 4 * (lane >> 4) + i; }
// u16 index of the scale of (row tile t, group g, row-in-tile r).
inline size_t WsIndex(int K, int G, int t, int g, int r) {
  return (static_cast<size_t>(t) * (K / G) + g) * 16 + r;
}

// Codes [N*K] (row-major) -> packed bytes, scattered through the forward map.
inline std::vector<uint8_t> PackWq(const std::vector<int8_t>& q, int N, int K) {
  if (N % kTileRows || K % kKBlock) throw std::runtime_error("w8a16_ref::PackWq: N % 16 or K % 32");
  std::vector<uint8_t> out(static_cast<size_t>(N) * K);
  for (int t = 0; t < N / 16; ++t)
    for (int kb = 0; kb < K / 32; ++kb)
      for (int l = 0; l < 32; ++l)
        for (int j = 0; j < 4; ++j)
          for (int i = 0; i < 4; ++i) {
            const int row = WqRow(t, l), k = WqK(kb, l, j, i);
            out[WqByteOffset(K, t, kb, l, j, i)] =
                static_cast<uint8_t>(q[static_cast<size_t>(row) * K + k] + kCodeOffset);
          }
  return out;
}
// Scales [N*(K/G)] row-major -> the kernel's u16 order.
inline std::vector<uint16_t> PackWs(const std::vector<uint16_t>& s16, int N, int K, int G) {
  const int ng = K / G;
  std::vector<uint16_t> out(static_cast<size_t>(N) * ng);
  for (int t = 0; t < N / 16; ++t)
    for (int g = 0; g < ng; ++g)
      for (int r = 0; r < 16; ++r) out[WsIndex(K, G, t, g, r)] = s16[static_cast<size_t>(16 * t + r) * ng + g];
  return out;
}

// Packed -> codes and scales of a set of weight rows (output columns), read the way the kernel reads them:
// for row r the kernel's lanes (r & 15) and (r & 15) + 16 of tile r/16 read, at every 32-K block kb, dword j
// byte i and treat it as (r, WqK(kb, lane, j, i)); the scale lane (r & 15) reads u16 WsIndex(K, G, t, g, r & 15).
// Unpacked row c holds weight row rows[c] (rows empty = all N rows in order).
struct Unpacked {
  int rows = 0;               // number of rows held
  std::vector<int> q;         // [rows*K] byte - 128
  std::vector<uint16_t> s16;  // [rows*(K/G)]
};
inline Unpacked UnpackRows(const std::vector<uint8_t>& wq, const std::vector<uint16_t>& ws, int N, int K, int G,
                           const std::vector<int>& rows = {}) {
  std::vector<int> all;
  const std::vector<int>* rs = &rows;
  if (rows.empty()) {
    all.resize(N);
    for (int n = 0; n < N; ++n) all[n] = n;
    rs = &all;
  }
  const int ng = K / G;
  Unpacked u;
  u.rows = static_cast<int>(rs->size());
  u.q.assign(static_cast<size_t>(u.rows) * K, 0);
  u.s16.assign(static_cast<size_t>(u.rows) * ng, 0);
  for (int c = 0; c < u.rows; ++c) {
    const int r = (*rs)[c], t = r / kTileRows, rl = r % kTileRows;
    if (r < 0 || r >= N) throw std::runtime_error("w8a16_ref::UnpackRows: row out of range");
    for (int kb = 0; kb < K / kKBlock; ++kb)
      for (int lh = 0; lh < 2; ++lh) {
        const int l = rl + 16 * lh;
        for (int j = 0; j < 4; ++j)
          for (int i = 0; i < 4; ++i)
            u.q[static_cast<size_t>(c) * K + WqK(kb, l, j, i)] =
                static_cast<int>(wq[WqByteOffset(K, t, kb, l, j, i)]) - kCodeOffset;
      }
    for (int g = 0; g < ng; ++g) u.s16[static_cast<size_t>(c) * ng + g] = ws[WsIndex(K, G, t, g, rl)];
  }
  return u;
}

// ---- the GEMM ----
inline void CheckLaunch(int M, int K, int G, int SK) {
  if (M < 1 || M > kMaxM || !IsGroupSupported(G) || SK < 1 || K < 1 || K % (SK * G))
    throw std::runtime_error("w8a16_ref::Gemm: illegal shape");
}

// One output element in the kernel's order (steps 1-7 of the header comment); returns v before the bf16
// store. a16: the row's f16 values as floats; q, s16: one weight row.
inline float Element(const float* a16, const int* q, const uint16_t* s16, int K, int G, int SK) {
  const int Ks = K / SK, gps = Ks / G;
  float v = 0.0f;
  for (int sp = 0; sp < SK; ++sp) {
    float acc = 0.0f;
    for (int gi = 0; gi < gps; ++gi) {
      const int g = sp * gps + gi;
      float gacc = 0.0f;
      for (int k0 = g * G; k0 < (g + 1) * G; k0 += 16) {  // one WMMA per 16-K step
        double sum = static_cast<double>(gacc);
        for (int k = k0; k < k0 + 16; ++k) sum += static_cast<double>(a16[k]) * static_cast<double>(q[k]);
        gacc = static_cast<float>(sum);
      }
      acc = std::fma(gacc, F16ToFloat(s16[g]), acc);
    }
    v = v + acc;
  }
  return v * std::ldexp(1.0f, -kScaleLog2Bias);
}

// a: [M*K] bf16 bits. Returns [M * w.rows] bf16 bits: out[m * w.rows + c] is the kernel's C[m, rows[c]].
inline std::vector<uint16_t> Gemm(const std::vector<uint16_t>& a, const Unpacked& w, int M, int K, int G, int SK) {
  CheckLaunch(M, K, G, SK);
  const int ng = K / G, R = w.rows;
  std::vector<uint16_t> out(static_cast<size_t>(M) * R, 0);
  std::vector<float> a16(K);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) a16[k] = ActivationF16(a[static_cast<size_t>(m) * K + k]);
    for (int c = 0; c < R; ++c)
      out[static_cast<size_t>(m) * R + c] = FloatToBf16Rne(
          Element(a16.data(), &w.q[static_cast<size_t>(c) * K], &w.s16[static_cast<size_t>(c) * ng], K, G, SK));
  }
  return out;
}

// fp64 truth over the f16-converted A and the dequantized weight, plus the two norms of the error bound
// (docs/w8a16.md 7.2: |c - ref| <= 2 bf16 ulp(|ref|) + 1e-6 * ||a_row|| * ||w_row||). Same [M * w.rows] layout.
struct NaiveOut {
  std::vector<double> c;       // [M * rows]
  std::vector<double> a_norm;  // [M]    ||a16 row||
  std::vector<double> w_norm;  // [rows] ||dequantized w row||
};
inline NaiveOut GemmNaive(const std::vector<uint16_t>& a, const Unpacked& w, int M, int K, int G) {
  const int ng = K / G, R = w.rows;
  NaiveOut o;
  o.c.assign(static_cast<size_t>(M) * R, 0.0);
  o.a_norm.assign(M, 0.0);
  o.w_norm.assign(R, 0.0);
  std::vector<double> a16(static_cast<size_t>(M) * K), wd(K);
  for (int m = 0; m < M; ++m) {
    double s = 0;
    for (int k = 0; k < K; ++k) {
      const double x = ActivationF16(a[static_cast<size_t>(m) * K + k]);
      a16[static_cast<size_t>(m) * K + k] = x;
      s += x * x;
    }
    o.a_norm[m] = std::sqrt(s);
  }
  const double unbias = std::ldexp(1.0, -kScaleLog2Bias);
  for (int c = 0; c < R; ++c) {
    double s = 0;
    for (int k = 0; k < K; ++k) {
      wd[k] = static_cast<double>(w.q[static_cast<size_t>(c) * K + k]) *
              F16ToFloat(w.s16[static_cast<size_t>(c) * ng + k / G]) * unbias;
      s += wd[k] * wd[k];
    }
    o.w_norm[c] = std::sqrt(s);
    for (int m = 0; m < M; ++m) {
      double acc = 0;
      const double* ar = &a16[static_cast<size_t>(m) * K];
      for (int k = 0; k < K; ++k) acc += ar[k] * wd[k];
      o.c[static_cast<size_t>(m) * R + c] = acc;
    }
  }
  return o;
}

// bf16 ulp of |x| (the spacing at x's binade; 2^-133 for zero / below the normal range).
inline double Bf16Ulp(double x) {
  x = std::fabs(x);
  if (x < std::ldexp(1.0, -126)) return std::ldexp(1.0, -133);
  int e;
  (void)std::frexp(x, &e);  // x in [2^(e-1), 2^e)
  return std::ldexp(1.0, e - 1 - 7);
}
// The design's bound for one element.
inline bool WithinBound(double got, double ref, double a_norm, double w_norm) {
  return std::fabs(got - ref) <= 2.0 * Bf16Ulp(ref) + 1e-6 * a_norm * w_norm;
}

}  // namespace w8a16_ref
