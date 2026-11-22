// tests/kernels/mxfp8_ref.hpp -- header-only CPU reference of r4dx_gemm_mxfp8_nt (src/kernels/include/r4dx/kernels/
// mxfp8_gemm.h, docs/inkling2-plan.md 4.1). No HIP. The format constants and the e4m3 / E8M0 codec come from the
// fixed src/convert/include/r4dx_convert/mxfp8_format.hpp; the bf16 / f16 conversions and the packed fragment index
// map are w8a16_ref.hpp's (the mxfp8 wq uses w8a16's byte order exactly).
//
// What it provides:
//   * PackWq / PackWs: row-major codes [N*K] and scale bytes [N*(K/32)] -> the kernel's order, scattered through the
//     forward map (t, kb, lane, j, i) -> (row, k); UnpackRows walks the kernel's lanes back;
//   * DeviceDecode(b): the kernel's in-register decode (perm into the high byte, shift / mask, * 2^8 in f16),
//     emulated bit for bit, which the CPU test proves equal to mxfp8::DecodeE4m3 for all 254 non-NaN codes;
//   * Gemm(): the kernel's result in the kernel's rounding order:
//       a16[k]   = f16_rtz(bf16 A[m,k]);  w16[k] = DecodeE4m3(code)  (exact)
//       per group g (32 K, two WMMA k steps of 16): gacc = f32(gacc + sum of the 16 exact products), from 0
//         -- ASSUMPTION: one rounding per WMMA; exact (order-free) whenever the true partial sums fit 24 bits, which
//         the GPU test's "exact class" inputs guarantee, so there the comparison is bit for bit;
//       per split: acc = fmaf(gacc, 2^(s - 127), acc) in group order, from 0 (f32 subnormals honoured)
//       v = ((0 + acc_0) + acc_1) + ... in split order;  out = bf16_rne(v)
//   * GemmNaive(): fp64 dot of a16 with the dequantized weights, for the error bound of general inputs.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "r4dx_convert/mxfp8_format.hpp"
#include "w8a16_ref.hpp"

namespace mxfp8_ref {

namespace fmt = r4dx_convert::mxfp8;
using w8a16_ref::WqByteOffset;
using w8a16_ref::WqK;
using w8a16_ref::WqRow;

inline constexpr int kGroup = fmt::kGroup;  // 32
inline constexpr int kMaxM = 256;
static_assert(kGroup == w8a16_ref::kKBlock, "an mxfp8 scale group is one packed 32-K block");

inline size_t WsIndex(int K, int t, int g, int r) { return (static_cast<size_t>(t) * (K / kGroup) + g) * 16 + r; }

inline std::vector<uint8_t> PackWq(const std::vector<uint8_t>& codes, int N, int K) {
  if (N % 16 || K % kGroup) throw std::runtime_error("mxfp8_ref::PackWq: N % 16 or K % 32");
  std::vector<uint8_t> out(static_cast<size_t>(N) * K);
  for (int t = 0; t < N / 16; ++t)
    for (int kb = 0; kb < K / 32; ++kb)
      for (int l = 0; l < 32; ++l)
        for (int j = 0; j < 4; ++j)
          for (int i = 0; i < 4; ++i)
            out[WqByteOffset(K, t, kb, l, j, i)] = codes[static_cast<size_t>(WqRow(t, l)) * K + WqK(kb, l, j, i)];
  return out;
}

inline std::vector<uint8_t> PackWs(const std::vector<uint8_t>& scales, int N, int K) {
  const int ng = K / kGroup;
  std::vector<uint8_t> out(static_cast<size_t>(N) * ng);
  for (int t = 0; t < N / 16; ++t)
    for (int g = 0; g < ng; ++g)
      for (int r = 0; r < 16; ++r) out[WsIndex(K, t, g, r)] = scales[static_cast<size_t>(16 * t + r) * ng + g];
  return out;
}

// Rows `rows` (all when empty) read the way the kernel reads them: lanes (r & 15) and (r & 15) + 16 of tile r / 16.
struct Unpacked {
  int rows = 0;
  std::vector<uint8_t> code;   // [rows*K]
  std::vector<uint8_t> scale;  // [rows*(K/32)]
};
inline Unpacked UnpackRows(const std::vector<uint8_t>& wq, const std::vector<uint8_t>& ws, int N, int K,
                           const std::vector<int>& rows = {}) {
  std::vector<int> all;
  const std::vector<int>* rs = &rows;
  if (rows.empty()) {
    all.resize(N);
    for (int n = 0; n < N; ++n) all[n] = n;
    rs = &all;
  }
  const int ng = K / kGroup;
  Unpacked u;
  u.rows = static_cast<int>(rs->size());
  u.code.assign(static_cast<size_t>(u.rows) * K, 0);
  u.scale.assign(static_cast<size_t>(u.rows) * ng, 0);
  for (int c = 0; c < u.rows; ++c) {
    const int r = (*rs)[c], t = r / 16, rl = r % 16;
    if (r < 0 || r >= N) throw std::runtime_error("mxfp8_ref::UnpackRows: row out of range");
    for (int kb = 0; kb < K / 32; ++kb)
      for (int lh = 0; lh < 2; ++lh) {
        const int l = rl + 16 * lh;
        for (int j = 0; j < 4; ++j)
          for (int i = 0; i < 4; ++i) u.code[static_cast<size_t>(c) * K + WqK(kb, l, j, i)] = wq[WqByteOffset(K, t, kb, l, j, i)];
      }
    for (int g = 0; g < ng; ++g) u.scale[static_cast<size_t>(c) * ng + g] = ws[WsIndex(K, t, g, rl)];
  }
  return u;
}

// The kernel's decode of one code: x = b << 8 (the byte in the high half of a 16-bit lane), f16 bits
// ((x >> 1) & 0x3F80) | (x & 0x8000) = value * 2^-8, then the f16 multiply by 2^8 (exact). Returns the f16 bits.
inline uint16_t DeviceDecodeF16Bits(uint8_t b) {
  const uint32_t x = static_cast<uint32_t>(b) << 8;
  const uint16_t scaled = static_cast<uint16_t>(((x >> 1) & 0x3F80u) | (x & 0x8000u));
  const float v = w8a16_ref::F16ToFloat(scaled) * 256.0f;  // exact: a power-of-two scaling of a value in range
  return w8a16_ref::FloatToF16Rne(v);
}
inline float DeviceDecode(uint8_t b) { return w8a16_ref::F16ToFloat(DeviceDecodeF16Bits(b)); }

// The kernel's E8M0 value: 2^(b - 127) as f32, b == 0 -> 2^-127 (subnormal).
inline float ScaleValue(uint8_t b) { return std::ldexp(1.0f, static_cast<int>(b) - 127); }

inline void CheckLaunch(int M, int K, int SK) {
  if (M < 1 || M > kMaxM || SK < 1 || K < 1 || K % (SK * kGroup)) throw std::runtime_error("mxfp8_ref::Gemm: illegal shape");
}

inline float Element(const float* a16, const float* w16, const uint8_t* sc, int K, int SK) {
  const int Ks = K / SK, gps = Ks / kGroup;
  float v = 0.0f;
  for (int sp = 0; sp < SK; ++sp) {
    float acc = 0.0f;
    for (int gi = 0; gi < gps; ++gi) {
      const int g = sp * gps + gi;
      float gacc = 0.0f;
      for (int k0 = g * kGroup; k0 < (g + 1) * kGroup; k0 += 16) {
        double sum = static_cast<double>(gacc);
        for (int k = k0; k < k0 + 16; ++k) sum += static_cast<double>(a16[k]) * static_cast<double>(w16[k]);
        gacc = static_cast<float>(sum);
      }
      acc = std::fma(gacc, ScaleValue(sc[g]), acc);
    }
    v = v + acc;
  }
  return v;
}

// a: [M*K] bf16 bits. Returns [M * w.rows] bf16 bits: out[m * w.rows + c] = the kernel's C[m, rows[c]].
inline std::vector<uint16_t> Gemm(const std::vector<uint16_t>& a, const Unpacked& w, int M, int K, int SK) {
  CheckLaunch(M, K, SK);
  const int ng = K / kGroup, R = w.rows;
  std::vector<uint16_t> out(static_cast<size_t>(M) * R, 0);
  std::vector<float> a16(K), w16(static_cast<size_t>(R) * K);
  for (size_t i = 0; i < w16.size(); ++i) w16[i] = fmt::DecodeE4m3(w.code[i]);
  for (int m = 0; m < M; ++m) {
    for (int k = 0; k < K; ++k) a16[k] = w8a16_ref::ActivationF16(a[static_cast<size_t>(m) * K + k]);
    for (int c = 0; c < R; ++c)
      out[static_cast<size_t>(m) * R + c] = w8a16_ref::FloatToBf16Rne(
          Element(a16.data(), &w16[static_cast<size_t>(c) * K], &w.scale[static_cast<size_t>(c) * ng], K, SK));
  }
  return out;
}

struct NaiveOut {
  std::vector<double> c, a_norm, w_norm;
};
inline NaiveOut GemmNaive(const std::vector<uint16_t>& a, const Unpacked& w, int M, int K) {
  const int ng = K / kGroup, R = w.rows;
  NaiveOut o;
  o.c.assign(static_cast<size_t>(M) * R, 0.0);
  o.a_norm.assign(M, 0.0);
  o.w_norm.assign(R, 0.0);
  std::vector<double> a16(static_cast<size_t>(M) * K), wd(K);
  for (int m = 0; m < M; ++m) {
    double s = 0;
    for (int k = 0; k < K; ++k) {
      const double x = w8a16_ref::ActivationF16(a[static_cast<size_t>(m) * K + k]);
      a16[static_cast<size_t>(m) * K + k] = x;
      s += x * x;
    }
    o.a_norm[m] = std::sqrt(s);
  }
  for (int c = 0; c < R; ++c) {
    double s = 0;
    for (int k = 0; k < K; ++k) {
      wd[k] = static_cast<double>(fmt::DecodeE4m3(w.code[static_cast<size_t>(c) * K + k])) *
              std::ldexp(1.0, static_cast<int>(w.scale[static_cast<size_t>(c) * ng + k / kGroup]) - 127);
      s += wd[k] * wd[k];
    }
    o.w_norm[c] = std::sqrt(s);
    for (int m = 0; m < M; ++m) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += a16[static_cast<size_t>(m) * K + k] * wd[k];
      o.c[static_cast<size_t>(m) * R + c] = acc;
    }
  }
  return o;
}

// |c - ref| <= 2 bf16 ulp(|ref|) + 1e-6 * ||a|| * ||w|| (docs/w8a16.md 7.2's bound).
inline bool WithinBound(double got, double ref, double a_norm, double w_norm) {
  return w8a16_ref::WithinBound(got, ref, a_norm, w_norm);
}

}  // namespace mxfp8_ref
