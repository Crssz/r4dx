// r4dx_convert::mxfp8_format -- the on-disk constants and the CPU reference codec of the mxfp8 layout
// (docs/inkling2-plan.md 4.1).
//
// Header-only and dependency-free (no r4dx headers), like w8a16_format.hpp, so the converter, the loader
// (src/model) and the kernel tests all include it. It is the single source of truth for the literals the
// container spells, and for the exact e4m3 / E8M0 rounding the converter writes and the kernels decode.
//
// Layout of one linear W [N, K] (K % 32 == 0):
//   <base>.mxfp8.wq  u8 {N*K}     OCP e4m3fn codes, in the SAME packed-fragment order as w8a16's wq
//                                 (16 rows x 32 K blocks, w8a16::kMetaFragmentPermutation), so one packed
//                                 block holds exactly one scale group per row;
//   <base>.mxfp8.ws  u8 {N*K/32}  E8M0 exponents (value 2^(byte - 127)), in the same element order as
//                                 w8a16's ws (one entry per (row, 32-K group)), one byte per entry.
// Element value: DecodeE4m3(wq) * 2^(ws - 127). No code is ever NaN (0x7F / 0xFF) and no scale is 255.
#pragma once

#include <cmath>
#include <cstdint>

namespace r4dx_convert {
namespace mxfp8 {

inline constexpr int kGroup = 32;           // K per (row, scale): the OCP MX block size
inline constexpr float kE4m3Max = 448.0f;   // largest finite e4m3fn magnitude (code 0x7E)
inline constexpr int kScaleBias = 127;      // E8M0: value = 2^(byte - 127)
inline constexpr int kScaleExpMin = -127;   // byte 0
inline constexpr int kScaleExpMax = 127;    // byte 254 (255 is NaN, never written)

// Tensor name suffixes appended to a linear's base name.
inline constexpr const char* kWqSuffix = ".mxfp8.wq";  // u8 {N*K}
inline constexpr const char* kWsSuffix = ".mxfp8.ws";  // u8 {N*K/32}

// __metadata__.quant.mxfp8 (version 1).
inline constexpr const char* kMetaQuantKey = "quant";
inline constexpr const char* kMetaBlockKey = "mxfp8";
inline constexpr int kMetaVersion = 1;
inline constexpr const char* kMetaElem = "e4m3fn";
inline constexpr const char* kMetaScale = "e8m0";
inline constexpr const char* kMetaFragmentPermutation = "wmma16x16x16_f16_b_lane16_k32block";  // = w8a16's

// LayoutSetId token: "mxfp8.g32".
inline constexpr const char* kLayoutId = "mxfp8.g32";

// e4m3fn code -> float (exact). Subnormals: m * 2^-9. 0x7F / 0xFF (NaN) decode to NaN.
inline float DecodeE4m3(uint8_t b) {
  const int s = b >> 7, e = (b >> 3) & 0xF, m = b & 7;
  float v;
  if (e == 15 && m == 7) {
    v = std::nanf("");
  } else if (e == 0) {
    v = std::ldexp(static_cast<float>(m), -9);
  } else {
    v = std::ldexp(static_cast<float>(8 + m), e - 10);  // (1 + m/8) * 2^(e-7)
  }
  return s ? -v : v;
}

// float -> e4m3fn code, round-to-nearest-even, saturating at +-448 (no NaN/inf is ever produced;
// a NaN input is the caller's bug and maps to 0). -0 maps to 0x80.
inline uint8_t EncodeE4m3(float x) {
  if (std::isnan(x)) return 0;
  const uint8_t sign = std::signbit(x) ? 0x80 : 0x00;
  const double a = std::fabs(static_cast<double>(x));
  if (a >= 448.0) return sign | 0x7E;
  if (a < std::ldexp(1.0, -6)) {  // subnormal range: code = round(a / 2^-9), 8 rounds up to the first normal
    const double q = std::nearbyint(std::ldexp(a, 9));
    return sign | static_cast<uint8_t>(q);
  }
  int e2 = 0;
  (void)std::frexp(a, &e2);  // a = f * 2^e2, f in [0.5, 1)
  int e = e2 - 1;            // a = (1 + mant) * 2^e
  double m = std::nearbyint(std::ldexp(a, 3 - e) - 8.0);
  if (m >= 8.0) {
    m = 0.0;
    ++e;
  }
  const int code = ((e + 7) << 3) | static_cast<int>(m);
  if (code > 0x7E) return sign | 0x7E;
  return sign | static_cast<uint8_t>(code);
}

// E8M0 exponent for a group whose largest magnitude is `amax`: the smallest e with 448 * 2^e >= amax
// (so no element is clipped), clamped to [-127, 127]; amax == 0 -> -127.
inline int ScaleExponent(float amax) {
  if (!(amax > 0.0f)) return kScaleExpMin;
  int e = static_cast<int>(std::ceil(std::log2(static_cast<double>(amax) / 448.0)));
  while (e < kScaleExpMax && std::ldexp(448.0, e) < amax) ++e;
  while (e > kScaleExpMin && std::ldexp(448.0, e - 1) >= amax) --e;
  if (e < kScaleExpMin) e = kScaleExpMin;
  if (e > kScaleExpMax) e = kScaleExpMax;
  return e;
}

inline uint8_t ScaleByte(int exponent) { return static_cast<uint8_t>(exponent + kScaleBias); }
inline float ScaleValue(uint8_t byte) { return std::ldexp(1.0f, static_cast<int>(byte) - kScaleBias); }

// Encode one group of `n` (<= 32) floats: writes n codes, returns the scale byte.
inline uint8_t EncodeGroup(const float* v, int n, uint8_t* codes) {
  float amax = 0.0f;
  for (int i = 0; i < n; ++i) amax = std::fmax(amax, std::fabs(v[i]));
  const int e = ScaleExponent(amax);
  for (int i = 0; i < n; ++i) codes[i] = EncodeE4m3(static_cast<float>(std::ldexp(static_cast<double>(v[i]), -e)));
  return ScaleByte(e);
}

}  // namespace mxfp8
}  // namespace r4dx_convert
