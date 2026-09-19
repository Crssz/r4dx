// r4dx::core dtype: the element types r4dx and r4d pass around, plus host<->fp32 conversion
// helpers. Every function is tagged __host__ __device__ so the exact same bit-for-bit math runs
// on the CPU (host test setup / CPU reference code) and inside r4dx's own HIP kernels
// (src/kernels/src/r4dx_kernels.hip #includes this header) -- there is only one implementation to
// keep in sync, not two that can silently drift.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace r4dx::core {

// std::memcpy is host-only (HIP device code cannot call it -- "reference to __host__ function
// 'memcpy' in __host__ __device__ function"), so every bit reinterpret below goes through this
// union-based punning helper instead, which hipcc compiles fine for both host and device.
inline __host__ __device__ uint32_t BitsOfF32(float f) {
  union {
    float f;
    uint32_t u;
  } c;
  c.f = f;
  return c.u;
}
inline __host__ __device__ float F32FromBits(uint32_t bits) {
  union {
    uint32_t u;
    float f;
  } c;
  c.u = bits;
  return c.f;
}

// Element types that flow through r4dx tensors / the r4d C ABI. `Dtype::kU8` is the generic
// "opaque packed byte" dtype used for fp8 e4m3, packed int4, etc. where the byte layout is
// interpreted by the consumer rather than by this enum.
enum class Dtype {
  kBF16,
  kF16,
  kF32,
  kI32,
  kI8,
  kU8,
  kFP8E4M3,
};

inline int64_t DtypeSize(Dtype dt) {
  switch (dt) {
    case Dtype::kBF16:
    case Dtype::kF16:
      return 2;
    case Dtype::kF32:
    case Dtype::kI32:
      return 4;
    case Dtype::kI8:
    case Dtype::kU8:
    case Dtype::kFP8E4M3:
      return 1;
  }
  return 0;
}

// ---- bf16: top 16 bits of an IEEE-754 fp32, round-to-nearest-even -------------------------
inline __host__ __device__ uint16_t FloatToBf16(float f) {
  uint32_t bits = BitsOfF32(f);
  // NaN must stay NaN after truncation (a naive round can turn a NaN's mantissa to all-zero,
  // which truncates to +/-inf).
  if ((bits & 0x7fffffffu) > 0x7f800000u) {
    return static_cast<uint16_t>((bits >> 16) | 0x0040u);  // quiet NaN, sign preserved
  }
  uint32_t rounded = bits + 0x7fffu + ((bits >> 16) & 1u);
  return static_cast<uint16_t>(rounded >> 16);
}

inline __host__ __device__ float Bf16ToFloat(uint16_t h) {
  uint32_t bits = static_cast<uint32_t>(h) << 16;
  return F32FromBits(bits);
}

// ---- f16 (IEEE binary16), round-to-nearest-even --------------------------------------------
inline __host__ __device__ uint16_t FloatToF16(float f) {
  uint32_t bits = BitsOfF32(f);
  const uint32_t sign = (bits >> 16) & 0x8000u;
  int32_t exp = static_cast<int32_t>((bits >> 23) & 0xffu) - 127 + 15;
  uint32_t mant = bits & 0x7fffffu;

  if (((bits >> 23) & 0xffu) == 0xffu) {
    // Inf / NaN.
    return static_cast<uint16_t>(sign | 0x7c00u | (mant ? 0x0200u : 0u));
  }
  if (exp >= 0x1f) {
    return static_cast<uint16_t>(sign | 0x7c00u);  // overflow -> inf
  }
  if (exp <= 0) {
    if (exp < -10) return static_cast<uint16_t>(sign);  // underflow -> 0
    // Subnormal f16: shift the implicit 1 in, then round to nearest even.
    mant |= 0x800000u;
    int32_t shift = 14 - exp;
    uint32_t rounded = mant + ((1u << (shift - 1)) - 1u) + ((mant >> shift) & 1u);
    uint32_t half = rounded >> shift;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t rounded_mant = mant + 0xfffu + ((mant >> 13) & 1u);
  if (rounded_mant & 0x800000u) {
    rounded_mant = 0;
    exp += 1;
    if (exp >= 0x1f) return static_cast<uint16_t>(sign | 0x7c00u);
  }
  return static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) | (rounded_mant >> 13));
}

inline __host__ __device__ float F16ToFloat(uint16_t h) {
  uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
  uint32_t exp = (h >> 10) & 0x1fu;
  uint32_t mant = h & 0x3ffu;
  uint32_t bits;
  if (exp == 0) {
    if (mant == 0) {
      bits = sign;
    } else {
      // Subnormal f16 -> normal fp32.
      int32_t e = -1;
      do {
        e++;
        mant <<= 1;
      } while (!(mant & 0x400u));
      mant &= 0x3ffu;
      bits = sign | (static_cast<uint32_t>(112 - e) << 23) | (mant << 13);
    }
  } else if (exp == 0x1fu) {
    bits = sign | 0x7f800000u | (mant << 13);
  } else {
    bits = sign | ((exp + 112u) << 23) | (mant << 13);
  }
  return F32FromBits(bits);
}

// ---- fp8 e4m3 (OCP E4M3FN: 1 sign, 4 exponent bits bias 7, 3 mantissa bits, no infinities,
// S.1111.111 is the only NaN, max finite magnitude 448) -------------------------------------
inline __host__ __device__ uint8_t FloatToFp8E4M3(float f) {
  uint32_t bits = BitsOfF32(f);
  const uint32_t sign = (bits >> 31) & 1u;
  const uint32_t abs_bits = bits & 0x7fffffffu;

  if (abs_bits == 0u) return static_cast<uint8_t>(sign << 7);
  if (abs_bits >= 0x7f800000u) return static_cast<uint8_t>((sign << 7) | 0x7fu);  // inf/NaN -> NaN

  float af = F32FromBits(abs_bits);
  const float kMax = 448.0f;
  if (af > kMax) af = kMax;  // saturate; e4m3fn has no infinity
  const float kMinNormal = 0.015625f;  // 2^-6

  uint32_t exp_field, mant3;
  if (af >= kMinNormal) {
    int exp2;
    float m = frexpf(af, &exp2);  // af = m * 2^exp2, m in [0.5, 1)
    int e = exp2 - 1;             // unbiased exponent for af in [2^e, 2^(e+1))
    float frac = m * 2.0f - 1.0f;  // mantissa fraction in [0, 1)
    int mi = static_cast<int>(rintf(frac * 8.0f));
    if (mi == 8) {
      mi = 0;
      e += 1;
    }
    if (e > 8) {
      e = 8;
      mi = 6;  // clamp to the max representable finite value (448), avoid the exp=15/mant=7 NaN
    }
    exp_field = static_cast<uint32_t>(e + 7);
    mant3 = static_cast<uint32_t>(mi);
  } else {
    // Subnormal: value = (mant3 / 8) * 2^-6.
    int mi = static_cast<int>(rintf(af / kMinNormal * 8.0f));
    if (mi > 7) {
      exp_field = 1;
      mant3 = 0;
    } else {
      exp_field = 0;
      mant3 = static_cast<uint32_t>(mi);
    }
  }
  return static_cast<uint8_t>((sign << 7) | (exp_field << 3) | (mant3 & 0x7u));
}

inline __host__ __device__ float Fp8E4M3ToFloat(uint8_t v) {
  const uint32_t sign = (v >> 7) & 1u;
  const uint32_t exp_field = (v >> 3) & 0xfu;
  const uint32_t mant3 = v & 0x7u;
  float value;
  if (exp_field == 0u) {
    value = (static_cast<float>(mant3) / 8.0f) * 0.015625f;  // subnormal, 2^-6
  } else if (exp_field == 0xfu && mant3 == 0x7u) {
    value = F32FromBits(0x7fc00000u);
    return sign ? -value : value;
  } else {
    int e = static_cast<int>(exp_field) - 7;
    value = (1.0f + static_cast<float>(mant3) / 8.0f) * exp2f(static_cast<float>(e));
  }
  return sign ? -value : value;
}

}  // namespace r4dx::core
