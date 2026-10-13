// Device-side helpers of the QSA kernels (qsa_*.hip). Included from .hip files only.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "r4dx/core/dtype.hpp"

namespace r4dx::model::qwen4exp::qsa {

using r4dx::core::Bf16ToFloat;
using r4dx::core::FloatToBf16;

constexpr int kWave = 32;

// Round an fp32 value to the nearest bf16 (RNE) and widen it back: one HF bf16 op boundary.
__device__ __forceinline__ float Rb(float v) { return Bf16ToFloat(FloatToBf16(v)); }

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int m = kWave / 2; m; m >>= 1) v += __shfl_xor(v, m, kWave);
  return v;
}

__device__ __forceinline__ float WaveMax(float v) {
#pragma unroll
  for (int m = kWave / 2; m; m >>= 1) v = fmaxf(v, __shfl_xor(v, m, kWave));
  return v;
}

// HF `Qwen4ExpTextRMSNorm` on one row held in LDS as bf16, in place, by one wave (lanes 0..31). rows of D
// elements, lane l owns elements l, l + 32, ...: out = Rb((x * rstd) * (1 + w)), rstd = rsqrt(mean(x^2) + eps).
// `w` is the raw bf16 weight in global memory. The caller guarantees LDS visibility of the row before and after.
template <int D>
__device__ __forceinline__ void NormRowInPlace(uint16_t* row, const uint16_t* __restrict__ w, float eps, int lane) {
  float ss = 0.0f;
#pragma unroll
  for (int i = lane; i < D; i += kWave) {
    const float x = Bf16ToFloat(row[i]);
    ss += x * x;
  }
  ss = WaveSum(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(D) + eps);
#pragma unroll
  for (int i = lane; i < D; i += kWave) {
    row[i] = FloatToBf16((Bf16ToFloat(row[i]) * rstd) * (1.0f + Bf16ToFloat(w[i])));
  }
}

// HF `apply_rotary_pos_emb` (bf16 cos / sin, op order of the header comment) on the first 2 * half elements of an
// LDS row, in place, one lane per rotation pair (half <= 32). pos is the rope position.
__device__ __forceinline__ void RopeRowInPlace(uint16_t* row, const float* __restrict__ inv_freq, int half, int pos,
                                                int lane) {
  if (lane < half) {
    float s, c;
    sincosf(static_cast<float>(pos) * inv_freq[lane], &s, &c);
    c = Rb(c);
    s = Rb(s);
    const float x0 = Bf16ToFloat(row[lane]);
    const float x1 = Bf16ToFloat(row[lane + half]);
    row[lane] = FloatToBf16(Rb(x0 * c) + Rb(-x1 * s));
    row[lane + half] = FloatToBf16(Rb(x1 * c) + Rb(x0 * s));
  }
}

}  // namespace r4dx::model::qwen4exp::qsa
