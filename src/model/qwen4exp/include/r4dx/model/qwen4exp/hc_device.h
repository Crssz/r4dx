// Device-side helpers shared by the qwen4_exp small kernels (hc_*, moe_*, ple_*, gdn_sigmoid_*, embed_*).
// Included from .hip files only.
//
// Rounding convention of these kernels: the HF reference runs every op in bfloat16 (torch rounds each op's
// result to bf16, with fp32 internal math for reductions / transcendental functions). Rb() is that
// per-op rounding (RNE, same as core::FloatToBf16); the kernels apply it exactly where HF has a bf16
// tensor between two ops and keep fp32 everywhere else. Each kernel documents its rounding points.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "r4dx/core/dtype.hpp"

namespace r4dx::model::qwen4exp::dev {

using r4dx::core::Bf16ToFloat;
using r4dx::core::FloatToBf16;

constexpr int kWave = 32;
constexpr int kBlock = 256;
constexpr int kBlockWaves = kBlock / kWave;

// Round an fp32 value to the nearest bf16 (RNE) and widen it back: one HF bf16 op boundary.
__device__ __forceinline__ float Rb(float v) { return Bf16ToFloat(FloatToBf16(v)); }

__device__ __forceinline__ float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float Silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ float WaveSum(float v) {
#pragma unroll
  for (int m = kWave / 2; m; m >>= 1) v += __shfl_xor(v, m, kWave);
  return v;
}

// Sum over the 256 threads of a block. `scratch` is kBlockWaves floats of LDS; every thread of the block
// must call it (it contains barriers) and gets the total. A trailing barrier makes `scratch` reusable.
__device__ __forceinline__ float BlockSum(float v, float* scratch) {
  v = WaveSum(v);
  const int lane = threadIdx.x & (kWave - 1), wave = threadIdx.x / kWave;
  if (lane == 0) scratch[wave] = v;
  __syncthreads();
  float t = 0.0f;
#pragma unroll
  for (int i = 0; i < kBlockWaves; ++i) t += scratch[i];
  __syncthreads();
  return t;
}

}  // namespace r4dx::model::qwen4exp::dev
