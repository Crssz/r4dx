// src/model/qwen4exp/kernels/gdn_sigmoid_norm.hip -- contract in include/r4dx/model/qwen4exp/gdn_sigmoid_norm.h.
// One wave per row, four channels per lane (the layout of third_party/libr4d/r4d_gdn_gated_rmsnorm_h128_bf16.hip).
#include <hip/hip_runtime.h>

#include "r4dx/model/qwen4exp/gdn_sigmoid_norm.h"
#include "r4dx/model/qwen4exp/hc_device.h"

namespace dev = r4dx::model::qwen4exp::dev;
using dev::Bf16ToFloat;
using dev::FloatToBf16;
using dev::Rb;

namespace {

constexpr int kWidth = 128;
constexpr int kPerLane = 4;
constexpr int kRowsPerBlock = 8;

__global__ __launch_bounds__(kRowsPerBlock * dev::kWave) void SigmoidGatedRmsNormKernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ z, const float* __restrict__ w, uint16_t* o,
    int64_t rows, int64_t xrow, int64_t zrow, int64_t orow, float eps) {
  const int lane = threadIdx.x & (dev::kWave - 1), wave = threadIdx.x / dev::kWave;
  const int64_t r = static_cast<int64_t>(blockIdx.x) * kRowsPerBlock + wave;
  if (r >= rows) return;  // whole wave exits together; nothing below needs a workgroup barrier
  const int d0 = lane * kPerLane;

  float v[kPerLane];
  float ss = 0.0f;
#pragma unroll
  for (int c = 0; c < kPerLane; ++c) {
    v[c] = Bf16ToFloat(x[r * xrow + d0 + c]);
    ss += v[c] * v[c];
  }
  ss = dev::WaveSum(ss);
  const float scale = 1.0f / sqrtf(ss * (1.0f / kWidth) + eps);

#pragma unroll
  for (int c = 0; c < kPerLane; ++c) {
    const float norm = Rb(v[c] * scale);
    const float prod = Rb(w[d0 + c] * norm);
    const float gate = dev::Sigmoid(Bf16ToFloat(z[r * zrow + d0 + c]));
    o[r * orow + d0 + c] = FloatToBf16(prod * gate);
  }
}

}  // namespace

extern "C" int q4x_gdn_sigmoid_gated_rmsnorm_h128_bf16(const void* x, const void* z, const void* w, void* o,
                                                        int64_t rows, int64_t xrow, int64_t zrow, int64_t orow,
                                                        int width, float eps, void* stream) {
  if (width != kWidth) return hipErrorInvalidValue;
  if (rows <= 0) return 0;
  hipLaunchKernelGGL(SigmoidGatedRmsNormKernel, dim3(static_cast<unsigned>((rows + kRowsPerBlock - 1) / kRowsPerBlock)),
                     dim3(kRowsPerBlock * dev::kWave), 0, static_cast<hipStream_t>(stream),
                     static_cast<const uint16_t*>(x), static_cast<const uint16_t*>(z), static_cast<const float*>(w),
                     static_cast<uint16_t*>(o), rows, xrow, zrow, orow, eps);
  return static_cast<int>(hipGetLastError());
}
