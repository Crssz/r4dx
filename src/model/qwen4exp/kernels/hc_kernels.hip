// src/model/qwen4exp/kernels/hc_kernels.hip -- see include/r4dx/model/qwen4exp/hc_kernels.h for the contract of every
// entry point and its HF source lines.
#include <hip/hip_runtime.h>

#include "r4dx/model/qwen4exp/hc_device.h"
#include "r4dx/model/qwen4exp/hc_kernels.h"

namespace dev = r4dx::model::qwen4exp::dev;
using dev::Bf16ToFloat;
using dev::FloatToBf16;
using dev::Rb;

namespace {

constexpr int kMaxPerThread = 16;  // group_size <= 4096 for the 256-thread norm

// One workgroup per (row, group): the whole group (2560 channels = 10 per thread) stays in registers between the
// two passes.
__global__ __launch_bounds__(dev::kBlock) void GroupedRmsNormKernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, uint16_t* out, int groups, int group_size,
    float eps) {
  __shared__ float scratch[dev::kBlockWaves];
  const int64_t row = blockIdx.x;
  const int g = blockIdx.y;
  const int64_t base = (row * groups + g) * group_size;
  const int64_t wbase = static_cast<int64_t>(g) * group_size;

  float v[kMaxPerThread];
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int d = threadIdx.x + j * dev::kBlock;
    v[j] = d < group_size ? Bf16ToFloat(x[base + d]) : 0.0f;
    ss += v[j] * v[j];
  }
  const float total = dev::BlockSum(ss, scratch);
  const float inv = 1.0f / sqrtf(total / static_cast<float>(group_size) + eps);
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int d = threadIdx.x + j * dev::kBlock;
    if (d < group_size) out[base + d] = FloatToBf16(v[j] * inv * (1.0f + Bf16ToFloat(w[wbase + d])));
  }
}

__global__ void SiluScaledKernel(const uint16_t* __restrict__ in, uint16_t* out, int64_t n, float scale) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = FloatToBf16(dev::Silu(Rb(Bf16ToFloat(in[i]) * scale)));
}

// One thread per (row, channel d): reads the hc stream values of channel d.
__global__ void HcMixKernel(const uint16_t* __restrict__ n, const uint16_t* __restrict__ up, uint16_t* x,
                            int64_t rows, int hc, int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * hidden) return;
  const int64_t r = i / hidden;
  const int d = static_cast<int>(i - r * hidden);
  float acc = 0.0f;
  for (int g = 0; g < hc; ++g) {
    const int64_t k = (r * hc + g) * hidden + d;
    const float m = Rb(dev::Sigmoid(Bf16ToFloat(up[k])));
    acc += Rb(m * Bf16ToFloat(n[k]));
  }
  x[i] = FloatToBf16(acc / static_cast<float>(hc));
}

__global__ void InjectGateKernel(const uint16_t* __restrict__ v, uint16_t* inj, int64_t n, int hc) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const float s = Rb(dev::Sigmoid(Rb(Bf16ToFloat(v[i]) / static_cast<float>(hc))));
    inj[i] = FloatToBf16(2.0f * s);
  }
}

__global__ void InjectKernel(const uint16_t* h, const uint16_t* __restrict__ block_out,
                             const uint16_t* __restrict__ inj, uint16_t* h_out, int64_t rows, int hc, int hidden) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t width = static_cast<int64_t>(hc) * hidden;
  if (i >= rows * width) return;
  const int64_t r = i / width;
  const int64_t c = i - r * width;
  const int g = static_cast<int>(c / hidden);
  const int d = static_cast<int>(c - static_cast<int64_t>(g) * hidden);
  const float injection = Rb(Bf16ToFloat(block_out[r * hidden + d]) * Bf16ToFloat(inj[r * hc + g]));
  h_out[i] = FloatToBf16(Bf16ToFloat(h[i]) + injection);
}

unsigned Blocks(int64_t n) { return static_cast<unsigned>((n + dev::kBlock - 1) / dev::kBlock); }

}  // namespace

extern "C" int q4x_grouped_rmsnorm_bf16(const void* x, const void* w, void* out, int64_t rows, int groups,
                                         int group_size, float eps, void* stream) {
  if (rows <= 0) return 0;
  if (groups < 1 || group_size < 1 || group_size > kMaxPerThread * dev::kBlock) return hipErrorInvalidValue;
  hipLaunchKernelGGL(GroupedRmsNormKernel, dim3(static_cast<unsigned>(rows), static_cast<unsigned>(groups)),
                     dim3(dev::kBlock), 0, static_cast<hipStream_t>(stream), static_cast<const uint16_t*>(x),
                     static_cast<const uint16_t*>(w), static_cast<uint16_t*>(out), groups, group_size, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int q4x_hc_silu_scaled_bf16(const void* in, void* out, int64_t n, float scale, void* stream) {
  if (n <= 0) return 0;
  hipLaunchKernelGGL(SiluScaledKernel, dim3(Blocks(n)), dim3(dev::kBlock), 0, static_cast<hipStream_t>(stream),
                     static_cast<const uint16_t*>(in), static_cast<uint16_t*>(out), n, scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int q4x_hc_mix_bf16(const void* n, const void* up, void* x, int64_t rows, int hc, int hidden,
                                void* stream) {
  if (rows <= 0) return 0;
  hipLaunchKernelGGL(HcMixKernel, dim3(Blocks(rows * hidden)), dim3(dev::kBlock), 0, static_cast<hipStream_t>(stream),
                     static_cast<const uint16_t*>(n), static_cast<const uint16_t*>(up), static_cast<uint16_t*>(x),
                     rows, hc, hidden);
  return static_cast<int>(hipGetLastError());
}

extern "C" int q4x_hc_inject_gate_bf16(const void* v, void* inj, int64_t n, int hc, void* stream) {
  if (n <= 0) return 0;
  hipLaunchKernelGGL(InjectGateKernel, dim3(Blocks(n)), dim3(dev::kBlock), 0, static_cast<hipStream_t>(stream),
                     static_cast<const uint16_t*>(v), static_cast<uint16_t*>(inj), n, hc);
  return static_cast<int>(hipGetLastError());
}

extern "C" int q4x_hc_inject_bf16(const void* h, const void* block_out, const void* inj, void* h_out, int64_t rows,
                                   int hc, int hidden, void* stream) {
  if (rows <= 0) return 0;
  hipLaunchKernelGGL(InjectKernel, dim3(Blocks(rows * hc * hidden)), dim3(dev::kBlock), 0,
                     static_cast<hipStream_t>(stream), static_cast<const uint16_t*>(h),
                     static_cast<const uint16_t*>(block_out), static_cast<const uint16_t*>(inj),
                     static_cast<uint16_t*>(h_out), rows, hc, hidden);
  return static_cast<int>(hipGetLastError());
}
