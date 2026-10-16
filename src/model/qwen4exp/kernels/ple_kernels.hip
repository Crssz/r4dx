// src/model/qwen4exp/kernels/ple_kernels.hip -- contracts in include/r4dx/model/qwen4exp/ple_kernels.h.
#include <hip/hip_runtime.h>

#include "r4dx/model/qwen4exp/hc_device.h"
#include "r4dx/model/qwen4exp/ple_kernels.h"

namespace dev = r4dx::model::qwen4exp::dev;
using dev::Bf16ToFloat;
using dev::FloatToBf16;
using dev::Rb;

namespace {

constexpr int kMaxPerThread = 16;  // hidden <= 4096
constexpr int kTaps = 4;
constexpr int kDilation = 3;
constexpr int kState = (kTaps - 1) * kDilation;  // 9

// One workgroup per (row, stream): the stream's key / residual group stays in registers over the two norms and the dot.
__global__ __launch_bounds__(dev::kBlock) void GateValueKernel(
    const uint16_t* __restrict__ kp, const uint16_t* __restrict__ h, const uint16_t* __restrict__ value,
    const uint16_t* __restrict__ wk, const uint16_t* __restrict__ wq, uint16_t* gv, int hc, int hidden, float eps) {
  __shared__ float scratch[dev::kBlockWaves];
  const int64_t row = blockIdx.x;
  const int g = blockIdx.y;
  const int64_t base = (row * hc + g) * hidden;
  const int64_t wbase = static_cast<int64_t>(g) * hidden;

  float kv[kMaxPerThread], qv[kMaxPerThread];
  float ssk = 0.0f, ssq = 0.0f;
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int d = threadIdx.x + j * dev::kBlock;
    kv[j] = d < hidden ? Bf16ToFloat(kp[base + d]) : 0.0f;
    qv[j] = d < hidden ? Bf16ToFloat(h[base + d]) : 0.0f;
    ssk += kv[j] * kv[j];
    ssq += qv[j] * qv[j];
  }
  const float inv_k = 1.0f / sqrtf(dev::BlockSum(ssk, scratch) / static_cast<float>(hidden) + eps);
  const float inv_q = 1.0f / sqrtf(dev::BlockSum(ssq, scratch) / static_cast<float>(hidden) + eps);

  float dot = 0.0f;
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int d = threadIdx.x + j * dev::kBlock;
    if (d < hidden) {
      const float kn = Rb(kv[j] * inv_k * (1.0f + Bf16ToFloat(wk[wbase + d])));
      const float qn = Rb(qv[j] * inv_q * (1.0f + Bf16ToFloat(wq[wbase + d])));
      dot += Rb(kn * qn);
    }
  }
  float gate = Rb(Rb(dev::BlockSum(dot, scratch)) / sqrtf(static_cast<float>(hidden)));
  const float sign = gate > 0.0f ? 1.0f : (gate < 0.0f ? -1.0f : 0.0f);
  gate = Rb(sqrtf(Rb(fmaxf(fabsf(gate), 1e-6f)))) * sign;
  const float s = Rb(dev::Sigmoid(gate));
#pragma unroll
  for (int j = 0; j < kMaxPerThread; ++j) {
    const int d = threadIdx.x + j * dev::kBlock;
    if (d < hidden) gv[base + d] = FloatToBf16(s * Bf16ToFloat(value[row * hidden + d]));
  }
}

__global__ __launch_bounds__(128) void ConvKernel(const uint16_t* __restrict__ gvn,
                                                  const uint16_t* __restrict__ gated_value,
                                                  const uint16_t* __restrict__ conv_w, uint16_t* state, uint16_t* h,
                                                  int64_t rows, int channels) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= channels) return;
  float w[kTaps], s[kState];
#pragma unroll
  for (int j = 0; j < kTaps; ++j) w[j] = Bf16ToFloat(conv_w[static_cast<int64_t>(c) * kTaps + j]);
#pragma unroll
  for (int i = 0; i < kState; ++i) s[i] = Bf16ToFloat(state[static_cast<int64_t>(c) * kState + i]);

  for (int64_t t = 0; t < rows; ++t) {
    const int64_t idx = t * channels + c;
    const float x = Bf16ToFloat(gvn[idx]);
    // window of 10: s[0..8], x; taps at window offsets 0, 3, 6, 9
    const float acc = ((w[0] * s[0] + w[1] * s[kDilation]) + w[2] * s[2 * kDilation]) + w[3] * x;
    const float conv = Rb(dev::Silu(Rb(acc)));
    const float o = Rb(Bf16ToFloat(gated_value[idx]) + conv);
    h[idx] = FloatToBf16(Bf16ToFloat(h[idx]) + o);
#pragma unroll
    for (int i = 0; i < kState - 1; ++i) s[i] = s[i + 1];
    s[kState - 1] = x;
  }
#pragma unroll
  for (int i = 0; i < kState; ++i) state[static_cast<int64_t>(c) * kState + i] = FloatToBf16(s[i]);
}

}  // namespace

extern "C" int q4x_ple_gate_value_bf16(const void* key_proj_out, const void* h, const void* value,
                                        const void* norm_key_w, const void* norm_query_w, void* gated_value,
                                        int64_t rows, int hc, int hidden, float eps, void* stream) {
  if (rows <= 0) return 0;
  if (hc < 1 || hidden < 1 || hidden > kMaxPerThread * dev::kBlock) return hipErrorInvalidValue;
  hipLaunchKernelGGL(GateValueKernel, dim3(static_cast<unsigned>(rows), static_cast<unsigned>(hc)),
                     dim3(dev::kBlock), 0, static_cast<hipStream_t>(stream),
                     static_cast<const uint16_t*>(key_proj_out), static_cast<const uint16_t*>(h),
                     static_cast<const uint16_t*>(value), static_cast<const uint16_t*>(norm_key_w),
                     static_cast<const uint16_t*>(norm_query_w), static_cast<uint16_t*>(gated_value), hc, hidden, eps);
  return static_cast<int>(hipGetLastError());
}

extern "C" int q4x_ple_conv_bf16(const void* gvn, const void* gated_value, const void* conv_w, void* state, void* h,
                                  int64_t rows, int channels, int kernel, int dilation, void* stream) {
  if (rows <= 0) return 0;
  if (kernel != kTaps || dilation != kDilation || channels < 1) return hipErrorInvalidValue;
  hipLaunchKernelGGL(ConvKernel, dim3(static_cast<unsigned>((channels + 127) / 128)), dim3(128), 0,
                     static_cast<hipStream_t>(stream), static_cast<const uint16_t*>(gvn),
                     static_cast<const uint16_t*>(gated_value), static_cast<const uint16_t*>(conv_w),
                     static_cast<uint16_t*>(state), static_cast<uint16_t*>(h), rows, channels);
  return static_cast<int>(hipGetLastError());
}
