// src/model/kernels/model_kernels.hip -- see model_kernels.h for why these two kernels live here
// rather than in src/kernels or r4d.h. Reuses r4dx::core's __host__ __device__ dtype helpers
// (dtype.hpp is designed for exactly this -- src/kernels/src/r4dx_kernels.hip does the same)
// rather than re-deriving bf16/f16 bit twiddling a third time.
#include <hip/hip_runtime.h>

#include "clock_probe_device.h"
#include "r4dx/core/dtype.hpp"
#include "model_kernels.h"

using r4dx::core::Bf16ToFloat;
using r4dx::core::FloatToF16;

namespace {

constexpr int kBlock = 256;

__global__ void CastBf16ToF16Kernel(const uint16_t* __restrict__ in, uint16_t* __restrict__ out,
                                     int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = FloatToF16(Bf16ToFloat(in[i]));
}

__global__ void WidenBf16ToF32Kernel(const uint16_t* __restrict__ in, float* __restrict__ out,
                                      int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = Bf16ToFloat(in[i]);
}

__global__ void WidenSoftcapBf16ToF32Kernel(const uint16_t* __restrict__ in, float* __restrict__ out,
                                             int64_t n, float cap) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = cap * tanhf(Bf16ToFloat(in[i]) / cap);
}

__global__ void WallStampKernel(unsigned long long* out) { r4dx_wall_stamp_device(out); }

__global__ void ClockProbeKernel(unsigned long long* out, int iters) {
  r4dx_clock_probe_device(out, iters);
}

}  // namespace

extern "C" void r4dx_model_cast_bf16_to_f16(int64_t in, int64_t out, int64_t n, int64_t stream) {
  if (n <= 0) return;
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(CastBf16ToF16Kernel, dim3(static_cast<unsigned int>(blocks)), dim3(kBlock), 0,
                      reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(in),
                      reinterpret_cast<uint16_t*>(out), n);
}

extern "C" void r4dx_model_widen_bf16_to_f32(int64_t in, int64_t out, int64_t n, int64_t stream) {
  if (n <= 0) return;
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(WidenBf16ToF32Kernel, dim3(static_cast<unsigned int>(blocks)), dim3(kBlock), 0,
                      reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(in),
                      reinterpret_cast<float*>(out), n);
}

extern "C" void r4dx_model_widen_softcap_bf16_to_f32(int64_t in, int64_t out, int64_t n, float cap,
                                                       int64_t stream) {
  if (n <= 0) return;
  const int64_t blocks = (n + kBlock - 1) / kBlock;
  hipLaunchKernelGGL(WidenSoftcapBf16ToF32Kernel, dim3(static_cast<unsigned int>(blocks)),
                      dim3(kBlock), 0, reinterpret_cast<hipStream_t>(stream),
                      reinterpret_cast<const uint16_t*>(in), reinterpret_cast<float*>(out), n, cap);
}

extern "C" void r4dx_model_wall_stamp(int64_t out, int64_t stream) {
  hipLaunchKernelGGL(WallStampKernel, dim3(1), dim3(1), 0, reinterpret_cast<hipStream_t>(stream),
                      reinterpret_cast<unsigned long long*>(out));
}

extern "C" void r4dx_model_clock_probe(int64_t out, int iters, int64_t stream) {
  hipLaunchKernelGGL(ClockProbeKernel, dim3(1), dim3(32), 0, reinterpret_cast<hipStream_t>(stream),
                      reinterpret_cast<unsigned long long*>(out), iters);
}
