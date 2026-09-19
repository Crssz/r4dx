// src/model/attention/src/attn_kernels.hip -- implements
// include/r4dx/model/attention/attn_kernels.h's two elementwise ops. Compiled by hipcc directly
// (see src/model/attention/CMakeLists.txt), same recipe as src/kernels/src/r4dx_kernels.hip.
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r4dx/core/dtype.hpp"
#include "r4dx/model/attention/attn_kernels.h"

namespace {

__global__ void SplitQgKernel(const uint16_t* __restrict__ qg, uint16_t* __restrict__ q,
                               uint16_t* __restrict__ gate, int T, int num_heads, int head_dim) {
  const int64_t total = static_cast<int64_t>(T) * num_heads * head_dim;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int d = static_cast<int>(idx % head_dim);
  const int64_t rest = idx / head_dim;
  const int h = static_cast<int>(rest % num_heads);
  const int64_t t = rest / num_heads;

  const int64_t qg_row_stride = static_cast<int64_t>(num_heads) * 2 * head_dim;
  const int64_t base = t * qg_row_stride + static_cast<int64_t>(h) * 2 * head_dim;
  q[idx] = qg[base + d];
  gate[idx] = qg[base + head_dim + d];
}

__global__ void GateMulKernel(const uint16_t* __restrict__ attn_out,
                               const uint16_t* __restrict__ gate, uint16_t* __restrict__ out,
                               int64_t n) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const float a = r4dx::core::Bf16ToFloat(attn_out[idx]);
  const float g = r4dx::core::Bf16ToFloat(gate[idx]);
  const float sg = 1.0f / (1.0f + expf(-g));
  out[idx] = r4dx::core::FloatToBf16(a * sg);
}

constexpr int kThreads = 256;

}  // namespace

extern "C" void r4dx_model_attn_split_qg_bf16(int64_t qg, int64_t q, int64_t gate, int T,
                                               int num_heads, int head_dim, int64_t stream) {
  const int64_t total = static_cast<int64_t>(T) * num_heads * head_dim;
  if (total <= 0) return;
  const int64_t blocks = (total + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(SplitQgKernel, dim3(static_cast<unsigned int>(blocks)), dim3(kThreads), 0,
                      reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(qg),
                      reinterpret_cast<uint16_t*>(q), reinterpret_cast<uint16_t*>(gate), T,
                      num_heads, head_dim);
}

extern "C" void r4dx_model_attn_gate_mul_bf16(int64_t attn_out, int64_t gate, int64_t out,
                                               int64_t n, int64_t stream) {
  if (n <= 0) return;
  const int64_t blocks = (n + kThreads - 1) / kThreads;
  hipLaunchKernelGGL(GateMulKernel, dim3(static_cast<unsigned int>(blocks)), dim3(kThreads), 0,
                      reinterpret_cast<hipStream_t>(stream),
                      reinterpret_cast<const uint16_t*>(attn_out),
                      reinterpret_cast<const uint16_t*>(gate), reinterpret_cast<uint16_t*>(out), n);
}
