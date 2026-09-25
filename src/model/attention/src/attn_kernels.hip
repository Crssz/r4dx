// src/model/attention/src/attn_kernels.hip -- implements
// include/r4dx/model/attention/attn_kernels.h's two elementwise ops, plus the output-gate
// multiply's quant2 Q2b Hadamard variant. Compiled by hipcc directly (see
// src/model/attention/CMakeLists.txt), same recipe as src/kernels/src/r4dx_kernels.hip.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4dx/core/dtype.hpp"
#include "r4dx/kernels/hadamard_device.h"  // src/kernels: the shared FwhtLds butterfly
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
constexpr int kGateHadMaxBlock = 1024;  // LDS floats per workgroup

// GateMulKernel's product, kept fp32, times the sign, then the block's FWHT / sqrt(block): one
// workgroup per (row, block), grid = dim3(rows, row_len / block). With block 256 = head_dim each
// workgroup is one (t, head) of the [T, H, D] contiguous buffer.
__global__ __launch_bounds__(256) void GateMulHadamardKernel(const uint16_t* __restrict__ attn_out,
                                                              const uint16_t* __restrict__ gate,
                                                              uint16_t* __restrict__ out,
                                                              int64_t row_len,
                                                              const float* __restrict__ signs,
                                                              int block, int log2_block,
                                                              float scale) {
  __shared__ float v[kGateHadMaxBlock];
  const int64_t k0 = static_cast<int64_t>(blockIdx.y) * block;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * row_len + k0;
  for (int i = threadIdx.x; i < block; i += blockDim.x) {
    const float a = r4dx::core::Bf16ToFloat(attn_out[base + i]);
    const float g = r4dx::core::Bf16ToFloat(gate[base + i]);
    const float sg = 1.0f / (1.0f + expf(-g));
    v[i] = a * sg * signs[k0 + i];
  }
  __syncthreads();
  r4dx::kernels::hadamard::FwhtLds(v, block, log2_block);
  for (int i = threadIdx.x; i < block; i += blockDim.x) {
    out[base + i] = r4dx::core::FloatToBf16(v[i] * scale);
  }
}

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

extern "C" void r4dx_model_attn_gate_mul_hadamard_bf16(int64_t attn_out, int64_t gate,
                                                        int64_t out, int64_t n, int64_t stream,
                                                        int64_t signs, int block,
                                                        int64_t row_len) {
  using namespace r4dx::kernels::hadamard;
  if (!IsPow2(block) || block < 2 || block > kGateHadMaxBlock) {
    throw std::runtime_error(
        "r4dx_model_attn_gate_mul_hadamard_bf16: block must be a power of two in [2, " +
        std::to_string(kGateHadMaxBlock) + "], got " + std::to_string(block));
  }
  if (row_len <= 0 || row_len % block != 0 || n % row_len != 0) {
    throw std::runtime_error("r4dx_model_attn_gate_mul_hadamard_bf16: need row_len (" +
                             std::to_string(row_len) + ") a positive multiple of block (" +
                             std::to_string(block) + ") and n (" + std::to_string(n) +
                             ") a multiple of row_len");
  }
  if (n <= 0) return;
  if (signs == 0) {
    throw std::runtime_error("r4dx_model_attn_gate_mul_hadamard_bf16: signs must be non-null");
  }
  const int64_t rows = n / row_len;
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(block)));
  hipLaunchKernelGGL(GateMulHadamardKernel,
                      dim3(static_cast<unsigned int>(rows), static_cast<unsigned int>(row_len / block)),
                      dim3(ThreadsForBlock(block)), 0, reinterpret_cast<hipStream_t>(stream),
                      reinterpret_cast<const uint16_t*>(attn_out),
                      reinterpret_cast<const uint16_t*>(gate), reinterpret_cast<uint16_t*>(out),
                      row_len, reinterpret_cast<const float*>(signs), block, Log2Pow2(block),
                      scale);
}
