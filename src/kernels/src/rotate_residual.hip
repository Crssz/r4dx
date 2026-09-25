// rotate_residual.hip -- quant2's residual-stream rotation Q / Q^T and the in-place blockwise
// Hadamard for gdn.out_proj's input (include/r4dx/kernels/rotate_residual.h; docs/quant2.md
// sections 3-4). Its own hipcc object (src/kernels/CMakeLists.txt), built with exactly
// r4dx_kernels.hip's flags -- in particular -ffp-contract=off, so every multiply-add below stays a
// separately rounded multiply and add, the same on every launch shape.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4dx/core/error.hpp"
#include "r4dx/kernels/hadamard_device.h"
#include "r4dx/kernels/kernels.h"  // r4dx_kernel_launch_counter_add
#include "r4dx/kernels/rotate_residual.h"

using r4dx::kernels::hadamard::FwhtLds;

namespace {

// The rotation contract's fixed shape: hidden = 5 blocks of 1024.
constexpr int kRotHidden = 5120;
constexpr int kRotBlock = 1024;
constexpr int kRotBlocks = 5;
constexpr int kRotLog2Block = 10;
constexpr float kRotScale = 1.0f / 32.0f;  // 1/sqrt(1024), exact in fp32
// 512 threads: 5 x 512 = 2560 butterflies per stage -> 5 per thread; a row is one workgroup, so at
// decode (rows = 1) this is the whole kernel's parallelism.
constexpr int kRotThreads = 512;

constexpr int kHadMaxBlock = 1024;  // LDS floats per workgroup for the Hb kernel

// x <- x Q (inverse == 0) or x Q^T (inverse != 0), one row per workgroup, in place. Every read of
// the row happens before the first barrier and every write after FwhtLds' last one, so the in-place
// update needs no second buffer.
__global__ __launch_bounds__(kRotThreads) void RotateResidualKernel(__hip_bfloat16* __restrict__ x,
                                                                      const float* __restrict__ signs,
                                                                      const float* __restrict__ mix5,
                                                                      int inverse) {
  __shared__ float v[kRotHidden];
  __shared__ float r[kRotBlocks * kRotBlocks];  // r[c*5 + b] = R[c][b]
  __hip_bfloat16* xr = x + static_cast<int64_t>(blockIdx.x) * kRotHidden;
  if (threadIdx.x < kRotBlocks * kRotBlocks) r[threadIdx.x] = mix5[threadIdx.x];
  __syncthreads();

  if (!inverse) {
    for (int i = threadIdx.x; i < kRotHidden; i += kRotThreads) {
      v[i] = __bfloat162float(xr[i]) * signs[i];
    }
  } else {
    // Q^T's first step is R's inverse (R^T) across the 5 blocks: y[c][i] = sum_b x[b][i] R[c][b].
    for (int i = threadIdx.x; i < kRotBlock; i += kRotThreads) {
      float xin[kRotBlocks];
#pragma unroll
      for (int b = 0; b < kRotBlocks; ++b) xin[b] = __bfloat162float(xr[b * kRotBlock + i]);
#pragma unroll
      for (int c = 0; c < kRotBlocks; ++c) {
        float acc = 0.0f;
#pragma unroll
        for (int b = 0; b < kRotBlocks; ++b) acc += xin[b] * r[c * kRotBlocks + b];
        v[c * kRotBlock + i] = acc;
      }
    }
  }
  __syncthreads();

  FwhtLds(v, kRotHidden, kRotLog2Block);

  if (!inverse) {
    // z[b][i] = sum_c y[c][i] R[c][b], with y already FWHT'd; the 1/32 is a power of two, so
    // applying it before the mix is exact.
    for (int i = threadIdx.x; i < kRotBlock; i += kRotThreads) {
      float y[kRotBlocks];
#pragma unroll
      for (int c = 0; c < kRotBlocks; ++c) y[c] = v[c * kRotBlock + i] * kRotScale;
#pragma unroll
      for (int b = 0; b < kRotBlocks; ++b) {
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < kRotBlocks; ++c) acc += y[c] * r[c * kRotBlocks + b];
        xr[b * kRotBlock + i] = __float2bfloat16(acc);
      }
    }
  } else {
    for (int i = threadIdx.x; i < kRotHidden; i += kRotThreads) {
      xr[i] = __float2bfloat16(v[i] * kRotScale * signs[i]);
    }
  }
}

// x <- x Hb, one (row, block) per workgroup: grid = dim3(rows, K / block). A workgroup reads its
// whole block into LDS before its first barrier and writes it back after the last, so in place is
// safe; blocks never overlap.
__global__ __launch_bounds__(256) void HadamardInplaceKernel(__hip_bfloat16* __restrict__ x,
                                                              int64_t K,
                                                              const float* __restrict__ signs,
                                                              int block, int log2_block,
                                                              float scale) {
  __shared__ float v[kHadMaxBlock];
  const int64_t k0 = static_cast<int64_t>(blockIdx.y) * block;
  __hip_bfloat16* xr = x + static_cast<int64_t>(blockIdx.x) * K + k0;
  const float* s = signs + k0;
  for (int i = threadIdx.x; i < block; i += blockDim.x) v[i] = __bfloat162float(xr[i]) * s[i];
  __syncthreads();
  FwhtLds(v, block, log2_block);
  for (int i = threadIdx.x; i < block; i += blockDim.x) xr[i] = __float2bfloat16(v[i] * scale);
}

}  // namespace

extern "C" {

void r4dx_rotate_residual_bf16(int64_t x, int64_t rows, int64_t hidden, int64_t signs,
                                int64_t mix5, int inverse, int64_t stream) {
  if (hidden != kRotHidden) {
    throw std::runtime_error("r4dx_rotate_residual_bf16: hidden must be " +
                             std::to_string(kRotHidden) + " (5 x 1024), got " +
                             std::to_string(hidden));
  }
  if (rows <= 0) return;
  if (signs == 0 || mix5 == 0) {
    throw std::runtime_error("r4dx_rotate_residual_bf16: signs and mix5 must be non-null");
  }
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(RotateResidualKernel, dim3(static_cast<unsigned>(rows)), dim3(kRotThreads), 0,
                     (hipStream_t)stream, (__hip_bfloat16*)x, (const float*)signs,
                     (const float*)mix5, inverse);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_hadamard_inplace_bf16(int64_t x, int64_t rows, int64_t K, int64_t signs, int block,
                                 int64_t stream) {
  using namespace r4dx::kernels::hadamard;
  if (!IsPow2(block) || block < 2 || block > kHadMaxBlock) {
    throw std::runtime_error("r4dx_hadamard_inplace_bf16: block must be a power of two in [2, " +
                             std::to_string(kHadMaxBlock) + "], got " + std::to_string(block));
  }
  if (K <= 0 || K % block != 0) {
    throw std::runtime_error("r4dx_hadamard_inplace_bf16: K (" + std::to_string(K) +
                             ") must be a positive multiple of block (" + std::to_string(block) +
                             ")");
  }
  if (rows <= 0) return;
  if (signs == 0) throw std::runtime_error("r4dx_hadamard_inplace_bf16: signs must be non-null");
  const int log2_block = Log2Pow2(block);
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(block)));
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(HadamardInplaceKernel,
                     dim3(static_cast<unsigned>(rows), static_cast<unsigned>(K / block)),
                     dim3(ThreadsForBlock(block)), 0, (hipStream_t)stream, (__hip_bfloat16*)x, K,
                     (const float*)signs, block, log2_block, scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

}  // extern "C"
