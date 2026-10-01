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

// The rotation geometry is `hidden = nblk * block` with block = the largest power of two dividing
// hidden, capped at 1024 (src/convert rotation.hpp ChooseRotationBlock; the loader refuses a container
// that says otherwise): 5120 = 5 x 1024 (Qwen), 3840 = 15 x 256 (Gemma 4), ... Any such hidden runs;
// nblk is a compile-time constant only for the two production shapes (unrolled mix loops, the exact
// instruction stream the 5120-specialized kernel had), a runtime value for the rest.
constexpr int kRotMaxBlock = 1024;
constexpr int kRotMaxNblk = 32;
constexpr int kRotMaxLdsBytes = 48 * 1024;
constexpr int kHadMaxBlock = 1024;  // LDS floats per workgroup for the Hb kernel

struct RotGeom {
  int hidden, block, nblk, log2_block, threads;
  float scale;  // 1 / sqrt(block): a power of two (exact) for block = 1024, 256, 64, ...
};

inline int RotThreads(int block) { return block >= 1024 ? 512 : 256; }

inline int ChooseBlock(int64_t hidden) {
  const int64_t low = hidden & -hidden;
  return static_cast<int>(low < kRotMaxBlock ? low : kRotMaxBlock);
}

// Validates `hidden` against the geometry limits; throws naming `who`.
RotGeom MakeRotGeom(const char* who, int64_t hidden) {
  if (hidden <= 0) {
    throw std::runtime_error(std::string(who) + ": hidden must be positive, got " + std::to_string(hidden));
  }
  const int block = ChooseBlock(hidden);
  const int64_t nblk = hidden / block;
  const int64_t lds = (hidden + nblk * nblk + 512) * static_cast<int64_t>(sizeof(float));
  if (block < 2 || nblk > kRotMaxNblk || lds > kRotMaxLdsBytes) {
    throw std::runtime_error(std::string(who) + ": unsupported hidden " + std::to_string(hidden) +
                             " (block " + std::to_string(block) + ", nblk " + std::to_string(nblk) +
                             "): needs block >= 2, nblk <= " + std::to_string(kRotMaxNblk) +
                             " and at most " + std::to_string(kRotMaxLdsBytes) + " bytes of LDS");
  }
  RotGeom g;
  g.hidden = static_cast<int>(hidden);
  g.block = block;
  g.nblk = static_cast<int>(nblk);
  g.log2_block = r4dx::kernels::hadamard::Log2Pow2(block);
  g.threads = RotThreads(block);
  g.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(block)));
  return g;
}

// Dynamic LDS layout of both kernels below: v[hidden] | r[nblk * nblk] (r[c*nblk + b] = R[c][b])
// | red[threads] (the fused kernel's reduction scratch; unused here but kept so one size serves both).
inline size_t RotLdsBytes(const RotGeom& g) {
  return (static_cast<size_t>(g.hidden) + static_cast<size_t>(g.nblk) * g.nblk + g.threads) * sizeof(float);
}

// x <- x Q (inverse == 0) or x Q^T (inverse != 0), one row per workgroup, in place. Every read of
// the row happens before the first barrier and every write after FwhtLds' last one, so the in-place
// update needs no second buffer. NBLK_CT > 0: nblk is that constant (unrolled); 0: runtime g.nblk.
template <int NBLK_CT>
__global__ __launch_bounds__(512) void RotateResidualKernel(__hip_bfloat16* __restrict__ x, const float* __restrict__ signs,
                                     const float* __restrict__ mix, int inverse, int hidden_rt,
                                     int block_rt, int nblk_rt, int log2_block, float scale) {
  constexpr int kMaxN = NBLK_CT > 0 ? NBLK_CT : kRotMaxNblk;
  const int nblk = NBLK_CT > 0 ? NBLK_CT : nblk_rt;
  const int block = block_rt;
  const int hidden = hidden_rt;
  extern __shared__ float lds[];
  float* v = lds;
  float* r = lds + hidden;
  __hip_bfloat16* xr = x + static_cast<int64_t>(blockIdx.x) * hidden;
  for (int i = threadIdx.x; i < nblk * nblk; i += blockDim.x) r[i] = mix[i];
  __syncthreads();

  if (!inverse) {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      v[i] = __bfloat162float(xr[i]) * signs[i];
    }
  } else {
    // Q^T's first step is R's inverse (R^T) across the blocks: y[c][i] = sum_b x[b][i] R[c][b].
    for (int i = threadIdx.x; i < block; i += blockDim.x) {
      float xin[kMaxN];
#pragma unroll
      for (int b = 0; b < kMaxN; ++b) {
        if (b < nblk) xin[b] = __bfloat162float(xr[b * block + i]);
      }
      for (int c = 0; c < nblk; ++c) {
        float acc = 0.0f;
#pragma unroll
        for (int b = 0; b < kMaxN; ++b) {
          if (b < nblk) acc += xin[b] * r[c * nblk + b];
        }
        v[c * block + i] = acc;
      }
    }
  }
  __syncthreads();

  r4dx::kernels::hadamard::FwhtLds(v, hidden, log2_block);

  if (!inverse) {
    // z[b][i] = sum_c y[c][i] R[c][b], with y already FWHT'd; for block = 4^k the 1/sqrt(block)
    // is a power of two, so applying it before the mix is exact.
    for (int i = threadIdx.x; i < block; i += blockDim.x) {
      float y[kMaxN];
#pragma unroll
      for (int c = 0; c < kMaxN; ++c) {
        if (c < nblk) y[c] = v[c * block + i] * scale;
      }
      for (int b = 0; b < nblk; ++b) {
        float acc = 0.0f;
#pragma unroll
        for (int c = 0; c < kMaxN; ++c) {
          if (c < nblk) acc += y[c] * r[c * nblk + b];
        }
        xr[b * block + i] = __float2bfloat16(acc);
      }
    }
  } else {
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
      xr[i] = __float2bfloat16(v[i] * scale * signs[i]);
    }
  }
}

// resid <- (resid + Q(rmsnorm_plain(y, w))) * layer_scale, one row per workgroup, in place over resid.
// y is the o_proj / down_proj output in the ORIGINAL basis (Gemma 4 option A, docs/gemma4-plan.md 4.4);
// the post-norm weight w is plain (x * rsqrt(mean(x^2) + eps) * w, no 1 + w); Q is the residual
// rotation above. The normed row never exists in memory: fp32 in LDS from the norm through the
// rotation, then ONE bf16 rounding of the final value. y and resid must not alias.
template <int NBLK_CT>
__global__ __launch_bounds__(512) void PostRmsnormRotateAddKernel(__hip_bfloat16* __restrict__ resid,
                                           const __hip_bfloat16* __restrict__ y,
                                           const __hip_bfloat16* __restrict__ w,
                                           const float* __restrict__ signs,
                                           const float* __restrict__ mix, int hidden_rt, int block_rt,
                                           int nblk_rt, int log2_block, float scale, float eps,
                                           float layer_scale) {
  constexpr int kMaxN = NBLK_CT > 0 ? NBLK_CT : kRotMaxNblk;
  const int nblk = NBLK_CT > 0 ? NBLK_CT : nblk_rt;
  const int block = block_rt;
  const int hidden = hidden_rt;
  extern __shared__ float lds[];
  float* v = lds;
  float* r = lds + hidden;
  float* red = r + nblk * nblk;  // blockDim.x floats
  const int64_t row = blockIdx.x;
  const __hip_bfloat16* yr = y + row * hidden;
  __hip_bfloat16* xr = resid + row * hidden;

  for (int i = threadIdx.x; i < nblk * nblk; i += blockDim.x) r[i] = mix[i];
  float ss = 0.0f;
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float t = __bfloat162float(yr[i]);
    ss += t * t;
  }
  // Fixed-order tree reduction (blockDim.x is a power of two): deterministic and independent of rows.
  red[threadIdx.x] = ss;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (static_cast<int>(threadIdx.x) < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  const float rstd = rsqrtf(red[0] / static_cast<float>(hidden) + eps);

  // x Q step 1: y = (normed * w) * d
  for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
    v[i] = (__bfloat162float(yr[i]) * rstd * __bfloat162float(w[i])) * signs[i];
  }
  __syncthreads();

  r4dx::kernels::hadamard::FwhtLds(v, hidden, log2_block);

  for (int i = threadIdx.x; i < block; i += blockDim.x) {
    float t[kMaxN];
#pragma unroll
    for (int c = 0; c < kMaxN; ++c) {
      if (c < nblk) t[c] = v[c * block + i] * scale;
    }
    for (int b = 0; b < nblk; ++b) {
      float acc = 0.0f;
#pragma unroll
      for (int c = 0; c < kMaxN; ++c) {
        if (c < nblk) acc += t[c] * r[c * nblk + b];
      }
      const int64_t idx = static_cast<int64_t>(b) * block + i;
      xr[idx] = __float2bfloat16((__bfloat162float(xr[idx]) + acc) * layer_scale);
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

template <int NBLK_CT>
void LaunchRotate(const RotGeom& g, int64_t x, int64_t rows, int64_t signs, int64_t mix, int inverse,
                  int64_t stream) {
  hipLaunchKernelGGL(RotateResidualKernel<NBLK_CT>, dim3(static_cast<unsigned>(rows)),
                     dim3(g.threads), RotLdsBytes(g), (hipStream_t)stream, (__hip_bfloat16*)x,
                     (const float*)signs, (const float*)mix, inverse, g.hidden, g.block, g.nblk,
                     g.log2_block, g.scale);
}

template <int NBLK_CT>
void LaunchPostNormRotateAdd(const RotGeom& g, int64_t resid, int64_t y, int64_t w, int64_t signs,
                             int64_t mix, int64_t rows, float eps, float layer_scale, int64_t stream) {
  hipLaunchKernelGGL(PostRmsnormRotateAddKernel<NBLK_CT>, dim3(static_cast<unsigned>(rows)),
                     dim3(g.threads), RotLdsBytes(g), (hipStream_t)stream, (__hip_bfloat16*)resid,
                     (const __hip_bfloat16*)y, (const __hip_bfloat16*)w, (const float*)signs,
                     (const float*)mix, g.hidden, g.block, g.nblk, g.log2_block, g.scale, eps,
                     layer_scale);
}

}  // namespace

extern "C" {

void r4dx_rotate_residual_bf16(int64_t x, int64_t rows, int64_t hidden, int64_t signs,
                                int64_t mix5, int inverse, int64_t stream) {
  const RotGeom g = MakeRotGeom("r4dx_rotate_residual_bf16", hidden);
  if (rows <= 0) return;
  if (signs == 0 || mix5 == 0) {
    throw std::runtime_error("r4dx_rotate_residual_bf16: signs and mix must be non-null");
  }
  r4dx_kernel_launch_counter_add(1);
  switch (g.nblk) {
    case 5: LaunchRotate<5>(g, x, rows, signs, mix5, inverse, stream); break;
    case 15: LaunchRotate<15>(g, x, rows, signs, mix5, inverse, stream); break;
    default: LaunchRotate<0>(g, x, rows, signs, mix5, inverse, stream); break;
  }
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_post_rmsnorm_rotate_add_bf16(int64_t resid, int64_t y, int64_t weight, int64_t signs,
                                        int64_t mix, int64_t rows, int64_t hidden, float eps,
                                        float layer_scale, int64_t stream) {
  const RotGeom g = MakeRotGeom("r4dx_post_rmsnorm_rotate_add_bf16", hidden);
  if (rows <= 0) return;
  if (resid == 0 || y == 0 || weight == 0 || signs == 0 || mix == 0) {
    throw std::runtime_error(
        "r4dx_post_rmsnorm_rotate_add_bf16: resid, y, weight, signs and mix must be non-null");
  }
  if (resid == y) {
    throw std::runtime_error("r4dx_post_rmsnorm_rotate_add_bf16: resid and y must not alias");
  }
  r4dx_kernel_launch_counter_add(1);
  switch (g.nblk) {
    case 5: LaunchPostNormRotateAdd<5>(g, resid, y, weight, signs, mix, rows, eps, layer_scale, stream); break;
    case 15: LaunchPostNormRotateAdd<15>(g, resid, y, weight, signs, mix, rows, eps, layer_scale, stream); break;
    default: LaunchPostNormRotateAdd<0>(g, resid, y, weight, signs, mix, rows, eps, layer_scale, stream); break;
  }
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
