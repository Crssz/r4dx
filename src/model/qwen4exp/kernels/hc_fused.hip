// src/model/qwen4exp/kernels/hc_fused.hip -- see include/r4dx/model/qwen4exp/hc_fused.h for the contract.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include "r4d_gdn_wmma.h"  // third_party/libr4d: v8bf / v8f, mma, rowGo (the GEMM's own helpers)
#include "r4dx/model/qwen4exp/hc_device.h"
#include "r4dx/model/qwen4exp/hc_fused.h"

namespace dev = r4dx::model::qwen4exp::dev;
using dev::Bf16ToFloat;
using dev::FloatToBf16;
using dev::Rb;

namespace {

constexpr int kHc = 4;
constexpr int kHidden = 2560;
constexpr int kLowrank = 320;
constexpr int kSkDown = 4;     // r4d_gemm_bf16_nt_m64 split of the down GEMM (ApplyLinear fallback: K % 256 == 0)
constexpr int kSkInject = 16;  // ... of the N = 4 inject GEMM (ApplyBf16Linear narrow path)
constexpr int kSkUp = 4;       // ... of the K = 320 up GEMM
constexpr int kKu = 20;        // k steps issued back to back per iteration (the accumulation order does not depend on it)

// The GEMM kernel's fragment load: an 8-byte scalar through a global-space pointer (see r4d_gemm_bf16_nt_m64.hip).
typedef __attribute__((address_space(1))) const unsigned long long* gu2;

__device__ __forceinline__ v8bf FragU(gu2 p, int u) {
  union { unsigned long long w[2]; v8bf f; } x;
  x.w[0] = p[4 * u];
  x.w[1] = p[4 * u + 2];
  return x.f;
}

// One wave's partial 16 x 16 tile of C[0, n0 .. n0 + 15] = A[0, :] . W[n0 .., :]^T over the K slice [kbase, kbase + 16 * steps):
// exactly the accumulation chain of one (column tile, k split) wave of r4d_gemm_bf16_nt_m64_kernel at M = 1. Row 0 of the tile
// is acc[0] of lanes 0..15. Rows past `nlim` are clamped (re-read, never stored).
template <int KU>
__device__ __forceinline__ float SliceRow0(const unsigned short* A, const unsigned short* W, unsigned K, int n0, int nlim,
                                            int kbase, int steps, int lane) {
  v8f acc = (v8f)(0.f);
  const unsigned woff = rowGo(K, n0, nlim, lane) + static_cast<unsigned>(kbase);
  const unsigned aoff = rowGo(K, 0, 0, lane) + static_cast<unsigned>(kbase);
  gu2 pw = (gu2)(const unsigned long long*)(W + woff);
  gu2 pa = (gu2)(const unsigned long long*)(A + aoff);
  int s = 0;
  for (; s + KU <= steps; s += KU) {
#pragma unroll
    for (int u = 0; u < KU; u++) {
      const v8bf b = FragU(pw, u);
      acc = mma(FragU(pa, u), b, acc);
    }
    pw += KU * 4;
    pa += KU * 4;
  }
  for (; s < steps; s++) {
    const v8bf b = FragU(pw, 0);
    pw += 4;
    acc = mma(FragU(pa, 0), b, acc);
    pa += 4;
  }
  return acc[0];
}

// the GEMM kernel's output conversion
__device__ __forceinline__ float RoundOut(float v) {
  const __hip_bfloat16 b = (__hip_bfloat16)v;
  return Bf16ToFloat(__builtin_bit_cast(unsigned short, b));
}

// grid = lowrank / 16 down tiles (4 waves = the 4 k splits each) + 1 inject block (4 waves x 4 splits of its 16 k splits).
__global__ __launch_bounds__(128) void HcPrepGemm1Kernel(const unsigned short* __restrict__ n,
                                                         const unsigned short* __restrict__ w_down,
                                                         const unsigned short* __restrict__ w_inject,
                                                         unsigned short* mid, unsigned short* inj, float silu_scale) {
  __shared__ float part[kSkInject][16];
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  constexpr unsigned K = kHc * kHidden;
  if (blockIdx.x < kLowrank / 16) {
    const int n0 = blockIdx.x * 16;
    const float p = SliceRow0<kKu>(n, w_down, K, n0, kLowrank - 1, wave * (K / kSkDown), (K / kSkDown) / 16, lane);
    if (lane < 16) part[wave][lane] = p;
    __syncthreads();
    if (threadIdx.x < 16) {
      float v = 0.f;
#pragma unroll
      for (int s = 0; s < kSkDown; s++) v += part[s][threadIdx.x];
      mid[n0 + threadIdx.x] = FloatToBf16(dev::Silu(Rb(RoundOut(v) * silu_scale)));
    }
  } else {
#pragma unroll
    for (int i = 0; i < kSkInject / 4; i++) {
      const int s = wave + 4 * i;
      const float p = SliceRow0<kKu>(n, w_inject, K, 0, kHc - 1, s * (K / kSkInject), (K / kSkInject) / 16, lane);
      if (lane < 16) part[s][lane] = p;
    }
    __syncthreads();
    if (threadIdx.x < kHc) {
      float v = 0.f;
#pragma unroll
      for (int s = 0; s < kSkInject; s++) v += part[s][threadIdx.x];
      const float g = Rb(dev::Sigmoid(Rb(RoundOut(v) / static_cast<float>(kHc))));
      inj[threadIdx.x] = FloatToBf16(2.0f * g);
    }
  }
}

// grid = hidden / 16 blocks of 16 waves: wave = g * 4 + s is (stream g's column tile d0 .. d0 + 15, k split s of 80).
__global__ __launch_bounds__(512) void HcPrepUpMixKernel(const unsigned short* __restrict__ mid,
                                                         const unsigned short* __restrict__ w_up,
                                                         const unsigned short* __restrict__ n, unsigned short* x) {
  __shared__ float part[kHc][kSkUp][16];
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int g = wave / kSkUp, s = wave % kSkUp;
  const int d0 = blockIdx.x * 16;
  const float p = SliceRow0<5>(mid, w_up, kLowrank, g * kHidden + d0, kHc * kHidden - 1, s * (kLowrank / kSkUp),
                               (kLowrank / kSkUp) / 16, lane);
  if (lane < 16) part[g][s][lane] = p;
  __syncthreads();
  if (threadIdx.x < 16) {
    const int d = d0 + threadIdx.x;
    float acc = 0.0f;
#pragma unroll
    for (int gg = 0; gg < kHc; gg++) {
      float v = 0.f;
#pragma unroll
      for (int ss = 0; ss < kSkUp; ss++) v += part[gg][ss][threadIdx.x];
      const float m = Rb(dev::Sigmoid(RoundOut(v)));
      acc += Rb(m * Bf16ToFloat(n[gg * kHidden + d]));
    }
    x[d] = FloatToBf16(acc / static_cast<float>(kHc));
  }
}

}  // namespace

extern "C" int q4x_hc_fused_supported(int hc, int hidden, int lowrank) {
  return hc == kHc && hidden == kHidden && lowrank == kLowrank ? 1 : 0;
}

extern "C" int q4x_hc_prep_gemm1_bf16(const void* n, const void* w_down, const void* w_inject, void* mid, void* inj, int hc,
                                       int hidden, int lowrank, float silu_scale, void* stream) {
  if (!q4x_hc_fused_supported(hc, hidden, lowrank)) return hipErrorInvalidValue;
  const unsigned blocks = kLowrank / 16 + (w_inject != nullptr ? 1 : 0);
  hipLaunchKernelGGL(HcPrepGemm1Kernel, dim3(blocks), dim3(128), 0, static_cast<hipStream_t>(stream),
                     static_cast<const unsigned short*>(n), static_cast<const unsigned short*>(w_down),
                     static_cast<const unsigned short*>(w_inject), static_cast<unsigned short*>(mid),
                     static_cast<unsigned short*>(inj), silu_scale);
  return static_cast<int>(hipGetLastError());
}

extern "C" int q4x_hc_prep_up_mix_bf16(const void* mid, const void* w_up, const void* n, void* x, int hc, int hidden,
                                        int lowrank, void* stream) {
  if (!q4x_hc_fused_supported(hc, hidden, lowrank)) return hipErrorInvalidValue;
  hipLaunchKernelGGL(HcPrepUpMixKernel, dim3(kHidden / 16), dim3(512), 0, static_cast<hipStream_t>(stream),
                     static_cast<const unsigned short*>(mid), static_cast<const unsigned short*>(w_up),
                     static_cast<const unsigned short*>(n), static_cast<unsigned short*>(x));
  return static_cast<int>(hipGetLastError());
}
