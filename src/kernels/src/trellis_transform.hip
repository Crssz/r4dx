// trellis_transform.hip -- the trellis linears' input transform (docs/trellis-kernel.md 4.8;
// declared in include/r4dx/kernels/kernels.h): A = f16(FWHT128(x * suh) * 2^s / sqrt(128)), the
// f16 activation r4d_gemm_trellis_nt_m64 consumes. It replaces the per-linear bf16 -> f16 cast a
// w4a16 linear pays (r4dx_model_cast_bf16_to_f16).
//
// Its own hipcc object (src/kernels/CMakeLists.txt), built with exactly r4dx_kernels.hip's flags --
// -ffp-contract=off among them, so x * suh stays a separately rounded multiply ahead of the first
// butterfly (fusing it into that butterfly as an fma would change the bits the golden vectors and
// the converter's KL runs assume), and the product by the scale stays a multiply ahead of the one
// rounding to f16.
//
// One wave32 per workgroup and one 128-block of one row per wave: lane l holds x[l + 32 r],
// r = 0..3 -- the layout libr4d's r4d_fwht128_wave (r4d_fwht128.h) transforms in registers (DPP /
// permlanex16 for the cross-lane stages), the same header the GEMM's output transform runs, so the
// two sides of a trellis linear share one butterfly. Every output of the call (a linear's parts, or
// several linears on one input) reuses the row's x from registers.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4d_fwht128.h"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/hadamard_device.h"
#include "r4dx/kernels/kernels.h"

namespace {

constexpr int kTqBlock = 128;
constexpr int kTqWave = 32;
constexpr int kTqMaxOut = 3;
constexpr int kTqMaxPrescale = 24;

struct TrellisInputPtrs {
  const float* suh[kTqMaxOut];
  uint16_t* out[kTqMaxOut];
};

__device__ __forceinline__ float Bf16BitsToFloat(uint16_t b) {
  return __builtin_bit_cast(float, static_cast<uint32_t>(b) << 16);
}

// grid (K / 128, M), one wave per workgroup.
__global__ __launch_bounds__(kTqWave) void TrellisInputKernel(const uint16_t* __restrict__ x,
                                                               int64_t K, int nout,
                                                               TrellisInputPtrs p, float scale) {
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t off = static_cast<int64_t>(blockIdx.y) * K +
                      static_cast<int64_t>(blockIdx.x) * kTqBlock + lane;
  float xv[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) xv[r] = Bf16BitsToFloat(x[off + kTqWave * r]);
  const int64_t k_off = static_cast<int64_t>(blockIdx.x) * kTqBlock + lane;
#pragma unroll
  for (int o = 0; o < kTqMaxOut; ++o) {
    if (o >= nout) break;   // uniform
    float v[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) v[r] = xv[r] * p.suh[o][k_off + kTqWave * r];
    r4d_fwht128_wave(v, lane);
#pragma unroll
    for (int r = 0; r < 4; ++r) {
      // The product is an fp32 value first, then rounded to f16 (the golden's and the converter's
      // f16(fp32(v * scale))). Left alone, the compiler folds the multiply and the conversion into
      // one v_fma_mixlo_f16 -- one rounding of the exact product, which differs in the last f16 bit
      // on ~1e-5 of the elements; the empty asm keeps the fp32 product (no instruction).
      float prod = v[r] * scale;
      asm("" : "+v"(prod));
      const _Float16 h = static_cast<_Float16>(prod);   // round to nearest even
      p.out[o][off + kTqWave * r] = __builtin_bit_cast(uint16_t, h);
    }
  }
}

// The diagnostic twins: one 128-block per workgroup.
__global__ __launch_bounds__(kTqWave) void Fwht128WaveKernel(float* __restrict__ x) {
  const int lane = static_cast<int>(threadIdx.x);
  float* b = x + static_cast<int64_t>(blockIdx.x) * kTqBlock;
  float v[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) v[r] = b[lane + kTqWave * r];
  r4d_fwht128_wave(v, lane);
#pragma unroll
  for (int r = 0; r < 4; ++r) b[lane + kTqWave * r] = v[r];
}

__global__ __launch_bounds__(64) void Fwht128LdsKernel(float* __restrict__ x) {
  __shared__ float v[kTqBlock];
  float* b = x + static_cast<int64_t>(blockIdx.x) * kTqBlock;
  for (int i = static_cast<int>(threadIdx.x); i < kTqBlock; i += static_cast<int>(blockDim.x)) v[i] = b[i];
  __syncthreads();
  r4dx::kernels::hadamard::FwhtLds(v, kTqBlock, 7);
  for (int i = static_cast<int>(threadIdx.x); i < kTqBlock; i += static_cast<int>(blockDim.x)) b[i] = v[i];
}

[[noreturn]] void TqFail(const std::string& what) {
  throw std::runtime_error("r4dx_trellis_input_bf16: " + what);
}

}  // namespace

extern "C" {

void r4dx_trellis_input_bf16(int64_t x, int64_t M, int64_t K, int nout, const int64_t* suh,
                              const int64_t* out, int prescale_log2, int64_t stream) {
  if (nout < 1 || nout > kTqMaxOut) TqFail("nout must be 1..3, got " + std::to_string(nout));
  if (K <= 0 || K % kTqBlock != 0) TqFail("K must be a positive multiple of 128, got " + std::to_string(K));
  if (M < 0 || M > 65535) TqFail("M must be 0..65535, got " + std::to_string(M));
  if (prescale_log2 < -kTqMaxPrescale || prescale_log2 > kTqMaxPrescale)
    TqFail("|prescale_log2| must be <= 24, got " + std::to_string(prescale_log2));
  if (suh == nullptr || out == nullptr) TqFail("suh and out must be host arrays of nout pointers");
  TrellisInputPtrs p{};
  for (int o = 0; o < nout; ++o) {
    if (suh[o] == 0 || out[o] == 0) TqFail("suh[" + std::to_string(o) + "] / out[" + std::to_string(o) + "] is null");
    p.suh[o] = reinterpret_cast<const float*>(suh[o]);
    p.out[o] = reinterpret_cast<uint16_t*>(out[o]);
  }
  if (x == 0) TqFail("x is null");
  if (M == 0) return;
  // 2^s / sqrt(128) in double, rounded once to fp32 (tools/reference/trellis_golden.py's
  // transform_scale): not a device rsqrt.
  const float scale = static_cast<float>(std::ldexp(1.0, prescale_log2) / std::sqrt(128.0));
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(TrellisInputKernel,
                     dim3(static_cast<unsigned>(K / kTqBlock), static_cast<unsigned>(M)),
                     dim3(kTqWave), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(x), K, nout, p, scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_fwht128_f32(int64_t x, int64_t blocks, int use_lds, int64_t stream) {
  if (blocks < 0 || blocks > 0x7FFFFFFF) throw std::runtime_error("r4dx_fwht128_f32: bad block count");
  if (blocks == 0) return;
  if (x == 0) throw std::runtime_error("r4dx_fwht128_f32: x is null");
  r4dx_kernel_launch_counter_add(1);
  if (use_lds) {
    hipLaunchKernelGGL(Fwht128LdsKernel, dim3(static_cast<unsigned>(blocks)), dim3(64), 0,
                       reinterpret_cast<hipStream_t>(stream), reinterpret_cast<float*>(x));
  } else {
    hipLaunchKernelGGL(Fwht128WaveKernel, dim3(static_cast<unsigned>(blocks)), dim3(kTqWave), 0,
                       reinterpret_cast<hipStream_t>(stream), reinterpret_cast<float*>(x));
  }
  R4DX_HIP_CHECK(hipGetLastError());
}

}  // extern "C"
