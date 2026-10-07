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
//
// The fused producers (M5, docs/trellis-kernel.md 4.8 / 5.4) run the same transform on a value they
// compute themselves -- silu_mul's or the attention gate-mul's bf16 product -- instead of loading x.
//
// r4dx_fake_quant_act_f16 (below) is not part of the transform: the R4DX_FAKEQ_ACT diagnostic that rounds
// the transform's f16 output to int8 and back in place (docs/int8-prefill.md).
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4d_fwht128.h"
#include "r4d_trellis_i8_fused.h"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/gemma_device.h"
#include "r4dx/kernels/gemma_kernels.h"
#include "r4dx/kernels/hadamard_device.h"
#include "r4dx/kernels/kernels.h"

namespace {

constexpr int kTqBlock = 128;
constexpr int kTqWave = 32;
constexpr int kTqMaxOut = 3;
constexpr int kTqMaxPrescale = 24;
constexpr int kTqRowsI8 = 256;   // rows of the int8 prefill GEMM's A8 layout (i8p::kM)

struct TrellisInputPtrs {
  const float* suh[kTqMaxOut];
  uint16_t* out[kTqMaxOut];
};

__device__ __forceinline__ float Bf16BitsToFloat(uint16_t b) {
  return __builtin_bit_cast(float, static_cast<uint32_t>(b) << 16);
}

// One output of one 128-block: v = xv * suh (fp32), FWHT, then f16(fp32(v * scale)) at
// out + off + 32 r. xv: the block's x, lane l holding elements l + 32 r; k_off: the block's first
// column plus the lane, for suh.
__device__ __forceinline__ void TransformBlock(const float (&xv)[4], const float* __restrict__ suh,
                                               uint16_t* __restrict__ out, int64_t off,
                                               int64_t k_off, float scale, int lane) {
  float v[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) v[r] = xv[r] * suh[k_off + kTqWave * r];
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
    out[off + kTqWave * r] = __builtin_bit_cast(uint16_t, h);
  }
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
    TransformBlock(xv, p.suh[o], p.out[o], off, k_off, scale, lane);
  }
}

// ---- fused producers (kernels.h: r4dx_silu_mul_trellis_bf16, r4dx_attn_gate_mul_trellis_bf16) --
// Each element is the producer's own bf16 value -- the same expression and the same rounding as its
// v1 kernel -- widened back to fp32, then the transform above. -ffp-contract=off (this object's
// flags, the same as r4dx_kernels.hip's and attn_kernels.hip's) keeps every product a separately
// rounded multiply, as there.

// r4dx_kernels.hip's Sigmoid, and SiluMulWideKernel's element: __float2bfloat16(g * Sigmoid(g) * u).
__device__ __forceinline__ float TqSigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ __launch_bounds__(kTqWave) void SiluMulTrellisKernel(const uint16_t* __restrict__ gate_up,
                                                                 int64_t intermediate,
                                                                 int64_t in_row_stride,
                                                                 const float* __restrict__ suh,
                                                                 uint16_t* __restrict__ out,
                                                                 float scale) {
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t k_off = static_cast<int64_t>(blockIdx.x) * kTqBlock + lane;
  const uint16_t* gu = gate_up + static_cast<int64_t>(blockIdx.y) * in_row_stride;
  float xv[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float g = Bf16BitsToFloat(gu[k_off + kTqWave * r]);
    const float u = Bf16BitsToFloat(gu[intermediate + k_off + kTqWave * r]);
    const __hip_bfloat16 h = __float2bfloat16(g * TqSigmoid(g) * u);
    xv[r] = __bfloat162float(h);
  }
  TransformBlock(xv, suh, out, static_cast<int64_t>(blockIdx.y) * intermediate + k_off, k_off, scale,
                 lane);
}

// Gemma 4's GeGLU (r4dx_gelu_tanh_mul_wide_bf16's element, gemma_device.h's GeGluElement), same shape
// as SiluMulTrellisKernel above.
__global__ __launch_bounds__(kTqWave) void GeluMulTrellisKernel(const uint16_t* __restrict__ gate_up,
                                                                 int64_t intermediate,
                                                                 int64_t in_row_stride,
                                                                 const float* __restrict__ suh,
                                                                 uint16_t* __restrict__ out,
                                                                 float scale) {
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t k_off = static_cast<int64_t>(blockIdx.x) * kTqBlock + lane;
  const uint16_t* gu = gate_up + static_cast<int64_t>(blockIdx.y) * in_row_stride;
  float xv[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float g = Bf16BitsToFloat(gu[k_off + kTqWave * r]);
    const float u = Bf16BitsToFloat(gu[intermediate + k_off + kTqWave * r]);
    xv[r] = __bfloat162float(r4dx::kernels::gemma::GeGluElement(g, u));
  }
  TransformBlock(xv, suh, out, static_cast<int64_t>(blockIdx.y) * intermediate + k_off, k_off, scale,
                 lane);
}

// attn_kernels.hip's GateMulKernel element: FloatToBf16(a * (1 / (1 + expf(-g)))).
__global__ __launch_bounds__(kTqWave) void GateMulTrellisKernel(const uint16_t* __restrict__ attn_out,
                                                                 const uint16_t* __restrict__ gate,
                                                                 int64_t K,
                                                                 const float* __restrict__ suh,
                                                                 uint16_t* __restrict__ out,
                                                                 float scale) {
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t k_off = static_cast<int64_t>(blockIdx.x) * kTqBlock + lane;
  const int64_t off = static_cast<int64_t>(blockIdx.y) * K + k_off;
  float xv[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float a = r4dx::core::Bf16ToFloat(attn_out[off + kTqWave * r]);
    const float g = r4dx::core::Bf16ToFloat(gate[off + kTqWave * r]);
    const float sg = 1.0f / (1.0f + expf(-g));
    xv[r] = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(a * sg));
  }
  TransformBlock(xv, suh, out, off, k_off, scale, lane);
}

// ---- the int8 variants (r4dx_*_i8; R4DX_PREFILL_INT8_FUSEDQ, docs/int8-prefill.md "The fused quantizer") --------
// The same transform and the same producers, but the f16 A is never stored: each wave rounds its block to f16
// exactly as TransformBlock does, then hands the four f16 values to libr4d's activation quantizer
// (r4d_trellis_i8_fused.h: per (row, 128 k) amax, s = amax / 127, int8 straight into the A8 fragment layout, SA), so
// A8 and SA are the bytes r4d_trellis_i8_quant_act would have made from the f16 A. TransformBlock itself, and the
// kernels above, are untouched (their ISA is what the f16 paths run). Fixed at 256 rows, the A8 layout's M.
struct TrellisInputI8Ptrs {
  const float* suh[kTqMaxOut];
  signed char* a8[kTqMaxOut];
  float* sa[kTqMaxOut];
};

// TransformBlock's arithmetic (x * suh in fp32, the butterfly, the product by `scale` as an fp32 value, one rounding
// to f16), the f16 bits handed to the quantizer. row: the block's row; kb: its index along K.
__device__ __forceinline__ void TransformBlockQ8(const float (&xv)[4], const float* __restrict__ suh,
                                                 signed char* __restrict__ a8, float* __restrict__ sa,
                                                 int64_t k_off, float scale, int lane, int row, int kb,
                                                 int64_t K) {
  float v[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) v[r] = xv[r] * suh[k_off + kTqWave * r];
  r4d_fwht128_wave(v, lane);
  unsigned short hb[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    float prod = v[r] * scale;
    asm("" : "+v"(prod));
    const _Float16 h = static_cast<_Float16>(prod);   // round to nearest even
    hb[r] = __builtin_bit_cast(unsigned short, h);
  }
  i8g_quant_wave_block(hb, lane, row, kb, static_cast<int>(K), a8, sa);
}

// TrellisInputKernel's int8 twin: grid (K / 128, 256), one wave32 per workgroup.
__global__ __launch_bounds__(kTqWave) void TrellisInputI8Kernel(const uint16_t* __restrict__ x, int64_t K,
                                                                 int nout, TrellisInputI8Ptrs p, float scale) {
  const int lane = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.y), kb = static_cast<int>(blockIdx.x);
  const int64_t off = static_cast<int64_t>(row) * K + static_cast<int64_t>(kb) * kTqBlock + lane;
  float xv[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) xv[r] = Bf16BitsToFloat(x[off + kTqWave * r]);
  const int64_t k_off = static_cast<int64_t>(kb) * kTqBlock + lane;
#pragma unroll
  for (int o = 0; o < kTqMaxOut; ++o) {
    if (o >= nout) break;   // uniform
    TransformBlockQ8(xv, p.suh[o], p.a8[o], p.sa[o], k_off, scale, lane, row, kb, K);
  }
}

// SiluMulTrellisKernel's int8 twin (the element is the same bf16 product).
__global__ __launch_bounds__(kTqWave) void SiluMulTrellisI8Kernel(const uint16_t* __restrict__ gate_up,
                                                                   int64_t intermediate, int64_t in_row_stride,
                                                                   const float* __restrict__ suh,
                                                                   signed char* __restrict__ a8,
                                                                   float* __restrict__ sa, float scale) {
  const int lane = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.y), kb = static_cast<int>(blockIdx.x);
  const int64_t k_off = static_cast<int64_t>(kb) * kTqBlock + lane;
  const uint16_t* gu = gate_up + static_cast<int64_t>(row) * in_row_stride;
  float xv[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float g = Bf16BitsToFloat(gu[k_off + kTqWave * r]);
    const float u = Bf16BitsToFloat(gu[intermediate + k_off + kTqWave * r]);
    const __hip_bfloat16 h = __float2bfloat16(g * TqSigmoid(g) * u);
    xv[r] = __bfloat162float(h);
  }
  TransformBlockQ8(xv, suh, a8, sa, k_off, scale, lane, row, kb, intermediate);
}

// GateMulTrellisKernel's int8 twin.
__global__ __launch_bounds__(kTqWave) void GateMulTrellisI8Kernel(const uint16_t* __restrict__ attn_out,
                                                                   const uint16_t* __restrict__ gate, int64_t K,
                                                                   const float* __restrict__ suh,
                                                                   signed char* __restrict__ a8,
                                                                   float* __restrict__ sa, float scale) {
  const int lane = static_cast<int>(threadIdx.x);
  const int row = static_cast<int>(blockIdx.y), kb = static_cast<int>(blockIdx.x);
  const int64_t k_off = static_cast<int64_t>(kb) * kTqBlock + lane;
  const int64_t off = static_cast<int64_t>(row) * K + k_off;
  float xv[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float a = r4dx::core::Bf16ToFloat(attn_out[off + kTqWave * r]);
    const float g = r4dx::core::Bf16ToFloat(gate[off + kTqWave * r]);
    const float sg = 1.0f / (1.0f + expf(-g));
    xv[r] = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(a * sg));
  }
  TransformBlockQ8(xv, suh, a8, sa, k_off, scale, lane, row, kb, K);
}

// ---- the per-ROW int8 variants (r4dx_*_i8r; R4DX_PREFILL_INT8_SCALES=coarse, docs/int8-prefill.md "Coarse scales") ------
// The activation scale is one per ROW over the whole K, so no wave can quantize its block alone: a workgroup of 8 waves
// owns one row. Wave w transforms the row's 128-blocks w, w + 8, ... exactly as above (the same TransformBlockHalfBits, the
// same elements) and parks the f16 bits in a row buffer in LDS (nout rows of K ushort, a linear's parts or the
// several linears on one input); one barrier; then libr4d's i8g_quant_row_wg (r4d_trellis_i8_fused.h, the very function
// r4d_trellis_i8_quant_act_row's kernel runs) takes the row's amax, s = amax / 127 and writes A8 + SA[row]. So A8 and SA are
// bytes r4d_trellis_i8_quant_act_row would make from the f16 A the f16 producers write. Fixed at 256 rows. Dynamic LDS:
// nout * K * 2 + 64 bytes (at most 64 KiB, checked on the host).
struct TrellisInputI8RowPtrs {
  const float* suh[kTqMaxOut];
  signed char* a8[kTqMaxOut];
  float* sa[kTqMaxOut];
};

// TransformBlockQ8's arithmetic again (x * suh in fp32, the butterfly, the product by `scale` as an fp32 value, one rounding
// to f16), returning the f16 bits instead of quantizing them; and the producers' elements as the kernels above compute
// them. (Copies, not shared helpers: moving the per-128 kernels' code into functions would move their ISA, which the
// build's comparisons pin; GPU test_trellis_input_i8 holds these to the f16 producers + the separate row quantizer.)
__device__ __forceinline__ void TransformBlockHalfBits(const float (&xv)[4], const float* __restrict__ suh,
                                                       int64_t k_off, float scale, int lane,
                                                       unsigned short (&hb)[4]) {
  float v[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) v[r] = xv[r] * suh[k_off + kTqWave * r];
  r4d_fwht128_wave(v, lane);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    float prod = v[r] * scale;
    asm("" : "+v"(prod));
    const _Float16 h = static_cast<_Float16>(prod);   // round to nearest even
    hb[r] = __builtin_bit_cast(unsigned short, h);
  }
}
__device__ __forceinline__ void SiluMulElems(const uint16_t* __restrict__ gu, int64_t intermediate, int64_t k_off,
                                             float (&xv)[4]) {
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float g = Bf16BitsToFloat(gu[k_off + kTqWave * r]);
    const float u = Bf16BitsToFloat(gu[intermediate + k_off + kTqWave * r]);
    const __hip_bfloat16 h = __float2bfloat16(g * TqSigmoid(g) * u);
    xv[r] = __bfloat162float(h);
  }
}
__device__ __forceinline__ void GateMulElems(const uint16_t* __restrict__ attn_out, const uint16_t* __restrict__ gate,
                                             int64_t off, float (&xv)[4]) {
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float a = r4dx::core::Bf16ToFloat(attn_out[off + kTqWave * r]);
    const float g = r4dx::core::Bf16ToFloat(gate[off + kTqWave * r]);
    const float sg = 1.0f / (1.0f + expf(-g));
    xv[r] = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(a * sg));
  }
}

__global__ __launch_bounds__(256) void TrellisInputI8RowKernel(const uint16_t* __restrict__ x, int64_t K, int nout,
                                                                TrellisInputI8RowPtrs p, float scale) {
  extern __shared__ __attribute__((aligned(16))) unsigned char tq_lds[];
  unsigned short* xl = reinterpret_cast<unsigned short*>(tq_lds);
  float* red = reinterpret_cast<float*>(tq_lds + static_cast<size_t>(nout) * static_cast<size_t>(K) * 2);
  const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, wave = tid >> 5;
  const int row = static_cast<int>(blockIdx.x);
  for (int kb = wave; kb < K / kTqBlock; kb += 8) {
    const int64_t k_off = static_cast<int64_t>(kb) * kTqBlock + lane;
    float xv[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) xv[r] = Bf16BitsToFloat(x[static_cast<int64_t>(row) * K + k_off + kTqWave * r]);
#pragma unroll
    for (int o = 0; o < kTqMaxOut; ++o) {
      if (o >= nout) break;   // uniform
      unsigned short hb[4];
      TransformBlockHalfBits(xv, p.suh[o], k_off, scale, lane, hb);
#pragma unroll
      for (int r = 0; r < 4; ++r) xl[static_cast<int64_t>(o) * K + k_off + kTqWave * r] = hb[r];
    }
  }
  __syncthreads();
#pragma unroll
  for (int o = 0; o < kTqMaxOut; ++o) {
    if (o >= nout) break;
    i8g_quant_row_wg(xl + static_cast<int64_t>(o) * K, static_cast<int>(K), row, p.a8[o], p.sa[o], red);
  }
}

__global__ __launch_bounds__(256) void SiluMulTrellisI8RowKernel(const uint16_t* __restrict__ gate_up,
                                                                  int64_t intermediate, int64_t in_row_stride,
                                                                  const float* __restrict__ suh,
                                                                  signed char* __restrict__ a8,
                                                                  float* __restrict__ sa, float scale) {
  extern __shared__ __attribute__((aligned(16))) unsigned char tq_lds[];
  unsigned short* xl = reinterpret_cast<unsigned short*>(tq_lds);
  float* red = reinterpret_cast<float*>(tq_lds + static_cast<size_t>(intermediate) * 2);
  const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, wave = tid >> 5;
  const int row = static_cast<int>(blockIdx.x);
  const uint16_t* gu = gate_up + static_cast<int64_t>(row) * in_row_stride;
  for (int kb = wave; kb < intermediate / kTqBlock; kb += 8) {
    const int64_t k_off = static_cast<int64_t>(kb) * kTqBlock + lane;
    float xv[4];
    SiluMulElems(gu, intermediate, k_off, xv);
    unsigned short hb[4];
    TransformBlockHalfBits(xv, suh, k_off, scale, lane, hb);
#pragma unroll
    for (int r = 0; r < 4; ++r) xl[k_off + kTqWave * r] = hb[r];
  }
  __syncthreads();
  i8g_quant_row_wg(xl, static_cast<int>(intermediate), row, a8, sa, red);
}

__global__ __launch_bounds__(256) void GateMulTrellisI8RowKernel(const uint16_t* __restrict__ attn_out,
                                                                  const uint16_t* __restrict__ gate, int64_t K,
                                                                  const float* __restrict__ suh,
                                                                  signed char* __restrict__ a8,
                                                                  float* __restrict__ sa, float scale) {
  extern __shared__ __attribute__((aligned(16))) unsigned char tq_lds[];
  unsigned short* xl = reinterpret_cast<unsigned short*>(tq_lds);
  float* red = reinterpret_cast<float*>(tq_lds + static_cast<size_t>(K) * 2);
  const int tid = static_cast<int>(threadIdx.x), lane = tid & 31, wave = tid >> 5;
  const int row = static_cast<int>(blockIdx.x);
  for (int kb = wave; kb < K / kTqBlock; kb += 8) {
    const int64_t k_off = static_cast<int64_t>(kb) * kTqBlock + lane;
    float xv[4];
    GateMulElems(attn_out, gate, static_cast<int64_t>(row) * K + k_off, xv);
    unsigned short hb[4];
    TransformBlockHalfBits(xv, suh, k_off, scale, lane, hb);
#pragma unroll
    for (int r = 0; r < 4; ++r) xl[k_off + kTqWave * r] = hb[r];
  }
  __syncthreads();
  i8g_quant_row_wg(xl, static_cast<int>(K), row, a8, sa, red);
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

// ---- fake int8 activation quantization (r4dx_fake_quant_act_f16; docs/int8-prefill.md) ----------------
// In place on a trellis linear's f16 A (the operand after the rotation): v = rint(x / s) * s, s = max|x| / 127
// over a scale group, fp32 arithmetic, written back as f16. src/model/fake_quant_act.h's FakeQuantGroupRef is
// the reference. Off the production path: only a Model with R4DX_FAKEQ_ACT set ever launches these.
constexpr int kFqRow = 1;
constexpr int kFqBlk128 = 2;
constexpr int kFqBlk32 = 3;
constexpr int kFqRowThreads = 256;

__device__ __forceinline__ float F16BitsToFloat(uint16_t b) {
  return static_cast<float>(__builtin_bit_cast(_Float16, b));
}

// |x| of an f16 value, or inf for a non-finite one (NaN included), so one max flags a group to leave alone.
__device__ __forceinline__ float FqKey(uint16_t b) {
  const float ax = fabsf(F16BitsToFloat(b));
  return ax <= 65504.0f ? ax : INFINITY;
}

// One element given its group's key max: zeros for an all-zero group, the input bits for a group with a
// non-finite element, else f16(rint(x / s) * s) with the product rounded to fp32 first (the empty asm keeps
// the multiply and the conversion apart, as TransformBlock does: a fused v_fma_mix would round once).
__device__ __forceinline__ uint16_t FqElement(uint16_t b, float amax) {
  if (amax == INFINITY) return b;
  if (amax == 0.0f) return 0;
  const float s = amax / 127.0f;
  float q = rintf(F16BitsToFloat(b) / s);
  q = fminf(fmaxf(q, -127.0f), 127.0f);
  float prod = q * s;
  asm("" : "+v"(prod));
  return __builtin_bit_cast(uint16_t, static_cast<_Float16>(prod));
}

__device__ __forceinline__ float WaveMax32(float v) {
#pragma unroll
  for (int o = kTqWave / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, kTqWave));
  return v;
}

// blk128 / blk32: grid (K / 128, rows, parts), one wave32 per 128-block of one row; lane l holds elements
// l + 32 r (r = 0..3). blk128 reduces all four, blk32 each r on its own (the lanes of one r are 32
// consecutive columns).
template <bool kBlk32>
__global__ __launch_bounds__(kTqWave) void FakeQuantBlkKernel(uint16_t* __restrict__ a, int64_t K,
                                                              int64_t part_stride) {
  const int lane = static_cast<int>(threadIdx.x);
  uint16_t* p = a + static_cast<int64_t>(blockIdx.z) * part_stride +
                static_cast<int64_t>(blockIdx.y) * K + static_cast<int64_t>(blockIdx.x) * kTqBlock + lane;
  uint16_t b[4];
  float key[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    b[r] = p[kTqWave * r];
    key[r] = FqKey(b[r]);
  }
  float amax[4];
  if (kBlk32) {
#pragma unroll
    for (int r = 0; r < 4; ++r) amax[r] = WaveMax32(key[r]);
  } else {
    const float m = WaveMax32(fmaxf(fmaxf(key[0], key[1]), fmaxf(key[2], key[3])));
#pragma unroll
    for (int r = 0; r < 4; ++r) amax[r] = m;
  }
#pragma unroll
  for (int r = 0; r < 4; ++r) p[kTqWave * r] = FqElement(b[r], amax[r]);
}

// row: grid (rows, parts), 256 threads; the row's max through LDS, then every element.
__global__ __launch_bounds__(kFqRowThreads) void FakeQuantRowKernel(uint16_t* __restrict__ a, int64_t K,
                                                                    int64_t part_stride) {
  __shared__ float red[kFqRowThreads / kTqWave];
  uint16_t* row = a + static_cast<int64_t>(blockIdx.y) * part_stride + static_cast<int64_t>(blockIdx.x) * K;
  const int t = static_cast<int>(threadIdx.x);
  float m = 0.0f;
  for (int64_t k = t; k < K; k += kFqRowThreads) m = fmaxf(m, FqKey(row[k]));
  m = WaveMax32(m);
  if ((t & (kTqWave - 1)) == 0) red[t / kTqWave] = m;
  __syncthreads();
  float amax = red[0];
#pragma unroll
  for (int w = 1; w < kFqRowThreads / kTqWave; ++w) amax = fmaxf(amax, red[w]);
  for (int64_t k = t; k < K; k += kFqRowThreads) row[k] = FqElement(row[k], amax);
}

[[noreturn]] void TqFail(const std::string& what) {
  throw std::runtime_error("r4dx_trellis_input_bf16: " + what);
}

// The fused producers' shared preconditions (r4dx_trellis_input_bf16's, for one output) and their
// transform scale.
float ProducerScale(const char* who, int64_t rows, int64_t K, int64_t suh, int64_t out,
                    int prescale_log2) {
  const auto fail = [&](const std::string& what) {
    throw std::runtime_error(std::string(who) + ": " + what);
  };
  if (K <= 0 || K % kTqBlock != 0) fail("K must be a positive multiple of 128, got " + std::to_string(K));
  if (rows < 0 || rows > 65535) fail("rows must be 0..65535, got " + std::to_string(rows));
  if (prescale_log2 < -kTqMaxPrescale || prescale_log2 > kTqMaxPrescale)
    fail("|prescale_log2| must be <= 24, got " + std::to_string(prescale_log2));
  if (suh == 0 || out == 0) fail("suh and out must be non-null");
  return static_cast<float>(std::ldexp(1.0, prescale_log2) / std::sqrt(128.0));
}

// The int8 entries' extra preconditions: the A8 layout is fixed at 256 rows, A8 / SA are device pointers.
void TqI8Check(const char* who, int64_t rows, int64_t a8, int64_t sa) {
  if (rows != kTqRowsI8) {
    throw std::runtime_error(std::string(who) + ": rows must be 256 (the int8 A8 layout), got " + std::to_string(rows));
  }
  if (a8 == 0 || sa == 0) throw std::runtime_error(std::string(who) + ": a8 and sa must be non-null");
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

void r4dx_silu_mul_trellis_bf16(int64_t gate_up, int64_t rows, int64_t intermediate,
                                 int64_t in_row_stride, int64_t suh, int64_t out, int prescale_log2,
                                 int64_t stream) {
  const float scale =
      ProducerScale("r4dx_silu_mul_trellis_bf16", rows, intermediate, suh, out, prescale_log2);
  if (gate_up == 0) throw std::runtime_error("r4dx_silu_mul_trellis_bf16: gate_up is null");
  if (in_row_stride < 2 * intermediate) {
    throw std::runtime_error("r4dx_silu_mul_trellis_bf16: in_row_stride must be >= 2 * intermediate");
  }
  if (rows == 0) return;
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(SiluMulTrellisKernel,
                     dim3(static_cast<unsigned>(intermediate / kTqBlock), static_cast<unsigned>(rows)),
                     dim3(kTqWave), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(gate_up), intermediate, in_row_stride,
                     reinterpret_cast<const float*>(suh), reinterpret_cast<uint16_t*>(out), scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_gelu_tanh_mul_trellis_bf16(int64_t gate_up, int64_t rows, int64_t intermediate,
                                      int64_t in_row_stride, int64_t suh, int64_t out,
                                      int prescale_log2, int64_t stream) {
  const float scale =
      ProducerScale("r4dx_gelu_tanh_mul_trellis_bf16", rows, intermediate, suh, out, prescale_log2);
  if (gate_up == 0) throw std::runtime_error("r4dx_gelu_tanh_mul_trellis_bf16: gate_up is null");
  if (in_row_stride < 2 * intermediate) {
    throw std::runtime_error("r4dx_gelu_tanh_mul_trellis_bf16: in_row_stride must be >= 2 * intermediate");
  }
  if (rows == 0) return;
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(GeluMulTrellisKernel,
                     dim3(static_cast<unsigned>(intermediate / kTqBlock), static_cast<unsigned>(rows)),
                     dim3(kTqWave), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(gate_up), intermediate, in_row_stride,
                     reinterpret_cast<const float*>(suh), reinterpret_cast<uint16_t*>(out), scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_attn_gate_mul_trellis_bf16(int64_t attn_out, int64_t gate, int64_t rows, int64_t K,
                                      int64_t suh, int64_t out, int prescale_log2, int64_t stream) {
  const float scale = ProducerScale("r4dx_attn_gate_mul_trellis_bf16", rows, K, suh, out, prescale_log2);
  if (attn_out == 0 || gate == 0) {
    throw std::runtime_error("r4dx_attn_gate_mul_trellis_bf16: attn_out and gate must be non-null");
  }
  if (rows == 0) return;
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(GateMulTrellisKernel,
                     dim3(static_cast<unsigned>(K / kTqBlock), static_cast<unsigned>(rows)),
                     dim3(kTqWave), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(attn_out), reinterpret_cast<const uint16_t*>(gate),
                     K, reinterpret_cast<const float*>(suh), reinterpret_cast<uint16_t*>(out), scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_trellis_input_i8(int64_t x, int64_t M, int64_t K, int nout, const int64_t* suh, const int64_t* a8,
                            const int64_t* sa, int prescale_log2, int64_t stream) {
  if (nout < 1 || nout > kTqMaxOut) TqFail("nout must be 1..3, got " + std::to_string(nout));
  if (K <= 0 || K % kTqBlock != 0) TqFail("K must be a positive multiple of 128, got " + std::to_string(K));
  if (prescale_log2 < -kTqMaxPrescale || prescale_log2 > kTqMaxPrescale)
    TqFail("|prescale_log2| must be <= 24, got " + std::to_string(prescale_log2));
  if (suh == nullptr || a8 == nullptr || sa == nullptr) TqFail("suh, a8 and sa must be host arrays of nout pointers");
  TrellisInputI8Ptrs p{};
  for (int o = 0; o < nout; ++o) {
    TqI8Check("r4dx_trellis_input_i8", M, a8[o], sa[o]);
    if (suh[o] == 0) TqFail("suh[" + std::to_string(o) + "] is null");
    p.suh[o] = reinterpret_cast<const float*>(suh[o]);
    p.a8[o] = reinterpret_cast<signed char*>(a8[o]);
    p.sa[o] = reinterpret_cast<float*>(sa[o]);
  }
  if (x == 0) TqFail("x is null");
  const float scale = static_cast<float>(std::ldexp(1.0, prescale_log2) / std::sqrt(128.0));
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(TrellisInputI8Kernel, dim3(static_cast<unsigned>(K / kTqBlock), static_cast<unsigned>(M)),
                     dim3(kTqWave), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(x),
                     K, nout, p, scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_silu_mul_trellis_i8(int64_t gate_up, int64_t rows, int64_t intermediate, int64_t in_row_stride,
                               int64_t suh, int64_t a8, int64_t sa, int prescale_log2, int64_t stream) {
  const float scale = ProducerScale("r4dx_silu_mul_trellis_i8", rows, intermediate, suh, a8, prescale_log2);
  TqI8Check("r4dx_silu_mul_trellis_i8", rows, a8, sa);
  if (gate_up == 0) throw std::runtime_error("r4dx_silu_mul_trellis_i8: gate_up is null");
  if (in_row_stride < 2 * intermediate) {
    throw std::runtime_error("r4dx_silu_mul_trellis_i8: in_row_stride must be >= 2 * intermediate");
  }
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(SiluMulTrellisI8Kernel,
                     dim3(static_cast<unsigned>(intermediate / kTqBlock), static_cast<unsigned>(rows)),
                     dim3(kTqWave), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(gate_up), intermediate, in_row_stride,
                     reinterpret_cast<const float*>(suh), reinterpret_cast<signed char*>(a8),
                     reinterpret_cast<float*>(sa), scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_attn_gate_mul_trellis_i8(int64_t attn_out, int64_t gate, int64_t rows, int64_t K, int64_t suh,
                                    int64_t a8, int64_t sa, int prescale_log2, int64_t stream) {
  const float scale = ProducerScale("r4dx_attn_gate_mul_trellis_i8", rows, K, suh, a8, prescale_log2);
  TqI8Check("r4dx_attn_gate_mul_trellis_i8", rows, a8, sa);
  if (attn_out == 0 || gate == 0) {
    throw std::runtime_error("r4dx_attn_gate_mul_trellis_i8: attn_out and gate must be non-null");
  }
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(GateMulTrellisI8Kernel, dim3(static_cast<unsigned>(K / kTqBlock), static_cast<unsigned>(rows)),
                     dim3(kTqWave), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(attn_out), reinterpret_cast<const uint16_t*>(gate), K,
                     reinterpret_cast<const float*>(suh), reinterpret_cast<signed char*>(a8),
                     reinterpret_cast<float*>(sa), scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

// The per-row entries: the same preconditions as the per-128 ones, and the row buffer in LDS has to fit (64 KiB).
static void TqI8RowCheck(const char* who, int nout, int64_t K) {
  if (static_cast<size_t>(nout) * static_cast<size_t>(K) * 2 + 64 > 65536) {
    throw std::runtime_error(std::string(who) + ": nout * K * 2 + 64 bytes of LDS exceed 64 KiB (nout " +
                             std::to_string(nout) + ", K " + std::to_string(K) + ")");
  }
}

void r4dx_trellis_input_i8r(int64_t x, int64_t M, int64_t K, int nout, const int64_t* suh, const int64_t* a8,
                             const int64_t* sa, int prescale_log2, int64_t stream) {
  if (nout < 1 || nout > kTqMaxOut) TqFail("nout must be 1..3, got " + std::to_string(nout));
  if (K <= 0 || K % kTqBlock != 0) TqFail("K must be a positive multiple of 128, got " + std::to_string(K));
  if (prescale_log2 < -kTqMaxPrescale || prescale_log2 > kTqMaxPrescale)
    TqFail("|prescale_log2| must be <= 24, got " + std::to_string(prescale_log2));
  if (suh == nullptr || a8 == nullptr || sa == nullptr) TqFail("suh, a8 and sa must be host arrays of nout pointers");
  TqI8RowCheck("r4dx_trellis_input_i8r", nout, K);
  TrellisInputI8RowPtrs p{};
  for (int o = 0; o < nout; ++o) {
    TqI8Check("r4dx_trellis_input_i8r", M, a8[o], sa[o]);
    if (suh[o] == 0) TqFail("suh[" + std::to_string(o) + "] is null");
    p.suh[o] = reinterpret_cast<const float*>(suh[o]);
    p.a8[o] = reinterpret_cast<signed char*>(a8[o]);
    p.sa[o] = reinterpret_cast<float*>(sa[o]);
  }
  if (x == 0) TqFail("x is null");
  const float scale = static_cast<float>(std::ldexp(1.0, prescale_log2) / std::sqrt(128.0));
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(TrellisInputI8RowKernel, dim3(static_cast<unsigned>(M)), dim3(256),
                     static_cast<size_t>(nout) * static_cast<size_t>(K) * 2 + 64, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(x), K, nout, p, scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_silu_mul_trellis_i8r(int64_t gate_up, int64_t rows, int64_t intermediate, int64_t in_row_stride,
                                int64_t suh, int64_t a8, int64_t sa, int prescale_log2, int64_t stream) {
  const float scale = ProducerScale("r4dx_silu_mul_trellis_i8r", rows, intermediate, suh, a8, prescale_log2);
  TqI8Check("r4dx_silu_mul_trellis_i8r", rows, a8, sa);
  TqI8RowCheck("r4dx_silu_mul_trellis_i8r", 1, intermediate);
  if (gate_up == 0) throw std::runtime_error("r4dx_silu_mul_trellis_i8r: gate_up is null");
  if (in_row_stride < 2 * intermediate) {
    throw std::runtime_error("r4dx_silu_mul_trellis_i8r: in_row_stride must be >= 2 * intermediate");
  }
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(SiluMulTrellisI8RowKernel, dim3(static_cast<unsigned>(rows)), dim3(256),
                     static_cast<size_t>(intermediate) * 2 + 64, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(gate_up), intermediate, in_row_stride,
                     reinterpret_cast<const float*>(suh), reinterpret_cast<signed char*>(a8),
                     reinterpret_cast<float*>(sa), scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_attn_gate_mul_trellis_i8r(int64_t attn_out, int64_t gate, int64_t rows, int64_t K, int64_t suh,
                                     int64_t a8, int64_t sa, int prescale_log2, int64_t stream) {
  const float scale = ProducerScale("r4dx_attn_gate_mul_trellis_i8r", rows, K, suh, a8, prescale_log2);
  TqI8Check("r4dx_attn_gate_mul_trellis_i8r", rows, a8, sa);
  TqI8RowCheck("r4dx_attn_gate_mul_trellis_i8r", 1, K);
  if (attn_out == 0 || gate == 0) {
    throw std::runtime_error("r4dx_attn_gate_mul_trellis_i8r: attn_out and gate must be non-null");
  }
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(GateMulTrellisI8RowKernel, dim3(static_cast<unsigned>(rows)), dim3(256),
                     static_cast<size_t>(K) * 2 + 64, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(attn_out), reinterpret_cast<const uint16_t*>(gate), K,
                     reinterpret_cast<const float*>(suh), reinterpret_cast<signed char*>(a8),
                     reinterpret_cast<float*>(sa), scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_fake_quant_act_f16(int64_t a, int64_t rows, int64_t K, int parts, int64_t part_stride, int mode,
                              int64_t stream) {
  const auto fail = [](const std::string& what) {
    throw std::runtime_error("r4dx_fake_quant_act_f16: " + what);
  };
  if (mode < kFqRow || mode > kFqBlk32) fail("mode must be 1 (row), 2 (blk128) or 3 (blk32), got " + std::to_string(mode));
  if (K <= 0 || K % kTqBlock != 0) fail("K must be a positive multiple of 128, got " + std::to_string(K));
  if (rows < 0 || rows > 65535) fail("rows must be 0..65535, got " + std::to_string(rows));
  if (parts < 1 || parts > 2) fail("parts must be 1 or 2, got " + std::to_string(parts));
  if (parts > 1 && part_stride < rows * K) fail("part_stride must be >= rows * K for two parts");
  if (a == 0) fail("a is null");
  if (rows == 0) return;
  r4dx_kernel_launch_counter_add(1);
  hipStream_t s = reinterpret_cast<hipStream_t>(stream);
  uint16_t* p = reinterpret_cast<uint16_t*>(a);
  if (mode == kFqRow) {
    hipLaunchKernelGGL(FakeQuantRowKernel, dim3(static_cast<unsigned>(rows), static_cast<unsigned>(parts)),
                       dim3(kFqRowThreads), 0, s, p, K, part_stride);
  } else {
    const dim3 grid(static_cast<unsigned>(K / kTqBlock), static_cast<unsigned>(rows), static_cast<unsigned>(parts));
    if (mode == kFqBlk32) {
      hipLaunchKernelGGL(FakeQuantBlkKernel<true>, grid, dim3(kTqWave), 0, s, p, K, part_stride);
    } else {
      hipLaunchKernelGGL(FakeQuantBlkKernel<false>, grid, dim3(kTqWave), 0, s, p, K, part_stride);
    }
  }
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
