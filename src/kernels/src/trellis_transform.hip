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
#include "r4d_gdn_wmma.h"
#include "r4d_trellis_dq.h"
#include "r4dx/kernels/lfm2_kernels.h"
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


// ---- LFM2 grouped MoE decode: device-indexed trellis input transform + GEMM (L4-3) ----------------------
constexpr int kTqMoeMaxPairs = 16384;
constexpr int kTqMoeMaxExperts = 256;

__global__ __launch_bounds__(kTqWave) void TrellisMoeInputKernel(
    const uint16_t* __restrict__ x, const uint64_t* __restrict__ suh_ptrs,
    const int32_t* __restrict__ ids, uint16_t* __restrict__ a0_out, uint16_t* __restrict__ a1_out,
    int a_row_group, int n_experts, int64_t K, int parts, float scale) {
  const int p = static_cast<int>(blockIdx.y);
  const int e = ids[p];
  if (e < 0 || e >= n_experts) return;
  const float* suh = reinterpret_cast<const float*>(suh_ptrs[e]);
  if (!suh) return;
  const int lane = static_cast<int>(threadIdx.x);
  const int64_t k_off = static_cast<int64_t>(blockIdx.x) * kTqBlock + lane;
  const int64_t x_off = static_cast<int64_t>(p / a_row_group) * K + k_off;
  const int64_t out_off = static_cast<int64_t>(p) * K + k_off;
  float xv[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) xv[r] = Bf16BitsToFloat(x[x_off + kTqWave * r]);
  TransformBlock(xv, suh, a0_out, out_off, k_off, scale, lane);
  if (parts > 1) {
    TransformBlock(xv, suh + K, a1_out, out_off, k_off, scale, lane);
  }
}

// Mirror of third_party/libr4d/r4d_gemm_trellis_nt_m64.hip's r4d_gemm_trellis_nt_m64_kernel at M = 1,
// MT = 1, full = 1, WQ = false, with pair_idx = blockIdx.z indexing ids[pair_idx] -> w_ptrs[e], svh_ptrs[e]
// and per-pair disjoint A0, A1, C, ws and self-resetting tickets.
typedef __attribute__((address_space(1))) const unsigned char* tq_gbyte;

template <typename T, bool NT>
__device__ __forceinline__ T tq_moe_load(tq_gbyte base, unsigned off, int imm) {
  auto p = (__attribute__((address_space(1))) const T*)(base + off + imm);
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

template <int KB, int NP, int U>
struct tq_moe_stage;
template <int NP, int U>
struct tq_moe_stage<4, NP, U> {
  unsigned long long ab[NP][U];
  unsigned p[NP][U];
  v8h af[U];
};
template <int NP, int U>
struct tq_moe_stage<5, NP, U> {
  unsigned w[NP][U][5];
  v8h af[U];
};

static constexpr bool tq_moe_prefetch(int KB, int NP, int U) {
  return 16 * NP + 2 * ((KB == 4 ? 3 : 5) * NP * U + 4 * U) <= 128;
}

__device__ __forceinline__ void tq_moe_opaque(tq_gbyte& p) {
  unsigned long long v = (unsigned long long)p;
  asm("" : "+s"(v));
  p = (tq_gbyte)v;
}

__device__ __forceinline__ void tq_moe_opaque_v(unsigned& v) { asm("" : "+v"(v)); }

__device__ __forceinline__ v8f tq_moe_mma(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ unsigned short tq_moe_bf16_rn(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (unsigned short)((u >> 16) | 0x40u);
  return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ void tq_moe_sync_lds() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

__device__ __forceinline__ void tq_moe_load_svh(float (&sv)[4], const float* __restrict__ svh, int n0,
                                                int lane) {
#pragma unroll
  for (int r = 0; r < 4; ++r) sv[r] = svh[n0 + lane + 32 * r];
}

__device__ __forceinline__ void tq_moe_finish(float (&v)[4], int lane, const float (&sv)[4],
                                              unsigned short* __restrict__ crow, int n0,
                                              float out_scale) {
  r4d_fwht128_wave(v, lane);
#pragma unroll
  for (int r = 0; r < 4; ++r) crow[n0 + lane + 32 * r] = tq_moe_bf16_rn((v[r] * sv[r]) * out_scale);
}

template <int KB, int NP, int U, bool NT>
__global__ __launch_bounds__(1024) void TrellisMoeDecodeGemmKernel(
    const unsigned short* __restrict__ A0_all, const unsigned short* __restrict__ A1_all, int n_split,
    const uint64_t* __restrict__ w_ptrs, const uint64_t* __restrict__ svh_ptrs,
    const int32_t* __restrict__ ids, unsigned short* __restrict__ Cb_all, float* __restrict__ ws_all,
    unsigned* __restrict__ tickets_all, int n_experts, int K, int N, int WV, int SK, int SKG,
    float out_scale) {
  constexpr int BLK = 64 * KB;
  extern __shared__ float red[];
  const int pair_idx = static_cast<int>(blockIdx.z);
  const int e = ids[pair_idx];
  if (e < 0 || e >= n_experts) return;
  const unsigned* W = reinterpret_cast<const unsigned*>(w_ptrs[e]);
  const float* svh = reinterpret_cast<const float*>(svh_ptrs[e]);
  if (!W || !svh) return;
  const unsigned short* A0 = A0_all + static_cast<size_t>(pair_idx) * static_cast<size_t>(K);
  const unsigned short* A1 = A1_all ? (A1_all + static_cast<size_t>(pair_idx) * static_cast<size_t>(K)) : nullptr;
  unsigned short* Cb = Cb_all + static_cast<size_t>(pair_idx) * static_cast<size_t>(N);
  float* ws = ws_all ? (ws_all + static_cast<size_t>(pair_idx) * static_cast<size_t>(SKG) * static_cast<size_t>(N))
                     : nullptr;
  unsigned* tickets =
      tickets_all ? (tickets_all + static_cast<size_t>(pair_idx) * static_cast<size_t>(N / 128)) : nullptr;

  const int tid = threadIdx.x, lane = tid % kTqWave;
  const int wave = __builtin_amdgcn_readfirstlane(tid / kTqWave);
  const int col_local = wave / SK, ksplit = wave - col_local * SK;
  const int Wc = WV * NP * 32;
  const int n0 = blockIdx.x * Wc;
  const int pair0 = n0 / 32 + col_local * NP;
  const int kt_total = K / 16;
  const int ktw = kt_total / (SK * SKG);
  const int kt0 = ((int)blockIdx.y * SK + ksplit) * ktw;
  const int steps = ktw / U;
  const unsigned short* A = n0 >= n_split ? A1 : A0;

  v8f acc[NP][2];
#pragma unroll
  for (int p = 0; p < NP; ++p)
#pragma unroll
    for (int f = 0; f < 2; ++f) acc[p][f] = (v8f)(0.f);

  {
    tq_gbyte wb[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p)
      wb[p] = (tq_gbyte)(const unsigned char*)W + ((size_t)(pair0 + p) * kt_total + kt0) * BLK;
    tq_gbyte abase = (tq_gbyte)(const unsigned char*)A + (size_t)kt0 * 32;
    constexpr int NOFF = KB == 4 ? 2 : 5;
    unsigned woff[NOFF];
    if constexpr (KB == 4) {
      woff[0] = r4d_trellis_k4_off_ab(lane);
      woff[1] = r4d_trellis_k4_off_p(lane);
    } else {
#pragma unroll
      for (int i = 0; i < 5; ++i) woff[i] = r4d_trellis_k5_word(lane, i) * 4u;
    }
    const unsigned k5sh = r4d_trellis_k5_shift(lane);
    unsigned aoff = rowGo((unsigned)K, 0, 0, lane) * 2u;

    auto load = [&](tq_moe_stage<KB, NP, U>& s, int ahead) {
      const size_t wo = (size_t)ahead * (U * BLK), ao = (size_t)ahead * (U * 32);
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        tq_gbyte b = wb[p] + wo;
        tq_moe_opaque(b);
#pragma unroll
        for (int u = 0; u < U; ++u) {
          if constexpr (KB == 4) {
            s.ab[p][u] = tq_moe_load<unsigned long long, NT>(b, woff[0], u * BLK);
            s.p[p][u] = tq_moe_load<unsigned, NT>(b, woff[1], u * BLK);
          } else {
#pragma unroll
            for (int i = 0; i < 5; ++i) s.w[p][u][i] = tq_moe_load<unsigned, NT>(b, woff[i], u * BLK);
          }
        }
      }
      tq_gbyte a = abase + ao;
      tq_moe_opaque(a);
#pragma unroll
      for (int u = 0; u < U; ++u) {
        union { unsigned long long w[2]; v8h f; } x;
        x.w[0] = tq_moe_load<unsigned long long, false>(a, aoff, u * 32);
        x.w[1] = tq_moe_load<unsigned long long, false>(a, aoff, u * 32 + 16);
        s.af[u] = x.f;
      }
    };
    auto advance = [&](int steps_done) {
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        wb[p] += (size_t)steps_done * (U * BLK);
        tq_moe_opaque(wb[p]);
      }
      abase += (size_t)steps_done * (U * 32);
      tq_moe_opaque(abase);
#pragma unroll
      for (int i = 0; i < NOFF; ++i) tq_moe_opaque_v(woff[i]);
      tq_moe_opaque_v(aoff);
    };
    auto compute = [&](tq_moe_stage<KB, NP, U>& s) {
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int p = 0; p < NP; ++p) {
          v8h f0, f1;
          if constexpr (KB == 4)
            r4d_trellis_k4_decode(s.p[p][u], (unsigned)s.ab[p][u], (unsigned)(s.ab[p][u] >> 32), f0, f1);
          else
            r4d_trellis_k5_decode(s.w[p][u], k5sh, f0, f1);
          acc[p][0] = tq_moe_mma(s.af[u], f0, acc[p][0]);
          acc[p][1] = tq_moe_mma(s.af[u], f1, acc[p][1]);
        }
    };
    if constexpr (tq_moe_prefetch(KB, NP, U)) {
      tq_moe_stage<KB, NP, U> s0, s1;
      load(s0, 0);
      int st = 0;
      for (; st + 1 < steps; st += 2) {
        load(s1, 1);
        __builtin_amdgcn_sched_barrier(0);
        compute(s0);
        load(s0, __builtin_amdgcn_readfirstlane(st + 2 < steps ? 2 : 1));
        __builtin_amdgcn_sched_barrier(0);
        compute(s1);
        advance(2);
      }
      if (st < steps) compute(s0);
    } else {
      for (int st = 0; st < steps; ++st) {
        tq_moe_stage<KB, NP, U> s0;
        load(s0, 0);
        compute(s0);
        advance(1);
      }
    }
  }

  const int nwaves = WV * SK, nthreads = nwaves * kTqWave;
  const bool split = SKG > 1 || Wc < 128;
  const int span = Wc > 128 ? Wc : 128;
  const int g0 = n0 / span * span;
  const int groups = span / 128;
  const int gw = wave % groups, rstep = nwaves / groups;
  float sv[4] = {0.f, 0.f, 0.f, 0.f};
  tq_moe_load_svh(sv, svh, g0 + gw * 128, lane);
  float* dst = split ? ws + (size_t)blockIdx.y * N : nullptr;
  const int t = (lane >> 3) & 1, c = lane & 7, hh_lane = lane >> 4;

  tq_moe_sync_lds();
  if (hh_lane == 0) {
#pragma unroll
    for (int p = 0; p < NP; ++p)
#pragma unroll
      for (int f = 0; f < 2; ++f) {
        const int col = (col_local * NP + p) * 32 + 16 * t + 8 * f + c;
#pragma unroll
        for (int el = 0; el < 8; ++el) red[(ksplit * 8 + el) * Wc + col] = acc[p][f][el];
      }
  }
  tq_moe_sync_lds();
  const bool keep = !split;
  for (int idx = tid; idx < 8 * Wc; idx += nthreads) {
    const int r = idx / Wc, n = idx - r * Wc;
    if (r >= 1) continue;
    float v = 0.f;
    for (int s = 0; s < SK; ++s) v += red[(s * 8 + r) * Wc + n];
    if (keep) red[r * Wc + n] = v;
    else dst[n0 + n] = v;
  }
  if (keep) {
    tq_moe_sync_lds();
    for (int r = wave / groups; r < 1; r += rstep) {
      float v[4];
#pragma unroll
      for (int q = 0; q < 4; ++q) v[q] = red[r * Wc + gw * 128 + lane + 32 * q];
      tq_moe_finish(v, lane, sv, Cb + (size_t)r * N, n0 + gw * 128, out_scale);
    }
  }
  if (!split) return;

  const unsigned contributors = (unsigned)(SKG * (span / Wc));
  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const unsigned old = atomicAdd(&tickets[g0 / 128], 1u);
    const bool last = old == contributors - 1;
    if (last) tickets[g0 / 128] = 0u;
    red[0] = last ? 1.f : 0.f;
  }
  __syncthreads();
  if (red[0] == 0.f) return;
  __threadfence();
  const int gn = g0 + gw * 128;
  for (int m = wave / groups; m < 1; m += rstep) {
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int n = gn + lane + 32 * q;
      float x = 0.f;
      for (int y = 0; y < SKG; ++y) x += ws[(size_t)y * N + n];
      v[q] = x;
    }
    tq_moe_finish(v, lane, sv, Cb + (size_t)m * N, gn, out_scale);
  }
}

template <int KB>
void LaunchTrellisMoeDecode(int64_t a0_f16, int64_t a1_f16, int n_split, int64_t w_ptrs, int64_t svh_ptrs,
                            int64_t ids, int64_t c_bf16, int64_t ws_f32, int64_t tickets_u32, int n_pairs,
                            int n_experts, int K, int N, int WV, int SK, int NP, int SKG, int U, int NT,
                            float out_scale, int64_t stream) {
  const int Wc = WV * NP * 32;
  const dim3 grid(static_cast<unsigned>(N / Wc), static_cast<unsigned>(SKG), static_cast<unsigned>(n_pairs));
  const dim3 block(static_cast<unsigned>(WV * SK * kTqWave));
  const size_t smem = static_cast<size_t>(SK) * 8 * Wc * sizeof(float);
  auto A0 = reinterpret_cast<const unsigned short*>(a0_f16);
  auto A1 = reinterpret_cast<const unsigned short*>(a1_f16);
  auto Wp = reinterpret_cast<const uint64_t*>(w_ptrs);
  auto Sp = reinterpret_cast<const uint64_t*>(svh_ptrs);
  auto Idp = reinterpret_cast<const int32_t*>(ids);
  auto Cp = reinterpret_cast<unsigned short*>(c_bf16);
  auto Wsp = reinterpret_cast<float*>(ws_f32);
  auto Tkp = reinterpret_cast<unsigned*>(tickets_u32);
  hipStream_t st = reinterpret_cast<hipStream_t>(stream);
#define TQ_MOE_LAUNCH(UU, TT)                                                                          \
  hipLaunchKernelGGL((TrellisMoeDecodeGemmKernel<KB, 1, UU, TT>), grid, block, smem, st, A0, A1,       \
                     n_split, Wp, Sp, Idp, Cp, Wsp, Tkp, n_experts, K, N, WV, SK, SKG, out_scale)
  if (NT) {
    if (U == 1) TQ_MOE_LAUNCH(1, true);
    else if (U == 2) TQ_MOE_LAUNCH(2, true);
    else TQ_MOE_LAUNCH(4, true);
  } else {
    if (U == 1) TQ_MOE_LAUNCH(1, false);
    else if (U == 2) TQ_MOE_LAUNCH(2, false);
    else TQ_MOE_LAUNCH(4, false);
  }
#undef TQ_MOE_LAUNCH
  R4DX_HIP_CHECK(hipGetLastError());
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


void r4dx_lfm2_trellis_moe_input_bf16(int64_t x_bf16, int64_t suh_ptrs, int64_t ids, int64_t a0_f16,
                                      int64_t a1_f16, int n_pairs, int a_row_group, int n_experts,
                                      int64_t K, int parts, int prescale_log2, int64_t stream) {
  const auto fail = [](const std::string& what) {
    throw std::runtime_error("r4dx_lfm2_trellis_moe_input_bf16: " + what);
  };
  if (n_pairs < 1 || n_pairs > kTqMoeMaxPairs) fail("n_pairs must be in [1, 16384]");
  if (a_row_group < 1 || a_row_group > n_pairs || n_pairs % a_row_group != 0)
    fail("a_row_group must divide n_pairs in [1, n_pairs]");
  if (n_experts < 1 || n_experts > kTqMoeMaxExperts) fail("n_experts must be in [1, 256]");
  if (K <= 0 || K % kTqBlock != 0) fail("K must be a positive multiple of 128");
  if (parts < 1 || parts > 2) fail("parts must be 1 or 2");
  if (prescale_log2 < -kTqMaxPrescale || prescale_log2 > kTqMaxPrescale) fail("|prescale_log2| <= 24");
  if (!x_bf16 || !suh_ptrs || !ids || !a0_f16 || (parts > 1 && !a1_f16)) fail("null pointer");
  const float scale = static_cast<float>(std::ldexp(1.0, prescale_log2) / std::sqrt(128.0));
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(TrellisMoeInputKernel,
                     dim3(static_cast<unsigned>(K / kTqBlock), static_cast<unsigned>(n_pairs)),
                     dim3(kTqWave), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(x_bf16), reinterpret_cast<const uint64_t*>(suh_ptrs),
                     reinterpret_cast<const int32_t*>(ids), reinterpret_cast<uint16_t*>(a0_f16),
                     reinterpret_cast<uint16_t*>(a1_f16), a_row_group, n_experts, K, parts, scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_lfm2_gemm_trellis_moe_decode(int64_t a0_f16, int64_t a1_f16, int n_split, int64_t w_ptrs,
                                       int64_t svh_ptrs, int64_t ids, int64_t c_bf16, int64_t ws_f32,
                                       int64_t tickets_u32, int n_pairs, int n_experts, int K, int N,
                                       int bits, int WV, int SK, int NP, int SKG, int U, int NT,
                                       float out_scale, int64_t stream) {
  const auto fail = [](const std::string& what) {
    throw std::runtime_error("r4dx_lfm2_gemm_trellis_moe_decode: " + what);
  };
  if (n_pairs < 1 || n_pairs > kTqMoeMaxPairs) fail("n_pairs must be in [1, 16384]");
  if (n_experts < 1 || n_experts > kTqMoeMaxExperts) fail("n_experts must be in [1, 256]");
  if (bits != 4 && bits != 5) fail("bits must be 4 or 5");
  if (NP != 1) fail("NP must be 1");
  if (U != 1 && U != 2 && U != 4) fail("U must be 1, 2 or 4");
  if (K < 128 || N < 128 || K % 128 != 0 || N % 128 != 0) fail("K and N must be multiples of 128");
  const int Wc = WV * NP * 32;
  if (WV < 1 || WV > 4 || (WV & (WV - 1)) != 0 || Wc > 256 || N % Wc != 0) fail("bad WV / Wc");
  if (SK < 1 || SK > 16 || (SK & (SK - 1)) != 0 || WV * SK * kTqWave > 1024) fail("bad SK");
  if (SKG < 1 || SKG > 8 || (SKG & (SKG - 1)) != 0) fail("bad SKG");
  if ((K / 16) % (SK * SKG * U) != 0) fail("K/16 not divisible by SK*SKG*U");
  if (static_cast<size_t>(SK) * Wc * 32 > 64 * 1024) fail("LDS > 64 KiB");
  if (n_split < 0 || n_split > N || n_split % Wc != 0) fail("bad n_split");
  const bool split = SKG > 1 || Wc < 128;
  if (!a0_f16 || (n_split < N && !a1_f16) || !w_ptrs || !svh_ptrs || !ids || !c_bf16 ||
      (split && (!ws_f32 || !tickets_u32))) {
    fail("null pointer");
  }
  r4dx_kernel_launch_counter_add(1);
  if (bits == 4) {
    LaunchTrellisMoeDecode<4>(a0_f16, a1_f16, n_split, w_ptrs, svh_ptrs, ids, c_bf16, ws_f32, tickets_u32,
                              n_pairs, n_experts, K, N, WV, SK, NP, SKG, U, NT, out_scale, stream);
  } else {
    LaunchTrellisMoeDecode<5>(a0_f16, a1_f16, n_split, w_ptrs, svh_ptrs, ids, c_bf16, ws_f32, tickets_u32,
                              n_pairs, n_experts, K, N, WV, SK, NP, SKG, U, NT, out_scale, stream);
  }
}
}  // extern "C"
