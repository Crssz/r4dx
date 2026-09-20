// r4dx_kernels.hip -- every r4dx-owned device kernel (docs/architecture.md "Own kernels"),
// compiled as one hipcc translation unit (mirrors third_party/CMakeLists.txt's per-unit recipe
// for r4d_core; see docs/build-windows.md). One row per workgroup for the row-reduction kernels
// (rmsnorm, quant_act_fp8), following r4d_quant_act_i8.hip's own convention in third_party/libr4d.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4d.h"  // third_party/libr4d: r4d_dflash_conv_t2_g16_bf16 (the DFlash2 conv wrapper below)
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using r4dx::core::FloatToFp8E4M3;
using r4dx::core::FloatToF16;

namespace {

constexpr int kThreads = 256;

// Kernel launch counter (docs/r9700.md P2/task item 4) -- see kernels.h's doc comment. Plain
// global, not atomic: Model is single-worker-thread (model.h's own SCOPE comment), matching
// linear.cpp's PickTuning cache's identical no-lock reasoning.
int64_t g_launch_count = 0;

__device__ __forceinline__ float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---- P6 vectorized-load helpers (docs/r9700.md P6, C24) ---------------------------------------
// gfx1201 has no v_cvt_pk_bf16_f32 (C24); packing two bf16 halves into one dword is done by hand.
// Unlike third_party/libr4d/r4d_gdn_wmma.h's f2bf2 (round-half-away, 2 adds + v_perm_b32), this
// pack keeps each half going through the SAME __float2bfloat16 (RTNE) the pre-P6 scalar kernels
// used, so the elementwise kernels/passes below (silu_mul, residual_rmsnorm's residual-add half)
// stay bit-exact against the pre-P6 golden (tests/kernels/test_kernel_bandwidth.cpp) -- the actual
// P6 win here is fewer, wider (16-byte) load/store instructions, not a cheaper convert. A
// round-half-away pack would be a legitimate ALU-cost win for the reduction-only outputs
// (rmsnorm's own output, residual_rmsnorm's normed output -- see that test's "Tolerance policy"
// header comment for why those are allowed to move by a last bf16 ulp already), but was not worth
// the extra code path for this pass; unpack/pack below is branch-free and inlines to a handful of
// VALU ops either way.
// NOTE: __hip_bfloat16's own `(unsigned short)` constructor and `operator unsigned short()`
// (amd_hip_bf16.h:187, 258) are VALUE conversions (static_cast<__bf16>(val), i.e. "the bf16
// nearest to the integer value val"), not bit reinterpretation -- using them here silently
// produces garbage (confirmed: an earlier version of this helper used them and every element
// differed from the pre-P6 golden). __ushort_as_bfloat16/__bfloat16_as_ushort
// (amd_hip_bf16.h:487, 618) are the bit-preserving pair, which is what packing/unpacking a raw
// bf16 payload out of a loaded dword needs.
__device__ __forceinline__ void UnpackBf16x2(unsigned dw, __hip_bfloat16* lo, __hip_bfloat16* hi) {
  *lo = __ushort_as_bfloat16(static_cast<unsigned short>(dw & 0xFFFFu));
  *hi = __ushort_as_bfloat16(static_cast<unsigned short>(dw >> 16));
}
__device__ __forceinline__ unsigned PackBf16x2(__hip_bfloat16 lo, __hip_bfloat16 hi) {
  const unsigned lo16 = __bfloat16_as_ushort(lo);
  const unsigned hi16 = __bfloat16_as_ushort(hi);
  return (hi16 << 16) | lo16;
}

__device__ __forceinline__ float BlockReduceSum(float v) {
  __shared__ float red[kThreads / 32];
  for (int o = 16; o; o >>= 1) v += __shfl_xor(v, o, 32);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float out = 0.0f;
  if (threadIdx.x < (blockDim.x + 31) / 32) out = red[threadIdx.x];
  if (threadIdx.x < 32) {
    for (int o = 16; o; o >>= 1) out += __shfl_xor(out, o, 32);
  }
  __shared__ float total;
  if (threadIdx.x == 0) total = out;
  __syncthreads();
  return total;
}

__device__ __forceinline__ float BlockReduceMax(float v) {
  __shared__ float red[kThreads / 32];
  for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float out = 0.0f;
  if (threadIdx.x < (blockDim.x + 31) / 32) out = red[threadIdx.x];
  if (threadIdx.x < 32) {
    for (int o = 16; o; o >>= 1) out = fmaxf(out, __shfl_xor(out, o, 32));
  }
  __shared__ float total;
  if (threadIdx.x == 0) total = out;
  __syncthreads();
  return total;
}

// ---- fused activation-quant epilogue (docs/r9700.md R2/P2) -------------------------------------
// Applied AFTER a producer kernel has fully written its bf16 output row to global memory (caller
// must __syncthreads() first: see call sites below), re-reading that row and running the IDENTICAL
// algorithm the corresponding standalone kernel uses, so the result is byte-identical to running
// that kernel separately on the same bf16 values (verified: tests/kernels/test_fused_quant.cpp).
// row_bf16: this row's just-written bf16 output, [K] contiguous. epilogue: an r4dx_epilogue value
// (kernels.h). epilogue_out: destination row, [K] contiguous, dtype selected by `epilogue`.
// epilogue_scale_row: this row's single fp32 scale slot (unused for r4dx_epilogue_f16).
__device__ __forceinline__ void ApplyEpilogueRow(const __hip_bfloat16* __restrict__ row_bf16,
                                                   int epilogue, void* __restrict__ epilogue_out,
                                                   float* __restrict__ epilogue_scale_row,
                                                   int64_t K) {
  if (epilogue == r4dx_epilogue_none) {
    return;
  } else if (epilogue == r4dx_epilogue_f16) {
    // Elementwise, no reduction -- byte-identical to r4dx_model_cast_bf16_to_f16's
    // CastBf16ToF16Kernel (src/model/kernels/model_kernels.hip), which calls the same
    // FloatToF16(Bf16ToFloat(v)) helper pair on the same bf16 value.
    uint16_t* __restrict__ out16 = reinterpret_cast<uint16_t*>(epilogue_out);
    for (int64_t i = threadIdx.x; i < K; i += kThreads) {
      out16[i] = FloatToF16(__bfloat162float(row_bf16[i]));
    }
  } else if (epilogue == r4dx_epilogue_fp8_e4m3_row) {
    // Byte-identical to QuantActFp8Kernel below (this file) / r4dx_quant_act_fp8e4m3_row: same
    // BlockReduceMax, same scale = max(1e-8, absmax)/448, same FloatToFp8E4M3(v*inv) quantize,
    // plain row-major byte order (no fragment permute -- matches the mxfp4 GEMM's flat A read).
    uint8_t* __restrict__ outq = reinterpret_cast<uint8_t*>(epilogue_out);
    float m = 0.0f;
    for (int64_t i = threadIdx.x; i < K; i += kThreads) {
      m = fmaxf(m, fabsf(__bfloat162float(row_bf16[i])));
    }
    m = BlockReduceMax(m);
    const float scale = fmaxf(m, 1e-8f) / 448.0f;
    const float inv = 1.0f / scale;
    if (threadIdx.x == 0) epilogue_scale_row[0] = scale;
    for (int64_t i = threadIdx.x; i < K; i += kThreads) {
      outq[i] = FloatToFp8E4M3(__bfloat162float(row_bf16[i]) * inv);
    }
  } else if (epilogue == r4dx_epilogue_int8_fraga8) {
    // Byte-identical to third_party/libr4d/r4d_quant_act_i8.hip's r4d_quant_act_i8_kernel: same
    // scale = max(1e-8, absmax)/127, same __float2int_rn + [-128,127] clamp, same WMMA-fragment
    // byte permute (src = (s<<4) + 8*((j&7)>>2) + 4*(j>>3) + (j&3), s=i>>4, j=i&15) -- see that
    // file's own header comment for the derivation. Requires K % 16 == 0 (checked by the host
    // wrapper below, matching r4d_quant_act_i8's own precondition).
    int8_t* __restrict__ outq = reinterpret_cast<int8_t*>(epilogue_out);
    float m = 0.0f;
    for (int64_t i = threadIdx.x; i < K; i += kThreads) {
      m = fmaxf(m, fabsf(__bfloat162float(row_bf16[i])));
    }
    m = BlockReduceMax(m);
    const float scale = fmaxf(m, 1e-8f) / 127.0f;
    const float inv = 1.0f / scale;
    if (threadIdx.x == 0) epilogue_scale_row[0] = scale;
    for (int64_t i = threadIdx.x; i < K; i += kThreads) {
      const int s = static_cast<int>(i >> 4);
      const int j = static_cast<int>(i & 15);
      const int src = (s << 4) + 8 * ((j & 7) >> 2) + 4 * (j >> 3) + (j & 3);
      const float v = __bfloat162float(row_bf16[src]) * inv;
      const int t = __float2int_rn(v);
      outq[i] = static_cast<int8_t>(t < -128 ? -128 : (t > 127 ? 127 : t));
    }
  }
}

// Shared precondition for r4dx_epilogue_int8_fraga8 (matches r4d_quant_act_i8's own throw) --
// checked once here rather than duplicated at every one of this epilogue's three call sites below.
void CheckEpiloguePrecondition(int epilogue, int64_t K) {
  if (epilogue == r4dx_epilogue_int8_fraga8 && (K % 16) != 0) {
    throw std::runtime_error(
        "r4dx epilogue int8_fraga8: hidden/intermediate must be a multiple of 16, got " +
        std::to_string(K));
  }
}

// ---- rmsnorm ----------------------------------------------------------------------------------
// P6 (docs/r9700.md task item 1-2): grid stays dim3(rows) -- a single ~10-35 KB row is far below
// what 64 CUs need to be bandwidth-bound (docs/r9700.md's own "honest goal is latency at 1 row"),
// and kThreads=256 already gives 8 waves/block, the task's other sanctioned fallback -- but every
// load/store below is now a 16-byte uint4 (8x bf16) instead of a 2-byte scalar, cutting the
// per-thread loop trip count (and therefore issued memory instructions) 8x. Falls back to the
// original scalar loop, bit-for-bit, whenever `hidden` is not a multiple of 8 or any pointer is
// not 16-byte aligned; every real caller's buffers are hipMalloc'd (>=256 B aligned) at hidden in
// {5120, 6144, 17408}, all multiples of 8 and with a 16-byte-aligned row stride, so the fallback
// is for correctness/generality, not expected to fire on the hot path.
// __restrict__ contract note (review finding, 2026-09-20): src/model/attention/
// attention_layer.hpp's qk_norm call site passes x==out (in-place, q->q / k->k) despite `x`/`out`
// both being declared __restrict__ here. This is formally UB per the C/C++ standard, but safe
// under this kernel's actual structure: every thread's own read of x[i] (the sum-of-squares
// grid-stride loop) happens strictly before that SAME thread's own write to out[i] (the second
// grid-stride loop), AND every thread's read completes before any thread's write, because
// BlockReduceSum's own __syncthreads() sits exactly at that boundary -- so no read can ever
// observe a value a write already clobbered. If this kernel is ever restructured (e.g. merged into
// one pass, or the barrier moved), that invariant must be re-verified before an in-place caller is
// safe again -- tests/kernels/test_rmsnorm.cpp's CheckInPlaceAliasing exercises exactly this case
// and would need to keep passing. Not removing __restrict__ itself: it is what makes the
// vectorized (uint4) loads/stores above legal in the non-aliased (out-of-place) call sites, which
// are the overwhelming majority.
__global__ __launch_bounds__(kThreads) void RmsNormKernel(const __hip_bfloat16* __restrict__ x,
                                                            const __hip_bfloat16* __restrict__ w,
                                                            __hip_bfloat16* __restrict__ out,
                                                            int64_t hidden, float eps, int epilogue,
                                                            void* __restrict__ epilogue_out,
                                                            float* __restrict__ epilogue_scale) {
  const int64_t row = blockIdx.x;
  const __hip_bfloat16* xr = x + row * hidden;
  __hip_bfloat16* outr = out + row * hidden;

  const bool vec_ok = (hidden % 8 == 0) &&
                       (reinterpret_cast<uintptr_t>(xr) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(w) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(outr) % 16 == 0);

  float ss = 0.0f;
  if (vec_ok) {
    const uint4* __restrict__ xr4 = reinterpret_cast<const uint4*>(xr);
    const int64_t n4 = hidden >> 3;
    for (int64_t i = threadIdx.x; i < n4; i += kThreads) {
      const uint4 c = xr4[i];
      const unsigned dw[4] = {c.x, c.y, c.z, c.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __hip_bfloat16 lo, hi;
        UnpackBf16x2(dw[j], &lo, &hi);
        const float fl = __bfloat162float(lo), fh = __bfloat162float(hi);
        ss += fl * fl + fh * fh;
      }
    }
  } else {
    for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
      float v = __bfloat162float(xr[i]);
      ss += v * v;
    }
  }
  ss = BlockReduceSum(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);

  if (vec_ok) {
    const uint4* __restrict__ xr4 = reinterpret_cast<const uint4*>(xr);
    const uint4* __restrict__ w4 = reinterpret_cast<const uint4*>(w);
    uint4* __restrict__ out4 = reinterpret_cast<uint4*>(outr);
    const int64_t n4 = hidden >> 3;
    for (int64_t i = threadIdx.x; i < n4; i += kThreads) {
      const uint4 xc = xr4[i];
      const uint4 wc = w4[i];
      const unsigned xdw[4] = {xc.x, xc.y, xc.z, xc.w};
      const unsigned wdw[4] = {wc.x, wc.y, wc.z, wc.w};
      unsigned odw[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __hip_bfloat16 xlo, xhi, wlo, whi;
        UnpackBf16x2(xdw[j], &xlo, &xhi);
        UnpackBf16x2(wdw[j], &wlo, &whi);
        const float ov_lo = __bfloat162float(xlo) * rstd * (1.0f + __bfloat162float(wlo));
        const float ov_hi = __bfloat162float(xhi) * rstd * (1.0f + __bfloat162float(whi));
        odw[j] = PackBf16x2(__float2bfloat16(ov_lo), __float2bfloat16(ov_hi));
      }
      out4[i] = make_uint4(odw[0], odw[1], odw[2], odw[3]);
    }
  } else {
    for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
      float v = __bfloat162float(xr[i]);
      float wv = 1.0f + __bfloat162float(w[i]);
      outr[i] = __float2bfloat16(v * rstd * wv);
    }
  }

  if (epilogue != r4dx_epilogue_none) {
    __syncthreads();  // every thread's own write to outr[] above must be visible before the
                       // epilogue (int8's byte permute reads across threads' columns) re-reads it.
    void* eo = static_cast<uint8_t*>(epilogue_out) +
               static_cast<size_t>(row) * hidden * (epilogue == r4dx_epilogue_f16 ? 2 : 1);
    float* es = epilogue_scale + row;
    ApplyEpilogueRow(outr, epilogue, eo, es, hidden);
  }
}

// P6 (docs/r9700.md task item 1-2): same grid/vectorization approach as RmsNormKernel above. The
// residual-add pass (out_residual) is a pure per-element a+b -> bf16(RTNE) conversion with no
// cross-thread dependency, so its vectorized form is bit-exact against the pre-P6 scalar kernel
// (tests/kernels/test_kernel_bandwidth.cpp asserts this); the normed pass (out_normed) depends on
// the row's sum-of-squares reduction, whose thread-local partial-sum grouping changes under
// vectorization (float addition is not associative), so it is held to that same test's last-ulp
// tolerance rather than bit-exactness, exactly like RmsNormKernel's own output.
// __restrict__ contract note (review finding, 2026-09-20): gdn_layer.cpp and mlp.cpp both call
// this in place with x==out_residual (`cur, cur`), despite both being declared __restrict__ here.
// Safe under this kernel's actual structure: the grid-stride loop that reads x[i]/residual[i] and
// writes out_residual[i] assigns each index i to exactly ONE thread (i = threadIdx.x, stride
// kThreads), and that thread reads x[i] then writes out_residual[i] in the same loop iteration --
// no other thread ever touches index i, so there is no cross-thread race, and this thread's own
// read strictly precedes its own write. See RmsNormKernel's own identical-purpose comment above
// for the out_normed/w pass, which additionally relies on the explicit __syncthreads() below.
// tests/kernels/test_rmsnorm.cpp's CheckInPlaceAliasing exercises this case; keep it passing if
// this kernel is ever restructured.
__global__ __launch_bounds__(kThreads) void ResidualRmsNormKernel(
    const __hip_bfloat16* __restrict__ x, const __hip_bfloat16* __restrict__ residual,
    const __hip_bfloat16* __restrict__ w, __hip_bfloat16* __restrict__ out_residual,
    __hip_bfloat16* __restrict__ out_normed, int64_t hidden, float eps, int epilogue,
    void* __restrict__ epilogue_out, float* __restrict__ epilogue_scale) {
  const int64_t row = blockIdx.x;
  const __hip_bfloat16* xr = x + row * hidden;
  const __hip_bfloat16* rr = residual + row * hidden;
  __hip_bfloat16* orr = out_residual + row * hidden;
  __hip_bfloat16* onr = out_normed + row * hidden;

  const bool vec_ok = (hidden % 8 == 0) &&
                       (reinterpret_cast<uintptr_t>(xr) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(rr) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(w) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(orr) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(onr) % 16 == 0);

  // No dynamic shared-memory scratch: each thread's strided elements are its own, so instead of
  // stashing the fp32 sum in an extern __shared__ buffer (which caps `hidden` at 16384 floats on
  // gfx1201's 64 KB LDS and silently fails the launch above that), the norm pass re-reads the
  // bf16 sum just written to out_residual. The extra bf16 round-trip costs well under this
  // kernel's existing tolerance and removes both the hidden-size cap and the occupancy hit from
  // reserving shared memory.
  float ss = 0.0f;
  if (vec_ok) {
    const uint4* __restrict__ xr4 = reinterpret_cast<const uint4*>(xr);
    const uint4* __restrict__ rr4 = reinterpret_cast<const uint4*>(rr);
    uint4* __restrict__ orr4 = reinterpret_cast<uint4*>(orr);
    const int64_t n4 = hidden >> 3;
    for (int64_t i = threadIdx.x; i < n4; i += kThreads) {
      const uint4 xc = xr4[i];
      const uint4 rc = rr4[i];
      const unsigned xdw[4] = {xc.x, xc.y, xc.z, xc.w};
      const unsigned rdw[4] = {rc.x, rc.y, rc.z, rc.w};
      unsigned odw[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __hip_bfloat16 xlo, xhi, rlo, rhi;
        UnpackBf16x2(xdw[j], &xlo, &xhi);
        UnpackBf16x2(rdw[j], &rlo, &rhi);
        const float slo = __bfloat162float(xlo) + __bfloat162float(rlo);
        const float shi = __bfloat162float(xhi) + __bfloat162float(rhi);
        odw[j] = PackBf16x2(__float2bfloat16(slo), __float2bfloat16(shi));
        ss += slo * slo + shi * shi;
      }
      orr4[i] = make_uint4(odw[0], odw[1], odw[2], odw[3]);
    }
  } else {
    for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
      float s = __bfloat162float(xr[i]) + __bfloat162float(rr[i]);
      orr[i] = __float2bfloat16(s);
      ss += s * s;
    }
  }
  __syncthreads();
  ss = BlockReduceSum(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);

  if (vec_ok) {
    const uint4* __restrict__ orr4 = reinterpret_cast<const uint4*>(orr);
    const uint4* __restrict__ w4 = reinterpret_cast<const uint4*>(w);
    uint4* __restrict__ onr4 = reinterpret_cast<uint4*>(onr);
    const int64_t n4 = hidden >> 3;
    for (int64_t i = threadIdx.x; i < n4; i += kThreads) {
      const uint4 sc = orr4[i];
      const uint4 wc = w4[i];
      const unsigned sdw[4] = {sc.x, sc.y, sc.z, sc.w};
      const unsigned wdw[4] = {wc.x, wc.y, wc.z, wc.w};
      unsigned odw[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __hip_bfloat16 slo, shi, wlo, whi;
        UnpackBf16x2(sdw[j], &slo, &shi);
        UnpackBf16x2(wdw[j], &wlo, &whi);
        const float ov_lo = __bfloat162float(slo) * rstd * (1.0f + __bfloat162float(wlo));
        const float ov_hi = __bfloat162float(shi) * rstd * (1.0f + __bfloat162float(whi));
        odw[j] = PackBf16x2(__float2bfloat16(ov_lo), __float2bfloat16(ov_hi));
      }
      onr4[i] = make_uint4(odw[0], odw[1], odw[2], odw[3]);
    }
  } else {
    for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
      float s = __bfloat162float(orr[i]);
      float wv = 1.0f + __bfloat162float(w[i]);
      onr[i] = __float2bfloat16(s * rstd * wv);
    }
  }

  if (epilogue != r4dx_epilogue_none) {
    __syncthreads();  // see RmsNormKernel's identical comment -- onr[] must be fully visible.
    void* eo = static_cast<uint8_t*>(epilogue_out) +
               static_cast<size_t>(row) * hidden * (epilogue == r4dx_epilogue_f16 ? 2 : 1);
    float* es = epilogue_scale + row;
    ApplyEpilogueRow(onr, epilogue, eo, es, hidden);
  }
}

// ---- residual add -------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void ResidualAddKernel(const __hip_bfloat16* __restrict__ a,
                                                                const __hip_bfloat16* __restrict__ b,
                                                                __hip_bfloat16* __restrict__ out,
                                                                int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (i < n) {
    out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
  }
}

// ---- silu_mul -----------------------------------------------------------------------------
// One workgroup per row, matching the fused mlp.gate_up layout docs/container-format.md and
// docs/architecture.md mandate: gate_up[row, 0:I) is gate, gate_up[row, I:2I) is up, WITHIN each
// row (row stride `in_row_stride`, ordinarily 2*intermediate but left explicit for any future
// padding). A flat two-pointer call (the previous signature) silently read across row boundaries
// for any rows>1 chunked-prefill call -- see kernels.h.
// P6 (docs/r9700.md task item 1-2): pure elementwise (silu(gate)*up has no cross-thread
// dependency), so the vectorized form below is bit-exact against the pre-P6 scalar kernel
// (tests/kernels/test_kernel_bandwidth.cpp asserts this) -- every half still goes through the
// same __float2bfloat16 (RTNE) convert, just packed 4 dwords (8 elements) at a time instead of
// stored one bf16 at a time. Falls back to the scalar loop, bit-for-bit, when `intermediate` is
// not a multiple of 8 or a pointer is not 16-byte aligned; see RmsNormKernel's comment for why
// that fallback is not expected to fire for this model's real shapes.
__global__ __launch_bounds__(kThreads) void SiluMulKernel(
    const __hip_bfloat16* __restrict__ gate_up, __hip_bfloat16* __restrict__ out,
    int64_t intermediate, int64_t in_row_stride, int epilogue,
    void* __restrict__ epilogue_out, float* __restrict__ epilogue_scale) {
  const int64_t row = blockIdx.x;
  const __hip_bfloat16* gu = gate_up + row * in_row_stride;
  __hip_bfloat16* outr = out + row * intermediate;

  const bool vec_ok = (intermediate % 8 == 0) &&
                       (reinterpret_cast<uintptr_t>(gu) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(gu + intermediate) % 16 == 0) &&
                       (reinterpret_cast<uintptr_t>(outr) % 16 == 0);

  if (vec_ok) {
    const uint4* __restrict__ g4 = reinterpret_cast<const uint4*>(gu);
    const uint4* __restrict__ u4 = reinterpret_cast<const uint4*>(gu + intermediate);
    uint4* __restrict__ out4 = reinterpret_cast<uint4*>(outr);
    const int64_t n4 = intermediate >> 3;
    for (int64_t i = threadIdx.x; i < n4; i += kThreads) {
      const uint4 gc = g4[i];
      const uint4 uc = u4[i];
      const unsigned gdw[4] = {gc.x, gc.y, gc.z, gc.w};
      const unsigned udw[4] = {uc.x, uc.y, uc.z, uc.w};
      unsigned odw[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        __hip_bfloat16 glo, ghi, ulo, uhi;
        UnpackBf16x2(gdw[j], &glo, &ghi);
        UnpackBf16x2(udw[j], &ulo, &uhi);
        const float gl = __bfloat162float(glo), gh = __bfloat162float(ghi);
        const float ul = __bfloat162float(ulo), uh = __bfloat162float(uhi);
        odw[j] = PackBf16x2(__float2bfloat16(gl * Sigmoid(gl) * ul),
                             __float2bfloat16(gh * Sigmoid(gh) * uh));
      }
      out4[i] = make_uint4(odw[0], odw[1], odw[2], odw[3]);
    }
  } else {
    for (int64_t i = threadIdx.x; i < intermediate; i += kThreads) {
      float g = __bfloat162float(gu[i]);
      float u = __bfloat162float(gu[intermediate + i]);
      outr[i] = __float2bfloat16(g * Sigmoid(g) * u);
    }
  }

  if (epilogue != r4dx_epilogue_none) {
    __syncthreads();  // see RmsNormKernel's identical comment -- outr[] must be fully visible.
    void* eo = static_cast<uint8_t*>(epilogue_out) +
               static_cast<size_t>(row) * intermediate * (epilogue == r4dx_epilogue_f16 ? 2 : 1);
    float* es = epilogue_scale + row;
    ApplyEpilogueRow(outr, epilogue, eo, es, intermediate);
  }
}

// ---- rope: partial rotary, NeoX half-split, text-only mrope -------------------------------
__global__ void RopePartialMropeKernel(__hip_bfloat16* __restrict__ q,
                                        __hip_bfloat16* __restrict__ k,
                                        const int32_t* __restrict__ pos_ids, int heads_q,
                                        int heads_k, int head_dim, int rotary_dim, float theta) {
  const int64_t t = blockIdx.x;
  const int half = rotary_dim / 2;
  const float pos = static_cast<float>(pos_ids[t]);

  for (int hd = threadIdx.x; hd < half; hd += blockDim.x) {
    const float inv_freq = exp2f(-(2.0f * hd) / static_cast<float>(rotary_dim) * log2f(theta));
    const float angle = pos * inv_freq;
    float c, s;
    // Accurate range-reduced sincos, not the __sincosf() fast-math intrinsic: at
    // max_position_embeddings (262144) the angle can reach ~2.6e5 rad, where __sincosf's
    // approximation error (~1.3e-2) exceeds bf16's own ~4e-3 resolution. The per-(token,
    // frequency-bin) sincos is amortized over 24 q heads + 4 k heads, so the accurate version
    // costs effectively nothing here.
    sincosf(angle, &s, &c);

    __hip_bfloat16* qbase = q + t * static_cast<int64_t>(heads_q) * head_dim;
    for (int h = 0; h < heads_q; ++h) {
      __hip_bfloat16* row = qbase + static_cast<int64_t>(h) * head_dim;
      float x0 = __bfloat162float(row[hd]);
      float x1 = __bfloat162float(row[hd + half]);
      row[hd] = __float2bfloat16(x0 * c - x1 * s);
      row[hd + half] = __float2bfloat16(x1 * c + x0 * s);
    }
    __hip_bfloat16* kbase = k + t * static_cast<int64_t>(heads_k) * head_dim;
    for (int h = 0; h < heads_k; ++h) {
      __hip_bfloat16* row = kbase + static_cast<int64_t>(h) * head_dim;
      float x0 = __bfloat162float(row[hd]);
      float x1 = __bfloat162float(row[hd + half]);
      row[hd] = __float2bfloat16(x0 * c - x1 * s);
      row[hd + half] = __float2bfloat16(x1 * c + x0 * s);
    }
  }
}

// ---- fp8 e4m3 per-row activation quant -----------------------------------------------------
__global__ __launch_bounds__(kThreads) void QuantActFp8Kernel(const __hip_bfloat16* __restrict__ x,
                                                                uint8_t* __restrict__ q,
                                                                float* __restrict__ scale_out,
                                                                int K) {
  const int row = blockIdx.x;
  const __hip_bfloat16* xr = x + static_cast<size_t>(row) * K;
  uint8_t* qr = q + static_cast<size_t>(row) * K;

  float m = 0.0f;
  for (int i = threadIdx.x; i < K; i += kThreads) {
    m = fmaxf(m, fabsf(__bfloat162float(xr[i])));
  }
  m = BlockReduceMax(m);
  const float scale = fmaxf(m, 1e-8f) / 448.0f;  // 448 = e4m3fn max finite magnitude
  const float inv = 1.0f / scale;
  if (threadIdx.x == 0) scale_out[row] = scale;

  for (int i = threadIdx.x; i < K; i += kThreads) {
    qr[i] = FloatToFp8E4M3(__bfloat162float(xr[i]) * inv);
  }
}

// ---- paged fp8 KV cache write ----------------------------------------------------------------
__global__ void KvWritePagedFp8Kernel(const __hip_bfloat16* __restrict__ k_new,
                                       const __hip_bfloat16* __restrict__ v_new,
                                       const int32_t* __restrict__ slot_mapping,
                                       const float* __restrict__ k_descale,
                                       const float* __restrict__ v_descale,
                                       uint8_t* __restrict__ kv_cache, int kv_heads, int head_dim,
                                       int block_size, int64_t kv_block_stride,
                                       int64_t kv_head_stride) {
  const int t = blockIdx.x;
  const int h = blockIdx.y;
  const int slot = slot_mapping[t];
  if (slot < 0) return;  // -1 = skip this padded token, the universal batching-layer convention
  const int block_id = slot / block_size;
  const int offset = slot % block_size;
  const int64_t base = static_cast<int64_t>(block_id) * kv_block_stride +
                        static_cast<int64_t>(h) * kv_head_stride +
                        static_cast<int64_t>(offset) * (2 * head_dim);
  const float kd_inv = 1.0f / k_descale[h];
  const float vd_inv = 1.0f / v_descale[h];
  const __hip_bfloat16* kk = k_new + (static_cast<size_t>(t) * kv_heads + h) * head_dim;
  const __hip_bfloat16* vv = v_new + (static_cast<size_t>(t) * kv_heads + h) * head_dim;

  for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
    kv_cache[base + d] = FloatToFp8E4M3(__bfloat162float(kk[d]) * kd_inv);
    kv_cache[base + head_dim + d] = FloatToFp8E4M3(__bfloat162float(vv[d]) * vd_inv);
  }
}

// ---- device argmax ------------------------------------------------------------------------
// One block, grid-stride loop over `vocab` (kernels.h: ~1MB at this model's vocab, comfortably one
// block's worth of work). Reduces (value, index) pairs together (not a separate max-then-search
// pass) so ties resolve toward the lowest index the same way r4dx::kernels::Argmax (sampler.hpp,
// CPU) does: a strict `>` comparison during the reduction only replaces the running best when a
// later element is STRICTLY greater, so among equal values the earliest index seen (which, thanks
// to the grid-stride loop and the deterministic warp-then-block shuffle-reduce order below, is
// always the lowest index for a genuine tie) wins.
// NOTE: indices are carried as int32_t throughout (not int64_t) -- this model's vocab (~250k) is
// far under INT32_MAX, and __shfl_xor has no int64 overload, so int32 avoids a lossy cast at every
// shuffle step.
__global__ __launch_bounds__(kThreads) void ArgmaxKernel(const float* __restrict__ logits,
                                                            int32_t* __restrict__ out_idx,
                                                            int64_t vocab) {
  float best_v = -INFINITY;
  int32_t best_i = 0;
  for (int64_t i = threadIdx.x; i < vocab; i += kThreads) {
    const float v = logits[i];
    if (v > best_v) {
      best_v = v;
      best_i = static_cast<int32_t>(i);
    }
  }

  __shared__ float red_v[kThreads / 32];
  __shared__ int32_t red_i[kThreads / 32];
  for (int o = 16; o; o >>= 1) {
    const float ov = __shfl_xor(best_v, o, 32);
    const int32_t oi = __shfl_xor(best_i, o, 32);
    if (ov > best_v || (ov == best_v && oi < best_i)) {
      best_v = ov;
      best_i = oi;
    }
  }
  if ((threadIdx.x & 31) == 0) {
    red_v[threadIdx.x >> 5] = best_v;
    red_i[threadIdx.x >> 5] = best_i;
  }
  __syncthreads();

  if (threadIdx.x < (blockDim.x + 31) / 32) {
    best_v = red_v[threadIdx.x];
    best_i = red_i[threadIdx.x];
  } else {
    best_v = -INFINITY;
    best_i = 0;
  }
  if (threadIdx.x < 32) {
    for (int o = 16; o; o >>= 1) {
      const float ov = __shfl_xor(best_v, o, 32);
      const int32_t oi = __shfl_xor(best_i, o, 32);
      if (ov > best_v || (ov == best_v && oi < best_i)) {
        best_v = ov;
        best_i = oi;
      }
    }
  }
  if (threadIdx.x == 0) out_idx[0] = best_i;
}

// ---- device-resident embedding gather (MTP device-residency pass) -------------------------
// One workgroup per output row: out[row,:] = table[ids[row],:]. See kernels.h for why `ids` must
// be a device pointer (feeds directly from r4dx_argmax_f32's own output with no host round trip).
// Bounds guard (review finding, 2026-09-20): an id outside [0, vocab) used to read straight past
// the table's own VRAM allocation -- every thread in the block computes the same `id` from global
// memory, so this compare is coherent across the whole warp/block (no divergent-branch cost) and
// clamps to row 0 instead of faulting or reading garbage. Cheap at 64 rows/call; not worth a
// separate host-side check given callers already trust this convention elsewhere in this header.
__global__ __launch_bounds__(kThreads) void EmbeddingGatherKernel(
    const __hip_bfloat16* __restrict__ table, const int32_t* __restrict__ ids,
    __hip_bfloat16* __restrict__ out, int64_t hidden, int64_t vocab) {
  const int64_t row = blockIdx.x;
  int32_t id = ids[row];
  if (id < 0 || id >= vocab) id = 0;
  const __hip_bfloat16* src = table + static_cast<int64_t>(id) * hidden;
  __hip_bfloat16* dst = out + row * hidden;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) dst[i] = src[i];
}

// ---- device-side single/few-element gather (docs/r9700.md R9, reduced-vocab MTP draft head) ----
// One thread per output element -- `n` is always tiny (1 in every current caller), so this is
// deliberately not a one-workgroup-per-row kernel like EmbeddingGatherKernel above. Same
// out-of-range-clamps-to-index-0 bounds guard as that kernel, for the same reason (a device fault
// mid-kernel is unrecoverable; a wrong-but-safe value is not).
__global__ void GatherI32Kernel(const int32_t* __restrict__ table, const int32_t* __restrict__ idx,
                                 int32_t* __restrict__ out, int64_t n, int64_t table_size) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int32_t id = idx[i];
  if (id < 0 || id >= table_size) id = 0;
  out[i] = table[id];
}

// ================ DFlash2 drafter device pieces (docs/dflash2.md "Kernels") ====================

// ---- rope: full-width NeoX split-half ---------------------------------------------------------
// One workgroup per row; `half` threads (capped at 256) each own one frequency bin and rotate that
// bin's (i, i+half) pair across every head of q and of k. `q`/`k` may independently be null (with
// the matching head count 0) -- the DFlash2 injection path ropes K only.
__global__ void RopeNeoxKernel(__hip_bfloat16* __restrict__ q, __hip_bfloat16* __restrict__ k,
                                const int32_t* __restrict__ pos_ids, int heads_q, int heads_k,
                                int head_dim, float theta) {
  const int64_t t = blockIdx.x;
  const int half = head_dim / 2;
  const float pos = static_cast<float>(pos_ids[t]);

  for (int hd = threadIdx.x; hd < half; hd += blockDim.x) {
    // Same expression order as RopePartialMropeKernel above (exp2f/log2f, then a float multiply,
    // then an ACCURATE sincosf -- not __sincosf; see that kernel's comment for why).
    const float inv_freq = exp2f(-(2.0f * hd) / static_cast<float>(head_dim) * log2f(theta));
    const float angle = pos * inv_freq;
    float c, s;
    sincosf(angle, &s, &c);

    if (heads_q > 0) {
      __hip_bfloat16* qbase = q + t * static_cast<int64_t>(heads_q) * head_dim;
      for (int h = 0; h < heads_q; ++h) {
        __hip_bfloat16* row = qbase + static_cast<int64_t>(h) * head_dim;
        const float x0 = __bfloat162float(row[hd]);
        const float x1 = __bfloat162float(row[hd + half]);
        row[hd] = __float2bfloat16(x0 * c - x1 * s);
        row[hd + half] = __float2bfloat16(x1 * c + x0 * s);
      }
    }
    if (heads_k > 0) {
      __hip_bfloat16* kbase = k + t * static_cast<int64_t>(heads_k) * head_dim;
      for (int h = 0; h < heads_k; ++h) {
        __hip_bfloat16* row = kbase + static_cast<int64_t>(h) * head_dim;
        const float x0 = __bfloat162float(row[hd]);
        const float x1 = __bfloat162float(row[hd + half]);
        row[hd] = __float2bfloat16(x0 * c - x1 * s);
        row[hd + half] = __float2bfloat16(x1 * c + x0 * s);
      }
    }
  }
}

// ---- per-row top-16 ---------------------------------------------------------------------------
// The single total order this kernel sorts by, used identically by the per-thread insertion and by
// the LDS merge so the result is the global top-16 under it, deterministically: higher value wins;
// equal values break toward the LOWER id (kernels.h's stated contract, matching `ggml_top_k` as
// the DFlash2 selector consumes it).
__device__ __forceinline__ bool TopkBetter(float a_v, int32_t a_i, float b_v, int32_t b_i) {
  return a_v > b_v || (a_v == b_v && a_i < b_i);
}

constexpr int kTopK = 16;

__global__ __launch_bounds__(kThreads) void Topk16Kernel(const float* __restrict__ logits,
                                                            int32_t* __restrict__ out_ids,
                                                            float* __restrict__ out_vals,
                                                            int64_t vocab) {
  const int64_t row = blockIdx.x;
  const float* __restrict__ lr = logits + row * vocab;
  const int tid = threadIdx.x;

  // Per-thread sorted-descending top-16, held in registers: every index below is a compile-time
  // constant (the shift loop is fully unrolled), so this does not spill to scratch. The sentinel
  // (-inf, INT32_MAX) loses to every real entry under TopkBetter, including a real -inf (tie
  // broken by the lower id), so short slices need no separate count.
  float bv[kTopK];
  int32_t bi[kTopK];
#pragma unroll
  for (int j = 0; j < kTopK; ++j) {
    bv[j] = -INFINITY;
    bi[j] = INT32_MAX;
  }
  for (int64_t i = tid; i < vocab; i += kThreads) {
    const float v = lr[i];
    const int32_t idx = static_cast<int32_t>(i);
    if (!TopkBetter(v, idx, bv[kTopK - 1], bi[kTopK - 1])) continue;
    bv[kTopK - 1] = v;
    bi[kTopK - 1] = idx;
#pragma unroll
    for (int s = kTopK - 1; s > 0; --s) {
      if (TopkBetter(bv[s], bi[s], bv[s - 1], bi[s - 1])) {
        const float tv = bv[s];
        bv[s] = bv[s - 1];
        bv[s - 1] = tv;
        const int32_t ti = bi[s];
        bi[s] = bi[s - 1];
        bi[s - 1] = ti;
      }
    }
  }

  __shared__ float s_v[kThreads * kTopK];
  __shared__ int32_t s_i[kThreads * kTopK];
#pragma unroll
  for (int j = 0; j < kTopK; ++j) {
    s_v[tid * kTopK + j] = bv[j];
    s_i[tid * kTopK + j] = bi[j];
  }
  __syncthreads();

  // Pairwise tree merge of the 256 sorted-16 lists (8 rounds). In a round, only threads with
  // tid < stride write, and each such thread writes ONLY its own slot after reading its own slot
  // and slot tid+stride (which no active thread writes this round) into registers -- so one
  // barrier per round, at the end, is sufficient.
  for (int stride = kThreads / 2; stride >= 1; stride >>= 1) {
    if (tid < stride) {
      float mv[kTopK];
      int32_t mi[kTopK];
      int pa = 0, pb = 0;
      const int base_a = tid * kTopK;
      const int base_b = (tid + stride) * kTopK;
#pragma unroll
      for (int j = 0; j < kTopK; ++j) {
        // pa/pb can each reach kTopK on the last step; clamp the READ (the guard below is what
        // decides) so a clamped read never leaves this block's LDS allocation.
        const int ra = pa < kTopK ? pa : kTopK - 1;
        const int rb = pb < kTopK ? pb : kTopK - 1;
        const float av = s_v[base_a + ra];
        const int32_t ai = s_i[base_a + ra];
        const float bvv = s_v[base_b + rb];
        const int32_t bii = s_i[base_b + rb];
        if (pa < kTopK && (pb >= kTopK || TopkBetter(av, ai, bvv, bii))) {
          mv[j] = av;
          mi[j] = ai;
          ++pa;
        } else {
          mv[j] = bvv;
          mi[j] = bii;
          ++pb;
        }
      }
#pragma unroll
      for (int j = 0; j < kTopK; ++j) {
        s_v[base_a + j] = mv[j];
        s_i[base_a + j] = mi[j];
      }
    }
    __syncthreads();
  }

  if (tid < kTopK) {
    out_vals[row * kTopK + tid] = s_v[tid];
    out_ids[row * kTopK + tid] = s_i[tid];
  }
}

// ---- DFlash2 draft-block attention: non-causal, windowed, GQA ---------------------------------
// One workgroup per (block row t, kv-head), 128 threads, serving that kv-head's up to 4 q-heads
// together so the K and V slices are each read exactly once per workgroup.
//
// Pass 1 (scores): one WAVE per key, 32 lanes x head_dim/32 dims each -> a fully coalesced
//   256-byte load per key, then one wave shuffle-reduce per q-head. Scores land in LDS.
// Pass 2a (softmax): wave r reduces q-head r's own score row (max, then exp-in-place and sum).
// Pass 2b (output): thread d owns dim d and sweeps the keys once, reading V coalesced and
//   accumulating all `ratio` q-heads' outputs in registers.
//
// This is a two-pass (materialise-then-normalise) softmax rather than a streaming online one: the
// whole visible row is at most 2055 scores (window 2048 + an 8-wide block), which fits LDS
// comfortably, and it is the numerically identical max-subtracted fp32 softmax with less state.
constexpr int kAttnThreads = 128;
constexpr int kAttnMaxKeys = 2056;   // (window - 1) + T <= 2047 + 8
constexpr int kAttnMaxRatio = 4;     // heads_q / heads_kv
constexpr int kAttnMaxHeadDim = 128;

__global__ __launch_bounds__(kAttnThreads) void DflashAttnKernel(
    const __hip_bfloat16* __restrict__ q, const __hip_bfloat16* __restrict__ k_block,
    const __hip_bfloat16* __restrict__ v_block, const __hip_bfloat16* __restrict__ k_store,
    const __hip_bfloat16* __restrict__ v_store, __hip_bfloat16* __restrict__ out, int T,
    int heads_q, int heads_kv, int head_dim, int n_injected, int window, int slots, float scale) {
  const int t = blockIdx.x;
  const int kvh = blockIdx.y;
  const int ratio = heads_q / heads_kv;
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int wave = tid >> 5;
  constexpr int kWaves = kAttnThreads / 32;
  const int dpl = head_dim / 32;  // dims per lane

  // Visible injected-store range for this query row: SWA says a key at position p is visible iff
  // q_pos - p < window, and only positions already injected exist -- so p in [lo, n_injected-1].
  const int q_pos = n_injected + t;
  int lo = q_pos - window + 1;
  if (lo < 0) lo = 0;
  const int n_store = n_injected > lo ? n_injected - lo : 0;
  const int total = n_store + T;  // + the block's own T keys, all unconditionally visible

  __shared__ float s_q[kAttnMaxRatio][kAttnMaxHeadDim];
  __shared__ float s_sc[kAttnMaxRatio][kAttnMaxKeys];
  __shared__ float s_m[kAttnMaxRatio], s_l[kAttnMaxRatio];

  if (tid < head_dim) {
    for (int r = 0; r < ratio; ++r) {
      const int hq = kvh * ratio + r;
      s_q[r][tid] = __bfloat162float(
          q[(static_cast<int64_t>(t) * heads_q + hq) * head_dim + tid]);
    }
  }
  __syncthreads();

  for (int j = wave; j < total; j += kWaves) {
    const __hip_bfloat16* kp;
    if (j < n_store) {
      const int slot = (lo + j) % slots;
      kp = k_store + (static_cast<int64_t>(slot) * heads_kv + kvh) * head_dim;
    } else {
      kp = k_block + (static_cast<int64_t>(j - n_store) * heads_kv + kvh) * head_dim;
    }
    float acc[kAttnMaxRatio] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int u = 0; u < dpl; ++u) {
      const int d = lane * dpl + u;
      const float kval = __bfloat162float(kp[d]);
#pragma unroll
      for (int r = 0; r < kAttnMaxRatio; ++r) {
        if (r < ratio) acc[r] += s_q[r][d] * kval;
      }
    }
#pragma unroll
    for (int r = 0; r < kAttnMaxRatio; ++r) {
      for (int o = 16; o; o >>= 1) acc[r] += __shfl_xor(acc[r], o, 32);
    }
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < kAttnMaxRatio; ++r) {
        if (r < ratio) s_sc[r][j] = acc[r] * scale;
      }
    }
  }
  __syncthreads();

  if (wave < ratio) {
    float m = -INFINITY;
    for (int j = lane; j < total; j += 32) m = fmaxf(m, s_sc[wave][j]);
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
    float l = 0.0f;
    for (int j = lane; j < total; j += 32) {
      const float e = expf(s_sc[wave][j] - m);
      s_sc[wave][j] = e;
      l += e;
    }
    for (int o = 16; o; o >>= 1) l += __shfl_xor(l, o, 32);
    if (lane == 0) {
      s_m[wave] = m;
      s_l[wave] = l;
    }
  }
  __syncthreads();

  if (tid < head_dim) {
    const int d = tid;
    float acc[kAttnMaxRatio] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int j = 0; j < total; ++j) {
      const __hip_bfloat16* vp;
      if (j < n_store) {
        const int slot = (lo + j) % slots;
        vp = v_store + (static_cast<int64_t>(slot) * heads_kv + kvh) * head_dim;
      } else {
        vp = v_block + (static_cast<int64_t>(j - n_store) * heads_kv + kvh) * head_dim;
      }
      const float vv = __bfloat162float(vp[d]);
#pragma unroll
      for (int r = 0; r < kAttnMaxRatio; ++r) {
        if (r < ratio) acc[r] += s_sc[r][j] * vv;
      }
    }
    for (int r = 0; r < ratio; ++r) {
      const int hq = kvh * ratio + r;
      out[(static_cast<int64_t>(t) * heads_q + hq) * head_dim + d] =
          __float2bfloat16(acc[r] / s_l[r]);
    }
  }
}

// ---- plain (non zero-centered) rmsnorm ---------------------------------------------------------
// out = x * rsqrt(mean(x^2) + eps) * w -- NO (1+w). See kernels.h for why this is a separate entry
// point from RmsNormKernel rather than a flag on it.
// In-place (out == x, out_fp32 == 0) is safe for the same reason RmsNormKernel's in-place use is:
// every thread's read of xr[i] in the sum-of-squares loop happens before BlockReduceSum's
// __syncthreads(), which happens before any thread's write in the output loop.
__global__ __launch_bounds__(kThreads) void RmsNormPlainKernel(
    const __hip_bfloat16* __restrict__ x, const __hip_bfloat16* __restrict__ w, void* __restrict__ out,
    int64_t hidden, float eps, int out_fp32) {
  const int64_t row = blockIdx.x;
  const __hip_bfloat16* xr = x + row * hidden;

  float ss = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    const float v = __bfloat162float(xr[i]);
    ss += v * v;
  }
  ss = BlockReduceSum(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);

  if (out_fp32) {
    float* outr = static_cast<float*>(out) + row * hidden;
    for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
      outr[i] = __bfloat162float(xr[i]) * rstd * __bfloat162float(w[i]);
    }
  } else {
    __hip_bfloat16* outr = static_cast<__hip_bfloat16*>(out) + row * hidden;
    for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
      outr[i] = __float2bfloat16(__bfloat162float(xr[i]) * rstd * __bfloat162float(w[i]));
    }
  }
}

}  // namespace

extern "C" {

void r4dx_rmsnorm_bf16(int64_t x, int64_t weight, int64_t out, int64_t rows, int64_t hidden,
                        float eps, int64_t stream, int epilogue, int64_t epilogue_out,
                        int64_t epilogue_scale) {
  if (rows <= 0) return;
  CheckEpiloguePrecondition(epilogue, hidden);
  ++g_launch_count;
  hipLaunchKernelGGL(RmsNormKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)x, (const __hip_bfloat16*)weight,
                     (__hip_bfloat16*)out, hidden, eps, epilogue, (void*)epilogue_out,
                     (float*)epilogue_scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_residual_rmsnorm_bf16(int64_t x, int64_t residual, int64_t weight,
                                 int64_t out_residual, int64_t out_normed, int64_t rows,
                                 int64_t hidden, float eps, int64_t stream, int epilogue,
                                 int64_t epilogue_out, int64_t epilogue_scale) {
  if (rows <= 0) return;
  CheckEpiloguePrecondition(epilogue, hidden);
  ++g_launch_count;
  hipLaunchKernelGGL(ResidualRmsNormKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)x,
                     (const __hip_bfloat16*)residual, (const __hip_bfloat16*)weight,
                     (__hip_bfloat16*)out_residual, (__hip_bfloat16*)out_normed, hidden, eps,
                     epilogue, (void*)epilogue_out, (float*)epilogue_scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_residual_add_bf16(int64_t a, int64_t b, int64_t out, int64_t n, int64_t stream) {
  if (n <= 0) return;
  ++g_launch_count;
  unsigned blocks = static_cast<unsigned>((n + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(ResidualAddKernel, dim3(blocks), dim3(kThreads), 0, (hipStream_t)stream,
                     (const __hip_bfloat16*)a, (const __hip_bfloat16*)b, (__hip_bfloat16*)out, n);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_silu_mul_bf16(int64_t gate_up, int64_t out, int64_t rows, int64_t intermediate,
                         int64_t in_row_stride, int64_t stream, int epilogue,
                         int64_t epilogue_out, int64_t epilogue_scale) {
  if (rows <= 0 || intermediate <= 0) return;
  CheckEpiloguePrecondition(epilogue, intermediate);
  ++g_launch_count;
  hipLaunchKernelGGL(SiluMulKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)gate_up, (__hip_bfloat16*)out,
                     intermediate, in_row_stride, epilogue, (void*)epilogue_out,
                     (float*)epilogue_scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_rope_partial_mrope_bf16(int64_t q, int64_t k, int64_t pos_ids, int tokens, int heads_q,
                                   int heads_k, int head_dim, int rotary_dim, float theta,
                                   int64_t stream) {
  if (tokens <= 0) return;
  ++g_launch_count;
  int threads = rotary_dim / 2;
  if (threads > 256) threads = 256;
  if (threads < 1) threads = 1;
  hipLaunchKernelGGL(RopePartialMropeKernel, dim3(static_cast<unsigned>(tokens)), dim3(threads), 0,
                     (hipStream_t)stream, (__hip_bfloat16*)q, (__hip_bfloat16*)k,
                     (const int32_t*)pos_ids, heads_q, heads_k, head_dim, rotary_dim, theta);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_quant_act_fp8e4m3_row(int64_t x, int64_t q, int64_t scale, int M, int K,
                                 int64_t stream) {
  if (M <= 0) return;
  ++g_launch_count;
  hipLaunchKernelGGL(QuantActFp8Kernel, dim3(static_cast<unsigned>(M)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)x, (uint8_t*)q, (float*)scale, K);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_kv_write_paged_fp8_hnd(int64_t k_new, int64_t v_new, int64_t slot_mapping,
                                  int64_t k_descale, int64_t v_descale, int64_t kv_cache, int T,
                                  int kv_heads, int head_dim, int block_size,
                                  int64_t kv_block_stride, int64_t kv_head_stride,
                                  int64_t stream) {
  if (T <= 0 || kv_heads <= 0) return;
  ++g_launch_count;
  int threads = head_dim < 256 ? head_dim : 256;
  hipLaunchKernelGGL(KvWritePagedFp8Kernel, dim3(static_cast<unsigned>(T), static_cast<unsigned>(kv_heads)),
                     dim3(threads), 0, (hipStream_t)stream, (const __hip_bfloat16*)k_new,
                     (const __hip_bfloat16*)v_new, (const int32_t*)slot_mapping,
                     (const float*)k_descale, (const float*)v_descale, (uint8_t*)kv_cache,
                     kv_heads, head_dim, block_size, kv_block_stride, kv_head_stride);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_argmax_f32(int64_t logits, int64_t out_idx, int64_t vocab, int64_t stream) {
  if (vocab <= 0) return;
  ++g_launch_count;
  hipLaunchKernelGGL(ArgmaxKernel, dim3(1), dim3(kThreads), 0, (hipStream_t)stream,
                     (const float*)logits, (int32_t*)out_idx, vocab);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_embedding_gather_bf16(int64_t table, int64_t ids, int64_t out, int64_t n, int64_t hidden,
                                 int64_t vocab, int64_t stream) {
  if (n <= 0) return;
  ++g_launch_count;
  hipLaunchKernelGGL(EmbeddingGatherKernel, dim3(static_cast<unsigned>(n)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)table, (const int32_t*)ids,
                     (__hip_bfloat16*)out, hidden, vocab);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_gather_i32(int64_t table, int64_t idx, int64_t out, int64_t n, int64_t table_size,
                      int64_t stream) {
  if (n <= 0) return;
  ++g_launch_count;
  const unsigned threads = n < 32 ? static_cast<unsigned>(n) : 32u;
  const unsigned blocks = static_cast<unsigned>((n + threads - 1) / threads);
  hipLaunchKernelGGL(GatherI32Kernel, dim3(blocks), dim3(threads), 0, (hipStream_t)stream,
                     (const int32_t*)table, (const int32_t*)idx, (int32_t*)out, n, table_size);
  R4DX_HIP_CHECK(hipGetLastError());
}

// ================ DFlash2 drafter device pieces (docs/dflash2.md "Kernels") ====================

void r4dx_rope_neox_bf16(int64_t q, int64_t k, int64_t pos_ids, int rows, int heads_q, int heads_k,
                          int head_dim, float theta, int64_t stream) {
  if (rows <= 0 || (heads_q <= 0 && heads_k <= 0)) return;
  if (head_dim <= 0 || (head_dim % 2) != 0) {
    throw std::runtime_error("r4dx_rope_neox_bf16: head_dim must be positive and even, got " +
                              std::to_string(head_dim));
  }
  ++g_launch_count;
  int threads = head_dim / 2;
  if (threads > 256) threads = 256;
  hipLaunchKernelGGL(RopeNeoxKernel, dim3(static_cast<unsigned>(rows)), dim3(threads), 0,
                     (hipStream_t)stream, (__hip_bfloat16*)q, (__hip_bfloat16*)k,
                     (const int32_t*)pos_ids, heads_q > 0 ? heads_q : 0, heads_k > 0 ? heads_k : 0,
                     head_dim, theta);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_topk16_f32(int64_t logits, int64_t out_ids, int64_t out_vals, int rows, int64_t vocab,
                      int64_t stream) {
  if (rows <= 0) return;
  if (vocab < kTopK) {
    throw std::runtime_error("r4dx_topk16_f32: vocab must be >= 16, got " + std::to_string(vocab));
  }
  ++g_launch_count;
  hipLaunchKernelGGL(Topk16Kernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const float*)logits, (int32_t*)out_ids,
                     (float*)out_vals, vocab);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_dflash_attn_bf16(int64_t q, int64_t k_block, int64_t v_block, int64_t k_store,
                            int64_t v_store, int64_t out, int T, int heads_q, int heads_kv,
                            int head_dim, int n_injected, int window, int slots, float scale,
                            int64_t stream) {
  if (T <= 0 || heads_kv <= 0) return;
  // Every one of these is a HARD precondition of the kernel's fixed LDS geometry: violating one
  // would read/write outside s_q/s_sc rather than merely produce a slow answer.
  if (T > 8) {
    throw std::runtime_error("r4dx_dflash_attn_bf16: T must be <= 8 (the DFlash2 block size), got " +
                              std::to_string(T));
  }
  if (head_dim <= 0 || head_dim > kAttnMaxHeadDim || (head_dim % 32) != 0) {
    throw std::runtime_error(
        "r4dx_dflash_attn_bf16: head_dim must be a positive multiple of 32 and <= 128, got " +
        std::to_string(head_dim));
  }
  if (heads_q <= 0 || (heads_q % heads_kv) != 0 || (heads_q / heads_kv) > kAttnMaxRatio) {
    throw std::runtime_error(
        "r4dx_dflash_attn_bf16: heads_q must be a multiple of heads_kv with ratio <= 4, got " +
        std::to_string(heads_q) + "/" + std::to_string(heads_kv));
  }
  if (window <= 0 || window > kAttnMaxKeys - 8) {
    throw std::runtime_error("r4dx_dflash_attn_bf16: window must be in [1, 2048], got " +
                              std::to_string(window));
  }
  if (slots < window) {
    throw std::runtime_error(
        "r4dx_dflash_attn_bf16: slots must be >= window (otherwise the visible store range aliases "
        "itself in the ring), got slots=" +
        std::to_string(slots) + " window=" + std::to_string(window));
  }
  if (n_injected < 0) {
    throw std::runtime_error("r4dx_dflash_attn_bf16: n_injected must be >= 0, got " +
                              std::to_string(n_injected));
  }
  ++g_launch_count;
  hipLaunchKernelGGL(DflashAttnKernel,
                     dim3(static_cast<unsigned>(T), static_cast<unsigned>(heads_kv)),
                     dim3(kAttnThreads), 0, (hipStream_t)stream, (const __hip_bfloat16*)q,
                     (const __hip_bfloat16*)k_block, (const __hip_bfloat16*)v_block,
                     (const __hip_bfloat16*)k_store, (const __hip_bfloat16*)v_store,
                     (__hip_bfloat16*)out, T, heads_q, heads_kv, head_dim, n_injected, window,
                     slots, scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_dflash_conv_bf16(int64_t x, int64_t dyn, int64_t base, int64_t out, int T, int H,
                            int side, int block_size, int64_t stream) {
  if (T <= 0 || H <= 0) return;
  constexpr int kTaps = 2, kGroup = 16;  // r4d_dflash_conv_t2_g16_bf16's compiled-in geometry
  if ((H % kGroup) != 0) {
    throw std::runtime_error("r4dx_dflash_conv_bf16: H must be a multiple of 16, got " +
                              std::to_string(H));
  }
  if (side != 0 && side != 1) {
    throw std::runtime_error("r4dx_dflash_conv_bf16: side must be 0 or 1, got " +
                              std::to_string(side));
  }
  if (block_size <= 0 || (block_size & (block_size - 1)) != 0 || block_size < T) {
    throw std::runtime_error(
        "r4dx_dflash_conv_bf16: block_size must be a power of two >= T, got " +
        std::to_string(block_size) + " (T=" + std::to_string(T) + ")");
  }
  const int NG = H / kGroup;
  const int dpitch = 2 * kTaps * NG;  // the FULL [T, 2, taps, NG] row pitch (r4d.h:118)
  const auto* delta = reinterpret_cast<const uint16_t*>(dyn) +
                       static_cast<int64_t>(side) * kTaps * NG;  // side-major slice
  const auto* base_s = reinterpret_cast<const uint16_t*>(base) +
                        static_cast<int64_t>(side) * kTaps * H;  // base is [2][taps][H]
  // No ++g_launch_count: the launch below is libr4d's, and that counter counts r4dx-owned
  // launches only (see its doc comment in kernels.h).
  const int rc = r4d_dflash_conv_t2_g16_bf16((const void*)x, (const void*)delta, (const void*)base_s,
                                              (void*)out, T, H, dpitch, NG, kTaps, kGroup,
                                              block_size, (void*)stream);
  if (rc != 0) {
    throw std::runtime_error("r4dx_dflash_conv_bf16: r4d_dflash_conv_t2_g16_bf16 returned " +
                              std::to_string(rc));
  }
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_rmsnorm_plain_bf16(int64_t x, int64_t weight, int64_t out, int64_t rows, int64_t hidden,
                              float eps, int out_fp32, int64_t stream) {
  if (rows <= 0 || hidden <= 0) return;
  ++g_launch_count;
  hipLaunchKernelGGL(RmsNormPlainKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)x, (const __hip_bfloat16*)weight,
                     (void*)out, hidden, eps, out_fp32);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_kernel_launch_counter_reset() { g_launch_count = 0; }

int64_t r4dx_kernel_launch_counter_get() { return g_launch_count; }

}  // extern "C"
