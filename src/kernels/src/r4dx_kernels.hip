// r4dx_kernels.hip -- every r4dx-owned device kernel (docs/architecture.md "Own kernels"),
// compiled as one hipcc translation unit (mirrors third_party/CMakeLists.txt's per-unit recipe
// for r4d_core; see docs/build-windows.md). One row per workgroup for the row-reduction kernels
// (rmsnorm, quant_act_fp8), following r4d_quant_act_i8.hip's own convention in third_party/libr4d.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

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

void r4dx_kernel_launch_counter_reset() { g_launch_count = 0; }

int64_t r4dx_kernel_launch_counter_get() { return g_launch_count; }

}  // extern "C"
