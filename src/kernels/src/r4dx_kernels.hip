// r4dx_kernels.hip -- every r4dx-owned device kernel (docs/architecture.md "Own kernels"),
// compiled as one hipcc translation unit (mirrors third_party/CMakeLists.txt's per-unit recipe
// for r4d_core; see docs/build-windows.md). One row per workgroup for the row-reduction kernels
// (rmsnorm, quant_act_fp8), following r4d_quant_act_i8.hip's own convention in third_party/libr4d.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cstdint>

#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using r4dx::core::FloatToFp8E4M3;

namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float Sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

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

// ---- rmsnorm ----------------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void RmsNormKernel(const __hip_bfloat16* __restrict__ x,
                                                            const __hip_bfloat16* __restrict__ w,
                                                            __hip_bfloat16* __restrict__ out,
                                                            int64_t hidden, float eps) {
  const int64_t row = blockIdx.x;
  const __hip_bfloat16* xr = x + row * hidden;
  __hip_bfloat16* outr = out + row * hidden;

  float ss = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    float v = __bfloat162float(xr[i]);
    ss += v * v;
  }
  ss = BlockReduceSum(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);

  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    float v = __bfloat162float(xr[i]);
    float wv = 1.0f + __bfloat162float(w[i]);
    outr[i] = __float2bfloat16(v * rstd * wv);
  }
}

__global__ __launch_bounds__(kThreads) void ResidualRmsNormKernel(
    const __hip_bfloat16* __restrict__ x, const __hip_bfloat16* __restrict__ residual,
    const __hip_bfloat16* __restrict__ w, __hip_bfloat16* __restrict__ out_residual,
    __hip_bfloat16* __restrict__ out_normed, int64_t hidden, float eps) {
  const int64_t row = blockIdx.x;
  const __hip_bfloat16* xr = x + row * hidden;
  const __hip_bfloat16* rr = residual + row * hidden;
  __hip_bfloat16* orr = out_residual + row * hidden;
  __hip_bfloat16* onr = out_normed + row * hidden;

  // No dynamic shared-memory scratch: each thread's strided elements are its own, so instead of
  // stashing the fp32 sum in an extern __shared__ buffer (which caps `hidden` at 16384 floats on
  // gfx1201's 64 KB LDS and silently fails the launch above that), the norm pass re-reads the
  // bf16 sum just written to out_residual. The extra bf16 round-trip costs well under this
  // kernel's existing tolerance and removes both the hidden-size cap and the occupancy hit from
  // reserving shared memory.
  float ss = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    float s = __bfloat162float(xr[i]) + __bfloat162float(rr[i]);
    orr[i] = __float2bfloat16(s);
    ss += s * s;
  }
  __syncthreads();
  ss = BlockReduceSum(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);

  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    float s = __bfloat162float(orr[i]);
    float wv = 1.0f + __bfloat162float(w[i]);
    onr[i] = __float2bfloat16(s * rstd * wv);
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
__global__ __launch_bounds__(kThreads) void SiluMulKernel(
    const __hip_bfloat16* __restrict__ gate_up, __hip_bfloat16* __restrict__ out,
    int64_t intermediate, int64_t in_row_stride) {
  const int64_t row = blockIdx.x;
  const __hip_bfloat16* gu = gate_up + row * in_row_stride;
  __hip_bfloat16* outr = out + row * intermediate;
  for (int64_t i = threadIdx.x; i < intermediate; i += kThreads) {
    float g = __bfloat162float(gu[i]);
    float u = __bfloat162float(gu[intermediate + i]);
    outr[i] = __float2bfloat16(g * Sigmoid(g) * u);
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

}  // namespace

extern "C" {

void r4dx_rmsnorm_bf16(int64_t x, int64_t weight, int64_t out, int64_t rows, int64_t hidden,
                        float eps, int64_t stream) {
  if (rows <= 0) return;
  hipLaunchKernelGGL(RmsNormKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)x, (const __hip_bfloat16*)weight,
                     (__hip_bfloat16*)out, hidden, eps);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_residual_rmsnorm_bf16(int64_t x, int64_t residual, int64_t weight,
                                 int64_t out_residual, int64_t out_normed, int64_t rows,
                                 int64_t hidden, float eps, int64_t stream) {
  if (rows <= 0) return;
  hipLaunchKernelGGL(ResidualRmsNormKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)x,
                     (const __hip_bfloat16*)residual, (const __hip_bfloat16*)weight,
                     (__hip_bfloat16*)out_residual, (__hip_bfloat16*)out_normed, hidden, eps);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_residual_add_bf16(int64_t a, int64_t b, int64_t out, int64_t n, int64_t stream) {
  if (n <= 0) return;
  unsigned blocks = static_cast<unsigned>((n + kThreads - 1) / kThreads);
  hipLaunchKernelGGL(ResidualAddKernel, dim3(blocks), dim3(kThreads), 0, (hipStream_t)stream,
                     (const __hip_bfloat16*)a, (const __hip_bfloat16*)b, (__hip_bfloat16*)out, n);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_silu_mul_bf16(int64_t gate_up, int64_t out, int64_t rows, int64_t intermediate,
                         int64_t in_row_stride, int64_t stream) {
  if (rows <= 0 || intermediate <= 0) return;
  hipLaunchKernelGGL(SiluMulKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)gate_up, (__hip_bfloat16*)out,
                     intermediate, in_row_stride);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_rope_partial_mrope_bf16(int64_t q, int64_t k, int64_t pos_ids, int tokens, int heads_q,
                                   int heads_k, int head_dim, int rotary_dim, float theta,
                                   int64_t stream) {
  if (tokens <= 0) return;
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
  hipLaunchKernelGGL(ArgmaxKernel, dim3(1), dim3(kThreads), 0, (hipStream_t)stream,
                     (const float*)logits, (int32_t*)out_idx, vocab);
  R4DX_HIP_CHECK(hipGetLastError());
}

}  // extern "C"
