// gemma_kernels.hip -- Gemma 4's device pieces (include/r4dx/kernels/gemma_kernels.h; docs/gemma4-plan.md
// 3.4 / 3.5, task M1-15). Its own hipcc object (src/kernels/CMakeLists.txt), built with exactly
// r4dx_kernels.hip's flags -- -ffp-contract=off among them, so every product below is a separately
// rounded multiply in the order written, and the rounding points follow the HF module chain.
//
// Everything here is plain scalar code: correctness first, a fused / vectorized perf pass is M1-34.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4dx/core/error.hpp"
#include "r4dx/kernels/gemma_device.h"
#include "r4dx/kernels/gemma_kernels.h"
#include "r4dx/kernels/kernels.h"  // r4dx_kernel_launch_counter_add

using r4dx::kernels::gemma::GeGluElement;

namespace {

constexpr int kThreads = 256;
constexpr int kGeluWideChunk = 2048;
constexpr int kMaxHeadDim = 1024;

// Same two-level reduction as r4dx_kernels.hip's BlockReduceSum (blockDim.x == kThreads).
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

[[noreturn]] void Fail(const char* who, const std::string& what) {
  throw std::runtime_error(std::string(who) + ": " + what);
}

// ---- proportional rope ---------------------------------------------------------------------------
// One workgroup per token; threads stride over the rotated pairs. The angle expression is
// RopeNeoxKernel's, with freq_dim in the exponent's denominator instead of head_dim.
__global__ void RopeProportionalKernel(__hip_bfloat16* __restrict__ q, __hip_bfloat16* __restrict__ k,
                                        const int32_t* __restrict__ pos_ids, int heads_q, int heads_k,
                                        int head_dim, int n_pairs, int freq_dim, float theta) {
  const int64_t t = blockIdx.x;
  const int half = head_dim / 2;
  const float pos = static_cast<float>(pos_ids[t]);
  for (int i = threadIdx.x; i < n_pairs; i += blockDim.x) {
    const float inv_freq = exp2f(-(2.0f * i) / static_cast<float>(freq_dim) * log2f(theta));
    const float angle = pos * inv_freq;
    float c, s;
    sincosf(angle, &s, &c);
    if (heads_q > 0) {
      __hip_bfloat16* base = q + t * static_cast<int64_t>(heads_q) * head_dim;
      for (int h = 0; h < heads_q; ++h) {
        __hip_bfloat16* row = base + static_cast<int64_t>(h) * head_dim;
        const float x0 = __bfloat162float(row[i]);
        const float x1 = __bfloat162float(row[i + half]);
        row[i] = __float2bfloat16(x0 * c - x1 * s);
        row[i + half] = __float2bfloat16(x1 * c + x0 * s);
      }
    }
    if (heads_k > 0) {
      __hip_bfloat16* base = k + t * static_cast<int64_t>(heads_k) * head_dim;
      for (int h = 0; h < heads_k; ++h) {
        __hip_bfloat16* row = base + static_cast<int64_t>(h) * head_dim;
        const float x0 = __bfloat162float(row[i]);
        const float x1 = __bfloat162float(row[i + half]);
        row[i] = __float2bfloat16(x0 * c - x1 * s);
        row[i + half] = __float2bfloat16(x1 * c + x0 * s);
      }
    }
  }
}

// ---- rmsnorm without weight ----------------------------------------------------------------------
// In place is safe: every read of the row precedes BlockReduceSum's barrier, every write follows it.
__global__ __launch_bounds__(kThreads) void RmsNormNoScaleKernel(const __hip_bfloat16* x,
                                                                   __hip_bfloat16* out, int64_t hidden,
                                                                   float eps) {
  const int64_t row = blockIdx.x;
  const __hip_bfloat16* xr = x + row * hidden;
  __hip_bfloat16* outr = out + row * hidden;
  float ss = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    const float v = __bfloat162float(xr[i]);
    ss += v * v;
  }
  ss = BlockReduceSum(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    outr[i] = __float2bfloat16(__bfloat162float(xr[i]) * rstd);
  }
}

// ---- GeGLU -----------------------------------------------------------------------------------------
// grid (rows, chunks); a workgroup takes `chunk` elements of its row, so the plain launch (chunks = 1,
// chunk = intermediate) and the wide one run the same per-element code and are bit-identical.
__global__ __launch_bounds__(kThreads) void GeluMulKernel(const __hip_bfloat16* __restrict__ gate_up,
                                                           __hip_bfloat16* __restrict__ out,
                                                           int64_t intermediate, int64_t in_row_stride,
                                                           int64_t chunk) {
  const int64_t row = blockIdx.x;
  const int64_t c0 = static_cast<int64_t>(blockIdx.y) * chunk;
  const int64_t n = intermediate - c0 < chunk ? intermediate - c0 : chunk;
  const __hip_bfloat16* gu = gate_up + row * in_row_stride + c0;
  __hip_bfloat16* outr = out + row * intermediate + c0;
  for (int64_t i = threadIdx.x; i < n; i += kThreads) {
    outr[i] = GeGluElement(__bfloat162float(gu[i]), __bfloat162float(gu[intermediate + i]));
  }
}

// ---- scaled embedding gather ---------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void EmbeddingGatherScaledKernel(
    const __hip_bfloat16* __restrict__ table, const int32_t* __restrict__ ids,
    __hip_bfloat16* __restrict__ out, int64_t hidden, int64_t vocab, float scale) {
  const int64_t row = blockIdx.x;
  int32_t id = ids[row];
  if (id < 0 || id >= vocab) id = 0;
  const __hip_bfloat16* src = table + static_cast<int64_t>(id) * hidden;
  __hip_bfloat16* dst = out + row * hidden;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    dst[i] = __float2bfloat16(__bfloat162float(src[i]) * scale);
  }
}

// ---- sandwich post-norm + residual (+ next pre-norm) -----------------------------------------------------
// Pass 1 reduces y^2. Pass 2 walks the row once per thread-strided element: normed -> sum -> scaled
// residual, stores out_res and accumulates its square. Pass 3 (only with out_normed) re-reads this
// thread's own just-stored elements (same stride, so no barrier is needed for visibility; the store
// is read back from global memory, not a register, so the value normalised is exactly the stored
// bf16) after the second reduction.
// res and out_res may alias: a thread reads res[i] and then writes out_res[i] and no other thread
// touches index i. They are deliberately not __restrict__.
__global__ __launch_bounds__(kThreads) void PostnormResidualKernel(
    const __hip_bfloat16* res, const __hip_bfloat16* __restrict__ y,
    const __hip_bfloat16* __restrict__ w_post, const __hip_bfloat16* __restrict__ w_next,
    __hip_bfloat16* out_res, __hip_bfloat16* __restrict__ out_normed, int64_t hidden, float eps,
    float scalar) {
  const int64_t row = blockIdx.x;
  const __hip_bfloat16* rr = res + row * hidden;
  const __hip_bfloat16* yr = y + row * hidden;
  __hip_bfloat16* orr = out_res + row * hidden;

  float ssy = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    const float v = __bfloat162float(yr[i]);
    ssy += v * v;
  }
  ssy = BlockReduceSum(ssy);
  const float rstd_y = rsqrtf(ssy / static_cast<float>(hidden) + eps);

  const bool scaled = scalar != 1.0f;
  float ss = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    const float normed =
        __bfloat162float(__float2bfloat16(__bfloat162float(yr[i]) * rstd_y * __bfloat162float(w_post[i])));
    float sum = __bfloat162float(__float2bfloat16(__bfloat162float(rr[i]) + normed));
    if (scaled) sum = __bfloat162float(__float2bfloat16(sum * scalar));
    orr[i] = __float2bfloat16(sum);  // sum is already bf16-exact, so this is the identity rounding
    ss += sum * sum;
  }
  if (out_normed == nullptr) return;
  ss = BlockReduceSum(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);
  __hip_bfloat16* onr = out_normed + row * hidden;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    onr[i] = __float2bfloat16(__bfloat162float(orr[i]) * rstd * __bfloat162float(w_next[i]));
  }
}

// ==== fp32-residual variants (R4DX_GEMMA_RESID=fp32, the default) =========================================
// The residual stream is fp32 [rows, hidden]; every GEMM input stays bf16. Nothing below rounds to bf16
// except the norm outputs that feed a GEMM (the `out_normed` / rmsnorm outputs). The bf16 kernels above
// are untouched and remain the R4DX_GEMMA_RESID=bf16 path.

// out (fp32) = float(table row) * scale. A bf16 value times the bf16-representable scale (62.0 = 31 * 2)
// is exact in fp32, so no rounding exists to skip or keep.
__global__ __launch_bounds__(kThreads) void EmbeddingGatherScaledF32Kernel(
    const __hip_bfloat16* __restrict__ table, const int32_t* __restrict__ ids, float* __restrict__ out,
    int64_t hidden, int64_t vocab, float scale) {
  const int64_t row = blockIdx.x;
  int32_t id = ids[row];
  if (id < 0 || id >= vocab) id = 0;
  const __hip_bfloat16* src = table + static_cast<int64_t>(id) * hidden;
  float* dst = out + row * hidden;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    dst[i] = __bfloat162float(src[i]) * scale;
  }
}

// out (bf16) = bf16(x * rsqrt(mean(x^2) + eps) * w), x fp32. Not in place (different element types).
__global__ __launch_bounds__(kThreads) void RmsNormPlainF32InKernel(const float* __restrict__ x,
                                                                      const __hip_bfloat16* __restrict__ w,
                                                                      __hip_bfloat16* __restrict__ out,
                                                                      int64_t hidden, float eps) {
  const int64_t row = blockIdx.x;
  const float* xr = x + row * hidden;
  float ss = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    const float v = xr[i];
    ss += v * v;
  }
  ss = BlockReduceSum(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);
  __hip_bfloat16* outr = out + row * hidden;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    outr[i] = __float2bfloat16(xr[i] * rstd * __bfloat162float(w[i]));
  }
}

// PostnormResidualKernel with an fp32 residual: sum = res + y*rstd_y*w_post (fp32, unrounded), times
// scalar when != 1, stored fp32; out_normed (bf16) is the next norm of the fp32 out_res. res and out_res
// may alias (same thread reads then writes index i; pass 3 reads back this thread's own stores).
__global__ __launch_bounds__(kThreads) void PostnormResidualF32Kernel(
    const float* res, const __hip_bfloat16* __restrict__ y, const __hip_bfloat16* __restrict__ w_post,
    const __hip_bfloat16* __restrict__ w_next, float* out_res, __hip_bfloat16* __restrict__ out_normed,
    int64_t hidden, float eps, float scalar) {
  const int64_t row = blockIdx.x;
  const float* rr = res + row * hidden;
  const __hip_bfloat16* yr = y + row * hidden;
  float* orr = out_res + row * hidden;

  float ssy = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    const float v = __bfloat162float(yr[i]);
    ssy += v * v;
  }
  ssy = BlockReduceSum(ssy);
  const float rstd_y = rsqrtf(ssy / static_cast<float>(hidden) + eps);

  const bool scaled = scalar != 1.0f;
  float ss = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    float sum = rr[i] + __bfloat162float(yr[i]) * rstd_y * __bfloat162float(w_post[i]);
    if (scaled) sum = sum * scalar;
    orr[i] = sum;
    ss += sum * sum;
  }
  if (out_normed == nullptr) return;
  ss = BlockReduceSum(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);
  __hip_bfloat16* onr = out_normed + row * hidden;
  for (int64_t i = threadIdx.x; i < hidden; i += kThreads) {
    onr[i] = __float2bfloat16(orr[i] * rstd * __bfloat162float(w_next[i]));
  }
}

__global__ __launch_bounds__(kThreads) void F32ToBf16Kernel(const float* __restrict__ x,
                                                              __hip_bfloat16* __restrict__ out, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < n; i += stride) {
    out[i] = __float2bfloat16(x[i]);
  }
}

}  // namespace

extern "C" {

void r4dx_embedding_gather_scaled_f32(int64_t table, int64_t ids, int64_t out, int64_t n, int64_t hidden,
                                       int64_t vocab, float scale, int64_t stream) {
  if (n <= 0) return;
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(EmbeddingGatherScaledF32Kernel, dim3(static_cast<unsigned>(n)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)table, (const int32_t*)ids, (float*)out,
                     hidden, vocab, scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_rmsnorm_plain_f32in_bf16(int64_t x, int64_t weight, int64_t out, int64_t rows, int64_t hidden,
                                    float eps, int64_t stream) {
  if (rows <= 0 || hidden <= 0) return;
  if (x == 0 || weight == 0 || out == 0) Fail("r4dx_rmsnorm_plain_f32in_bf16", "x, weight and out must be non-null");
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(RmsNormPlainF32InKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const float*)x, (const __hip_bfloat16*)weight, (__hip_bfloat16*)out,
                     hidden, eps);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_gemma_postnorm_residual_rmsnorm_f32res(int64_t res, int64_t y, int64_t w_post, int64_t w_next,
                                                  float eps, int64_t out_res, int64_t out_normed, float scalar,
                                                  int64_t rows, int64_t hidden, int64_t stream) {
  constexpr const char* kWho = "r4dx_gemma_postnorm_residual_rmsnorm_f32res";
  if (rows <= 0 || hidden <= 0) return;
  if (res == 0 || y == 0 || w_post == 0 || out_res == 0) {
    Fail(kWho, "res, y, w_post and out_res must be non-null");
  }
  if ((out_normed == 0) != (w_next == 0)) {
    Fail(kWho, "out_normed and w_next must be given together (or both null)");
  }
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(PostnormResidualF32Kernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const float*)res, (const __hip_bfloat16*)y,
                     (const __hip_bfloat16*)w_post, (const __hip_bfloat16*)w_next, (float*)out_res,
                     (__hip_bfloat16*)out_normed, hidden, eps, scalar);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_f32_to_bf16(int64_t x, int64_t out, int64_t n, int64_t stream) {
  if (n <= 0) return;
  r4dx_kernel_launch_counter_add(1);
  int64_t blocks = (n + kThreads - 1) / kThreads;
  if (blocks > 65535) blocks = 65535;
  hipLaunchKernelGGL(F32ToBf16Kernel, dim3(static_cast<unsigned>(blocks)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const float*)x, (__hip_bfloat16*)out, n);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_rope_proportional_bf16(int64_t q, int64_t k, int64_t pos, int tokens, int heads_q,
                                  int heads_k, int head_dim, int n_pairs, int freq_dim, float theta,
                                  int64_t stream) {
  constexpr const char* kWho = "r4dx_rope_proportional_bf16";
  if (head_dim <= 0 || (head_dim & 1) != 0 || head_dim > kMaxHeadDim) {
    Fail(kWho, "head_dim must be even and in [2, 1024], got " + std::to_string(head_dim));
  }
  if (n_pairs < 1 || n_pairs > head_dim / 2) {
    Fail(kWho, "n_pairs must be in [1, head_dim/2], got " + std::to_string(n_pairs));
  }
  if (freq_dim < 2 || (freq_dim & 1) != 0) {
    Fail(kWho, "freq_dim must be even and >= 2, got " + std::to_string(freq_dim));
  }
  if (tokens <= 0 || (heads_q <= 0 && heads_k <= 0)) return;
  if (heads_q > 0 && q == 0) Fail(kWho, "q is null with heads_q > 0");
  if (heads_k > 0 && k == 0) Fail(kWho, "k is null with heads_k > 0");
  r4dx_kernel_launch_counter_add(1);
  const int threads = n_pairs < kThreads ? n_pairs : kThreads;
  hipLaunchKernelGGL(RopeProportionalKernel, dim3(static_cast<unsigned>(tokens)), dim3(threads), 0,
                     (hipStream_t)stream, (__hip_bfloat16*)q, (__hip_bfloat16*)k, (const int32_t*)pos,
                     heads_q > 0 ? heads_q : 0, heads_k > 0 ? heads_k : 0, head_dim, n_pairs, freq_dim,
                     theta);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_rmsnorm_noscale_bf16(int64_t x, int64_t out, int64_t rows, int64_t hidden, float eps,
                                int64_t stream) {
  if (rows <= 0 || hidden <= 0) return;
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(RmsNormNoScaleKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)x, (__hip_bfloat16*)out, hidden, eps);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_gelu_tanh_mul_bf16(int64_t gate_up, int64_t out, int64_t rows, int64_t intermediate,
                              int64_t in_row_stride, int64_t stream) {
  if (rows <= 0 || intermediate <= 0) return;
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(GeluMulKernel, dim3(static_cast<unsigned>(rows), 1), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)gate_up, (__hip_bfloat16*)out,
                     intermediate, in_row_stride, intermediate);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_gelu_tanh_mul_wide_bf16(int64_t gate_up, int64_t out, int64_t rows, int64_t intermediate,
                                   int64_t in_row_stride, int64_t stream) {
  if (rows <= 0 || intermediate <= 0) return;
  r4dx_kernel_launch_counter_add(1);
  const int64_t chunks = (intermediate + kGeluWideChunk - 1) / kGeluWideChunk;
  hipLaunchKernelGGL(GeluMulKernel, dim3(static_cast<unsigned>(rows), static_cast<unsigned>(chunks)),
                     dim3(kThreads), 0, (hipStream_t)stream, (const __hip_bfloat16*)gate_up,
                     (__hip_bfloat16*)out, intermediate, in_row_stride,
                     static_cast<int64_t>(kGeluWideChunk));
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_embedding_gather_scaled_bf16(int64_t table, int64_t ids, int64_t out, int64_t n,
                                        int64_t hidden, int64_t vocab, float scale, int64_t stream) {
  if (n <= 0) return;
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(EmbeddingGatherScaledKernel, dim3(static_cast<unsigned>(n)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)table, (const int32_t*)ids,
                     (__hip_bfloat16*)out, hidden, vocab, scale);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_gemma_postnorm_residual_rmsnorm_bf16(int64_t res, int64_t y, int64_t w_post,
                                                int64_t w_next, float eps, int64_t out_res,
                                                int64_t out_normed, float scalar, int64_t rows,
                                                int64_t hidden, int64_t stream) {
  constexpr const char* kWho = "r4dx_gemma_postnorm_residual_rmsnorm_bf16";
  if (rows <= 0 || hidden <= 0) return;
  if (res == 0 || y == 0 || w_post == 0 || out_res == 0) {
    Fail(kWho, "res, y, w_post and out_res must be non-null");
  }
  if ((out_normed == 0) != (w_next == 0)) {
    Fail(kWho, "out_normed and w_next must be given together (or both null)");
  }
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(PostnormResidualKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)res, (const __hip_bfloat16*)y,
                     (const __hip_bfloat16*)w_post, (const __hip_bfloat16*)w_next,
                     (__hip_bfloat16*)out_res, (__hip_bfloat16*)out_normed, hidden, eps, scalar);
  R4DX_HIP_CHECK(hipGetLastError());
}

}  // extern "C"
