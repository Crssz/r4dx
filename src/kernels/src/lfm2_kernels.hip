// lfm2_kernels.hip -- LFM2.5-24B-A2B device pieces (include/r4dx/kernels/lfm2_kernels.h; docs/lfm2-plan.md
// 3.3 / 3.5, task L1-6). Its own hipcc object (src/kernels/CMakeLists.txt), built with exactly
// r4dx_kernels.hip's flags -- -ffp-contract=off among them, so every product below is a separately
// rounded multiply in the order written, and the rounding points follow the HF module chain.
//
// Correctness first: plain scalar code, a fused / vectorized perf pass is L4.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4dx/core/error.hpp"
#include "r4dx/kernels/lfm2_kernels.h"
#include "r4dx/kernels/kernels.h"  // r4dx_kernel_launch_counter_add

namespace {

constexpr int kConvThreads = 128;
constexpr int kConvTile = 16;       // time steps per thread; T <= kConvTile is a single launch
constexpr int kRouterThreads = 256;  // 8 waves of 32
constexpr int kMaxExperts = 64;
constexpr int kMaxTopK = 8;
constexpr int kSortMaxExperts = 256;
constexpr int kSortMaxPairs = 16384;
constexpr int kMaxHidden = 8192;

[[noreturn]] void Fail(const char* who, const std::string& what) {
  throw std::runtime_error(std::string(who) + ": " + what);
}

__device__ __forceinline__ float Bf16Round(float v) { return __bfloat162float(__float2bfloat16(v)); }

// ---- ShortConv -------------------------------------------------------------------------------------------
// One thread owns one channel and kConvTile consecutive time steps, keeping h_{t-2}, h_{t-1} in registers.
// The two halo values of a tile that does not start at row 0 are recomputed from bcx (B * x of the two rows
// before the tile) -- the same bf16 values the previous tile's thread computes -- so tiles are independent.
// When the whole chunk is one tile (gridDim.y == 1) the thread's register window ends as exactly the new
// state (h_{T-2}, h_{T-1}), and it writes it back itself: it is the only thread that reads or writes this
// channel's state column, so no ordering problem. Otherwise the state micro-kernel does it afterwards.
__global__ __launch_bounds__(kConvThreads) void ShortConvKernel(
    const __hip_bfloat16* __restrict__ bcx, __hip_bfloat16* __restrict__ state,
    const __hip_bfloat16* __restrict__ weight, __hip_bfloat16* __restrict__ out, int T, int H,
    int64_t stride, int update_state) {
  const int c = blockIdx.x * kConvThreads + threadIdx.x;
  if (c >= H) return;
  const int t0 = blockIdx.y * kConvTile;
  int t1 = t0 + kConvTile;
  if (t1 > T) t1 = T;

  float h2, h1;
  if (t0 == 0) {
    h2 = __bfloat162float(state[c]);
    h1 = __bfloat162float(state[H + c]);
  } else {
    const __hip_bfloat16* r2 = bcx + static_cast<int64_t>(t0 - 2) * stride;
    const __hip_bfloat16* r1 = bcx + static_cast<int64_t>(t0 - 1) * stride;
    h2 = Bf16Round(__bfloat162float(r2[c]) * __bfloat162float(r2[2 * H + c]));
    h1 = Bf16Round(__bfloat162float(r1[c]) * __bfloat162float(r1[2 * H + c]));
  }
  const float w0 = __bfloat162float(weight[static_cast<int64_t>(c) * 3 + 0]);
  const float w1 = __bfloat162float(weight[static_cast<int64_t>(c) * 3 + 1]);
  const float w2 = __bfloat162float(weight[static_cast<int64_t>(c) * 3 + 2]);

  for (int t = t0; t < t1; ++t) {
    const __hip_bfloat16* row = bcx + static_cast<int64_t>(t) * stride;
    const float b = __bfloat162float(row[c]);
    const float cc = __bfloat162float(row[H + c]);
    const float x = __bfloat162float(row[2 * H + c]);
    const float h0 = Bf16Round(b * x);
    const float p0 = w0 * h2;
    const float p1 = w1 * h1;
    const float p2 = w2 * h0;
    const float acc = Bf16Round((p0 + p1) + p2);
    out[static_cast<int64_t>(t) * H + c] = __float2bfloat16(cc * acc);
    h2 = h1;
    h1 = h0;
  }
  if (update_state) {
    state[c] = __float2bfloat16(h2);
    state[H + c] = __float2bfloat16(h1);
  }
}

// New state from the last two rows of the chunk (T >= 2). Runs after ShortConvKernel in stream order.
__global__ void ShortConvStateKernel(const __hip_bfloat16* __restrict__ bcx,
                                      __hip_bfloat16* __restrict__ state, int T, int H, int64_t stride) {
  const int c = blockIdx.x * kConvThreads + threadIdx.x;
  if (c >= H) return;
  const __hip_bfloat16* r2 = bcx + static_cast<int64_t>(T - 2) * stride;
  const __hip_bfloat16* r1 = bcx + static_cast<int64_t>(T - 1) * stride;
  state[c] = __float2bfloat16(__bfloat162float(r2[c]) * __bfloat162float(r2[2 * H + c]));
  state[H + c] = __float2bfloat16(__bfloat162float(r1[c]) * __bfloat162float(r1[2 * H + c]));
}

// ---- router ----------------------------------------------------------------------------------------------
// Block = 8 waves. Phase 1: x row -> LDS (fp32); wave w computes experts w, w+8, ..: lane l sums k = l,
// l+32, .. sequentially in fp32, then an xor-shuffle tree. Phase 2 (wave 0): sigmoid + bias, then top_k
// rounds of a wave-wide argmax (higher score wins, equal scores -> lower expert index). Phase 3 (lane 0):
// normalize.
__global__ __launch_bounds__(kRouterThreads) void RouterKernel(
    const __hip_bfloat16* __restrict__ x, const __hip_bfloat16* __restrict__ gate_w,
    const float* __restrict__ bias, int32_t* __restrict__ ids, float* __restrict__ wout, int hidden,
    int n_experts, int top_k, float scale, int64_t x_stride) {
  extern __shared__ float xs[];  // [hidden]
  __shared__ float sig[kMaxExperts];
  __shared__ int sel[kMaxTopK];

  const int64_t t = blockIdx.x;
  const __hip_bfloat16* xr = x + t * x_stride;
  for (int k = threadIdx.x; k < hidden; k += kRouterThreads) xs[k] = __bfloat162float(xr[k]);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  for (int e = wave; e < n_experts; e += kRouterThreads / 32) {
    const __hip_bfloat16* wr = gate_w + static_cast<int64_t>(e) * hidden;
    float acc = 0.0f;
    for (int k = lane; k < hidden; k += 32) {
      const float p = xs[k] * __bfloat162float(wr[k]);
      acc = acc + p;
    }
    for (int o = 16; o; o >>= 1) acc += __shfl_xor(acc, o, 32);
    if (lane == 0) sig[e] = 1.0f / (1.0f + expf(-acc));
  }
  __syncthreads();

  if (wave == 0) {
    // lane l owns experts l and l + 32
    float sc0 = (lane < n_experts) ? sig[lane] + bias[lane] : -INFINITY;
    float sc1 = (lane + 32 < n_experts) ? sig[lane + 32] + bias[lane + 32] : -INFINITY;
    for (int r = 0; r < top_k; ++r) {
      float best = sc0;
      int bi = lane;
      if (sc1 > sc0) {  // strict: an equal score keeps the lower index (lane < lane + 32)
        best = sc1;
        bi = lane + 32;
      }
      for (int o = 16; o; o >>= 1) {
        const float os = __shfl_xor(best, o, 32);
        const int oi = __shfl_xor(bi, o, 32);
        if (os > best || (os == best && oi < bi)) {
          best = os;
          bi = oi;
        }
      }
      if (lane == 0) sel[r] = bi;
      if (bi == lane) sc0 = -INFINITY;
      if (bi == lane + 32) sc1 = -INFINITY;
    }
    if (lane == 0) {
      float sum = 0.0f;
      for (int r = 0; r < top_k; ++r) sum = sum + sig[sel[r]];
      const float denom = sum + 1e-6f;
      for (int r = 0; r < top_k; ++r) {
        ids[t * top_k + r] = sel[r];
        const float q = sig[sel[r]] / denom;
        wout[t * top_k + r] = q * scale;
      }
    }
  }
}

// ---- expert counting sort ----------------------------------------------------------------------------------
// One workgroup, one thread per expert (<= 256). The ids are staged in LDS as bytes; thread e counts its
// pairs by scanning ascending p, thread 0 forms the exclusive scan, then thread e scans again and writes
// its pairs in ascending p. Deterministic by construction (no atomics, fixed scan order).
__global__ __launch_bounds__(kSortMaxExperts) void MoeSortKernel(
    const int32_t* __restrict__ ids, int P, int n_experts, int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets, int32_t* __restrict__ sorted_pairs, int32_t* __restrict__ pos_of_pair) {
  __shared__ uint8_t id_s[kSortMaxPairs];  // 255 = invalid
  __shared__ int32_t off_s[kSortMaxExperts + 1];
  for (int p = threadIdx.x; p < P; p += blockDim.x) {
    const int32_t e = ids[p];
    const bool ok = e >= 0 && e < n_experts;
    id_s[p] = ok ? static_cast<uint8_t>(e) : static_cast<uint8_t>(255);
    if (!ok) pos_of_pair[p] = -1;
  }
  __syncthreads();
  const int e = threadIdx.x;
  int cnt = 0;
  if (e < n_experts) {
    for (int p = 0; p < P; ++p) cnt += (id_s[p] == e) ? 1 : 0;
    counts[e] = cnt;
  }
  if (e < n_experts) off_s[e] = cnt;
  __syncthreads();
  if (threadIdx.x == 0) {
    int run = 0;
    for (int i = 0; i < n_experts; ++i) {
      const int c = off_s[i];
      off_s[i] = run;
      run += c;
    }
    off_s[n_experts] = run;
  }
  __syncthreads();
  if (e < n_experts) {
    offsets[e] = off_s[e];
    int pos = off_s[e];
    for (int p = 0; p < P; ++p) {
      if (id_s[p] == e) {
        sorted_pairs[pos] = p;
        pos_of_pair[p] = pos;
        ++pos;
      }
    }
  }
  if (e == 0) offsets[n_experts] = off_s[n_experts];
}

__global__ void MoeGatherKernel(const __hip_bfloat16* __restrict__ x, const int32_t* __restrict__ sorted_pairs,
                                 __hip_bfloat16* __restrict__ out, int n_pairs, int top_k, int hidden,
                                 int64_t x_stride, const int32_t* __restrict__ valid_pairs) {
  const int64_t i = blockIdx.x;
  const int n_valid = valid_pairs ? valid_pairs[0] : n_pairs;
  if (i >= n_valid) return;
  const int64_t tok = sorted_pairs[i] / top_k;
  const __hip_bfloat16* src = x + tok * x_stride;
  __hip_bfloat16* dst = out + i * hidden;
  for (int k = threadIdx.x; k < hidden; k += blockDim.x) dst[k] = src[k];
}

// ---- combine -----------------------------------------------------------------------------------------------
__global__ void MoeCombineKernel(const __hip_bfloat16* __restrict__ y, const int32_t* __restrict__ pos_of_pair,
                                  const float* __restrict__ w, __hip_bfloat16* __restrict__ out, int top_k,
                                  int hidden) {
  const int64_t t = blockIdx.x;
  for (int k = blockIdx.y * blockDim.x + threadIdx.x; k < hidden; k += gridDim.y * blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < top_k; ++s) {
      const int64_t p = t * top_k + s;
      const int64_t row = pos_of_pair ? static_cast<int64_t>(pos_of_pair[p]) : p;
      if (row < 0) continue;
      const float prod = w[p] * __bfloat162float(y[row * hidden + k]);
      acc = acc + prod;
    }
    out[t * hidden + k] = __float2bfloat16(acc);
  }
}

}  // namespace

extern "C" {

void r4dx_lfm2_shortconv_bf16(int64_t bcx, int64_t state, int64_t weight, int64_t out, int T, int H,
                               int64_t bcx_row_stride, int64_t stream) {
  if (T <= 0) return;
  if (H < 1) Fail("r4dx_lfm2_shortconv_bf16", "H must be >= 1, got " + std::to_string(H));
  if (bcx_row_stride < 3 * static_cast<int64_t>(H)) {
    Fail("r4dx_lfm2_shortconv_bf16", "bcx_row_stride " + std::to_string(bcx_row_stride) + " < 3H");
  }
  const unsigned gx = static_cast<unsigned>((H + kConvThreads - 1) / kConvThreads);
  const unsigned gy = static_cast<unsigned>((T + kConvTile - 1) / kConvTile);
  r4dx_kernel_launch_counter_add(gy > 1 ? 2 : 1);
  hipLaunchKernelGGL(ShortConvKernel, dim3(gx, gy), dim3(kConvThreads), 0, (hipStream_t)stream,
                     (const __hip_bfloat16*)bcx, (__hip_bfloat16*)state, (const __hip_bfloat16*)weight,
                     (__hip_bfloat16*)out, T, H, bcx_row_stride, gy == 1 ? 1 : 0);
  R4DX_HIP_CHECK(hipGetLastError());
  if (gy > 1) {
    hipLaunchKernelGGL(ShortConvStateKernel, dim3(gx), dim3(kConvThreads), 0, (hipStream_t)stream,
                       (const __hip_bfloat16*)bcx, (__hip_bfloat16*)state, T, H, bcx_row_stride);
    R4DX_HIP_CHECK(hipGetLastError());
  }
}

void r4dx_lfm2_router_f32(int64_t x, int64_t gate_w, int64_t expert_bias, int64_t ids, int64_t w, int T,
                           int hidden, int n_experts, int top_k, float routed_scaling_factor,
                           int64_t x_row_stride, int64_t stream) {
  if (T <= 0) return;
  if (n_experts < 1 || n_experts > kMaxExperts) {
    Fail("r4dx_lfm2_router_f32", "n_experts must be in [1, 64], got " + std::to_string(n_experts));
  }
  if (top_k < 1 || top_k > kMaxTopK || top_k > n_experts) {
    Fail("r4dx_lfm2_router_f32", "top_k must be in [1, min(8, n_experts)], got " + std::to_string(top_k));
  }
  if (hidden < 1 || hidden > kMaxHidden) {
    Fail("r4dx_lfm2_router_f32", "hidden must be in [1, 8192], got " + std::to_string(hidden));
  }
  if (x_row_stride < hidden) Fail("r4dx_lfm2_router_f32", "x_row_stride < hidden");
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(RouterKernel, dim3(static_cast<unsigned>(T)), dim3(kRouterThreads),
                     static_cast<size_t>(hidden) * sizeof(float), (hipStream_t)stream,
                     (const __hip_bfloat16*)x, (const __hip_bfloat16*)gate_w, (const float*)expert_bias,
                     (int32_t*)ids, (float*)w, hidden, n_experts, top_k, routed_scaling_factor,
                     x_row_stride);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_lfm2_moe_sort(int64_t ids, int T, int top_k, int n_experts, int64_t expert_counts,
                         int64_t expert_offsets, int64_t sorted_pairs, int64_t pos_of_pair,
                         int64_t stream) {
  if (T <= 0 || top_k <= 0) return;
  const int64_t P = static_cast<int64_t>(T) * top_k;
  if (n_experts < 1 || n_experts > kSortMaxExperts) {
    Fail("r4dx_lfm2_moe_sort", "n_experts must be in [1, 256], got " + std::to_string(n_experts));
  }
  if (P > kSortMaxPairs) Fail("r4dx_lfm2_moe_sort", "T * top_k must be <= 16384, got " + std::to_string(P));
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(MoeSortKernel, dim3(1), dim3(kSortMaxExperts), 0, (hipStream_t)stream,
                     (const int32_t*)ids, static_cast<int>(P), n_experts, (int32_t*)expert_counts,
                     (int32_t*)expert_offsets, (int32_t*)sorted_pairs, (int32_t*)pos_of_pair);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_lfm2_moe_gather_bf16(int64_t x, int64_t sorted_pairs, int64_t out, int n_pairs, int top_k,
                                int hidden, int64_t x_row_stride, int64_t valid_pairs, int64_t stream) {
  if (n_pairs <= 0 || hidden <= 0) return;
  if (top_k <= 0) Fail("r4dx_lfm2_moe_gather_bf16", "top_k must be >= 1");
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(MoeGatherKernel, dim3(static_cast<unsigned>(n_pairs)), dim3(256), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)x, (const int32_t*)sorted_pairs,
                     (__hip_bfloat16*)out, n_pairs, top_k, hidden, x_row_stride,
                     (const int32_t*)valid_pairs);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_lfm2_moe_combine_bf16(int64_t y, int64_t pos_of_pair, int64_t w, int64_t out, int T, int top_k,
                                 int hidden, int64_t stream) {
  if (T <= 0 || hidden <= 0) return;
  if (top_k <= 0) Fail("r4dx_lfm2_moe_combine_bf16", "top_k must be >= 1");
  r4dx_kernel_launch_counter_add(1);
  const unsigned gy = static_cast<unsigned>((hidden + 255) / 256);
  hipLaunchKernelGGL(MoeCombineKernel, dim3(static_cast<unsigned>(T), gy), dim3(256), 0,
                     (hipStream_t)stream, (const __hip_bfloat16*)y, (const int32_t*)pos_of_pair,
                     (const float*)w, (__hip_bfloat16*)out, top_k, hidden);
  R4DX_HIP_CHECK(hipGetLastError());
}

}  // extern "C"
