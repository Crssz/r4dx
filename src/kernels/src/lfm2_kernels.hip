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
#include "r4d_gdn_wmma.h"          // rowGo, v8f, v8bf, mma

namespace {

constexpr int kConvThreads = 128;
constexpr int kConvTile = 16;       // time steps per thread; T <= kConvTile is a single launch
constexpr int kRouterThreads = 1024;  // 32 waves of 32 (2 experts per wave at n_experts = 64)
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
  if (n_experts == 64) {
    const __hip_bfloat16* wr0 = gate_w + static_cast<int64_t>(wave) * hidden;
    const __hip_bfloat16* wr1 = gate_w + static_cast<int64_t>(wave + 32) * hidden;
    float acc0 = 0.0f, acc1 = 0.0f;
    for (int k = lane; k < hidden; k += 32) {
      const float xk = xs[k];
      const float p0 = xk * __bfloat162float(wr0[k]);
      const float p1 = xk * __bfloat162float(wr1[k]);
      acc0 = acc0 + p0;
      acc1 = acc1 + p1;
    }
    for (int o = 16; o; o >>= 1) {
      acc0 += __shfl_xor(acc0, o, 32);
      acc1 += __shfl_xor(acc1, o, 32);
    }
    if (lane == 0) {
      sig[wave] = 1.0f / (1.0f + expf(-acc0));
      sig[wave + 32] = 1.0f / (1.0f + expf(-acc1));
    }
  } else {
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

// ---- grouped MoE decode GEMM (bf16; docs/lfm2-plan.md 3.5 / L4-2) ---------------------------------------
// Mirror of third_party/libr4d/r4d_gemm_bf16_nt_m64.hip's r4d_gemm_bf16_nt_m64_kernel<1> at M = 1, with
// pair p = blockIdx.z indexing ids[p] -> w_ptrs[e], A + (p / a_row_group) * K, and C + p * N.
typedef __attribute__((address_space(1))) const unsigned long long* lfm2_gu2;

__device__ __forceinline__ v8bf fragU_bf16(lfm2_gu2 p, int u) {
  union { unsigned long long w[2]; v8bf f; } x;
  x.w[0] = p[4 * u];
  x.w[1] = p[4 * u + 2];
  return x.f;
}

constexpr int kMoeBf16Ku = 8;

__global__ __launch_bounds__(1024) void Bf16MoeDecodeKernel(
    const unsigned short* __restrict__ A, const uint64_t* __restrict__ w_ptrs,
    const int32_t* __restrict__ ids, __hip_bfloat16* __restrict__ C, int a_row_group,
    int n_experts, int K, int N, int WV, int SK) {
  extern __shared__ float red[];  // WV*SK*256 floats
  const int p = blockIdx.z;
  const int e = ids[p];
  if (e < 0 || e >= n_experts) return;
  const unsigned short* W = reinterpret_cast<const unsigned short*>(w_ptrs[e]);
  if (!W) return;
  const unsigned short* Ap = A + static_cast<size_t>(p / a_row_group) * static_cast<size_t>(K);
  __hip_bfloat16* Cp = C + static_cast<size_t>(p) * static_cast<size_t>(N);

  const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt = blockIdx.x * WV + col_local;
  const int n0 = nt * 16;
  const int Ks = K / SK, ksteps = Ks / 16, kbase = ksplit * Ks;

  v8f acc = (v8f)(0.f);
  const unsigned woff = rowGo((unsigned)K, n0, N - 1, lane) + (unsigned)kbase;
  const unsigned aoff = rowGo((unsigned)K, 0, 0, lane) + (unsigned)kbase;

  if (n0 < N) {
    lfm2_gu2 pw = (lfm2_gu2)(const unsigned long long*)(W + woff);
    lfm2_gu2 pa = (lfm2_gu2)(const unsigned long long*)(Ap + aoff);
    int s = 0;
    for (; s + kMoeBf16Ku <= ksteps; s += kMoeBf16Ku) {
#pragma unroll
      for (int u = 0; u < kMoeBf16Ku; u++) {
        const v8bf b = fragU_bf16(pw, u);
        acc = mma(fragU_bf16(pa, u), b, acc);
      }
      pw += kMoeBf16Ku * 4;
      pa += kMoeBf16Ku * 4;
    }
    for (; s < ksteps; s++) {
      const v8bf b = fragU_bf16(pw, 0);
      pw += 4;
      acc = mma(fragU_bf16(pa, 0), b, acc);
      pa += 4;
    }
  }

  const int slot = (col_local * SK + ksplit) * 256;
  const int nthreads = WV * SK * 32;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; j++) red[slot + j * 32 + lane] = acc[j];
  __syncthreads();
  for (int idx = tid; idx < WV * 256; idx += nthreads) {
    const int col = idx >> 8, off = idx & 255;
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
    const int l = off & 31;
    const int n = (blockIdx.x * WV + col) * 16 + (l & 15);
    const int m = 8 * (l >> 4) + (off >> 5);
    if (n < N && m < 1) Cp[n] = (__hip_bfloat16)v;
  }
}

__device__ __forceinline__ bool ResolveMoePrefillTile(const int32_t* __restrict__ offsets, int n_experts,
                                                      int n_pairs, int tile_idx, int& out_e,
                                                      int& out_row_start, int& out_m_rows) {
  int rem = tile_idx;
  for (int e = 0; e < n_experts; ++e) {
    const int r0 = offsets[e];
    const int r1 = offsets[e + 1];
    const int c = r1 - r0;
    if (c <= 0) continue;
    const int nt = (c + 15) >> 4;
    if (rem < nt) {
      out_e = e;
      out_row_start = r0 + (rem << 4);
      out_m_rows = (r1 - out_row_start) < 16 ? (r1 - out_row_start) : 16;
      return out_row_start >= 0 && out_row_start + out_m_rows <= n_pairs;
    }
    rem -= nt;
  }
  return false;
}

__global__ __launch_bounds__(1024) void Bf16MoePrefillKernel(
    const unsigned short* __restrict__ A, const uint64_t* __restrict__ w_ptrs,
    const int32_t* __restrict__ offsets, __hip_bfloat16* __restrict__ C, int n_pairs, int n_experts,
    int K, int N, int WV, int SK) {
  extern __shared__ float red[];  // WV*SK*256 floats
  int e = -1, row_start = 0, m_rows = 0;
  if (!ResolveMoePrefillTile(offsets, n_experts, n_pairs, static_cast<int>(blockIdx.y), e, row_start,
                             m_rows)) {
    return;
  }
  const unsigned short* W = reinterpret_cast<const unsigned short*>(w_ptrs[e]);
  if (!W) return;
  const unsigned short* Ap = A + static_cast<size_t>(row_start) * static_cast<size_t>(K);
  __hip_bfloat16* Cp = C + static_cast<size_t>(row_start) * static_cast<size_t>(N);

  const int tid = threadIdx.x, wave = tid / 32, lane = tid % 32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt = blockIdx.x * WV + col_local;
  const int n0 = nt * 16;
  const int Ks = K / SK, ksteps = Ks / 16, kbase = ksplit * Ks;

  v8f acc = (v8f)(0.f);
  const unsigned woff = rowGo((unsigned)K, n0, N - 1, lane) + (unsigned)kbase;
  const unsigned aoff = rowGo((unsigned)K, 0, m_rows - 1, lane) + (unsigned)kbase;

  if (n0 < N) {
    lfm2_gu2 pw = (lfm2_gu2)(const unsigned long long*)(W + woff);
    lfm2_gu2 pa = (lfm2_gu2)(const unsigned long long*)(Ap + aoff);
    int s = 0;
    for (; s + kMoeBf16Ku <= ksteps; s += kMoeBf16Ku) {
#pragma unroll
      for (int u = 0; u < kMoeBf16Ku; u++) {
        const v8bf b = fragU_bf16(pw, u);
        acc = mma(fragU_bf16(pa, u), b, acc);
      }
      pw += kMoeBf16Ku * 4;
      pa += kMoeBf16Ku * 4;
    }
    for (; s < ksteps; s++) {
      const v8bf b = fragU_bf16(pw, 0);
      pw += 4;
      acc = mma(fragU_bf16(pa, 0), b, acc);
      pa += 4;
    }
  }

  const int slot = (col_local * SK + ksplit) * 256;
  const int nthreads = WV * SK * 32;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < 8; j++) red[slot + j * 32 + lane] = acc[j];
  __syncthreads();
  for (int idx = tid; idx < WV * 256; idx += nthreads) {
    const int col = idx >> 8, off = idx & 255;
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
    const int l = off & 31;
    const int n = (blockIdx.x * WV + col) * 16 + (l & 15);
    const int m = 8 * (l >> 4) + (off >> 5);
    if (n < N && m < m_rows) Cp[static_cast<size_t>(m) * N + n] = (__hip_bfloat16)v;
  }
}

// ---- L4-5 fusions: residual_rmsnorm_plain + attn_precore -------------------------------------------------
constexpr int kNormThreads = 256;

__device__ __forceinline__ float BlockReduceSum256(float v) {
  __shared__ float red[kNormThreads / 32];
  for (int o = 16; o; o >>= 1) v += __shfl_xor(v, o, 32);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float out = 0.0f;
  if (threadIdx.x < kNormThreads / 32) out = red[threadIdx.x];
  if (threadIdx.x < 32) {
    for (int o = 16; o; o >>= 1) out += __shfl_xor(out, o, 32);
  }
  __shared__ float total;
  if (threadIdx.x == 0) total = out;
  __syncthreads();
  return total;
}

__global__ __launch_bounds__(kNormThreads) void ResidualRmsNormPlainKernel(
    __hip_bfloat16* __restrict__ h, const __hip_bfloat16* __restrict__ sub,
    const __hip_bfloat16* __restrict__ w, __hip_bfloat16* __restrict__ x_out, int64_t hidden,
    float eps) {
  const int64_t row = blockIdx.x;
  __hip_bfloat16* hr = h + row * hidden;
  const __hip_bfloat16* sr = sub + row * hidden;
  __hip_bfloat16* outr = x_out + row * hidden;

  float ss = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += kNormThreads) {
    const __hip_bfloat16 hn = __float2bfloat16(__bfloat162float(hr[i]) + __bfloat162float(sr[i]));
    hr[i] = hn;
    const float v = __bfloat162float(hn);
    ss += v * v;
  }
  ss = BlockReduceSum256(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);
  for (int64_t i = threadIdx.x; i < hidden; i += kNormThreads) {
    outr[i] = __float2bfloat16(__bfloat162float(hr[i]) * rstd * __bfloat162float(w[i]));
  }
}

// Grid (T, heads_q + heads_k), block (64): one workgroup per (token t, q-or-k head).
__global__ __launch_bounds__(64) void Lfm2AttnPrecoreKernel(
    __hip_bfloat16* __restrict__ q, __hip_bfloat16* __restrict__ k,
    const __hip_bfloat16* __restrict__ v, const __hip_bfloat16* __restrict__ q_norm,
    const __hip_bfloat16* __restrict__ k_norm, const int32_t* __restrict__ pos_ids,
    __hip_bfloat16* __restrict__ kv_cache, int heads_q, int heads_k, float rms_eps,
    float rope_theta, int block_size, int64_t kv_block_stride, int64_t kv_head_stride) {
  constexpr int kHd = 64;
  constexpr int kHalf = 32;
  __shared__ float red[2];
  __shared__ float sh_row[kHd];

  const int t = static_cast<int>(blockIdx.x);
  const int h_idx = static_cast<int>(blockIdx.y);
  const bool is_q = h_idx < heads_q;
  const int h = is_q ? h_idx : (h_idx - heads_q);
  __hip_bfloat16* row = is_q ? (q + (static_cast<int64_t>(t) * heads_q + h) * kHd)
                             : (k + (static_cast<int64_t>(t) * heads_k + h) * kHd);
  const __hip_bfloat16* nw = is_q ? q_norm : k_norm;

  const int d = static_cast<int>(threadIdx.x);
  const float val = __bfloat162float(row[d]);
  float ss = val * val;
  for (int o = 16; o; o >>= 1) ss += __shfl_xor(ss, o, 32);
  if ((d & 31) == 0) red[d >> 5] = ss;
  __syncthreads();
  const float total = red[0] + red[1];
  const float rstd = rsqrtf(total / 64.0f + rms_eps);
  const __hip_bfloat16 normed = __float2bfloat16(val * rstd * __bfloat162float(nw[d]));
  sh_row[d] = __bfloat162float(normed);
  __syncthreads();

  const int slot = pos_ids[t];
  if (d < kHalf) {
    const float inv_freq = exp2f(-(2.0f * d) / 64.0f * log2f(rope_theta));
    const float angle = static_cast<float>(slot) * inv_freq;
    float s, c;
    sincosf(angle, &s, &c);
    const float x0 = sh_row[d];
    const float x1 = sh_row[d + kHalf];
    const __hip_bfloat16 r0 = __float2bfloat16(x0 * c - x1 * s);
    const __hip_bfloat16 r1 = __float2bfloat16(x1 * c + x0 * s);
    row[d] = r0;
    row[d + kHalf] = r1;
    if (!is_q && slot >= 0) {
      const int64_t base = static_cast<int64_t>(slot / block_size) * kv_block_stride +
                           static_cast<int64_t>(h) * kv_head_stride +
                           static_cast<int64_t>(slot % block_size) * (2 * kHd);
      kv_cache[base + d] = r0;
      kv_cache[base + d + kHalf] = r1;
    }
  }
  if (!is_q && slot >= 0) {
    const int64_t base = static_cast<int64_t>(slot / block_size) * kv_block_stride +
                         static_cast<int64_t>(h) * kv_head_stride +
                         static_cast<int64_t>(slot % block_size) * (2 * kHd);
    const int64_t v_src = (static_cast<int64_t>(t) * heads_k + h) * kHd;
    kv_cache[base + kHd + d] = v[v_src + d];
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

void r4dx_lfm2_gemm_bf16_moe_decode(int64_t a_bf16, int64_t w_ptrs, int64_t ids, int64_t c_bf16,
                                    int n_pairs, int a_row_group, int n_experts, int K, int N,
                                    int WV, int SK, int64_t stream) {
  const char* who = "r4dx_lfm2_gemm_bf16_moe_decode";
  if (n_pairs < 1 || n_pairs > kSortMaxPairs) {
    Fail(who, "n_pairs must be in [1, 16384], got " + std::to_string(n_pairs));
  }
  if (a_row_group < 1 || a_row_group > n_pairs || n_pairs % a_row_group != 0) {
    Fail(who, "a_row_group must divide n_pairs in [1, n_pairs], got a_row_group " +
                  std::to_string(a_row_group) + ", n_pairs " + std::to_string(n_pairs));
  }
  if (n_experts < 1 || n_experts > kSortMaxExperts) {
    Fail(who, "n_experts must be in [1, 256], got " + std::to_string(n_experts));
  }
  if (N < 16 || N % 16 != 0) {
    Fail(who, "N must be a positive multiple of 16, got " + std::to_string(N));
  }
  if (WV < 1 || SK < 1) Fail(who, "WV and SK must be >= 1");
  if (K < 16 || K % (SK * 16) != 0) {
    Fail(who, "K must be divisible by SK*16 (K " + std::to_string(K) + ", SK " + std::to_string(SK) + ")");
  }
  if (WV * SK * 32 > 1024) Fail(who, "WV*SK*32 > 1024 threads per block");
  const size_t smem = static_cast<size_t>(WV) * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) Fail(who, "LDS WV*SK KiB > 64");
  if (!a_bf16 || !w_ptrs || !ids || !c_bf16) Fail(who, "null pointer");
  const int ntiles = N / 16;
  const dim3 grid(static_cast<unsigned>((ntiles + WV - 1) / WV), 1u, static_cast<unsigned>(n_pairs));
  const dim3 block(static_cast<unsigned>(WV * SK * 32));
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(Bf16MoeDecodeKernel, grid, block, smem, (hipStream_t)stream,
                     (const unsigned short*)a_bf16, (const uint64_t*)w_ptrs, (const int32_t*)ids,
                     (__hip_bfloat16*)c_bf16, a_row_group, n_experts, K, N, WV, SK);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_lfm2_gemm_bf16_moe_prefill(int64_t a_bf16, int64_t w_ptrs, int64_t expert_offsets,
                                     int64_t c_bf16, int n_pairs, int n_experts, int K, int N,
                                     int WV, int SK, int64_t stream) {
  const char* who = "r4dx_lfm2_gemm_bf16_moe_prefill";
  if (n_pairs < 1 || n_pairs > kSortMaxPairs) {
    Fail(who, "n_pairs must be in [1, 16384], got " + std::to_string(n_pairs));
  }
  if (n_experts < 1 || n_experts > kSortMaxExperts) {
    Fail(who, "n_experts must be in [1, 256], got " + std::to_string(n_experts));
  }
  if (N < 16 || N % 16 != 0) Fail(who, "N must be a positive multiple of 16, got " + std::to_string(N));
  if (WV < 1 || SK < 1) Fail(who, "WV and SK must be >= 1");
  if (K < 16 || K % (SK * 16) != 0) {
    Fail(who, "K must be divisible by SK*16 (K " + std::to_string(K) + ", SK " + std::to_string(SK) + ")");
  }
  if (WV * SK * 32 > 1024) Fail(who, "WV*SK*32 > 1024 threads per block");
  const size_t smem = static_cast<size_t>(WV) * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) Fail(who, "LDS WV*SK KiB > 64");
  if (!a_bf16 || !w_ptrs || !expert_offsets || !c_bf16) Fail(who, "null pointer");
  const int ntiles = N / 16;
  const int max_tiles = n_pairs < ((n_pairs + 15) / 16 + n_experts) ? n_pairs : ((n_pairs + 15) / 16 + n_experts);
  const dim3 grid(static_cast<unsigned>((ntiles + WV - 1) / WV), static_cast<unsigned>(max_tiles));
  const dim3 block(static_cast<unsigned>(WV * SK * 32));
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(Bf16MoePrefillKernel, grid, block, smem, (hipStream_t)stream,
                     (const unsigned short*)a_bf16, (const uint64_t*)w_ptrs,
                     (const int32_t*)expert_offsets, (__hip_bfloat16*)c_bf16, n_pairs, n_experts, K, N,
                     WV, SK);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_lfm2_residual_rmsnorm_plain_bf16(int64_t h, int64_t sub, int64_t weight, int64_t x_out,
                                           int64_t rows, int64_t hidden, float eps, int64_t stream) {
  if (rows <= 0 || hidden <= 0) return;
  if (!h || !sub || !weight || !x_out) {
    Fail("r4dx_lfm2_residual_rmsnorm_plain_bf16", "null pointer");
  }
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(ResidualRmsNormPlainKernel, dim3(static_cast<unsigned>(rows)),
                     dim3(kNormThreads), 0, (hipStream_t)stream, (__hip_bfloat16*)h,
                     (const __hip_bfloat16*)sub, (const __hip_bfloat16*)weight,
                     (__hip_bfloat16*)x_out, hidden, eps);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_lfm2_attn_precore_bf16(int64_t q, int64_t k, int64_t v, int64_t q_norm, int64_t k_norm,
                                 int64_t pos_ids, int64_t kv_cache, int T, int heads_q, int heads_k,
                                 int head_dim, float rms_eps, float rope_theta, int block_size,
                                 int64_t kv_block_stride, int64_t kv_head_stride, int64_t stream) {
  const char* who = "r4dx_lfm2_attn_precore_bf16";
  if (T <= 0) return;
  if (head_dim != 64) Fail(who, "head_dim must be 64, got " + std::to_string(head_dim));
  if (heads_q < 1 || heads_k < 1) Fail(who, "heads_q and heads_k must be >= 1");
  if (block_size < 1) Fail(who, "block_size must be >= 1");
  if (!q || !k || !v || !q_norm || !k_norm || !pos_ids || !kv_cache) Fail(who, "null pointer");
  r4dx_kernel_launch_counter_add(1);
  const dim3 grid(static_cast<unsigned>(T), static_cast<unsigned>(heads_q + heads_k));
  hipLaunchKernelGGL(Lfm2AttnPrecoreKernel, grid, dim3(64), 0, (hipStream_t)stream,
                     (__hip_bfloat16*)q, (__hip_bfloat16*)k, (const __hip_bfloat16*)v,
                     (const __hip_bfloat16*)q_norm, (const __hip_bfloat16*)k_norm,
                     (const int32_t*)pos_ids, (__hip_bfloat16*)kv_cache, heads_q, heads_k, rms_eps,
                     rope_theta, block_size, kv_block_stride, kv_head_stride);
  R4DX_HIP_CHECK(hipGetLastError());
}

}  // extern "C"
