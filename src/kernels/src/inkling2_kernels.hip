// inkling2_kernels.hip -- implements include/r4dx/kernels/inkling2_kernels.h (docs/inkling2-plan.md 3.3, 5). The spec
// of each entry point is include/r4dx/kernels/inkling2_ref.hpp; the expressions below are the same, term for term,
// with sums over long axes done by block reductions (different association, so the GPU tests use a tolerance).
// Built with r4dx_kernels.hip's flags (-ffp-contract=off: every a * b + c is two roundings, as in the reference).
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4d_attn_window.h"
#include "r4dx/core/dtype.hpp"
#include "r4dx/kernels/inkling2_kernels.h"
#include "r4dx/kernels/kernels.h"  // r4dx_kernel_launch_counter_add

namespace {

constexpr int kThreads = 256;
constexpr int kWaves = kThreads / 32;

[[noreturn]] void Fail(const char* fn, const std::string& what) { throw std::runtime_error(std::string(fn) + ": " + what); }

void CheckLaunch(const char* fn) {
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) Fail(fn, std::string("launch failed: ") + hipGetErrorString(e));
  r4dx_kernel_launch_counter_add(1);
}

__device__ __forceinline__ float LoadF(const void* p, int64_t i, int bf16) {
  return bf16 ? r4dx::core::Bf16ToFloat(static_cast<const uint16_t*>(p)[i]) : static_cast<const float*>(p)[i];
}

__device__ __forceinline__ float WaveSum(float v) {
  for (int o = 16; o; o >>= 1) v += __shfl_xor(v, o, 32);
  return v;
}

// Sum over the 256-thread block; `red` holds kWaves floats. Every thread gets the total.
__device__ __forceinline__ float BlockSum(float v, float* red) {
  v = WaveSum(v);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float out = red[0];
  for (int i = 1; i < kWaves; ++i) out += red[i];
  __syncthreads();
  return out;
}

__device__ __forceinline__ float SigmoidF(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---- attnres ---------------------------------------------------------------------------------------------------
constexpr int kMaxStates = 6;
constexpr int kMaxQueries = 4;
struct AttnResPtrs {
  const float* s[kMaxStates];
};

__global__ __launch_bounds__(kThreads) void AttnResKernel(AttnResPtrs st, int n_states, const float* __restrict__ q,
                                                          int nq, float* __restrict__ out, int T, int H, float eps) {
  __shared__ float red[kWaves * (kMaxStates + kMaxStates * kMaxQueries)];
  __shared__ float inv[kMaxStates];
  __shared__ float p[kMaxQueries][kMaxStates];
  __shared__ float lg_s[kMaxQueries][kMaxStates];
  const int t = blockIdx.x;
  // 1. per state: the sum of squares -> inv
  float ss[kMaxStates];
#pragma unroll
  for (int i = 0; i < kMaxStates; ++i) ss[i] = 0.0f;
  for (int h = threadIdx.x; h < H; h += kThreads) {
#pragma unroll
    for (int i = 0; i < kMaxStates; ++i) {
      if (i < n_states) {
        const float x = st.s[i][static_cast<int64_t>(t) * H + h];
        ss[i] += x * x;
      }
    }
  }
#pragma unroll
  for (int i = 0; i < kMaxStates; ++i) {
    if (i < n_states) {
      const float v = WaveSum(ss[i]);
      if ((threadIdx.x & 31) == 0) red[i * kWaves + (threadIdx.x >> 5)] = v;
    }
  }
  __syncthreads();
  if (threadIdx.x < n_states) {
    float a = 0.0f;
#pragma unroll
    for (int w = 0; w < kWaves; ++w) a += red[threadIdx.x * kWaves + w];
    inv[threadIdx.x] = 1.0f / sqrtf(a / static_cast<float>(H) + eps);
  }
  __syncthreads();
  // 2. per (query, state): q . rms(S_i)
  float d[kMaxQueries][kMaxStates];
#pragma unroll
  for (int qi = 0; qi < kMaxQueries; ++qi)
#pragma unroll
    for (int i = 0; i < kMaxStates; ++i) d[qi][i] = 0.0f;
  for (int h = threadIdx.x; h < H; h += kThreads) {
#pragma unroll
    for (int i = 0; i < kMaxStates; ++i) {
      if (i >= n_states) continue;
      const float r = st.s[i][static_cast<int64_t>(t) * H + h] * inv[i];
#pragma unroll
      for (int qi = 0; qi < kMaxQueries; ++qi) {
        if (qi < nq) d[qi][i] += q[static_cast<int64_t>(qi) * H + h] * r;
      }
    }
  }
  float* red2 = red + kWaves * kMaxStates;
#pragma unroll
  for (int qi = 0; qi < kMaxQueries; ++qi) {
    if (qi < nq) {
#pragma unroll
      for (int i = 0; i < kMaxStates; ++i) {
        if (i < n_states) {
          const float v = WaveSum(d[qi][i]);
          if ((threadIdx.x & 31) == 0) red2[(qi * kMaxStates + i) * kWaves + (threadIdx.x >> 5)] = v;
        }
      }
    }
  }
  __syncthreads();
  if (threadIdx.x < nq) {
    const int qi = threadIdx.x;
    const float scale = 1.0f / sqrtf(static_cast<float>(H));
    float m = -INFINITY;
    for (int i = 0; i < n_states; ++i) {
      float a = 0.0f;
#pragma unroll
      for (int w = 0; w < kWaves; ++w) a += red2[(qi * kMaxStates + i) * kWaves + w];
      const float v = a * scale;
      lg_s[qi][i] = v;
      m = fmaxf(m, v);
    }
    float l = 0.0f;
    for (int i = 0; i < n_states; ++i) {
      const float e = expf(lg_s[qi][i] - m);
      lg_s[qi][i] = e;
      l += e;
    }
    for (int i = 0; i < n_states; ++i) p[qi][i] = lg_s[qi][i] / l;
  }
  __syncthreads();
  // 3. the mix of the normalized states (statically unrolled over kMaxStates -> zero scratch spill)
  for (int h = threadIdx.x; h < H; h += kThreads) {
    float r[kMaxStates];
#pragma unroll
    for (int i = 0; i < kMaxStates; ++i) r[i] = i < n_states ? st.s[i][static_cast<int64_t>(t) * H + h] * inv[i] : 0.0f;
    for (int qi = 0; qi < nq; ++qi) {
      float a = 0.0f;
#pragma unroll
      for (int i = 0; i < kMaxStates; ++i) {
        if (i < n_states) a += p[qi][i] * r[i];
      }
      out[(static_cast<int64_t>(qi) * T + t) * H + h] = a;
    }
  }
}

// ---- short convolutions -----------------------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void SconvStaticKernel(const void* __restrict__ x, int x_bf16,
                                                              const float* __restrict__ w, float* __restrict__ state,
                                                              const float* residual, float* out, int T, int C) {
  const int c = blockIdx.x * kThreads + threadIdx.x;
  if (c >= C) return;
  const float inv = sqrtf(5.0f);
  float h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
  const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
  for (int t = 0; t < T; ++t) {
    const int64_t i = static_cast<int64_t>(t) * C + c;
    const float xt = LoadF(x, i, x_bf16);
    const float conv = w0 * h0 + w1 * h1 + w2 * h2 + w3 * xt;
    float y = (conv + xt) / inv;
    if (residual != nullptr) y = residual[i] + y;
    out[i] = y;
    h0 = h1;
    h1 = h2;
    h2 = xt;
  }
  state[c] = h0;
  state[C + c] = h1;
  state[2 * C + c] = h2;
}

__global__ __launch_bounds__(kThreads) void SconvDynamicKernel(const void* __restrict__ x, int x_bf16,
                                                               const void* __restrict__ dyn, int dyn_bf16,
                                                               const float* __restrict__ sw, float* __restrict__ state,
                                                               float* __restrict__ out, int T, int C, int heads) {
  const int c = blockIdx.x * kThreads + threadIdx.x;
  if (c >= C) return;
  const int hd = c / (C / heads);
  float h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
  const float s0 = sw[c], s1 = sw[C + c], s2 = sw[2 * C + c], s3 = sw[3 * C + c];
  for (int t = 0; t < T; ++t) {
    const int64_t di = static_cast<int64_t>(t) * heads * 4 + hd * 4;
    const float t0 = s0 + LoadF(dyn, di + 0, dyn_bf16), t1 = s1 + LoadF(dyn, di + 1, dyn_bf16);
    const float t2 = s2 + LoadF(dyn, di + 2, dyn_bf16), t3 = s3 + LoadF(dyn, di + 3, dyn_bf16);
    const int64_t i = static_cast<int64_t>(t) * C + c;
    const float xt = LoadF(x, i, x_bf16);
    const float sum = xt * t0 + h2 * t1 + h1 * t2 + h0 * t3;
    out[i] = (sum + xt) / 3.0f;
    h0 = h1;
    h1 = h2;
    h2 = xt;
  }
  state[c] = h0;
  state[C + c] = h1;
  state[2 * C + c] = h2;
}

// ---- per-head ops: one wave per (t, head), d / 32 elements per lane (d <= 512) ------------------------------------
constexpr int kMaxHeadDim = 512;
struct HeadArgs {
  const void* x;
  int x_bf16;
  int64_t x_row_stride;
  const float* scale;
  float threshold, log_alpha, log_floor, eps;
  int64_t pos0;
  int mode;  // 0 clip, 1 rms
  R4dxInkling2HeadOut o;
  int T, heads, d;
};

__global__ __launch_bounds__(kThreads) void HeadKernel(HeadArgs a) {
  const int wave = (blockIdx.x * kThreads + threadIdx.x) >> 5, lane = threadIdx.x & 31;
  if (wave >= a.T * a.heads) return;
  const int t = wave / a.heads, h = wave % a.heads;
  constexpr int kPer = kMaxHeadDim / 32;
  float v[kPer];
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int i = lane + 32 * j;
    v[j] = 0.0f;
    if (i < a.d) {
      float xv = LoadF(a.x, static_cast<int64_t>(t) * a.x_row_stride + static_cast<int64_t>(h) * a.d + i, a.x_bf16);
      if (a.mode == 0 && a.scale != nullptr) xv = xv * a.scale[static_cast<int64_t>(h) * a.d + i];
      v[j] = xv;
      ss += xv * xv;
    }
  }
  const int64_t pos0 = (a.o.pos_dev != 0) ? static_cast<int64_t>(*reinterpret_cast<const int32_t*>(a.o.pos_dev)) : a.pos0;
  ss = WaveSum(ss);
  float f;
  float tau = 1.0f;
  if (a.mode == 0) {
    f = fminf(1.0f, a.threshold / (sqrtf(ss) + 1e-6f));
    if (a.log_alpha != 0.0f) {
      const float r = static_cast<float>(pos0 + t + 1) / a.log_floor;
      tau = 1.0f + a.log_alpha * logf(fmaxf(r, 1.0f));
    }
  } else {
    f = 1.0f / sqrtf(ss / static_cast<float>(a.d) + a.eps);
  }
#pragma unroll
  for (int j = 0; j < kPer; ++j) {
    const int i = lane + 32 * j;
    if (i >= a.d) continue;
    float y = v[j] * f;
    if (a.mode == 0 && a.log_alpha != 0.0f) y = y * tau;
    if (a.o.kind == 0) {
      const int64_t idx = static_cast<int64_t>(t) * a.o.out_row_stride + static_cast<int64_t>(h) * a.d + i;
      if (a.o.out_bf16)
        reinterpret_cast<uint16_t*>(a.o.out)[idx] = r4dx::core::FloatToBf16(y);
      else
        reinterpret_cast<float*>(a.o.out)[idx] = y;
    } else {
      const int64_t p = pos0 + t;
      const int* bt = reinterpret_cast<const int*>(a.o.block_table);
      const int64_t idx = static_cast<int64_t>(bt[p / a.o.block_size]) * a.o.kv_block_stride +
                          static_cast<int64_t>(h) * a.o.kv_head_stride + (p % a.o.block_size) * 2 * a.d +
                          a.o.col_offset + i;
      reinterpret_cast<uint16_t*>(a.o.out)[idx] = r4dx::core::FloatToBf16(y);
    }
  }
}

// ---- rel logits: one block per row, threads over the extent --------------------------------------------------------
__global__ __launch_bounds__(kThreads) void RelLogitsKernel(const float* __restrict__ r, const float* __restrict__ proj,
                                                            float* __restrict__ rel, int d_rel, int extent) {
  __shared__ float rs[64];
  const int64_t row = blockIdx.x;
  if (threadIdx.x < d_rel) rs[threadIdx.x] = r[row * d_rel + threadIdx.x];
  __syncthreads();
  for (int e = threadIdx.x; e < extent; e += kThreads) {
    float a = 0.0f;
    for (int j = 0; j < d_rel; ++j) a += rs[j] * proj[static_cast<int64_t>(j) * extent + e];
    rel[row * extent + e] = a;
  }
}

__global__ __launch_bounds__(kThreads) void SigmoidGateMulKernel(const uint16_t* o, const uint16_t* __restrict__ g,
                                                                 uint16_t* out, int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (i >= n) return;
  const float ov = r4dx::core::Bf16ToFloat(o[i]);
  const float gv = r4dx::core::Bf16ToFloat(g[i]);
  out[i] = r4dx::core::FloatToBf16(ov * SigmoidF(gv));
}

// ---- message gate --------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void MsgPrepKernel(const float* __restrict__ x, const float* __restrict__ m,
                                                          uint16_t* __restrict__ out, int H, float eps) {
  __shared__ float red[kWaves];
  const int64_t t = blockIdx.x;
  const float* xr = x + t * H;
  const float* mr = m + t * H;
  float sx = 0.0f, sm = 0.0f;
  for (int h = threadIdx.x; h < H; h += kThreads) {
    sx += xr[h] * xr[h];
    sm += mr[h] * mr[h];
  }
  sx = BlockSum(sx, red);
  sm = BlockSum(sm, red);
  const float ix = 1.0f / sqrtf(sx / static_cast<float>(H) + eps);
  const float im = 1.0f / sqrtf(sm / static_cast<float>(H) + eps);
  uint16_t* o = out + t * 2 * H;
  for (int h = threadIdx.x; h < H; h += kThreads) {
    o[h] = r4dx::core::FloatToBf16(xr[h] * ix);
    o[H + h] = r4dx::core::FloatToBf16(mr[h] * im);
  }
}

constexpr int kMaxRank = 256;
// Coalesced, bit-exact MsgApply: each block handles 32 output channels h0..h0+31 across 4 waves (128 threads),
// loading [32][32] tiles of forget/gain coalesced along R into LDS and summing r = 0..R-1 in exact order.
constexpr int kMsgTile = 32;
constexpr int kMsgWaves = 4;
__global__ __launch_bounds__(kMsgWaves * 32) void MsgApplyKernel(
    const float* x, const float* m, const void* __restrict__ g, int g_bf16,
    const float* __restrict__ forget, const float* __restrict__ gain, float* out, int H, int R) {
  __shared__ float gs[kMaxRank];
  __shared__ float sf[kMsgTile][kMsgTile + 1];
  __shared__ float sg[kMsgTile][kMsgTile + 1];
  const int64_t t = blockIdx.y;
  const int tid = threadIdx.x;
  const int wave = tid >> 5;
  const int lane = tid & 31;
  for (int r = tid; r < R; r += kMsgWaves * 32) gs[r] = LoadF(g, t * R + r, g_bf16);
  __syncthreads();
  const int h0 = static_cast<int>(blockIdx.x) * kMsgTile;
  float f = 0.0f, a = 0.0f;
  for (int r0 = 0; r0 < R; r0 += kMsgTile) {
    const int rlen = (R - r0 < kMsgTile) ? (R - r0) : kMsgTile;
    for (int lh = wave; lh < kMsgTile; lh += kMsgWaves) {
      const int h = h0 + lh;
      if (h < H && lane < rlen) {
        const int64_t off = static_cast<int64_t>(h) * R + (r0 + lane);
        sf[lh][lane] = forget[off];
        sg[lh][lane] = gain[off];
      }
    }
    __syncthreads();
    if (wave == 0 && h0 + lane < H) {
      for (int j = 0; j < rlen; ++j) {
        const float gj = gs[r0 + j];
        f += gj * sf[lane][j];
        a += gj * sg[lane][j];
      }
    }
    __syncthreads();
  }
  if (wave == 0 && h0 + lane < H) {
    const int h = h0 + lane;
    f = tanhf(f);
    a = tanhf(a);
    const int64_t i = t * H + h;
    const float xv = x[i], mv = m[i];
    out[i] = (1.0f - f) * xv + a * mv;
  }
}

// ---- router: one block per token; wave w computes experts w, w + 8, ...; thread 0 selects ---------------------------
constexpr int kMaxExperts = 256;
constexpr int kMaxTopK = 16;
__global__ __launch_bounds__(kThreads) void RouterKernel(const uint16_t* __restrict__ x, const uint16_t* __restrict__ gw,
                                                         const float* __restrict__ bias, int32_t* __restrict__ ids,
                                                         float* __restrict__ wout, int H, int E, int K, float scale,
                                                         int64_t x_row_stride, float* __restrict__ logits_out) {
  __shared__ float s[kMaxExperts], sc[kMaxExperts];
  const int64_t t = blockIdx.x;
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const uint16_t* xr = x + t * x_row_stride;
  for (int e = wave; e < E; e += kWaves) {
    const uint16_t* wr = gw + static_cast<int64_t>(e) * H;
    float a = 0.0f;
    for (int h = lane; h < H; h += 32) a += r4dx::core::Bf16ToFloat(xr[h]) * r4dx::core::Bf16ToFloat(wr[h]);
    a = WaveSum(a);
    if (lane == 0) {
      if (logits_out != nullptr) logits_out[t * E + e] = a;
      const float sv = SigmoidF(a);
      s[e] = sv;
      sc[e] = sv + (bias != nullptr ? bias[e] : 0.0f);
    }
  }
  __syncthreads();
  if (threadIdx.x != 0) return;
  int sel[kMaxTopK];
  float sum = 0.0f;
  for (int j = 0; j < K; ++j) {
    int best = -1;
    for (int e = 0; e < E; ++e) {
      bool taken = false;
      for (int q = 0; q < j; ++q) taken = taken || sel[q] == e;
      if (taken) continue;
      if (best < 0 || sc[e] > sc[best]) best = e;  // strict >: ties keep the lower index
    }
    sel[j] = best;
    sum += s[best];
  }
  for (int j = 0; j < K; ++j) {
    ids[t * K + j] = sel[j];
    wout[t * K + j] = s[sel[j]] / (sum + 1e-20f) * scale;
  }
}

// Multi-block decode router (bit-identical to RouterKernel): 4 waves (128 threads) per block, 1 wave per expert ->
// E/4 blocks spread across CUs instead of 1 CU doing all E experts serially.
constexpr int kRouterWavesPerBlock = 4;
__global__ __launch_bounds__(kRouterWavesPerBlock * 32) void RouterDotsKernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ gw, float* __restrict__ logits_out,
    int H, int E, int64_t x_row_stride) {
  const int64_t t = blockIdx.y;
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int e = static_cast<int>(blockIdx.x) * kRouterWavesPerBlock + wave;
  if (e >= E) return;
  const uint16_t* xr = x + t * x_row_stride;
  const uint16_t* wr = gw + static_cast<int64_t>(e) * H;
  float a = 0.0f;
  for (int h = lane; h < H; h += 32) a += r4dx::core::Bf16ToFloat(xr[h]) * r4dx::core::Bf16ToFloat(wr[h]);
  a = WaveSum(a);
  if (lane == 0) logits_out[t * E + e] = a;
}

// Parallel wave32 top-k (bit-identical to RouterKernel: strict >, ties keep lower index; zero local array / scratch spill).
__global__ __launch_bounds__(32) void RouterTopkKernel(
    const float* __restrict__ logits, const float* __restrict__ bias, int32_t* __restrict__ ids,
    float* __restrict__ wout, int E, int K, float scale) {
  const int64_t t = blockIdx.x;
  const int lane = threadIdx.x;
  float sv[8], sc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int e = lane + i * 32;
    if (e < E) {
      sv[i] = SigmoidF(logits[t * E + e]);
      sc[i] = sv[i] + (bias != nullptr ? bias[e] : 0.0f);
    } else {
      sv[i] = 0.0f;
      sc[i] = 0.0f;
    }
  }
  unsigned taken = 0u;
  __shared__ int sel_id[kMaxTopK];
  __shared__ float sel_s[kMaxTopK];
  float sum = 0.0f;
  for (int j = 0; j < K; ++j) {
    float best_v = 0.0f;
    int best_e = E;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int e = lane + i * 32;
      if (e < E && !(taken & (1u << i)) && (best_e >= E || sc[i] > best_v)) {
        best_v = sc[i];
        best_e = e;
      }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor(best_v, o, 32);
      const int oe = __shfl_xor(best_e, o, 32);
      if (oe < E && (best_e >= E || ov > best_v || (ov == best_v && oe < best_e))) {
        best_v = ov;
        best_e = oe;
      }
    }
    const int win_lane = best_e & 31;
    const int win_slot = best_e >> 5;
    float my_s = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      if (i == win_slot) my_s = sv[i];
    }
    if (lane == win_lane) taken |= (1u << win_slot);
    const float win_s = __shfl(my_s, win_lane, 32);
    if (lane == 0) {
      sel_id[j] = best_e;
      sel_s[j] = win_s;
      sum += win_s;
    }
  }
  if (lane < K) {
    const float tot = __shfl(sum, 0, 32);
    ids[t * K + lane] = sel_id[lane];
    wout[t * K + lane] = sel_s[lane] / (tot + 1e-20f) * scale;
  }
}

__global__ __launch_bounds__(kThreads) void SwigluSoftcapKernel(const uint16_t* __restrict__ gu, uint16_t* __restrict__ out,
                                                                int64_t inter, int64_t in_row_stride) {
  const int64_t r = blockIdx.y;
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (i >= inter) return;
  const float g = r4dx::core::Bf16ToFloat(gu[r * in_row_stride + i]);
  const float u = r4dx::core::Bf16ToFloat(gu[r * in_row_stride + inter + i]);
  const float y = tanhf(g / 4.0f) * 4.0f * SigmoidF(g) * (tanhf(u / 25.0f) * 25.0f);
  out[r * inter + i] = r4dx::core::FloatToBf16(y);
}

__global__ __launch_bounds__(kThreads) void LogitSoftcapKernel(const void* x, int x_bf16, int64_t xs, float* out,
                                                               int64_t os, int n, float cap) {
  const int64_t r = blockIdx.y;
  const int i = blockIdx.x * kThreads + threadIdx.x;
  if (i >= n) return;
  const float v = LoadF(x, r * xs + i, x_bf16);
  out[r * os + i] = tanhf(v / cap) * cap;
}

// ---- reference attention with rel bias and sinks (r4dx_gemma_attn_ref_bf16kv + two terms) --------------------------
constexpr int kAttnMaxHeadDim = 512;

__device__ __forceinline__ float BlockReduce(float v, bool is_max, float* red) {
  for (int o = 16; o; o >>= 1) {
    const float w = __shfl_xor(v, o, 32);
    v = is_max ? fmaxf(v, w) : v + w;
  }
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
  __syncthreads();
  float out = red[0];
  for (int i = 1; i < kWaves; ++i) out = is_max ? fmaxf(out, red[i]) : out + red[i];
  __syncthreads();
  return out;
}

__global__ __launch_bounds__(kThreads) void Inkling2AttnRefKernel(const R4dxInkling2AttnRefArgs a) {
  __shared__ float sq[kAttnMaxHeadDim];
  __shared__ float sp[kThreads];
  __shared__ float red[kWaves];

  const int h = blockIdx.x, t = blockIdx.y;
  const int hd = a.head_dim, bs = a.block_size;
  const int kvh = h / (a.q_heads / a.kv_heads);
  const int qpos = a.ctx - a.q_len + t;
  const int* bt = reinterpret_cast<const int*>(a.block_table);
  const uint16_t* kv = reinterpret_cast<const uint16_t*>(a.kv);
  const float* rel = a.rel ? reinterpret_cast<const float*>(a.rel) + (static_cast<int64_t>(t) * a.q_heads + h) * a.rel_extent
                           : nullptr;
  const bool has_sink = a.sink != 0;
  const float sink = has_sink ? reinterpret_cast<const float*>(a.sink)[static_cast<int64_t>(t) * a.q_heads + h] : 0.0f;

  const int klow = r4d_win_klow(qpos, a.window);
  int khigh = qpos;
  if (khigh > a.ctx - 1) khigh = a.ctx - 1;

  const uint16_t* qrow = reinterpret_cast<const uint16_t*>(a.q) + (static_cast<int64_t>(t) * a.q_heads + h) * hd;
  for (int d = threadIdx.x; d < hd; d += kThreads) sq[d] = r4dx::core::Bf16ToFloat(qrow[d]);
  __syncthreads();

  auto key_base = [&](int k) -> int64_t {
    return static_cast<int64_t>(bt[k / bs]) * a.kv_block_stride + static_cast<int64_t>(kvh) * a.kv_head_stride +
           static_cast<int64_t>(k % bs) * 2 * hd;
  };
  auto score = [&](int k) -> float {
    const int64_t base = key_base(k);
    float acc = 0.0f;
    for (int d = 0; d < hd; ++d) acc += sq[d] * r4dx::core::Bf16ToFloat(kv[base + d]);
    acc = acc * a.scale;
    const int dist = qpos - k;
    if (rel != nullptr && dist < a.rel_extent) acc = acc + rel[dist];
    return acc;
  };

  float m = -INFINITY;
  for (int k0 = klow; k0 <= khigh; k0 += kThreads) {
    const int k = k0 + static_cast<int>(threadIdx.x);
    if (k <= khigh) m = fmaxf(m, score(k));
  }
  m = BlockReduce(m, true, red);
  if (has_sink) m = fmaxf(m, sink);

  float l = 0.0f, acc0 = 0.0f, acc1 = 0.0f;
  for (int k0 = klow; k0 <= khigh; k0 += kThreads) {
    const int k = k0 + static_cast<int>(threadIdx.x);
    float p = 0.0f;
    if (k <= khigh) p = expf(score(k) - m);
    sp[threadIdx.x] = p;
    l += p;
    __syncthreads();
    const int n = (khigh - k0 + 1 < kThreads) ? (khigh - k0 + 1) : kThreads;
    const int d0 = threadIdx.x, d1 = threadIdx.x + kThreads;
    for (int j = 0; j < n; ++j) {
      const int64_t vb = key_base(k0 + j) + hd;
      const float pj = sp[j];
      if (d0 < hd) acc0 += pj * r4dx::core::Bf16ToFloat(kv[vb + d0]);
      if (d1 < hd) acc1 += pj * r4dx::core::Bf16ToFloat(kv[vb + d1]);
    }
    __syncthreads();
  }
  l = BlockReduce(l, false, red);
  if (has_sink) l = l + expf(sink - m);

  const float inv = (l > 0.0f) ? (1.0f / l) : 0.0f;
  uint16_t* orow = reinterpret_cast<uint16_t*>(a.out) + (static_cast<int64_t>(t) * a.q_heads + h) * hd;
  if (threadIdx.x < hd) orow[threadIdx.x] = r4dx::core::FloatToBf16(acc0 * inv);
  if (threadIdx.x + kThreads < hd) orow[threadIdx.x + kThreads] = r4dx::core::FloatToBf16(acc1 * inv);
}

// ---- fused h128 attention with on-the-fly relative bias, sink, window, paged bf16 KV (S3) --------------------------
constexpr int kFusedHd = 128;
constexpr int kFusedMaxGqa = 8;
constexpr int kFusedMaxDrel = 32;
constexpr int kFusedWaves = 4;  // 128 threads = 4 wave32s

__device__ __forceinline__ void Unpack4Bf16(uint2 v, float& f0, float& f1, float& f2, float& f3) {
  f0 = r4dx::core::Bf16ToFloat(static_cast<uint16_t>(v.x & 0xFFFFu));
  f1 = r4dx::core::Bf16ToFloat(static_cast<uint16_t>(v.x >> 16));
  f2 = r4dx::core::Bf16ToFloat(static_cast<uint16_t>(v.y & 0xFFFFu));
  f3 = r4dx::core::Bf16ToFloat(static_cast<uint16_t>(v.y >> 16));
}

__device__ __forceinline__ uint2 Pack4Bf16(float f0, float f1, float f2, float f3) {
  const uint32_t lo = static_cast<uint32_t>(r4dx::core::FloatToBf16(f0)) |
                      (static_cast<uint32_t>(r4dx::core::FloatToBf16(f1)) << 16);
  const uint32_t hi = static_cast<uint32_t>(r4dx::core::FloatToBf16(f2)) |
                      (static_cast<uint32_t>(r4dx::core::FloatToBf16(f3)) << 16);
  return make_uint2(lo, hi);
}

// Decode kernel (q_len == 1): grid = (kv_heads, n_splits), block = (128) = 4 waves.
// Exact two-pass softmax per 256-key tile in LDS: K is read once (Pass 1 writes scores[GQA][256] and finds exact
// tile max), V is read once (Pass 2 accumulates exp(s - m) * V with zero per-key rescaling).
// If n_splits == 1, merges waves in LDS, applies sink, and writes bf16 out directly.
// If n_splits > 1, merges waves in LDS and writes (m, l, o[128]) per (q_head, split) to workspace.
constexpr int kDecTile = 128;
template <int GQA>
__global__ __launch_bounds__(kFusedWaves * 32) void Inkling2AttnDecodeSplitKernel(
    const R4dxInkling2AttnFusedArgs a, int n_splits) {
  __shared__ float s_tile[GQA][kDecTile];
  __shared__ float sm[kFusedWaves];
  __shared__ float sl[kFusedWaves];
  __shared__ float so[kFusedWaves][kFusedHd];

  const int kvh = static_cast<int>(blockIdx.x);
  const int split = static_cast<int>(blockIdx.y);
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int qh0 = kvh * GQA;
  const int qpos = (a.pos_dev != 0) ? (*reinterpret_cast<const int32_t*>(a.pos_dev)) : (a.ctx - 1);
  const int klow = r4d_win_klow(qpos, a.window);
  const int klen = qpos - klow + 1;
  int active_splits = (klen > kDecTile && n_splits > 1) ? ((klen + kDecTile - 1) / kDecTile) : 1;
  if (active_splits > n_splits) active_splits = n_splits;
  if (split >= active_splits) {
    if (n_splits > 1 && wave == 0 && lane == 0) {
      float* ws_ptr = reinterpret_cast<float*>(a.workspace);
#pragma unroll
      for (int g = 0; g < GQA; ++g) {
        float* dst = ws_ptr + (static_cast<int64_t>(qh0 + g) * n_splits + split) * (kFusedHd + 2);
        dst[0] = -INFINITY;
        dst[1] = 0.0f;
      }
    }
    return;
  }
  const int per_split = (klen + active_splits - 1) / active_splits;
  const int s_begin = klow + split * per_split;
  const int s_end = (s_begin + per_split < qpos + 1) ? (s_begin + per_split) : (qpos + 1);

  const uint16_t* q_ptr = reinterpret_cast<const uint16_t*>(a.q);
  const uint16_t* kv_ptr = reinterpret_cast<const uint16_t*>(a.kv);
  const int* bt = reinterpret_cast<const int*>(a.block_table);
  const float* r_ptr = a.r ? reinterpret_cast<const float*>(a.r) : nullptr;
  const float* proj_ptr = a.rel_proj ? reinterpret_cast<const float*>(a.rel_proj) : nullptr;
  const bool has_rel = (r_ptr != nullptr && proj_ptr != nullptr && a.d_rel > 0 && a.rel_extent > 0);

  float q0[GQA], q1[GQA], q2[GQA], q3[GQA];
  float o0[GQA], o1[GQA], o2[GQA], o3[GQA];
  float mv[GQA], lv[GQA];
  float rv[GQA];
#pragma unroll
  for (int g = 0; g < GQA; ++g) {
    const uint2 qv = *reinterpret_cast<const uint2*>(q_ptr + static_cast<int64_t>(qh0 + g) * kFusedHd + lane * 4);
    Unpack4Bf16(qv, q0[g], q1[g], q2[g], q3[g]);
    o0[g] = o1[g] = o2[g] = o3[g] = 0.0f;
    mv[g] = -INFINITY;
    lv[g] = 0.0f;
    rv[g] = (has_rel && lane < a.d_rel) ? r_ptr[static_cast<int64_t>(qh0 + g) * a.d_rel + lane] : 0.0f;
  }

  const int bs = a.block_size;
  for (int tb = s_begin; tb < s_end; tb += kDecTile) {
    const int tlen = (s_end - tb < kDecTile) ? (s_end - tb) : kDecTile;
    // Pass 1: read K once, compute scores and tile max per wave.
    float tm[GQA];
#pragma unroll
    for (int g = 0; g < GQA; ++g) tm[g] = -INFINITY;
    for (int j = wave; j < tlen; j += kFusedWaves) {
      const int k = tb + j;
      const int64_t base = static_cast<int64_t>(bt[k / bs]) * a.kv_block_stride +
                           static_cast<int64_t>(kvh) * a.kv_head_stride +
                           static_cast<int64_t>(k % bs) * (2 * kFusedHd);
      const uint2 kv_k = *reinterpret_cast<const uint2*>(kv_ptr + base + lane * 4);
      float k0, k1, k2, k3;
      Unpack4Bf16(kv_k, k0, k1, k2, k3);
      const int dist = qpos - k;
      const bool use_rel = has_rel && dist < a.rel_extent;
      const float pj = (use_rel && lane < a.d_rel) ? proj_ptr[static_cast<int64_t>(lane) * a.rel_extent + dist] : 0.0f;
#pragma unroll
      for (int g = 0; g < GQA; ++g) {
        float dot = q0[g] * k0 + q1[g] * k1 + q2[g] * k2 + q3[g] * k3;
        dot = WaveSum(dot) * a.scale;
        if (use_rel) {
          float rb = rv[g] * pj;
          rb = WaveSum(rb);
          dot += rb;
        }
        if (lane == 0) s_tile[g][j] = dot;
        tm[g] = fmaxf(tm[g], dot);
      }
    }
    // Reduce tile max across the 4 waves via sm[].
    float tile_m[GQA];
#pragma unroll
    for (int g = 0; g < GQA; ++g) {
      if (lane == 0) sm[wave] = tm[g];
      __syncthreads();
      float mg = sm[0];
#pragma unroll
      for (int w = 1; w < kFusedWaves; ++w) mg = fmaxf(mg, sm[w]);
      tile_m[g] = mg;
      __syncthreads();
    }
    // Pass 2: read V once, accumulate exp(s - tile_m) * V with zero per-key rescaling.
    float tl[GQA], to0[GQA], to1[GQA], to2[GQA], to3[GQA];
#pragma unroll
    for (int g = 0; g < GQA; ++g) {
      tl[g] = 0.0f;
      to0[g] = to1[g] = to2[g] = to3[g] = 0.0f;
    }
    for (int j = wave; j < tlen; j += kFusedWaves) {
      const int k = tb + j;
      const int64_t base = static_cast<int64_t>(bt[k / bs]) * a.kv_block_stride +
                           static_cast<int64_t>(kvh) * a.kv_head_stride +
                           static_cast<int64_t>(k % bs) * (2 * kFusedHd);
      const uint2 kv_v = *reinterpret_cast<const uint2*>(kv_ptr + base + kFusedHd + lane * 4);
      float v0, v1, v2, v3;
      Unpack4Bf16(kv_v, v0, v1, v2, v3);
#pragma unroll
      for (int g = 0; g < GQA; ++g) {
        const float p = expf(s_tile[g][j] - tile_m[g]);
        tl[g] += p;
        to0[g] += p * v0;
        to1[g] += p * v1;
        to2[g] += p * v2;
        to3[g] += p * v3;
      }
    }
    // Fold this tile into (mv, lv, o) (only happens >1 time if s_end - s_begin > 256).
#pragma unroll
    for (int g = 0; g < GQA; ++g) {
      if (mv[g] == -INFINITY) {
        mv[g] = tile_m[g];
        lv[g] = tl[g];
        o0[g] = to0[g];
        o1[g] = to1[g];
        o2[g] = to2[g];
        o3[g] = to3[g];
      } else {
        const float mn = fmaxf(mv[g], tile_m[g]);
        const float a0 = expf(mv[g] - mn);
        const float a1 = expf(tile_m[g] - mn);
        lv[g] = lv[g] * a0 + tl[g] * a1;
        o0[g] = o0[g] * a0 + to0[g] * a1;
        o1[g] = o1[g] * a0 + to1[g] * a1;
        o2[g] = o2[g] * a0 + to2[g] * a1;
        o3[g] = o3[g] * a0 + to3[g] * a1;
        mv[g] = mn;
      }
    }
    __syncthreads();
  }

  const float* sink_ptr = a.sink ? reinterpret_cast<const float*>(a.sink) : nullptr;
  uint16_t* out_ptr = reinterpret_cast<uint16_t*>(a.out);
  float* ws_ptr = reinterpret_cast<float*>(a.workspace);

#pragma unroll
  for (int g = 0; g < GQA; ++g) {
    if (lane == 0) {
      sm[wave] = mv[g];
      sl[wave] = lv[g];
    }
    so[wave][lane * 4 + 0] = o0[g];
    so[wave][lane * 4 + 1] = o1[g];
    so[wave][lane * 4 + 2] = o2[g];
    so[wave][lane * 4 + 3] = o3[g];
    __syncthreads();

    if (wave == 0) {
      float m_all = -INFINITY;
#pragma unroll
      for (int w = 0; w < kFusedWaves; ++w) m_all = fmaxf(m_all, sm[w]);
      float l_all = 0.0f;
      float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
#pragma unroll
      for (int w = 0; w < kFusedWaves; ++w) {
        if (sl[w] > 0.0f) {
          const float scale_w = expf(sm[w] - m_all);
          l_all += sl[w] * scale_w;
          r0 += so[w][lane * 4 + 0] * scale_w;
          r1 += so[w][lane * 4 + 1] * scale_w;
          r2 += so[w][lane * 4 + 2] * scale_w;
          r3 += so[w][lane * 4 + 3] * scale_w;
        }
      }
      const int h = qh0 + g;
      if (n_splits == 1) {
        if (sink_ptr != nullptr) {
          const float sv = sink_ptr[h];
          const float mn = fmaxf(m_all, sv);
          const float alpha = (m_all == -INFINITY) ? 0.0f : expf(m_all - mn);
          l_all = l_all * alpha + expf(sv - mn);
          r0 *= alpha;
          r1 *= alpha;
          r2 *= alpha;
          r3 *= alpha;
        }
        const float inv = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;
        *reinterpret_cast<uint2*>(out_ptr + static_cast<int64_t>(h) * kFusedHd + lane * 4) =
            Pack4Bf16(r0 * inv, r1 * inv, r2 * inv, r3 * inv);
      } else {
        // Workspace layout per (h, split): [m, l, o[0..127]] (130 floats).
        float* dst = ws_ptr + (static_cast<int64_t>(h) * n_splits + split) * (kFusedHd + 2);
        if (lane == 0) {
          dst[0] = m_all;
          dst[1] = l_all;
        }
        dst[2 + lane * 4 + 0] = r0;
        dst[2 + lane * 4 + 1] = r1;
        dst[2 + lane * 4 + 2] = r2;
        dst[2 + lane * 4 + 3] = r3;
      }
    }
    __syncthreads();
  }
}

// Combine kernel for split-K decode: grid = (q_heads), block = (32).
__global__ __launch_bounds__(32) void Inkling2AttnDecodeCombineKernel(
    const float* __restrict__ ws, const float* __restrict__ sink, uint16_t* __restrict__ out, int n_splits) {
  const int h = static_cast<int>(blockIdx.x);
  const int lane = threadIdx.x;
  const float* base = ws + static_cast<int64_t>(h) * n_splits * (kFusedHd + 2);
  float m_all = (sink != nullptr) ? sink[h] : -INFINITY;
  for (int s = 0; s < n_splits; ++s) {
    const float ms = base[static_cast<int64_t>(s) * (kFusedHd + 2) + 0];
    m_all = fmaxf(m_all, ms);
  }
  float l_all = (sink != nullptr) ? expf(sink[h] - m_all) : 0.0f;
  float r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
  for (int s = 0; s < n_splits; ++s) {
    const float* src = base + static_cast<int64_t>(s) * (kFusedHd + 2);
    const float ls = src[1];
    if (ls > 0.0f) {
      const float sc = expf(src[0] - m_all);
      l_all += ls * sc;
      r0 += src[2 + lane * 4 + 0] * sc;
      r1 += src[2 + lane * 4 + 1] * sc;
      r2 += src[2 + lane * 4 + 2] * sc;
      r3 += src[2 + lane * 4 + 3] * sc;
    }
  }
  const float inv = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;
  *reinterpret_cast<uint2*>(out + static_cast<int64_t>(h) * kFusedHd + lane * 4) =
      Pack4Bf16(r0 * inv, r1 * inv, r2 * inv, r3 * inv);
}

// Prefill kernel (q_len > 1): grid = (kv_heads, ceil(q_len / 4)), block = (128) = 4 waves.
// Wave w handles query row t = t0 + w across all GQA query heads; all 4 waves cooperatively load BK = 16
// K/V rows into LDS (8 KiB total) so every K/V load is reused across 4 * GQA = 12 query vectors.
constexpr int kPrefillBK = 16;
template <int GQA>
__global__ __launch_bounds__(kFusedWaves * 32) void Inkling2AttnPrefillKernel(const R4dxInkling2AttnFusedArgs a) {
  __shared__ uint2 sk[kPrefillBK][32];  // 16 x 128 bf16 = 4 KiB
  __shared__ uint2 sv[kPrefillBK][32];  // 16 x 128 bf16 = 4 KiB

  const int kvh = static_cast<int>(blockIdx.x);
  const int t0 = static_cast<int>(blockIdx.y) * kFusedWaves;
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int t = t0 + wave;
  const bool valid_t = t < a.q_len;
  const int qh0 = kvh * GQA;
  const int qpos = a.ctx - a.q_len + t;
  const int klow = valid_t ? r4d_win_klow(qpos, a.window) : a.ctx;
  const int khigh = valid_t ? (qpos < a.ctx - 1 ? qpos : a.ctx - 1) : -1;

  // Tile K range covering all valid queries in this block [t0 .. min(t0+3, q_len-1)].
  const int t_last = (t0 + kFusedWaves - 1 < a.q_len) ? (t0 + kFusedWaves - 1) : (a.q_len - 1);
  const int k_tile_begin = r4d_win_klow(a.ctx - a.q_len + t0, a.window);
  const int k_tile_end_raw = a.ctx - a.q_len + t_last;
  const int k_tile_end = (k_tile_end_raw < a.ctx - 1 ? k_tile_end_raw : a.ctx - 1) + 1;

  const uint16_t* q_ptr = reinterpret_cast<const uint16_t*>(a.q);
  const uint16_t* kv_ptr = reinterpret_cast<const uint16_t*>(a.kv);
  const int* bt = reinterpret_cast<const int*>(a.block_table);
  const float* r_ptr = a.r ? reinterpret_cast<const float*>(a.r) : nullptr;
  const float* proj_ptr = a.rel_proj ? reinterpret_cast<const float*>(a.rel_proj) : nullptr;
  const bool has_rel = (r_ptr != nullptr && proj_ptr != nullptr && a.d_rel > 0 && a.rel_extent > 0);

  float q0[GQA], q1[GQA], q2[GQA], q3[GQA];
  float o0[GQA], o1[GQA], o2[GQA], o3[GQA];
  float mv[GQA], lv[GQA];
  float rv[GQA];
#pragma unroll
  for (int g = 0; g < GQA; ++g) {
    if (valid_t) {
      const int64_t qoff = (static_cast<int64_t>(t) * a.q_heads + qh0 + g) * kFusedHd + lane * 4;
      Unpack4Bf16(*reinterpret_cast<const uint2*>(q_ptr + qoff), q0[g], q1[g], q2[g], q3[g]);
      rv[g] = (has_rel && lane < a.d_rel)
                  ? r_ptr[(static_cast<int64_t>(t) * a.q_heads + qh0 + g) * a.d_rel + lane]
                  : 0.0f;
    } else {
      q0[g] = q1[g] = q2[g] = q3[g] = 0.0f;
      rv[g] = 0.0f;
    }
    o0[g] = o1[g] = o2[g] = o3[g] = 0.0f;
    mv[g] = -INFINITY;
    lv[g] = 0.0f;
  }

  const int bs = a.block_size;
  for (int kb = k_tile_begin; kb < k_tile_end; kb += kPrefillBK) {
    const int n_tile = (k_tile_end - kb < kPrefillBK) ? (k_tile_end - kb) : kPrefillBK;
    // Cooperative load of n_tile K and V rows into LDS (4 waves x 32 lanes = 128 threads; 4 rows per wave).
    for (int j = wave; j < n_tile; j += kFusedWaves) {
      const int k = kb + j;
      const int64_t base = static_cast<int64_t>(bt[k / bs]) * a.kv_block_stride +
                           static_cast<int64_t>(kvh) * a.kv_head_stride +
                           static_cast<int64_t>(k % bs) * (2 * kFusedHd);
      sk[j][lane] = *reinterpret_cast<const uint2*>(kv_ptr + base + lane * 4);
      sv[j][lane] = *reinterpret_cast<const uint2*>(kv_ptr + base + kFusedHd + lane * 4);
    }
    __syncthreads();

    if (valid_t) {
      for (int j = 0; j < n_tile; ++j) {
        const int k = kb + j;
        if (k < klow || k > khigh) continue;
        float k0, k1, k2, k3, v0, v1, v2, v3;
        Unpack4Bf16(sk[j][lane], k0, k1, k2, k3);
        Unpack4Bf16(sv[j][lane], v0, v1, v2, v3);

        const int dist = qpos - k;
        const bool use_rel = has_rel && dist < a.rel_extent;
        const float pj = (use_rel && lane < a.d_rel) ? proj_ptr[static_cast<int64_t>(lane) * a.rel_extent + dist] : 0.0f;

#pragma unroll
        for (int g = 0; g < GQA; ++g) {
          float dot = q0[g] * k0 + q1[g] * k1 + q2[g] * k2 + q3[g] * k3;
          dot = WaveSum(dot) * a.scale;
          if (use_rel) {
            float rb = rv[g] * pj;
            rb = WaveSum(rb);
            dot += rb;
          }
          const float mn = fmaxf(mv[g], dot);
          const float alpha = (mv[g] == -INFINITY) ? 0.0f : expf(mv[g] - mn);
          const float p = expf(dot - mn);
          lv[g] = lv[g] * alpha + p;
          o0[g] = o0[g] * alpha + p * v0;
          o1[g] = o1[g] * alpha + p * v1;
          o2[g] = o2[g] * alpha + p * v2;
          o3[g] = o3[g] * alpha + p * v3;
          mv[g] = mn;
        }
      }
    }
    __syncthreads();
  }

  if (!valid_t) return;
  const float* sink_ptr = a.sink ? reinterpret_cast<const float*>(a.sink) : nullptr;
  uint16_t* out_ptr = reinterpret_cast<uint16_t*>(a.out);
#pragma unroll
  for (int g = 0; g < GQA; ++g) {
    const int h = qh0 + g;
    if (sink_ptr != nullptr) {
      const float sv_s = sink_ptr[static_cast<int64_t>(t) * a.q_heads + h];
      const float mn = fmaxf(mv[g], sv_s);
      const float alpha = (mv[g] == -INFINITY) ? 0.0f : expf(mv[g] - mn);
      lv[g] = lv[g] * alpha + expf(sv_s - mn);
      o0[g] *= alpha;
      o1[g] *= alpha;
      o2[g] *= alpha;
      o3[g] *= alpha;
    }
    const float inv = (lv[g] > 0.0f) ? (1.0f / lv[g]) : 0.0f;
    const int64_t ooff = (static_cast<int64_t>(t) * a.q_heads + h) * kFusedHd + lane * 4;
    *reinterpret_cast<uint2*>(out_ptr + ooff) = Pack4Bf16(o0[g] * inv, o1[g] * inv, o2[g] * inv, o3[g] * inv);
  }
}

// ---- small dense GEMM: one wave per output element (1 wave/block at small M for CU spread, bit-identical) ----------
__global__ __launch_bounds__(32) void DenseBf16DecodeKernel(const uint16_t* __restrict__ x, int64_t xs,
                                                            const uint16_t* __restrict__ w, void* __restrict__ out,
                                                            int out_bf16, int64_t os, int N, int K) {
  const int64_t m = blockIdx.y;
  const int n = static_cast<int>(blockIdx.x);
  const int lane = threadIdx.x;
  if (n >= N) return;
  const uint16_t* xr = x + m * xs;
  const uint16_t* wr = w + static_cast<int64_t>(n) * K;
  float a = 0.0f;
  for (int k = lane; k < K; k += 32) a += r4dx::core::Bf16ToFloat(xr[k]) * r4dx::core::Bf16ToFloat(wr[k]);
  a = WaveSum(a);
  if (lane != 0) return;
  if (out_bf16) {
    static_cast<uint16_t*>(out)[m * os + n] = r4dx::core::FloatToBf16(a);
  } else {
    static_cast<float*>(out)[m * os + n] = a;
  }
}

__global__ __launch_bounds__(kThreads) void DenseBf16Kernel(const uint16_t* __restrict__ x, int64_t xs,
                                                            const uint16_t* __restrict__ w, void* __restrict__ out,
                                                            int out_bf16, int64_t os, int N, int K) {
  const int64_t m = blockIdx.y;
  const int n = blockIdx.x * kWaves + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  if (n >= N) return;
  const uint16_t* xr = x + m * xs;
  const uint16_t* wr = w + static_cast<int64_t>(n) * K;
  float a = 0.0f;
  for (int k = lane; k < K; k += 32) a += r4dx::core::Bf16ToFloat(xr[k]) * r4dx::core::Bf16ToFloat(wr[k]);
  a = WaveSum(a);
  if (lane != 0) return;
  if (out_bf16) {
    static_cast<uint16_t*>(out)[m * os + n] = r4dx::core::FloatToBf16(a);
  } else {
    static_cast<float*>(out)[m * os + n] = a;
  }
}

void RunHead(const char* fn, HeadArgs a, int64_t stream) {
  if (a.T <= 0 || a.heads <= 0) return;
  if (a.x == nullptr) Fail(fn, "null x");
  if (a.d < 4 || a.d > kMaxHeadDim || a.d % 4 != 0) Fail(fn, "d must be a multiple of 4 in 4..512");
  if (a.x_row_stride < static_cast<int64_t>(a.heads) * a.d) Fail(fn, "x_row_stride < heads * d");
  if (a.o.out == 0) Fail(fn, "null output");
  if (a.o.kind == 0) {
    if (a.o.out_row_stride < static_cast<int64_t>(a.heads) * a.d) Fail(fn, "out_row_stride < heads * d");
  } else if (a.o.kind == 1) {
    if (a.o.block_table == 0 || a.o.block_size <= 0 || (a.o.pos_dev == 0 && a.pos0 < 0))
      Fail(fn, "paged output needs a block table, block_size > 0, pos0 >= 0");
    if (a.o.col_offset != 0 && a.o.col_offset != a.d) Fail(fn, "col_offset must be 0 (K) or d (V)");
  } else {
    Fail(fn, "out kind must be 0 (dense) or 1 (paged KV)");
  }
  const int64_t waves = static_cast<int64_t>(a.T) * a.heads;
  const unsigned blocks = static_cast<unsigned>((waves + kWaves - 1) / kWaves);
  hipLaunchKernelGGL(HeadKernel, dim3(blocks), dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), a);
  CheckLaunch(fn);
}

}  // namespace

extern "C" {

void r4dx_inkling2_attnres_f32(const int64_t* states, int n_states, int64_t queries, int nq, int64_t out, int T,
                               int H, float eps, int64_t stream) {
  const char* fn = "r4dx_inkling2_attnres_f32";
  if (T <= 0) return;
  if (states == nullptr || n_states < 1 || n_states > kMaxStates) Fail(fn, "n_states must be 1..6 with a pointer array");
  if (nq < 1 || nq > kMaxQueries) Fail(fn, "nq must be 1..4");
  if (H < 1 || H > 8192) Fail(fn, "H must be 1..8192");
  if (queries == 0 || out == 0) Fail(fn, "null queries / out");
  AttnResPtrs p{};
  for (int i = 0; i < n_states; ++i) {
    if (states[i] == 0) Fail(fn, "null state pointer");
    p.s[i] = reinterpret_cast<const float*>(states[i]);
  }
  hipLaunchKernelGGL(AttnResKernel, dim3(static_cast<unsigned>(T)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), p, n_states, reinterpret_cast<const float*>(queries), nq,
                     reinterpret_cast<float*>(out), T, H, eps);
  CheckLaunch(fn);
}

void r4dx_inkling2_sconv_static_f32(int64_t x, int x_bf16, int64_t weight, int64_t state, int64_t residual,
                                    int64_t out, int T, int C, int64_t stream) {
  const char* fn = "r4dx_inkling2_sconv_static_f32";
  if (T <= 0 || C <= 0) return;
  if (x == 0 || weight == 0 || state == 0 || out == 0) Fail(fn, "null pointer");
  if (x == out) Fail(fn, "out must not alias x");
  hipLaunchKernelGGL(SconvStaticKernel, dim3(static_cast<unsigned>((C + kThreads - 1) / kThreads)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const void*>(x), x_bf16,
                     reinterpret_cast<const float*>(weight), reinterpret_cast<float*>(state),
                     reinterpret_cast<const float*>(residual), reinterpret_cast<float*>(out), T, C);
  CheckLaunch(fn);
}

void r4dx_inkling2_sconv_dynamic_f32(int64_t x, int x_bf16, int64_t dyn, int dyn_bf16, int64_t static_w,
                                     int64_t state, int64_t out, int T, int C, int heads, int64_t stream) {
  const char* fn = "r4dx_inkling2_sconv_dynamic_f32";
  if (T <= 0 || C <= 0) return;
  if (x == 0 || dyn == 0 || static_w == 0 || state == 0 || out == 0) Fail(fn, "null pointer");
  if (heads < 1 || C % heads != 0) Fail(fn, "C must be a multiple of heads");
  if (x == out) Fail(fn, "out must not alias x");
  hipLaunchKernelGGL(SconvDynamicKernel, dim3(static_cast<unsigned>((C + kThreads - 1) / kThreads)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const void*>(x), x_bf16,
                     reinterpret_cast<const void*>(dyn), dyn_bf16, reinterpret_cast<const float*>(static_w),
                     reinterpret_cast<float*>(state), reinterpret_cast<float*>(out), T, C, heads);
  CheckLaunch(fn);
}

void r4dx_inkling2_head_clip(int64_t x, int x_bf16, int64_t x_row_stride, int64_t scale, float threshold,
                             int64_t pos0, float log_alpha, float log_floor, const R4dxInkling2HeadOut* out, int T,
                             int heads, int d, int64_t stream) {
  const char* fn = "r4dx_inkling2_head_clip";
  if (out == nullptr) Fail(fn, "null output descriptor");
  if (log_alpha != 0.0f && !(log_floor > 0.0f)) Fail(fn, "log_floor must be > 0");
  HeadArgs a{};
  a.x = reinterpret_cast<const void*>(x);
  a.x_bf16 = x_bf16;
  a.x_row_stride = x_row_stride;
  a.scale = reinterpret_cast<const float*>(scale);
  a.threshold = threshold;
  a.log_alpha = log_alpha;
  a.log_floor = log_floor;
  a.pos0 = pos0;
  a.mode = 0;
  a.o = *out;
  a.T = T;
  a.heads = heads;
  a.d = d;
  RunHead(fn, a, stream);
}

void r4dx_inkling2_head_rms(int64_t x, int x_bf16, int64_t x_row_stride, float eps, int64_t pos0,
                            const R4dxInkling2HeadOut* out, int T, int heads, int d, int64_t stream) {
  const char* fn = "r4dx_inkling2_head_rms";
  if (out == nullptr) Fail(fn, "null output descriptor");
  HeadArgs a{};
  a.x = reinterpret_cast<const void*>(x);
  a.x_bf16 = x_bf16;
  a.x_row_stride = x_row_stride;
  a.eps = eps;
  a.pos0 = pos0;
  a.mode = 1;
  a.o = *out;
  a.T = T;
  a.heads = heads;
  a.d = d;
  RunHead(fn, a, stream);
}

void r4dx_inkling2_rel_logits_f32(int64_t r, int64_t proj, int64_t rel, int rows, int d_rel, int extent,
                                  int64_t stream) {
  const char* fn = "r4dx_inkling2_rel_logits_f32";
  if (rows <= 0) return;
  if (r == 0 || proj == 0 || rel == 0) Fail(fn, "null pointer");
  if (d_rel < 1 || d_rel > 64 || extent < 1) Fail(fn, "d_rel must be 1..64 and extent >= 1");
  hipLaunchKernelGGL(RelLogitsKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const float*>(r),
                     reinterpret_cast<const float*>(proj), reinterpret_cast<float*>(rel), d_rel, extent);
  CheckLaunch(fn);
}

void r4dx_inkling2_sigmoid_gate_mul_bf16(int64_t o, int64_t g, int64_t out, int64_t n, int64_t stream) {
  const char* fn = "r4dx_inkling2_sigmoid_gate_mul_bf16";
  if (n <= 0) return;
  if (o == 0 || g == 0 || out == 0) Fail(fn, "null pointer");
  hipLaunchKernelGGL(SigmoidGateMulKernel, dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads)), dim3(kThreads),
                     0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(o),
                     reinterpret_cast<const uint16_t*>(g), reinterpret_cast<uint16_t*>(out), n);
  CheckLaunch(fn);
}

void r4dx_inkling2_msg_gate_prep_bf16(int64_t x, int64_t m, int64_t out, int T, int H, float eps, int64_t stream) {
  const char* fn = "r4dx_inkling2_msg_gate_prep_bf16";
  if (T <= 0) return;
  if (x == 0 || m == 0 || out == 0 || H < 1) Fail(fn, "null pointer or H < 1");
  hipLaunchKernelGGL(MsgPrepKernel, dim3(static_cast<unsigned>(T)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const float*>(x),
                     reinterpret_cast<const float*>(m), reinterpret_cast<uint16_t*>(out), H, eps);
  CheckLaunch(fn);
}

void r4dx_inkling2_msg_gate_apply_f32(int64_t x, int64_t m, int64_t g, int g_bf16, int64_t forget, int64_t gain,
                                      int64_t out, int T, int H, int R, int64_t stream) {
  const char* fn = "r4dx_inkling2_msg_gate_apply_f32";
  if (T <= 0) return;
  if (x == 0 || m == 0 || g == 0 || forget == 0 || gain == 0 || out == 0) Fail(fn, "null pointer");
  if (H < 1 || R < 1 || R > kMaxRank) Fail(fn, "H >= 1 and R in 1..256");
  hipLaunchKernelGGL(MsgApplyKernel, dim3(static_cast<unsigned>((H + kMsgTile - 1) / kMsgTile), static_cast<unsigned>(T)),
                     dim3(kMsgWaves * 32), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const float*>(x),
                     reinterpret_cast<const float*>(m), reinterpret_cast<const void*>(g), g_bf16,
                     reinterpret_cast<const float*>(forget), reinterpret_cast<const float*>(gain),
                     reinterpret_cast<float*>(out), H, R);
  CheckLaunch(fn);
}

void r4dx_inkling2_router_f32(int64_t x, int64_t gate_w, int64_t bias, int64_t ids, int64_t w, int T, int H, int E,
                              int top_k, float route_scale, int64_t x_row_stride, int64_t logits_out, int64_t stream) {
  const char* fn = "r4dx_inkling2_router_f32";
  if (T <= 0) return;
  if (x == 0 || gate_w == 0 || ids == 0 || w == 0) Fail(fn, "null pointer");
  if (E < 1 || E > kMaxExperts) Fail(fn, "E must be 1..256");
  if (top_k < 1 || top_k > kMaxTopK || top_k > E) Fail(fn, "top_k must be 1..min(16, E)");
  if (H < 1 || H > 8192 || x_row_stride < H) Fail(fn, "H must be 1..8192 and x_row_stride >= H");
  if (T < 16 && logits_out != 0 && E >= 16) {
    const unsigned blocks_e = static_cast<unsigned>((E + kRouterWavesPerBlock - 1) / kRouterWavesPerBlock);
    hipLaunchKernelGGL(RouterDotsKernel, dim3(blocks_e, static_cast<unsigned>(T)), dim3(kRouterWavesPerBlock * 32), 0,
                       reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(x),
                       reinterpret_cast<const uint16_t*>(gate_w), reinterpret_cast<float*>(logits_out), H, E,
                       x_row_stride);
    CheckLaunch(fn);
    hipLaunchKernelGGL(RouterTopkKernel, dim3(static_cast<unsigned>(T)), dim3(32), 0,
                       reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const float*>(logits_out),
                       reinterpret_cast<const float*>(bias), reinterpret_cast<int32_t*>(ids),
                       reinterpret_cast<float*>(w), E, top_k, route_scale);
    CheckLaunch(fn);
    return;
  }
  hipLaunchKernelGGL(RouterKernel, dim3(static_cast<unsigned>(T)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(x),
                     reinterpret_cast<const uint16_t*>(gate_w), reinterpret_cast<const float*>(bias),
                     reinterpret_cast<int32_t*>(ids), reinterpret_cast<float*>(w), H, E, top_k, route_scale,
                     x_row_stride, reinterpret_cast<float*>(logits_out));
  CheckLaunch(fn);
}

void r4dx_inkling2_swiglu_softcap_bf16(int64_t gate_up, int64_t out, int64_t rows, int64_t intermediate,
                                       int64_t in_row_stride, int64_t stream) {
  const char* fn = "r4dx_inkling2_swiglu_softcap_bf16";
  if (rows <= 0 || intermediate <= 0) return;
  if (gate_up == 0 || out == 0) Fail(fn, "null pointer");
  if (in_row_stride < 2 * intermediate) Fail(fn, "in_row_stride < 2 * intermediate");
  if (rows > 65535) Fail(fn, "rows > 65535");
  hipLaunchKernelGGL(SwigluSoftcapKernel,
                     dim3(static_cast<unsigned>((intermediate + kThreads - 1) / kThreads), static_cast<unsigned>(rows)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(gate_up),
                     reinterpret_cast<uint16_t*>(out), intermediate, in_row_stride);
  CheckLaunch(fn);
}

void r4dx_inkling2_logit_softcap_f32(int64_t x, int x_bf16, int64_t x_row_stride, int64_t out, int64_t out_row_stride,
                                     int rows, int n, float cap, int64_t stream) {
  const char* fn = "r4dx_inkling2_logit_softcap_f32";
  if (rows <= 0 || n <= 0) return;
  if (x == 0 || out == 0) Fail(fn, "null pointer");
  if (x_row_stride < n || out_row_stride < n || !(cap > 0.0f)) Fail(fn, "row strides >= n and cap > 0");
  if (rows > 65535) Fail(fn, "rows > 65535");
  hipLaunchKernelGGL(LogitSoftcapKernel, dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads), static_cast<unsigned>(rows)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const void*>(x), x_bf16,
                     x_row_stride, reinterpret_cast<float*>(out), out_row_stride, n, cap);
  CheckLaunch(fn);
}

void r4dx_inkling2_dense_bf16(int64_t x, int64_t x_row_stride, int64_t w, int64_t out, int out_bf16,
                              int64_t out_row_stride, int M, int N, int K, int64_t stream) {
  const char* fn = "r4dx_inkling2_dense_bf16";
  if (M <= 0 || N <= 0) return;
  if (x == 0 || w == 0 || out == 0) Fail(fn, "null pointer");
  if (K < 1 || x_row_stride < K || out_row_stride < N) Fail(fn, "K >= 1, x_row_stride >= K, out_row_stride >= N");
  if (M > 65535) Fail(fn, "M > 65535");
  if (M <= 4 && N >= 16) {
    hipLaunchKernelGGL(DenseBf16DecodeKernel, dim3(static_cast<unsigned>(N), static_cast<unsigned>(M)), dim3(32), 0,
                       reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(x), x_row_stride,
                       reinterpret_cast<const uint16_t*>(w), reinterpret_cast<void*>(out), out_bf16, out_row_stride, N, K);
    CheckLaunch(fn);
    return;
  }
  hipLaunchKernelGGL(DenseBf16Kernel, dim3(static_cast<unsigned>((N + kWaves - 1) / kWaves), static_cast<unsigned>(M)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(x),
                     x_row_stride, reinterpret_cast<const uint16_t*>(w), reinterpret_cast<void*>(out), out_bf16,
                     out_row_stride, N, K);
  CheckLaunch(fn);
}

int r4dx_inkling2_attn_ref_bf16kv(const R4dxInkling2AttnRefArgs* a, int64_t stream) {
  if (a == nullptr) return -1;
  if (a->q_len <= 0 || a->q_heads <= 0 || a->kv_heads <= 0 || a->block_size <= 0 || a->ctx <= 0 ||
      a->q_len > a->ctx || a->window < 0) {
    return -1;
  }
  if (a->head_dim < 8 || a->head_dim > kAttnMaxHeadDim || (a->head_dim % 8) != 0) return -2;
  if (a->q_heads % a->kv_heads != 0) return -3;
  if (a->q == 0 || a->kv == 0 || a->block_table == 0 || a->out == 0) return -4;
  if (a->rel != 0 && a->rel_extent <= 0) return -5;
  hipLaunchKernelGGL(Inkling2AttnRefKernel, dim3(static_cast<unsigned>(a->q_heads), static_cast<unsigned>(a->q_len)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), *a);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) return -100 - static_cast<int>(e);
  r4dx_kernel_launch_counter_add(1);
  return 0;
}

int r4dx_inkling2_attn_fused_bf16kv(const R4dxInkling2AttnFusedArgs* a, int64_t stream) {
  if (a == nullptr) return -1;
  if (a->q == 0 || a->kv == 0 || a->block_table == 0 || a->out == 0) return -2;
  if (a->q_len <= 0 || a->q_heads <= 0 || a->kv_heads <= 0 || a->head_dim != kFusedHd || a->block_size <= 0) return -3;
  if (a->q_heads % a->kv_heads != 0) return -4;
  const int gqa = a->q_heads / a->kv_heads;
  if (gqa < 1 || gqa > kFusedMaxGqa) return -4;
  if (a->pos_dev != 0) {
    if (a->q_len != 1 || a->max_splits < 1 || (a->max_splits > 1 && a->workspace == 0)) return -7;
  } else {
    if (a->ctx < a->q_len) return -5;
  }
  if (a->window < 0) return -5;
  if (a->r != 0 && (a->rel_proj == 0 || a->d_rel <= 0 || a->d_rel > kFusedMaxDrel || a->rel_extent <= 0)) return -6;
  hipStream_t st = reinterpret_cast<hipStream_t>(stream);
  if (a->q_len == 1) {
    int n_splits = 1;
    if (a->pos_dev != 0) {
      if (a->max_splits > 1) {
        n_splits = a->max_splits;
        if (a->window > 0) {
          const int win_splits = (a->window + kDecTile - 1) / kDecTile;
          if (win_splits < n_splits) n_splits = win_splits;
        }
        if (n_splits < 2) n_splits = 2;
      }
    } else {
      const int qpos = a->ctx - 1;
      const int klow = r4d_win_klow(qpos, a->window);
      const int klen = qpos - klow + 1;
      if (klen > kDecTile && a->workspace != 0 && a->max_splits > 1) {
        n_splits = (klen + kDecTile - 1) / kDecTile;
        if (n_splits > a->max_splits) n_splits = a->max_splits;
      }
    }
    const dim3 grid(static_cast<unsigned>(a->kv_heads), static_cast<unsigned>(n_splits));
    const dim3 block(kFusedWaves * 32);
#define INK2_DEC_CASE(G) \
  case G: hipLaunchKernelGGL((Inkling2AttnDecodeSplitKernel<G>), grid, block, 0, st, *a, n_splits); break
    switch (gqa) {
      INK2_DEC_CASE(1);
      INK2_DEC_CASE(2);
      INK2_DEC_CASE(3);
      INK2_DEC_CASE(4);
      INK2_DEC_CASE(6);
      INK2_DEC_CASE(8);
      default: return -4;
    }
#undef INK2_DEC_CASE
    hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -100 - static_cast<int>(e);
    r4dx_kernel_launch_counter_add(1);
    if (n_splits > 1) {
      hipLaunchKernelGGL(Inkling2AttnDecodeCombineKernel, dim3(static_cast<unsigned>(a->q_heads)), dim3(32), 0, st,
                         reinterpret_cast<const float*>(a->workspace), reinterpret_cast<const float*>(a->sink),
                         reinterpret_cast<uint16_t*>(a->out), n_splits);
      e = hipGetLastError();
      if (e != hipSuccess) return -100 - static_cast<int>(e);
      r4dx_kernel_launch_counter_add(1);
    }
    return 0;
  }
  const dim3 grid(static_cast<unsigned>(a->kv_heads), static_cast<unsigned>((a->q_len + kFusedWaves - 1) / kFusedWaves));
  const dim3 block(kFusedWaves * 32);
#define INK2_PF_CASE(G) \
  case G: hipLaunchKernelGGL((Inkling2AttnPrefillKernel<G>), grid, block, 0, st, *a); break
  switch (gqa) {
    INK2_PF_CASE(1);
    INK2_PF_CASE(2);
    INK2_PF_CASE(3);
    INK2_PF_CASE(4);
    INK2_PF_CASE(6);
    INK2_PF_CASE(8);
    default: return -4;
  }
#undef INK2_PF_CASE
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) return -100 - static_cast<int>(e);
  r4dx_kernel_launch_counter_add(1);
  return 0;
}

}  // extern "C"
