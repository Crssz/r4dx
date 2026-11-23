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
#include "r4d_gdn_wmma.h"
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

__device__ __forceinline__ uint8_t FloatToFp8E4m3Fn(float x) {
  const uint32_t bits = __float_as_uint(x);
  const uint32_t sign = (bits >> 24) & 0x80u;
  const float ax = fabsf(x);
  if (!(ax > 0.0f)) return static_cast<uint8_t>(sign);
  if (ax >= 448.0f) return static_cast<uint8_t>(sign | 0x7Eu);
  const uint32_t abits = __float_as_uint(ax);
  const int exp = static_cast<int>((abits >> 23) & 0xFFu) - 127;
  const uint32_t mant = abits & 0x7FFFFFu;
  if (exp < -6) {
    const int m = __float2int_rn(ax * 512.0f);
    if (m <= 0) return static_cast<uint8_t>(sign);
    if (m >= 8) return static_cast<uint8_t>(sign | 0x08u);
    return static_cast<uint8_t>(sign | static_cast<uint32_t>(m));
  }
  const int e4 = exp + 7;
  const uint32_t m3 = (mant + 0x7FFFFu + ((mant >> 20) & 1u)) >> 20;
  const uint32_t code = (static_cast<uint32_t>(e4) << 3) + m3;
  return static_cast<uint8_t>(sign | (code > 0x7Eu ? 0x7Eu : code));
}

__device__ __forceinline__ float Fp8E4m3FnToFloat(uint8_t c) {
  const uint32_t sign = (static_cast<uint32_t>(c) & 0x80u) << 24;
  const uint32_t e = (c >> 3) & 0xFu;
  const uint32_t m = c & 0x7u;
  if (e == 0u) {
    const float v = static_cast<float>(m) * (1.0f / 512.0f);
    return sign ? -v : v;
  }
  if (e == 15u && m == 7u) return 0.0f;
  return __uint_as_float(sign | ((e + 120u) << 23) | (m << 20));
}

__device__ __forceinline__ void Unpack4Fp8(uint32_t p, float& f0, float& f1, float& f2, float& f3) {
  f0 = Fp8E4m3FnToFloat(static_cast<uint8_t>(p & 0xFFu));
  f1 = Fp8E4m3FnToFloat(static_cast<uint8_t>((p >> 8) & 0xFFu));
  f2 = Fp8E4m3FnToFloat(static_cast<uint8_t>((p >> 16) & 0xFFu));
  f3 = Fp8E4m3FnToFloat(static_cast<uint8_t>((p >> 24) & 0xFFu));
}

__device__ __forceinline__ float SigmoidF(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---- attnres ---------------------------------------------------------------------------------------------------
constexpr int kMaxStates = 6;
constexpr int kMaxQueries = 4;
struct AttnResPtrs {
  const float* s[kMaxStates];
  const float* inv[kMaxStates];
};

// Fast pre-normalized AttnRes (+ optional fused RMSNorm -> bf16 when nq == 1 and norm_w != nullptr):
// When `st.inv[i] != nullptr`, `inv[i] = st.inv[i][t]` is already pre-computed once when state S_i was produced,
// eliminating the sum-of-squares reduction across all states and the barrier before softmax.
__global__ __launch_bounds__(kThreads) void AttnResPrenormKernel(
    AttnResPtrs st, int n_states, int all_prenorm, const float* __restrict__ q, int nq, float* __restrict__ out,
    const uint16_t* __restrict__ norm_w, uint16_t* __restrict__ out_bf16, int T, int H, float eps) {
  __shared__ float red[kWaves * (kMaxStates + kMaxStates * kMaxQueries)];
  __shared__ float inv[kMaxStates];
  __shared__ float p[kMaxQueries][kMaxStates];
  __shared__ float lg_s[kMaxQueries][kMaxStates];
  const int t = blockIdx.x;
  const int64_t row_off = static_cast<int64_t>(t) * H;
  float ss[kMaxStates];
  float d[kMaxQueries][kMaxStates];
#pragma unroll
  for (int i = 0; i < kMaxStates; ++i) {
    ss[i] = 0.0f;
#pragma unroll
    for (int qi = 0; qi < kMaxQueries; ++qi) d[qi][i] = 0.0f;
  }
  const bool vec4 = (H & 3) == 0;
  if (vec4) {
    const int H4 = H >> 2;
    const int64_t row4 = row_off >> 2;
    if (all_prenorm) {
      for (int h4 = threadIdx.x; h4 < H4; h4 += kThreads) {
        float4 qv[kMaxQueries];
#pragma unroll
        for (int qi = 0; qi < kMaxQueries; ++qi) {
          if (qi < nq) qv[qi] = reinterpret_cast<const float4*>(q)[static_cast<int64_t>(qi) * H4 + h4];
        }
#pragma unroll
        for (int i = 0; i < kMaxStates; ++i) {
          if (i < n_states) {
            const float4 x = reinterpret_cast<const float4*>(st.s[i])[row4 + h4];
#pragma unroll
            for (int qi = 0; qi < kMaxQueries; ++qi) {
              if (qi < nq) {
                d[qi][i] += (qv[qi].x * x.x + qv[qi].y * x.y) + (qv[qi].z * x.z + qv[qi].w * x.w);
              }
            }
          }
        }
      }
    } else {
      for (int h4 = threadIdx.x; h4 < H4; h4 += kThreads) {
        float4 qv[kMaxQueries];
#pragma unroll
        for (int qi = 0; qi < kMaxQueries; ++qi) {
          if (qi < nq) qv[qi] = reinterpret_cast<const float4*>(q)[static_cast<int64_t>(qi) * H4 + h4];
        }
#pragma unroll
        for (int i = 0; i < kMaxStates; ++i) {
          if (i < n_states) {
            const float4 x = reinterpret_cast<const float4*>(st.s[i])[row4 + h4];
            if (st.inv[i] == nullptr) ss[i] += (x.x * x.x + x.y * x.y) + (x.z * x.z + x.w * x.w);
#pragma unroll
            for (int qi = 0; qi < kMaxQueries; ++qi) {
              if (qi < nq) {
                d[qi][i] += (qv[qi].x * x.x + qv[qi].y * x.y) + (qv[qi].z * x.z + qv[qi].w * x.w);
              }
            }
          }
        }
      }
    }
  } else {
    for (int h = threadIdx.x; h < H; h += kThreads) {
      float qv[kMaxQueries];
#pragma unroll
      for (int qi = 0; qi < kMaxQueries; ++qi) {
        if (qi < nq) qv[qi] = q[static_cast<int64_t>(qi) * H + h];
      }
#pragma unroll
      for (int i = 0; i < kMaxStates; ++i) {
        if (i < n_states) {
          const float x = st.s[i][row_off + h];
          if (st.inv[i] == nullptr) ss[i] += x * x;
#pragma unroll
          for (int qi = 0; qi < kMaxQueries; ++qi) {
            if (qi < nq) d[qi][i] += qv[qi] * x;
          }
        }
      }
    }
  }
  if (!all_prenorm) {
#pragma unroll
    for (int i = 0; i < kMaxStates; ++i) {
      if (i < n_states && st.inv[i] == nullptr) {
        const float v = WaveSum(ss[i]);
        if ((threadIdx.x & 31) == 0) red[i * kWaves + (threadIdx.x >> 5)] = v;
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
  if (all_prenorm && threadIdx.x < n_states) {
    inv[threadIdx.x] = st.inv[threadIdx.x][t];
  }
  __syncthreads();
  if (!all_prenorm) {
    if (threadIdx.x < n_states) {
      if (st.inv[threadIdx.x] != nullptr) {
        inv[threadIdx.x] = st.inv[threadIdx.x][t];
      } else {
        float a = 0.0f;
#pragma unroll
        for (int w = 0; w < kWaves; ++w) a += red[threadIdx.x * kWaves + w];
        inv[threadIdx.x] = rsqrtf(a / static_cast<float>(H) + eps);
      }
    }
    __syncthreads();
  }
  if (threadIdx.x < nq) {
    const int qi = threadIdx.x;
    const float scale = rsqrtf(static_cast<float>(H));
    float m = -INFINITY;
    for (int i = 0; i < n_states; ++i) {
      float a = 0.0f;
#pragma unroll
      for (int w = 0; w < kWaves; ++w) a += red2[(qi * kMaxStates + i) * kWaves + w];
      const float v = (a * inv[i]) * scale;
      lg_s[qi][i] = v;
      m = fmaxf(m, v);
    }
    float l = 0.0f;
    for (int i = 0; i < n_states; ++i) {
      const float e = expf(lg_s[qi][i] - m);
      lg_s[qi][i] = e;
      l += e;
    }
    const float inv_l = 1.0f / l;
    for (int i = 0; i < n_states; ++i) p[qi][i] = (lg_s[qi][i] * inv_l) * inv[i];
  }
  __syncthreads();
  // Pass 2: mix of states (p[qi][i] already folded inv[i]), optionally fused with plain RMSNorm -> bf16 (when nq == 1).
  if (norm_w != nullptr && out_bf16 != nullptr && nq == 1) {
    float ss_out = 0.0f;
    if (vec4) {
      const int H4 = H >> 2;
      const int64_t row4 = row_off >> 2;
      for (int h4 = threadIdx.x; h4 < H4; h4 += kThreads) {
        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int i = 0; i < kMaxStates; ++i) {
          if (i < n_states) {
            const float4 sv = reinterpret_cast<const float4*>(st.s[i])[row4 + h4];
            const float w = p[0][i];
            a.x += w * sv.x;
            a.y += w * sv.y;
            a.z += w * sv.z;
            a.w += w * sv.w;
          }
        }
        if (out != nullptr) reinterpret_cast<float4*>(out)[row4 + h4] = a;
        ss_out += (a.x * a.x + a.y * a.y) + (a.z * a.z + a.w * a.w);
      }
      ss_out = BlockSum(ss_out, red);
      const float rstd = rsqrtf(ss_out / static_cast<float>(H) + eps);
      uint16_t* dst = out_bf16 + row_off;
      for (int h4 = threadIdx.x; h4 < H4; h4 += kThreads) {
        float4 a;
        if (out != nullptr) {
          a = reinterpret_cast<const float4*>(out)[row4 + h4];
        } else {
          a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
          for (int i = 0; i < kMaxStates; ++i) {
            if (i < n_states) {
              const float4 sv = reinterpret_cast<const float4*>(st.s[i])[row4 + h4];
              const float w = p[0][i];
              a.x += w * sv.x;
              a.y += w * sv.y;
              a.z += w * sv.z;
              a.w += w * sv.w;
            }
          }
        }
        const int h = h4 << 2;
        const uint2 w01 = *reinterpret_cast<const uint2*>(norm_w + h);
        const float w0 = r4dx::core::Bf16ToFloat(static_cast<uint16_t>(w01.x & 0xFFFFu));
        const float w1 = r4dx::core::Bf16ToFloat(static_cast<uint16_t>(w01.x >> 16));
        const float w2 = r4dx::core::Bf16ToFloat(static_cast<uint16_t>(w01.y & 0xFFFFu));
        const float w3 = r4dx::core::Bf16ToFloat(static_cast<uint16_t>(w01.y >> 16));
        const uint32_t o01 = static_cast<uint32_t>(r4dx::core::FloatToBf16(a.x * rstd * w0)) |
                             (static_cast<uint32_t>(r4dx::core::FloatToBf16(a.y * rstd * w1)) << 16);
        const uint32_t o23 = static_cast<uint32_t>(r4dx::core::FloatToBf16(a.z * rstd * w2)) |
                             (static_cast<uint32_t>(r4dx::core::FloatToBf16(a.w * rstd * w3)) << 16);
        *reinterpret_cast<uint2*>(dst + h) = make_uint2(o01, o23);
      }
    } else {
      for (int h = threadIdx.x; h < H; h += kThreads) {
        float a = 0.0f;
#pragma unroll
        for (int i = 0; i < kMaxStates; ++i) {
          if (i < n_states) a += p[0][i] * st.s[i][row_off + h];
        }
        if (out != nullptr) out[row_off + h] = a;
        ss_out += a * a;
      }
      ss_out = BlockSum(ss_out, red);
      const float rstd = rsqrtf(ss_out / static_cast<float>(H) + eps);
      uint16_t* dst = out_bf16 + row_off;
      for (int h = threadIdx.x; h < H; h += kThreads) {
        float a = 0.0f;
        if (out != nullptr) {
          a = out[row_off + h];
        } else {
#pragma unroll
          for (int i = 0; i < kMaxStates; ++i) {
            if (i < n_states) a += p[0][i] * st.s[i][row_off + h];
          }
        }
        dst[h] = r4dx::core::FloatToBf16(a * rstd * r4dx::core::Bf16ToFloat(norm_w[h]));
      }
    }
    return;
  }

  if (vec4) {
    const int H4 = H >> 2;
    const int64_t row4 = row_off >> 2;
    for (int h4 = threadIdx.x; h4 < H4; h4 += kThreads) {
      float4 s_v[kMaxStates];
#pragma unroll
      for (int i = 0; i < kMaxStates; ++i) {
        if (i < n_states) s_v[i] = reinterpret_cast<const float4*>(st.s[i])[row4 + h4];
      }
      for (int qi = 0; qi < nq; ++qi) {
        float4 a = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int i = 0; i < kMaxStates; ++i) {
          if (i < n_states) {
            const float w = p[qi][i];
            a.x += w * s_v[i].x;
            a.y += w * s_v[i].y;
            a.z += w * s_v[i].z;
            a.w += w * s_v[i].w;
          }
        }
        reinterpret_cast<float4*>(out)[(static_cast<int64_t>(qi) * T + t) * H4 + h4] = a;
      }
    }
  } else {
    for (int h = threadIdx.x; h < H; h += kThreads) {
      float s_v[kMaxStates];
#pragma unroll
      for (int i = 0; i < kMaxStates; ++i) s_v[i] = i < n_states ? st.s[i][row_off + h] : 0.0f;
      for (int qi = 0; qi < nq; ++qi) {
        float a = 0.0f;
#pragma unroll
        for (int i = 0; i < kMaxStates; ++i) {
          if (i < n_states) a += p[qi][i] * s_v[i];
        }
        out[(static_cast<int64_t>(qi) * T + t) * H + h] = a;
      }
    }
  }
}

__global__ __launch_bounds__(kThreads) void RowInvRmsKernel(const float* __restrict__ x, float* __restrict__ inv_out,
                                                            int H, float eps) {
  __shared__ float red[kWaves];
  const int t = blockIdx.x;
  const float* xr = x + static_cast<int64_t>(t) * H;
  float ss = 0.0f;
  if ((H & 3) == 0) {
    const int H4 = H >> 2;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    for (int h4 = threadIdx.x; h4 < H4; h4 += kThreads) {
      const float4 v = xr4[h4];
      ss += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
    }
  } else {
    for (int h = threadIdx.x; h < H; h += kThreads) {
      const float v = xr[h];
      ss += v * v;
    }
  }
  ss = WaveSum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = red[0];
#pragma unroll
    for (int i = 1; i < kWaves; ++i) total += red[i];
    inv_out[t] = rsqrtf(total / static_cast<float>(H) + eps);
  }
}

// Fused decode (T == 1) static short convolution + residual add + row inverse RMS in a single 256-thread block.
// Vectorized with float4 when C % 4 == 0 (cuts global load/store instructions by 4x).
__global__ __launch_bounds__(kThreads) void SconvStaticDecodeNormKernel(
    const void* __restrict__ x, int x_bf16, const float* __restrict__ w, float* __restrict__ state,
    const float* residual, float* out, float* hist, float* __restrict__ inv_out, int C, float eps) {
  __shared__ float red[kWaves];
  const float inv5 = sqrtf(5.0f);
  float ss = 0.0f;
  if ((C & 3) == 0 && (reinterpret_cast<uintptr_t>(state) & 15) == 0 && (reinterpret_cast<uintptr_t>(out) & 15) == 0 &&
      (residual == nullptr || (reinterpret_cast<uintptr_t>(residual) & 15) == 0) &&
      (hist == nullptr || (reinterpret_cast<uintptr_t>(hist) & 15) == 0)) {
    const int C4 = C >> 2;
    float4* st0 = reinterpret_cast<float4*>(state);
    float4* st1 = reinterpret_cast<float4*>(state + C);
    float4* st2 = reinterpret_cast<float4*>(state + 2 * C);
    const float4* res4 = residual != nullptr ? reinterpret_cast<const float4*>(residual) : nullptr;
    float4* out4 = reinterpret_cast<float4*>(out);
    float4* hist4 = hist != nullptr ? reinterpret_cast<float4*>(hist) : nullptr;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    for (int c4 = threadIdx.x; c4 < C4; c4 += kThreads) {
      const float4 h0 = st0[c4], h1 = st1[c4], h2 = st2[c4];
      float4 xt;
      if (x_bf16) {
        const uint2 xb = reinterpret_cast<const uint2*>(x)[c4];
        xt.x = r4dx::core::Bf16ToFloat(static_cast<uint16_t>(xb.x & 0xFFFFu));
        xt.y = r4dx::core::Bf16ToFloat(static_cast<uint16_t>(xb.x >> 16));
        xt.z = r4dx::core::Bf16ToFloat(static_cast<uint16_t>(xb.y & 0xFFFFu));
        xt.w = r4dx::core::Bf16ToFloat(static_cast<uint16_t>(xb.y >> 16));
      } else {
        xt = reinterpret_cast<const float4*>(x)[c4];
      }
      if (hist4 != nullptr) hist4[c4] = xt;
      const int c0 = c4 << 2;
      const float4 wv0 = w4[c0 + 0], wv1 = w4[c0 + 1], wv2 = w4[c0 + 2], wv3 = w4[c0 + 3];
      float4 y;
      y.x = ((wv0.x * h0.x + wv0.y * h1.x + wv0.z * h2.x + wv0.w * xt.x) + xt.x) / inv5;
      y.y = ((wv1.x * h0.y + wv1.y * h1.y + wv1.z * h2.y + wv1.w * xt.y) + xt.y) / inv5;
      y.z = ((wv2.x * h0.z + wv2.y * h1.z + wv2.z * h2.z + wv2.w * xt.z) + xt.z) / inv5;
      y.w = ((wv3.x * h0.w + wv3.y * h1.w + wv3.z * h2.w + wv3.w * xt.w) + xt.w) / inv5;
      if (res4 != nullptr) {
        const float4 rv = res4[c4];
        y.x = rv.x + y.x;
        y.y = rv.y + y.y;
        y.z = rv.z + y.z;
        y.w = rv.w + y.w;
      }
      out4[c4] = y;
      st0[c4] = h1;
      st1[c4] = h2;
      st2[c4] = xt;
      ss += (y.x * y.x + y.y * y.y) + (y.z * y.z + y.w * y.w);
    }
  } else {
    for (int c = threadIdx.x; c < C; c += kThreads) {
      const float h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
      const float4 wv = reinterpret_cast<const float4*>(w)[c];
      const float xt = LoadF(x, c, x_bf16);
      if (hist != nullptr) hist[c] = xt;
      const float conv = wv.x * h0 + wv.y * h1 + wv.z * h2 + wv.w * xt;
      float y = (conv + xt) / inv5;
      if (residual != nullptr) y = residual[c] + y;
      out[c] = y;
      state[c] = h1;
      state[C + c] = h2;
      state[2 * C + c] = xt;
      ss += y * y;
    }
  }
  ss = WaveSum(ss);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = ss;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = red[0];
#pragma unroll
    for (int i = 1; i < kWaves; ++i) total += red[i];
    inv_out[0] = rsqrtf(total / static_cast<float>(C) + eps);
  }
}

// Paired K and V static short convolution in a single launch: blockIdx.y == 0 -> K, blockIdx.y == 1 -> V.
__global__ __launch_bounds__(kThreads) void SconvStaticPairKernel(
    const void* __restrict__ xk, const void* __restrict__ xv, int x_bf16, const float* __restrict__ wk,
    const float* __restrict__ wv, float* __restrict__ state_k, float* __restrict__ state_v, float* __restrict__ out_k,
    float* __restrict__ out_v, float* __restrict__ hist_k, float* __restrict__ hist_v, int T, int C) {
  const int c = blockIdx.x * kThreads + threadIdx.x;
  if (c >= C) return;
  const bool is_v = (blockIdx.y != 0);
  const void* x = is_v ? xv : xk;
  const float* w = is_v ? wv : wk;
  float* state = is_v ? state_v : state_k;
  float* out = is_v ? out_v : out_k;
  float* hist = is_v ? hist_v : hist_k;
  const float inv = sqrtf(5.0f);
  float h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
  const float4 wc = reinterpret_cast<const float4*>(w)[c];
  for (int t = 0; t < T; ++t) {
    const int64_t i = static_cast<int64_t>(t) * C + c;
    const float xt = LoadF(x, i, x_bf16);
    if (hist != nullptr) hist[i] = xt;
    const float conv = wc.x * h0 + wc.y * h1 + wc.z * h2 + wc.w * xt;
    out[i] = (conv + xt) / inv;
    h0 = h1;
    h1 = h2;
    h2 = xt;
  }
  state[c] = h0;
  state[C + c] = h1;
  state[2 * C + c] = h2;
}

// Fused sliding KV static short-conv + K head_clip + V head_rms + paged bf16 KV cache write.
// Grid: (kv_heads, 2) where blockIdx.y == 0 is K (clip, col_offset 0) and blockIdx.y == 1 is V (rms, col_offset 128).
// Block: 32 threads (1 wave per head, 4 channels/lane: i = lane + 32*j), exact same WaveSum reduction order as HeadKernel.
__global__ __launch_bounds__(32) void SconvKvPagedBf16Kernel(
    const uint16_t* __restrict__ xk, const uint16_t* __restrict__ xv, const float* __restrict__ wk,
    const float* __restrict__ wv, float* __restrict__ state_k, float* __restrict__ state_v,
    float* __restrict__ hist_k, float* __restrict__ hist_v, float qk_clip, float eps, int64_t pos0_host,
    const int32_t* __restrict__ pos_dev, uint16_t* __restrict__ kv_out, const int32_t* __restrict__ block_table,
    int block_size, int64_t kv_block_stride, int64_t kv_head_stride, int T, int kv_heads) {
  constexpr int kHd = 128;
  const int h = static_cast<int>(blockIdx.x);
  const bool is_v = (blockIdx.y != 0);
  const int lane = threadIdx.x;
  const int C = kv_heads * kHd;
  const uint16_t* x = is_v ? xv : xk;
  const float4* w4 = reinterpret_cast<const float4*>(is_v ? wv : wk);
  float* state = is_v ? state_v : state_k;
  float* hist = is_v ? hist_v : hist_k;
  const int64_t pos0 = (pos_dev != nullptr) ? static_cast<int64_t>(*pos_dev) : pos0_host;
  const int col_offset = is_v ? kHd : 0;
  const float inv_s5 = sqrtf(5.0f);

  float h0[4], h1[4], h2[4];
  float4 wc[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = h * kHd + lane + 32 * j;
    h0[j] = state[c];
    h1[j] = state[C + c];
    h2[j] = state[2 * C + c];
    wc[j] = w4[c];
  }
  for (int t = 0; t < T; ++t) {
    float v[4];
    float ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int c = h * kHd + lane + 32 * j;
      const int64_t idx = static_cast<int64_t>(t) * C + c;
      const float xt = r4dx::core::Bf16ToFloat(x[idx]);
      if (hist != nullptr) hist[idx] = xt;
      const float conv = wc[j].x * h0[j] + wc[j].y * h1[j] + wc[j].z * h2[j] + wc[j].w * xt;
      const float y = (conv + xt) / inv_s5;
      v[j] = y;
      ss += y * y;
      h0[j] = h1[j];
      h1[j] = h2[j];
      h2[j] = xt;
    }
    ss = WaveSum(ss);
    const float f = is_v ? (1.0f / sqrtf(ss / static_cast<float>(kHd) + eps))
                         : fminf(1.0f, qk_clip / (sqrtf(ss) + 1e-6f));
    const int64_t p = pos0 + t;
    const int64_t base = static_cast<int64_t>(block_table[p / block_size]) * kv_block_stride +
                         static_cast<int64_t>(h) * kv_head_stride + (p % block_size) * (2 * kHd) + col_offset;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      kv_out[base + lane + 32 * j] = r4dx::core::FloatToBf16(v[j] * f);
    }
  }
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int c = h * kHd + lane + 32 * j;
    state[c] = h0[j];
    state[C + c] = h1[j];
    state[2 * C + c] = h2[j];
  }
}

// Fused Q (bf16 out) + R (fp32 out) HeadClip: blockIdx.y == 0 -> Q, blockIdx.y == 1 -> R.
__global__ __launch_bounds__(kThreads) void HeadClipQRKernel(
    const uint16_t* __restrict__ q_in, int64_t q_in_stride, const float* __restrict__ q_scale, float q_thresh,
    uint16_t* __restrict__ q_out, int q_d, const uint16_t* __restrict__ r_in, int64_t r_in_stride,
    const float* __restrict__ r_scale, float r_thresh, float* __restrict__ r_out, int r_d, int64_t pos0_host,
    const int32_t* __restrict__ pos_dev, float log_alpha, float log_floor, int T, int heads) {
  const int wave = (blockIdx.x * kThreads + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (wave >= T * heads) return;
  const int t = wave / heads;
  const int h = wave % heads;
  const bool is_r = (blockIdx.y != 0);
  const uint16_t* xin = is_r ? r_in : q_in;
  const int64_t stride = is_r ? r_in_stride : q_in_stride;
  const float* sc = is_r ? r_scale : q_scale;
  const float thresh = is_r ? r_thresh : q_thresh;
  const int d = is_r ? r_d : q_d;

  float v[4];
  float ss = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = lane + 32 * j;
    v[j] = 0.0f;
    if (i < d) {
      float xv = r4dx::core::Bf16ToFloat(xin[static_cast<int64_t>(t) * stride + static_cast<int64_t>(h) * d + i]);
      if (sc != nullptr) xv = xv * sc[static_cast<int64_t>(h) * d + i];
      v[j] = xv;
      ss += xv * xv;
    }
  }
  const int64_t pos0 = (pos_dev != nullptr) ? static_cast<int64_t>(*pos_dev) : pos0_host;
  ss = WaveSum(ss);
  const float f = fminf(1.0f, thresh / (sqrtf(ss) + 1e-6f));
  float tau = 1.0f;
  if (log_alpha != 0.0f) {
    const float r = static_cast<float>(pos0 + t + 1) / log_floor;
    tau = 1.0f + log_alpha * logf(fmaxf(r, 1.0f));
  }
  const int64_t out_base = (static_cast<int64_t>(t) * heads + h) * d;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int i = lane + 32 * j;
    if (i >= d) continue;
    float y = v[j] * f;
    if (log_alpha != 0.0f) y = y * tau;
    if (is_r) {
      r_out[out_base + i] = y;
    } else {
      q_out[out_base + i] = r4dx::core::FloatToBf16(y);
    }
  }
}

// ---- short convolutions -----------------------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void SconvStaticKernel(const void* __restrict__ x, int x_bf16,
                                                              const float* __restrict__ w, float* __restrict__ state,
                                                              const float* residual, float* out, float* hist, int T,
                                                              int C) {
  const int c = blockIdx.x * kThreads + threadIdx.x;
  if (c >= C) return;
  const float inv = sqrtf(5.0f);
  float h0 = state[c], h1 = state[C + c], h2 = state[2 * C + c];
  const float w0 = w[c * 4 + 0], w1 = w[c * 4 + 1], w2 = w[c * 4 + 2], w3 = w[c * 4 + 3];
  for (int t = 0; t < T; ++t) {
    const int64_t i = static_cast<int64_t>(t) * C + c;
    const float xt = LoadF(x, i, x_bf16);
    if (hist != nullptr) hist[i] = xt;
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
                                                               float* __restrict__ out, float* hist, int T, int C,
                                                               int heads) {
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
    if (hist != nullptr) hist[i] = xt;
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
        if (a.o.out_fp8) {
        reinterpret_cast<uint8_t*>(a.o.out)[idx] = FloatToFp8E4m3Fn(y);
      } else {
        reinterpret_cast<uint16_t*>(a.o.out)[idx] = r4dx::core::FloatToBf16(y);
      }
      else
        reinterpret_cast<float*>(a.o.out)[idx] = y;
    } else {
      const int64_t p = pos0 + t;
      const int* bt = reinterpret_cast<const int*>(a.o.block_table);
      const int64_t idx = static_cast<int64_t>(bt[p / a.o.block_size]) * a.o.kv_block_stride +
                          static_cast<int64_t>(h) * a.o.kv_head_stride + (p % a.o.block_size) * 2 * a.d +
                          a.o.col_offset + i;
      if (a.o.out_fp8) {
        reinterpret_cast<uint8_t*>(a.o.out)[idx] = FloatToFp8E4m3Fn(y);
      } else {
        reinterpret_cast<uint16_t*>(a.o.out)[idx] = r4dx::core::FloatToBf16(y);
      }
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
  if ((H & 3) == 0) {
    const int H4 = H >> 2;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    const float4* mr4 = reinterpret_cast<const float4*>(mr);
    for (int h4 = threadIdx.x; h4 < H4; h4 += kThreads) {
      const float4 xv = xr4[h4];
      const float4 mv = mr4[h4];
      sx += (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
      sm += (mv.x * mv.x + mv.y * mv.y) + (mv.z * mv.z + mv.w * mv.w);
    }
  } else {
    for (int h = threadIdx.x; h < H; h += kThreads) {
      sx += xr[h] * xr[h];
      sm += mr[h] * mr[h];
    }
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
// Fast zero-LDS-tile float4 MsgApply (R % 4 == 0): 16 lanes per output channel h (8 channels per
// 128-thread block), 100% of waves active, coalesced float4 loads along R.
__device__ __forceinline__ float HalfWaveSum(float v) {
  v += __shfl_xor(v, 8, 32);
  v += __shfl_xor(v, 4, 32);
  v += __shfl_xor(v, 2, 32);
  v += __shfl_xor(v, 1, 32);
  return v;
}

__device__ __forceinline__ float HalfWaveMax(float v) {
  v = fmaxf(v, __shfl_xor(v, 8, 32));
  v = fmaxf(v, __shfl_xor(v, 4, 32));
  v = fmaxf(v, __shfl_xor(v, 2, 32));
  v = fmaxf(v, __shfl_xor(v, 1, 32));
  return v;
}

// Batched 3-gate (Q, K, V) MsgGate kernels: grid.z = 3 (gates 0, 1, 2) -> 288 blocks across 64 WGPs instead of 1!
struct Ptr3F32 {
  const float* p[3];
};
struct Ptr3U16 {
  const uint16_t* p[3];
};
struct OutPtr3U16 {
  uint16_t* p[3];
};

__global__ __launch_bounds__(kThreads) void MsgPrep3Kernel(const float* __restrict__ x, int64_t x_gate_stride,
                                                           const float* __restrict__ m_all,
                                                           uint16_t* __restrict__ out_cat, int T, int H, float eps) {
  __shared__ float red[kWaves];
  const int64_t t = blockIdx.x;
  const int gate = static_cast<int>(blockIdx.y);
  const float* xr = x + static_cast<int64_t>(gate) * x_gate_stride + t * H;
  const float* mr = m_all + (static_cast<int64_t>(gate) * T + t) * H;
  float sx = 0.0f, sm = 0.0f;
  if ((H & 3) == 0) {
    const int H4 = H >> 2;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    const float4* mr4 = reinterpret_cast<const float4*>(mr);
    for (int h4 = threadIdx.x; h4 < H4; h4 += kThreads) {
      const float4 xv = xr4[h4];
      const float4 mv = mr4[h4];
      sx += (xv.x * xv.x + xv.y * xv.y) + (xv.z * xv.z + xv.w * xv.w);
      sm += (mv.x * mv.x + mv.y * mv.y) + (mv.z * mv.z + mv.w * mv.w);
    }
  } else {
    for (int h = threadIdx.x; h < H; h += kThreads) {
      sx += xr[h] * xr[h];
      sm += mr[h] * mr[h];
    }
  }
  sx = BlockSum(sx, red);
  sm = BlockSum(sm, red);
  const float ix = 1.0f / sqrtf(sx / static_cast<float>(H) + eps);
  const float im = 1.0f / sqrtf(sm / static_cast<float>(H) + eps);
  uint16_t* o = out_cat + (static_cast<int64_t>(gate) * T + t) * (2 * H);
  for (int h = threadIdx.x; h < H; h += kThreads) {
    o[h] = r4dx::core::FloatToBf16(xr[h] * ix);
    o[H + h] = r4dx::core::FloatToBf16(mr[h] * im);
  }
}

__global__ __launch_bounds__(128) void MsgApply3Vec4Kernel(
    const float* __restrict__ x, int64_t x_gate_stride, const float* __restrict__ m_all,
    const float* __restrict__ g_all, Ptr3F32 forget, Ptr3F32 gain, float* __restrict__ out_all, int T, int H, int R) {
  __shared__ float gs[kMaxRank];
  const int64_t t = blockIdx.y;
  const int gate = static_cast<int>(blockIdx.z);
  const int tid = threadIdx.x;
  const int64_t gt_row = static_cast<int64_t>(gate) * T + t;
  for (int r = tid; r < R; r += 128) gs[r] = g_all[gt_row * R + r];
  __syncthreads();
  const int sub = tid >> 4;
  const int lane16 = tid & 15;
  const int h = static_cast<int>(blockIdx.x) * 8 + sub;
  if (h >= H) return;
  const int R4 = R >> 2;
  const float4* fr = reinterpret_cast<const float4*>(forget.p[gate] + static_cast<int64_t>(h) * R);
  const float4* gr = reinterpret_cast<const float4*>(gain.p[gate] + static_cast<int64_t>(h) * R);
  const float4* g4 = reinterpret_cast<const float4*>(gs);
  float f = 0.0f, a = 0.0f;
  for (int r4 = lane16; r4 < R4; r4 += 16) {
    const float4 gv = g4[r4];
    const float4 fv = fr[r4];
    const float4 av = gr[r4];
    f += gv.x * fv.x; f += gv.y * fv.y; f += gv.z * fv.z; f += gv.w * fv.w;
    a += gv.x * av.x; a += gv.y * av.y; a += gv.z * av.z; a += gv.w * av.w;
  }
  f = HalfWaveSum(f);
  a = HalfWaveSum(a);
  if (lane16 == 0) {
    const float xv = x[static_cast<int64_t>(gate) * x_gate_stride + t * H + h];
    const float mv = m_all[gt_row * H + h];
    out_all[gt_row * H + h] = (1.0f - tanhf(f)) * xv + tanhf(a) * mv;
  }
}

__global__ __launch_bounds__(kThreads) void RmsNorm3F32InBf16Kernel(
    const float* __restrict__ x_all, Ptr3U16 nw, OutPtr3U16 out, int T, int H, float eps) {
  __shared__ float red[kWaves];
  const int64_t t = blockIdx.x;
  const int gate = static_cast<int>(blockIdx.y);
  const float* xr = x_all + (static_cast<int64_t>(gate) * T + t) * H;
  const uint16_t* w = nw.p[gate];
  uint16_t* dst = out.p[gate] + t * H;
  float ss = 0.0f;
  if ((H & 3) == 0) {
    const int H4 = H >> 2;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);
    for (int h4 = threadIdx.x; h4 < H4; h4 += kThreads) {
      const float4 v = xr4[h4];
      ss += (v.x * v.x + v.y * v.y) + (v.z * v.z + v.w * v.w);
    }
  } else {
    for (int h = threadIdx.x; h < H; h += kThreads) {
      const float v = xr[h];
      ss += v * v;
    }
  }
  ss = BlockSum(ss, red);
  const float rstd = rsqrtf(ss / static_cast<float>(H) + eps);
  for (int h = threadIdx.x; h < H; h += kThreads) {
    dst[h] = r4dx::core::FloatToBf16(xr[h] * rstd * r4dx::core::Bf16ToFloat(w[h]));
  }
}

__global__ __launch_bounds__(128) void MsgApplyVec4Kernel(
    const float* __restrict__ x, const float* __restrict__ m, const void* __restrict__ g, int g_bf16,
    const float* __restrict__ forget, const float* __restrict__ gain, float* __restrict__ out, int H, int R) {
  __shared__ float gs[kMaxRank];
  const int64_t t = blockIdx.y;
  const int tid = threadIdx.x;
  for (int r = tid; r < R; r += 128) gs[r] = LoadF(g, t * R + r, g_bf16);
  __syncthreads();
  const int sub = tid >> 4;   // 0..7
  const int lane16 = tid & 15;
  const int h = static_cast<int>(blockIdx.x) * 8 + sub;
  if (h >= H) return;
  const int R4 = R >> 2;
  const float4* fr = reinterpret_cast<const float4*>(forget + static_cast<int64_t>(h) * R);
  const float4* gr = reinterpret_cast<const float4*>(gain + static_cast<int64_t>(h) * R);
  const float4* g4 = reinterpret_cast<const float4*>(gs);
  float f = 0.0f, a = 0.0f;
  for (int r4 = lane16; r4 < R4; r4 += 16) {
    const float4 gv = g4[r4];
    const float4 fv = fr[r4];
    const float4 av = gr[r4];
    f += gv.x * fv.x; f += gv.y * fv.y; f += gv.z * fv.z; f += gv.w * fv.w;
    a += gv.x * av.x; a += gv.y * av.y; a += gv.z * av.z; a += gv.w * av.w;
  }
  f = HalfWaveSum(f);
  a = HalfWaveSum(a);
  if (lane16 == 0) {
    const int64_t i = t * H + h;
    out[i] = (1.0f - tanhf(f)) * x[i] + tanhf(a) * m[i];
  }
}

// 4-token-tiled float4 MsgApply for prefill (T >= 4, R % 4 == 0): reads forget/gain ONCE per 4 tokens.
__global__ __launch_bounds__(128) void MsgApplyTiled4Kernel(
    const float* __restrict__ x, const float* __restrict__ m, const void* __restrict__ g, int g_bf16,
    const float* __restrict__ forget, const float* __restrict__ gain, float* __restrict__ out, int T, int H, int R) {
  __shared__ float gs[4][kMaxRank];
  const int t0 = static_cast<int>(blockIdx.y) * 4;
  const int tid = threadIdx.x;
  const int nt = (T - t0 < 4) ? (T - t0) : 4;
  for (int idx = tid; idx < nt * R; idx += 128) {
    const int dt = idx / R;
    const int r = idx - dt * R;
    gs[dt][r] = LoadF(g, static_cast<int64_t>(t0 + dt) * R + r, g_bf16);
  }
  __syncthreads();
  const int sub = tid >> 4;
  const int lane16 = tid & 15;
  const int h = static_cast<int>(blockIdx.x) * 8 + sub;
  if (h >= H) return;
  const int R4 = R >> 2;
  const float4* fr = reinterpret_cast<const float4*>(forget + static_cast<int64_t>(h) * R);
  const float4* gr = reinterpret_cast<const float4*>(gain + static_cast<int64_t>(h) * R);
  float f0 = 0.0f, f1 = 0.0f, f2 = 0.0f, f3 = 0.0f;
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (int r4 = lane16; r4 < R4; r4 += 16) {
    const float4 fv = fr[r4];
    const float4 av = gr[r4];
    const float4 g0 = reinterpret_cast<const float4*>(gs[0])[r4];
    f0 += g0.x * fv.x; f0 += g0.y * fv.y; f0 += g0.z * fv.z; f0 += g0.w * fv.w;
    a0 += g0.x * av.x; a0 += g0.y * av.y; a0 += g0.z * av.z; a0 += g0.w * av.w;
    if (nt > 1) {
      const float4 g1 = reinterpret_cast<const float4*>(gs[1])[r4];
      f1 += g1.x * fv.x; f1 += g1.y * fv.y; f1 += g1.z * fv.z; f1 += g1.w * fv.w;
      a1 += g1.x * av.x; a1 += g1.y * av.y; a1 += g1.z * av.z; a1 += g1.w * av.w;
    }
    if (nt > 2) {
      const float4 g2 = reinterpret_cast<const float4*>(gs[2])[r4];
      f2 += g2.x * fv.x; f2 += g2.y * fv.y; f2 += g2.z * fv.z; f2 += g2.w * fv.w;
      a2 += g2.x * av.x; a2 += g2.y * av.y; a2 += g2.z * av.z; a2 += g2.w * av.w;
    }
    if (nt > 3) {
      const float4 g3 = reinterpret_cast<const float4*>(gs[3])[r4];
      f3 += g3.x * fv.x; f3 += g3.y * fv.y; f3 += g3.z * fv.z; f3 += g3.w * fv.w;
      a3 += g3.x * av.x; a3 += g3.y * av.y; a3 += g3.z * av.z; a3 += g3.w * av.w;
    }
  }
  f0 = HalfWaveSum(f0); a0 = HalfWaveSum(a0);
  if (nt > 1) { f1 = HalfWaveSum(f1); a1 = HalfWaveSum(a1); }
  if (nt > 2) { f2 = HalfWaveSum(f2); a2 = HalfWaveSum(a2); }
  if (nt > 3) { f3 = HalfWaveSum(f3); a3 = HalfWaveSum(a3); }
  if (lane16 == 0) {
    const int64_t i0 = static_cast<int64_t>(t0) * H + h;
    out[i0] = (1.0f - tanhf(f0)) * x[i0] + tanhf(a0) * m[i0];
    if (nt > 1) { const int64_t i1 = i0 + H; out[i1] = (1.0f - tanhf(f1)) * x[i1] + tanhf(a1) * m[i1]; }
    if (nt > 2) { const int64_t i2 = i0 + 2 * H; out[i2] = (1.0f - tanhf(f2)) * x[i2] + tanhf(a2) * m[i2]; }
    if (nt > 3) { const int64_t i3 = i0 + 3 * H; out[i3] = (1.0f - tanhf(f3)) * x[i3] + tanhf(a3) * m[i3]; }
  }
}

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

// Dot product of 8 packed bf16s (uint4) in fp32.
__device__ __forceinline__ float Dot8Bf16(const uint4& a, const uint4& b) {
  float s = 0.0f;
  s += r4dx::core::Bf16ToFloat(static_cast<uint16_t>(a.x & 0xFFFFu)) * r4dx::core::Bf16ToFloat(static_cast<uint16_t>(b.x & 0xFFFFu));
  s += r4dx::core::Bf16ToFloat(static_cast<uint16_t>(a.x >> 16)) * r4dx::core::Bf16ToFloat(static_cast<uint16_t>(b.x >> 16));
  s += r4dx::core::Bf16ToFloat(static_cast<uint16_t>(a.y & 0xFFFFu)) * r4dx::core::Bf16ToFloat(static_cast<uint16_t>(b.y & 0xFFFFu));
  s += r4dx::core::Bf16ToFloat(static_cast<uint16_t>(a.y >> 16)) * r4dx::core::Bf16ToFloat(static_cast<uint16_t>(b.y >> 16));
  s += r4dx::core::Bf16ToFloat(static_cast<uint16_t>(a.z & 0xFFFFu)) * r4dx::core::Bf16ToFloat(static_cast<uint16_t>(b.z & 0xFFFFu));
  s += r4dx::core::Bf16ToFloat(static_cast<uint16_t>(a.z >> 16)) * r4dx::core::Bf16ToFloat(static_cast<uint16_t>(b.z >> 16));
  s += r4dx::core::Bf16ToFloat(static_cast<uint16_t>(a.w & 0xFFFFu)) * r4dx::core::Bf16ToFloat(static_cast<uint16_t>(b.w & 0xFFFFu));
  s += r4dx::core::Bf16ToFloat(static_cast<uint16_t>(a.w >> 16)) * r4dx::core::Bf16ToFloat(static_cast<uint16_t>(b.w >> 16));
  return s;
}

// Multi-block decode router (uint4 vectorized when H % 8 == 0): 4 waves/block, 1 wave per expert.
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
  if ((H & 7) == 0 && (x_row_stride & 7) == 0) {
    const int H8 = H >> 3;
    const uint4* x8 = reinterpret_cast<const uint4*>(xr);
    const uint4* w8 = reinterpret_cast<const uint4*>(wr);
    for (int h = lane; h < H8; h += 32) a += Dot8Bf16(x8[h], w8[h]);
  } else {
    for (int h = lane; h < H; h += 32) a += r4dx::core::Bf16ToFloat(xr[h]) * r4dx::core::Bf16ToFloat(wr[h]);
  }
  a = WaveSum(a);
  if (lane == 0) logits_out[t * E + e] = a;
}

// 8-token-tiled prefill router (H % 8 == 0): reads gw[e] ONCE per 8 tokens (8x less weight traffic).
__global__ __launch_bounds__(kRouterWavesPerBlock * 32) void RouterDotsTiled8Kernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ gw, float* __restrict__ logits_out,
    int T, int H, int E, int64_t x_row_stride) {
  const int t0 = static_cast<int>(blockIdx.y) * 8;
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int e = static_cast<int>(blockIdx.x) * kRouterWavesPerBlock + wave;
  if (e >= E) return;
  const int nt = (T - t0 < 8) ? (T - t0) : 8;
  const int H8 = H >> 3;
  const int64_t s8 = x_row_stride >> 3;
  const uint4* x8 = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(t0) * x_row_stride);
  const uint4* w8 = reinterpret_cast<const uint4*>(gw + static_cast<int64_t>(e) * H);
  float a[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  for (int h = lane; h < H8; h += 32) {
    const uint4 wv = w8[h];
#pragma unroll
    for (int dt = 0; dt < 8; ++dt) {
      if (dt < nt) a[dt] += Dot8Bf16(x8[dt * s8 + h], wv);
    }
  }
#pragma unroll
  for (int dt = 0; dt < 8; ++dt) {
    if (dt < nt) {
      const float v = WaveSum(a[dt]);
      if (lane == 0) logits_out[static_cast<int64_t>(t0 + dt) * E + e] = v;
    }
  }
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

// ---- dedicated Inkling2 MoE sort, uint4 gather, and uint2 combine -------------------------------------------------
constexpr int kInkSortMaxExperts = 256;
constexpr int kInkSortMaxPairs = 16384;

// Segmented parallel deterministic counting sort (256 threads):
// When n_experts <= 128 and P >= 32, assigns 2 threads per expert (seg 0 = [0, P/2), seg 1 = [P/2, P))
// so all 256 threads are active and scan half the array in uint32 chunks (4x fewer LDS reads per thread).
__global__ __launch_bounds__(kInkSortMaxExperts) void Inkling2MoeSortKernel(
    const int32_t* __restrict__ ids, int P, int n_experts, int32_t* __restrict__ counts,
    int32_t* __restrict__ offsets, int32_t* __restrict__ sorted_pairs, int32_t* __restrict__ pos_of_pair) {
  __shared__ uint8_t id_s[kInkSortMaxPairs];
  __shared__ int32_t seg_cnt[2][kInkSortMaxExperts];
  __shared__ int32_t off_s[kInkSortMaxExperts + 1];
  for (int p = threadIdx.x; p < P; p += blockDim.x) {
    const int32_t e = ids[p];
    const bool ok = e >= 0 && e < n_experts;
    id_s[p] = ok ? static_cast<uint8_t>(e) : static_cast<uint8_t>(255);
    if (!ok) pos_of_pair[p] = -1;
  }
  __syncthreads();

  if (n_experts <= 128 && P >= 32) {
    const int e = threadIdx.x & 127;
    const int seg = threadIdx.x >> 7;  // 0 or 1
    const int mid = (P >> 1) & ~3;
    const int p0 = (seg == 0) ? 0 : mid;
    const int p1 = (seg == 0) ? mid : P;
    int cnt = 0;
    if (e < n_experts) {
      const uint8_t eb = static_cast<uint8_t>(e);
      int p = p0;
      for (; p + 4 <= p1; p += 4) {
        const uint32_t w = *reinterpret_cast<const uint32_t*>(&id_s[p]);
        cnt += ((w & 0xFFu) == eb) + (((w >> 8) & 0xFFu) == eb) + (((w >> 16) & 0xFFu) == eb) + ((w >> 24) == eb);
      }
      for (; p < p1; ++p) cnt += (id_s[p] == eb);
      seg_cnt[seg][e] = cnt;
    }
    __syncthreads();
    if (seg == 0 && e < n_experts) {
      const int total_e = seg_cnt[0][e] + seg_cnt[1][e];
      counts[e] = total_e;
      off_s[e] = total_e;
    }
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
      if (seg == 0) offsets[e] = off_s[e];
      int pos = off_s[e] + (seg == 1 ? seg_cnt[0][e] : 0);
      const uint8_t eb = static_cast<uint8_t>(e);
      for (int p = p0; p < p1; ++p) {
        if (id_s[p] == eb) {
          sorted_pairs[pos] = p;
          pos_of_pair[p] = pos;
          ++pos;
        }
      }
    }
    if (threadIdx.x == 0) offsets[n_experts] = off_s[n_experts];
    return;
  }

  const int e = threadIdx.x;
  int cnt = 0;
  if (e < n_experts) {
    for (int p = 0; p < P; ++p) cnt += (id_s[p] == e) ? 1 : 0;
    counts[e] = cnt;
    off_s[e] = cnt;
  }
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

// uint4-vectorized (128-bit, 8x bf16 per load/store) MoE gather.
__global__ __launch_bounds__(256) void Inkling2MoeGatherKernel(
    const uint16_t* __restrict__ x, const int32_t* __restrict__ sorted_pairs, uint16_t* __restrict__ out, int n_pairs,
    int top_k, int hidden, int64_t x_stride, const int32_t* __restrict__ valid_pairs) {
  const int64_t i = blockIdx.x;
  const int n_valid = valid_pairs ? valid_pairs[0] : n_pairs;
  if (i >= n_valid) return;
  const int64_t tok = sorted_pairs[i] / top_k;
  const uint16_t* src = x + tok * x_stride;
  uint16_t* dst = out + i * hidden;
  if ((hidden & 7) == 0 && (x_stride & 7) == 0 && (reinterpret_cast<uintptr_t>(src) & 15) == 0 &&
      (reinterpret_cast<uintptr_t>(dst) & 15) == 0) {
    const int H8 = hidden >> 3;
    const uint4* src8 = reinterpret_cast<const uint4*>(src);
    uint4* dst8 = reinterpret_cast<uint4*>(dst);
    for (int k8 = threadIdx.x; k8 < H8; k8 += 256) dst8[k8] = src8[k8];
  } else {
    for (int k = threadIdx.x; k < hidden; k += 256) dst[k] = src[k];
  }
}

// uint2-vectorized (4x bf16 per thread) MoE combine.
__global__ __launch_bounds__(256) void Inkling2MoeCombineKernel(
    const uint16_t* __restrict__ y, const int32_t* __restrict__ pos_of_pair, const float* __restrict__ w,
    uint16_t* __restrict__ out, int top_k, int hidden) {
  const int64_t t = blockIdx.x;
  if ((hidden & 3) == 0 && (reinterpret_cast<uintptr_t>(y) & 7) == 0 && (reinterpret_cast<uintptr_t>(out) & 7) == 0) {
    const int H4 = hidden >> 2;
    const int k4 = blockIdx.y * 256 + threadIdx.x;
    if (k4 >= H4) return;
    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (int s = 0; s < top_k; ++s) {
      const int64_t p = t * top_k + s;
      const int64_t row = pos_of_pair ? static_cast<int64_t>(pos_of_pair[p]) : p;
      if (row < 0) continue;
      const float wp = w[p];
      const uint2 yv = reinterpret_cast<const uint2*>(y + row * hidden)[k4];
      a0 += wp * r4dx::core::Bf16ToFloat(static_cast<uint16_t>(yv.x & 0xFFFFu));
      a1 += wp * r4dx::core::Bf16ToFloat(static_cast<uint16_t>(yv.x >> 16));
      a2 += wp * r4dx::core::Bf16ToFloat(static_cast<uint16_t>(yv.y & 0xFFFFu));
      a3 += wp * r4dx::core::Bf16ToFloat(static_cast<uint16_t>(yv.y >> 16));
    }
    const uint32_t o01 =
        static_cast<uint32_t>(r4dx::core::FloatToBf16(a0)) | (static_cast<uint32_t>(r4dx::core::FloatToBf16(a1)) << 16);
    const uint32_t o23 =
        static_cast<uint32_t>(r4dx::core::FloatToBf16(a2)) | (static_cast<uint32_t>(r4dx::core::FloatToBf16(a3)) << 16);
    reinterpret_cast<uint2*>(out + t * hidden)[k4] = make_uint2(o01, o23);
    return;
  }
  for (int k = blockIdx.y * 256 + threadIdx.x; k < hidden; k += gridDim.y * 256) {
    float acc = 0.0f;
    for (int s = 0; s < top_k; ++s) {
      const int64_t p = t * top_k + s;
      const int64_t row = pos_of_pair ? static_cast<int64_t>(pos_of_pair[p]) : p;
      if (row < 0) continue;
      acc += w[p] * r4dx::core::Bf16ToFloat(y[row * hidden + k]);
    }
    out[t * hidden + k] = r4dx::core::FloatToBf16(acc);
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
  const int t = static_cast<int>(blockIdx.z);
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int qh0 = kvh * GQA;
  const int64_t t_qoff = static_cast<int64_t>(t) * a.q_heads;
  const int qpos = (a.pos_dev != 0) ? (*reinterpret_cast<const int32_t*>(a.pos_dev) + t) : (a.ctx - a.q_len + t);
  const int klow = r4d_win_klow(qpos, a.window);
  const int klen = qpos - klow + 1;
  int active_splits = (klen > kDecTile && n_splits > 1) ? ((klen + kDecTile - 1) / kDecTile) : 1;
  if (active_splits > n_splits) active_splits = n_splits;
  if (split >= active_splits) {
    if (n_splits > 1 && wave == 0 && lane == 0) {
      float* ws_ptr = reinterpret_cast<float*>(a.workspace);
#pragma unroll
      for (int g = 0; g < GQA; ++g) {
        float* dst = ws_ptr + ((t_qoff + qh0 + g) * n_splits + split) * (kFusedHd + 2);
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
    const uint2 qv = *reinterpret_cast<const uint2*>(q_ptr + (t_qoff + qh0 + g) * kFusedHd + lane * 4);
    Unpack4Bf16(qv, q0[g], q1[g], q2[g], q3[g]);
    o0[g] = o1[g] = o2[g] = o3[g] = 0.0f;
    mv[g] = -INFINITY;
    lv[g] = 0.0f;
    rv[g] = (has_rel && lane < a.d_rel) ? r_ptr[(t_qoff + qh0 + g) * a.d_rel + lane] : 0.0f;
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
      float k0, k1, k2, k3;
      if (a.kv_fp8) {
        const uint32_t p = *reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(a.kv) + base + lane * 4);
        Unpack4Fp8(p, k0, k1, k2, k3);
      } else {
        const uint2 kv_k = *reinterpret_cast<const uint2*>(kv_ptr + base + lane * 4);
        Unpack4Bf16(kv_k, k0, k1, k2, k3);
      }
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
      float v0, v1, v2, v3;
      if (a.kv_fp8) {
        const uint32_t p =
            *reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(a.kv) + base + kFusedHd + lane * 4);
        Unpack4Fp8(p, v0, v1, v2, v3);
      } else {
        const uint2 kv_v = *reinterpret_cast<const uint2*>(kv_ptr + base + kFusedHd + lane * 4);
        Unpack4Bf16(kv_v, v0, v1, v2, v3);
      }
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
      const int64_t th = t_qoff + qh0 + g;
      if (n_splits == 1) {
        if (sink_ptr != nullptr) {
          const float sv = sink_ptr[th];
          const float mn = fmaxf(m_all, sv);
          const float alpha = (m_all == -INFINITY) ? 0.0f : expf(m_all - mn);
          l_all = l_all * alpha + expf(sv - mn);
          r0 *= alpha;
          r1 *= alpha;
          r2 *= alpha;
          r3 *= alpha;
        }
        const float inv = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;
        float y0 = r0 * inv, y1 = r1 * inv, y2 = r2 * inv, y3 = r3 * inv;
        if (a.gate != 0) {
          const uint2 gv =
              *reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(a.gate) + th * kFusedHd + lane * 4);
          float g0, g1, g2, g3;
          Unpack4Bf16(gv, g0, g1, g2, g3);
          y0 = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(y0)) * SigmoidF(g0);
          y1 = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(y1)) * SigmoidF(g1);
          y2 = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(y2)) * SigmoidF(g2);
          y3 = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(y3)) * SigmoidF(g3);
        }
        *reinterpret_cast<uint2*>(out_ptr + th * kFusedHd + lane * 4) = Pack4Bf16(y0, y1, y2, y3);
      } else {
        // Workspace layout per (t, h, split): [m, l, o[0..127]] (130 floats).
        float* dst = ws_ptr + (th * n_splits + split) * (kFusedHd + 2);
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

// Combine kernel for split-K decode: grid = (q_heads, q_len), block = (32).
__global__ __launch_bounds__(32) void Inkling2AttnDecodeCombineKernel(
    const float* __restrict__ ws, const float* __restrict__ sink, const uint16_t* __restrict__ gate,
    uint16_t* __restrict__ out, int q_heads, int n_splits) {
  const int h = static_cast<int>(blockIdx.x);
  const int t = static_cast<int>(blockIdx.y);
  const int64_t th = static_cast<int64_t>(t) * q_heads + h;
  const int lane = threadIdx.x;
  const float* base = ws + th * n_splits * (kFusedHd + 2);
  float m_all = (sink != nullptr) ? sink[th] : -INFINITY;
  for (int s = 0; s < n_splits; ++s) {
    const float ms = base[static_cast<int64_t>(s) * (kFusedHd + 2) + 0];
    m_all = fmaxf(m_all, ms);
  }
  float l_all = (sink != nullptr) ? expf(sink[th] - m_all) : 0.0f;
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
  float y0 = r0 * inv, y1 = r1 * inv, y2 = r2 * inv, y3 = r3 * inv;
  if (gate != nullptr) {
    const uint2 gv = *reinterpret_cast<const uint2*>(gate + th * kFusedHd + lane * 4);
    float g0, g1, g2, g3;
    Unpack4Bf16(gv, g0, g1, g2, g3);
    y0 = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(y0)) * SigmoidF(g0);
    y1 = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(y1)) * SigmoidF(g1);
    y2 = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(y2)) * SigmoidF(g2);
    y3 = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(y3)) * SigmoidF(g3);
  }
  *reinterpret_cast<uint2*>(out + th * kFusedHd + lane * 4) = Pack4Bf16(y0, y1, y2, y3);
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
  const int ctx = (a.pos_dev != 0) ? (*reinterpret_cast<const int32_t*>(a.pos_dev) + a.q_len) : a.ctx;
  const int qpos = ctx - a.q_len + t;
  const int klow = valid_t ? r4d_win_klow(qpos, a.window) : ctx;
  const int khigh = valid_t ? (qpos < ctx - 1 ? qpos : ctx - 1) : -1;

  // Tile K range covering all valid queries in this block [t0 .. min(t0+3, q_len-1)].
  const int t_last = (t0 + kFusedWaves - 1 < a.q_len) ? (t0 + kFusedWaves - 1) : (a.q_len - 1);
  const int k_tile_begin = r4d_win_klow(ctx - a.q_len + t0, a.window);
  const int k_tile_end_raw = ctx - a.q_len + t_last;
  const int k_tile_end = (k_tile_end_raw < ctx - 1 ? k_tile_end_raw : ctx - 1) + 1;

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
      if (a.kv_fp8) {
        const uint8_t* kv8 = reinterpret_cast<const uint8_t*>(a.kv);
        sk[j][lane].x = *reinterpret_cast<const uint32_t*>(kv8 + base + lane * 4);
        sv[j][lane].x = *reinterpret_cast<const uint32_t*>(kv8 + base + kFusedHd + lane * 4);
      } else {
        sk[j][lane] = *reinterpret_cast<const uint2*>(kv_ptr + base + lane * 4);
        sv[j][lane] = *reinterpret_cast<const uint2*>(kv_ptr + base + kFusedHd + lane * 4);
      }
    }
    __syncthreads();

    if (valid_t) {
      for (int j = 0; j < n_tile; ++j) {
        const int k = kb + j;
        if (k < klow || k > khigh) continue;
        float k0, k1, k2, k3, v0, v1, v2, v3;
        if (a.kv_fp8) {
          Unpack4Fp8(sk[j][lane].x, k0, k1, k2, k3);
          Unpack4Fp8(sv[j][lane].x, v0, v1, v2, v3);
        } else {
          Unpack4Bf16(sk[j][lane], k0, k1, k2, k3);
          Unpack4Bf16(sv[j][lane], v0, v1, v2, v3);
        }

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

// WMMA BF16 FlashAttention prefill kernel (gfx1201, head_dim == 128, d_rel <= 16):
// Grid = (q_heads, ceil(q_len / 64)), Block = (128) = 4 waves (16 query rows per wave).
// Uses 16x16x16 bf16 WMMA for both Q*K^T (8 ops) and P*V (8 ops) per 16-key tile.
__global__ __launch_bounds__(128) void Inkling2AttnPrefillWmmaKernel(const R4dxInkling2AttnFusedArgs a) {
  __shared__ unsigned short sk[16][128];       // 4 KiB: K[16][128]
  __shared__ unsigned short sv_t[128][16];     // 4 KiB: V^T[128][16]
  __shared__ unsigned short sp_hi[4][16][16];  // 2 KiB: P_hi[wave][16][16] (bf16)
  __shared__ unsigned short sp_lo[4][16][16];  // 2 KiB: P_lo[wave][16][16] (bf16 residual -> 14-bit P)
  __shared__ float sr[4][16][16];              // 4 KiB: r[wave][16][d_rel]

  const int h = static_cast<int>(blockIdx.x);
  const int gqa = a.q_heads / a.kv_heads;
  const int kvh = h / gqa;
  const int t0 = static_cast<int>(blockIdx.y) * 64;
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int tw = t0 + wave * 16;
  const int n = lane & 15;
  const int m_hi = (lane >> 4) * 8;

  const int ctx = (a.pos_dev != 0) ? (*reinterpret_cast<const int32_t*>(a.pos_dev) + a.q_len) : a.ctx;
  const int t_last = (t0 + 63 < a.q_len) ? (t0 + 63) : (a.q_len - 1);
  const int k_tile_begin = r4d_win_klow(ctx - a.q_len + t0, a.window);
  const int k_tile_end_raw = ctx - a.q_len + t_last;
  const int k_tile_end = (k_tile_end_raw < ctx - 1 ? k_tile_end_raw : ctx - 1) + 1;

  const unsigned short* q_ptr = reinterpret_cast<const unsigned short*>(a.q);
  const uint16_t* kv_ptr = reinterpret_cast<const uint16_t*>(a.kv);
  const uint8_t* kv8_ptr = reinterpret_cast<const uint8_t*>(a.kv);
  const int* bt = reinterpret_cast<const int*>(a.block_table);
  const float* r_ptr = a.r ? reinterpret_cast<const float*>(a.r) : nullptr;
  const float* proj_ptr = a.rel_proj ? reinterpret_cast<const float*>(a.rel_proj) : nullptr;
  const bool has_rel = (r_ptr != nullptr && proj_ptr != nullptr && a.d_rel > 0 && a.rel_extent > 0);

  // Load Q[16][128] into 8 v8bf fragments (32 VGPRs).
  v8bf q_frag[8];
  const int qr = tw + n;
  if (qr < a.q_len) {
    const unsigned short* qrow = q_ptr + (static_cast<size_t>(qr) * a.q_heads + h) * 128 + 4 * (lane >> 4);
#pragma unroll
    for (int s = 0; s < 8; ++s) q_frag[s] = fragP(qrow + s * 16);
  } else {
#pragma unroll
    for (int s = 0; s < 8; ++s) q_frag[s] = v8bf{};
  }

  // Load r[16][d_rel] into LDS once per wave.
  if (has_rel) {
    for (int idx = lane; idx < 16 * 16; idx += 32) {
      const int rm = idx >> 4;
      const int rd = idx & 15;
      const int rt = tw + rm;
      sr[wave][rm][rd] =
          (rt < a.q_len && rd < a.d_rel) ? r_ptr[(static_cast<int64_t>(rt) * a.q_heads + h) * a.d_rel + rd] : 0.0f;
    }
  }

  v8f o_frag[8];
#pragma unroll
  for (int d = 0; d < 8; ++d) o_frag[d] = v8f{};
  float m_row[8], l_row[8];
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    m_row[e] = -INFINITY;
    l_row[e] = 0.0f;
  }

  const int bs = a.block_size;
  const int w_t_last = (tw + 15 < a.q_len) ? (tw + 15) : (a.q_len - 1);
  const int w_klow = (tw < a.q_len) ? r4d_win_klow(ctx - a.q_len + tw, a.window) : ctx;
  const int w_khigh = (tw < a.q_len) ? ((ctx - a.q_len + w_t_last < ctx - 1) ? (ctx - a.q_len + w_t_last) : (ctx - 1)) : -1;

  for (int kb = k_tile_begin; kb < k_tile_end; kb += 16) {
    const int n_tile = (k_tile_end - kb < 16) ? (k_tile_end - kb) : 16;
    // Cooperative load of K[16][128] and V^T[128][16].
    for (int j = wave; j < 16; j += 4) {
      if (j < n_tile) {
        const int k = kb + j;
        const int64_t base = static_cast<int64_t>(bt[k / bs]) * a.kv_block_stride +
                             static_cast<int64_t>(kvh) * a.kv_head_stride +
                             static_cast<int64_t>(k % bs) * (2 * kFusedHd);
        uint2 kv_k, kv_v;
        if (a.kv_fp8) {
          float f0, f1, f2, f3;
          Unpack4Fp8(*reinterpret_cast<const uint32_t*>(kv8_ptr + base + lane * 4), f0, f1, f2, f3);
          kv_k = Pack4Bf16(f0, f1, f2, f3);
          Unpack4Fp8(*reinterpret_cast<const uint32_t*>(kv8_ptr + base + kFusedHd + lane * 4), f0, f1, f2, f3);
          kv_v = Pack4Bf16(f0, f1, f2, f3);
        } else {
          kv_k = *reinterpret_cast<const uint2*>(kv_ptr + base + lane * 4);
          kv_v = *reinterpret_cast<const uint2*>(kv_ptr + base + kFusedHd + lane * 4);
        }
        *reinterpret_cast<uint2*>(&sk[j][lane * 4]) = kv_k;
        sv_t[lane * 4 + 0][j] = static_cast<unsigned short>(kv_v.x & 0xFFFFu);
        sv_t[lane * 4 + 1][j] = static_cast<unsigned short>(kv_v.x >> 16);
        sv_t[lane * 4 + 2][j] = static_cast<unsigned short>(kv_v.y & 0xFFFFu);
        sv_t[lane * 4 + 3][j] = static_cast<unsigned short>(kv_v.y >> 16);
      } else {
        *reinterpret_cast<uint2*>(&sk[j][lane * 4]) = make_uint2(0u, 0u);
        sv_t[lane * 4 + 0][j] = 0;
        sv_t[lane * 4 + 1][j] = 0;
        sv_t[lane * 4 + 2][j] = 0;
        sv_t[lane * 4 + 3][j] = 0;
      }
    }
    __syncthreads();

    if (tw < a.q_len && w_khigh >= kb && w_klow < kb + n_tile) {
      // 1. Q * K^T (8 WMMA ops)
      v8f sc = {};
#pragma unroll
      for (int s = 0; s < 8; ++s) {
        sc = mma(q_frag[s], frag(&sk[0][0], 128, 0, s * 16, lane), sc);
      }
      // 2. Scale, rel bias, causal/window mask, online softmax -> sp_bf[wave][16][16]
      const int k = kb + n;
      const bool valid_k = (n < n_tile) && (k < ctx);
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int m = m_hi + e;
        const int t = tw + m;
        const int qpos = ctx - a.q_len + t;
        const bool valid = valid_k && (t < a.q_len) && (k >= r4d_win_klow(qpos, a.window)) && (k <= qpos);
        float val = -INFINITY;
        if (valid) {
          val = sc[e] * a.scale;
          if (has_rel) {
            const int dist = qpos - k;
            if (dist < a.rel_extent) {
              float rb = 0.0f;
              for (int rd = 0; rd < a.d_rel; ++rd) {
                rb += sr[wave][m][rd] * proj_ptr[static_cast<int64_t>(rd) * a.rel_extent + dist];
              }
              val += rb;
            }
          }
        }
        const float tm = HalfWaveMax(val);
        if (tm != -INFINITY) {
          const float mn = fmaxf(m_row[e], tm);
          const float alpha = (m_row[e] == -INFINITY) ? 0.0f : expf(m_row[e] - mn);
          const float pv = (val == -INFINITY) ? 0.0f : expf(val - mn);
          l_row[e] = l_row[e] * alpha + HalfWaveSum(pv);
          m_row[e] = mn;
          const uint16_t p_hi_u = r4dx::core::FloatToBf16(pv);
          const uint16_t p_lo_u = r4dx::core::FloatToBf16(pv - r4dx::core::Bf16ToFloat(p_hi_u));
          sp_hi[wave][m][n] = p_hi_u;
          sp_lo[wave][m][n] = p_lo_u;
          if (alpha != 1.0f) {
#pragma unroll
            for (int d = 0; d < 8; ++d) o_frag[d][e] *= alpha;
          }
        } else {
          sp_hi[wave][m][n] = 0;
          sp_lo[wave][m][n] = 0;
        }
      }
      asm volatile("s_wait_dscnt 0x0" ::: "memory");
      // 3. P * V (2-term bf16 split for 14-bit P precision -> <= 1-2 bf16 ULP, zero extra V loads)
      const v8bf p_hi_f = frag(&sp_hi[wave][0][0], 16, 0, 0, lane);
      const v8bf p_lo_f = frag(&sp_lo[wave][0][0], 16, 0, 0, lane);
#pragma unroll
      for (int d = 0; d < 8; ++d) {
        const v8bf vf = frag(&sv_t[0][0], 16, d * 16, 0, lane);
        o_frag[d] = mma(p_hi_f, vf, o_frag[d]);
        o_frag[d] = mma(p_lo_f, vf, o_frag[d]);
      }
    }
    __syncthreads();
  }

  // Epilogue: sink + normalize + store bf16.
  const float* sink_ptr = a.sink ? reinterpret_cast<const float*>(a.sink) : nullptr;
  uint16_t* out_ptr = reinterpret_cast<uint16_t*>(a.out);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const int m = m_hi + e;
    const int t = tw + m;
    if (t >= a.q_len) continue;
    float scale_o;
    if (sink_ptr != nullptr) {
      const float sv_s = sink_ptr[static_cast<int64_t>(t) * a.q_heads + h];
      const float mn = fmaxf(m_row[e], sv_s);
      const float alpha = (m_row[e] == -INFINITY) ? 0.0f : expf(m_row[e] - mn);
      const float l_tot = l_row[e] * alpha + expf(sv_s - mn);
      scale_o = (l_tot > 0.0f) ? (alpha / l_tot) : 0.0f;
    } else {
      scale_o = (l_row[e] > 0.0f) ? (1.0f / l_row[e]) : 0.0f;
    }
    uint16_t* orow = out_ptr + (static_cast<int64_t>(t) * a.q_heads + h) * 128;
#pragma unroll
    for (int d = 0; d < 8; ++d) {
      orow[d * 16 + n] = r4dx::core::FloatToBf16(o_frag[d][e] * scale_o);
    }
  }
}

// ---- small dense GEMM: uint4 vectorized (8x bf16/load) + 4-row prefill tiling ----------
__global__ __launch_bounds__(32) void Dense3Bf16DecodeKernel(const uint16_t* __restrict__ x_all, Ptr3U16 w,
                                                             float* __restrict__ out_all, int M, int N, int K) {
  const int64_t m = blockIdx.y;
  const int gate = static_cast<int>(blockIdx.z);
  const int n = static_cast<int>(blockIdx.x);
  const int lane = threadIdx.x;
  if (n >= N) return;
  const int64_t gm = static_cast<int64_t>(gate) * M + m;
  const uint16_t* xr = x_all + gm * K;
  const uint16_t* wr = w.p[gate] + static_cast<int64_t>(n) * K;
  float a = 0.0f;
  if ((K & 7) == 0) {
    const int K8 = K >> 3;
    const uint4* x8 = reinterpret_cast<const uint4*>(xr);
    const uint4* w8 = reinterpret_cast<const uint4*>(wr);
    for (int k = lane; k < K8; k += 32) a += Dot8Bf16(x8[k], w8[k]);
  } else {
    for (int k = lane; k < K; k += 32) a += r4dx::core::Bf16ToFloat(xr[k]) * r4dx::core::Bf16ToFloat(wr[k]);
  }
  a = WaveSum(a);
  if (lane == 0) out_all[gm * N + n] = a;
}

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
  if ((K & 7) == 0 && (xs & 7) == 0) {
    const int K8 = K >> 3;
    const uint4* x8 = reinterpret_cast<const uint4*>(xr);
    const uint4* w8 = reinterpret_cast<const uint4*>(wr);
    for (int k = lane; k < K8; k += 32) a += Dot8Bf16(x8[k], w8[k]);
  } else {
    for (int k = lane; k < K; k += 32) a += r4dx::core::Bf16ToFloat(xr[k]) * r4dx::core::Bf16ToFloat(wr[k]);
  }
  a = WaveSum(a);
  if (lane != 0) return;
  if (out_bf16) {
    static_cast<uint16_t*>(out)[m * os + n] = r4dx::core::FloatToBf16(a);
  } else {
    static_cast<float*>(out)[m * os + n] = a;
  }
}

__global__ __launch_bounds__(128) void DenseBf16Tiled4Kernel(const uint16_t* __restrict__ x, int64_t xs,
                                                             const uint16_t* __restrict__ w, void* __restrict__ out,
                                                             int out_bf16, int64_t os, int M, int N, int K) {
  const int m0 = static_cast<int>(blockIdx.y) * 4;
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = static_cast<int>(blockIdx.x) * 4 + wave;
  if (n >= N) return;
  const int nm = (M - m0 < 4) ? (M - m0) : 4;
  const int K8 = K >> 3;
  const int64_t xs8 = xs >> 3;
  const uint4* x8 = reinterpret_cast<const uint4*>(x + static_cast<int64_t>(m0) * xs);
  const uint4* w8 = reinterpret_cast<const uint4*>(w + static_cast<int64_t>(n) * K);
  float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
  for (int k = lane; k < K8; k += 32) {
    const uint4 wv = w8[k];
    a0 += Dot8Bf16(x8[k], wv);
    if (nm > 1) a1 += Dot8Bf16(x8[xs8 + k], wv);
    if (nm > 2) a2 += Dot8Bf16(x8[2 * xs8 + k], wv);
    if (nm > 3) a3 += Dot8Bf16(x8[3 * xs8 + k], wv);
  }
  a0 = WaveSum(a0);
  if (nm > 1) a1 = WaveSum(a1);
  if (nm > 2) a2 = WaveSum(a2);
  if (nm > 3) a3 = WaveSum(a3);
  if (lane != 0) return;
  auto store = [&](int dm, float v) {
    const int64_t idx = static_cast<int64_t>(m0 + dm) * os + n;
    if (out_bf16) static_cast<uint16_t*>(out)[idx] = r4dx::core::FloatToBf16(v);
    else static_cast<float*>(out)[idx] = v;
  };
  store(0, a0);
  if (nm > 1) store(1, a1);
  if (nm > 2) store(2, a2);
  if (nm > 3) store(3, a3);
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

void r4dx_inkling2_attnres_prenorm_f32(const int64_t* states, const int64_t* inv_rms, int n_states, int64_t queries,
                                       int nq, int64_t out, int64_t norm_weight, int64_t out_bf16, int T, int H,
                                       float eps, int64_t stream) {
  const char* fn = "r4dx_inkling2_attnres_f32";
  if (T <= 0) return;
  if (states == nullptr || n_states < 1 || n_states > kMaxStates) Fail(fn, "n_states must be 1..6 with a pointer array");
  if (nq < 1 || nq > kMaxQueries) Fail(fn, "nq must be 1..4");
  if (H < 1 || H > 8192) Fail(fn, "H must be 1..8192");
  const bool fuse_norm = (norm_weight != 0 && out_bf16 != 0 && nq == 1);
  if (queries == 0 || (out == 0 && !fuse_norm)) Fail(fn, "null queries / out");
  AttnResPtrs p{};
  int all_prenorm = 1;
  for (int i = 0; i < n_states; ++i) {
    if (states[i] == 0) Fail(fn, "null state pointer");
    p.s[i] = reinterpret_cast<const float*>(states[i]);
    p.inv[i] = (inv_rms != nullptr && inv_rms[i] != 0) ? reinterpret_cast<const float*>(inv_rms[i]) : nullptr;
    if (p.inv[i] == nullptr) all_prenorm = 0;
  }
  hipLaunchKernelGGL(AttnResPrenormKernel, dim3(static_cast<unsigned>(T)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), p, n_states, all_prenorm,
                     reinterpret_cast<const float*>(queries), nq, reinterpret_cast<float*>(out),
                     fuse_norm ? reinterpret_cast<const uint16_t*>(norm_weight) : nullptr,
                     fuse_norm ? reinterpret_cast<uint16_t*>(out_bf16) : nullptr, T, H, eps);
  CheckLaunch(fn);
}

void r4dx_inkling2_attnres_f32(const int64_t* states, int n_states, int64_t queries, int nq, int64_t out, int T,
                               int H, float eps, int64_t stream) {
  r4dx_inkling2_attnres_prenorm_f32(states, nullptr, n_states, queries, nq, out, 0, 0, T, H, eps, stream);
}

void r4dx_inkling2_row_inv_rms_f32(int64_t x, int64_t inv_out, int T, int H, float eps, int64_t stream) {
  const char* fn = "r4dx_inkling2_row_inv_rms_f32";
  if (T <= 0) return;
  if (x == 0 || inv_out == 0 || H < 1) Fail(fn, "null pointer or H < 1");
  hipLaunchKernelGGL(RowInvRmsKernel, dim3(static_cast<unsigned>(T)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const float*>(x),
                     reinterpret_cast<float*>(inv_out), H, eps);
  CheckLaunch(fn);
}

void r4dx_inkling2_sconv_static_hist_norm_f32(int64_t x, int x_bf16, int64_t weight, int64_t state, int64_t residual,
                                              int64_t out, int64_t hist, int64_t inv_out, int T, int C, float eps,
                                              int64_t stream) {
  const char* fn = "r4dx_inkling2_sconv_static_hist_norm_f32";
  if (T <= 0 || C <= 0) return;
  if (x == 0 || weight == 0 || state == 0 || out == 0) Fail(fn, "null pointer");
  if (x == out) Fail(fn, "out must not alias x");
  if (T == 1 && inv_out != 0) {
    hipLaunchKernelGGL(SconvStaticDecodeNormKernel, dim3(1), dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream),
                       reinterpret_cast<const void*>(x), x_bf16, reinterpret_cast<const float*>(weight),
                       reinterpret_cast<float*>(state), reinterpret_cast<const float*>(residual),
                       reinterpret_cast<float*>(out), reinterpret_cast<float*>(hist),
                       reinterpret_cast<float*>(inv_out), C, eps);
    CheckLaunch(fn);
    return;
  }
  r4dx_inkling2_sconv_static_hist_f32(x, x_bf16, weight, state, residual, out, hist, T, C, stream);
  if (inv_out != 0) r4dx_inkling2_row_inv_rms_f32(out, inv_out, T, C, eps, stream);
}

void r4dx_inkling2_sconv_static_hist_f32(int64_t x, int x_bf16, int64_t weight, int64_t state, int64_t residual,
                                         int64_t out, int64_t hist, int T, int C, int64_t stream) {
  const char* fn = "r4dx_inkling2_sconv_static_f32";
  if (T <= 0 || C <= 0) return;
  if (x == 0 || weight == 0 || state == 0 || out == 0) Fail(fn, "null pointer");
  if (x == out) Fail(fn, "out must not alias x");
  hipLaunchKernelGGL(SconvStaticKernel, dim3(static_cast<unsigned>((C + kThreads - 1) / kThreads)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const void*>(x), x_bf16,
                     reinterpret_cast<const float*>(weight), reinterpret_cast<float*>(state),
                     reinterpret_cast<const float*>(residual), reinterpret_cast<float*>(out),
                     reinterpret_cast<float*>(hist), T, C);
  CheckLaunch(fn);
}

void r4dx_inkling2_sconv_static_f32(int64_t x, int x_bf16, int64_t weight, int64_t state, int64_t residual,
                                    int64_t out, int T, int C, int64_t stream) {
  r4dx_inkling2_sconv_static_hist_f32(x, x_bf16, weight, state, residual, out, 0, T, C, stream);
}

void r4dx_inkling2_sconv_static_pair_hist_f32(int64_t x_k, int64_t x_v, int x_bf16, int64_t w_k, int64_t w_v,
                                              int64_t state_k, int64_t state_v, int64_t out_k, int64_t out_v,
                                              int64_t hist_k, int64_t hist_v, int T, int C, int64_t stream) {
  const char* fn = "r4dx_inkling2_sconv_static_pair_hist_f32";
  if (T <= 0 || C <= 0) return;
  if (x_k == 0 || x_v == 0 || w_k == 0 || w_v == 0 || state_k == 0 || state_v == 0 || out_k == 0 || out_v == 0)
    Fail(fn, "null pointer");
  hipLaunchKernelGGL(SconvStaticPairKernel, dim3(static_cast<unsigned>((C + kThreads - 1) / kThreads), 2u),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const void*>(x_k),
                     reinterpret_cast<const void*>(x_v), x_bf16, reinterpret_cast<const float*>(w_k),
                     reinterpret_cast<const float*>(w_v), reinterpret_cast<float*>(state_k),
                     reinterpret_cast<float*>(state_v), reinterpret_cast<float*>(out_k),
                     reinterpret_cast<float*>(out_v), reinterpret_cast<float*>(hist_k),
                     reinterpret_cast<float*>(hist_v), T, C);
  CheckLaunch(fn);
}

void r4dx_inkling2_sconv_kv_paged_bf16(int64_t x_k, int64_t x_v, int64_t w_k, int64_t w_v, int64_t state_k,
                                       int64_t state_v, int64_t hist_k, int64_t hist_v, float qk_clip, float eps,
                                       int64_t pos0, int64_t pos_dev, int64_t kv_out, int64_t block_table,
                                       int block_size, int64_t kv_block_stride, int64_t kv_head_stride, int T,
                                       int kv_heads, int64_t stream) {
  const char* fn = "r4dx_inkling2_sconv_kv_paged_bf16";
  if (T <= 0 || kv_heads <= 0) return;
  if (x_k == 0 || x_v == 0 || w_k == 0 || w_v == 0 || state_k == 0 || state_v == 0 || kv_out == 0 || block_table == 0 ||
      block_size <= 0)
    Fail(fn, "null pointer or invalid block_size");
  hipLaunchKernelGGL(SconvKvPagedBf16Kernel, dim3(static_cast<unsigned>(kv_heads), 2u), dim3(32), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(x_k),
                     reinterpret_cast<const uint16_t*>(x_v), reinterpret_cast<const float*>(w_k),
                     reinterpret_cast<const float*>(w_v), reinterpret_cast<float*>(state_k),
                     reinterpret_cast<float*>(state_v), reinterpret_cast<float*>(hist_k),
                     reinterpret_cast<float*>(hist_v), qk_clip, eps, pos0, reinterpret_cast<const int32_t*>(pos_dev),
                     reinterpret_cast<uint16_t*>(kv_out), reinterpret_cast<const int32_t*>(block_table), block_size,
                     kv_block_stride, kv_head_stride, T, kv_heads);
  CheckLaunch(fn);
}

void r4dx_inkling2_head_clip_qr(int64_t q_in, int64_t q_in_stride, int64_t q_scale, float q_thresh, int64_t q_out,
                                int q_d, int64_t r_in, int64_t r_in_stride, int64_t r_scale, float r_thresh,
                                int64_t r_out, int r_d, int64_t pos0, int64_t pos_dev, float log_alpha, float log_floor,
                                int T, int heads, int64_t stream) {
  const char* fn = "r4dx_inkling2_head_clip_qr";
  if (T <= 0 || heads <= 0) return;
  if (q_in == 0 || q_out == 0 || r_in == 0 || r_out == 0 || q_d < 1 || q_d > 128 || r_d < 1 || r_d > 128)
    Fail(fn, "null pointer or d > 128");
  const int64_t waves = static_cast<int64_t>(T) * heads;
  const unsigned bx = static_cast<unsigned>((waves + kWaves - 1) / kWaves);
  hipLaunchKernelGGL(HeadClipQRKernel, dim3(bx, 2u), dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(q_in), q_in_stride, reinterpret_cast<const float*>(q_scale),
                     q_thresh, reinterpret_cast<uint16_t*>(q_out), q_d, reinterpret_cast<const uint16_t*>(r_in),
                     r_in_stride, reinterpret_cast<const float*>(r_scale), r_thresh, reinterpret_cast<float*>(r_out), r_d,
                     pos0, reinterpret_cast<const int32_t*>(pos_dev), log_alpha, log_floor, T, heads);
  CheckLaunch(fn);
}

void r4dx_inkling2_msg_gate_prep3_bf16(int64_t x, int64_t x_gate_stride, int64_t m_all, int64_t out_cat, int T, int H,
                                       float eps, int64_t stream) {
  const char* fn = "r4dx_inkling2_msg_gate_prep3_bf16";
  if (T <= 0) return;
  if (x == 0 || m_all == 0 || out_cat == 0 || H < 1) Fail(fn, "null pointer or H < 1");
  hipLaunchKernelGGL(MsgPrep3Kernel, dim3(static_cast<unsigned>(T), 3u), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const float*>(x), x_gate_stride,
                     reinterpret_cast<const float*>(m_all), reinterpret_cast<uint16_t*>(out_cat), T, H, eps);
  CheckLaunch(fn);
}

void r4dx_inkling2_dense3_bf16(int64_t x_all, int64_t w0, int64_t w1, int64_t w2, int64_t out_all, int M, int N, int K,
                               int64_t stream) {
  const char* fn = "r4dx_inkling2_dense3_bf16";
  if (M <= 0 || N <= 0) return;
  if (x_all == 0 || w0 == 0 || w1 == 0 || w2 == 0 || out_all == 0) Fail(fn, "null pointer");
  if (M <= 4 && N >= 16) {
    Ptr3U16 w{{reinterpret_cast<const uint16_t*>(w0), reinterpret_cast<const uint16_t*>(w1),
               reinterpret_cast<const uint16_t*>(w2)}};
    hipLaunchKernelGGL(Dense3Bf16DecodeKernel, dim3(static_cast<unsigned>(N), static_cast<unsigned>(M), 3u), dim3(32),
                       0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(x_all), w,
                       reinterpret_cast<float*>(out_all), M, N, K);
    CheckLaunch(fn);
    return;
  }
  const int64_t x_step = static_cast<int64_t>(M) * K * sizeof(uint16_t);
  const int64_t o_step = static_cast<int64_t>(M) * N * sizeof(float);
  r4dx_inkling2_dense_bf16(x_all, K, w0, out_all, 0, N, M, N, K, stream);
  r4dx_inkling2_dense_bf16(x_all + x_step, K, w1, out_all + o_step, 0, N, M, N, K, stream);
  r4dx_inkling2_dense_bf16(x_all + 2 * x_step, K, w2, out_all + 2 * o_step, 0, N, M, N, K, stream);
}

void r4dx_inkling2_msg_gate_apply3_f32(int64_t x, int64_t x_gate_stride, int64_t m_all, int64_t g_all, int64_t forget0,
                                       int64_t forget1, int64_t forget2, int64_t gain0, int64_t gain1, int64_t gain2,
                                       int64_t out_all, int T, int H, int R, int64_t stream) {
  const char* fn = "r4dx_inkling2_msg_gate_apply3_f32";
  if (T <= 0) return;
  if (x == 0 || m_all == 0 || g_all == 0 || forget0 == 0 || forget1 == 0 || forget2 == 0 || gain0 == 0 || gain1 == 0 ||
      gain2 == 0 || out_all == 0)
    Fail(fn, "null pointer");
  if ((R & 3) == 0) {
    Ptr3F32 fg{{reinterpret_cast<const float*>(forget0), reinterpret_cast<const float*>(forget1),
                reinterpret_cast<const float*>(forget2)}};
    Ptr3F32 gn{{reinterpret_cast<const float*>(gain0), reinterpret_cast<const float*>(gain1),
                reinterpret_cast<const float*>(gain2)}};
    const unsigned bx = static_cast<unsigned>((H + 7) / 8);
    hipLaunchKernelGGL(MsgApply3Vec4Kernel, dim3(bx, static_cast<unsigned>(T), 3u), dim3(128), 0,
                       reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const float*>(x), x_gate_stride,
                       reinterpret_cast<const float*>(m_all), reinterpret_cast<const float*>(g_all), fg, gn,
                       reinterpret_cast<float*>(out_all), T, H, R);
    CheckLaunch(fn);
    return;
  }
  const int64_t f_step = static_cast<int64_t>(T) * H * sizeof(float);
  const int64_t g_step = static_cast<int64_t>(T) * R * sizeof(float);
  const int64_t x_step = x_gate_stride * static_cast<int64_t>(sizeof(float));
  r4dx_inkling2_msg_gate_apply_f32(x, m_all, g_all, 0, forget0, gain0, out_all, T, H, R, stream);
  r4dx_inkling2_msg_gate_apply_f32(x + x_step, m_all + f_step, g_all + g_step, 0, forget1, gain1, out_all + f_step, T,
                                   H, R, stream);
  r4dx_inkling2_msg_gate_apply_f32(x + 2 * x_step, m_all + 2 * f_step, g_all + 2 * g_step, 0, forget2, gain2,
                                   out_all + 2 * f_step, T, H, R, stream);
}

void r4dx_inkling2_rmsnorm3_f32in_bf16(int64_t x_all, int64_t w0, int64_t w1, int64_t w2, int64_t out0, int64_t out1,
                                       int64_t out2, int T, int H, float eps, int64_t stream) {
  const char* fn = "r4dx_inkling2_rmsnorm3_f32in_bf16";
  if (T <= 0) return;
  if (x_all == 0 || w0 == 0 || w1 == 0 || w2 == 0 || out0 == 0 || out1 == 0 || out2 == 0) Fail(fn, "null pointer");
  Ptr3U16 nw{{reinterpret_cast<const uint16_t*>(w0), reinterpret_cast<const uint16_t*>(w1),
              reinterpret_cast<const uint16_t*>(w2)}};
  OutPtr3U16 out{{reinterpret_cast<uint16_t*>(out0), reinterpret_cast<uint16_t*>(out1),
                  reinterpret_cast<uint16_t*>(out2)}};
  hipLaunchKernelGGL(RmsNorm3F32InBf16Kernel, dim3(static_cast<unsigned>(T), 3u), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const float*>(x_all), nw, out, T, H, eps);
  CheckLaunch(fn);
}

void r4dx_inkling2_moe_sort(int64_t ids, int T, int top_k, int n_experts, int64_t expert_counts,
                            int64_t expert_offsets, int64_t sorted_pairs, int64_t pos_of_pair, int64_t stream) {
  const char* fn = "r4dx_inkling2_moe_sort";
  if (T <= 0 || top_k <= 0) return;
  const int64_t P = static_cast<int64_t>(T) * top_k;
  if (n_experts < 1 || n_experts > kInkSortMaxExperts) Fail(fn, "n_experts must be 1..256");
  if (P > kInkSortMaxPairs) Fail(fn, "T * top_k must be <= 16384");
  hipLaunchKernelGGL(Inkling2MoeSortKernel, dim3(1), dim3(kInkSortMaxExperts), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const int32_t*>(ids), static_cast<int>(P),
                     n_experts, reinterpret_cast<int32_t*>(expert_counts), reinterpret_cast<int32_t*>(expert_offsets),
                     reinterpret_cast<int32_t*>(sorted_pairs), reinterpret_cast<int32_t*>(pos_of_pair));
  CheckLaunch(fn);
}

void r4dx_inkling2_moe_gather_bf16(int64_t x, int64_t sorted_pairs, int64_t out, int n_pairs, int top_k, int hidden,
                                   int64_t x_row_stride, int64_t valid_pairs, int64_t stream) {
  const char* fn = "r4dx_inkling2_moe_gather_bf16";
  if (n_pairs <= 0 || hidden <= 0) return;
  if (top_k <= 0) Fail(fn, "top_k must be >= 1");
  hipLaunchKernelGGL(Inkling2MoeGatherKernel, dim3(static_cast<unsigned>(n_pairs)), dim3(256), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(x),
                     reinterpret_cast<const int32_t*>(sorted_pairs), reinterpret_cast<uint16_t*>(out), n_pairs, top_k,
                     hidden, x_row_stride, reinterpret_cast<const int32_t*>(valid_pairs));
  CheckLaunch(fn);
}

void r4dx_inkling2_moe_combine_bf16(int64_t y, int64_t pos_of_pair, int64_t w, int64_t out, int T, int top_k,
                                    int hidden, int64_t stream) {
  const char* fn = "r4dx_inkling2_moe_combine_bf16";
  if (T <= 0 || hidden <= 0) return;
  if (top_k <= 0) Fail(fn, "top_k must be >= 1");
  const unsigned gy = ((hidden & 3) == 0) ? static_cast<unsigned>(((hidden >> 2) + 255) / 256)
                                          : static_cast<unsigned>((hidden + 255) / 256);
  hipLaunchKernelGGL(Inkling2MoeCombineKernel, dim3(static_cast<unsigned>(T), gy), dim3(256), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(y),
                     reinterpret_cast<const int32_t*>(pos_of_pair), reinterpret_cast<const float*>(w),
                     reinterpret_cast<uint16_t*>(out), top_k, hidden);
  CheckLaunch(fn);
}

void r4dx_inkling2_sconv_dynamic_hist_f32(int64_t x, int x_bf16, int64_t dyn, int dyn_bf16, int64_t static_w,
                                          int64_t state, int64_t out, int64_t hist, int T, int C, int heads,
                                          int64_t stream) {
  const char* fn = "r4dx_inkling2_sconv_dynamic_f32";
  if (T <= 0 || C <= 0) return;
  if (x == 0 || dyn == 0 || static_w == 0 || state == 0 || out == 0) Fail(fn, "null pointer");
  if (heads < 1 || C % heads != 0) Fail(fn, "C must be a multiple of heads");
  if (x == out) Fail(fn, "out must not alias x");
  hipLaunchKernelGGL(SconvDynamicKernel, dim3(static_cast<unsigned>((C + kThreads - 1) / kThreads)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const void*>(x), x_bf16,
                     reinterpret_cast<const void*>(dyn), dyn_bf16, reinterpret_cast<const float*>(static_w),
                     reinterpret_cast<float*>(state), reinterpret_cast<float*>(out), reinterpret_cast<float*>(hist), T,
                     C, heads);
  CheckLaunch(fn);
}

void r4dx_inkling2_sconv_dynamic_f32(int64_t x, int x_bf16, int64_t dyn, int dyn_bf16, int64_t static_w,
                                     int64_t state, int64_t out, int T, int C, int heads, int64_t stream) {
  r4dx_inkling2_sconv_dynamic_hist_f32(x, x_bf16, dyn, dyn_bf16, static_w, state, out, 0, T, C, heads, stream);
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
  if ((R & 3) == 0 && (forget & 15) == 0 && (gain & 15) == 0) {
    const unsigned bx = static_cast<unsigned>((H + 7) / 8);
    if (T >= 4) {
      hipLaunchKernelGGL(MsgApplyTiled4Kernel, dim3(bx, static_cast<unsigned>((T + 3) / 4)), dim3(128), 0,
                         reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const float*>(x),
                         reinterpret_cast<const float*>(m), reinterpret_cast<const void*>(g), g_bf16,
                         reinterpret_cast<const float*>(forget), reinterpret_cast<const float*>(gain),
                         reinterpret_cast<float*>(out), T, H, R);
    } else {
      hipLaunchKernelGGL(MsgApplyVec4Kernel, dim3(bx, static_cast<unsigned>(T)), dim3(128), 0,
                         reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const float*>(x),
                         reinterpret_cast<const float*>(m), reinterpret_cast<const void*>(g), g_bf16,
                         reinterpret_cast<const float*>(forget), reinterpret_cast<const float*>(gain),
                         reinterpret_cast<float*>(out), H, R);
    }
    CheckLaunch(fn);
    return;
  }
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
  if (logits_out != 0 && E >= 16) {
    const unsigned blocks_e = static_cast<unsigned>((E + kRouterWavesPerBlock - 1) / kRouterWavesPerBlock);
    if (T >= 8 && (H & 7) == 0 && (x_row_stride & 7) == 0 && (x & 15) == 0 && (gate_w & 15) == 0) {
      hipLaunchKernelGGL(RouterDotsTiled8Kernel, dim3(blocks_e, static_cast<unsigned>((T + 7) / 8)),
                         dim3(kRouterWavesPerBlock * 32), 0, reinterpret_cast<hipStream_t>(stream),
                         reinterpret_cast<const uint16_t*>(x), reinterpret_cast<const uint16_t*>(gate_w),
                         reinterpret_cast<float*>(logits_out), T, H, E, x_row_stride);
    } else {
      hipLaunchKernelGGL(RouterDotsKernel, dim3(blocks_e, static_cast<unsigned>(T)), dim3(kRouterWavesPerBlock * 32), 0,
                         reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(x),
                         reinterpret_cast<const uint16_t*>(gate_w), reinterpret_cast<float*>(logits_out), H, E,
                         x_row_stride);
    }
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
  if (M > 4 && (K & 7) == 0 && (x_row_stride & 7) == 0 && (x & 15) == 0 && (w & 15) == 0) {
    hipLaunchKernelGGL(DenseBf16Tiled4Kernel,
                       dim3(static_cast<unsigned>((N + 3) / 4), static_cast<unsigned>((M + 3) / 4)), dim3(128), 0,
                       reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(x), x_row_stride,
                       reinterpret_cast<const uint16_t*>(w), reinterpret_cast<void*>(out), out_bf16, out_row_stride, M,
                       N, K);
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
    if (a->q_len == 1 && (a->max_splits < 1 || (a->max_splits > 1 && a->workspace == 0))) return -7;
  } else {
    if (a->ctx < a->q_len) return -5;
  }
  if (a->window < 0) return -5;
  if (a->r != 0 && (a->rel_proj == 0 || a->d_rel <= 0 || a->d_rel > kFusedMaxDrel || a->rel_extent <= 0)) return -6;
  hipStream_t st = reinterpret_cast<hipStream_t>(stream);
  if (a->q_len == 1 || (a->q_len <= 16 && a->workspace != 0)) {
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
    const dim3 grid(static_cast<unsigned>(a->kv_heads), static_cast<unsigned>(n_splits),
                    static_cast<unsigned>(a->q_len));
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
      hipLaunchKernelGGL(Inkling2AttnDecodeCombineKernel,
                         dim3(static_cast<unsigned>(a->q_heads), static_cast<unsigned>(a->q_len)), dim3(32), 0, st,
                         reinterpret_cast<const float*>(a->workspace), reinterpret_cast<const float*>(a->sink),
                         reinterpret_cast<const uint16_t*>(a->gate), reinterpret_cast<uint16_t*>(a->out), a->q_heads,
                         n_splits);
      e = hipGetLastError();
      if (e != hipSuccess) return -100 - static_cast<int>(e);
      r4dx_kernel_launch_counter_add(1);
    }
    return 0;
  }
  if (a->q_len >= 16 && (a->r == 0 || a->d_rel <= 16)) {
    const dim3 wgrid(static_cast<unsigned>(a->q_heads), static_cast<unsigned>((a->q_len + 63) / 64));
    hipLaunchKernelGGL(Inkling2AttnPrefillWmmaKernel, wgrid, dim3(128), 0, st, *a);
    const hipError_t e = hipGetLastError();
    if (e != hipSuccess) return -100 - static_cast<int>(e);
    r4dx_kernel_launch_counter_add(1);
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

namespace {

// ---- DSpark / MTP drafter kernels ----------------------------------------------------------------------------------
__global__ __launch_bounds__(kThreads) void DsparkMixRmsNormKernel(
    const uint16_t* __restrict__ t0, const uint16_t* __restrict__ t1, const uint16_t* __restrict__ t2,
    const uint16_t* __restrict__ t3, float a0, float a1, float a2, float a3, const uint16_t* __restrict__ w,
    uint16_t* __restrict__ out, int H, float eps) {
  __shared__ float red[kWaves];
  const int64_t t = blockIdx.x;
  const int64_t off = t * H;
  float ss = 0.0f;
  for (int h = threadIdx.x; h < H; h += kThreads) {
    const float m = a0 * r4dx::core::Bf16ToFloat(t0[off + h]) + a1 * r4dx::core::Bf16ToFloat(t1[off + h]) +
                    a2 * r4dx::core::Bf16ToFloat(t2[off + h]) + a3 * r4dx::core::Bf16ToFloat(t3[off + h]);
    ss += m * m;
  }
  ss = BlockSum(ss, red);
  const float inv = rsqrtf(ss / static_cast<float>(H) + eps);
  uint16_t* dst = out + off;
  for (int h = threadIdx.x; h < H; h += kThreads) {
    const float m = a0 * r4dx::core::Bf16ToFloat(t0[off + h]) + a1 * r4dx::core::Bf16ToFloat(t1[off + h]) +
                    a2 * r4dx::core::Bf16ToFloat(t2[off + h]) + a3 * r4dx::core::Bf16ToFloat(t3[off + h]);
    dst[h] = r4dx::core::FloatToBf16(m * inv * r4dx::core::Bf16ToFloat(w[h]));
  }
}

__global__ __launch_bounds__(64) void DsparkHeadNormRopeKernel(
    uint16_t* __restrict__ x, int64_t row_stride, const uint16_t* __restrict__ w, int heads, int head_dim,
    int64_t pos0, const int64_t* __restrict__ pos_dev, float rope_theta, float eps) {
  const int t = static_cast<int>(blockIdx.y);
  const int h = static_cast<int>(blockIdx.x);
  const int lane = threadIdx.x;  // 0..head_dim/2 - 1 (64 threads for head_dim=128)
  const int half = head_dim >> 1;
  if (lane >= half) return;
  const int64_t p0 = (pos_dev != nullptr) ? *pos_dev : pos0;
  uint16_t* row = x + static_cast<int64_t>(t) * row_stride + static_cast<int64_t>(h) * head_dim;
  float x1 = r4dx::core::Bf16ToFloat(row[lane]);
  float x2 = r4dx::core::Bf16ToFloat(row[half + lane]);
  if (w != nullptr) {
    __shared__ float sm[2];
    const float sq = x1 * x1 + x2 * x2;
    float ws = WaveSum(sq);
    if ((lane & 31) == 0) sm[lane >> 5] = ws;
    __syncthreads();
    const float total = sm[0] + (half > 32 ? sm[1] : 0.0f);
    const float inv = 1.0f / sqrtf(total / static_cast<float>(head_dim) + eps);
    x1 = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(x1 * inv * r4dx::core::Bf16ToFloat(w[lane])));
    x2 = r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(x2 * inv * r4dx::core::Bf16ToFloat(w[half + lane])));
  }
  const float pos = static_cast<float>(p0 + t);
  const float inv_freq = powf(rope_theta, -2.0f * static_cast<float>(lane) / static_cast<float>(head_dim));
  const float angle = pos * inv_freq;
  float s, c;
  sincosf(angle, &s, &c);
  row[lane] = r4dx::core::FloatToBf16(x1 * c - x2 * s);
  row[half + lane] = r4dx::core::FloatToBf16(x2 * c + x1 * s);
}

__device__ __forceinline__ float RoundBf16F(float x) { return r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(x)); }

__global__ __launch_bounds__(256) void DsparkMarkovGateKernel(
    const uint16_t* __restrict__ raw, const uint16_t* __restrict__ bias, const uint16_t* __restrict__ state_in,
    uint16_t* __restrict__ state_out, uint16_t* __restrict__ latent_out, int R) {
  const int c = blockIdx.x * 256 + threadIdx.x;
  if (c >= R) return;
  float g_in = r4dx::core::Bf16ToFloat(raw[c]);
  float cand_in = r4dx::core::Bf16ToFloat(raw[R + c]);
  float out_in = r4dx::core::Bf16ToFloat(raw[2 * R + c]);
  if (bias != nullptr) {
    g_in = RoundBf16F(g_in + r4dx::core::Bf16ToFloat(bias[c]));
    cand_in = RoundBf16F(cand_in + r4dx::core::Bf16ToFloat(bias[R + c]));
    out_in = RoundBf16F(out_in + r4dx::core::Bf16ToFloat(bias[2 * R + c]));
  }
  const float st = r4dx::core::Bf16ToFloat(state_in[c]);
  const float gate = RoundBf16F(1.0f / (1.0f + expf(-g_in)));
  const float cand = RoundBf16F(tanhf(cand_in));
  const float keep = RoundBf16F(gate * st);
  const float upd = RoundBf16F(RoundBf16F(1.0f - gate) * cand);
  state_out[c] = r4dx::core::FloatToBf16(keep + upd);
  latent_out[c] = r4dx::core::FloatToBf16(tanhf(out_in));
}

__global__ __launch_bounds__(kThreads) void DsparkAddSoftcapKernel(
    const uint16_t* __restrict__ base, const uint16_t* __restrict__ delta, const uint16_t* __restrict__ bias,
    float* __restrict__ out, int64_t n, float softcap) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (i >= n) return;
  float x = r4dx::core::Bf16ToFloat(base[i]);
  if (delta != nullptr) x += r4dx::core::Bf16ToFloat(delta[i]);
  if (bias != nullptr) x += r4dx::core::Bf16ToFloat(bias[i]);
  out[i] = softcap * tanhf(x / softcap);
}

constexpr int kDsparkArgmaxBlocks = 256;

__global__ __launch_bounds__(kThreads) void DsparkAddArgmaxPartKernel(
    const uint16_t* __restrict__ base, const uint16_t* __restrict__ delta, const uint16_t* __restrict__ bias,
    float* __restrict__ part_val, int32_t* __restrict__ part_idx, int64_t n) {
  float best_v = -INFINITY;
  int32_t best_i = 0x7fffffff;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * kThreads;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x; i < n; i += stride) {
    float x = r4dx::core::Bf16ToFloat(base[i]);
    if (delta != nullptr) x += r4dx::core::Bf16ToFloat(delta[i]);
    if (bias != nullptr) x += r4dx::core::Bf16ToFloat(bias[i]);
    const int32_t idx = static_cast<int32_t>(i);
    if (x > best_v || (x == best_v && idx < best_i)) {
      best_v = x;
      best_i = idx;
    }
  }
  __shared__ float s_val[kThreads];
  __shared__ int32_t s_idx[kThreads];
  s_val[threadIdx.x] = best_v;
  s_idx[threadIdx.x] = best_i;
  __syncthreads();
  for (int off = kThreads / 2; off > 0; off >>= 1) {
    if (threadIdx.x < off) {
      const float ov = s_val[threadIdx.x + off];
      const int32_t oi = s_idx[threadIdx.x + off];
      const float mv = s_val[threadIdx.x];
      const int32_t mi = s_idx[threadIdx.x];
      if (ov > mv || (ov == mv && oi < mi)) {
        s_val[threadIdx.x] = ov;
        s_idx[threadIdx.x] = oi;
      }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    part_val[blockIdx.x] = s_val[0];
    part_idx[blockIdx.x] = s_idx[0];
  }
}

__global__ __launch_bounds__(kThreads) void DsparkAddArgmaxFinalKernel(
    const float* __restrict__ part_val, const int32_t* __restrict__ part_idx, int32_t* __restrict__ out_idx,
    float* __restrict__ out_val, int nblocks) {
  __shared__ float s_val[kThreads];
  __shared__ int32_t s_idx[kThreads];
  const int tid = threadIdx.x;
  s_val[tid] = (tid < nblocks) ? part_val[tid] : -INFINITY;
  s_idx[tid] = (tid < nblocks) ? part_idx[tid] : 0x7fffffff;
  __syncthreads();
  for (int off = kThreads / 2; off > 0; off >>= 1) {
    if (tid < off) {
      const float ov = s_val[tid + off];
      const int32_t oi = s_idx[tid + off];
      const float mv = s_val[tid];
      const int32_t mi = s_idx[tid];
      if (ov > mv || (ov == mv && oi < mi)) {
        s_val[tid] = ov;
        s_idx[tid] = oi;
      }
    }
    __syncthreads();
  }
  if (tid == 0) {
    out_idx[0] = s_idx[0];
    if (out_val != nullptr) out_val[0] = s_val[0];
  }
}

__global__ __launch_bounds__(kThreads) void DsparkMarkovPrepKernel(
    uint16_t* __restrict__ rnn_in, const uint16_t* __restrict__ prev_emb, const uint16_t* __restrict__ head_h,
    const uint16_t* __restrict__ f_rnn, int zero_state, int R, int H) {
  const int i = blockIdx.x * kThreads + threadIdx.x;
  const int total = 2 * R + 2 * H;
  if (i >= total) return;
  if (i < R) {
    if (zero_state != 0) rnn_in[i] = 0;
  } else if (i < 2 * R) {
    if (prev_emb != nullptr) rnn_in[i] = prev_emb[i - R];
  } else if (i < 2 * R + H) {
    if (head_h != nullptr) rnn_in[i] = head_h[i - 2 * R];
  } else {
    if (f_rnn != nullptr) rnn_in[i] = f_rnn[i - 2 * R - H];
  }
}

__global__ __launch_bounds__(kThreads) void DsparkTransposeBf16Kernel(
    const uint16_t* __restrict__ in, uint16_t* __restrict__ out, int64_t rows, int64_t cols) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  const int64_t total = rows * cols;
  if (idx >= total) return;
  const int64_t r = idx / cols;
  const int64_t c = idx - r * cols;
  out[c * rows + r] = in[idx];
}

__global__ __launch_bounds__(kThreads) void DsparkKvWriteKernel(
    const uint16_t* __restrict__ k_in, const uint16_t* __restrict__ v_in, uint16_t* __restrict__ k_store,
    uint16_t* __restrict__ v_store, int n, int kv_heads, int head_dim, int64_t pos0, int slots) {
  const int64_t elems_per_row = static_cast<int64_t>(kv_heads) * head_dim;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  const int64_t total = static_cast<int64_t>(n) * elems_per_row;
  if (idx >= total) return;
  const int64_t t = idx / elems_per_row;
  const int64_t rem = idx - t * elems_per_row;
  const int64_t slot = (pos0 + t) % slots;
  k_store[slot * elems_per_row + rem] = k_in[idx];
  v_store[slot * elems_per_row + rem] = v_in[idx];
}

__global__ __launch_bounds__(kThreads) void DsparkKvWriteFp8Kernel(
    const uint16_t* __restrict__ k_in, const uint16_t* __restrict__ v_in, uint8_t* __restrict__ k_store,
    uint8_t* __restrict__ v_store, int n, int kv_heads, int head_dim, int64_t pos0, int slots) {
  const int64_t elems_per_row = static_cast<int64_t>(kv_heads) * head_dim;
  const int64_t total = static_cast<int64_t>(n) * elems_per_row;
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (idx >= total) return;
  const int64_t r = idx / elems_per_row;
  const int64_t rem = idx - r * elems_per_row;
  const int64_t slot = (pos0 + r) % slots;
  k_store[slot * elems_per_row + rem] = FloatToFp8E4m3Fn(r4dx::core::Bf16ToFloat(k_in[idx]));
  v_store[slot * elems_per_row + rem] = FloatToFp8E4m3Fn(r4dx::core::Bf16ToFloat(v_in[idx]));
}

// Bidirectional block attention for DSpark draft (head_dim == 128): grid = (q_heads, T), block = 128 (4 waves).
__global__ __launch_bounds__(128) void DsparkAttnKernel(
    const uint16_t* __restrict__ q_ptr, const uint16_t* __restrict__ kb_ptr, const uint16_t* __restrict__ vb_ptr,
    const void* __restrict__ ks_ptr, const void* __restrict__ vs_ptr, uint16_t* __restrict__ out_ptr,
    int T, int q_heads, int kv_heads, int n_injected, const int64_t* __restrict__ pos_dev, int window, int slots,
    float scale, int kv_fp8) {
  const int n_inj = (pos_dev != nullptr) ? static_cast<int>(*pos_dev) : n_injected;
  __shared__ float sm[4];
  __shared__ float sl[4];
  __shared__ float so[4][128];
  const int qh = static_cast<int>(blockIdx.x);
  const int t = static_cast<int>(blockIdx.y);
  const int wave = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int gqa = q_heads / kv_heads;
  const int kvh = qh / gqa;
  const int64_t kv_row_stride = static_cast<int64_t>(kv_heads) * 128;

  const uint2 qv = *reinterpret_cast<const uint2*>(q_ptr + (static_cast<int64_t>(t) * q_heads + qh) * 128 + lane * 4);
  float q0, q1, q2, q3;
  Unpack4Bf16(qv, q0, q1, q2, q3);

  float mv = -INFINITY, lv = 0.0f;
  float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;

  const int k_begin = (window > 0 && n_inj > window) ? (n_inj - window) : 0;
  const int ctx_len = n_inj - k_begin;
  const int total_k = ctx_len + T;

  for (int idx = wave; idx < total_k; idx += 4) {
    float k0, k1, k2, k3, v0, v1, v2, v3;
    if (idx < ctx_len) {
      const int64_t slot = (k_begin + idx) % slots;
      const int64_t off = slot * kv_row_stride + static_cast<int64_t>(kvh) * 128;
      if (kv_fp8) {
        Unpack4Fp8(*reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(ks_ptr) + off + lane * 4),
                   k0, k1, k2, k3);
        Unpack4Fp8(*reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(vs_ptr) + off + lane * 4),
                   v0, v1, v2, v3);
      } else {
        Unpack4Bf16(*reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(ks_ptr) + off + lane * 4),
                    k0, k1, k2, k3);
        Unpack4Bf16(*reinterpret_cast<const uint2*>(reinterpret_cast<const uint16_t*>(vs_ptr) + off + lane * 4),
                    v0, v1, v2, v3);
      }
    } else {
      const int64_t off = static_cast<int64_t>(idx - ctx_len) * kv_row_stride + static_cast<int64_t>(kvh) * 128;
      Unpack4Bf16(*reinterpret_cast<const uint2*>(kb_ptr + off + lane * 4), k0, k1, k2, k3);
      Unpack4Bf16(*reinterpret_cast<const uint2*>(vb_ptr + off + lane * 4), v0, v1, v2, v3);
    }
    float dot = (q0 * k0 + q1 * k1 + q2 * k2 + q3 * k3);
    dot = WaveSum(dot) * scale;
    const float mn = fmaxf(mv, dot);
    const float alpha = (mv == -INFINITY) ? 0.0f : expf(mv - mn);
    const float p = expf(dot - mn);
    lv = lv * alpha + p;
    o0 = o0 * alpha + p * v0;
    o1 = o1 * alpha + p * v1;
    o2 = o2 * alpha + p * v2;
    o3 = o3 * alpha + p * v3;
    mv = mn;
  }

  if (lane == 0) {
    sm[wave] = mv;
    sl[wave] = lv;
  }
  so[wave][lane * 4 + 0] = o0;
  so[wave][lane * 4 + 1] = o1;
  so[wave][lane * 4 + 2] = o2;
  so[wave][lane * 4 + 3] = o3;
  __syncthreads();

  if (wave == 0) {
    float m_all = fmaxf(fmaxf(sm[0], sm[1]), fmaxf(sm[2], sm[3]));
    float l_all = 0.0f, r0 = 0.0f, r1 = 0.0f, r2 = 0.0f, r3 = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
      if (sl[w] > 0.0f) {
        const float sc = expf(sm[w] - m_all);
        l_all += sl[w] * sc;
        r0 += so[w][lane * 4 + 0] * sc;
        r1 += so[w][lane * 4 + 1] * sc;
        r2 += so[w][lane * 4 + 2] * sc;
        r3 += so[w][lane * 4 + 3] * sc;
      }
    }
    const float inv = (l_all > 0.0f) ? (1.0f / l_all) : 0.0f;
    *reinterpret_cast<uint2*>(out_ptr + (static_cast<int64_t>(t) * q_heads + qh) * 128 + lane * 4) =
        Pack4Bf16(r0 * inv, r1 * inv, r2 * inv, r3 * inv);
  }
}

}  // namespace

extern "C" {

void r4dx_inkling2_dspark_mix_rmsnorm_bf16(int64_t tap0, int64_t tap1, int64_t tap2, int64_t tap3,
                                           float a0, float a1, float a2, float a3, int64_t weight,
                                           int64_t out, int T, int H, float eps, int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_mix_rmsnorm_bf16";
  if (T <= 0) return;
  if (tap0 == 0 || tap1 == 0 || tap2 == 0 || tap3 == 0 || weight == 0 || out == 0 || H < 1)
    Fail(fn, "null pointer or H < 1");
  hipLaunchKernelGGL(DsparkMixRmsNormKernel, dim3(static_cast<unsigned>(T)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(tap0),
                     reinterpret_cast<const uint16_t*>(tap1), reinterpret_cast<const uint16_t*>(tap2),
                     reinterpret_cast<const uint16_t*>(tap3), a0, a1, a2, a3,
                     reinterpret_cast<const uint16_t*>(weight), reinterpret_cast<uint16_t*>(out), H, eps);
  CheckLaunch(fn);
}

void r4dx_inkling2_dspark_head_norm_rope_bf16(int64_t x, int64_t x_row_stride, int64_t weight, int T, int heads,
                                              int head_dim, int64_t pos0, float rope_theta, float eps, int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_head_norm_rope_bf16";
  if (T <= 0 || heads <= 0) return;
  if (x == 0 || head_dim != 128) Fail(fn, "null x or head_dim != 128");
  hipLaunchKernelGGL(DsparkHeadNormRopeKernel, dim3(static_cast<unsigned>(heads), static_cast<unsigned>(T)), dim3(64),
                     0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<uint16_t*>(x), x_row_stride,
                     reinterpret_cast<const uint16_t*>(weight), heads, head_dim, pos0,
                     static_cast<const int64_t*>(nullptr), rope_theta, eps);
  CheckLaunch(fn);
}

void r4dx_inkling2_dspark_head_norm_rope_posdev_bf16(int64_t x, int64_t x_row_stride, int64_t weight, int T, int heads,
                                                     int head_dim, int64_t pos_dev, float rope_theta, float eps,
                                                     int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_head_norm_rope_posdev_bf16";
  if (T <= 0 || heads <= 0) return;
  if (x == 0 || pos_dev == 0 || head_dim != 128) Fail(fn, "null x/pos_dev or head_dim != 128");
  hipLaunchKernelGGL(DsparkHeadNormRopeKernel, dim3(static_cast<unsigned>(heads), static_cast<unsigned>(T)), dim3(64),
                     0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<uint16_t*>(x), x_row_stride,
                     reinterpret_cast<const uint16_t*>(weight), heads, head_dim, int64_t{0},
                     reinterpret_cast<const int64_t*>(pos_dev), rope_theta, eps);
  CheckLaunch(fn);
}

void r4dx_inkling2_dspark_markov_gate_bf16(int64_t raw, int64_t bias, int64_t state_in, int64_t state_out,
                                           int64_t latent_out, int R, int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_markov_gate_bf16";
  if (raw == 0 || state_in == 0 || state_out == 0 || latent_out == 0 || R < 1) Fail(fn, "null pointer or R < 1");
  hipLaunchKernelGGL(DsparkMarkovGateKernel, dim3(static_cast<unsigned>((R + 255) / 256)), dim3(256), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(raw),
                     reinterpret_cast<const uint16_t*>(bias), reinterpret_cast<const uint16_t*>(state_in),
                     reinterpret_cast<uint16_t*>(state_out), reinterpret_cast<uint16_t*>(latent_out), R);
  CheckLaunch(fn);
}

void r4dx_inkling2_dspark_add_softcap_f32(int64_t base, int64_t delta, int64_t bias, int64_t out, int64_t n,
                                          float softcap, int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_add_softcap_f32";
  if (n <= 0) return;
  if (base == 0 || out == 0) Fail(fn, "null pointer");
  hipLaunchKernelGGL(DsparkAddSoftcapKernel, dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads)), dim3(kThreads),
                     0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(base),
                     reinterpret_cast<const uint16_t*>(delta), reinterpret_cast<const uint16_t*>(bias),
                     reinterpret_cast<float*>(out), n, softcap);
  CheckLaunch(fn);
}

void r4dx_inkling2_dspark_add_argmax_bf16(int64_t base, int64_t delta, int64_t bias, int64_t ws, int64_t out_idx,
                                          int64_t out_val, int64_t n, int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_add_argmax_bf16";
  if (n <= 0) return;
  if (base == 0 || ws == 0 || out_idx == 0) Fail(fn, "null pointer");
  const int nblocks =
      static_cast<int>(std::min<int64_t>(kDsparkArgmaxBlocks, (n + kThreads - 1) / kThreads));
  float* part_val = reinterpret_cast<float*>(ws);
  int32_t* part_idx = reinterpret_cast<int32_t*>(ws + kDsparkArgmaxBlocks * static_cast<int64_t>(sizeof(float)));
  hipStream_t sr = reinterpret_cast<hipStream_t>(stream);
  hipLaunchKernelGGL(DsparkAddArgmaxPartKernel, dim3(static_cast<unsigned>(nblocks)), dim3(kThreads), 0, sr,
                     reinterpret_cast<const uint16_t*>(base), reinterpret_cast<const uint16_t*>(delta),
                     reinterpret_cast<const uint16_t*>(bias), part_val, part_idx, n);
  CheckLaunch(fn);
  hipLaunchKernelGGL(DsparkAddArgmaxFinalKernel, dim3(1), dim3(kThreads), 0, sr, part_val, part_idx,
                     reinterpret_cast<int32_t*>(out_idx), reinterpret_cast<float*>(out_val), nblocks);
  CheckLaunch(fn);
}

void r4dx_inkling2_dspark_markov_prep_bf16(int64_t rnn_in, int64_t prev_emb, int64_t head_h, int64_t f_rnn,
                                           int zero_state, int R, int H, int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_markov_prep_bf16";
  if (rnn_in == 0 || R < 1 || H < 1) Fail(fn, "invalid args");
  const int total = 2 * R + 2 * H;
  hipLaunchKernelGGL(DsparkMarkovPrepKernel, dim3(static_cast<unsigned>((total + kThreads - 1) / kThreads)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<uint16_t*>(rnn_in),
                     reinterpret_cast<const uint16_t*>(prev_emb), reinterpret_cast<const uint16_t*>(head_h),
                     reinterpret_cast<const uint16_t*>(f_rnn), zero_state, R, H);
  CheckLaunch(fn);
}

void r4dx_inkling2_dspark_transpose_bf16(int64_t in, int64_t out, int64_t rows, int64_t cols, int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_transpose_bf16";
  if (rows <= 0 || cols <= 0) return;
  if (in == 0 || out == 0) Fail(fn, "null pointer");
  const int64_t total = rows * cols;
  hipLaunchKernelGGL(DsparkTransposeBf16Kernel, dim3(static_cast<unsigned>((total + kThreads - 1) / kThreads)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(in),
                     reinterpret_cast<uint16_t*>(out), rows, cols);
  CheckLaunch(fn);
}

void r4dx_inkling2_dspark_kv_write_bf16(int64_t k_in, int64_t v_in, int64_t k_store, int64_t v_store, int n,
                                        int kv_heads, int head_dim, int64_t pos0, int slots, int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_kv_write_bf16";
  if (n <= 0) return;
  if (k_in == 0 || v_in == 0 || k_store == 0 || v_store == 0 || slots <= 0) Fail(fn, "null pointer or slots <= 0");
  const int64_t total = static_cast<int64_t>(n) * kv_heads * head_dim;
  hipLaunchKernelGGL(DsparkKvWriteKernel, dim3(static_cast<unsigned>((total + kThreads - 1) / kThreads)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(k_in),
                     reinterpret_cast<const uint16_t*>(v_in), reinterpret_cast<uint16_t*>(k_store),
                     reinterpret_cast<uint16_t*>(v_store), n, kv_heads, head_dim, pos0, slots);
  CheckLaunch(fn);
}

void r4dx_inkling2_dspark_attn_bf16(int64_t q, int64_t k_block, int64_t v_block, int64_t k_store, int64_t v_store,
                                    int64_t out, int T, int q_heads, int kv_heads, int head_dim, int n_injected,
                                    int window, int slots, float scale, int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_attn_bf16";
  if (T <= 0) return;
  if (q == 0 || k_block == 0 || v_block == 0 || out == 0 || head_dim != 128 || kv_heads <= 0 ||
      q_heads % kv_heads != 0) {
    Fail(fn, "invalid args");
  }
  hipLaunchKernelGGL(DsparkAttnKernel, dim3(static_cast<unsigned>(q_heads), static_cast<unsigned>(T)), dim3(128), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(q),
                     reinterpret_cast<const uint16_t*>(k_block), reinterpret_cast<const uint16_t*>(v_block),
                     reinterpret_cast<const void*>(k_store), reinterpret_cast<const void*>(v_store),
                     reinterpret_cast<uint16_t*>(out), T, q_heads, kv_heads, n_injected,
                     static_cast<const int64_t*>(nullptr), window, slots, scale, 0);
  CheckLaunch(fn);
}

void r4dx_inkling2_dspark_attn_posdev_bf16(int64_t q, int64_t k_block, int64_t v_block, int64_t k_store, int64_t v_store,
                                           int64_t out, int T, int q_heads, int kv_heads, int head_dim, int64_t pos_dev,
                                           int window, int slots, float scale, int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_attn_posdev_bf16";
  if (T <= 0) return;
  if (q == 0 || k_block == 0 || v_block == 0 || out == 0 || pos_dev == 0 || head_dim != 128 || kv_heads <= 0 ||
      q_heads % kv_heads != 0) {
    Fail(fn, "invalid args");
  }
  hipLaunchKernelGGL(DsparkAttnKernel, dim3(static_cast<unsigned>(q_heads), static_cast<unsigned>(T)), dim3(128), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(q),
                     reinterpret_cast<const uint16_t*>(k_block), reinterpret_cast<const uint16_t*>(v_block),
                     reinterpret_cast<const void*>(k_store), reinterpret_cast<const void*>(v_store),
                     reinterpret_cast<uint16_t*>(out), T, q_heads, kv_heads, 0,
                     reinterpret_cast<const int64_t*>(pos_dev), window, slots, scale, 0);
  CheckLaunch(fn);
}

void r4dx_inkling2_dspark_kv_write_fp8(int64_t k_in, int64_t v_in, int64_t k_store_fp8, int64_t v_store_fp8, int n,
                                       int kv_heads, int head_dim, int64_t pos0, int slots, int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_kv_write_fp8";
  if (n <= 0) return;
  if (k_in == 0 || v_in == 0 || k_store_fp8 == 0 || v_store_fp8 == 0 || slots <= 0) Fail(fn, "null pointer or slots <= 0");
  const int64_t total = static_cast<int64_t>(n) * kv_heads * head_dim;
  hipLaunchKernelGGL(DsparkKvWriteFp8Kernel, dim3(static_cast<unsigned>((total + kThreads - 1) / kThreads)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(k_in),
                     reinterpret_cast<const uint16_t*>(v_in), reinterpret_cast<uint8_t*>(k_store_fp8),
                     reinterpret_cast<uint8_t*>(v_store_fp8), n, kv_heads, head_dim, pos0, slots);
  CheckLaunch(fn);
}

void r4dx_inkling2_dspark_attn_fp8kv(int64_t q, int64_t k_block, int64_t v_block, int64_t k_store_fp8,
                                     int64_t v_store_fp8, int64_t out, int T, int q_heads, int kv_heads, int head_dim,
                                     int n_injected, int64_t pos_dev, int window, int slots, float scale,
                                     int64_t stream) {
  const char* fn = "r4dx_inkling2_dspark_attn_fp8kv";
  if (T <= 0) return;
  if (q == 0 || k_block == 0 || v_block == 0 || out == 0 || head_dim != 128 || kv_heads <= 0 ||
      q_heads % kv_heads != 0) {
    Fail(fn, "invalid args");
  }
  hipLaunchKernelGGL(DsparkAttnKernel, dim3(static_cast<unsigned>(q_heads), static_cast<unsigned>(T)), dim3(128), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(q),
                     reinterpret_cast<const uint16_t*>(k_block), reinterpret_cast<const uint16_t*>(v_block),
                     reinterpret_cast<const void*>(k_store_fp8), reinterpret_cast<const void*>(v_store_fp8),
                     reinterpret_cast<uint16_t*>(out), T, q_heads, kv_heads, n_injected,
                     reinterpret_cast<const int64_t*>(pos_dev), window, slots, scale, 1);
  CheckLaunch(fn);
}

}  // extern "C"

namespace {

__global__ __launch_bounds__(kThreads) void SconvCaptureXKernel(const void* __restrict__ x, int x_bf16,
                                                                float* __restrict__ hist, int64_t total) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (i >= total) return;
  hist[i] = LoadF(x, i, x_bf16);
}

__global__ __launch_bounds__(kThreads) void SconvRollbackKernel(const float* __restrict__ bak,
                                                                const float* __restrict__ hist,
                                                                float* __restrict__ out, int n_keep, int C) {
  const int c = blockIdx.x * kThreads + threadIdx.x;
  if (c >= C) return;
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int idx = n_keep + k;
    out[static_cast<int64_t>(k) * C + c] =
        (idx < 3) ? bak[static_cast<int64_t>(idx) * C + c] : hist[static_cast<int64_t>(idx - 3) * C + c];
  }
}

__global__ __launch_bounds__(kThreads) void SconvRollbackBatchKernel(
    const float* __restrict__ state_bak, const float* __restrict__ hist, float* __restrict__ state_out,
    const int64_t* __restrict__ conv_off, const int64_t* __restrict__ conv_hist_off,
    const int32_t* __restrict__ conv_ch, int n_keep) {
  const int ci = blockIdx.y;
  const int C = conv_ch[ci];
  const int c = blockIdx.x * kThreads + threadIdx.x;
  if (c >= C) return;
  const int64_t off = conv_off[ci];
  const int64_t hoff = conv_hist_off[ci];
  const float* bak = state_bak + off;
  const float* h = hist + hoff;
  float* out = state_out + off;
#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int idx = n_keep + k;
    out[static_cast<int64_t>(k) * C + c] =
        (idx < 3) ? bak[static_cast<int64_t>(idx) * C + c] : h[static_cast<int64_t>(idx - 3) * C + c];
  }
}

__global__ __launch_bounds__(kThreads) void AddBf16Kernel(const uint16_t* __restrict__ a,
                                                          const uint16_t* __restrict__ b, uint16_t* __restrict__ out,
                                                          int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (i >= n) return;
  out[i] = r4dx::core::FloatToBf16(r4dx::core::Bf16ToFloat(a[i]) + r4dx::core::Bf16ToFloat(b[i]));
}

__global__ __launch_bounds__(kThreads) void AddF32Bf16Kernel(const float* __restrict__ a,
                                                             const uint16_t* __restrict__ b, float* __restrict__ out,
                                                             int64_t n) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (i >= n) return;
  out[i] = a[i] + r4dx::core::Bf16ToFloat(b[i]);
}

__global__ __launch_bounds__(kThreads) void QgrkvSplitBf16Kernel(
    const uint16_t* __restrict__ src, int64_t total_n, uint16_t* __restrict__ q_out, int64_t q_off, int64_t q_n,
    uint16_t* __restrict__ g_out, int64_t g_off, int64_t g_n, uint16_t* __restrict__ r_out, int64_t r_off, int64_t r_n,
    uint16_t* __restrict__ k_out, int64_t k_off, int64_t k_n, uint16_t* __restrict__ v_out, int64_t v_off, int64_t v_n,
    int T) {
  const int64_t t = static_cast<int64_t>(blockIdx.y);
  const int64_t c = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (t >= T) return;
  const uint16_t* row = src + t * total_n;
  if (c < q_n) q_out[t * q_n + c] = row[q_off + c];
  if (c < g_n) g_out[t * g_n + c] = row[g_off + c];
  if (c < r_n) r_out[t * r_n + c] = row[r_off + c];
  if (k_out != nullptr && c < k_n) k_out[t * k_n + c] = row[k_off + c];
  if (v_out != nullptr && c < v_n) v_out[t * v_n + c] = row[v_off + c];
}

}  // namespace

extern "C" {

void r4dx_inkling2_sconv_capture_x_f32(int64_t x, int x_bf16, int64_t hist, int T, int C, int64_t stream) {
  const char* fn = "r4dx_inkling2_sconv_capture_x_f32";
  if (T <= 0 || C <= 0) return;
  if (x == 0 || hist == 0) Fail(fn, "null pointer");
  const int64_t total = static_cast<int64_t>(T) * C;
  hipLaunchKernelGGL(SconvCaptureXKernel, dim3(static_cast<unsigned>((total + kThreads - 1) / kThreads)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const void*>(x), x_bf16,
                     reinterpret_cast<float*>(hist), total);
  CheckLaunch(fn);
}

void r4dx_inkling2_sconv_rollback_f32(int64_t state_bak, int64_t hist, int64_t state_out, int n_keep, int C,
                                      int64_t stream) {
  const char* fn = "r4dx_inkling2_sconv_rollback_f32";
  if (C <= 0) return;
  if (state_bak == 0 || state_out == 0 || (n_keep > 0 && hist == 0)) Fail(fn, "null pointer");
  hipLaunchKernelGGL(SconvRollbackKernel, dim3(static_cast<unsigned>((C + kThreads - 1) / kThreads)), dim3(kThreads),
                     0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const float*>(state_bak),
                     reinterpret_cast<const float*>(hist), reinterpret_cast<float*>(state_out), n_keep, C);
  CheckLaunch(fn);
}

void r4dx_inkling2_sconv_rollback_batch_f32(int64_t state_bak, int64_t hist, int64_t state_out, int64_t conv_off,
                                            int64_t conv_hist_off, int64_t conv_ch, int n_convs, int max_C, int n_keep,
                                            int64_t stream) {
  const char* fn = "r4dx_inkling2_sconv_rollback_batch_f32";
  if (n_convs <= 0 || max_C <= 0) return;
  if (state_bak == 0 || state_out == 0 || (n_keep > 0 && hist == 0) || conv_off == 0 || conv_hist_off == 0 ||
      conv_ch == 0)
    Fail(fn, "null pointer");
  hipLaunchKernelGGL(SconvRollbackBatchKernel,
                     dim3(static_cast<unsigned>((max_C + kThreads - 1) / kThreads), static_cast<unsigned>(n_convs)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const float*>(state_bak), reinterpret_cast<const float*>(hist),
                     reinterpret_cast<float*>(state_out), reinterpret_cast<const int64_t*>(conv_off),
                     reinterpret_cast<const int64_t*>(conv_hist_off), reinterpret_cast<const int32_t*>(conv_ch), n_keep);
  CheckLaunch(fn);
}

void r4dx_inkling2_add_bf16(int64_t a, int64_t b, int64_t out, int64_t n, int64_t stream) {
  const char* fn = "r4dx_inkling2_add_bf16";
  if (n <= 0) return;
  if (a == 0 || b == 0 || out == 0) Fail(fn, "null pointer");
  hipLaunchKernelGGL(AddBf16Kernel, dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(a),
                     reinterpret_cast<const uint16_t*>(b), reinterpret_cast<uint16_t*>(out), n);
  CheckLaunch(fn);
}

void r4dx_inkling2_add_f32_bf16(int64_t a, int64_t b, int64_t out, int64_t n, int64_t stream) {
  const char* fn = "r4dx_inkling2_add_f32_bf16";
  if (n <= 0) return;
  if (a == 0 || b == 0 || out == 0) Fail(fn, "null pointer");
  hipLaunchKernelGGL(AddF32Bf16Kernel, dim3(static_cast<unsigned>((n + kThreads - 1) / kThreads)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const float*>(a),
                     reinterpret_cast<const uint16_t*>(b), reinterpret_cast<float*>(out), n);
  CheckLaunch(fn);
}

void r4dx_inkling2_qgrkv_split_bf16(int64_t src, int64_t total_n, int64_t q_out, int64_t q_off, int64_t q_n,
                                    int64_t g_out, int64_t g_off, int64_t g_n, int64_t r_out, int64_t r_off,
                                    int64_t r_n, int64_t k_out, int64_t k_off, int64_t k_n, int64_t v_out,
                                    int64_t v_off, int64_t v_n, int T, int64_t stream) {
  const char* fn = "r4dx_inkling2_qgrkv_split_bf16";
  if (T <= 0) return;
  int64_t max_n = q_n;
  if (g_n > max_n) max_n = g_n;
  if (r_n > max_n) max_n = r_n;
  if (k_out != 0 && k_n > max_n) max_n = k_n;
  if (v_out != 0 && v_n > max_n) max_n = v_n;
  if (max_n <= 0) return;
  hipLaunchKernelGGL(QgrkvSplitBf16Kernel,
                     dim3(static_cast<unsigned>((max_n + kThreads - 1) / kThreads), static_cast<unsigned>(T)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(src),
                     total_n, reinterpret_cast<uint16_t*>(q_out), q_off, q_n, reinterpret_cast<uint16_t*>(g_out),
                     g_off, g_n, reinterpret_cast<uint16_t*>(r_out), r_off, r_n, reinterpret_cast<uint16_t*>(k_out),
                     k_off, k_n, reinterpret_cast<uint16_t*>(v_out), v_off, v_n, T);
  CheckLaunch(fn);
}

}  // extern "C"

namespace {

__global__ __launch_bounds__(kThreads) void VitPosClsKernel(
    const uint16_t* __restrict__ patch_out, const uint16_t* __restrict__ cls_emb,
    const uint16_t* __restrict__ cls_pos, const uint16_t* __restrict__ pos_emb, uint16_t* __restrict__ out,
    int gh, int gw, int grid_src, int D) {
  const int row = static_cast<int>(blockIdx.y);
  const int col = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  if (col >= D) return;
  if (row == 0) {
    const float v = r4dx::core::Bf16ToFloat(cls_emb[col]) + r4dx::core::Bf16ToFloat(cls_pos[col]);
    out[col] = r4dx::core::FloatToBf16(v);
    return;
  }
  const int p = row - 1;
  const int gy = p / gw;
  const int gx = p - gy * gw;
  const float fy = (gh > 1) ? ((static_cast<float>(gy) + 0.5f) * static_cast<float>(grid_src) / static_cast<float>(gh) - 0.5f) : 0.0f;
  const float fx = (gw > 1) ? ((static_cast<float>(gx) + 0.5f) * static_cast<float>(grid_src) / static_cast<float>(gw) - 0.5f) : 0.0f;
  int y0 = static_cast<int>(floorf(fy));
  int x0 = static_cast<int>(floorf(fx));
  float wy = fy - static_cast<float>(y0);
  float wx = fx - static_cast<float>(x0);
  if (y0 < 0) { y0 = 0; wy = 0.0f; }
  if (x0 < 0) { x0 = 0; wx = 0.0f; }
  if (y0 >= grid_src - 1) { y0 = grid_src - 1; wy = 0.0f; }
  if (x0 >= grid_src - 1) { x0 = grid_src - 1; wx = 0.0f; }
  const int y1 = (y0 + 1 < grid_src) ? (y0 + 1) : y0;
  const int x1 = (x0 + 1 < grid_src) ? (x0 + 1) : x0;
  const float p00 = r4dx::core::Bf16ToFloat(pos_emb[static_cast<int64_t>(y0 * grid_src + x0) * D + col]);
  const float p01 = r4dx::core::Bf16ToFloat(pos_emb[static_cast<int64_t>(y0 * grid_src + x1) * D + col]);
  const float p10 = r4dx::core::Bf16ToFloat(pos_emb[static_cast<int64_t>(y1 * grid_src + x0) * D + col]);
  const float p11 = r4dx::core::Bf16ToFloat(pos_emb[static_cast<int64_t>(y1 * grid_src + x1) * D + col]);
  const float pe = (1.0f - wy) * ((1.0f - wx) * p00 + wx * p01) + wy * ((1.0f - wx) * p10 + wx * p11);
  const float v = r4dx::core::Bf16ToFloat(patch_out[static_cast<int64_t>(p) * D + col]) + pe;
  out[static_cast<int64_t>(row) * D + col] = r4dx::core::FloatToBf16(v);
}

__global__ __launch_bounds__(kThreads) void VitLayerNormKernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ w, const uint16_t* __restrict__ b,
    uint16_t* __restrict__ out, int D, float eps) {
  __shared__ float red[kWaves];
  const int64_t row = blockIdx.x;
  const uint16_t* xr = x + row * D;
  float sum = 0.0f;
  for (int c = threadIdx.x; c < D; c += kThreads) sum += r4dx::core::Bf16ToFloat(xr[c]);
  const float mean = BlockSum(sum, red) / static_cast<float>(D);
  float var = 0.0f;
  for (int c = threadIdx.x; c < D; c += kThreads) {
    const float d = r4dx::core::Bf16ToFloat(xr[c]) - mean;
    var += d * d;
  }
  const float inv = rsqrtf(BlockSum(var, red) / static_cast<float>(D) + eps);
  uint16_t* orow = out + row * D;
  for (int c = threadIdx.x; c < D; c += kThreads) {
    float v = (r4dx::core::Bf16ToFloat(xr[c]) - mean) * inv * r4dx::core::Bf16ToFloat(w[c]);
    if (b != nullptr) v += r4dx::core::Bf16ToFloat(b[c]);
    orow[c] = r4dx::core::FloatToBf16(v);
  }
}

__global__ __launch_bounds__(kThreads) void VitBiasGeluKernel(
    const uint16_t* __restrict__ x, const uint16_t* __restrict__ b, uint16_t* __restrict__ out,
    int64_t total, int D, int do_gelu) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (i >= total) return;
  float v = r4dx::core::Bf16ToFloat(x[i]);
  if (b != nullptr) v += r4dx::core::Bf16ToFloat(b[i % D]);
  if (do_gelu) v = 0.5f * v * (1.0f + erff(v * 0.7071067811865475f));
  out[i] = r4dx::core::FloatToBf16(v);
}

__global__ __launch_bounds__(kThreads) void VitLayerScaleAddKernel(
    uint16_t* __restrict__ resid, const uint16_t* __restrict__ x, const uint16_t* __restrict__ b,
    const uint16_t* __restrict__ gamma, int64_t total, int D) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * kThreads + threadIdx.x;
  if (i >= total) return;
  const int c = static_cast<int>(i % D);
  float xv = r4dx::core::Bf16ToFloat(x[i]);
  if (b != nullptr) xv += r4dx::core::Bf16ToFloat(b[c]);
  const float rv = r4dx::core::Bf16ToFloat(resid[i]) + xv * r4dx::core::Bf16ToFloat(gamma[c]);
  resid[i] = r4dx::core::FloatToBf16(rv);
}

__global__ __launch_bounds__(32) void VitAttnH64Kernel(
    const uint16_t* __restrict__ q, const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
    uint16_t* __restrict__ out, int N, int heads, float scale) {
  const int h = static_cast<int>(blockIdx.x);
  const int qi = static_cast<int>(blockIdx.y);
  const int lane = threadIdx.x;
  const int64_t stride = static_cast<int64_t>(heads) * 64;
  const uint16_t* qr = q + static_cast<int64_t>(qi) * stride + static_cast<int64_t>(h) * 64;
  const float q0 = r4dx::core::Bf16ToFloat(qr[lane]);
  const float q1 = r4dx::core::Bf16ToFloat(qr[lane + 32]);
  float mv = -INFINITY, lv = 0.0f, o0 = 0.0f, o1 = 0.0f;
  for (int ki = 0; ki < N; ++ki) {
    const int64_t koff = static_cast<int64_t>(ki) * stride + static_cast<int64_t>(h) * 64;
    const float k0 = r4dx::core::Bf16ToFloat(k[koff + lane]);
    const float k1 = r4dx::core::Bf16ToFloat(k[koff + lane + 32]);
    float dot = WaveSum(q0 * k0 + q1 * k1) * scale;
    const float mn = fmaxf(mv, dot);
    const float alpha = (mv == -INFINITY) ? 0.0f : expf(mv - mn);
    const float p = expf(dot - mn);
    lv = lv * alpha + p;
    o0 = o0 * alpha + p * r4dx::core::Bf16ToFloat(v[koff + lane]);
    o1 = o1 * alpha + p * r4dx::core::Bf16ToFloat(v[koff + lane + 32]);
    mv = mn;
  }
  const float inv = (lv > 0.0f) ? (1.0f / lv) : 0.0f;
  uint16_t* orow = out + static_cast<int64_t>(qi) * stride + static_cast<int64_t>(h) * 64;
  orow[lane] = r4dx::core::FloatToBf16(o0 * inv);
  orow[lane + 32] = r4dx::core::FloatToBf16(o1 * inv);
}

__global__ __launch_bounds__(kThreads) void VitMerge2x2Kernel(
    const uint16_t* __restrict__ in_with_cls, uint16_t* __restrict__ out, int gh, int gw, int D) {
  const int mh = gh / 2, mw = gw / 2;
  const int m_idx = static_cast<int>(blockIdx.y);
  const int col = static_cast<int>(blockIdx.x) * kThreads + threadIdx.x;
  if (m_idx >= mh * mw || col >= 4 * D) return;
  const int my = m_idx / mw, mx = m_idx - my * mw;
  const int sub = col / D;
  const int d = col - sub * D;
  const int dy = sub >> 1, dx = sub & 1;
  const int src_p = (my * 2 + dy) * gw + (mx * 2 + dx);
  out[static_cast<int64_t>(m_idx) * (4 * D) + col] = in_with_cls[static_cast<int64_t>(src_p + 1) * D + d];
}

}  // namespace

extern "C" {

void r4dx_inkling2_vit_pos_cls_bf16(int64_t patch_out, int64_t cls_emb, int64_t cls_pos, int64_t pos_emb, int64_t out,
                                    int gh, int gw, int grid_src, int D, int64_t stream) {
  const char* fn = "r4dx_inkling2_vit_pos_cls_bf16";
  if (gh <= 0 || gw <= 0 || D <= 0) return;
  hipLaunchKernelGGL(VitPosClsKernel, dim3(static_cast<unsigned>((D + kThreads - 1) / kThreads), static_cast<unsigned>(1 + gh * gw)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(patch_out), reinterpret_cast<const uint16_t*>(cls_emb),
                     reinterpret_cast<const uint16_t*>(cls_pos), reinterpret_cast<const uint16_t*>(pos_emb),
                     reinterpret_cast<uint16_t*>(out), gh, gw, grid_src, D);
  CheckLaunch(fn);
}

void r4dx_inkling2_vit_layernorm_bf16(int64_t x, int64_t weight, int64_t bias, int64_t out, int rows, int D, float eps,
                                      int64_t stream) {
  const char* fn = "r4dx_inkling2_vit_layernorm_bf16";
  if (rows <= 0 || D <= 0) return;
  hipLaunchKernelGGL(VitLayerNormKernel, dim3(static_cast<unsigned>(rows)), dim3(kThreads), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(x),
                     reinterpret_cast<const uint16_t*>(weight), reinterpret_cast<const uint16_t*>(bias),
                     reinterpret_cast<uint16_t*>(out), D, eps);
  CheckLaunch(fn);
}

void r4dx_inkling2_vit_bias_gelu_bf16(int64_t x, int64_t bias, int64_t out, int rows, int D, int do_gelu,
                                      int64_t stream) {
  const char* fn = "r4dx_inkling2_vit_bias_gelu_bf16";
  const int64_t total = static_cast<int64_t>(rows) * D;
  if (total <= 0) return;
  hipLaunchKernelGGL(VitBiasGeluKernel, dim3(static_cast<unsigned>((total + kThreads - 1) / kThreads)), dim3(kThreads),
                     0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(x),
                     reinterpret_cast<const uint16_t*>(bias), reinterpret_cast<uint16_t*>(out), total, D, do_gelu);
  CheckLaunch(fn);
}

void r4dx_inkling2_vit_layerscale_add_bf16(int64_t resid, int64_t x, int64_t bias, int64_t gamma, int rows, int D,
                                           int64_t stream) {
  const char* fn = "r4dx_inkling2_vit_layerscale_add_bf16";
  const int64_t total = static_cast<int64_t>(rows) * D;
  if (total <= 0) return;
  hipLaunchKernelGGL(VitLayerScaleAddKernel, dim3(static_cast<unsigned>((total + kThreads - 1) / kThreads)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream), reinterpret_cast<uint16_t*>(resid),
                     reinterpret_cast<const uint16_t*>(x), reinterpret_cast<const uint16_t*>(bias),
                     reinterpret_cast<const uint16_t*>(gamma), total, D);
  CheckLaunch(fn);
}

void r4dx_inkling2_vit_attn_bf16(int64_t q, int64_t k, int64_t v, int64_t out, int N, int heads, int head_dim,
                                 float scale, int64_t stream) {
  const char* fn = "r4dx_inkling2_vit_attn_bf16";
  if (N <= 0 || heads <= 0) return;
  if (head_dim != 64) Fail(fn, "head_dim must be 64");
  hipLaunchKernelGGL(VitAttnH64Kernel, dim3(static_cast<unsigned>(heads), static_cast<unsigned>(N)), dim3(32), 0,
                     reinterpret_cast<hipStream_t>(stream), reinterpret_cast<const uint16_t*>(q),
                     reinterpret_cast<const uint16_t*>(k), reinterpret_cast<const uint16_t*>(v),
                     reinterpret_cast<uint16_t*>(out), N, heads, scale);
  CheckLaunch(fn);
}

void r4dx_inkling2_vit_merge2x2_bf16(int64_t in_with_cls, int64_t out, int gh, int gw, int D, int64_t stream) {
  const char* fn = "r4dx_inkling2_vit_merge2x2_bf16";
  if (gh <= 0 || gw <= 0 || (gh % 2) != 0 || (gw % 2) != 0 || D <= 0) return;
  const int m_rows = (gh / 2) * (gw / 2);
  hipLaunchKernelGGL(VitMerge2x2Kernel,
                     dim3(static_cast<unsigned>((4 * D + kThreads - 1) / kThreads), static_cast<unsigned>(m_rows)),
                     dim3(kThreads), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const uint16_t*>(in_with_cls), reinterpret_cast<uint16_t*>(out), gh, gw, D);
  CheckLaunch(fn);
}

}  // extern "C"
