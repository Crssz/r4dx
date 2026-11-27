// hy_v3_kernels.hip -- optimized gfx1201 kernels for hy_v3 (Hy-MT3-30B-A3B).
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4d_gdn_wmma.h"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/hy_v3_kernels.h"
#include "r4dx/kernels/kernels.h"

namespace {

using r4dx::core::FloatToFp8E4M3;

constexpr int kWave = 32;
constexpr int kW4KPB = 64;
constexpr int kRouterThreads = 1024;
constexpr int kMaxExperts = 256;
constexpr int kMaxRouterGroups = kMaxExperts / 32;
constexpr int kMaxTopK = 8;
constexpr int kNormThreads = 256;

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) const uint4v* gw4;
typedef __attribute__((address_space(1))) const unsigned long long* gu2;
typedef __attribute__((address_space(1))) const unsigned* gu1;

[[noreturn]] void Fail(const char* who, const std::string& what) {
  throw std::runtime_error(std::string(who) + ": " + what);
}

__device__ __forceinline__ unsigned bf2_to_h2(unsigned d) {
  const float lo = __uint_as_float(d << 16);
  const float hi = __uint_as_float(d & 0xffff0000u);
  union {
    decltype(__builtin_amdgcn_cvt_pkrtz(0.f, 0.f)) h;
    unsigned u;
  } x;
  x.h = __builtin_amdgcn_cvt_pkrtz(lo, hi);
  return x.u;
}

__device__ __forceinline__ v8h fragA_bf16(gu2 p, int u) {
  const unsigned long long w0 = p[4 * u];
  const unsigned long long w1 = p[4 * u + 2];
  union {
    unsigned u[4];
    v8h f;
  } x;
  x.u[0] = bf2_to_h2((unsigned)w0);
  x.u[1] = bf2_to_h2((unsigned)(w0 >> 32));
  x.u[2] = bf2_to_h2((unsigned)w1);
  x.u[3] = bf2_to_h2((unsigned)(w1 >> 32));
  return x.f;
}

__device__ __forceinline__ v8h dequant_w4(unsigned w, __half2 nz) {
  const unsigned K64 = 0x64646464u;
  const unsigned t0 = (w & 0x0F0F0F0Fu) ^ 0x08080808u;
  const unsigned t1 = ((w >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
  union {
    unsigned u[4];
    __half2 h2[4];
    v8h f;
  } x;
  x.u[0] = __builtin_amdgcn_perm(K64, t0, 0x05010400u);
  x.u[1] = __builtin_amdgcn_perm(K64, t0, 0x05030402u);
  x.u[2] = __builtin_amdgcn_perm(K64, t1, 0x05010400u);
  x.u[3] = __builtin_amdgcn_perm(K64, t1, 0x05030402u);
#pragma unroll
  for (int i = 0; i < 4; i++) x.h2[i] = __hadd2(x.h2[i], nz);
  return x.f;
}

__device__ __forceinline__ v8f mmah(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

// ---- 1. Fused Q/K norm + NeoX RoPE + paged KV write (head_dim = 128, bf16 or fp8 KV) ----
__global__ __launch_bounds__(128) void HyV3AttnPrecoreKernel(
    __hip_bfloat16* __restrict__ q, __hip_bfloat16* __restrict__ k,
    const __hip_bfloat16* __restrict__ v, const __hip_bfloat16* __restrict__ q_norm,
    const __hip_bfloat16* __restrict__ k_norm, const int32_t* __restrict__ pos_ids,
    void* __restrict__ kv_cache, int kv_fp8, const float* __restrict__ k_descale,
    const float* __restrict__ v_descale, int heads_q, int heads_k, float rms_eps,
    float rope_theta, int block_size, int64_t kv_block_stride, int64_t kv_head_stride) {
  constexpr int kHd = 128;
  constexpr int kHalf = 64;
  __shared__ float red[4];
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
  const float total = red[0] + red[1] + red[2] + red[3];
  const float rstd = rsqrtf(total / 128.0f + rms_eps);
  const __hip_bfloat16 normed = __float2bfloat16(val * rstd * __bfloat162float(nw[d]));
  sh_row[d] = __bfloat162float(normed);
  __syncthreads();

  const int slot = pos_ids[t];
  if (d < kHalf) {
    const float inv_freq = exp2f(-(2.0f * d) / 128.0f * log2f(rope_theta));
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
      if (kv_fp8) {
        const float kd_inv = k_descale ? (1.0f / k_descale[h]) : 1.0f;
        uint8_t* kv8 = static_cast<uint8_t*>(kv_cache);
        kv8[base + d] = FloatToFp8E4M3(__bfloat162float(r0) * kd_inv);
        kv8[base + d + kHalf] = FloatToFp8E4M3(__bfloat162float(r1) * kd_inv);
      } else {
        __hip_bfloat16* kv16 = static_cast<__hip_bfloat16*>(kv_cache);
        kv16[base + d] = r0;
        kv16[base + d + kHalf] = r1;
      }
    }
  }
  if (!is_q && slot >= 0) {
    const int64_t base = static_cast<int64_t>(slot / block_size) * kv_block_stride +
                         static_cast<int64_t>(h) * kv_head_stride +
                         static_cast<int64_t>(slot % block_size) * (2 * kHd);
    const int64_t v_src = (static_cast<int64_t>(t) * heads_k + h) * kHd;
    const __hip_bfloat16 vv = v[v_src + d];
    if (kv_fp8) {
      const float vd_inv = v_descale ? (1.0f / v_descale[h]) : 1.0f;
      uint8_t* kv8 = static_cast<uint8_t*>(kv_cache);
      kv8[base + kHd + d] = FloatToFp8E4M3(__bfloat162float(vv) * vd_inv);
    } else {
      __hip_bfloat16* kv16 = static_cast<__hip_bfloat16*>(kv_cache);
      kv16[base + kHd + d] = vv;
    }
  }
}

// ---- 2. Multi-block vectorized 128-expert router for T==1 (128 blocks x 1 wave, zero LDS, zero syncs) ----
__global__ __launch_bounds__(32) void HyV3RouterGemvM1Kernel(
    const __hip_bfloat16* __restrict__ x, const __hip_bfloat16* __restrict__ gate_w,
    float* __restrict__ sig_out, int hidden, int n_experts) {
  const int e = static_cast<int>(blockIdx.x);
  if (e >= n_experts) return;
  const int lane = threadIdx.x;
  const int n_vec = hidden >> 3;
  const uint4* xv = reinterpret_cast<const uint4*>(x);
  const uint4* wv = reinterpret_cast<const uint4*>(gate_w + static_cast<int64_t>(e) * hidden);
  float acc = 0.0f;
  for (int v = lane; v < n_vec; v += 32) {
    const uint4 ux = xv[v];
    const uint4 uw = wv[v];
    const __hip_bfloat16* bx = reinterpret_cast<const __hip_bfloat16*>(&ux);
    const __hip_bfloat16* bw = reinterpret_cast<const __hip_bfloat16*>(&uw);
#pragma unroll
    for (int i = 0; i < 8; ++i) acc += __bfloat162float(bx[i]) * __bfloat162float(bw[i]);
  }
  for (int o = 16; o; o >>= 1) acc += __shfl_xor(acc, o, 32);
  if (lane == 0) sig_out[e] = 1.0f / (1.0f + expf(-acc));
}

// Single-wave top-k + bias + normalize + shared slot for T==1.
__global__ __launch_bounds__(32) void HyV3RouterTopkM1Kernel(
    const float* __restrict__ sig, const float* __restrict__ bias,
    int32_t* __restrict__ ids, float* __restrict__ wout,
    int n_experts, int top_k, float scale, int append_shared) {
  const int lane = threadIdx.x;
  float sc[kMaxRouterGroups];
  float sv[kMaxRouterGroups];
  for (int g = 0; g < kMaxRouterGroups; ++g) {
    const int e = lane + g * 32;
    if (e < n_experts) {
      sv[g] = sig[e];
      sc[g] = sv[g] + bias[e];
    } else {
      sv[g] = 0.0f;
      sc[g] = -INFINITY;
    }
  }
  __shared__ int sel[kMaxTopK];
  __shared__ float sel_sig[kMaxTopK];
  for (int r = 0; r < top_k; ++r) {
    float best = -INFINITY;
    float bsig = 0.0f;
    int bi = lane;
    for (int g = 0; g < kMaxRouterGroups; ++g) {
      if (sc[g] > best) {
        best = sc[g];
        bsig = sv[g];
        bi = lane + g * 32;
      }
    }
    for (int o = 16; o; o >>= 1) {
      const float os = __shfl_xor(best, o, 32);
      const float ov = __shfl_xor(bsig, o, 32);
      const int oi = __shfl_xor(bi, o, 32);
      if (os > best || (os == best && oi < bi)) {
        best = os;
        bsig = ov;
        bi = oi;
      }
    }
    if (lane == 0) {
      sel[r] = bi;
      sel_sig[r] = bsig;
    }
    for (int g = 0; g < kMaxRouterGroups; ++g) {
      if (bi == lane + g * 32) sc[g] = -INFINITY;
    }
  }
  if (lane == 0) {
    float sum = 0.0f;
    for (int r = 0; r < top_k; ++r) sum += sel_sig[r];
    const float denom = sum + 1e-6f;
    for (int r = 0; r < top_k; ++r) {
      ids[r] = sel[r];
      wout[r] = (sel_sig[r] / denom) * scale;
    }
    if (append_shared) {
      ids[top_k] = n_experts;
      wout[top_k] = 1.0f;
    }
  }
}

// Prefill / general T > 1 router (1 block per token).
__global__ __launch_bounds__(kRouterThreads) void HyV3RouterKernel(
    const __hip_bfloat16* __restrict__ x, const __hip_bfloat16* __restrict__ gate_w,
    const float* __restrict__ bias, int32_t* __restrict__ ids, float* __restrict__ wout,
    int hidden, int n_experts, int top_k, float scale, int append_shared) {
  extern __shared__ float xs[];
  __shared__ float sig[kMaxExperts];
  __shared__ int sel[kMaxTopK];

  const int64_t t = blockIdx.x;
  const __hip_bfloat16* xr = x + t * hidden;
  for (int k = threadIdx.x; k < hidden; k += kRouterThreads) xs[k] = __bfloat162float(xr[k]);
  __syncthreads();

  const int lane = threadIdx.x & 31;
  const int wave = threadIdx.x >> 5;
  if (n_experts == 128 && (hidden & 7) == 0) {
    const int n_vec = hidden >> 3;
    const uint4* w0 = reinterpret_cast<const uint4*>(gate_w + static_cast<int64_t>(wave) * hidden);
    const uint4* w1 = reinterpret_cast<const uint4*>(gate_w + static_cast<int64_t>(wave + 32) * hidden);
    const uint4* w2 = reinterpret_cast<const uint4*>(gate_w + static_cast<int64_t>(wave + 64) * hidden);
    const uint4* w3 = reinterpret_cast<const uint4*>(gate_w + static_cast<int64_t>(wave + 96) * hidden);
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    for (int v = lane; v < n_vec; v += 32) {
      const float* xv = xs + (v << 3);
      auto dot8 = [&](const uint4& u) -> float {
        const __hip_bfloat16* b = reinterpret_cast<const __hip_bfloat16*>(&u);
        float s = 0.0f;
#pragma unroll
        for (int i = 0; i < 8; ++i) s += xv[i] * __bfloat162float(b[i]);
        return s;
      };
      acc0 += dot8(w0[v]);
      acc1 += dot8(w1[v]);
      acc2 += dot8(w2[v]);
      acc3 += dot8(w3[v]);
    }
    for (int o = 16; o; o >>= 1) {
      acc0 += __shfl_xor(acc0, o, 32);
      acc1 += __shfl_xor(acc1, o, 32);
      acc2 += __shfl_xor(acc2, o, 32);
      acc3 += __shfl_xor(acc3, o, 32);
    }
    if (lane == 0) {
      sig[wave] = 1.0f / (1.0f + expf(-acc0));
      sig[wave + 32] = 1.0f / (1.0f + expf(-acc1));
      sig[wave + 64] = 1.0f / (1.0f + expf(-acc2));
      sig[wave + 96] = 1.0f / (1.0f + expf(-acc3));
    }
  } else {
    for (int e = wave; e < n_experts; e += kRouterThreads / 32) {
      const __hip_bfloat16* wr = gate_w + static_cast<int64_t>(e) * hidden;
      float acc = 0.0f;
      for (int k = lane; k < hidden; k += 32) acc += xs[k] * __bfloat162float(wr[k]);
      for (int o = 16; o; o >>= 1) acc += __shfl_xor(acc, o, 32);
      if (lane == 0) sig[e] = 1.0f / (1.0f + expf(-acc));
    }
  }
  __syncthreads();

  if (wave == 0) {
    float sc[kMaxRouterGroups];
    for (int g = 0; g < kMaxRouterGroups; ++g) {
      const int e = lane + g * 32;
      sc[g] = (e < n_experts) ? sig[e] + bias[e] : -INFINITY;
    }
    for (int r = 0; r < top_k; ++r) {
      float best = -INFINITY;
      int bi = lane;
      for (int g = 0; g < kMaxRouterGroups; ++g) {
        if (sc[g] > best) {
          best = sc[g];
          bi = lane + g * 32;
        }
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
      for (int g = 0; g < kMaxRouterGroups; ++g) {
        if (bi == lane + g * 32) sc[g] = -INFINITY;
      }
    }
    if (lane == 0) {
      float sum = 0.0f;
      for (int r = 0; r < top_k; ++r) sum += sig[sel[r]];
      const float denom = sum + 1e-6f;
      const int stride = append_shared ? (top_k + 1) : top_k;
      for (int r = 0; r < top_k; ++r) {
        ids[t * stride + r] = sel[r];
        wout[t * stride + r] = (sig[sel[r]] / denom) * scale;
      }
      if (append_shared) {
        ids[t * stride + top_k] = n_experts;
        wout[t * stride + top_k] = 1.0f;
      }
    }
  }
}

// ---- 3. w4a16 M=1 body with SK==1 direct-register store (zero LDS, zero __syncthreads) ----
template <int G, int NPW, bool NT>
__device__ __forceinline__ void w4a16_m1_body(
    const unsigned short* __restrict__ Ap, const unsigned* __restrict__ Wq,
    const unsigned* __restrict__ Wsz, __hip_bfloat16* __restrict__ Cp,
    int nt0_in, int K, int N, int WV, int SK, float* red) {
  static_assert(G == 32 || G % kW4KPB == 0, "group must be 32 or multiple of 64");
  const int tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = nt0_in;
  const int n0 = nt0 * 16;
  const int Ks = K / SK;
  const int groups = Ks / G;
  const int kbase = ksplit * Ks;

  // Scalar M=1 accumulators: only row m=0 (g_acc[j][0]) is ever used, saving 7*NPW VGPRs & 7*NPW FMAs/group!
  float acc[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) acc[j] = 0.f;

  if (n0 < N) {
    const unsigned off = rowGo((unsigned)K, 0, 0, lane) + (unsigned)kbase;
    gu2 pa = (gu2)(const unsigned long long*)(Ap + off);

    const int kpb = K / kW4KPB;
    const int nsz = K / G;
    gw4 pw[NPW];
    gu1 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (gw4)(const uint4v*)(Wq) + (size_t)(t * kpb + kbase / kW4KPB) * kWave + lane;
      ps[j] = (gu1)(const unsigned*)(Wsz) + (size_t)(t * nsz + kbase / G) * 16 + (lane & 15);
    }

    if constexpr (G >= kW4KPB) {
      const int bpg = G / kW4KPB;
      for (int g = 0; g < groups; g++) {
        float sc[NPW];
        __half2 nz[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const unsigned sz = ps[j][(size_t)g * 16];
          sc[j] = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
          nz[j] = __half2half2(__ushort_as_half((unsigned short)(sz >> 16)));
        }
        v8f g_acc[NPW];
#pragma unroll
        for (int b = 0; b < bpg; b++) {
          uint4v wv[NPW];
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const size_t o = (size_t)(g * bpg + b) * kWave;
            wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
          }
#pragma unroll
          for (int s = 0; s < 4; s++) {
            const int u = b * 4 + s;
            const v8h afrag = fragA_bf16(pa, u);
#pragma unroll
            for (int j = 0; j < NPW; j++) {
              const v8h bfrag = dequant_w4(wv[j][s], nz[j]);
              g_acc[j] = u == 0 ? mmah(afrag, bfrag, (v8f)(0.f)) : mmah(afrag, bfrag, g_acc[j]);
            }
          }
        }
#pragma unroll
        for (int j = 0; j < NPW; j++) acc[j] = fmaf(g_acc[j][0], sc[j], acc[j]);
        pa += (G / 16) * 4;
      }
    } else {
      constexpr int gpb = kW4KPB / G;
      constexpr int spg = G / 16;
      const int blocks = groups / gpb;
      for (int b = 0; b < blocks; b++) {
        uint4v wv[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const size_t o = (size_t)b * kWave;
          wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        }
#pragma unroll
        for (int h = 0; h < gpb; h++) {
          const int g = b * gpb + h;
          float sc[NPW];
          __half2 nz[NPW];
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const unsigned sz = ps[j][(size_t)g * 16];
            sc[j] = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
            nz[j] = __half2half2(__ushort_as_half((unsigned short)(sz >> 16)));
          }
          v8f g_acc[NPW];
#pragma unroll
          for (int u = 0; u < spg; u++) {
            const int s = h * spg + u;
            const v8h afrag = fragA_bf16(pa, u);
#pragma unroll
            for (int j = 0; j < NPW; j++) {
              const v8h bfrag = dequant_w4(wv[j][s], nz[j]);
              g_acc[j] = u == 0 ? mmah(afrag, bfrag, (v8f)(0.f)) : mmah(afrag, bfrag, g_acc[j]);
            }
          }
#pragma unroll
          for (int j = 0; j < NPW; j++) acc[j] = fmaf(g_acc[j][0], sc[j], acc[j]);
          pa += spg * 4;
        }
      }
    }
  }

  // Fast path: SK == 1 needs NO LDS and NO __syncthreads()!
  // In gfx12 WMMA 16x16x16, row m=0 is held by lanes 0..15 in acc[j].
  if (SK == 1) {
    if (lane < 16) {
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const int n = (nt0 + j) * 16 + lane;
        if (n < N) Cp[n] = __float2bfloat16(acc[j]);
      }
    }
    return;
  }

  // Tiny M=1 SK>1 reduction: only row m=0 (lanes 0..15, acc[j]) -> 16x less LDS and traffic!
  if (lane < 16) {
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      red[((col_local * NPW + j) * SK + ksplit) * 16 + lane] = acc[j];
    }
  }
  __syncthreads();
  if (ksplit == 0 && lane < 16) {
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int n = (nt0 + j) * 16 + lane;
      if (n < N) {
        float v = 0.f;
        const int base = (col_local * NPW + j) * SK * 16 + lane;
        for (int s = 0; s < SK; s++) v += red[base + s * 16];
        Cp[n] = __float2bfloat16(v);
      }
    }
  }
}

// ---- 4. Fused Q+K+V w4a16 M=1 kernel (1 launch for all 3 projections) ----
template <int G, bool NT>
__global__ __launch_bounds__(1024) void W4a16QkvBf16AM1Kernel(
    const unsigned short* __restrict__ A,
    const unsigned* __restrict__ Wq_q, const unsigned* __restrict__ Wsz_q, __hip_bfloat16* __restrict__ C_q, int Nq,
    const unsigned* __restrict__ Wq_k, const unsigned* __restrict__ Wsz_k, __hip_bfloat16* __restrict__ C_k, int Nk,
    const unsigned* __restrict__ Wq_v, const unsigned* __restrict__ Wsz_v, __hip_bfloat16* __restrict__ C_v, int Nv,
    int K, int WV, int SK) {
  extern __shared__ float red[];
  const int wave = threadIdx.x / kWave;
  const int col_local = wave / SK;
  const int global_nt = blockIdx.x * WV + col_local;
  const int nt_q = Nq / 16;
  const int nt_k = Nk / 16;
  const int nt_v = Nv / 16;
  if (global_nt < nt_q) {
    w4a16_m1_body<G, 1, NT>(A, Wq_q, Wsz_q, C_q, global_nt, K, Nq, WV, SK, red);
  } else if (global_nt < nt_q + nt_k) {
    w4a16_m1_body<G, 1, NT>(A, Wq_k, Wsz_k, C_k, global_nt - nt_q, K, Nk, WV, SK, red);
  } else if (global_nt < nt_q + nt_k + nt_v) {
    w4a16_m1_body<G, 1, NT>(A, Wq_v, Wsz_v, C_v, global_nt - nt_q - nt_k, K, Nv, WV, SK, red);
  }
}

// ---- 5. Fused grouped w4a16 MoE gate_up + silu(gate)*up decode (supports SK >= 1, tiny 32-float LDS) ----
template <int G, bool NT>
__global__ __launch_bounds__(1024) void W4a16MoeGateUpSiluDecodeKernel(
    const unsigned short* __restrict__ A, const uint64_t* __restrict__ wq_ptrs,
    const uint64_t* __restrict__ wsz_ptrs, const int32_t* __restrict__ ids,
    __hip_bfloat16* __restrict__ Act, int a_row_group, int n_experts, int K, int Im, int WV, int SK) {
  static_assert(G == 64, "gate_up_silu expects G=64");
  extern __shared__ float red[];
  const int p = blockIdx.z;
  const int e = ids[p];
  if (e < 0 || e >= n_experts) return;
  const unsigned* Wq = reinterpret_cast<const unsigned*>(wq_ptrs[e]);
  const unsigned* Wsz = reinterpret_cast<const unsigned*>(wsz_ptrs[e]);
  if (!Wq || !Wsz) return;

  const int wave = threadIdx.x / kWave, lane = threadIdx.x % kWave;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt_g = blockIdx.x * WV + col_local;
  const int ntiles_im = Im / 16;
  if (nt_g >= ntiles_im) return;
  const int nt_u = nt_g + ntiles_im;

  const unsigned short* Ap = A + static_cast<size_t>(p / a_row_group) * static_cast<size_t>(K);
  __hip_bfloat16* Cp = Act + static_cast<size_t>(p) * static_cast<size_t>(Im);

  const int Ks = K / SK;
  const int kbase = ksplit * Ks;
  const int groups = Ks / G;
  const unsigned off = rowGo((unsigned)K, 0, 0, lane) + (unsigned)kbase;
  gu2 pa = (gu2)(const unsigned long long*)(Ap + off);

  const int kpb = K / kW4KPB;
  const int nsz = K / G;
  gw4 pw_g = (gw4)(const uint4v*)(Wq) + (size_t)(nt_g * kpb + kbase / kW4KPB) * kWave + lane;
  gw4 pw_u = (gw4)(const uint4v*)(Wq) + (size_t)(nt_u * kpb + kbase / kW4KPB) * kWave + lane;
  gu1 ps_g = (gu1)(const unsigned*)(Wsz) + (size_t)(nt_g * nsz + kbase / G) * 16 + (lane & 15);
  gu1 ps_u = (gu1)(const unsigned*)(Wsz) + (size_t)(nt_u * nsz + kbase / G) * 16 + (lane & 15);

  // Scalar M=1 accumulators: frees 14 VGPRs and removes 14 FMAs/group!
  float acc_g = 0.f;
  float acc_u = 0.f;

#pragma unroll 2
  for (int g = 0; g < groups; g++) {
    const unsigned sz_g = ps_g[(size_t)g * 16];
    const float sc_g = __half2float(__ushort_as_half((unsigned short)(sz_g & 0xFFFFu)));
    const __half2 nz_g = __half2half2(__ushort_as_half((unsigned short)(sz_g >> 16)));
    const unsigned sz_u = ps_u[(size_t)g * 16];
    const float sc_u = __half2float(__ushort_as_half((unsigned short)(sz_u & 0xFFFFu)));
    const __half2 nz_u = __half2half2(__ushort_as_half((unsigned short)(sz_u >> 16)));

    const size_t o = (size_t)g * kWave;
    const uint4v wv_g = NT ? __builtin_nontemporal_load(&pw_g[o]) : pw_g[o];
    const uint4v wv_u = NT ? __builtin_nontemporal_load(&pw_u[o]) : pw_u[o];

    v8f gg = (v8f)(0.f), gu = (v8f)(0.f);
#pragma unroll
    for (int s = 0; s < 4; s++) {
      const v8h afrag = fragA_bf16(pa, s);  // loaded ONCE, spent on BOTH gate and up!
      const v8h bg = dequant_w4(wv_g[s], nz_g);
      const v8h bu = dequant_w4(wv_u[s], nz_u);
      gg = s == 0 ? mmah(afrag, bg, (v8f)(0.f)) : mmah(afrag, bg, gg);
      gu = s == 0 ? mmah(afrag, bu, (v8f)(0.f)) : mmah(afrag, bu, gu);
    }
    acc_g = fmaf(gg[0], sc_g, acc_g);
    acc_u = fmaf(gu[0], sc_u, acc_u);
    pa += 16;  // (64 / 16) * 4
  }

  if (SK == 1) {
    if (lane < 16) {
      const int n = nt_g * 16 + lane;
      if (n < Im) {
        const float gv = __bfloat162float(__float2bfloat16(acc_g));
        const float uv = __bfloat162float(__float2bfloat16(acc_u));
        Cp[n] = __float2bfloat16((gv / (1.0f + expf(-gv))) * uv);
      }
    }
    return;
  }

  if (lane < 16) {
    const int slot = (col_local * SK + ksplit) * 32 + lane;
    red[slot] = acc_g;
    red[slot + 16] = acc_u;
  }
  __syncthreads();
  if (ksplit == 0 && lane < 16) {
    const int n = nt_g * 16 + lane;
    if (n < Im) {
      float ag = 0.f, au = 0.f;
      const int base = col_local * SK * 32 + lane;
      for (int s = 0; s < SK; s++) {
        ag += red[base + s * 32];
        au += red[base + s * 32 + 16];
      }
      const float gv = __bfloat162float(__float2bfloat16(ag));
      const float uv = __bfloat162float(__float2bfloat16(au));
      Cp[n] = __float2bfloat16((gv / (1.0f + expf(-gv))) * uv);
    }
  }
}

// ---- 5b. Fused grouped w4a16 MoE down + weighted combine + residual add (T==1, G=64) ----
// Grid (N/16 = 128 blocks = 2.0/CU), block (slots * 32 = 288 threads -> 1 wave per active expert).
// Eliminates the intermediate ys[9, 2048] VRAM write+read completely!
template <bool NT>
__global__ __launch_bounds__(512) void W4a16MoeDownCombineResDecodeKernel(
    const unsigned short* __restrict__ Act, const uint64_t* __restrict__ wq_ptrs,
    const uint64_t* __restrict__ wsz_ptrs, const int32_t* __restrict__ ids,
    const float* __restrict__ wts, __hip_bfloat16* __restrict__ h,
    int slots, int n_experts, int K, int N) {
  __shared__ float smem[16 * 16];  // up to 16 slots x 16 columns
  const int p = threadIdx.x / kWave;
  const int lane = threadIdx.x % kWave;
  const int nt = static_cast<int>(blockIdx.x);
  const int n = nt * 16 + lane;
  float val = 0.f;
  if (p < slots) {
    const int e = ids[p];
    if (e >= 0 && e < n_experts) {
      const unsigned* Wq = reinterpret_cast<const unsigned*>(wq_ptrs[e]);
      const unsigned* Wsz = reinterpret_cast<const unsigned*>(wsz_ptrs[e]);
      if (Wq && Wsz) {
        const unsigned short* Ap = Act + static_cast<size_t>(p) * static_cast<size_t>(K);
        const unsigned off = rowGo((unsigned)K, 0, 0, lane);
        gu2 pa = (gu2)(const unsigned long long*)(Ap + off);
        const int kpb = K / kW4KPB;
        const int groups = K / 64;
        gw4 pw = (gw4)(const uint4v*)(Wq) + (size_t)(nt * kpb) * kWave + lane;
        gu1 ps = (gu1)(const unsigned*)(Wsz) + (size_t)(nt * groups) * 16 + (lane & 15);
        float acc = 0.f;
        for (int g = 0; g < groups; g++) {
          const unsigned sz = ps[(size_t)g * 16];
          const float sc = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
          const __half2 nz = __half2half2(__ushort_as_half((unsigned short)(sz >> 16)));
          const size_t o = (size_t)g * kWave;
          const uint4v wv = NT ? __builtin_nontemporal_load(&pw[o]) : pw[o];
          v8f ga = (v8f)(0.f);
#pragma unroll
          for (int s = 0; s < 4; s++) {
            const v8h afrag = fragA_bf16(pa, s);
            const v8h bfrag = dequant_w4(wv[s], nz);
            ga = s == 0 ? mmah(afrag, bfrag, (v8f)(0.f)) : mmah(afrag, bfrag, ga);
          }
          acc = fmaf(ga[0], sc, acc);
          pa += 16;
        }
        val = wts[p] * __bfloat162float(__float2bfloat16(acc));
      }
    }
    if (lane < 16) smem[p * 16 + lane] = val;
  }
  __syncthreads();
  if (p == 0 && lane < 16 && n < N) {
    float sum = 0.f;
    for (int s = 0; s < slots; ++s) sum += smem[s * 16 + lane];
    const __hip_bfloat16 comb = __float2bfloat16(sum);
    h[n] = __float2bfloat16(__bfloat162float(h[n]) + __bfloat162float(comb));
  }
}

template <int G, int NPW, bool NT>
__global__ __launch_bounds__(1024) void W4a16MoeDecodeKernel(
    const unsigned short* __restrict__ A, const uint64_t* __restrict__ wq_ptrs,
    const uint64_t* __restrict__ wsz_ptrs, const int32_t* __restrict__ ids,
    __hip_bfloat16* __restrict__ C, int a_row_group, int n_experts, int K, int N, int WV, int SK) {
  extern __shared__ float red[];
  const int p = blockIdx.z;
  const int e = ids[p];
  if (e < 0 || e >= n_experts) return;
  const unsigned* Wq = reinterpret_cast<const unsigned*>(wq_ptrs[e]);
  const unsigned* Wsz = reinterpret_cast<const unsigned*>(wsz_ptrs[e]);
  if (!Wq || !Wsz) return;
  const unsigned short* Ap = A + static_cast<size_t>(p / a_row_group) * static_cast<size_t>(K);
  __hip_bfloat16* Cp = C + static_cast<size_t>(p) * static_cast<size_t>(N);
  const int wave = threadIdx.x / kWave;
  const int col_local = wave / SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  w4a16_m1_body<G, NPW, NT>(Ap, Wq, Wsz, Cp, nt0, K, N, WV, SK, red);
}

template <int G, int NPW, bool NT>
__global__ __launch_bounds__(1024) void W4a16Bf16AM1Kernel(
    const unsigned short* __restrict__ A, const unsigned* __restrict__ Wq,
    const unsigned* __restrict__ Wsz, __hip_bfloat16* __restrict__ C, int K, int N, int WV, int SK) {
  extern __shared__ float red[];
  const int wave = threadIdx.x / kWave;
  const int col_local = wave / SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  w4a16_m1_body<G, NPW, NT>(A, Wq, Wsz, C, nt0, K, N, WV, SK, red);
}

// ---- 6. Grouped w4a16 MoE prefill GEMM (M<=16 per tile, bf16 A, zero host syncs) ----
__device__ __forceinline__ bool ResolveW4MoePrefillTile(const int32_t* __restrict__ offsets,
                                                        int n_experts, int n_pairs, int tile_idx,
                                                        int& out_e, int& out_row_start,
                                                        int& out_m_rows) {
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

template <int G, int NPW, bool NT>
__global__ __launch_bounds__(1024) void W4a16MoePrefillKernel(
    const unsigned short* __restrict__ A, const uint64_t* __restrict__ wq_ptrs,
    const uint64_t* __restrict__ wsz_ptrs, const int32_t* __restrict__ offsets,
    __hip_bfloat16* __restrict__ C, int n_pairs, int n_experts, int K, int N, int WV, int SK) {
  static_assert(G == 32 || G % kW4KPB == 0, "group must be 32 or multiple of 64");
  extern __shared__ float red[];
  int e = -1, row_start = 0, m_rows = 0;
  if (!ResolveW4MoePrefillTile(offsets, n_experts, n_pairs, static_cast<int>(blockIdx.y), e,
                               row_start, m_rows)) {
    return;
  }
  const unsigned* Wq = reinterpret_cast<const unsigned*>(wq_ptrs[e]);
  const unsigned* Wsz = reinterpret_cast<const unsigned*>(wsz_ptrs[e]);
  if (!Wq || !Wsz) return;
  const unsigned short* Ap = A + static_cast<size_t>(row_start) * static_cast<size_t>(K);
  __hip_bfloat16* Cp = C + static_cast<size_t>(row_start) * static_cast<size_t>(N);

  const int tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int Ks = K / SK;
  const int groups = Ks / G;
  const int kbase = ksplit * Ks;

  v8f acc[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) acc[j] = (v8f)(0.f);

  if (n0 < N) {
    const unsigned off = rowGo((unsigned)K, 0, m_rows - 1, lane) + (unsigned)kbase;
    gu2 pa = (gu2)(const unsigned long long*)(Ap + off);

    const int kpb = K / kW4KPB;
    const int nsz = K / G;
    gw4 pw[NPW];
    gu1 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (gw4)(const uint4v*)(Wq) + (size_t)(t * kpb + kbase / kW4KPB) * kWave + lane;
      ps[j] = (gu1)(const unsigned*)(Wsz) + (size_t)(t * nsz + kbase / G) * 16 + (lane & 15);
    }

    if constexpr (G >= kW4KPB) {
      const int bpg = G / kW4KPB;
      for (int g = 0; g < groups; g++) {
        float sc[NPW];
        __half2 nz[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const unsigned sz = ps[j][(size_t)g * 16];
          sc[j] = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
          nz[j] = __half2half2(__ushort_as_half((unsigned short)(sz >> 16)));
        }
        v8f g_acc[NPW];
#pragma unroll
        for (int b = 0; b < bpg; b++) {
          uint4v wv[NPW];
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const size_t o = (size_t)(g * bpg + b) * kWave;
            wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
          }
#pragma unroll
          for (int s = 0; s < 4; s++) {
            const int u = b * 4 + s;
            const v8h afrag = fragA_bf16(pa, u);
#pragma unroll
            for (int j = 0; j < NPW; j++) {
              const v8h bfrag = dequant_w4(wv[j][s], nz[j]);
              g_acc[j] = u == 0 ? mmah(afrag, bfrag, (v8f)(0.f)) : mmah(afrag, bfrag, g_acc[j]);
            }
          }
        }
#pragma unroll
        for (int j = 0; j < NPW; j++)
#pragma unroll
          for (int el = 0; el < 8; el++) acc[j][el] = fmaf(g_acc[j][el], sc[j], acc[j][el]);
        pa += (G / 16) * 4;
      }
    } else {
      constexpr int gpb = kW4KPB / G;
      constexpr int spg = G / 16;
      const int blocks = groups / gpb;
      for (int b = 0; b < blocks; b++) {
        uint4v wv[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const size_t o = (size_t)b * kWave;
          wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        }
#pragma unroll
        for (int h = 0; h < gpb; h++) {
          const int g = b * gpb + h;
          float sc[NPW];
          __half2 nz[NPW];
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const unsigned sz = ps[j][(size_t)g * 16];
            sc[j] = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
            nz[j] = __half2half2(__ushort_as_half((unsigned short)(sz >> 16)));
          }
          v8f g_acc[NPW];
#pragma unroll
          for (int u = 0; u < spg; u++) {
            const int s = h * spg + u;
            const v8h afrag = fragA_bf16(pa, u);
#pragma unroll
            for (int j = 0; j < NPW; j++) {
              const v8h bfrag = dequant_w4(wv[j][s], nz[j]);
              g_acc[j] = u == 0 ? mmah(afrag, bfrag, (v8f)(0.f)) : mmah(afrag, bfrag, g_acc[j]);
            }
          }
#pragma unroll
          for (int j = 0; j < NPW; j++)
#pragma unroll
            for (int el = 0; el < 8; el++) acc[j][el] = fmaf(g_acc[j][el], sc[j], acc[j][el]);
          pa += spg * 4;
        }
      }
    }
  }

  // Fast path: SK == 1 needs NO LDS and NO __syncthreads()!
  // gfx12 WMMA 16x16x16: m = 8*(lane>>4) + el, n = (nt0 + j)*16 + (lane&15).
  if (SK == 1) {
    const int l15 = lane & 15;
    const int m_base = 8 * (lane >> 4);
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int n = (nt0 + j) * 16 + l15;
      if (n < N) {
#pragma unroll
        for (int el = 0; el < 8; el++) {
          const int m = m_base + el;
          if (m < m_rows) Cp[(size_t)m * (size_t)N + n] = __float2bfloat16(acc[j][el]);
        }
      }
    }
    return;
  }

  const int ncols = WV * NPW;
  const int nthreads = WV * SK * kWave;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
    for (int el = 0; el < 8; el++) red[slot + el * kWave + lane] = acc[j][el];
  }
  __syncthreads();
  for (int idx = tid; idx < ncols * 256; idx += nthreads) {
    const int col = idx >> 8, off = idx & 255;
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
    const int l = off & 31;
    const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
    const int m = 8 * (l >> 4) + (off >> 5);
    if (n < N && m < m_rows) Cp[(size_t)m * (size_t)N + n] = __float2bfloat16(v);
  }
}

// ---- 7. Vectorized fused MoE combine + residual add + optional next RMSNorm ----
__device__ __forceinline__ float BlockReduceSum256(float v) {
  __shared__ float smem[8];
  for (int o = 16; o; o >>= 1) v += __shfl_xor(v, o, 32);
  if ((threadIdx.x & 31) == 0) smem[threadIdx.x >> 5] = v;
  __syncthreads();
  float sum = 0.0f;
  if (threadIdx.x < 8) sum = smem[threadIdx.x];
  if (threadIdx.x < 32) {
    for (int o = 4; o; o >>= 1) sum += __shfl_xor(sum, o, 32);
    if (threadIdx.x == 0) smem[0] = sum;
  }
  __syncthreads();
  return smem[0];
}

// ---- 6a. Vectorized RMSNorm (hidden == 2048 fast path) ----
__global__ __launch_bounds__(kNormThreads) void HyV3RmsNormKernel(
    const __hip_bfloat16* __restrict__ h, const __hip_bfloat16* __restrict__ norm_w,
    __hip_bfloat16* __restrict__ out, int hidden, float eps) {
  const int64_t t = blockIdx.x;
  const __hip_bfloat16* hr = h + t * hidden;
  __hip_bfloat16* orow = out + t * hidden;
  if (hidden == 2048) {
    const int v = threadIdx.x;
    const uint4 uh = reinterpret_cast<const uint4*>(hr)[v];
    const __hip_bfloat16* bh = reinterpret_cast<const __hip_bfloat16*>(&uh);
    float hf[8];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      hf[i] = __bfloat162float(bh[i]);
      ss += hf[i] * hf[i];
    }
    ss = BlockReduceSum256(ss);
    const float rstd = rsqrtf(ss / 2048.0f + eps);
    const uint4 unw = reinterpret_cast<const uint4*>(norm_w)[v];
    const __hip_bfloat16* bnw = reinterpret_cast<const __hip_bfloat16*>(&unw);
    uint4 uo;
    __hip_bfloat16* bo = reinterpret_cast<__hip_bfloat16*>(&uo);
#pragma unroll
    for (int i = 0; i < 8; ++i) bo[i] = __float2bfloat16(hf[i] * rstd * __bfloat162float(bnw[i]));
    reinterpret_cast<uint4*>(orow)[v] = uo;
    return;
  }
  float ss = 0.0f;
  for (int k = threadIdx.x; k < hidden; k += kNormThreads) {
    const float v = __bfloat162float(hr[k]);
    ss += v * v;
  }
  ss = BlockReduceSum256(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);
  for (int k = threadIdx.x; k < hidden; k += kNormThreads) {
    orow[k] = __float2bfloat16(__bfloat162float(hr[k]) * rstd * __bfloat162float(norm_w[k]));
  }
}

// ---- 6b. Vectorized residual add + RMSNorm (hidden == 2048 fast path) ----
__global__ __launch_bounds__(kNormThreads) void HyV3ResidualRmsNormKernel(
    __hip_bfloat16* __restrict__ h, const __hip_bfloat16* __restrict__ sub,
    const __hip_bfloat16* __restrict__ norm_w, __hip_bfloat16* __restrict__ out,
    int hidden, float eps) {
  const int64_t t = blockIdx.x;
  __hip_bfloat16* hr = h + t * hidden;
  const __hip_bfloat16* sr = sub + t * hidden;
  __hip_bfloat16* orow = out + t * hidden;
  if (hidden == 2048) {
    const int v = threadIdx.x;
    const uint4 uh = reinterpret_cast<const uint4*>(hr)[v];
    const uint4 us = reinterpret_cast<const uint4*>(sr)[v];
    const __hip_bfloat16* bh = reinterpret_cast<const __hip_bfloat16*>(&uh);
    const __hip_bfloat16* bs = reinterpret_cast<const __hip_bfloat16*>(&us);
    uint4 uh_out;
    __hip_bfloat16* bho = reinterpret_cast<__hip_bfloat16*>(&uh_out);
    float hf[8];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const __hip_bfloat16 hn = __float2bfloat16(__bfloat162float(bh[i]) + __bfloat162float(bs[i]));
      bho[i] = hn;
      hf[i] = __bfloat162float(hn);
      ss += hf[i] * hf[i];
    }
    reinterpret_cast<uint4*>(hr)[v] = uh_out;
    ss = BlockReduceSum256(ss);
    const float rstd = rsqrtf(ss / 2048.0f + eps);
    const uint4 unw = reinterpret_cast<const uint4*>(norm_w)[v];
    const __hip_bfloat16* bnw = reinterpret_cast<const __hip_bfloat16*>(&unw);
    uint4 uo;
    __hip_bfloat16* bo = reinterpret_cast<__hip_bfloat16*>(&uo);
#pragma unroll
    for (int i = 0; i < 8; ++i) bo[i] = __float2bfloat16(hf[i] * rstd * __bfloat162float(bnw[i]));
    reinterpret_cast<uint4*>(orow)[v] = uo;
    return;
  }
  float ss = 0.0f;
  for (int k = threadIdx.x; k < hidden; k += kNormThreads) {
    const __hip_bfloat16 hn = __float2bfloat16(__bfloat162float(hr[k]) + __bfloat162float(sr[k]));
    hr[k] = hn;
    const float v = __bfloat162float(hn);
    ss += v * v;
  }
  ss = BlockReduceSum256(ss);
  const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);
  for (int k = threadIdx.x; k < hidden; k += kNormThreads) {
    orow[k] = __float2bfloat16(__bfloat162float(hr[k]) * rstd * __bfloat162float(norm_w[k]));
  }
}

__global__ __launch_bounds__(kNormThreads) void HyV3MoeCombineResidualRmsNormKernel(
    const __hip_bfloat16* __restrict__ y, const int32_t* __restrict__ pos_of_pair,
    const float* __restrict__ w, __hip_bfloat16* __restrict__ h,
    const __hip_bfloat16* __restrict__ next_norm_w, __hip_bfloat16* __restrict__ x_next,
    int slots, int hidden, float eps) {
  const int64_t t = blockIdx.x;
  __hip_bfloat16* hr = h + t * hidden;

  // Fast vectorized path for hidden == 2048 (256 threads x 8 bf16 = 2048, keeps hn[8] in registers!).
  if (hidden == 2048) {
    const int v = threadIdx.x;  // 0..255 -> elements [8v..8v+7]
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    for (int s = 0; s < slots; ++s) {
      const int64_t p = t * slots + s;
      const int64_t row = pos_of_pair ? static_cast<int64_t>(pos_of_pair[p]) : p;
      if (row < 0) continue;
      const float ws = w[p];
      const uint4 uy = reinterpret_cast<const uint4*>(y + row * 2048)[v];
      const __hip_bfloat16* by = reinterpret_cast<const __hip_bfloat16*>(&uy);
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] += ws * __bfloat162float(by[i]);
    }
    const uint4 uh = reinterpret_cast<const uint4*>(hr)[v];
    const __hip_bfloat16* bh = reinterpret_cast<const __hip_bfloat16*>(&uh);
    uint4 uh_out;
    __hip_bfloat16* bho = reinterpret_cast<__hip_bfloat16*>(&uh_out);
    float hf[8];
    float ss = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const __hip_bfloat16 comb = __float2bfloat16(acc[i]);
      const __hip_bfloat16 hn = __float2bfloat16(__bfloat162float(bh[i]) + __bfloat162float(comb));
      bho[i] = hn;
      hf[i] = __bfloat162float(hn);
      ss += hf[i] * hf[i];
    }
    reinterpret_cast<uint4*>(hr)[v] = uh_out;
    if (next_norm_w != nullptr && x_next != nullptr) {
      ss = BlockReduceSum256(ss);
      const float rstd = rsqrtf(ss / 2048.0f + eps);
      const uint4 unw = reinterpret_cast<const uint4*>(next_norm_w)[v];
      const __hip_bfloat16* bnw = reinterpret_cast<const __hip_bfloat16*>(&unw);
      uint4 ux_out;
      __hip_bfloat16* bxo = reinterpret_cast<__hip_bfloat16*>(&ux_out);
#pragma unroll
      for (int i = 0; i < 8; ++i) bxo[i] = __float2bfloat16(hf[i] * rstd * __bfloat162float(bnw[i]));
      reinterpret_cast<uint4*>(x_next + t * 2048)[v] = ux_out;
    }
    return;
  }

  float ss = 0.0f;
  for (int k = threadIdx.x; k < hidden; k += kNormThreads) {
    float acc = 0.0f;
    for (int s = 0; s < slots; ++s) {
      const int64_t p = t * slots + s;
      const int64_t row = pos_of_pair ? static_cast<int64_t>(pos_of_pair[p]) : p;
      if (row < 0) continue;
      acc += w[p] * __bfloat162float(y[row * hidden + k]);
    }
    const __hip_bfloat16 comb = __float2bfloat16(acc);
    const __hip_bfloat16 hn = __float2bfloat16(__bfloat162float(hr[k]) + __bfloat162float(comb));
    hr[k] = hn;
    if (next_norm_w != nullptr) {
      const float v = __bfloat162float(hn);
      ss += v * v;
    }
  }
  if (next_norm_w != nullptr && x_next != nullptr) {
    ss = BlockReduceSum256(ss);
    const float rstd = rsqrtf(ss / static_cast<float>(hidden) + eps);
    __hip_bfloat16* outr = x_next + t * hidden;
    for (int k = threadIdx.x; k < hidden; k += kNormThreads) {
      outr[k] = __float2bfloat16(__bfloat162float(hr[k]) * rstd * __bfloat162float(next_norm_w[k]));
    }
  }
}

template <int G>
void LaunchW4MoeDecode(int64_t a, int64_t wq_ptrs, int64_t wsz_ptrs, int64_t ids, int64_t c,
                       int n_pairs, int a_row_group, int n_experts, int K, int N, int WV, int SK,
                       int NPW, int NT, int64_t stream) {
  const int ntiles = N / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, 1, n_pairs), block(WV * SK * kWave);
  const size_t smem = (SK == 1) ? 0 : static_cast<size_t>(ncols) * SK * 16 * sizeof(float);
  auto Ap = reinterpret_cast<const unsigned short*>(a);
  auto Wqp = reinterpret_cast<const uint64_t*>(wq_ptrs);
  auto Wszp = reinterpret_cast<const uint64_t*>(wsz_ptrs);
  auto Idp = reinterpret_cast<const int32_t*>(ids);
  auto Cp = reinterpret_cast<__hip_bfloat16*>(c);
  hipStream_t st = reinterpret_cast<hipStream_t>(stream);
#define W4_DECODE_L(NP, TT) \
  hipLaunchKernelGGL((W4a16MoeDecodeKernel<G, NP, TT>), grid, block, smem, st, Ap, Wqp, Wszp, Idp, Cp, a_row_group, n_experts, K, N, WV, SK)
  if (NPW == 1) {
    if (NT) { W4_DECODE_L(1, true); } else { W4_DECODE_L(1, false); }
  } else {
    if (NT) { W4_DECODE_L(4, true); } else { W4_DECODE_L(4, false); }
  }
#undef W4_DECODE_L
  R4DX_HIP_CHECK(hipGetLastError());
}

template <int G>
void LaunchW4MoePrefill(int64_t a, int64_t wq_ptrs, int64_t wsz_ptrs, int64_t offsets, int64_t c,
                        int n_pairs, int n_experts, int K, int N, int WV, int SK, int NPW, int NT,
                        int64_t stream) {
  const int ntiles = N / 16;
  const int ncols = WV * NPW;
  const int max_m_tiles = (n_pairs + 15) / 16 + n_experts;
  dim3 grid((ntiles + ncols - 1) / ncols, max_m_tiles), block(WV * SK * kWave);
  const size_t smem = (SK == 1) ? 0 : static_cast<size_t>(ncols) * SK * 256 * sizeof(float);
  auto Ap = reinterpret_cast<const unsigned short*>(a);
  auto Wqp = reinterpret_cast<const uint64_t*>(wq_ptrs);
  auto Wszp = reinterpret_cast<const uint64_t*>(wsz_ptrs);
  auto Offp = reinterpret_cast<const int32_t*>(offsets);
  auto Cp = reinterpret_cast<__hip_bfloat16*>(c);
  hipStream_t st = reinterpret_cast<hipStream_t>(stream);
#define W4_PREFILL_L(NP, TT) \
  hipLaunchKernelGGL((W4a16MoePrefillKernel<G, NP, TT>), grid, block, smem, st, Ap, Wqp, Wszp, Offp, Cp, n_pairs, n_experts, K, N, WV, SK)
  if (NPW == 1) {
    if (NT) { W4_PREFILL_L(1, true); } else { W4_PREFILL_L(1, false); }
  } else {
    if (NT) { W4_PREFILL_L(4, true); } else { W4_PREFILL_L(4, false); }
  }
#undef W4_PREFILL_L
  R4DX_HIP_CHECK(hipGetLastError());
}

template <int G>
void LaunchW4Bf16AM1(int64_t a, int64_t wq, int64_t wsz, int64_t c, int K, int N, int WV, int SK,
                     int NPW, int NT, int64_t stream) {
  const int ntiles = N / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, 1), block(WV * SK * kWave);
  const size_t smem = (SK == 1) ? 0 : static_cast<size_t>(ncols) * SK * 16 * sizeof(float);
  auto Ap = reinterpret_cast<const unsigned short*>(a);
  auto Wqp = reinterpret_cast<const unsigned*>(wq);
  auto Wszp = reinterpret_cast<const unsigned*>(wsz);
  auto Cp = reinterpret_cast<__hip_bfloat16*>(c);
  hipStream_t st = reinterpret_cast<hipStream_t>(stream);
#define W4_M1_L(NP, TT) \
  hipLaunchKernelGGL((W4a16Bf16AM1Kernel<G, NP, TT>), grid, block, smem, st, Ap, Wqp, Wszp, Cp, K, N, WV, SK)
  if (NPW == 1) {
    if (NT) { W4_M1_L(1, true); } else { W4_M1_L(1, false); }
  } else {
    if (NT) { W4_M1_L(4, true); } else { W4_M1_L(4, false); }
  }
#undef W4_M1_L
  R4DX_HIP_CHECK(hipGetLastError());
}

}  // namespace

extern "C" {

void r4dx_hy_v3_attn_precore(int64_t q, int64_t k, int64_t v, int64_t q_norm, int64_t k_norm,
                             int64_t pos_ids, int64_t kv_cache, int kv_fp8,
                             int64_t k_descale, int64_t v_descale,
                             int T, int heads_q, int heads_k, float rms_eps, float rope_theta,
                             int block_size, int64_t kv_block_stride, int64_t kv_head_stride,
                             int64_t stream) {
  if (T <= 0) return;
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(HyV3AttnPrecoreKernel, dim3(static_cast<unsigned>(T), static_cast<unsigned>(heads_q + heads_k)),
                     dim3(128), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<__hip_bfloat16*>(q), reinterpret_cast<__hip_bfloat16*>(k),
                     reinterpret_cast<const __hip_bfloat16*>(v), reinterpret_cast<const __hip_bfloat16*>(q_norm),
                     reinterpret_cast<const __hip_bfloat16*>(k_norm), reinterpret_cast<const int32_t*>(pos_ids),
                     reinterpret_cast<void*>(kv_cache), kv_fp8, reinterpret_cast<const float*>(k_descale),
                     reinterpret_cast<const float*>(v_descale), heads_q, heads_k, rms_eps, rope_theta,
                     block_size, kv_block_stride, kv_head_stride);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_hy_v3_router_f32(int64_t x, int64_t gate_w, int64_t expert_bias, int64_t ids, int64_t w,
                           int64_t sig_scratch, int T, int hidden, int n_experts, int top_k,
                           float routed_scaling_factor, int append_shared, int64_t stream) {
  if (T <= 0) return;
  if (n_experts < 1 || n_experts > kMaxExperts) Fail("r4dx_hy_v3_router_f32", "n_experts out of range");
  if (top_k < 1 || top_k > kMaxTopK || top_k > n_experts) Fail("r4dx_hy_v3_router_f32", "top_k out of range");
  hipStream_t st = reinterpret_cast<hipStream_t>(stream);
  if (T == 1 && sig_scratch != 0 && (hidden & 7) == 0) {
    r4dx_kernel_launch_counter_add(2);
    hipLaunchKernelGGL(HyV3RouterGemvM1Kernel, dim3(static_cast<unsigned>(n_experts)), dim3(32), 0, st,
                       reinterpret_cast<const __hip_bfloat16*>(x),
                       reinterpret_cast<const __hip_bfloat16*>(gate_w),
                       reinterpret_cast<float*>(sig_scratch), hidden, n_experts);
    R4DX_HIP_CHECK(hipGetLastError());
    hipLaunchKernelGGL(HyV3RouterTopkM1Kernel, dim3(1), dim3(32), 0, st,
                       reinterpret_cast<const float*>(sig_scratch),
                       reinterpret_cast<const float*>(expert_bias),
                       reinterpret_cast<int32_t*>(ids), reinterpret_cast<float*>(w),
                       n_experts, top_k, routed_scaling_factor, append_shared);
    R4DX_HIP_CHECK(hipGetLastError());
    return;
  }
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(HyV3RouterKernel, dim3(static_cast<unsigned>(T)), dim3(kRouterThreads),
                     static_cast<size_t>(hidden) * sizeof(float), st,
                     reinterpret_cast<const __hip_bfloat16*>(x), reinterpret_cast<const __hip_bfloat16*>(gate_w),
                     reinterpret_cast<const float*>(expert_bias), reinterpret_cast<int32_t*>(ids),
                     reinterpret_cast<float*>(w), hidden, n_experts, top_k, routed_scaling_factor,
                     append_shared);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_gemm_w4a16_qkv_bf16a_m1(int64_t a_bf16,
                                  int64_t wq_q, int64_t wsz_q, int64_t c_q, int Nq,
                                  int64_t wq_k, int64_t wsz_k, int64_t c_k, int Nk,
                                  int64_t wq_v, int64_t wsz_v, int64_t c_v, int Nv,
                                  int K, int group, int WV, int SK, int NT, int64_t stream) {
  r4dx_kernel_launch_counter_add(1);
  const int total_nt = (Nq + Nk + Nv) / 16;
  dim3 grid((total_nt + WV - 1) / WV, 1), block(WV * SK * kWave);
  const size_t smem = (SK == 1) ? 0 : static_cast<size_t>(WV) * SK * 16 * sizeof(float);
  hipStream_t st = reinterpret_cast<hipStream_t>(stream);
#define QKV_L(G, TT) \
  hipLaunchKernelGGL((W4a16QkvBf16AM1Kernel<G, TT>), grid, block, smem, st, \
                     reinterpret_cast<const unsigned short*>(a_bf16), \
                     reinterpret_cast<const unsigned*>(wq_q), reinterpret_cast<const unsigned*>(wsz_q), reinterpret_cast<__hip_bfloat16*>(c_q), Nq, \
                     reinterpret_cast<const unsigned*>(wq_k), reinterpret_cast<const unsigned*>(wsz_k), reinterpret_cast<__hip_bfloat16*>(c_k), Nk, \
                     reinterpret_cast<const unsigned*>(wq_v), reinterpret_cast<const unsigned*>(wsz_v), reinterpret_cast<__hip_bfloat16*>(c_v), Nv, \
                     K, WV, SK)
  if (group == 64) {
    if (NT) { QKV_L(64, true); } else { QKV_L(64, false); }
  } else if (group == 32) {
    if (NT) { QKV_L(32, true); } else { QKV_L(32, false); }
  } else {
    Fail("r4dx_gemm_w4a16_qkv_bf16a_m1", "unsupported group " + std::to_string(group));
  }
#undef QKV_L
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_gemm_w4a16_moe_gate_up_silu_decode(int64_t a_bf16, int64_t wq_ptrs, int64_t wsz_ptrs,
                                             int64_t ids, int64_t act_bf16, int n_pairs,
                                             int a_row_group, int n_experts, int K, int Im,
                                             int group, int WV, int SK, int NT, int64_t stream) {
  if (n_pairs <= 0) return;
  if (group != 64) Fail("r4dx_gemm_w4a16_moe_gate_up_silu_decode", "only group 64 supported");
  r4dx_kernel_launch_counter_add(1);
  const int ntiles = Im / 16;
  dim3 grid((ntiles + WV - 1) / WV, 1, n_pairs), block(WV * SK * kWave);
  const size_t smem = (SK == 1) ? 0 : static_cast<size_t>(WV) * SK * 32 * sizeof(float);
  hipStream_t st = reinterpret_cast<hipStream_t>(stream);
  if (NT) {
    hipLaunchKernelGGL((W4a16MoeGateUpSiluDecodeKernel<64, true>), grid, block, smem, st,
                       reinterpret_cast<const unsigned short*>(a_bf16),
                       reinterpret_cast<const uint64_t*>(wq_ptrs),
                       reinterpret_cast<const uint64_t*>(wsz_ptrs),
                       reinterpret_cast<const int32_t*>(ids),
                       reinterpret_cast<__hip_bfloat16*>(act_bf16),
                       a_row_group, n_experts, K, Im, WV, SK);
  } else {
    hipLaunchKernelGGL((W4a16MoeGateUpSiluDecodeKernel<64, false>), grid, block, smem, st,
                       reinterpret_cast<const unsigned short*>(a_bf16),
                       reinterpret_cast<const uint64_t*>(wq_ptrs),
                       reinterpret_cast<const uint64_t*>(wsz_ptrs),
                       reinterpret_cast<const int32_t*>(ids),
                       reinterpret_cast<__hip_bfloat16*>(act_bf16),
                       a_row_group, n_experts, K, Im, WV, SK);
  }
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_gemm_w4a16_moe_decode(int64_t a_bf16, int64_t wq_ptrs, int64_t wsz_ptrs, int64_t ids,
                                int64_t c_bf16, int n_pairs, int a_row_group, int n_experts,
                                int K, int N, int group, int WV, int SK, int NPW, int NT,
                                int64_t stream) {
  if (n_pairs <= 0) return;
  r4dx_kernel_launch_counter_add(1);
  if (group == 64) {
    LaunchW4MoeDecode<64>(a_bf16, wq_ptrs, wsz_ptrs, ids, c_bf16, n_pairs, a_row_group, n_experts, K, N, WV, SK, NPW, NT, stream);
  } else if (group == 32) {
    LaunchW4MoeDecode<32>(a_bf16, wq_ptrs, wsz_ptrs, ids, c_bf16, n_pairs, a_row_group, n_experts, K, N, WV, SK, NPW, NT, stream);
  } else {
    Fail("r4dx_gemm_w4a16_moe_decode", "unsupported group " + std::to_string(group));
  }
}

void r4dx_gemm_w4a16_moe_down_combine_res_decode(int64_t act_bf16, int64_t wq_ptrs, int64_t wsz_ptrs,
                                                 int64_t ids, int64_t wts, int64_t h_bf16,
                                                 int slots, int n_experts, int K, int N,
                                                 int group, int NT, int64_t stream) {
  if (slots <= 0 || slots > 16) Fail("r4dx_gemm_w4a16_moe_down_combine_res_decode", "slots must be in [1, 16]");
  if (group != 64) Fail("r4dx_gemm_w4a16_moe_down_combine_res_decode", "only group 64 supported");
  r4dx_kernel_launch_counter_add(1);
  dim3 grid(static_cast<unsigned>(N / 16)), block(static_cast<unsigned>(slots * kWave));
  hipStream_t st = reinterpret_cast<hipStream_t>(stream);
  if (NT) {
    hipLaunchKernelGGL((W4a16MoeDownCombineResDecodeKernel<true>), grid, block, 0, st,
                       reinterpret_cast<const unsigned short*>(act_bf16),
                       reinterpret_cast<const uint64_t*>(wq_ptrs),
                       reinterpret_cast<const uint64_t*>(wsz_ptrs),
                       reinterpret_cast<const int32_t*>(ids),
                       reinterpret_cast<const float*>(wts),
                       reinterpret_cast<__hip_bfloat16*>(h_bf16), slots, n_experts, K, N);
  } else {
    hipLaunchKernelGGL((W4a16MoeDownCombineResDecodeKernel<false>), grid, block, 0, st,
                       reinterpret_cast<const unsigned short*>(act_bf16),
                       reinterpret_cast<const uint64_t*>(wq_ptrs),
                       reinterpret_cast<const uint64_t*>(wsz_ptrs),
                       reinterpret_cast<const int32_t*>(ids),
                       reinterpret_cast<const float*>(wts),
                       reinterpret_cast<__hip_bfloat16*>(h_bf16), slots, n_experts, K, N);
  }
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_gemm_w4a16_moe_prefill(int64_t a_bf16, int64_t wq_ptrs, int64_t wsz_ptrs,
                                 int64_t expert_offsets, int64_t c_bf16, int n_pairs,
                                 int n_experts, int K, int N, int group, int WV, int SK, int NPW,
                                 int NT, int64_t stream) {
  if (n_pairs <= 0) return;
  r4dx_kernel_launch_counter_add(1);
  if (group == 64) {
    LaunchW4MoePrefill<64>(a_bf16, wq_ptrs, wsz_ptrs, expert_offsets, c_bf16, n_pairs, n_experts, K, N, WV, SK, NPW, NT, stream);
  } else if (group == 32) {
    LaunchW4MoePrefill<32>(a_bf16, wq_ptrs, wsz_ptrs, expert_offsets, c_bf16, n_pairs, n_experts, K, N, WV, SK, NPW, NT, stream);
  } else {
    Fail("r4dx_gemm_w4a16_moe_prefill", "unsupported group " + std::to_string(group));
  }
}

void r4dx_gemm_w4a16_bf16a_m1(int64_t a_bf16, int64_t wq, int64_t wsz, int64_t c_bf16,
                              int K, int N, int group, int WV, int SK, int NPW, int NT,
                              int64_t stream) {
  r4dx_kernel_launch_counter_add(1);
  if (group == 64) {
    LaunchW4Bf16AM1<64>(a_bf16, wq, wsz, c_bf16, K, N, WV, SK, NPW, NT, stream);
  } else if (group == 32) {
    LaunchW4Bf16AM1<32>(a_bf16, wq, wsz, c_bf16, K, N, WV, SK, NPW, NT, stream);
  } else {
    Fail("r4dx_gemm_w4a16_bf16a_m1", "unsupported group " + std::to_string(group));
  }
}

void r4dx_hy_v3_rmsnorm_bf16(int64_t h, int64_t norm_w, int64_t out,
                             int T, int hidden, float eps, int64_t stream) {
  if (T <= 0 || hidden <= 0) return;
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(HyV3RmsNormKernel, dim3(static_cast<unsigned>(T)),
                     dim3(kNormThreads), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const __hip_bfloat16*>(h),
                     reinterpret_cast<const __hip_bfloat16*>(norm_w),
                     reinterpret_cast<__hip_bfloat16*>(out), hidden, eps);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_hy_v3_residual_rmsnorm_bf16(int64_t h, int64_t sub, int64_t norm_w, int64_t out,
                                      int T, int hidden, float eps, int64_t stream) {
  if (T <= 0 || hidden <= 0) return;
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(HyV3ResidualRmsNormKernel, dim3(static_cast<unsigned>(T)),
                     dim3(kNormThreads), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<__hip_bfloat16*>(h),
                     reinterpret_cast<const __hip_bfloat16*>(sub),
                     reinterpret_cast<const __hip_bfloat16*>(norm_w),
                     reinterpret_cast<__hip_bfloat16*>(out), hidden, eps);
  R4DX_HIP_CHECK(hipGetLastError());
}

void r4dx_hy_v3_moe_combine_residual_rmsnorm_bf16(int64_t y, int64_t pos_of_pair, int64_t w,
                                                  int64_t h, int64_t next_norm_w, int64_t x_next,
                                                  int T, int slots, int hidden, float eps,
                                                  int64_t stream) {
  if (T <= 0 || hidden <= 0) return;
  r4dx_kernel_launch_counter_add(1);
  hipLaunchKernelGGL(HyV3MoeCombineResidualRmsNormKernel, dim3(static_cast<unsigned>(T)),
                     dim3(kNormThreads), 0, reinterpret_cast<hipStream_t>(stream),
                     reinterpret_cast<const __hip_bfloat16*>(y), reinterpret_cast<const int32_t*>(pos_of_pair),
                     reinterpret_cast<const float*>(w), reinterpret_cast<__hip_bfloat16*>(h),
                     reinterpret_cast<const __hip_bfloat16*>(next_norm_w),
                     reinterpret_cast<__hip_bfloat16*>(x_next), slots, hidden, eps);
  R4DX_HIP_CHECK(hipGetLastError());
}

}  // extern "C"
