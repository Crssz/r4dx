// src/model/qwen4exp/kernels/moe_trellis_decode.hip
#include "r4dx/model/qwen4exp/moe_trellis_grouped.h"

#include <hip/hip_runtime.h>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "r4d.h"
#include "r4d_fwht128.h"
#include "r4d_gdn_wmma.h"
#include "r4d_trellis_dq.h"

#define WAVE32 32

namespace {

typedef __attribute__((address_space(1))) const unsigned char* r4d_gbyte;

template <typename T, bool NT>
__device__ __forceinline__ T r4d_tq_load(r4d_gbyte base, unsigned off, int imm) {
  auto p = (__attribute__((address_space(1))) const T*)(base + off + imm);
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

template <int KB, int NP, int U, int MT>
struct r4d_tq_stage;

template <int NP, int U, int MT>
struct r4d_tq_stage<4, NP, U, MT> {
  unsigned long long ab[NP][U];
  unsigned p[NP][U];
  v8h af[MT][U];
};

template <int NP, int U, int MT>
struct r4d_tq_stage<3, NP, U, MT> {
  unsigned w[NP][U][4];
  v8h af[MT][U];
};

__device__ __forceinline__ void r4d_tq_opaque(r4d_gbyte& p) {
  unsigned long long v = (unsigned long long)p;
  asm("" : "+s"(v));
  p = (r4d_gbyte)v;
}

__device__ __forceinline__ void r4d_tq_opaque_v(unsigned& v) {
  asm("" : "+v"(v));
}

__device__ __forceinline__ v8f r4d_tq_mma(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ unsigned short r4d_tq_bf16_rn(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (unsigned short)((u >> 16) | 0x40u);
  return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ void r4d_tq_sync_lds() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
__device__ __forceinline__ float Bf16BitsToFloat(uint16_t b) {
  return __builtin_bit_cast(float, static_cast<uint32_t>(b) << 16);
}

__device__ __forceinline__ float TqSigmoid(float x) {
  return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ void r4d_tq_finish(float (&v)[4], int lane, const float (&sv)[4],
                                              unsigned short* __restrict__ crow, int n0,
                                              float out_scale) {
  r4d_fwht128_wave(v, lane);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    crow[n0 + lane + 32 * r] = r4d_tq_bf16_rn((v[r] * sv[r]) * out_scale);
  }
}
__global__ void GroupedMoeDecodeSortKernel(
    const int32_t* __restrict__ ids_in,
    const float* __restrict__ weights_in,
    int32_t* __restrict__ ids_sorted_out,
    float* __restrict__ weights_sorted_out,
    int k) {
  __shared__ int32_t s_ids[16];
  __shared__ float s_w[16];
  const int lane = static_cast<int>(threadIdx.x);
  if (lane < k) {
    s_ids[lane] = ids_in[lane];
    s_w[lane] = weights_in[lane];
  }
  __syncthreads();
  if (lane < k) {
    const int32_t my_id = s_ids[lane];
    const float my_w = s_w[lane];
    int rank = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      if (i >= k) break;
      const int32_t other_id = s_ids[i];
      if (other_id < my_id || (other_id == my_id && i < lane)) {
        ++rank;
      }
    }
    ids_sorted_out[rank] = my_id;
    weights_sorted_out[rank] = my_w;
  }
}

// ------------------------------------------------------------------------------------------------
// 2. Grouped Trellis Input Transform for gate_up (K=2560, parts=2)
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(32) void GroupedTrellisInputGateUpKernel(
    const uint16_t* __restrict__ x,
    const int32_t* __restrict__ ids_sorted,
    const float* __restrict__ all_suh,
    uint16_t* __restrict__ out_a0,
    uint16_t* __restrict__ out_a1,
    float scale) {
  const int lane = static_cast<int>(threadIdx.x);
  const int k_blk = static_cast<int>(blockIdx.x);  // 0..19 (20 blocks of 128 = 2560)
  const int slot = static_cast<int>(blockIdx.y);   // 0..top_k-1
  const int expert = ids_sorted[slot];

  const int64_t k_off = static_cast<int64_t>(k_blk) * 128 + lane;
  const uint16_t* x_blk = x + k_off;

  float xv[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) xv[r] = Bf16BitsToFloat(x_blk[32 * r]);

  const float* suh0 = all_suh + ((size_t)expert * 5120 + k_off);
  const float* suh1 = suh0 + 2560;

  uint16_t* dst0 = out_a0 + ((size_t)slot * 2560 + k_off);
  uint16_t* dst1 = out_a1 + ((size_t)slot * 2560 + k_off);

  float v0[4], v1[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    v0[r] = xv[r] * suh0[32 * r];
    v1[r] = xv[r] * suh1[32 * r];
  }
  r4d_fwht128_wave(v0, lane);
  r4d_fwht128_wave(v1, lane);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    float p0 = v0[r] * scale;
    float p1 = v1[r] * scale;
    asm("" : "+v"(p0));
    asm("" : "+v"(p1));
    dst0[32 * r] = __builtin_bit_cast(uint16_t, static_cast<_Float16>(p0));
    dst1[32 * r] = __builtin_bit_cast(uint16_t, static_cast<_Float16>(p1));
  }
}

// ------------------------------------------------------------------------------------------------
// 3. Fused silu_mul + Trellis Input Transform for down (K=640, parts=1)
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(32) void GroupedSiluMulTrellisInputDownKernel(
    const uint16_t* __restrict__ gu,
    const int32_t* __restrict__ ids_sorted,
    const float* __restrict__ all_suh_down,
    uint16_t* __restrict__ out_a0,
    float scale) {
  const int lane = static_cast<int>(threadIdx.x);
  const int k_blk = static_cast<int>(blockIdx.x);  // 0..4 (5 blocks of 128 = 640)
  const int slot = static_cast<int>(blockIdx.y);   // 0..top_k-1
  const int expert = ids_sorted[slot];

  const int64_t k_off = static_cast<int64_t>(k_blk) * 128 + lane;
  const uint16_t* gu_expert = gu + ((size_t)slot * 1280 + k_off);

  float xv[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    const float g = Bf16BitsToFloat(gu_expert[32 * r]);
    const float u = Bf16BitsToFloat(gu_expert[640 + 32 * r]);
    const float h = Bf16BitsToFloat(r4d_tq_bf16_rn(g * TqSigmoid(g) * u));
    xv[r] = h;
  }

  const float* suh = all_suh_down + ((size_t)expert * 640 + k_off);
  uint16_t* dst = out_a0 + ((size_t)slot * 640 + k_off);

  float v[4];
#pragma unroll
  for (int r = 0; r < 4; ++r) v[r] = xv[r] * suh[32 * r];
  r4d_fwht128_wave(v, lane);
#pragma unroll
  for (int r = 0; r < 4; ++r) {
    float p = v[r] * scale;
    asm("" : "+v"(p));
    dst[32 * r] = __builtin_bit_cast(uint16_t, static_cast<_Float16>(p));
  }
}

// ------------------------------------------------------------------------------------------------
// 4. Grouped Trellis GEMV M=1 Kernel template
// ------------------------------------------------------------------------------------------------
template <int KB, int K, int N, int n_split, int SK, int SKG, int U,
          size_t w_per, size_t svh_per, size_t ws_per, size_t tickets_per>
__global__ __launch_bounds__(256) void GroupedGemmTrellisM1Kernel(
    const unsigned short* __restrict__ A0_base,
    const unsigned short* __restrict__ A1_base,
    const unsigned* __restrict__ W_base,
    const float* __restrict__ svh_base,
    const int32_t* __restrict__ expert_ids,
    uint16_t* __restrict__ Cv_base,
    float* __restrict__ ws_base,
    unsigned* __restrict__ tickets_base,
    float out_scale) {

  constexpr int WV = 4;
  constexpr int NP = 1;
  constexpr int MT = 1;
  constexpr int BLK = 64 * KB;
  constexpr int Wc = WV * NP * 32;  // 128
  constexpr int kt_total = K / 16;
  constexpr int ktw = kt_total / (SK * SKG);
  constexpr int steps = ktw / U;

  extern __shared__ float red[];
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int col_local = wave / SK, ksplit = wave - col_local * SK;

  const int expert_slot = blockIdx.z;
  const int expert = expert_ids[expert_slot];

  const int n0 = blockIdx.x * Wc;
  const int pair0 = n0 / 32 + col_local * NP;
  const int kt0 = ((int)blockIdx.y * SK + ksplit) * ktw;

  const unsigned* W = W_base + (size_t)expert * w_per;
  const float* svh = svh_base + (size_t)expert * svh_per;
  uint16_t* Cv = Cv_base + (size_t)expert_slot * N;
  float* ws = ws_base + (size_t)expert_slot * ws_per;
  unsigned* tickets = tickets_base + (size_t)expert * tickets_per;

  const unsigned short* A_part = (n0 >= n_split && A1_base != nullptr) ? (A1_base + (size_t)expert_slot * K)
                                                                      : (A0_base + (size_t)expert_slot * K);

  v8f acc[2];
  acc[0] = (v8f)(0.f);
  acc[1] = (v8f)(0.f);

  {
    r4d_gbyte wb = (r4d_gbyte)(const unsigned char*)W + ((size_t)pair0 * kt_total + kt0) * BLK;
    r4d_gbyte abase = (r4d_gbyte)(const unsigned char*)A_part + (size_t)kt0 * 32;

    constexpr int NOFF = KB == 4 ? 2 : KB == 3 ? 4 : 5;
    unsigned woff[NOFF];
    if constexpr (KB == 4) {
      woff[0] = r4d_trellis_k4_off_ab(lane);
      woff[1] = r4d_trellis_k4_off_p(lane);
    } else if constexpr (KB == 3) {
#pragma unroll
      for (int i = 0; i < 4; ++i) woff[i] = r4d_trellis_k3_word(lane, i) * 4u;
    }

    const unsigned k5sh = r4d_trellis_k5_shift(lane);
    unsigned aoff = (4u * (unsigned)(lane >> 4)) * 2u;

    auto load = [&](r4d_tq_stage<KB, 1, U, 1>& s, int ahead) {
      const size_t wo = (size_t)ahead * (U * BLK), ao = (size_t)ahead * (U * 32);
      r4d_gbyte b = wb + wo;
      r4d_tq_opaque(b);
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if constexpr (KB == 4) {
          s.ab[0][u] = r4d_tq_load<unsigned long long, true>(b, woff[0], u * BLK);
          s.p[0][u] = r4d_tq_load<unsigned, true>(b, woff[1], u * BLK);
        } else if constexpr (KB == 3) {
#pragma unroll
          for (int i = 0; i < 4; ++i) s.w[0][u][i] = r4d_tq_load<unsigned, true>(b, woff[i], u * BLK);
        }
      }
      r4d_gbyte a = abase + ao;
      r4d_tq_opaque(a);
#pragma unroll
      for (int u = 0; u < U; ++u) {
        union { unsigned long long w[2]; v8h f; } x;
        x.w[0] = r4d_tq_load<unsigned long long, false>(a, aoff, u * 32);
        x.w[1] = r4d_tq_load<unsigned long long, false>(a, aoff, u * 32 + 16);
        s.af[0][u] = x.f;
      }
    };

    auto advance = [&](int steps_done) {
      wb += (size_t)steps_done * (U * BLK);
      r4d_tq_opaque(wb);
      abase += (size_t)steps_done * (U * 32);
      r4d_tq_opaque(abase);
#pragma unroll
      for (int i = 0; i < NOFF; ++i) r4d_tq_opaque_v(woff[i]);
      r4d_tq_opaque_v(aoff);
    };

    auto compute = [&](r4d_tq_stage<KB, 1, U, 1>& s) {
#pragma unroll
      for (int u = 0; u < U; ++u) {
        v8h f0, f1;
        if constexpr (KB == 4)
          r4d_trellis_k4_decode(s.p[0][u], (unsigned)s.ab[0][u], (unsigned)(s.ab[0][u] >> 32), f0, f1);
        else if constexpr (KB == 3)
          r4d_trellis_k3_decode(s.w[0][u], k5sh, f0, f1);

        acc[0] = r4d_tq_mma(s.af[0][u], f0, acc[0]);
        acc[1] = r4d_tq_mma(s.af[0][u], f1, acc[1]);
      }
    };

    r4d_tq_stage<KB, 1, U, 1> s0, s1;
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
  }

  // Epilogue for M = 1:
  const int nthreads = WV * SK * WAVE32;
  const int pair_col = col_local * 32;

  if (lane < 16) {
    const int t = (lane >> 3) & 1, c = lane & 7;
    red[(ksplit * 8 + 0) * Wc + pair_col + 16 * t + c] = acc[0][0];
    red[(ksplit * 8 + 0) * Wc + pair_col + 16 * t + 8 + c] = acc[1][0];
  }
  r4d_tq_sync_lds();

  for (int idx = tid; idx < Wc; idx += nthreads) {
    float v = red[(0 * 8 + 0) * Wc + idx] + red[(1 * 8 + 0) * Wc + idx];
    ws[(size_t)blockIdx.y * N + n0 + idx] = v;
  }

  __threadfence();
  __syncthreads();
  if (tid == 0) {
    const unsigned old = atomicAdd(&tickets[n0 / 128], 1u);
    const bool last = (old == (unsigned)SKG - 1);
    if (last) tickets[n0 / 128] = 0u;
    red[0] = last ? 1.f : 0.f;
  }
  __syncthreads();
  if (red[0] == 0.f) return;

  __threadfence();

  if (wave == 0) {
    float v[4];
    float sv[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int n = n0 + lane + 32 * q;
      float x = 0.f;
#pragma unroll
      for (int y = 0; y < SKG; ++y) x += ws[(size_t)y * N + n];
      v[q] = x;
      sv[q] = svh[n];
    }
    r4d_tq_finish(v, lane, sv, Cv, n0, out_scale);
  }
}

// ------------------------------------------------------------------------------------------------
// 5. Grouped Combine Kernel
// ------------------------------------------------------------------------------------------------
__global__ __launch_bounds__(256) void GroupedMoeDecodeCombineKernel(
    const uint16_t* __restrict__ yp,
    const float* __restrict__ weights_sorted,
    const uint16_t* __restrict__ shared_y,
    const uint16_t* __restrict__ shared_gate_logit,
    uint16_t* __restrict__ out,
    int top_k, int hidden) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= hidden) return;
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    if (j >= top_k) break;
    acc += Bf16BitsToFloat(yp[(size_t)j * hidden + d]) * weights_sorted[j];
  }
  if (shared_y != nullptr) {
    const float gate = Bf16BitsToFloat(r4d_tq_bf16_rn(TqSigmoid(Bf16BitsToFloat(shared_gate_logit[0]))));
    acc += Bf16BitsToFloat(r4d_tq_bf16_rn(gate * Bf16BitsToFloat(shared_y[d])));
  }
  out[d] = r4d_tq_bf16_rn(acc);
}

}  // namespace

namespace r4dx {
namespace model {
namespace qwen4exp {

void GroupedMoeDecodeSort(const int32_t* ids_in, const float* weights_in,
                          int32_t* ids_sorted_out, float* weights_sorted_out,
                          int top_k, void* stream) {
  if (top_k <= 0) return;
  hipLaunchKernelGGL(GroupedMoeDecodeSortKernel, dim3(1), dim3(32), 0, static_cast<hipStream_t>(stream),
                     ids_in, weights_in, ids_sorted_out, weights_sorted_out, top_k);
}

void GroupedTrellisInputGateUp(const uint16_t* x, const int32_t* ids_sorted,
                              const float* all_suh, uint16_t* out_a0, uint16_t* out_a1,
                              int top_k, void* stream) {
  if (top_k <= 0) return;
  const float scale = static_cast<float>(1.0 / std::sqrt(128.0));
  hipLaunchKernelGGL(GroupedTrellisInputGateUpKernel, dim3(20, static_cast<unsigned>(top_k)), dim3(32), 0,
                     static_cast<hipStream_t>(stream), x, ids_sorted, all_suh, out_a0, out_a1, scale);
}

void GroupedTrellisGemvGateUp(int kb, const uint16_t* in_a0, const uint16_t* in_a1,
                             const int32_t* ids_sorted,
                             const uint32_t* all_w, const float* all_svh,
                             uint16_t* out_c, float* ws, unsigned* all_tickets,
                             int top_k, void* stream) {
  if (top_k <= 0) return;
  const float out_scale = static_cast<float>(1.0 / std::sqrt(128.0));
  const size_t smem = 2 * 128 * 8 * sizeof(float);
  const dim3 grid(10, 4, static_cast<unsigned>(top_k));
  const dim3 block(256);
  const auto st = static_cast<hipStream_t>(stream);

  if (kb == 3) {
    constexpr size_t w_per = 307200;
    constexpr size_t svh_per = 1280;
    constexpr size_t ws_per = 5120;
    constexpr size_t tickets_per = 10;
    hipLaunchKernelGGL((GroupedGemmTrellisM1Kernel<3, 2560, 1280, 640, 2, 4, 2, w_per, svh_per, ws_per, tickets_per>),
                       grid, block, smem, st,
                       (const unsigned short*)in_a0, (const unsigned short*)in_a1,
                       all_w, all_svh, ids_sorted, out_c, ws, all_tickets, out_scale);
  } else if (kb == 4) {
    constexpr size_t w_per = 409600;
    constexpr size_t svh_per = 1280;
    constexpr size_t ws_per = 5120;
    constexpr size_t tickets_per = 10;
    hipLaunchKernelGGL((GroupedGemmTrellisM1Kernel<4, 2560, 1280, 640, 2, 4, 2, w_per, svh_per, ws_per, tickets_per>),
                       grid, block, smem, st,
                       (const unsigned short*)in_a0, (const unsigned short*)in_a1,
                       all_w, all_svh, ids_sorted, out_c, ws, all_tickets, out_scale);
  } else {
    throw std::runtime_error("GroupedTrellisGemvGateUp: unsupported KB " + std::to_string(kb));
  }
}

void GroupedSiluMulTrellisInputDown(const uint16_t* gu, const int32_t* ids_sorted,
                                   const float* all_suh_down, uint16_t* out_a0,
                                   int top_k, void* stream) {
  if (top_k <= 0) return;
  const float scale = static_cast<float>(1.0 / std::sqrt(128.0));
  hipLaunchKernelGGL(GroupedSiluMulTrellisInputDownKernel, dim3(5, static_cast<unsigned>(top_k)), dim3(32), 0,
                     static_cast<hipStream_t>(stream), gu, ids_sorted, all_suh_down, out_a0, scale);
}

void GroupedTrellisGemvDown(int kb, const uint16_t* in_a0,
                           const int32_t* ids_sorted,
                           const uint32_t* all_w_down, const float* all_svh_down,
                           uint16_t* out_yp, float* ws, unsigned* all_tickets_down,
                           int top_k, void* stream) {
  if (top_k <= 0) return;
  const float out_scale = static_cast<float>(1.0 / std::sqrt(128.0));
  const size_t smem = 2 * 128 * 8 * sizeof(float);
  const dim3 grid(20, 2, static_cast<unsigned>(top_k));
  const dim3 block(256);
  const auto st = static_cast<hipStream_t>(stream);

  if (kb == 3) {
    constexpr size_t w_per = 153600;
    constexpr size_t svh_per = 2560;
    constexpr size_t ws_per = 5120;
    constexpr size_t tickets_per = 20;
    hipLaunchKernelGGL((GroupedGemmTrellisM1Kernel<3, 640, 2560, 2560, 2, 2, 2, w_per, svh_per, ws_per, tickets_per>),
                       grid, block, smem, st,
                       (const unsigned short*)in_a0, nullptr,
                       all_w_down, all_svh_down, ids_sorted, out_yp, ws, all_tickets_down, out_scale);
  } else if (kb == 4) {
    constexpr size_t w_per = 204800;
    constexpr size_t svh_per = 2560;
    constexpr size_t ws_per = 5120;
    constexpr size_t tickets_per = 20;
    hipLaunchKernelGGL((GroupedGemmTrellisM1Kernel<4, 640, 2560, 2560, 2, 2, 2, w_per, svh_per, ws_per, tickets_per>),
                       grid, block, smem, st,
                       (const unsigned short*)in_a0, nullptr,
                       all_w_down, all_svh_down, ids_sorted, out_yp, ws, all_tickets_down, out_scale);
  } else {
    throw std::runtime_error("GroupedTrellisGemvDown: unsupported KB " + std::to_string(kb));
  }
}

void GroupedMoeDecodeCombine(const uint16_t* yp, const float* weights_sorted,
                            const uint16_t* shared_y, const uint16_t* shared_gate_logit,
                            uint16_t* out, int top_k, int hidden, void* stream) {
  if (top_k <= 0 || hidden <= 0) return;
  const unsigned blocks = static_cast<unsigned>((hidden + 255) / 256);
  hipLaunchKernelGGL(GroupedMoeDecodeCombineKernel, dim3(blocks), dim3(256), 0, static_cast<hipStream_t>(stream),
                     yp, weights_sorted, shared_y, shared_gate_logit, out, top_k, hidden);
}

}  // namespace qwen4exp
}  // namespace model
}  // namespace r4dx
