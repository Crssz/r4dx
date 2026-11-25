// mxfp8_gemm.hip -- r4dx_gemm_mxfp8_nt (docs/inkling2-plan.md 4.1; ABI and numerics in
// include/r4dx/kernels/mxfp8_gemm.h). C[M,N] bf16 = A[M,K] bf16 @ dequant(W)[N,K]^T, M <= 256.
//
// A copy of w8a16_gemm.hip's w8a16_gemm_nt_m64_kernel (whose header and docs/w8a16.md 3-4 are the rationale for
// every structural choice: f16 WMMA, per-group f32 epilogue, NPW, clamped row / column tiles, the fixed-order
// in-block split-K) with these changes:
//
//  1. GROUP = ONE PACKED BLOCK. The E8M0 scale group is 32 K, exactly one 512-byte packed block (16 rows x 32 K, one
//     global_load_b128 per lane, two k steps). The group loop runs two groups per iteration (both groups' weight
//     and scale loads in flight before the first WMMA), plus a tail group when the split has an odd count.
//  2. UNPACK. e4m3fn byte b = s eeee mmm. v_perm_b32 puts b into the HIGH byte of a 16-bit half (b << 8); then
//     ((x >> 1) & 0x3F80) | (x & 0x8000) is the f16 with sign s, exponent field eeee and mantissa mmm0000000, i.e.
//     exactly value(b) * 2^-8 (an f16 subnormal when e == 0, since an e4m3 subnormal m * 2^-9 becomes m * 2^-17);
//     one v_pk_mul_f16 by 2^8 makes every value the exact normal f16 it is (|value| in [2^-9, 448]). 4 half pairs
//     per fragment: per pair 1 perm + shift/and/or + 1 pk_mul. No table, no NaN handling (never written).
//  3. SCALE. One byte per (row, group): float(2^(byte - 127)) built from its bits (byte 0 -> the f32 subnormal
//     2^-127), applied as acc = fmaf(g_acc, sc, acc) at the group's end. There is no output unbias: the scale is the
//     true power of two.
//  4. M up to 256: the grid's y covers ceil(M/16) row tiles in groups of MB, as w8a16's does at M <= 64.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "r4dx/kernels/mxfp8_gemm.h"
#include "r4dx/kernels/kernels.h"  // r4dx_kernel_launch_counter_add
#include "r4d_gdn_wmma.h"          // rowGo, v8f

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int kMaxM = 256;
constexpr int kGroup = 32;  // K per scale = K per packed block: 2 k steps, 16 bytes per lane
constexpr int kWave = 32;
constexpr int kMaxMoePairs = 16384;
constexpr int kMaxMoeExperts = 256;

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));
typedef unsigned uint2v __attribute__((ext_vector_type(2)));

typedef __attribute__((address_space(1))) const uint4v* gw4;
typedef __attribute__((address_space(1))) const uint2v* gw2;
typedef __attribute__((address_space(1))) const unsigned long long* gu2;
typedef __attribute__((address_space(1))) const unsigned char* gub;

// Two bf16 (low half = first element) -> two f16, RTZ (w8a16_gemm.hip's).
__device__ __forceinline__ unsigned bf2_to_h2(unsigned d) {
  const float lo = __uint_as_float(d << 16);
  const float hi = __uint_as_float(d & 0xffff0000u);
  union { decltype(__builtin_amdgcn_cvt_pkrtz(0.f, 0.f)) h; unsigned u; } x;
  x.h = __builtin_amdgcn_cvt_pkrtz(lo, hi);
  return x.u;
}

__device__ __forceinline__ v8h fragA_bf16(gu2 p, int u) {
  const unsigned long long w0 = p[4 * u];
  const unsigned long long w1 = p[4 * u + 2];
  union { unsigned u[4]; v8h f; } x;
  x.u[0] = bf2_to_h2((unsigned)w0);
  x.u[1] = bf2_to_h2((unsigned)(w0 >> 32));
  x.u[2] = bf2_to_h2((unsigned)w1);
  x.u[3] = bf2_to_h2((unsigned)(w1 >> 32));
  return x.f;
}

// Bytes (sel picks two of d's bytes into bits 8..15 and 24..31) -> two f16 of value * 2^-8, bit exact.
__device__ __forceinline__ unsigned e4m3x2_scaled(unsigned d, unsigned sel) {
  const unsigned x = __builtin_amdgcn_perm(0u, d, sel);
  return ((x >> 1) & 0x3F803F80u) | (x & 0x80008000u);
}

// Two dwords of four e4m3 codes (elements e0..e3, e4..e7) -> one f16 B fragment scaled by 2^-8.
// The 2^8 factor is folded into the group's E8M0 power-of-two scale (e8m0_value_p8), saving 4 v_pk_mul_f16 per 8 weights.
__device__ __forceinline__ v8h dequant8(unsigned d0, unsigned d1) {
  union { unsigned u[4]; v8h f; } x;
  x.u[0] = e4m3x2_scaled(d0, 0x010C000Cu);  // bytes 0, 1 -> e0, e1
  x.u[1] = e4m3x2_scaled(d0, 0x030C020Cu);  // bytes 2, 3 -> e2, e3
  x.u[2] = e4m3x2_scaled(d1, 0x010C000Cu);  // e4, e5
  x.u[3] = e4m3x2_scaled(d1, 0x030C020Cu);  // e6, e7
  return x.f;
}

// E8M0 byte -> 2^(b - 127 + 8) as f32 when b <= 246 (exact, including b == 0 -> 2^-119).
// For the rare extreme b in [247, 254] (where 2^(b - 119) overflows f32), encode -2^(b - 127)
// in the sign bit so group_step multiplies g_acc by 256.f before applying 2^(b - 127).
__device__ __forceinline__ float e8m0_value(unsigned b) {
  return __uint_as_float(b <= 246u ? ((b + 8u) << 23) : ((b << 23) | 0x80000000u));
}

__device__ __forceinline__ v8f mmah(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

// One 32-K group of NPW column tiles x MT row tiles: two WMMAs into g_acc (the first with a zero C), then the
// scaled fold into acc. `wv` holds the group's packed block per column tile, `sc` its scales, `u0` the k step of
// the group within the split (pa advances by 2 k steps per group).
template <int MT, int NPW>
__device__ __forceinline__ void group_step(v8f (&acc)[NPW][MT], const uint4v (&wv)[NPW], const float (&sc)[NPW],
                                           const gu2 (&pa)[MT], int u0) {
  v8f g_acc[NPW][MT];
#pragma unroll
  for (int s = 0; s < 2; s++) {
    v8h afrag[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) afrag[i] = fragA_bf16(pa[i], u0 + s);
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const v8h bfrag = dequant8(wv[j][2 * s], wv[j][2 * s + 1]);
#pragma unroll
      for (int i = 0; i < MT; i++)
        g_acc[j][i] = s == 0 ? mmah(afrag[i], bfrag, (v8f)(0.f)) : mmah(afrag[i], bfrag, g_acc[j][i]);
    }
  }
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    if (__builtin_expect(sc[j] < 0.0f, 0)) {
      const float s_pos = -sc[j];
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int e = 0; e < 8; e++) acc[j][i][e] = fmaf(g_acc[j][i][e] * 256.0f, s_pos, acc[j][i][e]);
    } else {
#pragma unroll
      for (int i = 0; i < MT; i++)
#pragma unroll
        for (int e = 0; e < 8; e++) acc[j][i][e] = fmaf(g_acc[j][i][e], sc[j], acc[j][i][e]);
    }
  }
}

template <int MT, int NPW, bool NT>
__global__ __launch_bounds__(1024) void mxfp8_gemm_nt_kernel(const unsigned short* __restrict__ A,
                                                            const unsigned char* __restrict__ Wq,
                                                            const unsigned char* __restrict__ Ws,
                                                            __hip_bfloat16* __restrict__ C, int M, int K, int N,
                                                            int WV, int SK) {
  extern __shared__ float red[];  // WV*NPW*SK*256 floats
  const int tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / kGroup;
  const int kbase = ksplit * Ks;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  if (n0 < N) {
    gu2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) {
      const unsigned off = rowGo((unsigned)K, (mt0 + i) * 16, M - 1, lane) + (unsigned)kbase;
      pa[i] = (gu2)(const unsigned long long*)(A + off);
    }

    const int kpb = K / kGroup;  // packed blocks (= scale groups) per row tile
    gw4 pw[NPW];
    gub ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;  // clamped, never masked
      pw[j] = (gw4)(const uint4v*)(Wq) + (size_t)(t * (size_t)kpb + kbase / kGroup) * kWave + lane;
      ps[j] = (gub)(Ws) + (size_t)(t * (size_t)kpb + kbase / kGroup) * 16 + (lane & 15);
    }

    // Two groups per iteration only while both groups' operands fit the 192-VGPR cap of a 1024-thread block
    // (hipcc -S: MT*NPW <= 8 is spill-free paired; at MT 3 / 4 with NPW 4 the pair spilled 2 / 123 VGPRs).
    constexpr bool kPair = MT * NPW <= 8;
    int g = 0;
    for (; kPair && g + 1 < groups; g += 2) {
      uint4v wv0[NPW], wv1[NPW];
      float sc0[NPW], sc1[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const size_t o = (size_t)g * kWave;
        wv0[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        wv1[j] = NT ? __builtin_nontemporal_load(&pw[j][o + kWave]) : pw[j][o + kWave];
        sc0[j] = e8m0_value(ps[j][(size_t)g * 16]);
        sc1[j] = e8m0_value(ps[j][(size_t)(g + 1) * 16]);
      }
      group_step<MT, NPW>(acc, wv0, sc0, pa, 2 * g);
      group_step<MT, NPW>(acc, wv1, sc1, pa, 2 * g + 2);
    }
    // The odd tail group when paired (an `if`: the loop form spilled at MT 2 NPW 4), every group otherwise.
    auto single = [&](int gg) {
      uint4v wv0[NPW];
      float sc0[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const size_t o = (size_t)gg * kWave;
        wv0[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        sc0[j] = e8m0_value(ps[j][(size_t)gg * 16]);
      }
      group_step<MT, NPW>(acc, wv0, sc0, pa, 2 * gg);
    };
    if constexpr (kPair) {
      if (g < groups) single(g);
    } else {
      for (; g < groups; ++g) single(g);
    }
  }

  // Split-K reduction and store (w8a16's, without the unbias): one row tile at a time, fixed split order, so the
  // result does not depend on wave scheduling.
  const int ncols = WV * NPW;
  const int nthreads = WV * SK * kWave;
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
      for (int e = 0; e < 8; e++) red[slot + e * kWave + lane] = acc[j][i][e];
    }
    __syncthreads();
    for (int idx = tid; idx < ncols * 256; idx += nthreads) {
      const int col = idx >> 8, off = idx & 255;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      const int l = off & 31;
      const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
      const int m = (mt0 + i) * 16 + 8 * (l >> 4) + (off >> 5);
      if (n < N && m < M) C[(size_t)m * N + n] = __float2bfloat16(v);
    }
  }
}

[[noreturn]] void Fail(const std::string& what) { throw std::runtime_error("r4dx_gemm_mxfp8_nt: " + what); }

void Launch(int64_t a, int64_t wq, int64_t ws, int64_t c, int M, int K, int N, int WV, int SK, int MB, int NPW,
            int NT, int64_t stream) {
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * kWave);
  const size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  auto Ap = (const unsigned short*)a;
  auto Wp = (const unsigned char*)wq;
  auto Sp = (const unsigned char*)ws;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define MX8_LAUNCH(MT, NP, TT) \
  hipLaunchKernelGGL((mxfp8_gemm_nt_kernel<MT, NP, TT>), grid, block, smem, st, Ap, Wp, Sp, Cp, M, K, N, WV, SK)
#define MX8_LAUNCH_MB(NP, TT)             \
  switch (MB) {                           \
    case 1: MX8_LAUNCH(1, NP, TT); break; \
    case 2: MX8_LAUNCH(2, NP, TT); break; \
    case 3: MX8_LAUNCH(3, NP, TT); break; \
    default: MX8_LAUNCH(4, NP, TT); break; \
  }
  if (NPW == 1) {
    if (NT) { MX8_LAUNCH_MB(1, true); } else { MX8_LAUNCH_MB(1, false); }
  } else {
    if (NT) { MX8_LAUNCH_MB(4, true); } else { MX8_LAUNCH_MB(4, false); }
  }
#undef MX8_LAUNCH_MB
#undef MX8_LAUNCH
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) Fail(std::string("launch failed: ") + hipGetErrorString(err));
}

template <int NPW, bool NT>
__global__ __launch_bounds__(1024) void mxfp8_gemm_moe_decode_kernel(
    const unsigned short* __restrict__ A, const uint64_t* __restrict__ wq_ptrs,
    const uint64_t* __restrict__ ws_ptrs, const int32_t* __restrict__ ids,
    __hip_bfloat16* __restrict__ C, int a_row_group, int n_experts, int K, int N, int WV, int SK) {
  extern __shared__ float red[];  // WV*NPW*SK*256 floats
  const int p = blockIdx.z;
  const int e = ids[p];
  if (e < 0 || e >= n_experts) return;
  const unsigned char* Wq = reinterpret_cast<const unsigned char*>(wq_ptrs[e]);
  const unsigned char* Ws = reinterpret_cast<const unsigned char*>(ws_ptrs[e]);
  if (!Wq || !Ws) return;
  const unsigned short* Ap = A + (size_t)(p / a_row_group) * (size_t)K;
  __hip_bfloat16* Cp = C + (size_t)p * (size_t)N;

  const int tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int Ks = K / SK;
  const int groups = Ks / kGroup;
  const int kbase = ksplit * Ks;

  v8f acc[NPW][1];
#pragma unroll
  for (int j = 0; j < NPW; j++) acc[j][0] = (v8f)(0.f);

  if (n0 < N) {
    const unsigned off = rowGo((unsigned)K, 0, 0, lane) + (unsigned)kbase;
    gu2 pa[1] = {(gu2)(const unsigned long long*)(Ap + off)};

    const int kpb = K / kGroup;
    gw4 pw[NPW];
    gub ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (gw4)(const uint4v*)(Wq) + (size_t)(t * (size_t)kpb + kbase / kGroup) * kWave + lane;
      ps[j] = (gub)(Ws) + (size_t)(t * (size_t)kpb + kbase / kGroup) * 16 + (lane & 15);
    }

    int g = 0;
    for (; g + 1 < groups; g += 2) {
      uint4v wv0[NPW], wv1[NPW];
      float sc0[NPW], sc1[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const size_t o = (size_t)g * kWave;
        wv0[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        wv1[j] = NT ? __builtin_nontemporal_load(&pw[j][o + kWave]) : pw[j][o + kWave];
        sc0[j] = e8m0_value(ps[j][(size_t)g * 16]);
        sc1[j] = e8m0_value(ps[j][(size_t)(g + 1) * 16]);
      }
      group_step<1, NPW>(acc, wv0, sc0, pa, 2 * g);
      group_step<1, NPW>(acc, wv1, sc1, pa, 2 * g + 2);
    }
    if (g < groups) {
      uint4v wv0[NPW];
      float sc0[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const size_t o = (size_t)g * kWave;
        wv0[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        sc0[j] = e8m0_value(ps[j][(size_t)g * 16]);
      }
      group_step<1, NPW>(acc, wv0, sc0, pa, 2 * g);
    }
  }

  const int ncols = WV * NPW;
  const int nthreads = WV * SK * kWave;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
    for (int e_idx = 0; e_idx < 8; e_idx++) red[slot + e_idx * kWave + lane] = acc[j][0][e_idx];
  }
  __syncthreads();
  for (int idx = tid; idx < ncols * 256; idx += nthreads) {
    const int col = idx >> 8, off = idx & 255;
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
    const int l = off & 31;
    const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
    const int m = 8 * (l >> 4) + (off >> 5);
    if (n < N && m < 1) Cp[n] = __float2bfloat16(v);
  }
}

[[noreturn]] void FailMoe(const std::string& what) {
  throw std::runtime_error("r4dx_gemm_mxfp8_moe_decode: " + what);
}

void LaunchMoeDecode(int64_t a, int64_t wq_ptrs, int64_t ws_ptrs, int64_t ids, int64_t c, int n_pairs,
                     int a_row_group, int n_experts, int K, int N, int WV, int SK, int NPW, int NT,
                     int64_t stream) {
  const int ntiles = N / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, 1, n_pairs), block(WV * SK * kWave);
  const size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  auto Ap = (const unsigned short*)a;
  auto Wqp = (const uint64_t*)wq_ptrs;
  auto Wsp = (const uint64_t*)ws_ptrs;
  auto Idp = (const int32_t*)ids;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define MX8_MOE_LAUNCH(NP, TT)                                                                          \
  hipLaunchKernelGGL((mxfp8_gemm_moe_decode_kernel<NP, TT>), grid, block, smem, st, Ap, Wqp, Wsp, Idp,  \
                     Cp, a_row_group, n_experts, K, N, WV, SK)
  if (NPW == 1) {
    if (NT) { MX8_MOE_LAUNCH(1, true); } else { MX8_MOE_LAUNCH(1, false); }
  } else {
    if (NT) { MX8_MOE_LAUNCH(4, true); } else { MX8_MOE_LAUNCH(4, false); }
  }
#undef MX8_MOE_LAUNCH
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) FailMoe(std::string("launch failed: ") + hipGetErrorString(err));
}

__device__ __forceinline__ bool ResolveMxfp8MoePrefillTile(const int32_t* __restrict__ offsets, int n_experts,
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

// Cooperatively caches offsets[0..n_experts] in the unused reduction LDS buffer (at least 1024 B)
// so lane 0 resolves (e, row_start, m_rows) once per block instead of every thread scanning global memory.
__device__ __forceinline__ bool ResolveMxfp8MoePrefillTileCached(
    float* red_lds, const int32_t* __restrict__ offsets, int n_experts, int n_pairs, int tile_idx,
    int& out_e, int& out_row_start, int& out_m_rows) {
  if (n_experts <= 251) {
    int32_t* lds = reinterpret_cast<int32_t*>(red_lds);
    const int tid = static_cast<int>(threadIdx.x);
    for (int i = tid; i <= n_experts; i += static_cast<int>(blockDim.x)) {
      lds[i] = offsets[i];
    }
    __syncthreads();
    if (tid == 0) {
      int e = -1, r0 = 0, mr = 0;
      const bool ok = ResolveMxfp8MoePrefillTile(lds, n_experts, n_pairs, tile_idx, e, r0, mr);
      lds[252] = ok ? 1 : 0;
      lds[253] = e;
      lds[254] = r0;
      lds[255] = mr;
    }
    __syncthreads();
    const bool ok = (lds[252] != 0);
    out_e = lds[253];
    out_row_start = lds[254];
    out_m_rows = lds[255];
    __syncthreads();
    return ok;
  }
  return ResolveMxfp8MoePrefillTile(offsets, n_experts, n_pairs, tile_idx, out_e, out_row_start, out_m_rows);
}

template <int NPW, bool NT>
__global__ __launch_bounds__(1024) void mxfp8_gemm_moe_prefill_kernel(
    const unsigned short* __restrict__ A, const uint64_t* __restrict__ wq_ptrs,
    const uint64_t* __restrict__ ws_ptrs, const int32_t* __restrict__ offsets,
    __hip_bfloat16* __restrict__ C, int n_pairs, int n_experts, int K, int N, int WV, int SK) {
  extern __shared__ float red[];  // WV*NPW*SK*256 floats
  int e = -1, row_start = 0, m_rows = 0;
  if (!ResolveMxfp8MoePrefillTileCached(red, offsets, n_experts, n_pairs, static_cast<int>(blockIdx.y), e,
                                        row_start, m_rows)) {
    return;
  }
  const unsigned char* Wq = reinterpret_cast<const unsigned char*>(wq_ptrs[e]);
  const unsigned char* Ws = reinterpret_cast<const unsigned char*>(ws_ptrs[e]);
  if (!Wq || !Ws) return;
  const unsigned short* Ap = A + (size_t)row_start * (size_t)K;
  __hip_bfloat16* Cp = C + (size_t)row_start * (size_t)N;

  const int tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int Ks = K / SK;
  const int groups = Ks / kGroup;
  const int kbase = ksplit * Ks;

  v8f acc[NPW][1];
#pragma unroll
  for (int j = 0; j < NPW; j++) acc[j][0] = (v8f)(0.f);

  if (n0 < N) {
    const unsigned off = rowGo((unsigned)K, 0, m_rows - 1, lane) + (unsigned)kbase;
    gu2 pa[1] = {(gu2)(const unsigned long long*)(Ap + off)};

    const int kpb = K / kGroup;
    gw4 pw[NPW];
    gub ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (gw4)(const uint4v*)(Wq) + (size_t)(t * (size_t)kpb + kbase / kGroup) * kWave + lane;
      ps[j] = (gub)(Ws) + (size_t)(t * (size_t)kpb + kbase / kGroup) * 16 + (lane & 15);
    }

    int g = 0;
    for (; g + 1 < groups; g += 2) {
      uint4v wv0[NPW], wv1[NPW];
      float sc0[NPW], sc1[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const size_t o = (size_t)g * kWave;
        wv0[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        wv1[j] = NT ? __builtin_nontemporal_load(&pw[j][o + kWave]) : pw[j][o + kWave];
        sc0[j] = e8m0_value(ps[j][(size_t)g * 16]);
        sc1[j] = e8m0_value(ps[j][(size_t)(g + 1) * 16]);
      }
      group_step<1, NPW>(acc, wv0, sc0, pa, 2 * g);
      group_step<1, NPW>(acc, wv1, sc1, pa, 2 * g + 2);
    }
    if (g < groups) {
      uint4v wv0[NPW];
      float sc0[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const size_t o = (size_t)g * kWave;
        wv0[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        sc0[j] = e8m0_value(ps[j][(size_t)g * 16]);
      }
      group_step<1, NPW>(acc, wv0, sc0, pa, 2 * g);
    }
  }

  const int ncols = WV * NPW;
  const int nthreads = WV * SK * kWave;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
    for (int e_idx = 0; e_idx < 8; e_idx++) red[slot + e_idx * kWave + lane] = acc[j][0][e_idx];
  }
  __syncthreads();
  for (int idx = tid; idx < ncols * 256; idx += nthreads) {
    const int col = idx >> 8, off = idx & 255;
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
    const int l = off & 31;
    const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
    const int m = 8 * (l >> 4) + (off >> 5);
    if (n < N && m < m_rows) Cp[(size_t)m * N + n] = __float2bfloat16(v);
  }
}

[[noreturn]] void FailMoePrefill(const std::string& what) {
  throw std::runtime_error("r4dx_gemm_mxfp8_moe_prefill: " + what);
}

void LaunchMoePrefill(int64_t a, int64_t wq_ptrs, int64_t ws_ptrs, int64_t offsets, int64_t c, int n_pairs,
                      int n_experts, int K, int N, int WV, int SK, int NPW, int NT, int64_t stream) {
  const int ntiles = N / 16;
  const int ncols = WV * NPW;
  const int max_tiles = n_pairs < ((n_pairs + 15) / 16 + n_experts) ? n_pairs : ((n_pairs + 15) / 16 + n_experts);
  dim3 grid((ntiles + ncols - 1) / ncols, max_tiles), block(WV * SK * kWave);
  const size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  auto Ap = (const unsigned short*)a;
  auto Wqp = (const uint64_t*)wq_ptrs;
  auto Wsp = (const uint64_t*)ws_ptrs;
  auto Ofp = (const int32_t*)offsets;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define MX8_MOE_PF_LAUNCH(NP, TT)                                                                          \
  hipLaunchKernelGGL((mxfp8_gemm_moe_prefill_kernel<NP, TT>), grid, block, smem, st, Ap, Wqp, Wsp, Ofp,    \
                     Cp, n_pairs, n_experts, K, N, WV, SK)
  if (NPW == 1) {
    if (NT) { MX8_MOE_PF_LAUNCH(1, true); } else { MX8_MOE_PF_LAUNCH(1, false); }
  } else {
    if (NT) { MX8_MOE_PF_LAUNCH(4, true); } else { MX8_MOE_PF_LAUNCH(4, false); }
  }
#undef MX8_MOE_PF_LAUNCH
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) FailMoePrefill(std::string("launch failed: ") + hipGetErrorString(err));
}

}  // namespace

extern "C" {

void r4dx_gemm_mxfp8_nt(int64_t a_bf16, int64_t wq, int64_t ws, int64_t c, int M, int K, int N, int WV, int SK,
                        int MB, int NPW, int NT, int64_t stream) {
  if (M < 1 || M > kMaxM) Fail("M must be 1..256 (got " + std::to_string(M) + ")");
  if (N < 16 || N % 16) Fail("N must be a positive multiple of 16 (got " + std::to_string(N) + ")");
  if (WV < 1 || SK < 1) Fail("WV and SK must be >= 1");
  if (K < 1 || K % (SK * kGroup))
    Fail("K must be a positive multiple of SK*32 (K " + std::to_string(K) + ", SK " + std::to_string(SK) + ")");
  if (WV * SK * kWave > 1024) Fail("WV*SK*32 > 1024 threads per block");
  if (MB < 1 || MB > 4) Fail("MB must be 1..4");
  if (NPW != 1 && NPW != 4) Fail("NPW must be 1 or 4");
  if ((size_t)WV * NPW * SK * 256 * sizeof(float) > 64 * 1024) Fail("LDS WV*NPW*SK KiB > 64");
  if (!a_bf16 || !wq || !ws || !c) Fail("null pointer");
  Launch(a_bf16, wq, ws, c, M, K, N, WV, SK, MB, NPW, NT, stream);
  r4dx_kernel_launch_counter_add(1);
}

int r4dx_gemm_mxfp8_group(void) { return kGroup; }
int r4dx_gemm_mxfp8_max_m(void) { return kMaxM; }

void r4dx_gemm_mxfp8_moe_decode(int64_t a_bf16, int64_t wq_ptrs, int64_t ws_ptrs, int64_t ids, int64_t c,
                                int n_pairs, int a_row_group, int n_experts, int K, int N, int WV, int SK,
                                int NPW, int NT, int64_t stream) {
  if (n_pairs < 1 || n_pairs > kMaxMoePairs) {
    FailMoe("n_pairs must be 1.." + std::to_string(kMaxMoePairs) + " (got " + std::to_string(n_pairs) + ")");
  }
  if (a_row_group < 1 || a_row_group > n_pairs || n_pairs % a_row_group != 0) {
    FailMoe("a_row_group must divide n_pairs in 1..n_pairs (a_row_group " + std::to_string(a_row_group) +
            ", n_pairs " + std::to_string(n_pairs) + ")");
  }
  if (n_experts < 1 || n_experts > kMaxMoeExperts) {
    FailMoe("n_experts must be 1.." + std::to_string(kMaxMoeExperts) + " (got " + std::to_string(n_experts) + ")");
  }
  if (N < 16 || N % 16) FailMoe("N must be a positive multiple of 16 (got " + std::to_string(N) + ")");
  if (WV < 1 || SK < 1) FailMoe("WV and SK must be >= 1");
  if (K < 1 || K % (SK * kGroup)) {
    FailMoe("K must be a positive multiple of SK*32 (K " + std::to_string(K) + ", SK " + std::to_string(SK) + ")");
  }
  if (WV * SK * kWave > 1024) FailMoe("WV*SK*32 > 1024 threads per block");
  if (NPW != 1 && NPW != 4) FailMoe("NPW must be 1 or 4");
  if ((size_t)WV * NPW * SK * 256 * sizeof(float) > 64 * 1024) FailMoe("LDS WV*NPW*SK KiB > 64");
  if (!a_bf16 || !wq_ptrs || !ws_ptrs || !ids || !c) FailMoe("null pointer");
  LaunchMoeDecode(a_bf16, wq_ptrs, ws_ptrs, ids, c, n_pairs, a_row_group, n_experts, K, N, WV, SK, NPW, NT, stream);
  r4dx_kernel_launch_counter_add(1);
}

void r4dx_gemm_mxfp8_moe_prefill(int64_t a_bf16, int64_t wq_ptrs, int64_t ws_ptrs, int64_t expert_offsets,
                                 int64_t c, int n_pairs, int n_experts, int K, int N, int WV, int SK,
                                 int NPW, int NT, int64_t stream) {
  if (n_pairs < 1 || n_pairs > kMaxMoePairs) {
    FailMoePrefill("n_pairs must be 1.." + std::to_string(kMaxMoePairs) + " (got " + std::to_string(n_pairs) + ")");
  }
  if (n_experts < 1 || n_experts > kMaxMoeExperts) {
    FailMoePrefill("n_experts must be 1.." + std::to_string(kMaxMoeExperts) + " (got " + std::to_string(n_experts) + ")");
  }
  if (N < 16 || N % 16) FailMoePrefill("N must be a positive multiple of 16 (got " + std::to_string(N) + ")");
  if (WV < 1 || SK < 1) FailMoePrefill("WV and SK must be >= 1");
  if (K < 1 || K % (SK * kGroup)) {
    FailMoePrefill("K must be a positive multiple of SK*32 (K " + std::to_string(K) + ", SK " + std::to_string(SK) + ")");
  }
  if (WV * SK * kWave > 1024) FailMoePrefill("WV*SK*32 > 1024 threads per block");
  if (NPW != 1 && NPW != 4) FailMoePrefill("NPW must be 1 or 4");
  if ((size_t)WV * NPW * SK * 256 * sizeof(float) > 64 * 1024) FailMoePrefill("LDS WV*NPW*SK KiB > 64");
  if (!a_bf16 || !wq_ptrs || !ws_ptrs || !expert_offsets || !c) FailMoePrefill("null pointer");
  LaunchMoePrefill(a_bf16, wq_ptrs, ws_ptrs, expert_offsets, c, n_pairs, n_experts, K, N, WV, SK, NPW, NT, stream);
  r4dx_kernel_launch_counter_add(1);
}

}  // extern "C"

namespace {

// ---- Native NVFP4 (FP4 E2M1 + F8_E4M3 per-16 scales + per-expert fp32 factor) grouped MoE GEMMs -------------------
__device__ __forceinline__ unsigned short fp4_to_f16_bits(unsigned nib) {
  const unsigned s = (nib & 8u) << 12;
  const unsigned m = nib & 7u;
  // FP4 E2M1 magnitudes -> FP16: 0->0x0000, 1->0x3800 (0.5), 2..7->0x3800+(m<<9) (1.0,1.5,2.0,3.0,4.0,6.0).
  const unsigned mag = (m == 0u) ? 0u : ((m == 1u) ? 0x3800u : (0x3800u + (m << 9)));
  return static_cast<unsigned short>(s | mag);
}

__device__ __forceinline__ v8h dequant8_fp4(unsigned d) {
  union { unsigned short h[8]; v8h f; } x;
  x.h[0] = fp4_to_f16_bits(d & 0xFu);
  x.h[1] = fp4_to_f16_bits((d >> 4) & 0xFu);
  x.h[2] = fp4_to_f16_bits((d >> 8) & 0xFu);
  x.h[3] = fp4_to_f16_bits((d >> 12) & 0xFu);
  x.h[4] = fp4_to_f16_bits((d >> 16) & 0xFu);
  x.h[5] = fp4_to_f16_bits((d >> 20) & 0xFu);
  x.h[6] = fp4_to_f16_bits((d >> 24) & 0xFu);
  x.h[7] = fp4_to_f16_bits((d >> 28) & 0xFu);
  return x.f;
}

__device__ __forceinline__ void e4m3x2_to_f32(unsigned b0, unsigned b1, float& f0, float& f1) {
  const unsigned u = e4m3x2_scaled((b0 & 0xFFu) | ((b1 & 0xFFu) << 8), 0x010C000Cu);
  f0 = __half2float(__ushort_as_half(static_cast<unsigned short>(u & 0xFFFFu))) * 256.0f;
  f1 = __half2float(__ushort_as_half(static_cast<unsigned short>(u >> 16))) * 256.0f;
}

template <int NPW>
__device__ __forceinline__ void group_step_nvfp4(v8f (&acc)[NPW], const uint2v (&wv)[NPW], const float (&sc0)[NPW],
                                                 const float (&sc1)[NPW], gu2 pa, int u0) {
  const v8h a0 = fragA_bf16(pa, u0);
  const v8h a1 = fragA_bf16(pa, u0 + 1);
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const v8f g0 = mmah(a0, dequant8_fp4(wv[j][0]), (v8f)(0.f));
    const v8f g1 = mmah(a1, dequant8_fp4(wv[j][1]), (v8f)(0.f));
#pragma unroll
    for (int e = 0; e < 8; e++) acc[j][e] = fmaf(g1[e], sc1[j], fmaf(g0[e], sc0[j], acc[j][e]));
  }
}

template <int NPW, bool NT>
__global__ __launch_bounds__(1024) void nvfp4_gemm_moe_decode_kernel(
    const unsigned short* __restrict__ A, const uint64_t* __restrict__ wq_ptrs, const uint64_t* __restrict__ ws_ptrs,
    const float* __restrict__ factors, const int32_t* __restrict__ ids, __hip_bfloat16* __restrict__ C, int n_pairs,
    int a_row_group, int n_experts, int K, int N, int WV, int SK) {
  extern __shared__ float red[];
  const int p = static_cast<int>(blockIdx.z);
  if (p >= n_pairs) return;
  const int e = ids[p];
  if (e < 0 || e >= n_experts) return;
  const unsigned char* Wq = reinterpret_cast<const unsigned char*>(wq_ptrs[e]);
  const unsigned char* Ws = reinterpret_cast<const unsigned char*>(ws_ptrs[e]);
  if (!Wq || !Ws) return;
  const float factor = factors[e];
  const unsigned short* Ap = A + static_cast<size_t>(p / a_row_group) * static_cast<size_t>(K);
  __hip_bfloat16* Cp = C + static_cast<size_t>(p) * static_cast<size_t>(N);

  const int tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int Ks = K / SK;
  const int groups = Ks / kGroup;
  const int kbase = ksplit * Ks;

  v8f acc[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) acc[j] = (v8f)(0.f);

  if (n0 < N) {
    const unsigned off = rowGo((unsigned)K, 0, 0, lane) + (unsigned)kbase;
    const gu2 pa = (gu2)(const unsigned long long*)(Ap + off);
    const int kpb = K / kGroup;
    const int gpr16 = K / 16;
    gw2 pw[NPW];
    gub ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (gw2)(const uint2v*)(Wq) + (size_t)(t * (size_t)kpb + kbase / kGroup) * kWave + lane;
      ps[j] = (gub)(Ws) + (size_t)(t * (size_t)gpr16 + kbase / 16) * 16 + (lane & 15);
    }
    for (int g = 0; g < groups; ++g) {
      uint2v wv[NPW];
      float sc0[NPW], sc1[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const size_t o = (size_t)g * kWave;
        wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        e4m3x2_to_f32(ps[j][(size_t)(2 * g + 0) * 16], ps[j][(size_t)(2 * g + 1) * 16], sc0[j], sc1[j]);
      }
      group_step_nvfp4<NPW>(acc, wv, sc0, sc1, pa, 2 * g);
    }
  }

  const int ncols = WV * NPW;
  const int nthreads = WV * SK * kWave;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
    for (int e_idx = 0; e_idx < 8; e_idx++) red[slot + e_idx * kWave + lane] = acc[j][e_idx];
  }
  __syncthreads();
  for (int idx = tid; idx < ncols * 16; idx += nthreads) {
    const int col = idx >> 4, r = idx & 15;
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + r];
    const int n = (blockIdx.x * ncols + col) * 16 + r;
    if (n < N) Cp[n] = __float2bfloat16(v * factor);
  }
}

template <int NPW, bool NT>
__global__ __launch_bounds__(1024) void nvfp4_gemm_moe_prefill_kernel(
    const unsigned short* __restrict__ A, const uint64_t* __restrict__ wq_ptrs, const uint64_t* __restrict__ ws_ptrs,
    const float* __restrict__ factors, const int32_t* __restrict__ offsets, __hip_bfloat16* __restrict__ C, int n_pairs,
    int n_experts, int K, int N, int WV, int SK) {
  extern __shared__ float red[];
  int e = 0, row_start = 0, m_rows = 0;
  if (!ResolveMxfp8MoePrefillTileCached(red, offsets, n_experts, n_pairs, static_cast<int>(blockIdx.y), e,
                                        row_start, m_rows)) {
    return;
  }
  const unsigned char* Wq = reinterpret_cast<const unsigned char*>(wq_ptrs[e]);
  const unsigned char* Ws = reinterpret_cast<const unsigned char*>(ws_ptrs[e]);
  if (!Wq || !Ws) return;
  const float factor = factors[e];
  const unsigned short* Ap = A + (size_t)row_start * (size_t)K;
  __hip_bfloat16* Cp = C + (size_t)row_start * (size_t)N;

  const int tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int Ks = K / SK;
  const int groups = Ks / kGroup;
  const int kbase = ksplit * Ks;

  v8f acc[NPW];
#pragma unroll
  for (int j = 0; j < NPW; j++) acc[j] = (v8f)(0.f);

  if (n0 < N) {
    const unsigned off = rowGo((unsigned)K, 0, m_rows - 1, lane) + (unsigned)kbase;
    const gu2 pa = (gu2)(const unsigned long long*)(Ap + off);
    const int kpb = K / kGroup;
    const int gpr16 = K / 16;
    gw2 pw[NPW];
    gub ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (gw2)(const uint2v*)(Wq) + (size_t)(t * (size_t)kpb + kbase / kGroup) * kWave + lane;
      ps[j] = (gub)(Ws) + (size_t)(t * (size_t)gpr16 + kbase / 16) * 16 + (lane & 15);
    }
    for (int g = 0; g < groups; ++g) {
      uint2v wv[NPW];
      float sc0[NPW], sc1[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) {
        const size_t o = (size_t)g * kWave;
        wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        e4m3x2_to_f32(ps[j][(size_t)(2 * g + 0) * 16], ps[j][(size_t)(2 * g + 1) * 16], sc0[j], sc1[j]);
      }
      group_step_nvfp4<NPW>(acc, wv, sc0, sc1, pa, 2 * g);
    }
  }

  const int ncols = WV * NPW;
  const int nthreads = WV * SK * kWave;
  __syncthreads();
#pragma unroll
  for (int j = 0; j < NPW; j++) {
    const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
    for (int e_idx = 0; e_idx < 8; e_idx++) red[slot + e_idx * kWave + lane] = acc[j][e_idx];
  }
  __syncthreads();
  for (int idx = tid; idx < ncols * 256; idx += nthreads) {
    const int col = idx >> 8, off = idx & 255;
    float v = 0.f;
    for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
    const int l = off & 31;
    const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
    const int m = 8 * (l >> 4) + (off >> 5);
    if (n < N && m < m_rows) Cp[(size_t)m * N + n] = __float2bfloat16(v * factor);
  }
}

}  // namespace

extern "C" {

void r4dx_gemm_nvfp4_moe_decode(int64_t a_bf16, int64_t wq_ptrs, int64_t ws_ptrs, int64_t factors_f32, int64_t ids,
                                int64_t c, int n_pairs, int a_row_group, int n_experts, int K, int N, int WV, int SK,
                                int NPW, int NT, int64_t stream) {
  if (n_pairs < 1 || n_pairs > kMaxMoePairs || n_experts < 1 || n_experts > kMaxMoeExperts || N < 16 || N % 16 ||
      WV < 1 || SK < 1 || K < 1 || K % (SK * kGroup) || WV * SK * kWave > 1024 || (NPW != 1 && NPW != 4) ||
      (size_t)WV * NPW * SK * 256 * sizeof(float) > 64 * 1024 || !a_bf16 || !wq_ptrs || !ws_ptrs || !factors_f32 ||
      !ids || !c) {
    throw std::runtime_error("r4dx_gemm_nvfp4_moe_decode: invalid args");
  }
  const int ntiles = N / 16, ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, 1, n_pairs), block(WV * SK * kWave);
  const size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  auto Ap = (const unsigned short*)a_bf16;
  auto Wqp = (const uint64_t*)wq_ptrs;
  auto Wsp = (const uint64_t*)ws_ptrs;
  auto Fp = (const float*)factors_f32;
  auto Idp = (const int32_t*)ids;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define NV4_DEC_LAUNCH(NP, TT)                                                                              \
  hipLaunchKernelGGL((nvfp4_gemm_moe_decode_kernel<NP, TT>), grid, block, smem, st, Ap, Wqp, Wsp, Fp, Idp, \
                     Cp, n_pairs, a_row_group, n_experts, K, N, WV, SK)
  if (NPW == 1) {
    if (NT) { NV4_DEC_LAUNCH(1, true); } else { NV4_DEC_LAUNCH(1, false); }
  } else {
    if (NT) { NV4_DEC_LAUNCH(4, true); } else { NV4_DEC_LAUNCH(4, false); }
  }
#undef NV4_DEC_LAUNCH
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) throw std::runtime_error(std::string("r4dx_gemm_nvfp4_moe_decode: launch failed: ") + hipGetErrorString(err));
  r4dx_kernel_launch_counter_add(1);
}

void r4dx_gemm_nvfp4_moe_prefill(int64_t a_bf16, int64_t wq_ptrs, int64_t ws_ptrs, int64_t factors_f32,
                                 int64_t expert_offsets, int64_t c, int n_pairs, int n_experts, int K, int N, int WV,
                                 int SK, int NPW, int NT, int64_t stream) {
  if (n_pairs < 1 || n_pairs > kMaxMoePairs || n_experts < 1 || n_experts > kMaxMoeExperts || N < 16 || N % 16 ||
      WV < 1 || SK < 1 || K < 1 || K % (SK * kGroup) || WV * SK * kWave > 1024 || (NPW != 1 && NPW != 4) ||
      (size_t)WV * NPW * SK * 256 * sizeof(float) > 64 * 1024 || !a_bf16 || !wq_ptrs || !ws_ptrs || !factors_f32 ||
      !expert_offsets || !c) {
    FailMoePrefill("r4dx_gemm_nvfp4_moe_prefill: invalid args");
  }
  const int ntiles = N / 16, ncols = WV * NPW;
  const int max_tiles = n_pairs < ((n_pairs + 15) / 16 + n_experts) ? n_pairs : ((n_pairs + 15) / 16 + n_experts);
  dim3 grid((ntiles + ncols - 1) / ncols, max_tiles), block(WV * SK * kWave);
  const size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  auto Ap = (const unsigned short*)a_bf16;
  auto Wqp = (const uint64_t*)wq_ptrs;
  auto Wsp = (const uint64_t*)ws_ptrs;
  auto Fp = (const float*)factors_f32;
  auto Ofp = (const int32_t*)expert_offsets;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define NV4_PF_LAUNCH(NP, TT)                                                                               \
  hipLaunchKernelGGL((nvfp4_gemm_moe_prefill_kernel<NP, TT>), grid, block, smem, st, Ap, Wqp, Wsp, Fp, Ofp, \
                     Cp, n_pairs, n_experts, K, N, WV, SK)
  if (NPW == 1) {
    if (NT) { NV4_PF_LAUNCH(1, true); } else { NV4_PF_LAUNCH(1, false); }
  } else {
    if (NT) { NV4_PF_LAUNCH(4, true); } else { NV4_PF_LAUNCH(4, false); }
  }
#undef NV4_PF_LAUNCH
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) FailMoePrefill(std::string("nvfp4 launch failed: ") + hipGetErrorString(err));
  r4dx_kernel_launch_counter_add(1);
}

}  // extern "C"
