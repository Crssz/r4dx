// tests/kernels/int8_gemm_proto_kernels.h -- the int8 x int8 prefill GEMM PROTOTYPE's device code and host launch
// (docs/int8-gemm-proto.md). A bench-only experiment: nothing here is wired into libr4d or the model, and the
// shipped kernels are untouched. Included by tool_int8_gemm_proto.hip (hipcc, gfx1201).
//
// WHAT IT IS.  One kernel, i8g_kernel, in the shape of r4d_gemm_trellis_nt_m256 (docs/trellis-m256.md): a
// workgroup is RG x SKW = 4 x SKW waves, wave (rg, ks) owns rows 64 rg .. and K slice ks of this launch's
// SKW x SKG slices with 4 row tiles x 2 accumulator fragments, and the weight block of each step is shared
// through a double-buffered LDS slot. What changes:
//   * the WMMA is v_wmma_i32_16x16x16_iu8 (A and B signed int8, int32 accumulate), A in A8 layout (one 8 B
//     load per lane per (row tile, k-tile)), B an int8 fragment pair (16 B per lane per block);
//   * a step is a whole 128-K block (8 k-tiles), because the scales are per 128 K: each wave decodes TWO of
//     the block's eight (tile pair, k-tile) blocks (k-tiles rg and rg + 4), stores them to LDS, one barrier,
//     then all four row-group waves run 4 passes (one row tile each) of 8 k-tiles x 2 fragments of WMMA on
//     int32, and rescale that pass into fp32: acc += float(int32) * (sa[row] * sw[col]);
//   * TRELLIS = false (the "dense ceiling"): the weights are already int8 (W8 layout, 1 B / weight) and the
//     decode is one 16 B load. TRELLIS = true: the weights are the trellis words of the shipped kernel, the
//     decode is r4d_trellis_k4_decode / k5_decode unchanged, and the f16 fragments are quantized to int8 on the
//     fly: v_pk_fma_f16(w, rs, 1536) puts round(w * rs) in the low bits of an f16 (the f16 spacing is 1 there),
//     whose low byte is the int8; one v_perm_b32 packs four (6 VALU per fragment, no cvt, no clamp);
//   * the epilogue is the shipped kernel's (LDS reduction over the SKW slices, fp32 partials in ws, ticket, the
//     last block sums the SKG partials, FWHT-128, svh, out_scale, bf16) when FWHT = true, or a plain
//     reduce-and-convert when FWHT = false (the dense ceiling without the trellis rotation);
//   * RESC = 1, 2, 3 are speed BOUNDS for coarser scales, NOT the per-128 math (each is verified against its own
//     reference, i8g_ref_cols with a scale mask, and only ever reported as a bound):
//       1  both scales are per row / per column only (block 0's stand for the whole K): the int32 accumulators run
//          over the whole K slice, one rescale at the end (no per-128 VALU at all);
//       2  the activation scale is per row only (block 0's): per 128 K one cvt + one fma per element (sw), the
//          row scale multiplied once at the end;
//       3  the weight scale is per column only (block 0's): per 128 K one cvt + one fma (sa), the column scale at the end.
//     RESC 2 and 3 exist for the trellis kernel only; RESC 1 for both.
// There is no bit-identity with the shipped kernel's summation order to keep, so SK = SKW (the slices of a
// workgroup are all the slices there are) and SKG is free; the slices are whole 128-K blocks.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#include "int8_gemm_proto_ref.h"
#include "r4d_fwht128.h"
#include "r4d_gdn_wmma.h"
#include "r4d_trellis_dq.h"

#define I8G_WAVE 32

typedef int i8g_v2i __attribute__((ext_vector_type(2)));
typedef int i8g_v8i __attribute__((ext_vector_type(8)));
typedef float i8g_v4f __attribute__((ext_vector_type(4)));
typedef unsigned i8g_v4u __attribute__((ext_vector_type(4)));
typedef __attribute__((address_space(1))) const unsigned char* i8g_gptr;

// ---- small helpers ----------------------------------------------------------------------------------------
template <typename T>
__device__ __forceinline__ T i8g_load(i8g_gptr base, unsigned off, int imm) {
  return *(__attribute__((address_space(1))) const T*)(base + off + imm);
}
__device__ __forceinline__ void i8g_opaque(i8g_gptr& p) {
  unsigned long long v = (unsigned long long)p;
  asm("" : "+s"(v));
  p = (i8g_gptr)v;
}
__device__ __forceinline__ void i8g_opaque_v(unsigned& v) { asm("" : "+v"(v)); }
__device__ __forceinline__ void i8g_sync_lds() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
__device__ __forceinline__ unsigned short i8g_bf16_rn(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (unsigned short)((u >> 16) | 0x40u);
  return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ i8g_v8i i8g_mma(i8g_v2i a, i8g_v2i b, i8g_v8i c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}
template <int MASK, int N>
__device__ __forceinline__ void i8g_lane_stage(v8f (&v)[N], int lane) {
  const float sgn = (lane & MASK) ? -1.f : 1.f;
#pragma unroll
  for (int k = 0; k < N; ++k)
#pragma unroll
    for (int e = 0; e < 8; ++e) v[k][e] = __builtin_fmaf(sgn, v[k][e], r4d_fwht_partner<MASK>(v[k][e]));
}
__device__ __forceinline__ void i8g_bfly(v8f& lo, v8f& hi) {
  const v8f a = lo, b = hi;
  lo = a + b;
  hi = a - b;
}

// ---- the quantizers (CPU twins: i8p::RsHalf, QuantWTrick, QuantAct) -------------------------------------------
// rs as a packed f16 pair, from the table's s_eff with the hardware reciprocal (idempotent, see i8p::RsHalf)
__device__ __forceinline__ unsigned i8g_rs_pair(float s_eff) {
  const _Float16 h = (_Float16)__builtin_fminf(__builtin_amdgcn_rcpf(s_eff), 60000.f);
  return __builtin_bit_cast(unsigned, (r4d_h2){h, h});
}
// four decoded f16 weights (two dwords) -> four int8 (one dword): the low byte of f16(fma(w, rs, 1536))
__device__ __forceinline__ unsigned i8g_q4(unsigned d0, unsigned d1, unsigned rs2) {
  const r4d_h2 c = {(_Float16)1536.f, (_Float16)1536.f};
  const r4d_h2 r = __builtin_bit_cast(r4d_h2, rs2);
  const r4d_h2 y0 = __builtin_elementwise_fma(__builtin_bit_cast(r4d_h2, d0), r, c);
  const r4d_h2 y1 = __builtin_elementwise_fma(__builtin_bit_cast(r4d_h2, d1), r, c);
  return __builtin_amdgcn_perm(__builtin_bit_cast(unsigned, y1), __builtin_bit_cast(unsigned, y0), 0x06040200u);
}
// a decoded fragment (8 f16: elements 0..7 in order) -> 8 int8 as two dwords (bytes = elements)
__device__ __forceinline__ uint2 i8g_qfrag(v8h f, unsigned rs2) {
  const r4d_u32x4 d = __builtin_bit_cast(r4d_u32x4, f);
  return make_uint2(i8g_q4(d[0], d[1], rs2), i8g_q4(d[2], d[3], rs2));
}
// the table entry of an (column, 128 k) group of amax m: 1 / f16(min(1 / (m / 127), 60000)), IEEE (builder only)
__device__ __forceinline__ float i8g_seff(float m) {
  const float s = m > 0.f ? m / 127.0f : 1.0f;
  const _Float16 h = (_Float16)__builtin_fminf(1.0f / s, 60000.f);
  return 1.0f / (float)h;
}

// ---- A producer: f16 [256][K] -> A8 + SA (the kernel a production transform would fuse) ------------------------
// One wave per (row, 128-block): lane l takes k = 4 l .. 4 l + 3 (8 B load), the block's amax by xor shuffles,
// s = amax / 127 (1 for all zero), q = clamp(rint(x / s)) (IEEE division); one 4 B store per lane into the
// fragment layout (the 4 k are 4 consecutive elements of one half, so they are one dword of one lane's 8 B).
// Ap (plain [256][K], true k) is written only for the references.
__global__ __launch_bounds__(256) void i8g_quant_act(const unsigned short* __restrict__ X, signed char* __restrict__ A8,
                                                     signed char* __restrict__ Ap, float* __restrict__ SA, int K) {
  const int lane = threadIdx.x & 31;
  const int NKB = K >> 7;
  const long long id = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (id >= (long long)i8p::kM * NKB) return;
  const int r = (int)(id / NKB), kb = (int)(id % NKB);
  const uint2 raw = *(const uint2*)(X + (size_t)r * K + kb * 128 + lane * 4);
  float v[4];
  v[0] = (float)__builtin_bit_cast(_Float16, (unsigned short)(raw.x & 0xFFFFu));
  v[1] = (float)__builtin_bit_cast(_Float16, (unsigned short)(raw.x >> 16));
  v[2] = (float)__builtin_bit_cast(_Float16, (unsigned short)(raw.y & 0xFFFFu));
  v[3] = (float)__builtin_bit_cast(_Float16, (unsigned short)(raw.y >> 16));
  float amax = __builtin_fmaxf(__builtin_fmaxf(__builtin_fabsf(v[0]), __builtin_fabsf(v[1])),
                               __builtin_fmaxf(__builtin_fabsf(v[2]), __builtin_fabsf(v[3])));
#pragma unroll
  for (int m = 16; m >= 1; m >>= 1) amax = __builtin_fmaxf(amax, __shfl_xor(amax, m, I8G_WAVE));
  const float s = amax > 0.f ? amax / 127.0f : 1.0f;
  unsigned p = 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    float q = __builtin_rintf(v[j] / s);
    q = __builtin_fminf(__builtin_fmaxf(q, -127.f), 127.f);
    p |= ((unsigned)(int)q & 0xFFu) << (8 * j);
  }
  const int m = lane & 3, h = m & 1, e0 = 4 * (m >> 1), kt = kb * 8 + (lane >> 2);
  const int lane_l = (r & 15) + 16 * h;
  *(unsigned*)(A8 + i8p::A8FragOffset(r >> 6, kt, (r >> 4) & 3, lane_l, K >> 4) + e0) = p;
  if (Ap) *(unsigned*)(Ap + (size_t)r * K + kb * 128 + lane * 4) = p;
  if (lane == 0) SA[(size_t)kb * i8p::kM + r] = s;
}

// ---- weight side: scale table and the int8 matrix from the trellis words --------------------------------------
template <int KB>
__device__ __forceinline__ void i8g_decode_block(const unsigned* b, int lane, v8h& f0, v8h& f1) {
  if constexpr (KB == 4) {
    const unsigned ia = r4d_trellis_k4_off_ab(lane) / 4, ip = r4d_trellis_k4_off_p(lane) / 4;
    r4d_trellis_k4_decode(b[ip], b[ia], b[ia + 1], f0, f1);
  } else {
    unsigned w5[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) w5[i] = b[r4d_trellis_k5_word(lane, i)];
    r4d_trellis_k5_decode(w5, r4d_trellis_k5_shift(lane), f0, f1);
  }
}
// SW[g][n], n a column of Q, g = k / 128: one wave per (tile pair, group), 8 k-tiles, lane xor 16 joins the halves
template <int KB>
__global__ __launch_bounds__(256) void i8g_wscale(const unsigned* __restrict__ W, float* __restrict__ SW, int K, int N) {
  const int lane = threadIdx.x & 31;
  const int kt_total = K >> 4, groups = kt_total >> 3;
  const long long id = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (id >= (long long)(N / 32) * groups) return;
  const int pair = (int)(id / groups), g = (int)(id - (long long)pair * groups);
  float m0 = 0.f, m1 = 0.f;
  for (int j = 0; j < 8; ++j) {
    const long long blk = (long long)pair * kt_total + g * 8 + j;
    v8h f0, f1;
    i8g_decode_block<KB>(W + blk * (16 * KB), lane, f0, f1);
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      m0 = __builtin_fmaxf(m0, __builtin_fabsf((float)f0[e]));
      m1 = __builtin_fmaxf(m1, __builtin_fabsf((float)f1[e]));
    }
  }
  m0 = __builtin_fmaxf(m0, __shfl_xor(m0, 16, I8G_WAVE));
  m1 = __builtin_fmaxf(m1, __shfl_xor(m1, 16, I8G_WAVE));
  if (lane < 16) {
    const int n = i8p::FragCol(pair, lane, 0);
    SW[(size_t)g * N + n] = i8g_seff(m0);
    SW[(size_t)g * N + n + 8] = i8g_seff(m1);
  }
}
// W8 (block layout, 16 B per lane) and, if Wp is non-null, the plain [N][K] matrix, from the words and the table
// with the quantizer the GEMM runs: the dense kernel's weights ARE the trellis kernel's.
template <int KB>
__global__ __launch_bounds__(256) void i8g_dump_w(const unsigned* __restrict__ W, const float* __restrict__ SW,
                                                  signed char* __restrict__ W8, signed char* __restrict__ Wp, int K, int N) {
  const int lane = threadIdx.x & 31;
  const int kt_total = K >> 4;
  const long long blk = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (blk >= (long long)(N / 32) * kt_total) return;
  const int pair = (int)(blk / kt_total), kt = (int)(blk - (long long)pair * kt_total);
  v8h f0, f1;
  i8g_decode_block<KB>(W + blk * (16 * KB), lane, f0, f1);
  const int n0 = i8p::FragCol(pair, lane, 0);
  const float* row = SW + (size_t)(kt >> 3) * N;
  const uint2 q0 = i8g_qfrag(f0, i8g_rs_pair(row[n0])), q1 = i8g_qfrag(f1, i8g_rs_pair(row[n0 + 8]));
  *(i8g_v4u*)(W8 + (size_t)blk * 512 + lane * 16) = (i8g_v4u){q0.x, q0.y, q1.x, q1.y};
  if (Wp) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const size_t k = (size_t)kt * 16 + i8p::FragK16(lane, e);
      const unsigned sh = 8 * (e & 3);
      Wp[(size_t)n0 * K + k] = (signed char)(((e < 4 ? q0.x : q0.y) >> sh) & 0xFFu);
      Wp[(size_t)(n0 + 8) * K + k] = (signed char)(((e < 4 ? q1.x : q1.y) >> sh) & 0xFFu);
    }
  }
}

// ---- the GEMM ---------------------------------------------------------------------------------------------------
// A8 / SA: the activations (A8 layout, SA [K/128][256]); Wsrc: trellis pair-grid words (TRELLIS) or W8;
// SW [K/128][N]; svh [N] fp32 and out_scale: the trellis epilogue's (FWHT); C bf16 [256][N]; ws: SKG x 256 x N
// fp32; tickets: N / 128 zeroed words (self-resetting). Grid (N / 32, SKG), block 128 SKW, LDS 8192 SKW bytes.
#ifndef I8G_MINWAVES
#define I8G_MINWAVES 8   // amdgpu_waves_per_eu lower bound: 8 caps the kernel at 192 VGPRs (two 16-wave workgroups per WGP)
#endif
template <bool TRELLIS, int KB, bool FWHT, int RESC, int SKW>
__global__ __launch_bounds__(128 * SKW) __attribute__((amdgpu_waves_per_eu(I8G_MINWAVES))) void i8g_kernel(
    const signed char* __restrict__ A8, const float* __restrict__ SA, const unsigned char* __restrict__ Wsrc,
    const float* __restrict__ SW, const float* __restrict__ svh, unsigned short* __restrict__ C, float* __restrict__ ws,
    unsigned* __restrict__ tickets, int K, int N, int SKG, float out_scale) {
  constexpr int RG = 4, MT = 4, M = 256, Wc = 32, NBLK = 4;
  constexpr int WBLK = TRELLIS ? 64 * KB : 512;           // bytes of one (tile pair, k-tile) block in Wsrc
  constexpr unsigned slot_bytes = 8 * 512;                // one K slice's staging per buffer: 8 blocks of 512 B
  constexpr unsigned buf_bytes = SKW * slot_bytes;
  static_assert(SKW == 2 || SKW == 4 || SKW == 8, "");
  extern __shared__ __attribute__((aligned(16))) unsigned char lds[];
  const int tid = threadIdx.x, lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int rg = wave & 3, ks = wave >> 2;
  const int pair0 = blockIdx.x, n0 = pair0 * Wc;
  const int t = (lane >> 3) & 1, c = lane & 7, hh = lane >> 4;
  const int KT = K >> 4;
  const int ktw = KT / (SKW * SKG);                       // k-tiles per slice, a multiple of 8
  const int nkb = ktw >> 3;
  const int kt0 = ((int)blockIdx.y * SKW + ks) * ktw;
  const int kbg0 = kt0 >> 3;
  const int ncol0 = n0 + 16 * t + c;                      // the true column of accumulator fragment 0 (+ 8 for 1)

  v8f accf[2][MT];
  i8g_v8i acci[2][MT];
#pragma unroll
  for (int f = 0; f < 2; ++f)
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      accf[f][i] = (v8f)(0.f);
      acci[f][i] = (i8g_v8i)(0);
    }

  {
    // ---- per-wave bases: uniform 64-bit base + one 32-bit lane offset per stream ---------------------------
    i8g_gptr abase = (i8g_gptr)(const unsigned char*)A8 + ((size_t)rg * KT + kt0) * 1024;   // + lane 8 + u 1024 + i 256
    const unsigned aoff = lane * 8;
    i8g_gptr sabase = (i8g_gptr)(const unsigned char*)SA + ((size_t)kbg0 * 256 + rg * 64) * 4;   // + hh 32 + i 64 (+16)
    const unsigned saoff = hh * 32;
    i8g_gptr swbase = (i8g_gptr)(const unsigned char*)SW + (size_t)kbg0 * N * 4;
    const unsigned swoff = ncol0 * 4;                     // fragment 1: + 32 bytes (column + 8)
    i8g_gptr wbase = (i8g_gptr)Wsrc + ((size_t)pair0 * KT + kt0 + rg) * WBLK;   // the wave's first block; the second is + 4 blocks
    constexpr int NOFF = TRELLIS ? (KB == 4 ? 2 : 5) : 1;
    unsigned woff[NOFF];
    if constexpr (TRELLIS) {
      if constexpr (KB == 4) {
        woff[0] = r4d_trellis_k4_off_ab(lane);
        woff[1] = r4d_trellis_k4_off_p(lane);
      } else {
#pragma unroll
        for (int i = 0; i < 5; ++i) woff[i] = r4d_trellis_k5_word(lane, i) * 4u;
      }
    } else {
      woff[0] = lane * 16;
    }
    const unsigned k5sh = r4d_trellis_k5_shift(lane);

    // the lane's words of the wave's two blocks of a kb, and the kb's two weight scales
    unsigned long long ab[2] = {0, 0};
    unsigned pw[2] = {0, 0};
    unsigned w5[2][5];
    i8g_v4u dd[2];
    float swc0 = 0.f, swc1 = 0.f;
#pragma unroll
    for (int b = 0; b < 2; ++b) {
      dd[b] = (i8g_v4u)(0u);
#pragma unroll
      for (int i = 0; i < 5; ++i) w5[b][i] = 0;
    }
    auto load_w = [&](int ahead) {
      i8g_gptr bp = wbase + (size_t)ahead * 8 * WBLK;
      i8g_opaque(bp);
#pragma unroll
      for (int b = 0; b < 2; ++b) {
        const int imm = b * 4 * WBLK;
        if constexpr (TRELLIS) {
          if constexpr (KB == 4) {
            ab[b] = i8g_load<unsigned long long>(bp, woff[0], imm);
            pw[b] = i8g_load<unsigned>(bp, woff[1], imm);
          } else {
#pragma unroll
            for (int i = 0; i < 5; ++i) w5[b][i] = i8g_load<unsigned>(bp, woff[i], imm);
          }
        } else {
          dd[b] = i8g_load<i8g_v4u>(bp, woff[0], imm);
        }
      }
    };
    auto load_sw = [&](int ahead) {
      i8g_gptr sp = swbase + (size_t)ahead * N * 4;
      i8g_opaque(sp);
      swc0 = i8g_load<float>(sp, swoff, 0);
      swc1 = i8g_load<float>(sp, swoff, 32);
    };
    i8g_v2i af[8];
    i8g_v4f sav[2];
    auto load_a = [&](int i) {
#pragma unroll
      for (int u = 0; u < 8; ++u) af[u] = i8g_load<i8g_v2i>(abase, aoff, u * 1024 + i * 256);
      sav[0] = i8g_load<i8g_v4f>(sabase, saoff, i * 64);
      sav[1] = i8g_load<i8g_v4f>(sabase, saoff, i * 64 + 16);
    };

    load_w(0);
    load_sw(0);
    int buf = 0;
    const unsigned lds_w = ks * slot_bytes + rg * 512 + lane * 16;      // block u = rg; u = rg + 4 is + 2048
    const unsigned lds_r = ks * slot_bytes + lane * 16;                 // block u at + 512 u
    for (int kbi = 0; kbi < nkb; ++kbi) {
      // ---- phase D: this kb's first activation pass, then decode + quantize + publish the two own blocks ----
      const float s0c = swc0, s1c = swc1;
      unsigned long long abc[2] = {ab[0], ab[1]};
      unsigned pwc[2] = {pw[0], pw[1]};
      unsigned c5[2][5];
      i8g_v4u ddc[2] = {dd[0], dd[1]};
#pragma unroll
      for (int b = 0; b < 2; ++b)
#pragma unroll
        for (int i = 0; i < 5; ++i) c5[b][i] = w5[b][i];
      const int more = (kbi + 1 < nkb) ? 1 : 0;
      load_w(more);
      load_sw(more);
      load_a(0);
      __builtin_amdgcn_sched_barrier(0);
      {
        unsigned rs0 = 0, rs1 = 0;
        if constexpr (TRELLIS) {
          rs0 = i8g_rs_pair(s0c);
          rs1 = i8g_rs_pair(s1c);
        }
#pragma unroll
        for (int b = 0; b < 2; ++b) {
          i8g_v4u q;
          if constexpr (TRELLIS) {
            v8h f0, f1;
            if constexpr (KB == 4) r4d_trellis_k4_decode(pwc[b], (unsigned)abc[b], (unsigned)(abc[b] >> 32), f0, f1);
            else r4d_trellis_k5_decode(c5[b], k5sh, f0, f1);
            const uint2 q0 = i8g_qfrag(f0, rs0), q1 = i8g_qfrag(f1, rs1);
            q = (i8g_v4u){q0.x, q0.y, q1.x, q1.y};
          } else {
            q = ddc[b];
          }
          *(i8g_v4u*)(lds + buf * buf_bytes + lds_w + b * 2048) = q;
        }
      }
      __builtin_amdgcn_sched_barrier(0);
      i8g_sync_lds();
      // ---- phase W: the eight blocks' fragments, then four passes (one row tile each) ----------------------
      i8g_v2i b0[8], b1[8];
#pragma unroll
      for (int u = 0; u < 8; ++u) {
        const i8g_v4u x = *(const i8g_v4u*)(lds + buf * buf_bytes + lds_r + u * 512);
        b0[u] = (i8g_v2i){(int)x[0], (int)x[1]};
        b1[u] = (i8g_v2i){(int)x[2], (int)x[3]};
      }
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        i8g_v2i afc[8];
        i8g_v4f sac[2];
#pragma unroll
        for (int u = 0; u < 8; ++u) afc[u] = af[u];
        sac[0] = sav[0];
        sac[1] = sav[1];
        if (i + 1 < MT) load_a(i + 1);                    // the next pass's A and sa, behind this pass's WMMA
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (RESC != 1) {
          i8g_v8i p0 = (i8g_v8i)(0), p1 = (i8g_v8i)(0);
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            p0 = i8g_mma(afc[u], b0[u], p0);
            p1 = i8g_mma(afc[u], b1[u], p1);
          }
          const v8f t0 = __builtin_convertvector(p0, v8f), t1 = __builtin_convertvector(p1, v8f);
          const v8f sa8 = __builtin_shufflevector(sac[0], sac[1], 0, 1, 2, 3, 4, 5, 6, 7);
          if constexpr (RESC == 0) {          // acc += float(int32) * (sa[row] * sw[col]): cvt + mul + fma per element
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              accf[0][i][e] = __builtin_fmaf(t0[e], sa8[e] * s0c, accf[0][i][e]);
              accf[1][i][e] = __builtin_fmaf(t1[e], sa8[e] * s1c, accf[1][i][e]);
            }
          } else if constexpr (RESC == 2) {   // activation scale of the first block for all (applied once at the end): cvt + fma
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              accf[0][i][e] = __builtin_fmaf(t0[e], s0c, accf[0][i][e]);
              accf[1][i][e] = __builtin_fmaf(t1[e], s1c, accf[1][i][e]);
            }
          } else {                            // RESC == 3: weight scale of the first block for all: cvt + fma
#pragma unroll
            for (int e = 0; e < 8; ++e) {
              accf[0][i][e] = __builtin_fmaf(t0[e], sa8[e], accf[0][i][e]);
              accf[1][i][e] = __builtin_fmaf(t1[e], sa8[e], accf[1][i][e]);
            }
          }
        } else {
#pragma unroll
          for (int u = 0; u < 8; ++u) {
            acci[0][i] = i8g_mma(afc[u], b0[u], acci[0][i]);
            acci[1][i] = i8g_mma(afc[u], b1[u], acci[1][i]);
          }
        }
        __builtin_amdgcn_sched_barrier(0);
      }
      buf ^= 1;
      abase = abase + 8192;
      sabase = sabase + 1024;
      wbase = wbase + (size_t)8 * WBLK;
      swbase = swbase + (size_t)N * 4;
      i8g_opaque(abase);
      i8g_opaque(sabase);
      i8g_opaque(wbase);
      i8g_opaque(swbase);
#pragma unroll
      for (int i = 0; i < NOFF; ++i) i8g_opaque_v(woff[i]);
      __builtin_amdgcn_sched_barrier(0);
    }
  }

  if constexpr (RESC != 0) {
    // the coarse variants' one multiply with the scale(s) of block 0 of the whole K (a per-row / per-column scale;
    // speed bounds, not the per-128 math): RESC 1 both scales (the int32 accumulators ran over the whole slice),
    // RESC 2 the activation scale, RESC 3 the weight scale (the fp32 accumulators already hold the other, per-block,
    // one). Every slice multiplies its own partial sum by the same factors, so the slices still add up.
    const float* swp = SW + ncol0;
    const float s0 = swp[0], s1 = swp[8];
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      if constexpr (RESC == 3) {
        accf[0][i] = accf[0][i] * s0;
        accf[1][i] = accf[1][i] * s1;
      } else {
        const i8g_v4f* sap = (const i8g_v4f*)(SA + (size_t)rg * 64 + 16 * i + 8 * hh);
        const v8f sa8 = __builtin_shufflevector(sap[0], sap[1], 0, 1, 2, 3, 4, 5, 6, 7);
        if constexpr (RESC == 1) {
          const v8f t0 = __builtin_convertvector(acci[0][i], v8f), t1 = __builtin_convertvector(acci[1][i], v8f);
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            accf[0][i][e] = t0[e] * (sa8[e] * s0);
            accf[1][i][e] = t1[e] * (sa8[e] * s1);
          }
        } else {
          accf[0][i] = accf[0][i] * sa8;
          accf[1][i] = accf[1][i] * sa8;
        }
      }
    }
  }

  // ---- 1. reduction over the SKW slices, in the accumulator layout (the shipped kernel's) ---------------------
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    const int o = i % SKW;
    i8g_sync_lds();
    if (ks != o) {
      const int kw = ks > o ? ks - 1 : ks;
      unsigned char* dst = lds + ((rg * (SKW - 1) + kw) * 4) * 512 + lane * 16;
#pragma unroll
      for (int f = 0; f < 2; ++f) {
        const v8f x = accf[f][i];
        *(i8g_v4f*)(dst + (f * 2 + 0) * 512) = __builtin_shufflevector(x, x, 0, 1, 2, 3);
        *(i8g_v4f*)(dst + (f * 2 + 1) * 512) = __builtin_shufflevector(x, x, 4, 5, 6, 7);
      }
    }
    i8g_sync_lds();
    if (ks == o) {
      const unsigned char* src = lds + ((rg * (SKW - 1)) * 4) * 512 + lane * 16;
#pragma unroll
      for (int f = 0; f < 2; ++f) {
        v8f s = (v8f)(0.f);
#pragma unroll
        for (int sl = 0; sl < SKW; ++sl) {
          v8f term;
          if (sl == o) {
            term = accf[f][i];
          } else {
            const int kw = sl > o ? sl - 1 : sl;
            const i8g_v4f lo = *(const i8g_v4f*)(src + (kw * 4 + f * 2 + 0) * 512);
            const i8g_v4f hi = *(const i8g_v4f*)(src + (kw * 4 + f * 2 + 1) * 512);
            term = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
          }
          s = s + term;
        }
        accf[f][i] = s;
      }
    }
  }

  if constexpr (!FWHT) {
    if (SKG == 1) {   // plain reduce-and-convert: the owners store the finished tile
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        if (ks != i % SKW) continue;
#pragma unroll
        for (int f = 0; f < 2; ++f)
#pragma unroll
          for (int e = 0; e < 8; ++e)
            C[(size_t)i8p::AccRow(rg, i, lane, e) * N + (n0 + 16 * t + 8 * f + c)] = i8g_bf16_rn(accf[f][i][e] * out_scale);
      }
      return;
    }
  }

  // ---- 2. hand the sums to ws in lane order, take a ticket ------------------------------------------------------
  float* wsy = ws + (size_t)blockIdx.y * M * N;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    if (ks != i % SKW) continue;
    float* unit = wsy + ((size_t)(rg * 4 + i) * (N / Wc) + n0 / Wc) * (16 * Wc);
#pragma unroll
    for (int f = 0; f < 2; ++f) {
      const v8f x = accf[f][i];
      *(i8g_v4f*)(unit + ((f * 2 + 0) * 32 + lane) * 4) = __builtin_shufflevector(x, x, 0, 1, 2, 3);
      *(i8g_v4f*)(unit + ((f * 2 + 1) * 32 + lane) * 4) = __builtin_shufflevector(x, x, 4, 5, 6, 7);
    }
  }
  const int g0 = n0 / 128 * 128;
  const unsigned contributors = (unsigned)(SKG * NBLK);
  __threadfence();
  __syncthreads();
  float* flag = (float*)lds;
  if (tid == 0) {
    const unsigned old = atomicAdd(&tickets[g0 / 128], 1u);
    const bool last = old == contributors - 1;
    if (last) tickets[g0 / 128] = 0u;
    flag[0] = last ? 1.f : 0.f;
  }
  __syncthreads();
  if (flag[0] == 0.f) return;
  __threadfence();

  // ---- 3. the last block finishes the 128-column group: y-sum, FWHT, svh, scale, bf16 ----------------------------
  const int cb0 = g0 / Wc;
  float sv[NBLK][2];
#pragma unroll
  for (int b = 0; b < NBLK; ++b)
#pragma unroll
    for (int f = 0; f < 2; ++f) sv[b][f] = FWHT ? svh[g0 + b * Wc + 16 * t + 8 * f + c] : 1.f;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    if (ks != i % SKW) continue;
    const int R = rg * 4 + i;
    v8f v[NBLK * 2];
#pragma unroll
    for (int k = 0; k < NBLK * 2; ++k) v[k] = (v8f)(0.f);
    for (int y = 0; y < SKG; ++y) {
#pragma unroll
      for (int b = 0; b < NBLK; ++b) {
        const float* unit = ws + (size_t)y * M * N + ((size_t)R * (N / Wc) + cb0 + b) * (16 * Wc);
#pragma unroll
        for (int f = 0; f < 2; ++f) {
          const i8g_v4f lo = *(const i8g_v4f*)(unit + ((f * 2 + 0) * 32 + lane) * 4);
          const i8g_v4f hi = *(const i8g_v4f*)(unit + ((f * 2 + 1) * 32 + lane) * 4);
          v[b * 2 + f] = v[b * 2 + f] + __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
        }
      }
    }
    if constexpr (FWHT) {
      // FWHT-128 over the group's columns: bits 0-2 lane bits 0-2, bit 3 the fragment pair, bit 4 lane bit 3,
      // bits 5 and 6 the block index
      i8g_lane_stage<1>(v, lane);
      i8g_lane_stage<2>(v, lane);
      i8g_lane_stage<4>(v, lane);
#pragma unroll
      for (int bp = 0; bp < NBLK; ++bp) i8g_bfly(v[bp * 2], v[bp * 2 + 1]);
      i8g_lane_stage<8>(v, lane);
#pragma unroll
      for (int f = 0; f < 2; ++f) {
        i8g_bfly(v[0 * 2 + f], v[1 * 2 + f]);
        i8g_bfly(v[2 * 2 + f], v[3 * 2 + f]);
      }
#pragma unroll
      for (int f = 0; f < 2; ++f) {
        i8g_bfly(v[0 * 2 + f], v[2 * 2 + f]);
        i8g_bfly(v[1 * 2 + f], v[3 * 2 + f]);
      }
    }
#pragma unroll
    for (int b = 0; b < NBLK; ++b)
#pragma unroll
      for (int f = 0; f < 2; ++f) {
        const int col = g0 + b * Wc + 16 * t + 8 * f + c;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
          const int row = R * 16 + 8 * hh + e;
          C[(size_t)row * N + col] = i8g_bf16_rn(FWHT ? (v[b * 2 + f][e] * sv[b][f]) * out_scale : v[b * 2 + f][e] * out_scale);
        }
      }
  }
}

// ---- reference: exact integer sums on the plain matrices, for chosen output columns --------------------------------
// out[idx * 256 + r] (fp64) = sum_kb sa[kb][r] sw[kb][cols[idx]] (sum_k Ap[r][k] Wp[col][k]); with mask bit 0 the
// activation scale of kb 0 stands for every kb, with bit 1 the weight scale of kb 0 (the coarse bound variants).
// One workgroup (256 threads, one per row) per chosen column.
__global__ __launch_bounds__(256) void i8g_ref_cols(const signed char* __restrict__ Ap, const float* __restrict__ SA,
                                                    const signed char* __restrict__ Wp, const float* __restrict__ SW,
                                                    const int* __restrict__ cols, double* __restrict__ out, int K, int N,
                                                    int mask) {
  const int r = threadIdx.x, col = cols[blockIdx.x];
  const signed char* a = Ap + (size_t)r * K;
  const signed char* w = Wp + (size_t)col * K;
  double acc = 0;
  for (int kb = 0; kb < (K >> 7); ++kb) {
    int s = 0;
    for (int k = kb * 128; k < kb * 128 + 128; k += 4) {
      const int av = *(const int*)(a + k), wv = *(const int*)(w + k);
#pragma unroll
      for (int j = 0; j < 4; ++j) s += (int)(signed char)(av >> (8 * j)) * (int)(signed char)(wv >> (8 * j));
    }
    const float sa_v = (mask & 1) ? SA[r] : SA[(size_t)kb * 256 + r];
    const float sw_v = (mask & 2) ? SW[col] : SW[(size_t)kb * N + col];
    acc += (double)sa_v * (double)sw_v * (double)s;
  }
  out[(size_t)blockIdx.x * 256 + r] = acc;
}

// ---- host: launch + dispatch -----------------------------------------------------------------------------------------
struct I8gCfg {
  bool trellis = false;
  int kb = 4;          // trellis rate (ignored when dense)
  bool fwht = true;
  int resc = 0;        // 0: per-128 rescale (the math), 1: one rescale per slice (the bound)
  int skw = 4;         // slices per workgroup (2, 4, 8) = SK
  int skg = 1;         // K groups across the grid (1, 2, 4, 8)
};

inline const char* I8gCheck(const I8gCfg& c, int K, int N) {
  static thread_local char msg[256];
#define I8G_BAD(...) do { std::snprintf(msg, sizeof msg, __VA_ARGS__); return msg; } while (0)
  if (K <= 0 || K % 128) I8G_BAD("K %d must be a multiple of 128", K);
  if (N <= 0 || N % 128) I8G_BAD("N %d must be a multiple of 128", N);
  if (c.skw != 2 && c.skw != 4 && c.skw != 8) I8G_BAD("skw %d must be 2, 4 or 8", c.skw);
  if (c.skg != 1 && c.skg != 2 && c.skg != 4 && c.skg != 8) I8G_BAD("skg %d must be 1, 2, 4 or 8", c.skg);
  if ((K / 128) % (c.skw * c.skg)) I8G_BAD("K/128 = %d must be divisible by skw * skg = %d", K / 128, c.skw * c.skg);
  if (c.trellis && c.kb != 4 && c.kb != 5) I8G_BAD("kb %d must be 4 or 5", c.kb);
  if (c.trellis && !c.fwht) I8G_BAD("the trellis kernel is only instantiated with the FWHT epilogue");
  if (c.resc < 0 || c.resc > 3) I8G_BAD("resc must be 0, 1, 2 or 3");
  if (!c.trellis && c.resc > 1) I8G_BAD("resc 2 and 3 are instantiated for the trellis kernel only");
#undef I8G_BAD
  return nullptr;
}

template <bool TR, int KB, bool FW, int RE, int SKW>
static void i8g_launch_t(const I8gCfg& c, const signed char* a8, const float* sa, const void* w, const float* sw,
                         const float* svh, unsigned short* C, float* ws, unsigned* tk, int K, int N, float out_scale,
                         hipStream_t st) {
  hipLaunchKernelGGL((i8g_kernel<TR, KB, FW, RE, SKW>), dim3(N / 32, c.skg), dim3(128 * SKW), 8192 * SKW, st, a8, sa,
                     (const unsigned char*)w, sw, svh, C, ws, tk, K, N, c.skg, out_scale);
}
template <bool TR, int KB, bool FW, int RE>
static bool i8g_dispatch_skw(const I8gCfg& c, const signed char* a8, const float* sa, const void* w, const float* sw,
                             const float* svh, unsigned short* C, float* ws, unsigned* tk, int K, int N, float os,
                             hipStream_t st) {
#ifdef I8G_LITE
  if (c.skw == 4) { i8g_launch_t<TR, KB, FW, RE, 4>(c, a8, sa, w, sw, svh, C, ws, tk, K, N, os, st); return true; }
#else
  switch (c.skw) {
    case 2: i8g_launch_t<TR, KB, FW, RE, 2>(c, a8, sa, w, sw, svh, C, ws, tk, K, N, os, st); return true;
    case 4: i8g_launch_t<TR, KB, FW, RE, 4>(c, a8, sa, w, sw, svh, C, ws, tk, K, N, os, st); return true;
    case 8: i8g_launch_t<TR, KB, FW, RE, 8>(c, a8, sa, w, sw, svh, C, ws, tk, K, N, os, st); return true;
    default: break;
  }
#endif
  return false;
}
// launches the kernel of `c`; throws if the configuration is illegal or not instantiated
inline void I8gRun(const I8gCfg& c, const signed char* a8, const float* sa, const void* w, const float* sw, const float* svh,
                   unsigned short* C, float* ws, unsigned* tk, int K, int N, float out_scale, hipStream_t st) {
  if (const char* why = I8gCheck(c, K, N)) throw std::runtime_error(std::string("i8g: ") + why);
  bool ok = false;
#define I8G_CASE(TR, KB, FW, RE) ok = i8g_dispatch_skw<TR, KB, FW, RE>(c, a8, sa, w, sw, svh, C, ws, tk, K, N, out_scale, st)
#define I8G_RESC4(TR, KB, FW) \
  switch (c.resc) { case 0: I8G_CASE(TR, KB, FW, 0); break; case 1: I8G_CASE(TR, KB, FW, 1); break; \
                    case 2: I8G_CASE(TR, KB, FW, 2); break; default: I8G_CASE(TR, KB, FW, 3); break; }
  if (!c.trellis) {
    if (c.fwht) { if (c.resc == 0) I8G_CASE(false, 4, true, 0); else I8G_CASE(false, 4, true, 1); }
    else { if (c.resc == 0) I8G_CASE(false, 4, false, 0); else I8G_CASE(false, 4, false, 1); }
  } else if (c.kb == 4) {
    I8G_RESC4(true, 4, true);
  } else {
    I8G_RESC4(true, 5, true);
  }
#undef I8G_RESC4
#undef I8G_CASE
  if (!ok) throw std::runtime_error("i8g: configuration not instantiated");
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) throw std::runtime_error(std::string("i8g: launch failed: ") + hipGetErrorString(e));
}
inline size_t I8gWsBytes(int N, int skg) { return (size_t)skg * i8p::kM * N * sizeof(float); }
