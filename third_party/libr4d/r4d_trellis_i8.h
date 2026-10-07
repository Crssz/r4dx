// r4d_trellis_i8.h -- the DEVICE code of libr4d's int8 x int8 trellis prefill GEMM: the kernel i8g_kernel, the
// activation quantizer i8g_quant_act, the weight scale table builder i8g_wscale and the int8 weight dump
// i8g_dump_w. r4d_gemm_trellis_nt_i8.hip instantiates the production kernels (TRELLIS, KB 4 / 5, FWHT, RESC 0 =
// per-128 scales and RESC 4 = coarse scales, SKW 2 / 4 / 8) and exports the host entries of r4d.h; tests/kernels/tool_int8_gemm_proto.hip (the bench, which
// also builds the dense and RESC 1..3 variants) and tests/kernels/test_int8_gemm_proto_emu.cpp (the kernel source
// compiled as plain C++) include it too. The design and the measurements: docs/int8-gemm-proto.md; the model
// side: docs/int8-prefill.md "Production path".
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
//   * TRELLIS = false (the "dense ceiling", bench only): the weights are already int8 (W8 layout, 1 B / weight)
//     and the decode is one 16 B load. TRELLIS = true: the weights are the trellis words of the shipped kernel,
//     the decode is r4d_trellis_k4_decode / k5_decode unchanged, and the f16 fragments are quantized to int8 on
//     the fly: v_pk_fma_f16(w, rs, 1536) puts round(w * rs) in the low bits of an f16 (the f16 spacing is 1
//     there), whose low byte is the int8; one v_perm_b32 packs four (6 VALU per fragment, no cvt, no clamp);
//   * the epilogue is the shipped kernel's (LDS reduction over the SKW slices, fp32 partials in ws, ticket, the
//     last block sums the SKG partials, FWHT-128, svh, out_scale, bf16) when FWHT = true, or a plain
//     reduce-and-convert when FWHT = false (the dense ceiling without the trellis rotation); the production
//     unit's copy of it is kept equal to the shipped kernel's by tools/reference/diff_epilogue.ps1;
//   * two A parts (A8_0 / SA_0 for output columns < n_split, A8_1 / SA_1 from n_split on: a fused gate / up
//     pair with different input transforms), chosen per block exactly as the shipped kernel does it;
//   * RESC = 1, 2, 3 are speed BOUNDS for coarser scales, NOT the per-128 math (bench only; each is verified
//     against its own reference, i8g_ref_cols with a scale mask, and only ever reported as a bound):
//       1  both scales are per row / per column only (block 0's stand for the whole K): the int32 accumulators run
//          over the whole K slice, one rescale at the end (no per-128 VALU at all);
//       2  the activation scale is per row only (block 0's): per 128 K one cvt + one fma per element (sw), the
//          row scale multiplied once at the end;
//       3  the weight scale is per column only (block 0's): per 128 K one cvt + one fma (sa), the column scale at the end.
//     RESC 2 and 3 exist for the trellis kernel only; RESC 1 for both.
//   * RESC = 4 is NOT a bound: it is the production coarse mode (R4DX_PREFILL_INT8_SCALES=coarse, docs/int8-prefill.md
//     "Coarse scales"). RESC 1's math with the tables it really is for: SA [256] (one scale per row, i8g_quant_act_row or
//     a fused producer), SW [N] (one per column over the whole K, i8g_wscale_col). The loop never loads SA or SW, the
//     weight scale's f16 rs is formed once, and the one multiply by sa[row] * sw[col] comes after the K loop (the int32
//     accumulators are exact over the whole slice: K 17408 x 127 x 127 = 2.8e8 < 2^31; a split-K slice is shorter).
//     Trellis only, FWHT only; verified against the exact reference like the per-128 kernel (bench, emulation, GPU test).
// There is no bit-identity with the shipped kernel's summation order to keep (the f16 kernel's chain of 16-k WMMA
// is gone: the int32 sums are exact and the fp32 rescale order is this kernel's own), so SK = SKW (the slices of
// a workgroup are all the slices there are) and SKG is free; the slices are whole 128-K blocks. A row's result
// depends on (skw, skg) and on nothing else of the launch (never on its position or on the other rows).
#pragma once

// I8G_EMU: compile the very same kernel source as plain C++ for the CPU (tests/kernels/i8g_host_emu.h supplies the
// device-side names: threads, workgroup and wave barriers, the iu8 WMMA, the cross-lane ops, stubbed trellis decode);
// test_int8_gemm_proto_emu.cpp runs the kernels that way against exact references. Unset (hipcc), nothing changes.
#ifdef I8G_EMU
#include "i8g_host_emu.h"
#else
#include <hip/hip_runtime.h>
#endif

#include <cstdint>
#include <type_traits>

#include "r4d_trellis_i8_fused.h"   // i8g_quant_row_wg (the per-row activation scale of RESC 4)
#include "r4d_trellis_i8_layout.h"
#ifndef I8G_EMU
#include "r4d_fwht128.h"
#include "r4d_gdn_wmma.h"
#include "r4d_trellis_dq.h"
#endif

#define I8G_WAVE 32

typedef int i8g_v2i __attribute__((ext_vector_type(2)));
typedef int i8g_v8i __attribute__((ext_vector_type(8)));
typedef float i8g_v4f __attribute__((ext_vector_type(4)));
typedef unsigned i8g_v4u __attribute__((ext_vector_type(4)));

// ---- small helpers ----------------------------------------------------------------------------------------
#ifdef I8G_EMU
typedef const unsigned char* i8g_gptr;
template <typename T>
inline T i8g_load(i8g_gptr base, unsigned off, int imm) { return *(const T*)(base + off + imm); }
inline void i8g_opaque(i8g_gptr&) {}
inline void i8g_opaque_v(unsigned&) {}
inline void i8g_sync_lds() { emu::wg_barrier(); }
#define I8G_LDS_DECL unsigned char* lds = emu::t_group->lds;
#else
typedef __attribute__((address_space(1))) const unsigned char* i8g_gptr;
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
#define I8G_LDS_DECL extern __shared__ __attribute__((aligned(16))) unsigned char lds[];
#endif
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
// Ap (plain [256][K], true k) is written only for the references. blockIdx.y selects the part (a two-part
// linear's second A, x_part_stride elements after the first in X; its A8 / Ap / SA follow the first's, 256 K
// bytes, 256 K bytes and K / 128 x 256 floats on).
__global__ __launch_bounds__(256) void i8g_quant_act(const unsigned short* __restrict__ X, signed char* __restrict__ A8,
                                                     signed char* __restrict__ Ap, float* __restrict__ SA, int K,
                                                     long long x_part_stride) {
  const int lane = threadIdx.x & 31;
  const int NKB = K >> 7;
  const long long id = (long long)blockIdx.x * 8 + (threadIdx.x >> 5);
  if (id >= (long long)i8p::kM * NKB) return;
  const int r = (int)(id / NKB), kb = (int)(id % NKB);
  const int part = (int)blockIdx.y;
  X += (long long)part * x_part_stride;
  A8 += (long long)part * i8p::kM * K;
  if (Ap) Ap += (long long)part * i8p::kM * K;
  SA += (long long)part * NKB * i8p::kM;
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

// The coarse (R4DX_PREFILL_INT8_SCALES=coarse) producer: one scale per ROW over the whole K instead of per (row, 128 k);
// the arithmetic is i8g_quant_row_wg's (r4d_trellis_i8_fused.h), the one function the fused producers of
// src/kernels/src/trellis_transform.hip call too. Grid (256 rows, parts), 256 threads; X as i8g_quant_act's (part p at
// x_part_stride elements), A8 part p at + p 256 K bytes, SA part p at + p 256 floats ([256] per part). 32 bytes of dynamic LDS.
__global__ __launch_bounds__(256) void i8g_quant_act_row(const unsigned short* __restrict__ X, signed char* __restrict__ A8,
                                                         float* __restrict__ SA, int K, long long x_part_stride) {
  I8G_LDS_DECL
  const int row = (int)blockIdx.x, part = (int)blockIdx.y;
  i8g_quant_row_wg(X + (long long)part * x_part_stride + (size_t)row * K, K, row, A8 + (long long)part * i8p::kM * K,
                   SA + (long long)part * i8p::kM, (float*)lds);
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
// The coarse table (R4DX_PREFILL_INT8_SCALES=coarse): SWC[n], ONE entry per column n of Q over the whole K, by the same
// rule as the per-128 table (s = amax / 127 with amax over all of K, 1 for an all-zero column, rs = f16(min(1 / s, 60000)),
// stored s_eff = 1 / rs). One workgroup (8 waves) per tile pair: wave w takes the 128-groups w, w + 8, ...; the amax is
// exact, so the order of the reduction (xor 16 joins the two k halves of a fragment, then 8 waves through LDS) is free.
// Grid N / 32, 256 threads, 8 waves x 16 lanes x 2 floats = 1 KiB of dynamic LDS.
template <int KB>
__global__ __launch_bounds__(256) void i8g_wscale_col(const unsigned* __restrict__ W, float* __restrict__ SWC, int K, int N) {
  (void)N;
  I8G_LDS_DECL
  float* red = (float*)lds;
  const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
  const int kt_total = K >> 4, groups = kt_total >> 3, pair = (int)blockIdx.x;
  float m0 = 0.f, m1 = 0.f;
  for (int g = wave; g < groups; g += 8)
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
    red[(wave * 16 + lane) * 2] = m0;
    red[(wave * 16 + lane) * 2 + 1] = m1;
  }
  i8g_sync_lds();
  if (tid < 16) {
    float a = 0.f, b = 0.f;
    for (int w = 0; w < 8; ++w) {
      a = __builtin_fmaxf(a, red[(w * 16 + tid) * 2]);
      b = __builtin_fmaxf(b, red[(w * 16 + tid) * 2 + 1]);
    }
    const int n = i8p::FragCol(pair, tid, 0);
    SWC[n] = i8g_seff(a);
    SWC[n + 8] = i8g_seff(b);
  }
}
// W8 (block layout, 16 B per lane) and, if Wp is non-null, the plain [N][K] matrix, from the words and the table
// with the quantizer the GEMM runs: the dense kernel's weights ARE the trellis kernel's. COARSE: the table is
// i8g_wscale_col's ([N], one entry per column), not the per-128 [K / 128][N].
template <int KB, bool COARSE = false>
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
  const float* row = COARSE ? SW : SW + (size_t)(kt >> 3) * N;
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
// A8_0 / SA_0 (output columns < n_split) and A8_1 / SA_1 (the rest; a single-part launch passes the first pair
// twice and n_split = N): the activations (A8 layout, SA [K/128][256]); Wsrc: trellis pair-grid words (TRELLIS) or
// W8; SW [K/128][N] indexed by the absolute output column; svh [N] fp32 and out_scale: the trellis epilogue's
// (FWHT); C bf16 [256][N]; ws: SKG x 256 x N fp32; tickets: N / 128 zeroed words (self-resetting). Grid
// (N / 32, SKG), block 128 SKW, LDS 8192 SKW bytes. n_split is a multiple of 128 (a 128-column group never
// straddles it), so a block's part is the part of its first column.
#ifndef I8G_MINWAVES
#define I8G_MINWAVES 8   // amdgpu_waves_per_eu lower bound: 8 caps the kernel at 192 VGPRs (two 16-wave workgroups per WGP)
#endif
#ifdef I8G_EMU
#define I8G_KATTR
#else
#define I8G_KATTR __attribute__((amdgpu_waves_per_eu(I8G_MINWAVES)))
#endif
template <bool TRELLIS, int KB, bool FWHT, int RESC, int SKW>
__global__ __launch_bounds__(128 * SKW) I8G_KATTR void i8g_kernel(
    const signed char* __restrict__ A8_0, const float* __restrict__ SA_0, const signed char* __restrict__ A8_1,
    const float* __restrict__ SA_1, int n_split, const unsigned char* __restrict__ Wsrc, const float* __restrict__ SW,
    const float* __restrict__ svh, unsigned short* __restrict__ C, float* __restrict__ ws,
    unsigned* __restrict__ tickets, int K, int N, int SKG, float out_scale) {
  constexpr int RG = 4, MT = 4, M = 256, Wc = 32, NBLK = 4;
  // RESC 4 is the PRODUCTION coarse mode (R4DX_PREFILL_INT8_SCALES=coarse): SA is [256] (one scale per row), SW is [N]
  // (one per column, the whole K), int32 accumulators over the whole K slice, one fp32 multiply by sa[row] * sw[col] at the
  // end. Its math is RESC 1's (RESC 1 reads block 0 of the per-128 tables, RESC 4 reads the whole table that exists), but
  // the loop touches neither SA nor SW: the weight scale is loaded once and its f16 rs formed once, not per 128 K.
  constexpr bool kColW = RESC == 4;
  constexpr bool kUseSA = RESC != 1 && RESC != 4;   // the variants whose loop reads the per-128 activation scales
  constexpr int WBLK = TRELLIS ? 64 * KB : 512;           // bytes of one (tile pair, k-tile) block in Wsrc
  constexpr unsigned slot_bytes = 8 * 512;                // one K slice's staging per buffer: 8 blocks of 512 B
  constexpr unsigned buf_bytes = SKW * slot_bytes;
  static_assert(SKW == 2 || SKW == 4 || SKW == 8, "");
  I8G_LDS_DECL
  const int tid = threadIdx.x, lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int rg = wave & 3, ks = wave >> 2;
  const int pair0 = blockIdx.x, n0 = pair0 * Wc;
  const signed char* A8 = n0 >= n_split ? A8_1 : A8_0;     // the part of this block (the shipped kernel's rule)
  const float* SA = n0 >= n_split ? SA_1 : SA_0;
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
    i8g_gptr swbase = (i8g_gptr)(const unsigned char*)SW + (kColW ? (size_t)0 : (size_t)kbg0 * N * 4);
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
      if constexpr (kUseSA) {
        sav[0] = i8g_load<i8g_v4f>(sabase, saoff, i * 64);
        sav[1] = i8g_load<i8g_v4f>(sabase, saoff, i * 64 + 16);
      }
    };

    unsigned rs0h = 0, rs1h = 0;   // RESC 4: the rs pairs of the column scales, formed once for the whole K
    if constexpr (kColW) {
      load_sw(0);
      if constexpr (TRELLIS) {
        rs0h = i8g_rs_pair(swc0);
        rs1h = i8g_rs_pair(swc1);
      }
    }
    load_w(0);
    if constexpr (!kColW) load_sw(0);
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
      if constexpr (!kColW) load_sw(more);
      load_a(0);
      __builtin_amdgcn_sched_barrier(0);
      {
        unsigned rs0 = 0, rs1 = 0;
        if constexpr (TRELLIS) {
          if constexpr (kColW) {
            rs0 = rs0h;
            rs1 = rs1h;
          } else {
            rs0 = i8g_rs_pair(s0c);
            rs1 = i8g_rs_pair(s1c);
          }
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
        if constexpr (kUseSA) {
          sac[0] = sav[0];
          sac[1] = sav[1];
        }
        if (i + 1 < MT) load_a(i + 1);                    // the next pass's A and sa, behind this pass's WMMA
        __builtin_amdgcn_sched_barrier(0);
        if constexpr (RESC != 1 && RESC != 4) {
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
      if constexpr (kUseSA) i8g_opaque(sabase);
      i8g_opaque(wbase);
      if constexpr (!kColW) i8g_opaque(swbase);
#pragma unroll
      for (int i = 0; i < NOFF; ++i) i8g_opaque_v(woff[i]);
      __builtin_amdgcn_sched_barrier(0);
    }
  }

  if constexpr (RESC != 0) {
    // the coarse variants' one multiply with the scale(s) of block 0 of the whole K (a per-row / per-column scale;
    // speed bounds, not the per-128 math): RESC 1 both scales (the int32 accumulators ran over the whole slice),
    // RESC 2 the activation scale, RESC 3 the weight scale (the fp32 accumulators already hold the other, per-block,
    // one). Every slice multiplies its own partial sum by the same factors, so the slices still add up. RESC 4 is
    // RESC 1's multiply with the tables that actually hold one scale per row / column ([256], [N]).
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
        if constexpr (RESC == 1 || RESC == 4) {
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
