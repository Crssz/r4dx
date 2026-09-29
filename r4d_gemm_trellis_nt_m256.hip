// r4d_gemm_trellis_nt_m256.hip - bigger-M sibling of r4d_gemm_trellis_nt_m64.hip (in R4D_UNITS since the
// 256-row prefill chunk spike, R4DX_PREFILL_CHUNK; r4d.h has the contract): the same
// trellis linear (C = bf16( (FWHT128(A @ Q) * svh) * out_scale )) for M = 64 RG rows in ONE launch,
// RG = 2 or 4, that gives every output element the bits the M = 64 kernel gives it at the same
// (SK, SKG): the K loop keeps each accumulator's chain of 16-k WMMA in k order, the SK partial sums are
// added in slice order from 0.f, the SKG partials in y order from 0.f, and the FWHT stages, svh, out
// scale and the one bf16 rounding are the stock ones. Row tiles are independent in WMMA, so nothing
// else can differ. Bench and check tools: tests/kernels/tool_trellis_m256_bench.hip, tool_trellis_m256_real_check.hip;
// ctest: tests/kernels/test_trellis_m256.cpp.
//
// WHY IT IS FASTER.  The M = 64 kernel is not DRAM-bound: it is the decode (62 VALU per weight block at
// KB = 4, 75 at KB = 5) plus the WMMA, whose times ADD (VALU and WMMA do not overlap on a SIMD). The
// decode is per weight, not per row, so at M = 256 it is shared: a workgroup is RG * SK waves, wave
// (rg, ks) owns rows 64 rg.. and K slice ks with the stock MT = 4 x NP accumulator tile, and per step of
// U k-tiles the RG waves of one K slice each decode ONE of the slice's NP * U (tile pair, k-tile)
// blocks, publish the two f16 fragments (ds_store_b128) in a double-buffered LDS slot, and after one
// workgroup barrier every wave reads all of them (ds_load_b128) and runs its NP * U * 2 * 4 WMMA.
// The decode and its weight loads are amortised over RG * 64 rows; LDS traffic costs the WMMA nothing
// (128 B/clk per WGP against the ~31 B/clk needed, docs/trellis-kernel.md 4.7 / linear A1).
//
// ANY SHIPPED SK (2, 4, 8, 16).  A workgroup is RG * SKW waves at most 32, so the K slices of a shipped row
// with SK > 32 / RG (mlp.down KB 4: SK 16, SKG 2) cannot all be resident. The kernel therefore walks PH =
// SK / SKW groups of SKW slices one after the other (slice = group * SKW + wave's K index, so every slice keeps
// its own k-tile range and its own chain of 16-k WMMA restarted from zero). The shipped SK sum is a left fold
// from 0.f in slice order; each group ends with the in-workgroup reduction (below) whose owner starts its sum
// from the running sum of the earlier groups (parked in the owner's ws unit, the unit the final sum lands in)
// instead of 0.f, which adds the same terms in the same order: the result is bit-identical to the shipped row.
// A slice's k-tile count need not be a multiple of U: TAIL = ktw % U trailing k-tiles run as one short step in
// which only the waves whose block index is below TAIL decode (mlp.down KB 4: 34 k-tiles, U 4, TAIL 2).
//
// EPILOGUE (all of it in the accumulator layout, so every LDS and global access is a lane-contiguous
// b128). Wc = 32 NP columns per block, so a 128-column FWHT group is spread over NBLK = 128 / Wc
// blocks (and SKG K-blocks): a "split" group in the stock kernel's words, finished by the LAST block to
// arrive (ticket). 1. SK reduction: tile i is owned by K slice i % SK; in pass i the other slices
// write their acc[.][.][i] to LDS (12 b128 per lane at most), the owner reads them and sums in slice
// order (its own slice's term taken from its registers). 2. The owner writes the sum to ws in lane
// order and takes a ticket. 3. The last block's owner of tile i sums the SKG x NBLK ws units in y
// order, runs all seven FWHT stages in registers (three cross-lane, one across the f pair, one across
// the t half-row, two across the p / block index), scales and stores the bf16.
#include "r4d.h"
#include "r4d_fwht128.h"
#include "r4d_gdn_wmma.h"
#include "r4d_trellis_dq.h"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <type_traits>

#define WAVE32 32

typedef __attribute__((address_space(1))) const unsigned char* r4d_g256byte;
typedef float r4d_t256_v4f __attribute__((ext_vector_type(4)));

template <typename T>
__device__ __forceinline__ T r4d_t256_load(r4d_g256byte base, unsigned off, int imm) {
  return *(__attribute__((address_space(1))) const T*)(base + off + imm);
}
__device__ __forceinline__ void r4d_t256_opaque(r4d_g256byte& p) {
  unsigned long long v = (unsigned long long)p;
  asm("" : "+s"(v));
  p = (r4d_g256byte)v;
}
__device__ __forceinline__ void r4d_t256_opaque_v(unsigned& v) { asm("" : "+v"(v)); }
__device__ __forceinline__ v8f r4d_t256_mma(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ void r4d_t256_sync_lds() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}
__device__ __forceinline__ unsigned short r4d_t256_bf16_rn(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (unsigned short)((u >> 16) | 0x40u);
  return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// One FWHT butterfly stage across lanes l and l ^ MASK (MASK = 1, 2, 4, 8: DPP row_xmask), on every
// float of the lane's values: the lower lane computes a + b, the upper a - b, both as one exact fma.
template <int MASK, int N>
__device__ __forceinline__ void r4d_t256_lane_stage(v8f (&v)[N], int lane) {
  const float sgn = (lane & MASK) ? -1.f : 1.f;
#pragma unroll
  for (int k = 0; k < N; ++k)
#pragma unroll
    for (int e = 0; e < 8; ++e) v[k][e] = __builtin_fmaf(sgn, v[k][e], r4d_fwht_partner<MASK>(v[k][e]));
}
// The in-register stage: (lo, hi) -> (lo + hi, lo - hi).
__device__ __forceinline__ void r4d_t256_bfly(v8f& lo, v8f& hi) {
  const v8f a = lo, b = hi;
  lo = a + b;
  hi = a - b;
}

// SK = SKW * PH is the K-slice count of the shipped M = 64 row this launch must reproduce; a workgroup holds
// only SKW slices at a time (RG * SKW <= 32 waves) and walks the PH groups of SKW slices one after the
// other. The shipped sum is a left fold over the SK slice partials from 0.f, so folding group by group
// (the running sum handed from one group to the next through the ws unit the group's owner will finish in)
// adds the very same terms in the very same order.
template <int KB, int NP, int U, int RG, int SKW, int PH, int TAIL>
__global__ __launch_bounds__(1024) void r4d_gemm_trellis_nt_m256_kernel(
    const unsigned short* __restrict__ A0, const unsigned short* __restrict__ A1, int n_split,
    const unsigned* __restrict__ W, const float* __restrict__ svh, unsigned short* __restrict__ C,
    float* __restrict__ ws, unsigned* __restrict__ tickets, int K, int N, int SKG, float out_scale) {
  constexpr int MT = 4, BLK = 64 * KB, M = RG * 64, Wc = NP * 32, NBLK = 128 / Wc;
  constexpr int NQ = NP * 4;                              // b128 per lane per accumulator tile
  constexpr int SK = SKW * PH;
  static_assert(NP * U == RG, "each wave of a K slice decodes exactly one block per step");
  static_assert(SKW == 2 || SKW == 4 || SKW == 8, "");
  static_assert(PH == 1 || PH == 2 || PH == 4, "");
  static_assert(TAIL >= 0 && TAIL < U, "");
  extern __shared__ __attribute__((aligned(16))) unsigned char lds[];
  const int tid = threadIdx.x, lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int rg = wave % RG, ks = wave / RG;
  const int n0 = blockIdx.x * Wc;
  const int pair0 = n0 / 32;
  const int kt_total = K / 16;
  const int ktw = kt_total / (SK * SKG);
  const int steps = ktw / U;                              // main steps of U k-tiles; TAIL = ktw % U more k-tiles in one last step
  const int p_d = rg % NP, u_d = rg / NP;                 // the block this wave decodes each step
  constexpr unsigned slot_bytes = NP * U * 1024;          // one K slice's staging per buffer
  constexpr unsigned buf_bytes = SKW * slot_bytes;
  const unsigned lds_w = ks * slot_bytes + rg * 1024 + lane * 16;   // own block's F0 (F1 at +512)
  const unsigned lds_r = ks * slot_bytes + lane * 16;               // block j's F0 at + j * 1024
  const unsigned short* A = n0 >= n_split ? A1 : A0;
  float* wsy = ws + (size_t)blockIdx.y * M * N;
  // the ws unit of (row group rg, tile i): the running sum lives here between groups, the final sum after
  auto ws_unit = [&](int i) {
    return wsy + ((size_t)(rg * 4 + i) * (N / Wc) + n0 / Wc) * (16 * Wc);
  };

  v8f acc[NP][2][MT];
#pragma unroll
  for (int ph = 0; ph < PH; ++ph) {
#pragma unroll
  for (int p = 0; p < NP; ++p)
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
      for (int i = 0; i < MT; ++i) acc[p][f][i] = (v8f)(0.f);
  const int kt0 = ((int)blockIdx.y * SK + ph * SKW + ks) * ktw;

  {
    r4d_g256byte wb = (r4d_g256byte)(const unsigned char*)W +
                      ((size_t)(pair0 + p_d) * kt_total + kt0 + u_d) * BLK;
    r4d_g256byte abase = (r4d_g256byte)(const unsigned char*)A + (size_t)kt0 * 32;
    const int lane_o = lane;
    constexpr int NOFF = KB == 4 ? 2 : 5;
    unsigned woff[NOFF];
    if constexpr (KB == 4) {
      woff[0] = r4d_trellis_k4_off_ab(lane_o);
      woff[1] = r4d_trellis_k4_off_p(lane_o);
    } else {
#pragma unroll
      for (int i = 0; i < 5; ++i) woff[i] = r4d_trellis_k5_word(lane_o, i) * 4u;
    }
    const unsigned k5sh = r4d_trellis_k5_shift(lane_o);
    unsigned aoff[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) aoff[i] = rowGo((unsigned)K, rg * 64 + i * 16, M - 1, lane_o) * 2u;

    // the lane's words of the wave's own block of a step
    unsigned long long ab;
    unsigned pw;
    unsigned w5[5];
    auto load_w = [&](int ahead) {
      r4d_g256byte b = wb + (size_t)ahead * (U * BLK);
      r4d_t256_opaque(b);
      if constexpr (KB == 4) {
        ab = r4d_t256_load<unsigned long long>(b, woff[0], 0);
        pw = r4d_t256_load<unsigned>(b, woff[1], 0);
      } else {
#pragma unroll
        for (int i = 0; i < 5; ++i) w5[i] = r4d_t256_load<unsigned>(b, woff[i], 0);
      }
    };
    load_w(0);
    int buf = 0;
    // One step over UU k-tiles of the slice (UU = U for the main steps; the tail of a slice whose k-tile
    // count is not a multiple of U runs one step of UU = ktw % U, in which only the waves whose block
    // index is below UU decode). The k-tile order of every accumulator chain is the slice's own.
    auto do_step = [&](auto ucst, int st, int nxt) {
      constexpr int UU = decltype(ucst)::value;
      // phase D: this step's activation fragments, then decode the own block and publish it
      v8h af[UU][MT];
      {
        r4d_g256byte a = abase + (size_t)st * (U * 32);
        r4d_t256_opaque(a);
#pragma unroll
        for (int u = 0; u < UU; ++u)
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            union { unsigned long long w[2]; v8h f; } x;
            x.w[0] = r4d_t256_load<unsigned long long>(a, aoff[i], u * 32);
            x.w[1] = r4d_t256_load<unsigned long long>(a, aoff[i], u * 32 + 16);
            af[u][i] = x.f;
          }
      }
      const unsigned long long ab_c = ab;
      const unsigned pw_c = pw;
      unsigned c5[5];
#pragma unroll
      for (int i = 0; i < 5; ++i) c5[i] = w5[i];
      if constexpr (UU == U) load_w(nxt);                 // next step's words (clamped)
      __builtin_amdgcn_sched_barrier(0);
      if (UU == U || u_d < UU) {
        v8h f0, f1;
        if constexpr (KB == 4) r4d_trellis_k4_decode(pw_c, (unsigned)ab_c, (unsigned)(ab_c >> 32), f0, f1);
        else r4d_trellis_k5_decode(c5, k5sh, f0, f1);
        *(v8h*)(lds + buf * buf_bytes + lds_w) = f0;
        *(v8h*)(lds + buf * buf_bytes + lds_w + 512) = f1;
      }
      __builtin_amdgcn_sched_barrier(0);
      r4d_t256_sync_lds();
      // phase W: every block of the slice, all four row tiles
#pragma unroll
      for (int u = 0; u < UU; ++u)
#pragma unroll
        for (int p = 0; p < NP; ++p) {
          const unsigned rd = buf * buf_bytes + lds_r + (u * NP + p) * 1024;
          const v8h b0 = *(const v8h*)(lds + rd);
          const v8h b1 = *(const v8h*)(lds + rd + 512);
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            acc[p][0][i] = r4d_t256_mma(af[u][i], b0, acc[p][0][i]);
            acc[p][1][i] = r4d_t256_mma(af[u][i], b1, acc[p][1][i]);
          }
        }
      buf ^= 1;
      wb += (size_t)U * BLK;
      r4d_t256_opaque(wb);
#pragma unroll
      for (int i = 0; i < NOFF; ++i) r4d_t256_opaque_v(woff[i]);
#pragma unroll
      for (int i = 0; i < MT; ++i) r4d_t256_opaque_v(aoff[i]);
      __builtin_amdgcn_sched_barrier(0);
    };
    for (int st = 0; st < steps; ++st) {
      // the last main step preloads the tail block (only the waves that will decode it), else re-reads
      const bool more = st + 1 < steps || (TAIL > 0 && u_d < TAIL);
      do_step(std::integral_constant<int, U>{}, st, more ? 1 : 0);
    }
    if constexpr (TAIL > 0) do_step(std::integral_constant<int, TAIL>{}, steps, 0);
  }

  // ---- 1. reduction over the group's SKW slices, in the accumulator layout ------------------------
  // pass i: slice i % SKW owns tile i; the others hand it their acc[.][.][i]. Region of (rg, writer kw):
  // NQ units of 512 B (32 lanes x 16 B). The owner's sum starts from 0.f (first group) or from the
  // running sum of the earlier groups (read back from its own ws unit).
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    const int o = i % SKW;
    r4d_t256_sync_lds();
    if (ks != o) {
      const int kw = ks > o ? ks - 1 : ks;
      unsigned char* dst = lds + ((rg * (SKW - 1) + kw) * NQ) * 512 + lane * 16;
#pragma unroll
      for (int p = 0; p < NP; ++p)
#pragma unroll
        for (int f = 0; f < 2; ++f) {
          const v8f x = acc[p][f][i];
          const int q = (p * 2 + f) * 2;
          *(r4d_t256_v4f*)(dst + (q + 0) * 512) = __builtin_shufflevector(x, x, 0, 1, 2, 3);
          *(r4d_t256_v4f*)(dst + (q + 1) * 512) = __builtin_shufflevector(x, x, 4, 5, 6, 7);
        }
    }
    r4d_t256_sync_lds();
    if (ks == o) {
      const unsigned char* src = lds + ((rg * (SKW - 1)) * NQ) * 512 + lane * 16;
      const float* unit_prev = ws_unit(i);
#pragma unroll
      for (int p = 0; p < NP; ++p)
#pragma unroll
        for (int f = 0; f < 2; ++f) {
          const int q = (p * 2 + f) * 2;
          v8f s = (v8f)(0.f);
          if (ph > 0) {
            const r4d_t256_v4f lo = *(const r4d_t256_v4f*)(unit_prev + ((q + 0) * 32 + lane) * 4);
            const r4d_t256_v4f hi = *(const r4d_t256_v4f*)(unit_prev + ((q + 1) * 32 + lane) * 4);
            s = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
          }
#pragma unroll
          for (int sl = 0; sl < SKW; ++sl) {
            v8f term;
            if (sl == o) {
              term = acc[p][f][i];
            } else {
              const int kw = sl > o ? sl - 1 : sl;
              const r4d_t256_v4f lo = *(const r4d_t256_v4f*)(src + (kw * NQ + q + 0) * 512);
              const r4d_t256_v4f hi = *(const r4d_t256_v4f*)(src + (kw * NQ + q + 1) * 512);
              term = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
            }
            s = s + term;
          }
          acc[p][f][i] = s;
        }
    }
  }

  // ---- 2. hand the sums to ws in lane order (the running sum, or the final one), take a ticket -----
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    if (ks == i % SKW) {
      float* unit = ws_unit(i);
#pragma unroll
      for (int p = 0; p < NP; ++p)
#pragma unroll
        for (int f = 0; f < 2; ++f) {
          const v8f x = acc[p][f][i];
          const int q = (p * 2 + f) * 2;
          *(r4d_t256_v4f*)(unit + ((q + 0) * 32 + lane) * 4) = __builtin_shufflevector(x, x, 0, 1, 2, 3);
          *(r4d_t256_v4f*)(unit + ((q + 1) * 32 + lane) * 4) = __builtin_shufflevector(x, x, 4, 5, 6, 7);
        }
    }
  }
  if (ph + 1 < PH) {
    // the running sum is read back by the same lanes next group; the barrier also keeps the next group's
    // staging stores behind every wave's reduction reads of the shared LDS
    __builtin_amdgcn_fence(__ATOMIC_ACQ_REL, "workgroup");
    r4d_t256_sync_lds();
  }
  }   // groups of SKW slices
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

  // ---- 3. the last block finishes the group: y-sum, FWHT, svh, scale, bf16 ----------------------
  const int t = (lane >> 3) & 1, c = lane & 7, hh = lane >> 4;
  const int cb0 = g0 / Wc;
  float sv[NBLK][NP][2];
#pragma unroll
  for (int b = 0; b < NBLK; ++b)
#pragma unroll
    for (int p = 0; p < NP; ++p)
#pragma unroll
      for (int f = 0; f < 2; ++f) sv[b][p][f] = svh[g0 + b * Wc + p * 32 + 16 * t + 8 * f + c];
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    if (ks != i % SKW) continue;
    const int R = rg * 4 + i;
    v8f v[NBLK * NP * 2];                                 // index (b * NP + p) * 2 + f
#pragma unroll
    for (int k = 0; k < NBLK * NP * 2; ++k) v[k] = (v8f)(0.f);
    for (int y = 0; y < SKG; ++y) {
#pragma unroll
      for (int b = 0; b < NBLK; ++b) {
        const float* unit = ws + (size_t)y * M * N + ((size_t)R * (N / Wc) + cb0 + b) * (16 * Wc);
#pragma unroll
        for (int p = 0; p < NP; ++p)
#pragma unroll
          for (int f = 0; f < 2; ++f) {
            const int q = (p * 2 + f) * 2;
            const r4d_t256_v4f lo = *(const r4d_t256_v4f*)(unit + ((q + 0) * 32 + lane) * 4);
            const r4d_t256_v4f hi = *(const r4d_t256_v4f*)(unit + ((q + 1) * 32 + lane) * 4);
            v[(b * NP + p) * 2 + f] = v[(b * NP + p) * 2 + f] + __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
          }
      }
    }
    // FWHT-128 over the group's columns: bit 0-2 lane bits 0-2 (c), bit 3 f, bit 4 lane bit 3 (t),
    // bit 5 p (NP = 2) or block bit 0 (NP = 1), bit 6 block bit 0 / 1.
    r4d_t256_lane_stage<1>(v, lane);
    r4d_t256_lane_stage<2>(v, lane);
    r4d_t256_lane_stage<4>(v, lane);
#pragma unroll
    for (int bp = 0; bp < NBLK * NP; ++bp) r4d_t256_bfly(v[bp * 2], v[bp * 2 + 1]);   // bit 3
    r4d_t256_lane_stage<8>(v, lane);                                                    // bit 4
    if constexpr (NP == 2) {
#pragma unroll
      for (int b = 0; b < NBLK; ++b)
#pragma unroll
        for (int f = 0; f < 2; ++f) r4d_t256_bfly(v[(b * 2 + 0) * 2 + f], v[(b * 2 + 1) * 2 + f]);  // bit 5
#pragma unroll
      for (int p = 0; p < 2; ++p)
#pragma unroll
        for (int f = 0; f < 2; ++f) r4d_t256_bfly(v[(0 * 2 + p) * 2 + f], v[(1 * 2 + p) * 2 + f]);  // bit 6
    } else {
#pragma unroll
      for (int f = 0; f < 2; ++f) {
        r4d_t256_bfly(v[0 * 2 + f], v[1 * 2 + f]);                                                // bit 5
        r4d_t256_bfly(v[2 * 2 + f], v[3 * 2 + f]);
      }
#pragma unroll
      for (int f = 0; f < 2; ++f) {
        r4d_t256_bfly(v[0 * 2 + f], v[2 * 2 + f]);                                                // bit 6
        r4d_t256_bfly(v[1 * 2 + f], v[3 * 2 + f]);
      }
    }
#pragma unroll
    for (int b = 0; b < NBLK; ++b)
#pragma unroll
      for (int p = 0; p < NP; ++p)
#pragma unroll
        for (int f = 0; f < 2; ++f) {
          const int col = g0 + b * Wc + p * 32 + 16 * t + 8 * f + c;
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            const int row = R * 16 + 8 * hh + e;
            C[(size_t)row * N + col] =
                r4d_t256_bf16_rn((v[(b * NP + p) * 2 + f][e] * sv[b][p][f]) * out_scale);
          }
        }
  }
}


// ---- host ------------------------------------------------------------------------------------

static void r4d_t256_fail(const std::string& what) {
  throw std::runtime_error("r4d gemm_trellis_nt_m256: " + what);
}

struct r4d_t256_args {
  dim3 grid, block;
  size_t smem;
  hipStream_t stream;
  const unsigned short *a0, *a1;
  int n_split;
  const unsigned* w;
  const float* svh;
  unsigned short* c;
  float* ws;
  unsigned* tickets;
  int K, N, SKG;
  float out_scale;
};

template <int KB, int NP, int U, int RG, int SKW, int PH, int TAIL>
static void r4d_t256_launch(const r4d_t256_args& a) {
  hipLaunchKernelGGL((r4d_gemm_trellis_nt_m256_kernel<KB, NP, U, RG, SKW, PH, TAIL>), a.grid, a.block, a.smem,
                     a.stream, a.a0, a.a1, a.n_split, a.w, a.svh, a.c, a.ws, a.tickets, a.K, a.N,
                     a.SKG, a.out_scale);
}

// THE instantiation table: which (KB, RG, NP, U, SKW, PH, TAIL) exist. One predicate serves the launch and
// the host check, so they cannot disagree, and it is the only place a kernel gets instantiated -- the build
// (third_party/check_trellis_isa.cmake) requires every instantiation of this unit to fit 190 VGPRs with no
// scratch, so a combination that does not (measured, resource report of docs/trellis-m256.md) is not listed:
//   * (RG, NP, U) = (4, 1, 4) for M = 256; (2, 2, 1) and (2, 1, 2) for M = 128 ((4, 2, 2) spills).
//   * SKW = 8 needs NP = 1 (the LDS of NP = 2 would not fit); PH > 1 exists for (RG 4, NP 1) only.
//   * PH = 1: any KB, TAIL 0.
//   * PH > 1 (SK = SKW PH > SKW: the shipped rows with SK 8 and 16, and SK 4 at SKW 2): KB 4 only, because
//     the KB 5 decode adds registers (192 VGPRs at PH 2 and 4, scratch at (4, 4)); (2,2), (2,4), (4,2),
//     (8,2) with TAIL 0 and the two k-tile-tail ones mlp.down KB 4 needs: (4,4) and (8,2) with TAIL 2
//     (34 k-tiles per slice = 8 steps of U = 4 and a tail of 2). (4,4) with TAIL 0 spills (192 VGPRs, 8 B
//     scratch) and is not used by any shape of the model, so it is not instantiated either.
template <int KB, int NP, int U, int RG, int SKW, int PH, int TAIL>
static constexpr bool r4d_t256_inst() {
  if (!((RG == 4 && NP == 1 && U == 4) || (RG == 2 && NP == 2 && U == 1) || (RG == 2 && NP == 1 && U == 2)))
    return false;
  if (SKW == 8 && NP != 1) return false;
  if (PH == 1) return TAIL == 0;
  if (!(RG == 4 && NP == 1) || KB != 4) return false;
  if (PH == 2) return TAIL == 0 || (SKW == 8 && TAIL == 2);
  return (SKW == 2 && TAIL == 0) || (SKW == 4 && TAIL == 2);   // PH == 4
}
template <int KB, int NP, int U, int RG, int SKW, int PH, int TAIL>
struct r4d_t256_cfg {
  static void launch(const r4d_t256_args& a) { r4d_t256_launch<KB, NP, U, RG, SKW, PH, TAIL>(a); }
};

// f(r4d_t256_cfg<...>{}) is called for the instantiated combination, if there is one (returns true), so the
// launch (f = launch) and the check (f = nothing) share the table above.
template <int KB, int NP, int U, int RG, class F>
static bool r4d_t256_dispatch_sk(int SKW, int PH, int TAIL, F& f) {
#define R4D_T256_CASE(W_, P_, T_)                                                                       \
  if (SKW == W_ && PH == P_ && TAIL == T_) {                                                            \
    if constexpr (r4d_t256_inst<KB, NP, U, RG, W_, P_, T_>()) {                                         \
      f(r4d_t256_cfg<KB, NP, U, RG, W_, P_, T_>{});                                                     \
      return true;                                                                                      \
    } else {                                                                                            \
      return false;                                                                                     \
    }                                                                                                   \
  }
  R4D_T256_CASE(2, 1, 0) R4D_T256_CASE(4, 1, 0) R4D_T256_CASE(8, 1, 0)
  R4D_T256_CASE(2, 2, 0) R4D_T256_CASE(4, 2, 0) R4D_T256_CASE(8, 2, 0) R4D_T256_CASE(8, 2, 2)
  R4D_T256_CASE(2, 4, 0) R4D_T256_CASE(4, 4, 0) R4D_T256_CASE(4, 4, 2)
#undef R4D_T256_CASE
  return false;
}
template <int KB, class F>
static bool r4d_t256_dispatch_kb(int RG, int NP, int U, int SKW, int PH, int TAIL, F& f) {
  if (RG == 4 && NP == 1 && U == 4) return r4d_t256_dispatch_sk<KB, 1, 4, 4>(SKW, PH, TAIL, f);
  if (RG == 2 && NP == 2 && U == 1) return r4d_t256_dispatch_sk<KB, 2, 1, 2>(SKW, PH, TAIL, f);
  if (RG == 2 && NP == 1 && U == 2) return r4d_t256_dispatch_sk<KB, 1, 2, 2>(SKW, PH, TAIL, f);
  return false;
}
template <class F>
static bool r4d_t256_dispatch(int KB, int RG, int NP, int U, int SKW, int PH, int TAIL, F&& f) {
  return KB == 4 ? r4d_t256_dispatch_kb<4>(RG, NP, U, SKW, PH, TAIL, f)
                 : r4d_t256_dispatch_kb<5>(RG, NP, U, SKW, PH, TAIL, f);
}

const char* r4d_gemm_trellis_nt_m256_check(int M, int K, int N, int n_split, int KB, int SK, int NP,
                                           int SKG, int U, int skw) {
  static thread_local char msg[320];
#define R4D_T256_BAD(...) do { std::snprintf(msg, sizeof msg, __VA_ARGS__); return msg; } while (0)
  if (M != 128 && M != 256) R4D_T256_BAD("M must be 128 or 256, got %d", M);
  const int RG = M / 64;
  if (K <= 0 || N <= 0 || K % 128 || N % 128) R4D_T256_BAD("K and N must be multiples of 128, got K %d N %d", K, N);
  if (KB != 4 && KB != 5) R4D_T256_BAD("KB must be 4 or 5, got %d", KB);
  if (SKG != 1 && SKG != 2 && SKG != 4 && SKG != 8) R4D_T256_BAD("SKG must be 1, 2, 4 or 8, got %d", SKG);
  if (NP != 1 && NP != 2) R4D_T256_BAD("NP must be 1 or 2, got %d", NP);
  if (N % (NP * 32)) R4D_T256_BAD("N %d must be a multiple of the block width 32 NP = %d", N, NP * 32);
  if (n_split < 0 || n_split > N || n_split % 128)
    R4D_T256_BAD("n_split %d must be in 0..N (%d) and a multiple of 128", n_split, N);
  if (NP * U != RG) R4D_T256_BAD("NP * U (%d) must equal M / 64 (%d)", NP * U, RG);
  if (SK != 2 && SK != 4 && SK != 8 && SK != 16) R4D_T256_BAD("SK must be 2, 4, 8 or 16, got %d", SK);
  if ((K / 16) % (SK * SKG)) R4D_T256_BAD("(K/16) = %d must be divisible by SK*SKG = %d", K / 16, SK * SKG);
  const int ktw = (K / 16) / (SK * SKG);
  if (ktw < U) R4D_T256_BAD("a K slice has %d k-tiles, fewer than U = %d", ktw, U);
  const int SKW = skw > 0 ? skw : (SK < 4 ? SK : 4);
  if (SKW < 2 || SK % SKW) R4D_T256_BAD("skw %d must be 2, 4 or 8 and divide SK %d", SKW, SK);
  if (RG * SKW > 32) R4D_T256_BAD("RG * skw = %d waves exceed the 32-wave workgroup", RG * SKW);
  const size_t stage = (size_t)2 * SKW * NP * U * 1024;
  const size_t red = (size_t)RG * (SKW - 1) * NP * 4 * 512;
  if ((stage > red ? stage : red) > 64 * 1024) R4D_T256_BAD("LDS %zu B exceeds 64 KiB", stage > red ? stage : red);
  const int PH = SK / SKW, TAIL = ktw % U;
  if (!r4d_t256_dispatch(KB, RG, NP, U, SKW, PH, TAIL, [](auto) {}))
    R4D_T256_BAD("(KB %d, RG %d, NP %d, U %d, skw %d x %d groups, k-tile tail %d) is not instantiated "
                 "(see r4d_t256_inst: PH > 1 only at KB 4, RG 4, NP 1; tails only (4,4,2) and (8,2,2))",
                 KB, RG, NP, U, SKW, PH, TAIL);
#undef R4D_T256_BAD
  return nullptr;
}

void r4d_gemm_trellis_nt_m256(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t svh, int64_t c,
                              int64_t ws, int64_t tickets, int M, int K, int N, int KB, int SK, int NP,
                              int SKG, int U, float out_scale, int64_t stream, int skw) {
  // SK is the shipped row's K-slice count. skw = K slices resident per workgroup (0 = the default,
  // min(SK, 4)); SK / skw groups are walked in turn.
  if (const char* why = r4d_gemm_trellis_nt_m256_check(M, K, N, n_split, KB, SK, NP, SKG, U, skw)) r4d_t256_fail(why);
  if (!w || !c || !svh || !ws || !tickets || !a0) r4d_t256_fail("null buffer");
  const int RG = M / 64, Wc = NP * 32;
  const int SKW = skw > 0 ? skw : (SK < 4 ? SK : 4);
  const int PH = SK / SKW;
  const int TAIL = ((K / 16) / (SK * SKG)) % U;
  r4d_t256_args a;
  a.grid = dim3(N / Wc, SKG, 1);
  a.block = dim3(RG * SKW * WAVE32);
  const size_t stage = (size_t)2 * SKW * NP * U * 1024;
  const size_t red = (size_t)RG * (SKW - 1) * NP * 4 * 512;
  a.smem = stage > red ? stage : red;
  a.stream = (hipStream_t)stream;
  a.a0 = (const unsigned short*)a0;
  a.a1 = (const unsigned short*)(a1 ? a1 : a0);
  a.n_split = n_split;
  a.w = (const unsigned*)w;
  a.svh = (const float*)svh;
  a.c = (unsigned short*)c;
  a.ws = (float*)ws;
  a.tickets = (unsigned*)tickets;
  a.K = K; a.N = N; a.SKG = SKG;
  a.out_scale = out_scale;
  r4d_t256_dispatch(KB, RG, NP, U, SKW, PH, TAIL, [&](auto cfg) { decltype(cfg)::launch(a); });
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) r4d_t256_fail(std::string("launch failed: ") + hipGetErrorString(e));
}

size_t r4d_gemm_trellis_nt_m256_ws_bytes(int M, int N, int SKG) {
  return (size_t)SKG * (size_t)M * (size_t)N * sizeof(float);
}
