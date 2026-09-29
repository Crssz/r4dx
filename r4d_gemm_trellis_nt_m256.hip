// r4d_gemm_trellis_nt_m256.hip - PROTOTYPE bigger-M sibling of r4d_gemm_trellis_nt_m64.hip: the same
// trellis linear (C = bf16( (FWHT128(A @ Q) * svh) * out_scale )) for M = 64 RG rows in ONE launch,
// RG = 2 or 4, that gives every output element the bits the M = 64 kernel gives it at the same
// (SK, SKG): the K loop keeps each accumulator's chain of 16-k WMMA in k order, the SK partial sums are
// added in slice order from 0.f, the SKG partials in y order from 0.f, and the FWHT stages, svh, out
// scale and the one bf16 rounding are the stock ones. Row tiles are independent in WMMA, so nothing
// else can differ. Bench-only: nothing in r4dx calls it (tests/kernels/tool_trellis_m256_bench.hip).
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
#include <stdexcept>
#include <string>

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

template <int KB, int NP, int U, int RG, int SK>
__global__ __launch_bounds__(1024) void r4d_gemm_trellis_nt_m256_kernel(
    const unsigned short* __restrict__ A0, const unsigned short* __restrict__ A1, int n_split,
    const unsigned* __restrict__ W, const float* __restrict__ svh, unsigned short* __restrict__ C,
    float* __restrict__ ws, unsigned* __restrict__ tickets, int K, int N, int SKG, float out_scale) {
  constexpr int MT = 4, BLK = 64 * KB, M = RG * 64, Wc = NP * 32, NBLK = 128 / Wc;
  constexpr int NQ = NP * 4;                              // b128 per lane per accumulator tile
  static_assert(NP * U == RG, "each wave of a K slice decodes exactly one block per step");
  static_assert(SK == 2 || SK == 4, "");
  extern __shared__ __attribute__((aligned(16))) unsigned char lds[];
  const int tid = threadIdx.x, lane = tid & 31;
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int rg = wave % RG, ks = wave / RG;
  const int n0 = blockIdx.x * Wc;
  const int pair0 = n0 / 32;
  const int kt_total = K / 16;
  const int ktw = kt_total / (SK * SKG);
  const int kt0 = ((int)blockIdx.y * SK + ks) * ktw;
  const int steps = ktw / U;
  const int p_d = rg % NP, u_d = rg / NP;                 // the block this wave decodes each step
  constexpr unsigned slot_bytes = NP * U * 1024;          // one K slice's staging per buffer
  constexpr unsigned buf_bytes = SK * slot_bytes;
  const unsigned lds_w = ks * slot_bytes + rg * 1024 + lane * 16;   // own block's F0 (F1 at +512)
  const unsigned lds_r = ks * slot_bytes + lane * 16;               // block j's F0 at + j * 1024
  const unsigned short* A = n0 >= n_split ? A1 : A0;

  v8f acc[NP][2][MT];
#pragma unroll
  for (int p = 0; p < NP; ++p)
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
      for (int i = 0; i < MT; ++i) acc[p][f][i] = (v8f)(0.f);

  {
    r4d_g256byte wb = (r4d_g256byte)(const unsigned char*)W +
                      ((size_t)(pair0 + p_d) * kt_total + kt0 + u_d) * BLK;
    r4d_g256byte abase = (r4d_g256byte)(const unsigned char*)A + (size_t)kt0 * 32;
    constexpr int NOFF = KB == 4 ? 2 : 5;
    unsigned woff[NOFF];
    if constexpr (KB == 4) {
      woff[0] = r4d_trellis_k4_off_ab(lane);
      woff[1] = r4d_trellis_k4_off_p(lane);
    } else {
#pragma unroll
      for (int i = 0; i < 5; ++i) woff[i] = r4d_trellis_k5_word(lane, i) * 4u;
    }
    const unsigned k5sh = r4d_trellis_k5_shift(lane);
    unsigned aoff[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) aoff[i] = rowGo((unsigned)K, rg * 64 + i * 16, M - 1, lane) * 2u;

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
    for (int st = 0; st < steps; ++st) {
      // phase D: this step's activation fragments, then decode the own block and publish it
      v8h af[U][MT];
      {
        r4d_g256byte a = abase + (size_t)st * (U * 32);
        r4d_t256_opaque(a);
#pragma unroll
        for (int u = 0; u < U; ++u)
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
      load_w(st + 1 < steps ? 1 : 0);                     // next step's words (clamped)
      __builtin_amdgcn_sched_barrier(0);
      v8h f0, f1;
      if constexpr (KB == 4) r4d_trellis_k4_decode(pw_c, (unsigned)ab_c, (unsigned)(ab_c >> 32), f0, f1);
      else r4d_trellis_k5_decode(c5, k5sh, f0, f1);
      *(v8h*)(lds + buf * buf_bytes + lds_w) = f0;
      *(v8h*)(lds + buf * buf_bytes + lds_w + 512) = f1;
      __builtin_amdgcn_sched_barrier(0);
      r4d_t256_sync_lds();
      // phase W: every block of the slice, all four row tiles
#pragma unroll
      for (int u = 0; u < U; ++u)
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
    }
  }

  // ---- 1. reduction over the SK slices, in the accumulator layout ---------------------------------
  // pass i: slice i % SK owns tile i; the others hand it their acc[.][.][i]. Region of (rg, writer kw):
  // NQ units of 512 B (32 lanes x 16 B).
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    constexpr int dummy = 0;
    (void)dummy;
    const int o = i % SK;
    r4d_t256_sync_lds();
    if (ks != o) {
      const int kw = ks > o ? ks - 1 : ks;
      unsigned char* dst = lds + ((rg * (SK - 1) + kw) * NQ) * 512 + lane * 16;
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
      const unsigned char* src = lds + ((rg * (SK - 1)) * NQ) * 512 + lane * 16;
#pragma unroll
      for (int p = 0; p < NP; ++p)
#pragma unroll
        for (int f = 0; f < 2; ++f) {
          const int q = (p * 2 + f) * 2;
          v8f s = (v8f)(0.f);
#pragma unroll
          for (int sl = 0; sl < SK; ++sl) {
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

  // ---- 2. hand the sums to ws in lane order, take a ticket ----------------------------------------
  float* wsy = ws + (size_t)blockIdx.y * M * N;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    if (ks == i % SK) {
      float* unit = wsy + ((size_t)(rg * 4 + i) * (N / Wc) + n0 / Wc) * (16 * Wc);
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
    if (ks != i % SK) continue;
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

template <int KB, int NP, int U, int RG, int SK>
static void r4d_t256_launch(const r4d_t256_args& a) {
  hipLaunchKernelGGL((r4d_gemm_trellis_nt_m256_kernel<KB, NP, U, RG, SK>), a.grid, a.block, a.smem,
                     a.stream, a.a0, a.a1, a.n_split, a.w, a.svh, a.c, a.ws, a.tickets, a.K, a.N,
                     a.SKG, a.out_scale);
}

template <int KB, int NP, int U, int RG>
static void r4d_t256_launch_sk(const r4d_t256_args& a, int SK) {
  if (SK == 2) r4d_t256_launch<KB, NP, U, RG, 2>(a);
  else if (SK == 4) r4d_t256_launch<KB, NP, U, RG, 4>(a);
  else r4d_t256_fail("SK must be 2 or 4");
}

template <int KB>
static void r4d_t256_launch_kb(const r4d_t256_args& a, int RG, int NP, int U, int SK) {
  if (RG == 4 && NP == 2 && U == 2) r4d_t256_launch_sk<KB, 2, 2, 4>(a, SK);
  else if (RG == 4 && NP == 1 && U == 4) r4d_t256_launch_sk<KB, 1, 4, 4>(a, SK);
  else if (RG == 2 && NP == 2 && U == 1) r4d_t256_launch_sk<KB, 2, 1, 2>(a, SK);
  else if (RG == 2 && NP == 1 && U == 2) r4d_t256_launch_sk<KB, 1, 2, 2>(a, SK);
  else r4d_t256_fail("(RG, NP, U) must be (4,2,2), (4,1,4), (2,2,1) or (2,1,2)");
}

void r4d_gemm_trellis_nt_m256(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t svh, int64_t c,
                              int64_t ws, int64_t tickets, int M, int K, int N, int KB, int SK, int NP,
                              int SKG, int U, float out_scale, int64_t stream) {
  if (M != 128 && M != 256) r4d_t256_fail("M must be 128 or 256");
  const int RG = M / 64;
  if (K <= 0 || N <= 0 || K % 128 || N % 128) r4d_t256_fail("K and N must be multiples of 128");
  if (KB != 4 && KB != 5) r4d_t256_fail("KB must be 4 or 5");
  if (SKG != 1 && SKG != 2 && SKG != 4 && SKG != 8) r4d_t256_fail("SKG must be 1, 2, 4 or 8");
  if (NP != 1 && NP != 2) r4d_t256_fail("NP must be 1 or 2");
  const int Wc = NP * 32;
  if (N % Wc) r4d_t256_fail("N must be a multiple of the block width 32 NP");
  if (n_split < 0 || n_split > N || n_split % 128) r4d_t256_fail("n_split must be in 0..N, a multiple of 128");
  if (NP * U != RG) r4d_t256_fail("NP * U must equal M / 64");
  if ((K / 16) % (SK * SKG * U)) r4d_t256_fail("(K/16) must be divisible by SK*SKG*U");
  if (!w || !c || !svh || !ws || !tickets || !a0) r4d_t256_fail("null buffer");
  r4d_t256_args a;
  a.grid = dim3(N / Wc, SKG, 1);
  a.block = dim3(RG * SK * WAVE32);
  const size_t stage = (size_t)2 * SK * NP * U * 1024;
  const size_t red = (size_t)RG * (SK - 1) * NP * 4 * 512;
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
  if (KB == 4) r4d_t256_launch_kb<4>(a, RG, NP, U, SK);
  else r4d_t256_launch_kb<5>(a, RG, NP, U, SK);
}

size_t r4d_gemm_trellis_nt_m256_ws_bytes(int M, int N, int SKG) {
  return (size_t)SKG * (size_t)M * (size_t)N * sizeof(float);
}
