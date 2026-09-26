// r4d_gemm_trellis_nt_m64.hip - the trellis-weight skinny GEMM for M up to 64, gfx1201, and the
// dense decode it shares its tile decode with:
//
//   r4d_gemm_trellis_nt_m64_raw   C[M,N] = A[M,K] @ Q[K,N], f16 A, fp32 C, fp32 accumulate
//   r4d_trellis_reconstruct_f16   Q[K,N] as f16, the whole matrix
//
// Q is the trellis-coded weight in its regularized domain: tiles of 16 k x 16 n, each a tail-biting
// ring of 256*KB bits whose 16-bit states hash to f16 values (EXL3 / QTIP, the "mul1" codebook; the
// format is docs/trellis.md in r4dx, the decode r4d_trellis_dq.h). The real linear is
// W^T = diag(suh) H Q H diag(svh) with 128-point Hadamards H on both sides; the input side is a
// separate kernel that hands this one an already transformed f16 A, and the output side (FWHT-128,
// svh, one bf16 rounding) is the production entry's epilogue. _raw stops before that epilogue and
// writes the fp32 sums, which is what a test can hold bit-exact (a one-hot A row returns a row of Q
// exactly) and what a bench can time without the transform. The design is docs/trellis-kernel.md
// 4.2-4.5 in r4dx.
//
// WHY A WMMA KERNEL AT M = 1.  The decode costs ~4 VALU per weight against w4a16's 1.25, and every
// cycle of it has to hide behind the weight stream for the format to pay: 4 bits per weight at KB = 4
// against w4a16's 4.5 with its group scales. A dot2 GEMV would be cheaper at M = 1 but would sum in a
// different order from the verify rows (M = 2..16) that must reproduce the decode row bit for bit, so
// the same WMMA kernel serves M = 1..64, and M <= 16 always runs the M = 1 band's tuning.
//
// DECOMPOSITION, as w4a16's.  A block is WV x SK waves: wave w owns column block w / SK and K slice
// w % SK. A wave owns NP tile PAIRS (32 NP columns: the lane map decodes two adjacent n-tiles at
// once), so a block covers Wc = 32 WV NP columns, one of 32 / 64 / 128 / 256. SKG blocks split K
// across the grid (grid.y) and grid.z covers the row tiles, MT per block. Every wave's K range is
// K / (SK SKG), walked U k-tiles per step.
//
// ADDRESSING.  Everything a lane loads is a loop-invariant 32-bit VGPR offset off a wave-UNIFORM
// base: the weight base of each pair and the activation base sit in SGPRs and step by a constant
// per iteration, and the U k-tiles of one step are immediate offsets. The design's first probe kept
// a 64-bit pointer per lane and paid 11 address VALU per (pair, k-tile) for it, 0.69 per weight on
// top of the decode's 3.875; this is what docs/trellis-kernel.md 4.2 asks for instead.
//
// ONE STEP AHEAD.  Each wave double-buffers its loads: the words and activation fragments of step
// i+1 are issued before step i is decoded, which is what lets a wave's own decode hide its next
// loads. The two stages are two named register sets, not one set copied at the loop edge, so no
// v_mov sits in the loop. At an even step count the last issue re-reads the final step (clamped,
// never past the tensor): one step of redundant, cache-resident loads per wave. That is cheaper
// than not issuing it: peeling the last one or two steps off the loop instead (an extra copy of
// the decode, more live registers) measured 0-1.2% slower on 9 of 10 M = 1 shapes and tunings,
// 1.6% faster on one. MT >= 3 (the prefill shapes) keeps one stage: 3-4 WMMA per decoded fragment
// hide the loads, and a second set of activation fragments would not fit the register budget; so
// do two MT = 2 shapes (r4d_tq_prefetch).
//
// THE WHOLE LOOP IS THE DECODE.  Compiled for gfx1201, every instantiation's K loop is 62 VALU per
// (tile pair, k-tile) -- 3.875 per weight, the decode and nothing else -- plus the WMMA, the loads
// and a few SALU (tools/reference/trellis_isa_report.py reads it from the build's listing). The
// decode is one asm block with a fixed schedule (r4d_trellis_k4_decode says why): inside it every
// dependency is 8 or more VALU apart, and it ends in an s_delay_alu, so no VALU reads an asm result
// 1-3 VALU later unmarked -- which the build checks (r4dx's third_party/check_trellis_isa.cmake).
//
// SPLIT 128-GROUPS.  The output transform works on 128-column groups, so a group whose sums come
// from more than one block -- SKG > 1, or blocks narrower than 128 columns -- is finished by the LAST
// block to arrive: every contributor writes its in-block sum to ws[blockIdx.y] and takes a ticket;
// the one that draws the last ticket re-zeroes it and adds ws[0..SKG-1] in that fixed order, its own
// slot included, so the result never depends on which block came last. Nobody waits for anybody,
// so there is no inter-block spin and nothing that can deadlock; the ticket resets itself, so the
// kernel is capture-safe. _raw writes that sum; the production entry will transform it first.
#include "r4d.h"
#include "r4d_gdn_wmma.h"
#include "r4d_trellis_dq.h"
#include <cstdint>
#include <stdexcept>
#include <string>

#define R4D_GEMM_TRELLIS_MAX_M 64
#define WAVE32 32

typedef __attribute__((address_space(1))) const unsigned char* r4d_gbyte;

// A load from a wave-uniform base at a per-lane 32-bit byte offset plus a compile-time immediate.
// Written as (base + off) + imm so the zero-extended offset and the constant stay separate operands
// and fold into global_load's voffset and immediate fields; (base + (off + imm)) would not, because
// an unsigned 32-bit add may wrap and LLVM cannot pull the constant out of the extension.
template <typename T, bool NT>
__device__ __forceinline__ T r4d_tq_load(r4d_gbyte base, unsigned off, int imm) {
  auto p = (__attribute__((address_space(1))) const T*)(base + off + imm);
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}

// One step's operands: for each of the wave's NP pairs and U k-tiles, the lane's b64 (A, B) and b32
// (P) ring words, and the activation fragment of each of the MT row tiles.
template <int NP, int U, int MT>
struct r4d_tq_stage {
  unsigned long long ab[NP][U];
  unsigned p[NP][U];
  v8h af[MT][U];
};

// Whether a wave double-buffers its loads (the K loop). At MT >= 3 each decoded fragment already
// feeds 3-4 WMMA per pair, which hides the loads by itself, and a second stage of 4 U MT activation
// VGPRs would spill the prefill shapes; so those take one stage. So do the two MT = 2 shapes whose
// accumulators (16 NP MT) and two stages (2 (3 NP U + 4 MT U)) come to more than 128 VGPRs,
// (NP, U) = (2, 4) and (4, 1): with the stages really both live (the K loop's sched_barriers) they
// compile to 192 VGPRs and scratch.
static constexpr bool r4d_tq_prefetch(int NP, int U, int MT) {
  return MT <= 2 && 16 * NP * MT + 2 * (3 * NP * U + 4 * MT * U) <= 128;
}

// Makes a wave-uniform pointer opaque to the optimizer (it stays in SGPRs, see the K loop).
__device__ __forceinline__ void r4d_tq_opaque(r4d_gbyte& p) {
  unsigned long long v = (unsigned long long)p;
  asm("" : "+s"(v));
  p = (r4d_gbyte)v;
}

// The same for a per-lane value (it stays where it is; the optimizer just cannot hoist uses of it).
__device__ __forceinline__ void r4d_tq_opaque_v(unsigned& v) { asm("" : "+v"(v)); }

__device__ __forceinline__ v8f r4d_tq_mma(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

__device__ __forceinline__ void r4d_tq_clock(unsigned long long* clk) {
  clk[0] = clock64();
  clk[1] = wall_clock64();
}

// KB = 4 only (the K = 5 fetch is in r4d_trellis_dq.h and the reconstruct kernel; its GEMM comes
// with the production epilogue). NT = non-temporal weight loads, as w4a16's.
template <int NP, int U, int MT, bool NT>
__global__ __launch_bounds__(1024) void r4d_gemm_trellis_nt_m64_raw_kernel(
    const unsigned short* __restrict__ A0, const unsigned short* __restrict__ A1, int n_split,
    const unsigned* __restrict__ W, float* __restrict__ C, float* __restrict__ ws,
    unsigned* __restrict__ tickets, int M, int K, int N, int WV, int SK, int SKG,
    unsigned long long* __restrict__ clk) {
  constexpr int BLK = 256;                                 // bytes per (pair, k-tile) block at KB = 4
  // SK x 8 rows x Wc columns, and nothing else: at SK = 16, Wc = 128 it is the whole 64 KiB, so the
  // split path's broadcast reuses its first word rather than a static __shared__ variable.
  extern __shared__ float red[];
  const int tid = threadIdx.x, lane = tid % WAVE32;
  const bool probe = clk != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 &&
                     tid == 0;
  if (probe) r4d_tq_clock(clk);
  // The wave index is uniform, but the compiler cannot prove it; saying so keeps every base below
  // in SGPRs.
  const int wave = __builtin_amdgcn_readfirstlane(tid / WAVE32);
  const int col_local = wave / SK, ksplit = wave - col_local * SK;
  const int Wc = WV * NP * 32;
  const int n0 = blockIdx.x * Wc;                          // the block's first column
  const int pair0 = n0 / 32 + col_local * NP;              // the wave's first tile pair
  const int kt_total = K / 16;
  const int ktw = kt_total / (SK * SKG);                   // k-tiles per wave
  const int kt0 = ((int)blockIdx.y * SK + ksplit) * ktw;
  const int mt0 = blockIdx.z * MT;
  const int steps = ktw / U;
  // A block never straddles the part boundary (the host checks n_split % Wc), so the part is uniform.
  const unsigned short* A = n0 >= n_split ? A1 : A0;

  v8f acc[NP][2][MT];
#pragma unroll
  for (int p = 0; p < NP; ++p)
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
      for (int i = 0; i < MT; ++i) acc[p][f][i] = (v8f)(0.f);

  {
    // Wave-uniform bases of the current step: one per pair, one for the activation.
    r4d_gbyte wb[NP];
#pragma unroll
    for (int p = 0; p < NP; ++p)
      wb[p] = (r4d_gbyte)(const unsigned char*)W + ((size_t)(pair0 + p) * kt_total + kt0) * BLK;
    r4d_gbyte abase = (r4d_gbyte)(const unsigned char*)A + (size_t)kt0 * 32;
    unsigned off_ab = r4d_trellis_k4_off_ab(lane), off_p = r4d_trellis_k4_off_p(lane);
    // Activation rows are clamped, never masked: a row tile past M re-reads the last valid row and
    // computes sums nobody stores.
    unsigned aoff[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) aoff[i] = rowGo((unsigned)K, (mt0 + i) * 16, M - 1, lane) * 2u;

    // `ahead` steps past the current bases (a scalar), U k-tiles as immediates. Each stage's bases
    // are made opaque too: otherwise the DAG shares one (base + lane offset) node between the two
    // stages, and a node with a scalar-offset user is materialized as a 64-bit VGPR pointer.
    auto load = [&](r4d_tq_stage<NP, U, MT>& s, int ahead) {
      const size_t wo = (size_t)ahead * (U * BLK), ao = (size_t)ahead * (U * 32);
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        r4d_gbyte b = wb[p] + wo;
        r4d_tq_opaque(b);
#pragma unroll
        for (int u = 0; u < U; ++u) {
          s.ab[p][u] = r4d_tq_load<unsigned long long, NT>(b, off_ab, u * BLK);
          s.p[p][u] = r4d_tq_load<unsigned, NT>(b, off_p, u * BLK);
        }
      }
      r4d_gbyte a = abase + ao;
      r4d_tq_opaque(a);
#pragma unroll
      for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int u = 0; u < U; ++u) {
          union { unsigned long long w[2]; v8h f; } x;
          x.w[0] = r4d_tq_load<unsigned long long, false>(a, aoff[i], u * 32);
          x.w[1] = r4d_tq_load<unsigned long long, false>(a, aoff[i], u * 32 + 16);
          s.af[i][u] = x.f;
        }
    };
    // Loop strength reduction would otherwise turn every (uniform base + lane offset) into a
    // per-lane 64-bit pointer, two VALU adds per load stream per step; hiding the bumped bases from
    // it keeps them scalar. The lane offsets are re-hidden every iteration for a second reason: the
    // saddr form needs the offset's zero-extension in the same basic block as the load, and LICM
    // would hoist it out of the loop. Neither costs an instruction.
    auto advance = [&](int steps_done) {
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        wb[p] += (size_t)steps_done * (U * BLK);
        r4d_tq_opaque(wb[p]);
      }
      abase += (size_t)steps_done * (U * 32);
      r4d_tq_opaque(abase);
      r4d_tq_opaque_v(off_ab);
      r4d_tq_opaque_v(off_p);
#pragma unroll
      for (int i = 0; i < MT; ++i) r4d_tq_opaque_v(aoff[i]);
    };
    auto compute = [&](const r4d_tq_stage<NP, U, MT>& s) {
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int p = 0; p < NP; ++p) {
          v8h f0, f1;
          r4d_trellis_k4_decode(s.p[p][u], (unsigned)s.ab[p][u], (unsigned)(s.ab[p][u] >> 32), f0,
                                f1);
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            acc[p][0][i] = r4d_tq_mma(s.af[i][u], f0, acc[p][0][i]);
            acc[p][1][i] = r4d_tq_mma(s.af[i][u], f1, acc[p][1][i]);
          }
        }
    };

    if constexpr (r4d_tq_prefetch(NP, U, MT)) {
      // The sched_barriers pin each stage's loads ahead of the other stage's decode. Without them
      // the scheduler sinks the loads toward their first use to shorten live ranges -- at (NP, U)
      // = (4, 1) all the way past the other decode, so the loop opened on s_wait_loadcnt 0 and took
      // 1.46x as long at M = 1 (0-5% longer on the other tunings measured).
      r4d_tq_stage<NP, U, MT> s0, s1;
      load(s0, 0);
      int st = 0;
      for (; st + 1 < steps; st += 2) {                    // bases at step st
        load(s1, 1);
        __builtin_amdgcn_sched_barrier(0);
        compute(s0);
        load(s0, st + 2 < steps ? 2 : 1);
        __builtin_amdgcn_sched_barrier(0);
        compute(s1);
        advance(2);
      }
      if (st < steps) compute(s0);
    } else {
      for (int st = 0; st < steps; ++st) {
        r4d_tq_stage<NP, U, MT> s0;
        load(s0, 0);
        compute(s0);
        advance(1);
      }
    }
  }

  // In-block reduction over the SK k-splits, 8 rows at a time, so the LDS is SK*Wc*32 B whatever MT
  // is: rows 0-7 of a row tile live on lanes 0-15, rows 8-15 on lanes 16-31 (the C layout). Summed
  // in split order, so no row depends on the schedule, on M, or on the other rows.
  const int nthreads = WV * SK * WAVE32;
  const bool split = SKG > 1 || Wc < 128;
  float* dst = split ? ws + (size_t)blockIdx.y * M * N : C;
  const int t = (lane >> 3) & 1, c = lane & 7, hh_lane = lane >> 4;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int mbase = (mt0 + i) * 16 + 8 * hh;
      if (mbase >= M) break;
      __syncthreads();
      if (hh_lane == hh) {
#pragma unroll
        for (int p = 0; p < NP; ++p)
#pragma unroll
          for (int f = 0; f < 2; ++f) {
            const int col = (col_local * NP + p) * 32 + 16 * t + 8 * f + c;   // true column
#pragma unroll
            for (int e = 0; e < 8; ++e) red[(ksplit * 8 + e) * Wc + col] = acc[p][f][i][e];
          }
      }
      __syncthreads();
      for (int idx = tid; idx < 8 * Wc; idx += nthreads) {
        const int r = idx / Wc, n = idx - r * Wc, m = mbase + r;
        if (m >= M) continue;
        float v = 0.f;
        for (int s = 0; s < SK; ++s) v += red[(s * 8 + r) * Wc + n];
        dst[(size_t)m * N + n0 + n] = v;
      }
    }
  }
  if (probe) r4d_tq_clock(clk + 2);
  if (!split) return;

  // The 128-group this block contributes to (for Wc = 256, both of its groups share one ticket: they
  // have the same contributors), its column span, and how many blocks contribute to it.
  const int span = Wc > 128 ? Wc : 128;
  const int g0 = n0 / span * span;
  const unsigned contributors = (unsigned)(SKG * (span / Wc));
  __threadfence();                                         // this thread's ws stores, device-wide
  __syncthreads();                                         // ... for every thread of the block (and
  if (tid == 0) {                                          // every read of red is done)
    const unsigned old = atomicAdd(&tickets[g0 / 128], 1u);
    const bool last = old == contributors - 1;
    if (last) tickets[g0 / 128] = 0u;                      // self-resetting: nobody else touches it
    red[0] = last ? 1.f : 0.f;                             // until the next launch
  }
  __syncthreads();
  if (red[0] == 0.f) return;
  __threadfence();                                         // acquire: see every contributor's ws
  for (int idx = tid; idx < M * span; idx += nthreads) {
    const int m = idx / span, n = g0 + (idx - m * span);
    float v = 0.f;
    for (int y = 0; y < SKG; ++y) v += ws[((size_t)y * M + m) * N + n];
    C[(size_t)m * N + n] = v;
  }
}

// One wave per (tile pair, k-tile) block, eight blocks per workgroup: the lane decodes its two
// fragments exactly as the GEMM does and scatters them to their true (k, n).
template <int KB>
__global__ __launch_bounds__(256) void r4d_trellis_reconstruct_f16_kernel(
    const unsigned* __restrict__ W, unsigned short* __restrict__ Q, int K, int N) {
  const int lane = threadIdx.x % WAVE32;
  const int kt_total = K / 16;
  const long long blk = (long long)blockIdx.x * 8 + threadIdx.x / WAVE32;
  if (blk >= (long long)(N / 32) * kt_total) return;
  const int pair = (int)(blk / kt_total), kt = (int)(blk - (long long)pair * kt_total);
  const unsigned* b = W + blk * (16 * KB);                // 2 tiles x 8 KB words
  v8h f0, f1;
  if constexpr (KB == 4) {
    const unsigned ia = r4d_trellis_k4_off_ab(lane) / 4, ip = r4d_trellis_k4_off_p(lane) / 4;
    r4d_trellis_k4_decode(b[ip], b[ia], b[ia + 1], f0, f1);
  } else {
    unsigned w5[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) w5[i] = b[r4d_trellis_k5_word(lane, i)];
    r4d_trellis_k5_decode(w5, lane, f0, f1);
  }
  const int t = (lane >> 3) & 1, c = lane & 7, h = lane >> 4;
  const int n = pair * 32 + 16 * t + c;
  // Whole-vector casts: clang lowers __builtin_bit_cast(unsigned short, f0[e]) to a read of element
  // 0 for every e (it takes the vector element's "address" as the vector's).
  typedef unsigned short u16x8 __attribute__((ext_vector_type(8)));
  const u16x8 q0 = __builtin_bit_cast(u16x8, f0), q1 = __builtin_bit_cast(u16x8, f1);
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    const size_t k = (size_t)kt * 16 + 8 * (e >> 2) + 4 * h + (e & 3);
    Q[k * N + n] = q0[e];
    Q[k * N + n + 8] = q1[e];
  }
}

// ---- host ------------------------------------------------------------------------------------

// The instantiated (NP, U, MT): NP * U <= 8 (docs/trellis-kernel.md 4.4), and per (NP, U) an MT
// cap that fits 190 VGPRs with no scratch -- 16 NP MT of them are accumulators, so the wide-pair,
// long-unroll combinations run out first. Set when the decode was one asm per hash (the next MT up
// spilled or reached 191-192 in every row); with the one-block decode the instantiated kernels take
// 54-188, the next MT up still spills at (NP, U) = (4, 1), and at (2, 2), (2, 4) and (4, 2) it would
// fit (173-174) but is not instantiated. The build checks the real count of every instantiation
// against 190 (8 waves per SIMD) and zero scratch (r4dx's third_party/check_trellis_isa.cmake), so
// a compiler that allocates differently fails there.
static constexpr int r4d_tq_max_mt(int NP, int U) {
  return NP == 1 ? (U == 1 || U == 2 || U == 4 ? 4 : 0)
       : NP == 2 ? (U == 1 ? 4 : U == 2 || U == 4 ? 3 : 0)
       : NP == 4 ? (U == 1 ? 2 : U == 2 ? 1 : 0)
       : 0;
}
static constexpr bool r4d_trellis_fits(int NP, int U, int MT) {
  return MT >= 1 && MT <= r4d_tq_max_mt(NP, U);
}
static bool r4d_trellis_instantiated(int NP, int U, int MT) { return r4d_trellis_fits(NP, U, MT); }

static void r4d_trellis_fail(const std::string& what) {
  throw std::runtime_error("r4d gemm_trellis_nt_m64: " + what);
}

// The legality rules of docs/trellis-kernel.md 4.3, in one place.
static void r4d_trellis_check(int M, int K, int N, int n_split, int KB, int WV, int SK, int MT,
                              int NP, int SKG, int U, int NT, bool have_ws) {
  if (M < 1 || M > R4D_GEMM_TRELLIS_MAX_M) r4d_trellis_fail("M must be 1..64");
  if (K <= 0 || N <= 0 || K % 128 || N % 128) r4d_trellis_fail("K and N must be multiples of 128");
  if (KB != 4) r4d_trellis_fail("KB " + std::to_string(KB) + " is not instantiated (4)");
  if (WV != 1 && WV != 2 && WV != 4) r4d_trellis_fail("WV must be 1, 2 or 4");
  if (SK != 1 && SK != 2 && SK != 4 && SK != 8 && SK != 16)
    r4d_trellis_fail("SK must be 1, 2, 4, 8 or 16");
  if (SKG != 1 && SKG != 2 && SKG != 4 && SKG != 8) r4d_trellis_fail("SKG must be 1, 2, 4 or 8");
  if (NT != 0 && NT != 1) r4d_trellis_fail("NT must be 0 or 1");
  if (!r4d_trellis_instantiated(NP, U, MT))
    r4d_trellis_fail("(NP, U, MT) = (" + std::to_string(NP) + ", " + std::to_string(U) + ", " +
                     std::to_string(MT) + ") is not instantiated (NP, U in {1,2,4}, NP*U <= 8; "
                     "MT up to 4 at NP 1 and at (2,1), 3 at (2,2) and (2,4), 2 at (4,1), 1 at "
                     "(4,2))");
  const int Wc = WV * NP * 32;
  if (Wc > 256) r4d_trellis_fail("WV*NP*32 must be <= 256 columns");
  if (N % Wc) r4d_trellis_fail("N must be a multiple of the block width WV*NP*32");
  if (n_split < 0 || n_split > N || n_split % 128 || n_split % Wc)
    r4d_trellis_fail("n_split must be in 0..N and a multiple of 128 and of WV*NP*32");
  if ((K / 16) % (SK * SKG * U)) r4d_trellis_fail("(K/16) must be divisible by SK*SKG*U");
  if (WV * SK * WAVE32 > 1024) r4d_trellis_fail("threads/block > 1024");
  if ((size_t)SK * Wc * 8 * sizeof(float) > 64 * 1024)
    r4d_trellis_fail("SK*Wc*32 B of LDS > 64 KiB");
  const int mtiles = (M + 15) / 16;
  const bool split = SKG > 1 || Wc < 128;
  if (split && mtiles > MT)
    r4d_trellis_fail("a split 128-group (SKG > 1 or WV*NP*32 < 128) needs every row tile in one "
                     "block (ceil(M/16) <= MT)");
  if (split && !have_ws) r4d_trellis_fail("a split 128-group needs ws and tickets");
}

struct r4d_tq_launch_args {
  dim3 grid, block;
  size_t smem;
  hipStream_t stream;
  const unsigned short *a0, *a1;
  int n_split;
  const unsigned* w;
  float *c, *ws;
  unsigned* tickets;
  int M, K, N, WV, SK, SKG, NT;
  unsigned long long* clk;
};

// One launch at a compile-time (NP, U, MT); combinations r4d_trellis_fits rejects are never
// instantiated (the host check has already thrown for them).
template <int NP, int U, int MT>
static void r4d_tq_launch(const r4d_tq_launch_args& a) {
  if constexpr (r4d_trellis_fits(NP, U, MT)) {
#define LAUNCH(NTv)                                                                                \
  hipLaunchKernelGGL((r4d_gemm_trellis_nt_m64_raw_kernel<NP, U, MT, NTv>), a.grid, a.block, a.smem, \
                     a.stream, a.a0, a.a1, a.n_split, a.w, a.c, a.ws, a.tickets, a.M, a.K, a.N,    \
                     a.WV, a.SK, a.SKG, a.clk)
    if (a.NT) { LAUNCH(true); } else { LAUNCH(false); }
#undef LAUNCH
  } else {
    r4d_trellis_fail("(NP, U, MT) not instantiated");
  }
}

template <int NP, int U>
static void r4d_tq_launch_mt(const r4d_tq_launch_args& a, int MT) {
  switch (MT) {
    case 1: r4d_tq_launch<NP, U, 1>(a); return;
    case 2: r4d_tq_launch<NP, U, 2>(a); return;
    case 3: r4d_tq_launch<NP, U, 3>(a); return;
    case 4: r4d_tq_launch<NP, U, 4>(a); return;
    default: r4d_trellis_fail("MT must be 1..4");
  }
}

void r4d_gemm_trellis_nt_m64_raw(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t c,
                                 int64_t ws, int64_t tickets, int M, int K, int N, int KB, int WV,
                                 int SK, int MT, int NP, int SKG, int U, int NT, int64_t clk,
                                 int64_t stream) {
  r4d_trellis_check(M, K, N, n_split, KB, WV, SK, MT, NP, SKG, U, NT, ws != 0 && tickets != 0);
  const int Wc = WV * NP * 32, mtiles = (M + 15) / 16;
  r4d_tq_launch_args a;
  a.grid = dim3(N / Wc, SKG, (mtiles + MT - 1) / MT);
  a.block = dim3(WV * SK * WAVE32);
  a.smem = (size_t)SK * Wc * 8 * sizeof(float);
  a.stream = (hipStream_t)stream;
  a.a0 = (const unsigned short*)a0;
  a.a1 = (const unsigned short*)(a1 ? a1 : a0);
  a.n_split = n_split;
  a.w = (const unsigned*)w;
  a.c = (float*)c;
  a.ws = (float*)ws;
  a.tickets = (unsigned*)tickets;
  a.M = M; a.K = K; a.N = N; a.WV = WV; a.SK = SK; a.SKG = SKG; a.NT = NT;
  a.clk = (unsigned long long*)clk;
  switch (NP * 16 + U) {
    case 1 * 16 + 1: r4d_tq_launch_mt<1, 1>(a, MT); return;
    case 1 * 16 + 2: r4d_tq_launch_mt<1, 2>(a, MT); return;
    case 1 * 16 + 4: r4d_tq_launch_mt<1, 4>(a, MT); return;
    case 2 * 16 + 1: r4d_tq_launch_mt<2, 1>(a, MT); return;
    case 2 * 16 + 2: r4d_tq_launch_mt<2, 2>(a, MT); return;
    case 2 * 16 + 4: r4d_tq_launch_mt<2, 4>(a, MT); return;
    case 4 * 16 + 1: r4d_tq_launch_mt<4, 1>(a, MT); return;
    case 4 * 16 + 2: r4d_tq_launch_mt<4, 2>(a, MT); return;
    default: r4d_trellis_fail("(NP, U) not instantiated");
  }
}

int r4d_gemm_trellis_nt_m64_has_rate(int KB) { return KB == 4; }
int r4d_gemm_trellis_nt_m64_max_m(void) { return R4D_GEMM_TRELLIS_MAX_M; }
size_t r4d_gemm_trellis_nt_m64_ws_bytes(int M, int N, int SKG) {
  return (size_t)SKG * (size_t)M * (size_t)N * sizeof(float);
}

void r4d_trellis_reconstruct_f16(int64_t w, int64_t q, int K, int N, int KB, int64_t stream) {
  if (K <= 0 || N <= 0 || K % 16 || N % 32)
    throw std::runtime_error("r4d trellis_reconstruct_f16: K must be a multiple of 16 and N of 32 "
                             "(whole tile pairs)");
  if (KB != 4 && KB != 5)
    throw std::runtime_error("r4d trellis_reconstruct_f16: KB must be 4 or 5");
  const long long blocks = (long long)(N / 32) * (K / 16);
  dim3 grid((unsigned)((blocks + 7) / 8)), block(8 * WAVE32);
  auto Wp = (const unsigned*)w;
  auto Qp = (unsigned short*)q;
  hipStream_t st = (hipStream_t)stream;
  if (KB == 4)
    hipLaunchKernelGGL(r4d_trellis_reconstruct_f16_kernel<4>, grid, block, 0, st, Wp, Qp, K, N);
  else
    hipLaunchKernelGGL(r4d_trellis_reconstruct_f16_kernel<5>, grid, block, 0, st, Wp, Qp, K, N);
}
