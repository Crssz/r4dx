// r4d_gemm_trellis_nt_m64.hip - the trellis-weight skinny GEMM for M up to 64, gfx1201, and the
// dense decode it shares its tile decode with:
//
//   r4d_gemm_trellis_nt_m64       C[M,N] = bf16( (FWHT128(A[M,K] @ Q[K,N]) * svh) * out_scale ), the
//                                 whole trellis linear given an input-transformed f16 A
//   r4d_gemm_trellis_nt_m64_raw   C[M,N] = A[M,K] @ Q[K,N], f16 A, fp32 C, fp32 accumulate
//   r4d_trellis_reconstruct_f16   Q[K,N] as f16, the whole matrix
//
// The kernel is also the source of r4d_gemm_trellis_nt_m64_wq.hip (R4DX_FAKEQ_W, a measurement hook: decoded
// weights rounded to int8 before the WMMA, the `WQ` template parameter below), which includes this file with
// R4D_TQ_WQ_UNIT defined: that unit has the `_wq` entry points and r4d_trellis_wscale_f32, this one the shipped
// ones. Nothing under `if constexpr (WQ)` is part of a shipped instantiation.
//
// Q is the trellis-coded weight in its regularized domain: tiles of 16 k x 16 n, each a tail-biting
// ring of 256*KB bits whose 16-bit states hash to f16 values (EXL3 / QTIP, the "mul1" codebook; the
// format is docs/trellis.md in r4dx, the decode r4d_trellis_dq.h). The real linear is
// W^T = diag(suh) H Q H diag(svh) with 128-point Hadamards H on both sides; the input side is a
// separate kernel (r4dx's r4dx_trellis_input_bf16) that hands this one an already transformed f16 A,
// and the output side -- FWHT-128 in fp32, svh, one bf16 rounding -- is this kernel's epilogue. _raw
// stops before that epilogue and writes the fp32 sums, which is what a test can hold bit-exact (a
// one-hot A row returns a row of Q exactly) and what a bench can time without the transform. The
// design is docs/trellis-kernel.md 4.2-4.5 in r4dx.
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
// K / (SK SKG), walked U k-tiles per step. KB = 4 and 5 are two instantiations of the same loop
// behind different fetches and decodes (3 dwords and 62 VALU per lane per (pair, k-tile) at KB = 4,
// 5 dwords and 75 VALU at KB = 5).
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
// do the MT <= 2 shapes whose two stages do not fit either (r4d_tq_prefetch).
//
// THE WHOLE LOOP IS THE DECODE.  Compiled for gfx1201, every instantiation's K loop is the decode
// and nothing else -- 62 VALU per (tile pair, k-tile) at KB = 4 (3.875 per weight), 75 at KB = 5 --
// plus the WMMA, the loads and a few SALU (tools/reference/trellis_isa_report.py reads it from the
// build's listing). Each decode is one asm block with a fixed schedule (r4d_trellis_k4_decode says
// why): no VALU reads an asm result 1-3 VALU later unmarked -- which the build checks (r4dx's
// third_party/check_trellis_isa.cmake).
//
// THE EPILOGUE (docs/trellis-kernel.md 4.5).  Per row tile, in passes of 8 rows (rows 0-7 of a tile
// live on lanes 0-15, rows 8-15 on lanes 16-31; a pass whose rows are all >= M is skipped, so
// M <= 8 costs one): every wave stores its accumulators for those rows at their TRUE columns into
// LDS red[SK][8][Wc], and the block sums the SK slices in slice order. The LDS is SK * Wc * 32 B
// whatever MT is. Then:
//   - a block that owns whole 128-column groups (SKG = 1 and Wc >= 128) keeps the sums in LDS and
//     finishes them itself: one wave per (row, 128-group), round-robin over the block's waves, runs
//     the fp32 FWHT-128 (r4d_fwht128.h) and writes bf16( (y * svh[n]) * out_scale ), the only
//     rounding after the fp32 accumulator;
//   - a SPLIT group -- SKG > 1, or blocks narrower than 128 columns -- is finished by the LAST block
//     to arrive: every contributor writes its in-block sums to ws[blockIdx.y] and takes a ticket;
//     the one that draws the last ticket re-zeroes it and adds ws[0..SKG-1] in that fixed order, its
//     own slot included, straight into the finishing waves' registers, then the same FWHT, svh and
//     rounding. The result never depends on which block came last. Nobody waits for anybody, so
//     there is no inter-block spin and nothing that can deadlock; the ticket resets itself, so the
//     kernel is capture-safe.
// No summation order depends on M or on the other rows, only on (SK, SKG, Wc), so one tuning for
// every M <= 16 makes a verify row equal to the decode row. _raw runs the same reduction and writes
// the fp32 sums instead of transforming them.
// Two things keep the epilogue short: the reduction's barriers order LDS only (r4d_tq_sync_lds;
// __syncthreads() waits for every outstanding global access and invalidates L0 at each one in WGP
// mode), and each wave loads the svh of the one group it finishes before the reduction starts
// rather than after its FWHT. Measured at M = 1 over one decode step's 336 GEMMs (r4dx's
// tool_trellis_gemm_bench --modes full, KB = 4), the output side went from 0.31-0.33 to 0.23 ms per
// token.
#include "r4d.h"
#include "r4d_fwht128.h"
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

// One step's operands: for each of the wave's NP pairs and U k-tiles, the lane's ring words -- at
// KB = 4 one b64 (A, B) and one b32 (P), at KB = 5 five b32 -- and the activation fragment of each
// of the MT row tiles.
template <int KB, int NP, int U, int MT>
struct r4d_tq_stage;
template <int NP, int U, int MT>
struct r4d_tq_stage<4, NP, U, MT> {
  unsigned long long ab[NP][U];
  unsigned p[NP][U];
  v8h af[MT][U];
};
template <int NP, int U, int MT>
struct r4d_tq_stage<5, NP, U, MT> {
  unsigned w[NP][U][5];
  v8h af[MT][U];
};

// Whether a wave double-buffers its loads (the K loop). At MT >= 3 each decoded fragment already
// feeds 3-4 WMMA per pair, which hides the loads by itself, and a second stage of 4 U MT activation
// VGPRs would spill the prefill shapes; so those take one stage. So do the MT <= 2 shapes whose
// accumulators (16 NP MT) and two stages (2 (W NP U + 4 MT U), W = 3 or 5 words per pair) come to
// more than 128 VGPRs -- at KB = 4 (NP, U) = (2, 4) and (4, 1) at MT = 2: with the stages really both
// live (the K loop's sched_barriers) they compiled to 192 VGPRs and scratch.
static constexpr bool r4d_tq_prefetch(int KB, int NP, int U, int MT) {
  return MT <= 2 && 16 * NP * MT + 2 * ((KB == 4 ? 3 : 5) * NP * U + 4 * MT * U) <= 128;
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

// fp32 -> bf16 bits, round to nearest even (a NaN stays a quiet NaN).
__device__ __forceinline__ unsigned short r4d_tq_bf16_rn(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (unsigned short)((u >> 16) | 0x40u);
  return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// A workgroup barrier that orders LDS only. __syncthreads() also orders global memory at workgroup
// scope, which in WGP mode (the block's waves on both CUs of a WGP, each CU with its own L0) means
// waiting for every outstanding global load and store before the barrier and invalidating L0 after
// it -- at every reduction barrier. The reduction exchanges nothing but LDS, and the one exchange
// through global memory (the split groups' ws and tickets) keeps its own device-scope fences and
// __syncthreads().
__device__ __forceinline__ void r4d_tq_sync_lds() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The svh of one 128-column group as its finishing wave holds it: sv[r] = svh[n0 + lane + 32 r].
__device__ __forceinline__ void r4d_tq_load_svh(float (&sv)[4], const float* __restrict__ svh, int n0,
                                                int lane) {
#pragma unroll
  for (int r = 0; r < 4; ++r) sv[r] = svh[n0 + lane + 32 * r];
}

// One wave finishes one (row, 128-column group): v[r] = the fp32 sum of column n0 + lane + 32 r,
// sv its svh (r4d_tq_load_svh); C[m][n] = bf16( (FWHT128(v)[n] * svh[n]) * out_scale ).
__device__ __forceinline__ void r4d_tq_finish(float (&v)[4], int lane, const float (&sv)[4],
                                              unsigned short* __restrict__ crow, int n0,
                                              float out_scale) {
  r4d_fwht128_wave(v, lane);
#pragma unroll
  for (int r = 0; r < 4; ++r) crow[n0 + lane + 32 * r] = r4d_tq_bf16_rn((v[r] * sv[r]) * out_scale);
}

// full = 0: _raw (fp32 C, no output transform); 1: the linear (bf16 C). A runtime flag, not a
// template parameter: it only selects the epilogue, after the K loop, so both entries share every
// instantiation (and its register count and schedule). NT = non-temporal weight loads, as w4a16's.
//
// WQ (R4DX_FAKEQ_W, docs/int8-prefill.md; a measurement hook): true rounds every decoded fragment to
// symmetric int8 with the scale of its (column, k group) from `wsc` (fp32 [K / 16 >> wq_gsh][N], built by
// r4d_trellis_wscale_f32; a group is 2^wq_gsh k-tiles) right before the WMMA. The two extra arguments exist
// only at WQ = true (a trailing parameter pack that is empty otherwise), so the shipped instantiations
// (WQ = false, the default) keep their exact signature and, with every `if constexpr (WQ)` gone, their exact
// code: the build's listing of them is identical to the one without this parameter (see docs/int8-prefill.md).
// A WQ = true instantiation always loads weights temporally (NT = false: a cache hint, the same bits) and
// is exempt from the 190-VGPR / zero-scratch rule (third_party/check_trellis_isa.cmake reports it only).
inline __device__ void r4d_tq_wq_unpack(const float*& wsc, int& gsh, const float* p, int g) {
  wsc = p;
  gsh = g;
}
template <int KB, int NP, int U, int MT, bool NT, bool WQ = false, typename... WqArgs>
__global__ __launch_bounds__(1024) void r4d_gemm_trellis_nt_m64_kernel(
    const unsigned short* __restrict__ A0, const unsigned short* __restrict__ A1, int n_split,
    const unsigned* __restrict__ W, const float* __restrict__ svh, void* __restrict__ Cv,
    float* __restrict__ ws, unsigned* __restrict__ tickets, int M, int K, int N, int WV, int SK,
    int SKG, int full, float out_scale, unsigned long long* __restrict__ clk, WqArgs... wq) {
  const float* wsc = nullptr;
  int wq_gsh = 0;
  if constexpr (WQ) r4d_tq_wq_unpack(wsc, wq_gsh, wq...);
  constexpr int BLK = 64 * KB;                             // bytes per (pair, k-tile) block
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
    // The lane's word offsets inside a block: (A, B) and P at KB = 4, five words at KB = 5.
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
    // Activation rows are clamped, never masked: a row tile past M re-reads the last valid row and
    // computes sums nobody stores.
    unsigned aoff[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) aoff[i] = rowGo((unsigned)K, (mt0 + i) * 16, M - 1, lane) * 2u;

    // `ahead` steps past the current bases (a scalar), U k-tiles as immediates. Each stage's bases
    // are made opaque too: otherwise the DAG shares one (base + lane offset) node between the two
    // stages, and a node with a scalar-offset user is materialized as a 64-bit VGPR pointer.
    auto load = [&](r4d_tq_stage<KB, NP, U, MT>& s, int ahead) {
      const size_t wo = (size_t)ahead * (U * BLK), ao = (size_t)ahead * (U * 32);
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        r4d_gbyte b = wb[p] + wo;
        r4d_tq_opaque(b);
#pragma unroll
        for (int u = 0; u < U; ++u) {
          if constexpr (KB == 4) {
            s.ab[p][u] = r4d_tq_load<unsigned long long, NT>(b, woff[0], u * BLK);
            s.p[p][u] = r4d_tq_load<unsigned, NT>(b, woff[1], u * BLK);
          } else {
#pragma unroll
            for (int i = 0; i < 5; ++i) s.w[p][u][i] = r4d_tq_load<unsigned, NT>(b, woff[i], u * BLK);
          }
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
#pragma unroll
      for (int i = 0; i < NOFF; ++i) r4d_tq_opaque_v(woff[i]);
#pragma unroll
      for (int i = 0; i < MT; ++i) r4d_tq_opaque_v(aoff[i]);
    };
    // The KB = 5 decode clobbers its words (they are reloaded before the stage is read again).
    // WQ: wq_kt is the first k-tile of the step compute() is on. The stages are computed strictly in step
    // order (s0, s1, s0, ... or the one stage of the plain loop), so compute() keeps the count itself and
    // the shipped instantiations (WQ = false) carry no extra argument, array or variable: their listing
    // must stay what it was.
    [[maybe_unused]] int wq_kt = kt0;
    auto compute = [&](r4d_tq_stage<KB, NP, U, MT>& s) {
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int p = 0; p < NP; ++p) {
          v8h f0, f1;
          if constexpr (KB == 4)
            r4d_trellis_k4_decode(s.p[p][u], (unsigned)s.ab[p][u], (unsigned)(s.ab[p][u] >> 32), f0,
                                  f1);
          else
            r4d_trellis_k5_decode(s.w[p][u], k5sh, f0, f1);
          if constexpr (WQ) {
            // A fragment's column is fixed per lane (f0 = 32 pair + 16 t + c, f1 = f0 + 8); its scale
            // group is a function of the k-tile, so the U k-tiles of a step may meet two groups when a
            // group is narrower than the step (col32 at U = 4).
            const float* row = wsc + (size_t)((wq_kt + u) >> wq_gsh) * N;
            const int n = (pair0 + p) * 32 + 16 * ((lane >> 3) & 1) + (lane & 7);
            f0 = r4d_trellis_wq_round(f0, row[n]);
            f1 = r4d_trellis_wq_round(f1, row[n + 8]);
          }
#pragma unroll
          for (int i = 0; i < MT; ++i) {
            acc[p][0][i] = r4d_tq_mma(s.af[i][u], f0, acc[p][0][i]);
            acc[p][1][i] = r4d_tq_mma(s.af[i][u], f1, acc[p][1][i]);
          }
        }
      if constexpr (WQ) wq_kt += U;
    };
    if constexpr (r4d_tq_prefetch(KB, NP, U, MT)) {
      // The sched_barriers pin each stage's loads ahead of the other stage's decode. Without them
      // the scheduler sinks the loads toward their first use to shorten live ranges -- at (NP, U)
      // = (4, 1) all the way past the other decode, so the loop opened on s_wait_loadcnt 0 and took
      // 1.46x as long at M = 1 (0-5% longer on the other tunings measured).
      r4d_tq_stage<KB, NP, U, MT> s0, s1;
      load(s0, 0);
      int st = 0;
      for (; st + 1 < steps; st += 2) {                    // bases at step st
        load(s1, 1);
        __builtin_amdgcn_sched_barrier(0);
        compute(s0);
        // readfirstlane: the select is uniform, but at KB = 5 (five b32 streams) the compiler
        // put it in a VGPR (v_cndmask + a 64-bit VALU pointer add per pair and step), which the
        // opaque base below then cannot take ("illegal VGPR to SGPR copy"). Said explicitly, it
        // is an s_cselect and no VALU; at KB = 4 it changes nothing.
        load(s0, __builtin_amdgcn_readfirstlane(st + 2 < steps ? 2 : 1));
        __builtin_amdgcn_sched_barrier(0);
        compute(s1);
        advance(2);
      }
      if (st < steps) compute(s0);
    } else {
      for (int st = 0; st < steps; ++st) {
        r4d_tq_stage<KB, NP, U, MT> s0;
        load(s0, 0);
        compute(s0);
        advance(1);
      }
    }
  }

  // In-block reduction over the SK k-splits, 8 rows at a time (see THE EPILOGUE above).
  const int nwaves = WV * SK, nthreads = nwaves * WAVE32;
  const bool split = SKG > 1 || Wc < 128;
  // The 128-column groups this block finishes or contributes to: its own columns at Wc >= 128 (for
  // Wc = 256 both groups share one ticket: they have the same contributors); a narrower block's
  // columns lie in one group with the blocks beside it.
  const int span = Wc > 128 ? Wc : 128;
  const int g0 = n0 / span * span;
  const int groups = span / 128;
  // The (row, group) finishing tasks go round-robin over the waves, task = wave + i nwaves, and a
  // task's group is task % groups. That is wave % groups for every task of a wave: groups = 2 only
  // at Wc = 256 = 32 WV NP with NP <= 4, so WV is even and so is nwaves = WV SK. Each wave therefore
  // finishes one group, gw, over rows wave / groups + i nwaves / groups, and its svh is loaded now,
  // so the latency hides behind the reduction instead of trailing the FWHT.
  const int gw = wave % groups, rstep = nwaves / groups;
  float sv[4] = {0.f, 0.f, 0.f, 0.f};
  if (full) r4d_tq_load_svh(sv, svh, g0 + gw * 128, lane);
  float* const Cf = (float*)Cv;
  unsigned short* const Cb = (unsigned short*)Cv;
  float* dst = split ? ws + (size_t)blockIdx.y * M * N : Cf;
  const int t = (lane >> 3) & 1, c = lane & 7, hh_lane = lane >> 4;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
#pragma unroll
    for (int hh = 0; hh < 2; ++hh) {
      const int mbase = (mt0 + i) * 16 + 8 * hh;
      if (mbase >= M) break;
      r4d_tq_sync_lds();
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
      r4d_tq_sync_lds();
      // A block that finishes its own groups keeps the sums in LDS, in slice 0's slot: each
      // (row, column) is read and rewritten by one thread only.
      const bool keep = full && !split;
      for (int idx = tid; idx < 8 * Wc; idx += nthreads) {
        const int r = idx / Wc, n = idx - r * Wc, m = mbase + r;
        if (m >= M) continue;
        float v = 0.f;
        for (int s = 0; s < SK; ++s) v += red[(s * 8 + r) * Wc + n];
        if (keep) red[r * Wc + n] = v;
        else dst[(size_t)m * N + n0 + n] = v;
      }
      if (keep) {
        r4d_tq_sync_lds();
        const int rows = M - mbase < 8 ? M - mbase : 8;   // groups = Wc / 128, g0 = n0 here
        for (int r = wave / groups; r < rows; r += rstep) {
          float v[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) v[q] = red[r * Wc + gw * 128 + lane + 32 * q];
          r4d_tq_finish(v, lane, sv, Cb + (size_t)(mbase + r) * N, n0 + gw * 128, out_scale);
        }
      }
    }
  }
  if (probe) r4d_tq_clock(clk + 2);
  if (!split) return;

  // How many blocks contribute to this block's group(s).
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
  if (!full) {
    for (int idx = tid; idx < M * span; idx += nthreads) {
      const int m = idx / span, n = g0 + (idx - m * span);
      float v = 0.f;
      for (int y = 0; y < SKG; ++y) v += ws[((size_t)y * M + m) * N + n];
      Cf[(size_t)m * N + n] = v;
    }
    return;
  }
  // One wave per (row, 128-group) sums the slots in y order into its registers and finishes it.
  const int gn = g0 + gw * 128;
  for (int m = wave / groups; m < M; m += rstep) {
    float v[4];
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const int n = gn + lane + 32 * q;
      float x = 0.f;
      for (int y = 0; y < SKG; ++y) x += ws[((size_t)y * M + m) * N + n];
      v[q] = x;
    }
    r4d_tq_finish(v, lane, sv, Cb + (size_t)m * N, gn, out_scale);
  }
}

// The per-(column, k group) scale table R4DX_FAKEQ_W's `_wq` kernels read: sc[g * N + n] = max |Q[k][n]|
// / 127 over the 2^gsh k-tiles of group g (fp32; 1.0f for an all-zero group). One wave per (tile pair,
// group), eight per workgroup; each decodes its blocks with the GEMM's own decode, so the values are the
// ones the GEMM's WMMA would see. A lane holds one column per fragment and half of each k-tile's 16 k
// (lane >> 4), so the two halves meet through a lane xor 16.
template <int KB>
__global__ __launch_bounds__(256) void r4d_trellis_wscale_kernel(
    const unsigned* __restrict__ W, float* __restrict__ sc, int K, int N, int gsh) {
  const int lane = threadIdx.x % WAVE32;
  const int kt_total = K / 16, groups = kt_total >> gsh;
  const long long id = (long long)blockIdx.x * 8 + threadIdx.x / WAVE32;
  if (id >= (long long)(N / 32) * groups) return;
  const int pair = (int)(id / groups), g = (int)(id - (long long)pair * groups);
  float m0 = 0.f, m1 = 0.f;
  for (int j = 0; j < (1 << gsh); ++j) {
    const long long blk = (long long)pair * kt_total + (g << gsh) + j;
    const unsigned* b = W + blk * (16 * KB);
    v8h f0, f1;
    if constexpr (KB == 4) {
      const unsigned ia = r4d_trellis_k4_off_ab(lane) / 4, ip = r4d_trellis_k4_off_p(lane) / 4;
      r4d_trellis_k4_decode(b[ip], b[ia], b[ia + 1], f0, f1);
    } else {
      unsigned w5[5];
#pragma unroll
      for (int i = 0; i < 5; ++i) w5[i] = b[r4d_trellis_k5_word(lane, i)];
      r4d_trellis_k5_decode(w5, r4d_trellis_k5_shift(lane), f0, f1);
    }
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      m0 = __builtin_fmaxf(m0, __builtin_fabsf((float)f0[e]));
      m1 = __builtin_fmaxf(m1, __builtin_fabsf((float)f1[e]));
    }
  }
  m0 = __builtin_fmaxf(m0, __shfl_xor(m0, 16, WAVE32));
  m1 = __builtin_fmaxf(m1, __shfl_xor(m1, 16, WAVE32));
  if (lane < 16) {
    const int n = pair * 32 + 16 * ((lane >> 3) & 1) + (lane & 7);
    sc[(size_t)g * N + n] = m0 > 0.f ? m0 / 127.0f : 1.0f;
    sc[(size_t)g * N + n + 8] = m1 > 0.f ? m1 / 127.0f : 1.0f;
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
    r4d_trellis_k5_decode(w5, r4d_trellis_k5_shift(lane), f0, f1);
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

// The instantiated (KB, NP, U, MT): NP * U <= 8 (docs/trellis-kernel.md 4.4), and per (NP, U) an MT
// cap that fits 190 VGPRs with no scratch -- 16 NP MT of them are accumulators and 3 (KB = 4) or 5
// (KB = 5) words per pair per k-tile per stage are loads, so the wide-pair, long-unroll combinations
// run out first. KB = 4: set when the decode was one asm per hash (the next MT up spilled or reached
// 191-192 in every row); with the one-block decode the instantiated kernels take 54-188, the next
// MT up still spills at (NP, U) = (4, 1), and at (2, 2), (2, 4) and (4, 2) it would fit (173-174)
// but is not instantiated. KB = 5 gives up one MT where its two extra words per pair and k-tile do
// not fit. The build checks the real count of every instantiation against 190 (8 waves per SIMD)
// and zero scratch (r4dx's third_party/check_trellis_isa.cmake), so a compiler that allocates
// differently fails there.
static constexpr int r4d_tq_max_mt(int KB, int NP, int U) {
  return KB == 4 ? (NP == 1 ? (U == 1 || U == 2 || U == 4 ? 4 : 0)
                  : NP == 2 ? (U == 1 ? 4 : U == 2 || U == 4 ? 3 : 0)
                  : NP == 4 ? (U == 1 ? 2 : U == 2 ? 1 : 0)
                  : 0)
       : KB == 5 ? (NP == 1 ? (U == 1 || U == 2 || U == 4 ? 4 : 0)
                  : NP == 2 ? (U == 1 ? 4 : U == 2 ? 3 : U == 4 ? 2 : 0)
                  : NP == 4 ? (U == 1 ? 2 : U == 2 ? 1 : 0)
                  : 0)
       : 0;
}
static constexpr bool r4d_trellis_fits(int KB, int NP, int U, int MT) {
  return MT >= 1 && MT <= r4d_tq_max_mt(KB, NP, U);
}

static void r4d_trellis_fail(const std::string& what) {
  throw std::runtime_error("r4d gemm_trellis_nt_m64: " + what);
}

// The legality rules of docs/trellis-kernel.md 4.3, in one place.
static void r4d_trellis_check(int M, int K, int N, int n_split, int KB, int WV, int SK, int MT,
                              int NP, int SKG, int U, int NT, bool have_ws) {
  if (M < 1 || M > R4D_GEMM_TRELLIS_MAX_M) r4d_trellis_fail("M must be 1..64");
  if (K <= 0 || N <= 0 || K % 128 || N % 128) r4d_trellis_fail("K and N must be multiples of 128");
  if (KB != 4 && KB != 5) r4d_trellis_fail("KB " + std::to_string(KB) + " is not instantiated (4, 5)");
  if (WV != 1 && WV != 2 && WV != 4) r4d_trellis_fail("WV must be 1, 2 or 4");
  if (SK != 1 && SK != 2 && SK != 4 && SK != 8 && SK != 16)
    r4d_trellis_fail("SK must be 1, 2, 4, 8 or 16");
  if (SKG != 1 && SKG != 2 && SKG != 4 && SKG != 8) r4d_trellis_fail("SKG must be 1, 2, 4 or 8");
  if (NT != 0 && NT != 1) r4d_trellis_fail("NT must be 0 or 1");
  if (!r4d_trellis_fits(KB, NP, U, MT))
    r4d_trellis_fail("(KB, NP, U, MT) = (" + std::to_string(KB) + ", " + std::to_string(NP) + ", " +
                     std::to_string(U) + ", " + std::to_string(MT) +
                     ") is not instantiated (NP, U in {1,2,4}, NP*U <= 8; MT up to 4 at NP 1 and "
                     "at (2,1), 3 at (2,2), 3 (KB 4) / 2 (KB 5) at (2,4), 2 at (4,1), 1 at (4,2))");
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
  const float* svh;
  void* c;
  float* ws;
  unsigned* tickets;
  int M, K, N, WV, SK, SKG, NT, full;
  float out_scale;
  unsigned long long* clk;
  const float* wsc;     // R4DX_FAKEQ_W: the scale table (nullptr = the shipped kernel)
  int wq_gsh;           // log2 of the k-tiles per scale group
};

// One launch at a compile-time (KB, NP, U, MT); combinations r4d_trellis_fits rejects are never
// instantiated (the host check has already thrown for them).
template <int KB, int NP, int U, int MT>
static void r4d_tq_launch(const r4d_tq_launch_args& a) {
  if constexpr (r4d_trellis_fits(KB, NP, U, MT)) {
#ifdef R4D_TQ_WQ_UNIT
    // R4DX_FAKEQ_W's unit (r4d_gemm_trellis_nt_m64_wq.hip) instantiates only the rounding kernel, and the
    // shipped unit only the shipped one: the shipped kernels' code must not depend on whether a rounding
    // instantiation shares their translation unit (measured: it does, by a few instructions).
    hipLaunchKernelGGL((r4d_gemm_trellis_nt_m64_kernel<KB, NP, U, MT, false, true, const float*, int>),
                       a.grid, a.block, a.smem, a.stream, a.a0, a.a1, a.n_split, a.w, a.svh, a.c, a.ws,
                       a.tickets, a.M, a.K, a.N, a.WV, a.SK, a.SKG, a.full, a.out_scale, a.clk, a.wsc,
                       a.wq_gsh);
#else
#define LAUNCH(NTv)                                                                              \
  hipLaunchKernelGGL((r4d_gemm_trellis_nt_m64_kernel<KB, NP, U, MT, NTv>), a.grid, a.block, a.smem, \
                     a.stream, a.a0, a.a1, a.n_split, a.w, a.svh, a.c, a.ws, a.tickets, a.M, a.K,  \
                     a.N, a.WV, a.SK, a.SKG, a.full, a.out_scale, a.clk)
    if (a.NT) { LAUNCH(true); } else { LAUNCH(false); }
#undef LAUNCH
#endif
  } else {
    r4d_trellis_fail("(KB, NP, U, MT) not instantiated");
  }
}

template <int KB, int NP, int U>
static void r4d_tq_launch_mt(const r4d_tq_launch_args& a, int MT) {
  switch (MT) {
    case 1: r4d_tq_launch<KB, NP, U, 1>(a); return;
    case 2: r4d_tq_launch<KB, NP, U, 2>(a); return;
    case 3: r4d_tq_launch<KB, NP, U, 3>(a); return;
    case 4: r4d_tq_launch<KB, NP, U, 4>(a); return;
    default: r4d_trellis_fail("MT must be 1..4");
  }
}

template <int KB>
static void r4d_tq_launch_kb(const r4d_tq_launch_args& a, int NP, int U, int MT) {
  switch (NP * 16 + U) {
    case 1 * 16 + 1: r4d_tq_launch_mt<KB, 1, 1>(a, MT); return;
    case 1 * 16 + 2: r4d_tq_launch_mt<KB, 1, 2>(a, MT); return;
    case 1 * 16 + 4: r4d_tq_launch_mt<KB, 1, 4>(a, MT); return;
    case 2 * 16 + 1: r4d_tq_launch_mt<KB, 2, 1>(a, MT); return;
    case 2 * 16 + 2: r4d_tq_launch_mt<KB, 2, 2>(a, MT); return;
    case 2 * 16 + 4: r4d_tq_launch_mt<KB, 2, 4>(a, MT); return;
    case 4 * 16 + 1: r4d_tq_launch_mt<KB, 4, 1>(a, MT); return;
    case 4 * 16 + 2: r4d_tq_launch_mt<KB, 4, 2>(a, MT); return;
    default: r4d_trellis_fail("(NP, U) not instantiated");
  }
}

static void r4d_tq_run(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t svh, int64_t c,
                       int64_t ws, int64_t tickets, int M, int K, int N, int KB, int WV, int SK,
                       int MT, int NP, int SKG, int U, int NT, int full, float out_scale,
                       int64_t clk, int64_t stream, int64_t wsc = 0, int wq_gsh = 0) {
  r4d_trellis_check(M, K, N, n_split, KB, WV, SK, MT, NP, SKG, U, NT, ws != 0 && tickets != 0);
  if (full && svh == 0) r4d_trellis_fail("svh must be non-null");
  if (wsc != 0 && (wq_gsh < 1 || wq_gsh > 3))
    r4d_trellis_fail("wq_gsh (log2 of k-tiles per scale group) must be 1, 2 or 3");
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
  a.svh = (const float*)svh;
  a.c = (void*)c;
  a.ws = (float*)ws;
  a.tickets = (unsigned*)tickets;
  a.M = M; a.K = K; a.N = N; a.WV = WV; a.SK = SK; a.SKG = SKG; a.NT = NT; a.full = full;
  a.out_scale = out_scale;
  a.clk = (unsigned long long*)clk;
  a.wsc = (const float*)wsc;
  a.wq_gsh = wq_gsh;
  if (KB == 4) r4d_tq_launch_kb<4>(a, NP, U, MT);
  else r4d_tq_launch_kb<5>(a, NP, U, MT);
}

#ifdef R4D_TQ_WQ_UNIT
// R4DX_FAKEQ_W entries: the same two launches with the weight rounding on. The scale table is mandatory
// (null throws: the measurement must never silently run unquantized).
static void r4d_tq_need_wscale(int64_t wscale, int gsh, int K) {
  if (wscale == 0) r4d_trellis_fail("the _wq entries need a scale table (r4d_trellis_wscale_f32)");
  if (gsh < 1 || gsh > 3 || K % (16 << gsh))
    r4d_trellis_fail("gsh (log2 of k-tiles per scale group) must be 1, 2 or 3 and 16 << gsh divide K");
}
void r4d_gemm_trellis_nt_m64_wq(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t svh,
                                int64_t c, int64_t ws, int64_t tickets, int M, int K, int N, int KB,
                                int WV, int SK, int MT, int NP, int SKG, int U, int NT,
                                float out_scale, int64_t wscale, int gsh, int64_t stream) {
  r4d_tq_need_wscale(wscale, gsh, K);
  r4d_tq_run(a0, a1, n_split, w, svh, c, ws, tickets, M, K, N, KB, WV, SK, MT, NP, SKG, U, NT, 1,
             out_scale, 0, stream, wscale, gsh);
}
void r4d_gemm_trellis_nt_m64_raw_wq(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t c,
                                    int64_t ws, int64_t tickets, int M, int K, int N, int KB, int WV,
                                    int SK, int MT, int NP, int SKG, int U, int NT, int64_t wscale,
                                    int gsh, int64_t stream) {
  r4d_tq_need_wscale(wscale, gsh, K);
  r4d_tq_run(a0, a1, n_split, w, 0, c, ws, tickets, M, K, N, KB, WV, SK, MT, NP, SKG, U, NT, 0, 1.f, 0,
             stream, wscale, gsh);
}

size_t r4d_trellis_wscale_count(int K, int N, int gsh) {
  return (size_t)(K / 16 >> gsh) * (size_t)N;
}
void r4d_trellis_wscale_f32(int64_t w, int64_t scale, int K, int N, int KB, int gsh, int64_t stream) {
  if (K <= 0 || N <= 0 || N % 32 || gsh < 1 || gsh > 3 || K % (16 << gsh))
    throw std::runtime_error("r4d trellis_wscale_f32: N must be a multiple of 32, gsh 1, 2 or 3, and "
                             "K a multiple of 16 << gsh");
  if (KB != 4 && KB != 5) throw std::runtime_error("r4d trellis_wscale_f32: KB must be 4 or 5");
  if (w == 0 || scale == 0) throw std::runtime_error("r4d trellis_wscale_f32: null buffer");
  const long long waves = (long long)(N / 32) * (K / 16 >> gsh);
  dim3 grid((unsigned)((waves + 7) / 8)), block(8 * WAVE32);
  auto Wp = (const unsigned*)w;
  auto Sp = (float*)scale;
  hipStream_t st = (hipStream_t)stream;
  if (KB == 4) hipLaunchKernelGGL(r4d_trellis_wscale_kernel<4>, grid, block, 0, st, Wp, Sp, K, N, gsh);
  else hipLaunchKernelGGL(r4d_trellis_wscale_kernel<5>, grid, block, 0, st, Wp, Sp, K, N, gsh);
}
#else  // the shipped unit's entries

void r4d_gemm_trellis_nt_m64(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t svh,
                             int64_t c, int64_t ws, int64_t tickets, int M, int K, int N, int KB,
                             int WV, int SK, int MT, int NP, int SKG, int U, int NT, float out_scale,
                             int64_t stream) {
  r4d_tq_run(a0, a1, n_split, w, svh, c, ws, tickets, M, K, N, KB, WV, SK, MT, NP, SKG, U, NT, 1,
             out_scale, 0, stream);
}

void r4d_gemm_trellis_nt_m64_raw(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t c,
                                 int64_t ws, int64_t tickets, int M, int K, int N, int KB, int WV,
                                 int SK, int MT, int NP, int SKG, int U, int NT, int64_t clk,
                                 int64_t stream) {
  r4d_tq_run(a0, a1, n_split, w, 0, c, ws, tickets, M, K, N, KB, WV, SK, MT, NP, SKG, U, NT, 0,
             1.f, clk, stream);
}

int r4d_gemm_trellis_nt_m64_has_rate(int KB) { return KB == 4 || KB == 5; }
int r4d_gemm_trellis_nt_m64_max_m(void) { return R4D_GEMM_TRELLIS_MAX_M; }
size_t r4d_gemm_trellis_nt_m64_ws_bytes(int M, int N, int SKG) {
  return (size_t)SKG * (size_t)M * (size_t)N * sizeof(float);
}
size_t r4d_gemm_trellis_nt_m64_tickets_bytes(int N) {
  return (size_t)(N > 0 ? N / 128 : 0) * sizeof(unsigned);
}
void r4d_gemm_trellis_nt_m64_zero_tickets(int64_t tickets, size_t bytes, int64_t stream) {
  if (bytes == 0) return;
  if (tickets == 0) throw std::runtime_error("r4d gemm_trellis_nt_m64_zero_tickets: null buffer");
  const hipError_t e = hipMemsetAsync((void*)tickets, 0, bytes, (hipStream_t)stream);
  if (e != hipSuccess)
    throw std::runtime_error(std::string("r4d gemm_trellis_nt_m64_zero_tickets: ") +
                             hipGetErrorString(e));
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
#endif  // R4D_TQ_WQ_UNIT
