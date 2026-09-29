// r4d_gemm_w4a16_nt_m64.hip - r4d_gemm_w4a16_nt_m64: 4-bit-weight skinny GEMM for M up to 64,
// gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T, f16 A, bf16 C, fp32 accumulate.
//
// Same problem and the same decomposition as r4d_gemm_bf16_nt_m64 -- a weight-streaming GEMM whose
// M is a speculative decode batch, so every output element is read once and the whole cost is the
// weight -- with the weight at 4 bits instead of 16.  That is the entire point: at M=8 a projection
// moves M*K*2 bytes of activation against N*K bytes of weight, 82 KiB against 44 MiB on the
// drafter's MLP, so the format of the WEIGHT is the only thing in the cost model.
//
// WHY f16 AND NOT bf16.  A 4-bit weight has to be widened to a WMMA input type, and the cheapest
// widening is a byte permute into a fixed exponent: `0x6400 | n` is the f16 for 1024 + n, exactly,
// for every n in 0..15, so four v_perm_b32 turn one dword of eight nibbles into a whole 16x16x16
// fragment.  The quantizer's zero point then has to be subtracted, and gfx1201 has NO packed bf16
// arithmetic (v_pk_add_bf16 is not an instruction on this part) while v_pk_add_f16 is -- so the
// subtract is four instructions in f16 and would be sixteen in bf16, through f32 and back.  Ten
// VALU per fragment against one WMMA leaves the kernel memory-bound over the whole M band; the
// bf16 route does not.  Everything the WMMA sees is exact: the activation is an f16 value, the
// weight is a small integer, and the group scale is applied to the f32 accumulator afterwards.
//
// AND WHY THAT CAPS IT.  Widening to f16 reaches the SLOWEST matrix instruction on this part:
// v_wmma_f32_16x16x16_f16 issues at 207 TF/s where v_wmma_i32_16x16x16_iu8 and
// v_wmma_f32_16x16x16_fp8_fp8 both issue at 407-412 and v_wmma_i32_16x16x32_iu4 at 830.  Below
// M=32 that costs nothing, because this kernel is at the DRAM roofline there -- 620-648 GB/s, 2.0x
// the block-fp8 path, and an ablation that deletes the unpack entirely moves M=8 by 0.25%.  At
// M=64 it is the whole story: 1.00-1.03x, 72-78 TF/s, and deleting the unpack only reaches 82-86.
// r4d_gemm_w4a8_nt_m64 runs the SAME packed weight through the int8 form for 1.5-1.9x there.  This
// kernel owns the small-M band and that band is finished.
//
// WHY THE SCALE IS NOT FOLDED INTO THE FRAGMENT.  It is tempting to fold both quantizer parameters
// into one v_pk_fma_f16 -- fma(1024 + n, s, -(1024 + z) * s) -- and drop the per-group epilogue.
// That is wrong by 12%: -(1024 + z) * s is not representable in f16, and the term it cancels is
// ~1024*s against a result of ~4*s, so half an f16 ULP of the addend is an eighth of the answer.
// The subtraction has to happen where 1024 + z is exact, which is the integer grid, and the scale
// has to be applied where the cancellation has already happened, which is the f32 accumulator.
//
// THE ACTIVATION IS NOT FREE, EVEN AT M=8.  A 16x16x16 fragment covers sixteen ROWS whether M is 8
// or 64, and every column tile re-reads the whole of A, so a wave that owns one column tile moves
// 16*K*2 bytes of activation against 16*K/2 bytes of weight -- four times as much, out of cache
// rather than DRAM, but four times as many load instructions. NPW (column tiles per WAVE) is the
// answer and it costs no LDS and no barrier: the wave loads one activation fragment and spends it
// on NPW weight fragments, so the ratio falls to 4*MT/NPW. WV still exists and still sets how many
// column tiles a BLOCK covers, but WV alone does nothing for this -- the waves of a block issue
// their own loads and only share through the cache.
//
// LAYOUT.  Wq is pre-permuted offline into exactly the order the fragment wants, so a lane's 16
// bytes are contiguous and a wave's weight load for four k steps is one 512-byte global_load_b128.
// Nothing is staged through LDS on the weight path and nothing is masked: with
//
//     idx = lane%16 , k = 8*(e>>2) + 4*(lane>>4) + (e&3)      (r4d_gdn_wmma.h)
//
// fragment element e of lane l is W[n0 + (l&15)][16s + 8*(e>>2) + 4*(l>>4) + (e&3)], and the dword
// holding it stores that value in nibble 2e for e<4 and nibble 2(e-4)+1 for e>=4 -- the order the
// two-mask, four-permute unpack below produces.  The packer in radiance_w4.py builds this and the
// isolation bench checks it against a dequantized reference, element by element.
//
// QUANTIZER.  Asymmetric, per output channel, per group of G contiguous K:
// w ~= scale * (q - zero) with q in 0..15 and an INTEGER zero, which is what makes the subtract
// exact.  Both parameters live in one dword per (row, group): the f16 scale in the low half and the
// f16 of -(1024 + zero) in the high half, ready to be broadcast into a v_pk_add_f16.
//
// THE GROUP IS A TEMPLATE PARAMETER.  G is 32 or 64 through r4d_gemm_w4a16_nt_m64_g, and the
// build default R4D_GEMM_W4_GROUP (64) through r4d_gemm_w4a16_nt_m64, which is the same template at
// that G -- so a caller that never asks for a group gets the instruction stream it always got.  The
// packed weight does not depend on G at all (a 64-K block is 64 K whatever the group), only the
// Wsz stride does, which is what lets one model mix groups tensor by tensor.  G >= 64 is the
// original loop: a group is bpg = G/64 whole packed blocks.  G = 32 splits a block instead -- k
// steps 0,1 belong to one group and 2,3 to the next -- so it has its own loop body, with one
// weight load per block and two scale loads, restarts and epilogues.  Both need every split to
// start on a packed block, hence K % (SK * max(G, 64)) == 0.
#include "r4d.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>
#include <string>

#define R4D_GEMM_W4_MAX_M 64
#ifndef R4D_GEMM_W4_GROUP
#define R4D_GEMM_W4_GROUP 64         // build default K per (scale, zero) pair: what the ungrouped
#endif                               // entry point and _group() serve
#define R4D_GEMM_W4_KPB   64         // K per packed block: 4 k steps, 16 bytes per lane
#define WAVE32 32
static_assert(R4D_GEMM_W4_GROUP == 32 || R4D_GEMM_W4_GROUP % R4D_GEMM_W4_KPB == 0,
              "R4D_GEMM_W4_GROUP must be 32 or a multiple of the 64-K packed block");

typedef _Float16 v8h  __attribute__((ext_vector_type(8)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));

// NT reads the weight non-temporally. The weight is streamed once per step and never reused, while
// the activation is re-read once per column tile and wants to stay resident, so telling the cache
// not to keep the weight decides which of the two occupies the last level. Measured on gfx1201
// against a POOLED working set -- one copy of any of these weights fits the 64 MiB last level, so a
// single-buffer bench measures the wrong machine -- it is worth +3 to +4% at M=8 and M=16
// (2.00x vs 1.94x of the block-fp8 path on the MLP) and it is a LOSS from M=24 up on the shapes
// with the longest K (down_proj: 1.05x vs 0.97x at M=64). So it is per-call, like every other
// geometry parameter, and the table in radiance_w4.py carries the measured value per M band.

// NPW (column tiles per WAVE) is a template parameter because it sizes the accumulator array, and a
// launch argument because the right value is a property of M: at M=8 the kernel is at the DRAM
// roofline and NPW>1 only costs blocks, while at M=64 the activation re-read is four times the
// weight and NPW=4 is what claws it back. Instantiated for the two values that were measured to
// win; anything else is rejected rather than silently rounded.

// Address-space-qualified so the loads stay global_load rather than flat_load: on gfx12 a flat
// load increments dscnt as well as loadcnt, so a later LDS wait would also wait on HBM.  See the
// same declaration in r4d_gemm_bf16_nt_m64.hip for why the element type is a scalar there.
typedef __attribute__((address_space(1))) const uint4v*          r4d_gw4;
typedef __attribute__((address_space(1))) const unsigned long long* r4d_gu2;
typedef __attribute__((address_space(1))) const unsigned*        r4d_gu1;

__device__ __forceinline__ v8h fragA(r4d_gu2 p, int u) {
  union { unsigned long long w[2]; v8h f; } x;
  x.w[0] = p[4 * u];
  x.w[1] = p[4 * u + 2];
  return x.f;
}

// One dword of eight nibbles -> one 16x16x16 f16 B fragment, minus the zero point.
// t0/t1 split the even and odd nibbles into bytes; each v_perm_b32 pairs two of those bytes with
// the constant high byte 0x64, which is the f16 exponent field for 1024.  nz carries
// -(1024 + zero) in BOTH halves so one v_pk_add_f16 corrects a pair.
// The unpack was priced once by deleting it: the WHOLE path is 0.13-0.54% of this kernel at M=8,
// where it is at the DRAM roofline and the arithmetic hides in the load latency. At M=64 it is
// 11.5-18.0%, but that band belongs to r4d_gemm_w4a8_nt_m64 -- and even with the unpack deleted
// entirely the f16 kernel is still 23-48% behind the int8 one there, because the gap is the matrix
// instruction and not the dequant. Nothing here is worth a knob.
#ifndef R4D_GEMM_W4_TWOS
#define R4D_GEMM_W4_TWOS 1
#endif

// REFUTED, and worth not trying again: folding the group scale into the B fragment -- one
// v_pk_mul_f16 per pair on the finished integer q, which is exact in f16 -- removes the second
// accumulator array entirely and takes the M=64 shape from 192 VGPRs to 135.  It is SLOWER on three
// of the four drafter shapes (qkv +1.2%, o +3.5%, down +5.5%, gate_up -2.3%).  The multiply lands
// inside the per-fragment dependent chain between the unpack and the WMMA, where the per-group
// epilogue it replaces sat outside it; this kernel is stall-bound, not register-bound.

__device__ __forceinline__ v8h dequant(unsigned w, __half2 nz) {
  const unsigned K64 = 0x64646464u;
#if R4D_GEMM_W4_TWOS
  // The stored nibble is a 4-bit TWO'S COMPLEMENT code, because that is what
  // r4d_gemm_w4a8_nt_m64 needs -- it shifts the nibble into the byte's high half and reads the
  // byte signed.  This kernel wants OFFSET BINARY, so that 1024 + n minus the constant 1024 + 8
  // is the value.  The two differ by bit 3 of the nibble, so one XOR per spread dword converts,
  // and the zero point is then pinned to 8 for every group.  Two instructions per fragment, on
  // the kernel whose whole unpack was measured at 0.25% of its time at M=8 -- which is why the
  // conversion lives here and not in the int8 kernel, where the unpack is three instructions and
  // does matter.  One quantised weight, two kernels, dispatched on M.
  const unsigned t0 = (w & 0x0F0F0F0Fu) ^ 0x08080808u;
  const unsigned t1 = ((w >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u;
#else
  const unsigned t0 = w & 0x0F0F0F0Fu;
  const unsigned t1 = (w >> 4) & 0x0F0F0F0Fu;
#endif
  union { unsigned u[4]; __half2 h2[4]; v8h f; } x;
  x.u[0] = __builtin_amdgcn_perm(K64, t0, 0x05010400u);   // nibbles 0, 2
  x.u[1] = __builtin_amdgcn_perm(K64, t0, 0x05030402u);   // nibbles 4, 6
  x.u[2] = __builtin_amdgcn_perm(K64, t1, 0x05010400u);   // nibbles 1, 3
  x.u[3] = __builtin_amdgcn_perm(K64, t1, 0x05030402u);   // nibbles 5, 7
#pragma unroll
  for (int i = 0; i < 4; i++) x.h2[i] = __hadd2(x.h2[i], nz);
  return x.f;
}

__device__ __forceinline__ v8f mmah(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

// MT = row tiles per BLOCK, as in the bf16 kernel: more of them amortise the weight fragment and
// its unpack over more WMMA, fewer put more blocks on the machine.  G = K per (scale, zero) pair.
template <int G, int MT, int NPW, bool NT>
__global__ __launch_bounds__(1024) void r4d_gemm_w4a16_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned* __restrict__ Wq,
    const unsigned* __restrict__ Wsz, __hip_bfloat16* __restrict__ C,
    int M, int K, int N, int WV, int SK) {
  static_assert(G == 32 || G % R4D_GEMM_W4_KPB == 0,
                "a group is half a packed block or a whole number of them");
  extern __shared__ float red[];        // WV*NPW*SK*256 floats: a 16x16 tile per (column, split)
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / G;                              // groups this split owns
  const int kbase = ksplit * Ks;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  if (n0 < N) {
    // Activation: rows clamped, never masked -- a tile past M re-reads the last valid row and
    // computes a result nobody stores.
    unsigned aoff[MT];
#pragma unroll
    for (int i = 0; i < MT; i++)
      aoff[i] = rowGo((unsigned)K, (mt0 + i) * 16, M - 1, lane) + (unsigned)kbase;
    r4d_gu2 pa[MT];
#pragma unroll
    for (int i = 0; i < MT; i++) pa[i] = (r4d_gu2)(const unsigned long long*)(A + aoff[i]);

    // Weight: one uint4 per lane per four k steps, tiles already permuted, so the whole k loop is
    // global_load_b128 with a constant stride and no address arithmetic per step.
    const int kpb = K / R4D_GEMM_W4_KPB;                  // packed blocks in a full row of tiles
    const int nsz = K / G;                                // (scale, zero) dwords per row of tiles
    r4d_gw4 pw[NPW];
    r4d_gu1 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      // Clamped, like the row tiles: a wave whose last column tiles run past N re-reads the last
      // real tile and computes a result the store discards, rather than reading memory it does not
      // own or diverging the whole k loop on a mask.
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (r4d_gw4)(const uint4v*)(Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W4_KPB) * WAVE32 + lane;
      // Quantizer parameters: 16 distinct dwords per group, read by the 16 lanes that own the rows.
      ps[j] = (r4d_gu1)(const unsigned*)(Wsz) +
              (size_t)(t * nsz + kbase / G) * 16 + (lane & 15);
    }

    if constexpr (G >= R4D_GEMM_W4_KPB) {
      // A group is bpg whole packed blocks.  This is the loop as it was before G was a template
      // parameter, token for token, so the build-default instantiation compiles to the same code.
      const int bpg = G / R4D_GEMM_W4_KPB;                // packed blocks per group

      for (int g = 0; g < groups; g++) {
        float sc[NPW];
        __half2 nz[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const unsigned sz = ps[j][(size_t)g * 16];
          sc[j] = __half2float(__ushort_as_half((unsigned short)(sz & 0xFFFFu)));
          nz[j] = __half2half2(__ushort_as_half((unsigned short)(sz >> 16)));
        }

        // The first WMMA of a group takes an inline zero for C, so the group accumulator costs no
        // instruction to clear.
        v8f g_acc[NPW][MT];
#pragma unroll
        for (int b = 0; b < bpg; b++) {
          uint4v wv[NPW];
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const size_t o = (size_t)(g * bpg + b) * WAVE32;
            wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
          }
#pragma unroll
          for (int s = 0; s < 4; s++) {
            const int u = b * 4 + s;
            v8h afrag[MT];
#pragma unroll
            for (int i = 0; i < MT; i++) afrag[i] = fragA(pa[i], u);   // loaded once, spent NPW times
#pragma unroll
            for (int j = 0; j < NPW; j++) {
              const v8h bfrag = dequant(wv[j][s], nz[j]);
#pragma unroll
              for (int i = 0; i < MT; i++)
                g_acc[j][i] = u == 0 ? mmah(afrag[i], bfrag, (v8f)(0.f))
                                     : mmah(afrag[i], bfrag, g_acc[j][i]);
            }
          }
        }
#pragma unroll
        for (int j = 0; j < NPW; j++)
#pragma unroll
          for (int i = 0; i < MT; i++)
#pragma unroll
            for (int e = 0; e < 8; e++) acc[j][i][e] = fmaf(g_acc[j][i][e], sc[j], acc[j][i][e]);
#pragma unroll
        for (int i = 0; i < MT; i++) pa[i] += (G / 16) * 4;
      }
    } else {
      // A packed block holds gpb = 64/G groups (two, at G = 32).  The weight is still loaded once
      // per block, as one b128 per lane; what repeats per group is the scale load, the restart of
      // the group accumulator on the group's first k step, and the f32 epilogue.  Twice the
      // epilogues and twice the scale bytes of G = 64, which is the price of the finer group.
      constexpr int gpb = R4D_GEMM_W4_KPB / G;            // groups per packed block
      constexpr int spg = G / 16;                         // k steps per group
      const int blocks = groups / gpb;                    // packed blocks this split owns

      for (int b = 0; b < blocks; b++) {
        uint4v wv[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const size_t o = (size_t)b * WAVE32;
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

          v8f g_acc[NPW][MT];
#pragma unroll
          for (int u = 0; u < spg; u++) {                 // u: k step within the group
            const int s = h * spg + u;                    // s: k step within the packed block
            v8h afrag[MT];
#pragma unroll
            for (int i = 0; i < MT; i++) afrag[i] = fragA(pa[i], u);
#pragma unroll
            for (int j = 0; j < NPW; j++) {
              const v8h bfrag = dequant(wv[j][s], nz[j]);
#pragma unroll
              for (int i = 0; i < MT; i++)
                g_acc[j][i] = u == 0 ? mmah(afrag[i], bfrag, (v8f)(0.f))
                                     : mmah(afrag[i], bfrag, g_acc[j][i]);
            }
          }
#pragma unroll
          for (int j = 0; j < NPW; j++)
#pragma unroll
            for (int i = 0; i < MT; i++)
#pragma unroll
              for (int e = 0; e < 8; e++) acc[j][i][e] = fmaf(g_acc[j][i][e], sc[j], acc[j][i][e]);
#pragma unroll
          for (int i = 0; i < MT; i++) pa[i] += spg * 4;
        }
      }
    }
  }

  // Split-K reduction and store: identical to r4d_gemm_bf16_nt_m64, one row tile at a time so the
  // LDS is WV*SK KiB whatever MT is, and in a fixed order over splits so the result does not depend
  // on how the waves were scheduled.
  const int ncols = WV * NPW;
  const int nthreads = WV * SK * WAVE32;
#pragma unroll
  for (int i = 0; i < MT; i++) {
    __syncthreads();
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int slot = ((col_local * NPW + j) * SK + ksplit) * 256;
#pragma unroll
      for (int e = 0; e < 8; e++) red[slot + e * WAVE32 + lane] = acc[j][i][e];
    }
    __syncthreads();
    for (int idx = tid; idx < ncols * 256; idx += nthreads) {
      const int col = idx >> 8, off = idx & 255;
      float v = 0.f;
      for (int s = 0; s < SK; s++) v += red[(col * SK + s) * 256 + off];
      const int l = off & 31;
      const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
      const int m = (mt0 + i) * 16 + 8 * (l >> 4) + (off >> 5);
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)v;
    }
  }
}

// The host side of both entry points, at one group.  Every split has to start on a packed block
// (the weight pointer is kbase/64 blocks in) and hold whole groups, so the K rule is
// K % (SK * max(G, 64)) == 0 -- SK * G for the groups that are whole blocks, as it always was.
template <int G>
static void r4d_gemm_w4a16_nt_m64_launch(int64_t a, int64_t wq, int64_t wsz, int64_t c, int M,
                                         int K, int N, int WV, int SK, int MB, int NPW, int NT,
                                         int64_t stream) {
  constexpr int kdiv = G > R4D_GEMM_W4_KPB ? G : R4D_GEMM_W4_KPB;
  if (M < 1 || M > R4D_GEMM_W4_MAX_M)
    throw std::runtime_error("r4d gemm_w4a16_nt_m64: M must be 1..64");
  if (N % 16) throw std::runtime_error("r4d gemm_w4a16_nt_m64: N must be a multiple of 16");
  if (K % (SK * kdiv))
    throw std::runtime_error("r4d gemm_w4a16_nt_m64: K must be divisible by SK*" +
                             std::to_string(kdiv) + " at group " + std::to_string(G));
  if (WV * SK * WAVE32 > 1024) throw std::runtime_error("r4d gemm_w4a16_nt_m64: threads/block > 1024");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_w4a16_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 4) throw std::runtime_error("r4d gemm_w4a16_nt_m64: NPW must be 1 or 4");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w4a16_nt_m64: WV*NPW*SK too large");
  auto Ap = (const unsigned short*)a;
  auto Wp = (const unsigned*)wq;
  auto Sp = (const unsigned*)wsz;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT, NP, TT) hipLaunchKernelGGL((r4d_gemm_w4a16_nt_m64_kernel<G, MT, NP, TT>), grid, \
                                              block, smem, st, Ap, Wp, Sp, Cp, M, K, N, WV, SK)
#define LAUNCH_MB(NP, TT) switch (MB) { \
    case 1: LAUNCH(1, NP, TT); break; \
    case 2: LAUNCH(2, NP, TT); break; \
    case 3: LAUNCH(3, NP, TT); break; \
    case 4: LAUNCH(4, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w4a16_nt_m64: MB unsupported"); }
  if (NPW == 1) {
    if (NT) { LAUNCH_MB(1, true); } else { LAUNCH_MB(1, false); }
  } else {
    if (NT) { LAUNCH_MB(4, true); } else { LAUNCH_MB(4, false); }
  }
#undef LAUNCH_MB
#undef LAUNCH
}

// The build default: the group this library was compiled for, and the entry every caller that
// predates per-tensor groups uses.
void r4d_gemm_w4a16_nt_m64(int64_t a, int64_t wq, int64_t wsz, int64_t c, int M, int K, int N,
                           int WV, int SK, int MB, int NPW, int NT, int64_t stream) {
  r4d_gemm_w4a16_nt_m64_launch<R4D_GEMM_W4_GROUP>(a, wq, wsz, c, M, K, N, WV, SK, MB, NPW, NT,
                                                  stream);
}

// Any instantiated group, chosen per call -- one model can carry tensors packed at different
// groups, and the packed weight is the same bytes whichever group its scales were fitted at.
// Exactly 32 and 64: a build default outside that set (128, 192, ...) is served by
// r4d_gemm_w4a16_nt_m64 alone, not here.  Asking for the build default here, when it is one of the
// two, runs the same kernel r4d_gemm_w4a16_nt_m64 does.
int r4d_gemm_w4a16_nt_m64_has_group(int group) {
  return group == 32 || group == 64;
}

void r4d_gemm_w4a16_nt_m64_g(int group, int64_t a, int64_t wq, int64_t wsz, int64_t c, int M, int K,
                             int N, int WV, int SK, int MB, int NPW, int NT, int64_t stream) {
#define LAUNCH_G(GG) r4d_gemm_w4a16_nt_m64_launch<GG>(a, wq, wsz, c, M, K, N, WV, SK, MB, NPW, NT, \
                                                      stream)
  switch (group) {
    case 32: LAUNCH_G(32); return;
    case 64: LAUNCH_G(64); return;
    default: break;
  }
#undef LAUNCH_G
  throw std::runtime_error("r4d gemm_w4a16_nt_m64_g: group " + std::to_string(group) +
                           " is not instantiated (32, 64; the build default " +
                           std::to_string(R4D_GEMM_W4_GROUP) +
                           " is r4d_gemm_w4a16_nt_m64's)");
}

int r4d_gemm_w4a16_nt_m64_max_m(void) { return R4D_GEMM_W4_MAX_M; }
int r4d_gemm_w4a16_nt_m64_group(void) { return R4D_GEMM_W4_GROUP; }
