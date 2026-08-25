// r4d_gemm_w4a8_nt_m64.hip - r4d_gemm_w4a8_nt_m64: 4-bit-weight, 8-bit-activation skinny GEMM for
// M up to 64, gfx1201.  C[M,N] = A[M,K] @ dequant(Wq)[N,K]^T with A quantised per row.
//
// WHY THIS EXISTS ALONGSIDE r4d_gemm_w4a16_nt_m64.  The f16 kernel is at the DRAM roofline at M=8
// (620-648 GB/s, 2.00x the block-fp8 path) and there is nothing left to win there.  At M=64 -- the
// drafter's batch at eight concurrent requests -- it is not near memory at all: 300-350 GB/s and
// 72-78 TF/s, against a MEASURED v_wmma_f32_16x16x16_f16 issue ceiling of 207 TF/s.  Removing the
// unpack entirely only moves it to 82-86.  The wall is the f16 matrix instruction, and on this part
// it is the slow one:
//
//     v_wmma_f32_16x16x16_f16       207 TF/s      5.95 cyc/instr
//     v_wmma_f32_16x16x16_fp8_fp8   412           2.99
//     v_wmma_i32_16x16x16_iu8       407           3.03
//     v_wmma_i32_16x16x32_iu4       830           2.97
//
// So the widening that made the f16 kernel cheap is what caps it: every 8-bit form runs at twice
// the rate. This kernel takes the int8 one, which needs no exponent games at all.
//
// THE UNPACK IS THREE INSTRUCTIONS AND NEEDS NO ZERO POINT.  Store the weight as a 4-bit TWO'S
// COMPLEMENT code and put the nibble in the HIGH half of its byte: for q in -8..7 the byte 16*q is
// exactly the signed int8 of 16*q, so
//
//     lo = (w << 4) & 0xF0F0F0F0     bytes <- nibbles 0,2,4,6      (elements 0..3)
//     hi =  w       & 0xF0F0F0F0     bytes <- nibbles 1,3,5,7      (elements 4..7)
//
// is the whole dequant: two v_and_b32 and one v_lshlrev_b32 per eight weights, against eleven for
// the f16 route, and the factor of 16 folds into the group scale offline.  A symmetric grid also
// deletes the zero point, which in an integer kernel would otherwise cost a per-group row-sum of A
// and a second correction term.  The acceptance sweep put a symmetric 4-bit grid at the top of the
// 4-bit block anyway, so this costs nothing to pick.
//
// The nibble ORDER is the one r4d_gemm_w4a16_nt_m64 already uses: its f16 unpack pairs nibble 2b
// into element b and nibble 2b+1 into element 4+b, which is exactly what the two masks above
// produce.  One packed layout, two kernels.
//
// IT SHARES ITS WEIGHT WITH r4d_gemm_w4a16_nt_m64, BYTE FOR BYTE.  The two kernels want the same
// nibbles in the same order, and a signed 4-bit code is what the f16 kernel computes anyway when
// its integer zero point is the constant 8: it builds 1024+n and adds -(1024+8), leaving n-8.  So
// one quantised weight feeds both, and the scale array is the f16 kernel's (scale, -(1024+zero))
// dword with only the low half read here.  The stored scale multiplies q-8 while the WMMA here
// sees 16*(q-8), so the factor of 1/16 rides along with the per-row activation scale at the store,
// where it costs nothing.  That matters because the dispatch is per call: below M=16 the f16
// kernel is already at the DRAM roofline and there is nothing for an int8 one to win, while the
// activation quantiser costs a launch -- so the small-M path stays on f16 and the same bytes serve
// both.
//
// THE ACTIVATION IS PRE-SHUFFLED TOO, BUT ONLY INSIDE A K STEP.  A fragment element e of lane l is
// A[row][16s + 8*(e>>2) + 4*(l>>4) + (e&3)], so read out of a row-major A a lane's eight bytes are
// two dwords eight bytes apart: the ISA showed 64 global_load_b32 against 64 WMMA for one group,
// and 32 s_wait_loadcnt with them.  A is OURS -- the quantiser writes it -- so it is written with
// the sixteen bytes of each k step reordered to [half 0's eight][half 1's eight].  A lane's
// fragment is then one global_load_b64 and the group costs 32 loads instead of 64.
//
// It is worth saying what did NOT work, because the obvious version is worse: laying A out as two
// PLANES, [2][M][K/2], makes a lane's four k steps contiguous and the group eight
// global_load_b128 -- and it LOST, 106.7 us against 96.5 at M=64.  Sixteen lanes of a wave then
// address one plane and sixteen the other, Mp*K/2 bytes away, so a single instruction spans two
// distant regions and doubles the cache lines it touches.  The shuffle below keeps every lane of a
// wave inside the same sixteen rows the row-major version touched, and only makes each lane's own
// eight bytes adjacent.
//
// THE ACTIVATION SCALE COMES OUT OF THE GROUP LOOP.  Per-row symmetric quantisation makes the row
// scale a constant in n and in the group index, so sum_g s_g * <a_row, w_g> * ar = ar * sum_g ...:
// it is applied once at the store and never inside the k loop.  What does stay in the loop is one
// v_cvt_f32_i32 per accumulator element per group, because group scales differ -- eight of them
// against eight WMMA, which is why R4D_GEMM_W4A8_GROUP is 256 and not 128.
#include "r4d.h"
#include "r4d_gdn_wmma.h"
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <cstdint>
#include <stdexcept>

#define R4D_GEMM_W4A8_MAX_M 64
#ifndef R4D_GEMM_W4A8_GROUP
#define R4D_GEMM_W4A8_GROUP 256      // K per scale
#endif
#define R4D_GEMM_W4A8_KPB   64       // K per packed block: 4 k steps, 16 bytes per lane
#define WAVE32 32

typedef int      v8i   __attribute__((ext_vector_type(8)));
typedef int      v2i   __attribute__((ext_vector_type(2)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));

typedef __attribute__((address_space(1))) const uint4v*   r4d_gw4;
typedef __attribute__((address_space(1))) const unsigned* r4d_gu1;
typedef __attribute__((address_space(1))) const unsigned* r4d_gh1;
typedef unsigned uint2v __attribute__((ext_vector_type(2)));
typedef __attribute__((address_space(1))) const uint2v*   r4d_ga2;

// A fragment out of the pre-permuted plane: eight contiguous bytes per k step, so a uint4 is two
// steps and the whole group is eight global_load_b128 per row tile.
#ifndef R4D_W4A8_APERM
#define R4D_W4A8_APERM 1        // 0 reads A row-major, for the isolation bench only
#endif

__device__ __forceinline__ v2i fragA8(r4d_ga2 p, int u) {
  const uint2v w = p[2 * u];
  v2i r;
  r[0] = (int)w[0];
  r[1] = (int)w[1];
  return r;
}

__device__ __forceinline__ v2i fragA8row(r4d_gu1 p, int u) {
  v2i r;
  r[0] = p[4 * u];
  r[1] = p[4 * u + 2];
  return r;
}

// The whole dequant: two ands and a shift, no zero point. Priced once against the rest of the
// kernel by deleting it -- 0.6% at qkv, 0.9% at o, 5.1% at gate_up and 1.2% at down for M=64, and
// under 0.2% at M=8 -- so there is nothing here worth a knob.
__device__ __forceinline__ v2i dequant8(unsigned w) {
  v2i r;
  r[0] = (int)((w << 4) & 0xF0F0F0F0u);
  r[1] = (int)(w & 0xF0F0F0F0u);
  return r;
}

__device__ __forceinline__ v8i mmai(v2i a, v2i b, v8i c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c, false);
}

template <int MT, int NPW, bool NT>
__global__ __launch_bounds__(1024) void r4d_gemm_w4a8_nt_m64_kernel(
    const signed char* __restrict__ A, const float* __restrict__ Ascale,
    const unsigned* __restrict__ Wq, const unsigned* __restrict__ Ws,
    __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV, int SK) {
  extern __shared__ float red[];
  const int tid = threadIdx.x, wave = tid / WAVE32, lane = tid % WAVE32;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / R4D_GEMM_W4A8_GROUP;
  const int kbase = ksplit * Ks;

  v8f acc[NPW][MT];
#pragma unroll
  for (int j = 0; j < NPW; j++)
#pragma unroll
    for (int i = 0; i < MT; i++) acc[j][i] = (v8f)(0.f);

  if (n0 < N) {
    // Rows clamped, never masked, as in the f16 kernel: a tile past M recomputes the last valid
    // row and the store discards it.  A is int8 so the row stride is K bytes, not 2K.
    // Plane (lane >> 4), row clamped into 0..M-1, and the group's byte offset is K/2 per row.
#if R4D_W4A8_APERM
    r4d_ga2 pa[MT];
#else
    r4d_gu1 pa[MT];
#endif
#pragma unroll
    for (int i = 0; i < MT; i++) {
      int r = (mt0 + i) * 16 + (lane & 15);
      r = r < M - 1 ? r : M - 1;
#if R4D_W4A8_APERM
      const size_t off = (size_t)r * K + kbase + 8 * (lane >> 4);
      pa[i] = (r4d_ga2)(const uint2v*)(A + off);
#else
      const size_t off = (size_t)r * K + kbase + 4 * (lane >> 4);
      pa[i] = (r4d_gu1)(const unsigned*)(A + off);
#endif
    }

    const int kpb = K / R4D_GEMM_W4A8_KPB;
    const int nsz = K / R4D_GEMM_W4A8_GROUP;
    r4d_gw4 pw[NPW];
    r4d_gh1 ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;
      pw[j] = (r4d_gw4)(const uint4v*)(Wq) +
              (size_t)(t * kpb + kbase / R4D_GEMM_W4A8_KPB) * WAVE32 + lane;
      ps[j] = (r4d_gh1)(const unsigned*)(Ws) +
              (size_t)(t * nsz + kbase / R4D_GEMM_W4A8_GROUP) * 16 + (lane & 15);
    }

    const int bpg = R4D_GEMM_W4A8_GROUP / R4D_GEMM_W4A8_KPB;

    for (int g = 0; g < groups; g++) {
      float sc[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++)
        sc[j] = __half2float(__ushort_as_half(
            (unsigned short)(ps[j][(size_t)g * 16] & 0xFFFFu)));

      v8i g_acc[NPW][MT];
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
          v2i afrag[MT];
#pragma unroll
#if R4D_W4A8_APERM
          for (int i = 0; i < MT; i++) afrag[i] = fragA8(pa[i], u);
#else
          for (int i = 0; i < MT; i++) afrag[i] = fragA8row(pa[i], u);
#endif
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const v2i bfrag = dequant8(wv[j][s]);
#pragma unroll
            for (int i = 0; i < MT; i++)
              g_acc[j][i] = (u == 0) ? mmai(afrag[i], bfrag, (v8i)(0))
                                     : mmai(afrag[i], bfrag, g_acc[j][i]);
          }
        }
      }
#pragma unroll
      for (int j = 0; j < NPW; j++)
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
          for (int e = 0; e < 8; e++)
            acc[j][i][e] = fmaf((float)g_acc[j][i][e], sc[j], acc[j][i][e]);
#pragma unroll
#if R4D_W4A8_APERM
      for (int i = 0; i < MT; i++) pa[i] += R4D_GEMM_W4A8_GROUP / 8;
#else
      for (int i = 0; i < MT; i++) pa[i] += R4D_GEMM_W4A8_GROUP / 4;
#endif
    }
  }

  // Split-K reduction and store, one row tile at a time.  The per-row activation scale is applied
  // here and only here: it is constant over n and over the group index, so it never entered the
  // k loop.
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
      // 0.0625 is the 1/16 the unpack introduced by putting the nibble in the byte's high half.
      if (n < N && m < M) C[(size_t)m * N + n] = (__hip_bfloat16)(v * Ascale[m] * 0.0625f);
    }
  }
}

void r4d_gemm_w4a8_nt_m64(long a, long ascale, long wq, long ws, long c, int M, int K, int N,
                          int WV, int SK, int MB, int NPW, int NT, long stream) {
  if (M < 1 || M > R4D_GEMM_W4A8_MAX_M)
    throw std::runtime_error("r4d gemm_w4a8_nt_m64: M must be 1..64");
  if (N % 16) throw std::runtime_error("r4d gemm_w4a8_nt_m64: N must be a multiple of 16");
  if (K % (SK * R4D_GEMM_W4A8_GROUP))
    throw std::runtime_error("r4d gemm_w4a8_nt_m64: K must be divisible by SK*group");
  if (WV * SK * WAVE32 > 1024) throw std::runtime_error("r4d gemm_w4a8_nt_m64: threads/block > 1024");
  if (MB < 1 || MB > 4) throw std::runtime_error("r4d gemm_w4a8_nt_m64: MB must be 1..4");
  if (NPW != 1 && NPW != 2 && NPW != 4 && NPW != 8)
    throw std::runtime_error("r4d gemm_w4a8_nt_m64: NPW must be 1, 2, 4 or 8");
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * WAVE32);
  size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  if (smem > 64 * 1024) throw std::runtime_error("r4d gemm_w4a8_nt_m64: WV*NPW*SK too large");
  auto Ap = (const signed char*)a;
  auto Sp = (const float*)ascale;
  auto Wp = (const unsigned*)wq;
  auto Zp = (const unsigned*)ws;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define LAUNCH(MT, NP, TT) hipLaunchKernelGGL((r4d_gemm_w4a8_nt_m64_kernel<MT, NP, TT>), grid, \
                                              block, smem, st, Ap, Sp, Wp, Zp, Cp, M, K, N, WV, SK)
#define LAUNCH_MB(NP, TT) switch (MB) { \
    case 1: LAUNCH(1, NP, TT); break; \
    case 2: LAUNCH(2, NP, TT); break; \
    case 3: LAUNCH(3, NP, TT); break; \
    case 4: LAUNCH(4, NP, TT); break; \
    default: throw std::runtime_error("r4d gemm_w4a8_nt_m64: MB unsupported"); }
#define LAUNCH_NPW(TT) switch (NPW) { \
    case 1: LAUNCH_MB(1, TT); break; \
    case 2: LAUNCH_MB(2, TT); break; \
    case 8: LAUNCH_MB(8, TT); break; \
    default: LAUNCH_MB(4, TT); break; }
  if (NT) { LAUNCH_NPW(true); } else { LAUNCH_NPW(false); }
#undef LAUNCH_NPW
#undef LAUNCH_MB
#undef LAUNCH
}

int r4d_gemm_w4a8_nt_m64_max_m(void) { return R4D_GEMM_W4A8_MAX_M; }
int r4d_gemm_w4a8_nt_m64_group(void) { return R4D_GEMM_W4A8_GROUP; }
int r4d_gemm_w4a8_nt_m64_aperm(void) { return R4D_W4A8_APERM; }
