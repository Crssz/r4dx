// w8a16_gemm.hip -- r4dx_gemm_w8a16_nt_m64 (docs/w8a16.md sections 3-4; ABI and numerics in
// include/r4dx/kernels/w8a16_gemm.h). C[M,N] bf16 = A[M,K] bf16 @ dequant(W)[N,K]^T, M <= 64.
//
// A copy of libr4d's r4d_gemm_w4a16_nt_m64_kernel (third_party/libr4d/r4d_gemm_w4a16_nt_m64.hip, whose
// header comment is the rationale for every structural choice here: f16 WMMA, per-group f32 epilogue, NPW,
// clamped row/column tiles, the fixed-order in-block split-K) with three changes (docs/w8a16.md 4.3):
//
//  1. WEIGHT LOAD. One global_load_b128 per lane per 32-K block (two k steps): dwords j = 0..3 of a lane
//     hold, byte i, code(16t + (l&15), 32kb + 16(j>>1) + 8(j&1) + 4(l>>4) + i), so dwords 2s and 2s+1 are
//     exactly elements e0..e3 and e4..e7 of k step s's B fragment (r4d_gdn_wmma.h's map). A group of G is
//     G/32 blocks.
//  2. UNPACK. Offset-binary byte b -> v_perm_b32 against 0x64646464 -> f16 0x64bb = 1024 + b (exact for
//     every b 0..255, the f16 ulp at 1024 is 1) -> one v_pk_add_f16 of (-1152, -1152) -> q = b - 128,
//     exact. 4 perm + 4 pk_add per fragment and no masks.
//  3. A OPERAND. bf16 in memory, converted in registers: a dword of two bf16 becomes the two f32
//     (d << 16, d & 0xffff0000) and one v_cvt_pk_rtz_f16_f32. Exact for |x| in [2^-14, 65504]; RTZ
//     saturates an overflow to +-65504 instead of inf; below 2^-14 an f16 subnormal truncated toward 0.
//
// The scale: f16 bits of s * 2^12 per (row, group). g_acc (the group's WMMA sum) is folded with
// acc = fmaf(g_acc, float(s16), acc); the split-K sum (fixed order) is multiplied by 2^-12 (exact) right
// before the bf16 store. Nothing is staged through scratch, there are no tickets, so two launches may
// overlap and a row's result depends on SK alone.
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>

#include "r4dx/kernels/w8a16_gemm.h"
#include "r4dx/kernels/kernels.h"  // r4dx_kernel_launch_counter_add
#include "r4d_gdn_wmma.h"          // rowGo, v8f

#include <cstdint>
#include <stdexcept>
#include <string>

namespace {

constexpr int kMaxM = 64;
constexpr int kKPB = 32;            // K per packed block: 2 k steps, 16 bytes per lane
constexpr int kScaleLog2Bias = 12;  // docs/w8a16.md 2: a FORMAT constant
constexpr int kWave = 32;

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef unsigned uint4v __attribute__((ext_vector_type(4)));

// Global address space so the loads stay global_load (see r4d_gemm_w4a16_nt_m64.hip).
typedef __attribute__((address_space(1))) const uint4v* gw4;
typedef __attribute__((address_space(1))) const unsigned long long* gu2;
typedef __attribute__((address_space(1))) const unsigned short* gus;

// Two bf16 (low half = first element) -> two f16, RTZ.
__device__ __forceinline__ unsigned bf2_to_h2(unsigned d) {
  const float lo = __uint_as_float(d << 16);
  const float hi = __uint_as_float(d & 0xffff0000u);
  union { decltype(__builtin_amdgcn_cvt_pkrtz(0.f, 0.f)) h; unsigned u; } x;
  x.h = __builtin_amdgcn_cvt_pkrtz(lo, hi);
  return x.u;
}

// The A fragment of k step u, from bf16 memory: the same two u64 loads as w4a16's fragA (elements
// 16u + 4(l>>4) + 0..3 and 16u + 8 + 4(l>>4) + 0..3 of the lane's row), converted to f16 in registers.
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

// Two dwords of four offset-binary bytes (elements e0..e3, e4..e7) -> one f16 B fragment of q = b - 128.
__device__ __forceinline__ v8h dequant8(unsigned d0, unsigned d1) {
  const unsigned K64 = 0x64646464u;
  const __half2 nb = __halves2half2(__ushort_as_half((unsigned short)0xE480u),   // -1152
                                    __ushort_as_half((unsigned short)0xE480u));
  union { unsigned u[4]; __half2 h2[4]; v8h f; } x;
  x.u[0] = __builtin_amdgcn_perm(K64, d0, 0x05010400u);  // bytes 0, 1 -> e0, e1
  x.u[1] = __builtin_amdgcn_perm(K64, d0, 0x05030402u);  // bytes 2, 3 -> e2, e3
  x.u[2] = __builtin_amdgcn_perm(K64, d1, 0x05010400u);  // e4, e5
  x.u[3] = __builtin_amdgcn_perm(K64, d1, 0x05030402u);  // e6, e7
#pragma unroll
  for (int i = 0; i < 4; i++) x.h2[i] = __hadd2(x.h2[i], nb);
  return x.f;
}

__device__ __forceinline__ v8f mmah(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

template <int G, int MT, int NPW, bool NT>
__global__ __launch_bounds__(1024) void w8a16_gemm_nt_m64_kernel(
    const unsigned short* __restrict__ A, const unsigned char* __restrict__ Wq,
    const unsigned short* __restrict__ Ws, __hip_bfloat16* __restrict__ C, int M, int K, int N, int WV,
    int SK) {
  static_assert(G % kKPB == 0, "a group is a whole number of 32-K packed blocks");
  constexpr int bpg = G / kKPB;  // packed blocks per group
  extern __shared__ float red[];  // WV*NPW*SK*256 floats
  const int tid = threadIdx.x, wave = tid / kWave, lane = tid % kWave;
  const int col_local = wave / SK, ksplit = wave % SK;
  const int nt0 = (blockIdx.x * WV + col_local) * NPW;
  const int n0 = nt0 * 16;
  const int mt0 = blockIdx.y * MT;
  const int Ks = K / SK;
  const int groups = Ks / G;
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

    const int kpb = K / kKPB;  // packed blocks per row tile
    const int nsc = K / G;     // scales per row (per row tile, x16)
    gw4 pw[NPW];
    gus ps[NPW];
    const int ntlast = N / 16 - 1;
#pragma unroll
    for (int j = 0; j < NPW; j++) {
      const int t = (nt0 + j) < ntlast ? (nt0 + j) : ntlast;  // clamped, never masked
      pw[j] = (gw4)(const uint4v*)(Wq) + (size_t)(t * (size_t)kpb + kbase / kKPB) * kWave + lane;
      ps[j] = (gus)(Ws) + (size_t)(t * (size_t)nsc + kbase / G) * 16 + (lane & 15);
    }

    for (int g = 0; g < groups; g++) {
      float sc[NPW];
#pragma unroll
      for (int j = 0; j < NPW; j++) sc[j] = __half2float(__ushort_as_half(ps[j][(size_t)g * 16]));

      v8f g_acc[NPW][MT];
#pragma unroll
      for (int b = 0; b < bpg; b++) {
        uint4v wv[NPW];
#pragma unroll
        for (int j = 0; j < NPW; j++) {
          const size_t o = (size_t)(g * bpg + b) * kWave;
          wv[j] = NT ? __builtin_nontemporal_load(&pw[j][o]) : pw[j][o];
        }
#pragma unroll
        for (int s = 0; s < 2; s++) {
          const int u = b * 2 + s;  // k step within the group
          v8h afrag[MT];
#pragma unroll
          for (int i = 0; i < MT; i++) afrag[i] = fragA_bf16(pa[i], u);
#pragma unroll
          for (int j = 0; j < NPW; j++) {
            const v8h bfrag = dequant8(wv[j][2 * s], wv[j][2 * s + 1]);
#pragma unroll
            for (int i = 0; i < MT; i++)
              g_acc[j][i] = u == 0 ? mmah(afrag[i], bfrag, (v8f)(0.f)) : mmah(afrag[i], bfrag, g_acc[j][i]);
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
  }

  // Split-K reduction and store (w4a16's, verbatim apart from the 2^-12): one row tile at a time, fixed
  // split order, so the result does not depend on wave scheduling.
  const float unbias = 1.0f / (float)(1 << kScaleLog2Bias);
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
      v = v * unbias;
      const int l = off & 31;
      const int n = (blockIdx.x * ncols + col) * 16 + (l & 15);
      const int m = (mt0 + i) * 16 + 8 * (l >> 4) + (off >> 5);
      if (n < N && m < M) C[(size_t)m * N + n] = __float2bfloat16(v);
    }
  }
}

[[noreturn]] void Fail(const std::string& what) { throw std::runtime_error("r4dx_gemm_w8a16_nt_m64: " + what); }

template <int G>
void Launch(int64_t a, int64_t wq, int64_t ws, int64_t c, int M, int K, int N, int WV, int SK, int MB, int NPW,
            int NT, int64_t stream) {
  const int ntiles = N / 16, mtiles = (M + 15) / 16;
  const int ncols = WV * NPW;
  dim3 grid((ntiles + ncols - 1) / ncols, (mtiles + MB - 1) / MB), block(WV * SK * kWave);
  const size_t smem = (size_t)ncols * SK * 256 * sizeof(float);
  auto Ap = (const unsigned short*)a;
  auto Wp = (const unsigned char*)wq;
  auto Sp = (const unsigned short*)ws;
  auto Cp = (__hip_bfloat16*)c;
  hipStream_t st = (hipStream_t)stream;
#define W8_LAUNCH(MT, NP, TT)                                                                         \
  hipLaunchKernelGGL((w8a16_gemm_nt_m64_kernel<G, MT, NP, TT>), grid, block, smem, st, Ap, Wp, Sp, Cp, M, \
                     K, N, WV, SK)
#define W8_LAUNCH_MB(NP, TT)            \
  switch (MB) {                         \
    case 1: W8_LAUNCH(1, NP, TT); break; \
    case 2: W8_LAUNCH(2, NP, TT); break; \
    case 3: W8_LAUNCH(3, NP, TT); break; \
    default: W8_LAUNCH(4, NP, TT); break; \
  }
  if (NPW == 1) {
    if (NT) { W8_LAUNCH_MB(1, true); } else { W8_LAUNCH_MB(1, false); }
  } else {
    if (NT) { W8_LAUNCH_MB(4, true); } else { W8_LAUNCH_MB(4, false); }
  }
#undef W8_LAUNCH_MB
#undef W8_LAUNCH
  const hipError_t err = hipGetLastError();
  if (err != hipSuccess) Fail(std::string("launch failed: ") + hipGetErrorString(err));
}

}  // namespace

extern "C" {

void r4dx_gemm_w8a16_nt_m64(int64_t a_bf16, int64_t wq, int64_t ws, int64_t c, int M, int K, int N, int group,
                            int WV, int SK, int MB, int NPW, int NT, int64_t stream) {
  if (M < 1 || M > kMaxM) Fail("M must be 1..64 (got " + std::to_string(M) + ")");
  if (group != 64 && group != 128) Fail("group " + std::to_string(group) + " is not instantiated (64, 128)");
  if (N < 16 || N % 16) Fail("N must be a positive multiple of 16 (got " + std::to_string(N) + ")");
  if (WV < 1 || SK < 1) Fail("WV and SK must be >= 1");
  if (K < 1 || K % (SK * group))
    Fail("K must be a positive multiple of SK*group (K " + std::to_string(K) + ", SK " + std::to_string(SK) +
         ", group " + std::to_string(group) + ")");
  if (WV * SK * kWave > 1024) Fail("WV*SK*32 > 1024 threads per block");
  if (MB < 1 || MB > 4) Fail("MB must be 1..4");
  if (NPW != 1 && NPW != 4) Fail("NPW must be 1 or 4");
  if ((size_t)WV * NPW * SK * 256 * sizeof(float) > 64 * 1024) Fail("LDS WV*NPW*SK KiB > 64");
  if (!a_bf16 || !wq || !ws || !c) Fail("null pointer");
  if (group == 128)
    Launch<128>(a_bf16, wq, ws, c, M, K, N, WV, SK, MB, NPW, NT, stream);
  else
    Launch<64>(a_bf16, wq, ws, c, M, K, N, WV, SK, MB, NPW, NT, stream);
  r4dx_kernel_launch_counter_add(1);
}

int r4dx_gemm_w8a16_has_group(int group) { return group == 64 || group == 128; }
int r4dx_gemm_w8a16_scale_log2_bias(void) { return kScaleLog2Bias; }
int r4dx_gemm_w8a16_max_m(void) { return kMaxM; }

}  // extern "C"
