// tests/kernels/tool_m256_mock_bench.hip -- standalone (host + device) TIMING MOCK for the A1 question
// "can an LDS-staged trellis GEMM at M = 256 be WMMA-bound and beat four M = 64 launches by >= 1.4x on
// the MLP pair?" (docs/trellis-kernel.md 4.7 "M8"). It is NOT a callable GEMM: KB = 4 or 5, random
// weights, no input transform, results thrown away. What it times is the K loop (plus, with kEPI, the
// stock epilogue and its M = 256 generalisation, structure only) of two kernels over the whole GPU,
// with the weights streamed from DRAM (a different copy of the weights for every launch of a batch, so
// no launch finds them in the 64 MB MALL / L2) and A in the L2:
//
//   S64  a copy of r4d_gemm_trellis_nt_m64_kernel's K loop (third_party/libr4d/r4d_gemm_trellis_nt_m64.hip
//        lines 191-352: same decode block, same loads, same one-stage schedule at MT = 4), with the
//        stock epilogue (kEPI) or one store per lane. Ablation bits switch the weight loads, the A loads
//        and the decode off one at a time (inputs made opaque, no instruction added).
//   L256 the M8 idea: RG row-group waves share every decoded B fragment through LDS. A workgroup of
//        WV * SK * RG waves; wave (rg, cb, ks) owns rows 64 rg.., column block cb, K slice ks with
//        MT = 4 row tiles and NP tile pairs, exactly S64's accumulator tile and K-slice order; per step
//        of U k-tiles each of the RG waves of a (cb, ks) group decodes ONE of the group's NP * U
//        (pair, k-tile) blocks, writes its two f16 fragments (ds_store_b128) to a double-buffered LDS
//        slot, one workgroup barrier, then every wave reads all NP * U * 2 fragments (ds_load_b128) and
//        runs NP * U * 2 * MT WMMA. Decode and its weight loads are amortised over RG * 64 rows.
//        `--set check` verifies bitwise that every accumulator equals S64's (same K order).
//
// Build (a hipcc one-liner; no CMake target):
//   hipcc.exe -O3 -std=c++17 --offload-arch=gfx1201 --rocm-path=C:/opt/rocm
//     --rocm-device-lib-path=C:/opt/rocm/lib/llvm/amdgcn/bitcode -I third_party/libr4d -ffp-contract=off
//     tests/kernels/tool_m256_mock_bench.hip -o tool_m256_mock_bench.exe
// Run (one device, the caller sets HIP_VISIBLE_DEVICES; every set is bounded):
//   tool_m256_mock_bench.exe --set main|kb5|epi|epiabl|cheap|cheap5|check [--shape gate_up|down|all]
//                            [--rounds 13] [--batch 6] [--out x.json]
// Every round times every job once (a batch of launches on different weight copies), so the printed
// speedups are medians of per-round ratios. The block-0 SCLK column is only indicative.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "r4d_fwht128.h"    // the epilogue's FWHT
#include "r4d_gdn_wmma.h"   // rowGo, v8f
#include "r4d_trellis_dq.h" // the decode, v8h

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e_)); \
  std::exit(1); } } while (0)

typedef __attribute__((address_space(1))) const unsigned char* gbyte;

template <typename T, bool NT>
__device__ __forceinline__ T Ld(gbyte base, unsigned off, int imm) {
  auto p = (__attribute__((address_space(1))) const T*)(base + off + imm);
  if constexpr (NT) return __builtin_nontemporal_load(p);
  else return *p;
}
__device__ __forceinline__ void OpaqueS(gbyte& p) {
  unsigned long long v = (unsigned long long)p;
  asm("" : "+s"(v));
  p = (gbyte)v;
}
__device__ __forceinline__ void OpaqueV(unsigned& v) { asm("" : "+v"(v)); }
__device__ __forceinline__ void OpaqueV(unsigned long long& v) { asm("" : "+v"(v)); }
__device__ __forceinline__ void OpaqueF(v8h& f) { asm("" : "+v"(f)); }
__device__ __forceinline__ v8f Mma(v8h a, v8h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}
__device__ __forceinline__ void SyncLds() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// The epilogue's operands (kEPI): svh [N] fp32, C bf16 [M][N], ws fp32 [SKG][M][N], tickets [N/128].
struct Epi {
  const float* svh;
  unsigned short* C;
  float* ws;
  unsigned* tickets;
  float out_scale;
};
__device__ __forceinline__ unsigned short Bf16Rn(float f) {
  const unsigned u = __builtin_bit_cast(unsigned, f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (unsigned short)((u >> 16) | 0x40u);
  return (unsigned short)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}
// One wave finishes one (row, 128-column group): FWHT-128 in fp32, svh, one bf16 rounding.
__device__ __forceinline__ void Finish(float (&v)[4], int lane, const float (&sv)[4],
                                       unsigned short* __restrict__ crow, int n0, float out_scale) {
  r4d_fwht128_wave(v, lane);
#pragma unroll
  for (int r = 0; r < 4; ++r) crow[n0 + lane + 32 * r] = Bf16Rn((v[r] * sv[r]) * out_scale);
}
// MODE bits
enum { kLW = 1, kLA = 2, kDEC = 4, kAC = 8, kNT = 16, kEPI = 32, kKB5 = 64, kEPI16 = 128, kNOTAIL = 256, kNOSTORE = 512 };   // kEPI16: with kEPI, L256 reduces 16 rows per pass
constexpr int kFull = kLW | kLA | kDEC;

// ---- S64: today's M = 64 K loop --------------------------------------------------------------------
template <int NP, int U, int MT, int MODE>
struct Stage {
  unsigned long long ab[NP][U];
  unsigned p[NP][U];
  unsigned w5[NP][U][5];
  v8h af[MT][U];
};

template <int NP, int U, int MT, int MODE>
__global__ __launch_bounds__(1024) void S64(const unsigned short* __restrict__ A,
                                            const unsigned* __restrict__ W, float* __restrict__ out,
                                            int K, int N, int WV, int SK, int SKG,
                                            unsigned long long* clk, Epi ep) {
  constexpr bool KB5 = MODE & kKB5;
  constexpr int BLK = KB5 ? 320 : 256;
  constexpr bool LW = MODE & kLW, LA = MODE & kLA, DEC = MODE & kDEC, AC = MODE & kAC, NT = MODE & kNT;
  extern __shared__ __attribute__((aligned(16))) float red[];
  const int tid = threadIdx.x, lane = tid & 31;
  const bool probe = clk != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0;
  if (probe) { clk[0] = clock64(); clk[1] = wall_clock64(); }
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int col_local = wave / SK, ksplit = wave - col_local * SK;
  const int Wc = WV * NP * 32;
  const int n0 = blockIdx.x * Wc;
  const int pair0 = n0 / 32 + col_local * NP;
  const int kt_total = K / 16;
  const int ktw = kt_total / (SK * SKG);
  const int kt0 = ((int)blockIdx.y * SK + ksplit) * ktw;
  const int steps = ktw / U;

  v8f acc[NP][2][MT];
#pragma unroll
  for (int p = 0; p < NP; ++p)
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
      for (int i = 0; i < MT; ++i) acc[p][f][i] = (v8f)(0.f);

  gbyte wb[NP];
#pragma unroll
  for (int p = 0; p < NP; ++p) wb[p] = (gbyte)(const unsigned char*)W + ((size_t)(pair0 + p) * kt_total + kt0) * BLK;
  gbyte abase;
  unsigned woff0 = r4d_trellis_k4_off_ab(lane), woff1 = r4d_trellis_k4_off_p(lane);
  unsigned woff5[5];
#pragma unroll
  for (int i = 0; i < 5; ++i) woff5[i] = KB5 ? r4d_trellis_k5_word(lane, i) * 4u : 0u;
  const unsigned k5sh = r4d_trellis_k5_shift(lane);
  unsigned aoff[MT];
  if constexpr (!AC) {
    abase = (gbyte)(const unsigned char*)A + (size_t)kt0 * 32;
#pragma unroll
    for (int i = 0; i < MT; ++i) aoff[i] = rowGo((unsigned)K, i * 16, 63, lane) * 2u;
  } else {
    // fragment-contiguous A: frag (mt, kt) is 512 B at ((mt * kt_total) + kt) * 512, lane * 16 inside
    abase = (gbyte)(const unsigned char*)A + (size_t)kt0 * 512;
#pragma unroll
    for (int i = 0; i < MT; ++i) aoff[i] = (unsigned)(i * kt_total) * 512u + (unsigned)lane * 16u;
  }

  auto load = [&](Stage<NP, U, MT, MODE>& s, int ahead) {
    const size_t wo = (size_t)ahead * (U * BLK), ao = (size_t)ahead * (U * (AC ? 512 : 32));
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      gbyte b = wb[p] + wo;
      OpaqueS(b);
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if constexpr (LW && KB5) {
#pragma unroll
          for (int i = 0; i < 5; ++i) s.w5[p][u][i] = Ld<unsigned, NT>(b, woff5[i], u * BLK);
        } else if constexpr (KB5) {
#pragma unroll
          for (int i = 0; i < 5; ++i) { s.w5[p][u][i] = woff0 ^ (woff1 + (unsigned)ahead * 977u + p * 31 + u + i * 7); OpaqueV(s.w5[p][u][i]); }
        } else if constexpr (LW) {
          s.ab[p][u] = Ld<unsigned long long, NT>(b, woff0, u * BLK);
          s.p[p][u] = Ld<unsigned, NT>(b, woff1, u * BLK);
        } else {
          s.ab[p][u] = ((unsigned long long)woff0 << 32) | (woff1 + (unsigned)ahead * 977u + p * 31 + u);
          s.p[p][u] = woff0 ^ (woff1 + (unsigned)ahead) ^ 0x5bd1e995u;
          OpaqueV(s.ab[p][u]);
          OpaqueV(s.p[p][u]);
        }
      }
    }
    gbyte a = abase + ao;
    OpaqueS(a);
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int u = 0; u < U; ++u) {
        if constexpr (LA) {
          union { unsigned long long w[2]; v8h f; } x;
          if constexpr (!AC) {
            x.w[0] = Ld<unsigned long long, false>(a, aoff[i], u * 32);
            x.w[1] = Ld<unsigned long long, false>(a, aoff[i], u * 32 + 16);
          } else {
            // one b128 per lane
            typedef unsigned v4u_ __attribute__((ext_vector_type(4)));
            const v4u_ q = Ld<v4u_, false>(a, aoff[i], u * 512);
            x.f = __builtin_bit_cast(v8h, q);
          }
          s.af[i][u] = x.f;
        } else {
          v8h f = __builtin_bit_cast(v8h, (r4d_u32x4){0x3C00u ^ (unsigned)lane, 0x3C003C00u, (unsigned)(i * 5 + u), 0x3800u});
          OpaqueF(f);
          s.af[i][u] = f;
        }
      }
  };
  auto advance = [&](int steps_done) {
#pragma unroll
    for (int p = 0; p < NP; ++p) {
      wb[p] += (size_t)steps_done * (U * BLK);
      OpaqueS(wb[p]);
    }
    abase += (size_t)steps_done * (U * (AC ? 512 : 32));
    OpaqueS(abase);
    OpaqueV(woff0);
    OpaqueV(woff1);
    if constexpr (KB5) {
#pragma unroll
      for (int i = 0; i < 5; ++i) OpaqueV(woff5[i]);
    }
#pragma unroll
    for (int i = 0; i < MT; ++i) OpaqueV(aoff[i]);
  };
  auto compute = [&](Stage<NP, U, MT, MODE>& s) {
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        v8h f0, f1;
        if constexpr (DEC && KB5) {
          r4d_trellis_k5_decode(s.w5[p][u], k5sh, f0, f1);
        } else if constexpr (KB5) {
          f0 = __builtin_bit_cast(v8h, (r4d_u32x4){s.w5[p][u][0], s.w5[p][u][1], s.w5[p][u][2], s.w5[p][u][3]});
          f1 = __builtin_bit_cast(v8h, (r4d_u32x4){s.w5[p][u][4], s.w5[p][u][1] ^ 1u, 0x3C003C00u, 0x3800u});
          OpaqueF(f0);
          OpaqueF(f1);
        } else if constexpr (DEC) {
          r4d_trellis_k4_decode(s.p[p][u], (unsigned)s.ab[p][u], (unsigned)(s.ab[p][u] >> 32), f0, f1);
        } else {
          f0 = __builtin_bit_cast(v8h, (r4d_u32x4){(unsigned)s.ab[p][u], s.p[p][u], 0x3C003C00u, 0x3800u});
          f1 = __builtin_bit_cast(v8h, (r4d_u32x4){(unsigned)(s.ab[p][u] >> 32), s.p[p][u] ^ 1u, 0x3C003C00u, 0x3800u});
          OpaqueF(f0);
          OpaqueF(f1);
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          acc[p][0][i] = Mma(s.af[i][u], f0, acc[p][0][i]);
          acc[p][1][i] = Mma(s.af[i][u], f1, acc[p][1][i]);
        }
      }
  };
  for (int st = 0; st < steps; ++st) {
    Stage<NP, U, MT, MODE> s0;
    load(s0, 0);
    compute(s0);
    advance(1);
  }
  if constexpr (MODE & kEPI) {
    // the stock epilogue (r4d_gemm_trellis_nt_m64.hip 354-453) at M = 64, full = 1
    constexpr int M = 16 * MT;
    const int nwaves = WV * SK, nthreads = nwaves * 32;
    const bool split = SKG > 1 || Wc < 128;
    const int span = Wc > 128 ? Wc : 128;
    const int g0 = n0 / span * span;
    const int groups = span / 128;
    const int gw = wave % groups, rstep = nwaves / groups;
    float sv[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) sv[r] = ep.svh[g0 + gw * 128 + lane + 32 * r];
    float* dst = split ? ep.ws + (size_t)blockIdx.y * M * N : nullptr;
    const int t = (lane >> 3) & 1, c = lane & 7, hh_lane = lane >> 4;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        const int mbase = i * 16 + 8 * hh;
        SyncLds();
        if (hh_lane == hh) {
#pragma unroll
          for (int p = 0; p < NP; ++p)
#pragma unroll
            for (int f = 0; f < 2; ++f) {
              const int col = (col_local * NP + p) * 32 + 16 * t + 8 * f + c;
#pragma unroll
              for (int e = 0; e < 8; ++e) red[(ksplit * 8 + e) * Wc + col] = acc[p][f][i][e];
            }
        }
        SyncLds();
        const bool keep = !split;
        for (int idx = tid; idx < 8 * Wc; idx += nthreads) {
          const int r = idx / Wc, n = idx - r * Wc, m = mbase + r;
          float v = 0.f;
          for (int s = 0; s < SK; ++s) v += red[(s * 8 + r) * Wc + n];
          if (keep) red[r * Wc + n] = v;
          else dst[(size_t)m * N + n0 + n] = v;
        }
        if (keep) {
          SyncLds();
          for (int r = wave / groups; r < 8; r += rstep) {
            float v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) v[q] = red[r * Wc + gw * 128 + lane + 32 * q];
            Finish(v, lane, sv, ep.C + (size_t)(mbase + r) * N, n0 + gw * 128, ep.out_scale);
          }
        }
      }
    }
    if (split) {
      const unsigned contributors = (unsigned)(SKG * (span / Wc));
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        const unsigned old = atomicAdd(&ep.tickets[g0 / 128], 1u);
        const bool last = old == contributors - 1;
        if (last) ep.tickets[g0 / 128] = 0u;
        red[0] = last ? 1.f : 0.f;
      }
      __syncthreads();
      if (red[0] != 0.f) {
        __threadfence();
        const int gn = g0 + gw * 128;
        for (int m = wave / groups; m < M; m += rstep) {
          float v[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int n = gn + lane + 32 * q;
            float x = 0.f;
            for (int y = 0; y < SKG; ++y) x += ep.ws[((size_t)y * M + m) * N + n];
            v[q] = x;
          }
          Finish(v, lane, sv, ep.C + (size_t)m * N, gn, ep.out_scale);
        }
      }
    }
  } else {
    float sum = 0.f;
#pragma unroll
    for (int p = 0; p < NP; ++p)
#pragma unroll
      for (int f = 0; f < 2; ++f)
#pragma unroll
        for (int i = 0; i < MT; ++i) sum += acc[p][f][i][0] + acc[p][f][i][7];
    out[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + tid] = sum;
  }
  if (probe) { clk[2] = clock64(); clk[3] = wall_clock64(); }
}

// ---- L256: RG row-group waves share each decoded fragment through LDS ------------------------------
template <int NP, int U, int RG, int MODE>
__global__ __launch_bounds__(1024) void L256(const unsigned short* __restrict__ A,
                                             const unsigned* __restrict__ W, float* __restrict__ out,
                                             int K, int N, int WV, int SK, int SKG,
                                             unsigned long long* clk, Epi ep) {
  constexpr bool KB5 = MODE & kKB5;
  constexpr int BLK = KB5 ? 320 : 256, MT = 4;
  static_assert(NP * U == RG, "each wave of a group decodes exactly one block per step");
  constexpr bool LW = MODE & kLW, LA = MODE & kLA, DEC = MODE & kDEC, AC = MODE & kAC, NT = MODE & kNT;
  extern __shared__ __attribute__((aligned(16))) unsigned char lds[];
  const int tid = threadIdx.x, lane = tid & 31;
  const bool probe = clk != nullptr && blockIdx.x == 0 && blockIdx.y == 0 && tid == 0;
  if (probe) { clk[0] = clock64(); clk[1] = wall_clock64(); }
  const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
  const int rg = wave % RG;
  const int cb = (wave / RG) % WV;
  const int ks = wave / (RG * WV);
  const int Wc = WV * NP * 32;
  const int n0 = blockIdx.x * Wc;
  const int pair0 = n0 / 32 + cb * NP;
  const int kt_total = K / 16;
  const int ktw = kt_total / (SK * SKG);
  const int kt0 = ((int)blockIdx.y * SK + ks) * ktw;
  const int steps = ktw / U;
  const int group = ks * WV + cb;
  const int ngroups = WV * SK;
  const int p_d = rg % NP, u_d = rg / NP;             // the block this wave decodes each step
  const unsigned slot_bytes = NP * U * 1024;          // one group's staging per buffer
  const unsigned buf_bytes = ngroups * slot_bytes;
  const unsigned lds_w = group * slot_bytes + rg * 1024 + lane * 16;   // own block's F0 (F1 at +512)
  const unsigned lds_r = group * slot_bytes + lane * 16;               // block j's F0 at + j * 1024

  v8f acc[NP][2][MT];
#pragma unroll
  for (int p = 0; p < NP; ++p)
#pragma unroll
    for (int f = 0; f < 2; ++f)
#pragma unroll
      for (int i = 0; i < MT; ++i) acc[p][f][i] = (v8f)(0.f);

  gbyte wb = (gbyte)(const unsigned char*)W + ((size_t)(pair0 + p_d) * kt_total + kt0 + u_d) * BLK;
  unsigned woff0 = r4d_trellis_k4_off_ab(lane), woff1 = r4d_trellis_k4_off_p(lane);
  unsigned woff5[5];
#pragma unroll
  for (int i = 0; i < 5; ++i) woff5[i] = KB5 ? r4d_trellis_k5_word(lane, i) * 4u : 0u;
  const unsigned k5sh = r4d_trellis_k5_shift(lane);
  gbyte abase;
  unsigned aoff[MT];
  if constexpr (!AC) {
    abase = (gbyte)(const unsigned char*)A + (size_t)kt0 * 32;
#pragma unroll
    for (int i = 0; i < MT; ++i) aoff[i] = rowGo((unsigned)K, rg * 64 + i * 16, RG * 64 - 1, lane) * 2u;
  } else {
    abase = (gbyte)(const unsigned char*)A + (size_t)kt0 * 512;
#pragma unroll
    for (int i = 0; i < MT; ++i) aoff[i] = (unsigned)((rg * 4 + i) * kt_total) * 512u + (unsigned)lane * 16u;
  }

  unsigned long long ab;
  unsigned pw;
  unsigned w5n[5];
  auto load_w = [&](int ahead) {
    gbyte b = wb + (size_t)ahead * (U * BLK);
    OpaqueS(b);
    if constexpr (KB5 && LW) {
#pragma unroll
      for (int i = 0; i < 5; ++i) w5n[i] = Ld<unsigned, NT>(b, woff5[i], 0);
    } else if constexpr (KB5) {
#pragma unroll
      for (int i = 0; i < 5; ++i) { w5n[i] = woff0 ^ (woff1 + (unsigned)ahead * 977u + i * 7); OpaqueV(w5n[i]); }
    } else if constexpr (LW) {
      ab = Ld<unsigned long long, NT>(b, woff0, 0);
      pw = Ld<unsigned, NT>(b, woff1, 0);
    } else {
      ab = ((unsigned long long)woff0 << 32) | (woff1 + (unsigned)ahead * 977u);
      pw = woff0 ^ (woff1 + (unsigned)ahead) ^ 0x5bd1e995u;
      OpaqueV(ab);
      OpaqueV(pw);
    }
  };
  load_w(0);
  int buf = 0;
  for (int st = 0; st < steps; ++st) {
    // ---- phase D: A loads for this step, decode own block, publish it
    v8h af[U][MT];
    {
      gbyte a = abase + (size_t)st * (U * (AC ? 512 : 32));
      OpaqueS(a);
#pragma unroll
      for (int u = 0; u < U; ++u)
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          if constexpr (LA) {
            union { unsigned long long w[2]; v8h f; } x;
            if constexpr (!AC) {
              x.w[0] = Ld<unsigned long long, false>(a, aoff[i], u * 32);
              x.w[1] = Ld<unsigned long long, false>(a, aoff[i], u * 32 + 16);
            } else {
              typedef unsigned v4u_ __attribute__((ext_vector_type(4)));
              const v4u_ q = Ld<v4u_, false>(a, aoff[i], u * 512);
              x.f = __builtin_bit_cast(v8h, q);
            }
            af[u][i] = x.f;
          } else {
            v8h f = __builtin_bit_cast(v8h, (r4d_u32x4){0x3C00u ^ (unsigned)lane, 0x3C003C00u, (unsigned)(i * 5 + u), 0x3800u});
            OpaqueF(f);
            af[u][i] = f;
          }
        }
    }
    const unsigned long long ab_c = ab;
    const unsigned pw_c = pw;
    unsigned c5[5];
#pragma unroll
    for (int i = 0; i < 5; ++i) c5[i] = w5n[i];
    load_w(st + 1 < steps ? 1 : 0);                    // next step's words (wb is this step's base; clamped)
    __builtin_amdgcn_sched_barrier(0);
    v8h f0, f1;
    if constexpr (KB5 && DEC) {
      r4d_trellis_k5_decode(c5, k5sh, f0, f1);
    } else if constexpr (KB5) {
      f0 = __builtin_bit_cast(v8h, (r4d_u32x4){c5[0], c5[1], c5[2], c5[3]});
      f1 = __builtin_bit_cast(v8h, (r4d_u32x4){c5[4], c5[1] ^ 1u, 0x3C003C00u, 0x3800u});
      OpaqueF(f0);
      OpaqueF(f1);
    } else if constexpr (DEC) {
      r4d_trellis_k4_decode(pw_c, (unsigned)ab_c, (unsigned)(ab_c >> 32), f0, f1);
    } else {
      f0 = __builtin_bit_cast(v8h, (r4d_u32x4){(unsigned)ab_c, pw_c, 0x3C003C00u, 0x3800u});
      f1 = __builtin_bit_cast(v8h, (r4d_u32x4){(unsigned)(ab_c >> 32), pw_c ^ 1u, 0x3C003C00u, 0x3800u});
      OpaqueF(f0);
      OpaqueF(f1);
    }
    *(v8h*)(lds + buf * buf_bytes + lds_w) = f0;
    *(v8h*)(lds + buf * buf_bytes + lds_w + 512) = f1;
    __builtin_amdgcn_sched_barrier(0);
    SyncLds();
    // ---- phase W: read every block's fragments of the group and multiply
#pragma unroll
    for (int u = 0; u < U; ++u)
#pragma unroll
      for (int p = 0; p < NP; ++p) {
        const unsigned rd = buf * buf_bytes + lds_r + (u * NP + p) * 1024;
        const v8h b0 = *(const v8h*)(lds + rd);
        const v8h b1 = *(const v8h*)(lds + rd + 512);
#pragma unroll
        for (int i = 0; i < MT; ++i) {
          acc[p][0][i] = Mma(af[u][i], b0, acc[p][0][i]);
          acc[p][1][i] = Mma(af[u][i], b1, acc[p][1][i]);
        }
      }
    buf ^= 1;
    wb += (size_t)U * BLK;
    OpaqueS(wb);
    OpaqueV(woff0);
    OpaqueV(woff1);
    if constexpr (KB5) {
#pragma unroll
      for (int i = 0; i < 5; ++i) OpaqueV(woff5[i]);
    }
#pragma unroll
    for (int i = 0; i < MT; ++i) OpaqueV(aoff[i]);
    __builtin_amdgcn_sched_barrier(0);
  }
  if constexpr ((MODE & kEPI) && (MODE & kEPI16)) {
    // L256 epilogue, 16 rows per pass (all lanes write, half the passes and barriers). LDS
    // RG * SK * 16 * Wc * 4 B (Wc <= 64 only).
    constexpr int M = RG * 64;
    float* red = reinterpret_cast<float*>(lds);
    const int nwaves_ = WV * SK * RG, nthreads = nwaves_ * 32;
    const bool split = SKG > 1 || Wc < 128;
    const int span = Wc > 128 ? Wc : 128;
    const int g0 = n0 / span * span;
    float sv[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) sv[r] = ep.svh[g0 + lane + 32 * r];
    float* dst = split ? ep.ws + (size_t)blockIdx.y * M * N : nullptr;
    const int t = (lane >> 3) & 1, c = lane & 7, hh_lane = lane >> 4;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
      SyncLds();
#pragma unroll
      for (int p = 0; p < NP; ++p)
#pragma unroll
        for (int f = 0; f < 2; ++f) {
          const int col = (cb * NP + p) * 32 + 16 * t + 8 * f + c;
#pragma unroll
          for (int e = 0; e < 8; ++e) red[(((rg * SK + ks) * 16) + hh_lane * 8 + e) * Wc + col] = acc[p][f][i][e];
        }
      SyncLds();
      const bool keep = !split;
      for (int idx = tid; idx < RG * 16 * Wc; idx += nthreads) {
        const int rgi = idx / (16 * Wc), rem = idx - rgi * 16 * Wc, r = rem / Wc, n = rem - r * Wc;
        const int m = rgi * 64 + i * 16 + r;
        float v = 0.f;
        for (int s = 0; s < SK; ++s) v += red[(((rgi * SK + s) * 16) + r) * Wc + n];
        if (keep) red[((rgi * SK) * 16 + r) * Wc + n] = v;
        else dst[(size_t)m * N + n0 + n] = v;
      }
      if (keep) {
        SyncLds();
        for (int task = wave; task < 16 * RG; task += nwaves_) {
          const int fr = task % 16, frg = task / 16;
          float v[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) v[q] = red[((frg * SK) * 16 + fr) * Wc + lane + 32 * q];
          Finish(v, lane, sv, ep.C + (size_t)(frg * 64 + i * 16 + fr) * N, n0, ep.out_scale);
        }
      }
    }
    if (split) {
      const unsigned contributors = (unsigned)(SKG * (span / Wc));
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        const unsigned old = atomicAdd(&ep.tickets[g0 / 128], 1u);
        const bool last = old == contributors - 1;
        if (last) ep.tickets[g0 / 128] = 0u;
        red[0] = last ? 1.f : 0.f;
      }
      __syncthreads();
      if (red[0] != 0.f) {
        __threadfence();
        for (int m = wave; m < M; m += nwaves_) {
          float v[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int n = g0 + lane + 32 * q;
            float x = 0.f;
            for (int y = 0; y < SKG; ++y) x += ep.ws[((size_t)y * M + m) * N + n];
            v[q] = x;
          }
          Finish(v, lane, sv, ep.C + (size_t)m * N, g0, ep.out_scale);
        }
      }
    }
  } else  if constexpr (MODE & kEPI) {
    // The stock epilogue generalised to RG row groups: every pass reduces 8 rows of each row group,
    // and the RG * 8 finishing (row, group) tasks of a pass go round-robin over the waves. A block
    // narrower than 128 columns or with SKG > 1 is "split": partials go to ws and the last block of
    // the 128-group to arrive (a ticket) sums them and finishes, as the stock kernel does.
    static_assert(RG >= 1, "");
    constexpr int M = RG * 64;
    float* red = reinterpret_cast<float*>(lds);
    const int nwaves_ = WV * SK * RG, nthreads = nwaves_ * 32;
    const bool split = SKG > 1 || Wc < 128;
    const int span = Wc > 128 ? Wc : 128;
    const int g0 = n0 / span * span;
    float sv[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) sv[r] = ep.svh[g0 + lane + 32 * r];
    float* dst = split ? ep.ws + (size_t)blockIdx.y * M * N : nullptr;
    const int t = (lane >> 3) & 1, c = lane & 7, hh_lane = lane >> 4;
#pragma unroll
    for (int i = 0; i < MT; ++i) {
#pragma unroll
      for (int hh = 0; hh < 2; ++hh) {
        SyncLds();
        if (hh_lane == hh) {
#pragma unroll
          for (int p = 0; p < NP; ++p)
#pragma unroll
            for (int f = 0; f < 2; ++f) {
              const int col = (cb * NP + p) * 32 + 16 * t + 8 * f + c;
#pragma unroll
              for (int e = 0; e < 8; ++e) red[(((rg * SK + ks) * 8) + e) * Wc + col] = acc[p][f][i][e];
            }
        }
        SyncLds();
        const bool keep = !split;
        for (int idx = tid; idx < RG * 8 * Wc; idx += nthreads) {
          const int rgi = idx / (8 * Wc), rem = idx - rgi * 8 * Wc, r = rem / Wc, n = rem - r * Wc;
          const int m = rgi * 64 + i * 16 + 8 * hh + r;
          float v = 0.f;
          for (int s = 0; s < SK; ++s) v += red[(((rgi * SK + s) * 8) + r) * Wc + n];
          if (keep) red[((rgi * SK) * 8 + r) * Wc + n] = v;
          else if constexpr (MODE & kNOSTORE) asm volatile("" :: "v"(v));
          else dst[(size_t)m * N + n0 + n] = v;
        }
        if (keep) {
          SyncLds();
          for (int task = wave; task < 8 * RG; task += nwaves_) {
            const int fr = task % 8, frg = task / 8;
            float v[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) v[q] = red[((frg * SK) * 8 + fr) * Wc + lane + 32 * q];
            Finish(v, lane, sv, ep.C + (size_t)(frg * 64 + i * 16 + 8 * hh + fr) * N, n0, ep.out_scale);
          }
        }
      }
    }
    if (split && !(MODE & kNOTAIL)) {
      const unsigned contributors = (unsigned)(SKG * (span / Wc));
      __threadfence();
      __syncthreads();
      if (tid == 0) {
        const unsigned old = atomicAdd(&ep.tickets[g0 / 128], 1u);
        const bool last = old == contributors - 1;
        if (last) ep.tickets[g0 / 128] = 0u;
        red[0] = last ? 1.f : 0.f;
      }
      __syncthreads();
      if (red[0] != 0.f) {
        __threadfence();
        for (int m = wave; m < M; m += nwaves_) {
          float v[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            const int n = g0 + lane + 32 * q;
            float x = 0.f;
            for (int y = 0; y < SKG; ++y) x += ep.ws[((size_t)y * M + m) * N + n];
            v[q] = x;
          }
          Finish(v, lane, sv, ep.C + (size_t)m * N, g0, ep.out_scale);
        }
      }
    }
  } else {
    float sum = 0.f;
#pragma unroll
    for (int p = 0; p < NP; ++p)
#pragma unroll
      for (int f = 0; f < 2; ++f)
#pragma unroll
        for (int i = 0; i < MT; ++i) sum += acc[p][f][i][0] + acc[p][f][i][7];
    out[((size_t)blockIdx.y * gridDim.x + blockIdx.x) * blockDim.x + tid] = sum;
  }
  if (probe) { clk[2] = clock64(); clk[3] = wall_clock64(); }
}

// ---- fills ----------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t Hash32(uint64_t i, uint32_t seed) {
  uint32_t h = (uint32_t)i * 0x9E3779B9u ^ (uint32_t)(i >> 32) * 0x85EBCA6Bu ^ seed;
  h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
  return h;
}
__global__ void FillW(uint32_t* p, int64_t n, uint32_t seed) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) p[i] = Hash32(i, seed);
}
__global__ void FillA(unsigned short* p, int64_t n, uint32_t seed) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
    p[i] = (unsigned short)(0x3800u | (Hash32(i, seed) & 0x3FFu));   // f16 in [0.5, 1)
}

// ---- host -----------------------------------------------------------------------------------------
#include <functional>

struct Bufs {
  unsigned char* w = nullptr;   // batch copies of the weights
  size_t w_bytes_each = 0;
  unsigned short* a = nullptr;
  float* out = nullptr;
  unsigned long long* clk = nullptr;
  float* svh = nullptr;
  unsigned short* c = nullptr;
  float* ws = nullptr;
  unsigned* tickets = nullptr;
  int batch = 0;
};
static Epi EpiOf(const Bufs& b) { return Epi{b.svh, b.c, b.ws, b.tickets, 0.0883883f}; }

struct Job {
  std::string name;
  int N, K, M;                 // rows computed per launch
  bool epi;                    // includes the epilogue
  int waves_per_wg, wgs;
  std::function<void(const unsigned char*)> launch;
  std::vector<double> t;       // us per launch, one entry per round
  double clk_mhz = 0;
};

static double Med(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

template <int NP, int U, int MT, int MODE>
static void AddS64(std::vector<Job>& jobs, const char* tag, Bufs& b, int N, int K, int WV, int SK, int SKG) {
  const int Wc = WV * NP * 32;
  char nm[160];
  std::snprintf(nm, sizeof nm, "S64  WV%d NP%d SK%d SKG%d U%d MT%d %s", WV, NP, SK, SKG, U, MT, tag);
  if ((K / 16) % (SK * SKG * U) || N % Wc) { std::printf("skip %s (illegal)\n", nm); return; }
  const int smem = (MODE & kEPI) ? SK * Wc * 32 : 0;
  Job j;
  j.name = nm; j.N = N; j.K = K; j.M = 16 * MT; j.epi = (MODE & kEPI) != 0;
  j.waves_per_wg = WV * SK; j.wgs = (N / Wc) * SKG;
  Bufs bb = b;
  j.launch = [=](const unsigned char* w) {
    hipLaunchKernelGGL((S64<NP, U, MT, MODE>), dim3(N / Wc, SKG), dim3(WV * SK * 32), smem, 0, bb.a,
                       (const unsigned*)w, bb.out, K, N, WV, SK, SKG, bb.clk, EpiOf(bb));
  };
  jobs.push_back(j);
}

template <int NP, int U, int RG, int MODE>
static void AddL256(std::vector<Job>& jobs, const char* tag, Bufs& b, int N, int K, int WV, int SK, int SKG) {
  const int Wc = WV * NP * 32;
  char nm[160];
  std::snprintf(nm, sizeof nm, "L%-3d WV%d NP%d SK%d SKG%d U%d RG%d %s", RG * 64, WV, NP, SK, SKG, U, RG, tag);
  if ((K / 16) % (SK * SKG * U) || N % Wc || WV * SK * RG > 32) { std::printf("skip %s (illegal)\n", nm); return; }
  int smem = 2 * (WV * SK) * (NP * U) * 1024;
  if (MODE & kEPI) {
    if (Wc != 128 && Wc != 64 && Wc != 32) { std::printf("skip %s (epilogue needs Wc 32/64/128)\n", nm); return; }
    smem = std::max(smem, RG * SK * ((MODE & kEPI16) ? 16 : 8) * Wc * 4);
  }
  if (smem > 64 * 1024) { std::printf("skip %s (LDS %d)\n", nm, smem); return; }
  CK(hipFuncSetAttribute(reinterpret_cast<const void*>(L256<NP, U, RG, MODE>), hipFuncAttributeMaxDynamicSharedMemorySize, smem));
  Job j;
  j.name = nm; j.N = N; j.K = K; j.M = 64 * RG; j.epi = (MODE & kEPI) != 0;
  j.waves_per_wg = WV * SK * RG; j.wgs = (N / Wc) * SKG;
  Bufs bb = b;
  j.launch = [=](const unsigned char* w) {
    hipLaunchKernelGGL((L256<NP, U, RG, MODE>), dim3(N / Wc, SKG), dim3(WV * SK * RG * 32), smem, 0, bb.a,
                       (const unsigned*)w, bb.out, K, N, WV, SK, SKG, bb.clk, EpiOf(bb));
  };
  jobs.push_back(j);
}

// Round-robin timing: every round times every job once (a batch of launches on different weight
// copies), so slow drifts of the power state hit all jobs alike and the per-round ratios are fair.
static void RunJobs(std::vector<Job>& jobs, Bufs& b, int rounds) {
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  for (Job& j : jobs)
    for (int i = 0; i < 2; ++i) j.launch(b.w);
  CK(hipDeviceSynchronize());
  for (int r = 0; r < rounds; ++r) {
    for (Job& j : jobs) {
      CK(hipEventRecord(e0, 0));
      for (int i = 0; i < b.batch; ++i) j.launch(b.w + (size_t)i * b.w_bytes_each);
      CK(hipEventRecord(e1, 0));
      CK(hipEventSynchronize(e1));
      float ms = 0;
      CK(hipEventElapsedTime(&ms, e0, e1));
      j.t.push_back(1000.0 * ms / b.batch);
      unsigned long long c[4];
      CK(hipMemcpy(c, b.clk, sizeof c, hipMemcpyDeviceToHost));
      if (c[1] > 0 && c[3] > c[1]) j.clk_mhz = (double)(c[2] - c[0]) / (double)(c[3] - c[1]) * 100.0;
    }
  }
  CK(hipEventDestroy(e0));
  CK(hipEventDestroy(e1));
}

static void Report(const std::vector<Job>& jobs, FILE* json, bool& first) {
  // baseline per (epi flag): the fastest M = 64 job of that flag (by median)
  const Job* base[2] = {nullptr, nullptr};
  for (const Job& j : jobs) {
    if (j.M != 64) continue;
    const int e = j.epi ? 1 : 0;
    if (!base[e] || Med(j.t) < Med(base[e]->t)) base[e] = &j;
  }
  for (const Job& j : jobs) {
    const double med = Med(j.t);
    const double mn = *std::min_element(j.t.begin(), j.t.end());
    const double mx = *std::max_element(j.t.begin(), j.t.end());
    double sp = 0, sp_lo = 0, sp_hi = 0;
    const Job* bs = base[j.epi ? 1 : 0];
    if (bs && (j.M > 64 || (&j != bs && j.M == 64))) {
      std::vector<double> r;
      for (size_t i = 0; i < j.t.size(); ++i) r.push_back((j.M / 64.0) * bs->t[i] / j.t[i]);
      sp = Med(r);
      sp_lo = *std::min_element(r.begin(), r.end());
      sp_hi = *std::max_element(r.begin(), r.end());
    }
    std::printf("%-46s M=%-4d %s %9.1f us [%8.1f..%8.1f] w/WG %2d WGs %4d blk0 %4.0f MHz", j.name.c_str(), j.M,
                j.epi ? "epi" : "   ", med, mn, mx, j.waves_per_wg, j.wgs, j.clk_mhz);
    if (sp > 0) std::printf("  x%.3f vs %d x %s [%.3f..%.3f]", sp, j.M / 64, bs->name.substr(5, 22).c_str(), sp_lo, sp_hi);
    std::printf("\n");
    if (json) {
      std::fprintf(json, "%s {\"name\":\"%s\",\"N\":%d,\"K\":%d,\"M\":%d,\"epilogue\":%d,\"us_median\":%.2f,\"us_min\":%.2f,\"us_max\":%.2f,"
                         "\"waves_per_wg\":%d,\"wgs\":%d,\"blk0_sclk_mhz\":%.0f,\"speedup_vs_M64_multiples\":%.4f,\"speedup_min\":%.4f,\"speedup_max\":%.4f}\n",
                   first ? "" : ",", j.name.c_str(), j.N, j.K, j.M, j.epi ? 1 : 0, med, mn, mx, j.waves_per_wg, j.wgs, j.clk_mhz, sp, sp_lo, sp_hi);
      first = false;
    }
  }
}

// The comparison set for one KB: today's M = 64 (the stock picks, plus the WV1 variant) against
// LDS-staged M = 256 in the 16-wave shapes that keep the stock tuned SK (and SKG) and in the others.
template <int F>
static void AddMain(std::vector<Job>& jobs, const char* kb, Bufs& b, int N, int K, int skg) {
  char tag[64];
  auto T = [&](const char* s) { std::snprintf(tag, sizeof tag, "%s %s", kb, s); return (const char*)tag; };
  // whole kernels (loop + epilogue)
  AddS64<2, 1, 4, kFull | kEPI | F>(jobs, T("full+epi"), b, N, K, 2, 4, skg);
  AddS64<2, 1, 4, kFull | kEPI | F>(jobs, T("full+epi"), b, N, K, 1, 4, skg);
  AddS64<2, 1, 4, kFull | kEPI | kNT | F>(jobs, T("full+epi NT"), b, N, K, 2, 4, skg);
  AddL256<2, 2, 4, kFull | kEPI | F>(jobs, T("full+epi"), b, N, K, 2, 4, skg);   // 32 waves, Wc 128, SK 4
  AddL256<2, 2, 4, kFull | kEPI | F>(jobs, T("full+epi"), b, N, K, 1, 4, skg);   // 16 waves, Wc 64 (split), SK 4
  AddL256<1, 4, 4, kFull | kEPI | F>(jobs, T("full+epi"), b, N, K, 2, 2, skg);   // 16 waves, Wc 64, SK 2
  AddL256<2, 2, 4, kFull | kEPI | F>(jobs, T("full+epi"), b, N, K, 2, 2, skg);   // 16 waves, Wc 128, SK 2
  AddL256<1, 4, 4, kFull | kEPI | F>(jobs, T("full+epi"), b, N, K, 4, 2, skg);   // 32 waves, Wc 128, SK 2
  AddL256<1, 4, 4, kFull | kEPI | F>(jobs, T("full+epi"), b, N, K, 2, 4, skg);   // 32 waves, Wc 64 (split), SK 4
  AddL256<1, 4, 4, kFull | kEPI | F>(jobs, T("full+epi"), b, N, K, 1, 4, skg);   // 16 waves, Wc 32 (split), SK 4
  AddL256<2, 2, 4, kFull | kEPI | kEPI16 | F>(jobs, T("full+epi16"), b, N, K, 1, 4, skg);
  AddL256<1, 4, 4, kFull | kEPI | kEPI16 | F>(jobs, T("full+epi16"), b, N, K, 2, 4, skg);
  AddL256<1, 4, 4, kFull | kEPI | kEPI16 | F>(jobs, T("full+epi16"), b, N, K, 1, 4, skg);
  // loop only
  AddS64<2, 1, 4, kFull | F>(jobs, T("loop"), b, N, K, 2, 4, skg);
  AddL256<2, 2, 4, kFull | F>(jobs, T("loop"), b, N, K, 1, 4, skg);
  AddL256<2, 2, 4, kFull | F>(jobs, T("loop"), b, N, K, 2, 4, skg);
}
int main(int argc, char** argv) {
  std::string outp, shape = "all", set = "main";
  int rounds = 11, batch = 6;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
    else if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) batch = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--shape") && i + 1 < argc) shape = argv[++i];
    else if (!std::strcmp(argv[i], "--set") && i + 1 < argc) set = argv[++i];
  }
  hipDeviceProp_t prop;
  CK(hipGetDeviceProperties(&prop, 0));
  std::printf("device %s, %d WGPs; set %s, rounds %d, batch %d\n", prop.name, prop.multiProcessorCount, set.c_str(), rounds, batch);
  if (set == "check") {
    // Data-movement check of the LDS staging: every accumulator of L256 must equal, bit for bit, the
    // one the same wave computes in S64 (same weights, same A rows, same K order).
    const int N = 34816, K = 5120;
    Bufs b;
    b.batch = 1;
    b.w_bytes_each = (size_t)N * K / 2;
    CK(hipMalloc(&b.w, b.w_bytes_each));
    CK(hipMalloc(&b.a, (size_t)256 * K * 2 + 4096));
    CK(hipMalloc(&b.out, (size_t)5 * 1024 * 1024 * 4));
    CK(hipMalloc(&b.clk, 64));
    hipLaunchKernelGGL(FillW, dim3(8192), dim3(256), 0, 0, (uint32_t*)b.w, (int64_t)(b.w_bytes_each / 4), 4242u);
    hipLaunchKernelGGL(FillA, dim3(2048), dim3(256), 0, 0, b.a, (int64_t)256 * K, 31337u);
    CK(hipDeviceSynchronize());
    Epi ep{nullptr, nullptr, nullptr, nullptr, 1.f};
    struct Cfg { int WV, SK; };
    long long bad_total = 0, n_total = 0;
    for (int cfg = 0; cfg < 2; ++cfg) {
      const int WV = cfg == 0 ? 2 : 1, SK = 4, Wc = WV * 64, nb = N / Wc;
      // S64 for the four row groups, into out[rg * 1M floats]
      for (int rg = 0; rg < 4; ++rg) {
        if (cfg == 0) hipLaunchKernelGGL((S64<2, 1, 4, kFull>), dim3(nb, 1), dim3(WV * SK * 32), 0, 0, b.a + (size_t)rg * 64 * K, (const unsigned*)b.w, b.out + (size_t)rg * (1 << 20), K, N, 2, 4, 1, b.clk, ep);
        else hipLaunchKernelGGL((S64<2, 1, 4, kFull>), dim3(nb, 1), dim3(WV * SK * 32), 0, 0, b.a + (size_t)rg * 64 * K, (const unsigned*)b.w, b.out + (size_t)rg * (1 << 20), K, N, 1, 4, 1, b.clk, ep);
      }
      const int smem = 2 * (WV * SK) * 4 * 1024;
      CK(hipFuncSetAttribute(reinterpret_cast<const void*>(L256<2, 2, 4, kFull>), hipFuncAttributeMaxDynamicSharedMemorySize, smem));
      hipLaunchKernelGGL((L256<2, 2, 4, kFull>), dim3(nb, 1), dim3(WV * SK * 4 * 32), smem, 0, b.a, (const unsigned*)b.w, b.out + (size_t)4 * (1 << 20), K, N, WV, SK, 1, b.clk, ep);
      CK(hipDeviceSynchronize());
      std::vector<float> h((size_t)5 * (1 << 20));
      CK(hipMemcpy(h.data(), b.out, h.size() * 4, hipMemcpyDeviceToHost));
      long long bad = 0, n = 0;
      for (int bx = 0; bx < nb; ++bx)
        for (int cb = 0; cb < WV; ++cb)
          for (int ks = 0; ks < SK; ++ks)
            for (int rg = 0; rg < 4; ++rg)
              for (int lane = 0; lane < 32; ++lane) {
                const float s = h[(size_t)rg * (1 << 20) + (size_t)bx * (WV * SK * 32) + (cb * SK + ks) * 32 + lane];
                const float l = h[(size_t)4 * (1 << 20) + (size_t)bx * (WV * SK * 4 * 32) + (rg + 4 * (cb + WV * ks)) * 32 + lane];
                ++n;
                if (std::memcmp(&s, &l, 4) != 0) ++bad;
              }
      std::printf("check cfg WV%d SK4 NP2 U2 RG4 vs 4 x S64 (WV%d SK4 NP2 U1 MT4): %lld of %lld accumulators differ\n", WV, WV, bad, n);
      bad_total += bad; n_total += n;
    }
    std::printf("CHECK %s (%lld / %lld)\n", bad_total == 0 ? "PASS" : "FAIL", bad_total, n_total);
    return bad_total == 0 ? 0 : 1;
  }
  FILE* json = outp.empty() ? nullptr : std::fopen(outp.c_str(), "wb");
  bool first = true;
  if (json) std::fprintf(json, "{\"device\":\"%s\",\"set\":\"%s\",\"rounds\":%d,\"batch\":%d,\"rows\":[\n", prop.name, set.c_str(), rounds, batch);

  struct Shape { const char* name; int N, K; };
  std::vector<Shape> shapes;
  if (shape == "gate_up" || shape == "all") shapes.push_back({"gate_up", 34816, 5120});
  if (shape == "down" || shape == "all") shapes.push_back({"down", 5120, 17408});
  for (const Shape& s : shapes) {
    Bufs b;
    b.batch = batch;
    b.w_bytes_each = (size_t)s.N * s.K * 5 / 8;   // sized for KB = 5 (KB = 4 uses the first 4/5)
    CK(hipMalloc(&b.w, b.w_bytes_each * batch));
    CK(hipMalloc(&b.a, (size_t)256 * s.K * 2 + 4096));
    CK(hipMalloc(&b.out, (size_t)64 * 1024 * 1024));
    CK(hipMalloc(&b.clk, 64));
    CK(hipMemset(b.clk, 0, 64));
    CK(hipMalloc(&b.svh, (size_t)s.N * 4));
    CK(hipMalloc(&b.c, (size_t)256 * s.N * 2));
    CK(hipMalloc(&b.ws, (size_t)2 * 256 * s.N * 4));
    CK(hipMalloc(&b.tickets, (size_t)(s.N / 128) * 4));
    CK(hipMemset(b.tickets, 0, (size_t)(s.N / 128) * 4));
    hipLaunchKernelGGL(FillW, dim3(8192), dim3(256), 0, 0, (uint32_t*)b.w, (int64_t)(b.w_bytes_each * batch / 4), 12345u);
    hipLaunchKernelGGL(FillW, dim3(256), dim3(256), 0, 0, (uint32_t*)b.svh, (int64_t)s.N, 99u);
    hipLaunchKernelGGL(FillA, dim3(2048), dim3(256), 0, 0, b.a, (int64_t)256 * s.K, 777u);
    CK(hipDeviceSynchronize());
    std::printf("---- %s N=%d K=%d, weights %.1f MiB per launch x %d copies\n", s.name, s.N, s.K,
                b.w_bytes_each / 1048576.0, batch);
    const bool gu = s.N == 34816;
    const int skg = gu ? 1 : 2;   // the stock M = 64 picks: gate_up WV2 NP2 SK4 SKG1 U1 MT4; down WV2 NP2 SK4 SKG2 U1 MT4
    for (int i = 0; i < 20; ++i)
      hipLaunchKernelGGL((S64<2, 1, 4, kFull>), dim3(s.N / 128, 1), dim3(256), 0, 0, b.a, (const unsigned*)b.w, b.out, s.K, s.N, 2, 4, 1, b.clk, EpiOf(b));
    CK(hipDeviceSynchronize());

    std::vector<Job> jobs;
    if (set == "main") {
      AddMain<0>(jobs, "KB4", b, s.N, s.K, skg);
    } else if (set == "cheap" || set == "cheap5") {
      // the cheap variants of today's M = 64 kernel: (1) A fragments stored fragment-contiguous (one b128 per lane
      // instead of two b64 over 16 row segments), (2) four consecutive M = 64 launches on the SAME weights (what a
      // Model chunk of 256 rows costs with today's kernels: the weights may hit the MALL / L2 on launches 2-4).
      if (set == "cheap") {
        AddS64<2, 1, 4, kFull | kEPI>(jobs, "KB4 full+epi", b, s.N, s.K, 2, 4, skg);
        AddS64<2, 1, 4, kFull | kEPI | kAC>(jobs, "KB4 full+epi A-contig", b, s.N, s.K, 2, 4, skg);
        AddS64<2, 1, 4, kFull | kEPI | kNT>(jobs, "KB4 full+epi NT", b, s.N, s.K, 2, 4, skg);
        AddS64<2, 1, 4, kFull | kEPI | kAC | kNT>(jobs, "KB4 full+epi NT A-contig", b, s.N, s.K, 2, 4, skg);
      } else {
        AddS64<2, 1, 4, kFull | kEPI | kKB5>(jobs, "KB5 full+epi", b, s.N, s.K, 2, 4, skg);
        AddS64<2, 1, 4, kFull | kEPI | kAC | kKB5>(jobs, "KB5 full+epi A-contig", b, s.N, s.K, 2, 4, skg);
      }
      if (!jobs.empty()) {
        Job base = jobs[0];
        Job j4 = base;
        j4.name = "S64  same-weights x4 (chunk 256, today's kernel)";
        j4.M = 256;
        auto one = base.launch;
        j4.launch = [one](const unsigned char* w) { for (int i = 0; i < 4; ++i) one(w); };
        jobs.push_back(j4);
      }
    } else if (set == "epiabl") {
      AddL256<2, 2, 4, kFull>(jobs, "KB4 loop", b, s.N, s.K, 1, 4, skg);
      AddL256<2, 2, 4, kFull | kEPI>(jobs, "KB4 full+epi", b, s.N, s.K, 1, 4, skg);
      AddL256<2, 2, 4, kFull | kEPI | kNOTAIL>(jobs, "KB4 epi no-tail", b, s.N, s.K, 1, 4, skg);
      AddL256<2, 2, 4, kFull | kEPI | kNOTAIL | kNOSTORE>(jobs, "KB4 epi passes-only", b, s.N, s.K, 1, 4, skg);
      AddL256<2, 2, 4, kFull>(jobs, "KB4 loop", b, s.N, s.K, 2, 4, skg);
      AddL256<2, 2, 4, kFull | kEPI>(jobs, "KB4 full+epi", b, s.N, s.K, 2, 4, skg);
      AddL256<2, 2, 4, kFull | kEPI | kNOTAIL>(jobs, "KB4 epi no-tail", b, s.N, s.K, 2, 4, skg);
      AddL256<2, 2, 4, kFull | kEPI | kNOTAIL | kNOSTORE>(jobs, "KB4 epi passes-only", b, s.N, s.K, 2, 4, skg);
      AddS64<2, 1, 4, kFull>(jobs, "KB4 loop", b, s.N, s.K, 2, 4, skg);
      AddS64<2, 1, 4, kFull | kEPI>(jobs, "KB4 full+epi", b, s.N, s.K, 2, 4, skg);
    } else if (set == "kb5") {
      AddMain<kKB5>(jobs, "KB5", b, s.N, s.K, skg);    } else if (set == "epi") {
      // epilogue in isolation: K = 512 (32 k-tiles), so the launch is the epilogue plus a tiny loop
      const int K2 = 512;
      AddS64<2, 1, 4, kFull | kEPI>(jobs, "epi-iso", b, s.N, K2, 2, 4, skg);
      AddS64<2, 1, 4, kFull>(jobs, "epi-iso-none", b, s.N, K2, 2, 4, skg);
      AddL256<2, 2, 4, kFull | kEPI>(jobs, "epi-iso", b, s.N, K2, 2, 4, skg);
      AddL256<2, 2, 4, kFull>(jobs, "epi-iso-none", b, s.N, K2, 2, 4, skg);
      AddL256<1, 4, 4, kFull | kEPI>(jobs, "epi-iso", b, s.N, K2, 4, 2, skg);
      AddL256<2, 2, 4, kFull | kEPI>(jobs, "epi-iso", b, s.N, K2, 1, 4, skg);
      AddL256<2, 2, 4, kFull | kEPI>(jobs, "epi-iso", b, s.N, K2, 2, 2, skg);
    }
    RunJobs(jobs, b, rounds);
    Report(jobs, json, first);
    CK(hipFree(b.w)); CK(hipFree(b.a)); CK(hipFree(b.out)); CK(hipFree(b.clk)); CK(hipFree(b.svh)); CK(hipFree(b.c)); CK(hipFree(b.ws)); CK(hipFree(b.tickets));
  }
  if (json) {
    std::fprintf(json, "]}\n");
    std::fclose(json);
    std::printf("wrote %s\n", outp.c_str());
  }
  return 0;
}
