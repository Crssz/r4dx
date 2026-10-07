// tests/kernels/int8_gemm_proto_kernels.h -- the BENCH layer of the int8 x int8 prefill GEMM (docs/int8-gemm-proto.md):
// what tool_int8_gemm_proto.hip (hipcc, gfx1201) and test_int8_gemm_proto_emu.cpp (I8G_EMU, plain C++) need on top of
// libr4d's device code. The kernels themselves (i8g_kernel, i8g_quant_act, i8g_wscale, i8g_dump_w) are
// third_party/libr4d/r4d_trellis_i8.h's, the very source the production unit r4d_gemm_trellis_nt_i8.hip builds, so
// the bench measures and the emulation checks the production code; this file adds the reference kernel
// i8g_ref_cols (exact integer sums for chosen columns) and the host launch + dispatch of EVERY variant the bench
// sweeps (the dense kernel and the RESC 1..3 speed bounds exist only here, the production unit instantiates
// TRELLIS, FWHT, RESC 0 only). A bench-only experiment: nothing here is wired into the model.
#pragma once

#include <cstdio>
#include <stdexcept>
#include <string>

#include "int8_gemm_proto_ref.h"   // the CPU references / quantizer definitions the bench and the tests compare against
#include "r4d_trellis_i8.h"
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
  int resc = 0;        // 0: per-128 rescale (the math), 1..3: speed bounds, 4: the production COARSE mode (SA [256], SW [N])
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
  if (c.resc < 0 || c.resc > 4) I8G_BAD("resc must be 0, 1, 2, 3 or 4");
  if (!c.trellis && c.resc > 1 && c.resc != 4) I8G_BAD("resc 2 and 3 are instantiated for the trellis kernel only");
#undef I8G_BAD
  return nullptr;
}

#ifndef I8G_EMU
template <bool TR, int KB, bool FW, int RE, int SKW>
static void i8g_launch_t(const I8gCfg& c, const signed char* a8, const float* sa, const void* w, const float* sw,
                         const float* svh, unsigned short* C, float* ws, unsigned* tk, int K, int N, float out_scale,
                         hipStream_t st) {
  hipLaunchKernelGGL((i8g_kernel<TR, KB, FW, RE, SKW>), dim3(N / 32, c.skg), dim3(128 * SKW), 8192 * SKW, st, a8, sa, a8, sa,
                     N, (const unsigned char*)w, sw, svh, C, ws, tk, K, N, c.skg, out_scale);   // one part: n_split = N
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
                    case 2: I8G_CASE(TR, KB, FW, 2); break; case 3: I8G_CASE(TR, KB, FW, 3); break; \
                    default: I8G_CASE(TR, KB, FW, 4); break; }
  if (!c.trellis) {
    if (c.fwht) { if (c.resc == 0) I8G_CASE(false, 4, true, 0); else if (c.resc == 1) I8G_CASE(false, 4, true, 1); else I8G_CASE(false, 4, true, 4); }
    else { if (c.resc == 0) I8G_CASE(false, 4, false, 0); else if (c.resc == 1) I8G_CASE(false, 4, false, 1); else I8G_CASE(false, 4, false, 4); }
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
#endif  // !I8G_EMU
inline size_t I8gWsBytes(int N, int skg) { return (size_t)skg * i8p::kM * N * sizeof(float); }
