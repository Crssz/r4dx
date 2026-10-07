// r4d_gemm_trellis_nt_i8.hip - the int8 x int8 sibling of r4d_gemm_trellis_nt_m256.hip (in R4D_UNITS since the
// int8 prefill, R4DX_PREFILL_INT8; r4d.h has the contract, docs/int8-prefill.md "Production path" the model
// side, docs/int8-gemm-proto.md the design and the measurements): the same trellis linear
// C = bf16( (FWHT128(A @ Q) * svh) * out_scale ) for M = 256 rows in ONE launch, with the activation A quantized
// to int8 per (row, 128 k) (r4d_trellis_i8_quant_act), the decoded trellis weight quantized to int8 per
// (column, 128 k) on the fly (scale table r4d_trellis_i8_wscale), v_wmma_i32_16x16x16_iu8 and a per-128 fp32
// rescale of the int32 partial sums. NOT bit-identical to the f16 kernel (it is a different, quantized, model of
// the same linear): its own contract is kernel exactness against the integer reference
// (tests/kernels/test_trellis_i8_gemm.cpp) and a KL budget against the f16 path (docs/int8-prefill.md).
//
// The device code is r4d_trellis_i8.h; this unit instantiates ONLY the production kernels,
// i8g_kernel<TRELLIS = true, KB in {4, 5}, FWHT = true, RESC = 0, SKW in {2, 4, 8}> (6 of them), and exports
// the host entries. A translation unit of its own for the reason r4d_gemm_trellis_nt_m64_wq.hip gives: the
// ISA of a kernel moved by a few instructions when other instantiations shared its unit. The dense kernel, the
// RESC 1..3 speed bounds and the reference kernels stay in the bench (tests/kernels/tool_int8_gemm_proto.hip).
// The build (third_party/check_trellis_isa.cmake, GEMM_KERNEL = i8g_kernel, MAX_VGPR 192) requires every
// instantiation of this unit to fit 192 VGPRs (the kernel's amdgpu_waves_per_eu(8) cap) with no scratch and no
// near dependency behind the decode's inline asm.
#include "r4d.h"
#include "r4d_trellis_i8.h"
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

static void r4d_i8_fail(const std::string& what) { throw std::runtime_error("r4d gemm_trellis_nt_i8: " + what); }

// ---- the production kernels' instantiation table --------------------------------------------------------------
template <int KB, int SKW>
static void r4d_i8_launch(const signed char* a0, const float* s0, const signed char* a1, const float* s1, int n_split,
                          const unsigned char* w, const float* sw, const float* svh, unsigned short* c, float* ws,
                          unsigned* tickets, int K, int N, int SKG, float out_scale, hipStream_t stream) {
  hipLaunchKernelGGL((i8g_kernel<true, KB, true, 0, SKW>), dim3(N / 32, SKG), dim3(128 * SKW), 8192 * SKW, stream, a0, s0,
                     a1, s1, n_split, w, sw, svh, c, ws, tickets, K, N, SKG, out_scale);
}
template <int KB>
static bool r4d_i8_dispatch_skw(int skw, const signed char* a0, const float* s0, const signed char* a1, const float* s1,
                                int n_split, const unsigned char* w, const float* sw, const float* svh,
                                unsigned short* c, float* ws, unsigned* tickets, int K, int N, int SKG, float out_scale,
                                hipStream_t stream) {
  switch (skw) {
    case 2: r4d_i8_launch<KB, 2>(a0, s0, a1, s1, n_split, w, sw, svh, c, ws, tickets, K, N, SKG, out_scale, stream); return true;
    case 4: r4d_i8_launch<KB, 4>(a0, s0, a1, s1, n_split, w, sw, svh, c, ws, tickets, K, N, SKG, out_scale, stream); return true;
    case 8: r4d_i8_launch<KB, 8>(a0, s0, a1, s1, n_split, w, sw, svh, c, ws, tickets, K, N, SKG, out_scale, stream); return true;
    default: return false;
  }
}

// ---- host entries ----------------------------------------------------------------------------------------------
const char* r4d_gemm_trellis_nt_i8_check(int M, int K, int N, int n_split, int KB, int skw, int skg) {
  static thread_local char msg[320];
#define R4D_I8_BAD(...) do { std::snprintf(msg, sizeof msg, __VA_ARGS__); return msg; } while (0)
  if (M != 256) R4D_I8_BAD("M must be 256, got %d", M);
  if (K <= 0 || N <= 0 || K % 128 || N % 128) R4D_I8_BAD("K and N must be multiples of 128, got K %d N %d", K, N);
  if (KB != 4 && KB != 5) R4D_I8_BAD("KB must be 4 or 5, got %d", KB);
  if (skw != 2 && skw != 4 && skw != 8) R4D_I8_BAD("skw must be 2, 4 or 8, got %d", skw);
  if (skg != 1 && skg != 2 && skg != 4 && skg != 8) R4D_I8_BAD("skg must be 1, 2, 4 or 8, got %d", skg);
  if (n_split < 0 || n_split > N || n_split % 128)
    R4D_I8_BAD("n_split %d must be in 0..N (%d) and a multiple of 128", n_split, N);
  if ((K / 128) % (skw * skg))
    R4D_I8_BAD("K / 128 = %d must be divisible by skw * skg = %d", K / 128, skw * skg);
  if (8192 * skw > 64 * 1024) R4D_I8_BAD("LDS %d B exceeds 64 KiB", 8192 * skw);
#undef R4D_I8_BAD
  return nullptr;
}

size_t r4d_gemm_trellis_nt_i8_ws_bytes(int M, int N, int SKG) {
  return (size_t)SKG * (size_t)M * (size_t)N * sizeof(float);
}

void r4d_gemm_trellis_nt_i8(int64_t a8_0, int64_t sa_0, int64_t a8_1, int64_t sa_1, int n_split, int64_t w,
                            int64_t sw, int64_t svh, int64_t c, int64_t ws, int64_t tickets, int M, int K, int N, int KB,
                            int skw, int skg, float out_scale, int64_t stream) {
  if (const char* why = r4d_gemm_trellis_nt_i8_check(M, K, N, n_split, KB, skw, skg)) r4d_i8_fail(why);
  if (!a8_0 || !sa_0 || !w || !sw || !svh || !c || !ws || !tickets) r4d_i8_fail("null buffer");
  if ((!a8_1) != (!sa_1)) r4d_i8_fail("a8_1 and sa_1 must be given together");
  if (n_split < N && !a8_1) r4d_i8_fail("n_split < N needs the second A part");
  if (!a8_1) {   // a single part: the first pair serves every column
    a8_1 = a8_0;
    sa_1 = sa_0;
  }
  const signed char* a0 = (const signed char*)a8_0;
  const signed char* a1 = (const signed char*)a8_1;
  const float* s0 = (const float*)sa_0;
  const float* s1 = (const float*)sa_1;
  const unsigned char* wp = (const unsigned char*)w;
  const bool ok = KB == 4 ? r4d_i8_dispatch_skw<4>(skw, a0, s0, a1, s1, n_split, wp, (const float*)sw, (const float*)svh,
                                                    (unsigned short*)c, (float*)ws, (unsigned*)tickets, K, N, skg,
                                                    out_scale, (hipStream_t)stream)
                          : r4d_i8_dispatch_skw<5>(skw, a0, s0, a1, s1, n_split, wp, (const float*)sw, (const float*)svh,
                                                    (unsigned short*)c, (float*)ws, (unsigned*)tickets, K, N, skg,
                                                    out_scale, (hipStream_t)stream);
  if (!ok) r4d_i8_fail("configuration not instantiated");
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) r4d_i8_fail(std::string("launch failed: ") + hipGetErrorString(e));
}

size_t r4d_trellis_i8_wscale_count(int K, int N) { return (size_t)(K / 128) * (size_t)N; }

static void r4d_i8_check_weights(const char* who, int K, int N, int KB) {
  if (K <= 0 || N <= 0 || K % 128 || N % 128) r4d_i8_fail(std::string(who) + ": K and N must be multiples of 128");
  if (KB != 4 && KB != 5) r4d_i8_fail(std::string(who) + ": KB must be 4 or 5");
}

void r4d_trellis_i8_wscale(int64_t w, int64_t sw, int K, int N, int KB, int64_t stream) {
  r4d_i8_check_weights("wscale", K, N, KB);
  if (!w || !sw) r4d_i8_fail("wscale: null buffer");
  const long long groups = (long long)(N / 32) * (K / 128);
  const dim3 grid((unsigned)((groups + 7) / 8)), block(256);
  if (KB == 4)
    hipLaunchKernelGGL(i8g_wscale<4>, grid, block, 0, (hipStream_t)stream, (const unsigned*)w, (float*)sw, K, N);
  else
    hipLaunchKernelGGL(i8g_wscale<5>, grid, block, 0, (hipStream_t)stream, (const unsigned*)w, (float*)sw, K, N);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) r4d_i8_fail(std::string("wscale launch failed: ") + hipGetErrorString(e));
}

void r4d_trellis_i8_dump_w(int64_t w, int64_t sw, int64_t w8, int64_t wp, int K, int N, int KB, int64_t stream) {
  r4d_i8_check_weights("dump_w", K, N, KB);
  if (!w || !sw || !w8) r4d_i8_fail("dump_w: null buffer");
  const long long blocks = (long long)(N / 32) * (K / 16);
  const dim3 grid((unsigned)((blocks + 7) / 8)), block(256);
  if (KB == 4)
    hipLaunchKernelGGL(i8g_dump_w<4>, grid, block, 0, (hipStream_t)stream, (const unsigned*)w, (const float*)sw,
                       (signed char*)w8, (signed char*)wp, K, N);
  else
    hipLaunchKernelGGL(i8g_dump_w<5>, grid, block, 0, (hipStream_t)stream, (const unsigned*)w, (const float*)sw,
                       (signed char*)w8, (signed char*)wp, K, N);
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) r4d_i8_fail(std::string("dump_w launch failed: ") + hipGetErrorString(e));
}

void r4d_trellis_i8_quant_act(int64_t x, int64_t a8, int64_t sa, int parts, int64_t part_stride, int K, int64_t stream) {
  if (K <= 0 || K % 128) r4d_i8_fail("quant_act: K must be a positive multiple of 128");
  if (parts < 1 || parts > 2) r4d_i8_fail("quant_act: parts must be 1 or 2");
  if (!x || !a8 || !sa) r4d_i8_fail("quant_act: null buffer");
  if (parts > 1 && part_stride < (int64_t)i8p::kM * K) r4d_i8_fail("quant_act: part_stride must be at least 256 * K elements");
  const long long waves = (long long)i8p::kM * (K / 128);
  hipLaunchKernelGGL(i8g_quant_act, dim3((unsigned)((waves + 7) / 8), (unsigned)parts), dim3(256), 0, (hipStream_t)stream,
                     (const unsigned short*)x, (signed char*)a8, (signed char*)nullptr, (float*)sa, K,
                     (long long)(parts > 1 ? part_stride : 0));
  const hipError_t e = hipGetLastError();
  if (e != hipSuccess) r4d_i8_fail(std::string("quant_act launch failed: ") + hipGetErrorString(e));
}
