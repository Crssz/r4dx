// tests/kernels/tool_int8_gemm_proto.hip -- the int8 x int8 prefill GEMM go / no-go bench (docs/int8-gemm-proto.md).
// Standalone (host + device), one device (HIP_VISIBLE_DEVICES set by the caller), every set bounded. Nothing here
// touches a production kernel or the model.
//
//   probe     peak WMMA rate of this part: f16, bf16, int8 (v_wmma_i32_16x16x16_iu8) and fp8, a register-only loop on
//             eight independent accumulators, TOPS from event time and the shader clock from clock64 / wall_clock64.
//   selftest  small shapes, every legal configuration of every kernel against CPU references: the single iu8 WMMA
//             (A, B and D layouts, signs), the activation quantizer, the scale table and the int8 weights against the
//             CPU trellis decode (trellis_ref.hpp), and the GEMMs (dense / trellis, FWHT epilogue or plain, per-128
//             or coarse rescale) against an exact-integer fp64 reference. Run it first: a FAIL here means the
//             timings below it mean nothing.
//   verify    the model's real linear classes: every configuration that gets timed is checked on five whole
//             128-column groups (the first, the last and three pseudo-random ones, all 256 rows) against the GPU
//             fp64-exact reference (i8g_ref_cols on the plain int8 matrices), the trellis kernel against the dense
//             kernel (they must agree byte for byte: same summation order, same int8 weights), and against the
//             shipped f16 trellis kernel (the fake-quant-level difference).
//   time      per class (KB 4 and 5): the shipped f16 M = 256 trellis kernel (the configuration PlanTrellisM256
//             picks, plus the best of the other legal ones), the dense int8 kernel (with the trellis epilogue, and
//             plain), the trellis int8 kernel, the coarse-rescale bound variants, and the activation quantizer; the
//             MLP-pair and all-class summaries and the decision rule. Same protocol as tool_trellis_m256_bench: weights
//             in `batch` cold copies, every round times every job once (a batch of launches on different copies), the
//             numbers are medians of per-round values and the ratios are medians of per-round ratios.
//
// Build: tests/kernels/build_int8_gemm_proto.ps1 (hipcc; links the shipped r4d_gemm_trellis_nt_m256 unit).
//   tool_int8_gemm_proto.exe [--mode all|probe|selftest|verify|time] [--shape gate_up|down|qkv|z|out|qg|kv|all]
//                            [--kb 4|5|both] [--rounds 11] [--batch 6] [--quick] [--noverify] [--no-resc]
//                            [--simds 128] [--verbose] [--out x.json]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "int8_gemm_proto_kernels.h"
#include "r4d.h"
#include "trellis_ref.hpp"
#include "trellis_tuning_rows.hpp"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e_)); \
  std::exit(1); } } while (0)

static const float kOutScale = 0.0883883f;   // 1 / sqrt(128), the trellis linear's

// ---- fills ------------------------------------------------------------------------------------------------------
__device__ __forceinline__ uint32_t Hash32(uint64_t i, uint32_t seed) {
  uint32_t h = (uint32_t)i * 0x9E3779B9u ^ (uint32_t)(i >> 32) * 0x85EBCA6Bu ^ seed;
  h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
  return h;
}
__global__ void FillW(uint32_t* p, int64_t n, uint32_t seed) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) p[i] = Hash32(i, seed);
}
// f16 N(0, sigma^2) (Box-Muller from two hashes): the rotated activations of a trellis linear are near Gaussian
__global__ void FillAg(unsigned short* p, int64_t n, float sigma, uint32_t seed) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const float u1 = ((float)(Hash32(2 * i, seed) >> 8) + 0.5f) * (1.0f / 16777216.0f);
    const float u2 = (float)(Hash32(2 * i + 1, seed) >> 8) * (1.0f / 16777216.0f);
    const float z = sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
    p[i] = __builtin_bit_cast(unsigned short, (_Float16)(z * sigma));
  }
}
__global__ void FillSvh(float* p, int64_t n, uint32_t seed) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint32_t h = Hash32(i, seed);
    const float m = 0.5f + (float)(h & 0xFFFFu) / 65536.f;
    p[i] = (h & 0x10000u) ? -m : m;
  }
}

// ---- WMMA probe ---------------------------------------------------------------------------------------------------
// KIND 0 f16, 1 bf16, 2 iu8, 3 fp8 (e4m3 x e4m3): 8 independent accumulators, operands in registers
typedef _Float16 pv8h __attribute__((ext_vector_type(8)));
typedef __bf16 pv8bf __attribute__((ext_vector_type(8)));
typedef int pv2i __attribute__((ext_vector_type(2)));
typedef int pv8i __attribute__((ext_vector_type(8)));
typedef float pv8f __attribute__((ext_vector_type(8)));
typedef unsigned pv4u __attribute__((ext_vector_type(4)));

template <int KIND>
__global__ __launch_bounds__(256) void ProbeKernel(float* sink, unsigned long long* clk, int iters) {
  const unsigned lane = threadIdx.x & 31, seed = blockIdx.x * 977u + (threadIdx.x >> 5) * 31u;
  const unsigned long long c0 = clock64(), w0 = wall_clock64();
  float s = 0.f;
  if constexpr (KIND == 0 || KIND == 1) {
    const pv4u ua = {0x3C003C00u ^ lane, 0x3C00u ^ seed, 0x3C003C00u, 0x3C00u ^ lane};
    const pv4u ub = {0x3C00u ^ seed, 0x3C003C00u ^ lane, 0x3C00u, 0x3C003C00u};
    pv8f acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = (pv8f)(0.f);
    if constexpr (KIND == 0) {
      const pv8h a = __builtin_bit_cast(pv8h, ua), b = __builtin_bit_cast(pv8h, ub);
      for (int it = 0; it < iters; ++it)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[j]);
    } else {
      const pv8bf a = __builtin_bit_cast(pv8bf, ua), b = __builtin_bit_cast(pv8bf, ub);
      for (int it = 0; it < iters; ++it)
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, acc[j]);
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) s += acc[j][0] + acc[j][7];
  } else if constexpr (KIND == 2) {
    const pv2i a = {(int)(0x01020304u ^ lane), (int)(0x05060708u ^ seed)}, b = {(int)(0x01010101u ^ seed), (int)(0x7F7F0101u ^ lane)};
    pv8i acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = (pv8i)(0);
    for (int it = 0; it < iters; ++it)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, acc[j], false);
#pragma unroll
    for (int j = 0; j < 8; ++j) s += (float)(acc[j][0] + acc[j][7]);
  } else {
    const pv2i a = {(int)(0x30303030u ^ (lane & 0x0F0F0F0Fu)), (int)(0x30303030u ^ (seed & 0x0F0F0F0Fu))}, b = {(int)0x30303030u, (int)(0x30303030u ^ (lane & 0x0F0F0F0Fu))};
    pv8f acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = (pv8f)(0.f);
    for (int it = 0; it < iters; ++it)
#pragma unroll
      for (int j = 0; j < 8; ++j) acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, acc[j]);
#pragma unroll
    for (int j = 0; j < 8; ++j) s += acc[j][0] + acc[j][7];
  }
  const unsigned long long c1 = clock64(), w1 = wall_clock64();
  if (lane == 0 && (threadIdx.x >> 5) == 0) {
    clk[blockIdx.x * 2] = c1 - c0;
    clk[blockIdx.x * 2 + 1] = w1 - w0;
  }
  if (s == 1234.5f) sink[blockIdx.x * 256 + threadIdx.x] = s;
}

static double Median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v.empty() ? 0.0 : (v.size() % 2 ? v[v.size() / 2] : 0.5 * (v[v.size() / 2 - 1] + v[v.size() / 2]));
}
static double MinOf(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }
static double MaxOf(const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); }

struct ProbeResult { double tops = 0, sclk = 0, clk_per_wmma_simd = 0; };
static std::map<std::string, ProbeResult> g_probe;

static void RunProbe(const hipDeviceProp_t& prop, int simds, int reps) {
  const int mp = prop.multiProcessorCount;
  const int grid = 8 * mp;
  float* sink;
  unsigned long long* clk;
  CK(hipMalloc(&sink, (size_t)grid * 256 * 4));
  CK(hipMalloc(&clk, (size_t)grid * 16));
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  std::printf("== probe: peak WMMA rate, %d blocks x 8 waves, 8 independent accumulators per wave, %d simds assumed ==\n", grid, simds);
  auto run = [&](const char* name, auto kernel, int iters) {
    std::vector<double> tops, sclk, cpw;
    for (int rep = 0; rep < reps + 1; ++rep) {
      CK(hipEventRecord(e0, 0));
      hipLaunchKernelGGL(kernel, dim3(grid), dim3(256), 0, 0, sink, clk, iters);
      CK(hipEventRecord(e1, 0));
      CK(hipEventSynchronize(e1));
      if (rep == 0) continue;   // warm-up
      float ms = 0;
      CK(hipEventElapsedTime(&ms, e0, e1));
      const double ops = (double)grid * 8 * iters * 8 * 8192.0;     // waves x iters x 8 WMMA x (16 x 16 x 16 x 2)
      tops.push_back(ops / (ms * 1e-3) / 1e12);
      std::vector<unsigned long long> h((size_t)grid * 2);
      CK(hipMemcpy(h.data(), clk, h.size() * 8, hipMemcpyDeviceToHost));
      std::vector<double> sc;
      for (int b = 0; b < grid; ++b) if (h[b * 2 + 1]) sc.push_back((double)h[b * 2] / (double)h[b * 2 + 1] * 100.0);
      sclk.push_back(Median(sc));
      cpw.push_back(sclk.back() * 1e6 * (ms * 1e-3) * simds / ((double)grid * 8 * iters * 8));   // clk per WMMA per SIMD
    }
    ProbeResult r;
    r.tops = *std::max_element(tops.begin(), tops.end());
    r.sclk = Median(sclk);
    r.clk_per_wmma_simd = *std::min_element(cpw.begin(), cpw.end());
    g_probe[name] = r;
    std::printf("  %-6s %8.1f TOPS (best of %d, median %.1f)  shader clock %.0f MHz  = %.2f clk per WMMA per SIMD\n", name, r.tops, reps,
                Median(tops), r.sclk, r.clk_per_wmma_simd);
  };
  run("f16", ProbeKernel<0>, 20000);
  run("bf16", ProbeKernel<1>, 20000);
  run("iu8", ProbeKernel<2>, 40000);
  run("fp8", ProbeKernel<3>, 40000);
  if (g_probe["f16"].tops > 0)
    std::printf("  ratios: iu8 / f16 = %.3f, fp8 / f16 = %.3f  (the f16 WMMA of the shipped kernel is the baseline's engine)\n",
                g_probe["iu8"].tops / g_probe["f16"].tops, g_probe["fp8"].tops / g_probe["f16"].tops);
  CK(hipEventDestroy(e0));
  CK(hipEventDestroy(e1));
  CK(hipFree(sink));
  CK(hipFree(clk));
}

// ---- the single iu8 WMMA, layouts and signs ----------------------------------------------------------------------------
__global__ void WmmaUnit(const signed char* Af, const signed char* Bf, const int* Cin, int* D) {
  const int lane = threadIdx.x;
  const i8g_v2i a = *(const i8g_v2i*)(Af + lane * 8), b = *(const i8g_v2i*)(Bf + lane * 8);
  i8g_v8i c;
  for (int e = 0; e < 8; ++e) c[e] = Cin[lane * 8 + e];
  c = i8g_mma(a, b, c);
  for (int e = 0; e < 8; ++e) D[lane * 8 + e] = c[e];
}

// ---- helpers ----------------------------------------------------------------------------------------------------------
struct Shape { const char* name; int N, K; int count; };
static const Shape kShapes[] = {
    {"gate_up", 34816, 5120, 64}, {"down", 5120, 17408, 64}, {"qkv", 10240, 5120, 48}, {"z", 6144, 5120, 48},
    {"out", 5120, 6144, 64},      {"qg", 12288, 5120, 16},   {"kv", 1024, 5120, 32},
};
struct Stock { int WV, SK, NP, SKG, U, NT, MT; };
static bool TableRow(const Shape& s, int kb, Stock& out) {
  for (const trellis_rows::GemmTuningRow& r : trellis_rows::kGemmTuningTable)
    if (r.layout == trellis_rows::Layout::kTrellis && r.N == s.N && r.K == s.K && r.M == 64 && r.rate == kb) {
      out = {r.tuning.WV, r.tuning.SK, r.tuning.NPW, r.tuning.SKG, r.tuning.U, r.tuning.NT, r.tuning.MB};
      return true;
    }
  return false;
}
// PlanTrellisM256 (src/model/linear.cpp), reproduced: the shipped M = 64 row's (SK, SKG), K slices resident per
// workgroup `pref` first, then 4, 8, 2, the first one the host check accepts
struct Plan { int SK = 0, SKG = 0, skw = 0; bool ok = false; };
static Plan ShippedPlan(const Shape& s, int kb) {
  Plan p;
  Stock t{};
  if (!TableRow(s, kb, t)) return p;
  p.SK = t.SK; p.SKG = t.SKG;
  const int pref = t.SK <= 2 ? 2 : (t.SK == 4 && kb == 4 && s.N == 34816) ? 2 : 4;
  const int order[4] = {pref, 4, 8, 2};
  for (int skw : order) {
    if (skw > t.SK) continue;
    if (r4d_gemm_trellis_nt_m256_check(256, s.K, s.N, s.N, kb, t.SK, 1, t.SKG, 4, skw) == nullptr) { p.skw = skw; p.ok = true; return p; }
  }
  return p;
}

static float Bf16ToF(unsigned short b) { return __builtin_bit_cast(float, (unsigned)b << 16); }
static std::string CfgName(const I8gCfg& c) {
  char b[112];
  static const char* const kResc[4] = {"", " COARSE-both", " COARSE-act", " COARSE-wgt"};
  std::snprintf(b, sizeof b, "%s%s%s skw%d skg%d%s", c.trellis ? "trellis" : "dense", c.trellis ? (c.kb == 4 ? "-kb4" : "-kb5") : "",
                c.trellis ? "" : (c.fwht ? "+fwht" : "-plain"), c.skw, c.skg, kResc[c.resc & 3]);
  return b;
}
// the scale mask of the reference (bit 0: the activation scale of kb 0 for all, bit 1: the weight scale of kb 0 for all)
static int MaskOfResc(int resc) { static const int m[4] = {0, 3, 1, 2}; return m[resc & 3]; }

// the whole bench state of one (shape, KB)
struct Case {
  int N = 0, K = 0, KB = 4, batch = 1;
  size_t wt_bytes = 0, w8_bytes = 0, sw_bytes = 0;
  unsigned char* wt = nullptr;          // trellis words, batch copies
  signed char* w8 = nullptr;            // dense int8 (block layout), batch copies
  float* sw = nullptr;                  // scale table, batch copies
  signed char* wp = nullptr;            // plain [N][K] (one copy, the references)
  unsigned short* a16 = nullptr;        // f16 activations [256][K]
  signed char* a8 = nullptr;
  signed char* ap = nullptr;            // plain [256][K]
  float* sa = nullptr;                  // [K/128][256]
  float* svh = nullptr;
  unsigned short* c_f16 = nullptr;      // shipped kernel output
  unsigned short* c_a = nullptr;        // scratch outputs for the verified kernel
  unsigned short* c_b = nullptr;
  float* ws = nullptr;                  // 8 x 256 x N fp32
  unsigned* tk = nullptr;
  unsigned* tk2 = nullptr;
};
static void AllocCase(Case& c, int N, int K, int KB, int batch) {
  c.N = N; c.K = K; c.KB = KB; c.batch = batch;
  c.wt_bytes = (size_t)N * K * KB / 8;
  c.w8_bytes = (size_t)N * K;
  c.sw_bytes = (size_t)(K / 128) * N * 4;
  CK(hipMalloc(&c.wt, c.wt_bytes * batch));
  CK(hipMalloc(&c.w8, c.w8_bytes * batch));
  CK(hipMalloc(&c.sw, c.sw_bytes * batch));
  CK(hipMalloc(&c.wp, c.w8_bytes));
  CK(hipMalloc(&c.a16, (size_t)256 * K * 2));
  CK(hipMalloc(&c.a8, (size_t)256 * K));
  CK(hipMalloc(&c.ap, (size_t)256 * K));
  CK(hipMalloc(&c.sa, (size_t)(K / 128) * 256 * 4));
  CK(hipMalloc(&c.svh, (size_t)N * 4));
  CK(hipMalloc(&c.c_f16, (size_t)256 * N * 2));
  CK(hipMalloc(&c.c_a, (size_t)256 * N * 2));
  CK(hipMalloc(&c.c_b, (size_t)256 * N * 2));
  CK(hipMalloc(&c.ws, (size_t)8 * 256 * N * 4));
  CK(hipMalloc(&c.tk, (size_t)(N / 128) * 4));
  CK(hipMalloc(&c.tk2, (size_t)(N / 128) * 4));
  CK(hipMemset(c.tk, 0, (size_t)(N / 128) * 4));
  CK(hipMemset(c.tk2, 0, (size_t)(N / 128) * 4));
}
static void FreeCase(Case& c) {
  void* ptrs[] = {c.wt, c.w8, c.sw, c.wp, c.a16, c.a8, c.ap, c.sa, c.svh, c.c_f16, c.c_a, c.c_b, c.ws, c.tk, c.tk2};
  for (void* p : ptrs) CK(hipFree(p));
  c = Case();
}
// words / activations / svh, the producers, the scale table and the int8 matrix; copies 1.. of the weights are
// device-to-device copies of copy 0 (distinct addresses = cold, same bytes)
static void FillCase(Case& c, uint32_t seed) {
  hipLaunchKernelGGL(FillW, dim3(8192), dim3(256), 0, 0, (uint32_t*)c.wt, (int64_t)(c.wt_bytes / 4), seed * 7919u + 1u);
  hipLaunchKernelGGL(FillAg, dim3(2048), dim3(256), 0, 0, c.a16, (int64_t)256 * c.K, 0.8f, seed * 104729u + 3u);
  hipLaunchKernelGGL(FillSvh, dim3(256), dim3(256), 0, 0, c.svh, (int64_t)c.N, seed * 1299709u + 5u);
  CK(hipDeviceSynchronize());
  const int NKB = c.K / 128;
  hipLaunchKernelGGL(i8g_quant_act, dim3((256 * NKB + 7) / 8), dim3(256), 0, 0, c.a16, c.a8, c.ap, c.sa, c.K);
  const long long groups = (long long)(c.N / 32) * NKB;
  if (c.KB == 4) hipLaunchKernelGGL(i8g_wscale<4>, dim3((unsigned)((groups + 7) / 8)), dim3(256), 0, 0, (const unsigned*)c.wt, c.sw, c.K, c.N);
  else hipLaunchKernelGGL(i8g_wscale<5>, dim3((unsigned)((groups + 7) / 8)), dim3(256), 0, 0, (const unsigned*)c.wt, c.sw, c.K, c.N);
  CK(hipDeviceSynchronize());
  const long long blocks = (long long)(c.N / 32) * (c.K / 16);
  if (c.KB == 4) hipLaunchKernelGGL(i8g_dump_w<4>, dim3((unsigned)((blocks + 7) / 8)), dim3(256), 0, 0, (const unsigned*)c.wt, c.sw, c.w8, c.wp, c.K, c.N);
  else hipLaunchKernelGGL(i8g_dump_w<5>, dim3((unsigned)((blocks + 7) / 8)), dim3(256), 0, 0, (const unsigned*)c.wt, c.sw, c.w8, c.wp, c.K, c.N);
  CK(hipDeviceSynchronize());
  for (int i = 1; i < c.batch; ++i) {
    CK(hipMemcpy(c.wt + (size_t)i * c.wt_bytes, c.wt, c.wt_bytes, hipMemcpyDeviceToDevice));
    CK(hipMemcpy(c.w8 + (size_t)i * c.w8_bytes, c.w8, c.w8_bytes, hipMemcpyDeviceToDevice));
    CK(hipMemcpy((unsigned char*)c.sw + (size_t)i * c.sw_bytes, c.sw, c.sw_bytes, hipMemcpyDeviceToDevice));
  }
  CK(hipDeviceSynchronize());
}

static void LaunchShipped(const Case& c, const Plan& p, int copy, unsigned short* out) {
  r4d_gemm_trellis_nt_m256((int64_t)c.a16, 0, c.N, (int64_t)(c.wt + (size_t)copy * c.wt_bytes), (int64_t)c.svh, (int64_t)out,
                           (int64_t)c.ws, (int64_t)c.tk, 256, c.K, c.N, c.KB, p.SK, 1, p.SKG, 4, kOutScale, 0, p.skw);
}
static void LaunchI8(const Case& c, const I8gCfg& g, int copy, unsigned short* out) {
  const void* w = g.trellis ? (const void*)(c.wt + (size_t)copy * c.wt_bytes) : (const void*)(c.w8 + (size_t)copy * c.w8_bytes);
  const float* sw = (const float*)((const unsigned char*)c.sw + (size_t)copy * c.sw_bytes);
  I8gRun(g, c.a8, c.sa, w, sw, c.svh, out, c.ws, c.tk2, c.K, c.N, g.fwht ? kOutScale : 1.0f, 0);
}

// ---- verification against the GPU fp64 reference ---------------------------------------------------------------------
struct Verifier {
  int N = 0, K = 0;
  std::vector<int> groups, cols;
  int* d_cols = nullptr;
  double* d_out = nullptr;
  std::vector<double> refs[4];         // by scale mask, each [ncols][256]
  std::vector<float> svh;
  void Build(const Case& c) {
    N = c.N; K = c.K;
    std::set<int> g = {0, c.N / 128 - 1};
    std::mt19937 rng(1234);
    while ((int)g.size() < std::min(5, c.N / 128)) g.insert((int)(rng() % (c.N / 128)));
    groups.assign(g.begin(), g.end());
    for (int gi : groups) for (int j = 0; j < 128; ++j) cols.push_back(gi * 128 + j);
    CK(hipMalloc(&d_cols, cols.size() * 4));
    CK(hipMemcpy(d_cols, cols.data(), cols.size() * 4, hipMemcpyHostToDevice));
    CK(hipMalloc(&d_out, cols.size() * 256 * 8));
    for (int mask = 0; mask < 4; ++mask) {
      hipLaunchKernelGGL(i8g_ref_cols, dim3((unsigned)cols.size()), dim3(256), 0, 0, c.ap, c.sa, c.wp, c.sw, d_cols, d_out, c.K, c.N, mask);
      CK(hipDeviceSynchronize());
      refs[mask].resize(cols.size() * 256);
      CK(hipMemcpy(refs[mask].data(), d_out, refs[mask].size() * 8, hipMemcpyDeviceToHost));
    }
    svh.resize(c.N);
    CK(hipMemcpy(svh.data(), c.svh, (size_t)c.N * 4, hipMemcpyDeviceToHost));
  }
  void Free() { CK(hipFree(d_cols)); CK(hipFree(d_out)); }
  // C: the whole [256][N] bf16 output
  struct Res { long long viol = 0, total = 0; double max_dev = 0; };
  Res Check(const std::vector<unsigned short>& C, bool fwht, int resc, float out_scale) const {
    Res r;
    const std::vector<double>& raw = refs[MaskOfResc(resc)];
    for (size_t gi = 0; gi < groups.size(); ++gi)
      for (int row = 0; row < 256; ++row) {
        double v[128], e[128], rms = 0;
        for (int j = 0; j < 128; ++j) v[j] = raw[(gi * 128 + j) * 256 + row];
        if (fwht) i8p::Fwht128(v);
        for (int j = 0; j < 128; ++j) {
          e[j] = fwht ? v[j] * (double)svh[groups[gi] * 128 + j] * (double)out_scale : v[j] * (double)out_scale;
          rms += e[j] * e[j];
        }
        rms = std::sqrt(rms / 128);
        for (int j = 0; j < 128; ++j) {
          const double got = Bf16ToF(C[(size_t)row * N + groups[gi] * 128 + j]);
          const double dev = std::fabs(got - e[j]), tol = 0.0085 * std::fabs(e[j]) + 1e-4 * rms;
          ++r.total;
          r.viol += dev > tol;
          r.max_dev = std::max(r.max_dev, dev / (rms > 0 ? rms : 1.0));
        }
      }
    return r;
  }
  // relative RMS difference of two outputs on the chosen columns
  double RelDiff(const std::vector<unsigned short>& a, const std::vector<unsigned short>& b) const {
    double d2 = 0, r2 = 0;
    for (int gi : groups)
      for (int row = 0; row < 256; ++row)
        for (int j = 0; j < 128; ++j) {
          const size_t i = (size_t)row * N + gi * 128 + j;
          const double x = Bf16ToF(a[i]), y = Bf16ToF(b[i]);
          d2 += (x - y) * (x - y);
          r2 += y * y;
        }
    return std::sqrt(d2 / r2);
  }
};

static std::vector<unsigned short> ReadC(const unsigned short* d, int N) {
  std::vector<unsigned short> h((size_t)256 * N);
  CK(hipMemcpy(h.data(), d, h.size() * 2, hipMemcpyDeviceToHost));
  return h;
}

// every legal (skw, skg) of a kernel kind for this shape
static std::vector<I8gCfg> Sweep(int K, int N, bool trellis, int kb, bool fwht, int resc) {
  std::vector<I8gCfg> v;
  for (int skw : {2, 4, 8})
    for (int skg : {1, 2, 4, 8}) {
      I8gCfg c;
      c.trellis = trellis; c.kb = kb; c.fwht = fwht; c.resc = resc; c.skw = skw; c.skg = skg;
      if (I8gCheck(c, K, N) == nullptr) v.push_back(c);
    }
  return v;
}

// ---- the selftest -----------------------------------------------------------------------------------------------------------
static int g_selftest_fails = 0;
static void SCheck(bool ok, const char* what) {
  std::printf("  %-100s %s\n", what, ok ? "ok" : "FAIL");
  if (!ok) ++g_selftest_fails;
}

static void SelftestWmma() {
  std::mt19937 rng(77);
  bool ok = true;
  for (int trial = 0; trial < 4; ++trial) {
    int8_t A[16][16], B[16][16];   // A[row][k], B[k][col]
    int32_t Cin[16][16], ref[16][16];
    for (int i = 0; i < 16; ++i)
      for (int j = 0; j < 16; ++j) {
        A[i][j] = (int8_t)((int)(rng() % 256) - 128 + (trial == 0 ? 0 : 0));
        B[i][j] = (int8_t)((int)(rng() % 256) - 128);
        Cin[i][j] = trial == 0 ? 0 : (int)(rng() % 2001) - 1000;
      }
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 16; ++c) {
        int s = Cin[r][c];
        for (int k = 0; k < 16; ++k) s += (int)A[r][k] * (int)B[k][c];
        ref[r][c] = s;
      }
    int8_t Af[32][8], Bf[32][8];
    int Cf[32][8];
    for (int L = 0; L < 32; ++L)
      for (int e = 0; e < 8; ++e) {
        Af[L][e] = A[L & 15][i8p::FragK16(L, e)];
        Bf[L][e] = B[i8p::FragK16(L, e)][L & 15];
        Cf[L][e] = Cin[8 * (L >> 4) + e][L & 15];
      }
    signed char *dA, *dB;
    int *dC, *dD;
    CK(hipMalloc(&dA, 256)); CK(hipMalloc(&dB, 256)); CK(hipMalloc(&dC, 1024)); CK(hipMalloc(&dD, 1024));
    CK(hipMemcpy(dA, Af, 256, hipMemcpyHostToDevice));
    CK(hipMemcpy(dB, Bf, 256, hipMemcpyHostToDevice));
    CK(hipMemcpy(dC, Cf, 1024, hipMemcpyHostToDevice));
    hipLaunchKernelGGL(WmmaUnit, dim3(1), dim3(32), 0, 0, dA, dB, dC, dD);
    CK(hipDeviceSynchronize());
    int Df[32][8];
    CK(hipMemcpy(Df, dD, 1024, hipMemcpyDeviceToHost));
    for (int L = 0; L < 32; ++L)
      for (int e = 0; e < 8; ++e) ok = ok && Df[L][e] == ref[8 * (L >> 4) + e][L & 15];
    CK(hipFree(dA)); CK(hipFree(dB)); CK(hipFree(dC)); CK(hipFree(dD));
  }
  SCheck(ok, "v_wmma_i32_16x16x16_iu8: D = A B + C (signed x signed) in the layouts the kernels use (4 random tiles)");
}

static float F16F(uint16_t h) { return i8p::F16ToF32(h); }

// returns the GEMM-config count run
static void SelftestGemms(int KB) {
  const int K = 1024, N = 256, M = 256, NKB = K / 128;
  std::mt19937 rng(900 + KB);
  // activations
  std::vector<uint16_t> xh((size_t)M * K);
  std::vector<float> X((size_t)M * K);
  {
    std::normal_distribution<float> nd(0.f, 0.8f);
    for (size_t i = 0; i < xh.size(); ++i) { xh[i] = i8p::F32ToF16(nd(rng)); X[i] = F16F(xh[i]); }
    // a few exact zeros and a dead 128-block (the all-zero group rule)
    for (int k = 0; k < 128; ++k) { xh[(size_t)5 * K + 128 + k] = 0; X[(size_t)5 * K + 128 + k] = 0.f; }
  }
  std::vector<int8_t> Ap;
  std::vector<float> sa;
  i8p::QuantizeActRef(X.data(), K, sa, Ap);
  const std::vector<int8_t> A8 = i8p::PackA8(Ap, K);
  // weights: random pair-grid words -> CPU decode
  std::vector<uint32_t> grid((size_t)K / 16 * (N / 16) * 8 * KB);
  for (uint32_t& w : grid) w = rng();
  const std::vector<uint16_t> Q = trellis_ref::DecodePairGrid(grid, K, N, KB);
  std::vector<int8_t> Wp;
  std::vector<float> sw;
  i8p::QuantizeWeightsRef(Q.data(), K, N, sw, Wp);
  const std::vector<int8_t> W8 = i8p::PackW8(Wp, K, N);
  std::vector<float> svh(N);
  for (int n = 0; n < N; ++n) svh[n] = (rng() & 1 ? -1.f : 1.f) * (0.5f + (rng() % 1000) / 1000.f);

  // device buffers
  uint16_t* d_x;
  signed char *d_a8, *d_ap, *d_w8, *d_wp;
  float *d_sa, *d_sw, *d_svh, *d_ws;
  unsigned *d_words, *d_tk;
  unsigned short* d_c;
  CK(hipMalloc(&d_x, xh.size() * 2)); CK(hipMalloc(&d_a8, (size_t)M * K)); CK(hipMalloc(&d_ap, (size_t)M * K));
  CK(hipMalloc(&d_sa, (size_t)NKB * M * 4)); CK(hipMalloc(&d_words, grid.size() * 4)); CK(hipMalloc(&d_sw, (size_t)NKB * N * 4));
  CK(hipMalloc(&d_w8, (size_t)N * K)); CK(hipMalloc(&d_wp, (size_t)N * K)); CK(hipMalloc(&d_svh, N * 4));
  CK(hipMalloc(&d_ws, (size_t)8 * M * N * 4)); CK(hipMalloc(&d_tk, (size_t)(N / 128) * 4)); CK(hipMalloc(&d_c, (size_t)M * N * 2));
  CK(hipMemset(d_tk, 0, (size_t)(N / 128) * 4));
  CK(hipMemcpy(d_x, xh.data(), xh.size() * 2, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_words, grid.data(), grid.size() * 4, hipMemcpyHostToDevice));
  CK(hipMemcpy(d_svh, svh.data(), N * 4, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(i8g_quant_act, dim3((M * NKB + 7) / 8), dim3(256), 0, 0, d_x, d_a8, d_ap, d_sa, K);
  const long long groups = (long long)(N / 32) * NKB, blocks = (long long)(N / 32) * (K / 16);
  if (KB == 4) {
    hipLaunchKernelGGL(i8g_wscale<4>, dim3((unsigned)((groups + 7) / 8)), dim3(256), 0, 0, d_words, d_sw, K, N);
    CK(hipDeviceSynchronize());
    hipLaunchKernelGGL(i8g_dump_w<4>, dim3((unsigned)((blocks + 7) / 8)), dim3(256), 0, 0, d_words, d_sw, d_w8, d_wp, K, N);
  } else {
    hipLaunchKernelGGL(i8g_wscale<5>, dim3((unsigned)((groups + 7) / 8)), dim3(256), 0, 0, d_words, d_sw, K, N);
    CK(hipDeviceSynchronize());
    hipLaunchKernelGGL(i8g_dump_w<5>, dim3((unsigned)((blocks + 7) / 8)), dim3(256), 0, 0, d_words, d_sw, d_w8, d_wp, K, N);
  }
  CK(hipDeviceSynchronize());
  auto same = [&](const void* dev, const void* host, size_t bytes) {
    std::vector<unsigned char> h(bytes);
    CK(hipMemcpy(h.data(), dev, bytes, hipMemcpyDeviceToHost));
    return std::memcmp(h.data(), host, bytes) == 0;
  };
  char what[160];
  std::snprintf(what, sizeof what, "KB %d: activation quantizer == CPU (A8 fragment layout, plain A, scales; incl. an all-zero block)", KB);
  SCheck(same(d_a8, A8.data(), A8.size()) && same(d_ap, Ap.data(), Ap.size()) && same(d_sa, sa.data(), sa.size() * 4), what);
  std::snprintf(what, sizeof what, "KB %d: scale table (GPU decode, amax, f16 reciprocal) == CPU decode + CPU table", KB);
  SCheck(same(d_sw, sw.data(), sw.size() * 4), what);
  std::snprintf(what, sizeof what, "KB %d: int8 weights (W8 block layout and plain) == CPU decode + CPU quantizer", KB);
  SCheck(same(d_w8, W8.data(), W8.size()) && same(d_wp, Wp.data(), Wp.size()), what);

  // CPU references by scale mask (0 per-128 both; 3 both coarse; 1 activation coarse; 2 weight coarse), with and
  // without the FWHT epilogue (applied below)
  std::vector<double> raw[4];
  for (int mask = 0; mask < 4; ++mask) {
    raw[mask].resize((size_t)M * N);
    i8p::RefGemmInt8(Ap.data(), sa.data(), Wp.data(), sw.data(), M, N, K, mask, raw[mask].data());
  }
  int ran = 0, bad = 0, bitdiff = 0;
  for (int kind = 0; kind < 3; ++kind) {   // 0 dense+fwht, 1 dense plain, 2 trellis
    for (int resc = 0; resc < (kind == 2 ? 4 : 2); ++resc) {
      for (int skw : {2, 4, 8})
        for (int skg : {1, 2, 4, 8}) {
          I8gCfg g;
          g.trellis = kind == 2; g.kb = KB; g.fwht = kind != 1; g.resc = resc; g.skw = skw; g.skg = skg;
          if (I8gCheck(g, K, N)) continue;
          const void* w = g.trellis ? (const void*)d_words : (const void*)d_w8;
          const float os = g.fwht ? kOutScale : 1.0f;
          CK(hipMemset(d_c, 0xFF, (size_t)M * N * 2));
          I8gRun(g, d_a8, d_sa, w, d_sw, d_svh, d_c, d_ws, d_tk, K, N, os, 0);
          CK(hipDeviceSynchronize());
          std::vector<unsigned short> C((size_t)M * N);
          CK(hipMemcpy(C.data(), d_c, C.size() * 2, hipMemcpyDeviceToHost));
          // expected
          long long viol = 0;
          for (int r = 0; r < M; ++r)
            for (int grp = 0; grp < N / 128; ++grp) {
              double v[128], e[128], rms = 0;
              for (int j = 0; j < 128; ++j) v[j] = raw[MaskOfResc(resc)][(size_t)r * N + grp * 128 + j];
              if (g.fwht) i8p::Fwht128(v);
              for (int j = 0; j < 128; ++j) {
                e[j] = g.fwht ? v[j] * (double)svh[grp * 128 + j] * (double)os : v[j] * (double)os;
                rms += e[j] * e[j];
              }
              rms = std::sqrt(rms / 128);
              for (int j = 0; j < 128; ++j) {
                const double got = Bf16ToF(C[(size_t)r * N + grp * 128 + j]);
                viol += std::fabs(got - e[j]) > 0.0085 * std::fabs(e[j]) + 1e-4 * rms;
              }
            }
          ++ran;
          if (viol) {
            ++bad;
            std::printf("    FAIL %s: %lld of %d outputs outside tolerance\n", CfgName(g).c_str(), viol, M * N);
          }
          // the trellis kernel must equal the dense one with the same flags byte for byte (RESC 0 and 1 have a dense twin)
          if (kind == 2 && resc <= 1) {
            I8gCfg dcfg = g;
            dcfg.trellis = false;
            CK(hipMemset(d_c, 0xFF, (size_t)M * N * 2));
            I8gRun(dcfg, d_a8, d_sa, d_w8, d_sw, d_svh, d_c, d_ws, d_tk, K, N, os, 0);
            CK(hipDeviceSynchronize());
            std::vector<unsigned short> D((size_t)M * N);
            CK(hipMemcpy(D.data(), d_c, D.size() * 2, hipMemcpyDeviceToHost));
            if (D != C) { ++bitdiff; std::printf("    trellis != dense (same int8 weights) for %s\n", CfgName(g).c_str()); }
          }
        }
    }
  }
  std::snprintf(what, sizeof what, "KB %d: %d GEMM configurations (dense / dense plain / trellis x RESC 0..3 x skw, skg) match the exact reference", KB, ran);
  SCheck(bad == 0 && ran > 0, what);
  std::snprintf(what, sizeof what, "KB %d: the trellis kernel is byte-identical to the dense kernel on the same int8 weights", KB);
  SCheck(bitdiff == 0, what);
  CK(hipFree(d_x)); CK(hipFree(d_a8)); CK(hipFree(d_ap)); CK(hipFree(d_sa)); CK(hipFree(d_words)); CK(hipFree(d_sw)); CK(hipFree(d_w8));
  CK(hipFree(d_wp)); CK(hipFree(d_svh)); CK(hipFree(d_ws)); CK(hipFree(d_tk)); CK(hipFree(d_c));
}

// ---- timing -----------------------------------------------------------------------------------------------------------------
struct Job {
  std::string name, kind;                 // kind: f16ship, f16alt, dense, denseplain, trellis, denseC, trellisC, quant
  I8gCfg cfg;
  std::function<void(int)> launch;        // launch(copy)
  std::vector<double> t;                  // us per call, one entry per round (a batch of launches / batch)
  bool verified = true;
};
static void RunJobs(std::vector<Job>& jobs, int batch, int rounds) {
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  for (Job& j : jobs) for (int i = 0; i < 2; ++i) j.launch(i % batch);
  CK(hipDeviceSynchronize());
  for (int r = 0; r < rounds; ++r)
    for (Job& j : jobs) {
      CK(hipEventRecord(e0, 0));
      for (int i = 0; i < batch; ++i) j.launch(i);
      CK(hipEventRecord(e1, 0));
      CK(hipEventSynchronize(e1));
      float ms = 0;
      CK(hipEventElapsedTime(&ms, e0, e1));
      j.t.push_back(1000.0 * ms / batch);
    }
  CK(hipEventDestroy(e0));
  CK(hipEventDestroy(e1));
}
static std::vector<double> Ratio(const std::vector<double>& num, const std::vector<double>& den) {
  std::vector<double> r;
  for (size_t i = 0; i < num.size(); ++i) r.push_back(num[i] / den[i]);
  return r;
}
static std::vector<double> SumRounds(const std::vector<const std::vector<double>*>& v, const std::vector<int>& w) {
  size_t n = (size_t)-1;
  for (auto* x : v) n = std::min(n, x->size());
  std::vector<double> r(n, 0.0);
  for (size_t k = 0; k < v.size(); ++k)
    for (size_t i = 0; i < n; ++i) r[i] += w[k] * (*v[k])[i];
  return r;
}

struct ClassResult {
  std::string name;
  int count = 0, N = 0, K = 0, KB = 0;
  std::vector<double> ship, f16best, dense, denseplain, trellis, denseC, trellisC, trellisA, trellisW, quant;
  std::string n_ship, n_f16best, n_dense, n_denseplain, n_trellis, n_denseC, n_trellisC, n_trellisA, n_trellisW;
  bool ok = true;     // every reported kernel verified
};

static void ShapeRun(const Shape& s, int kb, int rounds, int batch, bool do_verify, bool do_time, bool do_resc, bool verbose,
                     FILE* json, bool& json_first, std::map<std::string, ClassResult>& results, int& fails) {
  Case c;
  AllocCase(c, s.N, s.K, kb, do_time ? batch : 1);
  FillCase(c, 1);
  const Plan plan = ShippedPlan(s, kb);
  if (!plan.ok) { std::printf("---- %s KB%d: no shipped M = 256 plan (skipped)\n", s.name, kb); FreeCase(c); return; }
  std::printf("---- %s KB%d  N=%d K=%d  shipped plan: SK %d SKG %d skw %d (x%d)\n", s.name, kb, s.N, s.K, plan.SK, plan.SKG, plan.skw, plan.SK / plan.skw);

  // the shipped f16 kernel on copy 0 (the reference for the fake-quant-level difference)
  LaunchShipped(c, plan, 0, c.c_f16);
  CK(hipDeviceSynchronize());
  const std::vector<unsigned short> Cf16 = ReadC(c.c_f16, s.N);

  Verifier V;
  std::map<std::string, bool> verified;       // by config name
  if (do_verify) {
    V.Build(c);
    // the quantization itself: how far is the w8a8 product from the f16 product (the coarse view of "fake-quant level")
    std::vector<I8gCfg> all;
    for (int kind = 0; kind < 3; ++kind)
      for (int resc = 0; resc < (!do_resc ? 1 : (kind == 2 ? 4 : 2)); ++resc) {
        for (const I8gCfg& g : Sweep(s.K, s.N, kind == 2, kb, kind != 1, resc)) all.push_back(g);
      }
    int nbad = 0, nbit = 0, nrun = 0;
    double worst_rel = 0, best_rel = 1e9;
    for (const I8gCfg& g : all) {
      CK(hipMemset(c.c_a, 0xFF, (size_t)256 * s.N * 2));
      LaunchI8(c, g, 0, c.c_a);
      CK(hipDeviceSynchronize());
      const std::vector<unsigned short> C = ReadC(c.c_a, s.N);
      const auto r = V.Check(C, g.fwht, g.resc, g.fwht ? kOutScale : 1.0f);
      const bool ok = r.viol == 0;
      verified[CfgName(g)] = ok;
      ++nrun;
      if (!ok) { ++nbad; std::printf("    VERIFY FAIL %-34s %lld of %lld outputs outside tolerance (max dev %.3g rms)\n", CfgName(g).c_str(), r.viol, r.total, r.max_dev); }
      if (g.trellis) {
        if (g.resc <= 1) {
          I8gCfg d = g; d.trellis = false;
          LaunchI8(c, d, 0, c.c_b);
          CK(hipDeviceSynchronize());
          const std::vector<unsigned short> D = ReadC(c.c_b, s.N);
          if (D != C) { ++nbit; std::printf("    trellis != dense (same int8 weights): %s\n", CfgName(g).c_str()); }
        }
        if (g.resc == 0) {
          const double rel = V.RelDiff(C, Cf16);
          worst_rel = std::max(worst_rel, rel);
          best_rel = std::min(best_rel, rel);
        }
      }
    }
    std::printf("  verify: %d configurations against the exact reference (5 groups x 128 columns x 256 rows each): %s; trellis vs dense byte-identical: %s\n",
                nrun, nbad ? "FAILURES" : "all pass", nbit ? "NO" : "yes");
    if (best_rel < 1e8)
      std::printf("  trellis int8 vs shipped f16 trellis output: relative RMS difference %.4f .. %.4f (w8a8 fake-quant level; ~0.5-2%% expected for Gaussian data)\n", best_rel, worst_rel);
    fails += nbad + nbit;
    // the shipped kernel itself against the f16 exact? (not the subject) -- sanity: not all zero
    double nz = 0;
    for (size_t i = 0; i < Cf16.size(); i += 97) nz += Bf16ToF(Cf16[i]) != 0.f;
    if (nz == 0) { std::printf("    DATA PROBLEM: the shipped kernel's output is all zero\n"); ++fails; }
    V.Free();
  }
  if (!do_time) { FreeCase(c); return; }

  // ---- jobs ----
  std::vector<Job> jobs;
  {
    Job j; j.name = "f16 shipped plan"; j.kind = "f16ship";
    j.launch = [&, plan](int copy) { LaunchShipped(c, plan, copy, c.c_f16); };
    jobs.push_back(j);
    // the other legal skw for the shipped (SK, SKG): the best-of sweep that keeps the baseline honest
    for (int skw : {2, 4, 8}) {
      if (skw == plan.skw || skw > plan.SK || plan.SK % skw) continue;
      if (r4d_gemm_trellis_nt_m256_check(256, s.K, s.N, s.N, kb, plan.SK, 1, plan.SKG, 4, skw)) continue;
      Plan p2 = plan; p2.skw = skw;
      Job a; a.name = "f16 alt skw" + std::to_string(skw); a.kind = "f16alt";
      a.launch = [&, p2](int copy) { LaunchShipped(c, p2, copy, c.c_f16); };
      jobs.push_back(a);
    }
  }
  auto add = [&](const std::vector<I8gCfg>& cfgs, const char* kind) {
    for (const I8gCfg& g : cfgs) {
      Job j; j.name = CfgName(g); j.kind = kind; j.cfg = g;
      auto it = verified.find(j.name);
      j.verified = !do_verify || (it != verified.end() && it->second);
      j.launch = [&, g](int copy) { LaunchI8(c, g, copy, c.c_a); };
      jobs.push_back(j);
    }
  };
  add(Sweep(s.K, s.N, false, kb, true, 0), "dense");
  add(Sweep(s.K, s.N, false, kb, false, 0), "denseplain");
  add(Sweep(s.K, s.N, true, kb, true, 0), "trellis");
  if (do_resc) {
    add(Sweep(s.K, s.N, false, kb, true, 1), "denseC");
    add(Sweep(s.K, s.N, true, kb, true, 1), "trellisC");
    add(Sweep(s.K, s.N, true, kb, true, 2), "trellisA");
    add(Sweep(s.K, s.N, true, kb, true, 3), "trellisW");
  }
  {
    Job j; j.name = "quantize A (K=" + std::to_string(s.K) + ")"; j.kind = "quant";
    const int NKB = s.K / 128;
    j.launch = [&, NKB](int) { hipLaunchKernelGGL(i8g_quant_act, dim3((256 * NKB + 7) / 8), dim3(256), 0, 0, c.a16, c.a8, (signed char*)nullptr, c.sa, c.K); };
    jobs.push_back(j);
  }
  RunJobs(jobs, batch, rounds);
  CK(hipDeviceSynchronize());

  // ---- report ----
  const double flops = 2.0 * 256 * s.N * s.K;
  auto best = [&](const char* kind, bool verified_only) -> const Job* {
    const Job* b = nullptr;
    for (const Job& j : jobs)
      if (j.kind == kind && (!verified_only || j.verified) && (!b || Median(j.t) < Median(b->t))) b = &j;
    return b;
  };
  const Job* ship = best("f16ship", false);
  const Job* f16alt = best("f16alt", false);
  const Job* f16best = (f16alt && Median(f16alt->t) < Median(ship->t)) ? f16alt : ship;
  const Job* dense = best("dense", true);
  const Job* dplain = best("denseplain", true);
  const Job* trel = best("trellis", true);
  const Job* denseC = best("denseC", true);
  const Job* trelC = best("trellisC", true);
  const Job* trelA = best("trellisA", true);
  const Job* trelW = best("trellisW", true);
  const Job* quant = best("quant", false);
  const double t_ship = Median(ship->t);
  for (const Job& j : jobs) {
    const bool show = verbose || &j == ship || &j == f16best || &j == dense || &j == dplain || &j == trel || &j == denseC || &j == trelC ||
                      &j == trelA || &j == trelW || &j == quant;
    if (!show) continue;
    const double med = Median(j.t);
    std::printf("  %-34s %s %9.1f us [%8.1f..%8.1f] %7.1f TF/s", j.name.c_str(), j.verified ? " " : "!", med, MinOf(j.t), MaxOf(j.t), j.kind == "quant" ? 0.0 : flops / (med * 1e-6) / 1e12);
    if (j.kind != "quant" && j.kind != "f16ship") std::printf("  x%.3f vs shipped plan", Median(Ratio(ship->t, j.t)));
    if (&j == dense) std::printf("  [best dense, trellis epilogue]");
    if (&j == dplain) std::printf("  [best dense, plain epilogue: the ceiling]");
    if (&j == trel) std::printf("  [best trellis int8]");
    if (&j == denseC) std::printf("  [best dense, both scales coarse: bound]");
    if (&j == trelC) std::printf("  [best trellis, both scales coarse: bound]");
    if (&j == trelA) std::printf("  [best trellis, activation scale per row only: bound]");
    if (&j == trelW) std::printf("  [best trellis, weight scale per column only: bound]");
    if (&j == f16best && &j != ship) std::printf("  [best f16 config]");
    std::printf("\n");
    if (json && j.kind != "f16alt") {
      std::fprintf(json, "%s {\"kind\":\"time\",\"shape\":\"%s\",\"kb\":%d,\"job\":\"%s\",\"class\":\"%s\",\"verified\":%s,\"us_median\":%.2f,\"tflops\":%.2f,\"speedup_vs_shipped\":%.4f}\n",
                   json_first ? "" : ",", s.name, kb, j.name.c_str(), j.kind.c_str(), j.verified ? "true" : "false", med,
                   j.kind == "quant" ? 0.0 : flops / (med * 1e-6) / 1e12, j.kind == "quant" ? 0.0 : Median(Ratio(ship->t, j.t)));
      json_first = false;
    }
  }
  ClassResult r;
  r.name = s.name; r.count = s.count; r.N = s.N; r.K = s.K; r.KB = kb;
  r.ship = ship->t; r.f16best = f16best->t; r.n_ship = ship->name; r.n_f16best = f16best->name;
  if (dense) { r.dense = dense->t; r.n_dense = dense->name; }
  if (dplain) { r.denseplain = dplain->t; r.n_denseplain = dplain->name; }
  if (trel) { r.trellis = trel->t; r.n_trellis = trel->name; }
  if (denseC) { r.denseC = denseC->t; r.n_denseC = denseC->name; }
  if (trelC) { r.trellisC = trelC->t; r.n_trellisC = trelC->name; }
  if (trelA) { r.trellisA = trelA->t; r.n_trellisA = trelA->name; }
  if (trelW) { r.trellisW = trelW->t; r.n_trellisW = trelW->name; }
  if (quant) r.quant = quant->t;
  r.ok = dense && dplain && trel;
  results[std::string(s.name) + "/" + std::to_string(kb)] = r;
  (void)t_ship;
  FreeCase(c);
}

// combined report over classes (weights = layer counts, or 1 for the MLP pair)
static void Combined(const char* title, const std::map<std::string, ClassResult>& res, const std::vector<std::string>& names, bool weighted,
                     int kb, FILE* json, bool& json_first, bool decide) {
  std::vector<const std::vector<double>*> ship, f16b, dn, dp, tr, dnC, trC, trA, trW, qa;
  std::vector<int> w;
  std::string picks;
  bool haveC = true;
  for (const std::string& n : names) {
    auto it = res.find(n + "/" + std::to_string(kb));
    if (it == res.end() || !it->second.ok) return;
    const ClassResult& r = it->second;
    ship.push_back(&r.ship); f16b.push_back(&r.f16best); dn.push_back(&r.dense); dp.push_back(&r.denseplain); tr.push_back(&r.trellis); qa.push_back(&r.quant);
    if (r.denseC.empty() || r.trellisC.empty() || r.trellisA.empty() || r.trellisW.empty()) haveC = false;
    else { dnC.push_back(&r.denseC); trC.push_back(&r.trellisC); trA.push_back(&r.trellisA); trW.push_back(&r.trellisW); }
    w.push_back(weighted ? r.count : 1);
    picks += (picks.empty() ? "" : " + ") + n + ":[" + r.n_trellis + "]";
  }
  const auto S = SumRounds(ship, w), F = SumRounds(f16b, w), Dn = SumRounds(dn, w), Dp = SumRounds(dp, w), T = SumRounds(tr, w);
  // activation quantizer unfused: one per linear that has its own A (the MLP pair: gate_up and down each read their own)
  const auto Qa = SumRounds(qa, w);
  std::vector<double> Tq(T.size());
  for (size_t i = 0; i < T.size(); ++i) Tq[i] = T[i] + Qa[i];
  std::printf("==== %s KB%d%s\n", title, kb, weighted ? " (layer-weighted, one forward pass of the model's linears at M = 256)" : "");
  std::printf("     shipped f16 plan %9.1f us | best f16 config %9.1f us | dense+epi %9.1f us | dense plain %9.1f us | trellis int8 %9.1f us | A quantizer (unfused) %7.1f us\n",
              Median(S), Median(F), Median(Dn), Median(Dp), Median(T), Median(Qa));
  const auto r_t = Ratio(S, T), r_tq = Ratio(S, Tq), r_tb = Ratio(F, T), r_d = Ratio(S, Dn), r_dp = Ratio(S, Dp);
  std::printf("     speedup vs shipped plan: trellis int8 x%.3f [%.3f..%.3f]; with the unfused A quantizer added x%.3f; vs best f16 config x%.3f; dense+epilogue x%.3f; dense plain x%.3f\n",
              Median(r_t), MinOf(r_t), MaxOf(r_t), Median(r_tq), Median(r_tb), Median(r_d), Median(r_dp));
  double rc_t = 0, rc_d = 0, ra_t = 0, rw_t = 0;
  if (haveC) {
    const auto TC = SumRounds(trC, w), DC = SumRounds(dnC, w), TA = SumRounds(trA, w), TW = SumRounds(trW, w);
    rc_t = Median(Ratio(S, TC)); rc_d = Median(Ratio(S, DC)); ra_t = Median(Ratio(S, TA)); rw_t = Median(Ratio(S, TW));
    std::printf("     SPEED BOUNDS with coarser scales (NOT the per-128 math; each needs its own accuracy gate): trellis int8 with both scales coarse x%.3f (dense x%.3f), "
                "activation scale per row only x%.3f, weight scale per column only x%.3f, vs shipped plan\n", rc_t, rc_d, ra_t, rw_t);
  }
  if (decide) {
    const double sp = Median(r_t);
    std::printf("     DECISION RULE on this pair: GO if trellis int8 >= 1.40x, STOP if < 1.15x  ->  %s (x%.3f)\n",
                sp >= 1.40 ? "GO" : (sp < 1.15 ? "STOP" : "INCONCLUSIVE (between 1.15x and 1.40x)"), sp);
    const double sd = Median(r_dp);
    std::printf("     dense ceiling rule: dense int8 (plain) >= 1.5x shipped?  %s (x%.3f; with the trellis epilogue x%.3f)\n",
                sd >= 1.5 ? "yes" : "NO -- the idea is dead whatever the decode costs", sd, Median(r_d));
  }
  if (json) {
    std::fprintf(json, "%s {\"kind\":\"combined\",\"title\":\"%s\",\"kb\":%d,\"shipped_us\":%.2f,\"trellis_int8_us\":%.2f,\"speedup\":%.4f,\"speedup_with_aquant\":%.4f,\"speedup_vs_best_f16\":%.4f,\"dense_plain_speedup\":%.4f,\"dense_epi_speedup\":%.4f,\"coarse_trellis_speedup\":%.4f,\"coarse_dense_speedup\":%.4f,\"act_row_only_speedup\":%.4f,\"wgt_col_only_speedup\":%.4f}\n",
                 json_first ? "" : ",", title, kb, Median(S), Median(T), Median(r_t), Median(r_tq), Median(r_tb), Median(r_dp), Median(r_d), rc_t, rc_d, ra_t, rw_t);
    json_first = false;
  }
  (void)picks;
}

int main(int argc, char** argv) {
  std::string outp, shape = "all", mode = "all", kbs = "both";
  int rounds = 11, batch = 6, simds = 128;
  bool quick = false, noverify = false, no_resc = false, verbose = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
    else if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) batch = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--simds") && i + 1 < argc) simds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--shape") && i + 1 < argc) shape = argv[++i];
    else if (!std::strcmp(argv[i], "--kb") && i + 1 < argc) kbs = argv[++i];
    else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
    else if (!std::strcmp(argv[i], "--quick")) quick = true;
    else if (!std::strcmp(argv[i], "--noverify")) noverify = true;
    else if (!std::strcmp(argv[i], "--no-resc")) no_resc = true;
    else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  if (quick) { rounds = 5; batch = 3; }
  hipDeviceProp_t prop;
  CK(hipGetDeviceProperties(&prop, 0));
  std::printf("device %s (%s), multiProcessorCount %d, mode %s, rounds %d, batch %d\n", prop.name, prop.gcnArchName, prop.multiProcessorCount,
              mode.c_str(), rounds, batch);
  FILE* json = outp.empty() ? nullptr : std::fopen(outp.c_str(), "wb");
  bool json_first = true;
  if (json) std::fprintf(json, "{\"device\":\"%s\",\"mode\":\"%s\",\"rounds\":%d,\"batch\":%d,\"rows\":[\n", prop.name, mode.c_str(), rounds, batch);
  int fails = 0;
  const bool do_probe = mode == "all" || mode == "probe";
  const bool do_selftest = mode == "all" || mode == "selftest";
  const bool do_verify = (mode == "all" || mode == "verify" || mode == "time") && !noverify;
  const bool do_time = mode == "all" || mode == "time";
  const bool do_shapes = mode == "all" || mode == "verify" || mode == "time";
  if (do_probe) {
    RunProbe(prop, simds, quick ? 3 : 5);
    if (json)
      for (auto& kv : g_probe) {
        std::fprintf(json, "%s {\"kind\":\"probe\",\"wmma\":\"%s\",\"tops\":%.2f,\"sclk_mhz\":%.0f,\"clk_per_wmma_per_simd\":%.3f}\n", json_first ? "" : ",", kv.first.c_str(), kv.second.tops, kv.second.sclk, kv.second.clk_per_wmma_simd);
        json_first = false;
      }
  }
  if (do_selftest) {
    std::printf("== selftest (small shapes, CPU references) ==\n");
    SelftestWmma();
    for (int kb : {4, 5}) SelftestGemms(kb);
    std::printf("selftest: %s (%d failing)\n", g_selftest_fails ? "FAILED" : "all pass", g_selftest_fails);
    fails += g_selftest_fails;
    if (g_selftest_fails) {
      std::printf("selftest failed: the shape runs below would be meaningless; stopping\n");
      if (json) { std::fprintf(json, "]}\n"); std::fclose(json); }
      return 1;
    }
  }
  if (do_shapes) {
    std::vector<Shape> shapes;
    for (const Shape& s : kShapes)
      if (shape == "all" || shape == s.name) shapes.push_back(s);
    if (shapes.empty()) { std::fprintf(stderr, "no such shape %s\n", shape.c_str()); return 2; }
    std::vector<int> kbl;
    if (kbs == "4" || kbs == "both") kbl.push_back(4);
    if (kbs == "5" || kbs == "both") kbl.push_back(5);
    std::map<std::string, ClassResult> results;
    for (int kb : kbl) {
      for (const Shape& s : shapes) ShapeRun(s, kb, rounds, batch, do_verify, do_time, !no_resc, verbose, json, json_first, results, fails);
      if (do_time) {
        Combined("MLP PAIR (mlp.gate_up + mlp.down)", results, {"gate_up", "down"}, false, kb, json, json_first, true);
        Combined("ALL SEVEN LINEAR CLASSES", results, {"gate_up", "down", "qkv", "z", "out", "qg", "kv"}, true, kb, json, json_first, false);
      }
    }
  }
  if (json) { std::fprintf(json, "]}\n"); std::fclose(json); }
  std::printf("%s (%d failing checks)\n", fails == 0 ? "ALL OK" : "FAILED", fails);
  return fails == 0 ? 0 : 1;
}
