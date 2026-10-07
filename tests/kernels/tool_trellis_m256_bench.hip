// tests/kernels/tool_trellis_m256_bench.hip -- standalone (host + device) bench and bit-identity check of
// the PROTOTYPE bigger-M trellis GEMM (third_party/libr4d/r4d_gemm_trellis_nt_m256.hip) against the
// SHIPPED M = 64 kernel (r4d_gemm_trellis_nt_m64.hip), both through their real host entries.
//
//   verify  random weights / A / svh (several seeds), then C[256][N] bf16 of ONE M = 256 launch is compared
//           byte for byte with C of four M = 64 launches (one per 64-row slice of the same A) at EVERY
//           listed shipped-kernel tuning with the same (SK, SKG); M = 128 against two. A second pass per
//           seed splits the columns (n_split = N / 2) between two different activation matrices.
//   time    us per call, us per row, and the speedups of the M = 256 launch against
//             cold  4 x one M = 64 launch, each launch on a weight copy no launch of the batch has read (what
//                   64-row chunks cost today: every layer's weights stream from DRAM once per chunk),
//             same  four M = 64 launches back to back on the SAME weights (a Model chunk of 256 rows with
//                   today's kernel; the MALL / L2 may serve launches 2-4),
//           each against three M = 64 baselines: the SHIPPED tuning-table row (src/model/
//           gemm_tuning_table_trellis.inc, read through trellis_tuning_rows.hpp, M = 64 band), the fastest of
//           the listed tunings of the same (SK, SKG) as the M = 256 config (the only ones it can be
//           bit-identical to), and the fastest of every listed tuning. Every round times every job once (a
//           batch of launches, each on a different weight copy), so the ratios are medians of per-round
//           ratios. The MLP pair (mlp.gate_up + mlp.down) and a whole-chunk estimate are printed when the
//           shapes they need have run.
// It never touches the tuning table (no --inc-out here) and writes nothing but the --out JSON.
//
// --retune (docs/trellis-m256.md "Retuning the plan"): the sweep that decides a change of the M = 64 rows.
//   --retune --mode verify  every legal M = 256 (SK, SKG, SKW) of every family against a few stock M = 64 tunings
//                           of the SAME (SK, SKG): the identity a retuned row would rely on (correctness, any
//                           device, not a timing).
//   --retune --mode time    per class and rate: the shipped plan against every family, M = 256 and M = 64, FREE /
//                           TRADE verdicts, the .inc row to paste, a layer-weighted summary (idle device only).
//                           --tol64 <pct> (default 1) is the M = 64 slowdown a FREE family may have, --min-gain
//                           <pct> (default 1) the M = 256 speedup it needs.
//
// Build: tests/kernels/build_m256_bench.ps1 (hipcc, objects under <R4DX_MODELS_ROOT>\r4dx\linear\A2\obj).
// Run (one device; HIP_VISIBLE_DEVICES set by the caller; every set is bounded):
//   tool_trellis_m256_bench.exe [--shape gate_up|down|qkv|z|out|qg|kv|all] [--kb 4|5|both]
//                               [--mode verify|time|both] [--seeds 4] [--rounds 11] [--batch 6]
//                               [--m128] [--verbose] [--out x.json]
//                               [--retune [--tol64 PCT] [--min-gain PCT]]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4d.h"
#include "trellis_tuning_rows.hpp"

#define CK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
  std::fprintf(stderr, "HIP error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, hipGetErrorString(e_)); \
  std::exit(1); } } while (0)

__device__ __forceinline__ uint32_t Hash32(uint64_t i, uint32_t seed) {
  uint32_t h = (uint32_t)i * 0x9E3779B9u ^ (uint32_t)(i >> 32) * 0x85EBCA6Bu ^ seed;
  h ^= h >> 16; h *= 0x7FEB352Du; h ^= h >> 15; h *= 0x846CA68Bu; h ^= h >> 16;
  return h;
}
__global__ void FillW(uint32_t* p, int64_t n, uint32_t seed) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) p[i] = Hash32(i, seed);
}
// f16 with a random sign and |x| in [2^-3, 1)
__global__ void FillA(unsigned short* p, int64_t n, uint32_t seed) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint32_t h = Hash32(i, seed);
    p[i] = (unsigned short)((h & 0x8000u) | (0x3000u + ((h >> 4) & 0x0BFFu)));
  }
}
// svh: fp32 in [0.5, 1.5) with a random sign
__global__ void FillSvh(float* p, int64_t n, uint32_t seed) {
  const int64_t stride = (int64_t)gridDim.x * blockDim.x;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    const uint32_t h = Hash32(i, seed);
    const float m = 0.5f + (float)(h & 0xFFFFu) / 65536.f;
    p[i] = (h & 0x10000u) ? -m : m;
  }
}

struct Shape { const char* name; int N, K; int count; };   // count: layers per model using this linear class
struct Stock { int WV, SK, NP, SKG, U, NT, MT; };
struct Big { int M, NP, U, SK, SKG, SKW; };   // SK = the shipped slice count, SKW = slices resident per workgroup
static bool operator==(const Stock& a, const Stock& b) {
  return a.WV == b.WV && a.SK == b.SK && a.NP == b.NP && a.SKG == b.SKG && a.U == b.U && a.NT == b.NT && a.MT == b.MT;
}

static const Shape kShapes[] = {
    {"gate_up", 34816, 5120, 64}, {"down", 5120, 17408, 64}, {"qkv", 10240, 5120, 48}, {"z", 6144, 5120, 48},
    {"out", 5120, 6144, 64},      {"qg", 12288, 5120, 16},   {"kv", 1024, 5120, 32},
};

struct Bufs {
  unsigned char* w = nullptr;
  size_t w_bytes_each = 0;
  unsigned short* a = nullptr;             // two [256][K] matrices (A0 then A1)
  float* svh = nullptr;
  unsigned short* c_stock = nullptr;
  unsigned short* c_big = nullptr;
  float* ws_stock = nullptr;
  float* ws_big = nullptr;
  unsigned* tk_stock = nullptr;
  unsigned* tk_big = nullptr;
  int batch = 0;
  int N = 0, K = 0, KB = 4;
  int n_split = 0;                         // = N unless the split pass of verify runs
  const unsigned short* a1 = nullptr;      // 0 = same as a
};
static const float kOutScale = 0.0883883f;

static void LaunchStock(const Bufs& b, const Stock& t, const unsigned char* w, int row0, int M) {
  r4d_gemm_trellis_nt_m64((int64_t)(b.a + (size_t)row0 * b.K), (int64_t)(b.a1 ? b.a1 + (size_t)row0 * b.K : nullptr),
                          b.n_split, (int64_t)w, (int64_t)b.svh, (int64_t)(b.c_stock + (size_t)row0 * b.N),
                          (int64_t)b.ws_stock, (int64_t)b.tk_stock, M, b.K, b.N, b.KB, t.WV, t.SK, t.MT, t.NP,
                          t.SKG, t.U, t.NT, kOutScale, 0);
}
static void LaunchBig(const Bufs& b, const Big& g, const unsigned char* w) {
  r4d_gemm_trellis_nt_m256((int64_t)b.a, (int64_t)b.a1, b.n_split, (int64_t)w, (int64_t)b.svh, (int64_t)b.c_big,
                           (int64_t)b.ws_big, (int64_t)b.tk_big, g.M, b.K, b.N, b.KB, g.SK, g.NP, g.SKG, g.U,
                           kOutScale, 0, g.SKW);
}

static std::string StockName(const Stock& t) {
  char nm[96];
  std::snprintf(nm, sizeof nm, "WV%d NP%d SK%d SKG%d U%d MT%d NT%d", t.WV, t.NP, t.SK, t.SKG, t.U, t.MT, t.NT);
  return nm;
}
static std::string BigName(const Big& g) {
  char nm[96];
  std::snprintf(nm, sizeof nm, "M%d NP%d U%d SK%d(W%d x%d) SKG%d", g.M, g.NP, g.U, g.SK, g.SKW, g.SK / g.SKW, g.SKG);
  return nm;
}

// The shipped M = 64 row (tuning table, kTrellis, this N, K and rate), if there is one.
static bool TableRow(const Shape& s, int kb, Stock& out) {
  for (const trellis_rows::GemmTuningRow& r : trellis_rows::kGemmTuningTable)
    if (r.layout == trellis_rows::Layout::kTrellis && r.N == s.N && r.K == s.K && r.M == 64 && r.rate == kb) {
      out = {r.tuning.WV, r.tuning.SK, r.tuning.NPW, r.tuning.SKG, r.tuning.U, r.tuning.NT, r.tuning.MB};
      return true;
    }
  return false;
}

// M = 256 (or 128) configurations: every (SKW, SK / SKW) the kernel has for the shipped row's (SK, SKG), the
// only family that is bit-identical to the SHIPPED M = 64 row.
static std::vector<Big> BigList(const Shape& s, int kb, bool m128, const Stock& table) {
  std::vector<Big> v;
  const int SK = table.SK;
  // Only the instantiated configurations (r4d_gemm_trellis_nt_m256_check; e.g. KB 5 has none with SK / skw > 1).
  auto ok = [&](const Big& g) {
    return r4d_gemm_trellis_nt_m256_check(g.M, s.K, s.N, s.N, kb, g.SK, g.NP, g.SKG, g.U, g.SKW) == nullptr;
  };
  for (int skw : {2, 4, 8}) {
    if (SK % skw || skw > SK) continue;
    const int ph = SK / skw;
    if (ph > 4) continue;
    if (!m128) {
      if (256 / 64 * skw > 32) continue;
      const Big g{256, 1, 4, SK, table.SKG, skw};
      if (ok(g)) v.push_back(g);
    } else {
      if (ph > 1) continue;
      const Big g1{128, 1, 2, SK, table.SKG, skw}, g2{128, 2, 1, SK, table.SKG, skw};
      if (ok(g1)) v.push_back(g1);
      if (skw <= 4 && ok(g2)) v.push_back(g2);
    }
  }
  return v;
}

// --retune: EVERY (SK, SKG, SKW) the M = 256 kernel has for this shape, not only the shipped row's (SK, SKG).
// A config of another (SK, SKG) than the shipped M = 64 row is bit-identical to the 64-row kernel only at that
// same (SK, SKG), i.e. only if the M = 64 row is changed with it (docs/trellis-m256.md "Retuning the plan").
static std::vector<Big> BigListAll(const Shape& s, int kb) {
  std::vector<Big> v;
  for (int SK : {2, 4, 8, 16})
    for (int SKG : {1, 2, 4})
      for (int skw : {2, 4, 8}) {
        if (SK % skw || skw > SK) continue;
        if (SK / skw > 4) continue;
        if (256 / 64 * skw > 32) continue;
        const Big g{256, 1, 4, SK, SKG, skw};
        if (r4d_gemm_trellis_nt_m256_check(g.M, s.K, s.N, s.N, kb, g.SK, g.NP, g.SKG, g.U, g.SKW) == nullptr)
          v.push_back(g);
      }
  return v;
}

// The K slices resident per workgroup that PlanTrellisM256 (src/model/linear.cpp) picks for a row of this
// (SK, SKG): a copy of its rule (preferred value first, first instantiated one wins), so the sweep prices the
// configuration production would run, not only the best SKW. 0 when none is instantiated.
static int PlanSkw(const Shape& s, int kb, int SK, int SKG) {
  const int pref = SK <= 2 ? 2 : (SK == 4 && kb == 4 && s.N == 34816) ? 2 : 4;
  for (int skw : {pref, 4, 8, 2}) {
    if (skw > SK) continue;
    if (r4d_gemm_trellis_nt_m256_check(256, s.K, s.N, s.N, kb, SK, 1, SKG, 4, skw) == nullptr) return skw;
  }
  return 0;
}

static void Alloc(Bufs& b, const Shape& s, int kb, int batch) {
  b.N = s.N; b.K = s.K; b.KB = kb; b.batch = batch; b.n_split = s.N; b.a1 = nullptr;
  b.w_bytes_each = (size_t)s.N * s.K * kb / 8;
  CK(hipMalloc(&b.w, b.w_bytes_each * batch));
  CK(hipMalloc(&b.a, (size_t)2 * 256 * s.K * 2 + 4096));
  CK(hipMalloc(&b.svh, (size_t)s.N * 4));
  CK(hipMalloc(&b.c_stock, (size_t)256 * s.N * 2));
  CK(hipMalloc(&b.c_big, (size_t)256 * s.N * 2));
  CK(hipMalloc(&b.ws_stock, (size_t)4 * 64 * s.N * 4));
  CK(hipMalloc(&b.ws_big, (size_t)4 * 256 * s.N * 4));
  CK(hipMalloc(&b.tk_stock, (size_t)(s.N / 128) * 4));
  CK(hipMalloc(&b.tk_big, (size_t)(s.N / 128) * 4));
  CK(hipMemset(b.tk_stock, 0, (size_t)(s.N / 128) * 4));
  CK(hipMemset(b.tk_big, 0, (size_t)(s.N / 128) * 4));
}
static void Free(Bufs& b) {
  CK(hipFree(b.w)); CK(hipFree(b.a)); CK(hipFree(b.svh)); CK(hipFree(b.c_stock)); CK(hipFree(b.c_big));
  CK(hipFree(b.ws_stock)); CK(hipFree(b.ws_big)); CK(hipFree(b.tk_stock)); CK(hipFree(b.tk_big));
}
static void Fill(Bufs& b, uint32_t seed) {
  hipLaunchKernelGGL(FillW, dim3(8192), dim3(256), 0, 0, (uint32_t*)b.w, (int64_t)(b.w_bytes_each * b.batch / 4), seed * 7919u + 1u);
  hipLaunchKernelGGL(FillA, dim3(2048), dim3(256), 0, 0, b.a, (int64_t)2 * 256 * b.K, seed * 104729u + 3u);
  hipLaunchKernelGGL(FillSvh, dim3(256), dim3(256), 0, 0, b.svh, (int64_t)b.N, seed * 1299709u + 5u);
  CK(hipDeviceSynchronize());
}

static int TicketsNonZero(const unsigned* d, int n) {
  std::vector<unsigned> h(n);
  CK(hipMemcpy(h.data(), d, (size_t)n * 4, hipMemcpyDeviceToHost));
  int bad = 0;
  for (unsigned x : h) bad += x != 0;
  return bad;
}

// Every shipped-kernel tuning that is legal for this shape and shares (SK, SKG) with one of `fams`, plus the
// table row; found by launching each once (the host check throws for an illegal one). MT = 4 except the row's.
// `level` 0 is the original set (WV x NP x U x NT, MT 4); 1 (--retune verify) is a small one, WV x NP at U 1,
// NT 0, MT 4 (every family of the sweep is compared, so the reference copies must stay few); 2 (--retune
// time, the stage-1 screen of the M = 64 side) adds MT 2 as well, the way docs/trellis-kernel.md's ptune did.
static std::vector<Stock> Candidates(Bufs& b, const Stock* table, const std::vector<std::pair<int, int>>& fams,
                                     int level = 0) {
  std::vector<Stock> v;
  auto add = [&](const Stock& t) {
    for (const Stock& x : v) if (x == t) return;
    try {
      LaunchStock(b, t, b.w, 0, 64);
      CK(hipDeviceSynchronize());
      v.push_back(t);
    } catch (const std::exception&) {}
  };
  const std::vector<int> us = level == 1 ? std::vector<int>{1} : std::vector<int>{1, 2, 4};
  const std::vector<int> nts = level == 1 ? std::vector<int>{0} : std::vector<int>{0, 1};
  const std::vector<int> mts = level == 2 ? std::vector<int>{4, 2} : std::vector<int>{4};
  if (table) add(*table);
  for (auto& f : fams)
    for (int WV : {1, 2, 4})
      for (int NP : {1, 2})
        for (int U : us)
          for (int NT : nts)
            for (int MT : mts) add({WV, f.first, NP, f.second, U, NT, MT});
  return v;
}

// ---- verify ------------------------------------------------------------------------------------------
static int VerifyOne(Bufs& b, const Shape& s, int kb, uint32_t seed, bool m128, const std::vector<Big>& bigs,
                     const std::vector<Stock>& stocks, const Stock* table, const char* tag, FILE* json, bool& first) {
  Fill(b, seed);
  const int rows = m128 ? 128 : 256;
  const size_t bytes = (size_t)rows * s.N * 2;
  std::vector<unsigned short> hb(bytes / 2);
  int fails = 0;
  std::vector<std::vector<unsigned short>> ref(stocks.size(), std::vector<unsigned short>(bytes / 2));
  for (size_t i = 0; i < stocks.size(); ++i) {
    CK(hipMemset(b.c_stock, 0xFF, (size_t)256 * s.N * 2));
    for (int r = 0; r < rows; r += 64) LaunchStock(b, stocks[i], b.w, r, 64);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(ref[i].data(), b.c_stock, bytes, hipMemcpyDeviceToHost));
  }
  long long nan = 0, zero = 0;
  for (unsigned short x : ref[0]) { nan += (x & 0x7F80u) == 0x7F80u; zero += (x & 0x7FFFu) == 0; }
  if (nan) { std::printf("  DATA PROBLEM: %lld non-finite outputs\n", nan); ++fails; }
  for (const Big& g : bigs) {
    CK(hipMemset(b.c_big, 0xFF, (size_t)256 * s.N * 2));
    LaunchBig(b, g, b.w);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(hb.data(), b.c_big, bytes, hipMemcpyDeviceToHost));
    int same = 0, fam = 0;
    std::string bad;
    const char* shipped = "no shipped row";
    for (size_t i = 0; i < stocks.size(); ++i) {
      if (stocks[i].SK != g.SK || stocks[i].SKG != g.SKG) continue;
      ++fam;
      size_t ndiff = 0, first_bad = (size_t)-1;
      for (size_t k = 0; k < hb.size(); ++k)
        if (hb[k] != ref[i][k]) { if (first_bad == (size_t)-1) first_bad = k; ++ndiff; }
      if (table && stocks[i] == *table) shipped = ndiff == 0 ? "SHIPPED ROW IDENTICAL" : "SHIPPED ROW DIFFERS";
      if (ndiff == 0) ++same;
      else {
        bad += "    DIFFER vs " + StockName(stocks[i]) + ": " + std::to_string(ndiff) + " of " + std::to_string(hb.size()) +
               " bf16, first at row " + std::to_string(first_bad / s.N) + " col " + std::to_string(first_bad % s.N) + "\n";
        ++fails;
      }
      if (json) {
        std::fprintf(json, "%s {\"kind\":\"verify\",\"shape\":\"%s\",\"kb\":%d,\"seed\":%u,\"pass\":\"%s\",\"big\":\"%s\",\"stock\":\"%s\",\"ndiff\":%zu,\"n\":%zu}\n",
                     first ? "" : ",", s.name, kb, seed, tag, BigName(g).c_str(), StockName(stocks[i]).c_str(), ndiff, hb.size());
        first = false;
      }
    }
    // run-to-run: the running-sum handoff through ws (SK / SKW > 1) must not race; 8 more launches, same bytes
    for (int rep = 0; rep < 8; ++rep) {
      std::vector<unsigned short> hr(hb.size());
      LaunchBig(b, g, b.w);
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(hr.data(), b.c_big, bytes, hipMemcpyDeviceToHost));
      if (hr != hb) { std::printf("  REPEAT DIFFERS %s\n", BigName(g).c_str()); ++fails; break; }
    }
    if (fam == 0) { std::printf("  seed %u %-8s KB%d %-8s %-24s: NO stock tuning of this (SK, SKG) to compare with\n", seed, s.name, kb, tag, BigName(g).c_str()); ++fails; }
    else std::printf("  seed %u %-8s KB%d %-8s %-32s: %s (%d of %d stock tunings of (SK%d,SKG%d) byte-identical; %s; %lld zero outputs)\n",
                     seed, s.name, kb, tag, BigName(g).c_str(), same == fam ? "IDENTICAL" : "DIFFER", same, fam, g.SK, g.SKG, shipped, zero);
    std::fputs(bad.c_str(), stdout);
  }
  const int tk = TicketsNonZero(b.tk_stock, s.N / 128) + TicketsNonZero(b.tk_big, s.N / 128);
  if (tk) { std::printf("  TICKETS NOT RESET (%d)\n", tk); ++fails; }
  return fails;
}

static int Verify(Bufs& b, const Shape& s, int kb, int seeds, bool m128, bool retune, FILE* json, bool& first) {
  Stock table{};
  const bool have = TableRow(s, kb, table);
  // --retune: every M = 256 configuration of every (SK, SKG) family against a small set of stock M = 64 tunings
  // of THE SAME family (a retuned row must give the identical bytes whichever tuning of its family it takes).
  const std::vector<Big> bigs = retune ? BigListAll(s, kb) : BigList(s, kb, m128, table);
  std::vector<std::pair<int, int>> fams;
  for (const Big& g : bigs) {
    bool seen = false;
    for (const auto& f : fams) seen = seen || (f.first == g.SK && f.second == g.SKG);
    if (!seen) fams.push_back({g.SK, g.SKG});
  }
  const std::vector<Stock> stocks = Candidates(b, have ? &table : nullptr, fams, retune ? 1 : 0);
  int fails = 0;
  for (int seed = 1; seed <= seeds; ++seed) {
    b.n_split = s.N; b.a1 = nullptr;
    fails += VerifyOne(b, s, kb, (uint32_t)seed, m128, bigs, stocks, have ? &table : nullptr, "one-A", json, first);
    b.n_split = s.N / 2; b.a1 = b.a + (size_t)256 * s.K;     // the second half of the columns reads A1
    fails += VerifyOne(b, s, kb, (uint32_t)seed, m128, bigs, stocks, have ? &table : nullptr, "n_split", json, first);
    b.n_split = s.N; b.a1 = nullptr;
  }
  // Negative control: the same comparison against a shipped tuning of a DIFFERENT SK must find differing
  // bytes (the accumulation order is part of the bits), or the comparison above proves nothing.
  {
    const Big& g = bigs[0];
    const int rows = m128 ? 128 : 256;
    const size_t n = (size_t)rows * s.N;
    Fill(b, 1);
    const std::vector<Stock> ctrl = Candidates(b, nullptr, {{g.SK == 4 ? 2 : 4, g.SKG}});
    if (!ctrl.empty()) {
      std::vector<unsigned short> hs(n), hb(n);
      CK(hipMemset(b.c_big, 0xFF, n * 2));
      LaunchBig(b, g, b.w);
      for (int r = 0; r < rows; r += 64) LaunchStock(b, ctrl[0], b.w, r, 64);
      CK(hipDeviceSynchronize());
      CK(hipMemcpy(hb.data(), b.c_big, n * 2, hipMemcpyDeviceToHost));
      CK(hipMemcpy(hs.data(), b.c_stock, n * 2, hipMemcpyDeviceToHost));
      size_t nd = 0;
      for (size_t k = 0; k < n; ++k) nd += hb[k] != hs[k];
      std::printf("  control %-8s KB%d %s vs stock %s: %zu of %zu bf16 differ (must be > 0)\n", s.name, kb,
                  BigName(g).c_str(), StockName(ctrl[0]).c_str(), nd, n);
      if (nd == 0) ++fails;
    }
  }
  return fails;
}

// ---- timing -----------------------------------------------------------------------------------------
struct Job {
  std::string name;
  int rows;                                // rows computed per call
  int kind;                                // 0 = stock M=64 one launch, 1 = stock x (rows/64) same weights, 2 = big
  Stock st;
  Big big;
  bool table = false;
  std::function<void(const unsigned char*)> launch;
  std::vector<double> t;                   // us per call, one entry per round
};
static double Med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }
static double Min(const std::vector<double>& v) { return *std::min_element(v.begin(), v.end()); }
static double Max(const std::vector<double>& v) { return *std::max_element(v.begin(), v.end()); }

static void RunJobs(std::vector<Job>& jobs, const Bufs& b, int rounds) {
  hipEvent_t e0, e1;
  CK(hipEventCreate(&e0));
  CK(hipEventCreate(&e1));
  for (Job& j : jobs)
    for (int i = 0; i < 2; ++i) j.launch(b.w);
  CK(hipDeviceSynchronize());
  for (int r = 0; r < rounds; ++r)
    for (Job& j : jobs) {
      CK(hipEventRecord(e0, 0));
      for (int i = 0; i < b.batch; ++i) j.launch(b.w + (size_t)i * b.w_bytes_each);
      CK(hipEventRecord(e1, 0));
      CK(hipEventSynchronize(e1));
      float ms = 0;
      CK(hipEventElapsedTime(&ms, e0, e1));
      j.t.push_back(1000.0 * ms / b.batch);
    }
  CK(hipEventDestroy(e0));
  CK(hipEventDestroy(e1));
}

// Per-round times of the best M = 256 job and the baselines, all for M rows.
struct ShapeResult {
  std::vector<double> big, table_cold, fam_cold, fam_same, any_cold, table_same;
  // the M = 256 job that keeps the shipped row's (SK, SKG) (bit-identical to the SHIPPED M = 64 row); when the
  // row's SK is not one the kernel has (mlp.down KB4: SK 16) the best SK 4 job, and ship_exact is false
  std::vector<double> ship;
  std::string big_name, table_name, fam_name, ship_name;
  bool have_table = false, ship_exact = false;
  int count = 0;
};

static std::vector<double> Ratio(const std::vector<double>& num, const std::vector<double>& den) {
  std::vector<double> r;
  for (size_t i = 0; i < num.size(); ++i) r.push_back(num[i] / den[i]);
  return r;
}

static ShapeResult Time(Bufs& b, const Shape& s, int kb, int rounds, bool m128, bool verbose, FILE* json, bool& first) {
  Fill(b, 1);
  const int M = m128 ? 128 : 256, mult = M / 64;
  Stock table{};
  const bool have = TableRow(s, kb, table);
  const std::vector<Big> bigs = BigList(s, kb, m128, table);
  std::vector<std::pair<int, int>> fams;
  for (const Big& g : bigs) fams.push_back({g.SK, g.SKG});
  if (have) fams.push_back({table.SK, table.SKG});
  const std::vector<Stock> stocks = Candidates(b, have ? &table : nullptr, fams);
  std::vector<Job> jobs;
  for (const Stock& t : stocks) {
    Job j;
    j.name = "S64  " + StockName(t);
    j.rows = 64; j.kind = 0; j.st = t; j.big = {}; j.table = have && t == table;
    j.launch = [=, &b](const unsigned char* w) { LaunchStock(b, t, w, 0, 64); };
    jobs.push_back(j);
    Job j4 = j;
    j4.name = "S64x" + std::to_string(mult) + " same-weights " + StockName(t);
    j4.rows = M; j4.kind = 1;
    j4.launch = [=, &b](const unsigned char* w) { for (int r = 0; r < M; r += 64) LaunchStock(b, t, w, r, 64); };
    jobs.push_back(j4);
  }
  for (const Big& g : bigs) {
    Job j;
    j.name = "BIG  " + BigName(g);
    j.rows = g.M; j.kind = 2; j.st = {}; j.big = g;
    j.launch = [=, &b](const unsigned char* w) { LaunchBig(b, g, w); };
    jobs.push_back(j);
  }
  RunJobs(jobs, b, rounds);
  const Job* big = nullptr;
  for (const Job& j : jobs)
    if (j.kind == 2 && (!big || Med(j.t) < Med(big->t))) big = &j;
  auto best = [&](int kind, int mode) {   // mode 0 = the family of the best big job, 1 = any, 2 = the table row
    const Job* r = nullptr;
    for (const Job& j : jobs) {
      if (j.kind != kind) continue;
      if (mode == 0 && (j.st.SK != big->big.SK || j.st.SKG != big->big.SKG)) continue;
      if (mode == 2 && !j.table) continue;
      if (!r || Med(j.t) < Med(r->t)) r = &j;
    }
    return r;
  };
  const Job* fc = best(0, 0);
  const Job* fs = best(1, 0);
  const Job* ac = best(0, 1);
  const Job* tc = best(0, 2);
  const Job* ts = best(1, 2);
  std::printf("---- %s KB%d N=%d K=%d, weights %.1f MiB per launch x %d copies, M=%d\n", s.name, kb, s.N, s.K,
              b.w_bytes_each / 1048576.0, b.batch, M);
  for (const Job& j : jobs) {
    const bool show = verbose || j.kind == 2 || j.table || &j == fc || &j == fs || &j == ac;
    if (!show) continue;
    const double med = Med(j.t);
    std::printf("%-64s%s rows %3d %9.1f us [%8.1f..%8.1f]  %7.3f us/row", j.name.c_str(), j.table ? " *" : "  ",
                j.rows, med, Min(j.t), Max(j.t), med / j.rows);
    if (j.kind == 2) {
      std::vector<double> fcm, ftm;
      for (size_t i = 0; i < j.t.size(); ++i) fcm.push_back(mult * fc->t[i]);
      const auto rf = Ratio(fcm, j.t);
      const auto rs = Ratio(fs->t, j.t);
      std::printf("  x%.3f vs %dx cold same-SK/SKG [%.3f..%.3f], x%.3f vs same-weights", Med(rf), mult, Min(rf), Max(rf), Med(rs));
      if (tc) {
        std::vector<double> tcm;
        for (size_t i = 0; i < j.t.size(); ++i) tcm.push_back(mult * tc->t[i]);
        std::printf(", x%.3f vs shipped row x%d", Med(Ratio(tcm, j.t)), mult);
      }
      if (json) {
        std::vector<double> acm;
        for (size_t i = 0; i < j.t.size(); ++i) acm.push_back(mult * ac->t[i]);
        std::fprintf(json, "%s {\"kind\":\"time\",\"shape\":\"%s\",\"kb\":%d,\"M\":%d,\"job\":\"%s\",\"us_median\":%.2f,\"us_per_row\":%.4f,\"speedup_vs_cold_family\":%.4f,\"speedup_vs_same_family\":%.4f,\"speedup_vs_any_cold\":%.4f,\"family_cold_us\":%.2f,\"family_same_us\":%.2f}\n",
                     first ? "" : ",", s.name, kb, M, j.name.c_str(), med, med / j.rows, Med(rf), Med(rs), Med(Ratio(acm, j.t)), Med(fc->t), Med(fs->t));
        first = false;
      }
    }
    std::printf("\n");
  }
  ShapeResult r;
  r.count = s.count;
  r.big = big->t;
  for (size_t i = 0; i < big->t.size(); ++i) {
    r.fam_cold.push_back(mult * fc->t[i]);
    r.fam_same.push_back(fs->t[i]);
    r.any_cold.push_back(mult * ac->t[i]);
    if (tc) { r.table_cold.push_back(mult * tc->t[i]); r.table_same.push_back(ts->t[i]); }
  }
  r.have_table = tc != nullptr;
  {
    const Job* sh = nullptr;
    for (const Job& j : jobs)
      if (j.kind == 2 && have && j.big.SK == table.SK && j.big.SKG == table.SKG && (!sh || Med(j.t) < Med(sh->t))) sh = &j;
    r.ship_exact = sh != nullptr;
    if (!sh)
      for (const Job& j : jobs)
        if (j.kind == 2 && j.big.SK == 4 && (!sh || Med(j.t) < Med(sh->t))) sh = &j;
    if (!sh) sh = big;
    r.ship = sh->t;
    r.ship_name = sh->name.substr(5);
  }
  r.big_name = big->name.substr(5);
  r.fam_name = fc->name.substr(5);
  if (tc) r.table_name = tc->name.substr(5);
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

// Combined report over the shapes in `names` (weights = their layer counts, or 1 for the pair).
static void CombinedReport(const char* title, const std::map<std::string, ShapeResult>& res, const std::vector<std::string>& names,
                           bool weighted, int kb, int M, FILE* json, bool& first) {
  std::vector<const std::vector<double>*> big, fam, same, any, tab;
  std::vector<int> w;
  bool have_tab = true;
  std::string picks;
  for (const std::string& n : names) {
    auto it = res.find(n);
    if (it == res.end()) return;
    const ShapeResult& r = it->second;
    big.push_back(&r.big); fam.push_back(&r.fam_cold); same.push_back(&r.fam_same); any.push_back(&r.any_cold);
    if (r.have_table) tab.push_back(&r.table_cold); else have_tab = false;
    w.push_back(weighted ? r.count : 1);
    picks += (picks.empty() ? "" : " + ") + n + ":" + r.big_name;
  }
  const auto B = SumRounds(big, w), F = SumRounds(fam, w), S = SumRounds(same, w), A = SumRounds(any, w);
  const auto rf = Ratio(F, B), rs = Ratio(S, B), ra = Ratio(A, B);
  std::printf("==== %s KB%d M=%d: big %.1f us (%s); vs %dx cold same-SK/SKG x%.3f [%.3f..%.3f]; vs same-weights x%.3f [%.3f..%.3f]; vs any-tuning cold x%.3f",
              title, kb, M, Med(B), picks.c_str(), M / 64, Med(rf), Min(rf), Max(rf), Med(rs), Min(rs), Max(rs), Med(ra));
  double rt_med = 0, rt_lo = 0, rt_hi = 0;
  if (have_tab) {
    const auto T = SumRounds(tab, w);
    const auto rt = Ratio(T, B);
    rt_med = Med(rt); rt_lo = Min(rt); rt_hi = Max(rt);
    std::printf("; vs shipped table rows x%d cold x%.3f [%.3f..%.3f]", M / 64, rt_med, rt_lo, rt_hi);
  }
  std::printf("\n");
  if (have_tab) {   // the same for the jobs that keep the shipped rows' (SK, SKG), against the shipped rows
    std::vector<const std::vector<double>*> ship, tab2;
    std::string picks2, inexact;
    for (const std::string& n : names) {
      const ShapeResult& r = res.find(n)->second;
      ship.push_back(&r.ship); tab2.push_back(&r.table_cold);
      picks2 += (picks2.empty() ? "" : " + ") + n + ":" + r.ship_name;
      if (!r.ship_exact) inexact += (inexact.empty() ? "" : ", ") + n;
    }
    const auto Sh = SumRounds(ship, w), T = SumRounds(tab2, w);
    const auto rr = Ratio(T, Sh);
    std::printf("     bit-identical-to-shipped-row picks: %s\n     x%.3f [%.3f..%.3f] vs %dx shipped rows cold; %.1f -> %.1f us%s; %s%s\n",
                picks2.c_str(), Med(rr), Min(rr), Max(rr), M / 64, Med(T), Med(Sh), weighted ? " (sum over the model's layers)" : "",
                inexact.empty() ? "every class keeps the shipped SK, SKG" : "NOT identical to the shipped row (its SK is not one the kernel has, SK 4 used): ",
                inexact.c_str());
    if (weighted) std::printf("     per token: shipped %.4f ms -> %.4f ms, saves %.4f ms/token\n", Med(T) / M / 1000.0, Med(Sh) / M / 1000.0,
                              (Med(T) - Med(Sh)) / M / 1000.0);
    if (json) {
      std::fprintf(json, "%s {\"kind\":\"combined_shipped_sk\",\"title\":\"%s\",\"kb\":%d,\"M\":%d,\"speedup_vs_shipped_rows\":%.4f,\"lo\":%.4f,\"hi\":%.4f,\"shipped_us\":%.2f,\"big_us\":%.2f}\n",
                   first ? "" : ",", title, kb, M, Med(rr), Min(rr), Max(rr), Med(T), Med(Sh));
      first = false;
    }
  }
  if (json) {
    std::fprintf(json, "%s {\"kind\":\"combined\",\"title\":\"%s\",\"kb\":%d,\"M\":%d,\"big_us\":%.2f,\"speedup_vs_cold_family\":%.4f,\"lo\":%.4f,\"hi\":%.4f,\"speedup_vs_same_family\":%.4f,\"speedup_vs_any_cold\":%.4f,\"speedup_vs_shipped_rows\":%.4f}\n",
                 first ? "" : ",", title, kb, M, Med(B), Med(rf), Min(rf), Max(rf), Med(rs), Med(ra), rt_med);
    first = false;
  }
}

// ---- retune sweep (--retune --mode time) ------------------------------------------------------------------
// The question (docs/trellis-m256.md "Retuning the plan"): which (SK, SKG) should a class's M = 64 table row have, now
// that the 256-row chunk is the default and the M = 256 kernel must keep the M = 64 row's order? SK and SKG are the
// only order-defining parameters, so a retune moves BOTH kernels together. Per class and rate this times
//   M = 256: every legal (SK, SKG, SKW) of the unit (plan SKW = what PlanTrellisM256 would pick, and the best SKW),
//   M = 64:  every legal stock tuning (WV x NP x U x NT x MT) of each of those (SK, SKG), in two stages (a short
//            screen of all of them, then the best three per family and the shipped row in the full protocol),
// each job on cold weight copies, every round timing every job once, so the ratios are medians of per-round
// ratios. A family is FREE when its M = 256 plan is >= min_gain faster than the shipped plan and its best M = 64
// tuning is no more than tol64 slower than the shipped M = 64 row. A family that is faster at M = 256 and slower
// at M = 64 is a TRADE: the row says what fraction of rows may still run through the 64-row kernel (tails,
// R4DX_PREFILL_CHUNK=0, quant2) before the retune loses. The printed .inc row is the best M = 64 tuning of the
// chosen family, in src/model/gemm_tuning_table_trellis.inc's format, ready to paste over the class's M = 64 row.
struct RetuneTotals {
  double ship256 = 0, free256 = 0, best256 = 0;   // sum over classes of layers x us per 256-row super-chunk
  double ship64 = 0, free64 = 0, best64 = 0;      // sum over classes of layers x us per 64-row chunk
};

static void RetuneShape(Bufs& b, const Shape& s, int kb, int rounds, double tol64, double min_gain, FILE* json,
                        bool& first, RetuneTotals& tot) {
  Fill(b, 1);
  Stock table{};
  if (!TableRow(s, kb, table)) {
    std::printf("---- %s KB%d: no shipped M=64 row (fallback tuning), skipped\n", s.name, kb);
    return;
  }
  const std::vector<Big> bigs = BigListAll(s, kb);
  std::vector<std::pair<int, int>> fams;
  const auto add_fam = [&](int sk, int skg) {
    for (const auto& f : fams) if (f.first == sk && f.second == skg) return;
    fams.push_back({sk, skg});
  };
  add_fam(table.SK, table.SKG);
  for (const Big& g : bigs) add_fam(g.SK, g.SKG);

  // stage 1: the M = 64 screen, every legal stock tuning of every family, 3 rounds of 2 cold copies
  const std::vector<Stock> stocks = Candidates(b, &table, fams, 2);
  std::vector<Job> screen;
  for (const Stock& t : stocks) {
    Job j;
    j.name = "S64  " + StockName(t);
    j.rows = 64; j.kind = 0; j.st = t; j.big = {}; j.table = t == table;
    j.launch = [=, &b](const unsigned char* w) { LaunchStock(b, t, w, 0, 64); };
    screen.push_back(j);
  }
  Bufs bs = b;
  bs.batch = std::min(b.batch, 2);
  RunJobs(screen, bs, 3);

  // stage 2: the best three of each family (and the shipped row) plus every M = 256 configuration, full protocol
  std::vector<Job> jobs;
  for (const auto& f : fams) {
    std::vector<const Job*> fj;
    for (const Job& j : screen) if (j.st.SK == f.first && j.st.SKG == f.second) fj.push_back(&j);
    std::sort(fj.begin(), fj.end(), [](const Job* x, const Job* y) { return Med(x->t) < Med(y->t); });
    int kept = 0;
    for (const Job* j : fj) {
      if (kept < 3 || j->table) {
        Job c = *j;
        c.t.clear();
        jobs.push_back(c);
      }
      ++kept;
    }
  }
  for (const Big& g : bigs) {
    Job j;
    j.name = "BIG  " + BigName(g);
    j.rows = g.M; j.kind = 2; j.st = {}; j.big = g;
    j.launch = [=, &b](const unsigned char* w) { LaunchBig(b, g, w); };
    jobs.push_back(j);
  }
  RunJobs(jobs, b, rounds);

  const int ship_skw = PlanSkw(s, kb, table.SK, table.SKG);
  const Job* ship256 = nullptr;
  const Job* ship64 = nullptr;
  for (const Job& j : jobs) {
    if (j.kind == 2 && j.big.SK == table.SK && j.big.SKG == table.SKG && j.big.SKW == ship_skw) ship256 = &j;
    if (j.kind == 0 && j.table) ship64 = &j;
  }
  if (!ship256 || !ship64) {
    std::printf("---- %s KB%d: the shipped row (SK%d SKG%d, plan SKW %d) has no timed job on one of the two sides, skipped\n",
                s.name, kb, table.SK, table.SKG, ship_skw);
    return;
  }
  struct Fam { int SK, SKG; const Job *plan, *best256, *best64; double x256, x64; };
  std::vector<Fam> rows;
  for (const auto& f : fams) {
    Fam r{f.first, f.second, nullptr, nullptr, nullptr, 0.0, 0.0};
    const int pskw = PlanSkw(s, kb, f.first, f.second);
    for (const Job& j : jobs) {
      if (j.kind == 2 && j.big.SK == f.first && j.big.SKG == f.second) {
        if (j.big.SKW == pskw) r.plan = &j;
        if (!r.best256 || Med(j.t) < Med(r.best256->t)) r.best256 = &j;
      }
      if (j.kind == 0 && j.st.SK == f.first && j.st.SKG == f.second &&
          (!r.best64 || Med(j.t) < Med(r.best64->t))) r.best64 = &j;
    }
    if (!r.plan || !r.best64) continue;
    r.x256 = Med(Ratio(ship256->t, r.plan->t));
    r.x64 = Med(Ratio(ship64->t, r.best64->t));
    rows.push_back(r);
  }
  std::sort(rows.begin(), rows.end(), [](const Fam& x, const Fam& y) { return x.x256 > y.x256; });

  const double sh256 = Med(ship256->t), sh64 = Med(ship64->t);
  std::printf("---- %s KB%d N=%d K=%d (%d layers): shipped M=64 row (SK%d SKG%d, %s): M=256 plan W%d %.1f us, M=64 %.1f us (one 64-row launch)\n",
              s.name, kb, s.N, s.K, s.count, table.SK, table.SKG, StockName(table).c_str(), ship_skw, sh256, sh64);
  std::printf("     %-12s %-9s %9s %9s | %-9s %9s | %-26s %9s %9s  %s\n", "(SK,SKG)", "plan SKW", "M256 us", "x ship", "best SKW",
              "M256 us", "best M=64 tuning", "M64 us", "x ship", "verdict");
  const Fam* free_pick = nullptr;
  const Fam* best_any = nullptr;
  for (const Fam& r : rows) {
    const bool same = r.SK == table.SK && r.SKG == table.SKG;
    const bool is_free = !same && r.x64 >= 1.0 - tol64 && r.x256 >= 1.0 + min_gain;
    if (!best_any) best_any = &r;
    if (is_free && !free_pick) free_pick = &r;   // rows are sorted by x256
    char verdict[160];
    if (same) std::snprintf(verdict, sizeof verdict, "shipped row");
    else if (is_free) std::snprintf(verdict, sizeof verdict, "FREE");
    else if (r.x256 > 1.0 + min_gain) {
      const double gain = sh256 - Med(r.plan->t), cost = Med(r.best64->t) - sh64;
      if (cost <= 0) std::snprintf(verdict, sizeof verdict, "FREE (below the gain threshold)");
      else std::snprintf(verdict, sizeof verdict, "TRADE: wins while <= %.0f%% of rows use 64-row launches", 100.0 * gain / (gain + 4.0 * cost));
    } else std::snprintf(verdict, sizeof verdict, "no gain");
    std::printf("     (SK%-2d,SKG%d)  W%-8d %9.1f x%-8.3f | W%-8d %9.1f | %-26s %9.1f x%-8.3f %s\n", r.SK, r.SKG, r.plan->big.SKW,
                Med(r.plan->t), r.x256, r.best256->big.SKW, Med(r.best256->t), r.best64->name.substr(5).c_str(), Med(r.best64->t),
                r.x64, verdict);
    if (json) {
      std::fprintf(json, "%s {\"kind\":\"retune\",\"shape\":\"%s\",\"kb\":%d,\"SK\":%d,\"SKG\":%d,\"plan_skw\":%d,\"m256_plan_us\":%.2f,\"m256_best_skw\":%d,\"m256_best_us\":%.2f,\"m64_best\":\"%s\",\"m64_us\":%.2f,\"x256\":%.4f,\"x64\":%.4f,\"free\":%s,\"shipped\":%s}\n",
                   first ? "" : ",", s.name, kb, r.SK, r.SKG, r.plan->big.SKW, Med(r.plan->t), r.best256->big.SKW, Med(r.best256->t),
                   StockName(r.best64->st).c_str(), Med(r.best64->t), r.x256, r.x64, is_free ? "true" : "false", same ? "true" : "false");
      first = false;
    }
  }
  for (const Fam& r : rows) {   // class (A): the shipped (SK, SKG) at another SKW changes no bit
    if (r.SK != table.SK || r.SKG != table.SKG) continue;
    const double g = sh256 / Med(r.best256->t);
    if (r.best256->big.SKW != ship_skw && g >= 1.0 + min_gain)
      std::printf("     (A) scheduling-only gain: the shipped (SK%d, SKG%d) at SKW %d is x%.3f faster than the plan's SKW %d -- bit-identical, a PlanTrellisM256 rule change\n",
                  table.SK, table.SKG, r.best256->big.SKW, g, ship_skw);
    if (json) {
      std::fprintf(json, "%s {\"kind\":\"retune_A\",\"shape\":\"%s\",\"kb\":%d,\"plan_skw\":%d,\"best_skw\":%d,\"x\":%.4f}\n", first ? "" : ",", s.name, kb,
                   ship_skw, r.best256->big.SKW, g);
      first = false;
    }
  }
  const auto inc_row = [&](const Fam& r) {
    const Stock& t = r.best64->st;
    std::printf("       .inc row: {Layout::kTrellis, %d, %d, 64, {%d, %d, %d, %d, %d, %d, %d}, 0, %d},  // %s Wc %d, retuned from SK %d SKG %d\n",
                s.N, s.K, t.WV, t.SK, t.MT, t.NP, t.NT, t.SKG, t.U, kb, s.name, t.WV * t.NP * 32, table.SK, table.SKG);
  };
  if (free_pick) {
    std::printf("     FREE pick: (SK%d, SKG%d), M=256 x%.3f, M=64 x%.3f\n", free_pick->SK, free_pick->SKG, free_pick->x256, free_pick->x64);
    inc_row(*free_pick);
  } else {
    std::printf("     no FREE retune for this class (needs M=256 >= x%.3f and M=64 >= x%.3f against the shipped row)\n", 1.0 + min_gain, 1.0 - tol64);
  }
  if (best_any && !(best_any->SK == table.SK && best_any->SKG == table.SKG) && best_any != free_pick) {
    std::printf("     fastest at M=256 (not FREE): (SK%d, SKG%d) x%.3f, M=64 x%.3f\n", best_any->SK, best_any->SKG, best_any->x256, best_any->x64);
    inc_row(*best_any);
  }
  tot.ship256 += s.count * sh256;
  tot.ship64 += s.count * sh64;
  tot.free256 += s.count * (free_pick ? Med(free_pick->plan->t) : sh256);
  tot.free64 += s.count * (free_pick ? Med(free_pick->best64->t) : sh64);
  tot.best256 += s.count * (best_any ? Med(best_any->plan->t) : sh256);
  tot.best64 += s.count * (best_any ? Med(best_any->best64->t) : sh64);
}

static void RetuneSummary(int kb, const RetuneTotals& t) {
  if (t.ship256 <= 0) return;
  std::printf("==== RETUNE SUMMARY KB%d (every class at this rate, layer-weighted, one forward pass of the model's linears)\n", kb);
  std::printf("     M=256 super-chunk: shipped %.2f ms | FREE picks only %.2f ms (-%.2f%%) | fastest family of every class %.2f ms (-%.2f%%)\n",
              t.ship256 / 1000.0, t.free256 / 1000.0, 100.0 * (1.0 - t.free256 / t.ship256), t.best256 / 1000.0,
              100.0 * (1.0 - t.best256 / t.ship256));
  std::printf("     M=64 chunk:        shipped %.2f ms | FREE picks only %.2f ms (%+.2f%%) | fastest-at-M256 picks %.2f ms (%+.2f%%)\n",
              t.ship64 / 1000.0, t.free64 / 1000.0, 100.0 * (t.free64 / t.ship64 - 1.0), t.best64 / 1000.0,
              100.0 * (t.best64 / t.ship64 - 1.0));
}

int main(int argc, char** argv) {
  std::string outp, shape = "all", mode = "both", kbs = "both";
  int rounds = 11, batch = 6, seeds = 4;
  bool m128 = false, verbose = false, retune = false;
  double tol64 = 0.01, min_gain = 0.01;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
    else if (!std::strcmp(argv[i], "--retune")) retune = true;
    else if (!std::strcmp(argv[i], "--tol64") && i + 1 < argc) tol64 = std::atof(argv[++i]) / 100.0;
    else if (!std::strcmp(argv[i], "--min-gain") && i + 1 < argc) min_gain = std::atof(argv[++i]) / 100.0;
    else if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) batch = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--seeds") && i + 1 < argc) seeds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--shape") && i + 1 < argc) shape = argv[++i];
    else if (!std::strcmp(argv[i], "--kb") && i + 1 < argc) kbs = argv[++i];
    else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
    else if (!std::strcmp(argv[i], "--m128")) m128 = true;
    else if (!std::strcmp(argv[i], "--verbose")) verbose = true;
    else { std::fprintf(stderr, "unknown argument %s\n", argv[i]); return 2; }
  }
  hipDeviceProp_t prop;
  CK(hipGetDeviceProperties(&prop, 0));
  std::printf("device %s, %d WGPs; mode %s, rounds %d, batch %d, seeds %d, M=%d\n", prop.name, prop.multiProcessorCount,
              mode.c_str(), rounds, batch, seeds, m128 ? 128 : 256);
  FILE* json = outp.empty() ? nullptr : std::fopen(outp.c_str(), "wb");
  bool first = true;
  if (json) std::fprintf(json, "{\"device\":\"%s\",\"mode\":\"%s\",\"rounds\":%d,\"batch\":%d,\"rows\":[\n", prop.name, mode.c_str(), rounds, batch);
  std::vector<Shape> shapes;
  for (const Shape& s : kShapes)
    if (shape == "all" || shape == s.name) shapes.push_back(s);
  if (shapes.empty()) { std::fprintf(stderr, "no such shape %s\n", shape.c_str()); return 2; }
  std::vector<int> kbl;
  if (kbs == "4" || kbs == "both") kbl.push_back(4);
  if (kbs == "5" || kbs == "both") kbl.push_back(5);
  int fails = 0;
  const int M = m128 ? 128 : 256;
  if (retune && m128) { std::fprintf(stderr, "--retune is an M = 256 sweep; it does not combine with --m128\n"); return 2; }
  for (int kb : kbl) {
    std::map<std::string, ShapeResult> res;
    RetuneTotals rtot;
    for (const Shape& s : shapes) {
      {   // M = 128 has no configuration for a shipped SK 16 (RG 2 x 16 slices needs PH > 1, instantiated for RG 4 only)
        Stock tr{};
        if (m128 && (!TableRow(s, kb, tr) || BigList(s, kb, m128, tr).empty())) {
          std::printf("---- %s KB%d: no M=128 configuration for the shipped row's SK (skipped)\n", s.name, kb);
          continue;
        }
      }
      Bufs b;
      Alloc(b, s, kb, mode == "verify" ? 1 : batch);
      if (mode == "verify" || mode == "both") fails += Verify(b, s, kb, seeds, m128, retune, json, first);
      if (mode == "time" || mode == "both") {
        if (retune) RetuneShape(b, s, kb, rounds, tol64, min_gain, json, first, rtot);
        else res[s.name] = Time(b, s, kb, rounds, m128, verbose, json, first);
      }
      Free(b);
    }
    if (retune) {
      RetuneSummary(kb, rtot);
      continue;
    }
    CombinedReport("MLP PAIR", res, {"gate_up", "down"}, false, kb, M, json, first);
    CombinedReport("ALL LINEAR CLASSES (weighted by layer count)", res, {"gate_up", "down", "qkv", "z", "out", "qg", "kv"}, true, kb, M, json, first);
  }
  if (json) {
    std::fprintf(json, "]}\n");
    std::fclose(json);
  }
  std::printf("%s (%d failing comparisons)\n", fails == 0 ? "ALL IDENTICAL / OK" : "FAILED", fails);
  return fails == 0 ? 0 : 1;
}
