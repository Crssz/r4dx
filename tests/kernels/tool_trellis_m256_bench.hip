// tests/kernels/tool_trellis_m256_bench.hip -- standalone (host + device) bench and bit-identity check of
// the PROTOTYPE bigger-M trellis GEMM (third_party/libr4d/r4d_gemm_trellis_nt_m256.hip) against the
// SHIPPED M = 64 kernel (r4d_gemm_trellis_nt_m64.hip), both through their real host entries.
//
//   verify  random weights / A / svh (several seeds), then C[256][N] bf16 of ONE M = 256 launch is compared
//           byte for byte with C of four M = 64 launches (one per 64-row slice of the same A) at every
//           shipped-kernel tuning with the same (SK, SKG); M = 128 against two.
//   time    us per call, us per row, and the speedups: the M = 256 launch against
//             cold  4 x one M = 64 launch, each launch on a weight copy no launch of the batch has read (what
//                   64-row chunks cost today: every layer's weights stream from DRAM once per chunk),
//             same  four M = 64 launches back to back on the SAME weights (a Model chunk of 256 rows with
//                   today's kernel; the MALL / L2 may serve launches 2-4),
//           with the M = 64 side taken as the fastest of the listed shipped tunings of the same (SK, SKG)
//           (the only ones an M = 256 kernel with that (SK, SKG) can be bit-identical to) and, separately,
//           of every listed tuning. Every round times every job once (a batch of launches, each on a
//           different weight copy), so the ratios are medians of per-round ratios. --pair prints the MLP
//           pair (mlp.gate_up + mlp.down) once both shapes have run.
// It never touches the tuning table (no --inc-out here) and writes nothing but the --out JSON.
//
// Build: tests/kernels/build_m256_bench.ps1 (hipcc, objects under D:\models\r4dx\linear\A2\obj).
// Run (one device; HIP_VISIBLE_DEVICES set by the caller; every set is bounded):
//   tool_trellis_m256_bench.exe [--shape gate_up|down|all] [--kb 4|5|both] [--mode verify|time|both]
//                               [--seeds 4] [--rounds 11] [--batch 6] [--out x.json]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include "r4d.h"

void r4d_gemm_trellis_nt_m256(int64_t a0, int64_t a1, int n_split, int64_t w, int64_t svh, int64_t c,
                              int64_t ws, int64_t tickets, int M, int K, int N, int KB, int SK, int NP,
                              int SKG, int U, float out_scale, int64_t stream);

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

struct Stock { int WV, SK, NP, SKG, U, NT; };
struct Big { int M, NP, U, SK, SKG; };

struct Shape { const char* name; int N, K; };

// shipped-kernel tunings run as the M = 64 side. The first of each list is the tuning table's row for
// that (N, K, KB) at M = 64 (src/model/gemm_tuning_table_trellis.inc, MT = 4 always); the others are
// alternatives so the fastest of a given (SK, SKG) is a fair baseline.
static std::vector<Stock> StockList(const Shape& s, int kb) {
  std::vector<Stock> v;
  if (s.N == 34816) {
    v = {{2, 4, 2, 1, 1, 0}, {1, 4, 2, 1, 1, 0}, {2, 4, 2, 1, 1, 1}, {4, 4, 1, 1, 1, 0}, {2, 2, 2, 1, 1, 0}};
  } else if (kb == 4) {
    v = {{1, 16, 2, 2, 1, 0}, {2, 4, 2, 2, 1, 0}, {1, 4, 2, 2, 1, 0}, {2, 8, 2, 2, 1, 0}, {2, 4, 2, 1, 1, 0}};
  } else {
    v = {{1, 4, 2, 2, 1, 0}, {2, 4, 2, 2, 1, 0}, {1, 8, 2, 2, 1, 0}, {2, 4, 2, 1, 1, 0}};
  }
  return v;
}
// M = 256 (and 128) configurations
static std::vector<Big> BigList(const Shape& s, int kb, bool m128) {
  const int skg = s.N == 34816 ? 1 : 2;
  std::vector<Big> v;
  if (!m128) {
    v.push_back({256, 2, 2, 4, skg});
    v.push_back({256, 1, 4, 4, skg});
  } else {
    v.push_back({128, 2, 1, 4, skg});
    v.push_back({128, 1, 2, 4, skg});
  }
  (void)kb;
  return v;
}

struct Bufs {
  unsigned char* w = nullptr;
  size_t w_bytes_each = 0;
  unsigned short* a = nullptr;
  float* svh = nullptr;
  unsigned short* c_stock = nullptr;
  unsigned short* c_big = nullptr;
  float* ws_stock = nullptr;
  float* ws_big = nullptr;
  unsigned* tk_stock = nullptr;
  unsigned* tk_big = nullptr;
  int batch = 0;
  int N = 0, K = 0, KB = 4;
};
static const float kOutScale = 0.0883883f;

static void LaunchStock(const Bufs& b, const Stock& t, const unsigned char* w, int row0, int M) {
  r4d_gemm_trellis_nt_m64((int64_t)(b.a + (size_t)row0 * b.K), 0, b.N, (int64_t)w, (int64_t)b.svh,
                          (int64_t)(b.c_stock + (size_t)row0 * b.N), (int64_t)b.ws_stock, (int64_t)b.tk_stock,
                          M, b.K, b.N, b.KB, t.WV, t.SK, 4, t.NP, t.SKG, t.U, t.NT, kOutScale, 0);
}
static void LaunchBig(const Bufs& b, const Big& g, const unsigned char* w) {
  r4d_gemm_trellis_nt_m256((int64_t)b.a, 0, b.N, (int64_t)w, (int64_t)b.svh, (int64_t)b.c_big,
                           (int64_t)b.ws_big, (int64_t)b.tk_big, g.M, b.K, b.N, b.KB, g.SK, g.NP, g.SKG,
                           g.U, kOutScale, 0);
}

static std::string StockName(const Stock& t) {
  char nm[96];
  std::snprintf(nm, sizeof nm, "WV%d NP%d SK%d SKG%d U%d NT%d", t.WV, t.NP, t.SK, t.SKG, t.U, t.NT);
  return nm;
}
static std::string BigName(const Big& g) {
  char nm[96];
  std::snprintf(nm, sizeof nm, "M%d NP%d U%d SK%d SKG%d", g.M, g.NP, g.U, g.SK, g.SKG);
  return nm;
}

static void Alloc(Bufs& b, const Shape& s, int kb, int batch, int maxskg) {
  b.N = s.N; b.K = s.K; b.KB = kb; b.batch = batch;
  b.w_bytes_each = (size_t)s.N * s.K * kb / 8;
  CK(hipMalloc(&b.w, b.w_bytes_each * batch));
  CK(hipMalloc(&b.a, (size_t)256 * s.K * 2 + 4096));
  CK(hipMalloc(&b.svh, (size_t)s.N * 4));
  CK(hipMalloc(&b.c_stock, (size_t)256 * s.N * 2));
  CK(hipMalloc(&b.c_big, (size_t)256 * s.N * 2));
  CK(hipMalloc(&b.ws_stock, (size_t)maxskg * 64 * s.N * 4));
  CK(hipMalloc(&b.ws_big, (size_t)maxskg * 256 * s.N * 4));
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
  hipLaunchKernelGGL(FillA, dim3(2048), dim3(256), 0, 0, b.a, (int64_t)256 * b.K, seed * 104729u + 3u);
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

// One verify pass: returns the number of failing comparisons.
static int Verify(Bufs& b, const Shape& s, int kb, uint32_t seed, bool m128, FILE* json, bool& first) {
  Fill(b, seed);
  const int rows = m128 ? 128 : 256;
  const size_t bytes = (size_t)rows * s.N * 2;
  std::vector<unsigned short> hs(bytes / 2), hb(bytes / 2);
  int fails = 0;
  const std::vector<Stock> stocks = StockList(s, kb);
  // reference C of every stock tuning, once
  std::vector<std::vector<unsigned short>> ref(stocks.size(), std::vector<unsigned short>(bytes / 2));
  for (size_t i = 0; i < stocks.size(); ++i) {
    CK(hipMemset(b.c_stock, 0xFF, (size_t)256 * s.N * 2));
    for (int r = 0; r < rows; r += 64) LaunchStock(b, stocks[i], b.w, r, 64);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(ref[i].data(), b.c_stock, bytes, hipMemcpyDeviceToHost));
  }
  // sanity of the data: finite, varied
  {
    long long nan = 0, zero = 0;
    for (unsigned short x : ref[0]) { nan += (x & 0x7F80u) == 0x7F80u; zero += (x & 0x7FFFu) == 0; }
    std::printf("  seed %u data: %lld / %zu NaN-or-Inf, %lld zeros in stock C[0]\n", seed, nan, ref[0].size(), zero);
    if (nan) { std::printf("  DATA PROBLEM: non-finite outputs\n"); ++fails; }
  }
  for (const Big& g : BigList(s, kb, m128)) {
    CK(hipMemset(b.c_big, 0xFF, (size_t)256 * s.N * 2));
    LaunchBig(b, g, b.w);
    CK(hipDeviceSynchronize());
    CK(hipMemcpy(hb.data(), b.c_big, bytes, hipMemcpyDeviceToHost));
    for (size_t i = 0; i < stocks.size(); ++i) {
      if (stocks[i].SK != g.SK || stocks[i].SKG != g.SKG) continue;
      size_t ndiff = 0, first_bad = (size_t)-1;
      for (size_t k = 0; k < hb.size(); ++k)
        if (hb[k] != ref[i][k]) { if (first_bad == (size_t)-1) first_bad = k; ++ndiff; }
      std::printf("  %-4s %-8s KB%d %-26s vs stock %-26s: %s (%zu of %zu bf16 differ%s)\n", "seed", s.name, kb,
                  BigName(g).c_str(), StockName(stocks[i]).c_str(), ndiff == 0 ? "IDENTICAL" : "DIFFER",
                  ndiff, hb.size(),
                  first_bad == (size_t)-1 ? "" : (" first at row " + std::to_string(first_bad / s.N) + " col " +
                                                  std::to_string(first_bad % s.N)).c_str());
      if (ndiff) ++fails;
      if (json) {
        std::fprintf(json, "%s {\"kind\":\"verify\",\"shape\":\"%s\",\"kb\":%d,\"seed\":%u,\"big\":\"%s\",\"stock\":\"%s\",\"ndiff\":%zu,\"n\":%zu}\n",
                     first ? "" : ",", s.name, kb, seed, BigName(g).c_str(), StockName(stocks[i]).c_str(), ndiff, hb.size());
        first = false;
      }
    }
  }
  const int tk = TicketsNonZero(b.tk_stock, s.N / 128) + TicketsNonZero(b.tk_big, s.N / 128);
  if (tk) { std::printf("  TICKETS NOT RESET (%d)\n", tk); ++fails; }
  return fails;
}

// ---- timing -----------------------------------------------------------------------------------------
struct Job {
  std::string name;
  int rows;                                // rows computed per call
  int kind;                                // 0 = stock M=64 one launch, 1 = stock x (rows/64) same weights, 2 = big
  int SK, SKG;
  std::function<void(const unsigned char*)> launch;
  std::vector<double> t;                   // us per call, one entry per round
};
static double Med(std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; }

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

struct ShapeResult {
  std::vector<double> big_t;               // best big job, per round
  std::vector<double> s64_family_cold;     // best same-(SK,SKG) stock, one launch, per round (x4 = cold)
  std::vector<double> s64_family_same4;    // best same-(SK,SKG) stock, four launches on the same weights
  std::vector<double> s64_any_cold;
  std::string big_name, stock_name;
};

static ShapeResult Time(Bufs& b, const Shape& s, int kb, int rounds, bool m128, FILE* json, bool& first) {
  Fill(b, 1);
  const int M = m128 ? 128 : 256;
  std::vector<Job> jobs;
  for (const Stock& t : StockList(s, kb)) {
    Job j;
    j.name = "S64  " + StockName(t);
    j.rows = 64; j.kind = 0; j.SK = t.SK; j.SKG = t.SKG;
    j.launch = [=, &b](const unsigned char* w) { LaunchStock(b, t, w, 0, 64); };
    jobs.push_back(j);
    Job j4 = j;
    j4.name = std::string("S64x") + std::to_string(M / 64) + " same-weights " + StockName(t);
    j4.rows = M; j4.kind = 1;
    j4.launch = [=, &b](const unsigned char* w) { for (int r = 0; r < M; r += 64) LaunchStock(b, t, w, r, 64); };
    jobs.push_back(j4);
  }
  for (const Big& g : BigList(s, kb, m128)) {
    Job j;
    j.name = "BIG  " + BigName(g);
    j.rows = g.M; j.kind = 2; j.SK = g.SK; j.SKG = g.SKG;
    j.launch = [=, &b](const unsigned char* w) { LaunchBig(b, g, w); };
    jobs.push_back(j);
  }
  RunJobs(jobs, b, rounds);
  // pick the best big job by median, then its family's stock baselines
  const Job* big = nullptr;
  for (const Job& j : jobs)
    if (j.kind == 2 && (!big || Med(j.t) < Med(big->t))) big = &j;
  auto best = [&](int kind, bool family) {
    const Job* r = nullptr;
    for (const Job& j : jobs)
      if (j.kind == kind && (!family || (j.SK == big->SK && j.SKG == big->SKG)) && (!r || Med(j.t) < Med(r->t))) r = &j;
    return r;
  };
  const Job* fc = best(0, true);
  const Job* fs = best(1, true);
  const Job* ac = best(0, false);
  const Job* as = best(1, false);
  std::printf("---- %s KB%d N=%d K=%d, weights %.1f MiB per launch x %d copies, M=%d\n", s.name, kb, s.N, s.K,
              b.w_bytes_each / 1048576.0, b.batch, M);
  for (const Job& j : jobs) {
    const double med = Med(j.t);
    std::printf("%-58s rows %3d %9.1f us [%8.1f..%8.1f]  %7.3f us/row", j.name.c_str(), j.rows, med,
                *std::min_element(j.t.begin(), j.t.end()), *std::max_element(j.t.begin(), j.t.end()), med / j.rows);
    if (j.kind == 2) {
      std::vector<double> rc, rs, ra, ras;
      for (size_t i = 0; i < j.t.size(); ++i) {
        rc.push_back((M / 64.0) * fc->t[i] / j.t[i]);
        rs.push_back(fs->t[i] / j.t[i]);
        ra.push_back((M / 64.0) * ac->t[i] / j.t[i]);
        ras.push_back(as->t[i] / j.t[i]);
      }
      std::printf("  x%.3f vs %dx cold[%.3f..%.3f] x%.3f vs same-weights x%.3f vs any-tuning cold", Med(rc), M / 64,
                  *std::min_element(rc.begin(), rc.end()), *std::max_element(rc.begin(), rc.end()), Med(rs), Med(ra));
      if (json) {
        std::fprintf(json, "%s {\"kind\":\"time\",\"shape\":\"%s\",\"kb\":%d,\"M\":%d,\"job\":\"%s\",\"us_median\":%.2f,\"us_per_row\":%.4f,\"speedup_vs_cold\":%.4f,\"speedup_vs_same\":%.4f,\"speedup_vs_any_cold\":%.4f,\"stock_cold_us\":%.2f,\"stock_same_us\":%.2f}\n",
                     first ? "" : ",", s.name, kb, M, j.name.c_str(), med, med / j.rows, Med(rc), Med(rs), Med(ra), Med(fc->t), Med(fs->t));
        first = false;
      }
    }
    std::printf("\n");
  }
  ShapeResult r;
  r.big_t = big->t;
  for (size_t i = 0; i < big->t.size(); ++i) {
    r.s64_family_cold.push_back((M / 64.0) * fc->t[i]);
    r.s64_family_same4.push_back(fs->t[i]);
    r.s64_any_cold.push_back((M / 64.0) * ac->t[i]);
  }
  r.big_name = big->name;
  r.stock_name = fc->name;
  (void)as;
  return r;
}

static void PairReport(const ShapeResult& gu, const ShapeResult& dn, int kb, int M, FILE* json, bool& first) {
  auto ratio = [&](const std::vector<double>& a1, const std::vector<double>& a2) {
    std::vector<double> r;
    const size_t n = std::min(gu.big_t.size(), dn.big_t.size());
    for (size_t i = 0; i < n; ++i) r.push_back((a1[i] + a2[i]) / (gu.big_t[i] + dn.big_t[i]));
    return r;
  };
  const auto rc = ratio(gu.s64_family_cold, dn.s64_family_cold);
  const auto rs = ratio(gu.s64_family_same4, dn.s64_family_same4);
  const auto ra = ratio(gu.s64_any_cold, dn.s64_any_cold);
  const double pair_big = Med(gu.big_t) + Med(dn.big_t);
  std::printf("==== MLP PAIR KB%d M=%d: big %.1f us (%s + %s); vs %dx cold x%.3f [%.3f..%.3f]; vs same-weights x%.3f [%.3f..%.3f]; vs any-tuning cold x%.3f\n",
              kb, M, pair_big, gu.big_name.substr(5).c_str(), dn.big_name.substr(5).c_str(), M / 64, Med(rc),
              *std::min_element(rc.begin(), rc.end()), *std::max_element(rc.begin(), rc.end()), Med(rs),
              *std::min_element(rs.begin(), rs.end()), *std::max_element(rs.begin(), rs.end()), Med(ra));
  if (json) {
    std::fprintf(json, "%s {\"kind\":\"pair\",\"kb\":%d,\"M\":%d,\"big_us\":%.2f,\"speedup_vs_cold\":%.4f,\"lo\":%.4f,\"hi\":%.4f,\"speedup_vs_same\":%.4f,\"speedup_vs_any_cold\":%.4f}\n",
                 first ? "" : ",", kb, M, pair_big, Med(rc), *std::min_element(rc.begin(), rc.end()),
                 *std::max_element(rc.begin(), rc.end()), Med(rs), Med(ra));
    first = false;
  }
}

int main(int argc, char** argv) {
  std::string outp, shape = "all", mode = "both", kbs = "both";
  int rounds = 11, batch = 6, seeds = 4;
  bool m128 = false;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--out") && i + 1 < argc) outp = argv[++i];
    else if (!std::strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--batch") && i + 1 < argc) batch = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--seeds") && i + 1 < argc) seeds = std::atoi(argv[++i]);
    else if (!std::strcmp(argv[i], "--shape") && i + 1 < argc) shape = argv[++i];
    else if (!std::strcmp(argv[i], "--kb") && i + 1 < argc) kbs = argv[++i];
    else if (!std::strcmp(argv[i], "--mode") && i + 1 < argc) mode = argv[++i];
    else if (!std::strcmp(argv[i], "--m128")) m128 = true;
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
  if (shape == "gate_up" || shape == "all") shapes.push_back({"gate_up", 34816, 5120});
  if (shape == "down" || shape == "all") shapes.push_back({"down", 5120, 17408});
  std::vector<int> kbl;
  if (kbs == "4" || kbs == "both") kbl.push_back(4);
  if (kbs == "5" || kbs == "both") kbl.push_back(5);
  int fails = 0;
  for (int kb : kbl) {
    std::map<std::string, ShapeResult> res;
    for (const Shape& s : shapes) {
      Bufs b;
      Alloc(b, s, kb, mode == "verify" ? 1 : batch, 2);
      if (mode == "verify" || mode == "both")
        for (int seed = 1; seed <= seeds; ++seed) fails += Verify(b, s, kb, (uint32_t)seed, m128, json, first);
      if (mode == "time" || mode == "both") res[s.name] = Time(b, s, kb, rounds, m128, json, first);
      Free(b);
    }
    if (res.count("gate_up") && res.count("down")) PairReport(res["gate_up"], res["down"], kb, m128 ? 128 : 256, json, first);
  }
  if (json) {
    std::fprintf(json, "]}\n");
    std::fclose(json);
  }
  std::printf("%s (%d failing comparisons)\n", fails == 0 ? "ALL IDENTICAL / OK" : "FAILED", fails);
  return fails == 0 ? 0 : 1;
}
