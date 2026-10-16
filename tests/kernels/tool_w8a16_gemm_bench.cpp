// tests/kernels/tool_w8a16_gemm_bench.cpp -- r4dx_gemm_w8a16_nt_m64 speed (docs/w8a16.md 4.5, 7.2).
//
// For every (N, K) shape, group and M band: sweeps every legal w8a16 geometry (WV 1/2/4/8, SK 1..16, MB up to
// the row tiles, NPW 1/4, NT 0/1) and the bf16 (r4d_gemm_bf16_nt_m64) and w4a16 (r4d_gemm_w4a16_nt_m64_g,
// group 64) baselines over their own geometry sweeps, and prints the best time of each, the w8a16 GB/s
// (weight + scale + A + C bytes) and TF/s, and the speedups. The working set is POOLED: each kernel cycles
// through enough weight copies to exceed --pool-mib (default 512 MiB), because one copy of most of these
// weights fits the 64 MiB last-level cache and a single-buffer bench measures the wrong machine
// (r4d_gemm_w4a16_nt_m64.hip's NT comment). Weights are filled with a constant byte pattern (timing does not
// depend on the values; correctness is test_w8a16_gemm's job).
//
// Timing: per candidate, 3 warm-up launches, then 5 rounds of --reps launches between hip events; the median
// round / reps is the time.
//
// --inc-out FILE writes the best w8a16 geometry per (N, K, M, G) as GemmTuningRow initializers for
// src/model/gemm_tuning_table_w8a16.inc (lane W3's file; linear.cpp includes it in namespace w8a16):
//     {Layout::kW8a16, N, K, M, {WV, SK, MB, NPW, NT}, G},  // name us
// Default M bands are 1 (serves M = 1..16 by ResolveTuning's rule), 8, 16 (reported only), 32 and 64.
//
// Usage:
//   $env:HIP_VISIBLE_DEVICES='1'
//   build\win-hip\tests\kernels\tool_w8a16_gemm_bench.exe [--inc-out x.inc] [--groups 128,64] [--m 1,8,16,32,64]
//       [--shapes name:N:K,...] [--reps 20] [--pool-mib 512] [--no-baselines]
// GPU tool, WRITTEN, NOT YET RUN by its author.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/w8a16_gemm.h"

using namespace r4dx::core;

namespace {

int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

struct Shape {
  std::string name;
  int N, K;
};
struct Geo {
  int WV, SK, MB, NPW, NT;
};

// LFM2.5-24B-A2B at TP = 1 (docs/w8a16.md 3.3, docs/lfm2-semantics.md).
std::vector<Shape> DefaultShapes() {
  return {{"expert.gate_up", 3072, 2048}, {"expert.down", 2048, 1536},  {"conv.in_proj", 6144, 2048},
          {"conv.out_proj", 2048, 2048},  {"attn.kv", 512, 2048},       {"dense.gate_up", 23552, 2048},
          {"dense.down", 2048, 11776},    {"lm_head", 128000, 2048}};
}

std::vector<int> ParseInts(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) v.push_back(std::stoi(t));
  return v;
}

// A pool of `copies` buffers of `bytes` each, cycled per launch.
struct Pool {
  std::vector<DeviceBuffer<uint8_t>> bufs;
  Pool(size_t bytes, size_t pool_bytes, int fill) {
    const size_t copies = std::max<size_t>(1, (pool_bytes + bytes - 1) / bytes);
    for (size_t i = 0; i < std::min<size_t>(copies, 64); ++i) {
      bufs.emplace_back(bytes);
      R4DX_HIP_CHECK(hipMemset(bufs.back().data(), fill, bytes));
    }
  }
  int64_t at(size_t i) { return P(bufs[i % bufs.size()].data()); }
};

// Median per-launch microseconds of fn(i) over 5 rounds of `reps` launches; -1 if the first launch throws.
template <class F>
double TimeUs(F&& fn, int reps) {
  try {
    for (int i = 0; i < 3; ++i) fn(i);
  } catch (const std::exception&) {
    (void)hipGetLastError();
    return -1.0;
  }
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  hipEvent_t e0, e1;
  R4DX_HIP_CHECK(hipEventCreate(&e0));
  R4DX_HIP_CHECK(hipEventCreate(&e1));
  std::vector<double> rounds;
  size_t k = 3;
  for (int r = 0; r < 5; ++r) {
    R4DX_HIP_CHECK(hipEventRecord(e0, 0));
    for (int i = 0; i < reps; ++i) fn(k++);
    R4DX_HIP_CHECK(hipEventRecord(e1, 0));
    R4DX_HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0;
    R4DX_HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    rounds.push_back(1000.0 * ms / reps);
  }
  R4DX_HIP_CHECK(hipEventDestroy(e0));
  R4DX_HIP_CHECK(hipEventDestroy(e1));
  std::sort(rounds.begin(), rounds.end());
  return rounds[2];
}

bool LegalW8(const Geo& g, int M, int K, int G) {
  const int mtiles = (M + 15) / 16;
  return K % (g.SK * G) == 0 && g.WV * g.SK * 32 <= 1024 && g.MB <= std::max(1, mtiles) &&
         g.WV * g.NPW * g.SK <= 64;
}

}  // namespace

int main(int argc, char** argv) {
  std::string inc_out;
  std::vector<int> groups = {128};
  std::vector<int> Ms = {1, 8, 16, 32, 64};
  std::vector<Shape> shapes = DefaultShapes();
  int reps = 20;
  size_t pool_mib = 512;
  bool baselines = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--inc-out") inc_out = next();
    else if (a == "--groups") groups = ParseInts(next());
    else if (a == "--m") Ms = ParseInts(next());
    else if (a == "--reps") reps = std::stoi(next());
    else if (a == "--pool-mib") pool_mib = static_cast<size_t>(std::stoul(next()));
    else if (a == "--no-baselines") baselines = false;
    else if (a == "--shapes") {
      shapes.clear();
      std::stringstream ss(next());
      std::string t;
      while (std::getline(ss, t, ',')) {
        const auto c1 = t.find(':'), c2 = t.rfind(':');
        if (c1 == std::string::npos || c1 == c2) throw std::runtime_error("--shapes wants name:N:K");
        shapes.push_back({t.substr(0, c1), std::stoi(t.substr(c1 + 1, c2 - c1 - 1)), std::stoi(t.substr(c2 + 1))});
      }
    } else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  try {
    R4DX_HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop;
    R4DX_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s (%s), pool %zu MiB, reps %d\n", prop.name, prop.gcnArchName, pool_mib, reps);
    const size_t pool_bytes = pool_mib << 20;
    std::vector<std::string> rows;
    const int maxM = *std::max_element(Ms.begin(), Ms.end());

    for (const auto& s : shapes) {
      const int N = s.N, K = s.K;
      DeviceBuffer<uint16_t> a(static_cast<size_t>(maxM) * K), c(static_cast<size_t>(maxM) * N);
      R4DX_HIP_CHECK(hipMemset(a.data(), 0x3c, a.bytes()));  // bf16 0x3c3c ~ 0.0115
      std::unique_ptr<Pool> bf16_pool, w4_q, w4_s;
      if (baselines) {
        bf16_pool = std::make_unique<Pool>(static_cast<size_t>(N) * K * 2, pool_bytes, 0x3c);
        w4_q = std::make_unique<Pool>(static_cast<size_t>(N) * K / 2, pool_bytes, 0x37);
        w4_s = std::make_unique<Pool>(static_cast<size_t>(N) * (K / 64) * 4, pool_bytes / 16, 0x3c);
      }
      for (int G : groups) {
        if (K % G) {
          std::printf("%s [%dx%d]: K %% G%d != 0, skipped\n", s.name.c_str(), N, K, G);
          continue;
        }
        Pool wq(static_cast<size_t>(N) * K, pool_bytes, 0x93), ws(static_cast<size_t>(N) * (K / G) * 2,
                                                                     pool_bytes / 32, 0x3c);
        for (int M : Ms) {
          // w8a16 sweep
          double best = 1e30;
          Geo bg{};
          for (int WV : {1, 2, 4, 8})
            for (int SK : {1, 2, 4, 8, 16})
              for (int MB = 1; MB <= 4; ++MB)
                for (int NPW : {1, 4})
                  for (int NT : {0, 1}) {
                    const Geo g{WV, SK, MB, NPW, NT};
                    if (!LegalW8(g, M, K, G)) continue;
                    const double us = TimeUs(
                        [&](size_t i) {
                          r4dx_gemm_w8a16_nt_m64(P(a.data()), wq.at(i), ws.at(i), P(c.data()), M, K, N, G, WV, SK,
                                                 MB, NPW, NT, 0);
                        },
                        reps);
                    if (us > 0 && us < best) best = us, bg = g;
                  }
          double best_bf = -1, best_w4 = -1;
          if (baselines) {
            for (int WV : {1, 2, 4, 8})
              for (int SK : {1, 2, 4, 8, 16})
                for (int MB = 1; MB <= std::max(1, std::min(4, (M + 15) / 16)); ++MB) {
                  if (WV * SK * 32 > 1024) continue;
                  const double us = TimeUs(
                      [&](size_t i) {
                        r4d_gemm_bf16_nt_m64(P(a.data()), bf16_pool->at(i), P(c.data()), M, K, N, WV, SK, MB, 0);
                      },
                      reps);
                  if (us > 0 && (best_bf < 0 || us < best_bf)) best_bf = us;
                }
            if (K % 64 == 0)
              for (int WV : {1, 2, 4, 8})
                for (int SK : {1, 2, 4, 8, 16})
                  for (int MB = 1; MB <= std::max(1, std::min(4, (M + 15) / 16)); ++MB)
                    for (int NPW : {1, 4})
                      for (int NT : {0, 1}) {
                        if (WV * SK * 32 > 1024 || K % (SK * 64) || WV * NPW * SK > 64) continue;
                        const double us = TimeUs(
                            [&](size_t i) {
                              r4d_gemm_w4a16_nt_m64_g(64, P(a.data()), w4_q->at(i), w4_s->at(i), P(c.data()), M, K,
                                                      N, WV, SK, MB, NPW, NT, 0);
                            },
                            reps);
                        if (us > 0 && (best_w4 < 0 || us < best_w4)) best_w4 = us;
                      }
          }
          const double bytes = static_cast<double>(N) * K + static_cast<double>(N) * (K / G) * 2 +
                               static_cast<double>(M) * K * 2 + static_cast<double>(M) * N * 2;
          const double gbs = bytes / (best * 1e3), tfs = 2.0 * M * N * K / (best * 1e6);
          std::printf("%-16s [%6dx%5d] G%-3d M%-2d w8a16 %9.2f us %7.1f GB/s %6.2f TF/s  (WV%d SK%d MB%d NPW%d NT%d)",
                      s.name.c_str(), N, K, G, M, best, gbs, tfs, bg.WV, bg.SK, bg.MB, bg.NPW, bg.NT);
          if (baselines)
            std::printf("  bf16 %9.2f us (x%.2f)  w4a16g64 %9.2f us (x%.2f)", best_bf,
                        best_bf > 0 ? best_bf / best : 0.0, best_w4, best_w4 > 0 ? best_w4 / best : 0.0);
          std::printf("\n");
          std::fflush(stdout);
          if (M == 1 || M == 32 || M == 64) {
            char buf[256];
            std::snprintf(buf, sizeof buf, "    {Layout::kW8a16, %d, %d, %d, {%d, %d, %d, %d, %d}, %d},  // %s %.2f us",
                          N, K, M, bg.WV, bg.SK, bg.MB, bg.NPW, bg.NT, G, s.name.c_str(), best);
            rows.push_back(buf);
          }
        }
      }
    }
    if (!inc_out.empty()) {
      std::ofstream f(inc_out);
      f << "// GENERATED by tests/kernels/tool_w8a16_gemm_bench.exe --inc-out (docs/w8a16.md 4.5).\n"
        << "// Measured on " << prop.name << " (" << prop.gcnArchName << "), pooled working set " << pool_mib
        << " MiB, median of 5 rounds x " << reps << " launches.\n"
        << "// Best w8a16 geometry per (N, K, M, group) over WV 1/2/4/8, SK 1..16, MB, NPW 1/4, NT 0/1. linear.cpp\n"
        << "// includes this file inside namespace w8a16; M = 1 rows serve M = 1..16, M = 32 / 64 the prefill chunks.\n"
        << "static const GemmTuningRow kGemmTuningTable[] = {\n";
      for (const auto& r : rows) f << r << "\n";
      f << "};\n";
      std::printf("wrote %zu rows to %s\n", rows.size(), inc_out.c_str());
    }
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 1;
  }
  return 0;
}
