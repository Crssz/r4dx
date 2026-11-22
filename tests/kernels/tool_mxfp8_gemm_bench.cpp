// tests/kernels/tool_mxfp8_gemm_bench.cpp -- r4dx_gemm_mxfp8_nt / r4dx_gemm_mxfp8_moe_decode speed sweep (S6).
//
// Sweeps legal MXFP8 geometries (WV 1/2/4/8, SK 1..16, MB 1..min(4, ceil(M/16)), NPW 1/4, NT 0/1) over a pooled
// working set (--pool-mib, default 512 MiB) on Inkling2's shapes. With --bit-identical, pins SK to
// Mxfp8FallbackTuning's SK so every output is bit-for-bit identical to the KL-gated fallback.
//
// Usage:
//   $env:HIP_VISIBLE_DEVICES='1'
//   build\win-hip\tests\kernels\tool_mxfp8_gemm_bench.exe --bit-identical --m 1,32,64 --inc-out src/model/gemm_tuning_table_mxfp8.inc
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/mxfp8_gemm.h"

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

std::vector<Shape> DefaultShapes() {
  return {
      {"expert.gate_up", 2304, 2304},
      {"expert.down", 2304, 1152},
      {"attn.qgrkv", 8576, 2304},
      {"attn.qgr", 6528, 2304},
      {"attn.q_gate", 3072, 2304},
      {"attn.kv", 1024, 2304},
      {"attn.o", 2304, 3072},
      {"attn.rel", 384, 2304},
      {"lm_head", 200064, 2304},
  };
}

std::vector<int> ParseInts(const std::string& s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string t;
  while (std::getline(ss, t, ',')) v.push_back(std::stoi(t));
  return v;
}

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

bool LegalMx(const Geo& g, int M, int K) {
  const int mtiles = (M + 15) / 16;
  return K % (g.SK * 32) == 0 && g.WV * g.SK * 32 <= 1024 && g.MB <= std::max(1, mtiles) &&
         g.WV * g.NPW * g.SK <= 64;
}

int FallbackSk(int N, int K, int M) {
  const int wv = (N / 16 < 512) ? 1 : 4;
  const int npw = (M > 16) ? 4 : 1;
  for (int sk : {8, 4, 2, 1}) {
    if (K % (sk * 32) == 0 && wv * sk * 32 <= 1024 && wv * npw * sk <= 64) return sk;
  }
  return 1;
}

}  // namespace

int main(int argc, char** argv) {
  std::string inc_out;
  std::vector<int> Ms = {1, 32, 64};
  std::vector<Shape> shapes = DefaultShapes();
  int reps = 20;
  size_t pool_mib = 512;
  bool bit_identical = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      return argv[++i];
    };
    if (a == "--inc-out") inc_out = next();
    else if (a == "--m") Ms = ParseInts(next());
    else if (a == "--reps") reps = std::stoi(next());
    else if (a == "--pool-mib") pool_mib = static_cast<size_t>(std::stoul(next()));
    else if (a == "--bit-identical") bit_identical = true;
    else {
      std::fprintf(stderr, "unknown argument %s\n", a.c_str());
      return 2;
    }
  }
  try {
    R4DX_HIP_CHECK(hipSetDevice(0));
    hipDeviceProp_t prop;
    R4DX_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device: %s (%s), pool %zu MiB, reps %d, bit_identical=%d\n", prop.name, prop.gcnArchName, pool_mib,
                reps, bit_identical ? 1 : 0);
    const size_t pool_bytes = pool_mib << 20;
    std::vector<std::string> rows;
    const int maxM = *std::max_element(Ms.begin(), Ms.end());

    for (const auto& s : shapes) {
      const int N = s.N, K = s.K;
      DeviceBuffer<uint16_t> a(static_cast<size_t>(std::max(maxM, 8)) * K), c(static_cast<size_t>(maxM) * N);
      R4DX_HIP_CHECK(hipMemset(a.data(), 0x3c, a.bytes()));
      Pool wq(static_cast<size_t>(N) * K, pool_bytes, 0x38);
      Pool ws(static_cast<size_t>(N) * (K / 32), pool_bytes / 32, 0x7f);

      const bool is_expert = s.name.find("expert.") != std::string::npos;
      const int k_top = 8;
      const int input_div = (s.name == "expert.gate_up") ? k_top : 1;
      const int n_moe_sets = static_cast<int>(std::max<size_t>(1, wq.bufs.size() / k_top));
      std::vector<DeviceBuffer<uint64_t>> wq_ptrs_sets, ws_ptrs_sets;
      DeviceBuffer<int32_t> moe_ids(k_top);
      DeviceBuffer<uint16_t> moe_c(static_cast<size_t>(k_top) * N);
      // Prefill MoE (128 experts, 256 * 8 = 2048 pairs -> 16 rows/expert)
      constexpr int kPrefillE = 128;
      constexpr int kPrefillPairs = 256 * 8;
      DeviceBuffer<uint64_t> pf_wq_ptrs, pf_ws_ptrs;
      DeviceBuffer<int32_t> pf_offsets;
      DeviceBuffer<uint16_t> pf_a, pf_c;
      if (is_expert) {
        moe_ids.CopyFromHost(std::vector<int32_t>{0, 1, 2, 3, 4, 5, 6, 7});
        for (int si = 0; si < n_moe_sets; ++si) {
          std::vector<uint64_t> qh(k_top), sh(k_top);
          for (int j = 0; j < k_top; ++j) {
            qh[j] = static_cast<uint64_t>(wq.at(static_cast<size_t>(si * k_top + j)));
            sh[j] = static_cast<uint64_t>(ws.at(static_cast<size_t>(si * k_top + j)));
          }
          wq_ptrs_sets.emplace_back(k_top);
          wq_ptrs_sets.back().CopyFromHost(qh);
          ws_ptrs_sets.emplace_back(k_top);
          ws_ptrs_sets.back().CopyFromHost(sh);
        }
        std::vector<uint64_t> pq(kPrefillE), ps(kPrefillE);
        std::vector<int32_t> poff(kPrefillE + 1);
        for (int e = 0; e < kPrefillE; ++e) {
          pq[e] = static_cast<uint64_t>(wq.at(static_cast<size_t>(e)));
          ps[e] = static_cast<uint64_t>(ws.at(static_cast<size_t>(e)));
          poff[e] = e * (kPrefillPairs / kPrefillE);
        }
        poff[kPrefillE] = kPrefillPairs;
        pf_wq_ptrs = DeviceBuffer<uint64_t>(kPrefillE);
        pf_ws_ptrs = DeviceBuffer<uint64_t>(kPrefillE);
        pf_offsets = DeviceBuffer<int32_t>(kPrefillE + 1);
        pf_a = DeviceBuffer<uint16_t>(static_cast<size_t>(kPrefillPairs) * K);
        pf_c = DeviceBuffer<uint16_t>(static_cast<size_t>(kPrefillPairs) * N);
        pf_wq_ptrs.CopyFromHost(pq);
        pf_ws_ptrs.CopyFromHost(ps);
        pf_offsets.CopyFromHost(poff);
        R4DX_HIP_CHECK(hipMemset(pf_a.data(), 0x3c, pf_a.bytes()));
      }

      for (int M : Ms) {
        if (s.name == "lm_head" && M > 1) continue;  // lm_head only runs at M = 1
        const int runM = (M == 64 && !is_expert) ? 256 : M;  // real prefill chunks are M = 256
        if (runM > maxM) {
          a = DeviceBuffer<uint16_t>(static_cast<size_t>(runM) * K);
          c = DeviceBuffer<uint16_t>(static_cast<size_t>(runM) * N);
          R4DX_HIP_CHECK(hipMemset(a.data(), 0x3c, a.bytes()));
        }
        double best = 1e30;
        double fb_us = -1.0;
        Geo bg{};
        const int pinned_sk = FallbackSk(N, K, M);
        const Geo fb{(N / 16 < 512) ? 1 : 4, pinned_sk, (M > 16) ? std::min(4, (M + 15) / 16) : 1,
                     (M > 16) ? 4 : 1, (M > 16) ? 0 : 1};

        for (int WV : {1, 2, 4, 8})
          for (int SK : {1, 2, 4, 8, 16})
            for (int MB = 1; MB <= 4; ++MB)
              for (int NPW : {1, 4})
                for (int NT : {0, 1}) {
                  if (bit_identical && SK != pinned_sk) continue;
                  if (M == 1 && MB != 1) continue;
                  if (is_expert && M > 1 && MB != fb.MB) continue;  // moe_prefill ignores MB
                  const Geo g{WV, SK, MB, NPW, NT};
                  if (!LegalMx(g, M, K)) continue;
                  double us = -1.0;
                  if (is_expert && M == 1) {
                    us = TimeUs(
                        [&](size_t i) {
                          const size_t si = i % static_cast<size_t>(n_moe_sets);
                          r4dx_gemm_mxfp8_moe_decode(P(a.data()), P(wq_ptrs_sets[si].data()),
                                                     P(ws_ptrs_sets[si].data()), P(moe_ids.data()), P(moe_c.data()),
                                                     k_top, input_div, k_top, K, N, WV, SK, NPW, NT, 0);
                        },
                        reps);
                  } else if (is_expert && M > 1) {
                    us = TimeUs(
                        [&](size_t) {
                          r4dx_gemm_mxfp8_moe_prefill(P(pf_a.data()), P(pf_wq_ptrs.data()), P(pf_ws_ptrs.data()),
                                                      P(pf_offsets.data()), P(pf_c.data()), kPrefillPairs, kPrefillE, K,
                                                      N, WV, SK, NPW, NT, 0);
                        },
                        reps);
                  } else {
                    us = TimeUs(
                        [&](size_t i) {
                          r4dx_gemm_mxfp8_nt(P(a.data()), wq.at(i), ws.at(i), P(c.data()), runM, K, N, WV, SK, MB, NPW,
                                             NT, 0);
                        },
                        reps);
                  }
                  if (g.WV == fb.WV && g.SK == fb.SK && g.MB == fb.MB && g.NPW == fb.NPW && g.NT == fb.NT) fb_us = us;
                  if (us > 0 && us < best) best = us, bg = g;
                }
        const double mult = (is_expert && M == 1) ? static_cast<double>(k_top)
                          : (is_expert && M > 1)  ? static_cast<double>(kPrefillE)
                                                  : 1.0;
        const double bytes = mult * (static_cast<double>(N) * K + static_cast<double>(N) * (K / 32) +
                                     static_cast<double>(runM) * K * 2 + static_cast<double>(runM) * N * 2);
        const double gbs = bytes / (best * 1e3);
        std::printf("%-16s [%6dx%5d] M%-2d  fb %8.2f us -> best %8.2f us (x%.2f) %6.1f GB/s  (WV%d SK%d MB%d NPW%d NT%d)\n",
                    s.name.c_str(), N, K, M, fb_us, best, fb_us > 0 ? fb_us / best : 1.0, gbs, bg.WV, bg.SK, bg.MB,
                    bg.NPW, bg.NT);
        std::fflush(stdout);
        char buf[256];
        std::snprintf(buf, sizeof buf, "    {Layout::kMxfp8, %d, %d, %d, {%d, %d, %d, %d, %d}, 0},  // %s %.2f us (fb %.2f us)",
                      N, K, M, bg.WV, bg.SK, bg.MB, bg.NPW, bg.NT, s.name.c_str(), best, fb_us);
        rows.push_back(buf);
      }
    }
    if (!inc_out.empty()) {
      std::ofstream f(inc_out);
      f << "// GENERATED by tests/kernels/tool_mxfp8_gemm_bench.exe --bit-identical --m 1,32,64 --inc-out.\n"
        << "// Measured on " << prop.name << " (" << prop.gcnArchName << "), pooled working set " << pool_mib
        << " MiB, median of 5 rounds x " << reps << " launches; expert.* M=1 swept via r4dx_gemm_mxfp8_moe_decode\n"
        << "// (K_top=8), SK pinned to Mxfp8FallbackTuning so every output is bit-for-bit identical.\n"
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
