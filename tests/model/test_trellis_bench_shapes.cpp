// tests/model/test_trellis_bench_shapes.cpp -- host-only (no HIP call, no device, no container).
// tool_trellis_gemm_bench `--group gemma` tunes the shapes of tests/kernels/trellis_bench_shapes.hpp;
// this checks that table against docs/gemma4-plan.md 3.7 / tools/profile/tune_gemm.py's `gemma.*` group
// (hard-coded here, independently), against GemmaConfig::Shard's rules (column-parallel linears split N,
// row-parallel ones K, the single full-layer KV head is replicated), the loader's whole-128-block rule,
// the 48-layer 5 : 1 layout, and that production's PickTuning answers every (shape, KB 4, M) the bench
// would tune from (so the bench's fallback column is the production fallback).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <set>
#include <string>
#include <utility>

#include "../kernels/trellis_bench_shapes.hpp"
#include "linear.h"

namespace {
int g_failures = 0, g_checks = 0;
void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}
using NK = std::pair<int, int>;
}  // namespace

int main() try {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  using namespace trellis_bench;
  // tune_gemm.py's gemma.* (TP = 1) and tp2.gemma.* names, shape for shape.
  const std::set<NK> want1 = {{4096, 3840}, {8192, 3840}, {2048, 3840}, {512, 3840},
                              {3840, 4096}, {3840, 8192}, {30720, 3840}, {3840, 15360}};
  const std::set<NK> want2 = {{2048, 3840}, {4096, 3840}, {1024, 3840}, {512, 3840},
                              {3840, 2048}, {3840, 4096}, {15360, 3840}, {3840, 7680}};
  std::set<NK> got1, got2;
  for (int i = 0; i < kNumGemmaShapes; ++i) {
    const GemmaShape& s = kGemmaShapes[i];
    got1.insert({s.N1, s.K1});
    got2.insert({s.N2, s.K2});
    const std::string n = s.cls;
    Check(s.N1 % 128 == 0 && s.K1 % 128 == 0 && s.N2 % 128 == 0 && s.K2 % 128 == 0, n + ": whole 128-blocks");
    const bool row_parallel = n.find(".o_") != std::string::npos || n == "gemma.down";
    const bool replicated = n == "gemma.k_full";
    if (replicated) Check(s.N2 == s.N1 && s.K2 == s.K1, n + ": the one full-layer KV head is replicated");
    else if (row_parallel) Check(s.N2 == s.N1 && s.K2 * 2 == s.K1, n + ": row-parallel splits K");
    else Check(s.N2 * 2 == s.N1 && s.K2 == s.K1, n + ": column-parallel splits N");
    Check(s.parts == (n == "gemma.gate_up" ? 2 : 1), n + ": parts");
    if (s.parts == 2) Check((s.N1 / 2) % 128 == 0 && (s.N2 / 2) % 128 == 0, n + ": part boundary in whole 128-blocks");
    for (int j = 0; j < i; ++j) Check(std::strcmp(kGemmaShapes[j].cls, s.cls) != 0, n + ": duplicate class");
    // Production's fallback answers every row band the bench tunes (M = 1; 32 and 64 for ptune).
    for (int kb : {4})
      for (int m : {1, 32, 64}) {
        try {
          const r4dx::model::LinearTuning t =
              r4dx::model::PickTuning(r4dx::model::Layout::kTrellis, s.N1, s.K1, m, kb);
          Check(t.SK > 0 && t.WV > 0, n + ": PickTuning gives a tuning");
          const r4dx::model::LinearTuning t2 =
              r4dx::model::PickTuning(r4dx::model::Layout::kTrellis, s.N2, s.K2, m, kb);
          Check(t2.SK > 0 && t2.WV > 0, n + " tp2: PickTuning gives a tuning");
        } catch (const std::exception& e) {
          Check(false, n + ": PickTuning threw: " + e.what());
        }
      }
  }
  Check(got1 == want1, "TP = 1 shape set equals tune_gemm.py's gemma.* group");
  Check(got2 == want2, "TP = 2 shape set equals tune_gemm.py's tp2.gemma.* group");
  // 48 layers, 8 full (5, 11, ..., 47), 40 sliding.
  int full = 0;
  for (int L = 0; L < kGemmaLayers; ++L) full += GemmaFullLayer(L);
  Check(kGemmaLayers == 48 && full == 8 && GemmaFullLayer(5) && GemmaFullLayer(47) && !GemmaFullLayer(0) &&
            !GemmaFullLayer(46),
        "5 sliding : 1 full over 48 layers");
  std::printf("test_trellis_bench_shapes: %d checks, %d failed\n", g_checks, g_failures);
  return g_failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
  return 1;
}
