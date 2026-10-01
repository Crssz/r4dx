// tests/kernels/trellis_bench_shapes.hpp -- the Gemma 4 12B trellis linears tool_trellis_gemm_bench
// tunes with `--group gemma` (docs/gemma4-plan.md 3.7), as plain data so a host-only test
// (tests/model/test_trellis_bench_shapes.cpp) can check them without HIP.
//
// One row per linear class: its (N, K) at TP = 1 and for one TP = 2 rank (GemmaConfig::Shard: the
// column-parallel linears split N, the row-parallel ones K, the single full-layer KV head is
// replicated), and the number of parts (2: gate_up runs as gate and up). kv_sliding is attn.k and
// attn.v, one row each; a full layer has no v (k_eq_v).
#pragma once

namespace trellis_bench {

struct GemmaShape {
  const char* cls;
  int N1, K1;      // TP = 1
  int N2, K2;      // one TP = 2 rank
  int parts;
};

inline constexpr GemmaShape kGemmaShapes[] = {
    {"gemma.q_sliding", 4096, 3840, 2048, 3840, 1},
    {"gemma.k_sliding", 2048, 3840, 1024, 3840, 1},
    {"gemma.v_sliding", 2048, 3840, 1024, 3840, 1},
    {"gemma.o_sliding", 3840, 4096, 3840, 2048, 1},
    {"gemma.q_full", 8192, 3840, 4096, 3840, 1},
    {"gemma.k_full", 512, 3840, 512, 3840, 1},
    {"gemma.o_full", 3840, 8192, 3840, 4096, 1},
    {"gemma.gate_up", 30720, 3840, 15360, 3840, 2},
    {"gemma.down", 3840, 15360, 3840, 7680, 1},
};
inline constexpr int kNumGemmaShapes = sizeof(kGemmaShapes) / sizeof(kGemmaShapes[0]);

// 48 layers, 5 sliding : 1 full, the full layer last of each six (layers 5, 11, ..., 47).
inline constexpr int kGemmaLayers = 48;
inline constexpr bool GemmaFullLayer(int L) { return L % 6 == 5; }

}  // namespace trellis_bench
