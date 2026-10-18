// tests/model/test_w8a16_tuning.cpp -- CPU only (no HIP call; HIP_VISIBLE_DEVICES=-1 hides every device from the runtime that
// linking r4dx_model_linear loads). docs/w8a16.md 7.1 `model_w8a16_tuning`:
//   PickTuning(kW8a16, N, K, M, group) is LEGAL for r4dx_gemm_w8a16_nt_m64 (the kernel's own launch rules, restated here:
//   K % (SK * G) == 0, WV*SK*32 <= 1024, MB 1..4, NPW in {1, 4}, the LDS reduction WV*NPW*SK KiB <= 64) for every LFM2 linear
//   shape at TP = 1 and at the TP = 2 rank shapes (not a v1 deliverable, but the fallback must already be legal there), at
//   G 64 and 128 and M in {1, 8, 16, 17, 32, 64};
//   M = 2..16 resolve to the M = 1 row with NT 0 (the row identity a verify window / LFM2's kPerSlot == kSorted rely on: a
//   row's bits depend on SK alone);
//   the fallback's documented shape (docs/w8a16.md 4.5): WV 1 below 512 row tiles else 4; SK the largest of {8, 4, 2, 1}
//   that fits; M <= 16: MB 1 NPW 1 NT 1; M > 16: MB min(4, ceil(M/16)), NPW 4, NT 0;
//   the cache keys w8a16 apart from w4a16 / bf16 at the same (N, K, M) and apart by group;
//   gemm_tuning_table_w8a16.inc is the empty placeholder (so the above IS the fallback), and an illegal shape throws.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "linear.h"
#include "quant_linear.h"

using namespace r4dx::model;

namespace {

int g_failures = 0;
void Check(bool ok, const std::string& what) {
  if (!ok) std::printf("FAIL: %s\n", what.c_str());
  if (!ok) ++g_failures;
}

struct Shape {
  const char* name;
  int64_t N, K;
};

std::string Str(const LinearTuning& t) {
  return "WV " + std::to_string(t.WV) + " SK " + std::to_string(t.SK) + " MB " + std::to_string(t.MB) + " NPW " +
         std::to_string(t.NPW) + " NT " + std::to_string(t.NT);
}

bool Legal(const LinearTuning& t, int64_t K, int g) {
  return t.WV >= 1 && t.SK >= 1 && K % (static_cast<int64_t>(t.SK) * g) == 0 && t.WV * t.SK * 32 <= 1024 && t.MB >= 1 &&
         t.MB <= 4 && (t.NPW == 1 || t.NPW == 4) && t.WV * t.NPW * t.SK <= 64 && (t.NT == 0 || t.NT == 1);
}

}  // namespace

int main() {
  try {
    // LFM2.5-24B-A2B (docs/w8a16.md 3.3): TP = 1 and the TP = 2 rank shapes.
    const std::vector<Shape> shapes = {
        {"expert gate_up", 3072, 2048},  {"expert down", 2048, 1536},   {"conv.in_proj", 6144, 2048},
        {"attn.q/o, conv.out_proj", 2048, 2048}, {"attn.k/v", 512, 2048}, {"dense gate_up", 23552, 2048},
        {"dense down", 2048, 11776},     {"lm_head", 128000, 2048},
        // TP = 2 ranks: column-parallel N / 2, row-parallel K / 2.
        {"tp2 expert gate_up", 1536, 2048}, {"tp2 expert down", 2048, 768}, {"tp2 conv.in_proj", 3072, 2048},
        {"tp2 attn.q", 1024, 2048},         {"tp2 attn.o / conv.out_proj", 2048, 1024}, {"tp2 attn.k/v", 256, 2048},
        {"tp2 dense gate_up", 11776, 2048}, {"tp2 dense down", 2048, 5888}, {"tp2 lm_head", 64000, 2048},
        // The tiny fixture (hidden 512).
        {"tiny conv.in_proj", 1536, 512}, {"tiny 512x512", 512, 512}, {"tiny expert gate_up", 1024, 512},
        {"tiny attn.k/v", 128, 512}, {"tiny lm_head", 64, 512},
    };
    int picks = 0;
    for (const Shape& s : shapes) {
      for (int g : {64, 128}) {
        if (s.K % g != 0) continue;
        const LinearTuning t1 = PickTuning(Layout::kW8a16, s.N, s.K, 1, g);
        for (int64_t M : {1, 2, 4, 8, 16, 17, 32, 33, 64}) {
          const LinearTuning t = PickTuning(Layout::kW8a16, s.N, s.K, M, g);
          ++picks;
          const std::string tag = std::string(s.name) + " [" + std::to_string(s.N) + ", " + std::to_string(s.K) + "] g" +
                                  std::to_string(g) + " M=" + std::to_string(M) + ": " + Str(t);
          Check(Legal(t, s.K, g), tag + " is a legal r4dx_gemm_w8a16_nt_m64 launch");
          if (M <= 16) {
            Check(t.WV == t1.WV && t.SK == t1.SK && t.MB == t1.MB && t.NPW == t1.NPW,
                  tag + " resolves to the M = 1 row (" + Str(t1) + ")");
            Check(t.NT == (M == 1 ? t1.NT : 0), tag + ": NT 0 for M = 2..16 (w4a16's rule)");
          }
          // Bit-identical SK rule (docs/w8a16.md 4.5): SK always equals the fallback's SK, and shapes not in
          // gemm_tuning_table_w8a16.inc (g == 64 or the tiny K == 512 shapes) resolve to the exact fallback.
          const int wv = (s.N / 16 < 512) ? 1 : 4;
          const int npw = M > 16 ? 4 : 1;
          int sk = 1;
          for (int c : {8, 4, 2, 1}) {
            if (s.K % (static_cast<int64_t>(c) * g) == 0 && wv * c * 32 <= 1024 && wv * npw * c <= 64) {
              sk = c;
              break;
            }
          }
          Check(t.SK == sk, tag + ": SK matches the bit-identical fallback SK " + std::to_string(sk));
          if (g == 64 || s.K == 512) {
            const int mb = M > 16 ? static_cast<int>(std::min<int64_t>(4, (M + 15) / 16)) : 1;
            Check(t.WV == wv && t.SK == sk && t.MB == mb && t.NPW == npw && t.NT == (M == 1 ? 1 : 0),
                  tag + " is docs/w8a16.md 4.5's fallback (WV " + std::to_string(wv) + " SK " + std::to_string(sk) +
                      " MB " + std::to_string(mb) + " NPW " + std::to_string(npw) + ")");
          }
        }
      }
    }
    std::printf("PASS: %d w8a16 picks legal, M = 2..16 on the M = 1 row, fallback as documented\n", picks);

    // docs/w8a16.md 3.3's group counts at G = 128 (16, 12 and 92) and 4.5's "largest legal SK of {8, 4, 2, 1}": 8, 4, 4.
    Check(PickTuning(Layout::kW8a16, 3072, 2048, 1, 128).SK == 8, "expert gate_up K 2048 g128 (16 groups): SK 8 at M = 1");
    Check(PickTuning(Layout::kW8a16, 2048, 1536, 1, 128).SK == 4, "expert down K 1536 g128 (12 groups): SK 4 at M = 1");
    Check(PickTuning(Layout::kW8a16, 2048, 11776, 1, 128).SK == 4, "dense down K 11776 g128 (92 groups): SK 4 at M = 1");
    Check(PickTuning(Layout::kW8a16, 3072, 2048, 1, 128).WV == 2, "expert gate_up (grid.z=4 tuned row): WV 2");
    Check(PickTuning(Layout::kW8a16, 2048, 1536, 1, 128).WV == 4, "expert down (grid.z=4 tuned row): WV 4");
    Check(PickTuning(Layout::kW8a16, 23552, 2048, 64, 128).SK == 4,
          "dense gate_up M = 64: SK 4 (WV 4 x NPW 4 x SK 4 = 64 KiB of LDS, the cap)");

    // The cache keeps layouts and groups apart at one (N, K, M).
    {
      const LinearTuning a = PickTuning(Layout::kW8a16, 2048, 1536, 1, 128);
      const LinearTuning b = PickTuning(Layout::kW8a16, 2048, 1536, 1, 64);
      Check(a.SK == 4 && b.SK == 8, "g128 and g64 at the same shape resolve separately (SK 4 vs 8): " + Str(a) + " / " + Str(b));
      // bf16 at [2048, 2048] M = 1 is SK 4 (the generic fallback / table); w8a16 g128 there is SK 8: a shared cache slot
      // would hand one of them the other's pick.
      const LinearTuning c = PickTuning(Layout::kBf16, 2048, 2048, 1, 0);
      const LinearTuning d = PickTuning(Layout::kW8a16, 2048, 2048, 1, 128);
      Check(d.SK == 8, "w8a16 [2048, 2048] g128 after a bf16 pick at the same shape is its own: " + Str(d) +
                           " (bf16: " + Str(c) + ")");
      Check(PickTuning(Layout::kBf16, 2048, 2048, 1, 0).SK == c.SK, "and the bf16 pick is unchanged after it");
    }

    // Illegal shapes throw instead of handing out a launch the kernel rejects.
    const auto throws = [](auto&& f) {
      try {
        f();
      } catch (const std::exception&) {
        return true;
      }
      return false;
    };
    Check(throws([] { (void)PickTuning(Layout::kW8a16, 2040, 2048, 1, 128); }), "N % 16 != 0 throws");
    Check(throws([] { (void)PickTuning(Layout::kW8a16, 2048, 1600, 1, 128); }), "K % group != 0 throws");
    Check(throws([] { (void)PickTuning(Layout::kW8a16, 2048, 2048, 1, 600); }), "a group outside the cache key throws");

    // EpilogueForLayout: none (the conversion is in-kernel).
    Check(EpilogueForLayout(Layout::kW8a16) == EpilogueForLayout(Layout::kBf16),
          "EpilogueForLayout(kW8a16) is r4dx_epilogue_none, like bf16");
  } catch (const std::exception& e) {
    std::printf("FAIL: unexpected exception: %s\n", e.what());
    ++g_failures;
  }
  if (g_failures != 0) {
    std::printf("%d check(s) FAILED\n", g_failures);
    return 1;
  }
  std::printf("all checks passed\n");
  return 0;
}
