// tests/model/test_trellis_gemma_plan.cpp -- host-only (no HIP call, no device, no container).
// docs/gemma4-plan.md 3.7, task M1-29: every Gemma 4 12B trellis linear must get an EXPLICIT answer
// from the trellis GEMM path, never a silent fallback.
//
// For each linear shape at TP = 1 and at a TP = 2 rank, at both rates (KB 4 and 5), and for each row
// band ApplyLinear asks the tuning for (M = 1, 16, 32, 64):
//   * the tuning TrellisTuningFor / PickTuning hands out is legal for the 64-row kernel (whole blocks of
//     Wc = WV * NP * 32 columns on N and on a two-part linear's boundary, whole U-steps of every K slice,
//     a split 128-group only with every row tile in its block) -- the rules of libr4d's r4d_trellis_check
//     as linear.cpp mirrors them (TrellisRowFits);
//   * PlanTrellisM256 at M = 256 says ok (the one-launch wide kernel exists for the shape) and its
//     (SK, SKG) are the M = 64 pick's, so a 256-row super-chunk does not quietly slice into four launches;
//   * the plan's own check (libr4d's r4d_gemm_trellis_nt_m256_check, a pure host function) agrees.
// A shape whose plan is refused is printed with the refusal and FAILS: if a future tuning row makes one
// refuse (say an SK 16 row at KB 5), this test names it instead of the model slicing at runtime.
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <vector>

#include "linear.h"
#include "r4dx/core/r4d.hpp"

namespace {

using r4dx::model::Layout;
using r4dx::model::LinearTuning;

int g_failures = 0;
int g_checks = 0;
void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

struct Shape {
  const char* name;
  int64_t N, K;
  int parts;
  int64_t part_n0;   // parts == 2 only
  bool tp2;
};

// docs/gemma4-plan.md 3.7. head_dim 256 sliding (16 q heads, 8 kv heads), 512 full (16 q, 1 kv, k_eq_v).
const Shape kShapes[] = {
    // TP = 1
    {"q sliding", 4096, 3840, 1, 0, false},   {"q full", 8192, 3840, 1, 0, false},
    {"k/v sliding", 2048, 3840, 1, 0, false}, {"k full", 512, 3840, 1, 0, false},
    {"o sliding", 3840, 4096, 1, 0, false},   {"o full", 3840, 8192, 1, 0, false},
    {"gate_up", 30720, 3840, 2, 15360, false}, {"down", 3840, 15360, 1, 0, false},
    // TP = 2 rank (GemmaConfig::Shard): q 2048 / 4096, k/v 1024, k full replicated 512, o K = 2048 / 4096,
    // gate_up 15360 (two parts of 7680), down K = 7680.
    {"q sliding tp2", 2048, 3840, 1, 0, true},   {"q full tp2", 4096, 3840, 1, 0, true},
    {"k/v sliding tp2", 1024, 3840, 1, 0, true}, {"k full tp2", 512, 3840, 1, 0, true},
    {"o sliding tp2", 3840, 2048, 1, 0, true},   {"o full tp2", 3840, 4096, 1, 0, true},
    {"gate_up tp2", 15360, 3840, 2, 7680, true}, {"down tp2", 3840, 7680, 1, 0, true},
};

// The 64-row kernel's rules for tuning t on (N, K) at m rows (linear.cpp TrellisRowFits + the part
// boundary rule of TrellisTuningFor). Returns "" when legal, else the broken rule.
std::string KernelRule(const Shape& s, const LinearTuning& t, int m) {
  const int64_t wc = static_cast<int64_t>(t.WV) * t.NPW * 32;
  if (wc <= 0 || wc > 256) return "Wc out of range";
  if (s.N % wc != 0) return "N % Wc != 0";
  if (s.parts > 1 && s.part_n0 % wc != 0) return "part boundary % Wc != 0";
  if ((s.K / 16) % (static_cast<int64_t>(t.SK) * t.SKG * t.U) != 0) return "(K/16) % (SK*SKG*U) != 0";
  if (static_cast<int64_t>(t.WV) * t.SK * 32 > 1024) return "WV*SK waves > 1024 threads";
  if (static_cast<int64_t>(t.SK) * wc * 32 > 64 * 1024) return "LDS > 64 KiB";
  const bool split = t.SKG > 1 || wc < 128;
  if (split && (m + 15) / 16 > t.MB) return "split group with more row tiles than MB";
  return "";
}

}  // namespace

int main() try {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  int plans = 0;
  for (const Shape& s : kShapes) {
    r4dx::model::SetTp2TuningForThisThread(s.tp2);
    Check(s.N % 128 == 0 && s.K % 128 == 0 && (s.parts == 1 || s.part_n0 % 128 == 0),
          std::string(s.name) + ": trellis needs N, K and the part boundary in whole 128-blocks");
    for (int kb : {4, 5}) {
      for (int m : {1, 16, 32, 64}) {
        const LinearTuning t = r4dx::model::PickTuning(Layout::kTrellis, s.N, s.K, m, kb);
        const std::string rule = KernelRule(s, t, m);
        // A pick that breaks the part-boundary rule alone is what linear.cpp's TrellisTuningFor repairs
        // with the 4.4 fallback; every other rule must hold of the pick itself.
        const bool boundary_only = rule == "part boundary % Wc != 0";
        Check(rule.empty() || boundary_only,
              std::string(s.name) + " KB" + std::to_string(kb) + " M=" + std::to_string(m) + ": tuning WV" +
                  std::to_string(t.WV) + " SK" + std::to_string(t.SK) + " MB" + std::to_string(t.MB) + " NPW" +
                  std::to_string(t.NPW) + " SKG" + std::to_string(t.SKG) + " U" + std::to_string(t.U) +
                  " breaks: " + rule);
      }
      const r4dx::model::TrellisM256Plan plan =
          r4dx::model::PlanTrellisM256(s.N, s.K, kb, s.parts, s.part_n0);
      std::printf("%-16s N=%5lld K=%5lld KB%d parts=%d | M=256 plan %s SK%d SKG%d skw%d%s%s\n", s.name,
                  static_cast<long long>(s.N), static_cast<long long>(s.K), kb, s.parts,
                  plan.ok ? "ok" : "REFUSED", plan.SK, plan.SKG, plan.SKW, plan.ok ? "" : " why: ",
                  plan.ok ? "" : plan.why.c_str());
      Check(plan.ok, std::string(s.name) + " KB" + std::to_string(kb) +
                         ": PlanTrellisM256 refuses (ApplyLinear would slice into 64-row launches): " +
                         plan.why);
      if (!plan.ok) continue;
      ++plans;
      const LinearTuning t64 = r4dx::model::PickTuning(Layout::kTrellis, s.N, s.K, 64, kb);
      const int64_t wc = static_cast<int64_t>(t64.WV) * t64.NPW * 32;
      if (s.parts == 1 || s.part_n0 % wc == 0) {
        Check(plan.SK == t64.SK && plan.SKG == t64.SKG,
              std::string(s.name) + " KB" + std::to_string(kb) + ": plan (SK " + std::to_string(plan.SK) +
                  ", SKG " + std::to_string(plan.SKG) + ") is not the M = 64 pick's (SK " +
                  std::to_string(t64.SK) + ", SKG " + std::to_string(t64.SKG) + ")");
      }
      const int n_split = static_cast<int>(s.parts > 1 ? s.part_n0 : s.N);
      const char* why = r4dx::core::r4d::GemmTrellisM256Check(
          static_cast<int>(r4dx::model::kTrellisM256Rows), static_cast<int>(s.K), static_cast<int>(s.N),
          n_split, kb, plan.SK, /*NP=*/1, plan.SKG, /*U=*/4, plan.SKW);
      Check(why == nullptr, std::string(s.name) + " KB" + std::to_string(kb) +
                                ": the M = 256 check rejects the plan: " + (why ? why : ""));
    }
  }
  r4dx::model::SetTp2TuningForThisThread(false);

  // The refusals stay explicit and carry a reason (the model slices on them; nothing launches).
  {
    const auto bad_kb = r4dx::model::PlanTrellisM256(3840, 3840, 6, 1, 0);
    Check(!bad_kb.ok && !bad_kb.why.empty(), "KB 6 is refused with a reason");
    const auto bad_parts = r4dx::model::PlanTrellisM256(3840, 3840, 4, 3, 1920);
    Check(!bad_parts.ok && !bad_parts.why.empty(), "three parts is refused with a reason");
    const auto bad_n = r4dx::model::PlanTrellisM256(500, 3840, 4, 1, 0);
    Check(!bad_n.ok && !bad_n.why.empty(), "N = 500 (not whole 128-blocks) is refused with a reason");
  }

  std::printf("test_trellis_gemma_plan: %d checks, %d (shape, KB) plans ok, %d failed\n", g_checks, plans,
              g_failures);
  return g_failures == 0 ? 0 : 1;
} catch (const std::exception& e) {
  std::fprintf(stderr, "FAIL: exception: %s\n", e.what());
  return 1;
}
