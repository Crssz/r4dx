// tests/model/test_pick_tuning.cpp -- every tuning PickTuning can hand out, for every (layout, N, K)
// the GEMM tuning table covers and every chunk M in 1..64, must be one this build's r4d_gemm_*
// kernels accept. The table is swept at ONE w4a16 group (tools/profile/tune_gemm.py) but compiled
// into both R4DX_W4A16_GROUP builds, and r4d_gemm_w4a16_nt_m64 throws when K % (SK * group) != 0:
// a group-64 row with SK=16 at K=5120 is exactly that on a group-128 build, so ResolveTuning
// (src/model/linear.cpp) must skip it. And every chunk of M <= 16 rows must get the SK the M=1
// band gets (CheckRowTileSk below). Host-only -- PickTuning and the *_group() exports are plain
// host functions, no HIP call is made -- so it runs on every build, like test_mtp_round.
#include <cstdint>
#include <cstdio>

#include "linear.h"
#include "r4d.h"

namespace r4dx::model {
namespace {
// The test's own copy of the rows, for the set of (layout, N, K) shapes to probe.
#include "gemm_tuning_table.inc"
// And of the tensor-parallel per-rank rows (docs/tp.md 2.7), which a TP rank thread consults
// first: same sweep, same build-group hazard.
namespace tp2 {
#include "gemm_tuning_table_tp2.inc"
}  // namespace tp2
}  // namespace
}  // namespace r4dx::model

namespace {

using r4dx::model::Layout;
using r4dx::model::LinearTuning;

// The launch checks each kernel's entry point makes (third_party/libr4d/r4d_gemm_*_nt_m64.hip),
// against the groups this binary was built with.
bool Launchable(Layout layout, int64_t K, const LinearTuning& t) {
  if (t.WV * t.SK * 32 > 1024 || t.MB < 1 || t.MB > 4) return false;
  switch (layout) {
    case Layout::kBf16:
      return K % (t.SK * 16) == 0;
    case Layout::kW4a16:
      return K % (t.SK * r4d_gemm_w4a16_nt_m64_group()) == 0 && (t.NPW == 1 || t.NPW == 4) &&
             t.WV * t.NPW * t.SK <= 64;
    case Layout::kW4a8:
      return K % (t.SK * r4d_gemm_w4a8_nt_m64_group()) == 0 &&
             (t.NPW == 1 || t.NPW == 2 || t.NPW == 4 || t.NPW == 8) && t.WV * t.NPW * t.SK <= 64;
    case Layout::kMxfp4:
      return K % (t.SK * r4d_gemm_mxfp4a8_nt_m64_group()) == 0 &&
             (t.NPW == 1 || t.NPW == 2 || t.NPW == 4 || t.NPW == 8) && t.WV * t.NPW * t.SK <= 64;
  }
  return false;
}

template <size_t kRows>
void CheckTable(const r4dx::model::GemmTuningRow (&table)[kRows], const char* which, int& checked,
                int& failures) {
  for (const auto& row : table) {
    for (int64_t m = 1; m <= 64; ++m) {
      const LinearTuning t = r4dx::model::PickTuning(row.layout, row.N, row.K, m);
      ++checked;
      if (!Launchable(row.layout, row.K, t)) {
        std::fprintf(stderr,
                     "FAIL (%s) layout=%d N=%lld K=%lld M=%lld -> WV=%d SK=%d MB=%d NPW=%d is not "
                     "launchable at this build's groups (w4a16=%d w4a8=%d mxfp4=%d)\n",
                     which, static_cast<int>(row.layout), static_cast<long long>(row.N),
                     static_cast<long long>(row.K), static_cast<long long>(m), t.WV, t.SK, t.MB,
                     t.NPW, r4d_gemm_w4a16_nt_m64_group(), r4d_gemm_w4a8_nt_m64_group(),
                     r4d_gemm_mxfp4a8_nt_m64_group());
        ++failures;
      }
    }
  }
}

// A speculative verify window (M = k+1 <= 16 rows) must compute every row bit-identically to the
// single-row decode (M=1) of the same context, or a sampled round can emit a different token than
// plain sampled decode (docs/mtp.md, "Sampled rounds are bit-exact"). In these kernels a row's
// arithmetic depends on SK and nothing else (linear.cpp's kRowTile), so every M in 1..16 must get
// the SK that M=1 gets.
template <size_t kRows>
void CheckRowTileSk(const r4dx::model::GemmTuningRow (&table)[kRows], const char* which,
                    int& checked, int& failures) {
  for (const auto& row : table) {
    const LinearTuning t1 = r4dx::model::PickTuning(row.layout, row.N, row.K, 1);
    for (int64_t m = 2; m <= 16; ++m) {
      const LinearTuning t = r4dx::model::PickTuning(row.layout, row.N, row.K, m);
      ++checked;
      if (t.SK != t1.SK) {
        std::fprintf(stderr,
                     "FAIL (%s) layout=%d N=%lld K=%lld: M=%lld gets SK=%d but M=1 gets SK=%d -- a "
                     "verify row of that width would not sum in decode's order\n",
                     which, static_cast<int>(row.layout), static_cast<long long>(row.N),
                     static_cast<long long>(row.K), static_cast<long long>(m), t.SK, t1.SK);
        ++failures;
      }
    }
  }
}

}  // namespace

int main() {
  int failures = 0, checked = 0;
  CheckTable(r4dx::model::kGemmTuningTable, "main table", checked, failures);
  // A thread that loaded a tensor-parallel rank (docs/tp.md 2.7) sees the TP table first and the
  // main table behind it: every row of both must still be launchable there.
  r4dx::model::SetTp2TuningForThisThread(true);
  CheckTable(r4dx::model::tp2::kGemmTuningTable, "tp2 table, tp thread", checked, failures);
  CheckTable(r4dx::model::kGemmTuningTable, "main table, tp thread", checked, failures);
  r4dx::model::SetTp2TuningForThisThread(false);
  std::printf("test_pick_tuning: %d/%d PickTuning results launchable (w4a16 group %d)\n",
              checked - failures, checked, r4d_gemm_w4a16_nt_m64_group());

  int sk_failures = 0, sk_checked = 0;
  CheckRowTileSk(r4dx::model::kGemmTuningTable, "main table", sk_checked, sk_failures);
  r4dx::model::SetTp2TuningForThisThread(true);
  CheckRowTileSk(r4dx::model::tp2::kGemmTuningTable, "tp2 table, tp thread", sk_checked,
                 sk_failures);
  CheckRowTileSk(r4dx::model::kGemmTuningTable, "main table, tp thread", sk_checked, sk_failures);
  r4dx::model::SetTp2TuningForThisThread(false);
  std::printf("test_pick_tuning: %d/%d M=2..16 picks share M=1's SK\n", sk_checked - sk_failures,
              sk_checked);
  return failures == 0 && sk_failures == 0 ? 0 : 1;
}
