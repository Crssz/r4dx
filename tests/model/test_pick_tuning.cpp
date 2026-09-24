// tests/model/test_pick_tuning.cpp -- every tuning PickTuning can hand out, for every (layout, N, K)
// the GEMM tuning table covers and every chunk M in 1..64, must be one this build's r4d_gemm_*
// kernels accept. The table is swept at ONE w4a16 group (tools/profile/tune_gemm.py) but compiled
// into both R4DX_W4A16_GROUP builds, and r4d_gemm_w4a16_nt_m64 throws when K % (SK * group) != 0:
// a group-64 row with SK=16 at K=5120 is exactly that on a group-128 build, so ResolveTuning
// (src/model/linear.cpp) must skip it. Host-only -- PickTuning and the *_group() exports are plain
// host functions, no HIP call is made -- so it runs on every build, like test_mtp_round.
#include <cstdint>
#include <cstdio>

#include "linear.h"
#include "r4d.h"

namespace r4dx::model {
namespace {
// The test's own copy of the rows, for the set of (layout, N, K) shapes to probe.
#include "gemm_tuning_table.inc"
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

}  // namespace

int main() {
  int failures = 0, checked = 0;
  for (const auto& row : r4dx::model::kGemmTuningTable) {
    for (int64_t m = 1; m <= 64; ++m) {
      const LinearTuning t = r4dx::model::PickTuning(row.layout, row.N, row.K, m);
      ++checked;
      if (!Launchable(row.layout, row.K, t)) {
        std::fprintf(stderr,
                     "FAIL layout=%d N=%lld K=%lld M=%lld -> WV=%d SK=%d MB=%d NPW=%d is not "
                     "launchable at this build's groups (w4a16=%d w4a8=%d mxfp4=%d)\n",
                     static_cast<int>(row.layout), static_cast<long long>(row.N),
                     static_cast<long long>(row.K), static_cast<long long>(m), t.WV, t.SK, t.MB,
                     t.NPW, r4d_gemm_w4a16_nt_m64_group(), r4d_gemm_w4a8_nt_m64_group(),
                     r4d_gemm_mxfp4a8_nt_m64_group());
        ++failures;
      }
    }
  }
  std::printf("test_pick_tuning: %d/%d PickTuning results launchable (w4a16 group %d)\n",
              checked - failures, checked, r4d_gemm_w4a16_nt_m64_group());
  return failures == 0 ? 0 : 1;
}
