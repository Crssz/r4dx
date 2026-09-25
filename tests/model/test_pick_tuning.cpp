// tests/model/test_pick_tuning.cpp -- every tuning PickTuning can hand out, for every (layout, N, K)
// the GEMM tuning table covers and every chunk M in 1..64, must be one this build's r4d_gemm_*
// kernels accept. The table is swept at ONE w4a16 group (tools/profile/tune_gemm.py) but compiled
// into both R4DX_W4A16_GROUP builds, and r4d_gemm_w4a16_nt_m64 throws when K % (SK * group) != 0:
// a group-64 row with SK=16 at K=5120 is exactly that on a group-128 build, so ResolveTuning
// (src/model/linear.cpp) must skip it. And every chunk of M <= 16 rows must get the SK the M=1
// band gets (CheckRowTileSk below). Host-only -- PickTuning and the *_group() exports are plain
// host functions, no HIP call is made -- so it runs on every build, like test_mtp_round.
//
// quant2 Q3 (docs/quant2.md section 5.1): a w4a16 linear may carry its own group (32, 64 or 128,
// QuantLinear::w4a16_group), and the group is part of the tuning key. So, for every w4a16 shape:
//   - at the build default (w4a16_group 0 or the default itself) PickTuning must return exactly
//     what the pre-Q3 resolution returned (LegacyPick below is that code, verbatim) -- the default
//     path's tuning, like its kernel entry, does not move;
//   - at every group the kernel instantiates, the pick must be launchable at THAT group, whose rule
//     is K % (SK * max(group, 64)) (a split starts on a 64-K packed block, which 32 does not
//     guarantee), and M = 2..16 must still share M=1's SK;
//   - a non-default group has no table rows (every generated row is group 0 = the default), so it
//     must get FallbackTuning's WV4/SK4/MB1/NPW1 (NT=0 on the M = 2..16 row tile, NT=1 otherwise).
#include <algorithm>
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

using r4dx::model::GemmTuningRow;
using r4dx::model::Layout;
using r4dx::model::LinearTuning;

// Every w4a16 group the checks below run at: 0 is "the build default" (QuantLinear's default),
// then the three per-tensor groups r4d_gemm_w4a16_nt_m64_g instantiates.
constexpr int kW4a16Groups[] = {0, 32, 64, 128};

int EffectiveGroup(int g) { return g == 0 ? r4d_gemm_w4a16_nt_m64_group() : g; }

// The launch checks each kernel's entry point makes (third_party/libr4d/r4d_gemm_*_nt_m64.hip),
// against the groups this binary was built with -- for w4a16, at the linear's own group.
bool Launchable(Layout layout, int64_t K, const LinearTuning& t, int w4a16_group) {
  if (t.WV * t.SK * 32 > 1024 || t.MB < 1 || t.MB > 4) return false;
  switch (layout) {
    case Layout::kBf16:
      return K % (t.SK * 16) == 0;
    case Layout::kW4a16: {
      // The build default runs through the ungrouped entry (linear.cpp), which serves it whatever
      // it is; any other group needs the per-group entry to instantiate it.
      const int g = EffectiveGroup(w4a16_group);
      const bool served =
          g == r4d_gemm_w4a16_nt_m64_group() || r4d_gemm_w4a16_nt_m64_has_group(g) != 0;
      return served && K % (t.SK * std::max(g, 64)) == 0 &&
             (t.NPW == 1 || t.NPW == 4) && t.WV * t.NPW * t.SK <= 64;
    }
    case Layout::kW4a8:
      return K % (t.SK * r4d_gemm_w4a8_nt_m64_group()) == 0 &&
             (t.NPW == 1 || t.NPW == 2 || t.NPW == 4 || t.NPW == 8) && t.WV * t.NPW * t.SK <= 64;
    case Layout::kMxfp4:
      return K % (t.SK * r4d_gemm_mxfp4a8_nt_m64_group()) == 0 &&
             (t.NPW == 1 || t.NPW == 2 || t.NPW == 4 || t.NPW == 8) && t.WV * t.NPW * t.SK <= 64;
  }
  return false;
}

bool Same(const LinearTuning& a, const LinearTuning& b) {
  return a.WV == b.WV && a.SK == b.SK && a.MB == b.MB && a.NPW == b.NPW && a.NT == b.NT;
}

// The pre-Q3 resolution (linear.cpp at e18a3a9: BestRow + ResolveTuning, minus the cache), kept
// here as the reference the default group must still match.
template <size_t kRows>
const GemmTuningRow* LegacyBestRow(const GemmTuningRow (&table)[kRows], Layout layout, int64_t N,
                                   int64_t K, int64_t M) {
  static const int64_t kW4a16Group = r4d_gemm_w4a16_nt_m64_group();
  const GemmTuningRow* best = nullptr;
  for (const GemmTuningRow& row : table) {
    if (row.layout != layout || row.N != N || row.K != K) continue;
    if (row.M < M) continue;
    if (layout == Layout::kW4a16 && K % (row.tuning.SK * kW4a16Group) != 0) continue;
    if (best == nullptr || row.M < best->M) best = &row;
  }
  return best;
}

LinearTuning LegacyPick(Layout layout, int64_t N, int64_t K, int64_t M, bool tp2) {
  if (M > 1 && M <= 16) {
    LinearTuning t = LegacyPick(layout, N, K, 1, tp2);
    if (layout == Layout::kW4a16) t.NT = 0;
    return t;
  }
  if (tp2) {
    if (const GemmTuningRow* row =
            LegacyBestRow(r4dx::model::tp2::kGemmTuningTable, layout, N, K, M)) {
      return row->tuning;
    }
  }
  if (const GemmTuningRow* row = LegacyBestRow(r4dx::model::kGemmTuningTable, layout, N, K, M)) {
    return row->tuning;
  }
  return LinearTuning{4, 4, 1, 1, 1};  // FallbackTuning; every shape here has K % 512 == 0
}

// The groups to probe for a row: every w4a16 group for a w4a16 row, just 0 for the others (their
// PickTuning ignores the group).
template <class Fn>
void ForEachGroup(Layout layout, Fn fn) {
  if (layout != Layout::kW4a16) {
    fn(0);
    return;
  }
  for (int g : kW4a16Groups) fn(g);
}

template <size_t kRows>
void CheckTable(const GemmTuningRow (&table)[kRows], const char* which, bool tp2, int& checked,
                int& failures) {
  for (const auto& row : table) {
    ForEachGroup(row.layout, [&](int g) {
      for (int64_t m = 1; m <= 64; ++m) {
        const LinearTuning t = r4dx::model::PickTuning(row.layout, row.N, row.K, m, g);
        ++checked;
        if (!Launchable(row.layout, row.K, t, g)) {
          std::fprintf(stderr,
                       "FAIL (%s) layout=%d N=%lld K=%lld M=%lld w4a16_group=%d -> WV=%d SK=%d "
                       "MB=%d NPW=%d is not launchable at this build's groups (w4a16 default=%d "
                       "w4a8=%d mxfp4=%d)\n",
                       which, static_cast<int>(row.layout), static_cast<long long>(row.N),
                       static_cast<long long>(row.K), static_cast<long long>(m), g, t.WV, t.SK,
                       t.MB, t.NPW, r4d_gemm_w4a16_nt_m64_group(), r4d_gemm_w4a8_nt_m64_group(),
                       r4d_gemm_mxfp4a8_nt_m64_group());
          ++failures;
        }
        ++checked;
        if (row.layout != Layout::kW4a16 || EffectiveGroup(g) == r4d_gemm_w4a16_nt_m64_group()) {
          // The default group (0, or the default spelled out) resolves exactly as before Q3.
          const LinearTuning want = LegacyPick(row.layout, row.N, row.K, m, tp2);
          if (!Same(t, want)) {
            std::fprintf(stderr,
                         "FAIL (%s) layout=%d N=%lld K=%lld M=%lld w4a16_group=%d: WV=%d SK=%d "
                         "MB=%d NPW=%d NT=%d, the pre-Q3 pick was WV=%d SK=%d MB=%d NPW=%d NT=%d\n",
                         which, static_cast<int>(row.layout), static_cast<long long>(row.N),
                         static_cast<long long>(row.K), static_cast<long long>(m), g, t.WV, t.SK,
                         t.MB, t.NPW, t.NT, want.WV, want.SK, want.MB, want.NPW, want.NT);
            ++failures;
          }
        } else {
          // A non-default group: no row is keyed to it, so FallbackTuning (row-tile NT rule aside).
          const LinearTuning want{4, 4, 1, 1, (m > 1 && m <= 16) ? 0 : 1};
          if (!Same(t, want)) {
            std::fprintf(stderr,
                         "FAIL (%s) N=%lld K=%lld M=%lld w4a16_group=%d: WV=%d SK=%d MB=%d NPW=%d "
                         "NT=%d, expected FallbackTuning (a group-0 row must not serve group %d)\n",
                         which, static_cast<long long>(row.N), static_cast<long long>(row.K),
                         static_cast<long long>(m), g, t.WV, t.SK, t.MB, t.NPW, t.NT, g);
            ++failures;
          }
        }
      }
    });
  }
}

// A speculative verify window (M = k+1 <= 16 rows) must compute every row bit-identically to the
// single-row decode (M=1) of the same context, or a sampled round can emit a different token than
// plain sampled decode (docs/mtp.md, "Sampled rounds are bit-exact"). In these kernels a row's
// arithmetic depends on SK and nothing else (linear.cpp's kRowTile), so every M in 1..16 must get
// the SK that M=1 gets -- at every w4a16 group.
template <size_t kRows>
void CheckRowTileSk(const GemmTuningRow (&table)[kRows], const char* which, int& checked,
                    int& failures) {
  for (const auto& row : table) {
    ForEachGroup(row.layout, [&](int g) {
      const LinearTuning t1 = r4dx::model::PickTuning(row.layout, row.N, row.K, 1, g);
      for (int64_t m = 2; m <= 16; ++m) {
        const LinearTuning t = r4dx::model::PickTuning(row.layout, row.N, row.K, m, g);
        ++checked;
        if (t.SK != t1.SK) {
          std::fprintf(stderr,
                       "FAIL (%s) layout=%d N=%lld K=%lld w4a16_group=%d: M=%lld gets SK=%d but "
                       "M=1 gets SK=%d -- a verify row of that width would not sum in decode's "
                       "order\n",
                       which, static_cast<int>(row.layout), static_cast<long long>(row.N),
                       static_cast<long long>(row.K), g, static_cast<long long>(m), t.SK, t1.SK);
          ++failures;
        }
      }
    });
  }
}

// The kernel's per-group entry: which groups it serves (r4d.h), and the group normalisation
// ApplyLinear dispatches on (0 -> the historical entry's group).
void CheckGroupExports(int& checked, int& failures) {
  const auto expect = [&](bool ok, const char* what) {
    ++checked;
    if (!ok) {
      std::fprintf(stderr, "FAIL: %s\n", what);
      ++failures;
    }
  };
  expect(r4d_gemm_w4a16_nt_m64_has_group(32) == 1 && r4d_gemm_w4a16_nt_m64_has_group(64) == 1 &&
             r4d_gemm_w4a16_nt_m64_has_group(128) == 1,
         "r4d_gemm_w4a16_nt_m64_has_group(32/64/128) == 1");
  // Exactly 32/64/128 (r4d.h): a build default outside them (192, 256) is the ungrouped entry's
  // alone, so has_group() must not answer for it.
  expect(r4d_gemm_w4a16_nt_m64_has_group(16) == 0 && r4d_gemm_w4a16_nt_m64_has_group(48) == 0 &&
             r4d_gemm_w4a16_nt_m64_has_group(96) == 0 && r4d_gemm_w4a16_nt_m64_has_group(0) == 0 &&
             r4d_gemm_w4a16_nt_m64_has_group(-64) == 0 &&
             r4d_gemm_w4a16_nt_m64_has_group(192) == 0 && r4d_gemm_w4a16_nt_m64_has_group(256) == 0,
         "r4d_gemm_w4a16_nt_m64_has_group(16/48/96/0/-64/192/256) == 0");
  expect(r4dx::model::EffectiveW4a16Group(0) == r4d_gemm_w4a16_nt_m64_group() &&
             r4dx::model::EffectiveW4a16Group(32) == 32 &&
             r4dx::model::EffectiveW4a16Group(128) == 128,
         "EffectiveW4a16Group: 0 -> the build default, any other group unchanged");
  // PickTuning's cache is keyed on the effective group: the default spelled 0 and spelled out must
  // agree with the pre-Q3 pick, and neither may be served a non-default group's entry cached first.
  const LinearTuning g32 = r4dx::model::PickTuning(Layout::kW4a16, 5120, 17408, 1, 32);
  const LinearTuning d0 = r4dx::model::PickTuning(Layout::kW4a16, 5120, 17408, 1, 0);
  const LinearTuning dn =
      r4dx::model::PickTuning(Layout::kW4a16, 5120, 17408, 1, r4d_gemm_w4a16_nt_m64_group());
  expect(Same(d0, dn) && Same(d0, LegacyPick(Layout::kW4a16, 5120, 17408, 1, false)),
         "PickTuning: w4a16_group 0 and the default spelled out share the pre-Q3 resolution");
  expect(Same(g32, LinearTuning{4, 4, 1, 1, 1}),
         "PickTuning: mlp.down at g32 gets FallbackTuning, not the group-0 row");
}

}  // namespace

int main() {
  int failures = 0, checked = 0;
  CheckTable(r4dx::model::kGemmTuningTable, "main table", false, checked, failures);
  // A thread that loaded a tensor-parallel rank (docs/tp.md 2.7) sees the TP table first and the
  // main table behind it: every row of both must still be launchable there.
  r4dx::model::SetTp2TuningForThisThread(true);
  CheckTable(r4dx::model::tp2::kGemmTuningTable, "tp2 table, tp thread", true, checked, failures);
  CheckTable(r4dx::model::kGemmTuningTable, "main table, tp thread", true, checked, failures);
  r4dx::model::SetTp2TuningForThisThread(false);
  std::printf("test_pick_tuning: %d/%d PickTuning checks passed -- launchable, and the pre-Q3 pick "
              "at the default w4a16 group %d (w4a16 probed at groups 0/32/64/128)\n",
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

  int ex_failures = 0, ex_checked = 0;
  CheckGroupExports(ex_checked, ex_failures);
  std::printf("test_pick_tuning: %d/%d per-group export checks\n", ex_checked - ex_failures,
              ex_checked);
  return failures == 0 && sk_failures == 0 && ex_failures == 0 ? 0 : 1;
}
