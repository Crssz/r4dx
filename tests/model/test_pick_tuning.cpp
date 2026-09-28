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
//
// Trellis (docs/trellis-kernel.md 4.3, 4.4, 5.3, 6): src/model/gemm_tuning_table_trellis.inc's
// rows and, on a TP = 2 rank thread first, gemm_tuning_table_trellis_tp2.inc's per-rank rows (M5),
// keyed by rate (KB) as well as (N, K):
//   - every pick for M = 1..64 is one r4d_gemm_trellis_nt_m64 accepts (TrellisLaunchable, the
//     kernel's own r4d_trellis_check and instantiation table, transcribed), with the linear's part
//     boundary (mlp.gate_up: n_split = N / 2), on every row's shape, on every per-rank shape of a
//     TP = 2 thread, and on the tiny test container's shapes (the fallback alone);
//   - M = 2..16 get the M = 1 pick whole -- SK, SKG, block width, MT and U set the summation order,
//     and trellis keeps NT too (10.1) -- so a verify row is bit-identical to the decode row;
//   - every pick is the one ExpectedTrellis names, BestRow spelled out independently: the M = 1 row
//     of the linear's own rate for M <= 16, and for M > 16 (prefill chunks, M5's rows at M = 32 and
//     64) the row of the smallest M >= the chunk's that fits it -- the TP = 2 rows first on a TP
//     thread, never on a TP = 1 one -- else 4.4's M-aware fallback (SKG capped at 4; unsplit with
//     every row tile in its block for M > 16); every prefill row is the pick at its own M.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <iterator>
#include <vector>

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
// And the trellis rows (docs/trellis-kernel.md 5.3), and their TP = 2 per-rank sibling (M5), which
// a TP rank thread consults before them.
namespace trellis {
#include "gemm_tuning_table_trellis.inc"
}  // namespace trellis
namespace trellis_tp2 {
#include "gemm_tuning_table_trellis_tp2.inc"
}  // namespace trellis_tp2
}  // namespace
}  // namespace r4dx::model

// tests/kernels/trellis_tuning_rows.hpp re-declares Layout, LinearTuning and GemmTuningRow so the
// kernel tests read the same trellis rows without linking r4dx_model_linear. The rows are positional,
// so the copies must stay these structs field for field: a field reordered in one (SKG and U swapped,
// say) would still compile in both and hand the kernel tests other tunings than production runs.
// Checked here, where both are visible: the layouts at compile time, and the rows' values field by
// field in CheckTrellisRowsCopy below.
#include "../kernels/trellis_tuning_rows.hpp"

namespace {

using ModelTuning = r4dx::model::LinearTuning;
using CopyTuning = trellis_rows::LinearTuning;
using ModelRow = r4dx::model::GemmTuningRow;
using CopyRow = trellis_rows::GemmTuningRow;
static_assert(sizeof(ModelTuning) == sizeof(CopyTuning) &&
                  offsetof(ModelTuning, WV) == offsetof(CopyTuning, WV) &&
                  offsetof(ModelTuning, SK) == offsetof(CopyTuning, SK) &&
                  offsetof(ModelTuning, MB) == offsetof(CopyTuning, MB) &&
                  offsetof(ModelTuning, NPW) == offsetof(CopyTuning, NPW) &&
                  offsetof(ModelTuning, NT) == offsetof(CopyTuning, NT) &&
                  offsetof(ModelTuning, SKG) == offsetof(CopyTuning, SKG) &&
                  offsetof(ModelTuning, U) == offsetof(CopyTuning, U),
              "tests/kernels/trellis_tuning_rows.hpp's LinearTuning is not linear.h's, field for field");
static_assert(sizeof(ModelRow) == sizeof(CopyRow) &&
                  offsetof(ModelRow, layout) == offsetof(CopyRow, layout) &&
                  offsetof(ModelRow, N) == offsetof(CopyRow, N) &&
                  offsetof(ModelRow, K) == offsetof(CopyRow, K) &&
                  offsetof(ModelRow, M) == offsetof(CopyRow, M) &&
                  offsetof(ModelRow, tuning) == offsetof(CopyRow, tuning) &&
                  offsetof(ModelRow, group) == offsetof(CopyRow, group) &&
                  offsetof(ModelRow, rate) == offsetof(CopyRow, rate),
              "tests/kernels/trellis_tuning_rows.hpp's GemmTuningRow is not linear.h's, field for field");
constexpr bool SameLayoutValue(trellis_rows::Layout a, r4dx::model::Layout b) {
  return static_cast<int>(a) == static_cast<int>(b);
}
static_assert(SameLayoutValue(trellis_rows::Layout::kBf16, r4dx::model::Layout::kBf16) &&
                  SameLayoutValue(trellis_rows::Layout::kMxfp4, r4dx::model::Layout::kMxfp4) &&
                  SameLayoutValue(trellis_rows::Layout::kW4a16, r4dx::model::Layout::kW4a16) &&
                  SameLayoutValue(trellis_rows::Layout::kW4a8, r4dx::model::Layout::kW4a8) &&
                  SameLayoutValue(trellis_rows::Layout::kTrellis, r4dx::model::Layout::kTrellis),
              "tests/kernels/trellis_tuning_rows.hpp's Layout is not quant_linear.h's, value for value");

// The copy's rows read by field name equal production's: the same file through both declarations
// (the TP = 1 table, then the TP = 2 one).
template <size_t kProd, size_t kCopy>
void CheckTrellisRowsCopy(const ModelRow (&prod)[kProd], const CopyRow (&copy)[kCopy],
                          const char* which, int& checked, int& failures) {
  ++checked;
  if (kProd != kCopy) {
    std::fprintf(stderr, "FAIL trellis_tuning_rows.hpp (%s): %zu rows, production has %zu\n", which,
                 kCopy, kProd);
    ++failures;
    return;
  }
  for (size_t i = 0; i < kProd; ++i) {
    const ModelRow& a = prod[i];
    const CopyRow& b = copy[i];
    const ModelTuning& t = a.tuning;
    const CopyTuning& u = b.tuning;
    ++checked;
    if (!SameLayoutValue(b.layout, a.layout) || a.N != b.N || a.K != b.K || a.M != b.M ||
        a.group != b.group || a.rate != b.rate || t.WV != u.WV || t.SK != u.SK || t.MB != u.MB ||
        t.NPW != u.NPW || t.NT != u.NT || t.SKG != u.SKG || t.U != u.U) {
      std::fprintf(stderr, "FAIL trellis_tuning_rows.hpp (%s) row %zu (N=%lld K=%lld) differs from "
                           "production's by field name\n",
                   which, i, static_cast<long long>(a.N), static_cast<long long>(a.K));
      ++failures;
    }
  }
}

}  // namespace

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
    case Layout::kTrellis:
      return false;  // TrellisLaunchable below: it needs the chunk M, rate and part boundary
  }
  return false;
}

// third_party/libr4d/r4d_gemm_trellis_nt_m64.hip's r4d_tq_max_mt: the largest instantiated MT per
// (KB, NP, U), 0 where (NP, U) is not instantiated.
int TrellisMaxMt(int KB, int NP, int U) {
  if (KB == 4) {
    if (NP == 1) return U == 1 || U == 2 || U == 4 ? 4 : 0;
    if (NP == 2) return U == 1 ? 4 : (U == 2 || U == 4) ? 3 : 0;
    if (NP == 4) return U == 1 ? 2 : U == 2 ? 1 : 0;
    return 0;
  }
  if (KB == 5) {
    if (NP == 1) return U == 1 || U == 2 || U == 4 ? 4 : 0;
    if (NP == 2) return U == 1 ? 4 : U == 2 ? 3 : U == 4 ? 2 : 0;
    if (NP == 4) return U == 1 ? 2 : U == 2 ? 1 : 0;
    return 0;
  }
  return 0;
}

// r4d_trellis_check (the same file), rule for rule, for one M-row chunk of a linear [N, K] at rate
// KB whose second part starts at column n_split (N for one part), with ws and tickets given.
bool TrellisLaunchable(int64_t N, int64_t K, int64_t n_split, int64_t M, int KB,
                       const LinearTuning& t) {
  const int MT = t.MB, NP = t.NPW;
  if (M < 1 || M > r4d_gemm_trellis_nt_m64_max_m()) return false;
  if (K <= 0 || N <= 0 || K % 128 != 0 || N % 128 != 0) return false;
  if (r4d_gemm_trellis_nt_m64_has_rate(KB) == 0) return false;
  if (t.WV != 1 && t.WV != 2 && t.WV != 4) return false;
  if (t.SK != 1 && t.SK != 2 && t.SK != 4 && t.SK != 8 && t.SK != 16) return false;
  if (t.SKG != 1 && t.SKG != 2 && t.SKG != 4 && t.SKG != 8) return false;
  if (t.NT != 0 && t.NT != 1) return false;
  if (MT < 1 || MT > TrellisMaxMt(KB, NP, t.U)) return false;
  const int64_t wc = static_cast<int64_t>(t.WV) * NP * 32;
  if (wc > 256 || N % wc != 0) return false;
  if (n_split < 0 || n_split > N || n_split % 128 != 0 || n_split % wc != 0) return false;
  if ((K / 16) % (static_cast<int64_t>(t.SK) * t.SKG * t.U) != 0) return false;
  if (t.WV * t.SK * 32 > 1024) return false;
  if (static_cast<int64_t>(t.SK) * wc * 8 * 4 > 64 * 1024) return false;
  const bool split = t.SKG > 1 || wc < 128;
  return !split || (M + 15) / 16 <= MT;
}

bool SameTrellis(const LinearTuning& a, const LinearTuning& b) {
  return a.WV == b.WV && a.SK == b.SK && a.MB == b.MB && a.NPW == b.NPW && a.NT == b.NT &&
         a.SKG == b.SKG && a.U == b.U;
}

// docs/trellis-kernel.md 4.4's fallback, spelled out independently of linear.cpp.
LinearTuning TrellisFallback(int64_t N, int64_t K, int64_t M) {
  LinearTuning t{4, 2, 1, 1, 1};
  t.U = 2;
  if (M > 16) {
    t.SKG = 1;
    t.MB = static_cast<int>(std::min<int64_t>(4, (M + 15) / 16));
    t.NT = 0;
    return t;
  }
  int skg = static_cast<int>(std::max<int64_t>(1, std::min<int64_t>(4, 128 / (N / 128))));
  int p = 1;
  while (p * 2 <= skg) p *= 2;
  skg = p;
  while (skg > 1 && (K / 16) % (2 * skg * 2) != 0) skg /= 2;
  t.SKG = skg;
  return t;
}

// linear.cpp's BestRow over one trellis table, spelled out independently: the row of the smallest
// M >= m at (N, K, KB) that fits an m-row chunk (whole Wc blocks, whole U-steps of every K slice, a
// split 128-group only with every row tile in its block), or nullptr.
template <size_t kRows>
const GemmTuningRow* ExpectedRow(const GemmTuningRow (&table)[kRows], int64_t N, int64_t K, int kb,
                                 int64_t m) {
  const GemmTuningRow* best = nullptr;
  for (const GemmTuningRow& row : table) {
    if (row.layout != Layout::kTrellis || row.N != N || row.K != K || row.rate != kb || row.M < m) {
      continue;
    }
    const LinearTuning& t = row.tuning;
    const int64_t wc = static_cast<int64_t>(t.WV) * t.NPW * 32;
    if (N % wc != 0 || (K / 16) % (static_cast<int64_t>(t.SK) * t.SKG * t.U) != 0) continue;
    if ((t.SKG > 1 || wc < 128) && (m + 15) / 16 > t.MB) continue;
    if (best == nullptr || row.M < best->M) best = &row;
  }
  return best;
}

// The pick for an m-row chunk: a chunk of <= 16 rows resolves through the M = 1 band; on a TP
// thread the TP = 2 rows (M5) come first, then the TP = 1 rows (the M = 1 rows and M5's M = 32 / 64
// prefill rows), then 4.4's fallback.
LinearTuning ExpectedTrellis(int64_t N, int64_t K, int kb, int64_t m, bool tp2) {
  const int64_t band = m <= 16 ? 1 : m;
  if (tp2) {
    if (const GemmTuningRow* row =
            ExpectedRow(r4dx::model::trellis_tp2::kGemmTuningTable, N, K, kb, band)) {
      return row->tuning;
    }
  }
  if (const GemmTuningRow* row =
          ExpectedRow(r4dx::model::trellis::kGemmTuningTable, N, K, kb, band)) {
    return row->tuning;
  }
  return TrellisFallback(N, K, band);
}

struct TrellisShape {
  int64_t N, K, n_split;
  const char* what;
};

// One shape at one rate, on a TP thread when `tp2` (the caller sets the flag): M = 1..64 launchable
// and each the pick ExpectedTrellis names, M = 2..16 the M = 1 pick whole, and (when `want1` is
// given) the M = 1 pick equal to it.
void CheckTrellisShape(const TrellisShape& s, int kb, const LinearTuning* want1, bool tp2,
                       const char* which, int& checked, int& failures) {
  const LinearTuning t1 = r4dx::model::PickTuning(Layout::kTrellis, s.N, s.K, 1, kb);
  const auto fail = [&](int64_t m, const LinearTuning& t, const char* why) {
    std::fprintf(stderr,
                 "FAIL (%s) trellis %s N=%lld K=%lld n_split=%lld KB=%d M=%lld -> WV=%d SK=%d "
                 "MT=%d NP=%d NT=%d SKG=%d U=%d: %s\n",
                 which, s.what, static_cast<long long>(s.N), static_cast<long long>(s.K),
                 static_cast<long long>(s.n_split), kb, static_cast<long long>(m), t.WV, t.SK,
                 t.MB, t.NPW, t.NT, t.SKG, t.U, why);
    ++failures;
  };
  ++checked;
  if (want1 != nullptr && !SameTrellis(t1, *want1)) fail(1, t1, "not the expected M = 1 pick");
  for (int64_t m = 1; m <= 64; ++m) {
    const LinearTuning t = r4dx::model::PickTuning(Layout::kTrellis, s.N, s.K, m, kb);
    ++checked;
    if (!TrellisLaunchable(s.N, s.K, s.n_split, m, kb, t)) fail(m, t, "not launchable");
    ++checked;
    if (!SameTrellis(t, ExpectedTrellis(s.N, s.K, kb, m, tp2))) {
      fail(m, t, m <= 16 ? "not the M = 1 row (or 4.4's fallback)"
                         : "not the prefill row (or 4.4's fallback) for this M");
    }
    if (m <= 16) {
      ++checked;
      if (!SameTrellis(t, t1)) fail(m, t, "differs from the M = 1 pick (row identity)");
    }
  }
}

void CheckTrellis(int& checked, int& failures) {
  CheckTrellisRowsCopy(r4dx::model::trellis::kGemmTuningTable, trellis_rows::kGemmTuningTable,
                       "TP = 1 rows", checked, failures);
  CheckTrellisRowsCopy(r4dx::model::trellis_tp2::kGemmTuningTable,
                       trellis_rows::tp2::kGemmTuningTable, "TP = 2 rows", checked, failures);
  const auto n_split_for = [](int64_t N, int64_t K) {
    // mlp.gate_up's two parts (gate | up): 34816 at TP = 1, 17408 on a TP = 2 rank; a row of any
    // other shape has one part.
    return (N == 34816 || N == 17408) && K == 5120 ? N / 2 : N;
  };
  // Every row of a table at its own rate, on the thread kind that reads it: an M = 1 row is the
  // M = 1 pick, with every M checked as above; a prefill row (M5: M = 32 or 64) is launchable and
  // the pick at its own M.
  const auto check_rows = [&](const auto& table, bool tp2, const char* which) {
    r4dx::model::SetTp2TuningForThisThread(tp2);
    for (const GemmTuningRow& row : table) {
      ++checked;
      if (row.layout != Layout::kTrellis || (row.M != 1 && (row.M <= 16 || row.M > 64)) ||
          (row.rate != 4 && row.rate != 5)) {
        std::fprintf(stderr,
                     "FAIL %s row N=%lld K=%lld M=%lld: not a kTrellis row at KB 4/5 and M = 1 or "
                     "17..64\n",
                     which, static_cast<long long>(row.N), static_cast<long long>(row.K),
                     static_cast<long long>(row.M));
        ++failures;
        continue;
      }
      const TrellisShape s{row.N, row.K, n_split_for(row.N, row.K), "table row"};
      if (row.M == 1) {
        CheckTrellisShape(s, row.rate, &row.tuning, tp2, which, checked, failures);
        continue;
      }
      ++checked;
      const LinearTuning t =
          r4dx::model::PickTuning(Layout::kTrellis, row.N, row.K, row.M, row.rate);
      if (!TrellisLaunchable(s.N, s.K, s.n_split, row.M, row.rate, row.tuning) ||
          !SameTrellis(t, row.tuning)) {
        std::fprintf(stderr,
                     "FAIL %s prefill row N=%lld K=%lld M=%lld KB=%d: not launchable at its M, or "
                     "not the pick there\n",
                     which, static_cast<long long>(row.N), static_cast<long long>(row.K),
                     static_cast<long long>(row.M), row.rate);
        ++failures;
      }
    }
    r4dx::model::SetTp2TuningForThisThread(false);
  };
  check_rows(r4dx::model::trellis::kGemmTuningTable, false, "trellis table");
  check_rows(r4dx::model::trellis_tp2::kGemmTuningTable, true, "trellis TP = 2 table, tp thread");
  // The rate is part of the key: the gate_up shape's KB = 4 and KB = 5 rows differ, and each rate
  // gets its own.
  {
    const GemmTuningRow* r4 = nullptr;
    const GemmTuningRow* r5 = nullptr;
    for (const GemmTuningRow& row : r4dx::model::trellis::kGemmTuningTable) {
      if (row.N == 34816 && row.K == 5120 && row.M == 1) (row.rate == 4 ? r4 : r5) = &row;
    }
    ++checked;
    if (r4 == nullptr || r5 == nullptr || SameTrellis(r4->tuning, r5->tuning)) {
      std::fprintf(stderr, "FAIL trellis table: mlp.gate_up needs distinct KB 4 and KB 5 rows\n");
      ++failures;
    } else if (!SameTrellis(r4dx::model::PickTuning(Layout::kTrellis, 34816, 5120, 1, 4),
                            r4->tuning) ||
               !SameTrellis(r4dx::model::PickTuning(Layout::kTrellis, 34816, 5120, 1, 5),
                            r5->tuning)) {
      std::fprintf(stderr, "FAIL trellis: a rate was served the other rate's gate_up row\n");
      ++failures;
    }
  }
  // Shapes no row covers -- the tiny trellis test container (tests/convert's tiny_k4 / tiny_mix,
  // hidden 256, intermediate 1024) at TP = 1 and TP = 2 -- take the fallback at every M.
  const TrellisShape tiny[] = {{1536, 256, 1536, "tiny gdn.in_proj_qkv"},
                               {1024, 256, 1024, "tiny gdn.in_proj_z"},
                               {256, 1024, 256, "tiny gdn.out_proj / attn.o / mlp.down"},
                               {2048, 256, 1024, "tiny mlp.gate_up"},
                               {2048, 256, 2048, "tiny attn.qg"},
                               {512, 256, 512, "tiny attn.k / attn.v"},
                               {768, 256, 768, "tiny TP2 gdn.in_proj_qkv"},
                               {512, 256, 512, "tiny TP2 gdn.in_proj_z"},
                               {256, 512, 256, "tiny TP2 row-parallel"},
                               {1024, 256, 512, "tiny TP2 mlp.gate_up"},
                               {1024, 256, 1024, "tiny TP2 attn.qg"},
                               {256, 256, 256, "tiny TP2 attn.k / attn.v"}};
  for (const TrellisShape& s : tiny) {
    for (int kb : {4, 5}) {
      const LinearTuning want = TrellisFallback(s.N, s.K, 1);
      CheckTrellisShape(s, kb, &want, false, "trellis fallback", checked, failures);
    }
  }
  // A TP = 2 rank thread (docs/tp.md 2.7): the per-rank shapes of every trellis linear (2.4). The
  // per-rank trellis rows serve them first (the rank's attn.qg, 6144 x 5120, gets its own rows even
  // though gdn.in_proj_z's TP = 1 shape is the same), and every one of them must be what a TP
  // thread is served, at every M.
  r4dx::model::SetTp2TuningForThisThread(true);
  const TrellisShape tp2[] = {
      {5120, 5120, 5120, "TP2 gdn.in_proj_qkv"}, {3072, 5120, 3072, "TP2 gdn.in_proj_z"},
      {5120, 3072, 5120, "TP2 gdn.out_proj / attn.o"}, {6144, 5120, 6144, "TP2 attn.qg"},
      {512, 5120, 512, "TP2 attn.k / attn.v"}, {17408, 5120, 8704, "TP2 mlp.gate_up"},
      {5120, 8704, 5120, "TP2 mlp.down"}};
  for (const TrellisShape& s : tp2) {
    for (int kb : {4, 5}) {
      ++checked;
      if (ExpectedRow(r4dx::model::trellis_tp2::kGemmTuningTable, s.N, s.K, kb, 1) == nullptr) {
        std::fprintf(stderr, "FAIL trellis TP = 2 table: no M = 1 row for %s at KB %d\n", s.what,
                     kb);
        ++failures;
      }
      CheckTrellisShape(s, kb, nullptr, true, "trellis, tp thread", checked, failures);
    }
  }
  r4dx::model::SetTp2TuningForThisThread(false);
  // And a TP = 1 thread is never served a TP = 2 row: the shared shape (6144 x 5120) gets the
  // TP = 1 gdn.in_proj_z rows at every M.
  for (int kb : {4, 5}) {
    const TrellisShape s{6144, 5120, 6144, "TP1 gdn.in_proj_z (a TP2 attn.qg shape)"};
    CheckTrellisShape(s, kb, nullptr, false, "trellis, TP = 1 thread", checked, failures);
  }
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

  int tq_failures = 0, tq_checked = 0;
  CheckTrellis(tq_checked, tq_failures);
  std::printf("test_pick_tuning: %d/%d trellis checks -- launchable at M = 1..64 (TP = 1 rows, "
              "TP = 2 rank shapes, fallback), M = 2..16 the M = 1 pick, M > 16 the prefill row, "
              "the rate in the key\n",
              tq_checked - tq_failures, tq_checked);
  return failures == 0 && sk_failures == 0 && ex_failures == 0 && tq_failures == 0 ? 0 : 1;
}
