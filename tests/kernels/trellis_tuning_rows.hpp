// tests/kernels/trellis_tuning_rows.hpp -- src/model/gemm_tuning_table_trellis.inc and its TP = 2
// sibling gemm_tuning_table_trellis_tp2.inc (the trellis rows linear.cpp's PickTuning reads,
// docs/trellis-kernel.md 5.3; the per-rank ones in trellis_rows::tp2), readable from a kernel test that
// does not link r4dx_model_linear: this header declares the shapes the rows are written for --
// quant_linear.h's Layout (kTrellis appended last, 5.1) and linear.h's LinearTuning / GemmTuningRow
// (SKG, U, rate) -- in its own namespace and includes the table there, so test_trellis_gemm (row
// identity for every row) and tool_trellis_gemm_bench (replaying it) read the very file production
// runs. The declarations must stay field-for-field linear.h's: tests/model/test_pick_tuning.cpp,
// which sees both, static_asserts the layouts and compares the rows field by field, since a field
// reordered in only one copy would still compile.
#pragma once

#include <cstdint>

namespace trellis_rows {

// Values pinned to quant_linear.h's (its layout numbering has gaps at 1 and 3).
enum class Layout { kBf16 = 0, kW4a16 = 2, kTrellis = 4 };

struct LinearTuning {
  int WV, SK, MB, NPW, NT;
  int SKG = 1;
  int U = 2;
};

struct GemmTuningRow {
  Layout layout;
  int64_t N, K, M;
  LinearTuning tuning;
  int group = 0;
  int rate = 0;
};

#include "../../src/model/gemm_tuning_table_trellis.inc"

// The TP = 2 per-rank rows (milestone M5), same types.
namespace tp2 {
#include "../../src/model/gemm_tuning_table_trellis_tp2.inc"
}  // namespace tp2

}  // namespace trellis_rows
