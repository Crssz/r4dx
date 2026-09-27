// tests/kernels/trellis_tuning_rows.hpp -- src/model/gemm_tuning_table_trellis.inc (the trellis
// rows linear.cpp's PickTuning reads, docs/trellis-kernel.md 5.3), readable from a kernel test that
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

enum class Layout { kBf16, kMxfp4, kW4a16, kW4a8, kTrellis };

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

}  // namespace trellis_rows
