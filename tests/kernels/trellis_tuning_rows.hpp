// tests/kernels/trellis_tuning_rows.hpp -- tests/kernels/gemm_tuning_table_trellis.inc, readable
// before milestone M4 moves it into src/model and wires it into linear.cpp: the trellis rows are
// written for the structs as docs/trellis-kernel.md 5.3 extends them (LinearTuning += SKG, U;
// GemmTuningRow += rate; Layout += kTrellis, appended last per 5.1), which src/model does not have
// yet. This header declares those shapes -- quant_linear.h's Layout with kTrellis appended, and
// linear.h's two structs with the new fields -- in its own namespace and includes the table there,
// so test_trellis_gemm (row identity for every row) and tool_trellis_gemm_bench (replaying it) read
// the very file M4 will include, and a row that stops compiling against the 5.3 layout fails here
// first.
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

#include "gemm_tuning_table_trellis.inc"

}  // namespace trellis_rows
