// Host-side routing plan of one MoE layer (docs/qwen4exp-plan.md 4.11 item 5: host-routed MoE). Built from the router's
// top-k ids after the once-per-layer device -> host sync; describes the experts' rows as one expert-sorted list of
// (row, expert) PAIRS so that
//   * the gather (moe_kernels.h q4x_moe_gather_rows_bf16) puts expert e's input rows at
//     [expert_offset[e], expert_offset[e + 1]) of one contiguous [P, hidden] block,
//   * the per-expert GEMMs run on those slices in place, and
//   * the combine kernel (q4x_moe_combine_bf16) adds each row's k contributions back in ascending expert order.
#pragma once

#include <cstdint>
#include <vector>

namespace r4dx::model::qwen4exp {

struct MoePlan {
  int rows = 0;
  int k = 0;
  int experts = 0;
  int64_t pairs = 0;                   // rows * k
  std::vector<int32_t> expert_offset;  // [experts + 1]; expert e's pairs are [expert_offset[e], expert_offset[e+1])
  std::vector<int32_t> pair_row;       // [pairs]: source row of pair p (rows ascending within an expert)
  std::vector<int32_t> row_pairs;      // [rows * k]: pair index of row r's j-th selected expert, experts ascending
  std::vector<int32_t> row_w_idx;      // [rows * k]: index r * k + slot of that pair's routing weight

  int32_t Count(int expert) const { return expert_offset[expert + 1] - expert_offset[expert]; }
};

// ids: int32 [rows, k] (the router output, any slot order). Throws std::runtime_error naming the row for an expert id
// outside [0, experts) or a duplicate expert within a row (the router never produces either, so one means the device
// output is corrupt).
MoePlan BuildMoePlan(const int32_t* ids, int rows, int k, int experts);

}  // namespace r4dx::model::qwen4exp
