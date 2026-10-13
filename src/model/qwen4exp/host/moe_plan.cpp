#include "r4dx/model/qwen4exp/moe_plan.h"

#include <algorithm>
#include <numeric>
#include <stdexcept>
#include <string>

namespace r4dx::model::qwen4exp {

MoePlan BuildMoePlan(const int32_t* ids, int rows, int k, int experts) {
  if (rows < 0 || k < 1 || experts < 1) {
    throw std::runtime_error("qwen4exp::BuildMoePlan: bad shape rows=" + std::to_string(rows) +
                             " k=" + std::to_string(k) + " experts=" + std::to_string(experts));
  }
  MoePlan plan;
  plan.rows = rows;
  plan.k = k;
  plan.experts = experts;
  plan.pairs = static_cast<int64_t>(rows) * k;
  if (plan.pairs > INT32_MAX) throw std::runtime_error("qwen4exp::BuildMoePlan: too many pairs");
  plan.expert_offset.assign(static_cast<size_t>(experts) + 1, 0);
  plan.pair_row.assign(static_cast<size_t>(plan.pairs), 0);
  plan.row_pairs.assign(static_cast<size_t>(plan.pairs), 0);
  plan.row_w_idx.assign(static_cast<size_t>(plan.pairs), 0);

  for (int64_t i = 0; i < plan.pairs; ++i) {
    const int32_t e = ids[i];
    if (e < 0 || e >= experts) {
      throw std::runtime_error("qwen4exp::BuildMoePlan: row " + std::to_string(i / k) + " routes to expert " +
                               std::to_string(e) + " outside [0, " + std::to_string(experts) + ")");
    }
    ++plan.expert_offset[static_cast<size_t>(e) + 1];
  }
  std::partial_sum(plan.expert_offset.begin(), plan.expert_offset.end(), plan.expert_offset.begin());

  std::vector<int32_t> cursor(plan.expert_offset.begin(), plan.expert_offset.end() - 1);
  std::vector<int> slots(static_cast<size_t>(k));
  for (int r = 0; r < rows; ++r) {
    const int32_t* row_ids = ids + static_cast<int64_t>(r) * k;
    std::iota(slots.begin(), slots.end(), 0);
    std::sort(slots.begin(), slots.end(), [&](int a, int b) { return row_ids[a] < row_ids[b]; });
    for (int j = 0; j < k; ++j) {
      const int slot = slots[static_cast<size_t>(j)];
      const int32_t e = row_ids[slot];
      if (j > 0 && row_ids[slots[static_cast<size_t>(j) - 1]] == e) {
        throw std::runtime_error("qwen4exp::BuildMoePlan: row " + std::to_string(r) + " selects expert " +
                                 std::to_string(e) + " twice");
      }
      const int32_t p = cursor[static_cast<size_t>(e)]++;
      plan.pair_row[static_cast<size_t>(p)] = r;
      plan.row_pairs[static_cast<size_t>(r) * k + j] = p;
      plan.row_w_idx[static_cast<size_t>(r) * k + j] = r * k + slot;
    }
  }
  return plan;
}

}  // namespace r4dx::model::qwen4exp
