// MoeBlockBase -- the qwen4_exp sparse MoE block (HF Qwen4ExpTextSparseMoeBlock) minus the per-expert GEMMs
// (docs/qwen4exp-plan.md 4.7, 4.11 item 5, host-routed):
//
//   router GEMM (bf16, ApplyBf16Linear) -> q4x_moe_route_bf16 -> ids to the host (ONE device -> host sync per call)
//   -> BuildMoePlan -> q4x_moe_gather_rows_bf16 (all routed rows, expert-sorted, one launch)
//   -> per expert with rows: gate_up GEMM (callback)  -> q4x_moe_silu_mul_bf16 (one launch over all pairs)
//   -> per expert with rows: down GEMM (callback)
//   -> shared expert (gate / up / down bf16 linears + the N = 1 shared_gate dot, ApplyBf16Linear)
//   -> q4x_moe_combine_bf16: weighted sum of the expert outputs + sigmoid-gated shared expert, one rounding to bf16.
//
// The per-expert GEMMs are issued by the caller through an ExpertLinearFn, so the same code runs bf16 test weights
// (ApplyLinear on a dense [E, N, K] tensor) now and per-expert trellis QuantLinears later. NOTE the callback is the
// ONE LINEAR (gate_up or down), not the whole expert MLP: the silu * up between them is a kernel of this class.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>

#include "quant_linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/event.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/model/qwen4exp/moe_plan.h"

namespace r4dx::model::qwen4exp {

struct MoeConfig {
  int hidden = 2560;
  int experts = 512;
  int top_k = 10;
  int inter = 640;          // moe_intermediate_size (each of gate / up; the fused gate_up GEMM is N = 2 * inter)
  int shared_inter = 640;   // shared_expert_intermediate_size
  bool norm_topk = true;    // config norm_topk_prob
};

// Which linear of an expert the callback runs.
enum class ExpertLinear {
  kGateUp,  // y[n, 2 * inter] = x[n, hidden] @ W_gate_up[2 * inter, hidden]^T, gate = columns [0, inter), up = [inter, 2 * inter)
  kDown,    // y[n, hidden]    = x[n, inter]  @ W_down[hidden, inter]^T
};

// Computes y[rows, N] = x[rows, K] @ W(expert, which)^T for ONE expert: x, y are device bf16, contiguous, disjoint,
// valid on the stream passed to Forward. `arena` is the arena passed to Forward (for ApplyLinear's scratch). Called
// only for experts with rows > 0, gate_up calls first (ascending expert id), then the down calls.
using ExpertLinearFn = std::function<void(int expert, ExpertLinear which, const uint16_t* x, int64_t rows,
                                          uint16_t* y, core::Arena& arena)>;

struct MoeWeights {
  const QuantLinear* router = nullptr;       // bf16 [experts, hidden]
  const QuantLinear* shared_gate_proj = nullptr;  // [shared_inter, hidden]
  const QuantLinear* shared_up_proj = nullptr;    // [shared_inter, hidden]
  const QuantLinear* shared_down_proj = nullptr;  // [hidden, shared_inter]
  const QuantLinear* shared_gate = nullptr;  // bf16 [1, hidden] (shared_expert_gate)
};

class MoeBlockBase {
 public:
  // max_rows: the largest `rows` Forward will be called with (sizes the pinned id staging buffer).
  MoeBlockBase(const MoeConfig& cfg, int64_t max_rows);

  // Arena bytes one Forward call allocates for `rows` rows (the callback's own scratch comes on top).
  size_t ArenaBytes(int64_t rows) const;

  // x: device bf16 [rows, hidden]; y: device bf16 [rows, hidden] (disjoint from x). Blocks the calling thread once
  // (the ids' device -> host copy). Throws std::runtime_error on a shape or weight mismatch.
  // mark (profiling, R4DX_Q35_PROFILE=3): called with a phase name right after that phase's work is enqueued.
  void Forward(hipStream_t stream, core::Arena& arena, const MoeWeights& w, const uint16_t* x, uint16_t* y,
               int64_t rows, const ExpertLinearFn& run_expert, const std::function<void(const char*)>* mark = nullptr);

  // The plan of the most recent Forward (expert row counts etc.), for tests and diagnostics.
  const MoePlan& last_plan() const { return plan_; }
  // The router's ids of the most recent Forward, host copy [rows, k].
  const int32_t* last_ids() const { return ids_host_.data(); }

  const MoeConfig& config() const { return cfg_; }

 private:
  MoeConfig cfg_;
  int64_t max_rows_;
  core::PinnedBuffer<int32_t> ids_host_;
  core::Event ids_ready_;
  MoePlan plan_;
};

}  // namespace r4dx::model::qwen4exp
