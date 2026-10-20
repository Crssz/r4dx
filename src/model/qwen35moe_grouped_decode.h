// Qwen35GroupedMoeDecode -- the qwen3_5_moe sparse MoE block for ONE decode token (M = 1) entirely on the device, no host
// synchronization (docs/qwen35moe-plan.md section 7; the qwen4_exp counterpart is RunMoeGroupedDecode in qwen4exp_stage.cpp):
//
//   router GEMM (bf16) -> route (softmax, top-k, norm) -> shared expert (bf16 GEMVs, sigmoid gate logit) -> sort the k ids
//   -> input transform (gate_up) -> grouped trellis GEMV gate_up -> silu * up fused with the down input transform
//   -> grouped trellis GEMV down -> combine (k experts in ascending id order + sigmoid-gated shared expert, one rounding).
//
// Geometry is fixed by the instantiated kernels (moe_trellis_grouped.h): hidden 3072, routed and shared expert intermediate
// 1024, top_k <= 16, any expert count <= 1024, trellis bits 3 or 4 for both projections. Supported() says whether a layer
// qualifies; anything else keeps the host-routed MoeBlockBase.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>

#include "r4dx/core/arena.hpp"
#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"

namespace r4dx::model {

// One layer's stacked trellis experts (the container's ExpertClassStorage buffers: all experts back to back).
struct Qwen35MoeTrellisPtrs {
  const uint32_t* gate_up_w = nullptr;   // [experts, 2048 * 3072 * bits / 32] words
  const float* gate_up_suh = nullptr;    // [experts, 2 * 3072]
  const float* gate_up_svh = nullptr;    // [experts, 2048]
  int gate_up_bits = 0;
  const uint32_t* down_w = nullptr;      // [experts, 3072 * 1024 * bits / 32]
  const float* down_suh = nullptr;       // [experts, 1024]
  const float* down_svh = nullptr;       // [experts, 3072]
  int down_bits = 0;
};

// The router / shared-expert linears of the layer (bf16 [experts, hidden], [1024, hidden] x2, [hidden, 1024], [1, hidden]).
// (Not qwen4exp::MoeWeights: that name exists twice in namespace qwen4exp -- moe_block.h and qwen4exp_weights.h -- and the stage side
// must not see the container side's.)
struct Qwen35GroupedMoeWeights {
  const QuantLinear* router = nullptr;
  const QuantLinear* shared_gate_proj = nullptr;
  const QuantLinear* shared_up_proj = nullptr;
  const QuantLinear* shared_down_proj = nullptr;
  const QuantLinear* shared_gate = nullptr;
};

class Qwen35GroupedMoeDecode {
 public:
  static constexpr int kHidden = 3072;
  static constexpr int kInter = 1024;  // routed and shared

  // True when the geometry and the pointers match what the kernels are instantiated for.
  static bool Supported(int64_t hidden, int64_t inter, int64_t shared_inter, int64_t experts, int64_t top_k,
                        const Qwen35MoeTrellisPtrs& p);

  // Allocates the scratch on the calling thread's current device and zeroes the split-group tickets (synchronizing).
  Qwen35GroupedMoeDecode(int experts, int top_k, bool norm_topk);

  // x: device bf16 [1, hidden] (the post-attention-normed residual); y: device bf16 [1, hidden], disjoint from x. Only enqueues
  // on `stream`. `arena` serves the bf16 GEMVs' scratch (the caller resets it). Throws std::runtime_error on a launch error.
  // mark (benchmarks): called with a phase name right after each phase's launches are enqueued (the caller may record an event).
  void Run(hipStream_t stream, core::Arena& arena, const Qwen35GroupedMoeWeights& w, const Qwen35MoeTrellisPtrs& p,
           const uint16_t* x, uint16_t* y, const std::function<void(const char*)>* mark = nullptr);

  // The most recent selection in router order (descending score) -- for tests: copies top_k ids / weights to the host.
  void ReadSelection(hipStream_t stream, int32_t* ids, float* weights) const;

 private:
  void RunShared(hipStream_t s, core::Arena& arena, const Qwen35GroupedMoeWeights& w, const uint16_t* x);
  int experts_, top_k_;
  bool norm_topk_;
  core::DeviceBuffer<uint16_t> logits_;
  core::DeviceBuffer<int32_t> ids_, ids_sorted_, rank_;
  core::DeviceBuffer<float> weights_;
  core::DeviceBuffer<uint16_t> a0_gu_, a1_gu_, gu_, a0_down_, yp_;
  core::DeviceBuffer<float> ws_gu_, ws_down_;
  core::DeviceBuffer<uint16_t> shared_g_, shared_u_, shared_h_, shared_y_, shared_gate_logit_;
  core::DeviceBuffer<uint32_t> tickets_gu_, tickets_down_;
};

}  // namespace r4dx::model
