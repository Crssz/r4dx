// Qwen35PrefillMoe -- the qwen3_5_moe sparse MoE block for a PREFILL chunk (2..max_rows rows) entirely on the device
// (docs/qwen35moe-plan.md section 8, step a). The host-routed MoeBlockBase it replaces copies the router ids to the host, waits, builds the
// per-expert row lists there and launches two GEMMs per expert (~25,000 launches per 256-row chunk); this class does
//
//   router GEMM (bf16) -> route (softmax, top-k, norm) -> shared expert on all rows (bf16 GEMMs)
//   -> plan (device): the rows * top_k (token, expert) pairs sorted by expert, cut into groups of <= 16 pairs
//   -> gate_up input transform per pair -> grouped trellis GEMM (a group's pairs fill the WMMA tile, the expert's weights are decoded and
//      streamed once per group) -> silu * up fused with the down input transform -> grouped GEMM down -> router-order combine
//
// with no host round trip: 14 launches per layer whatever the chunk. The GEMMs are the M-row kernels of the speculative verify window
// (moe_trellis_grouped.h); only the plan differs (GroupedMoePlanLarge: several groups per expert). Geometry as Qwen35GroupedMoeDecode
// (hidden 3072, expert width 1024, or 512 per rank under --tp 2).
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

#include "qwen35moe_grouped_decode.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"

namespace r4dx::model {

class Qwen35PrefillMoe {
 public:
  // max_rows: the largest chunk Run takes (max_rows * top_k <= 8192). inter: kInter (TP=1) or kTp2Inter (a --tp 2 rank).
  // gu_skg / down_skg: split-group counts of the two GEMMs (1, 2 or 4; 1 needs no workspace). Allocates on the current device.
  Qwen35PrefillMoe(int experts, int top_k, bool norm_topk, int max_rows, int inter = Qwen35GroupedMoeDecode::kInter, int gu_skg = 1,
                   int down_skg = 1);

  // x / y: device bf16 [rows, hidden], disjoint; 2 <= rows <= max_rows. Only enqueues on `stream`. `arena` serves the bf16 GEMMs' scratch.
  void Run(hipStream_t stream, core::Arena& arena, const Qwen35GroupedMoeWeights& w, const Qwen35MoeTrellisPtrs& p, const uint16_t* x,
           uint16_t* y, int64_t rows, const std::function<void(const char*)>* mark = nullptr);

  size_t ScratchBytes() const { return scratch_bytes_; }
  // The router's ids of the last Run, device int32 [rows, top_k] (router order) -- for the capture hook.
  const int32_t* last_ids() const { return ids_.data(); }
  // The plan of the last Run, D2H (tests): kMoePlanInts int32.
  std::vector<int32_t> ReadPlan(hipStream_t stream) const;

 private:
  int experts_, top_k_, max_rows_, inter_, gu_skg_, down_skg_;
  bool norm_topk_;
  size_t scratch_bytes_ = 0;
  core::DeviceBuffer<uint16_t> logits_, a0_gu_, a1_gu_, gu_, a0_down_, yp_, shared_g_, shared_u_, shared_h_, shared_y_, shared_gate_logit_;
  core::DeviceBuffer<int32_t> ids_, plan_, counts_;
  core::DeviceBuffer<float> weights_, ws_gu_, ws_down_;
  core::DeviceBuffer<uint32_t> tickets_gu_, tickets_down_;
};

}  // namespace r4dx::model
