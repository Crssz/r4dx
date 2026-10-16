// r4dx::model -- the LFM2 (lfm2_moe) sublayers other than attention (docs/lfm2-plan.md 3.2, 3.3, 3.5; task L1-8):
//
//   Lfm2ConvLayer  the gated short convolution mixer:  bcx = in_proj(x) [T, 3C] (B | C | x);
//                  y = r4dx_lfm2_shortconv_bf16(bcx, state, weight) (h = B*x, causal depthwise conv k=3, C*conv,
//                  the [2, C] state updated in place);  out = out_proj(y)  [TP: row-parallel, all-reduced]
//   Lfm2DenseMlp   layers 0-1:  down( silu(gate(x)) * up(x) )  (mlp.gate_up gate rows first)  [TP: all-reduced]
//   Lfm2MoeFfn     layers 2-39: fp32 sigmoid router (r4dx_lfm2_router_f32, replicated), then the routed experts:
//                    decode (T == 1, kAuto):  one host readback of the 4 ids, then per slot gate_up -> silu_mul -> down
//                                             into a [1, 4, H] slot buffer
//                    prefill (T > 1, kAuto):  counting sort of the T x 4 pairs by expert (deterministic), one
//                                             readback of the counts / offsets, a gather of the sorted rows, per
//                                             non-empty expert one gate_up / silu_mul / down over its rows
//                  every expert GEMM is ApplyLinear on that expert's own QuantLinear, so the bf16 and the trellis
//                  layouts (docs/lfm2-plan.md L3-8: per-expert trellis gate_up [2 parts] / down, M = 1 at decode,
//                  M = the expert's row count <= 256 in prefill, sliced into <= 64-row launches) take the same path;
//                  the arena is rewound after each expert, so the GEMM scratch peak is one expert's;
//                  then r4dx_lfm2_moe_combine_bf16 (fp32 weighted sum over the slots in slot order, one rounding)
//                  [TP: every expert is split (gate|up halves, down K), so the combined rows are partial sums,
//                  all-reduced once].
//
// None of these owns device memory: every scratch comes from the caller's arena, the host readback goes through the
// caller's pinned Lfm2MoeHost (allocated at load: no allocation inside a TP collective). The residual add and the
// norms are the caller's (Lfm2Model::RunLayer).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

#include "lfm2_config.h"
#include "lfm2_container.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/tp_comm.hpp"

namespace r4dx::model {

class SpanAccumulator;  // profile_span.h

class Lfm2ConvLayer {
 public:
  // `cfg`: the RANK-LOCAL config (conv_dim = the rank's channels). `comm` non-null under TP.
  Lfm2ConvLayer(const Lfm2Config& cfg, const Lfm2ConvWeights& w, core::TpComm* comm = nullptr)
      : cfg_(cfg), w_(w), comm_(comm) {}
  // x_normed [T, hidden] bf16; out [T, hidden] bf16; state: device bf16 [2, conv_dim] of THIS layer, updated in place.
  void Forward(hipStream_t stream, core::Arena& arena, const uint16_t* x_normed, uint16_t* out, uint16_t* state,
               int64_t T, SpanAccumulator* prof = nullptr) const;

 private:
  const Lfm2Config& cfg_;
  const Lfm2ConvWeights& w_;
  core::TpComm* comm_;
};

class Lfm2DenseMlp {
 public:
  Lfm2DenseMlp(const Lfm2Config& cfg, const MlpWeights& w, core::TpComm* comm = nullptr) : cfg_(cfg), w_(w), comm_(comm) {}
  void Forward(hipStream_t stream, core::Arena& arena, const uint16_t* x_normed, uint16_t* out, int64_t T,
               SpanAccumulator* prof = nullptr) const;

 private:
  const Lfm2Config& cfg_;
  const MlpWeights& w_;
  core::TpComm* comm_;
};

// Host side of the MoE readbacks (pinned, sized at load for the largest chunk).
struct Lfm2MoeHost {
  core::PinnedBuffer<int32_t> ids;      // [rows * top_k]   (the decode path's selected expert ids)
  core::PinnedBuffer<int32_t> offsets;  // [n_experts + 1]  (the sorted path's exclusive prefix sum)
  void Allocate(int64_t rows, int64_t top_k, int64_t n_experts) {
    ids = core::PinnedBuffer<int32_t>(static_cast<size_t>(rows * top_k));
    offsets = core::PinnedBuffer<int32_t>(static_cast<size_t>(n_experts + 1));
  }
};

enum class Lfm2MoePath {
  kAuto,      // T == 1: per-slot (decode); T > 1: sorted
  kPerSlot,   // every row's top_k experts one by one, M = 1 GEMMs (the decode path, also for T > 1: test / A-B only)
  kSorted,    // the counting-sort path, also for T == 1
};

// What a debug caller wants back from Lfm2MoeFfn::Forward (host copies, taken after a stream synchronize).
struct Lfm2MoeDebug {
  std::vector<int32_t> ids;  // [T, top_k]
  std::vector<float> w;      // [T, top_k]
};

class Lfm2MoeFfn {
 public:
  Lfm2MoeFfn(const Lfm2Config& cfg, const Lfm2MoeWeights& w, Lfm2MoeHost* host, core::TpComm* comm = nullptr)
      : cfg_(cfg), w_(w), host_(host), comm_(comm) {}
  // x_normed [T, hidden] bf16; out [T, hidden] bf16. Synchronizes the stream once (the readback). `forced`: test /
  // warm-up only -- host [T, top_k] ids and weights used INSTEAD of the router's (the router still runs).
  void Forward(hipStream_t stream, core::Arena& arena, const uint16_t* x_normed, uint16_t* out, int64_t T,
               SpanAccumulator* prof = nullptr, Lfm2MoePath path = Lfm2MoePath::kAuto, Lfm2MoeDebug* dbg = nullptr,
               const std::vector<int32_t>* forced_ids = nullptr, const std::vector<float>* forced_w = nullptr) const;

 private:
  const Lfm2Config& cfg_;
  const Lfm2MoeWeights& w_;
  Lfm2MoeHost* host_;
  core::TpComm* comm_;
};

}  // namespace r4dx::model
