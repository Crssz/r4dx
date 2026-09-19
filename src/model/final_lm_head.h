// r4dx::model::FinalLmHead -- text.final_norm -> lm_head -> fp32 logits (docs/architecture.md's
// forward-pass tail). The lm_head GEMM itself always produces bf16 (every r4d_gemm_*_nt_m64
// variant's C operand is bf16 -- see each kernel's epilogue), so this widens to fp32 as its last
// step via r4dx_model_widen_bf16_to_f32 (kernels/model_kernels.h) to satisfy the task's "-> fp32
// logits on device".
#pragma once

#include <cstdint>

#include "container.h"
#include "model_config.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::model {

class FinalLmHead {
 public:
  FinalLmHead(const ModelConfig& cfg, const core::DeviceBuffer<uint16_t>& final_norm,
              const QuantLinear& lm_head)
      : cfg_(cfg), final_norm_(final_norm), lm_head_(lm_head) {}

  // x: device bf16 [T, hidden] -- the last decoder layer's residual output. logits_out: device
  // fp32 [T, vocab] (vocab == lm_head_.N), caller-allocated (a DeviceBuffer sized for the whole
  // call, not the arena -- logits are the pipeline's output, not scratch).
  void Forward(core::Stream& stream, core::Arena& arena, const uint16_t* x, float* logits_out,
               int64_t T);

 private:
  const ModelConfig& cfg_;
  const core::DeviceBuffer<uint16_t>& final_norm_;
  const QuantLinear& lm_head_;
};

}  // namespace r4dx::model
