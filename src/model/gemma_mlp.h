// r4dx::model::GemmaMlp -- the GeGLU MLP of one Gemma 4 decoder layer (docs/gemma4-plan.md 3.5, task M1-19):
//   down( gelu_tanh(gate(x)) * up(x) ),  intermediate 15360, no bias.
// The sandwich norms and the residual add are NOT here (the layer fuses post_feedforward_layernorm + residual
// + layer_scalar + the next pre-norm in one kernel); Qwen's Mlp (silu, post_attention_layernorm inside, residual
// inside) is left untouched.
//
//   x_normed = pre_feedforward_layernorm(residual)   (the caller / the previous fused kernel produced it)
//   gate_up  = x_normed @ [gate; up]^T               ([T, 2 * intermediate], gate rows first)
//   h        = bf16(gelu_tanh(gate)) * up            (r4dx_gelu_tanh_mul_*: the activation rounds to bf16 first)
//   [rotated container: h <- h Hb, 512-wide blocks, signs rotation.had_down_signs]
//   down_out = h @ down^T                            ([T, hidden], raw: post-norm + residual are the caller's)
#pragma once

#include <cstdint>

#include "container.h"  // MlpWeights
#include "gemma_config.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_comm.hpp"

namespace r4dx::model {

class SpanAccumulator;  // profile_span.h

class GemmaMlp {
 public:
  // `down_had_signs`: non-owning device fp32 [intermediate_size] (+-1), a rotated container's
  // rotation.had_down_signs; nullptr (every unrotated container) is the plain path.
  // `comm`: non-null under tensor parallelism (docs/gemma4-plan.md M1b-1): mlp.down is row-parallel, so its bf16 output
  // is a per-rank partial sum, all-reduced inside Forward before the caller's fp32 post-norm + residual add. `cfg` is the
  // RANK-LOCAL config (intermediate_size / world); nullptr (TP=1) is the unchanged path.
  GemmaMlp(const GemmaConfig& cfg, const MlpWeights& w, const float* down_had_signs = nullptr,
           core::TpComm* comm = nullptr)
      : cfg_(cfg), w_(w), down_had_signs_(down_had_signs), comm_(comm) {}

  // x_normed: device bf16 [T, hidden]. down_out: device bf16 [T, hidden] (disjoint from x_normed).
  void Forward(core::Stream& stream, core::Arena& arena, const uint16_t* x_normed, uint16_t* down_out, int64_t T,
               SpanAccumulator* prof = nullptr);

 private:
  const GemmaConfig& cfg_;
  const MlpWeights& w_;
  const float* down_had_signs_;
  core::TpComm* comm_;
};

}  // namespace r4dx::model
