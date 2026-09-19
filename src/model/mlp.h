// r4dx::model::Mlp -- the MLP half of a decoder block (docs/architecture.md's per-layer
// pseudocode, the part after the attention/GDN residual add): post_attention_layernorm ->
// gate_up -> silu_mul -> down -> residual add. Self-contained the same way GdnLayer is: it reads
// the block's current residual stream, applies post_attention_layernorm itself, and returns the
// new (summed) residual stream.
#pragma once

#include <cstdint>

#include "container.h"
#include "model_config.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::model {

class SpanAccumulator;  // profile_span.h -- forward-declared, see gdn_layer.h's identical comment.

class Mlp {
 public:
  Mlp(const ModelConfig& cfg, const core::DeviceBuffer<uint16_t>& post_attention_layernorm,
      const MlpWeights& w)
      : cfg_(cfg), post_attention_layernorm_(post_attention_layernorm), w_(w) {}

  // x: device bf16 [T, hidden] -- current residual stream. x_out: device bf16 [T, hidden], may
  // alias x.
  //
  // R3 fusion (docs/r9700.md P2/R3): `x_normed_in`, when non-null, is this block's ALREADY-NORMED
  // input (post_attention_layernorm applied), produced by the previous sub-block's (GdnLayer's or
  // AttentionLayer's) fused residual+rmsnorm epilogue -- Mlp skips its own initial rmsnorm launch
  // entirely and reads straight from it. When null (the pre-R3 behavior, still exercised by any
  // caller/test that does not wire the fusion), Mlp computes its own rmsnorm from `x` exactly as
  // before.
  // `next_norm_weight`/`x_normed_out`: when next_norm_weight is non-null, the final residual add is
  // replaced by r4dx_residual_rmsnorm_bf16(x, down_out, next_norm_weight, x_out, x_normed_out, ...)
  // -- x_out still gets the plain residual sum (unchanged contract), and x_normed_out additionally
  // gets that sum's rmsnorm under next_norm_weight, for the NEXT layer's sub-block to consume as
  // ITS x_normed_in. next_norm_weight is null for the last loaded layer (there is no next layer;
  // FinalLmHead applies its own final_norm separately, out of this fusion's scope) and for any
  // caller that has not wired the fusion, in which case behavior is the pre-R3 plain residual add.
  // `prof` (Milestone 3 profiling pass, docs/r9700.md R5/Q7): see gdn_layer.h's identical
  // parameter comment.
  void Forward(core::Stream& stream, core::Arena& arena, const uint16_t* x, uint16_t* x_out,
               int64_t T, const uint16_t* x_normed_in = nullptr,
               const uint16_t* next_norm_weight = nullptr, uint16_t* x_normed_out = nullptr,
               SpanAccumulator* prof = nullptr);

 private:
  const ModelConfig& cfg_;
  const core::DeviceBuffer<uint16_t>& post_attention_layernorm_;
  const MlpWeights& w_;
};

}  // namespace r4dx::model
