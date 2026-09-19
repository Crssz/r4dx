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

class Mlp {
 public:
  Mlp(const ModelConfig& cfg, const core::DeviceBuffer<uint16_t>& post_attention_layernorm,
      const MlpWeights& w)
      : cfg_(cfg), post_attention_layernorm_(post_attention_layernorm), w_(w) {}

  // x: device bf16 [T, hidden] -- current residual stream. x_out: device bf16 [T, hidden], may
  // alias x.
  void Forward(core::Stream& stream, core::Arena& arena, const uint16_t* x, uint16_t* x_out,
               int64_t T);

 private:
  const ModelConfig& cfg_;
  const core::DeviceBuffer<uint16_t>& post_attention_layernorm_;
  const MlpWeights& w_;
};

}  // namespace r4dx::model
