// r4dx-convert --gguf <hy_v3 .gguf> --out <container.r4dx>
//
// Converts a GGUF v3 hy_v3 (Hy-MT3-30B-A3B) checkpoint directly into an .r4dx container.
// Reconstructs __metadata__.model_config from the GGUF hy_v3.* keys and emits:
//   text.embed_tokens                           bf16 [vocab, hidden]
//   text.final_norm                             bf16 [hidden]
//   lm_head.{L}                                 [vocab, hidden]
//   text.layers.{i}.input_norm / post_attn_norm bf16 [hidden]
//   text.layers.{i}.attn.{q,k,v,o}.{L}
//   text.layers.{i}.attn.{q_norm,k_norm}        bf16 [head_dim]
//   layer 0 (dense): text.layers.0.mlp.gate_up.{L}, mlp.down.{L}
//   layers 1..N-1 (MoE):
//     text.layers.{i}.moe.router                bf16 [E, hidden]
//     text.layers.{i}.moe.expert_bias           fp32 [E, 4]
//     text.layers.{i}.moe.shared_gate_up.{L}    [2*I_moe, hidden]
//     text.layers.{i}.moe.shared_down.{L}       [hidden, I_moe]
//     text.layers.{i}.moe.experts.{e}.gate_up.{L} [2*I_moe, hidden]
//     text.layers.{i}.moe.experts.{e}.down.{L}    [hidden, I_moe]
#pragma once

#include <cstdint>
#include <string>

#include "nlohmann/json.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/quant_search.hpp"

namespace r4dx_convert {

struct HyV3GgufOptions {
  std::string gguf_path;
  std::string out_path;
  LayoutSet body_layouts;
  LayoutSet lm_head_layouts;
  QuantMode quant_mode = QuantMode::kSearch;
  int64_t layers = -1;  // -1 = all layers
  int threads = 0;
  nlohmann::json quant_metadata;
};

void ConvertHyV3Gguf(const HyV3GgufOptions& opts);

}  // namespace r4dx_convert
