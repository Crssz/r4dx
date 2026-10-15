// r4dx-convert's LFM2 branch (docs/lfm2-plan.md 4, task L1-2): which HF tensors of an lfm2_moe checkpoint
// (LiquidAI/LFM2.5-24B-A2B) become which container tensors, with the shape cross-checks and the coverage audit.
//
// Same shell as gemma_layout.hpp: the Qwen path of RunConvert (main.cpp) is not refactored; AddTextStack drives a
// `Kit` -- std::function handles onto RunConvert's own closures (add_bf16, add_linear, ...) -- so the quantization,
// --keep-bf16 / --ldlq machinery, the plan/emit job lists and the writer are the ones Qwen uses, and
// tests/convert/test_lfm2_layout.cpp can drive the layout against a recording Kit without a single weight byte.
//
// HF names (VERIFIED against the downloaded checkpoint's 7610 tensors, docs/lfm2-plan.md 2.1; config.json is FLAT,
// there is no text_config; NO lm_head tensor: the head is tied):
//   model.embed_tokens.weight                                bf16 [vocab, hidden]
//   model.embedding_norm.weight                              bf16 [hidden]   (the FINAL norm, despite the name)
//   model.layers.{i}.operator_norm.weight / ffn_norm.weight  bf16 [hidden]
//   conv layers:    conv.in_proj.weight [3H, H], conv.conv.weight [H, 1, 3], conv.out_proj.weight [H, H]
//   attn layers:    self_attn.{q,k,v,out}_proj.weight, self_attn.{q,k}_layernorm.weight [head_dim]
//   dense layers (i < num_dense_layers): feed_forward.{w1,w2,w3}.weight   (w1 = gate, w3 = up, w2 = down)
//   MoE layers:     feed_forward.gate.weight [E, H] bf16, feed_forward.expert_bias [E] F32,
//                   feed_forward.experts.{e}.{w1,w2,w3}.weight   (PER-EXPERT; the fused 3-D `experts.gate_up_proj`
//                   style of transformers 5 is refused by name, never converted wrongly)
//
// Container tensors ({L} = the layout suffix the converter appends: .bf16.w, .w4a16.wq/.wsz, .trellis.*):
//   text.embed_tokens                       bf16 [vocab, hidden], no scale
//   text.final_norm                         bf16 [hidden], plain weight
//   lm_head.{L}                             the SAME embed_tokens tensor (tied), written untied like Gemma's
//   text.layers.{i}.operator_norm / ffn_norm                          bf16 [hidden]
//   text.layers.{i}.attn.{q,k,v,o}.{L}      self_attn.{q,k,v,out}_proj
//   text.layers.{i}.attn.{q_norm,k_norm}    self_attn.{q,k}_layernorm            bf16 [head_dim]
//   text.layers.{i}.conv.in_proj.{L}        conv.in_proj (N = 3H, chunk order B, C, x kept)
//   text.layers.{i}.conv.weight             conv.conv.weight [H,1,3] SQUEEZED to bf16 [H, 3] (same bytes, never quantized)
//   text.layers.{i}.conv.out_proj.{L}
//   text.layers.{i}.mlp.gate_up.{L}, mlp.down.{L}                     (i < num_dense_layers) gate rows first
//   text.layers.{i}.moe.router              feed_forward.gate.weight             bf16 [E, H] (passthrough, never quantized)
//   text.layers.{i}.moe.expert_bias         feed_forward.expert_bias             fp32 [E, 4] (the F32 buffer, fp32 in the container)
//   text.layers.{i}.moe.experts.{e}.gate_up.{L}  experts.{e}.w1 rows then w3 rows (N = 2I)
//   text.layers.{i}.moe.experts.{e}.down.{L}     experts.{e}.w2
// Norms are stored raw: the runtime uses a plain RMSNorm (x * w, not 1 + w).
#pragma once

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace r4dx_convert::lfm2 {

constexpr const char* kModelArch = "lfm2_moe";

// What the converter needs of config.json. Shapes of the tensors come from the safetensors headers; this is only
// what they are cross-checked against.
struct TextShape {
  int64_t hidden = 0;
  int64_t layers_total = 0;
  int64_t heads = 0, kv_heads = 0, head_dim = 0;
  int64_t dense_intermediate = 0;  // `intermediate_size`: the dense FFN of the first num_dense_layers layers
  int64_t moe_intermediate = 0;
  int64_t num_dense_layers = 0;
  int64_t num_experts = 0;
  int64_t conv_kernel = 0;  // conv_L_cache
  int64_t vocab = 0;
  std::vector<std::string> layer_types;  // "conv" | "full_attention"

  bool IsAttn(int64_t i) const { return layer_types.at(static_cast<size_t>(i)) == "full_attention"; }
  bool IsDense(int64_t i) const { return i < num_dense_layers; }
};

// Strict: a missing key, a layer type other than conv/full_attention, an untied head (the head tensor would go
// unconsumed), conv_bias true, use_expert_bias false, or model_type other than lfm2_moe throws std::runtime_error.
// `head_dim` is hidden / heads when absent. The layer PATTERN is not asserted here (the runtime's Lfm2Config does;
// the tiny test checkpoint is conv, conv, attention, conv).
TextShape ParseConfig(const nlohmann::json& cfg);

struct Kit {
  // The checkpoint.
  std::function<std::vector<std::string>()> all_names;
  std::function<bool(const std::string&)> has;
  std::function<std::vector<int64_t>(const std::string&)> shape;
  // One plan job + one emit job each (RunConvert's add_* closures).
  std::function<void(const std::string& hf, const std::string& name)> add_bf16;
  // bf16 passthrough re-labelled with `shape` (no trailing element-width dim; same element count): conv.conv.weight.
  std::function<void(const std::string& hf, const std::string& name, const std::vector<int64_t>& shape)> add_bf16_shaped;
  // bf16 or F32 source read as float, stored fp32 with a trailing 4: expert_bias.
  std::function<void(const std::string& hf, const std::string& name)> add_fp32_widen;
  // A quantizable linear; several hf names are fused on the output axis in order. `head`: the tied lm_head (the
  // --lm-head layout set applies instead of --layouts).
  std::function<void(const std::vector<std::string>& hf, const std::string& base, bool head)> add_linear;
};

// Registers every tensor of the first `layers` layers, then the embedding table, the final norm and the tied lm_head.
// Throws before anything is planned when a tensor is missing, its shape disagrees with the config, or a fused
// experts.gate_up_proj / experts.down_proj tensor is present. Returns every HF tensor name it consumed.
std::set<std::string> AddTextStack(Kit& kit, const TextShape& s, int64_t layers);

// The coverage audit: checkpoint tensors AddTextStack did not consume and that are not allow-listed. Only layers >=
// `layers` (--layers) are allow-listed; anything else -- a mis-guessed name, a tensor the checkpoint grew, an
// `lm_head.weight` of an untied checkpoint -- must fail the run before a long conversion.
std::vector<std::string> UnconsumedTensors(const std::vector<std::string>& all_names,
                                           const std::set<std::string>& consumed, int64_t layers);

}  // namespace r4dx_convert::lfm2
