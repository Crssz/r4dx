// r4dx-convert's Llama branch (docs/nanbeige-plan.md 4, task N1-4): which HF tensors of a dense `llama` checkpoint
// (LlamaForCausalLM; first user Nanbeige/Nanbeige4.3-33B) become which container tensors, with the shape cross-checks
// and the coverage audit.
//
// Same shell as lfm2_layout.hpp: the Qwen path of RunConvert (main.cpp) is not refactored; AddTextStack drives a `Kit`
// -- std::function handles onto RunConvert's own closures (add_bf16, add_linear) -- so the quantization, --keep-bf16 /
// --ldlq machinery, the plan/emit job lists and the writer are the ones Qwen uses, and
// tests/convert/test_llama_layout.cpp can drive the layout against a recording Kit without a single weight byte.
//
// HF names (VERIFIED against the Nanbeige4.3-33B index, 471 tensors, docs/nanbeige-plan.md 11; config.json is FLAT):
//   model.embed_tokens.weight                                         bf16 [vocab, hidden]
//   model.norm.weight                                                 bf16 [hidden]
//   lm_head.weight                                                    bf16 [vocab, hidden]   (UNTIED)
//   model.layers.{i}.input_layernorm.weight / post_attention_layernorm.weight             bf16 [hidden]
//   model.layers.{i}.self_attn.{q,k,v,o}_proj.weight                  no bias
//   model.layers.{i}.mlp.{gate,up,down}_proj.weight                   no bias
//
// Container tensors ({L} = the layout suffix the converter appends: .bf16.w, .w4a16.wq/.wsz, .w8a16.wq/.ws, .trellis.*):
//   text.embed_tokens                       bf16 [vocab, hidden], no scale
//   text.final_norm                         bf16 [hidden], plain weight
//   lm_head.{L}                             lm_head.weight (the --lm-head layout set)
//   text.layers.{i}.input_norm / post_attn_norm                        bf16 [hidden], plain weight
//   text.layers.{i}.attn.{q,k,v,o}.{L}      self_attn.{q,k,v,o}_proj
//   text.layers.{i}.mlp.gate_up.{L}         gate_proj rows then up_proj rows (N = 2I)
//   text.layers.{i}.mlp.down.{L}            down_proj
//   text.layers.{i}.attn.{k,v}_descale      fp32 [kv_heads], --kv-calib only (amax / 448; K post-rope, V raw)
// Norms are stored raw: the runtime uses a plain RMSNorm (x * w, not 1 + w).
#pragma once

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace r4dx_convert::llama {

constexpr const char* kModelArch = "llama";

// What the converter needs of config.json. Shapes of the tensors come from the safetensors headers; this is only what
// they are cross-checked against.
struct TextShape {
  int64_t hidden = 0;
  int64_t layers_total = 0;
  int64_t heads = 0, kv_heads = 0, head_dim = 0;
  int64_t intermediate = 0;
  int64_t vocab = 0;
};

// Strict: a missing key, model_type other than llama, a tied head (lm_head.weight would be absent), attention or MLP
// biases, rope scaling, an activation other than silu, or pretraining_tp != 1 throws std::runtime_error. `head_dim` is
// hidden / heads when absent.
TextShape ParseConfig(const nlohmann::json& cfg);

struct Kit {
  // The checkpoint.
  std::function<std::vector<std::string>()> all_names;
  std::function<bool(const std::string&)> has;
  std::function<std::vector<int64_t>(const std::string&)> shape;
  // One plan job + one emit job each (RunConvert's add_* closures).
  std::function<void(const std::string& hf, const std::string& name)> add_bf16;
  // A quantizable linear; several hf names are fused on the output axis in order. `head`: lm_head (the --lm-head
  // layout set applies instead of --layouts).
  std::function<void(const std::vector<std::string>& hf, const std::string& base, bool head)> add_linear;
  // fp8 KV dequant scales (N5-2): set by RunConvert only under --kv-calib, so a container converted without it has no
  // descale tensors and is byte-identical to one from before. Empty = not called.
  std::function<void(const std::string& name, int n_kv, int layer_idx, const char* kind)> add_descale;
};

// Registers every tensor of the first `layers` layers, then the embedding table, the final norm and lm_head. Throws
// before anything is planned when a tensor is missing or its shape disagrees with the config. Returns every HF tensor
// name it consumed.
std::set<std::string> AddTextStack(Kit& kit, const TextShape& s, int64_t layers);

// The coverage audit: checkpoint tensors AddTextStack did not consume and that are not allow-listed. Allow-listed:
// layers >= `layers` (--layers) and `model.layers.{i}.self_attn.rotary_emb.inv_freq` (old checkpoints carry the rope
// table; the runtime computes it). Anything else -- a mis-guessed name, a tensor the checkpoint grew -- must fail the
// run before a long conversion.
std::vector<std::string> UnconsumedTensors(const std::vector<std::string>& all_names,
                                           const std::set<std::string>& consumed, int64_t layers);

}  // namespace r4dx_convert::llama
