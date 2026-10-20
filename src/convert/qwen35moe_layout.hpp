// r4dx-convert's qwen3_5_moe branch (docs/qwen35moe-plan.md M3): which HF tensors of an Ornith-1.5-122B checkpoint
// become which container tensors, with the shape cross-checks and the coverage audit.
//
// Same construction as qwen4exp_layout.hpp -- AddTextStack drives a Kit of std::function handles onto RunConvert's closures --
// and the same container tensor names, minus everything qwen4_exp-specific (hyper-connections, PLE, the QSA indexer) plus the
// plain residual stream's two norms per layer. The routed experts arrive as per-expert virtual tensors
// (fused_experts_source.hpp turns HF's fused gate_up_proj / down_proj into `mlp.experts.{e}.{gate,up,down}_proj.weight`).
//
// Container tensors (HF names under `model.language_model.`; {layout} = bf16 | w4a16 | trellis):
//   text.embed_tokens                       embed_tokens.weight                      bf16 [vocab, hidden]
//   text.final_norm                         norm.weight                              bf16 [hidden], raw (the kernel applies 1 + w)
//   lm_head.{layout}                        lm_head.weight (UNtied)                  the head recipe flags
//   text.layers.{i}.input_layernorm / post_attention_layernorm      bf16 [hidden], raw
//   text.layers.{i}.moe.router              mlp.gate.weight                          bf16 [E, hidden]
//   text.layers.{i}.moe.shared.{gate,up,down}   mlp.shared_expert.{gate,up,down}_proj.weight   bf16 linears (.bf16.w)
//   text.layers.{i}.moe.shared_gate         mlp.shared_expert_gate.weight            bf16 [1, hidden]
//   text.layers.{i}.moe.experts.{e}.gate_up.{layout}   gate rows, then up rows of the expert (one two-part linear)
//   text.layers.{i}.moe.experts.{e}.down.{layout}
//   GDN layers:   gdn.in_proj_qkv.{layout}, gdn.in_proj_z.{layout}, gdn.out_proj.{layout}   linear_attn.*
//                 gdn.in_proj_a, gdn.in_proj_b (bf16), gdn.conv1d_weight (bf16 [qkv,1,k]),
//                 gdn.A_log, gdn.dt_bias (fp32), gdn.norm_weight (bf16)
//   attention layers: attn.qg.{layout} (q_proj as is: per-head [q | gate] rows), attn.k, attn.v, attn.o ({layout}),
//                 attn.q_norm, attn.k_norm (bf16), attn.k_descale / attn.v_descale (--kv-calib, fp32 [kv_heads])
//
// Not converted, on purpose (the container says so in __metadata__.r4dx_convert_run.skipped): model.visual.* and mtp.*.
#pragma once

#include <string>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"
#include "qwen4exp_layout.hpp"

namespace r4dx_convert::qwen35moe {

constexpr const char* kModelArch = "qwen3_5_moe";

// The Kit and LinearRole are qwen4_exp's: the closures RunConvert hands over do not depend on the architecture.
using Kit = qwen4exp::Kit;
using LinearRole = qwen4exp::LinearRole;

struct TextShape {
  int hidden = 0;
  int layers_total = 0;
  int vocab = 0;
  int heads = 0, kv_heads = 0, head_dim = 0;
  int num_experts = 0, moe_inter = 0, shared_inter = 0;
  int linear_key_heads = 0, linear_value_heads = 0, linear_key_dim = 0, linear_value_dim = 0, conv_kernel = 0;
  std::vector<std::string> layer_types;  // "linear_attention" | "full_attention"

  bool IsFull(int i) const { return layer_types.at(static_cast<size_t>(i)) == "full_attention"; }
  int64_t QkvRows() const {
    return 2LL * linear_key_heads * linear_key_dim + static_cast<int64_t>(linear_value_heads) * linear_value_dim;
  }
};

// Strict: a missing key, a layer type other than linear_attention/full_attention, a non-empty mlp_only_layers or
// tie_word_embeddings true throws std::runtime_error naming it. `config` is the whole config.json.
TextShape ParseTextConfig(const nlohmann::json& config);

// Registers every tensor of the text stack and the globals; `layers` <= layers_total converts the first `layers`
// layers only (--layers). Throws before anything is planned when a tensor is missing or its shape disagrees with the
// config. Returns every HF tensor name it consumed, for UnconsumedTensors.
std::unordered_set<std::string> AddTextStack(Kit& kit, const TextShape& s, int layers);

// True for the checkpoint tensors this converter leaves out on purpose (model.visual.*, mtp.*).
bool IsSkipped(const std::string& hf_name);

// The coverage audit: checkpoint tensors neither consumed nor skipped on purpose. Layers >= `layers` (--layers) are
// skipped by request. Sorted.
std::vector<std::string> UnconsumedTensors(const std::vector<std::string>& all_names,
                                           const std::unordered_set<std::string>& consumed, int layers);

}  // namespace r4dx_convert::qwen35moe
