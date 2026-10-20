#include "qwen35moe_layout.hpp"

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace r4dx_convert::qwen35moe {

namespace {

const std::string kTextPrefix = "model.language_model.";
const std::string kLayerPrefix = kTextPrefix + "layers.";
const std::string kWho = "r4dx-convert (qwen3_5_moe): ";

std::string ShapeStr(const std::vector<int64_t>& s) { return nlohmann::json(s).dump(); }

// The layer index of "model.language_model.layers.<i>.<rest>", or -1 (also for an index that does not fit an int).
int LayerOf(const std::string& name) {
  if (name.compare(0, kLayerPrefix.size(), kLayerPrefix) != 0) return -1;
  size_t i = kLayerPrefix.size();
  int v = 0;
  bool any = false;
  while (i < name.size() && name[i] >= '0' && name[i] <= '9') {
    const int digit = name[i] - '0';
    if (v > (std::numeric_limits<int>::max() - digit) / 10) return -1;
    v = v * 10 + digit;
    any = true;
    ++i;
  }
  return any && i < name.size() && name[i] == '.' ? v : -1;
}

bool StartsWith(const std::string& s, const char* prefix) { return s.rfind(prefix, 0) == 0; }

const nlohmann::json& Req(const nlohmann::json& j, const char* key, const char* where) {
  if (!j.is_object() || !j.contains(key)) throw std::runtime_error(kWho + where + " has no '" + key + "'");
  return j.at(key);
}

int ReqInt(const nlohmann::json& j, const char* key, const char* where) {
  const nlohmann::json& v = Req(j, key, where);
  if (!v.is_number_integer() || v.get<int64_t>() <= 0 || v.get<int64_t>() > 0x7fffffff)
    throw std::runtime_error(kWho + where + "." + key + " is " + v.dump() + ", expected a positive integer");
  return v.get<int>();
}

}  // namespace

TextShape ParseTextConfig(const nlohmann::json& config) {
  const nlohmann::json& t = Req(config, "text_config", "config.json");
  const char* where = "text_config";
  TextShape s;
  s.hidden = ReqInt(t, "hidden_size", where);
  s.layers_total = ReqInt(t, "num_hidden_layers", where);
  s.vocab = ReqInt(t, "vocab_size", where);
  s.heads = ReqInt(t, "num_attention_heads", where);
  s.kv_heads = ReqInt(t, "num_key_value_heads", where);
  s.head_dim = ReqInt(t, "head_dim", where);
  s.num_experts = ReqInt(t, "num_experts", where);
  s.moe_inter = ReqInt(t, "moe_intermediate_size", where);
  s.shared_inter = ReqInt(t, "shared_expert_intermediate_size", where);
  s.linear_key_heads = ReqInt(t, "linear_num_key_heads", where);
  s.linear_value_heads = ReqInt(t, "linear_num_value_heads", where);
  s.linear_key_dim = ReqInt(t, "linear_key_head_dim", where);
  s.linear_value_dim = ReqInt(t, "linear_value_head_dim", where);
  s.conv_kernel = ReqInt(t, "linear_conv_kernel_dim", where);
  for (const auto& lt : Req(t, "layer_types", where)) {
    const std::string v = lt.is_string() ? lt.get<std::string>() : lt.dump();
    if (v != "linear_attention" && v != "full_attention")
      throw std::runtime_error(kWho + "layer_types has '" + v + "', expected linear_attention or full_attention");
    s.layer_types.push_back(v);
  }
  if (static_cast<int>(s.layer_types.size()) != s.layers_total)
    throw std::runtime_error(kWho + "layer_types length " + std::to_string(s.layer_types.size()) +
                             " != num_hidden_layers " + std::to_string(s.layers_total));
  if (t.contains("mlp_only_layers") && t.at("mlp_only_layers").is_array() && !t.at("mlp_only_layers").empty())
    throw std::runtime_error(kWho + "mlp_only_layers is not empty: dense layers inside the MoE stack are not converted");
  // The layout writes an untied head from lm_head.weight; a tied checkpoint has none.
  const nlohmann::json* tie = t.contains("tie_word_embeddings") ? &t.at("tie_word_embeddings")
                              : config.contains("tie_word_embeddings") ? &config.at("tie_word_embeddings")
                                                                       : nullptr;
  if (tie && tie->is_boolean() && tie->get<bool>())
    throw std::runtime_error(kWho + "tie_word_embeddings is true: the layout converts lm_head.weight, which a tied "
                                    "checkpoint does not have");
  return s;
}

std::unordered_set<std::string> AddTextStack(Kit& kit, const TextShape& s, int layers) {
  if (layers < 0 || layers > s.layers_total)
    throw std::runtime_error(kWho + "--layers " + std::to_string(layers) + " is outside [0, " +
                             std::to_string(s.layers_total) + "]");
  std::unordered_set<std::string> consumed;
  auto expect = [&](const std::string& name, const std::vector<int64_t>& want) {
    if (!kit.has(name)) throw std::runtime_error(kWho + "checkpoint has no tensor '" + name + "'");
    const std::vector<int64_t> got = kit.shape(name);
    if (got != want)
      throw std::runtime_error(kWho + "'" + name + "' has shape " + ShapeStr(got) + ", config says " + ShapeStr(want));
  };
  auto bf16 = [&](const std::string& hf, const std::string& name, const std::vector<int64_t>& want) {
    expect(hf, want);
    consumed.insert(hf);
    kit.add_bf16(hf, name);
  };
  auto fp32 = [&](const std::string& hf, const std::string& name, const std::vector<int64_t>& want) {
    expect(hf, want);
    consumed.insert(hf);
    kit.add_fp32_widen(hf, name);
  };
  auto linear = [&](const std::vector<std::string>& hf_names, const std::string& base, LinearRole role,
                    const std::vector<int64_t>& want_each) {
    for (const std::string& n : hf_names) {
      expect(n, want_each);
      consumed.insert(n);
    }
    kit.add_linear(hf_names, base, role);
  };

  const int64_t H = s.hidden;
  const LinearRole body = LinearRole::kBody, bf16_only = LinearRole::kBf16Only;
  for (int i = 0; i < layers; ++i) {
    const std::string hf = kLayerPrefix + std::to_string(i) + ".";
    const std::string base = "text.layers." + std::to_string(i) + ".";
    const bool full = s.IsFull(i);

    bf16(hf + "input_layernorm.weight", base + "input_layernorm", {H});
    bf16(hf + "post_attention_layernorm.weight", base + "post_attention_layernorm", {H});

    if (full) {
      if (kit.has(hf + "linear_attn.in_proj_qkv.weight"))
        throw std::runtime_error(kWho + "layer " + std::to_string(i) +
                                 " is full_attention in the config but the checkpoint has linear_attn tensors");
      const int64_t q_rows = static_cast<int64_t>(s.heads) * s.head_dim, kv_rows = static_cast<int64_t>(s.kv_heads) * s.head_dim;
      // q_proj is HF's fused query + output gate: per-head [q | gate] rows, the layout attn.qg has, converted as is.
      linear({hf + "self_attn.q_proj.weight"}, base + "attn.qg", body, {2 * q_rows, H});
      linear({hf + "self_attn.k_proj.weight"}, base + "attn.k", body, {kv_rows, H});
      linear({hf + "self_attn.v_proj.weight"}, base + "attn.v", body, {kv_rows, H});
      linear({hf + "self_attn.o_proj.weight"}, base + "attn.o", body, {H, q_rows});
      bf16(hf + "self_attn.q_norm.weight", base + "attn.q_norm", {s.head_dim});
      bf16(hf + "self_attn.k_norm.weight", base + "attn.k_norm", {s.head_dim});
      kit.add_descale(base + "attn.k_descale", s.kv_heads, i, "k");
      kit.add_descale(base + "attn.v_descale", s.kv_heads, i, "v");
    } else {
      if (kit.has(hf + "self_attn.q_proj.weight"))
        throw std::runtime_error(kWho + "layer " + std::to_string(i) +
                                 " is linear_attention in the config but the checkpoint has self_attn tensors");
      const int64_t qkv = s.QkvRows(), z = static_cast<int64_t>(s.linear_value_heads) * s.linear_value_dim;
      linear({hf + "linear_attn.in_proj_qkv.weight"}, base + "gdn.in_proj_qkv", body, {qkv, H});
      linear({hf + "linear_attn.in_proj_z.weight"}, base + "gdn.in_proj_z", body, {z, H});
      bf16(hf + "linear_attn.in_proj_b.weight", base + "gdn.in_proj_b", {s.linear_value_heads, H});
      bf16(hf + "linear_attn.in_proj_a.weight", base + "gdn.in_proj_a", {s.linear_value_heads, H});
      bf16(hf + "linear_attn.conv1d.weight", base + "gdn.conv1d_weight", {qkv, 1, s.conv_kernel});
      fp32(hf + "linear_attn.A_log", base + "gdn.A_log", {s.linear_value_heads});
      fp32(hf + "linear_attn.dt_bias", base + "gdn.dt_bias", {s.linear_value_heads});
      bf16(hf + "linear_attn.norm.weight", base + "gdn.norm_weight", {s.linear_value_dim});
      linear({hf + "linear_attn.out_proj.weight"}, base + "gdn.out_proj", body, {H, z});
    }

    // MoE: router and shared expert stay bf16; the routed experts are per-expert linears (virtual tensors over HF's fused
    // 3-D ones) and gate+up is ONE two-part container linear.
    bf16(hf + "mlp.gate.weight", base + "moe.router", {s.num_experts, H});
    linear({hf + "mlp.shared_expert.gate_proj.weight"}, base + "moe.shared.gate", bf16_only, {s.shared_inter, H});
    linear({hf + "mlp.shared_expert.up_proj.weight"}, base + "moe.shared.up", bf16_only, {s.shared_inter, H});
    linear({hf + "mlp.shared_expert.down_proj.weight"}, base + "moe.shared.down", bf16_only, {H, s.shared_inter});
    bf16(hf + "mlp.shared_expert_gate.weight", base + "moe.shared_gate", {1, H});
    for (int e = 0; e < s.num_experts; ++e) {
      const std::string es = std::to_string(e);
      const std::string h = hf + "mlp.experts." + es + ".";
      const std::string b = base + "moe.experts." + es + ".";
      linear({h + "gate_proj.weight", h + "up_proj.weight"}, b + "gate_up", body, {s.moe_inter, H});
      linear({h + "down_proj.weight"}, b + "down", body, {H, s.moe_inter});
    }
  }

  // Outside the layer stack.
  bf16(kTextPrefix + "embed_tokens.weight", "text.embed_tokens", {s.vocab, H});
  bf16(kTextPrefix + "norm.weight", "text.final_norm", {H});
  const LinearRole head = LinearRole::kHead;
  linear({"lm_head.weight"}, "lm_head", head, {s.vocab, H});
  return consumed;
}

bool IsSkipped(const std::string& hf_name) {
  return StartsWith(hf_name, "model.visual.") || StartsWith(hf_name, "mtp.");
}

std::vector<std::string> UnconsumedTensors(const std::vector<std::string>& all_names,
                                           const std::unordered_set<std::string>& consumed, int layers) {
  std::vector<std::string> out;
  for (const std::string& name : all_names) {
    if (consumed.count(name) != 0 || IsSkipped(name)) continue;
    const int layer = LayerOf(name);
    if (layer >= layers) continue;  // --layers: not converted by request
    out.push_back(name);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace r4dx_convert::qwen35moe
