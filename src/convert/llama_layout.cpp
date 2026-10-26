#include "llama_layout.hpp"

#include <algorithm>
#include <stdexcept>

namespace r4dx_convert::llama {

namespace {

const std::string kLayerPrefix = "model.layers.";
const std::string kWho = "r4dx-convert (llama): ";

std::string ShapeStr(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) out += (i ? "," : "") + std::to_string(s[i]);
  return out + "]";
}

// The layer index of "model.layers.<i>.<rest>", or -1.
int64_t LayerOf(const std::string& name) {
  if (name.compare(0, kLayerPrefix.size(), kLayerPrefix) != 0) return -1;
  size_t p = kLayerPrefix.size();
  if (p >= name.size() || name[p] < '0' || name[p] > '9') return -1;
  int64_t idx = 0;
  while (p < name.size() && name[p] >= '0' && name[p] <= '9') idx = idx * 10 + (name[p++] - '0');
  return p < name.size() && name[p] == '.' ? idx : -1;
}

bool EndsWith(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

TextShape ParseConfig(const nlohmann::json& cfg) {
  const std::string who = kWho + "config.json ";
  if (!cfg.is_object()) throw std::runtime_error(who + "is not an object");
  auto req = [&](const char* key) -> const nlohmann::json& {
    if (!cfg.contains(key)) throw std::runtime_error(who + "is missing '" + key + "'");
    return cfg.at(key);
  };
  auto req_int = [&](const char* key) {
    const nlohmann::json& v = req(key);
    if (!v.is_number_integer() || v.get<int64_t>() <= 0)
      throw std::runtime_error(who + "'" + std::string(key) + "' must be a positive integer");
    return v.get<int64_t>();
  };
  auto flag = [&](const char* key) {  // an absent or null boolean key is false
    return cfg.contains(key) && cfg.at(key).is_boolean() && cfg.at(key).get<bool>();
  };
  if (cfg.value("model_type", std::string()) != kModelArch)
    throw std::runtime_error(who + "model_type is not '" + kModelArch + "'");
  TextShape s;
  s.hidden = req_int("hidden_size");
  s.layers_total = req_int("num_hidden_layers");
  s.heads = req_int("num_attention_heads");
  s.kv_heads = req_int("num_key_value_heads");
  if (s.heads % s.kv_heads != 0)
    throw std::runtime_error(who + "num_attention_heads is not a multiple of num_key_value_heads");
  if (cfg.contains("head_dim") && !cfg.at("head_dim").is_null()) {
    s.head_dim = req_int("head_dim");
  } else {
    if (s.hidden % s.heads != 0)
      throw std::runtime_error(who + "hidden_size is not a multiple of num_attention_heads (no head_dim key)");
    s.head_dim = s.hidden / s.heads;
  }
  s.intermediate = req_int("intermediate_size");
  s.vocab = req_int("vocab_size");
  // Variants the layout does not carry: refuse them instead of converting them wrongly.
  if (flag("tie_word_embeddings"))
    throw std::runtime_error(who + "tie_word_embeddings is true: the layout reads lm_head.weight as its own tensor");
  if (flag("attention_bias")) throw std::runtime_error(who + "attention_bias is true: the layout has no bias tensors");
  if (flag("mlp_bias")) throw std::runtime_error(who + "mlp_bias is true: the layout has no bias tensors");
  if (cfg.contains("rope_scaling") && !cfg.at("rope_scaling").is_null())
    throw std::runtime_error(who + "has rope_scaling: the runtime implements the unscaled rope only");
  if (cfg.contains("hidden_act") && cfg.at("hidden_act") != "silu")
    throw std::runtime_error(who + "hidden_act is " + cfg.at("hidden_act").dump() + " (want \"silu\")");
  if (cfg.contains("pretraining_tp") && cfg.at("pretraining_tp").is_number_integer() &&
      cfg.at("pretraining_tp").get<int64_t>() != 1)
    throw std::runtime_error(who + "pretraining_tp is not 1");
  return s;
}

std::set<std::string> AddTextStack(Kit& kit, const TextShape& s, int64_t layers) {
  if (layers < 0 || layers > s.layers_total)
    throw std::runtime_error(kWho + "--layers " + std::to_string(layers) + " is outside [0, " +
                             std::to_string(s.layers_total) + "]");
  std::set<std::string> consumed;
  // Existence and shape against the config, before any job is registered: a mis-guessed name or a checkpoint that is
  // not the config's is a one-second error, not a mid-conversion one.
  auto expect = [&](const std::string& name, std::vector<int64_t> want) {
    if (!kit.has(name)) throw std::runtime_error(kWho + "checkpoint has no tensor '" + name + "'");
    const std::vector<int64_t> got = kit.shape(name);
    if (got != want)
      throw std::runtime_error(kWho + "'" + name + "' has shape " + ShapeStr(got) + ", config says " + ShapeStr(want));
  };
  auto bf16 = [&](const std::string& hf, const std::string& name, std::vector<int64_t> want) {
    expect(hf, std::move(want));
    consumed.insert(hf);
    kit.add_bf16(hf, name);
  };
  auto linear = [&](std::vector<std::string> hf_names, const std::string& base, std::vector<int64_t> want_each,
                    bool head = false) {
    for (const std::string& n : hf_names) {
      expect(n, want_each);
      consumed.insert(n);
    }
    kit.add_linear(hf_names, base, head);
  };

  const int64_t H = s.hidden, I = s.intermediate;
  const int64_t q_rows = s.heads * s.head_dim, kv_rows = s.kv_heads * s.head_dim;
  for (int64_t i = 0; i < layers; ++i) {
    const std::string hf = kLayerPrefix + std::to_string(i) + ".";
    const std::string base = "text.layers." + std::to_string(i) + ".";
    bf16(hf + "input_layernorm.weight", base + "input_norm", {H});
    bf16(hf + "post_attention_layernorm.weight", base + "post_attn_norm", {H});
    linear({hf + "self_attn.q_proj.weight"}, base + "attn.q", {q_rows, H});
    linear({hf + "self_attn.k_proj.weight"}, base + "attn.k", {kv_rows, H});
    linear({hf + "self_attn.v_proj.weight"}, base + "attn.v", {kv_rows, H});
    linear({hf + "self_attn.o_proj.weight"}, base + "attn.o", {H, q_rows});
    if (kit.add_descale) {
      kit.add_descale(base + "attn.k_descale", static_cast<int>(s.kv_heads), static_cast<int>(i), "k");
      kit.add_descale(base + "attn.v_descale", static_cast<int>(s.kv_heads), static_cast<int>(i), "v");
    }
    linear({hf + "mlp.gate_proj.weight", hf + "mlp.up_proj.weight"}, base + "mlp.gate_up", {I, H});
    linear({hf + "mlp.down_proj.weight"}, base + "mlp.down", {H, I});
  }

  // Outside the layer stack: the embedding table, the final norm and the (untied) head.
  bf16("model.embed_tokens.weight", "text.embed_tokens", {s.vocab, H});
  bf16("model.norm.weight", "text.final_norm", {H});
  linear({"lm_head.weight"}, "lm_head", {s.vocab, H}, /*head=*/true);
  return consumed;
}

std::vector<std::string> UnconsumedTensors(const std::vector<std::string>& all_names,
                                           const std::set<std::string>& consumed, int64_t layers) {
  std::vector<std::string> out;
  for (const std::string& name : all_names) {
    if (consumed.count(name) != 0) continue;
    const int64_t layer = LayerOf(name);
    if (layer >= layers) continue;  // --layers: not converted by request
    if (layer >= 0 && EndsWith(name, ".self_attn.rotary_emb.inv_freq")) continue;  // the runtime computes the rope
    out.push_back(name);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace r4dx_convert::llama
