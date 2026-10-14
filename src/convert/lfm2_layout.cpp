#include "lfm2_layout.hpp"

#include <algorithm>
#include <stdexcept>

namespace r4dx_convert::lfm2 {

namespace {

const std::string kLayerPrefix = "model.layers.";

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

}  // namespace

TextShape ParseConfig(const nlohmann::json& cfg) {
  const std::string who = "r4dx-convert (lfm2_moe): config.json ";
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
  if (cfg.value("model_type", std::string()) != kModelArch)
    throw std::runtime_error(who + "model_type is not '" + kModelArch + "'");
  TextShape s;
  s.hidden = req_int("hidden_size");
  s.layers_total = req_int("num_hidden_layers");
  s.heads = req_int("num_attention_heads");
  s.kv_heads = req_int("num_key_value_heads");
  if (cfg.contains("head_dim") && !cfg.at("head_dim").is_null()) {
    s.head_dim = req_int("head_dim");
  } else {
    if (s.hidden % s.heads != 0)
      throw std::runtime_error(who + "hidden_size is not a multiple of num_attention_heads (no head_dim key)");
    s.head_dim = s.hidden / s.heads;
  }
  s.dense_intermediate = req_int("intermediate_size");
  s.moe_intermediate = req_int("moe_intermediate_size");
  s.num_experts = req_int("num_experts");
  s.conv_kernel = req_int("conv_L_cache");
  s.vocab = req_int("vocab_size");
  const nlohmann::json& nd = req("num_dense_layers");
  if (!nd.is_number_integer() || nd.get<int64_t>() < 0 || nd.get<int64_t>() > s.layers_total)
    throw std::runtime_error(who + "'num_dense_layers' must be an integer in [0, num_hidden_layers]");
  s.num_dense_layers = nd.get<int64_t>();
  for (const auto& lt : req("layer_types")) {
    const std::string t = lt.get<std::string>();
    if (t != "conv" && t != "full_attention")
      throw std::runtime_error(who + "has layer type '" + t + "' (want conv | full_attention)");
    s.layer_types.push_back(t);
  }
  if (static_cast<int64_t>(s.layer_types.size()) != s.layers_total)
    throw std::runtime_error(who + "layer_types length " + std::to_string(s.layer_types.size()) +
                             " != num_hidden_layers " + std::to_string(s.layers_total));
  // Variants the layout does not carry: refuse them instead of converting them wrongly.
  if (req("conv_bias").get<bool>())
    throw std::runtime_error(who + "conv_bias is true: the layout has no conv bias tensor");
  if (!req("use_expert_bias").get<bool>())
    throw std::runtime_error(who + "use_expert_bias is false: the layout always carries moe.expert_bias");
  if (!req("tie_word_embeddings").get<bool>())
    throw std::runtime_error(who + "tie_word_embeddings is false: lm_head is written from the embedding table");
  return s;
}

std::set<std::string> AddTextStack(Kit& kit, const TextShape& s, int64_t layers) {
  if (layers < 0 || layers > s.layers_total)
    throw std::runtime_error("r4dx-convert (lfm2_moe): --layers " + std::to_string(layers) + " is outside [0, " +
                             std::to_string(s.layers_total) + "]");
  std::set<std::string> consumed;
  // Existence and shape against the config, before any job is registered: a mis-guessed name or a checkpoint that is
  // not the config's is a one-second error, not a mid-conversion one.
  auto expect = [&](const std::string& name, std::vector<int64_t> want) {
    if (!kit.has(name)) throw std::runtime_error("r4dx-convert (lfm2_moe): checkpoint has no tensor '" + name + "'");
    const std::vector<int64_t> got = kit.shape(name);
    if (got != want)
      throw std::runtime_error("r4dx-convert (lfm2_moe): '" + name + "' has shape " + ShapeStr(got) + ", config says " +
                               ShapeStr(want));
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

  const int64_t H = s.hidden;
  for (int64_t i = 0; i < layers; ++i) {
    const std::string hf = kLayerPrefix + std::to_string(i) + ".";
    const std::string base = "text.layers." + std::to_string(i) + ".";
    const bool attn = s.IsAttn(i), dense = s.IsDense(i);

    bf16(hf + "operator_norm.weight", base + "operator_norm", {H});
    bf16(hf + "ffn_norm.weight", base + "ffn_norm", {H});

    if (attn) {
      const int64_t q_rows = s.heads * s.head_dim, kv_rows = s.kv_heads * s.head_dim;
      linear({hf + "self_attn.q_proj.weight"}, base + "attn.q", {q_rows, H});
      linear({hf + "self_attn.k_proj.weight"}, base + "attn.k", {kv_rows, H});
      linear({hf + "self_attn.v_proj.weight"}, base + "attn.v", {kv_rows, H});
      linear({hf + "self_attn.out_proj.weight"}, base + "attn.o", {H, q_rows});
      bf16(hf + "self_attn.q_layernorm.weight", base + "attn.q_norm", {s.head_dim});
      bf16(hf + "self_attn.k_layernorm.weight", base + "attn.k_norm", {s.head_dim});
    } else {
      linear({hf + "conv.in_proj.weight"}, base + "conv.in_proj", {3 * H, H});
      // The depthwise conv taps [H, 1, L] -> [H, L]: a squeeze of the singleton group axis, the bytes are unchanged.
      const std::string cw = hf + "conv.conv.weight";
      expect(cw, {H, 1, s.conv_kernel});
      consumed.insert(cw);
      kit.add_bf16_shaped(cw, base + "conv.weight", {H, s.conv_kernel});
      linear({hf + "conv.out_proj.weight"}, base + "conv.out_proj", {H, H});
    }

    if (dense) {
      const int64_t I = s.dense_intermediate;
      linear({hf + "feed_forward.w1.weight", hf + "feed_forward.w3.weight"}, base + "mlp.gate_up", {I, H});
      linear({hf + "feed_forward.w2.weight"}, base + "mlp.down", {H, I});
    } else {
      const int64_t I = s.moe_intermediate;
      // transformers 5 can serialize the experts as two fused 3-D tensors; this layout converts the per-expert
      // checkpoint only, and says so instead of leaving them for the coverage audit.
      for (const char* fused : {"feed_forward.experts.gate_up_proj", "feed_forward.experts.down_proj"})
        if (kit.has(hf + fused))
          throw std::runtime_error("r4dx-convert (lfm2_moe): '" + hf + fused +
                                   "' is a fused-experts checkpoint; only the per-expert experts.{e}.{w1,w2,w3} layout is supported");
      bf16(hf + "feed_forward.gate.weight", base + "moe.router", {s.num_experts, H});
      const std::string eb = hf + "feed_forward.expert_bias";
      expect(eb, {s.num_experts});
      consumed.insert(eb);
      kit.add_fp32_widen(eb, base + "moe.expert_bias");
      for (int64_t e = 0; e < s.num_experts; ++e) {
        const std::string ex = hf + "feed_forward.experts." + std::to_string(e) + ".";
        const std::string eb2 = base + "moe.experts." + std::to_string(e) + ".";
        linear({ex + "w1.weight", ex + "w3.weight"}, eb2 + "gate_up", {I, H});
        linear({ex + "w2.weight"}, eb2 + "down", {H, I});
      }
    }
  }

  // Outside the layer stack: the embedding table, the final norm (named embedding_norm in HF) and the tied head.
  const std::string embed = "model.embed_tokens.weight";
  bf16(embed, "text.embed_tokens", {s.vocab, H});
  bf16("model.embedding_norm.weight", "text.final_norm", {H});
  // lm_head is the SAME tensor as the embedding (tied); the container carries it untied, as Gemma's does.
  linear({embed}, "lm_head", {s.vocab, H}, /*head=*/true);
  return consumed;
}

std::vector<std::string> UnconsumedTensors(const std::vector<std::string>& all_names,
                                           const std::set<std::string>& consumed, int64_t layers) {
  std::vector<std::string> out;
  for (const std::string& name : all_names) {
    if (consumed.count(name) != 0) continue;
    const int64_t layer = LayerOf(name);
    if (layer >= layers) continue;  // --layers: not converted by request
    out.push_back(name);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace r4dx_convert::lfm2
