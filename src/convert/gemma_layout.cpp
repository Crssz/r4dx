#include "gemma_layout.hpp"

#include <algorithm>
#include <stdexcept>

namespace r4dx_convert::gemma {

namespace {

const std::string kTextPrefix = "model.language_model.";
const std::string kLayerPrefix = kTextPrefix + "layers.";

std::string ShapeStr(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) out += (i ? "," : "") + std::to_string(s[i]);
  return out + "]";
}

// The layer index of "model.language_model.layers.<i>.<rest>", or -1.
int LayerOf(const std::string& name) {
  if (name.compare(0, kLayerPrefix.size(), kLayerPrefix) != 0) return -1;
  size_t p = kLayerPrefix.size();
  if (p >= name.size() || name[p] < '0' || name[p] > '9') return -1;
  int idx = 0;
  while (p < name.size() && name[p] >= '0' && name[p] <= '9') idx = idx * 10 + (name[p++] - '0');
  return p < name.size() && name[p] == '.' ? idx : -1;
}

}  // namespace

TextShape ParseTextConfig(const nlohmann::json& text_cfg) {
  const std::string who = "r4dx-convert (gemma4_unified): text_config ";
  auto req = [&](const char* key) -> const nlohmann::json& {
    if (!text_cfg.contains(key)) throw std::runtime_error(who + "is missing '" + key + "'");
    return text_cfg.at(key);
  };
  auto req_int = [&](const char* key) {
    const nlohmann::json& v = req(key);
    if (!v.is_number_integer() || v.get<int64_t>() <= 0)
      throw std::runtime_error(who + "'" + std::string(key) + "' must be a positive integer");
    return v.get<int>();
  };
  TextShape s;
  s.hidden = req_int("hidden_size");
  s.layers_total = req_int("num_hidden_layers");
  s.heads = req_int("num_attention_heads");
  s.kv_sliding = req_int("num_key_value_heads");
  s.kv_full = req_int("num_global_key_value_heads");
  s.head_dim = req_int("head_dim");
  s.global_head_dim = req_int("global_head_dim");
  s.intermediate = req_int("intermediate_size");
  s.vocab = req_int("vocab_size");
  for (const auto& lt : req("layer_types")) {
    const std::string t = lt.get<std::string>();
    if (t != "sliding_attention" && t != "full_attention")
      throw std::runtime_error(who + "has layer type '" + t + "' (want sliding_attention | full_attention)");
    s.layer_types.push_back(t);
  }
  if (static_cast<int>(s.layer_types.size()) != s.layers_total)
    throw std::runtime_error(who + "layer_types length " + std::to_string(s.layer_types.size()) +
                             " != num_hidden_layers " + std::to_string(s.layers_total));
  // The layout has no v_proj on full layers and an untied head's tensor would go unconsumed: refuse
  // the variants instead of converting them wrongly.
  if (!req("attention_k_eq_v").get<bool>())
    throw std::runtime_error(who + "attention_k_eq_v is false: full layers would carry a v_proj this layout omits");
  if (!req("tie_word_embeddings").get<bool>())
    throw std::runtime_error(who + "tie_word_embeddings is false: lm_head is written from the embedding table");
  return s;
}

std::string PassthroughName(const std::string& hf, bool* is_vision) {
  struct Map {
    const char* from;
    const char* to;
    bool vision;
  };
  static const Map kMaps[] = {{"model.vision_embedder.", "vision.vision_embedder.", true},
                              {"model.embed_vision.", "vision.embed_vision.", true},
                              {"model.embed_audio.", "audio.embed_audio.", false}};
  for (const Map& m : kMaps) {
    const std::string from = m.from;
    if (hf.compare(0, from.size(), from) == 0) {
      if (is_vision != nullptr) *is_vision = m.vision;
      return std::string(m.to) + hf.substr(from.size());
    }
  }
  return "";
}

std::set<std::string> AddTextStack(Kit& kit, const TextShape& s, int layers, bool do_vision, bool do_audio) {
  if (layers < 0 || layers > s.layers_total)
    throw std::runtime_error("r4dx-convert (gemma4_unified): --layers " + std::to_string(layers) +
                             " is outside [0, " + std::to_string(s.layers_total) + "]");
  std::set<std::string> consumed;
  // Existence and shape against the config, before any job is registered: a mis-guessed name or a
  // checkpoint that is not the config's is a one-second error, not a mid-conversion one.
  auto expect = [&](const std::string& name, std::vector<int64_t> want) {
    if (!kit.has(name)) throw std::runtime_error("r4dx-convert (gemma4_unified): checkpoint has no tensor '" + name + "'");
    const std::vector<int64_t> got = kit.shape(name);
    if (got != want)
      throw std::runtime_error("r4dx-convert (gemma4_unified): '" + name + "' has shape " + ShapeStr(got) +
                               ", config says " + ShapeStr(want));
  };
  auto bf16 = [&](const std::string& hf, const std::string& name, std::vector<int64_t> want) {
    expect(hf, std::move(want));
    consumed.insert(hf);
    kit.add_bf16(hf, name);
  };
  auto folded_norm = [&](const std::string& hf, const std::string& name) {
    expect(hf, {s.hidden});
    consumed.insert(hf);
    kit.add_folded_norm(hf, name);
  };
  auto linear = [&](std::vector<std::string> hf_names, const std::string& base, const Fold& fold,
                    std::vector<int64_t> want_each) {
    for (const std::string& n : hf_names) {
      expect(n, want_each);
      consumed.insert(n);
    }
    kit.add_linear(hf_names, base, fold);
  };

  const int64_t hidden = s.hidden, inter = s.intermediate;
  for (int i = 0; i < layers; ++i) {
    const std::string hf = kLayerPrefix + std::to_string(i) + ".";
    const std::string base = "text.layers." + std::to_string(i) + ".";
    const bool full = s.IsFull(i);
    const int64_t hd = s.HeadDim(i), kv = s.KvHeads(i);
    const int64_t q_rows = s.heads * hd, kv_rows = kv * hd;

    // v_proj: sliding layers have one; a full layer's V is the raw k_proj output (attention_k_eq_v),
    // so a v_proj there means the checkpoint is not the layout this branch knows.
    const bool has_v = kit.has(hf + "self_attn.v_proj.weight");
    if (full && has_v)
      throw std::runtime_error("r4dx-convert (gemma4_unified): layer " + std::to_string(i) +
                               " is a full-attention layer with attention_k_eq_v but the checkpoint has a v_proj");
    if (!full && !has_v)
      throw std::runtime_error("r4dx-convert (gemma4_unified): sliding layer " + std::to_string(i) +
                               " has no self_attn.v_proj.weight");

    // Folded by --rotate into the in-projections that follow them; stored as-is otherwise.
    const std::string input_norm = hf + "input_layernorm.weight";
    const std::string pre_ff_norm = hf + "pre_feedforward_layernorm.weight";
    folded_norm(input_norm, base + "input_layernorm");
    bf16(hf + "post_attention_layernorm.weight", base + "post_attention_layernorm", {hidden});
    folded_norm(pre_ff_norm, base + "pre_feedforward_layernorm");
    bf16(hf + "post_feedforward_layernorm.weight", base + "post_feedforward_layernorm", {hidden});

    Fold mixer_in;
    mixer_in.in_norm_hf = input_norm;
    Fold mlp_in;
    mlp_in.in_norm_hf = pre_ff_norm;
    Fold o_out;
    o_out.out = full ? OutSite::kOFull : OutSite::kOSliding;
    Fold down_out;
    down_out.out = OutSite::kDown;

    linear({hf + "self_attn.q_proj.weight"}, base + "attn.q", mixer_in, {q_rows, hidden});
    linear({hf + "self_attn.k_proj.weight"}, base + "attn.k", mixer_in, {kv_rows, hidden});
    if (has_v) linear({hf + "self_attn.v_proj.weight"}, base + "attn.v", mixer_in, {kv_rows, hidden});
    linear({hf + "self_attn.o_proj.weight"}, base + "attn.o", o_out, {hidden, q_rows});
    bf16(hf + "self_attn.q_norm.weight", base + "attn.q_norm", {hd});
    bf16(hf + "self_attn.k_norm.weight", base + "attn.k_norm", {hd});
    // fp8 KV dequant scales, per KV head of THIS layer (8 sliding, 1 full): the full layers' v_descale
    // is the scale of the raw k_proj output that is cached as V.
    kit.add_descale(base + "attn.k_descale", static_cast<int>(kv), i, "k");
    kit.add_descale(base + "attn.v_descale", static_cast<int>(kv), i, "v");

    linear({hf + "mlp.gate_proj.weight", hf + "mlp.up_proj.weight"}, base + "mlp.gate_up", mlp_in, {inter, hidden});
    linear({hf + "mlp.down_proj.weight"}, base + "mlp.down", down_out, {hidden, inter});

    // The [1] bf16 per-layer buffer, applied once after the MLP residual add; widened to fp32 [1, 4].
    const std::string scalar = hf + "layer_scalar";
    expect(scalar, {1});
    consumed.insert(scalar);
    kit.add_fp32_widen(scalar, base + "layer_scalar");
  }

  // Outside the layer stack, never folded by --rotate (the runtime applies x Q after the embedding gather
  // and x Q^T before final_norm).
  const std::string embed = kTextPrefix + "embed_tokens.weight";
  bf16(embed, "text.embed_tokens", {s.vocab, hidden});
  bf16(kTextPrefix + "norm.weight", "text.final_norm", {hidden});
  Fold head;
  head.head = true;
  // lm_head is the SAME tensor as the embedding (tied); the container carries it untied, as Qwen's does.
  linear({kTextPrefix + "embed_tokens.weight"}, "lm_head", head, {s.vocab, hidden});

  // Vision / audio: bf16 passthrough under their modality, `model.` stripped.
  if (do_vision || do_audio) {
    std::vector<std::string> names = kit.all_names();
    std::sort(names.begin(), names.end());
    int n_vision = 0, n_audio = 0;
    for (const std::string& name : names) {
      bool vision = false;
      const std::string out = PassthroughName(name, &vision);
      if (out.empty() || (vision && !do_vision) || (!vision && !do_audio)) continue;
      consumed.insert(name);
      kit.add_bf16(name, out);
      (vision ? n_vision : n_audio) += 1;
    }
    if (do_vision && n_vision == 0)
      throw std::runtime_error("r4dx-convert (gemma4_unified): --vision on but the checkpoint has no vision tensors");
    if (do_audio && n_audio == 0)
      throw std::runtime_error("r4dx-convert (gemma4_unified): --audio on but the checkpoint has no audio tensors");
  }
  return consumed;
}

std::vector<std::string> UnconsumedTensors(const std::vector<std::string>& all_names,
                                           const std::set<std::string>& consumed, int layers, bool do_vision,
                                           bool do_audio) {
  std::vector<std::string> out;
  for (const std::string& name : all_names) {
    if (consumed.count(name) != 0) continue;
    const int layer = LayerOf(name);
    if (layer >= layers) continue;  // --layers: not converted by request
    bool vision = false;
    if (!PassthroughName(name, &vision).empty() && ((vision && !do_vision) || (!vision && !do_audio))) continue;
    out.push_back(name);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace r4dx_convert::gemma
