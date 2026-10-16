// r4dx::model::Qwen4ExpConfig -- the text-model geometry of qwen4_exp (Qwen3.8-Flash-Next), parsed
// STRICTLY from a checkpoint's / container's config.json `text_config` (docs/qwen4exp-plan.md 2.1-2.4,
// 4.1, task M1-1). HF source of truth: transformers/models/qwen4_exp/modular_qwen4_exp.py
// (Qwen4ExpTextConfig and its validate_architecture).
//
// Header-only, HIP-free (like gemma_config.h): tests/model/test_qwen4exp_config.cpp runs it on the
// real Swift-1.5-Qwen3.8-Flash-Next config without a device.
//
// Shape of one decoder layer: a mixer (GDN, or QSA = full attention restricted by a block indexer),
// then a 512-expert top-10 MoE with a shared expert; each sub-block sits inside a hyper-connection
// gated residual over hc_count (4) residual streams. One layer (ple_layer_ids, 1-indexed) also adds a
// per-layer embedding fed by a hashed 3-gram table. Everything the runtime design assumes (4 streams,
// one indexer key head, 3-grams, every 4th layer QSA) is ASSERTED here rather than loaded wrongly.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "model_config.h"

namespace r4dx::model {

struct Qwen4ExpConfig {
  // HF spells a QSA layer "full_attention" in the checkpoint's config.json and rewrites it to
  // "indexed_attention" in __post_init__; both mean kQsa here.
  enum class LayerType { kGdn, kQsa };
  // The GDN gated norm's gate activation (output_gate_type, falling back to hidden_act as HF does).
  enum class OutputGate { kSigmoid, kSilu };

  static constexpr int64_t kFullAttentionInterval = 4;  // QSA at layers 3, 7, ..., 47 (plan 2.1)

  // ---- 2.1 shapes ------------------------------------------------------------------------------
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t vocab_size = 0;
  double rms_norm_eps = 0.0;
  std::vector<LayerType> layer_types;  // one per layer
  int64_t max_position_embeddings = 0;
  bool tie_word_embeddings = false;
  int64_t eos_token_id = 0;  // the first id when config.json carries a list (HF does the same)

  // GDN
  int64_t linear_num_key_heads = 0;
  int64_t linear_num_value_heads = 0;
  int64_t linear_key_head_dim = 0;
  int64_t linear_value_head_dim = 0;
  int64_t linear_conv_kernel_dim = 0;
  OutputGate output_gate_type = OutputGate::kSigmoid;

  // QSA attention
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  double partial_rotary_factor = 0.0;
  double rope_theta = 0.0;
  std::vector<int64_t> mrope_section;
  bool mrope_interleaved = false;

  // MoE
  int64_t num_experts = 0;
  int64_t num_experts_per_tok = 0;
  int64_t moe_intermediate_size = 0;
  int64_t shared_expert_intermediate_size = 0;
  bool norm_topk_prob = true;  // HF default when the key is absent

  // ---- 2.2 hyper-connections -------------------------------------------------------------------
  int64_t hc_count = 0;
  int64_t hc_lowrank = 0;

  // ---- 2.3 PLE + n-gram embedding --------------------------------------------------------------
  std::vector<int64_t> ple_layer_ids;      // 1-indexed, as config.json (sorted, unique)
  std::vector<int64_t> ple_layer_indices;  // the same layers, 0-based decoder layer indices
  int64_t ple_embed_dim = 0;
  int64_t ple_conv_kernel_size = 0;
  int64_t ngram_size = 0;
  int64_t heads_per_ngram = 0;
  int64_t ngram_vocab_size_base = 0;
  int64_t make_ngram_vocab_size_divisible_by = 0;
  int64_t seed = 1234;  // HF default when the key is absent
  int64_t split_ngram_parts = 0;

  // ---- 2.4 QSA indexer -------------------------------------------------------------------------
  int64_t indexer_n_heads = 0;
  int64_t indexer_kv_heads = 0;
  int64_t indexer_head_dim = 0;
  int64_t indexer_budget = 0;
  int64_t indexer_compress_ratio = 0;

  // ---- derived ---------------------------------------------------------------------------------
  // Hashed n-gram heads: orders 2..ngram_size, heads_per_ngram each (16), each of dim
  // ple_embed_dim / NgramHeads() (160).
  int64_t NgramHeads() const { return (ngram_size - 1) * heads_per_ngram; }
  int64_t NgramHeadDim() const { return ple_embed_dim / NgramHeads(); }
  // Blocks the indexer keeps per query (2048 / 4 = 512).
  int64_t BlockTopK() const { return indexer_budget / indexer_compress_ratio; }
  // The longest visible context at which dense attention equals QSA: every complete block selected
  // plus an incomplete tail of compress_ratio - 1 tokens (512 * 4 + 3 = 2051, plan 2.4).
  int64_t DenseExactMaxTokens() const { return BlockTopK() * indexer_compress_ratio + indexer_compress_ratio - 1; }
  // Rotated dims of a QSA head (256 * 0.25 = 64).
  int64_t RotaryDim() const { return static_cast<int64_t>(partial_rotary_factor * static_cast<double>(head_dim)); }

  bool IsQsaLayer(int64_t i) const {
    if (i < 0 || i >= static_cast<int64_t>(layer_types.size()))
      throw std::out_of_range("Qwen4ExpConfig::IsQsaLayer: layer index out of range");
    return layer_types[static_cast<size_t>(i)] == LayerType::kQsa;
  }
  bool IsGdnLayer(int64_t i) const { return !IsQsaLayer(i); }
  int64_t NumQsaLayers() const {
    int64_t n = 0;
    for (LayerType t : layer_types) n += t == LayerType::kQsa ? 1 : 0;
    return n;
  }
  int64_t NumGdnLayers() const { return static_cast<int64_t>(layer_types.size()) - NumQsaLayers(); }
  bool IsPleLayer(int64_t i) const {
    for (int64_t p : ple_layer_indices)
      if (p == i) return true;
    return false;
  }

  // The thin generic view TextModel::Config() hands the CLI, server and tools (they read hidden_size,
  // num_hidden_layers, vocab_size and arch): `arch` is kQwen4Exp, layer_types use the checkpoint's
  // "linear_attention" / "full_attention" spelling (a QSA layer is a full_attention one), the attention
  // geometry is the QSA layers', the GDN fields are the GDN layers'. intermediate_size is the routed
  // experts' (moe_intermediate_size): this arch has no dense MLP.
  ModelConfig ToModelConfig() const {
    ModelConfig m;
    m.arch = Arch::kQwen4Exp;
    m.hidden_size = hidden_size;
    m.num_hidden_layers = num_hidden_layers;
    m.layer_types.reserve(layer_types.size());
    for (LayerType t : layer_types) m.layer_types.push_back(t == LayerType::kQsa ? "full_attention" : "linear_attention");
    m.num_attention_heads = num_attention_heads;
    m.num_key_value_heads = num_key_value_heads;
    m.head_dim = head_dim;
    m.attn_output_gate = true;
    m.intermediate_size = moe_intermediate_size;
    m.partial_rotary_factor = partial_rotary_factor;
    m.rope_theta = rope_theta;
    m.mrope_interleaved = mrope_interleaved;
    m.mrope_section = mrope_section;
    m.linear_num_key_heads = linear_num_key_heads;
    m.linear_num_value_heads = linear_num_value_heads;
    m.linear_key_head_dim = linear_key_head_dim;
    m.linear_value_head_dim = linear_value_head_dim;
    m.linear_conv_kernel_dim = linear_conv_kernel_dim;
    m.rms_norm_eps = rms_norm_eps;
    m.vocab_size = vocab_size;
    m.tie_word_embeddings = tie_word_embeddings;
    m.mtp_num_hidden_layers = 0;
    return m;
  }

  // `config` is either the full config.json (a top-level object holding "text_config", as the
  // checkpoint and a container's __metadata__.model_config carry it) or the bare text_config.
  // Every geometry field is required (missing -> std::runtime_error naming the key), except the
  // three HF defaults: norm_topk_prob (true), seed (1234) and output_gate_type (hidden_act).
  static Qwen4ExpConfig FromJson(const nlohmann::json& config) {
    const std::string who = "Qwen4ExpConfig::FromJson: ";
    if (!config.is_object()) throw std::runtime_error(who + "config is not an object");
    if (config.contains("text_config")) {
      if (config.contains("model_type") && !StrEq(config.at("model_type"), "qwen4_exp"))
        throw std::runtime_error(who + "model_type " + config.at("model_type").dump() + " (want \"qwen4_exp\")");
      return FromTextConfig(config.at("text_config"));
    }
    return FromTextConfig(config);
  }

 private:
  static bool StrEq(const nlohmann::json& v, const char* s) {
    return v.is_string() && v.get_ref<const std::string&>() == s;
  }

  static Qwen4ExpConfig FromTextConfig(const nlohmann::json& t) {
    const std::string who = "Qwen4ExpConfig::FromJson: ";
    if (!t.is_object()) throw std::runtime_error(who + "text_config is not an object");
    auto fail = [&](const std::string& why) { throw std::runtime_error(who + why); };
    auto req_in = [&](const nlohmann::json& obj, const std::string& where, const char* key) -> const nlohmann::json& {
      if (!obj.contains(key) || obj.at(key).is_null()) fail(where + " missing '" + key + "'");
      return obj.at(key);
    };
    auto req = [&](const char* key) -> const nlohmann::json& { return req_in(t, "text_config", key); };
    auto as_int = [&](const nlohmann::json& v, const char* key) {
      if (!v.is_number_integer()) fail("'" + std::string(key) + "' is not an integer: " + v.dump());
      return v.get<int64_t>();
    };
    auto as_num = [&](const nlohmann::json& v, const char* key) {
      if (!v.is_number()) fail("'" + std::string(key) + "' is not a number: " + v.dump());
      return v.get<double>();
    };
    auto as_bool = [&](const nlohmann::json& v, const char* key) {
      if (!v.is_boolean()) fail("'" + std::string(key) + "' is not a boolean: " + v.dump());
      return v.get<bool>();
    };
    auto positive = [&](const char* key) {
      const int64_t v = as_int(req(key), key);
      if (v <= 0) fail("'" + std::string(key) + "' must be > 0, got " + std::to_string(v));
      return v;
    };

    if (t.contains("model_type") && !StrEq(t.at("model_type"), "qwen4_exp_text"))
      fail("text_config.model_type " + t.at("model_type").dump() + " (want \"qwen4_exp_text\")");

    Qwen4ExpConfig c;
    c.hidden_size = positive("hidden_size");
    c.num_hidden_layers = positive("num_hidden_layers");
    c.vocab_size = positive("vocab_size");
    c.rms_norm_eps = as_num(req("rms_norm_eps"), "rms_norm_eps");
    c.max_position_embeddings = positive("max_position_embeddings");
    c.tie_word_embeddings = as_bool(req("tie_word_embeddings"), "tie_word_embeddings");
    {
      const nlohmann::json& e = req("eos_token_id");
      if (e.is_array()) {
        if (e.empty()) fail("'eos_token_id' is an empty list");
        c.eos_token_id = as_int(e.at(0), "eos_token_id");
      } else {
        c.eos_token_id = as_int(e, "eos_token_id");
      }
    }

    // Layer pattern: GDN everywhere except every 4th layer (QSA).
    {
      const nlohmann::json& lts = req("layer_types");
      if (!lts.is_array()) fail("'layer_types' is not a list");
      for (const auto& lt : lts) {
        const std::string s = lt.is_string() ? lt.get<std::string>() : lt.dump();
        if (s == "linear_attention") {
          c.layer_types.push_back(LayerType::kGdn);
        } else if (s == "full_attention" || s == "indexed_attention") {
          c.layer_types.push_back(LayerType::kQsa);
        } else {
          fail("unknown layer type '" + s + "' (want linear_attention, full_attention or indexed_attention)");
        }
      }
    }
    if (static_cast<int64_t>(c.layer_types.size()) != c.num_hidden_layers)
      fail("layer_types length " + std::to_string(c.layer_types.size()) + " does not match num_hidden_layers " +
           std::to_string(c.num_hidden_layers));
    for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
      const bool want_qsa = (i + 1) % kFullAttentionInterval == 0;
      if ((c.layer_types[static_cast<size_t>(i)] == LayerType::kQsa) != want_qsa)
        fail("layer_types[" + std::to_string(i) + "] breaks the 3 linear_attention : 1 full_attention pattern " +
             "(QSA at 3, 7, ...; want " + (want_qsa ? "full_attention" : "linear_attention") + ")");
    }

    // GDN.
    c.linear_num_key_heads = positive("linear_num_key_heads");
    c.linear_num_value_heads = positive("linear_num_value_heads");
    c.linear_key_head_dim = positive("linear_key_head_dim");
    c.linear_value_head_dim = positive("linear_value_head_dim");
    c.linear_conv_kernel_dim = positive("linear_conv_kernel_dim");
    if (c.linear_num_value_heads % c.linear_num_key_heads != 0)
      fail("linear_num_value_heads is not a multiple of linear_num_key_heads");
    {
      const char* key = t.contains("output_gate_type") && !t.at("output_gate_type").is_null() ? "output_gate_type"
                                                                                               : "hidden_act";
      const nlohmann::json& g = req(key);
      const std::string s = g.is_string() ? g.get<std::string>() : g.dump();
      if (s == "sigmoid") {
        c.output_gate_type = OutputGate::kSigmoid;
      } else if (s == "silu") {
        c.output_gate_type = OutputGate::kSilu;
      } else {
        fail("unsupported output_gate_type '" + s + "' (from '" + key + "'; want sigmoid or silu)");
      }
    }

    // QSA attention + rope.
    c.num_attention_heads = positive("num_attention_heads");
    c.num_key_value_heads = positive("num_key_value_heads");
    c.head_dim = positive("head_dim");
    if (c.num_attention_heads % c.num_key_value_heads != 0)
      fail("num_attention_heads is not a multiple of num_key_value_heads");
    {
      const nlohmann::json& rp = req("rope_parameters");
      if (!rp.is_object()) fail("'rope_parameters' is not an object");
      const std::string where = "rope_parameters";
      if (rp.contains("rope_type") && !StrEq(rp.at("rope_type"), "default"))
        fail("rope_parameters.rope_type " + rp.at("rope_type").dump() + " (want \"default\")");
      c.rope_theta = as_num(req_in(rp, where, "rope_theta"), "rope_theta");
      // HF reads partial_rotary_factor from rope_parameters; a top-level copy must agree.
      c.partial_rotary_factor = as_num(req_in(rp, where, "partial_rotary_factor"), "partial_rotary_factor");
      if (t.contains("partial_rotary_factor") && !t.at("partial_rotary_factor").is_null() &&
          as_num(t.at("partial_rotary_factor"), "partial_rotary_factor") != c.partial_rotary_factor)
        fail("text_config.partial_rotary_factor disagrees with rope_parameters.partial_rotary_factor");
      const nlohmann::json& ms = req_in(rp, where, "mrope_section");
      if (!ms.is_array()) fail("rope_parameters.mrope_section is not a list");
      for (const auto& s : ms) c.mrope_section.push_back(as_int(s, "mrope_section"));
      c.mrope_interleaved = as_bool(req_in(rp, where, "mrope_interleaved"), "mrope_interleaved");
    }
    if (!(c.rope_theta > 0.0)) fail("rope_theta must be > 0");
    if (!(c.partial_rotary_factor > 0.0 && c.partial_rotary_factor <= 1.0))
      fail("partial_rotary_factor must be in (0, 1]");
    {
      const double rot = c.partial_rotary_factor * static_cast<double>(c.head_dim);
      if (rot != static_cast<double>(static_cast<int64_t>(rot)) || static_cast<int64_t>(rot) % 2 != 0)
        fail("partial_rotary_factor * head_dim is not an even integer");
      int64_t sections = 0;
      for (int64_t s : c.mrope_section) sections += s;
      if (2 * sections != static_cast<int64_t>(rot))
        fail("mrope_section sums to " + std::to_string(sections) + " but the rotary width needs " +
             std::to_string(static_cast<int64_t>(rot) / 2) + " frequency pairs");
    }

    // MoE.
    c.num_experts = positive("num_experts");
    c.num_experts_per_tok = positive("num_experts_per_tok");
    if (c.num_experts_per_tok > c.num_experts) fail("num_experts_per_tok exceeds num_experts");
    c.moe_intermediate_size = positive("moe_intermediate_size");
    c.shared_expert_intermediate_size = positive("shared_expert_intermediate_size");
    if (t.contains("norm_topk_prob") && !t.at("norm_topk_prob").is_null())
      c.norm_topk_prob = as_bool(t.at("norm_topk_prob"), "norm_topk_prob");

    // Hyper-connections.
    c.hc_count = positive("hc_count");
    if (c.hc_count != 4) fail("hc_count must be 4 (the 4-stream gated residual), got " + std::to_string(c.hc_count));
    c.hc_lowrank = positive("hc_lowrank");

    // PLE + n-gram embedding.
    {
      const nlohmann::json& ids = req("ple_layer_ids");
      if (!ids.is_array()) fail("'ple_layer_ids' is not a list");
      for (const auto& v : ids) {
        const int64_t id = as_int(v, "ple_layer_ids");
        if (id < 1 || id > c.num_hidden_layers)
          fail("ple_layer_ids entry " + std::to_string(id) + " is outside the 1-indexed range [1, " +
               std::to_string(c.num_hidden_layers) + "]");
        if (c.layer_types[static_cast<size_t>(id - 1)] != LayerType::kGdn)
          fail("ple_layer_ids entry " + std::to_string(id) + " is not a linear_attention layer");
        bool dup = false;
        for (int64_t p : c.ple_layer_ids) dup = dup || p == id;
        if (dup) fail("ple_layer_ids lists " + std::to_string(id) + " twice");
        if (!c.ple_layer_ids.empty() && id < c.ple_layer_ids.back()) fail("ple_layer_ids is not sorted");
        c.ple_layer_ids.push_back(id);
        c.ple_layer_indices.push_back(id - 1);
      }
    }
    c.ple_embed_dim = positive("ple_embed_dim");
    c.ple_conv_kernel_size = positive("ple_conv_kernel_size");
    c.ngram_size = positive("ngram_size");
    if (c.ngram_size != 3) fail("ngram_size must be 3, got " + std::to_string(c.ngram_size));
    c.heads_per_ngram = positive("heads_per_ngram");
    if (c.ple_embed_dim % ((c.ngram_size - 1) * c.heads_per_ngram) != 0)
      fail("ple_embed_dim " + std::to_string(c.ple_embed_dim) + " is not divisible by (ngram_size - 1) * heads_per_ngram = " +
           std::to_string((c.ngram_size - 1) * c.heads_per_ngram));
    c.ngram_vocab_size_base = positive("ngram_vocab_size_base");
    c.make_ngram_vocab_size_divisible_by = positive("make_ngram_vocab_size_divisible_by");
    if (t.contains("seed") && !t.at("seed").is_null()) c.seed = as_int(t.at("seed"), "seed");
    c.split_ngram_parts = positive("split_ngram_parts");

    // QSA indexer.
    c.indexer_n_heads = positive("indexer_n_heads");
    c.indexer_kv_heads = positive("indexer_kv_heads");
    if (c.indexer_kv_heads != 1) fail("indexer_kv_heads must be 1, got " + std::to_string(c.indexer_kv_heads));
    c.indexer_head_dim = positive("indexer_head_dim");
    c.indexer_budget = positive("indexer_budget");
    c.indexer_compress_ratio = positive("indexer_compress_ratio");
    if (c.indexer_budget % c.indexer_compress_ratio != 0)
      fail("indexer_budget is not divisible by indexer_compress_ratio");
    if (c.RotaryDim() > c.indexer_head_dim)
      fail("the attention rotary width " + std::to_string(c.RotaryDim()) + " exceeds indexer_head_dim " +
           std::to_string(c.indexer_head_dim));
    return c;
  }
};

}  // namespace r4dx::model
