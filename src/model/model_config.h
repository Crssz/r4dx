// r4dx::model::ModelConfig -- the subset of Qwen3_5's text_config the layer graph needs, parsed
// from a container's `__metadata__.model_config.text_config` (docs/container-format.md:
// `model_config` is a verbatim copy of the source HF config.json, so the container-format's
// top-level object nests the fields this struct reads under "text_config", exactly as
// C:\AI\models\Qwen3.8-27B\config.json does).
//
// This is a plain data holder plus a handful of derived shape helpers (KeyDim/ValueDim/ConvDim/
// RotaryDim) that every consumer (container loader, linear, GDN layer, MLP) would otherwise
// recompute from the same three or four raw fields -- keeping the arithmetic here means the whole
// model only ever states "key_dim = num_key_heads * key_head_dim" once.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace r4dx::model {

struct ModelConfig {
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  std::vector<std::string> layer_types;  // "linear_attention" | "full_attention", per layer

  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  bool attn_output_gate = true;

  int64_t intermediate_size = 0;

  double partial_rotary_factor = 0.25;
  double rope_theta = 1.0e7;
  bool mrope_interleaved = true;
  std::vector<int64_t> mrope_section;  // e.g. [11, 11, 10]

  int64_t linear_num_key_heads = 0;
  int64_t linear_num_value_heads = 0;
  int64_t linear_key_head_dim = 0;
  int64_t linear_value_head_dim = 0;
  int64_t linear_conv_kernel_dim = 4;

  double rms_norm_eps = 1e-6;
  int64_t vocab_size = 0;
  bool tie_word_embeddings = false;
  int64_t mtp_num_hidden_layers = 1;

  // ---- derived shapes, named exactly as docs/architecture.md / docs/container-format.md do ----
  int64_t KeyDim() const { return linear_num_key_heads * linear_key_head_dim; }
  int64_t ValueDim() const { return linear_num_value_heads * linear_value_head_dim; }
  int64_t ConvDim() const { return 2 * KeyDim() + ValueDim(); }
  int64_t RotaryDim() const {
    return static_cast<int64_t>(static_cast<double>(head_dim) * partial_rotary_factor + 0.5);
  }
  int64_t GqaRepeats() const {
    if (linear_num_key_heads == 0) return 1;
    return linear_num_value_heads / linear_num_key_heads;
  }
  int64_t AttnGqa() const {
    if (num_key_value_heads == 0) return 1;
    return num_attention_heads / num_key_value_heads;
  }

  bool IsGdnLayer(int64_t layer_idx) const {
    if (layer_idx < 0 || layer_idx >= static_cast<int64_t>(layer_types.size())) {
      throw std::out_of_range("ModelConfig::IsGdnLayer: layer_idx out of range");
    }
    return layer_types[static_cast<size_t>(layer_idx)] == "linear_attention";
  }

  // `text_cfg` is the JSON object at model_config["text_config"] (see file comment). Every field
  // is required except the ones with an obviously-safe default already set above (attn_output_gate,
  // mrope_interleaved, tie_word_embeddings, linear_conv_kernel_dim, mtp_num_hidden_layers) -- a
  // missing REQUIRED field fails loudly rather than silently defaulting to 0 and producing shapes
  // that don't match the container's actual tensors.
  static ModelConfig FromJson(const nlohmann::json& text_cfg) {
    ModelConfig c;
    auto req = [&](const char* key) -> const nlohmann::json& {
      if (!text_cfg.contains(key)) {
        throw std::runtime_error(std::string("ModelConfig::FromJson: text_config missing '") +
                                  key + "'");
      }
      return text_cfg.at(key);
    };
    c.hidden_size = req("hidden_size").get<int64_t>();
    c.num_hidden_layers = req("num_hidden_layers").get<int64_t>();
    for (const auto& lt : req("layer_types")) c.layer_types.push_back(lt.get<std::string>());
    if (static_cast<int64_t>(c.layer_types.size()) != c.num_hidden_layers) {
      throw std::runtime_error(
          "ModelConfig::FromJson: layer_types length does not match num_hidden_layers");
    }

    c.num_attention_heads = req("num_attention_heads").get<int64_t>();
    c.num_key_value_heads = req("num_key_value_heads").get<int64_t>();
    c.head_dim = req("head_dim").get<int64_t>();
    if (text_cfg.contains("attn_output_gate")) c.attn_output_gate = text_cfg.at("attn_output_gate").get<bool>();

    c.intermediate_size = req("intermediate_size").get<int64_t>();

    // rope_theta/partial_rotary_factor/mrope_* live under "rope_parameters" in the real
    // config.json; fall back to a top-level key (older/alternate config shapes) if present.
    if (text_cfg.contains("rope_parameters")) {
      const auto& rp = text_cfg.at("rope_parameters");
      if (rp.contains("rope_theta")) c.rope_theta = rp.at("rope_theta").get<double>();
      if (rp.contains("partial_rotary_factor"))
        c.partial_rotary_factor = rp.at("partial_rotary_factor").get<double>();
      if (rp.contains("mrope_interleaved")) c.mrope_interleaved = rp.at("mrope_interleaved").get<bool>();
      if (rp.contains("mrope_section")) {
        for (const auto& s : rp.at("mrope_section")) c.mrope_section.push_back(s.get<int64_t>());
      }
    } else {
      if (text_cfg.contains("rope_theta")) c.rope_theta = text_cfg.at("rope_theta").get<double>();
      if (text_cfg.contains("partial_rotary_factor"))
        c.partial_rotary_factor = text_cfg.at("partial_rotary_factor").get<double>();
    }
    if (text_cfg.contains("partial_rotary_factor"))  // top-level also carries a copy in this checkpoint
      c.partial_rotary_factor = text_cfg.at("partial_rotary_factor").get<double>();

    c.linear_num_key_heads = req("linear_num_key_heads").get<int64_t>();
    c.linear_num_value_heads = req("linear_num_value_heads").get<int64_t>();
    c.linear_key_head_dim = req("linear_key_head_dim").get<int64_t>();
    c.linear_value_head_dim = req("linear_value_head_dim").get<int64_t>();
    if (text_cfg.contains("linear_conv_kernel_dim"))
      c.linear_conv_kernel_dim = text_cfg.at("linear_conv_kernel_dim").get<int64_t>();

    c.rms_norm_eps = req("rms_norm_eps").get<double>();
    c.vocab_size = req("vocab_size").get<int64_t>();
    if (text_cfg.contains("tie_word_embeddings"))
      c.tie_word_embeddings = text_cfg.at("tie_word_embeddings").get<bool>();
    if (text_cfg.contains("mtp_num_hidden_layers"))
      c.mtp_num_hidden_layers = text_cfg.at("mtp_num_hidden_layers").get<int64_t>();

    return c;
  }
};

}  // namespace r4dx::model
