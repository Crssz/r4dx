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

  // ---- tensor parallel (docs/tp.md section 3.1) ----------------------------------------------
  // A config is either GLOBAL (tp_world == 1: the model as the container describes it) or the
  // rank-local view Shard() returns (tp_world > 1): the head counts and intermediate_size of that
  // rank's column-/row-parallel slices, everything else unchanged. vocab_size deliberately stays
  // GLOBAL in both (the embedding table, the sampler and every token-id range check need the full
  // vocabulary); the rank's lm_head width is VocabShardSize().
  int tp_world = 1;
  int tp_rank = 0;
  int64_t VocabShardSize() const { return vocab_size / tp_world; }
  int64_t VocabShardBegin() const { return tp_rank * VocabShardSize(); }
  bool IsShard() const { return tp_world > 1; }

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

  // mrope_section as the three ints r4dx_rope_partial_mrope3_bf16 takes, defaulting to the even
  // split-across-three-streams shape a config without the key implies. Throws when the config
  // carries a section vector that does not describe this model's rotary width -- a mismatch there
  // silently misassigns frequency bins to position streams, which is exactly the class of bug
  // docs/vision.md's "Text-side splicing" section warns produces plausible-looking garbage.
  void MropeSections(int* sec_t, int* sec_h, int* sec_w) const {
    const int64_t bins = RotaryDim() / 2;
    if (mrope_section.empty()) {
      *sec_h = static_cast<int>(bins / 3);
      *sec_w = static_cast<int>(bins / 3);
      *sec_t = static_cast<int>(bins - *sec_h - *sec_w);
      return;
    }
    if (mrope_section.size() != 3) {
      throw std::runtime_error("ModelConfig::MropeSections: mrope_section must have 3 entries, got " +
                                std::to_string(mrope_section.size()));
    }
    if (mrope_section[0] + mrope_section[1] + mrope_section[2] != bins) {
      throw std::runtime_error("ModelConfig::MropeSections: mrope_section must sum to rotary_dim/2 (" +
                                std::to_string(bins) + ")");
    }
    *sec_t = static_cast<int>(mrope_section[0]);
    *sec_h = static_cast<int>(mrope_section[1]);
    *sec_w = static_cast<int>(mrope_section[2]);
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

  // The rank-local config of `global` for tensor-parallel world `world`, rank `rank` (docs/tp.md
  // 3.1): num_attention_heads, num_key_value_heads, intermediate_size, linear_num_key_heads and
  // linear_num_value_heads are divided by `world`; tp_world/tp_rank are set; every other field
  // (including vocab_size) is copied. Throws std::invalid_argument naming the failing field unless
  // every shape the rank's kernels will see is legal:
  //   world in {1,2}; 0 <= rank < world; global.tp_world == 1 (no double sharding);
  //   !tie_word_embeddings (lm_head is vocab-split, the embedding table is replicated);
  //   the four head counts and intermediate_size divide by `world`; AttnGqa()/GqaRepeats()
  //   unchanged (the contiguous-head-block mapping of docs/tp.md 4.2 depends on it);
  //   every row-parallel rank K % 512 == 0 (attn.o, gdn.out_proj, mlp.down -- FallbackTuning's
  //   requirement, linear.cpp; it also implies every quant group and packed block divides it);
  //   every column-parallel rank segment % 16 == 0 (one 16-row tile never straddles two ranks);
  //   vocab_size % (16 * world) == 0 (the lm_head shard is whole tiles).
  // Every check applies at world 1 too, so Shard() accepts exactly the configs TP can serve. That
  // is why the TP=1 load path (tp_world == 1, docs/tp.md 5.1 step 7) must NOT call it: it keeps
  // the parsed config as is, and still loads e.g. a tied-embedding container Shard would refuse.
  static ModelConfig Shard(const ModelConfig& global, int world, int rank) {
    const std::string where = "ModelConfig::Shard(world=" + std::to_string(world) +
                              ", rank=" + std::to_string(rank) + "): ";
    const auto fail = [&](const std::string& what) { throw std::invalid_argument(where + what); };
    const auto require_div = [&](const char* field, int64_t value, int64_t divisor) {
      if (divisor <= 0 || value % divisor != 0) {
        fail(std::string(field) + " = " + std::to_string(value) + " is not divisible by " +
             std::to_string(divisor));
      }
    };

    if (world != 1 && world != 2) fail("world must be 1 or 2");
    if (rank < 0 || rank >= world) fail("rank must be in [0, world)");
    if (global.tp_world != 1) {
      fail("tp_world = " + std::to_string(global.tp_world) +
           " -- `global` is already a rank shard (no double sharding)");
    }
    if (global.tie_word_embeddings) {
      fail("tie_word_embeddings = true is not supported (lm_head is vocab-split while "
           "text.embed_tokens is replicated)");
    }

    require_div("num_attention_heads", global.num_attention_heads, world);
    require_div("num_key_value_heads", global.num_key_value_heads, world);
    require_div("linear_num_key_heads", global.linear_num_key_heads, world);
    require_div("linear_num_value_heads", global.linear_num_value_heads, world);
    require_div("intermediate_size", global.intermediate_size, world);

    ModelConfig r = global;
    r.tp_world = world;
    r.tp_rank = rank;
    r.num_attention_heads = global.num_attention_heads / world;
    r.num_key_value_heads = global.num_key_value_heads / world;
    r.linear_num_key_heads = global.linear_num_key_heads / world;
    r.linear_num_value_heads = global.linear_num_value_heads / world;
    r.intermediate_size = global.intermediate_size / world;

    if (r.AttnGqa() != global.AttnGqa()) {
      fail("AttnGqa() changes from " + std::to_string(global.AttnGqa()) + " to " +
           std::to_string(r.AttnGqa()) + " (num_attention_heads / num_key_value_heads)");
    }
    if (r.GqaRepeats() != global.GqaRepeats()) {
      fail("GqaRepeats() changes from " + std::to_string(global.GqaRepeats()) + " to " +
           std::to_string(r.GqaRepeats()) + " (linear_num_value_heads / linear_num_key_heads)");
    }

    // Row-parallel K per rank.
    require_div("attn.o rank K (num_attention_heads/world * head_dim)",
                r.num_attention_heads * r.head_dim, 512);
    require_div("gdn.out_proj rank K (ValueDim()/world)", r.ValueDim(), 512);
    require_div("mlp.down rank K (intermediate_size/world)", r.intermediate_size, 512);

    // Column-parallel rank segments.
    require_div("gdn.in_proj_qkv q/k rank segment (KeyDim()/world)", r.KeyDim(), 16);
    require_div("gdn.in_proj_qkv v / in_proj_z rank segment (ValueDim()/world)", r.ValueDim(),
                16);
    require_div("attn.qg rank rows (2 * num_attention_heads/world * head_dim)",
                2 * r.num_attention_heads * r.head_dim, 16);
    require_div("attn.k/v rank rows (num_key_value_heads/world * head_dim)",
                r.num_key_value_heads * r.head_dim, 16);
    require_div("mlp.gate_up rank segment (intermediate_size/world)", r.intermediate_size, 16);

    require_div("vocab_size (lm_head rank shard in whole 16-row tiles)", global.vocab_size,
                16 * static_cast<int64_t>(world));
    return r;
  }
};

}  // namespace r4dx::model
