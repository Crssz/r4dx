// r4dx::model::LlamaConfig -- the text-model geometry of a dense Llama (`model_type` "llama", LlamaForCausalLM;
// first user: Nanbeige4.3-33B), parsed STRICTLY from a container's `__metadata__.model_config` (the verbatim, flat
// config.json) (docs/nanbeige-plan.md 3.1, task N1-2).
//
// Header-only, HIP-free (like lfm2_config.h): tests/model/test_llama_config.cpp runs it on the real config without a
// device.
//
// The stack: `num_hidden_layers` identical layers, each `h += attn(input_layernorm(h)); h += mlp(post_attention_layernorm(h))`,
// full causal GQA attention (no bias, no QK norm, rotate-half rope over all of head_dim, no scaling) and a SwiGLU MLP
// (no bias); final `norm`, UNTIED lm_head. Everything the runtime does not implement is refused here rather than loaded
// wrongly: rope scaling, biases, a tied head, an activation other than silu, pretraining_tp > 1.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_config.h"
#include "nlohmann/json.hpp"

namespace r4dx::model {

struct LlamaConfig {
  // The context a run gets without --max-ctx. Not the checkpoint's max_position_embeddings (Nanbeige: 524288): KV is
  // 208 KiB per token at bf16, so the default is what both shipping formats hold (docs/nanbeige-plan.md 2.2, open
  // question 1).
  static constexpr int64_t kDefaultMaxCtx = 32768;
  static constexpr const char* kFull = "full_attention";

  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;

  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;  // `head_dim` when the config has it (Nanbeige: 128), else hidden_size / num_attention_heads

  int64_t intermediate_size = 0;

  int64_t vocab_size = 0;
  int64_t max_position_embeddings = 0;
  double rms_norm_eps = 1e-5;
  double rope_theta = 10000.0;  // no scaling, full rotary over head_dim
  bool tie_word_embeddings = false;

  int64_t bos_token_id = -1;
  int64_t eos_token_id = -1;
  int64_t pad_token_id = -1;

  // ---- tensor parallel -------------------------------------------------------------------------------------
  // GLOBAL (tp_world == 1) or the rank-local view Shard() returns: heads, kv heads and intermediate_size divided by
  // tp_world; vocab_size stays GLOBAL (VocabShardSize() is the rank's lm_head slice); hidden_size stays (the residual is
  // replicated).
  int tp_world = 1;
  int tp_rank = 0;
  int64_t VocabShardSize() const { return vocab_size / tp_world; }
  int64_t VocabShardBegin() const { return tp_rank * VocabShardSize(); }
  bool IsShard() const { return tp_world > 1; }

  // ---- derived shapes ----------------------------------------------------------------------------------------
  int64_t Gqa() const { return num_attention_heads / num_key_value_heads; }
  int64_t QDim() const { return num_attention_heads * head_dim; }   // q_proj rows == o_proj K
  int64_t KvDim() const { return num_key_value_heads * head_dim; }  // k_proj / v_proj rows
  int64_t GateUpRows() const { return 2 * intermediate_size; }      // gate rows first, then up
  // Every layer is full attention; ModelConfig::layer_types carries it for the generic consumers.
  std::vector<std::string> LayerTypes() const {
    return std::vector<std::string>(static_cast<size_t>(num_hidden_layers), kFull);
  }

  // The context a run gets: `requested` <= 0 -> min(kDefaultMaxCtx, max_position_embeddings); a value above the
  // checkpoint's max_position_embeddings throws (no rope scaling exists beyond it).
  int64_t ResolveMaxCtx(int64_t requested) const {
    const int64_t native = max_position_embeddings;
    if (requested <= 0) return std::min(kDefaultMaxCtx, native);
    if (requested <= native) return requested;
    throw std::invalid_argument("LlamaConfig::ResolveMaxCtx: " + std::to_string(requested) +
                                " exceeds the checkpoint's max_position_embeddings (" + std::to_string(native) +
                                "); the rope has no scaling, so there is no extended context");
  }

  // The thin generic view TextModel::Config() hands the CLI and server and tp::RuleFor reads: GDN and MoE fields 0,
  // `arch` says Llama.
  ModelConfig ToModelConfig() const {
    ModelConfig m;
    m.arch = Arch::kLlama;
    m.hidden_size = hidden_size;
    m.num_hidden_layers = num_hidden_layers;
    m.layer_types = LayerTypes();
    m.num_attention_heads = num_attention_heads;
    m.num_key_value_heads = num_key_value_heads;
    m.head_dim = head_dim;
    m.global_head_dim = 0;
    m.attn_output_gate = false;
    m.intermediate_size = intermediate_size;
    m.moe_intermediate_size = 0;
    m.num_experts = 0;
    m.num_dense_layers = num_hidden_layers;
    m.partial_rotary_factor = 1.0;
    m.rope_theta = rope_theta;
    m.mrope_interleaved = false;
    m.mrope_section.clear();
    m.linear_num_key_heads = m.linear_num_value_heads = 0;
    m.linear_key_head_dim = m.linear_value_head_dim = 0;
    m.linear_conv_kernel_dim = 0;
    m.rms_norm_eps = rms_norm_eps;
    m.vocab_size = vocab_size;
    m.tie_word_embeddings = tie_word_embeddings;
    m.mtp_num_hidden_layers = 0;
    m.tp_world = tp_world;
    m.tp_rank = tp_rank;
    return m;
  }

  // `model_config` is the container's verbatim config.json (flat); an object holding "text_config" is accepted too.
  static LlamaConfig FromModelConfig(const nlohmann::json& model_config) {
    if (model_config.is_object() && model_config.contains("text_config"))
      return FromJson(model_config.at("text_config"));
    return FromJson(model_config);
  }

  // `cfg` is config.json. Every field the geometry depends on is required; unsupported variants throw. Both rope
  // spellings are read: transformers 4.x `rope_theta` + `rope_scaling` (Nanbeige, saved by 4.51.1) and transformers 5
  // `rope_parameters {rope_type, rope_theta}`.
  static LlamaConfig FromJson(const nlohmann::json& cfg) {
    const std::string who = "LlamaConfig::FromJson: ";
    if (!cfg.is_object()) throw std::runtime_error(who + "config is not an object");
    auto req = [&](const char* key) -> const nlohmann::json& {
      if (!cfg.contains(key)) throw std::runtime_error(who + "config missing '" + key + "'");
      return cfg.at(key);
    };
    auto req_int = [&](const char* key) {
      const nlohmann::json& v = req(key);
      if (!v.is_number_integer()) throw std::runtime_error(who + "'" + key + "' is not an integer");
      return v.get<int64_t>();
    };
    auto positive = [&](const char* key) {
      const int64_t v = req_int(key);
      if (v <= 0) throw std::runtime_error(who + "'" + key + "' must be > 0, got " + std::to_string(v));
      return v;
    };
    auto refuse_if = [&](bool bad, const std::string& why) {
      if (bad) throw std::runtime_error(who + "unsupported: " + why);
    };
    auto flag = [&](const char* key) {  // an absent or null boolean key is false
      return cfg.contains(key) && cfg.at(key).is_boolean() && cfg.at(key).get<bool>();
    };

    if (cfg.contains("model_type")) {
      const std::string mt = cfg.at("model_type").get<std::string>();
      refuse_if(mt != "llama", "model_type '" + mt + "' (want llama)");
    }

    LlamaConfig c;
    c.hidden_size = positive("hidden_size");
    c.num_hidden_layers = positive("num_hidden_layers");

    c.num_attention_heads = positive("num_attention_heads");
    c.num_key_value_heads = positive("num_key_value_heads");
    if (c.num_attention_heads % c.num_key_value_heads != 0)
      throw std::runtime_error(who + "num_attention_heads is not a multiple of num_key_value_heads");
    if (cfg.contains("head_dim") && !cfg.at("head_dim").is_null()) {
      c.head_dim = positive("head_dim");
    } else {
      if (c.hidden_size % c.num_attention_heads != 0)
        throw std::runtime_error(who + "hidden_size is not a multiple of num_attention_heads (no head_dim key)");
      c.head_dim = c.hidden_size / c.num_attention_heads;
    }
    if (c.head_dim % 2 != 0) throw std::runtime_error(who + "head_dim must be even (rotate-half)");

    c.intermediate_size = positive("intermediate_size");
    const std::string act = cfg.contains("hidden_act") ? cfg.at("hidden_act").get<std::string>() : std::string("silu");
    refuse_if(act != "silu", "hidden_act '" + act + "' (want silu)");
    refuse_if(flag("attention_bias"), "attention_bias=true (the attention linears have no bias)");
    refuse_if(flag("mlp_bias"), "mlp_bias=true (the MLP linears have no bias)");
    if (cfg.contains("pretraining_tp") && cfg.at("pretraining_tp").is_number_integer())
      refuse_if(cfg.at("pretraining_tp").get<int64_t>() != 1, "pretraining_tp != 1");

    c.vocab_size = positive("vocab_size");
    c.max_position_embeddings = positive("max_position_embeddings");
    c.rms_norm_eps = req("rms_norm_eps").get<double>();
    c.tie_word_embeddings = flag("tie_word_embeddings");
    refuse_if(c.tie_word_embeddings, "tie_word_embeddings=true (the runtime reads lm_head as its own tensor)");

    if (cfg.contains("rope_parameters") && cfg.at("rope_parameters").is_object()) {
      const nlohmann::json& rp = cfg.at("rope_parameters");
      refuse_if(rp.value("rope_type", std::string("default")) != "default",
                "rope_type '" + rp.value("rope_type", std::string()) + "' (want default: no scaling)");
      if (!rp.contains("rope_theta")) throw std::runtime_error(who + "rope_parameters missing 'rope_theta'");
      c.rope_theta = rp.at("rope_theta").get<double>();
    } else {
      c.rope_theta = req("rope_theta").get<double>();
    }
    refuse_if(cfg.contains("rope_scaling") && !cfg.at("rope_scaling").is_null(), "rope_scaling");
    refuse_if(c.rope_theta <= 0.0, "rope_theta <= 0");

    auto opt_id = [&](const char* key, int64_t* out) {
      if (cfg.contains(key) && cfg.at(key).is_number_integer()) *out = cfg.at(key).get<int64_t>();
    };
    opt_id("bos_token_id", &c.bos_token_id);
    opt_id("eos_token_id", &c.eos_token_id);
    opt_id("pad_token_id", &c.pad_token_id);
    return c;
  }

  // The rank-local config of `global` for TP world `world`, rank `rank` (docs/nanbeige-plan.md 3.4). Throws
  // std::invalid_argument naming the failing field unless every shape a rank's kernels see is legal:
  //   world in {1,2}; 0 <= rank < world; global.tp_world == 1 (no double sharding);
  //   heads, kv heads and intermediate_size divide by world; the GQA is unchanged (Nanbeige: 8 kv heads -> 4, still 7);
  //   row-parallel rank K % 256 == 0 (FallbackTuning's SK 4 x 64 rule, linear.cpp): attn.o (heads/world * head_dim),
  //   mlp.down (intermediate_size/world);
  //   column-parallel rank rows % 16 == 0: q, k/v, each gate_up segment;
  //   vocab_size % (16 * world) == 0 (the lm_head shard is whole 16-row tiles).
  static LlamaConfig Shard(const LlamaConfig& global, int world, int rank) {
    const std::string where = "LlamaConfig::Shard(world=" + std::to_string(world) + ", rank=" + std::to_string(rank) + "): ";
    const auto fail = [&](const std::string& what) { throw std::invalid_argument(where + what); };
    const auto require_div = [&](const char* field, int64_t value, int64_t divisor) {
      if (divisor <= 0 || value % divisor != 0)
        fail(std::string(field) + " = " + std::to_string(value) + " is not divisible by " + std::to_string(divisor));
    };
    if (world != 1 && world != 2) fail("world must be 1 or 2");
    if (rank < 0 || rank >= world) fail("rank must be in [0, world)");
    if (global.tp_world != 1) fail("tp_world = " + std::to_string(global.tp_world) + " -- `global` is already a rank shard");

    require_div("num_attention_heads", global.num_attention_heads, world);
    require_div("num_key_value_heads", global.num_key_value_heads, world);
    require_div("intermediate_size", global.intermediate_size, world);

    LlamaConfig r = global;
    r.tp_world = world;
    r.tp_rank = rank;
    r.num_attention_heads = global.num_attention_heads / world;
    r.num_key_value_heads = global.num_key_value_heads / world;
    r.intermediate_size = global.intermediate_size / world;

    if (r.Gqa() != global.Gqa())
      fail("the GQA changes from " + std::to_string(global.Gqa()) + " to " + std::to_string(r.Gqa()));

    require_div("attn.o rank K (heads/world * head_dim)", r.QDim(), 256);
    require_div("mlp.down rank K (intermediate_size/world)", r.intermediate_size, 256);

    require_div("attn.q rank rows (heads/world * head_dim)", r.QDim(), 16);
    require_div("attn.k/v rank rows (kv/world * head_dim)", r.KvDim(), 16);
    require_div("mlp.gate_up rank segment (intermediate_size/world)", r.intermediate_size, 16);
    require_div("vocab_size (lm_head rank shard in whole 16-row tiles)", global.vocab_size, 16 * static_cast<int64_t>(world));
    return r;
  }
};

}  // namespace r4dx::model
