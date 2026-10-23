// r4dx::model::Qwen35MoeConfig -- the text-model geometry of `qwen3_5_moe` (Ornith-1.5-122B, the Qwen3.5
// hybrid with a sparse MoE FFN), parsed STRICTLY from a checkpoint's / container's config.json `text_config`
// (docs/qwen35moe-plan.md section 1).
//
// Same mixers as the Qwen3.5 27B (Gated DeltaNet and gated full attention, plain residual stream, input and
// post-attention RMSNorm) but every layer's FFN is a 256-expert top-8 MoE with a sigmoid-gated shared expert.
// ModelConfig::FromJson cannot parse it (it requires a dense `intermediate_size`), so this arch has its own
// parser; the 27B path is untouched.
//
// Header-only and HIP-free (like qwen4exp_config.h): tests/model/test_qwen35moe_config.cpp runs it on the real
// config without a device.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "model_config.h"

namespace r4dx::model {

struct Qwen35MoeConfig {
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t vocab_size = 0;
  double rms_norm_eps = 0.0;
  std::vector<std::string> layer_types;  // "linear_attention" | "full_attention"
  int64_t max_position_embeddings = 0;
  bool tie_word_embeddings = false;
  int64_t eos_token_id = 0;  // the first id when config.json carries a list

  // GDN
  int64_t linear_num_key_heads = 0;
  int64_t linear_num_value_heads = 0;
  int64_t linear_key_head_dim = 0;
  int64_t linear_value_head_dim = 0;
  int64_t linear_conv_kernel_dim = 0;

  // full attention
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  bool attn_output_gate = true;
  double partial_rotary_factor = 0.0;
  double rope_theta = 0.0;
  std::vector<int64_t> mrope_section;
  bool mrope_interleaved = false;

  // MoE
  int64_t num_experts = 0;
  int64_t num_experts_per_tok = 0;
  int64_t moe_intermediate_size = 0;
  int64_t shared_expert_intermediate_size = 0;
  bool norm_topk_prob = true;  // HF Qwen3.5-MoE normalises the top-k weights (VERIFY against the golden, plan M2)

  int64_t mtp_num_hidden_layers = 0;

  // ---- vision tower (config.json `vision_config`; 0 when absent) -------------------------------
  int64_t vision_depth = 0;
  int64_t vision_hidden_size = 0;
  int64_t vision_out_hidden_size = 0;

  // ---- tensor parallel (docs/qwen35moe-tp-design.md section 1.3) ---------------------------------
  // GLOBAL (tp_world == 1) or the rank-local view Shard() returns: attention heads / kv heads, GDN key / value heads,
  // moe_intermediate_size and shared_expert_intermediate_size divided by tp_world. vocab_size stays GLOBAL
  // (VocabShardSize() is the rank's lm_head slice); hidden_size, num_experts and num_experts_per_tok stay (the residual
  // and the router are replicated, every rank runs every expert at half width).
  int tp_world = 1;
  int tp_rank = 0;
  int64_t VocabShardSize() const { return vocab_size / tp_world; }
  int64_t VocabShardBegin() const { return tp_rank * VocabShardSize(); }
  bool IsShard() const { return tp_world > 1; }

  int64_t KeyDim() const { return linear_num_key_heads * linear_key_head_dim; }
  int64_t ValueDim() const { return linear_num_value_heads * linear_value_head_dim; }
  int64_t ConvDim() const { return 2 * KeyDim() + ValueDim(); }
  int64_t RotaryDim() const { return static_cast<int64_t>(static_cast<double>(head_dim) * partial_rotary_factor + 0.5); }
  int64_t AttnGqa() const { return num_key_value_heads == 0 ? 1 : num_attention_heads / num_key_value_heads; }
  int64_t GqaRepeats() const { return linear_num_key_heads == 0 ? 1 : linear_num_value_heads / linear_num_key_heads; }

  bool IsGdnLayer(int64_t i) const {
    if (i < 0 || i >= static_cast<int64_t>(layer_types.size()))
      throw std::out_of_range("Qwen35MoeConfig::IsGdnLayer: layer index out of range");
    return layer_types[static_cast<size_t>(i)] == "linear_attention";
  }
  int64_t NumGdnLayers() const {
    int64_t n = 0;
    for (const auto& t : layer_types) n += t == "linear_attention" ? 1 : 0;
    return n;
  }
  int64_t NumAttnLayers() const { return static_cast<int64_t>(layer_types.size()) - NumGdnLayers(); }

  // Routed expert parameters of the whole model (gate, up, down of every expert of every layer).
  int64_t RoutedExpertParams() const {
    return num_hidden_layers * num_experts * 3 * hidden_size * moe_intermediate_size;
  }

  // The generic view TextModel::Config() hands the CLI, server and tools. `arch` is kQwen35Moe; the
  // MoE fields reuse ModelConfig's LFM2-era `num_experts` / `moe_intermediate_size`; `intermediate_size` is the
  // routed experts' (this arch has no dense MLP); `mtp_num_hidden_layers` is 0 until the MTP head is wired.
  ModelConfig ToModelConfig() const {
    ModelConfig m;
    m.arch = Arch::kQwen35Moe;
    m.hidden_size = hidden_size;
    m.num_hidden_layers = num_hidden_layers;
    m.layer_types = layer_types;
    m.num_attention_heads = num_attention_heads;
    m.num_key_value_heads = num_key_value_heads;
    m.head_dim = head_dim;
    m.attn_output_gate = attn_output_gate;
    m.intermediate_size = moe_intermediate_size;
    m.moe_intermediate_size = moe_intermediate_size;
    m.num_experts = num_experts;
    m.shared_expert_intermediate_size = shared_expert_intermediate_size;
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

  // `config` is the full config.json (top-level object holding "text_config", as the checkpoint and a
  // container's __metadata__.model_config carry it) or the bare text_config.
  static Qwen35MoeConfig FromJson(const nlohmann::json& config) {
    const std::string who = "Qwen35MoeConfig::FromJson: ";
    if (!config.is_object()) throw std::runtime_error(who + "config is not an object");
    if (config.contains("text_config")) {
      if (config.contains("model_type") && !StrEq(config.at("model_type"), "qwen3_5_moe"))
        throw std::runtime_error(who + "model_type " + config.at("model_type").dump() + " (want \"qwen3_5_moe\")");
      Qwen35MoeConfig c = FromTextConfig(config.at("text_config"));
      if (config.contains("vision_config") && config.at("vision_config").is_object()) {
        const auto& v = config.at("vision_config");
        if (v.contains("depth")) c.vision_depth = v.at("depth").get<int64_t>();
        if (v.contains("hidden_size")) c.vision_hidden_size = v.at("hidden_size").get<int64_t>();
        if (v.contains("out_hidden_size")) c.vision_out_hidden_size = v.at("out_hidden_size").get<int64_t>();
      }
      return c;
    }
    return FromTextConfig(config);
  }

  // The rank-local config of `global` for TP world `world`, rank `rank`. Throws std::invalid_argument naming the failing
  // field unless every shape a rank's kernels, loader and byte plans see is legal:
  //   world in {1,2}; 0 <= rank < world; global.tp_world == 1 (no double sharding);
  //   attention heads, kv heads, GDN key / value heads, moe_intermediate_size and shared_expert_intermediate_size divide by world;
  //   both GQA ratios are unchanged (attention 16 q per kv head, GDN 4 value heads per key head);
  //   trellis rank ranges are whole 128-blocks (docs/trellis-kernel.md 2.4): every column-parallel segment's rows (the
  //   three in_proj_qkv segments, in_proj_z, qg, k / v, an expert's gate and up) and every row-parallel K (out_proj, attn.o,
  //   an expert's down);
  //   row-parallel rank K % 256 == 0 (the relaxed FallbackTuning rule: SK 4 x 64);
  //   the bf16 shared expert: rank rows % 16 and K % 16;
  //   vocab_size % (16 * world) == 0 (the lm_head slice is whole 16-row tiles).
  // Every check applies at world 1 too, so Shard() accepts exactly the configs TP can serve.
  static Qwen35MoeConfig Shard(const Qwen35MoeConfig& global, int world, int rank) {
    const std::string where = "Qwen35MoeConfig::Shard(world=" + std::to_string(world) + ", rank=" + std::to_string(rank) + "): ";
    const auto fail = [&](const std::string& what) { throw std::invalid_argument(where + what); };
    const auto require_div = [&](const std::string& field, int64_t value, int64_t divisor) {
      if (divisor <= 0 || value % divisor != 0)
        fail(field + " = " + std::to_string(value) + " is not divisible by " + std::to_string(divisor));
    };
    if (world != 1 && world != 2) fail("world must be 1 or 2");
    if (rank < 0 || rank >= world) fail("rank must be in [0, world)");
    if (global.tp_world != 1) fail("tp_world = " + std::to_string(global.tp_world) + " -- `global` is already a rank shard");

    require_div("num_attention_heads", global.num_attention_heads, world);
    require_div("num_key_value_heads", global.num_key_value_heads, world);
    require_div("linear_num_key_heads", global.linear_num_key_heads, world);
    require_div("linear_num_value_heads", global.linear_num_value_heads, world);
    require_div("moe_intermediate_size", global.moe_intermediate_size, world);
    require_div("shared_expert_intermediate_size", global.shared_expert_intermediate_size, world);

    Qwen35MoeConfig r = global;
    r.tp_world = world;
    r.tp_rank = rank;
    r.num_attention_heads = global.num_attention_heads / world;
    r.num_key_value_heads = global.num_key_value_heads / world;
    r.linear_num_key_heads = global.linear_num_key_heads / world;
    r.linear_num_value_heads = global.linear_num_value_heads / world;
    r.moe_intermediate_size = global.moe_intermediate_size / world;
    r.shared_expert_intermediate_size = global.shared_expert_intermediate_size / world;

    if (r.AttnGqa() != global.AttnGqa())
      fail("the attention GQA changes from " + std::to_string(global.AttnGqa()) + " to " + std::to_string(r.AttnGqa()));
    if (r.GqaRepeats() != global.GqaRepeats())
      fail("the GDN value-per-key head ratio changes from " + std::to_string(global.GqaRepeats()) + " to " +
           std::to_string(r.GqaRepeats()));

    const int64_t q_dim = r.num_attention_heads * r.head_dim, kv_dim = r.num_key_value_heads * r.head_dim;
    // Trellis: whole 128-blocks on every rank range (both Hadamards work in them).
    require_div("gdn.in_proj_qkv rank q / k segment (key heads/world * key_head_dim)", r.KeyDim(), 128);
    require_div("gdn.in_proj_qkv rank v segment and in_proj_z rows (value heads/world * value_head_dim)", r.ValueDim(), 128);
    require_div("gdn.out_proj rank K (value heads/world * value_head_dim)", r.ValueDim(), 128);
    require_div("attn.qg rank rows (2 * heads/world * head_dim)", 2 * q_dim, 128);
    require_div("attn.k/v rank rows (kv heads/world * head_dim)", kv_dim, 128);
    require_div("attn.o rank K (heads/world * head_dim)", q_dim, 128);
    require_div("expert gate_up rank segment (moe_intermediate_size/world)", r.moe_intermediate_size, 128);
    require_div("expert down rank K (moe_intermediate_size/world)", r.moe_intermediate_size, 128);
    // Row-parallel rank K % 256 (the relaxed FallbackTuning rule, linear.cpp: SK 4 x 64).
    require_div("gdn.out_proj rank K", r.ValueDim(), 256);
    require_div("attn.o rank K", q_dim, 256);
    require_div("expert down rank K", r.moe_intermediate_size, 256);
    // bf16 shared expert and the GDN conv channels.
    require_div("shared gate/up rank rows (shared_expert_intermediate_size/world)", r.shared_expert_intermediate_size, 16);
    require_div("shared down rank K (shared_expert_intermediate_size/world)", r.shared_expert_intermediate_size, 16);
    require_div("conv channels (2 * key dim + value dim)", r.ConvDim(), 16);
    require_div("vocab_size (lm_head rank shard in whole 16-row tiles)", global.vocab_size, 16 * static_cast<int64_t>(world));
    return r;
  }

 private:
  static bool StrEq(const nlohmann::json& v, const char* s) {
    return v.is_string() && v.get_ref<const std::string&>() == s;
  }

  static Qwen35MoeConfig FromTextConfig(const nlohmann::json& t) {
    const std::string who = "Qwen35MoeConfig::FromJson: ";
    if (!t.is_object()) throw std::runtime_error(who + "text_config is not an object");
    auto fail = [&](const std::string& why) { throw std::runtime_error(who + why); };
    auto req = [&](const char* key) -> const nlohmann::json& {
      if (!t.contains(key) || t.at(key).is_null()) fail(std::string("text_config missing '") + key + "'");
      return t.at(key);
    };
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

    if (t.contains("model_type") && !StrEq(t.at("model_type"), "qwen3_5_moe_text"))
      fail("text_config.model_type " + t.at("model_type").dump() + " (want \"qwen3_5_moe_text\")");

    Qwen35MoeConfig c;
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

    const nlohmann::json& lt = req("layer_types");
    if (!lt.is_array()) fail("'layer_types' is not an array");
    for (const auto& v : lt) {
      if (!v.is_string()) fail("'layer_types' holds a non-string: " + v.dump());
      const std::string s = v.get<std::string>();
      if (s != "linear_attention" && s != "full_attention")
        fail("unknown layer type '" + s + "' (want linear_attention | full_attention)");
      c.layer_types.push_back(s);
    }
    if (static_cast<int64_t>(c.layer_types.size()) != c.num_hidden_layers)
      fail("layer_types has " + std::to_string(c.layer_types.size()) + " entries, num_hidden_layers is " +
           std::to_string(c.num_hidden_layers));

    c.linear_num_key_heads = positive("linear_num_key_heads");
    c.linear_num_value_heads = positive("linear_num_value_heads");
    c.linear_key_head_dim = positive("linear_key_head_dim");
    c.linear_value_head_dim = positive("linear_value_head_dim");
    c.linear_conv_kernel_dim = positive("linear_conv_kernel_dim");
    if (c.linear_num_value_heads % c.linear_num_key_heads != 0)
      fail("linear_num_value_heads is not a multiple of linear_num_key_heads");

    c.num_attention_heads = positive("num_attention_heads");
    c.num_key_value_heads = positive("num_key_value_heads");
    if (c.num_attention_heads % c.num_key_value_heads != 0)
      fail("num_attention_heads is not a multiple of num_key_value_heads");
    c.head_dim = positive("head_dim");
    c.attn_output_gate = as_bool(req("attn_output_gate"), "attn_output_gate");

    // rope: rope_parameters, with the top-level partial_rotary_factor as the checkpoint also carries it
    if (!t.contains("rope_parameters") || !t.at("rope_parameters").is_object()) fail("text_config missing 'rope_parameters'");
    const nlohmann::json& rp = t.at("rope_parameters");
    auto rreq = [&](const char* key) -> const nlohmann::json& {
      if (!rp.contains(key) || rp.at(key).is_null()) fail(std::string("rope_parameters missing '") + key + "'");
      return rp.at(key);
    };
    c.rope_theta = as_num(rreq("rope_theta"), "rope_theta");
    c.partial_rotary_factor = as_num(rreq("partial_rotary_factor"), "partial_rotary_factor");
    if (t.contains("partial_rotary_factor") && as_num(t.at("partial_rotary_factor"), "partial_rotary_factor") != c.partial_rotary_factor)
      fail("partial_rotary_factor differs between text_config and rope_parameters");
    c.mrope_interleaved = as_bool(rreq("mrope_interleaved"), "mrope_interleaved");
    for (const auto& s : rreq("mrope_section")) c.mrope_section.push_back(as_int(s, "mrope_section"));
    if (c.mrope_section.size() != 3) fail("mrope_section must have 3 entries");
    {
      int64_t sum = 0;
      for (int64_t s : c.mrope_section) sum += s;
      if (sum != c.RotaryDim() / 2)
        fail("mrope_section sums to " + std::to_string(sum) + ", rotary_dim/2 is " + std::to_string(c.RotaryDim() / 2));
    }

    c.num_experts = positive("num_experts");
    c.num_experts_per_tok = positive("num_experts_per_tok");
    if (c.num_experts_per_tok > c.num_experts) fail("num_experts_per_tok exceeds num_experts");
    c.moe_intermediate_size = positive("moe_intermediate_size");
    c.shared_expert_intermediate_size = positive("shared_expert_intermediate_size");
    if (t.contains("norm_topk_prob")) c.norm_topk_prob = as_bool(t.at("norm_topk_prob"), "norm_topk_prob");
    if (t.contains("mlp_only_layers") && t.at("mlp_only_layers").is_array() && !t.at("mlp_only_layers").empty())
      fail("mlp_only_layers is not empty: dense layers inside the MoE stack are not supported");
    if (t.contains("mtp_num_hidden_layers")) c.mtp_num_hidden_layers = as_int(t.at("mtp_num_hidden_layers"), "mtp_num_hidden_layers");
    return c;
  }
};

}  // namespace r4dx::model
