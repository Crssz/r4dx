// r4dx::model::Lfm2Config -- the text-model geometry of LiquidAI's lfm2_moe (LFM2.5-24B-A2B), parsed STRICTLY
// from a container's `__metadata__.model_config` (the verbatim, FLAT config.json: there is no `text_config`)
// (docs/lfm2-plan.md 3.1, task L1-1).
//
// Header-only, HIP-free (like gemma_config.h): tests/model/test_lfm2_config.cpp runs it on the real config
// without a device.
//
// The stack: 40 layers, each `h += mixer(operator_norm(h)); h += ffn(ffn_norm(h))`, the mixer a ShortConv
// (layer_types "conv", 30 layers) or full attention ("full_attention", layers 2, 6, ..., 38), the FFN a dense
// SwiGLU for the first `num_dense_layers` layers and a 64-expert top-4 sigmoid-routed MoE after that. Final
// norm `embedding_norm`, tied head. Everything the runtime does not implement is refused here rather than
// loaded wrongly.
//
// Structure vs size: the structural invariants (layer pattern, dense-first, conv kernel 3, no conv bias,
// expert_bias, tied head, default rope) are asserted for EVERY config, the tiny converter fixture included; the
// sizes (2048 / 64 experts / ...) are parameters, pinned only by tests/model/test_lfm2_config.cpp on the real
// config.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_config.h"
#include "nlohmann/json.hpp"

namespace r4dx::model {

struct Lfm2Config {
  static constexpr int64_t kDefaultMaxCtx = 131072;  // the config's max_position_embeddings; no rope scaling exists
  static constexpr const char* kConv = "conv";
  static constexpr const char* kFull = "full_attention";
  static constexpr int64_t kAttnEvery = 4;  // attention at i % 4 == 2
  static constexpr int64_t kAttnPhase = 2;
  static constexpr int64_t kConvKernel = 3;  // conv_L_cache

  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  std::vector<std::string> layer_types;  // "conv" | "full_attention", per layer

  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;  // `head_dim` when the config has it, else hidden_size / num_attention_heads (VERIFIED: 64)

  int64_t intermediate_size = 0;      // the dense FFN (layers < num_dense_layers)
  int64_t moe_intermediate_size = 0;  // one expert
  int64_t num_dense_layers = 0;
  int64_t num_experts = 0;
  int64_t num_experts_per_tok = 0;
  double routed_scaling_factor = 1.0;
  bool norm_topk_prob = true;
  bool use_expert_bias = true;
  // The router runs in fp32 here (HF does it in bf16): a runtime decision, not a config key (docs/lfm2-plan.md 3.5).
  bool router_fp32 = true;

  int64_t conv_l_cache = kConvKernel;
  int64_t conv_dim = 0;  // channels of the depthwise conv == hidden_size globally; the rank's share under TP

  int64_t vocab_size = 0;
  int64_t max_position_embeddings = 0;
  double norm_eps = 1e-5;
  double rope_theta = 1.0e7;  // rope_type "default": no scaling, full rotary over head_dim
  bool tie_word_embeddings = true;

  int64_t bos_token_id = 124894;
  int64_t eos_token_id = 124900;
  int64_t pad_token_id = 124893;

  // ---- tensor parallel -------------------------------------------------------------------------------------
  // GLOBAL (tp_world == 1) or the rank-local view Shard() returns: heads, kv heads, intermediate_size,
  // moe_intermediate_size and conv_dim divided by tp_world; vocab_size stays GLOBAL (VocabShardSize() is the
  // rank's tied-head slice); hidden_size stays (the residual is replicated).
  int tp_world = 1;
  int tp_rank = 0;
  int64_t VocabShardSize() const { return vocab_size / tp_world; }
  int64_t VocabShardBegin() const { return tp_rank * VocabShardSize(); }
  bool IsShard() const { return tp_world > 1; }

  // ---- per-layer helpers -----------------------------------------------------------------------------------
  bool IsAttentionLayer(int64_t i) const {
    if (i < 0 || i >= static_cast<int64_t>(layer_types.size()))
      throw std::out_of_range("Lfm2Config::IsAttentionLayer: layer index out of range");
    return layer_types[static_cast<size_t>(i)] == kFull;
  }
  bool IsConvLayer(int64_t i) const { return !IsAttentionLayer(i); }
  bool IsDenseLayer(int64_t i) const { return i >= 0 && i < num_dense_layers; }
  bool IsMoeLayer(int64_t i) const { return i >= num_dense_layers && i < num_hidden_layers; }
  int64_t NumAttentionLayers() const {
    int64_t n = 0;
    for (const auto& t : layer_types) n += t == kFull ? 1 : 0;
    return n;
  }
  int64_t NumConvLayers() const { return static_cast<int64_t>(layer_types.size()) - NumAttentionLayers(); }
  int64_t NumMoeLayers() const { return num_hidden_layers - num_dense_layers; }
  int64_t Gqa() const { return num_attention_heads / num_key_value_heads; }
  int64_t QDim() const { return num_attention_heads * head_dim; }       // q_proj rows == out_proj K
  int64_t KvDim() const { return num_key_value_heads * head_dim; }      // k_proj / v_proj rows
  int64_t ConvInProjRows() const { return 3 * conv_dim; }               // in_proj N: segments B, C, x
  int64_t ExpertGateUpRows() const { return 2 * moe_intermediate_size; }  // gate rows first, then up

  // The context a run gets: `requested` <= 0 -> the config's max_position_embeddings; a larger value throws (the
  // rope has no scaling beyond it, so there is no extended-context opt-in).
  int64_t ResolveMaxCtx(int64_t requested) const {
    const int64_t native = max_position_embeddings;
    if (requested <= 0) return native;
    if (requested <= native) return requested;
    throw std::invalid_argument("Lfm2Config::ResolveMaxCtx: " + std::to_string(requested) +
                                " exceeds the checkpoint's max_position_embeddings (" + std::to_string(native) +
                                "); LFM2 has no rope scaling, so there is no extended context");
  }

  // The thin generic view TextModel::Config() hands the CLI and server (hidden_size, num_hidden_layers; text_model.h)
  // and tp::RuleFor reads: GDN fields stay 0, `arch` says LFM2, the MoE geometry rides on the new fields.
  ModelConfig ToModelConfig() const {
    ModelConfig m;
    m.arch = Arch::kLfm2Moe;
    m.hidden_size = hidden_size;
    m.num_hidden_layers = num_hidden_layers;
    m.layer_types = layer_types;
    m.num_attention_heads = num_attention_heads;
    m.num_key_value_heads = num_key_value_heads;
    m.head_dim = head_dim;
    m.global_head_dim = 0;
    m.attn_output_gate = false;
    m.intermediate_size = intermediate_size;
    m.moe_intermediate_size = moe_intermediate_size;
    m.num_experts = num_experts;
    m.num_dense_layers = num_dense_layers;
    m.partial_rotary_factor = 1.0;
    m.rope_theta = rope_theta;
    m.mrope_interleaved = false;
    m.mrope_section.clear();
    m.linear_num_key_heads = m.linear_num_value_heads = 0;
    m.linear_key_head_dim = m.linear_value_head_dim = 0;
    m.linear_conv_kernel_dim = conv_l_cache;
    m.rms_norm_eps = norm_eps;
    m.vocab_size = vocab_size;
    m.tie_word_embeddings = tie_word_embeddings;
    m.mtp_num_hidden_layers = 0;
    m.tp_world = tp_world;
    m.tp_rank = tp_rank;
    return m;
  }

  // `model_config` is the container's verbatim config.json (flat); an object holding "text_config" is accepted too.
  static Lfm2Config FromModelConfig(const nlohmann::json& model_config) {
    if (model_config.is_object() && model_config.contains("text_config"))
      return FromJson(model_config.at("text_config"));
    return FromJson(model_config);
  }

  // `cfg` is config.json. Every field the geometry depends on is required; unsupported variants throw.
  static Lfm2Config FromJson(const nlohmann::json& cfg) {
    const std::string who = "Lfm2Config::FromJson: ";
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

    if (cfg.contains("model_type")) {
      const std::string mt = cfg.at("model_type").get<std::string>();
      refuse_if(mt != "lfm2_moe", "model_type '" + mt + "' (want lfm2_moe)");
    }

    Lfm2Config c;
    c.hidden_size = positive("hidden_size");
    c.num_hidden_layers = positive("num_hidden_layers");
    for (const auto& lt : req("layer_types")) {
      const std::string t = lt.get<std::string>();
      if (t != kConv && t != kFull) throw std::runtime_error(who + "unknown layer type '" + t + "'");
      c.layer_types.push_back(t);
    }
    if (static_cast<int64_t>(c.layer_types.size()) != c.num_hidden_layers)
      throw std::runtime_error(who + "layer_types length " + std::to_string(c.layer_types.size()) +
                               " does not match num_hidden_layers " + std::to_string(c.num_hidden_layers));
    // Attention at 2, 6, 10, ...: the KV table (one entry per attention layer), the conv-state table and the
    // per-layer tensor shapes all assume it, so assert it rather than trust it.
    for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
      const bool want_attn = i % kAttnEvery == kAttnPhase;
      if ((c.layer_types[static_cast<size_t>(i)] == kFull) != want_attn)
        throw std::runtime_error(who + "layer_types[" + std::to_string(i) + "] is '" +
                                 c.layer_types[static_cast<size_t>(i)] + "' but the pattern (attention at i % 4 == 2) wants '" +
                                 (want_attn ? kFull : kConv) + "'");
    }

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
    c.moe_intermediate_size = positive("moe_intermediate_size");
    c.num_experts = positive("num_experts");
    c.num_experts_per_tok = positive("num_experts_per_tok");
    refuse_if(c.num_experts_per_tok > c.num_experts, "num_experts_per_tok " + std::to_string(c.num_experts_per_tok) +
                                                         " > num_experts " + std::to_string(c.num_experts));
    c.num_dense_layers = req_int("num_dense_layers");
    if (c.num_dense_layers < 0 || c.num_dense_layers > c.num_hidden_layers)
      throw std::runtime_error(who + "num_dense_layers is outside [0, num_hidden_layers]");
    if (cfg.contains("routed_scaling_factor") && !cfg.at("routed_scaling_factor").is_null())
      c.routed_scaling_factor = cfg.at("routed_scaling_factor").get<double>();
    c.norm_topk_prob = req("norm_topk_prob").get<bool>();
    refuse_if(!c.norm_topk_prob, "norm_topk_prob=false (the router normalizes the selected scores)");
    c.use_expert_bias = req("use_expert_bias").get<bool>();
    refuse_if(!c.use_expert_bias, "use_expert_bias=false (every MoE layer carries an expert_bias)");

    c.conv_l_cache = positive("conv_L_cache");
    refuse_if(c.conv_l_cache != kConvKernel, "conv_L_cache " + std::to_string(c.conv_l_cache) + " (want 3)");
    refuse_if(req("conv_bias").get<bool>(), "conv_bias=true (the conv has no bias)");
    c.conv_dim = c.hidden_size;

    c.vocab_size = positive("vocab_size");
    c.max_position_embeddings = positive("max_position_embeddings");
    c.norm_eps = req("norm_eps").get<double>();
    c.tie_word_embeddings = req("tie_word_embeddings").get<bool>();
    refuse_if(!c.tie_word_embeddings, "tie_word_embeddings=false (the runtime reads the head from the embedding)");

    const nlohmann::json& rp = req("rope_parameters");
    if (!rp.is_object()) throw std::runtime_error(who + "rope_parameters is not an object");
    refuse_if(rp.value("rope_type", std::string()) != "default",
              "rope_type '" + rp.value("rope_type", std::string()) + "' (want default: no scaling)");
    if (!rp.contains("rope_theta")) throw std::runtime_error(who + "rope_parameters missing 'rope_theta'");
    c.rope_theta = rp.at("rope_theta").get<double>();
    refuse_if(cfg.contains("rope_scaling") && !cfg.at("rope_scaling").is_null(), "rope_scaling");

    if (cfg.contains("bos_token_id") && cfg.at("bos_token_id").is_number_integer())
      c.bos_token_id = cfg.at("bos_token_id").get<int64_t>();
    if (cfg.contains("eos_token_id") && cfg.at("eos_token_id").is_number_integer())
      c.eos_token_id = cfg.at("eos_token_id").get<int64_t>();
    if (cfg.contains("pad_token_id") && cfg.at("pad_token_id").is_number_integer())
      c.pad_token_id = cfg.at("pad_token_id").get<int64_t>();
    return c;
  }

  // The rank-local config of `global` for TP world `world`, rank `rank` (docs/lfm2-plan.md 3.6). Throws
  // std::invalid_argument naming the failing field unless every shape a rank's kernels see is legal:
  //   world in {1,2}; 0 <= rank < world; global.tp_world == 1 (no double sharding);
  //   heads, kv heads, intermediate_size, moe_intermediate_size and conv channels divide by world;
  //   the GQA is unchanged (8 kv heads -> 4, still 4 q per kv);
  //   row-parallel rank K % 256 == 0 (the relaxed FallbackTuning rule, linear.cpp: SK 4 x 64): attn.o
  //   (heads/world * head_dim), conv out_proj (conv_dim/world), dense w2 (intermediate/world), expert down
  //   (moe_intermediate/world);
  //   column-parallel rank rows % 16 == 0: q, k/v, each conv in_proj segment, each dense / expert gate_up segment;
  //   vocab_size % (16 * world) == 0 (the tied head's shard is whole 16-row tiles).
  // The tied embedding is ALLOWED (the table is replicated and the head is a vocab slice of it).
  static Lfm2Config Shard(const Lfm2Config& global, int world, int rank) {
    const std::string where = "Lfm2Config::Shard(world=" + std::to_string(world) + ", rank=" + std::to_string(rank) + "): ";
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
    require_div("moe_intermediate_size", global.moe_intermediate_size, world);
    require_div("conv_dim", global.conv_dim, world);

    Lfm2Config r = global;
    r.tp_world = world;
    r.tp_rank = rank;
    r.num_attention_heads = global.num_attention_heads / world;
    r.num_key_value_heads = global.num_key_value_heads / world;
    r.intermediate_size = global.intermediate_size / world;
    r.moe_intermediate_size = global.moe_intermediate_size / world;
    r.conv_dim = global.conv_dim / world;

    if (r.Gqa() != global.Gqa())
      fail("the GQA changes from " + std::to_string(global.Gqa()) + " to " + std::to_string(r.Gqa()));

    require_div("attn.o rank K (heads/world * head_dim)", r.QDim(), 256);
    require_div("conv.out_proj rank K (conv_dim/world)", r.conv_dim, 256);
    require_div("dense w2 rank K (intermediate_size/world)", r.intermediate_size, 256);
    require_div("expert down rank K (moe_intermediate_size/world)", r.moe_intermediate_size, 256);

    require_div("attn.q rank rows (heads/world * head_dim)", r.QDim(), 16);
    require_div("attn.k/v rank rows (kv/world * head_dim)", r.KvDim(), 16);
    require_div("conv.in_proj rank segment (conv_dim/world)", r.conv_dim, 16);
    require_div("dense gate_up rank segment (intermediate_size/world)", r.intermediate_size, 16);
    require_div("expert gate_up rank segment (moe_intermediate_size/world)", r.moe_intermediate_size, 16);
    require_div("vocab_size (tied lm_head rank shard in whole 16-row tiles)", global.vocab_size,
                16 * static_cast<int64_t>(world));
    return r;
  }
};

}  // namespace r4dx::model
