// r4dx::model::GemmaConfig -- the text-model geometry of gemma4_unified, parsed STRICTLY from a
// container's `__metadata__.model_config.text_config` (docs/gemma4-plan.md 3.2, task M1-1; every
// fact below is settled in docs/gemma4-semantics.md).
//
// Header-only, HIP-free (like model_config.h): tests/model/test_gemma_config.cpp runs it on the real
// Huihui config without a device.
//
// Two attention geometries share one stack, five sliding layers then one full layer:
//   sliding: 16 q heads x 256, 8 kv heads (GQA 2), window 1024, default rope theta 1e4;
//   full:    16 q heads x 512 (global_head_dim), 1 kv head (GQA 16), V = raw k_proj output
//            (attention_k_eq_v, no v_proj), "proportional" rope theta 1e6, partial factor 0.25.
// Everything the runtime does not implement (MoE, per-layer inputs, shared KV, double-wide MLP, a
// config that is not k_eq_v / tied / gelu-tanh) is REFUSED here rather than loaded wrongly.
//
// Context length (docs/gemma4-plan.md section 9.1): the default is the config's
// max_position_embeddings (131072 for Huihui); 262144 is opt-in (ResolveMaxCtx). The difference is
// metadata and KV sizing only -- the rope has no scaling (docs/gemma4-semantics.md 1.9).
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_config.h"
#include "nlohmann/json.hpp"

namespace r4dx::model {

struct GemmaConfig {
  static constexpr int64_t kDefaultMaxCtx = 131072;   // docs/gemma4-plan.md section 9.1
  static constexpr int64_t kExtendedMaxCtx = 262144;  // opt-in
  static constexpr const char* kSliding = "sliding_attention";
  static constexpr const char* kFull = "full_attention";

  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  std::vector<std::string> layer_types;  // "sliding_attention" | "full_attention", per layer

  int64_t num_attention_heads = 0;
  int64_t num_kv_heads_sliding = 0;  // num_key_value_heads
  int64_t num_kv_heads_full = 0;     // num_global_key_value_heads
  int64_t head_dim_sliding = 0;      // head_dim
  int64_t head_dim_full = 0;         // global_head_dim
  int64_t sliding_window = 0;        // a query sees `sliding_window` keys INCLUDING itself

  int64_t intermediate_size = 0;
  int64_t vocab_size = 0;
  int64_t max_position_embeddings = 0;

  double rope_theta_sliding = 1.0e4;  // rope_type "default", full rotary over head_dim_sliding
  double rope_theta_full = 1.0e6;     // rope_type "proportional"
  double partial_rotary_factor_full = 0.25;

  double rms_norm_eps = 1e-6;
  double final_logit_softcapping = 0.0;  // tanh(x / cap) * cap; 0 = none
  bool tie_word_embeddings = true;
  bool attention_k_eq_v = true;
  std::string use_bidirectional_attention = "vision";  // image blocks only; sliding layers only (semantics 2)

  // Semantic flags (all VERIFIED in docs/gemma4-semantics.md; kept as flags so a golden mismatch is
  // a one-line switch, not a rewrite).
  bool v_norm_all_layers = true;  // v_norm (RMSNorm, no weight) on V of every layer (semantics 1.1)
  bool embed_scale_bf16 = true;   // sqrt(hidden) cast to bf16 before the multiply: 62.0 (semantics 1.3)

  int64_t bos_token_id = 2;
  int64_t pad_token_id = 0;

  // ---- tensor parallel -------------------------------------------------------------------------
  // GLOBAL (tp_world == 1) or the rank-local view Shard() returns: num_attention_heads,
  // num_kv_heads_sliding and intermediate_size divided by tp_world; the full layers' single kv head
  // stays replicated on every rank (so their GQA per rank is num_attention_heads/world / 1);
  // vocab_size stays GLOBAL (VocabShardSize() is the rank's tied-head slice).
  int tp_world = 1;
  int tp_rank = 0;
  int64_t VocabShardSize() const { return vocab_size / tp_world; }
  int64_t VocabShardBegin() const { return tp_rank * VocabShardSize(); }
  bool IsShard() const { return tp_world > 1; }

  // ---- per-layer helpers -----------------------------------------------------------------------
  bool IsFullLayer(int64_t i) const {
    if (i < 0 || i >= static_cast<int64_t>(layer_types.size()))
      throw std::out_of_range("GemmaConfig::IsFullLayer: layer index out of range");
    return layer_types[static_cast<size_t>(i)] == kFull;
  }
  int64_t NumFullLayers() const {
    int64_t n = 0;
    for (const auto& t : layer_types) n += t == kFull ? 1 : 0;
    return n;
  }
  int64_t NumSlidingLayers() const { return static_cast<int64_t>(layer_types.size()) - NumFullLayers(); }

  int64_t HeadDim(int64_t layer) const { return IsFullLayer(layer) ? head_dim_full : head_dim_sliding; }
  int64_t NumKvHeads(int64_t layer) const {
    return IsFullLayer(layer) ? num_kv_heads_full : num_kv_heads_sliding;
  }
  int64_t Gqa(int64_t layer) const { return num_attention_heads / NumKvHeads(layer); }
  int64_t QDim(int64_t layer) const { return num_attention_heads * HeadDim(layer); }  // q_proj rows == o_proj K
  int64_t KvDim(int64_t layer) const { return NumKvHeads(layer) * HeadDim(layer); }   // k_proj (and v_proj) rows
  double RopeTheta(int64_t layer) const { return IsFullLayer(layer) ? rope_theta_full : rope_theta_sliding; }
  // Rotated head dims: sliding rotates all of head_dim; full rotates partial_factor of it (the rest
  // is NoPE: inv_freq 0, cos 1 / sin 0). Frequencies use the FULL head_dim as their exponent
  // denominator (semantics 1.5), so RotaryAngles is "pairs", i.e. half of the rotated dims.
  int64_t RotaryAngles(int64_t layer) const {
    if (!IsFullLayer(layer)) return head_dim_sliding / 2;
    return static_cast<int64_t>(partial_rotary_factor_full * static_cast<double>(head_dim_full)) / 2;
  }

  // sqrt(hidden_size) as the embedding multiply uses it: in bf16 (the weight dtype) 61.9677 -> 62.0.
  float EmbedScale() const {
    const float s = std::sqrt(static_cast<float>(hidden_size));
    return embed_scale_bf16 ? Bf16Round(s) : s;
  }

  // The context a run gets: `requested` <= 0 -> the config's max_position_embeddings (131072 for
  // Huihui); a larger value is the opt-in, allowed only with `allow_extended` and only up to
  // kExtendedMaxCtx.
  int64_t ResolveMaxCtx(int64_t requested, bool allow_extended) const {
    const int64_t native = max_position_embeddings;
    if (requested <= 0) return native;
    if (requested <= native) return requested;
    if (!allow_extended) {
      throw std::invalid_argument("GemmaConfig::ResolveMaxCtx: " + std::to_string(requested) +
                                  " exceeds the checkpoint's max_position_embeddings (" +
                                  std::to_string(native) + "); the " +
                                  std::to_string(kExtendedMaxCtx) + "-token context is opt-in");
    }
    if (requested > kExtendedMaxCtx) {
      throw std::invalid_argument("GemmaConfig::ResolveMaxCtx: " + std::to_string(requested) +
                                  " exceeds the supported maximum " + std::to_string(kExtendedMaxCtx));
    }
    return requested;
  }

  // The thin generic view TextModel::Config() hands the CLI and server (they read hidden_size and
  // num_hidden_layers; text_model.h): GDN fields stay 0, `arch` says Gemma. Sliding-layer numbers
  // stand in for the per-layer-type ones.
  ModelConfig ToModelConfig() const {
    ModelConfig m;
    m.arch = Arch::kGemma4;
    m.hidden_size = hidden_size;
    m.num_hidden_layers = num_hidden_layers;
    m.layer_types = layer_types;
    m.num_attention_heads = num_attention_heads;
    m.num_key_value_heads = num_kv_heads_sliding;
    m.head_dim = head_dim_sliding;
    m.attn_output_gate = false;
    m.intermediate_size = intermediate_size;
    m.partial_rotary_factor = 1.0;  // sliding layers; the full layers' 0.25 is partial_rotary_factor_full
    m.rope_theta = rope_theta_sliding;
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

  // `model_config` is the container's verbatim config.json (a top-level object holding
  // "text_config"); a bare text_config object is accepted too.
  static GemmaConfig FromModelConfig(const nlohmann::json& model_config) {
    if (model_config.is_object() && model_config.contains("text_config"))
      return FromJson(model_config.at("text_config"));
    return FromJson(model_config);
  }

  // `text_cfg` is config.json's `text_config`. Every field the geometry depends on is required;
  // unsupported variants (MoE, per-layer inputs, KV sharing, double-wide MLP, untied head, a rope or
  // activation other than the checkpoint's) throw, and so does a layer_types list that is not five
  // sliding layers then one full layer.
  static GemmaConfig FromJson(const nlohmann::json& text_cfg) {
    const std::string who = "GemmaConfig::FromJson: ";
    if (!text_cfg.is_object()) throw std::runtime_error(who + "text_config is not an object");
    auto req = [&](const char* key) -> const nlohmann::json& {
      if (!text_cfg.contains(key)) throw std::runtime_error(who + "text_config missing '" + key + "'");
      return text_cfg.at(key);
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

    if (text_cfg.contains("model_type")) {
      const std::string mt = text_cfg.at("model_type").get<std::string>();
      refuse_if(mt != "gemma4_unified_text", "text_config.model_type '" + mt + "' (want gemma4_unified_text)");
    }

    GemmaConfig c;
    c.hidden_size = positive("hidden_size");
    c.num_hidden_layers = positive("num_hidden_layers");
    for (const auto& lt : req("layer_types")) {
      const std::string t = lt.get<std::string>();
      if (t != kSliding && t != kFull) throw std::runtime_error(who + "unknown layer type '" + t + "'");
      c.layer_types.push_back(t);
    }
    if (static_cast<int64_t>(c.layer_types.size()) != c.num_hidden_layers)
      throw std::runtime_error(who + "layer_types length " + std::to_string(c.layer_types.size()) +
                               " does not match num_hidden_layers " + std::to_string(c.num_hidden_layers));
    // The 5:1 pattern, full at 5, 11, ..., 47 (HF's default rule; semantics 1.8): the KV ring, the
    // kernel dispatch and the per-layer tensor shapes all assume it, so assert it rather than trust it.
    for (int64_t i = 0; i < c.num_hidden_layers; ++i) {
      const bool want_full = (i + 1) % 6 == 0;
      if ((c.layer_types[static_cast<size_t>(i)] == kFull) != want_full)
        throw std::runtime_error(who + "layer_types[" + std::to_string(i) + "] is '" +
                                 c.layer_types[static_cast<size_t>(i)] +
                                 "' but the 5 sliding : 1 full pattern (full at 5, 11, ...) wants '" +
                                 (want_full ? kFull : kSliding) + "'");
    }

    c.num_attention_heads = positive("num_attention_heads");
    c.num_kv_heads_sliding = positive("num_key_value_heads");
    c.num_kv_heads_full = positive("num_global_key_value_heads");
    c.head_dim_sliding = positive("head_dim");
    c.head_dim_full = positive("global_head_dim");
    c.sliding_window = positive("sliding_window");
    c.intermediate_size = positive("intermediate_size");
    c.vocab_size = positive("vocab_size");
    c.max_position_embeddings = positive("max_position_embeddings");
    if (c.num_attention_heads % c.num_kv_heads_sliding != 0 || c.num_attention_heads % c.num_kv_heads_full != 0)
      throw std::runtime_error(who + "num_attention_heads is not a multiple of both kv head counts");

    c.rms_norm_eps = req("rms_norm_eps").get<double>();
    if (text_cfg.contains("final_logit_softcapping") && !text_cfg.at("final_logit_softcapping").is_null())
      c.final_logit_softcapping = text_cfg.at("final_logit_softcapping").get<double>();
    c.tie_word_embeddings = req("tie_word_embeddings").get<bool>();
    refuse_if(!c.tie_word_embeddings, "tie_word_embeddings=false (the runtime reads the head from the embedding)");
    c.attention_k_eq_v = req("attention_k_eq_v").get<bool>();
    refuse_if(!c.attention_k_eq_v, "attention_k_eq_v=false (full layers would carry a v_proj)");
    const std::string act = req("hidden_activation").get<std::string>();
    refuse_if(act != "gelu_pytorch_tanh", "hidden_activation '" + act + "' (want gelu_pytorch_tanh)");
    if (text_cfg.contains("use_bidirectional_attention") && text_cfg.at("use_bidirectional_attention").is_string())
      c.use_bidirectional_attention = text_cfg.at("use_bidirectional_attention").get<std::string>();
    refuse_if(c.use_bidirectional_attention != "vision",
              "use_bidirectional_attention '" + c.use_bidirectional_attention +
                  "' (want vision: it halves the sliding window under 'all')");

    // Variants this checkpoint family can carry but the runtime does not implement.
    auto zero_or_absent = [&](const char* key) {
      if (!text_cfg.contains(key) || text_cfg.at(key).is_null()) return true;
      const nlohmann::json& v = text_cfg.at(key);
      if (v.is_boolean()) return !v.get<bool>();
      if (v.is_number()) return v.get<double>() == 0.0;
      return false;
    };
    refuse_if(!zero_or_absent("enable_moe_block"), "enable_moe_block");
    refuse_if(!zero_or_absent("hidden_size_per_layer_input"), "hidden_size_per_layer_input (per-layer inputs)");
    refuse_if(!zero_or_absent("num_kv_shared_layers"), "num_kv_shared_layers (shared KV)");
    refuse_if(!zero_or_absent("use_double_wide_mlp"), "use_double_wide_mlp");
    refuse_if(!zero_or_absent("attention_bias"), "attention_bias");

    const nlohmann::json& rp = req("rope_parameters");
    auto rope_for = [&](const char* type_key) -> const nlohmann::json& {
      if (!rp.is_object() || !rp.contains(type_key))
        throw std::runtime_error(who + "rope_parameters missing '" + type_key + "'");
      return rp.at(type_key);
    };
    const nlohmann::json& rs = rope_for("sliding_attention");
    const nlohmann::json& rf = rope_for("full_attention");
    refuse_if(rs.value("rope_type", "") != "default",
              "sliding rope_type '" + rs.value("rope_type", "") + "' (want default)");
    refuse_if(rf.value("rope_type", "") != "proportional",
              "full rope_type '" + rf.value("rope_type", "") + "' (want proportional)");
    c.rope_theta_sliding = rs.at("rope_theta").get<double>();
    c.rope_theta_full = rf.at("rope_theta").get<double>();
    c.partial_rotary_factor_full = rf.at("partial_rotary_factor").get<double>();
    if (!(c.partial_rotary_factor_full > 0.0 && c.partial_rotary_factor_full <= 1.0))
      throw std::runtime_error(who + "full-layer partial_rotary_factor must be in (0, 1]");
    // Rotated dims must pair up (rotate-half), and the sliding head must rotate fully.
    const double rot = c.partial_rotary_factor_full * static_cast<double>(c.head_dim_full);
    if (rot != std::floor(rot) || static_cast<int64_t>(rot) % 2 != 0)
      throw std::runtime_error(who + "partial_rotary_factor * global_head_dim is not an even integer");
    if (c.head_dim_sliding % 2 != 0 || c.head_dim_full % 2 != 0)
      throw std::runtime_error(who + "head dims must be even (rotate-half)");

    if (text_cfg.contains("bos_token_id") && text_cfg.at("bos_token_id").is_number_integer())
      c.bos_token_id = text_cfg.at("bos_token_id").get<int64_t>();
    if (text_cfg.contains("pad_token_id") && text_cfg.at("pad_token_id").is_number_integer())
      c.pad_token_id = text_cfg.at("pad_token_id").get<int64_t>();
    return c;
  }

  // The rank-local config of `global` for TP world `world`, rank `rank` (docs/gemma4-plan.md 3.2).
  // Throws std::invalid_argument naming the failing field unless every shape a rank's kernels see is
  // legal:
  //   world in {1,2}; 0 <= rank < world; global.tp_world == 1 (no double sharding);
  //   num_attention_heads, num_kv_heads_sliding and intermediate_size divide by world;
  //   the sliding GQA is unchanged (8 kv heads -> 4, still 2 q per kv) and the full layers' kv head
  //   is REPLICATED (num_kv_heads_full stays), so their per-rank GQA is heads/world / kv_full;
  //   row-parallel rank K % 512 == 0: o_proj sliding (heads/world * head_dim_sliding), o_proj full
  //   (heads/world * head_dim_full), mlp.down (intermediate/world);
  //   column-parallel rank rows % 16 == 0: q, sliding k/v, full k (replicated), gate_up;
  //   vocab_size % (16 * world) == 0 (the tied head's shard is whole 16-row tiles).
  // The tied embedding is ALLOWED (unlike ModelConfig::Shard): the table is replicated and the head
  // is a vocab slice of it.
  static GemmaConfig Shard(const GemmaConfig& global, int world, int rank) {
    const std::string where = "GemmaConfig::Shard(world=" + std::to_string(world) +
                              ", rank=" + std::to_string(rank) + "): ";
    const auto fail = [&](const std::string& what) { throw std::invalid_argument(where + what); };
    const auto require_div = [&](const char* field, int64_t value, int64_t divisor) {
      if (divisor <= 0 || value % divisor != 0)
        fail(std::string(field) + " = " + std::to_string(value) + " is not divisible by " +
             std::to_string(divisor));
    };
    if (world != 1 && world != 2) fail("world must be 1 or 2");
    if (rank < 0 || rank >= world) fail("rank must be in [0, world)");
    if (global.tp_world != 1)
      fail("tp_world = " + std::to_string(global.tp_world) + " -- `global` is already a rank shard");

    require_div("num_attention_heads", global.num_attention_heads, world);
    require_div("num_kv_heads_sliding", global.num_kv_heads_sliding, world);
    require_div("intermediate_size", global.intermediate_size, world);

    GemmaConfig r = global;
    r.tp_world = world;
    r.tp_rank = rank;
    r.num_attention_heads = global.num_attention_heads / world;
    r.num_kv_heads_sliding = global.num_kv_heads_sliding / world;
    r.intermediate_size = global.intermediate_size / world;
    // num_kv_heads_full stays: the single global kv head is replicated on every rank.

    if (r.num_attention_heads / r.num_kv_heads_sliding != global.num_attention_heads / global.num_kv_heads_sliding)
      fail("the sliding GQA changes from " +
           std::to_string(global.num_attention_heads / global.num_kv_heads_sliding) + " to " +
           std::to_string(r.num_attention_heads / r.num_kv_heads_sliding));
    if (r.num_attention_heads % r.num_kv_heads_full != 0)
      fail("per-rank heads " + std::to_string(r.num_attention_heads) + " do not divide by the replicated full kv heads " +
           std::to_string(r.num_kv_heads_full));

    require_div("attn.o sliding rank K (heads/world * head_dim)", r.num_attention_heads * r.head_dim_sliding, 512);
    require_div("attn.o full rank K (heads/world * global_head_dim)", r.num_attention_heads * r.head_dim_full, 512);
    require_div("mlp.down rank K (intermediate_size/world)", r.intermediate_size, 512);

    require_div("attn.q sliding rank rows (heads/world * head_dim)", r.num_attention_heads * r.head_dim_sliding, 16);
    require_div("attn.q full rank rows (heads/world * global_head_dim)", r.num_attention_heads * r.head_dim_full, 16);
    require_div("attn.k/v sliding rank rows (kv/world * head_dim)", r.num_kv_heads_sliding * r.head_dim_sliding, 16);
    require_div("attn.k full rows (replicated kv * global_head_dim)", r.num_kv_heads_full * r.head_dim_full, 16);
    require_div("mlp.gate_up rank segment (intermediate_size/world)", r.intermediate_size, 16);
    require_div("vocab_size (tied lm_head rank shard in whole 16-row tiles)", global.vocab_size,
                16 * static_cast<int64_t>(world));
    return r;
  }

 private:
  // Round-to-nearest-even to bf16 (the same algorithm as r4dx::core::FloatToBf16), returned as float.
  static float Bf16Round(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, 4);
    if ((bits & 0x7FFFFFFFu) > 0x7F800000u) return f;  // NaN
    const uint32_t rounded = (bits + 0x7FFFu + ((bits >> 16) & 1u)) & 0xFFFF0000u;
    float out;
    std::memcpy(&out, &rounded, 4);
    return out;
  }
};

}  // namespace r4dx::model
