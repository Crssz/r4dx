// r4dx::model::Inkling2Config -- the text-model geometry of an inkling2_loop container (Inkling-2-Mini; docs/inkling2-plan.md
// 3.1, 5), parsed STRICTLY from `__metadata__.model_config` (the verbatim config.json: the top-level object with
// model_type "inkling2_loop" holding `text_config`, which holds `loop_config`).
//
// Header-only and HIP-free (like lfm2_config.h / llama_config.h): tests/model/test_inkling2_config.cpp runs it on the real
// and the tiny config without a device.
//
// What it decides besides the plain fields:
//   * layer kinds: `layer_types` ("hybrid" = full attention, "hybrid_sliding" = sliding window) when present, else
//     `local_layer_ids` (every listed layer sliding, the rest full), else HF's default (i + 1) % 6 == 0 full; when both
//     lists are present they must agree (refused otherwise);
//   * the VISIT TABLE (Visits()): the order the HF Inkling2LoopTextModel.forward runs its layers in -- layers
//     [0, loop_start), then loop_passes passes over [loop_start, loop_end), then [loop_end, layers) -- each visit with its
//     KV / conv cache slot: HF's cache layer index (`layer_idx + cache_offset`, cache_offset = layers + (pass - 1) *
//     loop_layers - loop_start for pass >= 1), so slots [0, layers) are the stored layers' first visits and
//     [layers, layers + (passes - 1) * loop_layers) the extra loop passes. The real model: 48 visits, 48 slots.
//   * Shard(world, rank): the rank-local view for TP=2 (docs/inkling2-plan.md 5's table): q heads and kv heads / world
//     (GQA kept), one expert's intermediate / world, the lm_head's STORED rows (HeadRowsStored, a multiple of 16) / world;
//     everything replicated keeps its global size.
//   * ResolveMaxCtx: default 131072 (the KV budget, docs/inkling2-plan.md 0.4), explicit values up to 1048576.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_config.h"
#include "nlohmann/json.hpp"

namespace r4dx::model {

// One layer visit of the forward pass (docs/inkling2-plan.md 3.3 "Model loop").
struct Inkling2Visit {
  enum class Phase { kPre, kLoop, kPost };
  int index = 0;       // 0 .. Visits().size() - 1, in execution order
  int layer = 0;       // the stored layer whose weights run
  Phase phase = Phase::kPre;
  int pass = 0;        // loop pass 0 .. loop_passes - 1 (0 outside the loop)
  int loop_j = -1;     // position inside the loop, 0 .. loop_layers - 1 (-1 outside)
  int cache_slot = 0;  // the KV / conv state this visit owns (HF's cache layer index)
  bool HasMessages() const { return phase == Phase::kLoop && pass >= 1; }
};

struct Inkling2Config {
  static constexpr int64_t kDefaultMaxCtx = 131072;
  static constexpr int64_t kMaxCtx = 1048576;

  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  int64_t d_rel = 0;
  int64_t rel_extent = 0;           // full layers' relative-bias extent
  int64_t sliding_window = 0;       // sliding layers: window AND relative-bias extent (HF: rel_extent = sliding_window_size)
  int64_t conv_kernel = 4;          // sconv_kernel_size (the kernels implement 4 taps)
  int64_t vocab_size = 0;           // embedding rows (201024)
  int64_t unpadded_vocab_size = 0;  // logits width (200058)
  int64_t num_experts = 0;
  int64_t num_experts_per_tok = 0;
  int64_t moe_intermediate_size = 0;
  double route_scale = 1.0;
  double mlp_gate_softcap = 4.0, mlp_up_softcap = 25.0;
  double final_logit_softcap = 30.0;
  double rms_norm_eps = 1e-6;
  double qk_clip = 0.0, rel_clip = 0.0;
  double log_alpha = 0.0, log_floor = 1.0;
  int64_t attn_res_block = 6;
  int64_t loop_start = 0, loop_end = 0, loop_passes = 1, gate_rank = 0;
  std::vector<bool> full;  // per layer
  int64_t eos_token_id = -1, pad_token_id = -1, bos_token_id = -1;

  int tp_world = 1;
  int tp_rank = 0;

  // ---- derived ---------------------------------------------------------------------------------------------------
  bool IsFull(int64_t i) const { return full.at(static_cast<size_t>(i)); }
  bool IsLoop(int64_t i) const { return i >= loop_start && i < loop_end; }
  bool IsBoundary(int64_t i) const { return i % attn_res_block == 0; }
  int64_t LoopLayers() const { return loop_end - loop_start; }
  int64_t QDim() const { return num_attention_heads * head_dim; }
  int64_t KvDim() const { return num_key_value_heads * head_dim; }
  int64_t Gqa() const { return num_attention_heads / num_key_value_heads; }
  int64_t RelExtent(int64_t i) const { return IsFull(i) ? rel_extent : sliding_window; }
  int64_t Window(int64_t i) const { return IsFull(i) ? 0 : sliding_window; }
  // The lm_head rows the container stores (unpadded_vocab_size rounded up to the 16-row tile; zero rows past it).
  int64_t HeadRowsStored() const { return (unpadded_vocab_size + 15) / 16 * 16; }
  int64_t HeadRowsLocal() const { return HeadRowsStored() / tp_world; }
  int64_t HeadRowBegin() const { return tp_rank * HeadRowsLocal(); }
  int64_t NumCacheSlots() const { return num_hidden_layers + (loop_passes - 1) * LoopLayers(); }

  std::vector<Inkling2Visit> Visits() const {
    std::vector<Inkling2Visit> v;
    const auto add = [&](int layer, Inkling2Visit::Phase ph, int pass, int j, int slot) {
      Inkling2Visit x;
      x.index = static_cast<int>(v.size());
      x.layer = layer;
      x.phase = ph;
      x.pass = pass;
      x.loop_j = j;
      x.cache_slot = slot;
      v.push_back(x);
    };
    for (int64_t L = 0; L < loop_start; ++L)
      add(static_cast<int>(L), Inkling2Visit::Phase::kPre, 0, -1, static_cast<int>(L));
    for (int64_t p = 0; p < loop_passes; ++p) {
      for (int64_t L = loop_start; L < loop_end; ++L) {
        const int64_t slot = p == 0 ? L : num_hidden_layers + (p - 1) * LoopLayers() + (L - loop_start);
        add(static_cast<int>(L), Inkling2Visit::Phase::kLoop, static_cast<int>(p), static_cast<int>(L - loop_start),
            static_cast<int>(slot));
      }
    }
    for (int64_t L = loop_end; L < num_hidden_layers; ++L)
      add(static_cast<int>(L), Inkling2Visit::Phase::kPost, 0, -1, static_cast<int>(L));
    return v;
  }

  // The context a run gets: `requested` <= 0 -> 131072; explicit values in [1, 1048576] as given; above throws.
  static int64_t ResolveMaxCtx(int64_t requested) {
    if (requested <= 0) return kDefaultMaxCtx;
    if (requested <= kMaxCtx) return requested;
    throw std::invalid_argument("Inkling2Config::ResolveMaxCtx: " + std::to_string(requested) +
                                " exceeds the model's 1048576-token context (model_max_length)");
  }

  ModelConfig ToModelConfig() const {
    ModelConfig m;
    m.arch = Arch::kInkling2Loop;
    m.hidden_size = hidden_size;
    m.num_hidden_layers = num_hidden_layers;
    m.layer_types.clear();
    for (int64_t i = 0; i < num_hidden_layers; ++i) m.layer_types.push_back(IsFull(i) ? "full_attention" : "sliding_attention");
    m.num_attention_heads = num_attention_heads;
    m.num_key_value_heads = num_key_value_heads;
    m.head_dim = head_dim;
    m.global_head_dim = 0;
    m.attn_output_gate = true;  // the sigmoid output gate
    m.intermediate_size = 0;
    m.moe_intermediate_size = moe_intermediate_size;
    m.num_experts = num_experts;
    m.num_dense_layers = 0;
    m.partial_rotary_factor = 0.0;  // no RoPE
    m.rope_theta = 0.0;
    m.mrope_interleaved = false;
    m.mrope_section.clear();
    m.linear_num_key_heads = m.linear_num_value_heads = 0;
    m.linear_key_head_dim = m.linear_value_head_dim = 0;
    m.linear_conv_kernel_dim = conv_kernel;
    m.rms_norm_eps = rms_norm_eps;
    m.vocab_size = unpadded_vocab_size;  // the logits width every TextModel returns (token ids above it are never sampled)
    m.tie_word_embeddings = false;
    m.mtp_num_hidden_layers = 0;
    m.tp_world = tp_world;
    m.tp_rank = tp_rank;
    return m;
  }

  // `model_config`: the verbatim config.json (top level with `text_config`), or the text_config object itself.
  static Inkling2Config FromModelConfig(const nlohmann::json& model_config) {
    if (model_config.is_object() && model_config.contains("text_config")) {
      const std::string mt = model_config.value("model_type", std::string("inkling2_loop"));
      if (mt != "inkling2_loop")
        throw std::runtime_error("Inkling2Config::FromModelConfig: model_type '" + mt + "' (want inkling2_loop)");
      Inkling2Config c = FromJson(model_config.at("text_config"));
      if (c.eos_token_id < 0 && model_config.contains("eos_token_id") && model_config.at("eos_token_id").is_number_integer())
        c.eos_token_id = model_config.at("eos_token_id").get<int64_t>();
      return c;
    }
    return FromJson(model_config);
  }

  // `t` is text_config. Strict: every geometry field is required; what the runtime does not implement is refused.
  static Inkling2Config FromJson(const nlohmann::json& t) {
    const std::string who = "Inkling2Config::FromJson: ";
    if (!t.is_object()) throw std::runtime_error(who + "text_config is not an object");
    const auto req = [&](const nlohmann::json& o, const char* key) -> const nlohmann::json& {
      if (!o.contains(key) || o.at(key).is_null()) throw std::runtime_error(who + "missing '" + key + "'");
      return o.at(key);
    };
    const auto pos_int = [&](const nlohmann::json& o, const char* key) {
      const nlohmann::json& v = req(o, key);
      if (!v.is_number_integer() || v.get<int64_t>() <= 0)
        throw std::runtime_error(who + "'" + key + "' must be a positive integer");
      return v.get<int64_t>();
    };
    const auto num = [&](const nlohmann::json& o, const char* key) {
      const nlohmann::json& v = req(o, key);
      if (!v.is_number()) throw std::runtime_error(who + "'" + key + "' must be a number");
      return v.get<double>();
    };
    const auto refuse_if = [&](bool bad, const std::string& why) {
      if (bad) throw std::runtime_error(who + "unsupported: " + why);
    };
    if (t.contains("model_type")) {
      const std::string mt = t.at("model_type").get<std::string>();
      refuse_if(mt != "inkling2_loop_text" && mt != "inkling2_loop", "text model_type '" + mt + "'");
    }
    Inkling2Config c;
    c.hidden_size = pos_int(t, "hidden_size");
    c.num_hidden_layers = pos_int(t, "num_hidden_layers");
    c.num_attention_heads = pos_int(t, "num_attention_heads");
    c.num_key_value_heads = pos_int(t, "num_key_value_heads");
    c.head_dim = pos_int(t, "head_dim");
    refuse_if(pos_int(t, "swa_num_attention_heads") != c.num_attention_heads ||
                  pos_int(t, "swa_num_key_value_heads") != c.num_key_value_heads ||
                  pos_int(t, "swa_head_dim") != c.head_dim,
              "swa_* attention geometry differs from the full-attention one");
    if (c.num_attention_heads % c.num_key_value_heads != 0)
      throw std::runtime_error(who + "num_attention_heads is not a multiple of num_key_value_heads");
    c.d_rel = pos_int(t, "d_rel");
    c.rel_extent = pos_int(t, "rel_extent");
    c.sliding_window = pos_int(t, "sliding_window_size");
    c.conv_kernel = pos_int(t, "sconv_kernel_size");
    refuse_if(c.conv_kernel != 4, "sconv_kernel_size " + std::to_string(c.conv_kernel) + " (the kernels implement 4 taps)");
    c.vocab_size = pos_int(t, "vocab_size");
    c.unpadded_vocab_size = t.contains("unpadded_vocab_size") ? pos_int(t, "unpadded_vocab_size") : c.vocab_size;
    if (c.unpadded_vocab_size > c.vocab_size) throw std::runtime_error(who + "unpadded_vocab_size > vocab_size");
    c.num_experts = pos_int(t, "n_routed_experts");
    c.num_experts_per_tok = pos_int(t, "num_experts_per_tok");
    refuse_if(c.num_experts > 256 || c.num_experts_per_tok > 16 || c.num_experts_per_tok > c.num_experts,
              "router geometry (the router kernel takes <= 256 experts, top_k <= 16)");
    c.moe_intermediate_size = pos_int(t, "intermediate_size");
    refuse_if(t.contains("n_shared_experts") && t.at("n_shared_experts").is_number_integer() &&
                  t.at("n_shared_experts").get<int64_t>() != 0,
              "n_shared_experts != 0");
    refuse_if(t.contains("use_gate_bias") && t.at("use_gate_bias").is_boolean() && !t.at("use_gate_bias").get<bool>(),
              "use_gate_bias false (the container always carries moe.bias)");
    refuse_if(t.contains("q_bias") && t.at("q_bias").is_boolean() && t.at("q_bias").get<bool>(), "q_bias true");
    refuse_if(t.contains("o_bias") && t.at("o_bias").is_boolean() && t.at("o_bias").get<bool>(), "o_bias true");
    c.route_scale = num(t, "route_scale");
    c.mlp_gate_softcap = t.contains("mlp_gate_softcap") ? num(t, "mlp_gate_softcap") : 4.0;
    c.mlp_up_softcap = t.contains("mlp_up_softcap") ? num(t, "mlp_up_softcap") : 25.0;
    refuse_if(c.mlp_gate_softcap != 4.0 || c.mlp_up_softcap != 25.0,
              "mlp softcaps other than 4 / 25 (r4dx_inkling2_swiglu_softcap_bf16 has them fixed)");
    c.final_logit_softcap = num(t, "final_logit_softcapping");
    c.rms_norm_eps = num(t, "rms_norm_eps");
    c.qk_clip = num(t, "qk_l2_clip_threshold");
    c.rel_clip = num(t, "rel_l2_clip_threshold");
    c.log_alpha = num(t, "log_scaling_alpha");
    c.log_floor = num(t, "log_scaling_n_floor");
    refuse_if(!(c.log_floor > 0.0), "log_scaling_n_floor <= 0");
    c.attn_res_block = t.contains("n_layers_per_attn_res_block") ? pos_int(t, "n_layers_per_attn_res_block") : 6;
    refuse_if(t.contains("attn_res_normalize_values") && t.at("attn_res_normalize_values").is_boolean() &&
                  !t.at("attn_res_normalize_values").get<bool>(),
              "attn_res_normalize_values false");

    c.loop_start = 6;
    c.loop_end = 18;
    c.loop_passes = 3;
    c.gate_rank = 64;
    if (t.contains("loop_config")) {
      const nlohmann::json& l = t.at("loop_config");
      if (!l.is_object()) throw std::runtime_error(who + "loop_config is not an object");
      refuse_if(l.value("is_varloop", false), "loop_config.is_varloop true");
      c.loop_start = pos_int(l, "core_start");
      c.loop_end = pos_int(l, "core_end");
      c.loop_passes = pos_int(l, "num_passes");
      c.gate_rank = pos_int(l, "gate_rank");
    }
    refuse_if(c.gate_rank > 256, "loop gate_rank > 256 (r4dx_inkling2_msg_gate_apply_f32's limit)");
    if (c.loop_start % c.attn_res_block != 0 || c.loop_end % c.attn_res_block != 0 || c.loop_end <= c.loop_start ||
        c.loop_end > c.num_hidden_layers)
      throw std::runtime_error(who + "loop bounds [" + std::to_string(c.loop_start) + ", " + std::to_string(c.loop_end) +
                               ") must be non-empty multiples of n_layers_per_attn_res_block inside the layer count");

    // Layer kinds.
    const int64_t L = c.num_hidden_layers;
    std::vector<bool> from_types, from_local;
    if (t.contains("layer_types") && t.at("layer_types").is_array()) {
      const auto& lt = t.at("layer_types");
      if (static_cast<int64_t>(lt.size()) != L) throw std::runtime_error(who + "layer_types length != num_hidden_layers");
      for (const auto& k : lt) {
        const std::string s = k.get<std::string>();
        if (s != "hybrid" && s != "hybrid_sliding") throw std::runtime_error(who + "layer type '" + s + "'");
        from_types.push_back(s == "hybrid");
      }
    }
    if (t.contains("local_layer_ids") && t.at("local_layer_ids").is_array()) {
      from_local.assign(static_cast<size_t>(L), true);
      for (const auto& id : t.at("local_layer_ids")) {
        const int64_t i = id.get<int64_t>();
        if (i < 0 || i >= L) throw std::runtime_error(who + "local_layer_ids entry out of range");
        from_local[static_cast<size_t>(i)] = false;
      }
    }
    if (!from_types.empty() && !from_local.empty() && from_types != from_local)
      throw std::runtime_error(who + "layer_types and local_layer_ids disagree");
    if (!from_types.empty()) {
      c.full = from_types;
    } else if (!from_local.empty()) {
      c.full = from_local;
    } else {
      for (int64_t i = 0; i < L; ++i) c.full.push_back((i + 1) % 6 == 0);
    }
    // The attention-residual stack never exceeds the kernels' 6 states (blocks + the partial).
    int64_t max_blocks = 0;
    for (int64_t i = 0; i < L; ++i)
      if (c.IsBoundary(i)) ++max_blocks;
    refuse_if(max_blocks + 1 > 6, "more than 5 attention-residual blocks (r4dx_inkling2_attnres_f32 takes <= 6 states)");

    const auto opt_id = [&](const char* key, int64_t* out) {
      if (t.contains(key) && t.at(key).is_number_integer()) *out = t.at(key).get<int64_t>();
    };
    opt_id("eos_token_id", &c.eos_token_id);
    opt_id("pad_token_id", &c.pad_token_id);
    opt_id("bos_token_id", &c.bos_token_id);
    return c;
  }

  // The rank-local config of `global` for TP world `world`, rank `rank` (docs/inkling2-plan.md 5). Throws
  // std::invalid_argument naming the field unless heads, kv heads and one expert's intermediate divide by `world`, the
  // GQA is unchanged, and the stored lm_head rows split into whole 16-row tiles per rank. Per-layout alignment (mxfp8:
  // whole 16-row tiles and 32-K groups of every rank slice; bf16 needs nothing more) is the loader's check.
  static Inkling2Config Shard(const Inkling2Config& global, int world, int rank) {
    const std::string where =
        "Inkling2Config::Shard(world=" + std::to_string(world) + ", rank=" + std::to_string(rank) + "): ";
    const auto fail = [&](const std::string& what) { throw std::invalid_argument(where + what); };
    const auto require_div = [&](const char* field, int64_t value, int64_t divisor) {
      if (divisor <= 0 || value % divisor != 0)
        fail(std::string(field) + " = " + std::to_string(value) + " is not divisible by " + std::to_string(divisor));
    };
    if (world != 1 && world != 2) fail("world must be 1 or 2");
    if (rank < 0 || rank >= world) fail("rank must be in [0, world)");
    if (global.tp_world != 1) fail("`global` is already a rank shard");
    require_div("num_attention_heads", global.num_attention_heads, world);
    require_div("num_key_value_heads", global.num_key_value_heads, world);
    require_div("intermediate_size (one expert)", global.moe_intermediate_size, world);
    require_div("lm_head stored rows (whole 16-row tiles per rank)", global.HeadRowsStored(), 16 * static_cast<int64_t>(world));
    Inkling2Config r = global;
    r.tp_world = world;
    r.tp_rank = rank;
    r.num_attention_heads = global.num_attention_heads / world;
    r.num_key_value_heads = global.num_key_value_heads / world;
    r.moe_intermediate_size = global.moe_intermediate_size / world;
    if (r.Gqa() != global.Gqa()) fail("the GQA changes");
    return r;
  }
};

}  // namespace r4dx::model
