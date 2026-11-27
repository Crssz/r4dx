// r4dx::model::HyV3Config -- geometry of Tencent's hy_v3 (Hy-MT3-30B-A3B), parsed from
// __metadata__.model_config. Header-only and HIP-free.
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "model_config.h"
#include "nlohmann/json.hpp"

namespace r4dx::model {

struct HyV3Config {
  static constexpr int64_t kDefaultMaxCtx = 32768;

  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t intermediate_size = 0;      // dense FFN (layers < first_k_dense_replace)
  int64_t moe_intermediate_size = 0;  // per-expert and shared-expert FFN
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;
  int64_t head_dim = 0;
  int64_t num_experts = 0;
  int64_t num_experts_per_tok = 0;
  int64_t num_shared_experts = 1;
  int64_t first_k_dense_replace = 1;
  double router_scaling_factor = 1.0;
  bool route_norm = true;
  bool qk_norm = true;

  int64_t vocab_size = 0;
  int64_t max_position_embeddings = 262144;
  double norm_eps = 1e-5;
  double rope_theta = 11158840.0;

  int64_t bos_token_id = 120000;
  int64_t eos_token_id = 120025;
  int64_t pad_token_id = 120002;

  bool IsDenseLayer(int64_t i) const { return i >= 0 && i < first_k_dense_replace; }
  bool IsMoeLayer(int64_t i) const { return i >= first_k_dense_replace && i < num_hidden_layers; }
  int64_t Gqa() const { return num_attention_heads / num_key_value_heads; }
  int64_t QDim() const { return num_attention_heads * head_dim; }
  int64_t KvDim() const { return num_key_value_heads * head_dim; }
  int64_t DenseGateUpRows() const { return 2 * intermediate_size; }
  int64_t ExpertGateUpRows() const { return 2 * moe_intermediate_size; }

  int64_t ResolveMaxCtx(int64_t requested) const {
    const int64_t native = max_position_embeddings;
    if (requested <= 0) return std::min(kDefaultMaxCtx, native);
    if (requested <= native) return requested;
    throw std::invalid_argument("HyV3Config::ResolveMaxCtx: " + std::to_string(requested) +
                                " exceeds max_position_embeddings (" + std::to_string(native) + ")");
  }

  ModelConfig ToModelConfig() const {
    ModelConfig m;
    m.arch = Arch::kHyV3;
    m.hidden_size = hidden_size;
    m.num_hidden_layers = num_hidden_layers;
    m.num_attention_heads = num_attention_heads;
    m.num_key_value_heads = num_key_value_heads;
    m.head_dim = head_dim;
    m.intermediate_size = intermediate_size;
    m.moe_intermediate_size = moe_intermediate_size;
    m.num_experts = num_experts;
    m.num_dense_layers = first_k_dense_replace;
    m.vocab_size = vocab_size;
    m.rms_norm_eps = norm_eps;
    m.rope_theta = rope_theta;
    return m;
  }

  static HyV3Config FromModelConfig(const nlohmann::json& mc) {
    const std::string who = "HyV3Config::FromModelConfig: ";
    if (!mc.is_object()) throw std::runtime_error(who + "model_config is not an object");
    auto req_int = [&](const char* k) -> int64_t {
      if (!mc.contains(k) || !mc.at(k).is_number_integer())
        throw std::runtime_error(who + "missing/non-integer '" + k + "'");
      return mc.at(k).get<int64_t>();
    };
    HyV3Config c;
    c.hidden_size = req_int("hidden_size");
    c.num_hidden_layers = req_int("num_hidden_layers");
    c.intermediate_size = req_int("intermediate_size");
    c.moe_intermediate_size = req_int("moe_intermediate_size");
    c.num_attention_heads = req_int("num_attention_heads");
    c.num_key_value_heads = req_int("num_key_value_heads");
    c.head_dim = mc.contains("head_dim") ? req_int("head_dim") : (c.hidden_size / c.num_attention_heads);
    c.num_experts = req_int("num_experts");
    c.num_experts_per_tok = req_int("num_experts_per_tok");
    c.num_shared_experts = mc.value("num_shared_experts", int64_t{1});
    c.first_k_dense_replace = mc.value("first_k_dense_replace", int64_t{1});
    c.router_scaling_factor = mc.value("router_scaling_factor", 1.0);
    c.route_norm = mc.value("route_norm", true);
    c.qk_norm = mc.value("qk_norm", true);
    c.vocab_size = req_int("vocab_size");
    c.max_position_embeddings = mc.value("max_position_embeddings", int64_t{262144});
    c.norm_eps = mc.value("rms_norm_eps", 1e-5);
    c.rope_theta = mc.value("rope_theta", 11158840.0);
    c.bos_token_id = mc.value("bos_token_id", int64_t{120000});
    c.eos_token_id = mc.value("eos_token_id", int64_t{120025});
    c.pad_token_id = mc.value("pad_token_id", int64_t{120002});
    if (c.hidden_size <= 0 || c.num_hidden_layers <= 0 || c.num_attention_heads <= 0 ||
        c.num_key_value_heads <= 0 || c.num_attention_heads % c.num_key_value_heads != 0 || c.head_dim <= 0 ||
        c.head_dim % 2 != 0 || c.num_experts <= 0 || c.num_experts_per_tok <= 0) {
      throw std::runtime_error(who + "invalid geometry");
    }
    return c;
  }
};

}  // namespace r4dx::model
