// r4dx::model::Lfm2ContainerInfo -- the CPU half of the LFM2 container loader (docs/lfm2-plan.md 4, task L1-3):
// everything that can be decided from `__metadata__` and the safetensors tensor DIRECTORY, with no HIP call and no
// tensor byte read. Lfm2Container::Load (lfm2_container.cpp) runs these checks first and then uploads;
// tests/model/test_lfm2_container.cpp runs them alone on the tiny converter fixture (and on the real container
// header when one exists).
//
// Header-only and HIP-free (like gemma_container_info.h / lfm2_config.h).
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "lfm2_config.h"
#include "model_config.h"
#include "nlohmann/json.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

namespace r4dx::model {

struct Lfm2ContainerInfo {
  std::string path, model_id, config_sha256;
  nlohmann::json metadata;
  Lfm2Config config;         // GLOBAL config (tp_world 1)
  ModelConfig model_config;  // config.ToModelConfig(): what tp::RuleFor and TextModel::Config() read
  bool has_trellis = false;
};

// Container tensor base names (the converter's lfm2_layout.hpp is the contract).
inline std::string Lfm2LayerBase(int64_t layer) { return "text.layers." + std::to_string(layer) + "."; }
inline std::string Lfm2ExpertBase(int64_t layer, int64_t expert, const char* part /* "gate_up" | "down" */) {
  return Lfm2LayerBase(layer) + "moe.experts." + std::to_string(expert) + "." + part;
}

// `metadata` is the container's whole `__metadata__`. Throws std::runtime_error naming `path` unless it is an
// lfm2_moe container this build can run (model_arch, norm_kind plain, a parseable flat model_config, no rotation).
inline Lfm2ContainerInfo ParseLfm2ContainerMetadata(const nlohmann::json& metadata, const std::string& path) {
  const std::string who = "r4dx::model::Lfm2Container: " + path + ": ";
  Lfm2ContainerInfo info;
  info.path = path;
  info.metadata = metadata;
  if (!metadata.is_object()) throw std::runtime_error(who + "__metadata__ is not an object");
  if (metadata.value("model_arch", std::string()) != "lfm2_moe") {
    throw std::runtime_error(who + "__metadata__.model_arch is '" + metadata.value("model_arch", std::string()) +
                             "', not lfm2_moe");
  }
  if (metadata.value("norm_kind", std::string()) != "plain") {
    throw std::runtime_error(who + "__metadata__.norm_kind must be 'plain' (LFM2 norms are x * w, never 1 + w)");
  }
  if (!metadata.contains("model_config")) throw std::runtime_error(who + "no __metadata__.model_config");
  if (metadata.contains("rotation")) throw std::runtime_error(who + "a rotated container: LFM2 has no quant2 rotation");
  info.model_id = metadata.value("model_id", std::string());
  info.config_sha256 = metadata.value("config_sha256", std::string());
  info.config = Lfm2Config::FromModelConfig(metadata.at("model_config"));
  info.model_config = info.config.ToModelConfig();
  info.has_trellis = metadata.contains("quant") && metadata.at("quant").is_object() &&
                     metadata.at("quant").contains("trellis");
  return info;
}

// The names and shapes an LFM2 container of `layers` layers must carry. `bf16_body`: the body linears are required to
// carry their `.bf16.w` form with the exact [N, K, 2] shape; otherwise any one of the bf16 / w4a16 / trellis forms must
// exist. Returns one human-readable problem per mismatch (empty = the table is what Lfm2Container::Load will read),
// including tensors on the wrong kind of layer (attn.* on a conv layer, a dense mlp on an MoE layer, an expert past
// num_experts).
inline std::vector<std::string> Lfm2TensorTableProblems(const r4dx_convert::SafetensorsReader& r,
                                                        const Lfm2ContainerInfo& info, int64_t layers,
                                                        bool bf16_body = true) {
  const Lfm2Config& c = info.config;
  std::vector<std::string> bad;
  const auto shape_str = [](const std::vector<int64_t>& s) {
    std::string out = "[";
    for (size_t i = 0; i < s.size(); ++i) out += (i ? "," : "") + std::to_string(s[i]);
    return out + "]";
  };
  const auto need = [&](const std::string& n, std::vector<int64_t> shape) {
    if (!r.Has(n)) {
      bad.push_back("missing tensor '" + n + "'");
      return;
    }
    if (r.Meta(n).shape != shape)
      bad.push_back("tensor '" + n + "' has shape " + shape_str(r.Meta(n).shape) + ", expected " + shape_str(shape));
  };
  const auto has_linear = [&](const std::string& base) {
    return r.Has(base + ".bf16.w") || r.Has(base + ".w4a16.wq") || r.Has(base + ".trellis.w");
  };
  const auto linear = [&](const std::string& base, int64_t N, int64_t K) {
    // A bf16 form on disk is always shape-checked; other forms (w4a16 / trellis) are checked by their own loaders.
    if (r.Has(base + ".bf16.w") || bf16_body || !has_linear(base)) need(base + ".bf16.w", {N, K, 2});
  };
  const auto absent = [&](const std::string& base, const std::string& why) {
    if (has_linear(base) || r.Has(base)) bad.push_back("'" + base + "' is present but " + why);
  };
  const int64_t H = c.hidden_size;
  if (layers < 0 || layers > c.num_hidden_layers) bad.push_back("layer count out of range");
  need("text.embed_tokens", {c.vocab_size, H, 2});
  need("text.final_norm", {H, 2});
  if (!(r.Has("lm_head.bf16.w") || r.Has("lm_head.w4a16.wq"))) bad.push_back("missing lm_head (bf16 or w4a16)");
  else if (r.Has("lm_head.bf16.w")) need("lm_head.bf16.w", {c.vocab_size, H, 2});
  for (int64_t i = 0; i < layers && i < c.num_hidden_layers; ++i) {
    const std::string b = Lfm2LayerBase(i);
    const bool attn = c.IsAttentionLayer(i), dense = c.IsDenseLayer(i);
    need(b + "operator_norm", {H, 2});
    need(b + "ffn_norm", {H, 2});
    if (attn) {
      linear(b + "attn.q", c.QDim(), H);
      linear(b + "attn.k", c.KvDim(), H);
      linear(b + "attn.v", c.KvDim(), H);
      linear(b + "attn.o", H, c.QDim());
      need(b + "attn.q_norm", {c.head_dim, 2});
      need(b + "attn.k_norm", {c.head_dim, 2});
      absent(b + "conv.in_proj", "layer " + std::to_string(i) + " is an attention layer");
      absent(b + "conv.weight", "layer " + std::to_string(i) + " is an attention layer");
    } else {
      linear(b + "conv.in_proj", c.ConvInProjRows(), H);
      linear(b + "conv.out_proj", H, c.conv_dim);
      need(b + "conv.weight", {c.conv_dim, c.conv_l_cache, 2});
      absent(b + "attn.q", "layer " + std::to_string(i) + " is a ShortConv layer");
    }
    if (dense) {
      linear(b + "mlp.gate_up", 2 * c.intermediate_size, H);
      linear(b + "mlp.down", H, c.intermediate_size);
      absent(b + "moe.router", "layer " + std::to_string(i) + " is a dense-FFN layer");
    } else {
      need(b + "moe.router", {c.num_experts, H, 2});
      need(b + "moe.expert_bias", {c.num_experts, 4});  // fp32: the selection bias is an F32 buffer
      for (int64_t e = 0; e < c.num_experts; ++e) {
        linear(Lfm2ExpertBase(i, e, "gate_up"), c.ExpertGateUpRows(), H);
        linear(Lfm2ExpertBase(i, e, "down"), H, c.moe_intermediate_size);
      }
      absent(Lfm2ExpertBase(i, c.num_experts, "gate_up"), "the layer has only " + std::to_string(c.num_experts) + " experts");
      absent(b + "mlp.gate_up", "layer " + std::to_string(i) + " is an MoE layer");
    }
  }
  return bad;
}

// Reads `path`'s `__metadata__` and tensor directory (no weight byte) and checks the table: the CPU half of
// Lfm2Container::Load. Throws std::runtime_error listing up to 8 problems.
inline Lfm2ContainerInfo InspectLfm2Container(const std::string& path, int64_t layer_limit = -1) {
  r4dx_convert::SafetensorsReader reader(r4dx_convert::Utf8ToWide(path));
  const nlohmann::json header = nlohmann::json::parse(reader.HeaderJson());
  if (!header.contains("__metadata__"))
    throw std::runtime_error("r4dx::model::Lfm2Container: " + path + " has no __metadata__");
  Lfm2ContainerInfo info = ParseLfm2ContainerMetadata(header.at("__metadata__"), path);
  const int64_t layers = layer_limit >= 0 ? std::min(layer_limit, info.config.num_hidden_layers)
                                          : info.config.num_hidden_layers;
  const std::vector<std::string> bad = Lfm2TensorTableProblems(reader, info, layers, /*bf16_body=*/false);
  if (!bad.empty()) {
    std::string msg = "r4dx::model::Lfm2Container: " + path + ": the tensor table does not match the model (" +
                      std::to_string(bad.size()) + " problem(s)):";
    for (size_t i = 0; i < bad.size() && i < 8; ++i) msg += "\n  " + bad[i];
    throw std::runtime_error(msg);
  }
  return info;
}

}  // namespace r4dx::model
