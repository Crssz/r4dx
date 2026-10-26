// r4dx::model::LlamaContainerInfo -- the CPU half of the dense Llama container loader (docs/nanbeige-plan.md 4, task
// N1-5): everything that can be decided from `__metadata__` and the safetensors tensor DIRECTORY, with no HIP call and
// no tensor byte read. LlamaContainer::Load (llama_container.cpp) runs these checks first and then uploads;
// tests/model/test_llama_container.cpp runs them alone on a tiny converted container (and on the real Nanbeige
// container header when one exists).
//
// Header-only and HIP-free (like lfm2_container_info.h / llama_config.h).
#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "llama_config.h"
#include "model_config.h"
#include "nlohmann/json.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

namespace r4dx::model {

struct LlamaContainerInfo {
  std::string path, model_id, config_sha256;
  nlohmann::json metadata;
  LlamaConfig config;        // GLOBAL config (tp_world 1)
  ModelConfig model_config;  // config.ToModelConfig(): what tp::RuleFor and TextModel::Config() read
  bool has_trellis = false;
  bool has_w8a16 = false;  // __metadata__.quant.w8a16 present (parsed strictly at Load)
  bool has_kv_descale = false;  // every layer carries attn.{k,v}_descale (--kv-calib: fp8 KV can run, N5-2)
};

// Container tensor base names (the converter's llama_layout.hpp is the contract).
inline std::string LlamaLayerBase(int64_t layer) { return "text.layers." + std::to_string(layer) + "."; }

// `metadata` is the container's whole `__metadata__`. Throws std::runtime_error naming `path` unless it is a llama
// container this build can run (model_arch, norm_kind plain, a parseable flat model_config, no rotation).
inline LlamaContainerInfo ParseLlamaContainerMetadata(const nlohmann::json& metadata, const std::string& path) {
  const std::string who = "r4dx::model::LlamaContainer: " + path + ": ";
  LlamaContainerInfo info;
  info.path = path;
  info.metadata = metadata;
  if (!metadata.is_object()) throw std::runtime_error(who + "__metadata__ is not an object");
  if (metadata.value("model_arch", std::string()) != "llama") {
    throw std::runtime_error(who + "__metadata__.model_arch is '" + metadata.value("model_arch", std::string()) +
                             "', not llama");
  }
  if (metadata.value("norm_kind", std::string()) != "plain") {
    throw std::runtime_error(who + "__metadata__.norm_kind must be 'plain' (Llama norms are x * w, never 1 + w)");
  }
  if (!metadata.contains("model_config")) throw std::runtime_error(who + "no __metadata__.model_config");
  if (metadata.contains("rotation")) throw std::runtime_error(who + "a rotated container: the Llama trellis is unrotated");
  info.model_id = metadata.value("model_id", std::string());
  info.config_sha256 = metadata.value("config_sha256", std::string());
  info.config = LlamaConfig::FromModelConfig(metadata.at("model_config"));
  info.model_config = info.config.ToModelConfig();
  info.has_trellis = metadata.contains("quant") && metadata.at("quant").is_object() &&
                     metadata.at("quant").contains("trellis");
  info.has_w8a16 = metadata.contains("quant") && metadata.at("quant").is_object() &&
                   metadata.at("quant").contains("w8a16");
  return info;
}

// The names and shapes a Llama container of `layers` layers must carry. `bf16_body`: the body linears are required to
// carry their `.bf16.w` form with the exact [N, K, 2] shape; otherwise any one of the bf16 / w4a16 / w8a16 / trellis
// forms must exist (each is shape-checked by its own loader). Returns one human-readable problem per mismatch (empty =
// the table is what LlamaContainer::Load will read), including a layer past `layers` that the header says should not
// be there when `layers` is the whole model.
inline std::vector<std::string> LlamaTensorTableProblems(const r4dx_convert::SafetensorsReader& r,
                                                         const LlamaContainerInfo& info, int64_t layers,
                                                         bool bf16_body = true) {
  const LlamaConfig& c = info.config;
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
    return r.Has(base + ".bf16.w") || r.Has(base + ".w4a16.wq") || r.Has(base + ".trellis.w") ||
           r.Has(base + ".w8a16.wq");
  };
  const auto linear = [&](const std::string& base, int64_t N, int64_t K) {
    if (r.Has(base + ".bf16.w") || bf16_body || !has_linear(base)) need(base + ".bf16.w", {N, K, 2});
  };
  const int64_t H = c.hidden_size;
  if (layers < 0 || layers > c.num_hidden_layers) bad.push_back("layer count out of range");
  // fp8 KV descales (--kv-calib): all layers or none, decided by layer 0.
  const bool descales = r.Has(LlamaLayerBase(0) + "attn.k_descale");
  need("text.embed_tokens", {c.vocab_size, H, 2});
  need("text.final_norm", {H, 2});
  if (!(r.Has("lm_head.bf16.w") || r.Has("lm_head.w4a16.wq") || r.Has("lm_head.w8a16.wq")))
    bad.push_back("missing lm_head (bf16, w4a16 or w8a16)");
  else if (r.Has("lm_head.bf16.w")) need("lm_head.bf16.w", {c.vocab_size, H, 2});
  for (int64_t i = 0; i < layers && i < c.num_hidden_layers; ++i) {
    const std::string b = LlamaLayerBase(i);
    need(b + "input_norm", {H, 2});
    need(b + "post_attn_norm", {H, 2});
    linear(b + "attn.q", c.QDim(), H);
    linear(b + "attn.k", c.KvDim(), H);
    linear(b + "attn.v", c.KvDim(), H);
    linear(b + "attn.o", H, c.QDim());
    linear(b + "mlp.gate_up", c.GateUpRows(), H);
    linear(b + "mlp.down", H, c.intermediate_size);
    if (descales) {
      need(b + "attn.k_descale", {c.num_key_value_heads, 4});
      need(b + "attn.v_descale", {c.num_key_value_heads, 4});
    } else if (r.Has(b + "attn.k_descale") || r.Has(b + "attn.v_descale")) {
      bad.push_back("layer " + std::to_string(i) + " has a kv descale but layer 0 has none (all layers or none)");
    }
  }
  return bad;
}

// Reads `path`'s `__metadata__` and tensor directory (no weight byte) and checks the table: the CPU half of
// LlamaContainer::Load. Throws std::runtime_error listing up to 8 problems.
inline LlamaContainerInfo InspectLlamaContainer(const std::string& path, int64_t layer_limit = -1) {
  r4dx_convert::SafetensorsReader reader(r4dx_convert::Utf8ToWide(path));
  const nlohmann::json header = nlohmann::json::parse(reader.HeaderJson());
  if (!header.contains("__metadata__"))
    throw std::runtime_error("r4dx::model::LlamaContainer: " + path + " has no __metadata__");
  LlamaContainerInfo info = ParseLlamaContainerMetadata(header.at("__metadata__"), path);
  const int64_t layers = layer_limit >= 0 ? std::min(layer_limit, info.config.num_hidden_layers)
                                          : info.config.num_hidden_layers;
  const std::vector<std::string> bad = LlamaTensorTableProblems(reader, info, layers, /*bf16_body=*/false);
  if (!bad.empty()) {
    std::string msg = "r4dx::model::LlamaContainer: " + path + ": the tensor table does not match the model (" +
                      std::to_string(bad.size()) + " problem(s)):";
    for (size_t i = 0; i < bad.size() && i < 8; ++i) msg += "\n  " + bad[i];
    throw std::runtime_error(msg);
  }
  info.has_kv_descale = reader.Has(LlamaLayerBase(0) + "attn.k_descale");
  return info;
}

}  // namespace r4dx::model
