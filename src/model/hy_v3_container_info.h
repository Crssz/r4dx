// r4dx::model::HyV3ContainerInfo -- CPU-only header/directory inspection for hy_v3 .r4dx containers.
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#include "hy_v3_config.h"
#include "model_config.h"
#include "nlohmann/json.hpp"

namespace r4dx::model {

struct HyV3ContainerInfo {
  std::string path, model_id;
  nlohmann::json metadata;
  HyV3Config config;
  ModelConfig model_config;
  bool has_w8a16 = false;
};

inline std::string HyV3LayerBase(int64_t layer) { return "text.layers." + std::to_string(layer) + "."; }

inline HyV3ContainerInfo InspectHyV3Container(const std::string& path, int64_t layer_limit = -1) {
  const std::string who = "r4dx::model::HyV3Container: " + path + ": ";
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error(who + "cannot open file");
  uint64_t header_len = 0;
  f.read(reinterpret_cast<char*>(&header_len), 8);
  if (!f || header_len == 0 || header_len > (uint64_t{1} << 30)) {
    throw std::runtime_error(who + "bad safetensors header length");
  }
  std::string header_json(static_cast<size_t>(header_len), '\0');
  f.read(header_json.data(), static_cast<std::streamsize>(header_len));
  if (!f) throw std::runtime_error(who + "truncated header");
  const nlohmann::json header = nlohmann::json::parse(header_json, nullptr, /*allow_exceptions=*/false);
  if (!header.is_object() || !header.contains("__metadata__")) {
    throw std::runtime_error(who + "missing __metadata__");
  }
  const nlohmann::json metadata = header.at("__metadata__");
  if (!metadata.is_object()) throw std::runtime_error(who + "__metadata__ is not an object");
  if (metadata.value("model_arch", std::string()) != "hy_v3") {
    throw std::runtime_error(who + "__metadata__.model_arch is not 'hy_v3'");
  }
  if (!metadata.contains("model_config")) throw std::runtime_error(who + "no __metadata__.model_config");

  HyV3ContainerInfo info;
  info.path = path;
  info.metadata = metadata;
  info.model_id = metadata.value("model_id", std::string());
  info.config = HyV3Config::FromModelConfig(metadata.at("model_config"));
  info.model_config = info.config.ToModelConfig();
  info.has_w8a16 = metadata.contains("quant") && metadata.at("quant").is_object() &&
                   metadata.at("quant").contains("w8a16");
  const int64_t layers = layer_limit >= 0 ? layer_limit : info.config.num_hidden_layers;
  if (layers < 0 || layers > info.config.num_hidden_layers) {
    throw std::runtime_error(who + "layer_limit out of range");
  }
  return info;
}

}  // namespace r4dx::model
