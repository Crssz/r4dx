#include "arch.h"

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

namespace r4dx::model {

namespace {

bool IsGemma4Name(const nlohmann::json& v) {
  if (!v.is_string()) return false;
  const std::string s = v.get<std::string>();
  return s == "gemma4_unified" || s == "Gemma4UnifiedForConditionalGeneration";
}

}  // namespace

Arch DetectArchFromMetadata(const nlohmann::json& metadata) {
  if (!metadata.is_object()) return Arch::kQwen35;
  if (metadata.contains("model_arch")) {
    const nlohmann::json& a = metadata.at("model_arch");
    const std::string s = a.is_string() ? a.get<std::string>() : std::string();
    if (s == ArchName(Arch::kGemma4)) return Arch::kGemma4;
    if (s == ArchName(Arch::kQwen35)) return Arch::kQwen35;
    throw std::runtime_error("r4dx::model::DetectArch: unknown __metadata__.model_arch " + a.dump() +
                             " (this build knows 'gemma4_unified' and 'qwen3_5')");
  }
  if (metadata.contains("model_config") && metadata.at("model_config").is_object()) {
    const nlohmann::json& mc = metadata.at("model_config");
    if (mc.contains("model_type") && IsGemma4Name(mc.at("model_type"))) return Arch::kGemma4;
    if (mc.contains("architectures") && mc.at("architectures").is_array()) {
      for (const auto& a : mc.at("architectures"))
        if (IsGemma4Name(a)) return Arch::kGemma4;
    }
  }
  return Arch::kQwen35;
}

Arch DetectArch(const std::string& container_path) {
  std::ifstream f(container_path, std::ios::binary);
  if (!f) return Arch::kQwen35;
  uint64_t header_len = 0;
  f.read(reinterpret_cast<char*>(&header_len), 8);
  // A safetensors header is a JSON object of a few hundred KB at most here (Gemma 12B: ~90 KB). A
  // length past 1 GiB is not one: leave the file to the Qwen loader's own diagnostics.
  if (!f || header_len == 0 || header_len > (uint64_t{1} << 30)) return Arch::kQwen35;
  std::string header_json(static_cast<size_t>(header_len), '\0');
  f.read(header_json.data(), static_cast<std::streamsize>(header_len));
  if (!f) return Arch::kQwen35;
  const nlohmann::json header = nlohmann::json::parse(header_json, nullptr, /*allow_exceptions=*/false);
  if (!header.is_object() || !header.contains("__metadata__")) return Arch::kQwen35;
  return DetectArchFromMetadata(header.at("__metadata__"));
}

}  // namespace r4dx::model
