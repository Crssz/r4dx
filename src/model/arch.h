// r4dx::model::Arch -- which text-model family a container holds, decided from its header alone
// (docs/gemma4-plan.md 3.2, task M1-1 / M1-24).
//
// LoadTextModel (tp_model.cpp) calls DetectArch(container_path) before it builds anything: kQwen35
// goes down the unchanged Model / TpModel branches, kGemma4 to LoadGemmaTextModel. The enum is also
// carried on ModelConfig::arch (model_config.h), so a TextModel's Config() tells the server which
// family it loaded (server dialect cross-check, CheckDialectAgainstArch).
//
// CPU-only and HIP-free: arch.cpp reads the safetensors-shell header (8-byte length + JSON), never
// a tensor.
#pragma once

#include <cstdint>
#include <string>

#include "nlohmann/json.hpp"

namespace r4dx::model {

enum class Arch {
  kQwen35,  // Qwen3.5/3.8 hybrid GDN + full attention (every container before Gemma support)
  kGemma4,  // gemma4_unified: sliding + full attention, sandwich norms, GeGLU, tied head
};

// Stable on-disk / log spelling: `__metadata__.model_arch` for Gemma, "qwen3_5" for Qwen (which
// carries no such key). server/dialect.cpp's CheckDialectAgainstArch keys off the "gemma" prefix.
inline const char* ArchName(Arch a) { return a == Arch::kGemma4 ? "gemma4_unified" : "qwen3_5"; }

// The arch a `__metadata__` object describes:
//   1. `model_arch` present: "gemma4_unified" -> kGemma4, "qwen3_5" -> kQwen35, anything else throws
//      std::runtime_error (an unknown family must not be loaded as Qwen);
//   2. otherwise `model_config.model_type` / `model_config.architectures` naming Gemma 4
//      ("gemma4_unified", "Gemma4UnifiedForConditionalGeneration") -> kGemma4;
//   3. otherwise kQwen35: every container written before `model_arch` existed (and the selftest
//      containers, which have no model_config at all).
Arch DetectArchFromMetadata(const nlohmann::json& metadata);

// DetectArchFromMetadata of `container_path`'s `__metadata__`. A file that cannot be opened, is not
// a safetensors shell, or has no `__metadata__` yields kQwen35: that is today's behaviour, and the
// Qwen loader then reports its own (unchanged) error for the same file.
Arch DetectArch(const std::string& container_path);

// `container_path`'s `__metadata__` object (the safetensors-shell header only, never a tensor), or a
// null json when the file cannot be opened / is not a shell / has no `__metadata__`.
nlohmann::json ReadContainerMetadata(const std::string& container_path);

// The `max_ctx` a run on `container_path` gets (docs/gemma4-plan.md section 9.1, task M1-14), shared by
// r4dx-cli and r4dx-server so the two cannot drift:
//   * kQwen35: `requested` unchanged -- the callers' own default (262144) and every explicit value
//     behave exactly as before this function existed;
//   * kGemma4: GemmaConfig::ResolveMaxCtx on the container's text_config. `requested_given` false means
//     the user passed no --max-ctx: the default is the checkpoint's max_position_embeddings (131072
//     for Huihui), or kExtendedMaxCtx (262144) when `extended_ctx` (the opt-in flag) is set. An explicit
//     value above the checkpoint's own limit throws std::invalid_argument unless `extended_ctx`.
// Throws std::runtime_error when a Gemma container's model_config does not parse.
int64_t ResolveContainerMaxCtx(const std::string& container_path, Arch arch, bool requested_given,
                               int64_t requested, bool extended_ctx);

}  // namespace r4dx::model
