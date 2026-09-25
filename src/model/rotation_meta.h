// r4dx::model quant2 residual rotation -- the loader half of `__metadata__.rotation` (docs/quant2.md
// sections 3, 3.1 and 4; docs/container-format.md "Residual rotation"). The converter half is
// src/convert/include/r4dx_convert/rotation.hpp + RotationSource in src/convert/main.cpp.
//
// HIP-free and header-only on purpose: tests/model/test_rotation_meta.cpp checks the parse and every
// refusal without a device, and Container::Load (both the TP=1 path and the tensor-parallel shard
// path) is the only production caller. Nothing here touches tensor bytes -- this only decides
// WHETHER a container is rotated, which tensors it must then carry and how long each one is.
//
// The rule both directions of the contract hang on: a container without the `rotation` key gets no
// rotation op anywhere (ParseRotationMetadata returns nullopt, and every call site in Model degrades
// to exactly the pre-quant2 call), and a container whose rotation this binary does not implement is
// refused at load time -- its folded weights are only correct together with the online ops, and read
// as an ordinary container they produce garbage with no error.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "model_config.h"
#include "nlohmann/json.hpp"

namespace r4dx::model {

enum class RotationKind {
  kQ2a,   // residual Q only (docs/quant2.md section 3)
  kQ2ab,  // + online blockwise Hadamard on the mlp.down / attn.o / gdn.out_proj inputs (section 4)
};

// The contract's fixed geometry -- the only one the online kernels implement:
// r4dx_rotate_residual_bf16 is specialized for 5120 = 5 x 1024, and the three Hadamard blocks are
// chosen so every tensor-parallel rank's K is a whole number of blocks (docs/quant2.md section 0).
inline constexpr int64_t kRotationHidden = 5120;
inline constexpr int64_t kRotationBlock = 1024;
inline constexpr int64_t kHadDownBlock = 512;   // mlp.down: 34 blocks, 17 per TP=2 rank
inline constexpr int64_t kHadOBlock = 256;      // attn.o: one block per attention head (head_dim)
inline constexpr int64_t kHadGdnOutBlock = 128;  // gdn.out_proj: one block per GDN value head

inline constexpr char kRotationSigns[] = "rotation.signs";
inline constexpr char kRotationMix5[] = "rotation.mix5";
inline constexpr char kRotationHadDownSigns[] = "rotation.had_down_signs";
inline constexpr char kRotationHadOSigns[] = "rotation.had_o_signs";
inline constexpr char kRotationHadGdnOutSigns[] = "rotation.had_gdn_out_signs";

struct RotationSpec {
  RotationKind kind = RotationKind::kQ2a;
  uint64_t seed = 0;  // informational: the runtime reads the tensors, it never regenerates them
  bool Hadamard() const { return kind == RotationKind::kQ2ab; }
};

inline const char* RotationKindName(RotationKind k) {
  return k == RotationKind::kQ2ab ? "q2ab" : "q2a";
}

// One rotation.* tensor a rotated container must carry: its name and fp32 element count for the
// given config (the GLOBAL config for the on-disk tensor, a ModelConfig::Shard for a rank's slice --
// the three Hadamard sign vectors shrink with their linear's K, signs/mix5 do not).
struct RotationTensor {
  const char* name;
  int64_t elems;
};

inline std::vector<RotationTensor> RotationTensors(const RotationSpec& spec, const ModelConfig& cfg) {
  std::vector<RotationTensor> t = {{kRotationSigns, cfg.hidden_size}, {kRotationMix5, 25}};
  if (spec.Hadamard()) {
    t.push_back({kRotationHadDownSigns, cfg.intermediate_size});
    t.push_back({kRotationHadOSigns, cfg.num_attention_heads * cfg.head_dim});
    t.push_back({kRotationHadGdnOutSigns, cfg.ValueDim()});
  }
  return t;
}

// `metadata`: the container's whole `__metadata__`. `global`: the container's own (unsharded)
// text config. Returns nullopt when there is no `rotation` key (an unrotated container -- every
// container written before quant2 and every `--rotate none` one). Throws std::runtime_error naming
// `path` for a kind this binary does not implement (docs/quant2.md section 3's early draft name
// "hadamard1024x5" included: no converter ever wrote it) and for any field that disagrees with the
// contract or with the model's shape.
inline std::optional<RotationSpec> ParseRotationMetadata(const nlohmann::json& metadata,
                                                         const ModelConfig& global,
                                                         const std::string& path) {
  if (!metadata.is_object() || !metadata.contains("rotation")) return std::nullopt;
  const auto fail = [&](const std::string& what) {
    throw std::runtime_error("r4dx::model::Container: " + path + ": __metadata__.rotation " + what +
                             " -- refusing to load: a rotated container's weights are only correct "
                             "together with the matching online ops (docs/quant2.md section 3.1)");
  };
  const nlohmann::json& r = metadata.at("rotation");
  if (!r.is_object()) fail("is not an object");
  if (!r.contains("kind") || !r.at("kind").is_string()) fail("has no string \"kind\"");
  const std::string kind = r.at("kind").get<std::string>();
  RotationSpec spec;
  if (kind == "q2a") {
    spec.kind = RotationKind::kQ2a;
  } else if (kind == "q2ab") {
    spec.kind = RotationKind::kQ2ab;
  } else {
    fail("kind \"" + kind + "\" is not one this binary implements (q2a, q2ab)");
  }
  // nlohmann keeps a non-negative integer parsed from text as unsigned, but one built in C++ from a
  // signed literal as signed -- accept either, as long as it is not negative.
  if (!r.contains("seed") || !r.at("seed").is_number_integer() ||
      (!r.at("seed").is_number_unsigned() && r.at("seed").get<int64_t>() < 0)) {
    fail("has no unsigned integer \"seed\"");
  }
  spec.seed = r.at("seed").get<uint64_t>();
  const auto int_field = [&](const nlohmann::json& obj, const char* key, const std::string& where) {
    if (!obj.contains(key) || !obj.at(key).is_number_integer()) {
      fail("has no integer \"" + where + key + "\"");
    }
    return obj.at(key).get<int64_t>();
  };
  const int64_t hidden = int_field(r, "hidden", "");
  const int64_t block = int_field(r, "block", "");
  if (hidden != kRotationHidden || block != kRotationBlock) {
    fail("says hidden " + std::to_string(hidden) + " / block " + std::to_string(block) +
         ", but this runtime implements hidden 5120 = 5 x 1024 only");
  }
  if (global.hidden_size != hidden) {
    fail("says hidden " + std::to_string(hidden) + " but the model's hidden_size is " +
         std::to_string(global.hidden_size));
  }
  if (!spec.Hadamard()) {
    if (r.contains("had")) fail("is q2a but carries a q2ab \"had\" block");
    return spec;
  }
  if (!r.contains("had") || !r.at("had").is_object()) fail("is q2ab but has no \"had\" object");
  const nlohmann::json& had = r.at("had");
  const int64_t down = int_field(had, "down", "had.");
  const int64_t o = int_field(had, "o", "had.");
  const int64_t gdn_out = int_field(had, "gdn_out", "had.");
  if (down != kHadDownBlock || o != kHadOBlock || gdn_out != kHadGdnOutBlock) {
    fail("had blocks are down " + std::to_string(down) + " / o " + std::to_string(o) +
         " / gdn_out " + std::to_string(gdn_out) + ", but this runtime implements 512 / 256 / 128");
  }
  // The runtime applies the attn.o and gdn.out_proj Hadamards one HEAD per block (the fused gate
  // multiply and the in-place kernel after the gated norm both work per head), so the blocks must be
  // exactly the head dims; down's block must tile the intermediate width.
  if (global.head_dim != kHadOBlock || global.linear_value_head_dim != kHadGdnOutBlock ||
      global.intermediate_size % kHadDownBlock != 0) {
    fail("is q2ab but the model has head_dim " + std::to_string(global.head_dim) +
         ", linear_value_head_dim " + std::to_string(global.linear_value_head_dim) +
         " and intermediate_size " + std::to_string(global.intermediate_size) +
         " (needs 256, 128 and a multiple of 512)");
  }
  return spec;
}

}  // namespace r4dx::model
