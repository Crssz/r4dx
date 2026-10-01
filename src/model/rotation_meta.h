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

// The Qwen geometry (docs/quant2.md section 0): 5120 = 5 x 1024, with the three Hadamard blocks
// chosen so every tensor-parallel rank's K is a whole number of blocks. Other hidden sizes (Gemma 4's
// 3840 = 15 x 256) are accepted too -- r4dx_rotate_residual_bf16 derives the geometry from `hidden` --
// but only with the block the converter would choose (RotationBlockFor).
inline constexpr int64_t kRotationHidden = 5120;
inline constexpr int64_t kRotationBlock = 1024;   // also the cap of RotationBlockFor
inline constexpr int64_t kRotationMinBlock = 64;  // the converter refuses anything smaller
inline constexpr int64_t kRotationMaxNblk = 32;   // the kernels' limit
inline constexpr int64_t kHadDownBlock = 512;     // mlp.down: 34 blocks, 17 per TP=2 rank
inline constexpr int64_t kHadOBlock = 256;        // attn.o: one block per attention head (head_dim)
inline constexpr int64_t kHadGdnOutBlock = 128;   // gdn.out_proj: one block per GDN value head
inline constexpr int64_t kHadOFullBlock = 256;    // Gemma 4 full-attention o_proj (head_dim 512: 2 blocks)

// The residual rotation's block for a hidden width: the largest power of two dividing it, capped at
// kRotationBlock (src/convert rotation.hpp ChooseRotationBlock, which writes it). 5120 -> 1024,
// 3840 -> 256. 0 for hidden <= 0.
inline int64_t RotationBlockFor(int64_t hidden) {
  if (hidden <= 0) return 0;
  const int64_t low = hidden & -hidden;
  return low < kRotationBlock ? low : kRotationBlock;
}

inline constexpr char kRotationSigns[] = "rotation.signs";
inline constexpr char kRotationMix5[] = "rotation.mix5";  // nblk == 5 (Qwen); any other nblk: kRotationMix
inline constexpr char kRotationMix[] = "rotation.mix";
inline constexpr char kRotationHadDownSigns[] = "rotation.had_down_signs";
inline constexpr char kRotationHadOSigns[] = "rotation.had_o_signs";
inline constexpr char kRotationHadGdnOutSigns[] = "rotation.had_gdn_out_signs";
inline constexpr char kRotationHadOFullSigns[] = "rotation.had_o_full_signs";  // Gemma 4 only

struct RotationSpec {
  RotationKind kind = RotationKind::kQ2a;
  uint64_t seed = 0;  // informational: the runtime reads the tensors, it never regenerates them
  int64_t hidden = kRotationHidden;
  int64_t block = kRotationBlock;
  int64_t nblk = kRotationHidden / kRotationBlock;  // hidden / block
  // Gemma 4 "option A" (docs/gemma4-plan.md 4.4, metadata `out_fold: "had_only"`): o_proj / down_proj
  // carry only W Hb, their outputs stay in the original basis, and the runtime rotates AFTER the post
  // norm (r4dx_post_rmsnorm_rotate_add_bf16). False for every Qwen container.
  bool post_norm_rotate = false;
  bool has_gdn_out = true;  // q2ab: Qwen carries had_gdn_out_signs, Gemma does not
  bool has_o_full = false;  // q2ab: Gemma carries had_o_full_signs (full-attention o_proj)
  bool Hadamard() const { return kind == RotationKind::kQ2ab; }
  // The block-mixing matrix tensor: rotation.mix5 for the Qwen 5-block geometry, rotation.mix otherwise.
  const char* MixName() const { return nblk == 5 ? kRotationMix5 : kRotationMix; }
};

inline const char* RotationKindName(RotationKind k) {
  return k == RotationKind::kQ2ab ? "q2ab" : "q2a";
}

// One rotation.* tensor a rotated container must carry: its name and fp32 element count for the
// given config (the GLOBAL config for the on-disk tensor, a ModelConfig::Shard for a rank's slice --
// the Hadamard sign vectors shrink with their linear's K, signs/mix do not).
struct RotationTensor {
  const char* name;
  int64_t elems;
};

// Order: signs, mix (mix5), then (q2ab) had_down, had_o, [had_gdn_out], [had_o_full]. `o_full_elems` is
// the full-attention o_proj K (Gemma: heads * global_head_dim); ModelConfig has no such field, so a
// spec with has_o_full needs the caller to supply it.
inline std::vector<RotationTensor> RotationTensors(const RotationSpec& spec, const ModelConfig& cfg,
                                                   int64_t o_full_elems = 0) {
  std::vector<RotationTensor> t = {{kRotationSigns, cfg.hidden_size},
                                   {spec.MixName(), spec.nblk * spec.nblk}};
  if (spec.Hadamard()) {
    t.push_back({kRotationHadDownSigns, cfg.intermediate_size});
    t.push_back({kRotationHadOSigns, cfg.num_attention_heads * cfg.head_dim});
    if (spec.has_gdn_out) t.push_back({kRotationHadGdnOutSigns, cfg.ValueDim()});
    if (spec.has_o_full) t.push_back({kRotationHadOFullSigns, o_full_elems});
  }
  return t;
}

// `metadata`: the container's whole `__metadata__`. `global`: the container's own (unsharded)
// text config. Returns nullopt when there is no `rotation` key (an unrotated container -- every
// container written before quant2 and every `--rotate none` one). Throws std::runtime_error naming
// `path` for a kind this binary does not implement (docs/quant2.md section 3's early draft name
// "hadamard1024x5" included: no converter ever wrote it) and for any field that disagrees with the
// contract or with the model's shape.
//
// `allow_post_norm_rotate`: whether the caller implements Gemma's option A (`out_fold: "had_only"`,
// RotationSpec::post_norm_rotate). The Qwen loader (Container::Load) leaves it false, so a Gemma
// rotated container is refused there instead of being run with the wrong out-projection contract.
// `global_head_dim`: Gemma's full-attention head width (0 = unknown); with it, `had.o_full` is also
// checked to tile the full-attention o_proj K.
inline std::optional<RotationSpec> ParseRotationMetadata(const nlohmann::json& metadata,
                                                         const ModelConfig& global,
                                                         const std::string& path,
                                                         bool allow_post_norm_rotate = false,
                                                         int64_t global_head_dim = 0) {
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
  // Any hidden, but only with the block the converter chooses for it (the kernel derives the same
  // geometry from `hidden` alone, so a container that says otherwise was folded with a different Q).
  if (hidden <= 0 || block != RotationBlockFor(hidden) || block < kRotationMinBlock) {
    fail("says hidden " + std::to_string(hidden) + " / block " + std::to_string(block) +
         ", but this runtime implements block = the largest power of two dividing hidden, capped at " +
         std::to_string(kRotationBlock) + " (and >= " + std::to_string(kRotationMinBlock) + ")");
  }
  const int64_t nblk = hidden / block;
  // "nblk" is written only when it is not 5, so the Qwen geometry's header stays exactly as it was.
  if (r.contains("nblk")) {
    const int64_t got = int_field(r, "nblk", "");
    if (got != nblk) {
      fail("says nblk " + std::to_string(got) + " but hidden " + std::to_string(hidden) + " / block " +
           std::to_string(block) + " is " + std::to_string(nblk) + " blocks");
    }
  } else if (nblk != 5) {
    fail("has no \"nblk\" but hidden " + std::to_string(hidden) + " / block " + std::to_string(block) +
         " is " + std::to_string(nblk) + " blocks (only the 5-block geometry may omit it)");
  }
  if (nblk > kRotationMaxNblk) {
    fail("has " + std::to_string(nblk) + " blocks, the kernels implement at most " +
         std::to_string(kRotationMaxNblk));
  }
  if (global.hidden_size != hidden) {
    fail("says hidden " + std::to_string(hidden) + " but the model's hidden_size is " +
         std::to_string(global.hidden_size));
  }
  spec.hidden = hidden;
  spec.block = block;
  spec.nblk = nblk;
  if (r.contains("out_fold")) {
    if (!r.at("out_fold").is_string() || r.at("out_fold").get<std::string>() != "had_only") {
      fail("has an \"out_fold\" other than \"had_only\"");
    }
    if (!allow_post_norm_rotate) {
      fail("is Gemma option A (out_fold had_only: o/down outputs stay in the original basis and the "
           "runtime rotates after the post-norm), which this loader does not implement");
    }
    spec.post_norm_rotate = true;
  }
  if (!spec.Hadamard()) {
    if (r.contains("had")) fail("is q2a but carries a q2ab \"had\" block");
    return spec;
  }
  if (!r.contains("had") || !r.at("had").is_object()) fail("is q2ab but has no \"had\" object");
  const nlohmann::json& had = r.at("had");
  const int64_t down = int_field(had, "down", "had.");
  const int64_t o = int_field(had, "o", "had.");
  if (spec.post_norm_rotate) {
    // Gemma 4: down 512, o (sliding layers, one 256 block per head), o_full 256 (two blocks per
    // 512-wide head), no GDN.
    if (had.contains("gdn_out")) fail("has a \"had.gdn_out\" block, but option A models have no GDN");
    const int64_t o_full = int_field(had, "o_full", "had.");
    if (down != kHadDownBlock || o != kHadOBlock || o_full != kHadOFullBlock) {
      fail("had blocks are down " + std::to_string(down) + " / o " + std::to_string(o) + " / o_full " +
           std::to_string(o_full) + ", but this runtime implements 512 / 256 / 256");
    }
    if (global.head_dim != kHadOBlock || global.intermediate_size % kHadDownBlock != 0 ||
        (global_head_dim > 0 && global_head_dim % kHadOFullBlock != 0)) {
      fail("is q2ab but the model has head_dim " + std::to_string(global.head_dim) +
           ", global_head_dim " + std::to_string(global_head_dim) + " and intermediate_size " +
           std::to_string(global.intermediate_size) +
           " (needs 256, a multiple of 256 (when known) and a multiple of 512)");
    }
    spec.has_gdn_out = false;
    spec.has_o_full = true;
    return spec;
  }
  if (had.contains("o_full")) fail("has a \"had.o_full\" block, which only option A (out_fold) models have");
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
