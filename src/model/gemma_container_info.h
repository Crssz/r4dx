// r4dx::model::GemmaContainerInfo -- the CPU half of the Gemma 4 container loader (docs/gemma4-plan.md
// M1-19): everything that can be decided from `__metadata__` and the safetensors tensor DIRECTORY, with no
// HIP call and no tensor byte read. GemmaContainer::Load (gemma_container.cpp) runs these checks first and
// then uploads; tests/model/test_gemma_container.cpp runs them alone on the tiny converter fixtures and on
// the real D:\models\r4dx\huihui-gemma\bf16.r4dx header.
//
// Header-only and HIP-free (like gemma_config.h / rotation_meta.h).
#pragma once

#include <cstdint>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "gemma_config.h"
#include "model_config.h"
#include "nlohmann/json.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "rotation_meta.h"

namespace r4dx::model {

struct GemmaContainerInfo {
  std::string path, model_id, config_sha256;
  nlohmann::json metadata;
  GemmaConfig config;       // GLOBAL text config (tp_world 1)
  ModelConfig model_config;  // config.ToModelConfig()
  std::optional<RotationSpec> rotation;  // Gemma option A (post_norm_rotate) when present
  bool has_trellis = false;
  int64_t image_token_id = 258880;  // top-level config key; the Huihui / google default (docs/gemma4-plan.md 2)
  int64_t full_attn_out_elems = 0;  // heads * global_head_dim: the o_full Hadamard sign vector's length
};

// `metadata` is the container's whole `__metadata__`. Throws std::runtime_error naming `path` unless it is
// a gemma4_unified container this build can run (model_arch, norm_kind plain, a parseable text_config, a
// rotation block it implements).
inline GemmaContainerInfo ParseGemmaContainerMetadata(const nlohmann::json& metadata, const std::string& path) {
  const std::string who = "r4dx::model::GemmaContainer: " + path + ": ";
  GemmaContainerInfo info;
  info.path = path;
  info.metadata = metadata;
  if (!metadata.is_object()) throw std::runtime_error(who + "__metadata__ is not an object");
  if (metadata.value("model_arch", std::string()) != "gemma4_unified") {
    throw std::runtime_error(who + "__metadata__.model_arch is '" + metadata.value("model_arch", std::string()) +
                             "', not gemma4_unified");
  }
  if (metadata.value("norm_kind", std::string()) != "plain") {
    throw std::runtime_error(who + "__metadata__.norm_kind must be 'plain' (Gemma norms are x * w, never 1 + w)");
  }
  if (!metadata.contains("model_config")) throw std::runtime_error(who + "no __metadata__.model_config");
  info.model_id = metadata.value("model_id", std::string());
  info.config_sha256 = metadata.value("config_sha256", std::string());
  if (metadata.at("model_config").is_object()) {
    const nlohmann::json& mc = metadata.at("model_config");
    if (mc.contains("image_token_id") && mc.at("image_token_id").is_number_integer()) {
      info.image_token_id = mc.at("image_token_id").get<int64_t>();
    }
  }
  info.config = GemmaConfig::FromModelConfig(metadata.at("model_config"));
  info.model_config = info.config.ToModelConfig();
  info.full_attn_out_elems = info.config.num_attention_heads * info.config.head_dim_full;
  info.rotation = ParseRotationMetadata(metadata, info.model_config, path, /*allow_post_norm_rotate=*/true,
                                        info.config.head_dim_full);
  if (info.rotation && !info.rotation->post_norm_rotate) {
    throw std::runtime_error(who + "a rotated Gemma container must be option A (out_fold had_only)");
  }
  info.has_trellis = metadata.contains("quant") && metadata.at("quant").is_object() &&
                     metadata.at("quant").contains("trellis");
  return info;
}

// The names (and, where fixed, shapes) a Gemma container of `layers` layers must carry. `bf16_body`: the
// body linears are required to carry their `.bf16.w` form with the exact [N, K, 2] shape; otherwise any
// one of the bf16 / w4a16 / trellis forms must exist. Returns one human-readable problem per mismatch
// (empty = the table is what GemmaContainer::Load will read); also reports tensors the layout does not
// know (a stray `attn.v` on a full layer, a `v_proj`-shaped k_eq_v violation).
inline std::vector<std::string> GemmaTensorTableProblems(const r4dx_convert::SafetensorsReader& r,
                                                         const GemmaContainerInfo& info, int64_t layers,
                                                         bool bf16_body = true) {
  const GemmaConfig& c = info.config;
  std::vector<std::string> bad;
  const auto shape_of = [&](const std::string& n) -> const std::vector<int64_t>& { return r.Meta(n).shape; };
  const auto need = [&](const std::string& n, std::vector<int64_t> shape) {
    if (!r.Has(n)) {
      bad.push_back("missing tensor '" + n + "'");
      return;
    }
    if (shape_of(n) != shape) {
      std::string got = "[", want = "[";
      for (size_t i = 0; i < shape_of(n).size(); ++i) got += (i ? "," : "") + std::to_string(shape_of(n)[i]);
      for (size_t i = 0; i < shape.size(); ++i) want += (i ? "," : "") + std::to_string(shape[i]);
      bad.push_back("tensor '" + n + "' has shape " + got + "], expected " + want + "]");
    }
  };
  const auto linear = [&](const std::string& base, int64_t N, int64_t K) {
    const bool any = r.Has(base + ".bf16.w") || r.Has(base + ".w4a16.wq") || r.Has(base + ".trellis.w");
    // A bf16 form on disk is always shape-checked; other forms (w4a16 / trellis) are checked by their own loaders.
    if (r.Has(base + ".bf16.w") || bf16_body || !any) {
      need(base + ".bf16.w", {N, K, 2});
    }
  };
  const int64_t H = c.hidden_size;
  const std::string rot = info.rotation ? ".rotated" : "";
  if (layers < 0 || layers > c.num_hidden_layers) bad.push_back("layer count out of range");
  need("text.embed_tokens", {c.vocab_size, H, 2});
  need("text.final_norm", {H, 2});
  if (!(r.Has("lm_head.bf16.w") || r.Has("lm_head.w4a16.wq"))) bad.push_back("missing lm_head (bf16 or w4a16)");
  else if (r.Has("lm_head.bf16.w")) need("lm_head.bf16.w", {c.vocab_size, H, 2});
  for (int64_t i = 0; i < layers && i < c.num_hidden_layers; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    const bool full = c.IsFullLayer(i);
    const int64_t hd = c.HeadDim(i), kv = c.NumKvHeads(i);
    need(b + "input_layernorm" + rot, {H, 2});
    need(b + "post_attention_layernorm", {H, 2});
    need(b + "pre_feedforward_layernorm" + rot, {H, 2});
    need(b + "post_feedforward_layernorm", {H, 2});
    need(b + "attn.q_norm", {hd, 2});
    need(b + "attn.k_norm", {hd, 2});
    need(b + "attn.k_descale", {kv, 4});
    need(b + "attn.v_descale", {kv, 4});
    need(b + "layer_scalar", {1, 4});
    linear(b + "attn.q", c.QDim(i), H);
    linear(b + "attn.k", c.KvDim(i), H);
    if (full) {
      if (r.Has(b + "attn.v.bf16.w") || r.Has(b + "attn.v.w4a16.wq") || r.Has(b + "attn.v.trellis.w"))
        bad.push_back("layer " + std::to_string(i) + " is a k_eq_v full layer but the container carries attn.v");
    } else {
      linear(b + "attn.v", c.KvDim(i), H);
    }
    linear(b + "attn.o", H, c.QDim(i));
    linear(b + "mlp.gate_up", 2 * c.intermediate_size, H);
    linear(b + "mlp.down", H, c.intermediate_size);
  }
  if (info.rotation) {
    for (const RotationTensor& t : RotationTensors(*info.rotation, info.model_config, info.full_attn_out_elems)) {
      const std::string name = t.name;
      if (name == kRotationMix || name == kRotationMix5) {
        need(name, {info.rotation->nblk, info.rotation->nblk, 4});  // the mix is a square, not flat
      } else {
        need(name, {t.elems, 4});
      }
    }
  }
  return bad;
}

}  // namespace r4dx::model
