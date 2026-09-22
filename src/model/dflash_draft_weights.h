// r4dx::model::DflashDraftWeights -- CPU-only reader for a DFlash2 draft container
// (docs/container-format.md "DFlash2 draft container"). Opens the container's safetensors-shaped
// shell (r4dx_convert::SafetensorsReader -- the SAME mmap'd reader src/model/container.cpp reuses
// for the main text-model container) and its `__metadata__.dflash2` block, and hands back
// metadata + a tensor-shape/dtype directory lookup by name. NO HIP call anywhere in this class --
// fully CPU-unit-testable (tests/model/test_dflash_draft_weights.cpp).
//
// TODO(dflash2-forward): device upload + forward pass. This class deliberately stops at "the
// right tensor directory is available on the CPU" (task A1's own scope: "ONLY as far as it can be
// unit-tested on CPU"). A later stage should add a `ToDevice()`-style method that walks the same
// tensor names this header documents and uploads each through the existing `QuantLinear`/
// `core::DeviceBuffer` path (mirroring `src/model/container.cpp`'s `LoadQuantLinear`/
// `UploadRawU16` family), reusing the `SafetensorsReader` this class already holds rather than
// re-opening the container. The DFlash2 forward pass itself (per-token target-feature concat ->
// `fc` encoder -> per-layer KV injection into the draft's own cache -> block-diffusion draft
// attention/MLP with the fused dynamic depthwise conv -> shared-lm_head logits -> selector walk --
// docs/container-format.md's "DFlash2 draft container" section, cross-checked against
// `third_party/libr4d/r4d.h`'s `r4d_dflash_conv_t2_g16_bf16`) is a separate, later-stage task; NO
// code in this file executes on the GPU, and no existing test exercises the GPU through it.
#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "quant_linear.h"  // CheckW4a16Group -- the draft's linears carry the same .w4a16.* family
#include "r4dx_convert/safetensors_reader.hpp"

namespace r4dx::model {

struct Dflash2Attention {
  int64_t head_count = 0, head_count_kv = 0, key_length = 0, value_length = 0;
  bool causal = false;
  double rms_eps = 1e-6;
  int64_t sliding_window = 0;
  std::vector<bool> sliding_window_pattern;
};

struct Dflash2Rope {
  double freq_base = 1e7;
  std::vector<int64_t> dimension_sections;
  int64_t n_rot = 0;  // docs/container-format.md: FULL key_length for the real model (128), not a
                      // partial rotary factor -- see that doc's own derivation.
  std::string pairing;
};

// Mirrors docs/container-format.md's `__metadata__.dflash2` JSON block field-for-field --
// src/convert/include/r4dx_convert/dflash2_container.hpp's `BuildDflash2MetadataJson` is the
// writer side of this same contract; keep the two in lockstep.
struct Dflash2Config {
  int64_t hidden_size = 0, block_count = 0, feed_forward_length = 0;
  Dflash2Attention attention;
  Dflash2Rope rope;
  int64_t block_size = 0, conv_kernel_size = 0, conv_group_size = 0;
  int64_t selector_rank = 0, selector_top_k = 0;
  std::vector<int64_t> target_layers;  // 0-based TARGET-model layer INPUT indices, stored exactly
                                        // as the source GGUF gave them (see container-format.md).
  int64_t context_length = 0;
  int64_t mask_token_id = -1;
  int64_t vocab_size = 0;
  std::string layout;  // the quantized layout this container's linears were packed with
};

class DflashDraftWeights {
 public:
  static DflashDraftWeights Open(const std::string& path) {
    DflashDraftWeights w;
    w.metadata_ = ReadContainerMetadata(path);
    if (!w.metadata_.contains("container_kind") ||
        w.metadata_.at("container_kind").get<std::string>() != "dflash2_draft") {
      throw std::runtime_error("DflashDraftWeights::Open: '" + path +
                                "' is not a dflash2_draft container (missing/wrong container_kind)");
    }
    // Same group guard the main container gets (Container::Load -> CheckW4a16Group): a drafter
    // packed at a different w4a16 group than this build's kernel reads would produce silently
    // wrong draft logits, i.e. a collapsed acceptance rate with nothing else to see.
    if (w.metadata_.contains("quant") && w.metadata_.at("quant").contains("w4a16") &&
        w.metadata_.at("quant").at("w4a16").contains("group")) {
      CheckW4a16Group(w.metadata_.at("quant").at("w4a16").at("group").get<int>(), path);
    }
    w.reader_ = std::make_unique<r4dx_convert::SafetensorsReader>(r4dx_convert::Utf8ToWide(path));
    w.config_ = ParseConfig(w.metadata_.at("dflash2"));
    return w;
  }

  const Dflash2Config& Config() const { return config_; }
  const nlohmann::json& RawMetadata() const { return metadata_; }

  bool HasTensor(const std::string& name) const { return reader_->Has(name); }
  const r4dx_convert::TensorMeta& TensorMeta(const std::string& name) const { return reader_->Meta(name); }
  const uint8_t* TensorData(const std::string& name) const { return reader_->Data(name); }

  // Naming helpers matching docs/container-format.md's "Tensor naming" table -- a caller composes
  // e.g. LayerTensor(i, "self_attn.q_proj." + LayoutName(layout) + ".w4a16.wq") rather than
  // hand-formatting the "dflash.layers.{i}." prefix itself everywhere.
  static std::string LayerTensor(int64_t layer_idx, const std::string& suffix) {
    return "dflash.layers." + std::to_string(layer_idx) + "." + suffix;
  }

 private:
  DflashDraftWeights() = default;

  // Reads just the safetensors-shell header (8-byte length + JSON) for `__metadata__` -- the same
  // small amount of independent re-parsing src/model/container.cpp's own `ReadMetadata` does (see
  // that file's comment for why this isn't shared: SafetensorsReader itself parses this header too
  // but discards __metadata__, and the writer/reader contract is deliberately allowed this much
  // duplication rather than adding a dependency edge between components for one JSON blob).
  static nlohmann::json ReadContainerMetadata(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("DflashDraftWeights: cannot open " + path);
    uint64_t header_len = 0;
    f.read(reinterpret_cast<char*>(&header_len), 8);
    if (!f) throw std::runtime_error("DflashDraftWeights: " + path + " too small for a header");
    std::string header_json(static_cast<size_t>(header_len), '\0');
    f.read(header_json.data(), static_cast<std::streamsize>(header_len));
    if (!f) throw std::runtime_error("DflashDraftWeights: " + path + " header truncated");
    nlohmann::json header = nlohmann::json::parse(header_json);
    if (!header.contains("__metadata__"))
      throw std::runtime_error("DflashDraftWeights: " + path + " has no __metadata__");
    return header.at("__metadata__");
  }

  static Dflash2Config ParseConfig(const nlohmann::json& d) {
    Dflash2Config c;
    c.hidden_size = d.at("hidden_size").get<int64_t>();
    c.block_count = d.at("block_count").get<int64_t>();
    c.feed_forward_length = d.at("feed_forward_length").get<int64_t>();
    const auto& a = d.at("attention");
    c.attention.head_count = a.at("head_count").get<int64_t>();
    c.attention.head_count_kv = a.at("head_count_kv").get<int64_t>();
    c.attention.key_length = a.at("key_length").get<int64_t>();
    c.attention.value_length = a.at("value_length").get<int64_t>();
    c.attention.causal = a.at("causal").get<bool>();
    c.attention.rms_eps = a.at("rms_eps").get<double>();
    c.attention.sliding_window = a.at("sliding_window").get<int64_t>();
    c.attention.sliding_window_pattern = a.at("sliding_window_pattern").get<std::vector<bool>>();
    const auto& r = d.at("rope");
    c.rope.freq_base = r.at("freq_base").get<double>();
    c.rope.dimension_sections = r.at("dimension_sections").get<std::vector<int64_t>>();
    c.rope.n_rot = r.at("n_rot").get<int64_t>();
    c.rope.pairing = r.at("pairing").get<std::string>();
    c.block_size = d.at("block_size").get<int64_t>();
    c.conv_kernel_size = d.at("conv_kernel_size").get<int64_t>();
    c.conv_group_size = d.at("conv_group_size").get<int64_t>();
    c.selector_rank = d.at("selector_rank").get<int64_t>();
    c.selector_top_k = d.at("selector_top_k").get<int64_t>();
    c.target_layers = d.at("target_layers").get<std::vector<int64_t>>();
    c.context_length = d.at("context_length").get<int64_t>();
    c.mask_token_id = d.at("mask_token_id").get<int64_t>();
    c.vocab_size = d.at("vocab_size").get<int64_t>();
    if (d.contains("layout")) c.layout = d.at("layout").get<std::string>();
    return c;
  }

  std::unique_ptr<r4dx_convert::SafetensorsReader> reader_;
  nlohmann::json metadata_;
  Dflash2Config config_;
};

}  // namespace r4dx::model
