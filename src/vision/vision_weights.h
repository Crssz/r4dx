// src/vision/vision_weights.h -- the 333 `vision.*` tensors of a .r4dx container
// (docs/container-format.md "`vision.*` and `mtp.*`"), resident in device memory.
//
// Everything here is bf16 passthrough: `vision.*` mirrors the HF `Qwen3_5VisionModel` parameter
// names 1:1, is never quantized and is never permuted, so unlike the text side there is no
// `{layout}` fan-out, no QuantLinear and no layout selection -- one on-disk form, one numeric path.
// The only reshape is `patch_embed.proj.weight`, stored as the checkpoint's own Conv3d
// `[hidden, in_channels, temporal_patch_size, patch_size, patch_size]` and read here as the
// row-major `[hidden, 1536]` matrix that same memory already is (docs/vision.md "Patch embedding":
// the Conv3d is a dense matmul once `pixel_values` arrives pre-patchified).
//
// Loading is OPT-IN (Container::Load's `load_vision` / ModelOptions::vision): ~0.90 GiB of VRAM
// that a text-only run must not pay for. See docs/vision.md "Load policy".
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx_convert/safetensors_reader.hpp"

namespace r4dx::vision {

// The subset of `__metadata__.model_config.vision_config` the tower needs. Defaults are this
// checkpoint's real values, but every one of them is read from the container when present so a
// differently-shaped vision config fails on a tensor-shape mismatch rather than silently running
// the wrong geometry.
struct VisionConfig {
  int64_t depth = 27;
  int64_t hidden_size = 1152;
  int64_t num_heads = 16;
  int64_t intermediate_size = 4304;
  int64_t in_channels = 3;
  int64_t patch_size = 16;
  int64_t temporal_patch_size = 2;
  int64_t spatial_merge_size = 2;
  int64_t out_hidden_size = 5120;
  int64_t num_position_embeddings = 2304;
  // Vision's OWN rope base, 1e4 -- deliberately different from the text side's 1e7
  // (docs/vision.md "Model facts"). Absent from this checkpoint's stored vision_config, in which
  // case `Qwen3_5VisionConfig`'s own default (10000.0) applies, which is what this default is.
  double rope_theta = 10000.0;
  std::string hidden_act = "gelu_pytorch_tanh";
  double layer_norm_eps = 1e-6;  // Qwen3_5VisionBlock hardcodes nn.LayerNorm(..., eps=1e-6)

  int64_t HeadDim() const { return hidden_size / num_heads; }
  int64_t PatchDim() const {
    return in_channels * temporal_patch_size * patch_size * patch_size;  // 1536
  }
  // The merger's own width: four spatially-adjacent patches concatenated (4608).
  int64_t MergerHidden() const {
    return hidden_size * spatial_merge_size * spatial_merge_size;
  }
  // 48 for this checkpoint -- the learned position grid is square, side = sqrt(2304).
  int64_t NumGridPerSide() const;

  static VisionConfig FromJson(const nlohmann::json& vision_cfg);
};

// One `nn.Linear` of the tower: `y = x @ weight^T + bias`, weight [N, K] row-major bf16, which is
// exactly r4d_gemm_bf16_nt_m64's own operand contract.
struct VisionLinear {
  core::DeviceBuffer<uint16_t> weight;
  core::DeviceBuffer<uint16_t> bias;
  int64_t N = 0, K = 0;
};

// One `nn.LayerNorm`: weight AND bias, both [hidden] bf16 (unlike every norm on the text side).
struct VisionLayerNorm {
  core::DeviceBuffer<uint16_t> weight, bias;
};

struct VisionBlockWeights {
  VisionLayerNorm norm1, norm2;
  VisionLinear qkv;   // [3*hidden, hidden]
  VisionLinear proj;  // [hidden, hidden]
  VisionLinear fc1;   // [intermediate, hidden]
  VisionLinear fc2;   // [hidden, intermediate]
};

struct VisionWeights {
  VisionConfig config;
  VisionLinear patch_embed;                       // [hidden, patch_dim]
  core::DeviceBuffer<uint16_t> pos_embed_table;   // [num_position_embeddings, hidden]
  std::vector<VisionBlockWeights> blocks;         // depth entries
  VisionLayerNorm merger_norm;                    // over hidden (PRE-view), not merger_hidden
  VisionLinear merger_fc1;                        // [merger_hidden, merger_hidden]
  VisionLinear merger_fc2;                        // [out_hidden_size, merger_hidden]

  int64_t tensor_count = 0;  // 333 for this checkpoint -- reported, not assumed
  int64_t bytes = 0;         // total device bytes uploaded by LoadVisionWeights
};

// True iff `reader` carries a vision tower at all (probed by one representative tensor, the same
// way Container probes `mtp.norm` for the MTP head rather than trusting `__metadata__`).
bool HasVisionTensors(const r4dx_convert::SafetensorsReader& reader);

// Uploads every `vision.*` tensor onto the CURRENT HIP device, synchronously (this is a
// startup-path call, like Container::Load's own uploads). `vision_cfg` is the container's
// `__metadata__.model_config.vision_config` object. Throws -- naming the tensor -- on a missing
// tensor or a shape that does not match the config, rather than loading a tower that would
// produce plausible garbage.
VisionWeights LoadVisionWeights(const r4dx_convert::SafetensorsReader& reader,
                                 const nlohmann::json& vision_cfg);

}  // namespace r4dx::vision
