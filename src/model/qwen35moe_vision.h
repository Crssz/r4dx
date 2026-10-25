// r4dx::model::Qwen35MoeVision -- the Ornith-1.5-122B (qwen3_5_moe) vision tower on ONE device (docs/qwen35moe-vision.md).
//
// The tower is the Qwen3.5 27B's, unchanged (src/vision: depth 27, hidden 1152, 16 heads, patch 16 x temporal 2, 2x2 merger,
// learned 48 x 48 position grid); only the merger's out_hidden_size differs (3072 here). So this class is a thin owner: the
// `vision.*` weights (VisionWeights, bf16, ~0.85 GiB) plus the VisionTower stream / scratch on the device the model picked, and
// an Encode that hands the merged rows back as HOST bf16 rows [tokens, 3072]. Host rows because the text model's embedding
// path is a host gather (Qwen35MoeModel::RunChunk): the splice is a memcpy into the pinned chunk buffer, on one card, on both
// ranks of a tensor-parallel group, whatever device the tower lives on.
//
// Placement: the tower must not take the desktop card's headroom (the weights already fill it), so the default device is the
// LAST pipeline stage's (HIP device 1, headless); R4DX_Q35_VISION_DEVICE overrides.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "preprocess.h"  // vision::GridThw
#include "vision_tower.h"
#include "vision_weights.h"

namespace r4dx::model {

class Qwen35MoeVision {
 public:
  // True iff the container at `path` carries `vision.*` tensors (reads the header only).
  static bool ContainerHasVision(const std::string& path);

  // Uploads the tower onto `device` (the calling thread's current device is restored). Throws when the container has none, has
  // no model_config.vision_config, or the tower's out_hidden_size is not `text_hidden`.
  static std::unique_ptr<Qwen35MoeVision> Load(const std::string& path, int device, int64_t text_hidden);

  ~Qwen35MoeVision();

  int Device() const { return device_; }
  int64_t ImageTokenId() const { return image_token_id_; }
  int64_t MergeSize() const { return weights_.config.spatial_merge_size; }
  const vision::VisionConfig& Config() const { return weights_.config; }
  int64_t WeightBytes() const { return weights_.bytes; }
  int64_t ScratchBytes() const { return tower_->ScratchBytes(); }

  // pixel_values: HOST fp32 [total_patches, patch_dim] (PreprocessImages' output). `rows` becomes bf16 [merged tokens, hidden].
  void Encode(const float* pixel_values, int64_t total_patches, const std::vector<vision::GridThw>& grids,
              std::vector<uint16_t>* rows, vision::VisionEncodeStats* stats = nullptr);

 private:
  Qwen35MoeVision() = default;
  int device_ = 0;
  int64_t image_token_id_ = -1;
  int64_t text_hidden_ = 0;
  vision::VisionWeights weights_;
  std::unique_ptr<vision::VisionTower> tower_;  // created and destroyed with the tower's device current
};

}  // namespace r4dx::model
