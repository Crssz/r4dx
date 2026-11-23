#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "preprocess.h"
#include "quant_linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::model {

struct Inkling2VisionConfig {
  int64_t hidden_size = 1024;
  int64_t num_layers = 23;
  int64_t num_heads = 16;
  int64_t head_dim = 64;
  int64_t mlp_dim = 4096;
  int64_t native_patch_size = 20;
  int64_t adapter_merge_size = 2;
  int64_t posemb_grid_size = 32;
  int64_t proj_hidden_dim = 4096;
  int64_t text_hidden_size = 2304;
  float layer_norm_eps = 1e-5f;
};

class Inkling2VisionTower {
 public:
  // Returns empty (HasWeights() == false) if the container does not carry `vision.*` tensors.
  static Inkling2VisionTower TryLoad(const std::string& container_path);

  bool HasWeights() const { return loaded_; }
  const Inkling2VisionConfig& Config() const { return cfg_; }

  // Encode one or more preprocessed images ([total_patches, 1200] float32 HWC) into
  // [sum(gh/2 * gw/2), text_hidden_size] bf16 on device.
  void Encode(const float* pixel_values, int64_t total_patches, const std::vector<vision::GridThw>& grids,
              core::DeviceBuffer<uint16_t>* out_bf16, core::Stream& stream, core::Arena& arena) const;

 private:
  struct Layer {
    core::DeviceBuffer<uint16_t> ln1_w, ln1_b;
    core::DeviceBuffer<uint16_t> ln2_w, ln2_b;
    core::DeviceBuffer<uint16_t> ls1_g, ls2_g;
    QuantLinear q_proj, k_proj, v_proj, out_proj;
    core::DeviceBuffer<uint16_t> q_b, k_b, v_b, out_b;
    QuantLinear c_fc, c_proj;
    core::DeviceBuffer<uint16_t> fc_b, proj_b;
  };

  bool loaded_ = false;
  Inkling2VisionConfig cfg_;
  core::DeviceBuffer<uint16_t> class_emb_;       // [1024]
  core::DeviceBuffer<uint16_t> cls_pos_emb_;     // [1024]
  core::DeviceBuffer<uint16_t> pos_emb_;         // [1024, 1024]
  core::DeviceBuffer<uint16_t> patch_emb_w_;     // [1024, 1200]
  core::DeviceBuffer<uint16_t> ln_pre_w_, ln_pre_b_;
  std::vector<Layer> layers_;
  QuantLinear proj_in_;                          // [4096, 4096]
  QuantLinear proj_out_;                         // [2304, 4096]
};

}  // namespace r4dx::model
