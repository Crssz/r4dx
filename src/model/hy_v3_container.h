// r4dx::model::HyV3Container -- loads one hy_v3 .r4dx container into device memory.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "container.h"  // MlpWeights
#include "hy_v3_config.h"
#include "hy_v3_container_info.h"
#include "lfm2_container.h"  // Lfm2AttnWeights, Lfm2MoeWeights
#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"

namespace r4dx::model {

struct HyV3LayerWeights {
  bool dense = false;
  core::DeviceBuffer<uint16_t> input_norm, post_attn_norm;
  Lfm2AttnWeights attn;
  MlpWeights mlp;
  Lfm2MoeWeights moe;
  MlpWeights shared_mlp;
  // Device pointer tables of length (E + 1): indices [0..E-1] = routed experts, index E = shared_mlp.
  // Populated when every expert + shared_mlp loaded as kW4a16 at a uniform group.
  core::DeviceBuffer<uint64_t> gate_up_w4_wq_ptrs, gate_up_w4_wsz_ptrs;
  core::DeviceBuffer<uint64_t> down_w4_wq_ptrs, down_w4_wsz_ptrs;
  int moe_w4_group = 0;

  bool HasW4MoePtrs() const {
    return moe_w4_group > 0 && !gate_up_w4_wq_ptrs.empty() && !gate_up_w4_wsz_ptrs.empty() &&
           !down_w4_wq_ptrs.empty() && !down_w4_wsz_ptrs.empty();
  }
};

struct HyV3LoadOptions {
  Layout layout = Layout::kW4a16;
  Layout lm_head_layout = Layout::kW8a16;
  int64_t layer_limit = -1;
};

class HyV3Container {
 public:
  static HyV3ContainerInfo Inspect(const std::string& path, int64_t layer_limit = -1) {
    return InspectHyV3Container(path, layer_limit);
  }
  static HyV3Container Load(const std::string& path, const HyV3LoadOptions& o = HyV3LoadOptions());

  HyV3Container(HyV3Container&&) = default;
  HyV3Container& operator=(HyV3Container&&) = default;

  const HyV3ContainerInfo& Info() const { return info_; }
  const HyV3Config& Config() const { return info_.config; }
  const ModelConfig& GenericConfig() const { return info_.model_config; }
  const std::string& ModelId() const { return info_.model_id; }

  int64_t NumLoadedLayers() const { return static_cast<int64_t>(layers_.size()); }
  const HyV3LayerWeights& Layer(int64_t i) const { return layers_.at(static_cast<size_t>(i)); }
  const core::DeviceBuffer<uint16_t>& EmbedTokensDevice() const { return embed_tokens_; }
  const core::DeviceBuffer<uint16_t>& FinalNorm() const { return final_norm_; }
  const QuantLinear& LmHead() const { return lm_head_; }

 private:
  HyV3Container() = default;
  friend class HyV3Model;

  HyV3ContainerInfo info_;
  core::DeviceBuffer<uint16_t> embed_tokens_, final_norm_;
  QuantLinear lm_head_;
  std::vector<HyV3LayerWeights> layers_;
};

}  // namespace r4dx::model
