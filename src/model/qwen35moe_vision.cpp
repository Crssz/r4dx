#include "qwen35moe_vision.h"

#include <hip/hip_runtime.h>

#include <stdexcept>
#include <string>

#include "container_load_util.h"
#include "qwen35moe_stage.h"  // Qwen35MoeDeviceScope
#include "r4dx/core/error.hpp"

namespace r4dx::model {

namespace {
constexpr const char* kWho = "r4dx::model::Qwen35MoeVision: ";
}

bool Qwen35MoeVision::ContainerHasVision(const std::string& path) {
  const container_util::SafetensorsReader reader(container_util::Utf8ToWide(path));
  return vision::HasVisionTensors(reader);
}

std::unique_ptr<Qwen35MoeVision> Qwen35MoeVision::Load(const std::string& path, int device, int64_t text_hidden) {
  // The tower's stream and arena belong to the device that is current when they are created: scope it before construction.
  Qwen35MoeDeviceScope dev(device);
  std::unique_ptr<Qwen35MoeVision> v(new Qwen35MoeVision());
  v->device_ = device;
  v->tower_ = std::make_unique<vision::VisionTower>();
  v->text_hidden_ = text_hidden;
  const nlohmann::json metadata = container_util::ReadMetadata(path);
  const nlohmann::json& model_config = metadata.at("model_config");
  if (!model_config.contains("vision_config")) {
    throw std::runtime_error(std::string(kWho) + path + " carries no model_config.vision_config");
  }
  if (model_config.contains("image_token_id")) v->image_token_id_ = model_config.at("image_token_id").get<int64_t>();
  if (v->image_token_id_ < 0) throw std::runtime_error(std::string(kWho) + path + " carries no model_config.image_token_id");
  const container_util::SafetensorsReader reader(container_util::Utf8ToWide(path));
  if (!vision::HasVisionTensors(reader)) {
    throw std::runtime_error(std::string(kWho) + path + " carries no vision.* tensors (convert with r4dx-convert --vision on)");
  }
  v->weights_ = vision::LoadVisionWeights(reader, model_config.at("vision_config"));
  if (v->weights_.config.out_hidden_size != text_hidden) {
    throw std::runtime_error(std::string(kWho) + "vision_config.out_hidden_size (" +
                             std::to_string(v->weights_.config.out_hidden_size) + ") is not the text hidden size (" +
                             std::to_string(text_hidden) + "): the merger's rows are spliced straight into the embedding sequence");
  }
  return v;
}

Qwen35MoeVision::~Qwen35MoeVision() {
  // Frees (weights, scratch, the stream) on the tower's device.
  Qwen35MoeDeviceScope dev(device_);
  tower_.reset();
  weights_ = vision::VisionWeights();
}

void Qwen35MoeVision::Encode(const float* pixel_values, int64_t total_patches, const std::vector<vision::GridThw>& grids,
                             std::vector<uint16_t>* rows, vision::VisionEncodeStats* stats) {
  Qwen35MoeDeviceScope dev(device_);
  core::DeviceBuffer<uint16_t> out;
  // Per-block synchronization on the desktop card: the tower's whole encode (~160 ms for a 1024-token image) must not reach the
  // GPU as one unbroken queue (docs/tp.md Appendix B N57). A hook that leaves the stream alone bounds it to one encoder block.
  const vision::VisionPreBlock per_block = [](int64_t, void*, int64_t) { return false; };
  tower_->Encode(weights_, pixel_values, total_patches, grids, &out, stats, nullptr, device_ == 0 ? &per_block : nullptr);
  R4DX_HIP_CHECK(hipStreamSynchronize(tower_->StreamHandle()));
  rows->resize(out.size());
  if (!rows->empty()) out.CopyToHost(rows->data(), rows->size());
}

}  // namespace r4dx::model
