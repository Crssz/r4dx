#include "hy_v3_local_text_model.h"

#include <hip/hip_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "model.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"

namespace r4dx::model {

std::vector<VramReport> HyV3LocalTextModel::Vram() const {
  VramReport r;
  r.rank = 0;
  R4DX_HIP_CHECK(hipGetDevice(&r.device));
  size_t free_b = 0, total_b = 0;
  R4DX_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
  constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;
  r.used_gib = static_cast<double>(total_b - free_b) / kGiB;
  r.free_gib = static_cast<double>(free_b) / kGiB;
  r.total_gib = static_cast<double>(total_b) / kGiB;
  r.buffers_gib = static_cast<double>(core::DeviceBufferBytes(r.device)) / kGiB;
  return {r};
}

void HyV3LocalTextModel::EncodeImages(const float*, int64_t, const std::vector<vision::GridThw>&, ImageRows*,
                                      vision::VisionEncodeStats*) {
  throw std::runtime_error("HyV3LocalTextModel: hy_v3 is text only");
}

std::vector<float> HyV3LocalTextModel::PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                                         const std::vector<ImageSpan>& images) {
  if (!images.empty()) {
    throw std::runtime_error("HyV3LocalTextModel::PrefillMultimodal: hy_v3 is text only");
  }
  return m_.Prefill(token_ids);
}

std::vector<int32_t> HyV3LocalTextModel::DecodeStepMtpGreedy(int32_t, int64_t) {
  throw std::runtime_error("HyV3LocalTextModel: MTP is not supported for hy_v3");
}

std::vector<int32_t> HyV3LocalTextModel::DecodeStepMtpSampled(int32_t, int64_t, const kernels::SampleParams&,
                                                              std::mt19937_64&) {
  throw std::runtime_error("HyV3LocalTextModel: MTP is not supported for hy_v3");
}

std::vector<int32_t> HyV3LocalTextModel::DecodeStepDflashGreedy(int32_t, int64_t, float, int64_t, int64_t*) {
  throw std::runtime_error("HyV3LocalTextModel: DFlash is not supported for hy_v3");
}

std::vector<int32_t> HyV3LocalTextModel::DecodeStepDflashSampled(int32_t, int64_t, float, int64_t,
                                                                 const kernels::SampleParams&, std::mt19937_64&,
                                                                 int64_t*) {
  throw std::runtime_error("HyV3LocalTextModel: DFlash is not supported for hy_v3");
}

std::unique_ptr<TextModel> LoadHyV3TextModel(const ModelOptions& opts, const TpOptions& tp) {
  if (tp.world != 1) {
    throw std::invalid_argument("LoadTextModel: hy_v3 currently supports TP=1 only (single R9700)");
  }
  HyV3ModelOptions o;
  o.container_path = opts.container_path;
  o.layout = opts.layout;
  o.layer_limit = opts.layer_limit;
  o.prompt_checkpoint = opts.prompt_checkpoint;
  const ModelOptions defaults;
  if (opts.max_ctx != defaults.max_ctx) o.max_ctx = opts.max_ctx;
  if (opts.prefill_chunk >= 1 && opts.prefill_chunk <= 256) o.prefill_chunk = opts.prefill_chunk;
  return std::make_unique<HyV3LocalTextModel>(HyV3Model::Load(o));
}

}  // namespace r4dx::model
