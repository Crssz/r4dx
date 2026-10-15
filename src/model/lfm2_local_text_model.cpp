// LoadLfm2TextModel: the LFM2 branch of LoadTextModel (docs/lfm2-plan.md 3.1): TP=1 -> Lfm2LocalTextModel (L1-8),
// TP=2 -> Lfm2TpModel (L1-10).
#include "lfm2_local_text_model.h"

#include <hip/hip_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "lfm2_tp_model.h"
#include "model.h"  // ModelOptions
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"

namespace r4dx::model {

std::vector<VramReport> Lfm2LocalTextModel::Vram() const {
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

void Lfm2LocalTextModel::EncodeImages(const float*, int64_t, const std::vector<vision::GridThw>&, ImageRows*,
                                      vision::VisionEncodeStats*) {
  throw std::runtime_error("Lfm2LocalTextModel: LFM2.5-24B-A2B is a text-only model (no vision embedder)");
}

std::vector<float> Lfm2LocalTextModel::PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                                         const std::vector<ImageSpan>& images) {
  if (!images.empty()) {
    throw std::runtime_error("Lfm2LocalTextModel::PrefillMultimodal: LFM2.5-24B-A2B is a text-only model (image spans given)");
  }
  return m_.Prefill(token_ids);
}

std::vector<int32_t> Lfm2LocalTextModel::DecodeStepMtpGreedy(int32_t, int64_t) {
  throw std::runtime_error("Lfm2LocalTextModel: MTP is not available for LFM2 (the checkpoint has no MTP head)");
}
std::vector<int32_t> Lfm2LocalTextModel::DecodeStepMtpSampled(int32_t, int64_t, const kernels::SampleParams&,
                                                              std::mt19937_64&) {
  throw std::runtime_error("Lfm2LocalTextModel: MTP is not available for LFM2 (the checkpoint has no MTP head)");
}
std::vector<int32_t> Lfm2LocalTextModel::DecodeStepDflashGreedy(int32_t, int64_t, float, int64_t, int64_t*) {
  throw std::runtime_error("Lfm2LocalTextModel: DFlash is not available for LFM2 (no drafter, docs/lfm2-plan.md 1.3)");
}
std::vector<int32_t> Lfm2LocalTextModel::DecodeStepDflashSampled(int32_t, int64_t, float, int64_t,
                                                                 const kernels::SampleParams&, std::mt19937_64&, int64_t*) {
  throw std::runtime_error("Lfm2LocalTextModel: DFlash is not available for LFM2 (no drafter, docs/lfm2-plan.md 1.3)");
}

Lfm2ModelOptions MakeLfm2ModelOptions(const ModelOptions& opts) {
  Lfm2ModelOptions o;
  o.container_path = opts.container_path;
  o.layout = opts.layout;
  o.layer_limit = opts.layer_limit;
  o.prompt_checkpoint = opts.prompt_checkpoint;
  const ModelOptions defaults;
  if (opts.max_ctx != defaults.max_ctx) o.max_ctx = opts.max_ctx;  // ResolveMaxCtx refuses anything above 131072
  if (opts.prefill_chunk >= 1 && opts.prefill_chunk <= 256) o.prefill_chunk = opts.prefill_chunk;
  if (opts.mtp_draft_k > 0) {
    throw std::invalid_argument("LoadTextModel: MTP is not available for LFM2 (the checkpoint has no MTP head)");
  }
  if (!opts.dflash_container.empty() || opts.dflash_draft_k > 0) {
    throw std::invalid_argument("LoadTextModel: DFlash is not available for LFM2 (no drafter)");
  }
  if (opts.vision == ModelOptions::VisionMode::kOn) {
    throw std::invalid_argument("LoadTextModel: --vision on, but LFM2.5-24B-A2B is a text-only model");
  }
  return o;
}

std::unique_ptr<TextModel> LoadLfm2TextModel(const ModelOptions& opts, const TpOptions& tp) {
  if (tp.world == 2) return Lfm2TpModel::Load(opts, tp);
  if (tp.world != 1) {
    throw std::invalid_argument("LoadTextModel: TpOptions::world must be 1 or 2, got " + std::to_string(tp.world));
  }
  return std::make_unique<Lfm2LocalTextModel>(Lfm2Model::Load(MakeLfm2ModelOptions(opts)));
}

}  // namespace r4dx::model
