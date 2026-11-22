// LoadInkling2TextModel: the inkling2_loop branch of LoadTextModel (docs/inkling2-plan.md 5): TP=1 ->
// Inkling2LocalTextModel, TP=2 -> Inkling2TpModel.
#include "inkling2_local_text_model.h"

#include <hip/hip_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "inkling2_tp_model.h"
#include "model.h"  // ModelOptions
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"

namespace r4dx::model {

namespace {
const char* kTextOnly = "Inkling-2-Mini runs text only in r4dx (milestone 1: no vision tower)";
const char* kNoMtp = "MTP / the DSpark drafter is not available for inkling2_loop (milestone 1)";
const char* kNoDflash = "DFlash is not available for inkling2_loop (no drafter)";
}  // namespace

std::vector<VramReport> Inkling2LocalTextModel::Vram() const {
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

void Inkling2LocalTextModel::EncodeImages(const float*, int64_t, const std::vector<vision::GridThw>&, ImageRows*,
                                          vision::VisionEncodeStats*) {
  throw std::runtime_error(std::string("Inkling2LocalTextModel: ") + kTextOnly);
}

std::vector<float> Inkling2LocalTextModel::PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                                             const std::vector<ImageSpan>& images) {
  if (!images.empty()) throw std::runtime_error(std::string("Inkling2LocalTextModel::PrefillMultimodal: ") + kTextOnly);
  return m_.Prefill(token_ids);
}

std::vector<int32_t> Inkling2LocalTextModel::DecodeStepMtpGreedy(int32_t, int64_t) {
  throw std::runtime_error(std::string("Inkling2LocalTextModel: ") + kNoMtp);
}
std::vector<int32_t> Inkling2LocalTextModel::DecodeStepMtpSampled(int32_t, int64_t, const kernels::SampleParams&,
                                                                  std::mt19937_64&) {
  throw std::runtime_error(std::string("Inkling2LocalTextModel: ") + kNoMtp);
}
std::vector<int32_t> Inkling2LocalTextModel::DecodeStepDflashGreedy(int32_t token_id, int64_t k, float, int64_t,
                                                                    int64_t* walk_len_out) {
  return m_.DecodeStepDflashGreedy(token_id, k, walk_len_out);
}
std::vector<int32_t> Inkling2LocalTextModel::DecodeStepDflashSampled(int32_t token_id, int64_t k, float, int64_t,
                                                                     const kernels::SampleParams& params,
                                                                     std::mt19937_64& rng, int64_t* walk_len_out) {
  return m_.DecodeStepDflashSampled(token_id, k, params, rng, walk_len_out);
}

Inkling2ModelOptions MakeInkling2ModelOptions(const ModelOptions& opts) {
  Inkling2ModelOptions o;
  o.container_path = opts.container_path;
  o.layout = opts.layout;
  // ModelOptions' own default (262144, Qwen's) means "not given" (LFM2's precedent): the 131072 default, so a caller that
  // never resolved the context does not reserve 8 GiB of KV.
  const ModelOptions defaults;
  o.max_ctx = opts.max_ctx != defaults.max_ctx ? opts.max_ctx : 0;
  o.prompt_checkpoint = opts.prompt_checkpoint;
  if (opts.prefill_chunk >= 1 && opts.prefill_chunk <= 256) o.prefill_chunk = opts.prefill_chunk;
  if (opts.layer_limit >= 0)
    throw std::invalid_argument("LoadTextModel: --layers is not supported for inkling2_loop (the looped stack runs whole)");
  if (opts.mtp_draft_k > 0) throw std::invalid_argument(std::string("LoadTextModel: ") + kNoMtp);
  if (!opts.dflash_container.empty()) o.dspark_path = opts.dflash_container;
  if (opts.vision == ModelOptions::VisionMode::kOn)
    throw std::invalid_argument(std::string("LoadTextModel: --vision on, but ") + kTextOnly);
  if (opts.batch_slots > 0)
    throw std::invalid_argument("LoadTextModel: --batch " + std::to_string(opts.batch_slots) +
                                " does not support inkling2_loop (no batched decode path; drop --batch)");
  return o;
}

std::unique_ptr<TextModel> LoadInkling2TextModel(const ModelOptions& opts, const TpOptions& tp) {
  if (tp.world == 2) return Inkling2TpModel::Load(opts, tp);
  if (tp.world != 1) throw std::invalid_argument("LoadTextModel: TpOptions::world must be 1 or 2, got " + std::to_string(tp.world));
  return std::make_unique<Inkling2LocalTextModel>(Inkling2Model::Load(MakeInkling2ModelOptions(opts)));
}

}  // namespace r4dx::model
