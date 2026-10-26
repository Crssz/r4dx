// LoadLlamaTextModel: the Llama branch of LoadTextModel (docs/nanbeige-plan.md 3.1): TP=1 -> LlamaLocalTextModel (N1-8),
// TP=2 -> LlamaTpModel (N1-7).
#include "llama_local_text_model.h"

#include <hip/hip_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "llama_tp_model.h"
#include "model.h"  // ModelOptions
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"

namespace r4dx::model {

std::vector<VramReport> LlamaLocalTextModel::Vram() const {
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

void LlamaLocalTextModel::EncodeImages(const float*, int64_t, const std::vector<vision::GridThw>&, ImageRows*,
                                      vision::VisionEncodeStats*) {
  throw std::runtime_error("LlamaLocalTextModel: a Llama container is text only (no vision embedder)");
}

std::vector<float> LlamaLocalTextModel::PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                                         const std::vector<ImageSpan>& images) {
  if (!images.empty()) {
    throw std::runtime_error("LlamaLocalTextModel::PrefillMultimodal: a Llama container is text only (image spans given)");
  }
  return m_.Prefill(token_ids);
}

std::vector<int32_t> LlamaLocalTextModel::DecodeStepMtpGreedy(int32_t, int64_t) {
  throw std::runtime_error("LlamaLocalTextModel: MTP is not available for Llama (the checkpoint has no MTP head)");
}
std::vector<int32_t> LlamaLocalTextModel::DecodeStepMtpSampled(int32_t, int64_t, const kernels::SampleParams&,
                                                              std::mt19937_64&) {
  throw std::runtime_error("LlamaLocalTextModel: MTP is not available for Llama (the checkpoint has no MTP head)");
}
std::vector<int32_t> LlamaLocalTextModel::DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                                                int64_t* walk_len_out) {
  return m_.DecodeStepDflashGreedy(token_id, k, p_min, n_min, walk_len_out);
}
std::vector<int32_t> LlamaLocalTextModel::DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                                                 const kernels::SampleParams& params, std::mt19937_64& rng,
                                                                 int64_t* walk_len_out) {
  return m_.DecodeStepDflashSampled(token_id, k, p_min, n_min, params, rng, walk_len_out);
}

LlamaModelOptions MakeLlamaModelOptions(const ModelOptions& opts) {
  LlamaModelOptions o;
  o.container_path = opts.container_path;
  o.layout = opts.layout;
  o.layer_limit = opts.layer_limit;
  o.prompt_checkpoint = opts.prompt_checkpoint;
  const ModelOptions defaults;
  if (opts.max_ctx != defaults.max_ctx) o.max_ctx = opts.max_ctx;  // ResolveMaxCtx refuses anything above max_position_embeddings
  if (opts.prefill_chunk >= 1 && opts.prefill_chunk <= 256) o.prefill_chunk = opts.prefill_chunk;
  if (opts.mtp_draft_k > 0) {
    throw std::invalid_argument("LoadTextModel: MTP is not available for Llama (the checkpoint has no MTP head)");
  }
  if (!opts.dflash_container.empty() && opts.dflash_draft_k <= 0) {
    throw std::invalid_argument("LoadTextModel: dflash_container is set but dflash_draft_k <= 0");
  }
  o.dflash_container = opts.dflash_container;
  o.dflash_draft_k = opts.dflash_container.empty() ? 0 : opts.dflash_draft_k;
  if (opts.vision == ModelOptions::VisionMode::kOn) {
    throw std::invalid_argument("LoadTextModel: --vision on, but a Llama container is text only");
  }
  return o;
}

std::unique_ptr<TextModel> LoadLlamaTextModel(const ModelOptions& opts, const TpOptions& tp) {
  if (tp.world == 2) return LlamaTpModel::Load(opts, tp);
  if (tp.world != 1) {
    throw std::invalid_argument("LoadTextModel: TpOptions::world must be 1 or 2, got " + std::to_string(tp.world));
  }
  return std::make_unique<LlamaLocalTextModel>(LlamaModel::Load(MakeLlamaModelOptions(opts)));
}

}  // namespace r4dx::model
