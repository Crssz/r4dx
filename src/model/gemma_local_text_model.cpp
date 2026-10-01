#include "gemma_local_text_model.h"

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include "gemma_tp_model.h"
#include "model.h"  // ModelOptions
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"

namespace r4dx::model {

std::vector<VramReport> GemmaLocalTextModel::Vram() const {
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

void GemmaLocalTextModel::EncodeImages(const float*, int64_t, const std::vector<vision::GridThw>&, ImageRows*,
                                       vision::VisionEncodeStats*) {
  throw std::runtime_error("GemmaLocalTextModel::EncodeImages: Gemma 4 vision is not implemented yet (docs/gemma4-plan.md M2)");
}

std::vector<float> GemmaLocalTextModel::PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                                          const std::vector<ImageSpan>& images) {
  if (!images.empty()) {
    throw std::runtime_error("GemmaLocalTextModel::PrefillMultimodal: Gemma 4 vision is not implemented yet (M2)");
  }
  return m_.Prefill(token_ids);
}

std::vector<int32_t> GemmaLocalTextModel::DecodeStepMtpGreedy(int32_t, int64_t) {
  throw std::runtime_error("GemmaLocalTextModel: MTP is not available for Gemma 4 (the container carries no mtp.* head)");
}
std::vector<int32_t> GemmaLocalTextModel::DecodeStepMtpSampled(int32_t, int64_t, const kernels::SampleParams&,
                                                               std::mt19937_64&) {
  throw std::runtime_error("GemmaLocalTextModel: MTP is not available for Gemma 4 (the container carries no mtp.* head)");
}
std::vector<int32_t> GemmaLocalTextModel::DecodeStepDflashGreedy(int32_t, int64_t, float, int64_t, int64_t*) {
  throw std::runtime_error("GemmaLocalTextModel: DFlash is not wired for Gemma 4 yet (drafter track D-6)");
}
std::vector<int32_t> GemmaLocalTextModel::DecodeStepDflashSampled(int32_t, int64_t, float, int64_t,
                                                                  const kernels::SampleParams&, std::mt19937_64&,
                                                                  int64_t*) {
  throw std::runtime_error("GemmaLocalTextModel: DFlash is not wired for Gemma 4 yet (drafter track D-6)");
}

// The Gemma branch of LoadTextModel (text_model.h): TP=1 -> GemmaLocalTextModel, TP=2 -> GemmaTpModel (M1b-1).
//
// ModelOptions carries the Qwen knobs; the Gemma-specific ones come from the environment until the engine
// wiring (M1-14) adds flags: R4DX_GEMMA_KV=fp8|bf16_full|bf16, R4DX_GEMMA_ATTN=ref|r4d, and
// R4DX_GEMMA_EXTENDED_CTX=1 (the 262144 opt-in). Context: ModelOptions::max_ctx is honoured when it differs
// from its Qwen default (262144); the default itself is read as "not asked" and means the checkpoint's native
// 131072 (docs/gemma4-plan.md section 9.1) -- so `--max-ctx 262144` needs R4DX_GEMMA_EXTENDED_CTX=1 until M1-14.
GemmaModelOptions MakeGemmaModelOptions(const ModelOptions& opts) {
  GemmaModelOptions g;
  g.container_path = opts.container_path;
  g.layout = opts.layout;
  g.layer_limit = opts.layer_limit;
  g.prompt_checkpoint = opts.prompt_checkpoint;
  const char* ext = std::getenv("R4DX_GEMMA_EXTENDED_CTX");
  g.allow_extended_ctx = ext != nullptr && std::string(ext) == "1";
  const ModelOptions defaults;
  if (g.allow_extended_ctx) {
    g.max_ctx = GemmaConfig::kExtendedMaxCtx;
  } else if (opts.max_ctx != defaults.max_ctx) {
    g.max_ctx = opts.max_ctx;  // an explicit request: ResolveMaxCtx refuses anything above the native context
  }
  if (opts.mtp_draft_k > 0 || opts.dflash_draft_k > 0 || !opts.dflash_container.empty()) {
    throw std::invalid_argument("LoadTextModel: MTP and DFlash are not available for Gemma 4 yet");
  }
  ApplyGemmaEnv(&g);
  return g;
}

std::unique_ptr<TextModel> LoadGemmaTextModel(const ModelOptions& opts, const TpOptions& tp) {
  if (tp.world == 2) return GemmaTpModel::Load(opts, tp);
  if (tp.world != 1) {
    throw std::invalid_argument("LoadTextModel: TpOptions::world must be 1 or 2, got " + std::to_string(tp.world));
  }
  return std::make_unique<GemmaLocalTextModel>(GemmaModel::Load(MakeGemmaModelOptions(opts)));
}

}  // namespace r4dx::model
