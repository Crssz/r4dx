#include "gemma_local_text_model.h"

#include <hip/hip_runtime.h>

#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>

#include <chrono>

#include "model.h"  // ModelOptions
#include "vision_tower.h"  // vision::VisionEncodeStats
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

void GemmaLocalTextModel::EncodeImages(const float* pixel_values, int64_t total_patches,
                                       const std::vector<vision::GridThw>& grids, ImageRows* out,
                                       vision::VisionEncodeStats* stats) {
  const auto t0 = std::chrono::steady_clock::now();
  m_.EncodeImages(pixel_values, total_patches, grids, &out->dev);
  out->SetFilled(/*on_host=*/false, total_patches);
  if (stats != nullptr) {
    stats->total_patches = total_patches;
    stats->merged_tokens = total_patches;
    stats->num_segments = static_cast<int64_t>(grids.size());
    stats->encode_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  }
}

std::vector<float> GemmaLocalTextModel::PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                                          const std::vector<ImageSpan>& images) {
  return m_.PrefillMultimodal(token_ids, images);  // empty `images` is Prefill()
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

// The Gemma branch of LoadTextModel (text_model.h). TP=1 only: GemmaTpModel is M1b.
//
// ModelOptions carries the Qwen knobs; the Gemma-specific ones come from the environment until the engine
// wiring (M1-14) adds flags: R4DX_GEMMA_KV=fp8|bf16_full|bf16, R4DX_GEMMA_ATTN=ref|r4d, and
// R4DX_GEMMA_EXTENDED_CTX=1 (the 262144 opt-in). Context: ModelOptions::max_ctx is honoured when it differs
// from its Qwen default (262144); the default itself is read as "not asked" and means the checkpoint's native
// 131072 (docs/gemma4-plan.md section 9.1) -- so `--max-ctx 262144` needs R4DX_GEMMA_EXTENDED_CTX=1 until M1-14.
std::unique_ptr<TextModel> LoadGemmaTextModel(const ModelOptions& opts, const TpOptions& tp) {
  if (tp.world != 1) {
    throw std::invalid_argument("LoadTextModel: Gemma 4 runs at --tp 1 only (GemmaTpModel is M1b)");
  }
  GemmaModelOptions g;
  g.container_path = opts.container_path;
  g.layout = opts.layout;
  g.layer_limit = opts.layer_limit;
  g.prompt_checkpoint = opts.prompt_checkpoint;
  // --vision auto|on|off (docs/gemma4-plan.md M2): the encoder-free embedder's ~0.1 GB of vision.* tensors, from a
  // container converted with `r4dx-convert --vision on`; `on` requires them, `auto` loads them when present.
  g.vision = opts.vision == ModelOptions::VisionMode::kOn    ? GemmaVisionLoad::kOn
             : opts.vision == ModelOptions::VisionMode::kOff ? GemmaVisionLoad::kOff
                                                             : GemmaVisionLoad::kAuto;
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
  return std::make_unique<GemmaLocalTextModel>(GemmaModel::Load(g));
}

}  // namespace r4dx::model
