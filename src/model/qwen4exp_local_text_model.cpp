#include "qwen4exp_local_text_model.h"

#include <stdexcept>
#include <string>

#include "model.h"  // ModelOptions

namespace r4dx::model {

namespace {

[[noreturn]] void Unsupported(const std::string& feature, const std::string& why) {
  throw std::runtime_error("Qwen4ExpLocalTextModel: " + feature + " is not available for qwen4_exp (" + why + ")");
}

}  // namespace

void Qwen4ExpLocalTextModel::SetDflashInjectionEnabled(bool enabled) {
  // The server only calls this when a drafter container is configured, which MakeQwen4ExpModelOptions refused.
  if (enabled) Unsupported("DFlash", "milestone 1 is text-only, no drafter");
}

void Qwen4ExpLocalTextModel::EncodeImages(const float*, int64_t, const std::vector<vision::GridThw>&, ImageRows*,
                                          vision::VisionEncodeStats*) {
  Unsupported("vision", "the Qwen3-VL tower is milestone 4");
}

std::vector<float> Qwen4ExpLocalTextModel::PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                                             const std::vector<ImageSpan>& images) {
  if (!images.empty()) Unsupported("vision", "the Qwen3-VL tower is milestone 4");
  return m_->Prefill(token_ids);  // no images: a plain Prefill
}

std::vector<int32_t> Qwen4ExpLocalTextModel::DecodeStepMtpGreedy(int32_t, int64_t) {
  Unsupported("MTP", "no MTP head is wired up for this model");
}

std::vector<int32_t> Qwen4ExpLocalTextModel::DecodeStepMtpSampled(int32_t, int64_t, const kernels::SampleParams&,
                                                                  std::mt19937_64&) {
  Unsupported("MTP", "no MTP head is wired up for this model");
}

std::vector<int32_t> Qwen4ExpLocalTextModel::DecodeStepDflashGreedy(int32_t, int64_t, float, int64_t, int64_t*) {
  Unsupported("DFlash", "no drafter exists for this model");
}

std::vector<int32_t> Qwen4ExpLocalTextModel::DecodeStepDflashSampled(int32_t, int64_t, float, int64_t,
                                                                     const kernels::SampleParams&, std::mt19937_64&,
                                                                     int64_t*) {
  Unsupported("DFlash", "no drafter exists for this model");
}

StepProfile Qwen4ExpLocalTextModel::DecodeStepProfiled(int32_t) {
  Unsupported("the per-block step profile", "Qwen4ExpModel::LastTimings() reports per-stage wall times instead");
}

StepProfile Qwen4ExpLocalTextModel::PrefillProfiled(const std::vector<int32_t>&) {
  Unsupported("the per-block prefill profile", "Qwen4ExpModel::LastTimings() reports per-stage wall times instead");
}

Qwen4ExpModelOptions MakeQwen4ExpModelOptions(const ModelOptions& opts) {
  if (opts.mtp_draft_k > 0) {
    throw std::invalid_argument("LoadTextModel: MTP is not available for qwen4_exp (no MTP head is wired up for this model)");
  }
  if (!opts.dflash_container.empty() || opts.dflash_draft_k > 0) {
    throw std::invalid_argument("LoadTextModel: DFlash is not available for qwen4_exp (no drafter exists for this model)");
  }
  if (opts.vision == ModelOptions::VisionMode::kOn) {
    throw std::invalid_argument("LoadTextModel: --vision on is not available for qwen4_exp (the Qwen3-VL tower is milestone 4); "
                                "use --vision off or auto");
  }
  if (opts.layer_limit >= 0) {
    throw std::invalid_argument("LoadTextModel: layer_limit is not supported for qwen4_exp (the container is loaded whole)");
  }
  if (opts.tp.world != 1) {
    throw std::invalid_argument("LoadTextModel: a tensor-parallel rank is not supported for qwen4_exp");
  }
  if (opts.stage_only.role != stage::Role::kOff) {
    throw std::invalid_argument("LoadTextModel: stage-only loads (hybrid mode) are not supported for qwen4_exp");
  }
  Qwen4ExpModelOptions o;
  o.container_path = opts.container_path;
  o.layout = opts.layout;
  o.max_ctx = opts.max_ctx;
  o.prompt_checkpoint = opts.prompt_checkpoint;
  // devices / split: Qwen4ExpModel::Load reads R4DX_Q4_DEVICES / R4DX_Q4_SPLIT (ModelOptions has no field for them)
  return o;
}

std::unique_ptr<TextModel> LoadQwen4ExpTextModel(const ModelOptions& opts, const TpOptions& tp) {
  if (tp.world != 1) {
    throw std::invalid_argument("LoadTextModel: --tp " + std::to_string(tp.world) +
                                " is not supported for qwen4_exp: its two cards are a layer split inside the model, "
                                "not tensor parallelism");
  }
  const TpOptions d;
  if (tp.mode != d.mode || !tp.devices.empty() || tp.noop_rank != d.noop_rank || tp.ar_timeout_ms != d.ar_timeout_ms ||
      tp.ar_nb_small != d.ar_nb_small || tp.ar_nb_large != d.ar_nb_large || tp.submit_layers != d.submit_layers ||
      tp.max_inflight_units != d.max_inflight_units || tp.fault_rank != d.fault_rank ||
      tp.fault_at_allreduce != d.fault_at_allreduce || tp.fault_kind != d.fault_kind) {
    throw std::invalid_argument("LoadTextModel: the TpOptions besides `world` are tensor-parallel options (not supported "
                                "for qwen4_exp; the layer split is R4DX_Q4_DEVICES / R4DX_Q4_SPLIT)");
  }
  return std::make_unique<Qwen4ExpLocalTextModel>(Qwen4ExpModel::Load(MakeQwen4ExpModelOptions(opts)));
}

}  // namespace r4dx::model
