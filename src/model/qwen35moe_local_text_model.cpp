#include "qwen35moe_local_text_model.h"

#include <stdexcept>
#include <string>

#include "model.h"  // ModelOptions
#include "qwen35moe_tp_model.h"

namespace r4dx::model {

namespace {

[[noreturn]] void Unsupported(const std::string& feature, const std::string& why) {
  throw std::runtime_error("Qwen35MoeLocalTextModel: " + feature + " is not available for qwen3_5_moe yet (" + why + ")");
}

}  // namespace

void Qwen35MoeLocalTextModel::SetDflashInjectionEnabled(bool enabled) {
  // The server calls this with `true` only when a drafter container is configured (DflashEnabled()).
  m_->SetDflashInjectionEnabled(enabled);
}

void Qwen35MoeLocalTextModel::EncodeImages(const float*, int64_t, const std::vector<vision::GridThw>&, ImageRows*,
                                           vision::VisionEncodeStats*) {
  Unsupported("vision", "the vision tower is plan milestone M6");
}

std::vector<float> Qwen35MoeLocalTextModel::PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                                              const std::vector<ImageSpan>& images) {
  if (!images.empty()) Unsupported("vision", "the vision tower is plan milestone M6");
  return m_->Prefill(token_ids);  // no images: a plain Prefill
}

std::vector<int32_t> Qwen35MoeLocalTextModel::DecodeStepMtpGreedy(int32_t, int64_t) {
  Unsupported("MTP", "the MTP head is plan milestone M6");
}

std::vector<int32_t> Qwen35MoeLocalTextModel::DecodeStepMtpSampled(int32_t, int64_t, const kernels::SampleParams&,
                                                                   std::mt19937_64&) {
  Unsupported("MTP", "the MTP head is plan milestone M6");
}

std::vector<int32_t> Qwen35MoeLocalTextModel::DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                                                      int64_t* walk_len_out) {
  return m_->DecodeStepDflashGreedy(token_id, k, p_min, n_min, walk_len_out);
}

std::vector<int32_t> Qwen35MoeLocalTextModel::DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                                                      const kernels::SampleParams& params,
                                                                      std::mt19937_64& rng, int64_t* walk_len_out) {
  if (params.temperature > 0.0f) {
    // A sampled DFlash round (rejection sampling over the drafter) is not implemented for this model: take one plain sampled
    // step instead. It commits one row and injects it, so the drafter frontier stays == PositionCount() and a later greedy round
    // on the same request still works. Correct output, no speculation speedup.
    if (walk_len_out != nullptr) *walk_len_out = 0;
    return {m_->DecodeStepSampled(token_id, params, rng)};
  }
  return m_->DecodeStepDflashGreedy(token_id, k, p_min, n_min, walk_len_out);
}

StepProfile Qwen35MoeLocalTextModel::DecodeStepProfiled(int32_t) {
  Unsupported("the per-block step profile", "Qwen35MoeModel::LastTimings() reports per-stage wall times instead");
}

StepProfile Qwen35MoeLocalTextModel::PrefillProfiled(const std::vector<int32_t>&) {
  Unsupported("the per-block prefill profile", "Qwen35MoeModel::LastTimings() reports per-stage wall times instead");
}

Qwen35MoeModelOptions MakeQwen35MoeModelOptions(const ModelOptions& opts) {
  if (opts.mtp_draft_k > 0) {
    throw std::invalid_argument("LoadTextModel: MTP is not available for qwen3_5_moe yet (plan milestone M6)");
  }
  if (opts.dflash_container.empty() != (opts.dflash_draft_k <= 0)) {
    throw std::invalid_argument("LoadTextModel: --dflash <container> and --dflash-k N go together for qwen3_5_moe");
  }
  if (!opts.dflash_container.empty() && opts.mtp_draft_k > 0) {
    throw std::invalid_argument("LoadTextModel: --dflash and --mtp are exclusive");
  }
  if (opts.vision == ModelOptions::VisionMode::kOn) {
    throw std::invalid_argument("LoadTextModel: --vision on is not available for qwen3_5_moe yet (plan milestone M6); "
                                "use --vision off or auto");
  }
  if (opts.layer_limit >= 0) {
    throw std::invalid_argument("LoadTextModel: layer_limit is not supported for qwen3_5_moe (the container is loaded whole)");
  }
  if (opts.tp.world != 1) {
    throw std::invalid_argument("LoadTextModel: ModelOptions::tp must be default for qwen3_5_moe -- Qwen35MoeTpModel fills in every rank's options itself");
  }
  if (opts.stage_only.role != stage::Role::kOff) {
    throw std::invalid_argument("LoadTextModel: stage-only loads (hybrid mode) are not supported for qwen3_5_moe");
  }
  Qwen35MoeModelOptions o;
  o.container_path = opts.container_path;
  o.layout = opts.layout;
  o.max_ctx = opts.max_ctx;
  o.prompt_checkpoint = opts.prompt_checkpoint;
  o.dflash_container = opts.dflash_container;
  o.dflash_draft_k = opts.dflash_draft_k;
  // devices / split: Qwen35MoeModel::Load reads R4DX_Q35_DEVICES / R4DX_Q35_SPLIT (ModelOptions has no field for them)
  return o;
}

std::unique_ptr<TextModel> LoadQwen35MoeTextModel(const ModelOptions& opts, const TpOptions& tp) {
  // --tp 2 (docs/qwen35moe-tp-design.md P3): tensor parallelism over the two cards, opt-in; the default (--tp 1) stays the layer split inside
  // Qwen35MoeModel. The weights of both modes together do not fit, so the user picks one at load.
  if (tp.world == 2) return Qwen35MoeTpModel::Load(opts, tp);
  if (tp.world != 1) {
    throw std::invalid_argument("LoadTextModel: --tp " + std::to_string(tp.world) + " is not supported for qwen3_5_moe (1 = the layer split, 2 = tensor parallel)");
  }
  const TpOptions d;
  if (tp.mode != d.mode || !tp.devices.empty() || tp.noop_rank != d.noop_rank || tp.ar_timeout_ms != d.ar_timeout_ms ||
      tp.ar_nb_small != d.ar_nb_small || tp.ar_nb_large != d.ar_nb_large || tp.submit_layers != d.submit_layers ||
      tp.max_inflight_units != d.max_inflight_units || tp.fault_rank != d.fault_rank ||
      tp.fault_at_allreduce != d.fault_at_allreduce || tp.fault_kind != d.fault_kind) {
    throw std::invalid_argument("LoadTextModel: the TpOptions besides `world` are tensor-parallel options (not supported "
                                "for qwen3_5_moe; the layer split is R4DX_Q35_DEVICES / R4DX_Q35_SPLIT)");
  }
  return std::make_unique<Qwen35MoeLocalTextModel>(Qwen35MoeModel::Load(MakeQwen35MoeModelOptions(opts)));
}

}  // namespace r4dx::model
