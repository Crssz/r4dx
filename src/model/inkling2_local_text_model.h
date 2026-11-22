// r4dx::model::Inkling2LocalTextModel -- the TP=1 TextModel of an inkling2_loop container (docs/inkling2-plan.md 5): one
// Inkling2Model on the calling thread's current HIP device, every method a forward to the Inkling2Model method of the
// same name. Text only: HasVision() / MtpEnabled() / DflashEnabled() are false, an image span or an MTP / DFlash decode
// throws naming the feature. LoadInkling2TextModel (inkling2_local_text_model.cpp) is the inkling2_loop branch of
// LoadTextModel: --tp 1 -> this, --tp 2 -> Inkling2TpModel.
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "inkling2_model.h"
#include "text_model.h"

namespace r4dx::model {

struct ModelOptions;  // model.h

class Inkling2LocalTextModel final : public TextModel {
 public:
  explicit Inkling2LocalTextModel(Inkling2Model m) : m_(std::move(m)) {}

  Inkling2Model& model() { return m_; }
  const Inkling2Model& model() const { return m_; }

  const ModelConfig& Config() const override { return m_.GetContainer().GenericConfig(); }
  const std::string& ModelId() const override { return m_.GetContainer().ModelId(); }
  int64_t ImageTokenId() const override { return -1; }
  int64_t VisionMergeSize() const override { return 2; }
  bool HasVision() const override { return false; }
  bool MtpEnabled() const override { return false; }
  bool MtpUsingReducedVocabDraft() const override { return false; }
  bool DflashEnabled() const override { return m_.DsparkEnabled(); }
  int64_t SampledFallbackRows() const override { return m_.SampledFallbackRows(); }
  int64_t PositionCount() const override { return m_.PositionCount(); }
  int64_t NumLoadedLayers() const override { return m_.GetContainer().NumLoadedLayers(); }
  int TpWorld() const override { return 1; }
  std::vector<VramReport> Vram() const override;

  void Reset() override { m_.Reset(); }
  void SetDflashInjectionEnabled(bool) override {}
  void SaveCheckpoint() override { m_.SaveCheckpoint(); }
  void RestoreCheckpoint() override { m_.RestoreCheckpoint(); }

  void EncodeImages(const float*, int64_t, const std::vector<vision::GridThw>&, ImageRows*,
                    vision::VisionEncodeStats* = nullptr) override;

  std::vector<float> Prefill(const std::vector<int32_t>& token_ids) override { return m_.Prefill(token_ids); }
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                       const std::vector<ImageSpan>& images) override;
  std::vector<float> DecodeStep(int32_t token_id) override { return m_.DecodeStep(token_id); }
  int32_t DecodeStepGreedy(int32_t token_id) override { return m_.DecodeStepGreedy(token_id); }
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng) override {
    return m_.DecodeStepSampled(token_id, params, rng);
  }
  std::vector<int32_t> DecodeStepMtpGreedy(int32_t, int64_t) override;
  std::vector<int32_t> DecodeStepMtpSampled(int32_t, int64_t, const kernels::SampleParams&, std::mt19937_64&) override;
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t, int64_t, float, int64_t, int64_t* = nullptr) override;
  std::vector<int32_t> DecodeStepDflashSampled(int32_t, int64_t, float, int64_t, const kernels::SampleParams&,
                                               std::mt19937_64&, int64_t* = nullptr) override;

  StepProfile DecodeStepProfiled(int32_t token_id) override { return m_.DecodeStepProfiled(token_id); }
  StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids) override { return m_.PrefillProfiled(token_ids); }

 private:
  Inkling2Model m_;
};

// ModelOptions -> the engine's options (shared with Inkling2TpModel). Refuses what milestone 1 does not have: MTP, DFlash,
// --vision on, --layers (the looped stack cannot be truncated), --batch. max_ctx: ModelOptions::max_ctx as given (r4dx-cli /
// r4dx-server resolve it with ResolveContainerMaxCtx: default 131072, up to 1048576); <= 0 or ModelOptions' own default (262144) -> 131072.
Inkling2ModelOptions MakeInkling2ModelOptions(const ModelOptions& opts);

// The inkling2_loop branch of LoadTextModel (arch.h: DetectArch == kInkling2Loop).
std::unique_ptr<TextModel> LoadInkling2TextModel(const ModelOptions& opts, const TpOptions& tp);

}  // namespace r4dx::model
