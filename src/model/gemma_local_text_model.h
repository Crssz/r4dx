// r4dx::model::GemmaLocalTextModel -- the TP=1 TextModel of a gemma4_unified container (docs/gemma4-plan.md 3.2,
// task M1-20): one GemmaModel on the calling thread's current HIP device, every method a forward to the
// GemmaModel method of the same name. MTP, DFlash (until the drafter track lands its hooks) and images (M2)
// throw std::runtime_error naming the feature; MtpEnabled() / DflashEnabled() / HasVision() are false.
// LoadGemmaTextModel (gemma_local_text_model.cpp) is the Gemma branch of LoadTextModel.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "gemma_model.h"
#include "text_model.h"

namespace r4dx::model {

class GemmaLocalTextModel final : public TextModel {
 public:
  explicit GemmaLocalTextModel(GemmaModel m) : m_(std::move(m)) {}

  GemmaModel& model() { return m_; }
  const GemmaModel& model() const { return m_; }

  const ModelConfig& Config() const override { return m_.GetContainer().GenericConfig(); }
  const std::string& ModelId() const override { return m_.GetContainer().ModelId(); }
  int64_t ImageTokenId() const override { return m_.GetContainer().Info().image_token_id; }
  // Vision (M2): the Gemma processor already merges 3x3 teacher patches, so the grids this model sees are in
  // MERGED cells and GridThw::MergedTokenCount(1) is the soft-token count.
  int64_t VisionMergeSize() const override { return 1; }
  bool HasVision() const override { return m_.HasVision(); }
  int64_t ImageBoiTokenId() const override { return m_.GetContainer().Info().boi_token_id; }
  int64_t ImageEoiTokenId() const override { return m_.GetContainer().Info().eoi_token_id; }
  bool MtpEnabled() const override { return false; }
  bool MtpUsingReducedVocabDraft() const override { return false; }
  bool DflashEnabled() const override { return false; }
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
  GemmaModel m_;
};

}  // namespace r4dx::model
