// r4dx::model::Lfm2LocalTextModel -- the TP=1 TextModel of an lfm2_moe container (docs/lfm2-plan.md 3.1, task L1-8): one
// Lfm2Model on the calling thread's current HIP device, every method a forward to the Lfm2Model method of the same name.
// No vision, audio, MTP or DFlash: HasVision() / MtpEnabled() / DflashEnabled() are false, an image span or an MTP / DFlash
// decode throws naming the feature. LoadLfm2TextModel (lfm2_local_text_model.cpp) is the LFM2 branch of LoadTextModel.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "lfm2_model.h"
#include "text_model.h"

namespace r4dx::model {

struct ModelOptions;  // model.h

class Lfm2LocalTextModel final : public TextModel {
 public:
  explicit Lfm2LocalTextModel(Lfm2Model m) : m_(std::move(m)) {}

  Lfm2Model& model() { return m_; }
  const Lfm2Model& model() const { return m_; }

  const ModelConfig& Config() const override { return m_.GetContainer().GenericConfig(); }
  const std::string& ModelId() const override { return m_.GetContainer().ModelId(); }
  int64_t ImageTokenId() const override { return -1; }
  int64_t VisionMergeSize() const override { return 2; }  // "2 if no vision" (text_model.h)
  bool HasVision() const override { return false; }
  bool MtpEnabled() const override { return false; }
  bool MtpUsingReducedVocabDraft() const override { return false; }
  bool DflashEnabled() const override { return false; }
  int64_t SampledFallbackRows() const override { return m_.SampledFallbackRows(); }
  int64_t PositionCount() const override { return m_.PositionCount(); }
  int64_t NumLoadedLayers() const override { return m_.GetContainer().NumLoadedLayers(); }
  int TpWorld() const override { return 1; }
  std::vector<VramReport> Vram() const override;

  void Reset() override { m_.Reset(); }
  void SetDflashInjectionEnabled(bool) override {}  // no drafter: nothing to inject
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
  Lfm2Model m_;
};

// ModelOptions -> the engine's options (shared by the TP=1 wrapper and Lfm2TpModel, which then fills the per-rank tp_*
// fields). Refuses what LFM2 does not have: MTP, DFlash, --vision on. Context: ModelOptions::max_ctx is honoured when it
// differs from its Qwen default (262144); the default itself means the checkpoint's native 131072 (as Gemma). The prefill
// chunk: ModelOptions::prefill_chunk when in [1, 256], else 256.
Lfm2ModelOptions MakeLfm2ModelOptions(const ModelOptions& opts);

}  // namespace r4dx::model
