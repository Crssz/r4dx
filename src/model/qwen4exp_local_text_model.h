// r4dx::model::Qwen4ExpLocalTextModel -- the TextModel of a qwen4_exp container (docs/qwen4exp-plan.md 4.1, 4.11 item 4,
// task M1-16): one Qwen4ExpModel, which itself drives its one or two stages (devices) on the calling thread, every method a
// forward to the Qwen4ExpModel method of the same name. MTP, DFlash, vision and audio are not supported for this model
// (no MTP head or DFlash drafter is wired up; vision is a later milestone): the model options refuse them by name at
// load, their methods throw std::runtime_error naming the feature, and MtpEnabled() / HasVision() / DflashEnabled()
// are false. TpWorld() is 1: the two cards are a layer split inside the model, not tensor parallelism.
// LoadQwen4ExpTextModel (declared in text_model.h) is the qwen4_exp branch of LoadTextModel.
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "qwen4exp_model.h"
#include "text_model.h"

namespace r4dx::model {

struct ModelOptions;  // model.h

class Qwen4ExpLocalTextModel final : public TextModel {
 public:
  explicit Qwen4ExpLocalTextModel(std::unique_ptr<Qwen4ExpModel> m) : m_(std::move(m)) {}

  Qwen4ExpModel& model() { return *m_; }
  const Qwen4ExpModel& model() const { return *m_; }

  const ModelConfig& Config() const override { return m_->GenericConfig(); }
  const std::string& ModelId() const override { return m_->ModelId(); }
  int64_t ImageTokenId() const override { return -1; }
  int64_t VisionMergeSize() const override { return 2; }  // the default of a model without a vision tower
  bool HasVision() const override { return false; }
  bool MtpEnabled() const override { return false; }
  bool MtpUsingReducedVocabDraft() const override { return false; }
  bool DflashEnabled() const override { return false; }
  int64_t SampledFallbackRows() const override { return m_->SampledFallbackRows(); }
  int64_t PositionCount() const override { return m_->PositionCount(); }
  int64_t NumLoadedLayers() const override { return m_->NumLoadedLayers(); }
  int TpWorld() const override { return 1; }
  std::vector<VramReport> Vram() const override { return m_->Vram(); }

  void Reset() override { m_->Reset(); }
  void SetDflashInjectionEnabled(bool enabled) override;
  void SaveCheckpoint() override { m_->SaveCheckpoint(); }
  void RestoreCheckpoint() override { m_->RestoreCheckpoint(); }

  void EncodeImages(const float*, int64_t, const std::vector<vision::GridThw>&, ImageRows*,
                    vision::VisionEncodeStats* = nullptr) override;

  std::vector<float> Prefill(const std::vector<int32_t>& token_ids) override { return m_->Prefill(token_ids); }
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                       const std::vector<ImageSpan>& images) override;
  std::vector<float> DecodeStep(int32_t token_id) override { return m_->DecodeStep(token_id); }
  int32_t DecodeStepGreedy(int32_t token_id) override { return m_->DecodeStepGreedy(token_id); }
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng) override {
    return m_->DecodeStepSampled(token_id, params, rng);
  }
  std::vector<int32_t> DecodeStepMtpGreedy(int32_t token_id, int64_t k) override;
  std::vector<int32_t> DecodeStepMtpSampled(int32_t token_id, int64_t k, const kernels::SampleParams& params,
                                            std::mt19937_64& rng) override;
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t, int64_t, float, int64_t, int64_t* = nullptr) override;
  std::vector<int32_t> DecodeStepDflashSampled(int32_t, int64_t, float, int64_t, const kernels::SampleParams&,
                                               std::mt19937_64&, int64_t* = nullptr) override;

  StepProfile DecodeStepProfiled(int32_t token_id) override;
  StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids) override;

 private:
  std::unique_ptr<Qwen4ExpModel> m_;
};

// ModelOptions + the R4DX_Q4_* environment (devices, split) -> the engine's options, with the requests this arch does not
// serve (MTP, DFlash, vision on, a partial layer load, a tensor-parallel rank) refused by name. The layout is the
// container's own rule (Qwen4ExpContainer::Load): trellis for a trellis container, bf16 / w4a16 otherwise.
Qwen4ExpModelOptions MakeQwen4ExpModelOptions(const ModelOptions& opts);

}  // namespace r4dx::model
