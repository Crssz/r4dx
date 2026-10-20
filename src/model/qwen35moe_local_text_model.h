// r4dx::model::Qwen35MoeLocalTextModel -- the TextModel of a qwen3_5_moe container (docs/qwen35moe-plan.md M3): one
// Qwen35MoeModel, which itself drives its one or two stages (devices) on the calling thread, every method a forward to the
// Qwen35MoeModel method of the same name. MTP, DFlash, vision and audio are not supported yet (MTP and vision are plan M6;
// no DFlash drafter exists): the model options refuse them by name at load, their methods throw std::runtime_error naming
// the feature, and MtpEnabled() / HasVision() / DflashEnabled() are false. TpWorld() is 1: the two cards are a layer split
// inside the model, not tensor parallelism. LoadQwen35MoeTextModel (declared in text_model.h) is the qwen3_5_moe branch of
// LoadTextModel.
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "qwen35moe_model.h"
#include "text_model.h"

namespace r4dx::model {

struct ModelOptions;  // model.h

class Qwen35MoeLocalTextModel final : public TextModel {
 public:
  explicit Qwen35MoeLocalTextModel(std::unique_ptr<Qwen35MoeModel> m) : m_(std::move(m)) {}

  Qwen35MoeModel& model() { return *m_; }
  const Qwen35MoeModel& model() const { return *m_; }

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
  std::unique_ptr<Qwen35MoeModel> m_;
};

// ModelOptions + the R4DX_Q35_* environment (devices, split) -> the engine's options, with the requests this arch does not
// serve (MTP, DFlash, vision on, a partial layer load, a tensor-parallel rank) refused by name. The layout is the container's
// own rule (Qwen35MoeContainer::Load): trellis for a trellis container, bf16 / w4a16 otherwise, unless ModelOptions names one.
Qwen35MoeModelOptions MakeQwen35MoeModelOptions(const ModelOptions& opts);

}  // namespace r4dx::model
