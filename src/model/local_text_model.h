// r4dx::model::LocalTextModel -- the TP=1 TextModel (docs/tp.md 2.8): one Model, on the calling
// thread's current HIP device, every method a one-line forward to the Model method of the same
// name. No behaviour change: the CLI / tools going through this run exactly the calls, in exactly
// the order, they made on a Model before TextModel existed.
#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

#include "model.h"
#include "text_model.h"

namespace r4dx::model {

class LocalTextModel final : public TextModel {
 public:
  explicit LocalTextModel(Model m) : m_(std::move(m)) {}

  // The wrapped Model, for tests and tools that also need a Model-only accessor.
  Model& model() { return m_; }
  const Model& model() const { return m_; }

  const ModelConfig& Config() const override { return m_.Config(); }
  const std::string& ModelId() const override { return m_.GetContainer().ModelId(); }
  int64_t ImageTokenId() const override { return m_.GetContainer().ImageTokenId(); }
  int64_t VisionMergeSize() const override;
  bool HasVision() const override { return m_.HasVision(); }
  bool MtpEnabled() const override { return m_.MtpEnabled(); }
  bool MtpUsingReducedVocabDraft() const override { return m_.MtpUsingReducedVocabDraft(); }
  bool DflashEnabled() const override { return m_.DflashEnabled(); }
  int64_t SampledFallbackRows() const override { return m_.SampledFallbackRows(); }
  int64_t PositionCount() const override { return m_.PositionCount(); }
  int64_t NumLoadedLayers() const override { return m_.GetContainer().NumLoadedLayers(); }
  int TpWorld() const override { return 1; }
  // One hipMemGetInfo on the calling thread's current device.
  std::vector<VramReport> Vram() const override;

  void Reset() override { m_.Reset(); }
  void SetDflashInjectionEnabled(bool enabled) override { m_.SetDflashInjectionEnabled(enabled); }

  void EncodeImages(const float* pixel_values, int64_t total_patches,
                    const std::vector<vision::GridThw>& grids, ImageRows* out,
                    vision::VisionEncodeStats* stats = nullptr) override;

  std::vector<float> Prefill(const std::vector<int32_t>& token_ids) override { return m_.Prefill(token_ids); }
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                       const std::vector<ImageSpan>& images) override {
    return m_.PrefillMultimodal(token_ids, images);
  }
  std::vector<float> DecodeStep(int32_t token_id) override { return m_.DecodeStep(token_id); }
  int32_t DecodeStepGreedy(int32_t token_id) override { return m_.DecodeStepGreedy(token_id); }
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params,
                            std::mt19937_64& rng) override {
    return m_.DecodeStepSampled(token_id, params, rng);
  }
  std::vector<int32_t> DecodeStepMtpGreedy(int32_t token_id, int64_t k) override {
    return m_.DecodeStepMtpGreedy(token_id, k);
  }
  std::vector<int32_t> DecodeStepMtpSampled(int32_t token_id, int64_t k, const kernels::SampleParams& params,
                                            std::mt19937_64& rng) override {
    return m_.DecodeStepMtpSampled(token_id, k, params, rng);
  }
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                              int64_t* walk_len_out = nullptr) override {
    return m_.DecodeStepDflashGreedy(token_id, k, p_min, n_min, walk_len_out);
  }
  std::vector<int32_t> DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                               const kernels::SampleParams& params, std::mt19937_64& rng,
                                               int64_t* walk_len_out = nullptr) override {
    return m_.DecodeStepDflashSampled(token_id, k, p_min, n_min, params, rng, walk_len_out);
  }

  StepProfile DecodeStepProfiled(int32_t token_id) override { return m_.DecodeStepProfiled(token_id); }
  StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids) override {
    return m_.PrefillProfiled(token_ids);
  }

 private:
  Model m_;
};

}  // namespace r4dx::model
