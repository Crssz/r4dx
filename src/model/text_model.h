// r4dx::model::TextModel -- the model surface src/cli (and, from docs/tp.md P5, src/server) call
// (docs/tp.md 2.8): exactly the Model methods those callers use today, with two deliberate
// substitutions, because a device pointer and a Container& cannot mean anything across two devices:
// GetContainer() is replaced by the cached accessors ModelId() / ImageTokenId() / VisionMergeSize()
// / NumLoadedLayers(), and EncodeImages writes an ImageRows instead of a core::DeviceBuffer.
//
// Two implementations:
//   * LocalTextModel (local_text_model.h) -- TP=1: one Model, every method a one-line forward.
//     Byte-for-byte today's behaviour.
//   * TpModel (tp_model.h) -- tensor parallel (`--tp 2`): one rank thread per rank, each owning its
//     own Model, the facade broadcasting every call, reconciling the results and owning the error
//     state machine (docs/tp.md 2.2-2.9).
//
// LoadTextModel picks between them. This header deliberately does NOT include model.h: the forward
// contracts are Model's methods of the same name, documented there.
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "model_config.h"
#include "model_types.h"
#include "r4dx/kernels/sampler.hpp"  // kernels::SampleParams (header-only, HIP-free)

namespace r4dx::vision {
struct VisionEncodeStats;
}

namespace r4dx::model {

struct ModelOptions;  // model.h

// The `--tp*` flags (docs/tp.md 9.1).
struct TpOptions {
  int world = 1;  // 1 or 2
  // kReal: one rank per GPU (docs/tp.md P4); kEmulate: both ranks on one device, host-synchronized
  // exact adds (6.5); kNoop: ONE rank's shard, no-op all-reduce -- timing only, tokens meaningless.
  enum class Mode { kReal, kEmulate, kNoop } mode = Mode::kReal;
  std::vector<int> devices;  // process-visible HIP ordinals, rank r -> entry r; empty = auto (9.2)
  int noop_rank = 0;         // kNoop: which shard to load
  int ar_timeout_ms = 500;   // [10, 1500]
  int ar_nb_small = 4;       // channel 0 blocks
  int ar_nb_large = 4;       // channel 1 blocks
  // Test-only fault injection. Set by tests directly (and, from docs/tp.md P5, by TpModel::Load from
  // the environment variable R4DX_TP_FAULT="<rank>:<n>:<kind>"). Fires ONCE, at the n-th
  // AllReduceSumBf16 of `rank` counted from the END of warm-up. kind 0: the endpoint throws
  // std::runtime_error("tp fault injection"); kind 1: it sleeps 700 ms before enqueuing (the peer's
  // all-reduce times out). Logged loudly when armed.
  int fault_rank = -1;
  int64_t fault_at_allreduce = -1;
  int fault_kind = 0;
};

// One rank's device memory, as hipMemGetInfo reports it for that rank's device.
struct VramReport {
  int rank = 0;
  int device = 0;
  double used_gib = 0, free_gib = 0, total_gib = 0;
};

class TextModel {
 public:
  virtual ~TextModel() = default;

  // ---- identity / capabilities (cached at load; HasVision is safe from any thread) -------------
  virtual const ModelConfig& Config() const = 0;  // GLOBAL config (vocab, head counts: unsharded)
  virtual const std::string& ModelId() const = 0;
  virtual int64_t ImageTokenId() const = 0;
  virtual int64_t VisionMergeSize() const = 0;  // vision_config.spatial_merge_size, 2 if no vision
  virtual bool HasVision() const = 0;
  virtual bool MtpEnabled() const = 0;
  virtual bool MtpUsingReducedVocabDraft() const = 0;
  virtual bool DflashEnabled() const = 0;
  virtual int64_t SampledFallbackRows() const = 0;
  virtual int64_t PositionCount() const = 0;
  virtual int64_t NumLoadedLayers() const = 0;
  virtual int TpWorld() const = 0;
  virtual std::vector<VramReport> Vram() const = 0;  // one entry per rank

  // ---- sequence state -------------------------------------------------------------------------
  virtual void Reset() = 0;
  virtual void SetDflashInjectionEnabled(bool enabled) = 0;

  // ---- vision ---------------------------------------------------------------------------------
  // Model::EncodeImages, with the merged rows written to `out` (device rows at TP=1; host rows
  // under TP, docs/tp.md 8.3) and out->rows() set.
  virtual void EncodeImages(const float* pixel_values, int64_t total_patches,
                            const std::vector<vision::GridThw>& grids, ImageRows* out,
                            vision::VisionEncodeStats* stats = nullptr) = 0;

  // ---- forward (identical contracts to Model's methods of the same name) ----------------------
  virtual std::vector<float> Prefill(const std::vector<int32_t>& token_ids) = 0;
  virtual std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                               const std::vector<ImageSpan>& images) = 0;
  virtual std::vector<float> DecodeStep(int32_t token_id) = 0;
  virtual int32_t DecodeStepGreedy(int32_t token_id) = 0;
  virtual int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params,
                                    std::mt19937_64& rng) = 0;
  virtual std::vector<int32_t> DecodeStepMtpGreedy(int32_t token_id, int64_t k) = 0;
  virtual std::vector<int32_t> DecodeStepMtpSampled(int32_t token_id, int64_t k,
                                                    const kernels::SampleParams& params,
                                                    std::mt19937_64& rng) = 0;
  virtual std::vector<int32_t> DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min,
                                                      int64_t n_min, int64_t* walk_len_out = nullptr) = 0;
  virtual std::vector<int32_t> DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min,
                                                       int64_t n_min, const kernels::SampleParams& params,
                                                       std::mt19937_64& rng,
                                                       int64_t* walk_len_out = nullptr) = 0;

  // ---- diagnostics (TP: throws core::TpUnsupportedError) ---------------------------------------
  virtual StepProfile DecodeStepProfiled(int32_t token_id) = 0;
  virtual StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids) = 0;
};

// tp.world == 1 => LocalTextModel(Model::Load(opts)) -- exactly today's Model; every other TpOptions
// field must then keep its default. tp.world == 2 => TpModel::Load(opts, tp). Defined in
// tp_model.cpp.
std::unique_ptr<TextModel> LoadTextModel(const ModelOptions& opts, const TpOptions& tp);

}  // namespace r4dx::model
