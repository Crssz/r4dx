// r4dx::model::HyV3Model -- single-device hy_v3 (Hy-MT3-30B-A3B) text engine.
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "hy_v3_container.h"
#include "lfm2_layers.h"
#include "model_types.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/kernels/summary_sampler.hpp"
#include "r4dx/model/attention/lfm2_attention_layer.hpp"

namespace r4dx::model {

class SpanAccumulator;

struct HyV3ModelOptions {
  std::string container_path;
  Layout layout = Layout::kW4a16;
  int64_t layer_limit = -1;
  int64_t max_ctx = 0;
  int prefill_chunk = 256;
  bool prompt_checkpoint = false;
};

class HyV3Model {
 public:
  static HyV3Model Load(const HyV3ModelOptions& opts);

  HyV3Model(HyV3Model&&) = default;
  HyV3Model& operator=(HyV3Model&&) = default;

  const HyV3Config& Config() const { return container_.Config(); }
  const HyV3Container& GetContainer() const { return container_; }
  int64_t MaxCtx() const { return max_ctx_; }
  int64_t PositionCount() const { return pos_; }
  int64_t SampledFallbackRows() const { return sampled_fallback_rows_; }
  int64_t KvBytes() const { return kv_bytes_; }

  void Reset();
  void SaveCheckpoint();
  void RestoreCheckpoint();

  std::vector<float> Prefill(const std::vector<int32_t>& token_ids);
  std::vector<float> DecodeStep(int32_t token_id);
  int32_t DecodeStepGreedy(int32_t token_id);
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng);
  StepProfile DecodeStepProfiled(int32_t token_id);
  StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids);

 private:
  HyV3Model() = default;
  struct SummaryRequest {
    float inv_temperature = 1.0f;
    kernels::RowSummary* out = nullptr;
  };

  void UploadPositionsAsync(int64_t start_pos, int64_t T);
  void RunLayer(int64_t i, uint16_t* h, uint16_t* x_cur, const uint16_t* next_norm_w, uint16_t* x_next,
                int64_t T, int64_t start_pos, SpanAccumulator* prof);
  std::vector<float> RunChunk(const std::vector<int32_t>& token_ids, bool want_logits, int32_t* greedy_out,
                              const SummaryRequest* summary_out, SpanAccumulator* prof);
  void FetchRowSummary(float inv_temperature, kernels::RowSummary* out);

  struct GraphExecDeleter {
    void operator()(void* p) const noexcept;
  };
  void RunDecodeBody(uint16_t* h, uint16_t* x, int64_t start_pos, bool want_logits,
                     bool greedy, const SummaryRequest* summary_out, SpanAccumulator* prof);
  void InvalidateDecodeGraph() noexcept;

  HyV3ModelOptions opts_;
  HyV3Container container_;
  core::Stream stream_;
  core::Arena arena_;

  core::PinnedBuffer<int32_t> ids_host_, positions_host_, seqused_host_;
  core::DeviceBuffer<int32_t> ids_dev_, positions_dev_, seqused_dev_, argmax_dev_;
  core::DeviceBuffer<uint16_t> buf_h_, buf_x_;
  core::DeviceBuffer<float> logits_dev_, kv_descale_ones_;
  core::DeviceBuffer<int32_t> summary_ids_dev_;
  core::DeviceBuffer<float> summary_vals_dev_, summary_lse_dev_;
  Lfm2MoeHost moe_host_;

  std::vector<attention::GemmaKvCache> kv_;
  bool kv_fp8_ = true;
  bool decode_graph_enabled_ = true;
  std::unique_ptr<void, GraphExecDeleter> decode_graph_exec_;
  int64_t decode_graph_sb_ = -1;
  int decode_graph_mode_ = -1;
  float decode_graph_inv_temp_ = 0.0f;

  int64_t max_ctx_ = 0;
  int64_t max_chunk_ = 256;
  int64_t pos_ = 0;
  int64_t ckpt_pos_ = -1;
  int64_t kv_bytes_ = 0;
  int64_t sampled_fallback_rows_ = 0;
  bool started_ = false;
};

}  // namespace r4dx::model
