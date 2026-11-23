// r4dx::model::Inkling2Model -- the inkling2_loop (Inkling-2-Mini) text engine on one device, and one tensor-parallel rank
// of it (Inkling2TpModel drives two). docs/inkling2-plan.md 3.3 (the graph, with the lane-D corrections: the pre-loop
// block list holds ONE block, the embedding; layers 6 and 12 append in every pass; a pass >= 2 restarts from (h_pre,
// B_pre) and its messages attend over [the block appended at layer 12 of the previous pass, that pass's output]).
//
// RunChunk (T <= prefill_chunk <= 256 rows) walks the 48-entry visit table (Inkling2Config::Visits) once per chunk. The
// residual h, the block states B and the messages are fp32 [T, H] buffers from a small pool (the attention residual is
// per position, so a chunk needs nothing from the previous chunk but the caches). Per visit (stored layer i, slot s):
//   a     = i ? AttnRes(attn_query[i], B + [h]) : h                  (full layers: 3 queries -> aq, ak, av)
//   [pass >= 2: m = AttnRes(msg_query[j], [prev block, prev h]) (4 queries); aq/ak/av = gate_{q,k,v}(a*, m_*)]
//   xq, xk, xv = RMS_w(attn_norm | q_norm, k_norm, v_norm)  -> bf16 GEMM inputs
//   o     = Attention(xq, xk, xv) bf16 [T, H]   (K/V: linear -> short conv (static | dynamic) -> clip | rms -> the slot's
//                                              bf16 paged KV; relative bias + sinks; sigmoid gate; o_proj [TP: all-reduce])
//   boundary (i % block == 0): B += [h]; h = sconv(o)      else h = h + sconv(o)        (fp32, the slot's conv state 2)
//   y     = AttnRes(mlp_query[i], B + [h])  [pass >= 2: y = gate_mlp(y, m_mlp)];  x = RMS_w(mlp_norm, y)
//   f     = MoE(x) bf16 [T, H]   (sigmoid router + bias, top-8, one host sync, per-expert QuantLinears, fp32 combine
//                                [TP: all-reduce])
//   h     = h + sconv(f)                                                                (fp32, conv state 3)
// then h = RMS_w(final_norm, AttnRes(final_query, B + [h])) on the last row, the lm_head (the rank's stored-row slice),
// logits = 30 tanh(x / 30), the first unpadded_vocab_size (200058) columns.
//
// Caches, one set per visit slot (48): K/V bf16, paged in attn_ref.h's layout (block 256): a full-attention slot holds
// max_ctx positions, a sliding slot a ring of ceil((window + chunk - 1) / 256) + 1 blocks (its block table maps logical
// block b to b mod ring); 4 conv states fp32 [3, C] (k, v, post-attention, post-MLP). Reset zeroes the conv states (KV is
// positional). Save/RestoreCheckpoint (prompt_checkpoint) copy the conv states and the sliding rings (a ring position
// can be overwritten by the tokens after the checkpoint); the full-attention KV needs no copy.
#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "inkling2_container.h"
#include "inkling2_dspark.h"
#include "model_types.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/sampler.hpp"
#include "r4dx/kernels/summary_sampler.hpp"

namespace r4dx::model {

class SpanAccumulator;  // profile_span.h

struct Inkling2ModelOptions {
  std::string container_path;
  Layout layout = Layout::kMxfp8;
  int64_t max_ctx = 0;        // <= 0: Inkling2Config::ResolveMaxCtx default (131072)
  int prefill_chunk = 256;    // [1, 256]
  bool prompt_checkpoint = false;
  std::string dspark_path;    // optional path to mtp.safetensors (Inkling2 DSpark drafter)
  // Tensor parallel (Inkling2TpModel fills these per rank).
  int tp_world = 1;
  int tp_rank = 0;
  core::TpComm* tp_comm = nullptr;
};

// The router choice of one MoE visit (DebugRouterIds): [T, top_k] per visit, filled when enabled.
struct Inkling2RouterTrace {
  bool enabled = false;
  std::vector<std::vector<int32_t>> ids;  // [visit] -> [T * top_k]
};

class Inkling2Model {
 public:
  static Inkling2Model Load(const Inkling2ModelOptions& opts);

  Inkling2Model(Inkling2Model&&) = default;
  Inkling2Model& operator=(Inkling2Model&&) = default;

  const Inkling2Config& Config() const { return container_.Config(); }  // rank-local
  const Inkling2Container& GetContainer() const { return container_; }
  int64_t MaxCtx() const { return max_ctx_; }
  int64_t PositionCount() const { return pos_; }
  int64_t SampledFallbackRows() const { return sampled_fallback_rows_; }
  int64_t KvBytes() const { return kv_bytes_; }
  int64_t Vocab() const { return container_.GlobalConfig().unpadded_vocab_size; }  // the logits width
  int TpWorld() const { return opts_.tp_world; }
  bool DsparkEnabled() const { return dspark_ != nullptr; }

  void Reset();
  void SaveCheckpoint();
  void RestoreCheckpoint();

  std::vector<float> Prefill(const std::vector<int32_t>& token_ids);
  std::vector<float> DecodeStep(int32_t token_id);
  int32_t DecodeStepGreedy(int32_t token_id);
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params, std::mt19937_64& rng);
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t token_id, int64_t draft_k, int64_t* walk_len_out = nullptr);
  std::vector<int32_t> DecodeStepDflashSampled(int32_t token_id, int64_t draft_k, const kernels::SampleParams& params,
                                               std::mt19937_64& rng, int64_t* walk_len_out = nullptr);
  StepProfile DecodeStepProfiled(int32_t token_id);
  StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids);

  // TP only: one prefill chunk, decode greedy / sampled / full-row, then Reset (a lockstep collective).
  void TpWarmup();

  // Test hook: records every MoE visit's router ids from the next feed on (TP=1).
  void SetRouterTrace(bool on);
  const Inkling2RouterTrace& RouterTrace() const { return trace_; }

 private:
  Inkling2Model() = default;
  struct SummaryRequest {
    float inv_temperature = 1.0f;
    kernels::RowSummary* out = nullptr;
  };
  struct Chunk;  // the per-chunk fp32 state (inkling2_model.cpp)

  std::vector<float> RunChunk(const std::vector<int32_t>& ids, bool want_logits, int32_t* greedy_out,
                              const SummaryRequest* summary, SpanAccumulator* prof,
                              bool verify_conv_hist = false, std::vector<int32_t>* verify_greedy_out = nullptr,
                              float verify_inv_temp = 0.0f,
                              std::vector<kernels::RowSummary>* verify_summaries_out = nullptr);
  void RunVisit(const Inkling2Visit& v, Chunk& c, int64_t T, SpanAccumulator* prof, bool use_pos_dev = false,
                bool graph_long_ctx = true, bool capture_conv_hist = false);
  void Attention(const Inkling2Visit& v, const uint16_t* xq, const uint16_t* xk, const uint16_t* xv, uint16_t* out,
                 int64_t T, SpanAccumulator* prof, bool use_pos_dev = false, bool graph_long_ctx = true,
                 bool capture_conv_hist = false);
  void Moe(int64_t layer, const uint16_t* x, uint16_t* out, int64_t T, int visit, SpanAccumulator* prof);
  void MsgGate(const Inkling2MsgGateWeights& g, const float* x, const float* m, float* out, int64_t T);
  void Lin(const QuantLinear& w, const uint16_t* x, uint16_t* y, int64_t M);
  float* ConvState(int slot, int which);  // which: 0 k, 1 v, 2 post-attention, 3 post-MLP
  float* ConvHist(int slot, int which);
  int32_t LmHeadArgmaxFromBf16(const uint16_t* h_bf16);
  int32_t DsparkLogitsArgmax(const uint16_t* base_bf16, const uint16_t* delta_bf16, const uint16_t* bias_bf16);
  void Head(const float* h_row, const std::vector<const float*>& blocks, SpanAccumulator* prof);
  int32_t MergeGreedyPair();
  int32_t MergeGreedyPairAt(const int32_t* dev_pair);
  void MergeSummary(kernels::RowSummary* out);
  void GatherVocabRow(std::vector<float>* full);
  void GatherVerifyVocabRow(int64_t row, std::vector<float>* full);
  void FetchRowSummary(float inv_temperature, kernels::RowSummary* out);
  void RequireNotTp(const char* what) const;

  Inkling2ModelOptions opts_;
  Inkling2Container container_;
  std::vector<Inkling2Visit> visits_;
  core::Stream stream_;
  core::Arena arena_;
  core::TpComm* comm_ = nullptr;
  int64_t max_ctx_ = 0;
  int max_chunk_ = 256;
  int64_t pos_ = 0;
  bool started_ = false;
  int64_t sampled_fallback_rows_ = 0;
  int64_t kv_bytes_ = 0;
  int64_t ckpt_pos_ = -1;

  // KV: one paged bf16 cache per slot; the two block tables (full: identity, sliding: the ring).
  int kv_block_ = 256;
  int64_t full_blocks_ = 0, ring_blocks_ = 0;
  std::vector<core::DeviceBuffer<uint16_t>> kv_;       // [slot]
  std::vector<core::DeviceBuffer<uint16_t>> kv_ckpt_;  // [slot] sliding slots only (prompt_checkpoint)
  core::DeviceBuffer<int32_t> table_full_, table_ring_;
  // Conv states: [slot][4][3, C] fp32 in one buffer.
  std::vector<int64_t> conv_off_;  // [slot * 4 + which]
  std::vector<int64_t> conv_hist_off_;
  std::vector<int64_t> conv_ch_;
  core::DeviceBuffer<float> conv_, conv_ckpt_, conv_verify_bak_, conv_verify_hist_;

  // Optional Inkling2 DSpark / MTP drafter (mtp.safetensors).
  std::unique_ptr<Inkling2DsparkDraft> dspark_;
  std::array<core::DeviceBuffer<uint16_t>, 4> dspark_taps_;  // [4][max_chunk, H] bf16
  core::DeviceBuffer<float> verify_logits_dev_;              // [8, HeadRowsLocal]
  core::DeviceBuffer<int32_t> verify_argmax_dev_, verify_argmax_pair_dev_;
  core::DeviceBuffer<int32_t> verify_summary_ids_dev_;       // [8 * R4DX_TOPK_LSE_K]
  core::DeviceBuffer<float> verify_summary_vals_dev_;        // [8 * R4DX_TOPK_LSE_K]
  core::DeviceBuffer<float> verify_summary_lse_dev_;         // [8]

  // The fp32 state pool (h, blocks, h_pre, the previous pass's states) and the message buffer.
  std::vector<core::DeviceBuffer<float>> pool_;
  core::DeviceBuffer<float> msg_;  // [4, max_chunk, H]

  // MoE host readbacks.
  core::PinnedBuffer<int32_t> moe_ids_host_, moe_off_host_;
  Inkling2RouterTrace trace_;

  struct GraphExecDeleter {
    void operator()(void* p) const noexcept;
  };
  using GraphExecPtr = std::unique_ptr<void, GraphExecDeleter>;

  core::PinnedBuffer<int32_t> ids_host_, pos_host_;
  core::DeviceBuffer<int32_t> ids_dev_, pos_dev_;
  GraphExecPtr decode_graph_short_, decode_graph_long_;
  std::array<GraphExecPtr, 9> verify_graphs_;  // indexed by verify window T in [2..8]
  int64_t dspark_adapt_k_ = 3;
  bool decode_graph_eligible_ = false;
  bool decode_graph_disabled_ = false;
  int64_t decode_steps_seen_ = 0;
  core::DeviceBuffer<float> logits_dev_;  // [HeadRowsLocal]
  core::DeviceBuffer<int32_t> argmax_dev_, argmax_pair_dev_, summary_ids_dev_;
  core::DeviceBuffer<float> summary_vals_dev_, summary_lse_dev_;
  core::DeviceBuffer<uint8_t> topk_lse_ws_;
  int64_t vocab_local_valid_ = 0;  // this rank's logit columns below unpadded_vocab_size
  std::vector<float> gather_host_, sampled_row_scratch_;
};

}  // namespace r4dx::model
