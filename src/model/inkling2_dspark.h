// r4dx::model::Inkling2DsparkDraft -- Inkling-2-Mini DSpark / MTP speculative drafter (mtp.safetensors).
// 3-layer dense RoPE transformer over DFlare-mixed target feature taps [5, 23, 41, 48] + recurrent (rnn_hidden)
// Markov head (block_size = 8 -> 7 speculative tokens per round). Replicated per rank under TP=2 (no all-reduce
// inside the draft layers; only the shared target lm_head argmax merges across ranks via the caller's callback).
#pragma once

#include <array>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "quant_linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::model {

struct Inkling2DsparkConfig {
  int64_t hidden_size = 2304;
  int64_t num_layers = 3;
  int64_t num_attention_heads = 24;
  int64_t num_key_value_heads = 8;
  int64_t head_dim = 128;
  int64_t intermediate_size = 9216;
  int64_t block_size = 8;
  int64_t markov_rank = 256;
  int64_t sliding_window = 1024;
  int64_t vocab_size = 201024;
  float rms_norm_eps = 1e-6f;
  float rope_theta = 1000000.0f;
  std::array<bool, 3> is_local = {true, false, true};
  std::array<int, 4> target_layer_ids = {5, 23, 41, 48};
};

// Callback: given one normalized hidden state row [hidden_size] bf16 on device, runs the target's shared lm_head +
// softcap + argmax (with TP merge if TP=2) and returns the winning token id.
using DsparkLmHeadArgmaxFn = std::function<int32_t(const uint16_t* h_bf16)>;

// Callback: batched target lm_head projection h_bf16 [M, hidden_size] -> out_bf16 [M, HeadRowsLocal].
using DsparkLmHeadBatchFn = std::function<void(const uint16_t* h_bf16, int64_t M, uint16_t* out_bf16)>;

// Callback: given [HeadRowsLocal] bf16 base, optional delta, optional bias, computes softcap(base + delta + bias)
// into target's logits_dev_ + argmax (with TP merge if TP=2) and returns the winning token id.
using DsparkLogitsArgmaxFn =
    std::function<int32_t(const uint16_t* base_bf16, const uint16_t* delta_bf16, const uint16_t* bias_bf16)>;

// Callback: given anchor token id, writes target's embed_norm(embed_tokens(id)) [hidden_size] bf16 to out_bf16.
using DsparkAnchorEmbedFn = std::function<void(int32_t token_id, uint16_t* out_bf16)>;

class Inkling2DsparkDraft {
 public:
  static Inkling2DsparkDraft Load(const std::string& safetensors_path, int64_t max_ctx);

  Inkling2DsparkDraft() = default;
  ~Inkling2DsparkDraft();
  Inkling2DsparkDraft(Inkling2DsparkDraft&& other) noexcept;
  Inkling2DsparkDraft& operator=(Inkling2DsparkDraft&& other) noexcept;
  Inkling2DsparkDraft(const Inkling2DsparkDraft&) = delete;
  Inkling2DsparkDraft& operator=(const Inkling2DsparkDraft&) = delete;

  const Inkling2DsparkConfig& Config() const { return cfg_; }
  int64_t InjectedCount() const { return n_injected_; }
  int64_t BlockSize() const { return cfg_.block_size; }
  int64_t NumSpecTokens() const { return cfg_.block_size - 1; }

  void Reset();
  void SaveCheckpoint(core::Stream& stream);
  void RestoreCheckpoint(core::Stream& stream);

  // Precompute fold_w = lm_head @ hidden_out.weight ([vl, R] bf16) and fold_b = lm_head @ hidden_out.bias ([vl] bf16)
  // so DraftGreedy runs lm_head once (batched M = max_steps) and only [vl, 256] per Markov step.
  void InitFoldedLmHead(int64_t vl, core::Stream& stream, core::Arena& arena,
                        const DsparkLmHeadBatchFn& lm_head_batch);

  // Inject `n` newly committed target tap rows (4 separate contiguous [n, hidden_size] bf16 buffers) at positions
  // pos0..pos0+n-1, and update f_rnn_ from the last committed row (row n - 1).
  void InjectTaps(const uint16_t* tap0, const uint16_t* tap1, const uint16_t* tap2, const uint16_t* tap3,
                  int64_t n, int64_t pos0, core::Stream& stream, core::Arena& arena);

  // Draft `max_steps` (in 1..block_size-1; 0 = block_size - 1) speculative tokens starting after `anchor_token`
  // (at position `anchor_pos`). `n_injected_` must equal `anchor_pos`.
  std::vector<int32_t> DraftGreedy(int32_t anchor_token, int64_t anchor_pos, int64_t max_steps, core::Stream& stream,
                                   core::Arena& arena, const DsparkAnchorEmbedFn& embed_anchor,
                                   const DsparkLmHeadArgmaxFn& lm_head_argmax,
                                   const DsparkLmHeadBatchFn& lm_head_batch = {},
                                   const DsparkLogitsArgmaxFn& logits_argmax = {});

 private:
  struct Layer {
    core::DeviceBuffer<uint16_t> dflare_norm;
    core::DeviceBuffer<uint16_t> input_layernorm;
    core::DeviceBuffer<uint16_t> post_attention_layernorm;
    core::DeviceBuffer<uint16_t> q_norm, k_norm, k_norm_ctx;
    QuantLinear q_proj, k_proj, v_proj, o_proj;
    QuantLinear k_proj_ctx, v_proj_ctx;
    QuantLinear gate_up_proj, down_proj;
    core::DeviceBuffer<uint16_t> k_store, v_store;  // [slots, kv_heads, head_dim]
    core::DeviceBuffer<uint16_t> k_store_ckpt, v_store_ckpt;
    int slots = 0;
    int window = 0;
  };

  Inkling2DsparkConfig cfg_;
  std::array<std::array<float, 4>, 4> fuse_alpha_{};
  core::DeviceBuffer<uint16_t> mask_embed_;
  core::DeviceBuffer<uint16_t> norm_;
  std::vector<Layer> layers_;

  // Recurrent Markov head
  core::DeviceBuffer<uint16_t> markov_dflare_norm_;
  std::vector<uint16_t> markov_w1_host_;          // [vocab_size, markov_rank] bf16 on host (fallback)
  core::DeviceBuffer<uint16_t> markov_w1_dev_;    // [vocab_size, markov_rank] bf16 in VRAM (default, 103 MB)
  QuantLinear rnn_proj_;                  // [3*R, 2*R + 2*H] bf16
  core::DeviceBuffer<uint16_t> rnn_bias_bf16_;  // [3*R] bf16
  core::DeviceBuffer<uint16_t> hidden_out_w_;   // [H, R] bf16 (fallback if lm_head not folded)
  std::vector<uint16_t> hidden_out_w_t_host_;   // [R, H] bf16 (freed after InitFoldedLmHead)
  core::DeviceBuffer<uint16_t> hidden_out_b_;   // [H] bf16

  // Folded lm_head @ hidden_out (populated by InitFoldedLmHead).
  bool has_folded_lm_head_ = false;
  bool use_mxfp8_ = true;
  int64_t fold_vl_ = 0;
  QuantLinear fold_w_;                    // [vl, R] mxfp8 (or bf16)
  core::DeviceBuffer<uint16_t> fold_b_;   // [vl] bf16

  // Persistent draft state + HIP graph for the 3-layer DSpark transformer block
  core::DeviceBuffer<uint16_t> f_rnn_;          // [H] bf16 (last committed token's markov context)
  core::DeviceBuffer<uint16_t> f_rnn_ckpt_;
  core::PinnedBuffer<uint16_t> prev_emb_host_;  // [R] bf16
  core::DeviceBuffer<int64_t> draft_pos_dev_;   // [1] anchor_pos (= n_injected_) for draft_graph_
  core::DeviceBuffer<uint16_t> draft_resid_;    // [B, H] persistent input for draft_graph_
  core::DeviceBuffer<uint16_t> draft_head_h_;   // [B, H] persistent output for draft_graph_
  void* draft_graph_ = nullptr;                 // hipGraph_t
  void* draft_graph_exec_ = nullptr;            // hipGraphExec_t
  bool draft_graph_attempted_ = false;
  int64_t n_injected_ = 0;
  int64_t ckpt_injected_ = -1;

  void ReleaseDraftGraph();
};

}  // namespace r4dx::model
