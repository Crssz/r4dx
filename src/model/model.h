// r4dx::model::Model -- the full 64-layer forward pass (docs/architecture.md's "Forward pass, one
// text token step"), assembled from this directory's per-block primitives (GdnLayer, Mlp,
// FinalLmHead, EmbedTokens) plus src/model/attention/'s AttentionLayer, with a paged fp8 KV cache
// per full-attention layer and a GdnStateManager per GDN layer for a single sequence.
//
// SCOPE: single sequence (num_seqs==1), matching every per-block primitive this assembles
// (GdnLayer, AttentionLayer are both documented single-sequence-per-call). A multi-sequence
// serving layer (src/server's future job) batches N of these, or extends the per-block primitives
// to their own N/cu/slot arguments -- not done here.
//
// Prefill chunks the prompt into <=64-token pieces (docs/architecture.md "Interim chunked
// prefill"): each chunk runs every layer once, carrying GDN recurrent/conv state and the KV
// cache's running position across chunks. Decode processes one token (or a small window, if a
// future speculative-decoding caller wants it -- RunChunk's T is not hardcoded to 1) at a time.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "container.h"
#include "gdn_state.h"
#include "model_config.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/model/attention/paged_kv_cache.hpp"

namespace r4dx::model {

struct ModelOptions {
  std::string container_path;
  Layout layout = Layout::kBf16;       // body layout: GDN in_proj/out_proj, MLP gate_up/down,
                                        // lm_head (attn.qg/o are always bf16 -- see container.cpp)
  int64_t max_ctx = 131072;            // KV cache capacity per full-attention layer, in tokens
  int64_t layer_limit = -1;            // -1 = load Config().num_hidden_layers; >=0 for test
                                        // containers with fewer layers on disk
};

class Model {
 public:
  static Model Load(const ModelOptions& opts);

  Model(Model&&) = default;
  Model& operator=(Model&&) = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  const ModelConfig& Config() const { return container_.Config(); }
  const Container& GetContainer() const { return container_; }

  // Number of tokens already committed into the KV/GDN state (0 before the first Prefill call).
  int64_t PositionCount() const { return pos_; }

  // Feeds `token_ids` through the model in <=64-token chunks, continuing from whatever state
  // (GDN recurrent/conv, KV cache) this Model already holds. Despite the name this is not
  // restricted to a single call: it may be called once per prefix EXTENSION -- e.g. --chat's
  // per-turn re-prefill of only the new tail tokens (src/cli/main.cpp) -- as long as every call's
  // `token_ids` is the token sequence that immediately follows everything already fed via a prior
  // Prefill/DecodeStep call on this same Model (chunked-prefill's has_init/start_pos bookkeeping
  // handles the boundary correctly either way). Returns fp32 logits[vocab] for the token that
  // follows the prompt's last token.
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids);

  // Feeds one more token through the model, continuing state from the previous Prefill/DecodeStep
  // call. Returns fp32 logits[vocab] for the token that follows `token_id`.
  std::vector<float> DecodeStep(int32_t token_id);

 private:
  Model() = default;

  // Runs every layer once over `token_ids` (<=64 of them), advancing `pos_` by token_ids.size().
  // `is_prefill_path` selects GDN's chunked-scan kernels (true) vs its sequential recurrent-update
  // kernels (false, used for every DecodeStep and required whenever T does not represent a fresh
  // contiguous prefill chunk). `want_logits`: when false, skips final_norm+lm_head+the logits
  // readback entirely and returns an empty vector -- for Prefill()'s non-final chunks, whose
  // logits are never read (see model.cpp).
  std::vector<float> RunChunk(const std::vector<int32_t>& token_ids, bool is_prefill_path,
                               bool want_logits);

  Container container_;
  core::Stream stream_;
  core::Arena arena_;
  core::PinnedBuffer<uint16_t> embed_staging_;
  core::DeviceBuffer<uint16_t> buf_a_, buf_b_;  // ping-pong [max_chunk_, hidden] bf16 activations
  core::DeviceBuffer<float> logits_dev_;        // [vocab] fp32, one row at a time

  // Persistent (not arena-allocated) scratch every full-attention layer's AttentionLayer::Forward
  // shares within one RunChunk call: `positions[t] = pos_ + t` doubles as both the RoPE position
  // ids and the KV slot_mapping (contiguous block table => slot==pos), and `seqused_k[0] = pos_ +
  // T`. Neither depends on the layer, only on (pos_, T), so RunChunk uploads them once per chunk
  // and every attention layer in that chunk reuses the same device pointers -- see
  // AttentionLayer::Forward's doc comment for why these must NOT be arena-allocated.
  core::DeviceBuffer<int32_t> attn_positions_;  // [max_chunk_]
  core::DeviceBuffer<int32_t> attn_seqused_k_;  // [1]

  std::vector<std::optional<GdnStateManager>> gdn_states_;             // one per GDN layer
  std::vector<std::optional<attention::PagedKvCache>> kv_caches_;      // one per attn layer
  GdnControlCache gdn_control_;  // shared by every GDN layer -- see gdn_state.h

  int64_t max_chunk_ = 64;
  int64_t pos_ = 0;        // tokens already committed to KV/GDN state
  bool started_ = false;   // false only before the very first RunChunk call (GDN has_init gate)
};

}  // namespace r4dx::model
