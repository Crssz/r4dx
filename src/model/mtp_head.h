// r4dx::model::MtpHead -- Qwen3.5's MTP self-speculation draft head (docs/mtp.md,
// docs/container-format.md "mtp.*"): chains up to `draft_k` greedy single-token predictions,
// seeded from the MAIN model's own last hidden state and last-accepted token -- Model's
// DecodeStepMtpGreedy (model.cpp) verifies the drafts afterward in one batched call against the
// real 64-layer model.
//
// Math (vllm/model_executor/models/qwen3_5_mtp.py's Qwen3_5MultiTokenPredictor.forward -- the
// actual working reference for this module; modeling_qwen3_5.py's own Qwen3_5MTPLayer does NOT
// exist in the installed transformers, see docs/mtp.md's "Design" section for the correction) --
// for the h_i/t_{i+1} this model was trained with (h_i = backbone hidden state at position i,
// encoding tokens 0..i; t_{i+1} = the token at position i+1, i.e. the ALREADY-KNOWN "next" token
// whose own logits the backbone itself already produced from h_i),
//
//   x = concat(pre_fc_norm_embedding(embed(t_{i+1})), pre_fc_norm_hidden(h_i))   [2*hidden]
//   x = fc(x)                                                                    [hidden]
//   h' = one_full_attention_decoder_layer(x)     -- mtp.layer's own weights + REAL KV cache
//   logits = lm_head(norm(h'))                   -- mtp.norm, then the SHARED main-model lm_head
//   draft = argmax(logits)                       -- predicts t_{i+2}
//
// and chaining draft_k>1 tokens reuses THIS layer's own (h', draft) pair as the next (h_i, t_{i+1})
// -- MTP predicting its own MTP-layer output, not the backbone's, which is exactly what makes this
// self-speculative rather than requiring a second backbone forward pass per draft token. NOTE the
// embedding/hidden concat HALVES: qwen3_5_mtp.py's forward computes `inputs_embeds =
// pre_fc_norm_embedding(inputs_embeds)` FIRST, THEN `hidden_states = pre_fc_norm_hidden(hidden_states)`,
// THEN `torch.cat([inputs_embeds, hidden_states], dim=-1)` -- the embedding occupies fc's FIRST
// `hidden` input columns, the hidden state the SECOND `hidden` columns. An earlier version of this
// file had these swapped (mtp.fc.weight is [5120,10240]=[out,2*hidden], so the wrong operand hit
// each half of the weight), which made every draft uncorrelated with the model (0-1.2% measured
// acceptance) -- see docs/mtp.md's "Design" section for the full incident writeup.
//
// MTP's own KV cache is a REAL per-sequence cache built in LOCKSTEP with the main model's own KV/
// GDN state, sized to the same max_ctx (NOT reset every Draft() call, and NOT a tiny scratch
// block): qwen3_5_mtp.py's Qwen3_5MultiTokenPredictor.forward takes real `positions` and runs
// mtp_layer with a real per-sequence KV cache the vLLM runner prefills over the whole prompt and
// extends by every accepted token, exactly like any other decoder layer -- it is NOT a
// per-round scratch cache seeded solely from h_seed's folded context (that was this file's
// original, WRONG design premise; see docs/mtp.md's "Design"/"Known gaps" history). Model::RunChunk
// (model.cpp) therefore calls MtpHead::PrimeKv on every prefill chunk AND every plain decode step
// to extend this cache by exactly the positions that just became known (see PrimeKv's own comment
// below for the exact (h_i, t_{i+1}) pairing and why priming and Draft()'s own step-0 forward
// necessarily compute the identical thing). Positions written speculatively by Draft() that a
// verify round later REJECTS are harmless and require no explicit rewind: this cache is contiguous
// (slot == position, PagedKvCache's own convention), so a rejected position is simply overwritten
// the next time that same position is legitimately written (by the next Draft()/PrimeKv call,
// which always starts from the correctly-committed frontier) before anything could ever read it --
// the same "self-correcting via position overwrite" trick the main model's own KV cache already
// relies on (docs/mtp.md's "State commit and rewind").
#pragma once

#include <cstdint>
#include <vector>

#include "container.h"
#include "mlp.h"
#include "model_config.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/model/attention/attention_layer.hpp"
#include "r4dx/model/attention/paged_kv_cache.hpp"

namespace r4dx::model {

class MtpHead {
 public:
  // cfg: the main model's ModelConfig (MTP's inner layer shares its geometry exactly -- see file
  // comment); used only to size this object's own buffers at construction time, NOT stored (a
  // reference member here would dangle: Model -- which owns the Container `cfg`/`w` refer into --
  // is itself returned by value from Model::Load() and moved into the caller's variable, which may
  // relocate Container's sub-objects; Draft() below takes fresh cfg/w reference PARAMETERS every
  // call instead, always the caller's own live, current-address container_, never a captured one).
  // w: this container's mtp.* weights, used the same construction-time-only way (sizes nothing
  // itself here, but kept as a parameter for signature symmetry with Draft(); NOT stored).
  // max_draft: the largest draft_k this instance will ever be asked to draft (CLI --mtp's value).
  // max_ctx: this Model's ModelOptions::max_ctx -- MTP's own KV cache is sized to the SAME capacity
  // as every backbone attention layer's PagedKvCache (see file comment: this is a real, growing,
  // per-sequence cache now, not a tiny scratch block).
  MtpHead(const ModelConfig& cfg, const MtpWeights& w, int64_t max_draft, int64_t max_ctx);

  // h_seed: device bf16 [hidden], the main model's pre-final-norm hidden state at the row that
  // produced seed_token's own logits (see Model::DecodeStepMtpGreedy). seed_token: the last
  // already-accepted real token. cfg/w: the CURRENT (live) container config/mtp weights -- see
  // this class's own comment for why these are per-call parameters, not stored members. embed_table
  // /vocab: Container::EmbedTokensHost()/Config().vocab_size (MTP shares the main model's embedding
  // table -- mtp_use_dedicated_embeddings=false in the real checkpoint's config.json). lm_head:
  // Container::LmHead() (shared, not MTP's own). base_pos: the REAL sequence position this round's
  // first draft step writes into MTP's own KV cache -- always Model::pos_ - 1 at the moment
  // DecodeStepMtpGreedy is called (see PrimeKv's comment for the derivation: this is exactly the
  // position PrimeKv would prime next if no drafting happened at all). Returns exactly `k`
  // greedily-drafted token ids (k==0 returns empty without touching any state).
  std::vector<int32_t> Draft(core::Stream& stream, core::Arena& arena, const ModelConfig& cfg,
                              const MtpWeights& w, const uint16_t* h_seed, int32_t seed_token,
                              const uint16_t* embed_table, int64_t vocab,
                              const QuantLinear& lm_head, int64_t k, int64_t base_pos);

  // Extends MTP's own KV cache (file comment) by `n` real positions [base_pos, base_pos+n), using
  // the REAL (h_i, t_{i+1}) pair at each position -- i.e. exactly Draft()'s own first-step
  // computation (concat -> fc -> one attention-decoder-layer forward), but WITHOUT the mlp/norm/
  // lm_head/argmax tail Draft() needs to actually produce a token: priming only cares about the
  // attention sublayer's K/V write, and h_i here always comes from the REAL backbone (never a
  // previous MTP step's own chained output), so there is nothing to chain across positions within
  // one PrimeKv call -- every position's x can be computed independently and batched through one
  // attn_layer_.Forward(..., T=n, ...) call exactly like the backbone's own chunked prefill.
  // h_rows: device bf16 [n, hidden], CONTIGUOUS, row t = h_i for position base_pos+t (the backbone's
  // own pre-final-norm hidden state at that position). next_tokens: host, size n, next_tokens[t] =
  // the real token at position base_pos+t+1 (t_{i+1} for row t). Caller (Model::RunChunk) always
  // calls this in at most two pieces per chunk: one n=1 call for the single "boundary" position
  // left dangling by the PREVIOUS call (h_i = that call's own last-row hidden, held in
  // Model::mtp_seed_hidden_), then one n=T-1 call for the within-chunk positions (h_i = this
  // chunk's own rows 0..T-2, t_{i+1} = this chunk's own token_ids[1..T-1]) -- position T-1 is left
  // dangling for the NEXT call, same as the boundary case (see model.cpp's RunChunk comment).
  void PrimeKv(core::Stream& stream, core::Arena& arena, const ModelConfig& cfg,
               const MtpWeights& w, int64_t base_pos, const uint16_t* h_rows,
               const std::vector<int32_t>& next_tokens, const uint16_t* embed_table, int64_t vocab);

 private:
  attention::AttentionLayer attn_layer_;
  attention::PagedKvCache kv_;
  core::PinnedBuffer<uint16_t> embed_staging_host_;  // [hidden] -- one token's embedding row
  core::DeviceBuffer<uint16_t> embed_staging_dev_;   // [hidden]
  core::DeviceBuffer<int32_t> positions_;            // [1] -- this draft step's RoPE pos / KV slot
  core::DeviceBuffer<int32_t> seqused_k_;            // [1]
  core::DeviceBuffer<float> logits_dev_;             // [vocab] fp32 -- FinalLmHead's output
  core::DeviceBuffer<int32_t> argmax_dev_;           // [1]

  // PrimeKv's own scratch, sized for up to a 64-row chunk (Model::max_chunk_) so both the n=1
  // boundary call and the n<=63 within-chunk call reuse the same buffers -- safe without any extra
  // synchronization because every upload below goes through CopyFromHostAsync on the SAME stream
  // as the kernels that read it, so HIP's own stream-issue-order guarantee (not a host-side wait)
  // is what serializes a later call's upload against an earlier call's still-in-flight reads (see
  // PrimeKv's .cpp comment).
  static constexpr int64_t kMaxPrime = 64;
  core::PinnedBuffer<int32_t> prime_positions_host_;  // [kMaxPrime]
  core::DeviceBuffer<int32_t> prime_positions_dev_;   // [kMaxPrime]
  core::PinnedBuffer<int32_t> prime_seqused_host_;    // [1]
  core::DeviceBuffer<int32_t> prime_seqused_dev_;     // [1]
  core::PinnedBuffer<uint16_t> prime_embed_host_;     // [kMaxPrime * hidden]
  core::DeviceBuffer<uint16_t> prime_embed_dev_;      // [kMaxPrime * hidden]
};

}  // namespace r4dx::model
