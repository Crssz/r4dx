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
#include <utility>
#include <vector>

#include <memory>

#include "container.h"
#include "gdn_state.h"
#include "model_config.h"
#include "mtp_head.h"
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
  // MTP self-speculation (docs/mtp.md): draft this many tokens per step via the container's mtp.*
  // head before verifying them against the real model. 0 (default) disables MTP entirely -- the
  // GDN state layout and every decode call degenerate EXACTLY to this Model's pre-MTP behavior
  // (see gdn_state.h's file comment). >0 requires the container to have been converted with
  // --mtp on (Container::HasMtp()); Load() throws otherwise.
  int64_t mtp_draft_k = 0;
  // MTP head layout (docs/mtp.md "MTP head layout"): the layout for ONLY the MTP head's four
  // quantized linears (mtp.attn.qg/o, mtp.mlp.gate_up/down), independent of `layout`. nullopt
  // (default) tracks `layout` -- i.e. the head loads in whatever GEMM layout the body uses.
  // Measured (docs/mtp.md's "MTP head layout" table, K=1..4 x w4a8/w4a16/mxfp4): the
  // layout-matched head is FASTER than a bf16 head in 23/24 configurations (smaller GEMMs, no
  // extra VRAM/load time) and acceptance is a wash -- often slightly HIGHER, never meaningfully
  // lower, because a bf16 head sitting on top of h_seed (already carrying the body's own
  // quantization noise for w4a8/mxfp4) gains nothing from its own extra precision. Set explicitly
  // to Layout::kBf16 (CLI: `--mtp-head-layout bf16`) to force the exact-arithmetic head instead
  // (~0.5 GB extra VRAM). Ignored (no effect, no extra VRAM) when the container has no mtp.*
  // weights.
  std::optional<Layout> mtp_head_layout = std::nullopt;
  // Device-resident draft loop (docs/mtp.md "device-resident draft loop", docs/r9700.md P3):
  // mirror text.embed_tokens into VRAM (~2.54 GB bf16) so the decode/draft path can gather
  // embedding rows on-device instead of a host memcpy + H2D per step -- see Container::Load's own
  // comment for the free-VRAM fallback. Default true; set false to force host-only gather (e.g. a
  // VRAM-constrained run that would rather keep the margin for KV cache).
  bool embed_device_resident = true;
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

  // Like DecodeStep, but for a --temperature 0 (greedy) caller: computes the same next-token
  // logits on-device, then argmaxes THERE too (r4dx_argmax_f32) and reads back only the winning
  // vocab index (4 bytes) instead of the full vocab*4-byte logits vector -- the host-overhead
  // pass's (2026-09-19) only remaining per-token D2H copy that scaled with vocab rather than O(1).
  // Model has no notion of SampleParams/top-k/top-p; a caller that wants anything other than pure
  // greedy sampling must still call DecodeStep and sample over the full logits on the host.
  int32_t DecodeStepGreedy(int32_t token_id);

  // One named GPU timing span's accumulated result for one profiled decode step (tools/profile
  // pass, 2026-09-19): `count` calls of this op family summed to `ms` milliseconds of hipEvent-
  // measured device time (GDN kernels are one entry per GDN layer's whole Forward(), attention
  // likewise, one entry per distinct GEMM (N,K) shape, etc. -- see DecodeStepProfiled's own
  // comment for exactly what is and is not broken out).
  struct ProfileEntry {
    std::string name;
    double ms = 0.0;
    int count = 0;
  };
  // NOTE on what NOT to do with these numbers: `entries[*].ms` (hipEvent-measured) and
  // `finish_wait_ms` (host-chrono-measured) are NOT additive -- every kernel `entries` accounts
  // for was already enqueued (asynchronously) before `finish_wait_ms`'s clock starts, so
  // `finish_wait_ms` is host time spent BLOCKED waiting for that SAME already-queued GPU work to
  // finish (plus the final 4-byte D2H), not extra work on top of it. `entries[*].ms` summed
  // (`gpu_sum_ms`) is the right number for "which kernel family dominates GPU time" (this
  // pass's own top-3-cost ask); `finish_wait_ms` cross-validates it from the host's point of view
  // (the two should land close to each other) and is also the honest place to see the "only sync
  // per token" cost the host-overhead pass (item 5) was trying to minimize. `wall_ms` ~=
  // `host_enqueue_ms` (the host-only time spent issuing every async kernel launch + CPU-side work
  // like the embedding gather, measured BEFORE the Finish()/sync call below) + `finish_wait_ms`.
  struct StepProfile {
    std::vector<ProfileEntry> entries;  // per op-family, hipEvent-measured GPU device time, in
                                         // call order (first occurrence) -- does NOT include the
                                         // final stream-sync+readback (see finish_wait_ms)
    double gpu_sum_ms = 0.0;            // sum of entries[*].ms
    double host_enqueue_ms = 0.0;       // host time to issue every async launch (wall_t0 up to
                                         // the Finish()/sync call below), NOT GPU-blocked
    double finish_wait_ms = 0.0;        // host-chrono time for Finish()'s hipEventSynchronize +
                                         // the final 4-byte argmax D2H -- the actual "only sync
                                         // per token" cost, see NOTE above
    double wall_ms = 0.0;               // host wall-clock for the whole DecodeStepProfiled call
                                         // (~= host_enqueue_ms + finish_wait_ms)
  };

  // Runs exactly one decode step (T=1) like DecodeStep, but wraps each kernel-family call in a
  // hipEvent pair so the returned StepProfile breaks down where the step's time actually went --
  // tools/profile's own request ("per-op profile ... per kernel family per decode step"). NOT on
  // the hot path (DecodeStep/DecodeStepGreedy never call this): hipEventCreate/Record/Synchronize
  // per op adds real host-side overhead of its own, so this is diagnostic-only, invoked at most
  // once per r4dx-cli process via --profile (src/cli/main.cpp).
  StepProfile DecodeStepProfiled(int32_t token_id);

  // True iff this Model was Load()'d with mtp_draft_k > 0 (and the container had mtp.* weights).
  bool MtpEnabled() const { return static_cast<bool>(mtp_); }

  // MTP self-speculative decode (docs/mtp.md): drafts up to `k` tokens via the container's mtp.*
  // head (chained from this Model's own last-produced hidden state -- see mtp_seed_hidden_'s
  // comment below), verifies them against the real model in ONE q_len<=k+1 forward pass, and
  // commits however many of the prefix the real model's own greedy argmax confirms. `token_id`:
  // the last already-accepted real token (same convention as DecodeStep -- not yet reflected in
  // this Model's KV/GDN state). Returns 1..k+1 new committed tokens (the accepted drafts, plus
  // either a correction at the first mismatch or one "bonus" token if every draft was accepted).
  // Requires MtpEnabled(); k must be in [0, mtp_draft_k] (the value Load() was given -- this
  // Model's GDN state and MTP's own KV cache are sized for exactly that many, per gdn_state.h's
  // and mtp_head.h's file comments; k==0 degenerates to a single DecodeStepGreedy-equivalent
  // result, still going through the same verify path so num_accepted stays correctly threaded for
  // the NEXT call -- callers that never draft (k==0 for the whole run) should call plain
  // DecodeStepGreedy instead, which is cheaper and untouched by any of this).
  std::vector<int32_t> DecodeStepMtpGreedy(int32_t token_id, int64_t k);

  // Runs the real model over `candidates` (1..mtp_draft_k+1 tokens, is_prefill_path=false, the
  // same speculative-verify decode path DecodeStepMtpGreedy uses) and returns the greedy argmax
  // token predicted at EVERY position, WITHOUT committing any state: pos_ is not advanced and
  // mtp_num_accepted_dev_/mtp_seed_hidden_ are not updated (DecodeStepMtpGreedy does both once it
  // knows how many candidates were accepted -- see that method). Exposed publicly (rather than
  // kept as an internal helper) for tests/model/test_mtp.cpp, which needs to check this call's raw
  // per-position logits against sequential DecodeStep's own logits -- not just DecodeStepMtpGreedy's
  // argmax-only return value. `logits_out`, if non-null, is resized to
  // candidates.size()*Config().vocab_size and filled with this call's flat [T,vocab] fp32 logits
  // (an extra D2H a normal (non-test) caller does not need -- nullptr skips it). Requires
  // MtpEnabled() (the GDN state window bank and mtp_logits_dev_/mtp_argmax_dev_ scratch this needs
  // are only sized when mtp_draft_k>0).
  std::vector<int32_t> VerifyWindow(const std::vector<int32_t>& candidates,
                                     std::vector<float>* logits_out = nullptr);

 private:
  Model() = default;

  // Runs every layer once over `token_ids` (<=64 of them), advancing `pos_` by token_ids.size().
  // `is_prefill_path` selects GDN's chunked-scan kernels (true) vs its sequential recurrent-update
  // kernels (false, used for every DecodeStep and required whenever T does not represent a fresh
  // contiguous prefill chunk). `want_logits`: when false, skips final_norm+lm_head+the logits
  // readback entirely and returns an empty vector -- for Prefill()'s non-final chunks, whose
  // logits are never read (see model.cpp). `greedy_token_out`: when non-null (and want_logits),
  // skips the vocab-sized logits D2H entirely and instead argmaxes logits_dev_ ON DEVICE
  // (r4dx_argmax_f32), reading back only the single resulting index into `*greedy_token_out` --
  // the returned vector is empty in that mode (DecodeStepGreedy's caller wants the token id, not
  // the logits).
  std::vector<float> RunChunk(const std::vector<int32_t>& token_ids, bool is_prefill_path,
                               bool want_logits, int32_t* greedy_token_out = nullptr);

  Container container_;
  core::Stream stream_;
  core::Arena arena_;
  core::PinnedBuffer<uint16_t> embed_staging_;
  // Device-resident gather path (docs/mtp.md "device-resident draft loop"): a persistent [max_chunk_]
  // int32 id staging pair (pinned host + device), reused every RunChunk call instead of allocating
  // fresh -- same "persistent, not arena" reasoning as attn_positions_/attn_seqused_k_ below. Used
  // only when container_.EmbedTokensDeviceResident(); embed_staging_/EmbedTokens (host path) above
  // stays available unconditionally as the fallback.
  core::PinnedBuffer<int32_t> embed_ids_host_;
  core::DeviceBuffer<int32_t> embed_ids_dev_;
  core::DeviceBuffer<uint16_t> buf_a_, buf_b_;  // ping-pong [max_chunk_, hidden] bf16 activations
  core::DeviceBuffer<float> logits_dev_;        // [vocab] fp32, one row at a time
  core::DeviceBuffer<int32_t> argmax_dev_;      // [1] -- DecodeStepGreedy's on-device argmax result

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

  // ---- MTP self-speculation (docs/mtp.md), all empty/unused when mtp_draft_k==0 -----------------
  std::optional<MtpHead> mtp_;
  int64_t mtp_draft_k_ = 0;
  // The main model's own pre-final-norm hidden state at the row that produced the CURRENT
  // "last-accepted-token"'s own logits -- exactly what MtpHead::Draft's h_seed needs (see that
  // class's file comment), AND exactly the "boundary" h_i MtpHead::PrimeKv needs to prime the ONE
  // MTP position left dangling by the previous RunChunk/DecodeStepMtpGreedy call (see RunChunk's
  // own comment in model.cpp). Updated by EVERY RunChunk call (not just want_logits ones -- see
  // that method) and by every DecodeStepMtpGreedy() call's own verify pass (at the row that
  // produced the corrected/bonus token), so it is always valid by the time the NEXT call needs it.
  core::DeviceBuffer<uint16_t> mtp_seed_hidden_;
  // False only before the very first RunChunk/DecodeStepMtpGreedy call ever fills mtp_seed_hidden_
  // (there is no h_{-1} to prime a boundary position from yet).
  bool mtp_seed_valid_ = false;
  // GDN's cross-call acceptance-count thread (gdn_state.h's file comment): nullptr before the
  // first verify round of a generation (fresh seed from window index 0), then holds whatever the
  // PREVIOUS verify round's own num_committed was, forever after, for as long as MTP keeps
  // drafting.
  core::DeviceBuffer<int32_t> mtp_num_accepted_dev_;
  bool mtp_num_accepted_valid_ = false;
  // Scratch for VerifyWindow (below): logits for up to (mtp_draft_k_+1) candidate positions at
  // once, plus one argmax result per position.
  core::DeviceBuffer<float> mtp_logits_dev_;      // [(draft_k+1) * vocab]
  core::DeviceBuffer<int32_t> mtp_argmax_dev_;    // [draft_k+1]
  // Non-owning: whichever of buf_a_/buf_b_ the most recent VerifyWindow() call left its final
  // per-position hidden states in (which one depends on how many full-attention layers ran, so it
  // is not knowable statically) -- valid only until the NEXT RunChunk/VerifyWindow call's own
  // EmbedTokens overwrites it. DecodeStepMtpGreedy reads row `num_accepted_drafts` out of this
  // right after VerifyWindow returns, before anything else touches buf_a_/buf_b_.
  uint16_t* mtp_last_hidden_ = nullptr;

};

}  // namespace r4dx::model
