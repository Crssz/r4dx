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
//
// TENSOR PARALLEL (docs/tp.md 8.1): on a TP rank this head is sharded exactly like a body attention
// layer + MLP (half the heads, half the MLP width; mtp.fc, the norms and the optional reduced-vocab
// draft head replicate), `cfg` is the rank-local config, and every draft step all-reduces its
// attention o_proj and MLP down outputs through `comm` (docs/tp.md 6.2 A2/A3). PrimeKv runs NO
// all-reduce: it uses a second attention layer built with a null communicator, because only its
// (column-parallel, rank-local) K/V write matters. The shared lm_head is vocab-split, so the
// FULL-vocab draft head can no longer chain its argmax on the device: each step argmaxes this rank's
// shard, and the ranks merge the (index, value) pairs on the host (docs/tp.md 6.2 H6) before the
// next step's embedding gather. The reduced-vocab head is replicated and keeps its device chain.
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
  // logits_rows: the widest lm_head a draft step will ever run, max(lm_head.N, draft_lm_head.N)
  // (docs/tp.md 4.4) -- the whole vocabulary at TP=1, where the reduced head is never wider than
  // it; under TP the shared head is this rank's vocab shard while the replicated reduced head is
  // not capped at half the vocabulary by the container format.
  // comm: the rank's tensor-parallel communicator (docs/tp.md 8.1), or nullptr at TP=1.
  MtpHead(const ModelConfig& cfg, const MtpWeights& w, int64_t max_draft, int64_t max_ctx,
          int64_t logits_rows, core::TpComm* comm = nullptr);

  // h_seed: device bf16 [hidden], the main model's pre-final-norm hidden state at the row that
  // produced seed_token's own logits (see Model::DecodeStepMtpGreedy). seed_token: the last
  // already-accepted real token. cfg/w: the CURRENT (live) container config/mtp weights -- see
  // this class's own comment for why these are per-call parameters, not stored members. embed_table
  // /vocab: Container::EmbedTokensHost()/Config().vocab_size (MTP shares the main model's embedding
  // table -- mtp_use_dedicated_embeddings=false in the real checkpoint's config.json); `vocab` is
  // the EMBEDDING table's row count only -- the global vocabulary under TP too, where the table is
  // replicated. lm_head: Container::LmHead() (shared, not MTP's own); the full-vocab draft head's
  // argmax runs over its own lm_head.N rows, which is this rank's vocab shard under TP (docs/tp.md
  // 8.1) and the whole vocabulary at TP=1. base_pos: the REAL sequence position this round's
  // first draft step writes into MTP's own KV cache -- always Model::pos_ - 1 at the moment
  // DecodeStepMtpGreedy is called (see PrimeKv's comment for the derivation: this is exactly the
  // position PrimeKv would prime next if no drafting happened at all). Returns exactly `k`
  // greedily-drafted token ids (k==0 returns empty without touching any state).
  //
  // embed_table_dev (device-resident draft loop, docs/mtp.md): a DEVICE [vocab,hidden] bf16
  // mirror of `embed_table` (Container::EmbedTokensDevice()), or nullptr to fall back to the
  // original host-gather path. When non-null, every draft step after the first feeds THIS step's
  // own on-device argmax result (r4dx_argmax_f32's out_idx) straight into the next step's
  // embedding gather (r4dx_embedding_gather_bf16) with NO host sync/D2H/H2D in between -- only the
  // seed token (step 0's input, already known on the host as `seed_token`) needs one small H2D
  // upload, done ONCE before the loop starts, not once per step. The whole k-token draft result is
  // read back in exactly ONE stream.Synchronize() + D2H at the very end, replacing what was
  // previously k of each. positions_/seqused_k_ (this class's own per-step scratch) are likewise
  // preloaded for the whole window in one H2D upload each before the loop, not one CopyFromHost
  // (blocking) per step.
  // use_reduced_vocab (docs/r9700.md R9, "reduced-vocab draft head"): when true AND
  // w.HasDraftHead(), every draft step's own lm_head GEMM+argmax runs against w.draft_lm_head (N ==
  // w.draft_vocab_ids.size() rows instead of the full vocab) and the resulting subset-local index is
  // mapped back to a real vocab id via w.draft_vocab_ids (r4dx_gather_i32) BEFORE it is used for
  // anything else in this loop (the next step's embedding gather, the accumulated draft_ids_dev_/
  // drafts result) -- every other line of this method's own algorithm is unchanged, because from
  // that point on the value is, once again, a real vocab id like any other. This is why
  // VERIFICATION needs no changes at all: Draft() always returns real vocab ids, whichever head
  // produced them, and Model::VerifyWindow always checks them against the REAL model's own
  // full-vocab lm_head (container_.LmHead(), never w.draft_lm_head) -- a draft the reduced head
  // would only have made because its own subset omitted the correct token is simply an ordinary
  // rejected draft (lower acceptance, exactly like a wrong full-vocab-head guess), never a wrong
  // ACCEPTED one, so output quality cannot degrade by using this path. Silently falls back to the
  // full-vocab head (identical to use_reduced_vocab=false) when w.HasDraftHead() is false, so a
  // caller can request "reduced if available" unconditionally without checking the container itself.
  //
  // mrope_delta (vision milestone, docs/vision.md "Text-side splicing"): once an image has been
  // spliced into the conversation, a token's ROPE position is `sequence_index + mrope_delta`, not
  // `sequence_index`. Every position this method drafts at is past the prompt and therefore pure
  // text, so all three (t,h,w) streams still carry the same value -- but that value is no longer
  // base_pos+step. `positions_dev_` stays the sequence index (it is also the KV slot mapping for
  // this head's own cache); `rope3_dev_` carries the rope value. 0 (the default, and every
  // text-only conversation) leaves the pre-vision single-row path in place byte for byte.
  std::vector<int32_t> Draft(core::Stream& stream, core::Arena& arena, const ModelConfig& cfg,
                              const MtpWeights& w, const uint16_t* h_seed, int32_t seed_token,
                              const uint16_t* embed_table, const uint16_t* embed_table_dev,
                              int64_t vocab, const QuantLinear& lm_head, int64_t k,
                              int64_t base_pos, bool use_reduced_vocab = true,
                              bool mrope_active = false, int64_t mrope_delta = 0);

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
  //
  // host_staging_offset (elements, not bytes; default 0): where in this object's OWN pinned host
  // scratch (prime_positions_host_/prime_seqused_host_/prime_embed_host_) this call's data is
  // written before the async H2D upload below. This matters because HIP's stream-issue-order
  // guarantee serializes DEVICE-side reuse of a buffer, not the CPU's freedom to overwrite a
  // pinned HOST source buffer that a previous CopyFromHostAsync from it has not actually finished
  // reading yet (a real race found by review, 2026-09-20: RunChunk calls PrimeKv twice back to
  // back with no intervening sync -- the boundary n=1 call, then the within-chunk n=T-1 call --
  // and both calls' async H2D copies are stream-ordered behind ~64 layers of already-queued
  // kernel work, so by the time the FIRST call's DMA actually runs, the CPU has long since
  // overwritten the source buffers with the SECOND call's data). Giving each call in a back-to-back
  // pair a DISJOINT host_staging_offset (RunChunk: 0 for the boundary call, 1 for the within-chunk
  // call -- 1 + (T-1) == T <= kMaxPrime always, so this fits without enlarging any buffer) closes
  // the race without an extra stream.Synchronize() between them. The device-side destination
  // (prime_positions_dev_/prime_seqused_dev_/prime_embed_dev_) is NOT offset -- it is safely
  // reused at a fixed address across calls because HIP's stream-order guarantee DOES apply there
  // (each call's own kernels consume its device data before the next call's H2D can overwrite it).
  // Caller must ensure host_staging_offset + n <= kMaxPrime; PrimeKv throws otherwise.
  //
  // rope3_host (vision milestone, docs/vision.md): host int32[3, n] (compact, t row then h then w)
  // giving these n positions' 3-axis mrope rope positions. Unlike Draft's, THESE positions can
  // land inside an image run -- PrimeKv is called from RunChunk over the prompt itself -- so a
  // scalar delta is not enough and the caller (Model::RopePositionsHost) hands over the real rows.
  // nullptr (the default, and every text-only conversation) keeps the pre-vision path.
  void PrimeKv(core::Stream& stream, core::Arena& arena, const ModelConfig& cfg,
               const MtpWeights& w, int64_t base_pos, const uint16_t* h_rows,
               const std::vector<int32_t>& next_tokens, const uint16_t* embed_table, int64_t vocab,
               int64_t host_staging_offset = 0, const int32_t* rope3_host = nullptr);

#ifdef R4DX_TP_TESTING
  // Test hook (docs/tp.md 10.1, the H6 exactness check of tests/model/test_tp_emulation.cpp): while
  // capture is on, every step of a full-vocab draft that merges its argmax across ranks (TP only)
  // also gathers that step's full [vocab] logits row (a D2H of this rank's shard + one host
  // all-gather; every rank captures identically, so lockstep holds) and records the merged token.
  // DebugRows() is [steps][vocab] fp32 in global id order, DebugTokens() [steps]; both are cleared
  // at the start of every Draft() while capture is on, and stay empty on the reduced-vocab head and
  // at TP=1 (no merge happens there). DebugDrafts() is what the last Draft() returned while capture
  // was on, on every path -- so an empty DebugTokens() beside k DebugDrafts() shows a round that
  // drafted without a merge.
  void DebugSetCapture(bool on) { debug_capture_ = on; }
  const std::vector<int32_t>& DebugDrafts() const { return debug_drafts_; }
  const std::vector<float>& DebugRows() const { return debug_rows_; }
  const std::vector<int32_t>& DebugTokens() const { return debug_tokens_; }
#endif

 private:
  attention::AttentionLayer attn_layer_;
  // PrimeKv's layer (docs/tp.md 6.2): the same attention sublayer with a NULL communicator, so
  // priming -- whose output is discarded, only the K/V write matters -- runs no all-reduce under TP.
  // At TP=1 comm is null anyway and the two layers are identical.
  attention::AttentionLayer prime_attn_layer_;
  core::TpComm* comm_ = nullptr;  // non-owning; nullptr at TP=1
  attention::PagedKvCache kv_;
  core::PinnedBuffer<uint16_t> embed_staging_host_;  // [hidden] -- one token's embedding row (host
                                                      // gather fallback path only)
  core::DeviceBuffer<uint16_t> embed_staging_dev_;   // [hidden]
  core::DeviceBuffer<float> logits_dev_;             // [logits_rows] fp32 -- FinalLmHead's output
  core::DeviceBuffer<int32_t> argmax_dev_;           // [1] -- this step's own on-device argmax,
                                                      // ALWAYS a real vocab id by the time it is
                                                      // written (see subset_argmax_dev_ below for
                                                      // the reduced-vocab head's extra hop to get
                                                      // there); also feeds the next step's device
                                                      // gather directly when embed_table_dev !=
                                                      // nullptr
  // Reduced-vocab draft head scratch (docs/r9700.md R9): argmax_dev_ must always hold a REAL vocab
  // id (every other line of Draft()'s loop, and every caller of Draft(), assumes that) but the
  // reduced head's own r4dx_argmax_f32 call only ever sees w.draft_vocab_ids.size() logits, so its
  // result is a SUBSET-LOCAL index -- this one-element buffer holds that intermediate value for the
  // one extra r4dx_gather_i32 hop (subset index -> real id, via w.draft_vocab_ids) that writes the
  // real id into argmax_dev_ above. Unused (never allocated space needed beyond its fixed 1 element)
  // on the full-vocab path.
  core::DeviceBuffer<int32_t> subset_argmax_dev_;    // [1]
  // TP only (docs/tp.md 8.1): r4dx_argmax_val_f32's {int32 local index, float value} over this
  // rank's lm_head shard -- one tp::ArgmaxPair, merged across ranks on the host every draft step.
  // Empty at TP=1, which keeps its device-resident argmax chain.
  core::DeviceBuffer<int32_t> argmax_pair_dev_;      // [2]

  // Device-resident draft loop scratch (docs/mtp.md "device-resident draft loop"): positions_dev_/
  // seqused_dev_ hold the WHOLE k-step window's RoPE-pos/KV-slot and seqused_k values, preloaded in
  // one H2D upload each (Draft() before its loop) instead of one blocking CopyFromHost per step --
  // step `i` of the loop reads element `i` (pointer arithmetic, not a fresh buffer). Sized to
  // max_draft (the largest k this instance is ever asked to draft, ctor param). seed_token_dev_ is
  // the one small H2D upload Draft() still needs (step 0's input token, not yet known on-device);
  // draft_ids_dev_/draft_ids_host_ accumulate every step's own argmax result (D2D copy, no host
  // sync) for exactly ONE batched D2H readback at the end of Draft(), replacing what was previously
  // one stream.Synchronize()+D2H per drafted token.
  core::DeviceBuffer<int32_t> positions_dev_;   // [max_draft]
  core::PinnedBuffer<int32_t> positions_host_;  // [max_draft]
  core::DeviceBuffer<int32_t> seqused_dev_;     // [max_draft]
  core::PinnedBuffer<int32_t> seqused_host_;    // [max_draft]
  core::DeviceBuffer<int32_t> seed_token_dev_;  // [1]
  core::DeviceBuffer<int32_t> draft_ids_dev_;   // [max_draft]
  core::PinnedBuffer<int32_t> draft_ids_host_;  // [max_draft]
  // 3-axis rope positions for the same window (docs/vision.md), laid out STEP-MAJOR -- element
  // 3*step+axis, not axis*max_draft+step -- because each step ropes exactly one token and
  // r4dx_rope_partial_mrope3_bf16 reads its rows at stride `tokens`, which is 1 here. That makes
  // `rope3_dev_.data() + 3*step` a valid compact [3, 1] argument with no per-step upload.
  core::DeviceBuffer<int32_t> rope3_dev_;       // [3 * max_draft]
  core::PinnedBuffer<int32_t> rope3_host_;      // [3 * max_draft]

  // PrimeKv's own scratch, sized for up to a 64-row chunk (Model::max_chunk_) so both the n=1
  // boundary call and the n<=63 within-chunk call reuse the same buffers. Device-side reuse across
  // back-to-back calls IS safe purely from HIP's own stream-issue-order guarantee (each call's
  // kernels always consume its device data before the next call's H2D can overwrite it). The HOST
  // side is a different hazard class: a pinned host source buffer can be overwritten by the CPU as
  // soon as the issuing call returns, regardless of whether the async H2D copy that reads it has
  // actually run on the GPU yet -- stream order does not protect a host-side write the way it
  // protects a device-side one. PrimeKv's own host_staging_offset parameter is what actually closes
  // that gap (each back-to-back call in RunChunk uses a disjoint offset into these buffers); see
  // PrimeKv's own .h doc comment above and its .cpp comment for the incident this fixed.
  int64_t max_draft_ = 0;  // sizes positions_dev_/seqused_dev_/draft_ids_dev_ above

  static constexpr int64_t kMaxPrime = 64;
  core::PinnedBuffer<int32_t> prime_positions_host_;  // [kMaxPrime], sliced by host_staging_offset
  core::DeviceBuffer<int32_t> prime_positions_dev_;   // [kMaxPrime], dest always offset 0
  core::PinnedBuffer<int32_t> prime_seqused_host_;    // [kMaxPrime], sliced by host_staging_offset
  core::DeviceBuffer<int32_t> prime_seqused_dev_;     // [1], dest always offset 0
  core::PinnedBuffer<uint16_t> prime_embed_host_;     // [kMaxPrime * hidden]
  core::DeviceBuffer<uint16_t> prime_embed_dev_;      // [kMaxPrime * hidden]
  // PrimeKv's 3-axis rope rows, [3, n] compact at device offset 0; the host side is sliced by
  // 3*host_staging_offset for the same in-flight-H2D reason prime_positions_host_ is.
  core::PinnedBuffer<int32_t> prime_rope3_host_;      // [3 * kMaxPrime]
  core::DeviceBuffer<int32_t> prime_rope3_dev_;       // [3 * kMaxPrime]

#ifdef R4DX_TP_TESTING
  bool debug_capture_ = false;
  std::vector<float> debug_rows_;       // see DebugSetCapture
  std::vector<int32_t> debug_tokens_;
  std::vector<int32_t> debug_drafts_;
  std::vector<float> debug_shard_host_;
#endif
};

}  // namespace r4dx::model
