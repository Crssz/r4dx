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
//
// TENSOR PARALLEL (docs/tp.md, ModelOptions::tp): a Model can also be ONE RANK of a TP=2 group --
// it holds that rank's weight shard, runs on the rank-local config, all-reduces its three
// row-parallel outputs per layer through the rank's core::TpComm, and merges every vocab-split
// lm_head result across ranks itself, so each rank returns exactly what a full Model would (up to
// the documented numerics, docs/tp.md 1.2). Supported so far (P2a): Prefill, DecodeStep and
// DecodeStepGreedy. Sampled decode, every speculative-verify entry point (VerifyWindow,
// CommitVerifiedWindow, ReadVerifyLogitsRow, DecodeStepMtp*, DecodeStepDflash*) and the profiled
// methods throw core::TpUnsupportedError on a TP rank.
#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include <memory>

#include "container.h"
#include "dflash_draft.h"
#include "gdn_state.h"
#include "model_config.h"
#include "mtp_head.h"
#include "r4dx/core/arena.hpp"
// Sampled decode (docs/sampling.md): SampleParams/SampleCanonical/DrawUniform01 plus the RowSummary
// the device summary kernel fills and SampleFromSummary consumes. Header-only and HIP-free, so this
// costs every existing includer nothing but a couple of <cmath>/<random> declarations.
#include "r4dx/kernels/summary_sampler.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/model/attention/paged_kv_cache.hpp"
#include "vision_tower.h"  // src/vision: the device-side vision tower (docs/vision.md)

namespace r4dx::model {

// The DFlash2 drafter's two host codebooks, read once per tensor-parallel process and shared by
// every rank's drafter (docs/tp.md 8.2). Defined with DflashDraft::LoadHostCodebooks in P5; until
// then TpRankOptions::dflash_codebooks is always null.
struct DflashHostCodebooks;

// Tensor parallel (docs/tp.md 3.3): which rank of which world THIS Model is. Default-constructed ==
// TP=1, i.e. exactly the pre-TP Model. With world > 1, Model::Load loads rank `rank`'s shard of the
// container (Container::Load with ContainerLoadOptions::tp_world), runs every layer on the
// rank-local config with its row-parallel outputs all-reduced through `comm`, and merges every
// vocab-split lm_head result across ranks itself (docs/tp.md 7). Every rank's Model must be driven
// with the same sequence of calls and arguments (docs/tp.md 6.3.6).
struct TpRankOptions {
  int world = 1;
  int rank = 0;
  core::TpComm* comm = nullptr;  // non-owning; nullptr iff world == 1
  // TP: the process's one pinned host copy of text.embed_tokens (Container::LoadEmbedTokensHost).
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> shared_embed_host;
  // TP: the embedding device-mirror decision taken jointly for every rank (docs/tp.md 2.9 step 5):
  // 1 mirror, 0 host-only, -1 the Container's own free-VRAM heuristic (TP=1).
  int embed_device_resident_decided = -1;
  bool vision_weights_on_this_rank = true;  // TP: rank 0 only (docs/tp.md 8.3)
  std::shared_ptr<const DflashHostCodebooks> dflash_codebooks;  // TP: one host copy (P5)
};

struct ModelOptions {
  std::string container_path;
  Layout layout = Layout::kBf16;       // body layout: GDN in_proj/out_proj, MLP gate_up/down,
                                        // lm_head (attn.qg/o are always bf16 -- see container.cpp)
  // KV cache capacity per full-attention layer, in tokens. 262144 matches the checkpoint's own
  // config.json max_position_embeddings (docs/r9700.md R13, 2026-09-20 measurement) -- raised from
  // a stale 131072 self-imposed cap once real hardware showed the full 262144-token allocation
  // still leaves 21-24% of the card's VRAM free. See src/cli/cli_args.h's matching comment.
  int64_t max_ctx = 262144;
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
  // Reduced-vocab draft head (docs/r9700.md R9, docs/mtp.md "reduced-vocab draft head"): use the
  // container's OPTIONAL mtp.draft_head.* tensors (a smaller lm_head over a subset of the real
  // vocabulary) for MtpHead::Draft's own per-draft-token lm_head GEMM+argmax instead of the full
  // shared lm_head -- cuts the dominant cost of wide speculation (each drafted token otherwise pays
  // the FULL 248320-entry lm_head, docs/r9700.md §2.2's draft-side byte budget). Purely a drafting
  // SPEED lever: verification (Model::VerifyWindow) always runs the real model's full-vocab
  // lm_head regardless of this flag, so a draft the reduced head could not have produced (its own
  // subset omitted the correct token) is simply a rejected draft -- lower acceptance at high K,
  // never a wrong ACCEPTED token (see MtpHead::Draft's own doc comment for the full argument).
  // Default false (flipped 2026-09-20, Milestone 5 B2 item 8 + this review's fix -- was true):
  // M4 measured the reduced head SLOWER at its own best K than the full head at ITS own best K
  // (docs/mtp.md's "reduced-vocab draft head" K-sweep: 53.35 vs 65.02 tok/s) and this review's
  // matched-K=3 re-measurement confirms it at fixed K too (docs/mtp.md's "Flag default flipped"
  // note: 53.59/54.17 vs 68.20/68.64 tok/s, byte-identical generated text either way -- the flip
  // changes only speed, never correctness). Use the reduced head only by explicitly setting this true
  // (CLI/server: `--mtp-draft-head reduced`) once a higher-coverage calibration corpus (docs/mtp.md
  // "Known gaps") is measured and shown to beat the full head; has zero effect on a container
  // converted without --draft-vocab-ids, or (silently) when the container has no
  // mtp.draft_head.* tensors (Container::Mtp().HasDraftHead()).
  bool mtp_draft_reduced_vocab = false;
  // Device-resident draft loop (docs/mtp.md "device-resident draft loop", docs/r9700.md P3):
  // mirror text.embed_tokens into VRAM (~2.37-2.54 GiB bf16, depending on vocab/hidden -- see
  // docs/status.md's VRAM correction note for the measured delta) so the decode/draft path can
  // gather embedding rows on-device instead of a host memcpy + H2D per step -- see Container::
  // Load's own comment for the free-VRAM fallback. Default true; set false to force host-only
  // gather (e.g. a VRAM-constrained run that would rather keep the margin for KV cache). Exposed
  // as `--embed-device-resident {on|off}` on both r4dx-cli (src/cli/cli_args.h) and r4dx-server
  // (src/server/server_args.h) -- added 2026-09-20 (review finding: this doc comment promised the
  // escape hatch before either binary actually implemented it).
  bool embed_device_resident = true;
  // DFlash2 self-speculation (docs/dflash2.md): the largest number of DFlash2 draft tokens a
  // caller will ever verify in one VerifyWindow() call on this Model. Purely a SIZING knob -- it
  // allocates nothing of the drafter itself (that is r4dx::model::DflashDraft, a separate object
  // loaded from its own container), it only tells this Model how wide a speculative verify window
  // to size its GDN window bank and per-position logits/argmax scratch for, exactly as
  // `mtp_draft_k` does for MTP. The two are independent: the actual window capacity is
  // `1 + max(mtp_draft_k, dflash_draft_k)` (see Model::draft_window_), so a DFlash2 run needs no
  // MTP head at all and an MTP run is byte-identical to before this field existed. DFlash2's block
  // is 8 wide with the anchor at position 0, so the useful maximum here is 7.
  int64_t dflash_draft_k = 0;
  // Stage S3 (docs/dflash2.md section 7 item 5, "wire the drafter into generation"): path to a
  // DFlash2 draft container (docs/container-format.md "DFlash2 draft container"). Empty (default)
  // means no drafter -- Model behaves exactly as before this field existed. Non-empty requires
  // `dflash_draft_k > 0` (there is no point loading a drafter this Model's own verify window can
  // never accommodate) and `mtp_draft_k == 0` (docs/dflash2.md: DFlash2 and MTP are mutually
  // exclusive self-speculation families -- CLI/server also reject the combination up front, but
  // Model::Load re-checks it here so no caller can bypass the CLI/server layer and load both).
  // The container's own packed layout (its `__metadata__.dflash2.layout` field) is read directly
  // from the file and used to select which `<tensor>.{layout}.*` names to load -- this field does
  // NOT need to match ModelOptions::layout (the TARGET body's layout): a w4a16 target may run
  // against a bf16, w4a16, w4a8 or mxfp4 DFlash2 draft container, independently, exactly like the
  // draft's own weights are a completely separate set of tensors from the target's.
  std::string dflash_container;
  // Vision tower (docs/vision.md "Load policy"). The 333 `vision.*` tensors are ~0.90 GiB of
  // bf16 that a text-only run must not pay for, so this is a three-way choice rather than a bool:
  //   kAuto (default) -- load them iff the container actually has them. This is what the server
  //     and the CLI default to: a vision-capable container serves images out of the box, and a
  //     text-only container (the 4-layer test container, any --language-model-only convert) loads
  //     exactly as it did before this milestone, with no warning and no extra byte.
  //   kOn  -- load them, and THROW if the container has none, so a caller that meant to serve
  //     images finds out at startup instead of at the first request.
  //   kOff -- never load them, even from a vision-capable container. The escape hatch for a
  //     VRAM-constrained text-only run on the real container (measured: 0.8886 GiB reclaimed,
  //     docs/vision.md).
  // Exposed as `--vision {auto|on|off}` on both binaries.
  enum class VisionMode { kAuto, kOn, kOff };
  VisionMode vision = VisionMode::kAuto;
  // Tensor parallel (docs/tp.md 3.3); default-constructed == TP=1 == every pre-TP caller.
  // Staged (docs/tp.md 9.1, 2.9 step 1): with tp.world > 1, Load refuses `mtp_draft_k > 0`, a
  // non-empty `dflash_container` and `vision == kOn` (TpUnsupportedError) until P5, and loads
  // `kAuto` text-only.
  TpRankOptions tp;
};

class Model {
 public:
  static Model Load(const ModelOptions& opts);

  // Re-zeroes every piece of per-sequence state (GDN recurrent/conv state, KV/MTP-KV position
  // bookkeeping, pos_/started_, MTP head state) WITHOUT touching any weight -- the cheap
  // alternative to Model::Load() a caller (src/server's Engine) should use whenever a new
  // request's prompt does not extend the currently-fed token prefix, instead of paying a full
  // container reload (measured ~18.6s against the real 64-layer w4a16 container, docs/server.md's
  // "Reset cost") just to get back to "nothing committed yet".
  //
  // What this does NOT need to touch, and why: the paged KV caches (kv_caches_, and MtpHead's own
  // internal cache) have no separate "current length" bookkeeping of their own -- slot(t)==t
  // (contiguous, single-sequence, see paged_kv_cache.hpp's file comment) and every write always
  // happens at the CALLER-tracked position (pos_) BEFORE anything can read it, so a stale byte at
  // position >= the new pos_==0 is unreachable: the same "self-correcting via position overwrite"
  // property mtp_head.h's file comment already relies on for rejected speculative candidates.
  // Resetting pos_/started_ to their post-construction values is therefore sufficient "KV
  // bookkeeping" reset on its own. GDN state is different -- has_init reads the PREVIOUS token's
  // recurrent/conv state rather than always overwriting position-indexed slots, so it genuinely
  // needs to be re-zeroed (GdnStateManager::ZeroAll, matching Load()'s own initial zeroing).
  // GdnControlCache/arena_/buf_a_/buf_b_/buf_normed_/embed_staging_/logits_dev_/argmax_dev_/
  // attn_positions_/attn_seqused_k_ are all pure per-call scratch with no state that outlives one
  // RunChunk/VerifyWindow call, so none of them need touching either.
  void Reset();

  Model(Model&&) = default;
  Model& operator=(Model&&) = default;
  Model(const Model&) = delete;
  Model& operator=(const Model&) = delete;

  // Under tensor parallelism (ModelOptions::tp.world > 1) Config() is the RANK-local config (head
  // counts and intermediate_size divided by the world, docs/tp.md 3.2; vocab_size stays global) and
  // GlobalConfig() the container's own; at TP=1 the two are equal.
  const ModelConfig& Config() const { return container_.Config(); }
  const ModelConfig& GlobalConfig() const { return container_.GlobalConfig(); }
  const Container& GetContainer() const { return container_; }

  // ---- vision tower (docs/vision.md) -----------------------------------------------------------
  // True iff this Model can encode an image: the container carried vision.* weights AND
  // ModelOptions::vision resolved to "on". EncodeImages throws otherwise, rather than returning
  // an empty result a caller could mistake for a blank image.
  bool HasVision() const { return vision_.has_value() && container_.HasVision(); }

  // `pixel_values`/`grids` are r4dx::vision::PreprocessImages' output (host, fp32). Writes the
  // merged image embeddings, [sum(merged tokens), hidden_size] bf16, into `out` on the device --
  // the rows that a later milestone splices into the text embedding sequence at the image
  // placeholder positions.
  void EncodeImages(const float* pixel_values, int64_t total_patches,
                    const std::vector<vision::GridThw>& grids, core::DeviceBuffer<uint16_t>* out,
                    vision::VisionEncodeStats* stats = nullptr,
                    const vision::VisionTrace* trace = nullptr);

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
  //
  // `on_chunk_captured`, if non-null, is invoked once per internal RunChunk call (in chunk order),
  // AFTER that chunk has run and BEFORE the next chunk's RunChunk call overwrites DflashFeatureBuffer
  // / DflashFeatureRows() -- review finding, 2026-09-20: a capture attached via
  // AttachDflashFeatureCapture() and left to a plain Prefill() call with no drain hook would only
  // ever survive with the LAST <=64-token chunk's rows, silently discarding every earlier chunk's
  // features, because dflash_features_dev_ is sized max_chunk_ rows (not max_ctx -- a max_ctx-sized
  // capture buffer would cost gigabytes of VRAM per attached target layer, see
  // AttachDflashFeatureCapture's own doc comment) and every RunChunk call rewrites it starting at
  // row 0. A DFlash2 caller that needs every prefilled position's features (docs/dflash2.md) MUST
  // pass this callback and drain DflashFeatureBuffer()/DflashFeatureRows() inside it (e.g. inject
  // that chunk's rows into the draft's own KV store) before returning; a caller with no capture
  // attached, or one that only cares about the tail chunk, may pass nullptr (the default) exactly as
  // before -- zero cost, byte-identical to pre-this-fix behavior (a null std::function is a single
  // pointer compare per chunk, never invoked).
  std::vector<float> Prefill(const std::vector<int32_t>& token_ids,
                              const std::function<void()>& on_chunk_captured = nullptr);

  // ---- multimodal prefill (docs/vision.md "Text-side splicing") ---------------------------------
  // One image occurrence inside a PrefillMultimodal call's OWN token vector.
  struct ImageSpan {
    // Index of the first image-placeholder token (config.json's image_token_id, 248056) within
    // THIS call's `token_ids` -- not an absolute sequence index, so a continuation prefill states
    // its spans relative to the tail it is feeding, exactly as it states its tokens.
    int64_t offset = 0;
    // Length of the placeholder run == the image's MERGED token count
    // (grid.MergedTokenCount(merge_size)). Validated against `grid` and against the tokens
    // actually present at [offset, offset+tokens).
    int64_t tokens = 0;
    // The image's PATCH grid (pre-merge), which is what the mrope advance rule
    // `current_pos += max(grid.h, grid.w) / merge_size` reads -- NOT the merged grid, and NOT the
    // token count (docs/vision.md).
    vision::GridThw grid;
    // Device [tokens, hidden_size] bf16: this image's rows of Model::EncodeImages' output, which
    // overwrite the embed_tokens lookup at the placeholder positions.
    const uint16_t* embeds = nullptr;
  };

  // Prefill with images spliced in. Identical to Prefill() in every other respect (same chunking,
  // same GDN/KV bookkeeping, same returned tail logits, same `on_chunk_captured` contract), plus:
  //   (a) the embedding rows at each span's placeholder positions are overwritten with that span's
  //       merger rows instead of an embed_tokens lookup;
  //   (b) every full-attention layer's rope gets per-token 3-axis (t,h,w) position ids built by
  //       r4dx::vision::BuildMropePositionIds, continued from this Model's current mrope state;
  //   (c) the resulting mrope delta is recorded, so every later decode / MTP verify / MTP draft /
  //       DFlash2 step in this conversation ropes at `sequence_index + delta` while its KV slot
  //       stays the plain sequence index.
  // `images` must be sorted by `offset` and non-overlapping. An EMPTY `images` on a Model that has
  // never seen one takes the exact pre-vision code path (no extra upload, no extra kernel, the
  // single-row rope entry point) and is byte-identical to Prefill(); an empty `images` on a Model
  // that HAS seen one is the multi-turn continuation case and keeps roping at the carried delta.
  //
  // `rope_rows_out` (diagnostics, non-null only under the validation tooling): receives the
  // [3, token_ids.size()] rows this call actually fed the rope kernel, compact and indexed from 0.
  // These are the engine's OWN rows, not a re-derivation -- which is the point: it is what lets
  // tools/reference/rope_index_golden.py --verify-prompt check them against the unmodified
  // reference `get_rope_index` for a REAL rendered prompt rather than for a synthetic fixture.
  std::vector<float> PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                        const std::vector<ImageSpan>& images,
                                        const std::function<void()>& on_chunk_captured = nullptr,
                                        std::vector<int32_t>* rope_rows_out = nullptr);

  // True once an image has been spliced into this conversation, i.e. once rope positions and KV
  // slot indices have diverged. False for every text-only conversation, which is exactly the
  // condition under which every rope call site keeps its pre-vision single-row path.
  bool MropeActive() const { return mrope_active_; }
  // The rope position a token at absolute sequence index `s` gets is `s + MropeDelta()` on all
  // three axes, for every `s` past the last image (docs/vision.md). 0 for a text-only conversation.
  int64_t MropeDelta() const { return mrope_delta_; }

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

  // ---- Sampled decode (docs/sampling.md section 8, Milestone 6 stage S2) -----------------------
  // The sampled counterpart of DecodeStepGreedy: computes the same next-token logits on-device,
  // SUMMARISES them there (r4dx_topk_lse_f32 -- the top-64 raw logits + ids in canonical order plus
  // the row's logsumexp, 516 bytes) and reads back only that summary instead of the ~993 KB fp32
  // logits row, then resolves the token on the host with r4dx::kernels::SampleFromSummary. Exactly
  // ONE draw is taken from `rng` per emitted token.
  //
  // The returned token is BY CONSTRUCTION the token r4dx::kernels::SampleCanonical would return for
  // this row's full logits, the same `params` and that same draw (docs/sampling.md section 2):
  // whenever the summary cannot PROVE the answer (SampleFromSummary returns resolved==false -- e.g.
  // a top_p nucleus reaching past the top-64) this falls back for THIS row only, copying the full
  // fp32 row back out of the logits buffer the step already filled -- no second lm_head pass -- and
  // running the full-vocab canonical sampler with the SAME u. SampledFallbackRows() counts those.
  //
  // `params.temperature <= 0` (greedy) routes straight to DecodeStepGreedy and consumes NO draw, so
  // a greedy caller's generator state and emitted tokens are byte-identical to before this method
  // existed (docs/sampling.md design point E). A `temperature` so small that 1/temperature is not a
  // finite positive float (below ~1e-38) also has no device summary; that row takes the plain
  // full-logits path instead, still with exactly one draw.
  int32_t DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params,
                             std::mt19937_64& rng);

  // How many rows, since Load(), SampleFromSummary could not resolve from the device summary alone
  // and this Model therefore fell back to a full-row D2H + the full-vocab canonical sampler. A pure
  // PERFORMANCE counter (the emitted token is identical either way) -- a rate near 1.0 means the
  // request's filters routinely reach past the summary's top-64 (docs/sampling.md section 5.3's
  // measured fallback rates), which is worth knowing when a sampled request is slower than expected.
  // Not per-sequence state: Reset() deliberately leaves it alone.
  int64_t SampledFallbackRows() const { return sampled_fallback_rows_; }

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
    int64_t r4dx_kernel_launches = 0;   // r4dx::kernels::r4dx_kernel_launch_counter_get() delta for
                                         // just this one step (docs/r9700.md P2/task item 4) --
                                         // r4dx-owned launches only, NOT third_party/libr4d's own
                                         // r4d_gemm_*/r4d_gdn_*/r4d_attn_* launches; see kernels.h.
  };

  // Runs exactly one decode step (T=1) like DecodeStep, but wraps each kernel-family call in a
  // hipEvent pair so the returned StepProfile breaks down where the step's time actually went --
  // tools/profile's own request ("per-op profile ... per kernel family per decode step"). NOT on
  // the hot path (DecodeStep/DecodeStepGreedy never call this): hipEventCreate/Record/Synchronize
  // per op adds real host-side overhead of its own, so this is diagnostic-only, invoked at most
  // once per r4dx-cli process via --profile (src/cli/main.cpp).
  // Tensor parallel: throws core::TpUnsupportedError (profiling under TP is out of scope, docs/tp.md
  // 1.2) -- as does PrefillProfiled below.
  StepProfile DecodeStepProfiled(int32_t token_id);

  // Milestone 3 profiling pass (docs/r9700.md R5/Q7): prefill's analogue of DecodeStepProfiled --
  // chunks `token_ids` through the same <=max_chunk_-row prefill path Prefill() uses (is_prefill
  // GDN kernels, AttnPrefillFp8Kv once T exceeds the decode band), with `&acc` threaded into every
  // GdnLayer::Forward/AttentionLayer::Forward/Mlp::Forward call so every kernel family's GPU time
  // is summed by name ACROSS every chunk and every layer (e.g. "gemm:mlp.gate_up" accumulates one
  // hipEvent pair per layer per chunk into one bucket) -- divide entries[*].ms by the number of
  // chunks (callers know that: ceil(token_ids.size() / max_chunk) -- max_chunk is always 64 today,
  // see docs/architecture.md) for a "per T=64 chunk" figure, matching docs/r9700.md Q7's ask.
  // Diagnostic-only like DecodeStepProfiled: discards every chunk's logits (never computes
  // final_norm/lm_head at all, matching Prefill()'s own non-final-chunk skip -- see RunChunk's
  // want_logits comment) and skips MTP KV priming (out of this profiling pass's scope). Commits
  // real pos_/GDN/KV state exactly like Prefill() -- not meant to be combined with a real
  // generation afterward in the same process, same caveat as DecodeStepProfiled's own doc comment.
  // host_enqueue_ms/finish_wait_ms are not meaningfully split here (each chunk's positions/seqused_k
  // upload forces a stream_.Synchronize() before the next chunk, same hazard RunChunk's own comment
  // describes) -- both are left at 0; use wall_ms and gpu_sum_ms.
  StepProfile PrefillProfiled(const std::vector<int32_t>& token_ids);

  // True iff this Model was Load()'d with mtp_draft_k > 0 (and the container had mtp.* weights).
  bool MtpEnabled() const { return static_cast<bool>(mtp_); }

  // Diagnostic/measurement accessor (docs/r9700.md R9 task item 5, "report reduced-vs-full
  // acceptance at matched K"): true iff every subsequent DecodeStepMtpGreedy call on this Model will
  // actually use the reduced-vocab draft head (both ModelOptions::mtp_draft_reduced_vocab was true
  // AND the loaded container has one, container_.Mtp().HasDraftHead()) -- lets a caller/tool report
  // which path a given run took without duplicating Load()'s own two-condition check.
  bool MtpUsingReducedVocabDraft() const {
    return mtp_draft_reduced_vocab_ && mtp_ && container_.HasMtp() && container_.Mtp().HasDraftHead();
  }

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

  // The sampled counterpart of DecodeStepMtpGreedy -- "sample-and-match" (docs/sampling.md section
  // 9): drafts through the MTP head exactly as that method does, verifies the same
  // [token_id, d1..dm] window in ONE VerifyWindow pass, then walks the window SAMPLING one token per
  // row (one draw each, from the row's device summary) and stops at the first row whose sampled
  // token is not the draft that follows it. Commits matched+1 rows with EXACTLY the state updates
  // the greedy method performs for that accepted count -- the two share one implementation
  // (Model::SpeculativeRoundImpl in model.cpp) precisely so they cannot drift apart.
  //
  // Why this is exactly rejection sampling, i.e. why it is LOSSLESS: a greedy drafter's proposal
  // distribution is a point mass on its drafted token x, so the standard accept rule
  // min(1, p(x)/q(x)) degenerates to "accept with probability p(x)", and the residual distribution
  // on a rejection is p conditioned on y != x. Drawing y ~ p once and accepting iff y == x is
  // exactly that rule. Because exactly ONE draw is consumed per EMITTED token, the stronger
  // property also holds: for a fixed seed this emits the same token sequence, token for token, that
  // DecodeStepSampled emits from the same prompt (tests/model/test_mtp.cpp's
  // CheckSampledRoundsMatchPlain, tests/model/test_dflash_e2e.cpp's own equality check).
  //
  // Same contract as DecodeStepMtpGreedy otherwise (1..k+1 committed tokens, mtp_seed_hidden_
  // re-seeded from the last committed row). `params.temperature <= 0` routes straight to
  // DecodeStepMtpGreedy and consumes no draw.
  std::vector<int32_t> DecodeStepMtpSampled(int32_t token_id, int64_t k,
                                             const kernels::SampleParams& params,
                                             std::mt19937_64& rng);

  // Largest speculative-verify window this Model was sized for: 1 + max(mtp_draft_k,
  // dflash_draft_k) (ModelOptions). VerifyWindow accepts up to this many candidate rows.
  int64_t DraftWindow() const { return draft_window_; }

  // Commits `num_committed` (1..DraftWindow()) rows of the window the most recent VerifyWindow()
  // call ran -- the piece DecodeStepMtpGreedy does inline for MTP, exposed for a drafter that owns
  // its own round loop (DFlash2, docs/dflash2.md section 5 steps 4-5). Advances pos_ by exactly
  // that many positions and threads the count into the NEXT GDN decode/verify call's window-bank
  // seed (gdn_state.h's file comment), which is what keeps GDN state in step with the real
  // committed sequence. Deliberately does NOT touch mtp_seed_hidden_: a DFlash2 driver has no MTP
  // head, and a Model that has BOTH would be re-seeding MTP from a window this call knows nothing
  // about. Must be called at most once per VerifyWindow() call.
  void CommitVerifiedWindow(int64_t num_committed);

  // Runs the real model over `candidates` (1..DraftWindow() tokens, is_prefill_path=false, the
  // same speculative-verify decode path DecodeStepMtpGreedy uses) and returns the greedy argmax
  // token predicted at EVERY position, WITHOUT committing any state: pos_ is not advanced and
  // mtp_num_accepted_dev_/mtp_seed_hidden_ are not updated (DecodeStepMtpGreedy does both once it
  // knows how many candidates were accepted -- see that method). Exposed publicly (rather than
  // kept as an internal helper) for tests/model/test_mtp.cpp, which needs to check this call's raw
  // per-position logits against sequential DecodeStep's own logits -- not just DecodeStepMtpGreedy's
  // argmax-only return value. `logits_out`, if non-null, is resized to
  // candidates.size()*Config().vocab_size and filled with this call's flat [T,vocab] fp32 logits
  // (an extra D2H a normal (non-test) caller does not need -- nullptr skips it).
  //
  // Requires DraftWindow() > 1, i.e. this Model was Load()'d with mtp_draft_k > 0 OR
  // dflash_draft_k > 0 -- the GDN state window bank and the verify_logits_dev_/verify_argmax_dev_
  // scratch this needs are only sized then. An MTP head is NOT required (generalised for DFlash2,
  // which verifies through this same path with no mtp.* weights anywhere in the container); MTP's
  // own behaviour at a given mtp_draft_k is byte-identical to before that generalisation.
  //
  // `summaries_out` (Milestone 6 stage S2, docs/sampling.md section 8), if non-null, is resized to
  // candidates.size() and filled with each row's DEVICE row summary (r4dx_topk_lse_f32 over
  // verify_logits_dev_, at `summary_inv_temperature` = 1/temperature): the top-64 raw logits + ids
  // in canonical order plus that row's logsumexp, i.e. rows*516 bytes instead of the rows*~993 KB
  // `logits_out` costs. This is what a SAMPLED speculative round reads instead of the full window;
  // the summary kernel runs on the same stream right after the per-row argmaxes, so a summarised
  // verify pass is one extra pair of launches per round and no extra lm_head work. `logits_out` and
  // `summaries_out` are independent -- pass both (a test wanting to cross-check), either, or
  // neither. `summary_inv_temperature` is ignored when `summaries_out` is null and must otherwise
  // be finite and > 0.
  std::vector<int32_t> VerifyWindow(const std::vector<int32_t>& candidates,
                                     std::vector<float>* logits_out = nullptr,
                                     std::vector<kernels::RowSummary>* summaries_out = nullptr,
                                     float summary_inv_temperature = 1.0f);

  // Copies row `row` (0..DraftWindow()-1) of the logits the most recent VerifyWindow() call
  // produced back to the host, fp32, vocab-wide -- the ROW-GRANULAR counterpart of that method's
  // `logits_out`, for a sampled round whose summary could not resolve one single row (model.cpp's
  // SampleVerifyRow, and tests that re-check one row against the full-vocab sampler). Valid only
  // until the next RunChunk/VerifyWindow call overwrites verify_logits_dev_, exactly like
  // DflashFeatureBuffer()'s own lifetime. Requires DraftWindow() > 1.
  void ReadVerifyLogitsRow(int64_t row, std::vector<float>& out) const;

  // Diagnostic-only accessor (docs/mtp.md "Acceptance gap investigation", h_seed drift pass): reads
  // back `mtp_seed_hidden_` -- the exact [hidden] bf16 row `MtpHead::Draft`'s first step consumes --
  // as raw bf16 bit patterns (uint16_t), for direct cross-layout comparison against a bf16
  // (exact-arithmetic) reference. Requires MtpEnabled() (mtp_seed_hidden_ is only sized when
  // mtp_draft_k>0) and at least one prior RunChunk-driving call (Prefill/DecodeStep*/
  // DecodeStepMtpGreedy) -- throws otherwise. Not on any hot path; a plain host D2H copy of 5120
  // bf16 values (10 KB), cheap enough to call after every Prefill/DecodeStep in a diagnostic tool
  // without perturbing anything it measures.
  std::vector<uint16_t> DebugSeedHiddenBf16() const;

  // ---- DFlash2 target feature capture (docs/dflash2.md "Implementation", Milestone 5 B1 item 1)
  // ------------------------------------------------------------------------------------------
  // Attaches (or replaces) a DFlash2 feature-capture hook: every subsequent RunChunk (prefill
  // chunk or plain decode step) and VerifyWindow call copies the residual stream AS IT ENTERS
  // layer L, for every L in `target_layers` (0-based, sorted ascending, each < NumLoadedLayers()),
  // into a device buffer laid out [rows][target_layers.size()*hidden] bf16, column order matching
  // `target_layers`'s own order -- directly the DFlash2 encoder's expected input activation
  // (docs/dflash2.md section on FEATURES). `target_layers` empty is equivalent to Detach().
  // Cost when attached: exactly `target_layers.size()` hipMemcpy2DAsync device-to-device strided
  // copies per call (no host sync, no new kernel launch -- so
  // r4dx::kernels::r4dx_kernel_launch_counter_get() is unaffected either way). Cost when never
  // attached (the default, and every existing caller/test): a single `.empty()` check per layer
  // per call, no allocation, no copy, byte-identical to pre-B1 behavior.
  //
  // PRECONDITION (review finding, 2026-09-20): only call this (or DetachDflashFeatureCapture) when
  // this Model's stream is idle -- i.e. right after construction/Reset(), or after a prior
  // Prefill/DecodeStep*/VerifyWindow call has returned (every one of those already ends with a
  // stream_.Synchronize() before returning, per their own comments), never from inside a
  // Prefill on_chunk_captured callback or any other point where a RunChunk/VerifyWindow call could
  // still have stream-ordered work in flight. Both methods resize/free dflash_features_dev_
  // (core::DeviceBuffer::Resize -> hipFree, not stream-ordered), and any pointer previously returned
  // by DflashFeatureBuffer() is invalidated by either call -- do not retain it across an
  // Attach/Detach call.
  void AttachDflashFeatureCapture(std::vector<int64_t> target_layers);
  void DetachDflashFeatureCapture();
  bool DflashFeatureCaptureAttached() const { return !dflash_target_layers_.empty(); }
  // Valid until the NEXT RunChunk/VerifyWindow call overwrites it (same lifetime convention as
  // mtp_last_hidden_ above). Rows == the T of whichever RunChunk/VerifyWindow call last ran
  // (<=64 for a prefill chunk, <=mtp_draft_k_+1 for a verify window, 1 for plain decode).
  // UNDEFINED (do not read) after Reset() and before the next RunChunk/VerifyWindow call of the new
  // sequence: Reset() zeroes DflashFeatureRows() to 0 precisely so a caller cannot mistake the
  // previous sequence's stale buffer contents for the new sequence's -- see Reset()'s own comment.
  const uint16_t* DflashFeatureBuffer() const { return dflash_features_dev_.data(); }
  // 0 immediately after Load()/Reset(), before this Model has run any RunChunk/VerifyWindow call
  // for the current sequence.
  int64_t DflashFeatureRows() const { return dflash_feature_rows_; }
  int64_t DflashFeatureCols() const {
    return static_cast<int64_t>(dflash_target_layers_.size()) * Config().hidden_size;
  }
  const std::vector<int64_t>& DflashTargetLayers() const { return dflash_target_layers_; }

  // Per-RunChunk capture observer (review finding, 2026-09-20; generalises the narrower
  // `Prefill(..., on_chunk_captured)` drain hook below). When a capture is attached AND an observer
  // is set, it is invoked once at the end of EVERY RunChunk call -- every prefill chunk AND every
  // plain decode step, in order, after that chunk's captured rows are complete on the device and
  // BEFORE the next call overwrites `dflash_features_dev_` at row 0 again. Arguments: the capture
  // buffer, the number of rows this call filled, and the ABSOLUTE sequence position of row 0.
  //
  // This is the hook a DFlash2 driver actually wants: `dflash_features_dev_` is sized `max_chunk_`
  // rows, not `max_ctx` (a max_ctx-sized capture would cost gigabytes of VRAM per attached target
  // layer), so without it a >64-token prompt silently loses every chunk's features but the last,
  // and a plain decode step's single captured row is likewise lost before the next step.
  // The observer may enqueue device work: RunChunk has already synchronised its own stream by the
  // time it runs, so both a same-stream and a separate-stream consumer are safe.
  //
  // VerifyWindow deliberately does NOT invoke it. A verify window's rows are CANDIDATES, most of
  // which may be rejected; only the accepted prefix may ever be injected into a drafter's store
  // (docs/dflash2.md section 5 step 4), and only the driver knows how many that is -- so the driver
  // reads DflashFeatureBuffer()/DflashFeatureRows() itself after VerifyWindow returns.
  using DflashCaptureObserver =
      std::function<void(const uint16_t* features, int64_t rows, int64_t start_pos)>;
  void SetDflashCaptureObserver(DflashCaptureObserver observer) {
    dflash_observer_ = std::move(observer);
  }
  void ClearDflashCaptureObserver() { dflash_observer_ = nullptr; }

  // ---- DFlash2 self-speculative decode (docs/dflash2.md section 7 item 3) -----------------------
  // True iff this Model was Load()'d with a non-empty ModelOptions::dflash_container (and therefore
  // owns its own r4dx::model::DflashDraft, auto-fed by every RunChunk call -- see RunChunk's own
  // comment for the internal-vs-external-observer split).
  bool DflashEnabled() const { return dflash_.has_value(); }

  // ---- Per-request drafter-injection toggle (docs/server.md, docs/dflash2.md section 5) ---------
  // `--dflash` is a LOAD-time flag, so a long-lived server holds one Model and one drafter for
  // requests that will never speculate (`temperature>0` never calls DecodeStepDflashGreedy). While
  // this is false, RunChunk skips BOTH the per-layer target-feature capture AND the drafter
  // injection it feeds -- the point is to remove that cost, not to defer it -- so a sampled request
  // on a `--dflash` server runs at plain-decode speed.
  //
  // The drafter then LAGS: `dflash_->InjectedCount() < pos_` for as long as injection is off. That
  // is safe and explicitly supported: the next RunChunk after re-enabling injects at
  // `start_pos = pos_ > InjectedCount()`, which DflashDraft::InjectFeatures turns into a cold-ring
  // gap (its `ValidFrom()` moves up, and the skipped positions are never read again). After ANY
  // injection, `InjectedCount() == pos_` holds again, which is what DecodeStepDflashGreedy needs.
  //
  // DecodeStepDflashGreedy THROWS while this is false: it would otherwise draft a block at the
  // drafter's stale frontier instead of the model's real one. Re-enable it and run at least one
  // RunChunk (a prefill chunk or a plain decode step) first. `Reset()` does not change this flag --
  // it is a caller policy, not sequence state -- but it does reset the drafter, so an enabled
  // prefill after a Reset() starts cleanly from position 0 with no gap at all.
  //
  // Default true, i.e. every pre-existing caller (the CLI, every test, every tool) behaves exactly
  // as before. NOTE for a caller using the EXTERNAL dflash_observer_ mechanism instead of this
  // Model's own drafter: this flag gates the capture itself, so the observer stops being invoked
  // too -- do not turn it off while relying on that hook.
  void SetDflashInjectionEnabled(bool enabled) { dflash_injection_enabled_ = enabled; }
  bool DflashInjectionEnabled() const { return dflash_injection_enabled_; }

  // Drafter-ring bookkeeping, for tests/diagnostics. Both require DflashEnabled().
  // DflashInjectedCount() is the drafter's own frontier (== PositionCount() after any injection,
  // lagging it while injection is disabled); DflashValidFrom() is the lower bound of its visible
  // store, i.e. the position injection most recently RESUMED at after a gap (0 when there never
  // was one).
  int64_t DflashInjectedCount() const;
  int64_t DflashValidFrom() const;

  // The DFlash2 analogue of DecodeStepMtpGreedy, returning the SAME round-vector contract (1..k+1
  // committed tokens: 0..k accepted drafts followed by exactly one correction/bonus token) so
  // mtp_round.hpp's ProcessMtpRound and the CLI/server round loops plug in unchanged -- see that
  // header's file comment, which is already speculation-family-agnostic despite its name.
  //
  // Sequence (docs/dflash2.md section 5, section 7 item 3): DraftRound anchored at `token_id` ->
  // candidates [token_id, d1..dm] (m = however many the walk actually produced, 0..k) ->
  // VerifyWindow (<=8 rows) -> accepted `a` rows (the anchor is always accepted; a = 1 +
  // (longest confirmed draft prefix)) -> InjectFeatures the accepted prefix's own just-captured
  // target features (VerifyWindow's DflashFeatureBuffer()/DflashFeatureRows(), rows 0..a-1) at the
  // drafter's current InjectedCount() -> CommitVerifiedWindow(a) (advances pos_, threads GDN
  // acceptance exactly like MTP's own commit). The candidate block's own K/V is scratch inside
  // DflashDraft (never written to its ring), so a rejected suffix needs no rollback there, and
  // Model::VerifyWindow's own KV-rollback-via-position-overwrite already covers the target's KV/GDN
  // state -- same guarantee DecodeStepMtpGreedy already relies on.
  //
  // `token_id`: the last already-accepted real token (same convention as DecodeStep/
  // DecodeStepMtpGreedy). `k`: caps how many tokens DraftRound's walk may emit, 0..dflash_draft_k_
  // (the value Load() was given). `p_min`/`n_min`: DflashDraft::DraftRound's own early-stop/discard
  // gates (docs/dflash2.md section 4.3), <=0 disables either. Requires DflashEnabled().
  // `walk_len_out`, if non-null, is set to this round's OWN DflashDraftResult::walk_len -- how many
  // tokens the selector walk itself produced BEFORE verification (0..k), which unlike MTP's `k` is
  // not a constant every round (p_min/n_min can stop it early) -- a caller computing an acceptance
  // rate (accepted/offered) needs this per-round figure, not just `k`, to report it correctly. Does
  // not change the returned round vector's own contract at all.
  // `trace_out`, if non-null, is forwarded straight to the internal DflashDraft::DraftRound call
  // (diagnostics only -- costs one extra D2H + sync per round, same as DraftRound's own `trace`
  // param). `drafted_tokens_out`, if non-null, is set to DraftRound's own RAW drafted-token chain
  // (BEFORE verification/acceptance) -- distinct from this method's own return value, which is the
  // POST-verify committed round. Review finding (2026-09-21, item 5): these two are what let a
  // diagnostic tool compare the WIRED drafter (this Model's own internal dflash_, fed by RunChunk's
  // real capture hook) against the Python reference exactly the way
  // tests/model/tool_dflash_probe.cpp's original (external-observer, separately-owned DflashDraft)
  // mode already did -- dflash2_ref.py --real's `_compare_with_port` reads both the trace's tensors
  // and the raw `drafted_tokens` from the dump it produces.
  std::vector<int32_t> DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min,
                                                int64_t n_min, int64_t* walk_len_out = nullptr,
                                                DflashRoundTrace* trace_out = nullptr,
                                                std::vector<int32_t>* drafted_tokens_out = nullptr);

  // The sampled counterpart of DecodeStepDflashGreedy -- the identical "sample-and-match" round
  // DecodeStepMtpSampled runs (see that method for why it is lossless rejection sampling), with
  // DFlash2's DraftRound as the drafter: draft, verify, sample one token per row until one does not
  // match the draft that follows it, inject the committed rows' captured features and
  // CommitVerifiedWindow(matched+1) -- byte-for-byte the same commit tail the greedy method uses,
  // because both go through one implementation. Exactly one draw per emitted token, so for a fixed
  // seed this emits the same sequence DecodeStepSampled does (tests/model/test_dflash_e2e.cpp).
  // `params.temperature <= 0` routes straight to DecodeStepDflashGreedy and consumes no draw.
  std::vector<int32_t> DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min,
                                                 int64_t n_min, const kernels::SampleParams& params,
                                                 std::mt19937_64& rng,
                                                 int64_t* walk_len_out = nullptr);

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
  // `summary_out`: the sampled-decode counterpart of `greedy_token_out` (docs/sampling.md section
  // 8) -- when non-null (and want_logits, and greedy_token_out is null) this skips the vocab-sized
  // logits D2H too, instead summarising logits_dev_ ON DEVICE (r4dx_topk_lse_f32) and reading back
  // only that one 516-byte row summary into `summary_out->out`. The returned vector is empty in
  // that mode, exactly as it is in the greedy one.
  struct SummaryRequest {
    float inv_temperature = 1.0f;   // 1/temperature; must be finite and > 0
    kernels::RowSummary* out = nullptr;
  };
  std::vector<float> RunChunk(const std::vector<int32_t>& token_ids, bool is_prefill_path,
                               bool want_logits, int32_t* greedy_token_out = nullptr,
                               const SummaryRequest* summary_out = nullptr);

  // ---- sampled decode internals (docs/sampling.md sections 8-9) ---------------------------------
  // Enqueues r4dx_topk_lse_f32 over `rows` rows of `logits_dev` ([rows, vocab] fp32, row stride
  // vocab) into summary_{ids,vals,lse}_dev_, on stream_. `rows` may exceed the kernel's own
  // 8-row-per-call limit (a K=16 MTP window is 17 rows): it is issued in <=8-row calls on the SAME
  // stream, which serialises their shared module-scope device scratch (kernels.h's "one call in
  // flight" precondition). Caller must sync before FetchRowSummaries.
  void LaunchRowSummaries(const float* logits_dev, int64_t rows, float inv_temperature);
  // Copies those `rows` summaries back (device must be idle) into `out`, resized to `rows`.
  void FetchRowSummaries(int64_t rows, float inv_temperature, std::vector<kernels::RowSummary>& out);
  // 1/temperature, or 0 when `params` has no usable device summary (greedy, or a temperature so
  // small the reciprocal is not a finite positive float) -- callers take the full-logits path then.
  static float SummaryInvTemperature(const kernels::SampleParams& params);
  // One row of a verified window, sampled: ONE draw from `rng`, SampleFromSummary over
  // `summaries[row]`, and -- only when that cannot prove the answer -- a row-granular D2H out of
  // verify_logits_dev_ plus the full-vocab canonical sampler with the SAME draw.
  int32_t SampleVerifyRow(int64_t row, const std::vector<kernels::RowSummary>& summaries,
                           const kernels::SampleParams& params, std::mt19937_64& rng);

  // THE shared verify+acceptance middle of every speculative round -- MTP or DFlash2, greedy or
  // sampled (docs/sampling.md section 9.2). Verifies [anchor, drafts...] in ONE VerifyWindow pass
  // (asking for row summaries only in sampled mode), then walks the window emitting one token per
  // row and stopping at the first row whose emitted token is not the draft that follows it:
  //   greedy  (`params == nullptr`) -- the row's token is VerifyWindow's own argmax, so the walk is
  //                                    identical to the pre-S2 "longest confirmed draft prefix" loop
  //                                    and every greedy path stays byte-identical;
  //   sampled (`params != nullptr`) -- the row's token is SampleVerifyRow's, one draw per EMITTED
  //                                    token, in row order.
  // Returns the round vector (the matched drafts followed by one final token, size matched+1 -- the
  // contract both families' public methods already have) and sets *num_accepted_out to the number
  // of matched drafts. Commits NOTHING: the caller injects/commits matched+1 rows afterwards.
  std::vector<int32_t> VerifyAndResolveRound(int32_t anchor, const std::vector<int32_t>& drafts,
                                              const kernels::SampleParams* params,
                                              std::mt19937_64* rng, int64_t* num_accepted_out);
  // The one body behind DecodeStepMtpGreedy (params==nullptr) and DecodeStepMtpSampled.
  std::vector<int32_t> DecodeStepMtpImpl(int32_t token_id, int64_t k,
                                          const kernels::SampleParams* params,
                                          std::mt19937_64* rng);
  // The one body behind DecodeStepDflashGreedy (params==nullptr) and DecodeStepDflashSampled.
  std::vector<int32_t> DecodeStepDflashImpl(int32_t token_id, int64_t k, float p_min, int64_t n_min,
                                             const kernels::SampleParams* params,
                                             std::mt19937_64* rng, int64_t* walk_len_out,
                                             DflashRoundTrace* trace_out,
                                             std::vector<int32_t>* drafted_tokens_out);

  // ---- tensor parallel (docs/tp.md 7.2-7.5) -----------------------------------------------------
  // The full [vocab_size] fp32 row of one logits row whose [vocab_local_] shard sits at
  // `shard_row_dev` on this rank's device: D2H the shard, then HostAllGather -- the rank-order
  // concatenation IS the row in global id order, and every rank ends with the same bytes. The
  // device must be idle (after stream_.Synchronize()). Requires comm_.
  void GatherVocabRow(const float* shard_row_dev, float* full_host);
  // The merged greedy token of one row whose (local index, value) pair r4dx_argmax_val_f32 left at
  // `pair_dev` (8 bytes): D2H, + vocab_offset_, HostAllGather, tp::MergeArgmax (docs/tp.md 7.3).
  // Same preconditions as GatherVocabRow.
  int32_t MergeGreedyPair(const int32_t* pair_dev);
  // Throws core::TpUnsupportedError naming `what` when this Model is a tensor-parallel rank: the
  // paths TP does not support yet (sampled and speculative decode, P2b) or at all (profiling).
  void RequireNotTp(const char* what) const;

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
  // R3 fusion (docs/r9700.md P2/R3): persistent (NOT arena-allocated -- arena_.Reset() runs once
  // per layer iteration, but this buffer's write (one layer's Gdn/Attn or Mlp epilogue) and read
  // (the very next Forward call, same stream_, later in the same iteration or the next iteration)
  // must survive across that Reset(); it is a plain [max_chunk_, hidden] scratch buffer reused
  // (overwritten) at every fusion boundary in stream order, never read after being superseded, so
  // one buffer suffices -- see RunChunk/DecodeStepGreedy/DecodeStepMtpGreedy's per-layer loops.
  core::DeviceBuffer<uint16_t> buf_normed_;
  // R2/P2 (docs/r9700.md): buf_normed_'s fused quant-epilogue companion, same persistent
  // (not-arena) reuse-in-stream-order lifetime as buf_normed_ itself -- whichever layer boundary
  // most recently wrote buf_normed_ also writes its epilogue here (when body_epilogue_ !=
  // r4dx_epilogue_none), for the very next Forward call to consume as its x_normed_pre. Sized for
  // the widest epilogue format (f16, 2 bytes/element) at hidden width; a narrower format (fp8/int8,
  // 1 byte/element) just uses the buffer's first half.
  core::DeviceBuffer<uint8_t> buf_normed_pre_;        // [max_chunk_, hidden] bytes (f16-sized)
  core::DeviceBuffer<float> buf_normed_pre_scale_;    // [max_chunk_]
  // r4dx_epilogue (kernels.h) this Model's body layout wants every fused producer epilogue to
  // emit -- r4dx_epilogue_none for a bf16 body. Computed once in Load() from ModelOptions::layout;
  // every per-weight fusion site still independently verifies it against that specific weight's
  // OWN actual layout before using it (LoadQuantLinearWithFallback can fall one tensor back to
  // bf16 independently of this container-wide default -- see gdn_layer.cpp's defensive comment).
  int body_epilogue_ = 0;
  core::DeviceBuffer<float> logits_dev_;        // [vocab_local_] fp32, one row at a time
  core::DeviceBuffer<int32_t> argmax_dev_;      // [1] -- DecodeStepGreedy's on-device argmax result

  // ---- tensor parallel (docs/tp.md 4.4, 7.2); inert at TP=1 -------------------------------------
  core::TpComm* comm_ = nullptr;  // ModelOptions::tp.comm, non-owning; nullptr at TP=1
  // This rank's lm_head rows (container_.LmHead().N) and the global id of the first one
  // (Config().VocabShardBegin()): vocab_size and 0 at TP=1. Every device logits buffer, kernel
  // `vocab` argument and D2H count uses vocab_local_; every host row a sampler or caller sees is
  // a gathered full row of Config().vocab_size (GatherVocabRow) -- docs/tp.md 7.2's rule.
  int64_t vocab_local_ = 0;
  int64_t vocab_offset_ = 0;
  // TP only: [draft_window_][2] of {int32 local index, float value} -- r4dx_argmax_val_f32's
  // output, one 8-byte tp::ArgmaxPair per row. argmax_dev_/verify_argmax_dev_ serve TP=1.
  core::DeviceBuffer<int32_t> argmax_pair_dev_;
  std::vector<float> gather_shard_host_;  // TP only: GatherVocabRow's [vocab_local_] staging

  // Persistent (not arena-allocated) scratch every full-attention layer's AttentionLayer::Forward
  // shares within one RunChunk call: `positions[t] = pos_ + t` doubles as both the RoPE position
  // ids and the KV slot_mapping (contiguous block table => slot==pos), and `seqused_k[0] = pos_ +
  // T`. Neither depends on the layer, only on (pos_, T), so RunChunk uploads them once per chunk
  // and every attention layer in that chunk reuses the same device pointers -- see
  // AttentionLayer::Forward's doc comment for why these must NOT be arena-allocated.
  core::DeviceBuffer<int32_t> attn_positions_;  // [max_chunk_]
  core::DeviceBuffer<int32_t> attn_seqused_k_;  // [1]

  // ---- 3-axis mrope state (docs/vision.md "Text-side splicing") ---------------------------------
  // Once an image has been spliced into this conversation, a token's ROPE position stops being its
  // sequence index. `attn_rope_pos_` is the per-chunk [3, T] companion to attn_positions_ above --
  // same "uploaded once per chunk, shared by every full-attention layer, NOT arena-allocated"
  // lifetime and the same blocking-upload-only-when-the-device-is-idle hazard.
  //
  // Everything here stays at its post-construction value for a text-only conversation, and
  // `mrope_active_ == false` is what keeps every rope call site on its pre-vision single-row path
  // (RopePositionsForChunk returns nullptr), so a text-only run is unchanged byte for byte.
  core::DeviceBuffer<int32_t> attn_rope_pos_;  // [3 * max_chunk_], rows t/h/w, compact per chunk
  std::vector<int32_t> rope_pos_host_;         // [3 * max_chunk_] staging for the upload above
  bool mrope_active_ = false;
  // A token at absolute sequence index `s` past the last image ropes at `s + mrope_delta_`.
  int64_t mrope_delta_ = 0;
  // The block PrefillMultimodal is CURRENTLY feeding: its per-token [3, N] rows (compact, indexed
  // from 0) and the absolute sequence index its row 0 sits at. Empty outside such a call -- every
  // position outside this range is text and gets `s + mrope_delta_` on all three axes.
  std::vector<int32_t> mrope_block_;
  int64_t mrope_block_base_ = 0;
  // The image spans of that same block, same index space as mrope_block_.
  std::vector<ImageSpan> mrope_block_images_;

  // Fills rope_pos_host_/attn_rope_pos_ with the [3, T] rope rows for absolute sequence positions
  // [start, start+T) and returns the device pointer AttentionLayer::Forward's `rope_pos3` wants --
  // or nullptr when this conversation has no mrope divergence at all, which is what keeps a
  // text-only run on the pre-vision path. Blocking upload: same precondition as attn_positions_'
  // own (the device must be idle at the call site).
  const int32_t* RopePositionsForChunk(int64_t start, int64_t T);
  // Fills `out3` (resized to 3*T) with the same rows on the HOST, for a caller that does its own
  // upload (MtpHead::PrimeKv) or needs only the temporal row (DflashDraft). Always fills, even for
  // a text-only conversation (where every row is just `start + t`).
  void RopePositionsHost(int64_t start, int64_t T, std::vector<int32_t>* out3) const;
  // Overwrites `dst`'s rows with the merger embeddings of whichever pending image spans intersect
  // the chunk at absolute positions [start, start+T). No-op when mrope_block_images_ is empty.
  void SpliceImageEmbeddings(int64_t start, int64_t T, uint16_t* dst, int64_t hidden);

  std::vector<std::optional<GdnStateManager>> gdn_states_;             // one per GDN layer
  std::vector<std::optional<attention::PagedKvCache>> kv_caches_;      // one per attn layer
  GdnControlCache gdn_control_;  // shared by every GDN layer -- see gdn_state.h

  int64_t max_chunk_ = 64;
  int64_t pos_ = 0;        // tokens already committed to KV/GDN state
  bool started_ = false;   // false only before the very first RunChunk call (GDN has_init gate)

  // ---- MTP self-speculation (docs/mtp.md), all empty/unused when mtp_draft_k==0 -----------------
  std::optional<MtpHead> mtp_;
  int64_t mtp_draft_k_ = 0;
  int64_t dflash_draft_k_ = 0;  // ModelOptions::dflash_draft_k, copied at Load()
  // 1 + max(mtp_draft_k_, dflash_draft_k_): the widest speculative-verify window this Model's GDN
  // window bank and verify scratch were sized for, and the bound VerifyWindow enforces. 1 (no
  // speculation) makes every decode path degenerate exactly to the pre-speculation behaviour.
  int64_t draft_window_ = 1;
  bool mtp_draft_reduced_vocab_ = false;  // ModelOptions::mtp_draft_reduced_vocab, copied at Load()
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
  // Scratch for VerifyWindow: logits for up to draft_window_ candidate positions at once, plus one
  // argmax result per position. Shared by both speculation families (MTP and DFlash2) -- the verify
  // pass is identical for either draft source (docs/dflash2.md section 7 item 5).
  core::DeviceBuffer<float> verify_logits_dev_;    // [draft_window_ * vocab_local_]
  core::DeviceBuffer<int32_t> verify_argmax_dev_;  // [draft_window_]

  // ---- sampled decode scratch (docs/sampling.md section 8) -------------------------------------
  // r4dx_topk_lse_f32's outputs for up to draft_window_ rows at once: ~4 KB of VRAM at K=64 and the
  // widest window this Model was sized for, allocated unconditionally (a plain sampled decode step
  // summarises one row, a sampled speculative round summarises the whole window). `*_host_` are the
  // matching staging vectors, and `sampled_row_scratch_` is the ONE full fp32 row a fallback copies
  // back -- all three are plain host scratch, reused rather than reallocated per token.
  core::DeviceBuffer<int32_t> summary_ids_dev_;  // [rows * R4DX_TOPK_LSE_K]
  core::DeviceBuffer<float> summary_vals_dev_;   // [rows * R4DX_TOPK_LSE_K]
  core::DeviceBuffer<float> summary_lse_dev_;    // [rows]
  std::vector<int32_t> summary_ids_host_;
  std::vector<float> summary_vals_host_;
  std::vector<float> summary_lse_host_;
  std::vector<float> sampled_row_scratch_;
  std::vector<kernels::RowSummary> round_summaries_;  // one verified window's summaries
  int64_t sampled_fallback_rows_ = 0;                 // see SampledFallbackRows()
  // Non-owning: whichever of buf_a_/buf_b_ the most recent VerifyWindow() call left its final
  // per-position hidden states in (which one depends on how many full-attention layers ran, so it
  // is not knowable statically) -- valid only until the NEXT RunChunk/VerifyWindow call's own
  // EmbedTokens overwrites it. DecodeStepMtpGreedy reads row `num_accepted_drafts` out of this
  // right after VerifyWindow returns, before anything else touches buf_a_/buf_b_.
  uint16_t* mtp_last_hidden_ = nullptr;

  // ---- DFlash2 target feature capture (see AttachDflashFeatureCapture above) --------------------
  std::vector<int64_t> dflash_target_layers_;       // empty == capture disabled (the default)
  core::DeviceBuffer<uint16_t> dflash_features_dev_;  // [max_chunk_, target_layers_.size()*hidden]
  int64_t dflash_feature_rows_ = 0;  // rows filled by the most recent RunChunk/VerifyWindow call
  DflashCaptureObserver dflash_observer_ = nullptr;  // see SetDflashCaptureObserver
  bool dflash_injection_enabled_ = true;  // see SetDflashInjectionEnabled

  // Stage S3: the drafter this Model owns when Load()'d with a non-empty
  // ModelOptions::dflash_container, nullopt otherwise (the overwhelming common case, byte-identical
  // to before this field existed). Auto-fed by RunChunk (prefill chunks + plain decode steps) via a
  // direct member call -- NOT via dflash_observer_ above, which is reserved for a caller that owns
  // its OWN separate DflashDraft object (e.g. tests/model/tool_dflash_probe.cpp) and would otherwise
  // have no way to drain a capture Model itself does not know how to consume. VerifyWindow never
  // auto-injects into this (or any) drafter, by design -- see DecodeStepDflashGreedy.
  std::optional<DflashDraft> dflash_;

  // ---- vision tower (docs/vision.md) -----------------------------------------------------------
  // Present iff the container's vision.* weights were loaded (ModelOptions::vision resolved to on).
  // Holds only the encode stream and the scratch arena -- the weights themselves live in
  // container_, and are passed to Encode per call, so moving this Model cannot leave the tower
  // pointing at a moved-from container.
  std::optional<vision::VisionTower> vision_;
};

}  // namespace r4dx::model
