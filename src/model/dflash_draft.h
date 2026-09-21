// r4dx::model::DflashDraft -- the DFlash2 block-diffusion self-speculative drafter, on device.
// docs/dflash2.md is the authoritative spec for every convention this file implements (RoPE
// pairing/theta, the PLAIN rmsnorm weight form, the SWA visibility rule, the conv delta/base
// layout, the selector lattice and its greedy walk, the p_min/n_min policy, the block/anchor
// lifecycle). This class is the structural analogue of `MtpHead` (src/model/mtp_head.h) for the
// other self-speculation family: it owns its own weights, its own KV store and its own scratch,
// and it never touches the target `Model`'s state -- a driver feeds it target features and asks it
// for draft tokens, and `Model::VerifyWindow` verifies them exactly as it does MTP's.
//
// WHAT IT IS (docs/dflash2.md sections 1/4/5, compressed):
//   * FEATURES. The target's residual stream ENTERING layers {6,20,34,48,62}, concatenated into
//     one [25600] row per committed position. `Model::AttachDflashFeatureCapture` produces exactly
//     that, as a [rows][25600] bf16 device buffer.
//   * INJECT. `g = rmsnorm_plain(fc(features))` once, then per draft layer `K = rope(k_norm(Wk g))`
//     at the position's own ABSOLUTE index and `V = Wv g` (raw), written into this class's own
//     per-layer ring at `slot = pos % 2048`. There is no Wq in this path at all.
//   * DRAFT. A `block_size`(8)-wide "noise block" `[anchor, <mask> x7]` at positions `n..n+7`
//     (`n` == injected count), embedded from the TARGET's table, run through 5 layers of
//     {plain rmsnorm; dynamic grouped conv (side 0); q/k/v; q_norm/k_norm; rope; NON-causal windowed
//     GQA attention over [visible injected keys] + [the block's own 8 keys]; Wo; conv (side 1);
//     residual; plain rmsnorm; ffn conv (side 0); SwiGLU; conv (side 1); residual}, a final plain
//     rmsnorm, the TARGET's lm_head over all 8 rows, per-row top-16, and a host-side selector
//     lattice walk that chains one token per block position.
//   * BLOCK K/V IS SCRATCH. The block's own 8 K/V rows are passed to the attention kernel directly
//     and are NEVER written into the ring (see `r4dx_dflash_attn_bf16`'s own contract). That is
//     precisely why this ring needs no rollback after a partially-rejected verify round
//     (docs/dflash2.md section 5): nothing speculative was ever stored, and the next round's
//     injection always starts at a strictly higher position.
//   * THE VISIBLE STORE IS `[ValidFrom(), InjectedCount())`, intersected with the sliding window.
//     Injection is monotonic but need not be contiguous: a caller may stop feeding this drafter and
//     resume at a higher absolute position (the server's per-request injection toggle), which
//     leaves a GAP of stale ring bytes below the resume point. `ValidFrom()` is that resume point
//     and is handed to the attention kernel as `store_begin`, so the gap is never read -- the same
//     "self-correcting via position overwrite" argument as above, just with an explicit lower
//     bound instead of an implicit 0. See `InjectFeatures`.
//
// HOST/DEVICE SPLIT. Everything except the selector lattice runs on the GPU. One `DraftRound` call
// issues its whole layer stack asynchronously and then performs exactly ONE device->host copy (a
// single 5 KB staging blob holding `cand[8][16]` int32, `unary[8][16]` fp32 and `gate[8][256]`
// bf16) followed by exactly ONE stream synchronize; the lattice walk (16x256 dot products per
// block position, plus p_min/n_min) then runs on the host in fp32 against the two [vocab][256]
// row-gather codebooks, which are kept host-resident as bf16 (2 x 127 MB) rather than mirrored to
// VRAM -- a device gather would have to come back to the host anyway for the walk, which is
// inherently sequential across block positions.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "container.h"
#include "dflash_draft_weights.h"
#include "quant_linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::model {

// Gathers `n` TARGET embedding rows into `out_dev` ([n, hidden] bf16, device). BOTH id forms are
// handed over so either backing can be used with no extra allocation: `ids_host` for
// `r4dx::kernels::EmbeddingGatherHost` + an H2D upload, `ids_dev` (already uploaded on `stream` by
// the caller) for `r4dx_embedding_gather_bf16` against a VRAM-resident table. The provider must
// leave its result visible to subsequent work on `stream` and must not synchronize.
using DflashEmbeddingProvider = std::function<void(core::Stream& stream, const int32_t* ids_host,
                                                    const int32_t* ids_dev, int64_t n,
                                                    uint16_t* out_dev)>;

// Computes `logits_out` ([T, vocab] fp32, device) from `x_dev` ([T, hidden] bf16, device) using
// the TARGET's lm_head. NOTE: `x_dev` is ALREADY normalised by the drafter's own `output_norm`
// (docs/dflash2.md section 4.2) -- a production provider must therefore call the bare lm_head GEMM
// + widen, NOT `FinalLmHead::Forward`, which would apply the target's own `text.final_norm` a
// second time. Must not synchronize.
using DflashLmHeadProvider = std::function<void(core::Stream& stream, core::Arena& arena,
                                                 const uint16_t* x_dev, float* logits_out,
                                                 int64_t T)>;

// Production providers, backed by the TARGET container -- the drafter deliberately has no
// embedding table and no lm_head of its own (docs/container-format.md's "DFlash2 draft container":
// there is no `dflash.embed_tokens`/`dflash.lm_head` tensor at all).
//
// MakeTargetEmbeddingProvider picks the device-resident gather when the container has a VRAM mirror
// (`Container::EmbedTokensDeviceResident()`) and the host gather + one small H2D otherwise, so both
// paths work without the caller checking.
//
// MakeTargetLmHeadProvider runs the BARE lm_head GEMM + bf16->fp32 widen -- deliberately NOT
// `FinalLmHead::Forward`, which would additionally apply the target's own `text.final_norm` on top
// of the drafter's `dflash.output_norm` (docs/dflash2.md section 4.2 applies exactly one norm).
//
// Both closures capture raw pointers into `container` and must not outlive it. Build them from the
// live Container at the point of use; do not cache one across a Model move or reload.
DflashEmbeddingProvider MakeTargetEmbeddingProvider(const Container& container);
DflashLmHeadProvider MakeTargetLmHeadProvider(const Container& container);

struct DflashDraftOptions {
  std::string container_path;
  Layout layout = Layout::kW4a16;  // which packed layout of this container's linears to load
  // Largest `rows` any InjectFeatures call will be given. 64 matches Model's own max prefill chunk
  // (`Model::max_chunk_`), which is the widest a capture drain can ever hand over.
  int64_t max_inject_rows = 64;
  // Row count of the lm_head the LmHeadProvider will use; 0 means "the container's own
  // vocab_size" (248320 for the real drafter). A test driving a synthetic small-vocab target sets
  // this to that target's vocab so the logits buffer and the top-16 launch are sized correctly.
  int64_t lm_head_vocab = 0;
  // Overrides the container's `mask_token_id` (248070). Same reason as `lm_head_vocab`: the real
  // mask id is out of range for a synthetic small-vocab target, and the Python reference clamps it
  // to `vocab-1` in exactly that case (`dflash2_ref.py::_synthetic_mask_id`). <0 == use the
  // container's own value.
  int64_t mask_token_id_override = -1;
};

// Per-round intermediate capture, for tests/diagnostics ONLY: filling it costs one extra
// device->host copy per entry plus the stream synchronize they imply, so the production driver
// always passes nullptr and pays exactly the one D2H this class's file comment promises.
struct DflashRoundTrace {
  std::vector<int32_t> block_ids;                     // [block_size] the noise block's token ids
  std::vector<std::vector<uint16_t>> x_post_attn;     // [n_layer][block_size*hidden] bf16
  std::vector<std::vector<uint16_t>> x_post_ffn;      // [n_layer][block_size*hidden] bf16
  std::vector<uint16_t> x_final_normed;               // [block_size*hidden] bf16
  std::vector<float> logits;                          // [block_size*vocab] fp32
  std::vector<int32_t> cand;                          // [block_size*16]
  std::vector<float> unary;                           // [block_size*16]
  std::vector<float> gate;                            // [block_size*selector_rank] (widened)
  // score[t-1] is the FULL score[a][b] matrix for block position t (t = 1..block_size-1), row-major
  // `a`-major / `b`-minor, exactly `dflash2_ref.py`'s own `score_matrices[t]`: 1 row for t==1
  // (P == {anchor}), `selector_top_k` rows for t >= 2 (P == cand[t-1]).
  std::vector<std::vector<float>> score;
  // Per-position softmax probability at the argmax (the p_min statistic), and the argmax column.
  std::vector<float> walk_prob;
  std::vector<int32_t> walk_b;
};

struct DflashDraftResult {
  // The drafted token ids, 0..min(k, block_size-1) of them. EMPTY when the n_min gate discarded
  // the whole draft (docs/dflash2.md section 4.3's `n_min`): `walk_len` still reports what the walk
  // itself produced, so a caller can tell "the walk stopped early" from "the walk was thrown away".
  std::vector<int32_t> tokens;
  int64_t walk_len = 0;
  bool discarded_by_n_min = false;
  bool stopped_by_p_min = false;
};

class DflashDraft {
 public:
  static DflashDraft Load(const DflashDraftOptions& opts);

  DflashDraft(DflashDraft&&) = default;
  DflashDraft& operator=(DflashDraft&&) = default;
  DflashDraft(const DflashDraft&) = delete;
  DflashDraft& operator=(const DflashDraft&) = delete;

  const Dflash2Config& Config() const { return cfg_; }
  int64_t BlockSize() const { return cfg_.block_size; }
  int64_t FeatureCols() const {
    return static_cast<int64_t>(cfg_.target_layers.size()) * cfg_.hidden_size;
  }
  int64_t InjectedCount() const { return n_injected_; }
  // First position in the ring whose contents are valid -- see InjectFeatures below. 0 for a
  // drafter that has only ever been fed append-only (the overwhelming common case, and the only
  // case that existed before the server's per-request injection toggle).
  int64_t ValidFrom() const { return valid_from_; }
  int64_t MaskTokenId() const { return mask_token_id_; }
  int64_t LmHeadVocab() const { return lm_head_vocab_; }
  Layout GetLayout() const { return layout_; }

  // Drops every injected position (the ring's bytes are left alone -- nothing can read a slot at or
  // beyond `n_injected_`, and every slot is unconditionally overwritten before it is read again;
  // same self-correcting-via-position-overwrite argument `mtp_head.h` and `Model::Reset` already
  // make for their own caches).
  void Reset() {
    n_injected_ = 0;
    valid_from_ = 0;
  }

  // Encodes `rows` target-feature rows and writes their K/V into every draft layer's ring.
  // features_dev: [rows, FeatureCols()] bf16 DEVICE, contiguous (exactly what
  // `Model::DflashFeatureBuffer()` holds after a RunChunk/VerifyWindow call). Advances
  // `InjectedCount()` to `start_pos + rows`. Enqueues everything on `stream`; does not synchronize.
  //
  // `start_pos` must be >= `InjectedCount()` -- injection is still strictly monotonic, but no
  // longer strictly contiguous:
  //   * `start_pos == InjectedCount()` is the ordinary append, unchanged in every respect;
  //   * `start_pos > InjectedCount()` is a GAP: the caller stopped feeding this drafter for a while
  //     (the server's per-request injection toggle -- a `temperature>0` request never drafts, so it
  //     pays no capture/injection cost) and has now resumed at a higher absolute position. The
  //     skipped positions' ring bytes are stale, so `ValidFrom()` moves up to `start_pos` and
  //     `DraftRound` tells the attention kernel (`store_begin`) to start its visible range there.
  //     Nothing is cleared, because nothing reads them.
  //   * `start_pos < InjectedCount()` still throws. That direction would overwrite a position the
  //     drafter may already have attended to, which is the rollback docs/dflash2.md section 5's
  //     "no rollback needed" argument exists to rule out.
  void InjectFeatures(core::Stream& stream, core::Arena& arena, const uint16_t* features_dev,
                      int64_t rows, int64_t start_pos);

  // One draft round at the current frontier. `anchor_id` is the last committed real token (block
  // position 0); `k` caps how many tokens the walk may emit (0..block_size-1, clamped -- 0 drafts
  // nothing at all, which is how a driver falls back to verifying the anchor alone); `p_min` is the
  // early-stop probability gate (<=0 disables); `n_min` discards the WHOLE draft when the walk
  // produced fewer than that many tokens (<=0 disables). Exactly one D2H + one synchronize when
  // `trace` is null.
  //
  // `device_ms`, when non-null, is filled with this round's hipEvent-measured DEVICE time. The
  // event pair is recorded INSIDE this method for two reasons, both found by measuring rather than
  // assuming (2026-09-20, stage S2):
  //   * the closing record must come before this method's own stream synchronize, which a caller
  //     bracketing the whole call from outside cannot arrange; and
  //   * more importantly it must come before the round's final small async D2H into PINNED host
  //     memory. That copy can be serviced by the SDMA/blit engine instead of the compute queue, and
  //     an event recorded behind it then carries that queue's timestamp: the same instrumentation
  //     with the record one line later reported 0.03-0.27 ms (with an occasional correct sample at
  //     exactly the wall-clock figure) for a round whose wall clock is 6.2 ms and whose weight read
  //     alone is ~1.5 GB -- an impossible 50 TB/s. Recorded before the copy it reads 5.77 ms
  //     against a 6.18 ms wall clock, and InjectFeatures (which has no internal synchronize)
  //     independently agrees device-to-wall within 10%.
  // Costs two hipEventCreate/Destroy per call, so the production driver passes nullptr.
  DflashDraftResult DraftRound(core::Stream& stream, core::Arena& arena, int32_t anchor_id,
                               int64_t k, float p_min, int64_t n_min,
                               const DflashEmbeddingProvider& embed,
                               const DflashLmHeadProvider& lm_head,
                               DflashRoundTrace* trace = nullptr, double* device_ms = nullptr);

  // Diagnostics. `DebugEncodedG` returns the encoder output of the most recent InjectFeatures call
  // ([rows, hidden] bf16); `DebugStoreK`/`DebugStoreV` read `count` consecutive ring slots starting
  // at absolute position `pos_begin` for one layer ([count, kv_heads, head_dim] bf16). Both
  // synchronize; neither is on any hot path.
  std::vector<uint16_t> DebugEncodedG(core::Stream& stream, int64_t rows) const;
  std::vector<uint16_t> DebugStoreK(core::Stream& stream, int64_t layer, int64_t pos_begin,
                                    int64_t count) const;
  std::vector<uint16_t> DebugStoreV(core::Stream& stream, int64_t layer, int64_t pos_begin,
                                    int64_t count) const;

 private:
  DflashDraft() = default;

  struct LayerWeights {
    core::DeviceBuffer<uint16_t> input_layernorm, post_attention_layernorm;  // bf16 [hidden]
    core::DeviceBuffer<uint16_t> q_norm, k_norm;                             // bf16 [head_dim]
    core::DeviceBuffer<uint16_t> attn_conv_base, mlp_conv_base;              // bf16 [2][taps][hidden]
    QuantLinear q_proj, k_proj, v_proj, o_proj, attn_conv_proj;
    QuantLinear gate_proj, up_proj, down_proj, mlp_conv_proj;
  };

  // One layer of the draft block forward. `x` is the running residual ([block_size, hidden] bf16,
  // updated in place).
  void ForwardLayer(core::Stream& stream, core::Arena& arena, int64_t il,
                    DflashRoundTrace* trace);

  // The host-side selector lattice (docs/dflash2.md section 4.3). `cand`/`unary`/`gate` are this
  // round's already-read-back staging values.
  DflashDraftResult SelectorWalk(int32_t anchor_id, int64_t k, float p_min, int64_t n_min,
                                 const int32_t* cand, const float* unary, const uint16_t* gate,
                                 DflashRoundTrace* trace) const;

  const uint16_t* SelectorRow(const std::vector<uint16_t>& book, int64_t id) const {
    return book.data() + id * cfg_.selector_rank;
  }

  Dflash2Config cfg_;
  Layout layout_ = Layout::kW4a16;
  int64_t max_inject_rows_ = 64;
  int64_t lm_head_vocab_ = 0;
  int64_t mask_token_id_ = 0;
  int64_t n_injected_ = 0;
  // Lower bound of the ring's VALID contiguous run (see InjectFeatures / ValidFrom). Always
  // 0 <= valid_from_ <= n_injected_.
  int64_t valid_from_ = 0;
  int64_t last_inject_rows_ = 0;

  // ---- weights -------------------------------------------------------------------------------
  QuantLinear fc_;                                  // [hidden, len(target_layers)*hidden]
  core::DeviceBuffer<uint16_t> enc_output_norm_;    // bf16 [hidden]
  core::DeviceBuffer<uint16_t> output_norm_;        // bf16 [hidden]
  QuantLinear selector_hidden_;                     // [selector_rank, hidden]
  // Host-resident bf16 row-gather codebooks, [vocab][selector_rank] each (127 MB each for the real
  // drafter). Host, not VRAM: the lattice walk is sequential across block positions and runs on the
  // host, so a device copy would be gathered only to be copied straight back.
  std::vector<uint16_t> selector_predecessor_, selector_successor_;
  std::vector<LayerWeights> layers_;

  // ---- KV ring (docs/dflash2.md section 5) ------------------------------------------------------
  // [n_layer][slots][kv_heads][head_dim] bf16 each; `slots` == sliding_window (2048), so a position
  // is visible for exactly as long as the SWA rule says it is and its slot is reused only once it
  // has aged out. ~21 MB each for the real drafter.
  core::DeviceBuffer<uint16_t> k_store_, v_store_;
  int64_t slots_ = 0;

  // ---- injection scratch (persistent: sized for max_inject_rows) ---------------------------------
  core::DeviceBuffer<uint16_t> g_dev_;          // [max_inject_rows, hidden]
  core::DeviceBuffer<uint16_t> ik_dev_, iv_dev_;  // [max_inject_rows, kv_heads*head_dim]
  core::DeviceBuffer<int32_t> ipos_dev_;        // [max_inject_rows]
  core::PinnedBuffer<int32_t> ipos_host_;

  // ---- draft-block scratch (persistent: block_size rows, a few MB total) -------------------------
  core::DeviceBuffer<uint16_t> x_;        // [B, hidden] running residual
  core::DeviceBuffer<uint16_t> h_;        // [B, hidden] norm output
  core::DeviceBuffer<uint16_t> conv_;     // [B, hidden] conv output (never aliases its own input)
  core::DeviceBuffer<uint16_t> proj_;     // [B, hidden] o_proj / down_proj output
  core::DeviceBuffer<uint16_t> xf_;       // [B, hidden] final normed
  core::DeviceBuffer<uint16_t> dyn_;      // [B, 2*taps*n_groups]
  core::DeviceBuffer<uint16_t> q_, k_, v_, attn_;
  core::DeviceBuffer<uint16_t> gate_tmp_, up_tmp_, gate_up_, act_;
  core::DeviceBuffer<float> logits_dev_;  // [B, lm_head_vocab]
  core::DeviceBuffer<int32_t> block_ids_dev_;
  core::PinnedBuffer<int32_t> block_ids_host_;
  core::DeviceBuffer<int32_t> pos_dev_;
  core::PinnedBuffer<int32_t> pos_host_;
  // The ONE staging blob the round reads back: [0,512) unary fp32[B*16], [512,1024) cand int32
  // [B*16], [1024,1024+B*rank*2) gate bf16[B*rank]. Laid out as slices of a single allocation so
  // the three producers (top-16's two outputs, the selector-gate GEMM's output) write straight into
  // it and the readback is a single hipMemcpy.
  core::DeviceBuffer<uint8_t> sel_stage_dev_;
  core::PinnedBuffer<uint8_t> sel_stage_host_;
  int64_t sel_off_unary_ = 0, sel_off_cand_ = 0, sel_off_gate_ = 0, sel_stage_bytes_ = 0;
};

}  // namespace r4dx::model
