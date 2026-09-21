#include "model.h"

#include <algorithm>
#include <chrono>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "dflash_draft_weights.h"
#include "embedding.h"
#include "final_lm_head.h"
#include "gdn_layer.h"
#include "linear.h"
#include "mlp.h"
#include "profile_span.h"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/model/attention/attention_layer.hpp"
#include "r4dx/model/attention/types.hpp"

namespace r4dx::model {

namespace {

// SpanAccumulator/SpanEntry moved to profile_span.h (Milestone 3 profiling pass, docs/r9700.md
// R5/Q3/Q7, 2026-09-20) so GdnLayer::Forward/AttentionLayer::Forward/Mlp::Forward can record
// per-kernel spans into the SAME accumulator this file's DecodeStepProfiled/PrefillProfiled own,
// instead of this file only ever seeing one coarse per-block span. This local helper just converts
// SpanAccumulator::Finish()'s SpanEntry list into Model::ProfileEntry (identical fields).
std::vector<Model::ProfileEntry> ToProfileEntries(std::vector<SpanEntry> raw) {
  std::vector<Model::ProfileEntry> out;
  out.reserve(raw.size());
  for (auto& e : raw) out.push_back({std::move(e.name), e.ms, e.count});
  return out;
}

// DFlash2 target feature capture (docs/dflash2.md, Milestone 5 B1 item 1): called once per layer
// iteration in both RunChunk's and VerifyWindow's layer loops, BEFORE that layer's Gdn/Attn
// Forward call mutates `cur` -- `cur` at this point is exactly the residual stream as it enters
// layer `layer_idx`, which is what target_layers[*] names (0-based layer INPUT index). A no-op
// (single `.empty()` check, no allocation, no copy) when no drafter is attached, i.e. for every
// caller before this pass and every ctest/CLI/server invocation that never calls
// Model::AttachDflashFeatureCapture -- byte-identical behavior and r4dx-owned kernel launch count
// to pre-B1 (this uses hipMemcpy2DAsync, the plain HIP runtime strided D2D copy, not a new r4dx-
// owned kernel, so r4dx::kernels::r4dx_kernel_launch_counter_get() never counts it either way).
void CaptureDflashLayerInput(core::Stream& stream, int64_t layer_idx, const uint16_t* cur,
                              int64_t T, int64_t hidden,
                              const std::vector<int64_t>& target_layers, uint16_t* dst) {
  if (target_layers.empty()) return;
  const auto it = std::find(target_layers.begin(), target_layers.end(), layer_idx);
  if (it == target_layers.end()) return;
  const int64_t col = static_cast<int64_t>(std::distance(target_layers.begin(), it));
  const int64_t num_cols = static_cast<int64_t>(target_layers.size());
  const size_t elem = sizeof(uint16_t);
  R4DX_HIP_CHECK(hipMemcpy2DAsync(
      /*dst=*/dst + col * hidden, /*dpitch=*/static_cast<size_t>(num_cols * hidden) * elem,
      /*src=*/cur, /*spitch=*/static_cast<size_t>(hidden) * elem,
      /*width=*/static_cast<size_t>(hidden) * elem, /*height=*/static_cast<size_t>(T),
      hipMemcpyDeviceToDevice, stream.get()));
}

}  // namespace

void Model::AttachDflashFeatureCapture(std::vector<int64_t> target_layers) {
  if (target_layers.empty()) {
    DetachDflashFeatureCapture();
    return;
  }
  const int64_t num_layers = container_.NumLoadedLayers();
  for (size_t i = 0; i < target_layers.size(); ++i) {
    if (target_layers[i] < 0 || target_layers[i] >= num_layers) {
      throw std::out_of_range("Model::AttachDflashFeatureCapture: target layer index out of range");
    }
    if (i > 0 && target_layers[i] <= target_layers[i - 1]) {
      throw std::invalid_argument(
          "Model::AttachDflashFeatureCapture: target_layers must be sorted, strictly ascending");
    }
  }
  dflash_target_layers_ = std::move(target_layers);
  const int64_t hidden = container_.Config().hidden_size;
  dflash_features_dev_.Resize(static_cast<size_t>(max_chunk_) *
                               static_cast<size_t>(dflash_target_layers_.size()) *
                               static_cast<size_t>(hidden));
  dflash_feature_rows_ = 0;
}

void Model::DetachDflashFeatureCapture() {
  dflash_target_layers_.clear();
  dflash_features_dev_.Resize(0);
  dflash_feature_rows_ = 0;
}

namespace {

// Q13/R14 (docs/r9700.md): "one hipMemGetInfo + per-tensor accounting dump at end of load" --
// four snapshots bracketing Model::Load's three allocation phases (container weights, then
// per-layer GDN state + paged KV cache, then everything else: arena/staging/logits/MTP scratch)
// give a real weights/KV/arena/free breakdown from what the driver actually reports, rather than
// from summing this codebase's own byte-size arithmetic (which is exactly what Q13 flagged as
// insufficient to explain w4a8's "identical 15.75 GiB to w4a16 despite 0.35 GiB smaller weights").
struct VramSnap {
  size_t free_bytes = 0, total_bytes = 0;
  bool ok = false;
};
VramSnap SnapVram() {
  VramSnap s;
  s.ok = (hipMemGetInfo(&s.free_bytes, &s.total_bytes) == hipSuccess);
  return s;
}
double GiB(int64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

}  // namespace

Model Model::Load(const ModelOptions& opts) {
  Model m;
  const VramSnap vram0 = SnapVram();  // before any of this Load() call's own allocations
  m.container_ = Container::Load(opts.container_path, opts.layout, opts.layout, opts.layer_limit,
                                  opts.mtp_head_layout.value_or(opts.layout),
                                  opts.embed_device_resident);
  const VramSnap vram1 = SnapVram();  // after container weights are fully resident
  const ModelConfig& cfg = m.container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = m.container_.NumLoadedLayers();

  if (opts.mtp_draft_k > 0 && !m.container_.HasMtp()) {
    throw std::runtime_error(
        "Model::Load: mtp_draft_k > 0 but the container has no mtp.* weights (convert with "
        "--mtp on)");
  }
  if (opts.dflash_draft_k < 0) {
    throw std::runtime_error("Model::Load: dflash_draft_k must be >= 0");
  }
  if (!opts.dflash_container.empty()) {
    if (opts.dflash_draft_k <= 0) {
      throw std::runtime_error(
          "Model::Load: dflash_container is set but dflash_draft_k <= 0 (nothing would ever be "
          "drafted -- Model::VerifyWindow's own window would never exceed 1 row)");
    }
    if (opts.mtp_draft_k > 0) {
      throw std::runtime_error(
          "Model::Load: dflash_container cannot be combined with mtp_draft_k > 0 -- DFlash2 and "
          "MTP are mutually exclusive self-speculation families (docs/dflash2.md); the CLI/server "
          "already reject --dflash with --mtp > 0, this is Model::Load's own re-check so no caller "
          "can bypass that by constructing ModelOptions directly)");
    }
  }
  m.mtp_draft_k_ = opts.mtp_draft_k;
  m.dflash_draft_k_ = opts.dflash_draft_k;
  m.mtp_draft_reduced_vocab_ = opts.mtp_draft_reduced_vocab;
  // GDN's per-sequence window bank (gdn_state.h's file comment) must be sized for the largest
  // speculative-verify window this Model will ever run: the widest draft either speculation family
  // can produce, plus the seed/anchor token. ==1 (window index 0 only) when neither is enabled --
  // every decode call then degenerates exactly to this class's pre-speculation behavior, and a
  // Model with only MTP enabled sizes identically to before dflash_draft_k existed.
  m.draft_window_ = 1 + std::max(opts.mtp_draft_k, opts.dflash_draft_k);
  const int64_t max_decode_window = m.draft_window_;

  m.max_chunk_ = 64;
  m.embed_staging_ = core::PinnedBuffer<uint16_t>(static_cast<size_t>(m.max_chunk_ * hidden));
  m.embed_ids_host_ = core::PinnedBuffer<int32_t>(static_cast<size_t>(m.max_chunk_));
  m.embed_ids_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(m.max_chunk_));
  m.buf_a_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.max_chunk_ * hidden));
  m.buf_b_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.max_chunk_ * hidden));
  m.buf_normed_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(m.max_chunk_ * hidden));
  // R2/P2 (docs/r9700.md): buf_normed_'s fused quant-epilogue companion (model.h's doc comment) --
  // 2 bytes/element covers the widest epilogue format (f16); fp8/int8 use the same allocation's
  // first half.
  m.buf_normed_pre_ = core::DeviceBuffer<uint8_t>(static_cast<size_t>(m.max_chunk_ * hidden * 2));
  m.buf_normed_pre_scale_ = core::DeviceBuffer<float>(static_cast<size_t>(m.max_chunk_));
  // R2/P2 (docs/r9700.md): the original correctness issue (an `r4dx::core::Arena::Alloc` gap that
  // never rounded an allocation's END up to a 16-byte boundary, silently misaligning every
  // default-aligned buffer allocated right after a fused epilogue's scale scratch) was root-caused
  // and fixed in `arena.hpp` (docs/status.md's "R2/P2 fused activation-quant epilogues:
  // root-caused and enabled for w4a8/mxfp4" section). EpilogueForLayout (linear.cpp) now returns
  // `r4dx_epilogue_int8_fraga8`/`r4dx_epilogue_fp8_e4m3_row` for w4a8/mxfp4 (verified byte-identical
  // end to end against the fusion-disabled baseline, `tools/validate_fusion.ps1`) and still
  // `r4dx_epilogue_none` for w4a16/bf16 (w4a16's own separate, understood wall-clock regression --
  // see linear.cpp's own doc comment -- and bf16 never quantizes its activation input at all).
  m.body_epilogue_ = EpilogueForLayout(opts.layout);
  m.logits_dev_ = core::DeviceBuffer<float>(static_cast<size_t>(cfg.vocab_size));
  m.argmax_dev_ = core::DeviceBuffer<int32_t>(1);
  m.attn_positions_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(m.max_chunk_));
  m.attn_seqused_k_ = core::DeviceBuffer<int32_t>(1);
  // Per-layer activation scratch (rmsnorm output, gate_up, GDN conv/kkt/chunk-scan buffers,
  // activation-quant scratch): a few MB at T<=64 (see linear.h/gdn_layer.cpp's own buffer sizes).
  // 96MB gives headroom without materially affecting the ~15-35GB the weights themselves occupy.
  m.arena_.Reserve(96ull * 1024 * 1024);
  const VramSnap vram2 = SnapVram();  // after activation scratch (buf_a_/b_/logits/arena/etc.)

  const auto adims = core::r4d::GetAttnDims();  // head_dim=256, gqa=6, block_size=16

  m.gdn_states_.resize(static_cast<size_t>(num_layers));
  m.kv_caches_.resize(static_cast<size_t>(num_layers));
  for (int64_t i = 0; i < num_layers; ++i) {
    if (cfg.IsGdnLayer(i)) {
      m.gdn_states_[static_cast<size_t>(i)].emplace(
          /*max_seqs=*/1, /*H=*/cfg.linear_num_value_heads, /*V=*/cfg.linear_value_head_dim,
          /*K=*/cfg.linear_key_head_dim, cfg.ConvDim(), cfg.linear_conv_kernel_dim,
          max_decode_window);
      m.gdn_states_[static_cast<size_t>(i)]->ZeroAll(m.stream_);
    } else {
      m.kv_caches_[static_cast<size_t>(i)].emplace(
          static_cast<int>(cfg.num_key_value_heads), static_cast<int>(cfg.head_dim),
          adims.block_size, static_cast<int>(opts.max_ctx));
    }
  }

  if (opts.mtp_draft_k > 0) {
    m.mtp_.emplace(cfg, m.container_.Mtp(), opts.mtp_draft_k, opts.max_ctx);
    m.mtp_seed_hidden_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(hidden));
  }
  // Verify scratch + the GDN acceptance-count thread are shared by BOTH speculation families, so
  // they are sized off draft_window_ rather than off mtp_draft_k alone -- a DFlash2-only Model
  // (mtp_draft_k==0, dflash_draft_k>0) has no mtp.* weights anywhere but still verifies through
  // Model::VerifyWindow, which needs exactly these.
  if (m.draft_window_ > 1) {
    m.mtp_num_accepted_dev_ = core::DeviceBuffer<int32_t>(1);
    m.verify_logits_dev_ =
        core::DeviceBuffer<float>(static_cast<size_t>(m.draft_window_ * cfg.vocab_size));
    m.verify_argmax_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(m.draft_window_));
  }

  // Stage S3 (docs/dflash2.md section 7 item 1): load the DFlash2 drafter, if requested, as its own
  // phase so its VRAM (weights + KV ring + draft-block scratch, all allocated inside
  // DflashDraft::Load) can be reported as its own breakdown line rather than folded silently into
  // the generic "kv+gdn_state" bucket above.
  const VramSnap vram_dflash0 = SnapVram();
  bool has_dflash = false;
  if (!opts.dflash_container.empty()) {
    // The container's own packed layout lives in its metadata (docs/container-format.md), NOT in
    // ModelOptions -- a DFlash2 draft container is a completely separate tensor set from the
    // target body and may be converted to a different layout independently (model.h's own doc
    // comment on ModelOptions::dflash_container). Peek it with a throwaway DflashDraftWeights::Open
    // (a second mmap of the same small file, freed immediately) rather than adding a layout
    // parameter a caller would have to keep in sync with the file by hand.
    const DflashDraftWeights peek = DflashDraftWeights::Open(opts.dflash_container);
    const Dflash2Config& dcfg = peek.Config();
    if (dcfg.layout.empty()) {
      throw std::runtime_error("Model::Load: dflash container '" + opts.dflash_container +
                               "' has no 'layout' field in its __metadata__.dflash2 block");
    }
    const Layout dflash_layout = LayoutFromName(dcfg.layout);  // throws on an unrecognized name --
                                                                // this IS the "layout validated" step
    if (opts.dflash_draft_k > dcfg.block_size - 1) {
      throw std::runtime_error(
          "Model::Load: dflash_draft_k (" + std::to_string(opts.dflash_draft_k) +
          ") exceeds this container's own block_size-1 (" + std::to_string(dcfg.block_size - 1) +
          ") -- the drafter can never produce more than block_size-1 tokens per round");
    }
    // Review finding (2026-09-21): the drafter's fc encoder input is sized as
    // target_layers.size() * dcfg.hidden_size (DflashDraft::Load), while Model's own capture buffer
    // is sized as target_layers.size() * cfg.hidden_size (DflashFeatureCols(), AttachDflashFeatureCapture
    // below). A mismatched pair silently reads/writes past dflash_features_dev_ instead of throwing a
    // clear error -- catch it here, at load time, naming both values.
    if (dcfg.hidden_size != cfg.hidden_size) {
      throw std::runtime_error(
          "Model::Load: dflash container '" + opts.dflash_container + "' hidden_size (" +
          std::to_string(dcfg.hidden_size) + ") does not match the target model's hidden_size (" +
          std::to_string(cfg.hidden_size) + ") -- the drafter's feature encoder input width would not "
          "match the target's captured residual-stream width");
    }
    DflashDraftOptions dopts;
    dopts.container_path = opts.dflash_container;
    dopts.layout = dflash_layout;
    dopts.lm_head_vocab = cfg.vocab_size;  // the TARGET's own vocab (docs/dflash2.md: the drafter
                                            // has no lm_head of its own, MakeTargetLmHeadProvider
                                            // always runs the target's real head)
    m.dflash_ = DflashDraft::Load(dopts);
    // Auto-feed this Model-owned drafter from every RunChunk call (prefill chunk + plain decode
    // step) -- see RunChunk's own comment for exactly where, and model.h's dflash_ field comment
    // for why this is a direct member call rather than the external dflash_observer_ mechanism.
    m.AttachDflashFeatureCapture(m.dflash_->Config().target_layers);
    std::cerr << "[r4dx::model::Model] DFlash2 drafter loaded: " << opts.dflash_container
              << " (layout=" << LayoutName(dflash_layout)
              << ", block_size=" << dcfg.block_size << ", target_layers=[";
    for (size_t i = 0; i < dcfg.target_layers.size(); ++i) {
      std::cerr << (i ? "," : "") << dcfg.target_layers[i];
    }
    std::cerr << "], selector_rank=" << dcfg.selector_rank
              << ", sliding_window=" << dcfg.attention.sliding_window << ")\n";
    has_dflash = true;
  }
  const VramSnap vram_dflash1 = SnapVram();

  m.stream_.Synchronize();

  // Q13/R14 VRAM breakdown (docs/r9700.md): weights (Container::Load's own footprint) / KV+GDN
  // state (the per-layer paged-cache/recurrent-state loop above) / arena+activation scratch
  // (buf_a_/b_/logits_dev_/arena_/MTP scratch) / free, all from what the driver actually reports
  // at each phase boundary -- not from this codebase's own tensor-shape arithmetic (docs/r9700.md
  // Q13's own point: that arithmetic could not explain why w4a8's 0.35 GiB-smaller weights still
  // measured the identical 15.75 GiB as w4a16). Printed unconditionally (one line, stderr) rather
  // than gated behind a flag -- cheap, and exactly the diagnostic R14 asks every load to carry.
  // Stage S3: the dflash_drafter figure below (weights+KV-ring+draft-block-scratch, all allocated by
  // DflashDraft::Load above) is ALSO already included in kv_b's total (it was allocated between the
  // vram2 and vram3 snapshots) -- broken out here as its own line per docs/dflash2.md section 7
  // item 1's own ask, not double-counted in the reported total.
  const VramSnap vram3 = SnapVram();  // after KV/GDN state + MTP scratch + dflash: final steady state
  if (vram0.ok && vram1.ok && vram2.ok && vram3.ok) {
    const int64_t weights_b = static_cast<int64_t>(vram0.free_bytes) - static_cast<int64_t>(vram1.free_bytes);
    const int64_t arena_b = static_cast<int64_t>(vram1.free_bytes) - static_cast<int64_t>(vram2.free_bytes);
    const int64_t kv_b = static_cast<int64_t>(vram2.free_bytes) - static_cast<int64_t>(vram3.free_bytes);
    std::cerr << "[r4dx::model::Model] VRAM breakdown (layout=" << LayoutName(opts.layout)
              << "): weights=" << GiB(weights_b) << " GiB, kv+gdn_state=" << GiB(kv_b)
              << " GiB, arena+scratch=" << GiB(arena_b) << " GiB, free="
              << GiB(static_cast<int64_t>(vram3.free_bytes)) << " GiB (of "
              << GiB(static_cast<int64_t>(vram3.total_bytes)) << " GiB total)\n";
    if (has_dflash && vram_dflash0.ok && vram_dflash1.ok) {
      const int64_t dflash_b =
          static_cast<int64_t>(vram_dflash0.free_bytes) - static_cast<int64_t>(vram_dflash1.free_bytes);
      std::cerr << "[r4dx::model::Model] DFlash2 drafter VRAM: " << GiB(dflash_b)
                << " GiB (weights+kv_ring+draft_scratch; already included in kv+gdn_state above)\n";
    }
  }

  return m;
}

void Model::Reset() {
  // GDN's recurrent/conv state is read (not just overwritten) via has_init -- unlike the KV
  // caches (see model.h's Reset() doc comment), it must be explicitly zeroed, exactly like Load()
  // does the first time.
  for (auto& gs : gdn_states_) {
    if (gs) gs->ZeroAll(stream_);
  }
  // Block until the zeroing above has actually landed before returning -- Reset() is meant to be a
  // synchronous "the model is fresh now" call (Engine measures and logs its own latency around
  // this, docs/server.md), and the zeroing is tiny (a few MB at most), so this sync costs
  // microseconds, not the milliseconds a real work item would.
  stream_.Synchronize();

  pos_ = 0;
  started_ = false;

  // MTP head state (model.h's own field comments): mtp_seed_hidden_'s bytes are stale but harmless
  // -- mtp_seed_valid_==false means nothing will read them until the next RunChunk/
  // DecodeStepMtpGreedy call overwrites it first. mtp_num_accepted_dev_ is the same story via
  // mtp_num_accepted_valid_. MtpHead's own internal KV cache needs no explicit reset either, for
  // the identical self-correcting-via-position-overwrite reason kv_caches_ doesn't (see above).
  mtp_seed_valid_ = false;
  mtp_num_accepted_valid_ = false;
  mtp_last_hidden_ = nullptr;

  // DFlash2 target feature capture (review finding, 2026-09-20): dflash_features_dev_'s bytes are
  // now stale (the previous sequence's residual stream) but harmless for the identical reason
  // mtp_seed_hidden_'s bytes above are -- dflash_feature_rows_==0 means DflashFeatureRows() reports
  // "nothing captured yet" until the next RunChunk/VerifyWindow overwrites both the buffer and this
  // count together, so a caller cannot observe a stale row count pointing at stale data.
  dflash_feature_rows_ = 0;
  // DFlash2's own KV ring (stage S3): drop every injected position -- the ring's bytes are left
  // alone (same self-correcting-via-position-overwrite argument DflashDraft::Reset()'s own comment
  // makes), so this is a cheap host-only counter reset, not a device zero/sync. This also clears
  // any cold-ring gap (DflashDraft::ValidFrom() back to 0). Deliberately does NOT touch
  // dflash_injection_enabled_: that is a caller policy about the NEXT request, not state belonging
  // to the sequence being dropped (model.h's SetDflashInjectionEnabled).
  if (dflash_.has_value()) dflash_->Reset();
}

std::vector<float> Model::RunChunk(const std::vector<int32_t>& token_ids, bool is_prefill_path,
                                    bool want_logits, int32_t* greedy_token_out) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  if (T < 1 || T > max_chunk_) {
    throw std::runtime_error("Model::RunChunk: token_ids.size() must be in [1, " +
                              std::to_string(max_chunk_) + "]");
  }
  const ModelConfig& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = container_.NumLoadedLayers();
  const bool has_init = started_;
  // DFlash2 target feature capture + drafter injection, as ONE decision (model.h's
  // SetDflashInjectionEnabled): a chunk either captures and injects, or does neither. Capturing
  // without injecting would pay the whole per-layer strided D2D cost for rows nothing ever reads,
  // which is precisely the tax the toggle exists to remove.
  const bool dflash_capture_active = dflash_injection_enabled_ && !dflash_target_layers_.empty();

  // Device-resident draft loop (docs/mtp.md "device-resident draft loop"): when the container's
  // text.embed_tokens has a VRAM mirror (Container::EmbedTokensDeviceResident()), gather straight
  // from it via a device kernel instead of a host memcpy + async H2D of the whole [T,hidden]
  // staging buffer -- only a small [T] int32 id array needs to cross the H2D boundary. Falls back
  // to the original host-gather path when the container was loaded with embed_device_resident=
  // false or the free-VRAM heuristic decided the mirror would not fit (Container::Load's comment).
  if (container_.EmbedTokensDeviceResident()) {
    std::copy(token_ids.begin(), token_ids.end(), embed_ids_host_.begin());
    embed_ids_dev_.CopyFromHostAsync(embed_ids_host_.data(), static_cast<size_t>(T), stream_);
    EmbedTokensDeviceGather(stream_, container_.EmbedTokensDevice(), hidden, embed_ids_dev_.data(),
                             T, cfg.vocab_size, buf_a_.data());
  } else {
    EmbedTokens(stream_, container_.EmbedTokensHost(), cfg.vocab_size, hidden, token_ids,
                embed_staging_, buf_a_);
  }

  // Full-attention layers' positions/slot_mapping (== pos_+t, see model.h) and seqused_k (==
  // pos_+T) are identical for every attention layer in this chunk -- upload them once here rather
  // than once per attention layer. Plain (blocking) hipMemcpy is safe at this specific point only:
  // the previous RunChunk call (if any) ended with stream_.Synchronize() before returning its
  // logits, so the device is guaranteed idle here and nothing can race this write -- see
  // AttentionLayer::Forward's doc comment, which requires exactly that (a non-arena, ordered
  // upload) for these two buffers.
  std::vector<int32_t> positions_h(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) positions_h[static_cast<size_t>(t)] = static_cast<int32_t>(pos_ + t);
  attn_positions_.CopyFromHost(positions_h.data(), positions_h.size());
  const int32_t seqused_k_h = static_cast<int32_t>(pos_ + T);
  attn_seqused_k_.CopyFromHost(&seqused_k_h, 1);

  uint16_t* cur = buf_a_.data();
  uint16_t* other = buf_b_.data();
  // R3 fusion (docs/r9700.md): null for layer 0 (no previous Mlp to have fused its rmsnorm), then
  // set to buf_normed_.data() by every layer's own Mlp::Forward call below for i+1 to consume.
  const uint16_t* normed_in = nullptr;
  // R2/P2 (docs/r9700.md): companion to normed_in above -- r4dx_epilogue_none for layer 0 (nothing
  // to reuse yet), then body_epilogue_ once a Mlp::Forward call below has fused an epilogue into
  // buf_normed_pre_ for layer i+1 to consume (mirrors normed_in's own carry-forward exactly).
  int normed_in_epilogue = r4dx_epilogue_none;

  for (int64_t i = 0; i < num_layers; ++i) {
    const LayerWeights& lw = container_.Layer(i);
    const bool has_next_layer = (i + 1 < num_layers);
    // Every GDN/attention layer in this architecture is immediately followed by its own Mlp, so
    // the sub-block's fused epilogue always targets THIS layer's post_attention_layernorm.
    const uint16_t* mlp_norm_weight = lw.post_attention_layernorm.data();

    // DFlash2 target feature capture (see CaptureDflashLayerInput's comment above): `cur` right
    // here, before this iteration's Gdn/Attn Forward call, is exactly the residual stream entering
    // layer `i` -- a no-op when no drafter is attached, and skipped entirely while a caller has
    // turned injection off (model.h's SetDflashInjectionEnabled).
    if (dflash_capture_active) {
      CaptureDflashLayerInput(stream_, i, cur, T, hidden, dflash_target_layers_,
                               dflash_features_dev_.data());
    }

    if (cfg.IsGdnLayer(i)) {
      GdnLayer layer(cfg, lw.input_layernorm, *lw.gdn);
      GdnLayerParams p;
      p.slot = gdn_states_[static_cast<size_t>(i)]->SlotForSeq(0);
      p.is_prefill = is_prefill_path;
      p.has_init = has_init;
      // Plain-decode/MTP-desync guard (review finding, 2026-09-19): a decode step run through this
      // method (DecodeStep/DecodeStepGreedy, is_prefill_path==false) on an MTP-enabled Model must
      // seed from wherever the LAST verify round (DecodeStepMtpGreedy) actually left the window
      // state -- nullptr would silently reseed from window index 0 regardless of that round's own
      // num_accepted, desyncing GDN state from the real committed sequence. VerifyWindow/
      // DecodeStepMtpGreedy never call RunChunk at all, so this is the only place plain decode
      // needs to thread it. See the `mtp_` block below RunChunk's main loop, which sets
      // mtp_num_accepted_dev_=1/mtp_num_accepted_valid_=true right after this call commits, so the
      // NEXT plain decode step (or the next MTP round) seeds correctly regardless of which kind of
      // step follows.
      p.num_accepted = (!is_prefill_path && draft_window_ > 1 && mtp_num_accepted_valid_)
                            ? mtp_num_accepted_dev_.data()
                            : nullptr;
      layer.Forward(stream_, arena_, *gdn_states_[static_cast<size_t>(i)], gdn_control_, cur, cur,
                    T, p, normed_in, mlp_norm_weight, buf_normed_.data(), /*prof=*/nullptr,
                    normed_in_epilogue, buf_normed_pre_.data(), buf_normed_pre_scale_.data(),
                    body_epilogue_, buf_normed_pre_.data(), buf_normed_pre_scale_.data());
    } else {
      attention::AttnConfig acfg;
      acfg.hidden = static_cast<int>(hidden);
      acfg.num_heads = static_cast<int>(cfg.num_attention_heads);
      acfg.kv_heads = static_cast<int>(cfg.num_key_value_heads);
      acfg.head_dim = static_cast<int>(cfg.head_dim);
      acfg.rotary_dim = static_cast<int>(cfg.RotaryDim());
      acfg.rope_theta = static_cast<float>(cfg.rope_theta);
      acfg.rms_eps = static_cast<float>(cfg.rms_norm_eps);
      attention::AttentionLayer layer(acfg);

      attention::AttnWeights aw;
      aw.input_layernorm = lw.input_layernorm.data();
      aw.qg = &lw.attn->qg;
      aw.k = &lw.attn->k;
      aw.v = &lw.attn->v;
      aw.o = &lw.attn->o;
      aw.q_norm = lw.attn->q_norm.data();
      aw.k_norm = lw.attn->k_norm.data();
      aw.k_descale = lw.attn->k_descale.data();
      aw.v_descale = lw.attn->v_descale.data();

      layer.Forward(arena_, cur, other, aw, *kv_caches_[static_cast<size_t>(i)],
                    static_cast<int>(T), static_cast<int>(pos_), attn_positions_.data(),
                    attn_seqused_k_.data(), stream_.get(), normed_in, mlp_norm_weight,
                    buf_normed_.data(), /*prof=*/nullptr, normed_in_epilogue,
                    buf_normed_pre_.data(), buf_normed_pre_scale_.data(), body_epilogue_,
                    buf_normed_pre_.data(), buf_normed_pre_scale_.data());
      std::swap(cur, other);
    }

    Mlp mlp(cfg, lw.post_attention_layernorm, lw.mlp);
    // R3 fusion: x_normed_in is always buf_normed_ (this layer's Gdn/Attn just fused its own
    // rmsnorm epilogue into it above); next_norm_weight is null for the last layer (Mlp falls
    // back to a plain residual add, and FinalLmHead below applies its own final_norm separately).
    // R2/P2: normed_in_epilogue/buf_normed_pre_* is that SAME fused output's quant epilogue
    // companion (always body_epilogue_ by this point, since the Gdn/Attn call above always
    // requests it); next_epilogue/buf_normed_pre_* (output side) requests this Mlp's own
    // residual+rmsnorm epilogue ALSO emit x_normed_out's fused quant epilogue, for layer i+1's
    // Gdn/Attn sub-block to consume, unless this is the last layer.
    mlp.Forward(stream_, arena_, cur, cur, T, buf_normed_.data(),
                has_next_layer ? container_.Layer(i + 1).input_layernorm.data() : nullptr,
                has_next_layer ? buf_normed_.data() : nullptr, /*prof=*/nullptr,
                normed_in_epilogue, buf_normed_pre_.data(), buf_normed_pre_scale_.data(),
                has_next_layer ? body_epilogue_ : r4dx_epilogue_none,
                has_next_layer ? buf_normed_pre_.data() : nullptr,
                has_next_layer ? buf_normed_pre_scale_.data() : nullptr);
    normed_in = has_next_layer ? buf_normed_.data() : nullptr;
    normed_in_epilogue = has_next_layer ? body_epilogue_ : r4dx_epilogue_none;

    arena_.Reset();
  }
  // Rows 0..T-1 of dflash_features_dev_ are this chunk's captured features -- but ONLY if this
  // chunk actually captured. When injection is disabled the buffer still holds whatever the last
  // capturing call left there, so report 0 rows rather than let a caller mistake stale rows for
  // this chunk's (same reasoning as Reset()'s own zeroing of this counter).
  if (!dflash_target_layers_.empty()) dflash_feature_rows_ = dflash_capture_active ? T : 0;

  // ---- MTP lockstep KV priming (docs/mtp.md, mtp_head.h's PrimeKv comment) ----------------------
  // Extends MtpHead's own KV cache by exactly the real positions THIS call just made knowable --
  // unconditionally (not gated by want_logits: a prefill chunk that discards its own logits still
  // reveals real "next tokens" for MTP's boundary/within-chunk (h_i, t_{i+1}) pairs). No-op at
  // exactly zero extra cost when mtp_ is unset (the overwhelming common case).
  if (mtp_) {
    if (mtp_seed_valid_) {
      // The ONE position left dangling by the previous RunChunk/DecodeStepMtpGreedy call: h_i =
      // that call's own last-row hidden state (mtp_seed_hidden_), t_{i+1} = THIS call's own first
      // input token.
      // host_staging_offset=0: this is the FIRST of up to two PrimeKv calls in this RunChunk
      // invocation (see mtp_head.h's PrimeKv doc comment for why the two calls need disjoint
      // offsets into MtpHead's own pinned host scratch).
      mtp_->PrimeKv(stream_, arena_, cfg, container_.Mtp(), pos_ - 1, mtp_seed_hidden_.data(),
                    {token_ids[0]}, container_.EmbedTokensHost(), cfg.vocab_size,
                    /*host_staging_offset=*/0);
    }
    if (T > 1) {
      // Within-chunk pairs: h_i = this chunk's own rows 0..T-2 (`cur`, unmodified since the layer
      // loop above finished), t_{i+1} = this chunk's own token_ids[1..T-1]. Row T-1 is left
      // dangling for the NEXT call, exactly like the boundary case above.
      // host_staging_offset=1: the SECOND of up to two calls this invocation -- the boundary call
      // above (n=1) used offset 0, so offset 1 keeps this call's host source disjoint from it while
      // it may still be in flight. 1 + (T-1) == T <= max_chunk_ == MtpHead::kMaxPrime always.
      const std::vector<int32_t> next_toks(token_ids.begin() + 1, token_ids.end());
      mtp_->PrimeKv(stream_, arena_, cfg, container_.Mtp(), pos_, cur, next_toks,
                    container_.EmbedTokensHost(), cfg.vocab_size, /*host_staging_offset=*/1);
    }
    R4DX_HIP_CHECK(hipMemcpyAsync(mtp_seed_hidden_.data(), cur + (T - 1) * hidden,
                                   static_cast<size_t>(hidden) * sizeof(uint16_t),
                                   hipMemcpyDeviceToDevice, stream_.get()));
    mtp_seed_valid_ = true;
    arena_.Reset();
  }

  // Prefill discards every chunk's logits except the last (Prefill() below only keeps the final
  // RunChunk's return value) -- skip final_norm + the full-vocab lm_head GEMM + the bf16->fp32
  // widen + the 1MB D2H copy for a chunk whose logits nobody reads. On a long prompt this is the
  // difference between one full-vocab GEMM (~2.5GB bf16 lm_head weight read) and one per 64-token
  // chunk. The stream sync below still runs unconditionally (not folded into `want_logits`): the
  // NEXT RunChunk call's top-of-function attn_positions_/attn_seqused_k_ upload is a plain
  // (blocking) hipMemcpy that is only race-free because it assumes the device is fully idle when
  // it runs (see that comment) -- skipping the sync here would silently break that invariant for
  // every chunk but the last.
  if (want_logits) {
    FinalLmHead head(cfg, container_.FinalNorm(), container_.LmHead());
    head.Forward(stream_, arena_, cur + (T - 1) * hidden, logits_dev_.data(), /*T=*/1);
    // Greedy path (host-overhead pass, 2026-09-19): argmax logits_dev_ ON DEVICE while it's still
    // hot, so the only D2H this call ever does is 4 bytes instead of vocab*4 -- see
    // r4dx_argmax_f32 (src/kernels) and DecodeStepGreedy's own comment (model.h).
    if (greedy_token_out != nullptr) {
      r4dx_argmax_f32(reinterpret_cast<int64_t>(logits_dev_.data()),
                       reinterpret_cast<int64_t>(argmax_dev_.data()), cfg.vocab_size,
                       reinterpret_cast<int64_t>(stream_.get()));
    }
    arena_.Reset();
  }

  // stream_ is created with hipStreamNonBlocking (r4dx::core::Stream's default), which by design
  // does NOT implicitly synchronize against the legacy/null stream a plain (no-stream-argument)
  // hipMemcpy uses. logits_dev_.CopyToHost()/argmax_dev_.CopyToHost() below (and the next call's
  // attn_positions_/attn_seqused_k_ upload) is exactly that plain, synchronous D2H/H2D hipMemcpy --
  // it must be preceded by an explicit stream_.Synchronize() (not followed by one) or it can start
  // copying before this chunk's kernels (queued on stream_ above) have actually finished, silently
  // reading/overwriting stale/in-flight data. Same class of bug as the one fixed in
  // gdn_layer.cpp's UploadArray (see that file's comment) -- any plain hipMemcpy/
  // DeviceBuffer::CopyToHost/CopyFromHost call against a hipStreamNonBlocking stream's output
  // needs an explicit wait first, never an implicit one.
  stream_.Synchronize();

  std::vector<float> logits;
  if (want_logits) {
    if (greedy_token_out != nullptr) {
      argmax_dev_.CopyToHost(greedy_token_out, 1);
    } else {
      logits.resize(static_cast<size_t>(cfg.vocab_size));
      logits_dev_.CopyToHost(logits.data(), logits.size());
    }
  }

  // Plain-decode/MTP-desync guard (review finding, 2026-09-19; see the p.num_accepted comment
  // above in the GDN branch for the read side): a plain decode step through this method always
  // "accepts" exactly the one token it was given, so GDN's NEXT call (whether another plain decode
  // step or the next MTP verify round) must seed from window index 0 of THIS step's own commit --
  // num_accepted=1 encodes exactly that (gdn_state.h's file comment: sidx[naccept-1] with naccept=1
  // reads window index 0, the slot this step just wrote). Safe here (device is idle, the sync
  // above already waited for every kernel that read the PREVIOUS value of
  // mtp_num_accepted_dev_). Prefill never touches this: mtp_num_accepted_valid_ is reset by
  // Prefill() itself for the first verify round after a fresh prefill.
  if (!is_prefill_path && draft_window_ > 1) {
    const int32_t one = 1;
    mtp_num_accepted_dev_.CopyFromHost(&one, 1);
    mtp_num_accepted_valid_ = true;
  }

  // DFlash2 per-chunk capture drain (model.h's SetDflashCaptureObserver): invoked once per
  // RunChunk -- prefill chunk AND plain decode step alike -- after this call's captured rows are
  // complete on the device (the stream_.Synchronize() above already waited for them) and before any
  // later call can overwrite dflash_features_dev_ at row 0. `pos_` is still this chunk's own start
  // position here, which is exactly the absolute position of captured row 0.
  if (dflash_observer_ && dflash_capture_active) {
    dflash_observer_(dflash_features_dev_.data(), T, pos_);
  }
  // Stage S3: when THIS Model owns its own drafter (ModelOptions::dflash_container), feed it
  // directly -- a plain member call, not the external dflash_observer_ mechanism above (model.h's
  // dflash_ field comment explains why: dflash_observer_ closures are for a caller-owned drafter
  // object with its own, separately-managed lifetime; dflash_ is a member of this very Model, so a
  // direct call needs no captured pointer that a future Model move could invalidate). `pos_` is
  // still this chunk's own start position here, exactly InjectFeatures' own `start_pos` contract
  // (monotonic, >= DflashDraft::InjectedCount()). It EQUALS InjectedCount() whenever injection has
  // been on continuously, because Prefill/DecodeStep* never skip a chunk's worth of positions and
  // this call always injects every one of them; it is strictly GREATER exactly once after a caller
  // re-enables injection (model.h's SetDflashInjectionEnabled), which InjectFeatures turns into a
  // cold-ring gap. Either way `InjectedCount() == pos_` holds again on return.
  if (dflash_.has_value() && dflash_capture_active) {
    dflash_->InjectFeatures(stream_, arena_, dflash_features_dev_.data(), T, pos_);
    // MUST reset arena_ before returning: the per-layer loop above already left arena_ clean (its
    // own last iteration calls arena_.Reset() unconditionally), and every other terminal user of
    // arena_ in this codebase resets it when done -- InjectFeatures' own scratch allocations (the
    // fc encoder GEMM etc.) are the ONE exception, because leaving them un-reset means the NEXT
    // RunChunk call (the next prefill chunk, or the next plain decode step) starts ITS OWN layer
    // loop from a non-zero, dirty arena offset instead of the clean one every other code path
    // (including the `--mtp 0` baseline this must stay byte-identical to) always starts from.
    // FOUND BY validate_dflash.ps1 (docs/dflash2.md section 7 item 4): a >64-token prompt (>1
    // prefill chunk) diverged from `--mtp 0` on EVERY target layout including w4a16 (which has no
    // known batched-verify reduction-order sensitivity, ruling that mechanism out), while a
    // <=64-token (single-chunk) prompt matched exactly -- the single-chunk case only "worked" by
    // accident, because DecodeStepDflashGreedy's own first arena_.Reset() (after its DraftRound
    // call) wipes the leftover before anything reads it, which nothing does between two prefill
    // chunks of the SAME Prefill() call.
    arena_.Reset();
    // Review finding (2026-09-21, blocker-adjacent major): InjectFeatures enqueues real kernels
    // (the fc encoder GEMM, per-layer k/v projections, norms, rope, and the ring's D2D copies) on
    // stream_ AFTER the stream_.Synchronize() above (this function's one and only sync point until
    // now) -- so RunChunk was returning with the device NOT idle whenever a drafter is attached.
    // That breaks the invariant the top-of-function comment on attn_positions_/attn_seqused_k_
    // documents as load-bearing: "the previous RunChunk call (if any) ended with stream_.Synchronize()
    // ... so the device is guaranteed idle here" backs a plain (blocking, null-stream) hipMemcpy that
    // races an in-flight non-blocking-stream kernel if that invariant is false. No corruption was
    // observed because InjectFeatures happens not to touch those two buffers, but the invariant
    // itself was silently false with --dflash. Re-synchronize here so RunChunk keeps its documented
    // "device idle on return" contract for every caller, dflash or not.
    stream_.Synchronize();
  }

  pos_ += T;
  started_ = true;
  return logits;
}

std::vector<float> Model::Prefill(const std::vector<int32_t>& token_ids,
                                    const std::function<void()>& on_chunk_captured) {
  if (token_ids.empty()) throw std::runtime_error("Model::Prefill: token_ids is empty");
  // Prefill's chunked-scan GDN path always lands its result at window index 0 (GdnLayerParams::slot
  // == GdnStateManager::SlotForSeq, never a windowed verify slot -- see that class's file comment),
  // exactly what a nullptr-seeded MTP verify call reads. Any num_accepted carried from a PRIOR
  // generation (e.g. --chat's previous turn) refers to THAT turn's now-stale window slots, not this
  // fresh prefill's -- drop it so the first verify round after this call seeds correctly.
  mtp_num_accepted_valid_ = false;
  std::vector<float> logits;
  for (size_t off = 0; off < token_ids.size(); off += static_cast<size_t>(max_chunk_)) {
    const size_t n = std::min(static_cast<size_t>(max_chunk_), token_ids.size() - off);
    const std::vector<int32_t> chunk(token_ids.begin() + static_cast<ptrdiff_t>(off),
                                      token_ids.begin() + static_cast<ptrdiff_t>(off + n));
    const bool is_last_chunk = (off + n) == token_ids.size();
    std::vector<float> chunk_logits = RunChunk(chunk, /*is_prefill_path=*/true,
                                                /*want_logits=*/is_last_chunk);
    // DFlash2 target feature capture (review finding, 2026-09-20 -- see this method's own doc
    // comment in model.h): drain THIS chunk's captured rows before the next iteration's RunChunk
    // call overwrites dflash_features_dev_ starting at row 0 again. No-op cost when the caller
    // passed nullptr (the default) or no capture is attached.
    if (on_chunk_captured) on_chunk_captured();
    if (is_last_chunk) logits = std::move(chunk_logits);
  }
  return logits;
}

std::vector<float> Model::DecodeStep(int32_t token_id) {
  return RunChunk({token_id}, /*is_prefill_path=*/false, /*want_logits=*/true);
}

int32_t Model::DecodeStepGreedy(int32_t token_id) {
  int32_t next = -1;
  RunChunk({token_id}, /*is_prefill_path=*/false, /*want_logits=*/true, &next);
  return next;
}

// Coarser granularity than the task's ideal ("each GEMM by (N,K)", separate GDN-kernel/norm/rope/
// quant entries): this pass's time budget only extends to instrumenting the per-BLOCK boundaries
// model.cpp already calls directly (GdnLayer::Forward, AttentionLayer::Forward, Mlp::Forward,
// FinalLmHead::Forward each run one or more GEMMs plus several small kernels internally, all
// folded into that block's one entry here) rather than also threading a profiler handle into
// gdn_layer.cpp/attention_layer.hpp/mlp.cpp/final_lm_head.cpp to split GEMM-by-shape from
// norm/rope/quant within each block -- see docs/perf.md's profile table and "Known gaps" note for
// what this does and does not break out, and tools/profile/README.md for how to extend it.
Model::StepProfile Model::DecodeStepProfiled(int32_t token_id) {
  using Clock = std::chrono::steady_clock;
  const auto wall_t0 = Clock::now();
  r4dx_kernel_launch_counter_reset();  // docs/r9700.md P2/task item 4: count just this one step

  const ModelConfig& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = container_.NumLoadedLayers();
  const bool has_init = started_;
  const hipStream_t s = stream_.get();

  SpanAccumulator acc;

  // Same device-resident-vs-host branch real decode takes (RunChunk's own comment) -- review
  // finding, 2026-09-20: this used to always take the host path, so "259 r4dx-owned launches/
  // token" undercounted real decode by one launch (the device gather this profile never exercised)
  // whenever the container was loaded with embed_device_resident=true (the default).
  acc.Add(s, "embed", [&] {
    if (container_.EmbedTokensDeviceResident()) {
      std::copy_n(&token_id, 1, embed_ids_host_.begin());
      embed_ids_dev_.CopyFromHostAsync(embed_ids_host_.data(), 1, stream_);
      EmbedTokensDeviceGather(stream_, container_.EmbedTokensDevice(), hidden,
                               embed_ids_dev_.data(), 1, cfg.vocab_size, buf_a_.data());
    } else {
      EmbedTokens(stream_, container_.EmbedTokensHost(), cfg.vocab_size, hidden, {token_id},
                  embed_staging_, buf_a_);
    }
  });

  std::vector<int32_t> positions_h = {static_cast<int32_t>(pos_)};
  attn_positions_.CopyFromHost(positions_h.data(), positions_h.size());
  const int32_t seqused_k_h = static_cast<int32_t>(pos_ + 1);
  attn_seqused_k_.CopyFromHost(&seqused_k_h, 1);

  uint16_t* cur = buf_a_.data();
  uint16_t* other = buf_b_.data();
  const uint16_t* normed_in = nullptr;  // R3 fusion, see RunChunk's identical pattern
  int normed_in_epilogue = r4dx_epilogue_none;  // R2/P2, see RunChunk's identical pattern

  for (int64_t i = 0; i < num_layers; ++i) {
    const LayerWeights& lw = container_.Layer(i);
    const bool has_next_layer = (i + 1 < num_layers);
    const uint16_t* mlp_norm_weight = lw.post_attention_layernorm.data();

    // Milestone 3 profiling pass (docs/r9700.md R5/Q3): `&acc` is now threaded straight into
    // GdnLayer::Forward/AttentionLayer::Forward/Mlp::Forward instead of this loop wrapping each
    // call in one coarse "gdn_layers"/"attn_layers"/"mlp" span -- every kernel those methods launch
    // now records its OWN named span (profile_span.h's "gemm:" prefix convention), aggregated by
    // name across all num_layers calls into the same accumulator, so GDN's excess is attributable
    // per kernel (rmsnorm/conv_update/recurrent_update/in_proj_*/out_proj/residual) rather than as
    // one 88 us/layer lump. gpu_sum_ms is unaffected (same total, finer buckets).
    if (cfg.IsGdnLayer(i)) {
      GdnLayer layer(cfg, lw.input_layernorm, *lw.gdn);
      GdnLayerParams p;
      p.slot = gdn_states_[static_cast<size_t>(i)]->SlotForSeq(0);
      p.is_prefill = false;
      p.has_init = has_init;
      // Same plain-decode/MTP-desync guard RunChunk applies (that method's own comment) -- review
      // finding, 2026-09-20: this was omitted here, so profiling a DecodeStepProfiled call on an
      // MTP-enabled Model always seeded GDN from window index 0 regardless of the last verify
      // round's real num_accepted, silently profiling the wrong GDN window.
      p.num_accepted =
          (draft_window_ > 1 && mtp_num_accepted_valid_) ? mtp_num_accepted_dev_.data() : nullptr;
      layer.Forward(stream_, arena_, *gdn_states_[static_cast<size_t>(i)], gdn_control_, cur,
                    cur, 1, p, normed_in, mlp_norm_weight, buf_normed_.data(), &acc,
                    normed_in_epilogue, buf_normed_pre_.data(), buf_normed_pre_scale_.data(),
                    body_epilogue_, buf_normed_pre_.data(), buf_normed_pre_scale_.data());
    } else {
      attention::AttnConfig acfg;
      acfg.hidden = static_cast<int>(hidden);
      acfg.num_heads = static_cast<int>(cfg.num_attention_heads);
      acfg.kv_heads = static_cast<int>(cfg.num_key_value_heads);
      acfg.head_dim = static_cast<int>(cfg.head_dim);
      acfg.rotary_dim = static_cast<int>(cfg.RotaryDim());
      acfg.rope_theta = static_cast<float>(cfg.rope_theta);
      acfg.rms_eps = static_cast<float>(cfg.rms_norm_eps);
      attention::AttentionLayer layer(acfg);

      attention::AttnWeights aw;
      aw.input_layernorm = lw.input_layernorm.data();
      aw.qg = &lw.attn->qg;
      aw.k = &lw.attn->k;
      aw.v = &lw.attn->v;
      aw.o = &lw.attn->o;
      aw.q_norm = lw.attn->q_norm.data();
      aw.k_norm = lw.attn->k_norm.data();
      aw.k_descale = lw.attn->k_descale.data();
      aw.v_descale = lw.attn->v_descale.data();

      layer.Forward(arena_, cur, other, aw, *kv_caches_[static_cast<size_t>(i)], 1,
                    static_cast<int>(pos_), attn_positions_.data(), attn_seqused_k_.data(), s,
                    normed_in, mlp_norm_weight, buf_normed_.data(), &acc, normed_in_epilogue,
                    buf_normed_pre_.data(), buf_normed_pre_scale_.data(), body_epilogue_,
                    buf_normed_pre_.data(), buf_normed_pre_scale_.data());
      std::swap(cur, other);
    }

    {
      Mlp mlp(cfg, lw.post_attention_layernorm, lw.mlp);
      mlp.Forward(stream_, arena_, cur, cur, 1, buf_normed_.data(),
                  has_next_layer ? container_.Layer(i + 1).input_layernorm.data() : nullptr,
                  has_next_layer ? buf_normed_.data() : nullptr, &acc, normed_in_epilogue,
                  buf_normed_pre_.data(), buf_normed_pre_scale_.data(),
                  has_next_layer ? body_epilogue_ : r4dx_epilogue_none,
                  has_next_layer ? buf_normed_pre_.data() : nullptr,
                  has_next_layer ? buf_normed_pre_scale_.data() : nullptr);
    }
    normed_in = has_next_layer ? buf_normed_.data() : nullptr;
    normed_in_epilogue = has_next_layer ? body_epilogue_ : r4dx_epilogue_none;

    arena_.Reset();
  }

  acc.Add(s, "final_norm+lm_head (full-vocab GEMM + widen)", [&] {
    FinalLmHead head(cfg, container_.FinalNorm(), container_.LmHead());
    head.Forward(stream_, arena_, cur, logits_dev_.data(), /*T=*/1);
    r4dx_argmax_f32(reinterpret_cast<int64_t>(logits_dev_.data()),
                     reinterpret_cast<int64_t>(argmax_dev_.data()), cfg.vocab_size,
                     reinterpret_cast<int64_t>(s));
    arena_.Reset();
  });

  // Every kernel `acc` recorded above was already ENQUEUED (asynchronously) by this point --
  // `host_enqueue_ms` below is the pure host-side cost of issuing them (CPU work like the
  // embedding gather plus launch overhead), not GPU-blocked. acc.Finish() then does ONE
  // hipEventSynchronize (blocks until that already-queued GPU work actually completes) and reads
  // back every event pair's elapsed time; the subsequent 4-byte D2H is then safe with no further
  // wait (the GPU is already idle). Both are host-chrono-measured together as `finish_wait_ms` --
  // see StepProfile's own comment (model.h) for why this must NOT be summed with `gpu_sum_ms`
  // (they measure overlapping time from two different clocks, not sequential costs).
  const auto enqueue_done_t = Clock::now();
  std::vector<Model::ProfileEntry> entries =
      ToProfileEntries(acc.Finish());  // blocks until the GPU is idle
  int32_t next = -1;
  argmax_dev_.CopyToHost(&next, 1);  // GPU already idle (Finish() above already waited) -- safe
  const auto finish_t = Clock::now();

  const auto wall_t1 = Clock::now();
  StepProfile sp;
  sp.entries = std::move(entries);
  for (const auto& e : sp.entries) sp.gpu_sum_ms += e.ms;
  sp.host_enqueue_ms = std::chrono::duration<double, std::milli>(enqueue_done_t - wall_t0).count();
  sp.finish_wait_ms = std::chrono::duration<double, std::milli>(finish_t - enqueue_done_t).count();
  sp.wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();
  sp.r4dx_kernel_launches = r4dx_kernel_launch_counter_get();

  pos_ += 1;
  started_ = true;
  return sp;
}

Model::StepProfile Model::PrefillProfiled(const std::vector<int32_t>& token_ids) {
  using Clock = std::chrono::steady_clock;
  const auto wall_t0 = Clock::now();
  if (token_ids.empty()) throw std::runtime_error("Model::PrefillProfiled: token_ids is empty");
  r4dx_kernel_launch_counter_reset();

  const ModelConfig& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = container_.NumLoadedLayers();
  const hipStream_t s = stream_.get();

  SpanAccumulator acc;

  for (size_t off = 0; off < token_ids.size(); off += static_cast<size_t>(max_chunk_)) {
    const size_t n = std::min(static_cast<size_t>(max_chunk_), token_ids.size() - off);
    const std::vector<int32_t> chunk(token_ids.begin() + static_cast<ptrdiff_t>(off),
                                      token_ids.begin() + static_cast<ptrdiff_t>(off + n));
    const int64_t T = static_cast<int64_t>(chunk.size());
    const bool has_init = started_;

    // Same device-resident-vs-host branch real prefill (RunChunk) takes -- see DecodeStepProfiled's
    // identical fix above (review finding, 2026-09-20).
    acc.Add(s, "embed", [&] {
      if (container_.EmbedTokensDeviceResident()) {
        std::copy(chunk.begin(), chunk.end(), embed_ids_host_.begin());
        embed_ids_dev_.CopyFromHostAsync(embed_ids_host_.data(), static_cast<size_t>(T), stream_);
        EmbedTokensDeviceGather(stream_, container_.EmbedTokensDevice(), hidden,
                                 embed_ids_dev_.data(), T, cfg.vocab_size, buf_a_.data());
      } else {
        EmbedTokens(stream_, container_.EmbedTokensHost(), cfg.vocab_size, hidden, chunk,
                    embed_staging_, buf_a_);
      }
    });

    std::vector<int32_t> positions_h(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
      positions_h[static_cast<size_t>(t)] = static_cast<int32_t>(pos_ + t);
    }
    attn_positions_.CopyFromHost(positions_h.data(), positions_h.size());
    const int32_t seqused_k_h = static_cast<int32_t>(pos_ + T);
    attn_seqused_k_.CopyFromHost(&seqused_k_h, 1);

    uint16_t* cur = buf_a_.data();
    uint16_t* other = buf_b_.data();
    const uint16_t* normed_in = nullptr;  // R3 fusion, see RunChunk's identical pattern
    int normed_in_epilogue = r4dx_epilogue_none;  // R2/P2, see RunChunk's identical pattern

    for (int64_t i = 0; i < num_layers; ++i) {
      const LayerWeights& lw = container_.Layer(i);
      const bool has_next_layer = (i + 1 < num_layers);
      const uint16_t* mlp_norm_weight = lw.post_attention_layernorm.data();

      if (cfg.IsGdnLayer(i)) {
        GdnLayer layer(cfg, lw.input_layernorm, *lw.gdn);
        GdnLayerParams p;
        p.slot = gdn_states_[static_cast<size_t>(i)]->SlotForSeq(0);
        p.is_prefill = true;
        p.has_init = has_init;
        layer.Forward(stream_, arena_, *gdn_states_[static_cast<size_t>(i)], gdn_control_, cur,
                      cur, T, p, normed_in, mlp_norm_weight, buf_normed_.data(), &acc,
                      normed_in_epilogue, buf_normed_pre_.data(), buf_normed_pre_scale_.data(),
                      body_epilogue_, buf_normed_pre_.data(), buf_normed_pre_scale_.data());
      } else {
        attention::AttnConfig acfg;
        acfg.hidden = static_cast<int>(hidden);
        acfg.num_heads = static_cast<int>(cfg.num_attention_heads);
        acfg.kv_heads = static_cast<int>(cfg.num_key_value_heads);
        acfg.head_dim = static_cast<int>(cfg.head_dim);
        acfg.rotary_dim = static_cast<int>(cfg.RotaryDim());
        acfg.rope_theta = static_cast<float>(cfg.rope_theta);
        acfg.rms_eps = static_cast<float>(cfg.rms_norm_eps);
        attention::AttentionLayer layer(acfg);

        attention::AttnWeights aw;
        aw.input_layernorm = lw.input_layernorm.data();
        aw.qg = &lw.attn->qg;
        aw.k = &lw.attn->k;
        aw.v = &lw.attn->v;
        aw.o = &lw.attn->o;
        aw.q_norm = lw.attn->q_norm.data();
        aw.k_norm = lw.attn->k_norm.data();
        aw.k_descale = lw.attn->k_descale.data();
        aw.v_descale = lw.attn->v_descale.data();

        layer.Forward(arena_, cur, other, aw, *kv_caches_[static_cast<size_t>(i)],
                      static_cast<int>(T), static_cast<int>(pos_), attn_positions_.data(),
                      attn_seqused_k_.data(), s, normed_in, mlp_norm_weight, buf_normed_.data(),
                      &acc, normed_in_epilogue, buf_normed_pre_.data(),
                      buf_normed_pre_scale_.data(), body_epilogue_, buf_normed_pre_.data(),
                      buf_normed_pre_scale_.data());
        std::swap(cur, other);
      }

      {
        // x_normed_in must be buf_normed_.data() (this layer's Gdn/Attn call above just fused its
        // own rmsnorm epilogue into it, same as RunChunk's identical call), NOT `normed_in` (the
        // carry-forward variable that is null for layer 0 of every chunk) -- review finding,
        // 2026-09-20: this profiled a code path real prefill never runs (layer 0 of every chunk
        // recomputed its own rmsnorm from scratch here instead of consuming the fused one,
        // reporting a spurious "mlp.rmsnorm" launch --profile-prefill's own table showed once per
        // chunk that RunChunk never issues).
        Mlp mlp(cfg, lw.post_attention_layernorm, lw.mlp);
        mlp.Forward(stream_, arena_, cur, cur, T, buf_normed_.data(),
                    has_next_layer ? container_.Layer(i + 1).input_layernorm.data() : nullptr,
                    has_next_layer ? buf_normed_.data() : nullptr, &acc, normed_in_epilogue,
                    buf_normed_pre_.data(), buf_normed_pre_scale_.data(),
                    has_next_layer ? body_epilogue_ : r4dx_epilogue_none,
                    has_next_layer ? buf_normed_pre_.data() : nullptr,
                    has_next_layer ? buf_normed_pre_scale_.data() : nullptr);
      }
      normed_in = has_next_layer ? buf_normed_.data() : nullptr;
      normed_in_epilogue = has_next_layer ? body_epilogue_ : r4dx_epilogue_none;

      arena_.Reset();
    }

    // Diagnostic-only (see model.h's doc comment): no final_norm/lm_head, no MTP KV priming --
    // still synchronize before the next chunk's positions/seqused_k upload, same hazard RunChunk's
    // own comment describes (a plain hipMemcpy against a hipStreamNonBlocking stream's still-queued
    // writers is a race without this).
    stream_.Synchronize();
    pos_ += T;
    started_ = true;
  }

  StepProfile sp;
  sp.entries = ToProfileEntries(acc.Finish());
  for (const auto& e : sp.entries) sp.gpu_sum_ms += e.ms;
  sp.wall_ms = std::chrono::duration<double, std::milli>(Clock::now() - wall_t0).count();
  sp.r4dx_kernel_launches = r4dx_kernel_launch_counter_get();
  return sp;
}

// ---- MTP self-speculation (docs/mtp.md) --------------------------------------------------------

std::vector<int32_t> Model::VerifyWindow(const std::vector<int32_t>& candidates,
                                          std::vector<float>* logits_out) {
  // No MTP head required (generalised for DFlash2, 2026-09-20 stage S2): what this method actually
  // needs is the speculative-verify SIZING -- the GDN window bank, verify_logits_dev_/
  // verify_argmax_dev_ and mtp_num_accepted_dev_ -- all of which Load() allocates whenever
  // draft_window_ > 1, i.e. for mtp_draft_k > 0 OR dflash_draft_k > 0. A DFlash2 container has no
  // mtp.* weights at all and must still be able to verify here.
  if (draft_window_ <= 1) {
    throw std::runtime_error(
        "Model::VerifyWindow: this Model was not sized for speculative verification (Load() with "
        "ModelOptions::mtp_draft_k > 0 -- plus a container that has mtp.* weights -- or with "
        "ModelOptions::dflash_draft_k > 0)");
  }
  const int64_t T = static_cast<int64_t>(candidates.size());
  if (T < 1 || T > max_chunk_) {
    throw std::runtime_error("Model::VerifyWindow: candidates.size() must be in [1, " +
                              std::to_string(max_chunk_) + "]");
  }
  // verify_logits_dev_/verify_argmax_dev_ are sized for exactly draft_window_ candidates (this
  // class's own field comment, model.h) -- the max_chunk_ (64) check above does NOT enforce that
  // narrower bound, so a caller passing more than draft_window_ candidates (this method is public,
  // per its own doc comment, specifically for callers like tests/model/test_mtp.cpp) would
  // silently overflow those buffers (review finding, 2026-09-20: an 8-candidate call on a
  // draft_k=3 model writes ~8 MB into a 4 MB allocation). Today this is saved only
  // by an unrelated guard in a different component (gdn_layer.cpp's own `T > MaxDecodeWindow()`
  // throw, which happens to run first because layer 0 of the real container is a GDN layer) --
  // enforce the real precondition here directly rather than relying on that coincidence.
  if (T > draft_window_) {
    throw std::runtime_error("Model::VerifyWindow: candidates.size() must be in [1, " +
                              std::to_string(draft_window_) + "] (DraftWindow())");
  }
  const ModelConfig& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = container_.NumLoadedLayers();
  const bool has_init = started_;  // always true: VerifyWindow only ever runs after a Prefill

  // Same device-resident gather as RunChunk (model.cpp's own comment above) -- VerifyWindow is the
  // other hot-path caller (once per MTP round, docs/mtp.md).
  if (container_.EmbedTokensDeviceResident()) {
    std::copy(candidates.begin(), candidates.end(), embed_ids_host_.begin());
    embed_ids_dev_.CopyFromHostAsync(embed_ids_host_.data(), static_cast<size_t>(T), stream_);
    EmbedTokensDeviceGather(stream_, container_.EmbedTokensDevice(), hidden, embed_ids_dev_.data(),
                             T, cfg.vocab_size, buf_a_.data());
  } else {
    EmbedTokens(stream_, container_.EmbedTokensHost(), cfg.vocab_size, hidden, candidates,
                embed_staging_, buf_a_);
  }

  // Same reasoning/hazard as RunChunk's own upload (model.cpp's RunChunk comment): safe here
  // because the previous call (Prefill/DecodeStep*/VerifyWindow) always ends with
  // stream_.Synchronize() before returning.
  std::vector<int32_t> positions_h(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) positions_h[static_cast<size_t>(t)] = static_cast<int32_t>(pos_ + t);
  attn_positions_.CopyFromHost(positions_h.data(), positions_h.size());
  const int32_t seqused_k_h = static_cast<int32_t>(pos_ + T);
  attn_seqused_k_.CopyFromHost(&seqused_k_h, 1);

  const int32_t* num_accepted_ptr = mtp_num_accepted_valid_ ? mtp_num_accepted_dev_.data() : nullptr;

  uint16_t* cur = buf_a_.data();
  uint16_t* other = buf_b_.data();
  const uint16_t* normed_in = nullptr;  // R3 fusion, see RunChunk's identical pattern above
  int normed_in_epilogue = r4dx_epilogue_none;  // R2/P2, see RunChunk's identical pattern

  for (int64_t i = 0; i < num_layers; ++i) {
    const LayerWeights& lw = container_.Layer(i);
    const bool has_next_layer = (i + 1 < num_layers);
    const uint16_t* mlp_norm_weight = lw.post_attention_layernorm.data();

    // DFlash2 target feature capture -- see RunChunk's identical call site/comment above. A verify
    // window's `cur` right here is the residual stream entering layer `i` for these <=
    // (mtp_draft_k_+1) candidate rows.
    CaptureDflashLayerInput(stream_, i, cur, T, hidden, dflash_target_layers_,
                             dflash_features_dev_.data());

    if (cfg.IsGdnLayer(i)) {
      GdnLayer layer(cfg, lw.input_layernorm, *lw.gdn);
      GdnLayerParams p;
      p.slot = gdn_states_[static_cast<size_t>(i)]->SlotForSeq(0);
      p.is_prefill = false;
      p.has_init = has_init;
      p.num_accepted = num_accepted_ptr;
      layer.Forward(stream_, arena_, *gdn_states_[static_cast<size_t>(i)], gdn_control_, cur, cur,
                    T, p, normed_in, mlp_norm_weight, buf_normed_.data(), /*prof=*/nullptr,
                    normed_in_epilogue, buf_normed_pre_.data(), buf_normed_pre_scale_.data(),
                    body_epilogue_, buf_normed_pre_.data(), buf_normed_pre_scale_.data());
    } else {
      attention::AttnConfig acfg;
      acfg.hidden = static_cast<int>(hidden);
      acfg.num_heads = static_cast<int>(cfg.num_attention_heads);
      acfg.kv_heads = static_cast<int>(cfg.num_key_value_heads);
      acfg.head_dim = static_cast<int>(cfg.head_dim);
      acfg.rotary_dim = static_cast<int>(cfg.RotaryDim());
      acfg.rope_theta = static_cast<float>(cfg.rope_theta);
      acfg.rms_eps = static_cast<float>(cfg.rms_norm_eps);
      attention::AttentionLayer layer(acfg);

      attention::AttnWeights aw;
      aw.input_layernorm = lw.input_layernorm.data();
      aw.qg = &lw.attn->qg;
      aw.k = &lw.attn->k;
      aw.v = &lw.attn->v;
      aw.o = &lw.attn->o;
      aw.q_norm = lw.attn->q_norm.data();
      aw.k_norm = lw.attn->k_norm.data();
      aw.k_descale = lw.attn->k_descale.data();
      aw.v_descale = lw.attn->v_descale.data();

      // Note: this call unconditionally writes K/V for every one of the T candidate positions into
      // the paged cache BEFORE the attention math runs (AttentionLayer::Forward's own doc comment),
      // including whichever candidates turn out to be rejected below -- harmless (docs/mtp.md,
      // this class's own file comment): pos_ only ever advances by however many candidates
      // DecodeStepMtpGreedy actually commits, so a rejected candidate's stale KV entry is
      // unconditionally overwritten the next time that same slot (== that same position) is
      // written, before anything could ever read it.
      layer.Forward(arena_, cur, other, aw, *kv_caches_[static_cast<size_t>(i)],
                    static_cast<int>(T), static_cast<int>(pos_), attn_positions_.data(),
                    attn_seqused_k_.data(), stream_.get(), normed_in, mlp_norm_weight,
                    buf_normed_.data(), /*prof=*/nullptr, normed_in_epilogue,
                    buf_normed_pre_.data(), buf_normed_pre_scale_.data(), body_epilogue_,
                    buf_normed_pre_.data(), buf_normed_pre_scale_.data());
      std::swap(cur, other);
    }

    Mlp mlp(cfg, lw.post_attention_layernorm, lw.mlp);
    mlp.Forward(stream_, arena_, cur, cur, T, buf_normed_.data(),
                has_next_layer ? container_.Layer(i + 1).input_layernorm.data() : nullptr,
                has_next_layer ? buf_normed_.data() : nullptr, /*prof=*/nullptr,
                normed_in_epilogue, buf_normed_pre_.data(), buf_normed_pre_scale_.data(),
                has_next_layer ? body_epilogue_ : r4dx_epilogue_none,
                has_next_layer ? buf_normed_pre_.data() : nullptr,
                has_next_layer ? buf_normed_pre_scale_.data() : nullptr);
    normed_in = has_next_layer ? buf_normed_.data() : nullptr;
    normed_in_epilogue = has_next_layer ? body_epilogue_ : r4dx_epilogue_none;

    arena_.Reset();
  }
  if (!dflash_target_layers_.empty()) dflash_feature_rows_ = T;

  // Per-position logits + greedy argmax, for every one of the T candidate positions (not just the
  // last -- this is what distinguishes a verify window from Prefill/DecodeStep's own tail-only
  // want_logits path).
  FinalLmHead head(cfg, container_.FinalNorm(), container_.LmHead());
  head.Forward(stream_, arena_, cur, verify_logits_dev_.data(), T);
  for (int64_t t = 0; t < T; ++t) {
    r4dx_argmax_f32(
        reinterpret_cast<int64_t>(verify_logits_dev_.data() + t * cfg.vocab_size),
        reinterpret_cast<int64_t>(verify_argmax_dev_.data() + t), cfg.vocab_size,
        reinterpret_cast<int64_t>(stream_.get()));
  }
  arena_.Reset();

  mtp_last_hidden_ = cur;  // valid until the next RunChunk/VerifyWindow's EmbedTokens overwrites it

  stream_.Synchronize();  // same hazard class as RunChunk's own D2H -- see that method's comment

  std::vector<int32_t> preds(static_cast<size_t>(T));
  verify_argmax_dev_.CopyToHost(preds.data(), preds.size());
  if (logits_out != nullptr) {
    logits_out->resize(static_cast<size_t>(T * cfg.vocab_size));
    verify_logits_dev_.CopyToHost(logits_out->data(), logits_out->size());
  }
  return preds;
}

void Model::CommitVerifiedWindow(int64_t num_committed) {
  if (draft_window_ <= 1) {
    throw std::runtime_error(
        "Model::CommitVerifiedWindow: this Model was not sized for speculative verification");
  }
  if (num_committed < 1 || num_committed > draft_window_) {
    throw std::runtime_error("Model::CommitVerifiedWindow: num_committed must be in [1, " +
                              std::to_string(draft_window_) + "]");
  }
  // Identical to DecodeStepMtpGreedy's own commit block (minus MTP's h_seed reseed, which a
  // non-MTP drafter has nothing to do with): thread the acceptance count into the next GDN
  // decode/verify call's window-bank seed and advance pos_ by exactly what was committed -- NOT by
  // the window width, which would silently accept every candidate regardless of whether the real
  // model agreed. Safe to write mtp_num_accepted_dev_ synchronously here: VerifyWindow ended with
  // a stream_.Synchronize() before returning, so every kernel that read the previous value is done.
  const int32_t n = static_cast<int32_t>(num_committed);
  mtp_num_accepted_dev_.CopyFromHost(&n, 1);
  mtp_num_accepted_valid_ = true;
  pos_ += num_committed;
  started_ = true;
}

std::vector<int32_t> Model::DecodeStepMtpGreedy(int32_t token_id, int64_t k) {
  if (!mtp_) {
    throw std::runtime_error(
        "Model::DecodeStepMtpGreedy: MTP is not enabled on this Model (Load() with "
        "ModelOptions::mtp_draft_k > 0 and a container that has mtp.* weights)");
  }
  if (k < 0 || k > mtp_draft_k_) {
    throw std::runtime_error("Model::DecodeStepMtpGreedy: k must be in [0, " +
                              std::to_string(mtp_draft_k_) + "]");
  }
  const ModelConfig& cfg = container_.Config();
  const int64_t hidden = cfg.hidden_size;

  std::vector<int32_t> drafts;
  if (k > 0) {
    // base_pos: the REAL sequence position step 0 writes into MTP's own KV cache -- pos_-1 is
    // exactly the ONE position PrimeKv would otherwise prime from (mtp_seed_hidden_, token_id) if
    // this call drafted nothing at all (see mtp_head.h's PrimeKv comment and model.cpp's RunChunk).
    drafts = mtp_->Draft(stream_, arena_, cfg, container_.Mtp(), mtp_seed_hidden_.data(), token_id,
                          container_.EmbedTokensHost(),
                          container_.EmbedTokensDeviceResident() ? container_.EmbedTokensDevice()
                                                                  : nullptr,
                          cfg.vocab_size, container_.LmHead(), k,
                          /*base_pos=*/pos_ - 1, mtp_draft_reduced_vocab_);
    arena_.Reset();
  } else if (mtp_seed_valid_) {
    // k==0 degenerate call (doc's own "single DecodeStepGreedy-equivalent" case): Draft() is
    // skipped entirely, so nothing above would otherwise prime MTP's KV at position pos_-1 --
    // without this, a later k>0 call on the same Model would compute a correct base_pos but find
    // that position's KV entry missing (never written), breaking lockstep continuity. Mirrors
    // exactly what Draft()'s own step 0 would have computed.
    mtp_->PrimeKv(stream_, arena_, cfg, container_.Mtp(), pos_ - 1, mtp_seed_hidden_.data(),
                  {token_id}, container_.EmbedTokensHost(), cfg.vocab_size);
    arena_.Reset();
  }

  std::vector<int32_t> candidates;
  candidates.reserve(drafts.size() + 1);
  candidates.push_back(token_id);
  candidates.insert(candidates.end(), drafts.begin(), drafts.end());

  const std::vector<int32_t> preds = VerifyWindow(candidates);  // size == candidates.size()

  // Greedy acceptance: the longest prefix of drafts whose own predecessor's argmax matches it.
  // preds[i] is the real model's own next-token prediction after processing candidate i
  // (candidates[0]==token_id, candidates[1..]==drafts) -- so preds[i] should equal drafts[i] (the
  // draft that FOLLOWS candidate i) for the draft to be confirmed.
  int64_t num_accepted_drafts = 0;
  while (num_accepted_drafts < static_cast<int64_t>(drafts.size()) &&
         preds[static_cast<size_t>(num_accepted_drafts)] ==
             drafts[static_cast<size_t>(num_accepted_drafts)]) {
    ++num_accepted_drafts;
  }
  const int32_t corrected = preds[static_cast<size_t>(num_accepted_drafts)];
  const int64_t num_committed = num_accepted_drafts + 1;  // +1 for token_id itself

  // Thread this round's acceptance count into the NEXT GDN decode/verify call (gdn_state.h's file
  // comment) and advance pos_ by exactly what was committed -- NOT by candidates.size(), which
  // would silently accept every draft regardless of whether the real model agreed.
  const int32_t num_committed_i32 = static_cast<int32_t>(num_committed);
  mtp_num_accepted_dev_.CopyFromHost(&num_committed_i32, 1);
  mtp_num_accepted_valid_ = true;
  pos_ += num_committed;
  started_ = true;

  // Reseed MTP's h_seed from the row that produced `corrected` -- candidate index
  // num_accepted_drafts of THIS call's window, i.e. mtp_last_hidden_'s row num_accepted_drafts.
  R4DX_HIP_CHECK(hipMemcpyAsync(
      mtp_seed_hidden_.data(), mtp_last_hidden_ + num_accepted_drafts * hidden,
      static_cast<size_t>(hidden) * sizeof(uint16_t), hipMemcpyDeviceToDevice, stream_.get()));
  stream_.Synchronize();

  std::vector<int32_t> result(drafts.begin(), drafts.begin() + num_accepted_drafts);
  result.push_back(corrected);
  return result;
}

int64_t Model::DflashInjectedCount() const {
  if (!dflash_.has_value()) {
    throw std::runtime_error("Model::DflashInjectedCount: requires DflashEnabled()");
  }
  return dflash_->InjectedCount();
}

int64_t Model::DflashValidFrom() const {
  if (!dflash_.has_value()) {
    throw std::runtime_error("Model::DflashValidFrom: requires DflashEnabled()");
  }
  return dflash_->ValidFrom();
}

std::vector<int32_t> Model::DecodeStepDflashGreedy(int32_t token_id, int64_t k, float p_min,
                                                    int64_t n_min, int64_t* walk_len_out,
                                                    DflashRoundTrace* trace_out,
                                                    std::vector<int32_t>* drafted_tokens_out) {
  if (!dflash_.has_value()) {
    throw std::runtime_error(
        "Model::DecodeStepDflashGreedy: DFlash2 is not enabled on this Model (Load() with "
        "ModelOptions::dflash_container set and dflash_draft_k > 0)");
  }
  if (k < 0 || k > dflash_draft_k_) {
    throw std::runtime_error("Model::DecodeStepDflashGreedy: k must be in [0, " +
                              std::to_string(dflash_draft_k_) + "]");
  }
  if (!dflash_injection_enabled_) {
    throw std::runtime_error(
        "Model::DecodeStepDflashGreedy: drafter injection is disabled on this Model "
        "(SetDflashInjectionEnabled(false)), so the drafter's ring frontier lags pos_ and a draft "
        "block here would be built at the wrong absolute positions -- re-enable injection and run "
        "at least one RunChunk (a prefill chunk or a plain decode step) first");
  }
  // The drift check that backs the whole round path below: DraftRound builds its block at the
  // DRAFTER's own frontier (InjectedCount()) while VerifyWindow/CommitVerifiedWindow work at the
  // MODEL's (pos_). These are equal after every injection -- RunChunk injects every chunk's rows,
  // and the commit block at the end of this method injects every accepted row -- so a mismatch
  // means some path advanced one without the other, which would silently draft against the wrong
  // context rather than fail. Cheap (two integer loads) next to a round's own GEMMs.
  if (dflash_->InjectedCount() != pos_) {
    throw std::runtime_error(
        "Model::DecodeStepDflashGreedy: drafter frontier (" +
        std::to_string(dflash_->InjectedCount()) + ") != Model position (" + std::to_string(pos_) +
        ") -- the drafter's ring and this Model's committed sequence have drifted apart");
  }

  // Built fresh every call (docs/dflash2.md, dflash_draft.h's own doc comment on
  // MakeTargetEmbeddingProvider/MakeTargetLmHeadProvider): both closures capture raw pointers into
  // the LIVE container_, and must not be cached across a Model move/reload -- the construction cost
  // (two std::function allocations) is negligible next to one draft round's own GEMMs.
  const DflashEmbeddingProvider embed = MakeTargetEmbeddingProvider(container_);
  const DflashLmHeadProvider lm_head_provider = MakeTargetLmHeadProvider(container_);

  const DflashDraftResult draft = dflash_->DraftRound(stream_, arena_, token_id, k, p_min, n_min,
                                                       embed, lm_head_provider, trace_out);
  arena_.Reset();
  if (walk_len_out != nullptr) *walk_len_out = draft.walk_len;
  if (drafted_tokens_out != nullptr) *drafted_tokens_out = draft.tokens;

  std::vector<int32_t> candidates;
  candidates.reserve(draft.tokens.size() + 1);
  candidates.push_back(token_id);
  candidates.insert(candidates.end(), draft.tokens.begin(), draft.tokens.end());

  // VerifyWindow also (as a side effect of running every layer's own capture hook, since
  // AttachDflashFeatureCapture is attached whenever dflash_ is set) refills DflashFeatureBuffer()/
  // DflashFeatureRows() with THIS window's own captured target features, one row per candidate --
  // exactly what gets injected below for the accepted prefix. VerifyWindow itself never injects
  // anything (model.h's own doc comment on SetDflashCaptureObserver) -- only this method does, once
  // it knows how many rows were actually accepted.
  const std::vector<int32_t> preds = VerifyWindow(candidates);  // size == candidates.size()

  // Greedy acceptance: the longest prefix of drafts whose own predecessor's argmax matches it --
  // identical logic to DecodeStepMtpGreedy above (docs/dflash2.md section 5: "a rows accepted, the
  // anchor is always accepted").
  int64_t num_accepted_drafts = 0;
  while (num_accepted_drafts < static_cast<int64_t>(draft.tokens.size()) &&
         preds[static_cast<size_t>(num_accepted_drafts)] ==
             draft.tokens[static_cast<size_t>(num_accepted_drafts)]) {
    ++num_accepted_drafts;
  }
  const int32_t corrected = preds[static_cast<size_t>(num_accepted_drafts)];
  const int64_t num_committed = num_accepted_drafts + 1;  // +1 for token_id (the anchor) itself

  // Inject the accepted prefix's own just-captured target features into the drafter's ring BEFORE
  // CommitVerifiedWindow (which only moves this Model's pos_/GDN bookkeeping, never touches
  // DflashFeatureBuffer()/dflash_->InjectedCount()) -- docs/dflash2.md section 5 step 4: "inject
  // those a rows' captured features at positions n..n+a-1". Rows 0..num_committed-1 of the just-
  // captured window are exactly the accepted prefix (row 0 == the anchor, which is always accepted)
  // because VerifyWindow captures candidates in the same order they were verified.
  if (DflashFeatureRows() < num_committed) {
    throw std::runtime_error(
        "Model::DecodeStepDflashGreedy: internal error -- VerifyWindow captured fewer rows (" +
        std::to_string(DflashFeatureRows()) + ") than the accepted prefix (" +
        std::to_string(num_committed) + ")");
  }
  dflash_->InjectFeatures(stream_, arena_, DflashFeatureBuffer(), num_committed,
                          dflash_->InjectedCount());
  arena_.Reset();

  // Advances pos_ by exactly num_committed and threads the GDN acceptance count, identically to
  // DecodeStepMtpGreedy's own commit block above (this is exactly what CommitVerifiedWindow factors
  // out for a non-MTP drafter to call -- model.h's own doc comment on that method).
  CommitVerifiedWindow(num_committed);

  std::vector<int32_t> result(draft.tokens.begin(), draft.tokens.begin() + num_accepted_drafts);
  result.push_back(corrected);
  return result;
}

std::vector<uint16_t> Model::DebugSeedHiddenBf16() const {
  if (!mtp_.has_value()) {
    throw std::runtime_error("Model::DebugSeedHiddenBf16: requires MtpEnabled() (mtp_draft_k>0)");
  }
  if (!mtp_seed_valid_) {
    throw std::runtime_error(
        "Model::DebugSeedHiddenBf16: mtp_seed_hidden_ not yet valid -- call Prefill/DecodeStep* "
        "first");
  }
  return mtp_seed_hidden_.CopyToHost();
}

}  // namespace r4dx::model
