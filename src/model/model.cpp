#include "model.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <exception>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "attn_config.h"
#include "dflash_draft_weights.h"
#include "embedding.h"
#include "final_lm_head.h"
#include "gdn_layer.h"
#include "linear.h"
#include "mlp.h"
#include "position_ids.h"  // src/vision: BuildMropePositionIds (docs/vision.md)
#include "profile_span.h"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/model/attention/attention_layer.hpp"
#include "r4dx/model/attention/types.hpp"
#include "tp/tp_vocab.h"  // tensor-parallel vocab-split merges (docs/tp.md 7.3)

namespace r4dx::model {

namespace {

// ---- tensor parallel lockstep fingerprints (docs/tp.md 6.2 H1, H2) ------------------------------
// RunChunk's / VerifyWindow's entry check: {kind, T, pos_, FNV-1a of the tokens}. `kind` carries
// the call's own mode bits too -- RunChunk: 1 prefill path, 2 logits wanted, 4 greedy, 8 row
// summary; VerifyWindow: 1 full logits out, 2 row summaries -- because each mode issues a different
// sequence of collectives after the layers, and two ranks that agree on the tokens but not on the
// mode would otherwise only find out at a mismatched all-gather.
constexpr uint64_t kLockstepRunChunk = 0x52554e4300000000ull;  // "RUNC"
constexpr uint64_t kLockstepVerify = 0x5645524900000000ull;    // "VERI"

uint64_t Fnv1a64(const std::vector<int32_t>& tokens) {
  uint64_t h = 0xcbf29ce484222325ull;
  const auto* p = reinterpret_cast<const uint8_t*>(tokens.data());
  for (size_t i = 0; i < tokens.size() * sizeof(int32_t); ++i) {
    h ^= p[i];
    h *= 0x100000001b3ull;
  }
  return h;
}

static_assert(sizeof(tp::ArgmaxPair) == 2 * sizeof(int32_t),
              "argmax_pair_dev_ stores one tp::ArgmaxPair as two int32 slots");

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
  // ---- tensor parallel (docs/tp.md 3.3): validate this rank's options ---------------------------
  const TpRankOptions& tp = opts.tp;
  const bool is_tp_rank = tp.world > 1;
  if (tp.world < 1 || tp.world > 2 || tp.rank < 0 || tp.rank >= tp.world) {
    throw std::invalid_argument("Model::Load: ModelOptions::tp needs world in {1, 2} and 0 <= rank "
                                "< world, got world " + std::to_string(tp.world) + ", rank " +
                                std::to_string(tp.rank));
  }
  if (!is_tp_rank && (tp.comm != nullptr || tp.shared_embed_host || tp.dflash_codebooks ||
                   tp.embed_device_resident_decided >= 0 || !tp.vision_weights_on_this_rank)) {
    throw std::invalid_argument(
        "Model::Load: ModelOptions::tp.comm/shared_embed_host/dflash_codebooks/"
        "embed_device_resident_decided/vision_weights_on_this_rank are tensor-parallel options "
        "(tp.world > 1 only)");
  }
  ModelOptions::VisionMode vision_mode = opts.vision;
  if (is_tp_rank) {
    if (tp.comm == nullptr || tp.comm->World() != tp.world || tp.comm->Rank() != tp.rank) {
      throw std::invalid_argument(
          "Model::Load: tp.world > 1 needs a TpComm endpoint of the same world and rank");
    }
    // Staged rejections (docs/tp.md 2.9 step 1, 9.1): the MTP head, the DFlash2 drafter and the
    // vision tower get their tensor-parallel hooks in P5.
    if (opts.mtp_draft_k > 0) {
      throw core::TpUnsupportedError("Model::Load: MTP (mtp_draft_k > 0) is not supported under "
                                     "tensor parallelism yet (docs/tp.md P5)");
    }
    if (!opts.dflash_container.empty()) {
      throw core::TpUnsupportedError("Model::Load: DFlash2 (dflash_container) is not supported "
                                     "under tensor parallelism yet (docs/tp.md P5)");
    }
    if (opts.vision == ModelOptions::VisionMode::kOn) {
      throw core::TpUnsupportedError("Model::Load: --vision on is not supported under tensor "
                                     "parallelism yet (docs/tp.md P5)");
    }
    if (opts.vision == ModelOptions::VisionMode::kAuto) {
      std::cerr << "[r4dx::model::Model] tp rank " << tp.rank << "/" << tp.world
                << ": --vision auto loads text-only under tensor parallelism until docs/tp.md P5\n";
      vision_mode = ModelOptions::VisionMode::kOff;
    }
  }
  // docs/tp.md 2.7: a TP rank's thread consults the per-rank tuning table first. Set on EVERY load
  // (false at TP=1), so the flag follows this thread's latest successful Model::Load rather than any
  // earlier rank load; a TP load that throws clears it again (Appendix B N24).
  SetTp2TuningForThisThread(is_tp_rank);
  struct Tp2FlagOnThrow {
    int uncaught = std::uncaught_exceptions();
    ~Tp2FlagOnThrow() {
      if (std::uncaught_exceptions() > uncaught) SetTp2TuningForThisThread(false);
    }
  } tp2_flag_on_throw;

  Model m;
  m.comm_ = tp.comm;
  const VramSnap vram0 = SnapVram();  // before any of this Load() call's own allocations
  // The vision tower's ~0.90 GiB is loaded between two of its own snapshots so it gets its own
  // VRAM breakdown line -- the same treatment the DFlash2 drafter gets below, and the only honest
  // way to report a delta docs/vision.md quotes as a number (docs/vision.md "Load policy").
  const bool want_vision = vision_mode != ModelOptions::VisionMode::kOff;
  if (!is_tp_rank) {
    m.container_ = Container::Load(opts.container_path, opts.layout, opts.layout, opts.layer_limit,
                                    opts.mtp_head_layout.value_or(opts.layout),
                                    opts.embed_device_resident, want_vision);
  } else {
    ContainerLoadOptions co;
    co.layout = opts.layout;
    co.lm_head_layout = opts.layout;
    co.mtp_head_layout = opts.mtp_head_layout.value_or(opts.layout);
    co.layer_limit = opts.layer_limit;
    co.embed_device_resident = opts.embed_device_resident;
    co.embed_device_resident_decided = tp.embed_device_resident_decided;
    co.load_vision = want_vision && tp.vision_weights_on_this_rank;
    co.parse_vision_config = want_vision;
    co.tp_world = tp.world;
    co.tp_rank = tp.rank;
    co.shared_embed_host = tp.shared_embed_host;
    m.container_ = Container::Load(opts.container_path, co);
  }
  const VramSnap vram1 = SnapVram();  // after container weights are fully resident
  if (vision_mode == ModelOptions::VisionMode::kOn && !m.container_.HasVision()) {
    throw std::runtime_error(
        "Model::Load: --vision on was requested but the container has no vision.* tensors "
        "(convert without --language-model-only, or use --vision auto)");
  }
  if (m.container_.HasVision()) {
    m.vision_.emplace();
    const vision::VisionWeights& vw = m.container_.Vision();
    std::cerr << "[r4dx::model::Model] vision tower loaded: " << vw.tensor_count << " tensors, "
              << GiB(vw.bytes) << " GiB (depth=" << vw.config.depth
              << ", hidden=" << vw.config.hidden_size << ", out_hidden=" << vw.config.out_hidden_size
              << ")\n";
  } else if (vision_mode == ModelOptions::VisionMode::kAuto &&
             m.container_.ContainerHasVisionTensors()) {
    // Cannot happen with the current policy (auto asks for the load), but says so out loud rather
    // than silently leaving a vision-capable container text-only if that policy ever changes.
    std::cerr << "[r4dx::model::Model] container carries vision.* tensors but the tower was not "
                  "loaded -- image requests will be rejected\n";
  }
  const ModelConfig& cfg = m.container_.Config();
  const int64_t hidden = cfg.hidden_size;
  const int64_t num_layers = m.container_.NumLoadedLayers();
  // docs/tp.md 7.2: this rank's lm_head rows and their first global id -- the whole vocabulary and
  // 0 at TP=1, so every sizing below is unchanged there.
  m.vocab_local_ = m.container_.LmHead().N;
  m.vocab_offset_ = cfg.VocabShardBegin();
  if (m.vocab_local_ * tp.world != cfg.vocab_size) {
    throw std::runtime_error("Model::Load: lm_head has " + std::to_string(m.vocab_local_) +
                             " rows, not vocab_size / tp.world = " +
                             std::to_string(cfg.vocab_size / tp.world));
  }

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
  m.logits_dev_ = core::DeviceBuffer<float>(static_cast<size_t>(m.vocab_local_));
  m.argmax_dev_ = core::DeviceBuffer<int32_t>(1);
  m.attn_positions_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(m.max_chunk_));
  m.attn_seqused_k_ = core::DeviceBuffer<int32_t>(1);
  // 3-axis mrope companion (docs/vision.md): 3*64 int32 == 768 bytes, allocated unconditionally
  // rather than lazily -- a lazy hipMalloc would have to happen on the first multimodal chunk,
  // i.e. mid-request with the device live, which is exactly what every other per-chunk buffer in
  // this class is persistent to avoid.
  m.attn_rope_pos_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(3 * m.max_chunk_));
  m.rope_pos_host_.assign(static_cast<size_t>(3 * m.max_chunk_), 0);
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
        core::DeviceBuffer<float>(static_cast<size_t>(m.draft_window_ * m.vocab_local_));
    m.verify_argmax_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(m.draft_window_));
  }
  // Tensor parallel (docs/tp.md 4.4): one {local index, value} pair per row that can be argmaxed at
  // once -- a plain decode row or a whole verify window -- and this rank's own row-summary
  // workspace (docs/tp.md 2.7), allocated here with the rest of the load-time scratch rather than
  // at the first sampled row, so no sampled path ever allocates inside a collective (6.3.7).
  if (m.comm_ != nullptr) {
    m.argmax_pair_dev_ = core::DeviceBuffer<int32_t>(static_cast<size_t>(2 * m.draft_window_));
    m.topk_lse_ws_ = core::DeviceBuffer<uint8_t>(static_cast<size_t>(r4dx_topk_lse_workspace_bytes()));
  }

  // Sampled-decode row summaries (docs/sampling.md section 8), sized for the widest thing that can
  // ever be summarised at once: one plain decode row, or a whole verify window. Unconditional and
  // ~4 KB at draft_window_==8 -- a plain sampled decode step needs these on a Model that was never
  // sized for speculation at all, and a conditional allocation would be one more way for a caller
  // to reach a null pointer.
  {
    const size_t summary_rows = static_cast<size_t>(m.draft_window_ > 1 ? m.draft_window_ : 1);
    m.summary_ids_dev_ = core::DeviceBuffer<int32_t>(summary_rows * R4DX_TOPK_LSE_K);
    m.summary_vals_dev_ = core::DeviceBuffer<float>(summary_rows * R4DX_TOPK_LSE_K);
    m.summary_lse_dev_ = core::DeviceBuffer<float>(summary_rows);
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
    if (m.container_.HasVision()) {
      // Part of `weights` above (Container::Load uploads it), broken out because it is the one
      // component a caller can turn off with a flag. This is the SUMMED tensor size, not a
      // hipMemGetInfo delta -- the measured delta is the difference between this load's own
      // `weights=` figure and a `--vision off` load's, which docs/vision.md quotes.
      std::cerr << "[r4dx::model::Model] vision tower VRAM: "
                << GiB(m.container_.Vision().bytes)
                << " GiB (already included in weights above; --vision off reclaims it)\n";
    }
    if (has_dflash && vram_dflash0.ok && vram_dflash1.ok) {
      const int64_t dflash_b =
          static_cast<int64_t>(vram_dflash0.free_bytes) - static_cast<int64_t>(vram_dflash1.free_bytes);
      std::cerr << "[r4dx::model::Model] DFlash2 drafter VRAM: " << GiB(dflash_b)
                << " GiB (weights+kv_ring+draft_scratch; already included in kv+gdn_state above)\n";
    }
  }

  return m;
}

void Model::EncodeImages(const float* pixel_values, int64_t total_patches,
                          const std::vector<vision::GridThw>& grids,
                          core::DeviceBuffer<uint16_t>* out, vision::VisionEncodeStats* stats,
                          const vision::VisionTrace* trace) {
  if (!HasVision()) {
    throw std::runtime_error(
        "Model::EncodeImages: this model has no vision tower (the container carries no vision.* "
        "tensors, or it was loaded with --vision off)");
  }
  const vision::VisionWeights& vw = container_.Vision();
  if (vw.config.out_hidden_size != Config().hidden_size) {
    throw std::runtime_error(
        "Model::EncodeImages: vision_config.out_hidden_size (" +
        std::to_string(vw.config.out_hidden_size) + ") does not match the text hidden_size (" +
        std::to_string(Config().hidden_size) +
        ") -- the merger's rows are spliced straight into the text embedding sequence, with no "
        "extra projection (docs/vision.md)");
  }
  vision_->Encode(vw, pixel_values, total_patches, grids, out, stats, trace);
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
  // Every call resets the arena before it returns, so this is a no-op after a normal call; after a
  // call that THREW mid-layer (a tensor-parallel abort, docs/tp.md 2.4, or any other exception) it
  // drops the dead call's scratch, so the next sequence starts from the same clean arena offset a
  // fresh Model does.
  arena_.Reset();

  // 3-axis mrope state (docs/vision.md): the delta and the spliced spans describe the CONVERSATION,
  // not the KV bytes, so unlike the caches above they cannot be left to self-correct by position
  // overwrite -- a stale delta would rope the next conversation's every token at the wrong
  // position while looking entirely healthy. Back to the text-only values, which is also what puts
  // every rope call site back on its pre-vision single-row path.
  mrope_active_ = false;
  mrope_delta_ = 0;
  mrope_block_.clear();
  mrope_block_base_ = 0;
  mrope_block_images_.clear();

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
  if (dflash_.has_value()) {
    dflash_->Reset();
    dflash_->SetRopeDelta(0);  // back in step with mrope_delta_ above
  }
}

// ---- 3-axis mrope plumbing (docs/vision.md "Text-side splicing") -------------------------------
// The whole conversation's rope position assignment is two facts: a running `mrope_delta_` (every
// text token at absolute sequence index `s` ropes at `s + mrope_delta_` on all three axes) and,
// while PrefillMultimodal is feeding a block, that block's own explicit [3, N] rows. There is no
// full-sequence position table: `mrope_delta_` is by construction constant over every text run
// after the last image (position_ids.cpp derives it as exactly that running-counter difference),
// so a table would be 3 MiB of host memory restating one integer.

void Model::RopePositionsHost(int64_t start, int64_t T, std::vector<int32_t>* out3) const {
  out3->assign(static_cast<size_t>(3 * T), 0);
  const int64_t n_block = static_cast<int64_t>(mrope_block_.size() / 3);
  for (int64_t t = 0; t < T; ++t) {
    const int64_t s = start + t;
    const int64_t in_block = s - mrope_block_base_;
    if (n_block > 0 && in_block >= 0 && in_block < n_block) {
      for (int axis = 0; axis < 3; ++axis) {
        (*out3)[static_cast<size_t>(axis * T + t)] =
            mrope_block_[static_cast<size_t>(axis * n_block + in_block)];
      }
    } else {
      const int32_t p = static_cast<int32_t>(s + mrope_delta_);
      (*out3)[static_cast<size_t>(t)] = p;
      (*out3)[static_cast<size_t>(T + t)] = p;
      (*out3)[static_cast<size_t>(2 * T + t)] = p;
    }
  }
}

const int32_t* Model::RopePositionsForChunk(int64_t start, int64_t T) {
  if (!mrope_active_) return nullptr;
  std::vector<int32_t> rows;
  RopePositionsHost(start, T, &rows);
  std::copy(rows.begin(), rows.end(), rope_pos_host_.begin());
  // Blocking upload, at the same point in the call and for the same reason attn_positions_' own
  // is (see RunChunk's comment): the device is idle here because the previous call synchronized.
  attn_rope_pos_.CopyFromHost(rope_pos_host_.data(), static_cast<size_t>(3 * T));
  return attn_rope_pos_.data();
}

void Model::SpliceImageEmbeddings(int64_t start, int64_t T, uint16_t* dst, int64_t hidden) {
  if (mrope_block_images_.empty()) return;
  for (const ImageSpan& sp : mrope_block_images_) {
    const int64_t span_start = mrope_block_base_ + sp.offset;  // absolute sequence index
    const int64_t lo = std::max(span_start, start);
    const int64_t hi = std::min(span_start + sp.tokens, start + T);
    if (lo >= hi) continue;
    // Both sides are contiguous row-major [rows, hidden] bf16 -- the placeholder run is contiguous
    // in the prompt and the merger's rows are contiguous in EncodeImages' output -- so one D2D
    // copy per (span, chunk) intersection, no kernel and no per-row loop.
    const int64_t rows = hi - lo;
    R4DX_HIP_CHECK(hipMemcpyAsync(dst + (lo - start) * hidden,
                                   sp.embeds + (lo - span_start) * hidden,
                                   static_cast<size_t>(rows * hidden) * sizeof(uint16_t),
                                   hipMemcpyDeviceToDevice, stream_.get()));
  }
}

std::vector<float> Model::RunChunk(const std::vector<int32_t>& token_ids, bool is_prefill_path,
                                    bool want_logits, int32_t* greedy_token_out,
                                    const SummaryRequest* summary_out) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  if (T < 1 || T > max_chunk_) {
    throw std::runtime_error("Model::RunChunk: token_ids.size() must be in [1, " +
                              std::to_string(max_chunk_) + "]");
  }
  // Tensor parallel, H1 (docs/tp.md 6.2): every rank must be about to run the same chunk in the
  // same mode at the same position, and the group must be healthy, before this chunk's all-reduces
  // are enqueued. The device is idle here (the previous call ended synchronized).
  if (comm_ != nullptr) {
    const uint64_t kind = kLockstepRunChunk | (is_prefill_path ? 1u : 0u) |
                          (want_logits ? 2u : 0u) | (greedy_token_out != nullptr ? 4u : 0u) |
                          (summary_out != nullptr ? 8u : 0u);
    const uint64_t fingerprint[4] = {kind, static_cast<uint64_t>(T), static_cast<uint64_t>(pos_),
                                     Fnv1a64(token_ids)};
    comm_->CheckLockstep(fingerprint);
    comm_->CheckHealthy();
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
  // Image splice (docs/vision.md): whichever pending span rows fall in this chunk replace the
  // embed_tokens lookup that just ran for their placeholder token ids. Enqueued on stream_ after
  // the gather, so it is ordered behind it; a no-op (not even a loop iteration) outside a
  // PrefillMultimodal call carrying images.
  SpliceImageEmbeddings(pos_, T, buf_a_.data(), hidden);

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
  // The 3-axis rope rows for the same window -- nullptr (and no upload at all) unless an image has
  // been spliced into this conversation, which is what keeps a text-only run byte-identical.
  const int32_t* rope_pos3 = RopePositionsForChunk(pos_, T);

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
      GdnLayer layer(cfg, lw.input_layernorm, *lw.gdn, comm_);
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
      attention::AttentionLayer layer(MakeAttnConfig(cfg, comm_));

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
                    buf_normed_pre_.data(), buf_normed_pre_scale_.data(), rope_pos3);
      std::swap(cur, other);
    }

    Mlp mlp(cfg, lw.post_attention_layernorm, lw.mlp, comm_);
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
    // MTP's own attention layer ropes at the same 3-axis positions the backbone just did -- and
    // unlike its draft loop, these positions are INSIDE the prompt, so they can land on image rows
    // where the three axes genuinely differ (docs/vision.md). Built on the host per call because
    // the two PrimeKv calls below cover different, non-contiguous position windows.
    std::vector<int32_t> prime_rope3;
    const int32_t* boundary_rope3 = nullptr;
    const int32_t* within_rope3 = nullptr;
    std::vector<int32_t> boundary_rope3_rows;
    if (mrope_active_) {
      if (mtp_seed_valid_) {
        RopePositionsHost(pos_ - 1, 1, &boundary_rope3_rows);
        boundary_rope3 = boundary_rope3_rows.data();
      }
      if (T > 1) {
        RopePositionsHost(pos_, T - 1, &prime_rope3);
        within_rope3 = prime_rope3.data();
      }
    }
    if (mtp_seed_valid_) {
      // The ONE position left dangling by the previous RunChunk/DecodeStepMtpGreedy call: h_i =
      // that call's own last-row hidden state (mtp_seed_hidden_), t_{i+1} = THIS call's own first
      // input token.
      // host_staging_offset=0: this is the FIRST of up to two PrimeKv calls in this RunChunk
      // invocation (see mtp_head.h's PrimeKv doc comment for why the two calls need disjoint
      // offsets into MtpHead's own pinned host scratch).
      mtp_->PrimeKv(stream_, arena_, cfg, container_.Mtp(), pos_ - 1, mtp_seed_hidden_.data(),
                    {token_ids[0]}, container_.EmbedTokensHost(), cfg.vocab_size,
                    /*host_staging_offset=*/0, boundary_rope3);
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
                    container_.EmbedTokensHost(), cfg.vocab_size, /*host_staging_offset=*/1,
                    within_rope3);
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
    if (greedy_token_out != nullptr && comm_ != nullptr) {
      // Tensor parallel (docs/tp.md 7.3): argmax THIS rank's vocab shard, keeping the winning value
      // too; the host merges the per-rank pairs below.
      r4dx_argmax_val_f32(reinterpret_cast<int64_t>(logits_dev_.data()),
                           reinterpret_cast<int64_t>(argmax_pair_dev_.data()),
                           reinterpret_cast<int64_t>(argmax_pair_dev_.data() + 1), vocab_local_,
                           reinterpret_cast<int64_t>(stream_.get()));
    } else if (greedy_token_out != nullptr) {
      r4dx_argmax_f32(reinterpret_cast<int64_t>(logits_dev_.data()),
                       reinterpret_cast<int64_t>(argmax_dev_.data()), cfg.vocab_size,
                       reinterpret_cast<int64_t>(stream_.get()));
    } else if (summary_out != nullptr) {
      // Sampled path (docs/sampling.md section 8): the same "argmax it on device while it is still
      // hot" trick one step further -- summarise the row on device so the D2H below is 516 bytes
      // instead of vocab*4, and the host sampler gets to skip its own O(vocab) pass entirely.
      LaunchRowSummaries(logits_dev_.data(), /*rows=*/1, summary_out->inv_temperature);
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
  if (want_logits && comm_ != nullptr) {
    // Tensor parallel, H3 (docs/tp.md 6.2, 7.3-7.5): merge the vocab shards on the host -- the
    // greedy pair (8 B per rank), the row summary (536 B per rank), or the full row gathered in
    // global id order. Either way every rank ends with the same answer.
    if (greedy_token_out != nullptr) {
      *greedy_token_out = MergeGreedyPair(argmax_pair_dev_.data());
    } else if (summary_out != nullptr) {
      FetchRowSummaries(/*rows=*/1, summary_out->inv_temperature, round_summaries_);
      std::vector<tp::ArgmaxPair> no_pairs;
      MergeShardResults(no_pairs, &round_summaries_);
      *summary_out->out = round_summaries_[0];
    } else {
      logits.resize(static_cast<size_t>(cfg.vocab_size));
      GatherVocabRow(logits_dev_.data(), logits.data());
    }
  } else if (want_logits) {
    if (greedy_token_out != nullptr) {
      argmax_dev_.CopyToHost(greedy_token_out, 1);
    } else if (summary_out != nullptr) {
      // round_summaries_ is this Model's one summary staging vector; a plain decode step and a
      // verify round are never in flight at the same time (single sequence, single worker thread --
      // this class's own SCOPE comment), so they share it rather than allocating per token.
      FetchRowSummaries(/*rows=*/1, summary_out->inv_temperature, round_summaries_);
      *summary_out->out = round_summaries_[0];
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
    // The drafter ropes on the mrope TEMPORAL axis only (docs/dflash2.md "RoPE": sections
    // [64,0,0,0]) while its ring stays keyed on the sequence position -- so it gets the t row of
    // this chunk's rows, not the whole [3, T] block. `rope_delta_` covers its own draft blocks,
    // which are always past the prompt; injection can straddle an image, so it gets the real row.
    std::vector<int32_t> inject_rope3;
    const int32_t* inject_rope_t = nullptr;
    if (mrope_active_) {
      RopePositionsHost(pos_, T, &inject_rope3);
      inject_rope_t = inject_rope3.data();  // row 0 of [3, T] is the temporal row
    }
    dflash_->InjectFeatures(stream_, arena_, dflash_features_dev_.data(), T, pos_, inject_rope_t);
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

std::vector<float> Model::PrefillMultimodal(const std::vector<int32_t>& token_ids,
                                              const std::vector<ImageSpan>& images,
                                              const std::function<void()>& on_chunk_captured,
                                              std::vector<int32_t>* rope_rows_out) {
  if (token_ids.empty()) throw std::runtime_error("Model::PrefillMultimodal: token_ids is empty");
  // SpliceImageEmbeddings copies device to device; host-resident rows (TpModel's ImageRows) get
  // their H2D splice in docs/tp.md P5. Until then refuse them rather than D2D from a host pointer.
  for (const ImageSpan& sp : images) {
    if (sp.embeds_on_host) {
      throw std::invalid_argument("Model::PrefillMultimodal: host-resident image rows (ImageSpan::embeds_on_host) "
                                  "are not supported yet (docs/tp.md P5); pass device rows");
    }
  }
  if (images.empty() && !mrope_active_) {
    // Text-only, and nothing has ever diverged -- the pre-vision path, byte for byte. The rows a
    // diagnostic caller asked for are simply the sequence indices on all three axes.
    if (rope_rows_out != nullptr) {
      RopePositionsHost(pos_, static_cast<int64_t>(token_ids.size()), rope_rows_out);
    }
    return Prefill(token_ids, on_chunk_captured);
  }

  const int64_t seq_len = static_cast<int64_t>(token_ids.size());
  if (!images.empty() && !container_.HasVision()) {
    throw std::runtime_error(
        "Model::PrefillMultimodal: image spans supplied but this model has no vision tower loaded "
        "(--vision off, or a container with no vision.* tensors)");
  }
  // Only read when `images` is non-empty, which the check above already made imply HasVision().
  const int merge_size =
      container_.HasVision() ? static_cast<int>(container_.Vision().config.spatial_merge_size) : 2;
  const int32_t image_token_id = static_cast<int32_t>(container_.ImageTokenId());

  // ---- validate the spans, and derive the mm_token_type_ids the position walk needs -------------
  // Derived from the SPANS, not from a scan for image_token_id: the spans are the contract (they
  // carry the grid and the device rows), and checking that the tokens under each span really are
  // the placeholder id catches a caller whose offsets have drifted from its own rendered prompt --
  // which is the failure that would otherwise splice real embeddings over real text.
  std::vector<uint8_t> mm_ids(static_cast<size_t>(seq_len), vision::kMmTokenTypeText);
  std::vector<vision::GridThw> grids;
  grids.reserve(images.size());
  int64_t prev_end = 0;
  for (const ImageSpan& sp : images) {
    if (sp.offset < prev_end || sp.tokens <= 0 || sp.offset + sp.tokens > seq_len) {
      throw std::runtime_error(
          "Model::PrefillMultimodal: image spans must be sorted, non-overlapping and inside "
          "token_ids; got offset=" + std::to_string(sp.offset) + " tokens=" +
          std::to_string(sp.tokens) + " against a " + std::to_string(seq_len) + "-token prompt");
    }
    if (sp.embeds == nullptr) {
      throw std::runtime_error("Model::PrefillMultimodal: image span at offset " +
                                std::to_string(sp.offset) + " has no device embeddings");
    }
    if (sp.grid.MergedTokenCount(merge_size) != sp.tokens) {
      throw std::runtime_error(
          "Model::PrefillMultimodal: image span at offset " + std::to_string(sp.offset) +
          " claims " + std::to_string(sp.tokens) + " tokens but its grid merges to " +
          std::to_string(sp.grid.MergedTokenCount(merge_size)));
    }
    for (int64_t i = sp.offset; i < sp.offset + sp.tokens; ++i) {
      if (token_ids[static_cast<size_t>(i)] != image_token_id) {
        throw std::runtime_error(
            "Model::PrefillMultimodal: token at index " + std::to_string(i) +
            " is inside an image span but is not the image placeholder id " +
            std::to_string(image_token_id) + " (got " +
            std::to_string(token_ids[static_cast<size_t>(i)]) + ")");
      }
      mm_ids[static_cast<size_t>(i)] = vision::kMmTokenTypeImage;
    }
    grids.push_back(sp.grid);
    prev_end = sp.offset + sp.tokens;
  }

  // ---- this block's 3-axis rows + the delta every later step will rope at -----------------------
  const vision::MropePositions mp = vision::BuildMropePositionIds(
      mm_ids, grids, merge_size, /*seq_start=*/pos_, /*mrope_start=*/pos_ + mrope_delta_);

  mrope_block_ = mp.position_ids;
  mrope_block_base_ = pos_;
  mrope_block_images_ = images;
  mrope_active_ = true;
  // Cleared on EVERY exit, including an exception out of a RunChunk below: leaving a stale block
  // table behind would silently rope a LATER call's tokens from this call's rows.
  struct BlockGuard {
    Model* m;
    ~BlockGuard() {
      m->mrope_block_.clear();
      m->mrope_block_images_.clear();
      m->mrope_block_base_ = 0;
    }
  } guard{this};

  if (rope_rows_out != nullptr) *rope_rows_out = mrope_block_;

  mtp_num_accepted_valid_ = false;  // same reasoning as Prefill()'s own reset
  std::vector<float> logits;
  for (size_t off = 0; off < token_ids.size(); off += static_cast<size_t>(max_chunk_)) {
    const size_t n = std::min(static_cast<size_t>(max_chunk_), token_ids.size() - off);
    const std::vector<int32_t> chunk(token_ids.begin() + static_cast<ptrdiff_t>(off),
                                      token_ids.begin() + static_cast<ptrdiff_t>(off + n));
    const bool is_last_chunk = (off + n) == token_ids.size();
    std::vector<float> chunk_logits =
        RunChunk(chunk, /*is_prefill_path=*/true, /*want_logits=*/is_last_chunk);
    if (on_chunk_captured) on_chunk_captured();
    if (is_last_chunk) logits = std::move(chunk_logits);
  }
  // Only now: until this point the block's own explicit rows are what every chunk read, and the
  // NEW delta describes only positions at or after `pos_` (which the loop just advanced past the
  // whole block).
  mrope_delta_ = mp.mrope_position_delta;
  // The drafter ropes its own blocks on the temporal axis at `sequence index + delta`
  // (docs/dflash2.md "RoPE" -- sections [64,0,0,0]); its ring slots stay in sequence space.
  if (dflash_.has_value()) dflash_->SetRopeDelta(mrope_delta_);
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

// ---- sampled decode (docs/sampling.md sections 8-9, Milestone 6 stage S2) -----------------------

// Review fix (2026-09-21, minor): kRowSummaryLseRelTol's documented "~5x headroom"
// (summary_sampler.hpp) is conditional on |max_logit/temperature| staying under ~1e4 -- measured
// S_full error grows from 1.7e-6 (T=1.0) to 1.5e-3 (T=0.001, already past the 1e-3 band) on this
// model's logit scale. Below kMinSummaryTemperature the summary path is skipped in favor of the
// full-logits path (still exactly one draw, still the canonical sampler -- see DecodeStepSampled's
// existing inv_t==0.0f branch, which this reuses), so the band summary_sampler.hpp documents is
// never exercised outside where it was measured to hold. 0.01 leaves >2x margin below the measured
// safe point (T=0.005 -> 2.4e-4 error) and real chat traffic never asks for T this low.
inline constexpr float kMinSummaryTemperature = 0.01f;

float Model::SummaryInvTemperature(const kernels::SampleParams& params) {
  if (!(params.temperature > 0.0f)) return 0.0f;  // greedy -- no sampling, no summary
  if (params.temperature < kMinSummaryTemperature) return 0.0f;  // see kMinSummaryTemperature above
  const float inv = 1.0f / params.temperature;
  // The device kernel REJECTS a non-finite or non-positive inv_temperature by throwing out of an
  // extern "C" entry point, which under this build's /EHsc is a fail-fast rather than an exception
  // the caller could handle. Screen it here instead: a temperature below ~1e-38 overflows the
  // reciprocal, and such a request simply takes the full-logits path (it is greedy in all but name).
  // (kMinSummaryTemperature above already catches every temperature that could reach this, but the
  // check is kept as a second, independent guarantee against the device kernel's own precondition.)
  if (!(inv > 0.0f) || !(inv < std::numeric_limits<float>::infinity())) return 0.0f;
  return inv;
}

void Model::LaunchRowSummaries(const float* logits_dev, int64_t rows, float inv_temperature) {
  if (rows <= 0) return;
  // The device rows are this rank's [vocab_local_] shard (docs/tp.md 7.2's rule); at TP=1
  // vocab_local_ == Config().vocab_size.
  const int64_t vocab = vocab_local_;
  const int64_t capacity = static_cast<int64_t>(summary_lse_dev_.size());
  if (rows > capacity) {
    throw std::runtime_error("Model::LaunchRowSummaries: rows (" + std::to_string(rows) +
                              ") exceeds the summary scratch this Model was sized for (" +
                              std::to_string(capacity) + ")");
  }
  if (!(inv_temperature > 0.0f) ||
      !(inv_temperature < std::numeric_limits<float>::infinity())) {
    throw std::runtime_error("Model::LaunchRowSummaries: inv_temperature must be finite and > 0");
  }
  // r4dx_topk_lse_f32 takes at most 8 rows per call (kernels.h) and keeps its partials in a
  // module-scope device scratch, so only ONE call may be in flight. Both constraints are satisfied
  // by issuing <=8-row calls back to back on stream_: stream order makes call n+1's first kernel
  // wait for call n's merge kernel, which is exactly the serialisation that scratch needs. A K=16
  // MTP window (17 rows) is the only shape that needs more than one call today.
  constexpr int64_t kMaxRowsPerCall = 8;
  if (comm_ != nullptr && topk_lse_ws_.empty()) {
    throw std::logic_error("Model::LaunchRowSummaries: tensor-parallel rank has no topk_lse workspace");
  }
  for (int64_t off = 0; off < rows; off += kMaxRowsPerCall) {
    const int64_t n = std::min(kMaxRowsPerCall, rows - off);
    if (comm_ != nullptr) {
      // Tensor parallel (docs/tp.md 2.7): this rank's own workspace, never the kernels' module
      // scratch -- under emulation the other rank summarizes concurrently on the same device.
      r4dx_topk_lse_f32_ws(reinterpret_cast<int64_t>(logits_dev + off * vocab),
                           reinterpret_cast<int64_t>(summary_ids_dev_.data() + off * R4DX_TOPK_LSE_K),
                           reinterpret_cast<int64_t>(summary_vals_dev_.data() + off * R4DX_TOPK_LSE_K),
                           reinterpret_cast<int64_t>(summary_lse_dev_.data() + off),
                           static_cast<int>(n), vocab, inv_temperature,
                           reinterpret_cast<int64_t>(stream_.get()),
                           reinterpret_cast<int64_t>(topk_lse_ws_.data()));
      continue;
    }
    r4dx_topk_lse_f32(reinterpret_cast<int64_t>(logits_dev + off * vocab),
                       reinterpret_cast<int64_t>(summary_ids_dev_.data() + off * R4DX_TOPK_LSE_K),
                       reinterpret_cast<int64_t>(summary_vals_dev_.data() + off * R4DX_TOPK_LSE_K),
                       reinterpret_cast<int64_t>(summary_lse_dev_.data() + off),
                       static_cast<int>(n), vocab, inv_temperature,
                       reinterpret_cast<int64_t>(stream_.get()));
  }
}

void Model::FetchRowSummaries(int64_t rows, float inv_temperature,
                               std::vector<kernels::RowSummary>& out) {
  out.clear();
  if (rows <= 0) return;
  // The width the device summarized: this rank's shard under TP (ids are LOCAL until
  // MergeShardResults makes the merged summary global), the whole vocabulary at TP=1.
  const int64_t vocab = vocab_local_;
  const size_t k_total = static_cast<size_t>(rows) * R4DX_TOPK_LSE_K;
  summary_ids_host_.resize(k_total);
  summary_vals_host_.resize(k_total);
  summary_lse_host_.resize(static_cast<size_t>(rows));
  // Plain (blocking, null-stream) D2H copies, so the caller must already have synchronised stream_
  // -- the same rule RunChunk/VerifyWindow's own logits readback follows, for the same reason
  // (stream_ is hipStreamNonBlocking). rows*(64*8+4) bytes in total: 516 per row.
  summary_ids_dev_.CopyToHost(summary_ids_host_.data(), k_total);
  summary_vals_dev_.CopyToHost(summary_vals_host_.data(), k_total);
  summary_lse_dev_.CopyToHost(summary_lse_host_.data(), static_cast<size_t>(rows));

  out.resize(static_cast<size_t>(rows));
  for (int64_t r = 0; r < rows; ++r) {
    kernels::RowSummary& s = out[static_cast<size_t>(r)];
    s.k = R4DX_TOPK_LSE_K;
    s.vocab = vocab;
    s.inv_temperature = inv_temperature;
    s.lse = summary_lse_host_[static_cast<size_t>(r)];
    const size_t base = static_cast<size_t>(r) * R4DX_TOPK_LSE_K;
    for (int j = 0; j < R4DX_TOPK_LSE_K; ++j) {
      s.ids[j] = summary_ids_host_[base + static_cast<size_t>(j)];
      s.vals[j] = summary_vals_host_[base + static_cast<size_t>(j)];
    }
  }
}

int32_t Model::SampleVerifyRow(int64_t row, const std::vector<kernels::RowSummary>& summaries,
                                const kernels::SampleParams& params, std::mt19937_64& rng) {
  // ONE draw per emitted token, taken BEFORE either branch so that whether the summary happened to
  // resolve this row can never change the generator's state (docs/sampling.md section 2's
  // one-draw-per-token invariant, which is what makes a speculative round's emitted sequence equal
  // plain sampled decode's token for token).
  const double u = kernels::DrawUniform01(rng);
  const kernels::SummarySampleResult r =
      kernels::SampleFromSummary(summaries[static_cast<size_t>(row)], params, u);
  if (r.resolved) return r.token;
  ++sampled_fallback_rows_;
  ReadVerifyLogitsRow(row, sampled_row_scratch_);
  return kernels::SampleCanonical(sampled_row_scratch_.data(), container_.Config().vocab_size,
                                   params, u);
}

int32_t Model::DecodeStepSampled(int32_t token_id, const kernels::SampleParams& params,
                                  std::mt19937_64& rng) {
  const float inv_t = SummaryInvTemperature(params);
  if (params.temperature <= 0.0f) {
    // Design point E: a greedy request keeps the pre-S2 code path exactly, down to consuming no
    // draw at all -- so switching a greedy caller from DecodeStepGreedy to this method changes
    // nothing, and a caller alternating the two keeps one generator stream.
    return DecodeStepGreedy(token_id);
  }
  // Tensor parallel (docs/tp.md 7.4): the summary RunChunk returns is already the MERGED full-row
  // summary, the full-logits paths below gather the full row (GatherVocabRow), and the one draw `u`
  // comes from an rng every rank holds an identical copy of (TpModel, 2.3) -- so every rank makes
  // the same fallback decision and emits the same token.
  if (inv_t == 0.0f) {
    // Degenerate temperature (see SummaryInvTemperature): no device summary, full-vocab path. Still
    // exactly one draw, still the canonical sampler, so the emitted token is the same as any other
    // path would produce for this u.
    const std::vector<float> logits = DecodeStep(token_id);
    return kernels::SampleCanonical(logits.data(), container_.Config().vocab_size, params,
                                     kernels::DrawUniform01(rng));
  }

  kernels::RowSummary summary;
  const SummaryRequest req{inv_t, &summary};
  RunChunk({token_id}, /*is_prefill_path=*/false, /*want_logits=*/true,
           /*greedy_token_out=*/nullptr, &req);

  const double u = kernels::DrawUniform01(rng);
  const kernels::SummarySampleResult r = kernels::SampleFromSummary(summary, params, u);
  if (r.resolved) return r.token;
  // Row-granular fallback: logits_dev_ still holds this step's own full fp32 row (nothing has
  // touched it since RunChunk's lm_head, and RunChunk returned with the device idle), so this is a
  // plain 993 KB D2H -- never a second lm_head pass.
  ++sampled_fallback_rows_;
  const int64_t vocab = container_.Config().vocab_size;
  sampled_row_scratch_.resize(static_cast<size_t>(vocab));
  if (comm_ != nullptr) {
    // Tensor parallel, H5 (docs/tp.md 7.4 "unresolved row"): D2H this rank's shard and gather the
    // full row in global id order; SampleCanonical over it with the SAME u.
    GatherVocabRow(logits_dev_.data(), sampled_row_scratch_.data());
  } else {
    logits_dev_.CopyToHost(sampled_row_scratch_.data(), static_cast<size_t>(vocab));
  }
  return kernels::SampleCanonical(sampled_row_scratch_.data(), vocab, params, u);
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
  RequireNotTp("DecodeStepProfiled (profiling is not supported under tensor parallelism)");
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
  const int32_t* rope_pos3 = RopePositionsForChunk(pos_, 1);  // nullptr for a text-only run

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
      GdnLayer layer(cfg, lw.input_layernorm, *lw.gdn, comm_);
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
      attention::AttentionLayer layer(MakeAttnConfig(cfg, comm_));

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
                    buf_normed_pre_.data(), buf_normed_pre_scale_.data(), rope_pos3);
      std::swap(cur, other);
    }

    {
      Mlp mlp(cfg, lw.post_attention_layernorm, lw.mlp, comm_);
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
  RequireNotTp("PrefillProfiled (profiling is not supported under tensor parallelism)");
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
    const int32_t* rope_pos3 = RopePositionsForChunk(pos_, T);  // nullptr for a text-only run

    uint16_t* cur = buf_a_.data();
    uint16_t* other = buf_b_.data();
    const uint16_t* normed_in = nullptr;  // R3 fusion, see RunChunk's identical pattern
    int normed_in_epilogue = r4dx_epilogue_none;  // R2/P2, see RunChunk's identical pattern

    for (int64_t i = 0; i < num_layers; ++i) {
      const LayerWeights& lw = container_.Layer(i);
      const bool has_next_layer = (i + 1 < num_layers);
      const uint16_t* mlp_norm_weight = lw.post_attention_layernorm.data();

      if (cfg.IsGdnLayer(i)) {
        GdnLayer layer(cfg, lw.input_layernorm, *lw.gdn, comm_);
        GdnLayerParams p;
        p.slot = gdn_states_[static_cast<size_t>(i)]->SlotForSeq(0);
        p.is_prefill = true;
        p.has_init = has_init;
        layer.Forward(stream_, arena_, *gdn_states_[static_cast<size_t>(i)], gdn_control_, cur,
                      cur, T, p, normed_in, mlp_norm_weight, buf_normed_.data(), &acc,
                      normed_in_epilogue, buf_normed_pre_.data(), buf_normed_pre_scale_.data(),
                      body_epilogue_, buf_normed_pre_.data(), buf_normed_pre_scale_.data());
      } else {
        attention::AttentionLayer layer(MakeAttnConfig(cfg, comm_));

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
                      buf_normed_pre_scale_.data(), rope_pos3);
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
        Mlp mlp(cfg, lw.post_attention_layernorm, lw.mlp, comm_);
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
                                          std::vector<float>* logits_out,
                                          std::vector<kernels::RowSummary>* summaries_out,
                                          float summary_inv_temperature) {
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
  // Tensor parallel, H2 (docs/tp.md 6.2): every rank must be about to verify the same window in the
  // same mode at the same position, and the group must be healthy, before this window's all-reduces
  // are enqueued. The device is idle here (the previous call ended synchronized).
  if (comm_ != nullptr) {
    const uint64_t kind =
        kLockstepVerify | (logits_out != nullptr ? 1u : 0u) | (summaries_out != nullptr ? 2u : 0u);
    const uint64_t fingerprint[4] = {kind, static_cast<uint64_t>(T), static_cast<uint64_t>(pos_),
                                     Fnv1a64(candidates)};
    comm_->CheckLockstep(fingerprint);
    comm_->CheckHealthy();
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
  // The candidate rows are always past the prompt, so their 3-axis rope positions are just
  // `pos_ + t + mrope_delta_` on all three axes -- but they still have to be BUILT, because
  // `attn_positions_` above is the KV slot mapping and must stay the plain sequence index.
  const int32_t* rope_pos3 = RopePositionsForChunk(pos_, T);  // nullptr for a text-only run

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
      GdnLayer layer(cfg, lw.input_layernorm, *lw.gdn, comm_);
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
      attention::AttentionLayer layer(MakeAttnConfig(cfg, comm_));

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
                    buf_normed_pre_.data(), buf_normed_pre_scale_.data(), rope_pos3);
      std::swap(cur, other);
    }

    Mlp mlp(cfg, lw.post_attention_layernorm, lw.mlp, comm_);
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
    if (comm_ != nullptr) {
      // Tensor parallel (docs/tp.md 7.6): argmax this rank's [vocab_local_] shard of row t, keeping
      // the winning value; the host merges the per-rank pairs below.
      r4dx_argmax_val_f32(reinterpret_cast<int64_t>(verify_logits_dev_.data() + t * vocab_local_),
                           reinterpret_cast<int64_t>(argmax_pair_dev_.data() + 2 * t),
                           reinterpret_cast<int64_t>(argmax_pair_dev_.data() + 2 * t + 1),
                           vocab_local_, reinterpret_cast<int64_t>(stream_.get()));
      continue;
    }
    r4dx_argmax_f32(
        reinterpret_cast<int64_t>(verify_logits_dev_.data() + t * cfg.vocab_size),
        reinterpret_cast<int64_t>(verify_argmax_dev_.data() + t), cfg.vocab_size,
        reinterpret_cast<int64_t>(stream_.get()));
  }
  // Sampled rounds (docs/sampling.md section 9): summarise every row on device too, so the round's
  // whole D2H is T*516 bytes rather than T*~993 KB. Enqueued on the same stream as the argmaxes
  // above and read back below with them; never launched at all for a greedy round, which therefore
  // stays byte-identical (same kernels, same order, same launch count).
  if (summaries_out != nullptr) {
    LaunchRowSummaries(verify_logits_dev_.data(), T, summary_inv_temperature);
  }
  arena_.Reset();

  mtp_last_hidden_ = cur;  // valid until the next RunChunk/VerifyWindow's EmbedTokens overwrites it

  stream_.Synchronize();  // same hazard class as RunChunk's own D2H -- see that method's comment

  std::vector<int32_t> preds(static_cast<size_t>(T));
  if (comm_ != nullptr) {
    // Tensor parallel, H4 (docs/tp.md 6.2, 7.6): the T greedy pairs and (sampled) the T row
    // summaries of this rank's shard, merged across ranks in ONE host all-gather -- after which
    // `preds` and the summaries are global and identical on every rank, so the unchanged walk in
    // VerifyAndResolveRound runs identically everywhere.
    std::vector<tp::ArgmaxPair> pairs(static_cast<size_t>(T));
    R4DX_HIP_CHECK(hipMemcpy(pairs.data(), argmax_pair_dev_.data(), pairs.size() * sizeof(tp::ArgmaxPair),
                             hipMemcpyDeviceToHost));
    if (summaries_out != nullptr) FetchRowSummaries(T, summary_inv_temperature, *summaries_out);
    MergeShardResults(pairs, summaries_out);
    for (int64_t t = 0; t < T; ++t) preds[static_cast<size_t>(t)] = pairs[static_cast<size_t>(t)].idx;
    if (logits_out != nullptr) {
      // The full [T, vocab_size] rows in global id order: one gather per row (T x 496,640 B per
      // rank). A production path -- VerifyAndResolveRound asks for it on every sampled round below
      // kMinSummaryTemperature.
      logits_out->resize(static_cast<size_t>(T * cfg.vocab_size));
      for (int64_t t = 0; t < T; ++t) {
        GatherVocabRow(verify_logits_dev_.data() + t * vocab_local_, logits_out->data() + t * cfg.vocab_size);
      }
    }
    return preds;
  }
  verify_argmax_dev_.CopyToHost(preds.data(), preds.size());
  if (logits_out != nullptr) {
    logits_out->resize(static_cast<size_t>(T * cfg.vocab_size));
    verify_logits_dev_.CopyToHost(logits_out->data(), logits_out->size());
  }
  if (summaries_out != nullptr) {
    FetchRowSummaries(T, summary_inv_temperature, *summaries_out);
  }
  return preds;
}

void Model::ReadVerifyLogitsRow(int64_t row, std::vector<float>& out) const {
  if (draft_window_ <= 1) {
    throw std::runtime_error(
        "Model::ReadVerifyLogitsRow: this Model was not sized for speculative verification");
  }
  if (row < 0 || row >= draft_window_) {
    throw std::runtime_error("Model::ReadVerifyLogitsRow: row must be in [0, " +
                              std::to_string(draft_window_) + ")");
  }
  const int64_t vocab = container_.Config().vocab_size;
  out.resize(static_cast<size_t>(vocab));
  if (comm_ != nullptr) {
    // Tensor parallel, H5 (docs/tp.md 7.5): this rank's [vocab_local_] shard of row `row`, gathered
    // into the full row in global id order. Collective: every rank reads the same row (the caller's
    // decision comes from merged, replicated values -- SampleVerifyRow's unresolved summary).
    GatherVocabRow(verify_logits_dev_.data() + row * vocab_local_, out.data());
    return;
  }
  // Plain blocking D2H of ONE row out of the [draft_window_, vocab] verify buffer. Safe without a
  // sync of its own: VerifyWindow ends with stream_.Synchronize() and nothing between it and this
  // call (the host-side acceptance walk) enqueues device work -- the same "device is idle here"
  // invariant every other plain hipMemcpy in this file relies on.
  R4DX_HIP_CHECK(hipMemcpy(out.data(), verify_logits_dev_.data() + row * vocab,
                            static_cast<size_t>(vocab) * sizeof(float), hipMemcpyDeviceToHost));
}

// THE shared verify+acceptance middle of every speculative round -- see model.h for the contract
// and for why greedy and sampled rounds MUST come through here rather than through two loops that
// could drift apart.
std::vector<int32_t> Model::VerifyAndResolveRound(int32_t anchor,
                                                   const std::vector<int32_t>& drafts,
                                                   const kernels::SampleParams* params,
                                                   std::mt19937_64* rng,
                                                   int64_t* num_accepted_out) {
  std::vector<int32_t> candidates;
  candidates.reserve(drafts.size() + 1);
  candidates.push_back(anchor);
  candidates.insert(candidates.end(), drafts.begin(), drafts.end());

  const bool sampled = (params != nullptr);
  const float inv_t = sampled ? SummaryInvTemperature(*params) : 0.0f;
  // A sampled round whose temperature has no usable reciprocal (see SummaryInvTemperature) still
  // has to produce the canonical token, so it asks for the full window's logits instead of
  // summaries and samples from those rows directly. Vanishingly rare; correctness, not speed.
  const bool use_summaries = sampled && inv_t > 0.0f;
  std::vector<float>* logits_out = nullptr;
  std::vector<float> full_logits;
  if (sampled && !use_summaries) logits_out = &full_logits;

  const std::vector<int32_t> preds =
      VerifyWindow(candidates, logits_out, use_summaries ? &round_summaries_ : nullptr,
                    use_summaries ? inv_t : 1.0f);

  const int64_t vocab = container_.Config().vocab_size;
  const int64_t m = static_cast<int64_t>(drafts.size());
  std::vector<int32_t> round;
  round.reserve(drafts.size() + 1);
  int64_t i = 0;
  for (;; ++i) {
    int32_t tok;
    if (!sampled) {
      // Greedy: the row's own argmax. The walk below then reproduces exactly the pre-S2
      // "longest confirmed draft prefix, then the correction/bonus token" result.
      tok = preds[static_cast<size_t>(i)];
    } else if (use_summaries) {
      tok = SampleVerifyRow(i, round_summaries_, *params, *rng);
    } else {
      tok = kernels::SampleCanonical(full_logits.data() + i * vocab, vocab, *params,
                                      kernels::DrawUniform01(*rng));
    }
    round.push_back(tok);
    // Accept draft i (== candidates[i+1]) iff this row emitted exactly it; otherwise this token is
    // the round's last and everything after it in the window is discarded.
    if (i < m && tok == drafts[static_cast<size_t>(i)]) continue;
    break;
  }
  *num_accepted_out = i;  // == round.size() - 1
  return round;
}

void Model::CommitVerifiedWindow(int64_t num_committed) {
  // Tensor parallel (docs/tp.md 7.6): no collective here -- every rank commits the same count,
  // computed from the merged (replicated) verify results.
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
  return DecodeStepMtpImpl(token_id, k, /*params=*/nullptr, /*rng=*/nullptr);
}

std::vector<int32_t> Model::DecodeStepMtpSampled(int32_t token_id, int64_t k,
                                                  const kernels::SampleParams& params,
                                                  std::mt19937_64& rng) {
  // Design point E: greedy requests keep the greedy path exactly, including consuming no draw.
  if (params.temperature <= 0.0f) return DecodeStepMtpImpl(token_id, k, nullptr, nullptr);
  return DecodeStepMtpImpl(token_id, k, &params, &rng);
}

std::vector<int32_t> Model::DecodeStepMtpImpl(int32_t token_id, int64_t k,
                                               const kernels::SampleParams* params,
                                               std::mt19937_64* rng) {
  RequireNotTp("DecodeStepMtp{Greedy,Sampled} (MTP, docs/tp.md 8.1: P5)");
  if (!mtp_) {
    throw std::runtime_error(
        "Model::DecodeStepMtp{Greedy,Sampled}: MTP is not enabled on this Model (Load() with "
        "ModelOptions::mtp_draft_k > 0 and a container that has mtp.* weights)");
  }
  if (k < 0 || k > mtp_draft_k_) {
    throw std::runtime_error("Model::DecodeStepMtp{Greedy,Sampled}: k must be in [0, " +
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
                          /*base_pos=*/pos_ - 1, mtp_draft_reduced_vocab_, mrope_active_,
                          mrope_delta_);
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

  // Verify [token_id, d1..dk] in one pass and resolve the round -- greedy (argmax acceptance) or
  // sampled (sample-and-match), in the one shared implementation. Commits nothing.
  int64_t num_accepted_drafts = 0;
  std::vector<int32_t> result =
      VerifyAndResolveRound(token_id, drafts, params, rng, &num_accepted_drafts);
  const int64_t num_committed = num_accepted_drafts + 1;  // +1 for token_id itself

  // Thread this round's acceptance count into the NEXT GDN decode/verify call (gdn_state.h's file
  // comment) and advance pos_ by exactly what was committed -- NOT by candidates.size(), which
  // would silently accept every draft regardless of whether the real model agreed. This is exactly
  // CommitVerifiedWindow's body (that method is this block, factored out for a non-MTP drafter --
  // model.h), so MTP and DFlash2 cannot commit differently for the same accepted count.
  CommitVerifiedWindow(num_committed);

  // Reseed MTP's h_seed from the row that produced the round's last token -- candidate index
  // num_accepted_drafts of THIS call's window, i.e. mtp_last_hidden_'s row num_accepted_drafts.
  // (This is the one thing CommitVerifiedWindow deliberately does not do: a DFlash2 driver has no
  // MTP head.)
  R4DX_HIP_CHECK(hipMemcpyAsync(
      mtp_seed_hidden_.data(), mtp_last_hidden_ + num_accepted_drafts * hidden,
      static_cast<size_t>(hidden) * sizeof(uint16_t), hipMemcpyDeviceToDevice, stream_.get()));
  stream_.Synchronize();

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
  return DecodeStepDflashImpl(token_id, k, p_min, n_min, /*params=*/nullptr, /*rng=*/nullptr,
                               walk_len_out, trace_out, drafted_tokens_out);
}

std::vector<int32_t> Model::DecodeStepDflashSampled(int32_t token_id, int64_t k, float p_min,
                                                     int64_t n_min,
                                                     const kernels::SampleParams& params,
                                                     std::mt19937_64& rng, int64_t* walk_len_out) {
  // Design point E: greedy requests keep the greedy path exactly, including consuming no draw.
  const bool greedy = params.temperature <= 0.0f;
  return DecodeStepDflashImpl(token_id, k, p_min, n_min, greedy ? nullptr : &params,
                               greedy ? nullptr : &rng, walk_len_out, /*trace_out=*/nullptr,
                               /*drafted_tokens_out=*/nullptr);
}

std::vector<int32_t> Model::DecodeStepDflashImpl(int32_t token_id, int64_t k, float p_min,
                                                  int64_t n_min,
                                                  const kernels::SampleParams* params,
                                                  std::mt19937_64* rng, int64_t* walk_len_out,
                                                  DflashRoundTrace* trace_out,
                                                  std::vector<int32_t>* drafted_tokens_out) {
  RequireNotTp("DecodeStepDflash{Greedy,Sampled} (DFlash2, docs/tp.md 8.2: P5)");
  if (!dflash_.has_value()) {
    throw std::runtime_error(
        "Model::DecodeStepDflash{Greedy,Sampled}: DFlash2 is not enabled on this Model (Load() with "
        "ModelOptions::dflash_container set and dflash_draft_k > 0)");
  }
  if (k < 0 || k > dflash_draft_k_) {
    throw std::runtime_error("Model::DecodeStepDflash{Greedy,Sampled}: k must be in [0, " +
                              std::to_string(dflash_draft_k_) + "]");
  }
  if (!dflash_injection_enabled_) {
    throw std::runtime_error(
        "Model::DecodeStepDflash{Greedy,Sampled}: drafter injection is disabled on this Model "
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
        "Model::DecodeStepDflash{Greedy,Sampled}: drafter frontier (" +
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

  // Verify [token_id, d1..dm] in one pass and resolve the round through the SAME shared middle MTP
  // uses -- greedy acceptance (the longest prefix of drafts the target's own argmax confirms,
  // docs/dflash2.md section 5: "a rows accepted, the anchor is always accepted") or sample-and-match
  // (docs/sampling.md section 9). Commits nothing yet.
  //
  // That VerifyWindow call also (as a side effect of running every layer's own capture hook, since
  // AttachDflashFeatureCapture is attached whenever dflash_ is set) refills DflashFeatureBuffer()/
  // DflashFeatureRows() with THIS window's own captured target features, one row per candidate --
  // exactly what gets injected below for the accepted prefix. VerifyWindow itself never injects
  // anything (model.h's own doc comment on SetDflashCaptureObserver) -- only this method does, once
  // it knows how many rows were actually accepted.
  int64_t num_accepted_drafts = 0;
  std::vector<int32_t> result =
      VerifyAndResolveRound(token_id, draft.tokens, params, rng, &num_accepted_drafts);
  const int64_t num_committed = num_accepted_drafts + 1;  // +1 for token_id (the anchor) itself

  // Inject the accepted prefix's own just-captured target features into the drafter's ring BEFORE
  // CommitVerifiedWindow (which only moves this Model's pos_/GDN bookkeeping, never touches
  // DflashFeatureBuffer()/dflash_->InjectedCount()) -- docs/dflash2.md section 5 step 4: "inject
  // those a rows' captured features at positions n..n+a-1". Rows 0..num_committed-1 of the just-
  // captured window are exactly the accepted prefix (row 0 == the anchor, which is always accepted)
  // because VerifyWindow captures candidates in the same order they were verified.
  if (DflashFeatureRows() < num_committed) {
    throw std::runtime_error(
        "Model::DecodeStepDflash{Greedy,Sampled}: internal error -- VerifyWindow captured fewer rows (" +
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

// ---- tensor parallel (docs/tp.md 7.2-7.5) -------------------------------------------------------

void Model::GatherVocabRow(const float* shard_row_dev, float* full_host) const {
  // Plain (blocking, null-stream) D2H: the caller has synchronized stream_, the same rule every
  // other readback in this file follows. Staged in a separate vector rather than written straight
  // into this rank's slot of `full_host`, so the all-gather's source and destination never alias.
  gather_shard_host_.resize(static_cast<size_t>(vocab_local_));
  R4DX_HIP_CHECK(hipMemcpy(gather_shard_host_.data(), shard_row_dev,
                            static_cast<size_t>(vocab_local_) * sizeof(float),
                            hipMemcpyDeviceToHost));
  comm_->HostAllGather(gather_shard_host_.data(),
                       static_cast<size_t>(vocab_local_) * sizeof(float), full_host);
}

int32_t Model::MergeGreedyPair(const int32_t* pair_dev) {
  tp::ArgmaxPair mine{};
  R4DX_HIP_CHECK(hipMemcpy(&mine, pair_dev, sizeof(mine), hipMemcpyDeviceToHost));
  mine.idx += static_cast<int32_t>(vocab_offset_);  // local -> global id
  std::array<tp::ArgmaxPair, 2> all{};              // Model::Load allows world <= 2
  comm_->HostAllGather(&mine, sizeof(mine), all.data());
  return tp::MergeArgmax(all.data(), comm_->World());
}

void Model::MergeShardResults(std::vector<tp::ArgmaxPair>& pairs,
                              std::vector<kernels::RowSummary>* summaries) {
  static_assert(sizeof(tp::ArgmaxPair) == 8, "8 B per greedy row (docs/tp.md 7.3)");
  static_assert(std::is_trivially_copyable_v<kernels::RowSummary>,
                "row summaries cross the host exchange as raw bytes (same process, same layout)");
  const int world = comm_->World();
  const size_t n_pairs = pairs.size();
  const size_t n_sums = summaries != nullptr ? summaries->size() : 0;
  // Local -> global ids first (docs/tp.md 7.3, 7.4), so the merges below see global ids only.
  const int32_t offset = static_cast<int32_t>(vocab_offset_);
  for (tp::ArgmaxPair& p : pairs) p.idx += offset;
  for (size_t r = 0; r < n_sums; ++r) {
    kernels::RowSummary& s = (*summaries)[r];
    for (int j = 0; j < s.k; ++j) s.ids[j] += offset;
  }
  // One rendezvous for everything: [pairs][summaries] per rank, rank order.
  const size_t per_rank = n_pairs * sizeof(tp::ArgmaxPair) + n_sums * sizeof(kernels::RowSummary);
  std::vector<uint8_t> mine(per_rank);
  if (n_pairs > 0) std::memcpy(mine.data(), pairs.data(), n_pairs * sizeof(tp::ArgmaxPair));
  if (n_sums > 0) {
    std::memcpy(mine.data() + n_pairs * sizeof(tp::ArgmaxPair), summaries->data(),
                n_sums * sizeof(kernels::RowSummary));
  }
  merge_pack_host_.resize(per_rank * static_cast<size_t>(world));
  comm_->HostAllGather(mine.data(), per_rank, merge_pack_host_.data());

  std::vector<tp::ArgmaxPair> row_pairs(static_cast<size_t>(world));
  for (size_t t = 0; t < n_pairs; ++t) {
    for (int r = 0; r < world; ++r) {
      std::memcpy(&row_pairs[static_cast<size_t>(r)],
                  merge_pack_host_.data() + static_cast<size_t>(r) * per_rank + t * sizeof(tp::ArgmaxPair),
                  sizeof(tp::ArgmaxPair));
    }
    pairs[t].idx = tp::MergeArgmax(row_pairs.data(), world);
    pairs[t].val = 0.0f;  // the merged value is not needed; only the token id is
  }
  std::vector<kernels::RowSummary> row_sums(static_cast<size_t>(world));
  const int64_t global_vocab = container_.Config().vocab_size;
  for (size_t t = 0; t < n_sums; ++t) {
    for (int r = 0; r < world; ++r) {
      std::memcpy(&row_sums[static_cast<size_t>(r)],
                  merge_pack_host_.data() + static_cast<size_t>(r) * per_rank +
                      n_pairs * sizeof(tp::ArgmaxPair) + t * sizeof(kernels::RowSummary),
                  sizeof(kernels::RowSummary));
    }
    (*summaries)[t] = tp::MergeRowSummaries(row_sums.data(), world, global_vocab);
  }
}

void Model::TpWarmup() {
  if (comm_ == nullptr) {
    throw std::logic_error("Model::TpWarmup: only a tensor-parallel rank (ModelOptions::tp.world > 1) warms up");
  }
  if (gdn_control_.Frozen()) throw std::logic_error("Model::TpWarmup: called twice");
  // Every control-array key one sequence can ever ask for (docs/tp.md 2.7): cu for every chunk /
  // window length up to max_chunk_, and the one (slot, window) pair every GDN layer shares -- every
  // GdnStateManager is built with max_seqs 1 and the same max_decode_window (Load above).
  int32_t slot = 1;
  int64_t window = draft_window_;
  for (const auto& gs : gdn_states_) {
    if (gs) {
      slot = gs->SlotForSeq(0);
      window = gs->MaxDecodeWindow();
      break;
    }
  }
  gdn_control_.Prewarm(max_chunk_, slot, window);
  // The real paths once, through the public methods (2.9 step 9): a full 64-row prefill chunk (the
  // channel-1 all-reduce size) and a greedy decode step (channel 0). Fixed ids 0..63: the warm-up
  // is a lockstep collective, so every rank must feed the same tokens.
  std::vector<int32_t> ids(static_cast<size_t>(max_chunk_));
  for (int64_t i = 0; i < max_chunk_; ++i) ids[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  (void)Prefill(ids);
  (void)DecodeStepGreedy(0);
  Reset();
  gdn_control_.Freeze();
}

void Model::RequireNotTp(const char* what) const {
  if (comm_ != nullptr) {
    throw core::TpUnsupportedError(std::string("Model::") + what +
                                   " -- not supported on a tensor-parallel rank");
  }
}

}  // namespace r4dx::model
