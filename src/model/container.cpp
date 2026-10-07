#include "container.h"
#include "container_load_util.h"
#include "linear.h"  // BuildTrellisWScale (R4DX_FAKEQ_W)
#include "shard_loader.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"  // GemmTrellisTicketsBytes / GemmTrellisZeroTickets
#include "r4dx_convert/safetensors_reader.hpp"  // SafetensorsReader, Utf8ToWide -- see file comment
#include "tp/tp_shard.h"  // tensor-parallel shard rules and byte plans (LoadShard, docs/tp.md 5.1)
#include "trellis_meta.h"  // __metadata__.quant.trellis (docs/trellis-kernel.md 2.3, 2.5)
#include "w4a16_group_meta.h"  // quant2 Q3: __metadata__.quant.w4a16.groups (docs/quant2.md 5.1)

namespace r4dx::model {

// LayoutName/LayoutFromName now live in quant_linear.cpp (shared with r4dx_model_attention).

namespace {

// The container-loading helpers this file used to define here (ReadMetadata, W4a16LoadGroups, the trellis
// checks, the raw uploads, LoadQuantLinear*, LoadRotationWeights, ...) were moved verbatim to
// container_load_util.{h,cpp} so GemmaContainer shares them (docs/gemma4-plan.md M1-19); only their
// linkage changed.
using namespace container_util;
using r4dx_convert::SafetensorsReader;
using r4dx_convert::Utf8ToWide;

// docs/trellis-kernel.md 2.5 and 5.1: a head requested as kTrellis loads w4a16 -- the heads of a
// trellis container are never trellis -- so every caller gets the same answer (Model::Load passes
// the body layout for lm_head, tests pass whatever they load the body with).
ContainerLoadOptions MapTrellisHeads(const ContainerLoadOptions& o) {
  ContainerLoadOptions m = o;
  if (m.lm_head_layout == Layout::kTrellis) m.lm_head_layout = Layout::kW4a16;
  if (m.mtp_head_layout == Layout::kTrellis) m.mtp_head_layout = Layout::kW4a16;
  return m;
}

}  // namespace

namespace {

// R14/Q13 (docs/r9700.md): a `hipMemGetInfo` snapshot taken before any weight upload begins, so
// Container::Load can tell whether the layout it is about to load will fit in what's currently
// free -- the bf16 64-layer layout's 47.73 GiB against a fresh 31.86 GiB card is exactly the case
// this catches (§2.1's "bf16 does not fit" finding): the driver does not fail an oversubscribing
// hipMalloc outright, it silently pages the excess over PCIe (WDDM), so a loud stderr warning here
// is the only signal a caller gets before decode throughput craters. `hipMemGetInfo` failures
// (device not yet selected, etc.) are swallowed to 0/0 -- this diagnostic must never be why a load
// fails.
struct VramSnapshot {
  size_t free_bytes = 0;
  size_t total_bytes = 0;
  bool ok = false;
};
VramSnapshot SnapshotVram() {
  VramSnapshot s;
  s.ok = (hipMemGetInfo(&s.free_bytes, &s.total_bytes) == hipSuccess);
  return s;
}
double GiB(uint64_t bytes) { return static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0); }

}  // namespace

Container Container::Load(const std::string& path, Layout layout, Layout lm_head_layout,
                           int64_t layer_limit, Layout mtp_head_layout,
                           bool embed_device_resident, bool load_vision) {
  ContainerLoadOptions o;
  o.layout = layout;
  o.lm_head_layout = lm_head_layout;
  o.layer_limit = layer_limit;
  o.mtp_head_layout = mtp_head_layout;
  o.embed_device_resident = embed_device_resident;
  o.load_vision = load_vision;
  return Load(path, o);
}

Container Container::Load(const std::string& path, const ContainerLoadOptions& o_in) {
  // docs/trellis-kernel.md 5.1: the heads of a trellis container are w4a16 (or bf16) -- mapped
  // here, at the top, so the TP=1 path and LoadShard both see it.
  const ContainerLoadOptions o = MapTrellisHeads(o_in);
  if (o.tp_world < 1 || o.tp_world > 2 || o.tp_rank < 0 || o.tp_rank >= o.tp_world) {
    throw std::invalid_argument("r4dx::model::Container::Load: need tp_world in {1, 2} and 0 <= "
                                "tp_rank < tp_world, got tp_world " + std::to_string(o.tp_world) +
                                ", tp_rank " + std::to_string(o.tp_rank));
  }
  if (o.tp_world > 1) return LoadShard(path, o);
  // tp_world == 1 (docs/tp.md 5.1 step 7): the single-device loader below -- no rule lookup, no
  // staging. The TP-only options have no meaning here; refuse them rather than ignore them.
  if (o.shared_embed_host || o.embed_device_resident_decided >= 0 || o.parse_vision_config) {
    throw std::invalid_argument(
        "r4dx::model::Container::Load: shared_embed_host, embed_device_resident_decided and "
        "parse_vision_config are tensor-parallel options (tp_world > 1 only)");
  }
  const Layout layout = o.layout, lm_head_layout = o.lm_head_layout;
  const Layout mtp_head_layout = o.mtp_head_layout;
  const int64_t layer_limit = o.layer_limit;
  const bool embed_device_resident = o.embed_device_resident, load_vision = o.load_vision;

  const VramSnapshot vram_before = SnapshotVram();
  Container c;
  const nlohmann::json metadata = ReadMetadata(path);
  LinearLoadMeta meta;
  meta.w4a16 = CheckQuantGroups(metadata, path, layout, lm_head_layout, mtp_head_layout);
  meta.path = path;
  c.model_id_ = metadata.value("model_id", std::string());
  c.config_sha256_ = metadata.value("config_sha256", std::string());
  const nlohmann::json& model_config = metadata.at("model_config");
  const nlohmann::json& text_cfg = model_config.contains("text_config")
                                        ? model_config.at("text_config")
                                        : model_config;  // selftest containers have no text_config
  c.config_ = ModelConfig::FromJson(text_cfg);
  c.global_config_ = c.config_;  // tp_world == 1: the global config IS the config (docs/tp.md 3.2)
  // Top-level (not text_config/vision_config) -- see Container::ImageTokenId's doc comment.
  if (model_config.contains("image_token_id")) {
    c.image_token_id_ = model_config.at("image_token_id").get<int64_t>();
  }
  if (model_config.contains("video_token_id")) {
    c.video_token_id_ = model_config.at("video_token_id").get<int64_t>();
  }
  // quant2 (docs/quant2.md section 3.1): parsed before any upload, so an unknown rotation kind fails
  // here rather than after the whole container has been read into VRAM. nullopt: unrotated.
  const std::optional<RotationSpec> rotation =
      ParseRotationMetadata(metadata, c.global_config_, path);
  // docs/trellis-kernel.md 2.5: the trellis block, next to the rotation and with the same timing --
  // parsed and checked against the requested layout before any upload. nullopt: not trellis.
  meta.trellis = ParseTrellisMetadata(metadata, path);
  CheckTrellisChoice(meta.trellis, rotation.has_value(), layout, path);

  SafetensorsReader reader(Utf8ToWide(path));
  // quant2 Q3: a no-op for a container without a per-tensor group map.
  CheckW4a16GroupTensors(reader, meta.w4a16.groups, path);
  if (meta.w4a16.check_default_per_linear) LogW4a16Groups(meta.w4a16.groups, path);
  CheckTrellisTensors(reader, meta.trellis, path);

  const int64_t num_layers = (layer_limit >= 0)
                                  ? std::min(layer_limit, c.config_.num_hidden_layers)
                                  : c.config_.num_hidden_layers;

  // text.embed_tokens: host-resident, pinned so a future async H2D staging copy can overlap.
  {
    const int64_t n = ElemCountBySize(reader, "text.embed_tokens", 2);
    c.embed_tokens_ = core::PinnedBuffer<uint16_t>(static_cast<size_t>(n));
    std::memcpy(c.embed_tokens_.data(), reader.Data("text.embed_tokens"),
                static_cast<size_t>(n) * 2);

    // Device mirror (docs/mtp.md "device-resident draft loop") -- see Load()'s own comment for the
    // free-VRAM heuristic and why this is a best-effort ADDITION, never a replacement for the host
    // copy above.
    if (embed_device_resident) {
      size_t free_bytes = 0, total_bytes = 0;
      R4DX_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
      const size_t embed_bytes = static_cast<size_t>(n) * sizeof(uint16_t);
      if (free_bytes > embed_bytes * 2) {
        c.embed_tokens_dev_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(n));
        c.embed_tokens_dev_.CopyFromHost(c.embed_tokens_.data(), static_cast<size_t>(n));
      } else {
        std::fprintf(stderr,
                      "r4dx: only %.2f GiB free VRAM (need ~%.2f GiB for text.embed_tokens plus "
                      "headroom for the rest of the container) -- keeping embeddings host-only, "
                      "gather will go through the host path\n",
                      static_cast<double>(free_bytes) / (1024.0 * 1024 * 1024),
                      static_cast<double>(embed_bytes) / (1024.0 * 1024 * 1024));
      }
    }
  }

  const int64_t hidden = c.config_.hidden_size;
  const int64_t key_dim = c.config_.KeyDim();
  const int64_t value_dim = c.config_.ValueDim();
  const int64_t kv_heads = c.config_.num_key_value_heads;
  const int64_t attn_out = c.config_.num_attention_heads * c.config_.head_dim;

  // Milestone 11: how many linears did not carry the requested layout and loaded as bf16 instead
  // (LoadQuantLinearWithFallback's tier 2/3). Normally 0; non-zero exactly when the container was
  // built with r4dx-convert --keep-bf16, or is an old container predating a tensor's quantization.
  int bf16_fallbacks = 0;

  c.layers_.reserve(static_cast<size_t>(num_layers));
  for (int64_t i = 0; i < num_layers; ++i) {
    const std::string base = "text.layers." + std::to_string(i) + ".";
    LayerWeights lw;
    // A rotated container stores its (zeroed, folded) norms under `.rotated` names so that a binary
    // predating quant2 -- which never reads __metadata__.rotation -- fails on a missing tensor
    // instead of running rotated weights in the unrotated basis (docs/container-format.md).
    const char* norm_suffix = rotation ? ".rotated" : "";
    lw.input_layernorm = UploadRawU16(reader, base + "input_layernorm" + norm_suffix);
    lw.post_attention_layernorm =
        UploadRawU16(reader, base + "post_attention_layernorm" + norm_suffix);

    if (c.config_.IsGdnLayer(i)) {
      GdnWeights g;
      g.in_proj_qkv = LoadQuantLinearWithFallback(reader, meta,base + "gdn.in_proj_qkv", layout,
                                                   2 * key_dim + value_dim, hidden,
                                                   &bf16_fallbacks);
      g.in_proj_z = LoadQuantLinearWithFallback(reader, meta,base + "gdn.in_proj_z", layout, value_dim,
                                                 hidden, &bf16_fallbacks);
      g.in_proj_b = UploadRawU16(reader, base + "gdn.in_proj_b");
      g.in_proj_a = UploadRawU16(reader, base + "gdn.in_proj_a");
      g.conv1d_weight = UploadRawU16(reader, base + "gdn.conv1d_weight");
      g.A_log = UploadRawF32(reader, base + "gdn.A_log");
      g.dt_bias = UploadRawF32(reader, base + "gdn.dt_bias");
      g.norm_weight = UploadWidenedF32(reader, base + "gdn.norm_weight");
      g.out_proj = LoadQuantLinearWithFallback(reader, meta,base + "gdn.out_proj", layout, hidden,
                                                value_dim, &bf16_fallbacks);
      lw.gdn = std::move(g);
    } else {
      AttnWeights a;
      // attn.qg/o now honor the requested body `layout` the same way GDN's in_proj_qkv/out_proj
      // and MLP's gate_up/down do (decode-perf pass, 2026-09-19): AttentionLayer dispatches both
      // through the shared r4dx::model::ApplyLinear (src/model/linear.h), which every layout
      // already supports. See docs/perf.md for the measured per-layout VRAM/throughput delta this
      // unlocks (attn.qg/o account for 16 of 64 layers' full-attention projections).
      a.qg = LoadQuantLinearWithFallback(reader, meta,base + "attn.qg", layout, attn_out * 2, hidden,
                                          &bf16_fallbacks);
      a.k = LoadQuantLinearWithFallback(reader, meta,base + "attn.k", layout,
                                         kv_heads * c.config_.head_dim, hidden, &bf16_fallbacks);
      a.v = LoadQuantLinearWithFallback(reader, meta,base + "attn.v", layout,
                                         kv_heads * c.config_.head_dim, hidden, &bf16_fallbacks);
      a.o = LoadQuantLinearWithFallback(reader, meta,base + "attn.o", layout, hidden, attn_out,
                                         &bf16_fallbacks);
      a.q_norm = UploadRawU16(reader, base + "attn.q_norm");
      a.k_norm = UploadRawU16(reader, base + "attn.k_norm");
      a.k_descale = UploadRawF32(reader, base + "attn.k_descale");
      a.v_descale = UploadRawF32(reader, base + "attn.v_descale");
      lw.attn = std::move(a);
    }

    lw.mlp.gate_up = LoadQuantLinearWithFallback(reader, meta,base + "mlp.gate_up", layout,
                                                  2 * c.config_.intermediate_size, hidden,
                                                  &bf16_fallbacks);
    lw.mlp.down = LoadQuantLinearWithFallback(reader, meta,base + "mlp.down", layout, hidden,
                                               c.config_.intermediate_size, &bf16_fallbacks);
    c.layers_.push_back(std::move(lw));
  }

  c.final_norm_ = UploadRawU16(reader, "text.final_norm");
  // With fallback: a container converted with `--lm-head bf16` (rung 4 follow-up -- the 4-bit
  // lm_head is where the vocab-tail KL loss concentrates) carries only lm_head.bf16.w, and every
  // caller that asks for the body layout here should get that bf16 head rather than a throw.
  const int fallbacks_before_head = bf16_fallbacks;
  c.lm_head_ = LoadQuantLinearWithFallback(reader, meta,"lm_head", lm_head_layout, c.config_.vocab_size,
                                           hidden, &bf16_fallbacks);
  const bool trellis_bf16_head =
      TakeTrellisBf16Head(meta, c.lm_head_, fallbacks_before_head, &bf16_fallbacks);
  if (rotation) {
    c.rotation_ = LoadRotationWeights(reader, *rotation, c.global_config_, c.config_, path,
                                      [&](const std::string& name) { return UploadRawF32(reader, name); });
    LogRotation(*rotation, path);
  }

  // mtp.* (docs/container-format.md, docs/mtp.md): present only when the container was converted
  // with --mtp on -- probe with SafetensorsReader::Has rather than trusting __metadata__, so this
  // works uniformly for the real checkpoint's container and any hand-built/selftest fixture.
  // "mtp.norm" (add_bf16, src/convert/main.cpp) has no .{layout} suffix -- a bare bf16 passthrough
  // tensor, same naming convention as "text.final_norm" (UploadRawU16 below, not LoadQuantLinear).
  if (reader.Has("mtp.norm")) {
    MtpWeights mw;
    const std::string base = "mtp.";
    LayerWeights lw;
    lw.input_layernorm = UploadRawU16(reader, base + "input_layernorm");
    lw.post_attention_layernorm = UploadRawU16(reader, base + "post_attention_layernorm");
    // mtp.attn.qg/o and mtp.mlp.gate_up/down load in `mtp_head_layout`, NOT the body `layout` --
    // the draft head is a single decoder layer chained up to draft_k times, so its quantization
    // error compounds across chained drafts far more than one body-layer's own error does (task
    // rationale, docs/mtp.md "MTP head layout"). Every other mtp.* tensor (attn.k/v/q_norm/k_norm/
    // descales, fc, norm, pre_fc_norm_*) has only one on-disk form regardless of layout, same as
    // the body layers above.
    AttnWeights a;
    a.qg = LoadQuantLinearWithFallback(reader, meta,base + "attn.qg", mtp_head_layout, attn_out * 2,
                                        hidden, &bf16_fallbacks);
    // mtp.attn.k/v stay bf16 regardless of the requested body/head layout (docs/r9700.md's R1
    // task: "Keep mtp.* ... as they are") -- request Layout::kBf16 explicitly rather than
    // `layout`/`mtp_head_layout`, so this never picks up a quantized form even if a future
    // converter run ever quantized mtp.*. LoadQuantLinearWithFallback (not UploadRawU16) so this
    // still works against the OLD bare-tensor on-disk form (pre-R1 containers) as well as any
    // future `.bf16.w`-suffixed form -- see LoadQuantLinearWithFallback's own comment.
    a.k = LoadQuantLinearWithFallback(reader, meta,base + "attn.k", Layout::kBf16,
                                       kv_heads * c.config_.head_dim, hidden);
    a.v = LoadQuantLinearWithFallback(reader, meta,base + "attn.v", Layout::kBf16,
                                       kv_heads * c.config_.head_dim, hidden);
    a.o = LoadQuantLinearWithFallback(reader, meta,base + "attn.o", mtp_head_layout, hidden, attn_out,
                                       &bf16_fallbacks);
    a.q_norm = UploadRawU16(reader, base + "attn.q_norm");
    a.k_norm = UploadRawU16(reader, base + "attn.k_norm");
    a.k_descale = UploadRawF32(reader, base + "attn.k_descale");
    a.v_descale = UploadRawF32(reader, base + "attn.v_descale");
    lw.attn = std::move(a);
    lw.mlp.gate_up = LoadQuantLinearWithFallback(reader, meta,base + "mlp.gate_up", mtp_head_layout,
                                                  2 * c.config_.intermediate_size, hidden,
                                                  &bf16_fallbacks);
    lw.mlp.down = LoadQuantLinearWithFallback(reader, meta,base + "mlp.down", mtp_head_layout, hidden,
                                               c.config_.intermediate_size, &bf16_fallbacks);
    mw.layer = std::move(lw);
    mw.fc = UploadRawU16(reader, "mtp.fc");
    mw.norm = UploadRawU16(reader, "mtp.norm");
    mw.pre_fc_norm_hidden = UploadRawU16(reader, "mtp.pre_fc_norm_hidden");
    mw.pre_fc_norm_embedding = UploadRawU16(reader, "mtp.pre_fc_norm_embedding");

    // Reduced-vocab draft head (docs/r9700.md R9, container.h's MtpWeights own doc comment):
    // OPTIONAL, probed the same way HasMtp() probes for mtp.* itself -- present only when the
    // container was converted with `--draft-vocab-ids` (r4dx-convert). `mtp.draft_head.vocab_ids`
    // is the source of truth for the subset size (its own element count), read FIRST so the
    // subsequent LoadQuantLinear call knows N without a separate metadata round-trip; an old
    // container (or a run that omitted --draft-vocab-ids) simply lacks this tensor, leaving
    // draft_lm_head.N == 0 (HasDraftHead() false) -- MtpHead::Draft's own fallback then runs the
    // exact pre-R9 full-vocab path, unconditionally correct for every container ever produced.
    if (reader.Has("mtp.draft_head.vocab_ids")) {
      mw.draft_vocab_ids = UploadRawI32(reader, "mtp.draft_head.vocab_ids");
      const int64_t draft_vocab_size = static_cast<int64_t>(mw.draft_vocab_ids.size());
      // Same mtp_head_layout as every other mtp.* quantized linear (container.h's own comment on
      // why: the draft head's error compounds across chained draft steps) -- a run that chose to
      // build a draft head always emits it in the same LayoutSet as mtp.attn.qg/o and
      // mtp.mlp.gate_up/down, so this load-time layout selection just works.
      mw.draft_lm_head = LoadQuantLinearWithFallback(reader, meta,"mtp.draft_head.lm_head",
                                                      mtp_head_layout, draft_vocab_size, hidden,
                                                      &bf16_fallbacks);
    }
    c.mtp_ = std::move(mw);
  }

  // vision.* (docs/container-format.md, docs/vision.md "Load policy"): probed the same way mtp.*
  // is -- one representative tensor rather than a metadata field -- and uploaded only when the
  // caller asked, because it is ~0.90 GiB a text-only run must not pay for. `vision_config` comes
  // from the container's own metadata; a container that carries the tensors but no vision_config
  // block is a converter bug, so that combination throws rather than defaulting a geometry.
  // Milestone 11 (docs/validation.md "Milestone 11 / sensitivity"): say out loud how many linears
  // did not carry the requested layout. A `--keep-bf16` sensitivity container is supposed to have a
  // non-zero count here and the number is the experiment's own check that the regex selected the
  // class it meant to; a PRODUCTION container reporting anything other than 0 is a converter run
  // that quietly shipped bf16 weights (4x the bytes, and the throughput to match).
  if (bf16_fallbacks > 0) {
    std::fprintf(stderr,
                  "r4dx: %d linear(s) in %s do not carry the requested layout and were loaded as "
                  "bf16 (r4dx-convert --keep-bf16, or a container predating that tensor's "
                  "quantization)\n",
                  bf16_fallbacks, path.c_str());
  }
  // docs/trellis-kernel.md 4.5: every trellis linear's tickets, one zeroed buffer for the
  // container.
  if (meta.trellis) {
    c.trellis_ = std::move(meta.trellis);
    c.AssignTrellisTickets();
    LogTrellis(*c.trellis_, path);
    if (trellis_bf16_head) LogTrellisBf16Head(path);
  }

  c.container_has_vision_tensors_ = vision::HasVisionTensors(reader);
  if (load_vision && c.container_has_vision_tensors_) {
    if (!model_config.contains("vision_config")) {
      throw std::runtime_error("r4dx::model::Container: " + path +
                                " carries vision.* tensors but no model_config.vision_config");
    }
    c.vision_ = vision::LoadVisionWeights(reader, model_config.at("vision_config"));
  }

  // R14 (docs/r9700.md): warn, don't fail, if this load just consumed more VRAM than was free
  // when it started -- the bf16-on-64-layer case (47.73 GiB of weights on a 31.86 GiB card) is
  // exactly the scenario docs/r9700.md's §2.1 finding describes: hipMalloc does not error, WDDM
  // silently pages the excess over PCIe, and the only symptom is a 20-30x decode slowdown with no
  // diagnostic anywhere. Signal: measured against the real bf16 container on this card,
  // `hipMemGetInfo` CLAMPS free at ~0 rather than reporting the true 47.73 GiB logical footprint
  // against a 31.86 GiB card (WDDM's virtual/physical split hides the over-commit from this API),
  // so this flags "this call drove free VRAM to near-zero starting from headroom that was NOT
  // already near-zero", which is what was actually observed. (A "textbook" signal -- consumed
  // bytes exceeding what was free before loading started -- was tried first and removed: with
  // vram_after.free_bytes and vram_before.free_bytes both unsigned, `after < before` guarding
  // `(before - after) > before` is unreachable for any valid free-byte pair, so it could never
  // fire; see docs/status.md's R14 section for the removal note.)
  if (vram_before.ok) {
    const VramSnapshot vram_after = SnapshotVram();
    if (vram_after.ok) {
      constexpr uint64_t kNearZeroThreshold = 1ull << 30;  // 1 GiB
      const bool clamped_near_zero_signal =
          vram_after.free_bytes < kNearZeroThreshold && vram_before.free_bytes >= kNearZeroThreshold;
      if (clamped_near_zero_signal) {
        std::cerr << "[r4dx::model::Container] WARNING: layout '" << LayoutName(layout)
                  << "' left only " << GiB(vram_after.free_bytes) << " GiB free (was "
                  << GiB(vram_before.free_bytes) << " GiB free before this load) -- this looks like "
                  << "an over-committed load (docs/r9700.md §2.1's bf16-on-64-layer finding: 47.73 "
                  << "GiB of weights on a 31.86 GiB card). hipMalloc does not error on this; the "
                  << "driver (WDDM) silently pages the excess over PCIe, and the only symptom is "
                  << "decode throughput far below any quantized-layout number (\"bf16-layout "
                  << "performance work\" is explicitly out of scope for this reason).\n";
      }
    }
  }

  return c;
}

// ---- tensor-parallel shard loading (docs/tp.md 4.3, 5.1) ---------------------------------------

// ShardLoader / ShardLoadStats live in shard_loader.h (shared with GemmaContainer, M1b-1).
namespace {
using container_util::ShardLoader;
using container_util::ShardLoadStats;
}  // namespace

Container Container::LoadShard(const std::string& path, const ContainerLoadOptions& o) {
  if (o.load_vision && o.tp_rank != 0) {
    throw std::invalid_argument(
        "r4dx::model::Container::Load: the vision tower's weights live on tensor-parallel rank 0 "
        "only (docs/tp.md 4.2); load_vision was set on rank " + std::to_string(o.tp_rank));
  }
  const VramSnapshot vram_before = SnapshotVram();
  Container c;
  const nlohmann::json metadata = ReadMetadata(path);
  LinearLoadMeta meta;
  meta.w4a16 = CheckQuantGroups(metadata, path, o.layout, o.lm_head_layout, o.mtp_head_layout);
  meta.path = path;
  const W4a16LoadGroups& w4a16 = meta.w4a16;
  c.model_id_ = metadata.value("model_id", std::string());
  c.config_sha256_ = metadata.value("config_sha256", std::string());
  const nlohmann::json& model_config = metadata.at("model_config");
  const nlohmann::json& text_cfg = model_config.contains("text_config")
                                        ? model_config.at("text_config")
                                        : model_config;  // selftest containers have no text_config
  c.global_config_ = ModelConfig::FromJson(text_cfg);
  c.config_ = ModelConfig::Shard(c.global_config_, o.tp_world, o.tp_rank);
  if (model_config.contains("image_token_id")) {
    c.image_token_id_ = model_config.at("image_token_id").get<int64_t>();
  }
  if (model_config.contains("video_token_id")) {
    c.video_token_id_ = model_config.at("video_token_id").get<int64_t>();
  }
  // The groups the container's w4a16 scales were packed at -- the wsz stride per 16-row tile
  // (docs/tp.md 4.3) -- now per tensor (quant2 Q3): `w4a16` above. A container that predates the
  // `quant` block is group 128, CheckQuantGroups' own rule (ParseW4a16Groups' default); ShardLoader
  // sizes and slices each scale tensor at W4a16LoadGroups::KernelGroup, the group the kernel reads.
  // quant2: the TP=1 path's parse, before any upload (docs/quant2.md section 3.1).
  const std::optional<RotationSpec> rotation =
      ParseRotationMetadata(metadata, c.global_config_, path);
  // docs/trellis-kernel.md 2.5: the TP=1 path's trellis parse and refusals, before any upload; the
  // spec travels into ShardLoader, whose kTrellis case needs each linear's rate and parts.
  meta.trellis = ParseTrellisMetadata(metadata, path);
  CheckTrellisChoice(meta.trellis, rotation.has_value(), o.layout, path);

  SafetensorsReader reader(Utf8ToWide(path));
  CheckW4a16GroupTensors(reader, w4a16.groups, path);  // quant2 Q3; see Container::Load
  if (w4a16.check_default_per_linear && o.tp_rank == 0) LogW4a16Groups(w4a16.groups, path);
  CheckTrellisTensors(reader, meta.trellis, path);
  const ModelConfig& gc = c.global_config_;
  ShardLoader L(reader, gc, o.tp_world, o.tp_rank, meta);
  const int64_t num_layers =
      (o.layer_limit >= 0) ? std::min(o.layer_limit, gc.num_hidden_layers) : gc.num_hidden_layers;

  // text.embed_tokens: replicated. Host: the process's one shared pinned copy when the caller has
  // it (docs/tp.md 5.3 -- never duplicated), else this rank's own. Device mirror: the caller's joint
  // decision when given, so every rank takes the same gather path (docs/tp.md 2.9 step 5), else the
  // single-device 2x-headroom heuristic.
  {
    const int64_t n = ElemCountBySize(reader, "text.embed_tokens", 2);
    if (tp::RuleFor("text.embed_tokens", gc).split != tp::Split::kReplicate) {
      throw std::logic_error("r4dx::model::Container: text.embed_tokens must replicate");
    }
    if (o.shared_embed_host) {
      if (o.shared_embed_host->size() != static_cast<size_t>(n)) {
        throw std::invalid_argument("r4dx::model::Container::Load: shared_embed_host holds " +
                                    std::to_string(o.shared_embed_host->size()) +
                                    " elements, text.embed_tokens has " + std::to_string(n));
      }
      c.shared_embed_host_ = o.shared_embed_host;
    } else {
      c.embed_tokens_ = core::PinnedBuffer<uint16_t>(static_cast<size_t>(n));
      std::memcpy(c.embed_tokens_.data(), reader.Data("text.embed_tokens"),
                  static_cast<size_t>(n) * 2);
    }
    const size_t embed_bytes = static_cast<size_t>(n) * sizeof(uint16_t);
    bool resident = false;
    if (o.embed_device_resident_decided >= 0) {
      resident = o.embed_device_resident_decided != 0;
    } else if (o.embed_device_resident) {
      size_t free_bytes = 0, total_bytes = 0;
      R4DX_HIP_CHECK(hipMemGetInfo(&free_bytes, &total_bytes));
      resident = free_bytes > embed_bytes * 2;
      if (!resident) {
        std::fprintf(stderr,
                     "r4dx: only %.2f GiB free VRAM (need ~%.2f GiB for text.embed_tokens plus "
                     "headroom for the rest of the container) -- keeping embeddings host-only, "
                     "gather will go through the host path\n",
                     GiB(free_bytes), GiB(embed_bytes));
      }
    }
    if (resident) {
      c.embed_tokens_dev_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(n));
      c.embed_tokens_dev_.CopyFromHost(c.EmbedTokensHost(), static_cast<size_t>(n));
    }
    L.CountReplicated(resident ? embed_bytes : 0);
  }

  // GLOBAL shapes throughout: ShardLoader::Linear takes the logical [N, K] and returns the rank's.
  const int64_t hidden = gc.hidden_size;
  const int64_t key_dim = gc.KeyDim();
  const int64_t value_dim = gc.ValueDim();
  const int64_t kv_rows = gc.num_key_value_heads * gc.head_dim;
  const int64_t attn_out = gc.num_attention_heads * gc.head_dim;
  const int64_t intermediate = gc.intermediate_size;
  int bf16_fallbacks = 0;

  c.layers_.reserve(static_cast<size_t>(num_layers));
  for (int64_t i = 0; i < num_layers; ++i) {
    const std::string base = "text.layers." + std::to_string(i) + ".";
    LayerWeights lw;
    const char* norm_suffix = rotation ? ".rotated" : "";  // see Container::Load
    lw.input_layernorm = L.Raw<uint16_t>(base + "input_layernorm" + norm_suffix);
    lw.post_attention_layernorm = L.Raw<uint16_t>(base + "post_attention_layernorm" + norm_suffix);
    if (gc.IsGdnLayer(i)) {
      GdnWeights gw;
      gw.in_proj_qkv = L.Linear(base + "gdn.in_proj_qkv", o.layout, 2 * key_dim + value_dim,
                                hidden, &bf16_fallbacks);
      gw.in_proj_z = L.Linear(base + "gdn.in_proj_z", o.layout, value_dim, hidden, &bf16_fallbacks);
      gw.in_proj_b = L.Raw<uint16_t>(base + "gdn.in_proj_b");
      gw.in_proj_a = L.Raw<uint16_t>(base + "gdn.in_proj_a");
      gw.conv1d_weight = L.Raw<uint16_t>(base + "gdn.conv1d_weight");
      gw.A_log = L.Raw<float>(base + "gdn.A_log");
      gw.dt_bias = L.Raw<float>(base + "gdn.dt_bias");
      gw.norm_weight = L.WidenedF32(base + "gdn.norm_weight");
      gw.out_proj = L.Linear(base + "gdn.out_proj", o.layout, hidden, value_dim, &bf16_fallbacks);
      lw.gdn = std::move(gw);
    } else {
      AttnWeights a;
      a.qg = L.Linear(base + "attn.qg", o.layout, attn_out * 2, hidden, &bf16_fallbacks);
      a.k = L.Linear(base + "attn.k", o.layout, kv_rows, hidden, &bf16_fallbacks);
      a.v = L.Linear(base + "attn.v", o.layout, kv_rows, hidden, &bf16_fallbacks);
      a.o = L.Linear(base + "attn.o", o.layout, hidden, attn_out, &bf16_fallbacks);
      a.q_norm = L.Raw<uint16_t>(base + "attn.q_norm");
      a.k_norm = L.Raw<uint16_t>(base + "attn.k_norm");
      a.k_descale = L.Raw<float>(base + "attn.k_descale");
      a.v_descale = L.Raw<float>(base + "attn.v_descale");
      lw.attn = std::move(a);
    }
    lw.mlp.gate_up = L.Linear(base + "mlp.gate_up", o.layout, 2 * intermediate, hidden,
                              &bf16_fallbacks);
    lw.mlp.down = L.Linear(base + "mlp.down", o.layout, hidden, intermediate, &bf16_fallbacks);
    c.layers_.push_back(std::move(lw));
  }

  c.final_norm_ = L.Raw<uint16_t>("text.final_norm");
  // Vocab-split (docs/tp.md 7.1): this rank's [vocab/world, hidden] rows.
  const int fallbacks_before_head = bf16_fallbacks;
  c.lm_head_ = L.Linear("lm_head", o.lm_head_layout, gc.vocab_size, hidden, &bf16_fallbacks);
  const bool trellis_bf16_head =
      TakeTrellisBf16Head(meta, c.lm_head_, fallbacks_before_head, &bf16_fallbacks);
  // quant2 (docs/quant2.md section 3.1): every rank runs the same residual ops on its own full
  // replicated rows, so signs/mix5 replicate; the q2ab sign vectors are cut by tp::RuleFor with the
  // K range of their linear's columns (mlp.down 8704 = 17 x 512 per rank, attn.o 12 heads x 256,
  // gdn.out_proj 24 heads x 128), which LoadRotationWeights checks against the rank config.
  if (rotation) {
    c.rotation_ = LoadRotationWeights(reader, *rotation, gc, c.config_, path,
                                      [&](const std::string& name) { return L.Raw<float>(name); });
    if (o.tp_rank == 0) LogRotation(*rotation, path);
  }

  // mtp.* (docs/tp.md 4.2): the attention sublayer and MLP shard exactly like a body layer; fc, the
  // norms and the optional reduced-vocab draft head replicate. Same tensor set and layout choices
  // as the TP=1 path above.
  if (reader.Has("mtp.norm")) {
    MtpWeights mw;
    LayerWeights lw;
    lw.input_layernorm = L.Raw<uint16_t>("mtp.input_layernorm");
    lw.post_attention_layernorm = L.Raw<uint16_t>("mtp.post_attention_layernorm");
    AttnWeights a;
    a.qg = L.Linear("mtp.attn.qg", o.mtp_head_layout, attn_out * 2, hidden, &bf16_fallbacks);
    a.k = L.Linear("mtp.attn.k", Layout::kBf16, kv_rows, hidden);
    a.v = L.Linear("mtp.attn.v", Layout::kBf16, kv_rows, hidden);
    a.o = L.Linear("mtp.attn.o", o.mtp_head_layout, hidden, attn_out, &bf16_fallbacks);
    a.q_norm = L.Raw<uint16_t>("mtp.attn.q_norm");
    a.k_norm = L.Raw<uint16_t>("mtp.attn.k_norm");
    a.k_descale = L.Raw<float>("mtp.attn.k_descale");
    a.v_descale = L.Raw<float>("mtp.attn.v_descale");
    lw.attn = std::move(a);
    lw.mlp.gate_up = L.Linear("mtp.mlp.gate_up", o.mtp_head_layout, 2 * intermediate, hidden,
                              &bf16_fallbacks);
    lw.mlp.down = L.Linear("mtp.mlp.down", o.mtp_head_layout, hidden, intermediate,
                           &bf16_fallbacks);
    mw.layer = std::move(lw);
    mw.fc = L.Raw<uint16_t>("mtp.fc");
    mw.norm = L.Raw<uint16_t>("mtp.norm");
    mw.pre_fc_norm_hidden = L.Raw<uint16_t>("mtp.pre_fc_norm_hidden");
    mw.pre_fc_norm_embedding = L.Raw<uint16_t>("mtp.pre_fc_norm_embedding");
    if (reader.Has("mtp.draft_head.vocab_ids")) {
      mw.draft_vocab_ids = L.Raw<int32_t>("mtp.draft_head.vocab_ids");
      const int64_t draft_vocab_size = static_cast<int64_t>(mw.draft_vocab_ids.size());
      mw.draft_lm_head = L.Linear("mtp.draft_head.lm_head", o.mtp_head_layout, draft_vocab_size,
                                  hidden, &bf16_fallbacks);
    }
    c.mtp_ = std::move(mw);
  }

  if (bf16_fallbacks > 0) {
    std::fprintf(stderr,
                 "r4dx: %d linear(s) in %s do not carry the requested layout and were loaded as "
                 "bf16 (r4dx-convert --keep-bf16, or a container predating that tensor's "
                 "quantization)\n",
                 bf16_fallbacks, path.c_str());
  }
  // docs/trellis-kernel.md 4.5: this rank's own ticket buffer (each rank loads its own Container).
  if (meta.trellis) {
    c.trellis_ = meta.trellis;
    c.AssignTrellisTickets();
    if (o.tp_rank == 0) LogTrellis(*c.trellis_, path);
    if (o.tp_rank == 0 && trellis_bf16_head) LogTrellisBf16Head(path);
  }

  // vision.* (docs/tp.md 4.2, 8.3): rank-0-only weights; any rank may parse just the geometry.
  c.container_has_vision_tensors_ = vision::HasVisionTensors(reader);
  if (c.container_has_vision_tensors_ && (o.load_vision || o.parse_vision_config)) {
    if (!model_config.contains("vision_config")) {
      throw std::runtime_error("r4dx::model::Container: " + path +
                               " carries vision.* tensors but no model_config.vision_config");
    }
    if (o.load_vision) {
      c.vision_ = vision::LoadVisionWeights(reader, model_config.at("vision_config"));
    } else {
      c.vision_config_ = vision::VisionConfig::FromJson(model_config.at("vision_config"));
    }
  }

  const ShardLoadStats& st = L.Stats();
  std::cerr << "[r4dx::model::Container] rank " << o.tp_rank << "/" << o.tp_world << ": "
            << st.sharded << " tensors sharded, " << st.replicated << " replicated, "
            << GiB(st.uploaded_bytes) << " GiB uploaded, " << GiB(st.staged_bytes)
            << " GiB gathered through staging\n";

  // The TP=1 path's over-commit warning (docs/r9700.md R14), for this rank's device.
  if (vram_before.ok) {
    const VramSnapshot vram_after = SnapshotVram();
    constexpr uint64_t kNearZeroThreshold = 1ull << 30;  // 1 GiB
    if (vram_after.ok && vram_after.free_bytes < kNearZeroThreshold &&
        vram_before.free_bytes >= kNearZeroThreshold) {
      std::cerr << "[r4dx::model::Container] WARNING: rank " << o.tp_rank << " layout '"
                << LayoutName(o.layout) << "' left only " << GiB(vram_after.free_bytes)
                << " GiB free (was " << GiB(vram_before.free_bytes)
                << " GiB free before this load) -- this looks like an over-committed load; the "
                   "driver (WDDM) pages the excess over PCIe instead of failing hipMalloc\n";
    }
  }
  return c;
}

template <class Fn>
void Container::ForEachLinear(Fn&& fn) {
  const auto layer = [&](LayerWeights& lw) {
    if (lw.attn) {
      fn(lw.attn->qg);
      fn(lw.attn->k);
      fn(lw.attn->v);
      fn(lw.attn->o);
    }
    if (lw.gdn) {
      fn(lw.gdn->in_proj_qkv);
      fn(lw.gdn->in_proj_z);
      fn(lw.gdn->out_proj);
    }
    fn(lw.mlp.gate_up);
    fn(lw.mlp.down);
  };
  for (LayerWeights& lw : layers_) layer(lw);
  fn(lm_head_);
  if (mtp_) {
    layer(mtp_->layer);
    fn(mtp_->draft_lm_head);
  }
}

void Container::BuildTrellisWScales(int mode, hipStream_t stream) {
  if (mode == 0) return;
  ForEachLinear([&](QuantLinear& q) { BuildTrellisWScale(q, mode, stream); });
}

int Container::BuildTrellisI8Scales(hipStream_t stream, int* trellis_total) {
  int built = 0, total = 0;
  ForEachLinear([&](QuantLinear& q) {
    if (q.layout != Layout::kTrellis) return;
    ++total;
    if (BuildTrellisI8Scale(q, stream)) ++built;
  });
  if (trellis_total != nullptr) *trellis_total = total;
  return built;
}

void Container::AssignTrellisTickets() {
  size_t total = 0;
  ForEachLinear([&](QuantLinear& q) {
    if (q.layout == Layout::kTrellis) {
      total += core::r4d::GemmTrellisTicketsBytes(static_cast<int>(q.N)) / sizeof(uint32_t);
    }
  });
  trellis_tickets_ = core::DeviceBuffer<uint32_t>(total);
  trellis_tickets_.Zero();
  size_t off = 0;
  ForEachLinear([&](QuantLinear& q) {
    if (q.layout != Layout::kTrellis) return;
    q.trellis_tickets = trellis_tickets_.data() + off;
    off += core::r4d::GemmTrellisTicketsBytes(static_cast<int>(q.N)) / sizeof(uint32_t);
  });
}

void Container::ZeroTrellisTickets(hipStream_t stream) {
  if (trellis_tickets_.empty()) return;
  core::r4d::GemmTrellisZeroTickets(trellis_tickets_.data(), trellis_tickets_.bytes(), stream);
}

std::shared_ptr<const core::PinnedBuffer<uint16_t>> Container::LoadEmbedTokensHost(
    const std::string& path) {
  SafetensorsReader reader(Utf8ToWide(path));
  const int64_t n = ElemCountBySize(reader, "text.embed_tokens", 2);
  auto buf = std::make_shared<core::PinnedBuffer<uint16_t>>(static_cast<size_t>(n),
                                                            hipHostMallocPortable);
  std::memcpy(buf->data(), reader.Data("text.embed_tokens"), static_cast<size_t>(n) * 2);
  return buf;
}

}  // namespace r4dx::model
