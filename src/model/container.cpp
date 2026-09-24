#include "container.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx_convert/safetensors_reader.hpp"  // SafetensorsReader, Utf8ToWide -- see file comment
#include "tp/tp_shard.h"  // tensor-parallel shard rules and byte plans (LoadShard, docs/tp.md 5.1)

namespace r4dx::model {

// LayoutName/LayoutFromName now live in quant_linear.cpp (shared with r4dx_model_attention).

namespace {

// Reads just the safetensors-shell header (8-byte length + JSON) to pull out `__metadata__`.
// r4dx_convert::SafetensorsReader (reused below for the mmap'd tensor DATA reads) parses the same
// header but discards __metadata__ -- this is the small, deliberate amount of independent
// re-parsing docs/container-format.md's writer/reader contract implies (see container.h's file
// comment), not a second copy of the mmap machinery.
nlohmann::json ReadMetadata(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  if (!f) throw std::runtime_error("r4dx::model::Container: cannot open " + path);
  uint64_t header_len = 0;
  f.read(reinterpret_cast<char*>(&header_len), 8);
  if (!f) throw std::runtime_error("r4dx::model::Container: " + path + " too small for a header");
  std::string header_json(static_cast<size_t>(header_len), '\0');
  f.read(header_json.data(), static_cast<std::streamsize>(header_len));
  if (!f) throw std::runtime_error("r4dx::model::Container: " + path + " header truncated");
  nlohmann::json header = nlohmann::json::parse(header_json);
  if (!header.contains("__metadata__")) {
    throw std::runtime_error("r4dx::model::Container: " + path + " has no __metadata__");
  }
  return header.at("__metadata__");
}

// Refuses a container whose w4a16 group disagrees with the group this build's
// r4d_gemm_w4a16_nt_m64 was compiled with -- see CheckW4a16Group's comment in quant_linear.h for
// why a mismatch is silent-wrong-numbers rather than a crash.
//
// A container written before the group was recorded has no `quant` block at all; those are group
// 128 by construction (it was the only group that ever existed) and load unchanged on a group-128
// build, so nothing that loads today stops loading.
//
// SCOPE (adversarial-review fix): the check fires only when THIS load will actually read
// `.w4a16.wsz` bytes, i.e. when one of the three layout selections below is `kW4a16`. The hazard
// the guard exists to stop is a w4a16 GEMM striding the scales wrongly; a `--layout mxfp4` or
// `--layout w4a8` run never touches a `.w4a16.*` tensor (LoadQuantLinear reads only the requested
// layout's tensors, and LoadQuantLinearWithFallback's fallback chain is requested -> bf16 -> bare,
// never -> w4a16), and w4a8's and mxfp4's own groups do not move with R4DX_W4A16_GROUP. Refusing
// those runs bought no safety and cost real capability: `qwen38-27b-v5.r4dx` carries perfectly
// valid w4a8 and mxfp4 layouts that a group-64 build can read byte-for-byte correctly.
void CheckQuantGroups(const nlohmann::json& metadata, const std::string& path, Layout layout,
                       Layout lm_head_layout, Layout mtp_head_layout) {
  if (layout != Layout::kW4a16 && lm_head_layout != Layout::kW4a16 &&
      mtp_head_layout != Layout::kW4a16) {
    return;
  }
  if (!metadata.contains("quant")) return;
  const nlohmann::json& quant = metadata.at("quant");
  if (!quant.contains("w4a16") || !quant.at("w4a16").contains("group")) return;
  CheckW4a16Group(quant.at("w4a16").at("group").get<int>(), path);
}

using r4dx_convert::SafetensorsReader;
using r4dx_convert::Utf8ToWide;

int64_t ElemCountBySize(const SafetensorsReader& r, const std::string& name, int64_t elem_bytes) {
  const auto& m = r.Meta(name);
  const uint64_t span = m.end - m.begin;
  if (span % static_cast<uint64_t>(elem_bytes) != 0) {
    throw std::runtime_error("r4dx::model::Container: tensor '" + name +
                              "' byte span is not a multiple of " + std::to_string(elem_bytes));
  }
  return static_cast<int64_t>(span / static_cast<uint64_t>(elem_bytes));
}

core::DeviceBuffer<uint16_t> UploadRawU16(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 2);
  core::DeviceBuffer<uint16_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const uint16_t*>(r.Data(name)), static_cast<size_t>(n));
  return buf;
}

core::DeviceBuffer<float> UploadRawF32(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 4);
  core::DeviceBuffer<float> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const float*>(r.Data(name)), static_cast<size_t>(n));
  return buf;
}

core::DeviceBuffer<uint8_t> UploadRawU8(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 1);
  core::DeviceBuffer<uint8_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const uint8_t*>(r.Data(name)), static_cast<size_t>(n));
  return buf;
}

core::DeviceBuffer<int8_t> UploadRawI8(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 1);
  core::DeviceBuffer<int8_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const int8_t*>(r.Data(name)), static_cast<size_t>(n));
  return buf;
}

core::DeviceBuffer<uint32_t> UploadRawU32(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 4);
  core::DeviceBuffer<uint32_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const uint32_t*>(r.Data(name)), static_cast<size_t>(n));
  return buf;
}

core::DeviceBuffer<int32_t> UploadRawI32(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 4);
  core::DeviceBuffer<int32_t> buf(static_cast<size_t>(n));
  buf.CopyFromHost(reinterpret_cast<const int32_t*>(r.Data(name)), static_cast<size_t>(n));
  return buf;
}

// gdn.norm_weight is stored bf16 (docs/container-format.md), but both consumers
// (r4d_gdn_gated_rmsnorm_h128_bf16, r4d_gdn_recurrent_update_*) take a float* -- widen on load
// once rather than every layer call.
core::DeviceBuffer<float> UploadWidenedF32(const SafetensorsReader& r, const std::string& name) {
  const int64_t n = ElemCountBySize(r, name, 2);
  const auto* src = reinterpret_cast<const uint16_t*>(r.Data(name));
  std::vector<float> host(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) host[static_cast<size_t>(i)] = core::Bf16ToFloat(src[static_cast<size_t>(i)]);
  core::DeviceBuffer<float> buf(static_cast<size_t>(n));
  buf.CopyFromHost(host);
  return buf;
}

QuantLinear LoadQuantLinear(const SafetensorsReader& r, const std::string& base, Layout layout,
                             int64_t N, int64_t K) {
  QuantLinear q;
  q.layout = layout;
  q.N = N;
  q.K = K;
  switch (layout) {
    case Layout::kBf16:
      q.bf16_w = UploadRawU16(r, base + ".bf16.w");
      break;
    case Layout::kW4a16:
      q.wq = UploadRawU8(r, base + ".w4a16.wq");
      q.w4a16_wsz = UploadRawU32(r, base + ".w4a16.wsz");
      break;
    case Layout::kW4a8:
      q.wq = UploadRawU8(r, base + ".w4a8.wq");
      q.w4a8_ws = UploadRawU32(r, base + ".w4a8.ws");
      break;
    case Layout::kMxfp4:
      q.mxfp4_wq = UploadRawU8(r, base + ".mxfp4.wq");
      q.mxfp4_ws = UploadRawU8(r, base + ".mxfp4.ws");
      q.mxfp4_wref = UploadRawI8(r, base + ".mxfp4.wref");
      break;
  }
  return q;
}

// True iff `r` carries every tensor `LoadQuantLinear(r, base, layout, ...)` would read.
bool HasLayout(const SafetensorsReader& r, const std::string& base, Layout layout) {
  switch (layout) {
    case Layout::kBf16: return r.Has(base + ".bf16.w");
    case Layout::kW4a16: return r.Has(base + ".w4a16.wq") && r.Has(base + ".w4a16.wsz");
    case Layout::kW4a8: return r.Has(base + ".w4a8.wq") && r.Has(base + ".w4a8.ws");
    case Layout::kMxfp4:
      return r.Has(base + ".mxfp4.wq") && r.Has(base + ".mxfp4.ws") && r.Has(base + ".mxfp4.wref");
  }
  return false;
}

// R1 (docs/r9700.md): gdn.in_proj_z and attn.k/v join the quantized-linear family (previously
// bf16-only, no `.{layout}` suffix at all). Three tiers, in order:
//   1. `base` carries the requested `layout` -- load it normally (the common case for any
//      container converted with the new converter and --layouts including this layout).
//   2. `base` carries `.bf16.w` but not the requested layout (e.g. --layouts omitted this
//      quantized form, or the requested layout is bf16 itself) -- fall back to bf16 rather than
//      throwing, exactly like docs/container-format.md's other multi-layout linears already do
//      when a caller requests a layout the container didn't bake in.
//   3. `base` is a bare single tensor with no `.{layout}` suffix at all -- the OLD, pre-R1
//      on-disk form these three tensors used to have exclusively (every container converted
//      before this pass). Old containers keep working unmodified (task requirement).
//
// Milestone 11 (docs/validation.md "Milestone 11 / sensitivity"): EVERY quantized body linear now
// loads through this, not just the three R1 tensors. Tier 2 is what makes r4dx-convert's
// `--keep-bf16 <regex>` work -- that flag writes a matched linear as `<base>.bf16.w` and nothing
// else, so the container is quantized everywhere except the tensor class under test and this
// function is the only thing that has to notice. Before, those call sites used LoadQuantLinear
// directly and a bf16-only base was a hard "tensor not found" throw. `fallbacks`, when non-null, is
// incremented once per linear that did NOT have the requested layout, so Load() can report the
// count instead of falling back silently -- a container that accidentally quantized nothing and one
// that deliberately kept one class in bf16 must not look the same in a log.
QuantLinear LoadQuantLinearWithFallback(const SafetensorsReader& r, const std::string& base,
                                         Layout requested, int64_t N, int64_t K,
                                         int* fallbacks = nullptr) {
  if (HasLayout(r, base, requested)) return LoadQuantLinear(r, base, requested, N, K);
  if (fallbacks != nullptr) ++*fallbacks;
  if (HasLayout(r, base, Layout::kBf16)) return LoadQuantLinear(r, base, Layout::kBf16, N, K);
  if (r.Has(base)) {
    QuantLinear q;
    q.layout = Layout::kBf16;
    q.N = N;
    q.K = K;
    q.bf16_w = UploadRawU16(r, base);
    return q;
  }
  throw std::runtime_error("r4dx::model::Container: no tensor found for '" + base +
                            "' in any known on-disk form (requested layout, bf16, or bare)");
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

Container Container::Load(const std::string& path, const ContainerLoadOptions& o) {
  if (o.tp_world < 1 || o.tp_world > 2 || o.tp_rank < 0 || o.tp_rank >= o.tp_world) {
    throw std::invalid_argument("r4dx::model::Container::Load: need tp_world in {1, 2} and 0 <= "
                                "tp_rank < tp_world, got tp_world " + std::to_string(o.tp_world) +
                                ", tp_rank " + std::to_string(o.tp_rank));
  }
  if (o.tp_world > 1) return LoadShard(path, o);
  // tp_world == 1 (docs/tp.md 5.1 step 7): the pre-TP loader below, untouched -- no rule lookup, no
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
  CheckQuantGroups(metadata, path, layout, lm_head_layout, mtp_head_layout);
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

  SafetensorsReader reader(Utf8ToWide(path));

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
    lw.input_layernorm = UploadRawU16(reader, base + "input_layernorm");
    lw.post_attention_layernorm = UploadRawU16(reader, base + "post_attention_layernorm");

    if (c.config_.IsGdnLayer(i)) {
      GdnWeights g;
      g.in_proj_qkv = LoadQuantLinearWithFallback(reader, base + "gdn.in_proj_qkv", layout,
                                                   2 * key_dim + value_dim, hidden,
                                                   &bf16_fallbacks);
      g.in_proj_z = LoadQuantLinearWithFallback(reader, base + "gdn.in_proj_z", layout, value_dim,
                                                 hidden, &bf16_fallbacks);
      g.in_proj_b = UploadRawU16(reader, base + "gdn.in_proj_b");
      g.in_proj_a = UploadRawU16(reader, base + "gdn.in_proj_a");
      g.conv1d_weight = UploadRawU16(reader, base + "gdn.conv1d_weight");
      g.A_log = UploadRawF32(reader, base + "gdn.A_log");
      g.dt_bias = UploadRawF32(reader, base + "gdn.dt_bias");
      g.norm_weight = UploadWidenedF32(reader, base + "gdn.norm_weight");
      g.out_proj = LoadQuantLinearWithFallback(reader, base + "gdn.out_proj", layout, hidden,
                                                value_dim, &bf16_fallbacks);
      lw.gdn = std::move(g);
    } else {
      AttnWeights a;
      // attn.qg/o now honor the requested body `layout` the same way GDN's in_proj_qkv/out_proj
      // and MLP's gate_up/down do (decode-perf pass, 2026-09-19): AttentionLayer dispatches both
      // through the shared r4dx::model::ApplyLinear (src/model/linear.h), which every layout
      // already supports. See docs/perf.md for the measured per-layout VRAM/throughput delta this
      // unlocks (attn.qg/o account for 16 of 64 layers' full-attention projections).
      a.qg = LoadQuantLinearWithFallback(reader, base + "attn.qg", layout, attn_out * 2, hidden,
                                          &bf16_fallbacks);
      a.k = LoadQuantLinearWithFallback(reader, base + "attn.k", layout,
                                         kv_heads * c.config_.head_dim, hidden, &bf16_fallbacks);
      a.v = LoadQuantLinearWithFallback(reader, base + "attn.v", layout,
                                         kv_heads * c.config_.head_dim, hidden, &bf16_fallbacks);
      a.o = LoadQuantLinearWithFallback(reader, base + "attn.o", layout, hidden, attn_out,
                                         &bf16_fallbacks);
      a.q_norm = UploadRawU16(reader, base + "attn.q_norm");
      a.k_norm = UploadRawU16(reader, base + "attn.k_norm");
      a.k_descale = UploadRawF32(reader, base + "attn.k_descale");
      a.v_descale = UploadRawF32(reader, base + "attn.v_descale");
      lw.attn = std::move(a);
    }

    lw.mlp.gate_up = LoadQuantLinearWithFallback(reader, base + "mlp.gate_up", layout,
                                                  2 * c.config_.intermediate_size, hidden,
                                                  &bf16_fallbacks);
    lw.mlp.down = LoadQuantLinearWithFallback(reader, base + "mlp.down", layout, hidden,
                                               c.config_.intermediate_size, &bf16_fallbacks);
    c.layers_.push_back(std::move(lw));
  }

  c.final_norm_ = UploadRawU16(reader, "text.final_norm");
  // With fallback: a container converted with `--lm-head bf16` (rung 4 follow-up -- the 4-bit
  // lm_head is where the vocab-tail KL loss concentrates) carries only lm_head.bf16.w, and every
  // caller that asks for the body layout here should get that bf16 head rather than a throw.
  c.lm_head_ = LoadQuantLinearWithFallback(reader, "lm_head", lm_head_layout, c.config_.vocab_size,
                                           hidden, &bf16_fallbacks);

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
    a.qg = LoadQuantLinearWithFallback(reader, base + "attn.qg", mtp_head_layout, attn_out * 2,
                                        hidden, &bf16_fallbacks);
    // mtp.attn.k/v stay bf16 regardless of the requested body/head layout (docs/r9700.md's R1
    // task: "Keep mtp.* ... as they are") -- request Layout::kBf16 explicitly rather than
    // `layout`/`mtp_head_layout`, so this never picks up a quantized form even if a future
    // converter run ever quantized mtp.*. LoadQuantLinearWithFallback (not UploadRawU16) so this
    // still works against the OLD bare-tensor on-disk form (pre-R1 containers) as well as any
    // future `.bf16.w`-suffixed form -- see LoadQuantLinearWithFallback's own comment.
    a.k = LoadQuantLinearWithFallback(reader, base + "attn.k", Layout::kBf16,
                                       kv_heads * c.config_.head_dim, hidden);
    a.v = LoadQuantLinearWithFallback(reader, base + "attn.v", Layout::kBf16,
                                       kv_heads * c.config_.head_dim, hidden);
    a.o = LoadQuantLinearWithFallback(reader, base + "attn.o", mtp_head_layout, hidden, attn_out,
                                       &bf16_fallbacks);
    a.q_norm = UploadRawU16(reader, base + "attn.q_norm");
    a.k_norm = UploadRawU16(reader, base + "attn.k_norm");
    a.k_descale = UploadRawF32(reader, base + "attn.k_descale");
    a.v_descale = UploadRawF32(reader, base + "attn.v_descale");
    lw.attn = std::move(a);
    lw.mlp.gate_up = LoadQuantLinearWithFallback(reader, base + "mlp.gate_up", mtp_head_layout,
                                                  2 * c.config_.intermediate_size, hidden,
                                                  &bf16_fallbacks);
    lw.mlp.down = LoadQuantLinearWithFallback(reader, base + "mlp.down", mtp_head_layout, hidden,
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
      mw.draft_lm_head = LoadQuantLinearWithFallback(reader, "mtp.draft_head.lm_head",
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

namespace {

struct ShardLoadStats {
  int sharded = 0;     // on-disk tensors this rank uploaded a slice of
  int replicated = 0;  // on-disk tensors this rank uploaded whole
  uint64_t uploaded_bytes = 0;
  uint64_t staged_bytes = 0;  // the part of uploaded_bytes gathered through the host staging buffer
};

// The byte size a WHOLE on-disk part of logical shape [N, K] has (docs/tp.md 4.3, the converter's
// packers). A tensor whose span disagrees is refused rather than sliced with the wrong strides.
uint64_t PartBytes(const tp::PartShape& s) {
  const uint64_t N = static_cast<uint64_t>(s.N), K = static_cast<uint64_t>(s.K);
  switch (s.part) {
    case tp::Part::kBf16: return N * K * 2;
    case tp::Part::kW4Wq:
    case tp::Part::kMxWq: return N * K / 2;
    case tp::Part::kW4a16Wsz:
    case tp::Part::kW4a8Ws: return N * (K / static_cast<uint64_t>(s.group)) * 4;
    case tp::Part::kMxWs: return (K / static_cast<uint64_t>(s.group)) * N;
    case tp::Part::kMxWref: return N;
    case tp::Part::kElem: return N * static_cast<uint64_t>(s.row_bytes);
  }
  return 0;
}

// One rank's view of the container: every tensor is classified by tp::RuleFor on its base name and
// only this rank's byte runs (tp::PlanRows/PlanCols) are uploaded -- straight from the mmap when the
// plan is one contiguous range, through one reusable host staging buffer otherwise (docs/tp.md 5.1
// step 4). Every name must be known to RuleFor: an unclassified tensor throws, it is never silently
// replicated.
class ShardLoader {
 public:
  ShardLoader(const SafetensorsReader& r, const ModelConfig& global, int world, int rank,
              int w4a16_group)
      : r_(r), global_(global), world_(world), rank_(rank), w4a16_group_(w4a16_group) {}

  // A tensor with one on-disk form and no layout suffix (norms, gdn.in_proj_a/b, conv1d_weight,
  // A_log, dt_bias, the descales, mtp.fc, ...): its whole bytes when the rule replicates, else this
  // rank's row ranges of the row-major [N, ...] array (N = the rule's total rows).
  template <class T>
  core::DeviceBuffer<T> Raw(const std::string& name) {
    const tp::ShardRule rule = tp::RuleFor(name, global_);
    const uint64_t span = Span(name);
    if (rule.split == tp::Split::kReplicate) return Upload<T>(name, {tp::ByteRun{0, span}}, false);
    if (rule.split != tp::Split::kRows) {
      throw std::logic_error("r4dx::model::Container: '" + name +
                             "' has no layout suffix but its tensor-parallel rule is not a row split "
                             "or a replication");
    }
    int64_t rows = 0;
    for (const tp::Segment& s : rule.segments) rows += s.rows;
    if (rows <= 0 || span % static_cast<uint64_t>(rows) != 0) {
      throw std::runtime_error("r4dx::model::Container: tensor '" + name + "' (" +
                               std::to_string(span) + " bytes) is not " + std::to_string(rows) +
                               " equal rows, the row count its tensor-parallel rule splits");
    }
    tp::PartShape shape;
    shape.part = tp::Part::kElem;
    shape.N = rows;
    shape.row_bytes = static_cast<int64_t>(span / static_cast<uint64_t>(rows));
    return Upload<T>(name, tp::PlanRows(shape, tp::RankRows(rule, world_, rank_)), true);
  }

  // UploadWidenedF32's counterpart (gdn.norm_weight, bf16 on disk, fp32 on device): replicated.
  core::DeviceBuffer<float> WidenedF32(const std::string& name) {
    if (tp::RuleFor(name, global_).split != tp::Split::kReplicate) {
      throw std::logic_error("r4dx::model::Container: '" + name + "' is widened on load, which "
                             "only a replicated tensor supports");
    }
    core::DeviceBuffer<float> buf = UploadWidenedF32(r_, name);
    ++stats_.replicated;
    stats_.uploaded_bytes += buf.bytes();
    return buf;
  }

  // LoadQuantLinearWithFallback's sharded counterpart: the same requested -> .bf16.w -> bare
  // on-disk form chain, then this rank's slice of every part of that form. `N`/`K` are the GLOBAL
  // logical shape; the returned QuantLinear carries the RANK's (docs/tp.md 5.1 step 4).
  QuantLinear Linear(const std::string& base, Layout requested, int64_t N, int64_t K,
                     int* fallbacks = nullptr) {
    Layout form = requested;
    bool bare = false;
    if (!HasLayout(r_, base, requested)) {
      if (fallbacks != nullptr) ++*fallbacks;
      form = Layout::kBf16;
      if (!HasLayout(r_, base, Layout::kBf16)) {
        if (!r_.Has(base)) {
          throw std::runtime_error("r4dx::model::Container: no tensor found for '" + base +
                                   "' in any known on-disk form (requested layout, bf16, or bare)");
        }
        bare = true;
      }
    }

    const tp::ShardRule rule = tp::RuleFor(base, global_);
    QuantLinear q;
    q.layout = form;
    q.N = N;
    q.K = K;
    std::vector<tp::Range> rows;
    tp::Range cols{0, K};
    if (rule.split == tp::Split::kRows) {
      int64_t total = 0;
      for (const tp::Segment& s : rule.segments) total += s.rows;
      if (total != N) {
        throw std::logic_error("r4dx::model::Container: '" + base + "' has N = " +
                               std::to_string(N) + " but its tensor-parallel rule splits " +
                               std::to_string(total) + " rows");
      }
      rows = tp::RankRows(rule, world_, rank_);
      q.N = 0;
      for (const tp::Range& r : rows) q.N += r.count;
    } else if (rule.split == tp::Split::kCols) {
      if (rule.k_total != K) {
        throw std::logic_error("r4dx::model::Container: '" + base + "' has K = " +
                               std::to_string(K) + " but its tensor-parallel rule splits K = " +
                               std::to_string(rule.k_total));
      }
      cols = tp::RankCols(rule, world_, rank_);
      q.K = cols.count;
    } else if (rule.split != tp::Split::kReplicate) {
      throw std::logic_error("r4dx::model::Container: '" + base +
                             "' is not a text/mtp linear (rank-0-only rule)");
    }

    const auto shape = [&](tp::Part p, int group) {
      tp::PartShape s;
      s.part = p;
      s.N = N;
      s.K = K;
      s.group = group;
      return s;
    };
    switch (form) {
      case Layout::kBf16:
        q.bf16_w = Part<uint16_t>(bare ? base : base + ".bf16.w", shape(tp::Part::kBf16, 0), rule,
                                  rows, cols);
        break;
      case Layout::kW4a16:
        q.wq = Part<uint8_t>(base + ".w4a16.wq", shape(tp::Part::kW4Wq, 0), rule, rows, cols);
        q.w4a16_wsz = Part<uint32_t>(base + ".w4a16.wsz", shape(tp::Part::kW4a16Wsz, w4a16_group_),
                                     rule, rows, cols);
        break;
      case Layout::kW4a8:
        q.wq = Part<uint8_t>(base + ".w4a8.wq", shape(tp::Part::kW4Wq, 0), rule, rows, cols);
        q.w4a8_ws = Part<uint32_t>(base + ".w4a8.ws", shape(tp::Part::kW4a8Ws, 128), rule, rows,
                                   cols);
        break;
      case Layout::kMxfp4:
        q.mxfp4_wq = Part<uint8_t>(base + ".mxfp4.wq", shape(tp::Part::kMxWq, 0), rule, rows, cols);
        q.mxfp4_ws = Part<uint8_t>(base + ".mxfp4.ws", shape(tp::Part::kMxWs, 32), rule, rows, cols);
        // K-slice: the FULL-row wref (docs/tp.md 4.3 "The mxfp4 wref exception"), which PlanCols
        // returns for this part.
        q.mxfp4_wref = Part<int8_t>(base + ".mxfp4.wref", shape(tp::Part::kMxWref, 0), rule, rows,
                                    cols);
        break;
    }
    return q;
  }

  // The embedding's device mirror is uploaded by LoadShard itself; this only counts it.
  void CountReplicated(uint64_t bytes) {
    ++stats_.replicated;
    stats_.uploaded_bytes += bytes;
  }
  const ShardLoadStats& Stats() const { return stats_; }

 private:
  uint64_t Span(const std::string& name) const {
    const auto& m = r_.Meta(name);
    return m.end - m.begin;
  }

  template <class T>
  core::DeviceBuffer<T> Part(const std::string& name, const tp::PartShape& shape,
                             const tp::ShardRule& rule, const std::vector<tp::Range>& rows,
                             tp::Range cols) {
    const uint64_t span = Span(name);
    if (span != PartBytes(shape)) {
      throw std::runtime_error("r4dx::model::Container: tensor '" + name + "' is " +
                               std::to_string(span) + " bytes, but its [" +
                               std::to_string(shape.N) + ", " + std::to_string(shape.K) +
                               "] layout needs " + std::to_string(PartBytes(shape)));
    }
    switch (rule.split) {
      case tp::Split::kRows: return Upload<T>(name, tp::PlanRows(shape, rows), true);
      case tp::Split::kCols: return Upload<T>(name, tp::PlanCols(shape, cols), true);
      default: return Upload<T>(name, {tp::ByteRun{0, span}}, false);
    }
  }

  template <class T>
  core::DeviceBuffer<T> Upload(const std::string& name, const std::vector<tp::ByteRun>& runs,
                               bool sharded) {
    const uint64_t span = Span(name);
    const uint8_t* src = r_.Data(name);
    size_t total = 0;
    for (const tp::ByteRun& run : runs) {
      if (run.src_off > span || run.bytes > span - run.src_off) {
        throw std::out_of_range("r4dx::model::Container: a byte run of '" + name +
                                "' reaches past its " + std::to_string(span) + " bytes");
      }
      total += run.bytes;
    }
    if (total == 0 || total % sizeof(T) != 0) {
      throw std::runtime_error("r4dx::model::Container: this rank's slice of '" + name + "' is " +
                               std::to_string(total) + " bytes, not a positive multiple of " +
                               std::to_string(sizeof(T)));
    }
    const size_t count = total / sizeof(T);
    core::DeviceBuffer<T> buf(count);
    if (runs.size() == 1) {
      // One contiguous range: straight from the mmap, no staging copy.
      buf.CopyFromHost(reinterpret_cast<const T*>(src + runs[0].src_off), count);
    } else {
      if (staging_.size() < total) staging_.resize(total);
      size_t o = 0;
      for (const tp::ByteRun& run : runs) {
        std::memcpy(staging_.data() + o, src + run.src_off, run.bytes);
        o += run.bytes;
      }
      buf.CopyFromHost(reinterpret_cast<const T*>(staging_.data()), count);
      stats_.staged_bytes += total;
    }
    ++(sharded ? stats_.sharded : stats_.replicated);
    stats_.uploaded_bytes += total;
    return buf;
  }

  const SafetensorsReader& r_;
  const ModelConfig& global_;
  int world_, rank_, w4a16_group_;
  std::vector<uint8_t> staging_;  // grown to the largest gathered tensor; freed with the loader
  ShardLoadStats stats_;
};

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
  CheckQuantGroups(metadata, path, o.layout, o.lm_head_layout, o.mtp_head_layout);
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
  // The group the container's w4a16 scales were packed at -- the wsz stride per 16-row tile
  // (docs/tp.md 4.3). A container that predates the `quant` block is group 128, CheckQuantGroups'
  // own rule.
  int w4a16_group = 128;
  if (metadata.contains("quant") && metadata.at("quant").contains("w4a16") &&
      metadata.at("quant").at("w4a16").contains("group")) {
    w4a16_group = metadata.at("quant").at("w4a16").at("group").get<int>();
  }

  SafetensorsReader reader(Utf8ToWide(path));
  const ModelConfig& gc = c.global_config_;
  ShardLoader L(reader, gc, o.tp_world, o.tp_rank, w4a16_group);
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
    lw.input_layernorm = L.Raw<uint16_t>(base + "input_layernorm");
    lw.post_attention_layernorm = L.Raw<uint16_t>(base + "post_attention_layernorm");
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
  c.lm_head_ = L.Linear("lm_head", o.lm_head_layout, gc.vocab_size, hidden, &bf16_fallbacks);

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
