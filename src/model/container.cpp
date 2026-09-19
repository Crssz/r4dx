#include "container.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/safetensors_reader.hpp"  // SafetensorsReader, Utf8ToWide -- see file comment

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

}  // namespace

Container Container::Load(const std::string& path, Layout layout, Layout lm_head_layout,
                           int64_t layer_limit) {
  Container c;
  const nlohmann::json metadata = ReadMetadata(path);
  c.model_id_ = metadata.value("model_id", std::string());
  c.config_sha256_ = metadata.value("config_sha256", std::string());
  const nlohmann::json& model_config = metadata.at("model_config");
  const nlohmann::json& text_cfg = model_config.contains("text_config")
                                        ? model_config.at("text_config")
                                        : model_config;  // selftest containers have no text_config
  c.config_ = ModelConfig::FromJson(text_cfg);

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
  }

  const int64_t hidden = c.config_.hidden_size;
  const int64_t key_dim = c.config_.KeyDim();
  const int64_t value_dim = c.config_.ValueDim();
  const int64_t kv_heads = c.config_.num_key_value_heads;
  const int64_t attn_out = c.config_.num_attention_heads * c.config_.head_dim;

  c.layers_.reserve(static_cast<size_t>(num_layers));
  for (int64_t i = 0; i < num_layers; ++i) {
    const std::string base = "text.layers." + std::to_string(i) + ".";
    LayerWeights lw;
    lw.input_layernorm = UploadRawU16(reader, base + "input_layernorm");
    lw.post_attention_layernorm = UploadRawU16(reader, base + "post_attention_layernorm");

    if (c.config_.IsGdnLayer(i)) {
      GdnWeights g;
      g.in_proj_qkv = LoadQuantLinear(reader, base + "gdn.in_proj_qkv", layout,
                                       2 * key_dim + value_dim, hidden);
      g.in_proj_z = UploadRawU16(reader, base + "gdn.in_proj_z");
      g.in_proj_b = UploadRawU16(reader, base + "gdn.in_proj_b");
      g.in_proj_a = UploadRawU16(reader, base + "gdn.in_proj_a");
      g.conv1d_weight = UploadRawU16(reader, base + "gdn.conv1d_weight");
      g.A_log = UploadRawF32(reader, base + "gdn.A_log");
      g.dt_bias = UploadRawF32(reader, base + "gdn.dt_bias");
      g.norm_weight = UploadWidenedF32(reader, base + "gdn.norm_weight");
      g.out_proj = LoadQuantLinear(reader, base + "gdn.out_proj", layout, hidden, value_dim);
      lw.gdn = std::move(g);
    } else {
      AttnWeights a;
      // attn.qg/o now honor the requested body `layout` the same way GDN's in_proj_qkv/out_proj
      // and MLP's gate_up/down do (decode-perf pass, 2026-09-19): AttentionLayer dispatches both
      // through the shared r4dx::model::ApplyLinear (src/model/linear.h), which every layout
      // already supports. See docs/perf.md for the measured per-layout VRAM/throughput delta this
      // unlocks (attn.qg/o account for 16 of 64 layers' full-attention projections).
      a.qg = LoadQuantLinear(reader, base + "attn.qg", layout, attn_out * 2, hidden);
      a.k = UploadRawU16(reader, base + "attn.k");
      a.v = UploadRawU16(reader, base + "attn.v");
      a.o = LoadQuantLinear(reader, base + "attn.o", layout, hidden, attn_out);
      a.q_norm = UploadRawU16(reader, base + "attn.q_norm");
      a.k_norm = UploadRawU16(reader, base + "attn.k_norm");
      a.k_descale = UploadRawF32(reader, base + "attn.k_descale");
      a.v_descale = UploadRawF32(reader, base + "attn.v_descale");
      (void)kv_heads;
      lw.attn = std::move(a);
    }

    lw.mlp.gate_up = LoadQuantLinear(reader, base + "mlp.gate_up", layout,
                                      2 * c.config_.intermediate_size, hidden);
    lw.mlp.down = LoadQuantLinear(reader, base + "mlp.down", layout, hidden,
                                   c.config_.intermediate_size);
    c.layers_.push_back(std::move(lw));
  }

  c.final_norm_ = UploadRawU16(reader, "text.final_norm");
  c.lm_head_ = LoadQuantLinear(reader, "lm_head", lm_head_layout, c.config_.vocab_size, hidden);

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
    AttnWeights a;
    a.qg = LoadQuantLinear(reader, base + "attn.qg", layout, attn_out * 2, hidden);
    a.k = UploadRawU16(reader, base + "attn.k");
    a.v = UploadRawU16(reader, base + "attn.v");
    a.o = LoadQuantLinear(reader, base + "attn.o", layout, hidden, attn_out);
    a.q_norm = UploadRawU16(reader, base + "attn.q_norm");
    a.k_norm = UploadRawU16(reader, base + "attn.k_norm");
    a.k_descale = UploadRawF32(reader, base + "attn.k_descale");
    a.v_descale = UploadRawF32(reader, base + "attn.v_descale");
    lw.attn = std::move(a);
    lw.mlp.gate_up = LoadQuantLinear(reader, base + "mlp.gate_up", layout,
                                      2 * c.config_.intermediate_size, hidden);
    lw.mlp.down = LoadQuantLinear(reader, base + "mlp.down", layout, hidden,
                                   c.config_.intermediate_size);
    mw.layer = std::move(lw);
    mw.fc = UploadRawU16(reader, "mtp.fc");
    mw.norm = UploadRawU16(reader, "mtp.norm");
    mw.pre_fc_norm_hidden = UploadRawU16(reader, "mtp.pre_fc_norm_hidden");
    mw.pre_fc_norm_embedding = UploadRawU16(reader, "mtp.pre_fc_norm_embedding");
    c.mtp_ = std::move(mw);
  }

  return c;
}

}  // namespace r4dx::model
