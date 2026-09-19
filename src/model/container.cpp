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

const char* LayoutName(Layout l) {
  switch (l) {
    case Layout::kBf16: return "bf16";
    case Layout::kMxfp4: return "mxfp4";
    case Layout::kW4a16: return "w4a16";
    case Layout::kW4a8: return "w4a8";
  }
  return "?";
}

Layout LayoutFromName(const std::string& name) {
  if (name == "bf16") return Layout::kBf16;
  if (name == "mxfp4") return Layout::kMxfp4;
  if (name == "w4a16") return Layout::kW4a16;
  if (name == "w4a8") return Layout::kW4a8;
  throw std::runtime_error("r4dx::model::LayoutFromName: unrecognized layout '" + name + "'");
}

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

  // attn.qg/o always load bf16 regardless of `layout` -- see the loop below's comment. Surface
  // that at load time (not just in a source comment / docs/perf.md's "Known limitation" section
  // far below its own numbers) so `--layout mxfp4` (etc.) doesn't silently run ~3GiB of attention
  // projections at bf16 without the caller knowing why the VRAM/quality numbers don't fully match
  // the requested layout.
  if (layout != Layout::kBf16) {
    int64_t num_attn_layers = 0;
    for (int64_t i = 0; i < num_layers; ++i) {
      if (!c.config_.IsGdnLayer(i)) ++num_attn_layers;
    }
    if (num_attn_layers > 0) {
      std::fprintf(stderr,
                    "note: attn.qg/o load as bf16 regardless of --layout=%s (%lld full-attention "
                    "layer%s)\n",
                    LayoutName(layout), static_cast<long long>(num_attn_layers),
                    num_attn_layers == 1 ? "" : "s");
    }
  }

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
      // attn.qg/o are forced to bf16 regardless of the requested body `layout`: the assembly
      // stage's AttentionLayer (src/model/attention/**) only implements a bf16 GEMM dispatch for
      // these two linears (its own documented interim scope -- see
      // src/model/attention/include/r4dx/model/attention/linear.hpp's file comment and this
      // component's task open_issues). Every r4dx-convert run this project uses always emits the
      // bf16 variant for attn.qg/o alongside whichever quantized layouts were requested (verified
      // against the real D:\models\r4dx\qwen38-27b.r4dx container), so this is always loadable.
      // GDN's in_proj_qkv/out_proj and MLP's gate_up/down still honor `layout` -- only these two
      // attention linears (16 of 64 layers) stay bf16-precision until AttentionLayer gains
      // quantized-layout dispatch (see docs/perf.md's notes for this run's measured impact).
      a.qg = LoadQuantLinear(reader, base + "attn.qg", Layout::kBf16, attn_out * 2, hidden);
      a.k = UploadRawU16(reader, base + "attn.k");
      a.v = UploadRawU16(reader, base + "attn.v");
      a.o = LoadQuantLinear(reader, base + "attn.o", Layout::kBf16, hidden, attn_out);
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

  return c;
}

}  // namespace r4dx::model
