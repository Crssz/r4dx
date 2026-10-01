#include "gemma_container.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "container_load_util.h"
#include "r4dx/core/r4d.hpp"  // GemmTrellisTicketsBytes / GemmTrellisZeroTickets

namespace r4dx::model {

using namespace container_util;

GemmaContainerInfo GemmaContainer::Inspect(const std::string& path, int64_t layer_limit) {
  SafetensorsReader reader(Utf8ToWide(path));
  const nlohmann::json header = nlohmann::json::parse(reader.HeaderJson());
  if (!header.contains("__metadata__")) {
    throw std::runtime_error("r4dx::model::GemmaContainer: " + path + " has no __metadata__");
  }
  GemmaContainerInfo info = ParseGemmaContainerMetadata(header.at("__metadata__"), path);
  const int64_t layers = layer_limit >= 0 ? std::min(layer_limit, info.config.num_hidden_layers)
                                          : info.config.num_hidden_layers;
  const std::vector<std::string> bad = GemmaTensorTableProblems(reader, info, layers, /*bf16_body=*/false);
  if (!bad.empty()) {
    std::string msg = "r4dx::model::GemmaContainer: " + path + ": the tensor table does not match the model (" +
                      std::to_string(bad.size()) + " problem(s)):";
    for (size_t i = 0; i < bad.size() && i < 8; ++i) msg += "\n  " + bad[i];
    throw std::runtime_error(msg);
  }
  return info;
}

GemmaContainer GemmaContainer::Load(const std::string& path, const GemmaLoadOptions& o_in) {
  GemmaLoadOptions o = o_in;
  if (o.lm_head_layout == Layout::kTrellis) o.lm_head_layout = Layout::kW4a16;  // heads are never trellis
  GemmaContainer c;
  c.info_ = Inspect(path, o.layer_limit);
  const GemmaConfig& cfg = c.info_.config;
  const nlohmann::json& metadata = c.info_.metadata;

  LinearLoadMeta meta;
  meta.w4a16 = CheckQuantGroups(metadata, path, o.layout, o.lm_head_layout, Layout::kBf16);
  meta.path = path;
  meta.trellis = ParseTrellisMetadata(metadata, path);
  CheckTrellisChoice(meta.trellis, c.info_.rotation.has_value(), o.layout, path,
                     /*allow_rotated=*/true);

  SafetensorsReader reader(Utf8ToWide(path));
  CheckW4a16GroupTensors(reader, meta.w4a16.groups, path);
  if (meta.w4a16.check_default_per_linear) LogW4a16Groups(meta.w4a16.groups, path);
  CheckTrellisTensors(reader, meta.trellis, path);

  const int64_t layers = o.layer_limit >= 0 ? std::min(o.layer_limit, cfg.num_hidden_layers) : cfg.num_hidden_layers;
  const int64_t hidden = cfg.hidden_size;
  const std::string rot = c.info_.rotation ? ".rotated" : "";

  c.embed_tokens_ = UploadRawU16(reader, "text.embed_tokens");
  int fallbacks = 0;
  c.layers_.reserve(static_cast<size_t>(layers));
  for (int64_t i = 0; i < layers; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    GemmaLayerWeights lw;
    lw.full = cfg.IsFullLayer(i);
    lw.input_layernorm = UploadRawU16(reader, b + "input_layernorm" + rot);
    lw.post_attention_layernorm = UploadRawU16(reader, b + "post_attention_layernorm");
    lw.pre_feedforward_layernorm = UploadRawU16(reader, b + "pre_feedforward_layernorm" + rot);
    lw.post_feedforward_layernorm = UploadRawU16(reader, b + "post_feedforward_layernorm");
    GemmaAttnWeights& a = lw.attn;
    a.q = LoadQuantLinearWithFallback(reader, meta, b + "attn.q", o.layout, cfg.QDim(i), hidden, &fallbacks);
    a.k = LoadQuantLinearWithFallback(reader, meta, b + "attn.k", o.layout, cfg.KvDim(i), hidden, &fallbacks);
    if (!lw.full) {
      a.v = LoadQuantLinearWithFallback(reader, meta, b + "attn.v", o.layout, cfg.KvDim(i), hidden, &fallbacks);
    }
    a.o = LoadQuantLinearWithFallback(reader, meta, b + "attn.o", o.layout, hidden, cfg.QDim(i), &fallbacks);
    a.q_norm = UploadRawU16(reader, b + "attn.q_norm");
    a.k_norm = UploadRawU16(reader, b + "attn.k_norm");
    a.k_descale = UploadRawF32(reader, b + "attn.k_descale");
    a.v_descale = UploadRawF32(reader, b + "attn.v_descale");
    lw.mlp.gate_up = LoadQuantLinearWithFallback(reader, meta, b + "mlp.gate_up", o.layout,
                                                 2 * cfg.intermediate_size, hidden, &fallbacks);
    lw.mlp.down = LoadQuantLinearWithFallback(reader, meta, b + "mlp.down", o.layout, hidden,
                                              cfg.intermediate_size, &fallbacks);
    // layer_scalar: fp32 [1] on disk (a bf16 value widened); the fused post-norm kernel takes it by value.
    if (ElemCountBySize(reader, b + "layer_scalar", 4) != 1) {
      throw std::runtime_error("r4dx::model::GemmaContainer: " + path + ": " + b + "layer_scalar is not one element");
    }
    std::memcpy(&lw.layer_scalar, reader.Data(b + "layer_scalar"), sizeof(float));
    c.layers_.push_back(std::move(lw));
  }
  c.final_norm_ = UploadRawU16(reader, "text.final_norm");
  const int fb_before_head = fallbacks;
  c.lm_head_ = LoadQuantLinearWithFallback(reader, meta, "lm_head", o.lm_head_layout, cfg.vocab_size, hidden,
                                           &fallbacks);
  const bool trellis_bf16_head = TakeTrellisBf16Head(meta, c.lm_head_, fb_before_head, &fallbacks);
  if (c.info_.rotation) {
    c.rotation_ = LoadRotationWeights(
        reader, *c.info_.rotation, c.info_.model_config, c.info_.model_config, path,
        [&](const std::string& name) { return UploadRawF32(reader, name); }, c.info_.full_attn_out_elems);
    LogRotation(*c.info_.rotation, path);
  }
  if (fallbacks > 0) {
    std::fprintf(stderr,
                 "r4dx: %d linear(s) in %s do not carry the requested layout and were loaded as bf16\n",
                 fallbacks, path.c_str());
  }
  if (meta.trellis) {
    c.trellis_ = std::move(meta.trellis);
    c.AssignTrellisTickets();
    LogTrellis(*c.trellis_, path);
    if (trellis_bf16_head) LogTrellisBf16Head(path);
  }
  return c;
}

template <class Fn>
void GemmaContainer::ForEachLinear(Fn&& fn) {
  for (GemmaLayerWeights& lw : layers_) {
    fn(lw.attn.q);
    fn(lw.attn.k);
    fn(lw.attn.v);
    fn(lw.attn.o);
    fn(lw.mlp.gate_up);
    fn(lw.mlp.down);
  }
  fn(lm_head_);
}

void GemmaContainer::AssignTrellisTickets() {
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

void GemmaContainer::ZeroTrellisTickets(hipStream_t stream) {
  if (trellis_tickets_.empty()) return;
  core::r4d::GemmTrellisZeroTickets(trellis_tickets_.data(), trellis_tickets_.bytes(), stream);
}

}  // namespace r4dx::model
