#include "gemma_container.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <optional>
#include <stdexcept>

#include "container_load_util.h"
#include "shard_loader.h"
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
  if (o.tp_world < 1 || o.tp_world > 2 || o.tp_rank < 0 || o.tp_rank >= o.tp_world) {
    throw std::invalid_argument("r4dx::model::GemmaContainer::Load: need tp_world in {1, 2} and 0 <= tp_rank < "
                                "tp_world, got tp_world " + std::to_string(o.tp_world) + ", tp_rank " +
                                std::to_string(o.tp_rank));
  }
  const bool tp = o.tp_world > 1;
  GemmaContainer c;
  c.info_ = Inspect(path, o.layer_limit);
  const GemmaConfig& cfg = c.info_.config;  // GLOBAL: every ShardLoader / loader call takes the global [N, K]
  // The rank's config (docs/gemma4-plan.md 3.2): throws, naming the field, unless every per-rank shape is legal.
  c.local_config_ = tp ? GemmaConfig::Shard(cfg, o.tp_world, o.tp_rank) : cfg;
  const nlohmann::json& metadata = c.info_.metadata;

  LinearLoadMeta meta;
  meta.w4a16 = CheckQuantGroups(metadata, path, o.layout, o.lm_head_layout, Layout::kBf16);
  meta.path = path;
  meta.trellis = ParseTrellisMetadata(metadata, path);
  CheckTrellisChoice(meta.trellis, c.info_.rotation.has_value(), o.layout, path,
                     /*allow_rotated=*/true);

  SafetensorsReader reader(Utf8ToWide(path));
  CheckW4a16GroupTensors(reader, meta.w4a16.groups, path);
  if (meta.w4a16.check_default_per_linear && o.tp_rank == 0) LogW4a16Groups(meta.w4a16.groups, path);
  CheckTrellisTensors(reader, meta.trellis, path);

  const int64_t layers = o.layer_limit >= 0 ? std::min(o.layer_limit, cfg.num_hidden_layers) : cfg.num_hidden_layers;
  const int64_t hidden = cfg.hidden_size;
  const std::string rot = c.info_.rotation ? ".rotated" : "";

  // TP: one ShardLoader per rank, classifying every tensor by tp::RuleFor on the global ModelConfig view
  // (arch kGemma4 -> the Gemma table). At TP=1 the original single-device helpers run, unchanged.
  std::optional<container_util::ShardLoader> shard;
  if (tp) shard.emplace(reader, c.info_.model_config, o.tp_world, o.tp_rank, meta);
  int fallbacks = 0;
  const auto linear = [&](const std::string& base, Layout lay, int64_t N, int64_t K) {
    return shard ? shard->Linear(base, lay, N, K, &fallbacks)
                 : LoadQuantLinearWithFallback(reader, meta, base, lay, N, K, &fallbacks);
  };
  const auto raw_u16 = [&](const std::string& name) {
    return shard ? shard->Raw<uint16_t>(name) : UploadRawU16(reader, name);
  };
  const auto raw_f32 = [&](const std::string& name) {
    return shard ? shard->Raw<float>(name) : UploadRawF32(reader, name);
  };

  // The embedding table is replicated (the scaled gather runs on every rank); the tied head is the rank's own slice.
  c.embed_tokens_ = UploadRawU16(reader, "text.embed_tokens");
  if (shard) shard->CountReplicated(c.embed_tokens_.bytes());
  c.layers_.reserve(static_cast<size_t>(layers));
  for (int64_t i = 0; i < layers; ++i) {
    const std::string b = "text.layers." + std::to_string(i) + ".";
    GemmaLayerWeights lw;
    lw.full = cfg.IsFullLayer(i);
    lw.input_layernorm = raw_u16(b + "input_layernorm" + rot);
    lw.post_attention_layernorm = raw_u16(b + "post_attention_layernorm");
    lw.pre_feedforward_layernorm = raw_u16(b + "pre_feedforward_layernorm" + rot);
    lw.post_feedforward_layernorm = raw_u16(b + "post_feedforward_layernorm");
    GemmaAttnWeights& a = lw.attn;
    a.q = linear(b + "attn.q", o.layout, cfg.QDim(i), hidden);
    a.k = linear(b + "attn.k", o.layout, cfg.KvDim(i), hidden);
    if (!lw.full) {
      a.v = linear(b + "attn.v", o.layout, cfg.KvDim(i), hidden);
    }
    a.o = linear(b + "attn.o", o.layout, hidden, cfg.QDim(i));
    a.q_norm = raw_u16(b + "attn.q_norm");
    a.k_norm = raw_u16(b + "attn.k_norm");
    a.k_descale = raw_f32(b + "attn.k_descale");
    a.v_descale = raw_f32(b + "attn.v_descale");
    lw.mlp.gate_up = linear(b + "mlp.gate_up", o.layout, 2 * cfg.intermediate_size, hidden);
    lw.mlp.down = linear(b + "mlp.down", o.layout, hidden, cfg.intermediate_size);
    // layer_scalar: fp32 [1] on disk (a bf16 value widened); the fused post-norm kernel takes it by value.
    if (ElemCountBySize(reader, b + "layer_scalar", 4) != 1) {
      throw std::runtime_error("r4dx::model::GemmaContainer: " + path + ": " + b + "layer_scalar is not one element");
    }
    std::memcpy(&lw.layer_scalar, reader.Data(b + "layer_scalar"), sizeof(float));
    c.layers_.push_back(std::move(lw));
  }
  c.final_norm_ = raw_u16("text.final_norm");
  const int fb_before_head = fallbacks;
  c.lm_head_ = linear("lm_head", o.lm_head_layout, cfg.vocab_size, hidden);
  const int64_t head_rows = tp ? c.local_config_.VocabShardSize() : cfg.vocab_size;
  if (c.lm_head_.N != head_rows) {
    throw std::logic_error("r4dx::model::GemmaContainer: lm_head has " + std::to_string(c.lm_head_.N) +
                           " rows on this rank, expected " + std::to_string(head_rows));
  }
  const bool trellis_bf16_head = TakeTrellisBf16Head(meta, c.lm_head_, fb_before_head, &fallbacks);
  if (c.info_.rotation) {
    // Rotated container: signs / mix replicate; the q2ab Hadamard sign vectors are cut by RuleFor with the K range of
    // their linear's columns (had_o_signs 8 heads x 256, had_o_full_signs 8 heads x 512, had_down_signs 7680).
    const int64_t o_full_local = c.local_config_.num_attention_heads * c.local_config_.head_dim_full;
    c.rotation_ = LoadRotationWeights(reader, *c.info_.rotation, c.info_.model_config,
                                      c.local_config_.ToModelConfig(), path, raw_f32, c.info_.full_attn_out_elems,
                                      o_full_local);
    if (o.tp_rank == 0) LogRotation(*c.info_.rotation, path);
  }
  if (o.vision != GemmaVisionLoad::kOff) {
    if (GemmaContainerHasVisionTensors(reader)) {
      c.vision_.emplace(LoadGemmaVisionWeights(reader, hidden, path));
    } else if (o.vision == GemmaVisionLoad::kOn) {
      throw std::runtime_error("r4dx::model::GemmaContainer: " + path +
                               ": vision requested but the container has no vision.* tensors (convert with "
                               "r4dx-convert --vision on)");
    }
  }
  if (fallbacks > 0) {
    std::fprintf(stderr,
                 "r4dx: %d linear(s) in %s do not carry the requested layout and were loaded as bf16\n",
                 fallbacks, path.c_str());
  }
  if (meta.trellis) {
    c.trellis_ = std::move(meta.trellis);
    c.AssignTrellisTickets();
    if (o.tp_rank == 0) {
      LogTrellis(*c.trellis_, path);
      if (trellis_bf16_head) LogTrellisBf16Head(path);
    }
  }
  if (shard) {
    const container_util::ShardLoadStats& st = shard->Stats();
    std::fprintf(stderr, "[r4dx::model::GemmaContainer] rank %d/%d: %d tensors sharded, %d replicated, %.2f GiB uploaded, "
                         "%.2f GiB gathered through staging\n",
                 o.tp_rank, o.tp_world, st.sharded, st.replicated,
                 static_cast<double>(st.uploaded_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(st.staged_bytes) / (1024.0 * 1024.0 * 1024.0));
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
