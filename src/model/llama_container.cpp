#include "llama_container.h"

#include <algorithm>
#include <cstdio>
#include <optional>
#include <stdexcept>

#include "container_load_util.h"
#include "shard_loader.h"
#include "r4dx/core/r4d.hpp"  // GemmTrellisTicketsBytes / GemmTrellisZeroTickets

namespace r4dx::model {

using namespace container_util;

LlamaContainer LlamaContainer::Load(const std::string& path, const LlamaLoadOptions& o_in) {
  const core::DirectLoadScope direct_load;  // docs/direct-storage.md
  LlamaLoadOptions o = o_in;
  if (o.lm_head_layout == Layout::kTrellis) o.lm_head_layout = Layout::kW4a16;  // heads are never trellis
  if (o.tp_world < 1 || o.tp_world > 2 || o.tp_rank < 0 || o.tp_rank >= o.tp_world) {
    throw std::invalid_argument("r4dx::model::LlamaContainer::Load: need tp_world in {1, 2} and 0 <= tp_rank < "
                                "tp_world, got tp_world " + std::to_string(o.tp_world) + ", tp_rank " +
                                std::to_string(o.tp_rank));
  }
  const bool tp = o.tp_world > 1;
  LlamaContainer c;
  c.info_ = Inspect(path, o.layer_limit);
  const LlamaConfig& cfg = c.info_.config;  // GLOBAL: every ShardLoader / loader call takes the global [N, K]
  // The rank's config (docs/nanbeige-plan.md 3.4): throws, naming the field, unless every per-rank shape is legal.
  c.local_config_ = tp ? LlamaConfig::Shard(cfg, o.tp_world, o.tp_rank) : cfg;
  const nlohmann::json& metadata = c.info_.metadata;

  LinearLoadMeta meta;
  meta.w4a16 = CheckQuantGroups(metadata, path, o.layout, o.lm_head_layout, Layout::kBf16);
  meta.path = path;
  meta.trellis = ParseTrellisMetadata(metadata, path);
  CheckTrellisChoice(meta.trellis, /*rotated=*/false, o.layout, path);
  if (tp && meta.trellis) {  // RefuseLlamaTrellisTp says the same before any device work
    throw std::invalid_argument("r4dx::model::LlamaContainer::Load: " + path +
                                " is a trellis container; the Llama trellis runs at TP=1 only");
  }
  meta.w8a16 = CheckW8a16Choice(metadata, path, o.layout, o.lm_head_layout, Layout::kBf16);

  SafetensorsReader reader(Utf8ToWide(path));
  CheckW4a16GroupTensors(reader, meta.w4a16.groups, path);
  if (meta.w4a16.check_default_per_linear && o.tp_rank == 0) LogW4a16Groups(meta.w4a16.groups, path);
  CheckTrellisTensors(reader, meta.trellis, path);

  const int64_t layers = o.layer_limit >= 0 ? std::min(o.layer_limit, cfg.num_hidden_layers) : cfg.num_hidden_layers;
  const int64_t hidden = cfg.hidden_size;

  // TP: one ShardLoader per rank, classifying every tensor by tp::RuleFor on the global ModelConfig view (arch kLlama
  // -> the Llama table). At TP=1 the single-device helpers run.
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

  // The embedding table is replicated (the gather runs on every rank).
  c.embed_tokens_ = UploadRawU16(reader, "text.embed_tokens");
  if (shard) shard->CountReplicated(c.embed_tokens_.bytes());
  c.layers_.reserve(static_cast<size_t>(layers));
  for (int64_t i = 0; i < layers; ++i) {
    const std::string b = LlamaLayerBase(i);
    LlamaLayerWeights lw;
    lw.input_norm = raw_u16(b + "input_norm");
    lw.post_attn_norm = raw_u16(b + "post_attn_norm");
    lw.attn.q = linear(b + "attn.q", o.layout, cfg.QDim(), hidden);
    lw.attn.k = linear(b + "attn.k", o.layout, cfg.KvDim(), hidden);
    lw.attn.v = linear(b + "attn.v", o.layout, cfg.KvDim(), hidden);
    lw.attn.o = linear(b + "attn.o", o.layout, hidden, cfg.QDim());
    lw.mlp.gate_up = linear(b + "mlp.gate_up", o.layout, cfg.GateUpRows(), hidden);
    lw.mlp.down = linear(b + "mlp.down", o.layout, hidden, cfg.intermediate_size);
    c.layers_.push_back(std::move(lw));
  }
  c.final_norm_ = raw_u16("text.final_norm");
  const int fb_before_head = fallbacks;
  c.lm_head_ = linear("lm_head", o.lm_head_layout, cfg.vocab_size, hidden);
  const int64_t head_rows = tp ? c.local_config_.VocabShardSize() : cfg.vocab_size;
  if (c.lm_head_.N != head_rows) {
    throw std::logic_error("r4dx::model::LlamaContainer: lm_head has " + std::to_string(c.lm_head_.N) +
                           " rows on this rank, expected " + std::to_string(head_rows));
  }
  const bool trellis_bf16_head = TakeTrellisBf16Head(meta, c.lm_head_, fb_before_head, &fallbacks);
  const bool w8a16_bf16_head =
      o.lm_head_layout == Layout::kW8a16 && TakeW8a16Bf16Head(meta, c.lm_head_, fb_before_head, &fallbacks);
  if (fallbacks > 0) {
    std::fprintf(stderr, "r4dx: %d linear(s) in %s do not carry the requested layout and were loaded as bf16\n",
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
  if (meta.w8a16 && o.tp_rank == 0) {
    LogW8a16(*meta.w8a16, path);
    if (w8a16_bf16_head) LogW8a16Bf16Head(path);
  }
  if (shard) {
    const container_util::ShardLoadStats& st = shard->Stats();
    std::fprintf(stderr, "[r4dx::model::LlamaContainer] rank %d/%d: %d tensors sharded, %d replicated, %.2f GiB uploaded, "
                         "%.2f GiB gathered through staging\n",
                 o.tp_rank, o.tp_world, st.sharded, st.replicated,
                 static_cast<double>(st.uploaded_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(st.staged_bytes) / (1024.0 * 1024.0 * 1024.0));
  }
  return c;
}

template <class Fn>
void LlamaContainer::ForEachLinear(Fn&& fn) {
  for (LlamaLayerWeights& lw : layers_) {
    fn(lw.attn.q);
    fn(lw.attn.k);
    fn(lw.attn.v);
    fn(lw.attn.o);
    fn(lw.mlp.gate_up);
    fn(lw.mlp.down);
  }
  fn(lm_head_);
}

void LlamaContainer::AssignTrellisTickets() {
  size_t total = 0;
  ForEachLinear([&](QuantLinear& q) {
    if (q.layout == Layout::kTrellis) total += core::r4d::GemmTrellisTicketsBytes(static_cast<int>(q.N)) / sizeof(uint32_t);
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

void LlamaContainer::ZeroTrellisTickets(hipStream_t stream) {
  if (!trellis_tickets_.empty()) {
    core::r4d::GemmTrellisZeroTickets(trellis_tickets_.data(), trellis_tickets_.bytes(), stream);
  }
}

}  // namespace r4dx::model
