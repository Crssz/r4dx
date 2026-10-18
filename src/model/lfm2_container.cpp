#include "lfm2_container.h"

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

Lfm2Container Lfm2Container::Load(const std::string& path, const Lfm2LoadOptions& o_in) {
  Lfm2LoadOptions o = o_in;
  if (o.lm_head_layout == Layout::kTrellis) o.lm_head_layout = Layout::kW4a16;  // heads are never trellis
  if (o.tp_world < 1 || o.tp_world > 2 || o.tp_rank < 0 || o.tp_rank >= o.tp_world) {
    throw std::invalid_argument("r4dx::model::Lfm2Container::Load: need tp_world in {1, 2} and 0 <= tp_rank < "
                                "tp_world, got tp_world " + std::to_string(o.tp_world) + ", tp_rank " +
                                std::to_string(o.tp_rank));
  }
  const bool tp = o.tp_world > 1;
  Lfm2Container c;
  c.info_ = Inspect(path, o.layer_limit);
  const Lfm2Config& cfg = c.info_.config;  // GLOBAL: every ShardLoader / loader call takes the global [N, K]
  // The rank's config (docs/lfm2-plan.md 3.6): throws, naming the field, unless every per-rank shape is legal.
  c.local_config_ = tp ? Lfm2Config::Shard(cfg, o.tp_world, o.tp_rank) : cfg;
  const nlohmann::json& metadata = c.info_.metadata;

  LinearLoadMeta meta;
  meta.w4a16 = CheckQuantGroups(metadata, path, o.layout, o.lm_head_layout, Layout::kBf16);
  meta.path = path;
  meta.trellis = ParseTrellisMetadata(metadata, path);
  CheckTrellisChoice(meta.trellis, /*rotated=*/false, o.layout, path);
  if (tp && meta.trellis) {  // docs/lfm2-plan.md L3-8 (RefuseLfm2TrellisTp says the same before any device work)
    throw std::invalid_argument("r4dx::model::Lfm2Container::Load: " + path +
                                " is a trellis container; LFM2 trellis runs at TP=1 only");
  }
  // docs/w8a16.md 6.1 / 6.3: `--layout w8a16` needs the block (rule 1: a bf16 container never loads on one card
  // this way), a format this build's kernel reads (rule 2), and TP=1 (v1). A no-op for any other layout.
  meta.w8a16 = CheckW8a16Choice(metadata, path, o.layout, o.lm_head_layout, Layout::kBf16);  // LFM2 shards w8a16 (tp_world is not a refusal here)

  SafetensorsReader reader(Utf8ToWide(path));
  CheckW4a16GroupTensors(reader, meta.w4a16.groups, path);
  if (meta.w4a16.check_default_per_linear && o.tp_rank == 0) LogW4a16Groups(meta.w4a16.groups, path);
  CheckTrellisTensors(reader, meta.trellis, path);

  const int64_t layers = o.layer_limit >= 0 ? std::min(o.layer_limit, cfg.num_hidden_layers) : cfg.num_hidden_layers;
  const int64_t hidden = cfg.hidden_size;

  // TP: one ShardLoader per rank, classifying every tensor by tp::RuleFor on the global ModelConfig view (arch
  // kLfm2Moe -> the LFM table). At TP=1 the original single-device helpers run, unchanged.
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
  // uint64 base pointers of a vector of QuantLinears (the L4 grouped kernels' expert tables).
  const auto ptr_table_bf16 = [](const std::vector<QuantLinear>& v) {
    std::vector<uint64_t> host(v.size());
    for (size_t e = 0; e < v.size(); ++e) host[e] = reinterpret_cast<uint64_t>(v[e].bf16_w.data());
    core::DeviceBuffer<uint64_t> dev(host.size());
    dev.CopyFromHost(host.data(), host.size());
    return dev;
  };
  const auto ptr_table_w8_wq = [](const std::vector<QuantLinear>& v) {
    std::vector<uint64_t> host(v.size());
    for (size_t e = 0; e < v.size(); ++e) host[e] = reinterpret_cast<uint64_t>(v[e].w8_wq.data());
    core::DeviceBuffer<uint64_t> dev(host.size());
    dev.CopyFromHost(host.data(), host.size());
    return dev;
  };
  const auto ptr_table_w8_ws = [](const std::vector<QuantLinear>& v) {
    std::vector<uint64_t> host(v.size());
    for (size_t e = 0; e < v.size(); ++e) host[e] = reinterpret_cast<uint64_t>(v[e].w8_ws.data());
    core::DeviceBuffer<uint64_t> dev(host.size());
    dev.CopyFromHost(host.data(), host.size());
    return dev;
  };
  const auto ptr_table_trellis_w = [](const std::vector<QuantLinear>& v) {
    std::vector<uint64_t> host(v.size());
    for (size_t e = 0; e < v.size(); ++e) host[e] = reinterpret_cast<uint64_t>(v[e].trellis_w.data());
    core::DeviceBuffer<uint64_t> dev(host.size());
    dev.CopyFromHost(host.data(), host.size());
    return dev;
  };
  const auto ptr_table_trellis_suh = [](const std::vector<QuantLinear>& v) {
    std::vector<uint64_t> host(v.size());
    for (size_t e = 0; e < v.size(); ++e) host[e] = reinterpret_cast<uint64_t>(v[e].trellis_suh.data());
    core::DeviceBuffer<uint64_t> dev(host.size());
    dev.CopyFromHost(host.data(), host.size());
    return dev;
  };
  const auto ptr_table_trellis_svh = [](const std::vector<QuantLinear>& v) {
    std::vector<uint64_t> host(v.size());
    for (size_t e = 0; e < v.size(); ++e) host[e] = reinterpret_cast<uint64_t>(v[e].trellis_svh.data());
    core::DeviceBuffer<uint64_t> dev(host.size());
    dev.CopyFromHost(host.data(), host.size());
    return dev;
  };

  // The embedding table is replicated (the gather runs on every rank); the tied head is the rank's own slice.
  c.embed_tokens_ = UploadRawU16(reader, "text.embed_tokens");
  if (shard) shard->CountReplicated(c.embed_tokens_.bytes());
  c.layers_.reserve(static_cast<size_t>(layers));
  for (int64_t i = 0; i < layers; ++i) {
    const std::string b = Lfm2LayerBase(i);
    Lfm2LayerWeights lw;
    lw.attn_layer = cfg.IsAttentionLayer(i);
    lw.dense = cfg.IsDenseLayer(i);
    lw.operator_norm = raw_u16(b + "operator_norm");
    lw.ffn_norm = raw_u16(b + "ffn_norm");
    if (lw.attn_layer) {
      Lfm2AttnWeights& a = lw.attn;
      a.q = linear(b + "attn.q", o.layout, cfg.QDim(), hidden);
      a.k = linear(b + "attn.k", o.layout, cfg.KvDim(), hidden);
      a.v = linear(b + "attn.v", o.layout, cfg.KvDim(), hidden);
      a.o = linear(b + "attn.o", o.layout, hidden, cfg.QDim());
      a.q_norm = raw_u16(b + "attn.q_norm");
      a.k_norm = raw_u16(b + "attn.k_norm");
    } else {
      Lfm2ConvWeights& cv = lw.conv;
      cv.in_proj = linear(b + "conv.in_proj", o.layout, cfg.ConvInProjRows(), hidden);
      cv.out_proj = linear(b + "conv.out_proj", o.layout, hidden, cfg.conv_dim);
      cv.weight = raw_u16(b + "conv.weight");
      if (static_cast<int64_t>(cv.weight.size()) != c.local_config_.conv_dim * cfg.conv_l_cache) {
        throw std::logic_error("r4dx::model::Lfm2Container: " + b + "conv.weight has " + std::to_string(cv.weight.size()) +
                               " elements on this rank, expected " +
                               std::to_string(c.local_config_.conv_dim * cfg.conv_l_cache));
      }
    }
    if (lw.dense) {
      lw.mlp.gate_up = linear(b + "mlp.gate_up", o.layout, 2 * cfg.intermediate_size, hidden);
      lw.mlp.down = linear(b + "mlp.down", o.layout, hidden, cfg.intermediate_size);
    } else {
      Lfm2MoeWeights& m = lw.moe;
      m.router = raw_u16(b + "moe.router");
      m.expert_bias = raw_f32(b + "moe.expert_bias");
      m.gate_up.reserve(static_cast<size_t>(cfg.num_experts));
      m.down.reserve(static_cast<size_t>(cfg.num_experts));
      for (int64_t e = 0; e < cfg.num_experts; ++e) {
        m.gate_up.push_back(linear(Lfm2ExpertBase(i, e, "gate_up"), o.layout, cfg.ExpertGateUpRows(), hidden));
        m.down.push_back(linear(Lfm2ExpertBase(i, e, "down"), o.layout, hidden, cfg.moe_intermediate_size));
      }
      bool all_bf16 = !m.gate_up.empty();
      bool all_w8 = !m.gate_up.empty() && m.gate_up[0].w8a16_group > 0 && m.down[0].w8a16_group > 0;
      bool all_trellis = !m.gate_up.empty() &&
                         (m.gate_up[0].trellis_bits == 4 || m.gate_up[0].trellis_bits == 5) &&
                         (m.down[0].trellis_bits == 4 || m.down[0].trellis_bits == 5);
      for (size_t e = 0; e < m.gate_up.size(); ++e) {
        all_bf16 = all_bf16 && m.gate_up[e].layout == Layout::kBf16 && m.down[e].layout == Layout::kBf16;
        all_w8 = all_w8 && m.gate_up[e].layout == Layout::kW8a16 && m.down[e].layout == Layout::kW8a16 &&
                 m.gate_up[e].w8a16_group == m.gate_up[0].w8a16_group &&
                 m.down[e].w8a16_group == m.down[0].w8a16_group;
        all_trellis = all_trellis && m.gate_up[e].layout == Layout::kTrellis && m.down[e].layout == Layout::kTrellis &&
                      m.gate_up[e].trellis_bits == m.gate_up[0].trellis_bits &&
                      m.down[e].trellis_bits == m.down[0].trellis_bits &&
                      m.gate_up[e].trellis_prescale_log2 == m.gate_up[0].trellis_prescale_log2 &&
                      m.down[e].trellis_prescale_log2 == m.down[0].trellis_prescale_log2 &&
                      m.gate_up[e].trellis_parts == m.gate_up[0].trellis_parts &&
                      m.down[e].trellis_parts == m.down[0].trellis_parts &&
                      m.gate_up[e].trellis_part_n[0] == m.gate_up[0].trellis_part_n[0];
      }
      if (all_bf16) {
        m.gate_up_bf16_ptrs = ptr_table_bf16(m.gate_up);
        m.down_bf16_ptrs = ptr_table_bf16(m.down);
      } else if (all_w8) {
        m.gate_up_w8_wq_ptrs = ptr_table_w8_wq(m.gate_up);
        m.gate_up_w8_ws_ptrs = ptr_table_w8_ws(m.gate_up);
        m.down_w8_wq_ptrs = ptr_table_w8_wq(m.down);
        m.down_w8_ws_ptrs = ptr_table_w8_ws(m.down);
      } else if (all_trellis) {
        m.gate_up_trellis_w_ptrs = ptr_table_trellis_w(m.gate_up);
        m.gate_up_trellis_suh_ptrs = ptr_table_trellis_suh(m.gate_up);
        m.gate_up_trellis_svh_ptrs = ptr_table_trellis_svh(m.gate_up);
        m.down_trellis_w_ptrs = ptr_table_trellis_w(m.down);
        m.down_trellis_suh_ptrs = ptr_table_trellis_suh(m.down);
        m.down_trellis_svh_ptrs = ptr_table_trellis_svh(m.down);
      }
    }
    c.layers_.push_back(std::move(lw));
  }
  c.final_norm_ = raw_u16("text.final_norm");
  const int fb_before_head = fallbacks;
  c.lm_head_ = linear("lm_head", o.lm_head_layout, cfg.vocab_size, hidden);
  const int64_t head_rows = tp ? c.local_config_.VocabShardSize() : cfg.vocab_size;
  if (c.lm_head_.N != head_rows) {
    throw std::logic_error("r4dx::model::Lfm2Container: lm_head has " + std::to_string(c.lm_head_.N) +
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
    std::fprintf(stderr, "[r4dx::model::Lfm2Container] rank %d/%d: %d tensors sharded, %d replicated, %.2f GiB uploaded, "
                         "%.2f GiB gathered through staging\n",
                 o.tp_rank, o.tp_world, st.sharded, st.replicated,
                 static_cast<double>(st.uploaded_bytes) / (1024.0 * 1024.0 * 1024.0),
                 static_cast<double>(st.staged_bytes) / (1024.0 * 1024.0 * 1024.0));
  }
  return c;
}

template <class Fn>
void Lfm2Container::ForEachLinear(Fn&& fn) {
  for (Lfm2LayerWeights& lw : layers_) {
    if (lw.attn_layer) {
      fn(lw.attn.q);
      fn(lw.attn.k);
      fn(lw.attn.v);
      fn(lw.attn.o);
    } else {
      fn(lw.conv.in_proj);
      fn(lw.conv.out_proj);
    }
    if (lw.dense) {
      fn(lw.mlp.gate_up);
      fn(lw.mlp.down);
    } else {
      for (QuantLinear& q : lw.moe.gate_up) fn(q);
      for (QuantLinear& q : lw.moe.down) fn(q);
    }
  }
  fn(lm_head_);
}

void Lfm2Container::AssignTrellisTickets() {
  size_t total = 0;
  int64_t max_moe_n = 0;
  ForEachLinear([&](QuantLinear& q) {
    if (q.layout == Layout::kTrellis) {
      total += core::r4d::GemmTrellisTicketsBytes(static_cast<int>(q.N)) / sizeof(uint32_t);
      max_moe_n = std::max(max_moe_n, q.N);
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
  if (max_moe_n >= 128) {
    // Per-pair self-resetting tickets for r4dx_lfm2_gemm_trellis_moe_decode (up to 256 tokens * top_k = 1024 pairs).
    const size_t moe_tk = static_cast<size_t>(1024) * static_cast<size_t>(max_moe_n / 128);
    moe_trellis_tickets_ = core::DeviceBuffer<uint32_t>(moe_tk);
    moe_trellis_tickets_.Zero();
    for (Lfm2LayerWeights& lw : layers_) {
      if (!lw.dense && !lw.moe.gate_up_trellis_w_ptrs.empty()) {
        lw.moe.moe_trellis_tickets = moe_trellis_tickets_.data();
      }
    }
  }
}

void Lfm2Container::ZeroTrellisTickets(hipStream_t stream) {
  if (!trellis_tickets_.empty()) {
    core::r4d::GemmTrellisZeroTickets(trellis_tickets_.data(), trellis_tickets_.bytes(), stream);
  }
  if (!moe_trellis_tickets_.empty()) {
    core::r4d::GemmTrellisZeroTickets(moe_trellis_tickets_.data(), moe_trellis_tickets_.bytes(), stream);
  }
}

}  // namespace r4dx::model
