#include "hy_v3_container.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "container_load_util.h"

namespace r4dx::model {

using namespace container_util;

HyV3Container HyV3Container::Load(const std::string& path, const HyV3LoadOptions& o) {
  HyV3Container c;
  c.info_ = Inspect(path, o.layer_limit);
  const HyV3Config& cfg = c.info_.config;
  const nlohmann::json& metadata = c.info_.metadata;

  LinearLoadMeta meta;
  meta.path = path;
  meta.w4a16 = CheckQuantGroups(metadata, path, o.layout, o.lm_head_layout, Layout::kBf16);
  meta.w8a16 = CheckW8a16Choice(metadata, path, o.layout, o.lm_head_layout, Layout::kBf16);

  SafetensorsReader reader(Utf8ToWide(path));
  CheckW4a16GroupTensors(reader, meta.w4a16.groups, path);

  const int64_t layers = o.layer_limit >= 0 ? std::min(o.layer_limit, cfg.num_hidden_layers) : cfg.num_hidden_layers;
  const int64_t H = cfg.hidden_size;
  int fallbacks = 0;

  auto linear = [&](const std::string& base, Layout lay, int64_t N, int64_t K) {
    return LoadQuantLinearWithFallback(reader, meta, base, lay, N, K, &fallbacks);
  };

  c.embed_tokens_ = UploadRawU16(reader, "text.embed_tokens");
  c.layers_.reserve(static_cast<size_t>(layers));
  for (int64_t i = 0; i < layers; ++i) {
    const std::string b = HyV3LayerBase(i);
    HyV3LayerWeights lw;
    lw.dense = cfg.IsDenseLayer(i);
    lw.input_norm = UploadRawU16(reader, b + "input_norm");
    lw.post_attn_norm = UploadRawU16(reader, b + "post_attn_norm");
    lw.attn.q = linear(b + "attn.q", o.layout, cfg.QDim(), H);
    lw.attn.k = linear(b + "attn.k", o.layout, cfg.KvDim(), H);
    lw.attn.v = linear(b + "attn.v", o.layout, cfg.KvDim(), H);
    lw.attn.o = linear(b + "attn.o", o.layout, H, cfg.QDim());
    lw.attn.q_norm = UploadRawU16(reader, b + "attn.q_norm");
    lw.attn.k_norm = UploadRawU16(reader, b + "attn.k_norm");

    if (lw.dense) {
      lw.mlp.gate_up = linear(b + "mlp.gate_up", o.layout, cfg.DenseGateUpRows(), H);
      lw.mlp.down = linear(b + "mlp.down", o.layout, H, cfg.intermediate_size);
    } else {
      lw.moe.router = UploadRawU16(reader, b + "moe.router");
      lw.moe.expert_bias = UploadRawF32(reader, b + "moe.expert_bias");
      lw.shared_mlp.gate_up = linear(b + "moe.shared_gate_up", o.layout, cfg.ExpertGateUpRows(), H);
      lw.shared_mlp.down = linear(b + "moe.shared_down", o.layout, H, cfg.moe_intermediate_size);
      const int64_t E = cfg.num_experts;
      lw.moe.gate_up.reserve(static_cast<size_t>(E));
      lw.moe.down.reserve(static_cast<size_t>(E));
      for (int64_t e = 0; e < E; ++e) {
        const std::string eb = b + "moe.experts." + std::to_string(e) + ".";
        lw.moe.gate_up.push_back(linear(eb + "gate_up", o.layout, cfg.ExpertGateUpRows(), H));
        lw.moe.down.push_back(linear(eb + "down", o.layout, H, cfg.moe_intermediate_size));
      }
      // Build device pointer table [0..E] (routed experts + shared_mlp at index E) for grouped w4a16 MoE.
      bool all_w4 = lw.shared_mlp.gate_up.layout == Layout::kW4a16 && lw.shared_mlp.down.layout == Layout::kW4a16;
      const int g0 = lw.shared_mlp.gate_up.w4a16_group > 0 ? lw.shared_mlp.gate_up.w4a16_group : r4d_gemm_w4a16_nt_m64_group();
      if (all_w4) {
        for (int64_t e = 0; e < E; ++e) {
          const auto& gu = lw.moe.gate_up[static_cast<size_t>(e)];
          const auto& dn = lw.moe.down[static_cast<size_t>(e)];
          const int g_gu = gu.w4a16_group > 0 ? gu.w4a16_group : r4d_gemm_w4a16_nt_m64_group();
          const int g_dn = dn.w4a16_group > 0 ? dn.w4a16_group : r4d_gemm_w4a16_nt_m64_group();
          if (gu.layout != Layout::kW4a16 || dn.layout != Layout::kW4a16 || g_gu != g0 || g_dn != g0) {
            all_w4 = false;
            break;
          }
        }
      }
      if (all_w4) {
        std::vector<uint64_t> gu_wq(static_cast<size_t>(E + 1)), gu_wsz(static_cast<size_t>(E + 1));
        std::vector<uint64_t> dn_wq(static_cast<size_t>(E + 1)), dn_wsz(static_cast<size_t>(E + 1));
        for (int64_t e = 0; e < E; ++e) {
          const size_t ue = static_cast<size_t>(e);
          gu_wq[ue] = reinterpret_cast<uint64_t>(lw.moe.gate_up[ue].wq.data());
          gu_wsz[ue] = reinterpret_cast<uint64_t>(lw.moe.gate_up[ue].w4a16_wsz.data());
          dn_wq[ue] = reinterpret_cast<uint64_t>(lw.moe.down[ue].wq.data());
          dn_wsz[ue] = reinterpret_cast<uint64_t>(lw.moe.down[ue].w4a16_wsz.data());
        }
        gu_wq[static_cast<size_t>(E)] = reinterpret_cast<uint64_t>(lw.shared_mlp.gate_up.wq.data());
        gu_wsz[static_cast<size_t>(E)] = reinterpret_cast<uint64_t>(lw.shared_mlp.gate_up.w4a16_wsz.data());
        dn_wq[static_cast<size_t>(E)] = reinterpret_cast<uint64_t>(lw.shared_mlp.down.wq.data());
        dn_wsz[static_cast<size_t>(E)] = reinterpret_cast<uint64_t>(lw.shared_mlp.down.w4a16_wsz.data());
        lw.gate_up_w4_wq_ptrs = core::DeviceBuffer<uint64_t>(gu_wq.size());
        lw.gate_up_w4_wsz_ptrs = core::DeviceBuffer<uint64_t>(gu_wsz.size());
        lw.down_w4_wq_ptrs = core::DeviceBuffer<uint64_t>(dn_wq.size());
        lw.down_w4_wsz_ptrs = core::DeviceBuffer<uint64_t>(dn_wsz.size());
        lw.gate_up_w4_wq_ptrs.CopyFromHost(gu_wq.data(), gu_wq.size());
        lw.gate_up_w4_wsz_ptrs.CopyFromHost(gu_wsz.data(), gu_wsz.size());
        lw.down_w4_wq_ptrs.CopyFromHost(dn_wq.data(), dn_wq.size());
        lw.down_w4_wsz_ptrs.CopyFromHost(dn_wsz.data(), dn_wsz.size());
        lw.moe_w4_group = g0;
      }
    }
    c.layers_.push_back(std::move(lw));
  }

  c.final_norm_ = UploadRawU16(reader, "text.final_norm");
  Layout head_layout = o.lm_head_layout;
  if (reader.Has("lm_head.w8a16.wq")) {
    head_layout = Layout::kW8a16;
    if (!meta.w8a16.has_value()) {
      const int64_t ws_rows = reader.Meta("lm_head.w8a16.ws").shape.at(0);
      const int group = (ws_rows == (cfg.vocab_size * H / 128)) ? 128 : 64;
      const int bias = r4dx_gemm_w8a16_scale_log2_bias();
      CheckW8a16Meta(group, bias, path);
      W8a16Spec spec;
      spec.group = group;
      spec.scale_log2_bias = bias;
      meta.w8a16 = spec;
    }
  }
  c.lm_head_ = linear("lm_head", head_layout, cfg.vocab_size, H);
  if (fallbacks > 0) {
    std::fprintf(stderr, "r4dx: %d linear(s) in %s fell back to bf16\n", fallbacks, path.c_str());
  }
  return c;
}

}  // namespace r4dx::model
