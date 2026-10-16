// r4dx::model::Lfm2Container -- loads one lfm2_moe .r4dx container (docs/lfm2-plan.md 4, task L1-3) into device memory.
// Like GemmaContainer it reuses container_load_util (the per-linear layout fallback chain, the raw uploads, the
// trellis / w4a16 checks) and shard_loader.h (TP=2: every tensor cut by tp::RuleFor's LFM table). The CPU half --
// `__metadata__`, the tensor directory and every name / shape check -- is lfm2_container_info.h (InspectLfm2Container):
// it runs without a device, and Load() calls it first.
//
// What it loads (TP=1, or with Lfm2LoadOptions::tp_world 2 one rank's shard):
//   text.embed_tokens        bf16 [vocab, hidden], DEVICE-resident always, replicated (the gather and the tied head)
//   text.final_norm          bf16 [hidden] (plain weight)
//   lm_head                  QuantLinear [vocab, hidden] (the tied head, written untied; the rank's vocab slice under TP)
//   per layer: operator_norm / ffn_norm (bf16 [hidden], replicated), then
//     attention layers: attn.{q,k,v,o} + attn.{q,k}_norm (bf16 [head_dim], replicated)
//     ShortConv layers: conv.in_proj (three-segment column split B|C|x), conv.out_proj (row-parallel),
//                       conv.weight bf16 [conv_dim, 3] (the rank's channel rows)
//     dense layers (< num_dense_layers): mlp.gate_up / mlp.down
//     MoE layers: moe.router bf16 [E, hidden] + moe.expert_bias fp32 [E] (both replicated), and PER-EXPERT
//                 moe.experts.{e}.gate_up / .down QuantLinears (every expert TP-split). Per layer the expert table:
//                 `gate_up[e]` / `down[e]` plus, in the bf16 layout, a device array of the weight base pointers
//                 (uint64 [E] each: what the L4 grouped kernels index by the router's expert id).
//
// Trellis (docs/lfm2-plan.md L3-8, `--layout trellis`, TP=1 only -- RefuseLfm2TrellisTp): every body linear above,
// experts included, loads its `.trellis.{w,suh,svh}` through the same fallback chain (a linear the container left bf16
// loads bf16 and is counted), and gets its own ticket slice (AssignTrellisTickets). lm_head / embedding / router /
// expert_bias / conv taps / norms are never trellis. The experts' QuantLinears are called one by one through
// ApplyLinear (Lfm2MoeFfn), so no pointer table exists for a trellis body.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "container.h"  // MlpWeights
#include "lfm2_config.h"
#include "lfm2_container_info.h"
#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"
#include "trellis_meta.h"

namespace r4dx::model {

struct Lfm2AttnWeights {
  QuantLinear q;  // [heads * head_dim, hidden]
  QuantLinear k;  // [kv_heads * head_dim, hidden]
  QuantLinear v;  // [kv_heads * head_dim, hidden]
  QuantLinear o;  // [hidden, heads * head_dim]
  core::DeviceBuffer<uint16_t> q_norm, k_norm;  // bf16 [head_dim] (plain weight)
};

struct Lfm2ConvWeights {
  QuantLinear in_proj;                  // [3 * conv_dim, hidden]: segments B, C, x (the rank's channels of each under TP)
  QuantLinear out_proj;                 // [hidden, conv_dim]
  core::DeviceBuffer<uint16_t> weight;  // bf16 [conv_dim, 3]: the depthwise taps, oldest tap first
};

struct Lfm2MoeWeights {
  core::DeviceBuffer<uint16_t> router;     // bf16 [num_experts, hidden], replicated
  core::DeviceBuffer<float> expert_bias;   // fp32 [num_experts], replicated
  std::vector<QuantLinear> gate_up;        // [e]: [2 * moe_intermediate, hidden]  (gate rows first; the rank's halves under TP)
  std::vector<QuantLinear> down;           // [e]: [hidden, moe_intermediate]       (the rank's K slice under TP)
  // layout kBf16 only (empty otherwise; the trellis / w4a16 table is an L4 decision): the experts' bf16 weight base
  // pointers as uint64, indexed by expert id.
  core::DeviceBuffer<uint64_t> gate_up_bf16_ptrs, down_bf16_ptrs;
};

struct Lfm2LayerWeights {
  bool attn_layer = false;
  bool dense = false;  // dense FFN (mlp) vs MoE (moe)
  core::DeviceBuffer<uint16_t> operator_norm, ffn_norm;  // bf16 [hidden]
  Lfm2AttnWeights attn;                                  // attention layers
  Lfm2ConvWeights conv;                                  // ShortConv layers
  MlpWeights mlp;                                        // dense layers
  Lfm2MoeWeights moe;                                    // MoE layers
};

struct Lfm2LoadOptions {
  Layout layout = Layout::kBf16;          // body linears (trellis needs a trellis container)
  Layout lm_head_layout = Layout::kBf16;  // a trellis body's head loads as w4a16 / bf16
  int64_t layer_limit = -1;               // >= 0: load layers [0, layer_limit) (tiny fixtures, lfm2-l6)
  int tp_world = 1;                       // 1 or 2 (docs/lfm2-plan.md 3.6)
  int tp_rank = 0;
};

class Lfm2Container {
 public:
  // CPU only (no HIP call, no weight byte read): `__metadata__` + the tensor directory, checked.
  static Lfm2ContainerInfo Inspect(const std::string& path, int64_t layer_limit = -1) {
    return InspectLfm2Container(path, layer_limit);
  }

  // Inspect(), then the upload onto the current HIP device.
  static Lfm2Container Load(const std::string& path, const Lfm2LoadOptions& o = Lfm2LoadOptions());

  Lfm2Container(Lfm2Container&&) = default;
  Lfm2Container& operator=(Lfm2Container&&) = default;

  const Lfm2ContainerInfo& Info() const { return info_; }
  // The config the loaded weights have: Lfm2Config::Shard(global, tp_world, tp_rank) under TP (vocab_size stays global;
  // VocabShardSize() is the head's rows), else the global one. Info().config is always the GLOBAL config.
  const Lfm2Config& Config() const { return local_config_; }
  const Lfm2Config& GlobalConfig() const { return info_.config; }
  int TpWorld() const { return local_config_.tp_world; }
  int TpRank() const { return local_config_.tp_rank; }
  const ModelConfig& GenericConfig() const { return info_.model_config; }  // thin view for TextModel::Config()
  const std::string& ModelId() const { return info_.model_id; }

  int64_t NumLoadedLayers() const { return static_cast<int64_t>(layers_.size()); }
  const Lfm2LayerWeights& Layer(int64_t i) const { return layers_.at(static_cast<size_t>(i)); }
  const core::DeviceBuffer<uint16_t>& EmbedTokensDevice() const { return embed_tokens_; }
  const core::DeviceBuffer<uint16_t>& FinalNorm() const { return final_norm_; }
  const QuantLinear& LmHead() const { return lm_head_; }
  bool HasTrellis() const { return trellis_.has_value(); }
  void ZeroTrellisTickets(hipStream_t stream);

 private:
  Lfm2Container() = default;
  friend class Lfm2Model;  // default-constructs it, then assigns Lfm2Container::Load's result (as GemmaModel)
  template <class Fn>
  void ForEachLinear(Fn&& fn);
  void AssignTrellisTickets();

  Lfm2ContainerInfo info_;
  Lfm2Config local_config_;  // == info_.config at TP=1
  core::DeviceBuffer<uint16_t> embed_tokens_, final_norm_;
  QuantLinear lm_head_;
  std::vector<Lfm2LayerWeights> layers_;
  std::optional<TrellisSpec> trellis_;
  core::DeviceBuffer<uint32_t> trellis_tickets_;
};

// docs/lfm2-plan.md L3-8: the trellis body (the shipping format) runs at TP=1 only -- the TP=2 shard rules of the
// per-expert trellis linears (two-part gate_up halves, the pair-grid row split) are not built or validated for LFM2.
// Throws std::invalid_argument naming `who` when `layout` is trellis or `path`'s header carries
// __metadata__.quant.trellis (a CPU header read, no HIP call). Called before any device work.
inline void RefuseLfm2TrellisTp(const std::string& path, Layout layout, int64_t layer_limit, const std::string& who) {
  const bool trellis = layout == Layout::kTrellis || Lfm2Container::Inspect(path, layer_limit).has_trellis;
  if (!trellis) return;
  throw std::invalid_argument(who + ": " + path + " -- an LFM2 trellis container runs at TP=1 only (--tp 1 --layout "
                              "trellis); tensor parallelism (--tp 2) is supported for the bf16 LFM2 container only");
}

// docs/w8a16.md 6.2 (v1): `--layout w8a16` runs at TP=1 only (the w8a16 shard parts are not built). Throws
// std::invalid_argument naming `who`; no file read, no HIP call. Called before any device work, next to
// RefuseLfm2TrellisTp (Lfm2Container::Load refuses the same through CheckW8a16Choice).
inline void RefuseLfm2W8a16Tp(Layout layout, const std::string& who) {
  if (layout != Layout::kW8a16) return;
  throw std::invalid_argument(who + ": --layout w8a16 runs at TP=1 only (--tp 1 --layout w8a16); tensor parallelism "
                              "(--tp 2) of the w8a16 layout is not part of v1 (docs/w8a16.md 6.2)");
}

}  // namespace r4dx::model
