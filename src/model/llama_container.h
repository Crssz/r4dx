// r4dx::model::LlamaContainer -- loads one dense `llama` .r4dx container (docs/nanbeige-plan.md 4, task N1-5) into
// device memory. Like Lfm2Container it reuses container_load_util (the per-linear layout fallback chain, the raw
// uploads, the trellis / w4a16 / w8a16 checks) and shard_loader.h (TP=2: every tensor cut by tp::RuleFor's Llama
// table). The CPU half -- `__metadata__`, the tensor directory and every name / shape check -- is
// llama_container_info.h (InspectLlamaContainer): it runs without a device, and Load() calls it first.
//
// What it loads (TP=1, or with LlamaLoadOptions::tp_world 2 one rank's shard):
//   text.embed_tokens        bf16 [vocab, hidden], device-resident, replicated (the gather)
//   text.final_norm          bf16 [hidden] (plain weight)
//   lm_head                  QuantLinear [vocab, hidden] (untied; the rank's vocab slice under TP)
//   per layer: input_norm / post_attn_norm (bf16 [hidden], replicated), attn.{q,k,v} (column-parallel by head),
//              attn.o (row-parallel), mlp.gate_up (two-segment column split), mlp.down (row-parallel)
//
// Layouts: bf16, w4a16, w8a16 (TP=1 and TP=2: the w8a16 shard rule slices whole row tiles / K groups) and trellis
// (TP=1 only, RefuseLlamaTrellisTp: the trellis TP=2 shard rules are not built for this arch, docs/nanbeige-plan.md
// N5-5). The head is never trellis: a trellis body's head loads as w4a16 / bf16.
#pragma once

#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "container.h"  // MlpWeights
#include "llama_config.h"
#include "llama_container_info.h"
#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"
#include "trellis_meta.h"

namespace r4dx::model {

struct LlamaAttnWeights {
  QuantLinear q;  // [heads * head_dim, hidden]
  QuantLinear k;  // [kv_heads * head_dim, hidden]
  QuantLinear v;  // [kv_heads * head_dim, hidden]
  QuantLinear o;  // [hidden, heads * head_dim]
};

struct LlamaLayerWeights {
  core::DeviceBuffer<uint16_t> input_norm, post_attn_norm;  // bf16 [hidden]
  LlamaAttnWeights attn;
  MlpWeights mlp;  // gate_up [2 * intermediate, hidden] (gate rows first), down [hidden, intermediate]
};

struct LlamaLoadOptions {
  Layout layout = Layout::kBf16;          // body linears (trellis needs a trellis container)
  Layout lm_head_layout = Layout::kBf16;  // a trellis body's head loads as w4a16 / bf16
  int64_t layer_limit = -1;               // >= 0: load layers [0, layer_limit) (tiny fixtures, nanbeige-l8)
  int tp_world = 1;                       // 1 or 2 (docs/nanbeige-plan.md 3.4)
  int tp_rank = 0;
};

class LlamaContainer {
 public:
  // CPU only (no HIP call, no weight byte read): `__metadata__` + the tensor directory, checked.
  static LlamaContainerInfo Inspect(const std::string& path, int64_t layer_limit = -1) {
    return InspectLlamaContainer(path, layer_limit);
  }

  // Inspect(), then the upload onto the current HIP device.
  static LlamaContainer Load(const std::string& path, const LlamaLoadOptions& o = LlamaLoadOptions());

  LlamaContainer(LlamaContainer&&) = default;
  LlamaContainer& operator=(LlamaContainer&&) = default;

  const LlamaContainerInfo& Info() const { return info_; }
  // The config the loaded weights have: LlamaConfig::Shard(global, tp_world, tp_rank) under TP (vocab_size stays
  // global; VocabShardSize() is the head's rows), else the global one. Info().config is always the GLOBAL config.
  const LlamaConfig& Config() const { return local_config_; }
  const LlamaConfig& GlobalConfig() const { return info_.config; }
  int TpWorld() const { return local_config_.tp_world; }
  int TpRank() const { return local_config_.tp_rank; }
  const ModelConfig& GenericConfig() const { return info_.model_config; }  // thin view for TextModel::Config()
  const std::string& ModelId() const { return info_.model_id; }

  int64_t NumLoadedLayers() const { return static_cast<int64_t>(layers_.size()); }
  const LlamaLayerWeights& Layer(int64_t i) const { return layers_.at(static_cast<size_t>(i)); }
  const core::DeviceBuffer<uint16_t>& EmbedTokensDevice() const { return embed_tokens_; }
  const core::DeviceBuffer<uint16_t>& FinalNorm() const { return final_norm_; }
  const QuantLinear& LmHead() const { return lm_head_; }
  bool HasTrellis() const { return trellis_.has_value(); }
  void ZeroTrellisTickets(hipStream_t stream);

 private:
  LlamaContainer() = default;
  friend class LlamaModel;  // default-constructs it, then assigns LlamaContainer::Load's result (as Lfm2Model)
  template <class Fn>
  void ForEachLinear(Fn&& fn);
  void AssignTrellisTickets();

  LlamaContainerInfo info_;
  LlamaConfig local_config_;  // == info_.config at TP=1
  core::DeviceBuffer<uint16_t> embed_tokens_, final_norm_;
  QuantLinear lm_head_;
  std::vector<LlamaLayerWeights> layers_;
  std::optional<TrellisSpec> trellis_;
  core::DeviceBuffer<uint32_t> trellis_tickets_;
};

// The trellis body runs at TP=1 only (docs/nanbeige-plan.md 7, N5-5). Throws std::invalid_argument naming `who` when
// `layout` is trellis or `path`'s header carries __metadata__.quant.trellis (a CPU header read, no HIP call). Called
// before any device work.
inline void RefuseLlamaTrellisTp(const std::string& path, Layout layout, int64_t layer_limit, const std::string& who) {
  const bool trellis = layout == Layout::kTrellis || LlamaContainer::Inspect(path, layer_limit).has_trellis;
  if (!trellis) return;
  throw std::invalid_argument(who + ": " + path + " -- a Llama trellis container runs at TP=1 only (--tp 1 --layout "
                              "trellis); tensor parallelism (--tp 2) is supported for the bf16 / w8a16 Llama containers");
}

}  // namespace r4dx::model
