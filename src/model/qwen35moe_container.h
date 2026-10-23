// r4dx::model::Qwen35MoeContainer -- loads (a layer range of) one qwen3_5_moe .r4dx container onto one device
// (docs/qwen35moe-plan.md M3). A separate type on purpose, as Qwen4ExpContainer is: Container::Load cannot parse this arch
// (no dense MLP, 256 experts) and MUST NOT be able to reach it. The per-linear layout fallback chain, the raw uploads and
// the w4a16 / trellis checks are container_load_util's (the same code Container::Load, GemmaContainer::Load and
// Qwen4ExpContainer::Load run); the expert storage structs are qwen4_exp's (qwen4exp_weights.h).
//
// Two-stage placement: the model holds two loads, device 0 = layers [0, k) + the host embedding and device 1 = layers [k, L)
// + the final norm + lm_head; k comes from ChooseSplit over Inspect()'s per-layer device bytes.
//
// Tensors per layer (names are the converter's, src/convert/qwen35moe_layout):
//   text.layers.{i}.input_layernorm / post_attention_layernorm   bf16, bare
//   text.layers.{i}.moe.router / moe.shared_gate                 bf16, bare
//   text.layers.{i}.moe.shared.{gate,up,down}.bf16.w             bf16 linears
//   text.layers.{i}.moe.experts.{e}.gate_up.{layout}             [2 * moe_intermediate, hidden], trellis: two parts
//   text.layers.{i}.moe.experts.{e}.down.{layout}                [hidden, moe_intermediate]
//   GDN layers: gdn.in_proj_qkv/in_proj_z/out_proj ({layout}), gdn.in_proj_a/in_proj_b/conv1d_weight (bf16),
//     gdn.A_log/dt_bias (fp32), gdn.norm_weight (bf16, widened to fp32 on the device like Container)
//   attention layers: attn.qg/k/v/o ({layout}), attn.q_norm/k_norm (bf16), attn.k_descale/v_descale (fp32)
// Globals: text.embed_tokens (host-resident, pinned), text.final_norm, lm_head.{layout}.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "model_config.h"
#include "nlohmann/json.hpp"
#include "qwen35moe_config.h"
#include "qwen4exp_container.h"  // ChooseSplit, Qwen4ExpSplit, kQwen4ExpDefaultWeightCapBytes (arch-independent)
#include "qwen4exp_weights.h"
#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "trellis_meta.h"

namespace r4dx::model {

namespace qwen35moe {

// One full-attention layer's weights (the dense Qwen3.5 AttnWeights, container.h, minus the layer norm).
struct AttnWeights {
  QuantLinear qg;                                  // [2 * heads * head_dim, hidden], per-head interleaved q | gate rows
  QuantLinear k, v;                                // [kv_heads * head_dim, hidden]
  QuantLinear o;                                   // [hidden, heads * head_dim]
  core::DeviceBuffer<uint16_t> q_norm, k_norm;     // bf16 [head_dim]
  core::DeviceBuffer<float> k_descale, v_descale;  // fp32 [kv_heads]
};

struct LayerWeights {
  bool full_attention = false;
  core::DeviceBuffer<uint16_t> input_layernorm;           // bf16 [hidden]
  core::DeviceBuffer<uint16_t> post_attention_layernorm;  // bf16 [hidden]
  std::optional<qwen4exp::GdnWeights> gdn;                // populated iff !full_attention
  std::optional<AttnWeights> attn;                        // populated iff full_attention
  qwen4exp::MoeWeights moe;
};

}  // namespace qwen35moe

// Device bytes one layer occupies once loaded (exactly what Load allocates for it, tickets included).
struct Qwen35MoeLayerBytes {
  int64_t norms = 0;    // input_layernorm + post_attention_layernorm
  int64_t mixer = 0;    // GDN or attention weights
  int64_t dense = 0;    // router, shared expert, shared gate
  int64_t experts = 0;  // the two expert classes (words / sign vectors / bf16 / w4a16)
  int64_t tickets = 0;  // this layer's slice of the load's ticket buffer (trellis linears only)
  int64_t Total() const { return norms + mixer + dense + experts + tickets; }
};

struct Qwen35MoeContainerInfo {
  std::string path, model_id, config_sha256;
  nlohmann::json metadata;
  Qwen35MoeConfig config;
  ModelConfig model_config;  // config.ToModelConfig()
  // `__metadata__.layers_present` (r4dx-convert --layers N < the config's layer count): the container holds the globals and
  // layers [0, layers_present) only. A full container has no such key and layers_present == num_hidden_layers.
  int64_t layers_present = 0;
  bool has_trellis = false;
  std::optional<TrellisSpec> trellis;
  Layout body_layout = Layout::kBf16;  // the layout the per-linear table below was resolved for
  Layout head_layout = Layout::kBf16;  // lm_head's (never trellis)
  std::vector<std::string> tensor_names;  // every tensor of the container, sorted

  // Per layer (index = GLOBAL layer index, size layers_present): the trellis KB of the expert class, 0 when not trellis.
  std::vector<int> gate_up_kb, down_kb;
  std::vector<Qwen35MoeLayerBytes> layer_bytes;  // [layers_present]

  int64_t embed_host_bytes = 0;  // text.embed_tokens (pinned host memory, not VRAM)
  int64_t final_norm_bytes = 0;  // device 1
  int64_t lm_head_bytes = 0;     // device 1

  std::vector<int64_t> PerLayerDeviceBytes() const;
  // Device bytes of everything Load(load_head = true) uploads besides layers.
  int64_t TailDeviceBytes() const { return final_norm_bytes + lm_head_bytes; }
  int64_t TotalDeviceBytes() const;
};

struct Qwen35MoeLoadOptions {
  // Body linears' layout; empty = the container's own (trellis for a trellis container, else bf16).
  std::optional<Layout> layout;
  // lm_head's; empty = the container's (w4a16 when it carries `lm_head.w4a16.*`, else bf16); kTrellis reads as w4a16.
  std::optional<Layout> lm_head_layout;
  int device = -1;  // HIP device ordinal to upload to (restored afterwards); -1 = the calling thread's current one
  // Layers [layer_first, layer_last) are uploaded (layer_last < 0: through the last PRESENT layer). layer_last is EXCLUSIVE.
  int64_t layer_first = 0;
  int64_t layer_last = -1;
  bool load_embed = false;  // text.embed_tokens into pinned host memory
  bool load_head = false;   // text.final_norm + lm_head onto the device
  // Tensor parallelism (docs/qwen35moe-tp-design.md; src/model/qwen35moe_container_tp.cpp): tp_world > 1 loads THIS rank's slice of every
  // tensor (tp::RuleFor, exact byte runs of the one container -- no per-rank file) over the layer range, the final norm, the vocab slice
  // of lm_head and, with load_embed, the (replicated) embedding -- or `shared_embed_host`, the process's one pinned copy, when set.
  int tp_world = 1;
  int tp_rank = 0;
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> shared_embed_host;
};

class Qwen35MoeContainer {
 public:
  // CPU only (no weight byte read): `__metadata__` + the tensor directory, fully checked (every expected tensor, byte
  // sizes, layout tiers, per-(layer, class) KB uniformity), plus the device-byte accounting.
  static Qwen35MoeContainerInfo Inspect(const std::string& path, std::optional<Layout> body = std::nullopt,
                                        std::optional<Layout> head = std::nullopt);

  // Inspect(), then the upload of the requested range onto `o.device`.
  static Qwen35MoeContainer Load(const std::string& path, const Qwen35MoeLoadOptions& o = Qwen35MoeLoadOptions());

  // text.embed_tokens of `path` in one pinned host buffer: the process's ONE copy a --tp 2 group shares (Qwen35MoeLoadOptions::shared_embed_host).
  // Allocates pinned memory, so the calling thread needs a HIP context (any device).
  static std::shared_ptr<const core::PinnedBuffer<uint16_t>> LoadEmbeddingHost(const std::string& path);

  Qwen35MoeContainer(Qwen35MoeContainer&&) = default;
  Qwen35MoeContainer& operator=(Qwen35MoeContainer&&) = default;

  const Qwen35MoeContainerInfo& Info() const { return info_; }
  const Qwen35MoeConfig& Config() const { return info_.config; }
  const ModelConfig& GenericConfig() const { return info_.model_config; }
  // The config the loaded weights have: Config() (global) at TP=1, Qwen35MoeConfig::Shard(Config(), tp_world, tp_rank) under TP.
  const Qwen35MoeConfig& LocalConfig() const { return local_config_; }
  const std::string& ModelId() const { return info_.model_id; }
  int Device() const { return device_; }

  int64_t FirstLayer() const { return first_layer_; }
  int64_t EndLayer() const { return first_layer_ + static_cast<int64_t>(layers_.size()); }
  int64_t NumLoadedLayers() const { return static_cast<int64_t>(layers_.size()); }
  // `i` is the GLOBAL layer index; throws std::out_of_range outside [FirstLayer, EndLayer).
  const qwen35moe::LayerWeights& Layer(int64_t i) const;

  bool HasEmbed() const { return embed_host_ != nullptr; }
  const uint16_t* EmbedTokensHost() const;  // bf16 [vocab, hidden]; throws when load_embed was false
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> EmbedTokensHostShared() const { return embed_host_; }

  bool HasHead() const { return has_head_; }
  const core::DeviceBuffer<uint16_t>& FinalNorm() const;  // throws when load_head was false
  const QuantLinear& LmHead() const;

  bool HasTrellis() const { return info_.has_trellis; }
  void ZeroTrellisTickets(hipStream_t stream);
  int64_t NumTrellisLinears() const { return trellis_linears_; }

  // The VRAM report: the bytes of every OWNING device buffer this load holds (views are windows into the owners and
  // are not counted), tickets included. Equals what DeviceBufferBytes(device) grew by.
  int64_t DeviceBytes() const;

 private:
  Qwen35MoeContainer() = default;
  static Qwen35MoeContainer LoadShard(const std::string& path, const Qwen35MoeLoadOptions& o);  // qwen35moe_container_tp.cpp
  Qwen35MoeConfig local_config_;
  template <class Fn>
  void ForEachLinear(Fn&& fn);
  template <class Fn>
  void ForEachBuffer(Fn&& fn) const;
  void AssignTrellisTickets();

  Qwen35MoeContainerInfo info_;
  int device_ = -1;
  int64_t first_layer_ = 0;
  std::vector<qwen35moe::LayerWeights> layers_;
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> embed_host_;
  bool has_head_ = false;
  core::DeviceBuffer<uint16_t> final_norm_;
  QuantLinear lm_head_;
  core::DeviceBuffer<uint32_t> trellis_tickets_;
  int64_t trellis_linears_ = 0;
};

}  // namespace r4dx::model
