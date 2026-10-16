// r4dx::model::Qwen4ExpContainer -- loads (a layer range of) one qwen4_exp .r4dx container onto one device
// (docs/qwen4exp-plan.md 4.11 item 3, task M1-3). A separate type on purpose: Container::Load cannot parse this
// arch (no dense MLP, 4-stream hyper-connections, 512 experts, QSA indexer) and MUST NOT be able to reach it.
// The per-linear layout fallback chain, the raw uploads and the w4a16 / trellis checks are container_load_util's
// (the same code Container::Load and GemmaContainer::Load run).
//
// Two-stage placement (plan 3.3): the model holds two loads, device 0 = layers [0, k) + the host embedding and
// device 1 = layers [k, 48) + the final mixer (text.hc_final) + lm_head; k comes from ChooseSplit over
// Inspect()'s per-layer device bytes.
//
// Tensors per layer (names are the converter's, src/convert/qwen4exp_layout, plan 4.11 item 3):
//   text.layers.{i}.hc.{attn,mlp}.{norm,down,up,inject}     bf16, bare names
//   text.layers.{i}.moe.router / moe.shared_gate            bf16, bare
//   text.layers.{i}.moe.shared.{gate,up,down}.bf16.w        bf16 linears
//   text.layers.{i}.moe.experts.{e}.gate_up.{layout}        [2 * moe_intermediate, hidden], trellis: two parts
//   text.layers.{i}.moe.experts.{e}.down.{layout}           [hidden, moe_intermediate]
//   GDN layers: gdn.in_proj_qkv/in_proj_z/out_proj ({layout}), gdn.in_proj_a/in_proj_b/conv1d_weight (bf16),
//     gdn.A_log/dt_bias (fp32), gdn.norm_weight (bf16, widened to fp32 on the device like Container)
//   QSA layers: attn.qg/k/v/o ({layout}), attn.q_norm/k_norm (bf16), attn.k_descale/v_descale (fp32),
//     attn.indexer.qk / indexer.q_norm / indexer.k_norm (bf16)
//   the PLE layer: ple.key_proj/value_proj/norm_key/norm_query/norm_conv/conv1d (bf16)
// Globals: text.embed_tokens (host-resident, pinned), text.hc_final.{norm,down,up}, lm_head.{layout}.
//
// Expert storage: one device buffer per (layer, class, tensor kind) with per-expert QuantLinear views
// (DeviceBuffer::View) and ONE ticket buffer for the whole load sliced per trellis linear (the
// AssignTrellisTickets recipe of container.cpp / gemma_container.cpp).
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "model_config.h"
#include "nlohmann/json.hpp"
#include "qwen4exp_config.h"
#include "qwen4exp_weights.h"
#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "trellis_meta.h"

namespace r4dx::model {

// The per-card weight budget of plan Q14 (the model's own `--q4-weight-cap`-style override is a ChooseSplit arg).
inline constexpr int64_t kQwen4ExpDefaultWeightCapBytes = 26LL << 30;

// The record `__metadata__.ngram_sidecar` carries (plan 4.2). The file itself is opened by the model
// (host/ngram_sidecar.h); this is only what the container says about it.
struct Qwen4ExpNgramRef {
  std::string path;           // as recorded
  std::string resolved_path;  // `path` when absolute, else relative to the container's directory
  std::string sha256;
  int64_t rows = 0;
  int64_t row_bytes = 0;      // 0 when the record does not carry it
};

// Device bytes one layer occupies once loaded (exactly what Load allocates for it, tickets included).
struct Qwen4ExpLayerBytes {
  int64_t hc = 0;       // both gated-residual blocks
  int64_t mixer = 0;    // GDN or QSA weights
  int64_t dense = 0;    // router, shared expert, shared gate
  int64_t experts = 0;  // the two expert classes (words / sign vectors / bf16 / w4a16)
  int64_t ple = 0;
  int64_t tickets = 0;  // this layer's slice of the load's ticket buffer (trellis linears only)
  int64_t Total() const { return hc + mixer + dense + experts + ple + tickets; }
};

struct Qwen4ExpContainerInfo {
  std::string path, model_id, config_sha256;
  nlohmann::json metadata;
  Qwen4ExpConfig config;
  ModelConfig model_config;  // config.ToModelConfig()
  // `__metadata__.layers_present` (r4dx-convert --layers N < the config's layer count): the container holds the globals
  // and layers [0, layers_present) only. A full container has no such key and layers_present == num_hidden_layers.
  // Inspect checks and sizes the per-layer vectors below over [0, layers_present); Load refuses ranges beyond it.
  int64_t layers_present = 0;
  bool has_trellis = false;
  std::optional<TrellisSpec> trellis;
  Layout body_layout = Layout::kBf16;  // the layout the per-linear table below was resolved for
  Layout head_layout = Layout::kBf16;  // lm_head's (never trellis)
  std::optional<Qwen4ExpNgramRef> ngram;
  std::vector<std::string> tensor_names;  // every tensor of the container, sorted

  // Per layer (index = GLOBAL layer index, size layers_present): the trellis KB of the expert class (all experts of a
  // layer's class share it), 0 when the class is not trellis.
  std::vector<int> gate_up_kb, down_kb;
  std::vector<Qwen4ExpLayerBytes> layer_bytes;  // [layers_present]

  int64_t embed_host_bytes = 0;  // text.embed_tokens (pinned host memory, not VRAM)
  int64_t hc_final_bytes = 0;    // device 1
  int64_t lm_head_bytes = 0;     // device 1

  std::vector<int64_t> PerLayerDeviceBytes() const;
  // Device bytes of everything Load(load_head = true) uploads besides layers.
  int64_t TailDeviceBytes() const { return hc_final_bytes + lm_head_bytes; }
  int64_t TotalDeviceBytes() const;
};

// ChooseSplit's answer: device 0 holds layers [0, k), device 1 layers [k, L).
struct Qwen4ExpSplit {
  int64_t k = 0;
  int64_t dev0_bytes = 0;  // head_bytes + the layers below k
  int64_t dev1_bytes = 0;  // tail_bytes + the layers from k
};

// The layer boundary k in [1, L-1] that balances the two cards' weight bytes (|dev0 - dev1| minimal, the lower k on
// a tie) among the k that keep BOTH cards <= cap_bytes. `head_bytes` is what device 0 holds besides layers (0: the
// embedding is host-resident), `tail_bytes` what device 1 holds (hc_final + lm_head). Throws std::runtime_error
// (naming the totals and the cap) when no k fits, std::invalid_argument for fewer than two layers or a negative input.
Qwen4ExpSplit ChooseSplit(const std::vector<int64_t>& per_layer_bytes, int64_t head_bytes, int64_t tail_bytes,
                          int64_t cap_bytes = kQwen4ExpDefaultWeightCapBytes);

struct Qwen4ExpLoadOptions {
  // Body linears' layout; empty = the container's own (trellis for a trellis container, else bf16). A trellis
  // container loads with kTrellis only and vice versa (CheckTrellisChoice).
  std::optional<Layout> layout;
  // lm_head's; empty = the container's (w4a16 when it carries `lm_head.w4a16.*`, else bf16); kTrellis reads as w4a16.
  std::optional<Layout> lm_head_layout;
  int device = -1;  // HIP device ordinal to upload to (restored afterwards); -1 = the calling thread's current one
  // Layers [layer_first, layer_last) are uploaded (layer_last < 0: through the last PRESENT layer, Info().layers_present).
  // NOTE layer_last is EXCLUSIVE; a range beyond layers_present throws std::invalid_argument.
  int64_t layer_first = 0;
  int64_t layer_last = -1;
  bool load_embed = false;  // text.embed_tokens into pinned host memory
  bool load_head = false;   // text.hc_final + lm_head onto the device
};

class Qwen4ExpContainer {
 public:
  // CPU only (no HIP call besides pure libr4d queries, no weight byte read): `__metadata__` + the tensor
  // directory, fully checked (every expected tensor, byte sizes, layout tiers, per-(layer, class) KB uniformity),
  // plus the device-byte accounting. `body` / `head`: the layouts to resolve for (empty = the container's own).
  static Qwen4ExpContainerInfo Inspect(const std::string& path, std::optional<Layout> body = std::nullopt,
                                       std::optional<Layout> head = std::nullopt);

  // Inspect(), then the upload of the requested range onto `o.device`.
  static Qwen4ExpContainer Load(const std::string& path, const Qwen4ExpLoadOptions& o = Qwen4ExpLoadOptions());

  Qwen4ExpContainer(Qwen4ExpContainer&&) = default;
  Qwen4ExpContainer& operator=(Qwen4ExpContainer&&) = default;

  const Qwen4ExpContainerInfo& Info() const { return info_; }
  const Qwen4ExpConfig& Config() const { return info_.config; }
  const ModelConfig& GenericConfig() const { return info_.model_config; }
  const std::string& ModelId() const { return info_.model_id; }
  int Device() const { return device_; }

  int64_t FirstLayer() const { return first_layer_; }
  int64_t EndLayer() const { return first_layer_ + static_cast<int64_t>(layers_.size()); }
  int64_t NumLoadedLayers() const { return static_cast<int64_t>(layers_.size()); }
  // `i` is the GLOBAL layer index; throws std::out_of_range outside [FirstLayer, EndLayer).
  const qwen4exp::LayerWeights& Layer(int64_t i) const;

  bool HasEmbed() const { return embed_host_ != nullptr; }
  const uint16_t* EmbedTokensHost() const;  // bf16 [vocab, hidden]; throws when load_embed was false
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> EmbedTokensHostShared() const { return embed_host_; }

  bool HasHead() const { return has_head_; }
  const qwen4exp::HcWeights& HcFinal() const;  // throws when load_head was false
  const QuantLinear& LmHead() const;

  bool HasTrellis() const { return info_.has_trellis; }
  void ZeroTrellisTickets(hipStream_t stream);
  int64_t NumTrellisLinears() const { return trellis_linears_; }
  const core::DeviceBuffer<uint32_t>& TrellisTickets() const { return trellis_tickets_; }

  // The VRAM report: the bytes of every OWNING device buffer this load holds (views are windows into the
  // owners and are not counted), tickets included. Equals what DeviceBufferBytes(device) grew by.
  int64_t DeviceBytes() const;

 private:
  Qwen4ExpContainer() = default;
  template <class Fn>
  void ForEachLinear(Fn&& fn);
  template <class Fn>
  void ForEachBuffer(Fn&& fn) const;
  void AssignTrellisTickets();

  Qwen4ExpContainerInfo info_;
  int device_ = -1;
  int64_t first_layer_ = 0;
  std::vector<qwen4exp::LayerWeights> layers_;
  std::shared_ptr<const core::PinnedBuffer<uint16_t>> embed_host_;
  bool has_head_ = false;
  qwen4exp::HcWeights hc_final_;
  QuantLinear lm_head_;
  core::DeviceBuffer<uint32_t> trellis_tickets_;
  int64_t trellis_linears_ = 0;
};

}  // namespace r4dx::model
