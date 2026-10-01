// r4dx::model::GemmaContainer -- loads one gemma4_unified .r4dx container (docs/gemma4-plan.md 4.2, task
// M1-19) into device memory. The Qwen loader (Container) is not reused: it requires GDN fields in the
// config, one attention geometry, no sandwich norms. The per-linear layout fallback chain, the raw uploads,
// the trellis / w4a16 checks and the rotation.* loader are the SAME code Container::Load runs (they were
// moved verbatim to container_load_util.{h,cpp}).
//
// What it loads (TP=1; a rank shard is M1b):
//   text.embed_tokens        bf16 [vocab, hidden], DEVICE-resident always (the scaled gather and the tied head)
//   text.final_norm          bf16 [hidden] (plain weight)
//   lm_head                  QuantLinear [vocab, hidden] (the tied head, written untied; its own layout)
//   per layer: the four sandwich norms (the two folded ones are `.rotated` ones in a rotated container),
//     attn.{q,k,v,o} (no v on a full layer: V is the raw k_proj output), attn.{q,k}_norm,
//     attn.{k,v}_descale (fp32 [kv_heads of THAT layer]: 8 sliding, 1 full), mlp.gate_up / mlp.down, and
//     layer_scalar (read to the host as a float: the fused post-norm kernel takes it by value)
//   rotation.* (option A) when `__metadata__.rotation` is present.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "container.h"  // MlpWeights, RotationWeights
#include "gemma_config.h"
#include "gemma_container_info.h"
#include "gemma_vision_embedder.h"
#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"
#include "trellis_meta.h"

namespace r4dx::model {

struct GemmaAttnWeights {
  QuantLinear q;  // [heads * head_dim, hidden]
  QuantLinear k;  // [kv_heads * head_dim, hidden]
  QuantLinear v;  // sliding layers only (N == 0 on a full layer)
  QuantLinear o;  // [hidden, heads * head_dim]
  core::DeviceBuffer<uint16_t> q_norm, k_norm;    // bf16 [head_dim] (plain weight)
  core::DeviceBuffer<float> k_descale, v_descale;  // fp32 [kv_heads]
};

struct GemmaLayerWeights {
  bool full = false;
  core::DeviceBuffer<uint16_t> input_layernorm, post_attention_layernorm, pre_feedforward_layernorm,
      post_feedforward_layernorm;  // bf16 [hidden]
  GemmaAttnWeights attn;
  MlpWeights mlp;
  float layer_scalar = 1.0f;  // applied once, after the MLP residual add
};

// Vision embedder (docs/gemma4-plan.md M2): kAuto loads the vision.* tensors iff the container carries them,
// kOn requires them (throws when absent), kOff never reads them (~0.1 GB of VRAM saved).
enum class GemmaVisionLoad { kOff, kAuto, kOn };

struct GemmaLoadOptions {
  Layout layout = Layout::kBf16;          // body linears (trellis needs a trellis container)
  Layout lm_head_layout = Layout::kBf16;  // a trellis body's head loads as w4a16 / bf16
  int64_t layer_limit = -1;               // >= 0: load layers [0, layer_limit) (tiny fixtures)
  GemmaVisionLoad vision = GemmaVisionLoad::kOff;
};

class GemmaContainer {
 public:
  // CPU only (no HIP call, no weight byte read): `__metadata__` + the tensor directory, checked.
  static GemmaContainerInfo Inspect(const std::string& path, int64_t layer_limit = -1);

  // Inspect(), then the upload onto the current HIP device.
  static GemmaContainer Load(const std::string& path, const GemmaLoadOptions& o = GemmaLoadOptions());

  GemmaContainer(GemmaContainer&&) = default;
  GemmaContainer& operator=(GemmaContainer&&) = default;

  const GemmaContainerInfo& Info() const { return info_; }
  const GemmaConfig& Config() const { return info_.config; }
  const ModelConfig& GenericConfig() const { return info_.model_config; }  // thin view for TextModel::Config()
  const std::string& ModelId() const { return info_.model_id; }

  int64_t NumLoadedLayers() const { return static_cast<int64_t>(layers_.size()); }
  const GemmaLayerWeights& Layer(int64_t i) const { return layers_.at(static_cast<size_t>(i)); }
  const core::DeviceBuffer<uint16_t>& EmbedTokensDevice() const { return embed_tokens_; }
  const core::DeviceBuffer<uint16_t>& FinalNorm() const { return final_norm_; }
  const QuantLinear& LmHead() const { return lm_head_; }

  bool HasVision() const { return vision_.has_value(); }
  const GemmaVisionWeights& Vision() const { return vision_.value(); }
  bool HasRotation() const { return rotation_.has_value(); }
  const RotationWeights& Rotation() const { return rotation_.value(); }
  bool HasTrellis() const { return trellis_.has_value(); }
  void ZeroTrellisTickets(hipStream_t stream);

 private:
  GemmaContainer() = default;
  friend class GemmaModel;  // default-constructs it, then assigns GemmaContainer::Load's result
  template <class Fn>
  void ForEachLinear(Fn&& fn);
  void AssignTrellisTickets();

  GemmaContainerInfo info_;
  core::DeviceBuffer<uint16_t> embed_tokens_, final_norm_;
  QuantLinear lm_head_;
  std::vector<GemmaLayerWeights> layers_;
  std::optional<RotationWeights> rotation_;
  std::optional<GemmaVisionWeights> vision_;
  std::optional<TrellisSpec> trellis_;
  core::DeviceBuffer<uint32_t> trellis_tickets_;
};

}  // namespace r4dx::model
