// r4dx::model::Container -- loads one .r4dx weight container (docs/container-format.md) into
// device memory and exposes it by name/layer. Reader half of the writer/reader contract
// src/convert/include/r4dx_convert/container_writer.hpp implements the writer half of.
//
// Layout selection happens once, at Load() time: every quantized linear in the model (attn.qg/o,
// gdn.in_proj_qkv/out_proj, mlp.gate_up/down) is uploaded in the SAME chosen `Layout`, except
// lm_head, which gets its own independent `lm_head_layout` (the task's "lm_head loaded in the
// chosen layout (bf16 variant if requested)" -- logits are usually kept at higher precision than
// the rest of the model). Tensors that only ever have one on-disk form (embeddings, norms,
// attn.k/v, descales, A_log/dt_bias, conv1d_weight) are uploaded as that form regardless of
// `layout`.
#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "model_config.h"
#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::model {

// Layout and QuantLinear live in quant_linear.h (shared with r4dx_model_attention -- see that
// file's comment for why).

// One decoder layer's weights (docs/container-format.md "Tensor naming"). Exactly one of
// {attn, gdn} is populated, selected by ModelConfig::IsGdnLayer(layer_idx).
struct AttnWeights {
  QuantLinear qg;   // fused q_proj + output gate, [num_heads*head_dim*2, hidden]
  core::DeviceBuffer<uint16_t> k, v;              // bf16 [kv_heads*head_dim, hidden]
  QuantLinear o;                                  // [hidden, num_heads*head_dim]
  core::DeviceBuffer<uint16_t> q_norm, k_norm;    // bf16 [head_dim]
  core::DeviceBuffer<float> k_descale, v_descale;  // fp32 [kv_heads]
};

struct GdnWeights {
  QuantLinear in_proj_qkv;                         // [2*key_dim+value_dim, hidden]
  core::DeviceBuffer<uint16_t> in_proj_z;           // bf16 [value_dim, hidden]
  core::DeviceBuffer<uint16_t> in_proj_b, in_proj_a;  // bf16 [num_v_heads, hidden]
  core::DeviceBuffer<uint16_t> conv1d_weight;       // bf16 [conv_dim, width] (container's
                                                     // trailing singleton axis dropped)
  core::DeviceBuffer<float> A_log, dt_bias;         // fp32 [num_v_heads]
  core::DeviceBuffer<float> norm_weight;            // fp32 [head_v_dim] -- widened from the
                                                     // container's bf16 storage: both consumers
                                                     // (r4d_gdn_gated_rmsnorm_h128_bf16's `w` and
                                                     // r4d_gdn_recurrent_update_*'s `norm_weight`)
                                                     // take `const float*`, not bf16.
  QuantLinear out_proj;                             // [hidden, value_dim]
};

struct MlpWeights {
  QuantLinear gate_up;  // [2*intermediate, hidden]
  QuantLinear down;     // [hidden, intermediate]
};

struct LayerWeights {
  core::DeviceBuffer<uint16_t> input_layernorm;           // bf16 [hidden]
  core::DeviceBuffer<uint16_t> post_attention_layernorm;  // bf16 [hidden]
  std::optional<AttnWeights> attn;  // populated iff !IsGdnLayer(i)
  std::optional<GdnWeights> gdn;    // populated iff IsGdnLayer(i)
  MlpWeights mlp;
};

// MTP self-speculation head (docs/mtp.md, docs/container-format.md "mtp.*"): `layer` is the exact
// tensor set of one text full-attention decoder layer (src/convert/main.cpp's converter comment:
// "mtp.layers.0 is a COMPLETE full-attention decoder layer", verified against the real
// checkpoint's shard header -- NOT a GDN layer, so `layer.gdn` is always empty here and `layer.attn`
// always populated), plus the four tensors with no per-text-layer analogue: the hidden/embedding
// pre-norms and the fc projection that combines them (modeling_qwen3_5.py's Qwen3_5MTPLayer,
// cross-checked against the converter's own tensor-shape comment), and mtp's own final norm before
// the SHARED (main-model) lm_head. `fc` is a plain bf16 linear (container-format.md lists it with
// no `.{layout}` suffix, bf16-only) -- [hidden, 2*hidden], row-major, matching every other raw bf16
// linear this codebase calls via core::r4d::GemmBf16NtM64 (gdn_layer.cpp's in_proj_a/b/z).
struct MtpWeights {
  LayerWeights layer;
  core::DeviceBuffer<uint16_t> fc;                     // bf16 [hidden, 2*hidden]
  core::DeviceBuffer<uint16_t> norm;                   // bf16 [hidden]
  core::DeviceBuffer<uint16_t> pre_fc_norm_hidden;     // bf16 [hidden]
  core::DeviceBuffer<uint16_t> pre_fc_norm_embedding;  // bf16 [hidden]
};

class Container {
 public:
  // Loads `path` onto the current HIP device (caller must have already selected device 1 per the
  // project's GPU rule), synchronously (Container::Load is a startup-path call, not a hot-path
  // one -- every upload below is a plain synchronous DeviceBuffer::CopyFromHost). `layer_limit`,
  // when >= 0, loads only layers [0, layer_limit) -- for the 4-layer test container -- and leaves
  // Layers() sized to layer_limit rather than Config().num_hidden_layers. `mtp_head_layout`
  // (docs/mtp.md "MTP head layout"): the layout used ONLY for the MTP head's four quantized
  // linears (mtp.attn.qg/o, mtp.mlp.gate_up/down) -- independent of `layout`, since the draft
  // head's errors compound across chained draft steps (ModelOptions::mtp_head_layout's own
  // comment). Ignored when the container has no mtp.* weights. `embed_device_resident` (docs/
  // mtp.md "device-resident draft loop", r9700.md P3): if true (default), also mirrors
  // text.embed_tokens into VRAM (~2.54 GB bf16) so RunChunk/MtpHead::Draft can gather embedding
  // rows entirely on-device (EmbedTokensDevice() below) -- checked against free VRAM at the point
  // this call uploads it (before any layer, i.e. against the WHOLE GPU's free memory, not the
  // eventual steady-state footprint after every layer/KV cache is also loaded) with a conservative
  // 2x-headroom heuristic; falls back to host-only (a stderr warning, EmbedTokensDevice() returns
  // nullptr) if that heuristic says it would not fit. EmbedTokensHost() is always populated
  // regardless of this flag -- device residency is purely an additional mirror, never a
  // replacement, so every existing host-gather call site keeps working unchanged.
  static Container Load(const std::string& path, Layout layout, Layout lm_head_layout,
                         int64_t layer_limit = -1, Layout mtp_head_layout = Layout::kBf16,
                         bool embed_device_resident = true);

  const ModelConfig& Config() const { return config_; }
  const std::string& ModelId() const { return model_id_; }
  const std::string& ConfigSha256() const { return config_sha256_; }

  // text.embed_tokens: host-resident (docs/architecture.md), [vocab, hidden] bf16, row-major.
  const uint16_t* EmbedTokensHost() const { return embed_tokens_.data(); }

  // text.embed_tokens' device mirror (docs/mtp.md "device-resident draft loop") -- nullptr when
  // Load() was called with embed_device_resident=false, or when it was true but the free-VRAM
  // heuristic decided it would not fit (see Load()'s own comment); check
  // EmbedTokensDeviceResident() rather than relying on this being non-null implicitly.
  const uint16_t* EmbedTokensDevice() const { return embed_tokens_dev_.data(); }
  bool EmbedTokensDeviceResident() const { return !embed_tokens_dev_.empty(); }

  int64_t NumLoadedLayers() const { return static_cast<int64_t>(layers_.size()); }
  const LayerWeights& Layer(int64_t i) const { return layers_.at(static_cast<size_t>(i)); }

  const core::DeviceBuffer<uint16_t>& FinalNorm() const { return final_norm_; }
  const QuantLinear& LmHead() const { return lm_head_; }

  // True iff `path` was converted with --mtp on (docs/container-format.md "mtp.*") -- checked once
  // at Load() time via SafetensorsReader::Has, not inferred from `__metadata__.r4dx_convert_run.mtp`
  // (a JSON round-trip the loader would otherwise need just to answer this), so it stays correct
  // even against a hand-built or metadata-stripped test container.
  bool HasMtp() const { return mtp_.has_value(); }
  const MtpWeights& Mtp() const { return mtp_.value(); }

 private:
  Container() = default;
  // Model (model.h/model.cpp) default-constructs a Model whose Container member is filled in by
  // Container::Load() right after -- same "private default ctor, public static Load() factory"
  // pattern Container itself uses, one level up.
  friend class Model;

  ModelConfig config_;
  std::string model_id_, config_sha256_;
  core::PinnedBuffer<uint16_t> embed_tokens_;
  core::DeviceBuffer<uint16_t> embed_tokens_dev_;  // empty iff not device-resident (Load's own
                                                    // comment) -- see EmbedTokensDeviceResident()
  std::vector<LayerWeights> layers_;
  core::DeviceBuffer<uint16_t> final_norm_;
  QuantLinear lm_head_;
  std::optional<MtpWeights> mtp_;
};

}  // namespace r4dx::model
