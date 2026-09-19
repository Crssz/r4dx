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
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/pinned_buffer.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::model {

enum class Layout { kBf16, kMxfp4, kW4a16, kW4a8 };

const char* LayoutName(Layout l);
Layout LayoutFromName(const std::string& name);  // throws on an unrecognized name

// One linear weight W[N,K], uploaded in exactly one of the four on-disk layouts
// (docs/container-format.md "Quantized layout tensors"). linear.cpp's ApplyLinear is the only
// thing that reads the layout-specific buffers below; Container's job stops at "the right bytes
// are on the device in the container's documented byte order".
struct QuantLinear {
  Layout layout = Layout::kBf16;
  int64_t N = 0, K = 0;  // W is [N, K]: N output features, K input features

  // layout == kBf16: W itself, row-major [N, K], bf16.
  core::DeviceBuffer<uint16_t> bf16_w;

  // layout == kW4a16 or kW4a8: nibble-packed, WMMA-fragment-permuted weight, uint8[N*K/2].
  // Byte-identical between the two layouts is NOT assumed here (src/convert's quant_int4.hpp
  // quantizes w4a16 and w4a8 separately -- see its file comment) -- each QuantLinear holds only
  // the one layout it was loaded as.
  core::DeviceBuffer<uint8_t> wq;
  // layout == kW4a16: uint32[N*K/128], low16 = f16 scale, high16 = f16(-(1024+zero)).
  core::DeviceBuffer<uint32_t> w4a16_wsz;
  // layout == kW4a8: uint32[N*K/128], low16 = f16 scale (high16 unused).
  core::DeviceBuffer<uint32_t> w4a8_ws;

  // layout == kMxfp4: OCP MXFP4 weight.
  core::DeviceBuffer<uint8_t> mxfp4_wq;    // uint8[N*K/2], fragment-permuted e2m1 pairs
  core::DeviceBuffer<uint8_t> mxfp4_ws;    // uint8[(K/32)*N], E8M0 exponent per (group, row)
  core::DeviceBuffer<int8_t> mxfp4_wref;   // int8[N], per-row reference exponent
};

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

class Container {
 public:
  // Loads `path` onto the current HIP device (caller must have already selected device 1 per the
  // project's GPU rule), synchronously (Container::Load is a startup-path call, not a hot-path
  // one -- every upload below is a plain synchronous DeviceBuffer::CopyFromHost). `layer_limit`,
  // when >= 0, loads only layers [0, layer_limit) -- for the 4-layer test container -- and leaves
  // Layers() sized to layer_limit rather than Config().num_hidden_layers.
  static Container Load(const std::string& path, Layout layout, Layout lm_head_layout,
                         int64_t layer_limit = -1);

  const ModelConfig& Config() const { return config_; }
  const std::string& ModelId() const { return model_id_; }
  const std::string& ConfigSha256() const { return config_sha256_; }

  // text.embed_tokens: host-resident (docs/architecture.md), [vocab, hidden] bf16, row-major.
  const uint16_t* EmbedTokensHost() const { return embed_tokens_.data(); }

  int64_t NumLoadedLayers() const { return static_cast<int64_t>(layers_.size()); }
  const LayerWeights& Layer(int64_t i) const { return layers_.at(static_cast<size_t>(i)); }

  const core::DeviceBuffer<uint16_t>& FinalNorm() const { return final_norm_; }
  const QuantLinear& LmHead() const { return lm_head_; }

 private:
  Container() = default;
  // Model (model.h/model.cpp) default-constructs a Model whose Container member is filled in by
  // Container::Load() right after -- same "private default ctor, public static Load() factory"
  // pattern Container itself uses, one level up.
  friend class Model;

  ModelConfig config_;
  std::string model_id_, config_sha256_;
  core::PinnedBuffer<uint16_t> embed_tokens_;
  std::vector<LayerWeights> layers_;
  core::DeviceBuffer<uint16_t> final_norm_;
  QuantLinear lm_head_;
};

}  // namespace r4dx::model
