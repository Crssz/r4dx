// r4dx::model::qwen4exp weight structs -- what Qwen4ExpContainer::Load uploads (docs/qwen4exp-plan.md 4.11 item 3).
// One decoder layer = two hyper-connection blocks (around the mixer and around the MoE), a mixer (GDN or QSA),
// the MoE (router, shared expert, 512 routed experts) and, on the PLE layer, the PLE block. Norm weights are the
// raw checkpoint values (the kernels apply `1 + w`).
//
// The routed experts are NOT 1024 device allocations per layer: ExpertClassStorage owns ONE buffer per tensor kind
// for all experts of one (layer, class), and every expert's QuantLinear holds non-owning DeviceBuffer::View windows
// into it (docs/qwen4exp-plan.md 4.11 item 5). A QuantLinear of a view is used exactly like an owning one
// (ApplyLinear reads .data() / .size() only); it must not outlive the MoeWeights that own the storage.
#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"

namespace r4dx::model::qwen4exp {

// One gated-residual block (docs/qwen4exp-plan.md 2.2). All bf16. `inject` is empty on the final mixer.
struct HcWeights {
  core::DeviceBuffer<uint16_t> norm;    // [4 * hidden]
  core::DeviceBuffer<uint16_t> down;    // [hc_lowrank, 4 * hidden]
  core::DeviceBuffer<uint16_t> up;      // [4 * hidden, hc_lowrank]
  core::DeviceBuffer<uint16_t> inject;  // [hc_count, 4 * hidden]; empty for text.hc_final
};

struct GdnWeights {
  QuantLinear in_proj_qkv;                          // [2 * key_dim + value_dim, hidden]
  QuantLinear in_proj_z;                            // [value_dim, hidden]
  core::DeviceBuffer<uint16_t> in_proj_a, in_proj_b;  // bf16 [num_v_heads, hidden]
  core::DeviceBuffer<uint16_t> conv1d_weight;       // bf16 [conv_dim * 1 * kernel] (HF conv1d.weight)
  core::DeviceBuffer<float> A_log, dt_bias;         // fp32 [num_v_heads]
  core::DeviceBuffer<float> norm_weight;            // fp32 [head_v_dim], widened from the bf16 storage
  QuantLinear out_proj;                             // [hidden, value_dim]
};

struct QsaWeights {
  QuantLinear qg;                                  // [2 * heads * head_dim, hidden], per-head interleaved q|gate rows
  QuantLinear k, v;                                // [kv_heads * head_dim, hidden]
  QuantLinear o;                                   // [hidden, heads * head_dim]
  core::DeviceBuffer<uint16_t> q_norm, k_norm;     // bf16 [head_dim]
  core::DeviceBuffer<float> k_descale, v_descale;  // fp32 [kv_heads]
  core::DeviceBuffer<uint16_t> indexer_qk;         // bf16 [(indexer_n_heads + 1) * indexer_head_dim, hidden]
  core::DeviceBuffer<uint16_t> indexer_q_norm, indexer_k_norm;  // bf16 [indexer_head_dim]
};

struct SharedExpertWeights {
  QuantLinear gate, up;  // bf16 [shared_intermediate, hidden]
  QuantLinear down;      // bf16 [hidden, shared_intermediate]
};

// Storage of one (layer, expert class). Exactly the buffers of `layout` are populated; each holds the E experts'
// tensors back to back in expert order (expert e's bytes start at e times one expert's size).
struct ExpertClassStorage {
  Layout layout = Layout::kBf16;
  int64_t N = 0, K = 0;  // every expert's linear is [N, K]
  int64_t experts = 0;
  int bits = 0;          // trellis only: the class's one KB (3, 4 or 5); 0 otherwise
  core::DeviceBuffer<uint16_t> bf16_w;  // kBf16: E * N * K
  core::DeviceBuffer<uint8_t> wq;       // kW4a16: E * N * K / 2
  core::DeviceBuffer<uint32_t> wsz;     // kW4a16: E * N * K / group
  core::DeviceBuffer<uint32_t> trellis_w;  // kTrellis: E * N * K * bits / 32
  core::DeviceBuffer<float> trellis_suh;   // kTrellis: E * parts * K
  core::DeviceBuffer<float> trellis_svh;   // kTrellis: E * N
};

// One routed expert's two linears: views into the class storages of the same MoeWeights.
struct ExpertLinears {
  QuantLinear gate_up;  // [2 * moe_intermediate, hidden]; rows = gate then up (two trellis parts)
  QuantLinear down;     // [hidden, moe_intermediate]
};

struct MoeWeights {
  core::DeviceBuffer<uint16_t> router;       // bf16 [num_experts, hidden]
  SharedExpertWeights shared;
  core::DeviceBuffer<uint16_t> shared_gate;  // bf16 [1, hidden]
  ExpertClassStorage gate_up_store, down_store;
  std::vector<ExpertLinears> experts;        // [num_experts]: views into the two stores
};

// The PLE block (docs/qwen4exp-plan.md 2.3); only on the layers Qwen4ExpConfig::ple_layer_indices names.
struct PleWeights {
  core::DeviceBuffer<uint16_t> key_proj;    // bf16 [4 * hidden, hidden]
  core::DeviceBuffer<uint16_t> value_proj;  // bf16 [hidden, hidden]
  core::DeviceBuffer<uint16_t> norm_key, norm_query, norm_conv;  // bf16 [4 * hidden]
  core::DeviceBuffer<uint16_t> conv1d;      // bf16 [4 * hidden * 1 * kernel] (dilation 3)
};

struct LayerWeights {
  bool qsa = false;
  HcWeights hc_attn, hc_mlp;
  std::optional<GdnWeights> gdn;  // populated iff !qsa
  std::optional<QsaWeights> attn;  // populated iff qsa
  MoeWeights moe;
  std::optional<PleWeights> ple;
};

}  // namespace r4dx::model::qwen4exp
