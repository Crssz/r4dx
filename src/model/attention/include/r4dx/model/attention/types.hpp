// r4dx::model::attention::AttnConfig / AttnWeights -- the dims and device weight pointers one
// Qwen3.5 full-attention decoder layer needs (docs/architecture.md "Attention layer"). This
// component defines its own small structs rather than depending on the model-core agent's weight-
// loading types (task brief) -- a future assembly stage adapts real container-loaded pointers into
// an AttnWeights; the Integrate stage dedupes any overlap with model-core's own Linear/weight
// abstractions.
#pragma once

#include <cstdint>

namespace r4dx::model::attention {

// Geometry for Qwen3_5ForConditionalGeneration's full-attention layers (docs/architecture.md,
// third_party/libr4d/r4d.h's r4d_attn_dims(): head_dim=256, gqa=6, block_size=16). Defaults match
// the real model config (text_config: hidden_size=5120, num_attention_heads=24,
// num_key_value_heads=4, head_dim=256, partial_rotary_factor=0.25, rope_theta=1e7,
// rms_norm_eps=1e-6).
struct AttnConfig {
  int hidden = 5120;
  int num_heads = 24;
  int kv_heads = 4;
  int head_dim = 256;
  int rotary_dim = 64;       // head_dim * partial_rotary_factor (0.25); r4dx_rope_partial_mrope_bf16
  float rope_theta = 1.0e7f;
  float rms_eps = 1.0e-6f;

  int Gqa() const { return num_heads / kv_heads; }
};

// Device pointers for one full-attention layer's weights. All bf16 pointers point at row-major
// [N,K] linear weights (N=out features, K=in features, matching r4d_gemm_bf16_nt_m64's "nt" =
// W[N,K]^T convention) except q_norm/k_norm ([head_dim]) and input_layernorm ([hidden]).
// k_descale/v_descale are this WRITER's own per-layer [kv_heads] row (r4dx_kv_write_paged_fp8_hnd's
// convention), not R4DArgs' runtime [num_seqs,kv_heads] broadcast table -- AttentionLayer only
// supports num_seqs==1, where the two coincide (r4dx::core::r4d::AttnDecodeFp8Kv's doc comment in
// r4d.hpp).
//
// docs/container-format.md tensor names this maps to (bf16 layout): input_layernorm ->
// text.layers.{i}.input_layernorm; qg_w -> text.layers.{i}.attn.qg.bf16.w (fused q_proj + output
// gate, per-head-interleaved rows); k_w/v_w -> text.layers.{i}.attn.{k,v} (bf16-only, no
// .{layout} suffix); o_w -> text.layers.{i}.attn.o.bf16.w; q_norm/k_norm ->
// text.layers.{i}.attn.{q,k}_norm; k_descale/v_descale -> text.layers.{i}.attn.{k,v}_descale.
struct AttnWeights {
  const uint16_t* input_layernorm = nullptr;  // [hidden] bf16
  const uint16_t* qg_w = nullptr;             // [2*num_heads*head_dim, hidden] bf16
  const uint16_t* k_w = nullptr;              // [kv_heads*head_dim, hidden] bf16
  const uint16_t* v_w = nullptr;              // [kv_heads*head_dim, hidden] bf16
  const uint16_t* o_w = nullptr;              // [hidden, num_heads*head_dim] bf16
  const uint16_t* q_norm = nullptr;           // [head_dim] bf16
  const uint16_t* k_norm = nullptr;           // [head_dim] bf16
  const float* k_descale = nullptr;           // [kv_heads] fp32
  const float* v_descale = nullptr;           // [kv_heads] fp32
};

}  // namespace r4dx::model::attention
