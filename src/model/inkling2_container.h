// r4dx::model::Inkling2Container -- loads one inkling2_loop .r4dx container (docs/inkling2-plan.md 4, 5; the names are
// src/convert/inkling2_layout.hpp's) into device memory, whole (TP=1) or one tensor-parallel rank's shard (TP=2).
//
// Linears (QuantLinear, the per-linear layout fallback chain of container_load_util: the requested layout, else the
// linear's `.bf16.w`, so a `--keep-bf16` linear of an mxfp8 container loads bf16 and is counted):
//   text.layers.{i}.attn.{q,gate}  [heads*hd, H]  column-parallel by q head
//   text.layers.{i}.attn.{k,v}     [kv*hd, H]     column-parallel by kv head
//   text.layers.{i}.attn.o         [H, heads*hd]  row-parallel (K by q head) -> the all-reduce
//   text.layers.{i}.attn.rel       [heads*d_rel, H]  REPLICATED (every rank computes all heads' relative states and reads
//                                  its own heads': the 16-row-per-head slices of the tiny fixture are not whole tiles)
//   text.layers.{i}.moe.experts.{e}.gate_up [2I, H] (gate rows, then up rows: each half split), .down [H, I] (K split)
//   lm_head [HeadRowsStored, H]    the stored rows (unpadded vocab rounded up to 16) split in halves; rows past
//                                  unpadded_vocab_size are zero and never reach a logit
// Everything else is uploaded as the kernels read it (host conversion / slicing at load; none of it is large):
//   bf16 [H]:   attn_norm (sliding) | q_norm, k_norm, v_norm (full), mlp_norm, text.embed_norm, text.final_norm
//   bf16:       attn.sink [heads, H] (rank's heads), attn.{k,v}_conv_proj [kv*4, H] (rank's kv heads), moe.router [E, H],
//               msg.{q,k,v,mlp}.down [rank, 2H], text.embed_tokens [vocab, H] (replicated)
//   fp32 (widened from bf16): attn.q_scale [heads, hd], attn.rel_scale [heads, d_rel] (rank's heads),
//               attn.rel_proj [d_rel, extent], attn.{k,v}_conv [kv*hd, 4] (sliding, rank's channels),
//               attn.{k,v}_conv_static [4, kv*hd] (full, rank's columns), attn_conv / mlp_conv [H, 4],
//               msg.*.{forget,gain} [H, rank]
//   fp32:       attn_query [1|3, H] (none on layer 0), mlp_query [H], moe.bias [E], text.final_query [H],
//               text.msg_queries [loop_layers, 4, H]
// Anything missing, or of the wrong byte size, throws naming the tensor before the model runs.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "inkling2_config.h"
#include "model_config.h"
#include "nlohmann/json.hpp"
#include "quant_linear.h"
#include "r4dx/core/device_buffer.hpp"

namespace r4dx::model {

struct Inkling2AttnWeights {
  QuantLinear q, k, v, gate, o, rel;
  core::DeviceBuffer<uint16_t> sink;                  // bf16 [heads_local, H]
  core::DeviceBuffer<float> q_scale, rel_scale;       // fp32 [heads_local, hd] / [heads_local, d_rel]
  core::DeviceBuffer<float> rel_proj;                 // fp32 [d_rel, extent]
  core::DeviceBuffer<float> k_conv, v_conv;           // sliding: fp32 [kvd_local, 4]
  core::DeviceBuffer<float> k_conv_static, v_conv_static;  // full: fp32 [4, kvd_local]
  core::DeviceBuffer<uint16_t> k_conv_proj, v_conv_proj;   // full: bf16 [kv_local * 4, H]
  // S4.1: concatenated MXFP8 weight in row order [q, gate, rel, k, v] (zero duplicate VRAM: when populated,
  // q/gate/rel/k/v have their mx_wq/mx_ws cleared and slice into this contiguous 16-row-tile buffer).
  core::DeviceBuffer<uint8_t> qgrkv_wq, qgrkv_ws;
  int64_t qgrkv_n = 0, qgr_n = 0;
  int64_t q_row0 = 0, gate_row0 = 0, rel_row0 = 0, k_row0 = 0, v_row0 = 0;
};

struct Inkling2MsgGateWeights {
  core::DeviceBuffer<uint16_t> down;          // bf16 [rank, 2H]
  core::DeviceBuffer<uint16_t> forget, gain;  // bf16 [H, rank]
};

struct Inkling2LayerWeights {
  bool full = false, loop = false;
  int n_attn_queries = 0;                     // 0 (layer 0), 1 (sliding), 3 (full)
  core::DeviceBuffer<float> attn_query;       // fp32 [n_attn_queries, H]
  core::DeviceBuffer<float> mlp_query;        // fp32 [H]
  core::DeviceBuffer<uint16_t> attn_norm, q_norm, k_norm, v_norm, mlp_norm;  // bf16 [H]
  core::DeviceBuffer<float> attn_conv, mlp_conv;                              // fp32 [H, 4]
  Inkling2AttnWeights attn;
  Inkling2MsgGateWeights msg[4];              // loop layers: q, k, v, mlp
  core::DeviceBuffer<uint16_t> router;        // bf16 [E, H]
  core::DeviceBuffer<float> bias;             // fp32 [E]
  std::vector<QuantLinear> gate_up, down;     // [E]
  // uint64 [E] device tables of the experts' weight bases (the grouped MoE kernels of a later stage index them):
  // mxfp8/nvfp4: wq / ws, bf16: w. Empty when the experts' layouts are mixed.
  core::DeviceBuffer<uint64_t> gate_up_ptrs0, gate_up_ptrs1, down_ptrs0, down_ptrs1;
  // Native NVFP4 MoE experts (when loaded via R4DX_INK_NVFP4_EXPERTS): half the VRAM and bandwidth of MXFP8.
  bool nvfp4_experts = false;
  core::DeviceBuffer<uint8_t> nvfp4_gu_wq, nvfp4_gu_ws, nvfp4_dn_wq, nvfp4_dn_ws;
  core::DeviceBuffer<float> gate_up_factors, down_factors;  // [E]
};

struct Inkling2LoadOptions {
  Layout layout = Layout::kMxfp8;  // every linear incl. the head (per-linear bf16 fallback)
  int tp_world = 1;
  int tp_rank = 0;
};

class Inkling2Container {
 public:
  // CPU only: `__metadata__` -> the GLOBAL config (throws unless model_arch / model_config describe inkling2_loop).
  static Inkling2Config InspectConfig(const std::string& path, nlohmann::json* metadata_out = nullptr);
  static Inkling2Container Load(const std::string& path, const Inkling2LoadOptions& o = Inkling2LoadOptions());

  Inkling2Container(Inkling2Container&&) = default;
  Inkling2Container& operator=(Inkling2Container&&) = default;

  const Inkling2Config& Config() const { return local_; }       // rank-local (== global at TP=1)
  const Inkling2Config& GlobalConfig() const { return global_; }
  const ModelConfig& GenericConfig() const { return generic_; }  // global, for TextModel::Config()
  const std::string& ModelId() const { return model_id_; }
  int64_t NumLoadedLayers() const { return static_cast<int64_t>(layers_.size()); }
  const Inkling2LayerWeights& Layer(int64_t i) const { return layers_.at(static_cast<size_t>(i)); }
  const core::DeviceBuffer<uint16_t>& EmbedTokens() const { return embed_tokens_; }
  const core::DeviceBuffer<uint16_t>& EmbedNorm() const { return embed_norm_; }
  const core::DeviceBuffer<uint16_t>& FinalNorm() const { return final_norm_; }
  const core::DeviceBuffer<float>& FinalQuery() const { return final_query_; }
  const core::DeviceBuffer<float>& MsgQueries() const { return msg_queries_; }  // [loop_layers, 4, H]
  const QuantLinear& LmHead() const { return lm_head_; }  // [HeadRowsLocal, H]
  int BodyFallbacks() const { return fallbacks_; }

 private:
  Inkling2Container() = default;
  friend class Inkling2Model;

  Inkling2Config global_, local_;
  ModelConfig generic_;
  std::string model_id_;
  core::DeviceBuffer<uint16_t> embed_tokens_, embed_norm_, final_norm_;
  core::DeviceBuffer<float> final_query_, msg_queries_;
  QuantLinear lm_head_;
  std::vector<Inkling2LayerWeights> layers_;
  int fallbacks_ = 0;
};

}  // namespace r4dx::model
