// r4dx-convert's inkling2_loop branch (docs/inkling2-plan.md 3.2, 4; task lane A): which HF tensors of an
// Inkling-2-Mini checkpoint (OpenUncensored/Inkling-2-Mini-Uncensored, thinkingmachines/Inkling-2-Mini) become which
// container tensors, with the shape cross-checks and the coverage audit.
//
// Same shell as lfm2_layout.hpp: the Qwen path of RunConvert (main.cpp) is not refactored; AddInkling2TextStack drives a
// `Kit` -- std::function handles onto RunConvert's own closures -- so the --keep-bf16 machinery, the plan/emit job lists
// and the writer are the ones every other arch uses, and tests/convert/test_inkling2_layout.cpp can drive the layout
// against a recording Kit without a weight byte.
//
// HF names ({L} = model.llm.layers.{i}., the checkpoint's ORIGINAL naming; shapes for the real geometry in parens):
//   model.llm.embed.weight [vocab, H] bf16 (201024 rows, kept WHOLE) / embed_norm.weight [H] / norm.weight [H]
//   model.llm.unembed.weight [vocab, H] bf16: only the first unpadded_vocab_size (200058) rows are used
//   model.llm.attn_res.attn_queries.{i-1}   F32 [H], or [3, H] for a FULL layer i   (index j belongs to layer j + 1)
//   model.llm.attn_res.mlp_queries.{i}      F32 [H]
//   model.llm.attn_res.output_query         F32 [H]
//   model.llm.middle_cycle_cross_pass_queries / _kv_queries   F32 [2 * loop_layers, H] each
//   {L}attn.{wq_du,gate_attn}.weight [heads*hd, H], {wk_dv,wv_dv}.weight [kv*hd, H], wo_ud.weight [H, heads*hd],
//       wr_du.weight [heads*d_rel, H], attn_sink_proj.weight [heads, H], q_l2_clip.scale [heads, hd],
//       rel_l2_clip.scale [heads, d_rel], rel_logits_proj.proj [d_rel, rel_extent]
//   sliding layers: {L}attn.{k,v}_sconv.weight [kv*hd, 1, 4], {L}attn_norm.weight
//   full layers:    {L}attn.{k,v}_sconv.static_w [4, kv*hd] + .weight_proj [kv*4, H], {L}{q,k,v}_input_norm.weight
//   {L}mlp_norm.weight, {L}attn_sconv.weight / mlp_sconv.weight [H, 1, 4]
//   loop layers only: {L}controllers.{attn,k,v,mlp}.{down_weight [rank, 2H], forget_weight [H, rank], gain_weight [H, rank]}
//   {L}mlp.gate.weight [E, H] bf16, mlp.gate.bias [E] F32
//   {L}mlp.experts.w13_weight / w13_weight_scale [/ w13_weight_amax [E], w13_weight_outer_scale_denominator [1]], same for
//       w2: NVFP4 (U8 codes + F8_E4M3 scale + amax) or MXFP8 (F8_E4M3 codes + U8 E8M0 scale), see
//       inkling2_expert_codec.hpp
//   model.visual.*  (ViT; allow-listed in milestone 1, logged, not converted)
//
// Container tensors ({L} = the layout suffix: .bf16.w or .mxfp8.wq/.ws):
//   text.embed_tokens (bf16 [vocab, H]), text.embed_norm, text.final_norm (bf16), text.final_query (f32 [H])
//   lm_head.{L}   unembed rows [0, unpadded) -- stored with zero rows up to the next multiple of 16 (mxfp8 needs
//                 N % 16: 200058 -> 200064); metadata inkling2.lm_head_rows says how many rows are real
//   text.layers.{i}.attn_query (f32 [H] or [3,H]; none for i = 0), mlp_query (f32 [H])
//   text.layers.{i}.attn.{q,k,v,gate,o,rel}.{L}            wq_du, wk_dv, wv_dv, gate_attn, wo_ud, wr_du
//   text.layers.{i}.attn.sink                              attn_sink_proj      bf16 [heads, H] (never quantized)
//   text.layers.{i}.attn.{q_scale,rel_scale,rel_proj}      q_l2_clip.scale, rel_l2_clip.scale, rel_logits_proj.proj  bf16
//   sliding: attn.{k,v}_conv  bf16 [kv*hd, 4]  (the [C,1,4] taps SQUEEZED, same bytes)
//   full:    attn.{k,v}_conv_static  bf16 [4, kv*hd]; attn.{k,v}_conv_proj  bf16 [kv*4, H]
//   text.layers.{i}.attn_norm (sliding) | q_norm, k_norm, v_norm (full); mlp_norm; attn_conv, mlp_conv  bf16 ([H,4])
//   text.layers.{i}.msg.{q,k,v,mlp}.{down,forget,gain}     controllers.{attn,k,v,mlp}.*_weight   bf16 (loop layers)
//   text.msg_queries     f32 [loop_layers, 4, H] in the order q, k, v, mlp (Inkling2LoopMergeMessageQueries)
//   text.layers.{i}.moe.router (bf16 [E, H]), moe.bias (f32 [E])
//   text.layers.{i}.moe.experts.{e}.gate_up.{L}  [2I, H]  w13 slice, rows DE-INTERLEAVED to [gate; up]
//   text.layers.{i}.moe.experts.{e}.down.{L}     [H, I]   w2 slice
// Norm weights are stored raw: the runtime uses a plain RMSNorm (x * w, not 1 + w).
#pragma once

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx_convert/inkling2_expert_codec.hpp"

namespace r4dx_convert::inkling2 {

constexpr const char* kModelArch = "inkling2_loop";

// What the converter needs of config.json (text_config). Tensor shapes come from the safetensors headers; this is only
// what they are cross-checked against.
struct TextShape {
  int64_t hidden = 0, layers = 0;
  int64_t heads = 0, kv_heads = 0, head_dim = 0;  // swa_* must equal these (refused otherwise)
  int64_t d_rel = 0, rel_extent = 0, conv_kernel = 0;
  int64_t vocab = 0, unpadded_vocab = 0;
  int64_t experts = 0, moe_intermediate = 0;
  int64_t loop_start = 0, loop_end = 0, loop_passes = 0, gate_rank = 0;  // loop layers [start, end)
  int64_t attn_res_block = 0;
  std::vector<bool> full;  // per layer: full attention (HF "hybrid") vs sliding window ("hybrid_sliding")

  bool IsFull(int64_t i) const { return full.at(static_cast<size_t>(i)); }
  bool IsLoop(int64_t i) const { return i >= loop_start && i < loop_end; }
  int64_t LoopLayers() const { return loop_end - loop_start; }
  // The lm_head rows as stored: unpadded_vocab rounded up to the fragment tile (16).
  int64_t HeadRowsStored() const { return (unpadded_vocab + 15) / 16 * 16; }
};

// Strict. The top-level config (model_type "inkling2_loop") holds `text_config`; a missing key, swa_* different from
// the full-attention geometry, shared experts, loop bounds that are not multiples of n_layers_per_attn_res_block
// (HF's validate_architecture), or a layer-type list that disagrees with local_layer_ids throws std::runtime_error.
// Layer kinds: `layer_types` ("hybrid" = full) if present, else `local_layer_ids` (every other layer is full), else
// HF's default ((i + 1) % 6 == 0 is full).
TextShape ParseConfig(const nlohmann::json& cfg);

// One linear produced on demand as f32 [N, K] (experts, the padded head). `raw_mxfp8` is set only for a source that is
// ALREADY MXFP8: the codes [N*K] and the (row, group) E8M0 bytes [N*K/32] in the final row order, copied bit-exact when
// the container layout is mxfp8 (the f32 `read` then serves a bf16 container).
struct LinearSource {
  int64_t N = 0, K = 0;
  std::function<void(float* out)> read;
  std::function<void(uint8_t* codes, uint8_t* scales)> raw_mxfp8;
};

struct Kit {
  // The checkpoint.
  std::function<std::vector<std::string>()> all_names;
  std::function<bool(const std::string&)> has;
  std::function<std::vector<int64_t>(const std::string&)> shape;
  std::function<std::string(const std::string&)> dtype;           // "BF16", "F32", "U8", "F8_E4M3"
  std::function<const uint8_t*(const std::string&)> data;         // the tensor's bytes (read lazily, at emit time only)
  // One plan job + one emit job each (RunConvert's add_* closures).
  std::function<void(const std::string& hf, const std::string& name)> add_bf16;
  std::function<void(const std::string& hf, const std::string& name, const std::vector<int64_t>& shape)> add_bf16_shaped;
  // F32 (or bf16) source kept f32 at its natural shape (+ the trailing element width 4).
  std::function<void(const std::string& hf, const std::string& name)> add_fp32_widen;
  // The two cross-pass query tensors merged into f32 [loop_layers, 4, H] (MergeMessageQueries).
  std::function<void(const std::string& hf_queries, const std::string& hf_kv_queries, const std::string& name,
                     int64_t loop_layers, int64_t hidden)>
      add_msg_queries;
  // A bf16 checkpoint linear [N, K] in the run's layout (--layouts / --keep-bf16); `head` selects --lm-head's set.
  std::function<void(const std::vector<std::string>& hf, const std::string& base, bool head)> add_linear;
  // A linear computed from checkpoint bytes (an expert slice, the padded head), same layout rules as add_linear.
  std::function<void(const std::string& base, bool head, LinearSource source)> add_source_linear;
  // Optional raw passthrough (bytes copied verbatim with explicit container shape): used when keep_nvfp4_experts is
  // true to store layers 0..20 native NVFP4 experts directly in the .r4dx container (~15.8 GB vs 27.4 GB).
  std::function<void(const std::string& hf, const std::string& name, const std::vector<int64_t>& shape,
                     int64_t elem_bytes)>
      add_raw;
  bool keep_nvfp4_experts = false;
};

// f32 q, kv: [2 * loop_layers, H] each -> out [loop_layers, 4, H] = per loop layer j: q[2j], kv[2j], kv[2j + 1], q[2j + 1]
// (core_model_loading.py Inkling2LoopMergeMessageQueries.convert, line 843-849: query and mlp queries of each loop layer
// from `middle_cycle_cross_pass_queries`, key and value queries from `..._kv_queries`).
void MergeMessageQueries(const float* queries, const float* kv_queries, int64_t loop_layers, int64_t hidden, float* out);

// Registers every tensor of the model. Throws before anything is planned when a tensor is missing, its dtype or shape
// disagrees with the config, or an expert group is inconsistent. Returns every HF tensor name it consumed.
std::set<std::string> AddInkling2TextStack(Kit& kit, const TextShape& s);

// The allow-listed, deliberately unconverted tensors: the ViT (model.visual.*) in milestone 1.
bool IsAllowListed(const std::string& name);

// The coverage audit: checkpoint tensors AddInkling2TextStack did not consume and that are not allow-listed. Anything
// else -- a mis-guessed name, a tensor the checkpoint grew -- must fail the run before the output file is created.
std::vector<std::string> UnconsumedTensors(const std::vector<std::string>& all_names, const std::set<std::string>& consumed);

}  // namespace r4dx_convert::inkling2
