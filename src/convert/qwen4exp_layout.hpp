// r4dx-convert's qwen4_exp branch (docs/qwen4exp-plan.md 4.11 item 3, task M1-2): which HF tensors of a
// Qwen3.8-Flash-Next checkpoint become which container tensors, with the shape cross-checks, the
// coverage audit, the per-(layer, expert class) trellis KB rule and the n-gram sidecar record.
//
// Same construction as gemma_layout.hpp: RunConvert (main.cpp) is one function of closure lambdas that
// every byte-identity test pins; this branch does not refactor it. AddTextStack drives a `Kit` --
// std::function handles onto those closures -- so quantization, --keep-bf16 / --ldlq, the plan/emit job
// lists and the writer are the ones Qwen and Gemma use, and tests/convert/test_convert_qwen4exp_layout.cpp
// drives the layout against a recording Kit without a weight byte.
//
// Container tensors (HF names under `model.language_model.`; {layout} = bf16 | w4a16 | trellis):
//   text.embed_tokens                       embed_tokens.weight                      bf16 [vocab, hidden]
//   lm_head.{layout}                        lm_head.weight (UNtied)                  the head recipe flags
//   text.hc_final.{norm,down,up}            hyper_connection_mixer.{hc_norm, input_mix_weight_down,
//                                           input_mix_weight_up}.weight              bf16 [4h] [lowrank,4h] [4h,lowrank]
//   text.layers.{i}.hc.{attn,mlp}.{norm,down,up,inject}
//                                           {attn,mlp}_hyper_connection.{hc_norm, input_mix_weight_down,
//                                           input_mix_weight_up, block_inject_weight}.weight   bf16, inject [4, 4h]
//   text.layers.{i}.moe.router              mlp.gate.weight                          bf16 [E, hidden]
//   text.layers.{i}.moe.shared.{gate,up,down}   mlp.shared_expert.{gate,up,down}_proj.weight   bf16 linears (.bf16.w)
//   text.layers.{i}.moe.shared_gate         mlp.shared_expert_gate.weight            bf16 [1, hidden]
//   text.layers.{i}.moe.experts.{e}.gate_up.{layout}   mlp.experts.{e}.{gate,up}_proj.weight (gate rows, then up rows)
//   text.layers.{i}.moe.experts.{e}.down.{layout}      mlp.experts.{e}.down_proj.weight
//   GDN layers:   gdn.in_proj_qkv.{layout}, gdn.in_proj_z.{layout}, gdn.out_proj.{layout}   linear_attn.*
//                 gdn.in_proj_a, gdn.in_proj_b (bf16), gdn.conv1d_weight (bf16 [qkv,1,k]),
//                 gdn.A_log, gdn.dt_bias (fp32), gdn.norm_weight (bf16)
//   QSA layers:   attn.qg.{layout} (q_proj as is: its rows are per-head [q | gate], as Qwen3.5),
//                 attn.k.{layout}, attn.v.{layout}, attn.o.{layout}, attn.q_norm, attn.k_norm,
//                 attn.k_descale / attn.v_descale (--kv-calib, fp32 [kv_heads]),
//                 attn.indexer.qk (bf16 [(heads + kv) * head_dim, hidden]), attn.indexer.q_norm / k_norm
//   PLE layers:   ple.key_proj, ple.value_proj, ple.norm_key / norm_query / norm_conv, ple.conv1d (bf16)
// Norm weights are stored raw: the kernels apply (1 + w).
//
// Not converted, on purpose: model.visual.* and mtp.* (milestone 1 is text only; the container says so
// in __metadata__.r4dx_convert_run.skipped) and the n-gram table's tensors (.ple.ple_embedding.*:
// the shards live in the --ngram-sidecar file, the hash buffers in its header).
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "nlohmann/json.hpp"

namespace r4dx_convert::qwen4exp {

constexpr const char* kModelArch = "qwen4_exp";

// What the converter needs of `text_config`; tensor shapes come from the safetensors headers and are
// only cross-checked against these.
struct TextShape {
  int hidden = 0;
  int layers_total = 0;
  int vocab = 0;
  int heads = 0, kv_heads = 0, head_dim = 0;
  int num_experts = 0, moe_inter = 0, shared_inter = 0;
  int hc_count = 0, hc_lowrank = 0;
  int linear_key_heads = 0, linear_value_heads = 0, linear_key_dim = 0, linear_value_dim = 0, conv_kernel = 0;
  int indexer_heads = 0, indexer_kv_heads = 0, indexer_head_dim = 0;
  int ple_embed_dim = 0, ple_conv_kernel = 0;
  std::vector<int> ple_layers;           // 0-based layer indices (config's ple_layer_ids are 1-based)
  std::vector<std::string> layer_types;  // "linear_attention" | "full_attention"

  bool IsFull(int i) const { return layer_types.at(static_cast<size_t>(i)) == "full_attention"; }
  bool HasPle(int i) const;
  int64_t StreamWidth() const { return static_cast<int64_t>(hc_count) * hidden; }
  int64_t QkvRows() const {
    return 2LL * linear_key_heads * linear_key_dim + static_cast<int64_t>(linear_value_heads) * linear_value_dim;
  }
};

// Strict: a missing key, a layer type other than linear_attention/full_attention, tie_word_embeddings
// true (the layout writes an untied head from lm_head.weight) or a ple_layer_ids entry outside
// [1, num_hidden_layers] throws std::runtime_error naming it. `config` is the whole config.json: its
// `text_config` is read, and `tie_word_embeddings` from either level.
TextShape ParseTextConfig(const nlohmann::json& config);

// How a linear is written. kBody: whatever --layouts says, or the trellis manifest; kBf16Only: `.bf16.w`
// only and never trellis (the shared experts); kHead: lm_head's own layout set (--lm-head). Handed to
// add_linear as a VARIABLE, never inline: tools/reference/imatrix_capture.py's audit scans add_linear(...)
// calls and wants exactly one quoted string (the container base) after the HF-name list.
enum class LinearRole { kBody, kBf16Only, kHead };

struct Kit {
  // The checkpoint (with the AWQ overlay: the virtual `<b>.weight` tensors, not the packed quads).
  std::function<std::vector<std::string>()> all_names;
  std::function<bool(const std::string&)> has;
  std::function<std::vector<int64_t>(const std::string&)> shape;
  // One plan job + one emit job each (RunConvert's add_* closures).
  std::function<void(const std::string& hf, const std::string& name)> add_bf16;
  std::function<void(const std::string& hf, const std::string& name)> add_fp32_widen;
  std::function<void(const std::string& name, int n_kv, int layer_idx, const char* kind)> add_descale;
  std::function<void(const std::vector<std::string>& hf, const std::string& base, LinearRole role)> add_linear;
};

// Registers every tensor of the text stack and the globals; `layers` <= layers_total converts the first
// `layers` layers only (--layers). Throws before anything is planned when a tensor is missing or its
// shape disagrees with the config. Returns every HF tensor name it consumed, for UnconsumedTensors.
std::unordered_set<std::string> AddTextStack(Kit& kit, const TextShape& s, int layers);

// True for the checkpoint tensors this converter leaves out on purpose (see the header comment).
bool IsSkipped(const std::string& hf_name);

// The coverage audit: checkpoint tensors neither consumed nor skipped on purpose. Layers >= `layers`
// (--layers) are skipped by request. Sorted.
std::vector<std::string> UnconsumedTensors(const std::vector<std::string>& all_names,
                                           const std::unordered_set<std::string>& consumed, int layers);

// Plan 4.11 item 3: ONE trellis KB per (layer, expert class gate_up|down). `kb_by_base` maps every expert
// container base of a --trellis-from run (`text.layers.{i}.moe.experts.{e}.{gate_up|down}`) to the KB it
// is imported at, or 0 when it is not imported (--keep-bf16). Throws, naming the offending bases, when
// the experts of one layer and class disagree.
void CheckExpertKbUniform(const std::unordered_map<std::string, int>& kb_by_base);

// The sidecar header (tools/reference/qwen4exp/ngram_sidecar.py): what __metadata__.ngram_sidecar records.
struct NgramSidecar {
  std::string path;    // absolute
  std::string sha256;  // lowercase hex of the body, as the header stores it (never rehashed here)
  uint64_t rows = 0;
  uint32_t row_bytes = 0;
  uint32_t version = 0, heads = 0, head_dim = 0, ngram_size = 0, heads_per_ngram = 0;
  int64_t eos_token_id = 0;
  uint64_t seed = 0, padded_rows = 0;
  std::vector<int64_t> multipliers;
  std::vector<uint64_t> head_offsets, head_primes;

  nlohmann::json ToJson() const;
};

// Reads and checks the 4096-byte header of `path` (magic, version 1, size == 4096 + rows * row_bytes,
// zero padding) and cross-checks it against the config: row_bytes == 2 * ple_embed_dim / heads,
// eos id == the config's eos_token_id when that is a single integer. `path` is made absolute.
NgramSidecar ReadNgramSidecar(const std::string& path, const TextShape& s, const nlohmann::json& config);

}  // namespace r4dx_convert::qwen4exp
