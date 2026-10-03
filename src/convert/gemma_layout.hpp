// r4dx-convert's Gemma 4 branch (docs/gemma4-plan.md 4.2 / 4.3, task M1-6): which HF tensors of a
// gemma4_unified checkpoint become which container tensors, with the shape cross-checks and the
// coverage audit.
//
// The Qwen path of RunConvert (main.cpp) is one function of closure lambdas that every byte-identity
// test pins; this branch does NOT refactor it. AddTextStack drives a `Kit` -- std::function handles
// onto those same closures (add_bf16, add_linear, add_descale, ...) -- so the quantization,
// --keep-bf16 / --ldlq / --rotate machinery, the plan/emit job lists and the writer are exactly the
// ones Qwen uses, and tests/convert/test_gemma_layout.cpp can drive the layout against a recording
// Kit without a single weight byte.
//
// Container tensors (docs/gemma4-plan.md 4.2):
//   text.embed_tokens            model.language_model.embed_tokens.weight        bf16 [vocab, hidden]
//   text.final_norm              model.language_model.norm.weight                bf16 [hidden], plain
//   lm_head.{layout}             the SAME embed_tokens tensor (tied), written untied like Qwen's
//   text.layers.{i}.input_layernorm / post_attention_layernorm / pre_feedforward_layernorm /
//                  post_feedforward_layernorm                                      bf16 [hidden]
//   text.layers.{i}.attn.{q,k,v,o}.{layout}   self_attn.{q,k,v,o}_proj (no `v` on full layers)
//   text.layers.{i}.attn.{q_norm,k_norm}      self_attn.{q,k}_norm.weight         bf16 [head_dim]
//   text.layers.{i}.attn.{k,v}_descale        --kv-calib                          fp32 [kv_heads of THIS layer]
//   text.layers.{i}.mlp.gate_up.{layout}, mlp.down.{layout}
//   text.layers.{i}.layer_scalar              the [1] bf16 buffer, widened        fp32 [1, 4]
//   vision.<rest> / audio.<rest>              model.vision_embedder.* + model.embed_vision.*,
//                                             model.embed_audio.* (bf16 passthrough, `model.` stripped)
// v_norm stores nothing (no weight). Norms are stored raw: the runtime uses a plain RMSNorm.
#pragma once

#include <cstdint>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace r4dx_convert::gemma {

constexpr const char* kModelArch = "gemma4_unified";

// What the converter needs of `text_config`. Shapes of the tensors come from the safetensors headers;
// this is only what they are cross-checked against.
struct TextShape {
  int hidden = 0;
  int layers_total = 0;
  int heads = 0;
  int kv_sliding = 0, kv_full = 0;
  int head_dim = 0, global_head_dim = 0;
  int intermediate = 0;
  int vocab = 0;
  std::vector<std::string> layer_types;  // "sliding_attention" | "full_attention"

  bool IsFull(int i) const { return layer_types.at(static_cast<size_t>(i)) == "full_attention"; }
  int HeadDim(int i) const { return IsFull(i) ? global_head_dim : head_dim; }
  int KvHeads(int i) const { return IsFull(i) ? kv_full : kv_sliding; }
};

// Strict: a missing key, a layer type other than sliding/full, attention_k_eq_v false (the layout
// below has no v_proj on full layers) or tie_word_embeddings false throws std::runtime_error. The
// 5:1 pattern is NOT asserted here (the runtime's GemmaConfig does; a tiny test checkpoint is
// sliding + full).
TextShape ParseTextConfig(const nlohmann::json& text_cfg);

// What --rotate does to a linear, as RunConvert's glue resolves it into its LinearFold:
//   in_norm_hf non-empty: an in-projection; that norm's weight folds into it (W diag(w) Q);
//   out: the out-projection whose K side takes the block Hadamard (Gemma's "option A", Hadamard only);
//   head: this is the tied lm_head (the lm_head layout set applies, never folded).
// A plain struct handed to add_linear as a VARIABLE, never inline: tools/reference/imatrix_capture.py's
// audit scans add_linear(...) calls and wants exactly one quoted string (the container base) after
// the HF-name list.
enum class OutSite { kNone, kDown, kOSliding, kOFull };
struct Fold {
  std::string in_norm_hf;
  OutSite out = OutSite::kNone;
  bool head = false;
};

struct Kit {
  // The checkpoint.
  std::function<std::vector<std::string>()> all_names;
  std::function<bool(const std::string&)> has;
  std::function<std::vector<int64_t>(const std::string&)> shape;
  // One plan job + one emit job each (RunConvert's add_* closures).
  std::function<void(const std::string& hf, const std::string& name)> add_bf16;
  // A norm whose weight --rotate folds into the next linears: add_bf16 unrotated, ones under .rotated.
  std::function<void(const std::string& hf, const std::string& name)> add_folded_norm;
  std::function<void(const std::string& hf, const std::string& name)> add_fp32_widen;
  std::function<void(const std::string& name, int n_kv, int layer_idx, const char* kind)> add_descale;
  std::function<void(const std::vector<std::string>& hf, const std::string& base, const Fold& fold)> add_linear;
};

// Registers every tensor of the text stack, then the embedding table, final norm and tied lm_head, then
// (when asked) the vision / audio passthrough. `layers` <= layers_total converts the first `layers`
// layers only (--layers). Throws before anything is planned when a tensor is missing or its shape
// disagrees with the config, v_proj exists on a k_eq_v layer, or it is absent on a sliding layer.
// Returns every HF tensor name it consumed, for UnconsumedTensors.
std::set<std::string> AddTextStack(Kit& kit, const TextShape& s, int layers, bool do_vision, bool do_audio);

// The coverage audit: checkpoint tensors AddTextStack did not consume and that are not allow-listed.
// Allow-listed on purpose: layers >= `layers` (--layers), the vision tensors when !do_vision and the
// audio tensors when !do_audio. Anything else -- a mis-guessed name, a tensor the checkpoint grew --
// must fail the run before a 20-minute conversion, not vanish from the container.
std::vector<std::string> UnconsumedTensors(const std::vector<std::string>& all_names,
                                           const std::set<std::string>& consumed, int layers, bool do_vision,
                                           bool do_audio);

// "model.vision_embedder.X" -> "vision.vision_embedder.X", "model.embed_vision.X" -> "vision.embed_vision.X",
// "model.embed_audio.X" -> "audio.embed_audio.X": the HF name without `model.`, under its modality.
// Empty when `hf` is not a non-text tensor.
std::string PassthroughName(const std::string& hf, bool* is_vision);

}  // namespace r4dx_convert::gemma
