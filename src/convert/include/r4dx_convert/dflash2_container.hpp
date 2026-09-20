// r4dx_convert::dflash2_container -- reads a DFlash2 draft-model GGUF (docs/container-format.md
// "DFlash2 draft container") and writes it as an r4dx container of container_kind
// "dflash2_draft" (see that doc section for the full tensor-naming/metadata spec, cross-checked
// against C:\Users\user\dev\ROCmFPX\src\models\dflash.cpp, common\speculative.cpp, and
// gguf-py's tensor_mapping.py/constants.py -- read-only, not imported).
#pragma once

#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/gguf_reader.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/sha256.hpp"
#include "r4dx_convert/tensor_codec.hpp"

namespace r4dx_convert {

// sha256 of just the first `n` bytes of a file (provenance field -- the real GGUF is ~2GB, hashing
// the whole thing is unnecessary for "does this container still match the source file I built it
// from" purposes; the task spec asks for "sha256 of its first 1 MiB").
inline std::string Sha256HexOfFilePrefix(const std::string& path, size_t n) {
  std::ifstream f(Utf8ToWide(path).c_str(), std::ios::binary);
  if (!f) throw std::runtime_error("dflash2_container: cannot open " + path + " for hashing");
  std::vector<char> buf(n);
  f.read(buf.data(), static_cast<std::streamsize>(n));
  const std::streamsize got = f.gcount();
  Sha256 h;
  h.Update(reinterpret_cast<const uint8_t*>(buf.data()), static_cast<size_t>(got));
  return h.HexDigest();
}

// Every dflash.* metadata field this converter needs, pulled out of the GGUF once and validated
// (task A1: "confirm n_rot + pairing" -- see n_rot's own comment below).
struct Dflash2Metadata {
  int64_t hidden = 0, block_count = 0, ffn = 0;
  int64_t head_count = 0, head_count_kv = 0, key_length = 0, value_length = 0;
  bool causal = false;
  double rms_eps = 1e-6;
  int64_t sliding_window = 0;
  std::vector<bool> sliding_window_pattern;
  double rope_freq_base = 1e7;
  std::vector<int64_t> rope_dimension_sections;
  int64_t n_rot = 0;  // see below
  int64_t block_size = 0, conv_kernel_size = 0, conv_group_size = 0;
  int64_t selector_rank = 0, selector_top_k = 0;
  std::vector<int64_t> target_layers;
  int64_t context_length = 0;
  int64_t mask_token_id = -1;
  int64_t vocab_size = 0;  // derived from selector_predecessor's tensor shape, see below
  int64_t file_type = -1;
};

inline Dflash2Metadata ReadDflash2Metadata(const GgufReader& g) {
  if (g.GetString("general.architecture") != "dflash") {
    throw std::runtime_error("dflash2_container: general.architecture is not 'dflash' (got '" +
                              g.GetString("general.architecture") + "')");
  }
  Dflash2Metadata m;
  m.hidden = g.GetInt("dflash.embedding_length");
  m.block_count = g.GetInt("dflash.block_count");
  m.ffn = g.GetInt("dflash.feed_forward_length");
  m.head_count = g.GetInt("dflash.attention.head_count");
  m.head_count_kv = g.GetInt("dflash.attention.head_count_kv");
  m.key_length = g.GetInt("dflash.attention.key_length");
  m.value_length = g.GetInt("dflash.attention.value_length");
  m.causal = g.HasKey("dflash.attention.causal") ? g.GetBool("dflash.attention.causal") : true;
  m.rms_eps = g.HasKey("dflash.attention.layer_norm_rms_epsilon")
                  ? g.GetFloat("dflash.attention.layer_norm_rms_epsilon")
                  : 1e-6;
  m.sliding_window = g.HasKey("dflash.attention.sliding_window")
                         ? g.GetInt("dflash.attention.sliding_window")
                         : 0;
  if (g.HasKey("dflash.attention.sliding_window_pattern"))
    m.sliding_window_pattern = g.GetBoolArray("dflash.attention.sliding_window_pattern");
  m.rope_freq_base = g.GetFloat("dflash.rope.freq_base");
  m.rope_dimension_sections = g.GetIntArray("dflash.rope.dimension_sections");

  // n_rot (task A1: "state the confirmed n_rot + pairing"). llama-model.cpp's generic hparam load
  // (LLM_KV_ATTENTION_LAYER_NORM_RMS_EPS-adjacent block, `hparams.n_rot_full = n_embd_head_k_full;
  // ml.get_key(LLM_KV_ROPE_DIMENSION_COUNT, hparams.n_rot_full, /*required=*/false)`) defaults
  // n_rot to the FULL head_dim (attention.key_length) whenever a `dflash.rope.dimension_count` key
  // is absent -- CONFIRMED absent from the real Qwen3.8-27B-DFlash2-Q8_0.gguf's metadata dump (48
  // keys total, none named `dflash.rope.dimension_count`). So n_rot = key_length = 128 (the FULL
  // head_dim rotates, not a partial rotary factor like the main text model's 64-of-256).
  // `dimension_sections=[64,0,0,0]` (sum 64) is the M-RoPE PAIR-count split across (temporal,
  // height, width, extra) position axes -- rotated_dims = 2*sum(sections) = 128 = n_rot, i.e. this
  // is a full, single-section (temporal-only) rotation: only a plain sequential position id drives
  // every rotated pair (height/width sections are 0, so no 2D/3D image/video position input is
  // read). PAIRING IS NOT the main text model's interleaved M-RoPE: llama-model.cpp returns
  // LLAMA_ROPE_TYPE_MROPE (not IMROPE) for LLM_ARCH_DFLASH, and ggml's MROPE dispatch
  // (ggml-cpu/ops.cpp `rotate_pairs`, `n_dims/2` stride) pairs `src[ic]` with `src[ic + n_dims/2]`
  // -- GPT-NeoX SPLIT-HALF pairing `(i, i+64)` across all 128 dims. The interleaved scheme only
  // fires on ggml's separate `is_imrope` branch, which LLM_ARCH_DFLASH never takes. This differs
  // from the MAIN text model (0.25 partial rotary factor, 3 active sections [11,11,10],
  // interleaved pairing) in BOTH rotary width and pairing -- DFlash2's own draft attention needs no
  // spatial position awareness, only sequential order, and uses plain NeoX split-half pairs.
  if (g.HasKey("dflash.rope.dimension_count")) {
    m.n_rot = g.GetInt("dflash.rope.dimension_count");
  } else {
    m.n_rot = m.key_length;
  }

  m.block_size = g.GetInt("dflash.block_size");
  m.conv_kernel_size = g.GetInt("dflash.conv_kernel_size");
  m.conv_group_size = g.GetInt("dflash.conv_group_size");
  m.selector_rank = g.GetInt("dflash.selector_rank");
  m.selector_top_k = g.GetInt("dflash.selector_top_k");
  m.target_layers = g.GetIntArray("dflash.target_layers");
  m.context_length = g.GetInt("dflash.context_length");
  m.mask_token_id = g.HasKey("tokenizer.ggml.mask_token_id")
                         ? g.GetInt("tokenizer.ggml.mask_token_id")
                         : -1;
  m.file_type = g.HasKey("general.file_type") ? g.GetInt("general.file_type") : -1;

  // vocab_size: not a scalar dflash.* key (the real file has no dflash.vocab_size and no
  // tokenizer.ggml.tokens array in this draft-only GGUF -- the draft reuses the TARGET model's
  // vocabulary/embedding table wholesale, per the task's own "the draft has NO embedding table and
  // NO lm_head of its own" note) -- derive it from selector_predecessor.weight's own shape
  // (ne=[rank, vocab], one row per vocab id), which must exist in a valid DFlash2 GGUF.
  const auto& sel = g.TensorInfo("selector_predecessor.weight");
  if (sel.ne.size() != 2 || sel.ne[0] != m.selector_rank) {
    throw std::runtime_error("dflash2_container: selector_predecessor.weight shape mismatch");
  }
  m.vocab_size = sel.ne[1];
  return m;
}

// Builds the full __metadata__.dflash2 JSON block from a parsed Dflash2Metadata (docs/
// container-format.md "DFlash2 draft container" -- keep the two in lockstep).
inline nlohmann::json BuildDflash2MetadataJson(const Dflash2Metadata& m) {
  nlohmann::json j;
  j["hidden_size"] = m.hidden;
  j["block_count"] = m.block_count;
  j["feed_forward_length"] = m.ffn;
  j["attention"] = {
      {"head_count", m.head_count},       {"head_count_kv", m.head_count_kv},
      {"key_length", m.key_length},       {"value_length", m.value_length},
      {"causal", m.causal},               {"rms_eps", m.rms_eps},
      {"sliding_window", m.sliding_window},
  };
  j["attention"]["sliding_window_pattern"] = m.sliding_window_pattern;
  j["rope"] = {
      {"freq_base", m.rope_freq_base},
      {"dimension_sections", m.rope_dimension_sections},
      {"n_rot", m.n_rot},
      {"pairing", "neox_split_half"},
  };
  j["block_size"] = m.block_size;
  j["conv_kernel_size"] = m.conv_kernel_size;
  j["conv_group_size"] = m.conv_group_size;
  j["selector_rank"] = m.selector_rank;
  j["selector_top_k"] = m.selector_top_k;
  // Stored EXACTLY as the GGUF gives them (task A1 requirement): 0-based indices into the TARGET
  // model's layer stack marking where each draft encoder layer's feature is read, taken at that
  // layer's INPUT (i.e. target_layers[i]=L means "the residual stream as it enters target layer L"
  // == the OUTPUT of target layer L-1). For the real container: [6,20,34,48,62] = outputs of
  // target layers [5,19,33,47,61].
  j["target_layers"] = m.target_layers;
  j["context_length"] = m.context_length;
  j["mask_token_id"] = m.mask_token_id;
  j["vocab_size"] = m.vocab_size;
  j["source_file_type"] = m.file_type;
  return j;
}

// ---- tensor plumbing: GGUF (any of F32/F16/BF16/Q8_0) -> r4dx layout family --------------------
//
// Orientation (task A1, cross-checked against the real GGUF's own tensor shapes): every linear's
// GGUF `ne` is [K, N] (ne[0]=in/fastest, ne[1]=out) -- e.g. fc.weight ne=[25600,5120] (K=25600
// in, N=5120 out), attn_q.weight ne=[5120,4096] (K=5120 in, N=4096 out). GgufReader::DequantToF32
// returns that tensor's bytes flattened in GGUF's own storage order, which for a 2D tensor is
// exactly row-major [N][K] (ne[1] outer, ne[0] inner) -- BYTE-IDENTICAL to the row-major [N,K]
// ("row = output feature") convention r4dx_convert::PlanLinearLayouts/EmitLinearLayouts already
// consume for the main model's HF-sourced linears (HF's own nn.Linear.weight is also [out,in]
// row-major). So N=ne[1], K=ne[0], w=DequantToF32(name) needs NO transpose/permutation before
// going through the SAME packers (task A1 item 2's explicit instruction) -- verified with a
// dedicated orientation test (tests/convert/test_dflash_container.cpp).
struct Dflash2LinearSpec {
  std::string gguf_name;    // e.g. "blk.0.attn_q.weight"
  std::string container_base;  // e.g. "dflash.layers.0.self_attn.q_proj"
};

inline void PlanDflash2Linear(ContainerWriter& writer, const GgufReader& g,
                               const Dflash2LinearSpec& spec, const LayoutSet& layouts) {
  const auto& info = g.TensorInfo(spec.gguf_name);
  if (info.ne.size() != 2)
    throw std::runtime_error("dflash2_container: '" + spec.gguf_name + "' is not 2D");
  const int N = static_cast<int>(info.ne[1]);
  const int K = static_cast<int>(info.ne[0]);
  PlanLinearLayouts(writer, spec.container_base, N, K, layouts);
}

inline void EmitDflash2Linear(ContainerWriter& writer, const GgufReader& g,
                               const Dflash2LinearSpec& spec, const LayoutSet& layouts,
                               int nthreads) {
  const auto& info = g.TensorInfo(spec.gguf_name);
  const int N = static_cast<int>(info.ne[1]);
  const int K = static_cast<int>(info.ne[0]);
  std::vector<float> w = g.DequantToF32(spec.gguf_name);
  EmitLinearLayouts(writer, spec.container_base, w, N, K, layouts, nthreads);
}

// f32 norm/gate/misc tensor: dequant to f32, store as raw fp32 (task A1: "Norm weights f32" --
// deliberately NOT downcast to bf16, unlike the main text-model container's norms, since the
// source GGUF already stores them f32 and there's no reason to lose precision on a re-convert).
inline void PlanDflash2F32(ContainerWriter& writer, const GgufReader& g,
                            const std::string& gguf_name, const std::string& container_name) {
  const auto& info = g.TensorInfo(gguf_name);
  // Declared shape must be ROW-MAJOR (slowest-varying axis first), matching every other tensor
  // family in an r4dx container (PlanLinearLayouts/add_fp32_widen etc.) -- GGUF's `ne` is the
  // opposite order (ne[0] fastest), so reverse it before recording. For every current caller this
  // tensor is 1D (a norm vector) so the reversal is a no-op, but the multi-dim case must still be
  // right (see PlanDflash2Bf16's conv.base/selector tensors for where this actually bites).
  std::vector<int64_t> shape(info.ne.rbegin(), info.ne.rend());
  shape.push_back(4);  // trailing byte-width axis, matching main.cpp's add_fp32_widen convention
                       // (every container tensor's safetensors dtype is "U8"; the loader infers
                       // the real element width from this trailing axis).
  writer.Plan(container_name, shape, static_cast<uint64_t>(info.ElemCount()) * 4);
}
inline void EmitDflash2F32(ContainerWriter& writer, const GgufReader& g,
                            const std::string& gguf_name, const std::string& container_name) {
  auto w = g.DequantToF32(gguf_name);
  auto bytes = EncodeFp32(w);
  writer.WriteTensor(container_name, bytes.data(), bytes.size());
}

// bf16 passthrough tensor (conv_base, selector_predecessor/successor): dequant to bf16, keep GGUF's
// own element order verbatim (both callers below need no reshape -- see each call site's comment).
inline void PlanDflash2Bf16(ContainerWriter& writer, const GgufReader& g,
                             const std::string& gguf_name, const std::string& container_name) {
  const auto& info = g.TensorInfo(gguf_name);
  // Declared shape must be ROW-MAJOR (slowest-varying axis first) like every other tensor family
  // in an r4dx container -- GGUF's `ne` is fastest-first, the OPPOSITE order, so reverse it before
  // recording (only the byte layout is verbatim/unpermuted; the metadata shape is not GGUF's raw
  // `ne`). E.g. conv.base's GGUF ne=[hidden,2,2] (bytes are [side][tap][hidden], hidden fastest)
  // must declare shape [2,2,hidden,2] (side outer ... byte-width axis inner), not [hidden,2,2,2];
  // selector.predecessor/successor's GGUF ne=[rank,vocab] (bytes are [vocab][rank], rank fastest)
  // must declare [vocab,rank,2], not [rank,vocab,2].
  std::vector<int64_t> shape(info.ne.rbegin(), info.ne.rend());
  shape.push_back(2);
  writer.Plan(container_name, shape, static_cast<uint64_t>(info.ElemCount()) * 2);
}
inline void EmitDflash2Bf16(ContainerWriter& writer, const GgufReader& g,
                             const std::string& gguf_name, const std::string& container_name) {
  auto bf16 = g.DequantToBf16(gguf_name);
  writer.WriteTensor(container_name, bf16.data(), bf16.size() * 2);
}

}  // namespace r4dx_convert
