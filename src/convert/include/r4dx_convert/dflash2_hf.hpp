// r4dx_convert::dflash2_hf -- HF DFlash v1 checkpoint (z-lab/gemma4-12B-it-DFlash, a Qwen3-style
// `DFlashDraftModel`) -> r4dx dflash2 draft container, directly (not through a llama.cpp GGUF).
// docs/gemma4-plan.md 6.4 / D-3.
//
// v1 has no dynamic depthwise conv and no selector lattice, so both are SYNTHESIZED so that the dflash2
// runtime computes exactly the v1 forward and the v1 greedy draft:
//   * conv:     `conv.base[side][tap=0][:] = 1`, `conv.base[side][tap=1][:] = 0`, `conv.proj = 0` (dyn = 0):
//               out[t,c] = (1 + 0) * x[t,c] + (0 + 0) * x[t-1,c] = x[t,c] bit for bit (x is bf16, the
//               fp32 accumulate of 1*x + 0*xprev is exact, the bf16 round of a bf16 value is the value).
//   * selector: `selector.hidden = 0` and both codebooks 0: gate = 0, so every cross term is 0 and
//               score[a,b] = unary[b]; the walk emits each position's top-1 (candidate 0), i.e. the v1
//               per-position argmax (the lm_head top-16 is sorted, ties to the lower id like argmax).
//
// Tensor map (HF name -> container name; every linear [N,K] row-major, the same bytes both sides):
//   fc.weight                          -> dflash.fc                         (linear, K = n_target_layers * hidden)
//   hidden_norm.weight                 -> dflash.enc_output_norm            (f32)   z-lab: hidden_norm(fc(feats))
//   norm.weight                        -> dflash.output_norm                (f32)
//   layers.i.input_layernorm.weight    -> dflash.layers.i.input_layernorm   (f32)
//   layers.i.post_attention_layernorm  -> dflash.layers.i.post_attention_layernorm (f32)
//   layers.i.self_attn.{q,k,v,o}_proj  -> dflash.layers.i.self_attn.{q,k,v,o}_proj (linear)
//   layers.i.self_attn.{q,k}_norm      -> dflash.layers.i.self_attn.{q,k}_norm (f32)
//   layers.i.mlp.{gate,up,down}_proj   -> dflash.layers.i.mlp.{gate,up,down}_proj (linear)
//   (synthesized) layers.i.{self_attn,mlp}.conv.{base,proj}, selector.{hidden,predecessor,successor}
// z-lab's context path (k_proj / v_proj applied straight to hidden_norm(fc(features)), no input
// layernorm) is exactly dflash2's InjectFeatures; the draft block's own path is input_layernorm ->
// q/k/v -> q_norm/k_norm -> rope -> non-causal windowed attention -> o_proj, then post_attention_layernorm
// -> SwiGLU: dflash2's forward with an identity conv.
//
// Conventions read from z-lab/dflash (dflash/model.py), recorded here because they decide the metadata:
//   * target features: `hidden_states[layer_id + 1]` (offset = 1), i.e. the OUTPUT of target layer
//     `layer_id` == the INPUT of target layer `layer_id + 1`. The container stores layer-INPUT indices, so
//     target_layers = target_layer_ids + offset ([1,10,19,27,36,45] -> [2,11,20,28,37,46]).
//     `--dflash-target-layer-offset 0` stores the ids as given (the acceptance A/B knob).
//   * the draft block's embedding is `F.embedding(ids, target_embed_weight) * input_embedding_scale`
//     with `input_embedding_scale` defaulting to 1.0 and ABSENT from this checkpoint's config: the RAW
//     (unscaled) table rows, NOT Gemma's sqrt(hidden)-scaled ones. Stored as `embed_scale` (1.0).
//   * logits: target lm_head over the final-normed draft output, optionally `tanh(l / cap) * cap` when
//     config `final_logit_softcapping` is set (30 here); `output_multiplier` (default 1) is refused if != 1.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/dflash2_container.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/safetensors_reader.hpp"
#include "r4dx_convert/tensor_codec.hpp"

namespace r4dx_convert {

struct DflashHfConfig {
  int64_t hidden = 0, layers = 0, ffn = 0, heads = 0, kv_heads = 0, head_dim = 0;
  double rms_eps = 1e-6;
  int64_t sliding_window = 0;
  std::vector<bool> sliding;  // per layer: true == sliding_attention
  double rope_theta = 10000.0;
  int64_t block_size = 0;
  std::vector<int64_t> target_layer_ids;
  int64_t mask_token_id = -1;
  double softcap = 0.0;
  int64_t vocab = 0;
  int64_t context = 0;
  double input_embedding_scale = 1.0;
};

inline std::string ReadTextFileUtf8(const std::string& path) {
  std::ifstream f(Utf8ToWide(path).c_str(), std::ios::binary);
  if (!f) throw std::runtime_error("dflash2_hf: cannot open " + path);
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

inline DflashHfConfig ParseDflashHfConfig(const nlohmann::json& j) {
  DflashHfConfig c;
  auto get_int = [&](const nlohmann::json& o, const char* k) -> int64_t {
    if (!o.contains(k) || o.at(k).is_null())
      throw std::runtime_error(std::string("dflash2_hf: config.json is missing '") + k + "'");
    return o.at(k).get<int64_t>();
  };
  if (j.contains("architectures")) {
    bool ok = false;
    for (const auto& a : j.at("architectures")) ok |= a.get<std::string>() == "DFlashDraftModel";
    if (!ok) throw std::runtime_error("dflash2_hf: config.json architectures is not DFlashDraftModel");
  }
  c.hidden = get_int(j, "hidden_size");
  c.layers = get_int(j, "num_hidden_layers");
  c.ffn = get_int(j, "intermediate_size");
  c.heads = get_int(j, "num_attention_heads");
  c.kv_heads = get_int(j, "num_key_value_heads");
  c.head_dim = j.contains("head_dim") && !j.at("head_dim").is_null() ? j.at("head_dim").get<int64_t>()
                                                                     : c.hidden / c.heads;
  if (j.contains("rms_norm_eps")) c.rms_eps = j.at("rms_norm_eps").get<double>();
  c.block_size = get_int(j, "block_size");
  c.vocab = get_int(j, "vocab_size");
  c.context = j.contains("max_position_embeddings") ? j.at("max_position_embeddings").get<int64_t>() : 0;
  if (j.contains("rope_parameters") && j.at("rope_parameters").contains("rope_theta")) {
    c.rope_theta = j.at("rope_parameters").at("rope_theta").get<double>();
    if (j.at("rope_parameters").contains("rope_type") &&
        j.at("rope_parameters").at("rope_type").get<std::string>() != "default")
      throw std::runtime_error("dflash2_hf: only rope_type 'default' is supported");
  } else if (j.contains("rope_theta")) {
    c.rope_theta = j.at("rope_theta").get<double>();
  }
  const bool use_sw = j.contains("use_sliding_window") && j.at("use_sliding_window").get<bool>();
  c.sliding_window = (use_sw && j.contains("sliding_window") && !j.at("sliding_window").is_null())
                         ? j.at("sliding_window").get<int64_t>()
                         : 0;
  c.sliding.assign(static_cast<size_t>(c.layers), use_sw);
  if (j.contains("layer_types")) {
    const auto& lt = j.at("layer_types");
    if (static_cast<int64_t>(lt.size()) != c.layers)
      throw std::runtime_error("dflash2_hf: layer_types length != num_hidden_layers");
    for (int64_t i = 0; i < c.layers; ++i) {
      const std::string t = lt.at(static_cast<size_t>(i)).get<std::string>();
      if (t != "sliding_attention" && t != "full_attention")
        throw std::runtime_error("dflash2_hf: unknown layer type '" + t + "'");
      c.sliding[static_cast<size_t>(i)] = (t == "sliding_attention");
    }
  }
  if (!j.contains("dflash_config"))
    throw std::runtime_error("dflash2_hf: config.json has no dflash_config");
  const auto& d = j.at("dflash_config");
  c.target_layer_ids = d.at("target_layer_ids").get<std::vector<int64_t>>();
  c.mask_token_id = get_int(d, "mask_token_id");
  if (j.contains("final_logit_softcapping") && !j.at("final_logit_softcapping").is_null())
    c.softcap = j.at("final_logit_softcapping").get<double>();
  for (const nlohmann::json* o : {&j, &d}) {
    if (o->contains("output_multiplier") && o->at("output_multiplier").get<double>() != 1.0)
      throw std::runtime_error("dflash2_hf: output_multiplier != 1 is not supported");
    if (o->contains("input_embedding_scale"))
      c.input_embedding_scale = o->at("input_embedding_scale").get<double>();
  }
  if (c.hidden <= 0 || c.layers <= 0 || c.heads <= 0 || c.kv_heads <= 0 || c.head_dim <= 0 ||
      c.heads % c.kv_heads != 0 || c.block_size < 2 || c.target_layer_ids.empty() || c.vocab <= 0)
    throw std::runtime_error("dflash2_hf: config.json has a non-positive or inconsistent dimension");
  if (c.sliding_window <= 0) {
    // Every layer full: the runtime has no unbounded window; its finite cap is used (dflash_draft.cpp).
    c.sliding_window = 2048;
  }
  if (c.input_embedding_scale <= 0.0) throw std::runtime_error("dflash2_hf: input_embedding_scale <= 0");
  return c;
}

inline Dflash2Metadata MakeDflash2MetadataFromHf(const DflashHfConfig& c, int64_t layer_offset,
                                                 double embed_scale) {
  if (layer_offset != 0 && layer_offset != 1)
    throw std::runtime_error("dflash2_hf: --dflash-target-layer-offset must be 0 or 1");
  Dflash2Metadata m;
  m.hidden = c.hidden;
  m.block_count = c.layers;
  m.ffn = c.ffn;
  m.head_count = c.heads;
  m.head_count_kv = c.kv_heads;
  m.key_length = c.head_dim;
  m.value_length = c.head_dim;
  m.causal = false;
  m.rms_eps = c.rms_eps;
  m.sliding_window = c.sliding_window;
  m.sliding_window_pattern = c.sliding;
  m.rope_freq_base = c.rope_theta;
  m.rope_dimension_sections = {c.head_dim / 2, 0, 0, 0};
  m.n_rot = c.head_dim;
  m.block_size = c.block_size;
  m.conv_kernel_size = 2;
  m.conv_group_size = 16;
  m.selector_rank = 256;
  m.selector_top_k = 16;
  m.target_layers.clear();
  for (int64_t id : c.target_layer_ids) {
    if (id < 0) throw std::runtime_error("dflash2_hf: negative target_layer_id");
    m.target_layers.push_back(id + layer_offset);
  }
  m.context_length = c.context > 0 ? c.context : 131072;
  m.mask_token_id = c.mask_token_id;
  m.vocab_size = c.vocab;
  m.file_type = -1;
  m.logit_softcap = c.softcap;
  m.embed_scale = embed_scale > 0.0 ? embed_scale : c.input_embedding_scale;
  m.variant = "v1_identity";
  m.target_layer_offset = layer_offset;
  return m;
}

// bf16 conv.base for one sublayer: bytes [side][tap][hidden], tap 0 = 1.0, tap 1 = 0.0 (identity).
inline std::vector<uint16_t> IdentityConvBase(int64_t hidden) {
  std::vector<uint16_t> b(static_cast<size_t>(2 * 2 * hidden), 0);
  const uint16_t one = r4dx::core::FloatToBf16(1.0f);
  for (int side = 0; side < 2; ++side)
    for (int64_t c = 0; c < hidden; ++c) b[static_cast<size_t>((side * 2 + 0) * hidden + c)] = one;
  return b;
}

struct DflashHfOptions {
  std::string input_dir, output_path;
  std::string layout = "bf16";  // bf16 | w4a16
  int64_t target_layer_offset = 1;
  double embed_scale = 0.0;  // <= 0: the checkpoint's input_embedding_scale (1.0 when absent)
  int threads = 1;
  QuantOptions quant;
  // The container's `__metadata__.quant` block (src/convert/main.cpp's BuildQuantMetadata()); the
  // caller supplies it so there is one writer of that record. Null: the same default block.
  nlohmann::json quant_metadata;
  bool quiet = false;
};

// Convert. Throws on any shape mismatch or unconsumed tensor, before the output header exists where it can.
inline void ConvertDflashHf(const DflashHfOptions& o) {
  if (o.output_path.empty()) throw std::runtime_error("dflash2_hf: --out is required");
  LayoutSet layouts;
  layouts.bf16 = false;
  if (o.layout == "bf16") layouts.bf16 = true;
  else if (o.layout == "w4a16") layouts.w4a16 = true;
  else throw std::runtime_error("dflash2_hf: --layout must be bf16 or w4a16 (got '" + o.layout + "')");

  const DflashHfConfig c =
      ParseDflashHfConfig(nlohmann::json::parse(ReadTextFileUtf8(o.input_dir + "\\config.json")));
  const Dflash2Metadata meta = MakeDflash2MetadataFromHf(c, o.target_layer_offset, o.embed_scale);
  const int64_t feat = static_cast<int64_t>(c.target_layer_ids.size()) * c.hidden;
  const int64_t q_n = c.heads * c.head_dim, kv_n = c.kv_heads * c.head_dim;
  const int64_t n_groups = c.hidden / meta.conv_group_size;
  const int64_t dyn_n = 2 * meta.conv_kernel_size * n_groups;
  if (c.hidden % meta.conv_group_size != 0)
    throw std::runtime_error("dflash2_hf: hidden is not a multiple of the conv group size 16");

  ShardedModel src(o.input_dir);
  std::set<std::string> consumed;

  ContainerWriter writer;
  std::vector<std::function<void()>> emit;

  auto check_shape = [&](const std::string& hf, std::vector<int64_t> want) {
    if (!src.Has(hf)) throw std::runtime_error("dflash2_hf: checkpoint has no tensor '" + hf + "'");
    const TensorMeta& m = src.Meta(hf);
    if (m.shape != want) {
      auto s = [](const std::vector<int64_t>& v) {
        std::string r = "[";
        for (size_t i = 0; i < v.size(); ++i) r += (i ? "," : "") + std::to_string(v[i]);
        return r + "]";
      };
      throw std::runtime_error("dflash2_hf: tensor '" + hf + "' has shape " + s(m.shape) + ", expected " + s(want));
    }
    consumed.insert(hf);
  };
  auto linear = [&](const std::string& hf, const std::string& base, int64_t N, int64_t K) {
    check_shape(hf, {N, K});
    PlanLinearLayouts(writer, base, static_cast<int>(N), static_cast<int>(K), layouts);
    emit.push_back([&, hf, base, N, K] {
      std::vector<float> w = ReadTensorAsFloat(src, hf);
      EmitLinearLayouts(writer, base, w, static_cast<int>(N), static_cast<int>(K), layouts, o.threads, o.quant);
    });
  };
  auto norm = [&](const std::string& hf, const std::string& name, int64_t n) {
    check_shape(hf, {n});
    writer.Plan(name, {n, 4}, static_cast<uint64_t>(n) * 4);
    emit.push_back([&, hf, name] {
      auto b = EncodeFp32(ReadTensorAsFloat(src, hf));
      writer.WriteTensor(name, b.data(), b.size());
    });
  };
  auto zero_linear = [&](const std::string& base, int64_t N, int64_t K) {
    PlanLinearLayouts(writer, base, static_cast<int>(N), static_cast<int>(K), layouts);
    emit.push_back([&, base, N, K] {
      std::vector<float> w(static_cast<size_t>(N * K), 0.0f);
      EmitLinearLayouts(writer, base, w, static_cast<int>(N), static_cast<int>(K), layouts, o.threads, o.quant);
    });
  };
  auto conv_base = [&](const std::string& name) {
    writer.Plan(name, {2, 2, c.hidden, 2}, static_cast<uint64_t>(2 * 2 * c.hidden) * 2);
    emit.push_back([&, name] {
      const auto b = IdentityConvBase(c.hidden);
      writer.WriteTensor(name, b.data(), b.size() * 2);
    });
  };
  auto zero_bf16_rows = [&](const std::string& name, int64_t rows, int64_t cols) {
    writer.Plan(name, {rows, cols, 2}, static_cast<uint64_t>(rows * cols) * 2);
    emit.push_back([&, name, rows, cols] {
      // WriteTensor writes a whole tensor in one call (134 MB for a real codebook, freed on return).
      std::vector<uint8_t> all(static_cast<size_t>(rows * cols) * 2, 0);
      writer.WriteTensor(name, all.data(), all.size());
    });
  };

  linear("fc.weight", "dflash.fc", c.hidden, feat);
  norm("hidden_norm.weight", "dflash.enc_output_norm", c.hidden);
  norm("norm.weight", "dflash.output_norm", c.hidden);
  zero_linear("dflash.selector.hidden", meta.selector_rank, c.hidden);
  zero_bf16_rows("dflash.selector.predecessor", c.vocab, meta.selector_rank);
  zero_bf16_rows("dflash.selector.successor", c.vocab, meta.selector_rank);
  for (int64_t i = 0; i < c.layers; ++i) {
    const std::string h = "layers." + std::to_string(i) + ".";
    const std::string b = "dflash.layers." + std::to_string(i) + ".";
    norm(h + "input_layernorm.weight", b + "input_layernorm", c.hidden);
    linear(h + "self_attn.q_proj.weight", b + "self_attn.q_proj", q_n, c.hidden);
    linear(h + "self_attn.k_proj.weight", b + "self_attn.k_proj", kv_n, c.hidden);
    linear(h + "self_attn.v_proj.weight", b + "self_attn.v_proj", kv_n, c.hidden);
    linear(h + "self_attn.o_proj.weight", b + "self_attn.o_proj", c.hidden, q_n);
    norm(h + "self_attn.q_norm.weight", b + "self_attn.q_norm", c.head_dim);
    norm(h + "self_attn.k_norm.weight", b + "self_attn.k_norm", c.head_dim);
    conv_base(b + "self_attn.conv.base");
    zero_linear(b + "self_attn.conv.proj", dyn_n, c.hidden);
    norm(h + "post_attention_layernorm.weight", b + "post_attention_layernorm", c.hidden);
    linear(h + "mlp.gate_proj.weight", b + "mlp.gate_proj", c.ffn, c.hidden);
    linear(h + "mlp.up_proj.weight", b + "mlp.up_proj", c.ffn, c.hidden);
    linear(h + "mlp.down_proj.weight", b + "mlp.down_proj", c.hidden, c.ffn);
    conv_base(b + "mlp.conv.base");
    zero_linear(b + "mlp.conv.proj", dyn_n, c.hidden);
  }

  // Coverage audit: the drafter has no embedding / lm_head of its own (it uses the target's), so a
  // checkpoint that ships them is tolerated; any other tensor is a conversion bug waiting to happen.
  for (const std::string& n : src.AllNames()) {
    if (consumed.count(n)) continue;
    if (n == "embed_tokens.weight" || n == "lm_head.weight") {
      if (!o.quiet) std::cerr << "[r4dx-convert --dflash-hf] ignoring '" << n << "' (the target's table is used)\n";
      continue;
    }
    throw std::runtime_error("dflash2_hf: checkpoint tensor '" + n + "' is not consumed by the converter");
  }

  nlohmann::json metadata;
  metadata["r4dx_format_version"] = "1";
  metadata["container_kind"] = "dflash2_draft";
  const size_t slash = o.input_dir.find_last_of("/\\");
  metadata["model_id"] = "dflash-v1-identity:" + (slash == std::string::npos ? o.input_dir : o.input_dir.substr(slash + 1));
  metadata["produced_by"] = "r4dx-convert --dflash-hf (r4dx dev build)";
  metadata["source_hf"] = {{"dir", slash == std::string::npos ? o.input_dir : o.input_dir.substr(slash + 1)},
                           {"architectures", "DFlashDraftModel"}};
  metadata["dflash2"] = BuildDflash2MetadataJson(meta);
  metadata["dflash2"]["layout"] = o.layout;
  if (!o.quant_metadata.is_null()) {
    metadata["quant"] = o.quant_metadata;
  } else {
    metadata["quant"] = {{"w4a16",
                          {{"group", kW4A16Group},
                           {"zero_mode", "free_0_15"},
                           {"nibble_encoding", "offset_binary_xor8"},
                           {"fragment_permutation", "wmma16x16x16_lane16_koff_0_8_1_9_2_10_3_11"}}}};
  }

  writer.FinalizeHeader(o.output_path, metadata);
  if (!o.quiet)
    std::cout << "[r4dx-convert --dflash-hf] planned " << writer.PlannedTensorCount() << " tensors, "
              << writer.PlannedDataBytes() << " data bytes\n";
  for (auto& j : emit) j();
  writer.Finish();
}

}  // namespace r4dx_convert
