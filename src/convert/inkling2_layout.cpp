#include "inkling2_layout.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "r4dx/core/dtype.hpp"

namespace r4dx_convert::inkling2 {

namespace {

const std::string kRoot = "model.llm.";

std::string ShapeStr(const std::vector<int64_t>& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) out += (i ? "," : "") + std::to_string(s[i]);
  return out + "]";
}

[[noreturn]] void Fail(const std::string& what) { throw std::runtime_error("r4dx-convert (inkling2_loop): " + what); }

}  // namespace

TextShape ParseConfig(const nlohmann::json& cfg) {
  const std::string who = "config.json ";
  if (!cfg.is_object()) Fail(who + "is not an object");
  if (cfg.value("model_type", std::string()) != kModelArch) Fail(who + "model_type is not '" + kModelArch + "'");
  if (!cfg.contains("text_config") || !cfg.at("text_config").is_object()) Fail(who + "has no text_config object");
  const nlohmann::json& t = cfg.at("text_config");
  auto req = [&](const nlohmann::json& o, const char* key) -> const nlohmann::json& {
    if (!o.contains(key)) Fail(who + "is missing '" + key + "'");
    return o.at(key);
  };
  auto req_int = [&](const nlohmann::json& o, const char* key, int64_t min_value) {
    const nlohmann::json& v = req(o, key);
    if (!v.is_number_integer() || v.get<int64_t>() < min_value)
      Fail(who + "'" + std::string(key) + "' must be an integer >= " + std::to_string(min_value));
    return v.get<int64_t>();
  };
  TextShape s;
  s.hidden = req_int(t, "hidden_size", 1);
  s.layers = req_int(t, "num_hidden_layers", 1);
  s.heads = req_int(t, "num_attention_heads", 1);
  s.kv_heads = req_int(t, "num_key_value_heads", 1);
  s.head_dim = req_int(t, "head_dim", 1);
  if (req_int(t, "swa_num_attention_heads", 1) != s.heads || req_int(t, "swa_num_key_value_heads", 1) != s.kv_heads ||
      req_int(t, "swa_head_dim", 1) != s.head_dim)
    Fail(who + "swa_* attention geometry differs from the full-attention one: the layout assumes they are equal");
  s.d_rel = req_int(t, "d_rel", 1);
  s.rel_extent = req_int(t, "rel_extent", 1);
  s.conv_kernel = req_int(t, "sconv_kernel_size", 1);
  s.vocab = req_int(t, "vocab_size", 1);
  s.unpadded_vocab = t.contains("unpadded_vocab_size") ? req_int(t, "unpadded_vocab_size", 1) : 200058;
  if (s.unpadded_vocab > s.vocab) Fail(who + "unpadded_vocab_size exceeds vocab_size");
  s.experts = req_int(t, "n_routed_experts", 1);
  s.moe_intermediate = req_int(t, "intermediate_size", 1);
  if (t.contains("n_shared_experts") && t.at("n_shared_experts").get<int64_t>() != 0)
    Fail(who + "n_shared_experts != 0: the layout has no shared-expert tensors");
  s.attn_res_block = t.contains("n_layers_per_attn_res_block") ? req_int(t, "n_layers_per_attn_res_block", 1) : 6;

  s.loop_start = 6;
  s.loop_end = 18;
  s.loop_passes = 3;
  s.gate_rank = 64;
  if (t.contains("loop_config") && t.at("loop_config").is_object()) {
    const nlohmann::json& l = t.at("loop_config");
    if (l.value("is_varloop", false)) Fail(who + "loop_config.is_varloop is true: not supported");
    s.loop_start = l.value("core_start", s.loop_start);
    s.loop_end = l.value("core_end", s.loop_end);
    s.loop_passes = l.value("num_passes", s.loop_passes);
    s.gate_rank = l.value("gate_rank", s.gate_rank);
  }
  // HF's Inkling2LoopTextConfig.validate_architecture.
  if (s.loop_passes < 1) Fail(who + "loop num_passes must be >= 1");
  if (s.loop_start <= 0 || s.loop_start % s.attn_res_block != 0 || s.loop_end % s.attn_res_block != 0 ||
      s.loop_end <= s.loop_start || s.loop_end > s.layers)
    Fail(who + "loop bounds [" + std::to_string(s.loop_start) + ", " + std::to_string(s.loop_end) +
         ") must be multiples of n_layers_per_attn_res_block (" + std::to_string(s.attn_res_block) +
         "), non-empty and inside the layer count");

  // Layer kinds.
  s.full.assign(static_cast<size_t>(s.layers), false);
  bool have_kinds = false;
  if (t.contains("layer_types") && t.at("layer_types").is_array()) {
    const auto& lt = t.at("layer_types");
    if (static_cast<int64_t>(lt.size()) != s.layers) Fail(who + "layer_types length != num_hidden_layers");
    for (int64_t i = 0; i < s.layers; ++i) {
      const std::string k = lt[static_cast<size_t>(i)].get<std::string>();
      if (k != "hybrid" && k != "hybrid_sliding") Fail(who + "layer type '" + k + "' (want hybrid | hybrid_sliding)");
      s.full[static_cast<size_t>(i)] = (k == "hybrid");
    }
    have_kinds = true;
  }
  if (t.contains("local_layer_ids") && t.at("local_layer_ids").is_array()) {
    std::vector<bool> full(static_cast<size_t>(s.layers), true);
    for (const auto& id : t.at("local_layer_ids")) {
      const int64_t i = id.get<int64_t>();
      if (i < 0 || i >= s.layers) Fail(who + "local_layer_ids entry " + std::to_string(i) + " is out of range");
      full[static_cast<size_t>(i)] = false;
    }
    if (have_kinds && full != s.full) Fail(who + "layer_types and local_layer_ids disagree");
    s.full = full;
    have_kinds = true;
  }
  if (!have_kinds)
    for (int64_t i = 0; i < s.layers; ++i) s.full[static_cast<size_t>(i)] = ((i + 1) % 6 == 0);
  return s;
}

void MergeMessageQueries(const float* queries, const float* kv_queries, int64_t loop_layers, int64_t hidden, float* out) {
  const size_t row = static_cast<size_t>(hidden) * sizeof(float);
  for (int64_t j = 0; j < loop_layers; ++j) {
    float* o = out + static_cast<size_t>(j) * 4 * static_cast<size_t>(hidden);
    std::memcpy(o + 0 * hidden, queries + (2 * j) * hidden, row);       // query
    std::memcpy(o + 1 * hidden, kv_queries + (2 * j) * hidden, row);    // key
    std::memcpy(o + 2 * hidden, kv_queries + (2 * j + 1) * hidden, row);  // value
    std::memcpy(o + 3 * hidden, queries + (2 * j + 1) * hidden, row);   // mlp
  }
}

namespace {

// One (layer, w13 | w2) expert group: validates names / dtypes / shapes against the config and returns the source
// format. `rows`, `cols` are the logical decoded [rows, cols] of one expert.
struct ExpertGroup {
  ExpertFormat format;
  std::string weight, scale, amax, denom;  // amax / denom empty for MXFP8
  int64_t rows, cols;
};

ExpertGroup CheckExpertGroup(Kit& kit, std::set<std::string>& consumed, const std::string& hf_layer, const char* which,
                             int64_t experts, int64_t rows, int64_t cols) {
  const std::string pre = hf_layer + "mlp.experts." + which + "_weight";
  ExpertGroup g;
  g.weight = pre;
  g.scale = pre + "_scale";
  g.amax = pre + "_amax";
  g.denom = pre + "_outer_scale_denominator";
  g.rows = rows;
  g.cols = cols;
  auto need = [&](const std::string& n, const char* dt, std::vector<int64_t> want) {
    if (!kit.has(n)) Fail("checkpoint has no tensor '" + n + "'");
    const std::string got_dt = kit.dtype(n);
    if (got_dt != dt) Fail("'" + n + "' has dtype " + got_dt + ", want " + dt);
    const std::vector<int64_t> got = kit.shape(n);
    if (got != want) Fail("'" + n + "' has shape " + ShapeStr(got) + ", config says " + ShapeStr(want));
    consumed.insert(n);
  };
  const bool nvfp4 = kit.has(g.amax);
  if (nvfp4) {
    g.format = ExpertFormat::kNvfp4;
    if (cols % kNvfp4Group != 0) Fail("'" + pre + "': K=" + std::to_string(cols) + " is not a multiple of 16");
    need(g.weight, "U8", {experts, rows, cols / 2});
    need(g.scale, "F8_E4M3", {experts, rows, cols / kNvfp4Group});
    need(g.amax, "F32", {experts});
    if (!kit.has(g.denom)) Fail("checkpoint has no tensor '" + g.denom + "' (NVFP4 needs it next to amax)");
    if (kit.dtype(g.denom) != "F32" || kit.shape(g.denom) != std::vector<int64_t>({1}))
      Fail("'" + g.denom + "' must be F32 [1]");
    consumed.insert(g.denom);
  } else {
    g.format = ExpertFormat::kMxfp8;
    if (kit.has(g.denom)) Fail("'" + g.denom + "' is present without '" + g.amax + "'");
    if (cols % kMxfp8Group != 0) Fail("'" + pre + "': K=" + std::to_string(cols) + " is not a multiple of 32");
    need(g.weight, "F8_E4M3", {experts, rows, cols});
    need(g.scale, "U8", {experts, rows, cols / kMxfp8Group});
  }
  return g;
}

// The checkpoint-facing view, built at emit time (the bytes are read lazily).
using DataFn = std::function<const uint8_t*(const std::string&)>;

ExpertTensors OpenView(const DataFn& data, const ExpertGroup& g, int64_t experts) {
  ExpertTensors t;
  t.format = g.format;
  t.experts = experts;
  t.rows = g.rows;
  t.cols = g.cols;
  t.weight = data(g.weight);
  t.scale = data(g.scale);
  if (g.format == ExpertFormat::kNvfp4) {
    t.amax = data(g.amax);
    t.denominator = ReadF32(data(g.denom));
    if (!(t.denominator > 0.0f) || !std::isfinite(t.denominator))
      Fail("'" + g.denom + "' is not a positive finite number");
  }
  return t;
}

LinearSource ExpertSource(const DataFn& data, const ExpertGroup& g, int64_t experts, int64_t e, bool interleaved) {
  LinearSource src;
  src.N = g.rows;
  src.K = g.cols;
  src.read = [data, g, experts, e, interleaved](float* out) {
    const ExpertTensors t = OpenView(data, g, experts);
    if (!interleaved) {
      DecodeExpert(t, e, out);
      return;
    }
    std::vector<float> tmp(static_cast<size_t>(t.rows * t.cols));
    DecodeExpert(t, e, tmp.data());
    DeinterleaveGateUp(tmp.data(), out, t.rows, t.cols);
  };
  if (g.format == ExpertFormat::kMxfp8) {
    src.raw_mxfp8 = [data, g, experts, e, interleaved](uint8_t* codes, uint8_t* scales) {
      const ExpertTensors t = OpenView(data, g, experts);
      if (!interleaved) {
        RawMxfp8Expert(t, e, codes, scales);
        return;
      }
      std::vector<uint8_t> c(static_cast<size_t>(t.rows * t.cols)), s(static_cast<size_t>(t.rows * t.GroupsPerRow()));
      RawMxfp8Expert(t, e, c.data(), s.data());
      DeinterleaveGateUp(c.data(), codes, t.rows, t.cols);
      DeinterleaveGateUp(s.data(), scales, t.rows, t.GroupsPerRow());
    };
  }
  return src;
}

}  // namespace

std::set<std::string> AddInkling2TextStack(Kit& kit, const TextShape& s) {
  std::set<std::string> consumed;
  // Existence, dtype and shape against the config, before any job is registered: a mis-guessed name or a checkpoint
  // that is not the config's is a one-second error, not a mid-conversion one.
  auto expect = [&](const std::string& name, const char* dt, std::vector<int64_t> want) {
    if (!kit.has(name)) Fail("checkpoint has no tensor '" + name + "'");
    const std::string got_dt = kit.dtype(name);
    if (got_dt != dt) Fail("'" + name + "' has dtype " + got_dt + ", want " + dt);
    const std::vector<int64_t> got = kit.shape(name);
    if (got != want) Fail("'" + name + "' has shape " + ShapeStr(got) + ", config says " + ShapeStr(want));
    consumed.insert(name);
  };
  auto bf16 = [&](const std::string& hf, const std::string& name, std::vector<int64_t> want) {
    expect(hf, "BF16", std::move(want));
    kit.add_bf16(hf, name);
  };
  // A depthwise conv [C, 1, K] -> bf16 [C, K]: a squeeze of the singleton group axis, the bytes are unchanged.
  auto conv = [&](const std::string& hf, const std::string& name, int64_t channels) {
    expect(hf, "BF16", {channels, 1, s.conv_kernel});
    kit.add_bf16_shaped(hf, name, {channels, s.conv_kernel});
  };
  auto fp32 = [&](const std::string& hf, const std::string& name, std::vector<int64_t> want) {
    expect(hf, "F32", std::move(want));
    kit.add_fp32_widen(hf, name);
  };
  auto linear = [&](const std::string& hf, const std::string& base, int64_t n, int64_t k) {
    expect(hf, "BF16", {n, k});
    kit.add_linear({hf}, base, /*head=*/false);
  };

  const int64_t H = s.hidden, qd = s.heads * s.head_dim, kvd = s.kv_heads * s.head_dim;

  for (int64_t i = 0; i < s.layers; ++i) {
    const std::string hf = kRoot + "layers." + std::to_string(i) + ".";
    const std::string base = "text.layers." + std::to_string(i) + ".";
    const bool full = s.IsFull(i);

    // Attention-residual queries. attn_queries.{j} belongs to layer j + 1 (core_model_loading Inkling2IndexOffsetWeightRename
    // offset 1): layer 0 has none; a full layer mixes three (q, k, v).
    if (i >= 1)
      fp32(kRoot + "attn_res.attn_queries." + std::to_string(i - 1), base + "attn_query",
           full ? std::vector<int64_t>{3, H} : std::vector<int64_t>{H});
    fp32(kRoot + "attn_res.mlp_queries." + std::to_string(i), base + "mlp_query", {H});

    linear(hf + "attn.wq_du.weight", base + "attn.q", qd, H);
    linear(hf + "attn.wk_dv.weight", base + "attn.k", kvd, H);
    linear(hf + "attn.wv_dv.weight", base + "attn.v", kvd, H);
    linear(hf + "attn.gate_attn.weight", base + "attn.gate", qd, H);
    linear(hf + "attn.wo_ud.weight", base + "attn.o", H, qd);
    linear(hf + "attn.wr_du.weight", base + "attn.rel", s.heads * s.d_rel, H);
    bf16(hf + "attn.attn_sink_proj.weight", base + "attn.sink", {s.heads, H});
    bf16(hf + "attn.q_l2_clip.scale", base + "attn.q_scale", {s.heads, s.head_dim});
    bf16(hf + "attn.rel_l2_clip.scale", base + "attn.rel_scale", {s.heads, s.d_rel});
    bf16(hf + "attn.rel_logits_proj.proj", base + "attn.rel_proj", {s.d_rel, s.rel_extent});

    if (full) {
      for (const char* kv : {"k", "v"}) {
        bf16(hf + "attn." + kv + "_sconv.static_w", base + "attn." + kv + "_conv_static", {s.conv_kernel, kvd});
        bf16(hf + "attn." + kv + "_sconv.weight_proj", base + "attn." + kv + "_conv_proj", {s.kv_heads * s.conv_kernel, H});
      }
      for (const char* n : {"q", "k", "v"}) bf16(hf + n + "_input_norm.weight", base + n + "_norm", {H});
    } else {
      for (const char* kv : {"k", "v"}) conv(hf + "attn." + kv + "_sconv.weight", base + "attn." + kv + "_conv", kvd);
      bf16(hf + "attn_norm.weight", base + "attn_norm", {H});
    }
    bf16(hf + "mlp_norm.weight", base + "mlp_norm", {H});
    conv(hf + "attn_sconv.weight", base + "attn_conv", H);
    conv(hf + "mlp_sconv.weight", base + "mlp_conv", H);

    if (s.IsLoop(i)) {
      const char* hf_name[4] = {"attn", "k", "v", "mlp"};
      const char* ct_name[4] = {"q", "k", "v", "mlp"};
      for (int c = 0; c < 4; ++c) {
        const std::string h = hf + "controllers." + hf_name[c] + ".";
        const std::string b = base + "msg." + ct_name[c] + ".";
        bf16(h + "down_weight", b + "down", {s.gate_rank, 2 * H});
        bf16(h + "forget_weight", b + "forget", {H, s.gate_rank});
        bf16(h + "gain_weight", b + "gain", {H, s.gate_rank});
      }
    }

    // MoE: router (bf16, never quantized), selection bias (f32), the per-expert linears.
    const int64_t I = s.moe_intermediate;
    bf16(hf + "mlp.gate.weight", base + "moe.router", {s.experts, H});
    fp32(hf + "mlp.gate.bias", base + "moe.bias", {s.experts});
    const ExpertGroup w13 = CheckExpertGroup(kit, consumed, hf, "w13", s.experts, 2 * I, H);
    const ExpertGroup w2 = CheckExpertGroup(kit, consumed, hf, "w2", s.experts, H, I);
    if (kit.keep_nvfp4_experts && kit.add_raw && w13.format == ExpertFormat::kNvfp4 &&
        w2.format == ExpertFormat::kNvfp4) {
      const std::string nb = base + "moe.nvfp4.";
      kit.add_raw(w13.weight, nb + "w13_weight", {s.experts, 2 * I, H / 2}, 1);
      kit.add_raw(w13.scale, nb + "w13_weight_scale", {s.experts, 2 * I, H / kNvfp4Group}, 1);
      kit.add_raw(w13.amax, nb + "w13_weight_amax", {s.experts}, 4);
      kit.add_raw(w13.denom, nb + "w13_weight_outer_scale_denominator", {1}, 4);
      kit.add_raw(w2.weight, nb + "w2_weight", {s.experts, H, I / 2}, 1);
      kit.add_raw(w2.scale, nb + "w2_weight_scale", {s.experts, H, I / kNvfp4Group}, 1);
      kit.add_raw(w2.amax, nb + "w2_weight_amax", {s.experts}, 4);
      kit.add_raw(w2.denom, nb + "w2_weight_outer_scale_denominator", {1}, 4);
    } else {
      for (int64_t e = 0; e < s.experts; ++e) {
        const std::string eb = base + "moe.experts." + std::to_string(e) + ".";
        kit.add_source_linear(eb + "gate_up", false, ExpertSource(kit.data, w13, s.experts, e, /*interleaved=*/true));
        kit.add_source_linear(eb + "down", false, ExpertSource(kit.data, w2, s.experts, e, /*interleaved=*/false));
      }
    }
  }

  // Outside the layer stack.
  bf16(kRoot + "embed.weight", "text.embed_tokens", {s.vocab, H});
  bf16(kRoot + "embed_norm.weight", "text.embed_norm", {H});
  bf16(kRoot + "norm.weight", "text.final_norm", {H});
  fp32(kRoot + "attn_res.output_query", "text.final_query", {H});
  // The cross-pass queries: [q, mlp] and [k, v] pairs of each loop layer (message queries).
  {
    const int64_t L = s.LoopLayers();
    const std::string q = kRoot + "middle_cycle_cross_pass_queries", kv = kRoot + "middle_cycle_cross_pass_kv_queries";
    expect(q, "F32", {2 * L, H});
    expect(kv, "F32", {2 * L, H});
    kit.add_msg_queries(q, kv, "text.msg_queries", L, H);
  }

  // unembed: only the first unpadded_vocab rows, zero-padded to a multiple of 16 rows in the container.
  const std::string un = kRoot + "unembed.weight";
  expect(un, "BF16", {s.vocab, H});
  {
    LinearSource src;
    src.N = s.HeadRowsStored();
    src.K = H;
    const int64_t used = s.unpadded_vocab, stored = src.N;
    src.read = [data = kit.data, un, used, stored, H](float* out) {
      const uint16_t* w = reinterpret_cast<const uint16_t*>(data(un));
      for (int64_t i = 0; i < used * H; ++i) out[i] = r4dx::core::Bf16ToFloat(w[i]);
      std::fill(out + used * H, out + stored * H, 0.0f);
    };
    kit.add_source_linear("lm_head", /*head=*/true, std::move(src));
  }
  return consumed;
}

bool IsAllowListed(const std::string& name) { return name.compare(0, 13, "model.visual.") == 0; }

std::vector<std::string> UnconsumedTensors(const std::vector<std::string>& all_names, const std::set<std::string>& consumed) {
  std::vector<std::string> out;
  for (const std::string& name : all_names) {
    if (consumed.count(name) != 0 || IsAllowListed(name)) continue;
    out.push_back(name);
  }
  std::sort(out.begin(), out.end());
  return out;
}

}  // namespace r4dx_convert::inkling2
