#include "hy_v3_gguf.hpp"

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/container_writer.hpp"
#include "r4dx_convert/gguf_reader.hpp"
#include "r4dx_convert/sha256.hpp"

namespace r4dx_convert {

namespace {

struct HyV3GgufMeta {
  int64_t hidden = 0;
  int64_t layers = 0;
  int64_t intermediate = 0;
  int64_t heads = 0;
  int64_t kv_heads = 0;
  int64_t head_dim = 0;
  int64_t max_ctx = 0;
  double rope_theta = 11158840.0;
  double rms_norm_eps = 1e-5;
  int64_t num_experts = 0;
  int64_t num_experts_per_tok = 0;
  int64_t moe_intermediate = 0;
  int64_t shared_intermediate = 0;
  int64_t first_k_dense = 0;
  bool route_norm = true;
  double router_scaling_factor = 1.0;
  int64_t vocab_size = 0;
  int64_t bos_token_id = 120000;
  int64_t eos_token_id = 120025;
  int64_t pad_token_id = 120002;
};

HyV3GgufMeta ReadHyV3Meta(const GgufReader& gguf) {
  const std::string arch = gguf.GetString("general.architecture");
  if (arch != "hy_v3") {
    throw std::runtime_error("r4dx-convert --gguf: expected general.architecture == 'hy_v3', got '" + arch + "'");
  }
  HyV3GgufMeta m;
  m.hidden = gguf.GetInt("hy_v3.embedding_length");
  m.layers = gguf.GetInt("hy_v3.block_count");
  m.intermediate = gguf.GetInt("hy_v3.feed_forward_length");
  m.heads = gguf.GetInt("hy_v3.attention.head_count");
  m.kv_heads = gguf.GetInt("hy_v3.attention.head_count_kv");
  m.head_dim = gguf.HasKey("hy_v3.attention.key_length") ? gguf.GetInt("hy_v3.attention.key_length")
                                                         : (m.hidden / m.heads);
  m.max_ctx = gguf.GetInt("hy_v3.context_length");
  m.rope_theta = gguf.GetFloat("hy_v3.rope.freq_base");
  m.rms_norm_eps = gguf.GetFloat("hy_v3.attention.layer_norm_rms_epsilon");
  m.num_experts = gguf.GetInt("hy_v3.expert_count");
  m.num_experts_per_tok = gguf.GetInt("hy_v3.expert_used_count");
  m.moe_intermediate = gguf.GetInt("hy_v3.expert_feed_forward_length");
  m.shared_intermediate = gguf.HasKey("hy_v3.expert_shared_feed_forward_length")
                              ? gguf.GetInt("hy_v3.expert_shared_feed_forward_length")
                              : m.moe_intermediate;
  m.route_norm = gguf.HasKey("hy_v3.expert_weights_norm") ? gguf.GetBool("hy_v3.expert_weights_norm") : true;
  m.router_scaling_factor =
      gguf.HasKey("hy_v3.expert_weights_scale") ? gguf.GetFloat("hy_v3.expert_weights_scale") : 1.0;

  const auto& emb = gguf.TensorInfo("token_embd.weight");
  if (emb.ne.size() != 2 || emb.ne[0] != m.hidden) {
    throw std::runtime_error("r4dx-convert --gguf: unexpected token_embd.weight shape");
  }
  m.vocab_size = emb.ne[1];
  if (gguf.HasKey("tokenizer.ggml.bos_token_id")) m.bos_token_id = gguf.GetInt("tokenizer.ggml.bos_token_id");
  if (gguf.HasKey("tokenizer.ggml.eos_token_id")) m.eos_token_id = gguf.GetInt("tokenizer.ggml.eos_token_id");
  if (gguf.HasKey("tokenizer.ggml.padding_token_id")) m.pad_token_id = gguf.GetInt("tokenizer.ggml.padding_token_id");

  m.first_k_dense = 0;
  for (int64_t i = 0; i < m.layers; ++i) {
    if (gguf.HasTensor("blk." + std::to_string(i) + ".ffn_gate.weight")) {
      m.first_k_dense = i + 1;
    } else {
      break;
    }
  }
  return m;
}

nlohmann::json BuildHyV3ModelConfig(const HyV3GgufMeta& m) {
  nlohmann::json mc;
  mc["model_type"] = "hy_v3";
  mc["architectures"] = nlohmann::json::array({"HYV3ForCausalLM"});
  mc["hidden_size"] = m.hidden;
  mc["num_hidden_layers"] = m.layers;
  mc["intermediate_size"] = m.intermediate;
  mc["num_attention_heads"] = m.heads;
  mc["num_key_value_heads"] = m.kv_heads;
  mc["head_dim"] = m.head_dim;
  mc["max_position_embeddings"] = m.max_ctx;
  mc["rope_theta"] = m.rope_theta;
  mc["rms_norm_eps"] = m.rms_norm_eps;
  mc["num_experts"] = m.num_experts;
  mc["num_experts_per_tok"] = m.num_experts_per_tok;
  mc["moe_intermediate_size"] = m.moe_intermediate;
  mc["num_shared_experts"] = m.shared_intermediate / m.moe_intermediate;
  mc["first_k_dense_replace"] = m.first_k_dense;
  mc["moe_router_use_sigmoid"] = true;
  mc["moe_router_enable_expert_bias"] = true;
  mc["route_norm"] = m.route_norm;
  mc["router_scaling_factor"] = m.router_scaling_factor;
  mc["qk_norm"] = true;
  mc["tie_word_embeddings"] = false;
  mc["vocab_size"] = m.vocab_size;
  mc["bos_token_id"] = m.bos_token_id;
  mc["eos_token_id"] = m.eos_token_id;
  mc["pad_token_id"] = m.pad_token_id;
  return mc;
}

void Check2D(const GgufReader& gguf, const std::string& name, int64_t K, int64_t N) {
  const auto& info = gguf.TensorInfo(name);
  if (info.ne.size() != 2 || info.ne[0] != K || info.ne[1] != N) {
    throw std::runtime_error("r4dx-convert --gguf: '" + name + "' expected [" + std::to_string(K) + "," +
                             std::to_string(N) + "]");
  }
}

void Check1D(const GgufReader& gguf, const std::string& name, int64_t N) {
  const auto& info = gguf.TensorInfo(name);
  if (info.ne.size() != 1 || info.ne[0] != N) {
    throw std::runtime_error("r4dx-convert --gguf: '" + name + "' expected [" + std::to_string(N) + "]");
  }
}

void Check3D(const GgufReader& gguf, const std::string& name, int64_t K, int64_t N, int64_t E) {
  const auto& info = gguf.TensorInfo(name);
  if (info.ne.size() != 3 || info.ne[0] != K || info.ne[1] != N || info.ne[2] != E) {
    throw std::runtime_error("r4dx-convert --gguf: '" + name + "' expected [" + std::to_string(K) + "," +
                             std::to_string(N) + "," + std::to_string(E) + "]");
  }
}

}  // namespace

void ConvertHyV3Gguf(const HyV3GgufOptions& opts) {
  if (opts.out_path.empty()) throw std::runtime_error("r4dx-convert --gguf requires --out");
  GgufReader gguf(Utf8ToWide(opts.gguf_path));
  const HyV3GgufMeta m = ReadHyV3Meta(gguf);
  const int64_t n_layers = opts.layers >= 0 ? opts.layers : m.layers;
  if (n_layers < 0 || n_layers > m.layers) {
    throw std::runtime_error("r4dx-convert --gguf: --layers " + std::to_string(n_layers) + " out of range");
  }

  QuantOptions qopts;
  qopts.mode = opts.quant_mode;

  const int64_t H = m.hidden;
  const int64_t Q = m.heads * m.head_dim;
  const int64_t KV = m.kv_heads * m.head_dim;
  const int64_t I = m.intermediate;
  const int64_t Im = m.moe_intermediate;
  const int64_t E = m.num_experts;

  ContainerWriter writer;

  auto plan_bf16 = [&](const std::string& name, const std::vector<int64_t>& shape) {
    uint64_t elems = 1;
    for (int64_t d : shape) elems *= static_cast<uint64_t>(d);
    std::vector<int64_t> s = shape;
    s.push_back(2);
    writer.Plan(name, s, elems * 2);
  };
  auto plan_fp32 = [&](const std::string& name, int64_t len) {
    writer.Plan(name, {len, 4}, static_cast<uint64_t>(len) * 4);
  };

  // Plan global tensors
  Check2D(gguf, "token_embd.weight", H, m.vocab_size);
  Check1D(gguf, "output_norm.weight", H);
  Check2D(gguf, "output.weight", H, m.vocab_size);
  plan_bf16("text.embed_tokens", {m.vocab_size, H});
  plan_bf16("text.final_norm", {H});
  PlanLinearLayouts(writer, "lm_head", static_cast<int>(m.vocab_size), static_cast<int>(H), opts.lm_head_layouts);

  // Plan per-layer tensors
  for (int64_t i = 0; i < n_layers; ++i) {
    const std::string g = "blk." + std::to_string(i) + ".";
    const std::string b = "text.layers." + std::to_string(i) + ".";
    Check1D(gguf, g + "attn_norm.weight", H);
    Check1D(gguf, g + "ffn_norm.weight", H);
    Check1D(gguf, g + "attn_q_norm.weight", m.head_dim);
    Check1D(gguf, g + "attn_k_norm.weight", m.head_dim);
    Check2D(gguf, g + "attn_q.weight", H, Q);
    Check2D(gguf, g + "attn_k.weight", H, KV);
    Check2D(gguf, g + "attn_v.weight", H, KV);
    Check2D(gguf, g + "attn_output.weight", Q, H);

    plan_bf16(b + "input_norm", {H});
    plan_bf16(b + "post_attn_norm", {H});
    plan_bf16(b + "attn.q_norm", {m.head_dim});
    plan_bf16(b + "attn.k_norm", {m.head_dim});
    PlanLinearLayouts(writer, b + "attn.q", static_cast<int>(Q), static_cast<int>(H), opts.body_layouts);
    PlanLinearLayouts(writer, b + "attn.k", static_cast<int>(KV), static_cast<int>(H), opts.body_layouts);
    PlanLinearLayouts(writer, b + "attn.v", static_cast<int>(KV), static_cast<int>(H), opts.body_layouts);
    PlanLinearLayouts(writer, b + "attn.o", static_cast<int>(H), static_cast<int>(Q), opts.body_layouts);

    if (i < m.first_k_dense) {
      Check2D(gguf, g + "ffn_gate.weight", H, I);
      Check2D(gguf, g + "ffn_up.weight", H, I);
      Check2D(gguf, g + "ffn_down.weight", I, H);
      PlanLinearLayouts(writer, b + "mlp.gate_up", static_cast<int>(2 * I), static_cast<int>(H), opts.body_layouts);
      PlanLinearLayouts(writer, b + "mlp.down", static_cast<int>(H), static_cast<int>(I), opts.body_layouts);
    } else {
      Check2D(gguf, g + "ffn_gate_inp.weight", H, E);
      Check1D(gguf, g + "exp_probs_b", E);
      Check2D(gguf, g + "ffn_gate_shexp.weight", H, Im);
      Check2D(gguf, g + "ffn_up_shexp.weight", H, Im);
      Check2D(gguf, g + "ffn_down_shexp.weight", Im, H);
      Check3D(gguf, g + "ffn_gate_exps.weight", H, Im, E);
      Check3D(gguf, g + "ffn_up_exps.weight", H, Im, E);
      Check3D(gguf, g + "ffn_down_exps.weight", Im, H, E);

      plan_bf16(b + "moe.router", {E, H});
      plan_fp32(b + "moe.expert_bias", E);
      PlanLinearLayouts(writer, b + "moe.shared_gate_up", static_cast<int>(2 * Im), static_cast<int>(H),
                        opts.body_layouts);
      PlanLinearLayouts(writer, b + "moe.shared_down", static_cast<int>(H), static_cast<int>(Im), opts.body_layouts);
      for (int64_t e = 0; e < E; ++e) {
        const std::string eb = b + "moe.experts." + std::to_string(e) + ".";
        PlanLinearLayouts(writer, eb + "gate_up", static_cast<int>(2 * Im), static_cast<int>(H), opts.body_layouts);
        PlanLinearLayouts(writer, eb + "down", static_cast<int>(H), static_cast<int>(Im), opts.body_layouts);
      }
    }
  }

  nlohmann::json metadata;
  metadata["r4dx_format_version"] = "1";
  metadata["model_arch"] = "hy_v3";
  metadata["model_id"] = "orcarouter/Hy-MT3-30B-A3B-abliterated-GGUF";
  metadata["produced_by"] = "r4dx-convert --gguf (r4dx dev build)";
  metadata["source_gguf"] = {
      {"filename", opts.gguf_path.substr(opts.gguf_path.find_last_of("/\\") + 1)},
  };
  metadata["model_config"] = BuildHyV3ModelConfig(m);
  metadata["quant"] = opts.quant_metadata;

  const auto t0 = std::chrono::steady_clock::now();
  writer.FinalizeHeader(opts.out_path, metadata);
  std::cout << "[r4dx-convert --gguf] planned " << writer.PlannedTensorCount() << " tensors, "
            << writer.PlannedDataBytes() << " data bytes\n";

  auto emit_bf16 = [&](const std::string& gguf_name, const std::string& cname) {
    const std::vector<uint16_t> bf = gguf.DequantToBf16(gguf_name);
    writer.WriteTensor(cname, bf.data(), bf.size() * sizeof(uint16_t));
  };
  auto emit_fp32 = [&](const std::string& gguf_name, const std::string& cname) {
    const std::vector<float> f = gguf.DequantToF32(gguf_name);
    writer.WriteTensor(cname, f.data(), f.size() * sizeof(float));
  };
  auto emit_linear = [&](const std::string& gguf_name, const std::string& base, int N, int K,
                         const LayoutSet& ls) {
    const std::vector<float> w = gguf.DequantToF32(gguf_name);
    EmitLinearLayouts(writer, base, w, N, K, ls, opts.threads, qopts);
  };
  auto emit_fused_gate_up = [&](const std::string& g_name, const std::string& u_name, const std::string& base,
                                int I_dim, int K_dim, const LayoutSet& ls) {
    std::vector<float> g_w = gguf.DequantToF32(g_name);
    std::vector<float> u_w = gguf.DequantToF32(u_name);
    g_w.insert(g_w.end(), u_w.begin(), u_w.end());
    EmitLinearLayouts(writer, base, g_w, 2 * I_dim, K_dim, ls, opts.threads, qopts);
  };

  emit_bf16("token_embd.weight", "text.embed_tokens");
  emit_bf16("output_norm.weight", "text.final_norm");
  emit_linear("output.weight", "lm_head", static_cast<int>(m.vocab_size), static_cast<int>(H), opts.lm_head_layouts);

  for (int64_t i = 0; i < n_layers; ++i) {
    const std::string g = "blk." + std::to_string(i) + ".";
    const std::string b = "text.layers." + std::to_string(i) + ".";
    emit_bf16(g + "attn_norm.weight", b + "input_norm");
    emit_bf16(g + "ffn_norm.weight", b + "post_attn_norm");
    emit_bf16(g + "attn_q_norm.weight", b + "attn.q_norm");
    emit_bf16(g + "attn_k_norm.weight", b + "attn.k_norm");
    emit_linear(g + "attn_q.weight", b + "attn.q", static_cast<int>(Q), static_cast<int>(H), opts.body_layouts);
    emit_linear(g + "attn_k.weight", b + "attn.k", static_cast<int>(KV), static_cast<int>(H), opts.body_layouts);
    emit_linear(g + "attn_v.weight", b + "attn.v", static_cast<int>(KV), static_cast<int>(H), opts.body_layouts);
    emit_linear(g + "attn_output.weight", b + "attn.o", static_cast<int>(H), static_cast<int>(Q), opts.body_layouts);

    if (i < m.first_k_dense) {
      emit_fused_gate_up(g + "ffn_gate.weight", g + "ffn_up.weight", b + "mlp.gate_up", static_cast<int>(I),
                         static_cast<int>(H), opts.body_layouts);
      emit_linear(g + "ffn_down.weight", b + "mlp.down", static_cast<int>(H), static_cast<int>(I), opts.body_layouts);
    } else {
      emit_bf16(g + "ffn_gate_inp.weight", b + "moe.router");
      emit_fp32(g + "exp_probs_b", b + "moe.expert_bias");
      emit_fused_gate_up(g + "ffn_gate_shexp.weight", g + "ffn_up_shexp.weight", b + "moe.shared_gate_up",
                         static_cast<int>(Im), static_cast<int>(H), opts.body_layouts);
      emit_linear(g + "ffn_down_shexp.weight", b + "moe.shared_down", static_cast<int>(H), static_cast<int>(Im),
                  opts.body_layouts);
      const int64_t gu_elems = H * Im;
      const int64_t dn_elems = Im * H;
      for (int64_t e = 0; e < E; ++e) {
        const std::string eb = b + "moe.experts." + std::to_string(e) + ".";
        std::vector<float> eg = gguf.DequantSliceToF32(g + "ffn_gate_exps.weight", e, gu_elems);
        std::vector<float> eu = gguf.DequantSliceToF32(g + "ffn_up_exps.weight", e, gu_elems);
        eg.insert(eg.end(), eu.begin(), eu.end());
        EmitLinearLayouts(writer, eb + "gate_up", eg, static_cast<int>(2 * Im), static_cast<int>(H), opts.body_layouts,
                          opts.threads, qopts);
        std::vector<float> ed = gguf.DequantSliceToF32(g + "ffn_down_exps.weight", e, dn_elems);
        EmitLinearLayouts(writer, eb + "down", ed, static_cast<int>(H), static_cast<int>(Im), opts.body_layouts,
                          opts.threads, qopts);
      }
    }
    if ((i + 1) % 4 == 0 || i + 1 == n_layers) {
      std::cout << "[r4dx-convert --gguf] layer " << (i + 1) << "/" << n_layers << "\n";
    }
  }

  writer.Finish();
  const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::cout << "[r4dx-convert --gguf] wrote " << opts.out_path << " in " << secs << " s\n";
}

}  // namespace r4dx_convert
