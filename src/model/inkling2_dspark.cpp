#include "inkling2_dspark.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <unordered_map>

#include "linear.h"
#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/inkling2_kernels.h"
#include "r4dx/kernels/kernels.h"

namespace r4dx::model {

namespace {

inline int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

struct StEntry {
  std::string dtype;
  std::vector<int64_t> shape;
  uint64_t begin = 0, end = 0;
};

class SafetensorsReader {
 public:
  explicit SafetensorsReader(const std::string& path) : f_(path, std::ios::binary) {
    if (!f_) throw std::runtime_error("Inkling2DsparkDraft: cannot open " + path);
    uint64_t hdr_len = 0;
    f_.read(reinterpret_cast<char*>(&hdr_len), 8);
    if (!f_ || hdr_len == 0 || hdr_len > (64ull << 20))
      throw std::runtime_error("Inkling2DsparkDraft: bad safetensors header length in " + path);
    std::string hdr_str(static_cast<size_t>(hdr_len), '\0');
    f_.read(hdr_str.data(), static_cast<std::streamsize>(hdr_len));
    if (!f_) throw std::runtime_error("Inkling2DsparkDraft: truncated safetensors header in " + path);
    data_base_ = 8 + hdr_len;
    const auto j = nlohmann::json::parse(hdr_str);
    for (auto it = j.begin(); it != j.end(); ++it) {
      if (it.key() == "__metadata__") continue;
      StEntry e;
      e.dtype = it.value().at("dtype").get<std::string>();
      e.shape = it.value().at("shape").get<std::vector<int64_t>>();
      const auto& offs = it.value().at("data_offsets");
      e.begin = offs.at(0).get<uint64_t>();
      e.end = offs.at(1).get<uint64_t>();
      entries_.emplace(it.key(), std::move(e));
    }
  }

  std::vector<uint16_t> ReadBf16(const std::string& name, const std::vector<int64_t>& expect_shape) {
    auto it = entries_.find(name);
    if (it == entries_.end()) throw std::runtime_error("Inkling2DsparkDraft: missing tensor " + name);
    const StEntry& e = it->second;
    if (e.dtype != "BF16") throw std::runtime_error("Inkling2DsparkDraft: " + name + " expected BF16, got " + e.dtype);
    if (e.shape != expect_shape) throw std::runtime_error("Inkling2DsparkDraft: unexpected shape for " + name);
    int64_t elems = 1;
    for (int64_t d : e.shape) elems *= d;
    if (e.end - e.begin != static_cast<uint64_t>(elems) * 2)
      throw std::runtime_error("Inkling2DsparkDraft: byte length mismatch for " + name);
    std::vector<uint16_t> buf(static_cast<size_t>(elems));
    f_.seekg(static_cast<std::streamoff>(data_base_ + e.begin), std::ios::beg);
    f_.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(buf.size() * 2));
    if (!f_) throw std::runtime_error("Inkling2DsparkDraft: read failed for " + name);
    return buf;
  }

  core::DeviceBuffer<uint16_t> ReadDevBf16(const std::string& name, const std::vector<int64_t>& expect_shape) {
    const std::vector<uint16_t> h = ReadBf16(name, expect_shape);
    core::DeviceBuffer<uint16_t> d(h.size());
    d.CopyFromHost(h);
    return d;
  }

  QuantLinear ReadBf16Linear(const std::string& name, int64_t N, int64_t K) {
    QuantLinear q;
    q.layout = Layout::kBf16;
    q.N = N;
    q.K = K;
    q.bf16_w = ReadDevBf16(name, {N, K});
    return q;
  }

 private:
  std::ifstream f_;
  uint64_t data_base_ = 0;
  std::unordered_map<std::string, StEntry> entries_;
};

}  // namespace

Inkling2DsparkDraft Inkling2DsparkDraft::Load(const std::string& safetensors_path, int64_t max_ctx) {
  SafetensorsReader r(safetensors_path);
  Inkling2DsparkDraft d;
  const Inkling2DsparkConfig& c = d.cfg_;
  const int64_t H = c.hidden_size, Q = c.num_attention_heads * c.head_dim, KV = c.num_key_value_heads * c.head_dim;
  const int64_t I = c.intermediate_size, R = c.markov_rank;

  // fuse_weight [4, 4] -> softmax per row in fp32
  {
    const std::vector<uint16_t> fw = r.ReadBf16("fuse_weight", {4, 4});
    for (int row = 0; row < 4; ++row) {
      float v[4], m = -1e30f;
      for (int j = 0; j < 4; ++j) {
        v[j] = core::Bf16ToFloat(fw[static_cast<size_t>(row * 4 + j)]);
        m = std::max(m, v[j]);
      }
      float s = 0.0f;
      for (int j = 0; j < 4; ++j) {
        v[j] = std::exp(v[j] - m);
        s += v[j];
      }
      for (int j = 0; j < 4; ++j) d.fuse_alpha_[static_cast<size_t>(row)][static_cast<size_t>(j)] = v[j] / s;
    }
  }

  d.mask_embed_ = r.ReadDevBf16("mask_embed", {H});
  d.norm_ = r.ReadDevBf16("norm.weight", {H});

  d.layers_.resize(static_cast<size_t>(c.num_layers));
  for (int64_t l = 0; l < c.num_layers; ++l) {
    Layer& L = d.layers_[static_cast<size_t>(l)];
    const std::string p = "layers." + std::to_string(l) + ".";
    L.dflare_norm = r.ReadDevBf16(p + "dflare_norm.weight", {H});
    L.input_layernorm = r.ReadDevBf16(p + "input_layernorm.weight", {H});
    L.post_attention_layernorm = r.ReadDevBf16(p + "post_attention_layernorm.weight", {H});
    L.q_norm = r.ReadDevBf16(p + "self_attn.q_norm.weight", {c.head_dim});
    L.k_norm = r.ReadDevBf16(p + "self_attn.k_norm.weight", {c.head_dim});
    L.k_norm_ctx = r.ReadDevBf16(p + "self_attn.k_norm_ctx.weight", {c.head_dim});
    L.q_proj = r.ReadBf16Linear(p + "self_attn.q_proj.weight", Q, H);
    L.k_proj = r.ReadBf16Linear(p + "self_attn.k_proj.weight", KV, H);
    L.v_proj = r.ReadBf16Linear(p + "self_attn.v_proj.weight", KV, H);
    L.o_proj = r.ReadBf16Linear(p + "self_attn.o_proj.weight", H, Q);
    L.k_proj_ctx = r.ReadBf16Linear(p + "self_attn.k_proj_ctx.weight", KV, H);
    L.v_proj_ctx = r.ReadBf16Linear(p + "self_attn.v_proj_ctx.weight", KV, H);
    // Stack gate_proj + up_proj into gate_up_proj [2*I, H]
    {
      const std::vector<uint16_t> gp = r.ReadBf16(p + "mlp.gate_proj.weight", {I, H});
      const std::vector<uint16_t> up = r.ReadBf16(p + "mlp.up_proj.weight", {I, H});
      std::vector<uint16_t> gu;
      gu.reserve(gp.size() + up.size());
      gu.insert(gu.end(), gp.begin(), gp.end());
      gu.insert(gu.end(), up.begin(), up.end());
      L.gate_up_proj.layout = Layout::kBf16;
      L.gate_up_proj.N = 2 * I;
      L.gate_up_proj.K = H;
      L.gate_up_proj.bf16_w = core::DeviceBuffer<uint16_t>(gu.size());
      L.gate_up_proj.bf16_w.CopyFromHost(gu);
    }
    L.down_proj = r.ReadBf16Linear(p + "mlp.down_proj.weight", H, I);

    const bool local = c.is_local[static_cast<size_t>(l)];
    L.window = local ? static_cast<int>(c.sliding_window) : 0;
    L.slots = local ? static_cast<int>(std::min<int64_t>(max_ctx, 2048)) : static_cast<int>(max_ctx);
    const size_t kv_elems = static_cast<size_t>(L.slots) * static_cast<size_t>(KV);
    L.k_store = core::DeviceBuffer<uint16_t>(kv_elems);
    L.v_store = core::DeviceBuffer<uint16_t>(kv_elems);
  }

  // Recurrent Markov head
  d.markov_dflare_norm_ = r.ReadDevBf16("markov_head.dflare_norm.weight", {H});
  d.markov_w1_host_ = r.ReadBf16("markov_head.markov_w1.weight", {c.vocab_size, R});
  d.rnn_proj_ = r.ReadBf16Linear("markov_head.rnn_proj.weight", 3 * R, 2 * R + 2 * H);
  d.rnn_bias_bf16_ = r.ReadDevBf16("markov_head.rnn_proj.bias", {3 * R});
  {
    const std::vector<uint16_t> hw = r.ReadBf16("markov_head.hidden_out.weight", {H, R});
    d.hidden_out_w_ = core::DeviceBuffer<uint16_t>(hw.size());
    d.hidden_out_w_.CopyFromHost(hw);
    d.hidden_out_w_t_host_.resize(static_cast<size_t>(R * H));
    for (int64_t h = 0; h < H; ++h) {
      for (int64_t rk = 0; rk < R; ++rk) {
        d.hidden_out_w_t_host_[static_cast<size_t>(rk * H + h)] = hw[static_cast<size_t>(h * R + rk)];
      }
    }
  }
  d.hidden_out_b_ = r.ReadDevBf16("markov_head.hidden_out.bias", {H});

  d.f_rnn_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(H));
  d.f_rnn_.Zero();
  d.prev_emb_host_ = core::PinnedBuffer<uint16_t>(static_cast<size_t>(R));

  std::cerr << "[r4dx::model::Inkling2DsparkDraft] loaded " << safetensors_path << " (3 layers, block " << c.block_size
            << ", markov_rank " << R << ", markov_w1 host-resident 103 MB, KV slots " << d.layers_[0].slots << "/"
            << d.layers_[1].slots << "/" << d.layers_[2].slots << ")\n";
  return d;
}

void Inkling2DsparkDraft::Reset() {
  n_injected_ = 0;
  ckpt_injected_ = -1;
}

void Inkling2DsparkDraft::SaveCheckpoint(core::Stream& stream) {
  const hipStream_t sr = stream.get();
  if (f_rnn_ckpt_.empty()) f_rnn_ckpt_ = core::DeviceBuffer<uint16_t>(f_rnn_.size());
  R4DX_HIP_CHECK(hipMemcpyAsync(f_rnn_ckpt_.data(), f_rnn_.data(), f_rnn_.bytes(), hipMemcpyDeviceToDevice, sr));
  for (Layer& L : layers_) {
    if (L.window > 0) {
      if (L.k_store_ckpt.empty()) {
        L.k_store_ckpt = core::DeviceBuffer<uint16_t>(L.k_store.size());
        L.v_store_ckpt = core::DeviceBuffer<uint16_t>(L.v_store.size());
      }
      R4DX_HIP_CHECK(hipMemcpyAsync(L.k_store_ckpt.data(), L.k_store.data(), L.k_store.bytes(),
                                    hipMemcpyDeviceToDevice, sr));
      R4DX_HIP_CHECK(hipMemcpyAsync(L.v_store_ckpt.data(), L.v_store.data(), L.v_store.bytes(),
                                    hipMemcpyDeviceToDevice, sr));
    }
  }
  ckpt_injected_ = n_injected_;
}

void Inkling2DsparkDraft::RestoreCheckpoint(core::Stream& stream) {
  if (ckpt_injected_ < 0) return;
  const hipStream_t sr = stream.get();
  R4DX_HIP_CHECK(hipMemcpyAsync(f_rnn_.data(), f_rnn_ckpt_.data(), f_rnn_.bytes(), hipMemcpyDeviceToDevice, sr));
  for (Layer& L : layers_) {
    if (L.window > 0 && !L.k_store_ckpt.empty()) {
      R4DX_HIP_CHECK(hipMemcpyAsync(L.k_store.data(), L.k_store_ckpt.data(), L.k_store.bytes(),
                                    hipMemcpyDeviceToDevice, sr));
      R4DX_HIP_CHECK(hipMemcpyAsync(L.v_store.data(), L.v_store_ckpt.data(), L.v_store.bytes(),
                                    hipMemcpyDeviceToDevice, sr));
    }
  }
  n_injected_ = ckpt_injected_;
}

void Inkling2DsparkDraft::InitFoldedLmHead(int64_t vl, core::Stream& stream, core::Arena& arena,
                                           const DsparkLmHeadBatchFn& lm_head_batch) {
  const int64_t H = cfg_.hidden_size, R = cfg_.markov_rank;
  if (vl <= 0 || vl % 16 != 0 || R % 64 != 0 || hidden_out_w_t_host_.empty()) return;
  const hipStream_t sr = stream.get();
  const int64_t s = P(sr);
  const size_t mark = arena.Mark();
  uint16_t* w_t_dev = arena.Alloc<uint16_t>(static_cast<size_t>(R * H), 16);
  R4DX_HIP_CHECK(hipMemcpyAsync(w_t_dev, hidden_out_w_t_host_.data(), static_cast<size_t>(R * H) * 2,
                                hipMemcpyHostToDevice, sr));
  uint16_t* tmp_fold_t = arena.Alloc<uint16_t>(static_cast<size_t>(R * vl), 16);
  lm_head_batch(w_t_dev, R, tmp_fold_t);

  fold_w_.layout = Layout::kBf16;
  fold_w_.N = vl;
  fold_w_.K = R;
  fold_w_.bf16_w = core::DeviceBuffer<uint16_t>(static_cast<size_t>(vl * R));
  r4dx_inkling2_dspark_transpose_bf16(P(tmp_fold_t), P(fold_w_.bf16_w.data()), R, vl, s);

  fold_b_ = core::DeviceBuffer<uint16_t>(static_cast<size_t>(vl));
  lm_head_batch(hidden_out_b_.data(), 1, fold_b_.data());

  stream.Synchronize();
  arena.Rewind(mark);
  std::vector<uint16_t>().swap(hidden_out_w_t_host_);
  fold_vl_ = vl;
  has_folded_lm_head_ = true;
}

void Inkling2DsparkDraft::InjectTaps(const uint16_t* tap0, const uint16_t* tap1, const uint16_t* tap2,
                                     const uint16_t* tap3, int64_t n, int64_t pos0, core::Stream& stream,
                                     core::Arena& arena) {
  if (n <= 0) return;
  const hipStream_t sr = stream.get();
  const int64_t s = P(sr);
  const int64_t H = cfg_.hidden_size, KV = cfg_.num_key_value_heads * cfg_.head_dim;
  const size_t mark = arena.Mark();
  uint16_t* ctx = arena.Alloc<uint16_t>(static_cast<size_t>(n * H), 16);
  uint16_t* k = arena.Alloc<uint16_t>(static_cast<size_t>(n * KV), 16);
  uint16_t* v = arena.Alloc<uint16_t>(static_cast<size_t>(n * KV), 16);

  for (int64_t l = 0; l < cfg_.num_layers; ++l) {
    Layer& L = layers_[static_cast<size_t>(l)];
    const auto& al = fuse_alpha_[static_cast<size_t>(l)];
    r4dx_inkling2_dspark_mix_rmsnorm_bf16(P(tap0), P(tap1), P(tap2), P(tap3), al[0], al[1], al[2], al[3],
                                          P(L.dflare_norm.data()), P(ctx), static_cast<int>(n), static_cast<int>(H),
                                          cfg_.rms_norm_eps, s);
    ApplyLinear(sr, arena, L.k_proj_ctx, ctx, k, n);
    ApplyLinear(sr, arena, L.v_proj_ctx, ctx, v, n);
    r4dx_inkling2_dspark_head_norm_rope_bf16(P(k), KV, P(L.k_norm_ctx.data()), static_cast<int>(n),
                                             static_cast<int>(cfg_.num_key_value_heads), static_cast<int>(cfg_.head_dim),
                                             pos0, cfg_.rope_theta, cfg_.rms_norm_eps, s);
    r4dx_inkling2_dspark_kv_write_bf16(P(k), P(v), P(L.k_store.data()), P(L.v_store.data()), static_cast<int>(n),
                                       static_cast<int>(cfg_.num_key_value_heads), static_cast<int>(cfg_.head_dim), pos0,
                                       L.slots, s);
  }
  // Update f_rnn_ from the last committed token (row n - 1).
  const auto& al3 = fuse_alpha_[3];
  const int64_t last_off = (n - 1) * H;
  r4dx_inkling2_dspark_mix_rmsnorm_bf16(P(tap0 + last_off), P(tap1 + last_off), P(tap2 + last_off), P(tap3 + last_off),
                                        al3[0], al3[1], al3[2], al3[3], P(markov_dflare_norm_.data()), P(f_rnn_.data()),
                                        1, static_cast<int>(H), cfg_.rms_norm_eps, s);
  arena.Rewind(mark);
  n_injected_ = pos0 + n;
}

std::vector<int32_t> Inkling2DsparkDraft::DraftGreedy(int32_t anchor_token, int64_t anchor_pos, int64_t max_steps,
                                                      core::Stream& stream, core::Arena& arena,
                                                      const DsparkAnchorEmbedFn& embed_anchor,
                                                      const DsparkLmHeadArgmaxFn& lm_head_argmax,
                                                      const DsparkLmHeadBatchFn& lm_head_batch,
                                                      const DsparkLogitsArgmaxFn& logits_argmax) {
  const hipStream_t sr = stream.get();
  const int64_t s = P(sr);
  const int64_t B = cfg_.block_size;
  const int64_t n_spec = B - 1;
  const int64_t n_steps = (max_steps > 0 && max_steps < n_spec) ? max_steps : n_spec;
  const int64_t H = cfg_.hidden_size, Q = cfg_.num_attention_heads * cfg_.head_dim;
  const int64_t KV = cfg_.num_key_value_heads * cfg_.head_dim, I = cfg_.intermediate_size, R = cfg_.markov_rank;
  const float scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
  const size_t mark = arena.Mark();

  // 1. Build input block x[0..B-1]: row 0 = embed_norm(embed_tokens(anchor)), rows 1..B-1 = mask_embed.
  uint16_t* resid = arena.Alloc<uint16_t>(static_cast<size_t>(B * H), 16);
  uint16_t* x = arena.Alloc<uint16_t>(static_cast<size_t>(B * H), 16);
  embed_anchor(anchor_token, resid);
  for (int64_t r = 1; r < B; ++r) {
    R4DX_HIP_CHECK(hipMemcpyAsync(resid + r * H, mask_embed_.data(), static_cast<size_t>(H) * 2,
                                  hipMemcpyDeviceToDevice, sr));
  }

  // 2. Run 3 dense RoPE layers with bidirectional block attention over [context KV, block KV].
  uint16_t* q = arena.Alloc<uint16_t>(static_cast<size_t>(B * Q), 16);
  uint16_t* k = arena.Alloc<uint16_t>(static_cast<size_t>(B * KV), 16);
  uint16_t* v = arena.Alloc<uint16_t>(static_cast<size_t>(B * KV), 16);
  uint16_t* ao = arena.Alloc<uint16_t>(static_cast<size_t>(B * Q), 16);
  uint16_t* o = arena.Alloc<uint16_t>(static_cast<size_t>(B * H), 16);
  uint16_t* gu = arena.Alloc<uint16_t>(static_cast<size_t>(B * 2 * I), 16);
  uint16_t* act = arena.Alloc<uint16_t>(static_cast<size_t>(B * I), 16);

  for (int64_t l = 0; l < cfg_.num_layers; ++l) {
    const Layer& L = layers_[static_cast<size_t>(l)];
    r4dx_rmsnorm_plain_bf16(P(resid), P(L.input_layernorm.data()), P(x), B, H, cfg_.rms_norm_eps, 0, s);
    ApplyLinear(sr, arena, L.q_proj, x, q, B);
    ApplyLinear(sr, arena, L.k_proj, x, k, B);
    ApplyLinear(sr, arena, L.v_proj, x, v, B);
    r4dx_inkling2_dspark_head_norm_rope_bf16(P(q), Q, P(L.q_norm.data()), static_cast<int>(B),
                                             static_cast<int>(cfg_.num_attention_heads), static_cast<int>(cfg_.head_dim),
                                             anchor_pos, cfg_.rope_theta, cfg_.rms_norm_eps, s);
    r4dx_inkling2_dspark_head_norm_rope_bf16(P(k), KV, P(L.k_norm.data()), static_cast<int>(B),
                                             static_cast<int>(cfg_.num_key_value_heads), static_cast<int>(cfg_.head_dim),
                                             anchor_pos, cfg_.rope_theta, cfg_.rms_norm_eps, s);
    r4dx_inkling2_dspark_attn_bf16(P(q), P(k), P(v), P(L.k_store.data()), P(L.v_store.data()), P(ao),
                                   static_cast<int>(B), static_cast<int>(cfg_.num_attention_heads),
                                   static_cast<int>(cfg_.num_key_value_heads), static_cast<int>(cfg_.head_dim),
                                   static_cast<int>(n_injected_), L.window, L.slots, scale, s);
    ApplyLinear(sr, arena, L.o_proj, ao, o, B);
    r4dx_residual_add_bf16(P(resid), P(o), P(resid), B * H, s);
    r4dx_rmsnorm_plain_bf16(P(resid), P(L.post_attention_layernorm.data()), P(x), B, H, cfg_.rms_norm_eps, 0, s);
    ApplyLinear(sr, arena, L.gate_up_proj, x, gu, B);
    r4dx_silu_mul_bf16(P(gu), P(act), B, I, 2 * I, s);
    ApplyLinear(sr, arena, L.down_proj, act, o, B);
    r4dx_residual_add_bf16(P(resid), P(o), P(resid), B * H, s);
  }

  // 3. Final RMSNorm -> head_hidden[0..B-1]. We sample rows 1..B-1 (sample_from_anchor == false).
  uint16_t* head_h = arena.Alloc<uint16_t>(static_cast<size_t>(B * H), 16);
  r4dx_rmsnorm_plain_bf16(P(resid), P(norm_.data()), P(head_h), B, H, cfg_.rms_norm_eps, 0, s);

  // 4. Sequential recurrent Markov loop over steps i = 0..n_steps-1.
  // rnn_in = [state (R), prev_emb (R), head_h[1+i] (H), f_rnn (H)] -> [2*R + 2*H] bf16.
  const int64_t rnn_in_dim = 2 * R + 2 * H;
  uint16_t* rnn_in = arena.Alloc<uint16_t>(static_cast<size_t>(rnn_in_dim), 16);
  uint16_t* state_next = arena.Alloc<uint16_t>(static_cast<size_t>(R), 16);
  uint16_t* latent = arena.Alloc<uint16_t>(static_cast<size_t>(R), 16);
  uint16_t* raw_bf16 = arena.Alloc<uint16_t>(static_cast<size_t>(3 * R), 16);
  R4DX_HIP_CHECK(hipMemsetAsync(rnn_in, 0, static_cast<size_t>(R) * 2, sr));
  R4DX_HIP_CHECK(hipMemcpyAsync(rnn_in + 2 * R + H, f_rnn_.data(), static_cast<size_t>(H) * 2,
                                hipMemcpyDeviceToDevice, sr));

  const bool use_folded = has_folded_lm_head_ && static_cast<bool>(lm_head_batch) && static_cast<bool>(logits_argmax);
  uint16_t* base_lg = nullptr;
  uint16_t* delta_lg = nullptr;
  uint16_t* delta_h = nullptr;
  uint16_t* h_eff = nullptr;
  if (use_folded) {
    base_lg = arena.Alloc<uint16_t>(static_cast<size_t>(n_steps * fold_vl_), 16);
    delta_lg = arena.Alloc<uint16_t>(static_cast<size_t>(fold_vl_), 16);
    lm_head_batch(head_h + H, n_steps, base_lg);
  } else {
    delta_h = arena.Alloc<uint16_t>(static_cast<size_t>(H), 16);
    h_eff = arena.Alloc<uint16_t>(static_cast<size_t>(H), 16);
  }

  std::vector<int32_t> drafted;
  drafted.reserve(static_cast<size_t>(n_steps));
  int32_t prev = anchor_token;

  for (int64_t i = 0; i < n_steps; ++i) {
    const int64_t clamped = (prev >= 0 && prev < cfg_.vocab_size) ? prev : 0;
    std::memcpy(prev_emb_host_.data(), markov_w1_host_.data() + clamped * R, static_cast<size_t>(R) * 2);
    R4DX_HIP_CHECK(hipMemcpyAsync(rnn_in + R, prev_emb_host_.data(), static_cast<size_t>(R) * 2,
                                  hipMemcpyHostToDevice, sr));
    R4DX_HIP_CHECK(hipMemcpyAsync(rnn_in + 2 * R, head_h + (i + 1) * H, static_cast<size_t>(H) * 2,
                                  hipMemcpyDeviceToDevice, sr));
    ApplyLinear(sr, arena, rnn_proj_, rnn_in, raw_bf16, 1);
    r4dx_inkling2_dspark_markov_gate_bf16(P(raw_bf16), P(rnn_bias_bf16_.data()), P(rnn_in), P(state_next), P(latent),
                                          static_cast<int>(R), s);
    R4DX_HIP_CHECK(hipMemcpyAsync(rnn_in, state_next, static_cast<size_t>(R) * 2, hipMemcpyDeviceToDevice, sr));
    if (use_folded) {
      ApplyLinear(sr, arena, fold_w_, latent, delta_lg, 1);
      prev = logits_argmax(base_lg + i * fold_vl_, delta_lg, fold_b_.data());
    } else {
      r4dx_inkling2_dense_bf16(P(latent), R, P(hidden_out_w_.data()), P(delta_h), 1, H, 1, static_cast<int>(H),
                               static_cast<int>(R), s);
      r4dx_inkling2_add_bf16(P(delta_h), P(hidden_out_b_.data()), P(delta_h), H, s);
      r4dx_inkling2_add_bf16(P(head_h + (i + 1) * H), P(delta_h), P(h_eff), H, s);
      prev = lm_head_argmax(h_eff);
    }
    drafted.push_back(prev);
  }

  arena.Rewind(mark);
  return drafted;
}

}  // namespace r4dx::model
