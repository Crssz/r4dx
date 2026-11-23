#include "inkling2_vision_tower.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>

#include "container_load_util.h"
#include "linear.h"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/inkling2_kernels.h"

namespace r4dx::model {

namespace {

inline int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

core::DeviceBuffer<uint16_t> LoadDevBf16(const r4dx_convert::SafetensorsReader& r, const std::string& name,
                                         int64_t elems) {
  if (!r.Has(name)) throw std::runtime_error("Inkling2VisionTower: missing tensor " + name);
  const auto& m = r.Meta(name);
  if (m.end - m.begin != static_cast<uint64_t>(elems) * 2) {
    throw std::runtime_error("Inkling2VisionTower: byte length mismatch for " + name);
  }
  core::DeviceBuffer<uint16_t> d(static_cast<size_t>(elems));
  d.CopyFromHost(reinterpret_cast<const uint16_t*>(r.Data(name)), static_cast<size_t>(elems));
  return d;
}

QuantLinear LoadBf16Linear(const r4dx_convert::SafetensorsReader& r, const std::string& name, int64_t N, int64_t K) {
  QuantLinear q;
  q.layout = Layout::kBf16;
  q.N = N;
  q.K = K;
  q.bf16_w = LoadDevBf16(r, name, N * K);
  return q;
}

}  // namespace

Inkling2VisionTower Inkling2VisionTower::TryLoad(const std::string& container_path) {
  Inkling2VisionTower t;
  r4dx_convert::SafetensorsReader r(container_util::Utf8ToWide(container_path));
  if (!r.Has("vision.backbone.embeddings.patch_embedding.weight")) return t;

  const Inkling2VisionConfig& c = t.cfg_;
  const int64_t D = c.hidden_size, M = c.mlp_dim;
  const int64_t Pdim = c.native_patch_size * c.native_patch_size * 3;
  const int64_t G = c.posemb_grid_size;

  t.class_emb_ = LoadDevBf16(r, "vision.backbone.embeddings.class_embedding", D);
  t.cls_pos_emb_ = LoadDevBf16(r, "vision.backbone.embeddings.cls_position_embedding", D);
  t.pos_emb_ = LoadDevBf16(r, "vision.backbone.embeddings.position_embedding", G * G * D);
  t.patch_emb_w_ = LoadDevBf16(r, "vision.backbone.embeddings.patch_embedding.weight", D * Pdim);
  t.ln_pre_w_ = LoadDevBf16(r, "vision.backbone.ln_pre.weight", D);
  t.ln_pre_b_ = LoadDevBf16(r, "vision.backbone.ln_pre.bias", D);

  t.layers_.resize(static_cast<size_t>(c.num_layers));
  for (int64_t l = 0; l < c.num_layers; ++l) {
    Layer& L = t.layers_[static_cast<size_t>(l)];
    const std::string p = "vision.backbone.layers." + std::to_string(l) + ".";
    L.ln1_w = LoadDevBf16(r, p + "ln_1.weight", D);
    L.ln1_b = LoadDevBf16(r, p + "ln_1.bias", D);
    L.ln2_w = LoadDevBf16(r, p + "ln_2.weight", D);
    L.ln2_b = LoadDevBf16(r, p + "ln_2.bias", D);
    L.ls1_g = LoadDevBf16(r, p + "ls_1.gamma", D);
    L.ls2_g = LoadDevBf16(r, p + "ls_2.gamma", D);
    L.q_proj = LoadBf16Linear(r, p + "self_attn.q_proj.weight", D, D);
    L.k_proj = LoadBf16Linear(r, p + "self_attn.k_proj.weight", D, D);
    L.v_proj = LoadBf16Linear(r, p + "self_attn.v_proj.weight", D, D);
    L.out_proj = LoadBf16Linear(r, p + "self_attn.out_proj.weight", D, D);
    L.q_b = LoadDevBf16(r, p + "self_attn.q_proj.bias", D);
    L.k_b = LoadDevBf16(r, p + "self_attn.k_proj.bias", D);
    L.v_b = LoadDevBf16(r, p + "self_attn.v_proj.bias", D);
    L.out_b = LoadDevBf16(r, p + "self_attn.out_proj.bias", D);
    L.c_fc = LoadBf16Linear(r, p + "mlp.c_fc.weight", M, D);
    L.c_proj = LoadBf16Linear(r, p + "mlp.c_proj.weight", D, M);
    L.fc_b = LoadDevBf16(r, p + "mlp.c_fc.bias", M);
    L.proj_b = LoadDevBf16(r, p + "mlp.c_proj.bias", D);
  }

  t.proj_in_ = LoadBf16Linear(r, "vision.proj_in.weight", c.proj_hidden_dim, 4 * D);
  t.proj_out_ = LoadBf16Linear(r, "vision.proj_out.weight", c.text_hidden_size, c.proj_hidden_dim);
  t.loaded_ = true;
  std::cerr << "[r4dx::model::Inkling2VisionTower] loaded 23-layer ViT (hidden " << D << ", patch "
            << c.native_patch_size << ", merge " << c.adapter_merge_size << "x" << c.adapter_merge_size << " -> "
            << c.text_hidden_size << ")\n";
  return t;
}

void Inkling2VisionTower::Encode(const float* pixel_values, int64_t total_patches,
                                 const std::vector<vision::GridThw>& grids, core::DeviceBuffer<uint16_t>* out_bf16,
                                 core::Stream& stream, core::Arena& arena) const {
  if (!loaded_) throw std::runtime_error("Inkling2VisionTower::Encode: no vision weights loaded");
  if (pixel_values == nullptr || out_bf16 == nullptr)
    throw std::invalid_argument("Inkling2VisionTower::Encode: null pointer");

  int64_t sum_patches = 0, total_merged = 0;
  for (const auto& g : grids) {
    if (g.t != 1 || g.h <= 0 || g.w <= 0 || (g.h % 2) != 0 || (g.w % 2) != 0)
      throw std::invalid_argument("Inkling2VisionTower::Encode: invalid grid_thw");
    sum_patches += g.h * g.w;
    total_merged += (g.h / 2) * (g.w / 2);
  }
  if (sum_patches != total_patches)
    throw std::invalid_argument("Inkling2VisionTower::Encode: total_patches mismatch");

  const int64_t D = cfg_.hidden_size, M = cfg_.mlp_dim, Htxt = cfg_.text_hidden_size;
  const int64_t Pdim = cfg_.native_patch_size * cfg_.native_patch_size * 3;
  const float attn_scale = 1.0f / std::sqrt(static_cast<float>(cfg_.head_dim));
  const hipStream_t sr = stream.get();
  const int64_t s = P(sr);

  *out_bf16 = core::DeviceBuffer<uint16_t>(static_cast<size_t>(total_merged * Htxt));

  int64_t patch_off = 0, merged_off = 0;
  for (const auto& g : grids) {
    const int64_t np = g.h * g.w;
    const int64_t N = 1 + np;
    const int64_t nm = (g.h / 2) * (g.w / 2);
    const size_t mark = arena.Mark();

    // 1. Convert input patches float32 -> bf16 on host and upload [np, 1200].
    std::vector<uint16_t> pv_bf16(static_cast<size_t>(np * Pdim));
    const float* src_f = pixel_values + static_cast<size_t>(patch_off * Pdim);
    for (size_t i = 0; i < pv_bf16.size(); ++i) pv_bf16[i] = core::FloatToBf16(src_f[i]);
    uint16_t* pv_dev = arena.Alloc<uint16_t>(pv_bf16.size(), 16);
    R4DX_HIP_CHECK(hipMemcpyAsync(pv_dev, pv_bf16.data(), pv_bf16.size() * 2, hipMemcpyHostToDevice, sr));

    // 2. Patch embedding [np, 1200] @ [1024, 1200]^T -> [np, 1024] bf16.
    uint16_t* pe_out = arena.Alloc<uint16_t>(static_cast<size_t>(np * D), 16);
    r4dx_inkling2_dense_bf16(P(pv_dev), Pdim, P(patch_emb_w_.data()), P(pe_out), /*out_bf16=*/1, D,
                             static_cast<int>(np), static_cast<int>(D), static_cast<int>(Pdim), s);

    // 3. Prepend CLS + add 2D bilinear-interpolated pos_emb -> [N, D], then ln_pre.
    uint16_t* resid = arena.Alloc<uint16_t>(static_cast<size_t>(N * D), 16);
    uint16_t* x = arena.Alloc<uint16_t>(static_cast<size_t>(N * D), 16);
    r4dx_inkling2_vit_pos_cls_bf16(P(pe_out), P(class_emb_.data()), P(cls_pos_emb_.data()), P(pos_emb_.data()),
                                   P(x), static_cast<int>(g.h), static_cast<int>(g.w),
                                   static_cast<int>(cfg_.posemb_grid_size), static_cast<int>(D), s);
    r4dx_inkling2_vit_layernorm_bf16(P(x), P(ln_pre_w_.data()), P(ln_pre_b_.data()), P(resid), static_cast<int>(N),
                                     static_cast<int>(D), cfg_.layer_norm_eps, s);

    // 4. 23 ViT layers.
    uint16_t* q = arena.Alloc<uint16_t>(static_cast<size_t>(N * D), 16);
    uint16_t* k = arena.Alloc<uint16_t>(static_cast<size_t>(N * D), 16);
    uint16_t* v = arena.Alloc<uint16_t>(static_cast<size_t>(N * D), 16);
    uint16_t* ao = arena.Alloc<uint16_t>(static_cast<size_t>(N * D), 16);
    uint16_t* o = arena.Alloc<uint16_t>(static_cast<size_t>(N * D), 16);
    uint16_t* fc = arena.Alloc<uint16_t>(static_cast<size_t>(N * M), 16);

    for (int64_t l = 0; l < cfg_.num_layers; ++l) {
      const Layer& L = layers_[static_cast<size_t>(l)];
      r4dx_inkling2_vit_layernorm_bf16(P(resid), P(L.ln1_w.data()), P(L.ln1_b.data()), P(x), static_cast<int>(N),
                                       static_cast<int>(D), cfg_.layer_norm_eps, s);
      ApplyLinear(sr, arena, L.q_proj, x, q, N);
      ApplyLinear(sr, arena, L.k_proj, x, k, N);
      ApplyLinear(sr, arena, L.v_proj, x, v, N);
      r4dx_inkling2_vit_bias_gelu_bf16(P(q), P(L.q_b.data()), P(q), static_cast<int>(N), static_cast<int>(D), 0, s);
      r4dx_inkling2_vit_bias_gelu_bf16(P(k), P(L.k_b.data()), P(k), static_cast<int>(N), static_cast<int>(D), 0, s);
      r4dx_inkling2_vit_bias_gelu_bf16(P(v), P(L.v_b.data()), P(v), static_cast<int>(N), static_cast<int>(D), 0, s);
      r4dx_inkling2_vit_attn_bf16(P(q), P(k), P(v), P(ao), static_cast<int>(N), static_cast<int>(cfg_.num_heads),
                                  static_cast<int>(cfg_.head_dim), attn_scale, s);
      ApplyLinear(sr, arena, L.out_proj, ao, o, N);
      r4dx_inkling2_vit_layerscale_add_bf16(P(resid), P(o), P(L.out_b.data()), P(L.ls1_g.data()), static_cast<int>(N),
                                            static_cast<int>(D), s);

      r4dx_inkling2_vit_layernorm_bf16(P(resid), P(L.ln2_w.data()), P(L.ln2_b.data()), P(x), static_cast<int>(N),
                                       static_cast<int>(D), cfg_.layer_norm_eps, s);
      ApplyLinear(sr, arena, L.c_fc, x, fc, N);
      r4dx_inkling2_vit_bias_gelu_bf16(P(fc), P(L.fc_b.data()), P(fc), static_cast<int>(N), static_cast<int>(M), 1, s);
      ApplyLinear(sr, arena, L.c_proj, fc, o, N);
      r4dx_inkling2_vit_layerscale_add_bf16(P(resid), P(o), P(L.proj_b.data()), P(L.ls2_g.data()), static_cast<int>(N),
                                            static_cast<int>(D), s);
    }

    // 5. Drop CLS, 2x2 spatial merge -> [nm, 4096], proj_in -> GELU -> proj_out -> [nm, 2304] bf16.
    uint16_t* merged = arena.Alloc<uint16_t>(static_cast<size_t>(nm * 4 * D), 16);
    uint16_t* proj_h = arena.Alloc<uint16_t>(static_cast<size_t>(nm * cfg_.proj_hidden_dim), 16);
    r4dx_inkling2_vit_merge2x2_bf16(P(resid), P(merged), static_cast<int>(g.h), static_cast<int>(g.w),
                                    static_cast<int>(D), s);
    ApplyLinear(sr, arena, proj_in_, merged, proj_h, nm);
    r4dx_inkling2_vit_bias_gelu_bf16(P(proj_h), 0, P(proj_h), static_cast<int>(nm),
                                     static_cast<int>(cfg_.proj_hidden_dim), 1, s);
    ApplyLinear(sr, arena, proj_out_, proj_h, out_bf16->data() + static_cast<size_t>(merged_off * Htxt), nm);

    stream.Synchronize();
    arena.Rewind(mark);
    patch_off += np;
    merged_off += nm;
  }
}

}  // namespace r4dx::model
