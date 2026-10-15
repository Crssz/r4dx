// r4dx::model::attention::Lfm2AttnLayer -- the attention mixer of one LFM2 (lfm2_moe) attention layer
// (docs/lfm2-plan.md 3.4, task L1-8; semantics: docs/lfm2-semantics.md 1). GemmaAttnLayer (k_eq_v, v_norm, windows,
// proportional rope, rotation) and the Qwen AttentionLayer (q|gate split, mrope) are not reused; only GemmaKvCache (a
// contiguous bf16 paged cache) and the stage-0 reference kernel (attn_ref.h) are shared.
//
// What Forward computes, given the layer's already-normed input (`x_normed` = operator_norm(h), the caller's):
//   q = q_proj(x_normed);  k = k_proj(x_normed);  v = v_proj(x_normed)            (separate linears, no bias, no gate)
//   q = q_layernorm(q), k = k_layernorm(k)    per head, plain weight [head_dim], rows T * heads, hidden head_dim
//   q, k = rope(q, k)                          rotate_half pairs (i, i + head_dim/2), theta 1e7, all head_dim dims
//                                              (r4dx_rope_neox_bf16; fp32 cos/sin -- HF rounds them to bf16)
//   KV write (post-rope K, raw V) into the layer's cache at the absolute positions, BEFORE the attention call
//   o = softmax(q k^T * head_dim^-0.5 [causal]) v                       (GQA: q head h reads kv head h / gqa)
//   o_out = out_proj(o)                        [TP: row-parallel, all-reduced here]
// The residual add is the CALLER's.
//
// KV: bf16 only (docs/lfm2-plan.md 3.4: bf16 for all 10 layers), GemmaKvCache::Contiguous(kv_heads, head_dim, max_ctx).
// Attention backend: the reference kernel only (no libr4d h64 kernel yet, L4-1).
#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include "kernels/model_kernels.h"
#include "linear.h"
#include "profile_span.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/model/attention/attn_ref.h"
#include "r4dx/model/attention/gemma_attention_layer.hpp"  // GemmaKvCache (contiguous bf16 paged cache)

namespace r4dx::model::attention {

struct Lfm2AttnConfig {
  int hidden = 2048;
  int heads = 32;     // q heads (the rank's under TP)
  int kv_heads = 8;   // kv heads (the rank's under TP)
  int head_dim = 64;
  float rope_theta = 1.0e7f;
  float rms_eps = 1.0e-5f;
  int Gqa() const { return heads / kv_heads; }
  float Scale() const { return 1.0f / std::sqrt(static_cast<float>(head_dim)); }
};

struct Lfm2AttnLayerWeights {
  const QuantLinear* q = nullptr;
  const QuantLinear* k = nullptr;
  const QuantLinear* v = nullptr;
  const QuantLinear* o = nullptr;
  const uint16_t* q_norm = nullptr;  // [head_dim] bf16, plain weight
  const uint16_t* k_norm = nullptr;
};

class Lfm2AttnLayer {
 public:
  explicit Lfm2AttnLayer(const Lfm2AttnConfig& cfg) : cfg_(cfg) {
    if (cfg_.heads <= 0 || cfg_.kv_heads <= 0 || cfg_.head_dim <= 0 || cfg_.heads % cfg_.kv_heads != 0 ||
        cfg_.head_dim % 2 != 0) {
      throw std::invalid_argument("Lfm2AttnLayer: bad head geometry");
    }
  }

  // x_normed: [T, hidden] bf16. o_out: [T, hidden] bf16 (the out_proj output; all-reduced when comm != null).
  // positions: device int32 [T] -- the absolute positions, which are also the KV write slots (contiguous cache).
  // start_pos: tokens already cached before this call (ctx = start_pos + T).
  void Forward(core::Arena& arena, const uint16_t* x_normed, uint16_t* o_out, const Lfm2AttnLayerWeights& w,
               GemmaKvCache& kv, int T, int start_pos, const int32_t* positions, hipStream_t stream,
               SpanAccumulator* prof = nullptr, core::TpComm* comm = nullptr) const {
    const int H = cfg_.heads, Hk = cfg_.kv_heads, D = cfg_.head_dim;
    if (T < 1) throw std::invalid_argument("Lfm2AttnLayer::Forward: T must be >= 1");
    if (kv.KvHeads() != Hk || kv.HeadDim() != D || kv.Dtype() != GemmaKvDtype::kBf16 || kv.IsRing()) {
      throw std::invalid_argument("Lfm2AttnLayer::Forward: the cache must be a contiguous bf16 cache of this geometry");
    }
    if (w.q == nullptr || w.k == nullptr || w.v == nullptr || w.o == nullptr || w.q_norm == nullptr || w.k_norm == nullptr) {
      throw std::invalid_argument("Lfm2AttnLayer::Forward: missing weights");
    }
    const int64_t s = reinterpret_cast<int64_t>(stream);
    const auto P = [](const void* p) { return reinterpret_cast<int64_t>(p); };

    uint16_t* q = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D, 16);
    uint16_t* k = arena.Alloc<uint16_t>(static_cast<size_t>(T) * Hk * D, 16);
    uint16_t* v = arena.Alloc<uint16_t>(static_cast<size_t>(T) * Hk * D, 16);
    ProfiledCall(prof, stream, "gemm:attn.q_proj", [&] { ApplyLinear(stream, arena, *w.q, x_normed, q, T); });
    ProfiledCall(prof, stream, "gemm:attn.k_proj", [&] { ApplyLinear(stream, arena, *w.k, x_normed, k, T); });
    ProfiledCall(prof, stream, "gemm:attn.v_proj", [&] { ApplyLinear(stream, arena, *w.v, x_normed, v, T); });

    ProfiledCall(prof, stream, "attn.qk_norm", [&] {
      r4dx_rmsnorm_plain_bf16(P(q), P(w.q_norm), P(q), static_cast<int64_t>(T) * H, D, cfg_.rms_eps, 0, s);
      r4dx_rmsnorm_plain_bf16(P(k), P(w.k_norm), P(k), static_cast<int64_t>(T) * Hk, D, cfg_.rms_eps, 0, s);
    });
    ProfiledCall(prof, stream, "attn.rope", [&] {
      r4dx_rope_neox_bf16(P(q), P(k), P(positions), T, H, Hk, D, cfg_.rope_theta, s);
    });
    ProfiledCall(prof, stream, "attn.kv_write", [&] {
      r4dx_model_kv_write_paged_bf16_hnd(P(k), P(v), P(positions), P(kv.Data()), T, Hk, D, GemmaKvCache::kBlockSize,
                                         kv.KvBlockStride(), kv.KvHeadStride(), s);
    });

    uint16_t* attn_out = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D, 16);
    R4dxGemmaAttnRefArgs a{};
    a.q = P(q);
    a.kv = P(kv.Data());
    a.block_table = P(kv.BlockTable());
    a.out = P(attn_out);
    a.k_descale = 0;
    a.v_descale = 0;
    a.klimit_ext = 0;
    a.q_len = T;
    a.q_heads = H;
    a.kv_heads = Hk;
    a.head_dim = D;
    a.block_size = GemmaKvCache::kBlockSize;
    a.ctx = start_pos + T;
    a.window = 0;
    a.scale = cfg_.Scale();
    a.kv_block_stride = kv.KvBlockStride();
    a.kv_head_stride = kv.KvHeadStride();
    ProfiledCall(prof, stream, "attn.core_ref", [&] {
      const int rc = r4dx_gemma_attn_ref_bf16kv(&a, s);
      if (rc != 0) throw std::runtime_error("Lfm2AttnLayer: r4dx_gemma_attn_ref_bf16kv returned " + std::to_string(rc));
    });
    ProfiledCall(prof, stream, "gemm:attn.o_proj", [&] { ApplyLinear(stream, arena, *w.o, attn_out, o_out, T); });
    if (comm != nullptr) {
      ProfiledCall(prof, stream, "tp.allreduce", [&] { comm->AllReduceSumBf16Rows(o_out, T, cfg_.hidden, stream); });
    }
  }

  const Lfm2AttnConfig& Config() const { return cfg_; }

 private:
  Lfm2AttnConfig cfg_;
};

}  // namespace r4dx::model::attention
