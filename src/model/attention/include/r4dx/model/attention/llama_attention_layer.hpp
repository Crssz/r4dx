// r4dx::model::attention::LlamaAttnLayer -- the attention mixer of one dense Llama layer (docs/nanbeige-plan.md 3.3,
// task N1-6). Lfm2AttnLayer without the per-head q / k norms; GemmaKvCache (a contiguous bf16 paged cache) and the
// stage-0 reference kernel (attn_ref.h) are shared.
//
// What Forward computes, given the layer's already-normed input (`x_normed` = input_layernorm(h), the caller's):
//   q = q_proj(x_normed);  k = k_proj(x_normed);  v = v_proj(x_normed)            (separate linears, no bias)
//   q, k = rope(q, k)                          rotate_half pairs (i, i + head_dim/2) over all head_dim dims,
//                                              fp32 cos/sin of fp32 position * inv_freq (r4dx_rope_neox_bf16)
//   KV write (post-rope K, raw V) into the layer's cache at the absolute positions, BEFORE the attention call
//   o = softmax(q k^T * head_dim^-0.5 [causal]) v                       (GQA: q head h reads kv head h / gqa)
//   o_out = o_proj(o)                          [TP: row-parallel, all-reduced here]
// The residual add is the CALLER's.
//
// KV: bf16 (docs/nanbeige-plan.md 3.3; fp8 is N5-2), GemmaKvCache::Contiguous(kv_heads, head_dim, max_ctx).
// Attention backend (N5-1): libr4d's h128_gqa7 bf16-KV kernels when head_dim == 128 and GQA == 7 (Nanbeige, at TP=1
// and TP=2): split-KV decode for T <= 9 (its q_len x gqa <= 64 rows), prefill otherwise, split-KV prefill from
// kSplitKvMinCtx tokens of context (Lfm2PrefillSplits' law). R4DX_LLAMA_ATTN=ref, or any other geometry, runs the
// stage-0 reference kernel (exact, slow; the oracle of tests/kernels/test_attn_h128_gqa7.cpp).
#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <cstdlib>

#include "kernels/model_kernels.h"
#include "linear.h"
#include "profile_span.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/model/attention/attn_ref.h"
#include "r4dx/model/attention/gemma_attention_layer.hpp"  // GemmaKvCache (contiguous bf16 paged cache)
#include "r4dx/model/attention/lfm2_attention_layer.hpp"   // Lfm2PrefillSplits (the split-KV prefill law)

namespace r4dx::model::attention {

// The h128_gqa7 decode kernel holds q_len x gqa <= 64 rows: T <= 9 at GQA 7.
inline constexpr int kLlamaDecodeMaxRows = 9;

// R4DX_LLAMA_ATTN=ref forces the reference kernel (read once).
inline bool LlamaAttnForceRef() {
  static const bool v = [] {
    const char* e = std::getenv("R4DX_LLAMA_ATTN");
    return e != nullptr && std::string(e) == "ref";
  }();
  return v;
}

struct LlamaAttnConfig {
  int hidden = 7168;
  int heads = 56;     // q heads (the rank's under TP)
  int kv_heads = 8;   // kv heads (the rank's under TP)
  int head_dim = 128;
  float rope_theta = 1.2e8f;
  int Gqa() const { return heads / kv_heads; }
  float Scale() const { return 1.0f / std::sqrt(static_cast<float>(head_dim)); }
};

struct LlamaAttnLayerWeights {
  const QuantLinear* q = nullptr;
  const QuantLinear* k = nullptr;
  const QuantLinear* v = nullptr;
  const QuantLinear* o = nullptr;
};

class LlamaAttnLayer {
 public:
  explicit LlamaAttnLayer(const LlamaAttnConfig& cfg) : cfg_(cfg) {
    if (cfg_.heads <= 0 || cfg_.kv_heads <= 0 || cfg_.head_dim <= 0 || cfg_.heads % cfg_.kv_heads != 0 ||
        cfg_.head_dim % 2 != 0) {
      throw std::invalid_argument("LlamaAttnLayer: bad head geometry");
    }
  }

  // x_normed: [T, hidden] bf16. o_out: [T, hidden] bf16 (the o_proj output; all-reduced when comm != null).
  // positions: device int32 [T] -- the absolute positions, which are also the KV write slots (contiguous cache).
  // seqused_k: device int32 [1] == start_pos + T (the libr4d attention reads the context length on device).
  // start_pos: tokens already cached before this call (ctx = start_pos + T).
  void Forward(core::Arena& arena, const uint16_t* x_normed, uint16_t* o_out, const LlamaAttnLayerWeights& w,
               GemmaKvCache& kv, int T, int start_pos, const int32_t* positions, const int32_t* seqused_k,
               hipStream_t stream, SpanAccumulator* prof = nullptr, core::TpComm* comm = nullptr) const {
    const int H = cfg_.heads, Hk = cfg_.kv_heads, D = cfg_.head_dim;
    if (T < 1) throw std::invalid_argument("LlamaAttnLayer::Forward: T must be >= 1");
    if (kv.KvHeads() != Hk || kv.HeadDim() != D || kv.Dtype() != GemmaKvDtype::kBf16 || kv.IsRing()) {
      throw std::invalid_argument("LlamaAttnLayer::Forward: the cache must be a contiguous bf16 cache of this geometry");
    }
    if (w.q == nullptr || w.k == nullptr || w.v == nullptr || w.o == nullptr) {
      throw std::invalid_argument("LlamaAttnLayer::Forward: missing weights");
    }
    const int64_t s = reinterpret_cast<int64_t>(stream);
    const auto P = [](const void* p) { return reinterpret_cast<int64_t>(p); };

    uint16_t* q = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D, 16);
    uint16_t* k = arena.Alloc<uint16_t>(static_cast<size_t>(T) * Hk * D, 16);
    uint16_t* v = arena.Alloc<uint16_t>(static_cast<size_t>(T) * Hk * D, 16);
    ProfiledCall(prof, stream, "gemm:attn.q_proj", [&] { ApplyLinear(stream, arena, *w.q, x_normed, q, T); });
    ProfiledCall(prof, stream, "gemm:attn.k_proj", [&] { ApplyLinear(stream, arena, *w.k, x_normed, k, T); });
    ProfiledCall(prof, stream, "gemm:attn.v_proj", [&] { ApplyLinear(stream, arena, *w.v, x_normed, v, T); });
    ProfiledCall(prof, stream, "attn.rope", [&] { r4dx_rope_neox_bf16(P(q), P(k), P(positions), T, H, Hk, D, cfg_.rope_theta, s); });
    ProfiledCall(prof, stream, "attn.kv_write", [&] {
      r4dx_model_kv_write_paged_bf16_hnd(P(k), P(v), P(positions), P(kv.Data()), T, Hk, D, GemmaKvCache::kBlockSize,
                                         kv.KvBlockStride(), kv.KvHeadStride(), s);
    });

    uint16_t* attn_out = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D, 16);
    const int ctx = start_pos + T;
    if (D == 128 && cfg_.Gqa() == 7 && seqused_k != nullptr && !LlamaAttnForceRef()) {
      R4DArgs a{};
      a.q = q;
      a.kv = kv.Data();
      a.block_table = kv.BlockTable();
      a.seqused_k = seqused_k;
      a.out = attn_out;
      a.k_descale = nullptr;
      a.v_descale = nullptr;
      a.q_descale = nullptr;
      a.scratch = nullptr;
      a.num_seqs = 1;
      a.q_len = T;
      a.q_heads = H;
      a.kv_heads = Hk;
      a.head_dim = D;
      a.block_size = GemmaKvCache::kBlockSize;
      a.max_blocks = kv.TableEntries();
      a.kv_block_stride = kv.KvBlockStride();
      a.kv_head_stride = kv.KvHeadStride();
      a.scale = cfg_.Scale();
      a.splits = 0;
      a.max_ctx = ctx;
      if (T <= kLlamaDecodeMaxRows) {
        a.scratch = arena.Alloc<uint8_t>(static_cast<size_t>(core::r4d::AttnDecodeH128ScratchBytes(a)), 16);
        ProfiledCall(prof, stream, "attn.core_decode", [&] { core::r4d::AttnDecodeH128Bf16Kv(a, stream); });
      } else {
        const int splits = Lfm2PrefillSplits(ctx, T, Hk);
        if (splits > 1) {
          a.splits = splits;
          a.scratch = arena.Alloc<uint8_t>(static_cast<size_t>(core::r4d::AttnPrefillSplitKvH128ScratchBytes(a)), 256);
          ProfiledCall(prof, stream, "attn.core_prefill", [&] { core::r4d::AttnPrefillSplitKvH128Bf16Kv(a, stream); });
        } else {
          ProfiledCall(prof, stream, "attn.core_prefill", [&] { core::r4d::AttnPrefillH128Bf16Kv(a, stream); });
        }
      }
    } else {
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
      a.ctx = ctx;
      a.window = 0;
      a.scale = cfg_.Scale();
      a.kv_block_stride = kv.KvBlockStride();
      a.kv_head_stride = kv.KvHeadStride();
      ProfiledCall(prof, stream, "attn.core_ref", [&] {
        const int rc = r4dx_gemma_attn_ref_bf16kv(&a, s);
        if (rc != 0) throw std::runtime_error("LlamaAttnLayer: r4dx_gemma_attn_ref_bf16kv returned " + std::to_string(rc));
      });
    }
    ProfiledCall(prof, stream, "gemm:attn.o_proj", [&] { ApplyLinear(stream, arena, *w.o, attn_out, o_out, T); });
    if (comm != nullptr) {
      ProfiledCall(prof, stream, "tp.allreduce", [&] { comm->AllReduceSumBf16Rows(o_out, T, cfg_.hidden, stream); });
    }
  }

  const LlamaAttnConfig& Config() const { return cfg_; }

 private:
  LlamaAttnConfig cfg_;
};

}  // namespace r4dx::model::attention
