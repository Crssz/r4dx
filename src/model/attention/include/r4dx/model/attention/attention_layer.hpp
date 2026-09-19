// r4dx::model::attention::AttentionLayer -- one Qwen3.5 full-attention decoder layer
// (docs/architecture.md "Attention layer (16 of 64 layers)"), exactly per
// modeling_qwen3_5.py's Qwen3_5Attention.forward:
//
//   residual = x
//   x = rmsnorm(x, input_layernorm)
//   qg = gemm(x, attn.qg);  q, gate = split(qg, head_dim, head_dim) per head (per-head-interleaved)
//   k = gemm(x, attn.k);  v = gemm(x, attn.v)
//   q = rmsnorm_zero_centered(q, q_norm);  k = rmsnorm_zero_centered(k, k_norm)   (per head_dim row)
//   q, k = rope(q, k, pos_ids)             (partial rotary 0.25, mrope text positions)
//   kv_write(k, v)  -- into the fp8 paged cache, BEFORE the attention call
//   o = r4d_attn_{prefill,decode}(q, kv_cache, ...)
//   o = o * sigmoid(gate)
//   out = residual + gemm(o, attn.o)
//
// SCOPE: single sequence only (num_seqs==1) -- see PagedKvCache's own doc comment. This
// component owns src/model/attention/** and tests/model/attention/** only; it defines its own
// AttnWeights/Linear/PagedKvCache rather than depending on the model-core agent's equivalents
// (task brief), so the caller (an assembly/loader stage, or this component's own golden test)
// is responsible for pointing an AttnWeights at real container-loaded device buffers.
#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "linear.h"  // r4dx::model::ApplyLinear -- shared qg/o/k/v quantized-linear dispatch
#include "profile_span.h"  // r4dx::model::SpanAccumulator / ProfiledCall (Milestone 3 profiling)
#include "r4d.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/model/attention/attn_kernels.h"
#include "r4dx/model/attention/paged_kv_cache.hpp"
#include "r4dx/model/attention/types.hpp"

namespace r4dx::model::attention {

class AttentionLayer {
 public:
  explicit AttentionLayer(const AttnConfig& cfg) : cfg_(cfg) {
    if (cfg_.hidden <= 0 || cfg_.num_heads <= 0 || cfg_.kv_heads <= 0 || cfg_.head_dim <= 0) {
      throw std::invalid_argument("AttentionLayer: all dims must be positive");
    }
    if (cfg_.num_heads % cfg_.kv_heads != 0) {
      throw std::invalid_argument("AttentionLayer: num_heads must be a multiple of kv_heads");
    }
    if (cfg_.rotary_dim <= 0 || cfg_.rotary_dim > cfg_.head_dim) {
      throw std::invalid_argument("AttentionLayer: rotary_dim must be in (0, head_dim]");
    }
  }

  // hidden_in, out: device [T, hidden] bf16. `out` must not alias `hidden_in` -- the residual add
  // at the end reads hidden_in as the residual term after every projection has already been
  // derived from it, and this implementation does not guarantee those projections finish before
  // an in-place write to hidden_in would be visible.
  // kv: this sequence's cache; must already be sized (PagedKvCache's max_context_tokens) for at
  // least start_pos+T tokens.
  // start_pos: number of tokens already resident in `kv` for this sequence before this call
  // (0 for a fresh prefill; the prior call's start_pos+T when continuing decode).
  // arena: bump-allocator scratch for every activation buffer this call needs (all bounded by
  // T<=64) -- no hipMalloc/hipFree on this call's path; the caller Reset()s it between layers.
  // positions: device int32[T], positions[t] == start_pos+t -- doubles as both the RoPE position
  // ids and the KV slot_mapping (this cache's contiguous block table makes slot==pos, so the two
  // are literally the same array; see PagedKvCache's doc comment). seqused_k: device int32[1] ==
  // start_pos+T. Both are the caller's persistent buffers (typically uploaded once per
  // Model::RunChunk call and shared by every full-attention layer in that chunk, since neither
  // depends on the layer, only on (start_pos, T)) -- NOT arena-allocated, because a value written
  // by a plain/async host upload into arena-reused bytes would race the still-in-flight kernels
  // from a previous layer's use of those same bytes (see gdn_layer.cpp's UploadArray comment for
  // the same hazard class); a caller-owned, non-arena buffer sidesteps that entirely.
  // R3 fusion (docs/r9700.md P2/R3), mirrors Mlp::Forward/GdnLayer::Forward's trailing params:
  // `x_normed_in` non-null skips this block's own input rmsnorm; `next_norm_weight` non-null
  // replaces the final residual add with r4dx_residual_rmsnorm_bf16(...), additionally writing the
  // normed sum into `x_normed_out` for this SAME layer's own Mlp to consume as its x_normed_in.
  // `prof` (Milestone 3 profiling pass, docs/r9700.md R5/Q3/Q7): non-null only under r4dx-cli
  // --profile -- see profile_span.h's file comment for the naming convention ("gemm:" prefix) and
  // the zero-overhead guarantee when nullptr (every real decode/prefill call site).
  void Forward(core::Arena& arena, const uint16_t* hidden_in, uint16_t* out, const AttnWeights& w,
               PagedKvCache& kv, int T, int start_pos, const int32_t* positions,
               const int32_t* seqused_k, hipStream_t stream, const uint16_t* x_normed_in = nullptr,
               const uint16_t* next_norm_weight = nullptr, uint16_t* x_normed_out = nullptr,
               SpanAccumulator* prof = nullptr) {
    if (T < 1 || T > 64) {
      throw std::invalid_argument(
          "AttentionLayer::Forward: T must be 1..64 (interim skinny-GEMM/paged-attention band)");
    }
    kv.CheckCapacity(start_pos, T);

    const int hidden = cfg_.hidden;
    const int H = cfg_.num_heads;
    const int Hkv = cfg_.kv_heads;
    const int D = cfg_.head_dim;
    const int gqa = cfg_.Gqa();

    // ---- input rmsnorm ------------------------------------------------------------------------
    // R3 (docs/r9700.md): skip this launch when the previous layer's Mlp already fused it.
    const uint16_t* normed = x_normed_in;
    uint16_t* normed_scratch = nullptr;
    if (normed == nullptr) {
      normed_scratch = arena.Alloc<uint16_t>(static_cast<size_t>(T) * hidden);
      ProfiledCall(prof, stream, "attn.rmsnorm", [&] {
        r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(hidden_in),
                           reinterpret_cast<int64_t>(w.input_layernorm),
                           reinterpret_cast<int64_t>(normed_scratch), T, hidden, cfg_.rms_eps,
                           reinterpret_cast<int64_t>(stream));
      });
      normed = normed_scratch;
    }

    // ---- fused q_proj + output gate, then split per-head-interleaved --------------------------
    // Dispatched through the shared r4dx::model::ApplyLinear (decode-perf pass, 2026-09-19) --
    // whichever layout Container::Load loaded `w.qg` as (bf16/mxfp4/w4a16/w4a8), same as GDN's
    // in_proj_qkv/out_proj and MLP's gate_up/down. Replaces this component's own bf16-only Linear
    // (attention/linear.hpp) for this weight -- that wrapper is still used below for k/v, which
    // have no quantized on-disk form.
    uint16_t* qg_raw = arena.Alloc<uint16_t>(static_cast<size_t>(T) * 2 * H * D);
    ProfiledCall(prof, stream, "gemm:attn.qg_proj", [&] {
      ApplyLinear(stream, arena, *w.qg, normed, qg_raw, T);
    });

    uint16_t* q = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D);
    uint16_t* gate = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D);
    ProfiledCall(prof, stream, "attn.split_qg", [&] {
      r4dx_model_attn_split_qg_bf16(reinterpret_cast<int64_t>(qg_raw), reinterpret_cast<int64_t>(q),
                                     reinterpret_cast<int64_t>(gate), T, H, D,
                                     reinterpret_cast<int64_t>(stream));
    });

    // ---- k / v projections ---------------------------------------------------------------------
    // Dispatched through the shared ApplyLinear (R1, docs/r9700.md), same as qg/o above -- k/v now
    // honor `--layout` too (Container::Load falls back to bf16 when the requested layout's tensors
    // are absent, e.g. mtp.attn.k/v, which are always bf16 by design). Replaces this component's
    // own bf16-only Linear wrapper (attention/linear.hpp), which is now unused.
    uint16_t* k = arena.Alloc<uint16_t>(static_cast<size_t>(T) * Hkv * D);
    uint16_t* v = arena.Alloc<uint16_t>(static_cast<size_t>(T) * Hkv * D);
    ProfiledCall(prof, stream, "gemm:attn.k_proj", [&] { ApplyLinear(stream, arena, *w.k, normed, k, T); });
    ProfiledCall(prof, stream, "gemm:attn.v_proj", [&] { ApplyLinear(stream, arena, *w.v, normed, v, T); });

    // ---- per-head q_norm / k_norm (RMSNorm over head_dim, one shared weight per head) ----------
    ProfiledCall(prof, stream, "attn.qk_norm", [&] {
      r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(q), reinterpret_cast<int64_t>(w.q_norm),
                         reinterpret_cast<int64_t>(q), static_cast<int64_t>(T) * H, D, cfg_.rms_eps,
                         reinterpret_cast<int64_t>(stream));
      r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(k), reinterpret_cast<int64_t>(w.k_norm),
                         reinterpret_cast<int64_t>(k), static_cast<int64_t>(T) * Hkv, D,
                         cfg_.rms_eps, reinterpret_cast<int64_t>(stream));
    });

    // ---- partial-rotary mrope (text-only: all three position streams == token position) --------
    ProfiledCall(prof, stream, "attn.rope", [&] {
      r4dx_rope_partial_mrope_bf16(reinterpret_cast<int64_t>(q), reinterpret_cast<int64_t>(k),
                                    reinterpret_cast<int64_t>(positions), T, H, Hkv, D,
                                    cfg_.rotary_dim, cfg_.rope_theta,
                                    reinterpret_cast<int64_t>(stream));
    });

    // ---- fp8 paged KV cache write (post-rope K, per docs/architecture.md) -- BEFORE the attn call
    // `positions` doubles as the slot_mapping (contiguous block table: slot==pos, see above).
    ProfiledCall(prof, stream, "attn.kv_write", [&] {
      r4dx_kv_write_paged_fp8_hnd(
          reinterpret_cast<int64_t>(k), reinterpret_cast<int64_t>(v),
          reinterpret_cast<int64_t>(positions), reinterpret_cast<int64_t>(w.k_descale),
          reinterpret_cast<int64_t>(w.v_descale), reinterpret_cast<int64_t>(kv.Data()), T, Hkv, D,
          kv.BlockSize(), kv.KvBlockStride(), kv.KvHeadStride(), reinterpret_cast<int64_t>(stream));
    });

    // ---- r4d attention: prefill for q_len beyond the decode/verify-window band, decode otherwise
    // (r4d.h: decode/split-KV kernel serves q_len*gqa<=64) ---------------------------------------
    uint16_t* attn_out = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D);

    R4DArgs a{};
    a.q = q;
    a.kv = kv.Data();
    a.block_table = kv.BlockTable();
    a.seqused_k = seqused_k;
    a.out = attn_out;
    // num_seqs==1: this writer's own [kv_heads] row coincides with R4DArgs' runtime
    // [num_seqs,kv_heads] broadcast table (r4dx::core::r4d::AttnDecodeFp8Kv's doc comment).
    a.k_descale = w.k_descale;
    a.v_descale = w.v_descale;
    a.q_descale = nullptr;
    a.num_seqs = 1;
    a.q_len = T;
    a.q_heads = H;
    a.kv_heads = Hkv;
    a.head_dim = D;
    a.block_size = kv.BlockSize();
    a.max_blocks = kv.MaxBlocks();
    a.kv_block_stride = kv.KvBlockStride();
    a.kv_head_stride = kv.KvHeadStride();
    a.scale = 1.0f / std::sqrt(static_cast<float>(D));
    a.splits = 0;
    a.max_ctx = kv.CapacityTokens();

    const int max_decode_q_len = 64 / gqa;  // r4d.h A_MAX_DECODE_ROWS=64 (q_len*gqa)
    if (T <= max_decode_q_len) {
      const int64_t scratch_bytes = r4dx::core::r4d::AttnDecodeScratchBytes(a);
      a.scratch = scratch_bytes > 0 ? arena.Alloc<uint8_t>(static_cast<size_t>(scratch_bytes))
                                     : nullptr;
      ProfiledCall(prof, stream, "attn.core_decode",
                   [&] { r4dx::core::r4d::AttnDecodeFp8Kv(a, stream); });
    } else {
      a.scratch = nullptr;
      ProfiledCall(prof, stream, "attn.core_prefill",
                   [&] { r4dx::core::r4d::AttnPrefillFp8Kv(a, stream); });
    }

    // ---- output gate: attn_out * sigmoid(gate) --------------------------------------------------
    uint16_t* gated = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D);
    ProfiledCall(prof, stream, "attn.gate_mul", [&] {
      r4dx_model_attn_gate_mul_bf16(reinterpret_cast<int64_t>(attn_out),
                                     reinterpret_cast<int64_t>(gate),
                                     reinterpret_cast<int64_t>(gated),
                                     static_cast<int64_t>(T) * H * D,
                                     reinterpret_cast<int64_t>(stream));
    });

    // ---- o_proj ----------------------------------------------------------------------------------
    uint16_t* o_out = arena.Alloc<uint16_t>(static_cast<size_t>(T) * hidden);
    ProfiledCall(prof, stream, "gemm:attn.o_proj", [&] {
      ApplyLinear(stream, arena, *w.o, gated, o_out, T);
    });

    // ---- residual add ------------------------------------------------------------------------
    ProfiledCall(prof, stream, "attn.residual", [&] {
      if (next_norm_weight != nullptr) {
        r4dx_residual_rmsnorm_bf16(reinterpret_cast<int64_t>(hidden_in),
                                    reinterpret_cast<int64_t>(o_out),
                                    reinterpret_cast<int64_t>(next_norm_weight),
                                    reinterpret_cast<int64_t>(out),
                                    reinterpret_cast<int64_t>(x_normed_out), T, hidden, cfg_.rms_eps,
                                    reinterpret_cast<int64_t>(stream));
      } else {
        r4dx_residual_add_bf16(reinterpret_cast<int64_t>(hidden_in), reinterpret_cast<int64_t>(o_out),
                                reinterpret_cast<int64_t>(out), static_cast<int64_t>(T) * hidden,
                                reinterpret_cast<int64_t>(stream));
      }
    });
  }

  const AttnConfig& Config() const { return cfg_; }

 private:
  AttnConfig cfg_;
};

}  // namespace r4dx::model::attention
