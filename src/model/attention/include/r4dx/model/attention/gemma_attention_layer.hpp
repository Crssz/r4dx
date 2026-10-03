// r4dx::model::attention::GemmaAttnLayer -- the attention half of one Gemma 4 decoder layer
// (docs/gemma4-plan.md 3.3-3.5, task M1-19; semantics: docs/gemma4-semantics.md). AttentionLayer (Qwen:
// q|gate split, 1/sqrt(D) scale, mrope, residual add inside) is not reused.
//
// What Forward computes, given the layer's already-normed input (`x_normed` = input_layernorm(x), produced by
// the caller / the previous layer's fused post-norm kernel) -- exactly the HF Gemma4 attention:
//   q = q_proj(x_normed);  k = k_proj(x_normed);  v = v_proj(x_normed)            (sliding layers)
//   v = k  (the RAW k_proj output, before k_norm and rope)                          (full layers: k_eq_v)
//   q = q_norm(q) (plain weight);  k = k_norm(k) (plain weight);  v = v_norm(v) (NO weight)   every layer
//   q, k = rope(q, k)       sliding: theta 1e4, all 256 dims; full: "proportional", theta 1e6, 64 of 512 dims
//   KV write (post-rope K, normed V) into the layer's cache, BEFORE the attention call
//   o = softmax(q k^T * 1.0 [causal, + sliding window]) v            (scale 1.0, no logit softcap)
//   [rotated container: o <- o Hb, one 256 block per head / half head, signs rotation.had_o{,_full}_signs]
//   o_out = o_proj(o)
// The sandwich post-norm, the residual add and layer_scalar are the CALLER's (one fused kernel per sublayer).
//
// Attention backend: the stage-0 reference kernel (attn_ref.h; exact, slow at long context, any geometry) by
// default; the libr4d sliding-window gqa2 kernels for sliding layers with fp8 KV when asked for. The full
// layers (head_dim 512, gqa 16) have no libr4d kernel yet (M1-33) and always use the reference.
//
// KV: GemmaKvCache -- fp8 e4m3 (static per-(layer, head) descales) or bf16, a contiguous paged cache for a
// full layer, a ring (one block table shared by every sliding layer) for a sliding one.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels/model_kernels.h"
#include "linear.h"
#include "profile_span.h"
#include "r4d.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/gemma_kernels.h"
#include "r4dx/kernels/kernels.h"
#include "r4dx/kernels/rotate_residual.h"
#include "r4dx/model/attention/attn_ref.h"
#include "r4dx/model/attention/sliding_ring.hpp"

namespace r4dx::model::attention {

enum class GemmaKvDtype { kFp8, kBf16 };
enum class GemmaAttnBackend { kReference, kLibr4d };

inline GemmaAttnBackend ParseGemmaAttnBackend(const char* e) {
  if (e == nullptr || *e == '\0' || std::strcmp(e, "ref") == 0 || std::strcmp(e, "reference") == 0) {
    return GemmaAttnBackend::kReference;
  }
  if (std::strcmp(e, "r4d") == 0 || std::strcmp(e, "libr4d") == 0) return GemmaAttnBackend::kLibr4d;
  std::fprintf(stderr, "r4dx: R4DX_GEMMA_ATTN='%s' not recognized (ref|r4d); using ref\n", e);
  return GemmaAttnBackend::kReference;
}

// One layer's KV cache. Layout (R4DArgs.kv's): (num_blocks, kv_heads, block_size, 2 * head_dim), K columns
// first, V after; strides in ELEMENTS of the cache dtype. ZERO-initialised (the windowed kernels stage tiles
// below the window: finite slots only).
class GemmaKvCache {
 public:
  static constexpr int kBlockSize = 16;

  // Full layer: contiguous blocks for `max_ctx` tokens, own identity block table.
  static GemmaKvCache Contiguous(int kv_heads, int head_dim, int64_t max_ctx, GemmaKvDtype dtype) {
    GemmaKvCache c(kv_heads, head_dim, dtype);
    const int64_t blocks = (max_ctx + kBlockSize - 1) / kBlockSize;
    c.blocks_ = blocks;
    std::vector<int32_t> t(static_cast<size_t>(blocks + SlidingRingGeometry::kTableSlackEntries));
    for (size_t i = 0; i < t.size(); ++i) t[i] = static_cast<int32_t>(std::min<int64_t>(static_cast<int64_t>(i), blocks - 1));
    c.own_table_.Resize(t.size());
    c.own_table_.CopyFromHost(t);
    c.table_ = c.own_table_.data();
    c.table_entries_ = static_cast<int>(t.size());
    c.Alloc();
    return c;
  }

  // Sliding layer: RingBlocks() blocks, addressed through `shared_table` (SlidingBlockTable::Data(), owned
  // by the model; T[i] = i % RingBlocks()).
  static GemmaKvCache Ring(int kv_heads, int head_dim, const SlidingRingGeometry& geo, const int32_t* shared_table,
                           int table_entries, GemmaKvDtype dtype) {
    GemmaKvCache c(kv_heads, head_dim, dtype);
    c.blocks_ = geo.RingBlocks();
    c.table_ = shared_table;
    c.table_entries_ = table_entries;
    c.ring_ = true;
    c.Alloc();
    return c;
  }

  GemmaKvCache(GemmaKvCache&&) = default;
  GemmaKvCache& operator=(GemmaKvCache&&) = default;

  uint8_t* Data() { return bytes_.data(); }
  const uint8_t* Data() const { return bytes_.data(); }
  const int32_t* BlockTable() const { return table_; }
  int TableEntries() const { return table_entries_; }
  int KvHeads() const { return kv_heads_; }
  int HeadDim() const { return head_dim_; }
  GemmaKvDtype Dtype() const { return dtype_; }
  bool IsRing() const { return ring_; }
  int64_t KvHeadStride() const { return static_cast<int64_t>(kBlockSize) * 2 * head_dim_; }
  int64_t KvBlockStride() const { return static_cast<int64_t>(kv_heads_) * KvHeadStride(); }
  size_t Bytes() const { return bytes_.bytes(); }
  int64_t Blocks() const { return blocks_; }
  // Whole-cache D2D copy (a ring checkpoint / restore); same geometry required.
  void CopyFrom(const GemmaKvCache& o) {
    if (o.bytes_.size() != bytes_.size()) throw std::invalid_argument("GemmaKvCache::CopyFrom: geometry mismatch");
    R4DX_HIP_CHECK(hipMemcpy(bytes_.data(), o.bytes_.data(), bytes_.size(), hipMemcpyDeviceToDevice));
  }

 private:
  GemmaKvCache(int kv_heads, int head_dim, GemmaKvDtype dtype) : kv_heads_(kv_heads), head_dim_(head_dim), dtype_(dtype) {}
  void Alloc() {
    const size_t elem = dtype_ == GemmaKvDtype::kBf16 ? 2 : 1;
    bytes_.Resize(static_cast<size_t>(blocks_) * static_cast<size_t>(KvBlockStride()) * elem);
    bytes_.Zero();
  }
  int kv_heads_, head_dim_;
  GemmaKvDtype dtype_;
  int64_t blocks_ = 0;
  bool ring_ = false;
  core::DeviceBuffer<uint8_t> bytes_;
  core::DeviceBuffer<int32_t> own_table_;
  const int32_t* table_ = nullptr;
  int table_entries_ = 0;
};

struct GemmaAttnConfig {
  int hidden = 3840;
  int heads = 16;
  int kv_heads = 8;
  int head_dim = 256;
  int rope_pairs = 128;    // rotated (i, i + head_dim/2) pairs
  int rope_freq_dim = 256;  // exponent denominator of the frequencies (the FULL head_dim)
  float rope_theta = 1.0e4f;
  float rms_eps = 1.0e-6f;
  int window = 1024;  // sliding window (keys incl. the query); 0 = full causal
  bool k_eq_v = false;  // full layers: V = raw k_proj output, no v_proj
  bool v_norm = true;   // v_norm (no weight) on every layer's V (semantics 1.1)
  int Gqa() const { return heads / kv_heads; }
};

struct GemmaAttnWeights {
  const QuantLinear* q = nullptr;
  const QuantLinear* k = nullptr;
  const QuantLinear* v = nullptr;  // null on a k_eq_v layer
  const QuantLinear* o = nullptr;
  const uint16_t* q_norm = nullptr;  // [head_dim] bf16, plain weight
  const uint16_t* k_norm = nullptr;
  const float* k_descale = nullptr;  // [kv_heads] fp32 (fp8 KV only)
  const float* v_descale = nullptr;
  // Rotated container (option A): rotation.had_o_signs (sliding) / had_o_full_signs (full), fp32 [heads*head_dim];
  // non-null => attn_out is Hadamard-rotated in blocks of `o_had_block` (256) before o_proj.
  const float* o_had_signs = nullptr;
  int o_had_block = 256;
};

class GemmaAttnLayer {
 public:
  GemmaAttnLayer(const GemmaAttnConfig& cfg, GemmaAttnBackend backend = GemmaAttnBackend::kReference)
      : cfg_(cfg), backend_(backend) {
    if (cfg_.heads <= 0 || cfg_.kv_heads <= 0 || cfg_.head_dim <= 0 || cfg_.heads % cfg_.kv_heads != 0) {
      throw std::invalid_argument("GemmaAttnLayer: bad head geometry");
    }
    if (cfg_.k_eq_v && cfg_.kv_heads <= 0) throw std::invalid_argument("GemmaAttnLayer: k_eq_v needs kv heads");
  }

  // x_normed: [T, hidden] bf16, input_layernorm already applied. o_out: [T, hidden] bf16 (the raw o_proj
  // output). positions: device int32 [T] (absolute rope positions). slots: device int32 [T] (KV write slots:
  // the positions themselves for a full layer, the ring slots for a sliding one). seqused_k: device int32 [1]
  // == start_pos + T (the libr4d path). start_pos: tokens already cached before this call. klimit_ext: device
  // int32 [T] or null (bidirectional image block rows; M2).
  void Forward(core::Arena& arena, const uint16_t* x_normed, uint16_t* o_out, const GemmaAttnWeights& w,
               GemmaKvCache& kv, int T, int start_pos, const int32_t* positions, const int32_t* slots,
               const int32_t* seqused_k, hipStream_t stream, SpanAccumulator* prof = nullptr,
               const int32_t* klimit_ext = nullptr, core::TpComm* comm = nullptr) {
    const int H = cfg_.heads, Hk = cfg_.kv_heads, D = cfg_.head_dim;
    if (T < 1) throw std::invalid_argument("GemmaAttnLayer::Forward: T must be >= 1");
    if (kv.KvHeads() != Hk || kv.HeadDim() != D) throw std::invalid_argument("GemmaAttnLayer::Forward: cache geometry mismatch");
    if (cfg_.k_eq_v != (w.v == nullptr)) throw std::invalid_argument("GemmaAttnLayer::Forward: v_proj presence != k_eq_v");
    const int64_t s = reinterpret_cast<int64_t>(stream);
    const auto P = [](const void* p) { return reinterpret_cast<int64_t>(p); };

    uint16_t* q = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D, 16);
    uint16_t* k = arena.Alloc<uint16_t>(static_cast<size_t>(T) * Hk * D, 16);
    uint16_t* v = arena.Alloc<uint16_t>(static_cast<size_t>(T) * Hk * D, 16);
    ProfiledCall(prof, stream, "gemm:attn.q_proj", [&] { ApplyLinear(stream, arena, *w.q, x_normed, q, T); });
    ProfiledCall(prof, stream, "gemm:attn.k_proj", [&] { ApplyLinear(stream, arena, *w.k, x_normed, k, T); });
    if (w.v != nullptr) {
      ProfiledCall(prof, stream, "gemm:attn.v_proj", [&] { ApplyLinear(stream, arena, *w.v, x_normed, v, T); });
    } else {
      // k_eq_v: V is the raw k_proj output, BEFORE the in-place k_norm below (semantics 1.1).
      R4DX_HIP_CHECK(hipMemcpyAsync(v, k, static_cast<size_t>(T) * Hk * D * sizeof(uint16_t), hipMemcpyDeviceToDevice, stream));
    }

    ProfiledCall(prof, stream, "attn.qkv_norm", [&] {
      r4dx_rmsnorm_plain_bf16(P(q), P(w.q_norm), P(q), static_cast<int64_t>(T) * H, D, cfg_.rms_eps, 0, s);
      r4dx_rmsnorm_plain_bf16(P(k), P(w.k_norm), P(k), static_cast<int64_t>(T) * Hk, D, cfg_.rms_eps, 0, s);
      if (cfg_.v_norm) r4dx_rmsnorm_noscale_bf16(P(v), P(v), static_cast<int64_t>(T) * Hk, D, cfg_.rms_eps, s);
    });
    ProfiledCall(prof, stream, "attn.rope", [&] {
      r4dx_rope_proportional_bf16(P(q), P(k), P(positions), T, H, Hk, D, cfg_.rope_pairs, cfg_.rope_freq_dim,
                                  cfg_.rope_theta, s);
    });

    const bool bf16kv = kv.Dtype() == GemmaKvDtype::kBf16;
    ProfiledCall(prof, stream, "attn.kv_write", [&] {
      if (bf16kv) {
        r4dx_model_kv_write_paged_bf16_hnd(P(k), P(v), P(slots), P(kv.Data()), T, Hk, D, GemmaKvCache::kBlockSize,
                                           kv.KvBlockStride(), kv.KvHeadStride(), s);
      } else {
        r4dx_kv_write_paged_fp8_hnd(P(k), P(v), P(slots), P(w.k_descale), P(w.v_descale), P(kv.Data()), T, Hk, D,
                                    GemmaKvCache::kBlockSize, kv.KvBlockStride(), kv.KvHeadStride(), s);
      }
    });

    uint16_t* attn_out = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D, 16);
    const int ctx = start_pos + T;
    const bool use_r4d = backend_ == GemmaAttnBackend::kLibr4d && !bf16kv && cfg_.window > 0 && D == 256 && cfg_.Gqa() == 2;
    if (use_r4d) {
      R4DArgsW a{};
      a.q = q;
      a.kv = kv.Data();
      a.block_table = kv.BlockTable();
      a.seqused_k = seqused_k;
      a.out = attn_out;
      a.k_descale = w.k_descale;
      a.v_descale = w.v_descale;
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
      a.scale = 1.0f;
      a.splits = 0;
      a.max_ctx = ctx;
      a.window = cfg_.window;
      a.klimit_ext = klimit_ext;
      // The split-KV decode kernel has no klimit_ext (prefill only, r4d.h): a short image chunk takes the
      // prefill kernel, which handles any q_len (rows past q_len are clamped and dropped).
      if (T <= 32 && klimit_ext == nullptr) {
        const int64_t sb = core::r4d::AttnDecodeWindowScratchBytes(a);
        a.scratch = sb > 0 ? arena.Alloc<uint8_t>(static_cast<size_t>(sb), 16) : nullptr;
        ProfiledCall(prof, stream, "attn.core_decode", [&] { core::r4d::AttnDecodeWindowFp8Kv(a, stream); });
      } else {
        ProfiledCall(prof, stream, "attn.core_prefill", [&] { core::r4d::AttnPrefillWindowFp8Kv(a, stream); });
      }
    } else {
      R4dxGemmaAttnRefArgs a{};
      a.q = P(q);
      a.kv = P(kv.Data());
      a.block_table = P(kv.BlockTable());
      a.out = P(attn_out);
      a.k_descale = bf16kv ? 0 : P(w.k_descale);
      a.v_descale = bf16kv ? 0 : P(w.v_descale);
      a.klimit_ext = P(klimit_ext);
      a.q_len = T;
      a.q_heads = H;
      a.kv_heads = Hk;
      a.head_dim = D;
      a.block_size = GemmaKvCache::kBlockSize;
      a.ctx = ctx;
      a.window = cfg_.window;
      a.scale = 1.0f;
      a.kv_block_stride = kv.KvBlockStride();
      a.kv_head_stride = kv.KvHeadStride();
      ProfiledCall(prof, stream, "attn.core_ref", [&] {
        const int rc = bf16kv ? r4dx_gemma_attn_ref_bf16kv(&a, s) : r4dx_gemma_attn_ref_fp8kv(&a, s);
        if (rc != 0) throw std::runtime_error("GemmaAttnLayer: r4dx_gemma_attn_ref returned " + std::to_string(rc));
      });
    }

    if (w.o_had_signs != nullptr) {
      // Rotated container: o_proj was folded W Hb (one 256 block per head / half head), so the attention
      // output is rotated in place first.
      ProfiledCall(prof, stream, "attn.o_hadamard", [&] {
        r4dx_hadamard_inplace_bf16(P(attn_out), T, static_cast<int64_t>(H) * D, P(w.o_had_signs), w.o_had_block, s);
      });
    }
    ProfiledCall(prof, stream, "gemm:attn.o_proj", [&] { ApplyLinear(stream, arena, *w.o, attn_out, o_out, T); });
    // Tensor parallel (docs/gemma4-plan.md M1b-1): o_proj is row-parallel (K = this rank's heads), so o_out is a partial
    // sum; the bf16 sublayer output is summed across ranks BEFORE the (fp32) post-norm and residual add.
    if (comm != nullptr) {
      ProfiledCall(prof, stream, "tp.allreduce", [&] { comm->AllReduceSumBf16Rows(o_out, T, cfg_.hidden, stream); });
    }
  }

  const GemmaAttnConfig& Config() const { return cfg_; }
  GemmaAttnBackend Backend() const { return backend_; }

 private:
  GemmaAttnConfig cfg_;
  GemmaAttnBackend backend_;
};

}  // namespace r4dx::model::attention
