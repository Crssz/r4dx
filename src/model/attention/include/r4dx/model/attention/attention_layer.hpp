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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#include "linear.h"  // r4dx::model::ApplyLinear -- shared qg/o/k/v quantized-linear dispatch
#include "profile_span.h"  // r4dx::model::SpanAccumulator / ProfiledCall (Milestone 3 profiling)
#include "r4d.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/decode_legacy.hpp"  // R4DX_DECODE_LEGACY=attn
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/model/attention/attn_kernels.h"
#include "r4dx/model/attention/paged_kv_cache.hpp"
#include "r4dx/model/attention/types.hpp"

namespace r4dx::model::attention {

// prefill M1 split law (docs/prefill.md): how many KV segments one prefill attention call gets at
// context `ctx` (= start_pos + T, the call's seqused_k). A pure function of (ctx, q_len, kv_heads),
// so every rank of a TP group, every rerun and every prefix-reuse replay of the same chunk makes the
// same choice.
//   - Below kPrefillSplitKvMinCtx: 1, i.e. the plain launch, so short and medium prompts stay
//     bit-identical to the unsplit runtime.
//   - Above it: enough segments that q_blocks x kv_heads x splits reaches kPrefillSplitKvTargetWgs
//     workgroups (the device's 32 WGPs each hold one of these 54 KB-LDS workgroups), rounded down to
//     a power of two, and never segments thinner than kPrefillSplitKvMinTiles 48-key tiles.
// At TP=2 kv_heads per rank is half, so the same law gives each rank twice the segments.
inline constexpr int kPrefillSplitKvMinCtx = 8192;
inline constexpr int kPrefillSplitKvTargetWgs = 32;
inline constexpr int kPrefillSplitKvMinTiles = 8;
// `min_ctx` (default kPrefillSplitKvMinCtx) is the depth threshold; R4DX_PREFILL_SPLITKV_MIN sets it.
inline int PrefillSplitKvSplits(int ctx, int q_len, int kv_heads,
                                int min_ctx = kPrefillSplitKvMinCtx) {
  if (ctx < min_ctx || q_len < 1 || kv_heads < 1) return 1;
  const int units = ((q_len + 63) / 64) * kv_heads;
  const int tiles = (ctx + 47) / 48;
  int s = 1;
  while (s * 2 * units <= kPrefillSplitKvTargetWgs && tiles / (s * 2) >= kPrefillSplitKvMinTiles) s *= 2;
  return s;
}

// R4DX_PREFILL_SPLITKV (read once per process) picks the prompt-prefill attention path:
//   - unset, empty or "exact" = kPrefillAttnExact, THE DEFAULT: every prompt-prefill call takes
//     the exact-wide launch (r4d_attn_prefill_exact_*), which is the plain launch's output bit for
//     bit over 8x the workgroups -- lossless by construction, about 2x on the attention call.
//   - "split" (or "splitkv" / "auto") = kPrefillAttnSplitLaw: split-KV by PrefillSplitKvSplits'
//     law (opt-in: 5-6x on the attention call, rounding-class drift vs the dense bits).
//   - "0", "1", "off" or "dense" = 1: never split, the plain single-workgroup-per-(q-block, kv-head)
//     launch (the pre-M1 prefill, bit for bit -- same bits as exact, just slower).
//   - N > 1 = exactly N split-KV segments on every prompt-prefill call, at any depth (A/B and
//     calibration runs), capped at 32 (the fp32 partials are 64 x q_heads x N x 1032 B of the
//     layer's 96 MiB arena).
//   - anything else: a warning on stderr, then the default (exact).
inline constexpr int kPrefillAttnSplitLaw = -1;
inline constexpr int kPrefillAttnExact = -2;
inline int ParsePrefillAttnMode(const char* e) {
  if (e == nullptr || *e == '\0' || std::strcmp(e, "exact") == 0) return kPrefillAttnExact;
  if (std::strcmp(e, "split") == 0 || std::strcmp(e, "splitkv") == 0 || std::strcmp(e, "auto") == 0) {
    return kPrefillAttnSplitLaw;
  }
  if (std::strcmp(e, "off") == 0 || std::strcmp(e, "dense") == 0) return 1;
  bool digits = true;
  for (const char* p = e; *p != '\0'; ++p) digits = digits && *p >= '0' && *p <= '9';
  if (!digits) {
    std::fprintf(stderr,
                 "r4dx: R4DX_PREFILL_SPLITKV='%s' not recognized (exact|split|off|dense|N); "
                 "using the default (exact)\n",
                 e);
    return kPrefillAttnExact;
  }
  const int n = std::atoi(e);
  return n < 1 ? 1 : (n > 32 ? 32 : n);
}
inline int PrefillSplitKvOverride() {
  static const int v = ParsePrefillAttnMode(std::getenv("R4DX_PREFILL_SPLITKV"));
  return v;
}

// R4DX_PREFILL_SPLITKV_MIN (read once per process; only read in the "split" law mode): the context
// depth, in tokens, at which the split law engages. "Context" is the call's seqused_k, i.e. the
// tokens already in the KV cache plus the call's own rows (start_pos + 64 (j + 1) for 64-row slice
// j), the same quantity the law's built-in 8192 is compared to. Per rank at TP=2 (the same depth
// check, the law then gives that rank's kv_heads their segments).
//   - unset or empty = kPrefillSplitKvMinUnset (-1): today's behaviour, threshold 8192 and the plain
//     single-workgroup launch below it.
//   - a non-negative integer N: threshold N, and every call below N (and any call where the law
//     returns 1) takes the exact-wide launch instead of the plain one. Exact-wide is the plain
//     launch's output bit for bit and about 2.3x faster per call, so this only changes speed.
//   - anything else: a warning on stderr, then unset.
inline constexpr int kPrefillSplitKvMinUnset = -1;
inline int ParsePrefillSplitKvMin(const char* e) {
  if (e == nullptr || *e == '\0') return kPrefillSplitKvMinUnset;
  long long n = 0;
  for (const char* p = e; *p != '\0'; ++p) {
    if (*p < '0' || *p > '9' || n > 100000000LL) {
      std::fprintf(stderr,
                   "r4dx: R4DX_PREFILL_SPLITKV_MIN='%s' not a token count; using the default "
                   "(%d, plain launch below it)\n",
                   e, kPrefillSplitKvMinCtx);
      return kPrefillSplitKvMinUnset;
    }
    n = n * 10 + (*p - '0');
  }
  return static_cast<int>(n);
}
inline int PrefillSplitKvMinOverride() {
  static const int v = ParsePrefillSplitKvMin(std::getenv("R4DX_PREFILL_SPLITKV_MIN"));
  return v;
}

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
  // R2/P2 (docs/r9700.md), appended after `prof`, mirrors GdnLayer::Forward's identical trailing
  // params: `x_normed_pre_epilogue`/`_data` reuse a fused cast epilogue the producer of
  // `x_normed_in` already computed (skipping this call's own qg/k/v cast launches when the format
  // matches, checked per-weight -- see k_shares_pre/v_shares_pre below); `next_epilogue`/
  // `next_epilogue_out` request this call's own residual+rmsnorm epilogue also emit x_normed_out's
  // fused cast epilogue for the immediately-following Mlp.
  //
  // `rope_pos3` (vision milestone, docs/vision.md "Text-side splicing"): device int32[3, T]
  // (contiguous, t row then h then w) giving this chunk's 3-axis mrope ROPE positions, which a
  // prompt containing an image makes DIFFERENT from `positions`. `positions` keeps its original
  // meaning in that case -- it is the KV slot_mapping and the paged cache's sequence index, both
  // of which must stay the plain running token index -- and only the rope call switches to
  // r4dx_rope_partial_mrope3_bf16. nullptr (the default, and every text-only caller) keeps the
  // pre-vision single-row path, byte for byte.
  //
  // `prefill_split_kv` (prefill M1, docs/prefill.md): true only from a PROMPT prefill chunk
  // (Model::RunChunk on its prefill path, Model::PrefillProfiled). It lets the prefill-kernel branch
  // below take the R4DX_PREFILL_SPLITKV path (default: exact-wide, the plain launch's bits; opt-in:
  // split-KV); false -- the default, and every decode, verify-window (MTP/DFlash), MTP-priming and
  // test caller -- keeps the plain launch, bit for bit.
  void Forward(core::Arena& arena, const uint16_t* hidden_in, uint16_t* out, const AttnWeights& w,
               PagedKvCache& kv, int T, int start_pos, const int32_t* positions,
               const int32_t* seqused_k, hipStream_t stream, const uint16_t* x_normed_in = nullptr,
               const uint16_t* next_norm_weight = nullptr, uint16_t* x_normed_out = nullptr,
               SpanAccumulator* prof = nullptr, int x_normed_pre_epilogue = 0,
               const void* x_normed_pre_data = nullptr, int next_epilogue = 0,
               void* next_epilogue_out = nullptr, const int32_t* rope_pos3 = nullptr,
               bool prefill_split_kv = false, int attn_slice = 0,
               const int32_t* seqused_k_slices = nullptr) {
    // `attn_slice` (256-row prefill chunk, Model's R4DX_PREFILL_CHUNK=256, docs/trellis-m256.md): 0 --
    // the default, every caller before it -- is the T = 1..64 call as always. 64: T is a multiple of 64
    // (up to 256) and everything row-independent (norms, the linears, split_qg, qk-norm, rope, the KV
    // write, the gate, the residual) sees all T rows -- each kernel gives a row the same bits at any T --
    // while the attention core runs once per 64-row sub-slice, in order: sub-slice j is exactly the launch
    // a 64-row chunk at start_pos + 64 j makes (q_len 64, its own causal context). Its keys are already
    // in the cache (the KV write covers all T rows first), and the causal mask hides the later
    // sub-slices' rows from it, so the bytes equal T / 64 consecutive 64-row calls.
    // `seqused_k_slices`: device int32 [T / 64], entry j = start_pos + 64 (j + 1), the caller's persistent
    // buffer like `seqused_k`. `positions` then has T entries.
    const bool sliced = attn_slice > 0 && T > attn_slice;
    if (sliced) {
      if (attn_slice != 64 || T % 64 != 0 || T > 256 || seqused_k_slices == nullptr) {
        throw std::invalid_argument(
            "AttentionLayer::Forward: attn_slice must be 64, T a multiple of 64 up to 256, and "
            "seqused_k_slices set");
      }
    } else if (T < 1 || T > 64) {
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
    // R2/P2: when THIS call computes its own rmsnorm (x_normed_in==nullptr), it can fuse w.qg's
    // cast epilogue into the same launch (self-contained, like GdnLayer/Mlp's identical comment)
    // -- and since `normed` also feeds w.k/w.v below at the SAME K=hidden, the one fused buffer is
    // reused for all three GEMMs whenever they agree on layout (checked per-weight below, not
    // assumed, since Container::LoadQuantLinearWithFallback can fall an individual tensor back to
    // bf16 independently of its siblings -- see gdn_layer.cpp's identical z_shares_pre comment).
    const uint16_t* normed = x_normed_in;
    uint16_t* normed_scratch = nullptr;
    int qg_epilogue = x_normed_pre_epilogue;
    const void* qg_pre_data = x_normed_pre_data;
    // Cross-boundary reuse only valid if it matches THIS layer's own w.qg layout -- see
    // gdn_layer.cpp's identical defensive comment.
    if (normed != nullptr && qg_epilogue != EpilogueForLayout(w.qg->layout)) {
      qg_epilogue = r4dx_epilogue_none;
      qg_pre_data = nullptr;
    }
    if (normed == nullptr) {
      qg_epilogue = EpilogueForLayout(w.qg->layout);
      normed_scratch = arena.Alloc<uint16_t>(static_cast<size_t>(T) * hidden);
      uint8_t* qg_pre_data_local = nullptr;
      if (qg_epilogue != r4dx_epilogue_none) {
        // f16: 2 bytes per element. 16-byte alignment (not uint8_t's default 1): a GEMM's
        // A-operand read is a global_load_b64 fragment read -- an epilogue buffer allocated later
        // in a layer's own scratch sequence (after GdnLayer/AttentionLayer/Mlp's other
        // T/H/K/V-shaped allocations, several of which are not multiples of 8 bytes) is not
        // guaranteed to be 8-aligned, and an unaligned wide GPU load silently reads garbage rather
        // than faulting. Found by a real generated-text divergence (docs/status.md's R2/P2
        // section), not by any tolerance-based golden test.
        qg_pre_data_local =
            arena.Alloc<uint8_t>(static_cast<size_t>(T) * hidden * 2, /*align_bytes=*/16);
      }
      ProfiledCall(prof, stream, "attn.rmsnorm", [&] {
        r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(hidden_in),
                           reinterpret_cast<int64_t>(w.input_layernorm),
                           reinterpret_cast<int64_t>(normed_scratch), T, hidden, cfg_.rms_eps,
                           reinterpret_cast<int64_t>(stream), qg_epilogue,
                           reinterpret_cast<int64_t>(qg_pre_data_local));
      });
      normed = normed_scratch;
      qg_pre_data = qg_pre_data_local;
    }
    const bool have_qg_pre = qg_epilogue != r4dx_epilogue_none;
    PreQuantizedActivation normed_pre{qg_epilogue, qg_pre_data};

    // ---- fused q_proj + output gate, then split per-head-interleaved --------------------------
    // Dispatched through the shared r4dx::model::ApplyLinear (decode-perf pass, 2026-09-19) --
    // whichever layout Container::Load loaded `w.qg` as (bf16/w4a16/trellis), same as GDN's
    // in_proj_qkv/out_proj and MLP's gate_up/down. Replaces this component's own bf16-only Linear
    // (attention/linear.hpp) for this weight -- that wrapper is still used below for k/v, which
    // have no quantized on-disk form.
    // docs/trellis-kernel.md 4.9 / 5.4 (M5): trellis qg, k and v read the same `normed`, so one
    // input transform (nout = 3) serves all three -- each output the same bytes as the linear's own
    // -- inside qg's span, where qg's own transform would have been; k and v run none.
    PreQuantizedActivation trellis_in[3];
    const QuantLinear* const trellis_in_w[3] = {w.qg, w.k, w.v};
    bool trellis_shared = false;
    uint16_t* qg_raw = arena.Alloc<uint16_t>(static_cast<size_t>(T) * 2 * H * D);
    ProfiledCall(prof, stream, "gemm:attn.qg_proj", [&] {
      trellis_shared = !have_qg_pre &&
                       SharedTrellisInput(stream, arena, normed, T, trellis_in_w, 3, trellis_in);
      ApplyLinear(stream, arena, *w.qg, normed, qg_raw, T,
                  trellis_shared ? &trellis_in[0] : have_qg_pre ? &normed_pre : nullptr);
    });

    uint16_t* q = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D);
    uint16_t* gate = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D);
    uint16_t* k = arena.Alloc<uint16_t>(static_cast<size_t>(T) * Hkv * D);
    uint16_t* v = arena.Alloc<uint16_t>(static_cast<size_t>(T) * Hkv * D);
    // decode-t1 item 4 (docs/perf.md): split_qg + q/k norm + rope + the fp8 cache write as ONE launch
    // after the k/v projections (same bytes, r4dx_attn_precore_bf16's comment) -- on the single-row
    // rope path (no image in the prompt) wherever the replaced norm kernel took its vector path.
    // R4DX_DECODE_LEGACY=attn keeps the five launches.
    const bool precore_fused =
        rope_pos3 == nullptr && !core::DecodeLegacy(core::DecodeItem::kAttn) &&
        r4dx_attn_precore_supported(D, cfg_.rotary_dim, reinterpret_cast<int64_t>(qg_raw),
                                     reinterpret_cast<int64_t>(k), reinterpret_cast<int64_t>(w.q_norm),
                                     reinterpret_cast<int64_t>(w.k_norm), reinterpret_cast<int64_t>(q),
                                     reinterpret_cast<int64_t>(gate)) != 0;
    if (!precore_fused) {
      ProfiledCall(prof, stream, "attn.split_qg", [&] {
        r4dx_model_attn_split_qg_bf16(reinterpret_cast<int64_t>(qg_raw), reinterpret_cast<int64_t>(q),
                                       reinterpret_cast<int64_t>(gate), T, H, D,
                                       reinterpret_cast<int64_t>(stream));
      });
    }

    // ---- k / v projections ---------------------------------------------------------------------
    // Dispatched through the shared ApplyLinear (R1, docs/r9700.md), same as qg/o above -- k/v now
    // honor `--layout` too (Container::Load falls back to bf16 when the requested layout's tensors
    // are absent, e.g. mtp.attn.k/v, which are always bf16 by design). Replaces this component's
    // own bf16-only Linear wrapper (attention/linear.hpp), which is now unused.
    const bool k_shares_pre = have_qg_pre && (EpilogueForLayout(w.k->layout) == qg_epilogue);
    const bool v_shares_pre = have_qg_pre && (EpilogueForLayout(w.v->layout) == qg_epilogue);
    ProfiledCall(prof, stream, "gemm:attn.k_proj", [&] {
      ApplyLinear(stream, arena, *w.k, normed, k, T,
                  trellis_shared ? &trellis_in[1] : k_shares_pre ? &normed_pre : nullptr);
    });
    ProfiledCall(prof, stream, "gemm:attn.v_proj", [&] {
      ApplyLinear(stream, arena, *w.v, normed, v, T,
                  trellis_shared ? &trellis_in[2] : v_shares_pre ? &normed_pre : nullptr);
    });

    if (precore_fused) {
      // split_qg's q / gate, q_norm / k_norm, the single-row rope on q and k, and the fp8 cache write
      // of k and v -- one launch (see the comment at `precore_fused`). `positions` is both the rope
      // position and the slot mapping, as in the five-launch chain below.
      ProfiledCall(prof, stream, "attn.precore", [&] {
        r4dx_attn_precore_bf16(reinterpret_cast<int64_t>(qg_raw), reinterpret_cast<int64_t>(k),
                                reinterpret_cast<int64_t>(v), reinterpret_cast<int64_t>(w.q_norm),
                                reinterpret_cast<int64_t>(w.k_norm), reinterpret_cast<int64_t>(positions),
                                reinterpret_cast<int64_t>(q), reinterpret_cast<int64_t>(gate),
                                reinterpret_cast<int64_t>(w.k_descale),
                                reinterpret_cast<int64_t>(w.v_descale),
                                reinterpret_cast<int64_t>(kv.Data()), T, H, Hkv, D, cfg_.rotary_dim,
                                cfg_.rope_theta, cfg_.rms_eps, kv.BlockSize(), kv.KvBlockStride(),
                                kv.KvHeadStride(), reinterpret_cast<int64_t>(stream));
      });
    }

    // ---- per-head q_norm / k_norm (RMSNorm over head_dim, one shared weight per head) ----------
    if (!precore_fused) ProfiledCall(prof, stream, "attn.qk_norm", [&] {
      r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(q), reinterpret_cast<int64_t>(w.q_norm),
                         reinterpret_cast<int64_t>(q), static_cast<int64_t>(T) * H, D, cfg_.rms_eps,
                         reinterpret_cast<int64_t>(stream));
      r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(k), reinterpret_cast<int64_t>(w.k_norm),
                         reinterpret_cast<int64_t>(k), static_cast<int64_t>(T) * Hkv, D,
                         cfg_.rms_eps, reinterpret_cast<int64_t>(stream));
    });

    // ---- partial-rotary mrope ------------------------------------------------------------------
    // Text-only (rope_pos3 == nullptr): all three position streams equal the token position, so the
    // single-row entry point is exactly equivalent and is what every pre-vision caller keeps using.
    // Multimodal: the (t,h,w) rows diverge from `positions` -- see this method's doc comment.
    if (!precore_fused) ProfiledCall(prof, stream, "attn.rope", [&] {
      if (rope_pos3 != nullptr) {
        r4dx_rope_partial_mrope3_bf16(reinterpret_cast<int64_t>(q), reinterpret_cast<int64_t>(k),
                                       reinterpret_cast<int64_t>(rope_pos3), T, H, Hkv, D,
                                       cfg_.rotary_dim, cfg_.rope_theta, cfg_.mrope_section_t,
                                       cfg_.mrope_section_h, cfg_.mrope_section_w,
                                       reinterpret_cast<int64_t>(stream));
      } else {
        r4dx_rope_partial_mrope_bf16(reinterpret_cast<int64_t>(q), reinterpret_cast<int64_t>(k),
                                      reinterpret_cast<int64_t>(positions), T, H, Hkv, D,
                                      cfg_.rotary_dim, cfg_.rope_theta,
                                      reinterpret_cast<int64_t>(stream));
      }
    });

    // ---- fp8 paged KV cache write (post-rope K, per docs/architecture.md) -- BEFORE the attn call
    // `positions` doubles as the slot_mapping (contiguous block table: slot==pos, see above).
    if (!precore_fused) ProfiledCall(prof, stream, "attn.kv_write", [&] {
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
      // align_bytes=16 (review finding, 2026-09-20): this feeds third_party/libr4d's own attention
      // decode kernel, which reads it with wide (16-byte) vector loads -- same alignment the P2
      // epilogue buffers above (line ~145) already pass explicitly. Every allocation preceding this
      // one in a layer happens to land 16-aligned today (hidden/conv_dim/intermediate are all
      // multiples of 16 at 2 bytes/element), so this worked by incidental arithmetic rather than
      // any enforced guarantee; matching the epilogue buffers' explicit alignment removes that
      // dependency.
      a.scratch = scratch_bytes > 0
                      ? arena.Alloc<uint8_t>(static_cast<size_t>(scratch_bytes), /*align_bytes=*/16)
                      : nullptr;
      ProfiledCall(prof, stream, "attn.core_decode",
                   [&] { r4dx::core::r4d::AttnDecodeFp8Kv(a, stream); });
    } else {
      // prefill M1: a prompt-prefill chunk takes the exact-wide launch by default (the plain
      // launch's bits); split-KV (PrefillSplitKvSplits) and the plain launch are opt-in via
      // R4DX_PREFILL_SPLITKV. Every non-prefill caller takes the plain launch.
      // Sliced (attn_slice): one launch per 64-row sub-slice j, whose call is the 64-row chunk's own
      // (q_len 64, ctx start_pos + 64 (j + 1), so a split law sees what it would see chunk by chunk).
      const int forced = prefill_split_kv ? PrefillSplitKvOverride() : 1;
      const int slice_rows = sliced ? attn_slice : T;
      const int n_slices = T / slice_rows;
      // splits_of(j): > 1 split-KV segments, 1 the plain launch, 0 the exact-wide launch (only with
      // R4DX_PREFILL_SPLITKV_MIN set: the calls the split law leaves unsplit run exact-wide).
      const int min_env = (prefill_split_kv && forced == kPrefillAttnSplitLaw) ? PrefillSplitKvMinOverride()
                                                                              : kPrefillSplitKvMinUnset;
      const auto splits_of = [&](int j) {
        if (!prefill_split_kv) return 1;
        if (forced > 0) return forced;
        if (forced == kPrefillAttnExact) return 1;
        const int ctx = start_pos + slice_rows * (j + 1);
        if (min_env == kPrefillSplitKvMinUnset) return PrefillSplitKvSplits(ctx, slice_rows, Hkv);
        const int s = PrefillSplitKvSplits(ctx, slice_rows, Hkv, min_env);
        return s > 1 ? s : 0;
      };
      a.scratch = nullptr;
      a.q_len = slice_rows;
      if (forced != kPrefillAttnExact) {
        int max_splits = 1;
        for (int j = 0; j < n_slices; ++j) max_splits = std::max(max_splits, splits_of(j));
        if (max_splits > 1) {  // one scratch for every sub-slice (each uses its own, smaller or equal, count)
          a.splits = max_splits;
          const int64_t bytes = r4dx::core::r4d::AttnPrefillSplitKvScratchBytes(a);
          a.scratch = arena.Alloc<uint8_t>(static_cast<size_t>(bytes), /*align_bytes=*/256);
        }
      }
      for (int j = 0; j < n_slices; ++j) {
        a.q = q + static_cast<size_t>(j) * slice_rows * H * D;
        a.out = attn_out + static_cast<size_t>(j) * slice_rows * H * D;
        a.seqused_k = sliced ? seqused_k_slices + j : seqused_k;
        const int splits = splits_of(j);
        if (forced == kPrefillAttnExact || splits == 0) {
          a.splits = 0;  // the library's default exact-wide geometry
          ProfiledCall(prof, stream, "attn.core_prefill",
                       [&] { r4dx::core::r4d::AttnPrefillExactFp8Kv(a, stream); });
        } else if (splits > 1) {
          a.splits = splits;
          ProfiledCall(prof, stream, "attn.core_prefill",
                       [&] { r4dx::core::r4d::AttnPrefillSplitKvFp8Kv(a, stream); });
        } else {
          a.splits = 0;
          ProfiledCall(prof, stream, "attn.core_prefill",
                       [&] { r4dx::core::r4d::AttnPrefillFp8Kv(a, stream); });
        }
      }
    }

    // ---- output gate: attn_out * sigmoid(gate) --------------------------------------------------
    uint16_t* gated = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D);
    // docs/trellis-kernel.md 4.8 / 5.4 (M5): a trellis o_proj takes the gate-mul's product straight
    // into its input transform (r4dx_attn_gate_mul_trellis_bf16: the same bf16 product, then the
    // same transform, so the same bytes as the gate-mul and ApplyLinear's own transform) -- one
    // launch, and `gated` is never written.
    PreQuantizedActivation o_pre;
    const bool o_fused = w.o_had_signs == nullptr && TrellisFusionEnabled() &&
                         w.o->layout == Layout::kTrellis && w.o->trellis_parts == 1 &&
                         w.o->K == static_cast<int64_t>(H) * D;
    if (o_fused) {
      uint16_t* a_o = arena.Alloc<uint16_t>(static_cast<size_t>(T) * H * D, /*align_bytes=*/16);
      ProfiledCall(prof, stream, "attn.gate_mul_trellis", [&] {
        r4dx_attn_gate_mul_trellis_bf16(reinterpret_cast<int64_t>(attn_out),
                                        reinterpret_cast<int64_t>(gate), T,
                                        static_cast<int64_t>(H) * D,
                                        reinterpret_cast<int64_t>(w.o->trellis_suh.data()),
                                        reinterpret_cast<int64_t>(a_o), w.o->trellis_prescale_log2,
                                        reinterpret_cast<int64_t>(stream));
      });
      o_pre = PreQuantizedActivation{r4dx_epilogue_none, a_o, w.o->trellis_suh.data(), 0};
    } else if (w.o_had_signs != nullptr) {
      // quant2 Q2b (docs/quant2.md section 4): gated = (attn_out * sigmoid(gate)) Hb, one launch,
      // block = D (one head); o_proj's K order is head * D + d, the row layout of [T, H, D].
      ProfiledCall(prof, stream, "attn.gate_mul_hadamard", [&] {
        r4dx_model_attn_gate_mul_hadamard_bf16(
            reinterpret_cast<int64_t>(attn_out), reinterpret_cast<int64_t>(gate),
            reinterpret_cast<int64_t>(gated), static_cast<int64_t>(T) * H * D,
            reinterpret_cast<int64_t>(stream), reinterpret_cast<int64_t>(w.o_had_signs), D,
            static_cast<int64_t>(H) * D);
      });
    } else {
      ProfiledCall(prof, stream, "attn.gate_mul", [&] {
        r4dx_model_attn_gate_mul_bf16(reinterpret_cast<int64_t>(attn_out),
                                       reinterpret_cast<int64_t>(gate),
                                       reinterpret_cast<int64_t>(gated),
                                       static_cast<int64_t>(T) * H * D,
                                       reinterpret_cast<int64_t>(stream));
      });
    }

    // ---- o_proj ----------------------------------------------------------------------------------
    uint16_t* o_out = arena.Alloc<uint16_t>(static_cast<size_t>(T) * hidden);
    ProfiledCall(prof, stream, "gemm:attn.o_proj", [&] {
      ApplyLinear(stream, arena, *w.o, gated, o_out, T, o_fused ? &o_pre : nullptr);
    });
    // Tensor parallel (docs/tp.md 6.2, site A2): o_proj is row-parallel, so each rank holds a
    // partial sum of the output; sum it across ranks before the residual (and the fused next-norm
    // epilogue) consumes it. Absent at TP=1 (comm == nullptr).
    if (cfg_.comm != nullptr) {
      ProfiledCall(prof, stream, "tp.allreduce", [&] {
        cfg_.comm->AllReduceSumBf16Rows(o_out, static_cast<int64_t>(T), static_cast<int64_t>(hidden), stream);
      });
    }

    // ---- residual add ------------------------------------------------------------------------
    ProfiledCall(prof, stream, "attn.residual", [&] {
      if (next_norm_weight != nullptr) {
        r4dx_residual_rmsnorm_bf16(reinterpret_cast<int64_t>(hidden_in),
                                    reinterpret_cast<int64_t>(o_out),
                                    reinterpret_cast<int64_t>(next_norm_weight),
                                    reinterpret_cast<int64_t>(out),
                                    reinterpret_cast<int64_t>(x_normed_out), T, hidden, cfg_.rms_eps,
                                    reinterpret_cast<int64_t>(stream), next_epilogue,
                                    reinterpret_cast<int64_t>(next_epilogue_out));
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
