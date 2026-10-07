#include "gdn_layer.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "linear.h"
#include "prefill_chunk.h"  // kGdnConvV1 / kGdnConvV2 (GdnLayerParams::conv_prep)
#include "profile_span.h"
#include "r4d.h"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/core/tp_comm.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/kernels/rotate_residual.h"  // r4dx_hadamard_inplace_bf16 (quant2 Q2b)

namespace r4dx::model {

namespace {

// softplus's large-x cutoff, matching r4d_gdn_conv_w4_h128_bf16.hip / r4d_gdn_recurrent_update_*
// .hip's own `cp_softplus`/`ru_softplus` and tests/kernels/test_gdn_chunk_scan.cpp's CPU reference
// (all three use 20.0f).
constexpr float kSoftplusThr = 20.0f;
constexpr int kGdnActSilu = 0;  // r4d_gdn_gated_rmsnorm_h128_bf16 / recurrent_update's act code

}  // namespace

void GdnLayer::Forward(core::Stream& stream, core::Arena& arena, GdnStateManager& states,
                        GdnControlCache& control, const uint16_t* x, uint16_t* x_out, int64_t T,
                        const GdnLayerParams& p, const uint16_t* x_normed_in,
                        const uint16_t* next_norm_weight, uint16_t* x_normed_out,
                        SpanAccumulator* prof, int x_normed_pre_epilogue,
                        const void* x_normed_pre_data, int next_epilogue,
                        void* next_epilogue_out) {
  const int64_t hidden = cfg_.hidden_size;
  const int Hg = static_cast<int>(cfg_.linear_num_key_heads);     // 16
  const int H = static_cast<int>(cfg_.linear_num_value_heads);    // 48
  const int K = static_cast<int>(cfg_.linear_key_head_dim);       // 128
  const int V = static_cast<int>(cfg_.linear_value_head_dim);     // 128
  const int width = static_cast<int>(cfg_.linear_conv_kernel_dim);  // 4
  const int64_t value_dim = cfg_.ValueDim();
  const int64_t conv_dim = cfg_.ConvDim();
  const float scale = 1.0f / std::sqrt(static_cast<float>(K));
  const hipStream_t s = stream.get();
  const float eps = static_cast<float>(cfg_.rms_norm_eps);

  // ---- input rmsnorm --------------------------------------------------------------------------
  // R3 (docs/r9700.md): skip this launch when the previous layer's Mlp already fused it.
  // R2/P2: two ways x_normed's fused cast epilogue gets populated -- (a) x_normed_in==nullptr:
  // this call computes its own rmsnorm, and knows w_.in_proj_qkv/w_.in_proj_z locally, both sharing
  // ONE container-wide body layout (model.h's `layout` field), so ONE fused epilogue serves BOTH
  // consumers below; (b) x_normed_in!=nullptr: the previous layer's Mlp already fused an epilogue
  // (matching THIS Model's body layout) into x_normed_pre_epilogue/_data when it produced
  // x_normed_in -- reused here directly, no local cast launch needed either way. Either way,
  // in_proj_pre below is validated per-weight (z_shares_pre) before use, never assumed.
  const uint16_t* x_normed = x_normed_in;
  uint16_t* x_normed_scratch = nullptr;
  int in_proj_epilogue = x_normed_pre_epilogue;
  const void* in_proj_pre_data = x_normed_pre_data;
  // Cross-boundary reuse is only valid if the producer's epilogue matches THIS layer's own
  // in_proj_qkv layout (LoadQuantLinearWithFallback can fall an individual tensor back to bf16
  // independently of its container-wide sibling layout -- do not assume Model.cpp's caller-side
  // choice of next_epilogue already accounts for that; verify locally, same defensive pattern as
  // z_shares_pre below).
  if (x_normed != nullptr && in_proj_epilogue != EpilogueForLayout(w_.in_proj_qkv.layout)) {
    in_proj_epilogue = r4dx_epilogue_none;
    in_proj_pre_data = nullptr;
  }
  if (x_normed == nullptr) {
    in_proj_epilogue = EpilogueForLayout(w_.in_proj_qkv.layout);
    x_normed_scratch = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
    uint8_t* in_proj_pre_data_local = nullptr;
    if (in_proj_epilogue != r4dx_epilogue_none) {
      // f16: 2 bytes per element. 16-byte alignment: see attention_layer.hpp's identical comment
      // (a GEMM does a global_load_b64 fragment read; uint8_t's default 1-byte arena alignment
      // does not guarantee that for a buffer allocated this late in a layer's scratch sequence).
      in_proj_pre_data_local =
          arena.Alloc<uint8_t>(static_cast<size_t>(T * hidden * 2), /*align_bytes=*/16);
    }
    ProfiledCall(prof, s, "gdn.rmsnorm", [&] {
      r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(x),
                         reinterpret_cast<int64_t>(input_layernorm_.data()),
                         reinterpret_cast<int64_t>(x_normed_scratch), T, hidden, eps,
                         reinterpret_cast<int64_t>(s), in_proj_epilogue,
                         reinterpret_cast<int64_t>(in_proj_pre_data_local));
    });
    x_normed = x_normed_scratch;
    in_proj_pre_data = in_proj_pre_data_local;
  }
  const bool have_in_proj_pre = in_proj_epilogue != r4dx_epilogue_none;
  PreQuantizedActivation in_proj_pre{in_proj_epilogue, in_proj_pre_data};

  // ---- in_proj_qkv / in_proj_z / in_proj_b / in_proj_a -----------------------------------------
  // docs/trellis-kernel.md 4.9 / 5.4 (M5): trellis in_proj_qkv and in_proj_z read the same x_normed,
  // so one input transform (nout = 2) serves both -- each output the same bytes as the linear's own
  // -- and z's ApplyLinear runs no transform of its own. It runs inside qkv's span, where
  // ApplyLinear's own transform would have been.
  PreQuantizedActivation trellis_in[2];
  const QuantLinear* const trellis_in_w[2] = {&w_.in_proj_qkv, &w_.in_proj_z};
  bool trellis_shared = false;
  uint16_t* mixed_qkv = arena.Alloc<uint16_t>(static_cast<size_t>(T * conv_dim));
  ProfiledCall(prof, s, "gemm:gdn.in_proj_qkv", [&] {
    trellis_shared = !have_in_proj_pre &&
                     SharedTrellisInput(s, arena, x_normed, T, trellis_in_w, 2, trellis_in);
    ApplyLinear(stream, arena, w_.in_proj_qkv, x_normed, mixed_qkv, T,
                trellis_shared ? &trellis_in[0] : have_in_proj_pre ? &in_proj_pre : nullptr);
  });
  // in_proj_a/in_proj_b are plain bf16 linears (never quantized -- docs/r9700.md R1: "too small to
  // matter, feed the decay path -- leave them bf16"), so they go straight through the bf16 GEMM
  // rather than through ApplyLinear's QuantLinear dispatch.
  // The bf16 GEMM takes at most 64 rows per launch (a row's bits do not depend on the other rows or on
  // M): a 256-row prefill super-chunk runs it as 64-row slices, exactly the launches four 64-row chunks
  // would make, whatever GdnLayerParams::seq_slice says.
  // decode-t1 item 2 (docs/perf.md): one launch over the concatenated [a; b] weight (Container's
  // BuildGdnAb) with N = 2H -- output row t is [a_t (H), b_t (H)], so the consumers below read a at
  // ab_buf and b at ab_buf + H with a row stride of 2H -- instead of two N = H launches. Every output
  // column is the same bits (the GEMM's per-column reduction does not depend on N), so a and b are the
  // separate launches' bytes. R4DX_DECODE_LEGACY=ab (or a container without the merged weight) keeps
  // the two launches, with the original compact [T, H] buffers.
  const bool ab_merged = !w_.in_proj_ab.empty() && w_.in_proj_ab.size() == 2 * w_.in_proj_a.size();
  const int64_t ab_stride = ab_merged ? 2 * H : H;
  uint16_t* a_buf = nullptr;
  uint16_t* b_buf = nullptr;
  if (ab_merged) {
    uint16_t* ab_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * 2 * H));
    ProfiledCall(prof, s, "gemm:gdn.in_proj_ab", [&] {
      for (int64_t m0 = 0; m0 < T; m0 += 64) {
        core::r4d::GemmBf16NtM64(x_normed + m0 * hidden, w_.in_proj_ab.data(), ab_buf + m0 * 2 * H,
                                  static_cast<int>(std::min<int64_t>(64, T - m0)),
                                  static_cast<int>(hidden), static_cast<int>(2 * H), 4, 4, 1, s);
      }
    });
    a_buf = ab_buf;
    b_buf = ab_buf + H;
  } else {
    a_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H));
    ProfiledCall(prof, s, "gemm:gdn.in_proj_a", [&] {
      for (int64_t m0 = 0; m0 < T; m0 += 64) {
        core::r4d::GemmBf16NtM64(x_normed + m0 * hidden, w_.in_proj_a.data(), a_buf + m0 * H,
                                  static_cast<int>(std::min<int64_t>(64, T - m0)),
                                  static_cast<int>(hidden), static_cast<int>(H), 4, 4, 1, s);
      }
    });
    b_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H));
    ProfiledCall(prof, s, "gemm:gdn.in_proj_b", [&] {
      for (int64_t m0 = 0; m0 < T; m0 += 64) {
        core::r4d::GemmBf16NtM64(x_normed + m0 * hidden, w_.in_proj_b.data(), b_buf + m0 * H,
                                  static_cast<int>(std::min<int64_t>(64, T - m0)),
                                  static_cast<int>(hidden), static_cast<int>(H), 4, 4, 1, s);
      }
    });
  }
  // in_proj_z (R1, docs/r9700.md): now dispatched through ApplyLinear like in_proj_qkv/out_proj --
  // 3.02 GB/token of what used to be a forced-bf16 GEMM, now eligible for a quantized layout. Reuses
  // the SAME fused rmsnorm epilogue as in_proj_qkv above WHEN both weights agree on layout (the
  // ordinary case: one container-wide body layout, model.h's `layout` field) -- but
  // Container::LoadQuantLinearWithFallback (container.cpp) can fall a single tensor back to bf16
  // independently of its sibling if an older container lacks that layout's rows for it (R1's own
  // doc comment), so this does NOT assume the two always match; it only reuses the shared buffer
  // when they genuinely do, falling back to in_proj_z's own separate quant launch otherwise.
  uint16_t* z_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * value_dim));
  const bool z_shares_pre =
      have_in_proj_pre && (EpilogueForLayout(w_.in_proj_z.layout) == in_proj_epilogue);
  ProfiledCall(prof, s, "gemm:gdn.in_proj_z", [&] {
    ApplyLinear(stream, arena, w_.in_proj_z, x_normed, z_buf, T,
                trellis_shared ? &trellis_in[1] : z_shares_pre ? &in_proj_pre : nullptr);
  });

  // ---- control arrays ---------------------------------------------------------------------------
  // See GdnControlCache (gdn_state.h): these are pure functions of (T, p.slot), so every distinct
  // value is uploaded once ever (no per-call hipMemcpy, no per-call sync) rather than re-uploaded
  // on every single decode step.
  // The sequence-dependent prefill ops (conv prep, kkt solve, chunk scan + state commit, gated norm) run
  // once per `slice` rows: the whole call (slice == T: a 64-row chunk, and a 256-row super-chunk by
  // default) or, under R4DX_GDN_SLICE=64 (p.seq_slice 64), 64-row sub-slices in order -- each exactly the
  // call a 64-row chunk makes (has_init true after the first, the fp32 state handed on through the slot).
  // The bytes are the same either way (docs/trellis-m256.md "GDN sequence ops"): the kernels chunk by 64
  // internally with no cross-chunk state but the scan's fp32 state, which a sliced call stores and
  // reloads as fp32 (exact) where a whole call keeps it in registers, and a chunk's conv halo is the same
  // x shorts whether it is read from x or from the conv-state cache the previous slice wrote.
  const int64_t slice = (p.is_prefill && p.seq_slice > 0 && p.seq_slice < T) ? p.seq_slice : T;
  if (slice != T && (slice != 64 || T % slice != 0)) {
    throw std::runtime_error("GdnLayer::Forward: seq_slice must be 64 and divide T");
  }
  if (p.conv_prep < 0 || p.conv_prep > kGdnConvV2) {
    throw std::runtime_error("GdnLayer::Forward: conv_prep must be 0, 1 or 2");
  }
  const int32_t* cu_dev = control.CuPair(slice);
  const int32_t* cache_idx_dev = control.CacheIdx(p.slot);

  uint16_t* q_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * Hg * K));
  uint16_t* k_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * Hg * K));
  uint16_t* v_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H * V));
  uint16_t* out_core = arena.Alloc<uint16_t>(static_cast<size_t>(T * value_dim));  // [T,H,V]==[T,value_dim]

  if (p.is_prefill) {
    float* g_buf = arena.Alloc<float>(static_cast<size_t>(T * H));
    float* beta_buf = arena.Alloc<float>(static_cast<size_t>(T * H));

    constexpr int64_t kChunk = 64;  // r4d_gdn_dims().chunk
    uint16_t* A_buf = arena.Alloc<uint16_t>(static_cast<size_t>(slice * H * kChunk));
    float* h0 = states.RecurrentSlotPtr(p.slot);
    float* ht_scratch = arena.Alloc<float>(static_cast<size_t>(H * V * K));
    uint16_t* o_core = arena.Alloc<uint16_t>(static_cast<size_t>(T * H * V));
    const int ts = static_cast<int>(slice);
    // The original r4d_gdn_conv_prep (p.conv_prep 0 / 1) or r4d_gdn_conv_prep2 (2: a wide Model's default,
    // R4DX_GDN_CONV=1 opts out): the same bytes; prep2 only spreads the work over more of the device
    // (r4d.h).
    const auto conv_prep = p.conv_prep == kGdnConvV2 ? &core::r4d::GdnConvPrep2 : &core::r4d::GdnConvPrep;

    for (int64_t r0 = 0; r0 < T; r0 += slice) {
      // rows [r0, r0 + slice) of every [T, ...] buffer; the first slice of a call reads the slot's
      // history only if the call itself has one, every later slice always does
      const uint8_t* has_init_dev = (p.has_init || r0 > 0) ? control.HasInitTrue() : nullptr;

      ProfiledCall(prof, s, "gdn.conv_prep", [&] {
        conv_prep(mixed_qkv + r0 * conv_dim, conv_dim, w_.conv1d_weight.data(),
                  /*bias=*/nullptr, states.ConvBase(), states.ConvSeqStride(),
                  states.ConvDimStride(), states.ConvTokStride(), cache_idx_dev,
                  /*ci_stride=*/1, has_init_dev, a_buf + r0 * ab_stride, b_buf + r0 * ab_stride,
                  ab_stride, /*ab_is_bf16=*/1, w_.A_log.data(),
                  w_.dt_bias.data(), q_buf + r0 * Hg * K, k_buf + r0 * Hg * K,
                  v_buf + r0 * H * V, g_buf + r0 * H, beta_buf + r0 * H, cu_dev,
                  /*N=*/1, ts, H, Hg, K, V, static_cast<int>(width), kSoftplusThr, s);
      });

      ProfiledCall(prof, s, "gdn.kkt_solve", [&] {
        core::r4d::GdnKktSolve(k_buf + r0 * Hg * K, beta_buf + r0 * H, g_buf + r0 * H, A_buf, cu_dev,
                                /*N=*/1, ts, H, Hg, K, static_cast<int>(kChunk), s);
      });

      ProfiledCall(prof, s, "gdn.chunk_scan", [&] {
        core::r4d::GdnChunkScan(q_buf + r0 * Hg * K, k_buf + r0 * Hg * K, v_buf + r0 * H * V, A_buf,
                                 g_buf + r0 * H, beta_buf + r0 * H, h0, o_core + r0 * H * V,
                                 ht_scratch, cu_dev, /*N=*/1, H, Hg, K, V, static_cast<int>(kChunk),
                                 scale, s);
        // Commit the scanned state back into the sequence's slot (see gdn_state.h): both pointers
        // are persistent device buffers, so an async D2D copy on the same stream is safely ordered
        // after the kernel above and before any later call that reads this slot.
        R4DX_HIP_CHECK(hipMemcpyAsync(h0, ht_scratch, static_cast<size_t>(H * V * K) * sizeof(float),
                                       hipMemcpyDeviceToDevice, s));
      });

      ProfiledCall(prof, s, "gdn.gated_rmsnorm", [&] {
        core::r4d::GdnGatedRmsNorm(o_core + r0 * H * V, z_buf + r0 * value_dim, w_.norm_weight.data(),
                                    out_core + r0 * value_dim,
                                    /*rows=*/slice * H, /*xrow=*/V, /*zrow=*/V, /*orow=*/V,
                                    /*width=*/static_cast<int>(V), eps, kGdnActSilu, s);
      });
    }
  } else {
    // max_query_len must be the WINDOW BOUND (GdnStateManager::MaxDecodeWindow()), not the actual
    // row count T (review finding, 2026-09-19): r4d_gdn_conv_w4_h128_bf16.hip derives
    // slen_eff = state_len_max - (maxq - slen), and its cache-rewrite loop needs slen_eff-slen<=2
    // (CP_ST==width-1==3's own history depth) -- that only holds for every legal T when maxq is
    // pinned to the window this state was actually SIZED for (states.StateLenMax() ==
    // width-2+MaxDecodeWindow(), see GdnStateManager's file comment), not to whatever T this one
    // call happens to pass (T < MaxDecodeWindow() -- e.g. an MTP-enabled Model's plain decode step,
    // T=1, with mtp_draft_k>0 widening MaxDecodeWindow() beyond 1 -- previously read/wrote past the
    // kernel's fixed-size hist[]/x[] arrays; see gdn_state.h's own file comment for the identical
    // bug class that manifested as a hang for the analogous sidx array).
    if (T > states.MaxDecodeWindow()) {
      throw std::runtime_error(
          "GdnLayer::Forward: decode T exceeds GdnStateManager::MaxDecodeWindow()");
    }
    ProfiledCall(prof, s, "gdn.conv_update", [&] {
      core::r4d::GdnConvUpdate(mixed_qkv, conv_dim, w_.conv1d_weight.data(), /*bias=*/nullptr,
                                states.ConvBase(), states.ConvSeqStride(), states.ConvDimStride(),
                                states.ConvTokStride(), static_cast<int>(states.StateLenMax()),
                                cache_idx_dev, /*ci_stride=*/1, p.num_accepted, q_buf,
                                k_buf, v_buf, cu_dev, /*N=*/1, H, Hg, K, V,
                                static_cast<int>(width),
                                /*max_query_len=*/static_cast<int>(states.MaxDecodeWindow()), s);
    });

    // Every candidate token writes its own state slot (r4d_gdn_recurrent_update_*'s "one state
    // write per candidate token"): sidx is the STABLE, ascending {window 0, window 1, ...} array
    // for this sequence (GdnControlCache::SidxBase / GdnStateManager's file comment), not a T-sized
    // fresh array -- a plain sequential (non-speculative) decode (T==1, p.num_accepted==nullptr)
    // only ever touches window index 0, identical to this class's pre-MTP behavior; an MTP verify
    // call (T==draft_k+1) writes one snapshot per candidate into its own window slot, and
    // p.num_accepted (carried from the PREVIOUS call) selects which slot THIS call seeds from.
    const int64_t window = states.MaxDecodeWindow();
    const int32_t* sidx_dev = control.SidxBase(p.slot, window);

    if (!states.WriteOnce()) {
      ProfiledCall(prof, s, "gdn.recurrent_update", [&] {
        core::r4d::GdnRecurrentUpdate(
            q_buf, k_buf, v_buf, a_buf, b_buf, ab_stride, /*ab_is_bf16=*/1, w_.A_log.data(),
            w_.dt_bias.data(), states.RecurrentBase(), states.RecurrentSlotStride(),
            states.RecurrentHeadStride(), out_core, cu_dev, sidx_dev,
            /*indices_stride=*/window, p.num_accepted, z_buf, w_.norm_weight.data(), eps,
            kGdnActSilu, /*N=*/1, H, Hg, K, V, scale, kSoftplusThr, s);
      });
    } else {
      // Write-once state (docs/gdn-write-once.md 2.2): one state B per sequence, at sidx[0] == p.slot.
      //   T == 1, nothing pending : the legacy kernel, in place -- today's plain decode, byte for byte
      //                             (sidx = the window's ascending array, only sidx[0] is touched; no
      //                             num_accepted: there are no window slots to seed from)
      //   T == 1, pending         : replay the committed prefix of the last verify's log in the seed load,
      //                             run the row, store the final state into B (the speculation-to-plain
      //                             transition; every later plain step has nothing pending)
      //   T  > 1                  : log the rows (no state store), replaying the pending prefix first and
      //                             writing it back into B; the next ping-pong half takes the log
      // The log's key rows are sized from the manager's own key-head count (derived from its conv_dim); a layer
      // whose head geometry disagrees would index the log out of range.
      if (states.KeyHeads() != Hg) {
        throw std::runtime_error("GdnLayer::Forward: write-once state sized for " +
                                 std::to_string(states.KeyHeads()) + " key heads, this layer has " +
                                 std::to_string(Hg));
      }
      if (T == 1 && p.gdn_pending == nullptr) {
        ProfiledCall(prof, s, "gdn.recurrent_update", [&] {
          core::r4d::GdnRecurrentUpdate(
              q_buf, k_buf, v_buf, a_buf, b_buf, ab_stride, /*ab_is_bf16=*/1, w_.A_log.data(),
              w_.dt_bias.data(), states.RecurrentBase(), states.RecurrentSlotStride(),
              states.RecurrentHeadStride(), out_core, cu_dev, sidx_dev,
              /*indices_stride=*/window, /*num_accepted=*/nullptr, z_buf, w_.norm_weight.data(), eps,
              kGdnActSilu, /*N=*/1, H, Hg, K, V, scale, kSoftplusThr, s);
        });
      } else {
        const bool direct = (T == 1);
        const float* log_in = p.gdn_pending != nullptr ? states.LogIn() : nullptr;
        float* log_out = direct ? nullptr : states.LogOut();
        ProfiledCall(prof, s, direct ? "gdn.recurrent_update_wo_direct" : "gdn.recurrent_update_wo", [&] {
          core::r4d::GdnRecurrentUpdateWo(
              q_buf, k_buf, v_buf, a_buf, b_buf, ab_stride, /*ab_is_bf16=*/1, w_.A_log.data(),
              w_.dt_bias.data(), states.RecurrentBase(), states.RecurrentSlotStride(),
              states.RecurrentHeadStride(), out_core, cu_dev, sidx_dev, z_buf, w_.norm_weight.data(),
              eps, kGdnActSilu, log_in, log_out, p.gdn_pending, static_cast<int>(window),
              direct ? R4D_GDN_WO_DIRECT : R4D_GDN_WO_LOG, /*N=*/1, H, Hg, K, V, scale, kSoftplusThr, s);
        });
        if (!direct) states.FlipParity();
      }
    }
  }

  // ---- quant2 Q2b: out_proj's input Hadamard (docs/quant2.md section 4) ------------------------
  // Both branches above end with out_core = the gated norm's [T, H*V] output (prefill: the separate
  // GdnGatedRmsNorm; decode/verify: fused into the recurrent kernel's epilogue, which is why this
  // cannot be fused into a producer the way mlp.down's and attn.o's Hadamards are), so ONE in-place
  // launch here covers every path. Block V = one value head.
  if (p.out_had_signs != nullptr) {
    ProfiledCall(prof, s, "gdn.out_hadamard", [&] {
      r4dx_hadamard_inplace_bf16(reinterpret_cast<int64_t>(out_core), T, value_dim,
                                  reinterpret_cast<int64_t>(p.out_had_signs), V,
                                  reinterpret_cast<int64_t>(s));
    });
  }

  // ---- out_proj + residual ----------------------------------------------------------------------
  uint16_t* gdn_out = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
  // docs/trellis-kernel.md 10.6: out_proj is the first weight stream after the recurrent update,
  // which left this layer's new state (3 MB per token) dirty in the L2. A trellis GEMM loads its
  // weights non-temporally and evicts almost nothing, so those lines used to drain a few at a time
  // through every GEMM up to the next attention layer, 5-7% slower each; out_proj's weights loaded
  // normally (temporal_weight_loads, NT = 0: a cache hint, the same bits) flush them in one burst.
  // Measured per step (GPU span), K4m (mix4.5m): -0.64 (-1.43) ms at decode, -0.93 (-1.69) at a
  // 4-token and -0.72 (-1.74) at an 8-token verify window. w4a16 ignores the flag.
  ProfiledCall(prof, s, "gemm:gdn.out_proj", [&] {
    ApplyLinear(stream, arena, w_.out_proj, out_core, gdn_out, T, /*pre=*/nullptr,
                /*temporal_weight_loads=*/true);
  });
  // Tensor parallel (docs/tp.md 6.2, site A1): out_proj is row-parallel, so each rank holds a
  // partial sum; sum it across ranks before the residual (and the fused next-norm epilogue).
  if (comm_ != nullptr) {
    ProfiledCall(prof, s, "tp.allreduce", [&] { comm_->AllReduceSumBf16Rows(gdn_out, T, hidden, s); });
  }
  ProfiledCall(prof, s, "gdn.residual", [&] {
    if (next_norm_weight != nullptr) {
      r4dx_residual_rmsnorm_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(gdn_out),
                                  reinterpret_cast<int64_t>(next_norm_weight),
                                  reinterpret_cast<int64_t>(x_out),
                                  reinterpret_cast<int64_t>(x_normed_out), T, hidden, eps,
                                  reinterpret_cast<int64_t>(s), next_epilogue,
                                  reinterpret_cast<int64_t>(next_epilogue_out));
    } else {
      r4dx_residual_add_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(gdn_out),
                              reinterpret_cast<int64_t>(x_out), T * hidden,
                              reinterpret_cast<int64_t>(s));
    }
  });
}

}  // namespace r4dx::model
