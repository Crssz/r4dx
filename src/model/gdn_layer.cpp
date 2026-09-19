#include "gdn_layer.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <stdexcept>
#include <vector>

#include "linear.h"
#include "profile_span.h"
#include "r4d.h"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"

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
                        const void* x_normed_pre_data, const float* x_normed_pre_scale,
                        int next_epilogue, void* next_epilogue_out, float* next_epilogue_scale) {
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
  // R2/P2: two ways x_normed's fused quant epilogue gets populated -- (a) x_normed_in==nullptr:
  // this call computes its own rmsnorm, and knows w_.in_proj_qkv/w_.in_proj_z locally, both sharing
  // ONE container-wide body layout (model.h's `layout` field), so ONE fused epilogue serves BOTH
  // consumers below; (b) x_normed_in!=nullptr: the previous layer's Mlp already fused an epilogue
  // (matching THIS Model's body layout) into x_normed_pre_epilogue/_data/_scale when it produced
  // x_normed_in -- reused here directly, no local quant launch needed either way. Either way,
  // in_proj_pre below is validated per-weight (z_shares_pre) before use, never assumed.
  const uint16_t* x_normed = x_normed_in;
  uint16_t* x_normed_scratch = nullptr;
  int in_proj_epilogue = x_normed_pre_epilogue;
  const void* in_proj_pre_data = x_normed_pre_data;
  const float* in_proj_pre_scale = x_normed_pre_scale;
  // Cross-boundary reuse is only valid if the producer's epilogue matches THIS layer's own
  // in_proj_qkv layout (LoadQuantLinearWithFallback can fall an individual tensor back to bf16
  // independently of its container-wide sibling layout -- do not assume Model.cpp's caller-side
  // choice of next_epilogue already accounts for that; verify locally, same defensive pattern as
  // z_shares_pre below).
  if (x_normed != nullptr && in_proj_epilogue != EpilogueForLayout(w_.in_proj_qkv.layout)) {
    in_proj_epilogue = r4dx_epilogue_none;
    in_proj_pre_data = nullptr;
    in_proj_pre_scale = nullptr;
  }
  if (x_normed == nullptr) {
    in_proj_epilogue = EpilogueForLayout(w_.in_proj_qkv.layout);
    x_normed_scratch = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
    uint8_t* in_proj_pre_data_local = nullptr;
    float* in_proj_pre_scale_local = nullptr;
    if (in_proj_epilogue != r4dx_epilogue_none) {
      const int elem_size = (in_proj_epilogue == r4dx_epilogue_f16) ? 2 : 1;
      // 16-byte alignment: see attention_layer.hpp's identical comment (w4a8's GEMM does a
      // global_load_b64 fragment read; uint8_t's default 1-byte arena alignment does not
      // guarantee that for a buffer allocated this late in a layer's scratch sequence).
      in_proj_pre_data_local =
          arena.Alloc<uint8_t>(static_cast<size_t>(T * hidden * elem_size), /*align_bytes=*/16);
      if (in_proj_epilogue != r4dx_epilogue_f16) {
        in_proj_pre_scale_local = arena.Alloc<float>(static_cast<size_t>(T));
      }
    }
    ProfiledCall(prof, s, "gdn.rmsnorm", [&] {
      r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(x),
                         reinterpret_cast<int64_t>(input_layernorm_.data()),
                         reinterpret_cast<int64_t>(x_normed_scratch), T, hidden, eps,
                         reinterpret_cast<int64_t>(s), in_proj_epilogue,
                         reinterpret_cast<int64_t>(in_proj_pre_data_local),
                         reinterpret_cast<int64_t>(in_proj_pre_scale_local));
    });
    x_normed = x_normed_scratch;
    in_proj_pre_data = in_proj_pre_data_local;
    in_proj_pre_scale = in_proj_pre_scale_local;
  }
  const bool have_in_proj_pre = in_proj_epilogue != r4dx_epilogue_none;
  PreQuantizedActivation in_proj_pre{in_proj_epilogue, in_proj_pre_data, in_proj_pre_scale};

  // ---- in_proj_qkv / in_proj_z / in_proj_b / in_proj_a -----------------------------------------
  uint16_t* mixed_qkv = arena.Alloc<uint16_t>(static_cast<size_t>(T * conv_dim));
  ProfiledCall(prof, s, "gemm:gdn.in_proj_qkv", [&] {
    ApplyLinear(stream, arena, w_.in_proj_qkv, x_normed, mixed_qkv, T,
                have_in_proj_pre ? &in_proj_pre : nullptr);
  });
  uint16_t* a_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H));
  // in_proj_a/in_proj_b are plain bf16 linears (never quantized -- docs/r9700.md R1: "too small to
  // matter, feed the decay path -- leave them bf16"), so they go straight through the bf16 GEMM
  // rather than through ApplyLinear's QuantLinear dispatch.
  ProfiledCall(prof, s, "gemm:gdn.in_proj_a", [&] {
    core::r4d::GemmBf16NtM64(x_normed, w_.in_proj_a.data(), a_buf, static_cast<int>(T),
                              static_cast<int>(hidden), static_cast<int>(H), 4, 4, 1, s);
  });
  uint16_t* b_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H));
  ProfiledCall(prof, s, "gemm:gdn.in_proj_b", [&] {
    core::r4d::GemmBf16NtM64(x_normed, w_.in_proj_b.data(), b_buf, static_cast<int>(T),
                              static_cast<int>(hidden), static_cast<int>(H), 4, 4, 1, s);
  });
  // in_proj_z (R1, docs/r9700.md): now dispatched through ApplyLinear like in_proj_qkv/out_proj --
  // 3.02 GB/token of what used to be a forced-bf16 GEMM, now eligible for mxfp4/w4a16/w4a8. Reuses
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
                z_shares_pre ? &in_proj_pre : nullptr);
  });

  // ---- control arrays ---------------------------------------------------------------------------
  // See GdnControlCache (gdn_state.h): these are pure functions of (T, p.slot), so every distinct
  // value is uploaded once ever (no per-call hipMemcpy, no per-call sync) rather than re-uploaded
  // on every single decode step.
  const int32_t* cu_dev = control.CuPair(T);
  const int32_t* cache_idx_dev = control.CacheIdx(p.slot);

  uint16_t* q_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * Hg * K));
  uint16_t* k_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * Hg * K));
  uint16_t* v_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H * V));
  uint16_t* out_core = arena.Alloc<uint16_t>(static_cast<size_t>(T * value_dim));  // [T,H,V]==[T,value_dim]

  if (p.is_prefill) {
    float* g_buf = arena.Alloc<float>(static_cast<size_t>(T * H));
    float* beta_buf = arena.Alloc<float>(static_cast<size_t>(T * H));
    const uint8_t* has_init_dev = p.has_init ? control.HasInitTrue() : nullptr;

    ProfiledCall(prof, s, "gdn.conv_prep", [&] {
      core::r4d::GdnConvPrep(mixed_qkv, conv_dim, w_.conv1d_weight.data(), /*bias=*/nullptr,
                              states.ConvBase(), states.ConvSeqStride(), states.ConvDimStride(),
                              states.ConvTokStride(), cache_idx_dev, /*ci_stride=*/1, has_init_dev,
                              a_buf, b_buf, /*ab_stride=*/H, /*ab_is_bf16=*/1, w_.A_log.data(),
                              w_.dt_bias.data(), q_buf, k_buf, v_buf, g_buf, beta_buf, cu_dev,
                              /*N=*/1, static_cast<int>(T), H, Hg, K, V, static_cast<int>(width),
                              kSoftplusThr, s);
    });

    constexpr int64_t kChunk = 64;  // r4d_gdn_dims().chunk
    uint16_t* A_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H * kChunk));
    ProfiledCall(prof, s, "gdn.kkt_solve", [&] {
      core::r4d::GdnKktSolve(k_buf, beta_buf, g_buf, A_buf, cu_dev, /*N=*/1, static_cast<int>(T), H,
                              Hg, K, static_cast<int>(kChunk), s);
    });

    float* h0 = states.RecurrentSlotPtr(p.slot);
    float* ht_scratch = arena.Alloc<float>(static_cast<size_t>(H * V * K));
    uint16_t* o_core = arena.Alloc<uint16_t>(static_cast<size_t>(T * H * V));
    ProfiledCall(prof, s, "gdn.chunk_scan", [&] {
      core::r4d::GdnChunkScan(q_buf, k_buf, v_buf, A_buf, g_buf, beta_buf, h0, o_core, ht_scratch,
                               cu_dev, /*N=*/1, H, Hg, K, V, static_cast<int>(kChunk), scale, s);
      // Commit the scanned state back into the sequence's slot (see gdn_state.h): both pointers
      // are persistent device buffers, so an async D2D copy on the same stream is safely ordered
      // after the kernel above and before any later call that reads this slot.
      R4DX_HIP_CHECK(hipMemcpyAsync(h0, ht_scratch, static_cast<size_t>(H * V * K) * sizeof(float),
                                     hipMemcpyDeviceToDevice, s));
    });

    ProfiledCall(prof, s, "gdn.gated_rmsnorm", [&] {
      core::r4d::GdnGatedRmsNorm(o_core, z_buf, w_.norm_weight.data(), out_core,
                                  /*rows=*/T * H, /*xrow=*/V, /*zrow=*/V, /*orow=*/V,
                                  /*width=*/static_cast<int>(V), eps, kGdnActSilu, s);
    });
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

    ProfiledCall(prof, s, "gdn.recurrent_update", [&] {
      core::r4d::GdnRecurrentUpdate(
          q_buf, k_buf, v_buf, a_buf, b_buf, /*ab_stride=*/H, /*ab_is_bf16=*/1, w_.A_log.data(),
          w_.dt_bias.data(), states.RecurrentBase(), states.RecurrentSlotStride(),
          states.RecurrentHeadStride(), out_core, cu_dev, sidx_dev,
          /*indices_stride=*/window, p.num_accepted, z_buf, w_.norm_weight.data(), eps,
          kGdnActSilu, /*N=*/1, H, Hg, K, V, scale, kSoftplusThr, s);
    });
  }

  // ---- out_proj + residual ----------------------------------------------------------------------
  uint16_t* gdn_out = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
  ProfiledCall(prof, s, "gemm:gdn.out_proj", [&] {
    ApplyLinear(stream, arena, w_.out_proj, out_core, gdn_out, T);
  });
  ProfiledCall(prof, s, "gdn.residual", [&] {
    if (next_norm_weight != nullptr) {
      r4dx_residual_rmsnorm_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(gdn_out),
                                  reinterpret_cast<int64_t>(next_norm_weight),
                                  reinterpret_cast<int64_t>(x_out),
                                  reinterpret_cast<int64_t>(x_normed_out), T, hidden, eps,
                                  reinterpret_cast<int64_t>(s), next_epilogue,
                                  reinterpret_cast<int64_t>(next_epilogue_out),
                                  reinterpret_cast<int64_t>(next_epilogue_scale));
    } else {
      r4dx_residual_add_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(gdn_out),
                              reinterpret_cast<int64_t>(x_out), T * hidden,
                              reinterpret_cast<int64_t>(s));
    }
  });
}

}  // namespace r4dx::model
