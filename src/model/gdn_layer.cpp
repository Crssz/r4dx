#include "gdn_layer.h"

#include <hip/hip_runtime.h>

#include <cmath>
#include <vector>

#include "linear.h"
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
                        const GdnLayerParams& p) {
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
  uint16_t* x_normed = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
  r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(input_layernorm_.data()),
                     reinterpret_cast<int64_t>(x_normed), T, hidden, eps, reinterpret_cast<int64_t>(s));

  // ---- in_proj_qkv / in_proj_z / in_proj_b / in_proj_a -----------------------------------------
  uint16_t* mixed_qkv = arena.Alloc<uint16_t>(static_cast<size_t>(T * conv_dim));
  ApplyLinear(stream, arena, w_.in_proj_qkv, x_normed, mixed_qkv, T);
  uint16_t* a_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H));
  // in_proj_a/in_proj_b/in_proj_z are plain bf16 linears (never quantized -- container-format.md
  // lists only attn.qg|o / gdn.in_proj_qkv|out_proj / mlp.gate_up|down as multi-layout), so they
  // go straight through the bf16 GEMM rather than through ApplyLinear's QuantLinear dispatch.
  core::r4d::GemmBf16NtM64(x_normed, w_.in_proj_a.data(), a_buf, static_cast<int>(T),
                            static_cast<int>(hidden), static_cast<int>(H), 4, 4, 1, s);
  uint16_t* b_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H));
  core::r4d::GemmBf16NtM64(x_normed, w_.in_proj_b.data(), b_buf, static_cast<int>(T),
                            static_cast<int>(hidden), static_cast<int>(H), 4, 4, 1, s);
  uint16_t* z_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * value_dim));
  core::r4d::GemmBf16NtM64(x_normed, w_.in_proj_z.data(), z_buf, static_cast<int>(T),
                            static_cast<int>(hidden), static_cast<int>(value_dim), 4, 4, 1, s);

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

    core::r4d::GdnConvPrep(mixed_qkv, conv_dim, w_.conv1d_weight.data(), /*bias=*/nullptr,
                            states.ConvBase(), states.ConvSeqStride(), states.ConvDimStride(),
                            states.ConvTokStride(), cache_idx_dev, /*ci_stride=*/1, has_init_dev,
                            a_buf, b_buf, /*ab_stride=*/H, /*ab_is_bf16=*/1, w_.A_log.data(),
                            w_.dt_bias.data(), q_buf, k_buf, v_buf, g_buf, beta_buf, cu_dev,
                            /*N=*/1, static_cast<int>(T), H, Hg, K, V, static_cast<int>(width),
                            kSoftplusThr, s);

    constexpr int64_t kChunk = 64;  // r4d_gdn_dims().chunk
    uint16_t* A_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H * kChunk));
    core::r4d::GdnKktSolve(k_buf, beta_buf, g_buf, A_buf, cu_dev, /*N=*/1, static_cast<int>(T), H,
                            Hg, K, static_cast<int>(kChunk), s);

    float* h0 = states.RecurrentSlotPtr(p.slot);
    float* ht_scratch = arena.Alloc<float>(static_cast<size_t>(H * V * K));
    uint16_t* o_core = arena.Alloc<uint16_t>(static_cast<size_t>(T * H * V));
    core::r4d::GdnChunkScan(q_buf, k_buf, v_buf, A_buf, g_buf, beta_buf, h0, o_core, ht_scratch,
                             cu_dev, /*N=*/1, H, Hg, K, V, static_cast<int>(kChunk), scale, s);
    // Commit the scanned state back into the sequence's slot (see gdn_state.h): both pointers are
    // persistent device buffers, so an async D2D copy on the same stream is safely ordered after
    // the kernel above and before any later call that reads this slot.
    R4DX_HIP_CHECK(hipMemcpyAsync(h0, ht_scratch, static_cast<size_t>(H * V * K) * sizeof(float),
                                   hipMemcpyDeviceToDevice, s));

    core::r4d::GdnGatedRmsNorm(o_core, z_buf, w_.norm_weight.data(), out_core,
                                /*rows=*/T * H, /*xrow=*/V, /*zrow=*/V, /*orow=*/V,
                                /*width=*/static_cast<int>(V), eps, kGdnActSilu, s);
  } else {
    core::r4d::GdnConvUpdate(mixed_qkv, conv_dim, w_.conv1d_weight.data(), /*bias=*/nullptr,
                              states.ConvBase(), states.ConvSeqStride(), states.ConvDimStride(),
                              states.ConvTokStride(), static_cast<int>(states.StateLenMax()),
                              cache_idx_dev, /*ci_stride=*/1, /*num_accepted=*/nullptr, q_buf,
                              k_buf, v_buf, cu_dev, /*N=*/1, H, Hg, K, V,
                              static_cast<int>(width), /*max_query_len=*/static_cast<int>(T), s);

    // Every candidate token writes its own state slot (r4d_gdn_recurrent_update_*'s "one state
    // write per candidate token"); a plain sequential (non-speculative) decode of T tokens simply
    // reuses the SAME slot for every position, so the loop's carried register state ends up
    // committed there after the last token (see gdn_layer.h / gdn_state.h file comments).
    const int32_t* sidx_dev = control.Sidx(T, p.slot);

    core::r4d::GdnRecurrentUpdate(
        q_buf, k_buf, v_buf, a_buf, b_buf, /*ab_stride=*/H, /*ab_is_bf16=*/1, w_.A_log.data(),
        w_.dt_bias.data(), states.RecurrentBase(), states.RecurrentSlotStride(),
        states.RecurrentHeadStride(), out_core, cu_dev, sidx_dev, /*indices_stride=*/T,
        /*num_accepted=*/nullptr, z_buf, w_.norm_weight.data(), eps, kGdnActSilu,
        /*N=*/1, H, Hg, K, V, scale, kSoftplusThr, s);
  }

  // ---- out_proj + residual ----------------------------------------------------------------------
  uint16_t* gdn_out = arena.Alloc<uint16_t>(static_cast<size_t>(T * hidden));
  ApplyLinear(stream, arena, w_.out_proj, out_core, gdn_out, T);
  r4dx_residual_add_bf16(reinterpret_cast<int64_t>(x), reinterpret_cast<int64_t>(gdn_out),
                          reinterpret_cast<int64_t>(x_out), T * hidden, reinterpret_cast<int64_t>(s));
}

}  // namespace r4dx::model
