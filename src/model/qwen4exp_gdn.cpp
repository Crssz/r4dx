#include "qwen4exp_gdn.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "linear.h"
#include "r4d.h"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/model/qwen4exp/gdn_sigmoid_norm.h"
#include "r4dx/model/qwen4exp/common.h"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"

namespace r4dx::model {

namespace {

// libr4d's GDN kernels' own constants, as gdn_layer.cpp uses them.
constexpr float kSoftplusThr = 20.0f;
constexpr int64_t kScanChunk = 64;  // r4d_gdn_dims().chunk
constexpr int kActSilu = 0;         // r4d_gdn_gated_rmsnorm_h128_bf16 / recurrent_update act codes
constexpr int kActSigmoid = 1;

constexpr const char* kCtx = "Qwen4ExpGdn";

[[noreturn]] void Fail(const std::string& msg) { qwen4exp::Fail(kCtx, msg); }

using qwen4exp::Padded;  // Arena::Alloc pads every allocation end to 16

void CheckPtr(const void* p, const char* name) {
  if (p == nullptr) Fail(std::string("missing weight ") + name);
}

// The row slices of a prefill call: one slice when it fits a scan chunk or is a whole number of chunks (the 256-row
// super-chunk of the 27B path), else consecutive 64-row slices, the last one short -- each the call a chunk makes
// (gdn_layer.cpp's seq_slice contract: the bytes equal the consecutive 64-row calls).
int64_t SliceRows(int64_t T) { return (T <= kScanChunk || T % kScanChunk == 0) ? T : kScanChunk; }

}  // namespace

Qwen4ExpGdn::Qwen4ExpGdn(const Qwen4ExpConfig& cfg, const Qwen4ExpGdnRefs& w)
    : w_(w),
      hidden_(cfg.hidden_size),
      Hg_(cfg.linear_num_key_heads),
      H_(cfg.linear_num_value_heads),
      K_(cfg.linear_key_head_dim),
      V_(cfg.linear_value_head_dim),
      width_(cfg.linear_conv_kernel_dim),
      key_dim_(cfg.linear_num_key_heads * cfg.linear_key_head_dim),
      value_dim_(cfg.linear_num_value_heads * cfg.linear_value_head_dim),
      conv_dim_(2 * key_dim_ + value_dim_),
      eps_(static_cast<float>(cfg.rms_norm_eps)),
      sigmoid_gate_(cfg.output_gate_type == Qwen4ExpConfig::OutputGate::kSigmoid) {
  if (K_ != 128 || V_ != 128) Fail("libr4d's GDN kernels are written for key / value head dim 128");
  if (width_ != 4) Fail("libr4d's GDN conv kernels are written for a conv kernel of 4");
  if (H_ % Hg_ != 0) Fail("value heads are not a multiple of key heads");
  qwen4exp::CheckLinear(kCtx, w.in_proj_qkv, "in_proj_qkv", conv_dim_, hidden_);
  qwen4exp::CheckLinear(kCtx, w.in_proj_z, "in_proj_z", value_dim_, hidden_);
  qwen4exp::CheckLinear(kCtx, w.out_proj, "out_proj", hidden_, value_dim_);
  CheckPtr(w.in_proj_a, "in_proj_a");
  CheckPtr(w.in_proj_b, "in_proj_b");
  CheckPtr(w.conv1d_weight, "conv1d_weight");
  CheckPtr(w.A_log, "A_log");
  CheckPtr(w.dt_bias, "dt_bias");
  CheckPtr(w.norm_weight, "norm_weight");
  if (hidden_ % 64 != 0) Fail("hidden size must be a multiple of 64 for the bf16 in_proj_a / in_proj_b GEMM");
}

size_t Qwen4ExpGdn::ArenaBytes(int64_t T) const {
  const size_t t = static_cast<size_t>(T);
  const int64_t slice = T;  // the largest any slicing needs (A is sized per slice)
  size_t b = 0;
  b += Padded(t * static_cast<size_t>(conv_dim_) * 2);                 // mixed_qkv
  b += 2 * Padded(t * static_cast<size_t>(H_) * 2);                    // a, b
  b += Padded(t * static_cast<size_t>(value_dim_) * 2);                // z
  b += 2 * Padded(t * static_cast<size_t>(Hg_ * K_) * 2);              // q, k
  b += Padded(t * static_cast<size_t>(H_ * V_) * 2);                   // v
  b += Padded(t * static_cast<size_t>(value_dim_) * 2);                // out_core
  b += 2 * Padded(t * static_cast<size_t>(H_) * 4);                    // g, beta (prefill)
  b += Padded(static_cast<size_t>(slice) * static_cast<size_t>(H_) * kScanChunk * 2);  // A
  b += Padded(static_cast<size_t>(H_ * V_ * K_) * 4);                  // ht scratch
  b += Padded(t * static_cast<size_t>(H_ * V_) * 2);                   // o_core
  return b;
}

void Qwen4ExpGdn::Forward(hipStream_t s, core::Arena& arena, GdnStateManager& states, GdnControlCache& control,
                          const uint16_t* x, uint16_t* out, int64_t T, const Qwen4ExpGdnCall& call) const {
  if (T < 1) Fail("T must be >= 1");
  if (states.MaxDecodeWindow() != 1 || states.WriteOnce()) Fail("the state manager must be the window-1, single-slot one");
  if (states.NumHeads() != H_ || states.ValueDim() != V_ || states.KeyDim() != K_ || states.ConvDim() != conv_dim_) {
    Fail("the state manager's geometry does not match this layer");
  }
  const int Hi = static_cast<int>(H_), Hgi = static_cast<int>(Hg_), Ki = static_cast<int>(K_),
            Vi = static_cast<int>(V_);
  const float scale = 1.0f / std::sqrt(static_cast<float>(K_));
  const int32_t slot = states.SlotForSeq(0);

  // ---- projections ------------------------------------------------------------------------------------------------
  uint16_t* mixed_qkv = arena.Alloc<uint16_t>(static_cast<size_t>(T * conv_dim_));
  qwen4exp::ApplyBf16Linear(s, arena, *w_.in_proj_qkv, x, mixed_qkv, T);
  // in_proj_a / in_proj_b: plain bf16 GEMMs, <= 64 rows per launch, the 27B layer's own WV 4 / SK 4 / MB 1 (its
  // non-merged path, gdn_layer.cpp: a feeds conv_prep's `a`, b its `b`).
  uint16_t* a_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H_));
  uint16_t* b_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H_));
  for (int64_t m0 = 0; m0 < T; m0 += 64) {
    const int m = static_cast<int>(std::min<int64_t>(64, T - m0));
    core::r4d::GemmBf16NtM64(x + m0 * hidden_, w_.in_proj_a, a_buf + m0 * H_, m, static_cast<int>(hidden_),
                             Hi, 4, 4, 1, s);
    core::r4d::GemmBf16NtM64(x + m0 * hidden_, w_.in_proj_b, b_buf + m0 * H_, m, static_cast<int>(hidden_),
                             Hi, 4, 4, 1, s);
  }
  uint16_t* z_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * value_dim_));
  qwen4exp::ApplyBf16Linear(s, arena, *w_.in_proj_z, x, z_buf, T);

  // ---- the GDN core -----------------------------------------------------------------------------------------------
  const int32_t* cache_idx_dev = control.CacheIdx(slot);
  uint16_t* q_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * Hg_ * K_));
  uint16_t* k_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * Hg_ * K_));
  uint16_t* v_buf = arena.Alloc<uint16_t>(static_cast<size_t>(T * H_ * V_));
  uint16_t* out_core = arena.Alloc<uint16_t>(static_cast<size_t>(T * value_dim_));  // gated norm output [T, H * V]

  if (call.is_prefill && !call.prefill_recurrent) {
    float* g_buf = arena.Alloc<float>(static_cast<size_t>(T * H_));
    float* beta_buf = arena.Alloc<float>(static_cast<size_t>(T * H_));
    int64_t slice = call.seq_slice == 0 ? SliceRows(T) : (call.seq_slice < 0 ? T : call.seq_slice);
    if (slice > T) slice = T;
    if (slice != T && slice != kScanChunk) Fail("seq_slice must be 0 (auto), -1 (the whole call) or 64");
    if (call.conv_prep != 1 && call.conv_prep != 2) Fail("conv_prep must be 1 or 2");
    const auto conv_prep = call.conv_prep == 2 ? &core::r4d::GdnConvPrep2 : &core::r4d::GdnConvPrep;
    uint16_t* A_buf = arena.Alloc<uint16_t>(static_cast<size_t>(slice * H_ * kScanChunk));
    float* h0 = states.RecurrentSlotPtr(slot);
    float* ht_scratch = arena.Alloc<float>(static_cast<size_t>(H_ * V_ * K_));
    uint16_t* o_core = arena.Alloc<uint16_t>(static_cast<size_t>(T * H_ * V_));

    for (int64_t r0 = 0; r0 < T; r0 += slice) {
      const int ts = static_cast<int>(std::min<int64_t>(slice, T - r0));
      const int32_t* cu_dev = control.CuPair(ts);
      const uint8_t* has_init_dev = (call.has_init || r0 > 0) ? control.HasInitTrue() : nullptr;

      // conv_prep2 (default): the same bytes as conv_prep on a wider grid (r4d.h), what a wide 27B Model runs.
      conv_prep(mixed_qkv + r0 * conv_dim_, conv_dim_, w_.conv1d_weight, /*bias=*/nullptr,
                              states.ConvBase(), states.ConvSeqStride(), states.ConvDimStride(),
                              states.ConvTokStride(), cache_idx_dev, /*ci_stride=*/1, has_init_dev,
                              a_buf + r0 * H_, b_buf + r0 * H_, /*ab_stride=*/H_, /*ab_is_bf16=*/1, w_.A_log,
                              w_.dt_bias, q_buf + r0 * Hg_ * K_, k_buf + r0 * Hg_ * K_, v_buf + r0 * H_ * V_,
                              g_buf + r0 * H_, beta_buf + r0 * H_, cu_dev, /*N=*/1, ts, Hi, Hgi, Ki, Vi,
                              static_cast<int>(width_), kSoftplusThr, s);
      core::r4d::GdnKktSolve(k_buf + r0 * Hg_ * K_, beta_buf + r0 * H_, g_buf + r0 * H_, A_buf, cu_dev, /*N=*/1, ts,
                             Hi, Hgi, Ki, static_cast<int>(kScanChunk), s);
      // The span-safe scan, not the 27B's: this checkpoint's gate A = e^A_log reaches 158, so a 64-row chunk's gate span reaches the
      // thousands (layer 0 of the real model: 6505), where the shipped scan's midpoint split zeroes the near-diagonal weights
      // (r4d_gdn_chunk_scan_stable_*.hip; tools/reference/qwen4exp/gdn_probe.py reproduces the failure and the fix on the CPU).
      core::r4d::GdnChunkScanStable(q_buf + r0 * Hg_ * K_, k_buf + r0 * Hg_ * K_, v_buf + r0 * H_ * V_, A_buf,
                                    g_buf + r0 * H_, beta_buf + r0 * H_, h0, o_core + r0 * H_ * V_, ht_scratch, cu_dev,
                                    /*N=*/1, Hi, Hgi, Ki, Vi, static_cast<int>(kScanChunk), scale, s);
      // commit the scanned state into the slot (stream-ordered after the scan, before any later reader)
      R4DX_HIP_CHECK(hipMemcpyAsync(h0, ht_scratch, static_cast<size_t>(H_ * V_ * K_) * sizeof(float),
                                    hipMemcpyDeviceToDevice, s));
      if (sigmoid_gate_) {
        const int rc = q4x_gdn_sigmoid_gated_rmsnorm_h128_bf16(
            o_core + r0 * H_ * V_, z_buf + r0 * value_dim_, w_.norm_weight, out_core + r0 * value_dim_,
            /*rows=*/static_cast<int64_t>(ts) * H_, /*xrow=*/V_, /*zrow=*/V_, /*orow=*/V_, Vi, eps_, s);
        if (rc != 0) {
          Fail(std::string("q4x_gdn_sigmoid_gated_rmsnorm_h128_bf16 failed: ") +
               hipGetErrorString(static_cast<hipError_t>(rc)));
        }
      } else {
        core::r4d::GdnGatedRmsNorm(o_core + r0 * H_ * V_, z_buf + r0 * value_dim_, w_.norm_weight,
                                   out_core + r0 * value_dim_, static_cast<int64_t>(ts) * H_, V_, V_, V_, Vi, eps_,
                                   kActSilu, s);
      }
    }
  } else {
    // One token per kernel pair, the decode kernels: a decode step (T == 1), or a prefill that must not take the chunked
    // scan (prefill_recurrent: the same arithmetic as feeding the rows to decode one by one, see Qwen4ExpGdnCall).
    const int32_t* cu_dev = control.CuPair(1);
    const int32_t* sidx_dev = control.SidxBase(slot, states.MaxDecodeWindow());
    for (int64_t r = 0; r < T; ++r) {
      core::r4d::GdnConvUpdate(mixed_qkv + r * conv_dim_, conv_dim_, w_.conv1d_weight, /*bias=*/nullptr,
                               states.ConvBase(), states.ConvSeqStride(), states.ConvDimStride(), states.ConvTokStride(),
                               static_cast<int>(states.StateLenMax()), cache_idx_dev, /*ci_stride=*/1,
                               /*num_accepted=*/nullptr, q_buf, k_buf, v_buf, cu_dev, /*N=*/1, Hi, Hgi, Ki, Vi,
                               static_cast<int>(width_), /*max_query_len=*/static_cast<int>(states.MaxDecodeWindow()), s);
      core::r4d::GdnRecurrentUpdate(q_buf, k_buf, v_buf, a_buf + r * H_, b_buf + r * H_, /*ab_stride=*/H_, /*ab_is_bf16=*/1,
                                    w_.A_log, w_.dt_bias, states.RecurrentBase(), states.RecurrentSlotStride(),
                                    states.RecurrentHeadStride(), out_core + r * value_dim_, cu_dev, sidx_dev,
                                    /*indices_stride=*/states.MaxDecodeWindow(), /*num_accepted=*/nullptr,
                                    z_buf + r * value_dim_, w_.norm_weight, eps_, sigmoid_gate_ ? kActSigmoid : kActSilu,
                                    /*N=*/1, Hi, Hgi, Ki, Vi, scale, kSoftplusThr, s);
    }
  }

  // ---- out_proj ---------------------------------------------------------------------------------------------------
  // temporal weight loads: out_proj is the first weight stream after the recurrent update left the new state dirty in
  // the L2 (gdn_layer.cpp, docs/trellis-kernel.md 10.6); ApplyBf16Linear has no such flag, so a trellis out_proj takes
  // ApplyLinear directly.
  if (w_.out_proj->layout == Layout::kTrellis) {
    ApplyLinear(s, arena, *w_.out_proj, out_core, out, T, /*pre=*/nullptr, /*temporal_weight_loads=*/true);
  } else {
    qwen4exp::ApplyBf16Linear(s, arena, *w_.out_proj, out_core, out, T);
  }
}

}  // namespace r4dx::model
