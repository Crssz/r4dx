// r4dx::model::Qwen4ExpGdn -- the Gated DeltaNet mixer of one qwen4_exp layer (docs/qwen4exp-plan.md 2.1, 4.5, 4.11).
//
// The 27B's GdnLayer::Forward bakes its residual + input-rmsnorm contract into the layer (it reads the residual
// stream, applies input_layernorm and returns the new residual); here the block input is already the hyper-connection
// mix x [T, hidden] and the block output goes back to the caller's gated residual, so this class is the mixer alone:
//
//   mixed_qkv = in_proj_qkv(x), z = in_proj_z(x), a = in_proj_a(x), b = in_proj_b(x)      ApplyBf16Linear / bf16 GEMM
//   prefill  : conv_prep (causal conv1d + silu + split + gate prep), kkt_solve, chunk_scan (+ state commit),
//              gated RMSNorm                                                               libr4d, as gdn_layer.cpp
//   decode   : conv_update + recurrent_update with the gated norm fused into its epilogue  libr4d, as gdn_layer.cpp
//   out = out_proj(gated norm output)
//
// The ONE numeric difference to the 27B is the gated norm's gate activation (config output_gate_type): sigmoid here
// (prefill: the HF-rounded q4x_gdn_sigmoid_gated_rmsnorm_h128_bf16 of qwen4exp/gdn_sigmoid_norm.h; decode: libr4d's
// fused epilogue with act code 1), silu for a config that asks for it (libr4d act code 0 in both paths).
//
// State: the caller's GdnStateManager (max_seqs 1, window 1, one per GDN layer) and the stage-wide GdnControlCache.
// Single sequence, slot GdnStateManager::SlotForSeq(0).
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "gdn_state.h"
#include "qwen4exp_config.h"
#include "quant_linear.h"
#include "r4dx/core/arena.hpp"

namespace r4dx::model {

// Borrowed device weights (they must outlive the Qwen4ExpGdn); what qwen4exp::GdnWeights of the container holds.
struct Qwen4ExpGdnRefs {
  const QuantLinear* in_proj_qkv = nullptr;  // [2 * key_dim + value_dim, hidden]
  const QuantLinear* in_proj_z = nullptr;    // [value_dim, hidden]
  const uint16_t* in_proj_a = nullptr;       // bf16 [num_v_heads, hidden]
  const uint16_t* in_proj_b = nullptr;       // bf16 [num_v_heads, hidden]
  const uint16_t* conv1d_weight = nullptr;   // bf16 [conv_dim, 1, kernel]
  const float* A_log = nullptr;              // fp32 [num_v_heads]
  const float* dt_bias = nullptr;            // fp32 [num_v_heads]
  const float* norm_weight = nullptr;        // fp32 [head_v_dim] (the plain norm weight, not 1 + w)
  const QuantLinear* out_proj = nullptr;     // [hidden, value_dim]
};

struct Qwen4ExpGdnCall {
  bool is_prefill = true;  // chunked scan (any T >= 1) vs the sequential decode kernels (T == 1)
  bool has_init = false;   // prefill only: the state slot already holds the history of an earlier call
  // prefill only. conv_prep: 2 (default) = libr4d's r4d_gdn_conv_prep2, 1 = the original r4d_gdn_conv_prep (the same bytes
  // on a narrower grid). seq_slice: 0 = automatic (one slice when T <= 64 or a multiple of 64, else consecutive 64-row
  // slices), -1 = the whole call as one slice, 64 = consecutive 64-row slices (the last one short).
  int conv_prep = 2;
  int64_t seq_slice = 0;
  // prefill only: run the rows through the recurrent decode kernels one token at a time instead of the chunked scan (the
  // arithmetic of feeding them to a decode step; T rows cost T kernel pairs). The slow reference path: it equals HF's recurrent
  // rule to bf16 noise, and the chunked scan (libr4d's span-safe r4d_gdn_chunk_scan_stable) is held to it by the layer-isolated
  // tests (tests/model/qwen4exp/test_model_q4exp_layers.cpp). (The earlier shipped r4d_gdn_chunk_scan clamped e^{G_i-G_j} and
  // diverged from the recurrent rule when a 64-row chunk's gate span was large; that was a kernel defect, not model conditioning.)
  bool prefill_recurrent = false;

  // ---- speculative window (GdnLayer::Forward's protocol, docs/gdn-write-once.md) ---------------------------------------
  // Only with a GdnStateManager that has MaxDecodeWindow() > 1 AND write-once state (the stage's spec_window > 1); a window-1
  // manager ignores all three. Meaningful for !is_prefill calls (a decode step, T == 1, or a verify window, T <= window):
  //   verify        : log the T rows (conv_update + the write-once kernel in LOG mode), commit nothing -- the owner says how many
  //                   rows are committed afterwards (Qwen35MoeStage::CommitVerified) and the NEXT call replays that prefix.
  //   num_accepted  : device int32, the row count the previous window committed: the conv history offset (naccept - 1). null =
  //                   the history sits at offset 0 (nothing speculated since the last prefill).
  //   gdn_pending   : device int32, the committed prefix of the previous window's log still to be replayed into the state;
  //                   null when nothing is pending.
  bool verify = false;
  const int32_t* num_accepted = nullptr;
  const int32_t* gdn_pending = nullptr;
};

// The Gated DeltaNet geometry of one mixer, for architectures that are not qwen4_exp (qwen3_5_moe, docs/qwen35moe-plan.md):
// the plain fields Qwen4ExpGdn reads off Qwen4ExpConfig. `sigmoid_gate` is the gated norm's gate activation (sigmoid for
// qwen4_exp, silu for the Qwen3.5 hybrid).
struct GdnGeometry {
  int64_t hidden = 0;
  int64_t key_heads = 0, value_heads = 0, key_dim = 0, value_dim = 0;
  int64_t conv_kernel = 4;
  double rms_eps = 1e-6;
  bool sigmoid_gate = false;
};

class Qwen4ExpGdn {
 public:
  // Throws std::runtime_error for a geometry libr4d's GDN kernels do not serve (key / value head dim 128, conv width 4,
  // 16 | 48 head grouping is whatever the kernels check) or a missing / mis-shaped weight.
  Qwen4ExpGdn(const Qwen4ExpConfig& cfg, const Qwen4ExpGdnRefs& w);
  Qwen4ExpGdn(const GdnGeometry& geo, const Qwen4ExpGdnRefs& w);

  // Arena bytes one Forward(T) allocates (ApplyLinear's own scratch comes on top, as for any caller).
  size_t ArenaBytes(int64_t T) const;

  // x: device bf16 [T, hidden]; out: device bf16 [T, hidden], disjoint from x. T >= 1; decode (!is_prefill) needs
  // T == 1 (the state manager's window). Enqueues on `stream`, never synchronizes (the control-array cache's first use
  // of a key does one blocking upload, as in GdnLayer).
  void Forward(hipStream_t stream, core::Arena& arena, GdnStateManager& states, GdnControlCache& control,
               const uint16_t* x, uint16_t* out, int64_t T, const Qwen4ExpGdnCall& call) const;

  int64_t KeyHeads() const { return Hg_; }
  int64_t ValueHeads() const { return H_; }
  int64_t ConvDim() const { return conv_dim_; }
  int64_t ValueDim() const { return value_dim_; }

 private:
  Qwen4ExpGdnRefs w_;
  int64_t hidden_, Hg_, H_, K_, V_, width_, key_dim_, value_dim_, conv_dim_;
  float eps_;
  bool sigmoid_gate_;
};

}  // namespace r4dx::model
