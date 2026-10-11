// PleLayer -- the device part of the qwen4_exp PLE block (HF Qwen4ExpTextPLELayer, modular_qwen4_exp.py:711-783,
// docs/qwen4exp-plan.md 2.3, 4.8): everything after the n-gram table lookup. The caller (the host n-gram reader) hands in
// the gathered n-gram rows `emb` [rows, embed_dim] bf16 (16 heads x 160 channels concatenated), already on the device.
//
//   kp = key_proj(emb), v = value_proj(emb)                     two ApplyBf16Linear GEMMs
//   gate, gated value = sigmoid(gate) * v                       ONE fused launch, q4x_ple_gate_value_bf16: norm_key(kp),
//                                                               norm_query(h), their per-stream dot, the value broadcast
//   gvn = norm_conv(gated_value)                                q4x_grouped_rmsnorm_bf16
//   h  += gated_value + silu(dilated depthwise conv(gvn))       q4x_ple_conv_bf16, with the persistent conv state
//
// The layer owns the conv state [channels, 9] bf16 (the last (kernel - 1) * dilation = 9 steps of gvn per channel,
// oldest first), zero-initialised, which is what HF's cache holds after a first call (update_conv_state pads with
// zeros). Chunks of any size, including 1, compose exactly; the state is part of a prefix-reuse checkpoint, so it can
// be copied out and back.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>

#include "quant_linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"

namespace r4dx::model::qwen4exp {

struct PleConfig {
  int hc = 4;
  int hidden = 2560;
  int embed_dim = 2560;     // config ple_embed_dim
  int conv_kernel = 4;      // config ple_conv_kernel_size
  int conv_dilation = 3;    // config ngram_size (the conv's dilation, :728)
  float eps = 1e-6f;        // config rms_norm_eps
};

// Borrowed device weights; they must outlive the PleLayer.
struct PleWeights {
  const QuantLinear* key_proj = nullptr;    // bf16 [hc * hidden, embed_dim]
  const QuantLinear* value_proj = nullptr;  // bf16 [hidden, embed_dim]
  const uint16_t* norm_key = nullptr;       // bf16 [hc * hidden], raw weights (kernels apply 1 + w)
  const uint16_t* norm_query = nullptr;
  const uint16_t* norm_conv = nullptr;
  const uint16_t* conv1d = nullptr;         // bf16 [hc * hidden, conv_kernel]  (HF weight [C, 1, K])
};

class PleLayer {
 public:
  // Throws std::runtime_error on a config the kernels do not serve (conv_kernel 4, dilation 3) or a weight shape
  // mismatch. The conv state starts zeroed (zeroing is synchronous).
  PleLayer(const PleConfig& cfg, const PleWeights& w);

  // Elements (bf16) of the conv state: channels * (conv_kernel - 1) * conv_dilation.
  size_t StateElems() const { return state_.size(); }
  size_t StateBytes() const { return state_.bytes(); }

  // Arena bytes one Forward call allocates for `rows` rows.
  size_t ArenaBytes(int64_t rows) const;

  // Back to the "no history" state (as a fresh sequence / HF's empty cache).
  void Reset(hipStream_t stream);

  // Snapshot / restore of the conv state to / from a caller buffer of StateElems() bf16 (device, ordered on `stream`).
  void SaveState(hipStream_t stream, uint16_t* dst) const;
  void LoadState(hipStream_t stream, const uint16_t* src);

  // h [rows, hc * hidden] bf16 (in / out): h += PLE output for `rows` consecutive tokens; emb [rows, embed_dim] their
  // gathered n-gram rows. Advances the conv state by `rows` steps.
  void Forward(hipStream_t stream, core::Arena& arena, uint16_t* h, const uint16_t* emb, int64_t rows);

  const PleConfig& config() const { return cfg_; }

 private:
  PleConfig cfg_;
  PleWeights w_;
  core::DeviceBuffer<uint16_t> state_;
};

}  // namespace r4dx::model::qwen4exp
