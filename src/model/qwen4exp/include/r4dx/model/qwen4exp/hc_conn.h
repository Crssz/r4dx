// HyperConn -- one qwen4_exp gated-residual block (HF Qwen4ExpTextGatedResidual, modular_qwen4_exp.py:531-559) end to end:
//
//   n   = grouped_rmsnorm(h) * (1 + norm_w)                  hc_kernels.h q4x_grouped_rmsnorm_bf16
//   m   = sigmoid(up(silu(down(n) / hc)))                    two bf16 GEMMs + q4x_hc_silu_scaled_bf16
//   x   = mean over the hc streams of (m * n)                q4x_hc_mix_bf16          -> block input [rows, hidden]
//   inj = 2 * sigmoid(block_inject(n) / hc)                  one GEMM (N = hc) + q4x_hc_inject_gate_bf16
//   h  += broadcast(block_out) * inj  per stream             q4x_hc_inject_bf16       (after the attention / MoE block)
//
// The final hyper_connection_mixer (HF `use_combine=False`) is the same without block_inject and returns x only.
// All three GEMMs go through ApplyBf16Linear (hc_bf16_gemm.h): `up` has K = 320, which ApplyLinear's tuning fallback
// refuses, and `inject` has N = 4. The residual stream is bf16 (plan 4.11 item 4).
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>

#include "quant_linear.h"
#include "r4dx/core/arena.hpp"

namespace r4dx::model::qwen4exp {

struct HyperConnConfig {
  int hc = 4;          // residual streams (config hc_count)
  int hidden = 2560;   // channels per stream
  int lowrank = 320;   // config hc_lowrank
  float eps = 1e-6f;   // config rms_norm_eps
};

// Weights of one gated residual. The pointers are borrowed and must outlive the HyperConn.
struct HyperConnWeights {
  const uint16_t* norm = nullptr;      // device bf16 [hc * hidden], the RAW checkpoint weight (kernels apply 1 + w)
  const QuantLinear* down = nullptr;   // [lowrank, hc * hidden]
  const QuantLinear* up = nullptr;     // [hc * hidden, lowrank]
  const QuantLinear* inject = nullptr; // [hc, hc * hidden]; nullptr for the final mixer
};

class HyperConn {
 public:
  // Throws std::runtime_error when a weight's shape disagrees with `cfg` or a required weight is missing.
  HyperConn(const HyperConnConfig& cfg, const HyperConnWeights& w);

  // Arena bytes one Prep / Mix / ApplyBlock call allocates for `rows` rows (the scratch of the GEMMs themselves
  // comes on top, as for any ApplyLinear caller).
  size_t ArenaBytes(int64_t rows) const;

  // x_out [rows, hidden] = the block input; inj_out [rows, hc] = the injection weights (both bf16, device; inj_out
  // may be null only when the HyperConn has no inject weight).
  void Prep(hipStream_t stream, core::Arena& arena, const uint16_t* h, int64_t rows, uint16_t* x_out,
            uint16_t* inj_out) const;

  // h_out = h + broadcast(block_out) * inj (h_out may alias h). block_out [rows, hidden].
  void Inject(hipStream_t stream, const uint16_t* h, const uint16_t* block_out, const uint16_t* inj, int64_t rows,
              uint16_t* h_out) const;

  // The final mixer: x_out [rows, hidden] only. Requires a HyperConn without inject weights.
  void Mix(hipStream_t stream, core::Arena& arena, const uint16_t* h, int64_t rows, uint16_t* x_out) const;

  // The whole block in one call: Prep, `block(x, block_out)` (x [rows, hidden] in, block_out [rows, hidden] out; the
  // callback may allocate scratch from `arena` and runs on `stream`), then Inject in place: h [rows, hc * hidden] is
  // updated. Requires inject weights.
  using BlockFn = std::function<void(const uint16_t* x, uint16_t* block_out)>;
  void ApplyBlock(hipStream_t stream, core::Arena& arena, uint16_t* h, int64_t rows, const BlockFn& block) const;

  const HyperConnConfig& config() const { return cfg_; }

 private:
  HyperConnConfig cfg_;
  HyperConnWeights w_;
};

// h0[r, :] = embedding row ids[r] repeated hc times (`hidden_states.repeat(1, 1, hc_count)`), the gather fused with the
// repeat. table bf16 [vocab, hidden]; ids int32 device [rows]; out bf16 [rows, hc * hidden]. Out-of-range ids give NaN
// rows (hc_kernels.h q4x_embed_gather_repeat_bf16).
void EmbedRepeat(hipStream_t stream, const uint16_t* table, int64_t vocab, const int32_t* ids, int64_t rows,
                 int hidden, int hc, uint16_t* out);

}  // namespace r4dx::model::qwen4exp
