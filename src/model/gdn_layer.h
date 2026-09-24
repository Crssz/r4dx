// r4dx::model::GdnLayer -- one Gated DeltaNet decoder block (docs/architecture.md "GDN layer"):
// input rmsnorm -> in_proj_qkv/z/b/a -> conv+gate prep (or decode conv+recurrent update) ->
// out_proj -> residual add. Self-contained per the architecture doc's per-block pseudocode: it
// reads the block's raw INPUT hidden state (the residual stream), applies input_layernorm itself,
// and returns the NEW residual stream (already summed) -- the caller (the future assembly stage)
// does not apply input_layernorm or the residual add itself.
//
// Scope: one sequence per call (N=1) at a time. Batching multiple sequences through one launch
// (as the underlying r4d kernels support via their own N/cu/cache_idx arguments) is a scheduling
// concern the assembly stage owns; GdnLayer exposes the single-sequence primitive it needs.
#pragma once

#include <cstdint>

#include "container.h"
#include "gdn_state.h"
#include "model_config.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::core {
class TpComm;  // r4dx/core/tp_comm.hpp
}  // namespace r4dx::core

namespace r4dx::model {

class SpanAccumulator;  // profile_span.h -- forward-declared so this header does not need to
                         // include it; only gdn_layer.cpp's implementation does.

struct GdnLayerParams {
  int32_t slot = 1;        // GdnStateManager::SlotForSeq(seq) -- this sequence's WINDOW INDEX 0
                            // physical slot (>= 1); see GdnStateManager's file comment.
  bool is_prefill = true;  // true: conv_prep + kkt_solve + chunk_scan (chunked WY scan)
                            // false: conv_update + recurrent_update (sequential decode/verify)
  bool has_init = false;   // prefill only: does `slot` already hold valid conv/recurrent history
                            // from an earlier call (a continued prefill / chunked-prefill chunk)?
                            // false for a sequence's first-ever chunk.
  // decode/verify only (ignored when is_prefill): device int32[1], or nullptr. nullptr (the
  // default, and the only value any non-MTP caller ever passes) means "seed this call from window
  // index 0" -- exactly today's plain-sequential-decode behavior. A real MTP verify round passes
  // the PREVIOUS call's accepted-token count here (1..that call's own T) so this call's
  // conv_update/recurrent_update seed from the state snapshot that call left at window index
  // num_accepted-1 rather than window index 0 -- see GdnStateManager's file comment and
  // Model::DecodeStepMtpGreedy (model.cpp) for the caller side of this contract.
  const int32_t* num_accepted = nullptr;
};

class GdnLayer {
 public:
  // `comm` (docs/tp.md 6.2, site A1): non-owning; when non-null, Forward all-reduces the
  // row-parallel out_proj output across tensor-parallel ranks before the residual add, and `cfg`
  // is the RANK's config (ModelConfig::Shard). nullptr (TP=1) is exactly the pre-TP code path.
  GdnLayer(const ModelConfig& cfg, const core::DeviceBuffer<uint16_t>& input_layernorm,
           const GdnWeights& w, core::TpComm* comm = nullptr)
      : cfg_(cfg), input_layernorm_(input_layernorm), w_(w), comm_(comm) {}

  // x: device bf16 [T, hidden] -- the block's input residual stream. x_out: device bf16
  // [T, hidden] -- may alias x. T: token count for this call (prefill: a chunk of the prompt,
  // <=64 recommended per the interim chunked-prefill path this model's GEMMs are speced for;
  // decode: 1 or a speculative window, must be <= GdnStateManager's max_decode_window). `control`
  // supplies this call's tiny cu/cache_idx/has_init/sidx device arrays (see GdnControlCache);
  // typically one instance shared by every GDN layer in a Model since these arrays are pure
  // functions of (T, slot), not of the layer's own weights.
  // R3 fusion (docs/r9700.md P2/R3), mirrors Mlp::Forward's trailing params: `x_normed_in`
  // non-null skips this block's own input rmsnorm (the previous layer's Mlp already fused it).
  // `next_norm_weight` non-null replaces the final residual add with
  // r4dx_residual_rmsnorm_bf16(...), additionally writing this block's own Mlp-normed input into
  // `x_normed_out` (always the SAME layer's Mlp -- every GDN layer in this architecture is
  // immediately followed by its own Mlp, so a caller wiring this fusion passes that Mlp's
  // post_attention_layernorm here unconditionally).
  // `prof` (Milestone 3 profiling pass, docs/r9700.md R5/Q3): non-null only under r4dx-cli
  // --profile (Model::DecodeStepProfiled/PrefillProfiled) -- wraps every kernel this call launches
  // in its own named hipEvent span (see profile_span.h's file comment for the naming convention and
  // the zero-overhead guarantee when this is nullptr, the case on every real decode/prefill path).
  // R2/P2 (docs/r9700.md), appended after `prof` so every pre-existing positional call site
  // (which all end at `prof`) is unaffected: `x_normed_pre_epilogue`/`_data`/`_scale`, when
  // `x_normed_in` is also non-null, is the SAME producer's fused quant epilogue for `x_normed_in`
  // (kernels.h's r4dx_epilogue -- what the previous layer's Mlp emitted alongside its own
  // x_normed_out) -- consumed directly by in_proj_qkv/in_proj_z's ApplyLinear calls in place of
  // this call's own quant launch, WHEN it matches that weight's actual layout (checked per-weight,
  // see gdn_layer.cpp's z_shares_pre comment; falls back to a local quant launch otherwise, exactly
  // as if this had been left null). `next_epilogue`/`next_epilogue_out`/`next_epilogue_scale`,
  // when `next_norm_weight` is also non-null, requests this call's own residual+rmsnorm epilogue
  // (r4dx_residual_rmsnorm_bf16) ALSO emit x_normed_out's fused quant epilogue into
  // `next_epilogue_out`/`next_epilogue_scale` (for the immediately-following Mlp's gate_up to
  // consume as ITS x_normed_pre) -- ignored (no epilogue computed) when next_epilogue is
  // r4dx_epilogue_none, the default.
  void Forward(core::Stream& stream, core::Arena& arena, GdnStateManager& states,
               GdnControlCache& control, const uint16_t* x, uint16_t* x_out, int64_t T,
               const GdnLayerParams& p, const uint16_t* x_normed_in = nullptr,
               const uint16_t* next_norm_weight = nullptr, uint16_t* x_normed_out = nullptr,
               SpanAccumulator* prof = nullptr, int x_normed_pre_epilogue = 0,
               const void* x_normed_pre_data = nullptr, const float* x_normed_pre_scale = nullptr,
               int next_epilogue = 0, void* next_epilogue_out = nullptr,
               float* next_epilogue_scale = nullptr);

 private:
  const ModelConfig& cfg_;
  const core::DeviceBuffer<uint16_t>& input_layernorm_;
  const GdnWeights& w_;
  core::TpComm* comm_ = nullptr;
};

}  // namespace r4dx::model
