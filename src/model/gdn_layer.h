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

namespace r4dx::model {

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
  GdnLayer(const ModelConfig& cfg, const core::DeviceBuffer<uint16_t>& input_layernorm,
           const GdnWeights& w)
      : cfg_(cfg), input_layernorm_(input_layernorm), w_(w) {}

  // x: device bf16 [T, hidden] -- the block's input residual stream. x_out: device bf16
  // [T, hidden] -- may alias x. T: token count for this call (prefill: a chunk of the prompt,
  // <=64 recommended per the interim chunked-prefill path this model's GEMMs are speced for;
  // decode: 1 or a speculative window, must be <= GdnStateManager's max_decode_window). `control`
  // supplies this call's tiny cu/cache_idx/has_init/sidx device arrays (see GdnControlCache);
  // typically one instance shared by every GDN layer in a Model since these arrays are pure
  // functions of (T, slot), not of the layer's own weights.
  void Forward(core::Stream& stream, core::Arena& arena, GdnStateManager& states,
               GdnControlCache& control, const uint16_t* x, uint16_t* x_out, int64_t T,
               const GdnLayerParams& p);

 private:
  const ModelConfig& cfg_;
  const core::DeviceBuffer<uint16_t>& input_layernorm_;
  const GdnWeights& w_;
};

}  // namespace r4dx::model
