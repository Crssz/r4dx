// r4dx::model::Model::ChunkRun -- the state ONE RunChunk call carries between its three phases
// (docs/pp-prefill.md section 1): ChunkPrologue (ids / meta upload, embedding gather, image splice, the
// per-chunk scopes), RunLayerRange (layers [first, last): the residual stream `cur` and the R3 / R2
// fusion's normed carry go in and come out) and ChunkEpilogue (MTP priming, final norm + lm_head, the
// synchronize, the DFlash injection, pos_ += T). The monolithic RunChunk is exactly their composition
// (RunLayerRange once over every layer); the PP-emulate mode (Model::SetPpEmulate) runs it as two ranges
// with the carry copied through host staging between them, which is what the two-GPU pipeline will do.
//
// Everything here used to be a local variable of RunChunk, so the three phases are pure code motion: the
// same statements in the same order, reading and writing these fields instead of locals. Private to
// model.cpp and pp_stage.cpp (not included from model.h: it pulls in the probe and the linear scopes).
#pragma once

#include <functional>
#include <optional>
#include <vector>

#include "debug_probe.h"
#include "linear.h"
#include "model.h"

namespace r4dx::model {

struct Model::ChunkRun {
  // The call's own arguments, as RunChunk received them.
  const std::vector<int32_t>& token_ids;
  const bool is_prefill_path;
  const bool want_logits;
  int32_t* const greedy_token_out;
  const SummaryRequest* const summary_out;
  const std::function<void()>* const overlap;
  const int64_t T;
  // A 256-row prefill super-chunk (RunChunk's comment): T == prefill_wide_active_ on the prefill path.
  const bool wide;

  // R4DX_CLOCK_PROBE / R4DX_PROFILE_LINEARS: this call's GPU span (debug_probe.h); inert (null probe)
  // otherwise. Ended by the epilogue before its stream synchronize.
  ProbeScope probe_call;

  // Filled by ChunkPrologue.
  int64_t hidden = 0;
  int64_t num_layers = 0;        // layers LOADED on this Model (the global stack's depth)
  bool has_init = false;
  bool dflash_capture_active = false;
  const int32_t* positions_dev = nullptr;
  const int32_t* seqused_dev = nullptr;
  const int32_t* rope_pos3 = nullptr;
  bool bounded = false;          // TP only: bounded submission of this prefill chunk
  int64_t unit_layers = 0;
  BackboneHadSigns had;

  // The per-chunk thread-local scopes, opened where the prologue ends and closed with the call: the M = 256
  // trellis GEMM and the int8 GEMM for a super-chunk's layers, the fake-quant accuracy switches for a
  // prompt-prefill chunk's. Declared in RunChunk's original order (destroyed in reverse).
  std::optional<ScopedTrellisM256> trellis_m256_scope;
  std::optional<ScopedTrellisI8> trellis_i8_scope;
  std::optional<ScopedFakeQuantAct> fakeq_scope;
  std::optional<ScopedFakeQuantW> fakeqw_scope;

  // The layer-to-layer carry (docs/pp-prefill.md 1.3): `cur` is the residual stream (buf_a_ / buf_b_,
  // swapped by every attention layer), `other` the other ping-pong buffer, `normed_in` /
  // `normed_in_epilogue` the R3 / R2 fusion's hand-over (buf_normed_ / buf_normed_pre_, null / none before
  // layer 0 and after the last layer).
  uint16_t* cur = nullptr;
  uint16_t* other = nullptr;
  const uint16_t* normed_in = nullptr;
  int normed_in_epilogue = 0;

  // PP-emulate timing (Model::PpEmulateConfig::timing): the epilogue's MTP priming and DFlash injection,
  // filled only when `timing` is set (it adds a stream synchronize after the priming).
  bool timing = false;
  double mtp_ms = 0.0;
  double inject_ms = 0.0;
  // PP-emulate test control: fill the arena with 0xFF after every layer's scratch release (RunLayerRange).
  bool poison_arena_layers = false;

  ChunkRun(Model& m, const std::vector<int32_t>& ids, bool prefill, bool logits, int32_t* greedy,
           const SummaryRequest* summary, const std::function<void()>* ov, int64_t t, bool is_wide)
      : token_ids(ids),
        is_prefill_path(prefill),
        want_logits(logits),
        greedy_token_out(greedy),
        summary_out(summary),
        overlap(ov),
        T(t),
        wide(is_wide),
        probe_call(m.probe_, m.stream_.get(), prefill ? "prefill" : "decode", t) {}
  ChunkRun(const ChunkRun&) = delete;
  ChunkRun& operator=(const ChunkRun&) = delete;
};

}  // namespace r4dx::model
