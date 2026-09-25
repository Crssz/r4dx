// r4dx::model::ApplyLinear -- y[M,N] = x[M,K] @ W[N,K]^T for whichever layout `w` was loaded as,
// chunking M into <=64-row slices through the matching r4d skinny GEMM family
// (docs/architecture.md "Interim chunked prefill": every r4d_gemm_*_nt_m64 caps M at 64, so a
// wider prefill call is simply several of them back to back).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "quant_linear.h"
#include "r4dx/core/arena.hpp"

namespace r4dx::model {

// Tiling parameters for one r4d_gemm_*_nt_m64 launch.
struct LinearTuning {
  int WV, SK, MB, NPW, NT;
};

// One measured (layout, N, K, M) -> LinearTuning row (tools/profile/tune_gemm.py's sweep output,
// src/model/gemm_tuning_table.inc, included by linear.cpp). `M` is the exact chunk row count the
// row was measured at (one of the M-bands tools/profile/tune_gemm.py swept: 1,2,4,8,16,32,64) --
// PickTuning below looks up the smallest tabulated M that is >= the caller's chunk M (a tuning
// measured for a wider M is still legal, just not necessarily optimal, for a narrower call; the
// reverse -- a narrower M's tuning serving a wider call -- is also legal, but is more likely to
// underutilize the GPU on the wider call, so PickTuning rounds up, never down). One exception: a
// chunk of M <= 16 rows (one row tile) always gets the M=1 band's tuning (NT aside), whatever the
// M=2..16 bands measured. WV/MB/NPW/NT only choose how the work is laid out, but SK also sets the
// order the partial sums are added in, so two M-bands with different SK give the same row
// different last bits -- and a speculative verify row must equal the single-row decode row exactly
// (linear.cpp's kRowTile).
struct GemmTuningRow {
  Layout layout;
  int64_t N, K, M;
  LinearTuning tuning;
};

// Picks a WV/SK/MB/NPW/NT for a (layout, N, K) GEMM chunk of M rows. Looks up
// src/model/gemm_tuning_table.inc's measured table first (tools/profile/tune_gemm.py, keyed by
// (layout,N,K,M-band)); falls back to a hand-derived, constraint-legal-for-every-shape-this-model-
// has default (see linear.cpp) when the table has no row for this exact (layout,N,K) shape --
// e.g. a shape the sweep did not cover, or the table file is missing/empty (this model's tests use
// a 4-layer container with the same shapes as the real 64-layer one, so in practice every shape
// PickTuning ever sees during normal operation IS covered by the table once tune_gemm.py has run;
// the fallback exists for robustness, not because it is expected to fire in production).
LinearTuning PickTuning(Layout layout, int64_t N, int64_t K, int64_t M);

// Tensor parallel (docs/tp.md 2.7): marks the CALLING thread as one that runs a TP=2 rank's Model
// (Model::Load calls it on every load with ModelOptions::tp.world > 1 -- true on a rank thread in
// TpModel, the main thread in tool_tp_step_bench -- and clears it when that load throws). On such a
// thread PickTuning looks up src/model/gemm_tuning_table_tp2.inc (the per-rank (N,K) shapes, same
// M-band and w4a16-group rules) FIRST, then the main table, then the fallback; its cache is
// thread_local and keyed on the flag too. A TP=1 load sets it false, so a TP=1 Model consults the
// main table alone, exactly as before, even on a thread that loaded a rank earlier. The
// TP rows live in their own table because one per-rank key -- (w4a16, 17408, 5120) -- is also the
// TP=1 DFlash drafter's gate_proj/up_proj, which the main table deliberately leaves untuned.
void SetTp2TuningForThisThread(bool enabled);

// The r4dx_epilogue (kernels.h) a fused producer must emit to feed `layout`'s GEMM directly --
// r4dx_epilogue_none for kBf16 (which never quantizes its activation input), r4dx_epilogue_f16 for
// kW4a16, r4dx_epilogue_int8_fraga8 for kW4a8, r4dx_epilogue_fp8_e4m3_row for kMxfp4. Shared by
// every ApplyLinear caller that wants to pre-fuse its producer's quant epilogue (docs/r9700.md
// R2/P2) so both sides of the wiring agree on the mapping in exactly one place.
int EpilogueForLayout(Layout layout);

// A producer's already-quantized activation (docs/r9700.md R2/P2's fused epilogue output,
// kernels.h's r4dx_epilogue), ready to feed `w`'s GEMM directly. `epilogue` must equal
// EpilogueForLayout(w.layout) exactly -- ApplyLinear throws otherwise, rather than silently
// reinterpreting bytes in the wrong format. `data` is [M,K] contiguous in the format `epilogue`
// selects (f16 uint16_t, fp8e4m3 uint8_t, or int8 fragA8-permuted int8_t); `scale` is [M] fp32,
// unused (may be nullptr) for r4dx_epilogue_f16.
struct PreQuantizedActivation {
  int epilogue = 0;  // r4dx_epilogue_none means "no pre-quantized input provided"
  const void* data = nullptr;
  const float* scale = nullptr;
};

// x: device bf16 [M, K], row-major, CONTIGUOUS (row stride exactly K -- every r4d_gemm_*_nt_m64
// entry point reads its A/C operands at a hardcoded stride of K/N respectively; there is no
// strided-view form to call into). y: device bf16 [M, N], row-major, contiguous, disjoint from x.
// `arena` supplies this call's activation-quant scratch (w4a16's f16 cast, w4a8's int8 quant,
// mxfp4's fp8 quant) for up to a 64-row chunk; the caller is responsible for giving the arena
// enough headroom and Reset()-ing it between top-level forward-pass steps (Arena's own contract --
// see r4dx/core/arena.hpp), not between individual ApplyLinear calls (this function does not reset
// it, so several ApplyLinear calls in the same layer share one growing allocation, which is the
// intended usage).
// `stream` is a raw hipStream_t (not core::Stream&) so this can be called from components that
// only have a raw stream handle (e.g. r4dx::model::attention::AttentionLayer, which owns its
// stream as a plain hipStream_t parameter) without pulling in core::Stream's RAII ownership --
// r4dx::core::Stream converts implicitly (operator hipStream_t()), so every existing call site
// that passes a core::Stream& is unaffected.
// `pre`, when non-null and pre->epilogue != r4dx_epilogue_none, skips this call's own internal
// quant/cast launch entirely and feeds `pre->data`/`pre->scale` (offset per <=64-row chunk exactly
// like `x` is) straight to the GEMM -- see PreQuantizedActivation's doc above. `x` is still
// required even when `pre` is given (kBf16 always reads it directly; a caller that only produced a
// quantized epilogue for a NON-bf16 layout does not need to also keep the plain bf16 buffer alive
// for THIS call, but ApplyLinear does not special-case that -- every existing caller already has
// both).
void ApplyLinear(hipStream_t stream, core::Arena& arena, const QuantLinear& w, const uint16_t* x,
                  uint16_t* y, int64_t M, const PreQuantizedActivation* pre = nullptr);

}  // namespace r4dx::model
