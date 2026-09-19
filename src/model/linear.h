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
// reverse -- a narrower M's tuning serving a wider call -- is also legal since WV/SK/MB/NPW/NT
// only choose how the same sum is tiled, never what it computes, but is more likely to
// underutilize the GPU on the wider call, so PickTuning always rounds up, never down).
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
void ApplyLinear(hipStream_t stream, core::Arena& arena, const QuantLinear& w, const uint16_t* x,
                  uint16_t* y, int64_t M);

}  // namespace r4dx::model
