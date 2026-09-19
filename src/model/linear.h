// r4dx::model::ApplyLinear -- y[M,N] = x[M,K] @ W[N,K]^T for whichever layout `w` was loaded as,
// chunking M into <=64-row slices through the matching r4d skinny GEMM family
// (docs/architecture.md "Interim chunked prefill": every r4d_gemm_*_nt_m64 caps M at 64, so a
// wider prefill call is simply several of them back to back).
#pragma once

#include <cstdint>

#include "container.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/stream.hpp"

namespace r4dx::model {

// Tiling parameters for one r4d_gemm_*_nt_m64 launch. See linear.cpp's PickTuning for why a
// single constant satisfies every linear shape this model actually has.
struct LinearTuning {
  int WV, SK, MB, NPW, NT;
};
LinearTuning PickTuning(Layout layout, int64_t N, int64_t K);

// x: device bf16 [M, K], row-major, CONTIGUOUS (row stride exactly K -- every r4d_gemm_*_nt_m64
// entry point reads its A/C operands at a hardcoded stride of K/N respectively; there is no
// strided-view form to call into). y: device bf16 [M, N], row-major, contiguous, disjoint from x.
// `arena` supplies this call's activation-quant scratch (w4a16's f16 cast, w4a8's int8 quant,
// mxfp4's fp8 quant) for up to a 64-row chunk; the caller is responsible for giving the arena
// enough headroom and Reset()-ing it between top-level forward-pass steps (Arena's own contract --
// see r4dx/core/arena.hpp), not between individual ApplyLinear calls (this function does not reset
// it, so several ApplyLinear calls in the same layer share one growing allocation, which is the
// intended usage).
void ApplyLinear(core::Stream& stream, core::Arena& arena, const QuantLinear& w, const uint16_t* x,
                  uint16_t* y, int64_t M);

}  // namespace r4dx::model
