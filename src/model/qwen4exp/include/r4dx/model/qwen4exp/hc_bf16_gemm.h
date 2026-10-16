// ApplyBf16Linear -- y[M,N] = x[M,K] @ W[N,K]^T through r4dx::model::ApplyLinear, plus the shapes ApplyLinear's tuning
// fallback refuses for a bf16 weight. qwen4_exp has several bf16 linears the 27B never had:
//
//   hc up     N 10240, K  320      shared-expert down  N 2560, K 640      (K not a multiple of 256)
//   hc inject N     4, K 10240     shared_gate         N    1, K 2560     (N below one 16-row tile)
//
// linear.cpp's FallbackTuning requires K % 256 == 0 and N % 16 == 0 (those shapes have no tuning-table row), so
// ApplyLinear would throw for them although libr4d's r4d_gemm_bf16_nt_m64 serves them (K % (SK * 16) == 0, fragment
// rows past N are clamped). ApplyBf16Linear therefore sends a kBf16 weight through ApplyLinear whenever its fallback
// accepts the shape, and otherwise calls the same libr4d kernel itself with the same WV / SK / MB the fallback uses
// (WV 4, SK 4, MB 1; SK 16 and WV 1 for N <= 16 so a 4-row weight still fills a workgroup), chunking M by 64 exactly
// as ApplyLinear does. Every other layout (the trellis QuantLinears of later milestones) goes to ApplyLinear
// unchanged.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "linear.h"
#include "quant_linear.h"
#include "r4dx/core/arena.hpp"

namespace r4dx::model::qwen4exp {

// x: device bf16 [M, w.K] contiguous; y: device bf16 [M, w.N] contiguous, disjoint from x. Throws std::runtime_error
// on a shape the kernel rejects.
void ApplyBf16Linear(hipStream_t stream, core::Arena& arena, const QuantLinear& w, const uint16_t* x, uint16_t* y,
                     int64_t M);

// A bf16 QuantLinear owning a copy of host weights W[N, K] (row-major bf16 bits); for tests and loaders that hold
// plain bf16 tensors.
QuantLinear MakeBf16Linear(const uint16_t* host_w, int64_t N, int64_t K);

}  // namespace r4dx::model::qwen4exp
