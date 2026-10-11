// src/model/qwen4exp/include/r4dx/model/qwen4exp/moe_trellis_grouped.h
//
// Grouped (all selected experts in one launch) trellis decode MoE, M = 1. The kernels are instantiated for exactly one
// geometry: hidden 2560, expert intermediate 640 (gate_up N 1280, two parts), at most 16 experts per token, at most
// 1024 experts, trellis bits 3 or 4. Every launcher returns the launch's hipError_t (hipGetLastError()); a bits value
// outside {3, 4} returns hipErrorInvalidValue. Callers validate the geometry once (Qwen4ExpStage's constructor) and
// check the returned code with Check (common.h).
//
// Tickets: the GEMV kernels index the split-group tickets at `expert * 10` (gate_up) / `expert * 20` (down), so
// all_tickets / all_tickets_down must be dedicated zeroed arrays of experts * 10 / experts * 20 u32; a launch that
// completes leaves them zero again.
#pragma once
#include <hip/hip_runtime.h>

#include <cstdint>

namespace r4dx {
namespace model {
namespace qwen4exp {
// Input: ids [top_k] in descending score order, weights [top_k] in descending score order.
// Output: ids_sorted [top_k], weights_sorted [top_k] in ascending expert ID order.
hipError_t GroupedMoeDecodeSort(const int32_t* ids_in, const float* weights_in,
                                int32_t* ids_sorted_out, float* weights_sorted_out,
                                int top_k, void* stream);

// Transforms x [1, 2560] bf16 into A0 [top_k, 2560] fp16 and A1 [top_k, 2560] fp16
// for the top_k experts in a single launch.
hipError_t GroupedTrellisInputGateUp(const uint16_t* x, const int32_t* ids_sorted,
                                     const float* all_suh, uint16_t* out_a0, uint16_t* out_a1,
                                     int top_k, void* stream);

// Grouped trellis GEMV for gate_up (M=1, K=2560, N=1280, parts=2, top_k<=16).
// Output: out_c [top_k, 1280] bf16.
hipError_t GroupedTrellisGemvGateUp(int kb, const uint16_t* in_a0, const uint16_t* in_a1,
                                    const int32_t* ids_sorted,
                                    const uint32_t* all_w, const float* all_svh,
                                    uint16_t* out_c, float* ws, unsigned* all_tickets,
                                    int top_k, void* stream);

// Fuses silu(gate) * up (intermediate 640) with Trellis input transform for down
// into out_a0 [top_k, 640] fp16 in a single launch.
hipError_t GroupedSiluMulTrellisInputDown(const uint16_t* gu, const int32_t* ids_sorted,
                                          const float* all_suh_down, uint16_t* out_a0,
                                          int top_k, void* stream);

// Grouped trellis GEMV for down (M=1, K=640, N=2560, parts=1, top_k<=16).
// Output: out_yp [top_k, 2560] bf16.
hipError_t GroupedTrellisGemvDown(int kb, const uint16_t* in_a0,
                                  const int32_t* ids_sorted,
                                  const uint32_t* all_w_down, const float* all_svh_down,
                                  uint16_t* out_yp, float* ws, unsigned* all_tickets_down,
                                  int top_k, void* stream);

// Combines the top_k expert outputs and shared expert in-place into out [1, 2560] bf16.
hipError_t GroupedMoeDecodeCombine(const uint16_t* yp, const float* weights_sorted,
                                   const uint16_t* shared_y, const uint16_t* shared_gate_logit,
                                   uint16_t* out, int top_k, int hidden, void* stream);

}  // namespace qwen4exp
}  // namespace model
}  // namespace r4dx
