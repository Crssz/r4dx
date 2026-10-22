#include "qwen35moe_grouped_decode.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>

#include "r4dx/core/error.hpp"
#include "r4dx/model/qwen4exp/common.h"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/moe_kernels.h"
#include "r4dx/model/qwen4exp/moe_trellis_grouped.h"

namespace r4dx::model {

namespace {

constexpr const char* kCtx = "Qwen35GroupedMoeDecode";

void Check(int code, const char* what) { qwen4exp::Check(kCtx, code, what); }

}  // namespace

bool Qwen35GroupedMoeDecode::Supported(int64_t hidden, int64_t inter, int64_t shared_inter, int64_t experts, int64_t top_k,
                                       const Qwen35MoeTrellisPtrs& p) {
  const auto bits_ok = [](int b) { return b == 3 || b == 4; };
  return p.gate_up_w != nullptr && p.gate_up_suh != nullptr && p.gate_up_svh != nullptr && p.down_w != nullptr &&
         p.down_suh != nullptr && p.down_svh != nullptr && bits_ok(p.gate_up_bits) && bits_ok(p.down_bits) &&
         hidden == kHidden && inter == kInter && shared_inter == kInter && top_k >= 1 && top_k <= 16 && experts >= top_k &&
         experts <= 1024;
}

Qwen35GroupedMoeDecode::Qwen35GroupedMoeDecode(int experts, int top_k, bool norm_topk, int max_rows)
    : experts_(experts), top_k_(top_k), norm_topk_(norm_topk), max_rows_(max_rows) {
  if (max_rows < 1 || max_rows > kMaxRows || max_rows * top_k > qwen4exp::kMoePlanMaxPairs) {
    throw std::invalid_argument("Qwen35GroupedMoeDecode: max_rows must be in [1, 16] with max_rows * top_k <= 256");
  }
  const size_t k = static_cast<size_t>(top_k), E = static_cast<size_t>(experts);
  logits_ = core::DeviceBuffer<uint16_t>(E);
  ids_ = core::DeviceBuffer<int32_t>(k);
  weights_ = core::DeviceBuffer<float>(k);
  ids_sorted_ = core::DeviceBuffer<int32_t>(k);
  rank_ = core::DeviceBuffer<int32_t>(k);
  a0_gu_ = core::DeviceBuffer<uint16_t>(k * kHidden);
  a1_gu_ = core::DeviceBuffer<uint16_t>(k * kHidden);
  gu_ = core::DeviceBuffer<uint16_t>(k * 2 * kInter);
  ws_gu_ = core::DeviceBuffer<float>(k * qwen4exp::kGroupedQ35GuWsPerSlot);
  a0_down_ = core::DeviceBuffer<uint16_t>(k * kInter);
  yp_ = core::DeviceBuffer<uint16_t>(k * kHidden);
  ws_down_ = core::DeviceBuffer<float>(k * qwen4exp::kGroupedQ35DownWsPerSlot);
  shared_g_ = core::DeviceBuffer<uint16_t>(kInter);
  shared_u_ = core::DeviceBuffer<uint16_t>(kInter);
  shared_h_ = core::DeviceBuffer<uint16_t>(kInter);
  shared_y_ = core::DeviceBuffer<uint16_t>(kHidden);
  shared_gate_logit_ = core::DeviceBuffer<uint16_t>(1);
  // The GEMV kernels index the split-group tickets at expert * (N / 128): dedicated zeroed arrays.
  tickets_gu_ = core::DeviceBuffer<uint32_t>(E * qwen4exp::kGroupedQ35GuTickets);
  tickets_down_ = core::DeviceBuffer<uint32_t>(E * qwen4exp::kGroupedQ35DownTickets);
  tickets_gu_.Zero();
  tickets_down_.Zero();
  if (max_rows > 1) {
    const size_t R = static_cast<size_t>(max_rows), P = R * k;
    const size_t G = std::min<size_t>(E, P);  // the most distinct experts a call can pick
    r_logits_ = core::DeviceBuffer<uint16_t>(R * E);
    r_ids_ = core::DeviceBuffer<int32_t>(P);
    r_weights_ = core::DeviceBuffer<float>(P);
    r_plan_ = core::DeviceBuffer<int32_t>(qwen4exp::kMoePlanInts);
    r_a0_gu_ = core::DeviceBuffer<uint16_t>(P * kHidden);
    r_a1_gu_ = core::DeviceBuffer<uint16_t>(P * kHidden);
    r_gu_ = core::DeviceBuffer<uint16_t>(P * 2 * kInter);
    r_ws_gu_ = core::DeviceBuffer<float>(G * qwen4exp::kMoeRowsMaxSkg * 16 * 2 * kInter);
    r_tickets_gu_ = core::DeviceBuffer<uint32_t>(G * (2 * kInter / 128));
    r_a0_down_ = core::DeviceBuffer<uint16_t>(P * kInter);
    r_yp_ = core::DeviceBuffer<uint16_t>(P * kHidden);
    r_ws_down_ = core::DeviceBuffer<float>(G * qwen4exp::kMoeRowsMaxSkg * 16 * kHidden);
    r_tickets_down_ = core::DeviceBuffer<uint32_t>(G * (kHidden / 128));
    r_shared_g_ = core::DeviceBuffer<uint16_t>(R * kInter);
    r_shared_u_ = core::DeviceBuffer<uint16_t>(R * kInter);
    r_shared_h_ = core::DeviceBuffer<uint16_t>(R * kInter);
    r_shared_y_ = core::DeviceBuffer<uint16_t>(R * kHidden);
    r_shared_gate_logit_ = core::DeviceBuffer<uint16_t>(R);
    r_plan_.Zero();
    r_tickets_gu_.Zero();
    r_tickets_down_.Zero();
  }
  R4DX_HIP_CHECK(hipStreamSynchronize(nullptr));
}

void Qwen35GroupedMoeDecode::SetRowSplits(int gate_up_skg, int down_skg) {
  const auto ok = [](int v) { return v == 1 || v == 2 || v == 4; };
  if (!ok(gate_up_skg) || !ok(down_skg)) {
    throw std::invalid_argument("Qwen35GroupedMoeDecode::SetRowSplits: 1, 2 or 4");
  }
  rows_gu_skg_ = gate_up_skg;
  rows_down_skg_ = down_skg;
}

std::vector<int32_t> Qwen35GroupedMoeDecode::ReadPlan(hipStream_t stream) const {
  std::vector<int32_t> plan(static_cast<size_t>(qwen4exp::kMoePlanInts));
  R4DX_HIP_CHECK(hipMemcpyAsync(plan.data(), r_plan_.data(), plan.size() * 4, hipMemcpyDeviceToHost, stream));
  R4DX_HIP_CHECK(hipStreamSynchronize(stream));
  return plan;
}

void Qwen35GroupedMoeDecode::RunRows(hipStream_t s, core::Arena& arena, const Qwen35GroupedMoeWeights& w,
                                     const Qwen35MoeTrellisPtrs& p, const uint16_t* x, uint16_t* y, int64_t rows,
                                     const std::function<void(const char*)>* mark) {
  const auto phase = [&](const char* name) {
    if (mark != nullptr) (*mark)(name);
  };
  if (rows == 1) {
    Run(s, arena, w, p, x, y, mark);
    return;
  }
  if (rows < 1 || rows > max_rows_) throw std::invalid_argument("Qwen35GroupedMoeDecode::RunRows: rows out of range");
  const int R = static_cast<int>(rows), P = R * top_k_;
  const int max_groups = std::min(experts_, P);
  phase("start");
  // 1. router GEMM + route (one wave per row)
  qwen4exp::ApplyBf16Linear(s, arena, *w.router, x, r_logits_.data(), rows);
  Check(q4x_moe_route_bf16(r_logits_.data(), r_ids_.data(), r_weights_.data(), nullptr, rows, experts_, top_k_, norm_topk_ ? 1 : 0, s),
        "route");
  phase("router+route");
  // 2. shared expert, all rows
  qwen4exp::ApplyBf16Linear(s, arena, *w.shared_gate_proj, x, r_shared_g_.data(), rows);
  qwen4exp::ApplyBf16Linear(s, arena, *w.shared_up_proj, x, r_shared_u_.data(), rows);
  Check(q4x_moe_silu_mul_bf16(r_shared_g_.data(), r_shared_u_.data(), r_shared_h_.data(), rows, kInter, kInter, kInter, s), "shared silu_mul");
  qwen4exp::ApplyBf16Linear(s, arena, *w.shared_down_proj, r_shared_h_.data(), r_shared_y_.data(), rows);
  qwen4exp::ApplyBf16Linear(s, arena, *w.shared_gate, x, r_shared_gate_logit_.data(), rows);
  phase("shared expert");
  // 3. the routed experts, grouped by expert
  Check(qwen4exp::GroupedMoePlanRows(r_ids_.data(), r_plan_.data(), R, top_k_, s), "plan");
  phase("plan");
  Check(qwen4exp::GroupedTrellisInputGateUpQ35Rows(x, r_plan_.data(), p.gate_up_suh, r_a0_gu_.data(), r_a1_gu_.data(), P, s),
        "gate_up input transform");
  phase("gate_up transform");
  Check(qwen4exp::GroupedTrellisGemvGateUpQ35Rows(p.gate_up_bits, r_a0_gu_.data(), r_a1_gu_.data(), r_plan_.data(), p.gate_up_w,
                                                  p.gate_up_svh, r_gu_.data(), r_ws_gu_.data(), r_tickets_gu_.data(), max_groups,
                                                  rows_gu_skg_, s),
        "gate_up GEMM");
  phase("gate_up GEMM");
  Check(qwen4exp::GroupedSiluMulTrellisInputDownQ35Rows(r_gu_.data(), r_plan_.data(), p.down_suh, r_a0_down_.data(), P, s),
        "silu_mul / down input transform");
  phase("silu+down transform");
  Check(qwen4exp::GroupedTrellisGemvDownQ35Rows(p.down_bits, r_a0_down_.data(), r_plan_.data(), p.down_w, p.down_svh, r_yp_.data(),
                                                r_ws_down_.data(), r_tickets_down_.data(), max_groups, rows_down_skg_, s),
        "down GEMM");
  phase("down GEMM");
  // 4. combine
  Check(qwen4exp::GroupedMoeCombineRowsRouterOrder(r_yp_.data(), r_weights_.data(), r_plan_.data(), r_shared_y_.data(),
                                                   r_shared_gate_logit_.data(), y, R, top_k_, kHidden, s),
        "combine");
  phase("combine");
}

void Qwen35GroupedMoeDecode::RunShared(hipStream_t s, core::Arena& arena, const Qwen35GroupedMoeWeights& w, const uint16_t* x) {
  qwen4exp::ApplyBf16Linear(s, arena, *w.shared_gate_proj, x, shared_g_.data(), 1);
  qwen4exp::ApplyBf16Linear(s, arena, *w.shared_up_proj, x, shared_u_.data(), 1);
  Check(q4x_moe_silu_mul_bf16(shared_g_.data(), shared_u_.data(), shared_h_.data(), 1, kInter, kInter, kInter, s), "shared silu_mul");
  qwen4exp::ApplyBf16Linear(s, arena, *w.shared_down_proj, shared_h_.data(), shared_y_.data(), 1);
  qwen4exp::ApplyBf16Linear(s, arena, *w.shared_gate, x, shared_gate_logit_.data(), 1);
}

void Qwen35GroupedMoeDecode::Run(hipStream_t s, core::Arena& arena, const Qwen35GroupedMoeWeights& w,
                                 const Qwen35MoeTrellisPtrs& p, const uint16_t* x, uint16_t* y,
                                 const std::function<void(const char*)>* mark) {
  const auto phase = [&](const char* name) {
    if (mark != nullptr) (*mark)(name);
  };
  phase("start");
  // 1. router GEMM + route (softmax, top-k, norm)
  qwen4exp::ApplyBf16Linear(s, arena, *w.router, x, logits_.data(), 1);
  Check(q4x_moe_route_bf16(logits_.data(), ids_.data(), weights_.data(), nullptr, 1, experts_, top_k_, norm_topk_ ? 1 : 0, s),
        "route");
  phase("router+route");

  // 2. shared expert (bf16 GEMVs; independent of the routed experts, queued behind the route). A side stream for it was measured
  // 1.5x SLOWER (0.20 ms vs 0.129 per layer; the cross-stream waits cost more than the overlap saves).
  RunShared(s, arena, w, x);
  phase("shared expert");

  // 3. routed experts
  Check(qwen4exp::GroupedMoeDecodeSortRanked(ids_.data(), ids_sorted_.data(), rank_.data(), top_k_, s), "sort");
  phase("sort");
  Check(qwen4exp::GroupedTrellisInputGateUpQ35(x, ids_sorted_.data(), p.gate_up_suh, a0_gu_.data(), a1_gu_.data(), top_k_, s),
        "gate_up input transform");
  phase("gate_up transform");
  Check(qwen4exp::GroupedTrellisGemvGateUpQ35(p.gate_up_bits, a0_gu_.data(), a1_gu_.data(), ids_sorted_.data(), p.gate_up_w,
                                              p.gate_up_svh, gu_.data(), ws_gu_.data(), tickets_gu_.data(), top_k_, s),
        "gate_up GEMV");
  phase("gate_up GEMV");
  Check(qwen4exp::GroupedSiluMulTrellisInputDownQ35(gu_.data(), ids_sorted_.data(), p.down_suh, a0_down_.data(), top_k_, s),
        "silu_mul / down input transform");
  phase("silu+down transform");
  Check(qwen4exp::GroupedTrellisGemvDownQ35(p.down_bits, a0_down_.data(), ids_sorted_.data(), p.down_w, p.down_svh,
                                            yp_.data(), ws_down_.data(), tickets_down_.data(), top_k_, s),
        "down GEMV");
  phase("down GEMV");

  // 4. combine
  Check(qwen4exp::GroupedMoeDecodeCombineRouterOrder(yp_.data(), weights_.data(), rank_.data(), shared_y_.data(),
                                                     shared_gate_logit_.data(), y, top_k_, kHidden, s),
        "combine");
  phase("combine");
}

void Qwen35GroupedMoeDecode::ReadSelection(hipStream_t stream, int32_t* ids, float* weights) const {
  R4DX_HIP_CHECK(hipMemcpyAsync(ids, ids_.data(), static_cast<size_t>(top_k_) * 4, hipMemcpyDeviceToHost, stream));
  R4DX_HIP_CHECK(hipMemcpyAsync(weights, weights_.data(), static_cast<size_t>(top_k_) * 4, hipMemcpyDeviceToHost, stream));
  R4DX_HIP_CHECK(hipStreamSynchronize(stream));
}

}  // namespace r4dx::model

