#include "qwen35moe_prefill_moe.h"

#include <hip/hip_runtime.h>

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "r4dx/core/error.hpp"
#include "r4dx/model/qwen4exp/common.h"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/moe_kernels.h"
#include "r4dx/model/qwen4exp/moe_trellis_grouped.h"

namespace r4dx::model {

namespace {

constexpr const char* kCtx = "Qwen35PrefillMoe";

void Check(int code, const char* what) { qwen4exp::Check(kCtx, code, what); }

constexpr int kHidden = Qwen35GroupedMoeDecode::kHidden;

}  // namespace

Qwen35PrefillMoe::Qwen35PrefillMoe(int experts, int top_k, bool norm_topk, int max_rows, int inter, int gu_skg, int down_skg)
    : experts_(experts), top_k_(top_k), max_rows_(max_rows), inter_(inter), gu_skg_(gu_skg), down_skg_(down_skg), norm_topk_(norm_topk) {
  const auto skg_ok = [](int v) { return v == 1 || v == 2 || v == 4; };
  if (inter != Qwen35GroupedMoeDecode::kInter && inter != Qwen35GroupedMoeDecode::kTp2Inter) {
    throw std::invalid_argument("Qwen35PrefillMoe: inter must be 1024 (TP=1) or 512 (a --tp 2 rank)");
  }
  if (!skg_ok(gu_skg) || !skg_ok(down_skg)) throw std::invalid_argument("Qwen35PrefillMoe: split-group counts must be 1, 2 or 4");
  if (max_rows < 2 || top_k < 1 || top_k > 16 || experts < top_k || experts > 1024 || max_rows * top_k > qwen4exp::kMoePlanMaxPairs) {
    throw std::invalid_argument("Qwen35PrefillMoe: max_rows * top_k must be <= 8192 (and rows >= 2, top_k <= 16, experts <= 1024)");
  }
  const size_t R = static_cast<size_t>(max_rows), P = R * static_cast<size_t>(top_k), E = static_cast<size_t>(experts);
  const size_t G = std::min(P, P / 16 + E);  // the most groups a chunk can produce
  const size_t I = static_cast<size_t>(inter), Ngu = 2 * I;
  logits_ = core::DeviceBuffer<uint16_t>(R * E);
  ids_ = core::DeviceBuffer<int32_t>(P);
  weights_ = core::DeviceBuffer<float>(P);
  plan_ = core::DeviceBuffer<int32_t>(qwen4exp::kMoePlanInts);
  counts_ = core::DeviceBuffer<int32_t>(E);
  a0_gu_ = core::DeviceBuffer<uint16_t>(P * kHidden);
  a1_gu_ = core::DeviceBuffer<uint16_t>(P * kHidden);
  gu_ = core::DeviceBuffer<uint16_t>(P * Ngu);
  a0_down_ = core::DeviceBuffer<uint16_t>(P * I);
  yp_ = core::DeviceBuffer<uint16_t>(P * kHidden);
  shared_g_ = core::DeviceBuffer<uint16_t>(R * I);
  shared_u_ = core::DeviceBuffer<uint16_t>(R * I);
  shared_h_ = core::DeviceBuffer<uint16_t>(R * I);
  shared_y_ = core::DeviceBuffer<uint16_t>(R * kHidden);
  shared_gate_logit_ = core::DeviceBuffer<uint16_t>(R);
  if (gu_skg > 1) {
    ws_gu_ = core::DeviceBuffer<float>(G * gu_skg * 16 * Ngu);
    tickets_gu_ = core::DeviceBuffer<uint32_t>(G * (Ngu / 128));
    tickets_gu_.Zero();
  }
  if (down_skg > 1) {
    ws_down_ = core::DeviceBuffer<float>(G * down_skg * 16 * kHidden);
    tickets_down_ = core::DeviceBuffer<uint32_t>(G * (kHidden / 128));
    tickets_down_.Zero();
  }
  plan_.Zero();
  scratch_bytes_ = logits_.bytes() + ids_.bytes() + weights_.bytes() + plan_.bytes() + counts_.bytes() + a0_gu_.bytes() + a1_gu_.bytes() +
                   gu_.bytes() + a0_down_.bytes() + yp_.bytes() + shared_g_.bytes() * 3 + shared_y_.bytes() + shared_gate_logit_.bytes() +
                   ws_gu_.bytes() + ws_down_.bytes() + tickets_gu_.bytes() + tickets_down_.bytes();
  R4DX_HIP_CHECK(hipStreamSynchronize(nullptr));
}

std::vector<int32_t> Qwen35PrefillMoe::ReadPlan(hipStream_t stream) const {
  std::vector<int32_t> plan(static_cast<size_t>(qwen4exp::kMoePlanInts));
  R4DX_HIP_CHECK(hipMemcpyAsync(plan.data(), plan_.data(), plan.size() * 4, hipMemcpyDeviceToHost, stream));
  R4DX_HIP_CHECK(hipStreamSynchronize(stream));
  return plan;
}

void Qwen35PrefillMoe::Run(hipStream_t s, core::Arena& arena, const Qwen35GroupedMoeWeights& w, const Qwen35MoeTrellisPtrs& p,
                           const uint16_t* x, uint16_t* y, int64_t rows, const std::function<void(const char*)>* mark) {
  const auto phase = [&](const char* name) {
    if (mark != nullptr) (*mark)(name);
  };
  if (rows < 2 || rows > max_rows_) throw std::invalid_argument("Qwen35PrefillMoe::Run: rows out of range");
  const bool tp2 = inter_ == Qwen35GroupedMoeDecode::kTp2Inter;
  const int R = static_cast<int>(rows), P = R * top_k_;
  const int max_groups = std::min(P, P / 16 + experts_);
  // 1. router GEMM + route (one wave per row)
  qwen4exp::ApplyBf16Linear(s, arena, *w.router, x, logits_.data(), rows);
  Check(q4x_moe_route_bf16(logits_.data(), ids_.data(), weights_.data(), nullptr, rows, experts_, top_k_, norm_topk_ ? 1 : 0, s), "route");
  phase("moe.router");
  // 2. shared expert, all rows
  qwen4exp::ApplyBf16Linear(s, arena, *w.shared_gate_proj, x, shared_g_.data(), rows);
  qwen4exp::ApplyBf16Linear(s, arena, *w.shared_up_proj, x, shared_u_.data(), rows);
  Check(q4x_moe_silu_mul_bf16(shared_g_.data(), shared_u_.data(), shared_h_.data(), rows, inter_, inter_, inter_, s), "shared silu_mul");
  qwen4exp::ApplyBf16Linear(s, arena, *w.shared_down_proj, shared_h_.data(), shared_y_.data(), rows);
  qwen4exp::ApplyBf16Linear(s, arena, *w.shared_gate, x, shared_gate_logit_.data(), rows);
  phase("moe.shared");
  // 3. the routed experts, grouped by expert
  Check(qwen4exp::GroupedMoePlanLarge(ids_.data(), plan_.data(), counts_.data(), R, top_k_, experts_, s), "plan");
  Check(qwen4exp::GroupedTrellisInputGateUpQ35Rows(x, plan_.data(), p.gate_up_suh, a0_gu_.data(), a1_gu_.data(), P, s),
        "gate_up input transform");
  phase("moe.plan+transform");
  Check((tp2 ? qwen4exp::GroupedTrellisGemvGateUpQ35Tp2Rows : qwen4exp::GroupedTrellisGemvGateUpQ35Rows)(
            p.gate_up_bits, a0_gu_.data(), a1_gu_.data(), plan_.data(), p.gate_up_w, p.gate_up_svh, gu_.data(), ws_gu_.data(),
            tickets_gu_.data(), max_groups, gu_skg_, s),
        "gate_up GEMM");
  phase("moe.gate_up");
  Check((tp2 ? qwen4exp::GroupedSiluMulTrellisInputDownQ35Tp2Rows : qwen4exp::GroupedSiluMulTrellisInputDownQ35Rows)(
            gu_.data(), plan_.data(), p.down_suh, a0_down_.data(), P, s),
        "silu_mul / down input transform");
  phase("moe.silu");
  Check((tp2 ? qwen4exp::GroupedTrellisGemvDownQ35Tp2Rows : qwen4exp::GroupedTrellisGemvDownQ35Rows)(
            p.down_bits, a0_down_.data(), plan_.data(), p.down_w, p.down_svh, yp_.data(), ws_down_.data(), tickets_down_.data(), max_groups,
            down_skg_, s),
        "down GEMM");
  phase("moe.down");
  // 4. combine
  Check(qwen4exp::GroupedMoeCombineRowsRouterOrder(yp_.data(), weights_.data(), plan_.data(), shared_y_.data(), shared_gate_logit_.data(), y,
                                                   R, top_k_, kHidden, s),
        "combine");
  phase("moe.combine");
}

}  // namespace r4dx::model
