#include "r4dx/model/qwen4exp/moe_block.h"

#include <stdexcept>
#include <string>

#include "r4dx/core/error.hpp"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/moe_kernels.h"

namespace r4dx::model::qwen4exp {

namespace {

[[noreturn]] void Fail(const std::string& msg) { throw std::runtime_error("qwen4exp::MoeBlockBase: " + msg); }

void Check(int code, const char* what) {
  if (code != 0) {
    throw std::runtime_error(std::string("qwen4exp::MoeBlockBase: ") + what + " failed: " +
                             hipGetErrorString(static_cast<hipError_t>(code)));
  }
}

void CheckLinear(const QuantLinear* l, const char* name, int64_t n, int64_t k) {
  if (l == nullptr) Fail(std::string("missing weight ") + name);
  if (l->N != n || l->K != k) {
    Fail(std::string(name) + " is [" + std::to_string(l->N) + ", " + std::to_string(l->K) + "], expected [" +
         std::to_string(n) + ", " + std::to_string(k) + "]");
  }
}

constexpr size_t kAlign = 16;
size_t Padded(size_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }

}  // namespace

MoeBlockBase::MoeBlockBase(const MoeConfig& cfg, int64_t max_rows)
    : cfg_(cfg), max_rows_(max_rows), ids_host_(static_cast<size_t>(max_rows) * static_cast<size_t>(cfg.top_k)) {
  if (cfg.hidden < 1 || cfg.experts < 1 || cfg.top_k < 1 || cfg.top_k > cfg.experts || cfg.inter < 1 ||
      cfg.shared_inter < 1 || max_rows < 1) {
    Fail("invalid config / max_rows");
  }
}

size_t MoeBlockBase::ArenaBytes(int64_t rows) const {
  const size_t r = static_cast<size_t>(rows), E = static_cast<size_t>(cfg_.experts),
               k = static_cast<size_t>(cfg_.top_k), H = static_cast<size_t>(cfg_.hidden),
               I = static_cast<size_t>(cfg_.inter), SI = static_cast<size_t>(cfg_.shared_inter);
  const size_t P = r * k;
  return Padded(r * E * 2) + Padded(P * 4) + Padded(P * 4) + Padded(E * 4) +  // logits, ids, weights, counts
         3 * Padded(P * 4) +                                                   // pair_row, row_pairs, row_w_idx
         Padded(P * H * 2) + Padded(P * 2 * I * 2) + Padded(P * I * 2) + Padded(P * H * 2) +  // xg, gu, hh, yp
         3 * Padded(r * SI * 2) + Padded(r * H * 2) + Padded(r * 2);           // shared: gate, up, h, y, gate logit
}

void MoeBlockBase::Forward(hipStream_t stream, core::Arena& arena, const MoeWeights& w, const uint16_t* x,
                           uint16_t* y, int64_t rows, const ExpertLinearFn& run_expert) {
  if (rows <= 0) return;
  if (rows > max_rows_) {
    Fail("rows " + std::to_string(rows) + " exceeds max_rows " + std::to_string(max_rows_));
  }
  if (!run_expert) Fail("run_expert callback is empty");
  const int E = cfg_.experts, k = cfg_.top_k, H = cfg_.hidden, I = cfg_.inter, SI = cfg_.shared_inter;
  CheckLinear(w.router, "router", E, H);
  CheckLinear(w.shared_gate_proj, "shared_gate_proj", SI, H);
  CheckLinear(w.shared_up_proj, "shared_up_proj", SI, H);
  CheckLinear(w.shared_down_proj, "shared_down_proj", H, SI);
  CheckLinear(w.shared_gate, "shared_gate", 1, H);
  const int64_t P = rows * k;

  uint16_t* logits = arena.Alloc<uint16_t>(static_cast<size_t>(rows * E));
  int32_t* ids = arena.Alloc<int32_t>(static_cast<size_t>(P));
  float* weights = arena.Alloc<float>(static_cast<size_t>(P));
  int32_t* counts = arena.Alloc<int32_t>(static_cast<size_t>(E));
  int32_t* d_pair_row = arena.Alloc<int32_t>(static_cast<size_t>(P));
  int32_t* d_row_pairs = arena.Alloc<int32_t>(static_cast<size_t>(P));
  int32_t* d_row_w_idx = arena.Alloc<int32_t>(static_cast<size_t>(P));
  uint16_t* xg = arena.Alloc<uint16_t>(static_cast<size_t>(P * H));
  uint16_t* gu = arena.Alloc<uint16_t>(static_cast<size_t>(P * 2 * I));
  uint16_t* hh = arena.Alloc<uint16_t>(static_cast<size_t>(P * I));
  uint16_t* yp = arena.Alloc<uint16_t>(static_cast<size_t>(P * H));
  uint16_t* sg = arena.Alloc<uint16_t>(static_cast<size_t>(rows * SI));
  uint16_t* su = arena.Alloc<uint16_t>(static_cast<size_t>(rows * SI));
  uint16_t* sh = arena.Alloc<uint16_t>(static_cast<size_t>(rows * SI));
  uint16_t* sy = arena.Alloc<uint16_t>(static_cast<size_t>(rows * H));
  uint16_t* sgl = arena.Alloc<uint16_t>(static_cast<size_t>(rows));

  // Router. The ids go to the host (the one sync of the layer) while the shared expert is already queued behind them.
  ApplyBf16Linear(stream, arena, *w.router, x, logits, rows);
  R4DX_HIP_CHECK(hipMemsetAsync(counts, 0, static_cast<size_t>(E) * sizeof(int32_t), stream));
  Check(q4x_moe_route_bf16(logits, ids, weights, counts, rows, E, k, cfg_.norm_topk ? 1 : 0, stream), "route");
  R4DX_HIP_CHECK(hipMemcpyAsync(ids_host_.data(), ids, static_cast<size_t>(P) * sizeof(int32_t),
                                hipMemcpyDeviceToHost, stream));
  R4DX_HIP_CHECK(hipEventRecord(ids_ready_.get(), stream));

  ApplyBf16Linear(stream, arena, *w.shared_gate_proj, x, sg, rows);
  ApplyBf16Linear(stream, arena, *w.shared_up_proj, x, su, rows);
  Check(q4x_moe_silu_mul_bf16(sg, su, sh, rows, SI, SI, SI, stream), "shared silu_mul");
  ApplyBf16Linear(stream, arena, *w.shared_down_proj, sh, sy, rows);
  ApplyBf16Linear(stream, arena, *w.shared_gate, x, sgl, rows);

  R4DX_HIP_CHECK(hipEventSynchronize(ids_ready_.get()));
  // plan_'s vectors are rewritten only here, after this wait: every earlier upload from them has long completed.
  plan_ = BuildMoePlan(ids_host_.data(), static_cast<int>(rows), k, E);
  R4DX_HIP_CHECK(hipMemcpyAsync(d_pair_row, plan_.pair_row.data(), static_cast<size_t>(P) * sizeof(int32_t),
                                hipMemcpyHostToDevice, stream));
  R4DX_HIP_CHECK(hipMemcpyAsync(d_row_pairs, plan_.row_pairs.data(), static_cast<size_t>(P) * sizeof(int32_t),
                                hipMemcpyHostToDevice, stream));
  R4DX_HIP_CHECK(hipMemcpyAsync(d_row_w_idx, plan_.row_w_idx.data(), static_cast<size_t>(P) * sizeof(int32_t),
                                hipMemcpyHostToDevice, stream));

  Check(q4x_moe_gather_rows_bf16(x, d_pair_row, xg, P, H, stream), "gather rows");
  for (int e = 0; e < E; ++e) {
    const int64_t n = plan_.Count(e);
    if (n == 0) continue;
    const int64_t off = plan_.expert_offset[static_cast<size_t>(e)];
    run_expert(e, ExpertLinear::kGateUp, xg + off * H, n, gu + off * 2 * I, arena);
  }
  Check(q4x_moe_silu_mul_bf16(gu, gu + I, hh, P, I, 2 * I, 2 * I, stream), "expert silu_mul");
  for (int e = 0; e < E; ++e) {
    const int64_t n = plan_.Count(e);
    if (n == 0) continue;
    const int64_t off = plan_.expert_offset[static_cast<size_t>(e)];
    run_expert(e, ExpertLinear::kDown, hh + off * I, n, yp + off * H, arena);
  }
  Check(q4x_moe_combine_bf16(yp, d_row_pairs, d_row_w_idx, weights, sy, sgl, y, rows, k, H, stream), "combine");
}

}  // namespace r4dx::model::qwen4exp
