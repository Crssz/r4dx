// qwen4_exp router kernel (q4x_moe_route_bf16) and the host plan (BuildMoePlan) against the HF router math of
// tools/reference/qwen4exp/small_fixtures.py: random logits (E = 16 and the real E = 512), crafted ties at the top and at
// the top-k boundary, and heavy ties (logits on a 0.25 grid, E = 512). Ids must match exactly; the tie rule is "larger
// logit first, equal logits to the lower expert id" (moe_kernels.h). Exit 77 when the fixtures are missing.
#include <algorithm>
#include <cstdio>
#include <exception>
#include <numeric>
#include <vector>

#include "r4dx/core/stream.hpp"
#include "r4dx/model/qwen4exp/moe_kernels.h"
#include "r4dx/model/qwen4exp/moe_plan.h"
#include "small_fixture.h"

using namespace q4test;
using namespace r4dx;
using namespace r4dx::model::qwen4exp;

namespace {

bool g_ok = true;
void Fail(const std::string& msg) {
  std::printf("  FAIL: %s\n", msg.c_str());
  g_ok = false;
}

// Host reference of the routing weights for norm_topk = false: fp32 softmax of the bf16 logits at the selected ids,
// rounded to bf16 (an independent computation in double).
float HostProb(const uint16_t* logits, int E, int id) {
  double mx = -1e300;
  for (int i = 0; i < E; ++i) mx = std::max<double>(mx, F(logits[i]));
  double s = 0;
  for (int i = 0; i < E; ++i) s += std::exp(F(logits[i]) - mx);
  return F(r4dx::core::FloatToBf16(static_cast<float>(std::exp(F(logits[id]) - mx) / s)));
}

void CheckPlan(const MoePlan& plan, const int32_t* ids, int rows, int k, int E) {
  if (plan.pairs != int64_t(rows) * k || plan.expert_offset.back() != plan.pairs) Fail("plan: pair count");
  for (int e = 0; e < E; ++e) {
    for (int p = plan.expert_offset[e]; p < plan.expert_offset[e + 1]; ++p) {
      if (p > plan.expert_offset[e] && plan.pair_row[p] <= plan.pair_row[p - 1]) Fail("plan: rows not ascending in expert");
    }
  }
  for (int r = 0; r < rows; ++r) {
    int prev_e = -1;
    for (int j = 0; j < k; ++j) {
      const int p = plan.row_pairs[r * k + j];
      const int wi = plan.row_w_idx[r * k + j];
      const int e = ids[wi];
      if (wi / k != r) Fail("plan: row_w_idx row");
      if (!(p >= plan.expert_offset[e] && p < plan.expert_offset[e + 1])) Fail("plan: pair not in its expert slice");
      if (plan.pair_row[p] != r) Fail("plan: pair_row");
      if (e <= prev_e) Fail("plan: experts not ascending within a row");
      prev_e = e;
    }
  }
}

void Case(Fixtures& fx, const std::string& name) {
  const std::string p = "route." + name + ".";
  const int rows = static_cast<int>(fx.Param(p + "rows")), E = static_cast<int>(fx.Param(p + "E")), k = 10;
  std::printf("[%s] rows=%d E=%d (torch.topk disagrees with the stable rule on %d rows)\n", name.c_str(), rows, E,
              static_cast<int>(fx.Param(p + "torch_topk_ties_differ")));
  const Tensor& logits = fx.Get(p + "logits");
  const Tensor& ids_ref = fx.Get(p + "ids");
  const Tensor& w_ref = fx.Get(p + "weights");
  const Tensor& counts_ref = fx.Get(p + "counts");
  core::Stream stream;
  auto d_logits = Upload(logits);
  core::DeviceBuffer<int32_t> d_ids(static_cast<size_t>(rows * k)), d_counts(static_cast<size_t>(E));
  core::DeviceBuffer<float> d_w(static_cast<size_t>(rows * k));

  for (int norm = 1; norm >= 0; --norm) {
    d_counts.ZeroAsync(stream);
    if (q4x_moe_route_bf16(d_logits.data(), d_ids.data(), d_w.data(), d_counts.data(), rows, E, k, norm,
                           stream.get()) != 0) {
      throw std::runtime_error("q4x_moe_route_bf16 failed");
    }
    Sync();
    const std::vector<int32_t> ids = d_ids.CopyToHost();
    const std::vector<float> w = d_w.CopyToHost();
    const std::vector<int32_t> counts = d_counts.CopyToHost();

    int id_rows_bad = 0;
    for (int r = 0; r < rows; ++r) {
      if (!std::equal(ids.begin() + r * k, ids.begin() + (r + 1) * k, ids_ref.i32() + r * k)) {
        if (++id_rows_bad <= 3) {
          std::printf("  row %d ids differ: got", r);
          for (int j = 0; j < k; ++j) std::printf(" %d", ids[r * k + j]);
          std::printf(" ref");
          for (int j = 0; j < k; ++j) std::printf(" %d", ids_ref.i32()[r * k + j]);
          std::printf("\n");
        }
      }
    }
    // weights: bf16 values stored as fp32. norm: HF fp32 chain (one bf16 ulp of slack for exp / sum order); no norm:
    // the double reference above.
    double max_rel = 0;
    int exact = 0, bad = 0;
    for (int i = 0; i < rows * k; ++i) {
      const float ref = norm ? w_ref.f32()[i] : HostProb(logits.bf16() + (i / k) * E, E, ids[i]);
      if (w[i] == ref) ++exact;
      const double rel = ref != 0 ? std::fabs(w[i] - ref) / std::fabs(ref) : std::fabs(w[i]);
      max_rel = std::max(max_rel, rel);
      if (rel > 0.0079) ++bad;
    }
    int counts_bad = 0;
    for (int e = 0; e < E; ++e) counts_bad += counts[e] != counts_ref.i32()[e];
    if (counts_bad != 0 && E <= 16) {
      for (int e = 0; e < E; ++e) std::printf("   expert %d: device %d ref %d\n", e, counts[e], counts_ref.i32()[e]);
    }
    const bool pass = id_rows_bad == 0 && bad == 0 && counts_bad == 0;
    std::printf("  norm_topk=%d: id rows wrong=%d/%d  weights exact=%d/%d max_rel=%.3g (bad %d)  histogram wrong=%d/%d  %s\n",
                norm, id_rows_bad, rows, exact, rows * k, max_rel, bad, counts_bad, E, pass ? "OK" : "FAIL");
    if (!pass) g_ok = false;
    if (norm) {
      const MoePlan plan = BuildMoePlan(ids.data(), rows, k, E);
      CheckPlan(plan, ids.data(), rows, k, E);
      for (int e = 0; e < E; ++e) {
        if (plan.Count(e) != counts[e]) Fail("plan count != device histogram");
      }
    }
  }
}

void PlanErrors() {
  std::printf("[plan errors]\n");
  const int32_t dup[4] = {1, 2, 1, 3};
  const int32_t oob[4] = {1, 2, 3, 4};
  bool dup_threw = false, oob_threw = false;
  try { BuildMoePlan(dup, 1, 4, 8); } catch (const std::runtime_error&) { dup_threw = true; }
  try { BuildMoePlan(oob, 1, 4, 4); } catch (const std::runtime_error&) { oob_threw = true; }
  std::printf("  duplicate expert in a row throws: %s; id out of range throws: %s\n", dup_threw ? "yes" : "NO",
              oob_threw ? "yes" : "NO");
  if (!dup_threw || !oob_threw) g_ok = false;
}

}  // namespace

int main() {
  try {
    Fixtures fx;
    if (!fx.Open()) {
      std::fprintf(stderr, "[SKIP] fixtures not found in %s (run tools/reference/qwen4exp/small_fixtures.py)\n",
                   Fixtures::Dir().c_str());
      return kSkip;
    }
    for (const char* c : {"rand16", "rand512", "ties16", "ties512"}) Case(fx, c);
    PlanErrors();
    std::printf(g_ok ? "test_gpu_moe_route: PASS\n" : "test_gpu_moe_route: FAIL\n");
    return g_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gpu_moe_route: exception: %s\n", e.what());
    return 1;
  }
}
