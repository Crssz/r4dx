// qwen4_exp MoE block end to end (MoeBlockBase: router GEMM, route, host plan, gather, per-expert gate_up / down GEMMs
// through the callback, silu*up, shared expert, weighted combine) against HF Qwen4ExpTextSparseMoeBlock (16 experts,
// top-10, the real 2560 / 640 dims) from tools/reference/qwen4exp/small_fixtures.py. The callback runs bf16 dense
// per-expert weights through ApplyBf16Linear. Two row counts: 70 (> one 64-row GEMM chunk) and 1 (decode).
// Exit 77 when the fixtures are missing.
#include <cstdio>
#include <exception>
#include <vector>

#include "r4dx/core/arena.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/moe_block.h"
#include "small_fixture.h"

using namespace q4test;
using namespace r4dx;
using namespace r4dx::model::qwen4exp;
using r4dx::model::Layout;
using r4dx::model::QuantLinear;

namespace {

bool g_ok = true;

double Rms(const uint16_t* p, int64_t n) {
  double s = 0;
  for (int64_t i = 0; i < n; ++i) s += double(F(p[i])) * F(p[i]);
  return std::sqrt(s / double(n));
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
    MoeConfig cfg;
    cfg.experts = static_cast<int>(fx.Param("moe.E"));
    cfg.top_k = static_cast<int>(fx.Param("moe.k"));
    const int E = cfg.experts, H = cfg.hidden, I = cfg.inter;

    // weights
    QuantLinear router = MakeBf16Linear(fx.Get("moe.router_w").bf16(), E, H);
    QuantLinear sgate = MakeBf16Linear(fx.Get("moe.shared_gate_proj_w").bf16(), cfg.shared_inter, H);
    QuantLinear sup = MakeBf16Linear(fx.Get("moe.shared_up_proj_w").bf16(), cfg.shared_inter, H);
    QuantLinear sdown = MakeBf16Linear(fx.Get("moe.shared_down_proj_w").bf16(), H, cfg.shared_inter);
    QuantLinear sgl = MakeBf16Linear(fx.Get("moe.shared_gate_w").bf16(), 1, H);
    auto gate_up_all = Upload(fx.Get("moe.gate_up_w"));  // [E, 2I, H]
    auto down_all = Upload(fx.Get("moe.down_w"));        // [E, H, I]
    std::vector<QuantLinear> gate_up(E), down(E);
    for (int e = 0; e < E; ++e) {
      gate_up[e].layout = Layout::kBf16;
      gate_up[e].N = 2 * I;
      gate_up[e].K = H;
      gate_up[e].bf16_w = core::DeviceBuffer<uint16_t>::View(gate_up_all.data() + int64_t(e) * 2 * I * H, size_t(2 * I) * H);
      down[e].layout = Layout::kBf16;
      down[e].N = H;
      down[e].K = I;
      down[e].bf16_w = core::DeviceBuffer<uint16_t>::View(down_all.data() + int64_t(e) * H * I, size_t(H) * I);
    }
    MoeWeights w;
    w.router = &router;
    w.shared_gate_proj = &sgate;
    w.shared_up_proj = &sup;
    w.shared_down_proj = &sdown;
    w.shared_gate = &sgl;

    core::Stream stream;
    MoeBlockBase block(cfg, 70);
    core::Arena arena(block.ArenaBytes(70));

    int64_t expert_calls = 0;
    ExpertLinearFn run = [&](int e, ExpertLinear which, const uint16_t* x, int64_t rows, uint16_t* y,
                             core::Arena& a) {
      ++expert_calls;
      ApplyBf16Linear(stream.get(), a, which == ExpertLinear::kGateUp ? gate_up[e] : down[e], x, y, rows);
    };

    for (const char* tag : {"t70", "t1"}) {
      const std::string p = std::string("moe.") + tag + ".";
      const int rows = static_cast<int>(fx.Param(p + "rows"));
      std::printf("[%s] rows=%d E=%d k=%d\n", tag, rows, E, cfg.top_k);
      auto x = Upload(fx.Get(p + "x"));
      core::DeviceBuffer<uint16_t> y(static_cast<size_t>(rows) * H);
      arena.Reset();
      expert_calls = 0;
      block.Forward(stream.get(), arena, w, x.data(), y.data(), rows, run);
      Sync();

      // router ids vs HF's (exact: both ordered by score, ties to the lower id; fixture rows without ties)
      const Tensor& ids_ref = fx.Get(p + "ids");
      int id_rows_bad = 0;
      for (int r = 0; r < rows; ++r) {
        for (int j = 0; j < cfg.top_k; ++j) {
          if (block.last_ids()[r * cfg.top_k + j] != ids_ref.i32()[r * cfg.top_k + j]) {
            ++id_rows_bad;
            break;
          }
        }
      }
      int active = 0;
      for (int e = 0; e < E; ++e) active += block.last_plan().Count(e) > 0;
      std::printf("  router ids: rows differing from HF = %d/%d; experts with rows = %d/%d; callback calls = %lld\n",
                  id_rows_bad, rows, active, E, static_cast<long long>(expert_calls));
      if (id_rows_bad != 0) g_ok = false;
      if (expert_calls != 2 * active) {
        std::printf("  FAIL: callback calls != 2 x active experts\n");
        g_ok = false;
      }

      const Tensor& ref = fx.Get(p + "y");
      const std::vector<uint16_t> got = y.CopyToHost();
      const double rms = Rms(ref.bf16(), ref.count());
      const ErrorStats s = CompareBf16(got.data(), ref.bf16(), ref.count(), 0.03 * rms, 0.03);
      if (!Report("block output vs HF", s)) g_ok = false;

      // Against the fp32 module (same weights): HF's bf16 expert-by-expert accumulation rounds at every add, ours
      // accumulates in fp32 and rounds once, so ours must be at least as close to the fp32 truth as HF's own bf16 run.
      const Tensor& truth = fx.Get(p + "y_f32");
      double e_ours = 0, e_hf = 0;
      for (int64_t i = 0; i < ref.count(); ++i) {
        e_ours += std::fabs(double(F(got[i])) - truth.f32()[i]);
        e_hf += std::fabs(double(F(ref.bf16()[i])) - truth.f32()[i]);
      }
      e_ours /= double(ref.count());
      e_hf /= double(ref.count());
      const bool closer = e_ours <= e_hf * 1.02;
      std::printf("  mean |err| vs fp32 module: ours %.4g, HF bf16 %.4g  %s\n", e_ours, e_hf, closer ? "OK" : "FAIL");
      if (!closer) g_ok = false;
    }
    std::printf(g_ok ? "test_gpu_moe_block: PASS\n" : "test_gpu_moe_block: FAIL\n");
    return g_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gpu_moe_block: exception: %s\n", e.what());
    return 1;
  }
}
