// qwen4_exp's grouped decode MoE launchers (hidden 2560, expert width 640, top-10, bits 3 / 4) on SYNTHETIC trellis weights against
// the host-routed MoE (MoeBlockBase + per-expert ApplyLinear). Needs no fixture: it guards the shared kernels in moe_trellis_decode.hip
// (templated for qwen3_5_moe in 2026-10) when test_model_q4exp_moe_grouped skips for want of the swift15 container. The shared expert
// is zero (shared_y = nullptr on the grouped side), so only the routed experts are compared.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "linear.h"
#include "quant_linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/moe_block.h"
#include "r4dx/model/qwen4exp/moe_kernels.h"
#include "r4dx/model/qwen4exp/moe_trellis_grouped.h"

using namespace r4dx;
using namespace r4dx::model;

namespace {

constexpr int kE = 64, kTopK = 10, kH = 2560, kI = 640, kSkip = 77;
int g_failures = 0;

float Bf(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
uint16_t ToBf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

template <class T>
core::DeviceBuffer<T> Upload(const std::vector<T>& h) {
  core::DeviceBuffer<T> b(h.size());
  b.CopyFromHost(h);
  return b;
}

QuantLinear Bf16Linear(int64_t N, int64_t K, std::mt19937& rng, float sd) {
  std::normal_distribution<float> nd(0.0f, sd);
  std::vector<uint16_t> h(static_cast<size_t>(N * K));
  for (auto& v : h) v = ToBf(nd(rng));
  QuantLinear q;
  q.layout = Layout::kBf16;
  q.N = N;
  q.K = K;
  q.bf16_w = Upload(h);
  return q;
}

int RunBits(int gu_bits, int dn_bits) {
  std::mt19937 rng(7 + gu_bits * 10 + dn_bits);
  const size_t gu_per = static_cast<size_t>(2 * kI) * kH * gu_bits / 32, dn_per = static_cast<size_t>(kH) * kI * dn_bits / 32;
  std::vector<uint32_t> gu_w(kE * gu_per), dn_w(kE * dn_per);
  for (auto& w : gu_w) w = static_cast<uint32_t>(rng());
  for (auto& w : dn_w) w = static_cast<uint32_t>(rng());
  std::uniform_int_distribution<int> coin(0, 1);
  std::uniform_real_distribution<float> sc(0.02f, 0.06f);
  std::vector<float> gu_suh(static_cast<size_t>(kE) * 2 * kH), gu_svh(static_cast<size_t>(kE) * 2 * kI), dn_suh(static_cast<size_t>(kE) * kI),
      dn_svh(static_cast<size_t>(kE) * kH);
  for (auto& v : gu_suh) v = coin(rng) ? 1.0f : -1.0f;
  for (auto& v : dn_suh) v = coin(rng) ? 1.0f : -1.0f;
  for (auto& v : gu_svh) v = sc(rng);
  for (auto& v : dn_svh) v = sc(rng);
  auto d_guw = Upload(gu_w), d_dnw = Upload(dn_w);
  auto d_gusuh = Upload(gu_suh), d_gusvh = Upload(gu_svh), d_dnsuh = Upload(dn_suh), d_dnsvh = Upload(dn_svh);

  // host-routed reference: per-expert views
  const size_t tk_gu = core::r4d::GemmTrellisTicketsBytes(2 * kI) / 4, tk_dn = core::r4d::GemmTrellisTicketsBytes(kH) / 4;
  core::DeviceBuffer<uint32_t> tickets(kE * (tk_gu + tk_dn));
  tickets.Zero();
  std::vector<QuantLinear> gu, dn;
  for (int e = 0; e < kE; ++e) {
    QuantLinear g;
    g.layout = Layout::kTrellis;
    g.N = 2 * kI;
    g.K = kH;
    g.trellis_bits = gu_bits;
    g.trellis_parts = 2;
    g.trellis_part_n[0] = kI;
    g.trellis_part_n[1] = kI;
    g.trellis_w = core::DeviceBuffer<uint32_t>::View(d_guw.data() + e * gu_per, gu_per);
    g.trellis_suh = core::DeviceBuffer<float>::View(d_gusuh.data() + static_cast<size_t>(e) * 2 * kH, 2 * kH);
    g.trellis_svh = core::DeviceBuffer<float>::View(d_gusvh.data() + static_cast<size_t>(e) * 2 * kI, 2 * kI);
    g.trellis_tickets = tickets.data() + static_cast<size_t>(e) * (tk_gu + tk_dn);
    gu.push_back(std::move(g));
    QuantLinear d;
    d.layout = Layout::kTrellis;
    d.N = kH;
    d.K = kI;
    d.trellis_bits = dn_bits;
    d.trellis_parts = 1;
    d.trellis_part_n[0] = kH;
    d.trellis_w = core::DeviceBuffer<uint32_t>::View(d_dnw.data() + e * dn_per, dn_per);
    d.trellis_suh = core::DeviceBuffer<float>::View(d_dnsuh.data() + static_cast<size_t>(e) * kI, kI);
    d.trellis_svh = core::DeviceBuffer<float>::View(d_dnsvh.data() + static_cast<size_t>(e) * kH, kH);
    d.trellis_tickets = tickets.data() + static_cast<size_t>(e) * (tk_gu + tk_dn) + tk_gu;
    dn.push_back(std::move(d));
  }
  QuantLinear router = Bf16Linear(kE, kH, rng, 0.05f);
  QuantLinear sg = Bf16Linear(kI, kH, rng, 0.0f), su = Bf16Linear(kI, kH, rng, 0.0f), sd = Bf16Linear(kH, kI, rng, 0.0f),
              sgate = Bf16Linear(1, kH, rng, 0.05f);  // zero shared expert

  core::Stream stream;
  const hipStream_t s = stream.get();
  core::Arena arena, expert_arena, block_arena;
  arena.Reserve(size_t{32} << 20);
  expert_arena.Reserve(size_t{96} << 20);
  block_arena.Reserve(size_t{128} << 20);
  qwen4exp::MoeConfig mc;
  mc.hidden = kH;
  mc.experts = kE;
  mc.top_k = kTopK;
  mc.inter = kI;
  mc.shared_inter = kI;
  mc.norm_topk = true;
  qwen4exp::MoeBlockBase host(mc, 1);
  qwen4exp::MoeWeights mw;
  mw.router = &router;
  mw.shared_gate_proj = &sg;
  mw.shared_up_proj = &su;
  mw.shared_down_proj = &sd;
  mw.shared_gate = &sgate;
  const qwen4exp::ExpertLinearFn fn = [&](int e, qwen4exp::ExpertLinear which, const uint16_t* xin, int64_t rows, uint16_t* y, core::Arena&) {
    expert_arena.Reset();
    ApplyLinear(s, expert_arena, which == qwen4exp::ExpertLinear::kGateUp ? gu[static_cast<size_t>(e)] : dn[static_cast<size_t>(e)], xin, y, rows);
  };

  // grouped buffers, laid out as Qwen4ExpStage::RunMoeGroupedDecode
  core::DeviceBuffer<uint16_t> logits(kE), a0(kTopK * kH), a1(kTopK * kH), guo(kTopK * 2 * kI), a0d(kTopK * kI), yp(kTopK * kH);
  core::DeviceBuffer<int32_t> ids(kTopK), ids_sorted(kTopK);
  core::DeviceBuffer<float> wts(kTopK), wts_sorted(kTopK), ws_gu(kTopK * 5120), ws_dn(kTopK * 5120);
  core::DeviceBuffer<uint32_t> t_gu(kE * 10), t_dn(kE * 20);
  t_gu.Zero();
  t_dn.Zero();

  std::normal_distribution<float> nd(0.0f, 1.0f);
  double worst_max = 0, worst_mean = 0;
  core::DeviceBuffer<uint16_t> x(kH), y_g(kH), y_h(kH);
  std::vector<uint16_t> hx(kH), a(kH), b(kH);
  for (int it = 0; it < 24; ++it) {
    for (auto& v : hx) v = ToBf(nd(rng));
    x.CopyFromHost(hx);
    qwen4exp::ApplyBf16Linear(s, arena, router, x.data(), logits.data(), 1);
    if (q4x_moe_route_bf16(logits.data(), ids.data(), wts.data(), nullptr, 1, kE, kTopK, 1, s) != 0) throw std::runtime_error("route failed");
    R4DX_HIP_CHECK(qwen4exp::GroupedMoeDecodeSort(ids.data(), wts.data(), ids_sorted.data(), wts_sorted.data(), kTopK, s));
    R4DX_HIP_CHECK(qwen4exp::GroupedTrellisInputGateUp(x.data(), ids_sorted.data(), d_gusuh.data(), a0.data(), a1.data(), kTopK, s));
    R4DX_HIP_CHECK(qwen4exp::GroupedTrellisGemvGateUp(gu_bits, a0.data(), a1.data(), ids_sorted.data(), d_guw.data(), d_gusvh.data(),
                                                      guo.data(), ws_gu.data(), t_gu.data(), kTopK, s));
    R4DX_HIP_CHECK(qwen4exp::GroupedSiluMulTrellisInputDown(guo.data(), ids_sorted.data(), d_dnsuh.data(), a0d.data(), kTopK, s));
    R4DX_HIP_CHECK(qwen4exp::GroupedTrellisGemvDown(dn_bits, a0d.data(), ids_sorted.data(), d_dnw.data(), d_dnsvh.data(), yp.data(),
                                                    ws_dn.data(), t_dn.data(), kTopK, s));
    R4DX_HIP_CHECK(qwen4exp::GroupedMoeDecodeCombine(yp.data(), wts_sorted.data(), nullptr, nullptr, y_g.data(), kTopK, kH, s));
    arena.Reset();
    host.Forward(s, block_arena, mw, x.data(), y_h.data(), 1, fn);
    block_arena.Reset();
    R4DX_HIP_CHECK(hipStreamSynchronize(s));
    y_g.CopyToHost(a.data(), kH);
    y_h.CopyToHost(b.data(), kH);
    double sq = 0, sum = 0, mx = 0;
    for (int d = 0; d < kH; ++d) {
      const double r = Bf(b[static_cast<size_t>(d)]), g = Bf(a[static_cast<size_t>(d)]);
      if (!std::isfinite(g)) {
        std::fprintf(stderr, "FAIL: non-finite grouped output\n");
        ++g_failures;
      }
      sq += r * r;
      sum += std::fabs(g - r);
      mx = std::max(mx, std::fabs(g - r));
    }
    const double rms = std::sqrt(sq / kH);
    worst_max = std::max(worst_max, mx / rms);
    worst_mean = std::max(worst_mean, sum / kH / rms);
  }
  std::printf("qwen4exp geometry, gate_up %d-bit / down %d-bit, %d experts top-%d, 24 inputs: grouped vs host max|err|/rms %.4f mean %.5f\n",
              gu_bits, dn_bits, kE, kTopK, worst_max, worst_mean);
  if (worst_mean > 0.01 || worst_max > 0.06) {
    std::fprintf(stderr, "FAIL: grouped output differs from the host-routed one beyond bf16 noise\n");
    ++g_failures;
  }
  return 0;
}

}  // namespace

int main() {
  try {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    int n = 0;
    if (hipGetDeviceCount(&n) != hipSuccess || n < 1) {
      std::printf("SKIP: no HIP device\n");
      return kSkip;
    }
    RunBits(4, 4);
    RunBits(3, 4);
    RunBits(3, 3);
    std::printf("%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
  }
}
