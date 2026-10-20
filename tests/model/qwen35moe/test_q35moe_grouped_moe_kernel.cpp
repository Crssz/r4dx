// Qwen35GroupedMoeDecode at the REAL expert count (256 experts, top-8, hidden 3072, expert width 1024) on synthetic trellis
// weights, against the host-routed MoE (qwen4exp::MoeBlockBase with per-expert ApplyLinear), plus the per-layer decode benchmark.
//
// The weights are random trellis words (any bit pattern decodes to some value), random +-1 input signs and small positive output
// scales: they test indexing, split-group tickets and rounding at 256 experts and give the achieved bandwidth, not accuracy.
// The bench cycles through R4DX_Q35_BENCH_LAYERS (default 6) layers of ~1.25 GB each and varies the input per call, so the routed
// experts of consecutive calls differ and the 64 MB Infinity cache cannot hold the working set.
//
//   R4DX_Q35_BENCH_LAYERS   layers of synthetic weights (default 6; the first one is also the comparison layer)
//   R4DX_Q35_BENCH_BITS     "gate_up,down" bits, default "4,4" (3 or 4 each)
//   R4DX_Q35_BENCH_ITERS    timed calls (default 400)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "linear.h"
#include "qwen35moe_grouped_decode.h"
#include "quant_linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/model/qwen4exp/hc_bf16_gemm.h"
#include "r4dx/model/qwen4exp/moe_block.h"
#include "r4dx/model/qwen4exp/moe_trellis_grouped.h"

using namespace r4dx;
using namespace r4dx::model;

namespace {

constexpr int kE = 256, kTopK = 8, kH = 3072, kI = 1024;
constexpr int kSkip = 77;

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

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

int TestDevice() {
  const char* hv = std::getenv("HIP_VISIBLE_DEVICES");
  if (hv != nullptr && hv[0] != 0) return 0;
  int n = 0;
  if (hipGetDeviceCount(&n) != hipSuccess) return 0;
  return n >= 2 ? 1 : 0;
}

int EnvInt(const char* name, int def) {
  const char* e = std::getenv(name);
  return e != nullptr && e[0] != 0 ? std::atoi(e) : def;
}

// Fills `bytes` of device memory at `dst` from a host tile, starting `start` bytes into the tile (wrapping).
void FillFromTile(void* dst, size_t bytes, const std::vector<uint8_t>& tile, size_t start) {
  uint8_t* d = static_cast<uint8_t*>(dst);
  size_t done = 0;
  while (done < bytes) {
    const size_t off = (start + done) % tile.size();
    const size_t n = std::min(bytes - done, tile.size() - off);
    R4DX_HIP_CHECK(hipMemcpy(d + done, tile.data() + off, n, hipMemcpyHostToDevice));
    done += n;
  }
}

core::DeviceBuffer<uint16_t> Bf16Buffer(size_t n, std::mt19937& rng, float stddev) {
  std::vector<uint16_t> h(n);
  std::normal_distribution<float> nd(0.0f, stddev);
  for (auto& v : h) v = ToBf(nd(rng));
  core::DeviceBuffer<uint16_t> b(n);
  b.CopyFromHost(h);
  return b;
}

QuantLinear Bf16Linear(int64_t N, int64_t K, std::mt19937& rng, float stddev) {
  QuantLinear q;
  q.layout = Layout::kBf16;
  q.N = N;
  q.K = K;
  q.bf16_w = Bf16Buffer(static_cast<size_t>(N * K), rng, stddev);
  return q;
}

struct SynthLayer {
  QuantLinear router, sgate_proj, sup_proj, sdown_proj, sgate;
  core::DeviceBuffer<uint32_t> gu_w, dn_w;
  core::DeviceBuffer<float> gu_suh, gu_svh, dn_suh, dn_svh;
  Qwen35MoeTrellisPtrs p;
  int64_t bytes_per_call = 0;  // weights one call streams: top-8 experts + router + shared expert
};

SynthLayer MakeLayer(int gu_bits, int dn_bits, std::mt19937& rng, const std::vector<uint8_t>& tile, size_t seed_off) {
  SynthLayer L;
  L.router = Bf16Linear(kE, kH, rng, 0.05f);
  L.sgate_proj = Bf16Linear(kI, kH, rng, 0.02f);
  L.sup_proj = Bf16Linear(kI, kH, rng, 0.02f);
  L.sdown_proj = Bf16Linear(kH, kI, rng, 0.02f);
  L.sgate = Bf16Linear(1, kH, rng, 0.05f);
  const size_t gu_per = static_cast<size_t>(2 * kI) * kH * gu_bits / 32, dn_per = static_cast<size_t>(kH) * kI * dn_bits / 32;
  L.gu_w = core::DeviceBuffer<uint32_t>(kE * gu_per);
  L.dn_w = core::DeviceBuffer<uint32_t>(kE * dn_per);
  FillFromTile(L.gu_w.data(), L.gu_w.bytes(), tile, seed_off);
  FillFromTile(L.dn_w.data(), L.dn_w.bytes(), tile, seed_off * 7 + 12345);
  const auto signs = [&](size_t n, float mag) {
    std::vector<float> h(n);
    std::uniform_int_distribution<int> coin(0, 1);
    for (auto& v : h) v = coin(rng) ? mag : -mag;
    return h;
  };
  const auto scales = [&](size_t n) {
    std::vector<float> h(n);
    std::uniform_real_distribution<float> u(0.02f, 0.06f);
    for (auto& v : h) v = u(rng);
    return h;
  };
  L.gu_suh = core::DeviceBuffer<float>(static_cast<size_t>(kE) * 2 * kH);
  L.gu_suh.CopyFromHost(signs(L.gu_suh.size(), 1.0f));
  L.gu_svh = core::DeviceBuffer<float>(static_cast<size_t>(kE) * 2 * kI);
  L.gu_svh.CopyFromHost(scales(L.gu_svh.size()));
  L.dn_suh = core::DeviceBuffer<float>(static_cast<size_t>(kE) * kI);
  L.dn_suh.CopyFromHost(signs(L.dn_suh.size(), 1.0f));
  L.dn_svh = core::DeviceBuffer<float>(static_cast<size_t>(kE) * kH);
  L.dn_svh.CopyFromHost(scales(L.dn_svh.size()));
  L.p = {L.gu_w.data(), L.gu_suh.data(), L.gu_svh.data(), gu_bits, L.dn_w.data(), L.dn_suh.data(), L.dn_svh.data(), dn_bits};
  L.bytes_per_call = static_cast<int64_t>(kTopK) * (gu_per * 4 + dn_per * 4 + (2 * kH + 2 * kI + kI + kH) * 4) +
                     static_cast<int64_t>(kE) * kH * 2 + 3LL * kH * kI * 2 + kH * 2;
  return L;
}

// Per-expert views for the host-routed reference.
struct HostViews {
  std::vector<QuantLinear> gu, dn;
  core::DeviceBuffer<uint32_t> tickets;
};

HostViews MakeViews(const SynthLayer& L) {
  HostViews v;
  const size_t tk_gu = core::r4d::GemmTrellisTicketsBytes(2 * kI) / 4, tk_dn = core::r4d::GemmTrellisTicketsBytes(kH) / 4;
  v.tickets = core::DeviceBuffer<uint32_t>(kE * (tk_gu + tk_dn));
  v.tickets.Zero();
  const size_t gu_per = static_cast<size_t>(2 * kI) * kH * L.p.gate_up_bits / 32, dn_per = static_cast<size_t>(kH) * kI * L.p.down_bits / 32;
  for (int e = 0; e < kE; ++e) {
    QuantLinear g;
    g.layout = Layout::kTrellis;
    g.N = 2 * kI;
    g.K = kH;
    g.trellis_bits = L.p.gate_up_bits;
    g.trellis_parts = 2;
    g.trellis_part_n[0] = kI;
    g.trellis_part_n[1] = kI;
    g.trellis_w = core::DeviceBuffer<uint32_t>::View(const_cast<uint32_t*>(L.p.gate_up_w) + e * gu_per, gu_per);
    g.trellis_suh = core::DeviceBuffer<float>::View(const_cast<float*>(L.p.gate_up_suh) + static_cast<size_t>(e) * 2 * kH, 2 * kH);
    g.trellis_svh = core::DeviceBuffer<float>::View(const_cast<float*>(L.p.gate_up_svh) + static_cast<size_t>(e) * 2 * kI, 2 * kI);
    g.trellis_tickets = v.tickets.data() + static_cast<size_t>(e) * (tk_gu + tk_dn);
    v.gu.push_back(std::move(g));
    QuantLinear d;
    d.layout = Layout::kTrellis;
    d.N = kH;
    d.K = kI;
    d.trellis_bits = L.p.down_bits;
    d.trellis_parts = 1;
    d.trellis_part_n[0] = kH;
    d.trellis_w = core::DeviceBuffer<uint32_t>::View(const_cast<uint32_t*>(L.p.down_w) + e * dn_per, dn_per);
    d.trellis_suh = core::DeviceBuffer<float>::View(const_cast<float*>(L.p.down_suh) + static_cast<size_t>(e) * kI, kI);
    d.trellis_svh = core::DeviceBuffer<float>::View(const_cast<float*>(L.p.down_svh) + static_cast<size_t>(e) * kH, kH);
    d.trellis_tickets = v.tickets.data() + static_cast<size_t>(e) * (tk_gu + tk_dn) + tk_gu;
    v.dn.push_back(std::move(d));
  }
  return v;
}

}  // namespace

int RunTest();
int main() {
  try {
    return RunTest();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 2;
  }
}

int RunTest() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  int ndev = 0;
  if (hipGetDeviceCount(&ndev) != hipSuccess || ndev < 1) {
    std::printf("SKIP: no HIP device\n");
    return kSkip;
  }
  R4DX_HIP_CHECK(hipSetDevice(TestDevice()));
  const int layers = std::max(1, EnvInt("R4DX_Q35_BENCH_LAYERS", 6));
  int gu_bits = 4, dn_bits = 4;
  if (const char* b = std::getenv("R4DX_Q35_BENCH_BITS"); b != nullptr && std::sscanf(b, "%d,%d", &gu_bits, &dn_bits) != 2) {
    std::fprintf(stderr, "R4DX_Q35_BENCH_BITS wants \"gate_up,down\"\n");
    return 2;
  }
  const int iters = std::max(10, EnvInt("R4DX_Q35_BENCH_ITERS", 400));
  {
    const int gu_skg = EnvInt("R4DX_Q35_GU_SKG", qwen4exp::kGroupedQ35GuSkg), dn_skg = EnvInt("R4DX_Q35_DOWN_SKG", qwen4exp::kGroupedQ35DownSkg);
    if (qwen4exp::GroupedQ35SetSplits(gu_skg, dn_skg) != hipSuccess) {
      std::fprintf(stderr, "unsupported split-group counts %d / %d\n", gu_skg, dn_skg);
      return 2;
    }
    std::printf("split groups: gate_up %d, down %d\n", gu_skg, dn_skg);
  }
  {
    size_t free_b = 0, total_b = 0;
    R4DX_HIP_CHECK(hipMemGetInfo(&free_b, &total_b));
    const double need = layers * 1.3e9 + 1.5e9;
    if (static_cast<double>(free_b) < need) {
      std::printf("SKIP: %.1f GB free, %.1f GB needed for %d synthetic layers\n", free_b / 1e9, need / 1e9, layers);
      return kSkip;
    }
  }

  std::mt19937 rng(1234);
  std::vector<uint8_t> tile(size_t{48} << 20);
  {
    std::mt19937_64 r64(99);
    for (size_t i = 0; i + 8 <= tile.size(); i += 8) {
      const uint64_t v = r64();
      std::memcpy(tile.data() + i, &v, 8);
    }
  }
  std::printf("building %d synthetic layer(s): 256 experts, gate_up %d-bit, down %d-bit ...\n", layers, gu_bits, dn_bits);
  std::vector<SynthLayer> L;
  L.reserve(static_cast<size_t>(layers));
  for (int i = 0; i < layers; ++i) L.push_back(MakeLayer(gu_bits, dn_bits, rng, tile, 4096u * (i + 1) + 77));

  core::Stream stream;
  const hipStream_t s = stream.get();
  core::Arena arena;
  arena.Reserve(size_t{64} << 20);
  Qwen35GroupedMoeDecode moe(kE, kTopK, /*norm_topk=*/true);

  // 64 distinct inputs
  constexpr int kInputs = 64;
  core::DeviceBuffer<uint16_t> xs = Bf16Buffer(static_cast<size_t>(kInputs) * kH, rng, 1.0f);
  core::DeviceBuffer<uint16_t> y_grp(kH), y_host(kH);

  const auto weights_of = [](const SynthLayer& l) {
    return Qwen35GroupedMoeWeights{&l.router, &l.sgate_proj, &l.sup_proj, &l.sdown_proj, &l.sgate};
  };

  // ---- correctness against the host-routed MoE (layer 0) ------------------------------------------------------------------
  {
    HostViews hv = MakeViews(L[0]);
    qwen4exp::MoeConfig mc;
    mc.hidden = kH;
    mc.experts = kE;
    mc.top_k = kTopK;
    mc.inter = kI;
    mc.shared_inter = kI;
    mc.norm_topk = true;
    qwen4exp::MoeBlockBase host(mc, 1);
    qwen4exp::MoeWeights mw;
    mw.router = &L[0].router;
    mw.shared_gate_proj = &L[0].sgate_proj;
    mw.shared_up_proj = &L[0].sup_proj;
    mw.shared_down_proj = &L[0].sdown_proj;
    mw.shared_gate = &L[0].sgate;
    core::Arena expert_arena;
    expert_arena.Reserve(size_t{96} << 20);
    core::Arena block_arena;
    block_arena.Reserve(size_t{256} << 20);
    const qwen4exp::ExpertLinearFn fn = [&](int e, qwen4exp::ExpertLinear which, const uint16_t* xin, int64_t rows, uint16_t* y,
                                            core::Arena&) {
      expert_arena.Reset();
      ApplyLinear(s, expert_arena, which == qwen4exp::ExpertLinear::kGateUp ? hv.gu[static_cast<size_t>(e)] : hv.dn[static_cast<size_t>(e)],
                  xin, y, rows);
    };
    std::set<int> used;
    double worst_max = 0, worst_mean = 0, sum_mean = 0;
    int nan_rows = 0;
    double rms_sum = 0;
    std::vector<uint16_t> a(kH), b(kH);
    std::vector<int32_t> ids(kTopK);
    std::vector<float> wts(kTopK);
    for (int i = 0; i < kInputs; ++i) {
      const uint16_t* x = xs.data() + static_cast<size_t>(i) * kH;
      moe.Run(s, arena, weights_of(L[0]), L[0].p, x, y_grp.data());
      moe.ReadSelection(s, ids.data(), wts.data());
      for (int j = 0; j < kTopK; ++j) used.insert(ids[static_cast<size_t>(j)]);
      arena.Reset();
      host.Forward(s, block_arena, mw, x, y_host.data(), 1, fn);
      block_arena.Reset();
      R4DX_HIP_CHECK(hipStreamSynchronize(s));
      y_grp.CopyToHost(a.data(), kH);
      y_host.CopyToHost(b.data(), kH);
      double sq = 0, sum = 0, mx = 0;
      bool bad = false;
      for (int d = 0; d < kH; ++d) {
        const double r = Bf(b[static_cast<size_t>(d)]), g = Bf(a[static_cast<size_t>(d)]);
        if (!std::isfinite(g) || !std::isfinite(r)) bad = true;
        sq += r * r;
        sum += std::fabs(g - r);
        mx = std::max(mx, std::fabs(g - r));
      }
      const double rms = std::sqrt(sq / kH);
      rms_sum += rms;
      nan_rows += bad ? 1 : 0;
      if (rms > 0) {
        worst_max = std::max(worst_max, mx / rms);
        worst_mean = std::max(worst_mean, sum / kH / rms);
        sum_mean += sum / kH / rms;
      }
    }
    std::printf("grouped vs host-routed (layer 0, %d inputs, %zu distinct experts selected, max id %d): max|err|/rms worst %.4f, "
                "mean|err|/rms worst %.5f avg %.5f, non-finite rows %d (reference output rms %.4g)\n",
                kInputs, used.size(), *used.rbegin(), worst_max, worst_mean, sum_mean / kInputs, nan_rows, rms_sum / kInputs);
    Check(nan_rows == 0, "non-finite MoE output");
    Check(worst_mean <= 0.01, "grouped output differs from the host-routed one beyond bf16 noise (mean/rms " + std::to_string(worst_mean) + ")");
    Check(used.size() > 100, "the router should have spread the 64 inputs over many experts");

    // ---- host-routed timing (layer 0) -------------------------------------------------------------------------------------
    for (int i = 0; i < 10; ++i) {
      host.Forward(s, block_arena, mw, xs.data() + static_cast<size_t>(i) * kH, y_host.data(), 1, fn);
      block_arena.Reset();
    }
    R4DX_HIP_CHECK(hipStreamSynchronize(s));
    const int hit = 100;
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < hit; ++i) {
      host.Forward(s, block_arena, mw, xs.data() + static_cast<size_t>(i % kInputs) * kH, y_host.data(), 1, fn);
      block_arena.Reset();
    }
    R4DX_HIP_CHECK(hipStreamSynchronize(s));
    const double host_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() / hit;
    std::printf("host-routed MoE layer (one layer's weights, so the 8 experts partly stay cached): %.3f ms per call\n", host_ms);
  }

  // ---- grouped benchmark over `layers` layers --------------------------------------------------------------------------
  {
    for (int i = 0; i < 40; ++i) {
      moe.Run(s, arena, weights_of(L[static_cast<size_t>(i % layers)]), L[static_cast<size_t>(i % layers)].p,
              xs.data() + static_cast<size_t>(i % kInputs) * kH, y_grp.data());
      arena.Reset();
    }
    R4DX_HIP_CHECK(hipStreamSynchronize(s));
    hipEvent_t e0, e1;
    R4DX_HIP_CHECK(hipEventCreate(&e0));
    R4DX_HIP_CHECK(hipEventCreate(&e1));
    R4DX_HIP_CHECK(hipStreamSynchronize(s));
    const auto wall0 = std::chrono::steady_clock::now();
    R4DX_HIP_CHECK(hipEventRecord(e0, s));
    for (int i = 0; i < iters; ++i) {
      const SynthLayer& l = L[static_cast<size_t>(i % layers)];
      moe.Run(s, arena, weights_of(l), l.p, xs.data() + static_cast<size_t>(i % kInputs) * kH, y_grp.data());
      arena.Reset();
    }
    R4DX_HIP_CHECK(hipEventRecord(e1, s));
    R4DX_HIP_CHECK(hipEventSynchronize(e1));
    R4DX_HIP_CHECK(hipStreamSynchronize(s));
    const double wall_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
    float ms = 0;
    R4DX_HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    std::printf("   (hipEvent span %.4f ms per call, host wall %.4f ms per call; the wall number is the one reported)\n", ms / iters, wall_ms / iters);
    const double per = wall_ms / iters;
    const double gb = static_cast<double>(L[0].bytes_per_call) / 1e9;
    std::printf("\ngrouped MoE decode, %d calls over %d layers (gate_up %d-bit, down %d-bit): %.4f ms per layer-call\n", iters, layers,
                gu_bits, dn_bits, per);
    std::printf("   weights streamed per call: %.1f MB (top-8 experts %.1f MB + router/shared expert %.1f MB) -> %.0f GB/s achieved\n",
                gb * 1e3, (gb - (static_cast<double>(kE) * kH * 2 + 3.0 * kH * kI * 2) / 1e9) * 1e3,
                (static_cast<double>(kE) * kH * 2 + 3.0 * kH * kI * 2) / 1e6, gb / (per * 1e-3));

    // phase breakdown: an event after every phase (adds a little overhead per phase)
    std::vector<hipEvent_t> evs;
    std::vector<std::string> names;
    const std::function<void(const char*)> mark = [&](const char* name) {
      hipEvent_t e;
      static_cast<void>(hipEventCreate(&e));
      static_cast<void>(hipEventRecord(e, s));
      evs.push_back(e);
      names.push_back(name);
    };
    std::map<std::string, double> acc;
    std::vector<std::string> order;
    const int pit = std::max(20, iters / 4);
    for (int i = 0; i < pit; ++i) {
      evs.clear();
      names.clear();
      const SynthLayer& l = L[static_cast<size_t>(i % layers)];
      moe.Run(s, arena, weights_of(l), l.p, xs.data() + static_cast<size_t>(i % kInputs) * kH, y_grp.data(), &mark);
      arena.Reset();
      R4DX_HIP_CHECK(hipStreamSynchronize(s));
      for (size_t k = 1; k < evs.size(); ++k) {
        float t = 0;
        R4DX_HIP_CHECK(hipEventElapsedTime(&t, evs[k - 1], evs[k]));
        if (acc.find(names[k]) == acc.end()) order.push_back(names[k]);
        acc[names[k]] += t;
      }
      for (hipEvent_t e : evs) static_cast<void>(hipEventDestroy(e));
    }
    std::printf("   phases (event-bracketed, sequential kernels):\n");
    for (const std::string& n : order) std::printf("     %-22s %.4f ms\n", n.c_str(), acc[n] / pit);
    static_cast<void>(hipEventDestroy(e0));
    static_cast<void>(hipEventDestroy(e1));
  }

  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
