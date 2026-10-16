// tests/model/test_lfm2_trellis_shapes.cpp -- docs/lfm2-plan.md L3-8 "trellis input tests for K 1536 / 11776": ApplyLinear on
// trellis linears of the REAL LFM2.5-24B-A2B shapes, which no tuning table row covers (every launch takes linear.cpp's
// FallbackTuning: M <= 16 split 128-column groups with SKG up to 4 and the tickets, M > 16 unsplit), at KB 4 and 5:
//
//   attn.q / attn.o / conv.out_proj   [2048, 2048]          attn.k / attn.v      [512, 2048]
//   conv.in_proj                      [6144, 2048]          dense mlp.down       [2048, 11776]
//   dense mlp.gate_up (two parts)     [23552, 2048] (parts 11776 + 11776: a part boundary that is an ODD multiple of 128)
//   expert gate_up (two parts)        [3072, 2048] (1536 + 1536)        expert down    [2048, 1536]
//
// Synthetic weights (random pair-grid words, fp16 suh / svh), so no container is needed. Per shape and rate:
//   - every element of the M in {1, 8, 16, 17, 65} calls (and M = 256 for the expert shapes: the sorted MoE path's
//     largest expert) within 4 bf16 ulp + 1e-4 row rms of trellis_ref's fp64 linear of the bit-exact transformed A
//     (test_trellis_linear's criterion);
//   - row identity: every row of the M = 8 call (a top-4 x 2-token sorted expert) equals the M = 1 call on that row --
//     what makes Lfm2MoeFfn's per-slot and sorted paths agree byte for byte;
//   - a second identical call gives identical bytes (the split-group tickets reset themselves).
// GPU test (one device). Not run by its author.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "linear.h"
#include "quant_linear.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/core/stream.hpp"
#include "test_common.h"
#include "trellis_ref.hpp"

using r4dx::core::Bf16ToFloat;
using r4dx::core::DeviceBuffer;
using r4dx::core::F16ToFloat;
using r4dx::core::FloatToBf16;
using r4dx::core::FloatToF16;
using r4dx::model::ApplyLinear;
using r4dx::model::Layout;
using r4dx::model::QuantLinear;

namespace {

int g_checks = 0, g_failures = 0;
void Check(bool ok, const std::string& what) {
  ++g_checks;
  std::printf("%s: %s\n", ok ? "PASS" : "FAIL", what.c_str());
  if (!ok) ++g_failures;
}

struct Shape {
  const char* name;
  int64_t N, K, part0;  // part0 > 0: two parts [part0, N - part0]
  bool expert;
};

struct Host {
  std::vector<uint16_t> q;          // decoded [K][N] f16
  std::vector<float> suh, svh;      // [parts][K], [N]
};

std::vector<uint16_t> RandomX(std::mt19937_64& rng, int64_t M, int64_t K) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(-8.f, 3.f), od(20.f, 60.f);
  std::uniform_int_distribution<int64_t> kd(0, K - 1);
  std::vector<uint16_t> x(static_cast<size_t>(M * K));
  std::vector<float> row(static_cast<size_t>(K));
  for (int64_t m = 0; m < M; ++m) {
    const float s = std::exp2(ud(rng));
    for (auto& v : row) v = nd(rng) * s;
    for (int i = 0; i < 2; ++i) row[static_cast<size_t>(kd(rng))] *= od(rng);
    for (int64_t k = 0; k < K; ++k) x[static_cast<size_t>(m * K + k)] = FloatToBf16(row[static_cast<size_t>(k)]);
  }
  return x;
}

// A synthetic trellis linear: random pair-grid words, fp16 suh (+-[0.5, 1.5]) / svh (+-[0.005, 0.02]) widened.
QuantLinear MakeLinear(const Shape& s, int kb, std::mt19937_64& rng, Host* h, DeviceBuffer<uint32_t>* tickets) {
  const int parts = s.part0 > 0 ? 2 : 1;
  std::vector<uint32_t> grid(static_cast<size_t>(s.N * s.K * kb / 32));
  for (auto& w : grid) w = static_cast<uint32_t>(rng());
  std::uniform_real_distribution<double> u(0.0, 1.0);
  const auto sgn = [&] { return (rng() & 1) ? 1.0 : -1.0; };
  h->suh.clear();
  h->svh.clear();
  for (int64_t i = 0; i < parts * s.K; ++i) h->suh.push_back(F16ToFloat(FloatToF16(static_cast<float>(sgn() * (0.5 + u(rng))))));
  for (int64_t i = 0; i < s.N; ++i) h->svh.push_back(F16ToFloat(FloatToF16(static_cast<float>(sgn() * (0.005 + 0.015 * u(rng))))));
  h->q = trellis_ref::DecodePairGrid(grid, s.K, s.N, kb);
  QuantLinear q;
  q.layout = Layout::kTrellis;
  q.N = s.N;
  q.K = s.K;
  q.trellis_bits = kb;
  q.trellis_parts = parts;
  q.trellis_prescale_log2 = 0;
  q.trellis_part_n[0] = parts > 1 ? s.part0 : s.N;
  q.trellis_part_n[1] = parts > 1 ? s.N - s.part0 : 0;
  q.trellis_w = DeviceBuffer<uint32_t>(grid.size());
  q.trellis_w.CopyFromHost(grid);
  q.trellis_suh = DeviceBuffer<float>(h->suh.size());
  q.trellis_suh.CopyFromHost(h->suh);
  q.trellis_svh = DeviceBuffer<float>(h->svh.size());
  q.trellis_svh.CopyFromHost(h->svh);
  *tickets = DeviceBuffer<uint32_t>(r4dx::core::r4d::GemmTrellisTicketsBytes(static_cast<int>(s.N)) / sizeof(uint32_t));
  tickets->Zero();
  q.trellis_tickets = tickets->data();
  return q;
}

std::vector<uint16_t> Run(hipStream_t st, r4dx::core::Arena& arena, const QuantLinear& q, const std::vector<uint16_t>& x,
                          int64_t M) {
  DeviceBuffer<uint16_t> xd(x.size()), yd(static_cast<size_t>(M * q.N));
  xd.CopyFromHost(x);
  std::vector<uint16_t> sent(yd.size(), 0x7FC1u);
  yd.CopyFromHost(sent);
  arena.Reset();
  ApplyLinear(st, arena, q, xd.data(), yd.data(), M);
  R4DX_HIP_CHECK(hipStreamSynchronize(st));
  return yd.CopyToHost();
}

}  // namespace

int main() {
  return r4dx_test::RunGuardedMain("test_lfm2_trellis_shapes", []() -> int {
    const Shape shapes[] = {
        {"attn.q/o, conv.out_proj", 2048, 2048, 0, false}, {"attn.k/v", 512, 2048, 0, false},
        {"conv.in_proj", 6144, 2048, 0, false},           {"mlp.gate_up (dense)", 23552, 2048, 11776, false},
        {"mlp.down (dense)", 2048, 11776, 0, false},       {"expert gate_up", 3072, 2048, 1536, true},
        {"expert down", 2048, 1536, 0, true},
    };
    r4dx::core::Stream stream;
    r4dx::core::Arena arena(64ull * 1024 * 1024);
    std::mt19937_64 rng(0x1F42ull);
    double worst = 0.0;
    for (const Shape& s : shapes) {
      for (int kb : {4, 5}) {
        Host h;
        DeviceBuffer<uint32_t> tickets;
        const QuantLinear q = MakeLinear(s, kb, rng, &h, &tickets);
        const std::string tag = std::string(s.name) + " [" + std::to_string(s.N) + ", " + std::to_string(s.K) + "] KB" +
                                std::to_string(kb);
        for (int64_t M : {int64_t{1}, int64_t{8}, int64_t{16}, int64_t{17}, int64_t{65}, int64_t{256}}) {
          if (M == 256 && !s.expert) continue;
          const r4dx::model::LinearTuning t = r4dx::model::PickTuning(Layout::kTrellis, s.N, s.K, std::min<int64_t>(M, 64), kb);
          const std::vector<uint16_t> x = RandomX(rng, M, s.K);
          const std::vector<uint16_t> y = Run(stream.get(), arena, q, x, M);
          std::vector<std::vector<uint16_t>> a;
          std::vector<const uint16_t*> ap;
          for (int p = 0; p < q.trellis_parts; ++p) a.push_back(trellis_ref::TransformInput(x, M, s.K, h.suh.data() + p * s.K, 0));
          for (const auto& v : a) ap.push_back(v.data());
          const std::vector<double> ya = trellis_ref::LinearFromA(ap, M, s.K, s.N, q.trellis_part_n[0], h.q, h.svh.data(), 0);
          size_t bad = 0;
          double w = 0.0;
          for (int64_t m = 0; m < M; ++m) {
            double ss = 0.0;
            for (int64_t n = 0; n < s.N; ++n) ss += ya[m * s.N + n] * ya[m * s.N + n];
            const double floor = 1e-4 * std::sqrt(ss / static_cast<double>(s.N));
            for (int64_t n = 0; n < s.N; ++n) {
              const size_t i = static_cast<size_t>(m * s.N + n);
              const double lim = 4.0 * trellis_ref::Bf16Ulp(ya[i]) + floor;
              const double err = std::fabs(static_cast<double>(Bf16ToFloat(y[i])) - ya[i]);
              if (!(err <= lim)) ++bad;
              w = std::max(w, err / lim);
            }
          }
          worst = std::max(worst, w);
          Check(bad == 0, tag + " M=" + std::to_string(M) + " (WV " + std::to_string(t.WV) + " SK " + std::to_string(t.SK) +
                              " MT " + std::to_string(t.MB) + " NP " + std::to_string(t.NPW) + " SKG " + std::to_string(t.SKG) +
                              " U " + std::to_string(t.U) + "): " + std::to_string(bad) + " element(s) beyond 4 bf16 ulp (worst " +
                              std::to_string(w) + " of the tolerance)");
          if (M == 8) {
            bool same = true;
            for (int64_t m = 0; m < M; ++m) {
              const std::vector<uint16_t> row(x.begin() + m * s.K, x.begin() + (m + 1) * s.K);
              const std::vector<uint16_t> y1 = Run(stream.get(), arena, q, row, 1);
              same = same && std::memcmp(y1.data(), y.data() + m * s.N, static_cast<size_t>(s.N) * 2) == 0;
            }
            Check(same, tag + " M=8: every row equals the M = 1 call on it (row identity)");
            Check(Run(stream.get(), arena, q, x, M) == y, tag + " M=8: a second call gives identical bytes (tickets)");
          }
        }
      }
    }
    std::printf("worst element: %.3f of the tolerance; %d check(s), %d failed\n", worst, g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
  });
}
