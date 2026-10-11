// tests/kernels/test_gdn_chunk_scan_stable.cpp -- r4d_gdn_chunk_scan_stable_k128_v128_c64_bf16 (the span-safe GDN chunk scan, a unit of
// its own: third_party/libr4d/r4d_gdn_chunk_scan_stable_*.hip) and the shipped r4d_gdn_chunk_scan_k128_v128_c64_bf16, both after
// r4d_gdn_kkt_solve, against the exact fp64 per-token recurrence of HF's torch_recurrent_gated_delta_rule:
//
//   state <- state * exp(g_t);  kv_mem = state^T k_t;  delta = (v_t - kv_mem) * beta_t;  state += k_t delta^T;  out_t = scale state^T q_t
//
// T = 100 rows (a 64-row chunk and a 36-row one: the state is carried between chunks), H = 6 value heads over Hg = 2 key heads (the
// real models' 3x grouping), h0 = 0. Two gate regimes:
//   mild    per-token g in the range tests/kernels/test_gdn_chunk_scan.cpp uses (a 64-row chunk's span stays under ~25 nats): the
//           shipped scan and the span-safe one must both match the recurrence.
//   strong  A = e^A_log from 0.03 to 158 (Qwen3.8-Flash-Next layer 0's range; tools/reference/qwen4exp/gdn_probe.py): per-token g down
//           to -hundreds and a chunk gate span in the thousands. The shipped scan splits e^{G_i-G_j} around the chunk midpoint and
//           clamps each half at e^80 (its source says "attenuated, not exact" beyond a span of ~176), so its error here is REPORTED, not
//           bounded; the span-safe scan must match the recurrence at bf16 level.
// Also the model's own call shape: the 100 rows as one 64-row call and one 36-row call, the second continuing from the first's ht
// (h0 = ht), which must give the same result as the single 100-row call.
// Every head is compared on its own (a head that fails is not hidden by the other five).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"

using namespace r4dx::core;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

void L2NormRows(std::vector<uint16_t>* x, int rows, int dim) {
  for (int r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int d = 0; d < dim; ++d) {
      const float v = Bf16ToFloat((*x)[static_cast<size_t>(r) * dim + d]);
      ss += static_cast<double>(v) * v;
    }
    const float inv = 1.0f / std::sqrt(static_cast<float>(ss) + 1e-6f);
    for (int d = 0; d < dim; ++d) {
      (*x)[static_cast<size_t>(r) * dim + d] = FloatToBf16(Bf16ToFloat((*x)[static_cast<size_t>(r) * dim + d]) * inv);
    }
  }
}

constexpr int kT = 100, kH = 6, kHg = 2, kK = 128, kV = 128, kChunk = 64;

struct Inputs {
  std::vector<uint16_t> q, k, v;  // q, k [T, Hg, K], v [T, H, V]
  std::vector<float> g_raw, beta;  // [T, H]
  double span_max = 0;             // largest chunk gate span over (chunk, head)
};

// log_a_lo / log_a_hi: A = e^A_log is drawn log-uniform per head in [e^lo, e^hi]; dt_bias in [dt_lo, dt_hi]; a ~ U(-a_mag, a_mag).
Inputs MakeInputs(unsigned seed, float log_a_lo, float log_a_hi, float dt_lo, float dt_hi, float a_mag) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f), b_dist(-2.0f, 2.0f);
  Inputs in;
  in.q.resize(static_cast<size_t>(kT) * kHg * kK);
  in.k.resize(in.q.size());
  in.v.resize(static_cast<size_t>(kT) * kH * kV);
  for (auto& x : in.q) x = FloatToBf16(dist(rng));
  for (auto& x : in.k) x = FloatToBf16(dist(rng));
  for (auto& x : in.v) x = FloatToBf16(dist(rng));
  L2NormRows(&in.q, kT * kHg, kK);
  L2NormRows(&in.k, kT * kHg, kK);
  std::uniform_real_distribution<float> alog(log_a_lo, log_a_hi), dt(dt_lo, dt_hi), a(-a_mag, a_mag);
  std::vector<float> A_log(kH), dt_bias(kH);
  for (int h = 0; h < kH; ++h) {
    A_log[h] = alog(rng);
    dt_bias[h] = dt(rng);
  }
  in.g_raw.resize(static_cast<size_t>(kT) * kH);
  in.beta.resize(in.g_raw.size());
  for (int t = 0; t < kT; ++t) {
    for (int h = 0; h < kH; ++h) {
      const float x = a(rng) + dt_bias[h];
      const float softplus = x > 20.0f ? x : std::log1p(std::exp(x));
      in.g_raw[static_cast<size_t>(t) * kH + h] = -std::exp(A_log[h]) * softplus;
      in.beta[static_cast<size_t>(t) * kH + h] = 1.0f / (1.0f + std::exp(-b_dist(rng)));
    }
  }
  for (int h = 0; h < kH; ++h) {
    for (int c0 = 0; c0 < kT; c0 += kChunk) {
      double s = 0;
      for (int t = c0 + 1; t < std::min(kT, c0 + kChunk); ++t) s -= in.g_raw[static_cast<size_t>(t) * kH + h];
      in.span_max = std::max(in.span_max, s);
    }
  }
  return in;
}

// The exact recurrence in fp64. Returns out [T, H, V].
std::vector<double> Reference(const Inputs& in, float scale) {
  std::vector<double> out(static_cast<size_t>(kT) * kH * kV, 0.0);
  std::vector<double> state(static_cast<size_t>(kH) * kK * kV, 0.0);  // [H, K, V]
  const int ratio = kH / kHg;
  for (int t = 0; t < kT; ++t) {
    for (int h = 0; h < kH; ++h) {
      const int hq = h / ratio;
      double* s = &state[static_cast<size_t>(h) * kK * kV];
      const double decay = std::exp(static_cast<double>(in.g_raw[static_cast<size_t>(t) * kH + h]));
      for (size_t i = 0; i < static_cast<size_t>(kK) * kV; ++i) s[i] *= decay;
      std::vector<double> kv_mem(kV, 0.0), delta(kV);
      for (int k = 0; k < kK; ++k) {
        const double kt = Bf16ToFloat(in.k[(static_cast<size_t>(t) * kHg + hq) * kK + k]);
        for (int v = 0; v < kV; ++v) kv_mem[v] += s[static_cast<size_t>(k) * kV + v] * kt;
      }
      const double beta = in.beta[static_cast<size_t>(t) * kH + h];
      for (int v = 0; v < kV; ++v) {
        delta[v] = (static_cast<double>(Bf16ToFloat(in.v[(static_cast<size_t>(t) * kH + h) * kV + v])) - kv_mem[v]) * beta;
      }
      for (int k = 0; k < kK; ++k) {
        const double kt = Bf16ToFloat(in.k[(static_cast<size_t>(t) * kHg + hq) * kK + k]);
        for (int v = 0; v < kV; ++v) s[static_cast<size_t>(k) * kV + v] += kt * delta[v];
      }
      for (int v = 0; v < kV; ++v) {
        double acc = 0.0;
        for (int k = 0; k < kK; ++k) {
          acc += s[static_cast<size_t>(k) * kV + v] * Bf16ToFloat(in.q[(static_cast<size_t>(t) * kHg + hq) * kK + k]);
        }
        out[(static_cast<size_t>(t) * kH + h) * kV + v] = scale * acc;
      }
    }
  }
  return out;
}

// The kernels' gate input: the cumsum of g_raw within each call's 64-row chunks (rows [r0, r1) are one call).
std::vector<float> ChunkCumsum(const Inputs& in, int r0, int r1) {
  std::vector<float> g(static_cast<size_t>(r1 - r0) * kH);
  for (int h = 0; h < kH; ++h) {
    float run = 0.0f;
    for (int t = r0; t < r1; ++t) {
      if ((t - r0) % kChunk == 0) run = 0.0f;
      run += in.g_raw[static_cast<size_t>(t) * kH + h];
      g[static_cast<size_t>(t - r0) * kH + h] = run;
    }
  }
  return g;
}

// Runs kkt_solve + the scan on rows [r0, r1) as ONE call with state h0 (fp32 [H, V, K], nullptr = zeros); writes out rows and ht.
void RunCall(bool stable, const Inputs& in, int r0, int r1, const std::vector<float>* h0, float scale,
             std::vector<uint16_t>* out_rows, std::vector<float>* ht) {
  const int T = r1 - r0;
  const std::vector<float> g = ChunkCumsum(in, r0, r1);
  std::vector<uint16_t> q(in.q.begin() + static_cast<size_t>(r0) * kHg * kK, in.q.begin() + static_cast<size_t>(r1) * kHg * kK);
  std::vector<uint16_t> k(in.k.begin() + static_cast<size_t>(r0) * kHg * kK, in.k.begin() + static_cast<size_t>(r1) * kHg * kK);
  std::vector<uint16_t> v(in.v.begin() + static_cast<size_t>(r0) * kH * kV, in.v.begin() + static_cast<size_t>(r1) * kH * kV);
  std::vector<float> beta(in.beta.begin() + static_cast<size_t>(r0) * kH, in.beta.begin() + static_cast<size_t>(r1) * kH);
  std::vector<int32_t> cu = {0, T};
  DeviceBuffer<uint16_t> q_d(q.size()), k_d(k.size()), v_d(v.size()), A_d(static_cast<size_t>(T) * kH * kChunk),
      o_d(static_cast<size_t>(T) * kH * kV);
  DeviceBuffer<float> g_d(g.size()), beta_d(beta.size()), h0_d(static_cast<size_t>(kH) * kV * kK), ht_d(static_cast<size_t>(kH) * kV * kK);
  DeviceBuffer<int32_t> cu_d(cu.size());
  q_d.CopyFromHost(q);
  k_d.CopyFromHost(k);
  v_d.CopyFromHost(v);
  g_d.CopyFromHost(g);
  beta_d.CopyFromHost(beta);
  cu_d.CopyFromHost(cu);
  if (h0 != nullptr) {
    h0_d.CopyFromHost(*h0);
  } else {
    h0_d.Zero();
  }
  r4d::GdnKktSolve(k_d.data(), beta_d.data(), g_d.data(), A_d.data(), cu_d.data(), 1, T, kH, kHg, kK, kChunk, nullptr);
  if (stable) {
    r4d::GdnChunkScanStable(q_d.data(), k_d.data(), v_d.data(), A_d.data(), g_d.data(), beta_d.data(), h0_d.data(), o_d.data(),
                            ht_d.data(), cu_d.data(), 1, kH, kHg, kK, kV, kChunk, scale, nullptr);
  } else {
    r4d::GdnChunkScan(q_d.data(), k_d.data(), v_d.data(), A_d.data(), g_d.data(), beta_d.data(), h0_d.data(), o_d.data(), ht_d.data(),
                      cu_d.data(), 1, kH, kHg, kK, kV, kChunk, scale, nullptr);
  }
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  *out_rows = o_d.CopyToHost();
  *ht = ht_d.CopyToHost();
}

struct Err {
  double norm_rel = 0, worst_head = 0;  // sqrt(sum err^2 / sum ref^2) over everything / over the worst head
  bool finite = true;
};
Err Compare(const std::vector<uint16_t>& got, const std::vector<double>& ref, int row0, int rows) {
  Err e;
  double num = 0, den = 0;
  std::vector<double> hn(kH, 0.0), hd(kH, 0.0);
  for (int t = 0; t < rows; ++t) {
    for (int h = 0; h < kH; ++h) {
      for (int v = 0; v < kV; ++v) {
        const double g = Bf16ToFloat(got[(static_cast<size_t>(t) * kH + h) * kV + v]);
        const double r = ref[(static_cast<size_t>(row0 + t) * kH + h) * kV + v];
        if (!std::isfinite(g)) e.finite = false;
        num += (g - r) * (g - r);
        den += r * r;
        hn[h] += (g - r) * (g - r);
        hd[h] += r * r;
      }
    }
  }
  e.norm_rel = std::sqrt(num) / std::max(1e-12, std::sqrt(den));
  for (int h = 0; h < kH; ++h) e.worst_head = std::max(e.worst_head, std::sqrt(hn[h]) / std::max(1e-12, std::sqrt(hd[h])));
  return e;
}

void Scenario(const char* name, const Inputs& in, bool bound_shipped) {
  const float scale = 1.0f / std::sqrt(static_cast<float>(kK));
  const std::vector<double> ref = Reference(in, scale);
  std::printf("\n-- %s: chunk gate span up to %.1f nats; min per-token g %.1f\n", name, in.span_max,
              static_cast<double>(*std::min_element(in.g_raw.begin(), in.g_raw.end())));
  for (const bool stable : {false, true}) {
    const char* kn = stable ? "span-safe" : "shipped  ";
    std::vector<uint16_t> whole, a, b;
    std::vector<float> ht_whole, ht_a, ht_b;
    RunCall(stable, in, 0, kT, nullptr, scale, &whole, &ht_whole);
    RunCall(stable, in, 0, kChunk, nullptr, scale, &a, &ht_a);
    RunCall(stable, in, kChunk, kT, &ht_a, scale, &b, &ht_b);
    const Err ew = Compare(whole, ref, 0, kT);
    const Err ea = Compare(a, ref, 0, kChunk);
    const Err eb = Compare(b, ref, kChunk, kT - kChunk);
    std::printf("  %s  one 100-row call: norm rel err %.3e (worst head %.3e);  64 + 36 rows, h0 = ht: %.3e / %.3e (worst head %.3e / %.3e)\n", kn,
                ew.norm_rel, ew.worst_head, ea.norm_rel, eb.norm_rel, ea.worst_head, eb.worst_head);
    Check(ew.finite && ea.finite && eb.finite, std::string(name) + " " + kn + ": non-finite output");
    if (stable || bound_shipped) {
      Check(ew.worst_head < 5e-2, std::string(name) + " " + kn + " one call: worst head above 5e-2");
      Check(ea.worst_head < 5e-2 && eb.worst_head < 5e-2, std::string(name) + " " + kn + " 64+36 calls: worst head above 5e-2");
    }
    // the final state of the two-call path equals the one-call path's (both carry fp32 state; the second call reads h0 = ht)
    double dn = 0, dd = 0;
    for (size_t i = 0; i < ht_whole.size(); ++i) {
      dn += static_cast<double>(ht_whole[i] - ht_b[i]) * (ht_whole[i] - ht_b[i]);
      dd += static_cast<double>(ht_whole[i]) * ht_whole[i];
    }
    std::printf("            final state, one call vs 64 + 36: rel diff %.3e\n", std::sqrt(dn) / std::max(1e-12, std::sqrt(dd)));
    if (stable || bound_shipped) Check(std::sqrt(dn) / std::max(1e-12, std::sqrt(dd)) < 5e-2, std::string(name) + " " + kn + ": final state differs");
  }
}

// Informational (no bound): the scan kernels alone on a 4096-row prompt at the real models' shape (H 48, Hg 16), mild gate; kkt_solve is run once.
void Timing() {
  const int T = 4096, H = 48, Hg = 16, iters = 50;
  const float scale = 1.0f / std::sqrt(static_cast<float>(kK));
  std::mt19937 rng(5);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<uint16_t> q(static_cast<size_t>(T) * Hg * kK), k(q.size()), v(static_cast<size_t>(T) * H * kV);
  for (auto& x : q) x = FloatToBf16(dist(rng));
  for (auto& x : k) x = FloatToBf16(dist(rng));
  for (auto& x : v) x = FloatToBf16(dist(rng));
  L2NormRows(&q, T * Hg, kK);
  L2NormRows(&k, T * Hg, kK);
  std::vector<float> g(static_cast<size_t>(T) * H), beta(g.size());
  for (int h = 0; h < H; ++h) {
    for (int t = 0; t < T; ++t) {
      float run = (t % kChunk == 0) ? 0.0f : g[static_cast<size_t>(t - 1) * H + h];
      run += -0.05f * (1.0f + 0.5f * dist(rng));
      g[static_cast<size_t>(t) * H + h] = run;
      beta[static_cast<size_t>(t) * H + h] = 0.5f;
    }
  }
  std::vector<int32_t> cu = {0, T};
  DeviceBuffer<uint16_t> q_d(q.size()), k_d(k.size()), v_d(v.size()), A_d(static_cast<size_t>(T) * H * kChunk), o_d(v.size());
  DeviceBuffer<float> g_d(g.size()), beta_d(beta.size()), h0_d(static_cast<size_t>(H) * kV * kK), ht_d(static_cast<size_t>(H) * kV * kK);
  DeviceBuffer<int32_t> cu_d(cu.size());
  q_d.CopyFromHost(q);
  k_d.CopyFromHost(k);
  v_d.CopyFromHost(v);
  g_d.CopyFromHost(g);
  beta_d.CopyFromHost(beta);
  cu_d.CopyFromHost(cu);
  h0_d.Zero();
  r4d::GdnKktSolve(k_d.data(), beta_d.data(), g_d.data(), A_d.data(), cu_d.data(), 1, T, H, Hg, kK, kChunk, nullptr);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  hipEvent_t e0, e1;
  R4DX_HIP_CHECK(hipEventCreate(&e0));
  R4DX_HIP_CHECK(hipEventCreate(&e1));
  double us[2] = {0, 0};
  for (int s = 0; s < 2; ++s) {
    const bool stable = s == 1;
    for (int rep = 0; rep < 2; ++rep) {  // rep 0 warms up
      R4DX_HIP_CHECK(hipEventRecord(e0, nullptr));
      for (int i = 0; i < iters; ++i) {
        if (stable) {
          r4d::GdnChunkScanStable(q_d.data(), k_d.data(), v_d.data(), A_d.data(), g_d.data(), beta_d.data(), h0_d.data(), o_d.data(),
                                  ht_d.data(), cu_d.data(), 1, H, Hg, kK, kV, kChunk, scale, nullptr);
        } else {
          r4d::GdnChunkScan(q_d.data(), k_d.data(), v_d.data(), A_d.data(), g_d.data(), beta_d.data(), h0_d.data(), o_d.data(), ht_d.data(),
                            cu_d.data(), 1, H, Hg, kK, kV, kChunk, scale, nullptr);
        }
      }
      R4DX_HIP_CHECK(hipEventRecord(e1, nullptr));
      R4DX_HIP_CHECK(hipEventSynchronize(e1));
      float ms = 0;
      R4DX_HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
      if (rep == 1) us[s] = 1000.0 * ms / iters;
    }
  }
  std::printf("\n-- timing (informational), T=%d H=%d Hg=%d, scan kernel alone: shipped %.1f us, span-safe %.1f us (x%.2f)\n", T, H, Hg, us[0],
              us[1], us[1] / us[0]);
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  const r4d::GdnDims gd = r4d::GetGdnDims();
  Check(gd.head_k == kK && gd.head_v == kV && gd.chunk == kChunk, "r4d_gdn_dims is 128 / 128 / 64");
  // mild: test_gdn_chunk_scan.cpp's gate (a 64-row span of a few nats). strong: A = e^A_log in [0.03, 158], dt_bias -8 .. 2.5 (layer 0 of
  // the real model), a ~ U(-3, 3) so a fraction of the tokens decays by e^-100 and below.
  Scenario("mild gate", MakeInputs(41, std::log(0.01f), std::log(0.3f), -0.2f, 0.2f, 0.3f), /*bound_shipped=*/true);
  Scenario("strong gate (the real model's range)", MakeInputs(7, std::log(0.03f), std::log(158.0f), -8.0f, 2.5f, 3.0f), /*bound_shipped=*/false);
  Scenario("very strong gate", MakeInputs(11, std::log(50.0f), std::log(160.0f), -1.0f, 1.0f, 3.0f), /*bound_shipped=*/false);
  Timing();
  std::printf("\n%s (%d failure(s))\n", g_failures == 0 ? "PASS" : "FAIL", g_failures);
  return g_failures == 0 ? 0 : 1;
}
