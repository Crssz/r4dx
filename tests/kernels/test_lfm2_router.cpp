// tests/kernels/test_lfm2_router.cpp -- r4dx_lfm2_router_f32 (docs/lfm2-plan.md 3.5, task L1-7) against
// lfm2_ref.hpp's Router (logits and sigmoid in double there, fp32 in the kernel):
//   * LFM geometry (hidden 2048, 64 experts, top-4) at T = 1, 7, 64, 256 with random bf16 data and an fp32
//     bias: ids equal the reference except where the reference's own scores are within 1e-5 of each other
//     (an fp32-vs-double near tie), weights within rel 1e-3 (plan 3.5), and the number of differing tokens
//     is reported (expected 0);
//   * routed_scaling_factor 2.5 and other shapes (hidden 128 / 1000, 8 experts top-2, strided x);
//   * EXACT ties: duplicated gate rows with equal bias (bitwise equal logits on the device) and an all-zero
//     gate with a patterned bias -> the LOWER expert index wins, ids exactly equal to the reference;
//   * the bias moves the selection only: a large-bias expert is selected but its weight is its unbiased
//     sigmoid's share;
//   * determinism (two launches, same bytes); precondition throws.
// GPU test (HIP device 1 via HIP_VISIBLE_DEVICES=1 in CMakeLists.txt); WRITTEN, NOT YET RUN. Built with /EHc-.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "lfm2_test_util.hpp"
#include "r4dx/kernels/lfm2_kernels.h"

using namespace r4dx::core;
using namespace lfm2_test;

namespace {

struct Got {
  std::vector<int32_t> ids;
  std::vector<float> w;
};

Got OnDevice(const std::vector<float>& x, const std::vector<float>& gw, const std::vector<float>& bias, int T,
             int hidden, int E, int K, float scale, int64_t x_stride) {
  DeviceBuffer<uint16_t> x_d(x.size()), g_d(gw.size());
  DeviceBuffer<float> b_d(bias.size()), w_d(static_cast<size_t>(T) * K);
  DeviceBuffer<int32_t> id_d(static_cast<size_t>(T) * K);
  x_d.CopyFromHost(ToBits(x));
  g_d.CopyFromHost(ToBits(gw));
  b_d.CopyFromHost(bias);
  r4dx_lfm2_router_f32(P(x_d.data()), P(g_d.data()), P(b_d.data()), P(id_d.data()), P(w_d.data()), T, hidden, E, K,
                       scale, x_stride, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  return {id_d.CopyToHost(), w_d.CopyToHost()};
}

// Compares a kernel result to the reference: tokens whose ids differ must be fp32-vs-double near ties
// (every differing slot's reference score within 1e-5 of the reference's pick); equal tokens must match in
// weights (rel 1e-3). Returns the number of tokens whose ids differ at all.
int Verify(const std::string& name, const Got& g, const lfm2_ref::RouterOut& r, int T, int E, int K, bool exact_ids) {
  bool ids_ok = true, w_ok = true;
  int differing = 0;
  for (int t = 0; t < T; ++t) {
    bool same = true;
    for (int s = 0; s < K; ++s) same &= g.ids[t * K + s] == r.ids[t * K + s];
    if (same) {
      for (int s = 0; s < K; ++s) {
        const float a = g.w[t * K + s], b = r.w[t * K + s];
        w_ok &= std::fabs(a - b) <= 1e-3f * std::fabs(b) + 1e-9f;
      }
      continue;
    }
    ++differing;
    if (exact_ids) {
      ids_ok = false;
      continue;
    }
    for (int s = 0; s < K; ++s) {
      const int gi = g.ids[t * K + s], ri = r.ids[t * K + s];
      if (gi == ri) continue;
      if (gi < 0 || gi >= E) {
        ids_ok = false;
        continue;
      }
      const float sg = r.score[static_cast<size_t>(t) * E + gi], sr = r.score[static_cast<size_t>(t) * E + ri];
      ids_ok &= std::fabs(sg - sr) < 1e-5f;
    }
  }
  CheckF(ids_ok, name + ": ids == reference" + (exact_ids ? " (exact)" : " (up to fp32 near ties)"));
  CheckF(w_ok, name + ": weights within rel 1e-3 of the reference");
  std::printf("      %d of %d tokens differ in ids\n", differing, T);
  return differing;
}

void TestRandom(std::mt19937& rng, int T, int hidden, int E, int K, float scale, int64_t extra_stride) {
  const int64_t stride = hidden + extra_stride;
  auto x = RandBf16(rng, static_cast<size_t>(T) * stride, 2.0f);
  auto gw = RandBf16(rng, static_cast<size_t>(E) * hidden, 0.05f);
  std::uniform_real_distribution<float> db(-0.2f, 0.2f);
  std::vector<float> bias(E);
  for (auto& b : bias) b = db(rng);
  const auto ref = lfm2_ref::Router(x, gw, bias, T, hidden, E, K, scale, stride);
  const Got g = OnDevice(x, gw, bias, T, hidden, E, K, scale, stride);
  Verify("router T=" + std::to_string(T) + " hidden=" + std::to_string(hidden) + " E=" + std::to_string(E) +
             " K=" + std::to_string(K) + " scale=" + std::to_string(scale).substr(0, 4) + " stride+" +
             std::to_string(extra_stride),
         g, ref, T, E, K, false);
  const Got g2 = OnDevice(x, gw, bias, T, hidden, E, K, scale, stride);
  Check(g2.ids == g.ids && g2.w == g.w, "router: two launches give the same bytes (determinism)");
}

void TestExactTies(std::mt19937& rng) {
  const int T = 16, hidden = 2048, E = 64, K = 4;
  auto x = RandBf16(rng, static_cast<size_t>(T) * hidden, 2.0f);
  auto gw = RandBf16(rng, static_cast<size_t>(E) * hidden, 0.05f);
  std::uniform_real_distribution<float> db(-0.2f, 0.2f);
  std::vector<float> bias(E);
  for (auto& b : bias) b = db(rng);
  // duplicated rows + equal bias: bitwise-equal scores on the device; pairs (7, 40) (12, 13) (0, 63)
  const int pairs[3][2] = {{7, 40}, {12, 13}, {0, 63}};
  for (const auto& pr : pairs) {
    std::copy(gw.begin() + pr[0] * hidden, gw.begin() + (pr[0] + 1) * hidden, gw.begin() + pr[1] * hidden);
    bias[pr[1]] = bias[pr[0]];
  }
  // push the pairs to the top for some tokens so the ties are decisive: large equal bias on 40 and 7, 13 and 12
  bias[40] = bias[7] = 5.0f;
  bias[12] = bias[13] = 4.0f;
  const auto ref = lfm2_ref::Router(x, gw, bias, T, hidden, E, K, 1.0f, hidden);
  const Got g = OnDevice(x, gw, bias, T, hidden, E, K, 1.0f, hidden);
  Verify("router duplicated gate rows + equal bias", g, ref, T, E, K, true);
  bool low_first = true;
  for (int t = 0; t < T; ++t) {
    // with bias 5 / 4 the slots are 7, 40, 12, 13 in that order (descending score, equal -> lower index)
    low_first &= g.ids[t * K] == 7 && g.ids[t * K + 1] == 40 && g.ids[t * K + 2] == 12 && g.ids[t * K + 3] == 13;
  }
  Check(low_first, "router ties: equal scores order by ascending expert index (7, 40, 12, 13)");

  // all-zero gate: s = 0.5 exactly everywhere, bias pattern with repeats -> deterministic lowest-index picks
  std::vector<float> zero_gw(static_cast<size_t>(E) * hidden, 0.0f), bias2(E);
  for (int e = 0; e < E; ++e) bias2[e] = static_cast<float>(e % 5) * 0.25f;  // 0, .25, .5, .75, 1 repeating
  const auto ref2 = lfm2_ref::Router(x, zero_gw, bias2, T, hidden, E, K, 1.0f, hidden);
  const Got g2 = OnDevice(x, zero_gw, bias2, T, hidden, E, K, 1.0f, hidden);
  Verify("router all-zero gate, bias pattern e % 5 (tie among 13 experts)", g2, ref2, T, E, K, true);
  Check(g2.ids[0] == 4 && g2.ids[1] == 9 && g2.ids[2] == 14 && g2.ids[3] == 19, "router ties: the four lowest experts of the top bias class (4, 9, 14, 19)");

  // no bias, zero gate: experts 0..3
  std::vector<float> nobias(E, 0.0f);
  const Got g3 = OnDevice(x, zero_gw, nobias, T, hidden, E, K, 1.0f, hidden);
  bool lowest = true;
  for (int t = 0; t < T; ++t)
    for (int s = 0; s < K; ++s) lowest &= g3.ids[t * K + s] == s && std::fabs(g3.w[t * K + s] - 0.25f) < 1e-5f;
  Check(lowest, "router ties: all scores equal -> experts 0, 1, 2, 3 with weight 1/4 each");
}

void TestBiasSelectionOnly(std::mt19937& rng) {
  const int T = 4, hidden = 256, E = 8, K = 2;
  auto x = RandBf16(rng, static_cast<size_t>(T) * hidden, 1.0f);
  auto gw = RandBf16(rng, static_cast<size_t>(E) * hidden, 0.05f);
  std::vector<float> bias(E, 0.0f);
  bias[5] = 100.0f;  // always selected first
  const auto ref = lfm2_ref::Router(x, gw, bias, T, hidden, E, K, 1.0f, hidden);
  const Got g = OnDevice(x, gw, bias, T, hidden, E, K, 1.0f, hidden);
  bool first5 = true, share = true;
  for (int t = 0; t < T; ++t) {
    first5 &= g.ids[t * K] == 5;
    // the weights are the two unbiased sigmoids' shares: each in (0, 1), summing to 1, NOT dominated by the bias
    share &= std::fabs(g.w[t * K] + g.w[t * K + 1] - 1.0f) < 1e-4f && g.w[t * K] < 0.9f;
  }
  Check(first5, "router: a huge bias forces the selection of that expert");
  Check(share, "router: weights stay the unbiased sigmoids' normalized shares (bias is selection-only)");
  Verify("router bias selection-only case", g, ref, T, E, K, false);
}

void TestThrows() {
  DeviceBuffer<uint16_t> a(64);
  DeviceBuffer<float> f(64);
  DeviceBuffer<int32_t> i(64);
  auto throws = [&](int hidden, int E, int K, int64_t stride) {
    try {
      r4dx_lfm2_router_f32(P(a.data()), P(a.data()), P(f.data()), P(i.data()), P(f.data()), 1, hidden, E, K, 1.0f, stride, 0);
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };
  Check(throws(32, 65, 4, 32), "router: n_experts 65 throws");
  Check(throws(32, 8, 9, 32), "router: top_k 9 throws");
  Check(throws(32, 4, 5, 32), "router: top_k > n_experts throws");
  Check(throws(0, 8, 2, 32), "router: hidden 0 throws");
  Check(throws(32, 8, 2, 16), "router: x_row_stride < hidden throws");
  r4dx_lfm2_router_f32(0, 0, 0, 0, 0, 0, 32, 8, 2, 1.0f, 32, 0);
  Check(true, "router: T = 0 is a no-op");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(77);
  for (int T : {1, 7, 64, 256}) TestRandom(rng, T, 2048, 64, 4, 1.0f, 0);
  TestRandom(rng, 33, 2048, 64, 4, 2.5f, 0);
  TestRandom(rng, 20, 2048, 64, 4, 1.0f, 40);   // strided x rows
  TestRandom(rng, 9, 128, 64, 4, 1.0f, 0);
  TestRandom(rng, 9, 1000, 64, 4, 1.0f, 0);     // hidden not a multiple of 32
  TestRandom(rng, 9, 512, 8, 2, 1.0f, 0);
  TestRandom(rng, 5, 64, 3, 3, 1.0f, 0);        // E < 32: lanes beyond E own no expert
  TestExactTies(rng);
  TestBiasSelectionOnly(rng);
  TestThrows();
  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
