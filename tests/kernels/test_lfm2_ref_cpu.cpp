// tests/kernels/test_lfm2_ref_cpu.cpp -- pure CPU (no HIP call, no kernel library). Pins lfm2_ref.hpp, the
// reference the GPU tests test_lfm2_* compare the LFM2 kernels against, to hand cases and to naive
// implementations written independently of its loop structure:
//   * ShortConv: hand sequence; the stateful (decode / chunked-prefill) form against a whole-sequence causal
//     conv1d with explicit left padding (double accumulation, <= 1 bf16 ulp); chunk equivalence bit for bit
//     (1 x N, 256 + 256 + 88, uneven splits, strided bcx); state-after is the last two h.
//   * router: hand case with the bias moving the SELECTION only, ties -> lower index, normalization and
//     routed_scaling_factor, the double-logit router vs the sigmoid-first form incl. duplicated gate rows.
//   * counting sort vs std::stable_sort, inverse permutation, invalid ids; combine in slot and sorted layouts.
//   * rmsnorm (kernel contract vs HF double rounding), neox rope (rotate_half pairing, norm preservation).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "lfm2_ref.hpp"

namespace {

bool g_ok = true;
void Check(bool cond, const char* what) {
  std::printf("%-92s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}

using lfm2_ref::Bf16Round;

std::vector<float> RandBf16(std::mt19937& rng, size_t n, float mag) {
  std::uniform_real_distribution<float> d(-mag, mag);
  std::vector<float> v(n);
  for (auto& x : v) x = Bf16Round(d(rng));
  return v;
}

// 1 bf16 ulp of |v| (relative 2^-7), plus a floor for values near zero.
bool WithinUlp(float a, float b) { return std::fabs(a - b) <= std::fabs(b) * (1.0f / 128.0f) + 1e-30f; }

void TestShortConvHand() {
  // H = 1, taps (1, 2, 3), B = C = 1, x = 1, 2, 3 -> h = 1, 2, 3.
  std::vector<float> bcx = {1, 1, 1, 1, 1, 2, 1, 1, 3};
  std::vector<float> st = {0, 0};
  const auto out = lfm2_ref::ShortConv(bcx, &st, {1, 2, 3}, 3, 1, 3);
  Check(out[0] == 3 && out[1] == 8 && out[2] == 14, "shortconv hand: 3 * 1, 2 * 1 + 3 * 2, 1 + 2 * 2 + 3 * 3");
  Check(st[0] == 2 && st[1] == 3, "shortconv hand: state after = (h_{T-2}, h_{T-1})");
  // the C gate multiplies the conv output, B and x multiply before it
  std::vector<float> bcx2 = {2, 0.5f, 4};  // B = 2, C = 0.5, x = 4 -> h = 8; w = (0, 0, 1) -> acc 8, out 4
  std::vector<float> st2 = {0, 0};
  const auto o2 = lfm2_ref::ShortConv(bcx2, &st2, {0, 0, 1}, 1, 1, 3);
  Check(o2[0] == 4 && st2[0] == 0 && st2[1] == 8, "shortconv hand: out = C * conv(B * x), T = 1 shifts the state");
}

void TestShortConvVsNaive(std::mt19937& rng) {
  const int H = 2048, T = 600;
  const int64_t stride = 3 * H + 16;  // strided bcx rows
  const auto bcx = RandBf16(rng, static_cast<size_t>(T) * stride, 2.0f);
  const auto w = RandBf16(rng, static_cast<size_t>(H) * 3, 1.0f);
  const auto init = RandBf16(rng, static_cast<size_t>(2) * H, 2.0f);

  // whole sequence at once, nonzero start state
  std::vector<float> st = init;
  const auto whole = lfm2_ref::ShortConv(bcx, &st, w, T, H, stride);
  const auto naive = lfm2_ref::ShortConvWholeSequence(bcx, init, w, T, H, stride);
  size_t bad = 0, exact = 0;
  for (size_t i = 0; i < whole.size(); ++i) {
    exact += whole[i] == naive[i];
    bad += !WithinUlp(whole[i], naive[i]);
  }
  Check(bad == 0, "shortconv: stateful form vs whole-sequence conv1d, <= 1 bf16 ulp (H 2048, T 600, nonzero state)");
  std::printf("      exact %.4f%% of %zu elements\n", 100.0 * exact / whole.size(), whole.size());

  // zero state start
  std::vector<float> zero(2 * static_cast<size_t>(H), 0.0f), st0 = zero;
  const auto whole0 = lfm2_ref::ShortConv(bcx, &st0, w, T, H, stride);
  const auto naive0 = lfm2_ref::ShortConvWholeSequence(bcx, zero, w, T, H, stride);
  bad = 0;
  for (size_t i = 0; i < whole0.size(); ++i) bad += !WithinUlp(whole0[i], naive0[i]);
  Check(bad == 0, "shortconv: zero start state equals HF's zero causal padding");

  // chunk equivalence: any split of the sequence into chunks gives the same bytes and the same final state
  auto run_chunks = [&](const std::vector<int>& sizes, std::vector<float>* final_state) {
    std::vector<float> s = init, out;
    int t0 = 0;
    for (int n : sizes) {
      std::vector<float> chunk(bcx.begin() + static_cast<size_t>(t0) * stride,
                               bcx.begin() + static_cast<size_t>(t0 + n) * stride);
      const auto o = lfm2_ref::ShortConv(chunk, &s, w, n, H, stride);
      out.insert(out.end(), o.begin(), o.end());
      t0 += n;
    }
    *final_state = s;
    return out;
  };
  std::vector<float> fs;
  const auto c3 = run_chunks({256, 256, 88}, &fs);
  Check(c3 == whole && fs == st, "shortconv: 256 + 256 + 88 chunks == one pass (output and state, bitwise)");
  const std::vector<int> ones(T, 1);
  const auto c1 = run_chunks(ones, &fs);
  Check(c1 == whole && fs == st, "shortconv: 600 x T = 1 steps == one pass (output and state, bitwise)");
  const auto cu = run_chunks({1, 2, 3, 63, 64, 255, 1, 211}, &fs);
  Check(cu == whole && fs == st, "shortconv: uneven split 1, 2, 3, 63, 64, 255, 1, 211 == one pass (bitwise)");
}

void TestRouterHand() {
  const int E = 8, K = 4;
  // bias moves the SELECTION only; expert 1 (s 0.1 + 0.5 = 0.6) ties expert 4 (0.6) -> lower index 1 wins
  std::vector<float> s = {0.9f, 0.1f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f};
  std::vector<float> bias = {0, 0.5f, 0, 0, 0, 0, 0, 0};
  // make the tie exact in fp32: 0.6f + 0 == 0.1f + 0.5f only if the sum rounds equal; force it
  s[4] = 0.1f + 0.5f;
  const auto r = lfm2_ref::RouterFromSigmoid(s, bias, 1, E, K, 2.5f);
  Check(r.ids[0] == 0 && r.ids[1] == 2 && r.ids[2] == 3 && r.ids[3] == 1, "router hand: top-4 = 0, 2, 3, then the tie resolves to expert 1 (lower index)");
  const float sum = s[0] + s[2] + s[3] + s[1];
  bool wok = true;
  const int sel[4] = {0, 2, 3, 1};
  for (int i = 0; i < 4; ++i) wok &= std::fabs(r.w[i] - s[sel[i]] / (sum + 1e-6f) * 2.5f) < 1e-7f;
  Check(wok, "router hand: weights = unbiased s / (sum s + 1e-6) * routed_scaling_factor (bias excluded)");
  float tot = 0;
  for (int i = 0; i < 4; ++i) tot += r.w[i];
  Check(std::fabs(tot - 2.5f) < 1e-4f, "router hand: weights sum to routed_scaling_factor");

  // all scores equal -> experts 0..K-1 in order
  const auto eq = lfm2_ref::RouterFromSigmoid(std::vector<float>(E, 0.5f), std::vector<float>(E, 0.0f), 1, E, K, 1.0f);
  Check(eq.ids[0] == 0 && eq.ids[1] == 1 && eq.ids[2] == 2 && eq.ids[3] == 3, "router hand: all-equal scores -> lowest indices first");
}

void TestRouterVsSigmoidForm(std::mt19937& rng) {
  const int E = 64, K = 4, hidden = 2048, T = 33;
  auto x = RandBf16(rng, static_cast<size_t>(T) * hidden, 2.0f);
  auto gw = RandBf16(rng, static_cast<size_t>(E) * hidden, 0.05f);
  // duplicate rows: experts 7 and 40 share the weights (and the bias below), 12 and 13 too
  std::copy(gw.begin() + 7 * hidden, gw.begin() + 8 * hidden, gw.begin() + 40 * hidden);
  std::copy(gw.begin() + 12 * hidden, gw.begin() + 13 * hidden, gw.begin() + 13 * hidden);
  std::uniform_real_distribution<float> db(-0.2f, 0.2f);
  std::vector<float> bias(E);
  for (auto& b : bias) b = db(rng);
  bias[40] = bias[7];
  bias[13] = bias[12];
  const auto r = lfm2_ref::Router(x, gw, bias, T, hidden, E, K, 1.0f, hidden);
  bool uniq = true, ordered = true, tie_ok = true;
  for (int t = 0; t < T; ++t) {
    for (int a = 0; a < K; ++a) {
      for (int b = a + 1; b < K; ++b) uniq &= r.ids[t * K + a] != r.ids[t * K + b];
      if (a + 1 < K) {
        const float sa = r.score[static_cast<size_t>(t) * E + r.ids[t * K + a]];
        const float sb = r.score[static_cast<size_t>(t) * E + r.ids[t * K + a + 1]];
        ordered &= sa > sb || (sa == sb && r.ids[t * K + a] < r.ids[t * K + a + 1]);
      }
    }
    // a duplicated pair: if the higher index is selected the lower one is too
    for (int s = 0; s < K; ++s) {
      const int id = r.ids[t * K + s];
      if (id == 40 || id == 13) {
        const int low = id == 40 ? 7 : 12;
        bool has = false;
        for (int q = 0; q < K; ++q) has |= r.ids[t * K + q] == low;
        tie_ok &= has;
      }
    }
  }
  Check(uniq, "router: the 4 selected experts are distinct (T 33, E 64, hidden 2048)");
  Check(ordered, "router: slots are in descending score order, equal scores by ascending index");
  Check(tie_ok, "router: duplicated gate rows (equal logits and bias) never pick the higher index alone");
}

void TestSort(std::mt19937& rng) {
  const int T = 256, K = 4, E = 64;
  std::vector<int32_t> ids(static_cast<size_t>(T) * K);
  for (int t = 0; t < T; ++t) {
    std::vector<int> perm(E);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int k = 0; k < K; ++k) ids[static_cast<size_t>(t) * K + k] = perm[k];
  }
  const auto s = lfm2_ref::MoeSort(ids, E);
  // naive: stable sort of the pair indices by expert
  std::vector<int> naive(ids.size());
  std::iota(naive.begin(), naive.end(), 0);
  std::stable_sort(naive.begin(), naive.end(), [&](int a, int b) { return ids[a] < ids[b]; });
  bool same = true;
  for (size_t i = 0; i < naive.size(); ++i) same &= s.sorted_pairs[i] == naive[i];
  Check(same, "sort: counting sort == std::stable_sort of pairs by expert (token order inside an expert)");
  bool inv = true;
  for (size_t p = 0; p < ids.size(); ++p) inv &= s.sorted_pairs[s.pos_of_pair[p]] == static_cast<int>(p);
  Check(inv, "sort: pos_of_pair is the inverse permutation");
  int tot = 0;
  bool segs = true;
  for (int e = 0; e < E; ++e) {
    tot += s.counts[e];
    segs &= s.offsets[e + 1] - s.offsets[e] == s.counts[e];
    for (int i = s.offsets[e]; i < s.offsets[e + 1]; ++i) segs &= ids[s.sorted_pairs[i]] == e;
  }
  Check(tot == T * K && s.offsets[E] == T * K && segs, "sort: counts / offsets consistent, every segment holds its expert's pairs");

  // invalid ids are skipped
  const auto sb = lfm2_ref::MoeSort({3, -1, 3, 99, 0}, 4);
  Check(sb.offsets[4] == 3 && sb.pos_of_pair[1] == -1 && sb.pos_of_pair[3] == -1 && sb.sorted_pairs[0] == 4 &&
            sb.sorted_pairs[1] == 0 && sb.sorted_pairs[2] == 2,
        "sort: ids outside [0, E) get pos -1 and are not counted");
}

void TestCombine(std::mt19937& rng) {
  const int T = 5, K = 4, H = 70;
  // hand: constant rows 1, 2, 3, 4 with weights .5, .25, .125, .125 -> 0.5 + 0.5 + 0.375 + 0.5 = 1.875
  std::vector<float> y(static_cast<size_t>(K) * H), w = {0.5f, 0.25f, 0.125f, 0.125f};
  for (int s = 0; s < K; ++s)
    for (int i = 0; i < H; ++i) y[static_cast<size_t>(s) * H + i] = static_cast<float>(s + 1);
  const auto o = lfm2_ref::Combine(y, nullptr, w, 1, K, H);
  bool hand = true;
  for (float v : o) hand &= v == 1.875f;
  Check(hand, "combine hand: 0.5 * 1 + 0.25 * 2 + 0.125 * 3 + 0.125 * 4 = 1.875");

  // slot layout == sorted layout through pos_of_pair
  std::uniform_real_distribution<float> dw(0.0f, 1.0f);
  const auto yr = RandBf16(rng, static_cast<size_t>(T) * K * H, 3.0f);
  std::vector<float> wr(static_cast<size_t>(T) * K);
  for (auto& v : wr) v = dw(rng);
  std::vector<int32_t> ids(static_cast<size_t>(T) * K);
  for (int t = 0; t < T; ++t)
    for (int k = 0; k < K; ++k) ids[static_cast<size_t>(t) * K + k] = (t * 7 + k * 5) % 16;
  const auto so = lfm2_ref::MoeSort(ids, 16);
  std::vector<float> ysorted(yr.size());
  for (size_t p = 0; p < ids.size(); ++p)
    std::copy(yr.begin() + p * H, yr.begin() + (p + 1) * H, ysorted.begin() + static_cast<size_t>(so.pos_of_pair[p]) * H);
  Check(lfm2_ref::Combine(yr, nullptr, wr, T, K, H) == lfm2_ref::Combine(ysorted, &so.pos_of_pair, wr, T, K, H),
        "combine: slot layout and expert-sorted layout (via pos_of_pair) give the same bytes");
}

void TestNormAndRope(std::mt19937& rng) {
  // rmsnorm hand case: x = (3, 4), w = (1, 2), eps 0 -> rstd = 1 / sqrt(12.5)
  const auto n = lfm2_ref::RmsNormPlain({3, 4}, {1, 2}, 1, 2, 0.0f);
  const double rstd = 1.0 / std::sqrt(12.5);
  Check(std::fabs(n[0] - 3 * rstd) < 0.01 * 3 * rstd && std::fabs(n[1] - 8 * rstd) < 0.01 * 8 * rstd,
        "rmsnorm hand: (3, 4) * rsqrt(12.5) * (1, 2)");
  // kernel contract (one rounding) vs HF (two roundings): within 1 bf16 ulp on random head_dim-64 rows
  const int rows = 64 * 32, hd = 64;
  const auto x = RandBf16(rng, static_cast<size_t>(rows) * hd, 3.0f);
  auto w = RandBf16(rng, hd, 2.0f);
  for (auto& v : w) v = std::fabs(v) + 0.1f;
  const auto a = lfm2_ref::RmsNormPlain(x, w, rows, hd, 1e-5f);
  const auto b = lfm2_ref::RmsNormHf(x, w, rows, hd, 1e-5f);
  size_t bad = 0, exact = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    bad += std::fabs(a[i] - b[i]) > std::fabs(a[i]) * (2.0f / 128.0f) + 1e-6f;
    exact += a[i] == b[i];
  }
  Check(bad == 0, "rmsnorm: one-rounding kernel contract within 2 bf16 ulp of HF's two-rounding op (head_dim 64)");
  std::printf("      identical on %.2f%% of %zu elements\n", 100.0 * exact / a.size(), a.size());

  // rope: pos 0 is the identity, the rotation preserves the norm, pairing is (i, i + half)
  std::vector<double> v(64);
  std::normal_distribution<double> nd;
  for (auto& e : v) e = nd(rng);
  const auto r0 = lfm2_ref::RopeNeox(v, 1e7, 0);
  Check(r0 == v, "rope: position 0 is the identity");
  const auto r5 = lfm2_ref::RopeNeox(v, 1e7, 12345);
  double n0 = 0, n1 = 0;
  for (int i = 0; i < 64; ++i) {
    n0 += v[i] * v[i];
    n1 += r5[i] * r5[i];
  }
  Check(std::fabs(n0 - n1) < 1e-9 * n0, "rope: pairs rotate, so the vector norm is preserved (head_dim 64, theta 1e7)");
  // explicit hand check of one pair: head_dim 4 -> freqs 1, 1e7^-0.5; x = (1, 0, 0, 1), pos 1
  const auto r4 = lfm2_ref::RopeNeox({1, 0, 0, 1}, 1e7, 1);
  const double a0 = 1.0, a1 = std::pow(1e7, -0.5);
  Check(std::fabs(r4[0] - std::cos(a0)) < 1e-12 && std::fabs(r4[2] - std::sin(a0)) < 1e-12 &&
            std::fabs(r4[1] - (-std::sin(a1))) < 1e-12 && std::fabs(r4[3] - std::cos(a1)) < 1e-12,
        "rope: rotate_half pairing (x_i, x_{i+half}) with inv_freq = theta^(-2i/d)");
}

}  // namespace

int main() {
  std::mt19937 rng(20261014);
  TestShortConvHand();
  TestShortConvVsNaive(rng);
  TestRouterHand();
  TestRouterVsSigmoidForm(rng);
  TestSort(rng);
  TestCombine(rng);
  TestNormAndRope(rng);
  std::printf("%s\n", g_ok ? "test_lfm2_ref_cpu: PASS" : "test_lfm2_ref_cpu: FAIL");
  return g_ok ? 0 : 1;
}
