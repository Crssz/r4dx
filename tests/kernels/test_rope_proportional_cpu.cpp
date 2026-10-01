// tests/kernels/test_rope_proportional_cpu.cpp -- pure CPU (no HIP call). Pins the Gemma 4 rope
// contract that r4dx_rope_proportional_bf16 implements (docs/gemma4-plan.md 3.4) against HF's own
// construction (inv_freq with a zero tail, cat(freqs, freqs), rotate_half) in gemma_ref.hpp::HfRope:
//   * the full layer's proportional rope (head_dim 512, 64 rotated pairs (i, i + 256), freq_dim 512,
//     theta 1e6) leaves dims 64..255 and 320..511 untouched and rotates pair i by pos * theta^(-2i/512);
//   * the sliding layer's default rope (256, 128 pairs (i, i + 128), theta 1e4) is the same
//     expression with n_pairs = head_dim / 2 -- the kernel's pair loop at that setting is plain NeoX
//     rope, so r4dx_rope_neox_bf16 is a special case of it;
//   * the kernel's own fp32 angle expression (exp2f / log2f / multiply) stays within 1e-6 relative of
//     the exact angle at the positions the tests use.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "gemma_ref.hpp"

namespace {

bool g_ok = true;
void Check(bool cond, const char* what) {
  std::printf("%-72s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}

// The kernel's loop, on the host: pair (i, i + half), angle = pos * exp2(-(2i / freq_dim) * log2(theta)).
std::vector<double> PairLoop(std::vector<double> x, int n_pairs, int freq_dim, float theta, int pos) {
  const int half = static_cast<int>(x.size()) / 2;
  for (int i = 0; i < n_pairs; ++i) {
    const float inv_freq = std::exp2(-(2.0f * i) / static_cast<float>(freq_dim) * std::log2(theta));
    const double angle = static_cast<double>(static_cast<float>(pos) * inv_freq);
    const double c = std::cos(angle), s = std::sin(angle);
    const double x0 = x[i], x1 = x[i + half];
    x[i] = x0 * c - x1 * s;
    x[i + half] = x1 * c + x0 * s;
  }
  return x;
}

double MaxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) m = std::max(m, std::abs(a[i] - b[i]));
  return m;
}

void CheckGeometry(const char* name, int head_dim, int n_pairs, int freq_dim, float theta) {
  std::mt19937 rng(11);
  std::normal_distribution<double> nd(0.0, 1.0);
  const int half = head_dim / 2;
  for (int pos : {0, 1, 7, 513, 4096}) {
    std::vector<double> x(head_dim);
    for (double& v : x) v = nd(rng);
    const std::vector<double> hf = gemma_ref::HfRope(x, n_pairs, freq_dim, theta, pos, true);
    const std::vector<double> loop = PairLoop(x, n_pairs, freq_dim, theta, pos);
    char msg[160];
    // The two build the angle in fp32 from different (std::pow vs exp2/log2) inv_freq roundings: they
    // agree to well under a bf16 ulp (4e-3 of the value) even at pos 4096.
    std::snprintf(msg, sizeof msg, "%s pos %d: pair loop == HF construction (max |d| %.2e)", name, pos,
                  MaxAbsDiff(hf, loop));
    Check(MaxAbsDiff(hf, loop) < 5e-3, msg);

    // Dims outside the rotated pairs are bit-untouched (a zero frequency: cos 1, sin 0).
    bool tail_same = true;
    for (int j = 0; j < head_dim; ++j) {
      const int i = j % half;
      if (i >= n_pairs && hf[j] != x[j]) tail_same = false;
      if (i >= n_pairs && loop[j] != x[j]) tail_same = false;
    }
    std::snprintf(msg, sizeof msg, "%s pos %d: dims outside the %d rotated pairs pass through", name,
                  pos, n_pairs);
    Check(tail_same, msg);

    // A rotation preserves each pair's norm.
    double worst = 0.0;
    for (int i = 0; i < half; ++i) {
      const double n0 = x[i] * x[i] + x[i + half] * x[i + half];
      const double n1 = loop[i] * loop[i] + loop[i + half] * loop[i + half];
      worst = std::max(worst, std::abs(n0 - n1));
    }
    std::snprintf(msg, sizeof msg, "%s pos %d: pair norms preserved (max |d| %.2e)", name, pos, worst);
    Check(worst < 1e-9, msg);
    if (pos == 0) Check(MaxAbsDiff(hf, x) == 0.0 && MaxAbsDiff(loop, x) == 0.0, "  pos 0 is the identity");
  }
}

}  // namespace

int main() {
  CheckGeometry("full    (512, 64 pairs, freq 512, 1e6)", 512, 64, 512, 1.0e6f);
  CheckGeometry("sliding (256, 128 pairs, freq 256, 1e4)", 256, 128, 256, 1.0e4f);

  // Pair identity: i pairs with i + head_dim/2 -- element 0 of a full-layer head pairs with element
  // 256, and element 64 is not rotated at all.
  {
    std::vector<double> x(512, 0.0);
    x[0] = 1.0;
    const std::vector<double> y = gemma_ref::HfRope(x, 64, 512, 1.0e6, 3, true);
    Check(std::abs(y[0] - std::cos(3.0)) < 1e-12 && std::abs(y[256] - std::sin(3.0)) < 1e-12,
          "full layer: element 0 rotates with element 256 (angle = pos * theta^0)");
    x.assign(512, 0.0);
    x[64] = 1.0;
    x[320] = 2.0;
    const std::vector<double> z = gemma_ref::HfRope(x, 64, 512, 1.0e6, 12345, true);
    Check(z[64] == 1.0 && z[320] == 2.0, "full layer: pair 64 (NoPE) is not rotated");
  }

  // The proportional frequencies divide by freq_dim (512), NOT by the rotated width (128): pair 1 of
  // the full layer has inv_freq = 1e6^(-2/512), not 1e6^(-2/128).
  {
    std::vector<double> x(512, 0.0);
    x[1] = 1.0;
    const double inv = std::pow(1.0e6, -2.0 / 512.0);
    const std::vector<double> y = gemma_ref::HfRope(x, 64, 512, 1.0e6, 100, false);
    Check(std::abs(y[1] - std::cos(100.0 * inv)) < 1e-12, "full layer: exponent denominator is freq_dim");
  }

  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
