// tests/kernels/test_rotation_ref_cpu.cpp -- CPU-only checks of rotation_ref.hpp, the fp64 reference
// the quant2 rotation kernel tests (test_rotate_residual, test_attn_gate_mul_hadamard) compare
// against. Those tests are only as good as this reference, so it is pinned here to the contract's
// definitions by independent routes:
//   1. Fwht == the dense Sylvester definition sum_i v[i] (-1)^popcount(i & j).
//   2. ApplyQ(e_k) == the closed form of row k of Q = D (I_5 (x) H/32) (R (x) I_1024):
//        Q[c*1024 + i][b*1024 + j] = d[c*1024 + i] * (-1)^popcount(i & j) / 32 * R[c][b]
//      (pins the R[c][b] orientation and the sign/transform order, which norm checks cannot).
//   3. Q and Hb preserve norms, Q^T / Hb^T invert them.
//   4. TP slicing: a rank's K-slice of h Hb equals the slice of the full-K result (block-aligned
//      halves never share a Hadamard block), which is what lets a rank use its own sign slice.
// No GPU, no container: always on.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "rotation_ref.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

int Popcount(int64_t v) {
  int c = 0;
  while (v) {
    c += static_cast<int>(v & 1);
    v >>= 1;
  }
  return c;
}

double Norm(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x * x;
  return std::sqrt(s);
}

double MaxAbsDiff(const std::vector<double>& a, const std::vector<double>& b) {
  double m = 0.0;
  for (size_t i = 0; i < a.size(); ++i) m = std::fmax(m, std::fabs(a[i] - b[i]));
  return m;
}

std::vector<double> RandomVec(std::mt19937_64& rng, int64_t n) {
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> v(static_cast<size_t>(n));
  for (auto& x : v) x = nd(rng);
  return v;
}

void CheckFwhtDense(std::mt19937_64& rng) {
  for (int64_t n : {2, 8, 128, 256, 512}) {
    std::vector<double> v = RandomVec(rng, n);
    std::vector<double> fast = v;
    rotation_ref::Fwht(fast.data(), n);
    std::vector<double> dense(static_cast<size_t>(n), 0.0);
    for (int64_t j = 0; j < n; ++j) {
      for (int64_t i = 0; i < n; ++i) dense[j] += (Popcount(i & j) & 1) ? -v[i] : v[i];
    }
    const double err = MaxAbsDiff(fast, dense);
    Check(err < 1e-9 * std::sqrt(static_cast<double>(n)),
          "Fwht(n=" + std::to_string(n) + ") != dense Sylvester definition, max err " +
              std::to_string(err));
  }
}

void CheckQClosedForm(const std::vector<float>& d, const std::vector<float>& R) {
  using rotation_ref::kBlock;
  using rotation_ref::kHidden;
  // Rows spread over all five input blocks, both ends of a block and an interior index.
  for (int64_t k : {int64_t{0}, int64_t{1}, int64_t{1023}, int64_t{1024}, int64_t{2047},
                    int64_t{2 * 1024 + 517}, int64_t{3 * 1024 + 300}, int64_t{5119}}) {
    std::vector<double> e(kHidden, 0.0);
    e[k] = 1.0;
    const std::vector<double> row = rotation_ref::ApplyQ(e, d.data(), R.data(), false);
    const int64_t c = k / kBlock;
    const int64_t i = k % kBlock;
    double err = 0.0;
    for (int b = 0; b < 5; ++b) {
      for (int64_t j = 0; j < kBlock; ++j) {
        const double h = (Popcount(i & j) & 1) ? -1.0 : 1.0;
        const double want = static_cast<double>(d[k]) * h / 32.0 * static_cast<double>(R[c * 5 + b]);
        err = std::fmax(err, std::fabs(row[b * kBlock + j] - want));
      }
    }
    Check(err < 1e-12, "ApplyQ(e_" + std::to_string(k) + ") != closed-form row of Q, max err " +
                           std::to_string(err));
    // And the inverse path takes that row back to e_k (R is fp32-orthogonal, hence 1e-6).
    const std::vector<double> back = rotation_ref::ApplyQ(row, d.data(), R.data(), true);
    Check(MaxAbsDiff(back, e) < 1e-6, "e_" + std::to_string(k) + " Q Q^T != e_k");
  }
}

void CheckNormsAndInverses(std::mt19937_64& rng, const std::vector<float>& d,
                           const std::vector<float>& R) {
  // R is fp32, so it is orthogonal to ~1e-7, not exactly: the tolerances below allow for that.
  for (int t = 0; t < 3; ++t) {
    const std::vector<double> x = RandomVec(rng, rotation_ref::kHidden);
    const std::vector<double> z = rotation_ref::ApplyQ(x, d.data(), R.data(), false);
    const std::vector<double> zt = rotation_ref::ApplyQ(x, d.data(), R.data(), true);
    Check(std::fabs(Norm(z) / Norm(x) - 1.0) < 1e-6, "||x Q|| != ||x||");
    Check(std::fabs(Norm(zt) / Norm(x) - 1.0) < 1e-6, "||x Q^T|| != ||x||");
    Check(MaxAbsDiff(rotation_ref::ApplyQ(z, d.data(), R.data(), true), x) < 1e-5, "x Q Q^T != x");
    Check(MaxAbsDiff(rotation_ref::ApplyQ(zt, d.data(), R.data(), false), x) < 1e-5, "x Q^T Q != x");
  }
  for (int64_t B : {128, 256, 512}) {
    const int64_t K = B * 12;
    const std::vector<float> s = rotation_ref::RandomSigns(rng, K);
    const std::vector<double> h = RandomVec(rng, K);
    const std::vector<double> hb = rotation_ref::ApplyHb(h, s.data(), B);
    Check(std::fabs(Norm(hb) / Norm(h) - 1.0) < 1e-12, "||h Hb|| != ||h||, B=" + std::to_string(B));
    Check(MaxAbsDiff(rotation_ref::ApplyHbT(hb, s.data(), B), h) < 1e-12,
          "h Hb Hb^T != h, B=" + std::to_string(B));
  }
}

void CheckTpSlices(std::mt19937_64& rng) {
  struct Case {
    int64_t K;
    int64_t B;
    const char* name;
  };
  // mlp.down (17408 -> 8704 per rank = 17 x 512), attn.o and gdn.out_proj (6144 -> 3072 per rank).
  for (const Case& c : {Case{17408, 512, "down"}, Case{6144, 256, "o"}, Case{6144, 128, "gdn_out"}}) {
    const std::vector<float> s = rotation_ref::RandomSigns(rng, c.K);
    const std::vector<double> h = RandomVec(rng, c.K);
    const std::vector<double> full = rotation_ref::ApplyHb(h, s.data(), c.B);
    const int64_t half = c.K / 2;
    Check(half % c.B == 0, std::string(c.name) + ": TP=2 rank K is not whole blocks");
    for (int r = 0; r < 2; ++r) {
      std::vector<double> hr(h.begin() + r * half, h.begin() + (r + 1) * half);
      const std::vector<double> part = rotation_ref::ApplyHb(hr, s.data() + r * half, c.B);
      std::vector<double> want(full.begin() + r * half, full.begin() + (r + 1) * half);
      Check(MaxAbsDiff(part, want) == 0.0,
            std::string(c.name) + ": rank " + std::to_string(r) + " slice of h Hb differs");
    }
  }
}

}  // namespace

int main() {
  std::mt19937_64 rng(0x5EED2025ull);
  const std::vector<float> d = rotation_ref::RandomSigns(rng, rotation_ref::kHidden);
  const std::vector<float> R = rotation_ref::RandomOrthogonal5(rng);

  // The reference's own R must be orthogonal (to fp32) and not symmetric (see RandomOrthogonal5).
  double orth = 0.0, asym = 0.0;
  for (int a = 0; a < 5; ++a) {
    for (int b = 0; b < 5; ++b) {
      double dot = 0.0;
      for (int c = 0; c < 5; ++c) dot += static_cast<double>(R[a * 5 + c]) * R[b * 5 + c];
      orth = std::fmax(orth, std::fabs(dot - (a == b ? 1.0 : 0.0)));
      asym = std::fmax(asym, std::fabs(static_cast<double>(R[a * 5 + b]) - R[b * 5 + a]));
    }
  }
  Check(orth < 1e-6, "RandomOrthogonal5 is not orthogonal");
  Check(asym > 1e-3, "RandomOrthogonal5 came out symmetric");

  CheckFwhtDense(rng);
  CheckQClosedForm(d, R);
  CheckNormsAndInverses(rng, d, R);
  CheckTpSlices(rng);

  if (g_failures == 0) {
    std::printf("PASS (%d checks)\n", g_checks);
    return 0;
  }
  std::printf("FAIL (%d of %d checks)\n", g_failures, g_checks);
  return 1;
}
