// r4dx-convert --rotate (src/convert/include/r4dx_convert/rotation.hpp, docs/quant2.md sections 3-4):
// the residual rotation Q, the block Hadamards Hb, the weight folds, and the Hessian / imatrix
// changes of basis. Everything here is CPU-only.
//
//   (a) generation. splitmix64 against its published reference values (seed 0), rotation.signs being
//       exactly the top bits of the stream's first `hidden` draws, mix5 row 0 being the normalized
//       first Box-Muller row, R R^T = I (fp32 values, 1e-6), every sign +-1, the same seed giving the
//       same bytes, another seed different ones, and q2a's signs / mix5 equal to q2ab's.
//   (b) Q. At miniature sizes (5 x 16, 4 x 64) the dense matrix built by pushing every basis row
//       through the fast transform must equal the documented closed form
//       Q[(c,i)][(b,j)] = d[c*B+i] H[i][j] / sqrt(B) R[c][b] (1e-12), be orthogonal (Q Q^T = I,
//       1e-5), and ApplyT must be x Q^T of that same matrix. At the production size (5120 = 5 x 1024,
//       default seed) the closed form is checked on EVERY entry, x Q Q^T = x and ||x Q|| = ||x|| to
//       1e-5 relative on random rows, and x Q^T Q = x.
//   (c) Hb. Miniature (K = 96, B = 32): dense vs closed form s[m] H[i][j] / sqrt(B) inside the block
//       and 0 outside, Hb Hb^T = I, ApplyT = x Hb^T. Production (down 17408/512, o 6144/256,
//       gdn_out 6144/128): round trip, norm preservation, and sampled rows against the closed form
//       with exact block support.
//   (d) fold exactness on a random miniature layer (hidden 5 x 16, and the production hidden with
//       small N / K): for residual x, zero-centred norm weight w, W_in, W_out,
//         rms-normed(x) (1 + w) W_in^T  ==  rms-normed(x Q) W_in'^T            (norm stored as 0)
//         r + o W_out^T                 ==  (r Q + o W_out'^T) Q^T
//       and, for the q2ab sites (down / o / gdn_out, miniature blocks),
//         r + h W^T                     ==  (r Q + (h Hb) W''^T) Q^T,   W'' = (Q^T W) Hb
//       to 1e-4 relative. Also: the fold of a row-concatenated matrix (mlp.gate_up) is the
//       concatenation of the folds, and all folds are bit-identical at 1 and 7 threads.
//   (e) Hessians. TransformHessianQ(H, w) must equal the Hessian captured DIRECTLY on the folded
//       linear's input x_n' = (x / rms) Q (1e-4), and the dense Q^T D^-1 H D^-1 Q; it must be
//       exactly symmetric, and tr(W' H' W'^T) = tr(W H W^T), and the same for a random error matrix
//       mapped the same way (the LDLQ proxy of the same error is basis-free). Likewise
//       TransformHessianHadamard vs a direct capture on h Hb and the trace identity for W'' = W Hb.
//       The in-projection case is repeated at the production hidden (K = 5120) against a direct
//       capture on a rank-48 input. |1 + w| < 1e-3 must throw.
//   (f) HessianStore's transform hook on fixtures/hess_small (K = 256, t.b and t.c share bc.hess):
//       the transform runs once per (file, damp, id); t.b then t.c under the same id share the
//       cached factor; a different id or no transform re-factors; the factor equals FactorHessian of
//       the transformed ReadHessFile bit for bit; an empty id throws.
//   (g) imatrix under the diagonal model: TransformImportanceQ / TransformImportanceHadamard equal
//       diag(M^T diag(v) M) of the dense M = D^-1 Q / Hb (1e-6 relative).
//
// What this file does NOT cover, and what does: that `r4dx-convert` WITHOUT --rotate writes the
// same bytes as before the flag existed is a property of main.cpp's call sites (every new path is
// gated on RotationSource::Enabled(), and no header key is added when it is off) -- checked by
// converting the real checkpoint (--layers 4) with the pre-rotation binary and this one and
// comparing the files byte for byte; see docs/container-format.md "Residual rotation". The
// weightless rms Hessian path for in-projections (TransformHessianRmsQ, hessian.json "rms_keys",
// dead norm channels) and the exe end to end on a synthetic checkpoint are test_rms_hessian.cpp's.
//
// R4DX_CONVERT_FIXTURES_DIR is injected by tests/convert/CMakeLists.txt as an absolute path.
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx_convert/hessian_store.hpp"
#include "r4dx_convert/quant_ldlq.hpp"
#include "r4dx_convert/rotation.hpp"
#include "r4dx_convert/threadpool.hpp"

namespace {

using namespace r4dx_convert;
namespace fs = std::filesystem;

int g_failures = 0;

std::string Fmt(const char* fmt, ...) {
  char buf[1024];
  va_list ap;
  va_start(ap, fmt);
  std::vsnprintf(buf, sizeof(buf), fmt, ap);
  va_end(ap);
  return buf;
}

bool Gate(bool cond, const std::string& label) {
  if (cond) {
    std::printf("OK   %s\n", label.c_str());
  } else {
    std::fprintf(stderr, "FAIL %s\n", label.c_str());
    ++g_failures;
  }
  return cond;
}

template <typename F>
void ExpectThrow(const std::string& label, F&& f) {
  try {
    f();
  } catch (const std::exception& e) {
    Gate(true, label + " (threw: " + e.what() + ")");
    return;
  }
  Gate(false, label + " (did not throw)");
}

// max |a - b| / max |b|.
template <typename A, typename B>
double RelErr(const A& a, const B& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    num = std::max(num, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    den = std::max(den, std::fabs(static_cast<double>(b[i])));
  }
  return den > 0.0 ? num / den : num;
}

std::vector<double> RandomVec(std::mt19937_64& rng, int64_t n, double sigma = 1.0) {
  std::normal_distribution<double> nd(0.0, sigma);
  std::vector<double> v(static_cast<size_t>(n));
  for (auto& x : v) x = nd(rng);
  return v;
}

std::vector<float> RandomVecF(std::mt19937_64& rng, int64_t n, double sigma = 1.0) {
  std::normal_distribution<double> nd(0.0, sigma);
  std::vector<float> v(static_cast<size_t>(n));
  for (auto& x : v) x = static_cast<float>(nd(rng));
  return v;
}

int Popcount(uint64_t v) {
  int c = 0;
  for (; v; v &= v - 1) ++c;
  return c;
}
double Had(int64_t i, int64_t j) { return (Popcount(static_cast<uint64_t>(i & j)) & 1) ? -1.0 : 1.0; }

// Dense n x n matrix whose row m is op(e_m): for x -> x M, that is exactly M.
template <typename Op>
std::vector<double> DenseFromRowOp(int64_t n, const Op& op) {
  std::vector<double> M(static_cast<size_t>(n * n), 0.0), tmp(static_cast<size_t>(n));
  for (int64_t m = 0; m < n; ++m) {
    double* row = M.data() + m * n;
    row[m] = 1.0;
    op(row, tmp.data());
  }
  return M;
}

// y = x M (x length n, M n x n).
std::vector<double> RowTimes(const std::vector<double>& x, const std::vector<double>& M, int64_t n) {
  std::vector<double> y(static_cast<size_t>(n), 0.0);
  for (int64_t m = 0; m < n; ++m)
    for (int64_t k = 0; k < n; ++k) y[static_cast<size_t>(k)] += x[static_cast<size_t>(m)] * M[static_cast<size_t>(m * n + k)];
  return y;
}

// max |M M^T - I|.
double OrthoErr(const std::vector<double>& M, int64_t n) {
  double e = 0.0;
  for (int64_t a = 0; a < n; ++a)
    for (int64_t b = 0; b < n; ++b) {
      double s = 0.0;
      for (int64_t k = 0; k < n; ++k) s += M[static_cast<size_t>(a * n + k)] * M[static_cast<size_t>(b * n + k)];
      e = std::max(e, std::fabs(s - (a == b ? 1.0 : 0.0)));
    }
  return e;
}

double Norm(const std::vector<double>& v) {
  double s = 0.0;
  for (double x : v) s += x * x;
  return std::sqrt(s);
}

double QClosedForm(const ResidualRotation& q, int64_t m, int64_t k) {
  const int64_t c = m / q.block, i = m % q.block, b = k / q.block, j = k % q.block;
  return q.signs[static_cast<size_t>(m)] * Had(i, j) / std::sqrt(static_cast<double>(q.block)) *
         q.mix[static_cast<size_t>(c * q.nblk + b)];
}

double HbClosedForm(const BlockHadamard& hb, int64_t m, int64_t k) {
  if (m / hb.block != k / hb.block) return 0.0;
  return hb.signs[static_cast<size_t>(m)] * Had(m % hb.block, k % hb.block) /
         std::sqrt(static_cast<double>(hb.block));
}

RotationShape ProductionShape() {
  RotationShape s;
  s.hidden = 5120;
  s.k_down = 17408;
  s.k_o = 6144;
  s.k_gdn_out = 6144;
  return s;
}

// ---- (a) generation ---------------------------------------------------------------------------

void TestGeneration() {
  std::printf("---- (a) generation ----\n");
  {
    uint64_t s = 0;
    const uint64_t a = rotation_detail::SplitMix64(s), b = rotation_detail::SplitMix64(s),
                   c = rotation_detail::SplitMix64(s);
    Gate(a == 0xE220A8397B1DCDAFull && b == 0x6E789E6AA1B965F4ull && c == 0x06C45D188009454Full,
         "(a) splitmix64(seed 0) = e220a8397b1dcdaf, 6e789e6aa1b965f4, 06c45d188009454f");
  }
  const RotationSet ab = GenerateRotationSet(RotationKind::kQ2ab, kDefaultRotationSeed, ProductionShape());
  const RotationSet a = GenerateRotationSet(RotationKind::kQ2a, kDefaultRotationSeed, ProductionShape());
  const RotationSet ab2 = GenerateRotationSet(RotationKind::kQ2ab, kDefaultRotationSeed, ProductionShape());
  const RotationSet other = GenerateRotationSet(RotationKind::kQ2ab, kDefaultRotationSeed + 1, ProductionShape());

  Gate(ab.q.hidden == 5120 && ab.q.block == 1024 && ab.q.nblk == 5 && ab.q.signs.size() == 5120 &&
           ab.q.mix.size() == 25 && ab.had_down.K == 17408 && ab.had_down.block == 512 &&
           ab.had_o.K == 6144 && ab.had_o.block == 256 && ab.had_gdn_out.K == 6144 &&
           ab.had_gdn_out.block == 128,
       "(a) production shapes: signs[5120], mix5[5x5], had_down[17408]/512, had_o[6144]/256, "
       "had_gdn_out[6144]/128");
  Gate(a.had_down.Empty() && a.had_o.Empty() && a.had_gdn_out.Empty(), "(a) q2a has no Hadamards");

  // signs[i] = top bit of draw i.
  {
    uint64_t s = kDefaultRotationSeed;
    bool ok = true;
    for (int64_t i = 0; i < 5120; ++i) {
      const uint64_t d = rotation_detail::SplitMix64(s);
      ok = ok && ab.q.signs[static_cast<size_t>(i)] == ((d >> 63) ? -1.0f : 1.0f);
    }
    // Draws 5120/5121 are mix row 0's first Box-Muller pair: row 0 is just the normalized normals.
    const uint64_t d0 = rotation_detail::SplitMix64(s), d1 = rotation_detail::SplitMix64(s);
    const uint64_t d2 = rotation_detail::SplitMix64(s), d3 = rotation_detail::SplitMix64(s);
    const uint64_t d4 = rotation_detail::SplitMix64(s), d5 = rotation_detail::SplitMix64(s);
    auto bm = [](uint64_t x, uint64_t y, double* g0, double* g1) {
      const double u1 = (static_cast<double>(x >> 11) + 1.0) / 9007199254740992.0;
      const double u2 = static_cast<double>(y >> 11) / 9007199254740992.0;
      const double r = std::sqrt(-2.0 * std::log(u1));
      *g0 = r * std::cos(6.283185307179586 * u2);
      *g1 = r * std::sin(6.283185307179586 * u2);
    };
    double g[6];
    bm(d0, d1, &g[0], &g[1]);
    bm(d2, d3, &g[2], &g[3]);
    bm(d4, d5, &g[4], &g[5]);
    double n = 0.0;
    for (int j = 0; j < 5; ++j) n += g[j] * g[j];
    n = std::sqrt(n);
    double e = 0.0;
    for (int j = 0; j < 5; ++j) e = std::max(e, std::fabs(ab.q.mix[static_cast<size_t>(j)] - g[j] / n));
    Gate(ok, "(a) rotation.signs[i] = top bit of the stream's draw i, for all 5120");
    Gate(e < 1e-7, Fmt("(a) mix5 row 0 = normalized first 5 Box-Muller normals (max err %.2e)", e));
  }

  double rr = 0.0;
  for (int c = 0; c < 5; ++c)
    for (int d = 0; d < 5; ++d) {
      double s = 0.0;
      for (int b = 0; b < 5; ++b)
        s += static_cast<double>(ab.q.mix[static_cast<size_t>(c * 5 + b)]) * ab.q.mix[static_cast<size_t>(d * 5 + b)];
      rr = std::max(rr, std::fabs(s - (c == d ? 1.0 : 0.0)));
    }
  Gate(rr < 1e-6, Fmt("(a) R R^T = I with R the stored fp32 mix5 (max err %.2e)", rr));

  auto all_pm1 = [](const std::vector<float>& v) {
    int64_t neg = 0;
    for (float x : v) {
      if (x != 1.0f && x != -1.0f) return false;
      neg += x < 0;
    }
    // a fair coin: 5120 draws put this at 2560 +- ~36; 8 sigma is a generation bug, not luck
    const double mean = static_cast<double>(v.size()) / 2;
    return std::fabs(neg - mean) < 8 * std::sqrt(mean / 2);
  };
  Gate(all_pm1(ab.q.signs) && all_pm1(ab.had_down.signs) && all_pm1(ab.had_o.signs) &&
           all_pm1(ab.had_gdn_out.signs),
       "(a) every sign vector is +-1 with a plausible balance");
  Gate(ab.q.signs == ab2.q.signs && ab.q.mix == ab2.q.mix && ab.had_down.signs == ab2.had_down.signs &&
           ab.had_o.signs == ab2.had_o.signs && ab.had_gdn_out.signs == ab2.had_gdn_out.signs,
       "(a) same seed -> identical tensors");
  Gate(ab.q.signs != other.q.signs && ab.q.mix != other.q.mix && ab.had_down.signs != other.had_down.signs,
       "(a) seed + 1 -> different tensors");
  Gate(a.q.signs == ab.q.signs && a.q.mix == ab.q.mix, "(a) q2a's signs / mix5 equal q2ab's");
  Gate(ab.had_o.signs != ab.had_gdn_out.signs, "(a) had_o and had_gdn_out are independent draws");
  ExpectThrow("(a) GenerateResidualRotation rejects a non-power-of-two block", []() {
    uint64_t s = 1;
    GenerateResidualRotation(s, 5 * 96, 96);
  });
  ExpectThrow("(a) GenerateBlockHadamard rejects K not a multiple of the block", []() {
    uint64_t s = 1;
    GenerateBlockHadamard(s, 100, 32);
  });
  ExpectThrow("(a) ParseRotationKind rejects 'q2b'", []() { ParseRotationKind("q2b"); });
}

// ---- (b) Q -------------------------------------------------------------------------------------

void TestQMiniature(int64_t nblk, int64_t block, uint64_t seed) {
  uint64_t st = seed;
  const ResidualRotation q = GenerateResidualRotation(st, nblk * block, block);
  const int64_t n = q.hidden;
  const std::vector<double> M = DenseFromRowOp(n, [&](double* x, double* t) { q.Apply(x, t); });
  double cf = 0.0;
  for (int64_t m = 0; m < n; ++m)
    for (int64_t k = 0; k < n; ++k)
      cf = std::max(cf, std::fabs(M[static_cast<size_t>(m * n + k)] - QClosedForm(q, m, k)));
  Gate(cf < 1e-12, Fmt("(b) %lldx%lld: fast x Q == closed form d H R / sqrt(B), every entry (%.2e)",
                       (long long)nblk, (long long)block, cf));
  const double oe = OrthoErr(M, n);
  Gate(oe < 1e-5, Fmt("(b) %lldx%lld: Q Q^T = I (max err %.2e)", (long long)nblk, (long long)block, oe));

  std::mt19937_64 rng(seed);
  std::vector<double> tmp(static_cast<size_t>(n));
  double fe = 0.0, te = 0.0, rt = 0.0;
  for (int rep = 0; rep < 4; ++rep) {
    const std::vector<double> x = RandomVec(rng, n);
    std::vector<double> y = x;
    q.Apply(y.data(), tmp.data());
    fe = std::max(fe, RelErr(y, RowTimes(x, M, n)));
    // x Q^T via the dense transpose
    std::vector<double> Mt(M.size());
    for (int64_t a = 0; a < n; ++a)
      for (int64_t b = 0; b < n; ++b) Mt[static_cast<size_t>(b * n + a)] = M[static_cast<size_t>(a * n + b)];
    std::vector<double> z = x;
    q.ApplyT(z.data(), tmp.data());
    te = std::max(te, RelErr(z, RowTimes(x, Mt, n)));
    q.ApplyT(y.data(), tmp.data());
    rt = std::max(rt, RelErr(y, x));
  }
  Gate(fe < 1e-12 && te < 1e-5 && rt < 1e-5,
       Fmt("(b) %lldx%lld: Apply = x Q (%.2e), ApplyT = x Q^T (%.2e), x Q Q^T = x (%.2e)",
           (long long)nblk, (long long)block, fe, te, rt));
}

void TestQProduction(const RotationSet& rs) {
  const ResidualRotation& q = rs.q;
  const int64_t n = q.hidden;
  // Every entry against the closed form, one basis row at a time (26M entries).
  std::atomic<int64_t> bad{0};
  std::atomic<uint64_t> worst_bits{0};
  ParallelFor(0, n, 16, [&](int64_t m0, int64_t m1) {
    std::vector<double> row(static_cast<size_t>(n)), tmp(static_cast<size_t>(n));
    double worst = 0.0;
    for (int64_t m = m0; m < m1; ++m) {
      std::fill(row.begin(), row.end(), 0.0);
      row[static_cast<size_t>(m)] = 1.0;
      q.Apply(row.data(), tmp.data());
      for (int64_t k = 0; k < n; ++k) worst = std::max(worst, std::fabs(row[static_cast<size_t>(k)] - QClosedForm(q, m, k)));
    }
    uint64_t bits;
    std::memcpy(&bits, &worst, 8);
    uint64_t cur = worst_bits.load();
    while (bits > cur && !worst_bits.compare_exchange_weak(cur, bits)) {
    }
    if (worst > 1e-12) ++bad;
  });
  double worst;
  const uint64_t wb = worst_bits.load();
  std::memcpy(&worst, &wb, 8);
  Gate(bad.load() == 0, Fmt("(b) 5120: fast x Q == closed form on all 26,214,400 entries (%.2e)", worst));

  std::mt19937_64 rng(7);
  std::vector<double> tmp(static_cast<size_t>(n));
  double rt = 0.0, rt2 = 0.0, nm = 0.0;
  for (int rep = 0; rep < 8; ++rep) {
    std::vector<double> x = RandomVec(rng, n);
    x[static_cast<size_t>(rep * 97)] = 300.0;  // an outlier channel, as real residuals have
    std::vector<double> y = x;
    q.Apply(y.data(), tmp.data());
    nm = std::max(nm, std::fabs(Norm(y) - Norm(x)) / Norm(x));
    q.ApplyT(y.data(), tmp.data());
    rt = std::max(rt, RelErr(y, x));
    std::vector<double> z = x;
    q.ApplyT(z.data(), tmp.data());
    q.Apply(z.data(), tmp.data());
    rt2 = std::max(rt2, RelErr(z, x));
  }
  Gate(rt < 1e-5 && rt2 < 1e-5 && nm < 1e-5,
       Fmt("(b) 5120: x Q Q^T = x (%.2e), x Q^T Q = x (%.2e), ||x Q|| = ||x|| (%.2e)", rt, rt2, nm));
}

// ---- (c) Hb ------------------------------------------------------------------------------------

void TestHbMiniature() {
  uint64_t st = 99;
  const BlockHadamard hb = GenerateBlockHadamard(st, 96, 32);
  const int64_t n = hb.K;
  const std::vector<double> M = DenseFromRowOp(n, [&](double* x, double*) { hb.Apply(x); });
  double cf = 0.0;
  for (int64_t m = 0; m < n; ++m)
    for (int64_t k = 0; k < n; ++k)
      cf = std::max(cf, std::fabs(M[static_cast<size_t>(m * n + k)] - HbClosedForm(hb, m, k)));
  Gate(cf < 1e-12, Fmt("(c) K=96 B=32: fast h Hb == closed form s H / sqrt(B), 0 off-block (%.2e)", cf));
  const double oe = OrthoErr(M, n);
  Gate(oe < 1e-12, Fmt("(c) K=96 B=32: Hb Hb^T = I (%.2e)", oe));
  std::mt19937_64 rng(3);
  std::vector<double> Mt(M.size());
  for (int64_t a = 0; a < n; ++a)
    for (int64_t b = 0; b < n; ++b) Mt[static_cast<size_t>(b * n + a)] = M[static_cast<size_t>(a * n + b)];
  const std::vector<double> x = RandomVec(rng, n);
  std::vector<double> z = x;
  hb.ApplyT(z.data());
  const double te = RelErr(z, RowTimes(x, Mt, n));
  hb.Apply(z.data());
  const double rt = RelErr(z, x);
  Gate(te < 1e-12 && rt < 1e-12, Fmt("(c) K=96 B=32: ApplyT = x Hb^T (%.2e), round trip (%.2e)", te, rt));
}

void TestHbProduction(const BlockHadamard& hb, const char* name) {
  const int64_t n = hb.K;
  std::mt19937_64 rng(11);
  double rt = 0.0, nm = 0.0, cf = 0.0;
  for (int rep = 0; rep < 4; ++rep) {
    std::vector<double> x = RandomVec(rng, n);
    x[static_cast<size_t>(rep * 1001)] = 500.0;
    std::vector<double> y = x;
    hb.Apply(y.data());
    nm = std::max(nm, std::fabs(Norm(y) - Norm(x)) / Norm(x));
    hb.ApplyT(y.data());
    rt = std::max(rt, RelErr(y, x));
  }
  bool support = true;
  for (int64_t m : {int64_t{0}, int64_t{1}, hb.block - 1, hb.block, n / 2 + 3, n - 1}) {
    std::vector<double> row(static_cast<size_t>(n), 0.0);
    row[static_cast<size_t>(m)] = 1.0;
    hb.Apply(row.data());
    for (int64_t k = 0; k < n; ++k) {
      const double want = HbClosedForm(hb, m, k);
      if (want == 0.0 && row[static_cast<size_t>(k)] != 0.0) support = false;
      cf = std::max(cf, std::fabs(row[static_cast<size_t>(k)] - want));
    }
  }
  Gate(rt < 1e-5 && nm < 1e-5 && support && cf < 1e-12,
       Fmt("(c) %s (K=%lld, B=%lld): round trip %.2e, norm %.2e, sampled rows = closed form %.2e "
           "with exact block support",
           name, (long long)n, (long long)hb.block, rt, nm, cf));
}

// ---- (d) fold exactness ------------------------------------------------------------------------

// y[t][n] = sum_k (x[t][k] / rms(x_t)) * g[k] * W[n][k]; g == nullptr means 1.
std::vector<double> NormedLinear(const std::vector<double>& x, int64_t T, int64_t K, const float* w_norm,
                                 const std::vector<float>& W, int64_t N) {
  std::vector<double> y(static_cast<size_t>(T * N), 0.0);
  for (int64_t t = 0; t < T; ++t) {
    const double* xt = x.data() + t * K;
    double ss = 0.0;
    for (int64_t k = 0; k < K; ++k) ss += xt[k] * xt[k];
    const double inv = 1.0 / std::sqrt(ss / static_cast<double>(K));
    for (int64_t n = 0; n < N; ++n) {
      double s = 0.0;
      for (int64_t k = 0; k < K; ++k) {
        const double g = w_norm ? 1.0 + static_cast<double>(w_norm[k]) : 1.0;
        s += xt[k] * inv * g * W[static_cast<size_t>(n * K + k)];
      }
      y[static_cast<size_t>(t * N + n)] = s;
    }
  }
  return y;
}

// z = r + o W^T, W [N, K].
std::vector<double> ResidualAdd(const std::vector<double>& r, const std::vector<double>& o,
                                const std::vector<float>& W, int64_t N, int64_t K) {
  std::vector<double> z = r;
  for (int64_t n = 0; n < N; ++n) {
    double s = 0.0;
    for (int64_t k = 0; k < K; ++k) s += o[static_cast<size_t>(k)] * W[static_cast<size_t>(n * K + k)];
    z[static_cast<size_t>(n)] += s;
  }
  return z;
}

void TestFoldIn(const ResidualRotation& q, int64_t N, const char* label) {
  const int64_t K = q.hidden, T = 3;
  std::mt19937_64 rng(21 + K);
  std::vector<double> x = RandomVec(rng, T * K);
  x[5] = 80.0;  // outlier
  const std::vector<float> w = RandomVecF(rng, K, 0.3);
  const std::vector<float> W = RandomVecF(rng, N * K, 0.05);
  const std::vector<double> y = NormedLinear(x, T, K, w.data(), W, N);

  std::vector<float> Wf = W;
  FoldRowsQ(Wf, N, K, w.data(), q, 4);
  std::vector<double> xr = x, tmp(static_cast<size_t>(K));
  for (int64_t t = 0; t < T; ++t) q.Apply(xr.data() + t * K, tmp.data());
  const std::vector<double> yr = NormedLinear(xr, T, K, /*stored norm = 0*/ nullptr, Wf, N);
  const double e = RelErr(yr, y);
  Gate(e < 1e-4, Fmt("(d) %s in-projection: rms(x)(1+w) W^T == rms(xQ) W'^T, W' = W diag(1+w) Q (%.2e)",
                     label, e));

  // row fusion (mlp.gate_up = [gate; up]) and thread-count independence
  const int64_t Na = N / 2;
  std::vector<float> A(W.begin(), W.begin() + Na * K), B(W.begin() + Na * K, W.end());
  FoldRowsQ(A, Na, K, w.data(), q, 1);
  FoldRowsQ(B, N - Na, K, w.data(), q, 7);
  std::vector<float> cat = A;
  cat.insert(cat.end(), B.begin(), B.end());
  Gate(cat == Wf, Fmt("(d) %s: fold of a row-concatenation == concatenated folds, 1 vs 4 vs 7 threads "
                      "bit-identical", label));
}

void TestFoldOut(const ResidualRotation& q, const BlockHadamard* hb, int64_t K, const char* label) {
  const int64_t N = q.hidden;
  std::mt19937_64 rng(33 + K);
  const std::vector<double> r = RandomVec(rng, N, 2.0);
  std::vector<double> o = RandomVec(rng, K);
  o[1] = 40.0;
  const std::vector<float> W = RandomVecF(rng, N * K, 0.05);
  const std::vector<double> z = ResidualAdd(r, o, W, N, K);

  std::vector<float> Wf = W;
  FoldColumnsQt(Wf, N, K, q, 4);
  std::vector<float> W1 = W;
  FoldColumnsQt(W1, N, K, q, 1);
  bool threads_same = (W1 == Wf);
  if (hb) {
    FoldRowsHadamard(Wf, N, K, *hb, 4);
    FoldRowsHadamard(W1, N, K, *hb, 7);
    threads_same = threads_same && (W1 == Wf);
  }
  std::vector<double> rq = r, tmp(static_cast<size_t>(N));
  q.Apply(rq.data(), tmp.data());
  std::vector<double> oin = o;
  if (hb) hb->Apply(oin.data());  // the runtime's online h -> h Hb
  std::vector<double> zr = ResidualAdd(rq, oin, Wf, N, K);
  q.ApplyT(zr.data(), tmp.data());  // stack exit
  const double e = RelErr(zr, z);
  Gate(e < 1e-4 && threads_same,
       Fmt("(d) %s: r + o W^T == (rQ + %s W'^T) Q^T, W' = %s (%.2e; 1 vs 4 vs 7 threads bit-identical)",
           label, hb ? "(o Hb)" : "o", hb ? "(Q^T W) Hb" : "Q^T W", e));
}

void TestFolds() {
  std::printf("---- (d) fold exactness ----\n");
  uint64_t st = 0xABCDEF;
  const ResidualRotation qm = GenerateResidualRotation(st, 5 * 16, 16);
  const BlockHadamard down = GenerateBlockHadamard(st, 4 * 32, 32);  // "intermediate" 128, B 32
  const BlockHadamard o = GenerateBlockHadamard(st, 3 * 16, 16);      // 3 heads x 16
  const BlockHadamard gdn = GenerateBlockHadamard(st, 4 * 8, 8);      // 4 heads x 8
  TestFoldIn(qm, 12, "mini 5x16");
  TestFoldOut(qm, nullptr, 48, "mini 5x16 q2a out-projection");
  TestFoldOut(qm, &down, down.K, "mini 5x16 q2ab mlp.down");
  TestFoldOut(qm, &o, o.K, "mini 5x16 q2ab attn.o");
  TestFoldOut(qm, &gdn, gdn.K, "mini 5x16 q2ab gdn.out_proj");

  const RotationSet rs = GenerateRotationSet(RotationKind::kQ2a, kDefaultRotationSeed, ProductionShape());
  TestFoldIn(rs.q, 16, "5120");
  TestFoldOut(rs.q, nullptr, 40, "5120 q2a out-projection");

  ExpectThrow("(d) FoldRowsQ rejects K != hidden", [&]() {
    std::vector<float> W(4 * 64);
    FoldRowsQ(W, 4, 64, nullptr, qm, 1);
  });
  ExpectThrow("(d) FoldColumnsQt rejects N != hidden", [&]() {
    std::vector<float> W(64 * 4);
    FoldColumnsQt(W, 64, 4, qm, 1);
  });
  ExpectThrow("(d) FoldRowsHadamard rejects K != Hb's K", [&]() {
    std::vector<float> W(2 * 64);
    FoldRowsHadamard(W, 2, 64, down, 1);
  });
}

// ---- (e) Hessians ------------------------------------------------------------------------------

// H = X^T X / rows, fp64 accumulate, fp32 out (hessian_capture.py's convention).
std::vector<float> Capture(const std::vector<double>& X, int64_t rows, int64_t K, int threads) {
  std::vector<float> H(static_cast<size_t>(K * K));
  ParallelFor(0, K, threads, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i)
      for (int64_t j = 0; j < K; ++j) {
        double s = 0.0;
        for (int64_t r = 0; r < rows; ++r) s += X[static_cast<size_t>(r * K + i)] * X[static_cast<size_t>(r * K + j)];
        H[static_cast<size_t>(i * K + j)] = static_cast<float>(s / static_cast<double>(rows));
      }
  });
  return H;
}

// tr(W H W^T), W [N, K].
double Proxy(const std::vector<float>& W, const std::vector<float>& H, int64_t N, int64_t K, int threads) {
  std::vector<double> part(static_cast<size_t>(N), 0.0);
  ParallelFor(0, N, threads, [&](int64_t n0, int64_t n1) {
    std::vector<double> hw(static_cast<size_t>(K));
    for (int64_t n = n0; n < n1; ++n) {
      const float* w = W.data() + n * K;
      double s = 0.0;
      for (int64_t i = 0; i < K; ++i) {
        double t = 0.0;
        const float* h = H.data() + i * K;
        for (int64_t j = 0; j < K; ++j) t += static_cast<double>(h[j]) * w[j];
        s += static_cast<double>(w[i]) * t;
      }
      part[static_cast<size_t>(n)] = s;
    }
  });
  double s = 0.0;
  for (double p : part) s += p;
  return s;
}

bool ExactlySymmetric(const std::vector<float>& H, int64_t K) {
  for (int64_t i = 0; i < K; ++i)
    for (int64_t j = i + 1; j < K; ++j)
      if (H[static_cast<size_t>(i * K + j)] != H[static_cast<size_t>(j * K + i)]) return false;
  return true;
}

// Correlated inputs: AR(1) along channels plus heavy channels, like the LDLQ fixture.
std::vector<double> CorrelatedRows(std::mt19937_64& rng, int64_t rows, int64_t K) {
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> X(static_cast<size_t>(rows * K));
  for (int64_t r = 0; r < rows; ++r) {
    double prev = nd(rng);
    for (int64_t k = 0; k < K; ++k) {
      prev = 0.8 * prev + 0.6 * nd(rng);
      X[static_cast<size_t>(r * K + k)] = prev * ((k % 29 == 3) ? 12.0 : 1.0);
    }
  }
  return X;
}

void TestHessianIn(const ResidualRotation& q, int64_t rows, int64_t N, const char* label, int threads) {
  const int64_t K = q.hidden;
  std::mt19937_64 rng(41 + K);
  const std::vector<double> x = CorrelatedRows(rng, rows, K);  // raw residual rows
  const std::vector<float> w = RandomVecF(rng, K, 0.3);
  // HF's linear input: x_n = x / rms(x) * (1 + w). The folded linear's input: x_n' = (x / rms) Q.
  std::vector<double> xn(x.size()), xr(x.size()), tmp(static_cast<size_t>(K));
  for (int64_t r = 0; r < rows; ++r) {
    const double* xt = x.data() + r * K;
    double ss = 0.0;
    for (int64_t k = 0; k < K; ++k) ss += xt[k] * xt[k];
    const double inv = 1.0 / std::sqrt(ss / static_cast<double>(K));
    for (int64_t k = 0; k < K; ++k) {
      xn[static_cast<size_t>(r * K + k)] = xt[k] * inv * (1.0 + static_cast<double>(w[static_cast<size_t>(k)]));
      xr[static_cast<size_t>(r * K + k)] = xt[k] * inv;
    }
    q.Apply(xr.data() + r * K, tmp.data());
  }
  const std::vector<float> H = Capture(xn, rows, K, threads);
  const std::vector<float> H_direct = Capture(xr, rows, K, threads);
  std::vector<float> Ht = H;
  TransformHessianQ(Ht, K, w.data(), q, threads);
  const double ed = RelErr(Ht, H_direct);
  Gate(ed < 1e-4, Fmt("(e) %s: TransformHessianQ(H) == H captured on the folded input (x/rms) Q (%.2e)",
                      label, ed));
  Gate(ExactlySymmetric(Ht, K), Fmt("(e) %s: H' exactly symmetric", label));

  const std::vector<float> W = RandomVecF(rng, N * K, 0.05);
  std::vector<float> Wf = W;
  FoldRowsQ(Wf, N, K, w.data(), q, threads);
  const double p0 = Proxy(W, H, N, K, threads), p1 = Proxy(Wf, Ht, N, K, threads);
  // A quantization-like error, mapped into the folded basis the way the weight is: dW' = dW D Q.
  std::vector<float> dW = RandomVecF(rng, N * K, 0.004);
  std::vector<float> dWf = dW;
  FoldRowsQ(dWf, N, K, w.data(), q, threads);
  const double e0 = Proxy(dW, H, N, K, threads), e1 = Proxy(dWf, Ht, N, K, threads);
  Gate(std::fabs(p1 - p0) <= 1e-4 * std::fabs(p0) && std::fabs(e1 - e0) <= 1e-4 * std::fabs(e0),
       Fmt("(e) %s: tr(W' H' W'^T) = tr(W H W^T) (%.6g vs %.6g), same for an error dW (%.6g vs %.6g)",
           label, p1, p0, e1, e0));
}

void TestHessians() {
  std::printf("---- (e) Hessians ----\n");
  uint64_t st = 0x1234;
  const ResidualRotation qm = GenerateResidualRotation(st, 5 * 16, 16);
  TestHessianIn(qm, 400, 6, "mini 5x16 in-projection", 4);

  // Dense cross-check at the miniature size: H' == Q^T D^-1 H D^-1 Q.
  {
    const int64_t K = qm.hidden;
    std::mt19937_64 rng(5);
    const std::vector<double> X = CorrelatedRows(rng, 300, K);
    const std::vector<float> H = Capture(X, 300, K, 4);
    const std::vector<float> w = RandomVecF(rng, K, 0.3);
    const std::vector<double> Q = DenseFromRowOp(K, [&](double* x, double* t) { qm.Apply(x, t); });
    // M = D^-1 Q ; H' = M^T H M
    std::vector<double> M(Q.size());
    for (int64_t m = 0; m < K; ++m)
      for (int64_t k = 0; k < K; ++k)
        M[static_cast<size_t>(m * K + k)] = Q[static_cast<size_t>(m * K + k)] / (1.0 + w[static_cast<size_t>(m)]);
    std::vector<double> HM(static_cast<size_t>(K * K), 0.0), ref(static_cast<size_t>(K * K), 0.0);
    for (int64_t i = 0; i < K; ++i)
      for (int64_t j = 0; j < K; ++j) {
        double s = 0.0;
        for (int64_t m = 0; m < K; ++m) s += H[static_cast<size_t>(i * K + m)] * M[static_cast<size_t>(m * K + j)];
        HM[static_cast<size_t>(i * K + j)] = s;
      }
    for (int64_t i = 0; i < K; ++i)
      for (int64_t j = 0; j < K; ++j) {
        double s = 0.0;
        for (int64_t m = 0; m < K; ++m) s += M[static_cast<size_t>(m * K + i)] * HM[static_cast<size_t>(m * K + j)];
        ref[static_cast<size_t>(i * K + j)] = s;
      }
    std::vector<float> Ht = H;
    TransformHessianQ(Ht, K, w.data(), qm, 3);
    const double e = RelErr(Ht, ref);
    Gate(e < 1e-5, Fmt("(e) mini: TransformHessianQ == dense Q^T D^-1 H D^-1 Q (%.2e)", e));
    std::vector<float> H1 = H, H7 = H;
    TransformHessianQ(H1, K, w.data(), qm, 1);
    TransformHessianQ(H7, K, w.data(), qm, 7);
    Gate(H1 == H7 && H1 == Ht, "(e) mini: TransformHessianQ bit-identical at 1, 3 and 7 threads");
  }

  // q2ab K side: H' = Hb^T H Hb vs a direct capture on h Hb, and the trace identity.
  {
    uint64_t s2 = 77;
    const BlockHadamard hb = GenerateBlockHadamard(s2, 128, 32);
    const int64_t K = hb.K, rows = 500, N = 5;
    std::mt19937_64 rng(9);
    const std::vector<double> X = CorrelatedRows(rng, rows, K);
    std::vector<double> Xh = X;
    for (int64_t r = 0; r < rows; ++r) hb.Apply(Xh.data() + r * K);
    const std::vector<float> H = Capture(X, rows, K, 4);
    std::vector<float> Ht = H;
    TransformHessianHadamard(Ht, K, hb, 4);
    const double e = RelErr(Ht, Capture(Xh, rows, K, 4));
    const std::vector<float> W = RandomVecF(rng, N * K, 0.05);
    std::vector<float> Wf = W;
    FoldRowsHadamard(Wf, N, K, hb, 2);
    const double p0 = Proxy(W, H, N, K, 2), p1 = Proxy(Wf, Ht, N, K, 2);
    Gate(e < 1e-4 && ExactlySymmetric(Ht, K) && std::fabs(p1 - p0) <= 1e-4 * std::fabs(p0),
         Fmt("(e) q2ab K=128 B=32: Hb^T H Hb == capture on h Hb (%.2e), symmetric, "
             "tr(W Hb H' (W Hb)^T) = tr(W H W^T) (%.6g vs %.6g)",
             e, p1, p0));
  }

  // Production hidden: a rank-48 input is enough to pin H' against a direct capture.
  const RotationSet rs = GenerateRotationSet(RotationKind::kQ2a, kDefaultRotationSeed, ProductionShape());
  TestHessianIn(rs.q, 48, 4, "5120 in-projection", 16);

  ExpectThrow("(e) TransformHessianQ throws on |1 + w| < 1e-3", [&]() {
    std::vector<float> H(static_cast<size_t>(qm.hidden * qm.hidden), 0.0f);
    for (int64_t i = 0; i < qm.hidden; ++i) H[static_cast<size_t>(i * qm.hidden + i)] = 1.0f;
    std::vector<float> w(static_cast<size_t>(qm.hidden), 0.1f);
    w[17] = -0.9995f;
    TransformHessianQ(H, qm.hidden, w.data(), qm, 1);
  });
}

// ---- (f) HessianStore hook -------------------------------------------------------------------

void TestStoreHook(const std::string& fixtures) {
  std::printf("---- (f) HessianStore transform hook ----\n");
  const std::string dir = (fs::u8path(fixtures) / "hess_small").u8string();
  try {
    HessianStore store(dir);
    const int64_t K = store.KOf("t.b");
    uint64_t st = 5;
    const BlockHadamard hb = GenerateBlockHadamard(st, K, 128);
    const ResidualRotation q = GenerateResidualRotation(st, K, 64);  // 4 x 64
    int calls_a = 0, calls_b = 0;
    HessianTransform xa{"test:had128", [&](std::vector<float>& H, int64_t k, int t) {
                          ++calls_a;
                          TransformHessianHadamard(H, k, hb, t);
                        }};
    HessianTransform xb{"test:q4x64", [&](std::vector<float>& H, int64_t k, int t) {
                          ++calls_b;
                          TransformHessianQ(H, k, nullptr, q, t);
                        }};

    const LdlqFactor& fa = store.Factor("t.b", K, 0.01f, 4, &xa);
    const std::vector<float> Ua = fa.U;
    Gate(calls_a == 1, "(f) the transform ran once for the first Factor");
    store.Factor("t.c", K, 0.01f, 4, &xa);
    Gate(calls_a == 1, "(f) t.c under the same transform id reuses t.b's cached factor");
    const LdlqFactor& fb = store.Factor("t.c", K, 0.01f, 4, &xb);
    Gate(calls_b == 1 && fb.U != Ua, "(f) a different transform id re-factors (and differs)");
    const LdlqFactor& fn = store.Factor("t.c", K, 0.01f, 4, nullptr);
    Gate(calls_a == 1 && calls_b == 1 && fn.U != Ua, "(f) no transform re-factors the plain H");
    const std::vector<float> Un = fn.U;
    store.Factor("t.b", K, 0.01f, 4, &xa);
    Gate(calls_a == 2, "(f) switching back to the transform re-applies it (one-entry cache)");

    // Bit-exact against doing it by hand.
    int64_t fk = 0;
    std::vector<float> H = ReadHessFile((fs::u8path(dir) / "bc.hess").u8string(), &fk, nullptr);
    std::vector<float> H0 = H;
    TransformHessianHadamard(H, K, hb, 4);
    const LdlqFactor ref = FactorHessian(std::move(H), K, 0.01f, 4);
    const LdlqFactor refn = FactorHessian(std::move(H0), K, 0.01f, 4);
    Gate(ref.U == Ua && ref.diag_h == store.Factor("t.b", K, 0.01f, 4, &xa).diag_h,
         "(f) Factor(xf) == FactorHessian(transform(ReadHessFile)) bit for bit");
    Gate(refn.U == Un, "(f) Factor(nullptr) == FactorHessian(ReadHessFile) bit for bit (unchanged path)");

    HessianTransform bad{"", [](std::vector<float>&, int64_t, int) {}};
    ExpectThrow("(f) a transform with an empty id throws", [&]() { store.Factor("t.b", K, 0.01f, 1, &bad); });
  } catch (const std::exception& e) {
    Gate(false, std::string("(f) threw: ") + e.what());
  }
}

// ---- (g) imatrix diagonal model ----------------------------------------------------------------

void TestImportance() {
  std::printf("---- (g) imatrix under the diagonal model ----\n");
  uint64_t st = 0x55;
  const ResidualRotation q = GenerateResidualRotation(st, 5 * 16, 16);
  const BlockHadamard hb = GenerateBlockHadamard(st, 96, 32);
  std::mt19937_64 rng(13);
  auto positive = [&](int64_t n) {
    std::vector<float> v = RandomVecF(rng, n);
    for (auto& x : v) x = x * x + 0.01f;
    v[3] = 900.0f;
    return v;
  };
  {
    const int64_t K = q.hidden;
    const std::vector<float> v = positive(K), w = RandomVecF(rng, K, 0.3);
    const std::vector<double> Q = DenseFromRowOp(K, [&](double* x, double* t) { q.Apply(x, t); });
    std::vector<double> ref(static_cast<size_t>(K), 0.0);
    for (int64_t k = 0; k < K; ++k)
      for (int64_t m = 0; m < K; ++m) {
        const double mk = Q[static_cast<size_t>(m * K + k)] / (1.0 + w[static_cast<size_t>(m)]);
        ref[static_cast<size_t>(k)] += mk * mk * v[static_cast<size_t>(m)];
      }
    const std::vector<float> got = TransformImportanceQ(v.data(), K, w.data(), q);
    const double e = RelErr(got, ref);
    Gate(e < 1e-6, Fmt("(g) TransformImportanceQ == diag((D^-1 Q)^T diag(v) D^-1 Q) (%.2e)", e));
  }
  {
    const int64_t K = hb.K;
    const std::vector<float> v = positive(K);
    const std::vector<double> M = DenseFromRowOp(K, [&](double* x, double*) { hb.Apply(x); });
    std::vector<double> ref(static_cast<size_t>(K), 0.0);
    for (int64_t k = 0; k < K; ++k)
      for (int64_t m = 0; m < K; ++m) {
        const double mk = M[static_cast<size_t>(m * K + k)];
        ref[static_cast<size_t>(k)] += mk * mk * v[static_cast<size_t>(m)];
      }
    const std::vector<float> got = TransformImportanceHadamard(v.data(), K, hb);
    const double e = RelErr(got, ref);
    Gate(e < 1e-6, Fmt("(g) TransformImportanceHadamard == diag(Hb^T diag(v) Hb) (%.2e)", e));
  }
}

}  // namespace

int main() {
  try {
    TestGeneration();
    std::printf("---- (b) Q ----\n");
    TestQMiniature(5, 16, 0x5EED);
    TestQMiniature(4, 64, 0xBEEF);
    const RotationSet rs = GenerateRotationSet(RotationKind::kQ2ab, kDefaultRotationSeed, ProductionShape());
    TestQProduction(rs);
    std::printf("---- (c) Hb ----\n");
    TestHbMiniature();
    TestHbProduction(rs.had_down, "had_down");
    TestHbProduction(rs.had_o, "had_o");
    TestHbProduction(rs.had_gdn_out, "had_gdn_out");
    TestFolds();
    TestHessians();
    TestStoreHook(R4DX_CONVERT_FIXTURES_DIR);
    TestImportance();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL unexpected exception: %s\n", e.what());
    ++g_failures;
  }
  if (g_failures) {
    std::fprintf(stderr, "convert_rotation: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("convert_rotation: all gates passed\n");
  return 0;
}
