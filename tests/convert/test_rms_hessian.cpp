// r4dx-convert's weightless rms Hessian for rotated in-projections (docs/quant2.md 3.2;
// rotation.hpp's TransformHessianRmsQ, hessian_store.hpp's "rms_keys", main.cpp's add_linear).
// CPU-only.
//
// A rotated container runs every zero-centred norm weightless and folds its (1 + w) into the next
// in-projections, so their input is r' = rms(x) Q and LDLQ needs H' = Q^T H_rms Q. The captured
// post-norm H = D H_rms D (D = diag(1 + w)) can only give that by division, which is impossible
// where (1 + w) == 0 (H has a zero row/column there) and noisy where it is small. hessian_capture.py
// --rms-taps / --rms-only captures H_rms itself and lists it under hessian.json "rms_keys" (its own
// CPU test is tests/reference/test_hessian_rms.py).
//
//   (a) TransformHessianRmsQ against the DENSE closed-form Q: the whole Q^T H Q at the miniature
//       sizes (5 x 16, 4 x 64; 1e-5), a 48 x 48 block of sampled rows/columns spanning all five
//       blocks at the production 5120 (1e-5); exactly symmetric; bit-identical across thread counts
//       and to TransformHessianQ(H, nullptr); refuses a K that is not Q's width.
//   (b) a dead channel ((1 + w_j) == 0, the real checkpoint has one: layer 7
//       post_attention_layernorm[3994]). On captured activations (miniature and production K): the
//       division path (TransformHessianQ on the post-norm H) throws and its message says
//       "--rms-only"; the rms path equals a direct capture on the folded input rms(x) Q (1e-4),
//       factors without a damping retry, keeps the LDLQ proxy tr(W' H' W'^T) = tr(W H W^T) with
//       W' = W D Q, and gives the dead channel's rotated direction e_j Q its full energy
//       H_rms[j][j] > 0 -- the energy a division path with the channel exempted would have zeroed.
//   (c) HessianStore "rms_keys" on synthetic .hess files: HasRms / RmsFile / FileFor, CheckFile of
//       either source, Factor(kRms) == FactorHessian(ReadHessFile(rms file)) bit for bit, the cache
//       shared by two bases on one rms file and NOT between the keys and rms files of one base, a
//       manifest without the map (HasRms false, RmsFile / Factor(kRms) throw), and every refusal:
//       not an object, a non-string value, an unlisted file, a base missing from "keys", a K that
//       differs from the "keys" file's, the "keys" file itself, another base's "keys" file,
//       another tap's rms file of the same K (the rms file must be the base's own "keys" file with
//       ".rms" before ".hess"), rows that differ from the "keys" file's, a truncated rms file, and
//       (at Factor(kRms)) an rms file with a zero diagonal -- what a norm-OUTPUT capture gives on
//       a dead channel.
//   (d) r4dx-convert (when built) on a synthetic 2-layer checkpoint (layer 0 GDN, layer 1 full
//       attention, hidden 5120, row counts shrunk) whose post_attention_layernorm has a dead
//       channel, with --ldlq ".*" against three Hessian directories: A (keys only), B (keys +
//       rms_keys for all seven in-projections), C (rms_keys for the two mlp.gate_up only). The
//       K = 5120 factorizations dominate the run time, so the runs that do not need layer 1 use
//       --layers 1.
//         unrotated A (--layers 1): the container's tensor digest equals the golden taken with the
//             build from before this path existed -- unrotated output is byte-identical;
//         unrotated B (--layers 1): every tensor byte-identical to A's, the header identical except
//             hessian_dir / hessian_manifest_sha256, no ldlq_rms_linears key;
//         q2ab A: refused during PLANNING (no output file), naming text.layers.0.mlp.gate_up,
//             norm_weight[3994] and --rms-only;
//         q2ab B: converts; r4dx_convert_run.ldlq_rms_linears lists exactly the seven
//             in-projections; gdn.in_proj_z and attn.k / v reuse the cached rms factor; the
//             dead-channel mlp.gate_up and attn.k w4a16 bytes equal fold -> Q^T H_rms Q -> LDLQ;
//         q2ab C (--layers 1): both paths in one run: ldlq_rms_linears == [mlp.gate_up];
//             gdn.in_proj_qkv's bytes equal fold -> Q^T D^-1 H D^-1 Q -> LDLQ, gate_up's equal B's.
//
// Modes, to regenerate (d)'s golden with another r4dx-convert build:
//   convert_rms_hessian --make-fixture <dir>   writes (d)'s checkpoint and Hessian directories
//   convert_rms_hessian --digest <container>   prints the digest (d) compares
// then run that build's r4dx-convert with the command --make-fixture prints (CommonArgs() below).
// Every file this test writes goes to a fresh directory under %TEMP% and is removed at the end.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/hessian_store.hpp"
#include "r4dx_convert/linear_layouts.hpp"
#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/quant_ldlq.hpp"
#include "r4dx_convert/rotation.hpp"
#include "r4dx_convert/sha256.hpp"
#include "r4dx_convert/threadpool.hpp"

namespace {

using namespace r4dx_convert;
namespace fs = std::filesystem;

int g_failures = 0;
constexpr int kThreads = 6;  // this machine is shared; every parallel section here uses at most this

std::string Fmt(const char* fmt, ...) {
  char buf[2048];
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
  std::fflush(stdout);
  return cond;
}

// Passes when f() throws and the message contains `needle` (empty: any message).
template <typename F>
void ExpectThrow(const std::string& label, const std::string& needle, F&& f) {
  try {
    f();
  } catch (const std::exception& e) {
    const std::string msg = e.what();
    Gate(needle.empty() || msg.find(needle) != std::string::npos,
         label + " (threw: " + msg + ")" + (needle.empty() ? "" : " [must mention '" + needle + "']"));
    return;
  }
  Gate(false, label + " (did not throw)");
}

bool Contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
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

// Q[m][k] from its documented closed form (rotation.hpp): d[m] H[i][j] / sqrt(B) R[c][b].
double QClosedForm(const ResidualRotation& q, int64_t m, int64_t k) {
  const int64_t c = m / q.block, i = m % q.block, b = k / q.block, j = k % q.block;
  const double h = (Popcount(static_cast<uint64_t>(i & j)) & 1) ? -1.0 : 1.0;
  return q.signs[static_cast<size_t>(m)] * h / std::sqrt(static_cast<double>(q.block)) *
         q.mix[static_cast<size_t>(c * q.nblk + b)];
}

// max |a - b| over the given entries / max |b|.
double RelErr(const std::vector<float>& a, const std::vector<double>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    num = std::max(num, std::fabs(static_cast<double>(a[i]) - b[i]));
    den = std::max(den, std::fabs(b[i]));
  }
  return den > 0.0 ? num / den : num;
}

double RelErrF(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < b.size(); ++i) {
    num = std::max(num, std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i])));
    den = std::max(den, std::fabs(static_cast<double>(b[i])));
  }
  return den > 0.0 ? num / den : num;
}

bool ExactlySymmetric(const std::vector<float>& H, int64_t K) {
  for (int64_t i = 0; i < K; ++i)
    for (int64_t j = i + 1; j < K; ++j)
      if (H[static_cast<size_t>(i * K + j)] != H[static_cast<size_t>(j * K + i)]) return false;
  return true;
}

// Random symmetric K x K (fp32), with one heavy channel and one all-zero channel: the identity
// Q^T H Q holds for any symmetric H, so nothing here is special-cased.
std::vector<float> RandomSymmetric(std::mt19937_64& rng, int64_t K, int64_t heavy, int64_t zero) {
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<float> H(static_cast<size_t>(K * K));
  for (int64_t i = 0; i < K; ++i)
    for (int64_t j = i; j < K; ++j) {
      double v = nd(rng) + (i == j ? 3.0 : 0.0);
      if (i == heavy || j == heavy) v *= 40.0;
      if (i == zero || j == zero) v = 0.0;
      H[static_cast<size_t>(i * K + j)] = H[static_cast<size_t>(j * K + i)] = static_cast<float>(v);
    }
  return H;
}

// H = X^T X / rows over double rows, fp32 out (hessian_capture.py's convention).
std::vector<float> Capture(const std::vector<double>& X, int64_t rows, int64_t K) {
  std::vector<float> H(static_cast<size_t>(K * K));
  ParallelFor(0, K, kThreads, [&](int64_t i0, int64_t i1) {
    std::vector<double> acc(static_cast<size_t>(K));
    for (int64_t i = i0; i < i1; ++i) {
      std::fill(acc.begin(), acc.end(), 0.0);
      for (int64_t r = 0; r < rows; ++r) {
        const double* x = X.data() + r * K;
        const double xi = x[i];
        for (int64_t j = 0; j < K; ++j) acc[static_cast<size_t>(j)] += xi * x[j];
      }
      for (int64_t j = 0; j < K; ++j)
        H[static_cast<size_t>(i * K + j)] = static_cast<float>(acc[static_cast<size_t>(j)] / rows);
    }
  });
  return H;
}

// tr(W H W^T), W [N, K].
double Proxy(const std::vector<float>& W, const std::vector<float>& H, int64_t N, int64_t K) {
  std::vector<double> part(static_cast<size_t>(N), 0.0);
  ParallelFor(0, N, kThreads, [&](int64_t n0, int64_t n1) {
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

// ---- (a) TransformHessianRmsQ vs the dense Q --------------------------------------------------------

void TestDenseMini(int64_t nblk, int64_t block, uint64_t seed) {
  uint64_t st = seed;
  const ResidualRotation q = GenerateResidualRotation(st, nblk * block, block);
  const int64_t K = q.hidden;
  std::mt19937_64 rng(seed);
  const std::vector<float> H = RandomSymmetric(rng, K, 3, K - 2);
  // ref = Q^T H Q with the closed-form Q, all in double.
  std::vector<double> Q(static_cast<size_t>(K * K)), HQ(static_cast<size_t>(K * K), 0.0),
      ref(static_cast<size_t>(K * K), 0.0);
  for (int64_t m = 0; m < K; ++m)
    for (int64_t k = 0; k < K; ++k) Q[static_cast<size_t>(m * K + k)] = QClosedForm(q, m, k);
  for (int64_t i = 0; i < K; ++i)
    for (int64_t m = 0; m < K; ++m) {
      const double h = H[static_cast<size_t>(i * K + m)];
      for (int64_t j = 0; j < K; ++j) HQ[static_cast<size_t>(i * K + j)] += h * Q[static_cast<size_t>(m * K + j)];
    }
  for (int64_t m = 0; m < K; ++m)
    for (int64_t i = 0; i < K; ++i) {
      const double qm = Q[static_cast<size_t>(m * K + i)];
      for (int64_t j = 0; j < K; ++j) ref[static_cast<size_t>(i * K + j)] += qm * HQ[static_cast<size_t>(m * K + j)];
    }
  std::vector<float> Ht = H, H1 = H, H7 = H, Hd = H;
  TransformHessianRmsQ(Ht, K, q, 3);
  TransformHessianRmsQ(H1, K, q, 1);
  TransformHessianRmsQ(H7, K, q, 7);
  TransformHessianQ(Hd, K, nullptr, q, 3);
  const double e = RelErr(Ht, ref);
  Gate(e < 1e-5, Fmt("(a) %lldx%lld: TransformHessianRmsQ(H) == dense Q^T H Q, every entry (%.2e)",
                     (long long)nblk, (long long)block, e));
  Gate(ExactlySymmetric(Ht, K) && Ht == H1 && Ht == H7 && Ht == Hd,
       Fmt("(a) %lldx%lld: exactly symmetric; bit-identical at 1 / 3 / 7 threads and to "
           "TransformHessianQ(H, nullptr)",
           (long long)nblk, (long long)block));
}

void TestDenseProduction() {
  const RotationShape shape = [] {
    RotationShape s;
    s.hidden = 5120;
    return s;
  }();
  const RotationSet rs = GenerateRotationSet(RotationKind::kQ2a, kDefaultRotationSeed, shape);
  const ResidualRotation& q = rs.q;
  const int64_t K = q.hidden;
  std::mt19937_64 rng(0xA11CE);
  const std::vector<float> H = RandomSymmetric(rng, K, 3994, 17);
  // Sampled indices: both edges of every 1024-block plus random interior ones.
  std::vector<int64_t> idx;
  for (int64_t b = 0; b < 5; ++b) {
    idx.push_back(b * 1024);
    idx.push_back(b * 1024 + 1023);
  }
  idx.push_back(17);
  idx.push_back(3994);
  std::uniform_int_distribution<int64_t> ud(0, K - 1);
  while (idx.size() < 48) idx.push_back(ud(rng));
  const int64_t S = static_cast<int64_t>(idx.size());
  // Qs = the S sampled COLUMNS of the dense Q; Y = H Qs (K x S); ref = Qs^T Y (S x S).
  std::vector<double> Qs(static_cast<size_t>(K * S));
  for (int64_t m = 0; m < K; ++m)
    for (int64_t s = 0; s < S; ++s) Qs[static_cast<size_t>(m * S + s)] = QClosedForm(q, m, idx[static_cast<size_t>(s)]);
  std::vector<double> Y(static_cast<size_t>(K * S), 0.0);
  ParallelFor(0, K, kThreads, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      double* y = Y.data() + i * S;
      const float* h = H.data() + i * K;
      for (int64_t m = 0; m < K; ++m) {
        const double hv = h[m];
        const double* qs = Qs.data() + m * S;
        for (int64_t s = 0; s < S; ++s) y[s] += hv * qs[s];
      }
    }
  });
  std::vector<double> ref(static_cast<size_t>(S * S), 0.0);
  for (int64_t m = 0; m < K; ++m)
    for (int64_t a = 0; a < S; ++a) {
      const double qa = Qs[static_cast<size_t>(m * S + a)];
      for (int64_t b = 0; b < S; ++b) ref[static_cast<size_t>(a * S + b)] += qa * Y[static_cast<size_t>(m * S + b)];
    }
  std::vector<float> Ht = H;
  TransformHessianRmsQ(Ht, K, q, kThreads);
  std::vector<float> got(static_cast<size_t>(S * S));
  for (int64_t a = 0; a < S; ++a)
    for (int64_t b = 0; b < S; ++b)
      got[static_cast<size_t>(a * S + b)] =
          Ht[static_cast<size_t>(idx[static_cast<size_t>(a)] * K + idx[static_cast<size_t>(b)])];
  const double e = RelErr(got, ref);
  Gate(e < 1e-5, Fmt("(a) 5120: TransformHessianRmsQ(H) == dense Q^T H Q on a %lldx%lld block of "
                     "sampled rows/columns spanning all five blocks (%.2e)",
                     (long long)S, (long long)S, e));
  std::vector<float> H3 = H;
  TransformHessianRmsQ(H3, K, q, 3);
  Gate(ExactlySymmetric(Ht, K) && H3 == Ht, "(a) 5120: exactly symmetric; bit-identical at 3 and 6 threads");
  ExpectThrow("(a) TransformHessianRmsQ refuses K != Q's width", "TransformHessianRmsQ", [&]() {
    std::vector<float> Hs(64 * 64, 0.0f);
    TransformHessianRmsQ(Hs, 64, q, 1);
  });
}

// ---- (b) dead channel ---------------------------------------------------------------------------

// Residual rows: AR(1) along channels, a few heavy channels, and a large value on `dead` -- a dead
// norm channel is exactly where the residual may carry something big that the norm then discards.
std::vector<double> ResidualRows(std::mt19937_64& rng, int64_t rows, int64_t K, int64_t dead) {
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> X(static_cast<size_t>(rows * K));
  for (int64_t r = 0; r < rows; ++r) {
    double prev = nd(rng);
    for (int64_t k = 0; k < K; ++k) {
      prev = 0.8 * prev + 0.6 * nd(rng);
      double v = prev * ((k % 29 == 3) ? 12.0 : 1.0);
      if (k == dead) v = 30.0 + 5.0 * nd(rng);
      X[static_cast<size_t>(r * K + k)] = v;
    }
  }
  return X;
}

void TestDeadChannel(const ResidualRotation& q, int64_t rows, int64_t dead, const char* label) {
  const int64_t K = q.hidden, N = 6;
  std::mt19937_64 rng(77 + K);
  const std::vector<double> x = ResidualRows(rng, rows, K, dead);
  std::vector<float> w = RandomVecF(rng, K, 0.3);
  w[static_cast<size_t>(dead)] = -1.0f;  // (1 + w) == 0, exactly as in the checkpoint's bf16
  // HF's post-norm input x_n = rms(x) (1 + w); the weightless r = rms(x); the folded input r Q.
  const double eps = 1e-6;
  std::vector<double> xn(x.size()), xr(x.size()), xq(x.size()), tmp(static_cast<size_t>(K));
  for (int64_t r = 0; r < rows; ++r) {
    const double* xt = x.data() + r * K;
    double ss = 0.0;
    for (int64_t k = 0; k < K; ++k) ss += xt[k] * xt[k];
    const double inv = 1.0 / std::sqrt(ss / static_cast<double>(K) + eps);
    for (int64_t k = 0; k < K; ++k) {
      xr[static_cast<size_t>(r * K + k)] = xt[k] * inv;
      xn[static_cast<size_t>(r * K + k)] = xt[k] * inv * (1.0 + static_cast<double>(w[static_cast<size_t>(k)]));
    }
    std::copy(xr.begin() + r * K, xr.begin() + (r + 1) * K, xq.begin() + r * K);
    q.Apply(xq.data() + r * K, tmp.data());
  }
  const std::vector<float> H_post = Capture(xn, rows, K);
  const std::vector<float> H_rms = Capture(xr, rows, K);
  const std::vector<float> H_direct = Capture(xq, rows, K);

  bool zero_rowcol = true;
  for (int64_t k = 0; k < K; ++k)
    zero_rowcol = zero_rowcol && H_post[static_cast<size_t>(dead * K + k)] == 0.0f &&
                  H_post[static_cast<size_t>(k * K + dead)] == 0.0f;
  Gate(zero_rowcol && H_rms[static_cast<size_t>(dead * K + dead)] > 0.0f,
       Fmt("(b) %s: the post-norm H has an all-zero row/column at the dead channel %lld; H_rms[%lld][%lld] "
           "= %.4g > 0",
           label, (long long)dead, (long long)dead, (long long)dead,
           static_cast<double>(H_rms[static_cast<size_t>(dead * K + dead)])));

  ExpectThrow(Fmt("(b) %s: the division path (TransformHessianQ on the post-norm H) refuses it", label),
              "--rms-only", [&]() {
                std::vector<float> Hc = H_post;
                TransformHessianQ(Hc, K, w.data(), q, kThreads);
              });

  std::vector<float> Ht = H_rms;
  TransformHessianRmsQ(Ht, K, q, kThreads);
  const double ed = RelErrF(Ht, H_direct);
  Gate(ed < 1e-4, Fmt("(b) %s: TransformHessianRmsQ(H_rms) == H captured on the folded input rms(x) Q "
                      "(%.2e)",
                      label, ed));
  const LdlqFactor f = FactorHessian(Ht, K, 0.01f, kThreads);
  Gate(f.retries == 0, Fmt("(b) %s: H' factors at damp 0.01 without a retry", label));

  // The LDLQ proxy of the folded weight (and of a folded error) equals the original's against the
  // post-norm H -- dead channel included, where both sides have zero weight.
  const std::vector<float> W = RandomVecF(rng, N * K, 0.05);
  std::vector<float> Wf = W;
  FoldRowsQ(Wf, N, K, w.data(), q, kThreads);
  const double p0 = Proxy(W, H_post, N, K), p1 = Proxy(Wf, Ht, N, K);
  const std::vector<float> dW = RandomVecF(rng, N * K, 0.004);
  std::vector<float> dWf = dW;
  FoldRowsQ(dWf, N, K, w.data(), q, kThreads);
  const double e0 = Proxy(dW, H_post, N, K), e1 = Proxy(dWf, Ht, N, K);
  Gate(std::fabs(p1 - p0) <= 1e-4 * std::fabs(p0) && std::fabs(e1 - e0) <= 1e-4 * std::fabs(e0),
       Fmt("(b) %s: tr(W' H' W'^T) = tr(W H_post W^T) with W' = W D Q (%.6g vs %.6g), same for dW "
           "(%.6g vs %.6g)",
           label, p1, p0, e1, e0));

  // In the rotated basis the rounding error is free along every direction; the dead channel's own
  // direction u = e_j Q must carry its full energy u H' u^T = H_rms[j][j]. A division path that
  // exempted the channel (D^-1 := 0 there) would give 0 and let error leak along u unpenalized.
  std::vector<double> u(static_cast<size_t>(K), 0.0);
  u[static_cast<size_t>(dead)] = 1.0;
  q.Apply(u.data(), tmp.data());
  double uhu = 0.0;
  for (int64_t i = 0; i < K; ++i) {
    double t = 0.0;
    for (int64_t j = 0; j < K; ++j) t += static_cast<double>(Ht[static_cast<size_t>(i * K + j)]) * u[static_cast<size_t>(j)];
    uhu += u[static_cast<size_t>(i)] * t;
  }
  const double hjj = H_rms[static_cast<size_t>(dead * K + dead)];
  Gate(std::fabs(uhu - hjj) <= 1e-4 * hjj,
       Fmt("(b) %s: the dead channel's rotated direction e_j Q has energy %.6g in H' == H_rms[j][j] = "
           "%.6g",
           label, uhu, hjj));
}

// ---- .hess writing -------------------------------------------------------------------------------

std::string ReadWhole(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  if (!f) throw std::runtime_error("cannot open " + p.u8string());
  return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
}

void WriteWhole(const fs::path& p, const std::string& bytes) {
  std::ofstream f(p, std::ios::binary | std::ios::trunc);
  f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!f) throw std::runtime_error("cannot write " + p.u8string());
}

// hessian_store.hpp's format from a packed upper triangle; returns the header trace (the fp64 sum
// of the stored fp32 diagonal), which the manifest repeats.
double WriteHessPacked(const fs::path& path, int64_t K, uint64_t rows, const std::vector<float>& packed) {
  if (packed.size() != static_cast<size_t>(K * (K + 1) / 2)) throw std::logic_error("WriteHessPacked: size");
  double trace = 0.0;
  size_t off = 0;
  for (int64_t i = 0; i < K; ++i) {
    trace += static_cast<double>(packed[off]);
    off += static_cast<size_t>(K - i);
  }
  std::string bytes(64, '\0');
  std::memcpy(&bytes[0], "R4DXHES1", 8);
  const uint32_t k32 = static_cast<uint32_t>(K), flags = 1;
  std::memcpy(&bytes[8], &k32, 4);
  std::memcpy(&bytes[12], &flags, 4);
  std::memcpy(&bytes[16], &rows, 8);
  std::memcpy(&bytes[24], &trace, 8);
  bytes.append(reinterpret_cast<const char*>(packed.data()), packed.size() * sizeof(float));
  WriteWhole(path, bytes);
  return trace;
}

std::vector<float> PackUpper(const std::vector<float>& H, int64_t K) {
  std::vector<float> p;
  p.reserve(static_cast<size_t>(K * (K + 1) / 2));
  for (int64_t i = 0; i < K; ++i)
    for (int64_t j = i; j < K; ++j) p.push_back(H[static_cast<size_t>(i * K + j)]);
  return p;
}

// A positive definite H_rms = diag(s) + V V^T (V: K x r), lognormal channel scales, `heavy` (>= 0)
// a large channel. Packed upper triangle, fp32. Built row-parallel; the values do not depend on it.
std::vector<float> LowRankPdPacked(uint64_t seed, int64_t K, int r, int64_t heavy) {
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> nd(0.0, 1.0);
  std::uniform_real_distribution<double> ud(0.0, 1.0);
  std::vector<double> V(static_cast<size_t>(K * r)), s(static_cast<size_t>(K));
  for (int64_t k = 0; k < K; ++k) {
    const double c = (k == heavy) ? 6.0 : std::exp(0.5 * nd(rng));
    for (int t = 0; t < r; ++t) V[static_cast<size_t>(k * r + t)] = nd(rng) * c / std::sqrt(static_cast<double>(r));
    s[static_cast<size_t>(k)] = (0.05 + 0.1 * ud(rng)) * c * c;
  }
  std::vector<float> p(static_cast<size_t>(K * (K + 1) / 2));
  ParallelFor(0, K, kThreads, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      size_t off = static_cast<size_t>(i * K - i * (i - 1) / 2);
      const double* vi = V.data() + i * r;
      for (int64_t j = i; j < K; ++j) {
        const double* vj = V.data() + j * r;
        double h = (i == j) ? s[static_cast<size_t>(i)] : 0.0;
        for (int t = 0; t < r; ++t) h += vi[t] * vj[t];
        p[off++] = static_cast<float>(h);
      }
    }
  });
  return p;
}

// The post-norm capture of the same activations: H_post[i][j] = (1 + w_i) (1 + w_j) H_rms[i][j],
// computed in double from the stored fp32 H_rms (so a dead channel's row/column is exactly 0).
std::vector<float> PostFromRms(const std::vector<float>& rms_packed, int64_t K, const std::vector<float>& w) {
  std::vector<float> p(rms_packed.size());
  ParallelFor(0, K, kThreads, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      size_t off = static_cast<size_t>(i * K - i * (i - 1) / 2);
      const double di = 1.0 + static_cast<double>(w[static_cast<size_t>(i)]);
      for (int64_t j = i; j < K; ++j, ++off)
        p[off] = static_cast<float>(static_cast<double>(rms_packed[off]) * di *
                                    (1.0 + static_cast<double>(w[static_cast<size_t>(j)])));
    }
  });
  return p;
}

// ---- (c) HessianStore rms_keys -------------------------------------------------------------------

fs::path TempDir(const std::string& tag) {
  const fs::path d = fs::temp_directory_path() /
                     ("r4dx_test_rms_hessian_" + tag + "_" +
                      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  fs::create_directories(d);
  return d;
}

void TestStore() {
  std::printf("---- (c) HessianStore rms_keys ----\n");
  const int64_t K = 128;
  const uint64_t rows = 4096;
  const fs::path dir = TempDir("store");
  try {
    std::mt19937_64 rng(4242);
    std::vector<float> w = RandomVecF(rng, K, 0.3);
    w[5] = -1.0f;
    const std::vector<float> rms = LowRankPdPacked(11, K, 8, 5);
    const std::vector<float> post = PostFromRms(rms, K, w);
    const std::vector<float> other = LowRankPdPacked(12, K, 8, -1);
    const std::vector<float> small = LowRankPdPacked(13, 64, 4, -1);
    const double tr_rms = WriteHessPacked(dir / "t.in.rms.hess", K, rows, rms);
    const double tr_post = WriteHessPacked(dir / "t.in.hess", K, rows, post);
    const double tr_other = WriteHessPacked(dir / "t.out.hess", K, rows, other);
    const double tr_small = WriteHessPacked(dir / "small.hess", 64, rows, small);
    nlohmann::json base = {
        {"format", "r4dx-hessian"},
        {"version", 1},
        {"files",
         {{"t.in.hess", {{"K", K}, {"rows", rows}, {"trace", tr_post}}},
          {"t.in.rms.hess", {{"K", K}, {"rows", rows}, {"trace", tr_rms}}},
          {"t.out.hess", {{"K", K}, {"rows", rows}, {"trace", tr_other}}},
          {"small.hess", {{"K", 64}, {"rows", rows}, {"trace", tr_small}}}}},
        {"keys", {{"t.qkv", "t.in.hess"}, {"t.z", "t.in.hess"}, {"t.out", "t.out.hess"}, {"t.small", "small.hess"}}},
    };
    auto write_manifest = [&](const nlohmann::json& j) { WriteWhole(dir / "hessian.json", j.dump(2)); };

    // Without the map: the pre-rms manifest reads exactly as before.
    write_manifest(base);
    {
      HessianStore s(dir.u8string());
      Gate(!s.HasRms("t.qkv") && !s.HasRms("t.out") && s.Has("t.qkv"),
           "(c) a manifest without rms_keys: HasRms is false everywhere, keys unchanged");
      ExpectThrow("(c) ... RmsFile throws", "rms_keys", [&]() { s.RmsFile("t.qkv"); });
      ExpectThrow("(c) ... Factor(kRms) throws", "rms_keys",
                  [&]() { s.Factor("t.qkv", K, 0.01f, 2, nullptr, HessianSource::kRms); });
    }

    nlohmann::json good = base;
    good["rms_keys"] = {{"t.qkv", "t.in.rms.hess"}, {"t.z", "t.in.rms.hess"}};
    write_manifest(good);
    {
      HessianStore s(dir.u8string());
      Gate(s.HasRms("t.qkv") && s.HasRms("t.z") && !s.HasRms("t.out") && !s.HasRms("nope"),
           "(c) HasRms: t.qkv, t.z yes; t.out (no entry), unknown no");
      Gate(s.RmsFile("t.qkv") == "t.in.rms.hess" && s.File("t.qkv") == "t.in.hess" &&
               s.FileFor("t.z", HessianSource::kRms) == "t.in.rms.hess" &&
               s.FileFor("t.z", HessianSource::kKeys) == "t.in.hess" && s.KOf("t.qkv") == K,
           "(c) RmsFile / File / FileFor resolve the two maps independently");
      s.CheckFile("t.qkv");
      s.CheckFile("t.qkv", HessianSource::kRms);
      Gate(true, "(c) CheckFile passes for both sources");

      int64_t fk = 0;
      const LdlqFactor ref_rms = FactorHessian(ReadHessFile((dir / "t.in.rms.hess").u8string(), &fk, nullptr), K, 0.01f, 2);
      const LdlqFactor ref_post = FactorHessian(ReadHessFile((dir / "t.in.hess").u8string(), &fk, nullptr), K, 0.01f, 2);
      const LdlqFactor& fr = s.Factor("t.qkv", K, 0.01f, 2, nullptr, HessianSource::kRms);
      Gate(fr.U == ref_rms.U && fr.diag_h == ref_rms.diag_h,
           "(c) Factor(kRms) == FactorHessian(ReadHessFile(rms file)) bit for bit");
      const_cast<LdlqFactor&>(fr).retries = 555;  // cache marker (see test_quant_ldlq (f))
      const LdlqFactor& fz = s.Factor("t.z", K, 0.01f, 2, nullptr, HessianSource::kRms);
      Gate(&fz == &fr && fz.retries == 555, "(c) t.z on the same rms file reuses t.qkv's cached factor");
      const LdlqFactor& fp = s.Factor("t.z", K, 0.01f, 2, nullptr, HessianSource::kKeys);
      Gate(fp.retries != 555 && fp.U == ref_post.U,
           "(c) the same base's \"keys\" file re-factors (the file is part of the cache key) and "
           "equals FactorHessian of the post-norm file");
      Gate(ref_post.diag_h[5] == 0.0f && ref_rms.diag_h[5] > 0.0f,
           "(c) the fixture's post-norm H has the dead channel's zero diagonal; the rms H does not");
      // With transforms: same id + same rms file shares; the id alone does not make a hit.
      int calls = 0;
      HessianTransform xa{"x", [&](std::vector<float>&, int64_t, int) { ++calls; }};
      s.Factor("t.qkv", K, 0.01f, 2, &xa, HessianSource::kRms);
      s.Factor("t.z", K, 0.01f, 2, &xa, HessianSource::kRms);
      s.Factor("t.z", K, 0.01f, 2, &xa, HessianSource::kKeys);
      Gate(calls == 2, "(c) with a transform: qkv/z share (one call), then the keys file with the "
                       "same id re-applies it");
      ExpectThrow("(c) RmsFile of a base without an entry throws", "t.out", [&]() { s.RmsFile("t.out"); });
      ExpectThrow("(c) CheckFile(kRms) of a base without an entry throws", "rms_keys",
                  [&]() { s.CheckFile("t.out", HessianSource::kRms); });
    }

    auto refused = [&](const std::string& what, const std::string& needle, const nlohmann::json& j) {
      write_manifest(j);
      ExpectThrow("(c) refuses rms_keys " + what, needle, [&]() { HessianStore s(dir.u8string()); });
    };
    {
      nlohmann::json j = base;
      j["rms_keys"] = nlohmann::json::array({"t.in.rms.hess"});
      refused("that is not an object", "not an object", j);
    }
    {
      nlohmann::json j = good;
      j["rms_keys"]["t.qkv"] = 3;
      refused("with a non-string value", "not a file name", j);
    }
    {
      nlohmann::json j = good;
      j["rms_keys"]["t.qkv"] = "t.in.rsm.hess";
      refused("naming a file not listed under \"files\"", "not listed under \"files\"", j);
    }
    {
      nlohmann::json j = good;
      j["rms_keys"]["t.gate_up"] = "t.in.rms.hess";
      refused("for a base with no \"keys\" entry", "no \"keys\" entry", j);
    }
    {
      nlohmann::json j = good;  // correctly paired by name, but K 128 for a K=64 base
      j["files"]["small.rms.hess"] = {{"K", K}, {"rows", rows}, {"trace", tr_rms}};
      j["rms_keys"]["t.small"] = "small.rms.hess";
      refused("whose K differs from the \"keys\" file's", "K=64", j);
    }
    {
      nlohmann::json j = good;
      j["rms_keys"]["t.qkv"] = "t.in.hess";
      refused("naming the base's own \"keys\" file", "same file", j);
    }
    {
      nlohmann::json j = good;  // K matches (every norm-fed tap is K = hidden): only the pairing tells
      j["rms_keys"]["t.qkv"] = "t.out.hess";
      refused("naming another base's \"keys\" file", "is a \"keys\" file", j);
    }
    {
      nlohmann::json j = good;  // t.out reads t.out.hess; t.in.rms.hess is the in tap's rms(x)
      j["rms_keys"]["t.out"] = "t.in.rms.hess";
      refused("naming another tap's rms file (same K)", "whose rms Hessian is \"t.out.rms.hess\"", j);
    }
    {
      nlohmann::json j = good;  // correctly paired and K-equal, but from a different token count
      j["files"]["t.out.rms.hess"] = {{"K", K}, {"rows", rows / 2}, {"trace", tr_other}};
      j["rms_keys"]["t.out"] = "t.out.rms.hess";
      refused("whose rows differ from the \"keys\" file's", "rows=2048", j);
    }
    // A zero diagonal in an rms file: what a capture of the norm's OUTPUT gives on a dead channel.
    // The manifest is fine; Factor(kRms) refuses it (the keys source of the same base still works).
    {
      const std::string bytes = ReadWhole(dir / "t.in.rms.hess");
      // H_post: row/column 5 are exactly 0. The manifest lists the file's own trace, as a capture
      // would -- CheckFile holds the header's trace to it exactly.
      nlohmann::json j = good;
      j["files"]["t.in.rms.hess"]["trace"] = WriteHessPacked(dir / "t.in.rms.hess", K, rows, post);
      write_manifest(j);
      HessianStore s(dir.u8string());
      s.CheckFile("t.qkv", HessianSource::kRms);
      ExpectThrow("(c) Factor(kRms) refuses an rms file with a zero diagonal", "H[5][5] = 0", [&]() {
        s.Factor("t.qkv", K, 0.01f, 2, nullptr, HessianSource::kRms);
      });
      bool ok = true;
      try {
        s.Factor("t.qkv", K, 0.01f, 2, nullptr, HessianSource::kKeys);
      } catch (const std::exception&) {
        ok = false;
      }
      Gate(ok, "(c) ... while Factor(kKeys) of the post-norm file (zero at the dead channel) still factors");
      WriteWhole(dir / "t.in.rms.hess", bytes);
    }
    // Two same-K files swapped under one manifest: K and rows agree, only the header trace (one f64,
    // written to both by the capture) tells them apart -- CheckFile and Factor refuse it exactly.
    {
      write_manifest(good);
      const std::string rms_bytes = ReadWhole(dir / "t.in.rms.hess");
      const std::string out_bytes = ReadWhole(dir / "t.out.hess");
      WriteWhole(dir / "t.in.rms.hess", out_bytes);
      WriteWhole(dir / "t.out.hess", rms_bytes);
      HessianStore s(dir.u8string());
      ExpectThrow("(c) CheckFile refuses a file whose header trace is not the manifest's", "trace=",
                  [&]() { s.CheckFile("t.qkv", HessianSource::kRms); });
      ExpectThrow("(c) ... and Factor refuses it too", "trace=",
                  [&]() { s.Factor("t.out", K, 0.01f, 2, nullptr, HessianSource::kKeys); });
      WriteWhole(dir / "t.in.rms.hess", rms_bytes);
      WriteWhole(dir / "t.out.hess", out_bytes);
    }
    // A truncated rms file passes the manifest but fails the planning-time CheckFile(kRms).
    {
      write_manifest(good);
      const std::string bytes = ReadWhole(dir / "t.in.rms.hess");
      WriteWhole(dir / "t.in.rms.hess", bytes.substr(0, bytes.size() - 8));
      HessianStore s(dir.u8string());
      s.CheckFile("t.qkv");
      ExpectThrow("(c) CheckFile(kRms) rejects a truncated rms file", "size", [&]() {
        s.CheckFile("t.qkv", HessianSource::kRms);
      });
      WriteWhole(dir / "t.in.rms.hess", bytes);
    }
  } catch (const std::exception& e) {
    Gate(false, std::string("(c) threw unexpectedly: ") + e.what());
  }
  std::error_code ec;
  fs::remove_all(dir, ec);
}

// ---- (d) the exe on a synthetic checkpoint -------------------------------------------------------

constexpr int64_t kHidden = 5120;
constexpr int64_t kInter = 512;     // q2ab's had_down block: intermediate_size must be a multiple of it
constexpr int64_t kHeadDim = 256;   // q2ab: attn.o's Hadamard block is one head
constexpr int64_t kVDim = 128;      // q2ab: gdn.out_proj's Hadamard block is one value head
constexpr int64_t kVocab = 16;
constexpr int64_t kDead = 3994;     // the real checkpoint's dead post_attention_layernorm channel
constexpr uint64_t kRows = 172156;
const char* const kL0 = "model.language_model.layers.0.";
const char* const kL1 = "model.language_model.layers.1.";

float Bf16Round(float x) { return r4dx::core::Bf16ToFloat(r4dx::core::FloatToBf16(x)); }

struct SynthTensor {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<float> v;  // bf16-exact
};

struct Synth {
  std::vector<SynthTensor> tensors;
  std::map<std::string, size_t> idx;
  void Add(const std::string& name, std::vector<int64_t> shape, std::vector<float> v) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    if (n != static_cast<int64_t>(v.size())) throw std::logic_error("Synth::Add: " + name);
    idx[name] = tensors.size();
    tensors.push_back({name, std::move(shape), std::move(v)});
  }
  const std::vector<float>& Get(const std::string& name) const { return tensors.at(idx.at(name)).v; }
};

// The layer shapes the converter checks are real (K = hidden for in-projections; out-projection K =
// one Hadamard block per head / 512 for mlp.down; N % 16 == 0 for every quantized linear); the row
// counts it does not check are shrunk to keep the test fast.
Synth MakeCheckpoint() {
  Synth s;
  std::mt19937_64 rng(0x5EED0415ull);
  std::normal_distribution<double> nd(0.0, 1.0);
  auto rnd = [&](int64_t n, double sigma) {
    std::vector<float> v(static_cast<size_t>(n));
    for (auto& x : v) x = Bf16Round(static_cast<float>(sigma * nd(rng)));
    return v;
  };
  // Zero-centred norm weights, clamped so the only |1 + w| < 1e-3 is the planted one (an unclamped
  // N(0, 0.3) draw of 5120 rounds to -1.0 in bf16 about one time in ten).
  auto norm_w = [&]() {
    std::vector<float> v = rnd(kHidden, 0.3);
    for (auto& x : v) x = Bf16Round(std::min(0.9f, std::max(-0.9f, x)));  // stays bf16-exact
    return v;
  };
  std::vector<float> in_norm = norm_w();
  std::vector<float> post_norm = norm_w();
  post_norm[static_cast<size_t>(kDead)] = -1.0f;  // (1 + w) == 0
  for (const char* L : {kL0, kL1}) {
    // Both layers share their norms, so they share their norm-fed taps (and Hessian files).
    s.Add(std::string(L) + "input_layernorm.weight", {kHidden}, in_norm);
    s.Add(std::string(L) + "post_attention_layernorm.weight", {kHidden}, post_norm);
    s.Add(std::string(L) + "mlp.gate_proj.weight", {64, kHidden}, rnd(64 * kHidden, 0.02));
    s.Add(std::string(L) + "mlp.up_proj.weight", {64, kHidden}, rnd(64 * kHidden, 0.02));
    s.Add(std::string(L) + "mlp.down_proj.weight", {kHidden, kInter}, rnd(kHidden * kInter, 0.02));
  }
  const std::string g = std::string(kL0) + "linear_attn.";
  s.Add(g + "in_proj_qkv.weight", {64, kHidden}, rnd(64 * kHidden, 0.02));
  s.Add(g + "in_proj_z.weight", {kVDim, kHidden}, rnd(kVDim * kHidden, 0.02));
  s.Add(g + "in_proj_b.weight", {1, kHidden}, rnd(kHidden, 0.02));
  s.Add(g + "in_proj_a.weight", {1, kHidden}, rnd(kHidden, 0.02));
  s.Add(g + "conv1d.weight", {64, 1, 4}, rnd(64 * 4, 0.2));
  s.Add(g + "A_log", {1}, rnd(1, 0.5));
  s.Add(g + "dt_bias", {1}, rnd(1, 0.5));
  s.Add(g + "norm.weight", {kVDim}, rnd(kVDim, 0.1));
  s.Add(g + "out_proj.weight", {kHidden, kVDim}, rnd(kHidden * kVDim, 0.02));
  const std::string a = std::string(kL1) + "self_attn.";
  s.Add(a + "q_proj.weight", {128, kHidden}, rnd(128 * kHidden, 0.02));
  s.Add(a + "k_proj.weight", {64, kHidden}, rnd(64 * kHidden, 0.02));
  s.Add(a + "v_proj.weight", {64, kHidden}, rnd(64 * kHidden, 0.02));
  s.Add(a + "o_proj.weight", {kHidden, kHeadDim}, rnd(kHidden * kHeadDim, 0.02));
  s.Add(a + "q_norm.weight", {kHeadDim}, rnd(kHeadDim, 0.1));
  s.Add(a + "k_norm.weight", {kHeadDim}, rnd(kHeadDim, 0.1));
  s.Add("model.language_model.embed_tokens.weight", {kVocab, kHidden}, rnd(kVocab * kHidden, 0.02));
  s.Add("model.language_model.norm.weight", {kHidden}, rnd(kHidden, 0.1));
  s.Add("lm_head.weight", {kVocab, kHidden}, rnd(kVocab * kHidden, 0.02));
  return s;
}

void WriteCheckpoint(const Synth& s, const fs::path& dir) {
  fs::create_directories(dir);
  nlohmann::json hdr = nlohmann::json::object(), weight_map = nlohmann::json::object();
  std::string data;
  for (const auto& t : s.tensors) {
    const uint64_t b = data.size();
    for (float x : t.v) {
      const uint16_t h = r4dx::core::FloatToBf16(x);
      data.append(reinterpret_cast<const char*>(&h), 2);
    }
    hdr[t.name] = {{"dtype", "BF16"}, {"shape", t.shape}, {"data_offsets", {b, data.size()}}};
    weight_map[t.name] = "model-00001-of-00001.safetensors";
  }
  const std::string h = hdr.dump();
  const uint64_t hl = h.size();
  std::string file(reinterpret_cast<const char*>(&hl), 8);
  file += h;
  file += data;
  WriteWhole(dir / "model-00001-of-00001.safetensors", file);
  WriteWhole(dir / "model.safetensors.index.json", nlohmann::json{{"weight_map", weight_map}}.dump(2));
  const nlohmann::json config = {
      {"architectures", {"SyntheticRmsHessianTest"}},
      {"text_config",
       {{"hidden_size", kHidden},
        {"num_hidden_layers", 2},
        {"layer_types", {"linear_attention", "full_attention"}},
        {"num_attention_heads", 1},
        {"num_key_value_heads", 1},
        {"head_dim", kHeadDim},
        {"linear_num_value_heads", 1},
        {"linear_value_head_dim", kVDim},
        {"intermediate_size", kInter}}}};
  WriteWhole(dir / "config.json", config.dump(2));
}

std::vector<std::string> InProjBases() {
  return {"text.layers.0.gdn.in_proj_qkv", "text.layers.0.gdn.in_proj_z", "text.layers.0.mlp.gate_up",
          "text.layers.1.attn.qg",         "text.layers.1.attn.k",        "text.layers.1.attn.v",
          "text.layers.1.mlp.gate_up"};
}

// Writes the Hessian directories: A (keys only -- the manifest before rms taps existed), B (plus
// rms_keys for every norm-fed in-projection), C (rms_keys for the two mlp.gate_up only). The big
// files live in A and are hard-linked (copied if that fails) into B and C.
void WriteHessianDirs(const Synth& s, const fs::path& root) {
  const fs::path A = root / "hess_A", B = root / "hess_B", C = root / "hess_C";
  for (const auto& d : {A, B, C}) fs::create_directories(d);
  nlohmann::json files = nlohmann::json::object();
  auto put = [&](const std::string& name, int64_t K, const std::vector<float>& packed) {
    const double tr = WriteHessPacked(A / name, K, kRows, packed);
    files[name] = {{"K", K}, {"rows", kRows}, {"trace", tr}};
  };
  {
    const std::vector<float> in_rms = LowRankPdPacked(101, kHidden, 16, 77);
    put("in.rms.hess", kHidden, in_rms);
    put("in.hess", kHidden, PostFromRms(in_rms, kHidden, s.Get(std::string(kL0) + "input_layernorm.weight")));
  }
  {
    // The dead channel carries a large residual value (heavy), which the post-norm capture zeroes.
    const std::vector<float> mlp_rms = LowRankPdPacked(102, kHidden, 16, kDead);
    put("mlp_in.rms.hess", kHidden, mlp_rms);
    put("mlp_in.hess", kHidden,
        PostFromRms(mlp_rms, kHidden, s.Get(std::string(kL0) + "post_attention_layernorm.weight")));
  }
  put("L00.out.hess", kVDim, LowRankPdPacked(103, kVDim, 8, -1));
  put("L01.out.hess", kHeadDim, LowRankPdPacked(104, kHeadDim, 8, -1));
  put("mlp_mid.hess", kInter, LowRankPdPacked(105, kInter, 8, -1));

  const nlohmann::json keys = {
      {"text.layers.0.gdn.in_proj_qkv", "in.hess"}, {"text.layers.0.gdn.in_proj_z", "in.hess"},
      {"text.layers.0.gdn.out_proj", "L00.out.hess"}, {"text.layers.0.mlp.gate_up", "mlp_in.hess"},
      {"text.layers.0.mlp.down", "mlp_mid.hess"},     {"text.layers.1.attn.qg", "in.hess"},
      {"text.layers.1.attn.k", "in.hess"},            {"text.layers.1.attn.v", "in.hess"},
      {"text.layers.1.attn.o", "L01.out.hess"},       {"text.layers.1.mlp.gate_up", "mlp_in.hess"},
      {"text.layers.1.mlp.down", "mlp_mid.hess"}};
  nlohmann::json files_a = files;
  files_a.erase("in.rms.hess");
  files_a.erase("mlp_in.rms.hess");
  const nlohmann::json ma = {{"format", "r4dx-hessian"}, {"version", 1}, {"files", files_a}, {"keys", keys},
                             {"tool", "tests/convert/test_rms_hessian.cpp"}};
  nlohmann::json mb = ma, mc = ma;
  mb["files"] = files;
  mc["files"] = files;
  mb["rms_keys"] = nlohmann::json::object();
  for (const auto& base : InProjBases())
    mb["rms_keys"][base] = Contains(base, "gate_up") ? "mlp_in.rms.hess" : "in.rms.hess";
  mc["rms_keys"] = {{"text.layers.0.mlp.gate_up", "mlp_in.rms.hess"},
                    {"text.layers.1.mlp.gate_up", "mlp_in.rms.hess"}};
  WriteWhole(A / "hessian.json", ma.dump(2));
  WriteWhole(B / "hessian.json", mb.dump(2));
  WriteWhole(C / "hessian.json", mc.dump(2));
  for (auto it = files.begin(); it != files.end(); ++it) {
    for (const auto& d : {B, C}) {
      std::error_code ec;
      fs::create_hard_link(A / it.key(), d / it.key(), ec);
      if (ec) fs::copy_file(A / it.key(), d / it.key(), fs::copy_options::overwrite_existing);
    }
  }
}

struct Container {
  bool ok = false;
  nlohmann::json header;
  std::string data;
  std::map<std::string, std::string> tensors;
};

Container ReadContainer(const fs::path& path) {
  Container c;
  std::string bytes;
  try {
    bytes = ReadWhole(path);
  } catch (const std::exception&) {
    return c;
  }
  if (bytes.size() < 8) return c;
  uint64_t hl = 0;
  std::memcpy(&hl, bytes.data(), 8);
  if (8 + hl > bytes.size()) return c;
  c.header = nlohmann::json::parse(bytes.substr(8, static_cast<size_t>(hl)));
  c.data = bytes.substr(static_cast<size_t>(8 + hl));
  for (auto it = c.header.begin(); it != c.header.end(); ++it) {
    if (it.key() == "__metadata__") continue;
    const uint64_t b = it.value()["data_offsets"][0].get<uint64_t>();
    const uint64_t e = it.value()["data_offsets"][1].get<uint64_t>();
    c.tensors[it.key()] = c.data.substr(static_cast<size_t>(b), static_cast<size_t>(e - b));
  }
  c.ok = true;
  return c;
}

// sha256 over every tensor (name, size, bytes; in name order): the container's DATA, independent of
// its __metadata__ (which records where the test wrote its Hessians).
std::string Digest(const Container& c) {
  std::string all;
  for (const auto& kv : c.tensors) {
    all += kv.first;
    all.push_back('\0');
    all += std::to_string(kv.second.size());
    all.push_back('\0');
    all += kv.second;
  }
  return Sha256Hex(all);
}

// Every conversion (d) runs: w4a16 only, every quantized linear LDLQ'd. `layers` 1 converts the GDN
// layer only (two K=5120 factorizations instead of four: this is the slow part of the test).
std::string CommonArgs(const fs::path& ckpt, const fs::path& out, const fs::path& hdir, int layers) {
  return "--input \"" + ckpt.u8string() + "\" --output \"" + out.u8string() + "\" --layers " +
         std::to_string(layers) +
         " --vision off --mtp off --layouts w4a16 --no-bf16 --lm-head bf16 --threads " +
         std::to_string(kThreads) + " --hessian-dir \"" + hdir.u8string() + "\" --ldlq .*";
}

// Digest() of `r4dx-convert CommonArgs(<ckpt>, <out>, <root>/hess_A, 1)` on --make-fixture's output,
// taken with r4dx-convert built from bcebb21, whose w4a16 / bf16 tensors are the ones this build
// must still write; w4a16 group 64. (The previous
// golden, of the whole file with its header, was taken at 0190d80, before rms_keys existed; that
// build's and bcebb21's tensors were identical.)
const char* const kUnrotatedGolden = "02af5a27e54276c7d6bc88fea494941ce7c90c16e380b6e9ac0ab4c9265c4e92";

std::string W4a16Bytes(const std::vector<uint32_t>& v) {
  return std::string(reinterpret_cast<const char*>(v.data()), v.size() * 4);
}

// What r4dx-convert must write for one LDLQ'd linear's w4a16 layout, from its (already folded)
// fp32 weight and the Hessian it rounds against (already in the linear's input basis).
std::pair<std::string, std::string> ExpectW4a16(const std::vector<float>& w, int64_t N, int64_t K,
                                                std::vector<float> H) {
  const LdlqFactor f = FactorHessian(std::move(H), K, 0.01f, kThreads);
  std::vector<uint8_t> q, zero;
  std::vector<float> scale;
  QuantizeInt4AsymmetricLdlq(w.data(), static_cast<int>(N), static_cast<int>(K), kW4A16Group, f,
                             kThreads, q, scale, zero);
  return {W4a16Bytes(PackW4Nibbles(q, static_cast<int>(N), static_cast<int>(K), kThreads)),
          W4a16Bytes(PackW4A16Scales(scale, zero, static_cast<int>(N), static_cast<int>(K), kW4A16Group))};
}

bool SameW4a16(const Container& c, const std::string& base, const std::pair<std::string, std::string>& want) {
  const std::string wq = base + ".w4a16.wq", wsz = W4a16WszName(base, kW4A16Group);
  return c.ok && c.tensors.count(wq) && c.tensors.count(wsz) && c.tensors.at(wq) == want.first &&
         c.tensors.at(wsz) == want.second;
}

std::vector<float> Concat(const Synth& s, std::initializer_list<std::string> names) {
  std::vector<float> w;
  for (const auto& n : names) {
    const auto& v = s.Get(n);
    w.insert(w.end(), v.begin(), v.end());
  }
  return w;
}

#if defined(R4DX_CONVERT_EXE)

// cmd.exe strips the outermost pair of quotes of a command that starts with one, so wrap it whole.
int Run(const std::string& args, const fs::path& log) {
  const std::string cmd = "\"\"" + std::string(R4DX_CONVERT_EXE) + "\" " + args + " > \"" +
                          log.u8string() + "\" 2>&1\"";
  return std::system(cmd.c_str());
}

void TestExe() {
  std::printf("---- (d) r4dx-convert on a synthetic 2-layer checkpoint ----\n");
  const fs::path root = TempDir("exe");
  const fs::path ckpt = root / "ckpt", out = root / "out.r4dx", log = root / "convert.log";
  const fs::path A = root / "hess_A", B = root / "hess_B", C = root / "hess_C";
  try {
    const auto t0 = std::chrono::steady_clock::now();
    const Synth s = MakeCheckpoint();
    WriteCheckpoint(s, ckpt);
    WriteHessianDirs(s, root);
    std::printf("     fixture written in %.1f s\n",
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());

    auto convert = [&](const std::string& label, const fs::path& hdir, int layers,
                       const std::string& extra) {
      std::error_code ec;
      fs::remove(out, ec);
      const auto t = std::chrono::steady_clock::now();
      const int rc = Run(CommonArgs(ckpt, out, hdir, layers) + extra, log);
      std::printf("     %s: rc=%d in %.1f s\n", label.c_str(), rc,
                  std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count());
      return rc;
    };
    const std::vector<std::string> rms_all = InProjBases();
    const std::string qkv = "text.layers.0.gdn.in_proj_qkv", gate0 = "text.layers.0.mlp.gate_up",
                      k1 = "text.layers.1.attn.k";
    const std::vector<float> w_qkv = s.Get(std::string(kL0) + "linear_attn.in_proj_qkv.weight");
    const std::vector<float> w_gate0 = Concat(s, {std::string(kL0) + "mlp.gate_proj.weight",
                                                  std::string(kL0) + "mlp.up_proj.weight"});
    const std::vector<float> w_k1 = s.Get(std::string(kL1) + "self_attn.k_proj.weight");
    const std::vector<float>& in_norm = s.Get(std::string(kL0) + "input_layernorm.weight");
    const std::vector<float>& post_norm = s.Get(std::string(kL0) + "post_attention_layernorm.weight");
    int64_t fk = 0;
    auto read_h = [&](const char* name) { return ReadHessFile((A / name).u8string(), &fk, nullptr); };

    // ---- unrotated (GDN layer only): A, then B (the rms files must change nothing) ----
    int rc = convert("unrotated, hess_A, --layers 1", A, 1, "");
    const Container ua = ReadContainer(out);
    Gate(rc == 0 && ua.ok, "(d) unrotated A: exit 0");
    if (rc != 0) std::printf("%s", ReadWhole(log).c_str());
    const std::string digest = ua.ok ? Digest(ua) : std::string("(unreadable)");
    if (kW4A16Group == 64) {
      Gate(digest == kUnrotatedGolden,
           "(d) unrotated A: container byte-identical (header modulo hessian_dir, and data) to the "
           "build before rms_keys existed");
      if (digest != kUnrotatedGolden) std::printf("     digest %s\n", digest.c_str());
    } else {
      std::printf("SKIP (d) unrotated golden: taken at w4a16 group 64, this build is %d (digest %s)\n",
                  kW4A16Group, digest.c_str());
    }

    rc = convert("unrotated, hess_B, --layers 1", B, 1, "");
    const Container ub = ReadContainer(out);
    Gate(rc == 0 && ub.ok && ua.ok && ub.tensors == ua.tensors,
         "(d) unrotated B (with rms_keys): every tensor byte-identical to A's");
    if (ua.ok && ub.ok) {
      nlohmann::json ha = ua.header, hb = ub.header;
      auto& ra = ha["__metadata__"]["r4dx_convert_run"];
      auto& rb = hb["__metadata__"]["r4dx_convert_run"];
      const bool sha_differs = ra["hessian_manifest_sha256"] != rb["hessian_manifest_sha256"];
      for (auto* r : {&ra, &rb}) {
        (*r)["hessian_dir"] = "";
        (*r)["hessian_manifest_sha256"] = "";
      }
      Gate(ha == hb && sha_differs && !ra.contains("ldlq_rms_linears"),
           "(d) unrotated B: header identical to A's except hessian_dir / hessian_manifest_sha256; "
           "no ldlq_rms_linears key");
    }

    // ---- q2ab against A: the dead channel has no rms Hessian -> refused while planning ----
    rc = convert("q2ab, hess_A", A, 2, " --rotate q2ab");
    {
      const std::string text = ReadWhole(log);
      Gate(rc != 0 && !fs::exists(out) && Contains(text, "text.layers.0.mlp.gate_up") &&
               Contains(text, "norm_weight[3994]") && Contains(text, "--rms-only"),
           "(d) q2ab A: refused during planning (no output file), naming text.layers.0.mlp.gate_up, "
           "norm_weight[3994] and --rms-only");
      if (rc == 0 || !Contains(text, "--rms-only")) std::printf("%s", text.c_str());
    }

    // The rotation the exe uses (default seed): Q is drawn first, so its shape args do not matter.
    RotationShape shape;
    shape.hidden = kHidden;
    shape.k_down = kInter;
    shape.k_o = kHeadDim;
    shape.k_gdn_out = kVDim;
    const RotationSet rs = GenerateRotationSet(RotationKind::kQ2ab, kDefaultRotationSeed, shape);
    auto fold = [&](std::vector<float> w, int64_t N, const std::vector<float>& norm) {
      FoldRowsQ(w, N, kHidden, norm.data(), rs.q, kThreads);
      return w;
    };

    // ---- q2ab against B, both layers: every in-projection on the rms path ----
    rc = convert("q2ab, hess_B", B, 2, " --rotate q2ab");
    const Container rb = ReadContainer(out);
    const std::string logb = ReadWhole(log);
    Gate(rc == 0 && rb.ok, "(d) q2ab B: exit 0");
    if (rc != 0) std::printf("%s", logb.c_str());
    std::pair<std::string, std::string> gate_rms;
    if (rb.ok) {
      const nlohmann::json& run = rb.header["__metadata__"]["r4dx_convert_run"];
      Gate(run.contains("ldlq_rms_linears") && run["ldlq_rms_linears"] == nlohmann::json(rms_all) &&
               run["ldlq_linears"].size() == 11 && run["rotate"] == "q2ab",
           "(d) q2ab B: r4dx_convert_run.ldlq_rms_linears == the seven in-projections (of 11 LDLQ'd)");
      Gate(Contains(logb, "ldlq: text.layers.0.gdn.in_proj_z [128,5120] rms Hessian in.rms.hess: factor") &&
               Contains(logb, "(shared tap, cached)") &&
               Contains(logb, "7 rotated in-projection(s) round against the weightless rms Hessian") &&
               !Contains(logb, "dividing the norm out"),
           "(d) q2ab B: the log names the rms file, z / k / v reuse the cached rms factor, nothing divided");
      std::vector<float> H = read_h("mlp_in.rms.hess");
      TransformHessianRmsQ(H, kHidden, rs.q, kThreads);
      gate_rms = ExpectW4a16(fold(w_gate0, 128, post_norm), 128, kHidden, std::move(H));
      Gate(SameW4a16(rb, gate0, gate_rms),
           "(d) q2ab B: dead-channel mlp.gate_up w4a16 == fold W diag(1+w) Q -> LDLQ against Q^T H_rms Q");
      std::vector<float> Hi = read_h("in.rms.hess");
      TransformHessianRmsQ(Hi, kHidden, rs.q, kThreads);
      Gate(SameW4a16(rb, k1, ExpectW4a16(fold(w_k1, 64, in_norm), 64, kHidden, std::move(Hi))),
           "(d) q2ab B: attn.k (cached rms factor) w4a16 == fold -> LDLQ against Q^T H_rms Q");
    }

    // ---- q2ab against C, GDN layer only: rms for gate_up, division for in_proj_qkv / z ----
    rc = convert("q2ab, hess_C, --layers 1", C, 1, " --rotate q2ab");
    const Container rc_ = ReadContainer(out);
    const std::string logc = ReadWhole(log);
    Gate(rc == 0 && rc_.ok, "(d) q2ab C: exit 0");
    if (rc != 0) std::printf("%s", logc.c_str());
    if (rc_.ok) {
      const nlohmann::json& run = rc_.header["__metadata__"]["r4dx_convert_run"];
      Gate(run["ldlq_rms_linears"] == nlohmann::json::array({gate0}) &&
               Contains(logc, "(2 in-projection Hessian(s) by dividing the norm out"),
           "(d) q2ab C: both paths in one run: ldlq_rms_linears == [mlp.gate_up]; in_proj_qkv / z divided");
      std::vector<float> H = read_h("in.hess");
      TransformHessianQ(H, kHidden, in_norm.data(), rs.q, kThreads);
      Gate(SameW4a16(rc_, qkv, ExpectW4a16(fold(w_qkv, 64, in_norm), 64, kHidden, std::move(H))),
           "(d) q2ab C: gdn.in_proj_qkv w4a16 == fold -> LDLQ against Q^T D^-1 H D^-1 Q (post-norm)");
      Gate(!gate_rms.first.empty() && SameW4a16(rc_, gate0, gate_rms),
           "(d) q2ab C: mlp.gate_up w4a16 byte-identical to B's (same rms Hessian)");
    }
  } catch (const std::exception& e) {
    Gate(false, std::string("(d) threw unexpectedly: ") + e.what());
  }
  std::error_code ec;
  fs::remove_all(root, ec);
}

#endif  // R4DX_CONVERT_EXE

}  // namespace

int main(int argc, char** argv) {
  if (argc == 3 && std::string(argv[1]) == "--make-fixture") {
    const fs::path root = fs::u8path(argv[2]);
    const Synth s = MakeCheckpoint();
    WriteCheckpoint(s, root / "ckpt");
    WriteHessianDirs(s, root);
    std::printf("r4dx-convert %s\n",
                CommonArgs(root / "ckpt", root / "out.r4dx", root / "hess_A", 1).c_str());
    return 0;
  }
  if (argc == 3 && std::string(argv[1]) == "--digest") {
    const Container c = ReadContainer(fs::u8path(argv[2]));
    if (!c.ok) {
      std::fprintf(stderr, "cannot read %s\n", argv[2]);
      return 1;
    }
    std::printf("%s\n", Digest(c).c_str());
    return 0;
  }
  try {
    std::printf("---- (a) TransformHessianRmsQ vs the dense Q ----\n");
    TestDenseMini(5, 16, 0x5EED);
    TestDenseMini(4, 64, 0xBEEF);
    TestDenseProduction();
    std::printf("---- (b) dead channel ----\n");
    {
      uint64_t st = 0x1234;
      const ResidualRotation qm = GenerateResidualRotation(st, 5 * 16, 16);
      TestDeadChannel(qm, 400, 37, "mini 5x16");
      RotationShape shape;
      shape.hidden = kHidden;
      const RotationSet rs = GenerateRotationSet(RotationKind::kQ2a, kDefaultRotationSeed, shape);
      TestDeadChannel(rs.q, 48, kDead, "5120 (rank-48 capture)");
    }
    TestStore();
#if defined(R4DX_CONVERT_EXE)
    TestExe();
#else
    std::printf("SKIP (d) r4dx-convert is not part of this build (R4DX_BUILD_CONVERT off)\n");
#endif
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL unexpected exception: %s\n", e.what());
    ++g_failures;
  }
  if (g_failures) {
    std::fprintf(stderr, "convert_rms_hessian: %d failure(s)\n", g_failures);
    return 1;
  }
  std::printf("convert_rms_hessian: all gates passed\n");
  return 0;
}
