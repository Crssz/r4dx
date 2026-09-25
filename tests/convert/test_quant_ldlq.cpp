// r4dx-convert --ldlq (src/convert/include/r4dx_convert/{dense_linalg,quant_ldlq,hessian_store}.hpp):
// gate G1 of docs/quant2.md section 1.3. Everything here is CPU-only.
//
//   (a) linalg. SubMatMul / SubMatMulSerial, CholeskyLower and InvertLower against naive
//       double-precision references at n in {1, 7, 128, 129, 300}, on shapes that are not multiples
//       of the 8-row / 32-column register tile, and with padded row strides (the padding must come
//       back untouched). The error is Frobenius-relative and must be <= 1e-4. Thread count 1 vs 16
//       must give BIT-IDENTICAL output. So must the AVX-512 path vs linalg::ForceScalar(), when the CPU
//       has AVX-512F; otherwise that half is skipped and says so. A non-SPD matrix and a NaN matrix
//       must make CholeskyLower return false. The GFLOP/s of SubMatMul at 128x4096x128 (64 calls,
//       16 threads) is printed for information and is not gated.
//   (b) FactorHessian. U is upper triangular with a positive, finite diagonal. With
//       H_d = H + damp*mean(diag H)*I, ||H_d U^T U - I||_F / ||I||_F must be <= 1e-3 at K = 256.
//       damp_used / retries / diag_h say what happened. A rank-deficient H with dead channels
//       succeeds at damp 0.01, and at damp 0 it succeeds only through a retry (an exactly-zero pivot
//       is guaranteed to fail the first attempt). A NaN anywhere in H, or a wrong-sized H, throws.
//   (c) identity Hessian. With H = I and damp = 0, U is exactly I, so the error feedback is exactly
//       zero. QuantizeInt4AsymmetricLdlq / QuantizeInt4Pinned8Ldlq must then equal, byte for byte,
//       a per-(row, group) application of SearchInt4AsymGroup / SearchInt4Pinned8Group with
//       refit = false and unit weights, at groups 32, 64 and 128. QuantizeMxfp4Ldlq must equal
//       SearchMxfp4Group, which is also QuantizeMxfp4Search with no imatrix. Rows span a partial
//       128-row tile, and the matrix carries an all-zero group, a constant group and an outlier, so
//       the degenerate branches are covered too.
//   (d) proxy loss. X has strongly correlated columns: AR(1) along the channels, a shared low-rank
//       part, lognormal channel scales and a few 10x heavy channels. rows = 4096, K in {256, 512},
//       H = X^T X / rows, W random [256, K]. proxy(Wq) = tr((W - deq Wq) H (W - deq Wq)^T), in
//       double. The LDLQ proxy must be <= 0.2x that of the search quantizer weighted by diag(H)
//       (the --quant search --imatrix path; docs/quant2.md's G1 floor is 0.8, this data leaves the
//       correct loop at ~0.1), for w4a16 at groups 32, 64 and 128 (32: a per-tensor group,
//       docs/quant2.md section 5), w4a8 and mxfp4. RTN is printed
//       alongside for scale. The ratio cannot see a broken lazy-block update on its own, so each
//       layout is also checked against an UNBLOCKED per-row GPTQ reference (full-row feedback after
//       every column, group parameters chosen from the current weights with the no-refit search;
//       K = 512 is 4 column blocks): codes may differ at <= 0.1% (fp summation order) and the proxy
//       by <= 0.5%.
//   (e) determinism. LDLQ output bytes are identical at nthreads 1, 3 and 16 for all three layouts
//       (w4a16 at groups 32, 64 and 128), and with the scalar path forced. So are FactorHessian's U and diag_h.
//   (f) HessianStore on tests/convert/fixtures/hess_small, which
//       tools/reference/hessian_capture.py --write-fixture wrote. Checked against expected.json:
//       the file size and sha256, K / rows / trace, the fp64 sum of the upper triangle (to 1e-6
//       relative), min / max diag and exact sample entries. H must be exactly symmetric.
//       Has / KOf / File for t.a, t.b and t.c, and ManifestSha256. Factor must share one cached
//       factorization between t.b and t.c (proved by a marker planted in the cached object, not by
//       its address, which is always the same member), rebuild it on a different damp or file, and
//       throw on a K mismatch. Corrupted copies (magic, flags, trace, truncation) are rejected, as
//       are manifests with a bad version, a key naming an unlisted file, or a K the file header
//       contradicts.
//   (g) packers. PackW4Nibbles / PackW4A16Scales / PackW4A8Scales / PackMxfp4Wq / PackMxfp4Ws accept
//       the (d) LDLQ outputs. The bytes are then decoded by the kernels' own literal indexing (the
//       same derivation as test_kernel_decode.cpp) and must give back exactly the LDLQ codes, zeros
//       and exponents. The kernel-side dequant (f16 scales, mxfp4 Wref / dsh fold) must agree with
//       the fp32 dequant to f16 precision (exactly, for mxfp4), and the proxy loss recomputed from
//       the packed bytes must agree with (d)'s to 2%.
//
// R4DX_CONVERT_FIXTURES_DIR is injected by tests/convert/CMakeLists.txt as an absolute path.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/dense_linalg.hpp"
#include "r4dx_convert/hessian_store.hpp"
#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/quant_ldlq.hpp"
#include "r4dx_convert/quant_mxfp4.hpp"
#include "r4dx_convert/quant_search.hpp"
#include "r4dx_convert/sha256.hpp"
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

bool SameBytes(const std::string& label, const void* a, size_t na, const void* b, size_t nb) {
  if (na != nb) {
    std::fprintf(stderr, "FAIL %s: size mismatch %zu vs %zu bytes\n", label.c_str(), na, nb);
    ++g_failures;
    return false;
  }
  const uint8_t* pa = static_cast<const uint8_t*>(a);
  const uint8_t* pb = static_cast<const uint8_t*>(b);
  for (size_t i = 0; i < na; ++i) {
    if (pa[i] != pb[i]) {
      std::fprintf(stderr, "FAIL %s: first mismatch at byte %zu: 0x%02x vs 0x%02x\n", label.c_str(),
                   i, pa[i], pb[i]);
      ++g_failures;
      return false;
    }
  }
  std::printf("OK   %s (%zu bytes identical)\n", label.c_str(), na);
  return true;
}

template <typename T>
bool SameVec(const std::string& label, const std::vector<T>& a, const std::vector<T>& b) {
  return SameBytes(label, a.data(), a.size() * sizeof(T), b.data(), b.size() * sizeof(T));
}

// Runs fn, which must throw std::exception. The message is printed so a log shows WHICH check fired.
template <typename Fn>
bool ExpectThrow(const std::string& label, Fn&& fn) {
  try {
    fn();
  } catch (const std::exception& e) {
    std::printf("OK   %s (threw: %s)\n", label.c_str(), e.what());
    return true;
  }
  std::fprintf(stderr, "FAIL %s: did not throw\n", label.c_str());
  ++g_failures;
  return false;
}

// Restores linalg::ForceScalar() on scope exit, so a failing check cannot leak the scalar path into
// the checks that follow it.
struct ScopedForceScalar {
  bool prev;
  explicit ScopedForceScalar(bool on) : prev(linalg::ForceScalar()) { linalg::ForceScalar() = on; }
  ~ScopedForceScalar() { linalg::ForceScalar() = prev; }
};

std::vector<float> RandomNormal(size_t n, uint32_t seed, float sd = 1.0f) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> d(0.0f, sd);
  std::vector<float> v(n);
  for (auto& x : v) x = d(rng);
  return v;
}

// ||got - ref||_F / ||ref||_F over an M x N window, got / ref with their own row strides.
double RelErr(const float* got, int64_t ldg, const double* ref, int64_t ldr, int64_t M, int64_t N) {
  double num = 0.0, den = 0.0;
  for (int64_t i = 0; i < M; ++i) {
    for (int64_t j = 0; j < N; ++j) {
      const double r = ref[i * ldr + j];
      const double d = static_cast<double>(got[i * ldg + j]) - r;
      num += d * d;
      den += r * r;
    }
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

double RelL2(const std::vector<double>& got, const std::vector<double>& ref) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double d = got[i] - ref[i];
    num += d * d;
    den += ref[i] * ref[i];
  }
  return den > 0.0 ? std::sqrt(num / den) : std::sqrt(num);
}

// ---- references and generators -----------------------------------------------------------------

// Symmetric positive definite, well conditioned: G G^T / n + 0.5 I (eigenvalues ~ 0.5 .. 4.5).
std::vector<float> MakeSpd(int64_t n, uint32_t seed) {
  const std::vector<float> G = RandomNormal(static_cast<size_t>(n * n), seed);
  std::vector<float> A(static_cast<size_t>(n * n));
  for (int64_t i = 0; i < n; ++i) {
    for (int64_t j = 0; j <= i; ++j) {
      double s = 0.0;
      for (int64_t t = 0; t < n; ++t)
        s += static_cast<double>(G[i * n + t]) * static_cast<double>(G[j * n + t]);
      s = s / static_cast<double>(n) + (i == j ? 0.5 : 0.0);
      A[i * n + j] = A[j * n + i] = static_cast<float>(s);
    }
  }
  return A;
}

// Textbook right-looking Cholesky in double on the float input. Returns an empty vector on failure.
std::vector<double> NaiveCholesky(const std::vector<float>& A, int64_t n) {
  std::vector<double> L(static_cast<size_t>(n * n), 0.0);
  for (int64_t j = 0; j < n; ++j) {
    double s = A[j * n + j];
    for (int64_t t = 0; t < j; ++t) s -= L[j * n + t] * L[j * n + t];
    if (!(s > 0.0)) return {};
    const double ljj = std::sqrt(s);
    L[j * n + j] = ljj;
    for (int64_t i = j + 1; i < n; ++i) {
      double v = A[i * n + j];
      for (int64_t t = 0; t < j; ++t) v -= L[i * n + t] * L[j * n + t];
      L[i * n + j] = v / ljj;
    }
  }
  return L;
}

// Inverse of a lower-triangular matrix, column by column forward substitution, in double.
std::vector<double> NaiveInvLower(const std::vector<double>& L, int64_t n) {
  std::vector<double> X(static_cast<size_t>(n * n), 0.0);
  for (int64_t j = 0; j < n; ++j) {
    X[j * n + j] = 1.0 / L[j * n + j];
    for (int64_t i = j + 1; i < n; ++i) {
      double s = 0.0;
      for (int64_t t = j; t < i; ++t) s += L[i * n + t] * X[t * n + j];
      X[i * n + j] = -s / L[i * n + i];
    }
  }
  return X;
}

// H = X^T X / rows (X row-major [rows, K]), accumulated in double, stored fp32 and exactly symmetric,
// the way hessian_capture.py stores it.
std::vector<float> HessianOf(const std::vector<float>& X, int64_t rows, int64_t K, int nthreads) {
  std::vector<double> XT(static_cast<size_t>(K * rows));
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t k = 0; k < K; ++k) XT[k * rows + r] = X[r * K + k];
  std::vector<float> H(static_cast<size_t>(K * K));
  ParallelFor(0, K, nthreads, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      const double* xi = XT.data() + i * rows;
      for (int64_t j = i; j < K; ++j) {
        const double* xj = XT.data() + j * rows;
        double s = 0.0;
        for (int64_t r = 0; r < rows; ++r) s += xi[r] * xj[r];
        H[i * K + j] = H[j * K + i] = static_cast<float>(s / static_cast<double>(rows));
      }
    }
  });
  return H;
}

// Mildly mixed activations for (b): X = Z (I + 0.5 G / sqrt(K)). Conditioning stays in the tens, so
// the 1e-3 residual gate measures the factorization, not the input's condition number.
std::vector<float> MixedX(int64_t rows, int64_t K, uint32_t seed) {
  const std::vector<float> Z = RandomNormal(static_cast<size_t>(rows * K), seed);
  const std::vector<float> G = RandomNormal(static_cast<size_t>(K * K), seed + 1);
  const double c = 0.5 / std::sqrt(static_cast<double>(K));
  std::vector<float> X(static_cast<size_t>(rows * K));
  std::vector<double> acc(static_cast<size_t>(K));
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t k = 0; k < K; ++k) acc[k] = Z[r * K + k];  // the I term
    for (int64_t t = 0; t < K; ++t) {
      const double z = static_cast<double>(Z[r * K + t]) * c;
      for (int64_t k = 0; k < K; ++k) acc[k] += z * static_cast<double>(G[t * K + k]);
    }
    for (int64_t k = 0; k < K; ++k) X[r * K + k] = static_cast<float>(acc[k]);
  }
  return X;
}

// Strongly correlated activations for (d) / (e) / (g). AR(1) along the channel axis (rho = 0.9;
// neighbouring channels, i.e. members of the same group, move together), plus an 8-factor shared
// low-rank part (long-range correlation across groups and blocks), times lognormal per-channel scales
// (sigma 0.5), with every 64th channel 10x heavy. The heavy channels are the outlier-channel pattern
// diag(H) already rewards, so the search baseline is a fair, imatrix-aware opponent.
std::vector<float> CorrelatedX(int64_t rows, int64_t K, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::lognormal_distribution<float> ld(0.0f, 0.5f);
  std::vector<float> chan(static_cast<size_t>(K));
  for (int64_t k = 0; k < K; ++k) chan[k] = ld(rng) * (k % 64 == 17 ? 10.0f : 1.0f);
  constexpr int kFactors = 8;
  std::vector<float> B(static_cast<size_t>(kFactors * K));
  for (auto& b : B) b = nd(rng);
  const float rho = 0.9f;
  const float inn = std::sqrt(1.0f - rho * rho);
  std::vector<float> X(static_cast<size_t>(rows * K));
  for (int64_t r = 0; r < rows; ++r) {
    float u[kFactors];
    for (float& f : u) f = nd(rng);
    float ar = nd(rng);
    for (int64_t k = 0; k < K; ++k) {
      if (k > 0) ar = rho * ar + inn * nd(rng);
      float lr = 0.0f;
      for (int f = 0; f < kFactors; ++f) lr += u[f] * B[f * K + k];
      X[r * K + k] = (ar + 0.5f * lr) * chan[k];
    }
  }
  return X;
}

std::vector<double> ToDouble(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

// tr((W - D) H (W - D)^T) in double: sum over rows of d^T H d. Rows are reduced in a fixed order.
double Proxy(const std::vector<float>& W, const std::vector<double>& deq, const std::vector<double>& Hd,
             int N, int K) {
  std::vector<double> per_row(static_cast<size_t>(N));
  ParallelFor(0, N, 16, [&](int64_t r0, int64_t r1) {
    std::vector<double> d(static_cast<size_t>(K));
    for (int64_t r = r0; r < r1; ++r) {
      for (int k = 0; k < K; ++k) d[k] = static_cast<double>(W[r * K + k]) - deq[r * K + k];
      double s = 0.0;
      for (int i = 0; i < K; ++i) {
        const double* h = Hd.data() + static_cast<size_t>(i) * K;
        double t = 0.0;
        for (int j = 0; j < K; ++j) t += h[j] * d[j];
        s += d[i] * t;
      }
      per_row[r] = s;
    }
  });
  double total = 0.0;
  for (double v : per_row) total += v;
  return total;
}

// fp32-scale dequantization of each layout's quantizer outputs (the error model the quantizers use).
std::vector<double> DeqAsym(const std::vector<uint8_t>& q, const std::vector<float>& scale,
                            const std::vector<uint8_t>& zero, int N, int K, int group) {
  const int gpr = K / group;
  std::vector<double> out(static_cast<size_t>(N) * K);
  for (int r = 0; r < N; ++r)
    for (int k = 0; k < K; ++k) {
      const size_t gi = static_cast<size_t>(r) * gpr + k / group;
      out[static_cast<size_t>(r) * K + k] =
          static_cast<double>(scale[gi]) *
          (static_cast<double>(q[static_cast<size_t>(r) * K + k]) - static_cast<double>(zero[gi]));
    }
  return out;
}

std::vector<double> DeqPinned8(const std::vector<uint8_t>& q, const std::vector<float>& scale, int N,
                               int K, int group) {
  const int gpr = K / group;
  std::vector<double> out(static_cast<size_t>(N) * K);
  for (int r = 0; r < N; ++r)
    for (int k = 0; k < K; ++k) {
      const size_t gi = static_cast<size_t>(r) * gpr + k / group;
      out[static_cast<size_t>(r) * K + k] =
          static_cast<double>(scale[gi]) *
          (static_cast<double>(q[static_cast<size_t>(r) * K + k]) - 8.0);
    }
  return out;
}

double E2M1Value(uint8_t code, int raw) {
  const double mag = static_cast<double>(kE2M1Magnitude[code & 0x7]);
  return ((code & 0x8) ? -mag : mag) * std::ldexp(1.0, raw - 127);
}

std::vector<double> DeqMxfp4(const Mxfp4Quantized& m, int N, int K, int group) {
  const int gpr = K / group;
  std::vector<double> out(static_cast<size_t>(N) * K);
  for (int r = 0; r < N; ++r)
    for (int k = 0; k < K; ++k) {
      const uint8_t byte = m.packed[static_cast<size_t>(r) * (K / 2) + k / 2];
      const uint8_t code = (k % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
      out[static_cast<size_t>(r) * K + k] =
          E2M1Value(code, m.escale[static_cast<size_t>(r) * gpr + k / group]);
    }
  return out;
}

// ---- (a) linalg ---------------------------------------------------------------------------------

void TestSubMatMul(int64_t M, int64_t N, int64_t K, uint32_t seed) {
  // Padded strides: a masked-tail bug that writes past column N shows up as changed padding.
  const int64_t lda = K + 3, ldb = N + 5, ldc = N + 2;
  const std::vector<float> A = RandomNormal(static_cast<size_t>(M * lda), seed);
  const std::vector<float> B = RandomNormal(static_cast<size_t>(std::max<int64_t>(K, 1) * ldb), seed + 1);
  const std::vector<float> C0 = RandomNormal(static_cast<size_t>(M * ldc), seed + 2);
  std::vector<double> ref(static_cast<size_t>(M * N));
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = 0; j < N; ++j) {
      double s = 0.0;
      for (int64_t k = 0; k < K; ++k)
        s += static_cast<double>(A[i * lda + k]) * static_cast<double>(B[k * ldb + j]);
      ref[i * N + j] = static_cast<double>(C0[i * ldc + j]) - s;
    }

  const std::string label = Fmt("(a) SubMatMul M=%lld N=%lld K=%lld", static_cast<long long>(M),
                                static_cast<long long>(N), static_cast<long long>(K));
  std::vector<float> c1 = C0;
  linalg::SubMatMulSerial(M, N, K, A.data(), lda, B.data(), ldb, c1.data(), ldc);
  std::vector<float> c16 = C0;
  linalg::SubMatMul(M, N, K, A.data(), lda, B.data(), ldb, c16.data(), ldc, 16);

  const double e = RelErr(c1.data(), ldc, ref.data(), N, M, N);
  Gate(e <= 1e-4, label + Fmt(": rel err %.3g <= 1e-4", e));
  bool pad_ok = true;
  for (int64_t i = 0; i < M; ++i)
    for (int64_t j = N; j < ldc; ++j)
      if (std::memcmp(&c1[i * ldc + j], &C0[i * ldc + j], sizeof(float)) != 0 ||
          std::memcmp(&c16[i * ldc + j], &C0[i * ldc + j], sizeof(float)) != 0)
        pad_ok = false;
  Gate(pad_ok, label + ": row padding past N untouched");
  SameVec(label + ": 16 threads == serial", c16, c1);
  if (linalg::CpuHasAvx512f()) {
    ScopedForceScalar s(true);
    std::vector<float> cs = C0;
    linalg::SubMatMulSerial(M, N, K, A.data(), lda, B.data(), ldb, cs.data(), ldc);
    SameVec(label + ": scalar path == AVX-512 path", cs, c1);
  }
}

void TestCholeskyInverse(int64_t n, uint32_t seed) {
  const std::string tag = Fmt("n=%lld", static_cast<long long>(n));
  const std::vector<float> A = MakeSpd(n, seed);
  const std::vector<double> Lref = NaiveCholesky(A, n);
  if (Lref.empty()) {
    Gate(false, "(a) reference Cholesky of the test matrix " + tag);
    return;
  }

  // CholeskyLower
  std::vector<float> L16 = A;
  const bool ok16 = linalg::CholeskyLower(L16.data(), n, 16);
  linalg::ZeroStrictUpper(L16.data(), n);
  Gate(ok16, "(a) CholeskyLower " + tag + " succeeds on an SPD matrix");
  const double e = RelErr(L16.data(), n, Lref.data(), n, n, n);
  Gate(e <= 1e-4, "(a) CholeskyLower " + tag + Fmt(": rel err %.3g <= 1e-4", e));
  std::vector<float> L1 = A;
  linalg::CholeskyLower(L1.data(), n, 1);
  linalg::ZeroStrictUpper(L1.data(), n);
  SameVec("(a) CholeskyLower " + tag + ": 1 thread == 16 threads", L1, L16);
  if (linalg::CpuHasAvx512f()) {
    ScopedForceScalar s(true);
    std::vector<float> Ls = A;
    linalg::CholeskyLower(Ls.data(), n, 16);
    linalg::ZeroStrictUpper(Ls.data(), n);
    SameVec("(a) CholeskyLower " + tag + ": scalar path == AVX-512 path", Ls, L16);
  }

  // InvertLower, on the fp32 rounding of the reference factor (so only the inverse is measured).
  std::vector<float> Lf(static_cast<size_t>(n * n), 0.0f);
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j <= i; ++j) Lf[i * n + j] = static_cast<float>(Lref[i * n + j]);
  const std::vector<double> Iref = NaiveInvLower(ToDouble(Lf), n);
  std::vector<float> X16 = Lf;
  linalg::InvertLower(X16.data(), n, 16);
  const double ei = RelErr(X16.data(), n, Iref.data(), n, n, n);
  Gate(ei <= 1e-4, "(a) InvertLower " + tag + Fmt(": rel err %.3g <= 1e-4", ei));
  bool lower = true;
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = i + 1; j < n; ++j)
      if (X16[i * n + j] != 0.0f) lower = false;
  Gate(lower, "(a) InvertLower " + tag + ": result stays lower triangular");
  std::vector<float> X1 = Lf;
  linalg::InvertLower(X1.data(), n, 1);
  SameVec("(a) InvertLower " + tag + ": 1 thread == 16 threads", X1, X16);
  if (linalg::CpuHasAvx512f()) {
    ScopedForceScalar s(true);
    std::vector<float> Xs = Lf;
    linalg::InvertLower(Xs.data(), n, 16);
    SameVec("(a) InvertLower " + tag + ": scalar path == AVX-512 path", Xs, X16);
  }
}

void TestLinalg() {
  std::printf("---- (a) linalg (AVX-512F %s) ----\n",
              linalg::CpuHasAvx512f() ? "present: scalar-vs-AVX-512 bit identity is gated"
                                      : "ABSENT: scalar-vs-AVX-512 bit identity SKIPPED");
  struct Shape {
    int64_t M, N, K;
  };
  const Shape shapes[] = {{1, 1, 1},     {7, 7, 7},    {128, 128, 128}, {129, 129, 129},
                          {300, 300, 300}, {13, 45, 37}, {9, 33, 17},     {8, 32, 64},
                          {5, 17, 3},    {31, 70, 129}, {4, 9, 0}};
  uint32_t seed = 100;
  for (const Shape& s : shapes) TestSubMatMul(s.M, s.N, s.K, seed += 10);

  for (int64_t n : {1, 7, 45, 128, 129, 200, 300}) TestCholeskyInverse(n, static_cast<uint32_t>(1000 + n));

  {  // failure reporting
    const int64_t n = 129;
    std::vector<float> A = MakeSpd(n, 77);
    A[5 * n + 5] = -1.0f;
    Gate(!linalg::CholeskyLower(A.data(), n, 16), "(a) CholeskyLower returns false on a non-SPD matrix");
    const int64_t m = 300;
    std::vector<float> B = MakeSpd(m, 78);
    B[200 * m + 3] = B[3 * m + 200] = std::numeric_limits<float>::quiet_NaN();
    Gate(!linalg::CholeskyLower(B.data(), m, 16), "(a) CholeskyLower returns false on a NaN matrix");
  }

  {  // informational throughput, the LDLQ block-update shape
    const int64_t M = 128, N = 4096, K = 128;
    const std::vector<float> A = RandomNormal(static_cast<size_t>(M * K), 5);
    const std::vector<float> B = RandomNormal(static_cast<size_t>(K * N), 6);
    std::vector<float> C(static_cast<size_t>(M * N), 0.0f);
    linalg::SubMatMul(M, N, K, A.data(), K, B.data(), N, C.data(), N, 16);  // warm-up
    const auto t0 = std::chrono::steady_clock::now();
    constexpr int kCalls = 64;
    for (int it = 0; it < kCalls; ++it)
      linalg::SubMatMul(M, N, K, A.data(), K, B.data(), N, C.data(), N, 16);
    const double sec =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double flops = 2.0 * static_cast<double>(M) * N * K * kCalls;
    std::printf("INFO (a) SubMatMul %lldx%lldx%lld x %d calls, 16 threads, %s path: %.1f GFLOP/s "
                "(%.3f s; not gated)\n",
                static_cast<long long>(M), static_cast<long long>(N), static_cast<long long>(K),
                kCalls, linalg::UseAvx512() ? "AVX-512" : "scalar",
                sec > 0.0 ? flops / sec / 1e9 : 0.0, sec);
  }
}

// ---- (b) FactorHessian --------------------------------------------------------------------------

// U upper triangular (strict lower EXACTLY zero), finite everywhere, positive diagonal.
bool FactorShapeOk(const LdlqFactor& f, int64_t K) {
  if (f.K != K || f.U.size() != static_cast<size_t>(K * K) || f.diag_h.size() != static_cast<size_t>(K))
    return false;
  for (int64_t i = 0; i < K; ++i) {
    const float* u = f.U.data() + i * K;
    if (!(u[i] > 0.0f) || !std::isfinite(u[i])) return false;
    for (int64_t j = 0; j < i; ++j)
      if (u[j] != 0.0f) return false;
    for (int64_t j = i + 1; j < K; ++j)
      if (!std::isfinite(u[j])) return false;
  }
  return true;
}

// ||H_d U^T U - I||_F / ||I||_F, H_d rebuilt with the exact float shift FactorHessian applies.
double FactorResidual(const std::vector<float>& H, const LdlqFactor& f, int64_t K) {
  double dsum = 0.0;
  for (int64_t i = 0; i < K; ++i) dsum += static_cast<double>(H[i * K + i]);
  const double mean = dsum / static_cast<double>(K);
  const float add = static_cast<float>(static_cast<double>(f.damp_used) * mean);
  std::vector<double> Hd = ToDouble(H);
  for (int64_t i = 0; i < K; ++i) Hd[i * K + i] = static_cast<double>(H[i * K + i] + add);

  std::vector<double> UtU(static_cast<size_t>(K * K), 0.0);  // (U^T U)[i][j] = sum_t U[t][i] U[t][j]
  ParallelFor(0, K, 16, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i)
      for (int64_t j = 0; j < K; ++j) {
        double s = 0.0;
        const int64_t tmax = std::min(i, j);
        for (int64_t t = 0; t <= tmax; ++t)
          s += static_cast<double>(f.U[t * K + i]) * static_cast<double>(f.U[t * K + j]);
        UtU[i * K + j] = s;
      }
  });
  std::vector<double> row_sq(static_cast<size_t>(K));
  ParallelFor(0, K, 16, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      double acc = 0.0;
      for (int64_t j = 0; j < K; ++j) {
        double s = 0.0;
        for (int64_t t = 0; t < K; ++t) s += Hd[i * K + t] * UtU[t * K + j];
        const double v = s - (i == j ? 1.0 : 0.0);
        acc += v * v;
      }
      row_sq[i] = acc;
    }
  });
  double total = 0.0;
  for (double v : row_sq) total += v;
  return std::sqrt(total) / std::sqrt(static_cast<double>(K));
}

void TestFactorHessian() {
  std::printf("---- (b) FactorHessian ----\n");
  const int64_t K = 256;
  const std::vector<float> H = HessianOf(MixedX(1024, K, 11), 1024, K, 16);

  {
    const LdlqFactor f = FactorHessian(H, K, 0.01f, 16);
    Gate(FactorShapeOk(f, K), "(b) K=256 damp=0.01: U upper triangular, finite, positive diagonal");
    Gate(f.damp_used == 0.01f && f.retries == 0,
         Fmt("(b) K=256 damp=0.01: damp_used=%g retries=%d (no retry needed)",
             static_cast<double>(f.damp_used), f.retries));
    bool diag_ok = true;
    for (int64_t i = 0; i < K; ++i)
      if (std::memcmp(&f.diag_h[i], &H[i * K + i], sizeof(float)) != 0) diag_ok = false;
    Gate(diag_ok, "(b) diag_h is the UNDAMPED diag(H), bit for bit");
    const double res = FactorResidual(H, f, K);
    Gate(res <= 1e-3, Fmt("(b) K=256: ||H_d U^T U - I||_F / ||I||_F = %.3g <= 1e-3", res));
  }
  {  // the strongly correlated (d) input, informational: how far fp32 goes on a harder matrix
    const std::vector<float> Hc = HessianOf(CorrelatedX(4096, K, 21), 4096, K, 16);
    const LdlqFactor f = FactorHessian(Hc, K, 0.01f, 16);
    std::printf("INFO (b) K=256 correlated (d) input, damp=0.01: residual %.3g, retries %d\n",
                FactorResidual(Hc, f, K), f.retries);
  }

  // Rank-deficient: fewer rows than K, and channels 10 and 200 dead (exactly zero row/column in H).
  std::vector<float> Xs = MixedX(128, K, 31);
  for (int64_t r = 0; r < 128; ++r) Xs[r * K + 10] = Xs[r * K + 200] = 0.0f;
  const std::vector<float> Hs = HessianOf(Xs, 128, K, 16);
  try {
    const LdlqFactor f = FactorHessian(Hs, K, 0.01f, 16);
    Gate(FactorShapeOk(f, K),
         Fmt("(b) rank-deficient H (128 rows, 2 dead channels) at damp=0.01 succeeds "
             "(damp_used=%g, retries=%d) with a valid U",
             static_cast<double>(f.damp_used), f.retries));
  } catch (const std::exception& e) {
    Gate(false, std::string("(b) rank-deficient H at damp=0.01 threw: ") + e.what());
  }
  try {
    // A dead channel is an exactly-zero pivot, so damp 0 MUST fail once and recover by retrying.
    const LdlqFactor f = FactorHessian(Hs, K, 0.0f, 16);
    Gate(FactorShapeOk(f, K) && f.retries >= 1 && f.damp_used > 0.0f,
         Fmt("(b) rank-deficient H at damp=0 recovers by retry (damp_used=%g, retries=%d)",
             static_cast<double>(f.damp_used), f.retries));
  } catch (const std::exception& e) {
    Gate(false, std::string("(b) rank-deficient H at damp=0 threw: ") + e.what());
  }

  {
    std::vector<float> Hn = H;
    Hn[3 * K + 5] = Hn[5 * K + 3] = std::numeric_limits<float>::quiet_NaN();
    ExpectThrow("(b) NaN off-diagonal in H throws",
                [&]() { FactorHessian(Hn, K, 0.01f, 16); });
    std::vector<float> Hd = H;
    Hd[7 * K + 7] = std::numeric_limits<float>::quiet_NaN();
    ExpectThrow("(b) NaN diagonal in H throws", [&]() { FactorHessian(Hd, K, 0.01f, 16); });
    ExpectThrow("(b) wrong-sized H throws",
                [&]() { FactorHessian(std::vector<float>(10, 1.0f), K, 0.01f, 16); });
  }
}

// ---- (c) identity Hessian == the no-refit group search --------------------------------------------

void TestIdentity() {
  std::printf("---- (c) H = I, damp = 0 ----\n");
  const int N = 144, K = 256;  // 144 rows = one full 128-row tile + a partial one
  std::vector<float> w = RandomNormal(static_cast<size_t>(N) * K, 3);
  for (int k = 0; k < 128; ++k) w[3 * K + k] = 0.0f;         // all-zero groups (every group size)
  for (int k = 128; k < 256; ++k) w[5 * K + k] = -0.75f;     // constant groups
  w[7 * K + 40] = 25.0f;                                     // outlier

  std::vector<float> I(static_cast<size_t>(K) * K, 0.0f);
  for (int i = 0; i < K; ++i) I[static_cast<size_t>(i) * K + i] = 1.0f;
  const LdlqFactor f = FactorHessian(I, K, 0.0f, 4);
  SameVec("(c) FactorHessian(I, damp=0).U == I exactly", f.U, I);
  Gate(f.damp_used == 0.0f && f.retries == 0, "(c) FactorHessian(I, damp=0): no damping, no retry");

  for (int group : {32, 64, 128}) {
    const int gpr = K / group;
    const std::string tag = Fmt(" g%d", group);
    {  // w4a16
      std::vector<uint8_t> q, zero, rq_ref(static_cast<size_t>(N) * K), rz_ref(static_cast<size_t>(N) * gpr);
      std::vector<float> scale, rs_ref(static_cast<size_t>(N) * gpr);
      QuantizeInt4AsymmetricLdlq(w.data(), N, K, group, f, 4, q, scale, zero);
      std::vector<int32_t> rq(static_cast<size_t>(group));
      std::vector<uint8_t> best(static_cast<size_t>(group));
      for (int r = 0; r < N; ++r)
        for (int g = 0; g < gpr; ++g) {
          const size_t gi = static_cast<size_t>(r) * gpr + g;
          const size_t off = static_cast<size_t>(r) * K + static_cast<size_t>(g) * group;
          int zp = 0;
          // nullptr = unit weights, the same 1.0f LDLQ reads from diag_h here
          SearchInt4AsymGroup(w.data() + off, nullptr, group, /*refit=*/false, rq.data(), best.data(),
                              rq_ref.data() + off, &rs_ref[gi], &zp);
          rz_ref[gi] = static_cast<uint8_t>(zp);
        }
      SameVec("(c) w4a16" + tag + " q == SearchInt4AsymGroup(refit=false)", q, rq_ref);
      SameVec("(c) w4a16" + tag + " scale", scale, rs_ref);
      SameVec("(c) w4a16" + tag + " zero", zero, rz_ref);
    }
    {  // w4a8
      std::vector<uint8_t> q, rq_ref(static_cast<size_t>(N) * K);
      std::vector<float> scale, rs_ref(static_cast<size_t>(N) * gpr);
      QuantizeInt4Pinned8Ldlq(w.data(), N, K, group, f, 4, q, scale);
      std::vector<uint8_t> best(static_cast<size_t>(group));
      for (int r = 0; r < N; ++r)
        for (int g = 0; g < gpr; ++g) {
          const size_t gi = static_cast<size_t>(r) * gpr + g;
          const size_t off = static_cast<size_t>(r) * K + static_cast<size_t>(g) * group;
          SearchInt4Pinned8Group(w.data() + off, nullptr, group, /*refit=*/false, best.data(),
                                 rq_ref.data() + off, &rs_ref[gi]);
        }
      SameVec("(c) w4a8" + tag + " q == SearchInt4Pinned8Group(refit=false)", q, rq_ref);
      SameVec("(c) w4a8" + tag + " scale", scale, rs_ref);
    }
  }

  {  // mxfp4 (its group is its own constant, 32)
    const int group = kMxfp4Group, gpr = K / group;
    const Mxfp4Quantized m = QuantizeMxfp4Ldlq(w.data(), N, K, group, f, 4);
    Mxfp4Quantized ref;
    ref.packed.assign(static_cast<size_t>(N) * (K / 2), 0);
    ref.escale.assign(static_cast<size_t>(N) * gpr, 0);
    ref.wref.assign(static_cast<size_t>(N), 0);
    std::vector<uint8_t> codes(static_cast<size_t>(group)), best(static_cast<size_t>(group));
    for (int r = 0; r < N; ++r) {
      for (int g = 0; g < gpr; ++g) {
        int raw = 0;
        SearchMxfp4Group(w.data() + static_cast<size_t>(r) * K + static_cast<size_t>(g) * group,
                         nullptr, group, codes.data(), best.data(), &raw);
        ref.escale[static_cast<size_t>(r) * gpr + g] = static_cast<uint8_t>(raw);
        ref.wref[r] = std::max(ref.wref[r], static_cast<uint8_t>(raw));
        for (int k = 0; k < group; ++k) {
          const int kk = g * group + k;
          uint8_t& byte = ref.packed[static_cast<size_t>(r) * (K / 2) + kk / 2];
          byte = (kk % 2 == 0) ? static_cast<uint8_t>((byte & 0xF0u) | (best[k] & 0x0Fu))
                               : static_cast<uint8_t>((byte & 0x0Fu) | ((best[k] & 0x0Fu) << 4));
        }
      }
    }
    SameVec("(c) mxfp4 packed == SearchMxfp4Group", m.packed, ref.packed);
    SameVec("(c) mxfp4 escale", m.escale, ref.escale);
    SameVec("(c) mxfp4 wref", m.wref, ref.wref);
    const Mxfp4Quantized s = QuantizeMxfp4Search(w.data(), N, K, group, ImportanceVector{}, 4);
    SameVec("(c) mxfp4 packed == QuantizeMxfp4Search (no imatrix)", m.packed, s.packed);
    SameVec("(c) mxfp4 escale == QuantizeMxfp4Search", m.escale, s.escale);
  }
}

// ---- (g) the packers accept LDLQ output and decode back to it ------------------------------------
// Decode indexing is the kernels' own, as re-derived in test_kernel_decode.cpp's header comment.

int DecodeRow(int t, int lane) { return t * 16 + (lane & 15); }
int DecodeK(int kb, int s, int e, int lane) {
  return (kb * 4 + s) * 16 + 8 * (e >> 2) + 4 * (lane >> 4) + (e & 3);
}
int NibblePos(int e) { return (e < 4) ? (2 * e) : (2 * (e - 4) + 1); }

void CheckPackedW4A16(const std::string& label, const std::vector<float>& W,
                      const std::vector<double>& Hd, const std::vector<uint8_t>& q,
                      const std::vector<float>& scale, const std::vector<uint8_t>& zero, int N, int K,
                      int group, const std::vector<double>& deq32, double proxy32) {
  std::vector<uint32_t> wq, wsz;
  try {
    wq = PackW4Nibbles(q, N, K, 4);
    wsz = PackW4A16Scales(scale, zero, N, K, group);
  } catch (const std::exception& e) {
    Gate(false, label + ": packers threw: " + e.what());
    return;
  }
  const int ntiles = N / 16, kblocks = K / 64, gpr = K / group;
  std::vector<double> deqp(static_cast<size_t>(N) * K);
  bool codes_ok = true, zeros_ok = true;
  for (int t = 0; t < ntiles; ++t)
    for (int kb = 0; kb < kblocks; ++kb)
      for (int lane = 0; lane < 32; ++lane)
        for (int s = 0; s < 4; ++s) {
          const uint32_t dword = wq[((static_cast<size_t>(t) * kblocks + kb) * 32 + lane) * 4 + s];
          const int row = DecodeRow(t, lane);
          for (int e = 0; e < 8; ++e) {
            const int code = static_cast<int>(((dword >> (4 * NibblePos(e))) & 0xFu) ^ 0x8u);
            const int k = DecodeK(kb, s, e, lane);
            if (code != q[static_cast<size_t>(row) * K + k]) codes_ok = false;
            const int g = k / group;
            const uint32_t sz = wsz[(static_cast<size_t>(t) * gpr + g) * 16 + (lane & 15)];
            const float sc = r4dx::core::F16ToFloat(static_cast<uint16_t>(sz & 0xFFFFu));
            const float nz = r4dx::core::F16ToFloat(static_cast<uint16_t>(sz >> 16));
            const int z = static_cast<int>(-nz - 1024.0f);
            if (z != zero[static_cast<size_t>(row) * gpr + g]) zeros_ok = false;
            deqp[static_cast<size_t>(row) * K + k] =
                static_cast<double>(sc) * static_cast<double>(code - z);
          }
        }
  Gate(codes_ok && zeros_ok, label + ": PackW4Nibbles / PackW4A16Scales decode to the LDLQ codes and zeros");
  const double rel = RelL2(deqp, deq32);
  Gate(rel <= 1e-3, label + Fmt(": f16-scale dequant vs fp32 dequant rel diff %.3g <= 1e-3", rel));
  const double pp = Proxy(W, deqp, Hd, N, K);
  Gate(std::fabs(pp / proxy32 - 1.0) <= 0.02,
       label + Fmt(": proxy from packed bytes %.6g vs %.6g (within 2%%)", pp, proxy32));
}

void CheckPackedW4A8(const std::string& label, const std::vector<float>& W,
                     const std::vector<double>& Hd, const std::vector<uint8_t>& q,
                     const std::vector<float>& scale, int N, int K, int group,
                     const std::vector<double>& deq32, double proxy32) {
  std::vector<uint32_t> wq, ws;
  try {
    wq = PackW4Nibbles(q, N, K, 4);
    ws = PackW4A8Scales(scale, N, K, group);
  } catch (const std::exception& e) {
    Gate(false, label + ": packers threw: " + e.what());
    return;
  }
  const int ntiles = N / 16, kblocks = K / 64, gpr = K / group;
  std::vector<double> deqp(static_cast<size_t>(N) * K);
  bool codes_ok = true, hi_ok = true;
  for (int t = 0; t < ntiles; ++t)
    for (int kb = 0; kb < kblocks; ++kb)
      for (int lane = 0; lane < 32; ++lane)
        for (int s = 0; s < 4; ++s) {
          const uint32_t dword = wq[((static_cast<size_t>(t) * kblocks + kb) * 32 + lane) * 4 + s];
          const int row = DecodeRow(t, lane);
          for (int e = 0; e < 8; ++e) {
            const uint32_t nibble = (dword >> (4 * NibblePos(e))) & 0xFu;
            const int sv = static_cast<int8_t>(nibble << 4) / 16;  // dequant8's two's-complement read
            const int k = DecodeK(kb, s, e, lane);
            if (sv != static_cast<int>(q[static_cast<size_t>(row) * K + k]) - 8) codes_ok = false;
            const uint32_t word = ws[(static_cast<size_t>(t) * gpr + k / group) * 16 + (lane & 15)];
            if ((word >> 16) != 0) hi_ok = false;
            const float sc = r4dx::core::F16ToFloat(static_cast<uint16_t>(word & 0xFFFFu));
            deqp[static_cast<size_t>(row) * K + k] = static_cast<double>(sc) * sv;
          }
        }
  Gate(codes_ok && hi_ok, label + ": PackW4Nibbles / PackW4A8Scales decode to the LDLQ codes");
  const double rel = RelL2(deqp, deq32);
  Gate(rel <= 1e-3, label + Fmt(": f16-scale dequant vs fp32 dequant rel diff %.3g <= 1e-3", rel));
  const double pp = Proxy(W, deqp, Hd, N, K);
  Gate(std::fabs(pp / proxy32 - 1.0) <= 0.02,
       label + Fmt(": proxy from packed bytes %.6g vs %.6g (within 2%%)", pp, proxy32));
}

void CheckPackedMxfp4(const std::string& label, const Mxfp4Quantized& m, int N, int K,
                      const std::vector<double>& deq32) {
  const int group = kMxfp4Group, gpr = K / group;
  std::vector<uint8_t> wq, ws;
  try {
    wq = PackMxfp4Wq(m.packed, N, K, 4);
    ws = PackMxfp4Ws(m.escale, N, K, group);
  } catch (const std::exception& e) {
    Gate(false, label + ": packers threw: " + e.what());
    return;
  }
  bool wref_ok = true;
  int max_dsh = 0;
  for (int r = 0; r < N; ++r) {
    uint8_t mx = 0;
    for (int g = 0; g < gpr; ++g) {
      const uint8_t raw = m.escale[static_cast<size_t>(r) * gpr + g];
      mx = std::max(mx, raw);
    }
    if (mx != m.wref[r]) wref_ok = false;
    for (int g = 0; g < gpr; ++g)
      max_dsh = std::max(max_dsh, static_cast<int>(m.wref[r]) - m.escale[static_cast<size_t>(r) * gpr + g]);
  }
  Gate(wref_ok, label + ": wref == max escale per row");
  Gate(max_dsh <= 15, label + Fmt(": max Wref - escale = %d fits the kernel's 0..15 dsh clamp", max_dsh));

  const int ntiles = N / 16, ksteps = K / 16;
  bool codes_ok = true, values_ok = true;
  for (int nt = 0; nt < ntiles; ++nt)
    for (int ks = 0; ks < ksteps; ++ks)
      for (int lane = 0; lane < 32; ++lane) {
        const int row = nt * 16 + (lane & 15), h = lane >> 4;
        const size_t base = (static_cast<size_t>(nt) * ksteps + ks) * 32 * 4 + static_cast<size_t>(lane) * 4;
        for (int j = 0; j < 4; ++j)
          for (int half = 0; half < 2; ++half) {
            const uint8_t byte = wq[base + j];
            const uint8_t code = half == 0 ? (byte & 0xF) : ((byte >> 4) & 0xF);
            const int k = ks * 16 + 8 * h + 2 * j + half;
            const uint8_t src = m.packed[static_cast<size_t>(row) * (K / 2) + k / 2];
            if (code != ((k % 2 == 0) ? (src & 0xF) : ((src >> 4) & 0xF))) codes_ok = false;
            const uint8_t escale = ws[static_cast<size_t>(k / group) * N + row];
            if (escale != m.escale[static_cast<size_t>(row) * gpr + k / group]) codes_ok = false;
            // the kernel's Wref / clamped-dsh fold (test_quantize_roundtrip.cpp's mxfp4 comment)
            int dsh = static_cast<int>(m.wref[row]) - static_cast<int>(escale);
            dsh = dsh < 0 ? 0 : (dsh > 15 ? 15 : dsh);
            const double v = E2M1Value(code, static_cast<int>(m.wref[row]) - dsh);
            if (v != deq32[static_cast<size_t>(row) * K + k]) values_ok = false;
          }
      }
  Gate(codes_ok, label + ": PackMxfp4Wq / PackMxfp4Ws decode to the LDLQ codes and exponents");
  Gate(values_ok, label + ": kernel-side dequant (Wref/dsh fold) == fp32 dequant exactly");
}

// ---- (d) proxy loss, and (g) on the same outputs ------------------------------------------------

// The unblocked reference for (d): textbook per-row GPTQ / LDLQ with NO lazy batching. After every
// column the error is pushed onto the WHOLE rest of the row (w_j = fma(-err, U[i][j], w_j) for every
// j > i), and each group's parameters are chosen at its first column from the CURRENT weights with
// the no-refit search weighted by diag(H). Mathematically this is quant_ldlq.hpp's blocked loop; only
// the fp summation order of the feedback differs (one fma per column here, an in-block fma plus one
// per-block GEMM there), so a few codes sitting on a rounding knife-edge may differ and nothing else.
// The ratio gate alone cannot see a broken block update: dropping it entirely still leaves LDLQ at
// ~0.3-0.6x the search baseline on this data, because the in-block feedback carries most of the gain.
enum class RefLayout { kAsym, kPinned8, kMxfp4 };

struct RefResult {
  std::vector<uint8_t> codes;  // one unpacked code per element (w4a16 q, w4a8 q, mxfp4 nibble)
  std::vector<double> deq;     // the fp32 dequantized value the reference rounded to
};

RefResult UnblockedLdlqRef(RefLayout layout, const std::vector<float>& W, int N, int K, int group,
                           const LdlqFactor& f) {
  RefResult out;
  out.codes.assign(static_cast<size_t>(N) * K, 0);
  out.deq.assign(static_cast<size_t>(N) * K, 0.0);
  ParallelFor(0, N, 16, [&](int64_t r0, int64_t r1) {
    R4DX_NO_FP_CONTRACT
    std::vector<float> w(static_cast<size_t>(K));
    std::vector<int32_t> rq(static_cast<size_t>(group));
    std::vector<uint8_t> bq(static_cast<size_t>(group)), qg(static_cast<size_t>(group));
    for (int64_t r = r0; r < r1; ++r) {
      std::copy(W.begin() + r * K, W.begin() + (r + 1) * K, w.begin());
      float sc = 0.0f;
      int zp = 0;
      for (int i = 0; i < K; ++i) {
        if (i % group == 0) {
          const float* wg = w.data() + i;
          const float* wt = f.diag_h.data() + i;
          if (layout == RefLayout::kAsym) {
            SearchInt4AsymGroup(wg, wt, group, /*refit=*/false, rq.data(), bq.data(), qg.data(), &sc,
                                &zp);
          } else if (layout == RefLayout::kPinned8) {
            SearchInt4Pinned8Group(wg, wt, group, /*refit=*/false, bq.data(), qg.data(), &sc);
          } else {
            int raw = 0;
            SearchMxfp4Group(wg, wt, group, bq.data(), qg.data(), &raw);
            sc = std::ldexp(1.0f, raw - 127);
          }
        }
        const float x = w[i];
        uint8_t code = 0;
        float dq = 0.0f;
        if (layout == RefLayout::kAsym) {
          if (sc == 0.0f) {
            code = static_cast<uint8_t>(zp);
          } else {
            const int qi = ClampInt(RoundHalfAwayFromZero(x / sc) + zp, 0, 15);
            code = static_cast<uint8_t>(qi);
            dq = sc * (static_cast<float>(qi) - static_cast<float>(zp));
          }
        } else if (layout == RefLayout::kPinned8) {
          const int qs = ClampInt(RoundHalfAwayFromZero(x / sc), -8, 7);
          code = static_cast<uint8_t>(qs + 8);
          dq = sc * static_cast<float>(qs);
        } else {
          code = EncodeE2M1(x / sc);
          const float mag = kE2M1Magnitude[code & 0x7];
          dq = ((code & 0x8) ? -mag : mag) * sc;
        }
        out.codes[static_cast<size_t>(r) * K + i] = code;
        out.deq[static_cast<size_t>(r) * K + i] = static_cast<double>(dq);
        const float* u = f.U.data() + static_cast<size_t>(i) * K;
        const float err = (x - dq) / u[i];
        for (int j = i + 1; j < K; ++j) w[j] = std::fma(-err, u[j], w[j]);
      }
    }
  });
  return out;
}

std::vector<uint8_t> UnpackMxfp4Codes(const Mxfp4Quantized& m, int N, int K) {
  std::vector<uint8_t> c(static_cast<size_t>(N) * K);
  for (size_t e = 0; e < c.size(); ++e) {
    const uint8_t byte = m.packed[e / 2];
    c[e] = (e % 2 == 0) ? (byte & 0xF) : ((byte >> 4) & 0xF);
  }
  return c;
}

// The blocked LDLQ output against UnblockedLdlqRef: code mismatch rate <= 0.1% and proxy within
// 0.5%. Measured when this was written: 0-0.04% of codes, proxy within 1e-4 -- summation order only.
// Mutants of quant_ldlq.hpp, each caught: dropping the cross-block update (14-22% of codes, proxy
// 2-6x), an off-by-one column offset into U in that update (18-29%, 6-14x), and choosing a group's
// parameters from the ORIGINAL instead of the current weights (5-21% of codes, proxy only ~1-2% off
// -- the code count is what catches this one).
void CheckAgainstRef(const std::string& label, const RefResult& ref,
                     const std::vector<uint8_t>& codes, double p_ldlq, double p_ref) {
  size_t mism = 0;
  for (size_t e = 0; e < codes.size(); ++e) mism += (codes[e] != ref.codes[e]) ? 1 : 0;
  const double rate = static_cast<double>(mism) / static_cast<double>(codes.size());
  Gate(codes.size() == ref.codes.size() && rate <= 1e-3,
       Fmt("(d) %s: codes vs the unblocked reference differ at %zu / %zu (%.4f%%) <= 0.1%%",
           label.c_str(), mism, codes.size(), 100.0 * rate));
  const double rel = std::fabs(p_ldlq - p_ref) / p_ref;
  Gate(rel <= 5e-3, Fmt("(d) %s: proxy %.6g vs the unblocked reference %.6g (rel %.2e) <= 0.5%%",
                        label.c_str(), p_ldlq, p_ref, rel));
}

void TestProxyAndPackers() {
  std::printf("---- (d) proxy loss vs search+diag(H), (g) packers on the LDLQ outputs ----\n");
  const int N = 256;
  const int64_t rows = 4096;
  for (int K : {256, 512}) {
    const std::vector<float> H = HessianOf(CorrelatedX(rows, K, 40 + static_cast<uint32_t>(K)), rows, K, 16);
    const std::vector<double> Hd = ToDouble(H);
    const std::vector<float> W = RandomNormal(static_cast<size_t>(N) * K, 50 + static_cast<uint32_t>(K));
    const LdlqFactor f = FactorHessian(H, K, 0.01f, 16);
    ImportanceVector imp;
    imp.data = f.diag_h.data();
    imp.size = K;
    const std::string kt = Fmt("K=%d", K);

    auto report = [&](const std::string& what, double rtn, double search, double ldlq) {
      const double ratio = ldlq / search;
      std::printf("     (d) %s %-10s proxy: rtn %.6g  search+diag(H) %.6g  ldlq %.6g  "
                  "(ldlq/rtn %.3f)\n",
                  kt.c_str(), what.c_str(), rtn, search, ldlq, ldlq / rtn);
      // docs/quant2.md's G1 floor is 0.8. On THIS data the correct loop sits at ~0.08-0.11 and one
      // with the cross-block update dropped at >= 0.30, so 0.2 is a regression tripwire with margin
      // both ways; the unblocked-reference check below is the precise one.
      Gate(ratio <= 0.2, Fmt("(d) %s %s: ldlq / search+diag(H) = %.3f <= 0.2 (G1 floor 0.8)",
                             kt.c_str(), what.c_str(), ratio));
    };
    auto vs_ref = [&](const std::string& what, RefLayout layout, int group,
                      const std::vector<uint8_t>& codes, double p_ldlq) {
      const RefResult ref = UnblockedLdlqRef(layout, W, N, K, group, f);
      CheckAgainstRef(kt + " " + what, ref, codes, p_ldlq, Proxy(W, ref.deq, Hd, N, K));
    };

    // 32 is a per-tensor w4a16 group (--w4a16-group-rule, docs/quant2.md section 5); 64 / 128 the
    // two build defaults.
    for (int group : {32, 64, 128}) {
      const std::string what = Fmt("w4a16 g%d", group);
      std::vector<uint8_t> q0, z0, q1, z1, q2, z2;
      std::vector<float> s0, s1, s2;
      QuantizeInt4Asymmetric(W.data(), N, K, group, 16, q0, s0, z0);
      QuantizeInt4AsymmetricSearch(W.data(), N, K, group, imp, 16, q1, s1, z1);
      QuantizeInt4AsymmetricLdlq(W.data(), N, K, group, f, 16, q2, s2, z2);
      const std::vector<double> d2 = DeqAsym(q2, s2, z2, N, K, group);
      const double p2 = Proxy(W, d2, Hd, N, K);
      report(what, Proxy(W, DeqAsym(q0, s0, z0, N, K, group), Hd, N, K),
             Proxy(W, DeqAsym(q1, s1, z1, N, K, group), Hd, N, K), p2);
      vs_ref(what, RefLayout::kAsym, group, q2, p2);
      CheckPackedW4A16("(g) " + kt + " " + what, W, Hd, q2, s2, z2, N, K, group, d2, p2);
    }
    {
      const int group = kW4A8Group;
      const std::string what = Fmt("w4a8 g%d", group);
      std::vector<uint8_t> q0, q1, q2;
      std::vector<float> s0, s1, s2;
      QuantizeInt4SymmetricPinned8(W.data(), N, K, group, 16, q0, s0);
      QuantizeInt4Pinned8Search(W.data(), N, K, group, imp, 16, q1, s1);
      QuantizeInt4Pinned8Ldlq(W.data(), N, K, group, f, 16, q2, s2);
      const std::vector<double> d2 = DeqPinned8(q2, s2, N, K, group);
      const double p2 = Proxy(W, d2, Hd, N, K);
      report(what, Proxy(W, DeqPinned8(q0, s0, N, K, group), Hd, N, K),
             Proxy(W, DeqPinned8(q1, s1, N, K, group), Hd, N, K), p2);
      vs_ref(what, RefLayout::kPinned8, group, q2, p2);
      CheckPackedW4A8("(g) " + kt + " " + what, W, Hd, q2, s2, N, K, group, d2, p2);
    }
    {
      const int group = kMxfp4Group;
      const Mxfp4Quantized m0 = QuantizeMxfp4(W.data(), N, K, group, 16);
      const Mxfp4Quantized m1 = QuantizeMxfp4Search(W.data(), N, K, group, imp, 16);
      const Mxfp4Quantized m2 = QuantizeMxfp4Ldlq(W.data(), N, K, group, f, 16);
      const std::vector<double> d2 = DeqMxfp4(m2, N, K, group);
      const double p2 = Proxy(W, d2, Hd, N, K);
      report("mxfp4", Proxy(W, DeqMxfp4(m0, N, K, group), Hd, N, K),
             Proxy(W, DeqMxfp4(m1, N, K, group), Hd, N, K), p2);
      vs_ref("mxfp4", RefLayout::kMxfp4, group, UnpackMxfp4Codes(m2, N, K), p2);
      CheckPackedMxfp4("(g) " + kt + " mxfp4", m2, N, K, d2);
    }
  }
}

// ---- (e) determinism ----------------------------------------------------------------------------

void TestDeterminism() {
  std::printf("---- (e) determinism ----\n");
  const int N = 400, K = 256;  // 4 tiles, the last one partial (16 rows)
  const std::vector<float> H = HessianOf(CorrelatedX(2048, K, 61), 2048, K, 16);
  const std::vector<float> W = RandomNormal(static_cast<size_t>(N) * K, 62);

  const LdlqFactor f = FactorHessian(H, K, 0.01f, 16);
  for (int t : {1, 3}) {
    const LdlqFactor ft = FactorHessian(H, K, 0.01f, t);
    SameVec(Fmt("(e) FactorHessian U: %d thread(s) == 16", t), ft.U, f.U);
  }
  const bool avx = linalg::CpuHasAvx512f();
  if (avx) {
    ScopedForceScalar s(true);
    const LdlqFactor fs_ = FactorHessian(H, K, 0.01f, 3);
    SameVec("(e) FactorHessian U: scalar path == AVX-512 path", fs_.U, f.U);
    SameVec("(e) FactorHessian diag_h: scalar path == AVX-512 path", fs_.diag_h, f.diag_h);
  } else {
    std::printf("SKIP (e) scalar-vs-AVX-512 comparisons: CPU has no AVX-512F\n");
  }

  // Each layout at 16 threads is the reference; 1 and 3 threads, and the scalar path, must match it.
  for (int group : {32, 64, 128}) {
    std::vector<uint8_t> q16, z16;
    std::vector<float> s16;
    QuantizeInt4AsymmetricLdlq(W.data(), N, K, group, f, 16, q16, s16, z16);
    auto check = [&](const std::string& how, int t) {
      std::vector<uint8_t> q, z;
      std::vector<float> s;
      QuantizeInt4AsymmetricLdlq(W.data(), N, K, group, f, t, q, s, z);
      const std::string l = Fmt("(e) w4a16 g%d ", group) + how;
      SameVec(l + " q", q, q16);
      SameVec(l + " scale", s, s16);
      SameVec(l + " zero", z, z16);
    };
    check("1 thread == 16", 1);
    check("3 threads == 16", 3);
    if (avx) {
      ScopedForceScalar s(true);
      check("scalar path == AVX-512 path", 3);
    }
  }
  {
    const int group = kW4A8Group;
    std::vector<uint8_t> q16;
    std::vector<float> s16;
    QuantizeInt4Pinned8Ldlq(W.data(), N, K, group, f, 16, q16, s16);
    auto check = [&](const std::string& how, int t) {
      std::vector<uint8_t> q;
      std::vector<float> s;
      QuantizeInt4Pinned8Ldlq(W.data(), N, K, group, f, t, q, s);
      SameVec("(e) w4a8 " + how + " q", q, q16);
      SameVec("(e) w4a8 " + how + " scale", s, s16);
    };
    check("1 thread == 16", 1);
    check("3 threads == 16", 3);
    if (avx) {
      ScopedForceScalar s(true);
      check("scalar path == AVX-512 path", 3);
    }
  }
  {
    const Mxfp4Quantized m16 = QuantizeMxfp4Ldlq(W.data(), N, K, kMxfp4Group, f, 16);
    auto check = [&](const std::string& how, int t) {
      const Mxfp4Quantized m = QuantizeMxfp4Ldlq(W.data(), N, K, kMxfp4Group, f, t);
      SameVec("(e) mxfp4 " + how + " packed", m.packed, m16.packed);
      SameVec("(e) mxfp4 " + how + " escale", m.escale, m16.escale);
      SameVec("(e) mxfp4 " + how + " wref", m.wref, m16.wref);
    };
    check("1 thread == 16", 1);
    check("3 threads == 16", 3);
    if (avx) {
      ScopedForceScalar s(true);
      check("scalar path == AVX-512 path", 3);
    }
  }
}

// ---- (f) HessianStore on the capture fixture ------------------------------------------------------

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

void WriteManifest(const fs::path& dir, const nlohmann::json& j) {
  WriteWhole(dir / "hessian.json", j.dump(2));
}

void TestStore(const std::string& fixtures) {
  std::printf("---- (f) HessianStore on fixtures/hess_small ----\n");
  const fs::path dir = fs::u8path(fixtures) / "hess_small";
  nlohmann::json expected;
  {
    std::ifstream f(dir / "expected.json");
    if (!f) {
      Gate(false, "(f) open " + (dir / "expected.json").u8string());
      return;
    }
    f >> expected;
  }

  // Per-file content, straight through ReadHessFile.
  for (auto it = expected["files"].begin(); it != expected["files"].end(); ++it) {
    const std::string name = it.key();
    const nlohmann::json& e = it.value();
    const fs::path path = dir / name;
    const std::string tag = "(f) " + name;

    const std::string bytes = ReadWhole(path);
    Gate(bytes.size() == e.at("bytes").get<uint64_t>(),
         tag + Fmt(": file is %zu bytes", bytes.size()));
    // A mismatch here with every value check below passing means git rewrote the file
    // (fixtures/hess_small/.gitattributes must say -text).
    Gate(Sha256Hex(bytes) == e.at("sha256").get<std::string>(), tag + ": sha256 matches expected.json");

    int64_t K = 0;
    uint64_t rows = 0;
    std::vector<float> H;
    try {
      H = ReadHessFile(path.u8string(), &K, &rows);
    } catch (const std::exception& ex) {
      Gate(false, tag + ": ReadHessFile threw: " + ex.what());
      continue;
    }
    Gate(K == e.at("K").get<int64_t>() && rows == e.at("rows").get<uint64_t>(),
         tag + Fmt(": K=%lld rows=%llu", static_cast<long long>(K), static_cast<unsigned long long>(rows)));
    if (H.size() != static_cast<size_t>(K * K)) {
      Gate(false, tag + ": H has K*K elements");
      continue;
    }
    double trace = 0.0, sum_upper = 0.0;
    float dmin = std::numeric_limits<float>::infinity(), dmax = -dmin;
    bool sym = true;
    for (int64_t i = 0; i < K; ++i) {
      trace += static_cast<double>(H[i * K + i]);
      dmin = std::min(dmin, H[i * K + i]);
      dmax = std::max(dmax, H[i * K + i]);
      for (int64_t j = i; j < K; ++j) {
        sum_upper += static_cast<double>(H[i * K + j]);
        if (std::memcmp(&H[i * K + j], &H[j * K + i], sizeof(float)) != 0) sym = false;
      }
    }
    const double trace_e = e.at("trace").get<double>();
    const double sum_e = e.at("sum_upper").get<double>();
    Gate(std::fabs(trace - trace_e) <= 1e-9 * std::fabs(trace_e),
         tag + Fmt(": trace %.12g vs %.12g", trace, trace_e));
    Gate(std::fabs(sum_upper - sum_e) <= 1e-6 * std::fabs(sum_e),
         tag + Fmt(": fp64 sum of the upper triangle %.12g vs %.12g (1e-6 rel)", sum_upper, sum_e));
    Gate(dmin == static_cast<float>(e.at("min_diag").get<double>()) &&
             dmax == static_cast<float>(e.at("max_diag").get<double>()),
         tag + Fmt(": min/max diag %.9g / %.9g", static_cast<double>(dmin), static_cast<double>(dmax)));
    Gate(sym, tag + ": H exactly symmetric");
    bool samples_ok = true;
    for (const auto& s : e.at("samples")) {
      const int64_t i = s.at(0).get<int64_t>(), j = s.at(1).get<int64_t>();
      const float want = static_cast<float>(s.at(2).get<double>());
      if (H[i * K + j] != want) {
        std::fprintf(stderr, "     %s: H[%lld][%lld] = %.9g, expected %.9g\n", name.c_str(),
                     static_cast<long long>(i), static_cast<long long>(j),
                     static_cast<double>(H[i * K + j]), static_cast<double>(want));
        samples_ok = false;
      }
    }
    Gate(samples_ok, tag + ": sample entries (both triangles) exact");
  }

  // The manifest-level API.
  try {
    HessianStore store(dir.u8string());
    Gate(store.ManifestSha256() == expected.at("manifest_sha256").get<std::string>(),
         "(f) ManifestSha256 == sha256(hessian.json)");
    Gate(store.Has("t.a") && store.Has("t.b") && store.Has("t.c") && !store.Has("t.d"),
         "(f) Has: t.a, t.b, t.c yes; t.d no");
    Gate(store.KOf("t.a") == 128 && store.KOf("t.b") == 256 && store.KOf("t.c") == 256,
         "(f) KOf: 128 / 256 / 256");
    Gate(store.File("t.a") == "a.hess" && store.File("t.b") == "bc.hess" && store.File("t.c") == "bc.hess",
         "(f) File: a.hess / bc.hess / bc.hess");
    ExpectThrow("(f) File of an unknown key throws", [&]() { store.File("t.d"); });

    const LdlqFactor& fb = store.Factor("t.b", 256, 0.01f, 4);
    Gate(FactorShapeOk(fb, 256), "(f) Factor(t.b): valid K=256 factor");
    // Plant a marker in the cached object. A cache hit hands back the SAME object with the marker
    // still in it; a silent refactorization would overwrite it. The address alone proves nothing:
    // the cache is one member, so every call returns the same address either way.
    const_cast<LdlqFactor&>(fb).retries = 777;
    const LdlqFactor& fc = store.Factor("t.c", 256, 0.01f, 4);
    Gate(&fc == &fb && fc.retries == 777, "(f) Factor(t.c) reuses t.b's cached factorization (same file, same damp)");
    const LdlqFactor& fb2 = store.Factor("t.b", 256, 0.02f, 4);
    Gate(fb2.retries != 777 && fb2.damp_used == 0.02f, "(f) Factor(t.b) at a different damp refactors");
    {
      int64_t K = 0;
      uint64_t rows = 0;
      const std::vector<float> Ha = ReadHessFile((dir / "a.hess").u8string(), &K, &rows);
      const LdlqFactor& fa = store.Factor("t.a", 128, 0.01f, 4);
      bool diag_ok = FactorShapeOk(fa, 128);
      for (int64_t i = 0; diag_ok && i < 128; ++i) diag_ok = fa.diag_h[i] == Ha[i * 128 + i];
      Gate(diag_ok, "(f) Factor(t.a): valid K=128 factor whose diag_h is the file's diagonal");
    }
    ExpectThrow("(f) Factor with a K that contradicts the manifest throws",
                [&]() { store.Factor("t.a", 256, 0.01f, 4); });
    ExpectThrow("(f) Factor of an unknown key throws", [&]() { store.Factor("t.d", 128, 0.01f, 4); });
  } catch (const std::exception& e) {
    Gate(false, std::string("(f) HessianStore on the fixture threw: ") + e.what());
  }

  // Corruptions, in a private temp directory.
  const fs::path tmp = fs::temp_directory_path() /
                       ("r4dx_test_quant_ldlq_" +
                        std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
  try {
    fs::create_directories(tmp);
    const std::string good = ReadWhole(dir / "a.hess");
    auto corrupt = [&](const std::string& what, const std::string& bytes) {
      const fs::path p = tmp / "bad.hess";
      WriteWhole(p, bytes);
      ExpectThrow("(f) ReadHessFile rejects " + what, [&]() {
        int64_t K = 0;
        uint64_t rows = 0;
        ReadHessFile(p.u8string(), &K, &rows);
      });
    };
    {
      std::string b = good;
      b[0] = 'X';
      corrupt("a corrupted magic", b);
    }
    {
      std::string b = good;
      b[12] = 3;  // flags: packed upper + an unknown bit
      corrupt("unknown flags", b);
    }
    {
      std::string b = good;
      const float big = 1000.0f;  // H[0][0], the first packed value: the trace no longer matches
      std::memcpy(&b[64], &big, sizeof(float));
      corrupt("a trace mismatch", b);
    }
    corrupt("a truncated file", good.substr(0, good.size() - 4));

    // Manifests.
    const nlohmann::json base = nlohmann::json::parse(ReadWhole(dir / "hessian.json"));
    auto manifest_case = [&](const std::string& sub, const nlohmann::json& j) {
      const fs::path d = tmp / sub;
      fs::create_directories(d);
      WriteWhole(d / "a.hess", good);
      WriteManifest(d, j);
      return d;
    };
    {
      nlohmann::json j = base;
      j["version"] = 2;
      const fs::path d = manifest_case("version2", j);
      ExpectThrow("(f) HessianStore rejects version 2", [&]() { HessianStore s(d.u8string()); });
    }
    {
      nlohmann::json j = base;
      j["keys"]["t.x"] = "nope.hess";
      const fs::path d = manifest_case("missing_file", j);
      ExpectThrow("(f) HessianStore rejects a key naming an unlisted file",
                  [&]() { HessianStore s(d.u8string()); });
    }
    {
      nlohmann::json j;
      j["format"] = "r4dx-hessian";
      j["version"] = 1;
      j["files"]["a.hess"] = {{"K", 256}, {"rows", 1024}};  // the file header says K=128
      j["keys"]["t.x"] = "a.hess";
      const fs::path d = manifest_case("k_mismatch", j);
      HessianStore s(d.u8string());
      Gate(s.KOf("t.x") == 256, "(f) KOf comes from the manifest alone (no file read)");
      ExpectThrow("(f) Factor rejects a file whose header K contradicts the manifest",
                  [&]() { s.Factor("t.x", 256, 0.01f, 4); });
    }
  } catch (const std::exception& e) {
    Gate(false, std::string("(f) corruption cases threw unexpectedly: ") + e.what());
  }
  std::error_code ec;
  fs::remove_all(tmp, ec);
}

}  // namespace

int main() {
  const std::string fixtures = R4DX_CONVERT_FIXTURES_DIR;
  try {
    TestLinalg();
    TestFactorHessian();
    TestIdentity();
    TestProxyAndPackers();
    TestDeterminism();
    TestStore(fixtures);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "FAIL uncaught exception: %s\n", e.what());
    return 1;
  }
  if (g_failures != 0) {
    std::fprintf(stderr, "FAIL: %d check(s) failed\n", g_failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
