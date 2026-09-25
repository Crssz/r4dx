// r4dx_convert::rotation -- the residual-stream rotation Q (docs/quant2.md section 3, "Q2a") and
// the K-side block Hadamards Hb on the mlp.down / attn.o / gdn.out_proj inputs (section 4, "Q2b"):
// the seeded generation of both, the weight folds r4dx-convert --rotate applies before quantizing,
// and the matching change of basis for the LDLQ Hessians and the imatrix vectors.
//
// ---- Convention --------------------------------------------------------------------------------
// Residual rows are ROW vectors x (hidden = 5120 = 5 x 1024 on this checkpoint). Q is orthogonal:
//
//   x Q    step 1  y = x * d                                     d = rotation.signs, +-1
//          step 2  y[b*B : (b+1)*B] = FWHT(y[b*B : (b+1)*B]) / sqrt(B)   per block b (B = 1024)
//          step 3  z[b*B + i] = sum_c y[c*B + i] * R[c][b]      R = rotation.mix5, 5x5 orthogonal
//   x Q^T  the exact inverse: y[c*B + i] = sum_b x[b*B + i] * R[c][b]; FWHT / sqrt(B) per block;
//          then * d.
//
// FWHT is the unnormalized Walsh-Hadamard transform in natural (Sylvester) order,
// H[i][j] = (-1)^popcount(i & j): symmetric, and H H = B I, so FWHT / sqrt(B) is its own inverse.
// As a matrix, Q[(c,i)][(b,j)] = d[c*B + i] * H[i][j] / sqrt(B) * R[c][b] -- tests/convert/
// test_rotation.cpp checks the fast transform against exactly that, entry by entry.
//
// The block Hadamard of a linear with input width K and block B (B | K, B a power of two), sign
// vector s (+-1, length K):
//
//   h Hb   := (h * s), then FWHT / sqrt(B) on each contiguous block of B     Hb Hb^T = I
//   h Hb^T := FWHT / sqrt(B) per block, then * s
//
// Q2b uses B = 512 for mlp.down (K = 17408), 256 for attn.o (K = 6144, one head each) and 128 for
// gdn.out_proj (K = 6144, one head each): every block divides the TP=2 per-rank K too.
//
// ---- The folds (fp32 in, double math, fp32 out; applied BEFORE quantization) ------------------
// With the residual carried as x' = x Q inside the stack, and every zero-centred norm stored as 0
// (so the kernel computes rms(x) * 1, which commutes with Q because Q is orthogonal):
//
//   in-projection  W [N, hidden]:  W' = W diag(1 + w_norm) Q    FoldRowsQ: each ROW r -> (r*(1+w)) Q
//       since (x'/rms) W'^T = (x/rms) Q Q^T diag(1+w) W^T = (x/rms)(1+w) W^T, the original output.
//   out-projection W [hidden, K]:  W' = Q^T W                   FoldColumnsQt: each COLUMN c, taken
//       as a row, -> c Q (that is (Q^T c)^T). The residual add r' + o W'^T = (r + o W^T) Q.
//   Q2b K side     W [N, K]:       W'' = W' Hb                  FoldRowsHadamard: each ROW -> row Hb,
//       and the runtime feeds h Hb, so (h Hb) W''^T = h Hb Hb^T W'^T = h W'^T.
//
// ---- Hessians and importance vectors ------------------------------------------------------------
// A folded linear sees its input in the new basis, so what LDLQ must minimize is the proxy against
// the Hessian OF THAT INPUT: with M the input change of basis (x_in' = x_in M),
//   H' = E[x_in'^T x_in'] = M^T H M,  and the fold is W' = W M^-T (it keeps x_in' W'^T = x_in W^T),
//   so W' H' W'^T = W M^-T M^T H M M^-1 W^T = W H W^T: the proxy of any error mapped the same way
//   (dW' = dW M^-T) is unchanged, which is what makes LDLQ in the new basis the same problem.
//   in-projection: x_in = x_n (the HF normed input, (1+w) included), x_in' = x_n D^-1 Q, so
//                  M = D^-1 Q (M^-T = D Q, the fold above) and H' = Q^T D^-1 H D^-1 Q
//                  (TransformHessianQ; |1+w| >= 1e-3 or throw)
//   Q2b K side:    M = Hb, H' = Hb^T H Hb                 (TransformHessianHadamard)
//   out-projection's N-side Q^T does not change its input: H unchanged.
// Both are computed with the fast transforms, never a dense K x K product: the row transform on
// every row of H (A -> A M), then the same transform on every column (-> M^T A M), then an exact
// symmetrization (the two halves differ only by rounding).
//
// The --imatrix vector is diag(H) of the UNrotated input, and diag(M^T H M) is not a function of
// diag(H). What the imatrix-weighted search already assumes, though, is H ~= diag(v) -- and under
// that model the rotated diagonal is exact and cheap, because every |Hadamard entry| is 1/sqrt(B):
//   Q:   v'[(b,j)] = sum_c R[c][b]^2 * mean_i( v[(c,i)] / (1+w[(c,i)])^2 )     (constant per block)
//   Hb:  v'[k] = mean of v over k's block
// (TransformImportanceQ / TransformImportanceHadamard). It is the diagonal model carried into the
// new basis, not diag(H'): the rotation spreads every channel's energy over its whole block, which is
// exactly why an imatrix says so little after it. --ldlq with the exact H' is the real answer there.
//
// ---- Generation (converter only; the runtime reads the stored tensors, never regenerates) -------
// One splitmix64 stream seeded with --rotation-seed, drawn in this fixed order:
//   1. rotation.signs[hidden]            sign = top bit of each draw: 1 -> -1.0f, 0 -> +1.0f
//   2. rotation.mix5[n][n] (n = hidden / B)   n*n standard normals, Box-Muller from pairs of draws
//        (u1 = ((a >> 11) + 1) * 2^-53 in (0, 1], u2 = (b >> 11) * 2^-53; the pair gives
//        r cos(2 pi u2), r sin(2 pi u2) with r = sqrt(-2 ln u1)), filled row-major; an odd count
//        drops the last sine. Rows are then orthonormalized IN ORDER by modified Gram-Schmidt in
//        double (two passes -- "twice is enough" -- so fp64 orthogonality is at the rounding floor),
//        rounded to fp32, and used as R[c][b] = G[c][b].
//   3. (q2ab only) rotation.had_down_signs[K_down], rotation.had_o_signs[K_o],
//      rotation.had_gdn_out_signs[K_gdn_out], top bit as in 1.
// q2ab's signs / mix5 are therefore identical to q2a's for the same seed. Every fold here uses the
// fp32 values exactly as stored, so the runtime (which reads those tensors) and the folded weights
// agree on Q to the last bit of R; R's fp32 rounding makes Q orthogonal to ~1e-7, far below the
// 4-bit weight error.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx_convert/threadpool.hpp"

namespace r4dx_convert {

inline constexpr uint64_t kDefaultRotationSeed = 0x5EED2025ull;
inline constexpr int64_t kRotationBlock = 1024;   // Q's Hadamard block (5120 = 5 x 1024)
inline constexpr int64_t kHadBlockDown = 512;     // mlp.down input
inline constexpr int64_t kHadBlockO = 256;        // attn.o input (one head)
inline constexpr int64_t kHadBlockGdnOut = 128;   // gdn.out_proj input (one head)

enum class RotationKind { kNone, kQ2a, kQ2ab };

inline RotationKind ParseRotationKind(const std::string& s) {
  if (s == "none") return RotationKind::kNone;
  if (s == "q2a") return RotationKind::kQ2a;
  if (s == "q2ab") return RotationKind::kQ2ab;
  throw std::runtime_error("--rotate must be one of none, q2a, q2ab (got '" + s + "')");
}

inline const char* RotationKindName(RotationKind k) {
  switch (k) {
    case RotationKind::kQ2a: return "q2a";
    case RotationKind::kQ2ab: return "q2ab";
    default: return "none";
  }
}

namespace rotation_detail {

inline uint64_t SplitMix64(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ull;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

inline float SignFromTopBit(uint64_t v) { return (v >> 63) ? -1.0f : 1.0f; }

inline bool IsPow2(int64_t n) { return n > 0 && (n & (n - 1)) == 0; }

// In place, unnormalized, natural (Sylvester) order: afterwards x = x_in H, H[i][j] =
// (-1)^popcount(i & j). n must be a power of two.
inline void Fwht(double* x, int64_t n) {
  for (int64_t len = 1; len < n; len <<= 1) {
    for (int64_t i = 0; i < n; i += 2 * len) {
      double* a = x + i;
      double* b = x + i + len;
      for (int64_t j = 0; j < len; ++j) {
        const double u = a[j], v = b[j];
        a[j] = u + v;
        b[j] = u - v;
      }
    }
  }
}

inline void FwhtBlocks(double* x, int64_t K, int64_t B) {
  const double inv = 1.0 / std::sqrt(static_cast<double>(B));
  for (int64_t b0 = 0; b0 < K; b0 += B) {
    Fwht(x + b0, B);
    for (int64_t j = 0; j < B; ++j) x[b0 + j] *= inv;
  }
}

inline std::vector<float> DrawSigns(uint64_t& state, int64_t n) {
  std::vector<float> s(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) s[static_cast<size_t>(i)] = SignFromTopBit(SplitMix64(state));
  return s;
}

}  // namespace rotation_detail

// ---- Q -------------------------------------------------------------------------------------------

struct ResidualRotation {
  int64_t hidden = 0;
  int64_t block = 0;
  int64_t nblk = 0;
  std::vector<float> signs;  // hidden, +-1
  std::vector<float> mix;    // nblk * nblk row-major: mix[c * nblk + b] = R[c][b]

  // x (length hidden) <- x Q. `tmp` is caller scratch of length hidden.
  void Apply(double* x, double* tmp) const {
    for (int64_t m = 0; m < hidden; ++m) x[m] *= signs[static_cast<size_t>(m)];
    rotation_detail::FwhtBlocks(x, hidden, block);
    for (int64_t b = 0; b < nblk; ++b) {
      double* z = tmp + b * block;
      for (int64_t i = 0; i < block; ++i) z[i] = 0.0;
      for (int64_t c = 0; c < nblk; ++c) {
        const double r = mix[static_cast<size_t>(c * nblk + b)];
        const double* y = x + c * block;
        for (int64_t i = 0; i < block; ++i) z[i] += y[i] * r;
      }
    }
    std::memcpy(x, tmp, static_cast<size_t>(hidden) * sizeof(double));
  }

  // x <- x Q^T.
  void ApplyT(double* x, double* tmp) const {
    for (int64_t c = 0; c < nblk; ++c) {
      double* y = tmp + c * block;
      for (int64_t i = 0; i < block; ++i) y[i] = 0.0;
      for (int64_t b = 0; b < nblk; ++b) {
        const double r = mix[static_cast<size_t>(c * nblk + b)];
        const double* z = x + b * block;
        for (int64_t i = 0; i < block; ++i) y[i] += z[i] * r;
      }
    }
    std::memcpy(x, tmp, static_cast<size_t>(hidden) * sizeof(double));
    rotation_detail::FwhtBlocks(x, hidden, block);
    for (int64_t m = 0; m < hidden; ++m) x[m] *= signs[static_cast<size_t>(m)];
  }
};

// ---- Hb ------------------------------------------------------------------------------------------

struct BlockHadamard {
  int64_t K = 0;
  int64_t block = 0;
  std::vector<float> signs;  // K, +-1

  bool Empty() const { return K == 0; }

  // h (length K) <- h Hb.
  void Apply(double* h) const {
    for (int64_t k = 0; k < K; ++k) h[k] *= signs[static_cast<size_t>(k)];
    rotation_detail::FwhtBlocks(h, K, block);
  }

  // h <- h Hb^T.
  void ApplyT(double* h) const {
    rotation_detail::FwhtBlocks(h, K, block);
    for (int64_t k = 0; k < K; ++k) h[k] *= signs[static_cast<size_t>(k)];
  }
};

// ---- generation ------------------------------------------------------------------------------------

// Step 1 + 2 of the header's generation order, drawing from `state`.
inline ResidualRotation GenerateResidualRotation(uint64_t& state, int64_t hidden, int64_t block) {
  using namespace rotation_detail;
  if (!IsPow2(block) || hidden <= 0 || hidden % block != 0) {
    throw std::runtime_error("GenerateResidualRotation: hidden=" + std::to_string(hidden) +
                             " must be a positive multiple of a power-of-two block (block=" +
                             std::to_string(block) + ")");
  }
  ResidualRotation q;
  q.hidden = hidden;
  q.block = block;
  q.nblk = hidden / block;
  q.signs = DrawSigns(state, hidden);

  const int64_t n = q.nblk;
  std::vector<double> g(static_cast<size_t>(n * n));
  const double kTwoPi = 6.283185307179586476925286766559;
  const double k2m53 = 1.0 / 9007199254740992.0;  // 2^-53
  for (int64_t p = 0; p < n * n; p += 2) {
    const uint64_t a = SplitMix64(state), b = SplitMix64(state);
    const double u1 = (static_cast<double>(a >> 11) + 1.0) * k2m53;  // (0, 1]
    const double u2 = static_cast<double>(b >> 11) * k2m53;          // [0, 1)
    const double r = std::sqrt(-2.0 * std::log(u1));
    g[static_cast<size_t>(p)] = r * std::cos(kTwoPi * u2);
    if (p + 1 < n * n) g[static_cast<size_t>(p + 1)] = r * std::sin(kTwoPi * u2);
  }
  // Modified Gram-Schmidt over the rows, in order, twice.
  for (int64_t r = 0; r < n; ++r) {
    double* gr = g.data() + r * n;
    for (int pass = 0; pass < 2; ++pass) {
      for (int64_t p = 0; p < r; ++p) {
        const double* gp = g.data() + p * n;
        double dot = 0.0;
        for (int64_t j = 0; j < n; ++j) dot += gr[j] * gp[j];
        for (int64_t j = 0; j < n; ++j) gr[j] -= dot * gp[j];
      }
      double nrm = 0.0;
      for (int64_t j = 0; j < n; ++j) nrm += gr[j] * gr[j];
      nrm = std::sqrt(nrm);
      if (!(nrm > 1e-12))
        throw std::runtime_error("GenerateResidualRotation: degenerate Gram-Schmidt row " +
                                 std::to_string(r) + " (try another --rotation-seed)");
      for (int64_t j = 0; j < n; ++j) gr[j] /= nrm;
    }
  }
  q.mix.resize(static_cast<size_t>(n * n));
  for (size_t i = 0; i < q.mix.size(); ++i) q.mix[i] = static_cast<float>(g[i]);
  return q;
}

// Step 3's building block: one sign vector of length K, block B.
inline BlockHadamard GenerateBlockHadamard(uint64_t& state, int64_t K, int64_t block) {
  if (!rotation_detail::IsPow2(block) || K <= 0 || K % block != 0) {
    throw std::runtime_error("GenerateBlockHadamard: K=" + std::to_string(K) +
                             " must be a positive multiple of a power-of-two block (block=" +
                             std::to_string(block) + ")");
  }
  BlockHadamard h;
  h.K = K;
  h.block = block;
  h.signs = rotation_detail::DrawSigns(state, K);
  return h;
}

// The input widths of the three Q2b sites (q2ab only; ignored for q2a).
struct RotationShape {
  int64_t hidden = 0;
  int64_t block = kRotationBlock;
  int64_t k_down = 0, b_down = kHadBlockDown;
  int64_t k_o = 0, b_o = kHadBlockO;
  int64_t k_gdn_out = 0, b_gdn_out = kHadBlockGdnOut;
};

struct RotationSet {
  RotationKind kind = RotationKind::kNone;
  uint64_t seed = 0;
  ResidualRotation q;
  BlockHadamard had_down, had_o, had_gdn_out;  // Empty() unless kind == kQ2ab
};

inline RotationSet GenerateRotationSet(RotationKind kind, uint64_t seed, const RotationShape& shape) {
  RotationSet set;
  set.kind = kind;
  set.seed = seed;
  if (kind == RotationKind::kNone) return set;
  uint64_t state = seed;
  set.q = GenerateResidualRotation(state, shape.hidden, shape.block);
  if (kind == RotationKind::kQ2ab) {
    set.had_down = GenerateBlockHadamard(state, shape.k_down, shape.b_down);
    set.had_o = GenerateBlockHadamard(state, shape.k_o, shape.b_o);
    set.had_gdn_out = GenerateBlockHadamard(state, shape.k_gdn_out, shape.b_gdn_out);
  }
  return set;
}

// ---- generic row / column drivers --------------------------------------------------------------

// op(double* v, double* tmp) transforms one length-K row in place (tmp: scratch of length K).
// Rows are independent and each result depends only on its own row, so the output does not depend
// on nthreads.
template <typename Op>
void ApplyToRows(float* W, int64_t N, int64_t K, int nthreads, const Op& op) {
  ParallelFor(0, N, nthreads, [&](int64_t r0, int64_t r1) {
    std::vector<double> v(static_cast<size_t>(K)), tmp(static_cast<size_t>(K));
    for (int64_t r = r0; r < r1; ++r) {
      float* row = W + r * K;
      for (int64_t k = 0; k < K; ++k) v[static_cast<size_t>(k)] = row[k];
      op(v.data(), tmp.data());
      for (int64_t k = 0; k < K; ++k) row[k] = static_cast<float>(v[static_cast<size_t>(k)]);
    }
  });
}

// Same, on every length-N COLUMN of W [N, K] (taken as a row vector). Columns are gathered 16 at a
// time, so each row of W is read as one 64-byte run per tile rather than one float per cache line.
template <typename Op>
void ApplyToColumns(float* W, int64_t N, int64_t K, int nthreads, const Op& op) {
  constexpr int64_t kTile = 16;
  const int64_t tiles = (K + kTile - 1) / kTile;
  ParallelFor(0, tiles, nthreads, [&](int64_t t0, int64_t t1) {
    std::vector<double> buf(static_cast<size_t>(kTile * N)), tmp(static_cast<size_t>(N));
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t k0 = t * kTile;
      const int64_t nk = std::min(kTile, K - k0);
      for (int64_t m = 0; m < N; ++m) {
        const float* src = W + m * K + k0;
        for (int64_t c = 0; c < nk; ++c) buf[static_cast<size_t>(c * N + m)] = src[c];
      }
      for (int64_t c = 0; c < nk; ++c) op(buf.data() + c * N, tmp.data());
      for (int64_t m = 0; m < N; ++m) {
        float* dst = W + m * K + k0;
        for (int64_t c = 0; c < nk; ++c) dst[c] = static_cast<float>(buf[static_cast<size_t>(c * N + m)]);
      }
    }
  });
}

// ---- weight folds ------------------------------------------------------------------------------

// In-projection, W [N, hidden] row-major: every row r -> (r * (1 + norm_w)) Q. norm_w == nullptr
// folds Q alone (D = I).
inline void FoldRowsQ(std::vector<float>& W, int64_t N, int64_t K, const float* norm_w,
                      const ResidualRotation& q, int nthreads) {
  if (K != q.hidden || W.size() != static_cast<size_t>(N * K))
    throw std::runtime_error("FoldRowsQ: W is [" + std::to_string(N) + ", " + std::to_string(K) +
                             "] but Q is " + std::to_string(q.hidden) + " wide");
  ApplyToRows(W.data(), N, K, nthreads, [&](double* v, double* tmp) {
    if (norm_w) {
      for (int64_t k = 0; k < K; ++k) v[k] *= 1.0 + static_cast<double>(norm_w[k]);
    }
    q.Apply(v, tmp);
  });
}

// Out-projection, W [hidden, K]: W' = Q^T W, i.e. every column c (as a row) -> c Q.
inline void FoldColumnsQt(std::vector<float>& W, int64_t N, int64_t K, const ResidualRotation& q,
                          int nthreads) {
  if (N != q.hidden || W.size() != static_cast<size_t>(N * K))
    throw std::runtime_error("FoldColumnsQt: W is [" + std::to_string(N) + ", " +
                             std::to_string(K) + "] but Q is " + std::to_string(q.hidden) + " wide");
  ApplyToColumns(W.data(), N, K, nthreads, [&](double* v, double* tmp) { q.Apply(v, tmp); });
}

// Q2b K side, W [N, K]: every row -> row Hb.
inline void FoldRowsHadamard(std::vector<float>& W, int64_t N, int64_t K, const BlockHadamard& hb,
                             int nthreads) {
  if (K != hb.K || W.size() != static_cast<size_t>(N * K))
    throw std::runtime_error("FoldRowsHadamard: W is [" + std::to_string(N) + ", " +
                             std::to_string(K) + "] but Hb is " + std::to_string(hb.K) + " wide");
  ApplyToRows(W.data(), N, K, nthreads, [&](double* v, double*) { hb.Apply(v); });
}

// ---- Hessians ------------------------------------------------------------------------------------

// Throws if any |1 + w| < 1e-3: D^-1 would blow that channel's Hessian row/column up by > 1e3
// (a norm weight of exactly -1 zeroes the channel, and no finite H' represents that).
inline void CheckNormInvertible(const float* norm_w, int64_t K, const std::string& who) {
  for (int64_t k = 0; k < K; ++k) {
    const double d = 1.0 + static_cast<double>(norm_w[k]);
    if (!(std::fabs(d) >= 1e-3)) {
      throw std::runtime_error(who + ": |1 + norm_weight[" + std::to_string(k) + "]| = " +
                               std::to_string(std::fabs(d)) +
                               " < 1e-3 -- the norm cannot be divided out of this linear's Hessian");
    }
  }
}

namespace rotation_detail {

// H (symmetric K x K) <- (H + H^T) / 2, exactly (each pair written by one thread).
inline void SymmetrizeInPlace(float* H, int64_t K, int nthreads) {
  ParallelFor(0, K, nthreads, [&](int64_t r0, int64_t r1) {
    for (int64_t i = r0; i < r1; ++i) {
      for (int64_t j = i + 1; j < K; ++j) {
        const float a = H[i * K + j], b = H[j * K + i];
        const float m = static_cast<float>((static_cast<double>(a) + static_cast<double>(b)) * 0.5);
        H[i * K + j] = m;
        H[j * K + i] = m;
      }
    }
  });
}

}  // namespace rotation_detail

// In-projection tap: H' = Q^T D^-1 H D^-1 Q, D = diag(1 + norm_w) (nullptr: D = I).
inline void TransformHessianQ(std::vector<float>& H, int64_t K, const float* norm_w,
                              const ResidualRotation& q, int nthreads) {
  if (K != q.hidden || H.size() != static_cast<size_t>(K * K))
    throw std::runtime_error("TransformHessianQ: H is " + std::to_string(K) + "^2 but Q is " +
                             std::to_string(q.hidden) + " wide");
  std::vector<double> dinv;
  if (norm_w) {
    CheckNormInvertible(norm_w, K, "TransformHessianQ");
    dinv.resize(static_cast<size_t>(K));
    for (int64_t k = 0; k < K; ++k)
      dinv[static_cast<size_t>(k)] = 1.0 / (1.0 + static_cast<double>(norm_w[k]));
  }
  // Rows: A = D^-1 H D^-1, then A Q. The row index is needed for D^-1's left factor, so this pass
  // is spelled out instead of going through ApplyToRows.
  ParallelFor(0, K, nthreads, [&](int64_t r0, int64_t r1) {
    std::vector<double> v(static_cast<size_t>(K)), tmp(static_cast<size_t>(K));
    for (int64_t r = r0; r < r1; ++r) {
      float* row = H.data() + r * K;
      if (norm_w) {
        const double dr = dinv[static_cast<size_t>(r)];
        for (int64_t k = 0; k < K; ++k)
          v[static_cast<size_t>(k)] = static_cast<double>(row[k]) * dr * dinv[static_cast<size_t>(k)];
      } else {
        for (int64_t k = 0; k < K; ++k) v[static_cast<size_t>(k)] = row[k];
      }
      q.Apply(v.data(), tmp.data());
      for (int64_t k = 0; k < K; ++k) row[k] = static_cast<float>(v[static_cast<size_t>(k)]);
    }
  });
  // Columns: Q^T (A Q).
  ApplyToColumns(H.data(), K, K, nthreads, [&](double* v, double* tmp) { q.Apply(v, tmp); });
  rotation_detail::SymmetrizeInPlace(H.data(), K, nthreads);
}

// Q2b tap: H' = Hb^T H Hb.
inline void TransformHessianHadamard(std::vector<float>& H, int64_t K, const BlockHadamard& hb,
                                     int nthreads) {
  if (K != hb.K || H.size() != static_cast<size_t>(K * K))
    throw std::runtime_error("TransformHessianHadamard: H is " + std::to_string(K) +
                             "^2 but Hb is " + std::to_string(hb.K) + " wide");
  ApplyToRows(H.data(), K, K, nthreads, [&](double* v, double*) { hb.Apply(v); });
  ApplyToColumns(H.data(), K, K, nthreads, [&](double* v, double*) { hb.Apply(v); });
  rotation_detail::SymmetrizeInPlace(H.data(), K, nthreads);
}

// ---- importance vectors (the diagonal model, see the header) --------------------------------------

inline std::vector<float> TransformImportanceQ(const float* v, int64_t K, const float* norm_w,
                                               const ResidualRotation& q) {
  if (K != q.hidden)
    throw std::runtime_error("TransformImportanceQ: K=" + std::to_string(K) + " but Q is " +
                             std::to_string(q.hidden) + " wide");
  if (norm_w) CheckNormInvertible(norm_w, K, "TransformImportanceQ");
  std::vector<double> block_mean(static_cast<size_t>(q.nblk), 0.0);
  for (int64_t c = 0; c < q.nblk; ++c) {
    double s = 0.0;
    for (int64_t i = 0; i < q.block; ++i) {
      const int64_t m = c * q.block + i;
      double e = v[m];
      if (norm_w) {
        const double d = 1.0 + static_cast<double>(norm_w[m]);
        e /= d * d;
      }
      s += e;
    }
    block_mean[static_cast<size_t>(c)] = s / static_cast<double>(q.block);
  }
  std::vector<float> out(static_cast<size_t>(K));
  for (int64_t b = 0; b < q.nblk; ++b) {
    double s = 0.0;
    for (int64_t c = 0; c < q.nblk; ++c) {
      const double r = q.mix[static_cast<size_t>(c * q.nblk + b)];
      s += r * r * block_mean[static_cast<size_t>(c)];
    }
    for (int64_t j = 0; j < q.block; ++j) out[static_cast<size_t>(b * q.block + j)] = static_cast<float>(s);
  }
  return out;
}

inline std::vector<float> TransformImportanceHadamard(const float* v, int64_t K,
                                                      const BlockHadamard& hb) {
  if (K != hb.K)
    throw std::runtime_error("TransformImportanceHadamard: K=" + std::to_string(K) +
                             " but Hb is " + std::to_string(hb.K) + " wide");
  std::vector<float> out(static_cast<size_t>(K));
  for (int64_t b0 = 0; b0 < K; b0 += hb.block) {
    double s = 0.0;
    for (int64_t j = 0; j < hb.block; ++j) s += v[b0 + j];
    const float m = static_cast<float>(s / static_cast<double>(hb.block));
    for (int64_t j = 0; j < hb.block; ++j) out[static_cast<size_t>(b0 + j)] = m;
  }
  return out;
}

}  // namespace r4dx_convert
