// tests/model/numpy_legacy_rng.hpp -- a bit-exact C++ reimplementation of
// `numpy.random.RandomState(seed).randn(...)`, so a C++ test can reconstruct the SYNTHETIC inputs
// `tools/reference/dflash2_ref.py` generates for its golden fixtures without those inputs having to
// be dumped to disk.
//
// WHY THIS EXISTS. Two of the DFlash2 fixtures' inputs are deliberately NOT stored
// (docs/dflash2.md section 9): the synthetic target's `[4096, 5120]` embedding table and lm_head
// (~80 MB each as fp32, far over that directory's 60 MB budget) and fixture B's `[2100, 25600]`
// feature matrix (~215 MB). The manifest instead records the exact formula
// (`RandomState(seed).randn(vocab, hidden) * 0.02` for the embedding table, then the lm_head drawn
// from the SAME RandomState immediately after; `RandomState(seed).randn(n, 25600)` for features),
// and `dflash2_selftest.py` reconstructs them in-memory rather than loading them. This header lets
// `tests/model/test_dflash_draft.cpp` do exactly the same thing.
//
// WHY IT IS TRUSTWORTHY. It is not assumed correct: the test that uses it first regenerates
// fixture A's own `features.npy` (which IS dumped, `RandomState(0).randn(40, 25600)`) and asserts
// BIT-EXACT equality against the stored file before relying on this generator for anything that is
// not dumped. Fixture B's features are the same RNG stream continued (C-order fill, same seed), so
// that check covers the identical code path.
//
// WHAT IS REPRODUCED, from numpy's own C sources (the legacy RandomState path, unchanged since
// numpy 1.17 moved it to `_bounded_integers`/`legacy-distributions.c` precisely so old streams stay
// reproducible):
//   * `mt19937_seed`: key[i] = s; s = 1812433253*(s ^ (s>>30)) + i + 1  (identical in effect to
//     Matsumoto's `init_genrand`, which writes mt[i] = 1812433253*(mt[i-1]^(mt[i-1]>>30)) + i).
//   * `mt19937_next`: the standard MT19937 twist + tempering.
//   * `mt19937_next_double`: a = next()>>5, b = next()>>6, (a*67108864 + b) / 2^53.
//   * `legacy_gauss`: Marsaglia polar method over pairs of those doubles, caching the second
//     variate -- NOT the Ziggurat that `numpy.random.Generator` (the NEW API) uses, which would
//     produce a completely different stream.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstddef>

namespace r4dx_test {

class NumpyLegacyRng {
 public:
  explicit NumpyLegacyRng(uint32_t seed) {
    uint32_t s = seed;
    for (int i = 0; i < kN; ++i) {
      key_[i] = s;
      s = (1812433253u * (s ^ (s >> 30)) + static_cast<uint32_t>(i) + 1u);
    }
    pos_ = kN;
    has_gauss_ = false;
    gauss_ = 0.0;
  }

  uint32_t NextUint32() {
    if (pos_ >= kN) Twist();
    uint32_t y = key_[pos_++];
    y ^= (y >> 11);
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= (y >> 18);
    return y;
  }

  double NextDouble() {
    const uint32_t a = NextUint32() >> 5;
    const uint32_t b = NextUint32() >> 6;
    return (static_cast<double>(a) * 67108864.0 + static_cast<double>(b)) / 9007199254740992.0;
  }

  // numpy's `legacy_gauss`.
  double Gauss() {
    if (has_gauss_) {
      has_gauss_ = false;
      const double t = gauss_;
      gauss_ = 0.0;
      return t;
    }
    double x1, x2, r2;
    do {
      x1 = 2.0 * NextDouble() - 1.0;
      x2 = 2.0 * NextDouble() - 1.0;
      r2 = x1 * x1 + x2 * x2;
    } while (r2 >= 1.0 || r2 == 0.0);
    const double f = std::sqrt(-2.0 * std::log(r2) / r2);
    gauss_ = f * x1;
    has_gauss_ = true;
    return f * x2;
  }

  // `rng.randn(...).astype(np.float32)` over `n` elements in C order.
  void FillRandnF32(float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(Gauss());
  }

  // `(rng.randn(...) * scale).astype(np.float32)` -- the multiply happens in float64 FIRST (the
  // order SyntheticTarget.__init__ uses), so do not fold `scale` into the float32 result.
  void FillRandnScaledF32(float* out, size_t n, double scale) {
    for (size_t i = 0; i < n; ++i) out[i] = static_cast<float>(Gauss() * scale);
  }

 private:
  static constexpr int kN = 624;
  static constexpr int kM = 397;

  void Twist() {
    static const uint32_t mag01[2] = {0x0u, 0x9908b0dfu};
    uint32_t y;
    int i = 0;
    for (; i < kN - kM; ++i) {
      y = (key_[i] & 0x80000000u) | (key_[i + 1] & 0x7fffffffu);
      key_[i] = key_[i + kM] ^ (y >> 1) ^ mag01[y & 0x1u];
    }
    for (; i < kN - 1; ++i) {
      y = (key_[i] & 0x80000000u) | (key_[i + 1] & 0x7fffffffu);
      key_[i] = key_[i + (kM - kN)] ^ (y >> 1) ^ mag01[y & 0x1u];
    }
    y = (key_[kN - 1] & 0x80000000u) | (key_[0] & 0x7fffffffu);
    key_[kN - 1] = key_[kM - 1] ^ (y >> 1) ^ mag01[y & 0x1u];
    pos_ = 0;
  }

  uint32_t key_[kN];
  int pos_ = kN;
  bool has_gauss_ = false;
  double gauss_ = 0.0;
};

}  // namespace r4dx_test
