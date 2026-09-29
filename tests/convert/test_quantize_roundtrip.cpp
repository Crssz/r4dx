// Quantizer round-trip error bounds on random data (task item 5). Not a byte-exactness test (that
// is test_pack_bytes.cpp) -- this catches a quantizer regression (e.g. a broken scale/zero
// formula) by checking the dequantized reconstruction stays within the error a 4-bit
// grid is expected to produce on Gaussian data, generously bounded so the test isn't flaky.
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "r4dx_convert/quant_int4.hpp"

namespace {

double RelL2Error(const std::vector<float>& a, const std::vector<float>& b) {
  double num = 0.0, den = 0.0;
  for (size_t i = 0; i < a.size(); ++i) {
    double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
    num += d * d;
    den += static_cast<double>(a[i]) * static_cast<double>(a[i]);
  }
  return den > 0.0 ? std::sqrt(num / den) : 0.0;
}

bool CheckBound(const std::string& label, double rel_err, double bound) {
  std::printf("%-28s rel_l2_err=%.4f  (bound %.4f)\n", label.c_str(), rel_err, bound);
  if (!(rel_err < bound)) {
    std::fprintf(stderr, "FAIL: %s exceeded bound\n", label.c_str());
    return false;
  }
  return true;
}

// The int4 quantizers take the group as a plain argument, so both grids are exercised at every
// group r4d_gemm_w4a16_nt_m64 can be built with (R4DX_W4A16_GROUP; the kernel needs a multiple of
// its 64-wide packed block, so 64 and 128 are the whole set). Running both HERE rather than only
// at whatever group this build happens to be configured with means the default build gates the
// group-64 path too.
const int kGroups[] = {128, 64};

bool CheckW4A16(const std::vector<float>& w, int N, int K, int group) {
  using namespace r4dx_convert;
  std::vector<uint8_t> q, zero;
  std::vector<float> scale;
  QuantizeInt4Asymmetric(w.data(), N, K, group, /*nthreads=*/4, q, scale, zero);
  std::vector<float> recon(w.size());
  const int gpr = K / group;
  for (int r = 0; r < N; ++r) {
    for (int k = 0; k < K; ++k) {
      const int g = k / group;
      const size_t gi = static_cast<size_t>(r) * gpr + g;
      recon[static_cast<size_t>(r) * K + k] =
          scale[gi] * (static_cast<float>(q[static_cast<size_t>(r) * K + k]) - zero[gi]);
    }
  }
  // Bound tightened from 0.25 to ~1.3x the observed 0.0999 (review finding, minor: 0.25 was
  // ~2.5x observed, loose enough that a regression doubling quantization error would still pass).
  // A smaller group only ever lowers the error (0.0999 at 128, 0.0836 at 64), so one bound covers
  // both -- it is the loosest group that has to clear it.
  return CheckBound("w4a16 asymmetric g" + std::to_string(group), RelL2Error(w, recon), 0.13);
}

}  // namespace

int main() {
  using namespace r4dx_convert;

  const int N = 64, K = 512;  // N multiple of 16, K multiple of 128 and 32
  std::mt19937 rng(42);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> w(static_cast<size_t>(N) * K);
  for (auto& v : w) v = dist(rng);

  bool ok = true;

  // ---- w4a16 (asymmetric, free zero), at every group ---------------------------------------------
  for (int group : kGroups) {
    ok &= CheckW4A16(w, N, K, group);
  }


  if (!ok) return 1;
  std::printf("PASS\n");
  return 0;
}
