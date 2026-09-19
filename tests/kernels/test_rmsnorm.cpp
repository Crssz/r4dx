// tests/kernels/test_rmsnorm.cpp -- r4dx_rmsnorm_bf16 and its fused r4dx_residual_rmsnorm_bf16
// variant against a CPU fp32 reference implementing Qwen3_5RMSNorm's zero-centered-weight
// convention: out = x * rsqrt(mean(x^2) + eps) * (1 + weight).
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

namespace {

void RmsNormRef(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w, int64_t rows,
                 int64_t hidden, float eps, std::vector<float>* out) {
  out->resize(rows * hidden);
  for (int64_t r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int64_t i = 0; i < hidden; ++i) {
      float v = Bf16ToFloat(x[r * hidden + i]);
      ss += static_cast<double>(v) * v;
    }
    float rstd = 1.0f / std::sqrt(static_cast<float>(ss / hidden) + eps);
    for (int64_t i = 0; i < hidden; ++i) {
      float v = Bf16ToFloat(x[r * hidden + i]);
      float wv = 1.0f + Bf16ToFloat(w[i]);
      (*out)[r * hidden + i] = v * rstd * wv;
    }
  }
}

double MaxRelErr(const std::vector<uint16_t>& got, const std::vector<float>& ref) {
  double max_rel = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    double g = Bf16ToFloat(got[i]);
    double r = ref[i];
    double denom = std::max(1e-3, std::abs(r));
    max_rel = std::max(max_rel, std::abs(g - r) / denom);
  }
  return max_rel;
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));

  const int64_t rows = 17, hidden = 5120;  // 5120 = the model's hidden size
  const float eps = 1e-6f;

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> dist(-2.0f, 2.0f);
  std::vector<uint16_t> x_h(rows * hidden), w_h(hidden);
  for (auto& v : x_h) v = FloatToBf16(dist(rng));
  for (auto& v : w_h) v = FloatToBf16(dist(rng) * 0.1f);  // small: zero-centered weight ~1+w

  DeviceBuffer<uint16_t> x_d(x_h.size()), w_d(w_h.size()), out_d(x_h.size());
  x_d.CopyFromHost(x_h);
  w_d.CopyFromHost(w_h);

  r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(x_d.data()), reinterpret_cast<int64_t>(w_d.data()),
                     reinterpret_cast<int64_t>(out_d.data()), rows, hidden, eps, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> out_h = out_d.CopyToHost();

  std::vector<float> ref;
  RmsNormRef(x_h, w_h, rows, hidden, eps, &ref);
  double rel = MaxRelErr(out_h, ref);
  std::printf("rmsnorm max rel err=%.4e\n", rel);
  bool ok = rel < 2e-2;

  // Fused residual + rmsnorm.
  std::vector<uint16_t> resid_h(rows * hidden);
  for (auto& v : resid_h) v = FloatToBf16(dist(rng));
  DeviceBuffer<uint16_t> resid_d(resid_h.size()), out_resid_d(x_h.size()), out_norm_d(x_h.size());
  resid_d.CopyFromHost(resid_h);

  r4dx_residual_rmsnorm_bf16(
      reinterpret_cast<int64_t>(x_d.data()), reinterpret_cast<int64_t>(resid_d.data()),
      reinterpret_cast<int64_t>(w_d.data()), reinterpret_cast<int64_t>(out_resid_d.data()),
      reinterpret_cast<int64_t>(out_norm_d.data()), rows, hidden, eps, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> out_resid_h = out_resid_d.CopyToHost();
  std::vector<uint16_t> out_norm_h = out_norm_d.CopyToHost();

  std::vector<uint16_t> sum_h(rows * hidden);
  for (int64_t i = 0; i < rows * hidden; ++i) {
    sum_h[i] = FloatToBf16(Bf16ToFloat(x_h[i]) + Bf16ToFloat(resid_h[i]));
  }
  std::vector<float> ref2;
  RmsNormRef(sum_h, w_h, rows, hidden, eps, &ref2);
  double rel_resid = MaxRelErr(out_resid_h, [&] {
    std::vector<float> f(sum_h.size());
    for (size_t i = 0; i < sum_h.size(); ++i) f[i] = Bf16ToFloat(sum_h[i]);
    return f;
  }());
  double rel_norm = MaxRelErr(out_norm_h, ref2);
  std::printf("residual_rmsnorm: residual rel err=%.4e, normed rel err=%.4e\n", rel_resid,
              rel_norm);
  ok = ok && rel_resid < 2e-2 && rel_norm < 2e-2;

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
