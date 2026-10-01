// tests/kernels/test_rotate_residual_f32.cpp -- r4dx_rotate_residual_f32 and r4dx_post_rmsnorm_rotate_add_f32res
// (the fp32-residual twins of the Gemma option-A rotation kernels) against rotation_ref.hpp's fp64 reference:
// hidden 3840 (15 x 256, Gemma 4), 1280 and 5120; forward and inverse Q, the round trip, the fused post-norm +
// rotate + add with scalar 1.0 and a layer_scalar, row independence (bit-exact), and precondition throws.
// Values are held to fp32 precision (no bf16 rounding exists in these kernels): 1e-5 of the row's rms.
// GPU test (HIP device 1 via HIP_VISIBLE_DEVICES=1); WRITTEN, NOT YET RUN. Built with /EHc-.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/rotate_residual.h"
#include "rotation_ref.hpp"

using namespace r4dx::core;

namespace {

int g_failures = 0;
void Check(bool cond, const std::string& what) {
  std::printf("%-90s %s\n", what.c_str(), cond ? "ok" : "FAIL");
  if (!cond) ++g_failures;
}
template <typename T>
int64_t P(T* p) {
  return reinterpret_cast<int64_t>(p);
}

double MaxRelToRms(const std::vector<float>& got, const std::vector<double>& ref) {
  double ss = 0.0;
  for (double v : ref) ss += v * v;
  const double rms = std::sqrt(ss / static_cast<double>(ref.size())) + 1e-30;
  double w = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) w = std::max(w, std::abs(got[i] - ref[i]) / rms);
  return w;
}

void RunHidden(std::mt19937_64& rng, int64_t hidden) {
  using namespace rotation_ref;
  const int64_t block = ChooseBlock(hidden), nblk = hidden / block;
  const std::vector<float> d = RandomSigns(rng, hidden);
  const std::vector<float> R = RandomOrthogonal(rng, static_cast<int>(nblk));
  DeviceBuffer<float> d_d(hidden), R_d(nblk * nblk);
  d_d.CopyFromHost(d);
  R_d.CopyFromHost(R);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  const int64_t rows = 3;
  std::vector<float> x(rows * hidden);
  for (auto& v : x) v = nd(rng) * 150.0f;  // residual-scale, genuinely fp32 (not bf16-representable)

  for (int inverse = 0; inverse <= 1; ++inverse) {
    DeviceBuffer<float> x_d(x.size());
    x_d.CopyFromHost(x);
    r4dx_rotate_residual_f32(P(x_d.data()), rows, hidden, P(d_d.data()), P(R_d.data()), inverse, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const std::vector<float> got = x_d.CopyToHost();
    double worst = 0.0;
    for (int64_t r = 0; r < rows; ++r) {
      std::vector<double> xr(x.begin() + r * hidden, x.begin() + (r + 1) * hidden);
      const auto ref = ApplyQGeneral(xr, block, d.data(), R.data(), inverse != 0);
      std::vector<float> gr(got.begin() + r * hidden, got.begin() + (r + 1) * hidden);
      worst = std::max(worst, MaxRelToRms(gr, ref));
    }
    char msg[160];
    std::snprintf(msg, sizeof msg, "hidden %lld rotate_residual_f32 %s: max err / rms = %.2e", (long long)hidden,
                  inverse ? "Q^T" : "Q", worst);
    Check(worst < 1e-5, msg);
  }
  {  // round trip
    DeviceBuffer<float> x_d(x.size());
    x_d.CopyFromHost(x);
    r4dx_rotate_residual_f32(P(x_d.data()), rows, hidden, P(d_d.data()), P(R_d.data()), 0, 0);
    r4dx_rotate_residual_f32(P(x_d.data()), rows, hidden, P(d_d.data()), P(R_d.data()), 1, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const auto got = x_d.CopyToHost();
    double w = 0.0;
    for (size_t i = 0; i < x.size(); ++i) w = std::max(w, static_cast<double>(std::abs(got[i] - x[i])));
    char msg[120];
    std::snprintf(msg, sizeof msg, "hidden %lld Q then Q^T round trip: max abs err %.2e on values ~150", (long long)hidden, w);
    Check(w < 5e-3, msg);
  }

  // fused post-norm + rotate + add
  for (float layer_scale : {1.0f, Bf16ToFloat(FloatToBf16(0.37f))}) {
    std::vector<float> resid(rows * hidden);
    for (auto& v : resid) v = nd(rng) * 150.0f;
    std::vector<uint16_t> y(rows * hidden), w(hidden);
    for (auto& v : y) v = FloatToBf16(nd(rng) * 20.0f);
    for (auto& v : w) v = FloatToBf16(nd(rng));
    DeviceBuffer<float> r_d(resid.size());
    DeviceBuffer<uint16_t> y_d(y.size()), w_d(hidden);
    r_d.CopyFromHost(resid);
    y_d.CopyFromHost(y);
    w_d.CopyFromHost(w);
    r4dx_post_rmsnorm_rotate_add_f32res(P(r_d.data()), P(y_d.data()), P(w_d.data()), P(d_d.data()), P(R_d.data()), rows,
                                         hidden, 1e-6f, layer_scale, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const auto got = r_d.CopyToHost();
    double worst = 0.0;
    std::vector<double> wd(hidden);
    for (int64_t i = 0; i < hidden; ++i) wd[i] = Bf16ToFloat(w[i]);
    for (int64_t r = 0; r < rows; ++r) {
      std::vector<double> rr(resid.begin() + r * hidden, resid.begin() + (r + 1) * hidden), yr(hidden);
      for (int64_t i = 0; i < hidden; ++i) yr[i] = Bf16ToFloat(y[r * hidden + i]);
      const auto ref = PostNormRotateAddRef(rr, yr, wd, block, d.data(), R.data(), 1e-6, layer_scale);
      std::vector<float> gr(got.begin() + r * hidden, got.begin() + (r + 1) * hidden);
      worst = std::max(worst, MaxRelToRms(gr, ref));
    }
    char msg[160];
    std::snprintf(msg, sizeof msg, "hidden %lld post_rmsnorm_rotate_add_f32res layer_scale %.3g: max err / rms = %.2e",
                  (long long)hidden, layer_scale, worst);
    Check(worst < 1e-5, msg);

    // row independence: row 1 alone gives the same bits as inside the 3-row launch.
    DeviceBuffer<float> one(hidden);
    one.CopyFromHost(std::vector<float>(resid.begin() + hidden, resid.begin() + 2 * hidden));
    r4dx_post_rmsnorm_rotate_add_f32res(P(one.data()), P(y_d.data() + hidden), P(w_d.data()), P(d_d.data()), P(R_d.data()),
                                         1, hidden, 1e-6f, layer_scale, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const auto g1 = one.CopyToHost();
    Check(std::memcmp(g1.data(), got.data() + hidden, hidden * sizeof(float)) == 0,
          "    row independence (bit-exact, 1 row vs 3 rows)");
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937_64 rng(53);
  for (int64_t hidden : {int64_t{3840}, int64_t{1280}, int64_t{5120}}) RunHidden(rng, hidden);
  {
    DeviceBuffer<float> a(3840);
    DeviceBuffer<uint16_t> y(3840), w(3840);
    const auto throws = [](auto fn) {
      try {
        fn();
      } catch (const std::runtime_error&) {
        return true;
      }
      return false;
    };
    Check(throws([&] { r4dx_rotate_residual_f32(0, 1, 3840, 0, 0, 0, 0); }), "rotate_residual_f32: null pointers throw");
    Check(throws([&] { r4dx_post_rmsnorm_rotate_add_f32res(P(a.data()), 0, P(w.data()), 0, 0, 1, 3840, 1e-6f, 1.0f, 0); }),
          "post_rmsnorm_rotate_add_f32res: null y throws");
    Check(throws([&] { r4dx_rotate_residual_f32(P(a.data()), 1, 7 * 1024 * 33, 1, 1, 0, 0); }),
          "rotate_residual_f32: unsupported hidden throws");
  }
  std::printf(g_failures == 0 ? "PASS\n" : "FAIL\n");
  return g_failures == 0 ? 0 : 1;
}
