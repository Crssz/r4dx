// tests/kernels/test_rmsnorm_noscale.cpp -- r4dx_rmsnorm_noscale_bf16 (Gemma 4's v_norm, docs/gemma4-
// plan.md 3.5) vs a double-precision CPU reference, at the widths it runs at (v_norm rows are
// T * kv_heads with hidden = head_dim 256 / 512; 3840 is the model width), out of place and in
// place, plus a cross-check against r4dx_rmsnorm_plain_bf16 with an all-ones weight (the same
// arithmetic, so they must agree bit for bit). GPU test.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "gemma_ref.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/gemma_kernels.h"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

namespace {

bool g_ok = true;
void Check(bool cond, const char* what) {
  std::printf("%-76s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}
int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

// Within one bf16 step of the double reference (relative 2^-7 plus a tiny absolute floor).
bool WithinOneStep(const std::vector<uint16_t>& got, const std::vector<float>& ref, double* worst) {
  *worst = 0.0;
  bool ok = true;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::abs(static_cast<double>(Bf16ToFloat(got[i])) - ref[i]);
    const double tol = std::abs(static_cast<double>(ref[i])) * (1.0 / 128.0) + 1e-6;
    *worst = std::max(*worst, d / tol);
    if (d > tol) ok = false;
  }
  return ok;
}

void Run(std::mt19937& rng, int64_t rows, int64_t hidden, float scale_mag) {
  std::uniform_real_distribution<float> dist(-scale_mag, scale_mag);
  std::vector<uint16_t> x_h(static_cast<size_t>(rows * hidden));
  for (auto& v : x_h) v = FloatToBf16(dist(rng));
  const float eps = 1e-6f;

  std::vector<float> ref(x_h.size());
  for (int64_t r = 0; r < rows; ++r) {
    std::vector<float> row(hidden);
    for (int64_t i = 0; i < hidden; ++i) row[i] = Bf16ToFloat(x_h[r * hidden + i]);
    const std::vector<float> o = gemma_ref::RmsNormRow(row, nullptr, eps);
    std::copy(o.begin(), o.end(), ref.begin() + r * hidden);
  }

  DeviceBuffer<uint16_t> x_d(x_h.size()), out_d(x_h.size()), inpl_d(x_h.size());
  x_d.CopyFromHost(x_h);
  inpl_d.CopyFromHost(x_h);
  r4dx_rmsnorm_noscale_bf16(P(x_d.data()), P(out_d.data()), rows, hidden, eps, 0);
  r4dx_rmsnorm_noscale_bf16(P(inpl_d.data()), P(inpl_d.data()), rows, hidden, eps, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<uint16_t> got = out_d.CopyToHost();

  double worst;
  char msg[160];
  std::snprintf(msg, sizeof msg, "rows %lld hidden %lld mag %.2g: matches the reference", (long long)rows,
                (long long)hidden, scale_mag);
  Check(WithinOneStep(got, ref, &worst), msg);
  std::snprintf(msg, sizeof msg, "rows %lld hidden %lld: in place equals out of place", (long long)rows,
                (long long)hidden);
  Check(inpl_d.CopyToHost() == got, msg);

  // All-ones weight through r4dx_rmsnorm_plain_bf16: the same reduction and the same expression
  // order (x * rstd * 1.0f), so the bytes are equal.
  std::vector<uint16_t> ones(static_cast<size_t>(hidden), FloatToBf16(1.0f));
  DeviceBuffer<uint16_t> w_d(ones.size()), plain_d(x_h.size());
  w_d.CopyFromHost(ones);
  r4dx_rmsnorm_plain_bf16(P(x_d.data()), P(w_d.data()), P(plain_d.data()), rows, hidden, eps, 0, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::snprintf(msg, sizeof msg, "rows %lld hidden %lld: equals r4dx_rmsnorm_plain_bf16 with w = 1",
                (long long)rows, (long long)hidden);
  Check(plain_d.CopyToHost() == got, msg);
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(5);
  Run(rng, 13 * 8, 256, 3.0f);   // v_norm on a sliding layer: T = 13, 8 kv heads
  Run(rng, 13, 512, 3.0f);       // v_norm on a full layer: 1 kv head, head_dim 512
  Run(rng, 5, 3840, 8.0f);       // model width
  Run(rng, 3, 256, 1.0e-3f);     // small magnitudes: eps matters
  Run(rng, 2, 4096 + 7, 2.0f);   // a width that is not a multiple of the block
  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
