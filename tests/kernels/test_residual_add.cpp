// tests/kernels/test_residual_add.cpp -- r4dx_residual_add_bf16 vs a CPU fp32 reference.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));

  const int64_t n = 5120 * 13 + 7;  // deliberately not a multiple of the block size
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

  std::vector<uint16_t> a_h(n), b_h(n);
  for (auto& v : a_h) v = FloatToBf16(dist(rng));
  for (auto& v : b_h) v = FloatToBf16(dist(rng));

  DeviceBuffer<uint16_t> a_d(n), b_d(n), out_d(n);
  a_d.CopyFromHost(a_h);
  b_d.CopyFromHost(b_h);

  r4dx_residual_add_bf16(reinterpret_cast<int64_t>(a_d.data()), reinterpret_cast<int64_t>(b_d.data()),
                          reinterpret_cast<int64_t>(out_d.data()), n, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> out_h = out_d.CopyToHost();

  double max_rel = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    float ref = Bf16ToFloat(a_h[i]) + Bf16ToFloat(b_h[i]);
    float got = Bf16ToFloat(out_h[i]);
    max_rel = std::max(max_rel, static_cast<double>(std::abs(got - ref) / std::max(1e-3f, std::abs(ref))));
  }
  std::printf("residual_add max rel err=%.4e\n", max_rel);
  bool ok = max_rel < 5e-3;
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
