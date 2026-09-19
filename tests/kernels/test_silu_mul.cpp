// tests/kernels/test_silu_mul.cpp -- r4dx_silu_mul_bf16 (the MLP's silu(gate) * up) vs a CPU
// fp32 reference. Exercises the FUSED [rows, 2*intermediate] gate_up layout docs/container-
// format.md mandates, with rows>1 (a chunked-prefill shape) -- a flat two-pointer call, which the
// kernel used to require, is only correct for rows==1 and silently reads across row boundaries
// otherwise; per-row distinct random data means any such cross-row read shows up as an error here.
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
float Silu(float x) { return x / (1.0f + std::exp(-x)); }
}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));

  const int64_t rows = 5;
  const int64_t intermediate = 17408;  // the model's MLP intermediate size
  const int64_t row_stride = 2 * intermediate;
  const int64_t n_in = rows * row_stride;
  const int64_t n_out = rows * intermediate;

  std::mt19937 rng(13);
  std::uniform_real_distribution<float> dist(-4.0f, 4.0f);

  std::vector<uint16_t> gate_up_h(n_in);
  for (auto& v : gate_up_h) v = FloatToBf16(dist(rng));

  DeviceBuffer<uint16_t> gate_up_d(n_in), out_d(n_out);
  gate_up_d.CopyFromHost(gate_up_h);

  r4dx_silu_mul_bf16(reinterpret_cast<int64_t>(gate_up_d.data()), reinterpret_cast<int64_t>(out_d.data()),
                      rows, intermediate, row_stride, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> out_h = out_d.CopyToHost();

  double max_rel = 0.0;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t i = 0; i < intermediate; ++i) {
      float g = Bf16ToFloat(gate_up_h[r * row_stride + i]);
      float u = Bf16ToFloat(gate_up_h[r * row_stride + intermediate + i]);
      float ref = Silu(g) * u;
      float got = Bf16ToFloat(out_h[r * intermediate + i]);
      max_rel = std::max(max_rel,
                          static_cast<double>(std::abs(got - ref) / std::max(1e-2f, std::abs(ref))));
    }
  }
  std::printf("silu_mul max rel err=%.4e (rows=%lld, intermediate=%lld)\n", max_rel,
              static_cast<long long>(rows), static_cast<long long>(intermediate));
  bool ok = max_rel < 1e-2;
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
