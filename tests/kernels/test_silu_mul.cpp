// tests/kernels/test_silu_mul.cpp -- r4dx_silu_mul_bf16 (the MLP's silu(gate) * up) vs a CPU
// fp32 reference. Exercises the FUSED [rows, 2*intermediate] gate_up layout docs/container-
// format.md mandates, with rows>1 (a chunked-prefill shape) -- a flat two-pointer call, which the
// kernel used to require, is only correct for rows==1 and silently reads across row boundaries
// otherwise; per-row distinct random data means any such cross-row read shows up as an error here.
//
// Then r4dx_silu_mul_wide_bf16 (the trellis body's wide-grid form, docs/trellis-kernel.md 10.2)
// byte-identical to r4dx_silu_mul_bf16 on the vectorized path (the model's widths, one row and
// several, a width that is not a whole number of 2048-element slices) and on the scalar fallback (a
// width that is not a multiple of 8, and a gate_up buffer that is not 16-byte aligned).
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

// The wide form against the plain one on the same input: every output byte equal. `offset`
// elements are skipped at the start of the gate_up allocation (1 = misaligned rows).
bool WideMatchesPlain(std::mt19937& rng, int64_t rows, int64_t intermediate, int64_t offset) {
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
  const int64_t stride = 2 * intermediate;
  std::vector<uint16_t> gu_h(static_cast<size_t>(offset + rows * stride));
  for (auto& v : gu_h) v = FloatToBf16(dist(rng));
  DeviceBuffer<uint16_t> gu_d(gu_h.size()), plain_d(static_cast<size_t>(rows * intermediate)),
      wide_d(static_cast<size_t>(rows * intermediate));
  gu_d.CopyFromHost(gu_h);
  plain_d.Zero();
  wide_d.Zero();
  const int64_t gu = reinterpret_cast<int64_t>(gu_d.data() + offset);
  r4dx_silu_mul_bf16(gu, reinterpret_cast<int64_t>(plain_d.data()), rows, intermediate, stride, 0);
  r4dx_silu_mul_wide_bf16(gu, reinterpret_cast<int64_t>(wide_d.data()), rows, intermediate, stride,
                          0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<uint16_t> plain = plain_d.CopyToHost(), wide = wide_d.CopyToHost();
  int64_t diff = 0;
  for (size_t i = 0; i < plain.size(); ++i) diff += plain[i] != wide[i];
  std::printf("silu_mul_wide rows=%lld intermediate=%lld offset=%lld: %lld differing element(s)\n",
              static_cast<long long>(rows), static_cast<long long>(intermediate),
              static_cast<long long>(offset), static_cast<long long>(diff));
  return diff == 0;
}
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

  // rows x intermediate x gate_up offset: the model's widths (17408 and its TP = 2 half 8704) at
  // decode and prefill rows, a width with a partial last slice (2056 = 2048 + 8), the tiny test
  // container's 1024, the scalar path (1000 is not a multiple of 8; offset 1 breaks alignment).
  struct Case {
    int64_t rows, intermediate, offset;
  };
  const Case cases[] = {{1, 17408, 0}, {3, 17408, 0}, {64, 17408, 0}, {1, 8704, 0}, {7, 2056, 0},
                        {2, 1024, 0},  {3, 1000, 0},  {2, 17408, 1}, {1, 8, 0},    {1, 5, 0}};
  for (const Case& c : cases) ok = WideMatchesPlain(rng, c.rows, c.intermediate, c.offset) && ok;

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
