// tests/kernels/test_gelu_mul.cpp -- Gemma 4's GeGLU kernels (docs/gemma4-plan.md 3.5):
//   r4dx_gelu_tanh_mul_bf16 vs a CPU reference that follows HF's op sequence (bf16(gelu_tanh(gate)),
//   then bf16(that * up)) on the fused [rows, 2*I] gate_up layout with rows > 1 and a padded row
//   stride; r4dx_gelu_tanh_mul_wide_bf16 byte-identical to it; and r4dx_gelu_tanh_mul_trellis_bf16
//   byte-identical to the v1 pair (the wide GeGLU, then r4dx_trellis_input_bf16 with nout = 1).
// GPU test.
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
  std::printf("%-84s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}
int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

void Run(std::mt19937& rng, int64_t rows, int64_t intermediate, int64_t pad) {
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);
  const int64_t stride = 2 * intermediate + pad;
  std::vector<uint16_t> gu_h(static_cast<size_t>(rows * stride));
  for (auto& v : gu_h) v = FloatToBf16(dist(rng));
  DeviceBuffer<uint16_t> gu_d(gu_h.size()), plain_d(static_cast<size_t>(rows * intermediate)),
      wide_d(static_cast<size_t>(rows * intermediate));
  gu_d.CopyFromHost(gu_h);
  plain_d.Zero();
  wide_d.Zero();
  r4dx_gelu_tanh_mul_bf16(P(gu_d.data()), P(plain_d.data()), rows, intermediate, stride, 0);
  r4dx_gelu_tanh_mul_wide_bf16(P(gu_d.data()), P(wide_d.data()), rows, intermediate, stride, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<uint16_t> plain = plain_d.CopyToHost(), wide = wide_d.CopyToHost();

  // CPU reference: allow one bf16 step (the double gelu can land on the other side of a rounding tie
  // than the device's fp32 tanhf).
  int64_t bad = 0;
  double worst = 0.0;
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t i = 0; i < intermediate; ++i) {
      const double g = Bf16ToFloat(gu_h[r * stride + i]);
      const double u = Bf16ToFloat(gu_h[r * stride + intermediate + i]);
      const double a = gemma_ref::Bf16Round(static_cast<float>(gemma_ref::GeluTanh(g)));
      const double ref = a * u;
      const double got = Bf16ToFloat(plain[r * intermediate + i]);
      const double tol = std::abs(ref) / 64.0 + std::abs(u) * 0.01 + 1e-6;  // activation step, then product step
      worst = std::max(worst, std::abs(got - ref) / tol);
      if (std::abs(got - ref) > tol) ++bad;
    }
  }
  char msg[200];
  std::snprintf(msg, sizeof msg, "rows %lld I %lld stride %lld: gelu_tanh_mul matches the reference (%lld bad, worst %.2f)",
                (long long)rows, (long long)intermediate, (long long)stride, (long long)bad, worst);
  Check(bad == 0, msg);
  std::snprintf(msg, sizeof msg, "rows %lld I %lld: wide is byte-identical to plain", (long long)rows,
                (long long)intermediate);
  Check(plain == wide, msg);

  // The trellis producer against the v1 pair, only for K a multiple of 128.
  if (intermediate % 128 != 0) return;
  std::vector<float> suh_h(static_cast<size_t>(intermediate));
  for (auto& v : suh_h) v = (rng() & 1) ? 1.0f : -1.0f;
  DeviceBuffer<float> suh_d(suh_h.size());
  suh_d.CopyFromHost(suh_h);
  DeviceBuffer<uint16_t> fused_d(static_cast<size_t>(rows * intermediate)), pair_d(fused_d.size());
  fused_d.Zero();
  pair_d.Zero();
  for (int prescale : {0, -3}) {
    r4dx_gelu_tanh_mul_trellis_bf16(P(gu_d.data()), rows, intermediate, stride, P(suh_d.data()),
                                     P(fused_d.data()), prescale, 0);
    const int64_t suh_p = P(suh_d.data()), out_p = P(pair_d.data());
    r4dx_trellis_input_bf16(P(wide_d.data()), rows, intermediate, 1, &suh_p, &out_p, prescale, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    std::snprintf(msg, sizeof msg, "rows %lld I %lld prescale %d: trellis producer == wide GeGLU + trellis_input",
                  (long long)rows, (long long)intermediate, prescale);
    Check(fused_d.CopyToHost() == pair_d.CopyToHost(), msg);
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(17);
  Run(rng, 1, 15360, 0);   // the model's intermediate at decode
  Run(rng, 5, 15360, 0);
  Run(rng, 3, 7680, 0);    // TP=2 per rank
  Run(rng, 4, 2056, 0);    // partial last 2048 slice
  Run(rng, 2, 1000, 0);    // not a multiple of 128 (no trellis check)
  Run(rng, 3, 1024, 16);   // padded row stride
  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
