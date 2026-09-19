// tests/kernels/test_quant_act_fp8.cpp -- r4dx_quant_act_fp8e4m3_row (feeds
// r4d_gemm_mxfp4a8_nt_m64) vs a CPU reference: per-row scale = max(1e-8,absmax)/448, plain
// row-major fp8e4m3 bytes (see kernels.h's comment on why this differs from r4d_quant_act_i8's
// WMMA-fragment byte reorder).
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

  const int M = 9, K = 5120;
  std::mt19937 rng(23);
  std::uniform_real_distribution<float> dist(-6.0f, 6.0f);

  std::vector<uint16_t> x_h(static_cast<size_t>(M) * K);
  for (auto& v : x_h) v = FloatToBf16(dist(rng));

  DeviceBuffer<uint16_t> x_d(x_h.size());
  DeviceBuffer<uint8_t> q_d(x_h.size());
  DeviceBuffer<float> scale_d(M);
  x_d.CopyFromHost(x_h);

  r4dx_quant_act_fp8e4m3_row(reinterpret_cast<int64_t>(x_d.data()),
                              reinterpret_cast<int64_t>(q_d.data()),
                              reinterpret_cast<int64_t>(scale_d.data()), M, K, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());

  std::vector<uint8_t> q_got = q_d.CopyToHost();
  std::vector<float> scale_got = scale_d.CopyToHost();

  double max_rel = 0.0, max_scale_rel = 0.0;
  for (int m = 0; m < M; ++m) {
    float absmax = 0.0f;
    for (int k = 0; k < K; ++k) absmax = std::max(absmax, std::abs(Bf16ToFloat(x_h[m * K + k])));
    float scale_ref = std::max(1e-8f, absmax) / 448.0f;
    max_scale_rel = std::max(max_scale_rel, static_cast<double>(
                              std::abs(scale_got[m] - scale_ref) / std::max(1e-8f, scale_ref)));
    for (int k = 0; k < K; ++k) {
      float x = Bf16ToFloat(x_h[m * K + k]);
      uint8_t q_ref = FloatToFp8E4M3(x / scale_ref);
      float dequant_ref = Fp8E4M3ToFloat(q_ref) * scale_ref;
      float dequant_got = Fp8E4M3ToFloat(q_got[m * K + k]) * scale_got[m];
      max_rel = std::max(max_rel, static_cast<double>(std::abs(dequant_got - dequant_ref) /
                                       std::max(1e-2f, std::abs(dequant_ref))));
    }
  }
  std::printf("quant_act_fp8 scale max rel err=%.4e, dequant max rel err=%.4e\n", max_scale_rel,
              max_rel);
  bool ok = max_scale_rel < 1e-4 && max_rel < 1e-3;
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
