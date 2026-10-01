// tests/kernels/test_widen_softcap.cpp -- r4dx_model_widen_softcap_bf16_to_f32 (Gemma 4's final logit
// softcap fused into the lm_head widen, docs/gemma4-plan.md 3.5): out = cap * tanh(x / cap) in fp32,
// vs a double reference over logits spanning well past the cap, the cap-dominated tail (|out| <= cap
// always), x = 0, and the cap = 30 / non-default cap cases. GPU test.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "kernels/model_kernels.h"

using namespace r4dx::core;

namespace {
bool g_ok = true;
void Check(bool cond, const char* what) {
  std::printf("%-72s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}
}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(3);
  for (float cap : {30.0f, 12.5f}) {
    // A vocab-sized row plus a size that is not a multiple of the block.
    for (int64_t n : {int64_t{262144}, int64_t{131072 + 5}}) {
      std::uniform_real_distribution<float> dist(-150.0f, 150.0f);
      std::vector<uint16_t> in_h(static_cast<size_t>(n));
      for (auto& v : in_h) v = FloatToBf16(dist(rng));
      in_h[0] = FloatToBf16(0.0f);
      in_h[1] = FloatToBf16(1.0e-3f);
      DeviceBuffer<uint16_t> in_d(in_h.size());
      DeviceBuffer<float> out_d(in_h.size());
      in_d.CopyFromHost(in_h);
      r4dx_model_widen_softcap_bf16_to_f32(reinterpret_cast<int64_t>(in_d.data()),
                                            reinterpret_cast<int64_t>(out_d.data()), n, cap, 0);
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      const std::vector<float> got = out_d.CopyToHost();
      double worst = 0.0;
      bool bounded = true;
      for (size_t i = 0; i < got.size(); ++i) {
        const double x = Bf16ToFloat(in_h[i]);
        const double ref = static_cast<double>(cap) * std::tanh(x / cap);
        worst = std::max(worst, std::abs(got[i] - ref));
        if (std::abs(got[i]) > cap) bounded = false;
      }
      char msg[160];
      std::snprintf(msg, sizeof msg, "cap %.1f n %lld: max |d| vs double %.3e", cap, (long long)n, worst);
      Check(worst < 1e-4, msg);
      Check(bounded, "    |out| <= cap everywhere");
      Check(got[0] == 0.0f, "    softcap(0) == 0");
    }
  }
  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
