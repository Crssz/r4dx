// tests/model/qwen4exp/test_gpu_w4a16_k640.cpp -- a w4a16 (and bf16) linear whose K is 640 = 5 * 128
// (qwen4_exp's expert `down`, N = 2560) through ApplyLinear. linear.cpp's FallbackTuning used to demand
// K % 256 and so refused to pick a tuning for it; it now takes SK = 2 for K % 128 == 0. This runs the
// real path (PickTuning -> r4d_gemm_w4a16_nt_m64[_g] / r4d_gemm_bf16_nt_m64) on random weights and
// compares with a CPU reference:
//   w4a16: random codes q (0..15), f16 scales and zeros, packed exactly as the kernel reads them (the
//          layout of tests/convert/test_kernel_decode.cpp: dword ((t*(K/64)+kb)*32+lane)*4+s, nibble
//          2e / 2(e-4)+1 of element e holding code ^ 8; wsz[(t*(K/g)+g)*16+r] = f16 scale | f16(-(1024 +
//          zero)) << 16), W[n][k] = scale * (q - zero) in fp64, A = x rounded to f16 (exact for bf16 x),
//          at the build's default group 64 and at the per-tensor group 32;
//   bf16:  W bf16 [N][K].
// M in {1, 2, 8, 16, 17, 64}. Tolerance per element: 2 bf16 ulp of the reference + 4e-3 * rms(row)
// (the weights are rounded to f16 inside the kernel and the sum has 640 terms; an element that
// cancels has no meaningful ulp). Prints the max abs error and the worst error as a fraction of the
// tolerance. Also asserts the picked tuning is SK = 2 at K = 640.
//
// GPU test: HIP device 1 via HIP_VISIBLE_DEVICES=1 (tests/model/qwen4exp/CMakeLists.txt).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <random>
#include <string>
#include <vector>

#include "linear.h"
#include "quant_linear.h"
#include "r4d.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/stream.hpp"

using r4dx::core::Bf16ToFloat;
using r4dx::core::DeviceBuffer;
using r4dx::core::F16ToFloat;
using r4dx::core::FloatToBf16;
using r4dx::core::FloatToF16;
using r4dx::model::ApplyLinear;
using r4dx::model::Layout;
using r4dx::model::QuantLinear;

namespace {

int g_checks = 0, g_failures = 0;

void Check(bool ok, const std::string& what) {
  ++g_checks;
  if (ok) return;
  ++g_failures;
  if (g_failures <= 40) std::fprintf(stderr, "[FAIL] %s\n", what.c_str());
}

constexpr int kN = 2560, kK = 640;
constexpr int kNibbleKOffset[8] = {0, 8, 1, 9, 2, 10, 3, 11};

struct Weights {
  std::vector<double> w;  // [N][K] the linear's exact weights
};

// Random codes / scales / zeros -> the packed buffers and the exact weights.
Weights MakeW4a16(std::mt19937_64& rng, int group, QuantLinear& q) {
  const int ntiles = kN / 16, kblocks = kK / 64, gpr = kK / group;
  std::uniform_int_distribution<int> cd(0, 15);
  std::uniform_real_distribution<float> sd(0.002f, 0.02f);
  std::vector<uint8_t> code(static_cast<size_t>(kN) * kK);
  std::vector<uint16_t> sc(static_cast<size_t>(kN) * gpr);
  std::vector<int> zero(static_cast<size_t>(kN) * gpr);
  for (auto& c : code) c = static_cast<uint8_t>(cd(rng));
  for (auto& s : sc) s = FloatToF16(sd(rng));
  for (auto& z : zero) z = cd(rng);
  std::vector<uint32_t> wq(static_cast<size_t>(ntiles) * kblocks * 32 * 4);
  for (int t = 0; t < ntiles; ++t)
    for (int kb = 0; kb < kblocks; ++kb)
      for (int lane = 0; lane < 32; ++lane)
        for (int s = 0; s < 4; ++s) {
          uint32_t dword = 0;
          const int row = t * 16 + (lane & 15);
          for (int e = 0; e < 8; ++e) {
            const int k = (kb * 4 + s) * 16 + 8 * (e >> 2) + 4 * (lane >> 4) + (e & 3);
            const int pos = e < 4 ? 2 * e : 2 * (e - 4) + 1;
            dword |= static_cast<uint32_t>((code[static_cast<size_t>(row) * kK + k] ^ 8) & 0xF) << (4 * pos);
          }
          wq[((static_cast<size_t>(t) * kblocks + kb) * 32 + lane) * 4 + s] = dword;
        }
  std::vector<uint32_t> wsz(static_cast<size_t>(ntiles) * gpr * 16);
  for (int t = 0; t < ntiles; ++t)
    for (int g = 0; g < gpr; ++g)
      for (int r = 0; r < 16; ++r) {
        const size_t gi = static_cast<size_t>(t * 16 + r) * gpr + g;
        wsz[(static_cast<size_t>(t) * gpr + g) * 16 + r] =
            sc[gi] | (static_cast<uint32_t>(FloatToF16(-(1024.0f + static_cast<float>(zero[gi])))) << 16);
      }
  q.layout = Layout::kW4a16;
  q.N = kN;
  q.K = kK;
  q.w4a16_group = group == r4d_gemm_w4a16_nt_m64_group() ? 0 : group;
  q.wq = DeviceBuffer<uint8_t>(wq.size() * 4);
  R4DX_HIP_CHECK(hipMemcpy(q.wq.data(), wq.data(), wq.size() * 4, hipMemcpyHostToDevice));
  q.w4a16_wsz = DeviceBuffer<uint32_t>(wsz.size());
  q.w4a16_wsz.CopyFromHost(wsz);
  Weights W;
  W.w.resize(static_cast<size_t>(kN) * kK);
  for (int n = 0; n < kN; ++n)
    for (int k = 0; k < kK; ++k) {
      const size_t gi = static_cast<size_t>(n) * gpr + k / group;
      W.w[static_cast<size_t>(n) * kK + k] =
          static_cast<double>(F16ToFloat(sc[gi])) * (code[static_cast<size_t>(n) * kK + k] - zero[gi]);
    }
  return W;
}

Weights MakeBf16(std::mt19937_64& rng, QuantLinear& q) {
  std::normal_distribution<float> nd(0.f, 0.02f);
  std::vector<uint16_t> h(static_cast<size_t>(kN) * kK);
  Weights W;
  W.w.resize(h.size());
  for (size_t i = 0; i < h.size(); ++i) {
    h[i] = FloatToBf16(nd(rng));
    W.w[i] = Bf16ToFloat(h[i]);
  }
  q.layout = Layout::kBf16;
  q.N = kN;
  q.K = kK;
  q.bf16_w = DeviceBuffer<uint16_t>(h.size());
  q.bf16_w.CopyFromHost(h);
  return W;
}

void RunCase(hipStream_t s, r4dx::core::Arena& arena, std::mt19937_64& rng, const char* label,
             const QuantLinear& q, const Weights& W, int group) {
  const int variant = q.layout == Layout::kW4a16 ? q.w4a16_group : 0;
  for (int64_t M : {1, 2, 8, 16, 17, 64}) {
    const auto t = r4dx::model::PickTuning(q.layout, kN, kK, M, variant);
    Check(t.SK == 2, std::string(label) + ": picked SK " + std::to_string(t.SK) + " at K = 640, expected 2");
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<uint16_t> x(static_cast<size_t>(M * kK));
    for (auto& v : x) v = FloatToBf16(nd(rng));
    DeviceBuffer<uint16_t> x_d(x.size()), y_d(static_cast<size_t>(M * kN));
    x_d.CopyFromHost(x);
    y_d.CopyFromHost(std::vector<uint16_t>(y_d.size(), 0x7FC1));
    arena.Reset();
    ApplyLinear(s, arena, q, x_d.data(), y_d.data(), M);
    R4DX_HIP_CHECK(hipStreamSynchronize(s));
    const std::vector<uint16_t> y = y_d.CopyToHost();
    double max_abs = 0.0, worst = 0.0;
    size_t bad = 0;
    for (int64_t m = 0; m < M; ++m) {
      std::vector<double> ref(kN, 0.0);
      for (int n = 0; n < kN; ++n) {
        double acc = 0.0;
        const double* wr = &W.w[static_cast<size_t>(n) * kK];
        for (int k = 0; k < kK; ++k) acc += wr[k] * F16ToFloat(FloatToF16(Bf16ToFloat(x[m * kK + k])));
        ref[n] = acc;
      }
      double ss = 0.0;
      for (double v : ref) ss += v * v;
      const double floor = 4e-3 * std::sqrt(ss / kN);
      for (int n = 0; n < kN; ++n) {
        const double a = std::fabs(ref[n]);
        const double ulp = a > 1e-30 ? std::ldexp(1.0, static_cast<int>(std::floor(std::log2(a))) - 7) : 0.0;
        const double lim = 2.0 * ulp + floor;
        const double err = std::fabs(static_cast<double>(Bf16ToFloat(y[m * kN + n])) - ref[n]);
        max_abs = std::max(max_abs, err);
        worst = std::max(worst, err / lim);
        bad += !(err <= lim);
      }
    }
    Check(bad == 0, std::string(label) + " M=" + std::to_string(M) + ": " + std::to_string(bad) +
                        " element(s) out of tolerance");
    std::printf("  %-14s M=%2lld: tuning WV%d SK%d MB%d NPW%d NT%d, max|err| %.4g, worst %.3f of the tolerance\n",
                label, static_cast<long long>(M), t.WV, t.SK, t.MB, t.NPW, t.NT, max_abs, worst);
  }
  (void)group;
}

int RunTest() {
  int n_devices = 0;
  if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices < 1) {
    std::printf("[SKIP] no HIP device\n");
    return 77;
  }
  R4DX_HIP_CHECK(hipSetDevice(0));
  r4dx::core::Stream stream;
  r4dx::core::Arena arena;
  arena.Reserve(64ull << 20);
  std::mt19937_64 rng(20261011);
  std::printf("w4a16 / bf16 linear N=%d K=%d through ApplyLinear (FallbackTuning SK 2)\n", kN, kK);
  for (int group : {r4d_gemm_w4a16_nt_m64_group(), 32}) {
    if (group != r4d_gemm_w4a16_nt_m64_group() && r4d_gemm_w4a16_nt_m64_has_group(group) == 0) continue;
    QuantLinear q;
    const Weights W = MakeW4a16(rng, group, q);
    RunCase(stream.get(), arena, rng, group == 32 ? "w4a16 g32" : "w4a16 g64", q, W, group);
  }
  {
    QuantLinear q;
    const Weights W = MakeBf16(rng, q);
    RunCase(stream.get(), arena, rng, "bf16", q, W, 0);
  }
  std::printf("test_gpu_w4a16_k640: %d/%d checks passed\n", g_checks - g_failures, g_checks);
  return g_failures == 0 ? 0 : 1;
}

}  // namespace

int main() {
  try {
    return RunTest();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gpu_w4a16_k640: exception: %s\n", e.what());
    return 1;
  }
}
