// qwen4_exp GDN gated RMSNorm with the sigmoid output gate (q4x_gdn_sigmoid_gated_rmsnorm_h128_bf16) against HF
// Qwen4ExpTextRMSNormGated(128, activation="sigmoid") from tools/reference/qwen4exp/small_fixtures.py: 7 tokens x 48
// value heads = 336 rows of 128. Also reported: libr4d's r4d_gdn_gated_rmsnorm_h128_bf16 with act = 1 (sigmoid) against
// the same reference, to show what the HF rounding points buy. Exit 77 when the fixtures are missing.
#include <cstdio>
#include <exception>
#include <vector>

#include "r4dx/core/r4d.hpp"
#include "r4dx/core/stream.hpp"
#include "r4dx/model/qwen4exp/gdn_sigmoid_norm.h"
#include "small_fixture.h"

using namespace q4test;
using namespace r4dx;

namespace {

bool g_ok = true;

double Rms(const uint16_t* p, int64_t n) {
  double s = 0;
  for (int64_t i = 0; i < n; ++i) s += double(F(p[i])) * F(p[i]);
  return std::sqrt(s / double(n));
}

}  // namespace

int main() {
  try {
    Fixtures fx;
    if (!fx.Open()) {
      std::fprintf(stderr, "[SKIP] fixtures not found in %s (run tools/reference/qwen4exp/small_fixtures.py)\n",
                   Fixtures::Dir().c_str());
      return kSkip;
    }
    const int64_t rows = static_cast<int64_t>(fx.Param("gdn.rows"));
    const float eps = static_cast<float>(fx.Param("gdn.eps"));
    const Tensor& wt = fx.Get("gdn.weight");
    std::vector<float> w32(128);
    for (int i = 0; i < 128; ++i) w32[i] = F(wt.bf16()[i]);  // the bf16 weight widened, as the GDN layer holds it
    auto d_w = UploadF32(w32.data(), 128);
    auto d_x = Upload(fx.Get("gdn.x"));
    auto d_z = Upload(fx.Get("gdn.z"));
    core::Stream stream;
    const Tensor& ref = fx.Get("gdn.y_sigmoid");
    const Tensor& ref_silu = fx.Get("gdn.y_silu");
    std::printf("[gdn] rows=%lld width=128\n", static_cast<long long>(rows));

    core::DeviceBuffer<uint16_t> out(static_cast<size_t>(rows * 128));
    if (q4x_gdn_sigmoid_gated_rmsnorm_h128_bf16(d_x.data(), d_z.data(), d_w.data(), out.data(), rows, 128, 128, 128,
                                                128, eps, stream.get()) != 0) {
      throw std::runtime_error("q4x_gdn_sigmoid_gated_rmsnorm_h128_bf16 failed");
    }
    const std::vector<uint16_t> got = Download(out);
    const double rms = Rms(ref.bf16(), ref.count());
    // The two bf16 roundings before the gate are reproduced, so only the fp32 reduction order / sigmoid ulp can differ:
    // a handful of one-ulp differences at most.
    if (!Report("sigmoid norm (new) vs HF", CompareBf16(got.data(), ref.bf16(), ref.count(), 1e-3 * rms, 0.0079))) g_ok = false;

    // strided rows: x padded to 256, z to 192, o to 160, same values
    {
      std::vector<uint16_t> xp(static_cast<size_t>(rows * 256), 0x7fc0), zp(static_cast<size_t>(rows * 192), 0x7fc0);
      for (int64_t r = 0; r < rows; ++r) {
        for (int d = 0; d < 128; ++d) {
          xp[r * 256 + d] = fx.Get("gdn.x").bf16()[r * 128 + d];
          zp[r * 192 + d] = fx.Get("gdn.z").bf16()[r * 128 + d];
        }
      }
      auto d_xp = UploadBf16(xp.data(), xp.size());
      auto d_zp = UploadBf16(zp.data(), zp.size());
      core::DeviceBuffer<uint16_t> op(static_cast<size_t>(rows * 160));
      op.ZeroAsync(stream);
      if (q4x_gdn_sigmoid_gated_rmsnorm_h128_bf16(d_xp.data(), d_zp.data(), d_w.data(), op.data(), rows, 256, 192, 160,
                                                  128, eps, stream.get()) != 0) {
        throw std::runtime_error("strided launch failed");
      }
      const std::vector<uint16_t> o = Download(op);
      bool same = true, pad_clean = true;
      for (int64_t r = 0; r < rows; ++r) {
        for (int d = 0; d < 128; ++d) same &= o[r * 160 + d] == got[r * 128 + d];
        for (int d = 128; d < 160; ++d) pad_clean &= o[r * 160 + d] == 0;
      }
      std::printf("  %-34s %s (padding untouched: %s)\n", "strided rows == contiguous (exact)", same ? "OK" : "FAIL",
                  pad_clean ? "yes" : "NO");
      if (!same || !pad_clean) g_ok = false;
    }

    // a width other than 128 is refused
    {
      const int rc = q4x_gdn_sigmoid_gated_rmsnorm_h128_bf16(d_x.data(), d_z.data(), d_w.data(), out.data(), rows, 128,
                                                              128, 128, 64, eps, stream.get());
      std::printf("  %-34s %s\n", "width != 128 rejected", rc != 0 ? "OK" : "FAIL");
      if (rc == 0) g_ok = false;
    }

    // reference points: libr4d's kernel at act = 1 (sigmoid) and act = 0 (silu, the 27B) vs the same HF outputs
    {
      core::DeviceBuffer<uint16_t> lo(static_cast<size_t>(rows * 128));
      core::r4d::GdnGatedRmsNorm(d_x.data(), d_z.data(), d_w.data(), lo.data(), rows, 128, 128, 128, 128, eps, 1,
                                 stream.get());
      const std::vector<uint16_t> a = Download(lo);
      const ErrorStats sa = CompareBf16(a.data(), ref.bf16(), ref.count(), 1e-3 * rms, 0.0079);
      std::printf("  [info] libr4d act=1 vs HF sigmoid: max_abs=%.4g mean_abs=%.4g exact=%.2f%% (> 1 ulp: %lld)\n",
                  sa.max_abs, sa.mean_abs, 100.0 * double(sa.exact) / double(sa.n), static_cast<long long>(sa.bad));
      core::r4d::GdnGatedRmsNorm(d_x.data(), d_z.data(), d_w.data(), lo.data(), rows, 128, 128, 128, 128, eps, 0,
                                 stream.get());
      const std::vector<uint16_t> b = Download(lo);
      const ErrorStats sb = CompareBf16(b.data(), ref_silu.bf16(), ref_silu.count(), 1e-3 * Rms(ref_silu.bf16(), ref_silu.count()), 0.0079);
      std::printf("  [info] libr4d act=0 vs HF silu:    max_abs=%.4g mean_abs=%.4g exact=%.2f%% (> 1 ulp: %lld)\n",
                  sb.max_abs, sb.mean_abs, 100.0 * double(sb.exact) / double(sb.n), static_cast<long long>(sb.bad));
    }

    std::printf(g_ok ? "test_gpu_gdn_sigmoid: PASS\n" : "test_gpu_gdn_sigmoid: FAIL\n");
    return g_ok ? 0 : 1;
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gpu_gdn_sigmoid: exception: %s\n", e.what());
    return 1;
  }
}
