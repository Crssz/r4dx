// tests/kernels/test_rmsnorm_plain.cpp -- r4dx_rmsnorm_plain_bf16 (docs/dflash2.md "Kernels").
//
// The point of this kernel is the WEIGHT FORM. r4dx's existing r4dx_rmsnorm_bf16 computes the
// target model's zero-centered `x * rstd * (1 + w)`; every DFlash2 norm site is ggml's plain
// `ggml_rms_norm` + `ggml_mul`, i.e. `x * rstd * w` with no offset (docs/dflash2.md's "RMSNorm
// convention" row). Feeding a DFlash2 norm weight to the target's kernel would add a spurious +1
// to every channel -- which is exactly what check [3] below rules out, by asserting the two
// kernels DISAGREE on the same inputs and that only the plain one matches the reference.
//
// Checks: (1) a CPU fp64 reference across both output dtypes and several shapes, (2) in-place
// operation (out == x), (3) the contrast against r4dx_rmsnorm_bf16, (4) fixture A's own
// `x_post_ffn_l4` -> `x_final_normed` under `output_norm_w`, which is a real DFlash2 norm weight
// applied by tools/reference/dflash2_ref.py with no C++ in the loop.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "npy_fixture.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;
using namespace r4dx_test;

namespace {

// y = x * rsqrt(mean(x^2) + eps) * w, fp64 accumulation, from the same bf16 bits the kernel reads.
std::vector<float> RmsNormPlainRef(const std::vector<uint16_t>& x, const std::vector<uint16_t>& w,
                                   int64_t rows, int64_t hidden, float eps) {
  std::vector<float> out(static_cast<size_t>(rows * hidden));
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* xr = x.data() + r * hidden;
    double ss = 0.0;
    for (int64_t i = 0; i < hidden; ++i) {
      const double v = Bf16ToFloat(xr[i]);
      ss += v * v;
    }
    const double rstd = 1.0 / std::sqrt(ss / static_cast<double>(hidden) + eps);
    for (int64_t i = 0; i < hidden; ++i) {
      out[static_cast<size_t>(r * hidden + i)] =
          static_cast<float>(Bf16ToFloat(xr[i]) * rstd * Bf16ToFloat(w[i]));
    }
  }
  return out;
}

std::vector<uint16_t> RandomBf16(size_t n, std::mt19937* rng, float lo, float hi) {
  std::uniform_real_distribution<float> d(lo, hi);
  std::vector<uint16_t> out(n);
  for (auto& v : out) v = FloatToBf16(d(*rng));
  return out;
}

}  // namespace

int main() {
  // Unbuffered: a GPU test that dies (a kernel fault, or an abort out of a failed HIP check)
  // takes the CRT's stdout buffer with it, and a crash report with zero output says nothing about
  // which check was running. Costs nothing at this output volume.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  bool ok = true;
  const float eps = 1.0e-6f;  // docs/dflash2.md: DFlash2's rms_eps
  std::mt19937 rng(41);

  // ---- 1. CPU reference, both output dtypes, the shapes the drafter actually uses ----
  // 5120 = draft hidden; 17408 = draft ffn; 128 = per-head q_norm/k_norm width; 8 rows = the
  // draft block; 40 rows = a fixture-sized injection batch.
  std::printf("[1] cpu fp64 reference (eps=%g)\n", eps);
  for (int64_t hidden : {int64_t(128), int64_t(5120), int64_t(17408)}) {
    for (int64_t rows : {int64_t(1), int64_t(8), int64_t(40)}) {
      const std::vector<uint16_t> x =
          RandomBf16(static_cast<size_t>(rows * hidden), &rng, -3.0f, 3.0f);
      const std::vector<uint16_t> w = RandomBf16(static_cast<size_t>(hidden), &rng, -1.5f, 1.5f);
      const std::vector<float> ref = RmsNormPlainRef(x, w, rows, hidden, eps);

      DeviceBuffer<uint16_t> x_d(x.size()), w_d(w.size()), out_bf_d(x.size());
      DeviceBuffer<float> out_f32_d(x.size());
      x_d.CopyFromHost(x);
      w_d.CopyFromHost(w);
      r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(x_d.data()),
                              reinterpret_cast<int64_t>(w_d.data()),
                              reinterpret_cast<int64_t>(out_bf_d.data()), rows, hidden, eps, 0, 0);
      r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(x_d.data()),
                              reinterpret_cast<int64_t>(w_d.data()),
                              reinterpret_cast<int64_t>(out_f32_d.data()), rows, hidden, eps, 1,
                              0);
      R4DX_HIP_CHECK(hipDeviceSynchronize());

      const ErrStats sb = CompareToRef(out_bf_d.CopyToHost(), ref);
      const ErrStats sf = CompareF32ToRef(out_f32_d.CopyToHost(), ref);
      // bf16 out: bounded by the final RTNE (2^-9 relative). fp32 out: only the fp32 vs fp64
      // sum-of-squares over up to 17408 terms, which is ~1e-6 relative on rstd.
      const bool pass = sb.norm_rel < 3e-3 && sf.norm_rel < 1e-5;
      std::printf("  rows=%-3lld hidden=%-6lld  bf16 norm_rel=%.3e  f32 norm_rel=%.3e  %s\n",
                  (long long)rows, (long long)hidden, sb.norm_rel, sf.norm_rel,
                  pass ? "PASS" : "FAIL");
      ok = ok && pass;
    }
  }

  // ---- 2. In-place (out == x, bf16) must equal the out-of-place result bit for bit ----
  {
    const int64_t rows = 8, hidden = 5120;
    const std::vector<uint16_t> x =
        RandomBf16(static_cast<size_t>(rows * hidden), &rng, -3.0f, 3.0f);
    const std::vector<uint16_t> w = RandomBf16(static_cast<size_t>(hidden), &rng, -1.5f, 1.5f);
    DeviceBuffer<uint16_t> x_d(x.size()), w_d(w.size()), out_d(x.size()), inplace_d(x.size());
    x_d.CopyFromHost(x);
    w_d.CopyFromHost(w);
    inplace_d.CopyFromHost(x);
    r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(x_d.data()),
                            reinterpret_cast<int64_t>(w_d.data()),
                            reinterpret_cast<int64_t>(out_d.data()), rows, hidden, eps, 0, 0);
    r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(inplace_d.data()),
                            reinterpret_cast<int64_t>(w_d.data()),
                            reinterpret_cast<int64_t>(inplace_d.data()), rows, hidden, eps, 0, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const bool same = out_d.CopyToHost() == inplace_d.CopyToHost();
    std::printf("[2] in-place == out-of-place, bit for bit: %s\n", same ? "PASS" : "FAIL");
    ok = ok && same;
  }

  // ---- 3. Contrast with the target model's (1+w) kernel: they must NOT agree ----
  // This is the whole reason the entry point exists, so it gets an explicit assertion rather than
  // only a comment. With `w` drawn around 0, `(1+w)` roughly doubles-and-shifts every channel, so
  // the two outputs differ by O(1) relative -- and only the plain one matches the reference.
  {
    const int64_t rows = 4, hidden = 5120;
    const std::vector<uint16_t> x =
        RandomBf16(static_cast<size_t>(rows * hidden), &rng, -3.0f, 3.0f);
    const std::vector<uint16_t> w = RandomBf16(static_cast<size_t>(hidden), &rng, -1.5f, 1.5f);
    const std::vector<float> ref = RmsNormPlainRef(x, w, rows, hidden, eps);

    DeviceBuffer<uint16_t> x_d(x.size()), w_d(w.size()), plain_d(x.size()), centered_d(x.size());
    x_d.CopyFromHost(x);
    w_d.CopyFromHost(w);
    r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(x_d.data()),
                            reinterpret_cast<int64_t>(w_d.data()),
                            reinterpret_cast<int64_t>(plain_d.data()), rows, hidden, eps, 0, 0);
    r4dx_rmsnorm_bf16(reinterpret_cast<int64_t>(x_d.data()),
                      reinterpret_cast<int64_t>(w_d.data()),
                      reinterpret_cast<int64_t>(centered_d.data()), rows, hidden, eps, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());

    const ErrStats sp = CompareToRef(plain_d.CopyToHost(), ref);
    const ErrStats sc = CompareToRef(centered_d.CopyToHost(), ref);
    const bool pass = sp.norm_rel < 3e-3 && sc.norm_rel > 0.5;
    std::printf("[3] plain norm_rel=%.3e (must be small) | (1+w) norm_rel=%.3e (must be large)"
                "  %s\n",
                sp.norm_rel, sc.norm_rel, pass ? "PASS" : "FAIL");
    ok = ok && pass;
  }

  // ---- 4. Fixture A: x_post_ffn_l4 -> x_final_normed under the real output_norm weight ----
  if (!FixtureAvailable()) {
    std::fprintf(stderr,
                 "[SKIP] fixture A not found at %s -- regenerate with\n"
                 "       <reference venv>/python.exe tools/reference/dflash2_ref.py "
                 "--gen-fixtures all --seed 0\n",
                 FixtureDir().c_str());
    return ok ? kSkipReturnCode : 1;
  }
  {
    // manifest.json's own `x_final_normed_source`/`x_final_normed_weight`/`x_final_normed_rms_eps`
    // name this triple; the values below match those fields for the seed-0 fixture.
    const std::string d = FixtureDir();
    const std::vector<uint16_t> x = ToBf16(LoadNpyF32(d + "/x_post_ffn_l4.npy", {8, 5120}));
    const std::vector<uint16_t> w = ToBf16(LoadNpyF32(d + "/output_norm_w.npy", {5120}));
    const std::vector<float> ref = LoadNpyF32(d + "/x_final_normed.npy", {8, 5120});

    DeviceBuffer<uint16_t> x_d(x.size()), w_d(w.size()), out_d(x.size());
    x_d.CopyFromHost(x);
    w_d.CopyFromHost(w);
    r4dx_rmsnorm_plain_bf16(reinterpret_cast<int64_t>(x_d.data()),
                            reinterpret_cast<int64_t>(w_d.data()),
                            reinterpret_cast<int64_t>(out_d.data()), 8, 5120, eps, 0, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());

    const ErrStats s = CompareToRef(out_d.CopyToHost(), ref);
    std::printf("[4] fixture A x_post_ffn_l4 -> x_final_normed: max_abs=%.3e max_rel=%.3e "
                "norm_rel=%.3e\n",
                s.max_abs, s.max_rel, s.norm_rel);
    // x and w are both bf16-rounded from the fp32 fixture, so the product carries ~2 * 2^-9; the
    // (1+w) form would land at O(1) here, not at 1e-2.
    const bool pass = s.norm_rel < 1e-2;
    std::printf("[4] %s\n", pass ? "PASS" : "FAIL");
    ok = ok && pass;
  }

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
