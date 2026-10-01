// tests/kernels/test_gemma_resid_f32_cpu.cpp -- pure CPU (no HIP call). Pins the fp32-residual contract of
// GemmaModel (R4DX_GEMMA_RESID=fp32) that the *_f32 / *_f32res kernels implement, against the fp64
// references in gemma_ref.hpp / rotation_ref.hpp:
//   * the fp32 post-norm + residual reference agrees with the bf16-rounded chain to within bf16 rounding
//     (it is the same math, minus the rounding points), for scalar 1 and a layer_scalar;
//   * a large (|h| ~ 300) residual absorbs a sub-ulp update in a bf16 stream but not in fp32 -- the failure
//     mode that motivated the fp32 stream;
//   * scaled embedding: bf16(row) * 62.0 is exact in fp32 (so the fp32 gather has no rounding to match);
//   * rotated fp32 stream: rotate(h) + rotate(normed), un-rotated, equals h + normed to fp32 precision
//     (the Q round trip is exact up to ~1e-6, while bf16 storage of the rotated stream is ~4e-3);
//   * the fp32 pre-norm reference is the bf16 one evaluated on the unrounded input.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "gemma_ref.hpp"
#include "rotation_ref.hpp"

namespace {

bool g_ok = true;
void Check(bool cond, const char* what) {
  std::printf("%-86s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}

using gemma_ref::Bf16Round;

void TestPostnormAgreesWithBf16Chain(std::mt19937& rng) {
  const int n = 3840;
  const double eps = 1e-6;
  std::uniform_real_distribution<float> dres(-300.0f, 300.0f), dy(-30.0f, 30.0f), dw(-2.0f, 2.0f);
  for (double scalar : {1.0, static_cast<double>(Bf16Round(0.37f))}) {
    std::vector<float> resf(n), yf(n), wpf(n), wnf(n);
    std::vector<double> res(n), y(n), wp(n), wn(n);
    for (int i = 0; i < n; ++i) {
      resf[i] = Bf16Round(dres(rng));
      yf[i] = Bf16Round(dy(rng));
      wpf[i] = Bf16Round(dw(rng));
      wnf[i] = Bf16Round(dw(rng));
      res[i] = resf[i];
      y[i] = yf[i];
      wp[i] = wpf[i];
      wn[i] = wnf[i];
    }
    const auto f32 = gemma_ref::PostnormResidualF32Ref(res, y, wp, &wn, eps, scalar);
    // The bf16 chain (what the HF module / the bf16 kernel does).
    const std::vector<float> normed = gemma_ref::RmsNormRow(yf, &wpf, eps);
    double worst = 0.0;
    for (int i = 0; i < n; ++i) {
      float sum = Bf16Round(resf[i] + normed[i]);
      if (scalar != 1.0) sum = Bf16Round(sum * static_cast<float>(scalar));
      // three bf16 roundings of values of magnitude |res| + |normed|: <= ~3 * 2^-8 relative
      const double bound = (std::abs(resf[i]) + std::abs(normed[i])) * std::abs(scalar) * 3.0 / 256.0 + 1e-6;
      worst = std::max(worst, std::abs(f32.out_res[i] - sum) / bound);
    }
    char msg[160];
    std::snprintf(msg, sizeof msg, "fp32 post-norm residual within bf16-chain rounding (scalar %.3g, worst %.2f of bound)", scalar,
                  worst);
    Check(worst <= 1.0, msg);
    // out_normed of the fp32 reference equals the bf16 pre-norm applied to the unrounded out_res.
    std::vector<double> nref = gemma_ref::RmsNormF32InRef(f32.out_res, wn, eps);
    double maxd = 0.0;
    for (int i = 0; i < n; ++i) maxd = std::max(maxd, std::abs(nref[i] - f32.out_normed[i]));
    Check(maxd < 1e-12, "next-norm from the fp32 out_res == fp32 pre-norm reference");
  }
}

void TestSubUlpUpdateSurvives() {
  // |h| = 300: the bf16 ulp there is 2.0, so adding 0.4 a thousand times does nothing in bf16, +400 in fp32.
  float h_bf16 = Bf16Round(300.0f);
  float h_f32 = 300.0f;
  for (int i = 0; i < 1000; ++i) {
    h_bf16 = Bf16Round(h_bf16 + 0.4f);
    h_f32 = h_f32 + 0.4f;
  }
  Check(h_bf16 == 300.0f, "bf16 stream: 1000 x (+0.4) on |h| = 300 is absorbed entirely (the motivation)");
  Check(std::abs(h_f32 - 700.0f) < 0.1f, "fp32 stream: the same updates accumulate");
}

void TestEmbedScaleExact() {
  std::mt19937 rng(7);
  std::uniform_real_distribution<float> d(-0.5f, 0.5f);
  const float scale = Bf16Round(static_cast<float>(std::sqrt(3840.0)));
  bool exact = scale == 62.0f;
  Check(exact, "bf16(sqrt(3840)) == 62.0");
  bool all_exact = true;
  for (int i = 0; i < 100000; ++i) {
    const float x = Bf16Round(d(rng));
    all_exact = all_exact && (static_cast<double>(x) * 62.0 == static_cast<double>(x * scale));
  }
  Check(all_exact, "bf16 table value * 62.0 is exact in fp32 (the fp32 gather rounds nothing)");
}

void TestRotatedStreamRoundTrip() {
  using namespace rotation_ref;
  std::mt19937_64 rng(11);
  const int64_t hidden = 3840, block = ChooseBlock(hidden);  // 256
  const int64_t nblk = hidden / block;
  const std::vector<float> d = RandomSigns(rng, hidden);
  const std::vector<float> R = RandomOrthogonal(rng, static_cast<int>(nblk));
  std::normal_distribution<double> nd(0.0, 1.0);
  std::vector<double> h(hidden), y(hidden), w(hidden);
  for (int64_t i = 0; i < hidden; ++i) {
    h[i] = nd(rng) * 100.0;  // residual-scale values
    y[i] = nd(rng) * 10.0;
    w[i] = nd(rng);
  }
  // Rotated stream in FP32 storage: hq = fp32(h Q); then one fused post-norm/rotate/add step, fp32 stored.
  std::vector<double> hq = ApplyQGeneral(h, block, d.data(), R.data(), false);
  for (double& v : hq) v = static_cast<double>(static_cast<float>(v));
  std::vector<double> next = PostNormRotateAddRef(hq, y, w, block, d.data(), R.data(), 1e-6, 1.0);
  for (double& v : next) v = static_cast<double>(static_cast<float>(v));
  const std::vector<double> back = ApplyQGeneral(next, block, d.data(), R.data(), true);
  // Expected unrotated: h + normed(y)
  double ss = 0.0;
  for (double v : y) ss += v * v;
  const double rstd = 1.0 / std::sqrt(ss / hidden + 1e-6);
  double worst = 0.0, scale = 0.0;
  for (int64_t i = 0; i < hidden; ++i) {
    worst = std::max(worst, std::abs(back[i] - (h[i] + y[i] * rstd * w[i])));
    scale = std::max(scale, std::abs(h[i]));
  }
  char msg[160];
  std::snprintf(msg, sizeof msg, "fp32 rotated stream round trip: max abs err %.2e on |h|max %.0f (rel %.1e)", worst, scale,
                worst / scale);
  Check(worst / scale < 5e-6, msg);

  // The same with the rotated stream rounded to bf16 between sublayers is far coarser.
  std::vector<double> hq_b = ApplyQGeneral(h, block, d.data(), R.data(), false);
  for (double& v : hq_b) v = Bf16Round(static_cast<float>(v));
  std::vector<double> next_b = PostNormRotateAddRef(hq_b, y, w, block, d.data(), R.data(), 1e-6, 1.0);
  for (double& v : next_b) v = Bf16Round(static_cast<float>(v));
  const std::vector<double> back_b = ApplyQGeneral(next_b, block, d.data(), R.data(), true);
  double worst_b = 0.0;
  for (int64_t i = 0; i < hidden; ++i) worst_b = std::max(worst_b, std::abs(back_b[i] - (h[i] + y[i] * rstd * w[i])));
  Check(worst_b > 100.0 * worst, "bf16 rotated stream is >= 100x noisier than fp32 (sanity of the comparison)");
}

}  // namespace

int main() {
  std::mt19937 rng(23);
  TestPostnormAgreesWithBf16Chain(rng);
  TestSubUlpUpdateSurvives();
  TestEmbedScaleExact();
  TestRotatedStreamRoundTrip();
  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
