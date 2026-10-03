// tests/kernels/test_gemma_resid_f32.cpp -- the fp32-residual Gemma kernels (gemma_kernels.h, "fp32-residual
// variants"; GemmaModel R4DX_GEMMA_RESID=fp32) against the fp64 references in gemma_ref.hpp:
//   r4dx_embedding_gather_scaled_f32          exact (bf16 * 62.0 is exact in fp32), bounds guard
//   r4dx_rmsnorm_plain_f32in_bf16             pre-norms / final norm of the fp32 stream
//   r4dx_gemma_postnorm_residual_rmsnorm_f32res   scalar 1 / layer_scalar, in place / out of place, +-next norm
//   r4dx_f32_to_bf16                          RNE, the drafter feature narrowing
//   precondition throws.
// GPU test (HIP device 1 via HIP_VISIBLE_DEVICES=1 in CMakeLists.txt); WRITTEN, NOT YET RUN. Built with /EHc-.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "gemma_ref.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/gemma_kernels.h"

using namespace r4dx::core;
using gemma_ref::Bf16Round;

namespace {

bool g_ok = true;
void Check(bool cond, const char* what) {
  std::printf("%-86s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}
int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

std::vector<uint16_t> RandBf16(std::mt19937& rng, size_t n, float mag) {
  std::uniform_real_distribution<float> dist(-mag, mag);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = FloatToBf16(dist(rng));
  return v;
}
std::vector<float> RandF32(std::mt19937& rng, size_t n, float mag) {
  std::uniform_real_distribution<float> dist(-mag, mag);
  std::vector<float> v(n);
  for (auto& x : v) x = dist(rng);
  return v;
}
std::vector<double> Row(const std::vector<float>& v, int64_t r, int64_t n) {
  return std::vector<double>(v.begin() + r * n, v.begin() + (r + 1) * n);
}
std::vector<double> RowB(const std::vector<uint16_t>& v, int64_t r, int64_t n) {
  std::vector<double> o(n);
  for (int64_t i = 0; i < n; ++i) o[i] = Bf16ToFloat(v[r * n + i]);
  return o;
}

void TestEmbed(std::mt19937& rng) {
  const int64_t vocab = 97, hidden = 3840, n = 5;
  const auto table = RandBf16(rng, vocab * hidden, 0.5f);
  std::vector<int32_t> ids = {3, 96, 0, 50, -1};  // -1: out of range reads row 0
  DeviceBuffer<uint16_t> t_d(table.size());
  DeviceBuffer<int32_t> id_d(n);
  DeviceBuffer<float> out_d(n * hidden);
  t_d.CopyFromHost(table);
  id_d.CopyFromHost(ids);
  r4dx_embedding_gather_scaled_f32(P(t_d.data()), P(id_d.data()), P(out_d.data()), n, hidden, vocab, 62.0f, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<float> got = out_d.CopyToHost();
  int64_t bad = 0;
  for (int64_t r = 0; r < n; ++r) {
    const int64_t id = (ids[r] < 0 || ids[r] >= vocab) ? 0 : ids[r];
    for (int64_t i = 0; i < hidden; ++i) {
      const double ref = static_cast<double>(Bf16ToFloat(table[id * hidden + i])) * 62.0;
      if (static_cast<double>(got[r * hidden + i]) != ref) ++bad;
    }
  }
  char msg[120];
  std::snprintf(msg, sizeof msg, "embedding_gather_scaled_f32 exact (%lld bad)", (long long)bad);
  Check(bad == 0, msg);
}

void TestNorm(std::mt19937& rng, int64_t rows, int64_t hidden) {
  const float eps = 1e-6f;
  const auto x = RandF32(rng, rows * hidden, 250.0f);
  const auto w = RandBf16(rng, hidden, 2.0f);
  DeviceBuffer<float> x_d(x.size());
  DeviceBuffer<uint16_t> w_d(hidden), o_d(x.size());
  x_d.CopyFromHost(x);
  w_d.CopyFromHost(w);
  r4dx_rmsnorm_plain_f32in_bf16(P(x_d.data()), P(w_d.data()), P(o_d.data()), rows, hidden, eps, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const auto got = o_d.CopyToHost();
  std::vector<double> wd(hidden);
  for (int64_t i = 0; i < hidden; ++i) wd[i] = Bf16ToFloat(w[i]);
  int64_t bad = 0;
  for (int64_t r = 0; r < rows; ++r) {
    const auto ref = gemma_ref::RmsNormF32InRef(Row(x, r, hidden), wd, eps);
    for (int64_t i = 0; i < hidden; ++i) {
      const double g = Bf16ToFloat(got[r * hidden + i]);
      if (std::abs(g - ref[i]) > std::abs(ref[i]) / 128.0 + 1e-6) ++bad;  // one bf16 rounding + fp32 noise
    }
  }
  char msg[120];
  std::snprintf(msg, sizeof msg, "rmsnorm_plain_f32in_bf16 rows %lld hidden %lld (%lld bad)", (long long)rows,
                (long long)hidden, (long long)bad);
  Check(bad == 0, msg);
}

void TestPostnorm(std::mt19937& rng, int64_t rows, int64_t hidden, float scalar, bool in_place, bool with_next) {
  const float eps = 1e-6f;
  const auto res_h = RandF32(rng, rows * hidden, 300.0f);  // a genuine fp32 residual (not bf16-representable)
  const auto y_h = RandBf16(rng, rows * hidden, 30.0f);
  const auto wp_h = RandBf16(rng, hidden, 2.0f);
  const auto wn_h = RandBf16(rng, hidden, 2.0f);
  DeviceBuffer<float> res_d(res_h.size()), out_d(res_h.size());
  DeviceBuffer<uint16_t> y_d(y_h.size()), wp_d(hidden), wn_d(hidden), nrm_d(res_h.size());
  res_d.CopyFromHost(res_h);
  y_d.CopyFromHost(y_h);
  wp_d.CopyFromHost(wp_h);
  wn_d.CopyFromHost(wn_h);
  nrm_d.Zero();
  float* out_p = in_place ? res_d.data() : out_d.data();
  r4dx_gemma_postnorm_residual_rmsnorm_f32res(P(res_d.data()), P(y_d.data()), P(wp_d.data()),
                                               with_next ? P(wn_d.data()) : 0, eps, P(out_p),
                                               with_next ? P(nrm_d.data()) : 0, scalar, rows, hidden, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<float> got = (in_place ? res_d : out_d).CopyToHost();
  const auto nrm = nrm_d.CopyToHost();
  std::vector<double> wp(hidden), wn(hidden);
  for (int64_t i = 0; i < hidden; ++i) {
    wp[i] = Bf16ToFloat(wp_h[i]);
    wn[i] = Bf16ToFloat(wn_h[i]);
  }
  int64_t bad = 0, bad_n = 0;
  for (int64_t r = 0; r < rows; ++r) {
    const auto ref = gemma_ref::PostnormResidualF32Ref(Row(res_h, r, hidden), RowB(y_h, r, hidden), wp,
                                                       with_next ? &wn : nullptr, eps, scalar);
    for (int64_t i = 0; i < hidden; ++i) {
      // fp32 arithmetic against fp64: a few fp32 ulps of the operands, no bf16 rounding anywhere.
      const double tol = (std::abs(res_h[r * hidden + i]) + 40.0) * std::abs(scalar) * 4e-7 + 1e-6;
      if (std::abs(got[r * hidden + i] - ref.out_res[i]) > tol) ++bad;
      if (with_next) {
        const double g = Bf16ToFloat(nrm[r * hidden + i]);
        if (std::abs(g - ref.out_normed[i]) > std::abs(ref.out_normed[i]) / 128.0 + 1e-5) ++bad_n;
      }
    }
  }
  char msg[200];
  std::snprintf(msg, sizeof msg, "postnorm_f32res rows %lld hidden %lld scalar %.3g %s%s: out_res (%lld bad)",
                (long long)rows, (long long)hidden, scalar, in_place ? "in-place" : "out-of-place",
                with_next ? " +next" : "", (long long)bad);
  Check(bad == 0, msg);
  if (with_next) {
    std::snprintf(msg, sizeof msg, "    out_normed (%lld bad)", (long long)bad_n);
    Check(bad_n == 0, msg);
  } else {
    bool untouched = true;
    for (uint16_t v : nrm) untouched = untouched && v == 0;
    Check(untouched, "    out_normed untouched when not requested");
  }
}

void TestNarrow(std::mt19937& rng) {
  const int64_t n = 3840 * 5 + 7;
  const auto x = RandF32(rng, n, 400.0f);
  DeviceBuffer<float> x_d(n);
  DeviceBuffer<uint16_t> o_d(n);
  x_d.CopyFromHost(x);
  r4dx_f32_to_bf16(P(x_d.data()), P(o_d.data()), n, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const auto got = o_d.CopyToHost();
  int64_t bad = 0;
  for (int64_t i = 0; i < n; ++i) bad += Bf16ToFloat(got[i]) != Bf16Round(x[i]);
  char msg[100];
  std::snprintf(msg, sizeof msg, "f32_to_bf16 == RNE (%lld bad)", (long long)bad);
  Check(bad == 0, msg);
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(41);
  TestEmbed(rng);
  TestNorm(rng, 1, 3840);
  TestNorm(rng, 7, 3840);
  TestNorm(rng, 3, 1003);
  const float ls = Bf16Round(0.37f);
  for (bool in_place : {false, true}) {
    for (bool with_next : {false, true}) {
      TestPostnorm(rng, 5, 3840, 1.0f, in_place, with_next);
      TestPostnorm(rng, 5, 3840, ls, in_place, with_next);
    }
  }
  TestPostnorm(rng, 1, 3840, ls, false, true);
  TestPostnorm(rng, 3, 1003, 1.0f, false, true);
  TestNarrow(rng);
  {
    DeviceBuffer<float> a(64), o(64);
    DeviceBuffer<uint16_t> b(64), w(64), on(64);
    const auto throws = [&](int64_t res, int64_t y, int64_t wp, int64_t wn, int64_t out, int64_t nrm) {
      try {
        r4dx_gemma_postnorm_residual_rmsnorm_f32res(res, y, wp, wn, 1e-6f, out, nrm, 1.0f, 1, 64, 0);
      } catch (const std::runtime_error&) {
        return true;
      }
      return false;
    };
    Check(throws(0, P(b.data()), P(w.data()), 0, P(o.data()), 0), "null res throws");
    Check(throws(P(a.data()), P(b.data()), P(w.data()), P(w.data()), P(o.data()), 0), "w_next without out_normed throws");
    Check(throws(P(a.data()), P(b.data()), P(w.data()), 0, P(o.data()), P(on.data())), "out_normed without w_next throws");
  }
  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
