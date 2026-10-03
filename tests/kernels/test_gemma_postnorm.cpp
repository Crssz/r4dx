// tests/kernels/test_gemma_postnorm.cpp -- r4dx_gemma_postnorm_residual_rmsnorm_bf16 (docs/gemma4-plan.md
// 3.5): out_res = bf16(bf16(res + bf16(rms(y) * w_post)) * scalar), out_normed = bf16(rms(out_res) *
// w_next), each rounded where the HF module chain rounds. Checked against a CPU reference that spells
// out those roundings, with scalar == 1 (the attention half: the identity) and a layer_scalar value
// (the MLP half), out of place and in place (res == out_res), with and without the fused next norm,
// plus the precondition throws. GPU test (built with /EHc-).
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
  std::printf("%-84s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}
int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

std::vector<uint16_t> RandBf16(std::mt19937& rng, size_t n, float mag) {
  std::uniform_real_distribution<float> dist(-mag, mag);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = FloatToBf16(dist(rng));
  return v;
}

void Run(std::mt19937& rng, int64_t rows, int64_t hidden, float scalar, bool in_place, bool with_next) {
  const float eps = 1e-6f;
  const auto res_h = RandBf16(rng, rows * hidden, 4.0f);
  const auto y_h = RandBf16(rng, rows * hidden, 30.0f);  // sublayer outputs are large before the post-norm
  const auto wp_h = RandBf16(rng, hidden, 2.0f);
  const auto wn_h = RandBf16(rng, hidden, 2.0f);

  DeviceBuffer<uint16_t> res_d(res_h.size()), y_d(y_h.size()), wp_d(hidden), wn_d(hidden),
      out_d(res_h.size()), nrm_d(res_h.size());
  res_d.CopyFromHost(res_h);
  y_d.CopyFromHost(y_h);
  wp_d.CopyFromHost(wp_h);
  wn_d.CopyFromHost(wn_h);
  nrm_d.Zero();
  const uint16_t* res_p = res_d.data();
  uint16_t* out_p = in_place ? res_d.data() : out_d.data();
  r4dx_gemma_postnorm_residual_rmsnorm_bf16(P(res_p), P(y_d.data()), P(wp_d.data()),
                                             with_next ? P(wn_d.data()) : 0, eps, P(out_p),
                                             with_next ? P(nrm_d.data()) : 0, scalar, rows, hidden, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<uint16_t> got = (in_place ? res_d : out_d).CopyToHost();
  const std::vector<uint16_t> nrm = nrm_d.CopyToHost();

  int64_t bad = 0, bad_n = 0;
  for (int64_t r = 0; r < rows; ++r) {
    std::vector<float> yr(hidden), wpf(hidden), wnf(hidden), got_row(hidden);
    for (int64_t i = 0; i < hidden; ++i) {
      yr[i] = Bf16ToFloat(y_h[r * hidden + i]);
      wpf[i] = Bf16ToFloat(wp_h[i]);
      wnf[i] = Bf16ToFloat(wn_h[i]);
      got_row[i] = Bf16ToFloat(got[r * hidden + i]);
    }
    const std::vector<float> normed = gemma_ref::RmsNormRow(yr, &wpf, eps);  // bf16-rounded
    for (int64_t i = 0; i < hidden; ++i) {
      const float sum = Bf16Round(Bf16ToFloat(res_h[r * hidden + i]) + normed[i]);
      const float ref = scalar != 1.0f ? Bf16Round(sum * scalar) : sum;
      // Two bf16 steps of the larger operand: a rounding tie in `normed` flips one step upstream.
      const double tol = (std::abs(Bf16ToFloat(res_h[r * hidden + i])) + std::abs(normed[i])) / 100.0 + 1e-5;
      if (std::abs(got_row[i] - ref) > tol * (scalar != 1.0f ? std::abs(scalar) : 1.0f)) ++bad;
    }
    if (with_next) {
      const std::vector<float> nref = gemma_ref::RmsNormRow(got_row, &wnf, eps);  // from the kernel's own out_res
      for (int64_t i = 0; i < hidden; ++i) {
        const double g = Bf16ToFloat(nrm[r * hidden + i]);
        if (std::abs(g - nref[i]) > std::abs(nref[i]) / 100.0 + 1e-6) ++bad_n;
      }
    }
  }
  char msg[200];
  std::snprintf(msg, sizeof msg, "rows %lld hidden %lld scalar %.3g %s%s: out_res matches (%lld bad)",
                (long long)rows, (long long)hidden, scalar, in_place ? "in-place" : "out-of-place",
                with_next ? " +next" : "", (long long)bad);
  Check(bad == 0, msg);
  if (with_next) {
    std::snprintf(msg, sizeof msg, "    out_normed matches (%lld bad)", (long long)bad_n);
    Check(bad_n == 0, msg);
  } else {
    bool untouched = true;
    for (uint16_t v : nrm) untouched = untouched && v == 0;
    Check(untouched, "    out_normed untouched when not requested");
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(29);
  const float ls = Bf16Round(0.37f);  // layer_scalar is a bf16 buffer
  for (bool in_place : {false, true}) {
    for (bool with_next : {false, true}) {
      Run(rng, 5, 3840, 1.0f, in_place, with_next);  // attention half: the identity scalar
      Run(rng, 5, 3840, ls, in_place, with_next);    // MLP half: layer_scalar
    }
  }
  Run(rng, 1, 3840, ls, false, true);   // decode row
  Run(rng, 3, 1000 + 3, 1.0f, false, true);  // odd width

  // Preconditions.
  {
    DeviceBuffer<uint16_t> a(64), b(64), w(64), o(64);
    const auto throws = [&](int64_t res, int64_t y, int64_t wp, int64_t wn, int64_t out, int64_t nrm) {
      try {
        r4dx_gemma_postnorm_residual_rmsnorm_bf16(res, y, wp, wn, 1e-6f, out, nrm, 1.0f, 1, 64, 0);
      } catch (const std::runtime_error&) {
        return true;
      }
      return false;
    };
    Check(throws(0, P(b.data()), P(w.data()), 0, P(o.data()), 0), "null res throws");
    Check(throws(P(a.data()), P(b.data()), P(w.data()), P(w.data()), P(o.data()), 0),
          "w_next without out_normed throws");
    Check(throws(P(a.data()), P(b.data()), P(w.data()), 0, P(o.data()), P(o.data())),
          "out_normed without w_next throws");
  }

  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
