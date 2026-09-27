// tests/kernels/test_trellis_input.cpp -- the trellis linears' input transform, bit for bit
// (docs/trellis-kernel.md 4.8 and 6, milestone M2):
//
//   r4d_fwht128_wave (libr4d's r4d_fwht128.h, through r4dx_fwht128_f32) equals r4dx's FwhtLds and
//   the CPU FwhtLdsF32 bitwise, on 4096 random fp32 blocks over a wide exponent range;
//
//   r4dx_trellis_input_bf16 equals trellis_ref::TransformInput -- f16_rn(FWHT128_fp32(fp32(x) *
//   suh) * float(2^s / sqrt(128))) -- bit for bit: nout 1-3, prescale 0 / 4 / -3, every output
//   placed part_stride elements after the previous one in one buffer (the layout a multi-part
//   linear's A takes) with every element outside the outputs left untouched, M = 1..64 (all of
//   them at K = 5120) and K in {3072, 5120, 6144, 8704, 17408}; activations whose small rows land
//   in f16's subnormal range, so that rounding path is covered;
//
//   against the Python goldens (tools/reference/trellis_golden.py -> golden/trellis/): the random
//   case's A at prescale 0 and 4 and the three real oracle blocks' A, bit for bit;
//
//   the host's precondition throws (hence /EHc-).
//
// GPU test: HIP device 1 via HIP_VISIBLE_DEVICES=1 (tests/kernels/CMakeLists.txt). The golden part
// runs last and returns 77 (SKIPPED) when the golden directory is absent, only if everything else
// passed.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "npy_fixture.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"
#include "trellis_ref.hpp"

using namespace r4dx::core;

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    if (g_failures < 40) std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

template <typename T>
int64_t P(T* p) {
  return reinterpret_cast<int64_t>(p);
}

template <typename Fn>
bool Throws(Fn fn) {
  try {
    fn();
  } catch (const std::exception&) {
    return true;
  }
  return false;
}

std::string GoldenDir() { return std::string(R4DX_SOURCE_DIR) + "/tests/kernels/golden/trellis"; }

bool GoldenAvailable() {
  std::ifstream f(GoldenDir() + "/manifest.json", std::ios::binary);
  return static_cast<bool>(f);
}

// bf16 activations as trellis_golden.py's random_x: row scales log-uniform in [2^-8, 2^3] (the small
// rows reach the f16 subnormal range at prescale 0) and two 20-60x outliers per row.
std::vector<uint16_t> RandomX(std::mt19937_64& rng, int64_t M, int64_t K) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(-8.f, 3.f), od(20.f, 60.f);
  std::uniform_int_distribution<int64_t> kd(0, K - 1);
  std::vector<uint16_t> x(static_cast<size_t>(M * K));
  for (int64_t m = 0; m < M; ++m) {
    const float s = std::exp2(ud(rng));
    std::vector<float> row(static_cast<size_t>(K));
    for (auto& v : row) v = nd(rng) * s;
    for (int i = 0; i < 2; ++i) row[static_cast<size_t>(kd(rng))] *= od(rng);
    for (int64_t k = 0; k < K; ++k) x[static_cast<size_t>(m * K + k)] = FloatToBf16(row[static_cast<size_t>(k)]);
  }
  return x;
}

// fp16 +-scales with magnitudes log-uniform in [lo, hi], widened to fp32 (what the loader uploads).
std::vector<float> RandomScales(std::mt19937_64& rng, int64_t n, double lo, double hi) {
  std::uniform_real_distribution<double> ud(std::log(lo), std::log(hi));
  std::uniform_int_distribution<int> sd(0, 1);
  std::vector<float> s(static_cast<size_t>(n));
  for (auto& v : s) v = F16ToFloat(FloatToF16(static_cast<float>((sd(rng) ? -1.0 : 1.0) * std::exp(ud(rng)))));
  return s;
}

size_t CountDiff(const uint16_t* a, const uint16_t* b, size_t n) {
  size_t d = 0;
  for (size_t i = 0; i < n; ++i) d += a[i] != b[i];
  return d;
}

// ---- FWHT: the wave butterfly against FwhtLds ------------------------------------------------------
void TestFwht(std::mt19937_64& rng) {
  const int blocks = 4096;
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ed(-60.f, 60.f);
  std::vector<float> x(static_cast<size_t>(blocks) * 128);
  for (int b = 0; b < blocks; ++b) {
    const float s = std::exp2(ed(rng));
    for (int i = 0; i < 128; ++i) x[static_cast<size_t>(b) * 128 + i] = nd(rng) * s;
  }
  // A few blocks of exact values, where every butterfly is exact and signed zeros appear.
  for (int i = 0; i < 128; ++i) {
    x[i] = 0.f;
    x[128 + i] = (i & 1) ? -0.f : 0.f;
    x[256 + i] = static_cast<float>((i * 37) % 11) - 5.f;
  }
  std::vector<float> cpu = x;
  for (int b = 0; b < blocks; ++b) trellis_ref::FwhtLdsF32(&cpu[static_cast<size_t>(b) * 128]);
  DeviceBuffer<float> d_wave(x.size()), d_lds(x.size());
  d_wave.CopyFromHost(x);
  d_lds.CopyFromHost(x);
  r4dx_fwht128_f32(P(d_wave.data()), blocks, 0, 0);
  r4dx_fwht128_f32(P(d_lds.data()), blocks, 1, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<float> wave = d_wave.CopyToHost(), lds = d_lds.CopyToHost();
  size_t wv_lds = 0, wv_cpu = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    uint32_t a, b, c;
    std::memcpy(&a, &wave[i], 4);
    std::memcpy(&b, &lds[i], 4);
    std::memcpy(&c, &cpu[i], 4);
    wv_lds += a != b;
    wv_cpu += a != c;
  }
  std::printf("  FWHT-128 on %d random fp32 blocks: wave vs FwhtLds %zu, wave vs CPU %zu values differ\n",
              blocks, wv_lds, wv_cpu);
  Check(wv_lds == 0, "r4d_fwht128_wave differs from FwhtLds in " + std::to_string(wv_lds) + " values");
  Check(wv_cpu == 0, "r4d_fwht128_wave differs from the CPU FwhtLds loop in " + std::to_string(wv_cpu) + " values");
}

// ---- the transform against the CPU emulation ---------------------------------------------------
// One call: nout outputs into one buffer part_stride elements apart, with guard elements before,
// between and after them. Returns the number of wrong elements (outputs and guards).
size_t RunTransform(const std::vector<uint16_t>& x, int64_t M, int64_t K, int nout,
                    const std::vector<std::vector<float>>& suh, int prescale, int64_t* subnormals) {
  constexpr uint16_t kGuard = 0x7E5Au;   // a NaN no transform of finite values can produce
  const int64_t guard = 256;
  const int64_t part_stride = M * K + 3 * 128;   // parts not back to back: a gap to catch overruns
  const size_t total = static_cast<size_t>(guard + nout * part_stride + guard);
  DeviceBuffer<uint16_t> d_x(x.size()), d_out(total);
  d_x.CopyFromHost(x);
  std::vector<uint16_t> init(total, kGuard);
  d_out.CopyFromHost(init);
  std::vector<DeviceBuffer<float>> d_suh;
  int64_t suh_p[3] = {0, 0, 0}, out_p[3] = {0, 0, 0};
  for (int o = 0; o < nout; ++o) {
    d_suh.emplace_back(suh[o].size());
    d_suh.back().CopyFromHost(suh[o]);
    suh_p[o] = P(d_suh.back().data());
    out_p[o] = P(d_out.data() + guard + o * part_stride);
  }
  r4dx_trellis_input_bf16(P(d_x.data()), M, K, nout, suh_p, out_p, prescale, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<uint16_t> got = d_out.CopyToHost();
  std::vector<uint16_t> want(init);
  for (int o = 0; o < nout; ++o) {
    const std::vector<uint16_t> a = trellis_ref::TransformInput(x, M, K, suh[o].data(), prescale);
    std::copy(a.begin(), a.end(), want.begin() + guard + o * part_stride);
    for (uint16_t h : a) *subnormals += (h & 0x7C00u) == 0 && (h & 0x03FFu) != 0;
  }
  const size_t bad = CountDiff(got.data(), want.data(), total);
  static int reported = 0;
  for (size_t i = 0; i < total && bad && reported < 6; ++i) {
    if (got[i] == want[i]) continue;
    ++reported;
    std::printf("    mismatch at %zu (output %lld, row %lld, k %lld): got 0x%04x (%.9g) want 0x%04x (%.9g)\n", i,
                static_cast<long long>((static_cast<int64_t>(i) - guard) / part_stride),
                static_cast<long long>(((static_cast<int64_t>(i) - guard) % part_stride) / K),
                static_cast<long long>(((static_cast<int64_t>(i) - guard) % part_stride) % K), got[i],
                F16ToFloat(got[i]), want[i], F16ToFloat(want[i]));
  }
  return bad;
}

void TestTransform(std::mt19937_64& rng) {
  int configs = 0;
  size_t bad_total = 0;
  int64_t subnormals = 0;
  const int64_t Ks[] = {3072, 5120, 6144, 8704, 17408};
  const int64_t Ms[] = {1, 3, 8, 16, 17, 64};
  for (int64_t K : Ks) {
    std::vector<std::vector<float>> suh;
    for (int o = 0; o < 3; ++o) suh.push_back(RandomScales(rng, K, 5e-3, 4e-2));
    for (int64_t M : Ms) {
      const std::vector<uint16_t> x = RandomX(rng, M, K);
      for (int nout = 1; nout <= 3; ++nout)
        for (int s : {0, 4, -3}) {
          const size_t bad = RunTransform(x, M, K, nout, suh, s, &subnormals);
          ++configs;
          bad_total += bad;
          Check(bad == 0, "transform K=" + std::to_string(K) + " M=" + std::to_string(M) + " nout=" +
                              std::to_string(nout) + " s=" + std::to_string(s) + ": " +
                              std::to_string(bad) + " elements wrong");
        }
    }
  }
  // Every M = 1..64 at K = 5120, two parts (mlp.gate_up's call).
  {
    const int64_t K = 5120;
    std::vector<std::vector<float>> suh;
    for (int o = 0; o < 2; ++o) suh.push_back(RandomScales(rng, K, 5e-3, 4e-2));
    const std::vector<uint16_t> x64 = RandomX(rng, 64, K);
    for (int64_t M = 1; M <= 64; ++M) {
      const std::vector<uint16_t> x(x64.begin(), x64.begin() + M * K);
      const size_t bad = RunTransform(x, M, K, 2, suh, 0, &subnormals);
      ++configs;
      bad_total += bad;
      Check(bad == 0, "transform K=5120 nout=2 M=" + std::to_string(M) + ": " + std::to_string(bad) +
                          " elements wrong");
    }
  }
  std::printf("  transform: %d configurations (K x M x nout x prescale, and M = 1..64) bit-exact%s; "
              "%lld subnormal f16 outputs among them\n",
              configs, bad_total ? " (NOT all)" : "", static_cast<long long>(subnormals));
  Check(subnormals > 0, "the random activations never reached the f16 subnormal range");
}

void TestPreconditions() {
  int64_t suh[3] = {16, 16, 16}, out[3] = {16, 16, 16}, zero[3] = {16, 0, 16};
  auto call = [&](int64_t x, int64_t M, int64_t K, int nout, const int64_t* s, const int64_t* o, int p) {
    r4dx_trellis_input_bf16(x, M, K, nout, s, o, p, 0);
  };
  Check(Throws([&] { call(16, 1, 5120, 0, suh, out, 0); }), "nout = 0 accepted");
  Check(Throws([&] { call(16, 1, 5120, 4, suh, out, 0); }), "nout = 4 accepted");
  Check(Throws([&] { call(16, 1, 5000, 1, suh, out, 0); }), "K % 128 accepted");
  Check(Throws([&] { call(16, 1, 0, 1, suh, out, 0); }), "K = 0 accepted");
  Check(Throws([&] { call(16, -1, 5120, 1, suh, out, 0); }), "M = -1 accepted");
  Check(Throws([&] { call(16, 65536, 5120, 1, suh, out, 0); }), "M = 65536 accepted");
  Check(Throws([&] { call(16, 1, 5120, 1, suh, out, 25); }), "prescale 25 accepted");
  Check(Throws([&] { call(16, 1, 5120, 1, nullptr, out, 0); }), "null suh array accepted");
  Check(Throws([&] { call(16, 1, 5120, 1, suh, nullptr, 0); }), "null out array accepted");
  Check(Throws([&] { call(16, 1, 5120, 2, suh, zero, 0); }), "null out[1] accepted");
  Check(Throws([&] { call(0, 1, 5120, 1, suh, out, 0); }), "null x accepted");
  Check(!Throws([&] { call(16, 0, 5120, 1, suh, out, 0); }), "M = 0 is a no-op, not an error");
  Check(Throws([&] { r4dx_fwht128_f32(0, 4, 0, 0); }), "fwht128 null x accepted");
}

// ---- goldens (trellis_golden.py) ---------------------------------------------------------------
std::vector<uint16_t> Bf16FromF32(const std::vector<float>& v) {
  std::vector<uint16_t> b(v.size());
  for (size_t i = 0; i < v.size(); ++i) b[i] = FloatToBf16(v[i]);
  return b;
}

std::vector<float> Widen(const std::vector<uint16_t>& h, size_t off, size_t n) {
  std::vector<float> f(n);
  for (size_t i = 0; i < n; ++i) f[i] = F16ToFloat(h[off + i]);
  return f;
}

// x (fp32 holding bf16 values) [M][K], suh f16 [parts][K], a f16 [parts][M][K]: one call with
// nout = parts.
void GoldenCase(const std::string& label, const std::vector<float>& xf, const std::vector<uint16_t>& suh16,
                const std::vector<uint16_t>& a, int64_t M, int64_t K, int parts, int prescale) {
  const std::vector<uint16_t> x = Bf16FromF32(xf);
  size_t inexact = 0;
  for (size_t i = 0; i < xf.size(); ++i) inexact += Bf16ToFloat(x[i]) != xf[i];
  Check(inexact == 0, label + ": golden x is not bf16-exact");
  DeviceBuffer<uint16_t> d_x(x.size()), d_out(static_cast<size_t>(parts * M * K));
  d_x.CopyFromHost(x);
  std::vector<DeviceBuffer<float>> d_suh;
  d_suh.reserve(3);
  int64_t sp[3] = {0, 0, 0}, op[3] = {0, 0, 0};
  for (int p = 0; p < parts; ++p) {
    d_suh.emplace_back(static_cast<size_t>(K));
    d_suh.back().CopyFromHost(Widen(suh16, static_cast<size_t>(p * K), static_cast<size_t>(K)));
    sp[p] = P(d_suh.back().data());
    op[p] = P(d_out.data() + p * M * K);
  }
  r4dx_trellis_input_bf16(P(d_x.data()), M, K, parts, sp, op, prescale, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<uint16_t> got = d_out.CopyToHost();
  const size_t diff = CountDiff(got.data(), a.data(), a.size());
  std::printf("  golden %-18s M=%lld K=%lld P=%d s=%d: %zu of %zu values differ\n", label.c_str(),
              static_cast<long long>(M), static_cast<long long>(K), parts, prescale, diff, a.size());
  Check(diff == 0, "golden " + label + ": " + std::to_string(diff) + " values differ");
}

void TestGoldens() {
  using r4dx_test::LoadNpyF16Bits;
  using r4dx_test::LoadNpyF32;
  const std::string d = GoldenDir() + "/";
  {
    const int64_t M = 64, K = 1024;
    const std::vector<float> x = LoadNpyF32(d + "rand_x.npy", {M, K});
    const std::vector<uint16_t> suh = LoadNpyF16Bits(d + "rand_suh.npy", {2, K});
    for (int s : {0, 4}) {
      const std::vector<uint16_t> a =
          LoadNpyF16Bits(d + "rand_a_s" + std::to_string(s) + ".npy", {2, M, K});
      GoldenCase("rand", x, suh, a, M, K, 2, s);
    }
  }
  const int64_t M = 64, K = 256;
  const std::vector<float> x = LoadNpyF32(d + "real_x.npy", {M, K});
  for (const char* c : {"attn_k", "mlp_down", "mlp_gate_up"}) {
    const std::string base = d + "real_" + c;
    const int parts = std::string(c) == "mlp_gate_up" ? 2 : 1;
    const std::vector<uint16_t> suh = LoadNpyF16Bits(base + "_suh.npy", {parts, K});
    const std::vector<uint16_t> a = LoadNpyF16Bits(base + "_a.npy", {parts, M, K});
    GoldenCase(std::string("real ") + c, x, suh, a, M, K, parts, 0);
  }
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937_64 rng(20260927);

  std::printf("trellis input transform (docs/trellis-kernel.md 4.8, M2)\n");
  TestPreconditions();
  TestFwht(rng);
  TestTransform(rng);

  const bool golden = GoldenAvailable();
  if (golden) {
    TestGoldens();
  } else {
    std::printf("  goldens: %s absent (tools/reference/trellis_golden.py writes them)\n",
                GoldenDir().c_str());
  }

  if (g_failures != 0) {
    std::printf("FAIL (%d of %d checks)\n", g_failures, g_checks);
    return 1;
  }
  if (!golden) {
    std::printf("PASS without goldens (%d checks) -> SKIPPED\n", g_checks);
    return r4dx_test::kSkipReturnCode;
  }
  std::printf("PASS (%d checks)\n", g_checks);
  return 0;
}
