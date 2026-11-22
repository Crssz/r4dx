// tests/kernels/test_mxfp8_gemm.cpp -- r4dx_gemm_mxfp8_nt (docs/inkling2-plan.md 4.1, mxfp8_gemm.h) against
// mxfp8_ref.hpp, at Inkling2's shapes ([N, K]: attn q / gate 3072x2304, k / v 1024x2304, o 2304x3072, rel 384x2304,
// expert gate_up 2304x2304, expert down 2304x1152, the TP = 2 rank halves 1152x2304 and 2304x576, and lm_head
// 200064x2304 = the 200058 used rows rounded up to whole 16-row tiles, at M = 1 with sampled columns).
//
//   * EXACT CLASS (A rows m % 4 != 3): activations in {0, +-1, +-2}, so every WMMA partial sum is exact whatever its
//     internal order; there the kernel must equal mxfp8_ref::Gemm BIT FOR BIT. Weight rows (n % 64): random codes
//     over all 254 non-NaN e4m3 values (subnormals, +-448, -0 included) at scale bytes 110..130; row 5 every group at
//     scale byte 0 (2^-127, an f32 subnormal: results in the f32 subnormal range); row 6 group 0 at scale byte 254
//     (2^127) with codes +-2^-9 only; row 7 subnormal codes only; row 8 codes {+-448, +-0} only;
//   * GENERAL rows (m % 4 == 3): Gaussian bf16 activations (sd 1, clipped to +-4) and a tiny row (1e-5): within
//     |c - ref| <= 2 bf16 ulp(|ref|) + 1e-6 ||a|| ||w|| of the fp64 GEMM; the bitwise count is reported;
//   * M in {1, 2, 3, 8, 16, 17, 63, 64, 65, 255, 256} (decode GEMV rows and m64-tiled prefill up to 256 rows) at
//     SK in {1, 2, 4, 8} where legal and the fallback geometry; rows of every launch == the M = 256 launch's rows;
//     M = 8 / 16 launches at an A row offset == their M = 1 launches; two runs identical; WV / MB / NPW / NT do not
//     change a byte at fixed SK (M = 256 and M = 17);
//   * hand cases: every e4m3 code through the kernel (code n at k = 0 of row n, a = 1) == bf16(DecodeE4m3(n)) bit for
//     bit, with scale bytes 127, 0 and 254; single activations (1e5 -> 65504 RTZ, 1e-6 -> f16 subnormal, 1e-9 -> 0);
//   * illegal launches throw.
// GPU test (HIP device 1 via HIP_VISIBLE_DEVICES=1 in CMakeLists.txt). WRITTEN, NOT YET RUN by its author. Built with
// /EHc-. Optional argument: --quick (M in {1, 17, 256}, SK fallback only, no lm_head).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "mxfp8_ref.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/mxfp8_gemm.h"

using namespace r4dx::core;
using w8a16_ref::Bf16ToFloat;
using w8a16_ref::Bf16Ulp;
using w8a16_ref::FloatToBf16Rne;

namespace {

bool g_ok = true;
int g_checks = 0, g_fails = 0;
void Check(bool c, const std::string& what) {
  std::printf("%-118s %s\n", what.c_str(), c ? "ok" : "FAIL");
  std::fflush(stdout);
  ++g_checks;
  if (!c) ++g_fails;
  g_ok = g_ok && c;
}
int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

constexpr int kRows = 256;  // A rows

struct Geo {
  int WV, SK, MB, NPW, NT;
};
std::string GeoStr(const Geo& g) {
  return "WV" + std::to_string(g.WV) + " SK" + std::to_string(g.SK) + " MB" + std::to_string(g.MB) + " NPW" +
         std::to_string(g.NPW) + " NT" + std::to_string(g.NT);
}
bool Legal(const Geo& g, int K) {
  return g.WV >= 1 && g.SK >= 1 && K % (g.SK * 32) == 0 && g.WV * g.SK * 32 <= 1024 && g.MB >= 1 && g.MB <= 4 &&
         (g.NPW == 1 || g.NPW == 4) && g.WV * g.NPW * g.SK <= 64;
}
// The model's Mxfp8FallbackTuning at a given SK (src/model/linear.cpp).
Geo Fallback(int M, int N, int SK) {
  Geo g;
  g.SK = SK;
  g.WV = (N / 16 < 512) ? 1 : 4;
  if (M <= 16) {
    g.MB = 1, g.NPW = 1, g.NT = 1;
  } else {
    g.MB = std::min(4, (M + 15) / 16), g.NPW = 4, g.NT = 0;
  }
  while (g.WV > 1 && (g.WV * g.NPW * SK > 64 || g.WV * SK * 32 > 1024)) --g.WV;
  if (g.WV * g.NPW * SK > 64) g.NPW = 1;
  return g;
}
int FallbackSk(int K, int M, int N) {
  for (int sk : {8, 4, 2, 1}) {
    const Geo g = Fallback(M, N, sk);
    if (Legal(g, K)) return sk;
  }
  return 1;
}

struct Problem {
  std::string name;
  int N, K;
  std::vector<int> cols;
  DeviceBuffer<uint8_t> wq, ws;
  mxfp8_ref::Unpacked w;
};

uint8_t RandomCode(std::mt19937& rng) {
  std::uniform_int_distribution<int> d(0, 255);
  for (;;) {
    const int c = d(rng);
    if (c != 0x7F && c != 0xFF) return static_cast<uint8_t>(c);
  }
}

void MakeWeight(std::mt19937& rng, Problem& p) {
  const int N = p.N, K = p.K, ng = K / 32;
  std::vector<uint8_t> code(static_cast<size_t>(N) * K), scale(static_cast<size_t>(N) * ng);
  std::uniform_int_distribution<int> ds(110, 130), d7(1, 7), d01(0, 1), d4(0, 3);
  const uint8_t big_zero[4] = {0x7E, 0xFE, 0x00, 0x80};
  for (int n = 0; n < N; ++n) {
    const int kind = n % 64;
    for (int k = 0; k < K; ++k) {
      uint8_t c = RandomCode(rng);
      if (kind == 6 && k < 32) c = d01(rng) ? 0x01 : 0x81;
      if (kind == 7) c = static_cast<uint8_t>(d7(rng) | (d01(rng) ? 0x80 : 0));
      if (kind == 8) c = big_zero[d4(rng)];
      code[static_cast<size_t>(n) * K + k] = c;
    }
    for (int g = 0; g < ng; ++g) {
      uint8_t s = static_cast<uint8_t>(ds(rng));
      if (kind == 5) s = 0;
      if (kind == 6 && g == 0) s = 254;
      if (kind == 7) s = 127;
      if (kind == 8) s = 120;
      scale[static_cast<size_t>(n) * ng + g] = s;
    }
  }
  const auto wq = mxfp8_ref::PackWq(code, N, K);
  const auto ws = mxfp8_ref::PackWs(scale, N, K);
  p.wq = DeviceBuffer<uint8_t>(wq.size());
  p.wq.CopyFromHost(wq);
  p.ws = DeviceBuffer<uint8_t>(ws.size());
  p.ws.CopyFromHost(ws);
  p.w = mxfp8_ref::UnpackRows(wq, ws, N, K, p.cols);
}

std::vector<int> SampleCols(std::mt19937& rng, int N) {
  if (N <= 256) {
    std::vector<int> c(N);
    for (int i = 0; i < N; ++i) c[i] = i;
    return c;
  }
  std::set<int> s;
  for (int i = 0; i < 32; ++i) s.insert(i);
  for (int i = N - 32; i < N; ++i) s.insert(i);
  for (int i = 64; i < 80; ++i) s.insert(i);
  std::uniform_int_distribution<int> d(0, N - 1);
  while (s.size() < 256) s.insert(d(rng));
  return std::vector<int>(s.begin(), s.end());
}

bool ExactRow(int m) { return m % 4 != 3; }

std::vector<uint16_t> MakeA(std::mt19937& rng, int K) {
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::uniform_int_distribution<int> de(-2, 2);
  std::vector<uint16_t> a(static_cast<size_t>(kRows) * K);
  for (int m = 0; m < kRows; ++m)
    for (int k = 0; k < K; ++k) {
      float x;
      if (ExactRow(m)) {
        x = static_cast<float>(de(rng));
      } else {
        x = std::clamp(d(rng), -4.0f, 4.0f);
        if (m == 7) x *= 1e-5f;
      }
      a[static_cast<size_t>(m) * K + k] = FloatToBf16Rne(x);
    }
  return a;
}

std::vector<uint16_t> Run(DeviceBuffer<uint16_t>& a_d, size_t a_off_rows, Problem& p, int M, const Geo& g,
                          DeviceBuffer<uint16_t>& c_d) {
  const int64_t a = P(a_d.data()) + static_cast<int64_t>(a_off_rows * p.K * sizeof(uint16_t));
  R4DX_HIP_CHECK(hipMemset(c_d.data(), 0xff, static_cast<size_t>(M) * p.N * sizeof(uint16_t)));
  r4dx_gemm_mxfp8_nt(a, P(p.wq.data()), P(p.ws.data()), P(c_d.data()), M, p.K, p.N, g.WV, g.SK, g.MB, g.NPW, g.NT, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> out(static_cast<size_t>(M) * p.N);
  R4DX_HIP_CHECK(hipMemcpy(out.data(), c_d.data(), out.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
  return out;
}

bool RowsEqual(const std::vector<uint16_t>& x, size_t xrow, const std::vector<uint16_t>& y, size_t yrow, int N) {
  return std::memcmp(&x[xrow * N], &y[yrow * N], static_cast<size_t>(N) * sizeof(uint16_t)) == 0;
}

void TestProblem(std::mt19937& rng, const std::string& name, int N, int K, const std::vector<int>& Ms, bool quick) {
  Problem p;
  p.name = name;
  p.N = N, p.K = K;
  p.cols = SampleCols(rng, N);
  MakeWeight(rng, p);
  const auto a = MakeA(rng, K);
  DeviceBuffer<uint16_t> a_d(a.size());
  a_d.CopyFromHost(a);
  const int Mmax = *std::max_element(Ms.begin(), Ms.end());
  DeviceBuffer<uint16_t> c_d(static_cast<size_t>(Mmax) * N);
  const mxfp8_ref::NaiveOut nv = mxfp8_ref::GemmNaive(a, p.w, Mmax, K);
  const int R = p.w.rows;
  const std::string tag = name + " [" + std::to_string(N) + "x" + std::to_string(K) + "]";

  std::vector<int> sks;
  const int fsk = FallbackSk(K, Mmax, N);
  if (quick) {
    sks = {fsk};
  } else {
    for (int sk : {1, 2, 4, 8})
      if (K % (sk * 32) == 0) sks.push_back(sk);
    if (std::find(sks.begin(), sks.end(), fsk) == sks.end()) sks.push_back(fsk);
  }
  for (int SK : sks) {
    const auto ref = mxfp8_ref::Gemm(a, p.w, Mmax, K, SK);
    const std::string tsk = tag + " SK" + std::to_string(SK);
    const Geo ganchor = Fallback(Mmax, N, SK);
    if (!Legal(ganchor, K)) continue;
    const auto canchor = Run(a_d, 0, p, Mmax, ganchor, c_d);
    Check(canchor == Run(a_d, 0, p, Mmax, ganchor, c_d), tsk + " M" + std::to_string(Mmax) + ": two runs identical");
    for (int M : Ms) {
      const Geo g = Fallback(M, N, SK);
      if (!Legal(g, K)) continue;
      const auto c = M == Mmax ? canchor : Run(a_d, 0, p, M, g, c_d);
      size_t bad = 0, exact_bad = 0, exact_n = 0, gen_bit = 0, gen_n = 0;
      double worst = 0;
      for (int m = 0; m < M; ++m)
        for (int ci = 0; ci < R; ++ci) {
          const uint16_t gotb = c[static_cast<size_t>(m) * N + p.cols[ci]];
          const size_t ri = static_cast<size_t>(m) * R + ci;
          if (ExactRow(m)) {
            ++exact_n;
            exact_bad += gotb != ref[ri];
          } else {
            const double got = Bf16ToFloat(gotb);
            bad += !mxfp8_ref::WithinBound(got, nv.c[ri], nv.a_norm[m], nv.w_norm[ci]);
            gen_bit += gotb == ref[ri];
            worst = std::max(worst, std::fabs(got - nv.c[ri]) / (2.0 * Bf16Ulp(nv.c[ri]) + 1e-6 * nv.a_norm[m] * nv.w_norm[ci]));
            ++gen_n;
          }
        }
      char buf[200];
      std::snprintf(buf, sizeof buf, " M%d %s: exact class %zu/%zu bitwise; general %zu/%zu in bound (worst %.3f), %zu bitwise", M,
                    GeoStr(g).c_str(), exact_n - exact_bad, exact_n, gen_n - bad, gen_n, worst, gen_bit);
      Check(exact_bad == 0 && bad == 0, tsk + buf);
      bool rows = true;
      for (int m = 0; m < M; ++m) rows &= RowsEqual(c, m, canchor, m, N);
      Check(rows, tsk + " M" + std::to_string(M) + ": rows == the M" + std::to_string(Mmax) + " launch's rows");
    }
    for (int M : {8, 16}) {
      if (std::find(Ms.begin(), Ms.end(), M) == Ms.end() || Mmax < 64) continue;
      const size_t off = M == 8 ? 40 : 7;
      const auto cm = Run(a_d, off, p, M, Fallback(M, N, SK), c_d);
      bool ok = true;
      for (int m = 0; m < M; ++m) {
        const auto c1 = Run(a_d, off + m, p, 1, Fallback(1, N, SK), c_d);
        ok &= RowsEqual(cm, m, c1, 0, N) && RowsEqual(cm, m, canchor, off + m, N);
      }
      Check(ok, tsk + " M" + std::to_string(M) + " (A row offset " + std::to_string(off) + "): every row == its M1 launch");
    }
    if (quick || Mmax < 17) continue;
    for (int M : {Mmax, 17}) {
      size_t tried = 0, differ = 0;
      std::string first_bad;
      for (int WV : {1, 2, 3, 4})
        for (int MB = 1; MB <= 4; ++MB)
          for (int NPW : {1, 4})
            for (int NT : {0, 1}) {
              const Geo g{WV, SK, MB, NPW, NT};
              if (!Legal(g, K)) continue;
              const auto c = Run(a_d, 0, p, M, g, c_d);
              ++tried;
              bool same = true;
              for (int m = 0; m < M; ++m) same &= RowsEqual(c, m, canchor, m, N);
              if (!same && first_bad.empty()) first_bad = GeoStr(g);
              differ += !same;
            }
      Check(differ == 0, tsk + " M" + std::to_string(M) + ": " + std::to_string(tried) +
                             " geometries bit-identical" + (first_bad.empty() ? "" : " (first differing " + first_bad + ")"));
    }
  }
}

// Every code through the kernel: row n has code n at k = 0 (NaN codes replaced by 0), a = [1, 0, ...].
void TestEveryCode() {
  const int N = 256, K = 64;
  for (int sbyte : {127, 0, 254}) {
    std::vector<uint8_t> code(static_cast<size_t>(N) * K, 0), scale(static_cast<size_t>(N) * (K / 32), 127);
    for (int n = 0; n < N; ++n) {
      uint8_t c = static_cast<uint8_t>(n);
      if (c == 0x7F || c == 0xFF) c = 0;
      // 2^127 times a code above 1.75 would overflow f32: keep |value| <= 1 at scale 254
      if (sbyte == 254 && std::fabs(r4dx_convert::mxfp8::DecodeE4m3(c)) > 1.0f) c = 0x38;
      code[static_cast<size_t>(n) * K] = c;
      scale[static_cast<size_t>(n) * (K / 32)] = static_cast<uint8_t>(sbyte);
    }
    const auto wq = mxfp8_ref::PackWq(code, N, K);
    const auto ws = mxfp8_ref::PackWs(scale, N, K);
    std::vector<uint16_t> a(K, 0);
    a[0] = FloatToBf16Rne(1.0f);
    DeviceBuffer<uint8_t> wq_d(wq.size()), ws_d(ws.size());
    wq_d.CopyFromHost(wq);
    ws_d.CopyFromHost(ws);
    DeviceBuffer<uint16_t> a_d(a.size()), c_d(N);
    a_d.CopyFromHost(a);
    r4dx_gemm_mxfp8_nt(P(a_d.data()), P(wq_d.data()), P(ws_d.data()), P(c_d.data()), 1, K, N, 1, 1, 1, 1, 1, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const auto c = c_d.CopyToHost();
    bool ok = true;
    int first = -1;
    for (int n = 0; n < N; ++n) {
      const uint8_t cd = code[static_cast<size_t>(n) * K];
      const float want = std::ldexp(r4dx_convert::mxfp8::DecodeE4m3(cd), sbyte - 127);
      const uint16_t wb = FloatToBf16Rne(want);
      // -0 * 1 summed from +0: the kernel's split sum starts at 0.0f, so -0 comes out +0 (as the reference)
      const uint16_t wref = (wb == 0x8000) ? 0 : wb;
      // A value below the bf16 range (e.g. -2^-136 at scale byte 0) may come out as either signed zero (the f32
      // product is flushed with its sign kept); the bitwise check against mxfp8_ref::Gemm below pins the exact bits.
      const bool both_zero = (wref & 0x7FFF) == 0 && (c[n] & 0x7FFF) == 0;
      if (c[n] != wref && !both_zero) {
        ok = false;
        if (first < 0) first = n;
      }
    }
    Check(ok, "every e4m3 code at scale byte " + std::to_string(sbyte) + " -> bf16(decode * 2^" + std::to_string(sbyte - 127) +
                  ") bit for bit" + (first < 0 ? "" : " (first bad code " + std::to_string(first) + ")"));
    const auto ref = mxfp8_ref::Gemm(a, mxfp8_ref::UnpackRows(wq, ws, N, K), 1, K, 1);
    Check(c == ref, "every e4m3 code at scale byte " + std::to_string(sbyte) + ": == mxfp8_ref::Gemm bitwise");
  }
}

void TestHandActivations() {
  const int N = 16, K = 64;
  std::vector<uint8_t> code(static_cast<size_t>(N) * K, 0), scale(static_cast<size_t>(N) * (K / 32), 127);
  for (int n = 0; n < N; ++n) code[static_cast<size_t>(n) * K + n] = 0x38;  // 1.0
  const auto wq = mxfp8_ref::PackWq(code, N, K);
  const auto ws = mxfp8_ref::PackWs(scale, N, K);
  const std::vector<float> vals = {1e5f, -1e5f, 3e38f, 65504.0f, 1e-6f, -1e-6f, 1e-9f, 1.0f,
                                   -0.0f, 65280.0f, 6e4f, 6.1e-5f, 3.0f, -7.5f, 1e-4f, 2.5e4f};
  std::vector<uint16_t> a(K, 0);
  for (int n = 0; n < N; ++n) a[n] = FloatToBf16Rne(vals[n]);
  DeviceBuffer<uint8_t> wq_d(wq.size()), ws_d(ws.size());
  wq_d.CopyFromHost(wq);
  ws_d.CopyFromHost(ws);
  DeviceBuffer<uint16_t> a_d(a.size()), c_d(N);
  a_d.CopyFromHost(a);
  r4dx_gemm_mxfp8_nt(P(a_d.data()), P(wq_d.data()), P(ws_d.data()), P(c_d.data()), 1, K, N, 1, 1, 1, 1, 1, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const auto c = c_d.CopyToHost();
  const auto ref = mxfp8_ref::Gemm(a, mxfp8_ref::UnpackRows(wq, ws, N, K), 1, K, 1);
  Check(c == ref, "hand: single activations through code 1.0 == reference bitwise");
  Check(c[0] == FloatToBf16Rne(65504.0f) && c[1] == FloatToBf16Rne(-65504.0f) && c[2] == FloatToBf16Rne(65504.0f),
        "hand: 1e5, -1e5, 3e38 -> +-65504 (RTZ saturation)");
  Check(Bf16ToFloat(c[4]) == w8a16_ref::F16ToFloat(0x0010) && Bf16ToFloat(c[6]) == 0.0f,
        "hand: 1e-6 -> 16 * 2^-24 (f16 subnormal), 1e-9 -> 0");
}

void TestIllegal() {
  DeviceBuffer<uint16_t> a(256 * 256), c(256 * 64);
  DeviceBuffer<uint8_t> wq(64 * 256), ws(64 * 8);
  auto throws = [&](int M, int K, int N, int WV, int SK, int MB, int NPW, int NT, bool null_a = false) {
    try {
      r4dx_gemm_mxfp8_nt(null_a ? 0 : P(a.data()), P(wq.data()), P(ws.data()), P(c.data()), M, K, N, WV, SK, MB, NPW, NT, 0);
    } catch (const std::exception&) {
      return true;
    }
    return false;
  };
  Check(throws(0, 256, 64, 1, 1, 1, 1, 1), "illegal: M = 0 throws");
  Check(throws(257, 256, 64, 1, 1, 1, 1, 1), "illegal: M = 257 throws");
  Check(throws(1, 256, 24, 1, 1, 1, 1, 1), "illegal: N = 24 (not % 16) throws");
  Check(throws(1, 96, 64, 1, 2, 1, 1, 1), "illegal: K 96 % (SK 2 * 32) throws");
  Check(throws(1, 80, 64, 1, 1, 1, 1, 1), "illegal: K 80 % 32 throws");
  Check(throws(1, 256, 64, 9, 4, 1, 1, 1), "illegal: WV*SK*32 > 1024 throws");
  Check(throws(1, 256, 64, 0, 1, 1, 1, 1), "illegal: WV = 0 throws");
  Check(throws(1, 256, 64, 1, 0, 1, 1, 1), "illegal: SK = 0 throws");
  Check(throws(1, 256, 64, 1, 1, 5, 1, 1), "illegal: MB = 5 throws");
  Check(throws(1, 256, 64, 1, 1, 1, 2, 1), "illegal: NPW = 2 throws");
  Check(throws(1, 256, 64, 8, 4, 1, 4, 1), "illegal: LDS WV 8 * NPW 4 * SK 4 = 128 KiB throws");
  Check(throws(1, 256, 64, 1, 1, 1, 1, 1, true), "illegal: null A throws");
  Check(!throws(256, 256, 64, 1, 8, 4, 4, 0), "legal: M 256 K 256 SK 8 MB 4 NPW 4 runs");
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  Check(r4dx_gemm_mxfp8_group() == 32 && r4dx_gemm_mxfp8_max_m() == 256, "exports: group 32, max_m 256");

  DeviceBuffer<uint64_t> ptrs(8);
  DeviceBuffer<int32_t> ids(16);
  auto throws_moe = [&](int n_pairs, int a_row_group, int n_experts, int K, int N, int WV, int SK, int NPW, int NT,
                        bool null_ids = false) {
    try {
      r4dx_gemm_mxfp8_moe_decode(P(a.data()), P(ptrs.data()), P(ptrs.data()), null_ids ? 0 : P(ids.data()), P(c.data()),
                                 n_pairs, a_row_group, n_experts, K, N, WV, SK, NPW, NT, 0);
    } catch (const std::exception&) {
      return true;
    }
    return false;
  };
  Check(throws_moe(0, 1, 8, 256, 64, 1, 1, 1, 1), "illegal moe_decode: n_pairs = 0 throws");
  Check(throws_moe(16385, 1, 8, 256, 64, 1, 1, 1, 1), "illegal moe_decode: n_pairs = 16385 throws");
  Check(throws_moe(8, 3, 8, 256, 64, 1, 1, 1, 1), "illegal moe_decode: a_row_group 3 not dividing 8 throws");
  Check(throws_moe(8, 0, 8, 256, 64, 1, 1, 1, 1), "illegal moe_decode: a_row_group = 0 throws");
  Check(throws_moe(8, 1, 0, 256, 64, 1, 1, 1, 1), "illegal moe_decode: n_experts = 0 throws");
  Check(throws_moe(8, 1, 257, 256, 64, 1, 1, 1, 1), "illegal moe_decode: n_experts = 257 throws");
  Check(throws_moe(8, 1, 8, 96, 64, 1, 2, 1, 1), "illegal moe_decode: K % (SK*32) != 0 throws");
  Check(throws_moe(8, 1, 8, 256, 24, 1, 1, 1, 1), "illegal moe_decode: N % 16 != 0 throws");
  Check(throws_moe(8, 1, 8, 256, 64, 1, 1, 2, 1), "illegal moe_decode: NPW = 2 throws");
  Check(throws_moe(8, 1, 8, 256, 64, 1, 1, 1, 1, true), "illegal moe_decode: null ids throws");
}

void TestMoeDecode(std::mt19937& rng, bool quick) {
  struct MoeShape {
    const char* name;
    int N, K, a_row_group, n_pairs;
  };
  const MoeShape shapes[] = {
      {"tiny.gate_up", 64, 64, 4, 8},
      {"tiny.down", 64, 32, 1, 8},
      {"inkling.gate_up", 2304, 2304, 8, 8},
      {"inkling.down", 2304, 1152, 1, 8},
      {"inkling.gate_up.t3", 2304, 2304, 8, 24},
      {"tp2.gate_up", 1152, 2304, 8, 8},
      {"tp2.down", 2304, 576, 1, 8},
  };
  const int n_experts = 10;
  for (const auto& s : shapes) {
    if (quick && std::string(s.name).find("tp2") != std::string::npos) continue;
    std::vector<Problem> experts(static_cast<size_t>(n_experts));
    std::vector<uint64_t> wq_h(static_cast<size_t>(n_experts)), ws_h(static_cast<size_t>(n_experts));
    for (int e = 0; e < n_experts; ++e) {
      experts[static_cast<size_t>(e)].N = s.N;
      experts[static_cast<size_t>(e)].K = s.K;
      experts[static_cast<size_t>(e)].cols = {0};
      MakeWeight(rng, experts[static_cast<size_t>(e)]);
      wq_h[static_cast<size_t>(e)] = static_cast<uint64_t>(P(experts[static_cast<size_t>(e)].wq.data()));
      ws_h[static_cast<size_t>(e)] = static_cast<uint64_t>(P(experts[static_cast<size_t>(e)].ws.data()));
    }
    DeviceBuffer<uint64_t> wq_ptrs(wq_h.size()), ws_ptrs(ws_h.size());
    wq_ptrs.CopyFromHost(wq_h);
    ws_ptrs.CopyFromHost(ws_h);

    const int n_a_rows = s.n_pairs / s.a_row_group;
    std::vector<uint16_t> a_h(static_cast<size_t>(n_a_rows) * s.K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : a_h) v = FloatToBf16Rne(std::clamp(nd(rng), -4.0f, 4.0f));
    DeviceBuffer<uint16_t> a_d(a_h.size());
    a_d.CopyFromHost(a_h);

    // Random ids including repeats and out-of-range sentinel ids (-1 and n_experts).
    std::vector<int32_t> ids_h(static_cast<size_t>(s.n_pairs));
    std::uniform_int_distribution<int> de(0, n_experts - 1);
    for (int p = 0; p < s.n_pairs; ++p) ids_h[static_cast<size_t>(p)] = de(rng);
    if (s.n_pairs >= 4) {
      ids_h[1] = ids_h[0];  // repeat
      ids_h[2] = -1;        // out-of-range low
      ids_h[3] = n_experts; // out-of-range high
    }
    DeviceBuffer<int32_t> ids_d(ids_h.size());
    ids_d.CopyFromHost(ids_h);

    const uint16_t kSentinel = 0x5A5Au;
    std::vector<uint16_t> init_c(static_cast<size_t>(s.n_pairs) * s.N, kSentinel);
    DeviceBuffer<uint16_t> c_grp(init_c.size()), c_ref(init_c.size());

    const int fsk = FallbackSk(s.K, 1, s.N);
    std::vector<Geo> geos = {Fallback(1, s.N, fsk)};
    if (!quick) {
      for (int sk : {1, 2, 4, 8}) {
        for (int npw : {1, 4}) {
          for (int nt : {0, 1}) {
            Geo g{1, sk, 1, npw, nt};
            if (Legal(g, s.K)) geos.push_back(g);
          }
        }
      }
    }
    for (const Geo& g : geos) {
      c_grp.CopyFromHost(init_c);
      c_ref.CopyFromHost(init_c);
      r4dx_gemm_mxfp8_moe_decode(P(a_d.data()), P(wq_ptrs.data()), P(ws_ptrs.data()), P(ids_d.data()), P(c_grp.data()),
                                 s.n_pairs, s.a_row_group, n_experts, s.K, s.N, g.WV, g.SK, g.NPW, g.NT, 0);
      for (int p = 0; p < s.n_pairs; ++p) {
        const int e = ids_h[static_cast<size_t>(p)];
        if (e < 0 || e >= n_experts) continue;
        const int64_t ap = P(a_d.data() + static_cast<size_t>(p / s.a_row_group) * s.K);
        const int64_t cp = P(c_ref.data() + static_cast<size_t>(p) * s.N);
        r4dx_gemm_mxfp8_nt(ap, P(experts[static_cast<size_t>(e)].wq.data()), P(experts[static_cast<size_t>(e)].ws.data()),
                           cp, 1, s.K, s.N, g.WV, g.SK, 1, g.NPW, g.NT, 0);
      }
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      const auto got = c_grp.CopyToHost();
      const auto ref = c_ref.CopyToHost();
      Check(got == ref, std::string("moe_decode ") + s.name + " [" + std::to_string(s.N) + "x" + std::to_string(s.K) +
                            "] pairs=" + std::to_string(s.n_pairs) + " grp=" + std::to_string(s.a_row_group) + " " +
                            GeoStr(g) + ": bit-identical to per-pair nt");
    }
  }
}

void TestMoePrefill(std::mt19937& rng, bool quick) {
  struct PfShape {
    const char* name;
    int N, K;
  };
  const PfShape shapes[] = {
      {"tiny.gate_up", 64, 64},
      {"tiny.down", 64, 32},
      {"inkling.gate_up", 2304, 2304},
      {"inkling.down", 2304, 1152},
      {"tp2.gate_up", 1152, 2304},
      {"tp2.down", 2304, 576},
  };
  // Counts per expert covering 0, 1, 7, 16, 17, 35 rows (empty, partial tile, exact tile, multi-tile).
  const std::vector<int> counts = {0, 1, 0, 7, 16, 0, 17, 35, 3, 0};
  const int n_experts = static_cast<int>(counts.size());
  std::vector<int32_t> off_h(static_cast<size_t>(n_experts + 1), 0);
  for (int e = 0; e < n_experts; ++e) off_h[static_cast<size_t>(e + 1)] = off_h[static_cast<size_t>(e)] + counts[static_cast<size_t>(e)];
  const int n_pairs = off_h[static_cast<size_t>(n_experts)];
  DeviceBuffer<int32_t> off_d(off_h.size());
  off_d.CopyFromHost(off_h);

  for (const auto& s : shapes) {
    if (quick && std::string(s.name).find("tp2") != std::string::npos) continue;
    std::vector<Problem> experts(static_cast<size_t>(n_experts));
    std::vector<uint64_t> wq_h(static_cast<size_t>(n_experts)), ws_h(static_cast<size_t>(n_experts));
    for (int e = 0; e < n_experts; ++e) {
      experts[static_cast<size_t>(e)].N = s.N;
      experts[static_cast<size_t>(e)].K = s.K;
      experts[static_cast<size_t>(e)].cols = {0};
      MakeWeight(rng, experts[static_cast<size_t>(e)]);
      wq_h[static_cast<size_t>(e)] = static_cast<uint64_t>(P(experts[static_cast<size_t>(e)].wq.data()));
      ws_h[static_cast<size_t>(e)] = static_cast<uint64_t>(P(experts[static_cast<size_t>(e)].ws.data()));
    }
    DeviceBuffer<uint64_t> wq_ptrs(wq_h.size()), ws_ptrs(ws_h.size());
    wq_ptrs.CopyFromHost(wq_h);
    ws_ptrs.CopyFromHost(ws_h);

    std::vector<uint16_t> a_h(static_cast<size_t>(n_pairs) * s.K);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : a_h) v = FloatToBf16Rne(std::clamp(nd(rng), -4.0f, 4.0f));
    DeviceBuffer<uint16_t> a_d(a_h.size());
    a_d.CopyFromHost(a_h);

    DeviceBuffer<uint16_t> c_grp(static_cast<size_t>(n_pairs) * s.N), c_ref(static_cast<size_t>(n_pairs) * s.N);
    const int fsk = FallbackSk(s.K, 64, s.N);
    std::vector<Geo> geos = {Fallback(64, s.N, fsk)};
    if (!quick) {
      for (int sk : {1, 2, 4, 8}) {
        for (int npw : {1, 4}) {
          Geo g{1, sk, 1, npw, 0};
          if (Legal(g, s.K)) geos.push_back(g);
        }
      }
    }
    for (const Geo& g : geos) {
      R4DX_HIP_CHECK(hipMemset(c_grp.data(), 0x5A, c_grp.bytes()));
      R4DX_HIP_CHECK(hipMemset(c_ref.data(), 0x5A, c_ref.bytes()));
      r4dx_gemm_mxfp8_moe_prefill(P(a_d.data()), P(wq_ptrs.data()), P(ws_ptrs.data()), P(off_d.data()), P(c_grp.data()),
                                  n_pairs, n_experts, s.K, s.N, g.WV, g.SK, g.NPW, g.NT, 0);
      for (int e = 0; e < n_experts; ++e) {
        const int r0 = off_h[static_cast<size_t>(e)], cnt = counts[static_cast<size_t>(e)];
        if (cnt == 0) continue;
        const int64_t ap = P(a_d.data() + static_cast<size_t>(r0) * s.K);
        const int64_t cp = P(c_ref.data() + static_cast<size_t>(r0) * s.N);
        r4dx_gemm_mxfp8_nt(ap, P(experts[static_cast<size_t>(e)].wq.data()), P(experts[static_cast<size_t>(e)].ws.data()),
                           cp, cnt, s.K, s.N, g.WV, g.SK, std::min(4, (cnt + 15) / 16), g.NPW, g.NT, 0);
      }
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      const auto got = c_grp.CopyToHost();
      const auto ref = c_ref.CopyToHost();
      Check(got == ref, std::string("moe_prefill ") + s.name + " [" + std::to_string(s.N) + "x" + std::to_string(s.K) +
                            "] pairs=" + std::to_string(n_pairs) + " " + GeoStr(g) + ": bit-identical to per-expert nt");
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  try {
    R4DX_HIP_CHECK(hipSetDevice(0));
    TestIllegal();
    TestEveryCode();
    TestHandActivations();
    std::mt19937 rng(20261122);
    const std::vector<int> Ms =
        quick ? std::vector<int>{1, 17, 256} : std::vector<int>{1, 2, 3, 8, 16, 17, 63, 64, 65, 255, 256};
    struct S {
      const char* name;
      int N, K;
    };
    const S shapes[] = {
        {"attn.q/gate", 3072, 2304},   {"attn.k/v", 1024, 2304},     {"attn.o", 2304, 3072},
        {"attn.rel", 384, 2304},       {"expert.gate_up", 2304, 2304}, {"expert.down", 2304, 1152},
        {"tp2.col1152", 1152, 2304},   {"tp2.expert.down576", 2304, 576},
    };
    for (const auto& s : shapes) TestProblem(rng, s.name, s.N, s.K, Ms, quick);
    if (!quick) TestProblem(rng, "lm_head (200058 rounded to 200064)", 200064, 2304, {1}, false);
    TestMoeDecode(rng, quick);
    TestMoePrefill(rng, quick);
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 1;
  }
  std::printf("test_mxfp8_gemm: %d checks, %d failed -> %s\n", g_checks, g_fails, g_ok ? "PASS" : "FAIL");
  return g_ok ? 0 : 1;
}
