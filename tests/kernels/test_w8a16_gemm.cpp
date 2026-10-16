// tests/kernels/test_w8a16_gemm.cpp -- r4dx_gemm_w8a16_nt_m64 (docs/w8a16.md 7.2) against w8a16_ref.hpp.
//
// At the real LFM2 shapes ([N, K]: expert gate_up 3072x2048, expert down 2048x1536, conv in_proj 6144x2048,
// conv out_proj / attn q / o 2048x2048, attn k / v 512x2048, dense w1 / w3 11776x2048, fused dense gate_up
// 23552x2048, dense down 2048x11776, lm_head 128000x2048 (sampled columns), the TP = 2 rank shapes 2048x768 and
// 2048x5888) plus an odd one (592x384: 37 row tiles, 3 groups of 128), at G = 64 and 128:
//   * M in {1, 8, 16, 17, 64} at every legal SK (groups % SK == 0, SK*32 <= 1024), with the fallback
//     geometry of docs/w8a16.md 4.5: every sampled output within |c - ref| <= 2 bf16 ulp(|ref|) +
//     1e-6 ||a_row|| ||w_row|| of the fp64 naive GEMM; the count that equals w8a16_ref::Gemm bit for bit is
//     REPORTED (the WMMA's internal sum order is the one step the reference has to assume);
//   * EXACT: row i of every M launch == row i of the M = 64 launch at the same SK; the rows of the M = 8 and
//     M = 16 launches == M = 1 launches of those rows (A pointer offset into the middle of the matrix); two
//     runs give the same bytes; the output does not depend on WV (1, 2, 3, 4), MB (1..4), NPW (1, 4) or
//     NT (0, 1) at fixed SK (WV 3 leaves a partial last block of column tiles);
//   * activations: Gaussian rows, a row with magnitudes up to 6e4, a row of tiny (1e-5) values, a row mixing
//     1e5 (saturates to 65504) and 1e-9 (f16 zero); weights: random codes, typical biased scales (0.2..23),
//     rows of tiny / subnormal biased scales and zero-scale groups;
//   * a hand case per group that isolates single activations (1e5, -1e5, 3e38, 65504 -> +-65504 by RTZ;
//     1e-6 -> the f16 subnormal 16 * 2^-24; 1e-9 -> 0), compared bit for bit (one product per output);
//   * illegal launches throw.
// GPU test (HIP device 1 via HIP_VISIBLE_DEVICES=1 in CMakeLists.txt). WRITTEN, NOT YET RUN by its author.
// Built with /EHc-. Optional argument: --quick (G 128 only, M in {1, 17, 64}, no head).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/w8a16_gemm.h"
#include "w8a16_ref.hpp"

using namespace r4dx::core;
using namespace w8a16_ref;

namespace {

bool g_ok = true;
int g_checks = 0, g_fails = 0;
void Check(bool c, const std::string& what) {
  std::printf("%-110s %s\n", what.c_str(), c ? "ok" : "FAIL");
  std::fflush(stdout);
  ++g_checks;
  if (!c) ++g_fails;
  g_ok = g_ok && c;
}
int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

struct Geo {
  int WV, SK, MB, NPW, NT;
};
std::string GeoStr(const Geo& g) {
  return "WV" + std::to_string(g.WV) + " SK" + std::to_string(g.SK) + " MB" + std::to_string(g.MB) + " NPW" +
         std::to_string(g.NPW) + " NT" + std::to_string(g.NT);
}
bool Legal(const Geo& g, int K, int G) {
  return g.WV >= 1 && g.SK >= 1 && K % (g.SK * G) == 0 && g.WV * g.SK * 32 <= 1024 && g.MB >= 1 && g.MB <= 4 &&
         (g.NPW == 1 || g.NPW == 4) && g.WV * g.NPW * g.SK <= 64;
}
// docs/w8a16.md 4.5's fallback at a given SK.
Geo Fallback(int M, int N, int SK) {
  Geo g;
  g.SK = SK;
  g.WV = (N / 16 < 512) ? 1 : 4;
  if (g.WV * SK > 32) g.WV = std::max(1, 32 / SK);
  if (M <= 16) {
    g.MB = 1, g.NPW = 1, g.NT = 1;
  } else {
    g.MB = std::min(4, (M + 15) / 16), g.NPW = 4, g.NT = 0;
  }
  while (g.WV > 1 && g.WV * g.NPW * SK > 64) --g.WV;
  if (g.WV * g.NPW * SK > 64) g.NPW = 1;
  return g;
}

struct Problem {
  std::string name;
  int N, K, G;
  std::vector<int> cols;  // sampled output columns (all when N is small)
  DeviceBuffer<uint8_t> wq;
  DeviceBuffer<uint16_t> ws;
  Unpacked w;  // the sampled rows of the weight, read in the kernel's order
};

// Random codes and biased scales. Rows 0..15 of every 64-row band: typical scales; row 5: tiny normal
// biased scales (~1e-3); row 6: subnormal biased scales; row 7: every other group zero-scale.
void MakeWeight(std::mt19937& rng, Problem& p) {
  const int N = p.N, K = p.K, G = p.G, ng = K / G;
  std::vector<int8_t> q(static_cast<size_t>(N) * K);
  std::uniform_int_distribution<int> dq(-127, 127);
  for (auto& c : q) c = static_cast<int8_t>(dq(rng));
  std::vector<uint16_t> s(static_cast<size_t>(N) * ng);
  std::uniform_real_distribution<float> ds(0.2f, 23.0f), dt(5e-4f, 2e-3f);
  std::uniform_int_distribution<int> dsub(1, 0x3ff);
  for (int n = 0; n < N; ++n)
    for (int g = 0; g < ng; ++g) {
      uint16_t v = FloatToF16Rne(ds(rng));
      switch (n % 64) {
        case 5: v = FloatToF16Rne(dt(rng)); break;
        case 6: v = static_cast<uint16_t>(dsub(rng)); break;
        case 7: if (g % 2) v = 0; break;
        default: break;
      }
      s[static_cast<size_t>(n) * ng + g] = v;
    }
  const auto wq = PackWq(q, N, K);
  const auto ws = PackWs(s, N, K, G);
  std::vector<int8_t>().swap(q);
  p.wq = DeviceBuffer<uint8_t>(wq.size());
  p.wq.CopyFromHost(wq);
  p.ws = DeviceBuffer<uint16_t>(ws.size());
  p.ws.CopyFromHost(ws);
  p.w = UnpackRows(wq, ws, N, K, G, p.cols);
}

std::vector<int> SampleCols(std::mt19937& rng, int N) {
  if (N <= 256) {
    std::vector<int> c(N);
    for (int i = 0; i < N; ++i) c[i] = i;
    return c;
  }
  std::set<int> s;
  for (int i = 0; i < 32; ++i) s.insert(i);              // first two tiles (incl. the special rows 5..7)
  for (int i = N - 32; i < N; ++i) s.insert(i);          // last two tiles (clamped tail)
  for (int i = 64; i < 80; ++i) s.insert(i);             // a second band of special rows
  std::uniform_int_distribution<int> d(0, N - 1);
  while (s.size() < 256) s.insert(d(rng));
  return std::vector<int>(s.begin(), s.end());
}

// A: 64 rows. Row 3: Gaussian * 1e4 clipped to +-6e4, with 6e4 itself; row 4: tiny; row 9: a mix of 1e5 and
// 1e-9 spikes in Gaussian data; others Gaussian sd 1.
std::vector<uint16_t> MakeA(std::mt19937& rng, int K) {
  std::normal_distribution<float> d(0.0f, 1.0f);
  std::vector<uint16_t> a(static_cast<size_t>(64) * K);
  for (int m = 0; m < 64; ++m)
    for (int k = 0; k < K; ++k) {
      float x = d(rng);
      if (m == 3) x = std::clamp(x * 1e4f, -6e4f, 6e4f);
      if (m == 4) x *= 1e-5f;
      if (m == 9 && k % 97 == 0) x = (k % 2 ? 1e5f : -1e5f);
      if (m == 9 && k % 89 == 0) x = 1e-9f;
      a[static_cast<size_t>(m) * K + k] = FloatToBf16Rne(x);
    }
  a[static_cast<size_t>(3) * K] = FloatToBf16Rne(6e4f);
  return a;
}

std::vector<uint16_t> Run(DeviceBuffer<uint16_t>& a_d, size_t a_off_rows, Problem& p, int M,
                          const Geo& g, DeviceBuffer<uint16_t>& c_d) {
  const int64_t a = P(a_d.data()) + static_cast<int64_t>(a_off_rows * p.K * sizeof(uint16_t));
  // poison C so an unwritten element is visible
  R4DX_HIP_CHECK(hipMemset(c_d.data(), 0xff, static_cast<size_t>(M) * p.N * sizeof(uint16_t)));
  r4dx_gemm_w8a16_nt_m64(a, P(p.wq.data()), P(p.ws.data()), P(c_d.data()), M, p.K, p.N, p.G, g.WV, g.SK, g.MB,
                         g.NPW, g.NT, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> out(static_cast<size_t>(M) * p.N);
  R4DX_HIP_CHECK(hipMemcpy(out.data(), c_d.data(), out.size() * sizeof(uint16_t), hipMemcpyDeviceToHost));
  return out;
}

bool RowsEqual(const std::vector<uint16_t>& x, size_t xrow, const std::vector<uint16_t>& y, size_t yrow, int N) {
  return std::memcmp(&x[xrow * N], &y[yrow * N], static_cast<size_t>(N) * sizeof(uint16_t)) == 0;
}

void TestProblem(std::mt19937& rng, const std::string& name, int N, int K, int G, const std::vector<int>& Ms) {
  Problem p;
  p.name = name;
  p.N = N, p.K = K, p.G = G;
  p.cols = SampleCols(rng, N);
  MakeWeight(rng, p);
  const auto a = MakeA(rng, K);
  DeviceBuffer<uint16_t> a_d(a.size());
  a_d.CopyFromHost(a);
  DeviceBuffer<uint16_t> c_d(static_cast<size_t>(64) * N);
  const NaiveOut nv = GemmNaive(a, p.w, 64, K, G);
  const int R = p.w.rows;
  const std::string tag = name + " [" + std::to_string(N) + "x" + std::to_string(K) + "] G" + std::to_string(G);

  const int groups = K / G;
  for (int SK = 1; SK <= 32; ++SK) {
    if (groups % SK || SK * 32 > 1024) continue;
    const auto ref = Gemm(a, p.w, 64, K, G, SK);
    const std::string tsk = tag + " SK" + std::to_string(SK);
    // M = 64 at the fallback geometry: the anchor of the exact checks
    const Geo g64 = Fallback(64, N, SK);
    const auto c64 = Run(a_d, 0, p, 64, g64, c_d);
    const auto c64b = Run(a_d, 0, p, 64, g64, c_d);
    Check(c64 == c64b, tsk + " M64: two runs identical");
    for (int M : Ms) {
      const Geo g = Fallback(M, N, SK);
      const auto c = M == 64 ? c64 : Run(a_d, 0, p, M, g, c_d);
      size_t bad = 0, bitwise = 0, n = 0;
      double worst = 0;
      for (int m = 0; m < M; ++m)
        for (int ci = 0; ci < R; ++ci) {
          const double got = Bf16ToFloat(c[static_cast<size_t>(m) * N + p.cols[ci]]);
          const size_t ri = static_cast<size_t>(m) * R + ci;
          const bool okb = WithinBound(got, nv.c[ri], nv.a_norm[m], nv.w_norm[ci]);
          bad += !okb;
          bitwise += c[static_cast<size_t>(m) * N + p.cols[ci]] == ref[ri];
          worst = std::max(worst, std::fabs(got - nv.c[ri]) / (2.0 * Bf16Ulp(nv.c[ri]) +
                                                                 1e-6 * nv.a_norm[m] * nv.w_norm[ci]));
          ++n;
        }
      char buf[160];
      std::snprintf(buf, sizeof buf, " M%d %s: %zu/%zu within bound (worst %.3f of it), %zu/%zu == ref bitwise", M,
                    GeoStr(g).c_str(), n - bad, n, worst, bitwise, n);
      Check(bad == 0, tsk + buf);
      bool rows = true;
      for (int m = 0; m < M; ++m) rows &= RowsEqual(c, m, c64, m, N);
      Check(rows, tsk + " M" + std::to_string(M) + ": rows == the M64 launch's rows (exact)");
    }
    // M = 1 launches of rows of the M = 8 / 16 launches, A offset into the matrix
    for (int M : {8, 16}) {
      if (std::find(Ms.begin(), Ms.end(), M) == Ms.end()) continue;
      const size_t off = M == 8 ? 40 : 7;  // rows 40..47 / 7..22 of A
      const auto cm = Run(a_d, off, p, M, Fallback(M, N, SK), c_d);
      bool ok = true;
      for (int m = 0; m < M; ++m) {
        const auto c1 = Run(a_d, off + m, p, 1, Fallback(1, N, SK), c_d);
        ok &= RowsEqual(cm, m, c1, 0, N) && RowsEqual(cm, m, c64, off + m, N);
      }
      Check(ok, tsk + " M" + std::to_string(M) + " (A row offset " + std::to_string(off) +
                    "): every row == its M1 launch (exact)");
    }
    // geometry independence at fixed SK (M = 64 and M = 17)
    for (int M : {64, 17}) {
      const auto& anchor = c64;
      size_t tried = 0, differ = 0;
      std::string first_bad;
      for (int WV : {1, 2, 3, 4})
        for (int MB = 1; MB <= 4; ++MB)
          for (int NPW : {1, 4})
            for (int NT : {0, 1}) {
              const Geo g{WV, SK, MB, NPW, NT};
              if (!Legal(g, K, G)) continue;
              const auto c = Run(a_d, 0, p, M, g, c_d);
              ++tried;
              bool same = true;
              for (int m = 0; m < M; ++m) same &= RowsEqual(c, m, anchor, m, N);
              if (!same && first_bad.empty()) first_bad = GeoStr(g);
              differ += !same;
            }
      Check(differ == 0, tsk + " M" + std::to_string(M) + ": " + std::to_string(tried) +
                             " geometries (WV 1-4, MB 1-4, NPW 1/4, NT 0/1) bit-identical" +
                             (first_bad.empty() ? "" : " (first differing " + first_bad + ")"));
    }
  }
}

void TestHandActivations(int G) {
  // column n reads A[0, n] alone (code 1 at k = n, scale 1 = biased 0x6c00): out[n] = bf16(f16_rtz(A[n]))
  const int N = 16, K = 128;
  std::vector<int8_t> q(static_cast<size_t>(N) * K, 0);
  for (int n = 0; n < N; ++n) q[static_cast<size_t>(n) * K + n] = 1;
  std::vector<uint16_t> s(static_cast<size_t>(N) * (K / G), 0x6c00);
  const auto wq = PackWq(q, N, K);
  const auto ws = PackWs(s, N, K, G);
  std::vector<float> vals = {1e5f, -1e5f, 3e38f, 65504.0f, 1e-6f, -1e-6f, 1e-9f, 1.0f,
                             -0.0f, 65280.0f, 6e4f, 6.1e-5f, 3.0f, -7.5f, 1e-4f, 2.5e4f};
  std::vector<uint16_t> a(K, 0);
  for (int n = 0; n < N; ++n) a[n] = FloatToBf16Rne(vals[n]);
  DeviceBuffer<uint8_t> wq_d(wq.size());
  wq_d.CopyFromHost(wq);
  DeviceBuffer<uint16_t> ws_d(ws.size()), a_d(a.size()), c_d(N);
  ws_d.CopyFromHost(ws);
  a_d.CopyFromHost(a);
  r4dx_gemm_w8a16_nt_m64(P(a_d.data()), P(wq_d.data()), P(ws_d.data()), P(c_d.data()), 1, K, N, G, 1, 1, 1, 1, 1, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const auto c = c_d.CopyToHost();
  const auto ref = Gemm(a, UnpackRows(wq, ws, N, K, G), 1, K, G, 1);
  Check(c == ref, "hand G" + std::to_string(G) + ": single activations through the kernel == reference bitwise");
  Check(c[0] == FloatToBf16Rne(65504.0f) && c[1] == FloatToBf16Rne(-65504.0f) && c[2] == FloatToBf16Rne(65504.0f) &&
            c[3] == FloatToBf16Rne(65504.0f),
        "hand G" + std::to_string(G) + ": 1e5, -1e5, 3e38, bf16(65504)=65536 -> +-65504 (RTZ saturation, finite)");
  Check(Bf16ToFloat(c[4]) == F16ToFloat(0x0010) && Bf16ToFloat(c[5]) == -F16ToFloat(0x0010),
        "hand G" + std::to_string(G) + ": +-1e-6 -> +-16 * 2^-24 (f16 subnormal, RTZ)");
  Check(Bf16ToFloat(c[6]) == 0.0f, "hand G" + std::to_string(G) + ": 1e-9 -> 0");
}

void TestIllegal() {
  DeviceBuffer<uint16_t> a(64 * 256), ws(64 * 4), c(64 * 64);
  DeviceBuffer<uint8_t> wq(64 * 256);
  auto throws = [&](int M, int K, int N, int G, int WV, int SK, int MB, int NPW, int NT, bool null_a = false) {
    try {
      r4dx_gemm_w8a16_nt_m64(null_a ? 0 : P(a.data()), P(wq.data()), P(ws.data()), P(c.data()), M, K, N, G, WV, SK,
                             MB, NPW, NT, 0);
    } catch (const std::exception&) {
      return true;
    }
    return false;
  };
  Check(throws(0, 256, 64, 128, 1, 1, 1, 1, 1), "illegal: M = 0 throws");
  Check(throws(65, 256, 64, 128, 1, 1, 1, 1, 1), "illegal: M = 65 throws");
  Check(throws(1, 256, 24, 128, 1, 1, 1, 1, 1), "illegal: N = 24 (not % 16) throws");
  Check(throws(1, 256, 64, 128, 1, 4, 1, 1, 1), "illegal: K 256 % (SK 4 * G 128) throws");
  Check(throws(1, 192, 64, 128, 1, 1, 1, 1, 1), "illegal: K 192 % G 128 throws");
  Check(throws(1, 256, 64, 96, 1, 1, 1, 1, 1), "illegal: group 96 throws");
  Check(throws(1, 256, 64, 32, 1, 1, 1, 1, 1), "illegal: group 32 throws");
  Check(throws(1, 256, 64, 64, 9, 4, 1, 1, 1), "illegal: WV*SK*32 > 1024 throws");
  Check(throws(1, 256, 64, 64, 0, 1, 1, 1, 1), "illegal: WV = 0 throws");
  Check(throws(1, 256, 64, 64, 1, 0, 1, 1, 1), "illegal: SK = 0 throws");
  Check(throws(1, 256, 64, 64, 1, 1, 5, 1, 1), "illegal: MB = 5 throws");
  Check(throws(1, 256, 64, 64, 1, 1, 0, 1, 1), "illegal: MB = 0 throws");
  Check(throws(1, 256, 64, 64, 1, 1, 1, 2, 1), "illegal: NPW = 2 throws");
  Check(throws(1, 4096, 64, 64, 8, 4, 1, 4, 1), "illegal: LDS WV 8 * NPW 4 * SK 4 = 128 KiB throws");
  Check(throws(1, 256, 64, 64, 1, 1, 1, 1, 1, true), "illegal: null A throws");
  Check(!throws(64, 256, 64, 64, 1, 4, 4, 4, 0), "legal: M 64 K 256 G 64 SK 4 MB 4 NPW 4 runs");
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  Check(r4dx_gemm_w8a16_has_group(64) && r4dx_gemm_w8a16_has_group(128) && !r4dx_gemm_w8a16_has_group(32) &&
            !r4dx_gemm_w8a16_has_group(256),
        "exports: has_group 64/128 only");
  Check(r4dx_gemm_w8a16_scale_log2_bias() == kScaleLog2Bias && r4dx_gemm_w8a16_max_m() == kMaxM,
        "exports: scale_log2_bias 12, max_m 64");
}

}  // namespace

int main(int argc, char** argv) {
  const bool quick = argc > 1 && std::string(argv[1]) == "--quick";
  try {
    R4DX_HIP_CHECK(hipSetDevice(0));
    TestIllegal();
    TestHandActivations(64);
    TestHandActivations(128);
    std::mt19937 rng(20261016);
    const std::vector<int> Ms = quick ? std::vector<int>{1, 17, 64} : std::vector<int>{1, 8, 16, 17, 64};
    struct S {
      const char* name;
      int N, K;
      bool big;
    };
    const S shapes[] = {
        {"expert.gate_up", 3072, 2048, false}, {"expert.down", 2048, 1536, false},
        {"conv.in_proj", 6144, 2048, false},   {"conv.out_proj/attn.q/o", 2048, 2048, false},
        {"attn.k/v", 512, 2048, false},        {"dense.w1/w3", 11776, 2048, false},
        {"dense.gate_up", 23552, 2048, false}, {"dense.down", 2048, 11776, false},
        {"tp2.row768", 2048, 768, false},      {"tp2.row5888", 2048, 5888, false},
        {"odd", 592, 384, false},              {"lm_head", 128000, 2048, true},
    };
    for (int G : {128, 64}) {
      if (quick && G == 64) continue;
      for (const auto& s : shapes) {
        if (quick && s.big) continue;
        TestProblem(rng, s.name, s.N, s.K, G, Ms);
      }
    }
  } catch (const std::exception& e) {
    std::printf("EXCEPTION: %s\n", e.what());
    return 1;
  }
  std::printf("test_w8a16_gemm: %d checks, %d failed -> %s\n", g_checks, g_fails, g_ok ? "PASS" : "FAIL");
  return g_ok ? 0 : 1;
}
