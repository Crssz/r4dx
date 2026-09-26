// tests/kernels/test_trellis_decode.cpp -- libr4d's trellis decode, bit for bit
// (docs/trellis-kernel.md 4.2-4.5 and 6, milestone M1):
//
//   trellis_ref.hpp             the C++ CPU decode, against the Python goldens (trellis_golden.py ->
//                               tests/kernels/golden/trellis/): random tiles at KB = 4 and 5 and the
//                               real L03 attn.k / L10 mlp.down / L07 mlp.gate_up blocks, plus the
//                               oracle-layout -> pair-grid regrid of the real words;
//   r4d_trellis_reconstruct_f16 the whole decoded Q against decode_words (goldens) and the CPU decode
//                               (random shapes), KB = 4 and 5;
//   r4d_gemm_trellis_nt_m64_raw one-hot A rows return rows of Q bit for bit:
//                               - every instantiated (NP, U, MT) x NT, split and unsplit, over every k
//                                 of a 1024 x 512 weight (even step counts) and of a 1152 x 512 one
//                                 (9 steps: the loop, then its odd tail) (full coverage);
//                               - the whole legal tuning space (WV, SK, SKG, MT, NP, U, NT) on a
//                                 K = 8192 weight, with both A parts (each with both NT values),
//                                 rows >= M left untouched, and the tickets back at zero after every
//                                 call;
//                               - the real gate_up block as a 2-part linear (n_split);
//                               plus, on random (not one-hot) A at K = 5120: the M = 1 rows against
//                               an fp64 reference, row identity for M <= 16 (row r of an M-row call
//                               == the M = 1 call on that row) and repeat determinism of a split
//                               group; and the host's precondition throws (hence /EHc-).
// The fp32 C of a one-hot row is exact whatever the summation order (every other product is zero),
// so these checks hold the lane map, the pair grid, the in-block and cross-block reductions and the
// part selection to bit-exactness. The full linear (transforms, FWHT epilogue) is M2's
// test_trellis_gemm.
//
// GPU test: HIP device 1 via HIP_VISIBLE_DEVICES=1 (tests/kernels/CMakeLists.txt). The golden
// comparisons run after everything else and return 77 (SKIPPED) when the golden directory is absent,
// only if every other check passed.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "npy_fixture.hpp"
#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
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

std::vector<uint32_t> RandomWords(std::mt19937_64& rng, size_t n) {
  std::vector<uint32_t> w(n);
  for (auto& x : w) x = static_cast<uint32_t>(rng());
  return w;
}

size_t CountDiff(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  if (a.size() != b.size()) return a.size() + b.size();
  size_t n = 0;
  for (size_t i = 0; i < a.size(); ++i) n += a[i] != b[i];
  return n;
}

std::vector<uint16_t> Reconstruct(const std::vector<uint32_t>& grid, int K, int N, int KB) {
  DeviceBuffer<uint32_t> w_d(grid.size());
  DeviceBuffer<uint16_t> q_d(static_cast<size_t>(K) * N);
  w_d.CopyFromHost(grid);
  R4DX_HIP_CHECK(hipMemset(q_d.data(), 0xFF, q_d.bytes()));
  r4d_trellis_reconstruct_f16(P(w_d.data()), P(q_d.data()), K, N, KB, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  return q_d.CopyToHost();
}

// ---- the _raw GEMM ------------------------------------------------------------------------------

struct Tuning {
  int WV, SK, MT, NP, SKG, U, NT;
  int Wc() const { return WV * NP * 32; }
  bool Split() const { return SKG > 1 || Wc() < 128; }
  std::string Str() const {
    char b[96];
    std::snprintf(b, sizeof b, "WV%d SK%d MT%d NP%d SKG%d U%d NT%d", WV, SK, MT, NP, SKG, U, NT);
    return b;
  }
};

// One weight on the device plus the buffers a call needs, for M up to 64.
struct Gemm {
  int K, N;
  std::vector<uint16_t> q;               // the CPU decode, [K][N] f16 bits
  DeviceBuffer<uint32_t> w;
  DeviceBuffer<uint16_t> a0, a1;         // [64][K] f16
  DeviceBuffer<float> c;                 // [64][N]
  DeviceBuffer<float> ws;                // [8][64][N]
  DeviceBuffer<uint32_t> tickets;        // [N/128]
  Gemm(const std::vector<uint32_t>& grid, std::vector<uint16_t> q_, int K_, int N_)
      : K(K_), N(N_), q(std::move(q_)), w(grid.size()), a0(static_cast<size_t>(64) * K_),
        a1(static_cast<size_t>(64) * K_), c(static_cast<size_t>(64) * N_),
        ws(static_cast<size_t>(8) * 64 * N_), tickets(static_cast<size_t>(N_ / 128)) {
    w.CopyFromHost(grid);
    tickets.Zero();
  }
  void Run(const Tuning& t, int M, int n_split, bool two_parts) {
    r4d_gemm_trellis_nt_m64_raw(P(a0.data()), two_parts ? P(a1.data()) : 0, n_split, P(w.data()),
                                P(c.data()), P(ws.data()), P(tickets.data()), M, K, N, 4, t.WV, t.SK,
                                t.MT, t.NP, t.SKG, t.U, t.NT, 0, 0);
  }
};

constexpr uint16_t kOneF16 = 0x3C00;
constexpr uint32_t kSentinel = 0x7FC0DEADu;   // a NaN no sum can produce

// Row m of part p is one-hot at k = ks[p][m].
std::vector<uint16_t> OneHotA(int K, const std::vector<int>& ks) {
  std::vector<uint16_t> a(static_cast<size_t>(64) * K, 0);
  for (size_t m = 0; m < ks.size(); ++m) a[m * K + ks[m]] = kOneF16;
  return a;
}

// Runs one call and checks C against Q: rows < M exact, rows >= M untouched, tickets zero again.
// Returns the number of bad elements.
size_t RunOneHotCall(Gemm& g, const Tuning& t, int M, const std::vector<int>& k0,
                     const std::vector<int>& k1, bool two_parts, int n_split) {
  std::vector<uint32_t> sentinel(static_cast<size_t>(64) * g.N, kSentinel);
  R4DX_HIP_CHECK(hipMemcpy(g.c.data(), sentinel.data(), sentinel.size() * 4, hipMemcpyHostToDevice));
  g.Run(t, M, n_split, two_parts);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<float> c = g.c.CopyToHost();
  size_t bad = 0;
  for (int m = 0; m < 64; ++m)
    for (int n = 0; n < g.N; ++n) {
      uint32_t got;
      std::memcpy(&got, &c[static_cast<size_t>(m) * g.N + n], 4);
      uint32_t want = kSentinel;
      if (m < M) {
        const int k = (two_parts && n >= n_split) ? k1[m] : k0[m];
        const float v = F16ToFloat(g.q[static_cast<size_t>(k) * g.N + n]);
        std::memcpy(&want, &v, 4);
      }
      bad += got != want;
    }
  if (t.Split()) {
    std::vector<uint32_t> tk = g.tickets.CopyToHost();
    for (uint32_t x : tk) bad += x != 0;
  }
  return bad;
}

// Full coverage: every k of the weight, for one tuning, through ceil(K / M) one-hot calls.
void FullCoverage(Gemm& g, const Tuning& t, bool two_parts, int n_split, const std::string& label) {
  const int M = t.Split() ? 16 * t.MT : 64;
  size_t bad = 0;
  for (int kb = 0; kb < g.K; kb += M) {
    std::vector<int> k0(M), k1(M);
    for (int m = 0; m < M; ++m) {
      k0[m] = (kb + m) % g.K;
      k1[m] = (kb + m + g.K / 2 + 5) % g.K;   // a different row of Q for the second part
    }
    g.a0.CopyFromHost(OneHotA(g.K, k0));
    g.a1.CopyFromHost(OneHotA(g.K, k1));
    bad += RunOneHotCall(g, t, M, k0, k1, two_parts, n_split);
  }
  Check(bad == 0, label + " " + t.Str() + ": " + std::to_string(bad) + " elements differ from Q");
}

// The whole knob space of docs/trellis-kernel.md 4.4; the host decides what is legal on this shape
// (a rejected tuning throws before launching), and every tuning it accepts must be bit-exact.
void TestRawSweep(std::mt19937_64& rng) {
  const int K = 8192, N = 512, KB = 4;
  std::vector<uint32_t> words = RandomWords(rng, static_cast<size_t>(K / 16) * (N / 16) * 8 * KB);
  std::vector<uint32_t> grid = trellis_ref::ToPairGrid(words, K, N, KB);
  Gemm g(grid, trellis_ref::DecodePairGrid(grid, K, N, KB), K, N);

  // Eight one-hot activations per part, uploaded once and cycled: rows spread over the whole K so
  // every K slice of every split sees some.
  constexpr int kVariants = 8;
  std::vector<std::vector<int>> ks(kVariants, std::vector<int>(64));
  std::uniform_int_distribution<int> kd(0, K - 1);
  for (auto& v : ks)
    for (auto& k : v) k = kd(rng);
  std::vector<DeviceBuffer<uint16_t>> a_d;
  for (int v = 0; v < kVariants; ++v) {
    a_d.emplace_back(static_cast<size_t>(64) * K);
    a_d.back().CopyFromHost(OneHotA(K, ks[v]));
  }

  const int ms[] = {1, 3, 8, 16, 17, 33, 64};
  int legal = 0, illegal = 0, idx = 0;
  size_t bad_total = 0;
  for (int WV : {1, 2, 4})
    for (int NP : {1, 2, 4})
      for (int SK : {1, 2, 4, 8, 16})
        for (int SKG : {1, 2, 4, 8})
          for (int MT : {1, 2, 3, 4})
            for (int U : {1, 2, 4})
              for (int NT : {0, 1}) {
                const Tuning t{WV, SK, MT, NP, SKG, U, NT};
                int M = ms[idx % 7];
                if (t.Split()) M = std::min(M, 16 * MT);
                // idx & 1 is NT (the innermost loop); the part count alternates with idx >> 1 as
                // well, so both part counts meet both NT values.
                const bool two = (((idx >> 1) ^ idx) & 1) != 0;
                const int n_split = two ? N / 2 : N;
                const int v0 = idx % kVariants, v1 = (idx + 3) % kVariants;
                ++idx;
                std::vector<uint32_t> sentinel(static_cast<size_t>(64) * N, kSentinel);
                R4DX_HIP_CHECK(hipMemcpy(g.c.data(), sentinel.data(), sentinel.size() * 4,
                                         hipMemcpyHostToDevice));
                bool threw = false;
                try {
                  r4d_gemm_trellis_nt_m64_raw(P(a_d[v0].data()), two ? P(a_d[v1].data()) : 0,
                                              n_split, P(g.w.data()), P(g.c.data()), P(g.ws.data()),
                                              P(g.tickets.data()), M, K, N, KB, WV, SK, MT, NP, SKG,
                                              U, NT, 0, 0);
                } catch (const std::exception&) {
                  threw = true;
                }
                if (threw) {
                  ++illegal;
                  continue;
                }
                ++legal;
                R4DX_HIP_CHECK(hipDeviceSynchronize());
                std::vector<float> c = g.c.CopyToHost();
                size_t bad = 0;
                for (int m = 0; m < 64; ++m)
                  for (int n = 0; n < N; ++n) {
                    uint32_t got, want = kSentinel;
                    std::memcpy(&got, &c[static_cast<size_t>(m) * N + n], 4);
                    if (m < M) {
                      const int k = (two && n >= n_split) ? ks[v1][m] : ks[v0][m];
                      const float val = F16ToFloat(g.q[static_cast<size_t>(k) * N + n]);
                      std::memcpy(&want, &val, 4);
                    }
                    bad += got != want;
                  }
                if (t.Split()) {
                  for (uint32_t x : g.tickets.CopyToHost()) bad += x != 0;
                }
                bad_total += bad;
                Check(bad == 0, "raw sweep " + t.Str() + " M=" + std::to_string(M) +
                                    (two ? " P=2" : " P=1") + ": " + std::to_string(bad) +
                                    " elements wrong");
              }
  std::printf("  raw sweep K=%d N=%d: %d legal tunings run bit-exact%s, %d rejected by the host\n", K,
              N, legal, bad_total ? " (NOT all)" : "", illegal);
  Check(legal > 1000, "raw sweep: suspiciously few legal tunings (" + std::to_string(legal) + ")");
}

// Two weights, so every instantiation runs both ends of its K loop:
//   K = 1024 (64 k-tiles): SK*SKG*U = 2U, 32 / U steps -- even, so the double-buffered loop ends on
//     a full iteration (and at MT <= 2 re-reads its last step);
//   K = 1152 (72 = 8 * 9 k-tiles): SK*SKG*U = 8, 9 steps -- the loop runs 4 iterations and then the
//     odd tail. Every production K has an odd factor (5120 = 320 k-tiles, 6144 = 384, 17408 = 1088),
//     so e.g. SK*SKG*U = 64 at K = 5120 gives 5 steps: this path.
void TestRawFullCoverage(std::mt19937_64& rng) {
  const int N = 512, KB = 4;
  auto weight = [&](int K) {
    std::vector<uint32_t> words = RandomWords(rng, static_cast<size_t>(K / 16) * (N / 16) * 8 * KB);
    std::vector<uint32_t> grid = trellis_ref::ToPairGrid(words, K, N, KB);
    return std::make_unique<Gemm>(grid, trellis_ref::DecodePairGrid(grid, K, N, KB), K, N);
  };
  std::unique_ptr<Gemm> even = weight(1024), odd = weight(1152);
  int kernels = 0;
  for (int NP : {1, 2, 4})
    for (int U : {1, 2, 4})
      for (int MT = 1; MT <= 4; ++MT)
        for (int NT : {0, 1}) {
          // Unsplit: a 128-column block (WV = 4 / NP). Split: the narrowest block of this NP (WV 1)
          // with SKG 2, so both split kinds meet in one ticket. At K = 1024 SK is 2; at K = 1152 it
          // makes SK*SKG*U = 8.
          const int WV = 4 / NP;
          const Tuning unsplit{WV, 2, MT, NP, 1, U, NT}, split{1, 2, MT, NP, 2, U, NT};
          const Tuning odd_unsplit{WV, 8 / U, MT, NP, 1, U, NT}, odd_split{1, 4 / U, MT, NP, 2, U, NT};
          if (Throws([&] { even->Run(unsplit, 1, N, false); })) continue;
          R4DX_HIP_CHECK(hipDeviceSynchronize());
          ++kernels;
          FullCoverage(*even, unsplit, (NT + MT) % 2 == 1, N / 2, "full coverage K=1024");
          FullCoverage(*even, split, (NT + MT) % 2 == 0, N / 2, "full coverage K=1024");
          FullCoverage(*odd, odd_unsplit, (NT + MT) % 2 == 0, N / 2, "full coverage K=1152 (9 steps)");
          FullCoverage(*odd, odd_split, (NT + MT) % 2 == 1, N / 2, "full coverage K=1152 (9 steps)");
        }
  std::printf("  full coverage N=%d, K=1024 (even step counts) and K=1152 (9 steps, odd tail): %d "
              "instantiations x {unsplit, split}\n", N, kernels);
  Check(kernels == 50, "full coverage: expected 50 instantiated kernels, ran " +
                           std::to_string(kernels));
}

// On dense random A at a production shape (K = 5120: 320 k-tiles, so SK*SKG*U = 64 runs 5 steps,
// the odd tail), where sums of many products make the order matter:
//   - the M = 1 rows against an fp64 dot product with the CPU decode, within the fp32 summation
//     bound 2 n 2^-24 sum|a q| (n = K/16 + 16 + SK + SKG additions per element, generous);
//   - row identity (rows of M = 2..16 calls equal their M = 1 calls) and 20-repeat determinism.
void TestRawOrder(std::mt19937_64& rng) {
  const int K = 5120, N = 1024, KB = 4;
  std::vector<uint32_t> words = RandomWords(rng, static_cast<size_t>(K / 16) * (N / 16) * 8 * KB);
  std::vector<uint32_t> grid = trellis_ref::ToPairGrid(words, K, N, KB);
  Gemm g(grid, trellis_ref::DecodePairGrid(grid, K, N, KB), K, N);
  std::uniform_real_distribution<float> ud(-1.f, 1.f);
  std::vector<uint16_t> a(static_cast<size_t>(64) * K);
  for (auto& x : a) x = FloatToF16(ud(rng));
  g.a0.CopyFromHost(a);
  g.a1.CopyFromHost(a);
  // fp64 reference of rows 0..15: ref[m][n] = sum_k a[m][k] q[k][n], and sum_k |a q| for the bound.
  std::vector<double> ref(static_cast<size_t>(16) * N, 0.0), mag(static_cast<size_t>(16) * N, 0.0);
  {
    std::vector<float> qf(static_cast<size_t>(K) * N);
    for (size_t i = 0; i < qf.size(); ++i) qf[i] = F16ToFloat(g.q[i]);
    for (int m = 0; m < 16; ++m)
      for (int k = 0; k < K; ++k) {
        const double av = F16ToFloat(a[static_cast<size_t>(m) * K + k]);
        const float* qr = &qf[static_cast<size_t>(k) * N];
        double* rr = &ref[static_cast<size_t>(m) * N];
        double* mr = &mag[static_cast<size_t>(m) * N];
        for (int n = 0; n < N; ++n) {
          rr[n] += av * qr[n];
          mr[n] += std::fabs(av * qr[n]);
        }
      }
  }
  double worst_ratio = 0.0;
  const Tuning tunings[] = {{4, 2, 1, 1, 1, 2, 1}, {1, 4, 1, 2, 8, 2, 1}, {1, 16, 1, 1, 4, 1, 0},
                            {2, 8, 1, 4, 2, 1, 1}, {4, 1, 1, 2, 1, 4, 0}};
  for (const Tuning& t : tunings) {
    // M = 1 on each row alone: row m of A copied to row 0 of a scratch A.
    std::vector<std::vector<float>> single(16);
    DeviceBuffer<uint16_t> a1row(static_cast<size_t>(64) * K);
    for (int m = 0; m < 16; ++m) {
      R4DX_HIP_CHECK(hipMemcpy(a1row.data(), g.a0.data() + static_cast<size_t>(m) * K,
                               static_cast<size_t>(K) * 2, hipMemcpyDeviceToDevice));
      r4d_gemm_trellis_nt_m64_raw(P(a1row.data()), 0, N, P(g.w.data()), P(g.c.data()),
                                  P(g.ws.data()), P(g.tickets.data()), 1, K, N, KB, t.WV, t.SK, t.MT,
                                  t.NP, t.SKG, t.U, t.NT, 0, 0);
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      std::vector<float> c = g.c.CopyToHost();
      single[m].assign(c.begin(), c.begin() + N);
    }
    const double n_adds = K / 16 + 16 + t.SK + t.SKG;
    size_t off = 0;
    for (int m = 0; m < 16; ++m)
      for (int n = 0; n < N; ++n) {
        const size_t i = static_cast<size_t>(m) * N + n;
        const double err = std::fabs(static_cast<double>(single[m][n]) - ref[i]);
        const double bound = 2.0 * n_adds * std::ldexp(1.0, -24) * mag[i] + 1e-30;
        worst_ratio = std::max(worst_ratio, err / bound);
        off += err > bound;
      }
    Check(off == 0, "fp64 reference " + t.Str() + ": " + std::to_string(off) +
                        " of the M=1 rows' values outside the fp32 summation bound");
    for (int M : {2, 5, 8, 16}) {
      g.Run(t, M, N, false);
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      std::vector<float> c = g.c.CopyToHost();
      size_t bad = 0;
      for (int m = 0; m < M; ++m)
        bad += std::memcmp(&c[static_cast<size_t>(m) * N], single[m].data(), N * 4) != 0;
      Check(bad == 0, "row identity " + t.Str() + " M=" + std::to_string(M) + ": " +
                          std::to_string(bad) + " rows differ from their M=1 call");
    }
    // Repeats of the same call give the same bytes (the split ones exercise the ticket path).
    g.Run(t, 16, N, false);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const std::vector<float> first = g.c.CopyToHost();
    int differ = 0;
    for (int r = 0; r < 20; ++r) {
      g.Run(t, 16, N, false);
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      differ += std::memcmp(g.c.CopyToHost().data(), first.data(), first.size() * 4) != 0;
    }
    Check(differ == 0, "determinism " + t.Str() + ": " + std::to_string(differ) + " of 20 repeats differ");
  }
  std::printf("  K=%d N=%d dense A, %zu tunings: M=1 rows vs fp64 (worst error %.3f of the bound), row "
              "identity (M 2/5/8/16 vs M=1), 20-repeat determinism\n",
              K, N, sizeof(tunings) / sizeof(tunings[0]), worst_ratio);
}

void TestPreconditions() {
  const int K = 1024, N = 512;
  // Buffers are never touched: every call below must throw before launching.
  auto call = [&](int M, int K_, int N_, int n_split, int KB, int WV, int SK, int MT, int NP, int SKG,
                  int U, int64_t ws) {
    r4d_gemm_trellis_nt_m64_raw(16, 0, n_split, 16, 16, ws, ws, M, K_, N_, KB, WV, SK, MT, NP, SKG, U,
                                0, 0, 0);
  };
  Check(Throws([&] { call(0, K, N, N, 4, 4, 2, 1, 1, 1, 2, 16); }), "M = 0 accepted");
  Check(Throws([&] { call(65, K, N, N, 4, 4, 2, 1, 1, 1, 2, 16); }), "M = 65 accepted");
  Check(Throws([&] { call(1, 1000, N, N, 4, 4, 2, 1, 1, 1, 2, 16); }), "K % 128 accepted");
  Check(Throws([&] { call(1, K, 500, 500, 4, 4, 2, 1, 1, 1, 2, 16); }), "N % 128 accepted");
  Check(Throws([&] { call(1, K, N, N, 5, 4, 2, 1, 1, 1, 2, 16); }), "KB = 5 accepted (not instantiated)");
  Check(Throws([&] { call(1, K, N, N, 4, 1, 2, 1, 4, 1, 4, 16); }), "NP*U = 16 accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 1, 2, 2, 4, 1, 2, 16); }), "(NP 4, U 2, MT 2) accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 4, 2, 1, 4, 1, 1, 16); }), "Wc = 512 accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 2, 16, 1, 4, 1, 1, 16); }), "SK*Wc*32 = 128 KiB of LDS accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 4, 16, 1, 1, 1, 1, 16); }), "2048 threads accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 4, 4, 1, 1, 8, 4, 16); }), "(K/16) % (SK*SKG*U) accepted");
  Check(Throws([&] { call(17, K, N, N, 4, 4, 2, 1, 1, 2, 2, 16); }), "split group with 2 row tiles at MT 1 accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 1, 2, 1, 1, 1, 2, 0); }), "split group without ws/tickets accepted");
  Check(Throws([&] { call(1, K, N, 192, 4, 4, 2, 1, 1, 1, 2, 16); }), "n_split % 128 accepted");
  Check(Throws([&] { call(1, K, N, 128, 4, 4, 2, 1, 2, 1, 2, 16); }), "n_split % Wc (256) accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 3, 2, 1, 1, 1, 2, 16); }), "WV = 3 accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 4, 2, 1, 1, 3, 2, 16); }), "SKG = 3 accepted");
  Check(Throws([&] { r4d_trellis_reconstruct_f16(16, 16, 1024, 48, 4, 0); }), "reconstruct N % 32 accepted");
  Check(Throws([&] { r4d_trellis_reconstruct_f16(16, 16, 1024, 64, 3, 0); }), "reconstruct KB = 3 accepted");
  Check(r4d_gemm_trellis_nt_m64_has_rate(4) == 1 && r4d_gemm_trellis_nt_m64_has_rate(5) == 0,
        "has_rate: 4 yes, 5 not yet");
  Check(r4d_gemm_trellis_nt_m64_max_m() == 64, "max_m 64");
  Check(r4d_gemm_trellis_nt_m64_ws_bytes(16, 5120, 8) == static_cast<size_t>(8) * 16 * 5120 * 4,
        "ws_bytes = SKG*M*N*4");
}

void TestReconstructRandom(std::mt19937_64& rng) {
  for (int KB : {4, 5}) {
    const int K = 512, N = 384;
    std::vector<uint32_t> words = RandomWords(rng, static_cast<size_t>(K / 16) * (N / 16) * 8 * KB);
    std::vector<uint32_t> grid = trellis_ref::ToPairGrid(words, K, N, KB);
    const std::vector<uint16_t> ref = trellis_ref::DecodeOracle(words, K, N, KB);
    Check(CountDiff(trellis_ref::DecodePairGrid(grid, K, N, KB), ref) == 0,
          "CPU decode: pair grid and oracle layout disagree at KB " + std::to_string(KB));
    const size_t diff = CountDiff(Reconstruct(grid, K, N, KB), ref);
    std::printf("  reconstruct KB=%d random %dx%d: %zu of %d values differ from the CPU decode\n", KB,
                K, N, diff, K * N);
    Check(diff == 0, "reconstruct KB " + std::to_string(KB) + " random: " + std::to_string(diff) +
                         " values differ");
  }
}

// ---- goldens (trellis_golden.py) -------------------------------------------------------------
void TestGoldens() {
  const std::string d = GoldenDir() + "/";
  using r4dx_test::LoadNpyF16Bits;
  using r4dx_test::LoadNpyU32;
  for (int KB : {4, 5}) {
    const std::string kb = std::to_string(KB);
    const int K = 1024, N = 512;
    const std::vector<uint32_t> grid = LoadNpyU32(d + "rand_k" + kb + "_w.npy", {K * N * KB / 32});
    const std::vector<uint16_t> q = LoadNpyF16Bits(d + "rand_k" + kb + "_q.npy", {K, N});
    const size_t cpu = CountDiff(trellis_ref::DecodePairGrid(grid, K, N, KB), q);
    const size_t gpu = CountDiff(Reconstruct(grid, K, N, KB), q);
    std::printf("  golden rand KB=%d %dx%d: CPU decode %zu, reconstruct %zu values differ\n", KB, K, N,
                cpu, gpu);
    Check(cpu == 0, "golden rand KB " + kb + ": CPU decode differs from decode_words");
    Check(gpu == 0, "golden rand KB " + kb + ": reconstruct differs from decode_words");
  }
  struct Real {
    const char* name;
    std::vector<const char*> words;   // oracle-layout parts, concatenated along N
    int N;
  };
  const Real reals[] = {{"attn_k", {"real_attn_k_words.npy"}, 256},
                        {"mlp_down", {"real_mlp_down_words.npy"}, 256},
                        {"mlp_gate_up", {"real_mlp_gate_words.npy", "real_mlp_up_words.npy"}, 512}};
  const int K = 256, KB = 4;
  for (const Real& r : reals) {
    const std::string base = d + "real_" + r.name;
    const std::vector<uint32_t> grid = LoadNpyU32(base + "_w.npy", {K * r.N * KB / 32});
    const std::vector<uint16_t> q = LoadNpyF16Bits(base + "_q.npy", {K, r.N});
    // The regrid: each part's oracle words, concatenated along N, into the pair grid.
    std::vector<uint32_t> regrid(grid.size());
    const int pn = r.N / static_cast<int>(r.words.size());
    for (size_t pi = 0; pi < r.words.size(); ++pi) {
      const std::vector<uint32_t> w = LoadNpyU32(d + r.words[pi], {K / 16, pn / 16, 8 * KB});
      for (int tk = 0; tk < K / 16; ++tk)
        for (int tn = 0; tn < pn / 16; ++tn)
          for (int x = 0; x < 8 * KB; ++x)
            regrid[trellis_ref::PairGridIndex(K, KB, static_cast<int64_t>(pi) * (pn / 16) + tn, tk) + x] =
                w[(static_cast<size_t>(tk) * (pn / 16) + tn) * 8 * KB + x];
    }
    Check(regrid == grid, std::string("golden real ") + r.name + ": the regrid of the oracle words "
                                                               "differs from the stored pair grid");
    const size_t cpu = CountDiff(trellis_ref::DecodePairGrid(grid, K, r.N, KB), q);
    const size_t gpu = CountDiff(Reconstruct(grid, K, r.N, KB), q);
    Check(cpu == 0, std::string("golden real ") + r.name + ": CPU decode differs from decode_words");
    Check(gpu == 0, std::string("golden real ") + r.name + ": reconstruct differs from decode_words");
    // And through the GEMM: one-hot rows over every k, as a 2-part linear for gate_up.
    Gemm g(grid, q, K, r.N);
    const bool two = r.words.size() == 2;
    FullCoverage(g, Tuning{4, 2, 1, 1, 1, 2, 1}, two, r.N / 2, std::string("golden real ") + r.name);
    FullCoverage(g, Tuning{1, 1, 1, 2, 4, 1, 0}, two, r.N / 2, std::string("golden real ") + r.name);
    std::printf("  golden real %-11s %dx%d: CPU decode %zu, reconstruct %zu values differ; regrid %s; "
                "raw GEMM one-hot over every k%s\n",
                r.name, K, r.N, cpu, gpu, regrid == grid ? "ok" : "WRONG", two ? " (2 parts)" : "");
  }
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937_64 rng(20260926);

  std::printf("trellis decode (docs/trellis-kernel.md 4.2-4.5, M1)\n");
  TestPreconditions();
  TestReconstructRandom(rng);
  TestRawFullCoverage(rng);
  TestRawSweep(rng);
  TestRawOrder(rng);

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
