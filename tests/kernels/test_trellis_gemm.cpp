// tests/kernels/test_trellis_gemm.cpp -- the whole trellis linear: r4dx_trellis_input_bf16, then
// libr4d's r4d_gemm_trellis_nt_m64 with its epilogue (in-block reduction, cross-block tickets for a
// split 128-column group, FWHT-128 in fp32, svh, one bf16 rounding), docs/trellis-kernel.md 4.3-4.5
// and 6, milestone M2:
//
//   accuracy     random weights at K = 5120, N = 1024, KB = 4 and 5, one part and two (n_split
//                512), prescale 0 and 4, M in {1, 3, 8, 16, 17, 64}, seven tunings (unsplit 128- and
//                256-wide blocks; Wc = 32 / 64 blocks with SKG 1-4; MT up to 4; SK up to 16):
//                every element within 4 bf16 ulp of the fp64 reference computed from the f16 A the
//                GEMM was given, plus a floor of 1e-4 * rms(row) (M0's tolerance study: an element
//                that cancels to near zero has no meaningful ulp); ||C - y|| / ||y|| <= 2e-3 against
//                the fp64 linear of the exact x; rows >= M untouched; tickets zero after every call.
//                The transform's A is checked bit-exact against trellis_ref on the way;
//   epilogue     bit for bit: the full entry's C equals bf16_rn((FwhtLds(S)[n] * svh[n]) * out_scale)
//                of the fp32 sums S its _raw entry returns at the same tuning (both entries run one
//                instantiation and one reduction; the flag only selects the epilogue), KB 4 and 5,
//                P 1 and 2, prescale 0 and 4, M 1..64, every block width and SKG 1-8 -- which pins
//                the output side's operation order (FWHT stages, the two products, no contraction,
//                one rounding) that the 4-ulp tolerance above cannot see;
//   invariance   every legal tuning of a K = 2048, N = 512 two-part weight at KB = 4 and 5, M = 5 and
//                16, grouped by (SK, SKG): all tunings of a group give the same bytes -- a result
//                depends on the summation order alone, and an unsplit block's in-LDS finish and a
//                split group's last-block finish agree bit for bit;
//   row identity for every row of tests/kernels/gemm_tuning_table_trellis.inc and 4.4's fallback
//                tuning (SKG up to 4; and up to 8, the first revision's rule, where that differs), at
//                the row's real (N, K, KB): row r of an M-row call (M = 2..16) is bit-identical to
//                the M = 1 call on that row;
//                (M5: the TP = 2 per-rank table's M = 1 rows too, at their rank shapes);
//   prefill rows every M > 16 row of both tables (M5) at both ends of the chunk sizes it serves, on
//                its K and KB (N narrowed to 1024): within the accuracy tolerance (TestPrefillRows);
//   determinism  100 repeats give identical bytes at SKG = 4 and at Wc = 32 on mlp.down's shape, and
//                at SKG = 8 (128- and 32-wide blocks, 8 and 32 per group) on attn.k/v's, M = 8 and 16;
//   tickets      tickets left non-zero (as a launch that never completed would leave them) make
//                the result wrong -- no block recognizes itself as the last -- and after
//                r4d_gemm_trellis_nt_m64_zero_tickets the next call is bit-exact again;
//   preconditions the host's throws (hence /EHc-), including NP*U > 8 and the LDS rule;
//   goldens      trellis_golden.py's random case (KB 4 and 5, P 1 and 2, prescale 0 and 4) and
//                the three real oracle blocks, from the goldens' own A, against their ya / y.
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
#include <iterator>
#include <map>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "npy_fixture.hpp"
#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"
#include "trellis_ref.hpp"
#include "trellis_tuning_rows.hpp"

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

constexpr uint16_t kSentinel = 0x7FC1u;   // a bf16 NaN no linear of finite values produces

float OutScale(int prescale_log2) {
  return static_cast<float>(std::ldexp(1.0, -prescale_log2) / std::sqrt(128.0));
}

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

// A split group needs every row tile in one block: raise MT to ceil(M / 16) there.
Tuning ForM(Tuning t, int M) {
  if (t.Split()) t.MT = std::max(t.MT, (M + 15) / 16);
  return t;
}

// docs/trellis-kernel.md 4.4's fallback for M <= 16: SKG capped at 4 (10.1 dropped SKG = 8);
// max_skg = 8 gives the first revision's rule, which the kernel still accepts.
Tuning Fallback(int N, int K, int max_skg = 4) {
  Tuning t{4, 2, 1, 1, 1, 2, 1};
  int skg = std::max(1, std::min(max_skg, 128 / (N / 128)));
  int p = 1;
  while (p * 2 <= skg) p *= 2;
  skg = p;
  while (skg > 1 && (K / 16) % (t.SK * skg * t.U) != 0) skg /= 2;
  t.SKG = skg;
  return t;
}

std::vector<uint32_t> RandomWords(std::mt19937_64& rng, size_t n) {
  std::vector<uint32_t> w(n);
  for (auto& x : w) x = static_cast<uint32_t>(rng());
  return w;
}

// Activations as trellis_golden.py's random_x (bf16; row scales 2^-8..2^3, two outliers per row).
std::vector<uint16_t> RandomX(std::mt19937_64& rng, int64_t M, int64_t K) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(-8.f, 3.f), od(20.f, 60.f);
  std::uniform_int_distribution<int64_t> kd(0, K - 1);
  std::vector<uint16_t> x(static_cast<size_t>(M * K));
  std::vector<float> row(static_cast<size_t>(K));
  for (int64_t m = 0; m < M; ++m) {
    const float s = std::exp2(ud(rng));
    for (auto& v : row) v = nd(rng) * s;
    for (int i = 0; i < 2; ++i) row[static_cast<size_t>(kd(rng))] *= od(rng);
    for (int64_t k = 0; k < K; ++k) x[static_cast<size_t>(m * K + k)] = FloatToBf16(row[static_cast<size_t>(k)]);
  }
  return x;
}

std::vector<float> RandomScales(std::mt19937_64& rng, int64_t n, double lo, double hi) {
  std::uniform_real_distribution<double> ud(std::log(lo), std::log(hi));
  std::uniform_int_distribution<int> sd(0, 1);
  std::vector<float> s(static_cast<size_t>(n));
  for (auto& v : s) v = F16ToFloat(FloatToF16(static_cast<float>((sd(rng) ? -1.0 : 1.0) * std::exp(ud(rng)))));
  return s;
}

// Random f16 activations of the size a transformed A has (|a| ~ 1e-3..1e-1).
std::vector<uint16_t> RandomA(std::mt19937_64& rng, size_t n) {
  std::normal_distribution<float> nd(0.f, 0.03f);
  std::vector<uint16_t> a(n);
  for (auto& v : a) v = FloatToF16(nd(rng));
  return a;
}

// One trellis linear on the device (words and svh) and the buffers a call needs for M <= 64.
struct Linear {
  int K, N, KB, n_split;
  std::vector<uint16_t> q;   // the CPU decode [K][N] (empty when not needed)
  std::vector<float> svh;
  DeviceBuffer<uint32_t> w;
  DeviceBuffer<float> svh_d;
  DeviceBuffer<uint16_t> a0, a1, c;
  DeviceBuffer<float> cf;   // the _raw entry's fp32 C
  DeviceBuffer<float> ws;
  DeviceBuffer<uint32_t> tickets;
  Linear(const std::vector<uint32_t>& grid, std::vector<uint16_t> q_, std::vector<float> svh_, int K_,
         int N_, int KB_, int n_split_)
      : K(K_), N(N_), KB(KB_), n_split(n_split_), q(std::move(q_)), svh(std::move(svh_)), w(grid.size()),
        svh_d(static_cast<size_t>(N_)), a0(static_cast<size_t>(64) * K_), a1(static_cast<size_t>(64) * K_),
        c(static_cast<size_t>(64) * N_), cf(static_cast<size_t>(64) * N_),
        ws(r4d_gemm_trellis_nt_m64_ws_bytes(64, N_, 8) / 4), tickets(r4d_gemm_trellis_nt_m64_tickets_bytes(N_) / 4) {
    w.CopyFromHost(grid);
    svh_d.CopyFromHost(svh);
    tickets.Zero();
  }
  bool TwoParts() const { return n_split < N; }
  void Run(const Tuning& t, int M, int prescale = 0) {
    r4d_gemm_trellis_nt_m64(P(a0.data()), TwoParts() ? P(a1.data()) : 0, n_split, P(w.data()),
                            P(svh_d.data()), P(c.data()), P(ws.data()), P(tickets.data()), M, K, N, KB,
                            t.WV, t.SK, t.MT, t.NP, t.SKG, t.U, t.NT, OutScale(prescale), 0);
  }
  void Sentinel() {
    std::vector<uint16_t> s(c.size(), kSentinel);
    c.CopyFromHost(s);
  }
  // Presets C, runs, returns C [64][N].
  std::vector<uint16_t> Call(const Tuning& t, int M, int prescale = 0) {
    Sentinel();
    Run(t, M, prescale);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    return c.CopyToHost();
  }
  // The _raw entry at the same tuning: the fp32 sums [64][N] (rows >= M hold whatever was there).
  std::vector<float> CallRaw(const Tuning& t, int M) {
    r4d_gemm_trellis_nt_m64_raw(P(a0.data()), TwoParts() ? P(a1.data()) : 0, n_split, P(w.data()), P(cf.data()),
                                P(ws.data()), P(tickets.data()), M, K, N, KB, t.WV, t.SK, t.MT, t.NP, t.SKG, t.U,
                                t.NT, 0, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    return cf.CopyToHost();
  }
  bool TicketsZero() const {
    for (uint32_t x : tickets.CopyToHost())
      if (x != 0) return false;
    return true;
  }
};

std::unique_ptr<Linear> RandomLinear(std::mt19937_64& rng, int K, int N, int KB, int n_split, bool decode) {
  const std::vector<uint32_t> words = RandomWords(rng, static_cast<size_t>(K / 16) * (N / 16) * 8 * KB);
  std::vector<uint32_t> grid = trellis_ref::ToPairGrid(words, K, N, KB);
  std::vector<uint16_t> q;
  if (decode) q = trellis_ref::DecodePairGrid(grid, K, N, KB);
  return std::make_unique<Linear>(grid, std::move(q), RandomScales(rng, N, 0.6, 1.9), K, N, KB, n_split);
}

// Per-element and Frobenius check of C rows < M against ya (from the f16 A) and y (exact x, may be
// empty); rows >= M must still hold the sentinel.
struct Tol {
  size_t bad = 0, untouched_bad = 0;
  double worst = 0.0;   // max err / (4 ulp + floor)
  double fro = 0.0;     // ||C - y|| / ||y||
};
// `k_scale`: the 4-ulp + floor limit was calibrated at K = 5120 (M0); fp32 accumulation error grows ~sqrt(K), so
// longer reductions (Gemma's o_full K 8192, down K 15360) get the limit times sqrt(K / 5120). K <= 5120 is unchanged.
double KScale(int K) { return std::max(1.0, std::sqrt(K / 5120.0)); }

Tol CheckTolerance(const std::vector<uint16_t>& c, int M, int N, const std::vector<double>& ya,
                   const std::vector<double>& y, double k_scale = 1.0) {
  Tol r;
  double num = 0.0, den = 0.0;
  for (int m = 0; m < 64; ++m) {
    if (m >= M) {
      for (int n = 0; n < N; ++n) r.untouched_bad += c[static_cast<size_t>(m) * N + n] != kSentinel;
      continue;
    }
    double ss = 0.0;
    for (int n = 0; n < N; ++n) ss += ya[static_cast<size_t>(m) * N + n] * ya[static_cast<size_t>(m) * N + n];
    const double floor = 1e-4 * std::sqrt(ss / N);
    for (int n = 0; n < N; ++n) {
      const size_t i = static_cast<size_t>(m) * N + n;
      const double got = Bf16ToFloat(c[i]);
      const double lim = (4.0 * trellis_ref::Bf16Ulp(ya[i]) + floor) * k_scale;
      const double err = std::fabs(got - ya[i]);
      if (!(err <= lim)) ++r.bad;
      r.worst = std::max(r.worst, err / lim);
      if (!y.empty()) {
        num += (got - y[i]) * (got - y[i]);
        den += y[i] * y[i];
      }
    }
  }
  r.fro = den > 0 ? std::sqrt(num / den) : 0.0;
  return r;
}

// ---- accuracy -------------------------------------------------------------------------------------
const Tuning kAccTunings[] = {
    {4, 2, 1, 1, 1, 2, 1},    // Wc 128, unsplit (the fallback's shape)
    {1, 4, 1, 1, 2, 4, 1},    // Wc 32, SKG 2 (M1's sweep winners' shape)
    {2, 8, 1, 4, 1, 2, 0},    // Wc 256, NP 4, unsplit
    {1, 16, 1, 2, 4, 1, 1},   // Wc 64, SKG 4, SK 16
    {2, 4, 4, 1, 1, 4, 0},    // Wc 64 (split, one ticket for two blocks), MT 4
    {4, 1, 2, 2, 1, 1, 1},    // Wc 256, NP 2, MT 2, SK 1
    {1, 8, 1, 1, 4, 2, 1},    // Wc 32, SKG 4
};

void TestAccuracyAt(std::mt19937_64& rng, int K, int N, int min_calls) {
  const int Ms[] = {1, 3, 8, 16, 17, 64};
  double worst = 0.0, worst_fro = 0.0;
  int calls = 0, skipped = 0;
  for (int KB : {4, 5}) {
    for (int parts : {1, 2}) {
      const int n_split = parts == 2 ? N / 2 : N;
      std::unique_ptr<Linear> L = RandomLinear(rng, K, N, KB, n_split, true);
      std::vector<std::vector<float>> suh;
      for (int p = 0; p < parts; ++p) suh.push_back(RandomScales(rng, K, 5e-3, 4e-2));
      const std::vector<uint16_t> x = RandomX(rng, 64, K);
      DeviceBuffer<uint16_t> x_d(x.size());
      x_d.CopyFromHost(x);
      std::vector<DeviceBuffer<float>> suh_d;
      suh_d.reserve(2);
      int64_t sp[3] = {0, 0, 0}, op[3] = {P(L->a0.data()), P(L->a1.data()), 0};
      for (int p = 0; p < parts; ++p) {
        suh_d.emplace_back(static_cast<size_t>(K));
        suh_d.back().CopyFromHost(suh[p]);
        sp[p] = P(suh_d.back().data());
      }
      std::vector<const float*> suh_ptrs;
      for (auto& s : suh) suh_ptrs.push_back(s.data());
      const std::vector<double> y = trellis_ref::LinearExact(x, 64, K, N, n_split, suh_ptrs, L->q, L->svh.data());
      for (int s : (parts == 2 ? std::vector<int>{0, 4} : std::vector<int>{0})) {
        // The transform (both parts in one call), checked bit for bit on the way.
        r4dx_trellis_input_bf16(P(x_d.data()), 64, K, parts, sp, op, s, 0);
        R4DX_HIP_CHECK(hipDeviceSynchronize());
        std::vector<std::vector<uint16_t>> a;
        std::vector<const uint16_t*> a_ptrs;
        for (int p = 0; p < parts; ++p) {
          a.push_back((p == 0 ? L->a0 : L->a1).CopyToHost());
          const std::vector<uint16_t> want = trellis_ref::TransformInput(x, 64, K, suh[p].data(), s);
          Check(a.back() == want, "transform part " + std::to_string(p) + " differs from trellis_ref");
        }
        for (auto& v : a) a_ptrs.push_back(v.data());
        const std::vector<double> ya = trellis_ref::LinearFromA(a_ptrs, 64, K, N, n_split, L->q, L->svh.data(), s);
        for (const Tuning& t0 : kAccTunings)
          for (int M : Ms) {
            const Tuning t = ForM(t0, M);
            std::vector<uint16_t> c;
            try {
              c = L->Call(t, M, s);
            } catch (const std::exception&) {
              ++skipped;   // not instantiated at this (KB, NP, U, MT)
              continue;
            }
            ++calls;
            const Tol r = CheckTolerance(c, M, N, ya, y, KScale(K));
            worst = std::max(worst, r.worst);
            worst_fro = std::max(worst_fro, r.fro);
            const std::string what = "accuracy K=" + std::to_string(K) + " KB=" + std::to_string(KB) + " P=" +
                                     std::to_string(parts) + " s=" + std::to_string(s) + " M=" + std::to_string(M) +
                                     " " + t.Str();
            Check(r.bad == 0, what + ": " + std::to_string(r.bad) + " elements outside (4 bf16 ulp + 1e-4 rms) x " +
                                  std::to_string(KScale(K)));
            Check(r.fro <= 2e-3, what + ": ||C-y||/||y|| = " + std::to_string(r.fro));
            Check(r.untouched_bad == 0, what + ": rows >= M written");
            if (t.Split()) Check(L->TicketsZero(), what + ": tickets not zero after the call");
          }
      }
    }
  }
  std::printf("  accuracy K=%d N=%d, KB 4/5, P 1/2, prescale 0/4, M 1..64, %zu tunings: %d calls (%d not "
              "instantiated), worst element %.3f of the tolerance, worst ||C-y||/||y|| %.2e\n",
              K, N, sizeof(kAccTunings) / sizeof(kAccTunings[0]), calls, skipped, worst, worst_fro);
  Check(calls >= min_calls, "accuracy: too few calls ran (" + std::to_string(calls) + ")");
}

void TestAccuracy(std::mt19937_64& rng) {
  TestAccuracyAt(rng, 5120, 1024, 150);
  // Gemma 4 12B (docs/gemma4-plan.md 3.7) K and narrow-N edges: hidden 3840 (full-attention k is N = 512),
  // o_proj sliding 4096 and full 8192, down 15360. A tuning the kernel does not instantiate at a K is
  // skipped (and counted), so the floor is lower than at 5120.
  TestAccuracyAt(rng, 3840, 512, 40);
  TestAccuracyAt(rng, 3840, 1024, 40);
  TestAccuracyAt(rng, 4096, 1024, 40);
  TestAccuracyAt(rng, 8192, 1024, 40);
  TestAccuracyAt(rng, 15360, 1024, 40);
}

// ---- epilogue, bit for bit ----------------------------------------------------------------------------
// The full entry against trellis_ref::EpilogueFromRaw of its own _raw sums. The tunings cover every
// finishing path: unsplit 128- and 256-wide blocks (the in-LDS finish, one and two groups per block),
// split groups of Wc 32 / 64 / 128 / 256 blocks at SKG 1-8 (the last block's finish, the 256-wide one
// with two groups per ticket), MT up to 4 and SK up to 16.
void TestEpilogueExact(std::mt19937_64& rng) {
  const int K = 2048, N = 1024;
  std::vector<Tuning> tunings(std::begin(kAccTunings), std::end(kAccTunings));
  tunings.push_back({1, 8, 1, 1, 1, 4, 1});   // Wc 32, SKG 1
  tunings.push_back({2, 4, 1, 2, 8, 1, 1});   // Wc 128, SKG 8
  tunings.push_back({4, 2, 1, 2, 2, 2, 1});   // Wc 256, split (SKG 2): two groups, one ticket
  tunings.push_back({2, 2, 1, 4, 4, 1, 0});   // Wc 256, NP 4, SKG 4
  int calls = 0, bad_calls = 0;
  for (int KB : {4, 5})
    for (int parts : {1, 2}) {
      std::unique_ptr<Linear> L = RandomLinear(rng, K, N, KB, parts == 2 ? N / 2 : N, false);
      L->a0.CopyFromHost(RandomA(rng, static_cast<size_t>(64) * K));
      L->a1.CopyFromHost(RandomA(rng, static_cast<size_t>(64) * K));
      for (int s : {0, 4})
        for (const Tuning& t0 : tunings)
          for (int M : {1, 5, 9, 16, 17, 64}) {
            const Tuning t = ForM(t0, M);
            std::vector<uint16_t> c;
            std::vector<float> raw;
            try {
              c = L->Call(t, M, s);
              raw = L->CallRaw(t, M);
            } catch (const std::exception&) {
              continue;   // not instantiated at this (KB, NP, U, MT)
            }
            ++calls;
            const std::vector<uint16_t> want = trellis_ref::EpilogueFromRaw(raw, M, N, L->svh.data(), OutScale(s));
            size_t differ = 0;
            for (size_t i = 0; i < static_cast<size_t>(M) * N; ++i) differ += c[i] != want[i];
            bad_calls += differ != 0;
            Check(differ == 0, "epilogue KB=" + std::to_string(KB) + " P=" + std::to_string(parts) + " s=" +
                                   std::to_string(s) + " M=" + std::to_string(M) + " " + t.Str() + ": " +
                                   std::to_string(differ) + " elements differ from bf16((FwhtLds(raw) * svh) * "
                                   "out_scale)");
          }
    }
  std::printf("  epilogue K=%d N=%d, KB 4/5, P 1/2, prescale 0/4, M 1..64, %zu tunings: %d calls, %d with an "
              "element off bf16((FwhtLds(raw) * svh) * out_scale)\n",
              K, N, tunings.size(), calls, bad_calls);
  Check(calls >= 300, "epilogue: too few calls ran (" + std::to_string(calls) + ")");
}

// ---- tuning invariance ------------------------------------------------------------------------------
void TestInvariance(std::mt19937_64& rng) {
  const int K = 2048, N = 512;
  int legal_total = 0, classes_total = 0;
  double worst = 0.0;
  for (int KB : {4, 5}) {
    std::unique_ptr<Linear> L = RandomLinear(rng, K, N, KB, N / 2, true);
    const std::vector<uint16_t> a0 = RandomA(rng, static_cast<size_t>(64) * K), a1 = RandomA(rng, static_cast<size_t>(64) * K);
    L->a0.CopyFromHost(a0);
    L->a1.CopyFromHost(a1);
    const std::vector<double> ya = trellis_ref::LinearFromA({a0.data(), a1.data()}, 64, K, N, N / 2, L->q, L->svh.data(), 0);
    for (int M : {5, 16}) {
      std::map<std::pair<int, int>, std::vector<uint16_t>> ref;
      int legal = 0, differ = 0;
      for (int WV : {1, 2, 4})
        for (int NP : {1, 2, 4})
          for (int SK : {1, 2, 4, 8, 16})
            for (int SKG : {1, 2, 4, 8})
              for (int MT : {1, 2, 3, 4})
                for (int U : {1, 2, 4})
                  for (int NT : {0, 1}) {
                    const Tuning t{WV, SK, MT, NP, SKG, U, NT};
                    std::vector<uint16_t> c;
                    try {
                      c = L->Call(t, M);
                    } catch (const std::exception&) {
                      continue;
                    }
                    ++legal;
                    const auto key = std::make_pair(SK, SKG);
                    auto it = ref.find(key);
                    if (it == ref.end()) {
                      const Tol r = CheckTolerance(c, M, N, ya, {});
                      worst = std::max(worst, r.worst);
                      Check(r.bad == 0 && r.untouched_bad == 0,
                            "invariance KB=" + std::to_string(KB) + " M=" + std::to_string(M) + " " + t.Str() +
                                ": the class reference is outside the tolerance");
                      ref.emplace(key, std::move(c));
                    } else if (c != it->second) {
                      ++differ;
                      Check(false, "invariance KB=" + std::to_string(KB) + " M=" + std::to_string(M) + " " +
                                       t.Str() + ": bytes differ from its (SK, SKG) class");
                    }
                    if (t.Split()) Check(L->TicketsZero(), "invariance " + t.Str() + ": tickets not zero");
                  }
      legal_total += legal;
      classes_total += static_cast<int>(ref.size());
      std::printf("  invariance KB=%d M=%d K=%d N=%d (2 parts): %d legal tunings in %zu (SK, SKG) classes, %d "
                  "differ from their class\n",
                  KB, M, K, N, legal, ref.size(), differ);
    }
  }
  std::printf("  invariance: class references within %.3f of the tolerance\n", worst);
  Check(legal_total > 2000, "invariance: suspiciously few legal tunings (" + std::to_string(legal_total) + ")");
  Check(classes_total >= 60, "invariance: too few (SK, SKG) classes (" + std::to_string(classes_total) + ")");
}

// ---- row identity -----------------------------------------------------------------------------------
// mlp.gate_up's two parts: 34816 x 5120 at TP = 1, 17408 x 5120 on a TP = 2 rank.
bool GateUp(int64_t N, int64_t K) {
  // Qwen: 34816 (TP = 1) / 17408 (a TP = 2 rank) x 5120; Gemma 4 12B: 30720 / 15360 x 3840.
  return ((N == 34816 || N == 17408) && K == 5120) || ((N == 30720 || N == 15360) && K == 3840);
}

void RowIdentityImpl(std::mt19937_64& rng, int N, int K, int KB, const Tuning& t, const std::string& label);

void RowIdentity(std::mt19937_64& rng, int N, int K, int KB, const Tuning& t, const std::string& label) {
  try {
    RowIdentityImpl(rng, N, K, KB, t, label);
  } catch (const std::exception& e) {  // a rejected tuning is a failure to report, not a reason to abort the run
    Check(false, "row identity " + label + " N=" + std::to_string(N) + " K=" + std::to_string(K) + " " + t.Str() +
                     ": rejected (" + e.what() + ")");
  }
}

void RowIdentityImpl(std::mt19937_64& rng, int N, int K, int KB, const Tuning& t, const std::string& label) {
  const int n_split = GateUp(N, K) ? N / 2 : N;
  std::unique_ptr<Linear> L = RandomLinear(rng, K, N, KB, n_split, false);
  const std::vector<uint16_t> a0 = RandomA(rng, static_cast<size_t>(16) * K), a1 = RandomA(rng, static_cast<size_t>(16) * K);
  // M = 1 on each row alone.
  std::vector<std::vector<uint16_t>> single(16);
  for (int r = 0; r < 16; ++r) {
    L->a0.CopyFromHost(a0.data() + static_cast<size_t>(r) * K, static_cast<size_t>(K));
    L->a1.CopyFromHost(a1.data() + static_cast<size_t>(r) * K, static_cast<size_t>(K));
    const std::vector<uint16_t> c = L->Call(t, 1);
    single[r].assign(c.begin(), c.begin() + N);
  }
  L->a0.CopyFromHost(a0.data(), a0.size());
  L->a1.CopyFromHost(a1.data(), a1.size());
  int bad_rows = 0;
  for (int M : {2, 3, 4, 5, 8, 9, 13, 16}) {
    const std::vector<uint16_t> c = L->Call(t, M);
    for (int r = 0; r < M; ++r)
      bad_rows += std::memcmp(&c[static_cast<size_t>(r) * N], single[r].data(), static_cast<size_t>(N) * 2) != 0;
  }
  Check(bad_rows == 0, "row identity " + label + " " + t.Str() + ": " + std::to_string(bad_rows) +
                           " rows differ from their M = 1 call");
  std::printf("  row identity %-24s N=%5d K=%5d KB=%d %s: %s\n", label.c_str(), N, K, KB, t.Str().c_str(),
              bad_rows ? "ROWS DIFFER" : "rows of M = 2..16 == M = 1");
}

void TestRowIdentity(std::mt19937_64& rng) {
  int rows = 0;
  const auto table = [&](const auto& rows_of, const char* label) {
    for (const trellis_rows::GemmTuningRow& row : rows_of) {
      if (row.layout != trellis_rows::Layout::kTrellis || row.M > 16) continue;
      const trellis_rows::LinearTuning& lt = row.tuning;
      const Tuning t{lt.WV, lt.SK, lt.MB, lt.NPW, lt.SKG, lt.U, lt.NT};
      RowIdentity(rng, static_cast<int>(row.N), static_cast<int>(row.K), row.rate, t, label);
      ++rows;
    }
  };
  table(trellis_rows::kGemmTuningTable, "table row");
  table(trellis_rows::tp2::kGemmTuningTable, "TP = 2 table row");
  Check(rows > 0, "the trellis tuning table has no M <= 16 rows");
  const std::pair<int, int> shapes[] = {{10240, 5120}, {6144, 5120}, {5120, 6144}, {12288, 5120},
                                        {1024, 5120},  {34816, 5120}, {5120, 17408},
                                        // Gemma 4 12B at TP = 1 (docs/gemma4-plan.md 3.7): sliding q/k/v/o, full
                                        // q/k/o, gate_up (two parts), down; then the TP = 2 rank shapes.
                                        {4096, 3840},  {2048, 3840},  {512, 3840},   {8192, 3840},
                                        {3840, 4096},  {3840, 8192},  {30720, 3840}, {3840, 15360},
                                        {15360, 3840}, {3840, 7680},  {1024, 3840},  {3840, 2048}};
  for (auto nk : shapes) {
    const Tuning f4 = Fallback(nk.first, nk.second), f8 = Fallback(nk.first, nk.second, 8);
    RowIdentity(rng, nk.first, nk.second, 4, f4, "4.4 fallback");
    if (f8.SKG != f4.SKG) RowIdentity(rng, nk.first, nk.second, 4, f8, "fallback at SKG <= 8");
  }
}

// ---- prefill rows ------------------------------------------------------------------------------------
// Every M > 16 row of both tables (M5's prefill picks, which production runs for every prefill
// chunk; the TP = 2 per-rank table's too) at the smallest and the largest chunk M it serves (17..32
// for an M = 32 row, 33..64 for M = 64: linear.cpp's BestRow takes the smallest row M >= the
// chunk's), on the row's K and KB and a narrower N of 1024 (every block width divides it;
// mlp.gate_up keeps two parts, n_split 512):
// within the accuracy tolerance of the fp64 linear of its f16 A, rows >= M untouched, tickets zero
// after a split call.
void TestPrefillRows(std::mt19937_64& rng) {
  const int N = 1024;
  struct Ref {
    std::unique_ptr<Linear> L;
    std::vector<double> ya;
  };
  std::map<std::tuple<int, int, int>, Ref> refs;   // (K, KB, parts)
  int calls = 0;
  double worst = 0.0;
  // The TP = 1 table's rows, then the TP = 2 per-rank table's (M5), each at the M band its own
  // table's rows give it.
  std::vector<std::pair<const trellis_rows::GemmTuningRow*, const char*>> all;
  for (const trellis_rows::GemmTuningRow& row : trellis_rows::kGemmTuningTable) all.push_back({&row, "TP1"});
  for (const trellis_rows::GemmTuningRow& row : trellis_rows::tp2::kGemmTuningTable) all.push_back({&row, "TP2"});
  for (const auto& entry : all) {
    const trellis_rows::GemmTuningRow& row = *entry.first;
    if (row.layout != trellis_rows::Layout::kTrellis || row.M <= 16) continue;
    const int K = static_cast<int>(row.K), KB = row.rate, parts = GateUp(row.N, row.K) ? 2 : 1;
    const int n_split = parts == 2 ? N / 2 : N;
    Ref& ref = refs[{K, KB, parts}];
    if (!ref.L) {
      ref.L = RandomLinear(rng, K, N, KB, n_split, true);
      const std::vector<uint16_t> a0 = RandomA(rng, static_cast<size_t>(64) * K);
      const std::vector<uint16_t> a1 = RandomA(rng, static_cast<size_t>(64) * K);
      ref.L->a0.CopyFromHost(a0);
      ref.L->a1.CopyFromHost(a1);
      std::vector<const uint16_t*> a_ptrs{a0.data()};
      if (parts == 2) a_ptrs.push_back(a1.data());
      ref.ya = trellis_ref::LinearFromA(a_ptrs, 64, K, N, n_split, ref.L->q, ref.L->svh.data(), 0);
    }
    int lo = 17;
    for (const auto& other : all) {
      const trellis_rows::GemmTuningRow& o = *other.first;
      if (other.second == entry.second && o.N == row.N && o.K == row.K && o.rate == row.rate && o.M > 16 &&
          o.M < row.M)
        lo = std::max(lo, static_cast<int>(o.M) + 1);
    }
    const trellis_rows::LinearTuning& lt = row.tuning;
    const Tuning t{lt.WV, lt.SK, lt.MB, lt.NPW, lt.SKG, lt.U, lt.NT};
    for (int M : {lo, static_cast<int>(row.M)}) {
      const std::string what = std::string(entry.second) + " prefill row N=" + std::to_string(row.N) + " K=" +
                               std::to_string(K) + " KB=" + std::to_string(KB) + " M=" + std::to_string(M) + " " +
                               t.Str();
      std::vector<uint16_t> c;
      try {
        c = ref.L->Call(t, M);
      } catch (const std::exception& e) {
        Check(false, what + ": rejected (" + e.what() + ")");
        continue;
      }
      ++calls;
      const Tol r = CheckTolerance(c, M, N, ref.ya, {});
      worst = std::max(worst, r.worst);
      Check(r.bad == 0, what + ": " + std::to_string(r.bad) + " elements outside 4 bf16 ulp + 1e-4 rms");
      Check(r.untouched_bad == 0, what + ": rows >= M written");
      if (t.Split()) Check(ref.L->TicketsZero(), what + ": tickets not zero after the call");
    }
  }
  std::printf("  prefill rows: %d calls (every M > 16 row of both tables at the ends of its M band, N = 1024), "
              "worst element %.3f of the tolerance\n",
              calls, worst);
  Check(calls > 0, "the trellis tuning table has no M > 16 rows");
}

// ---- determinism --------------------------------------------------------------------------------------
void TestDeterminism(std::mt19937_64& rng) {
  struct Case {
    const char* label;
    int N, K;
    std::vector<Tuning> tunings;
  };
  const Case cases[] = {
      // SKG 4 on 128-wide blocks, and Wc 32 at SKG 1 and 2.
      {"mlp.down", 5120, 17408, {{4, 4, 1, 1, 4, 4, 1}, {1, 4, 1, 1, 1, 4, 1}, {1, 4, 1, 1, 2, 4, 1}}},
      // SKG 8, the kernel's largest (the first revision's fallback for this shape: 8 blocks per
      // group), and Wc 32 at SKG 8 (32 per group).
      {"attn.k/v", 1024, 5120, {Fallback(1024, 5120, 8), {1, 4, 1, 1, 8, 2, 1}}},
      // Gemma 4 12B: full-attention k (N 512, K 3840) and down (N 3840, K 15360).
      // (SK 1 at SKG 8: K/16 = 240 must divide by SK*SKG*U, so SK 4 / U 2 would be illegal at K 3840.)
      {"g.k_full", 512, 3840, {Fallback(512, 3840, 8), {1, 1, 1, 1, 8, 2, 1}}},
      {"g.down", 3840, 15360, {Fallback(3840, 15360), {1, 4, 1, 1, 2, 4, 1}}},
  };
  for (const Case& cs : cases)
    for (int KB : {4, 5}) {
      std::unique_ptr<Linear> L = RandomLinear(rng, cs.K, cs.N, KB, cs.N, false);
      L->a0.CopyFromHost(RandomA(rng, static_cast<size_t>(64) * cs.K));
      for (const Tuning& t : cs.tunings) {
        for (int M : {8, 16}) {
          const std::vector<uint16_t> first = L->Call(t, M);
          int differ = 0;
          for (int r = 0; r < 100; ++r) {
            L->Run(t, M);
            R4DX_HIP_CHECK(hipDeviceSynchronize());
            differ += L->c.CopyToHost() != first;
          }
          Check(differ == 0, std::string("determinism ") + cs.label + " KB=" + std::to_string(KB) + " M=" +
                                 std::to_string(M) + " " + t.Str() + ": " + std::to_string(differ) +
                                 " of 100 repeats differ");
          Check(L->TicketsZero(), "determinism " + t.Str() + ": tickets not zero");
          std::printf("  determinism %-8s KB=%d M=%2d %s (Wc %d, SKG %d): %d of 100 repeats differ\n", cs.label,
                      KB, M, t.Str().c_str(), t.Wc(), t.SKG, differ);
        }
      }
    }
}

// ---- tickets --------------------------------------------------------------------------------------------
void TestTicketReset(std::mt19937_64& rng) {
  const int N = 5120, K = 6144;
  std::unique_ptr<Linear> L = RandomLinear(rng, K, N, 4, N, false);
  L->a0.CopyFromHost(RandomA(rng, static_cast<size_t>(64) * K));
  const size_t tb = r4d_gemm_trellis_nt_m64_tickets_bytes(N);
  for (const Tuning& t : {Tuning{4, 2, 1, 1, 2, 2, 1}, Tuning{1, 4, 1, 1, 1, 4, 1}}) {
    const std::vector<uint16_t> good = L->Call(t, 8);
    // What a launch that never completed leaves behind: counters no block will recognize as the last.
    R4DX_HIP_CHECK(hipMemset(L->tickets.data(), 0x40, tb));
    const std::vector<uint16_t> stale = L->Call(t, 8);
    size_t wrong = 0;
    for (size_t i = 0; i < good.size(); ++i) wrong += stale[i] != good[i];
    Check(wrong > 0, "tickets left non-zero did not change the result (" + t.Str() + ")");
    Check(!L->TicketsZero(), "tickets left non-zero came back zero (" + t.Str() + ")");
    r4d_gemm_trellis_nt_m64_zero_tickets(P(L->tickets.data()), tb, 0);
    const std::vector<uint16_t> again = L->Call(t, 8);
    Check(again == good, "after zero_tickets the result is not bit-exact again (" + t.Str() + ")");
    Check(L->TicketsZero(), "tickets not zero after the recovered call (" + t.Str() + ")");
    std::printf("  tickets %s: stale tickets -> %zu of %zu outputs wrong; after zero_tickets bit-exact %s\n",
                t.Str().c_str(), wrong, good.size(), again == good ? "yes" : "NO");
  }
}

// ---- preconditions ---------------------------------------------------------------------------------------
void TestPreconditions() {
  const int K = 1024, N = 512;
  auto call = [&](int M, int K_, int N_, int n_split, int KB, int WV, int SK, int MT, int NP, int SKG, int U,
                  int64_t ws, int64_t svh) {
    r4d_gemm_trellis_nt_m64(16, 0, n_split, 16, svh, 16, ws, ws, M, K_, N_, KB, WV, SK, MT, NP, SKG, U, 0,
                            1.f, 0);
  };
  Check(Throws([&] { call(1, K, N, N, 4, 4, 2, 1, 1, 1, 2, 16, 0); }), "null svh accepted");
  Check(Throws([&] { call(0, K, N, N, 4, 4, 2, 1, 1, 1, 2, 16, 16); }), "M = 0 accepted");
  Check(Throws([&] { call(65, K, N, N, 4, 4, 2, 1, 1, 1, 2, 16, 16); }), "M = 65 accepted");
  Check(Throws([&] { call(1, 1000, N, N, 4, 4, 2, 1, 1, 1, 2, 16, 16); }), "K % 128 accepted");
  Check(Throws([&] { call(1, K, 500, 500, 4, 4, 2, 1, 1, 1, 2, 16, 16); }), "N % 128 accepted");
  Check(Throws([&] { call(1, K, N, N, 3, 4, 2, 1, 1, 1, 2, 16, 16); }), "KB = 3 accepted");
  Check(Throws([&] { call(1, K, N, N, 6, 4, 2, 1, 1, 1, 2, 16, 16); }), "KB = 6 accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 1, 2, 1, 4, 1, 4, 16, 16); }), "NP*U = 16 accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 1, 2, 1, 2, 1, 8, 16, 16); }), "U = 8 accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 1, 2, 2, 4, 1, 2, 16, 16); }), "(NP 4, U 2, MT 2) accepted");
  Check(Throws([&] { call(1, K, N, N, 5, 1, 2, 3, 2, 1, 4, 16, 16); }), "(KB 5, NP 2, U 4, MT 3) accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 4, 2, 1, 4, 1, 1, 16, 16); }), "Wc = 512 accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 2, 16, 1, 4, 1, 1, 16, 16); }), "SK*Wc*32 = 128 KiB of LDS accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 4, 16, 1, 1, 1, 1, 16, 16); }), "2048 threads accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 4, 4, 1, 1, 8, 4, 16, 16); }), "(K/16) % (SK*SKG*U) accepted");
  Check(Throws([&] { call(17, K, N, N, 4, 4, 2, 1, 1, 2, 2, 16, 16); }), "split group with 2 row tiles at MT 1 accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 1, 2, 1, 1, 1, 2, 0, 16); }), "split group without ws/tickets accepted");
  Check(Throws([&] { call(1, K, N, 192, 4, 4, 2, 1, 1, 1, 2, 16, 16); }), "n_split % 128 accepted");
  Check(Throws([&] { call(1, K, N, 128, 4, 4, 2, 1, 2, 1, 2, 16, 16); }), "n_split % Wc (256) accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 3, 2, 1, 1, 1, 2, 16, 16); }), "WV = 3 accepted");
  Check(Throws([&] { call(1, K, N, N, 4, 4, 2, 1, 1, 3, 2, 16, 16); }), "SKG = 3 accepted");
  Check(r4d_gemm_trellis_nt_m64_has_rate(4) == 1 && r4d_gemm_trellis_nt_m64_has_rate(5) == 1 &&
            r4d_gemm_trellis_nt_m64_has_rate(3) == 0 && r4d_gemm_trellis_nt_m64_has_rate(6) == 0,
        "has_rate: 4 and 5 yes, 3 and 6 no");
  Check(r4d_gemm_trellis_nt_m64_max_m() == 64, "max_m 64");
  Check(r4d_gemm_trellis_nt_m64_ws_bytes(16, 5120, 4) == static_cast<size_t>(4) * 16 * 5120 * 4, "ws_bytes");
  Check(r4d_gemm_trellis_nt_m64_tickets_bytes(5120) == 160, "tickets_bytes(5120) = 40 u32");
  Check(!Throws([&] { r4d_gemm_trellis_nt_m64_zero_tickets(0, 0, 0); }), "zero_tickets of 0 bytes is a no-op");
  Check(Throws([&] { r4d_gemm_trellis_nt_m64_zero_tickets(0, 16, 0); }), "zero_tickets of a null buffer accepted");
}

// ---- goldens (trellis_golden.py) -----------------------------------------------------------------------
std::vector<double> ToF64(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }
std::vector<float> WidenF16(const std::vector<uint16_t>& h) {
  std::vector<float> f(h.size());
  for (size_t i = 0; i < h.size(); ++i) f[i] = F16ToFloat(h[i]);
  return f;
}

// a: f16 [parts][64][K]. Runs every tuning of `tunings` legal at this shape, M in {1,3,8,16,17,64}.
void GoldenLinear(const std::string& label, const std::vector<uint32_t>& grid, int K, int N, int KB, int parts,
                  const std::vector<uint16_t>& a, const std::vector<uint16_t>& svh16, const std::vector<float>& ya,
                  const std::vector<float>& y, int prescale, const std::vector<Tuning>& tunings) {
  Linear L(grid, {}, WidenF16(svh16), K, N, KB, parts == 2 ? N / 2 : N);
  L.a0.CopyFromHost(a.data(), static_cast<size_t>(64) * K);
  if (parts == 2) L.a1.CopyFromHost(a.data() + static_cast<size_t>(64) * K, static_cast<size_t>(64) * K);
  const std::vector<double> ya64 = ToF64(ya), y64 = ToF64(y);
  int calls = 0;
  double worst = 0.0, fro = 0.0;
  for (const Tuning& t0 : tunings)
    for (int M : {1, 3, 8, 16, 17, 64}) {
      const Tuning t = ForM(t0, M);
      std::vector<uint16_t> c;
      try {
        c = L.Call(t, M, prescale);
      } catch (const std::exception&) {
        continue;
      }
      ++calls;
      const Tol r = CheckTolerance(c, M, N, ya64, y64);
      worst = std::max(worst, r.worst);
      fro = std::max(fro, r.fro);
      const std::string what = "golden " + label + " M=" + std::to_string(M) + " " + t.Str();
      Check(r.bad == 0, what + ": " + std::to_string(r.bad) + " elements outside the tolerance");
      Check(r.fro <= 2e-3, what + ": ||C-y||/||y|| = " + std::to_string(r.fro));
      Check(r.untouched_bad == 0, what + ": rows >= M written");
    }
  Check(calls > 0, "golden " + label + ": no tuning was legal");
  std::printf("  golden %-22s K=%4d N=%3d KB=%d P=%d s=%d: %d calls, worst element %.3f of the tolerance, "
              "worst ||C-y||/||y|| %.2e\n",
              label.c_str(), K, N, KB, parts, prescale, calls, worst, fro);
}

void TestGoldens() {
  using r4dx_test::LoadNpyF16Bits;
  using r4dx_test::LoadNpyF32;
  using r4dx_test::LoadNpyU32;
  const std::string d = GoldenDir() + "/";
  {
    const int K = 1024, N = 512;
    const std::vector<uint16_t> svh = LoadNpyF16Bits(d + "rand_svh.npy", {N});
    const std::vector<Tuning> tunings = {{4, 2, 1, 1, 1, 2, 1}, {1, 4, 1, 1, 2, 4, 1}, {1, 16, 1, 2, 4, 1, 1},
                                         {2, 8, 1, 4, 1, 2, 0}};
    for (int KB : {4, 5}) {
      const std::string kb = std::to_string(KB);
      const std::vector<uint32_t> grid = LoadNpyU32(d + "rand_k" + kb + "_w.npy", {K * N * KB / 32});
      for (int parts : {1, 2})
        for (int s : (parts == 2 ? std::vector<int>{0, 4} : std::vector<int>{0})) {
          const std::vector<uint16_t> a = LoadNpyF16Bits(d + "rand_a_s" + std::to_string(s) + ".npy", {2, 64, K});
          const std::string p = std::to_string(parts);
          const std::vector<float> ya =
              LoadNpyF32(d + "rand_k" + kb + "_ya_p" + p + "_s" + std::to_string(s) + ".npy", {64, N});
          const std::vector<float> y = LoadNpyF32(d + "rand_k" + kb + "_y_p" + p + ".npy", {64, N});
          GoldenLinear("rand", grid, K, N, KB, parts, a, svh, ya, y, s, tunings);
        }
    }
  }
  const int K = 256;
  const std::vector<Tuning> tunings = {{4, 2, 1, 1, 1, 2, 1}, {1, 2, 1, 1, 2, 2, 1}, {2, 1, 1, 4, 1, 1, 1}};
  for (const char* c : {"attn_k", "mlp_down", "mlp_gate_up"}) {
    const bool two = std::string(c) == "mlp_gate_up";
    const int N = two ? 512 : 256, parts = two ? 2 : 1;
    const std::string base = d + "real_" + c;
    GoldenLinear(std::string("real ") + c, LoadNpyU32(base + "_w.npy", {K * N * 4 / 32}), K, N, 4, parts,
                 LoadNpyF16Bits(base + "_a.npy", {parts, 64, K}), LoadNpyF16Bits(base + "_svh.npy", {N}),
                 LoadNpyF32(base + "_ya.npy", {64, N}), LoadNpyF32(base + "_y.npy", {64, N}), 0, tunings);
  }
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937_64 rng(20260928);

  std::printf("trellis linear: input transform + GEMM + epilogue (docs/trellis-kernel.md 4.3-4.5, M2)\n");
  TestPreconditions();
  TestTicketReset(rng);
  TestAccuracy(rng);
  TestEpilogueExact(rng);
  TestInvariance(rng);
  TestRowIdentity(rng);
  TestPrefillRows(rng);
  TestDeterminism(rng);

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
