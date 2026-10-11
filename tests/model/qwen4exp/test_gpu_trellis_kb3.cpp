// tests/model/qwen4exp/test_gpu_trellis_kb3.cpp -- trellis KB = 3 through ApplyLinear
// (docs/qwen4exp-plan.md 4.3 / 4.11 item 6; docs/trellis-kernel.md "KB = 3").
//
// Random packed ring words (pair grid) and random suh / svh form a QuantLinear of layout kTrellis, which
// is run through r4dx::model::ApplyLinear -- the production path: input transform, tuning pick
// (FallbackTuning at KB 3: the shipped table has no KB 3 rows), r4d_gemm_trellis_nt_m64 -- for the shapes
// qwen4_exp uses, at M in {1, 2, 8, 16, 17, 64}, against the CPU reference of tests/kernels/trellis_ref.hpp
// (CPU decode of the same words, the transform's own arithmetic, fp64 linear of the f16 A the GEMM was
// given): every element within 4 bf16 ulp + 1e-4 rms(row), as test_trellis_linear. The max abs error, the
// worst error as a fraction of the tolerance and the mean abs error are printed. Shapes (N, K):
//   expert gate_up (2 parts of 640, N = 1280, K = 2560), expert down (2560, 640), attn.qg (12288, 2560),
//   gdn.in_proj_qkv (10240, 2560), gdn.in_proj_z (6144, 2560), gdn.out_proj / attn.o (2560, 6144),
//   attn.k / v (512, 2560).
// Also, at KB 3: r4d_trellis_reconstruct_f16 (what verify uses) equals the CPU decode bit for bit on every
// shape; rows of an M <= 16 call equal the M = 1 call on that row (the verify-equals-decode identity);
// PlanTrellisM256 / PlanTrellisI8 refuse rate 3 without throwing (the arch never opens those scopes).
// KB 4 and 5 run the same shapes (M in {1, 17, 64}) with the same tolerance, and print an FNV-1a digest of
// their outputs, for comparison across a libr4d change (their kernels' listings are byte-identical to the
// pre-KB-3 libr4d's, which the change's report checks separately).
//
// R4DX_KB3_BENCH=1 adds the measurement of the plan's item 6: the M = 1 GEMM time of the expert gate_up
// (N 1280, K 2560) and down (N 2560, K 640) at KB 3 against KB 4 (and 5), the shipped fallback tuning and
// the best of a tuning sweep, with the weights cycled through 96 copies (~100-150 MB: DRAM, not cache,
// like 512 experts) and, for the cache-warm case, a single copy. Printed only.
//
// GPU test: HIP device 1 via HIP_VISIBLE_DEVICES=1 (tests/model/qwen4exp/CMakeLists.txt).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "linear.h"
#include "quant_linear.h"
#include "r4d.h"
#include "r4dx/core/arena.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/stream.hpp"
#include "trellis_ref.hpp"

using r4dx::core::Bf16ToFloat;
using r4dx::core::DeviceBuffer;
using r4dx::core::FloatToBf16;
using r4dx::model::ApplyLinear;
using r4dx::model::Layout;
using r4dx::model::QuantLinear;

namespace {

int g_checks = 0, g_failures = 0;

void Check(bool ok, const std::string& what) {
  ++g_checks;
  if (ok) return;
  ++g_failures;
  if (g_failures <= 40) std::fprintf(stderr, "[FAIL] %s\n", what.c_str());
}

template <typename T>
int64_t P(const T* p) {
  return reinterpret_cast<int64_t>(p);
}

struct Shape {
  const char* name;
  int64_t N, K;
  int parts;
  int64_t part0;   // output columns of part 0 (== N for one part)
};

const Shape kShapes[] = {
    {"expert gate_up", 1280, 2560, 2, 640}, {"expert down", 2560, 640, 1, 2560},
    {"attn.qg", 12288, 2560, 1, 12288},     {"gdn.in_proj_qkv", 10240, 2560, 1, 10240},
    {"gdn.in_proj_z", 6144, 2560, 1, 6144}, {"gdn.out_proj / attn.o", 2560, 6144, 1, 2560},
    {"attn.k / v", 512, 2560, 1, 512},
};

// A trellis linear of random words and scales, with the CPU decode of its words.
struct Lin {
  Shape s;
  int KB;
  QuantLinear q;
  std::vector<uint32_t> grid;
  std::vector<uint16_t> qdec;        // decoded Q [K][N] f16 bits (CPU)
  std::vector<float> suh, svh;       // [parts][K], [N]
  DeviceBuffer<uint32_t> tickets;
};

std::unique_ptr<Lin> MakeLin(std::mt19937_64& rng, const Shape& s, int KB, bool decode = true) {
  auto L = std::make_unique<Lin>();
  L->s = s;
  L->KB = KB;
  L->grid.resize(static_cast<size_t>(s.N) * s.K * KB / 32);
  for (auto& w : L->grid) w = static_cast<uint32_t>(rng());
  if (decode) L->qdec = trellis_ref::DecodePairGrid(L->grid, s.K, s.N, KB);
  std::uniform_real_distribution<float> sd(5e-3f, 4e-2f), vd(0.6f, 1.9f);
  L->suh.resize(static_cast<size_t>(s.parts) * s.K);
  for (auto& v : L->suh) v = sd(rng);
  L->svh.resize(static_cast<size_t>(s.N));
  for (auto& v : L->svh) v = vd(rng);
  QuantLinear& q = L->q;
  q.layout = Layout::kTrellis;
  q.N = s.N;
  q.K = s.K;
  q.trellis_w = DeviceBuffer<uint32_t>(L->grid.size());
  q.trellis_w.CopyFromHost(L->grid);
  q.trellis_suh = DeviceBuffer<float>(L->suh.size());
  q.trellis_suh.CopyFromHost(L->suh);
  q.trellis_svh = DeviceBuffer<float>(L->svh.size());
  q.trellis_svh.CopyFromHost(L->svh);
  L->tickets = DeviceBuffer<uint32_t>(static_cast<size_t>(s.N / 128));
  L->tickets.Zero();
  q.trellis_tickets = L->tickets.data();
  q.trellis_bits = KB;
  q.trellis_parts = s.parts;
  q.trellis_part_n[0] = s.parts > 1 ? s.part0 : s.N;
  q.trellis_part_n[1] = s.parts > 1 ? s.N - s.part0 : 0;
  q.trellis_prescale_log2 = 0;
  return L;
}

// Activations as trellis_golden.py's random_x (the generator of test_trellis_linear / test_trellis_gemm).
std::vector<uint16_t> RandomX(std::mt19937_64& rng, int64_t M, int64_t K) {
  std::normal_distribution<float> nd(0.f, 1.f);
  std::uniform_real_distribution<float> ud(-8.f, 3.f), od(20.f, 60.f);
  std::uniform_int_distribution<int64_t> kd(0, K - 1);
  std::vector<uint16_t> x(static_cast<size_t>(M * K));
  std::vector<float> row(static_cast<size_t>(K));
  for (int64_t m = 0; m < M; ++m) {
    const float sc = std::exp2(ud(rng));
    for (auto& v : row) v = nd(rng) * sc;
    for (int i = 0; i < 2; ++i) row[static_cast<size_t>(kd(rng))] *= od(rng);
    for (int64_t k = 0; k < K; ++k) x[static_cast<size_t>(m * K + k)] = FloatToBf16(row[static_cast<size_t>(k)]);
  }
  return x;
}

constexpr uint16_t kSentinel = 0x7FC1u;

std::vector<uint16_t> Run(hipStream_t s, r4dx::core::Arena& arena, const QuantLinear& q,
                          const DeviceBuffer<uint16_t>& x, int64_t M) {
  DeviceBuffer<uint16_t> y(static_cast<size_t>(M * q.N));
  y.CopyFromHost(std::vector<uint16_t>(y.size(), kSentinel));
  arena.Reset();
  ApplyLinear(s, arena, q, x.data(), y.data(), M);
  R4DX_HIP_CHECK(hipStreamSynchronize(s));
  return y.CopyToHost();
}

uint64_t Fnv(uint64_t h, const std::vector<uint16_t>& v) {
  for (uint16_t x : v) {
    h ^= x & 0xFF;
    h *= 1099511628211ull;
    h ^= x >> 8;
    h *= 1099511628211ull;
  }
  return h;
}

struct Err {
  size_t bad = 0;
  double worst = 0.0, max_abs = 0.0, sum_abs = 0.0, max_rel_rms = 0.0;
  size_t n = 0;
};

Err Compare(const std::vector<uint16_t>& y, const std::vector<double>& ya, int64_t M, int64_t N) {
  Err e;
  for (int64_t m = 0; m < M; ++m) {
    double ss = 0.0;
    for (int64_t n = 0; n < N; ++n) ss += ya[m * N + n] * ya[m * N + n];
    const double rms = std::sqrt(ss / static_cast<double>(N)), floor = 1e-4 * rms;
    for (int64_t n = 0; n < N; ++n) {
      const size_t i = static_cast<size_t>(m * N + n);
      const double lim = 4.0 * trellis_ref::Bf16Ulp(ya[i]) + floor;
      const double err = std::fabs(static_cast<double>(Bf16ToFloat(y[i])) - ya[i]);
      if (!(err <= lim)) ++e.bad;
      e.worst = std::max(e.worst, err / lim);
      e.max_abs = std::max(e.max_abs, err);
      e.sum_abs += err;
      e.max_rel_rms = std::max(e.max_rel_rms, rms > 0 ? err / rms : 0.0);
      ++e.n;
    }
  }
  return e;
}

// One linear, the given M list. Returns the digest of its outputs.
uint64_t CheckLinear(hipStream_t s, r4dx::core::Arena& arena, const Lin& L, const std::vector<int64_t>& Ms,
                     std::mt19937_64& rng, bool identity) {
  const Shape& sh = L.s;
  uint64_t digest = 1469598103934665603ull;
  Err all;
  for (int64_t M : Ms) {
    const std::vector<uint16_t> x = RandomX(rng, M, sh.K);
    DeviceBuffer<uint16_t> x_d(x.size());
    x_d.CopyFromHost(x);
    const std::vector<uint16_t> y = Run(s, arena, L.q, x_d, M);
    digest = Fnv(digest, y);
    std::vector<std::vector<uint16_t>> a;
    std::vector<const uint16_t*> ap;
    for (int p = 0; p < sh.parts; ++p)
      a.push_back(trellis_ref::TransformInput(x, M, sh.K, L.suh.data() + static_cast<size_t>(p) * sh.K, 0));
    for (const auto& v : a) ap.push_back(v.data());
    const std::vector<double> ya =
        trellis_ref::LinearFromA(ap, M, sh.K, sh.N, sh.parts > 1 ? sh.part0 : sh.N, L.qdec, L.svh.data(), 0);
    const Err e = Compare(y, ya, M, sh.N);
    Check(e.bad == 0, std::string(sh.name) + " KB=" + std::to_string(L.KB) + " M=" + std::to_string(M) + ": " +
                          std::to_string(e.bad) + " element(s) beyond 4 bf16 ulp (worst " +
                          std::to_string(e.worst) + " of the tolerance)");
    all.worst = std::max(all.worst, e.worst);
    all.max_abs = std::max(all.max_abs, e.max_abs);
    all.sum_abs += e.sum_abs;
    all.n += e.n;
    all.max_rel_rms = std::max(all.max_rel_rms, e.max_rel_rms);

    if (identity && M <= 16 && M > 1) {
      // verify rows == decode rows: every row of the M-row call is the M = 1 call on that row
      bool same = true;
      for (int64_t m = 0; m < M; ++m) {
        DeviceBuffer<uint16_t> row(static_cast<size_t>(sh.K));
        R4DX_HIP_CHECK(hipMemcpy(row.data(), x_d.data() + m * sh.K, static_cast<size_t>(sh.K) * 2,
                                 hipMemcpyDeviceToDevice));
        const std::vector<uint16_t> y1 = Run(s, arena, L.q, row, 1);
        same = same && std::memcmp(y1.data(), y.data() + m * sh.N, static_cast<size_t>(sh.N) * 2) == 0;
      }
      Check(same, std::string(sh.name) + " KB=3 M=" + std::to_string(M) + ": rows differ from their M = 1 calls");
    }
  }
  std::printf("  %-22s N=%5lld K=%4lld KB=%d P=%d M{%zu}: max|err| %.4g, mean|err| %.4g, max err/rms(row) %.3g, "
              "worst %.3f of the 4-ulp tolerance, digest %016llx\n",
              sh.name, static_cast<long long>(sh.N), static_cast<long long>(sh.K), L.KB, sh.parts, Ms.size(),
              all.max_abs, all.sum_abs / static_cast<double>(all.n), all.max_rel_rms, all.worst,
              static_cast<unsigned long long>(digest));
  return digest;
}

void CheckReconstruct(const Lin& L) {
  const Shape& sh = L.s;
  DeviceBuffer<uint16_t> q_d(static_cast<size_t>(sh.N * sh.K));
  R4DX_HIP_CHECK(hipMemset(q_d.data(), 0xFF, q_d.bytes()));
  r4d_trellis_reconstruct_f16(P(L.q.trellis_w.data()), P(q_d.data()), static_cast<int>(sh.K),
                              static_cast<int>(sh.N), L.KB, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<uint16_t> got = q_d.CopyToHost();
  size_t diff = 0;
  for (size_t i = 0; i < got.size(); ++i) diff += got[i] != L.qdec[i];
  std::printf("  reconstruct %-22s KB=%d %lldx%lld: %zu of %zu values differ from the CPU decode\n", sh.name, L.KB,
              static_cast<long long>(sh.K), static_cast<long long>(sh.N), diff, got.size());
  Check(diff == 0, std::string("reconstruct ") + sh.name + " KB=" + std::to_string(L.KB) + ": " +
                       std::to_string(diff) + " values differ");
}

// ---- R4DX_KB3_BENCH ---------------------------------------------------------------------------------

struct Cfg {
  int WV, SK, MT, NP, SKG, U, NT;
};

// Median per-launch time (us) of `reps` M = 1 GEMMs cycling through `copies` weight copies. Launches go
// through a captured graph of `batch` launches when the runtime allows it (the host cannot enqueue a 5 us
// kernel faster than it runs), else back to back.
double TimeGemm(hipStream_t st, const std::vector<DeviceBuffer<uint32_t>>& w, const DeviceBuffer<uint16_t>& a,
                const DeviceBuffer<float>& svh, DeviceBuffer<uint16_t>& c, DeviceBuffer<float>& ws,
                DeviceBuffer<uint32_t>& tickets, int K, int N, int n_split, int KB, const Cfg& t, bool two_parts,
                const DeviceBuffer<uint16_t>& a1) {
  const int batch = 64;
  auto launch_all = [&]() {
    for (int i = 0; i < batch; ++i)
      r4d_gemm_trellis_nt_m64(P(a.data()), two_parts ? P(a1.data()) : 0, n_split, P(w[static_cast<size_t>(i) % w.size()].data()),
                              P(svh.data()), P(c.data()), P(ws.data()), P(tickets.data()), 1, K, N, KB, t.WV, t.SK,
                              t.MT, t.NP, t.SKG, t.U, t.NT, 1.f / std::sqrt(128.f), P(st));
  };
  launch_all();   // throws when the tuning is illegal here
  R4DX_HIP_CHECK(hipStreamSynchronize(st));
  hipGraph_t g = nullptr;
  hipGraphExec_t ge = nullptr;
  bool graph = hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal) == hipSuccess;
  if (graph) {
    launch_all();
    graph = hipStreamEndCapture(st, &g) == hipSuccess && hipGraphInstantiate(&ge, g, nullptr, nullptr, 0) == hipSuccess;
  }
  hipEvent_t e0, e1;
  R4DX_HIP_CHECK(hipEventCreate(&e0));
  R4DX_HIP_CHECK(hipEventCreate(&e1));
  std::vector<double> us;
  const int iters = 25;
  for (int it = 0; it < iters + 3; ++it) {
    R4DX_HIP_CHECK(hipEventRecord(e0, st));
    if (graph) R4DX_HIP_CHECK(hipGraphLaunch(ge, st));
    else launch_all();
    R4DX_HIP_CHECK(hipEventRecord(e1, st));
    R4DX_HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0.f;
    R4DX_HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    if (it >= 3) us.push_back(1000.0 * ms / batch);
  }
  if (ge) (void)hipGraphExecDestroy(ge);
  if (g) (void)hipGraphDestroy(g);
  (void)hipEventDestroy(e0);
  (void)hipEventDestroy(e1);
  std::sort(us.begin(), us.end());
  return us[us.size() / 2];
}

void Bench(hipStream_t st, std::mt19937_64& rng) {
  struct B {
    const char* name;
    int N, K, n_split;
  };
  const B shapes[] = {{"expert gate_up (1280 x 2560, 2 parts)", 1280, 2560, 640},
                      {"expert down (2560 x 640)", 2560, 640, 2560}};
  std::printf("bench (M = 1, per-launch GPU time, median of 25 graph replays of 64 launches)\n");
  for (const B& b : shapes) {
    const bool two = b.n_split != b.N;
    DeviceBuffer<uint16_t> a(static_cast<size_t>(b.K)), a1(static_cast<size_t>(b.K)), c(static_cast<size_t>(b.N));
    std::vector<uint16_t> ah(static_cast<size_t>(b.K));
    for (auto& v : ah) v = r4dx::core::FloatToF16(std::uniform_real_distribution<float>(-1.f, 1.f)(rng));
    a.CopyFromHost(ah);
    a1.CopyFromHost(ah);
    DeviceBuffer<float> svh(static_cast<size_t>(b.N)), ws(static_cast<size_t>(8) * b.N);
    svh.CopyFromHost(std::vector<float>(static_cast<size_t>(b.N), 1.f));
    DeviceBuffer<uint32_t> tickets(static_cast<size_t>(b.N / 128));
    tickets.Zero();
    for (int KB : {3, 4, 5}) {
      const size_t words = static_cast<size_t>(b.N) * b.K * KB / 32;
      std::vector<uint32_t> h(words);
      for (auto& v : h) v = static_cast<uint32_t>(rng());
      for (int copies : {1, 96}) {
        std::vector<DeviceBuffer<uint32_t>> w;
        for (int i = 0; i < copies; ++i) {
          w.emplace_back(words);
          w.back().CopyFromHost(h);
        }
        // the shipped fallback (linear.cpp FallbackTuning at M = 1): WV 4, SK 2, MT 1, NP 1, U 2, NT 1, SKG per N
        const r4dx::model::LinearTuning ft = r4dx::model::PickTuning(Layout::kTrellis, b.N, b.K, 1, KB);
        const Cfg fb{ft.WV, ft.SK, ft.MB, ft.NPW, ft.SKG, ft.U, ft.NT};
        const double t_fb = TimeGemm(st, w, a, svh, c, ws, tickets, b.K, b.N, b.n_split, KB, fb, two, a1);
        double best = 1e30;
        Cfg bc{};
        for (int WV : {1, 2, 4})
          for (int NP : {1, 2})
            for (int SK : {1, 2, 4, 8})
              for (int SKG : {1, 2, 4})
                for (int U : {1, 2, 4}) {
                  const Cfg t{WV, SK, 1, NP, SKG, U, 1};
                  double us;
                  try {
                    us = TimeGemm(st, w, a, svh, c, ws, tickets, b.K, b.N, b.n_split, KB, t, two, a1);
                  } catch (const std::exception&) {
                    continue;
                  }
                  if (us < best) {
                    best = us;
                    bc = t;
                  }
                }
        std::printf("  %-38s KB=%d %-9s fallback (WV%d SK%d NP%d SKG%d U%d) %.2f us, best %.2f us (WV%d SK%d NP%d SKG%d "
                    "U%d), weights %.2f MB\n",
                    b.name, KB, copies == 1 ? "warm" : "cycled-96", fb.WV, fb.SK, fb.NP, fb.SKG, fb.U, t_fb, best,
                    bc.WV, bc.SK, bc.NP, bc.SKG, bc.U, static_cast<double>(words) * 4 / 1e6);
      }
    }
  }
}

int RunTest() {
  int n_devices = 0;
  if (hipGetDeviceCount(&n_devices) != hipSuccess || n_devices < 1) {
    std::printf("[SKIP] no HIP device\n");
    return 77;
  }
  R4DX_HIP_CHECK(hipSetDevice(0));
  r4dx::core::Stream stream;
  r4dx::core::Arena arena;
  arena.Reserve(256ull << 20);
  std::mt19937_64 rng(20261010);

  std::printf("trellis KB 3 through ApplyLinear (random words, qwen4_exp shapes)\n");
  Check(r4d_gemm_trellis_nt_m64_has_rate(3) == 1 && r4d_gemm_trellis_nt_m64_has_rate(4) == 1 &&
            r4d_gemm_trellis_nt_m64_has_rate(5) == 1 && r4d_gemm_trellis_nt_m64_has_rate(2) == 0,
        "has_rate");
  for (const Shape& s : kShapes) {
    // the M = 256 and int8 plans refuse rate 3 cleanly (no throw, ok == false)
    try {
      const auto p256 = r4dx::model::PlanTrellisM256(s.N, s.K, 3, s.parts, s.part0);
      Check(!p256.ok, std::string(s.name) + ": PlanTrellisM256 must refuse rate 3");
      const auto p8 = r4dx::model::PlanTrellisI8(s.N, s.K, 3, s.parts, s.part0);
      Check(!p8.ok, std::string(s.name) + ": PlanTrellisI8 must refuse rate 3");
    } catch (const std::exception& e) {
      Check(false, std::string(s.name) + ": a plan threw for rate 3: " + e.what());
    }
  }

  const std::vector<int64_t> m3 = {1, 2, 8, 16, 17, 64}, m45 = {1, 17, 64};
  for (const Shape& s : kShapes) {
    for (int KB : {3, 4, 5}) {
      const std::unique_ptr<Lin> L = MakeLin(rng, s, KB);
      if (KB == 3) CheckReconstruct(*L);
      CheckLinear(stream.get(), arena, *L, KB == 3 ? m3 : m45, rng, /*identity=*/KB == 3);
    }
  }

  if (const char* b = std::getenv("R4DX_KB3_BENCH"); b != nullptr && b[0] == '1') Bench(stream.get(), rng);

  std::printf("test_gpu_trellis_kb3: %d/%d checks passed\n", g_checks - g_failures, g_checks);
  return g_failures == 0 ? 0 : 1;
}

}  // namespace

int main() {
  try {
    return RunTest();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "test_gpu_trellis_kb3: exception: %s\n", e.what());
    return 1;
  }
}
