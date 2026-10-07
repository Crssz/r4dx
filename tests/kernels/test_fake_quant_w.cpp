// tests/kernels/test_fake_quant_w.cpp -- R4DX_FAKEQ_W's kernels (docs/int8-prefill.md): libr4d's `_wq` trellis
// GEMMs, which round every decoded weight fragment to symmetric int8 and back before the WMMA, against
// src/model/fake_quant_w.h's CPU reference. BUILT FOR THE GPU AND NOT RUN BY THE AUTHOR OF THE BRANCH (the
// session that wrote it had no GPU to spare); the command that runs it is in docs/int8-prefill.md.
//
//   A  r4d_trellis_wscale_f32 (the scale-table builder, which decodes with the GEMM's own decode) bit for bit
//      against FakeQuantWTableRef on the CPU decode of random trellis words (any 32-bit word is a valid ring),
//      KB 4 and 5, 128-k and 32-k groups, K = 1024 and 1152;
//   B  r4d_gemm_trellis_nt_m64_raw_wq: one-hot A rows return rows of the ROUNDED weight Q' bit for bit (the
//      raw entry sums in fp32 with a single non-zero product), for every (NP, U, MT) instantiation, unsplit and
//      split, one part and two, K = 1024 (even step counts) and 1152 (9 steps, the odd tail), KB 4 and 5,
//      both group widths -- against f16(FakeQuantWApplyRef(CPU decode)); the instantiation counts are
//      checked (KB 4: 25, KB 5: 24; NT does not select a kernel in the _wq unit);
//   C  r4d_gemm_trellis_nt_m256_wq against four 64-row r4d_gemm_trellis_nt_m64_wq launches of the shipped
//      M = 64 tuning row, byte for byte, on the model's trellis linear classes at both rates and both group
//      widths (as tests/kernels/test_trellis_m256.cpp does for the shipped kernels), repeatability of the
//      M = 256 launch, tickets back at zero, and: the rounding changes the output (the stock kernel's bytes
//      differ), so the comparison is known to see the switch;
//   D  the whole linear with rounding: r4d_gemm_trellis_nt_m64_wq's bf16 output equals the CPU epilogue of
//      the _raw_wq sums bit for bit, and the _raw_wq sums agree with an fp64 dot product against Q' within the
//      fp32 summation bound;
//   E  the guards: a null table, a bad group shift, and a null table through the M = 256 entry all throw.
// Skips (77) without a HIP device. HIP device 1 via HIP_VISIBLE_DEVICES=1 (tests/kernels/CMakeLists.txt); a
// few seconds to a minute.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "fake_quant_w.h"
#include "linear.h"
#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "trellis_ref.hpp"

using namespace r4dx::core;
using r4dx::model::Layout;
using r4dx::model::LinearTuning;

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    if (g_failures < 60) std::printf("FAIL: %s\n", what.c_str());
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

std::vector<uint32_t> RandomWords(std::mt19937_64& rng, size_t n) {
  std::vector<uint32_t> w(n);
  for (auto& x : w) x = static_cast<uint32_t>(rng());
  return w;
}

constexpr int kModes[2] = {r4dx::model::kFakeQuantWCol128, r4dx::model::kFakeQuantWCol32};

// One random trellis weight: pair-grid words on the device, the CPU decode, and per group width the reference
// scale table (also uploaded, so the GEMM tests do not depend on the GPU builder) and the rounded Q' as f16 bits.
struct Weight {
  int K, N, KB;
  std::vector<uint32_t> grid;
  std::vector<float> qf;  // CPU decode, [K][N], f16 widened
  DeviceBuffer<uint32_t> w;
  struct Ref {
    int gsh = 0;
    std::vector<float> table;
    std::vector<uint16_t> q16;  // Q' = f16(round(Q)), [K][N]
    DeviceBuffer<float> dtable;
  };
  std::unique_ptr<Ref> ref[2];  // [0] col128 (gsh 3), [1] col32 (gsh 1)

  Weight(std::mt19937_64& rng, int K_, int N_, int KB_, bool want_cpu_ref) : K(K_), N(N_), KB(KB_) {
    std::vector<uint32_t> words = RandomWords(rng, static_cast<size_t>(K / 16) * (N / 16) * 8 * KB);
    // Any 32-bit word is a valid ring, so without a CPU reference the random words are the pair grid as they are.
    grid = want_cpu_ref ? trellis_ref::ToPairGrid(words, K, N, KB) : std::move(words);
    w = DeviceBuffer<uint32_t>(grid.size());
    w.CopyFromHost(grid);
    if (!want_cpu_ref) return;
    const std::vector<uint16_t> q16 = trellis_ref::DecodePairGrid(grid, K, N, KB);
    qf.resize(q16.size());
    for (size_t i = 0; i < q16.size(); ++i) qf[i] = F16ToFloat(q16[i]);
    for (int i = 0; i < 2; ++i) {
      auto r = std::make_unique<Ref>();
      r->gsh = r4dx::model::FakeQuantWGroupShift(kModes[i]);
      const int group = r4dx::model::FakeQuantWGroup(kModes[i]);
      r->table = r4dx::model::FakeQuantWTableRef(qf.data(), K, N, group);
      const std::vector<float> applied = r4dx::model::FakeQuantWApplyRef(qf.data(), K, N, group, r->table);
      r->q16.resize(applied.size());
      for (size_t j = 0; j < applied.size(); ++j) r->q16[j] = FloatToF16(applied[j]);
      r->dtable = DeviceBuffer<float>(r->table.size());
      r->dtable.CopyFromHost(r->table);
      ref[i] = std::move(r);
    }
  }
};

// ---- A: the scale-table builder --------------------------------------------------------------------------------

void TestScaleTable(std::mt19937_64& rng) {
  int cases = 0;
  for (int KB : {4, 5})
    for (int K : {1024, 1152}) {
      const int N = 512;
      Weight wt(rng, K, N, KB, /*want_cpu_ref=*/true);
      for (int i = 0; i < 2; ++i) {
        const Weight::Ref& r = *wt.ref[i];
        const size_t count = r4d_trellis_wscale_count(K, N, r.gsh);
        Check(count == r.table.size(), "wscale_count K=" + std::to_string(K) + " gsh=" + std::to_string(r.gsh));
        DeviceBuffer<float> d(count);
        R4DX_HIP_CHECK(hipMemset(d.data(), 0xFF, d.bytes()));  // NaN: an unwritten slot cannot match
        r4d_trellis_wscale_f32(P(wt.w.data()), P(d.data()), K, N, KB, r.gsh, 0);
        R4DX_HIP_CHECK(hipDeviceSynchronize());
        const std::vector<float> got = d.CopyToHost();
        size_t bad = 0;
        for (size_t j = 0; j < count; ++j) {
          uint32_t a, b;
          std::memcpy(&a, &got[j], 4);
          std::memcpy(&b, &r.table[j], 4);
          bad += a != b;
        }
        Check(bad == 0, "scale table KB=" + std::to_string(KB) + " K=" + std::to_string(K) + " gsh=" +
                            std::to_string(r.gsh) + ": " + std::to_string(bad) + " of " +
                            std::to_string(count) + " scales differ from the CPU reference");
        ++cases;
      }
    }
  std::printf("  A scale table: %d (KB, K, group) cases bit-exact against the CPU reference\n", cases);
}

// ---- B: the raw entry, one-hot rows ----------------------------------------------------------------------------

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

struct RawRig {
  Weight& wt;
  DeviceBuffer<uint16_t> a0, a1;  // [64][K] f16
  DeviceBuffer<float> c;          // [64][N]
  DeviceBuffer<float> ws;         // [8][64][N]
  DeviceBuffer<uint32_t> tickets;
  explicit RawRig(Weight& w_)
      : wt(w_), a0(static_cast<size_t>(64) * w_.K), a1(static_cast<size_t>(64) * w_.K),
        c(static_cast<size_t>(64) * w_.N), ws(static_cast<size_t>(8) * 64 * w_.N),
        tickets(static_cast<size_t>(w_.N / 128)) {
    tickets.Zero();
  }
  void Run(const Tuning& t, int M, int n_split, bool two_parts, int mode) {
    const Weight::Ref& r = *wt.ref[mode];
    r4d_gemm_trellis_nt_m64_raw_wq(P(a0.data()), two_parts ? P(a1.data()) : 0, n_split, P(wt.w.data()),
                                   P(c.data()), P(ws.data()), P(tickets.data()), M, wt.K, wt.N, wt.KB, t.WV, t.SK,
                                   t.MT, t.NP, t.SKG, t.U, t.NT, P(r.dtable.data()), r.gsh, 0);
  }
};

constexpr uint16_t kOneF16 = 0x3C00;
constexpr uint32_t kSentinel = 0x7FC0DEADu;  // a NaN no sum can produce

std::vector<uint16_t> OneHotA(int K, const std::vector<int>& ks) {
  std::vector<uint16_t> a(static_cast<size_t>(64) * K, 0);
  for (size_t m = 0; m < ks.size(); ++m) a[m * K + ks[m]] = kOneF16;
  return a;
}

size_t RunOneHotCall(RawRig& g, const Tuning& t, int M, const std::vector<int>& k0, const std::vector<int>& k1,
                     bool two_parts, int n_split, int mode) {
  const int N = g.wt.N;
  const Weight::Ref& r = *g.wt.ref[mode];
  std::vector<uint32_t> sentinel(static_cast<size_t>(64) * N, kSentinel);
  R4DX_HIP_CHECK(hipMemcpy(g.c.data(), sentinel.data(), sentinel.size() * 4, hipMemcpyHostToDevice));
  g.Run(t, M, n_split, two_parts, mode);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<float> c = g.c.CopyToHost();
  size_t bad = 0;
  for (int m = 0; m < 64; ++m)
    for (int n = 0; n < N; ++n) {
      uint32_t got;
      std::memcpy(&got, &c[static_cast<size_t>(m) * N + n], 4);
      if (m >= M) {
        bad += got != kSentinel;
        continue;
      }
      const int k = (two_parts && n >= n_split) ? k1[m] : k0[m];
      // value comparison: a one-hot sum 0 + 1 * (-0) is +0, and -0 == +0 here is right
      bad += c[static_cast<size_t>(m) * N + n] != F16ToFloat(r.q16[static_cast<size_t>(k) * N + n]);
    }
  if (t.Split())
    for (uint32_t x : g.tickets.CopyToHost()) bad += x != 0;
  return bad;
}

void FullCoverage(RawRig& g, const Tuning& t, bool two_parts, int n_split, int mode, const std::string& label) {
  const int M = t.Split() ? 16 * t.MT : 64;
  size_t bad = 0;
  for (int kb = 0; kb < g.wt.K; kb += M) {
    std::vector<int> k0(M), k1(M);
    for (int m = 0; m < M; ++m) {
      k0[m] = (kb + m) % g.wt.K;
      k1[m] = (kb + m + g.wt.K / 2 + 5) % g.wt.K;  // a different row of Q' for the second part
    }
    g.a0.CopyFromHost(OneHotA(g.wt.K, k0));
    g.a1.CopyFromHost(OneHotA(g.wt.K, k1));
    bad += RunOneHotCall(g, t, M, k0, k1, two_parts, n_split, mode);
  }
  Check(bad == 0, label + " " + t.Str() + ": " + std::to_string(bad) + " elements differ from the rounded Q'");
}

void TestRawFullCoverage(std::mt19937_64& rng) {
  const int N = 512;
  for (int KB : {4, 5}) {
    Weight even(rng, 1024, N, KB, true), odd(rng, 1152, N, KB, true);
    RawRig re(even), ro(odd);
    int kernels = 0;
    const std::string kb = " KB=" + std::to_string(KB);
    for (int NP : {1, 2, 4})
      for (int U : {1, 2, 4})
        for (int MT = 1; MT <= 4; ++MT) {
          const int NT = (NP + U + MT) & 1;  // ignored by the _wq unit; both values must give the same bytes
          const int WV = 4 / NP;
          const Tuning unsplit{WV, 2, MT, NP, 1, U, NT}, split{1, 2, MT, NP, 2, U, NT};
          const Tuning odd_unsplit{WV, 8 / U, MT, NP, 1, U, NT}, odd_split{1, 4 / U, MT, NP, 2, U, NT};
          if (Throws([&] { re.Run(unsplit, 1, N, false, 0); })) continue;
          R4DX_HIP_CHECK(hipDeviceSynchronize());
          ++kernels;
          const int mode = (NP + MT) & 1;  // the group width alternates over the instantiations
          FullCoverage(re, unsplit, (NT + MT) % 2 == 1, N / 2, mode, "wq full coverage K=1024" + kb);
          FullCoverage(re, split, (NT + MT) % 2 == 0, N / 2, 1 - mode, "wq full coverage K=1024" + kb);
          FullCoverage(ro, odd_unsplit, (NT + MT) % 2 == 0, N / 2, mode, "wq full coverage K=1152 (9 steps)" + kb);
          FullCoverage(ro, odd_split, (NT + MT) % 2 == 1, N / 2, 1 - mode, "wq full coverage K=1152 (9 steps)" + kb);
        }
    std::printf("  B raw one-hot KB=%d N=%d, K=1024 and K=1152 (9 steps), both group widths: %d instantiations x "
                "{unsplit, split}\n", KB, N, kernels);
    const int want = KB == 4 ? 25 : 24;
    Check(kernels == want, "wq full coverage KB " + std::to_string(KB) + ": expected " + std::to_string(want) +
                               " instantiated kernels, ran " + std::to_string(kernels));
  }
}

// ---- C: the M = 256 kernel against four 64-row launches ----------------------------------------------------------

struct Cls {
  const char* name;
  int N, K, parts, part_n0;
};
const Cls kClasses[] = {
    {"mlp.gate_up", 34816, 5120, 2, 17408},
    {"mlp.down", 5120, 17408, 1, 0},
    {"gdn.in_proj_qkv", 10240, 5120, 1, 0},
    {"attn.k/attn.v", 1024, 5120, 1, 0},
};
constexpr float kOutScale = 0.0883883f;  // 1 / sqrt(128), prescale 0

void TestM256(std::mt19937_64& rng) {
  int cases = 0;
  for (const Cls& c : kClasses)
    for (int kb : {4, 5}) {
      const int M = 256, N = c.N, K = c.K;
      const LinearTuning t = r4dx::model::PickTuning(Layout::kTrellis, N, K, 64, kb);
      const r4dx::model::TrellisM256Plan plan = r4dx::model::PlanTrellisM256(N, K, kb, c.parts, c.part_n0);
      const std::string tag = std::string(c.name) + " KB" + std::to_string(kb);
      Check(plan.ok, tag + ": no M = 256 plan (" + plan.why + ")");
      if (!plan.ok) continue;
      Weight wt(rng, K, N, kb, /*want_cpu_ref=*/false);
      std::vector<uint16_t> ha(static_cast<size_t>(2) * M * K);
      for (uint16_t& x : ha) {
        const uint32_t h = static_cast<uint32_t>(rng());
        x = static_cast<uint16_t>((h & 0x8000u) | (0x3000u + ((h >> 4) & 0x0BFFu)));  // finite f16, |x| in [2^-3, 1)
      }
      std::vector<float> hs(static_cast<size_t>(N));
      for (float& x : hs) {
        const uint32_t h = static_cast<uint32_t>(rng());
        const float m = 0.5f + static_cast<float>(h & 0xFFFFu) / 65536.f;
        x = (h & 0x10000u) ? -m : m;
      }
      DeviceBuffer<uint16_t> da(ha.size()), dc0(static_cast<size_t>(M) * N), dc1(static_cast<size_t>(M) * N),
          dcs(static_cast<size_t>(M) * N);
      DeviceBuffer<float> dsvh(hs.size());
      DeviceBuffer<float> ws0(r4dx::core::r4d::GemmTrellisWsBytes(64, N, t.SKG) / sizeof(float));
      DeviceBuffer<float> ws1(r4dx::core::r4d::GemmTrellisM256WsBytes(M, N, plan.SKG) / sizeof(float));
      DeviceBuffer<unsigned> tk0(static_cast<size_t>(N / 128)), tk1(static_cast<size_t>(N / 128));
      da.CopyFromHost(ha);
      dsvh.CopyFromHost(hs);
      tk0.Zero();
      tk1.Zero();
      struct Form {
        const char* name;
        int n_split;
        const uint16_t* a1;
      };
      std::vector<Form> forms = {{"one-A", N, nullptr}};
      if (c.parts > 1) forms.push_back({"two-part", c.part_n0, da.data() + static_cast<size_t>(M) * K});
      for (int mi = 0; mi < 2; ++mi) {
        const int gsh = r4dx::model::FakeQuantWGroupShift(kModes[mi]);
        DeviceBuffer<float> dtab(r4d_trellis_wscale_count(K, N, gsh));
        r4d_trellis_wscale_f32(P(wt.w.data()), P(dtab.data()), K, N, kb, gsh, 0);
        for (const Form& f : forms) {
          const std::string what = tag + " " + f.name + " gsh" + std::to_string(gsh);
          R4DX_HIP_CHECK(hipMemset(dc0.data(), 0xFF, dc0.bytes()));
          R4DX_HIP_CHECK(hipMemset(dc1.data(), 0xFF, dc1.bytes()));
          R4DX_HIP_CHECK(hipMemset(dcs.data(), 0xFF, dcs.bytes()));
          for (int m0 = 0; m0 < M; m0 += 64) {
            r4dx::core::r4d::GemmTrellisNtM64Wq(
                da.data() + static_cast<size_t>(m0) * K, f.a1 ? f.a1 + static_cast<size_t>(m0) * K : nullptr, f.n_split,
                wt.w.data(), dsvh.data(), dc0.data() + static_cast<size_t>(m0) * N, ws0.data(), tk0.data(), 64, K, N,
                kb, t.WV, t.SK, t.MB, t.NPW, t.SKG, t.U, t.NT, kOutScale, dtab.data(), gsh, nullptr);
            r4dx::core::r4d::GemmTrellisNtM64(
                da.data() + static_cast<size_t>(m0) * K, f.a1 ? f.a1 + static_cast<size_t>(m0) * K : nullptr, f.n_split,
                wt.w.data(), dsvh.data(), dcs.data() + static_cast<size_t>(m0) * N, ws0.data(), tk0.data(), 64, K, N,
                kb, t.WV, t.SK, t.MB, t.NPW, t.SKG, t.U, t.NT, kOutScale, nullptr);
          }
          const auto big = [&] {
            r4dx::core::r4d::GemmTrellisNtM256Wq(da.data(), f.a1, f.n_split, wt.w.data(), dsvh.data(), dc1.data(),
                                                 ws1.data(), tk1.data(), M, K, N, kb, plan.SK, 1, plan.SKG, 4,
                                                 kOutScale, plan.SKW, dtab.data(), gsh, nullptr);
          };
          big();
          R4DX_HIP_CHECK(hipDeviceSynchronize());
          const std::vector<uint16_t> ref = dc0.CopyToHost(), got = dc1.CopyToHost(), stock = dcs.CopyToHost();
          size_t bad = 0, differs_from_stock = 0;
          for (size_t j = 0; j < ref.size(); ++j) {
            bad += ref[j] != got[j];
            differs_from_stock += ref[j] != stock[j];
          }
          Check(bad == 0, what + ": the M = 256 _wq launch differs from four 64-row _wq launches in " +
                              std::to_string(bad) + " of " + std::to_string(ref.size()) + " outputs");
          Check(differs_from_stock > ref.size() / 100,
                what + ": the rounding barely changed the output (" + std::to_string(differs_from_stock) +
                    " outputs differ from the shipped kernel's): the comparison cannot see the switch");
          for (int rep = 0; rep < 3; ++rep) {  // the running-sum handoff must not race
            R4DX_HIP_CHECK(hipMemset(dc1.data(), 0xFF, dc1.bytes()));
            big();
            R4DX_HIP_CHECK(hipDeviceSynchronize());
            Check(dc1.CopyToHost() == got, what + ": the M = 256 _wq launch is not repeatable");
          }
          size_t tickets_left = 0;
          for (unsigned x : tk0.CopyToHost()) tickets_left += x != 0;
          for (unsigned x : tk1.CopyToHost()) tickets_left += x != 0;
          Check(tickets_left == 0, what + ": tickets not back at zero");
          ++cases;
        }
      }
    }
  std::printf("  C M = 256 _wq vs four 64-row _wq launches: %d (class, KB, group, form) cases byte-identical\n", cases);
}

// ---- D: the whole linear with rounding ---------------------------------------------------------------------------

void TestLinear(std::mt19937_64& rng) {
  const int K = 5120, N = 1024, KB = 4, M = 64;
  Weight wt(rng, K, N, KB, true);
  std::uniform_real_distribution<float> ud(-1.f, 1.f);
  std::vector<uint16_t> ha(static_cast<size_t>(M) * K);
  for (auto& x : ha) x = FloatToF16(ud(rng));
  std::vector<float> hs(static_cast<size_t>(N));
  for (float& x : hs) x = 0.5f + std::fabs(ud(rng));
  DeviceBuffer<uint16_t> da(ha.size()), dc(static_cast<size_t>(M) * N);
  DeviceBuffer<float> draw(static_cast<size_t>(M) * N), dsvh(hs.size()), ws(static_cast<size_t>(8) * M * N);
  DeviceBuffer<uint32_t> tk(static_cast<size_t>(N / 128));
  da.CopyFromHost(ha);
  dsvh.CopyFromHost(hs);
  tk.Zero();
  const Tuning tunings[] = {{4, 2, 4, 1, 1, 4, 0}, {1, 4, 4, 1, 2, 2, 0}};  // unsplit; split (Wc 32, SKG 2)
  int cases = 0;
  for (int mi = 0; mi < 2; ++mi) {
    const Weight::Ref& r = *wt.ref[mi];
    for (const Tuning& t : tunings) {
      const std::string what = "linear gsh" + std::to_string(r.gsh) + " " + t.Str();
      R4DX_HIP_CHECK(hipMemset(draw.data(), 0xFF, draw.bytes()));
      r4d_gemm_trellis_nt_m64_raw_wq(P(da.data()), 0, N, P(wt.w.data()), P(draw.data()), P(ws.data()), P(tk.data()),
                                     M, K, N, KB, t.WV, t.SK, t.MT, t.NP, t.SKG, t.U, t.NT, P(r.dtable.data()), r.gsh,
                                     0);
      r4d_gemm_trellis_nt_m64_wq(P(da.data()), 0, N, P(wt.w.data()), P(dsvh.data()), P(dc.data()), P(ws.data()),
                                 P(tk.data()), M, K, N, KB, t.WV, t.SK, t.MT, t.NP, t.SKG, t.U, t.NT, kOutScale,
                                 P(r.dtable.data()), r.gsh, 0);
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      const std::vector<float> sums = draw.CopyToHost();
      const std::vector<uint16_t> want = trellis_ref::EpilogueFromRaw(sums, M, N, hs.data(), kOutScale);
      const std::vector<uint16_t> got = dc.CopyToHost();
      size_t bad = 0;
      for (size_t j = 0; j < want.size(); ++j) bad += want[j] != got[j];
      Check(bad == 0, what + ": the linear's bf16 output differs from the CPU epilogue of the _raw_wq sums in " +
                          std::to_string(bad) + " elements");
      // the sums against fp64 with Q' (rows 0..3): the fp32 summation bound
      const int n_adds = K / 16 + 16 + t.SK + t.SKG;
      size_t out_of_bound = 0;
      for (int m = 0; m < 4; ++m)
        for (int n = 0; n < N; ++n) {
          double ref = 0.0, mag = 0.0;
          for (int k = 0; k < K; ++k) {
            const double p = static_cast<double>(F16ToFloat(ha[static_cast<size_t>(m) * K + k])) *
                             F16ToFloat(r.q16[static_cast<size_t>(k) * N + n]);
            ref += p;
            mag += std::fabs(p);
          }
          const double lim = 2.0 * n_adds * std::ldexp(1.0, -24) * mag + 1e-30;
          out_of_bound += std::fabs(static_cast<double>(sums[static_cast<size_t>(m) * N + n]) - ref) > lim;
        }
      Check(out_of_bound == 0, what + ": " + std::to_string(out_of_bound) +
                                   " raw sums are outside the fp32 summation bound of the fp64 dot product with Q'");
      ++cases;
    }
  }
  std::printf("  D whole linear: %d (group, tuning) cases: bf16 output = CPU epilogue of the raw sums, sums within "
              "the fp64 bound\n", cases);
}

// ---- E: guards -----------------------------------------------------------------------------------------------------

void TestGuards() {
  const int K = 1024, N = 512, KB = 4;
  Check(Throws([&] {
          r4d_gemm_trellis_nt_m64_raw_wq(16, 0, N, 16, 16, 16, 16, 1, K, N, KB, 4, 2, 1, 1, 1, 1, 0, /*wscale=*/0, 3, 0);
        }),
        "a null table through the M <= 64 raw entry did not throw");
  Check(Throws([&] {
          r4d_gemm_trellis_nt_m64_wq(16, 0, N, 16, 16, 16, 16, 16, 1, K, N, KB, 4, 2, 1, 1, 1, 1, 0, 1.f,
                                     /*wscale=*/0, 3, 0);
        }),
        "a null table through the M <= 64 entry did not throw");
  for (int gsh : {0, 4, -1}) {
    Check(Throws([&] {
            r4d_gemm_trellis_nt_m64_raw_wq(16, 0, N, 16, 16, 16, 16, 1, K, N, KB, 4, 2, 1, 1, 1, 1, 0, 16, gsh, 0);
          }),
          "group shift " + std::to_string(gsh) + " through the raw entry did not throw");
  }
  Check(Throws([&] {
          r4d_gemm_trellis_nt_m256_wq(16, 0, N, 16, 16, 16, 16, 16, 256, 5120, 5120, 4, 4, 1, 1, 4, 0.1f, 0, 4,
                                      /*wscale=*/0, 3);
        }),
        "a null table through the M = 256 entry did not throw");
  Check(Throws([&] { r4d_trellis_wscale_f32(16, 16, K, 48, KB, 3, 0); }), "wscale_f32 with N % 32 != 0 did not throw");
  Check(Throws([&] { r4d_trellis_wscale_f32(16, 16, K, N, KB, 0, 0); }), "wscale_f32 with gsh 0 did not throw");
  Check(Throws([&] { r4d_trellis_wscale_f32(16, 16, K, N, 3, 3, 0); }), "wscale_f32 with KB 3 did not throw");
  Check(Throws([&] { r4d_trellis_wscale_f32(0, 16, K, N, KB, 3, 0); }), "wscale_f32 with a null weight did not throw");
  Check(r4d_trellis_wscale_count(5120, 5120, 3) == static_cast<size_t>(5120 / 128) * 5120 &&
            r4d_trellis_wscale_count(5120, 5120, 1) == static_cast<size_t>(5120 / 32) * 5120,
        "wscale_count");
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices < 1) {
    std::printf("SKIP: no HIP device\n");
    return 77;
  }
  if (hipSetDevice(0) != hipSuccess) {
    std::printf("SKIP: hipSetDevice(0) failed\n");
    return 77;
  }
  std::mt19937_64 rng(20261007);
  std::printf("R4DX_FAKEQ_W kernels (docs/int8-prefill.md)\n");
  TestGuards();
  TestScaleTable(rng);
  TestRawFullCoverage(rng);
  TestLinear(rng);
  TestM256(rng);
  if (g_failures != 0) {
    std::printf("test_fake_quant_w: %d of %d checks FAILED\n", g_failures, g_checks);
    return 1;
  }
  std::printf("test_fake_quant_w: PASS (%d checks)\n", g_checks);
  return 0;
}