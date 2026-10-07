// test_trellis_i8_gemm: libr4d's int8 x int8 trellis prefill GEMM (r4d_gemm_trellis_nt_i8.hip, R4DX_PREFILL_INT8;
// docs/int8-prefill.md "Production path") on the GPU, against exact CPU models of what it computes. Built, and NOT
// RUN by the session that wrote it (CPU only): run it on device 1 with the machine idle (ctest -R trellis_i8_gemm).
//
// For every trellis linear class the model wires to it (the seven TP = 1 classes of the 27B, and the per-rank
// shapes of TP = 2, which the model does not use yet but the kernel is bit-tested at) at both rates, on synthetic
// data (any 32-bit word is a valid trellis ring; A is Gaussian):
//   A. the producers, bit for bit against the CPU rules: the weight scale table (r4d_trellis_i8_wscale) and the int8
//      weights r4d_trellis_i8_dump_w prints (block layout and plain) on three whole 128-column groups against the CPU
//      trellis decode (trellis_ref.hpp) and quantizer (int8_gemm_proto_ref.h's QuantizeWeightsRef); the activation
//      quantizer (r4d_trellis_i8_quant_act, one and two parts) on every row against QuantizeActRef + PackA8;
//   B. every legal (skw, skg) of the GEMM on every one of those shapes against the exact integer reference: the
//      int8 sums in int32 on the CPU, the per-128 rescale, the FWHT, svh and out_scale in fp64, within the bench's
//      tolerance (0.85% of the value + 1e-4 of the group's rms: one bf16 rounding and the fp32 sum order), on three
//      groups (the first, the middle, the last: the last is in the second A part of a two-part linear) x 256 rows;
//   C. an EXACT check, bit for bit: every row has a single nonzero activation, so every int32 partial sum is one
//      product and the kernel's fp32 chain is rounding-order free; the output must equal the CPU's fp32 emulation of
//      the epilogue (FWHT stages, svh, out_scale, bf16) byte for byte -- this sees a single wrong int8 weight, a
//      misplaced k, a wrong row or a wrong scale, which B's tolerance cannot;
//   D. a two-part launch is byte-identical to two single-part launches (columns below n_split from the first
//      A part's, the rest from the second's), at n_split of 128, N - 128, the class's own boundary and 8704;
//   E. three repeats give the same bytes and the tickets are back at zero after every launch;
//   F. an f16, int8, f16 interleave on one linear (one tickets buffer, one ws): the f16 bytes before and after the
//      int8 launch are equal, and the int8 bytes equal the ones of a launch without the f16 neighbours;
//   G. row independence: the same activation rows in another order give the same output rows, byte for byte.
// The same A..G again for the COARSE scales (R4DX_PREFILL_INT8_SCALES=coarse: r4d_gemm_trellis_nt_i8c, one activation
// scale per row, one weight scale per column, both over the whole K; "RunCaseCoarse" below).
// Negative controls: the tolerance check must reject a reference with a perturbed weight scale, and D's comparison
// must see two-part against single-part launches of the wrong A part. Also the refusals (illegal configurations
// say which rule they break and throw), the one-launch-per-class timing is NOT measured here.
//
// Skips (77) when there is no HIP device. Needs no golden data. Device 1 only, through the ctest environment
// (HIP_VISIBLE_DEVICES=1).
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "int8_gemm_proto_ref.h"
#include "linear.h"
#include "r4d.h"
#include "r4dx/core/r4d.hpp"
#include "trellis_ref.hpp"

namespace {

int g_fail = 0;
int g_checks = 0;
#define CHECK(cond, ...)                                                    \
  do {                                                                      \
    ++g_checks;                                                             \
    if (!(cond)) {                                                          \
      std::printf("FAIL %s:%d: ", __FILE__, __LINE__);                      \
      std::printf(__VA_ARGS__);                                             \
      std::printf("\n");                                                    \
      ++g_fail;                                                             \
    }                                                                       \
  } while (0)
#define HIP(x)                                                                            \
  do {                                                                                    \
    const hipError_t e_ = (x);                                                            \
    if (e_ != hipSuccess) {                                                               \
      std::printf("HIP error %s at %d: %s\n", #x, __LINE__, hipGetErrorString(e_));       \
      return 2;                                                                           \
    }                                                                                     \
  } while (0)

template <class T>
struct Dev {
  T* p = nullptr;
  size_t n = 0;
  hipError_t Alloc(size_t count) {
    n = count;
    return hipMalloc(reinterpret_cast<void**>(&p), count * sizeof(T));
  }
  ~Dev() {
    if (p != nullptr) (void)hipFree(p);
  }
};

struct Cls {
  const char* name;
  int N, K, parts, part_n0;
};
// the seven linear classes of the 27B at TP = 1 (src/model's trellis linears)
const Cls kClasses[] = {
    {"mlp.gate_up", 34816, 5120, 2, 17408}, {"mlp.down", 5120, 17408, 1, 0},
    {"gdn.in_proj_qkv", 10240, 5120, 1, 0}, {"gdn.in_proj_z", 6144, 5120, 1, 0},
    {"gdn.out_proj/attn.o", 5120, 6144, 1, 0}, {"attn.qg", 12288, 5120, 1, 0},
    {"attn.k/attn.v", 1024, 5120, 1, 0},
    // the per-rank shapes of TP = 2 (docs/tp.md 2.7): column-parallel linears halve N, row-parallel ones halve K
    {"tp2 mlp.gate_up", 17408, 5120, 2, 8704}, {"tp2 mlp.down", 5120, 8704, 1, 0},
    {"tp2 gdn.in_proj_qkv", 5120, 5120, 1, 0}, {"tp2 gdn.in_proj_z", 3072, 5120, 1, 0},
    {"tp2 gdn.out_proj/attn.o", 5120, 3072, 1, 0}, {"tp2 attn.qg", 6144, 5120, 1, 0},
    {"tp2 attn.k/attn.v", 512, 5120, 1, 0},
};

constexpr float kOutScale = 0.0883883f;  // 1 / sqrt(128), prescale 0
constexpr int kM = 256;

float Bf16ToF(uint16_t b) { return __builtin_bit_cast(float, static_cast<uint32_t>(b) << 16); }
// i8g_bf16_rn (r4d_trellis_i8.h): round to nearest even, NaN quieted
uint16_t Bf16Rn(float f) {
  const uint32_t u = __builtin_bit_cast(uint32_t, f);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return static_cast<uint16_t>((u >> 16) | 0x40u);
  return static_cast<uint16_t>((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

// The 128 columns of group g decoded to f16 on the CPU: q[k][j], j = column - 128 g (only the 8 tile columns it
// needs are read from the pair-grid words).
std::vector<uint16_t> DecodeGroup(const std::vector<uint32_t>& grid, int K, int N, int KB, int g) {
  (void)N;
  std::vector<uint16_t> q(static_cast<size_t>(K) * 128);
  for (int tk = 0; tk < K / 16; ++tk)
    for (int t = 0; t < 8; ++t)
      trellis_ref::DecodeTile(&grid[trellis_ref::PairGridIndex(K, KB, g * 8 + t, tk)], KB, q.data(), 128, 16 * tk, 16 * t);
  return q;
}

// everything the CPU knows about one 128-column group of one linear
struct GroupRef {
  int g = 0;
  std::vector<float> sw;     // [K/128][128]
  std::vector<int8_t> Wp;    // [128][K]
};

// the quantized activations of one part on the CPU
struct ActRef {
  std::vector<float> sa;     // [K/128][256]
  std::vector<int8_t> Ap;    // [256][K] plain
  std::vector<int8_t> A8;    // fragment layout
};
ActRef QuantAct(const std::vector<uint16_t>& xh, int K) {
  std::vector<float> X(xh.size());
  for (size_t i = 0; i < xh.size(); ++i) X[i] = i8p::F16ToF32(xh[i]);
  ActRef a;
  i8p::QuantizeActRef(X.data(), K, a.sa, a.Ap);
  a.A8 = i8p::PackA8(a.Ap, K);
  return a;
}

// raw[row][j] (fp64) = sum_kb sa[kb][row] sw[kb][col] (sum_{k in kb} Ap[row][k] Wp[j][k]) for the group's 128 columns
std::vector<double> RefRaw(const ActRef& a, const GroupRef& gr, int K, double sw_perturb = 1.0) {
  std::vector<double> raw(static_cast<size_t>(kM) * 128, 0.0);
  const int nkb = K / 128;
  const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&, t] {
      for (int r = static_cast<int>(t); r < kM; r += static_cast<int>(nt)) {
        const int8_t* ar = &a.Ap[static_cast<size_t>(r) * K];
        for (int j = 0; j < 128; ++j) {
          const int8_t* wj = &gr.Wp[static_cast<size_t>(j) * K];
          double acc = 0;
          for (int kb = 0; kb < nkb; ++kb) {
            int s = 0;
            for (int k = kb * 128; k < kb * 128 + 128; ++k) s += static_cast<int>(ar[k]) * static_cast<int>(wj[k]);
            acc += static_cast<double>(a.sa[static_cast<size_t>(kb) * kM + r]) *
                   (static_cast<double>(gr.sw[static_cast<size_t>(kb) * 128 + j]) * sw_perturb) * static_cast<double>(s);
          }
          raw[static_cast<size_t>(r) * 128 + j] = acc;
        }
      }
    });
  for (auto& x : th) x.join();
  return raw;
}

// violations of the bench's tolerance for the group's columns of C
long long CountViolations(const std::vector<double>& raw, const std::vector<float>& svh, int g, const std::vector<uint16_t>& C,
                          int N) {
  long long viol = 0;
  for (int r = 0; r < kM; ++r) {
    double v[128], e[128], rms = 0;
    for (int j = 0; j < 128; ++j) v[j] = raw[static_cast<size_t>(r) * 128 + j];
    i8p::Fwht128(v);
    for (int j = 0; j < 128; ++j) {
      e[j] = v[j] * static_cast<double>(svh[g * 128 + j]) * static_cast<double>(kOutScale);
      rms += e[j] * e[j];
    }
    rms = std::sqrt(rms / 128);
    for (int j = 0; j < 128; ++j) {
      const double got = Bf16ToF(C[static_cast<size_t>(r) * N + g * 128 + j]);
      viol += std::fabs(got - e[j]) > 0.0085 * std::fabs(e[j]) + 1e-4 * rms;
    }
  }
  return viol;
}

// the f16 activations of one part: Gaussian, sigma 0.8, a zero 128-block in row 7 (the all-zero scale rule)
std::vector<uint16_t> GaussianAct(int K, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> nd(0.f, 0.8f);
  std::vector<uint16_t> x(static_cast<size_t>(kM) * K);
  for (uint16_t& v : x) v = i8p::F32ToF16(nd(rng));
  for (int k = 0; k < 128; ++k) x[static_cast<size_t>(7) * K + 128 + k] = 0;
  return x;
}

struct Cfg {
  int skw, skg;
};
std::vector<Cfg> LegalConfigs(int K, int N, int n_split, int KB) {
  std::vector<Cfg> v;
  for (int skw : {2, 4, 8})
    for (int skg : {1, 2, 4, 8})
      if (r4d_gemm_trellis_nt_i8_check(kM, K, N, n_split, KB, skw, skg) == nullptr) v.push_back({skw, skg});
  return v;
}

// One (class, KB): the device state and the CPU models.
struct Case {
  Cls c;
  int KB = 4;
  int N = 0, K = 0, parts = 1, n_split = 0;
  std::vector<uint32_t> grid;
  std::vector<float> svh;
  std::vector<std::vector<uint16_t>> xh;   // per part
  std::vector<ActRef> act;                  // per part
  std::vector<GroupRef> groups;
  Dev<uint32_t> dw;
  Dev<float> dsvh, dsw, dsa, dws;
  Dev<float> dswc, dsac;                    // the COARSE tables: SWC [N], SAC [parts][256] (RunCaseCoarse)
  Dev<uint16_t> dx;                         // parts x 256 x K f16
  Dev<int8_t> da8, dw8, dwp;
  Dev<uint16_t> dc, dc2, dc3;
  Dev<unsigned> dtk;
};

int GroupsOf(int N, std::vector<int>* g) {
  std::vector<int> v = {0, N / 128 / 2, N / 128 - 1};
  v.erase(std::unique(v.begin(), v.end()), v.end());
  *g = v;
  return static_cast<int>(v.size());
}

bool SameBytes(const std::vector<uint16_t>& a, const std::vector<uint16_t>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * 2) == 0;
}

int ReadC(const Dev<uint16_t>& d, std::vector<uint16_t>* out, size_t n) {
  out->resize(n);
  HIP(hipMemcpy(out->data(), d.p, n * 2, hipMemcpyDeviceToHost));
  return 0;
}

// the launch under test: both parts quantized once into da8 / dsa by the caller
int Launch(Case& cs, int n_split, int skw, int skg, const int8_t* a0, const float* s0, const int8_t* a1, const float* s1,
           uint16_t* out, const char* what) {
  try {
    r4dx::core::r4d::GemmTrellisNtI8(a0, s0, a1, s1, n_split, cs.dw.p, cs.dsw.p, cs.dsvh.p, out, cs.dws.p, cs.dtk.p, kM, cs.K,
                                     cs.N, cs.KB, skw, skg, kOutScale, nullptr);
  } catch (const std::exception& e) {
    CHECK(false, "%s: launch (skw %d, skg %d) threw: %s", what, skw, skg, e.what());
    return 1;
  }
  HIP(hipDeviceSynchronize());
  return 0;
}

int TicketsZero(Case& cs, const char* what) {
  std::vector<unsigned> t(static_cast<size_t>(cs.N) / 128);
  HIP(hipMemcpy(t.data(), cs.dtk.p, t.size() * 4, hipMemcpyDeviceToHost));
  size_t bad = 0;
  for (unsigned x : t) bad += x != 0;
  CHECK(bad == 0, "%s: %zu tickets not reset", what, bad);
  return 0;
}

int RunCase(const Cls& c, int KB, uint32_t seed) {
  using namespace r4dx;
  Case cs;
  cs.c = c;
  cs.KB = KB;
  cs.N = c.N;
  cs.K = c.K;
  cs.parts = c.parts;
  cs.n_split = c.parts > 1 ? c.part_n0 : c.N;
  const int N = cs.N, K = cs.K, parts = cs.parts, KT = K / 16;
  const std::string tag = std::string(c.name) + " KB" + std::to_string(KB);
  const std::vector<Cfg> cfgs = LegalConfigs(K, N, cs.n_split, KB);
  std::printf("%-26s N %5d K %5d parts %d | %zu legal (skw, skg)", tag.c_str(), N, K, parts, cfgs.size());
  CHECK(!cfgs.empty(), "%s: no legal (skw, skg) at all", tag.c_str());
  if (cfgs.empty()) return 0;
  // the production plan's row is one of them (when the class is a TP = 1 one the model uses)
  const model::TrellisI8Plan plan = model::PlanTrellisI8(N, K, KB, parts, cs.n_split);
  if (plan.ok) {
    bool found = false;
    for (const Cfg& g : cfgs) found = found || (g.skw == plan.skw && g.skg == plan.skg);
    CHECK(found, "%s: the plan's (skw %d, skg %d) is not a legal configuration", tag.c_str(), plan.skw, plan.skg);
  }
  std::printf(" | plan %s", plan.ok ? "ok" : "none");
  const Cfg main_cfg = plan.ok ? Cfg{plan.skw, plan.skg} : cfgs.front();
  std::printf(" (skw %d skg %d)\n", main_cfg.skw, main_cfg.skg);

  // ---- data ----
  std::mt19937 rng(seed);
  cs.grid.resize(static_cast<size_t>(N) * K * KB / 32);
  for (uint32_t& x : cs.grid) x = rng();
  cs.svh.resize(N);
  for (float& x : cs.svh) x = ((rng() & 1u) ? -1.f : 1.f) * (0.5f + static_cast<float>(rng() % 1000u) / 1000.f);
  for (int p = 0; p < parts; ++p) {
    cs.xh.push_back(GaussianAct(K, seed * 31u + static_cast<uint32_t>(p) * 977u + 5u));
    cs.act.push_back(QuantAct(cs.xh.back(), K));
  }
  std::vector<int> gsel;
  GroupsOf(N, &gsel);
  for (int g : gsel) {
    GroupRef gr;
    gr.g = g;
    const std::vector<uint16_t> q = DecodeGroup(cs.grid, K, N, KB, g);
    i8p::QuantizeWeightsRef(q.data(), K, 128, gr.sw, gr.Wp);
    cs.groups.push_back(std::move(gr));
  }
  const size_t a8_part = static_cast<size_t>(kM) * K, sa_part = static_cast<size_t>(K / 128) * kM;
  HIP(cs.dw.Alloc(cs.grid.size()));
  HIP(cs.dsvh.Alloc(N));
  HIP(cs.dsw.Alloc(static_cast<size_t>(K / 128) * N));
  HIP(cs.dx.Alloc(static_cast<size_t>(parts) * kM * K));
  HIP(cs.da8.Alloc(static_cast<size_t>(parts) * a8_part));
  HIP(cs.dsa.Alloc(static_cast<size_t>(parts) * sa_part));
  HIP(cs.dw8.Alloc(static_cast<size_t>(N) * K));
  HIP(cs.dwp.Alloc(static_cast<size_t>(N) * K));
  HIP(cs.dc.Alloc(static_cast<size_t>(kM) * N));
  HIP(cs.dc2.Alloc(static_cast<size_t>(kM) * N));
  HIP(cs.dc3.Alloc(static_cast<size_t>(kM) * N));
  HIP(cs.dws.Alloc(core::r4d::GemmTrellisNtI8WsBytes(kM, N, 8) / sizeof(float)));
  HIP(cs.dtk.Alloc(static_cast<size_t>(N) / 128));
  HIP(hipMemcpy(cs.dw.p, cs.grid.data(), cs.grid.size() * 4, hipMemcpyHostToDevice));
  HIP(hipMemcpy(cs.dsvh.p, cs.svh.data(), cs.svh.size() * 4, hipMemcpyHostToDevice));
  HIP(hipMemset(cs.dtk.p, 0, static_cast<size_t>(N / 128) * 4));
  for (int p = 0; p < parts; ++p)
    HIP(hipMemcpy(cs.dx.p + static_cast<size_t>(p) * kM * K, cs.xh[p].data(), cs.xh[p].size() * 2, hipMemcpyHostToDevice));

  // ---- A. the producers ----
  try {
    core::r4d::TrellisI8WscaleBuild(cs.dw.p, cs.dsw.p, K, N, KB, nullptr);
    core::r4d::TrellisI8DumpW(cs.dw.p, cs.dsw.p, cs.dw8.p, cs.dwp.p, K, N, KB, nullptr);
    core::r4d::TrellisI8QuantAct(cs.dx.p, cs.da8.p, cs.dsa.p, parts, static_cast<int64_t>(kM) * K, K, nullptr);
  } catch (const std::exception& e) {
    CHECK(false, "%s: a producer threw: %s", tag.c_str(), e.what());
    return 0;
  }
  HIP(hipDeviceSynchronize());
  {
    std::vector<float> sw(static_cast<size_t>(K / 128) * N);
    HIP(hipMemcpy(sw.data(), cs.dsw.p, sw.size() * 4, hipMemcpyDeviceToHost));
    size_t bad_sw = 0, bad_wp = 0, bad_w8 = 0;
    const int KTB = KT * 512;   // bytes of one tile pair's blocks in W8
    for (const GroupRef& gr : cs.groups) {
      for (int kb = 0; kb < K / 128; ++kb)
        for (int j = 0; j < 128; ++j)
          bad_sw += std::memcmp(&sw[static_cast<size_t>(kb) * N + gr.g * 128 + j], &gr.sw[static_cast<size_t>(kb) * 128 + j], 4) != 0;
      std::vector<int8_t> wp(static_cast<size_t>(128) * K);
      HIP(hipMemcpy(wp.data(), cs.dwp.p + static_cast<size_t>(gr.g) * 128 * K, wp.size(), hipMemcpyDeviceToHost));
      bad_wp += std::memcmp(wp.data(), gr.Wp.data(), wp.size()) != 0;
      for (int pl = 0; pl < 4; ++pl) {   // the four tile pairs of the group
        const int pair = gr.g * 4 + pl;
        std::vector<int8_t> seg(static_cast<size_t>(KTB));
        HIP(hipMemcpy(seg.data(), cs.dw8.p + static_cast<size_t>(pair) * KTB, seg.size(), hipMemcpyDeviceToHost));
        for (int jn = 0; jn < 32; ++jn)
          for (int k = 0; k < K; ++k) {
            const int n = pair * 32 + jn;
            const size_t off = i8p::W8Offset(n, k, K) - static_cast<size_t>(pair) * KTB;
            bad_w8 += seg[off] != gr.Wp[static_cast<size_t>(n - gr.g * 128) * K + k];
          }
      }
    }
    CHECK(bad_sw == 0, "%s: %zu scale table entries differ from the CPU rule", tag.c_str(), bad_sw);
    CHECK(bad_wp == 0, "%s: the plain int8 weights of %zu group(s) differ from the CPU decode + quantizer", tag.c_str(), bad_wp);
    CHECK(bad_w8 == 0, "%s: %zu int8 weights in the block layout differ from the CPU decode + quantizer", tag.c_str(), bad_w8);
    for (int p = 0; p < parts; ++p) {
      std::vector<int8_t> a8(a8_part);
      std::vector<float> sa(sa_part);
      HIP(hipMemcpy(a8.data(), cs.da8.p + static_cast<size_t>(p) * a8_part, a8.size(), hipMemcpyDeviceToHost));
      HIP(hipMemcpy(sa.data(), cs.dsa.p + static_cast<size_t>(p) * sa_part, sa.size() * 4, hipMemcpyDeviceToHost));
      CHECK(std::memcmp(a8.data(), cs.act[p].A8.data(), a8.size()) == 0, "%s: A8 of part %d differs from the CPU quantizer", tag.c_str(), p);
      CHECK(std::memcmp(sa.data(), cs.act[p].sa.data(), sa.size() * 4) == 0, "%s: SA of part %d differs from the CPU quantizer", tag.c_str(), p);
    }
  }
  const int8_t* a0 = cs.da8.p;
  const float* s0 = cs.dsa.p;
  const int8_t* a1 = parts > 1 ? cs.da8.p + a8_part : nullptr;
  const float* s1 = parts > 1 ? cs.dsa.p + sa_part : nullptr;

  // ---- B. every legal (skw, skg) against the exact integer reference ----
  std::vector<std::vector<double>> raw_for_group;   // by cs.groups
  for (const GroupRef& gr : cs.groups) {
    const int part = (gr.g * 128 >= cs.n_split && parts > 1) ? 1 : 0;
    raw_for_group.push_back(RefRaw(cs.act[part], gr, K));
  }
  std::vector<uint16_t> C;
  size_t viol_total = 0;
  for (const Cfg& g : cfgs) {
    HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
    if (Launch(cs, cs.n_split, g.skw, g.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) != 0) continue;
    if (ReadC(cs.dc, &C, static_cast<size_t>(kM) * N) != 0) return 2;
    long long viol = 0;
    for (size_t i = 0; i < cs.groups.size(); ++i) viol += CountViolations(raw_for_group[i], cs.svh, cs.groups[i].g, C, N);
    CHECK(viol == 0, "%s skw %d skg %d: %lld of %zu outputs outside the exact reference's tolerance", tag.c_str(), g.skw, g.skg, viol,
          cs.groups.size() * 128 * kM);
    viol_total += static_cast<size_t>(viol);
    if (TicketsZero(cs, tag.c_str()) != 0) return 2;
  }
  std::printf("    B  %zu configurations vs the exact reference (%zu groups x 128 cols x 256 rows each): %zu outside tolerance\n",
              cfgs.size(), cs.groups.size(), viol_total);
  // negative control of B: a reference with the weight scales 2% off must be rejected by the same tolerance
  {
    const GroupRef& gr = cs.groups.front();
    const int part = (gr.g * 128 >= cs.n_split && parts > 1) ? 1 : 0;
    HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
    if (Launch(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) == 0) {
      if (ReadC(cs.dc, &C, static_cast<size_t>(kM) * N) != 0) return 2;
      const std::vector<double> bad_ref = RefRaw(cs.act[part], gr, K, 1.02);
      const long long v = CountViolations(bad_ref, cs.svh, gr.g, C, N);
      CHECK(v > 0, "%s: negative control: a 2%% weight-scale error in the reference was not seen by the tolerance", tag.c_str());
    }
  }

  // ---- C. the exact one-hot check ----
  {
    // per part, per row: one nonzero activation at k = k_r (stratified over the whole K so every K slice and step is hit)
    std::vector<std::vector<uint16_t>> xo(parts, std::vector<uint16_t>(static_cast<size_t>(kM) * K, 0));
    std::vector<std::vector<int>> kk(parts, std::vector<int>(kM));
    std::vector<std::vector<float>> vv(parts, std::vector<float>(kM));
    std::mt19937 r2(seed ^ 0xA5A5u);
    for (int p = 0; p < parts; ++p)
      for (int r = 0; r < kM; ++r) {
        const int stride = K / kM;
        kk[p][r] = r * stride + static_cast<int>(r2() % static_cast<uint32_t>(stride));
        const float m = 0.25f + static_cast<float>(r2() % 4000u) / 1000.f;   // 0.25 .. 4.25
        vv[p][r] = (r2() & 1u) ? -m : m;
        xo[p][static_cast<size_t>(r) * K + kk[p][r]] = i8p::F32ToF16(vv[p][r]);
      }
    Dev<uint16_t> dxo;
    HIP(dxo.Alloc(static_cast<size_t>(parts) * kM * K));
    for (int p = 0; p < parts; ++p)
      HIP(hipMemcpy(dxo.p + static_cast<size_t>(p) * kM * K, xo[p].data(), xo[p].size() * 2, hipMemcpyHostToDevice));
    core::r4d::TrellisI8QuantAct(dxo.p, cs.da8.p, cs.dsa.p, parts, static_cast<int64_t>(kM) * K, K, nullptr);
    HIP(hipDeviceSynchronize());
    std::vector<ActRef> ao;
    for (int p = 0; p < parts; ++p) ao.push_back(QuantAct(xo[p], K));
    size_t nbad = 0, ntot = 0;
    for (const Cfg& g : {main_cfg, cfgs.back()}) {
      HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
      if (Launch(cs, cs.n_split, g.skw, g.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) != 0) continue;
      if (ReadC(cs.dc, &C, static_cast<size_t>(kM) * N) != 0) return 2;
      for (const GroupRef& gr : cs.groups) {
        const int p = (gr.g * 128 >= cs.n_split && parts > 1) ? 1 : 0;
        for (int r = 0; r < kM; ++r) {
          // the kernel's chain for a single nonzero product: v_n = f32(t * f32(sa * sw)), t = q_a * q_w
          float v[128];
          const int k = kk[p][r], kb = k / 128;
          const float sa = ao[p].sa[static_cast<size_t>(kb) * kM + r];
          for (int j = 0; j < 128; ++j) {
            const int t = static_cast<int>(ao[p].Ap[static_cast<size_t>(r) * K + k]) * static_cast<int>(gr.Wp[static_cast<size_t>(j) * K + k]);
            const float prod = sa * gr.sw[static_cast<size_t>(kb) * 128 + j];
            v[j] = static_cast<float>(t) * prod;
          }
          trellis_ref::FwhtLdsF32(v);
          for (int j = 0; j < 128; ++j) {
            const float o = (v[j] * cs.svh[gr.g * 128 + j]) * kOutScale;
            ++ntot;
            nbad += Bf16Rn(o) != C[static_cast<size_t>(r) * N + gr.g * 128 + j];
          }
        }
      }
    }
    CHECK(nbad == 0, "%s: %zu of %zu one-hot outputs differ from the CPU's fp32 emulation (must be byte-exact)", tag.c_str(), nbad, ntot);
    std::printf("    C  one-hot rows, byte-exact vs the fp32 emulation of the epilogue: %zu compared, %zu differ\n", ntot, nbad);
    // restore the dense activations for D..G
    core::r4d::TrellisI8QuantAct(cs.dx.p, cs.da8.p, cs.dsa.p, parts, static_cast<int64_t>(kM) * K, K, nullptr);
    HIP(hipDeviceSynchronize());
  }

  // ---- D. two parts == two single-part launches (classes with two A parts) ----
  if (parts > 1) {
    std::vector<uint16_t> ca, cb, cp;
    size_t nd_total = 0;
    std::vector<int> splits = {128, N - 128, c.part_n0, 8704};
    std::sort(splits.begin(), splits.end());
    splits.erase(std::unique(splits.begin(), splits.end()), splits.end());
    for (int ns : splits) {
      if (ns <= 0 || ns >= N || ns % 128 != 0) continue;
      if (r4d_gemm_trellis_nt_i8_check(kM, K, N, ns, KB, main_cfg.skw, main_cfg.skg) != nullptr) continue;
      HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
      HIP(hipMemset(cs.dc2.p, 0xFF, static_cast<size_t>(kM) * N * 2));
      HIP(hipMemset(cs.dc3.p, 0xFF, static_cast<size_t>(kM) * N * 2));
      if (Launch(cs, ns, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) != 0) continue;
      if (Launch(cs, N, main_cfg.skw, main_cfg.skg, a0, s0, nullptr, nullptr, cs.dc2.p, tag.c_str()) != 0) continue;
      // the second part alone: its pair passed as the first, n_split = N (a single-part launch)
      if (Launch(cs, N, main_cfg.skw, main_cfg.skg, a1, s1, nullptr, nullptr, cs.dc3.p, tag.c_str()) != 0) continue;
      if (ReadC(cs.dc, &cp, static_cast<size_t>(kM) * N) != 0) return 2;
      if (ReadC(cs.dc2, &ca, static_cast<size_t>(kM) * N) != 0) return 2;
      if (ReadC(cs.dc3, &cb, static_cast<size_t>(kM) * N) != 0) return 2;
      size_t nd = 0, nd_wrong = 0;
      for (int r = 0; r < kM; ++r)
        for (int n = 0; n < N; ++n) {
          const size_t i = static_cast<size_t>(r) * N + n;
          nd += cp[i] != (n < ns ? ca[i] : cb[i]);
          nd_wrong += cp[i] != ca[i];   // the wrong composition: every column from the first part
        }
      nd_total += nd;
      CHECK(nd == 0, "%s: two-part launch (n_split %d) differs from the two single-part launches in %zu outputs", tag.c_str(), ns, nd);
      CHECK(nd_wrong > 0, "%s: negative control: n_split %d two-part output equals a single-part launch of the first part everywhere", tag.c_str(), ns);
    }
    std::printf("    D  two-part launches vs two single-part launches at %zu boundaries: %zu bytes differ\n", splits.size(), nd_total);
  }

  // ---- E. repeats, tickets ----
  {
    std::vector<uint16_t> first, again;
    HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
    if (Launch(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) == 0) {
      if (ReadC(cs.dc, &first, static_cast<size_t>(kM) * N) != 0) return 2;
      size_t diff = 0;
      for (int rep = 0; rep < 3; ++rep) {
        HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
        if (Launch(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) != 0) continue;
        if (ReadC(cs.dc, &again, static_cast<size_t>(kM) * N) != 0) return 2;
        diff += !SameBytes(first, again);
        if (TicketsZero(cs, tag.c_str()) != 0) return 2;
      }
      CHECK(diff == 0, "%s: %zu of 3 repeats gave other bytes", tag.c_str(), diff);
      std::printf("    E  3 repeats byte-identical: %s, tickets back at zero\n", diff == 0 ? "yes" : "NO");
    }
  }

  // ---- F. f16, int8, f16 on one linear ----
  {
    const model::TrellisM256Plan p16 = model::PlanTrellisM256(N, K, KB, parts, cs.n_split);
    if (!p16.ok) {
      std::printf("    F  no f16 M = 256 plan for this shape (%s): skipped\n", p16.why.c_str());
    } else {
      std::vector<uint16_t> f_before, f_after, i_alone, i_between;
      const uint16_t* x0 = cs.dx.p;
      const uint16_t* x1 = parts > 1 ? cs.dx.p + static_cast<size_t>(kM) * K : nullptr;
      const auto f16 = [&](uint16_t* out) {
        core::r4d::GemmTrellisNtM256(x0, x1, cs.n_split, cs.dw.p, cs.dsvh.p, out, cs.dws.p, cs.dtk.p, kM, K, N, KB, p16.SK, 1,
                                     p16.SKG, 4, kOutScale, p16.SKW, nullptr);
      };
      const size_t cn = static_cast<size_t>(kM) * N;
      try {
        // the int8 launch alone (its own bytes), then f16, int8, f16 on the same tickets and ws
        HIP(hipMemset(cs.dc.p, 0xFF, cn * 2));
        if (Launch(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) != 0) return 0;
        if (ReadC(cs.dc, &i_alone, cn) != 0) return 2;
        HIP(hipMemset(cs.dc2.p, 0xFF, cn * 2));
        f16(cs.dc2.p);
        HIP(hipDeviceSynchronize());
        if (ReadC(cs.dc2, &f_before, cn) != 0) return 2;
        HIP(hipMemset(cs.dc3.p, 0xFF, cn * 2));
        if (Launch(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc3.p, tag.c_str()) != 0) return 0;
        if (ReadC(cs.dc3, &i_between, cn) != 0) return 2;
        HIP(hipMemset(cs.dc2.p, 0xFF, cn * 2));
        f16(cs.dc2.p);
        HIP(hipDeviceSynchronize());
        if (ReadC(cs.dc2, &f_after, cn) != 0) return 2;
      } catch (const std::exception& e) {
        CHECK(false, "%s: the interleave threw: %s", tag.c_str(), e.what());
        return 0;
      }
      CHECK(SameBytes(f_before, f_after), "%s: the f16 bytes after an int8 launch on the same tickets / ws differ", tag.c_str());
      CHECK(SameBytes(i_alone, i_between), "%s: the int8 bytes next to f16 launches differ from the int8 launch alone", tag.c_str());
      if (TicketsZero(cs, tag.c_str()) != 0) return 2;
      std::printf("    F  f16 / int8 / f16 on one linear: f16 bytes %s, int8 bytes %s\n", SameBytes(f_before, f_after) ? "unchanged" : "CHANGED",
                  SameBytes(i_alone, i_between) ? "unchanged" : "CHANGED");
    }
  }

  // ---- G. row independence: the same rows in another order ----
  {
    std::vector<int> perm(kM);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 r3(seed + 17u);
    std::shuffle(perm.begin(), perm.end(), r3);
    // part p row i of the permuted input = row perm[i] of the original
    std::vector<uint16_t> xp(static_cast<size_t>(parts) * kM * K);
    for (int p = 0; p < parts; ++p)
      for (int i = 0; i < kM; ++i)
        std::memcpy(&xp[(static_cast<size_t>(p) * kM + i) * K], &cs.xh[p][static_cast<size_t>(perm[i]) * K], static_cast<size_t>(K) * 2);
    Dev<uint16_t> dxp;
    HIP(dxp.Alloc(xp.size()));
    HIP(hipMemcpy(dxp.p, xp.data(), xp.size() * 2, hipMemcpyHostToDevice));
    std::vector<uint16_t> c_orig, c_perm;
    HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
    if (Launch(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) != 0) return 0;
    core::r4d::TrellisI8QuantAct(dxp.p, cs.da8.p, cs.dsa.p, parts, static_cast<int64_t>(kM) * K, K, nullptr);
    HIP(hipDeviceSynchronize());
    HIP(hipMemset(cs.dc2.p, 0xFF, static_cast<size_t>(kM) * N * 2));
    if (Launch(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc2.p, tag.c_str()) != 0) return 0;
    if (ReadC(cs.dc, &c_orig, static_cast<size_t>(kM) * N) != 0) return 2;
    if (ReadC(cs.dc2, &c_perm, static_cast<size_t>(kM) * N) != 0) return 2;
    size_t nd = 0;
    for (int i = 0; i < kM; ++i)
      nd += std::memcmp(&c_perm[static_cast<size_t>(i) * N], &c_orig[static_cast<size_t>(perm[i]) * N], static_cast<size_t>(N) * 2) != 0;
    CHECK(nd == 0, "%s: %zu of 256 permuted rows differ from the same row in the original order", tag.c_str(), nd);
    std::printf("    G  row permutation: %zu of 256 rows differ\n", nd);
  }
  return 0;
}

// ===== the COARSE scales (R4DX_PREFILL_INT8_SCALES=coarse, docs/int8-prefill.md "Coarse scales") ============================
// The same classes at both rates through r4d_gemm_trellis_nt_i8c with ONE activation scale per row (r4d_trellis_i8_quant_act_row)
// and ONE weight scale per column (r4d_trellis_i8_wscale_col), both over the whole K:
//   A. the producers against the CPU (QuantizeWeightsColRef / QuantizeActRowRef): the column table and the int8 weights
//      (r4d_trellis_i8_dump_w_col, block layout and plain) on three whole groups, the row quantizer (one and two parts);
//   B. every legal (skw, skg) against the exact integer reference of the coarse math (int32 sum over the whole K, one
//      multiply by sa[row] * sw[col], fp64, then the FWHT, svh and out_scale), within the same tolerance as the per-128
//      kernel's, with a negative control (a 2% weight-scale error is seen);
//   C. the one-hot byte-exact check against the CPU's fp32 emulation of the epilogue (v = f32(t * f32(sa * sw)));
//   D. two A parts == two single-part launches; E. repeats and tickets; F. an f16, coarse, f16 interleave on one linear;
//   G. row independence (the same rows in another order).
struct GroupRefC {
  int g = 0;
  std::vector<float> sw;     // [128]
  std::vector<int8_t> Wp;    // [128][K]
};
struct ActRefC {
  std::vector<float> sa;     // [256]
  std::vector<int8_t> Ap;    // [256][K] plain
  std::vector<int8_t> A8;    // fragment layout
};
ActRefC QuantActRow(const std::vector<uint16_t>& xh, int K) {
  std::vector<float> X(xh.size());
  for (size_t i = 0; i < xh.size(); ++i) X[i] = i8p::F16ToF32(xh[i]);
  ActRefC a;
  i8p::QuantizeActRowRef(X.data(), K, a.sa, a.Ap);
  a.A8 = i8p::PackA8(a.Ap, K);
  return a;
}
// raw[row][j] (fp64) = sa[row] sw[j] (sum_{k < K} Ap[row][k] Wp[j][k]), the integer sum exact
std::vector<double> RefRawC(const ActRefC& a, const GroupRefC& gr, int K, double sw_perturb = 1.0) {
  std::vector<double> raw(static_cast<size_t>(kM) * 128, 0.0);
  const unsigned nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::vector<std::thread> th;
  for (unsigned t = 0; t < nt; ++t)
    th.emplace_back([&, t] {
      for (int r = static_cast<int>(t); r < kM; r += static_cast<int>(nt)) {
        const int8_t* ar = &a.Ap[static_cast<size_t>(r) * K];
        for (int j = 0; j < 128; ++j) {
          const int8_t* wj = &gr.Wp[static_cast<size_t>(j) * K];
          long long s = 0;
          for (int k = 0; k < K; ++k) s += static_cast<int>(ar[k]) * static_cast<int>(wj[k]);
          raw[static_cast<size_t>(r) * 128 + j] = static_cast<double>(a.sa[static_cast<size_t>(r)]) *
                                                  (static_cast<double>(gr.sw[static_cast<size_t>(j)]) * sw_perturb) * static_cast<double>(s);
        }
      }
    });
  for (auto& x : th) x.join();
  return raw;
}

int LaunchC(Case& cs, int n_split, int skw, int skg, const int8_t* a0, const float* s0, const int8_t* a1, const float* s1,
            uint16_t* out, const char* what) {
  try {
    r4dx::core::r4d::GemmTrellisNtI8c(a0, s0, a1, s1, n_split, cs.dw.p, cs.dswc.p, cs.dsvh.p, out, cs.dws.p, cs.dtk.p, kM, cs.K,
                                      cs.N, cs.KB, skw, skg, kOutScale, nullptr);
  } catch (const std::exception& e) {
    CHECK(false, "%s [coarse]: launch (skw %d, skg %d) threw: %s", what, skw, skg, e.what());
    return 1;
  }
  HIP(hipDeviceSynchronize());
  return 0;
}

int RunCaseCoarse(const Cls& c, int KB, uint32_t seed) {
  using namespace r4dx;
  Case cs;
  cs.c = c;
  cs.KB = KB;
  cs.N = c.N;
  cs.K = c.K;
  cs.parts = c.parts;
  cs.n_split = c.parts > 1 ? c.part_n0 : c.N;
  const int N = cs.N, K = cs.K, parts = cs.parts;
  const std::string tag = std::string(c.name) + " KB" + std::to_string(KB) + " [coarse]";
  const std::vector<Cfg> cfgs = LegalConfigs(K, N, cs.n_split, KB);
  CHECK(!cfgs.empty(), "%s: no legal (skw, skg) at all", tag.c_str());
  if (cfgs.empty()) return 0;
  const model::TrellisI8Plan plan = model::PlanTrellisI8(N, K, KB, parts, cs.n_split, /*coarse=*/true);
  if (plan.ok) {
    bool found = false;
    for (const Cfg& g : cfgs) found = found || (g.skw == plan.skw && g.skg == plan.skg);
    CHECK(found, "%s: the plan's (skw %d, skg %d) is not a legal configuration", tag.c_str(), plan.skw, plan.skg);
  }
  const Cfg main_cfg = plan.ok ? Cfg{plan.skw, plan.skg} : cfgs.front();
  std::printf("%-34s N %5d K %5d parts %d | %zu legal (skw, skg) | coarse plan %s (skw %d skg %d)\n", tag.c_str(), N, K, parts, cfgs.size(),
              plan.ok ? "ok" : "none", main_cfg.skw, main_cfg.skg);

  // ---- data ----
  std::mt19937 rng(seed);
  cs.grid.resize(static_cast<size_t>(N) * K * KB / 32);
  for (uint32_t& x : cs.grid) x = rng();
  cs.svh.resize(N);
  for (float& x : cs.svh) x = ((rng() & 1u) ? -1.f : 1.f) * (0.5f + static_cast<float>(rng() % 1000u) / 1000.f);
  std::vector<ActRefC> act;
  for (int p = 0; p < parts; ++p) {
    cs.xh.push_back(GaussianAct(K, seed * 31u + static_cast<uint32_t>(p) * 977u + 5u));
    act.push_back(QuantActRow(cs.xh.back(), K));
  }
  // one row of exact zeros (the scale-1 rule) and one with a single huge element (a row scale far above the rest)
  for (int p = 0; p < parts; ++p) {
    std::fill(cs.xh[p].begin() + static_cast<size_t>(11) * K, cs.xh[p].begin() + static_cast<size_t>(12) * K, static_cast<uint16_t>(0));
    cs.xh[p][static_cast<size_t>(21) * K + K / 3] = i8p::F32ToF16(900.f);
    act[p] = QuantActRow(cs.xh[p], K);
  }
  std::vector<int> gsel;
  GroupsOf(N, &gsel);
  std::vector<GroupRefC> groups;
  for (int g : gsel) {
    GroupRefC gr;
    gr.g = g;
    const std::vector<uint16_t> q = DecodeGroup(cs.grid, K, N, KB, g);
    i8p::QuantizeWeightsColRef(q.data(), K, 128, gr.sw, gr.Wp);
    groups.push_back(std::move(gr));
  }
  const size_t a8_part = static_cast<size_t>(kM) * K, sa_part = static_cast<size_t>(kM);
  HIP(cs.dw.Alloc(cs.grid.size()));
  HIP(cs.dsvh.Alloc(N));
  HIP(cs.dswc.Alloc(static_cast<size_t>(N)));
  HIP(cs.dx.Alloc(static_cast<size_t>(parts) * kM * K));
  HIP(cs.da8.Alloc(static_cast<size_t>(parts) * a8_part));
  HIP(cs.dsac.Alloc(static_cast<size_t>(parts) * sa_part));
  HIP(cs.dw8.Alloc(static_cast<size_t>(N) * K));
  HIP(cs.dwp.Alloc(static_cast<size_t>(N) * K));
  HIP(cs.dc.Alloc(static_cast<size_t>(kM) * N));
  HIP(cs.dc2.Alloc(static_cast<size_t>(kM) * N));
  HIP(cs.dc3.Alloc(static_cast<size_t>(kM) * N));
  HIP(cs.dws.Alloc(core::r4d::GemmTrellisNtI8WsBytes(kM, N, 8) / sizeof(float)));
  HIP(cs.dtk.Alloc(static_cast<size_t>(N) / 128));
  HIP(hipMemcpy(cs.dw.p, cs.grid.data(), cs.grid.size() * 4, hipMemcpyHostToDevice));
  HIP(hipMemcpy(cs.dsvh.p, cs.svh.data(), cs.svh.size() * 4, hipMemcpyHostToDevice));
  HIP(hipMemset(cs.dtk.p, 0, static_cast<size_t>(N / 128) * 4));
  for (int p = 0; p < parts; ++p)
    HIP(hipMemcpy(cs.dx.p + static_cast<size_t>(p) * kM * K, cs.xh[p].data(), cs.xh[p].size() * 2, hipMemcpyHostToDevice));

  // ---- A. the producers ----
  try {
    core::r4d::TrellisI8WscaleColBuild(cs.dw.p, cs.dswc.p, K, N, KB, nullptr);
    core::r4d::TrellisI8DumpWCol(cs.dw.p, cs.dswc.p, cs.dw8.p, cs.dwp.p, K, N, KB, nullptr);
    core::r4d::TrellisI8QuantActRow(cs.dx.p, cs.da8.p, cs.dsac.p, parts, static_cast<int64_t>(kM) * K, K, nullptr);
  } catch (const std::exception& e) {
    CHECK(false, "%s: a producer threw: %s", tag.c_str(), e.what());
    return 0;
  }
  HIP(hipDeviceSynchronize());
  {
    std::vector<float> swc(static_cast<size_t>(N));
    HIP(hipMemcpy(swc.data(), cs.dswc.p, swc.size() * 4, hipMemcpyDeviceToHost));
    size_t bad_sw = 0, bad_wp = 0, bad_w8 = 0;
    const int KT = K / 16, KTB = KT * 512;
    for (const GroupRefC& gr : groups) {
      for (int j = 0; j < 128; ++j) bad_sw += std::memcmp(&swc[static_cast<size_t>(gr.g) * 128 + j], &gr.sw[static_cast<size_t>(j)], 4) != 0;
      std::vector<int8_t> wp(static_cast<size_t>(128) * K);
      HIP(hipMemcpy(wp.data(), cs.dwp.p + static_cast<size_t>(gr.g) * 128 * K, wp.size(), hipMemcpyDeviceToHost));
      bad_wp += std::memcmp(wp.data(), gr.Wp.data(), wp.size()) != 0;
      for (int pl = 0; pl < 4; ++pl) {
        const int pair = gr.g * 4 + pl;
        std::vector<int8_t> seg(static_cast<size_t>(KTB));
        HIP(hipMemcpy(seg.data(), cs.dw8.p + static_cast<size_t>(pair) * KTB, seg.size(), hipMemcpyDeviceToHost));
        for (int jn = 0; jn < 32; ++jn)
          for (int k = 0; k < K; ++k) {
            const int n = pair * 32 + jn;
            const size_t off = i8p::W8Offset(n, k, K) - static_cast<size_t>(pair) * KTB;
            bad_w8 += seg[off] != gr.Wp[static_cast<size_t>(n - gr.g * 128) * K + k];
          }
      }
    }
    CHECK(bad_sw == 0, "%s: %zu column-table entries differ from the CPU rule", tag.c_str(), bad_sw);
    CHECK(bad_wp == 0, "%s: the plain int8 weights of %zu group(s) differ from the CPU decode + column quantizer", tag.c_str(), bad_wp);
    CHECK(bad_w8 == 0, "%s: %zu int8 weights in the block layout differ from the CPU decode + column quantizer", tag.c_str(), bad_w8);
    for (int p = 0; p < parts; ++p) {
      std::vector<int8_t> a8(a8_part);
      std::vector<float> sa(sa_part);
      HIP(hipMemcpy(a8.data(), cs.da8.p + static_cast<size_t>(p) * a8_part, a8.size(), hipMemcpyDeviceToHost));
      HIP(hipMemcpy(sa.data(), cs.dsac.p + static_cast<size_t>(p) * sa_part, sa.size() * 4, hipMemcpyDeviceToHost));
      CHECK(std::memcmp(a8.data(), act[p].A8.data(), a8.size()) == 0, "%s: A8 of part %d differs from the CPU row quantizer", tag.c_str(), p);
      CHECK(std::memcmp(sa.data(), act[p].sa.data(), sa.size() * 4) == 0, "%s: SA of part %d differs from the CPU row quantizer", tag.c_str(), p);
    }
    std::printf("    A  column table + int8 weights on %zu groups and the row quantizer vs the CPU: %zu / %zu / %zu bad\n", groups.size(), bad_sw,
                bad_wp, bad_w8);
  }
  const int8_t* a0 = cs.da8.p;
  const float* s0 = cs.dsac.p;
  const int8_t* a1 = parts > 1 ? cs.da8.p + a8_part : nullptr;
  const float* s1 = parts > 1 ? cs.dsac.p + sa_part : nullptr;

  // ---- B. every legal (skw, skg) against the exact integer reference ----
  std::vector<std::vector<double>> raw_for_group;
  for (const GroupRefC& gr : groups) {
    const int part = (gr.g * 128 >= cs.n_split && parts > 1) ? 1 : 0;
    raw_for_group.push_back(RefRawC(act[part], gr, K));
  }
  std::vector<uint16_t> C;
  size_t viol_total = 0;
  for (const Cfg& g : cfgs) {
    HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
    if (LaunchC(cs, cs.n_split, g.skw, g.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) != 0) continue;
    if (ReadC(cs.dc, &C, static_cast<size_t>(kM) * N) != 0) return 2;
    long long viol = 0;
    for (size_t i = 0; i < groups.size(); ++i) viol += CountViolations(raw_for_group[i], cs.svh, groups[i].g, C, N);
    CHECK(viol == 0, "%s skw %d skg %d: %lld of %zu outputs outside the exact reference's tolerance", tag.c_str(), g.skw, g.skg, viol,
          groups.size() * 128 * kM);
    viol_total += static_cast<size_t>(viol);
    if (TicketsZero(cs, tag.c_str()) != 0) return 2;
  }
  std::printf("    B  %zu configurations vs the exact coarse reference (%zu groups x 128 cols x 256 rows each): %zu outside tolerance\n", cfgs.size(),
              groups.size(), viol_total);
  {
    const GroupRefC& gr = groups.front();
    const int part = (gr.g * 128 >= cs.n_split && parts > 1) ? 1 : 0;
    HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
    if (LaunchC(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) == 0) {
      if (ReadC(cs.dc, &C, static_cast<size_t>(kM) * N) != 0) return 2;
      const std::vector<double> bad_ref = RefRawC(act[part], gr, K, 1.02);
      CHECK(CountViolations(bad_ref, cs.svh, gr.g, C, N) > 0, "%s: negative control: a 2%% weight-scale error in the reference was not seen", tag.c_str());
    }
  }

  // ---- C. the exact one-hot check ----
  {
    std::vector<std::vector<uint16_t>> xo(parts, std::vector<uint16_t>(static_cast<size_t>(kM) * K, 0));
    std::vector<std::vector<int>> kk(parts, std::vector<int>(kM));
    std::mt19937 r2(seed ^ 0x5A5Au);
    for (int p = 0; p < parts; ++p)
      for (int r = 0; r < kM; ++r) {
        const int stride = K / kM;
        kk[p][r] = r * stride + static_cast<int>(r2() % static_cast<uint32_t>(stride));
        const float m = 0.25f + static_cast<float>(r2() % 4000u) / 1000.f;
        xo[p][static_cast<size_t>(r) * K + kk[p][r]] = i8p::F32ToF16((r2() & 1u) ? -m : m);
      }
    Dev<uint16_t> dxo;
    HIP(dxo.Alloc(static_cast<size_t>(parts) * kM * K));
    for (int p = 0; p < parts; ++p)
      HIP(hipMemcpy(dxo.p + static_cast<size_t>(p) * kM * K, xo[p].data(), xo[p].size() * 2, hipMemcpyHostToDevice));
    core::r4d::TrellisI8QuantActRow(dxo.p, cs.da8.p, cs.dsac.p, parts, static_cast<int64_t>(kM) * K, K, nullptr);
    HIP(hipDeviceSynchronize());
    std::vector<ActRefC> ao;
    for (int p = 0; p < parts; ++p) ao.push_back(QuantActRow(xo[p], K));
    size_t nbad = 0, ntot = 0;
    for (const Cfg& g : {main_cfg, cfgs.back()}) {
      HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
      if (LaunchC(cs, cs.n_split, g.skw, g.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) != 0) continue;
      if (ReadC(cs.dc, &C, static_cast<size_t>(kM) * N) != 0) return 2;
      for (const GroupRefC& gr : groups) {
        const int p = (gr.g * 128 >= cs.n_split && parts > 1) ? 1 : 0;
        for (int r = 0; r < kM; ++r) {
          float v[128];
          const int k = kk[p][r];
          const float sa = ao[p].sa[static_cast<size_t>(r)];
          for (int j = 0; j < 128; ++j) {
            const int t = static_cast<int>(ao[p].Ap[static_cast<size_t>(r) * K + k]) * static_cast<int>(gr.Wp[static_cast<size_t>(j) * K + k]);
            v[j] = static_cast<float>(t) * (sa * gr.sw[static_cast<size_t>(j)]);
          }
          trellis_ref::FwhtLdsF32(v);
          for (int j = 0; j < 128; ++j) {
            const float o = (v[j] * cs.svh[gr.g * 128 + j]) * kOutScale;
            ++ntot;
            nbad += Bf16Rn(o) != C[static_cast<size_t>(r) * N + gr.g * 128 + j];
          }
        }
      }
    }
    CHECK(nbad == 0, "%s: %zu of %zu one-hot outputs differ from the CPU's fp32 emulation (must be byte-exact)", tag.c_str(), nbad, ntot);
    std::printf("    C  one-hot rows, byte-exact vs the fp32 emulation of the epilogue: %zu compared, %zu differ\n", ntot, nbad);
    core::r4d::TrellisI8QuantActRow(cs.dx.p, cs.da8.p, cs.dsac.p, parts, static_cast<int64_t>(kM) * K, K, nullptr);
    HIP(hipDeviceSynchronize());
  }

  // ---- D. two parts == two single-part launches ----
  if (parts > 1) {
    std::vector<uint16_t> ca, cb, cp;
    size_t nd_total = 0;
    std::vector<int> splits = {128, N - 128, c.part_n0, 8704};
    std::sort(splits.begin(), splits.end());
    splits.erase(std::unique(splits.begin(), splits.end()), splits.end());
    for (int ns : splits) {
      if (ns <= 0 || ns >= N || ns % 128 != 0) continue;
      if (r4d_gemm_trellis_nt_i8_check(kM, K, N, ns, KB, main_cfg.skw, main_cfg.skg) != nullptr) continue;
      HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
      HIP(hipMemset(cs.dc2.p, 0xFF, static_cast<size_t>(kM) * N * 2));
      HIP(hipMemset(cs.dc3.p, 0xFF, static_cast<size_t>(kM) * N * 2));
      if (LaunchC(cs, ns, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) != 0) continue;
      if (LaunchC(cs, N, main_cfg.skw, main_cfg.skg, a0, s0, nullptr, nullptr, cs.dc2.p, tag.c_str()) != 0) continue;
      if (LaunchC(cs, N, main_cfg.skw, main_cfg.skg, a1, s1, nullptr, nullptr, cs.dc3.p, tag.c_str()) != 0) continue;
      if (ReadC(cs.dc, &cp, static_cast<size_t>(kM) * N) != 0) return 2;
      if (ReadC(cs.dc2, &ca, static_cast<size_t>(kM) * N) != 0) return 2;
      if (ReadC(cs.dc3, &cb, static_cast<size_t>(kM) * N) != 0) return 2;
      size_t nd = 0, nd_wrong = 0;
      for (int r = 0; r < kM; ++r)
        for (int n = 0; n < N; ++n) {
          const size_t i = static_cast<size_t>(r) * N + n;
          nd += cp[i] != (n < ns ? ca[i] : cb[i]);
          nd_wrong += cp[i] != ca[i];
        }
      nd_total += nd;
      CHECK(nd == 0, "%s: two-part launch (n_split %d) differs from the two single-part launches in %zu outputs", tag.c_str(), ns, nd);
      CHECK(nd_wrong > 0, "%s: negative control: n_split %d two-part output equals a single-part launch of the first part everywhere", tag.c_str(), ns);
    }
    std::printf("    D  two-part launches vs two single-part launches at %zu boundaries: %zu bytes differ\n", splits.size(), nd_total);
  }

  // ---- E. repeats, tickets ----
  {
    std::vector<uint16_t> first, again;
    HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
    if (LaunchC(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) == 0) {
      if (ReadC(cs.dc, &first, static_cast<size_t>(kM) * N) != 0) return 2;
      size_t diff = 0;
      for (int rep = 0; rep < 3; ++rep) {
        HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
        if (LaunchC(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) != 0) continue;
        if (ReadC(cs.dc, &again, static_cast<size_t>(kM) * N) != 0) return 2;
        diff += !SameBytes(first, again);
        if (TicketsZero(cs, tag.c_str()) != 0) return 2;
      }
      CHECK(diff == 0, "%s: %zu of 3 repeats gave other bytes", tag.c_str(), diff);
      std::printf("    E  3 repeats byte-identical: %s, tickets back at zero\n", diff == 0 ? "yes" : "NO");
    }
  }

  // ---- F. f16, coarse, f16 on one linear (one tickets buffer, one ws) ----
  {
    const model::TrellisM256Plan p16 = model::PlanTrellisM256(N, K, KB, parts, cs.n_split);
    if (!p16.ok) {
      std::printf("    F  no f16 M = 256 plan for this shape (%s): skipped\n", p16.why.c_str());
    } else {
      std::vector<uint16_t> f_before, f_after, i_alone, i_between;
      const uint16_t* x0 = cs.dx.p;
      const uint16_t* x1 = parts > 1 ? cs.dx.p + static_cast<size_t>(kM) * K : nullptr;
      const auto f16 = [&](uint16_t* out) {
        core::r4d::GemmTrellisNtM256(x0, x1, cs.n_split, cs.dw.p, cs.dsvh.p, out, cs.dws.p, cs.dtk.p, kM, K, N, KB, p16.SK, 1, p16.SKG, 4,
                                     kOutScale, p16.SKW, nullptr);
      };
      const size_t cn = static_cast<size_t>(kM) * N;
      try {
        HIP(hipMemset(cs.dc.p, 0xFF, cn * 2));
        if (LaunchC(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) != 0) return 0;
        if (ReadC(cs.dc, &i_alone, cn) != 0) return 2;
        HIP(hipMemset(cs.dc2.p, 0xFF, cn * 2));
        f16(cs.dc2.p);
        HIP(hipDeviceSynchronize());
        if (ReadC(cs.dc2, &f_before, cn) != 0) return 2;
        HIP(hipMemset(cs.dc3.p, 0xFF, cn * 2));
        if (LaunchC(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc3.p, tag.c_str()) != 0) return 0;
        if (ReadC(cs.dc3, &i_between, cn) != 0) return 2;
        HIP(hipMemset(cs.dc2.p, 0xFF, cn * 2));
        f16(cs.dc2.p);
        HIP(hipDeviceSynchronize());
        if (ReadC(cs.dc2, &f_after, cn) != 0) return 2;
      } catch (const std::exception& e) {
        CHECK(false, "%s: the interleave threw: %s", tag.c_str(), e.what());
        return 0;
      }
      CHECK(SameBytes(f_before, f_after), "%s: the f16 bytes after a coarse launch on the same tickets / ws differ", tag.c_str());
      CHECK(SameBytes(i_alone, i_between), "%s: the coarse bytes next to f16 launches differ from the coarse launch alone", tag.c_str());
      if (TicketsZero(cs, tag.c_str()) != 0) return 2;
      std::printf("    F  f16 / coarse / f16 on one linear: f16 bytes %s, coarse bytes %s\n", SameBytes(f_before, f_after) ? "unchanged" : "CHANGED",
                  SameBytes(i_alone, i_between) ? "unchanged" : "CHANGED");
    }
  }

  // ---- G. row independence ----
  {
    std::vector<int> perm(kM);
    std::iota(perm.begin(), perm.end(), 0);
    std::mt19937 r3(seed + 19u);
    std::shuffle(perm.begin(), perm.end(), r3);
    std::vector<uint16_t> xp(static_cast<size_t>(parts) * kM * K);
    for (int p = 0; p < parts; ++p)
      for (int i = 0; i < kM; ++i)
        std::memcpy(&xp[(static_cast<size_t>(p) * kM + i) * K], &cs.xh[p][static_cast<size_t>(perm[i]) * K], static_cast<size_t>(K) * 2);
    Dev<uint16_t> dxp;
    HIP(dxp.Alloc(xp.size()));
    HIP(hipMemcpy(dxp.p, xp.data(), xp.size() * 2, hipMemcpyHostToDevice));
    std::vector<uint16_t> c_orig, c_perm;
    HIP(hipMemset(cs.dc.p, 0xFF, static_cast<size_t>(kM) * N * 2));
    if (LaunchC(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc.p, tag.c_str()) != 0) return 0;
    core::r4d::TrellisI8QuantActRow(dxp.p, cs.da8.p, cs.dsac.p, parts, static_cast<int64_t>(kM) * K, K, nullptr);
    HIP(hipDeviceSynchronize());
    HIP(hipMemset(cs.dc2.p, 0xFF, static_cast<size_t>(kM) * N * 2));
    if (LaunchC(cs, cs.n_split, main_cfg.skw, main_cfg.skg, a0, s0, a1, s1, cs.dc2.p, tag.c_str()) != 0) return 0;
    if (ReadC(cs.dc, &c_orig, static_cast<size_t>(kM) * N) != 0) return 2;
    if (ReadC(cs.dc2, &c_perm, static_cast<size_t>(kM) * N) != 0) return 2;
    size_t nd = 0;
    for (int i = 0; i < kM; ++i)
      nd += std::memcmp(&c_perm[static_cast<size_t>(i) * N], &c_orig[static_cast<size_t>(perm[i]) * N], static_cast<size_t>(N) * 2) != 0;
    CHECK(nd == 0, "%s: %zu of 256 permuted rows differ from the same row in the original order", tag.c_str(), nd);
    std::printf("    G  row permutation: %zu of 256 rows differ\n", nd);
  }
  return 0;
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
  // ---- the refusals: host-side, a message names the rule, a launch throws ----
  {
    const auto refused = [](const char* msg, const char* needle) { return msg != nullptr && std::strstr(msg, needle) != nullptr; };
    CHECK(refused(r4d_gemm_trellis_nt_i8_check(64, 5120, 5120, 5120, 4, 4, 1), "M must be 256"), "M = 64 must be refused");
    CHECK(refused(r4d_gemm_trellis_nt_i8_check(256, 5120 + 64, 5120, 5120, 4, 4, 1), "multiples of 128"), "a K that is not a multiple of 128 must be refused");
    CHECK(refused(r4d_gemm_trellis_nt_i8_check(256, 5120, 5120 + 64, 5120 + 64, 4, 4, 1), "multiples of 128"), "an N that is not a multiple of 128 must be refused");
    CHECK(refused(r4d_gemm_trellis_nt_i8_check(256, 5120, 5120, 5120, 6, 4, 1), "KB must be"), "KB 6 must be refused");
    CHECK(refused(r4d_gemm_trellis_nt_i8_check(256, 5120, 5120, 5120, 4, 3, 1), "skw must be"), "skw 3 must be refused");
    CHECK(refused(r4d_gemm_trellis_nt_i8_check(256, 5120, 5120, 5120, 4, 4, 3), "skg must be"), "skg 3 must be refused");
    CHECK(refused(r4d_gemm_trellis_nt_i8_check(256, 5120, 5120, 5120 - 64, 4, 4, 1), "n_split"), "an n_split that is not a multiple of 128 must be refused");
    CHECK(refused(r4d_gemm_trellis_nt_i8_check(256, 5120, 5120, 5120, 4, 8, 8), "divisible"), "K / 128 = 40 not divisible by 64 must be refused");
    CHECK(r4d_gemm_trellis_nt_i8_check(256, 17408, 5120, 5120, 4, 4, 1) == nullptr, "mlp.down (skw 4, skg 1) must be accepted");
    CHECK(r4d_gemm_trellis_nt_i8_check(256, 6144, 5120, 5120, 4, 2, 8) == nullptr, "attn.o (K / 128 = 48, skw 2, skg 8) must be accepted");
    bool threw = false;
    try {
      r4dx::core::r4d::GemmTrellisNtI8(reinterpret_cast<const void*>(16), reinterpret_cast<const void*>(16), nullptr, nullptr, 5120,
                                       reinterpret_cast<const void*>(16), reinterpret_cast<const void*>(16),
                                       reinterpret_cast<const void*>(16), reinterpret_cast<void*>(16), reinterpret_cast<void*>(16),
                                       reinterpret_cast<void*>(16), 256, 5120, 5120, 4, 3, 1, kOutScale, nullptr);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw, "a launch with skw 3 must throw, not run");
    std::printf("host checks: the refusals name their rule and a launch throws\n");
  }
  // the refusals of the coarse entry are the per-128 entry's (one check function); it throws on an illegal launch too
  {
    bool threw = false;
    try {
      r4dx::core::r4d::GemmTrellisNtI8c(reinterpret_cast<const void*>(16), reinterpret_cast<const void*>(16), nullptr, nullptr, 5120,
                                        reinterpret_cast<const void*>(16), reinterpret_cast<const void*>(16),
                                        reinterpret_cast<const void*>(16), reinterpret_cast<void*>(16), reinterpret_cast<void*>(16),
                                        reinterpret_cast<void*>(16), 256, 5120, 5120, 4, 3, 1, kOutScale, nullptr);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw, "a coarse launch with skw 3 must throw, not run");
    threw = false;
    try {
      r4dx::core::r4d::TrellisI8QuantActRow(reinterpret_cast<const void*>(16), reinterpret_cast<void*>(16), reinterpret_cast<void*>(16), 3, 0, 5120, nullptr);
    } catch (const std::runtime_error&) {
      threw = true;
    }
    CHECK(threw, "a row quantizer launch with 3 parts must throw");
    CHECK(r4dx::core::r4d::TrellisI8WscaleColCount(17408, 5120) == 5120, "the column table is N floats");
  }
  for (int kb : {4, 5}) {
    for (const Cls& c : kClasses) {
      const int rc = RunCase(c, kb, 4000u + static_cast<uint32_t>(kb) * 13u + static_cast<uint32_t>(c.N % 997));
      if (rc != 0) return rc;
    }
  }
  // R4DX_PREFILL_INT8_SCALES=coarse: the same classes through the coarse entry
  for (int kb : {4, 5}) {
    for (const Cls& c : kClasses) {
      const int rc = RunCaseCoarse(c, kb, 6000u + static_cast<uint32_t>(kb) * 17u + static_cast<uint32_t>(c.N % 991));
      if (rc != 0) return rc;
    }
  }
  if (g_fail != 0) {
    std::printf("test_trellis_i8_gemm: %d of %d checks FAILED\n", g_fail, g_checks);
    return 1;
  }
  std::printf("test_trellis_i8_gemm: PASS (%zu classes x KB 4/5, %d checks)\n", sizeof(kClasses) / sizeof(kClasses[0]), g_checks);
  return 0;
}
