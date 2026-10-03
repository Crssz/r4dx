// test_trellis_m256: libr4d's M = 256 trellis GEMM (r4d_gemm_trellis_nt_m256, docs/trellis-m256.md) against
// the SHIPPED M = 64 kernel, byte for byte, on synthetic data, for every trellis linear class the model
// wires to it (ApplyLinear's PlanTrellisM256) at both rates.
//
// For each (class, KB) of the model -- mlp.gate_up (two A parts split at 17408), mlp.down, gdn.in_proj_qkv,
// gdn.in_proj_z, gdn.out_proj / attn.o, attn.qg, attn.k / attn.v, at KB 4 and 5 -- the shipped M = 64
// tuning row (PickTuning at M = 64, what a 64-row chunk of that linear runs today) is launched on the four
// 64-row slices of one 256-row activation, and the M = 256 kernel once with the plan ApplyLinear uses for
// that linear; the 256 x N bf16 outputs must be identical, every launch repeated three more times must give
// the same bytes (the running-sum handoff must not race), and the tickets must be back at zero. Weights, A
// and svh are random (any 32-bit word is a valid trellis ring). A negative control launches a shipped
// kernel with a DIFFERENT K-slice count and requires the bytes to differ, so the comparison is known to
// see an accumulation-order change. Also: every class must HAVE a plan, and a shape no configuration
// covers (KB 5 with SK 16) must be refused with a message, not launched.
//
// Skips (77) when there is no HIP device. Needs no golden data. Device 1 only, through the ctest
// environment (HIP_VISIBLE_DEVICES=1).
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "linear.h"
#include "r4d.h"
#include "r4dx/core/r4d.hpp"

namespace {

using r4dx::model::Layout;
using r4dx::model::LinearTuning;

int g_fail = 0;
#define CHECK(cond, ...)                                                    \
  do {                                                                      \
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

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 0x1234567ull) {}
  uint32_t Next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<uint32_t>(s >> 16);
  }
};

struct Cls {
  const char* name;
  int N, K, parts, part_n0;
};
const Cls kClasses[] = {
    {"mlp.gate_up", 34816, 5120, 2, 17408}, {"mlp.down", 5120, 17408, 1, 0},
    {"gdn.in_proj_qkv", 10240, 5120, 1, 0}, {"gdn.in_proj_z", 6144, 5120, 1, 0},
    {"gdn.out_proj/attn.o", 5120, 6144, 1, 0}, {"attn.qg", 12288, 5120, 1, 0},
    {"attn.k/attn.v", 1024, 5120, 1, 0},
    // Gemma 4 12B at TP = 1 (docs/gemma4-plan.md 3.7; the CPU legality test is test_trellis_gemma_plan):
    {"g.gate_up", 30720, 3840, 2, 15360}, {"g.down", 3840, 15360, 1, 0},
    {"g.q_sliding", 4096, 3840, 1, 0},    {"g.q_full", 8192, 3840, 1, 0},
    {"g.kv_sliding", 2048, 3840, 1, 0},   {"g.k_full", 512, 3840, 1, 0},
    {"g.o_sliding", 3840, 4096, 1, 0},    {"g.o_full", 3840, 8192, 1, 0},
};

constexpr float kOutScale = 0.0883883f;  // 1 / sqrt(128), prescale 0

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

int RunCase(const Cls& c, int kb, uint32_t seed) {
  using namespace r4dx;
  const int M = 256, N = c.N, K = c.K;
  const LinearTuning t = model::PickTuning(Layout::kTrellis, N, K, 64, kb);
  const model::TrellisM256Plan plan = model::PlanTrellisM256(N, K, kb, c.parts, c.part_n0);
  const std::string tag = std::string(c.name) + " KB" + std::to_string(kb);
  std::printf("%-20s KB%d shipped M=64 row WV%d SK%d MT%d NP%d SKG%d U%d | plan %s SK%d SKG%d skw%d%s%s\n",
              c.name, kb, t.WV, t.SK, t.MB, t.NPW, t.SKG, t.U, plan.ok ? "ok" : "REFUSED", plan.SK,
              plan.SKG, plan.SKW, plan.ok ? "" : " why: ", plan.ok ? "" : plan.why.c_str());
  CHECK(plan.ok, "%s: the model wires this class to the M = 256 kernel but PlanTrellisM256 refuses it: %s",
        tag.c_str(), plan.why.c_str());
  if (!plan.ok) return 0;
  CHECK(plan.SK == t.SK && plan.SKG == t.SKG, "%s: plan (SK %d, SKG %d) is not the shipped row's (SK %d, SKG %d)",
        tag.c_str(), plan.SK, plan.SKG, t.SK, t.SKG);
  const int wc = t.WV * t.NPW * 32;
  if (c.parts > 1) CHECK(c.part_n0 % wc == 0, "%s: shipped row's block width %d crosses the part boundary", tag.c_str(), wc);

  const size_t w_words = static_cast<size_t>(N) * K * kb / 32;
  std::vector<uint32_t> hw(w_words);
  Rng rng(seed);
  for (uint32_t& x : hw) x = rng.Next() ^ (rng.Next() << 16);
  std::vector<uint16_t> ha(static_cast<size_t>(2) * M * K);
  for (uint16_t& x : ha) {
    const uint32_t h = rng.Next();
    x = static_cast<uint16_t>((h & 0x8000u) | (0x3000u + ((h >> 4) & 0x0BFFu)));  // finite f16, |x| in [2^-3, 1)
  }
  std::vector<float> hs(N);
  for (float& x : hs) {
    const uint32_t h = rng.Next();
    const float m = 0.5f + static_cast<float>(h & 0xFFFFu) / 65536.f;
    x = (h & 0x10000u) ? -m : m;
  }
  Dev<uint32_t> dw;
  Dev<uint16_t> da, dc0, dc1;
  Dev<float> dsvh, ws0, ws1;
  Dev<unsigned> tk0, tk1;
  HIP(dw.Alloc(w_words));
  HIP(da.Alloc(ha.size()));
  HIP(dsvh.Alloc(N));
  HIP(dc0.Alloc(static_cast<size_t>(M) * N));
  HIP(dc1.Alloc(static_cast<size_t>(M) * N));
  HIP(ws0.Alloc(core::r4d::GemmTrellisWsBytes(64, N, t.SKG) / sizeof(float)));
  HIP(ws1.Alloc(core::r4d::GemmTrellisM256WsBytes(M, N, plan.SKG) / sizeof(float)));
  HIP(tk0.Alloc(N / 128));
  HIP(tk1.Alloc(N / 128));
  HIP(hipMemcpy(dw.p, hw.data(), hw.size() * 4, hipMemcpyHostToDevice));
  HIP(hipMemcpy(da.p, ha.data(), ha.size() * 2, hipMemcpyHostToDevice));
  HIP(hipMemcpy(dsvh.p, hs.data(), hs.size() * 4, hipMemcpyHostToDevice));
  HIP(hipMemset(tk0.p, 0, static_cast<size_t>(N / 128) * 4));
  HIP(hipMemset(tk1.p, 0, static_cast<size_t>(N / 128) * 4));

  std::vector<uint16_t> ref(static_cast<size_t>(M) * N), got(ref.size());
  const auto stock = [&](const LinearTuning& tt, int n_split, const uint16_t* a1) {
    for (int m0 = 0; m0 < M; m0 += 64) {
      core::r4d::GemmTrellisNtM64(da.p + static_cast<size_t>(m0) * K, a1 ? a1 + static_cast<size_t>(m0) * K : nullptr,
                                  n_split, dw.p, dsvh.p, dc0.p + static_cast<size_t>(m0) * N, ws0.p, tk0.p, 64, K, N,
                                  kb, tt.WV, tt.SK, tt.MB, tt.NPW, tt.SKG, tt.U, tt.NT, kOutScale, nullptr);
    }
  };
  const auto big = [&](int n_split, const uint16_t* a1) {
    core::r4d::GemmTrellisNtM256(da.p, a1, n_split, dw.p, dsvh.p, dc1.p, ws1.p, tk1.p, M, K, N, kb, plan.SK, 1,
                                 plan.SKG, 4, kOutScale, plan.SKW, nullptr);
  };
  struct Form {
    const char* name;
    int n_split;
    const uint16_t* a1;
  };
  std::vector<Form> forms = {{"one-A", N, nullptr}};
  if (c.parts > 1) forms.push_back({"two-part", c.part_n0, da.p + static_cast<size_t>(M) * K});
  for (const Form& f : forms) {
    HIP(hipMemset(dc0.p, 0xFF, static_cast<size_t>(M) * N * 2));
    HIP(hipMemset(dc1.p, 0xFF, static_cast<size_t>(M) * N * 2));
    try {
      stock(t, f.n_split, f.a1);
      big(f.n_split, f.a1);
    } catch (const std::exception& e) {
      CHECK(false, "%s %s: launch threw: %s", tag.c_str(), f.name, e.what());
      continue;
    }
    HIP(hipDeviceSynchronize());
    HIP(hipMemcpy(ref.data(), dc0.p, ref.size() * 2, hipMemcpyDeviceToHost));
    HIP(hipMemcpy(got.data(), dc1.p, got.size() * 2, hipMemcpyDeviceToHost));
    size_t nd = 0, nonfinite = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
      nd += ref[i] != got[i];
      nonfinite += (ref[i] & 0x7F80u) == 0x7F80u;
    }
    CHECK(nd == 0, "%s %s: %zu of %zu bf16 outputs of one M=256 launch differ from four shipped M=64 launches",
          tag.c_str(), f.name, nd, ref.size());
    CHECK(nonfinite == 0, "%s %s: %zu non-finite outputs (bad test data)", tag.c_str(), f.name, nonfinite);
    size_t rep_diff = 0;
    for (int rep = 0; rep < 3; ++rep) {
      HIP(hipMemset(dc1.p, 0xFF, static_cast<size_t>(M) * N * 2));
      big(f.n_split, f.a1);
      HIP(hipDeviceSynchronize());
      std::vector<uint16_t> again(got.size());
      HIP(hipMemcpy(again.data(), dc1.p, again.size() * 2, hipMemcpyDeviceToHost));
      rep_diff += again != got;
    }
    CHECK(rep_diff == 0, "%s %s: %zu of 3 repeats of the M=256 launch gave other bytes", tag.c_str(), f.name, rep_diff);
    std::vector<unsigned> t0(N / 128), t1(N / 128);
    HIP(hipMemcpy(t0.data(), tk0.p, t0.size() * 4, hipMemcpyDeviceToHost));
    HIP(hipMemcpy(t1.data(), tk1.p, t1.size() * 4, hipMemcpyDeviceToHost));
    size_t tbad = 0;
    for (unsigned x : t0) tbad += x != 0;
    for (unsigned x : t1) tbad += x != 0;
    CHECK(tbad == 0, "%s %s: %zu tickets not reset", tag.c_str(), f.name, tbad);
    std::printf("    %-9s %zu bf16 compared, %zu differ%s\n", f.name, ref.size(), nd, nd == 0 ? " (identical)" : "");
  }

  // negative control: a shipped kernel with another K-slice count must NOT match (an accumulation-order
  // change is visible to this comparison). Try SK / 2, then SK * 2; the first legal one is the control.
  {
    LinearTuning alt = t;
    bool have = false;
    for (int sk : {t.SK / 2, t.SK * 2}) {
      if (sk < 1 || sk > 16) continue;
      alt.SK = sk;
      HIP(hipMemset(dc0.p, 0xFF, static_cast<size_t>(M) * N * 2));
      try {
        stock(alt, N, nullptr);
        HIP(hipDeviceSynchronize());
        have = true;
        break;
      } catch (const std::exception&) {
        HIP(hipMemset(tk0.p, 0, static_cast<size_t>(N / 128) * 4));
      }
    }
    if (have) {
      HIP(hipMemset(dc1.p, 0xFF, static_cast<size_t>(M) * N * 2));
      big(N, nullptr);
      HIP(hipDeviceSynchronize());
      HIP(hipMemcpy(ref.data(), dc0.p, ref.size() * 2, hipMemcpyDeviceToHost));
      HIP(hipMemcpy(got.data(), dc1.p, got.size() * 2, hipMemcpyDeviceToHost));
      size_t nd = 0;
      for (size_t i = 0; i < ref.size(); ++i) nd += ref[i] != got[i];
      CHECK(nd > 0, "%s: negative control (shipped kernel at SK %d instead of %d) found no differing byte -- the "
            "comparison cannot see an accumulation-order change", tag.c_str(), alt.SK, t.SK);
      std::printf("    control   SK %d vs M=256 at SK %d: %zu of %zu bf16 differ (must be > 0)\n", alt.SK, plan.SK, nd,
                  ref.size());
    } else {
      std::printf("    control   no legal shipped kernel at another SK for this shape (skipped)\n");
    }
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
  // Host-side: the refusals are messages, never launches.
  {
    const auto plan = r4dx::model::PlanTrellisM256(5120, 17408, /*kb=*/6, 1, 0);
    CHECK(!plan.ok && !plan.why.empty(), "KB 6 must be refused with a reason");
    const char* bad_m = r4d_gemm_trellis_nt_m256_check(64, 5120, 5120, 5120, 4, 4, 1, 1, 4, 0);
    CHECK(bad_m != nullptr && std::strstr(bad_m, "M must be") != nullptr, "M = 64 must be refused: %s",
          bad_m ? bad_m : "(accepted)");
    // KB 5 with SK 16 (mlp.down's KB 4 row shape) has no instantiation: 192 VGPRs and scratch
    const char* kb5 = r4d_gemm_trellis_nt_m256_check(256, 17408, 5120, 5120, 5, 16, 1, 2, 4, 0);
    CHECK(kb5 != nullptr && std::strstr(kb5, "not instantiated") != nullptr, "KB 5 SK 16 must be refused: %s",
          kb5 ? kb5 : "(accepted)");
    // a K whose per-slice k-tile count leaves a tail no instantiation covers (K/16 = 80 tiles / (SK 8 x SKG 2)... = 5)
    const char* tail = r4d_gemm_trellis_nt_m256_check(256, 1280, 5120, 5120, 4, 8, 1, 2, 4, 0);
    CHECK(tail != nullptr, "an uncovered k-tile tail must be refused");
    const char* ok = r4d_gemm_trellis_nt_m256_check(256, 17408, 5120, 5120, 4, 16, 1, 2, 4, 4);
    CHECK(ok == nullptr, "mlp.down KB 4 (SK 16, SKG 2, skw 4) must be accepted: %s", ok ? ok : "");
    std::printf("host checks: refusals name their rule; mlp.down KB 4 accepted\n");
  }
  for (int kb : {4, 5}) {
    for (const Cls& c : kClasses) {
      const int rc = RunCase(c, kb, 1000u + static_cast<uint32_t>(kb));
      if (rc != 0) return rc;
    }
  }
  if (g_fail != 0) {
    std::printf("test_trellis_m256: %d FAILED\n", g_fail);
    return 1;
  }
  std::printf("test_trellis_m256: PASS (%zu classes x KB 4/5: one M=256 launch == four shipped M=64 launches, byte for byte)\n",
              sizeof(kClasses) / sizeof(kClasses[0]));
  return 0;
}
