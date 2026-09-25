// tests/kernels/test_attn_gate_mul_hadamard.cpp -- r4dx_model_attn_gate_mul_hadamard_bf16
// (src/model/attention, quant2 Q2b: attn.o's input rotated inside the output-gate multiply) against
// the fp64 CPU reference in rotation_ref.hpp:
//   out[t, :] = (attn_out[t, :] * sigmoid(gate[t, :])) Hb,  Hb blockwise with block 256 = one head
// over the [T, H, D] contiguous buffers, at the full model's K = 24 x 256 and a TP=2 rank's
// 12 x 256, T = 1 / 3. Plus, bit-exact: row independence, TP rank-1 K-slice == the full result's
// slice, and the precondition throws (hence /EHc- in CMakeLists.txt). Lives here rather than under
// tests/model/attention because it is a kernel-vs-reference test like its r4dx_kernels siblings
// (test_rotate_residual) and shares their reference header.
// GPU test: HIP device 1 via HIP_VISIBLE_DEVICES=1 (tests/kernels/CMakeLists.txt), always on.
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/model/attention/attn_kernels.h"
#include "rotation_ref.hpp"

using namespace r4dx::core;

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool cond, const std::string& what) {
  ++g_checks;
  if (!cond) {
    std::printf("FAIL: %s\n", what.c_str());
    ++g_failures;
  }
}

template <typename T>
int64_t P(T* p) {
  return reinterpret_cast<int64_t>(p);
}

std::vector<uint16_t> RandomBf16(std::mt19937_64& rng, size_t n, float lo, float hi) {
  std::uniform_real_distribution<float> dist(lo, hi);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = FloatToBf16(dist(rng));
  return v;
}

std::vector<uint16_t> Run(const std::vector<uint16_t>& a, const std::vector<uint16_t>& g,
                          int64_t row_len, const float* signs_dev, int block) {
  DeviceBuffer<uint16_t> a_d(a.size()), g_d(g.size()), o_d(a.size());
  a_d.CopyFromHost(a);
  g_d.CopyFromHost(g);
  r4dx_model_attn_gate_mul_hadamard_bf16(P(a_d.data()), P(g_d.data()), P(o_d.data()),
                                          static_cast<int64_t>(a.size()), 0, P(signs_dev), block,
                                          row_len);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  return o_d.CopyToHost();
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

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937_64 rng(20260926);
  constexpr int kHeadDim = 256;  // attn.o's Hb block: one head

  struct Case {
    int heads;
    const char* what;
  };
  for (const Case& c : {Case{24, "o TP=1"}, Case{12, "o TP=2 rank"}}) {
    const int64_t K = static_cast<int64_t>(c.heads) * kHeadDim;
    const std::vector<float> s = rotation_ref::RandomSigns(rng, K);
    DeviceBuffer<float> s_d(s.size());
    s_d.CopyFromHost(s);
    for (int64_t T : {1, 3}) {
      const std::string name = std::string("attn gate_mul_hadamard ") + c.what + " T=" +
                               std::to_string(T);
      // attn_out ~ an attention output's range; gate over a range that exercises both sigmoid tails.
      const std::vector<uint16_t> a = RandomBf16(rng, static_cast<size_t>(T * K), -3.0f, 3.0f);
      const std::vector<uint16_t> g = RandomBf16(rng, static_cast<size_t>(T * K), -8.0f, 8.0f);
      const std::vector<uint16_t> got = Run(a, g, K, s_d.data(), kHeadDim);

      for (int64_t t = 0; t < T; ++t) {
        std::vector<double> h(static_cast<size_t>(K));
        for (int64_t k = 0; k < K; ++k) {
          const double av = Bf16ToFloat(a[t * K + k]);
          const double gv = Bf16ToFloat(g[t * K + k]);
          h[k] = av / (1.0 + std::exp(-gv));
        }
        const std::vector<double> ref = rotation_ref::ApplyHb(h, s.data(), kHeadDim);
        std::vector<float> gf(static_cast<size_t>(K));
        for (int64_t k = 0; k < K; ++k) gf[k] = Bf16ToFloat(got[t * K + k]);
        double worst = 0.0;
        const int64_t bad = rotation_ref::CountOutOfTolerance(gf, ref, &worst);
        if (t == 0) std::printf("  %-52s worst err/bound %.3f\n", name.c_str(), worst);
        Check(bad == 0, name + " row " + std::to_string(t) + ": " + std::to_string(bad) +
                            " elements out of tolerance (worst err/bound " + std::to_string(worst) +
                            ")");
      }

      // Row independence: each row alone == its row of the batch.
      for (int64_t t = 0; t < T && T > 1; ++t) {
        std::vector<uint16_t> a1(a.begin() + t * K, a.begin() + (t + 1) * K);
        std::vector<uint16_t> g1(g.begin() + t * K, g.begin() + (t + 1) * K);
        const std::vector<uint16_t> o1 = Run(a1, g1, K, s_d.data(), kHeadDim);
        Check(std::memcmp(o1.data(), &got[t * K], sizeof(uint16_t) * K) == 0,
              name + ": row " + std::to_string(t) + " launched alone differs from the batch");
      }

      // TP=2 slicing: rank 1's heads [H/2, H) with the sign slice [K/2, K) == the full result's
      // columns [K/2, K) (head-split o keeps whole heads = whole Hb blocks per rank).
      const int64_t half = K / 2;
      std::vector<uint16_t> ar(static_cast<size_t>(T * half)), gr(static_cast<size_t>(T * half));
      for (int64_t t = 0; t < T; ++t) {
        std::memcpy(&ar[t * half], &a[t * K + half], sizeof(uint16_t) * half);
        std::memcpy(&gr[t * half], &g[t * K + half], sizeof(uint16_t) * half);
      }
      const std::vector<uint16_t> part = Run(ar, gr, half, s_d.data() + half, kHeadDim);
      bool same = true;
      for (int64_t t = 0; t < T && same; ++t) {
        same = std::memcmp(&part[t * half], &got[t * K + half], sizeof(uint16_t) * half) == 0;
      }
      Check(same, name + ": TP rank-1 K-slice differs from the full-K result's slice");
    }
  }

  DeviceBuffer<float> s_d(6144);
  Check(Throws([&] {
          r4dx_model_attn_gate_mul_hadamard_bf16(1, 1, 1, 6144, 0, P(s_d.data()), 384, 6144);
        }),
        "non-power-of-two block must throw");
  Check(Throws([&] {
          r4dx_model_attn_gate_mul_hadamard_bf16(1, 1, 1, 6144 + 256, 0, P(s_d.data()), 256, 6144);
        }),
        "n % row_len != 0 must throw");
  Check(Throws([&] {
          r4dx_model_attn_gate_mul_hadamard_bf16(1, 1, 1, 6144, 0, P(s_d.data()), 256, 6144 + 128);
        }),
        "row_len % block != 0 must throw");

  if (g_failures == 0) {
    std::printf("PASS (%d checks)\n", g_checks);
    return 0;
  }
  std::printf("FAIL (%d of %d checks)\n", g_failures, g_checks);
  return 1;
}
