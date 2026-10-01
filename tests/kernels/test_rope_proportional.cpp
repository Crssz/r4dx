// tests/kernels/test_rope_proportional.cpp -- r4dx_rope_proportional_bf16 (docs/gemma4-plan.md 3.4)
// vs gemma_ref.hpp::HfRope (HF's inv_freq-with-zero-tail construction), for both Gemma 4 geometries:
// the full layer (16 q / 1 kv head, head_dim 512, 64 rotated pairs, freq_dim 512, theta 1e6) and the
// sliding layer (16 / 8, head_dim 256, 128 pairs, freq_dim 256, theta 1e4). Also: unrotated dims are
// bit-untouched, n_pairs = head_dim / 2 equals r4dx_rope_neox_bf16 bit for bit, the Q-only / K-only
// calling forms, and the precondition throws.
// GPU test (HIP device 1 via ctest's ENVIRONMENT).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <stdexcept>
#include <vector>

#include "gemma_ref.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/gemma_kernels.h"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

namespace {

bool g_ok = true;
void Check(bool cond, const char* what) {
  std::printf("%-76s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}

int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

struct Geo {
  const char* name;
  int heads_q, heads_k, head_dim, n_pairs, freq_dim;
  float theta;
};

// One head vector through HfRope, rounded to bf16 once like the kernel.
std::vector<uint16_t> RefRope(const std::vector<uint16_t>& in, int tokens, int heads, const Geo& g,
                              const std::vector<int32_t>& pos) {
  std::vector<uint16_t> out(in.size());
  for (int t = 0; t < tokens; ++t) {
    for (int h = 0; h < heads; ++h) {
      const size_t base = (static_cast<size_t>(t) * heads + h) * g.head_dim;
      std::vector<double> x(g.head_dim);
      for (int j = 0; j < g.head_dim; ++j) x[j] = Bf16ToFloat(in[base + j]);
      const std::vector<double> y = gemma_ref::HfRope(x, g.n_pairs, g.freq_dim, g.theta, pos[t], true);
      for (int j = 0; j < g.head_dim; ++j) out[base + j] = FloatToBf16(static_cast<float>(y[j]));
    }
  }
  return out;
}

// bf16 step at |x| ~ 4 is 3.1e-2; the angle's fp32 formation differs by an inv_freq ulp * pos.
bool Close(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref, double* worst) {
  *worst = 0.0;
  bool ok = true;
  for (size_t i = 0; i < got.size(); ++i) {
    const double d = std::abs(static_cast<double>(Bf16ToFloat(got[i])) - Bf16ToFloat(ref[i]));
    *worst = std::max(*worst, d);
    if (d > 0.04 + 0.01 * std::abs(static_cast<double>(Bf16ToFloat(ref[i])))) ok = false;
  }
  return ok;
}

void RunGeometry(const Geo& g, std::mt19937& rng) {
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
  const int tokens = 11;
  std::vector<uint16_t> q_h(static_cast<size_t>(tokens) * g.heads_q * g.head_dim);
  std::vector<uint16_t> k_h(static_cast<size_t>(tokens) * g.heads_k * g.head_dim);
  for (auto& v : q_h) v = FloatToBf16(dist(rng));
  for (auto& v : k_h) v = FloatToBf16(dist(rng));
  std::uniform_int_distribution<int> pd(0, 4096);
  std::vector<int32_t> pos(tokens);
  for (auto& p : pos) p = pd(rng);
  pos[0] = 0;

  DeviceBuffer<uint16_t> q_d(q_h.size()), k_d(k_h.size());
  DeviceBuffer<int32_t> p_d(pos.size());
  q_d.CopyFromHost(q_h);
  k_d.CopyFromHost(k_h);
  p_d.CopyFromHost(pos);
  r4dx_rope_proportional_bf16(P(q_d.data()), P(k_d.data()), P(p_d.data()), tokens, g.heads_q,
                               g.heads_k, g.head_dim, g.n_pairs, g.freq_dim, g.theta, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<uint16_t> q_got = q_d.CopyToHost(), k_got = k_d.CopyToHost();

  double wq, wk;
  char msg[160];
  std::snprintf(msg, sizeof msg, "%s: q matches HF construction", g.name);
  const bool qok = Close(q_got, RefRope(q_h, tokens, g.heads_q, g, pos), &wq);
  Check(qok, msg);
  std::snprintf(msg, sizeof msg, "%s: k matches HF construction", g.name);
  const bool kok = Close(k_got, RefRope(k_h, tokens, g.heads_k, g, pos), &wk);
  Check(kok, msg);
  std::printf("    max |d| q %.3e k %.3e\n", wq, wk);

  // Unrotated dims (pair index >= n_pairs, both halves) are bit-untouched; row 0 (pos 0) is the identity.
  const int half = g.head_dim / 2;
  bool untouched = true, identity = true;
  for (size_t i = 0; i < q_h.size(); ++i) {
    const int j = static_cast<int>(i % g.head_dim);
    if (j % half >= g.n_pairs && q_got[i] != q_h[i]) untouched = false;
    if (i < static_cast<size_t>(g.heads_q) * g.head_dim && q_got[i] != q_h[i]) identity = false;
  }
  std::snprintf(msg, sizeof msg, "%s: dims outside the rotated pairs are bit-untouched", g.name);
  Check(untouched, msg);
  std::snprintf(msg, sizeof msg, "%s: position 0 is the bit-exact identity", g.name);
  Check(identity, msg);

  // Q-only and K-only forms equal the combined call.
  DeviceBuffer<uint16_t> qo(q_h.size()), ko(k_h.size());
  qo.CopyFromHost(q_h);
  ko.CopyFromHost(k_h);
  r4dx_rope_proportional_bf16(P(qo.data()), 0, P(p_d.data()), tokens, g.heads_q, 0, g.head_dim,
                               g.n_pairs, g.freq_dim, g.theta, 0);
  r4dx_rope_proportional_bf16(0, P(ko.data()), P(p_d.data()), tokens, 0, g.heads_k, g.head_dim,
                               g.n_pairs, g.freq_dim, g.theta, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::snprintf(msg, sizeof msg, "%s: q-only and k-only calls equal the combined call", g.name);
  Check(qo.CopyToHost() == q_got && ko.CopyToHost() == k_got, msg);
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(21);

  RunGeometry({"full    (16/1 x 512, 64 pairs, 1e6)", 16, 1, 512, 64, 512, 1.0e6f}, rng);
  RunGeometry({"sliding (16/8 x 256, 128 pairs, 1e4)", 16, 8, 256, 128, 256, 1.0e4f}, rng);

  // n_pairs = head_dim / 2, freq_dim = head_dim is r4dx_rope_neox_bf16, bit for bit.
  {
    const int tokens = 9, heads = 8, head_dim = 256;
    std::uniform_real_distribution<float> dist(-3.0f, 3.0f);
    std::vector<uint16_t> h(static_cast<size_t>(tokens) * heads * head_dim);
    for (auto& v : h) v = FloatToBf16(dist(rng));
    std::vector<int32_t> pos(tokens);
    for (int t = 0; t < tokens; ++t) pos[t] = 100 * t + 3;
    DeviceBuffer<uint16_t> a(h.size()), b(h.size());
    DeviceBuffer<int32_t> p_d(pos.size());
    a.CopyFromHost(h);
    b.CopyFromHost(h);
    p_d.CopyFromHost(pos);
    r4dx_rope_proportional_bf16(P(a.data()), 0, P(p_d.data()), tokens, heads, 0, head_dim, head_dim / 2,
                                 head_dim, 1.0e4f, 0);
    r4dx_rope_neox_bf16(P(b.data()), 0, P(p_d.data()), tokens, heads, 0, head_dim, 1.0e4f, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    Check(a.CopyToHost() == b.CopyToHost(), "n_pairs = head_dim/2 is r4dx_rope_neox_bf16 bit for bit");
  }

  // Preconditions throw (test built with /EHc-, see tests/kernels/CMakeLists.txt).
  {
    DeviceBuffer<uint16_t> x(512 * 4);
    DeviceBuffer<int32_t> p_d(1);
    const auto throws = [&](int head_dim, int n_pairs, int freq_dim) {
      try {
        r4dx_rope_proportional_bf16(P(x.data()), 0, P(p_d.data()), 1, 1, 0, head_dim, n_pairs, freq_dim,
                                     1.0e4f, 0);
      } catch (const std::runtime_error&) {
        return true;
      }
      return false;
    };
    Check(throws(255, 64, 256), "odd head_dim throws");
    Check(throws(512, 0, 512), "n_pairs 0 throws");
    Check(throws(512, 257, 512), "n_pairs > head_dim/2 throws");
    Check(throws(512, 64, 511), "odd freq_dim throws");
  }

  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
