// tests/kernels/test_lfm2_rope_norm64.cpp -- the EXISTING kernels at LFM2's head_dim 64 (docs/lfm2-plan.md 3.4,
// task L1-7; there is no new kernel: r4dx_rope_neox_bf16 and r4dx_rmsnorm_plain_bf16 are generic in the
// width). Pins them against lfm2_ref.hpp / an fp32-angle NeoX rope:
//   * rope: 32 q heads x 8 kv heads x 64, theta 1e7, positions 0..69 and large ones (up to 131071), q and k
//     in one call, q-only and k-only forms; the angle is formed in fp32 like the kernel (pos * inv_freq as
//     floats, a 131071-position angle is NOT exact in fp32 -- that is the kernel's and HF's own rounding),
//     rotate_half pairing (i, i + 32); tolerance 2^-7 relative + 0.01 absolute;
//   * qk rmsnorm: per-head plain rmsnorm over rows T * heads with hidden = 64, eps 1e-5, bf16 weight [64]
//     (LFM2's q_layernorm / k_layernorm), out of place and in place, vs the one-rounding reference; the
//     distance to HF's two-rounding op (lfm2_ref::RmsNormHf) is printed;
//   * the full attention pre-core order: norm then rope (LFM2: qk-norm BEFORE rope) on [T, 32, 64] / [T, 8, 64].
// GPU test (HIP device 1 via HIP_VISIBLE_DEVICES=1 in CMakeLists.txt); WRITTEN, NOT YET RUN.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "lfm2_test_util.hpp"
#include "r4dx/kernels/kernels.h"
#include "kernels/model_kernels.h"
#include "r4dx/kernels/lfm2_kernels.h"

using namespace r4dx::core;
using namespace lfm2_test;

namespace {

constexpr int kHd = 64;
constexpr float kTheta = 1.0e7f;

// NeoX rotate_half rope with the kernel's fp32 angle: inv_freq = exp2(-(2i)/hd * log2(theta)) in fp32,
// angle = double(float(pos) * inv_freq).
std::vector<float> RopeRef(const std::vector<float>& x, int rows, int heads, const std::vector<int32_t>& pos) {
  std::vector<float> o = x;
  const int half = kHd / 2;
  for (int t = 0; t < rows; ++t) {
    for (int i = 0; i < half; ++i) {
      const float inv_freq = std::exp2(-(2.0f * i) / static_cast<float>(kHd) * std::log2(kTheta));
      const double angle = static_cast<double>(static_cast<float>(pos[t]) * inv_freq);
      const double c = std::cos(angle), s = std::sin(angle);
      for (int h = 0; h < heads; ++h) {
        const size_t base = (static_cast<size_t>(t) * heads + h) * kHd;
        const double x0 = x[base + i], x1 = x[base + i + half];
        o[base + i] = lfm2_ref::Bf16Round(static_cast<float>(x0 * c - x1 * s));
        o[base + i + half] = lfm2_ref::Bf16Round(static_cast<float>(x1 * c + x0 * s));
      }
    }
  }
  return o;
}

bool Close(const std::vector<float>& got, const std::vector<float>& want, double rel, double abs_tol, size_t* worst_i = nullptr) {
  bool ok = true;
  double worst = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double e = std::fabs(static_cast<double>(got[i]) - want[i]);
    const double tol = abs_tol + rel * std::fabs(static_cast<double>(want[i]));
    if (e > tol) ok = false;
    if (e - tol > worst) {
      worst = e - tol;
      if (worst_i) *worst_i = i;
    }
  }
  return ok;
}

std::vector<float> RunRope(const std::vector<float>& q, const std::vector<float>& k, int T, int hq, int hk,
                           const std::vector<int32_t>& pos, std::vector<float>* k_out, bool do_q, bool do_k) {
  DeviceBuffer<uint16_t> q_d(std::max<size_t>(q.size(), 1)), k_d(std::max<size_t>(k.size(), 1));
  DeviceBuffer<int32_t> p_d(pos.size());
  q_d.CopyFromHost(ToBits(q));
  k_d.CopyFromHost(ToBits(k));
  p_d.CopyFromHost(pos);
  r4dx_rope_neox_bf16(do_q ? P(q_d.data()) : 0, do_k ? P(k_d.data()) : 0, P(p_d.data()), T, do_q ? hq : 0,
                      do_k ? hk : 0, kHd, kTheta, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  *k_out = FromBits(k_d.CopyToHost());
  return FromBits(q_d.CopyToHost());
}

void TestRope(std::mt19937& rng) {
  const int T = 70, hq = 32, hk = 8;
  const auto q = RandBf16(rng, static_cast<size_t>(T) * hq * kHd, 2.0f);
  const auto k = RandBf16(rng, static_cast<size_t>(T) * hk * kHd, 2.0f);
  for (int variant = 0; variant < 2; ++variant) {
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = variant == 0 ? t : (t * 1873) % 131072;
    pos[0] = 0;
    pos[T - 1] = 131071;
    std::vector<float> k_got;
    const auto q_got = RunRope(q, k, T, hq, hk, pos, &k_got, true, true);
    const auto q_want = RopeRef(q, T, hq, pos), k_want = RopeRef(k, T, hk, pos);
    const std::string nm = variant == 0 ? "rope64 positions 0..69" : "rope64 positions up to 131071";
    CheckF(Close(q_got, q_want, 1.0 / 128, 0.01), nm + ": q (32 heads) == fp32-angle NeoX reference");
    CheckF(Close(k_got, k_want, 1.0 / 128, 0.01), nm + ": k (8 heads) == fp32-angle NeoX reference");
  }
  // q only / k only leave the other untouched
  std::vector<int32_t> pos(T);
  for (int t = 0; t < T; ++t) pos[t] = 5 * t + 3;
  std::vector<float> k_got;
  const auto q_only = RunRope(q, k, T, hq, hk, pos, &k_got, true, false);
  Check(k_got == k && Close(q_only, RopeRef(q, T, hq, pos), 1.0 / 128, 0.01), "rope64: q-only form rotates q and leaves k bytes untouched");
  const auto q_same = RunRope(q, k, T, hq, hk, pos, &k_got, false, true);
  Check(q_same == q && Close(k_got, RopeRef(k, T, hk, pos), 1.0 / 128, 0.01), "rope64: k-only form rotates k and leaves q bytes untouched");
  // position 0 is the identity, bitwise
  std::vector<int32_t> zero(T, 0);
  const auto q0 = RunRope(q, k, T, hq, hk, zero, &k_got, true, true);
  Check(q0 == q && k_got == k, "rope64: position 0 leaves q and k bitwise unchanged");
}

std::vector<float> RunNorm(const std::vector<float>& x, const std::vector<float>& w, int rows, float eps, bool in_place) {
  DeviceBuffer<uint16_t> x_d(x.size()), w_d(w.size()), o_d(in_place ? 1 : x.size());
  x_d.CopyFromHost(ToBits(x));
  w_d.CopyFromHost(ToBits(w));
  const int64_t out = in_place ? P(x_d.data()) : P(o_d.data());
  r4dx_rmsnorm_plain_bf16(P(x_d.data()), P(w_d.data()), out, rows, kHd, eps, 0, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  return FromBits(in_place ? x_d.CopyToHost() : o_d.CopyToHost());
}

void TestNorm(std::mt19937& rng) {
  for (int heads : {32, 8}) {
    const int T = 256, rows = T * heads;
    const auto x = RandBf16(rng, static_cast<size_t>(rows) * kHd, 3.0f);
    auto w = RandBf16(rng, kHd, 2.0f);
    for (auto& v : w) v = std::fabs(v) + 0.1f;
    const auto want = lfm2_ref::RmsNormPlain(x, w, rows, kHd, 1e-5f);
    const auto got = RunNorm(x, w, rows, 1e-5f, false);
    const auto got_ip = RunNorm(x, w, rows, 1e-5f, true);
    const Diff d = Compare(got, want, 1.0f);
    CheckF(d.over == 0, "qk rmsnorm hidden 64, " + std::to_string(rows) + " rows (T 256 x " + std::to_string(heads) +
                           " heads): within 1 bf16 ulp of the one-rounding reference");
    CheckF(got_ip == got, "qk rmsnorm hidden 64: in place == out of place (bitwise)");
    const auto hf = lfm2_ref::RmsNormHf(x, w, rows, kHd, 1e-5f);
    const Diff dh = Compare(got, hf, 2.0f);
    std::printf("      vs HF two-rounding op: %zu / %zu differ, %zu beyond 2 ulp\n", dh.differ, dh.n, dh.over);
  }
}

void TestPreCoreOrder(std::mt19937& rng) {
  // LFM2 attention: q = q_layernorm(q_proj(x) viewed [T, 32, 64]) THEN rope. Chain the two kernels on the device
  // and compare with the chain of references.
  const int T = 40, hq = 32;
  const auto q = RandBf16(rng, static_cast<size_t>(T) * hq * kHd, 3.0f);
  auto w = RandBf16(rng, kHd, 2.0f);
  for (auto& v : w) v = std::fabs(v) + 0.1f;
  std::vector<int32_t> pos(T);
  for (int t = 0; t < T; ++t) pos[t] = 1000 * t + 7;
  DeviceBuffer<uint16_t> q_d(q.size()), w_d(w.size());
  DeviceBuffer<int32_t> p_d(T);
  q_d.CopyFromHost(ToBits(q));
  w_d.CopyFromHost(ToBits(w));
  p_d.CopyFromHost(pos);
  r4dx_rmsnorm_plain_bf16(P(q_d.data()), P(w_d.data()), P(q_d.data()), static_cast<int64_t>(T) * hq, kHd, 1e-5f, 0, 0);
  r4dx_rope_neox_bf16(P(q_d.data()), 0, P(p_d.data()), T, hq, 0, kHd, kTheta, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const auto got = FromBits(q_d.CopyToHost());
  const auto want = RopeRef(lfm2_ref::RmsNormPlain(q, w, T * hq, kHd, 1e-5f), T, hq, pos);
  Check(Close(got, want, 3.0 / 128, 0.02), "qk-norm then rope chain (q [40, 32, 64]) == reference chain");
}

void TestFusions(std::mt19937& rng) {
  // 1. r4dx_lfm2_residual_rmsnorm_plain_bf16 == r4dx_residual_add_bf16 + r4dx_rmsnorm_plain_bf16 (bitwise)
  {
    const int64_t rows = 16, hidden = 2048;
    const auto h0 = ToBits(RandBf16(rng, rows * hidden, 2.0f));
    const auto sub = ToBits(RandBf16(rng, rows * hidden, 1.5f));
    const auto w = ToBits(RandBf16(rng, hidden, 1.0f));
    DeviceBuffer<uint16_t> h_a(h0.size()), h_b(h0.size()), sub_d(sub.size()), w_d(w.size());
    DeviceBuffer<uint16_t> x_a(h0.size()), x_b(h0.size());
    h_a.CopyFromHost(h0);
    h_b.CopyFromHost(h0);
    sub_d.CopyFromHost(sub);
    w_d.CopyFromHost(w);
    r4dx_residual_add_bf16(P(h_a.data()), P(sub_d.data()), P(h_a.data()), rows * hidden, 0);
    r4dx_rmsnorm_plain_bf16(P(h_a.data()), P(w_d.data()), P(x_a.data()), rows, hidden, 1e-5f, 0, 0);
    r4dx_lfm2_residual_rmsnorm_plain_bf16(P(h_b.data()), P(sub_d.data()), P(w_d.data()), P(x_b.data()), rows, hidden,
                                          1e-5f, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    Check(h_a.CopyToHost() == h_b.CopyToHost() && x_a.CopyToHost() == x_b.CopyToHost(),
          "r4dx_lfm2_residual_rmsnorm_plain_bf16 == residual_add + rmsnorm_plain (bitwise on h and x_out)");
  }
  // 2. r4dx_lfm2_attn_precore_bf16 == q_norm + k_norm + rope_neox + kv_write_paged_bf16_hnd (bitwise)
  {
    const int T = 16, hq = 32, hk = 8, block_size = 16, max_blocks = 8;
    const int64_t head_stride = static_cast<int64_t>(block_size) * 2 * kHd;
    const int64_t block_stride = static_cast<int64_t>(hk) * head_stride;
    const auto q0 = ToBits(RandBf16(rng, static_cast<size_t>(T) * hq * kHd, 2.0f));
    const auto k0 = ToBits(RandBf16(rng, static_cast<size_t>(T) * hk * kHd, 2.0f));
    const auto v0 = ToBits(RandBf16(rng, static_cast<size_t>(T) * hk * kHd, 2.0f));
    const auto qn = ToBits(RandBf16(rng, kHd, 1.0f));
    const auto kn = ToBits(RandBf16(rng, kHd, 1.0f));
    std::vector<int32_t> pos(T);
    for (int t = 0; t < T; ++t) pos[t] = t * 5 + 3;
    DeviceBuffer<uint16_t> qa(q0.size()), qb(q0.size()), ka(k0.size()), kb(k0.size()), vd(v0.size());
    DeviceBuffer<uint16_t> qnd(kHd), knd(kHd);
    DeviceBuffer<int32_t> pd(T);
    DeviceBuffer<uint16_t> kva(static_cast<size_t>(max_blocks * block_stride)), kvb(kva.size());
    qa.CopyFromHost(q0);
    qb.CopyFromHost(q0);
    ka.CopyFromHost(k0);
    kb.CopyFromHost(k0);
    vd.CopyFromHost(v0);
    qnd.CopyFromHost(qn);
    knd.CopyFromHost(kn);
    pd.CopyFromHost(pos);
    R4DX_HIP_CHECK(hipMemset(kva.data(), 0x5a, kva.bytes()));
    R4DX_HIP_CHECK(hipMemset(kvb.data(), 0x5a, kvb.bytes()));

    r4dx_rmsnorm_plain_bf16(P(qa.data()), P(qnd.data()), P(qa.data()), static_cast<int64_t>(T) * hq, kHd, 1e-5f, 0, 0);
    r4dx_rmsnorm_plain_bf16(P(ka.data()), P(knd.data()), P(ka.data()), static_cast<int64_t>(T) * hk, kHd, 1e-5f, 0, 0);
    r4dx_rope_neox_bf16(P(qa.data()), P(ka.data()), P(pd.data()), T, hq, hk, kHd, kTheta, 0);
    r4dx_model_kv_write_paged_bf16_hnd(P(ka.data()), P(vd.data()), P(pd.data()), P(kva.data()), T, hk, kHd, block_size,
                                       block_stride, head_stride, 0);

    r4dx_lfm2_attn_precore_bf16(P(qb.data()), P(kb.data()), P(vd.data()), P(qnd.data()), P(knd.data()), P(pd.data()),
                                P(kvb.data()), T, hq, hk, kHd, 1e-5f, kTheta, block_size, block_stride, head_stride, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    Check(qa.CopyToHost() == qb.CopyToHost() && ka.CopyToHost() == kb.CopyToHost() &&
              kva.CopyToHost() == kvb.CopyToHost(),
          "r4dx_lfm2_attn_precore_bf16 == q_norm + k_norm + rope_neox + kv_write (bitwise on q, k and kv_cache)");
  }
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(64);
  TestRope(rng);
  TestNorm(rng);
  TestPreCoreOrder(rng);
  TestFusions(rng);
  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
