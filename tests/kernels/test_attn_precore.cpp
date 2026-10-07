// tests/kernels/test_attn_precore.cpp -- decode-t1 item 4: r4dx_attn_precore_bf16 (split_qg + q/k norm +
// partial rope + fp8 paged cache write in one launch) against the five launches it replaces, in the order
// AttentionLayer::Forward runs them -- r4dx_model_attn_split_qg_bf16, r4dx_rmsnorm_bf16 on q and on k,
// r4dx_rope_partial_mrope_bf16, r4dx_kv_write_paged_fp8_hnd -- BIT FOR BIT: the q rows, the gate rows and
// every byte of the fp8 cache (the rope'd k is not written back by the fused kernel; only the cache write
// reads it). Shapes: the model's (24 q / 4 kv heads, head_dim 256, rotary 64, theta 1e7) and the TP=2
// shard's (12 / 2), T = 1 (decode) up to 64 (a chunk), small and very large start positions (the rope's
// slow-path angles), a skipped slot (-1), stray outlier values. Plus r4dx_attn_precore_supported's refusals.
// Needs a HIP device.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/model/attention/attn_kernels.h"

using namespace r4dx::core;

namespace {

int g_fail = 0;
int g_cases = 0;

std::vector<uint16_t> RandomBf16(std::mt19937& rng, size_t n, float stddev, float outlier_rate = 0.0f) {
  std::normal_distribution<float> nd(0.0f, stddev);
  std::uniform_real_distribution<float> u(0.0f, 1.0f);
  std::vector<uint16_t> v(n);
  for (auto& x : v) {
    float f = nd(rng);
    if (outlier_rate > 0.0f && u(rng) < outlier_rate) f *= 30.0f;
    x = FloatToBf16(f);
  }
  return v;
}

void CheckCase(std::mt19937& rng, int T, int H, int Hkv, int D, int rotary, int start_pos, bool skip_one) {
  const float theta = 1.0e7f, eps = 1.0e-6f;
  const int block_size = 16;
  const int max_pos = start_pos + T + 1;
  const int num_blocks = (max_pos + block_size - 1) / block_size + 1;
  const int64_t kv_head_stride = static_cast<int64_t>(block_size) * 2 * D;
  const int64_t kv_block_stride = static_cast<int64_t>(Hkv) * kv_head_stride;
  const size_t cache_bytes = static_cast<size_t>(num_blocks) * kv_block_stride;

  const std::vector<uint16_t> qg_h = RandomBf16(rng, static_cast<size_t>(T) * H * 2 * D, 2.0f, 0.002f);
  const std::vector<uint16_t> k_h = RandomBf16(rng, static_cast<size_t>(T) * Hkv * D, 2.0f, 0.002f);
  const std::vector<uint16_t> v_h = RandomBf16(rng, static_cast<size_t>(T) * Hkv * D, 1.0f);
  const std::vector<uint16_t> qn_h = RandomBf16(rng, static_cast<size_t>(D), 0.3f);
  const std::vector<uint16_t> kn_h = RandomBf16(rng, static_cast<size_t>(D), 0.3f);
  std::uniform_real_distribution<float> dd(0.5f, 2.0f);
  std::vector<float> kd_h(static_cast<size_t>(Hkv)), vd_h(static_cast<size_t>(Hkv));
  for (auto& x : kd_h) x = dd(rng);
  for (auto& x : vd_h) x = dd(rng);
  std::vector<int32_t> pos_h(static_cast<size_t>(T));
  for (int t = 0; t < T; ++t) pos_h[static_cast<size_t>(t)] = start_pos + t;
  if (skip_one && T > 1) pos_h[static_cast<size_t>(T / 2)] = -1;

  DeviceBuffer<uint16_t> qg(qg_h.size()), k(k_h.size()), kold(k_h.size()), v(v_h.size()), qn(qn_h.size()),
      kn(kn_h.size());
  DeviceBuffer<float> kd(kd_h.size()), vd(vd_h.size());
  DeviceBuffer<int32_t> pos(pos_h.size());
  qg.CopyFromHost(qg_h);
  k.CopyFromHost(k_h);
  kold.CopyFromHost(k_h);
  v.CopyFromHost(v_h);
  qn.CopyFromHost(qn_h);
  kn.CopyFromHost(kn_h);
  kd.CopyFromHost(kd_h);
  vd.CopyFromHost(vd_h);
  pos.CopyFromHost(pos_h);
  const size_t qn_elems = static_cast<size_t>(T) * H * D;
  DeviceBuffer<uint16_t> q_old(qn_elems), gate_old(qn_elems), q_new(qn_elems), gate_new(qn_elems);
  DeviceBuffer<uint8_t> cache_old(cache_bytes), cache_new(cache_bytes);
  cache_old.Zero();
  cache_new.Zero();
  q_new.Zero();
  gate_new.Zero();
  const auto P = [](const auto& b) { return reinterpret_cast<int64_t>(b.data()); };

  // the five launches, as AttentionLayer::Forward runs them
  r4dx_model_attn_split_qg_bf16(P(qg), P(q_old), P(gate_old), T, H, D, 0);
  r4dx_rmsnorm_bf16(P(q_old), P(qn), P(q_old), static_cast<int64_t>(T) * H, D, eps, 0);
  r4dx_rmsnorm_bf16(P(kold), P(kn), P(kold), static_cast<int64_t>(T) * Hkv, D, eps, 0);
  r4dx_rope_partial_mrope_bf16(P(q_old), P(kold), P(pos), T, H, Hkv, D, rotary, theta, 0);
  r4dx_kv_write_paged_fp8_hnd(P(kold), P(v), P(pos), P(kd), P(vd), P(cache_old), T, Hkv, D, block_size,
                               kv_block_stride, kv_head_stride, 0);

  // the one launch (run twice: determinism, and the cache write is idempotent)
  for (int rep = 0; rep < 2; ++rep) {
    r4dx_attn_precore_bf16(P(qg), P(k), P(v), P(qn), P(kn), P(pos), P(q_new), P(gate_new), P(kd), P(vd),
                            P(cache_new), T, H, Hkv, D, rotary, theta, eps, block_size, kv_block_stride,
                            kv_head_stride, 0);
  }
  R4DX_HIP_CHECK(hipDeviceSynchronize());

  const std::vector<uint16_t> qo = q_old.CopyToHost(), qnw = q_new.CopyToHost();
  const std::vector<uint16_t> go = gate_old.CopyToHost(), gnw = gate_new.CopyToHost();
  const std::vector<uint8_t> co = cache_old.CopyToHost(), cnw = cache_new.CopyToHost();
  size_t bad_q = 0, bad_g = 0, bad_c = 0;
  for (size_t i = 0; i < qo.size(); ++i) bad_q += qo[i] != qnw[i];
  for (size_t i = 0; i < go.size(); ++i) bad_g += go[i] != gnw[i];
  for (size_t i = 0; i < co.size(); ++i) bad_c += co[i] != cnw[i];
  ++g_cases;
  if (bad_q != 0 || bad_g != 0 || bad_c != 0) {
    ++g_fail;
    std::fprintf(stderr,
                 "FAIL T=%d H=%d Hkv=%d D=%d rotary=%d start=%d skip=%d: %zu q, %zu gate, %zu cache bytes differ\n", T,
                 H, Hkv, D, rotary, start_pos, skip_one ? 1 : 0, bad_q, bad_g, bad_c);
  }
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(4102026);
  for (const auto& heads : {std::pair<int, int>{24, 4}, std::pair<int, int>{12, 2}}) {
    for (int T : {1, 2, 4, 8, 17, 64}) {
      for (int start : {0, 7, 4093, 40000}) CheckCase(rng, T, heads.first, heads.second, 256, 64, start, false);
    }
    CheckCase(rng, 8, heads.first, heads.second, 256, 64, 100, true);
  }
  CheckCase(rng, 4, 8, 2, 128, 32, 50, false);  // another head_dim / rotary
  CheckCase(rng, 3, 8, 2, 64, 64, 50, false);   // rotary == head_dim
  CheckCase(rng, 256, 24, 4, 256, 64, 1000, false);  // a 256-row super-chunk's rows

  // r4dx_attn_precore_supported: the shapes and alignments the kernel does NOT reproduce
  DeviceBuffer<uint16_t> b(4096);
  const int64_t p = reinterpret_cast<int64_t>(b.data());
  const auto ok = [&](int hd, int rd, int64_t qg, int64_t k, int64_t qn, int64_t kn, int64_t q, int64_t g) {
    return r4dx_attn_precore_supported(hd, rd, qg, k, qn, kn, q, g) != 0;
  };
  ++g_cases;
  if (!ok(256, 64, p, p, p, p, p, p)) { ++g_fail; std::fprintf(stderr, "FAIL supported: the model's own shape refused\n"); }
  if (ok(257, 64, p, p, p, p, p, p) || ok(512, 64, p, p, p, p, p, p) || ok(256, 63, p, p, p, p, p, p) ||
      ok(256, 0, p, p, p, p, p, p) || ok(256, 64, p + 2, p, p, p, p, p) || ok(256, 64, p, p, p, p, 0, p) ||
      ok(256, 64, p, p, p + 8, p, p, p) || ok(256, 64, p, p + 4, p, p, p, p)) {
    ++g_fail;
    std::fprintf(stderr, "FAIL supported: an unsupported shape / alignment was accepted\n");
  }

  if (g_fail != 0) {
    std::fprintf(stderr, "test_attn_precore: %d of %d cases FAILED\n", g_fail, g_cases);
    return 1;
  }
  std::printf("test_attn_precore: OK (%d cases, one launch == split_qg + qk_norm + rope + kv_write bit for bit)\n", g_cases);
  return 0;
}
