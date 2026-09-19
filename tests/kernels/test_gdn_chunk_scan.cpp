// tests/kernels/test_gdn_chunk_scan.cpp -- r4d_gdn_kkt_solve_k128_c64_bf16 +
// r4d_gdn_chunk_scan_k128_v128_c64_bf16 (via r4dx::core::r4d) against a CPU fp32 reference that
// implements modeling_qwen3_5.py's `torch_recurrent_gated_delta_rule` directly (the exact O(T)
// per-token recurrence the chunked WY kernel is an equivalent, chunk-parallel rewrite of):
//
//   state <- state * exp(g_t)                          (decay, per (t, head))
//   kv_mem[v] = sum_k state[k,v] * k_t[k]
//   delta[v] = (v_t[v] - kv_mem[v]) * beta_t
//   state[k,v] += k_t[k] * delta[v]
//   out_t[v] = scale * sum_k state[k,v] * q_t[k]
//
// One chunk (T <= 64 = r4d_gdn_dims().chunk), N=1, h0 = 0, so the kernel's chunk-local cumulative
// gate `g` it wants (r4d_gdn_kkt_solve_k128_c64_bf16's doc: "g already summed along the chunk")
// is simply the running cumsum of this reference's per-token instantaneous g over the whole
// (single-chunk) sequence -- no cross-chunk bookkeeping needed. q, k are L2-normalized per
// (token, head) before use, matching `use_qk_l2norm_in_kernel=True`
// (docs/architecture.md "GDN layer").
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"

using namespace r4dx::core;

namespace {

void L2NormRows(std::vector<uint16_t>* x, int rows, int dim) {
  for (int r = 0; r < rows; ++r) {
    double ss = 0.0;
    for (int d = 0; d < dim; ++d) {
      float v = Bf16ToFloat((*x)[r * dim + d]);
      ss += static_cast<double>(v) * v;
    }
    float inv = 1.0f / std::sqrt(static_cast<float>(ss) + 1e-6f);
    for (int d = 0; d < dim; ++d) {
      float v = Bf16ToFloat((*x)[r * dim + d]) * inv;
      (*x)[r * dim + d] = FloatToBf16(v);
    }
  }
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));

  r4d::GdnDims gdims = r4d::GetGdnDims();
  const int K = gdims.head_k, V = gdims.head_v, bt = gdims.chunk;  // 128, 128, 64
  const int N = 1, T = 48, H = 2, Hg = 2;  // one chunk, no GQA broadening in this test
  const float scale = 1.0f / std::sqrt(static_cast<float>(K));

  std::mt19937 rng(41);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  // A_log/dt_bias/a are deliberately much smaller in magnitude than the model's real
  // uniform(0.01,16) weight-init range for A: chunk_scan's per-chunk decay reference `c` (see
  // r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip's header comment and its "80 rather than 88"
  // clamping note) only reproduces the UNCLAMPED recurrence exactly while the chunk's total
  // cumulative-gate span stays well inside that clamp; past it the kernel's own comment says its
  // result is "attenuated, not exact" by design. This test is about verifying the WY chunked
  // rewrite matches the textbook recurrence, not about stressing that documented approximation,
  // so the per-token decay here is kept small enough that the 48-token chunk's total span stays
  // under ~10 nats, far inside the safe range.
  std::uniform_real_distribution<float> a_log_dist(0.01f, 0.3f);
  std::uniform_real_distribution<float> dt_bias_dist(-0.2f, 0.2f);
  std::uniform_real_distribution<float> a_dist(-0.3f, 0.3f);
  std::uniform_real_distribution<float> b_dist(-2.0f, 2.0f);

  std::vector<uint16_t> q_h(static_cast<size_t>(T) * Hg * K);
  std::vector<uint16_t> k_h(static_cast<size_t>(T) * Hg * K);
  std::vector<uint16_t> v_h(static_cast<size_t>(T) * H * V);
  for (auto& x : q_h) x = FloatToBf16(dist(rng));
  for (auto& x : k_h) x = FloatToBf16(dist(rng));
  for (auto& x : v_h) x = FloatToBf16(dist(rng));
  L2NormRows(&q_h, T * Hg, K);
  L2NormRows(&k_h, T * Hg, K);

  std::vector<float> A_log(H), dt_bias(H);
  for (int h = 0; h < H; ++h) {
    A_log[h] = std::log(a_log_dist(rng));
    dt_bias[h] = dt_bias_dist(rng);
  }
  std::vector<float> a_raw(static_cast<size_t>(T) * H), b_raw(static_cast<size_t>(T) * H);
  for (auto& x : a_raw) x = a_dist(rng);
  for (auto& x : b_raw) x = b_dist(rng);

  std::vector<float> g_raw(static_cast<size_t>(T) * H), beta(static_cast<size_t>(T) * H);
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < H; ++h) {
      float a_val = a_raw[t * H + h] + dt_bias[h];
      float softplus = a_val > 20.0f ? a_val : std::log1p(std::exp(a_val));
      g_raw[t * H + h] = -std::exp(A_log[h]) * softplus;
      beta[t * H + h] = 1.0f / (1.0f + std::exp(-b_raw[t * H + h]));
    }
  }
  // Chunk-local cumulative gate (single chunk == a running cumsum over the whole sequence).
  std::vector<float> g_cumsum(static_cast<size_t>(T) * H);
  for (int h = 0; h < H; ++h) {
    float running = 0.0f;
    for (int t = 0; t < T; ++t) {
      running += g_raw[t * H + h];
      g_cumsum[t * H + h] = running;
    }
  }

  std::vector<int32_t> cu = {0, T};

  // ---- device: kkt_solve then chunk_scan ------------------------------------------------------
  DeviceBuffer<uint16_t> q_d(q_h.size()), k_d(k_h.size()), v_d(v_h.size());
  DeviceBuffer<float> g_d(g_cumsum.size()), beta_d(beta.size());
  DeviceBuffer<uint16_t> A_d(static_cast<size_t>(T) * H * bt);
  DeviceBuffer<float> h0_d(static_cast<size_t>(N) * H * V * K), ht_d(static_cast<size_t>(N) * H * V * K);
  DeviceBuffer<uint16_t> o_d(static_cast<size_t>(T) * H * V);
  DeviceBuffer<int32_t> cu_d(cu.size());

  q_d.CopyFromHost(q_h);
  k_d.CopyFromHost(k_h);
  v_d.CopyFromHost(v_h);
  g_d.CopyFromHost(g_cumsum);
  beta_d.CopyFromHost(beta);
  h0_d.Zero();
  cu_d.CopyFromHost(cu);

  r4d::GdnKktSolve(k_d.data(), beta_d.data(), g_d.data(), A_d.data(), cu_d.data(), N, T, H, Hg, K,
                    bt, nullptr);
  R4DX_HIP_CHECK(hipDeviceSynchronize());

  r4d::GdnChunkScan(q_d.data(), k_d.data(), v_d.data(), A_d.data(), g_d.data(), beta_d.data(),
                     h0_d.data(), o_d.data(), ht_d.data(), cu_d.data(), N, H, Hg, K, V, bt, scale,
                     nullptr);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> o_h = o_d.CopyToHost();

  // ---- CPU reference: exact O(T) recurrence ---------------------------------------------------
  std::vector<float> ref_out(static_cast<size_t>(T) * H * V, 0.0f);
  std::vector<double> state(static_cast<size_t>(H) * K * V, 0.0);  // [H,K,V], zero initial state
  for (int t = 0; t < T; ++t) {
    for (int h = 0; h < H; ++h) {
      double decay = std::exp(static_cast<double>(g_raw[t * H + h]));
      double* s = &state[static_cast<size_t>(h) * K * V];
      for (size_t i = 0; i < static_cast<size_t>(K) * V; ++i) s[i] *= decay;

      std::vector<double> kv_mem(V, 0.0);
      for (int k = 0; k < K; ++k) {
        float kt = Bf16ToFloat(k_h[(static_cast<size_t>(t) * Hg + h) * K + k]);
        for (int v = 0; v < V; ++v) kv_mem[v] += s[static_cast<size_t>(k) * V + v] * kt;
      }
      std::vector<double> delta(V);
      for (int v = 0; v < V; ++v) {
        float vt = Bf16ToFloat(v_h[(static_cast<size_t>(t) * H + h) * V + v]);
        delta[v] = (static_cast<double>(vt) - kv_mem[v]) * beta[t * H + h];
      }
      for (int k = 0; k < K; ++k) {
        float kt = Bf16ToFloat(k_h[(static_cast<size_t>(t) * Hg + h) * K + k]);
        for (int v = 0; v < V; ++v) s[static_cast<size_t>(k) * V + v] += kt * delta[v];
      }
      for (int v = 0; v < V; ++v) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k) {
          float qt = Bf16ToFloat(q_h[(static_cast<size_t>(t) * Hg + h) * K + k]);
          acc += s[static_cast<size_t>(k) * V + v] * qt;
        }
        ref_out[(static_cast<size_t>(t) * H + h) * V + v] = static_cast<float>(scale * acc);
      }
    }
  }

  double max_rel = 0.0;
  double norm_num = 0.0, norm_den = 0.0;
  for (size_t i = 0; i < ref_out.size(); ++i) {
    double got = Bf16ToFloat(o_h[i]);
    double ref = ref_out[i];
    max_rel = std::max(max_rel, std::abs(got - ref) / std::max(1e-2, std::abs(ref)));
    norm_num += (got - ref) * (got - ref);
    norm_den += ref * ref;
  }
  double norm_rel = std::sqrt(norm_num) / std::max(1e-9, std::sqrt(norm_den));
  std::printf("gdn_chunk_scan T=%d H=%d K=%d V=%d: max elementwise rel err=%.4e, norm rel err=%.4e\n",
              T, H, K, V, max_rel, norm_rel);
  bool ok = norm_rel < 5e-2;
  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
