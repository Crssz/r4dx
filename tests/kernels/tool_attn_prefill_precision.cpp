// tests/kernels/tool_attn_prefill_precision.cpp -- prefill M1 diagnosis: is split-KV prefill
// attention LESS accurate than the unsplit kernel, or just differently rounded?
//
// One 64-row prefill chunk at depth D over a synthetic paged fp8 cache (identity block table,
// descales 1, so every dequantized K/V value is exact in fp32), run through
//   plain    r4d_attn_prefill_h256_gqa6_fp8kv (today's kernel, the "dense" reference of the KL gate)
//   exact    r4d_attn_prefill_exact_h256_gqa6_fp8kv (default geometry; must equal plain bit for bit)
//   splitS   r4d_attn_prefill_splitkv_h256_gqa6_fp8kv with S segments (fp32 partials + merge)
// and scored against an fp64 CPU reference over the SAME dequantized cache and the same bf16 query
// (QK, softmax and PV all in double). Per variant:
//   - rms / max abs error, and rms error relative to the output's rms;
//   - error in units of the bf16 ulp at the reference value: mean and p99;
//   - the fraction of outputs that are the correctly rounded (RNE) bf16 of the fp64 value;
//   - head to head vs plain: the fraction of outputs where the variant is closer to fp64, farther,
//     or tied (identical error).
// Two data sets: "uniform" (the ctest's i.i.d. [-1,1] data: flat attention) and "peaky" (keys share
// a mean direction and every query has a few strongly aligned keys spread over the whole depth, so
// the softmax is dominated by a handful of positions, as in real long-context attention).
//
// Built, never add_test()'d (a measurement). Example:
//   $env:HIP_VISIBLE_DEVICES='1'; tool_attn_prefill_precision.exe --depths 8192,32768,122880 `
//       --splits 8,16,32 --out precision.json
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

namespace {

std::vector<int> ParseList(const char* s) {
  std::vector<int> v;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) v.push_back(std::atoi(item.c_str()));
  }
  return v;
}

// bf16 round-to-nearest-even of a double (via float, then RNE to 16 bits; the double->float step
// can double-round only when the double sits within 2^-24 relative of a bf16 midpoint, which is
// below anything this tool resolves).
uint16_t Bf16Rne(double x) {
  const float f = static_cast<float>(x);
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t lsb = (u >> 16) & 1u;
  u += 0x7FFFu + lsb;
  return static_cast<uint16_t>(u >> 16);
}

double Bf16Ulp(double x) {
  const double a = std::max(std::abs(x), 1e-30);
  int e = 0;
  std::frexp(a, &e);           // a = m * 2^e, m in [0.5, 1)
  return std::ldexp(1.0, e - 8);  // 8 significant bits
}

struct Score {
  std::string name;
  double sum_sq = 0, max_abs = 0, ref_sq = 0;
  std::vector<double> ulp;
  size_t correct = 0, n = 0, closer = 0, farther = 0, tied = 0, bits_vs_plain = 0;
};

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> depths = {8192, 32768, 122880};
  std::vector<int> splits = {8, 16, 32};
  std::vector<std::string> dists = {"uniform", "peaky"};
  int kv_heads = 4, q_len = 64;
  const char* out_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const char* v = i + 1 < argc ? argv[i + 1] : nullptr;
    if (a == "--depths" && v) { depths = ParseList(v); ++i; }
    else if (a == "--splits" && v) { splits = ParseList(v); ++i; }
    else if (a == "--kv-heads" && v) { kv_heads = std::atoi(v); ++i; }
    else if (a == "--dist" && v) { dists = {v}; ++i; }
    else if (a == "--out" && v) { out_path = v; ++i; }
    else {
      std::fprintf(stderr, "usage: tool_attn_prefill_precision [--depths a,b] [--splits a,b] "
                           "[--kv-heads N] [--dist uniform|peaky] [--out f.json]\n");
      return 2;
    }
  }
  R4DX_HIP_CHECK(hipSetDevice(0));
  int head_dim = 0, gqa = 0, block_size = 0, max_decode_rows = 0;
  r4d_attn_dims(&head_dim, &gqa, &block_size, &max_decode_rows);
  const int q_heads = kv_heads * gqa;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t kv_head_stride = static_cast<int64_t>(block_size) * 2 * head_dim;
  const int64_t kv_block_stride = static_cast<int64_t>(kv_heads) * kv_head_stride;
  const unsigned nthreads = std::max(1u, std::min(32u, std::thread::hardware_concurrency()));

  std::string json = "{\"results\": [";
  bool first = true, exact_ok = true;
  for (const std::string& dist_name : dists) {
    const bool peaky = dist_name == "peaky";
    for (int depth : depths) {
      const int ctx = depth + q_len;
      const int max_blocks = (ctx + block_size - 1) / block_size;
      std::mt19937 rng(20260928u + static_cast<unsigned>(depth) + (peaky ? 7u : 0u));
      std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
      std::normal_distribution<float> gauss(0.0f, 1.0f);

      std::vector<uint16_t> k_h(static_cast<size_t>(ctx) * kv_heads * head_dim), v_h(k_h.size());
      std::vector<uint16_t> q_h(static_cast<size_t>(q_len) * q_heads * head_dim);
      if (!peaky) {
        for (auto& x : k_h) x = FloatToBf16(uni(rng));
        for (auto& x : v_h) x = FloatToBf16(uni(rng));
        for (auto& x : q_h) x = FloatToBf16(uni(rng));
      } else {
        // Keys: a per-head shared direction (|mu| ~ 2) plus unit noise; values: unit noise.
        std::vector<float> mu(static_cast<size_t>(kv_heads) * head_dim);
        for (auto& x : mu) x = 0.12f * gauss(rng);
        for (int t = 0; t < ctx; ++t)
          for (int h = 0; h < kv_heads; ++h)
            for (int d = 0; d < head_dim; ++d) {
              const size_t i = (static_cast<size_t>(t) * kv_heads + h) * head_dim + d;
              k_h[i] = FloatToBf16(mu[static_cast<size_t>(h) * head_dim + d] + 0.5f * gauss(rng));
              v_h[i] = FloatToBf16(gauss(rng));
            }
        // Queries: noise with a component along mu (so every key gets a common offset) ...
        for (int r = 0; r < q_len; ++r)
          for (int qh = 0; qh < q_heads; ++qh)
            for (int d = 0; d < head_dim; ++d) {
              const size_t i = (static_cast<size_t>(r) * q_heads + qh) * head_dim + d;
              q_h[i] = FloatToBf16(1.5f * gauss(rng) + 4.0f * mu[static_cast<size_t>(qh / gqa) * head_dim + d]);
            }
        // ... and 6 strongly aligned keys per (row, kv head), anywhere in the causal range: the
        // key is the first query of that group, scaled, so its score is ~8-20 nats above the background.
        std::uniform_int_distribution<int> pos(0, ctx - 1);
        for (int r = 0; r < q_len; ++r)
          for (int h = 0; h < kv_heads; ++h)
            for (int j = 0; j < 6; ++j) {
              const int t = std::min(pos(rng), depth + r);
              const float s = 0.2f + 0.06f * static_cast<float>(j);
              for (int d = 0; d < head_dim; ++d) {
                const size_t src = (static_cast<size_t>(r) * q_heads + h * gqa) * head_dim + d;
                const size_t dst = (static_cast<size_t>(t) * kv_heads + h) * head_dim + d;
                k_h[dst] = FloatToBf16(s * Bf16ToFloat(q_h[src]));
              }
            }
      }

      std::vector<int32_t> bt(static_cast<size_t>(max_blocks)), slots(static_cast<size_t>(ctx));
      for (int b = 0; b < max_blocks; ++b) bt[b] = b;
      for (int t = 0; t < ctx; ++t) slots[t] = t;
      const std::vector<float> ones(static_cast<size_t>(kv_heads), 1.0f);
      DeviceBuffer<uint16_t> k_d(k_h.size()), v_d(v_h.size()), q_d(q_h.size()), out_d(q_h.size());
      DeviceBuffer<int32_t> slot_d(slots.size()), bt_d(bt.size()), seq_d(1);
      DeviceBuffer<float> kd_d(kv_heads), vd_d(kv_heads);
      DeviceBuffer<uint8_t> cache_d(static_cast<size_t>(max_blocks) * kv_block_stride);
      k_d.CopyFromHost(k_h);
      v_d.CopyFromHost(v_h);
      q_d.CopyFromHost(q_h);
      slot_d.CopyFromHost(slots);
      bt_d.CopyFromHost(bt);
      seq_d.CopyFromHost(std::vector<int32_t>{ctx});
      kd_d.CopyFromHost(ones);
      vd_d.CopyFromHost(ones);
      cache_d.Zero();
      r4dx_kv_write_paged_fp8_hnd(reinterpret_cast<int64_t>(k_d.data()), reinterpret_cast<int64_t>(v_d.data()),
                                   reinterpret_cast<int64_t>(slot_d.data()), reinterpret_cast<int64_t>(kd_d.data()),
                                   reinterpret_cast<int64_t>(vd_d.data()), reinterpret_cast<int64_t>(cache_d.data()),
                                   ctx, kv_heads, head_dim, block_size, kv_block_stride, kv_head_stride, 0);
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      const std::vector<uint8_t> cache_h = cache_d.CopyToHost();

      // ---- fp64 reference ------------------------------------------------------------------------
      std::vector<float> kf(static_cast<size_t>(ctx) * kv_heads * head_dim), vf(kf.size());
      for (int t = 0; t < ctx; ++t)
        for (int h = 0; h < kv_heads; ++h) {
          const int64_t base = static_cast<int64_t>(t / block_size) * kv_block_stride + h * kv_head_stride +
                               (t % block_size) * 2 * head_dim;
          const size_t dst = (static_cast<size_t>(t) * kv_heads + h) * head_dim;
          for (int d = 0; d < head_dim; ++d) {
            kf[dst + d] = Fp8E4M3ToFloat(cache_h[base + d]);
            vf[dst + d] = Fp8E4M3ToFloat(cache_h[base + head_dim + d]);
          }
        }
      std::vector<double> ref(q_h.size());
      double ent_sum = 0.0, top_sum = 0.0;  // mean softmax entropy (nats) and top-weight, for the record
      std::vector<double> ent_part(nthreads, 0.0), top_part(nthreads, 0.0);
      auto ref_rows = [&](unsigned tid) {
        std::vector<double> logit(static_cast<size_t>(ctx)), o(static_cast<size_t>(head_dim)), q(head_dim);
        for (int job = static_cast<int>(tid); job < q_len * q_heads; job += static_cast<int>(nthreads)) {
          const int r = job / q_heads, qh = job % q_heads, h = qh / gqa;
          const size_t qoff = (static_cast<size_t>(r) * q_heads + qh) * head_dim;
          for (int d = 0; d < head_dim; ++d) q[d] = Bf16ToFloat(q_h[qoff + d]);
          const int klimit = depth + r;
          double mx = -1e300;
          for (int t = 0; t <= klimit; ++t) {
            const float* kp = &kf[(static_cast<size_t>(t) * kv_heads + h) * head_dim];
            double dot = 0.0;
            for (int d = 0; d < head_dim; ++d) dot += q[d] * kp[d];
            logit[t] = dot * static_cast<double>(scale);
            mx = std::max(mx, logit[t]);
          }
          double sum = 0.0, top = 0.0;
          std::fill(o.begin(), o.end(), 0.0);
          for (int t = 0; t <= klimit; ++t) {
            const double p = std::exp(logit[t] - mx);
            sum += p;
            top = std::max(top, p);
            const float* vp = &vf[(static_cast<size_t>(t) * kv_heads + h) * head_dim];
            for (int d = 0; d < head_dim; ++d) o[d] += p * vp[d];
          }
          double ent = 0.0;
          for (int t = 0; t <= klimit; ++t) {
            const double p = std::exp(logit[t] - mx) / sum;
            if (p > 0) ent -= p * std::log(p);
          }
          ent_part[tid] += ent;
          top_part[tid] += top / sum;
          for (int d = 0; d < head_dim; ++d) ref[qoff + d] = o[d] / sum;
        }
      };
      {
        std::vector<std::thread> th;
        for (unsigned t = 0; t < nthreads; ++t) th.emplace_back(ref_rows, t);
        for (auto& t : th) t.join();
      }
      for (unsigned t = 0; t < nthreads; ++t) { ent_sum += ent_part[t]; top_sum += top_part[t]; }

      // ---- kernels -------------------------------------------------------------------------------
      R4DArgs a{};
      a.q = q_d.data();
      a.kv = cache_d.data();
      a.block_table = bt_d.data();
      a.seqused_k = seq_d.data();
      a.k_descale = kd_d.data();
      a.v_descale = vd_d.data();
      a.num_seqs = 1;
      a.q_len = q_len;
      a.q_heads = q_heads;
      a.kv_heads = kv_heads;
      a.head_dim = head_dim;
      a.block_size = block_size;
      a.max_blocks = max_blocks;
      a.kv_block_stride = kv_block_stride;
      a.kv_head_stride = kv_head_stride;
      a.scale = scale;
      a.max_ctx = ctx;
      auto run = [&](int kind, int sp) -> std::vector<uint16_t> {  // 0 plain, 1 exact, 2 split
        R4DArgs b = a;
        b.out = out_d.data();
        b.splits = kind == 2 ? sp : 0;
        const int64_t bytes = kind == 2 ? r4d_attn_prefill_splitkv_h256_gqa6_scratch_bytes(&b) : 0;
        DeviceBuffer<uint8_t> scratch(bytes > 0 ? static_cast<size_t>(bytes) : 1);
        b.scratch = bytes > 0 ? scratch.data() : nullptr;
        R4DX_HIP_CHECK(hipMemset(out_d.data(), 0xFF, out_d.size() * 2));
        const int rc = kind == 0 ? r4d_attn_prefill_h256_gqa6_fp8kv(&b, nullptr)
                     : kind == 1 ? r4d_attn_prefill_exact_h256_gqa6_fp8kv(&b, nullptr)
                                 : r4d_attn_prefill_splitkv_h256_gqa6_fp8kv(&b, nullptr);
        if (rc != 0) {
          std::fprintf(stderr, "launch rc %d\n", rc);
          std::exit(1);
        }
        R4DX_HIP_CHECK(hipDeviceSynchronize());
        return out_d.CopyToHost();
      };

      const std::vector<uint16_t> plain = run(0, 0);
      std::vector<std::pair<std::string, std::vector<uint16_t>>> vars;
      vars.push_back({"plain", plain});
      vars.push_back({"exact", run(1, 0)});
      for (int sp : splits) vars.push_back({"split" + std::to_string(sp), run(2, sp)});

      std::printf("[%s depth %6d] softmax: mean entropy %.2f nats, mean top weight %.3f\n",
                  dist_name.c_str(), depth, ent_sum / (q_len * q_heads), top_sum / (q_len * q_heads));
      for (auto& [name, got] : vars) {
        Score s;
        s.name = name;
        s.ulp.reserve(got.size());
        for (size_t i = 0; i < got.size(); ++i) {
          const double g = Bf16ToFloat(got[i]), r = ref[i];
          const double e = std::abs(g - r), ep = std::abs(static_cast<double>(Bf16ToFloat(plain[i])) - r);
          s.sum_sq += e * e;
          s.ref_sq += r * r;
          s.max_abs = std::max(s.max_abs, e);
          s.ulp.push_back(e / Bf16Ulp(r));
          s.correct += got[i] == Bf16Rne(r);
          s.closer += e < ep;
          s.farther += e > ep;
          s.tied += e == ep;
          s.bits_vs_plain += got[i] != plain[i];
          ++s.n;
        }
        std::vector<double> u = s.ulp;
        const double mean_ulp = std::accumulate(u.begin(), u.end(), 0.0) / u.size();
        std::nth_element(u.begin(), u.begin() + static_cast<long>(u.size() * 0.99), u.end());
        const double p99 = u[static_cast<size_t>(u.size() * 0.99)];
        const double rms = std::sqrt(s.sum_sq / s.n), rel = rms / std::sqrt(s.ref_sq / s.n);
        const double n = static_cast<double>(s.n);
        std::printf("  %-8s rms %.3e (rel %.3e) max %.3e | ulp mean %.4f p99 %.3f | correctly rounded "
                    "%.2f%% | vs plain: closer %.2f%% farther %.2f%% tied %.2f%% (%zu bf16 differ)\n",
                    name.c_str(), rms, rel, s.max_abs, mean_ulp, p99, 100.0 * s.correct / n,
                    100.0 * s.closer / n, 100.0 * s.farther / n, 100.0 * s.tied / n, s.bits_vs_plain);
        if (name == "exact" && s.bits_vs_plain != 0) exact_ok = false;
        char buf[768];
        std::snprintf(buf, sizeof(buf),
                      "%s\n {\"dist\": \"%s\", \"depth\": %d, \"variant\": \"%s\", \"rms\": %.6e, \"rel_rms\": %.6e, "
                      "\"max_abs\": %.6e, \"ulp_mean\": %.6f, \"ulp_p99\": %.6f, \"correct_pct\": %.4f, "
                      "\"closer_pct\": %.4f, \"farther_pct\": %.4f, \"tied_pct\": %.4f, \"bits_vs_plain\": %zu, "
                      "\"entropy\": %.4f, \"top_weight\": %.4f}",
                      first ? "" : ",", dist_name.c_str(), depth, name.c_str(), rms, rel, s.max_abs, mean_ulp, p99,
                      100.0 * s.correct / n, 100.0 * s.closer / n, 100.0 * s.farther / n, 100.0 * s.tied / n,
                      s.bits_vs_plain, ent_sum / (q_len * q_heads), top_sum / (q_len * q_heads));
        json += buf;
        first = false;
      }
    }
  }
  json += "\n]}\n";
  if (out_path != nullptr) {
    FILE* f = std::fopen(out_path, "w");
    if (f != nullptr) {
      std::fputs(json.c_str(), f);
      std::fclose(f);
      std::printf("wrote %s\n", out_path);
    }
  }
  std::printf(exact_ok ? "exact == plain: yes\n" : "exact == plain: NO\n");
  return exact_ok ? 0 : 1;
}
