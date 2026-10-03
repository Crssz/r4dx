// tests/kernels/test_attn_ref_gemma.cpp -- r4dx_gemma_attn_ref_{fp8,bf16}kv (docs/gemma4-plan.md 3.3,
// stage 0) against a CPU double-precision reference, for both Gemma 4 geometries:
//   sliding : 16 q / 8 kv heads, head_dim 256, window 1024, scale 1.0, the KV in a 1536-token RING
//             (96 blocks) addressed through the shared block table, contexts well past the ring size
//             (the ring wraps twice), q_len 1 (decode), 8 (verify), 37 (a prefill chunk tail);
//   full    : 16 q / 1 kv head, head_dim 512, no window, contiguous table, q_len 1 / 5 / 30;
// plus a bidirectional-extension case (klimit_ext: an image block inside a prefill chunk, sliding
// ring), a non-unit scale with non-unit descales, a bf16-KV case per geometry, and the shape refusals.
// The oracle reads the cache BYTES the device holds (so it isolates the attention kernel from the
// fp8 write kernel) and a separate check proves the ring still holds the TRUE history for every key
// of every window (so an aliasing / ring-write bug cannot hide in the oracle).
// GPU test (ctest sets HIP_VISIBLE_DEVICES=1); needs no data files.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/model/attention/attn_ref.h"
#include "r4dx/model/attention/sliding_kv_cache.hpp"
#include "r4dx/model/attention/sliding_ring.hpp"

using namespace r4dx::core;
using r4dx::model::attention::SlidingBlockTable;
using r4dx::model::attention::SlidingKvCache;
using r4dx::model::attention::SlidingRingGeometry;

namespace {

int g_fail = 0;
void Check(bool cond, const std::string& what) {
  std::printf("%-100s %s\n", what.c_str(), cond ? "ok" : "FAIL");
  if (!cond) ++g_fail;
}
int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

struct Case {
  std::string name;
  int q_heads, kv_heads, head_dim, window, ctx, q_len;
  bool ring;      // sliding ring + shared table (fp8 only), else contiguous
  bool bf16kv;
  bool ext;       // bidirectional image block inside the chunk
  float scale;
  bool unit_descale;
};

const int kBs = 16;

void RunCase(const Case& c, std::mt19937& rng) {
  const int hd = c.head_dim, kvh = c.kv_heads, qh = c.q_heads;
  const float sigma = std::sqrt(1.5f / std::sqrt(static_cast<float>(hd)));
  std::normal_distribution<float> nd(0.0f, sigma);

  // ---- true history (bf16-rounded), per position / kv head / dim ----
  const size_t row = static_cast<size_t>(kvh) * hd;
  std::vector<uint16_t> k_hist(static_cast<size_t>(c.ctx) * row), v_hist(k_hist.size());
  for (auto& v : k_hist) v = FloatToBf16(nd(rng));
  for (auto& v : v_hist) v = FloatToBf16(nd(rng));
  std::vector<float> kd(kvh, 1.0f), vd(kvh, 1.0f);
  if (!c.bf16kv && !c.unit_descale) {
    for (int h = 0; h < kvh; ++h) { kd[h] = 0.01f * (1.0f + 0.15f * h); vd[h] = 0.012f * (1.0f + 0.1f * h); }
  } else if (!c.bf16kv) {
    for (int h = 0; h < kvh; ++h) { kd[h] = 0.01f; vd[h] = 0.01f; }   // fp8 range wants a descale
  } else if (!c.unit_descale) {
    for (int h = 0; h < kvh; ++h) { kd[h] = 1.0f + 0.1f * h; vd[h] = 0.9f + 0.05f * h; }
  }

  // ---- the cache ----
  const int ring_tokens = c.ring ? 1536 : 0;
  std::unique_ptr<SlidingRingGeometry> geo;
  std::vector<int32_t> table_h;
  int blocks;
  if (c.ring) {
    geo = std::make_unique<SlidingRingGeometry>(1024, 288, kBs, c.ctx, ring_tokens);
    table_h = geo->BuildBlockTable();
    blocks = geo->RingBlocks();
  } else {
    blocks = (c.ctx + kBs - 1) / kBs;
    table_h.resize(blocks);
    for (int i = 0; i < blocks; ++i) table_h[i] = i;
  }
  const int64_t head_stride = static_cast<int64_t>(kBs) * 2 * hd;
  const int64_t block_stride = static_cast<int64_t>(kvh) * head_stride;
  const size_t cache_elems = static_cast<size_t>(blocks) * block_stride;

  DeviceBuffer<uint8_t> cache8;
  DeviceBuffer<uint16_t> cache16;
  DeviceBuffer<float> kd_d(kvh), vd_d(kvh);
  kd_d.CopyFromHost(kd);
  vd_d.CopyFromHost(vd);
  auto slot_of = [&](int p) { return c.ring ? p % ring_tokens : p; };

  std::vector<uint8_t> host8;
  std::vector<uint16_t> host16;
  if (!c.bf16kv) {
    cache8.Resize(cache_elems);
    cache8.Zero();
    // Write the history in chunks of <= 256 (a chunk is written before the next one's attention
    // would run), through the production fp8 write kernel with the ring's slot mapping.
    for (int pos = 0; pos < c.ctx; pos += 256) {
      const int T = std::min(256, c.ctx - pos);
      DeviceBuffer<uint16_t> kn(static_cast<size_t>(T) * row), vn(kn.size());
      DeviceBuffer<int32_t> sm(T);
      kn.CopyFromHost(k_hist.data() + static_cast<size_t>(pos) * row, kn.size());
      vn.CopyFromHost(v_hist.data() + static_cast<size_t>(pos) * row, vn.size());
      std::vector<int32_t> slots(T);
      for (int t = 0; t < T; ++t) slots[t] = slot_of(pos + t);
      sm.CopyFromHost(slots);
      r4dx_kv_write_paged_fp8_hnd(P(kn.data()), P(vn.data()), P(sm.data()), P(kd_d.data()), P(vd_d.data()),
                                   P(cache8.data()), T, kvh, hd, kBs, block_stride, head_stride, 0);
    }
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    host8 = cache8.CopyToHost();
  } else {
    // bf16 cache assembled on the host (K then V per slot), contiguous table.
    host16.assign(cache_elems, 0);
    for (int p = 0; p < c.ctx; ++p) {
      for (int h = 0; h < kvh; ++h) {
        const size_t base = static_cast<size_t>(p / kBs) * block_stride + h * head_stride +
                            static_cast<size_t>(p % kBs) * 2 * hd;
        for (int d = 0; d < hd; ++d) {
          host16[base + d] = k_hist[static_cast<size_t>(p) * row + h * hd + d];
          host16[base + hd + d] = v_hist[static_cast<size_t>(p) * row + h * hd + d];
        }
      }
    }
    cache16.Resize(cache_elems);
    cache16.CopyFromHost(host16);
  }

  // value of cache element (block of key p, head h, column col) as the kernel sees it
  auto cache_at = [&](int p, int h, int col) -> double {
    const size_t idx = static_cast<size_t>(table_h[p / kBs]) * block_stride + h * head_stride +
                       static_cast<size_t>(p % kBs) * 2 * hd + col;
    return c.bf16kv ? Bf16ToFloat(host16[idx]) : Fp8E4M3ToFloat(host8[idx]);
  };

  // ---- ring really holds the true history for every window key of every query ----
  {
    const int qpos_lo = c.ctx - c.q_len;
    const int klo = c.window > 0 ? std::max(0, qpos_lo - c.window + 1) : 0;
    int bad = 0;
    for (int p = klo; p < c.ctx; p += 7) {
      for (int h = 0; h < kvh; ++h) {
        for (int d = 0; d < hd; d += 13) {
          const double want_k = Bf16ToFloat(k_hist[static_cast<size_t>(p) * row + h * hd + d]);
          const double got_k = cache_at(p, h, d) * (c.bf16kv ? 1.0 : kd[h]);
          if (std::abs(got_k - want_k) > 0.07 * std::abs(want_k) + 2e-3) ++bad;
          const double want_v = Bf16ToFloat(v_hist[static_cast<size_t>(p) * row + h * hd + d]);
          const double got_v = cache_at(p, h, hd + d) * (c.bf16kv ? 1.0 : vd[h]);
          if (std::abs(got_v - want_v) > 0.07 * std::abs(want_v) + 2e-3) ++bad;
        }
      }
    }
    Check(bad == 0, c.name + ": the cache holds the true history for the whole window (" +
                        std::to_string(bad) + " bad)");
  }

  // ---- queries ----
  std::vector<uint16_t> q_h(static_cast<size_t>(c.q_len) * qh * hd);
  for (auto& v : q_h) v = FloatToBf16(nd(rng));
  std::vector<int32_t> ext_h(c.q_len, -1);
  if (c.ext) {
    // Rows 5..24 of the chunk are an image block: they all see up to its last row.
    const int a = std::min(5, c.q_len - 1), b = std::min(24, c.q_len - 1);
    for (int t = a; t <= b; ++t) ext_h[t] = c.ctx - c.q_len + b;
  }

  DeviceBuffer<uint16_t> q_d(q_h.size()), out_d(q_h.size());
  DeviceBuffer<int32_t> bt_d(table_h.size()), ext_d(ext_h.size());
  q_d.CopyFromHost(q_h);
  bt_d.CopyFromHost(table_h);
  ext_d.CopyFromHost(ext_h);
  out_d.Zero();

  R4dxGemmaAttnRefArgs a{};
  a.q = P(q_d.data());
  a.kv = c.bf16kv ? P(cache16.data()) : P(cache8.data());
  a.block_table = P(bt_d.data());
  a.out = P(out_d.data());
  a.k_descale = c.unit_descale && c.bf16kv ? 0 : P(kd_d.data());
  a.v_descale = c.unit_descale && c.bf16kv ? 0 : P(vd_d.data());
  a.klimit_ext = c.ext ? P(ext_d.data()) : 0;
  a.q_len = c.q_len; a.q_heads = qh; a.kv_heads = kvh; a.head_dim = hd; a.block_size = kBs;
  a.ctx = c.ctx; a.window = c.window; a.scale = c.scale;
  a.kv_block_stride = block_stride; a.kv_head_stride = head_stride;
  const int rc = c.bf16kv ? r4dx_gemma_attn_ref_bf16kv(&a, 0) : r4dx_gemma_attn_ref_fp8kv(&a, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  Check(rc == 0, c.name + ": launch returns 0");
  const std::vector<uint16_t> got = out_d.CopyToHost();

  // ---- CPU oracle: double precision over the bytes the device reads ----
  double max_abs = 0.0, num = 0.0, den = 0.0, max_ref = 0.0;
  for (int t = 0; t < c.q_len; ++t) {
    const int qpos = c.ctx - c.q_len + t;
    const int klow = c.window > 0 ? std::max(0, qpos - c.window + 1) : 0;
    int khigh = std::max(qpos, ext_h[t]);
    khigh = std::min(khigh, c.ctx - 1);
    for (int h = 0; h < qh; ++h) {
      const int g = h / (qh / kvh);
      std::vector<double> s(khigh - klow + 1);
      double m = -1e300;
      for (int p = klow; p <= khigh; ++p) {
        double dot = 0.0;
        for (int d = 0; d < hd; ++d) {
          dot += static_cast<double>(Bf16ToFloat(q_h[(static_cast<size_t>(t) * qh + h) * hd + d])) *
                 cache_at(p, g, d);
        }
        s[p - klow] = dot * c.scale * (a.k_descale ? kd[g] : 1.0f);
        m = std::max(m, s[p - klow]);
      }
      double l = 0.0;
      for (double& v : s) { v = std::exp(v - m); l += v; }
      for (int d = 0; d < hd; ++d) {
        double acc = 0.0;
        for (int p = klow; p <= khigh; ++p) acc += s[p - klow] * cache_at(p, g, hd + d);
        const double ref = acc / l * (a.v_descale ? vd[g] : 1.0f);
        const double out = Bf16ToFloat(got[(static_cast<size_t>(t) * qh + h) * hd + d]);
        max_abs = std::max(max_abs, std::abs(out - ref));
        max_ref = std::max(max_ref, std::abs(ref));
        num += (out - ref) * (out - ref);
        den += ref * ref;
      }
    }
  }
  const double norm_rel = std::sqrt(num / std::max(den, 1e-30));
  std::printf("    %s: norm_rel %.3e, max_abs %.3e (max |ref| %.3e)\n", c.name.c_str(), norm_rel, max_abs, max_ref);
  // The output is one bf16 rounding of an exact value: relative 2^-9 per element.
  Check(norm_rel < 4e-3 && max_abs < 8e-3 * std::max(max_ref, 1e-3), c.name + ": matches the CPU reference");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(41);

  // sliding, ring, fp8 -- decode / verify / prefill-tail, at contexts past the ring's 1536 (wrap x2)
  const Case cases[] = {
      {"sliding ring decode  ctx 3500 q1", 16, 8, 256, 1024, 3500, 1, true, false, false, 1.0f, false},
      {"sliding ring verify  ctx 3500 q8", 16, 8, 256, 1024, 3500, 8, true, false, false, 1.0f, false},
      {"sliding ring chunk   ctx 3500 q37", 16, 8, 256, 1024, 3500, 37, true, false, false, 1.0f, false},
      {"sliding ring short   ctx 600 q96 (no wrap)", 16, 8, 256, 1024, 600, 96, true, false, false, 1.0f, false},
      {"sliding ring edge    ctx 1537 q5 (first wrap)", 16, 8, 256, 1024, 1537, 5, true, false, false, 1.0f, false},
      {"sliding ring ext     ctx 2100 q64 (image block)", 16, 8, 256, 1024, 2100, 64, true, false, true, 1.0f, false},
      {"sliding ring scale   ctx 2600 q3 scale 0.0625", 16, 8, 256, 1024, 2600, 3, true, false, false, 0.0625f, false},
      {"sliding small window ctx 700 q9 W 100 (contig)", 16, 8, 256, 100, 700, 9, false, false, false, 1.0f, false},
      {"full decode          ctx 900 q1", 16, 1, 512, 0, 900, 1, false, false, false, 1.0f, false},
      {"full verify          ctx 900 q5", 16, 1, 512, 0, 900, 5, false, false, false, 1.0f, false},
      {"full chunk           ctx 900 q30 + descales", 16, 1, 512, 0, 900, 30, false, false, false, 1.0f, false},
      {"sliding bf16 kv      ctx 500 q20 W 100", 16, 8, 256, 100, 500, 20, false, true, false, 1.0f, true},
      {"full bf16 kv         ctx 500 q4", 16, 1, 512, 0, 500, 4, false, true, false, 1.0f, false},
  };
  for (const Case& c : cases) RunCase(c, rng);

  // Shape refusals.
  {
    R4dxGemmaAttnRefArgs a{};
    DeviceBuffer<uint16_t> x(64);
    a.q = a.kv = a.block_table = a.out = P(x.data());
    a.q_len = 1; a.q_heads = 16; a.kv_heads = 8; a.head_dim = 256; a.block_size = 16; a.ctx = 10;
    a.scale = 1.0f; a.kv_block_stride = 1; a.kv_head_stride = 1;
    R4dxGemmaAttnRefArgs b = a; b.head_dim = 520;
    Check(r4dx_gemma_attn_ref_fp8kv(&b, 0) < 0, "head_dim 520 refused");
    b = a; b.head_dim = 12;
    Check(r4dx_gemma_attn_ref_fp8kv(&b, 0) < 0, "head_dim not a multiple of 8 refused");
    b = a; b.q_heads = 12; b.kv_heads = 8;
    Check(r4dx_gemma_attn_ref_fp8kv(&b, 0) < 0, "q_heads % kv_heads != 0 refused");
    b = a; b.q_len = 11;
    Check(r4dx_gemma_attn_ref_fp8kv(&b, 0) < 0, "q_len > ctx refused");
    b = a; b.window = -1;
    Check(r4dx_gemma_attn_ref_bf16kv(&b, 0) < 0, "negative window refused");
  }

  std::printf(g_fail ? "FAIL\n" : "PASS\n");
  return g_fail ? 1 : 0;
}
