// tests/kernels/test_attn_prefill_splitkv.cpp -- prefill M1: libr4d's split-KV prefill attention
// (r4d_attn_prefill_splitkv_h256_gqa6_fp8kv, via r4dx::core::r4d) against the plain prefill kernel
// and a CPU fp32 attention reference over the SAME fp8 cache (so the comparison isolates the
// attention algorithm, not the KV quantization -- test_attn_decode's convention).
//
// The contract being pinned (r4d.h):
//   1. splits <= 1 through the split-KV entry IS the plain prefill launch: bit-identical output.
//   2. split, the only differences from the unsplit kernel are where each segment's softmax
//      reference max starts (so which f16 P roundings happen) and the fp32 order of the merge; so
//      the split output's error against the fp32 reference must be of the unsplit kernel's own size
//      (checked: split max-abs error <= 1.5 x unsplit's + 2e-3, and max-rel <= 5e-2 like
//      test_attn_decode), and the split-vs-unsplit difference is printed for the record.
//   3. deterministic: the same split launch twice gives the same bits (fixed-order merge).
// Cases cover: chunk at depth 0 (causal diagonal only), depths that are not multiples of the
// 16-token page or the 48-key tile (partial pages / partial tiles), a partial query block
// (q_len 37), several query blocks (q_len 150), a split count above the tile count (empty
// segments), the 64 clamp, and a cache whose block table is a random permutation (paging). One
// case plants keys aligned with specific queries inside ONE segment so that segment's max sits
// ~30 octaves above the others: the merge's reweighting then carries the whole result.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <random>
#include <thread>
#include <vector>

#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

namespace {

struct Case {
  int depth, q_len;
  std::vector<int> splits;
  bool plant;
};

struct Err {
  double max_abs = 0, max_rel = 0, sum_sq = 0;
  size_t n = 0;
  void Add(float got, float ref) {
    const double d = std::abs(static_cast<double>(got) - ref);
    max_abs = std::max(max_abs, d);
    max_rel = std::max(max_rel, d / std::max(1e-2, std::abs(static_cast<double>(ref))));
    sum_sq += d * d;
    ++n;
  }
  double Rms() const { return n ? std::sqrt(sum_sq / n) : 0.0; }
};

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  const r4d::AttnDims dims = r4d::GetAttnDims();
  const int head_dim = dims.head_dim, gqa = dims.gqa, block_size = dims.block_size;
  const int kv_heads = 4, q_heads = kv_heads * gqa;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
  const int64_t kv_head_stride = static_cast<int64_t>(block_size) * 2 * head_dim;
  const int64_t kv_block_stride = static_cast<int64_t>(kv_heads) * kv_head_stride;

  const std::vector<Case> cases = {
      {0, 64, {2, 8}, false},              // one tile-and-a-bit, all causal diagonal
      {701, 64, {2, 3, 8, 16}, false},     // partial page and partial tile at both ends
      {3001, 64, {4, 8, 16, 32}, false},
      {2000, 37, {8}, false},              // partial query block
      {1500, 150, {2, 8}, false},          // three query blocks, the first two see fewer tiles
      {9000, 64, {8, 16, 64, 200}, false}, // 200 clamps to 64 (and segments of one tile)
      {5000, 64, {8}, true},               // one segment dominates the merge
      {32700, 64, {8, 16}, false},         // the depth band the model splits at (tp=1 / tp=2 counts)
  };

  std::mt19937 rng(20260928);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::uniform_real_distribution<float> descale_dist(0.5f, 2.0f);
  bool ok = true;
  const unsigned nthreads = std::max(1u, std::min(24u, std::thread::hardware_concurrency()));

  for (const Case& cs : cases) {
    const int q_len = cs.q_len, ctx = cs.depth + q_len;
    const int max_blocks = (ctx + block_size - 1) / block_size;

    // ---- K/V for every position, through r4dx's own paged fp8 writer, permuted block table ----
    std::vector<uint16_t> k_h(static_cast<size_t>(ctx) * kv_heads * head_dim);
    std::vector<uint16_t> v_h(k_h.size());
    for (auto& x : k_h) x = FloatToBf16(dist(rng));
    for (auto& x : v_h) x = FloatToBf16(dist(rng));
    std::vector<uint16_t> q_h(static_cast<size_t>(q_len) * q_heads * head_dim);
    for (auto& x : q_h) x = FloatToBf16(dist(rng));
    std::vector<float> k_desc(kv_heads), v_desc(kv_heads);
    for (auto& d : k_desc) d = descale_dist(rng);
    for (auto& d : v_desc) d = descale_dist(rng);
    if (cs.plant) {
      // Keys at depth ~3600..3700 (one segment at splits 8: tps = ceil(106 tiles / 8) = 14 tiles =
      // 672 keys, so keys 3360..4031 are segment 5) aligned with four queries each, scaled so the
      // score is ~30 octaves above the uniform background. The key's V stays its own.
      const int plant_key[] = {3600, 3650, 3700, 3701};
      const int plant_q[] = {3, 20, 41, 63};
      for (int j = 0; j < 4; ++j) {
        for (int kvh = 0; kvh < kv_heads; ++kvh) {
          const size_t dst = (static_cast<size_t>(plant_key[j]) * kv_heads + kvh) * head_dim;
          const size_t src = (static_cast<size_t>(plant_q[j]) * q_heads + kvh * gqa) * head_dim;
          for (int d = 0; d < head_dim; ++d) {
            k_h[dst + d] = FloatToBf16(4.0f * Bf16ToFloat(q_h[src + d]));
          }
        }
      }
    }
    std::vector<int32_t> perm(static_cast<size_t>(max_blocks));
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    std::vector<int32_t> slots(static_cast<size_t>(ctx));
    for (int t = 0; t < ctx; ++t) slots[t] = perm[t / block_size] * block_size + t % block_size;

    DeviceBuffer<uint16_t> k_d(k_h.size()), v_d(v_h.size()), q_d(q_h.size());
    DeviceBuffer<int32_t> slot_d(slots.size()), bt_d(perm.size()), seq_d(1);
    DeviceBuffer<float> kd_d(kv_heads), vd_d(kv_heads);
    DeviceBuffer<uint8_t> cache_d(static_cast<size_t>(max_blocks) * kv_block_stride);
    k_d.CopyFromHost(k_h);
    v_d.CopyFromHost(v_h);
    q_d.CopyFromHost(q_h);
    slot_d.CopyFromHost(slots);
    bt_d.CopyFromHost(perm);
    const std::vector<int32_t> seq_h = {ctx};
    seq_d.CopyFromHost(seq_h);
    kd_d.CopyFromHost(k_desc);
    vd_d.CopyFromHost(v_desc);
    cache_d.Zero();
    r4dx_kv_write_paged_fp8_hnd(reinterpret_cast<int64_t>(k_d.data()),
                                 reinterpret_cast<int64_t>(v_d.data()),
                                 reinterpret_cast<int64_t>(slot_d.data()),
                                 reinterpret_cast<int64_t>(kd_d.data()),
                                 reinterpret_cast<int64_t>(vd_d.data()),
                                 reinterpret_cast<int64_t>(cache_d.data()), ctx, kv_heads, head_dim,
                                 block_size, kv_block_stride, kv_head_stride, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const std::vector<uint8_t> cache_h = cache_d.CopyToHost();

    // ---- fp32 reference over the dequantized cache ---------------------------------------------
    std::vector<float> kf(static_cast<size_t>(ctx) * kv_heads * head_dim), vf(kf.size());
    for (int t = 0; t < ctx; ++t) {
      const int64_t blk = perm[t / block_size], off = t % block_size;
      for (int kvh = 0; kvh < kv_heads; ++kvh) {
        const int64_t base = blk * kv_block_stride + kvh * kv_head_stride + off * 2 * head_dim;
        const size_t dst = (static_cast<size_t>(t) * kv_heads + kvh) * head_dim;
        for (int d = 0; d < head_dim; ++d) {
          kf[dst + d] = Fp8E4M3ToFloat(cache_h[base + d]) * k_desc[kvh];
          vf[dst + d] = Fp8E4M3ToFloat(cache_h[base + head_dim + d]) * v_desc[kvh];
        }
      }
    }
    std::vector<float> ref(q_h.size());
    auto ref_heads = [&](int h0, int h1) {
      std::vector<float> logit(static_cast<size_t>(ctx));
      std::vector<double> o(static_cast<size_t>(head_dim));
      for (int qh = h0; qh < h1; ++qh) {
        const int kvh = qh / gqa;
        for (int qi = 0; qi < q_len; ++qi) {
          const float* qp = nullptr;
          std::vector<float> qv(static_cast<size_t>(head_dim));
          const size_t qoff = (static_cast<size_t>(qi) * q_heads + qh) * head_dim;
          for (int d = 0; d < head_dim; ++d) qv[d] = Bf16ToFloat(q_h[qoff + d]);
          qp = qv.data();
          const int klimit = ctx - q_len + qi;
          float mx = -1e30f;
          for (int t = 0; t <= klimit; ++t) {
            const float* kp = &kf[(static_cast<size_t>(t) * kv_heads + kvh) * head_dim];
            float dot = 0.0f;
            for (int d = 0; d < head_dim; ++d) dot += qp[d] * kp[d];
            logit[t] = dot * scale;
            mx = std::max(mx, logit[t]);
          }
          double sum = 0.0;
          std::fill(o.begin(), o.end(), 0.0);
          for (int t = 0; t <= klimit; ++t) {
            const double p = std::exp(static_cast<double>(logit[t]) - mx);
            sum += p;
            const float* vp = &vf[(static_cast<size_t>(t) * kv_heads + kvh) * head_dim];
            for (int d = 0; d < head_dim; ++d) o[d] += p * vp[d];
          }
          for (int d = 0; d < head_dim; ++d) ref[qoff + d] = static_cast<float>(o[d] / sum);
        }
      }
    };
    {
      std::vector<std::thread> th;
      const int per = (q_heads + static_cast<int>(nthreads) - 1) / static_cast<int>(nthreads);
      for (int h0 = 0; h0 < q_heads; h0 += per) th.emplace_back(ref_heads, h0, std::min(q_heads, h0 + per));
      for (auto& t : th) t.join();
    }

    // ---- kernels -----------------------------------------------------------------------------
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

    DeviceBuffer<uint16_t> out_d(q_h.size());
    auto run = [&](int splits, bool via_splitkv) -> std::vector<uint16_t> {
      R4DArgs b = a;
      b.out = out_d.data();
      b.splits = splits;
      const int64_t bytes = r4d::AttnPrefillSplitKvScratchBytes(b);
      DeviceBuffer<uint8_t> scratch(bytes > 0 ? static_cast<size_t>(bytes) : 1);
      b.scratch = bytes > 0 ? scratch.data() : nullptr;
      R4DX_HIP_CHECK(hipMemset(out_d.data(), 0xFF, out_d.size() * 2));  // NaN poison
      if (via_splitkv) r4d::AttnPrefillSplitKvFp8Kv(b, nullptr);
      else r4d::AttnPrefillFp8Kv(b, nullptr);
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      return out_d.CopyToHost();
    };

    const std::vector<uint16_t> base = run(0, false);
    Err ebase;
    for (size_t i = 0; i < ref.size(); ++i) ebase.Add(Bf16ToFloat(base[i]), ref[i]);
    std::printf("[depth %5d q_len %3d ctx %5d%s] unsplit   vs fp32 ref: max abs %.3e  max rel %.3e  rms %.3e\n",
                cs.depth, q_len, ctx, cs.plant ? " planted" : "", ebase.max_abs, ebase.max_rel,
                ebase.Rms());
    bool case_ok = std::isfinite(ebase.max_abs) && ebase.max_rel < 5e-2;

    for (int s1 : {0, 1}) {
      const std::vector<uint16_t> same = run(s1, true);
      size_t diff = 0;
      for (size_t i = 0; i < same.size(); ++i) diff += same[i] != base[i];
      std::printf("    splitkv entry, splits %d: %zu/%zu bf16 differ from the plain launch\n", s1,
                  diff, same.size());
      case_ok = case_ok && diff == 0;
    }

    for (int sp : cs.splits) {
      const std::vector<uint16_t> got = run(sp, true);
      const std::vector<uint16_t> again = run(sp, true);
      Err eref, eunsplit;
      size_t diff = 0, nondet = 0, nonfinite = 0;
      for (size_t i = 0; i < got.size(); ++i) {
        const float g = Bf16ToFloat(got[i]);
        if (!std::isfinite(g)) ++nonfinite;
        eref.Add(g, ref[i]);
        eunsplit.Add(g, Bf16ToFloat(base[i]));
        diff += got[i] != base[i];
        nondet += got[i] != again[i];
      }
      const bool pass = nonfinite == 0 && nondet == 0 && eref.max_rel < 5e-2 &&
                        eref.max_abs <= 1.5 * ebase.max_abs + 2e-3;
      std::printf("    splits %3d vs fp32 ref: max abs %.3e  max rel %.3e  rms %.3e | vs unsplit: "
                  "max abs %.3e rms %.3e, %zu/%zu bf16 differ | rerun differs %zu | %s\n",
                  sp, eref.max_abs, eref.max_rel, eref.Rms(), eunsplit.max_abs, eunsplit.Rms(),
                  diff, got.size(), nondet, pass ? "ok" : "FAIL");
      case_ok = case_ok && pass;
    }
    ok = ok && case_ok;
  }

  // A split launch without scratch is a rejected shape, not a silent unsplit run.
  {
    R4DArgs b{};
    b.head_dim = head_dim;
    b.block_size = block_size;
    b.q_heads = q_heads;
    b.kv_heads = kv_heads;
    b.num_seqs = 1;
    b.q_len = 64;
    b.splits = 4;
    const int rc = r4d_attn_prefill_splitkv_h256_gqa6_fp8kv(&b, nullptr);
    std::printf("split launch with null scratch returns %d (want -5)\n", rc);
    ok = ok && rc == -5;
  }

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
