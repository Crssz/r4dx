// tests/kernels/test_attn_h128_gqa7.cpp -- libr4d's h128_gqa7 attention (docs/nanbeige-plan.md 3.3, tasks N5-1 / N5-2)
// against the stage-0 reference kernel on the SAME cache, bf16 KV (r4dx_gemma_attn_ref_bf16kv) and fp8-e4m3 KV with random
// per-head descales (r4dx_gemma_attn_ref_fp8kv; the suite runs once per dtype): Nanbeige's geometry, head_dim 128,
// scale 128^-0.5, contiguous table, at TP=1 (56 q / 8 kv heads) and at a TP=2 rank's shapes (28 q / 4 kv heads).
//   * decode / verify: q_len 1, 2, 7, 9 at contexts 1..4097 and the split law's own count plus explicit counts 1 and 16;
//   * prefill (plain): q_len 1..256 as a chunk tail at several contexts (a partial 32-query workgroup, a partial tile;
//     q_len 33 and 100 put queries in a second workgroup, the case a 7-head row map gets wrong without whole queries);
//   * prefill split-KV: counts 2, 5, 32 and the library's clamp, at deep contexts;
//   * determinism (two launches, same bytes), the unsplit == splits 1 identity, and the argument refusals.
// GPU test (ctest sets HIP_VISIBLE_DEVICES=1); needs no data files.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/model/attention/attn_ref.h"

using namespace r4dx::core;

namespace {
int kQH = 56, kKVH = 8;  // the geometry under test (TP=1 first, then a TP=2 rank)
bool g_fp8 = false;      // the KV dtype under test
constexpr int kD = 128, kBs = 16;
const float kScale = 1.0f / std::sqrt(128.0f);

int g_fail = 0;
void Check(bool cond, const std::string& what) {
  std::printf("%-100s %s\n", what.c_str(), cond ? "ok" : "FAIL");
  if (!cond) ++g_fail;
}
int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

struct Cache {
  int ctx;
  DeviceBuffer<uint16_t> kv;   // bf16 cache
  DeviceBuffer<uint8_t> kv8;   // fp8 cache (g_fp8)
  DeviceBuffer<float> k_descale, v_descale;  // fp32 [kv_heads] (fp8 only)
  DeviceBuffer<int32_t> table;
  const void* Data() const { return g_fp8 ? static_cast<const void*>(kv8.data()) : static_cast<const void*>(kv.data()); }
  int64_t head_stride, block_stride;
  int blocks;
};

Cache MakeCache(int ctx, std::mt19937& rng) {
  Cache c;
  c.ctx = ctx;
  c.blocks = (ctx + kBs - 1) / kBs;
  c.head_stride = static_cast<int64_t>(kBs) * 2 * kD;
  c.block_stride = kKVH * c.head_stride;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  if (g_fp8) {
    // Stored values use e4m3's range (|x| up to about 400 of 448); the descales (0.002..0.02) put the real K / V near
    // the bf16 case's scale, different per head so a head mix-up shows.
    std::vector<uint8_t> h(static_cast<size_t>(c.blocks) * c.block_stride, 0);
    for (auto& v : h) v = FloatToFp8E4M3(std::clamp(100.0f * nd(rng), -440.0f, 440.0f));
    c.kv8.Resize(h.size());
    c.kv8.CopyFromHost(h);
    std::uniform_real_distribution<float> ud(0.002f, 0.02f);
    std::vector<float> kd(kKVH), vd(kKVH);
    for (int i = 0; i < kKVH; ++i) { kd[i] = ud(rng); vd[i] = ud(rng); }
    c.k_descale.Resize(kd.size());
    c.k_descale.CopyFromHost(kd);
    c.v_descale.Resize(vd.size());
    c.v_descale.CopyFromHost(vd);
  } else {
    std::vector<uint16_t> h(static_cast<size_t>(c.blocks) * c.block_stride, 0);
    for (auto& v : h) v = FloatToBf16(nd(rng));  // finite everywhere, incl. the unused tail of the last block
    c.kv.Resize(h.size());
    c.kv.CopyFromHost(h);
  }
  const int entries = c.blocks + 4;  // the layers' table has 4 slack entries, clamped to the last block
  std::vector<int32_t> t(entries);
  for (int i = 0; i < entries; ++i) t[i] = std::min(i, c.blocks - 1);
  c.table.Resize(t.size());
  c.table.CopyFromHost(t);
  return c;
}

std::vector<uint16_t> RandomQ(int q_len, std::mt19937& rng) {
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint16_t> q(static_cast<size_t>(q_len) * kQH * kD);
  for (auto& v : q) v = FloatToBf16(nd(rng));
  return q;
}

std::vector<uint16_t> Reference(const Cache& c, const DeviceBuffer<uint16_t>& q, int q_len) {
  DeviceBuffer<uint16_t> out(static_cast<size_t>(q_len) * kQH * kD);
  out.Zero();
  R4dxGemmaAttnRefArgs a{};
  a.q = P(q.data());
  a.kv = P(c.Data());
  a.block_table = P(c.table.data());
  a.out = P(out.data());
  if (g_fp8) {
    a.k_descale = P(c.k_descale.data());
    a.v_descale = P(c.v_descale.data());
  }
  a.q_len = q_len; a.q_heads = kQH; a.kv_heads = kKVH; a.head_dim = kD; a.block_size = kBs;
  a.ctx = c.ctx; a.window = 0; a.scale = kScale;
  a.kv_block_stride = c.block_stride; a.kv_head_stride = c.head_stride;
  Check((g_fp8 ? r4dx_gemma_attn_ref_fp8kv(&a, 0) : r4dx_gemma_attn_ref_bf16kv(&a, 0)) == 0, "reference launch returns 0");
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  return out.CopyToHost();
}

R4DArgs Args(const Cache& c, const DeviceBuffer<uint16_t>& q, DeviceBuffer<uint16_t>& out, DeviceBuffer<int32_t>& used,
             int q_len, int splits) {
  R4DArgs a{};
  a.q = q.data();
  a.kv = c.Data();
  a.block_table = c.table.data();
  a.seqused_k = used.data();
  if (g_fp8) {
    a.k_descale = c.k_descale.data();
    a.v_descale = c.v_descale.data();
  }
  a.out = out.data();
  a.num_seqs = 1; a.q_len = q_len; a.q_heads = kQH; a.kv_heads = kKVH; a.head_dim = kD; a.block_size = kBs;
  a.max_blocks = c.blocks + 4;
  a.kv_block_stride = c.block_stride; a.kv_head_stride = c.head_stride;
  a.scale = kScale; a.splits = splits; a.max_ctx = c.ctx;
  return a;
}

struct Err { double norm_rel, max_abs; };
Err Compare(const std::vector<uint16_t>& got, const std::vector<uint16_t>& want) {
  double num = 0, den = 0, mx = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = Bf16ToFloat(got[i]), w = Bf16ToFloat(want[i]);
    num += (g - w) * (g - w);
    den += w * w;
    mx = std::max(mx, std::abs(g - w));
  }
  return {std::sqrt(num / std::max(den, 1e-30)), mx};
}
// bf16 output, f16 P operands, fp32 accumulation: a few bf16 ulps of the reference in rel L2.
bool Within(const Err& e) { return e.norm_rel < 1.5e-2 && e.max_abs < 0.06; }

enum class Mode { kDecode, kPrefill, kSplit };

void Run(Mode m, int ctx, int q_len, int splits, std::mt19937& rng) {
  Cache c = MakeCache(ctx, rng);
  const std::vector<uint16_t> qh = RandomQ(q_len, rng);
  DeviceBuffer<uint16_t> q(qh.size());
  q.CopyFromHost(qh);
  DeviceBuffer<int32_t> used(1);
  used.CopyFromHost(std::vector<int32_t>{ctx});
  DeviceBuffer<uint16_t> out(qh.size()), out2(qh.size());
  out.Zero();
  out2.Zero();
  const std::vector<uint16_t> want = Reference(c, q, q_len);

  DeviceBuffer<uint8_t> scratch;
  R4DArgs a = Args(c, q, out, used, q_len, splits);
  auto launch = [&](R4DArgs* args) {
    if (m == Mode::kDecode) {
      const int64_t sb = r4d_attn_decode_h128_gqa7_scratch_bytes(args);
      if (static_cast<int64_t>(scratch.size()) < sb) scratch.Resize(static_cast<size_t>(sb));
      args->scratch = scratch.data();
      return g_fp8 ? r4d_attn_decode_h128_gqa7_fp8kv(args, nullptr) : r4d_attn_decode_h128_gqa7_bf16kv(args, nullptr);
    }
    if (m == Mode::kSplit) {
      const int64_t sb = r4d_attn_prefill_splitkv_h128_gqa7_scratch_bytes(args);
      if (static_cast<int64_t>(scratch.size()) < sb) scratch.Resize(static_cast<size_t>(std::max<int64_t>(sb, 1)));
      args->scratch = scratch.data();
      return g_fp8 ? r4d_attn_prefill_splitkv_h128_gqa7_fp8kv(args, nullptr)
                   : r4d_attn_prefill_splitkv_h128_gqa7_bf16kv(args, nullptr);
    }
    return g_fp8 ? r4d_attn_prefill_h128_gqa7_fp8kv(args, nullptr) : r4d_attn_prefill_h128_gqa7_bf16kv(args, nullptr);
  };
  const char* kind = m == Mode::kDecode ? "decode" : (m == Mode::kSplit ? "prefill split-KV" : "prefill");
  int rc = launch(&a);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  char name[160];
  std::snprintf(name, sizeof name, "[%s %d/%d heads] %s ctx %d q_len %d splits %d", g_fp8 ? "fp8" : "bf16", kQH, kKVH,
                kind, ctx, q_len, splits);
  if (rc != 0) {
    Check(false, std::string(name) + ": returned " + std::to_string(rc));
    return;
  }
  const std::vector<uint16_t> got = out.CopyToHost();
  const Err e = Compare(got, want);
  std::printf("    norm_rel %.3e max_abs %.3e\n", e.norm_rel, e.max_abs);
  Check(Within(e), std::string(name) + ": matches the reference kernel");

  R4DArgs b = Args(c, q, out2, used, q_len, splits);
  rc = launch(&b);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  Check(rc == 0 && out2.CopyToHost() == got, std::string(name) + ": a second launch gives the same bytes");
}

void Suite(std::mt19937& rng) {
  for (int q : {1, 2, 7, 9}) {
    for (int ctx : {16, 17, 100, 1000, 4097}) {
      if (q > ctx) continue;
      Run(Mode::kDecode, ctx, q, 0, rng);
    }
  }
  for (int s : {1, 16}) Run(Mode::kDecode, 1000, 4, s, rng);

  for (int ctx : {300, 1000, 2100}) {
    for (int q : {10, 32, 33, 100, 256}) {
      if (q > ctx) continue;
      Run(Mode::kPrefill, ctx, q, 0, rng);
    }
  }
  Run(Mode::kPrefill, 1, 1, 0, rng);
  Run(Mode::kPrefill, 64, 64, 0, rng);
  Run(Mode::kPrefill, 79, 79, 0, rng);  // the diagonal key of query 32 opens the second 64-key tile

  for (int s : {2, 5, 32, 100}) Run(Mode::kSplit, 4000, 64, s, rng);
  Run(Mode::kSplit, 4000, 256, 8, rng);
  Run(Mode::kSplit, 300, 64, 1, rng);  // splits <= 1 is the plain launch
}
}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(128);

  for (bool fp8 : {false, true}) {
    g_fp8 = fp8;
    kQH = 56;
    kKVH = 8;
    Suite(rng);        // TP=1: 56 q / 8 kv heads
    kQH = 28;
    kKVH = 4;
    Suite(rng);        // a TP=2 rank: 28 q / 4 kv heads
  }
  g_fp8 = false;
  kQH = 56;
  kKVH = 8;

  // fp8: splits 1 is the plain launch bit for bit, as for bf16 below.
  {
    g_fp8 = true;
    Cache c = MakeCache(1500, rng);
    const std::vector<uint16_t> qh = RandomQ(64, rng);
    DeviceBuffer<uint16_t> q(qh.size()), o1(qh.size()), o2(qh.size());
    q.CopyFromHost(qh);
    DeviceBuffer<int32_t> used(1);
    used.CopyFromHost(std::vector<int32_t>{1500});
    R4DArgs a = Args(c, q, o1, used, 64, 0), b = Args(c, q, o2, used, 64, 1);
    const int r1 = r4d_attn_prefill_h128_gqa7_fp8kv(&a, nullptr);
    const int r2 = r4d_attn_prefill_splitkv_h128_gqa7_fp8kv(&b, nullptr);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    Check(r1 == 0 && r2 == 0 && o1.CopyToHost() == o2.CopyToHost(),
          "fp8: split-KV prefill with splits 1 == the plain launch");
    g_fp8 = false;
  }

  // splits 1 is the plain launch bit for bit (same instantiation).
  {
    Cache c = MakeCache(1500, rng);
    const std::vector<uint16_t> qh = RandomQ(64, rng);
    DeviceBuffer<uint16_t> q(qh.size()), o1(qh.size()), o2(qh.size());
    q.CopyFromHost(qh);
    DeviceBuffer<int32_t> used(1);
    used.CopyFromHost(std::vector<int32_t>{1500});
    R4DArgs a = Args(c, q, o1, used, 64, 0), b = Args(c, q, o2, used, 64, 1);
    const int r1 = r4d_attn_prefill_h128_gqa7_bf16kv(&a, nullptr);
    const int r2 = r4d_attn_prefill_splitkv_h128_gqa7_bf16kv(&b, nullptr);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    Check(r1 == 0 && r2 == 0 && o1.CopyToHost() == o2.CopyToHost(), "split-KV prefill with splits 1 == the plain launch");
  }

  // Refusals (a negative code, nothing launched).
  {
    Cache c = MakeCache(200, rng);
    const std::vector<uint16_t> qh = RandomQ(20, rng);
    DeviceBuffer<uint16_t> q(qh.size()), o(qh.size());
    q.CopyFromHost(qh);
    DeviceBuffer<int32_t> used(1);
    used.CopyFromHost(std::vector<int32_t>{200});
    DeviceBuffer<uint8_t> scratch(1 << 20);
    R4DArgs a = Args(c, q, o, used, 20, 0);
    a.scratch = scratch.data();
    R4DArgs b = a; b.head_dim = 64;
    Check(r4d_attn_prefill_h128_gqa7_bf16kv(&b, nullptr) == -1, "head_dim 64 refused (-1)");
    b = a; b.block_size = 32;
    Check(r4d_attn_decode_h128_gqa7_bf16kv(&b, nullptr) == -1, "block 32 refused (-1)");
    b = a; b.q_heads = 48;  // gqa 6
    Check(r4d_attn_prefill_h128_gqa7_bf16kv(&b, nullptr) == -2, "gqa 6 refused (-2)");
    b = a; b.q_len = 10;
    Check(r4d_attn_decode_h128_gqa7_bf16kv(&b, nullptr) == -4, "q_len 10 (70 rows) refused by decode (-4)");
    b = a; b.q_len = 4; b.scratch = nullptr;
    Check(r4d_attn_decode_h128_gqa7_bf16kv(&b, nullptr) == -5, "decode without scratch refused (-5)");
    b = a; b.splits = 4; b.scratch = nullptr;
    Check(r4d_attn_prefill_splitkv_h128_gqa7_bf16kv(&b, nullptr) == -5, "split-KV prefill without scratch refused (-5)");
  }

  // The geometry registry lists the family.
  {
    bool found = false;
    for (int i = 0; i < r4d_attn_geom_count(); ++i) {
      const R4DAttnGeom* g = r4d_attn_geom_at(i);
      found = found || (g->gqa == 7 && !g->windowed && g->head_dim == 128 && g->block_size == 16);
    }
    Check(found, "r4d_attn_geom lists the h128_gqa7 geometry");
  }

  std::printf(g_fail ? "FAIL\n" : "PASS\n");
  return g_fail ? 1 : 0;
}
