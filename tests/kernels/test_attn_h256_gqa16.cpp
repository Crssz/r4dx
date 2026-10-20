// tests/kernels/test_attn_h256_gqa16.cpp -- libr4d's h256_gqa16 fp8-KV attention (docs/qwen35moe-plan.md M3) against the stage-0
// reference kernel r4dx_gemma_attn_ref_fp8kv on the SAME fp8 cache: the Qwen3.5 MoE family's geometry, 32 q / 2 kv heads,
// head_dim 256, scale 1/16, contiguous table.
//   * decode / verify: q_len 1..4 at contexts 1, 17, 100, 1000, 4097 and the split law's own count plus explicit counts;
//   * prefill (plain): q_len 1..300 as a chunk tail at several contexts (a partial 24-position workgroup, a partial tile);
//   * prefill split-KV: counts 2, 5, 32 and the library's clamp, at deep contexts;
//   * determinism (two launches, same bytes), the unsplit == splits 1 identity, the descales (non-trivial per-head values),
//     and the argument refusals;
//   * the core wrappers' gqa dispatch (r4dx::core::r4d::Attn*Fp8Kv serve this geometry and still the 27B's gqa 6).
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
#include "r4dx/core/r4d.hpp"
#include "r4dx/model/attention/attn_ref.h"

using namespace r4dx::core;

namespace {
constexpr int kQH = 32, kKVH = 2, kD = 256, kBs = 16;
constexpr float kScale = 1.0f / 16.0f;

int g_fail = 0;
void Check(bool cond, const std::string& what) {
  std::printf("%-100s %s\n", what.c_str(), cond ? "ok" : "FAIL");
  if (!cond) ++g_fail;
}
int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

struct Cache {
  int ctx;
  DeviceBuffer<uint8_t> kv;
  DeviceBuffer<int32_t> table;
  DeviceBuffer<float> kd, vd;
  int64_t head_stride, block_stride;
  int blocks;
};

Cache MakeCache(int ctx, std::mt19937& rng, bool descales) {
  Cache c;
  c.ctx = ctx;
  c.blocks = (ctx + kBs - 1) / kBs;
  c.head_stride = static_cast<int64_t>(kBs) * 2 * kD;
  c.block_stride = kKVH * c.head_stride;
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<uint8_t> h(static_cast<size_t>(c.blocks) * c.block_stride, 0);
  for (auto& v : h) v = FloatToFp8E4M3(nd(rng));  // finite everywhere, incl. the unused tail of the last block
  c.kv.Resize(h.size());
  c.kv.CopyFromHost(h);
  const int entries = c.blocks + 4;
  std::vector<int32_t> t(entries);
  for (int i = 0; i < entries; ++i) t[i] = std::min(i, c.blocks - 1);
  c.table.Resize(t.size());
  c.table.CopyFromHost(t);
  if (descales) {
    c.kd.Resize(kKVH);
    c.vd.Resize(kKVH);
    c.kd.CopyFromHost(std::vector<float>{0.5f, 2.0f});
    c.vd.CopyFromHost(std::vector<float>{1.5f, 0.25f});
  }
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
  a.kv = P(c.kv.data());
  a.block_table = P(c.table.data());
  a.out = P(out.data());
  a.k_descale = c.kd.size() ? P(c.kd.data()) : 0;
  a.v_descale = c.vd.size() ? P(c.vd.data()) : 0;
  a.q_len = q_len; a.q_heads = kQH; a.kv_heads = kKVH; a.head_dim = kD; a.block_size = kBs;
  a.ctx = c.ctx; a.window = 0; a.scale = kScale;
  a.kv_block_stride = c.block_stride; a.kv_head_stride = c.head_stride;
  Check(r4dx_gemma_attn_ref_fp8kv(&a, 0) == 0, "reference launch returns 0");
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  return out.CopyToHost();
}

R4DArgs Args(const Cache& c, const DeviceBuffer<uint16_t>& q, DeviceBuffer<uint16_t>& out, DeviceBuffer<int32_t>& used,
             int q_len, int splits) {
  R4DArgs a{};
  a.q = q.data();
  a.kv = c.kv.data();
  a.block_table = c.table.data();
  a.seqused_k = used.data();
  a.out = out.data();
  a.k_descale = c.kd.size() ? c.kd.data() : nullptr;
  a.v_descale = c.vd.size() ? c.vd.data() : nullptr;
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

void Run(Mode m, int ctx, int q_len, int splits, std::mt19937& rng, bool descales = false) {
  Cache c = MakeCache(ctx, rng, descales);
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
      const int64_t sb = r4d_attn_decode_h256_gqa16_scratch_bytes(args);
      if (static_cast<int64_t>(scratch.size()) < sb) scratch.Resize(static_cast<size_t>(sb));
      args->scratch = scratch.data();
      return r4d_attn_decode_h256_gqa16_fp8kv(args, nullptr);
    }
    if (m == Mode::kSplit) {
      const int64_t sb = r4d_attn_prefill_splitkv_h256_gqa16_scratch_bytes(args);
      if (static_cast<int64_t>(scratch.size()) < sb) scratch.Resize(static_cast<size_t>(std::max<int64_t>(sb, 1)));
      args->scratch = scratch.data();
      return r4d_attn_prefill_splitkv_h256_gqa16_fp8kv(args, nullptr);
    }
    return r4d_attn_prefill_h256_gqa16_fp8kv(args, nullptr);
  };
  const char* kind = m == Mode::kDecode ? "decode" : (m == Mode::kSplit ? "prefill split-KV" : "prefill");
  const int rc = launch(&a);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  char name[160];
  std::snprintf(name, sizeof name, "%s ctx %d q_len %d splits %d%s", kind, ctx, q_len, splits, descales ? " descales" : "");
  if (rc != 0) {
    Check(false, std::string(name) + ": returned " + std::to_string(rc));
    return;
  }
  const std::vector<uint16_t> got = out.CopyToHost();
  const Err e = Compare(got, want);
  std::printf("    norm_rel %.3e max_abs %.3e\n", e.norm_rel, e.max_abs);
  Check(Within(e), std::string(name) + ": matches the reference kernel");

  R4DArgs b = Args(c, q, out2, used, q_len, splits);
  const int rc2 = launch(&b);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  Check(rc2 == 0 && out2.CopyToHost() == got, std::string(name) + ": a second launch gives the same bytes");
}
}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(256);

  for (int q : {1, 2, 3, 4}) {
    for (int ctx : {16, 17, 100, 1000, 4097}) {
      if (q > ctx) continue;
      Run(Mode::kDecode, ctx, q, 0, rng);
    }
  }
  for (int s : {1, 16}) Run(Mode::kDecode, 1000, 4, s, rng);
  Run(Mode::kDecode, 1000, 1, 0, rng, /*descales=*/true);

  for (int ctx : {300, 1000, 2100}) {
    for (int q : {5, 24, 25, 64, 100, 256}) {
      if (q > ctx) continue;
      Run(Mode::kPrefill, ctx, q, 0, rng);
    }
  }
  Run(Mode::kPrefill, 1, 1, 0, rng);
  Run(Mode::kPrefill, 64, 64, 0, rng);
  Run(Mode::kPrefill, 300, 64, 0, rng, /*descales=*/true);

  for (int s : {2, 5, 32, 100}) Run(Mode::kSplit, 4000, 64, s, rng);
  Run(Mode::kSplit, 4000, 256, 8, rng);
  Run(Mode::kSplit, 300, 64, 1, rng);  // splits <= 1 is the plain launch

  // splits 1 is the plain launch bit for bit (same instantiation).
  {
    Cache c = MakeCache(1500, rng, false);
    const std::vector<uint16_t> qh = RandomQ(64, rng);
    DeviceBuffer<uint16_t> q(qh.size()), o1(qh.size()), o2(qh.size());
    q.CopyFromHost(qh);
    DeviceBuffer<int32_t> used(1);
    used.CopyFromHost(std::vector<int32_t>{1500});
    R4DArgs a = Args(c, q, o1, used, 64, 0), b = Args(c, q, o2, used, 64, 1);
    const int r1 = r4d_attn_prefill_h256_gqa16_fp8kv(&a, nullptr);
    const int r2 = r4d_attn_prefill_splitkv_h256_gqa16_fp8kv(&b, nullptr);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    Check(r1 == 0 && r2 == 0 && o1.CopyToHost() == o2.CopyToHost(), "split-KV prefill with splits 1 == the plain launch");
  }

  // The core wrappers pick the gqa 16 kernels from the call's own head counts.
  {
    Cache c = MakeCache(500, rng, false);
    const std::vector<uint16_t> qh = RandomQ(40, rng);
    DeviceBuffer<uint16_t> q(qh.size()), o1(qh.size()), o2(qh.size()), o3(qh.size());
    q.CopyFromHost(qh);
    DeviceBuffer<int32_t> used(1);
    used.CopyFromHost(std::vector<int32_t>{500});
    R4DArgs a = Args(c, q, o1, used, 40, 0);
    r4dx::core::r4d::AttnPrefillFp8Kv(a, nullptr);
    R4DArgs b = Args(c, q, o2, used, 40, 0);
    r4d_attn_prefill_h256_gqa16_fp8kv(&b, nullptr);
    R4DArgs d = Args(c, q, o3, used, 40, 0);
    r4dx::core::r4d::AttnPrefillExactFp8Kv(d, nullptr);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    Check(o1.CopyToHost() == o2.CopyToHost(), "core AttnPrefillFp8Kv at gqa 16 == r4d_attn_prefill_h256_gqa16_fp8kv");
    Check(o3.CopyToHost() == o2.CopyToHost(), "core AttnPrefillExactFp8Kv at gqa 16 == the plain gqa 16 launch");
  }

  // Refusals (a negative code, nothing launched).
  {
    Cache c = MakeCache(200, rng, false);
    const std::vector<uint16_t> qh = RandomQ(20, rng);
    DeviceBuffer<uint16_t> q(qh.size()), o(qh.size());
    q.CopyFromHost(qh);
    DeviceBuffer<int32_t> used(1);
    used.CopyFromHost(std::vector<int32_t>{200});
    DeviceBuffer<uint8_t> scratch(1 << 20);
    R4DArgs a = Args(c, q, o, used, 20, 0);
    a.scratch = scratch.data();
    R4DArgs b = a; b.head_dim = 128;
    Check(r4d_attn_prefill_h256_gqa16_fp8kv(&b, nullptr) == -1, "head_dim 128 refused (-1)");
    b = a; b.block_size = 32;
    Check(r4d_attn_decode_h256_gqa16_fp8kv(&b, nullptr) == -1, "block 32 refused (-1)");
    b = a; b.q_heads = 24;  // gqa 12
    Check(r4d_attn_prefill_h256_gqa16_fp8kv(&b, nullptr) == -2, "gqa 12 refused (-2)");
    Check(r4d_attn_decode_h256_gqa16_fp8kv(&a, nullptr) == -4, "q_len 20 (320 rows) refused by decode (-4)");
    b = a; b.q_len = 4; b.scratch = nullptr;
    Check(r4d_attn_decode_h256_gqa16_fp8kv(&b, nullptr) == -5, "decode without scratch refused (-5)");
    b = a; b.splits = 4; b.scratch = nullptr;
    Check(r4d_attn_prefill_splitkv_h256_gqa16_fp8kv(&b, nullptr) == -5, "split-KV prefill without scratch refused (-5)");
  }

  // The geometry registry lists the family.
  {
    bool found = false;
    for (int i = 0; i < r4d_attn_geom_count(); ++i) {
      const R4DAttnGeom* g = r4d_attn_geom_at(i);
      found = found || (g->gqa == 16 && !g->windowed && g->head_dim == 256 && g->block_size == 16);
    }
    Check(found, "r4d_attn_geom lists the h256_gqa16 geometry");
  }

  std::printf(g_fail ? "FAIL\n" : "PASS\n");
  return g_fail ? 1 : 0;
}
