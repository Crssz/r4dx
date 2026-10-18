// tests/kernels/tool_attn_h64_bench.cpp -- libr4d h64_gqa4 bf16-KV attention vs the stage-0 reference kernel
// (docs/lfm2-plan.md 3.4, task L4-1): per-launch time and KV read rate at LFM2's geometry (32 q / 8 kv heads,
// head_dim 64) for decode (q_len 1), a 64-row prefill chunk plain and split by the layer's law, over contexts
// 2k / 8k / 32k / 128k. Usage: $env:HIP_VISIBLE_DEVICES='1'; tool_attn_h64_bench.exe [--ctx 2048,8192] [--reps N]
// Times are the median of --reps launches (hip events); the cache holds random bf16 (timing does not depend on values).
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "r4d.h"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/model/attention/attn_ref.h"
#include "r4dx/model/attention/lfm2_attention_layer.hpp"

using namespace r4dx::core;

namespace {
constexpr int kQH = 32, kKVH = 8, kD = 64, kBs = 16;
int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }

template <class F>
double MedianUs(F&& f, int reps) {
  hipEvent_t e0, e1;
  R4DX_HIP_CHECK(hipEventCreate(&e0));
  R4DX_HIP_CHECK(hipEventCreate(&e1));
  for (int i = 0; i < 3; ++i) f();
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) {
    R4DX_HIP_CHECK(hipEventRecord(e0, nullptr));
    f();
    R4DX_HIP_CHECK(hipEventRecord(e1, nullptr));
    R4DX_HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0;
    R4DX_HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    t.push_back(ms * 1000.0);
  }
  R4DX_HIP_CHECK(hipEventDestroy(e0));
  R4DX_HIP_CHECK(hipEventDestroy(e1));
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

std::vector<int> ParseInts(const std::string& s) {
  std::vector<int> v;
  size_t p = 0;
  while (p < s.size()) {
    const size_t c = s.find(',', p);
    v.push_back(std::atoi(s.substr(p, c == std::string::npos ? c : c - p).c_str()));
    if (c == std::string::npos) break;
    p = c + 1;
  }
  return v;
}
}  // namespace

int Run(int argc, char** argv) {
  std::vector<int> ctxs = {2048, 8192, 32768, 131072};
  int reps = 30;
  bool with_ref = true;
  int decode_splits = 0;  // 0 = the unit's split law
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--ctx" && i + 1 < argc) ctxs = ParseInts(argv[++i]);
    else if (a == "--reps" && i + 1 < argc) reps = std::atoi(argv[++i]);
    else if (a == "--no-ref") with_ref = false;
    else if (a == "--splits" && i + 1 < argc) decode_splits = std::atoi(argv[++i]);
    else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
  }
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(1);
  std::normal_distribution<float> nd(0.0f, 1.0f);

  std::printf("%8s | %-22s | %10s %9s | %10s %9s | %6s\n", "ctx", "case", "ref us", "ref GB/s", "r4d us", "r4d GB/s",
              "speedup");
  for (const int ctx : ctxs) {
    const int blocks = (ctx + kBs - 1) / kBs;
    const int64_t head_stride = static_cast<int64_t>(kBs) * 2 * kD;
    const int64_t block_stride = kKVH * head_stride;
    const size_t elems = static_cast<size_t>(blocks) * block_stride;
    DeviceBuffer<uint16_t> kv(elems);
    {
      std::vector<uint16_t> h(elems);
      for (auto& v : h) v = FloatToBf16(nd(rng));
      kv.CopyFromHost(h);
    }
    std::vector<int32_t> tbl(blocks + 4);
    for (size_t i = 0; i < tbl.size(); ++i) tbl[i] = std::min<int>(static_cast<int>(i), blocks - 1);
    DeviceBuffer<int32_t> table(tbl.size()), used(1);
    table.CopyFromHost(tbl);
    used.CopyFromHost(std::vector<int32_t>{ctx});

    struct Case { const char* name; int q_len; int mode; };  // mode 0 decode, 1 prefill law
    for (const Case c : {Case{"decode q_len 1", 1, 0}, Case{"prefill chunk 64 rows", 64, 1}}) {
      std::vector<uint16_t> qh(static_cast<size_t>(c.q_len) * kQH * kD);
      for (auto& v : qh) v = FloatToBf16(nd(rng));
      DeviceBuffer<uint16_t> q(qh.size()), out(qh.size());
      q.CopyFromHost(qh);

      R4dxGemmaAttnRefArgs ra{};
      ra.q = P(q.data()); ra.kv = P(kv.data()); ra.block_table = P(table.data()); ra.out = P(out.data());
      ra.q_len = c.q_len; ra.q_heads = kQH; ra.kv_heads = kKVH; ra.head_dim = kD; ra.block_size = kBs;
      ra.ctx = ctx; ra.window = 0; ra.scale = 0.125f; ra.kv_block_stride = block_stride; ra.kv_head_stride = head_stride;
      const double ref_us = with_ref ? MedianUs([&] { r4dx_gemma_attn_ref_bf16kv(&ra, 0); }, reps) : 0.0;

      R4DArgs a{};
      a.q = q.data(); a.kv = kv.data(); a.block_table = table.data(); a.seqused_k = used.data(); a.out = out.data();
      a.num_seqs = 1; a.q_len = c.q_len; a.q_heads = kQH; a.kv_heads = kKVH; a.head_dim = kD; a.block_size = kBs;
      a.max_blocks = blocks + 4; a.kv_block_stride = block_stride; a.kv_head_stride = head_stride;
      a.scale = 0.125f; a.max_ctx = ctx;
      if (c.mode == 0) a.splits = decode_splits;
      DeviceBuffer<uint8_t> scratch;
      double us;
      if (c.mode == 0) {
        scratch.Resize(static_cast<size_t>(r4d_attn_decode_h64_gqa4_scratch_bytes(&a)));
        a.scratch = scratch.data();
        us = MedianUs([&] { r4d_attn_decode_h64_gqa4_bf16kv(&a, nullptr); }, reps);
      } else {
        a.splits = r4dx::model::attention::Lfm2PrefillSplits(ctx, c.q_len, kKVH);
        if (a.splits > 1) {
          scratch.Resize(static_cast<size_t>(r4d_attn_prefill_splitkv_h64_gqa4_scratch_bytes(&a)));
          a.scratch = scratch.data();
          us = MedianUs([&] { r4d_attn_prefill_splitkv_h64_gqa4_bf16kv(&a, nullptr); }, reps);
        } else {
          us = MedianUs([&] { r4d_attn_prefill_h64_gqa4_bf16kv(&a, nullptr); }, reps);
        }
      }
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      const double kv_bytes = static_cast<double>(ctx) * kKVH * 2 * kD * 2;  // the cache's K and V, once
      if (with_ref) {
        std::printf("%8d | %-22s | %10.1f %9.1f | %10.1f %9.1f | %5.2fx%s\n", ctx, c.name, ref_us, kv_bytes / ref_us / 1e3, us,
                    kv_bytes / us / 1e3, ref_us / us, c.mode == 1 ? (a.splits > 1 ? "  (split)" : "  (plain)") : "");
      } else {
        std::printf("%8d | %-22s | %10s %9s | %10.1f %9.1f | %6s%s\n", ctx, c.name, "-", "-", us, kv_bytes / us / 1e3, "-",
                    c.mode == 1 ? (a.splits > 1 ? "  (split)" : "  (plain)") : "");
      }
    }
  }
  return 0;
}

int main(int argc, char** argv) {
  try {
    return Run(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "EXCEPTION: %s\n", e.what());
    return 1;
  }
}
