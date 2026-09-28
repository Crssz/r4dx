// tests/kernels/tool_attn_prefill_bench.cpp -- prefill M0 (tools/prefill, D:\models\r4dx\prefill-m0):
// what the production prefill attention call (r4d_attn_prefill_h256_gqa6_fp8kv, the kernel every
// full-attention layer runs once per 64-row prefill chunk) costs at context depth D, and what it
// WOULD cost with more parallelism, without changing any kernel.
//
// Three shapes per depth D, all on a synthetic paged fp8 cache (no NaN codes) with the model's
// geometry (head_dim 256, 4 KV heads, GQA 6, block 16):
//   chunk   q_len = Q rows appended at depth D (ctx = D + Q), Q in --qlens (default 64 .. 1024).
//           Q = 64 is today's call (max_chunk_ = 64: grid ceil(Q/64) x 4 x 1 = 4 workgroups).
//           Q > 64 is one call for a bigger chunk; "ms per 64 rows" is what the same rows cost if
//           prefill ran Q-row chunks instead (the q-tile loop and causal tail are the real ones).
//   split   q_len = 64 at depth D, the KV range cut into S segments emulated as S sequences of
//           ctx D/S + 64 each (grid 1 x 4 x S): the parallel partial pass of a split-KV prefill
//           (each segment writes normalized bf16 rather than f16 partials + LSE, same size class).
//           The LSE merge is not included: it is one small pass over 64 x 24 x S x ~520 B.
//   splitkv prefill M1 (docs/prefill.md): the REAL split-KV prefill, r4d_attn_prefill_splitkv_*,
//           q_len 64 at depth D cut into S segments (--splitkv, default 2..32), fp32 partials and
//           the merge included. --kv-heads 2 is one TP=2 rank's shape (default 4).
// Timing: hipEvents around --reps back-to-back launches after --warmup, reported as the mean per
// call. Output: one line per shape on stdout and, with --out, a JSON file.
//
// Built, never add_test()'d (a measurement, not a contract):
//   $env:HIP_VISIBLE_DEVICES='1'; build\win-hip\tests\kernels\tool_attn_prefill_bench.exe `
//       --depths 0,8192,32768,65536,122880 --out attn_bench.json
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "r4d.h"
#include "r4dx/core/error.hpp"

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

struct Result {
  std::string kind;
  int depth, q_len, splits, grid;
  double ms;
};

}  // namespace

int main(int argc, char** argv) {
  std::vector<int> depths = {0, 8192, 32768, 65536, 122880};
  std::vector<int> qlens = {64, 128, 256, 512, 1024};
  std::vector<int> splits = {2, 4, 8, 16};
  std::vector<int> splitkv = {2, 4, 8, 16, 32};
  int reps = 20, warmup = 3, kv_heads_arg = 4;
  const char* out_path = nullptr;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    const char* v = i + 1 < argc ? argv[i + 1] : nullptr;
    if (a == "--depths" && v) { depths = ParseList(v); ++i; }
    else if (a == "--qlens" && v) { qlens = ParseList(v); ++i; }
    else if (a == "--splits" && v) { splits = ParseList(v); ++i; }
    else if (a == "--splitkv" && v) { splitkv = ParseList(v); ++i; }
    else if (a == "--reps" && v) { reps = std::atoi(v); ++i; }
    else if (a == "--kv-heads" && v) { kv_heads_arg = std::atoi(v); ++i; }  // 2 = one TP=2 rank
    else if (a == "--warmup" && v) { warmup = std::atoi(v); ++i; }
    else if (a == "--out" && v) { out_path = v; ++i; }
    else {
      std::fprintf(stderr,
                   "usage: tool_attn_prefill_bench [--depths a,b] [--qlens a,b] [--splits a,b] "
                   "[--splitkv a,b] [--kv-heads N] [--reps N] [--warmup N] [--out f.json]\n");
      return 2;
    }
  }
  R4DX_HIP_CHECK(hipSetDevice(0));
  int head_dim = 0, gqa = 0, block_size = 0, max_decode_rows = 0;
  r4d_attn_dims(&head_dim, &gqa, &block_size, &max_decode_rows);
  const int kv_heads = kv_heads_arg, q_heads = kv_heads * gqa;
  const int max_q = std::max(64, *std::max_element(qlens.begin(), qlens.end()));
  const int max_depth = *std::max_element(depths.begin(), depths.end());
  const int max_splits = splits.empty() ? 1 : *std::max_element(splits.begin(), splits.end());
  const int64_t max_ctx = static_cast<int64_t>(max_depth) + max_q;
  const int max_blocks = static_cast<int>((max_ctx + block_size - 1) / block_size);
  const int64_t kv_head_stride = static_cast<int64_t>(block_size) * 2 * head_dim;  // elements
  const int64_t kv_block_stride = kv_head_stride * kv_heads;
  const size_t kv_bytes = static_cast<size_t>(max_blocks) * kv_block_stride;  // fp8: 1 B/elt

  hipDeviceProp_t prop{};
  R4DX_HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  std::printf("[attn-bench] device %s, %d CUs; head_dim %d gqa %d block %d kv_heads %d; KV cache "
              "%.1f MiB (fp8, %d blocks)\n",
              prop.name, prop.multiProcessorCount, head_dim, gqa, block_size, kv_heads,
              kv_bytes / 1048576.0, max_blocks);

  // fp8 e4m3 with the top exponent bit cleared: |x| < 2, and never the NaN code S.1111.111.
  std::vector<uint8_t> kv_h(kv_bytes);
  std::mt19937 rng(7);
  for (size_t i = 0; i < kv_bytes; i += 4) {
    const uint32_t r = rng();
    for (size_t j = 0; j < 4 && i + j < kv_bytes; ++j) kv_h[i + j] = ((r >> (8 * j)) & 0xFF) & 0xBF;
  }
  const int q_rows_max = std::max(max_q, 64 * max_splits);
  std::vector<uint16_t> q_h(static_cast<size_t>(q_rows_max) * q_heads * head_dim);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& x : q_h) {
    const float f = dist(rng);
    uint32_t u;
    std::memcpy(&u, &f, 4);
    x = static_cast<uint16_t>(u >> 16);
  }
  // Identity block table, one row per emulated sequence (all rows identical).
  std::vector<int> bt_h(static_cast<size_t>(max_splits) * max_blocks);
  for (int s = 0; s < max_splits; ++s)
    for (int b = 0; b < max_blocks; ++b) bt_h[static_cast<size_t>(s) * max_blocks + b] = b;
  std::vector<float> desc_h(static_cast<size_t>(max_splits) * kv_heads, 1.0f);

  void *kv_d, *q_d, *out_d;
  int *bt_d, *seq_d;
  float* desc_d;
  R4DX_HIP_CHECK(hipMalloc(&kv_d, kv_bytes));
  R4DX_HIP_CHECK(hipMalloc(&q_d, q_h.size() * 2));
  R4DX_HIP_CHECK(hipMalloc(&out_d, q_h.size() * 2));
  R4DX_HIP_CHECK(hipMalloc(&bt_d, bt_h.size() * sizeof(int)));
  R4DX_HIP_CHECK(hipMalloc(&seq_d, static_cast<size_t>(max_splits) * sizeof(int)));
  R4DX_HIP_CHECK(hipMalloc(&desc_d, desc_h.size() * sizeof(float)));
  R4DX_HIP_CHECK(hipMemcpy(kv_d, kv_h.data(), kv_bytes, hipMemcpyHostToDevice));
  R4DX_HIP_CHECK(hipMemcpy(q_d, q_h.data(), q_h.size() * 2, hipMemcpyHostToDevice));
  R4DX_HIP_CHECK(hipMemcpy(bt_d, bt_h.data(), bt_h.size() * sizeof(int), hipMemcpyHostToDevice));
  R4DX_HIP_CHECK(hipMemcpy(desc_d, desc_h.data(), desc_h.size() * sizeof(float), hipMemcpyHostToDevice));
  hipStream_t s;
  R4DX_HIP_CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
  hipEvent_t e0, e1;
  R4DX_HIP_CHECK(hipEventCreate(&e0));
  R4DX_HIP_CHECK(hipEventCreate(&e1));

  // splitkv > 0: the split-KV entry with that many segments (scratch_d holds its partials).
  const int max_splitkv = splitkv.empty() ? 1 : *std::max_element(splitkv.begin(), splitkv.end());
  void* scratch_d = nullptr;
  {
    const size_t scratch_bytes = static_cast<size_t>(64) * q_heads * std::min(64, std::max(1, max_splitkv)) *
                                 (head_dim + 2) * 4;
    R4DX_HIP_CHECK(hipMalloc(&scratch_d, scratch_bytes));
  }
  auto time_call = [&](int num_seqs, int q_len, int ctx, int splitkv_n = 0) -> double {
    std::vector<int> seq(static_cast<size_t>(num_seqs), ctx);
    R4DX_HIP_CHECK(hipMemcpy(seq_d, seq.data(), seq.size() * sizeof(int), hipMemcpyHostToDevice));
    R4DArgs a{};
    a.q = q_d;
    a.kv = kv_d;
    a.block_table = bt_d;
    a.seqused_k = seq_d;
    a.out = out_d;
    a.k_descale = desc_d;
    a.v_descale = desc_d;
    a.num_seqs = num_seqs;
    a.q_len = q_len;
    a.q_heads = q_heads;
    a.kv_heads = kv_heads;
    a.head_dim = head_dim;
    a.block_size = block_size;
    a.max_blocks = max_blocks;
    a.kv_block_stride = kv_block_stride;
    a.kv_head_stride = kv_head_stride;
    a.scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    a.max_ctx = ctx;
    a.splits = splitkv_n;
    a.scratch = splitkv_n > 0 ? scratch_d : nullptr;
    auto launch = [&]() {
      return splitkv_n > 0 ? r4d_attn_prefill_splitkv_h256_gqa6_fp8kv(&a, s)
                           : r4d_attn_prefill_h256_gqa6_fp8kv(&a, s);
    };
    for (int i = 0; i < warmup; ++i) {
      if (launch() != 0) {
        std::fprintf(stderr, "[attn-bench] launch rejected\n");
        std::exit(1);
      }
    }
    R4DX_HIP_CHECK(hipEventRecord(e0, s));
    for (int i = 0; i < reps; ++i) launch();
    R4DX_HIP_CHECK(hipEventRecord(e1, s));
    R4DX_HIP_CHECK(hipEventSynchronize(e1));
    float ms = 0.0f;
    R4DX_HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
    return static_cast<double>(ms) / reps;
  };

  std::vector<Result> results;
  for (int d : depths) {
    double base = 0.0;
    for (int q : qlens) {
      const double ms = time_call(1, q, d + q);
      const int grid = ((q + 63) / 64) * kv_heads;
      if (q == 64) base = ms;
      results.push_back({"chunk", d, q, 1, grid, ms});
      std::printf("[attn-bench] depth %6d chunk q_len %5d grid %4d: %9.4f ms/call  %9.4f ms per 64 rows"
                  "%s\n",
                  d, q, grid, ms, ms * 64.0 / q,
                  (q != 64 && base > 0) ? (" (x" + std::to_string(base / (ms * 64.0 / q)).substr(0, 5) +
                                           " vs q_len 64)").c_str()
                                        : "");
    }
    // prefill M1: the real split-KV prefill (r4d_attn_prefill_splitkv_*: one sequence, the KV range
    // cut into S segments, fp32 partials AND the merge), q_len 64 at depth D.
    for (int sp : splitkv) {
      const double ms = time_call(1, 64, d + 64, sp);
      results.push_back({"splitkv", d, 64, sp, kv_heads * sp, ms});
      std::printf("[attn-bench] depth %6d splitkv %2d (q_len 64) grid %4d: %9.4f ms/call incl. merge%s\n",
                  d, sp, kv_heads * sp, ms,
                  base > 0 ? (" (x" + std::to_string(base / ms).substr(0, 5) + " vs unsplit)").c_str()
                           : "");
    }
    for (int sp : splits) {
      if (d / sp < 64) continue;
      const double ms = time_call(sp, 64, d / sp + 64);
      results.push_back({"split", d, 64, sp, kv_heads * sp, ms});
      std::printf("[attn-bench] depth %6d split %2d (q_len 64) grid %4d: %9.4f ms/call partial pass%s\n",
                  d, sp, kv_heads * sp, ms,
                  base > 0 ? (" (x" + std::to_string(base / ms).substr(0, 5) + " vs unsplit)").c_str()
                           : "");
    }
  }
  if (out_path != nullptr) {
    FILE* f = std::fopen(out_path, "w");
    if (f == nullptr) {
      std::fprintf(stderr, "[attn-bench] cannot write %s\n", out_path);
      return 1;
    }
    std::fprintf(f, "{\"device\": \"%s\", \"cus\": %d, \"reps\": %d, \"results\": [", prop.name,
                 prop.multiProcessorCount, reps);
    for (size_t i = 0; i < results.size(); ++i) {
      const Result& r = results[i];
      std::fprintf(f,
                   "%s\n {\"kind\": \"%s\", \"depth\": %d, \"q_len\": %d, \"splits\": %d, "
                   "\"grid\": %d, \"ms\": %.6f}",
                   i ? "," : "", r.kind.c_str(), r.depth, r.q_len, r.splits, r.grid, r.ms);
    }
    std::fprintf(f, "\n]}\n");
    std::fclose(f);
    std::printf("[attn-bench] wrote %s\n", out_path);
  }
  return 0;
}
