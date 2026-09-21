// tests/kernels/tool_sampler_bench.cpp -- Milestone 6 S1 item 4 (docs/sampling.md "Cost").
//
// Three numbers, one process, HIP device 1 only:
//   1. r4dx_topk_lse_f32 kernel time at V = 248320 for rows in {1, 8} (hip events, 200 iterations
//      after a warm-up), plus the D2H time for the summary it produces versus the D2H time for the
//      full fp32 logits it replaces;
//   2. the host cost of SampleFromSummary per token;
//   3. the host cost of the full-vocab path per token -- both the canonical rewrite
//      (SampleCanonical) and the pre-M6 SampleLegacy, for every filter combination the server
//      actually sends.
//
// Prints numbers for a human/doc to read; asserts no pass/fail contract, hence "tool_", never
// registered with add_test (see tests/kernels/CMakeLists.txt).
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/kernels/summary_sampler.hpp"

using namespace r4dx::core;
using namespace r4dx::kernels;

namespace {

constexpr int64_t kVocab = 248320;
constexpr int kK = R4DX_TOPK_LSE_K;
constexpr int kIters = 200;

struct Combo {
  const char* name;
  SampleParams p;
};

// A Zipf-like row with a realistic peak, the shape a real decode step produces.
std::vector<float> MakeRealisticRow(int rows, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> jitter(-0.75f, 0.75f);
  std::vector<float> out(static_cast<size_t>(rows) * kVocab);
  for (int r = 0; r < rows; ++r) {
    for (int64_t i = 0; i < kVocab; ++i) {
      out[static_cast<size_t>(r) * kVocab + i] =
          static_cast<float>(12.0 - 2.2 * std::log(static_cast<double>(i + 1))) + jitter(rng);
    }
  }
  return out;
}

// Same kernel at a smaller vocabulary but the SAME chunk count (the wrapper caps chunks at 64 for
// any vocab >= 64*1024), which isolates the phase-1 memory pass from the fixed phase-2 merge.
double TimeKernelAtVocabMs(int rows, int64_t vocab, const std::vector<float>& logits) {
  DeviceBuffer<float> l_d(static_cast<size_t>(rows) * vocab);
  DeviceBuffer<int32_t> ids_d(static_cast<size_t>(rows) * kK);
  DeviceBuffer<float> vals_d(static_cast<size_t>(rows) * kK);
  DeviceBuffer<float> lse_d(static_cast<size_t>(rows));
  R4DX_HIP_CHECK(hipMemcpy(l_d.data(), logits.data(),
                           static_cast<size_t>(rows) * vocab * sizeof(float),
                           hipMemcpyHostToDevice));
  const int64_t l = reinterpret_cast<int64_t>(l_d.data());
  const int64_t i = reinterpret_cast<int64_t>(ids_d.data());
  const int64_t v = reinterpret_cast<int64_t>(vals_d.data());
  const int64_t s = reinterpret_cast<int64_t>(lse_d.data());
  for (int w = 0; w < 20; ++w) r4dx_topk_lse_f32(l, i, v, s, rows, vocab, 1.0f / 0.7f, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  hipEvent_t beg, end;
  R4DX_HIP_CHECK(hipEventCreate(&beg));
  R4DX_HIP_CHECK(hipEventCreate(&end));
  R4DX_HIP_CHECK(hipEventRecord(beg, nullptr));
  for (int it = 0; it < kIters; ++it) r4dx_topk_lse_f32(l, i, v, s, rows, vocab, 1.0f / 0.7f, 0);
  R4DX_HIP_CHECK(hipEventRecord(end, nullptr));
  R4DX_HIP_CHECK(hipEventSynchronize(end));
  float ms = 0.0f;
  R4DX_HIP_CHECK(hipEventElapsedTime(&ms, beg, end));
  R4DX_HIP_CHECK(hipEventDestroy(beg));
  R4DX_HIP_CHECK(hipEventDestroy(end));
  return static_cast<double>(ms) / kIters;
}

double TimeKernelMs(int rows, const std::vector<float>& logits) {
  DeviceBuffer<float> l_d(logits.size());
  DeviceBuffer<int32_t> ids_d(static_cast<size_t>(rows) * kK);
  DeviceBuffer<float> vals_d(static_cast<size_t>(rows) * kK);
  DeviceBuffer<float> lse_d(static_cast<size_t>(rows));
  l_d.CopyFromHost(logits);
  const int64_t l = reinterpret_cast<int64_t>(l_d.data());
  const int64_t i = reinterpret_cast<int64_t>(ids_d.data());
  const int64_t v = reinterpret_cast<int64_t>(vals_d.data());
  const int64_t s = reinterpret_cast<int64_t>(lse_d.data());

  for (int w = 0; w < 20; ++w) r4dx_topk_lse_f32(l, i, v, s, rows, kVocab, 1.0f / 0.7f, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());

  hipEvent_t beg, end;
  R4DX_HIP_CHECK(hipEventCreate(&beg));
  R4DX_HIP_CHECK(hipEventCreate(&end));
  R4DX_HIP_CHECK(hipEventRecord(beg, nullptr));
  for (int it = 0; it < kIters; ++it) r4dx_topk_lse_f32(l, i, v, s, rows, kVocab, 1.0f / 0.7f, 0);
  R4DX_HIP_CHECK(hipEventRecord(end, nullptr));
  R4DX_HIP_CHECK(hipEventSynchronize(end));
  float ms = 0.0f;
  R4DX_HIP_CHECK(hipEventElapsedTime(&ms, beg, end));
  R4DX_HIP_CHECK(hipEventDestroy(beg));
  R4DX_HIP_CHECK(hipEventDestroy(end));
  return static_cast<double>(ms) / kIters;
}

// Host-side D2H cost of the summary (rows * (K*8 + 4) bytes) versus the full fp32 logits it
// replaces (rows * vocab * 4 bytes), measured the way the engine would pay it: a blocking copy.
void TimeD2H(int rows, const std::vector<float>& logits) {
  DeviceBuffer<float> l_d(logits.size());
  DeviceBuffer<int32_t> ids_d(static_cast<size_t>(rows) * kK);
  DeviceBuffer<float> vals_d(static_cast<size_t>(rows) * kK);
  DeviceBuffer<float> lse_d(static_cast<size_t>(rows));
  l_d.CopyFromHost(logits);
  r4dx_topk_lse_f32(reinterpret_cast<int64_t>(l_d.data()), reinterpret_cast<int64_t>(ids_d.data()),
                     reinterpret_cast<int64_t>(vals_d.data()),
                     reinterpret_cast<int64_t>(lse_d.data()), rows, kVocab, 1.0f / 0.7f, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> host_full(logits.size());
  std::vector<int32_t> host_ids(static_cast<size_t>(rows) * kK);
  std::vector<float> host_vals(static_cast<size_t>(rows) * kK);
  std::vector<float> host_lse(static_cast<size_t>(rows));

  auto bench = [&](const char* what, auto&& fn) {
    for (int w = 0; w < 5; ++w) fn();
    const auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < kIters; ++it) fn();
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / kIters;
    std::printf("    %-34s %8.4f ms\n", what, ms);
  };

  bench("D2H full fp32 logits", [&] {
    R4DX_HIP_CHECK(hipMemcpy(host_full.data(), l_d.data(), logits.size() * sizeof(float),
                             hipMemcpyDeviceToHost));
  });
  bench("D2H row summary (ids+vals+lse)", [&] {
    R4DX_HIP_CHECK(hipMemcpy(host_ids.data(), ids_d.data(), host_ids.size() * sizeof(int32_t),
                             hipMemcpyDeviceToHost));
    R4DX_HIP_CHECK(hipMemcpy(host_vals.data(), vals_d.data(), host_vals.size() * sizeof(float),
                             hipMemcpyDeviceToHost));
    R4DX_HIP_CHECK(hipMemcpy(host_lse.data(), lse_d.data(), host_lse.size() * sizeof(float),
                             hipMemcpyDeviceToHost));
  });
}

RowSummary FetchSummary(const std::vector<float>& logits, float inv_t) {
  DeviceBuffer<float> l_d(logits.size());
  DeviceBuffer<int32_t> ids_d(kK);
  DeviceBuffer<float> vals_d(kK);
  DeviceBuffer<float> lse_d(1);
  l_d.CopyFromHost(logits);
  r4dx_topk_lse_f32(reinterpret_cast<int64_t>(l_d.data()), reinterpret_cast<int64_t>(ids_d.data()),
                     reinterpret_cast<int64_t>(vals_d.data()),
                     reinterpret_cast<int64_t>(lse_d.data()), 1, kVocab, inv_t, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const std::vector<int32_t> ids = ids_d.CopyToHost();
  const std::vector<float> vals = vals_d.CopyToHost();
  const std::vector<float> lse = lse_d.CopyToHost();
  RowSummary s;
  s.k = kK;
  s.vocab = kVocab;
  s.inv_temperature = inv_t;
  s.lse = lse[0];
  for (int j = 0; j < kK; ++j) {
    s.ids[j] = ids[j];
    s.vals[j] = vals[j];
  }
  return s;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));

  std::printf("tool_sampler_bench -- V=%lld, K=%d, %d iterations after warm-up\n\n",
              static_cast<long long>(kVocab), kK, kIters);

  for (int rows : {1, 8}) {
    const std::vector<float> logits = MakeRealisticRow(rows, 7u);
    const double a = TimeKernelMs(rows, logits);
    const double b = TimeKernelMs(rows, logits);
    std::printf("[kernel] r4dx_topk_lse_f32 rows=%d : %.4f ms / %.4f ms (run 1 / run 2)\n", rows, a,
                b);
    // Reference points on the same data: the one-pass top-16 this kernel's extraction rounds are
    // built from, and the single-int32 device argmax the greedy path already uses.
    {
      DeviceBuffer<float> l_d(logits.size());
      DeviceBuffer<int32_t> ids_d(static_cast<size_t>(rows) * 16);
      DeviceBuffer<float> vals_d(static_cast<size_t>(rows) * 16);
      l_d.CopyFromHost(logits);
      const int64_t l = reinterpret_cast<int64_t>(l_d.data());
      const int64_t i = reinterpret_cast<int64_t>(ids_d.data());
      const int64_t v = reinterpret_cast<int64_t>(vals_d.data());
      auto ev = [&](auto&& fn) {
        for (int w = 0; w < 20; ++w) fn();
        R4DX_HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t beg, end;
        R4DX_HIP_CHECK(hipEventCreate(&beg));
        R4DX_HIP_CHECK(hipEventCreate(&end));
        R4DX_HIP_CHECK(hipEventRecord(beg, nullptr));
        for (int it = 0; it < kIters; ++it) fn();
        R4DX_HIP_CHECK(hipEventRecord(end, nullptr));
        R4DX_HIP_CHECK(hipEventSynchronize(end));
        float ms = 0.0f;
        R4DX_HIP_CHECK(hipEventElapsedTime(&ms, beg, end));
        R4DX_HIP_CHECK(hipEventDestroy(beg));
        R4DX_HIP_CHECK(hipEventDestroy(end));
        return static_cast<double>(ms) / kIters;
      };
      const double t16 = ev([&] { r4dx_topk16_f32(l, i, v, rows, kVocab, 0); });
      const double targmax = ev([&] { r4dx_argmax_f32(l, i, kVocab, 0); });
      // Launch-overhead floor on this toolchain: a one-thread kernel that touches 4 bytes.
      const double tnull = ev([&] { r4dx_gather_i32(i, i, i, 1, 16, 0); });
      std::printf("    (reference) r4dx_topk16_f32 %.4f ms, r4dx_argmax_f32 %.4f ms, "
                  "one-thread launch %.4f ms\n",
                  t16, targmax, tnull);
    }
    TimeD2H(rows, logits);
    std::printf("\n");
  }

  // ---- phase split: same chunk count (64), shrinking phase-1 memory --------------------------
  {
    const std::vector<float> big = MakeRealisticRow(1, 7u);
    std::printf("[kernel] rows=1, chunks pinned at 64 -- isolates the fixed merge from the pass\n");
    for (int64_t vocab : {int64_t(1024), int64_t(4096), int64_t(16384), int64_t(65536),
                          int64_t(131072), int64_t(248320)}) {
      int chunks = static_cast<int>(vocab / 1024);
      if (chunks < 1) chunks = 1;
      if (chunks > 64) chunks = 64;
      const double a = TimeKernelAtVocabMs(1, vocab, big);
      const double b = TimeKernelAtVocabMs(1, vocab, big);
      // Same vocabulary through the one-block one-pass kernels, as a launch/floor reference.
      DeviceBuffer<float> l_d(static_cast<size_t>(vocab));
      DeviceBuffer<int32_t> ids_d(16);
      DeviceBuffer<float> vals_d(16);
      R4DX_HIP_CHECK(hipMemcpy(l_d.data(), big.data(), static_cast<size_t>(vocab) * sizeof(float),
                               hipMemcpyHostToDevice));
      const int64_t l = reinterpret_cast<int64_t>(l_d.data());
      const int64_t ii = reinterpret_cast<int64_t>(ids_d.data());
      const int64_t vv = reinterpret_cast<int64_t>(vals_d.data());
      auto ev = [&](auto&& fn) {
        for (int w = 0; w < 20; ++w) fn();
        R4DX_HIP_CHECK(hipDeviceSynchronize());
        hipEvent_t bg, en;
        R4DX_HIP_CHECK(hipEventCreate(&bg));
        R4DX_HIP_CHECK(hipEventCreate(&en));
        R4DX_HIP_CHECK(hipEventRecord(bg, nullptr));
        for (int it = 0; it < kIters; ++it) fn();
        R4DX_HIP_CHECK(hipEventRecord(en, nullptr));
        R4DX_HIP_CHECK(hipEventSynchronize(en));
        float ms = 0.0f;
        R4DX_HIP_CHECK(hipEventElapsedTime(&ms, bg, en));
        R4DX_HIP_CHECK(hipEventDestroy(bg));
        R4DX_HIP_CHECK(hipEventDestroy(en));
        return static_cast<double>(ms) / kIters;
      };
      const double t16 = ev([&] { r4dx_topk16_f32(l, ii, vv, 1, vocab, 0); });
      const double tam = ev([&] { r4dx_argmax_f32(l, ii, vocab, 0); });
      std::printf("    V=%-7lld chunks=%-3d %.4f ms / %.4f ms   (topk16 %.4f, argmax %.4f)\n",
                  static_cast<long long>(vocab), chunks, a, b, t16, tam);
    }
    std::printf("\n");
  }

  // ---- host sampler cost, one row --------------------------------------------------------------
  const std::vector<float> row = MakeRealisticRow(1, 7u);
  const Combo combos[] = {
      {"T=1.0 (pure temperature)", [] { SampleParams p; p.temperature = 1.0f; return p; }()},
      {"T=0.7 top_k=20 top_p=0.8",
       [] { SampleParams p; p.temperature = 0.7f; p.top_k = 20; p.top_p = 0.8f; return p; }()},
      {"T=0.6 top_k=20 top_p=0.95",
       [] { SampleParams p; p.temperature = 0.6f; p.top_k = 20; p.top_p = 0.95f; return p; }()},
      {"T=1.0 top_p=0.9",
       [] { SampleParams p; p.temperature = 1.0f; p.top_p = 0.9f; return p; }()},
      {"T=0.8 min_p=0.05",
       [] { SampleParams p; p.temperature = 0.8f; p.min_p = 0.05f; return p; }()},
      {"T=0.7 top_k=50 min_p=0.02 top_p=0.9",
       [] {
         SampleParams p;
         p.temperature = 0.7f;
         p.top_k = 50;
         p.min_p = 0.02f;
         p.top_p = 0.9f;
         return p;
       }()},
  };

  std::printf("[host] per-token sampling cost on a realistic peaked row (V=%lld)\n",
              static_cast<long long>(kVocab));
  std::printf("    %-38s %12s %12s %12s %10s\n", "filters", "summary(ms)", "canonical(ms)",
              "legacy(ms)", "fallback");
  for (const Combo& c : combos) {
    const RowSummary summary = FetchSummary(row, 1.0f / c.p.temperature);

    const int n_summary = 20000;
    int fallbacks = 0;
    volatile int32_t sink = 0;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_summary; ++i) {
      const double u = (static_cast<double>(i) + 0.5) / n_summary;
      const SummarySampleResult r = SampleFromSummary(summary, c.p, u);
      if (!r.resolved) ++fallbacks;
      sink = r.token;
    }
    auto t1 = std::chrono::steady_clock::now();
    const double ms_summary =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / n_summary;

    const int n_full = 200;
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_full; ++i) {
      sink = SampleCanonical(row.data(), kVocab, c.p, (static_cast<double>(i) + 0.5) / n_full);
    }
    t1 = std::chrono::steady_clock::now();
    const double ms_canon = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_full;

    std::mt19937_64 rng(1234);
    t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_full; ++i) sink = SampleLegacy(row.data(), kVocab, c.p, rng);
    t1 = std::chrono::steady_clock::now();
    const double ms_legacy = std::chrono::duration<double, std::milli>(t1 - t0).count() / n_full;
    (void)sink;

    std::printf("    %-38s %12.5f %12.4f %12.4f %9.2f%%\n", c.name, ms_summary, ms_canon, ms_legacy,
                100.0 * fallbacks / n_summary);
  }
  std::printf("\ndone\n");
  return 0;
}
