// tests/kernels/test_topk16.cpp -- r4dx_topk16_f32 (docs/dflash2.md "Kernels") against a CPU
// reference (including a deliberately tie-heavy input, since the tie-break toward the LOWER id is
// part of the contract the DFlash2 selector depends on) and against fixture A's own
// `logits` -> `cand`/`unary`, where the match must be EXACT: the kernel only copies fp32 values it
// read, and the reference's `top_k_desc` sorts by the same total order.
#include <algorithm>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "npy_fixture.hpp"
#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;
using namespace r4dx_test;

namespace {

constexpr int kK = 16;

// The kernel's documented total order: higher value wins, ties break toward the LOWER id.
struct Ref {
  std::vector<int32_t> ids;
  std::vector<float> vals;
};

Ref TopK16Ref(const std::vector<float>& logits, int rows, int64_t vocab) {
  Ref out;
  out.ids.resize(static_cast<size_t>(rows) * kK);
  out.vals.resize(static_cast<size_t>(rows) * kK);
  std::vector<int32_t> order(static_cast<size_t>(vocab));
  for (int r = 0; r < rows; ++r) {
    const float* lr = logits.data() + static_cast<size_t>(r) * vocab;
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + kK, order.end(),
                      [lr](int32_t a, int32_t b) {
                        return lr[a] > lr[b] || (lr[a] == lr[b] && a < b);
                      });
    for (int j = 0; j < kK; ++j) {
      out.ids[static_cast<size_t>(r) * kK + j] = order[j];
      out.vals[static_cast<size_t>(r) * kK + j] = lr[order[j]];
    }
  }
  return out;
}

bool RunCase(const char* label, const std::vector<float>& logits, int rows, int64_t vocab) {
  const Ref ref = TopK16Ref(logits, rows, vocab);

  DeviceBuffer<float> l_d(logits.size());
  DeviceBuffer<int32_t> ids_d(static_cast<size_t>(rows) * kK);
  DeviceBuffer<float> vals_d(static_cast<size_t>(rows) * kK);
  l_d.CopyFromHost(logits);
  r4dx_topk16_f32(reinterpret_cast<int64_t>(l_d.data()), reinterpret_cast<int64_t>(ids_d.data()),
                   reinterpret_cast<int64_t>(vals_d.data()), rows, vocab, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());

  const std::vector<int32_t> ids = ids_d.CopyToHost();
  const std::vector<float> vals = vals_d.CopyToHost();
  int bad_ids = 0, bad_vals = 0, first_bad = -1;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] != ref.ids[i]) {
      if (first_bad < 0) first_bad = static_cast<int>(i);
      ++bad_ids;
    }
    if (vals[i] != ref.vals[i]) ++bad_vals;
  }
  const bool pass = bad_ids == 0 && bad_vals == 0;
  std::printf("  %-34s rows=%d V=%-7lld  id mismatches=%d  val mismatches=%d  %s\n", label, rows,
              (long long)vocab, bad_ids, bad_vals, pass ? "PASS" : "FAIL");
  if (!pass && first_bad >= 0) {
    std::printf("    first mismatch at [%d/%d]: got id=%d val=%.9g, want id=%d val=%.9g\n",
                first_bad / kK, first_bad % kK, ids[first_bad], vals[first_bad],
                ref.ids[first_bad], ref.vals[first_bad]);
  }
  return pass;
}

}  // namespace

int main() {
  // Unbuffered: a GPU test that dies (a kernel fault, or an abort out of a failed HIP check)
  // takes the CRT's stdout buffer with it, and a crash report with zero output says nothing about
  // which check was running. Costs nothing at this output volume.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  bool ok = true;
  std::mt19937 rng(11);

  // ---- 1. Random data, both the fixture vocab and the real drafter's vocab ----
  std::printf("[1] cpu-ref, random values\n");
  for (int64_t vocab : {int64_t(4096), int64_t(248320)}) {
    std::vector<float> logits(static_cast<size_t>(8) * vocab);
    std::uniform_real_distribution<float> d(-20.0f, 20.0f);
    for (auto& v : logits) v = d(rng);
    ok = RunCase("random", logits, 8, vocab) && ok;
  }

  // ---- 2. Tie-break: only 12 distinct values across the whole row, so the top 16 is decided
  //         almost entirely by the "lower id wins" rule rather than by value ordering. ----
  std::printf("[2] cpu-ref, heavily duplicated values (tie-break by lower id)\n");
  for (int64_t vocab : {int64_t(4096), int64_t(248320)}) {
    std::vector<float> logits(static_cast<size_t>(8) * vocab);
    std::uniform_int_distribution<int> d(0, 11);
    for (auto& v : logits) v = static_cast<float>(d(rng));
    ok = RunCase("12 distinct levels", logits, 8, vocab) && ok;
  }
  {
    // The degenerate extreme: every entry identical, so the answer is exactly ids 0..15.
    const int64_t vocab = 248320;
    std::vector<float> logits(static_cast<size_t>(8) * vocab, 1.25f);
    ok = RunCase("all values identical", logits, 8, vocab) && ok;
  }
  {
    // A row of -inf: the sentinel the kernel seeds its per-thread lists with is (-inf, INT32_MAX),
    // so this is the case where a naive sentinel would leak INT32_MAX into the output.
    const int64_t vocab = 4096;
    std::vector<float> logits(static_cast<size_t>(2) * vocab,
                              -std::numeric_limits<float>::infinity());
    ok = RunCase("all -inf", logits, 2, vocab) && ok;
  }

  // ---- 3. rows sweep 1..8 (the DFlash2 block never exceeds 8 rows) ----
  std::printf("[3] cpu-ref, rows sweep\n");
  for (int rows = 1; rows <= 8; ++rows) {
    const int64_t vocab = 4096;
    std::vector<float> logits(static_cast<size_t>(rows) * vocab);
    std::uniform_real_distribution<float> d(-5.0f, 5.0f);
    for (auto& v : logits) v = d(rng);
    char label[64];
    std::snprintf(label, sizeof(label), "rows=%d", rows);
    ok = RunCase(label, logits, rows, vocab) && ok;
  }

  // ---- 4. Fixture A: logits -> cand/unary, EXACT ----
  if (!FixtureAvailable()) {
    std::fprintf(stderr,
                 "[SKIP] fixture A not found at %s -- regenerate with\n"
                 "       <reference venv>/python.exe tools/reference/dflash2_ref.py "
                 "--gen-fixtures all --seed 0\n",
                 FixtureDir().c_str());
    return ok ? kSkipReturnCode : 1;
  }
  {
    const std::string d = FixtureDir();
    const std::vector<float> logits = LoadNpyF32(d + "/logits.npy", {8, 4096});
    const std::vector<int64_t> cand = LoadNpyI64(d + "/cand.npy", {8, 16});
    const std::vector<float> unary = LoadNpyF32(d + "/unary.npy", {8, 16});

    DeviceBuffer<float> l_d(logits.size());
    DeviceBuffer<int32_t> ids_d(8 * kK);
    DeviceBuffer<float> vals_d(8 * kK);
    l_d.CopyFromHost(logits);
    r4dx_topk16_f32(reinterpret_cast<int64_t>(l_d.data()), reinterpret_cast<int64_t>(ids_d.data()),
                     reinterpret_cast<int64_t>(vals_d.data()), 8, 4096, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const std::vector<int32_t> ids = ids_d.CopyToHost();
    const std::vector<float> vals = vals_d.CopyToHost();

    int bad = 0;
    for (size_t i = 0; i < ids.size(); ++i) {
      // EXACT on both: the fixture's logits are fp32 and the kernel reads them as fp32, so the
      // selected values are the same bits, not merely close.
      if (ids[i] != static_cast<int32_t>(cand[i]) || vals[i] != unary[i]) ++bad;
    }
    std::printf("[4] fixture A logits -> cand/unary: %d/%d mismatches  %s\n", bad,
                static_cast<int>(ids.size()), bad == 0 ? "PASS" : "FAIL");
    ok = ok && bad == 0;
  }

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
