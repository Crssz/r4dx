// tests/kernels/test_rep_penalty.cpp -- r4dx_apply_repetition_penalty_f32 (rep_penalty.h; docs/lfm2-plan.md 5.4)
// against the host rule in rep_penalty.hpp, on a vocab-sized fp32 row. GPU test (HIP device 1 via ctest's
// ENVIRONMENT): `ctest -R rep_penalty` also runs the pure-CPU twin test_rep_penalty_cpu.
//   * for several penalties (> 1, < 1) and seen-list sizes (1, around the 256-thread block edges, thousands,
//     a large fraction of the vocab): the row equals the host result BIT FOR BIT (the multiply is exact; the
//     divide is IEEE on both sides), and every id that is not listed is unchanged;
//   * negative, zero, -0, +-inf and NaN logits behave like the host rule;
//   * out-of-range ids (-1, vocab, INT_MAX) are skipped and write nothing (a guard canary after the row);
//   * penalty == 1 and n_seen == 0 leave the row alone (and launch nothing);
//   * a delta upload (two launches over the two halves of the id list) equals one launch over the whole list;
//   * the preconditions throw (penalty <= 0, non-finite, null pointers, vocab <= 0).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/rep_penalty.h"
#include "r4dx/kernels/rep_penalty.hpp"

using namespace r4dx::core;
using namespace r4dx::kernels;

namespace {

bool g_ok = true;
void Check(bool cond, const char* what) {
  std::printf("%-76s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}

bool SameBits(float a, float b) {
  if (std::isnan(a) && std::isnan(b)) return true;
  return std::memcmp(&a, &b, sizeof a) == 0;
}

constexpr int64_t kVocab = 128000;
constexpr int kCanary = 64;  // floats after the row; must stay untouched

std::vector<float> MakeRow(std::mt19937& rng) {
  std::normal_distribution<float> n(0.0f, 6.0f);
  std::vector<float> row(static_cast<size_t>(kVocab) + kCanary);
  for (int64_t i = 0; i < kVocab; ++i) row[static_cast<size_t>(i)] = n(rng);
  for (int i = 0; i < kCanary; ++i) row[static_cast<size_t>(kVocab) + i] = 123.0f + static_cast<float>(i);
  // the awkward values
  row[0] = 0.0f;
  row[1] = -0.0f;
  row[2] = std::numeric_limits<float>::infinity();
  row[3] = -std::numeric_limits<float>::infinity();
  row[4] = std::numeric_limits<float>::quiet_NaN();
  row[5] = 1.0e-30f;
  row[6] = -1.0e-30f;
  row[7] = 3.0e38f;
  return row;
}

// Runs the kernel on a copy of `row` with `ids`, returns the row read back.
std::vector<float> RunDevice(const std::vector<float>& row, const std::vector<int32_t>& ids, float p,
                             int64_t vocab = kVocab) {
  DeviceBuffer<float> d_row(row.size());
  d_row.CopyFromHost(row);
  DeviceBuffer<int32_t> d_ids(ids.empty() ? 1 : ids.size());
  if (!ids.empty()) d_ids.CopyFromHost(ids);
  r4dx_apply_repetition_penalty_f32(reinterpret_cast<int64_t>(d_row.data()),
                                    reinterpret_cast<int64_t>(d_ids.data()),
                                    static_cast<int64_t>(ids.size()), vocab, p, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  return d_row.CopyToHost();
}

std::vector<int32_t> DistinctIds(std::mt19937& rng, size_t n) {
  std::vector<int32_t> all(static_cast<size_t>(kVocab));
  std::iota(all.begin(), all.end(), 0);
  std::shuffle(all.begin(), all.end(), rng);
  all.resize(n);
  return all;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(20261014);
  const std::vector<float> row = MakeRow(rng);

  for (float p : {1.05f, 1.5f, 0.8f, 2.0f, 1.0000001f}) {
    for (size_t n : {size_t{1}, size_t{255}, size_t{256}, size_t{257}, size_t{5000}, size_t{100000}}) {
      std::vector<int32_t> ids = DistinctIds(rng, n);
      // make sure the awkward positions are in the list for the bigger lists
      if (n >= 5000) {
        for (int32_t k = 0; k < 8; ++k) {
          if (std::find(ids.begin(), ids.end(), k) == ids.end()) ids[static_cast<size_t>(k)] = k;
        }
        std::sort(ids.begin(), ids.end());
        ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
      }
      std::vector<float> want = row;
      ApplyRepetitionPenalty(want.data(), kVocab, ids.data(), ids.size(), p);
      const std::vector<float> got = RunDevice(row, ids, p);
      size_t bad = 0;
      for (size_t i = 0; i < got.size(); ++i) bad += SameBits(got[i], want[i]) ? 0 : 1;
      char msg[160];
      std::snprintf(msg, sizeof msg, "p %.7g, %zu ids: bit-identical to the host rule (%zu differing)", p, ids.size(), bad);
      Check(bad == 0, msg);
    }
  }

  // Unlisted ids unchanged, canary untouched, out-of-range ids skipped.
  {
    std::vector<int32_t> ids = {10, 20, 30, -1, static_cast<int32_t>(kVocab), std::numeric_limits<int32_t>::max(), 40};
    const std::vector<float> got = RunDevice(row, ids, 1.7f);
    size_t changed = 0;
    for (size_t i = 0; i < got.size(); ++i) changed += SameBits(got[i], row[i]) ? 0 : 1;
    Check(changed == 4, "out-of-range ids skipped: exactly the 4 valid ids changed");
    bool canary = true;
    for (int i = 0; i < kCanary; ++i) canary = canary && got[static_cast<size_t>(kVocab) + i] == row[static_cast<size_t>(kVocab) + i];
    Check(canary, "nothing written past the row");
    Check(got[10] == PenalizeLogit(row[10], 1.7f) && got[40] == PenalizeLogit(row[40], 1.7f), "valid ids penalized");
  }

  // p == 1 and an empty list: no change.
  {
    const std::vector<int32_t> ids = DistinctIds(rng, 1000);
    const std::vector<float> a = RunDevice(row, ids, 1.0f);
    bool same = true;
    for (size_t i = 0; i < row.size(); ++i) same = same && SameBits(a[i], row[i]);
    Check(same, "penalty 1.0: row unchanged");
    const std::vector<float> b = RunDevice(row, {}, 1.4f);
    same = true;
    for (size_t i = 0; i < row.size(); ++i) same = same && SameBits(b[i], row[i]);
    Check(same, "n_seen 0: row unchanged");
  }

  // Delta uploads: two launches over the halves equal one launch over the whole list.
  {
    const std::vector<int32_t> ids = DistinctIds(rng, 3000);
    const std::vector<float> whole = RunDevice(row, ids, 1.3f);
    DeviceBuffer<float> d_row(row.size());
    d_row.CopyFromHost(row);
    DeviceBuffer<int32_t> d_ids(ids.size());
    d_ids.CopyFromHost(ids);
    r4dx_apply_repetition_penalty_f32(reinterpret_cast<int64_t>(d_row.data()),
                                      reinterpret_cast<int64_t>(d_ids.data()), 1700, kVocab, 1.3f, 0);
    r4dx_apply_repetition_penalty_f32(reinterpret_cast<int64_t>(d_row.data()),
                                      reinterpret_cast<int64_t>(d_ids.data() + 1700), 1300, kVocab, 1.3f, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const std::vector<float> halves = d_row.CopyToHost();
    bool same = true;
    for (size_t i = 0; i < row.size(); ++i) same = same && SameBits(halves[i], whole[i]);
    Check(same, "two delta launches == one launch");
  }

  // Preconditions.
  {
    DeviceBuffer<float> d_row(row.size());
    DeviceBuffer<int32_t> d_ids(4);
    const auto lg = reinterpret_cast<int64_t>(d_row.data());
    const auto id = reinterpret_cast<int64_t>(d_ids.data());
    auto throws = [&](int64_t l, int64_t i, int64_t n, int64_t v, float p) {
      try {
        r4dx_apply_repetition_penalty_f32(l, i, n, v, p, 0);
      } catch (const std::exception&) {
        return true;
      }
      return false;
    };
    Check(throws(lg, id, 4, kVocab, 0.0f), "penalty 0 throws");
    Check(throws(lg, id, 4, kVocab, -1.0f), "negative penalty throws");
    Check(throws(lg, id, 4, kVocab, std::numeric_limits<float>::infinity()), "infinite penalty throws");
    Check(throws(lg, id, 4, kVocab, std::numeric_limits<float>::quiet_NaN()), "NaN penalty throws");
    Check(throws(0, id, 4, kVocab, 1.2f), "null logits throw");
    Check(throws(lg, 0, 4, kVocab, 1.2f), "null ids throw");
    Check(throws(lg, id, 4, 0, 1.2f), "vocab 0 throws");
    Check(!throws(lg, id, 0, kVocab, 0.0f), "n_seen 0 returns before validation (nothing to do)");
  }

  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}