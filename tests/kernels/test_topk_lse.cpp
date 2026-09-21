// tests/kernels/test_topk_lse.cpp -- r4dx_topk_lse_f32 (docs/sampling.md "Row summary kernel")
// against a CPU fp64 reference.
//
// Two contracts, checked separately:
//   * the top-K ids AND values must be EXACT (the kernel only copies fp32 values it read, and the
//     reference sorts by the same total order -- value descending, ties toward the LOWER id), so
//     every mismatch here is a real defect, not a tolerance question;
//   * the logsumexp is a reduction over the whole row, so it is checked as an ABSOLUTE error
//     against an fp64 Kahan reference, with the measured worst case printed for docs/sampling.md.
//
// Tie-heavy inputs get their own cases because the tie-break is what the canonical-order
// definition in docs/sampling.md rests on: 12 distinct levels, all-identical rows, and rows of
// -inf (which is also where the kernel's own (-inf, INT32_MAX) per-thread sentinel would leak into
// the output if the total order were not strict).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"
#include "r4dx/kernels/summary_sampler.hpp"

using namespace r4dx::core;

static_assert(R4DX_TOPK_LSE_K == r4dx::kernels::kRowSummaryMaxK,
              "kernels.h and summary_sampler.hpp must agree on K");

namespace {

constexpr int kK = R4DX_TOPK_LSE_K;
constexpr float kNegInf = -std::numeric_limits<float>::infinity();

int g_failures = 0;
double g_worst_lse_err = 0.0;
const char* g_worst_lse_case = "(none)";

void Check(bool cond, const std::string& what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what.c_str());
    ++g_failures;
  } else {
    std::printf("PASS: %s\n", what.c_str());
  }
}

// The kernel's documented total order: higher value wins, ties break toward the LOWER id.
bool Better(float av, int32_t ai, float bv, int32_t bi) {
  return av > bv || (av == bv && ai < bi);
}

struct Ref {
  std::vector<int32_t> ids;   // [rows, kK]
  std::vector<float> vals;    // [rows, kK]
  std::vector<double> lse;    // [rows]
};

Ref Reference(const std::vector<float>& logits, int rows, int64_t vocab, double inv_t) {
  Ref out;
  out.ids.resize(static_cast<size_t>(rows) * kK);
  out.vals.resize(static_cast<size_t>(rows) * kK);
  out.lse.resize(static_cast<size_t>(rows));
  std::vector<int32_t> order(static_cast<size_t>(vocab));
  for (int r = 0; r < rows; ++r) {
    const float* lr = logits.data() + static_cast<size_t>(r) * vocab;
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + kK, order.end(),
                      [lr](int32_t a, int32_t b) { return Better(lr[a], a, lr[b], b); });
    for (int j = 0; j < kK; ++j) {
      out.ids[static_cast<size_t>(r) * kK + j] = order[j];
      out.vals[static_cast<size_t>(r) * kK + j] = lr[order[j]];
    }
    // fp64 Kahan-compensated logsumexp of the SCALED row.
    double m = -std::numeric_limits<double>::infinity();
    for (int64_t i = 0; i < vocab; ++i) m = std::max(m, static_cast<double>(lr[i]));
    if (m == -std::numeric_limits<double>::infinity()) {
      out.lse[r] = m;
      continue;
    }
    double sum = 0.0, comp = 0.0;
    for (int64_t i = 0; i < vocab; ++i) {
      const double term = std::exp((static_cast<double>(lr[i]) - m) * inv_t);
      const double y = term - comp;
      const double t = sum + y;
      comp = (t - sum) - y;
      sum = t;
    }
    out.lse[r] = m * inv_t + std::log(sum);
  }
  return out;
}

bool RunCase(const char* label, const std::vector<float>& logits, int rows, int64_t vocab,
             float inv_t, double lse_tol = 1e-4) {
  const Ref ref = Reference(logits, rows, vocab, static_cast<double>(inv_t));

  DeviceBuffer<float> l_d(logits.size());
  DeviceBuffer<int32_t> ids_d(static_cast<size_t>(rows) * kK);
  DeviceBuffer<float> vals_d(static_cast<size_t>(rows) * kK);
  DeviceBuffer<float> lse_d(static_cast<size_t>(rows));
  l_d.CopyFromHost(logits);
  r4dx_topk_lse_f32(reinterpret_cast<int64_t>(l_d.data()), reinterpret_cast<int64_t>(ids_d.data()),
                     reinterpret_cast<int64_t>(vals_d.data()),
                     reinterpret_cast<int64_t>(lse_d.data()), rows, vocab, inv_t, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());

  const std::vector<int32_t> ids = ids_d.CopyToHost();
  const std::vector<float> vals = vals_d.CopyToHost();
  const std::vector<float> lse = lse_d.CopyToHost();

  int bad_ids = 0, bad_vals = 0, first_bad = -1;
  for (size_t i = 0; i < ids.size(); ++i) {
    if (ids[i] != ref.ids[i]) {
      if (first_bad < 0) first_bad = static_cast<int>(i);
      ++bad_ids;
    }
    if (vals[i] != ref.vals[i]) ++bad_vals;
  }
  double worst = 0.0;
  int bad_lse = 0;
  for (int r = 0; r < rows; ++r) {
    const double want = ref.lse[r];
    const double got = static_cast<double>(lse[r]);
    if (std::isinf(want) && want < 0.0) {
      if (!(std::isinf(got) && got < 0.0)) ++bad_lse;
      continue;
    }
    const double err = std::fabs(got - want);
    worst = std::max(worst, err);
    if (!(err <= lse_tol)) ++bad_lse;
  }
  if (worst > g_worst_lse_err) {
    g_worst_lse_err = worst;
    g_worst_lse_case = label;
  }

  const bool pass = bad_ids == 0 && bad_vals == 0 && bad_lse == 0;
  std::printf("  %-32s rows=%d V=%-7lld invT=%-7.4f  ids=%d vals=%d lse=%d (worst |dlse|=%.3g)  %s\n",
              label, rows, static_cast<long long>(vocab), inv_t, bad_ids, bad_vals, bad_lse, worst,
              pass ? "PASS" : "FAIL");
  if (!pass && first_bad >= 0) {
    std::printf("    first id/val mismatch at [%d/%d]: got id=%d val=%.9g, want id=%d val=%.9g\n",
                first_bad / kK, first_bad % kK, ids[first_bad], vals[first_bad], ref.ids[first_bad],
                ref.vals[first_bad]);
  }
  if (!pass && bad_lse > 0) {
    for (int r = 0; r < rows; ++r) {
      std::printf("    lse[%d]: got %.9g want %.9g\n", r, static_cast<double>(lse[r]), ref.lse[r]);
    }
  }
  if (!pass) ++g_failures;
  return pass;
}

const float kInvTemps[] = {0.5f, 1.0f, 1.0f / 0.6f, 5.0f};

}  // namespace

int main() {
  // Unbuffered: a GPU test that dies takes the CRT's stdout buffer with it, and a crash report
  // with zero output says nothing about which check was running.
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(20260921u);

  // ---- 1. Random values, the real vocab and a small one, 1 and 8 rows, every inv_temperature ---
  std::printf("[1] random values\n");
  for (int64_t vocab : {int64_t(4096), int64_t(248320)}) {
    for (int rows : {1, 8}) {
      std::vector<float> logits(static_cast<size_t>(rows) * vocab);
      std::uniform_real_distribution<float> d(-20.0f, 20.0f);
      for (auto& v : logits) v = d(rng);
      for (float inv_t : kInvTemps) RunCase("random", logits, rows, vocab, inv_t);
    }
  }

  // ---- 2. Tie-heavy inputs ---------------------------------------------------------------------
  std::printf("[2] tie-heavy inputs\n");
  for (int64_t vocab : {int64_t(4096), int64_t(248320)}) {
    std::vector<float> logits(static_cast<size_t>(8) * vocab);
    std::uniform_int_distribution<int> d(0, 11);
    for (auto& v : logits) v = static_cast<float>(d(rng));
    RunCase("12 distinct levels", logits, 8, vocab, 1.0f);
    RunCase("12 distinct levels", logits, 8, vocab, 5.0f);
  }
  {
    // Every entry identical: the answer is exactly ids 0..63, and lse = log(vocab) + v*invT.
    const int64_t vocab = 248320;
    std::vector<float> logits(static_cast<size_t>(8) * vocab, 1.25f);
    RunCase("all values identical", logits, 8, vocab, 1.0f);
  }
  {
    // A whole row of -inf: lse must be -inf, not NaN, and the top-K must be ids 0..63 (the
    // per-thread sentinel is (-inf, INT32_MAX) and must lose to every real -inf entry).
    const int64_t vocab = 4096;
    std::vector<float> logits(static_cast<size_t>(2) * vocab, kNegInf);
    RunCase("all -inf", logits, 2, vocab, 1.0f);
    RunCase("all -inf", logits, 2, vocab, 1.0f / 0.6f);
  }
  {
    // Half the row masked out with -inf (the shape a logit-bias / banned-token mask produces).
    const int64_t vocab = 248320;
    const int rows = 4;
    std::vector<float> logits(static_cast<size_t>(rows) * vocab);
    std::uniform_real_distribution<float> d(-10.0f, 10.0f);
    for (int r = 0; r < rows; ++r) {
      for (int64_t i = 0; i < vocab; ++i) {
        logits[static_cast<size_t>(r) * vocab + i] = (i % 2 == 0) ? d(rng) : kNegInf;
      }
    }
    for (float inv_t : kInvTemps) RunCase("half the row is -inf", logits, rows, vocab, inv_t);
  }

  // ---- 3. Peaked and flat rows ------------------------------------------------------------------
  std::printf("[3] peaked / flat rows\n");
  {
    const int64_t vocab = 248320;
    const int rows = 8;
    std::vector<float> logits(static_cast<size_t>(rows) * vocab);
    std::uniform_real_distribution<float> d(-5.0f, 5.0f);
    for (auto& v : logits) v = d(rng);
    for (int r = 0; r < rows; ++r) {
      // One logit +40 above everything else in its row: the softmax is a point mass and the whole
      // lse is carried by a single term, the worst case for the max-subtraction being wrong.
      logits[static_cast<size_t>(r) * vocab + (12345 + r * 7919)] = 45.0f;
    }
    for (float inv_t : kInvTemps) RunCase("one logit +40 above the rest", logits, rows, vocab, inv_t);
  }
  {
    // Perfectly flat but NOT identical: 248320 equal values would be the tie case above, so use a
    // tiny deterministic ramp -- every probability within a factor of e^0.001 of every other, the
    // worst case for the summation (no term dominates).
    const int64_t vocab = 248320;
    const int rows = 2;
    std::vector<float> logits(static_cast<size_t>(rows) * vocab);
    for (int r = 0; r < rows; ++r) {
      for (int64_t i = 0; i < vocab; ++i) {
        logits[static_cast<size_t>(r) * vocab + i] =
            static_cast<float>(1e-3 * static_cast<double>((i * 7919 + r) % 1000) / 1000.0);
      }
    }
    for (float inv_t : kInvTemps) RunCase("flat row", logits, rows, vocab, inv_t);
  }

  // ---- 4. rows sweep 1..8 -----------------------------------------------------------------------
  std::printf("[4] rows sweep\n");
  for (int rows = 1; rows <= 8; ++rows) {
    const int64_t vocab = 4096;
    std::vector<float> logits(static_cast<size_t>(rows) * vocab);
    std::uniform_real_distribution<float> d(-5.0f, 5.0f);
    for (auto& v : logits) v = d(rng);
    char label[64];
    std::snprintf(label, sizeof(label), "rows=%d", rows);
    RunCase(label, logits, rows, vocab, 1.0f / 0.6f);
  }

  // ---- 5. Preconditions must THROW, not produce a wrong summary ---------------------------------
  // (this TU is compiled with /EHc- -- see tests/kernels/CMakeLists.txt for why)
  std::printf("[5] preconditions\n");
  {
    const int64_t vocab = 4096;
    DeviceBuffer<float> l_d(static_cast<size_t>(vocab));
    DeviceBuffer<int32_t> ids_d(8 * kK);
    DeviceBuffer<float> vals_d(8 * kK);
    DeviceBuffer<float> lse_d(8);
    const int64_t l = reinterpret_cast<int64_t>(l_d.data());
    const int64_t i = reinterpret_cast<int64_t>(ids_d.data());
    const int64_t v = reinterpret_cast<int64_t>(vals_d.data());
    const int64_t s = reinterpret_cast<int64_t>(lse_d.data());
    auto throws = [&](int rows, int64_t vc, float inv_t) {
      try {
        r4dx_topk_lse_f32(l, i, v, s, rows, vc, inv_t, 0);
      } catch (const std::exception&) {
        return true;
      }
      R4DX_HIP_CHECK(hipDeviceSynchronize());
      return false;
    };
    Check(throws(9, vocab, 1.0f), "rows > 8 throws");
    Check(throws(1, kK, 1.0f), "vocab == K throws");
    Check(throws(1, kK - 1, 1.0f), "vocab < K throws");
    Check(throws(1, vocab, 0.0f), "inv_temperature == 0 throws");
    Check(throws(1, vocab, -1.0f), "inv_temperature < 0 throws");
    Check(throws(1, vocab, std::numeric_limits<float>::infinity()), "inv_temperature == inf throws");
    Check(throws(1, vocab, std::numeric_limits<float>::quiet_NaN()), "inv_temperature NaN throws");
    // rows <= 0 is a no-op, not an error (every other kernel in this header behaves that way).
    bool no_op_ok = true;
    try {
      r4dx_topk_lse_f32(l, i, v, s, 0, vocab, 1.0f, 0);
    } catch (const std::exception&) {
      no_op_ok = false;
    }
    Check(no_op_ok, "rows == 0 is a no-op");
  }

  std::printf("\nworst logsumexp |error| over every case: %.4g (%s), tolerance 1e-4\n",
              g_worst_lse_err, g_worst_lse_case);
  if (g_failures > 0) {
    std::fprintf(stderr, "%d check(s) FAILED\n", g_failures);
    std::printf("FAIL\n");
    return 1;
  }
  std::printf("ALL PASS\n");
  return 0;
}
