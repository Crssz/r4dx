// tests/kernels/test_argmax_order_cpu.cpp -- pure CPU. The algebra that makes the multi-workgroup argmax
// (decode-t1 item 1) bit-identical to the one-workgroup kernel it replaced: both compute the maximum, under
// "larger value first, then lower index", of the elements greater than the seed (-inf, 0) (argmax_ref.hpp),
// so the result cannot depend on how a row is partitioned or in which order partials are combined.
//
// Checked here on the host, against argmax_ref.hpp's emulation of the one-workgroup kernel's own order
// (strided per-thread scan + xor trees) and against the plain definition, for rows built to hurt:
// ties spread over every position class, NaNs, -inf, +inf, +/-0, rows with nothing greater than -inf.
// The device kernels are checked against the same references by tests/kernels/test_argmax_multi.cpp.
//
// Also pins the property the tie-break rests on for callers that merge shards (TP: rank r's local index
// plus its vocab offset, lower rank on equal values): partitioning a row into contiguous shards and
// merging the shard winners by (value, then lower global index) gives the whole row's answer.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "r4dx/kernels/argmax_ref.hpp"
#include "r4dx/kernels/sampler.hpp"  // r4dx::kernels::Argmax, the host reference decode used before the device argmax

using namespace r4dx::kernels;

namespace {

int g_fail = 0;
void Check(bool ok, const char* what, long long a = 0, long long b = 0) {
  if (!ok) {
    ++g_fail;
    std::fprintf(stderr, "FAIL: %s (%lld vs %lld)\n", what, a, b);
  }
}

bool SamePair(const ArgmaxPairRef& a, const ArgmaxPairRef& b) {
  if (a.i != b.i) return false;
  // equal as values (or both the seed): compare bits so -0 vs +0 would show
  uint32_t ba, bb;
  std::memcpy(&ba, &a.v, 4);
  std::memcpy(&bb, &b.v, 4);
  return ba == bb;
}

float Nan() { return std::numeric_limits<float>::quiet_NaN(); }
float Inf() { return std::numeric_limits<float>::infinity(); }

void CheckRow(const std::vector<float>& x, std::mt19937& rng, const char* what) {
  const int64_t n = static_cast<int64_t>(x.size());
  const ArgmaxPairRef direct = ArgmaxDirectRef(x.data(), n);
  const ArgmaxPairRef single = ArgmaxSingleWgRef(x.data(), n);
  Check(SamePair(direct, single), what, direct.i, single.i);
  // random partitions, random combine orders
  for (int trial = 0; trial < 12; ++trial) {
    std::vector<int64_t> cuts;
    const int parts = 1 + static_cast<int>(rng() % 40);
    for (int p = 0; p < parts - 1 && n > 1; ++p) cuts.push_back(1 + static_cast<int64_t>(rng() % static_cast<uint64_t>(n - 1)));
    std::sort(cuts.begin(), cuts.end());
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    std::vector<size_t> order(cuts.size() + 1);
    for (size_t k = 0; k < order.size(); ++k) order[k] = k;
    std::shuffle(order.begin(), order.end(), rng);
    const ArgmaxPairRef part = ArgmaxPartitionedRef(x.data(), n, cuts, order);
    Check(SamePair(direct, part), what, direct.i, part.i);
  }
  // the host sampler's Argmax agrees wherever its own contract applies (no NaN at index 0, something
  // greater than -inf: it returns the first maximum, which is the lowest index)
  bool plain = n > 0 && !std::isnan(x[0]);
  bool any_finite_max = false;
  for (float v : x) any_finite_max = any_finite_max || v > -Inf();
  if (plain && any_finite_max) Check(Argmax(x.data(), n) == direct.i, "host Argmax agrees", Argmax(x.data(), n), direct.i);
}

}  // namespace

int main() {
  std::mt19937 rng(20261008);
  std::uniform_real_distribution<float> uni(-3.0f, 3.0f);

  // ---- random rows of the sizes the callers use (and awkward ones) ----
  for (int64_t n : {1, 2, 31, 32, 33, 255, 256, 257, 1000, 4096, 32768, 248320, 248321}) {
    std::vector<float> x(static_cast<size_t>(n));
    for (float& v : x) v = uni(rng);
    CheckRow(x, rng, "random row");
  }

  // ---- ties: a plateau of the maximum at assorted positions, including block / thread boundaries ----
  for (int64_t n : {256, 1024, 5000, 248320}) {
    for (int rep = 0; rep < 8; ++rep) {
      std::vector<float> x(static_cast<size_t>(n));
      for (float& v : x) v = std::round(uni(rng));  // few distinct values: ties everywhere
      const float top = 5.0f;
      const int copies = 1 + static_cast<int>(rng() % 6);
      for (int c = 0; c < copies; ++c) x[rng() % static_cast<uint64_t>(n)] = top;
      CheckRow(x, rng, "tied maxima");
    }
    std::vector<float> flat(static_cast<size_t>(n), 1.25f);  // all equal: index 0
    CheckRow(flat, rng, "all equal");
    Check(ArgmaxDirectRef(flat.data(), n).i == 0, "all equal -> 0");
  }

  // ---- NaN never wins; -inf / all-NaN rows give 0; +inf and signed zeros ----
  {
    std::vector<float> x(1000, Nan());
    CheckRow(x, rng, "all NaN");
    Check(ArgmaxDirectRef(x.data(), 1000).i == 0, "all NaN -> 0");
    std::vector<float> ninf(1000, -Inf());
    CheckRow(ninf, rng, "all -inf");
    Check(ArgmaxDirectRef(ninf.data(), 1000).i == 0, "all -inf -> 0");
    std::vector<float> mix(1000, Nan());
    mix[7] = -Inf();
    mix[500] = -Inf();  // the maximum is -inf, the answer is the seed's index 0, not 7
    CheckRow(mix, rng, "NaN and -inf");
    Check(ArgmaxDirectRef(mix.data(), 1000).i == 0, "max -inf -> 0, not the first -inf");
    std::vector<float> one(1000, Nan());
    one[993] = -1e30f;
    CheckRow(one, rng, "NaN with one finite");
    Check(ArgmaxDirectRef(one.data(), 1000).i == 993, "one finite among NaN");
    std::vector<float> nan_then(1000, -2.0f);
    nan_then[0] = Nan();
    nan_then[300] = -0.5f;
    nan_then[301] = -0.5f;
    CheckRow(nan_then, rng, "NaN at 0");
    Check(ArgmaxDirectRef(nan_then.data(), 1000).i == 300, "NaN at 0 does not win, lowest tie wins");
    std::vector<float> pinf(1000, 1.0f);
    pinf[640] = Inf();
    pinf[641] = Inf();
    pinf[10] = Nan();
    CheckRow(pinf, rng, "+inf plateau");
    Check(ArgmaxDirectRef(pinf.data(), 1000).i == 640, "+inf plateau -> lowest");
    std::vector<float> zeros(1000, -0.0f);
    zeros[400] = 0.0f;  // +0 equals -0: a tie, the lowest index (0) wins
    CheckRow(zeros, rng, "signed zeros");
    Check(ArgmaxDirectRef(zeros.data(), 1000).i == 0, "-0 / +0 tie -> index 0");
  }

  // ---- random garbage mixes of everything above ----
  for (int rep = 0; rep < 40; ++rep) {
    const int64_t n = 1 + static_cast<int64_t>(rng() % 3000);
    std::vector<float> x(static_cast<size_t>(n));
    for (float& v : x) {
      switch (rng() % 9) {
        case 0: v = Nan(); break;
        case 1: v = -Inf(); break;
        case 2: v = Inf(); break;
        case 3: v = 0.0f; break;
        case 4: v = -0.0f; break;
        case 5: v = 2.0f; break;
        default: v = std::round(uni(rng) * 4) / 4; break;
      }
    }
    CheckRow(x, rng, "garbage mix");
  }

  // ---- the TP shard merge: shard winners by (value, then lower global index) = the whole row ----
  for (int rep = 0; rep < 60; ++rep) {
    const int64_t n = 64 + static_cast<int64_t>(rng() % 4000);
    std::vector<float> x(static_cast<size_t>(n));
    for (float& v : x) v = (rng() % 11 == 0) ? Nan() : std::round(uni(rng) * 2) / 2;
    const int64_t split = 1 + static_cast<int64_t>(rng() % static_cast<uint64_t>(n - 1));
    const ArgmaxPairRef lo = ArgmaxDirectRef(x.data(), split);
    ArgmaxPairRef hi = ArgmaxDirectRef(x.data() + split, n - split);
    hi.i += static_cast<int32_t>(split);  // local -> global, as the host merge does
    // an all-seed shard reports (-inf, local 0): its global index is the shard's first, which only matters
    // if nothing anywhere beats -inf (then the whole row's answer is 0, the first shard's seed)
    ArgmaxPairRef merged = lo;
    if (hi.v > merged.v) merged = hi;  // the strictly larger value wins; ties keep the lower rank
    Check(SamePair(ArgmaxDirectRef(x.data(), n), merged) ||
              (merged.v == -Inf() && ArgmaxDirectRef(x.data(), n).i == 0),
          "shard merge", merged.i, ArgmaxDirectRef(x.data(), n).i);
  }

  if (g_fail != 0) {
    std::fprintf(stderr, "%d check(s) failed\n", g_fail);
    return 1;
  }
  std::printf("test_argmax_order_cpu: OK\n");
  return 0;
}
