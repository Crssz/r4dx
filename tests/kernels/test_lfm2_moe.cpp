// tests/kernels/test_lfm2_moe.cpp -- the MoE bookkeeping kernels (docs/lfm2-plan.md 3.5, task L1-7):
// r4dx_lfm2_moe_sort, r4dx_lfm2_moe_gather_bf16 and r4dx_lfm2_moe_combine_bf16 against lfm2_ref.hpp.
//   * sort: counts / offsets / sorted_pairs / pos_of_pair equal the CPU counting sort EXACTLY for random
//     top-4-of-64 ids at T = 1, 5, 64, 256 (P up to 1024), a skewed case (every token routed to the same
//     four experts: one expert holds all T rows), an empty-experts case, and invalid ids (skipped, pos -1);
//     the order inside an expert is ascending token order; two launches give the same bytes;
//   * gather: rows equal x[sorted_pairs[i] / top_k] bitwise; with valid_pairs the rows past the valid count
//     are left untouched;
//   * combine: slot layout ([T, k, hidden], the M = 1 decode path) and expert-sorted layout (through
//     pos_of_pair) both equal the reference (<= 1 bf16 ulp; the report prints how many are bitwise) and
//     each other bitwise; hidden 2048 and an odd 1000; two launches give the same bytes; negative positions
//     contribute nothing.
// GPU test (HIP device 1 via HIP_VISIBLE_DEVICES=1 in CMakeLists.txt); WRITTEN, NOT YET RUN. Built with /EHc-.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "lfm2_test_util.hpp"
#include "r4dx/kernels/lfm2_kernels.h"

using namespace r4dx::core;
using namespace lfm2_test;

namespace {

struct SortGot {
  std::vector<int32_t> counts, offsets, sorted, pos;
};

SortGot SortOnDevice(const std::vector<int32_t>& ids, int T, int K, int E) {
  const size_t Pn = static_cast<size_t>(T) * K;
  DeviceBuffer<int32_t> id_d(Pn), c_d(E), o_d(E + 1), s_d(Pn), p_d(Pn);
  id_d.CopyFromHost(ids);
  // poison the outputs so an unwritten entry is visible
  std::vector<int32_t> poison(Pn, -7);
  s_d.CopyFromHost(poison);
  p_d.CopyFromHost(poison);
  r4dx_lfm2_moe_sort(P(id_d.data()), T, K, E, P(c_d.data()), P(o_d.data()), P(s_d.data()), P(p_d.data()), 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  return {c_d.CopyToHost(), o_d.CopyToHost(), s_d.CopyToHost(), p_d.CopyToHost()};
}

std::vector<int32_t> RandomIds(std::mt19937& rng, int T, int K, int E) {
  std::vector<int32_t> ids(static_cast<size_t>(T) * K);
  for (int t = 0; t < T; ++t) {
    std::vector<int> perm(E);
    std::iota(perm.begin(), perm.end(), 0);
    std::shuffle(perm.begin(), perm.end(), rng);
    for (int k = 0; k < K; ++k) ids[static_cast<size_t>(t) * K + k] = perm[k];
  }
  return ids;
}

void TestSort(const std::string& name, const std::vector<int32_t>& ids, int T, int K, int E, bool has_invalid) {
  const auto want = lfm2_ref::MoeSort(ids, E);
  const SortGot g = SortOnDevice(ids, T, K, E);
  const SortGot g2 = SortOnDevice(ids, T, K, E);
  CheckF(g.counts == want.counts && g.offsets == want.offsets, name + ": counts and offsets == reference");
  const int valid = want.offsets[E];
  // only the first `valid` sorted entries are defined when ids are invalid
  bool sorted_ok = std::equal(want.sorted_pairs.begin(), want.sorted_pairs.begin() + valid, g.sorted.begin());
  CheckF(sorted_ok && g.pos == want.pos_of_pair, name + ": sorted_pairs and pos_of_pair == reference (exact)");
  bool asc = true;
  for (int e = 0; e < E; ++e)
    for (int i = g.offsets[e] + 1; i < g.offsets[e + 1]; ++i) asc &= g.sorted[i - 1] < g.sorted[i];
  CheckF(asc, name + ": ascending pair (token) order inside every expert");
  CheckF(g2.counts == g.counts && g2.offsets == g.offsets && g2.sorted == g.sorted && g2.pos == g.pos,
         name + ": second launch gives the same bytes");
  if (has_invalid) CheckF(valid < T * K, name + ": invalid ids were skipped");
}

void TestGather(std::mt19937& rng, int T, int K, int E, int hidden, int64_t extra) {
  const int64_t stride = hidden + extra;
  const auto ids = RandomIds(rng, T, K, E);
  const auto so = lfm2_ref::MoeSort(ids, E);
  const auto x = RandBf16(rng, static_cast<size_t>(T) * stride, 2.0f);
  const int Pn = T * K;
  DeviceBuffer<uint16_t> x_d(x.size()), out_d(static_cast<size_t>(Pn) * hidden);
  DeviceBuffer<int32_t> s_d(Pn);
  x_d.CopyFromHost(ToBits(x));
  s_d.CopyFromHost(so.sorted_pairs);
  out_d.CopyFromHost(std::vector<uint16_t>(out_d.size(), 0x7F31));
  r4dx_lfm2_moe_gather_bf16(P(x_d.data()), P(s_d.data()), P(out_d.data()), Pn, K, hidden, stride, 0, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const auto got = out_d.CopyToHost();
  bool ok = true;
  for (int i = 0; i < Pn && ok; ++i) {
    const int tok = so.sorted_pairs[i] / K;
    for (int j = 0; j < hidden; ++j) ok &= Bf16ToFloat(got[static_cast<size_t>(i) * hidden + j]) == x[static_cast<size_t>(tok) * stride + j];
  }
  CheckF(ok, "gather T=" + std::to_string(T) + " hidden=" + std::to_string(hidden) + " stride+" + std::to_string(extra) +
                 ": rows == x[sorted_pairs[i] / top_k] (bitwise)");
}

void TestGatherValidCount(std::mt19937& rng) {
  const int T = 6, K = 4, E = 16, hidden = 64;
  auto ids = RandomIds(rng, T, K, E);
  ids[3] = -1;
  ids[10] = 99;  // two skipped pairs
  const auto so = lfm2_ref::MoeSort(ids, E);
  const int valid = so.offsets[E];
  const auto x = RandBf16(rng, static_cast<size_t>(T) * hidden, 2.0f);
  const int Pn = T * K;
  DeviceBuffer<uint16_t> x_d(x.size()), out_d(static_cast<size_t>(Pn) * hidden);
  DeviceBuffer<int32_t> s_d(Pn), v_d(1);
  x_d.CopyFromHost(ToBits(x));
  s_d.CopyFromHost(so.sorted_pairs);
  v_d.CopyFromHost(std::vector<int32_t>{valid});
  out_d.CopyFromHost(std::vector<uint16_t>(out_d.size(), 0x7F31));
  r4dx_lfm2_moe_gather_bf16(P(x_d.data()), P(s_d.data()), P(out_d.data()), Pn, K, hidden, hidden, P(v_d.data()), 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const auto got = out_d.CopyToHost();
  bool tail_untouched = true, head_ok = true;
  for (int i = 0; i < Pn; ++i)
    for (int j = 0; j < hidden; ++j) {
      const uint16_t v = got[static_cast<size_t>(i) * hidden + j];
      if (i >= valid) tail_untouched &= v == 0x7F31;
      else head_ok &= Bf16ToFloat(v) == x[static_cast<size_t>(so.sorted_pairs[i] / K) * hidden + j];
    }
  Check(head_ok && tail_untouched, "gather with valid_pairs: valid rows gathered, the rest left untouched");
}

void TestCombine(std::mt19937& rng, int T, int K, int hidden, bool inject_invalid) {
  const int E = 64;
  auto ids = RandomIds(rng, T, K, E);
  if (inject_invalid) ids[1] = -1;
  const auto so = lfm2_ref::MoeSort(ids, E);
  const size_t Pn = static_cast<size_t>(T) * K;
  const auto y = RandBf16(rng, Pn * hidden, 3.0f);  // slot layout [T, K, hidden]
  std::uniform_real_distribution<float> dw(0.0f, 1.0f);
  std::vector<float> w(Pn);
  for (auto& v : w) v = dw(rng);
  std::vector<float> ysorted(y.size(), 0.0f);
  for (size_t p = 0; p < Pn; ++p) {
    if (so.pos_of_pair[p] < 0) continue;
    std::copy(y.begin() + p * hidden, y.begin() + (p + 1) * hidden, ysorted.begin() + static_cast<size_t>(so.pos_of_pair[p]) * hidden);
  }
  // reference: invalid pairs contribute nothing in the sorted layout; in the slot layout they are just rows
  const auto want_sorted = lfm2_ref::Combine(ysorted, &so.pos_of_pair, w, T, K, hidden);
  const auto want_slot = lfm2_ref::Combine(y, nullptr, w, T, K, hidden);

  DeviceBuffer<uint16_t> y_d(y.size()), ys_d(ysorted.size()), out_d(static_cast<size_t>(T) * hidden), out2_d(out_d.size());
  DeviceBuffer<int32_t> p_d(Pn);
  DeviceBuffer<float> w_d(Pn);
  y_d.CopyFromHost(ToBits(y));
  ys_d.CopyFromHost(ToBits(ysorted));
  p_d.CopyFromHost(so.pos_of_pair);
  w_d.CopyFromHost(w);
  r4dx_lfm2_moe_combine_bf16(P(y_d.data()), 0, P(w_d.data()), P(out_d.data()), T, K, hidden, 0);
  r4dx_lfm2_moe_combine_bf16(P(ys_d.data()), P(p_d.data()), P(w_d.data()), P(out2_d.data()), T, K, hidden, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  const auto got_slot = FromBits(out_d.CopyToHost());
  const auto got_sorted = FromBits(out2_d.CopyToHost());
  const std::string nm = "combine T=" + std::to_string(T) + " hidden=" + std::to_string(hidden) + (inject_invalid ? " invalid pos" : "");
  const Diff d1 = Compare(got_slot, want_slot, 1.0f), d2 = Compare(got_sorted, want_sorted, 1.0f);
  CheckF(d1.over == 0, nm + ": slot layout within 1 bf16 ulp of the reference");
  CheckF(d2.over == 0, nm + ": sorted layout (pos_of_pair) within 1 bf16 ulp of the reference");
  std::printf("      bitwise identical: slot %zu / %zu, sorted %zu / %zu\n", d1.n - d1.differ, d1.n, d2.n - d2.differ, d2.n);
  if (!inject_invalid) CheckF(got_slot == got_sorted, nm + ": slot and sorted layouts give the same bytes");
  r4dx_lfm2_moe_combine_bf16(P(ys_d.data()), P(p_d.data()), P(w_d.data()), P(out_d.data()), T, K, hidden, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  CheckF(FromBits(out_d.CopyToHost()) == got_sorted, nm + ": repeat launch gives the same bytes (determinism)");
}

void TestThrows() {
  DeviceBuffer<int32_t> a(64);
  auto throws = [&](int T, int K, int E) {
    try {
      r4dx_lfm2_moe_sort(P(a.data()), T, K, E, P(a.data()), P(a.data()), P(a.data()), P(a.data()), 0);
    } catch (const std::runtime_error&) {
      return true;
    }
    return false;
  };
  Check(throws(1, 4, 257), "sort: n_experts 257 throws");
  Check(throws(5000, 4, 64), "sort: T * top_k > 16384 throws");
  Check(throws(1, 4, 0), "sort: n_experts 0 throws");
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(314);
  for (int T : {1, 5, 64, 256}) {
    TestSort("sort T=" + std::to_string(T) + " top-4 of 64", RandomIds(rng, T, 4, 64), T, 4, 64, false);
  }
  {
    // skewed: every token routed to experts 3, 9, 20, 63 -> four experts hold all T rows, 60 are empty
    const int T = 256;
    std::vector<int32_t> ids(static_cast<size_t>(T) * 4);
    for (int t = 0; t < T; ++t) {
      ids[t * 4 + 0] = 63;
      ids[t * 4 + 1] = 3;
      ids[t * 4 + 2] = 20;
      ids[t * 4 + 3] = 9;
    }
    TestSort("sort skewed (4 experts hold all rows)", ids, T, 4, 64, false);
  }
  {
    auto ids = RandomIds(rng, 64, 4, 64);
    ids[0] = -1;
    ids[17] = 64;
    ids[100] = 1000;
    TestSort("sort with invalid ids", ids, 64, 4, 64, true);
  }
  TestSort("sort 2 experts top-2 (E == top_k)", RandomIds(rng, 9, 2, 2), 9, 2, 2, false);
  TestGather(rng, 1, 4, 64, 2048, 0);
  TestGather(rng, 64, 4, 64, 2048, 0);
  TestGather(rng, 256, 4, 64, 2048, 16);
  TestGather(rng, 7, 4, 64, 1000, 0);
  TestGatherValidCount(rng);
  TestCombine(rng, 1, 4, 2048, false);   // decode: [1, 4, 2048] slot layout
  TestCombine(rng, 256, 4, 2048, false);
  TestCombine(rng, 17, 4, 1000, false);  // hidden not a multiple of the block
  TestCombine(rng, 8, 4, 2048, true);
  TestThrows();
  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
