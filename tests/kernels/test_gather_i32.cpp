// tests/kernels/test_gather_i32.cpp -- r4dx_gather_i32 (GatherI32Kernel, r4dx_kernels.hip) against
// a CPU reference gather, plus the in-kernel bounds guard: an idx outside [0, table_size) must
// clamp to table[0] instead of reading out-of-bounds device memory. Built for docs/r9700.md R9's
// reduced-vocab MTP draft head (MtpHead::Draft maps a subset-local argmax index back to a real
// vocab id via this kernel), same "device-side lookup, bounds-checked, clamp not fault" contract
// as r4dx_embedding_gather_bf16 -- see tests/kernels/test_embedding_gather.cpp for that kernel's
// identical test shape, which this file mirrors.
#include <cstdio>
#include <random>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

namespace {

bool CheckInRangeGatherMatchesReference() {
  const int64_t table_size = 8192, n = 37;
  std::mt19937 rng(41);
  std::uniform_int_distribution<int32_t> val_dist(0, 250000);

  std::vector<int32_t> table_h(static_cast<size_t>(table_size));
  for (auto& v : table_h) v = val_dist(rng);

  std::uniform_int_distribution<int32_t> idx_dist(0, static_cast<int32_t>(table_size - 1));
  std::vector<int32_t> idx_h(static_cast<size_t>(n));
  for (auto& i : idx_h) i = idx_dist(rng);

  DeviceBuffer<int32_t> table_d(table_h.size()), idx_d(idx_h.size()), out_d(static_cast<size_t>(n));
  table_d.CopyFromHost(table_h);
  idx_d.CopyFromHost(idx_h);
  out_d.Zero();

  r4dx_gather_i32(reinterpret_cast<int64_t>(table_d.data()), reinterpret_cast<int64_t>(idx_d.data()),
                   reinterpret_cast<int64_t>(out_d.data()), n, table_size, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int32_t> out_h = out_d.CopyToHost();

  bool ok = true;
  for (int64_t i = 0; i < n; ++i) {
    const int32_t expect = table_h[static_cast<size_t>(idx_h[static_cast<size_t>(i)])];
    if (out_h[static_cast<size_t>(i)] != expect) {
      std::printf("FAIL: gather mismatch at i=%lld: got %d, expected %d\n",
                  static_cast<long long>(i), out_h[static_cast<size_t>(i)], expect);
      ok = false;
    }
  }
  return ok;
}

bool CheckOutOfRangeIdxClampsToTableZero() {
  const int64_t table_size = 100;
  std::vector<int32_t> table_h(static_cast<size_t>(table_size));
  for (int64_t i = 0; i < table_size; ++i) table_h[static_cast<size_t>(i)] = static_cast<int32_t>(1000 + i);

  const std::vector<int32_t> idx_h = {0, -1, static_cast<int32_t>(table_size),
                                       static_cast<int32_t>(table_size) + 500, -1000000, 5};
  const int64_t n = static_cast<int64_t>(idx_h.size());

  DeviceBuffer<int32_t> table_d(table_h.size()), idx_d(idx_h.size()), out_d(static_cast<size_t>(n));
  table_d.CopyFromHost(table_h);
  idx_d.CopyFromHost(idx_h);
  out_d.Zero();

  r4dx_gather_i32(reinterpret_cast<int64_t>(table_d.data()), reinterpret_cast<int64_t>(idx_d.data()),
                   reinterpret_cast<int64_t>(out_d.data()), n, table_size, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<int32_t> out_h = out_d.CopyToHost();

  bool ok = true;
  for (int64_t i = 0; i < n; ++i) {
    const int32_t expect = (i == 5) ? table_h[5] : table_h[0];  // idx_h[5]==5 is the one in-range idx
    if (out_h[static_cast<size_t>(i)] != expect) {
      std::printf("FAIL: out-of-range idx %d (i=%lld) did not clamp to table[0], got %d expected %d\n",
                  idx_h[static_cast<size_t>(i)], static_cast<long long>(i), out_h[static_cast<size_t>(i)],
                  expect);
      ok = false;
    }
  }
  return ok;
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));

  bool ok = true;
  const bool in_range_ok = CheckInRangeGatherMatchesReference();
  ok &= in_range_ok;
  std::printf(in_range_ok ? "[PASS] CheckInRangeGatherMatchesReference\n"
                           : "[FAIL] CheckInRangeGatherMatchesReference\n");

  const bool oob_ok = CheckOutOfRangeIdxClampsToTableZero();
  ok &= oob_ok;
  std::printf(oob_ok ? "[PASS] CheckOutOfRangeIdxClampsToTableZero\n"
                      : "[FAIL] CheckOutOfRangeIdxClampsToTableZero\n");

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
