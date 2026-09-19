// tests/kernels/test_embedding_gather.cpp -- r4dx_embedding_gather_bf16 (EmbeddingGatherKernel,
// src/kernels/src/r4dx_kernels.hip) against a CPU reference gather, plus the in-kernel bounds
// guard (review finding, 2026-09-20): an id outside [0, vocab) must clamp to row 0 instead of
// reading out-of-bounds device memory.
#include <cstdio>
#include <random>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

namespace {

bool CheckInRangeGatherMatchesReference() {
  const int64_t vocab = 1000, hidden = 64, n = 17;
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

  std::vector<uint16_t> table_h(static_cast<size_t>(vocab * hidden));
  for (auto& v : table_h) v = FloatToBf16(dist(rng));

  std::vector<int32_t> ids_h(static_cast<size_t>(n));
  std::uniform_int_distribution<int32_t> id_dist(0, static_cast<int32_t>(vocab - 1));
  for (auto& id : ids_h) id = id_dist(rng);

  DeviceBuffer<uint16_t> table_d(table_h.size()), out_d(static_cast<size_t>(n * hidden));
  DeviceBuffer<int32_t> ids_d(static_cast<size_t>(n));
  table_d.CopyFromHost(table_h);
  ids_d.CopyFromHost(ids_h);
  out_d.Zero();

  r4dx_embedding_gather_bf16(reinterpret_cast<int64_t>(table_d.data()),
                              reinterpret_cast<int64_t>(ids_d.data()),
                              reinterpret_cast<int64_t>(out_d.data()), n, hidden, vocab, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> out_h = out_d.CopyToHost();

  bool ok = true;
  for (int64_t t = 0; t < n; ++t) {
    const uint16_t* ref_row = table_h.data() + static_cast<size_t>(ids_h[static_cast<size_t>(t)]) * hidden;
    const uint16_t* got_row = out_h.data() + static_cast<size_t>(t) * hidden;
    for (int64_t d = 0; d < hidden; ++d) {
      if (ref_row[d] != got_row[d]) {
        std::printf("FAIL: in-range gather mismatch at row %lld dim %lld\n",
                    static_cast<long long>(t), static_cast<long long>(d));
        ok = false;
      }
    }
  }
  return ok;
}

// Out-of-range ids (negative and >= vocab) must clamp to row 0 -- NOT read out-of-bounds device
// memory. This is the direct regression test for the review finding: before the fix, this
// launched with no guard at all, and this test (or a real out-of-range token id reaching the
// device path, e.g. a tokenizer/vocab-size mismatch or a malformed server request) would read
// past table_d's own allocation.
bool CheckOutOfRangeIdsClampToRowZero() {
  const int64_t vocab = 100, hidden = 32;
  std::mt19937 rng(23);
  std::uniform_real_distribution<float> dist(-3.0f, 3.0f);

  std::vector<uint16_t> table_h(static_cast<size_t>(vocab * hidden));
  for (auto& v : table_h) v = FloatToBf16(dist(rng));

  const std::vector<int32_t> ids_h = {0, -1, static_cast<int32_t>(vocab), static_cast<int32_t>(vocab) + 500,
                                       -1000000, 5};
  const int64_t n = static_cast<int64_t>(ids_h.size());

  DeviceBuffer<uint16_t> table_d(table_h.size()), out_d(static_cast<size_t>(n * hidden));
  DeviceBuffer<int32_t> ids_d(static_cast<size_t>(n));
  table_d.CopyFromHost(table_h);
  ids_d.CopyFromHost(ids_h);
  out_d.Zero();

  r4dx_embedding_gather_bf16(reinterpret_cast<int64_t>(table_d.data()),
                              reinterpret_cast<int64_t>(ids_d.data()),
                              reinterpret_cast<int64_t>(out_d.data()), n, hidden, vocab, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  std::vector<uint16_t> out_h = out_d.CopyToHost();

  bool ok = true;
  const uint16_t* row0 = table_h.data();  // row 0 -- the clamp target for every id in [1,2,3,4]
  const uint16_t* row5 = table_h.data() + static_cast<size_t>(5) * hidden;  // id 5 (index 5) is in-range
  for (int64_t t = 0; t < n; ++t) {
    const uint16_t* got_row = out_h.data() + static_cast<size_t>(t) * hidden;
    const uint16_t* expect_row = (t == 5) ? row5 : row0;  // ids_h[5]==5 is the one in-range id
    for (int64_t d = 0; d < hidden; ++d) {
      if (got_row[d] != expect_row[d]) {
        std::printf("FAIL: out-of-range id %d (row %lld) did not clamp to expected row, dim %lld\n",
                    ids_h[static_cast<size_t>(t)], static_cast<long long>(t),
                    static_cast<long long>(d));
        ok = false;
      }
    }
  }
  return ok;
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));

  bool ok = true;
  ok &= CheckInRangeGatherMatchesReference();
  std::printf(ok ? "[PASS] CheckInRangeGatherMatchesReference\n"
                 : "[FAIL] CheckInRangeGatherMatchesReference\n");

  const bool oob_ok = CheckOutOfRangeIdsClampToRowZero();
  ok &= oob_ok;
  std::printf(oob_ok ? "[PASS] CheckOutOfRangeIdsClampToRowZero\n"
                      : "[FAIL] CheckOutOfRangeIdsClampToRowZero\n");

  std::printf(ok ? "PASS\n" : "FAIL\n");
  return ok ? 0 : 1;
}
