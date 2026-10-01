// tests/kernels/test_embed_scaled.cpp -- r4dx_embedding_gather_scaled_bf16 (Gemma 4's
// ScaledWordEmbedding, docs/gemma4-plan.md 3.5): out[i,:] = bf16(table[ids[i],:] * scale). The bf16
// product is checked EXACTLY against the host computation of the same float product (one rounding),
// at scale 62.0 (bf16(sqrt(3840)); the unrounded 61.968 gives different bytes, which is asserted so
// the parity flag cannot silently stop mattering), scale 1 (== r4dx_embedding_gather_bf16 bit for
// bit), and the out-of-range-id guard (reads row 0). GPU test.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/kernels/gemma_kernels.h"
#include "r4dx/kernels/kernels.h"

using namespace r4dx::core;

namespace {
bool g_ok = true;
void Check(bool cond, const char* what) {
  std::printf("%-80s %s\n", what, cond ? "ok" : "FAIL");
  g_ok = g_ok && cond;
}
int64_t P(const void* p) { return reinterpret_cast<int64_t>(p); }
}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(9);
  const int64_t vocab = 4096, hidden = 3840, n = 7;
  std::uniform_real_distribution<float> dist(-0.2f, 0.2f);
  std::vector<uint16_t> table(static_cast<size_t>(vocab * hidden));
  for (auto& v : table) v = FloatToBf16(dist(rng));
  std::vector<int32_t> ids = {0, 5, 4095, 1234, 5, 77, 2048};
  DeviceBuffer<uint16_t> table_d(table.size()), out_d(static_cast<size_t>(n * hidden));
  DeviceBuffer<int32_t> ids_d(ids.size());
  table_d.CopyFromHost(table);
  ids_d.CopyFromHost(ids);

  const float scale62 = Bf16ToFloat(FloatToBf16(std::sqrt(3840.0f)));
  Check(scale62 == 62.0f, "bf16(sqrt(3840)) is 62.0");

  const auto run = [&](float scale) {
    r4dx_embedding_gather_scaled_bf16(P(table_d.data()), P(ids_d.data()), P(out_d.data()), n, hidden,
                                       vocab, scale, 0);
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    return out_d.CopyToHost();
  };

  auto got = run(scale62);
  bool exact = true;
  for (int64_t r = 0; r < n; ++r)
    for (int64_t i = 0; i < hidden; ++i) {
      const uint16_t ref = FloatToBf16(Bf16ToFloat(table[ids[r] * hidden + i]) * scale62);
      if (got[r * hidden + i] != ref) exact = false;
    }
  Check(exact, "scale 62.0: every element is bf16(row * 62.0), exactly");

  const auto got_unrounded = run(std::sqrt(3840.0f));
  Check(got_unrounded != got, "scale 61.968 (unrounded) differs from 62.0 (the parity flag matters)");

  got = run(1.0f);
  DeviceBuffer<uint16_t> plain_d(out_d.size());
  r4dx_embedding_gather_bf16(P(table_d.data()), P(ids_d.data()), P(plain_d.data()), n, hidden, vocab, 0);
  R4DX_HIP_CHECK(hipDeviceSynchronize());
  Check(got == plain_d.CopyToHost(), "scale 1.0 equals r4dx_embedding_gather_bf16 bit for bit");

  // Out-of-range ids read row 0 (the shared guard).
  ids = {-1, 4096, 1 << 30, 3, 0, 1, 2};
  ids_d.CopyFromHost(ids);
  got = run(scale62);
  bool guard = true;
  for (int64_t i = 0; i < hidden; ++i) {
    const uint16_t ref = FloatToBf16(Bf16ToFloat(table[i]) * scale62);
    for (int r = 0; r < 3; ++r) guard = guard && got[r * hidden + i] == ref;
  }
  Check(guard, "ids -1, vocab and 2^30 read row 0");

  std::printf(g_ok ? "PASS\n" : "FAIL\n");
  return g_ok ? 0 : 1;
}
