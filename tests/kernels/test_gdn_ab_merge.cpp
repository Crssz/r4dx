// tests/kernels/test_gdn_ab_merge.cpp -- decode-t1 item 2: gdn.in_proj_a and gdn.in_proj_b as ONE
// r4d_gemm_bf16_nt_m64 launch over the concatenated [a; b] weight (N = 2H, Container's BuildGdnAb) against
// the two launches it replaces (N = H each), BIT FOR BIT, in the exact geometry GdnLayer::Forward uses
// (WV 4, SK 4, MB 1; at most 64 rows per launch, the merged output's row stride 2H with b at column H).
// The GEMM's output column is one WMMA tile column reduced over the same K splits whatever N is, so the
// columns of the merged output must be the separate launches' bits -- including when H is not a multiple
// of the 16-wide tile (a TP shard's 24 or 8 heads: a tile then straddles a and b).
// Needs a HIP device.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "r4dx/core/device_buffer.hpp"
#include "r4dx/core/dtype.hpp"
#include "r4dx/core/error.hpp"
#include "r4dx/core/r4d.hpp"

using namespace r4dx::core;

namespace {

int g_fail = 0;
int g_cases = 0;

std::vector<uint16_t> RandomBf16(std::mt19937& rng, size_t n, float stddev) {
  std::normal_distribution<float> nd(0.0f, stddev);
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = FloatToBf16(nd(rng));
  return v;
}

void CheckCase(std::mt19937& rng, int M, int K, int H) {
  const std::vector<uint16_t> a_h = RandomBf16(rng, static_cast<size_t>(M) * K, 1.0f);
  const std::vector<uint16_t> wa_h = RandomBf16(rng, static_cast<size_t>(H) * K, 0.02f);
  const std::vector<uint16_t> wb_h = RandomBf16(rng, static_cast<size_t>(H) * K, 0.02f);
  std::vector<uint16_t> wab_h(wa_h);  // a's rows, then b's -- BuildGdnAb's layout
  wab_h.insert(wab_h.end(), wb_h.begin(), wb_h.end());

  DeviceBuffer<uint16_t> a(a_h.size()), wa(wa_h.size()), wb(wb_h.size()), wab(wab_h.size());
  a.CopyFromHost(a_h);
  wa.CopyFromHost(wa_h);
  wb.CopyFromHost(wb_h);
  wab.CopyFromHost(wab_h);
  DeviceBuffer<uint16_t> ca(static_cast<size_t>(M) * H), cb(static_cast<size_t>(M) * H),
      cab(static_cast<size_t>(M) * 2 * H);

  // GdnLayer::Forward's loop: 64-row slices, WV 4, SK 4, MB 1
  for (int m0 = 0; m0 < M; m0 += 64) {
    const int m = std::min(64, M - m0);
    r4d::GemmBf16NtM64(a.data() + static_cast<size_t>(m0) * K, wa.data(), ca.data() + static_cast<size_t>(m0) * H, m, K, H,
                        4, 4, 1, nullptr);
    r4d::GemmBf16NtM64(a.data() + static_cast<size_t>(m0) * K, wb.data(), cb.data() + static_cast<size_t>(m0) * H, m, K, H,
                        4, 4, 1, nullptr);
  }
  for (int rep = 0; rep < 3; ++rep) {
    cab.Zero();
    for (int m0 = 0; m0 < M; m0 += 64) {
      const int m = std::min(64, M - m0);
      r4d::GemmBf16NtM64(a.data() + static_cast<size_t>(m0) * K, wab.data(),
                          cab.data() + static_cast<size_t>(m0) * 2 * H, m, K, 2 * H, 4, 4, 1, nullptr);
    }
    R4DX_HIP_CHECK(hipDeviceSynchronize());
    const std::vector<uint16_t> ca_h = ca.CopyToHost(), cb_h = cb.CopyToHost(), cab_h = cab.CopyToHost();
    for (int m = 0; m < M; ++m) {
      for (int n = 0; n < H; ++n) {
        ++g_cases;
        const uint16_t ea = ca_h[static_cast<size_t>(m) * H + n], eb = cb_h[static_cast<size_t>(m) * H + n];
        const uint16_t ga = cab_h[static_cast<size_t>(m) * 2 * H + n];
        const uint16_t gb = cab_h[static_cast<size_t>(m) * 2 * H + H + n];
        if (ea != ga || eb != gb) {
          if (++g_fail <= 10) {
            std::fprintf(stderr, "FAIL M=%d K=%d H=%d row %d col %d: a %04x vs %04x, b %04x vs %04x\n", M, K, H, m, n,
                         ea, ga, eb, gb);
          }
        }
      }
    }
  }
}

}  // namespace

int main() {
  R4DX_HIP_CHECK(hipSetDevice(0));
  std::mt19937 rng(7102026);
  // 48 = the model's value heads, 24 / 8 = TP shards (tiles straddle a and b), 16 = tile aligned
  for (int H : {48, 24, 16, 8}) {
    for (int M : {1, 2, 5, 8, 16, 17, 33, 64, 65, 130, 256}) CheckCase(rng, M, 5120, H);
  }
  CheckCase(rng, 1, 2048, 48);  // another K
  if (g_fail != 0) {
    std::fprintf(stderr, "test_gdn_ab_merge: %d of %d outputs differ\n", g_fail, g_cases);
    return 1;
  }
  std::printf("test_gdn_ab_merge: OK (%d outputs, merged N = 2H launch == two N = H launches bit for bit)\n", g_cases);
  return 0;
}
