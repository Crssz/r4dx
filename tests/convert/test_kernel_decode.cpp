// Kernel-literal decode test (review finding, major: "the gating tests do not gate the thing that
// matters" -- test_pack_bytes.cpp only checks the C++ packer against fixtures produced by the same
// author's Python references, so a permutation bug present in BOTH would pass silently, and the
// only check against the real GEMM kernels' actual pointer arithmetic (tools/convert_ref/
// kernel_crosscheck.py) is a manual GPU script nobody's build runs).
//
// This test decodes the packed bytes by following each kernel's OWN indexing (re-derived here from
// third_party/libr4d/r4d_gemm_w4a16_nt_m64.hip source, not by inverting PackW4Nibbles' own loops) and
// asserts the decode recovers exactly the quantizer's intended (row,k) code and exactly the
// packer's intended f16 scale/zero bit pattern. CPU-only, no HIP device needed, so it runs on every
// build -- the permanent, always-on replacement for the gap the review flagged.
//
// w4a16 (packed by PackW4Nibbles):
//   dword index for (tile t, k-block kb of 64, lane 0..31, k-step s of 0..3) is
//     ((t*(K/64)+kb)*32+lane)*4+s
//   row = t*16 + (lane&15); within that dword, nibble at bit position 4*nibble_pos holds element
//   e (0..7) where nibble_pos = 2e (e<4) or 2(e-4)+1 (e>=4) -- r4d_gemm_w4a16_nt_m64.hip's literal
//   "nibble 2e (e<4) / 2(e-4)+1 (e>=4)" statement -- and that element's k is
//     k = (kb*4+s)*16 + 8*(e>>2) + 4*(lane>>4) + (e&3)
//   w4a16's dequant XORs the stored nibble by 8 to recover the 0..15 code.
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/quant_int4.hpp"

namespace {

using namespace r4dx_convert;

int g_failures = 0;

void Expect(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_failures;
  }
}

// ---- w4a16 index math (independently derived from the kernel, see file header) -----------------

int DecodeRow(int t, int lane) { return t * 16 + (lane & 15); }

int DecodeK(int kb, int s, int e, int lane) {
  return (kb * 4 + s) * 16 + 8 * (e >> 2) + 4 * (lane >> 4) + (e & 3);
}

int NibblePos(int e) { return (e < 4) ? (2 * e) : (2 * (e - 4) + 1); }

void CheckW4A16(const std::vector<float>& w, int N, int K, int group) {
  std::vector<uint8_t> q, zero;
  std::vector<float> scale;
  QuantizeInt4Asymmetric(w.data(), N, K, group, /*nthreads=*/4, q, scale, zero);
  auto wq = PackW4Nibbles(q, N, K, /*nthreads=*/4);
  auto wsz = PackW4A16Scales(scale, zero, N, K, group);

  const int ntiles = N / 16, kblocks = K / 64, groups_per_row = K / group;
  for (int t = 0; t < ntiles; ++t) {
    for (int kb = 0; kb < kblocks; ++kb) {
      for (int lane = 0; lane < 32; ++lane) {
        const int row = DecodeRow(t, lane);
        for (int s = 0; s < 4; ++s) {
          const size_t idx = ((static_cast<size_t>(t) * kblocks + kb) * 32 + lane) * 4 + s;
          const uint32_t dword = wq[idx];
          for (int e = 0; e < 8; ++e) {
            const uint32_t nibble = (dword >> (4 * NibblePos(e))) & 0xFu;
            const uint8_t decoded_q = static_cast<uint8_t>(nibble ^ 0x8u);  // kernel's XOR-by-8
            const int k = DecodeK(kb, s, e, lane);
            const uint8_t expect_q = q[static_cast<size_t>(row) * K + k];
            if (decoded_q != expect_q) {
              std::fprintf(stderr, "w4a16 code mismatch row=%d k=%d got=%d want=%d\n", row, k,
                           decoded_q, expect_q);
              ++g_failures;
            }
            const int g = k / group;
            const size_t gidx = static_cast<size_t>(row) * groups_per_row + g;
            const int r = lane & 15;
            const size_t wsz_idx = (static_cast<size_t>(t) * groups_per_row + g) * 16 + r;
            const uint16_t sc16 = static_cast<uint16_t>(wsz[wsz_idx] & 0xFFFFu);
            const uint16_t nz16 = static_cast<uint16_t>(wsz[wsz_idx] >> 16);
            const uint16_t expect_sc16 = r4dx::core::FloatToF16(scale[gidx]);
            const uint16_t expect_nz16 =
                r4dx::core::FloatToF16(-(1024.0f + static_cast<float>(zero[gidx])));
            if (sc16 != expect_sc16 || nz16 != expect_nz16) {
              std::fprintf(stderr, "w4a16 wsz mismatch row=%d g=%d\n", row, g);
              ++g_failures;
            }
          }
        }
      }
    }
  }
  std::printf("w4a16 kernel-literal decode: N=%d K=%d group=%d checked, failures so far=%d\n", N, K,
              group, g_failures);
}

}  // namespace

int main() {
  const int N = 64, K = 512;  // N multiple of 16; K multiple of 64 and 32
  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> w(static_cast<size_t>(N) * K);
  for (auto& v : w) v = dist(rng);

  // The default w4a16 group. The decode below re-derives the kernel's own indexing from
  // R4D_GEMM_W4_GROUP-parameterized source, so it is exactly the group-64 wsz stride
  // (`t*nsz + kbase/GROUP`, `bpg = GROUP/64`) that has to be gated here.
  CheckW4A16(w, N, K, 64);
  // Per-tensor w4a16 group 32 (docs/quant2.md section 5, --w4a16-group-rule): the kernel's G = 32
  // body splits each packed 64-K block into two groups -- k steps 0,1 read (scale, zero) dword
  // `t*nsz + kbase/32 + 2b`, k steps 2,3 dword `... + 2b + 1` -- which is k / 32 for every k of the
  // block, i.e. the same `(t * K/g + k/g) * 16 + r` index CheckW4A16 decodes by.
  CheckW4A16(w, N, K, 32);

  if (g_failures != 0) {
    std::fprintf(stderr, "FAIL: %d kernel-literal decode mismatches\n", g_failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}

