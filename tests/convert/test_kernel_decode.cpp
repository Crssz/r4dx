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
#include <cstring>

#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/quant_int8.hpp"

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

// ---- w8a16 (docs/w8a16.md 3.2, 3.4; the kernel itself is src/kernels/src/w8a16_gemm.hip) ---------
//
// Decoded from the f16 WMMA B-fragment map, NOT from the packer's closed-form (and not from
// DequantW8a16): lane l, element e = 0..7 of k-step s holds W[n0 + (l&15)][16s + 8*(e>>2) + 4*(l>>4) +
// (e&3)]. A lane loads ONE b128 (4 dwords) per 32-K block = 2 k-steps: dword j = 2*(s&1) + (e>>2),
// byte (e&3) of it holds element e. The b128 of (row tile t, 32-K block kb, lane) is at uint4 index
// (t*(K/32) + kb)*32 + lane. Scales: u16 index (t*(K/G) + k/G)*16 + (l&15). The kernel widens a byte b
// as the f16 0x6400|b (= 1024 + b exactly) minus 1152, i.e. the code b - 128.
void CheckW8a16(const std::vector<float>& w, int N, int K, int group) {
  std::vector<uint8_t> q;
  std::vector<uint16_t> s16;
  QuantizeInt8Symmetric(w.data(), N, K, group, /*nthreads=*/4, q, s16);
  const auto wq = PackW8Bytes(q, N, K, 4);
  const auto ws = PackW8a16Scales(s16, N, K, group);
  Expect(wq.size() == static_cast<size_t>(N) * K, "w8a16 wq size");
  Expect(ws.size() == static_cast<size_t>(N) * (K / group), "w8a16 ws size");

  const int ntiles = N / 16, kblocks = K / 32, gpr = K / group;
  size_t checked = 0;
  for (int t = 0; t < ntiles; ++t)
    for (int kb = 0; kb < kblocks; ++kb)
      for (int lane = 0; lane < 32; ++lane) {
        const size_t uint4_idx = (static_cast<size_t>(t) * kblocks + kb) * 32 + lane;
        uint32_t dw[4];
        std::memcpy(dw, wq.data() + uint4_idx * 16, 16);  // one global_load_b128, little endian
        const int row = t * 16 + (lane & 15);
        for (int s = 0; s < 2; ++s)
          for (int e = 0; e < 8; ++e) {
            const uint32_t d = dw[2 * s + (e >> 2)];
            const uint8_t byte = static_cast<uint8_t>((d >> (8 * (e & 3))) & 0xFFu);
            const int k = 16 * (2 * kb + s) + 8 * (e >> 2) + 4 * (lane >> 4) + (e & 3);
            // magic-exponent widening: f16 bits 0x6400 | byte is 1024 + byte; minus 1152 is the code
            const float widened =
                r4dx::core::F16ToFloat(static_cast<uint16_t>(0x6400u | byte)) - 1152.0f;
            const float want_code = static_cast<float>(static_cast<int>(q[static_cast<size_t>(row) * K + k]) - 128);
            if (byte != q[static_cast<size_t>(row) * K + k] || widened != want_code) {
              std::fprintf(stderr, "w8a16 code mismatch row=%d k=%d byte=%d want=%d widened=%g\n", row,
                           k, byte, q[static_cast<size_t>(row) * K + k], widened);
              ++g_failures;
            }
            const uint16_t sc = ws[(static_cast<size_t>(t) * gpr + k / group) * 16 + (lane & 15)];
            if (sc != s16[static_cast<size_t>(row) * gpr + k / group]) {
              std::fprintf(stderr, "w8a16 scale mismatch row=%d k=%d\n", row, k);
              ++g_failures;
            }
            ++checked;
          }
      }
  Expect(checked == static_cast<size_t>(N) * K, "w8a16 fragment decode visited every (row, k) once");
  // The widening trick is exact for every byte.
  for (int b = 0; b < 256; ++b)
    Expect(r4dx::core::F16ToFloat(static_cast<uint16_t>(0x6400u | b)) - 1152.0f == static_cast<float>(b - 128),
           "w8a16 magic exponent widening is exact for every byte");
  // And the reference dequant agrees with the fragment-decoded weights.
  const auto dq = DequantW8a16(wq, ws, N, K, group);
  for (size_t i = 0; i < dq.size(); ++i) {
    const int row = static_cast<int>(i / K), k = static_cast<int>(i % K);
    const float want = static_cast<float>(static_cast<int>(q[i]) - 128) *
                       (r4dx::core::F16ToFloat(s16[static_cast<size_t>(row) * gpr + k / group]) *
                        (1.0f / 4096.0f));
    if (dq[i] != want) {
      ++g_failures;
      std::fprintf(stderr, "w8a16 DequantW8a16 mismatch at %zu\n", i);
      break;
    }
  }
  std::printf("w8a16 kernel-literal decode: N=%d K=%d group=%d checked %zu, failures so far=%d\n", N,
              K, group, checked, g_failures);
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

  // w8a16 at both instantiated groups (K = 512: 16 blocks of 32, 4 groups of 128 / 8 of 64).
  CheckW8a16(w, N, K, 128);
  CheckW8a16(w, N, K, 64);

  if (g_failures != 0) {
    std::fprintf(stderr, "FAIL: %d kernel-literal decode mismatches\n", g_failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}

