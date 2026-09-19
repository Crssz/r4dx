// Kernel-literal decode test (review finding, major: "the gating tests do not gate the thing that
// matters" -- test_pack_bytes.cpp only checks the C++ packer against fixtures produced by the same
// author's Python references, so a permutation bug present in BOTH would pass silently, and the
// only check against the real GEMM kernels' actual pointer arithmetic (tools/convert_ref/
// kernel_crosscheck.py) is a manual GPU script nobody's build runs).
//
// This test decodes the packed bytes by following each kernel's OWN indexing (re-derived here from
// third_party/libr4d/r4d_gemm_w4a16_nt_m64.hip, r4d_gemm_w4a8_nt_m64.hip and
// r4d_gemm_mxfp4a8_nt_m64.hip source, not by inverting PackW4Nibbles/PackMxfp4Wq's own loops) and
// asserts the decode recovers exactly the quantizer's intended (row,k) code and exactly the
// packer's intended f16 scale/zero bit pattern. CPU-only, no HIP device needed, so it runs on every
// build -- the permanent, always-on replacement for the gap the review flagged.
//
// w4a16 / w4a8 (identical fragment layout, both packed by PackW4Nibbles):
//   dword index for (tile t, k-block kb of 64, lane 0..31, k-step s of 0..3) is
//     ((t*(K/64)+kb)*32+lane)*4+s
//   row = t*16 + (lane&15); within that dword, nibble at bit position 4*nibble_pos holds element
//   e (0..7) where nibble_pos = 2e (e<4) or 2(e-4)+1 (e>=4) -- r4d_gemm_w4a16_nt_m64.hip's literal
//   "nibble 2e (e<4) / 2(e-4)+1 (e>=4)" statement -- and that element's k is
//     k = (kb*4+s)*16 + 8*(e>>2) + 4*(lane>>4) + (e&3)
//   w4a16's dequant XORs the stored nibble by 8 to recover the 0..15 code; w4a8's dequant8 reads
//   the nibble as a 4-bit two's-complement signed value ((nibble<8)?nibble:nibble-16), which is
//   algebraically q-8 for the same offset-binary q PackW4Nibbles stored (verified in this file's
//   CheckW4A8, not assumed).
//
// mxfp4 (packed by PackMxfp4Wq/PackMxfp4Ws, per r4d_gemm_mxfp4a8_nt_m64.hip's Wp/Ws indexing):
//   a lane's 4-byte word for (n-tile nt, k-step ks of 16) is at byte offset
//     ((nt*ksteps+ks)*32+lane)*4
//   row = nt*16 + (lane&15); byte j (0..3) of that word holds e2m1 codes for
//     k = ks*16 + 8*(lane>>4) + 2*j (low nibble) and +1 (high nibble)
//   and the kernel's dequant is sign*magnitude*2^(escale-127) where escale = Ws[(k/group)*N+row]
//   (the kernel folds this through a per-row reference exponent Wref and a clamped exponent
//   difference dsh purely as a perf trick -- dsh = Wref[row]-escale rearranges to the same
//   2^(escale-127) whenever dsh is not clamped, which review's own measurement on the real
//   checkpoint confirms is always the case here, max dsh=5 of a 0..15 clamp range).
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/quant_int4.hpp"
#include "r4dx_convert/quant_mxfp4.hpp"

namespace {

using namespace r4dx_convert;

int g_failures = 0;

void Expect(bool cond, const char* what) {
  if (!cond) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++g_failures;
  }
}

// ---- w4a16 / w4a8 shared index math (independently derived from the kernels, see file header) --

int DecodeRow(int t, int lane) { return t * 16 + (lane & 15); }

int DecodeK(int kb, int s, int e, int lane) {
  return (kb * 4 + s) * 16 + 8 * (e >> 2) + 4 * (lane >> 4) + (e & 3);
}

int NibblePos(int e) { return (e < 4) ? (2 * e) : (2 * (e - 4) + 1); }

void CheckW4A16(const std::vector<float>& w, int N, int K) {
  std::vector<uint8_t> q, zero;
  std::vector<float> scale;
  QuantizeInt4Asymmetric(w.data(), N, K, kInt4Group, /*nthreads=*/4, q, scale, zero);
  auto wq = PackW4Nibbles(q, N, K, /*nthreads=*/4);
  auto wsz = PackW4A16Scales(scale, zero, N, K, kInt4Group);

  const int ntiles = N / 16, kblocks = K / 64, groups_per_row = K / kInt4Group;
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
            const int g = k / kInt4Group;
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
  std::printf("w4a16 kernel-literal decode: N=%d K=%d checked, failures so far=%d\n", N, K,
              g_failures);
}

void CheckW4A8(const std::vector<float>& w, int N, int K) {
  std::vector<uint8_t> q;
  std::vector<float> scale;
  QuantizeInt4SymmetricPinned8(w.data(), N, K, kInt4Group, /*nthreads=*/4, q, scale);
  auto wq = PackW4Nibbles(q, N, K, /*nthreads=*/4);
  auto ws = PackW4A8Scales(scale, N, K, kInt4Group);

  const int ntiles = N / 16, kblocks = K / 64, groups_per_row = K / kInt4Group;
  for (int t = 0; t < ntiles; ++t) {
    for (int kb = 0; kb < kblocks; ++kb) {
      for (int lane = 0; lane < 32; ++lane) {
        const int row = DecodeRow(t, lane);
        for (int s = 0; s < 4; ++s) {
          const size_t idx = ((static_cast<size_t>(t) * kblocks + kb) * 32 + lane) * 4 + s;
          const uint32_t dword = wq[idx];
          for (int e = 0; e < 8; ++e) {
            const uint32_t nibble = (dword >> (4 * NibblePos(e))) & 0xFu;
            // dequant8's literal two's-complement read: nibble<<4 as a signed byte, /16.
            const int8_t as_i8 = static_cast<int8_t>(nibble << 4);
            const int decoded_signed = as_i8 / 16;
            const int k = DecodeK(kb, s, e, lane);
            const int expect_signed =
                static_cast<int>(q[static_cast<size_t>(row) * K + k]) - 8;
            if (decoded_signed != expect_signed) {
              std::fprintf(stderr, "w4a8 code mismatch row=%d k=%d got=%d want=%d\n", row, k,
                           decoded_signed, expect_signed);
              ++g_failures;
            }
            const int g = k / kInt4Group;
            const size_t gidx = static_cast<size_t>(row) * groups_per_row + g;
            const int r = lane & 15;
            const size_t ws_idx = (static_cast<size_t>(t) * groups_per_row + g) * 16 + r;
            const uint16_t sc16 = static_cast<uint16_t>(ws[ws_idx] & 0xFFFFu);
            const uint16_t hi16 = static_cast<uint16_t>(ws[ws_idx] >> 16);
            const uint16_t expect_sc16 = r4dx::core::FloatToF16(scale[gidx]);
            if (sc16 != expect_sc16 || hi16 != 0) {
              std::fprintf(stderr, "w4a8 ws mismatch row=%d g=%d\n", row, g);
              ++g_failures;
            }
          }
        }
      }
    }
  }
  std::printf("w4a8 kernel-literal decode: N=%d K=%d checked, failures so far=%d\n", N, K,
              g_failures);
}

void CheckMxfp4(const std::vector<float>& w, int N, int K) {
  Mxfp4Quantized mq = QuantizeMxfp4(w.data(), N, K, kMxfp4Group, /*nthreads=*/4);
  auto wq = PackMxfp4Wq(mq.packed, N, K, /*nthreads=*/4);
  auto ws = PackMxfp4Ws(mq.escale, N, K, kMxfp4Group);

  const int ntiles = N / 16, ksteps = K / 16, groups_per_row = K / kMxfp4Group;
  for (int nt = 0; nt < ntiles; ++nt) {
    for (int ks = 0; ks < ksteps; ++ks) {
      for (int lane = 0; lane < 32; ++lane) {
        const int r = lane & 15, h = lane >> 4;
        const int row = nt * 16 + r;
        const size_t word_base = (static_cast<size_t>(nt) * ksteps + ks) * 32 * 4 + lane * 4;
        for (int j = 0; j < 4; ++j) {
          const uint8_t byte = wq[word_base + j];
          for (int half = 0; half < 2; ++half) {
            const uint8_t code = half == 0 ? (byte & 0xF) : ((byte >> 4) & 0xF);
            const int k = ks * 16 + 8 * h + 2 * j + half;
            const uint8_t src_byte = mq.packed[static_cast<size_t>(row) * (K / 2) + k / 2];
            const uint8_t expect_code = (k % 2 == 0) ? (src_byte & 0xF) : ((src_byte >> 4) & 0xF);
            if (code != expect_code) {
              std::fprintf(stderr, "mxfp4 code mismatch row=%d k=%d got=%d want=%d\n", row, k,
                           code, expect_code);
              ++g_failures;
            }
            const int g = k / kMxfp4Group;
            const uint8_t escale = ws[static_cast<size_t>(g) * N + row];
            const uint8_t expect_escale = mq.escale[static_cast<size_t>(row) * groups_per_row + g];
            if (escale != expect_escale) {
              std::fprintf(stderr, "mxfp4 ws mismatch row=%d g=%d\n", row, g);
              ++g_failures;
            }
            // Value check: sign*magnitude*2^(escale-127) (see file header for the dsh-fold
            // algebra) reconstructs the same code this row/k already produced, i.e. decode(pack(x))
            // has not silently swapped an element.
            const bool neg = code & 0x8;
            const float mag = kE2M1Magnitude[code & 0x7];
            const float value = (neg ? -1.0f : 1.0f) * mag * std::ldexp(1.0f, escale - 127);
            const bool expect_neg = expect_code & 0x8;
            const float expect_mag = kE2M1Magnitude[expect_code & 0x7];
            const float expect_value =
                (expect_neg ? -1.0f : 1.0f) * expect_mag * std::ldexp(1.0f, expect_escale - 127);
            if (value != expect_value) {
              std::fprintf(stderr, "mxfp4 value mismatch row=%d k=%d\n", row, k);
              ++g_failures;
            }
          }
        }
      }
    }
  }
  std::printf("mxfp4 kernel-literal decode: N=%d K=%d checked, failures so far=%d\n", N, K,
              g_failures);
}

}  // namespace

int main() {
  const int N = 64, K = 512;  // N multiple of 16; K multiple of 64, 128 and 32
  std::mt19937 rng(7);
  std::normal_distribution<float> dist(0.0f, 1.0f);
  std::vector<float> w(static_cast<size_t>(N) * K);
  for (auto& v : w) v = dist(rng);

  CheckW4A16(w, N, K);
  CheckW4A8(w, N, K);
  CheckMxfp4(w, N, K);

  if (g_failures != 0) {
    std::fprintf(stderr, "FAIL: %d kernel-literal decode mismatches\n", g_failures);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
