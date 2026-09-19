// r4dx_convert::quant_mxfp4 -- the OCP MXFP4 (e2m1 weight + E8M0 per-32 scale) quantizer and
// packer for r4d_gemm_mxfp4a8_nt_m64.
//
// GROUND TRUTH: third_party/libr4d/r4d_gemm_mxfp4a8_nt_m64.hip (dequant math: final value is
// e2m1_magnitude * sign * 2^(E8M0_block-127), read via a per-row reference exponent Wref[n] =
// max_k E8M0[n][k] and a folded difference dsh = Wref[n]-E8M0[blk][n]) and
// C:\Users\user\dev\libr4d\mxfp4_layout.py (permute_w -- the fragment permutation, re-derived
// here as explicit index loops rather than the tensor-reshape form so the C++ has no library
// dependency).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "r4dx_convert/quant_int4.hpp"  // RoundHalfAwayFromZero, ClampInt
#include "r4dx_convert/threadpool.hpp"

namespace r4dx_convert {

// K per E8M0 exponent, fixed by the OCP MX format (R4D_GEMM_MXFP4_GROUP in
// r4d_gemm_mxfp4a8_nt_m64.hip).
inline constexpr int kMxfp4Group = 32;

// The eight OCP E2M1 magnitude codes, index = the 3 low bits of the packed nibble; bit 3 is sign.
inline constexpr float kE2M1Magnitude[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

inline uint8_t EncodeE2M1(float v) {
  const bool neg = v < 0.0f || (v == 0.0f && std::signbit(v));
  const float av = std::fabs(v);
  int best = 0;
  float best_d = std::fabs(av - kE2M1Magnitude[0]);
  for (int i = 1; i < 8; ++i) {
    const float d = std::fabs(av - kE2M1Magnitude[i]);
    // Nearest code; on an exact tie (equidistant between two grid points, e.g. |v|==0.25 is
    // equidistant from codes 0 and .5) keep the lower index, matching tools/convert_ref/
    // mxfp4_ref.py's `np.argmin` (first-minimum) tie-break so the two implementations land on the
    // same code from the same input -- byte-exactness (tests/convert/test_pack_bytes.cpp) depends
    // on this.
    //
    // Review finding (minor): this is round-half-toward-zero, not the OCP-recommended round-half-
    // to-even, so it carries a small systematic bias toward zero at exact midpoints (which DO occur
    // in practice: bf16 source weights with a power-of-two group scale make w/scale exactly
    // representable). Deliberately left as-is per the review's own suggested fallback -- changing
    // it requires updating mxfp4_ref.py and regenerating tests/convert/fixtures/ in lockstep, and
    // the bias is a rounding-mode choice, not a correctness bug the loader/kernel cares about.
    if (d < best_d) {
      best_d = d;
      best = i;
    }
  }
  return static_cast<uint8_t>((neg ? 0x8u : 0u) | static_cast<uint32_t>(best));
}

struct Mxfp4Quantized {
  std::vector<uint8_t> packed;  // [N, K/2] checkpoint-order e2m1, byte = elem(2b) | elem(2b+1)<<4
  std::vector<uint8_t> escale;  // [N, K/group] E8M0 raw byte (biased exponent) per (row, group)
  std::vector<uint8_t> wref;    // [N] max escale over the row's groups
};

// Shared e8m0 scale = 2^ceil(log2(amax/6)) (task brief's documented rounding: 6 is the largest
// finite e2m1 magnitude, so this is the smallest power-of-two group scale that keeps amax
// representable without saturating -- the OCP-recommended "round the scale up" convention rather
// than round-to-nearest, so a group's amax never clips). A group whose amax is exactly 0 gets raw
// exponent 0 (scale 2^-127): every element quantizes to code 0 regardless, and it never wins the
// row's Wref max against any group that has real content.
inline Mxfp4Quantized QuantizeMxfp4(const float* w, int N, int K, int group, int nthreads) {
  RequireDivisible(K, group, "K", "QuantizeMxfp4");
  const int groups_per_row = K / group;
  Mxfp4Quantized out;
  out.packed.assign(static_cast<size_t>(N) * (K / 2), 0);
  out.escale.assign(static_cast<size_t>(N) * groups_per_row, 0);
  out.wref.assign(static_cast<size_t>(N), 0);

  ParallelFor(0, N, nthreads, [&](int64_t r0, int64_t r1) {
    for (int64_t row = r0; row < r1; ++row) {
      const float* wr = w + row * K;
      uint8_t row_max_raw = 0;
      for (int g = 0; g < groups_per_row; ++g) {
        const float* wg = wr + static_cast<int64_t>(g) * group;
        float amax = 0.0f;
        for (int k = 0; k < group; ++k) amax = std::max(amax, std::fabs(wg[k]));

        int raw;
        float scale;
        if (amax <= 0.0f) {
          raw = 0;
          scale = std::ldexp(1.0f, -127);
        } else {
          const int e = static_cast<int>(std::ceil(std::log2(amax / 6.0)));
          raw = ClampInt(e + 127, 0, 254);  // 255 is the OCP E8M0 NaN encoding, never emitted
          scale = std::ldexp(1.0f, raw - 127);
        }
        out.escale[static_cast<size_t>(row) * groups_per_row + g] = static_cast<uint8_t>(raw);
        row_max_raw = std::max(row_max_raw, static_cast<uint8_t>(raw));

        uint8_t* packed_row = out.packed.data() + row * static_cast<int64_t>(K / 2);
        for (int k = 0; k < group; ++k) {
          const int kk = g * group + k;
          const uint8_t code = EncodeE2M1(wg[k] / scale);
          uint8_t& byte = packed_row[kk / 2];
          if (kk % 2 == 0) {
            byte = static_cast<uint8_t>((byte & 0xF0u) | (code & 0x0Fu));
          } else {
            byte = static_cast<uint8_t>((byte & 0x0Fu) | ((code & 0x0Fu) << 4));
          }
        }
      }
      out.wref[row] = row_max_raw;
    }
  });
  return out;
}

// Checkpoint-order [N, K/2] packed e2m1 -> fragment order (mxfp4_layout.py::permute_w, expanded
// to explicit loops): slot l (0..31) of tile (nt, ks) is packed[16*nt + (l&15)][8*ks + 4*(l>>4)
// .. +4] (byte offsets), i.e. one lane's 4-byte (8-element) slice; 32 lanes = 128 contiguous
// output bytes per (n-tile, k-step of 16 elements).
inline std::vector<uint8_t> PackMxfp4Wq(const std::vector<uint8_t>& packed, int N, int K,
                                         int nthreads) {
  RequireDivisible(N, 16, "N", "PackMxfp4Wq");
  RequireDivisible(K, 16, "K", "PackMxfp4Wq");
  const int ntiles = N / 16;
  const int ksteps = K / 16;  // 16 elements = 8 bytes per step
  std::vector<uint8_t> wq(static_cast<size_t>(ntiles) * ksteps * 32 * 4);

  ParallelFor(0, ntiles, nthreads, [&](int64_t t0, int64_t t1) {
    for (int64_t nt = t0; nt < t1; ++nt) {
      for (int ks = 0; ks < ksteps; ++ks) {
        const size_t base = (static_cast<size_t>(nt) * ksteps + ks) * 32 * 4;
        for (int l = 0; l < 32; ++l) {
          const int r = l & 15, h = l >> 4;
          const int row = static_cast<int>(nt) * 16 + r;
          const size_t src = static_cast<size_t>(row) * (K / 2) + ks * 8 + 4 * h;
          uint8_t* dst = wq.data() + base + static_cast<size_t>(l) * 4;
          dst[0] = packed[src + 0];
          dst[1] = packed[src + 1];
          dst[2] = packed[src + 2];
          dst[3] = packed[src + 3];
        }
      }
    }
  });
  return wq;
}

// Ws: [K/group][N] row-major (r4d_registry.hip's mxfp4 shape comment / r4d_gemm_mxfp4a8_nt_m64.hip
// `Ws[blk*N+n]`) -- transpose of `escale`'s [N, K/group] storage.
inline std::vector<uint8_t> PackMxfp4Ws(const std::vector<uint8_t>& escale, int N, int K,
                                         int group) {
  const int groups_per_row = K / group;
  std::vector<uint8_t> ws(static_cast<size_t>(groups_per_row) * N);
  for (int n = 0; n < N; ++n) {
    for (int g = 0; g < groups_per_row; ++g) {
      ws[static_cast<size_t>(g) * N + n] = escale[static_cast<size_t>(n) * groups_per_row + g];
    }
  }
  return ws;
}

}  // namespace r4dx_convert
