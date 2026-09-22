// r4dx_convert::quant_int4 -- the w4a16 / w4a8 quantizer and packer.
//
// GROUND TRUTH for the byte layout (cited in full in docs/container-format.md and re-derived here
// from the kernel sources, not assumed):
//   - third_party/libr4d/r4d_gemm_w4a16_nt_m64.hip ("LAYOUT" / "QUANTIZER" comments, dequant())
//   - third_party/libr4d/r4d_gemm_w4a8_nt_m64.hip (dequant8(), the Ws pointer type)
//   - C:\Users\user\dev\vllm-radiance\radiance_w4.py pack() -- the reference packer for a
//     SYMMETRIC grid; this file generalizes it to an ASYMMETRIC per-group zero (task requirement)
//     and re-derives, element by element, that radiance's `_KOFF` table and nibble order are
//     exactly r4d_gemm_w4a16_nt_m64.hip's fragment map, independent of symmetric-vs-asymmetric.
//
// ---- Why w4a16 and w4a8 do NOT share wq bytes here (a deliberate deviation from
// docs/container-format.md, flagged for the loader/doc owners) ------------------------------
//
// r4d_gemm_w4a16_nt_m64's dequant() always XORs the stored nibble with 0x8 before use (
// R4D_GEMM_W4_TWOS=1, the flag this build compiles with), i.e. it decodes
// n = stored_nibble ^ 8, then computes w = scale*n - scale*(1024+zero) worth of arithmetic that
// nets out to w = scale*(n - zero), where `zero` is whatever this converter wrote into `wsz`'s
// high 16 bits. `zero` is free to be any of 0..15 -- that is the whole point of storing it.
//
// r4d_gemm_w4a8_nt_m64's dequant8() has NO zero input at all: it reads the stored nibble directly
// as a two's-complement nibble (`nibble<<4` as a signed byte, /16 at the store). That is only
// correct when the value the stored nibble represents was produced with zero PINNED to 8 (the
// kernel comment says so explicitly: "one quantised weight feeds both... when its integer zero
// point is the constant 8"). XOR-by-8 and "subtract 8 mod 16" are the identical operation on a
// 4-bit code (verified: (q^8) == (q-8)&0xF for every q in 0..15), so sharing bytes is EXACT only
// when w4a16's zero happens to equal 8 for that group.
//
// This converter's w4a16 quantizer computes a genuine per-(row,group) asymmetric zero from the
// data (0..15, not pinned) -- that is what the task brief asks for and it is measurably more
// accurate than a symmetric grid. So w4a16.wq and w4a8.wq are quantized SEPARATELY: w4a16 with a
// free zero, w4a8 with zero pinned to 8 (a plain symmetric grid, scale = amax/7, exactly
// radiance_w4.py's `_grid`). Both still use the identical fragment permutation (PackW4Nibbles
// below), because that part of the byte layout has nothing to do with the quantization grid.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/threadpool.hpp"

namespace r4dx_convert {

// ---- group sizes: one constant PER KERNEL, each driven by that kernel's own build flag ---------
//
// These two used to be a single `kInt4Group = 128`, which was only correct because both kernels
// happened to be built with 128. They are separate knobs in libr4d (R4D_GEMM_W4_GROUP for w4a16,
// R4D_GEMM_W4A8_GROUP for w4a8) and they are separate constants here, each asserted against its
// own kernel export at startup (src/convert/main.cpp ValidateKernelGroupSizes).
//
// w4a16's group is a BUILD OPTION: the CMake cache variable R4DX_W4A16_GROUP (root
// CMakeLists.txt, default 128) is passed both to the kernel as -DR4D_GEMM_W4_GROUP and to this
// header as -DR4DX_W4A16_GROUP (src/convert/CMakeLists.txt), so one switch moves both sides at
// once. The kernel packs R4D_GEMM_W4_KPB=64 contiguous K per weight block and derives
// `bpg = R4D_GEMM_W4_GROUP / R4D_GEMM_W4_KPB`, so the group must be a multiple of 64 -- i.e. 64 is
// the only value below the 128 default the kernel accepts as-is. Smaller group = more (scale,zero)
// dwords per row = more bits per weight: 4 + 32/group bits, so 4.25 at 128 and 4.5 at 64.
#ifndef R4DX_W4A16_GROUP
#define R4DX_W4A16_GROUP 128
#endif
inline constexpr int kW4A16Group = R4DX_W4A16_GROUP;
static_assert(kW4A16Group > 0 && kW4A16Group % 64 == 0,
              "R4DX_W4A16_GROUP must be a positive multiple of R4D_GEMM_W4_KPB (64)");

// w4a8's group is NOT an option: third_party/CMakeLists.txt pins the kernel to
// -DR4D_GEMM_W4A8_GROUP=128 (its in-kernel default is 256). Change one and the startup assert in
// src/convert/main.cpp fires naming both numbers.
inline constexpr int kW4A8Group = 128;

// k-offset (within a 16-wide WMMA step, before the lane's own 4*(lane>>4) term) of fragment
// element e = 0..7, i.e. the inverse of r4d_gemm_w4a16_nt_m64.hip's "nibble 2e (e<4) / 2(e-4)+1
// (e>=4)" statement: nibble index i (0..7, the dword's i-th 4-bit lane) holds element
// e = kNibbleToElement[i], and that element's k-offset is kElementKOff[e]. Composed together this
// is exactly radiance_w4.py's `_KOFF = (0, 8, 1, 9, 2, 10, 3, 11)` table indexed by nibble i --
// re-derived independently here from the kernel's dequant() so the two sources cross-check.
inline constexpr int kNibbleKOffset[8] = {0, 8, 1, 9, 2, 10, 3, 11};

// Round-half-away-from-zero in float32, matching tools/convert_ref/w4_ref.py's `_round_haz`
// bit-for-bit (both are single IEEE-754 add/floor or subtract/ceil, no library rounding-mode
// dependence) so the C++ and Python quantizers land on the identical integer code from the same
// input -- the byte-exactness gate (task item 3) depends on this.
inline int32_t RoundHalfAwayFromZero(float x) {
  return x >= 0.0f ? static_cast<int32_t>(std::floor(x + 0.5f))
                    : static_cast<int32_t>(std::ceil(x - 0.5f));
}

inline int ClampInt(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

// Review finding (minor): none of the quantizers/packers validated their shape preconditions --
// `K / group` and `N / 16` just truncate, so K not a multiple of `group` (or 64, for the packer's
// k-block) silently drops the tail of every row as zeros instead of failing. No shape in this
// checkpoint hits it (every quantized K is 5120/6144/17408, all multiples of 128 and 64; every N
// is a multiple of 16), but the converter is generic, so fail loudly instead of writing a
// plausible-looking container full of zeros.
inline void RequireDivisible(int64_t value, int64_t divisor, const char* what,
                              const char* tensor_hint) {
  if (divisor <= 0 || value % divisor != 0) {
    throw std::runtime_error(std::string("r4dx_convert: ") + what + " (" + std::to_string(value) +
                              ") is not divisible by " + std::to_string(divisor) + " for " +
                              tensor_hint);
  }
}

// ---- quantizers -------------------------------------------------------------------------------

// Asymmetric, per (row, `group` contiguous K): scale = (max-min)/15 (floored at 1e-12 so a
// constant group never divides by zero), zero = round(-min/scale) clipped to 0..15,
// q = clamp(round(w/scale) + zero, 0, 15). w ~= scale*(q-zero). This is w4a16's grid.
//
// `w` is row-major [N,K] float32. Outputs: `q` [N*K] in 0..15, `scale`/`zero` [N*(K/group)],
// group-major within a row (group index g covers k in [g*group, (g+1)*group)).
inline void QuantizeInt4Asymmetric(const float* w, int N, int K, int group, int nthreads,
                                    std::vector<uint8_t>& q, std::vector<float>& scale,
                                    std::vector<uint8_t>& zero) {
  RequireDivisible(K, group, "K", "QuantizeInt4Asymmetric");
  const int groups_per_row = K / group;
  q.assign(static_cast<size_t>(N) * K, 0);
  scale.assign(static_cast<size_t>(N) * groups_per_row, 0.0f);
  zero.assign(static_cast<size_t>(N) * groups_per_row, 0);

  ParallelFor(0, N, nthreads, [&](int64_t r0, int64_t r1) {
    for (int64_t row = r0; row < r1; ++row) {
      const float* wr = w + row * K;
      for (int g = 0; g < groups_per_row; ++g) {
        const float* wg = wr + static_cast<int64_t>(g) * group;
        float wmin = wg[0], wmax = wg[0];
        for (int k = 1; k < group; ++k) {
          wmin = std::min(wmin, wg[k]);
          wmax = std::max(wmax, wg[k]);
        }
        const float range = wmax - wmin;
        // Review finding (minor): a constant-valued group used to hit `sc = 1e-12/15`, which
        // FloatToF16 flushes to 0 and zp saturates to 0 or 15 -- the group silently dequantizes to
        // exactly 0 regardless of what the constant actually was, rather than reconstructing it.
        // Special-case it: a genuine all-zero group still gets scale 0 (nothing to represent), but
        // a nonzero constant gets an exact reconstruction (q=1,zero=0 for w>=0 or q=14,zero=15 for
        // w<0, scale=|w|, so scale*(q-zero) == w for every element in the group).
        float sc;
        int zp;
        bool degenerate_zero = false;
        if (range <= 0.0f) {
          if (wmax == 0.0f) {
            sc = 0.0f;
            zp = 0;
            degenerate_zero = true;
          } else {
            sc = std::fabs(wmax);
            zp = (wmax >= 0.0f) ? 0 : 15;
          }
        } else {
          // Keep the original 1e-12 floor for a tiny-but-nonzero range (a near-constant group, as
          // opposed to the exactly-constant case handled above) so a division by a near-zero range
          // can't itself overflow scale to inf.
          sc = std::max(range, 1e-12f) / 15.0f;
          zp = ClampInt(RoundHalfAwayFromZero(-wmin / sc), 0, 15);
        }
        const size_t gidx = static_cast<size_t>(row) * groups_per_row + g;
        scale[gidx] = sc;
        zero[gidx] = static_cast<uint8_t>(zp);
        uint8_t* qg = q.data() + row * static_cast<int64_t>(K) + static_cast<int64_t>(g) * group;
        for (int k = 0; k < group; ++k) {
          const int qi =
              degenerate_zero ? 0 : ClampInt(RoundHalfAwayFromZero(wg[k] / sc) + zp, 0, 15);
          qg[k] = static_cast<uint8_t>(qi);
        }
      }
    }
  });
}

// Symmetric, zero pinned to 8: scale = max(|w|)/7 per (row,group), q_signed = clamp(round(w/scale),
// -8,7), q = q_signed+8 (0..15, the "offset binary" code PackW4Nibbles expects). This is w4a8's
// grid -- identical formula to radiance_w4.py's `_grid` (without its clip search: the task's
// self-test uses small random matrices where a single min/max/7 grid already round-trips inside
// the harness's 2e-2 tolerance, and the clip search is a serving-time quality knob, not a byte
// layout requirement).
inline void QuantizeInt4SymmetricPinned8(const float* w, int N, int K, int group, int nthreads,
                                          std::vector<uint8_t>& q, std::vector<float>& scale) {
  RequireDivisible(K, group, "K", "QuantizeInt4SymmetricPinned8");
  const int groups_per_row = K / group;
  q.assign(static_cast<size_t>(N) * K, 0);
  scale.assign(static_cast<size_t>(N) * groups_per_row, 0.0f);

  ParallelFor(0, N, nthreads, [&](int64_t r0, int64_t r1) {
    for (int64_t row = r0; row < r1; ++row) {
      const float* wr = w + row * K;
      for (int g = 0; g < groups_per_row; ++g) {
        const float* wg = wr + static_cast<int64_t>(g) * group;
        float amax = std::fabs(wg[0]);
        for (int k = 1; k < group; ++k) amax = std::max(amax, std::fabs(wg[k]));
        const float sc = std::max(amax, 1e-12f) / 7.0f;
        scale[static_cast<size_t>(row) * groups_per_row + g] = sc;
        uint8_t* qg = q.data() + row * static_cast<int64_t>(K) + static_cast<int64_t>(g) * group;
        for (int k = 0; k < group; ++k) {
          const int qs = ClampInt(RoundHalfAwayFromZero(wg[k] / sc), -8, 7);
          qg[k] = static_cast<uint8_t>(qs + 8);
        }
      }
    }
  });
}

// ---- packer -------------------------------------------------------------------------------
//
// `q` [N,K] in 0..15 is the OFFSET-BINARY code the kernel's dequant produces after its own
// internal XOR-by-8 (i.e. exactly QuantizeInt4Asymmetric's or QuantizeInt4SymmetricPinned8's `q`,
// unmodified). Output `wq` is uint32[N*K/8], in the fragment order both r4d_gemm_w4a16_nt_m64 and
// r4d_gemm_w4a8_nt_m64 read: outer-to-inner (row-tile t, k-block kb of 64, lane-half lh, row-in-
// tile r, k-step s), 8 nibbles per dword (nibble i = element e=kNibbleToElement[i], stored as
// q[row,k]^8 so the kernel's XOR undoes it back to q[row,k]).
inline std::vector<uint32_t> PackW4Nibbles(const std::vector<uint8_t>& q, int N, int K,
                                            int nthreads) {
  constexpr int kKpb = 64;  // K per packed block (4 k-steps of 16)
  RequireDivisible(N, 16, "N", "PackW4Nibbles");
  RequireDivisible(K, kKpb, "K", "PackW4Nibbles");
  const int ntiles = N / 16;
  const int kblocks = K / kKpb;
  std::vector<uint32_t> wq(static_cast<size_t>(ntiles) * kblocks * 2 * 16 * 4);

  ParallelFor(0, ntiles, nthreads, [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      for (int kb = 0; kb < kblocks; ++kb) {
        for (int lh = 0; lh < 2; ++lh) {
          for (int r = 0; r < 16; ++r) {
            const int row = static_cast<int>(t) * 16 + r;
            const size_t dword_idx =
                (((static_cast<size_t>(t) * kblocks + kb) * 2 + lh) * 16 + r) * 4;
            for (int s = 0; s < 4; ++s) {
              uint32_t dword = 0;
              for (int i = 0; i < 8; ++i) {
                const int kk = 4 * lh + kNibbleKOffset[i];
                const int k = kb * kKpb + s * 16 + kk;
                const uint8_t code = q[static_cast<size_t>(row) * K + k];
                const uint32_t nibble = static_cast<uint32_t>(code ^ 0x8u) & 0xFu;
                dword |= nibble << (4 * i);
              }
              wq[dword_idx + s] = dword;
            }
          }
        }
      }
    }
  });
  return wq;
}

// wsz for w4a16: uint32[N*K/group], (row-tile t, group g, row-in-tile r) order -- low 16 bits =
// f16(scale), high 16 bits = f16(-(1024+zero)), exactly the dword r4d_gemm_w4a16_nt_m64.hip's
// epilogue reads (`sc[j] = ...& 0xFFFF`, `nz[j] = ...>>16`).
// Review finding (minor): a scale above 65504 (f16 max) previously encoded silently as f16 inf
// with no diagnostic anywhere; a scale of exactly 0 encoded from a nonzero input scale (a flush,
// not a deliberate "unused group" 0) is the same kind of silent-data-loss failure. Neither is
// active on the real checkpoint (min/max scale observed there is 8.4e-4 / 0.173), but a future
// model or an unscaled/mis-normalized weight should fail loudly here instead of shipping a group
// of silently-wrong weights.
inline uint16_t EncodeScaleF16Checked(float scale_f32, const char* tensor_hint) {
  const uint16_t f16 = r4dx::core::FloatToF16(scale_f32);
  const float back = r4dx::core::F16ToFloat(f16);
  if (!std::isfinite(back)) {
    throw std::runtime_error(std::string("r4dx_convert: scale ") + std::to_string(scale_f32) +
                              " overflowed f16 (inf) in " + tensor_hint);
  }
  if (scale_f32 != 0.0f && back == 0.0f) {
    throw std::runtime_error(std::string("r4dx_convert: scale ") + std::to_string(scale_f32) +
                              " flushed to 0 in f16 in " + tensor_hint);
  }
  return f16;
}

inline std::vector<uint32_t> PackW4A16Scales(const std::vector<float>& scale,
                                              const std::vector<uint8_t>& zero, int N, int K,
                                              int group) {
  RequireDivisible(N, 16, "N", "PackW4A16Scales");
  RequireDivisible(K, group, "K", "PackW4A16Scales");
  const int ntiles = N / 16;
  const int groups_per_row = K / group;
  std::vector<uint32_t> wsz(static_cast<size_t>(ntiles) * groups_per_row * 16);
  size_t out = 0;
  for (int t = 0; t < ntiles; ++t) {
    for (int g = 0; g < groups_per_row; ++g) {
      for (int r = 0; r < 16; ++r, ++out) {
        const int row = t * 16 + r;
        const size_t gidx = static_cast<size_t>(row) * groups_per_row + g;
        const uint16_t sc16 = EncodeScaleF16Checked(scale[gidx], "w4a16.wsz");
        const uint16_t nz16 = r4dx::core::FloatToF16(-(1024.0f + static_cast<float>(zero[gidx])));
        wsz[out] = static_cast<uint32_t>(sc16) | (static_cast<uint32_t>(nz16) << 16);
      }
    }
  }
  return wsz;
}

// ws for w4a8: uint32[N*K/group] (same (t,g,r) order and same 4-byte stride as wsz -- the kernel
// reads `Ws` through an `unsigned*`, `sz & 0xFFFF` only, see r4d_gemm_w4a8_nt_m64.hip; the doc's
// "uint16[...]" is the logical content, not the physical stride). High 16 bits are 0 (unread).
inline std::vector<uint32_t> PackW4A8Scales(const std::vector<float>& scale, int N, int K,
                                             int group) {
  RequireDivisible(N, 16, "N", "PackW4A8Scales");
  RequireDivisible(K, group, "K", "PackW4A8Scales");
  const int ntiles = N / 16;
  const int groups_per_row = K / group;
  std::vector<uint32_t> ws(static_cast<size_t>(ntiles) * groups_per_row * 16);
  size_t out = 0;
  for (int t = 0; t < ntiles; ++t) {
    for (int g = 0; g < groups_per_row; ++g) {
      for (int r = 0; r < 16; ++r, ++out) {
        const int row = t * 16 + r;
        const uint16_t sc16 = EncodeScaleF16Checked(scale[static_cast<size_t>(row) * groups_per_row + g],
                                                      "w4a8.ws");
        ws[out] = static_cast<uint32_t>(sc16);
      }
    }
  }
  return ws;
}

}  // namespace r4dx_convert
