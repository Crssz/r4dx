// r4dx_convert::quant_int8 -- the w8a16 quantizer, packer and CPU reference dequant
// (docs/w8a16.md sections 2, 3.2).
//
// w8a16 = symmetric int8 weights, one f16 scale per (row, group of G along K), stored biased by
// 2^kScaleLog2Bias, codes stored offset-binary (q + 128). The constants live in w8a16_format.hpp.
//
// The CPU REFERENCE other code reuses (kernel tests, loader tests, the tiny golden):
//   DequantW8a16(wq, ws, N, K, group)  -> float[N*K]
// It is written from the KERNEL's indexing (docs/w8a16.md 3.2: dword ((t*(K/32)+kb)*32+l)*4+j, byte i
// holds code(row, k) with row = 16t + (l&15), k = 32kb + 16*(j>>1) + 8*(j&1) + 4*(l>>4) + i; scale u16
// index (t*(K/G)+g)*16 + r), by walking the lanes, NOT by inverting PackW8Bytes' loops.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx/core/dtype.hpp"
#include "r4dx_convert/quant_int4.hpp"  // RoundHalfAwayFromZero, ClampInt, RequireDivisible
#include "r4dx_convert/threadpool.hpp"
#include "r4dx_convert/w8a16_format.hpp"

namespace r4dx_convert {

inline constexpr int kW8A16Group = w8a16::kDefaultGroup;
inline constexpr int kW8A16ScaleLog2Bias = w8a16::kScaleLog2Bias;
inline constexpr bool IsW8A16GroupSupported(int group) { return w8a16::IsGroupSupported(group); }

// Run-wide tallies of the quantizer (every QuantizeInt8Symmetric call adds to them), for the
// end-of-run log. "zero groups" = groups whose biased scale flushed to f16 zero (written as scale 0,
// codes 0).
struct W8a16Tally {
  std::atomic<uint64_t> groups{0};
  std::atomic<uint64_t> zero_groups{0};
};
inline W8a16Tally& W8a16RunTally() {
  static W8a16Tally t;
  return t;
}

inline constexpr float kW8A16ScaleBias = 4096.0f;      // 2^12
inline constexpr float kW8A16ScaleUnbias = 1.0f / 4096.0f;  // 2^-12, exact

// Symmetric int8, per (row, `group` contiguous K), RTN against the STORED f16 scale. Exact recipe
// (mirrored bit for bit by tools/convert_ref/w8_ref.py):
//   amax = max|w| (f32);  s = amax/127;  s16 = f16(s * 2^12) (RNE; inf -> error);
//   s16 == 0 -> scale 0, every code 0 (zero group);
//   s_use = float(s16) * 2^-12;  q = clamp(RoundHalfAwayFromZero(w / s_use), -127, 127).
// Outputs: `q` [N*K] offset-binary bytes (q + 128, so 1..255; 128 = zero), logical [N,K] row-major;
// `scale16` [N*(K/group)] f16 bits of s*2^12, row-major (row, group). `zero_groups` (optional) is
// incremented by this call's zero groups; the run-wide tally is updated too.
inline void QuantizeInt8Symmetric(const float* w, int N, int K, int group, int nthreads,
                                   std::vector<uint8_t>& q, std::vector<uint16_t>& scale16,
                                   uint64_t* zero_groups = nullptr,
                                   const char* tensor_hint = "w8a16") {
  if (!IsW8A16GroupSupported(group))
    throw std::runtime_error(std::string("r4dx_convert: w8a16 group ") + std::to_string(group) +
                             " is not 64 or 128 (" + tensor_hint + ")");
  RequireDivisible(K, group, "K", tensor_hint);
  const int groups_per_row = K / group;
  q.assign(static_cast<size_t>(N) * K, static_cast<uint8_t>(w8a16::kCodeOffset));
  scale16.assign(static_cast<size_t>(N) * groups_per_row, 0);
  std::atomic<uint64_t> zeros{0};
  std::atomic<int> err_kind{0};  // 1 = non-finite weight, 2 = biased scale overflows f16
  std::atomic<float> err_amax{0.0f};

  ParallelFor(0, N, nthreads, [&](int64_t r0, int64_t r1) {
    uint64_t local_zero = 0;
    for (int64_t row = r0; row < r1; ++row) {
      const float* wr = w + row * K;
      for (int g = 0; g < groups_per_row; ++g) {
        const float* wg = wr + static_cast<int64_t>(g) * group;
        float amax = 0.0f;
        for (int k = 0; k < group; ++k) {
          const float a = std::fabs(wg[k]);
          if (!std::isfinite(a)) {  // no throw inside a worker thread: report after the join
            err_kind.store(1);
            return;
          }
          amax = std::max(amax, a);
        }
        const float s = amax / 127.0f;
        const uint16_t s16 = r4dx::core::FloatToF16(s * kW8A16ScaleBias);
        const float s16f = r4dx::core::F16ToFloat(s16);
        if (!std::isfinite(s16f)) {
          err_amax.store(amax);
          err_kind.store(2);
          return;
        }
        const size_t gidx = static_cast<size_t>(row) * groups_per_row + g;
        if (s16 == 0) {  // codes stay 128 (q = 0), scale 0
          ++local_zero;
          continue;
        }
        scale16[gidx] = s16;
        const float s_use = s16f * kW8A16ScaleUnbias;
        uint8_t* qg = q.data() + row * static_cast<int64_t>(K) + static_cast<int64_t>(g) * group;
        for (int k = 0; k < group; ++k) {
          const int qi = ClampInt(RoundHalfAwayFromZero(wg[k] / s_use), -w8a16::kCodeMax,
                                  w8a16::kCodeMax);
          qg[k] = static_cast<uint8_t>(qi + w8a16::kCodeOffset);
        }
      }
    }
    zeros.fetch_add(local_zero, std::memory_order_relaxed);
  });
  if (err_kind.load() == 1)
    throw std::runtime_error(std::string("r4dx_convert: non-finite weight in ") + tensor_hint);
  if (err_kind.load() == 2)
    throw std::runtime_error(std::string("r4dx_convert: w8a16 group amax ") +
                             std::to_string(err_amax.load()) +
                             " overflows the biased f16 scale (max amax " +
                             std::to_string(w8a16::kMaxAmax) + ") in " + tensor_hint);
  const uint64_t z = zeros.load();
  if (zero_groups) *zero_groups += z;
  W8a16RunTally().groups.fetch_add(static_cast<uint64_t>(N) * groups_per_row);
  W8a16RunTally().zero_groups.fetch_add(z);
}

// Byte offset of code(row r16 in tile, lane l, dword j, byte i) -- the kernel's order (header comment).
inline constexpr size_t W8WqByteIndex(int64_t t, int kblocks, int kb, int lane, int j, int i) {
  return (((static_cast<size_t>(t) * kblocks + kb) * 32 + lane) * 4 + j) * 4 + i;
}
// Logical k of byte i of dword j of lane `lane` in 32-K block kb.
inline constexpr int W8WqK(int kb, int lane, int j, int i) {
  return 32 * kb + 16 * (j >> 1) + 8 * (j & 1) + 4 * (lane >> 4) + i;
}

// `q` [N,K] offset-binary bytes (QuantizeInt8Symmetric's) -> the kernel's fragment order: N*K bytes.
// Each 16-row tile is one contiguous run of 16*K bytes; each (tile, 32-K block) is 512 bytes.
inline std::vector<uint8_t> PackW8Bytes(const std::vector<uint8_t>& q, int N, int K, int nthreads) {
  RequireDivisible(N, w8a16::kTileRows, "N", "PackW8Bytes");
  RequireDivisible(K, w8a16::kKBlock, "K", "PackW8Bytes");
  const int ntiles = N / 16, kblocks = K / 32;
  std::vector<uint8_t> wq(static_cast<size_t>(N) * K);
  ParallelFor(0, ntiles, nthreads, [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t)
      for (int kb = 0; kb < kblocks; ++kb)
        for (int lane = 0; lane < 32; ++lane) {
          const int row = static_cast<int>(t) * 16 + (lane & 15);
          for (int j = 0; j < 4; ++j)
            for (int i = 0; i < 4; ++i)
              wq[W8WqByteIndex(t, kblocks, kb, lane, j, i)] =
                  q[static_cast<size_t>(row) * K + W8WqK(kb, lane, j, i)];
        }
  });
  return wq;
}

// `scale16` [N*(K/group)] (row, group) -> u16 [ntiles * (K/group) * 16] in (tile t, group g, row r) order.
inline std::vector<uint16_t> PackW8a16Scales(const std::vector<uint16_t>& scale16, int N, int K,
                                              int group) {
  RequireDivisible(N, 16, "N", "PackW8a16Scales");
  RequireDivisible(K, group, "K", "PackW8a16Scales");
  const int ntiles = N / 16, gpr = K / group;
  std::vector<uint16_t> ws(static_cast<size_t>(ntiles) * gpr * 16);
  size_t out = 0;
  for (int t = 0; t < ntiles; ++t)
    for (int g = 0; g < gpr; ++g)
      for (int r = 0; r < 16; ++r, ++out) ws[out] = scale16[static_cast<size_t>(t * 16 + r) * gpr + g];
  return ws;
}

// The CPU reference: dequantized weights W[N,K] (float32) from the packed tensors. Walks the kernel's
// lanes (header comment); a group with scale bits 0 dequantizes to exactly 0.
inline std::vector<float> DequantW8a16(const uint8_t* wq, const uint16_t* ws, int N, int K,
                                        int group) {
  RequireDivisible(N, 16, "N", "DequantW8a16");
  RequireDivisible(K, group, "K", "DequantW8a16");
  RequireDivisible(K, 32, "K", "DequantW8a16");
  const int ntiles = N / 16, kblocks = K / 32, gpr = K / group;
  std::vector<float> w(static_cast<size_t>(N) * K, 0.0f);
  for (int t = 0; t < ntiles; ++t)
    for (int kb = 0; kb < kblocks; ++kb)
      for (int lane = 0; lane < 32; ++lane) {
        const int row = t * 16 + (lane & 15);
        for (int j = 0; j < 4; ++j) {
          const uint32_t dword = static_cast<uint32_t>(wq[W8WqByteIndex(t, kblocks, kb, lane, j, 0)]) |
                                 (static_cast<uint32_t>(wq[W8WqByteIndex(t, kblocks, kb, lane, j, 1)]) << 8) |
                                 (static_cast<uint32_t>(wq[W8WqByteIndex(t, kblocks, kb, lane, j, 2)]) << 16) |
                                 (static_cast<uint32_t>(wq[W8WqByteIndex(t, kblocks, kb, lane, j, 3)]) << 24);
          for (int i = 0; i < 4; ++i) {
            const int k = 32 * kb + 16 * (j >> 1) + 8 * (j & 1) + 4 * (lane >> 4) + i;
            const int code = static_cast<int>((dword >> (8 * i)) & 0xFFu) - w8a16::kCodeOffset;
            const uint16_t s16 = ws[(static_cast<size_t>(t) * gpr + k / group) * 16 + (lane & 15)];
            const float s_use = r4dx::core::F16ToFloat(s16) * kW8A16ScaleUnbias;
            w[static_cast<size_t>(row) * K + k] = static_cast<float>(code) * s_use;
          }
        }
      }
  return w;
}

inline std::vector<float> DequantW8a16(const std::vector<uint8_t>& wq,
                                        const std::vector<uint16_t>& ws, int N, int K, int group) {
  if (wq.size() != static_cast<size_t>(N) * K ||
      ws.size() != static_cast<size_t>(N) * (K / group))
    throw std::runtime_error("DequantW8a16: tensor sizes do not match N, K, group");
  return DequantW8a16(wq.data(), ws.data(), N, K, group);
}

}  // namespace r4dx_convert
