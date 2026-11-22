// r4dx_convert::quant_mxfp8 -- the mxfp8 quantizer, packers and CPU reference dequant
// (docs/inkling2-plan.md 4.1; the on-disk constants and the e4m3 / E8M0 codec are mxfp8_format.hpp).
//
// One linear W [N, K] (N % 16, K % 32) becomes
//   <base>.mxfp8.wq  u8 {N*K}     e4m3 codes in w8a16's packed fragment order (PackW8Bytes: the SAME permutation,
//                                 only the byte is an OCP e4m3fn code instead of an offset-binary int8);
//   <base>.mxfp8.ws  u8 {N*K/32}  one E8M0 byte per (row, 32-K group), in w8a16's ws ELEMENT order: index
//                                 (t * (K/32) + g) * 16 + r for row 16t + r and group g (PackW8a16Scales' order).
// The CPU REFERENCE (DequantMxfp8) is written from the kernel's indexing by walking the lanes, like
// DequantW8a16, not by inverting the packers.
#pragma once

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "r4dx_convert/mxfp8_format.hpp"
#include "r4dx_convert/quant_int8.hpp"  // PackW8Bytes, W8WqByteIndex, W8WqK, RequireDivisible, ParallelFor

namespace r4dx_convert {

// f32 [N, K] -> codes [N*K] (logical row-major) and scale bytes [N*K/32] (row, group), through
// mxfp8::EncodeGroup (per row and 32-K group: ceil rule, no clipping, RNE e4m3). A non-finite weight throws.
inline void QuantizeMxfp8(const float* w, int N, int K, int nthreads, std::vector<uint8_t>& codes,
                          std::vector<uint8_t>& scales, const char* hint = "mxfp8") {
  RequireDivisible(K, mxfp8::kGroup, "K", hint);
  const int gpr = K / mxfp8::kGroup;
  codes.assign(static_cast<size_t>(N) * K, 0);
  scales.assign(static_cast<size_t>(N) * gpr, 0);
  std::atomic<bool> bad{false};
  ParallelFor(0, N, nthreads, [&](int64_t r0, int64_t r1) {
    for (int64_t row = r0; row < r1; ++row) {
      const float* wr = w + row * K;
      for (int k = 0; k < K; ++k)
        if (!std::isfinite(wr[k])) {
          bad.store(true);
          return;
        }
      for (int g = 0; g < gpr; ++g)
        scales[static_cast<size_t>(row) * gpr + g] =
            mxfp8::EncodeGroup(wr + static_cast<int64_t>(g) * mxfp8::kGroup, mxfp8::kGroup,
                               codes.data() + row * static_cast<int64_t>(K) + static_cast<int64_t>(g) * mxfp8::kGroup);
    }
  });
  if (bad.load()) throw std::runtime_error(std::string("r4dx_convert: non-finite weight in ") + hint);
}

// (row, group) scale bytes [N*K/32] -> w8a16's ws element order (tile t, group g, row r), one byte each.
inline std::vector<uint8_t> PackMxfp8Scales(const std::vector<uint8_t>& scales, int N, int K) {
  RequireDivisible(N, 16, "N", "PackMxfp8Scales");
  RequireDivisible(K, mxfp8::kGroup, "K", "PackMxfp8Scales");
  const int ntiles = N / 16, gpr = K / mxfp8::kGroup;
  if (scales.size() != static_cast<size_t>(N) * gpr)
    throw std::runtime_error("PackMxfp8Scales: scale count does not match N, K");
  std::vector<uint8_t> ws(scales.size());
  size_t out = 0;
  for (int t = 0; t < ntiles; ++t)
    for (int g = 0; g < gpr; ++g)
      for (int r = 0; r < 16; ++r, ++out) ws[out] = scales[static_cast<size_t>(t * 16 + r) * gpr + g];
  return ws;
}

// The codes' fragment packing: w8a16's, unchanged (codes are plain bytes here).
inline std::vector<uint8_t> PackMxfp8Codes(const std::vector<uint8_t>& codes, int N, int K, int nthreads) {
  return PackW8Bytes(codes, N, K, nthreads);
}

// The CPU reference: W[N, K] (f32) from the packed tensors, by walking the kernel's lanes. Element value
// DecodeE4m3(wq) * 2^(ws - 127); a NaN code or scale byte 255 is rejected (never written).
inline std::vector<float> DequantMxfp8(const uint8_t* wq, const uint8_t* ws, int N, int K) {
  RequireDivisible(N, 16, "N", "DequantMxfp8");
  RequireDivisible(K, mxfp8::kGroup, "K", "DequantMxfp8");
  const int ntiles = N / 16, kblocks = K / 32, gpr = K / mxfp8::kGroup;
  std::vector<float> w(static_cast<size_t>(N) * K, 0.0f);
  for (int t = 0; t < ntiles; ++t)
    for (int kb = 0; kb < kblocks; ++kb)
      for (int lane = 0; lane < 32; ++lane) {
        const int row = t * 16 + (lane & 15);
        for (int j = 0; j < 4; ++j)
          for (int i = 0; i < 4; ++i) {
            const int k = W8WqK(kb, lane, j, i);
            const uint8_t code = wq[W8WqByteIndex(t, kblocks, kb, lane, j, i)];
            const uint8_t sb = ws[(static_cast<size_t>(t) * gpr + k / mxfp8::kGroup) * 16 + (lane & 15)];
            if (sb == 255 || (code & 0x7F) == 0x7F)
              throw std::runtime_error("DequantMxfp8: NaN code or scale byte in a packed tensor");
            w[static_cast<size_t>(row) * K + k] = mxfp8::DecodeE4m3(code) * mxfp8::ScaleValue(sb);
          }
      }
  return w;
}

inline std::vector<float> DequantMxfp8(const std::vector<uint8_t>& wq, const std::vector<uint8_t>& ws, int N, int K) {
  if (wq.size() != static_cast<size_t>(N) * K || ws.size() != static_cast<size_t>(N) * (K / mxfp8::kGroup))
    throw std::runtime_error("DequantMxfp8: tensor sizes do not match N, K");
  return DequantMxfp8(wq.data(), ws.data(), N, K);
}

}  // namespace r4dx_convert
