// tests/kernels/trellis_ref.hpp -- CPU reference decode of the trellis (EXL3 / QTIP "mul1") weight
// format, for the libr4d trellis kernels' tests (docs/trellis-kernel.md 2.1, 4.2, 6).
//
// A C++ transcription of tools/reference/trellis_quant.py's decode, function for function:
//   Mul1CodebookF16   codebook_np("mul1"): x = s * 0x83DCD12D, f16((1024 + bytesum(x)) * f16(0x1EEE)
//                     + f16(0xC931)) with one rounding -- the product and the sum are exact in
//                     float (11-bit by 11-bit, and |result| * 2^19 < 2^21), so FloatToF16 of the
//                     float result IS the single f16 rounding;
//   TrellisState      unpack_states: state(p) = the 16 ring bits ending at bit KB*(p+1), tail-biting;
//   TensorCorePerm    tensor_core_perm: position 8t + j -> row-major tile element (r*16 + c,
//                     r = k inside the tile, c = n);
//   DecodeOracle      decode_words: oracle-layout words [K/16][N/16][8KB] -> Q[K][N] f16 bits;
//   ToPairGrid        trellis_golden.py's to_pair_grid: word w of tile (tn, tk) at
//                     (((tn>>1)*(K/16) + tk)*2 + (tn&1))*8KB + w;
//   DecodePairGrid    the same decode, read from the pair grid.
// test_trellis_decode checks it against the committed goldens (tests/kernels/golden/trellis, from
// the Python) bit for bit before trusting it on shapes the goldens do not cover.
#pragma once

#include <cstdint>
#include <vector>

#include "r4dx/core/dtype.hpp"

namespace trellis_ref {

inline const std::vector<uint16_t>& Mul1CodebookF16() {
  static const std::vector<uint16_t> cb = [] {
    const double kinv = r4dx::core::F16ToFloat(0x1EEE), kbias = r4dx::core::F16ToFloat(0xC931);
    std::vector<uint16_t> out(1u << 16);
    for (uint32_t s = 0; s < (1u << 16); ++s) {
      const uint32_t x = s * 0x83DCD12Du;
      const uint32_t bs = (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24);
      const double v = (1024.0 + bs) * kinv + kbias;   // exact (see the header comment)
      out[s] = r4dx::core::FloatToF16(static_cast<float>(v));
    }
    return out;
  }();
  return cb;
}

// state(p) of one tile's ring `w` (8*KB words), integer KB.
inline uint32_t TrellisState(const uint32_t* w, int KB, int p) {
  const int R = 256 * KB, nw = 8 * KB;
  const int start = ((KB * (p + 1) - 16) % R + R) % R;
  const int i0 = start / 32, off = start % 32, i1 = (i0 + 1) % nw;
  const uint64_t w0 = w[i0], w1 = w[i1];
  return static_cast<uint32_t>((((w0 << off) & 0xFFFFFFFFull) >> 16) | (w1 >> (48 - off))) & 0xFFFFu;
}

// Position p -> row-major tile element r*16 + c.
inline int TensorCorePerm(int p) {
  const int t = p / 8, j = p % 8;
  const int r0 = (t % 4) * 2, c0 = t / 4;
  const int rows[4] = {r0, r0 + 1, r0 + 8, r0 + 9};
  return j < 4 ? rows[j] * 16 + c0 : rows[j - 4] * 16 + c0 + 8;
}

// One tile's ring -> its 256 f16 values at (r, c) of out[(k0 + r) * N + n0 + c].
inline void DecodeTile(const uint32_t* w, int KB, uint16_t* out, int64_t N, int64_t k0, int64_t n0) {
  const std::vector<uint16_t>& cb = Mul1CodebookF16();
  for (int p = 0; p < 256; ++p) {
    const int e = TensorCorePerm(p);
    out[(k0 + e / 16) * N + n0 + e % 16] = cb[TrellisState(w, KB, p)];
  }
}

inline size_t PairGridIndex(int64_t K, int KB, int64_t tn, int64_t tk) {
  return static_cast<size_t>((((tn >> 1) * (K / 16) + tk) * 2 + (tn & 1)) * 8 * KB);
}

// Oracle layout [K/16][N/16][8KB] -> Q[K][N] f16 bits.
inline std::vector<uint16_t> DecodeOracle(const std::vector<uint32_t>& words, int64_t K, int64_t N,
                                          int KB) {
  std::vector<uint16_t> q(static_cast<size_t>(K * N));
  const int nw = 8 * KB;
  for (int64_t tk = 0; tk < K / 16; ++tk)
    for (int64_t tn = 0; tn < N / 16; ++tn)
      DecodeTile(&words[static_cast<size_t>((tk * (N / 16) + tn) * nw)], KB, q.data(), N, 16 * tk,
                 16 * tn);
  return q;
}

inline std::vector<uint32_t> ToPairGrid(const std::vector<uint32_t>& words, int64_t K, int64_t N,
                                        int KB) {
  std::vector<uint32_t> grid(words.size());
  const int nw = 8 * KB;
  for (int64_t tk = 0; tk < K / 16; ++tk)
    for (int64_t tn = 0; tn < N / 16; ++tn)
      for (int w = 0; w < nw; ++w)
        grid[PairGridIndex(K, KB, tn, tk) + w] = words[static_cast<size_t>((tk * (N / 16) + tn) * nw + w)];
  return grid;
}

inline std::vector<uint16_t> DecodePairGrid(const std::vector<uint32_t>& grid, int64_t K, int64_t N,
                                            int KB) {
  std::vector<uint16_t> q(static_cast<size_t>(K * N));
  for (int64_t tk = 0; tk < K / 16; ++tk)
    for (int64_t tn = 0; tn < N / 16; ++tn)
      DecodeTile(&grid[PairGridIndex(K, KB, tn, tk)], KB, q.data(), N, 16 * tk, 16 * tn);
  return q;
}

}  // namespace trellis_ref
