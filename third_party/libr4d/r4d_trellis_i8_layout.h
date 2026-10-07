// r4d_trellis_i8_layout.h -- the operand layouts of libr4d's int8 x int8 trellis prefill GEMM
// (r4d_gemm_trellis_nt_i8.hip, r4d_trellis_i8.h; docs/int8-gemm-proto.md, docs/int8-prefill.md "Production
// path"). Pure index arithmetic, no includes beyond <cstddef>: compiled by hipcc (the kernels and the host
// entries share ONE definition of every formula, I8P_HD = __host__ __device__) and by a plain C++ compiler
// (the CPU tests, tests/kernels/int8_gemm_proto_ref.h).
//
// THE PROBLEM THIS FILE PINS DOWN.  A gfx12 WMMA fragment gives lane L one row (A) or column (B) of the
// 16 x 16 tile, L & 15, and 8 of its 16 k, picked by the lane's half h = L >> 4. Which 8 is a property of
// the hardware that only has to be the SAME for A and B (a dot product does not care about the order of its
// terms), so the int8 kernel uses the order the trellis decode produces its f16 fragments in, element e of
// half h <-> k = 8 (e >> 2) + 4 h + (e & 3) (FragK16): the decoded B fragment needs no permutation, and the
// activation producer writes A in that order too. D: lane L holds column L & 15 and rows 8 (L >> 4) + e.
//
// LAYOUTS (all byte offsets; K a multiple of 128, N a multiple of 32, M = 256):
//   A8   [rg][kt][i][lane][8]   rg = row / 64, i = (row / 16) % 4, kt = k / 16: one wave's A fragment of one
//        (16-row tile, k-tile) is 256 contiguous bytes, the 4 tiles of a row group 1 KiB, a row group's
//        K slice contiguous. Row = 64 rg + 16 i + (lane & 15).
//   SA   [kb][256]  fp32, the activation scale of (kb = k / 128, row)
//   W8   [pair][kt][lane][16]   the dense int8 weights in the trellis GEMM's own block order: a (tile pair,
//        k-tile) block is 512 B, lane L's 16 B = fragment F0 (8 B) then F1 (8 B); the column of element e of
//        F on lane L is n = 32 pair + 16 ((L >> 3) & 1) + 8 F + (L & 7).
//   SW   [kb][N]    fp32, the weight scale of (kb, column n of Q). The table holds s_eff = 1 / rs, rs the f16
//        the quantizer multiplies by, so the dequantization uses exactly the grid it rounded on.
//   plain A [256][K] and plain W [N][K] (true k, true n) exist for the references.
#pragma once

#include <cstddef>

#if defined(__HIPCC__)
#define I8P_HD __host__ __device__
#else
#define I8P_HD
#endif

namespace i8p {

constexpr int kM = 256;          // rows of the prefill super-chunk

// ---- fragment position map ------------------------------------------------------------------------
I8P_HD inline int FragK16(int lane, int e) { return 8 * (e >> 2) + 4 * (lane >> 4) + (e & 3); }
I8P_HD inline void K16ToFrag(int k16, int& h, int& e) {
  h = (k16 >> 2) & 1;
  e = 4 * (k16 >> 3) + (k16 & 3);
}

// ---- A8 / W8 offsets --------------------------------------------------------------------------------
// The 8 bytes of lane `lane`'s A fragment of (row group rg, k-tile kt, row tile i); KT = K / 16.
I8P_HD inline size_t A8FragOffset(int rg, int kt, int i, int lane, int KT) {
  return ((((size_t)rg * KT + kt) * 4 + i) * 32 + lane) * 8;
}
// byte offset of the element (row r, k) of the plain A in A8
I8P_HD inline size_t A8Offset(int r, int k, int K) {
  int h, e;
  K16ToFrag(k & 15, h, e);
  const int lane = (r & 15) + 16 * h;
  return A8FragOffset(r >> 6, k >> 4, (r >> 4) & 3, lane, K >> 4) + e;
}
// the 512-byte block of (tile pair, k-tile); lane L's 16 B are at + 16 L
I8P_HD inline size_t W8BlockOffset(int pair, int kt, int KT) { return ((size_t)pair * KT + kt) * 512; }
// the true column of element e of fragment F (0 or 1) on lane L of pair `pair`
I8P_HD inline int FragCol(int pair, int lane, int F) { return pair * 32 + 16 * ((lane >> 3) & 1) + 8 * F + (lane & 7); }
// byte offset of the element (column n, k) of the plain W in W8
I8P_HD inline size_t W8Offset(int n, int k, int K) {
  int h, e;
  K16ToFrag(k & 15, h, e);
  const int pair = n >> 5, j = n & 31, t = j >> 4, F = (j >> 3) & 1, c = j & 7;
  const int lane = 16 * h + 8 * t + c;
  return W8BlockOffset(pair, k >> 4, K >> 4) + (size_t)lane * 16 + F * 8 + e;
}
// accumulator layout: lane L's element e of tile (rg, i) is the row below; its column (tile pair `pair`,
// accumulator fragment f) is FragCol(pair, L, f)
I8P_HD inline int AccRow(int rg, int i, int lane, int e) { return rg * 64 + i * 16 + 8 * (lane >> 4) + e; }

}  // namespace i8p
