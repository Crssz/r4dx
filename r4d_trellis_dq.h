// r4d_trellis_dq.h - the trellis (EXL3 / QTIP, "mul1" codebook) tile decode, shared by
// r4d_gemm_trellis_nt_m64 and r4d_trellis_reconstruct_f16: which ring words one lane of a wave32
// WMMA B fragment reads, and how they become that fragment's eight f16 values -- bit for bit the
// values tools/reference/trellis_quant.py's decode_words produces (codebook_np, unpack_states,
// tensor_core_perm). The design and its CPU checks are docs/trellis-kernel.md 4.2 and 4.4 in r4dx;
// tools/reference/trellis_lane_check.py there emulates every function below lane by lane.
//
// THE FORMAT, AS FAR AS THE DECODE CARES.  A tile is 16 k x 16 n weights. Its 256 positions are a
// ring of 256*KB bits held in 8*KB uint32 words, stream bit 32w at bit 31 of word w; position p owns
// bits [KB*p, KB*(p+1)) and its 16-bit STATE is the 16 ring bits ending there (tail-biting: the ring
// wraps). The value is a hash of the state: x = s * 0x83DCD12D (mod 2^32), and the weight is the f16
// of (1024 + bytesum(x)) * f16(0x1EEE) + f16(0xC931), rounded once. Positions are in EXL3's NVIDIA
// tensor-core order: 8t..8t+7 are rows (k) 2m, 2m+1, 2m+8, 2m+9 of column c and then the same four
// rows of column c+8, for t = 4c + m.
//
// THE PAIR GRID.  Word w of tile (tn, tk) -- HF rows n = 16tn.., columns k = 16tk.. -- is uint32
// (((tn >> 1) * (K/16) + tk) * 2 + (tn & 1)) * 8KB + w: a (tile pair, k-tile) BLOCK is the two tiles'
// rings back to back, 64*KB bytes, and a pair's k-tiles are consecutive blocks.
//
// THE LANE MAP (the "column split").  A wave owns a tile pair (t0, t1) and builds two fragments per
// k-tile: F0 holds columns 0-7 of t0 on lanes 0-7 and of t1 on lanes 8-15, F1 columns 8-15 of both.
// A gfx12 fragment gives lane L one column and eight k, k = 8(e>>2) + 4(L>>4) + (e&3), so with
// t = (L>>3)&1, c = L&7, h = L>>4 the lane needs runs m = 2h and 2h+1 of column c -- ring words
// 4c+2h and 4c+2h+1 of tile t -- whose positions 0-3 are F0's and 4-7 F1's (column c+8's), plus the
// word before (mod 32, the tail-biting wrap) for the 12 history bits. At KB = 4 every state is a
// 16-bit window at a 4-bit offset of the 96 bits P:A:B, and a state for F0 and the state 16 bits
// later for F1 are ONE v_alignbit: its hi16 and lo16. Six alignbits give the lane's 16 states, and
// the fragment's element order falls out of the word order, so neither operand needs a k
// permutation. The output columns are permuted inside the pair: element e of F on lane L is the
// true column 32*pair + 16t + 8F + c, which the GEMM's epilogue puts back.
//
// THE VALUE PATH, 3.875 VALU PER WEIGHT.  The 32-bit multiply is quarter rate on this part, so it is
// split: v_mad_u32_u16 gives s * 0xD12D exactly, and one v_pk_mad_u16 adds s * 0x83DC to its high
// half (its low half adds 0 * s). op_sel picks the state's half, so the F0 / F1 states never have to
// be moved out of the alignbit result. v_sad_u8 against zero is a byte sum, and it adds its third
// operand: seeded with 0x64006400 (f16 1024 in both halves) a v_sad_u8 / v_sad_hi_u8 pair leaves the
// exact f16 bits of 1024 + bytesum for two states in one dword, and one v_pk_fma_f16 applies the
// codebook's affine map to both with the single rounding codebook_np specifies. Per lane per (tile
// pair, k-tile): 6 alignbit + 16 mad_u32_u16 + 16 pk_mad_u16 + 8 sad_u8 + 8 sad_hi_u8 + 8 pk_fma.
//
// KB = 5 (40-word tiles) is the same value path behind a different fetch: five words from
// ring[(5c - 2 + 3h + i) mod 40], pre-aligned by 16h so that state q (position 32c + 16h + q) is
// big-endian bits [21 + 5q, 37 + 5q) of V0..V3 on every lane.
#pragma once
#include <hip/hip_runtime.h>
#include <cstdint>

typedef _Float16 v8h __attribute__((ext_vector_type(8)));
typedef _Float16 r4d_h2 __attribute__((ext_vector_type(2)));

#define R4D_TRELLIS_MUL1_KMLO 0x0000D12Du   // 0x83DCD12D = KMHI:KMLO
#define R4D_TRELLIS_MUL1_KMHI 0x83DC0000u
#define R4D_TRELLIS_ONES_F16X2 0x64006400u  // f16 1024.0 in both halves: the sad seed
#define R4D_TRELLIS_KINV_F16X2 0x1EEE1EEEu  // the mul1 affine, f16 bits in both halves
#define R4D_TRELLIS_KBIAS_F16X2 0xC931C931u

// ---- lane map --------------------------------------------------------------------------------
// Byte offsets, inside one (pair, k-tile) block, of the words lane `lane` reads. Loop-invariant:
// the GEMM keeps them in a VGPR and walks the blocks with a scalar base.
__device__ __forceinline__ unsigned r4d_trellis_k4_off_ab(int lane) {   // words A, B (one b64)
  const int t = (lane >> 3) & 1, c = lane & 7, h = lane >> 4;
  return (unsigned)(t * 32 + 4 * c + 2 * h) * 4u;
}
__device__ __forceinline__ unsigned r4d_trellis_k4_off_p(int lane) {    // word P, the history
  const int t = (lane >> 3) & 1, c = lane & 7, h = lane >> 4;
  return (unsigned)(t * 32 + ((4 * c + 2 * h + 31) & 31)) * 4u;
}
// KB = 5: word i (0..4) of lane `lane`, as a word index inside the 80-word block.
__device__ __forceinline__ unsigned r4d_trellis_k5_word(int lane, int i) {
  const int t = (lane >> 3) & 1, c = lane & 7, h = lane >> 4;
  return (unsigned)(t * 40 + (5 * c - 2 + 3 * h + i + 40) % 40);
}

// ---- value path ------------------------------------------------------------------------------
// x = s * 0x83DCD12D (mod 2^32) for the 16-bit state s in the HIGH (HI) or LOW half of r. Inline
// asm because nothing in the builtins reaches v_mad_u32_u16's op_sel, and plain C lowers the
// 32-bit product to the quarter-rate v_mul_lo_u32. Not volatile: the scheduler may move and CSE it.
// These per-op helpers serve KB = 5, which only the reconstruct test entry decodes today, so their
// schedule (see r4d_trellis_k4_decode on why it matters) is not tuned; KB = 4 is one asm block.
template <bool HI>
__device__ __forceinline__ unsigned r4d_trellis_hash(unsigned r) {
  unsigned t, x;
  if constexpr (HI) {
    asm("v_mad_u32_u16 %0, %1, %2, 0 op_sel:[1,0,0,0]" : "=v"(t) : "v"(r), "s"(R4D_TRELLIS_MUL1_KMLO));
    asm("v_pk_mad_u16 %0, %1, %2, %3 op_sel_hi:[1,1,1]"
        : "=v"(x) : "s"(R4D_TRELLIS_MUL1_KMHI), "v"(r), "v"(t));
  } else {
    asm("v_mad_u32_u16 %0, %1, %2, 0" : "=v"(t) : "v"(r), "s"(R4D_TRELLIS_MUL1_KMLO));
    asm("v_pk_mad_u16 %0, %1, %2, %3 op_sel_hi:[1,0,1]"
        : "=v"(x) : "s"(R4D_TRELLIS_MUL1_KMHI), "v"(r), "v"(t));
  }
  return x;
}

// Two hashes -> the f16 bits of 1024 + bytesum(xa) (low half) and 1024 + bytesum(xb) (high half).
__device__ __forceinline__ unsigned r4d_trellis_pack(unsigned xa, unsigned xb) {
  return __builtin_amdgcn_sad_hi_u8(xb, 0u, __builtin_amdgcn_sad_u8(xa, 0u, R4D_TRELLIS_ONES_F16X2));
}

// The codebook's affine map on both halves, one rounding (== codebook_np bit for bit).
__device__ __forceinline__ unsigned r4d_trellis_affine(unsigned d) {
  const r4d_h2 kinv = __builtin_bit_cast(r4d_h2, R4D_TRELLIS_KINV_F16X2);
  const r4d_h2 kbias = __builtin_bit_cast(r4d_h2, R4D_TRELLIS_KBIAS_F16X2);
  return __builtin_bit_cast(unsigned, __builtin_elementwise_fma(__builtin_bit_cast(r4d_h2, d), kinv, kbias));
}

// Eight hashed states in fragment element order -> the fragment, built as four dwords and bit-cast
// whole. (A caller that wants single elements bit-casts the whole fragment too: this compiler reads
// __builtin_bit_cast(unsigned short, f[e]) from the vector's first element whatever e is.)
typedef unsigned r4d_u32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ v8h r4d_trellis_frag(const unsigned (&x)[8]) {
  r4d_u32x4 d;
#pragma unroll
  for (int i = 0; i < 4; ++i) d[i] = r4d_trellis_affine(r4d_trellis_pack(x[2 * i], x[2 * i + 1]));
  return __builtin_bit_cast(v8h, d);
}

// ---- KB = 4 ------------------------------------------------------------------------------------
// P, A, B: this lane's words (r4d_trellis_k4_off_p / _off_ab) of one block. f0 / f1: the lane's
// elements of fragments F0 and F1. Element order e0,e1 = A j0,j1; e2,e3 = B j0,j1; e4,e5 = A j2,j3;
// e6,e7 = B j2,j3, where RA[j] = alignbit(P, A, 12 - 4j) (RA[3] = A) and RB likewise on A:B.
//
// ONE asm block with a fixed schedule, because of what the compiler does NOT know about inline asm:
// its latency. gfx12 resolves a VALU -> VALU dependency by stalling the SIMD's VALU (every wave on
// it, not just the dependent one) unless an s_delay_alu tells the sequencer to hold that one wave
// instead, and LLVM inserts those only between instructions it generated itself. (Measured with
// r4dx's tool_trellis_gemm_bench --modes ops: ops each reading the one before issue at 0.20 per
// clock per SIMD with 1 wave and 0.23 with 4; with s_delay_alu, 0.18 and 0.71.) With one asm per
// hash (the first version) the scheduler put consumers 1-3 slots after their asm producers -- up to
// 249 such pairs per 992 VALU in a K loop, a different number in every instantiation -- and the
// decode issued at ~0.52 VALU per clock whatever the wave count. Here every dependency inside the
// block is at least 8 VALU apart, the inputs are loads (s_wait_loadcnt, which the compiler does
// insert), and the trailing s_delay_alu makes whatever follows -- usually the WMMA reading f1 --
// wait for the last pk_fma in its own wave rather than stall the SIMD. Same 62 VALU, same results
// bit for bit; the decode loop now issues at 0.95 VALU per clock per SIMD with 4 waves.
//
// Registers: 19 besides P, A and B; the block overwrites P (dead after its third alignbit). The
// states are hashed four at a time (8 mad, then 8 pk_mad), so four t registers serve all the
// low-half products; a state's high-half hash (F0) lands in its h register and its low-half hash
// (F1) in the state's own register; the fragment dwords are then built in place in the registers
// of the even elements (A0, B0, A2, B2).
__device__ __forceinline__ void r4d_trellis_k4_decode(unsigned P, unsigned A, unsigned B, v8h& f0,
                                                      v8h& f1) {
  // States: A0 A1 A2 A3 = RA[0..3] live in ra0, p (RA1 overwrites P, its last use), ra2, A;
  // B0 B1 B2 B3 = RB[0..3] in rb0, rb1, rb2, B. hX: the high-half hash of state X (F0's element);
  // the low-half hash (F1's) replaces X in its own register -- except A3 and B3's, which go to lA3
  // and lB3: A and B are halves of one 64-bit load, and tying either to an output costs v_movs.
  // t0..t3: the low-half products, reused.
  unsigned p = P;
  unsigned ra0, ra2, rb0, rb1, rb2, hA0, hA1, hA2, hA3, hB0, hB1, hB2, hB3, lA3, lB3, t0, t1, t2, t3;
  asm("v_alignbit_b32 %[ra0], %[p], %[a], 12\n\t"
      "v_alignbit_b32 %[rb0], %[a], %[b], 12\n\t"
      "v_alignbit_b32 %[ra2], %[p], %[a], 4\n\t"
      "v_alignbit_b32 %[rb2], %[a], %[b], 4\n\t"
      "v_alignbit_b32 %[rb1], %[a], %[b], 8\n\t"
      "v_alignbit_b32 %[p], %[p], %[a], 8\n\t"                        // RA1
      // s * 0xD12D (mad; op_sel picks the high half), then + s * 0x83DC << 16 (pk_mad) = the hash
      // s * 0x83DCD12D. First A3, B3, A0, B0: the loaded words, then the oldest alignbits.
      "v_mad_u32_u16 %[hA3], %[a], %[klo], 0 op_sel:[1,0,0,0]\n\t"
      "v_mad_u32_u16 %[t0], %[a], %[klo], 0\n\t"
      "v_mad_u32_u16 %[hB3], %[b], %[klo], 0 op_sel:[1,0,0,0]\n\t"
      "v_mad_u32_u16 %[t1], %[b], %[klo], 0\n\t"
      "v_mad_u32_u16 %[hA0], %[ra0], %[klo], 0 op_sel:[1,0,0,0]\n\t"
      "v_mad_u32_u16 %[t2], %[ra0], %[klo], 0\n\t"
      "v_mad_u32_u16 %[hB0], %[rb0], %[klo], 0 op_sel:[1,0,0,0]\n\t"
      "v_mad_u32_u16 %[t3], %[rb0], %[klo], 0\n\t"
      "v_pk_mad_u16 %[hA3], %[khi], %[a], %[hA3] op_sel_hi:[1,1,1]\n\t"
      "v_pk_mad_u16 %[lA3], %[khi], %[a], %[t0] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[hB3], %[khi], %[b], %[hB3] op_sel_hi:[1,1,1]\n\t"
      "v_pk_mad_u16 %[lB3], %[khi], %[b], %[t1] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[hA0], %[khi], %[ra0], %[hA0] op_sel_hi:[1,1,1]\n\t"
      "v_pk_mad_u16 %[ra0], %[khi], %[ra0], %[t2] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[hB0], %[khi], %[rb0], %[hB0] op_sel_hi:[1,1,1]\n\t"
      "v_pk_mad_u16 %[rb0], %[khi], %[rb0], %[t3] op_sel_hi:[1,0,1]\n\t"
      // Then A2, B2, B1, A1.
      "v_mad_u32_u16 %[hA2], %[ra2], %[klo], 0 op_sel:[1,0,0,0]\n\t"
      "v_mad_u32_u16 %[t0], %[ra2], %[klo], 0\n\t"
      "v_mad_u32_u16 %[hB2], %[rb2], %[klo], 0 op_sel:[1,0,0,0]\n\t"
      "v_mad_u32_u16 %[t1], %[rb2], %[klo], 0\n\t"
      "v_mad_u32_u16 %[hB1], %[rb1], %[klo], 0 op_sel:[1,0,0,0]\n\t"
      "v_mad_u32_u16 %[t2], %[rb1], %[klo], 0\n\t"
      "v_mad_u32_u16 %[hA1], %[p], %[klo], 0 op_sel:[1,0,0,0]\n\t"
      "v_mad_u32_u16 %[t3], %[p], %[klo], 0\n\t"
      "v_pk_mad_u16 %[hA2], %[khi], %[ra2], %[hA2] op_sel_hi:[1,1,1]\n\t"
      "v_pk_mad_u16 %[ra2], %[khi], %[ra2], %[t0] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[hB2], %[khi], %[rb2], %[hB2] op_sel_hi:[1,1,1]\n\t"
      "v_pk_mad_u16 %[rb2], %[khi], %[rb2], %[t1] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[hB1], %[khi], %[rb1], %[hB1] op_sel_hi:[1,1,1]\n\t"
      "v_pk_mad_u16 %[rb1], %[khi], %[rb1], %[t2] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[hA1], %[khi], %[p], %[hA1] op_sel_hi:[1,1,1]\n\t"
      "v_pk_mad_u16 %[p], %[khi], %[p], %[t3] op_sel_hi:[1,0,1]\n\t"
      // Dword j of a fragment = (1024 + bytesum) of elements 2j (low half) and 2j + 1 (high half):
      // j = 0 (A0, A1), 1 (B0, B1), 2 (A2, A3), 3 (B2, B3); F0 from the h registers, F1 from the
      // states' own.
      "v_sad_u8 %[hA0], %[hA0], 0, 0x64006400\n\t"
      "v_sad_u8 %[hB0], %[hB0], 0, 0x64006400\n\t"
      "v_sad_u8 %[hA2], %[hA2], 0, 0x64006400\n\t"
      "v_sad_u8 %[hB2], %[hB2], 0, 0x64006400\n\t"
      "v_sad_u8 %[ra0], %[ra0], 0, 0x64006400\n\t"
      "v_sad_u8 %[rb0], %[rb0], 0, 0x64006400\n\t"
      "v_sad_u8 %[ra2], %[ra2], 0, 0x64006400\n\t"
      "v_sad_u8 %[rb2], %[rb2], 0, 0x64006400\n\t"
      "v_sad_hi_u8 %[hA0], %[hA1], 0, %[hA0]\n\t"
      "v_sad_hi_u8 %[hB0], %[hB1], 0, %[hB0]\n\t"
      "v_sad_hi_u8 %[hA2], %[hA3], 0, %[hA2]\n\t"
      "v_sad_hi_u8 %[hB2], %[hB3], 0, %[hB2]\n\t"
      "v_sad_hi_u8 %[ra0], %[p], 0, %[ra0]\n\t"
      "v_sad_hi_u8 %[rb0], %[rb1], 0, %[rb0]\n\t"
      "v_sad_hi_u8 %[ra2], %[lA3], 0, %[ra2]\n\t"
      "v_sad_hi_u8 %[rb2], %[lB3], 0, %[rb2]\n\t"
      // The codebook's affine map, one rounding: d * f16(0x1EEE) + f16(0xC931) on both halves (the
      // literal's low half serves the high half through op_sel_hi).
      "v_pk_fma_f16 %[hA0], %[hA0], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[hB0], %[hB0], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[hA2], %[hA2], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[hB2], %[hB2], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[ra0], %[ra0], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[rb0], %[rb0], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[ra2], %[ra2], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[rb2], %[rb2], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "s_delay_alu instid0(VALU_DEP_1)"
      : [p] "+v"(p), [ra0] "=&v"(ra0), [ra2] "=&v"(ra2), [rb0] "=&v"(rb0), [rb1] "=&v"(rb1),
        [rb2] "=&v"(rb2), [hA0] "=&v"(hA0), [hA1] "=&v"(hA1), [hA2] "=&v"(hA2), [hA3] "=&v"(hA3),
        [hB0] "=&v"(hB0), [hB1] "=&v"(hB1), [hB2] "=&v"(hB2), [hB3] "=&v"(hB3), [lA3] "=&v"(lA3),
        [lB3] "=&v"(lB3), [t0] "=&v"(t0), [t1] "=&v"(t1), [t2] "=&v"(t2), [t3] "=&v"(t3)
      : [a] "v"(A), [b] "v"(B), [klo] "s"(R4D_TRELLIS_MUL1_KMLO), [khi] "s"(R4D_TRELLIS_MUL1_KMHI),
        [kinv] "s"(R4D_TRELLIS_KINV_F16X2));
  f0 = __builtin_bit_cast(v8h, (r4d_u32x4){hA0, hB0, hA2, hB2});
  f1 = __builtin_bit_cast(v8h, (r4d_u32x4){ra0, rb0, ra2, rb2});
}

// ---- KB = 5 ------------------------------------------------------------------------------------
// W[0..4]: this lane's words (r4d_trellis_k5_word). State q is big-endian bits [21 + 5q, 37 + 5q)
// of V0..V3; F0 takes q {0,1,8,9,2,3,10,11}, F1 q {4,5,12,13,6,7,14,15}. q = 15 starts on a word
// boundary (bit 96), so its state is V3's high half and hashes through op_sel with no extract.
template <int Q>
__device__ __forceinline__ unsigned r4d_trellis_k5_hash(const unsigned (&V)[4]) {
  constexpr int r = 21 + 5 * Q, i = r >> 5, s = r & 31;
  if constexpr (s == 0) {
    return r4d_trellis_hash<true>(V[i]);
  } else if constexpr (s <= 16) {
    return r4d_trellis_hash<false>(V[i] >> (16 - s));
  } else {
    return r4d_trellis_hash<false>(__builtin_amdgcn_alignbit(V[i], V[i + 1], 48 - s));
  }
}

__device__ __forceinline__ void r4d_trellis_k5_decode(const unsigned (&W)[5], int lane, v8h& f0,
                                                      v8h& f1) {
  const unsigned sh = 16u * (unsigned)(lane >> 4);
  unsigned V[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) V[i] = __builtin_amdgcn_alignbit(W[i], W[i + 1], sh);
  const unsigned x0[8] = {r4d_trellis_k5_hash<0>(V), r4d_trellis_k5_hash<1>(V),
                          r4d_trellis_k5_hash<8>(V), r4d_trellis_k5_hash<9>(V),
                          r4d_trellis_k5_hash<2>(V), r4d_trellis_k5_hash<3>(V),
                          r4d_trellis_k5_hash<10>(V), r4d_trellis_k5_hash<11>(V)};
  const unsigned x1[8] = {r4d_trellis_k5_hash<4>(V), r4d_trellis_k5_hash<5>(V),
                          r4d_trellis_k5_hash<12>(V), r4d_trellis_k5_hash<13>(V),
                          r4d_trellis_k5_hash<6>(V), r4d_trellis_k5_hash<7>(V),
                          r4d_trellis_k5_hash<14>(V), r4d_trellis_k5_hash<15>(V)};
  f0 = r4d_trellis_frag(x0);
  f1 = r4d_trellis_frag(x1);
}
