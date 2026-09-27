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
// Both decodes below are single inline-asm blocks (r4d_trellis_k4_decode says why). Inline asm at
// all because nothing in the builtins reaches v_mad_u32_u16's op_sel, and plain C lowers the 32-bit
// product to the quarter-rate v_mul_lo_u32. A fragment is built as four dwords and bit-cast whole:
// this compiler reads __builtin_bit_cast(unsigned short, f[e]) from the vector's first element
// whatever e is, so a caller that wants single elements bit-casts the whole fragment too.
typedef unsigned r4d_u32x4 __attribute__((ext_vector_type(4)));

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
// The per-lane alignment of the five words: 16 h bits (h = lane >> 4). Loop-invariant.
__device__ __forceinline__ unsigned r4d_trellis_k5_shift(int lane) { return 16u * (unsigned)(lane >> 4); }

// W[0..4]: this lane's words (r4d_trellis_k5_word), CLOBBERED (the block reuses them as scratch).
// sh = r4d_trellis_k5_shift(lane). V_i = alignbit(W_i, W_i+1, sh); state q is then big-endian bits
// [21 + 5q, 37 + 5q) of V0..V3 on every lane: V_i >> (16 - s) when it lies inside V_i (s = the bit
// offset in V_i <= 16), alignbit(V_i, V_i+1, 48 - s) when it straddles, and q = 15 (bit 96) is V3's
// high half, hashed through op_sel with no extract. F0 takes q {0,1,8,9,2,3,10,11} and F1
// {4,5,12,13,6,7,14,15} as elements e0..e7, i.e. dwords (q0,q1) (q8,q9) (q2,q3) (q10,q11) and
// (q4,q5) (q12,q13) (q6,q7) (q14,q15).
//
// One asm block for the reason r4d_trellis_k4_decode gives, 75 VALU: 4 alignbit (V), 15 extracts,
// 16 mad_u32_u16, 16 pk_mad_u16, 8 sad_u8, 8 sad_hi_u8, 8 pk_fma -- 4.69 per weight. Every
// dependency inside is at least 8 VALU apart except the first four extracts' on V1 / V2 (4 and 6);
// the block ends in s_delay_alu for the WMMA that reads the last dword. The low products of the 16
// hashes go through eight temporaries in two batches: W0..W4 (dead after V3) and V0..V2 (dead after
// the extracts). Registers: 24 (the five words, V0..V3 and 15 states; q15's hash lands in V3).
__device__ __forceinline__ void r4d_trellis_k5_decode(unsigned (&W)[5], unsigned sh, v8h& f0,
                                                      v8h& f1) {
  unsigned v0, v1, v2, v3, s0, s1, s2, s3, s4, s5, s6, s7, s8, s9, s10, s11, s12, s13, s14;
  asm("v_alignbit_b32 %[v1], %[w1], %[w2], %[sh]\n\t"
      "v_alignbit_b32 %[v2], %[w2], %[w3], %[sh]\n\t"
      "v_alignbit_b32 %[v0], %[w0], %[w1], %[sh]\n\t"
      "v_alignbit_b32 %[v3], %[w3], %[w4], %[sh]\n\t"
      // The 15 extracts, V1 / V2 shifts first (their V is oldest).
      "v_lshrrev_b32 %[s3], 12, %[v1]\n\t"
      "v_lshrrev_b32 %[s9], 14, %[v2]\n\t"
      "v_lshrrev_b32 %[s4], 7, %[v1]\n\t"
      "v_lshrrev_b32 %[s10], 9, %[v2]\n\t"
      "v_alignbit_b32 %[s0], %[v0], %[v1], 27\n\t"
      "v_alignbit_b32 %[s12], %[v2], %[v3], 31\n\t"
      "v_lshrrev_b32 %[s5], 2, %[v1]\n\t"
      "v_lshrrev_b32 %[s11], 4, %[v2]\n\t"
      "v_alignbit_b32 %[s1], %[v0], %[v1], 22\n\t"
      "v_alignbit_b32 %[s13], %[v2], %[v3], 26\n\t"
      "v_alignbit_b32 %[s6], %[v1], %[v2], 29\n\t"
      "v_alignbit_b32 %[s2], %[v0], %[v1], 17\n\t"
      "v_alignbit_b32 %[s14], %[v2], %[v3], 21\n\t"
      "v_alignbit_b32 %[s7], %[v1], %[v2], 24\n\t"
      "v_alignbit_b32 %[s8], %[v1], %[v2], 19\n\t"
      // Hash batch 1 (s3 s9 s4 s10 s0 s12 s5 s11): s * 0xD12D into a temporary, then
      // + s * 0x83DC << 16 back into the state's register.
      "v_mad_u32_u16 %[w0], %[s3], %[klo], 0\n\t"
      "v_mad_u32_u16 %[w1], %[s9], %[klo], 0\n\t"
      "v_mad_u32_u16 %[w2], %[s4], %[klo], 0\n\t"
      "v_mad_u32_u16 %[w3], %[s10], %[klo], 0\n\t"
      "v_mad_u32_u16 %[w4], %[s0], %[klo], 0\n\t"
      "v_mad_u32_u16 %[v0], %[s12], %[klo], 0\n\t"
      "v_mad_u32_u16 %[v1], %[s5], %[klo], 0\n\t"
      "v_mad_u32_u16 %[v2], %[s11], %[klo], 0\n\t"
      "v_pk_mad_u16 %[s3], %[khi], %[s3], %[w0] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s9], %[khi], %[s9], %[w1] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s4], %[khi], %[s4], %[w2] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s10], %[khi], %[s10], %[w3] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s0], %[khi], %[s0], %[w4] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s12], %[khi], %[s12], %[v0] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s5], %[khi], %[s5], %[v1] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s11], %[khi], %[s11], %[v2] op_sel_hi:[1,0,1]\n\t"
      // Hash batch 2 (s1 s13 s6 s2 s14 s7 s8, and q15 = V3's high half into V3).
      "v_mad_u32_u16 %[w0], %[s1], %[klo], 0\n\t"
      "v_mad_u32_u16 %[w1], %[s13], %[klo], 0\n\t"
      "v_mad_u32_u16 %[w2], %[s6], %[klo], 0\n\t"
      "v_mad_u32_u16 %[w3], %[s2], %[klo], 0\n\t"
      "v_mad_u32_u16 %[w4], %[s14], %[klo], 0\n\t"
      "v_mad_u32_u16 %[v0], %[s7], %[klo], 0\n\t"
      "v_mad_u32_u16 %[v1], %[s8], %[klo], 0\n\t"
      "v_mad_u32_u16 %[v2], %[v3], %[klo], 0 op_sel:[1,0,0,0]\n\t"
      "v_pk_mad_u16 %[s1], %[khi], %[s1], %[w0] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s13], %[khi], %[s13], %[w1] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s6], %[khi], %[s6], %[w2] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s2], %[khi], %[s2], %[w3] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s14], %[khi], %[s14], %[w4] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s7], %[khi], %[s7], %[v0] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[s8], %[khi], %[s8], %[v1] op_sel_hi:[1,0,1]\n\t"
      "v_pk_mad_u16 %[v3], %[khi], %[v3], %[v2] op_sel_hi:[1,1,1]\n\t"
      // Dword = (1024 + bytesum) of its low element (sad_u8) and high element (sad_hi_u8).
      "v_sad_u8 %[s4], %[s4], 0, 0x64006400\n\t"
      "v_sad_u8 %[s10], %[s10], 0, 0x64006400\n\t"
      "v_sad_u8 %[s0], %[s0], 0, 0x64006400\n\t"
      "v_sad_u8 %[s12], %[s12], 0, 0x64006400\n\t"
      "v_sad_u8 %[s6], %[s6], 0, 0x64006400\n\t"
      "v_sad_u8 %[s2], %[s2], 0, 0x64006400\n\t"
      "v_sad_u8 %[s14], %[s14], 0, 0x64006400\n\t"
      "v_sad_u8 %[s8], %[s8], 0, 0x64006400\n\t"
      "v_sad_hi_u8 %[s4], %[s5], 0, %[s4]\n\t"
      "v_sad_hi_u8 %[s10], %[s11], 0, %[s10]\n\t"
      "v_sad_hi_u8 %[s0], %[s1], 0, %[s0]\n\t"
      "v_sad_hi_u8 %[s12], %[s13], 0, %[s12]\n\t"
      "v_sad_hi_u8 %[s6], %[s7], 0, %[s6]\n\t"
      "v_sad_hi_u8 %[s2], %[s3], 0, %[s2]\n\t"
      "v_sad_hi_u8 %[s14], %[v3], 0, %[s14]\n\t"
      "v_sad_hi_u8 %[s8], %[s9], 0, %[s8]\n\t"
      // The codebook's affine map, one rounding (see r4d_trellis_k4_decode).
      "v_pk_fma_f16 %[s4], %[s4], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[s10], %[s10], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[s0], %[s0], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[s12], %[s12], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[s6], %[s6], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[s2], %[s2], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[s14], %[s14], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "v_pk_fma_f16 %[s8], %[s8], %[kinv], 0xc931 op_sel_hi:[1,0,0]\n\t"
      "s_delay_alu instid0(VALU_DEP_1)"
      : [w0] "+v"(W[0]), [w1] "+v"(W[1]), [w2] "+v"(W[2]), [w3] "+v"(W[3]), [w4] "+v"(W[4]),
        [v0] "=&v"(v0), [v1] "=&v"(v1), [v2] "=&v"(v2), [v3] "=&v"(v3), [s0] "=&v"(s0),
        [s1] "=&v"(s1), [s2] "=&v"(s2), [s3] "=&v"(s3), [s4] "=&v"(s4), [s5] "=&v"(s5),
        [s6] "=&v"(s6), [s7] "=&v"(s7), [s8] "=&v"(s8), [s9] "=&v"(s9), [s10] "=&v"(s10),
        [s11] "=&v"(s11), [s12] "=&v"(s12), [s13] "=&v"(s13), [s14] "=&v"(s14)
      : [sh] "v"(sh), [klo] "s"(R4D_TRELLIS_MUL1_KMLO), [khi] "s"(R4D_TRELLIS_MUL1_KMHI),
        [kinv] "s"(R4D_TRELLIS_KINV_F16X2));
  f0 = __builtin_bit_cast(v8h, (r4d_u32x4){s0, s8, s2, s10});
  f1 = __builtin_bit_cast(v8h, (r4d_u32x4){s4, s12, s6, s14});
}
