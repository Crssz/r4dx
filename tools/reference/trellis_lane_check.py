"""tools/reference/trellis_lane_check.py -- CPU check of the trellis GEMM's lane map (no GPU).

docs/trellis-kernel.md 4.2 and 4.4 give each lane of a wave32 WMMA B fragment its words and its
extraction; this emulates that per lane, from the pair grid (2.1), and compares every result with
trellis_quant.py's own functions:

  column split  a wave owns tile pair (t0, t1); fragment F0 = columns 0-7 of both tiles, F1 =
                columns 8-15; lane L: t = (L >> 3) & 1, c = L & 7, h = L >> 4, n = 16t + c (+ 8 in F1)
  KB = 4        A, B = ring words 4c + 2h, +1 and P = word (4c + 2h - 1) mod 32 of tile t;
                RA_j = alignbit(P, A, 12 - 4j), RB_j = alignbit(A, B, 12 - 4j) (j < 3), RA_3 = A,
                RB_3 = B; F0 states = hi16, F1 = lo16; e0,e1 = A j0,j1; e2,e3 = B j0,j1;
                e4,e5 = A j2,j3; e6,e7 = B j2,j3
  KB = 5        5 words ring[(5c - 2 + 3h + i) mod 40]; V_i = alignbit(W_i, W_{i+1}, 16h); state q
                = big-endian bits [21 + 5q, 37 + 5q) of V0..V3; F0 = q {0,1,8,9,2,3,10,11},
                F1 = q {4,5,12,13,6,7,14,15}
  fragment      element e of lane L is k = 8(e >> 2) + 4h + (e & 3) (r4d_gdn_wmma.h:2-6)
  value path    tlo = v_mad_u32_u16(s, 0xD12D, 0); x = v_pk_mad_u16(0x83DC0000, s, tlo) (hi16 +=
                s * 0x83DC); d = v_sad_hi_u8(x_b, 0, v_sad_u8(x_a, 0, 0x64006400)) (the f16 bits of
                1024 + bytesum, lo16 = e even, hi16 = e odd); v_pk_fma_f16(d, 0x1EEE, 0xC931)

Checked: every (tile, k, n) is produced exactly once; its 16-bit state equals unpack_states +
tensor_core_perm; its f16 value equals decode_words' Q bit for bit; the value path equals
codebook_np for all 65,536 states (both halves). Also reported: the AFFINE variant's deviation (4.6).
Exit code 1 on any mismatch.

  & $py tools\\reference\\trellis_lane_check.py [--tiles 64] [--seed 0]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import trellis_quant as tq  # noqa: E402  (torch on the CPU only)
import torch  # noqa: E402
from trellis_golden import to_pair_grid  # noqa: E402

KMLO, KMHI = 0x0000D12D, 0x83DC0000        # 0x83DCD12D = KMHI:KMLO
ONESBIAS = 0x64006400                      # f16 1024.0 in both halves
KINV, KBIAS = 0x1EEE, 0xC931               # the mul1 affine, f16 bits
F0_Q5 = (0, 1, 8, 9, 2, 3, 10, 11)
F1_Q5 = (4, 5, 12, 13, 6, 7, 14, 15)


def alignbit(hi: int, lo: int, s: int) -> int:
    """v_alignbit_b32: ({hi, lo} >> (s & 31)) & 0xFFFFFFFF."""
    return (((hi << 32) | lo) >> (s & 31)) & 0xFFFFFFFF


def lane_states_k4(block: np.ndarray, L: int) -> tuple[list[int], list[int]]:
    """block: one (pair, k-tile) of the pair grid, 2 x 32 uint32. -> F0, F1 states in e order."""
    t, c, h = (L >> 3) & 1, L & 7, L >> 4
    wa = 4 * c + 2 * h
    wp = (wa + 31) & 31                    # tail-biting wrap
    A, B, P = (int(block[t * 32 + wa]), int(block[t * 32 + wa + 1]), int(block[t * 32 + wp]))
    RA = [alignbit(P, A, 12 - 4 * j) for j in range(3)] + [A]
    RB = [alignbit(A, B, 12 - 4 * j) for j in range(3)] + [B]
    order = [(RA, 0), (RA, 1), (RB, 0), (RB, 1), (RA, 2), (RA, 3), (RB, 2), (RB, 3)]
    return [R[j] >> 16 for R, j in order], [R[j] & 0xFFFF for R, j in order]


def lane_states_k5(block: np.ndarray, L: int) -> tuple[list[int], list[int]]:
    """block: one (pair, k-tile) of the pair grid, 2 x 40 uint32."""
    t, c, h = (L >> 3) & 1, L & 7, L >> 4
    wb = 5 * c - 2 + 3 * h
    W = [int(block[t * 40 + (wb + i) % 40]) for i in range(5)]
    V = [alignbit(W[i], W[i + 1], 16 * h) for i in range(4)]
    st = []
    for q in range(16):                    # the kernel's per-state extraction (4.4)
        r = 21 + 5 * q
        i, s = r >> 5, r & 31
        v = (V[i] >> (16 - s)) if s <= 16 else alignbit(V[i], V[i + 1], 48 - s)
        st.append(v & 0xFFFF)              # the hash reads lo16 (op_sel)
    return [st[q] for q in F0_Q5], [st[q] for q in F1_Q5]


def hash_x(s: int) -> int:
    """x = s * 0x83DCD12D mod 2^32 as the kernel builds it."""
    tlo = (s * (KMLO & 0xFFFF)) & 0xFFFFFFFF                   # v_mad_u32_u16(s, KMLO, 0)
    lo = ((KMHI & 0xFFFF) * s + (tlo & 0xFFFF)) & 0xFFFF      # v_pk_mad_u16 lo half: 0 * s + tlo.lo
    hi = ((KMHI >> 16) * s + (tlo >> 16)) & 0xFFFF            # hi half: 0x83DC * s + tlo.hi
    return (hi << 16) | lo


def bytesum(x: int) -> int:
    return (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + (x >> 24)


def f16_bits_to_f64(b: int) -> float:
    return float(np.array([b], dtype=np.uint16).view(np.float16)[0])


KINV_F, KBIAS_F = f16_bits_to_f64(KINV), f16_bits_to_f64(KBIAS)


def fma_f16(d16: int) -> int:
    """v_pk_fma_f16 on one half: fp16(d * KINV + KBIAS), exact product and sum, one rounding."""
    v = np.float64(f16_bits_to_f64(d16) * KINV_F + KBIAS_F)
    return int(np.array([v]).astype(np.float16).view(np.uint16)[0])


def fragment_values(states: list[int]) -> list[int]:
    """8 states in e order -> 8 f16 bit patterns, through the sad/sad_hi packing and pk_fma."""
    out = []
    for e in range(0, 8, 2):
        d = (ONESBIAS + bytesum(hash_x(states[e]))) & 0xFFFFFFFF          # v_sad_u8
        d = (d + (bytesum(hash_x(states[e + 1])) << 16)) & 0xFFFFFFFF     # v_sad_hi_u8
        out += [fma_f16(d & 0xFFFF), fma_f16(d >> 16)]
    return out


def check_lanes(KB: int, tiles: int, seed: int) -> tuple[int, int, int]:
    """Random oracle-layout words [tk][tn][8KB] (tiles = tk * tn), regridded into the pair grid;
    returns (state mismatches, value mismatches, elements checked)."""
    tk = 4
    tn = max(2, tiles // tk) // 2 * 2
    K, N, nw = 16 * tk, 16 * tn, 8 * KB
    rng = np.random.default_rng([seed, KB])
    words = rng.integers(0, 1 << 32, size=(tk, tn, nw), dtype=np.uint64).astype(np.uint32)
    wt = torch.from_numpy(words.view(np.int32).copy())
    # reference states in row-major tile order: position p holds element perm[p]
    ref = tq.unpack_states(wt, float(KB)).numpy()[..., tq.tensor_core_perm_inv()]   # [tk][tn][256]
    q = tq.decode_words(wt, float(KB), torch.from_numpy(tq.codebook_np("mul1").copy())).numpy()
    qbits = q.astype(np.float16).view(np.uint16)                                     # [K][N]
    grid = to_pair_grid(words)
    blk = 2 * nw
    lane_fn = lane_states_k4 if KB == 4 else lane_states_k5
    seen = np.zeros((K, N), dtype=np.int64)
    bad_state = bad_val = 0
    for pr in range(tn // 2):
        for kt in range(tk):
            block = grid[(pr * tk + kt) * blk:(pr * tk + kt + 1) * blk]
            for L in range(32):
                t, c, h = (L >> 3) & 1, L & 7, L >> 4
                for F, states in enumerate(lane_fn(block, L)):
                    vals = fragment_values(states)
                    col = c + 8 * F
                    for e in range(8):
                        kl = 8 * (e >> 2) + 4 * h + (e & 3)
                        k, n = 16 * kt + kl, 32 * pr + 16 * t + col
                        seen[k, n] += 1
                        bad_state += states[e] != ref[kt, 2 * pr + t, kl * 16 + col]
                        bad_val += vals[e] != qbits[k, n]
    if not np.all(seen == 1):
        raise AssertionError(f"KB={KB}: the lane map does not cover every (k, n) exactly once")
    return bad_state, bad_val, K * N


def check_value_path() -> tuple[int, float]:
    """Both halves of the value path against codebook_np for all 65,536 states (vectorized with the
    same integer steps); returns (mismatches, the AFFINE variant's max |codebook - exact affine|)."""
    s = np.arange(1 << 16, dtype=np.uint64)
    tlo = (s * np.uint64(KMLO & 0xFFFF)) & np.uint64(0xFFFFFFFF)
    lo = (np.uint64(KMHI & 0xFFFF) * s + (tlo & np.uint64(0xFFFF))) & np.uint64(0xFFFF)
    hi = (np.uint64(KMHI >> 16) * s + (tlo >> np.uint64(16))) & np.uint64(0xFFFF)
    x = (hi << np.uint64(16)) | lo
    if not np.array_equal(x, (s * np.uint64(tq.MUL1_MULT)) & np.uint64(0xFFFFFFFF)):
        return 1 << 16, float("nan")
    bs = sum((x >> np.uint64(8 * i)) & np.uint64(0xFF) for i in range(4))
    a, b = bs, np.roll(bs, 1)              # lo16 from state s, hi16 from state s - 1
    d = (np.uint64(ONESBIAS) + a + (b << np.uint64(16))) & np.uint64(0xFFFFFFFF)
    cb = tq.codebook_np("mul1").astype(np.float16).view(np.uint16)
    bad = 0
    for half, src in ((d & np.uint64(0xFFFF), a), (d >> np.uint64(16), b)):
        h16 = half.astype(np.uint16).view(np.float16).astype(np.float64)
        if not np.array_equal(h16, 1024.0 + src.astype(np.float64)):
            bad += 1 << 16
            continue
        v = (h16 * KINV_F + KBIAS_F).astype(np.float16).view(np.uint16)
        bad += int(np.sum(v != (cb if src is a else np.roll(cb, 1))))
    exact = (1024.0 + bs.astype(np.float64)) * KINV_F + KBIAS_F
    return bad, float(np.max(np.abs(tq.codebook_np("mul1").astype(np.float64) - exact)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tiles", type=int, default=64, help="random tiles per KB (4 k-tiles x tiles/4 n-tiles)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    bad = 0
    for KB in (4, 5):
        bs, bv, n = check_lanes(KB, args.tiles, args.seed)
        bad += bs + bv
        print(f"KB={KB}: {n} weights ({n // 256} tiles, 32 lanes x 8 elements x 2 fragments per tile pair "
              f"and k-tile): state mismatches {bs}, value mismatches {bv} "
              f"(vs unpack_states + tensor_core_perm, decode_words)")
    vb, dev = check_value_path()
    bad += vb
    print(f"value path (v_mad_u32_u16, v_pk_mad_u16, v_sad_u8/v_sad_hi_u8, v_pk_fma_f16) vs codebook_np, "
          f"all 65536 states, both halves: {vb} mismatches")
    print(f"AFFINE variant (B = 1024 + sum, affine in the epilogue): max |codebook - exact affine| = {dev:.3e}")
    print("LANE CHECK " + ("PASSED" if bad == 0 else f"FAILED ({bad})"))
    if torch.cuda.is_initialized():
        raise AssertionError("torch.cuda was initialized: this check must stay on the CPU")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
