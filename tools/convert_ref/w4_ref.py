"""Python reference for the w4a16 / w4a8 quantizer and packer -- the byte-exact ground truth
tools/convert_ref/selftest_compare.py diffs r4dx-convert's C++ output against.

Mirrors src/convert/include/r4dx_convert/quant_int4.hpp function-for-function; see that header's
comment block for why w4a16 (asymmetric, free zero) and w4a8 (symmetric, zero pinned to 8) are
quantized independently rather than sharing one packed weight, and for the provenance of the
`_KOFF` fragment-order table (third_party/libr4d/r4d_gemm_w4a16_nt_m64.hip's dequant(),
cross-checked against C:\\Users\\user\\dev\\vllm-radiance\\radiance_w4.py's pack()).
"""
import numpy as np

from common import F32, round_half_away_from_zero, f16_bits

GROUP = 128
KPB = 64
KOFF = (0, 8, 1, 9, 2, 10, 3, 11)


def quantize_asymmetric(w: np.ndarray, group: int = GROUP):
    """w [N,K] float32 -> (q uint8 [N,K] in 0..15, scale f32 [N,K/group], zero uint8 [N,K/group])."""
    w = w.astype(F32)
    N, K = w.shape
    gpr = K // group
    w3 = w.reshape(N, gpr, group)
    wmin = w3.min(axis=2).astype(F32)
    wmax = w3.max(axis=2).astype(F32)
    rng = (wmax - wmin).astype(F32)
    scale = (np.maximum(rng, F32(1e-12)) / F32(15.0)).astype(F32)
    zero = np.clip(round_half_away_from_zero(-wmin / scale), 0, 15).astype(np.uint8)
    # Task brief's literal formula: q = clamp(round(w/scale) + zero, 0, 15) -- round FIRST, then
    # shift by the integer zero point (not round(w/scale + zero); the two differ exactly on a
    # w/scale + zero_fraction... case, i.e. whenever w/scale lands on a half-integer, which a
    # uniform random fixture hits often enough to matter for byte-exactness).
    rq = round_half_away_from_zero((w3 / scale[:, :, None]).astype(F32))
    q = np.clip(rq + zero[:, :, None].astype(np.int32), 0, 15).astype(np.uint8).reshape(N, K)
    return q, scale, zero


def quantize_symmetric_pinned8(w: np.ndarray, group: int = GROUP):
    """w [N,K] float32 -> (q uint8 [N,K] in 0..15 offset-binary, scale f32 [N,K/group])."""
    w = w.astype(F32)
    N, K = w.shape
    gpr = K // group
    w3 = w.reshape(N, gpr, group)
    amax = np.abs(w3).max(axis=2).astype(F32)
    scale = (np.maximum(amax, F32(1e-12)) / F32(7.0)).astype(F32)
    qs = np.clip(round_half_away_from_zero((w3 / scale[:, :, None]).astype(F32)), -8, 7)
    q = (qs + 8).astype(np.uint8).reshape(N, K)
    return q, scale


# ---- error-minimizing search (r4dx-convert --quant search) -------------------------------------
#
# Byte-exact mirror of src/convert/include/r4dx_convert/quant_search.hpp. The BYTE LAYOUT is
# unchanged -- these return the same (q, scale, zero) triples the RTN quantizers above return, for
# the same pack_* functions; only the chosen values differ.
#
# Bit-exactness needs the C++'s evaluation order, and the C++ accumulates the error sequentially
# over k = 0 .. group-1 in float32. numpy's `np.sum` is PAIRWISE, so summing an axis here would
# silently disagree on near-ties. Every accumulator below therefore loops over k and vectorizes
# across GROUPS instead -- same arithmetic order as the C++, one numpy op per k.

SEARCH_STEPS = 21


def search_scale_mult(step: int) -> np.float32:
    """0.85 .. 1.15 in 21 binary32 steps; step 10 is EXACTLY 1.0f, so the RTN grid is a genuine
    member of the candidate set (quant_search.hpp kW4SearchSteps)."""
    return (F32(0.85) + F32(0.3) * (F32(step) / F32(20.0))).astype(F32)


def _weights(imatrix, N: int, K: int, group: int) -> np.ndarray:
    """-> float32 [N*K/group, group] importance, row i = (row n, group g) with i = n*gpr + g."""
    gpr = K // group
    if imatrix is None:
        return np.ones((N * gpr, group), dtype=F32)
    imat = np.asarray(imatrix, dtype=F32).reshape(gpr, group)
    return np.tile(imat, (N, 1)).astype(F32)


def _werr(x, wt, sc, q, z) -> np.ndarray:
    """sum_k wt_k * (x_k - sc*(q_k - z_k))^2, float32, accumulated sequentially over k."""
    group = x.shape[1]
    t = (q - z[:, None]).astype(np.int32).astype(F32)
    err = np.zeros(x.shape[0], dtype=F32)
    for k in range(group):
        d = (x[:, k] - (sc * t[:, k]).astype(F32)).astype(F32)
        err = (err + (wt[:, k] * (d * d).astype(F32)).astype(F32)).astype(F32)
    return err


def _refit(x, wt, sc_in, q, z, err_in):
    """One weighted least-squares refit of the float scale with q and z held fixed; kept only if it
    lowers the error (quant_search.hpp's final block)."""
    group = x.shape[1]
    t = (q - z[:, None]).astype(np.int32).astype(F32)
    num = np.zeros(x.shape[0], dtype=F32)
    den = np.zeros(x.shape[0], dtype=F32)
    for k in range(group):
        num = (num + ((wt[:, k] * x[:, k]).astype(F32) * t[:, k]).astype(F32)).astype(F32)
        den = (den + (wt[:, k] * (t[:, k] * t[:, k]).astype(F32)).astype(F32)).astype(F32)
    ok = den > F32(0.0)
    sc2 = np.where(ok, num / np.where(ok, den, F32(1.0)).astype(F32), sc_in).astype(F32)
    ok = ok & (sc2 > F32(0.0))
    sc2 = np.where(ok, sc2, sc_in).astype(F32)
    err2 = _werr(x, wt, sc2, q, z)
    better = ok & (err2 < err_in)
    return np.where(better, sc2, sc_in).astype(F32)


def quantize_asymmetric_search(w: np.ndarray, group: int = GROUP, imatrix=None):
    """w [N,K] float32 (+ optional float32 [K] importance) -> (q, scale, zero), same shapes and
    same meaning as quantize_asymmetric."""
    w = w.astype(F32)
    N, K = w.shape
    gpr = K // group
    x = w.reshape(N * gpr, group).astype(F32)
    wt = _weights(imatrix, N, K, group)

    wmin = x.min(axis=1).astype(F32)
    wmax = x.max(axis=1).astype(F32)
    rng = (wmax - wmin).astype(F32)
    degenerate = rng <= F32(0.0)

    # Degenerate rows take the special case at the bottom; give them a harmless placeholder scale
    # here so `x / s0` cannot overflow int32 on the way to being discarded (the C++ short-circuits
    # them before the search instead).
    s0 = (np.maximum(rng, F32(1e-12)) / F32(15.0)).astype(F32)
    s0 = np.where(degenerate, F32(1.0), s0).astype(F32)

    # candidate 0 = the RTN grid (incumbent; later candidates must beat it STRICTLY)
    best_z = np.clip(round_half_away_from_zero((-wmin / s0).astype(F32)), 0, 15).astype(np.int32)
    best_q = np.clip(round_half_away_from_zero((x / s0[:, None]).astype(F32)) + best_z[:, None],
                     0, 15).astype(np.int32)
    best_sc = s0.copy()
    best_err = _werr(x, wt, best_sc, best_q, best_z)

    for step in range(SEARCH_STEPS):
        sc = (s0 * search_scale_mult(step)).astype(F32)
        rq = round_half_away_from_zero((x / sc[:, None]).astype(F32))
        zc = round_half_away_from_zero((-wmin / sc).astype(F32))
        for dz in (-1, 0, 1):
            z = np.clip(zc + dz, 0, 15).astype(np.int32)
            q = np.clip(rq + z[:, None], 0, 15).astype(np.int32)
            err = _werr(x, wt, sc, q, z)
            better = err < best_err
            best_err = np.where(better, err, best_err).astype(F32)
            best_sc = np.where(better, sc, best_sc).astype(F32)
            best_z = np.where(better, z, best_z).astype(np.int32)
            best_q = np.where(better[:, None], q, best_q).astype(np.int32)

    best_sc = _refit(x, wt, best_sc, best_q, best_z, best_err)

    # Degenerate (max == min) groups bypass the search and take QuantizeInt4Asymmetric's own
    # special case verbatim -- searching a zero-width range only picks an arbitrary member of a set
    # of equally-perfect candidates.
    if degenerate.any():
        zero_grp = degenerate & (wmax == F32(0.0))
        const_grp = degenerate & ~zero_grp
        sc_d = np.abs(wmax).astype(F32)
        zp_d = np.where(wmax >= F32(0.0), 0, 15).astype(np.int32)
        sc_safe = np.where(sc_d > F32(0.0), sc_d, F32(1.0)).astype(F32)
        q_d = np.clip(round_half_away_from_zero((x / sc_safe[:, None]).astype(F32))
                      + zp_d[:, None], 0, 15).astype(np.int32)
        best_sc = np.where(const_grp, sc_d, np.where(zero_grp, F32(0.0), best_sc)).astype(F32)
        best_z = np.where(const_grp, zp_d, np.where(zero_grp, 0, best_z)).astype(np.int32)
        best_q = np.where(const_grp[:, None], q_d,
                          np.where(zero_grp[:, None], 0, best_q)).astype(np.int32)

    q_out = best_q.astype(np.uint8).reshape(N, K)
    scale = best_sc.reshape(N, gpr).astype(F32)
    zero = best_z.astype(np.uint8).reshape(N, gpr)
    return q_out, scale, zero


def quantize_symmetric_pinned8_search(w: np.ndarray, group: int = GROUP, imatrix=None):
    """w [N,K] float32 (+ optional float32 [K] importance) -> (q, scale). Zero stays PINNED to 8 --
    r4d_gemm_w4a8_nt_m64's dequant8() has no zero-point input -- so only the scale is searched."""
    w = w.astype(F32)
    N, K = w.shape
    gpr = K // group
    x = w.reshape(N * gpr, group).astype(F32)
    wt = _weights(imatrix, N, K, group)
    z8 = np.full(x.shape[0], 8, dtype=np.int32)

    amax = np.abs(x).max(axis=1).astype(F32)
    all_zero = amax <= F32(0.0)
    s0 = (np.maximum(amax, F32(1e-12)) / F32(7.0)).astype(F32)

    best_sc = s0.copy()
    best_q = (np.clip(round_half_away_from_zero((x / s0[:, None]).astype(F32)), -8, 7)
              + 8).astype(np.int32)
    best_err = _werr(x, wt, best_sc, best_q, z8)

    for step in range(SEARCH_STEPS):
        sc = (s0 * search_scale_mult(step)).astype(F32)
        q = (np.clip(round_half_away_from_zero((x / sc[:, None]).astype(F32)), -8, 7)
             + 8).astype(np.int32)
        err = _werr(x, wt, sc, q, z8)
        better = err < best_err
        best_err = np.where(better, err, best_err).astype(F32)
        best_sc = np.where(better, sc, best_sc).astype(F32)
        best_q = np.where(better[:, None], q, best_q).astype(np.int32)

    best_sc = _refit(x, wt, best_sc, best_q, z8, best_err)

    if all_zero.any():  # every code is 8 and the scale stays at the 1e-12 floor, as in RTN
        best_sc = np.where(all_zero, s0, best_sc).astype(F32)
        best_q = np.where(all_zero[:, None], 8, best_q).astype(np.int32)

    return best_q.astype(np.uint8).reshape(N, K), best_sc.reshape(N, gpr).astype(F32)


def pack_nibbles(q: np.ndarray, N: int, K: int) -> np.ndarray:
    """q [N,K] uint8 in 0..15 (offset-binary) -> wq uint32[N*K/8], fragment order."""
    ntiles, kblocks = N // 16, K // KPB
    wq = np.zeros(ntiles * kblocks * 2 * 16 * 4, dtype=np.uint32)
    idx = 0
    for t in range(ntiles):
        for kb in range(kblocks):
            for lh in range(2):
                for r in range(16):
                    row = t * 16 + r
                    for s in range(4):
                        dword = 0
                        for i in range(8):
                            kk = 4 * lh + KOFF[i]
                            k = kb * KPB + s * 16 + kk
                            nib = (int(q[row, k]) ^ 0x8) & 0xF
                            dword |= nib << (4 * i)
                        wq[idx] = dword
                        idx += 1
    return wq


def pack_w4a16_scales(scale: np.ndarray, zero: np.ndarray, N: int, K: int, group: int = GROUP) -> np.ndarray:
    ntiles, gpr = N // 16, K // group
    wsz = np.zeros(ntiles * gpr * 16, dtype=np.uint32)
    idx = 0
    for t in range(ntiles):
        for g in range(gpr):
            for r in range(16):
                row = t * 16 + r
                sc16 = f16_bits(scale[row, g])
                nz16 = f16_bits(-(1024.0 + float(zero[row, g])))
                wsz[idx] = (sc16 & 0xFFFF) | ((nz16 & 0xFFFF) << 16)
                idx += 1
    return wsz


def pack_w4a8_scales(scale: np.ndarray, N: int, K: int, group: int = GROUP) -> np.ndarray:
    ntiles, gpr = N // 16, K // group
    ws = np.zeros(ntiles * gpr * 16, dtype=np.uint32)
    idx = 0
    for t in range(ntiles):
        for g in range(gpr):
            for r in range(16):
                row = t * 16 + r
                ws[idx] = f16_bits(scale[row, g]) & 0xFFFF
                idx += 1
    return ws
