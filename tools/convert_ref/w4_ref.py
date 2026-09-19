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
