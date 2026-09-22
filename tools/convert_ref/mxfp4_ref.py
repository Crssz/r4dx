"""Python reference for the OCP MXFP4 quantizer and packer -- byte-exact ground truth for
tools/convert_ref/selftest_compare.py.

Mirrors src/convert/include/r4dx_convert/quant_mxfp4.hpp; the fragment permutation is the same
math as C:\\Users\\user\\dev\\libr4d\\mxfp4_layout.py::permute_w, re-derived as index loops so
this file has no torch dependency (the reference venv's torch would work too, but this reference
is meant to be a from-scratch cross-check, not a re-import of the thing under test).
"""
import math

import numpy as np

F32 = np.float32
GROUP = 32
MAGNITUDE = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def encode_e2m1(v: float) -> int:
    neg = v < 0.0
    av = abs(v)
    best, best_d = 0, abs(av - MAGNITUDE[0])
    for i in range(1, 8):
        d = abs(av - MAGNITUDE[i])
        if d < best_d:
            best_d, best = d, i
    return (0x8 if neg else 0) | best


def quantize(w: np.ndarray, group: int = GROUP):
    """w [N,K] float32 -> (packed uint8 [N,K/2] checkpoint order, escale uint8 [N,K/group],
    wref uint8 [N])."""
    w = w.astype(F32)
    N, K = w.shape
    gpr = K // group
    packed = np.zeros((N, K // 2), dtype=np.uint8)
    escale = np.zeros((N, gpr), dtype=np.uint8)
    wref = np.zeros(N, dtype=np.uint8)
    for row in range(N):
        row_max_raw = 0
        for g in range(gpr):
            seg = w[row, g * group:(g + 1) * group]
            amax = float(np.max(np.abs(seg)))
            if amax <= 0.0:
                raw, scale = 0, 2.0 ** -127
            else:
                e = math.ceil(math.log2(amax / 6.0))
                raw = min(max(e + 127, 0), 254)
                scale = 2.0 ** (raw - 127)
            escale[row, g] = raw
            row_max_raw = max(row_max_raw, raw)
            for k in range(group):
                kk = g * group + k
                code = encode_e2m1(float(seg[k]) / scale)
                byte_idx = kk // 2
                if kk % 2 == 0:
                    packed[row, byte_idx] = (packed[row, byte_idx] & 0xF0) | (code & 0xF)
                else:
                    packed[row, byte_idx] = (packed[row, byte_idx] & 0x0F) | ((code & 0xF) << 4)
        wref[row] = row_max_raw
    return packed, escale, wref


def quantize_search(w: np.ndarray, group: int = GROUP, imatrix=None):
    """E8M0 exponent search (r4dx-convert --quant search): try `quantize`'s round-up exponent and
    that exponent minus one, keep whichever has the lower (optionally importance-weighted) squared
    error. Byte-exact mirror of r4dx_convert::QuantizeMxfp4Search.

    Unlike the w4 searches this accumulates in float64 on BOTH sides: `quantize` above already works
    in Python floats, and `sign * magnitude * 2**(raw-127)` is exactly representable either way, so
    float64 is the easier bit-exact contract here. The round-up exponent is scored first and kept on
    ties, so the search is never worse than plain `quantize`."""
    w = w.astype(F32)
    N, K = w.shape
    gpr = K // group
    imp = None if imatrix is None else np.asarray(imatrix, dtype=F32)
    packed = np.zeros((N, K // 2), dtype=np.uint8)
    escale = np.zeros((N, gpr), dtype=np.uint8)
    wref = np.zeros(N, dtype=np.uint8)
    for row in range(N):
        row_max_raw = 0
        for g in range(gpr):
            seg = w[row, g * group:(g + 1) * group]
            amax = float(np.max(np.abs(seg)))
            if amax <= 0.0:
                best_raw = 0
                best_codes = [encode_e2m1(float(seg[k]) / (2.0 ** -127)) for k in range(group)]
            else:
                e = math.ceil(math.log2(amax / 6.0))
                raw0 = min(max(e + 127, 0), 254)
                best_raw, best_err, best_codes = raw0, None, None
                for cand in range(2):
                    raw = raw0 if cand == 0 else raw0 - 1
                    if raw < 0:
                        continue
                    scale = 2.0 ** (raw - 127)
                    codes, err = [], 0.0
                    for k in range(group):
                        code = encode_e2m1(float(seg[k]) / scale)
                        codes.append(code)
                        mag = MAGNITUDE[code & 0x7]
                        recon = (-mag if (code & 0x8) else mag) * scale
                        d = float(seg[k]) - recon
                        wt = 1.0 if imp is None else float(imp[g * group + k])
                        err = err + wt * (d * d)
                    if cand == 0:
                        best_err, best_codes = err, codes
                    elif err < best_err:
                        best_err, best_raw, best_codes = err, raw, codes
            escale[row, g] = best_raw
            row_max_raw = max(row_max_raw, best_raw)
            for k in range(group):
                kk = g * group + k
                byte_idx = kk // 2
                if kk % 2 == 0:
                    packed[row, byte_idx] = (packed[row, byte_idx] & 0xF0) | (best_codes[k] & 0xF)
                else:
                    packed[row, byte_idx] = (packed[row, byte_idx] & 0x0F) | ((best_codes[k] & 0xF) << 4)
        wref[row] = row_max_raw
    return packed, escale, wref


def permute_wq(packed: np.ndarray, N: int, K: int) -> np.ndarray:
    ntiles, ksteps = N // 16, K // 16
    wq = np.zeros(ntiles * ksteps * 32 * 4, dtype=np.uint8)
    idx = 0
    for nt in range(ntiles):
        for ks in range(ksteps):
            for l in range(32):
                r, h = l & 15, l >> 4
                row = nt * 16 + r
                src = ks * 8 + 4 * h
                wq[idx:idx + 4] = packed[row, src:src + 4]
                idx += 4
    return wq


def pack_ws(escale: np.ndarray, N: int, K: int, group: int = GROUP) -> np.ndarray:
    gpr = K // group
    ws = np.zeros(gpr * N, dtype=np.uint8)
    for n in range(N):
        for g in range(gpr):
            ws[g * N + n] = escale[n, g]
    return ws
