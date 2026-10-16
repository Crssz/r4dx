"""Python reference for the w8a16 quantizer, packer and dequantizer -- the byte-exact ground truth
tools/convert_ref/selftest_compare.py (--w8a16) and tests/convert/test_quant_int8.cpp diff
r4dx-convert's C++ output against.

Mirrors src/convert/include/r4dx_convert/quant_int8.hpp (docs/w8a16.md sections 2 and 3.2) step for
step, in float32 throughout (a stray float64 changes the last bit of a division):

    amax  = max |w| over the group (f32)
    s     = amax / 127
    s16   = f16(s * 2^12)                    # RNE; inf -> error
    if s16 == 0: scale 0, every code 0, counted as a zero group
    s_use = float(s16) * 2^-12               # quantize against the STORED scale
    q     = clamp(round_half_away(w / s_use), -127, 127)
    byte  = uint8(q + 128)                   # offset binary

Packed weight order (the kernel's fragment order, docs/w8a16.md 3.2): byte index
    (((t*(K/32) + kb)*32 + l)*4 + j)*4 + i      t = row tile, kb = 32-K block, l = lane, j = dword, i = byte
holds code(row = 16t + (l & 15), k = 32kb + 16*(j >> 1) + 8*(j & 1) + 4*(l >> 4) + i).
Scales: u16 index ((t*(K/G) + g)*16 + r) = f16 bits of s*2^12 of row 16t + r, group g.
"""
import numpy as np

from common import F32, round_half_away_from_zero

SCALE_LOG2_BIAS = 12
BIAS = F32(2.0 ** SCALE_LOG2_BIAS)
UNBIAS = F32(2.0 ** -SCALE_LOG2_BIAS)
GROUP = 128
CODE_OFFSET = 128


def quantize_int8_symmetric(w: np.ndarray, group: int = GROUP):
    """w [N,K] float32 -> (q uint8 [N,K] offset-binary, scale16 uint16 [N,K/group], zero_groups int).
    Raises ValueError on a non-finite weight or a biased scale that overflows f16."""
    w = w.astype(F32)
    N, K = w.shape
    if K % group != 0:
        raise ValueError(f"K={K} is not divisible by group={group}")
    gpr = K // group
    if not np.isfinite(w).all():
        raise ValueError("non-finite weight")
    w3 = w.reshape(N, gpr, group)
    amax = np.abs(w3).max(axis=2).astype(F32)
    s = (amax / F32(127.0)).astype(F32)
    biased = (s * BIAS).astype(F32)
    with np.errstate(over="ignore"):
        s16 = biased.astype(np.float16)
    if np.isinf(s16).any():
        raise ValueError(f"w8a16 group amax {float(amax.max())} overflows the biased f16 scale")
    zero = s16 == np.float16(0.0)
    s_use = (s16.astype(F32) * UNBIAS).astype(F32)
    s_safe = np.where(zero, F32(1.0), s_use).astype(F32)
    rq = round_half_away_from_zero((w3 / s_safe[:, :, None]).astype(F32))
    q = np.clip(rq, -127, 127)
    q = np.where(zero[:, :, None], 0, q)
    q8 = (q + CODE_OFFSET).astype(np.uint8).reshape(N, K)
    return q8, s16.view(np.uint16).reshape(N, gpr), int(zero.sum())


def pack_w8_bytes(q: np.ndarray, N: int, K: int) -> np.ndarray:
    """q [N,K] uint8 offset-binary -> wq uint8 [N*K] in the kernel's fragment order."""
    ntiles, kblocks = N // 16, K // 32
    wq = np.zeros(N * K, dtype=np.uint8)
    for t in range(ntiles):
        for kb in range(kblocks):
            for lane in range(32):
                row = t * 16 + (lane & 15)
                for j in range(4):
                    for i in range(4):
                        k = 32 * kb + 16 * (j >> 1) + 8 * (j & 1) + 4 * (lane >> 4) + i
                        wq[(((t * kblocks + kb) * 32 + lane) * 4 + j) * 4 + i] = q[row, k]
    return wq


def pack_w8a16_scales(scale16: np.ndarray, N: int, K: int, group: int = GROUP) -> np.ndarray:
    """scale16 uint16 [N,K/group] -> ws uint16 [ntiles*(K/group)*16] in (tile, group, row) order."""
    ntiles, gpr = N // 16, K // group
    ws = np.zeros(ntiles * gpr * 16, dtype=np.uint16)
    idx = 0
    for t in range(ntiles):
        for g in range(gpr):
            for r in range(16):
                ws[idx] = scale16[t * 16 + r, g]
                idx += 1
    return ws


def dequant_w8a16(wq: np.ndarray, ws: np.ndarray, N: int, K: int, group: int = GROUP) -> np.ndarray:
    """Packed tensors -> float32 [N,K], walking the kernel's lanes."""
    ntiles, kblocks, gpr = N // 16, K // 32, K // group
    w = np.zeros((N, K), dtype=F32)
    ws_f = ws.view(np.float16).astype(F32)
    for t in range(ntiles):
        for kb in range(kblocks):
            for lane in range(32):
                row = t * 16 + (lane & 15)
                for j in range(4):
                    for i in range(4):
                        k = 32 * kb + 16 * (j >> 1) + 8 * (j & 1) + 4 * (lane >> 4) + i
                        code = int(wq[(((t * kblocks + kb) * 32 + lane) * 4 + j) * 4 + i]) - CODE_OFFSET
                        s_use = F32(ws_f[(t * gpr + k // group) * 16 + (lane & 15)]) * UNBIAS
                        w[row, k] = F32(code) * s_use
    return w
