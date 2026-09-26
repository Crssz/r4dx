"""tools/reference/gguf_dequant.py

Memory-mapped GGUF reader and numpy dequantizers, plus the llama.cpp `qwen35` <-> HF
`Qwen3_5ForConditionalGeneration` tensor map with its inverse layout transforms. Used to put a
llama.cpp GGUF's *quantized weights* into the bf16 reference forward (`full_logits_golden.py
--weights-gguf`), so the GGUF can be scored with `kl_report.py` on our own `kl_corpus/` tokens.
`gguf_validate.py` checks every mapped tensor against the bf16 checkpoint.

numpy only: no `gguf` package, no torch. The file is never read whole. The header is parsed out of an
`mmap` and each tensor's bytes are `numpy.frombuffer` views into the same map, so a 16 GiB GGUF costs
only the pages actually touched.

Block layouts (ggml-common.h; little-endian; `f16` is IEEE half). Each block covers `block_size`
consecutive elements of one row (`ne[0]` is the fastest axis and a multiple of the block size):

  Q8_0   (32 elts, 34 B)   d:f16, qs:int8[32]                      y = d*q
  Q4_K  (256 elts, 144 B)  d:f16, dmin:f16, scales:u8[12], qs:u8[128]
         8 sub-blocks of 32. sub-block s has a 6-bit scale sc[s] and min m[s], packed:
           s<4:  sc = scales[s] & 63,            m = scales[s+4] & 63
           s>=4: sc = (scales[s+4] & 15) | (scales[s-4] >> 6) << 4
                 m  = (scales[s+4] >> 4) | (scales[s]   >> 6) << 4
         q (4 bit): sub-blocks 2k and 2k+1 share qs[32k:32k+32], low nibble then high nibble.
         y = (d*sc)*q - (dmin*m)
  Q5_K  (256 elts, 176 B)  d, dmin, scales:u8[12], qh:u8[32], qs:u8[128]
         as Q4_K, plus the 5th bit of element 32s+l in bit s of qh[l].
  Q6_K  (256 elts, 210 B)  ql:u8[128], qh:u8[64], scales:int8[16], d:f16
         16 sub-blocks of 16, scale sc[e//16]. For each 128-element half n (ql += 64, qh += 32) and
         l<32: element l uses the low nibble of ql[l] and qh[l] bits 0-1, l+32 the low nibble of
         ql[l+32] and bits 2-3, l+64 the high nibble of ql[l] and bits 4-5, l+96 the high nibble of
         ql[l+32] and bits 6-7. y = (d*sc)*(q - 32)
  Q3_K  (256 elts, 110 B)  hmask:u8[32], qs:u8[64], scales:u8[12], d:f16
         16 sub-blocks of 16. 6-bit scale j: low 4 bits = scales[j] & 15 (j<8) or scales[j-8] >> 4
         (j>=8); high 2 bits = (scales[8 + j%4] >> 2*(j//4)) & 3. Element e = 128n + 32j + l:
         low 2 bits = (qs[32n + l] >> 2j) & 3, high bit = (hmask[l] >> (4n + j)) & 1.
         y = d*(sc - 32) * (low2 - (hbit ? 0 : 4))
  IQ4_NL (32 elts, 18 B)   d:f16, qs:u8[16]                        y = d*KV[idx]
         elements 0-15 are the low nibbles of qs, 16-31 the high nibbles; KV = kvalues_iq4nl.
  IQ4_XS (256 elts, 136 B) d:f16, scales_h:u16, scales_l:u8[4], qs:u8[128]
         8 sub-blocks of 32, each laid out like one IQ4_NL block's qs; 6-bit scale
         ls = ((scales_l[ib//2] >> 4*(ib%2)) & 15) | ((scales_h >> 2*ib) & 3) << 4.
         y = d*(ls - 32) * KV[idx]
  IQ3_S (256 elts, 110 B)  d:f16, qs:u8[64], qh:u8[8], signs:u8[32], scales:u8[4]
         8 sub-blocks of 32 = 8 grid points of 4. Grid index of point k of sub-block ib is
         qs[8ib + k] | ((qh[ib] >> k) & 1) << 8, into the 512-entry `iq3s_grid` (4 unsigned bytes
         per entry). Element e of the sub-block is negated when bit e%8 of signs[4ib + e//8] is set.
         y = d*(1 + 2*((scales[ib//2] >> 4*(ib%2)) & 15)) * grid[idx][e%4] * sign
  F32 / F16 / BF16: plain little-endian values.

The two lookup tables: kvalues_iq4nl (16 int8) is small and stated in the spec, so it is written
out below. iq3s_grid is NOT derivable. It is a searched selection of 512 of the 8^4 points with odd
coordinates 1..15, so nothing here reconstructs it. It is read from a local ggml source tree
(ggml-common.h; `$env:R4DX_GGML_COMMON_H`, else ~/dev/ROCmFPX/ggml/src/ggml-common.h) and checked
against a pinned SHA-256, which is also the hash of gguf-py's independently encoded copy of the
same table. Without that file, IQ3_S tensors raise instead of decoding.

Every dequantizer does its arithmetic in float32 in the same order as ggml's
`dequantize_row_*`, so the output is bit-identical to ggml's reference dequantization.
"""

from __future__ import annotations

import hashlib
import mmap
import os
import re
import struct
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

GGUF_MAGIC = 0x46554747  # b"GGUF" little-endian
GGUF_VERSIONS = (2, 3)
DEFAULT_ALIGNMENT = 32

# GGUF metadata value types.
(VT_UINT8, VT_INT8, VT_UINT16, VT_INT16, VT_UINT32, VT_INT32, VT_FLOAT32, VT_BOOL, VT_STRING,
 VT_ARRAY, VT_UINT64, VT_INT64, VT_FLOAT64) = range(13)
_VT_STRUCT = {VT_UINT8: "<B", VT_INT8: "<b", VT_UINT16: "<H", VT_INT16: "<h", VT_UINT32: "<I",
              VT_INT32: "<i", VT_FLOAT32: "<f", VT_BOOL: "<?", VT_UINT64: "<Q", VT_INT64: "<q",
              VT_FLOAT64: "<d"}
_VT_NUMPY = {VT_UINT8: "<u1", VT_INT8: "<i1", VT_UINT16: "<u2", VT_INT16: "<i2", VT_UINT32: "<u4",
             VT_INT32: "<i4", VT_FLOAT32: "<f4", VT_BOOL: "<u1", VT_UINT64: "<u8", VT_INT64: "<i8",
             VT_FLOAT64: "<f8"}


@dataclass(frozen=True)
class GGMLType:
    name: str
    block_size: int  # elements per block
    type_size: int   # bytes per block


#: ggml_type enum (ggml.h) -> layout. Sizes are needed to walk any file's tensor table; only the
#: types in DEQUANTIZERS below can be decoded.
GGML_TYPES: dict[int, GGMLType] = {t[0]: GGMLType(*t[1:]) for t in [
    (0, "F32", 1, 4), (1, "F16", 1, 2), (2, "Q4_0", 32, 18), (3, "Q4_1", 32, 20),
    (6, "Q5_0", 32, 22), (7, "Q5_1", 32, 24), (8, "Q8_0", 32, 34), (9, "Q8_1", 32, 36),
    (10, "Q2_K", 256, 84), (11, "Q3_K", 256, 110), (12, "Q4_K", 256, 144), (13, "Q5_K", 256, 176),
    (14, "Q6_K", 256, 210), (15, "Q8_K", 256, 292), (16, "IQ2_XXS", 256, 66),
    (17, "IQ2_XS", 256, 74), (18, "IQ3_XXS", 256, 98), (19, "IQ1_S", 256, 50),
    (20, "IQ4_NL", 32, 18), (21, "IQ3_S", 256, 110), (22, "IQ2_S", 256, 82),
    (23, "IQ4_XS", 256, 136), (24, "I8", 1, 1), (25, "I16", 1, 2), (26, "I32", 1, 4),
    (27, "I64", 1, 8), (28, "F64", 1, 8), (29, "IQ1_M", 256, 56), (30, "BF16", 1, 2),
    (34, "TQ1_0", 256, 54), (35, "TQ2_0", 256, 66), (39, "MXFP4", 32, 17),
]}

#: Types stored without loss relative to a bf16 checkpoint (F32 holds every bf16 value exactly).
LOSSLESS_TYPES = ("F32", "BF16")

#: Per-type bounds on a dequantized tensor's error against the bf16 checkpoint:
#: (max relative Frobenius error ||deq - W|| / ||W||, min cosine similarity).
#:
#: These bounds are FITTED to the one file they are used on. Over all 506 quantized tensors of
#: Unsloth's Qwen3.8-27B-UD-Q4_K_XL, the worst per type was Q8_0 0.0069, Q6_K 0.0242, Q5_K 0.0401,
#: Q4_K 0.0789, IQ4_NL 0.0853, IQ4_XS 0.0904, Q3_K 0.161 and IQ3_S 0.147. The rel bounds leave
#: ~25-45% headroom over those; each cos bound sits just under 1/sqrt(1 + rel_max^2), the cosine
#: of an error of that size orthogonal to W. So "every tensor is inside its bound" is partly
#: circular for this file and is only a tripwire. It catches a mapping or layout error (a missed
#: V-head reorder, a transposed or misnamed tensor, which lands at rel 0.28-1.4 with a cosine far
#: below these) and a GGUF of a different model. The independent evidence that the decode is right
#: is elsewhere: bit-exactness against ggml's own C `dequantize_row_*` (tests/reference/
#: test_gguf_dequant.py part (f)) and against gguf-py (part (b)), and the V-head reorder's
#: rel_no_inverse >= 1.05 in gguf_validate.py.
WEIGHT_ERROR_BOUNDS: dict[str, tuple[float, float]] = {
    "BF16": (0.0, 1.0),
    "F16": (2e-3, 0.99999),
    "Q8_0": (0.010, 0.9999),
    "Q6_K": (0.035, 0.999),
    "Q5_K": (0.055, 0.998),
    "Q4_K": (0.100, 0.995),
    "IQ4_NL": (0.110, 0.994),
    "IQ4_XS": (0.115, 0.993),
    "Q3_K": (0.200, 0.980),
    "IQ3_S": (0.200, 0.980),
}

#: A tensor-level rel error pools every row, so one catastrophic row is invisible in it. A ROW is
#: an outlier when its own rel error ||deq_r - W_r|| / ||W_r|| exceeds this factor times its type's
#: tensor bound (Q4_K: 0.2). In the Unsloth file the worst is token_embd (Q4_K) row 107517. Its
#: bf16 values are ~1e-5 (||w|| 8.1e-4, the median row's is 0.93), so every one of its 20
#: super-blocks' f16 d underflowed to 0. The row decodes to its -dmin*m offsets only, and its rel
#: error is 1.40. ggml's C decodes it identically. No kl_corpus token uses it; the worst used rows
#: are at 0.106 (tokens.json) and 0.102 (tokens_thai_canon.json). gguf_validate.py lists every
#: outlier row of every tensor: in this file 9 rows of 5 tensors, all decoded identically by ggml's
#: C (see tools/reference/README.md).
ROW_ERROR_FACTOR = 2.0

#: Byte offset of the f16 scale `d` in each type's block.
SCALE_D_OFFSET = {"Q8_0": 0, "Q4_K": 0, "Q5_K": 0, "Q6_K": 208, "Q3_K": 108, "IQ4_NL": 0,
                  "IQ4_XS": 0, "IQ3_S": 0}
#: f16 subnormals are multiples of 2^-24, so a block scale d below 2^-20 (~9.5e-7) keeps at most 4
#: significant bits, and one of ~1e-7 keeps 1 or 2, or is 0: the scale alone is off by up to
#: 6-50%. A row whose blocks ALL have such a coarse d holds values of ~1e-5 or less, below what the
#: type's f16 scale resolves, so a large relative error on it is the file's, not the decoder's.
#: (Merely subnormal is not enough: Q6_K's d = amax / 4064 is subnormal for any block with
#: amax < 0.25, which is most of them, and still keeps ~9-10 bits.)
F16_COARSE_SCALE = 2.0 ** -20


def row_error_bound(type_name: str) -> float | None:
    """The per-row rel error above which a row of a `type_name` tensor is an outlier."""
    b = WEIGHT_ERROR_BOUNDS.get(type_name)
    return None if b is None else ROW_ERROR_FACTOR * b[0]


class GGUFError(RuntimeError):
    pass


# ================================================================================================
# Dequantizers: uint8 blocks [nb, type_size] -> float32 [nb, block_size]
# ================================================================================================

#: kvalues_iq4nl (ggml-common.h): the non-linear 4-bit codebook of IQ4_NL and IQ4_XS.
KVALUES_IQ4NL = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113],
                         dtype=np.float32)

#: SHA-256 of iq3s_grid as 512 little-endian uint32 (ggml-common.h's table, byte-identical to the
#: grid gguf-py's quants.py decodes from its own hex string).
IQ3S_GRID_SHA256 = "bd1af4945a1717c65610b0284e4628b9a1ba3b306fae3a06f6e5f597356e349f"
_iq3s_grid_cache: np.ndarray | None = None


def ggml_common_h_candidates() -> list[Path]:
    env = os.environ.get("R4DX_GGML_COMMON_H")
    cands = [Path(env)] if env else []
    cands.append(Path.home() / "dev" / "ROCmFPX" / "ggml" / "src" / "ggml-common.h")
    return cands


def parse_iq3s_grid(text: str) -> np.ndarray:
    """iq3s_grid from ggml-common.h source text -> uint8 [512, 4] (entry k's byte j is element j)."""
    m = re.search(r"GGML_TABLE_BEGIN\(\s*uint32_t\s*,\s*iq3s_grid\s*,\s*512\s*\)(.*?)GGML_TABLE_END",
                  text, re.S)
    if not m:
        raise GGUFError("no `GGML_TABLE_BEGIN(uint32_t, iq3s_grid, 512)` table in that ggml-common.h")
    vals = [int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", m.group(1))]
    table = np.array(vals, dtype="<u4")
    digest = hashlib.sha256(table.tobytes()).hexdigest()
    if len(vals) != 512 or digest != IQ3S_GRID_SHA256:
        raise GGUFError(f"iq3s_grid has {len(vals)} entries and sha256 {digest}; expected 512 "
                        f"entries hashing to {IQ3S_GRID_SHA256}")
    return table.view(np.uint8).reshape(512, 4)


def iq3s_grid() -> np.ndarray:
    global _iq3s_grid_cache
    if _iq3s_grid_cache is None:
        tried = []
        for p in ggml_common_h_candidates():
            tried.append(str(p))
            if p.is_file():
                _iq3s_grid_cache = parse_iq3s_grid(p.read_text(encoding="utf-8", errors="replace"))
                break
        else:
            raise GGUFError(
                "IQ3_S needs ggml's 512-entry iq3s_grid lookup table, which is not derivable from a "
                "formula and is not bundled here. Point $env:R4DX_GGML_COMMON_H at a ggml-common.h "
                f"(tried: {tried}).")
    return _iq3s_grid_cache


def _f16_field(b: np.ndarray, off: int) -> np.ndarray:
    """The f16 at byte `off` of every block -> float32 [nb]."""
    return np.ascontiguousarray(b[:, off:off + 2]).view("<f2")[:, 0].astype(np.float32)


def _nibbles(qs: np.ndarray) -> np.ndarray:
    """uint8 [..., n] -> uint8 [..., 2, n]: the low nibbles, then the high nibbles."""
    out = np.empty(qs.shape[:-1] + (2, qs.shape[-1]), dtype=np.uint8)
    np.bitwise_and(qs, 0x0F, out=out[..., 0, :])
    np.right_shift(qs, 4, out=out[..., 1, :])
    return out


def _k4_scales_mins(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Q4_K / Q5_K 12-byte scale field [nb, 12] -> (sc, m) float32 [nb, 8] (get_scale_min_k4)."""
    nb = s.shape[0]
    sc = np.empty((nb, 8), dtype=np.uint8)
    mn = np.empty((nb, 8), dtype=np.uint8)
    sc[:, :4] = s[:, 0:4] & 63
    mn[:, :4] = s[:, 4:8] & 63
    sc[:, 4:] = (s[:, 8:12] & 0x0F) | ((s[:, 0:4] >> 6) << 4)
    mn[:, 4:] = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return sc.astype(np.float32), mn.astype(np.float32)


def dequant_f32(b):
    return np.ascontiguousarray(b).view("<f4").astype(np.float32, copy=False)


def dequant_f16(b):
    return np.ascontiguousarray(b).view("<f2").astype(np.float32)


def dequant_bf16(b):
    u = np.ascontiguousarray(b).view("<u2").astype(np.uint32) << 16
    return u.view(np.float32)


def dequant_q8_0(b):
    d = _f16_field(b, 0)
    return b[:, 2:34].view(np.int8).astype(np.float32) * d[:, None]


def dequant_q4_k(b):
    nb = b.shape[0]
    d, dmin = _f16_field(b, 0), _f16_field(b, 2)
    sc, mn = _k4_scales_mins(b[:, 4:16])
    q = _nibbles(b[:, 16:144].reshape(nb, 4, 32)).reshape(nb, 8, 32).astype(np.float32)
    q *= (d[:, None] * sc)[:, :, None]
    q -= (dmin[:, None] * mn)[:, :, None]
    return q.reshape(nb, 256)


def dequant_q5_k(b):
    nb = b.shape[0]
    d, dmin = _f16_field(b, 0), _f16_field(b, 2)
    sc, mn = _k4_scales_mins(b[:, 4:16])
    qh = b[:, 16:48]
    q = _nibbles(b[:, 48:176].reshape(nb, 4, 32)).reshape(nb, 8, 32)
    q |= ((qh[:, None, :] >> np.arange(8, dtype=np.uint8)[None, :, None]) & 1) << 4
    q = q.astype(np.float32)
    q *= (d[:, None] * sc)[:, :, None]
    q -= (dmin[:, None] * mn)[:, :, None]
    return q.reshape(nb, 256)


def dequant_q6_k(b):
    nb = b.shape[0]
    ql = b[:, 0:128].reshape(nb, 2, 64)
    qh = b[:, 128:192].reshape(nb, 2, 32)
    sc = b[:, 192:208].view(np.int8).astype(np.float32)
    d = _f16_field(b, 208)
    q = np.empty((nb, 2, 4, 32), dtype=np.uint8)
    q[:, :, 0] = (ql[:, :, 0:32] & 0x0F) | ((qh & 3) << 4)
    q[:, :, 1] = (ql[:, :, 32:64] & 0x0F) | (((qh >> 2) & 3) << 4)
    q[:, :, 2] = (ql[:, :, 0:32] >> 4) | (((qh >> 4) & 3) << 4)
    q[:, :, 3] = (ql[:, :, 32:64] >> 4) | (((qh >> 6) & 3) << 4)
    y = q.reshape(nb, 16, 16).astype(np.float32)
    y -= 32.0
    y *= (d[:, None] * sc)[:, :, None]
    return y.reshape(nb, 256)


def dequant_q3_k(b):
    nb = b.shape[0]
    hm = b[:, 0:32]
    qs = b[:, 32:96].reshape(nb, 2, 32)
    s = b[:, 96:108]
    d = _f16_field(b, 108)
    low = np.concatenate([s[:, 0:8] & 0x0F, s[:, 0:8] >> 4], axis=1)                    # [nb, 16]
    hs = s[:, 8:12]
    high = np.concatenate([hs & 3, (hs >> 2) & 3, (hs >> 4) & 3, (hs >> 6) & 3], axis=1)  # [nb, 16]
    sc = (low | (high << 4)).astype(np.float32) - 32.0
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8)[None, None, :, None]
    q = (qs[:, :, None, :] >> shifts) & 3                                                # [nb,2,4,32]
    hbit = (hm[:, None, None, :] >> np.arange(8, dtype=np.uint8).reshape(1, 2, 4, 1)) & 1
    y = (q | (hbit << 2)).astype(np.float32) - 4.0  # low2 - (hbit ? 0 : 4)
    y = y.reshape(nb, 16, 16)
    y *= (d[:, None] * sc)[:, :, None]
    return y.reshape(nb, 256)


def dequant_iq4_nl(b):
    nb = b.shape[0]
    d = _f16_field(b, 0)
    y = KVALUES_IQ4NL[_nibbles(b[:, 2:18])].reshape(nb, 32)
    y *= d[:, None]
    return y


def dequant_iq4_xs(b):
    nb = b.shape[0]
    d = _f16_field(b, 0)
    sh = np.ascontiguousarray(b[:, 2:4]).view("<u2")[:, 0].astype(np.int32)
    sl = b[:, 4:8].astype(np.int32)
    ib = np.arange(8)
    ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0x0F) | (((sh[:, None] >> (2 * ib)) & 3) << 4)
    dl = d[:, None] * (ls - 32).astype(np.float32)                                       # [nb, 8]
    y = KVALUES_IQ4NL[_nibbles(b[:, 8:136].reshape(nb, 8, 16))].reshape(nb, 8, 32)
    y *= dl[:, :, None]
    return y.reshape(nb, 256)


def dequant_iq3_s(b):
    grid = iq3s_grid()
    nb = b.shape[0]
    d = _f16_field(b, 0)
    qs = b[:, 2:66].reshape(nb, 8, 8).astype(np.int32)
    qh = b[:, 66:74].astype(np.int32)
    signs = b[:, 74:106].reshape(nb, 8, 4)
    scales = b[:, 106:110].astype(np.int32)
    ib = np.arange(8)
    db = d[:, None] * (1 + 2 * ((scales[:, ib // 2] >> (4 * (ib % 2))) & 0x0F)).astype(np.float32)
    idx = qs | (((qh[:, :, None] >> np.arange(8)[None, None, :]) & 1) << 8)            # [nb, 8, 8]
    g = grid[idx].reshape(nb, 8, 32).astype(np.float32)
    neg = np.unpackbits(signs, axis=2, bitorder="little").reshape(nb, 8, 32).astype(bool)
    y = db[:, :, None] * g
    np.negative(y, out=y, where=neg)
    return y.reshape(nb, 256)


DEQUANTIZERS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "F32": dequant_f32, "F16": dequant_f16, "BF16": dequant_bf16, "Q8_0": dequant_q8_0,
    "Q3_K": dequant_q3_k, "Q4_K": dequant_q4_k, "Q5_K": dequant_q5_k, "Q6_K": dequant_q6_k,
    "IQ4_NL": dequant_iq4_nl, "IQ4_XS": dequant_iq4_xs, "IQ3_S": dequant_iq3_s,
}


def dequantize_blocks(type_name: str, blocks: np.ndarray) -> np.ndarray:
    """uint8 [nb, type_size] -> float32 [nb * block_size]."""
    fn = DEQUANTIZERS.get(type_name)
    if fn is None:
        raise GGUFError(f"no dequantizer for ggml type {type_name}")
    return np.asarray(fn(blocks), dtype=np.float32).reshape(-1)


# ================================================================================================
# The container
# ================================================================================================


@dataclass(frozen=True)
class TensorInfo:
    name: str
    ne: tuple[int, ...]   # ggml order: ne[0] is the fastest-varying axis
    type_id: int
    offset: int           # absolute byte offset of the data in the file

    @property
    def gtype(self) -> GGMLType:
        t = GGML_TYPES.get(self.type_id)
        if t is None:
            raise GGUFError(f"tensor {self.name!r}: unknown ggml type id {self.type_id}")
        return t

    @property
    def type_name(self) -> str:
        return self.gtype.name

    @property
    def shape(self) -> tuple[int, ...]:
        """numpy row-major shape (ne reversed). A 2D weight [out, in] is stored ne = [in, out]."""
        return tuple(reversed(self.ne))

    @property
    def n_elements(self) -> int:
        return int(np.prod(self.ne, dtype=np.int64))

    @property
    def row_len(self) -> int:
        return int(self.ne[0])

    @property
    def n_rows(self) -> int:
        return int(np.prod(self.ne[1:], dtype=np.int64)) if len(self.ne) > 1 else 1

    @property
    def row_bytes(self) -> int:
        t = self.gtype
        if self.row_len % t.block_size:
            raise GGUFError(f"tensor {self.name!r}: ne[0]={self.row_len} is not a multiple of the "
                            f"{t.name} block size {t.block_size}")
        return self.row_len // t.block_size * t.type_size

    @property
    def nbytes(self) -> int:
        return self.row_bytes * self.n_rows


class _Cursor:
    __slots__ = ("buf", "pos", "end")

    def __init__(self, buf, pos: int = 0):
        self.buf, self.pos, self.end = buf, pos, len(buf)

    def take(self, n: int) -> int:
        p = self.pos
        if n < 0 or p + n > self.end:
            raise GGUFError(f"truncated GGUF header at byte {p} (wanted {n} more bytes)")
        self.pos = p + n
        return p

    def unpack(self, fmt: str):
        return struct.unpack_from(fmt, self.buf, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.unpack("<Q")
        p = self.take(n)
        return bytes(self.buf[p:p + n]).decode("utf-8", errors="replace")

    def value(self, vtype: int):
        if vtype == VT_STRING:
            return self.string()
        if vtype == VT_ARRAY:
            etype = self.unpack("<I")
            n = self.unpack("<Q")
            if etype in _VT_NUMPY:
                size = np.dtype(_VT_NUMPY[etype]).itemsize
                p = self.take(n * size)
                arr = np.frombuffer(self.buf, dtype=_VT_NUMPY[etype], count=n, offset=p).tolist()
                return [bool(x) for x in arr] if etype == VT_BOOL else arr
            return [self.value(etype) for _ in range(n)]
        fmt = _VT_STRUCT.get(vtype)
        if fmt is None:
            raise GGUFError(f"unknown GGUF metadata value type {vtype} at byte {self.pos}")
        return self.unpack(fmt)


class GGUFFile:
    """A GGUF v2/v3 file, memory-mapped read-only. `tensors` keeps the file's order."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fh = open(self.path, "rb")
        try:
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
        except Exception:
            self._fh.close()
            raise
        self.file_size = len(self._mm)
        try:
            self._parse()
        except Exception:
            self.close()
            raise

    # -- header ----------------------------------------------------------------------------------

    def _parse(self) -> None:
        c = _Cursor(self._mm)
        magic = c.unpack("<I")
        if magic != GGUF_MAGIC:
            raise GGUFError(f"{self.path}: bad magic 0x{magic:08x} (not a GGUF file)")
        self.version = c.unpack("<I")
        if self.version not in GGUF_VERSIONS:
            raise GGUFError(f"{self.path}: unsupported GGUF version {self.version}")
        n_tensors = c.unpack("<Q")
        n_kv = c.unpack("<Q")
        self.metadata: dict[str, Any] = {}
        self.metadata_types: dict[str, int] = {}
        for _ in range(n_kv):
            key = c.string()
            vtype = c.unpack("<I")
            self.metadata[key] = c.value(vtype)
            self.metadata_types[key] = vtype
        infos = []
        for _ in range(n_tensors):
            name = c.string()
            n_dims = c.unpack("<I")
            ne = tuple(int(c.unpack("<Q")) for _ in range(n_dims))
            type_id = c.unpack("<I")
            rel = c.unpack("<Q")
            infos.append((name, ne, type_id, rel))
        self.alignment = int(self.metadata.get("general.alignment", DEFAULT_ALIGNMENT))
        if self.alignment <= 0 or self.alignment & (self.alignment - 1):
            raise GGUFError(f"{self.path}: general.alignment={self.alignment} is not a power of two")
        self.header_end = c.pos
        self.data_start = -(-c.pos // self.alignment) * self.alignment
        self.tensors: dict[str, TensorInfo] = {}
        for name, ne, type_id, rel in infos:
            if name in self.tensors:
                raise GGUFError(f"{self.path}: duplicate tensor name {name!r}")
            if rel % self.alignment:
                raise GGUFError(f"{self.path}: tensor {name!r} offset {rel} is not "
                                f"{self.alignment}-aligned")
            self.tensors[name] = TensorInfo(name, ne, type_id, self.data_start + rel)
        # Every tensor must lie inside the file, and no two may overlap.
        spans = sorted((t.offset, t.offset + t.nbytes, t.name) for t in self.tensors.values()
                       if t.type_id in GGML_TYPES)
        prev_end, prev_name = self.data_start, None
        for start, end, name in spans:
            if start < prev_end:
                raise GGUFError(f"{self.path}: tensor {name!r} overlaps {prev_name!r}")
            if end > self.file_size:
                raise GGUFError(f"{self.path}: tensor {name!r} ends at byte {end}, past the end of "
                                f"the {self.file_size}-byte file")
            prev_end, prev_name = end, name

    @property
    def header_sha256(self) -> str:
        """SHA-256 of bytes [0, data_start): the metadata and the tensor table, not the weights."""
        return hashlib.sha256(self._mm[:self.data_start]).hexdigest()

    def sha256(self, chunk: int = 1 << 24) -> str:
        """SHA-256 of the whole file (reads all of it)."""
        h = hashlib.sha256()
        for p in range(0, self.file_size, chunk):
            h.update(self._mm[p:p + chunk])
        return h.hexdigest()

    def close(self) -> None:
        """Unmap and close. If numpy views of the map are still alive the map cannot be closed
        (BufferError); it is then dropped and freed with the last view."""
        try:
            self._mm.close()
        except BufferError:
            pass
        finally:
            self._fh.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def meta(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.metadata:
            raise GGUFError(f"{self.path}: metadata key {key!r} is missing")
        return self.metadata[key]

    def tensor(self, name: str) -> TensorInfo:
        t = self.tensors.get(name)
        if t is None:
            raise GGUFError(f"{self.path}: no tensor named {name!r}")
        return t

    def type_counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.tensors.values():
            out[t.type_name] = out.get(t.type_name, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: -kv[1]))

    def check_decodable(self, names=None) -> None:
        """Raise now, not halfway through a run, if any of `names` (default: all) cannot be decoded."""
        needed = {self.tensor(n).type_name for n in (names if names is not None else self.tensors)}
        missing = sorted(t for t in needed if t not in DEQUANTIZERS)
        if missing:
            raise GGUFError(f"no dequantizer for ggml type(s) {missing}")
        if "IQ3_S" in needed:
            iq3s_grid()

    # -- data ------------------------------------------------------------------------------------

    def raw_rows(self, name: str, r0: int = 0, r1: int | None = None) -> np.ndarray:
        """uint8 view [r1 - r0, row_bytes] of rows [r0, r1) of the tensor seen as [n_rows, ne[0]]."""
        t = self.tensor(name)
        r1 = t.n_rows if r1 is None else r1
        if not 0 <= r0 <= r1 <= t.n_rows:
            raise GGUFError(f"{name}: row range [{r0}, {r1}) outside [0, {t.n_rows})")
        rb = t.row_bytes
        return np.frombuffer(self._mm, dtype=np.uint8, count=(r1 - r0) * rb,
                             offset=t.offset + r0 * rb).reshape(r1 - r0, rb)

    def dequantize_rows(self, name: str, r0: int = 0, r1: int | None = None,
                        threads: int = 1, out: np.ndarray | None = None) -> np.ndarray:
        """float32 [r1 - r0, ne[0]]. `threads` > 1 splits the rows over a thread pool (numpy
        releases the GIL inside its kernels)."""
        t = self.tensor(name)
        r1 = t.n_rows if r1 is None else r1
        raw = self.raw_rows(name, r0, r1)
        n, ts = raw.shape[0], t.gtype.type_size
        if out is None:
            out = np.empty((n, t.row_len), dtype=np.float32)
        elif out.shape != (n, t.row_len) or out.dtype != np.float32:
            raise GGUFError(f"{name}: out has shape {out.shape} {out.dtype}, need {(n, t.row_len)} float32")
        step = max(1, (1 << 21) // max(1, t.row_len))  # ~2M elements per piece

        def work(a: int) -> None:
            b = min(n, a + step)
            blocks = raw[a:b].reshape(-1, ts)
            out[a:b] = dequantize_blocks(t.type_name, blocks).reshape(b - a, t.row_len)

        starts = range(0, n, step)
        if threads > 1 and n > step:
            with ThreadPoolExecutor(max_workers=threads) as ex:
                list(ex.map(work, starts))
        else:
            for a in starts:
                work(a)
        return out

    def dequantize_row_ids(self, name: str, row_ids) -> np.ndarray:
        """float32 [len(row_ids), ne[0]] for arbitrary row indices (an embedding gather)."""
        t = self.tensor(name)
        ids = np.asarray(row_ids, dtype=np.int64)
        if ids.size and (ids.min() < 0 or ids.max() >= t.n_rows):
            raise GGUFError(f"{name}: row id out of range [0, {t.n_rows})")
        raw = self.raw_rows(name)[ids]
        return dequantize_blocks(t.type_name, raw.reshape(-1, t.gtype.type_size)).reshape(ids.size, t.row_len)

    def dequantize(self, name: str, threads: int = 1) -> np.ndarray:
        """The whole tensor, float32, numpy shape (ne reversed), GGUF layout."""
        return self.dequantize_rows(name, threads=threads).reshape(self.tensor(name).shape)

    def row_scales(self, name: str, row: int) -> np.ndarray | None:
        """The f16 block scale d of every block of GGUF row `row`, as float32; None for a type
        without one (F32/F16/BF16)."""
        t = self.tensor(name)
        off = SCALE_D_OFFSET.get(t.type_name)
        if off is None:
            return None
        return _f16_field(self.raw_rows(name, row, row + 1).reshape(-1, t.gtype.type_size), off)


# ================================================================================================
# llama.cpp qwen35 <-> HF Qwen3_5ForConditionalGeneration
# ================================================================================================

#: GGUF per-block suffix -> (HF suffix under the layer prefix, transform tag). Transforms, established
#: tensor by tensor against the bf16 checkpoint (F32 tensors bit-exact; see gguf_validate.py):
#:   id              same values, same layout (ggml's ne order is only the reversed numpy shape)
#:   plus1           zero-centred RMSNorm weight stored as float32(w) + 1
#:   qkv_vrows       in_proj_qkv: Q and K rows as is, V rows in tiled head order (below)
#:   vrows           every row block of head_v_dim rows is one V head, in tiled order
#:   vheads          one entry (or row) per V head, tiled order
#:   neg_exp_vheads  A_log stored as -exp(A_log) in float32, tiled order
#:   conv_vch        conv1d [C, 1, K] squeezed to [C, K]; the V channels in tiled order
#:   vcols           out_proj's input columns in tiled V-head order
#: Tiled order: HF groups V heads per K head (k-head j owns v-heads r*j .. r*j + r-1, r = nv/nk,
#: torch repeat_interleave); llama.cpp broadcasts with ggml_repeat (tiling), so GGUF V head h holds
#: HF V head (h % nk) * r + h // nk.
_BLOCK_MAP: dict[str, tuple[str, str]] = {
    "attn_norm.weight": ("input_layernorm.weight", "plus1"),
    "post_attention_norm.weight": ("post_attention_layernorm.weight", "plus1"),
    "attn_q.weight": ("self_attn.q_proj.weight", "id"),
    "attn_k.weight": ("self_attn.k_proj.weight", "id"),
    "attn_v.weight": ("self_attn.v_proj.weight", "id"),
    "attn_output.weight": ("self_attn.o_proj.weight", "id"),
    "attn_q_norm.weight": ("self_attn.q_norm.weight", "plus1"),
    "attn_k_norm.weight": ("self_attn.k_norm.weight", "plus1"),
    "attn_qkv.weight": ("linear_attn.in_proj_qkv.weight", "qkv_vrows"),
    "attn_gate.weight": ("linear_attn.in_proj_z.weight", "vrows"),
    "ssm_alpha.weight": ("linear_attn.in_proj_a.weight", "vheads"),
    "ssm_beta.weight": ("linear_attn.in_proj_b.weight", "vheads"),
    "ssm_a": ("linear_attn.A_log", "neg_exp_vheads"),
    "ssm_dt.bias": ("linear_attn.dt_bias", "vheads"),
    "ssm_conv1d.weight": ("linear_attn.conv1d.weight", "conv_vch"),
    "ssm_norm.weight": ("linear_attn.norm.weight", "id"),
    "ssm_out.weight": ("linear_attn.out_proj.weight", "vcols"),
    "ffn_gate.weight": ("mlp.gate_proj.weight", "id"),
    "ffn_up.weight": ("mlp.up_proj.weight", "id"),
    "ffn_down.weight": ("mlp.down_proj.weight", "id"),
}
_NEXTN_MAP: dict[str, tuple[str, str]] = {
    "nextn.eh_proj.weight": ("fc.weight", "id"),
    "nextn.enorm.weight": ("pre_fc_norm_embedding.weight", "plus1"),
    "nextn.hnorm.weight": ("pre_fc_norm_hidden.weight", "plus1"),
    "nextn.shared_head_norm.weight": ("norm.weight", "plus1"),
}
_TOP_MAP: dict[str, tuple[str, str]] = {
    "token_embd.weight": ("model.language_model.embed_tokens.weight", "id"),
    "output.weight": ("lm_head.weight", "id"),
    "output_norm.weight": ("model.language_model.norm.weight", "plus1"),
}
HF_TEXT_LAYER_PREFIX = "model.language_model.layers."


@dataclass
class Qwen35Map:
    """Name map and layout transforms for one qwen35 GGUF, with the head geometry read from its
    metadata."""

    n_trunk: int      # decoder layers (block_count - nextn_predict_layers)
    n_nextn: int
    nk: int           # linear-attention K heads (ssm.group_count)
    nv: int           # linear-attention V heads (ssm.time_step_rank)
    dk: int           # K head dim (ssm.state_size)
    dv: int           # V head dim (ssm.inner_size / nv)
    hf_of: dict[str, tuple[str, str]] = field(default_factory=dict)    # gguf -> (hf, tag)
    gguf_of: dict[str, tuple[str, str]] = field(default_factory=dict)  # hf -> (gguf, tag)
    unmapped: list[str] = field(default_factory=list)

    @classmethod
    def from_gguf(cls, g: GGUFFile) -> "Qwen35Map":
        arch = g.meta("general.architecture")
        if arch != "qwen35":
            raise GGUFError(f"{g.path}: general.architecture={arch!r}; this map is for 'qwen35'")
        blocks = int(g.require("qwen35.block_count"))
        nextn = int(g.meta("qwen35.nextn_predict_layers", 0))
        nk = int(g.require("qwen35.ssm.group_count"))
        nv = int(g.require("qwen35.ssm.time_step_rank"))
        dk = int(g.require("qwen35.ssm.state_size"))
        inner = int(g.require("qwen35.ssm.inner_size"))
        if nv % nk or inner % nv:
            raise GGUFError(f"inconsistent ssm geometry: nk={nk} nv={nv} inner={inner}")
        m = cls(n_trunk=blocks - nextn, n_nextn=nextn, nk=nk, nv=nv, dk=dk, dv=inner // nv)
        for name in g.tensors:
            hit = m.map_name(name)
            if hit is None:
                m.unmapped.append(name)
                continue
            if hit[0] in m.gguf_of:
                raise GGUFError(f"{name} and {m.gguf_of[hit[0]][0]} both map to {hit[0]}")
            m.hf_of[name] = hit
            m.gguf_of[hit[0]] = (name, hit[1])
        return m

    def map_name(self, gname: str) -> tuple[str, str] | None:
        if gname in _TOP_MAP:
            return _TOP_MAP[gname]
        mt = re.fullmatch(r"blk\.(\d+)\.(.+)", gname)
        if not mt:
            return None
        il, suffix = int(mt.group(1)), mt.group(2)
        if il < self.n_trunk:
            prefix = f"{HF_TEXT_LAYER_PREFIX}{il}."
        elif il < self.n_trunk + self.n_nextn:
            prefix = f"mtp.layers.{il - self.n_trunk}."
            if suffix in _NEXTN_MAP:
                hf, tag = _NEXTN_MAP[suffix]
                return f"mtp.{hf}", tag
        else:
            return None
        if suffix in _BLOCK_MAP:
            hf, tag = _BLOCK_MAP[suffix]
            return prefix + hf, tag
        return None

    # -- the V-head permutation --------------------------------------------------------------------

    @property
    def tiled(self) -> np.ndarray:
        """GGUF V head h -> HF V head."""
        r = self.nv // self.nk
        return np.array([(h % self.nk) * r + h // self.nk for h in range(self.nv)], dtype=np.int64)

    @property
    def untiled(self) -> np.ndarray:
        """HF V head h -> GGUF V head (the inverse permutation)."""
        return np.argsort(self.tiled)

    def _heads(self, x: np.ndarray, perm: np.ndarray, head_rows: int) -> np.ndarray:
        """Permute leading-axis blocks of `head_rows` rows: out block h = x block perm[h]."""
        s = x.shape
        if s[0] != self.nv * head_rows:
            raise GGUFError(f"expected {self.nv} x {head_rows} rows, got {s[0]}")
        return x.reshape(self.nv, head_rows, *s[1:])[perm].reshape(s)

    def _v_part(self, x: np.ndarray, perm: np.ndarray) -> np.ndarray:
        """Rows [2*nk*dk:] (the V part of in_proj_qkv / conv1d) head-permuted, the rest kept."""
        kd = 2 * self.nk * self.dk
        out = np.array(x, dtype=np.float32, copy=True)
        out[kd:] = self._heads(x[kd:], perm, self.dv)
        return out

    def to_hf(self, tag: str, g: np.ndarray, hf_shape) -> np.ndarray:
        """Dequantized GGUF tensor (float32, GGUF numpy layout) -> HF layout and values (float32)."""
        hf_shape = tuple(int(s) for s in hf_shape)
        if g.size != int(np.prod(hf_shape, dtype=np.int64)):
            raise GGUFError(f"{tag}: {g.shape} ({g.size} elements) cannot become HF {hf_shape}")
        g = np.asarray(g, dtype=np.float32)
        u = self.untiled
        if tag == "id":
            out = g
        elif tag == "plus1":
            out = g - np.float32(1.0)
        elif tag == "qkv_vrows":
            out = self._v_part(g, u)
        elif tag == "vrows":
            out = self._heads(g, u, self.dv)
        elif tag == "vheads":
            out = self._heads(g, u, 1)
        elif tag == "neg_exp_vheads":
            out = np.log(-self._heads(g, u, 1).astype(np.float64)).astype(np.float32)
        elif tag == "conv_vch":
            out = self._v_part(g.reshape(g.shape[0], -1), u)
        elif tag == "vcols":
            out = np.ascontiguousarray(self._heads(np.ascontiguousarray(g.T), u, self.dv).T)
        else:
            raise GGUFError(f"unknown transform tag {tag!r}")
        return out.reshape(hf_shape)

    def from_hf(self, tag: str, w: np.ndarray, g_shape) -> np.ndarray:
        """HF tensor (float32 values) -> the float32 values the converter stores in the GGUF. Used to
        check lossless (F32) tensors bit for bit, which is stricter than inverting them."""
        g_shape = tuple(int(s) for s in g_shape)
        w = np.asarray(w, dtype=np.float32)
        t = self.tiled
        if tag == "id":
            out = w
        elif tag == "plus1":
            out = w + np.float32(1.0)
        elif tag == "qkv_vrows":
            out = self._v_part(w, t)
        elif tag == "vrows":
            out = self._heads(w, t, self.dv)
        elif tag == "vheads":
            out = self._heads(w, t, 1)
        elif tag == "neg_exp_vheads":
            out = (-np.exp(self._heads(w, t, 1).astype(np.float64))).astype(np.float32)
        elif tag == "conv_vch":
            out = self._v_part(w.reshape(w.shape[0], -1), t)
        elif tag == "vcols":
            out = np.ascontiguousarray(self._heads(np.ascontiguousarray(w.T), t, self.dv).T)
        else:
            raise GGUFError(f"unknown transform tag {tag!r}")
        return out.reshape(g_shape)

    def gguf_row(self, tag: str, hf_row: int) -> int:
        """The GGUF row (numpy layout: an output row of a 2D weight) that holds row `hf_row` of the
        HF tensor after `to_hf`. Only the V-head row reorders move rows; `vcols` moves columns."""
        r, kd = int(hf_row), 2 * self.nk * self.dk
        if tag == "qkv_vrows" and r >= kd:
            q = r - kd
            return kd + int(self.untiled[q // self.dv]) * self.dv + q % self.dv
        if tag == "vrows":
            return int(self.untiled[r // self.dv]) * self.dv + r % self.dv
        if tag == "vheads":
            return int(self.untiled[r])
        return r

    def hf_tensor(self, g: GGUFFile, hf_name: str, hf_shape, threads: int = 1) -> np.ndarray:
        """The GGUF's value of HF tensor `hf_name`: dequantized, inverse-transformed, float32."""
        gname, tag = self.gguf_of[hf_name]
        return self.to_hf(tag, g.dequantize(gname, threads=threads), hf_shape)


def f32_ulp_distance(a: np.ndarray, b: np.ndarray) -> int:
    """Max distance in float32 units in the last place (same-sign finite values)."""
    ia = np.ascontiguousarray(a, dtype=np.float32).view(np.int32).astype(np.int64)
    ib = np.ascontiguousarray(b, dtype=np.float32).view(np.int32).astype(np.int64)
    return int(np.abs(ia - ib).max()) if ia.size else 0


#: Lossless tensors must equal `from_hf(checkpoint value)` exactly, except -exp(A_log): numpy's
#: float64 exp rounded to float32 may sit 1 ulp from the converter's float32 exp.
LOSSLESS_ULP_TOLERANCE = {"neg_exp_vheads": 1}


def lossless_matches(qmap: Qwen35Map, tag: str, gguf_value: np.ndarray, hf_value: np.ndarray) -> tuple[bool, int]:
    """(ok, ulp distance) for a lossless GGUF tensor against its checkpoint tensor."""
    expect = qmap.from_hf(tag, hf_value, gguf_value.shape)
    ulp = f32_ulp_distance(expect, gguf_value)
    return ulp <= LOSSLESS_ULP_TOLERANCE.get(tag, 0), ulp


def row_rel_errors(row_sse, row_ssw) -> np.ndarray:
    """Per-row ||deq_r - W_r|| / ||W_r|| from per-row sums of squares (0 for an exact all-zero row,
    inf for a nonzero error on an all-zero row)."""
    sse = np.asarray(row_sse, dtype=np.float64)
    ssw = np.asarray(row_ssw, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rel = np.sqrt(sse / ssw)
    rel[ssw == 0] = np.where(sse[ssw == 0] == 0, 0.0, np.inf)
    return rel


def row_outliers(g: GGUFFile, qmap: Qwen35Map, gname: str, tag: str, row_rel: np.ndarray,
                 row_norm: np.ndarray, rows=None, row_cos: np.ndarray | None = None) -> list[dict]:
    """Rows whose rel error exceeds `row_error_bound` for the tensor's type, worst first.
    `row_rel` / `row_norm` / `row_cos` are per HF row (||deq_r - W_r|| / ||W_r||, ||W_r||, and the
    cosine of deq_r with W_r); `rows` gives their HF row indices when they are not 0..n-1 (an
    embedding gather). Each outlier carries its GGUF row, its blocks' f16 scales (how many are 0,
    how many coarse, the largest |d|), and `coarse_scale`: every block's d is below
    F16_COARSE_SCALE, so the row is too small for the type's scale format."""
    bound = row_error_bound(g.tensor(gname).type_name)
    if bound is None:
        return []
    row_rel = np.asarray(row_rel, dtype=np.float64)
    ids = np.arange(row_rel.size) if rows is None else np.asarray(rows, dtype=np.int64)
    bad = np.nonzero(row_rel > bound)[0]
    out = []
    for i in bad[np.argsort(-row_rel[bad], kind="stable")]:
        grow = qmap.gguf_row(tag, int(ids[i]))
        d = g.row_scales(gname, grow)
        o = {"row": int(ids[i]), "gguf_row": grow, "rel": float(row_rel[i]), "norm": float(row_norm[i])}
        if row_cos is not None:
            o["cos"] = float(row_cos[i])
        if d is None:
            o.update(blocks=None, zero_scale_blocks=None, coarse_scale_blocks=None, max_abs_scale=None,
                     coarse_scale=False)
        else:
            ad = np.abs(d)
            o.update(blocks=int(d.size), zero_scale_blocks=int((ad == 0).sum()),
                     coarse_scale_blocks=int((ad < F16_COARSE_SCALE).sum()),
                     max_abs_scale=float(ad.max()), coarse_scale=bool((ad < F16_COARSE_SCALE).all()))
        out.append(o)
    return out


# ================================================================================================
# ggml's own C dequantizers, as an oracle (optional)
# ================================================================================================

#: ggml type -> its C `void dequantize_row_<t>(const block_<t> * x, float * y, int64_t k)`.
GGML_C_DEQUANT = {"Q8_0": "dequantize_row_q8_0", "Q3_K": "dequantize_row_q3_K",
                  "Q4_K": "dequantize_row_q4_K", "Q5_K": "dequantize_row_q5_K",
                  "Q6_K": "dequantize_row_q6_K", "IQ4_NL": "dequantize_row_iq4_nl",
                  "IQ4_XS": "dequantize_row_iq4_xs", "IQ3_S": "dequantize_row_iq3_s"}


def ggml_base_candidates() -> list[Path]:
    env = os.environ.get("R4DX_GGML_BASE_DLL")
    names = ("ggml-base.dll",) if sys.platform == "win32" else ("libggml-base.so", "libggml-base.dylib")
    root = Path.home() / "dev" / "ROCmFPX"
    return ([Path(env)] if env else []) + [root / b / sub / n for b in ("build-hip", "build")
                                          for sub in ("bin", "lib") for n in names]


class GgmlC:
    """ggml's reference C dequantizers (`ggml-quants.c`, exported by the ggml-base library of a local
    llama.cpp build: CPU code only, no GPU) called through ctypes. The numpy decoders above must
    match them bit for bit; tests/reference/test_gguf_dequant.py and gguf_validate.py check that
    when a build is on disk ($env:R4DX_GGML_BASE_DLL, else ~/dev/ROCmFPX/build-hip/bin)."""

    def __init__(self, path: str | Path):
        import ctypes

        self.path = Path(path)
        if sys.platform == "win32":  # its own dependencies (the CRT) resolve next to it
            self._dll_dir = os.add_dll_directory(str(self.path.parent))
        self._lib = ctypes.CDLL(str(self.path))
        self._fns = {}
        for tname, sym in GGML_C_DEQUANT.items():
            fn = getattr(self._lib, sym, None)
            if fn is not None:
                fn.restype = None
                fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
                self._fns[tname] = fn

    @property
    def types(self) -> list[str]:
        return sorted(self._fns)

    def dequantize(self, type_name: str, blocks: np.ndarray) -> np.ndarray:
        """uint8 [nb, type_size] (or [rows, row_bytes]) -> float32 [nb * block_size], by ggml's C."""
        t = next(v for v in GGML_TYPES.values() if v.name == type_name)
        raw = np.ascontiguousarray(blocks, dtype=np.uint8)
        if raw.size % t.type_size:
            raise GGUFError(f"{raw.size} bytes is not a whole number of {type_name} blocks")
        out = np.empty(raw.size // t.type_size * t.block_size, dtype=np.float32)
        self._fns[type_name](raw.ctypes.data, out.ctypes.data, out.size)
        return out


def load_ggml_c() -> GgmlC | None:
    """The first ggml-base library of `ggml_base_candidates` that loads, or None."""
    for p in ggml_base_candidates():
        if p.is_file():
            try:
                return GgmlC(p)
            except OSError:
                continue
    return None


def tensor_class(hf_name: str) -> str:
    """A short role label for the tables: 'mlp.gate_proj', 'gdn.in_proj_qkv', 'attn.q_proj', ..."""
    if hf_name.endswith("embed_tokens.weight"):
        return "embed_tokens"
    if hf_name == "lm_head.weight":
        return "lm_head"
    if hf_name == "model.language_model.norm.weight":
        return "final_norm"
    s = re.sub(r"^(model\.language_model\.layers\.\d+|mtp\.layers\.\d+|mtp)\.", "", hf_name)
    s = re.sub(r"\.weight$", "", s).replace("linear_attn.", "gdn.").replace("self_attn.", "attn.")
    return ("mtp." if hf_name.startswith("mtp.") else "") + s


# ================================================================================================


def _main(argv=None) -> int:  # pragma: no cover - inspection helper
    import argparse

    ap = argparse.ArgumentParser(description="Summarize a GGUF: header, tensor types, qwen35 map.")
    ap.add_argument("gguf", type=Path)
    ap.add_argument("--tensors", action="store_true", help="list every tensor")
    args = ap.parse_args(argv)
    with GGUFFile(args.gguf) as g:
        print(f"{g.path}: GGUF v{g.version}, {len(g.metadata)} metadata keys, {len(g.tensors)} tensors, "
              f"data at {g.data_start} (alignment {g.alignment}), {g.file_size} bytes")
        for k, v in g.metadata.items():
            if isinstance(v, list) and len(v) > 8:
                v = f"[{len(v)} items]"
            elif isinstance(v, str) and len(v) > 100:
                v = v[:100] + "..."
            print(f"  {k} = {v}")
        print("types:", g.type_counts())
        if g.meta("general.architecture") == "qwen35":
            m = Qwen35Map.from_gguf(g)
            print(f"qwen35 map: {len(m.hf_of)} mapped, {len(m.unmapped)} unmapped {m.unmapped[:10]}; "
                  f"trunk={m.n_trunk} nextn={m.n_nextn} nk={m.nk} nv={m.nv} dk={m.dk} dv={m.dv}")
        if args.tensors:
            for t in g.tensors.values():
                print(f"  {t.name:48s} {t.type_name:7s} {t.shape}")
    return 0


if __name__ == "__main__":
    sys.exit(_main())
