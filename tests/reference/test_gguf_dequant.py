"""tests/reference/test_gguf_dequant.py

CPU-only tests of tools/reference/gguf_dequant.py (GGUF reader, dequantizers, qwen35 map) and
gguf_validate.py. numpy only: no torch, no GPU.

    (a) hand-built blocks for every implemented type: an encoder written from the ggml block layouts
        (the packing in ggml-quants.c's quantize_row_*_ref) packs known integer quants and scales,
        and the decoder must return exactly the values those integers define. d is a small odd
        integer times a power of two, so every expected value is exact in float32 and the
        comparison is bit for bit.
    (b) if a ROCmFPX checkout's gguf-py is on disk (loaded from source, nothing installed): the same
        hand-built blocks and random-byte blocks decode bit-identically in gguf-py, and the ggml
        type ids agree. SKIPped otherwise.
    (c) a tiny synthetic GGUF v3 written here: every metadata value type (incl. arrays and a
        nested array), a non-default general.alignment, tensors of several types; header parsing,
        offsets, shapes (ne reversed) and dequantized values; refusals for bad magic, truncation,
        misaligned and overlapping tensors, and a tensor past EOF; the per-row scale diagnostics
        (row_scales / row_outliers: a zero and a coarse subnormal d).
    (d) Qwen35Map on a small geometry: the tiled V-head order, from_hf / to_hf round trips for every
        transform tag, gguf_row against to_hf, name mapping (trunk, MTP, top level),
        lossless_matches with a 1-ulp change.
    (e) the real Unsloth GGUF and the bf16 checkpoint, if both are on disk (SKIP otherwise): header
        facts, the full name map, gguf_validate.check_tensor on a handful of tensors (one per
        transform tag, plus IQ3_S, Q3_K, IQ4_NL, IQ4_XS), and token_embd row 107517 (the file's one
        catastrophic embedding row: every f16 scale 0) found by row_outliers.
    (f) if a local llama.cpp build's ggml-base library is on disk (gguf_dequant.GgmlC; SKIPped
        otherwise): ggml's own C `dequantize_row_*` on the hand-built and random blocks of (a) and
        on gathered and contiguous rows of a real tensor of every quantized type, token_embd
        (incl. row 107517) and output, all bit-identical; one flipped quant bit moves exactly one
        value. This is the evidence that does not depend on bounds fitted to the file.

Plain script, no pytest (like test_manifest.py):

    python tests\\reference\\test_gguf_dequant.py

Exits 0 and prints "OK (<n> checks)" on success; 1 and every failed check otherwise.
"""

from __future__ import annotations

import os
import struct
import sys
import tempfile
import traceback
import zlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "reference"))

try:
    import numpy as np
except ImportError as e:
    print(f"SKIP: {e} -- run this with the reference venv's python.exe (see docstring)")
    raise SystemExit(0)

import gguf_dequant as gd  # noqa: E402

REAL_GGUF = Path(os.environ.get(
    "R4DX_TEST_GGUF",
    r"D:\huggingface\hub\models--unsloth--Qwen3.8-27B-GGUF\snapshots"
    r"\4ca720788d1e01f1bff70c033e0d0028fd02e502\Qwen3.8-27B-UD-Q4_K_XL.gguf"))
REAL_MODEL_DIR = Path(os.environ.get("R4DX_TEST_MODEL_DIR", r"C:\AI\models\Qwen3.8-27B"))
GGUF_PY = Path(os.environ.get("R4DX_GGUF_PY", str(Path.home() / "dev" / "ROCmFPX" / "gguf-py")))

CHECKS = 0
FAILURES: list[str] = []


def check(cond, label: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILURES.append(label)
        print(f"FAIL {label}")


def raises(exc_type, needle: str, fn, label: str) -> None:
    try:
        fn()
    except exc_type as e:
        check(needle in str(e), f"{label} (message {str(e)[:160]!r} lacks {needle!r})")
        return
    check(False, f"{label} (did not raise {exc_type.__name__})")


def f16b(x) -> bytes:
    v = np.float16(x)
    assert float(v) == float(x), f"{x} is not exact in f16"
    return v.tobytes()


def exact_scale(rng) -> float:
    """A nonzero d exact in f16 whose products with the small ints below stay exact in float32."""
    return float(rng.choice([-7, -5, -3, -1, 1, 3, 5, 7]) * 2.0 ** int(rng.integers(-12, -3)))


# ================================================================================================
# (a) hand-built blocks: encoders written from the ggml layouts
# ================================================================================================


def enc_q8_0(rng):
    d = exact_scale(rng)
    q = rng.integers(-128, 128, 32)
    return f16b(d) + q.astype(np.int8).tobytes(), d * q


def _k4_scale_bytes(sc, m) -> bytearray:
    s = bytearray(12)
    for j in range(8):
        if j < 4:
            s[j] = sc[j]
            s[j + 4] = m[j]
        else:
            s[j + 4] = (sc[j] & 0xF) | ((m[j] & 0xF) << 4)
            s[j - 4] |= (sc[j] >> 4) << 6
            s[j] |= (m[j] >> 4) << 6
    return s


def enc_q4_k(rng):
    d, dmin = exact_scale(rng), exact_scale(rng)
    sc, m = rng.integers(0, 64, 8), rng.integers(0, 64, 8)
    q = rng.integers(0, 16, 256)
    qs = bytearray(128)
    for j in range(0, 256, 64):
        for l in range(32):
            qs[j // 2 + l] = q[j + l] | (q[j + l + 32] << 4)
    expect = np.array([d * sc[e // 32] * q[e] - dmin * m[e // 32] for e in range(256)])
    return f16b(d) + f16b(dmin) + bytes(_k4_scale_bytes(sc, m)) + bytes(qs), expect


def enc_q5_k(rng):
    d, dmin = exact_scale(rng), exact_scale(rng)
    sc, m = rng.integers(0, 64, 8), rng.integers(0, 64, 8)
    q = rng.integers(0, 32, 256)
    qh, qs = bytearray(32), bytearray(128)
    for n in range(0, 256, 64):
        for j in range(32):
            l1, l2 = q[n + j], q[n + j + 32]
            s = n // 32  # sub-block of l1; l2 is s + 1
            if l1 > 15:
                l1 -= 16
                qh[j] |= 1 << s
            if l2 > 15:
                l2 -= 16
                qh[j] |= 1 << (s + 1)
            qs[n // 2 + j] = l1 | (l2 << 4)
    expect = np.array([d * sc[e // 32] * q[e] - dmin * m[e // 32] for e in range(256)])
    return f16b(d) + f16b(dmin) + bytes(_k4_scale_bytes(sc, m)) + bytes(qh) + bytes(qs), expect


def enc_q6_k(rng):
    d = exact_scale(rng)
    sc = rng.integers(-128, 128, 16)
    L = rng.integers(0, 64, 256)
    ql, qh = bytearray(128), bytearray(64)
    for j in range(0, 256, 128):
        for l in range(32):
            q1, q2, q3, q4 = (L[j + l] & 0xF, L[j + l + 32] & 0xF, L[j + l + 64] & 0xF, L[j + l + 96] & 0xF)
            ql[j // 2 + l] = q1 | (q3 << 4)
            ql[j // 2 + l + 32] = q2 | (q4 << 4)
            qh[j // 4 + l] = ((L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2)
                              | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6))
    expect = np.array([d * sc[e // 16] * (L[e] - 32) for e in range(256)])
    return bytes(ql) + bytes(qh) + sc.astype(np.int8).tobytes() + f16b(d), expect


def enc_q3_k(rng):
    d = exact_scale(rng)
    ls = rng.integers(0, 64, 16)  # stored scale, value ls - 32
    L = rng.integers(0, 8, 256)   # stored quant, value L - 4
    scales = bytearray(12)
    for j in range(16):
        lj = int(ls[j])
        if j < 8:
            scales[j] = lj & 0xF
        else:
            scales[j - 8] |= (lj & 0xF) << 4
        scales[j % 4 + 8] |= (lj >> 4) << (2 * (j // 4))
    hmask = bytearray(32)
    Lw = L.copy()
    m, hm = 0, 1
    for j in range(256):
        if Lw[j] > 3:
            hmask[m] |= hm
            Lw[j] -= 4
        m += 1
        if m == 32:
            m, hm = 0, hm << 1
    qs = bytearray(64)
    for j in range(0, 256, 128):
        for l in range(32):
            qs[j // 4 + l] = Lw[j + l] | (Lw[j + l + 32] << 2) | (Lw[j + l + 64] << 4) | (Lw[j + l + 96] << 6)
    expect = np.array([d * (ls[e // 16] - 32) * (L[e] - 4) for e in range(256)])
    return bytes(hmask) + bytes(qs) + bytes(scales) + f16b(d), expect


KV = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113]


def enc_iq4_nl(rng):
    d = exact_scale(rng)
    idx = rng.integers(0, 16, 32)
    qs = bytes(int(idx[j] | (idx[j + 16] << 4)) for j in range(16))
    return f16b(d) + qs, np.array([d * KV[i] for i in idx])


def enc_iq4_xs(rng):
    d = exact_scale(rng)
    ls = rng.integers(0, 64, 8)
    idx = rng.integers(0, 16, 256)
    sh, sl = 0, bytearray(4)
    for ib in range(8):
        lo, hi = int(ls[ib]) & 0xF, int(ls[ib]) >> 4
        sl[ib // 2] |= lo << (4 * (ib % 2))
        sh |= hi << (2 * ib)
    qs = bytearray(128)
    for ib in range(8):
        for j in range(16):
            qs[16 * ib + j] = idx[32 * ib + j] | (idx[32 * ib + 16 + j] << 4)
    expect = np.array([d * (ls[e // 32] - 32) * KV[idx[e]] for e in range(256)])
    return f16b(d) + struct.pack("<H", sh) + bytes(sl) + bytes(qs), expect


def enc_iq3_s(rng):
    grid = gd.iq3s_grid()
    d = exact_scale(rng)
    s = rng.integers(0, 16, 8)
    idx = rng.integers(0, 512, (8, 8))
    sign = rng.integers(0, 2, (8, 32))
    qs, qh, signs, scales = bytearray(64), bytearray(8), bytearray(32), bytearray(4)
    for ib in range(8):
        for k in range(8):
            qs[8 * ib + k] = idx[ib, k] & 0xFF
            qh[ib] |= (idx[ib, k] >> 8) << k
        for e in range(32):
            signs[4 * ib + e // 8] |= sign[ib, e] << (e % 8)
        scales[ib // 2] |= s[ib] << (4 * (ib % 2))
    expect = np.array([d * (1 + 2 * s[ib]) * grid[idx[ib, e // 4], e % 4] * (-1 if sign[ib, e] else 1)
                       for ib in range(8) for e in range(32)])
    return f16b(d) + bytes(qs) + bytes(qh) + bytes(signs) + bytes(scales), expect


def enc_f32(rng):
    v = rng.standard_normal(1).astype(np.float32)
    return v.tobytes(), v.astype(np.float64)


def enc_f16(rng):
    v = rng.standard_normal(1).astype(np.float16)
    return v.tobytes(), v.astype(np.float64)


def enc_bf16(rng):
    v = rng.standard_normal(1).astype(np.float32)
    u = (v.view(np.uint32) >> 16).astype(np.uint16)  # truncation: any bf16 bit pattern will do
    return u.tobytes(), (u.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


ENCODERS = {"Q8_0": enc_q8_0, "Q4_K": enc_q4_k, "Q5_K": enc_q5_k, "Q6_K": enc_q6_k, "Q3_K": enc_q3_k,
            "IQ4_NL": enc_iq4_nl, "IQ4_XS": enc_iq4_xs, "IQ3_S": enc_iq3_s, "F32": enc_f32,
            "F16": enc_f16, "BF16": enc_bf16}
TYPE_ID = {t.name: i for i, t in gd.GGML_TYPES.items()}


def have_iq3s_grid() -> bool:
    try:
        gd.iq3s_grid()
        return True
    except gd.GGUFError as e:
        print(f"SKIP IQ3_S parts: {e}")
        return False


def build_blocks(tname: str, n: int, seed: int):
    rng = np.random.default_rng(seed)
    raws, expects = [], []
    for _ in range(n):
        b, e = ENCODERS[tname](rng)
        raws.append(b)
        expects.append(e)
    t = gd.GGML_TYPES[TYPE_ID[tname]]
    blob = b"".join(raws)
    assert len(blob) == n * t.type_size, (tname, len(blob))
    return np.frombuffer(blob, dtype=np.uint8).reshape(n, t.type_size), np.concatenate(expects)


def test_hand_built(iq3s: bool) -> dict:
    print("(a) hand-built blocks")
    built = {}
    check(set(ENCODERS) == set(gd.DEQUANTIZERS), f"every dequantizer has an encoder test "
          f"({sorted(set(gd.DEQUANTIZERS) ^ set(ENCODERS))})")
    for tname in ENCODERS:
        if tname == "IQ3_S" and not iq3s:
            continue
        blocks, expect = build_blocks(tname, 7, seed=zlib.crc32(tname.encode()))
        got = gd.dequantize_blocks(tname, blocks)
        exp32 = expect.astype(np.float32)
        check(np.array_equal(exp32.astype(np.float64), expect), f"{tname}: expected values exact in f32")
        check(got.dtype == np.float32 and got.shape == exp32.shape, f"{tname}: shape {got.shape}")
        ok = np.array_equal(got.view(np.uint32), exp32.view(np.uint32))
        check(ok, f"{tname}: decode == the encoded integers' values "
                  f"(max err {float(np.abs(got - exp32).max()) if got.shape == exp32.shape else 'n/a'})")
        built[tname] = blocks
    # A block with every quant and scale at its maximum, and one of all zeros.
    b = bytearray(144)
    b[0:2], b[2:4] = f16b(1.0), f16b(1.0)
    b[4:16] = bytes([0xFF] * 12)  # all scales and mins 63
    b[16:144] = bytes([0xFF] * 128)  # all quants 15
    y = gd.dequantize_blocks("Q4_K", np.frombuffer(bytes(b), np.uint8).reshape(1, 144))
    check(np.all(y == 63 * 15 - 63), "Q4_K saturated block = 63*15 - 63")
    y = gd.dequantize_blocks("Q6_K", np.zeros((1, 210), np.uint8))
    check(np.all(y == 0), "Q6_K all-zero block (d = 0) decodes to 0")
    return built


# ================================================================================================
# (b) the same blocks, and random bytes, through gguf-py (independent implementation), if present
# ================================================================================================


#: Byte offsets of each quantized type's f16 fields (d, and dmin for Q4_K/Q5_K).
F16_FIELDS = {"Q8_0": [0], "Q4_K": [0, 2], "Q5_K": [0, 2], "Q6_K": [208], "Q3_K": [108],
              "IQ4_NL": [0], "IQ4_XS": [0], "IQ3_S": [0]}


def random_blocks(rng, tname: str, n: int) -> np.ndarray:
    """n blocks of random bytes whose f16 fields are finite (NaN != NaN would make a bitwise
    comparison moot)."""
    t = gd.GGML_TYPES[TYPE_ID[tname]]
    rnd = rng.integers(0, 256, (n, t.type_size), dtype=np.uint8)
    for off in F16_FIELDS[tname]:
        rnd[:, off:off + 2] = np.frombuffer(
            (rng.standard_normal(n) * 0.02).astype(np.float16).tobytes(), np.uint8).reshape(n, 2)
    return rnd


def test_against_gguf_py(built: dict, iq3s: bool) -> None:
    print("(b) gguf-py cross-check")
    if not (GGUF_PY / "gguf" / "quants.py").is_file():
        print(f"SKIP (b): no gguf-py source at {GGUF_PY}")
        return
    sys.path.insert(0, str(GGUF_PY))
    try:
        from gguf import quants as gq
        from gguf.constants import GGMLQuantizationType as QT
    except Exception as e:  # its own imports may be missing in this venv
        print(f"SKIP (b): gguf-py at {GGUF_PY} does not import ({type(e).__name__}: {e})")
        return
    finally:
        sys.path.pop(0)
    print(f"    gguf-py from {GGUF_PY}: {sorted(built)}")
    rng = np.random.default_rng(1234)
    for tname, blocks in built.items():
        check(QT[tname].value == TYPE_ID[tname], f"{tname}: ggml type id {TYPE_ID[tname]} == gguf-py's")
        if tname in ("F32", "F16", "BF16"):
            continue
        rnd = random_blocks(rng, tname, 64)
        for what, b in (("hand-built", blocks), ("random bytes", rnd)):
            mine = gd.dequantize_blocks(tname, b)
            ref = np.asarray(gq.dequantize(np.ascontiguousarray(b), QT[tname]), dtype=np.float32).reshape(-1)
            check(np.array_equal(mine.view(np.uint32), ref.view(np.uint32)),
                  f"{tname} {what}: bit-identical to gguf-py")


# ================================================================================================
# (c) a tiny synthetic GGUF
# ================================================================================================


def _s(x: str) -> bytes:
    b = x.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _kv(key: str, vtype: int, payload: bytes) -> bytes:
    return _s(key) + struct.pack("<I", vtype) + payload


def _arr(etype: int, items: list[bytes]) -> bytes:
    return struct.pack("<IQ", etype, len(items)) + b"".join(items)


def write_gguf(path: Path, kvs: list[bytes], tensors: list[tuple[str, tuple, int, bytes]],
               alignment: int = 32, magic: int = gd.GGUF_MAGIC, offsets=None) -> None:
    """tensors: (name, ne, type_id, data). Data blocks are laid out back to back, each padded to the
    alignment, unless `offsets` overrides the recorded offsets."""
    infos, datas, cur = [], [], 0
    for i, (name, ne, tid, data) in enumerate(tensors):
        off = cur if offsets is None else offsets[i]
        infos.append(_s(name) + struct.pack("<I", len(ne)) + struct.pack(f"<{len(ne)}Q", *ne)
                     + struct.pack("<IQ", tid, off))
        pad = (-len(data)) % alignment
        datas.append(data + b"\0" * pad)
        cur += len(data) + pad
    head = struct.pack("<IIQQ", magic, 3, len(tensors), len(kvs)) + b"".join(kvs) + b"".join(infos)
    head += b"\0" * ((-len(head)) % alignment)
    path.write_bytes(head + b"".join(datas))


QWEN35_SMALL = dict(blocks=3, nextn=1, nk=2, nv=6, dk=4, dv=2)


def qwen35_kvs(align: int | None = 64) -> list[bytes]:
    g = QWEN35_SMALL
    kv = [
        _kv("general.architecture", gd.VT_STRING, _s("qwen35")),
        _kv("qwen35.block_count", gd.VT_UINT32, struct.pack("<I", g["blocks"])),
        _kv("qwen35.nextn_predict_layers", gd.VT_UINT32, struct.pack("<I", g["nextn"])),
        _kv("qwen35.ssm.group_count", gd.VT_UINT32, struct.pack("<I", g["nk"])),
        _kv("qwen35.ssm.time_step_rank", gd.VT_UINT32, struct.pack("<I", g["nv"])),
        _kv("qwen35.ssm.state_size", gd.VT_UINT32, struct.pack("<I", g["dk"])),
        _kv("qwen35.ssm.inner_size", gd.VT_UINT32, struct.pack("<I", g["nv"] * g["dv"])),
        _kv("t.u8", gd.VT_UINT8, struct.pack("<B", 200)),
        _kv("t.i8", gd.VT_INT8, struct.pack("<b", -5)),
        _kv("t.u16", gd.VT_UINT16, struct.pack("<H", 60000)),
        _kv("t.i16", gd.VT_INT16, struct.pack("<h", -30000)),
        _kv("t.i32", gd.VT_INT32, struct.pack("<i", -7)),
        _kv("t.f32", gd.VT_FLOAT32, struct.pack("<f", 0.5)),
        _kv("t.bool", gd.VT_BOOL, struct.pack("<?", True)),
        _kv("t.u64", gd.VT_UINT64, struct.pack("<Q", 2 ** 40)),
        _kv("t.i64", gd.VT_INT64, struct.pack("<q", -(2 ** 40))),
        _kv("t.f64", gd.VT_FLOAT64, struct.pack("<d", 0.25)),
        _kv("t.str_utf8", gd.VT_STRING, _s("ภาษาไทย")),
        _kv("t.arr_i32", gd.VT_ARRAY, _arr(gd.VT_INT32, [struct.pack("<i", v) for v in (1, -2, 3)])),
        _kv("t.arr_bool", gd.VT_ARRAY, _arr(gd.VT_BOOL, [b"\x01", b"\x00"])),
        _kv("t.arr_str", gd.VT_ARRAY, _arr(gd.VT_STRING, [_s("a"), _s(""), _s("bc")])),
        _kv("t.arr_arr", gd.VT_ARRAY, _arr(gd.VT_ARRAY, [_arr(gd.VT_UINT8, [b"\x07"]),
                                                         _arr(gd.VT_FLOAT32, [struct.pack("<f", 1.5)])])),
    ]
    if align is not None:
        kv.append(_kv("general.alignment", gd.VT_UINT32, struct.pack("<I", align)))
    return kv


def test_synthetic_gguf(tmp: Path, iq3s: bool) -> None:
    print("(c) synthetic GGUF")
    rng = np.random.default_rng(5)
    f32 = rng.standard_normal((3, 5)).astype(np.float32)           # numpy [3,5] -> ne (5, 3)
    f16 = rng.standard_normal((2, 32)).astype(np.float16)
    q8, q8_exp = build_blocks("Q8_0", 4, 11)                        # 2 rows x 64 -> ne (64, 2)
    q4, q4_exp = build_blocks("Q4_K", 2, 12)                        # 2 rows x 256
    tensors = [("a.f32", (5, 3), 0, f32.tobytes()), ("b.f16", (32, 2), 1, f16.tobytes()),
               ("c.q8_0", (64, 2), 8, q8.tobytes()), ("d.q4_k", (256, 2), 12, q4.tobytes())]
    p = tmp / "tiny.gguf"
    write_gguf(p, qwen35_kvs(64), tensors, alignment=64)
    with gd.GGUFFile(p) as g:
        m = g.metadata
        check(g.version == 3 and g.alignment == 64 and g.data_start % 64 == 0, "version/alignment/data_start")
        check((m["t.u8"], m["t.i8"], m["t.u16"], m["t.i16"], m["t.i32"]) == (200, -5, 60000, -30000, -7),
              "integer scalars")
        check(m["t.f32"] == 0.5 and m["t.f64"] == 0.25 and m["t.bool"] is True, "float/bool scalars")
        check(m["t.u64"] == 2 ** 40 and m["t.i64"] == -(2 ** 40), "64-bit scalars")
        check(m["t.str_utf8"] == "ภาษาไทย", "UTF-8 string")
        check(m["t.arr_i32"] == [1, -2, 3] and m["t.arr_bool"] == [True, False], "numeric arrays")
        check(m["t.arr_str"] == ["a", "", "bc"], "string array")
        check(m["t.arr_arr"] == [[7], [1.5]], "nested array")
        check(g.metadata_types["t.arr_arr"] == gd.VT_ARRAY, "metadata value types recorded")
        check(list(g.tensors) == ["a.f32", "b.f16", "c.q8_0", "d.q4_k"], "tensor order kept")
        check(g.tensor("a.f32").shape == (3, 5) and g.tensor("a.f32").ne == (5, 3), "shape is ne reversed")
        check(all((t.offset - g.data_start) % 64 == 0 for t in g.tensors.values()), "offsets aligned")
        check(np.array_equal(g.dequantize("a.f32"), f32), "F32 tensor")
        check(np.array_equal(g.dequantize("b.f16"), f16.astype(np.float32)), "F16 tensor")
        check(np.array_equal(g.dequantize("c.q8_0").reshape(-1), q8_exp.astype(np.float32)), "Q8_0 tensor")
        check(np.array_equal(g.dequantize("d.q4_k", threads=4).reshape(-1), q4_exp.astype(np.float32)),
              "Q4_K tensor (threaded)")
        check(np.array_equal(g.dequantize_rows("d.q4_k", 1, 2), q4_exp[256:].astype(np.float32)[None]),
              "row range")
        check(np.array_equal(g.dequantize_row_ids("c.q8_0", [1, 0, 1]),
                             q8_exp.astype(np.float32).reshape(2, 64)[[1, 0, 1]]), "row gather")
        check(g.type_counts() == {"F32": 1, "F16": 1, "Q8_0": 1, "Q4_K": 1}, "type counts")
        check(len(g.header_sha256) == 64, "header sha256")
        raises(gd.GGUFError, "outside", lambda: g.raw_rows("c.q8_0", 1, 3), "row range past the end")
        raises(gd.GGUFError, "no tensor", lambda: g.tensor("nope"), "unknown tensor name")

    # default alignment (32) when general.alignment is absent
    p2 = tmp / "tiny32.gguf"
    write_gguf(p2, qwen35_kvs(None), tensors[:1], alignment=32)
    with gd.GGUFFile(p2) as g:
        check(g.alignment == 32 and np.array_equal(g.dequantize("a.f32"), f32), "default alignment 32")

    bad = tmp / "bad.gguf"
    write_gguf(bad, qwen35_kvs(64), tensors, alignment=64, magic=0x12345678)
    raises(gd.GGUFError, "bad magic", lambda: gd.GGUFFile(bad), "bad magic")
    good = p.read_bytes()
    bad.write_bytes(good[:200])
    raises(gd.GGUFError, "truncated", lambda: gd.GGUFFile(bad), "truncated header")
    bad.write_bytes(good[:-64])
    raises(gd.GGUFError, "past the end", lambda: gd.GGUFFile(bad), "tensor past EOF")
    write_gguf(bad, qwen35_kvs(64), tensors, alignment=64, offsets=[0, 64, 64, 256])
    raises(gd.GGUFError, "overlaps", lambda: gd.GGUFFile(bad), "overlapping tensors")
    write_gguf(bad, qwen35_kvs(64), tensors, alignment=64, offsets=[0, 96, 192, 448])
    raises(gd.GGUFError, "aligned", lambda: gd.GGUFFile(bad), "misaligned offset")
    raises(gd.GGUFError, "no dequantizer", lambda: gd.dequantize_blocks("Q2_K", np.zeros((1, 84), np.uint8)),
           "unimplemented type refused")

    # Per-row scale diagnostics: a 3-row Q4_K tensor whose row 1 has d = 0 (the Unsloth embedding's
    # row 107517 in miniature) and whose row 2 has a coarse subnormal d.
    q4r, _ = build_blocks("Q4_K", 3, 13)
    q4r = q4r.copy()
    q4r[1, 0:2] = np.frombuffer(f16b(0.0), np.uint8)
    q4r[2, 0:2] = np.frombuffer(np.float16(3 * 2.0 ** -24).tobytes(), np.uint8)
    p3 = tmp / "rows.gguf"
    write_gguf(p3, qwen35_kvs(32), [("z.q4_k", (256, 3), 12, q4r.tobytes())])
    qm = gd.Qwen35Map(n_trunk=2, n_nextn=1, nk=2, nv=6, dk=4, dv=2)
    with gd.GGUFFile(p3) as g:
        d = g.row_scales("z.q4_k", 1)
        check(d.shape == (1,) and d[0] == 0, "row_scales: the zero d")
        out = gd.row_outliers(g, qm, "z.q4_k", "id", np.array([0.05, 1.3, 0.9]), np.array([1.0, 1e-3, 2e-3]),
                              row_cos=np.array([0.99, 0.0, 0.5]))
        check([o["row"] for o in out] == [1, 2], f"row_outliers: rows above 0.2, worst first ({out})")
        check(out[0]["zero_scale_blocks"] == 1 and out[0]["coarse_scale"] and out[0]["cos"] == 0.0,
              "row_outliers: d = 0 is a coarse scale")
        check(out[1]["zero_scale_blocks"] == 0 and out[1]["coarse_scale_blocks"] == 1 and out[1]["coarse_scale"],
              "row_outliers: a d of 3 * 2^-24 is a coarse scale")
        out = gd.row_outliers(g, qm, "z.q4_k", "id", np.array([0.3]), np.array([1.0]), rows=[0])
        check(len(out) == 1 and out[0]["row"] == 0 and not out[0]["coarse_scale"],
              "row_outliers: a normal d is not a coarse scale; `rows` maps gathered rows")
    rel = gd.row_rel_errors([4.0, 0.0, 1.0], [16.0, 0.0, 0.0])
    check(rel[0] == 0.5 and rel[1] == 0.0 and np.isinf(rel[2]), f"row_rel_errors {rel}")


# ================================================================================================
# (d) the qwen35 map
# ================================================================================================


def test_qwen35_map(tmp: Path) -> None:
    print("(d) qwen35 map")
    g0 = QWEN35_SMALL
    p = tmp / "map.gguf"
    names = ["token_embd.weight", "output.weight", "output_norm.weight", "blk.0.attn_qkv.weight",
             "blk.1.attn_q.weight", "blk.2.nextn.eh_proj.weight", "blk.2.ffn_up.weight", "blk.2.attn_norm.weight",
             "blk.0.mystery.weight", "blk.9.ffn_up.weight"]
    write_gguf(p, qwen35_kvs(32), [(n, (4,), 0, np.zeros(4, np.float32).tobytes()) for n in names])
    with gd.GGUFFile(p) as g:
        m = gd.Qwen35Map.from_gguf(g)
    check((m.n_trunk, m.n_nextn, m.nk, m.nv, m.dk, m.dv) == (2, 1, 2, 6, 4, 2), "geometry from metadata")
    check(m.tiled.tolist() == [0, 3, 1, 4, 2, 5], f"tiled order nk=2 nv=6: {m.tiled.tolist()}")
    check(m.tiled[m.untiled].tolist() == list(range(6)), "untiled inverts tiled")
    L = "model.language_model.layers."
    check(m.hf_of["token_embd.weight"] == ("model.language_model.embed_tokens.weight", "id"), "embed name")
    check(m.hf_of["output_norm.weight"] == ("model.language_model.norm.weight", "plus1"), "final norm name")
    check(m.hf_of["blk.0.attn_qkv.weight"] == (L + "0.linear_attn.in_proj_qkv.weight", "qkv_vrows"), "GDN qkv")
    check(m.hf_of["blk.1.attn_q.weight"] == (L + "1.self_attn.q_proj.weight", "id"), "attn q")
    check(m.hf_of["blk.2.nextn.eh_proj.weight"] == ("mtp.fc.weight", "id"), "MTP fc")
    check(m.hf_of["blk.2.ffn_up.weight"] == ("mtp.layers.0.mlp.up_proj.weight", "id"), "MTP mlp")
    check(m.hf_of["blk.2.attn_norm.weight"] == ("mtp.layers.0.input_layernorm.weight", "plus1"), "MTP norm")
    check(sorted(m.unmapped) == ["blk.0.mystery.weight", "blk.9.ffn_up.weight"], f"unmapped {m.unmapped}")
    check(m.gguf_of[L + "0.linear_attn.in_proj_qkv.weight"] == ("blk.0.attn_qkv.weight", "qkv_vrows"), "reverse map")

    nk, nv, dk, dv = g0["nk"], g0["nv"], g0["dk"], g0["dv"]
    kd, hidden = 2 * nk * dk, 5
    rng = np.random.default_rng(3)
    shapes = {  # tag -> (HF shape, GGUF numpy shape)
        "id": ((7, hidden), (7, hidden)), "plus1": ((hidden,), (hidden,)),
        "qkv_vrows": ((kd + nv * dv, hidden), (kd + nv * dv, hidden)),
        "vrows": ((nv * dv, hidden), (nv * dv, hidden)), "vheads": ((nv, hidden), (nv, hidden)),
        "neg_exp_vheads": ((nv,), (nv,)), "conv_vch": ((kd + nv * dv, 1, 4), (kd + nv * dv, 4)),
        "vcols": ((hidden, nv * dv), (hidden, nv * dv)),
    }
    for tag, (hs, gs) in shapes.items():
        w = rng.standard_normal(hs).astype(np.float32)
        gv = m.from_hf(tag, w, gs)
        back = m.to_hf(tag, gv, hs)
        check(gv.shape == gs and back.shape == hs, f"{tag}: shapes {gv.shape} / {back.shape}")
        if tag in ("neg_exp_vheads", "plus1"):  # float rounding in the +1 / exp: not an identity
            check(np.allclose(back, w, rtol=1e-6, atol=1e-6), f"{tag}: to_hf(from_hf(w)) ~= w")
        else:
            check(np.array_equal(back, w), f"{tag}: to_hf(from_hf(w)) == w")
    # gguf_row: row r of to_hf(tag, g) is row gguf_row(tag, r) of the GGUF tensor g
    for tag, rows in (("id", 7), ("vrows", nv * dv), ("vheads", nv), ("qkv_vrows", kd + nv * dv)):
        gidx = np.arange(rows, dtype=np.float32)[:, None].repeat(3, axis=1)
        back = m.to_hf(tag, gidx, gidx.shape)
        check(all(back[r, 0] == m.gguf_row(tag, r) for r in range(rows)), f"gguf_row inverts to_hf for {tag}")
    check(m.gguf_row("vcols", 3) == 3, "gguf_row: vcols moves columns, not rows")
    # the direction of the reorder, spelled out: GGUF V head h holds HF V head tiled[h]
    w = np.arange(nv * dv, dtype=np.float32).reshape(nv * dv, 1)
    gv = m.from_hf("vrows", w, w.shape)
    check(all(gv[h * dv, 0] == m.tiled[h] * dv for h in range(nv)), "vrows: GGUF head h = HF head tiled[h]")
    w = rng.standard_normal((kd + nv * dv, hidden)).astype(np.float32)
    gv = m.from_hf("qkv_vrows", w, w.shape)
    check(np.array_equal(gv[:kd], w[:kd]) and not np.array_equal(gv[kd:], w[kd:]), "qkv: only V rows move")
    wc = rng.standard_normal((hidden, nv * dv)).astype(np.float32)
    gc = m.from_hf("vcols", wc, wc.shape)
    check(np.array_equal(gc[:, dv:2 * dv], wc[:, m.tiled[1] * dv:(m.tiled[1] + 1) * dv]), "vcols permute columns")
    a_log = rng.standard_normal(nv).astype(np.float32)
    g_a = m.from_hf("neg_exp_vheads", a_log, (nv,))
    check(np.all(g_a < 0), "ssm_a = -exp(A_log) < 0")
    ok, ulp = gd.lossless_matches(m, "neg_exp_vheads", g_a, a_log)
    check(ok and ulp == 0, "lossless_matches exact")
    norm = rng.standard_normal(hidden).astype(np.float32)
    g_n = m.from_hf("plus1", norm, (hidden,))
    bumped = g_n.copy()
    bumped.view(np.int32)[2] += 1
    ok, ulp = gd.lossless_matches(m, "plus1", bumped, norm)
    check(not ok and ulp == 1, "lossless_matches rejects a 1-ulp change of a plus1 tensor")
    bumped = g_a.copy()
    bumped.view(np.int32)[0] += 1
    check(gd.lossless_matches(m, "neg_exp_vheads", bumped, a_log)[0], "neg_exp_vheads allows 1 ulp")
    bumped.view(np.int32)[0] += 1
    check(not gd.lossless_matches(m, "neg_exp_vheads", bumped, a_log)[0], "neg_exp_vheads rejects 2 ulp")
    check(gd.tensor_class(L + "5.linear_attn.in_proj_qkv.weight") == "gdn.in_proj_qkv", "class gdn")
    check(gd.tensor_class(L + "3.self_attn.q_proj.weight") == "attn.q_proj", "class attn")
    check(gd.tensor_class("mtp.fc.weight") == "mtp.fc", "class mtp.fc")
    check(gd.tensor_class(L + "0.linear_attn.A_log") == "gdn.A_log", "class A_log")


# ================================================================================================
# (e) the real GGUF against the real checkpoint
# ================================================================================================


def test_real_file(iq3s: bool) -> None:
    print("(e) real GGUF")
    if not REAL_GGUF.is_file():
        print(f"SKIP (e): {REAL_GGUF} not on disk")
        return
    import gguf_validate as gv

    with gd.GGUFFile(REAL_GGUF) as g:
        m = gd.Qwen35Map.from_gguf(g)
        check(g.version == 3 and len(g.tensors) == 866, f"866 tensors ({len(g.tensors)})")
        check(g.type_counts() == {"F32": 360, "Q5_K": 191, "Q8_0": 110, "IQ4_XS": 70, "Q4_K": 69, "Q6_K": 56,
                                  "IQ4_NL": 6, "Q3_K": 3, "IQ3_S": 1}, f"type counts {g.type_counts()}")
        check(len(m.hf_of) == 866 and not m.unmapped, "every tensor maps")
        check((m.n_trunk, m.n_nextn, m.nk, m.nv, m.dk, m.dv) == (64, 1, 16, 48, 128, 128), "geometry")
        last = max(g.tensors.values(), key=lambda t: t.offset)
        check(last.offset + last.nbytes == g.file_size, "tensors tile the file exactly")
        if not (REAL_MODEL_DIR / "model.safetensors.index.json").is_file():
            print(f"SKIP (e) tensor checks: no checkpoint at {REAL_MODEL_DIR}")
            return
        ck = gv.SafetensorsCheckpoint(REAL_MODEL_DIR)
        picks = ["blk.0.attn_qkv.weight", "blk.0.attn_gate.weight", "blk.0.ssm_alpha.weight", "blk.0.ssm_beta.weight",
                 "blk.0.ssm_out.weight", "blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_conv1d.weight",
                 "blk.0.ssm_norm.weight", "blk.0.attn_norm.weight", "blk.3.attn_q.weight", "blk.3.attn_k.weight",
                 "blk.3.attn_q_norm.weight", "blk.11.attn_k.weight", "blk.0.ffn_up.weight", "blk.1.ffn_down.weight",
                 "blk.0.ffn_down.weight", "output_norm.weight"]
        if iq3s:
            picks.append("blk.14.ffn_down.weight")
        types = set()
        for name in picks:
            r = gv.check_tensor(name, g, m, ck, threads=8)
            types.add(r["type"])
            extra = f" rel_no_inverse={r['rel_no_inverse']:.3f}" if "rel_no_inverse" in r else ""
            print(f"    {name:28s} {r['type']:7s} {r.get('tag', ''):15s} {r.get('kind', ''):9s} "
                  f"rel={r.get('rel', float('nan')):.4f} cos={r.get('cos', float('nan')):.6f}{extra}")
            check(r.get("ok"), f"{name}: {r.get('why')}")
            if "rel_no_inverse" in r:
                check(r["rel_no_inverse"] > 0.5, f"{name}: skipping the inverse reorder is caught "
                                                 f"(rel {r['rel_no_inverse']:.3f})")
            if r.get("kind") == "quantized":
                check(r["row_rel_max"] >= r["row_rel_median"] > 0 and r["n_rows"] >= 1,
                      f"{name}: per-row statistics recorded")
        want = {"F32", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "Q3_K", "IQ4_NL", "IQ4_XS"} | ({"IQ3_S"} if iq3s else set())
        check(want <= types, f"real-tensor checks cover {sorted(want - types)}")

        # Per row: token_embd row 107517 is the file's one catastrophic embedding row. Its bf16
        # values are ~1e-5, every f16 d of its 20 super-blocks is 0, and it decodes to rel ~1.40.
        # Rows 494 and 0 are ordinary (494 is the worst row tokens.json uses, rel ~0.106).
        hf_embed = "model.language_model.embed_tokens.weight"
        ids = [0, 494, 107517]
        deq = g.dequantize_row_ids("token_embd.weight", ids).astype(np.float64)
        w = np.concatenate([ck.rows(hf_embed, i, i + 1) for i in ids]).astype(np.float64)
        rel = np.linalg.norm(deq - w, axis=1) / np.linalg.norm(w, axis=1)
        out = gd.row_outliers(g, m, "token_embd.weight", "id", rel, np.linalg.norm(w, axis=1), rows=ids)
        print(f"    token_embd rows {ids}: rel {np.round(rel, 4).tolist()}; outliers {[o['row'] for o in out]}")
        check(0.08 < rel[0] < 0.1 and 0.1 < rel[1] < 0.11 and 1.3 < rel[2] < 1.5, f"embedding row errors {rel}")
        check(len(out) == 1 and out[0]["row"] == 107517 and out[0]["coarse_scale"]
              and out[0]["zero_scale_blocks"] == out[0]["blocks"] == 20,
              "token_embd row 107517 is the one outlier, with all 20 scales 0")


# ================================================================================================
# (f) ggml's own C dequantizers, if a local llama.cpp build is on disk
# ================================================================================================


def test_against_ggml_c(built: dict, iq3s: bool) -> None:
    print("(f) ggml C cross-check")
    c = gd.load_ggml_c()
    if c is None:
        print(f"SKIP (f): no ggml-base library at {[str(p) for p in gd.ggml_base_candidates()]}")
        return
    print(f"    {c.path}: {c.types}")
    check(c.types == sorted(gd.GGML_C_DEQUANT), f"ggml C exports every dequantizer ({c.types})")
    rng = np.random.default_rng(4321)
    for tname, blocks in built.items():
        if tname not in c.types:
            continue
        for what, b in (("hand-built", blocks), ("random bytes", random_blocks(rng, tname, 64))):
            mine = gd.dequantize_blocks(tname, b)
            ref = c.dequantize(tname, b)
            check(np.array_equal(mine.view(np.uint32), ref.view(np.uint32)),
                  f"{tname} {what}: bit-identical to ggml's C")
    # The comparison has teeth: one flipped quant bit changes exactly one C output value.
    if "Q4_K" in built:
        b = built["Q4_K"].copy()
        b[0, 20] ^= 0x10  # the high nibble of qs[4]: element 36 of block 0
        diff = np.nonzero(c.dequantize("Q4_K", b) != gd.dequantize_blocks("Q4_K", built["Q4_K"]))[0]
        check(diff.tolist() == [36], f"a flipped Q4_K bit moves exactly element 36 ({diff.tolist()[:5]})")

    if not REAL_GGUF.is_file():
        print(f"SKIP (f) real tensors: {REAL_GGUF} not on disk")
        return
    with gd.GGUFFile(REAL_GGUF) as g:
        picks: dict[str, str] = {}
        for name, t in g.tensors.items():  # the first tensor of every quantized type
            if t.type_name in c.types and (t.type_name != "IQ3_S" or iq3s):
                picks.setdefault(t.type_name, name)
        names = sorted(set(picks.values()) | {"token_embd.weight", "output.weight"})
        n_vals = 0
        for name in names:
            t = g.tensor(name)
            n = t.n_rows
            ids = sorted({0, n - 1, *rng.choice(n, size=min(24, n), replace=False).tolist()}
                         | ({494, 107517} if name == "token_embd.weight" else set()))
            mine = g.dequantize_row_ids(name, ids)
            ref = c.dequantize(t.type_name, g.raw_rows(name)[ids]).reshape(mine.shape)
            r0 = int(rng.integers(0, max(1, n - 256)))
            mine_r = g.dequantize_rows(name, r0, min(n, r0 + 256), threads=4)
            ref_r = c.dequantize(t.type_name, g.raw_rows(name, r0, min(n, r0 + 256))).reshape(mine_r.shape)
            n_vals += mine.size + mine_r.size
            check(np.array_equal(mine.view(np.uint32), ref.view(np.uint32))
                  and np.array_equal(mine_r.view(np.uint32), ref_r.view(np.uint32)),
                  f"{name} ({t.type_name}): {len(ids)} gathered + 256 contiguous rows bit-identical to ggml's C")
        print(f"    real tensors: {names} ({n_vals / 1e6:.1f} M values)")
        check({g.tensor(n).type_name for n in names} >= set(c.types) - (set() if iq3s else {"IQ3_S"}),
              "real-tensor C checks cover every quantized type")


def main() -> int:
    iq3s = have_iq3s_grid()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            built = test_hand_built(iq3s)
        except Exception:
            traceback.print_exc()
            check(False, "hand-built raised")
            built = {}
        for label, fn in (("gguf-py", lambda: test_against_gguf_py(built, iq3s)),
                          ("synthetic", lambda: test_synthetic_gguf(tmp, iq3s)),
                          ("qwen35 map", lambda: test_qwen35_map(tmp)),
                          ("real file", lambda: test_real_file(iq3s)),
                          ("ggml C", lambda: test_against_ggml_c(built, iq3s))):
            try:
                fn()
            except Exception:
                traceback.print_exc()
                check(False, f"{label} raised")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)} of {CHECKS} checks):")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    print(f"OK ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
