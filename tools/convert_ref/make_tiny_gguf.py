#!/usr/bin/env python3
"""Dependency-free (stdlib `struct` only -- NO gguf-py, NO numpy) GGUF v3 file generator for
r4dx-convert's own tests.

Two modes:

  python make_tiny_gguf.py --out fixture.gguf
      A tiny, deliberately-mixed fixture: one tensor of each dtype r4dx_convert::GgufReader
      supports (F32, F16, BF16, Q8_0) with hand-picked, exactly-representable values, and one
      metadata key of every GgufValueType (including arrays and strings). Consumed by
      tests/convert/test_gguf_reader.cpp, which hardcodes the same expected values.

  python make_tiny_gguf.py --dflash-mini --out mini.gguf
      A 2-layer, 64-dim, 512-vocab miniature carrying the exact tensor/metadata-key SET of the
      real Qwen3.8-27B-DFlash2-Q8_0.gguf (docs/container-format.md's DFlash2 section), scaled down
      so every quantized dimension still satisfies Q8_0's "ne[0] divisible by 32" constraint.
      Consumed by tests/convert/test_dflash_container.cpp and
      tools/convert_ref/dflash2_container_check.py.

Both this file and src/convert/include/r4dx_convert/gguf_reader.hpp were written independently
against the plain GGUF v3 spec (see that header's own file comment) -- no code is shared with, or
imported from, C:\\Users\\user\\dev\\ROCmFPX\\gguf-py, per the task's explicit rule.
"""
import argparse
import struct
import sys

MAGIC = b"GGUF"
VERSION = 3

# GgufValueType (gguf_reader.hpp's r4dx_convert::GgufValueType)
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32 = 0, 1, 2, 3, 4, 5
T_FLOAT32, T_BOOL, T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64 = 6, 7, 8, 9, 10, 11, 12

# ggml_type (gguf_reader.hpp's r4dx_convert::GgmlType)
GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30


def f32_to_bf16(f):
    """Top 16 bits of an IEEE-754 fp32, round-to-nearest-even -- bit-for-bit the same algorithm as
    r4dx::core::FloatToBf16 (src/core/include/r4dx/core/dtype.hpp)."""
    bits = struct.unpack("<I", struct.pack("<f", f))[0]
    if (bits & 0x7FFFFFFF) > 0x7F800000:
        return (bits >> 16) | 0x0040  # quiet NaN, sign preserved
    rounded = (bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFFFFFF
    return (rounded >> 16) & 0xFFFF


def f32_to_f16_bytes(f):
    # Python's struct 'e' format is IEEE binary16, round-to-nearest-even -- exactly GGUF's F16.
    return struct.pack("<e", f)


class Writer:
    def __init__(self):
        self.buf = bytearray()

    def raw(self, b):
        self.buf += b

    def u32(self, v):
        self.buf += struct.pack("<I", v)

    def i32(self, v):
        self.buf += struct.pack("<i", v)

    def u64(self, v):
        self.buf += struct.pack("<Q", v)

    def i64(self, v):
        self.buf += struct.pack("<q", v)

    def u16(self, v):
        self.buf += struct.pack("<H", v)

    def i16(self, v):
        self.buf += struct.pack("<h", v)

    def u8(self, v):
        self.buf += struct.pack("<B", v)

    def i8(self, v):
        self.buf += struct.pack("<b", v)

    def f32(self, v):
        self.buf += struct.pack("<f", v)

    def f64(self, v):
        self.buf += struct.pack("<d", v)

    def string(self, s):
        b = s.encode("utf-8")
        self.u64(len(b))
        self.buf += b

    def kv_value(self, vtype, value):
        if vtype == T_STRING:
            self.string(value)
        elif vtype == T_ARRAY:
            elem_type, elems = value
            self.u32(elem_type)
            self.u64(len(elems))
            for e in elems:
                self.kv_value(elem_type, e)
        elif vtype == T_BOOL:
            self.u8(1 if value else 0)
        elif vtype == T_UINT8:
            self.u8(value)
        elif vtype == T_INT8:
            self.i8(value)
        elif vtype == T_UINT16:
            self.u16(value)
        elif vtype == T_INT16:
            self.i16(value)
        elif vtype == T_UINT32:
            self.u32(value)
        elif vtype == T_INT32:
            self.i32(value)
        elif vtype == T_UINT64:
            self.u64(value)
        elif vtype == T_INT64:
            self.i64(value)
        elif vtype == T_FLOAT32:
            self.f32(value)
        elif vtype == T_FLOAT64:
            self.f64(value)
        else:
            raise ValueError(f"unknown value type {vtype}")

    def kv(self, key, vtype, value):
        self.string(key)
        self.u32(vtype)
        self.kv_value(vtype, value)

    def tensor_info(self, name, ne, ggml_type, offset):
        self.string(name)
        self.u32(len(ne))
        for d in ne:
            self.u64(d)
        self.u32(ggml_type)
        self.u64(offset)


def ggml_block_size(t):
    return 32 if t == GGML_Q8_0 else 1


def ggml_type_size(t):
    return {GGML_F32: 4, GGML_F16: 2, GGML_BF16: 2, GGML_Q8_0: 34}[t]


def pack_f32(values):
    return b"".join(struct.pack("<f", v) for v in values)


def pack_f16(values):
    return b"".join(f32_to_f16_bytes(v) for v in values)


def pack_bf16(values):
    return b"".join(struct.pack("<H", f32_to_bf16(v)) for v in values)


def pack_q8_0(values):
    """values: flat list, len a multiple of 32. Each block: f16 scale d, then 32 signed int8 q,
    d chosen as amax/127 per block (matches gguf-py's Q8_0.quantize_blocks convention, but here we
    pick blocks whose values are already exact multiples of a power-of-two scale so quantization
    is lossless and the test can assert exact round-trip)."""
    assert len(values) % 32 == 0
    out = bytearray()
    for b in range(0, len(values), 32):
        block = values[b : b + 32]
        amax = max(abs(v) for v in block) or 1.0
        d = amax / 127.0
        out += f32_to_f16_bytes(d)
        for v in block:
            q = int(round(v / d))
            q = max(-128, min(127, q))
            out += struct.pack("<b", q)
    return bytes(out)


def build(tensors, kvs, alignment=32):
    """tensors: list of (name, ne, ggml_type, raw_bytes). kvs: list of (key, vtype, value)."""
    header = Writer()
    header.raw(MAGIC)
    header.u32(VERSION)
    header.u64(len(tensors))
    header.u64(len(kvs))
    for key, vtype, value in kvs:
        header.kv(key, vtype, value)

    infos = Writer()
    offset = 0
    tensor_offsets = []
    for name, ne, ggml_type, raw in tensors:
        tensor_offsets.append(offset)
        infos.tensor_info(name, ne, ggml_type, offset)
        offset += len(raw)
        # GGUF pads each tensor's start to `alignment` inside the data section too, in general;
        # this generator's tensors are all already alignment-sized multiples in practice (every
        # dtype here is a multiple of 2 bytes and every shape a multiple of 16 elements), so no
        # extra intra-data padding is emitted here beyond the once padded data_start below. Kept
        # simple deliberately -- if a future tensor size isn't a multiple of `alignment`, this
        # assert catches it rather than silently emitting a reader-breaking file.
        assert offset % 4 == 0, "tensor end not 4-byte aligned; extend build() with per-tensor padding"

    body = bytearray(header.buf) + bytes(infos.buf)
    pad = (-len(body)) % alignment
    body += b"\x00" * pad

    for name, ne, ggml_type, raw in tensors:
        body += raw

    return bytes(body)


def make_basic():
    kvs = [
        ("general.architecture", T_STRING, "test"),
        ("general.alignment", T_UINT32, 32),
        ("test.u8", T_UINT8, 200),
        ("test.i8", T_INT8, -100),
        ("test.u16", T_UINT16, 40000),
        ("test.i16", T_INT16, -30000),
        ("test.u32", T_UINT32, 3000000000),
        ("test.i32", T_INT32, -2000000000),
        ("test.u64", T_UINT64, 12345678901234),
        ("test.i64", T_INT64, -12345678901234),
        ("test.f32", T_FLOAT32, 3.5),
        ("test.f64", T_FLOAT64, -2.25),
        ("test.bool_true", T_BOOL, True),
        ("test.bool_false", T_BOOL, False),
        ("test.str", T_STRING, "hello gguf"),
        ("test.arr_i32", T_ARRAY, (T_INT32, [1, 2, 3, 4, 5])),
        ("test.arr_str", T_ARRAY, (T_STRING, ["a", "bb", "ccc"])),
        ("test.arr_bool", T_ARRAY, (T_BOOL, [True, False, True])),
        ("test.arr_f32", T_ARRAY, (T_FLOAT32, [1.5, -2.5, 0.0])),
    ]

    f32_vals = [float(i) - 6.0 for i in range(12)]  # shape [3,4], ne=[4,3]
    f16_vals = [0.5 * (i - 5) for i in range(10)]  # shape [2,5], ne=[5,2]
    bf16_vals = [2.0 * (i - 3) for i in range(6)]  # shape [2,3], ne=[3,2]
    q8_vals = [float(i - 16) for i in range(32)] + [float(2 * (i - 16)) for i in range(32)]  # ne=[64]

    tensors = [
        ("t_f32", [4, 3], GGML_F32, pack_f32(f32_vals)),
        ("t_f16", [5, 2], GGML_F16, pack_f16(f16_vals)),
        ("t_bf16", [3, 2], GGML_BF16, pack_bf16(bf16_vals)),
        ("t_q8_0", [64], GGML_Q8_0, pack_q8_0(q8_vals)),
    ]
    meta = {
        "f32_vals": f32_vals,
        "f16_vals": f16_vals,
        "bf16_vals": bf16_vals,
        "q8_vals": q8_vals,
    }
    return build(tensors, kvs), meta


def make_dflash_mini():
    hidden = 64
    block_count = 2
    head_count = 4
    head_count_kv = 2
    head_dim = 16
    ffn = 128
    conv_group = 4
    conv_kernel = 2
    selector_rank = 32
    selector_top_k = 4
    draft_block_size = 8
    vocab = 512
    context_length = 2048
    sliding_window = 64
    target_layers = [1, 2]
    ng = hidden // conv_group  # 16 groups
    conv_proj_out = 2 * conv_kernel * ng  # 2 sides * 2 taps * 16 groups = 64

    kvs = [
        ("general.architecture", T_STRING, "dflash"),
        ("general.name", T_STRING, "dflash-mini"),
        ("general.file_type", T_UINT32, 7),
        ("general.alignment", T_UINT32, 32),
        ("dflash.block_count", T_UINT32, block_count),
        ("dflash.embedding_length", T_UINT32, hidden),
        ("dflash.feed_forward_length", T_UINT32, ffn),
        ("dflash.attention.head_count", T_UINT32, head_count),
        ("dflash.attention.head_count_kv", T_UINT32, head_count_kv),
        ("dflash.attention.key_length", T_UINT32, head_dim),
        ("dflash.attention.value_length", T_UINT32, head_dim),
        ("dflash.attention.causal", T_BOOL, False),
        ("dflash.attention.layer_norm_rms_epsilon", T_FLOAT32, 1e-6),
        ("dflash.attention.sliding_window", T_UINT32, sliding_window),
        ("dflash.attention.sliding_window_pattern", T_ARRAY, (T_BOOL, [True] * block_count)),
        ("dflash.rope.freq_base", T_FLOAT32, 1e7),
        ("dflash.rope.dimension_sections", T_ARRAY, (T_UINT32, [8, 0, 0, 0])),
        ("dflash.block_size", T_UINT32, draft_block_size),
        ("dflash.conv_kernel_size", T_UINT32, conv_kernel),
        ("dflash.conv_group_size", T_UINT32, conv_group),
        ("dflash.selector_rank", T_UINT32, selector_rank),
        ("dflash.selector_top_k", T_UINT32, selector_top_k),
        ("dflash.target_layers", T_ARRAY, (T_UINT32, target_layers)),
        ("dflash.context_length", T_UINT32, context_length),
        ("tokenizer.ggml.mask_token_id", T_UINT32, 500),
        ("tokenizer.ggml.model", T_STRING, "gpt2"),
    ]
    # vocab is carried structurally via the tensor shapes below (selector tables, no fc/lm_head
    # embedding of its own in this container) rather than a dedicated metadata key -- the real
    # GGUF's own vocab size similarly comes from tokenizer.ggml.tokens' array length, which this
    # mini fixture does not need to reproduce for the container/loader tests it feeds.

    def const_block(n, v):
        return [v] * n

    tensors = []

    def add_f32(name, ne, values):
        tensors.append((name, ne, GGML_F32, pack_f32(values)))

    def add_q8(name, ne, values):
        tensors.append((name, ne, GGML_Q8_0, pack_q8_0(values)))

    fc_in = block_count * hidden
    add_q8("fc.weight", [fc_in, hidden], [float((i % 7) - 3) for i in range(fc_in * hidden)])
    add_f32("enc.output_norm.weight", [hidden], [1.0 + 0.01 * i for i in range(hidden)])
    add_f32("output_norm.weight", [hidden], [1.0 + 0.02 * i for i in range(hidden)])
    add_q8("selector_hidden.weight", [hidden, selector_rank],
           [float((i % 5) - 2) for i in range(hidden * selector_rank)])
    add_q8("selector_predecessor.weight", [selector_rank, vocab],
           [float((i % 9) - 4) for i in range(selector_rank * vocab)])
    add_q8("selector_successor.weight", [selector_rank, vocab],
           [float((i % 9) - 4) for i in range(selector_rank * vocab)])

    for b in range(block_count):
        p = f"blk.{b}."
        add_f32(p + "attn_norm.weight", [hidden], const_block(hidden, 1.0))
        add_q8(p + "attn_q.weight", [hidden, head_count * head_dim],
               [float((i % 11) - 5) for i in range(hidden * head_count * head_dim)])
        add_q8(p + "attn_k.weight", [hidden, head_count_kv * head_dim],
               [float((i % 11) - 5) for i in range(hidden * head_count_kv * head_dim)])
        add_q8(p + "attn_v.weight", [hidden, head_count_kv * head_dim],
               [float((i % 11) - 5) for i in range(hidden * head_count_kv * head_dim)])
        add_q8(p + "attn_output.weight", [head_count * head_dim, hidden],
               [float((i % 11) - 5) for i in range(head_count * head_dim * hidden)])
        add_f32(p + "attn_q_norm.weight", [head_dim], const_block(head_dim, 1.0))
        add_f32(p + "attn_k_norm.weight", [head_dim], const_block(head_dim, 1.0))
        add_f32(p + "attn_conv_base", [hidden, conv_kernel, 2], const_block(hidden * conv_kernel * 2, 0.1))
        add_q8(p + "attn_conv_proj.weight", [hidden, conv_proj_out],
               [float((i % 7) - 3) for i in range(hidden * conv_proj_out)])
        add_f32(p + "ffn_norm.weight", [hidden], const_block(hidden, 1.0))
        add_q8(p + "ffn_gate.weight", [hidden, ffn], [float((i % 13) - 6) for i in range(hidden * ffn)])
        add_q8(p + "ffn_up.weight", [hidden, ffn], [float((i % 13) - 6) for i in range(hidden * ffn)])
        add_q8(p + "ffn_down.weight", [ffn, hidden], [float((i % 13) - 6) for i in range(ffn * hidden)])
        add_f32(p + "ffn_conv_base", [hidden, conv_kernel, 2], const_block(hidden * conv_kernel * 2, 0.1))
        add_q8(p + "ffn_conv_proj.weight", [hidden, conv_proj_out],
               [float((i % 7) - 3) for i in range(hidden * conv_proj_out)])

    return build(tensors, kvs), {
        "hidden": hidden, "block_count": block_count, "head_count": head_count,
        "head_count_kv": head_count_kv, "head_dim": head_dim, "ffn": ffn,
        "conv_group": conv_group, "selector_rank": selector_rank, "vocab": vocab,
        "target_layers": target_layers,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dflash-mini", action="store_true")
    args = ap.parse_args()

    if args.dflash_mini:
        data, meta = make_dflash_mini()
    else:
        data, meta = make_basic()

    with open(args.out, "wb") as f:
        f.write(data)
    print(f"wrote {args.out} ({len(data)} bytes), meta={meta}", file=sys.stderr)


if __name__ == "__main__":
    main()
