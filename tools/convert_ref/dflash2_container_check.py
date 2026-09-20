"""Independent check for a DFlash2 draft r4dx container (docs/container-format.md "DFlash2 draft
container") against its source GGUF -- task A1 item 5. Own minimal GGUF v3 parser (struct-only,
shares no code with gguf-py; the reader half mirrors tools/convert_ref/make_tiny_gguf.py's writer
half and src/convert/include/r4dx_convert/gguf_reader.hpp's C++ reader, all three written
independently against the plain GGUF v3 spec).

Checks:
  1. bf16-layout container: EVERY tensor's bytes are bit-exact to Q8_0-dequant-RTNE-bf16 (or
     F32/F16/BF16-dequant-RTNE-bf16, whichever the source tensor's own dtype is) of the GGUF.
     Fully vectorized (numpy), so this covers the real, full-size container in seconds.
  2. Quantized layouts (w4a16/w4a8/mxfp4): reuses this repo's existing byte-exact Python references
     (w4_ref.py, mxfp4_ref.py -- the SAME ones tools/convert_ref/selftest_compare.py diffs the main
     model's converter against) to re-quantize+re-pack each linear's FIRST tile (16 rows x one
     k-block: 64 cols for w4a16/w4a8, 32 cols for mxfp4) and byte-compare against the container's
     own bytes at that tile's offset. A full-tensor byte-exact re-pack in pure Python is not
     tractable at these tensor sizes (w4_ref.pack_nibbles is an O(N*K) nested Python loop, written
     for selftest_compare.py's tiny 48x384 fixture) -- the first-tile spot check exercises the
     IDENTICAL quantize+pack code path (quantization is per-row/group, packing is per-tile, both
     purely local, so tile (0,0) of the full matrix is byte-identical to what a from-scratch
     N=16,K=blocksize re-pack of the tensor's own top-left corner produces) without re-deriving a
     vectorized packer. Reported per-tensor, not a full-tensor guarantee -- see this script's own
     printed caveat and the task's open_issues.

Usage:
  python dflash2_container_check.py --gguf <DFlash2 gguf> --container <....r4dx> [--layout {bf16,w4a16,w4a8,mxfp4}]
"""
import argparse
import json
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mxfp4_ref
import w4_ref
from common import F32

# ---- minimal GGUF v3 reader (no gguf-py) -------------------------------------------------------

T_STRING, T_ARRAY, T_BOOL = 8, 9, 7
SCALAR_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 10: "<Q", 11: "<q", 12: "<d"}
SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 10: 8, 11: 8, 12: 8}

GGML_F32, GGML_F16, GGML_Q8_0, GGML_BF16 = 0, 1, 8, 30


class GgufFile:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        pos = 0
        assert self.data[0:4] == b"GGUF"
        pos = 4
        (version,) = struct.unpack_from("<I", self.data, pos); pos += 4
        assert version == 3
        (n_tensors,) = struct.unpack_from("<Q", self.data, pos); pos += 8
        (n_kv,) = struct.unpack_from("<Q", self.data, pos); pos += 8
        self.kv = {}
        for _ in range(n_kv):
            key, pos = self._read_str(pos)
            val, pos = self._read_val(pos)
            self.kv[key] = val
        self.alignment = self.kv.get("general.alignment", 32)
        self.tensors = {}
        self.order = []
        for _ in range(n_tensors):
            name, pos = self._read_str(pos)
            (n_dims,) = struct.unpack_from("<I", self.data, pos); pos += 4
            ne = struct.unpack_from("<" + "Q" * n_dims, self.data, pos); pos += 8 * n_dims
            (ttype,) = struct.unpack_from("<I", self.data, pos); pos += 4
            (offset,) = struct.unpack_from("<Q", self.data, pos); pos += 8
            self.tensors[name] = (list(ne), ttype, offset)
            self.order.append(name)
        rem = pos % self.alignment
        self.data_start = pos if rem == 0 else pos + (self.alignment - rem)

    def _read_str(self, pos):
        (n,) = struct.unpack_from("<Q", self.data, pos); pos += 8
        s = self.data[pos:pos + n].decode("utf-8"); pos += n
        return s, pos

    def _read_val(self, pos):
        (vtype,) = struct.unpack_from("<I", self.data, pos); pos += 4
        return self._read_val_payload(pos, vtype)

    def _read_val_payload(self, pos, vtype):
        if vtype == T_STRING:
            return self._read_str(pos)
        if vtype == T_ARRAY:
            (elem_type,) = struct.unpack_from("<I", self.data, pos); pos += 4
            (count,) = struct.unpack_from("<Q", self.data, pos); pos += 8
            out = []
            for _ in range(count):
                v, pos = self._read_val_payload(pos, elem_type)
                out.append(v)
            return out, pos
        if vtype == T_BOOL:
            v = self.data[pos] != 0
            return v, pos + 1
        fmt, size = SCALAR_FMT[vtype], SCALAR_SIZE[vtype]
        (v,) = struct.unpack_from(fmt, self.data, pos)
        return v, pos + size

    def tensor_bytes(self, name):
        ne, ttype, offset = self.tensors[name]
        n = 1
        for d in ne:
            n *= d
        block = 32 if ttype == GGML_Q8_0 else 1
        tsize = {GGML_F32: 4, GGML_F16: 2, GGML_BF16: 2, GGML_Q8_0: 34}[ttype]
        nbytes = (n // block) * tsize
        start = self.data_start + offset
        return self.data[start:start + nbytes], ne, ttype

    def dequant_f32(self, name):
        raw, ne, ttype = self.tensor_bytes(name)
        if ttype == GGML_F32:
            arr = np.frombuffer(raw, dtype="<f4").astype(F32)
        elif ttype == GGML_F16:
            arr = np.frombuffer(raw, dtype="<f2").astype(F32)
        elif ttype == GGML_BF16:
            u16 = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            arr = (u16 << 16).view(np.uint32).view(F32)
        elif ttype == GGML_Q8_0:
            blocks = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 34)
            d = blocks[:, 0:2].copy().view("<f2").astype(F32).reshape(-1)
            qs = blocks[:, 2:34].view(np.int8).astype(F32)
            arr = (qs * d[:, None]).reshape(-1)
        else:
            raise ValueError(f"unsupported ggml_type {ttype}")
        # GGUF ne is [K,N] (ne[0] fastest/innermost); flat storage is already row-major [N,K].
        if len(ne) == 2:
            K, N = ne
            return arr.reshape(N, K)
        return arr.reshape(ne[::-1])

    def dequant_bf16_bytes(self, name):
        f = self.dequant_f32(name)
        bits = f.view(np.uint32)
        nan_mask = (bits & 0x7FFFFFFF) > 0x7F800000
        rounded = (bits + np.uint32(0x7FFF) + ((bits >> 16) & np.uint32(1))).astype(np.uint32)
        bf16 = (rounded >> 16).astype(np.uint16)
        bf16_nan = ((bits >> 16) | 0x0040).astype(np.uint16)
        return np.where(nan_mask, bf16_nan, bf16)


# ---- r4dx container reader (safetensors-shaped shell) ------------------------------------------

class ContainerFile:
    def __init__(self, path):
        with open(path, "rb") as f:
            self.data = f.read()
        (n,) = struct.unpack_from("<Q", self.data, 0)
        self.header = json.loads(self.data[8:8 + n])
        self.data_start = 8 + n

    def has(self, name):
        return name in self.header

    def bytes(self, name):
        b, e = self.header[name]["data_offsets"]
        return self.data[self.data_start + b:self.data_start + e]

    def shape(self, name):
        return self.header[name]["shape"]

    @property
    def metadata(self):
        return self.header["__metadata__"]


DFLASH2_TENSORS = ["fc", "enc_output_norm", "output_norm", "selector.hidden",
                   "selector.predecessor", "selector.successor"]


def layer_tensors(block_count):
    out = []
    for i in range(block_count):
        p = f"layers.{i}."
        out += [p + "input_layernorm", p + "self_attn.q_proj", p + "self_attn.k_proj",
                p + "self_attn.v_proj", p + "self_attn.o_proj", p + "self_attn.q_norm",
                p + "self_attn.k_norm", p + "self_attn.conv.base", p + "self_attn.conv.proj",
                p + "post_attention_layernorm", p + "mlp.gate_proj", p + "mlp.up_proj",
                p + "mlp.down_proj", p + "mlp.conv.base", p + "mlp.conv.proj"]
    return out


GGUF_NAME = {
    "fc": "fc.weight", "enc_output_norm": "enc.output_norm.weight", "output_norm": "output_norm.weight",
    "selector.hidden": "selector_hidden.weight", "selector.predecessor": "selector_predecessor.weight",
    "selector.successor": "selector_successor.weight",
}


def gguf_name_for(container_suffix, block_count):
    if container_suffix in GGUF_NAME:
        return GGUF_NAME[container_suffix]
    # "layers.{i}.X" -> "blk.{i}.<gguf-name>"
    parts = container_suffix.split(".", 2)
    i = parts[1]
    rest = parts[2]
    mapping = {
        "input_layernorm": "attn_norm.weight", "self_attn.q_proj": "attn_q.weight",
        "self_attn.k_proj": "attn_k.weight", "self_attn.v_proj": "attn_v.weight",
        "self_attn.o_proj": "attn_output.weight", "self_attn.q_norm": "attn_q_norm.weight",
        "self_attn.k_norm": "attn_k_norm.weight", "self_attn.conv.base": "attn_conv_base",
        "self_attn.conv.proj": "attn_conv_proj.weight", "post_attention_layernorm": "ffn_norm.weight",
        "mlp.gate_proj": "ffn_gate.weight", "mlp.up_proj": "ffn_up.weight",
        "mlp.down_proj": "ffn_down.weight", "mlp.conv.base": "ffn_conv_base",
        "mlp.conv.proj": "ffn_conv_proj.weight",
    }
    return f"blk.{i}." + mapping[rest]


LINEAR_TENSORS = {"fc", "selector.hidden", "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                  "self_attn.o_proj", "self_attn.conv.proj", "mlp.gate_proj", "mlp.up_proj",
                  "mlp.down_proj", "mlp.conv.proj"}
BF16_ALWAYS = {"selector.predecessor", "selector.successor", "self_attn.conv.base", "mlp.conv.base"}
F32_ALWAYS = {"enc_output_norm", "output_norm", "input_layernorm", "self_attn.q_norm",
              "self_attn.k_norm", "post_attention_layernorm"}


def is_linear(container_suffix):
    tail = container_suffix.split(".", 2)[-1] if container_suffix.startswith("layers.") else container_suffix
    return tail in LINEAR_TENSORS or container_suffix in LINEAR_TENSORS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--container", required=True)
    ap.add_argument("--layout", required=True, choices=["bf16", "w4a16", "w4a8", "mxfp4"])
    args = ap.parse_args()

    gguf = GgufFile(args.gguf)
    container = ContainerFile(args.container)
    block_count = container.metadata["dflash2"]["block_count"]

    names = DFLASH2_TENSORS + layer_tensors(block_count)
    n_checked = n_ok = n_fail = n_skipped_norm_dtype = 0

    for name in names:
        suffix = name
        gname = gguf_name_for(suffix, block_count)
        tail = suffix.split(".", 2)[-1] if suffix.startswith("layers.") else suffix

        if tail in F32_ALWAYS:
            got = container.bytes(f"dflash.{name}")
            want = gguf.dequant_f32(gname).astype(F32).tobytes()
            n_checked += 1
            ok = got == want
            n_ok += ok
            n_fail += not ok
            print(f"{'OK  ' if ok else 'FAIL'} f32   {name} ({len(want)} bytes)")
            continue

        if tail in BF16_ALWAYS:
            got = container.bytes(f"dflash.{name}")
            want = gguf.dequant_bf16_bytes(gname).tobytes()
            n_checked += 1
            ok = got == want
            n_ok += ok
            n_fail += not ok
            print(f"{'OK  ' if ok else 'FAIL'} bf16  {name} ({len(want)} bytes, bit-exact vs GGUF dequant)")
            continue

        # linear (GEMM) tensor
        w = gguf.dequant_f32(gname)  # [N, K]
        N, K = w.shape
        if args.layout == "bf16":
            got = container.bytes(f"dflash.{name}.bf16.w")
            want = gguf.dequant_bf16_bytes(gname).tobytes()
            n_checked += 1
            ok = got == want
            n_ok += ok
            n_fail += not ok
            print(f"{'OK  ' if ok else 'FAIL'} bf16  {name} [{N},{K}] ({len(want)} bytes, bit-exact)")
            continue

        # Quantized layouts: first-tile spot check (see file header comment). The slice width MUST
        # equal the real quantization group size (w4a16/w4a8: 128, mxfp4: 32) -- quantization scale
        # is computed per (row, full group), so a narrower slice would compute a DIFFERENT scale
        # than the real converter did for that row's actual group 0, breaking byte-exactness for a
        # reason that has nothing to do with a real bug. A slice exactly one group wide starting at
        # column 0 reproduces the real row's own group-0 scale exactly (same float values, same
        # group boundary), and PackW4Nibbles/PackMxfp4Wq's tile-major loop order guarantees tile
        # (t=0, kb=0[, kb=1]) is emitted first and is independent of how many tiles/k-blocks the
        # full tensor has -- so this slice's packed bytes are bit-identical to the real container's
        # own first bytes for this tensor.
        n_checked += 1
        if args.layout == "w4a16":
            tile = w[0:16, 0:128]
            q, sc, zero = w4_ref.quantize_asymmetric(tile)  # default group=128
            want_wq = w4_ref.pack_nibbles(q, 16, 128).tobytes()
            got_wq = container.bytes(f"dflash.{name}.w4a16.wq")[0:len(want_wq)]
            ok = got_wq == want_wq
        elif args.layout == "w4a8":
            tile = w[0:16, 0:128]
            q, sc = w4_ref.quantize_symmetric_pinned8(tile)  # default group=128
            want_wq = w4_ref.pack_nibbles(q, 16, 128).tobytes()
            got_wq = container.bytes(f"dflash.{name}.w4a8.wq")[0:len(want_wq)]
            ok = got_wq == want_wq
        else:  # mxfp4
            tile = w[0:16, 0:32]
            packed, escale, wref = mxfp4_ref.quantize(tile)  # default group=32
            want_wq = mxfp4_ref.permute_wq(packed, 16, 32).tobytes()
            got_wq = container.bytes(f"dflash.{name}.mxfp4.wq")[0:len(want_wq)]
            ok = got_wq == want_wq
        n_ok += ok
        n_fail += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {args.layout:5s} {name} [{N},{K}] (first-tile spot check, "
              f"{len(want_wq)} bytes)")

    print(f"\n{args.layout}: {n_ok}/{n_checked} tensors OK, {n_fail} FAIL "
          f"(container={args.container})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
