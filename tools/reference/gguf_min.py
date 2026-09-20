"""Minimal, dependency-free (numpy/struct only) GGUF reader + Q8_0 dequantizer.

Deliberately does NOT import `gguf-py` (the ROCmFPX checkout's own Python package) -- the task
that produced this file requires an independent parser so the reference implementation isn't
trusting the same code path the reference *inference engine* uses. Covers exactly the subset of
the GGUF v3 format DFlash2 containers need: scalar/array metadata (uint8/int8/uint16/int16/
uint32/int32/float32/bool/string/uint64/int64/float64/array-of-any-of-those) and tensor data in
F32 or Q8_0.

GGUF container layout (little-endian throughout):

    uint32  magic       'GGUF' (0x46554747)
    uint32  version      (3 for every file this reader has seen)
    int64   tensor_count
    int64   metadata_kv_count
    <metadata_kv_count> key/value pairs:
        gguf_string  key
        uint32       value_type   (GGUFValueType)
        <value>                    (scalar, string, or array -- arrays carry their own
                                     element-type uint32 + int64 count prefix)
    <tensor_count> tensor infos:
        gguf_string  name
        uint32       n_dims
        uint64[n_dims] ne          ne[0] is the FASTEST-varying (contiguous) dimension --
                                    the opposite of numpy/C row-major axis order. A 2D weight
                                    W with logical shape [out, in] (row-major, as GEMM expects
                                    it) is stored with ne = [in, out].
        uint32       dtype         (GGML type enum; this reader only decodes F32=0 and Q8_0=8)
        uint64       offset        byte offset into the (alignment-padded) tensor data block
    <padding to general.alignment (default 32)>
    <tensor data, back to back at the recorded offsets>

`gguf_string` is `uint64 len` followed by `len` raw (not NUL-terminated) UTF-8 bytes.

Q8_0 block (llama.cpp ggml-quants convention, QK8_0=32 elements/block):
    float16  d       per-block scale
    int8[32] qs      per-block quantized values
    dequant: value[i] = d * qs[i]

Every read here targets the ~2 GB DFlash2 container end to end (`--gguf-info` below); nothing
streams partial tensors, since a 27B-parameter *target* checkpoint is handled separately by
`tools/reference/dflash2_ref.py`'s safetensors path, not by this module. Concretely:
`load_gguf` reads the whole file into RAM (`Path.read_bytes()`, ~1.9 GB for the real DFlash2 GGUF)
and `get_array`/`_dequant_q8_0` allocate a fresh float32 copy per tensor on top of that (the
largest, `fc.weight`, is ~139 MB packed / ~530 MB as float32) -- peak RSS for a full read-through is
in the ~3-4 GB range, not "streaming". Fine for this ~2 GB draft container on a normal dev machine;
would need an `np.memmap`-backed rewrite before pointing this module at the ~27B *target*
checkpoint instead.
"""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

GGUF_MAGIC = 0x46554747
GGUF_SUPPORTED_VERSIONS = (2, 3)

# GGUFValueType
_T_UINT8, _T_INT8, _T_UINT16, _T_INT16, _T_UINT32, _T_INT32, _T_FLOAT32, _T_BOOL, \
    _T_STRING, _T_ARRAY, _T_UINT64, _T_INT64, _T_FLOAT64 = range(13)

_SCALAR_STRUCT = {
    _T_UINT8: "<B", _T_INT8: "<b", _T_UINT16: "<H", _T_INT16: "<h",
    _T_UINT32: "<I", _T_INT32: "<i", _T_FLOAT32: "<f", _T_BOOL: "<?",
    _T_UINT64: "<Q", _T_INT64: "<q", _T_FLOAT64: "<d",
}

# GGML tensor dtype enum values this reader knows how to dequantize.
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8

QK8_0 = 32
_Q8_0_BLOCK_BYTES = 2 + QK8_0  # fp16 scale + 32 int8 values


class GGUFError(RuntimeError):
    pass


@dataclass
class TensorInfo:
    name: str
    ne: tuple[int, ...]  # ne[0] fastest-varying, as stored in the file
    dtype: int
    offset: int  # byte offset from the start of the tensor-data block

    @property
    def numpy_shape(self) -> tuple[int, ...]:
        """Row-major numpy shape: reverse of `ne` (ne[0] becomes the LAST/fastest numpy axis)."""
        return tuple(reversed(self.ne))

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.ne:
            n *= d
        return n


class _Reader:
    __slots__ = ("buf", "pos")

    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def bytes(self, n: int) -> bytes:
        b = self.buf[self.pos : self.pos + n]
        if len(b) != n:
            raise GGUFError(f"truncated GGUF file at offset {self.pos} (wanted {n} bytes)")
        self.pos += n
        return b

    def u32(self) -> int:
        return struct.unpack_from("<I", self.bytes(4))[0]

    def u64(self) -> int:
        return struct.unpack_from("<Q", self.bytes(8))[0]

    def i64(self) -> int:
        return struct.unpack_from("<q", self.bytes(8))[0]

    def string(self) -> str:
        n = self.u64()
        return self.bytes(n).decode("utf-8")

    def scalar(self, vtype: int) -> Any:
        if vtype == _T_STRING:
            return self.string()
        fmt = _SCALAR_STRUCT.get(vtype)
        if fmt is None:
            raise GGUFError(f"unsupported GGUF scalar value type {vtype}")
        size = struct.calcsize(fmt)
        return struct.unpack_from(fmt, self.bytes(size))[0]

    def value(self, vtype: int) -> Any:
        if vtype == _T_ARRAY:
            elem_type = self.u32()
            count = self.i64()
            return [self.value(elem_type) for _ in range(count)]
        return self.scalar(vtype)


@dataclass
class GGUFFile:
    path: Path
    version: int
    metadata: dict[str, Any]
    tensors: dict[str, TensorInfo]
    _data: bytes  # the full file content (tensor payloads read as views into this)
    _data_start: int  # byte offset of the tensor-data block within `_data`

    # ---- metadata helpers -------------------------------------------------

    def meta(self, key: str, default: Any = None) -> Any:
        return self.metadata.get(key, default)

    def require_meta(self, key: str) -> Any:
        if key not in self.metadata:
            raise GGUFError(f"required GGUF metadata key {key!r} is missing from {self.path}")
        return self.metadata[key]

    # ---- tensor helpers -----------------------------------------------------

    def tensor_names(self) -> list[str]:
        return list(self.tensors.keys())

    def has_tensor(self, name: str) -> bool:
        return name in self.tensors

    def get_array(self, name: str) -> np.ndarray:
        """Dequantized tensor as float32, numpy row-major shape (see `TensorInfo.numpy_shape`)."""
        info = self.tensors.get(name)
        if info is None:
            raise GGUFError(f"tensor {name!r} not found in {self.path}")

        n_elem = info.n_elements
        start = self._data_start + info.offset

        if info.dtype == GGML_TYPE_F32:
            raw = self._data[start : start + 4 * n_elem]
            arr = np.frombuffer(raw, dtype="<f4", count=n_elem)
        elif info.dtype == GGML_TYPE_F16:
            raw = self._data[start : start + 2 * n_elem]
            arr = np.frombuffer(raw, dtype="<f2", count=n_elem).astype(np.float32)
        elif info.dtype == GGML_TYPE_Q8_0:
            arr = _dequant_q8_0(self._data, start, n_elem)
        else:
            raise GGUFError(
                f"tensor {name!r} has dtype {info.dtype}, which this minimal reader does not "
                "decode (only F32/F16/Q8_0 are implemented -- add a case here if a DFlash2 "
                "container ever ships a different weight dtype)"
            )
        return arr.reshape(info.numpy_shape).astype(np.float32, copy=False)


def _dequant_q8_0(data: bytes, start: int, n_elem: int) -> np.ndarray:
    if n_elem % QK8_0 != 0:
        raise GGUFError(f"Q8_0 tensor element count {n_elem} is not a multiple of {QK8_0}")
    n_blocks = n_elem // QK8_0
    raw = data[start : start + n_blocks * _Q8_0_BLOCK_BYTES]
    block_dtype = np.dtype([("d", "<f2"), ("qs", "<i1", (QK8_0,))])
    blocks = np.frombuffer(raw, dtype=block_dtype, count=n_blocks)
    scales = blocks["d"].astype(np.float32)  # [n_blocks]
    qs = blocks["qs"].astype(np.float32)  # [n_blocks, QK8_0]
    out = qs * scales[:, None]
    return out.reshape(n_elem)


def load_gguf(path: str | Path) -> GGUFFile:
    path = Path(path)
    data = path.read_bytes()
    r = _Reader(data)

    magic = r.u32()
    if magic != GGUF_MAGIC:
        raise GGUFError(f"{path}: bad magic 0x{magic:08x}, expected GGUF (0x{GGUF_MAGIC:08x})")

    version = r.u32()
    if version not in GGUF_SUPPORTED_VERSIONS:
        raise GGUFError(f"{path}: unsupported GGUF version {version}")

    tensor_count = r.i64()
    kv_count = r.i64()

    metadata: dict[str, Any] = {}
    for _ in range(kv_count):
        key = r.string()
        vtype = r.u32()
        metadata[key] = r.value(vtype)

    tensors: dict[str, TensorInfo] = {}
    for _ in range(tensor_count):
        name = r.string()
        n_dims = r.u32()
        ne = tuple(r.u64() for _ in range(n_dims))
        dtype = r.u32()
        offset = r.u64()
        tensors[name] = TensorInfo(name=name, ne=ne, dtype=dtype, offset=offset)

    alignment = int(metadata.get("general.alignment", 32))
    data_start = r.pos
    if alignment > 0 and data_start % alignment != 0:
        data_start += alignment - (data_start % alignment)

    return GGUFFile(
        path=path,
        version=version,
        metadata=metadata,
        tensors=tensors,
        _data=data,
        _data_start=data_start,
    )


def _main() -> None:  # pragma: no cover -- manual inspection helper, not part of any test
    if len(sys.argv) != 2:
        print("usage: gguf_min.py <path.gguf>", file=sys.stderr)
        raise SystemExit(2)
    g = load_gguf(sys.argv[1])
    print(f"version={g.version}  tensors={len(g.tensors)}  metadata_keys={len(g.metadata)}")
    for k, v in sorted(g.metadata.items()):
        v_repr = v if not isinstance(v, list) or len(v) <= 8 else f"{v[:8]}... ({len(v)} items)"
        print(f"  {k} = {v_repr}")
    print("tensors:")
    for name, info in sorted(g.tensors.items()):
        print(f"  {name:40s} ne={info.ne} dtype={info.dtype} shape(numpy)={info.numpy_shape}")


if __name__ == "__main__":  # pragma: no cover
    _main()
