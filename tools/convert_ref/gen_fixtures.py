"""Generates tests/convert/fixtures/: a small random bf16 weight (as a one-tensor .safetensors
file r4dx-convert's --selftest / the CTest packer test can both read) plus every reference
quantized/packed byte blob (w4a16, w4a8, mxfp4), computed by the Python references in this
directory. tests/convert/test_pack_bytes.cpp loads input.safetensors, runs the C++ quantizer/
packer on it, and memcmp's the result against these files -- byte-exactness is the gate (task item
3 / item 5).

The weight is generated as float32 THEN rounded to bf16 and read back before it is quantized, so
whatever bytes land in input.safetensors are exactly the float values both the C++ and the Python
side quantize -- no bf16-rounding discrepancy between "the value I quantized" and "the value the
C++ side reads back out of the file" to worry about.

Run with the read-only reference venv:
  C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe gen_fixtures.py
"""
import json
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mxfp4_ref
import w4_ref


def float_to_bf16_u16(x: np.ndarray) -> np.ndarray:
    """Matches r4dx::core::FloatToBf16 (src/core/include/r4dx/core/dtype.hpp): round-to-nearest-
    even truncation, NaN-preserving."""
    x = x.astype(np.float32)
    bits = x.view(np.uint32)
    is_nan = (bits & 0x7fffffff) > 0x7f800000
    rounded = (bits + np.uint32(0x7fff) + ((bits >> 16) & np.uint32(1))).astype(np.uint32)
    bf16 = (rounded >> 16).astype(np.uint16)
    nan_bf16 = ((bits >> 16) | np.uint32(0x0040)).astype(np.uint16)
    return np.where(is_nan, nan_bf16, bf16).astype(np.uint16)


def bf16_u16_to_float(h: np.ndarray) -> np.ndarray:
    bits = h.astype(np.uint32) << 16
    return bits.view(np.float32)


def write_bf16_safetensors(path: str, name: str, bf16_u16: np.ndarray, shape) -> None:
    data = bf16_u16.astype("<u2").tobytes()
    header = {name: {"dtype": "BF16", "shape": list(shape), "data_offsets": [0, len(data)]}}
    hbytes = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        f.write(data)


def main() -> None:
    out_dir = pathlib.Path(__file__).resolve().parents[2] / "tests" / "convert" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    N, K = 32, 256  # N multiple of 16 (2 row-tiles), K multiple of 128 and 32 (int4 + mxfp4 groups)
    rng = np.random.default_rng(1234)
    raw = rng.normal(0.0, 1.0, size=(N, K)).astype(np.float32)
    bf16_u16 = float_to_bf16_u16(raw).reshape(N, K)
    w = bf16_u16_to_float(bf16_u16)

    write_bf16_safetensors(str(out_dir / "input.safetensors"), "w", bf16_u16, (N, K))

    manifest = {"N": N, "K": K, "int4_group": w4_ref.GROUP, "mxfp4_group": mxfp4_ref.GROUP}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    q16, sc16, z16 = w4_ref.quantize_asymmetric(w)
    w4_ref.pack_nibbles(q16, N, K).tofile(str(out_dir / "w4a16_wq.bin"))
    w4_ref.pack_w4a16_scales(sc16, z16, N, K).tofile(str(out_dir / "w4a16_wsz.bin"))

    q8, sc8 = w4_ref.quantize_symmetric_pinned8(w)
    w4_ref.pack_nibbles(q8, N, K).tofile(str(out_dir / "w4a8_wq.bin"))
    w4_ref.pack_w4a8_scales(sc8, N, K).tofile(str(out_dir / "w4a8_ws.bin"))

    packed, escale, wref = mxfp4_ref.quantize(w)
    mxfp4_ref.permute_wq(packed, N, K).tofile(str(out_dir / "mxfp4_wq.bin"))
    mxfp4_ref.pack_ws(escale, N, K).tofile(str(out_dir / "mxfp4_ws.bin"))
    wref.tofile(str(out_dir / "mxfp4_wref.bin"))

    print("fixtures written to", out_dir)


if __name__ == "__main__":
    main()
