"""Generates tests/convert/fixtures/: a small random bf16 weight (as a one-tensor .safetensors
file r4dx-convert's --selftest / the CTest packer test can both read) plus every reference
quantized/packed byte blob (w4a16), computed by the Python references in this
directory. tests/convert/test_pack_bytes.cpp loads input.safetensors, runs the C++ quantizer/
packer on it, and memcmp's the result against these files -- byte-exactness is the gate (task item
3 / item 5).

The weight is generated as float32 THEN rounded to bf16 and read back before it is quantized, so
whatever bytes land in input.safetensors are exactly the float values both the C++ and the Python
side quantize -- no bf16-rounding discrepancy between "the value I quantized" and "the value the
C++ side reads back out of the file" to worry about.

Run with the read-only reference venv:
  python gen_fixtures.py
"""
import json
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
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

    N, K = 32, 256  # N multiple of 16 (2 row-tiles); K multiple of 64 and 32 (the int4 groups)
    rng = np.random.default_rng(1234)
    raw = rng.normal(0.0, 1.0, size=(N, K)).astype(np.float32)
    bf16_u16 = float_to_bf16_u16(raw).reshape(N, K)
    w = bf16_u16_to_float(bf16_u16)

    write_bf16_safetensors(str(out_dir / "input.safetensors"), "w", bf16_u16, (N, K))

    # The int4 fixtures exist at the default w4a16 group, 64 (kW4A16Group); each group g gets a
    # "_g{g}" suffix.
    int4_groups = [64]
    manifest = {"N": N, "K": K, "int4_group": 64, "int4_groups": int4_groups}
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    def gsfx(group: int) -> str:
        return f"_g{group}"

    for group in int4_groups:
        g = gsfx(group)
        q16, sc16, z16 = w4_ref.quantize_asymmetric(w, group=group)
        w4_ref.pack_nibbles(q16, N, K).tofile(str(out_dir / f"w4a16_wq{g}.bin"))
        w4_ref.pack_w4a16_scales(sc16, z16, N, K, group=group).tofile(
            str(out_dir / f"w4a16_wsz{g}.bin"))

    # ---- --quant search fixtures (tests/convert/test_quant_search.cpp) ------------------------
    # Same input tensor, both weightings. `search_imatrix.bin` is the float32[K] importance vector
    # the C++ test feeds to the search -- lognormal, so it has the heavy per-channel tail a real
    # tools/reference/imatrix_capture.py vector has and actually changes the chosen values.
    imat = rng.lognormal(0.0, 2.0, size=K).astype(np.float32)
    imat.tofile(str(out_dir / "search_imatrix.bin"))

    for suffix, im in (("", None), ("_imat", imat)):
        for group in int4_groups:
            g = gsfx(group)
            q16s, sc16s, z16s = w4_ref.quantize_asymmetric_search(w, group=group, imatrix=im)
            w4_ref.pack_nibbles(q16s, N, K).tofile(
                str(out_dir / f"search{suffix}_w4a16_wq{g}.bin"))
            w4_ref.pack_w4a16_scales(sc16s, z16s, N, K, group=group).tofile(
                str(out_dir / f"search{suffix}_w4a16_wsz{g}.bin"))

    print("fixtures written to", out_dir)


if __name__ == "__main__":
    main()
