"""Drives `r4dx-convert --selftest` on a random bf16 weight and diffs its output byte-for-byte
against the Python references (tools/convert_ref/w4_ref.py, mxfp4_ref.py) -- the task item 3 gate,
run through the actual CLI rather than the CTest-internal comparison (tests/convert/
test_pack_bytes.cpp exercises the same headers in-process; this script is the end-to-end version
against the built r4dx-convert.exe, on a freshly generated random input each run).

Covers all three ways r4dx-convert can choose its quantized values, on the same input tensor:
  * `--quant rtn`                      -- the historical min/max + round-to-nearest grid
  * `--quant search`                   -- error-minimizing search, unweighted MSE
  * `--quant search --imatrix <npz>`   -- the same search weighted by a synthetic importance
                                          vector this script generates and saves with numpy.savez
                                          (keyed "selftest", the selftest tensor's container base)
The byte LAYOUT is identical in all three; only the values differ, which is exactly what makes a
byte-for-byte diff the right gate for the search.

Usage (from the reference venv):
  C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe selftest_compare.py
      [--exe <path to r4dx-convert.exe>] [--n 48] [--k 384] [--seed 7]
"""
import argparse
import json
import pathlib
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import mxfp4_ref
import w4_ref
from gen_fixtures import bf16_u16_to_float, float_to_bf16_u16, write_bf16_safetensors

# Resolved relative to this file's location in the repo, so the path holds for any checkout.
DEFAULT_EXE = str(
    pathlib.Path(__file__).resolve().parents[2] / "build" / "win-hip" / "src" / "convert" / "r4dx-convert.exe"
)


class ContainerReader:
    """Minimal reader for r4dx's safetensors-shaped container (docs/container-format.md): every
    tensor is dtype U8, so this only needs the header offsets, not a dtype table."""

    def __init__(self, path: str):
        with open(path, "rb") as f:
            (n,) = struct.unpack("<Q", f.read(8))
            header = json.loads(f.read(n))
            self._data_start = 8 + n
            self._path = path
            self._header = header

    def __getitem__(self, name: str) -> bytes:
        entry = self._header[name]
        b, e = entry["data_offsets"]
        with open(self._path, "rb") as f:
            f.seek(self._data_start + b)
            return f.read(e - b)

    def names(self):
        return [k for k in self._header if k != "__metadata__"]


def compare(label: str, got: bytes, want: bytes) -> bool:
    if got == want:
        print(f"OK   {label} ({len(got)} bytes byte-exact)")
        return True
    n = min(len(got), len(want))
    first = next((i for i in range(n) if got[i] != want[i]), n)
    print(f"FAIL {label}: len got={len(got)} want={len(want)}, first diff at byte {first}"
          f" (got=0x{got[first] if first < len(got) else -1:02x}"
          f" want=0x{want[first] if first < len(want) else -1:02x})")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exe", default=DEFAULT_EXE)
    ap.add_argument("--n", type=int, default=48)
    ap.add_argument("--k", type=int, default=384)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    if not pathlib.Path(args.exe).exists():
        print(f"selftest_compare: r4dx-convert.exe not found at {args.exe} -- build it first "
              f"(cmake --build --preset win-hip --target r4dx-convert)", file=sys.stderr)
        return 2

    rng = np.random.default_rng(args.seed)
    raw = rng.normal(0.0, 1.0, size=(args.n, args.k)).astype(np.float32)
    bf16_u16 = float_to_bf16_u16(raw).reshape(args.n, args.k)
    w = bf16_u16_to_float(bf16_u16)

    # Synthetic importance vector: lognormal, so it carries the heavy per-channel tail a real
    # imatrix has (tools/reference/imatrix_capture.py measured max/median ratios of 80x-90000x) and
    # actually moves the search's choices, rather than being a near-uniform no-op.
    imat = rng.lognormal(0.0, 2.0, size=args.k).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        in_path = str(pathlib.Path(tmp) / "selftest_input.safetensors")
        imat_path = str(pathlib.Path(tmp) / "selftest_imatrix.npz")
        write_bf16_safetensors(in_path, "w", bf16_u16, (args.n, args.k))
        # np.savez writes ZIP_STORED members, which is what r4dx_convert::NpzReader accepts.
        np.savez(imat_path, **{"selftest": imat})

        ok = True
        for label, extra_args, expect in (
            ("--quant rtn", ["--quant", "rtn"], None),
            ("--quant search", ["--quant", "search"], "none"),
            ("--quant search --imatrix", ["--quant", "search", "--imatrix", imat_path], "imat"),
        ):
            print(f"\n=== {label} ===")
            out_path = str(pathlib.Path(tmp) / f"selftest_output_{len(extra_args)}_{expect}.r4dx")
            proc = subprocess.run(
                [args.exe, "--selftest", "--selftest-input", in_path,
                 "--selftest-output", out_path, "--layouts", "mxfp4,w4a16,w4a8", "--threads", "4"]
                + extra_args,
                capture_output=True, text=True)
            print(proc.stdout, end="")
            if proc.returncode != 0:
                print(proc.stderr, file=sys.stderr)
                return 1

            container = ContainerReader(out_path)
            if expect is None:
                q16, sc16, z16 = w4_ref.quantize_asymmetric(w)
                q8, sc8 = w4_ref.quantize_symmetric_pinned8(w)
                packed, escale, wref = mxfp4_ref.quantize(w)
            else:
                im = imat if expect == "imat" else None
                q16, sc16, z16 = w4_ref.quantize_asymmetric_search(w, imatrix=im)
                q8, sc8 = w4_ref.quantize_symmetric_pinned8_search(w, imatrix=im)
                packed, escale, wref = mxfp4_ref.quantize_search(w, imatrix=im)

            ok &= compare("w4a16.wq", container["selftest.w4a16.wq"],
                          w4_ref.pack_nibbles(q16, args.n, args.k).tobytes())
            ok &= compare("w4a16.wsz", container["selftest.w4a16.wsz"],
                          w4_ref.pack_w4a16_scales(sc16, z16, args.n, args.k).tobytes())
            ok &= compare("w4a8.wq", container["selftest.w4a8.wq"],
                          w4_ref.pack_nibbles(q8, args.n, args.k).tobytes())
            ok &= compare("w4a8.ws", container["selftest.w4a8.ws"],
                          w4_ref.pack_w4a8_scales(sc8, args.n, args.k).tobytes())
            ok &= compare("mxfp4.wq", container["selftest.mxfp4.wq"],
                          mxfp4_ref.permute_wq(packed, args.n, args.k).tobytes())
            ok &= compare("mxfp4.ws", container["selftest.mxfp4.ws"],
                          mxfp4_ref.pack_ws(escale, args.n, args.k).tobytes())
            ok &= compare("mxfp4.wref", container["selftest.mxfp4.wref"], wref.tobytes())

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
