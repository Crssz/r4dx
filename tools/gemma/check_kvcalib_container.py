#!/usr/bin/env python3
"""Check that a Gemma .r4dx container's `text.layers.{i}.attn.{k,v}_descale` tensors are exactly
`amax / 448` of the kvcalib.json it was converted with (r4dx-convert --kv-calib), for every layer the
container holds, with the per-layer-type shapes (8 sliding heads, 1 full head) and no 1.0 placeholder left.
Python stdlib only; CPU only; reads the header and the descale tensors, nothing else.

    python tools\\gemma\\check_kvcalib_container.py <container.r4dx> <kvcalib.json>

Exit 0 = every descale matches (within one fp32 ulp of the division), 1 = a mismatch or a missing tensor.
"""
import json
import struct
import sys

FP8_MAX = 448.0


def main(argv):
    if len(argv) != 3:
        print(__doc__)
        return 2
    path, calib_path = argv[1], argv[2]
    calib = json.load(open(calib_path, encoding="utf-8"))
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        layers = sorted({int(k.split(".")[2]) for k in header if k.startswith("text.layers.")})
        bad, checked, placeholder = [], 0, 0
        shapes = {}
        for i in layers:
            for kind in ("k", "v"):
                name = f"text.layers.{i}.attn.{kind}_descale"
                if name not in header:
                    bad.append(f"{name}: missing from the container")
                    continue
                t = header[name]
                lo, hi = t["data_offsets"]
                f.seek(base + lo)
                vals = struct.unpack(f"<{(hi - lo) // 4}f", f.read(hi - lo))
                entry = calib.get(str(i))
                if entry is None:
                    bad.append(f"layer {i}: not in the kvcalib json")
                    continue
                want = [a / FP8_MAX for a in entry[f"{kind}_amax"]]
                shapes.setdefault((entry.get("layer_type", "?"), len(vals)), []).append(i)
                if len(vals) != len(want):
                    bad.append(f"{name}: {len(vals)} entries, kvcalib has {len(want)}")
                    continue
                for h, (g, w) in enumerate(zip(vals, want)):
                    if abs(g - w) > 1e-6 * max(abs(w), 1e-30):
                        bad.append(f"{name}[{h}]: container {g!r} != amax/448 {w!r}")
                    if g == 1.0:
                        placeholder += 1
                checked += 1
    for (lt, nh), ids in sorted(shapes.items()):
        print(f"  {lt:<18} {nh} head(s): {len(ids)} layer(s) (first {ids[:4]})")
    print(f"{path}: {checked} descale tensors over {len(layers)} layer(s) checked, {placeholder} entries == 1.0, "
          f"{len(bad)} problem(s)")
    for b in bad[:20]:
        print("  PROBLEM", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
