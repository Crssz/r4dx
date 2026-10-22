#!/usr/bin/env python3
"""Check that a qwen3_5_moe .r4dx container's `text.layers.{i}.attn.{k,v}_descale` tensors (fp32 [kv_heads], full-attention layers only)
are exactly `amax / 448` of the kvcalib.json it was converted with (r4dx-convert --kv-calib), or - with no json - report them
(1.0 everywhere = the placeholder container). Python stdlib only; CPU only; reads the header and the descale tensors.

    python tools\\quant2\\qwen35moe\\check_kvcalib_q35.py <container.r4dx> [kvcalib.json]

Exit 0 = every descale matches (within 1e-6 relative), 1 = a mismatch or a missing tensor. (tools/gemma/check_kvcalib_container.py
is the Gemma version: it expects descales on every layer.)
"""
import json
import struct
import sys

FP8_MAX = 448.0


def descales(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, t in header.items():
            if name.startswith("text.layers.") and name.endswith(("attn.k_descale", "attn.v_descale")):
                lo, hi = t["data_offsets"]
                f.seek(base + lo)
                out[name] = struct.unpack(f"<{(hi - lo) // 4}f", f.read(hi - lo))
        return out


def main(argv):
    if len(argv) not in (2, 3):
        print(__doc__)
        return 2
    path = argv[1]
    got = descales(path)
    layers = sorted({int(k.split(".")[2]) for k in got})
    if len(argv) == 2:
        for i in layers:
            print(f"layer {i}: k_descale {list(got[f'text.layers.{i}.attn.k_descale'])} v_descale {list(got[f'text.layers.{i}.attn.v_descale'])}")
        return 0
    calib = json.load(open(argv[2], encoding="utf-8"))
    bad, placeholder = [], 0
    if layers != sorted(int(k) for k in calib):
        bad.append(f"container has descales on layers {layers}, the json on {sorted(int(k) for k in calib)}")
    for i in layers:
        entry = calib.get(str(i))
        for kind in ("k", "v"):
            name = f"text.layers.{i}.attn.{kind}_descale"
            if entry is None or name not in got:
                continue
            vals, want = got[name], [a / FP8_MAX for a in entry[f"{kind}_amax"]]
            if len(vals) != len(want):
                bad.append(f"{name}: {len(vals)} entries, kvcalib has {len(want)}")
                continue
            for h, (g, w) in enumerate(zip(vals, want)):
                if abs(g - w) > 1e-6 * max(abs(w), 1e-30):
                    bad.append(f"{name}[{h}]: container {g!r} != amax/448 {w!r}")
                placeholder += g == 1.0
    print(f"{path}: {len(got)} descale tensors on {len(layers)} full-attention layers {layers[:3]}..., {placeholder} entries == 1.0, "
          f"{len(bad)} problem(s)")
    for b in bad[:20]:
        print("  PROBLEM", b)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
