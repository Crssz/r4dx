#!/usr/bin/env python3
"""Decode bytes of .r4dx containers: what one decode step streams, measured from the file.

The definition is docs/quant2.md 7 ("decode bytes"): every text-layer weight plus lm_head -- the
embedding is a row gather, and the MTP head, the draft head and vision are not part of a plain decode
step. Each linear counts ONE layout, the one the runtime loads at the given --layout
(src/model/container.cpp): the body layout's tensors when the linear has them, else its bf16
fallback (`.bf16.w`, a --keep-bf16 base). lm_head follows the body layout too, except that a trellis
body reads a w4a16 lm_head (docs/trellis-kernel.md 5.1 maps the trellis head layouts to w4a16).
Everything else under text.layers.* (norms, conv1d, A_log, dt_bias, descales, gdn.in_proj_a/b, a
pre-R1 container's bare attn.k/v) counts as it is. `<base>.trellis.w|suh|svh` are one layout: the
words and both scale vectors (docs/trellis-kernel.md 2.1, 2.5 "Bytes").

  python tools/quant2/decode_bytes.py <models root>\\r4dx\\huihui-qwen38-27b-abl-trellis-mix45m.r4dx `
      [<another container>.r4dx ...] [--layout auto|w4a16|trellis|bf16] [--by-class] [--json-out f]

`--layout auto` (default) is trellis for a container with __metadata__.quant.trellis, else w4a16.
Python stdlib only.
"""
import argparse
import json
import re
import struct
import sys

GIB = 1 << 30
LAYOUT_RE = re.compile(r"^(?P<base>.+?)\.(?P<layout>w4a16|w4a8|mxfp4|bf16|trellis)\.(?P<rest>.+)$")


def read_header(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        if n == 0 or n > (1 << 30):
            raise SystemExit(f"{path}: implausible header length {n}")
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", None) or {}
    return header, meta


def decode_bytes(path, layout="auto"):
    header, meta = read_header(path)
    has_trellis = isinstance(meta.get("quant"), dict) and "trellis" in meta["quant"]
    body = layout if layout != "auto" else ("trellis" if has_trellis else "w4a16")
    head = "w4a16" if body == "trellis" else body
    linears = {}  # base -> layout -> bytes
    other = {}
    for name, t in header.items():
        if not (name.startswith("text.layers.") or name == "lm_head" or name.startswith("lm_head.")):
            continue
        b = t["data_offsets"][1] - t["data_offsets"][0]
        m = LAYOUT_RE.match(name)
        if m:
            linears.setdefault(m.group("base"), {}).setdefault(m.group("layout"), 0)
            linears[m.group("base")][m.group("layout")] += b
        else:
            other[name] = b
    total, by_class, picked, trellis_bits = 0, {}, {}, {}
    trellis_lin = (meta.get("quant", {}).get("trellis", {}) or {}).get("linears", {}) if has_trellis else {}
    for base, lay in linears.items():
        want = head if base == "lm_head" else body
        use = want if want in lay else ("bf16" if "bf16" in lay else None)
        if use is None:
            raise SystemExit(f"{path}: linear {base} has neither {want} nor bf16 ({sorted(lay)}) -- the runtime "
                             f"could not load it at --layout {body}")
        picked[use] = picked.get(use, 0) + 1
        total += lay[use]
        cls = "lm_head" if base == "lm_head" else re.sub(r"^text\.layers\.\d+\.", "", base)
        by_class[cls] = by_class.get(cls, 0) + lay[use]
        if use == "trellis":
            kb = trellis_lin.get(base, {}).get("bits")
            trellis_bits[str(kb)] = trellis_bits.get(str(kb), 0) + 1
    for name, b in other.items():
        total += b
        cls = "other (" + re.sub(r"^text\.layers\.\d+\.", "", name).split(".")[0] + ".*)"
        by_class[cls] = by_class.get(cls, 0) + b
    return {"path": path, "layout": body, "head_layout": head, "bytes": total, "gib": total / GIB,
            "linears_by_layout": picked, "trellis_linears_by_bits": trellis_bits, "by_class": by_class}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("containers", nargs="+")
    ap.add_argument("--layout", default="auto", choices=["auto", "w4a16", "trellis", "bf16"])
    ap.add_argument("--by-class", action="store_true", help="print the per-class breakdown")
    ap.add_argument("--json-out")
    args = ap.parse_args()
    results = []
    for p in args.containers:
        r = decode_bytes(p, args.layout)
        results.append(r)
        tb = "".join(f", trellis KB={k} x{v}" for k, v in sorted(r["trellis_linears_by_bits"].items()))
        print(f"{p}: decode bytes {r['bytes']} = {r['gib']:.3f} GiB at --layout {r['layout']} "
              f"(lm_head {r['head_layout']}; linears by layout {r['linears_by_layout']}{tb})")
        if args.by_class:
            for k in sorted(r["by_class"], key=lambda k: -r["by_class"][k]):
                print(f"    {k:40s} {r['by_class'][k] / GIB:8.4f} GiB")
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
