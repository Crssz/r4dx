"""tools/prefill/probe_depth.py -- per-chunk prefill time vs context depth, from the in-model probe's
timeline (src/model/debug_probe.h: R4DX_PROFILE_LINEARS=all + R4DX_PROBE_TIMELINE=<csv>).

Each prefill RunChunk is one "call" (64 rows, the last one shorter); the probe writes every
ProfiledCall span of it (name, layer, duration from wall_clock64 stamps) plus one "call" row with
the call's whole GPU span. The k-th prefill call (in serial order) starts at depth 64*k.

Classes:
  linear:<name>  every "gemm:" span (the trellis/w4a16 body linears incl. their input transform)
  gdn            gdn.conv_prep + gdn.kkt_solve + gdn.chunk_scan + gdn.gated_rmsnorm (the chunked GDN)
  attn_core      attn.core_prefill (the paged fp8 prefill attention kernel, r4d_attn_prefill_*)
  other          every other span (norms, rope, kv_write, residuals, silu, gate mul, embed...) plus the
                 call's GPU time outside any span (call - sum of spans: launch gaps, un-spanned kernels)

Usage:
  python tools\\prefill\\probe_depth.py --timeline t.csv --out summary.json [--window 16]
      [--depths 0,8192,32768,65536,122880] [--wall-s <unprofiled prefill seconds>]
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

GDN = {"gdn.conv_prep", "gdn.kkt_solve", "gdn.chunk_scan", "gdn.gated_rmsnorm"}


def classify(name: str) -> str:
    if name.startswith("gemm:"):
        return "linear:" + name[5:]
    if name in GDN:
        return "gdn"
    if name == "attn.core_prefill":
        return "attn_core"
    return "other"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--timeline", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--window", type=int, default=16, help="chunks averaged around each reported depth")
    ap.add_argument("--depths", default="0,8192,32768,65536,122880")
    ap.add_argument("--wall-s", type=float, default=0.0, help="unprofiled prefill seconds (overhead check)")
    args = ap.parse_args(argv)

    calls: dict[int, dict] = {}
    spans: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    span_names: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    with open(args.timeline, newline="") as f:
        for row in csv.reader(f):
            if len(row) < 7 or row[0] != "prefill":
                continue
            kind, T, serial, name, layer, start, dur = row[:7]
            serial = int(serial)
            dur_ms = float(dur) / 1000.0
            if name == "call":
                calls[serial] = {"T": int(T), "gpu_ms": dur_ms}
            else:
                spans[serial][classify(name)] += dur_ms
                span_names[serial][name] += dur_ms
    order = sorted(calls)
    chunks = []
    depth = 0
    for s in order:
        c = calls[s]
        cls = dict(spans[s])
        spanned = sum(cls.values())
        cls["other"] = cls.get("other", 0.0) + max(0.0, c["gpu_ms"] - spanned)
        lin = sum(v for k, v in cls.items() if k.startswith("linear:"))
        chunks.append({"depth": depth, "T": c["T"], "gpu_ms": c["gpu_ms"], "linear": lin,
                       "gdn": cls.get("gdn", 0.0), "attn_core": cls.get("attn_core", 0.0),
                       "other": cls["other"], "classes": cls, "unspanned_ms": c["gpu_ms"] - spanned})
        depth += c["T"]
    if not chunks:
        raise SystemExit("no prefill calls in the timeline")
    total_tokens = depth

    # Depth table.
    table = []
    for d in [int(x) for x in args.depths.split(",")]:
        k0 = min(range(len(chunks)), key=lambda i: abs(chunks[i]["depth"] - d))
        lo, hi = max(0, k0 - args.window // 2), min(len(chunks), k0 + args.window // 2)
        sel = [c for c in chunks[lo:hi] if c["T"] == 64] or chunks[lo:hi]
        n = len(sel)
        agg = defaultdict(float)
        for c in sel:
            for k, v in c["classes"].items():
                agg[k] += v / n
        row = {"depth": d, "depth_range": [sel[0]["depth"], sel[-1]["depth"] + sel[-1]["T"]], "chunks": n,
               "gpu_ms": sum(c["gpu_ms"] for c in sel) / n,
               "linear": sum(v for k, v in agg.items() if k.startswith("linear:")),
               "gdn": agg.get("gdn", 0.0), "attn_core": agg.get("attn_core", 0.0), "other": agg.get("other", 0.0),
               "linear_by_class": {k[7:]: round(v, 4) for k, v in sorted(agg.items()) if k.startswith("linear:")}}
        table.append(row)

    # Cumulative shares for a prompt of length L = the first L/64 chunks.
    cumulative = []
    for L in (8192, 32768, 65536, 131072):
        sel = [c for c in chunks if c["depth"] < L]
        if not sel or sel[-1]["depth"] + sel[-1]["T"] < L * 0.95:
            continue
        tot = sum(c["gpu_ms"] for c in sel)
        cumulative.append({"prompt_tokens": sel[-1]["depth"] + sel[-1]["T"], "gpu_s": tot / 1000.0,
                           **{k: sum(c[k] for c in sel) / tot for k in ("linear", "gdn", "attn_core", "other")}})

    # Least-squares line of attention-core ms per 64-row chunk vs depth, and of the whole chunk.
    full = [c for c in chunks if c["T"] == 64]

    def fit(key):
        xs = [c["depth"] for c in full]
        ys = [c[key] for c in full]
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
        return {"ms_at_0": my - b * mx, "ms_per_1k_depth": b * 1000.0}

    res = {"timeline": str(args.timeline), "prefill_tokens": total_tokens, "chunks": len(chunks),
           "sum_call_gpu_s": sum(c["gpu_ms"] for c in chunks) / 1000.0,
           "unprofiled_prefill_s": args.wall_s or None,
           "table": table, "cumulative": cumulative,
           "fit_per_chunk": {k: fit(k) for k in ("gpu_ms", "linear", "gdn", "attn_core", "other")},
           "per_chunk": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items() if k != "classes"}
                         for c in chunks]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")

    print(f"[depth] {len(chunks)} prefill chunks, {total_tokens} tokens; sum of call GPU spans "
          f"{res['sum_call_gpu_s']:.2f}s" + (f" (unprofiled prefill {args.wall_s:.2f}s)" if args.wall_s else ""))
    print(f"{'depth':>8} {'chunk ms':>9} {'linear':>8} {'gdn':>7} {'attn':>7} {'other':>7} | attn% linear%")
    for r in table:
        print(f"{r['depth']:>8} {r['gpu_ms']:>9.2f} {r['linear']:>8.2f} {r['gdn']:>7.2f} {r['attn_core']:>7.2f} "
              f"{r['other']:>7.2f} | {100 * r['attn_core'] / r['gpu_ms']:5.1f}% {100 * r['linear'] / r['gpu_ms']:5.1f}%")
    for c in cumulative:
        print(f"[depth] prompt {c['prompt_tokens']:>6}: GPU {c['gpu_s']:.1f}s  attn {100 * c['attn_core']:.1f}%  "
              f"linear {100 * c['linear']:.1f}%  gdn {100 * c['gdn']:.1f}%  other {100 * c['other']:.1f}%")
    print(f"[depth] attn_core fit: {res['fit_per_chunk']['attn_core']}")
    print(f"[depth] chunk fit:     {res['fit_per_chunk']['gpu_ms']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
