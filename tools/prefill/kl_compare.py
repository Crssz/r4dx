"""tools/prefill/kl_compare.py -- compares two long-prefix teacher-forced dumps (run_kl.ps1 output:
tool_teacher_forced_logprobs --tail-rows R) row for row: a dense reference run vs a variant (new
prefill kernel, bigger chunks, sparse attention, TP=2, ...).

Per row (fp64, full vocabulary; the math is tools/reference/kl_report.py's chunk_stats, reused):
    KL(P_ref || Q_test), top-1 agreement, top-5 containment, NLL of the actual next token.
Reported per segment, per length (pooled over kinds) and overall: mean / median / p99 / max KL,
top-1 %, top-5 %, perplexity of each side over the continuation rows, and each side's prefix
prefill seconds (from the sidecars) with the speed ratio.

Both sides must have scored the same tokens (sidecar sha256_of_token_ids_json) with the same
first_row/rows; mismatches are refused.

Usage:
  python tools\\prefill\\kl_compare.py --ref <models root>\\r4dx\\prefill-m0\\kl\\dense `
      --test <models root>\\r4dx\\prefill-m0\\kl\\variant `
      --tokens <models root>\\r4dx\\prefill-m0\\kl\\tokens_long.json [--json out.json]
A dense-vs-dense rerun gives the run-to-run noise floor (0 when the engine is deterministic).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import pf_common as C

sys.path.insert(0, str(C.REPO / "tools" / "reference"))
from kl_report import chunk_stats  # noqa: E402


def load_side(d: Path, name: str):
    meta = json.loads((d / f"{name}.meta.json").read_text(encoding="utf-8"))
    rows, V = int(meta["rows"]), int(meta["V"])
    arr = np.memmap(d / f"{name}.logprobs.f16", dtype=np.float16, mode="r", shape=(rows, V))
    return meta, arr


def agg(parts: dict) -> dict:
    kl = np.concatenate(parts["kl"])
    nll_r = np.concatenate(parts["nll_ref"])
    nll_t = np.concatenate(parts["nll_test"])
    return {
        "rows": int(kl.size),
        "kl_mean": float(kl.mean()), "kl_median": float(np.median(kl)),
        "kl_p99": float(np.percentile(kl, 99)), "kl_max": float(kl.max()),
        "top1_pct": 100.0 * float(np.concatenate(parts["top1"]).mean()),
        "top5_pct": 100.0 * float(np.concatenate(parts["top5"]).mean()),
        "ppl_ref": float(math.exp(nll_r.mean())), "ppl_test": float(math.exp(nll_t.mean())),
        "prefill_s_ref": float(sum(parts["pf_ref"])), "prefill_s_test": float(sum(parts["pf_test"])),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", type=Path, required=True)
    ap.add_argument("--test", type=Path, required=True)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--segment", default="", help="comma-separated subset")
    ap.add_argument("--row-chunk", type=int, default=32)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)

    doc = json.loads(args.tokens.read_text(encoding="utf-8"))
    want = {s for s in args.segment.split(",") if s}
    per_seg, per_len, overall = {}, defaultdict(lambda: defaultdict(list)), defaultdict(list)
    skipped: list[str] = []
    for seg in doc["segments"]:
        name = seg["name"]
        if want and name not in want:
            continue
        if not (args.ref / f"{name}.meta.json").exists() or not (args.test / f"{name}.meta.json").exists():
            if want:
                print(f"[kl] {name}: missing on one side, skipped")
            skipped.append(name)
            continue
        mr, ar = load_side(args.ref, name)
        mt, at = load_side(args.test, name)
        for key in ("sha256_of_token_ids_json", "rows", "V", "first_row"):
            if mr.get(key) != mt.get(key):
                raise SystemExit(f"[kl] {name}: {key} differs (ref {mr.get(key)} vs test {mt.get(key)})")
        first = int(mr.get("first_row", 0))
        ids = np.asarray(seg["token_ids"], dtype=np.int64)
        parts = defaultdict(list)
        for r0 in range(0, ar.shape[0], args.row_chunk):
            r1 = min(ar.shape[0], r0 + args.row_chunk)
            nxt = ids[first + 1 + r0:first + 1 + r1]
            st = chunk_stats(ar[r0:r1], at[r0:r1], nxt)
            parts["kl"].append(st["kl"])
            parts["top1"].append(st["top1_agree"])
            parts["top5"].append(st["top5_contain"])
            parts["nll_ref"].append(st["nll_ref"])
            parts["nll_test"].append(st["nll_test"])
        parts["pf_ref"].append(float(mr.get("prefill_seconds", 0.0)))
        parts["pf_test"].append(float(mt.get("prefill_seconds", 0.0)))
        per_seg[name] = agg(parts)
        per_seg[name]["prefix_tokens"] = int(mr.get("prefix_tokens", first + 1))
        length = name.rsplit("_", 1)[-1]
        for k, v in parts.items():
            per_len[length][k].extend(v)
            overall[k].extend(v)
        s = per_seg[name]
        print(f"[kl] {name:<14} rows={s['rows']} KL mean={s['kl_mean']:.6f} p99={s['kl_p99']:.6f} "
              f"max={s['kl_max']:.4f} top1={s['top1_pct']:.2f}% ppl ref/test={s['ppl_ref']:.4f}/{s['ppl_test']:.4f} "
              f"prefill {s['prefill_s_ref']:.2f}s/{s['prefill_s_test']:.2f}s", flush=True)
    if skipped:
        print(f"[kl] {len(skipped)} segment(s) not present on both sides: {', '.join(skipped)}")
    if not per_seg:
        raise SystemExit("[kl] no segment present on both sides")
    lens = {k: agg(v) for k, v in per_len.items()}
    tot = agg(overall)
    print("\n| scope | rows | mean KL | median KL | p99 KL | max KL | top-1 % | top-5 % | ppl ref | ppl test "
          "| prefill ref s | prefill test s | speedup |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for scope, s in list(per_seg.items()) + [(f"len {k}", v) for k, v in lens.items()] + [("ALL", tot)]:
        sp = s["prefill_s_ref"] / s["prefill_s_test"] if s["prefill_s_test"] > 0 else float("nan")
        print(f"| {scope} | {s['rows']} | {s['kl_mean']:.6f} | {s['kl_median']:.6f} | {s['kl_p99']:.6f} | "
              f"{s['kl_max']:.4f} | {s['top1_pct']:.2f} | {s['top5_pct']:.2f} | {s['ppl_ref']:.4f} | "
              f"{s['ppl_test']:.4f} | {s['prefill_s_ref']:.2f} | {s['prefill_s_test']:.2f} | {sp:.3f} |")
    if args.json:
        C.write_json(args.json, {"ref": str(args.ref), "test": str(args.test), "tokens": str(args.tokens),
                                 "segments": per_seg, "lengths": lens, "overall": tot})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
