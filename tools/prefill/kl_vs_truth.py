"""tools/prefill/kl_vs_truth.py -- the paired quality check of a prefill variant against the fp32 truth.

Two long-prefix teacher-forced dumps of the same tokens (tool_teacher_forced_logprobs --tail-rows R: only the last R rows of every
segment are written) are each compared with the HF fp32 truth dump of the whole segment (tools/reference/qwen35moe/ref_run.py
logprobs; rows 0..T-2). For every variant V and every scored row:

    KL(truth || V)        and top-1 agreement of truth and V,

and the PAIRED difference KL(truth || V) - KL(truth || BASE) per row, with its standard error. A variant that only re-rounds the
same computation moves the mean paired difference by ~0 +- a few standard errors; a real accuracy loss shows as a positive mean.
(KL(BASE || V) alone is not a quality number for this model: its own bf16 noise floor, KL(truth || stock bf16), is 0.046.)

    python tools\\prefill\\kl_vs_truth.py --tokens E:\\models\\r4dx\\q35moe\\kl\\tokens_gate.json --truth E:\\models\\r4dx\\q35moe\\kl\\hf_fp32 `
        --base <dump A> --test B=<dump B> --test C=<dump C>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "reference"))
from kl_report import chunk_stats  # noqa: E402

GROUP = {"english": "english", "thai": "thai", "python": "code", "cpp": "code", "tool": "tool", "english_prose": "english",
         "thai_prose": "thai", "python_source": "code", "cpp_source": "code", "mixed_long": "long"}


def group_of(name: str) -> str:
    key = name.split("_", 1)[1] if "_" in name else name
    if name.startswith("chat_"):
        key = key.split("_")[0]
    return GROUP.get(key, GROUP.get(key.split("_")[0], "other"))


def load(d: Path, name: str):
    meta = json.loads((d / f"{name}.meta.json").read_text(encoding="utf-8"))
    rows, V = int(meta["rows"]), int(meta["V"])
    return meta, np.memmap(d / f"{name}.logprobs.f16", dtype=np.float16, mode="r", shape=(rows, V))


def per_row(truth_dir: Path, name: str, ids: np.ndarray, side_dir: Path, row_chunk: int = 16):
    meta, arr = load(side_dir, name)
    first = int(meta.get("first_row", 0))
    tmeta, tarr = load(truth_dir, name)
    kl, top1 = [], []
    for r0 in range(0, arr.shape[0], row_chunk):
        r1 = min(arr.shape[0], r0 + row_chunk)
        st = chunk_stats(tarr[first + r0:first + r1], arr[r0:r1], ids[first + 1 + r0:first + 1 + r1])
        kl.append(st["kl"])
        top1.append(st["top1_agree"])
    return np.concatenate(kl), np.concatenate(top1).astype(np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--test", action="append", default=[], help="NAME=DIR")
    args = ap.parse_args()
    tests = [(t.split("=", 1)[0], Path(t.split("=", 1)[1])) for t in args.test]
    doc = json.loads(args.tokens.read_text(encoding="utf-8"))
    rows_by = defaultdict(lambda: defaultdict(list))  # variant -> group -> [kl rows]
    top_by = defaultdict(lambda: defaultdict(list))
    for seg in doc["segments"]:
        name = seg["name"]
        if not (args.base / f"{name}.meta.json").exists() or not (args.truth / f"{name}.meta.json").exists():
            continue
        ids = np.asarray(seg["token_ids"], dtype=np.int64)
        g = group_of(name)
        for label, d in [("BASE", args.base)] + tests:
            if not (d / f"{name}.meta.json").exists():
                continue
            kl, t1 = per_row(args.truth, name, ids, d)
            for key in (g, "ALL"):
                rows_by[label][key].append(kl)
                top_by[label][key].append(t1)
        print(f"[kvt] {name}", flush=True)
    groups = ["english", "thai", "code", "tool", "long", "ALL"]
    print("\n| group | rows | variant | KL(truth||V) | top-1(truth,V) % | paired dKL vs BASE | +- s.e. | dTop-1 pts |")
    print("|---|---:|---|---:|---:|---:|---:|---:|")
    for g in groups:
        if g not in rows_by["BASE"]:
            continue
        base = np.concatenate(rows_by["BASE"][g])
        btop = np.concatenate(top_by["BASE"][g])
        print(f"| {g} | {base.size} | BASE | {base.mean():.5f} | {100 * btop.mean():.2f} | | | |")
        for label, _ in tests:
            if g not in rows_by[label]:
                continue
            v = np.concatenate(rows_by[label][g])
            vt = np.concatenate(top_by[label][g])
            d = v - base
            se = d.std(ddof=1) / math.sqrt(d.size) if d.size > 1 else float("nan")
            print(f"| {g} | {v.size} | {label} | {v.mean():.5f} | {100 * vt.mean():.2f} | {d.mean():+.5f} | {se:.5f} | {100 * (vt.mean() - btop.mean()):+.2f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
