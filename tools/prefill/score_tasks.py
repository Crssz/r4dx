"""tools/prefill/score_tasks.py -- scores run_tasks.py results against the task set's ground truth.

Match rules (per item's "match" field):
  all        every answer string (7-digit numbers) must appear in the output, not as part of a
             longer digit run; score = fraction found
  all_words  every answer word must appear as a whole word (case-insensitive); score = fraction
  number     some numeric literal in the output equals the answer (C/C++ suffixes u/l/f, digit
             separators ' and _, and hex are normalized); score 0/1
  word       the answer identifier appears as a whole word (case-sensitive); score 0/1
  path       the answer file's basename appears in the output (slashes normalized); score 0/1
An item "passes" when its score is 1.0.

Also summarizes prefill speed from the server's `timings` (prompt_n / prompt_ms per request -- a
cold prefill, since every item starts with a unique line).

Usage:
  python tools\\prefill\\score_tasks.py --results <run_dir>\\results.jsonl [--json <out.json>]
  python tools\\prefill\\score_tasks.py --results A\\results.jsonl --compare B\\results.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

import pf_common as C

LEN_ORDER = list(C.LENGTHS)


def _num(s: str):
    s = s.replace("'", "").replace("_", "").rstrip("uUlLfF")
    try:
        return int(s, 16) if s.lower().startswith(("0x", "-0x")) else float(s)
    except ValueError:
        return None


def score_item(item: dict, output: str) -> float:
    m = item["match"]
    ans = item["answers"]
    out = output or ""
    if m == "all":
        hits = sum(1 for a in ans if re.search(rf"(?<!\d){re.escape(a)}(?!\d)", out))
        return hits / len(ans)
    if m == "all_words":
        hits = sum(1 for a in ans if re.search(rf"\b{re.escape(a)}\b", out, re.I))
        return hits / len(ans)
    if m == "number":
        want = _num(ans[0])
        for tok in re.findall(r"-?(?:0[xX][0-9a-fA-F']+|\d[\d'_]*(?:\.\d+)?(?:[eE][-+]?\d+)?)[uUlLfF]*", out):
            got = _num(tok)
            if got is not None and want is not None and abs(got - want) <= 1e-9 * max(1.0, abs(want)):
                return 1.0
        return 0.0
    if m == "word":
        return 1.0 if re.search(rf"(?<![\w]){re.escape(ans[0])}(?![\w])", out) else 0.0
    if m == "path":
        base = ans[0].replace("\\", "/").rsplit("/", 1)[-1]
        return 1.0 if base in out.replace("\\", "/") else 0.0
    raise ValueError(f"unknown match rule {m!r}")


def load_results(path: Path) -> list[dict]:
    rows = C.read_jsonl(path)
    for r in rows:  # always re-scored, so a fixed matcher applies to old runs too
        r["score"] = 0.0 if r.get("error") else score_item(r, r.get("output", ""))
    return rows


def summarize(rows: list[dict]) -> dict:
    by = defaultdict(list)
    speed = defaultdict(list)
    for r in rows:
        by[(r["task"], r["length"])].append(r)
        t = r.get("timings") or {}
        if t.get("prompt_ms"):
            speed[r["length"]].append((t["prompt_n"], t["prompt_ms"]))
    tasks = sorted({k[0] for k in by}, key=lambda t: (t != "niah_single", t))
    lengths = sorted({k[1] for k in by}, key=LEN_ORDER.index)
    cells = {}
    for (task, length), rs in by.items():
        cells[f"{task}/{length}"] = {
            "n": len(rs), "score": sum(r["score"] for r in rs) / len(rs),
            "pass": sum(1 for r in rs if r["score"] >= 1.0) / len(rs),
            "failed_ids": [r["id"] for r in rs if r["score"] < 1.0]}
    per_len = {}
    for length in lengths:
        rs = [r for r in rows if r["length"] == length]
        sp = speed.get(length, [])
        per_len[length] = {
            "n": len(rs), "mean_score": sum(r["score"] for r in rs) / len(rs),
            "prefill_tok_s_median": statistics.median(n / (ms / 1000.0) for n, ms in sp) if sp else None,
            "prefill_s_median": statistics.median(ms / 1000.0 for _, ms in sp) if sp else None,
            "prompt_n_median": statistics.median(n for n, _ in sp) if sp else None}
    return {"tasks": tasks, "lengths": lengths, "cells": cells, "per_length": per_len,
            "n_items": len(rows), "errors": sum(1 for r in rows if r.get("error"))}


def print_table(s: dict, title: str) -> None:
    print(f"## {title}  ({s['n_items']} items, {s['errors']} errors)")
    hdr = "| task | " + " | ".join(s["lengths"]) + " |"
    print(hdr)
    print("|---" * (len(s["lengths"]) + 1) + "|")
    for t in s["tasks"]:
        cells = []
        for length in s["lengths"]:
            c = s["cells"].get(f"{t}/{length}")
            cells.append("-" if c is None else f"{100 * c['score']:.1f} ({c['n']})")
        print(f"| {t} | " + " | ".join(cells) + " |")
    print("| **mean** | " + " | ".join(f"{100 * s['per_length'][l]['mean_score']:.1f}" for l in s["lengths"]) + " |")
    print("| prefill tok/s (median) | " + " | ".join(
        "-" if s["per_length"][l]["prefill_tok_s_median"] is None
        else f"{s['per_length'][l]['prefill_tok_s_median']:.1f}" for l in s["lengths"]) + " |")
    print("| prefill s (median) | " + " | ".join(
        "-" if s["per_length"][l]["prefill_s_median"] is None
        else f"{s['per_length'][l]['prefill_s_median']:.2f}" for l in s["lengths"]) + " |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--compare", type=Path, default=None, help="a second results.jsonl (variant)")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    a = load_results(args.results)
    sa = summarize(a)
    print_table(sa, str(args.results))
    out = {"a": sa}
    if args.compare:
        b = load_results(args.compare)
        sb = summarize(b)
        print()
        print_table(sb, str(args.compare))
        bi = {r["id"]: r for r in b}
        flips = [(r["id"], r["score"], bi[r["id"]]["score"]) for r in a
                 if r["id"] in bi and abs(r["score"] - bi[r["id"]]["score"]) > 1e-9]
        same_text = sum(1 for r in a if r["id"] in bi and r.get("output") == bi[r["id"]].get("output"))
        common = sum(1 for r in a if r["id"] in bi)
        print(f"\nidentical outputs: {same_text}/{common}; score changes: {len(flips)}")
        for iid, x, y in flips:
            print(f"  {iid}: {x:.2f} -> {y:.2f}")
        out.update(b=sb, identical_outputs=same_text, common=common, flips=flips)
    if args.json:
        C.write_json(args.json, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
