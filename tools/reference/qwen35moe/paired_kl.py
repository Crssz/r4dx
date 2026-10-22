"""Paired comparison of two r4dx dumps against one reference dump: per-row KL difference (A - B), mean and a block-bootstrap interval.

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\paired_kl.py --tokens tokens_gate.json --ref <truth dir> --a <dir> --b <dir>

The rows of one sequence are correlated, so the interval resamples whole segments (and, for the long segment, 512-row blocks), not
single rows. Used to say whether the calibrated-KV container differs from the placeholder-KV one beyond the sampling noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import kl_report  # noqa: E402

BLOCK = 512


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--row-chunk", type=int, default=256)
    a = ap.parse_args()
    segs = json.loads(Path(a.tokens).read_text(encoding="utf-8"))["segments"]
    blocks_d, blocks_n, t1 = [], [], []
    for s in segs:
        mask = kl_report.row_mask_from_score_mask(s["score_mask"]) if "score_mask" in s else None
        ra = kl_report.compare_segment(s["name"], s["token_ids"], Path(a.ref), Path(a.a), a.row_chunk, True, row_mask=mask)
        rb = kl_report.compare_segment(s["name"], s["token_ids"], Path(a.ref), Path(a.b), a.row_chunk, True, row_mask=mask)
        d = ra["_kl"] - rb["_kl"]
        t1.append(ra["_top1"].astype(float) - rb["_top1"].astype(float))
        for lo in range(0, d.size, BLOCK):
            blocks_d.append(float(d[lo:lo + BLOCK].sum()))
            blocks_n.append(int(d[lo:lo + BLOCK].size))
    bd, bn = np.array(blocks_d), np.array(blocks_n)
    n = bn.sum()
    mean = bd.sum() / n
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(a.boot):
        pick = rng.integers(0, bd.size, bd.size)
        boots.append(bd[pick].sum() / bn[pick].sum())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    t1 = np.concatenate(t1)
    print(f"rows {n}, blocks {bd.size}: mean KL(ref||A) - KL(ref||B) = {mean:+.5f}  (95% block-bootstrap {lo:+.5f} .. {hi:+.5f}); "
          f"top-1(A) - top-1(B) = {100 * t1.mean():+.3f} pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
