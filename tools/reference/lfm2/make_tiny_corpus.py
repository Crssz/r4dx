"""tools/reference/lfm2/make_tiny_corpus.py -- a synthetic token-id samples JSONL for the CPU tests of the LFM2 quant
tools (the format of gen_samples.py / gemma.capture.load_token_samples: id "<category>/<NNN>", category, rejected,
token_ids, kl_token_overlap, calib_kind). Random ids with a Zipf-ish skew (so the random tiny router sees a varying
input), `--seqs` samples of `--len` tokens spread over the five categories.

    python make_tiny_corpus.py --out D:\\r4dx-scratch\\lfm2\\tiny\\samples.jsonl --vocab 512 --seqs 20 --len 1024
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CATS = ("thai_prose", "english_prose", "chat", "code", "multilingual")


def make(path: Path, vocab: int, seqs: int, length: int, seed: int = 7) -> int:
    rng = np.random.default_rng(seed)
    p = 1.0 / np.arange(1, vocab + 1) ** 0.8
    p /= p.sum()
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i in range(seqs):
            cat = CATS[i % len(CATS)]
            ids = [1] + [int(t) for t in rng.choice(vocab, size=length - 1, p=p)]
            f.write(json.dumps({"id": f"{cat}/{i // len(CATS):03d}", "category": cat, "format": "raw",
                                "rejected": False, "reject_reason": None, "token_ids": ids,
                                "calib_kind": "raw_text", "kl_token_overlap": 0}) + "\n")
            n += len(ids)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--vocab", type=int, default=512)
    ap.add_argument("--seqs", type=int, default=20)
    ap.add_argument("--len", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    print(f"[tiny-corpus] wrote {make(a.out, a.vocab, a.seqs, a.len, a.seed)} tokens to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
