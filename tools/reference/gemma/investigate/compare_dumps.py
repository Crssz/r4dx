"""Mean KL / top-1 agreement between two tool_teacher_forced_logprobs dumps of the same segment (CPU).

  python compare_dumps.py <dirA> <dirB> <segment>
"""
import json
import sys
from pathlib import Path

import numpy as np

a_dir, b_dir, seg = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]


def load(d: Path) -> np.ndarray:
    meta = json.loads((d / f"{seg}.meta.json").read_text(encoding="utf-8"))
    rows = meta.get("rows") or meta.get("n_rows")
    vocab = meta.get("vocab") or meta.get("vocab_size")
    x = np.fromfile(d / f"{seg}.logprobs.f16", dtype=np.float16).astype(np.float32)
    if rows is None or vocab is None:
        vocab = 262144
        rows = x.size // vocab
    return x.reshape(int(rows), int(vocab))


a, b = load(a_dir), load(b_dir)
pa = np.exp(a)
kl = (pa * (a - b)).sum(axis=1)
top1 = (a.argmax(1) == b.argmax(1)).mean()
print(f"{seg}: rows {a.shape[0]}  mean KL(A||B) {kl.mean():.5f}  max {kl.max():.4f}  top-1 agree {top1 * 100:.2f}%")
