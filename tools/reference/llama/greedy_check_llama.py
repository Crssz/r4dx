"""Greedy smoke by teacher-forced truth (docs/nanbeige-plan.md 6.3, N2-4).  r4dx greedy-generates (`r4dx-cli --temperature 0
--dump-token-ids`), full_logits_llama_cpu.py --variant truth teacher-forces the same ids, and this script checks every
generated token: it must be the truth's top-1, or a near-tie (truth top-1 log-prob minus the chosen token's < --tie).

  python -I tools\\reference\\llama\\greedy_check_llama.py --tokens greedy_tokens.json --prompt-lens prompt_lens.json `
      --truth <dir> [--tie 0.5]

prompt_lens.json maps a segment's name without the "greedy_" prefix to its prompt length (r4dx-cli's [stats] prefill count).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=Path, required=True)
    ap.add_argument("--prompt-lens", type=Path, required=True)
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--tie", type=float, default=0.5)
    a = ap.parse_args(argv)
    doc = json.loads(a.tokens.read_text(encoding="utf-8"))
    plens = json.loads(a.prompt_lens.read_text(encoding="utf-8"))
    ok_all = True
    for seg in doc["segments"]:
        name, ids = seg["name"], seg["token_ids"]
        meta = json.loads((a.truth / f"{name}.meta.json").read_text(encoding="utf-8"))
        lp = np.fromfile(a.truth / f"{name}.logprobs.f16", dtype=np.float16).reshape(-1, meta["V"])
        p = plens[name.removeprefix("greedy_")]
        exact, ties, bad = 0, [], []
        for i in range(p, len(ids)):
            row = lp[i - 1].astype(np.float32)  # row i-1 predicts token i
            top = int(row.argmax())
            gap = float(row[top] - row[ids[i]])
            if top == ids[i]:
                exact += 1
            elif gap < a.tie:
                ties.append((i - p, round(gap, 3)))
            else:
                bad.append((i - p, ids[i], top, round(gap, 3)))
        n = len(ids) - p
        ok = not bad
        ok_all &= ok
        print(f"{name}: {n} generated, {exact} top-1, near-ties {ties}, misses {bad} -> {'PASS' if ok else 'FAIL'}")
    print("greedy smoke:", "PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
