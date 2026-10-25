"""tools/reference/llama/kl_gate_llama.py  (docs/nanbeige-plan.md 6.3 / N2-3: the Nanbeige w8a16 KL gate)

Reads three dump directories in the shared format (`<seg>.logprobs.f16` [T-1, V] fp16 log-probs + `<seg>.meta.json`):
  --truth      tools/reference/llama/full_logits_llama_cpu.py --variant truth
  --noise      the same tool, --variant bf16sdpa (the bf16-engine yardstick)
  --candidate  tests/model/tool_teacher_forced_logprobs.exe (r4dx; w8a16 at --tp 2)
and the chat tokens file (score masks and groups). For every gate group it pools the SCORED rows (row i is scored when
score_mask[i + 1] == 1: it predicts a model-turn token) and reports mean / median / p99 KL(truth || x) for x in {noise,
candidate} and top-1 agreement with the truth, then applies the plan's rule per group:

    KL(truth || candidate) <= 1.5 x KL(truth || noise) + epsilon        (epsilon 0.005 by default)
    top1(truth, candidate) >= top1(truth, noise) - 1.0 point

Groups: english, thai, code (python + cpp), tool, zh, and ALL (every scored row). Raw segments (--raw tokens files,
mask-less: every row scored) are reported, not gated. Writes --out (JSON) and prints a markdown table; exit 1 when a
gated group fails.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\llama\\kl_gate_llama.py --chat <chat_nanbeige.json> `
        --raw <tokens_nanbeige.json> --raw <tokens_nanbeige_long.json> --truth <dir> --noise <dir> --candidate <dir> --out gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

GROUP_OF = {"python": "code", "cpp": "code"}
GATED = ("english", "thai", "code", "tool", "zh", "ALL")


def load(d: Path, name: str, V: int | None = None):
    meta = json.loads((d / f"{name}.meta.json").read_text(encoding="utf-8"))
    V = V or int(meta["V"])
    a = np.fromfile(d / f"{name}.logprobs.f16", dtype=np.float16).reshape(-1, V)
    return a, meta


def row_stats(t: np.ndarray, x: np.ndarray, rows: np.ndarray, block: int = 256):
    """KL(t || x) per row and argmax agreement, over `rows`, in blocks (fp64)."""
    kls, agree = [], []
    for s in range(0, len(rows), block):
        r = rows[s:s + block]
        lt, lx = t[r].astype(np.float64), x[r].astype(np.float64)
        p = np.exp(lt)
        kls.append((p * (lt - lx)).sum(-1))
        agree.append(lt.argmax(-1) == lx.argmax(-1))
    return np.concatenate(kls), np.concatenate(agree)


def summarize(kl: np.ndarray, agree: np.ndarray) -> dict:
    return {"rows": int(kl.size), "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)),
            "p99_kl": float(np.percentile(kl, 99)), "max_kl": float(kl.max()), "top1": float(agree.mean() * 100.0)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chat", type=Path, required=True)
    ap.add_argument("--raw", type=Path, action="append", default=[])
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--noise", type=Path, required=True)
    ap.add_argument("--candidate", type=Path, required=True)
    ap.add_argument("--epsilon", type=float, default=0.005)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    pooled = {g: {"noise": ([], []), "cand": ([], [])} for g in GATED}
    per_segment = []
    chat = json.loads(args.chat.read_text(encoding="utf-8"))
    for s in chat["segments"]:
        n, ids, mask = s["name"], s["token_ids"], s["score_mask"]
        t, tm = load(args.truth, n)
        x, _ = load(args.noise, n, tm["V"])
        c, cm = load(args.candidate, n, tm["V"])
        if not (t.shape == x.shape == c.shape) or len(ids) - 1 != t.shape[0]:
            raise SystemExit(f"{n}: shapes differ (truth {t.shape}, noise {x.shape}, candidate {c.shape}, T {len(ids)})")
        if cm.get("sha256_of_token_ids_json") not in (None, tm["sha256_of_token_ids_json"]):
            raise SystemExit(f"{n}: candidate token ids differ from the truth's")
        rows = np.array([i for i in range(len(ids) - 1) if mask[i + 1] == 1])
        g = GROUP_OF.get(s["group"], s["group"])
        kn, an = row_stats(t, x, rows)
        kc, ac = row_stats(t, c, rows)
        per_segment.append({"segment": n, "group": g, "noise": summarize(kn, an), "candidate": summarize(kc, ac)})
        for grp in (g, "ALL"):
            if grp in pooled:
                pooled[grp]["noise"][0].append(kn); pooled[grp]["noise"][1].append(an)
                pooled[grp]["cand"][0].append(kc); pooled[grp]["cand"][1].append(ac)

    groups, ok_all = [], True
    for g in GATED:
        if not pooled[g]["noise"][0]:
            continue
        nz = summarize(np.concatenate(pooled[g]["noise"][0]), np.concatenate(pooled[g]["noise"][1]))
        cd = summarize(np.concatenate(pooled[g]["cand"][0]), np.concatenate(pooled[g]["cand"][1]))
        limit = 1.5 * nz["mean_kl"] + args.epsilon
        ok = cd["mean_kl"] <= limit and cd["top1"] >= nz["top1"] - 1.0
        ok_all = ok_all and ok
        groups.append({"name": g, "noise": nz, "candidate": cd, "kl_limit": limit, "pass": ok})

    raws = []
    for rp in args.raw:
        doc = json.loads(rp.read_text(encoding="utf-8"))
        for s in doc["segments"]:
            n = s["name"]
            if not (args.candidate / f"{n}.meta.json").exists():
                continue
            t, tm = load(args.truth, n)
            x, _ = load(args.noise, n, tm["V"])
            c, _ = load(args.candidate, n, tm["V"])
            r = min(t.shape[0], x.shape[0], c.shape[0])
            rows = np.arange(r)
            kn, an = row_stats(t, x, rows)
            kc, ac = row_stats(t, c, rows)
            raws.append({"segment": n, "noise": summarize(kn, an), "candidate": summarize(kc, ac)})

    res = {"rule": f"KL(truth||candidate) <= 1.5 x KL(truth||noise) + {args.epsilon}; top-1 >= noise - 1 pt",
           "truth": str(args.truth), "noise": str(args.noise), "candidate": str(args.candidate), "chat": str(args.chat),
           "groups": groups, "segments": per_segment, "raw": raws, "pass": ok_all}
    args.out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("| group | rows | KL noise | KL r4dx | limit | top-1 noise | top-1 r4dx | p99 r4dx | pass |")
    print("|---|--:|--:|--:|--:|--:|--:|--:|---|")
    for g in groups:
        n, c = g["noise"], g["candidate"]
        print(f"| {g['name']} | {c['rows']} | {n['mean_kl']:.5f} | {c['mean_kl']:.5f} | {g['kl_limit']:.5f} | "
              f"{n['top1']:.2f} | {c['top1']:.2f} | {c['p99_kl']:.4f} | {'PASS' if g['pass'] else 'FAIL'} |")
    for r in raws:
        print(f"| raw {r['segment']} | {r['candidate']['rows']} | {r['noise']['mean_kl']:.5f} | {r['candidate']['mean_kl']:.5f} "
              f"| - | {r['noise']['top1']:.2f} | {r['candidate']['top1']:.2f} | {r['candidate']['p99_kl']:.4f} | (reported) |")
    print(f"\ngate: {'PASS' if ok_all else 'FAIL'} -> {args.out}")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
