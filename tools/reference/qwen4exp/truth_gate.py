"""Noise-relative gate numbers of qwen4_exp against the fp32 truth (plan 4.10 / Q10; the Gemma lane's vocabulary, docs/gemma4-plan.md 9.9-9.11).

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen4exp\\truth_gate.py [--root E:\\models\\r4dx\\q4exp\\ref] [--truth truth-fp32]

Inputs (real_ref.py dumps under --root): `truth-fp32` (ref.py dtype=float32, recent rule), `canon` / `ref-hf` (stock bf16 reference, recent / hf rule),
`canon_p1` (the same bf16 stack, every segment with ONE more token: a second, independent bf16 rounding path), `probes\\{recent,hf}`
(probe_8k), `probes\\recent\\probe_8k_p1`. Everything through kl_report's own functions (fp64 over the full vocabulary).

Reported, per scope (canon = 4 segments x 1023 rows, probe_8k = 8191 rows):
    stock       KL(truth || bf16 stack) and top-1: what an independent, correct bf16 implementation shows against the truth
                (stock = the pooled rows of both bf16 runs, the reference and the +1 token run)
    bf16 pair   KL(bf16 run || bf16 +1 run) (noise_floor.shape_noise): what two correct bf16 implementations show against each other
    gate        the Gemma rule applied to the stock term: KL(truth || r4dx) <= 1.5 * stock + 0.005 and
                top-1(truth, r4dx) >= stock top-1 - 1 point, per group english / thai / code / ALL (kl_report.gate_thresholds)
Exit status 0; the numbers are the product.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import kl_report  # noqa: E402
from qwen4exp import noise_floor  # noqa: E402
from qwen4exp.common_qwen4exp import KL_CORPUS_DIR  # noqa: E402

GROUPS = {"english_prose": "english", "thai_prose": "thai", "python_source": "code", "cpp_source": "code"}


def seg_stats(res: dict) -> dict:
    return {"rows": res["rows"], "mean_kl": res["mean_kl"], "median_kl": res["median_kl"], "p99_kl": res["p99_kl"],
            "max_kl": res["max_kl"], "top1_agreement_pct": res["top1_agreement_pct"], "positions_kl_gt_1": res["positions_kl_gt_1"]}


def compare(ref_dir: Path, test_dir: Path, tokens: Path, segments: list[str] | None, row_chunk: int = 32) -> list[dict]:
    doc = json.loads(Path(tokens).read_text(encoding="utf-8"))
    return [kl_report.compare_segment(s["name"], s["token_ids"], ref_dir, test_dir, row_chunk, strict=True)
            for s in doc["segments"] if segments is None or s["name"] in segments]


def prefix_results(short_dir: Path, short_tokens: Path, long_dir: Path, long_tokens: Path, segments: list[str] | None,
                   row_chunk: int = 32) -> dict:
    """KL(short || long) over the short rows per segment (the `_p1` runs), with _kl/_top1 arrays kept for pooling."""
    sdoc = json.loads(Path(short_tokens).read_text(encoding="utf-8"))
    ldoc = {s["name"]: s["token_ids"] for s in json.loads(Path(long_tokens).read_text(encoding="utf-8"))["segments"]}
    out = {}
    for seg in sdoc["segments"]:
        if segments is not None and seg["name"] not in segments:
            continue
        r = noise_floor.prefix_check(Path(long_dir), seg["name"] + "_p1", Path(short_dir), seg["name"], ldoc[seg["name"] + "_p1"], row_chunk)
        out[seg["name"]] = r
    return out


def pool_prefix(res: dict) -> dict:
    kl = np.concatenate([r["_kl"] for r in res.values()])
    top1 = np.concatenate([r["_top1"] for r in res.values()])
    return {"rows": int(kl.size), "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)), "p99_kl": float(np.percentile(kl, 99)),
            "max_kl": float(kl.max()), "top1_agreement_pct": 100.0 * float(top1.mean()), "positions_kl_gt_1": int((kl > 1.0).sum())}


def pool_any(groups: list[dict]) -> dict:
    """Pool several {segment: result-with-_kl/_top1} dicts (row-weighted)."""
    kl = np.concatenate([r["_kl"] for g in groups for r in g.values()])
    top1 = np.concatenate([r["_top1"] for g in groups for r in g.values()])
    return {"rows": int(kl.size), "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)), "p99_kl": float(np.percentile(kl, 99)),
            "max_kl": float(kl.max()), "top1_agreement_pct": 100.0 * float(top1.mean())}


def group_noise(per_run: list[dict]) -> dict:
    """{group: pooled stats} over english / thai / code / ALL from several runs' {segment: result}."""
    out = {}
    names = {"ALL": None, "english": ["english_prose"], "thai": ["thai_prose"], "code": ["python_source", "cpp_source"]}
    for g, segs in names.items():
        sel = [{k: v for k, v in run.items() if segs is None or k in segs} for run in per_run]
        sel = [s for s in sel if s]
        if sel:
            out[g] = pool_any(sel)
    return out


def gate_rows(noise: dict) -> dict:
    return {g: {"noise_mean_kl": n["mean_kl"], "noise_top1_pct": n["top1_agreement_pct"], **kl_report.gate_thresholds(
        {"mean_kl": n["mean_kl"], "top1_pct": n["top1_agreement_pct"]})} for g, n in noise.items()}


def as_dict(results: list[dict]) -> dict:
    return {r["name"]: r for r in results}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path(r"E:\models\r4dx\q4exp\ref"))
    ap.add_argument("--truth", default="truth-fp32", help="truth directory under --root (truth-fp32, truth-fp32-w16)")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--row-chunk", type=int, default=32)
    args = ap.parse_args(argv)
    R, T = args.root, args.root / args.truth
    canon_tokens = KL_CORPUS_DIR / "tokens_canon.json"
    probe_tokens = R / "probes" / "tokens_probes.json"
    p1_canon_tokens = R / "canon_p1" / "tokens_canon_p1.json"
    p1_probe_tokens = R / "probes" / "tokens_probes_p1.json"
    rep: dict = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "truth_dir": str(T), "scopes": {}}

    scopes = {
        "canon": {"tokens": canon_tokens, "segments": None, "bf16": R / "canon", "bf16_hf": R / "ref-hf",
                  "p1_dir": R / "canon_p1", "p1_tokens": p1_canon_tokens},
        "probe_8k": {"tokens": probe_tokens, "segments": ["probe_8k"], "bf16": R / "probes" / "recent", "bf16_hf": R / "probes" / "hf",
                     "p1_dir": R / "probes" / "recent", "p1_tokens": p1_probe_tokens},
    }
    for name, sc in scopes.items():
        if not (T / ("cpp_source.logprobs.f16" if name == "canon" else "probe_8k.logprobs.f16")).exists():
            print(f"[truth_gate] {T} has no {name} dump yet; skipped")
            continue
        print(f"[truth_gate] {name}: KL(truth || bf16) ...", flush=True)
        a = as_dict(compare(T, sc["bf16"], sc["tokens"], sc["segments"], args.row_chunk))
        ah = as_dict(compare(T, sc["bf16_hf"], sc["tokens"], sc["segments"], args.row_chunk))
        print(f"[truth_gate] {name}: KL(truth || bf16 +1 token) ...", flush=True)
        b = prefix_results(T, sc["tokens"], sc["p1_dir"], sc["p1_tokens"], sc["segments"], args.row_chunk)
        stock = group_noise([a, b])
        rep["scopes"][name] = {
            "truth_vs_bf16_recent": {"segments": {k: seg_stats(v) for k, v in a.items()}, "pooled": pool_any([a])},
            "truth_vs_bf16_hf_rule": {"segments": {k: seg_stats(v) for k, v in ah.items()}, "pooled": pool_any([ah])},
            "truth_vs_bf16_plus1": {"segments": {k: {x: y for x, y in v.items() if not x.startswith("_")} for k, v in b.items()},
                                    "pooled": pool_prefix(b)},
            "stock_noise_groups": stock,
            "gate_proposal": gate_rows(stock),
        }
        print(json.dumps({k: rep["scopes"][name][k]["pooled"] for k in ("truth_vs_bf16_recent", "truth_vs_bf16_hf_rule", "truth_vs_bf16_plus1")}, indent=1))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"[truth_gate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
