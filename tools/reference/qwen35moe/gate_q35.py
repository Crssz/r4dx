"""Rung 4 KL gate of the Ornith-1.5-122B (qwen3_5_moe) trellis container: held-out text, gate A + noise-relative gate B, per position.

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\gate_q35.py prep --out E:\\models\\r4dx\\q35moe\\kl\\tokens_gate.json
    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\gate_q35.py eval --tokens ...\\tokens_gate.json \\
        --stock <HF bf16 dumps> --truth <HF fp32 dumps> --test placeholder-kv=<r4dx dumps> --test calibrated-kv=<r4dx dumps> --out report

* `prep` merges the held-out corpora of tools/reference/kl_corpus (chat_q35moe.json: twelve thinking / tool conversations scored on
  model-turn tokens; tokens_q35moe.json: four raw 1024-token texts; tokens_q35moe_long.json: ~10k tokens of the four texts joined) into
  one token file with unique segment names, a `group` and, for chat segments, the `score_mask`.
* `eval` compares every `--test` dump with (A) the stock HF bf16 stack and (B) the fp32 truth, and measures the noise N = KL(truth ||
  stock bf16) on the same rows:
    Gate A  (the original gate): pooled KL(bf16 || r4dx) mean <= 0.02 and top-1(bf16, r4dx) >= 93 %.
    Gate B  (noise-relative):    per group KL(truth || r4dx) <= 1.5 * N + 0.005 and top-1(truth, r4dx) >= N_top1 - 1 point.
  Groups: english, thai, code (python + cpp), ALL are gated (as in tools/reference/qwen4exp/gate_eval.py); tool, long and each
  segment are reported. Chat segments contribute their model-turn rows only, raw segments every row. The same numbers are also
  bucketed by position in the sequence (early vs late tokens) because the fp8 KV cache and the GDN state errors grow with context.
Everything is `kl_report.compare_segment` / `chunk_stats` arithmetic (fp64 over the full vocabulary).
Exit status: 0 = every counted gate passed, 1 = a counted gate failed, 2 = a dump is missing / inconsistent.
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

KL_CORPUS = HERE.parent / "kl_corpus"
GATE_A_KL_MAX = 0.02
GATE_A_TOP1_MIN = 93.0
GATE_B_FACTOR = 1.5
GATE_B_SLACK = 0.005
GATE_B_TOP1_SLACK_PTS = 1.0
CHAT_GROUP = {"english": "english", "thai": "thai", "python": "code", "cpp": "code", "tool": "tool"}
RAW_GROUP = {"english_prose": "english", "thai_prose": "thai", "python_source": "code", "cpp_source": "code", "mixed_long": "long"}
GATED = ["english", "thai", "code", "ALL"]
REPORTED = ["tool", "long"]
BUCKETS = [(0, 128), (128, 512), (512, 1024), (1024, 2048), (2048, 4096), (4096, 8192), (8192, 1 << 30)]


def prep(out: Path) -> int:
    chat = json.loads((KL_CORPUS / "chat_q35moe.json").read_text(encoding="utf-8"))
    raw = json.loads((KL_CORPUS / "tokens_q35moe.json").read_text(encoding="utf-8"))
    long = json.loads((KL_CORPUS / "tokens_q35moe_long.json").read_text(encoding="utf-8"))
    segs = []
    for s in chat["segments"]:
        segs.append({"name": "chat_" + s["name"], "kind": "chat", "group": CHAT_GROUP[s["group"]], "token_ids": s["token_ids"],
                     "score_mask": s["score_mask"]})
    for doc in (raw, long):
        for s in doc["segments"]:
            segs.append({"name": "raw_" + s["name"], "kind": "raw", "group": RAW_GROUP[s["name"]], "token_ids": s["token_ids"]})
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tokenizer": chat["tokenizer"], "tokenizer_mode": "canonical", "chat_template": False,
                               "add_special_tokens": False, "segments": segs}), encoding="utf-8")
    scored = sum(sum(s["score_mask"]) if "score_mask" in s else len(s["token_ids"]) - 1 for s in segs)
    print(f"wrote {out}: {len(segs)} segments, {sum(len(s['token_ids']) for s in segs)} tokens, {scored} scored rows")
    return 0


def mask_of(seg: dict):
    return kl_report.row_mask_from_score_mask(seg["score_mask"]) if "score_mask" in seg else None


def compare_all(segs: list[dict], ref_dir: Path, test_dir: Path, row_chunk: int) -> dict[str, dict]:
    out = {}
    for s in segs:
        r = kl_report.compare_segment(s["name"], s["token_ids"], ref_dir, test_dir, row_chunk, strict=True, row_mask=mask_of(s))
        r["_group"] = s["group"]
        out[s["name"]] = r
    return out


def pooled(results: dict[str, dict], group: str | None) -> dict | None:
    items = [r for r in results.values() if group in (None, "ALL") or r["_group"] == group]
    return kl_report.pool_stats(items) if items else None


def bucketed(results: dict[str, dict]) -> list[dict]:
    rows = []
    for lo, hi in BUCKETS:
        kl, top1 = [], []
        for r in results.values():
            sel = (r["_pos"] >= lo) & (r["_pos"] < hi)
            if sel.any():
                kl.append(r["_kl"][sel])
                top1.append(r["_top1"][sel])
        if kl:
            kl, top1 = np.concatenate(kl), np.concatenate(top1)
            rows.append({"from": lo, "to": None if hi >= 1 << 30 else hi, "rows": int(kl.size), "mean_kl": float(kl.mean()),
                         "p99_kl": float(np.percentile(kl, 99)), "top1_pct": 100.0 * float(top1.mean())})
    return rows


def brief(p: dict) -> dict:
    return {k: p[k] for k in ("rows", "mean_kl", "median_kl", "p99_kl", "max_kl", "top1_pct")}


def evaluate(a) -> int:
    doc = json.loads(Path(a.tokens).read_text(encoding="utf-8"))
    segs = doc["segments"]
    stock, truth = Path(a.stock), Path(a.truth)
    tests = []
    for t in a.test:
        label, _, path = t.partition("=")
        if not path:
            raise SystemExit(f"--test wants label=dir, got {t!r}")
        tests.append((label, Path(path)))
    names = [s["name"] for s in segs]
    for d in [stock, truth] + [p for _, p in tests]:
        missing = [n for n in names if not ((d / f"{n}.logprobs.f16").exists() and (d / f"{n}.meta.json").exists())]
        if missing:
            print(f"[gate_q35] {d}: missing dumps for {missing}")
            return 2

    print("[gate_q35] noise: truth vs stock bf16 ...", flush=True)
    noise = compare_all(segs, truth, stock, a.row_chunk)
    noise_g = {g: pooled(noise, g) for g in GATED + REPORTED}
    noise_b = bucketed(noise)
    report = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "tokens": str(a.tokens),
              "stock": str(stock), "truth": str(truth),
              "rules": {"gate_a": f"pooled mean KL(bf16||r4dx) <= {GATE_A_KL_MAX} and top-1 >= {GATE_A_TOP1_MIN}%",
                        "gate_b": f"per group KL(truth||r4dx) <= {GATE_B_FACTOR} * N + {GATE_B_SLACK} and top-1 >= N_top1 - "
                                  f"{GATE_B_TOP1_SLACK_PTS} pt, N = KL(truth||stock bf16)"},
              "noise": {g: brief(v) for g, v in noise_g.items() if v}, "noise_by_position": noise_b,
              "noise_segments": {n: {"rows": r["rows"], "mean_kl": r["mean_kl"], "top1_pct": r["top1_agreement_pct"]} for n, r in noise.items()},
              "tests": {}}
    ok_all = True
    for label, tdir in tests:
        print(f"[gate_q35] {label}: vs stock bf16 ...", flush=True)
        va = compare_all(segs, stock, tdir, a.row_chunk)
        print(f"[gate_q35] {label}: vs truth fp32 ...", flush=True)
        vb = compare_all(segs, truth, tdir, a.row_chunk)
        pa = {g: pooled(va, g) for g in GATED + REPORTED}
        pb = {g: pooled(vb, g) for g in GATED + REPORTED}
        gate_a = {"pooled": brief(pa["ALL"]), "pass": bool(pa["ALL"]["mean_kl"] <= GATE_A_KL_MAX and pa["ALL"]["top1_pct"] >= GATE_A_TOP1_MIN)}
        gb = {}
        for g in GATED + REPORTED:
            if pb[g] is None:
                continue
            n = noise_g[g]
            thr = {"kl_max": GATE_B_FACTOR * n["mean_kl"] + GATE_B_SLACK, "top1_min_pct": n["top1_pct"] - GATE_B_TOP1_SLACK_PTS}
            gb[g] = {**brief(pb[g]), "noise_mean_kl": n["mean_kl"], "noise_top1_pct": n["top1_pct"], **thr,
                     "counted": g in GATED, "pass": bool(pb[g]["mean_kl"] <= thr["kl_max"] and pb[g]["top1_pct"] >= thr["top1_min_pct"])}
        gate_b = {"groups": gb, "pass": all(v["pass"] for v in gb.values() if v["counted"])}
        seg_rows = {n: {"group": r["_group"], "rows": r["rows"], "vs_bf16_kl": r["mean_kl"], "vs_bf16_top1": r["top1_agreement_pct"],
                        "vs_truth_kl": vb[n]["mean_kl"], "vs_truth_top1": vb[n]["top1_agreement_pct"],
                        "noise_kl": noise[n]["mean_kl"], "vs_bf16_max_kl": r["max_kl"], "vs_bf16_kl_gt_1": r["positions_kl_gt_1"]}
                    for n, r in va.items()}
        report["tests"][label] = {
            "dir": str(tdir), "gate_a": gate_a, "gate_b": gate_b,
            "vs_bf16": {g: brief(v) for g, v in pa.items() if v}, "vs_truth": {g: brief(v) for g, v in pb.items() if v},
            "vs_bf16_by_position": bucketed(va), "vs_truth_by_position": bucketed(vb), "segments": seg_rows,
            "worst_rows_vs_bf16": kl_report.worst_rows([dict(r, _ids=segs[names.index(n)]["token_ids"]) for n, r in va.items()], 8)}
        ok_all &= gate_a["pass"] and gate_b["pass"]
    out = Path(a.out)
    out.with_suffix(".json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    out.with_suffix(".md").write_text(markdown(report), encoding="utf-8")
    print(markdown(report))
    print(f"[gate_q35] wrote {out.with_suffix('.json')} and {out.with_suffix('.md')}")
    return 0 if ok_all else 1


def markdown(rep: dict) -> str:
    L = ["# Ornith-1.5-122B trellis container: Rung 4 KL gate", "", f"- gate A: {rep['rules']['gate_a']}", f"- gate B: {rep['rules']['gate_b']}",
         f"- stock bf16 dumps: `{rep['stock']}`; fp32 truth dumps: `{rep['truth']}`", ""]
    L += ["## Noise: KL(truth fp32 || stock HF bf16)", "", "| group | rows | mean KL | p99 KL | top-1 |", "|---|---:|---:|---:|---:|"]
    for g, v in rep["noise"].items():
        L.append(f"| {g} | {v['rows']} | {v['mean_kl']:.5f} | {v['p99_kl']:.4f} | {v['top1_pct']:.2f}% |")
    for label, t in rep["tests"].items():
        L += ["", f"## {label}", "", f"container dumps: `{t['dir']}`", "",
              f"**Gate A** (vs HF bf16, pooled): mean KL {t['gate_a']['pooled']['mean_kl']:.5f} (max {GATE_A_KL_MAX}), top-1 "
              f"{t['gate_a']['pooled']['top1_pct']:.2f}% (min {GATE_A_TOP1_MIN}) over {t['gate_a']['pooled']['rows']} rows: "
              f"**{'PASS' if t['gate_a']['pass'] else 'FAIL'}**", "",
              "| group (vs bf16) | rows | mean KL | median | p99 | max | top-1 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for g, v in t["vs_bf16"].items():
            L.append(f"| {g} | {v['rows']} | {v['mean_kl']:.5f} | {v['median_kl']:.5f} | {v['p99_kl']:.4f} | {v['max_kl']:.3f} | {v['top1_pct']:.2f}% |")
        L += ["", f"**Gate B** (vs fp32 truth, noise-relative): **{'PASS' if t['gate_b']['pass'] else 'FAIL'}**", "",
              "| group | rows | KL(truth\\|\\|r4dx) | limit 1.5N+0.005 | N | top-1 | min top-1 | N top-1 | |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
        for g, v in t["gate_b"]["groups"].items():
            tag = ("PASS" if v["pass"] else "FAIL") + ("" if v["counted"] else " (reported)")
            L.append(f"| {g} | {v['rows']} | {v['mean_kl']:.5f} | {v['kl_max']:.5f} | {v['noise_mean_kl']:.5f} | {v['top1_pct']:.2f}% | "
                     f"{v['top1_min_pct']:.2f}% | {v['noise_top1_pct']:.2f}% | {tag} |")
        L += ["", "| position in sequence | rows | KL vs bf16 | top-1 vs bf16 | KL vs truth | top-1 vs truth | noise KL | noise top-1 |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
        nz = {(b["from"]): b for b in rep["noise_by_position"]}
        vt = {(b["from"]): b for b in t["vs_truth_by_position"]}
        for b in t["vs_bf16_by_position"]:
            hi = "+" if b["to"] is None else f"-{b['to']}"
            c, n = vt.get(b["from"]), nz.get(b["from"])
            L.append(f"| {b['from']}{hi} | {b['rows']} | {b['mean_kl']:.5f} | {b['top1_pct']:.2f}% | "
                     f"{c['mean_kl']:.5f} | {c['top1_pct']:.2f}% | {n['mean_kl']:.5f} | {n['top1_pct']:.2f}% |" if c and n else
                     f"| {b['from']}{hi} | {b['rows']} | {b['mean_kl']:.5f} | {b['top1_pct']:.2f}% | | | | |")
        L += ["", "| segment | group | rows | KL vs bf16 | top-1 vs bf16 | KL vs truth | noise KL | max KL vs bf16 | KL>1 |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
        for n, r in t["segments"].items():
            L.append(f"| {n} | {r['group']} | {r['rows']} | {r['vs_bf16_kl']:.5f} | {r['vs_bf16_top1']:.2f}% | {r['vs_truth_kl']:.5f} | "
                     f"{r['noise_kl']:.5f} | {r['vs_bf16_max_kl']:.3f} | {r['vs_bf16_kl_gt_1']} |")
    return "\n".join(L) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("eval")
    p.add_argument("--tokens", required=True)
    p.add_argument("--stock", required=True, help="HF bf16 dumps (ref_run.py logprobs --dtype bf16)")
    p.add_argument("--truth", required=True, help="HF fp32 dumps (ref_run.py logprobs --dtype fp32)")
    p.add_argument("--test", action="append", required=True, help="label=dir of r4dx dumps (repeatable)")
    p.add_argument("--out", required=True, help="report path without extension (.json and .md are written)")
    p.add_argument("--row-chunk", type=int, default=256)
    a = ap.parse_args()
    return prep(a.out) if a.cmd == "prep" else evaluate(a)


if __name__ == "__main__":
    raise SystemExit(main())
