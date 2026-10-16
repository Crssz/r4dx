"""Both qwen4_exp quality gates over one r4dx teacher-forced dump directory (plan 4.10 / Q10, REAL_REF.md "Proposed gate").

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen4exp\\gate_eval.py --test-dir <dump dir> [--scopes canon,probe8k,probe32k]
    ... gate_eval.py --self-test                       # CPU, no GPU: the reference dumps against themselves

Gate A (the ORIGINAL gate): KL(bf16 reference || r4dx), pooled over the scope's rows, mean KL <= 0.02 and top-1 >= 93 %.
    canon is the gate; probe8k / probe32k are reported against the same thresholds (never counted).
Gate B (the noise-relative gate, REAL_REF.md section 5): KL(truth-fp32 || r4dx) <= 1.5 * N_KL + 0.005 and top-1 >= N_top1 - 1 point,
    per group english / thai / code / ALL, N = the stock-bf16 noise term `truth_gate.py` computes (cached in
    <root>/noise_floor/truth_gate_exact.json, recomputed through truth_gate's own functions when that file is missing or
    --recompute-noise is given). probe8k is reported only; probe32k has no truth dump.

Everything is `kl_report` arithmetic (fp64 over the full vocabulary) through `truth_gate`'s functions. Exit status: 0 = every
counted gate passed, 1 = a counted gate failed, 2 = a dump is missing / inconsistent.
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
sys.path.insert(0, str(HERE))

import kl_report  # noqa: E402
import truth_gate  # noqa: E402
from qwen4exp.common_qwen4exp import KL_CORPUS_DIR  # noqa: E402

DEFAULT_ROOT = Path(r"E:\models\r4dx\q4exp\ref")
GATE_A_KL_MAX = 0.02
GATE_A_TOP1_MIN = 93.0
GROUP_ORDER = ["english", "thai", "code", "ALL"]
#: REAL_REF.md section 1 (a): KL(truth || bf16 reference) on canon, what --self-test must reproduce.
REAL_REF_CANON = {"mean_kl": 0.01386, "top1_pct": 95.19, "rows": 4092}


def scopes(root: Path, truth: str) -> dict:
    """name -> where the tokens, the bf16 reference, the truth and the dump segments are."""
    probes = root / "probes"
    return {
        "canon": {"tokens": KL_CORPUS_DIR / "tokens_canon.json", "segments": None, "bf16": root / "canon",
                  "truth": root / truth, "gate_a_counts": True, "gate_b_counts": True, "noise_key": "canon",
                  "p1_dir": root / "canon_p1", "p1_tokens": root / "canon_p1" / "tokens_canon_p1.json"},
        "probe8k": {"tokens": probes / "tokens_probes.json", "segments": ["probe_8k"], "bf16": probes / "recent",
                    "truth": root / truth, "gate_a_counts": False, "gate_b_counts": False, "noise_key": "probe_8k",
                    "p1_dir": probes / "recent", "p1_tokens": probes / "tokens_probes_p1.json"},
        "probe32k": {"tokens": probes / "tokens_probes.json", "segments": ["probe_32k"], "bf16": probes / "recent",
                     "truth": None, "gate_a_counts": False, "gate_b_counts": False, "noise_key": None},
    }


def seg_names(sc: dict) -> list[str]:
    doc = json.loads(Path(sc["tokens"]).read_text(encoding="utf-8"))
    return [s["name"] for s in doc["segments"] if sc["segments"] is None or s["name"] in sc["segments"]]


def brief(res: dict) -> dict:
    return {"rows": res["rows"], "mean_kl": res["mean_kl"], "median_kl": res["median_kl"], "p99_kl": res["p99_kl"],
            "max_kl": res["max_kl"], "top1_pct": res["top1_agreement_pct"], "positions_kl_gt_1": res.get("positions_kl_gt_1", 0)}


def gate_a(results: dict) -> dict:
    """The original gate over {segment: compare_segment result}: pooled mean KL / top-1 against 0.02 / 93 %."""
    pooled = brief(truth_gate.pool_any([results]))
    return {"kl_max": GATE_A_KL_MAX, "top1_min_pct": GATE_A_TOP1_MIN, "pooled": pooled,
            "segments": {k: brief(v) for k, v in results.items()},
            "pass": bool(pooled["mean_kl"] <= GATE_A_KL_MAX and pooled["top1_pct"] >= GATE_A_TOP1_MIN)}


def gate_b(results: dict, thresholds: dict) -> dict:
    """The noise-relative gate over {segment: result vs truth}; thresholds = {group: {kl_max, top1_min_pct, noise_*}}."""
    groups = truth_gate.group_noise([results])
    out = {}
    for g in GROUP_ORDER:
        if g not in groups or g not in thresholds:
            continue
        s, t = groups[g], thresholds[g]
        out[g] = {"rows": s["rows"], "mean_kl": s["mean_kl"], "top1_pct": s["top1_agreement_pct"], "p99_kl": s["p99_kl"],
                  "max_kl": s["max_kl"], "noise_mean_kl": t["noise_mean_kl"], "noise_top1_pct": t["noise_top1_pct"],
                  "kl_max": t["kl_max"], "top1_min_pct": t["top1_min_pct"],
                  "pass": bool(s["mean_kl"] <= t["kl_max"] and s["top1_agreement_pct"] >= t["top1_min_pct"])}
    return {"groups": out, "segments": {k: brief(v) for k, v in results.items()},
            "pass": bool(out) and all(v["pass"] for v in out.values())}


def compute_noise(root: Path, truth: str, scope: str, row_chunk: int) -> dict:
    """truth_gate.py's stock-noise thresholds of one scope, from the dumps (what --recompute-noise / a missing cache use)."""
    sc = scopes(root, truth)[scope]
    a = truth_gate.as_dict(truth_gate.compare(sc["truth"], sc["bf16"], sc["tokens"], sc["segments"], row_chunk))
    b = truth_gate.prefix_results(sc["truth"], sc["tokens"], sc["p1_dir"], sc["p1_tokens"], sc["segments"], row_chunk)
    return truth_gate.gate_rows(truth_gate.group_noise([a, b]))


def noise_thresholds(root: Path, truth: str, scope: str, row_chunk: int, recompute: bool) -> tuple[dict, str]:
    cache = root / "noise_floor" / "truth_gate_exact.json"
    key = scopes(root, truth)[scope]["noise_key"]
    if not recompute and truth == "truth-fp32" and cache.exists():
        doc = json.loads(cache.read_text(encoding="utf-8"))
        if key in doc.get("scopes", {}):
            return doc["scopes"][key]["gate_proposal"], f"cached {cache}"
    return compute_noise(root, truth, scope, row_chunk), "recomputed from the dumps (truth_gate functions)"


def check_dump(test_dir: Path, names: list[str]) -> list[str]:
    return [n for n in names if not ((test_dir / f"{n}.logprobs.f16").exists() and (test_dir / f"{n}.meta.json").exists())]


def evaluate(test_dir: Path, scope_names: list[str], root: Path, truth: str, row_chunk: int, recompute_noise: bool,
             custom: dict | None = None) -> dict:
    rep: dict = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "test_dir": str(test_dir), "root": str(root),
                 "truth": truth, "gate_a_rule": f"pooled mean KL <= {GATE_A_KL_MAX} and top-1 >= {GATE_A_TOP1_MIN}%",
                 "scopes": {}}
    table = scopes(root, truth)
    if custom:
        table = {"custom": {"tokens": custom["tokens"], "segments": None, "bf16": custom["ref_dir"], "truth": None,
                            "gate_a_counts": True, "gate_b_counts": False, "noise_key": None}}
        scope_names = ["custom"]
    for name in scope_names:
        sc = table[name]
        names = seg_names(sc)
        missing = check_dump(test_dir, names)
        if missing:
            raise FileNotFoundError(f"{test_dir}: no dump for segment(s) {missing} of scope {name}")
        entry: dict = {"segments_run": names}
        print(f"[gate_eval] {name}: KL(bf16 reference || r4dx) ...", flush=True)
        a = truth_gate.as_dict(truth_gate.compare(sc["bf16"], test_dir, sc["tokens"], sc["segments"], row_chunk))
        entry["gate_a"] = {**gate_a(a), "counts": sc["gate_a_counts"], "ref_dir": str(sc["bf16"])}
        if sc["truth"] is not None and (sc["truth"] / f"{names[0]}.logprobs.f16").exists():
            print(f"[gate_eval] {name}: KL(truth || r4dx) ...", flush=True)
            thr, src = noise_thresholds(root, truth, name, row_chunk, recompute_noise)
            b = truth_gate.as_dict(truth_gate.compare(sc["truth"], test_dir, sc["tokens"], sc["segments"], row_chunk))
            entry["gate_b"] = {**gate_b(b, thr), "counts": sc["gate_b_counts"], "noise_source": src, "truth_dir": str(sc["truth"])}
        else:
            entry["gate_b"] = None
        rep["scopes"][name] = entry
    counted = [(n, k) for n, e in rep["scopes"].items() for k in ("gate_a", "gate_b") if e[k] is not None and e[k]["counts"]]
    rep["pass_a"] = all(rep["scopes"][n]["gate_a"]["pass"] for n, k in counted if k == "gate_a") if any(k == "gate_a" for _, k in counted) else None
    rep["pass_b"] = all(rep["scopes"][n]["gate_b"]["pass"] for n, k in counted if k == "gate_b") if any(k == "gate_b" for _, k in counted) else None
    rep["pass"] = all(v for v in (rep["pass_a"], rep["pass_b"]) if v is not None) and any(v is not None for v in (rep["pass_a"], rep["pass_b"]))
    return rep


def verdict(ok: bool, counts: bool) -> str:
    return ("PASS" if ok else "FAIL") if counts else ("within" if ok else "outside") + " (report)"


def render_markdown(rep: dict) -> str:
    L = [f"# qwen4_exp gate report", "", f"* dump: `{rep['test_dir']}`", f"* generated: {rep['generated_at']}",
         f"* gate A (original): {rep['gate_a_rule']} vs the bf16 reference",
         f"* gate B (noise-relative): vs `{rep['truth']}`, KL <= 1.5 x N + 0.005 and top-1 >= N - 1 point per group", ""]
    L += ["## Verdict", "", "| gate | scope | result | detail |", "|---|---|---|---|"]
    for name, e in rep["scopes"].items():
        a = e["gate_a"]
        L.append(f"| A original | {name} | **{verdict(a['pass'], a['counts'])}** | KL {a['pooled']['mean_kl']:.5f} (<= {a['kl_max']}), "
                 f"top-1 {a['pooled']['top1_pct']:.2f}% (>= {a['top1_min_pct']}%), {a['pooled']['rows']} rows |")
        b = e["gate_b"]
        if b is None:
            L.append(f"| B noise-relative | {name} | n/a | no truth dump for this scope |")
            continue
        for g in GROUP_ORDER:
            if g in b["groups"]:
                r = b["groups"][g]
                L.append(f"| B noise-relative | {name} / {g} | **{verdict(r['pass'], b['counts'])}** | KL {r['mean_kl']:.5f} (<= {r['kl_max']:.5f}), "
                         f"top-1 {r['top1_pct']:.2f}% (>= {r['top1_min_pct']:.2f}%), noise N {r['noise_mean_kl']:.5f} / {r['noise_top1_pct']:.2f}%, {r['rows']} rows |")
    L += ["", f"Overall: gate A **{_pf(rep['pass_a'])}**, gate B **{_pf(rep['pass_b'])}**.", "", "## Per segment"]
    for name, e in rep["scopes"].items():
        L += ["", f"### {name}: vs the bf16 reference (gate A)", "", "| segment | rows | mean KL | p99 | max | top-1 % | rows KL>1 |", "|---|---:|---:|---:|---:|---:|---:|"]
        for s, v in e["gate_a"]["segments"].items():
            L.append(f"| {s} | {v['rows']} | {v['mean_kl']:.5f} | {v['p99_kl']:.4f} | {v['max_kl']:.3f} | {v['top1_pct']:.2f} | {v['positions_kl_gt_1']} |")
        if e["gate_b"]:
            L += ["", f"### {name}: vs the fp32 truth (gate B)", "", "| segment | rows | mean KL | p99 | max | top-1 % | rows KL>1 |", "|---|---:|---:|---:|---:|---:|---:|"]
            for s, v in e["gate_b"]["segments"].items():
                L.append(f"| {s} | {v['rows']} | {v['mean_kl']:.5f} | {v['p99_kl']:.4f} | {v['max_kl']:.3f} | {v['top1_pct']:.2f} | {v['positions_kl_gt_1']} |")
            L.append(f"\nnoise term: {e['gate_b']['noise_source']}")
    return "\n".join(L) + "\n"


def _pf(v) -> str:
    return "n/a" if v is None else ("PASS" if v else "FAIL")


def print_table(rep: dict) -> None:
    print("\n" + render_markdown(rep))


def self_test(root: Path, truth: str, row_chunk: int, recompute_noise: bool) -> int:
    """No GPU: the reference dumps stand in for an r4dx dump. Asserts REAL_REF.md section 1 (a) and the identity cases."""
    canon_t = root / truth
    bf16 = root / "canon"
    failures = []

    def expect(cond: bool, msg: str) -> None:
        print(f"[self-test] {'ok  ' if cond else 'FAIL'} {msg}")
        if not cond:
            failures.append(msg)

    print("[self-test] 1/3 truth-fp32 as the 'r4dx' dump: gate B must be exactly 0 KL / 100% top-1 (identity)", flush=True)
    rep1 = evaluate(canon_t, ["canon"], root, truth, row_chunk, recompute_noise)
    b = rep1["scopes"]["canon"]["gate_b"]
    expect(b is not None and all(g["mean_kl"] < 1e-9 and g["top1_pct"] == 100.0 for g in b["groups"].values()), "truth vs truth: KL 0, top-1 100% in every group")
    expect(bool(b["pass"]), "truth vs truth passes gate B")
    a = rep1["scopes"]["canon"]["gate_a"]["pooled"]
    print(f"[self-test]      (truth as test vs the bf16 reference, gate A numbers: KL {a['mean_kl']:.5f}, top-1 {a['top1_pct']:.2f}%, {a['rows']} rows)")

    print("[self-test] 2/3 bf16 reference as the 'r4dx' dump against the truth (REAL_REF.md section 1 (a): 0.01386 / 95.19%)", flush=True)
    rep2 = evaluate(bf16, ["canon"], root, truth, row_chunk, recompute_noise)
    all_ = rep2["scopes"]["canon"]["gate_b"]["groups"]["ALL"]
    expect(all_["rows"] == REAL_REF_CANON["rows"], f"canon ALL rows {all_['rows']} == {REAL_REF_CANON['rows']}")
    expect(round(all_["mean_kl"], 5) == REAL_REF_CANON["mean_kl"], f"canon KL(truth||bf16) {all_['mean_kl']:.5f} == {REAL_REF_CANON['mean_kl']}")
    expect(round(all_["top1_pct"], 2) == REAL_REF_CANON["top1_pct"], f"canon top-1 {all_['top1_pct']:.2f}% == {REAL_REF_CANON['top1_pct']}%")
    expect(bool(rep2["scopes"]["canon"]["gate_b"]["pass"]), "the stock bf16 reference passes gate B (its own noise term + 0.5 x N + 0.005)")
    ra = rep2["scopes"]["canon"]["gate_a"]
    expect(ra["pooled"]["mean_kl"] < 1e-9 and ra["pass"], "bf16 vs bf16: gate A KL 0, passes")
    print_table(rep2)
    print("[self-test] 3/3 thresholds: ALL kl_max / top1_min vs REAL_REF.md section 5 (0.02610 / 94.23%)", flush=True)
    thr = rep2["scopes"]["canon"]["gate_b"]["groups"]["ALL"]
    expect(round(thr["kl_max"], 5) == 0.02610 and round(thr["top1_min_pct"], 2) == 94.23,
           f"canon ALL gate {thr['kl_max']:.5f} / {thr['top1_min_pct']:.2f}%")
    print(f"[self-test] {'PASS' if not failures else 'FAIL: ' + '; '.join(failures)}")
    return 0 if not failures else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-dir", type=Path, help="the r4dx dump directory (<segment>.logprobs.f16 + .meta.json)")
    ap.add_argument("--scopes", default="canon", help="comma list of canon,probe8k,probe32k")
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--truth", default="truth-fp32")
    ap.add_argument("--row-chunk", type=int, default=32)
    ap.add_argument("--recompute-noise", action="store_true", help="recompute the stock-noise thresholds instead of the cached ones")
    ap.add_argument("--out-json", type=Path)
    ap.add_argument("--out-md", type=Path)
    ap.add_argument("--custom-tokens", type=Path, help="plumbing mode: this tokens.json with --custom-ref, gate A only")
    ap.add_argument("--custom-ref", type=Path, help="plumbing mode: the reference dump directory of --custom-tokens")
    ap.add_argument("--self-test", action="store_true", help="CPU: the reference dumps against themselves, REAL_REF.md numbers")
    args = ap.parse_args(argv)
    try:
        if args.self_test:
            return self_test(args.root, args.truth, args.row_chunk, args.recompute_noise)
        if args.test_dir is None:
            ap.error("--test-dir is required (or --self-test)")
        custom = {"tokens": args.custom_tokens, "ref_dir": args.custom_ref} if args.custom_tokens else None
        rep = evaluate(args.test_dir, [s for s in args.scopes.split(",") if s], args.root, args.truth, args.row_chunk,
                       args.recompute_noise, custom)
    except (FileNotFoundError, SystemExit) as e:
        print(f"[gate_eval] ERROR {e}", file=sys.stderr)
        return 2
    print_table(rep)
    if args.out_json:
        args.out_json.write_text(json.dumps(rep, indent=1), encoding="utf-8")
        print(f"[gate_eval] wrote {args.out_json}")
    if args.out_md:
        args.out_md.write_text(render_markdown(rep), encoding="utf-8")
        print(f"[gate_eval] wrote {args.out_md}")
    return 0 if rep["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
