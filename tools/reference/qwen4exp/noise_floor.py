"""QSA tie-rule noise floor of the qwen4_exp gate (plan Q23) and consistency checks of the real-model reference dumps.

    KL(P_hf || P_recent) and top-1 agreement per segment, `kl_report.py`'s own functions (fp64 over the full vocab):

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen4exp\\noise_floor.py `
        --hf-dir E:\\models\\r4dx\\q4exp\\ref\\ref-hf --recent-dir E:\\models\\r4dx\\q4exp\\ref\\canon `
        --out E:\\models\\r4dx\\q4exp\\ref\\noise_floor

ref side = the dump made with the HF-exact tie rule (`qsa_indexer` tie_rule "hf", HF's torch.topk choice reproduced),
test side = the dump made with the rule r4dx implements ("recent"). The gate compares r4dx against "recent"; this is
the part of any measured KL that is rule noise, not quantization. Exit status 2 (and a banner) when the mean KL over
all rows exceeds `--stop-mean-kl` (0.01): the gate thresholds would need revisiting with the user.

    noise_floor.py prefix --long-dir <dir of probe_32k> --long-name probe_32k --short-dir <dir of probe_8k> --short-name probe_8k --tokens <file>

compares the first rows of two dumps of the same token prefix (the 8k probe is the first 8192 ids of the 32k probe): a
causal model gives the same distributions for those rows, so the KL is the numerical difference of the two runs
(different sequence lengths -> different sdpa / GEMM shapes), a sanity check of the long-context path.

Only numpy (and kl_report); no GPU.
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

KL_CORPUS_DIR = HERE.parent / "kl_corpus"
STOP_MEAN_KL = 0.01


def noise_floor(hf_dir: Path, recent_dir: Path, tokens_path: Path, segments: list[str] | None = None,
                row_chunk: int = 32) -> dict:
    """KL(hf || recent) per segment and over all rows, from two dump directories (`kl_report` format)."""
    doc = json.loads(Path(tokens_path).read_text(encoding="utf-8"))
    segs = [s for s in doc["segments"] if segments is None or s["name"] in segments]
    if not segs:
        raise SystemExit(f"no segments selected in {tokens_path}")
    results = [kl_report.compare_segment(s["name"], s["token_ids"], Path(hf_dir), Path(recent_dir), row_chunk, strict=True)
               for s in segs]
    total = kl_report.overall(results)
    return {"results": results, "overall": total, "markdown": kl_report.markdown(results, total, Path(hf_dir), Path(recent_dir))}


def plain(results: list[dict]) -> list[dict]:
    return [{k: v for k, v in r.items() if not k.startswith("_")} for r in results]


def prefix_check(long_dir: Path, long_name: str, short_dir: Path, short_name: str, token_ids: list[int],
                 row_chunk: int = 32) -> dict:
    """KL(short || long) over the rows of the short dump (it covers the first len(short) - 1 rows of the long one)."""
    sm, lm = kl_report.read_meta(short_dir, short_name), kl_report.read_meta(long_dir, long_name)
    vocab = int(sm["V"])
    rows = int(sm["rows"])
    if int(lm["V"]) != vocab or int(lm["rows"]) < rows:
        raise SystemExit(f"cannot compare {short_name} ({sm['rows']} rows) with {long_name} ({lm['rows']} rows)")
    short = kl_report.open_rows(short_dir, short_name, rows, vocab)
    long_ = kl_report.open_rows(long_dir, long_name, int(lm["rows"]), vocab)
    nxt = np.asarray(token_ids[1: rows + 1], dtype=np.int64)
    kl = np.empty(rows)
    top1 = np.empty(rows, dtype=bool)
    for s in range(0, rows, row_chunk):
        e = min(s + row_chunk, rows)
        st = kl_report.chunk_stats(short[s:e], long_[s:e], nxt[s:e])
        kl[s:e], top1[s:e] = st["kl"], st["top1_agree"]
    return {"rows": rows, "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)), "p99_kl": float(np.percentile(kl, 99)),
            "max_kl": float(kl.max()), "max_kl_position": int(kl.argmax()), "top1_agreement_pct": 100.0 * float(top1.mean()),
            "_kl": kl, "_top1": top1}


def shape_noise(short_dir: Path, short_tokens: Path, long_dir: Path, long_tokens: Path, suffix: str = "_p1",
                segments: list[str] | None = None, row_chunk: int = 32) -> dict:
    """Pure numerical-noise control: every segment `<name>` of `short_tokens` was also run as `<name><suffix>` (the same ids
    plus the true next token(s), same tie rule). KL(short || long) over the short rows, per segment and pooled."""
    sdoc = json.loads(Path(short_tokens).read_text(encoding="utf-8"))
    ldoc = {s["name"]: s["token_ids"] for s in json.loads(Path(long_tokens).read_text(encoding="utf-8"))["segments"]}
    per = {}
    for seg in sdoc["segments"]:
        if segments is not None and seg["name"] not in segments:
            continue
        long_ids = ldoc[seg["name"] + suffix]
        if long_ids[: len(seg["token_ids"])] != seg["token_ids"]:
            raise SystemExit(f"{seg['name']}{suffix} does not start with {seg['name']}")
        per[seg["name"]] = prefix_check(Path(long_dir), seg["name"] + suffix, Path(short_dir), seg["name"], long_ids, row_chunk)
    kl = np.concatenate([r["_kl"] for r in per.values()])
    top1 = np.concatenate([r["_top1"] for r in per.values()])
    pooled = {"rows": int(kl.size), "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)), "p99_kl": float(np.percentile(kl, 99)),
              "max_kl": float(kl.max()), "top1_agreement_pct": 100.0 * float(top1.mean())}
    return {"segments": {k: {a: b for a, b in v.items() if not a.startswith("_")} for k, v in per.items()}, "pooled": pooled}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    ap.add_argument("--hf-dir", type=Path)
    ap.add_argument("--recent-dir", type=Path)
    ap.add_argument("--tokens", type=Path, default=KL_CORPUS_DIR / "tokens_canon.json")
    ap.add_argument("--segment", action="append", default=None)
    ap.add_argument("--row-chunk", type=int, default=32)
    ap.add_argument("--stop-mean-kl", type=float, default=STOP_MEAN_KL)
    ap.add_argument("--out", type=Path, default=None, help="write <out>.json and <out>.md")
    p = sub.add_parser("shape-noise", help="KL between each segment and its `_p1` re-run with one more token (pure bf16 shape noise)")
    p.add_argument("--short-dir", type=Path, required=True)
    p.add_argument("--short-tokens", type=Path, required=True)
    p.add_argument("--long-dir", type=Path, required=True)
    p.add_argument("--long-tokens", type=Path, required=True)
    p.add_argument("--suffix", default="_p1")
    p.add_argument("--out", type=Path, default=None)
    p = sub.add_parser("prefix")
    p.add_argument("--long-dir", type=Path, required=True)
    p.add_argument("--long-name", required=True)
    p.add_argument("--short-dir", type=Path, required=True)
    p.add_argument("--short-name", required=True)
    p.add_argument("--tokens", type=Path, required=True, help="tokens file holding the long segment")
    p.add_argument("--short-tokens", type=Path, default=None, help="tokens file holding the short segment (default: --tokens)")
    p.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    if args.cmd == "shape-noise":
        res = shape_noise(args.short_dir, args.short_tokens, args.long_dir, args.long_tokens, args.suffix)
        print(json.dumps(res, indent=1))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "short": str(args.short_dir),
                                            "long": str(args.long_dir), **res}, indent=1), encoding="utf-8")
        return 0
    if args.cmd == "prefix":
        doc = json.loads(args.tokens.read_text(encoding="utf-8"))
        ids = next(s["token_ids"] for s in doc["segments"] if s["name"] == args.long_name)
        sdoc = json.loads(args.short_tokens.read_text(encoding="utf-8")) if args.short_tokens else doc
        short_ids = next(s["token_ids"] for s in sdoc["segments"] if s["name"] == args.short_name)
        if ids[: len(short_ids)] != short_ids:
            raise SystemExit(f"{args.short_name} is not a prefix of {args.long_name}")
        res = prefix_check(args.long_dir, args.long_name, args.short_dir, args.short_name, ids, args.row_chunk)
        res = {k: v for k, v in res.items() if not k.startswith("_")}
        print(json.dumps(res, indent=1))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "long": str(args.long_dir),
                                            "short": str(args.short_dir), **res}, indent=1), encoding="utf-8")
        return 0

    if args.hf_dir is None or args.recent_dir is None:
        raise SystemExit("noise_floor.py needs --hf-dir and --recent-dir (or the `prefix` subcommand)")
    nf = noise_floor(args.hf_dir, args.recent_dir, args.tokens, args.segment, args.row_chunk)
    total = nf["overall"]
    print(nf["markdown"])
    stop = total["mean_kl"] > args.stop_mean_kl
    banner = ""
    if stop:
        banner = (f"\n!!! STOP: the tie-rule noise floor alone is mean KL {total['mean_kl']:.5f} > {args.stop_mean_kl} "
                  f"(top-1 {total['top1_agreement_pct']:.2f}%): the gate thresholds (plan Q10/Q23) need revisiting with the user. !!!")
        print(banner)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "what": "KL(hf-rule || recent-rule), plan Q23 noise floor",
                   "hf_dir": str(args.hf_dir), "recent_dir": str(args.recent_dir), "tokens_file": str(args.tokens),
                   "stop_mean_kl": args.stop_mean_kl, "exceeds_stop": stop, "segments": plain(nf["results"]), "overall": total}
        args.out.with_suffix(".json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
        args.out.with_suffix(".md").write_text(nf["markdown"] + banner + "\n", encoding="utf-8")
        print(f"[noise_floor] wrote {args.out.with_suffix('.json')} and {args.out.with_suffix('.md')}")
    return 2 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
