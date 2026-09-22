"""tools/reference/kl_report.py

Pairs the bf16 reference's per-position log-probabilities (`full_logits_golden.py`, source
"reference") with the r4dx engine's own dump of the same token sequences (source "r4dx") and
reports how far the quantized engine has drifted from the reference, in nats.

For every position i of every segment (row i = `log p(next token | token_ids[0..i])`), in fp64:

    KL(P_ref || Q_test) = sum_v p_ref[v] * (logp_ref[v] - logq_test[v])       over the FULL vocab
    top-1 agreement      = argmax_v logp_ref[v] == argmax_v logq_test[v]
    top-5 containment    = argmax_v logp_ref[v] is among the test's five largest
    NLL_ref / NLL_test   = -logp[token_ids[i+1]]

and aggregates per segment and overall: mean / median / 99th-percentile / max KL (with the position
and token where the max occurs), top-1 agreement %, top-5 containment %, each model's perplexity
over the corpus, and the number of positions with KL > 1.0 nat.

Both files are `[T-1, V]` fp16 -- at V=248320 that is 488 MiB per 1024-token segment, so neither is
ever loaded whole, let alone widened to fp64 whole: `numpy.memmap` + a row chunk (`--row-chunk`,
default 32 rows = 2 x 60 MiB of fp64 scratch) walks them in lockstep.

Usage:

    <venv>\\Scripts\\python.exe tools\\reference\\kl_report.py `
        --ref-dir tools\\reference\\kl_out\\ref --test-dir tools\\reference\\kl_out\\r4dx `
        --tokens tools\\reference\\kl_corpus\\tokens.json --out tools\\reference\\kl_out\\kl.json

    <venv>\\Scripts\\python.exe tools\\reference\\kl_report.py --self-test

No GPU, no checkpoint, no torch -- numpy only.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

#: Must match `full_logits_golden.LOGPROB_CLAMP` / the engine-side writer: log-probabilities are
#: floored here before the fp16 cast, because fp16 cannot represent anything below about -65504.
LOGPROB_CLAMP = -1e4

#: A position whose KL exceeds this is counted separately -- 1 nat is already a gross disagreement
#: (the reference's own distribution is ~e times less likely under the test model).
KL_ALARM = 1.0


def token_ids_sha256(token_ids) -> str:
    """Identical to `full_logits_golden.token_ids_sha256` and the C++ `TokenIdsSha256`."""
    payload = json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------------------------
# The core: one chunk of rows, in fp64
# --------------------------------------------------------------------------------------------


def chunk_stats(logp_ref: np.ndarray, logq_test: np.ndarray, next_ids: np.ndarray) -> dict:
    """Per-row statistics for a block of rows. Inputs are `[rows, V]` (any float dtype; widened to
    fp64 here) and `next_ids` is `[rows]` -- the token each row is predicting.

    Returns per-row arrays: `kl`, `top1_agree`, `top5_contain`, `nll_ref`, `nll_test`,
    `ref_argmax`, `p_ref_sum` (a diagnostic: the fp16 round trip means `sum_v exp(logp_ref[v])` is
    only approximately 1).
    """
    ref = np.asarray(logp_ref, dtype=np.float64)
    test = np.asarray(logq_test, dtype=np.float64)
    if ref.shape != test.shape:
        raise ValueError(f"row block shape mismatch: ref {ref.shape} vs test {test.shape}")
    p_ref = np.exp(ref)
    # exp(LOGPROB_CLAMP) is exactly 0.0 in fp64, so clamped entries contribute nothing and the
    # difference (which can be 0 - 0) is never multiplied by a non-zero weight: no NaN, no inf.
    kl = np.einsum("ij,ij->i", p_ref, ref - test)
    ref_argmax = ref.argmax(axis=1)
    test_argmax = test.argmax(axis=1)
    top1 = ref_argmax == test_argmax
    k = min(5, test.shape[1])
    top5_idx = np.argpartition(test, -k, axis=1)[:, -k:]
    top5 = (top5_idx == ref_argmax[:, None]).any(axis=1)
    rows = np.arange(ref.shape[0])
    nll_ref = -ref[rows, next_ids]
    nll_test = -test[rows, next_ids]
    return {
        "kl": kl,
        "top1_agree": top1,
        "top5_contain": top5,
        "nll_ref": nll_ref,
        "nll_test": nll_test,
        "ref_argmax": ref_argmax,
        "test_argmax": test_argmax,
        "p_ref_sum": p_ref.sum(axis=1),
    }


# --------------------------------------------------------------------------------------------
# Segment / corpus driver
# --------------------------------------------------------------------------------------------


def read_meta(directory: Path, name: str) -> dict:
    path = directory / f"{name}.meta.json"
    if not path.exists():
        raise FileNotFoundError(f"missing sidecar {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def open_rows(directory: Path, name: str, rows: int, vocab: int) -> np.memmap:
    path = directory / f"{name}.logprobs.f16"
    if not path.exists():
        raise FileNotFoundError(f"missing log-prob file {path}")
    expected = rows * vocab * 2
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path} is {actual} bytes, expected {expected} "
                         f"({rows} rows x {vocab} vocab x 2 bytes fp16)")
    return np.memmap(path, dtype="<f2", mode="r", shape=(rows, vocab))


def compare_segment(name: str, token_ids: list[int], ref_dir: Path, test_dir: Path,
                    row_chunk: int, strict: bool) -> dict:
    total_len = len(token_ids)
    rows = total_len - 1
    ref_meta = read_meta(ref_dir, name)
    test_meta = read_meta(test_dir, name)

    problems = []
    sha = token_ids_sha256(token_ids)
    for side, meta in (("ref", ref_meta), ("test", test_meta)):
        if int(meta.get("T", -1)) != total_len:
            problems.append(f"{side}.T={meta.get('T')} but the tokens file has {total_len}")
        if int(meta.get("rows", -1)) != rows:
            problems.append(f"{side}.rows={meta.get('rows')} but T-1={rows}")
        if meta.get("dtype") != "float16":
            problems.append(f"{side}.dtype={meta.get('dtype')!r}, expected 'float16'")
        if meta.get("sha256_of_token_ids_json") not in (None, sha):
            problems.append(f"{side}.sha256_of_token_ids_json={meta.get('sha256_of_token_ids_json')} "
                            f"but the tokens file hashes to {sha}")
    vocab = int(ref_meta["V"])
    if int(test_meta["V"]) != vocab:
        problems.append(f"vocab mismatch: ref V={vocab}, test V={test_meta['V']}")
    if problems:
        msg = f"segment {name!r}: " + "; ".join(problems)
        if strict:
            raise SystemExit("[kl_report] " + msg)
        print(f"[kl_report] WARNING {msg}")

    ref = open_rows(ref_dir, name, rows, vocab)
    test = open_rows(test_dir, name, rows, vocab)
    next_ids = np.asarray(token_ids[1:], dtype=np.int64)

    kl = np.empty(rows, dtype=np.float64)
    nll_ref = np.empty(rows, dtype=np.float64)
    nll_test = np.empty(rows, dtype=np.float64)
    top1 = np.empty(rows, dtype=bool)
    top5 = np.empty(rows, dtype=bool)
    p_sum_dev = 0.0
    for start in range(0, rows, row_chunk):
        stop = min(start + row_chunk, rows)
        st = chunk_stats(ref[start:stop], test[start:stop], next_ids[start:stop])
        kl[start:stop] = st["kl"]
        nll_ref[start:stop] = st["nll_ref"]
        nll_test[start:stop] = st["nll_test"]
        top1[start:stop] = st["top1_agree"]
        top5[start:stop] = st["top5_contain"]
        p_sum_dev = max(p_sum_dev, float(np.abs(st["p_ref_sum"] - 1.0).max()))

    worst = int(kl.argmax())
    return {
        "name": name,
        "T": total_len,
        "V": vocab,
        "rows": rows,
        "sha256_of_token_ids_json": sha,
        "mean_kl": float(kl.mean()),
        "median_kl": float(np.median(kl)),
        "p99_kl": float(np.percentile(kl, 99)),
        "max_kl": float(kl[worst]),
        "max_kl_position": worst,
        "max_kl_context_token": int(token_ids[worst]),
        "max_kl_next_token": int(token_ids[worst + 1]),
        "top1_agreement_pct": 100.0 * float(top1.mean()),
        "top5_containment_pct": 100.0 * float(top5.mean()),
        "nll_ref": float(nll_ref.mean()),
        "nll_test": float(nll_test.mean()),
        "ppl_ref": float(np.exp(nll_ref.mean())),
        "ppl_test": float(np.exp(nll_test.mean())),
        "positions_kl_gt_1": int((kl > KL_ALARM).sum()),
        "max_abs_p_ref_sum_minus_1": p_sum_dev,
        "problems": problems,
        "_kl": kl,
        "_nll_ref": nll_ref,
        "_nll_test": nll_test,
        "_top1": top1,
        "_top5": top5,
    }


def overall(segs: list[dict]) -> dict:
    kl = np.concatenate([s["_kl"] for s in segs])
    nll_ref = np.concatenate([s["_nll_ref"] for s in segs])
    nll_test = np.concatenate([s["_nll_test"] for s in segs])
    top1 = np.concatenate([s["_top1"] for s in segs])
    top5 = np.concatenate([s["_top5"] for s in segs])
    worst = int(kl.argmax())
    # Map the flat index back to (segment, position within that segment).
    off = 0
    worst_seg, worst_pos = segs[0]["name"], 0
    for s in segs:
        if worst < off + s["rows"]:
            worst_seg, worst_pos = s["name"], worst - off
            break
        off += s["rows"]
    return {
        "name": "ALL",
        "rows": int(kl.size),
        "mean_kl": float(kl.mean()),
        "median_kl": float(np.median(kl)),
        "p99_kl": float(np.percentile(kl, 99)),
        "max_kl": float(kl[worst]),
        "max_kl_segment": worst_seg,
        "max_kl_position": worst_pos,
        "top1_agreement_pct": 100.0 * float(top1.mean()),
        "top5_containment_pct": 100.0 * float(top5.mean()),
        "nll_ref": float(nll_ref.mean()),
        "nll_test": float(nll_test.mean()),
        "ppl_ref": float(np.exp(nll_ref.mean())),
        "ppl_test": float(np.exp(nll_test.mean())),
        "positions_kl_gt_1": int((kl > KL_ALARM).sum()),
    }


def markdown(segs: list[dict], total: dict, ref_dir: Path, test_dir: Path) -> str:
    head = ("| segment | rows | mean KL | median KL | p99 KL | max KL | max @ | top-1 | top-5 | "
            "ppl ref | ppl test | KL>1 |")
    sep = "|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|"
    lines = [
        f"# KL(reference || r4dx) -- {len(segs)} segment(s)",
        "",
        f"- reference log-probs: `{ref_dir}`",
        f"- test log-probs:      `{test_dir}`",
        f"- KL is in **nats**, summed over the full {segs[0]['V']}-way vocabulary, computed in fp64 "
        f"from fp16 inputs.",
        f"- `KL>1` counts positions whose KL exceeds {KL_ALARM} nat.",
        "",
        head, sep,
    ]
    for s in segs:
        lines.append(
            f"| {s['name']} | {s['rows']} | {s['mean_kl']:.5f} | {s['median_kl']:.5f} | "
            f"{s['p99_kl']:.5f} | {s['max_kl']:.5f} | pos {s['max_kl_position']} "
            f"(tok {s['max_kl_next_token']}) | {s['top1_agreement_pct']:.2f}% | "
            f"{s['top5_containment_pct']:.2f}% | {s['ppl_ref']:.3f} | {s['ppl_test']:.3f} | "
            f"{s['positions_kl_gt_1']} |")
    lines.append(
        f"| **{total['name']}** | {total['rows']} | **{total['mean_kl']:.5f}** | "
        f"{total['median_kl']:.5f} | {total['p99_kl']:.5f} | {total['max_kl']:.5f} | "
        f"{total['max_kl_segment']} pos {total['max_kl_position']} | "
        f"**{total['top1_agreement_pct']:.2f}%** | {total['top5_containment_pct']:.2f}% | "
        f"{total['ppl_ref']:.3f} | {total['ppl_test']:.3f} | {total['positions_kl_gt_1']} |")
    lines += ["", "Per-segment fp16 round-trip diagnostic "
              "(`max |sum_v exp(logp_ref[v]) - 1|`, should be ~1e-3, not ~1):"]
    for s in segs:
        lines.append(f"- `{s['name']}`: {s['max_abs_p_ref_sum_minus_1']:.3e}")
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------------------------


def self_test() -> int:
    """Two checks with an analytically known answer.

    1. `chunk_stats` directly, in fp64, on a 4-way distribution pair whose KL is exactly
       `0.25 * ln 2`:
           p = [1/2, 1/4, 1/8, 1/8], q = [1/4, 1/4, 1/4, 1/4]
           KL(p||q) = 1/2*ln2 + 1/4*ln1 + 1/8*ln(1/2) + 1/8*ln(1/2) = (1/4) ln 2
    2. the whole file pipeline, by writing that same pair out as real `.logprobs.f16` +
       `.meta.json` + `tokens.json` files in a temp directory and running `compare_segment` over
       them. This one only has to hold to fp16 precision, which is where the ~1e-4 tolerance below
       comes from (fp16 has ~3 decimal digits; a log-prob near -2 rounds to ~1e-3 absolute, and the
       KL is a p-weighted difference of two such roundings).
    """
    ok = True
    p = np.array([0.5, 0.25, 0.125, 0.125])
    q = np.array([0.25, 0.25, 0.25, 0.25])
    analytic = 0.25 * np.log(2.0)

    logp = np.log(p)[None, :].repeat(3, axis=0)
    logq = np.log(q)[None, :].repeat(3, axis=0)
    next_ids = np.array([0, 1, 2])
    st = chunk_stats(logp, logq, next_ids)
    err = float(np.abs(st["kl"] - analytic).max())
    print(f"[self-test] 1. chunk_stats fp64 KL     = {st['kl'][0]:.12f} "
          f"(analytic {analytic:.12f}, max err {err:.3e})")
    ok &= err < 1e-12
    # p's argmax is 0; q is uniform so its argmax is 0 too (first maximum wins), and 0 is trivially
    # inside a 4-element "top 5". Check the NLL wiring instead, which is unambiguous.
    nll_err = float(np.abs(st["nll_ref"] - (-np.log(p[next_ids]))).max())
    print(f"[self-test]    NLL_ref wiring          max err {nll_err:.3e}")
    ok &= nll_err < 1e-12

    # A case where the two argmaxes genuinely differ, to exercise top1/top5.
    p2 = np.array([0.1, 0.1, 0.7, 0.1])
    q2 = np.array([0.7, 0.1, 0.1, 0.1])
    st2 = chunk_stats(np.log(p2)[None, :], np.log(q2)[None, :], np.array([2]))
    analytic2 = float((p2 * (np.log(p2) - np.log(q2))).sum())
    print(f"[self-test]    asymmetric pair KL      = {st2['kl'][0]:.12f} (analytic {analytic2:.12f}), "
          f"top1_agree={bool(st2['top1_agree'][0])} (expect False), "
          f"top5_contain={bool(st2['top5_contain'][0])} (expect True, V=4 < 5)")
    ok &= abs(float(st2["kl"][0]) - analytic2) < 1e-12
    ok &= not bool(st2["top1_agree"][0])
    ok &= bool(st2["top5_contain"][0])

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ref_dir, test_dir = tmp / "ref", tmp / "test"
        ref_dir.mkdir(), test_dir.mkdir()
        rows, vocab = 8, 4
        token_ids = [0, 1, 2, 3, 0, 1, 2, 3, 0]  # T = rows + 1
        for directory, dist, source in ((ref_dir, p, "reference"), (test_dir, q, "r4dx")):
            arr = np.log(dist)[None, :].repeat(rows, axis=0)
            np.clip(arr, LOGPROB_CLAMP, None).astype("<f2").tofile(directory / "selftest.logprobs.f16")
            with open(directory / "selftest.meta.json", "w", encoding="utf-8") as f:
                json.dump({"T": len(token_ids), "V": vocab, "dtype": "float16", "rows": rows,
                           "source": source,
                           "sha256_of_token_ids_json": token_ids_sha256(token_ids)}, f)
        seg = compare_segment("selftest", token_ids, ref_dir, test_dir, row_chunk=3, strict=True)
        err2 = abs(seg["mean_kl"] - analytic)
        print(f"[self-test] 2. end-to-end file KL     = {seg['mean_kl']:.9f} "
              f"(analytic {analytic:.9f}, err {err2:.3e}, fp16 inputs)")
        ok &= err2 < 1e-4
        ok &= seg["rows"] == rows and seg["V"] == vocab and not seg["problems"]
        ppl_ref_analytic = float(np.exp(np.mean(-np.log(p[np.array(token_ids[1:])]))))
        # Perplexity is exp() of an fp16-rounded mean NLL, so it inherits fp16's *relative*
        # precision (~1e-3), not an absolute 1e-3 on a number of order 5.
        ppl_rel = abs(seg["ppl_ref"] - ppl_ref_analytic) / ppl_ref_analytic
        print(f"[self-test]    ppl_ref               = {seg['ppl_ref']:.9f} "
              f"(analytic {ppl_ref_analytic:.9f}, rel err {ppl_rel:.3e})")
        ok &= ppl_rel < 1e-3

        # A size/shape mismatch must be caught, not silently misread.
        (test_dir / "selftest.logprobs.f16").write_bytes(b"\x00" * 10)
        try:
            compare_segment("selftest", token_ids, ref_dir, test_dir, row_chunk=3, strict=True)
            print("[self-test] 3. truncated file NOT detected -- FAIL")
            ok = False
        except ValueError as exc:
            print(f"[self-test] 3. truncated file detected: {type(exc).__name__}")

    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


# --------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", type=Path, help="directory of the reference's .logprobs.f16/.meta.json")
    ap.add_argument("--test-dir", type=Path, help="directory of the engine's .logprobs.f16/.meta.json")
    ap.add_argument("--tokens", type=Path, help="the shared tokens.json both sides were run on")
    ap.add_argument("--out", type=Path, default=None, help="write the full report as JSON here")
    ap.add_argument("--segment", action="append", default=None,
                    help="only this segment (repeatable, or comma-separated)")
    ap.add_argument("--row-chunk", type=int, default=32, help="rows widened to fp64 at a time")
    ap.add_argument("--allow-mismatch", action="store_true",
                    help="warn instead of aborting when a sidecar disagrees with the tokens file")
    ap.add_argument("--self-test", action="store_true", help="run the analytic self-test and exit")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    missing = [n for n, v in (("--ref-dir", args.ref_dir), ("--test-dir", args.test_dir),
                              ("--tokens", args.tokens)) if v is None]
    if missing:
        raise SystemExit(f"[kl_report] missing required argument(s): {', '.join(missing)} "
                         "(or pass --self-test)")

    with open(args.tokens, "r", encoding="utf-8") as f:
        doc = json.load(f)
    wanted = None
    if args.segment:
        wanted = [s for arg in args.segment for s in arg.split(",") if s]
    segments = [s for s in doc["segments"] if wanted is None or s["name"] in wanted]
    if not segments:
        raise SystemExit(f"no segments selected (available: {[s['name'] for s in doc['segments']]})")

    results = []
    for seg in segments:
        results.append(compare_segment(seg["name"], seg["token_ids"], args.ref_dir, args.test_dir,
                                       args.row_chunk, strict=not args.allow_mismatch))
    total = overall(results)
    print(markdown(results, total, args.ref_dir, args.test_dir))

    if args.out:
        payload = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "ref_dir": str(args.ref_dir),
            "test_dir": str(args.test_dir),
            "tokens_file": str(args.tokens),
            "tokenizer": doc.get("tokenizer"),
            "kl_alarm_nats": KL_ALARM,
            "logprob_clamp": LOGPROB_CLAMP,
            "segments": [{k: v for k, v in s.items() if not k.startswith("_")} for s in results],
            "overall": total,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[kl_report] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
