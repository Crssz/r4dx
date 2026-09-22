"""tools/reference/kl_audit.py

The adversarial companion to `kl_report.py`. `kl_report.py` answers "how far apart are these two
log-prob dumps"; this answers "is that number measuring what it claims to measure, and where does
it come from". Run it on the same `--ref-dir`/`--test-dir`/`--tokens` triple.

Five checks, all numpy-only (no GPU, no torch, no checkpoint) -- the GPU-side half of the audit,
which validates the *reference* itself, is `tools/reference/reference_selfcheck.py`:

  1. IDENTITY      Both sidecars' `sha256_of_token_ids_json` recomputed from the tokens file, plus
                   `T`/`rows`/`V`. Proves both halves scored the same ids.
  2. ALIGNMENT     The deliberate off-by-one control. Pairs `ref[i]` against `test[i+1]` and
                   `ref[i+1]` against `test[i]` and reports top-1 agreement and mean KL for both.
                   A correctly aligned pair must beat both shifts by orders of magnitude; a dump
                   whose rows are off by one would score ~the shifted numbers instead. This is the
                   check that a "merely large" KL cannot hide from.
  3. CLAMP         How much reference probability mass actually sits on tokens the test side floored
                   at `LOGPROB_CLAMP` (and vice versa), and how many entries per row were clamped at
                   all. Turns the fp16-clamp *argument* in docs/validation.md into a measurement.
                   Also reports `sum_v exp(logp_ref[v])`, the fp16 round-trip diagnostic.
  4. ENTROPY       Mean KL and top-1 agreement binned by the REFERENCE's own per-position entropy,
                   per segment. KL grows with how flat the reference distribution is, so a segment
                   with more high-entropy positions has a higher mean KL for free. The
                   entropy-matched mean KL (each segment reweighted onto the first segment's entropy
                   histogram) separates that composition effect from a genuinely worse segment.
  5. VOCAB         The KL sum split by vocabulary region at `--vocab-split`, with the reference
                   probability mass in each region and the mass-weighted mean `|logp_ref - logp_test|`.
                   Says whether the divergence comes from the part of the vocabulary a segment
                   actually uses, and whether one region of the lm_head is reconstructed worse.

Usage:

    <venv>\\Scripts\\python.exe tools\\reference\\kl_audit.py `
        --ref-dir tools\\reference\\kl_out\\ref --test-dir tools\\reference\\kl_out\\w4a16 `
        --tokens tools\\reference\\kl_corpus\\tokens.json

    <venv>\\Scripts\\python.exe tools\\reference\\kl_audit.py --self-test
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

#: Must match `full_logits_golden.LOGPROB_CLAMP`, `kl_report.LOGPROB_CLAMP` and the C++ writer.
LOGPROB_CLAMP = -1e4

#: Default reference-entropy bin edges, in nats. The first bin is where the model is essentially
#: certain (p_max > ~0.9) and the last is where it is spread over hundreds of tokens.
ENTROPY_EDGES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 1e9)


def token_ids_sha256(token_ids) -> str:
    payload = json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def open_rows(directory: Path, name: str, rows: int, vocab: int) -> np.memmap:
    path = directory / f"{name}.logprobs.f16"
    expected = rows * vocab * 2
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(f"{path} is {actual} bytes, expected {expected} "
                         f"({rows} rows x {vocab} vocab x 2 bytes fp16)")
    return np.memmap(path, dtype="<f2", mode="r", shape=(rows, vocab))


# --------------------------------------------------------------------------------------------
# One segment, one pass over the two files
# --------------------------------------------------------------------------------------------


def audit_segment(name: str, token_ids: list[int], ref_dir: Path, test_dir: Path,
                  row_chunk: int, vocab_split: int) -> dict:
    """Everything checks 1-5 need for one segment, in a single chunked walk of both memmaps."""
    total_len = len(token_ids)
    rows = total_len - 1
    sha = token_ids_sha256(token_ids)
    metas = {}
    for side, directory in (("ref", ref_dir), ("test", test_dir)):
        with open(directory / f"{name}.meta.json", "r", encoding="utf-8") as f:
            metas[side] = json.load(f)
    vocab = int(metas["ref"]["V"])
    identity = {
        "sha256_of_token_ids_json": sha,
        "ref_sha_matches": metas["ref"].get("sha256_of_token_ids_json") == sha,
        "test_sha_matches": metas["test"].get("sha256_of_token_ids_json") == sha,
        "ref_T": int(metas["ref"]["T"]), "test_T": int(metas["test"]["T"]), "tokens_T": total_len,
        "ref_rows": int(metas["ref"]["rows"]), "test_rows": int(metas["test"]["rows"]),
        "ref_V": vocab, "test_V": int(metas["test"]["V"]),
        "ref_source": metas["ref"].get("source"), "test_source": metas["test"].get("source"),
        "first5": [int(t) for t in token_ids[:5]], "last5": [int(t) for t in token_ids[-5:]],
    }

    ref = open_rows(ref_dir, name, rows, vocab)
    test = open_rows(test_dir, name, rows, vocab)
    nxt = np.asarray(token_ids[1:], dtype=np.int64)

    kl = np.empty(rows); entropy = np.empty(rows)
    nll_ref = np.empty(rows); nll_test = np.empty(rows)
    top1 = np.zeros(rows, dtype=bool)
    ref_argmax = np.empty(rows, dtype=np.int64); test_argmax = np.empty(rows, dtype=np.int64)
    p_sum = np.empty(rows)
    clamp_mass_test = np.zeros(rows); clamp_mass_ref = np.zeros(rows)
    clamped_test = np.zeros(rows, dtype=np.int64); clamped_ref = np.zeros(rows, dtype=np.int64)
    split = min(max(int(vocab_split), 0), vocab)
    reg = {"kl_lo": 0.0, "kl_hi": 0.0, "mass_lo": 0.0, "mass_hi": 0.0,
           "absd_lo": 0.0, "absd_hi": 0.0}
    # A clamped entry is one at (or numerically below) the floor; the +1.0 slack is wider than any
    # fp16 rounding of -1e4 and far narrower than the gap to any real log-probability.
    floor = LOGPROB_CLAMP + 1.0

    for start in range(0, rows, row_chunk):
        stop = min(start + row_chunk, rows)
        r = np.asarray(ref[start:stop], dtype=np.float64)
        t = np.asarray(test[start:stop], dtype=np.float64)
        p = np.exp(r)
        d = r - t
        kl[start:stop] = np.einsum("ij,ij->i", p, d)
        entropy[start:stop] = -np.einsum("ij,ij->i", p, r)
        ra = r.argmax(axis=1); ta = t.argmax(axis=1)
        ref_argmax[start:stop] = ra; test_argmax[start:stop] = ta
        top1[start:stop] = ra == ta
        idx = np.arange(stop - start)
        nll_ref[start:stop] = -r[idx, nxt[start:stop]]
        nll_test[start:stop] = -t[idx, nxt[start:stop]]
        p_sum[start:stop] = p.sum(axis=1)
        mt = t <= floor; mr = r <= floor
        clamp_mass_test[start:stop] = (p * mt).sum(axis=1)
        clamp_mass_ref[start:stop] = (p * mr).sum(axis=1)
        clamped_test[start:stop] = mt.sum(axis=1); clamped_ref[start:stop] = mr.sum(axis=1)
        reg["kl_lo"] += float((p[:, :split] * d[:, :split]).sum())
        reg["kl_hi"] += float((p[:, split:] * d[:, split:]).sum())
        reg["mass_lo"] += float(p[:, :split].sum()); reg["mass_hi"] += float(p[:, split:].sum())
        reg["absd_lo"] += float((p[:, :split] * np.abs(d[:, :split])).sum())
        reg["absd_hi"] += float((p[:, split:] * np.abs(d[:, split:])).sum())
        del r, t, p, d

    # Check 2: the off-by-one controls. The shifted KL needs a second pass (different row pairing).
    kl_plus = np.empty(rows - 1)
    for start in range(0, rows - 1, row_chunk):
        stop = min(start + row_chunk, rows - 1)
        r = np.asarray(ref[start:stop], dtype=np.float64)
        t = np.asarray(test[start + 1:stop + 1], dtype=np.float64)
        kl_plus[start:stop] = np.einsum("ij,ij->i", np.exp(r), r - t)
        del r, t

    def per_mass(num: float, mass: float) -> float:
        return num / mass if mass > 1e-12 else float("nan")

    return {
        "name": name, "rows": rows, "V": vocab, "identity": identity,
        "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)),
        "p99_kl": float(np.percentile(kl, 99)), "max_kl": float(kl.max()),
        "max_kl_position": int(kl.argmax()),
        "top1_agreement_pct": 100.0 * float(top1.mean()),
        "alignment": {
            "top1_aligned_pct": 100.0 * float(top1.mean()),
            "top1_ref_i_vs_test_i_plus_1_pct": 100.0 * float((ref_argmax[:-1] == test_argmax[1:]).mean()),
            "top1_ref_i_plus_1_vs_test_i_pct": 100.0 * float((ref_argmax[1:] == test_argmax[:-1]).mean()),
            "mean_kl_aligned": float(kl.mean()),
            "mean_kl_shifted_plus_1": float(kl_plus.mean()),
            "ref_predicts_actual_next_pct": 100.0 * float((ref_argmax == nxt).mean()),
            "test_predicts_actual_next_pct": 100.0 * float((test_argmax == nxt).mean()),
        },
        "clamp": {
            "max_ref_mass_on_test_clamped": float(clamp_mass_test.max()),
            "mean_ref_mass_on_test_clamped": float(clamp_mass_test.mean()),
            "max_ref_mass_on_ref_clamped": float(clamp_mass_ref.max()),
            "max_clamped_entries_per_row_test": int(clamped_test.max()),
            "max_clamped_entries_per_row_ref": int(clamped_ref.max()),
            "total_clamped_entries_test": int(clamped_test.sum()),
            "total_clamped_entries_ref": int(clamped_ref.sum()),
            "min_sum_p_ref": float(p_sum.min()), "max_sum_p_ref": float(p_sum.max()),
        },
        "vocab_region": {
            "split": split,
            "mass_lo": reg["mass_lo"] / rows, "mass_hi": reg["mass_hi"] / rows,
            "kl_lo": reg["kl_lo"] / rows, "kl_hi": reg["kl_hi"] / rows,
            "kl_per_unit_mass_lo": per_mass(reg["kl_lo"], reg["mass_lo"]),
            "kl_per_unit_mass_hi": per_mass(reg["kl_hi"], reg["mass_hi"]),
            "abs_dlogp_lo": per_mass(reg["absd_lo"], reg["mass_lo"]),
            "abs_dlogp_hi": per_mass(reg["absd_hi"], reg["mass_hi"]),
        },
        "ppl_ref": float(np.exp(nll_ref.mean())), "ppl_test": float(np.exp(nll_test.mean())),
        "mean_entropy": float(entropy.mean()), "median_entropy": float(np.median(entropy)),
        "_kl": kl, "_entropy": entropy, "_top1": top1,
    }


def entropy_table(segs: list[dict], edges=ENTROPY_EDGES) -> dict:
    """Check 4. Mean KL and top-1 per reference-entropy bin, plus the entropy-matched mean KL: each
    segment's per-bin KL reweighted onto the FIRST segment's bin histogram, which is what isolates
    "this segment is genuinely worse" from "this segment simply has flatter distributions"."""
    edges = list(edges)
    nb = len(edges) - 1
    per_seg = {}
    for s in segs:
        h, k, t = s["_entropy"], s["_kl"], s["_top1"]
        bins = []
        for i in range(nb):
            m = (h >= edges[i]) & (h < edges[i + 1])
            bins.append({"n": int(m.sum()),
                         "mean_kl": float(k[m].mean()) if m.any() else None,
                         "top1_pct": 100.0 * float(t[m].mean()) if m.any() else None})
        per_seg[s["name"]] = bins
    base = per_seg[segs[0]["name"]]
    base_n = np.array([b["n"] for b in base], dtype=np.float64)
    matched = {}
    for name, bins in per_seg.items():
        w = base_n.copy()
        vals = np.array([b["mean_kl"] if b["mean_kl"] is not None else np.nan for b in bins])
        w[np.isnan(vals)] = 0.0
        matched[name] = float(np.nansum(w * vals) / w.sum()) if w.sum() > 0 else float("nan")
    return {"edges": edges, "per_segment": per_seg, "matched_to": segs[0]["name"],
            "entropy_matched_mean_kl": matched}


# --------------------------------------------------------------------------------------------
# Printing
# --------------------------------------------------------------------------------------------


def report(segs: list[dict], ent: dict, ref_dir: Path, test_dir: Path) -> None:
    print(f"# kl_audit -- {len(segs)} segment(s)")
    print(f"- reference dumps: {ref_dir}")
    print(f"- test dumps:      {test_dir}\n")

    print("## 1. Identity -- did both halves score the same token ids?")
    ok = True
    for s in segs:
        i = s["identity"]
        good = (i["ref_sha_matches"] and i["test_sha_matches"] and i["ref_T"] == i["tokens_T"]
                and i["test_T"] == i["tokens_T"] and i["ref_V"] == i["test_V"])
        ok &= good
        print(f"  {s['name']:<16} sha {i['sha256_of_token_ids_json'][:16]}...  "
              f"ref_match={i['ref_sha_matches']} test_match={i['test_sha_matches']}  "
              f"T={i['tokens_T']} rows={s['rows']} V={i['ref_V']}  "
              f"src={i['ref_source']}/{i['test_source']}  {'OK' if good else 'MISMATCH'}")
        print(f"  {'':16} first5={i['first5']} last5={i['last5']}")

    print("\n## 2. Alignment -- the deliberate off-by-one control")
    print(f"  {'segment':<16}{'top1 aligned':>14}{'top1 ref[i]/test[i+1]':>24}"
          f"{'top1 ref[i+1]/test[i]':>24}{'meanKL aligned':>16}{'meanKL shift+1':>16}")
    for s in segs:
        a = s["alignment"]
        print(f"  {s['name']:<16}{a['top1_aligned_pct']:>13.2f}%{a['top1_ref_i_vs_test_i_plus_1_pct']:>23.2f}%"
              f"{a['top1_ref_i_plus_1_vs_test_i_pct']:>23.2f}%{a['mean_kl_aligned']:>16.5f}"
              f"{a['mean_kl_shifted_plus_1']:>16.5f}")
    for s in segs:
        a = s["alignment"]
        print(f"  {s['name']:<16} reference's own argmax hits the actual next token "
              f"{a['ref_predicts_actual_next_pct']:.2f}% of the time (test: "
              f"{a['test_predicts_actual_next_pct']:.2f}%)")

    print("\n## 3. fp16 clamp -- how much does the -1e4 floor actually touch?")
    for s in segs:
        c = s["clamp"]
        print(f"  {s['name']:<16} clamped entries: test {c['total_clamped_entries_test']} "
              f"(max {c['max_clamped_entries_per_row_test']}/row), ref {c['total_clamped_entries_ref']} "
              f"(max {c['max_clamped_entries_per_row_ref']}/row) of {s['rows']}x{s['V']}")
        print(f"  {'':16} ref probability mass on a clamped test entry: max "
              f"{c['max_ref_mass_on_test_clamped']:.3e}, mean {c['mean_ref_mass_on_test_clamped']:.3e}"
              f"   |sum_v p_ref - 1| in [{abs(c['min_sum_p_ref']-1):.2e}, {abs(c['max_sum_p_ref']-1):.2e}]")

    print("\n## 4. KL vs the reference's own entropy")
    names = [s["name"] for s in segs]
    print(f"  {'mean/median entropy':<22}" + "".join(f"{n[:18]:>20}" for n in names))
    print(f"  {'':<22}" + "".join(f"{s['mean_entropy']:>10.4f}/{s['median_entropy']:<9.4f}" for s in segs))
    print(f"\n  mean KL per reference-entropy bin (nats), n = rows in bin:")
    print(f"  {'bin':<16}" + "".join(f"{n[:18]:>22}" for n in names))
    e = ent["edges"]
    for i in range(len(e) - 1):
        hi = "inf" if e[i + 1] >= 1e9 else f"{e[i+1]:.2f}"
        cells = ""
        for n in names:
            b = ent["per_segment"][n][i]
            cell = "-" if b["mean_kl"] is None else "%.4f (n=%d)" % (b["mean_kl"], b["n"])
            cells += f"{cell:>22}"
        print(f"  [{e[i]:.2f},{hi})".ljust(18) + cells)
    print(f"\n  top-1 agreement per reference-entropy bin:")
    print(f"  {'bin':<16}" + "".join(f"{n[:18]:>22}" for n in names))
    for i in range(len(e) - 1):
        hi = "inf" if e[i + 1] >= 1e9 else f"{e[i+1]:.2f}"
        cells = ""
        for n in names:
            b = ent["per_segment"][n][i]
            cell = "-" if b["top1_pct"] is None else "%.1f%% (n=%d)" % (b["top1_pct"], b["n"])
            cells += f"{cell:>22}"
        print(f"  [{e[i]:.2f},{hi})".ljust(18) + cells)
    print(f"\n  entropy-matched mean KL (every segment reweighted onto {ent['matched_to']}'s "
          f"entropy histogram):")
    for n in names:
        raw = next(s for s in segs if s["name"] == n)["mean_kl"]
        print(f"    {n:<16} raw {raw:.5f}  ->  entropy-matched {ent['entropy_matched_mean_kl'][n]:.5f}")

    print("\n## 5. Where in the vocabulary does the divergence live?")
    sp = segs[0]["vocab_region"]["split"]
    print(f"  split at id {sp}: low = [0,{sp}), high = [{sp},V)")
    for s in segs:
        v = s["vocab_region"]
        print(f"  {s['name']:<16} mean KL {s['mean_kl']:.5f}")
        print(f"  {'':16}   low : ref mass {v['mass_lo']:.5f}  KL {v['kl_lo']:+.5f}  "
              f"KL/mass {v['kl_per_unit_mass_lo']:+.5f}  mass-weighted |dlogp| {v['abs_dlogp_lo']:.5f}")
        print(f"  {'':16}   high: ref mass {v['mass_hi']:.5f}  KL {v['kl_hi']:+.5f}  "
              f"KL/mass {v['kl_per_unit_mass_hi']:+.5f}  mass-weighted |dlogp| {v['abs_dlogp_hi']:.5f}")

    print("\n## Overall")
    kl = np.concatenate([s["_kl"] for s in segs])
    t1 = np.concatenate([s["_top1"] for s in segs])
    print(f"  rows {kl.size}  mean KL {kl.mean():.17g}  median {np.median(kl):.5f}  "
          f"p99 {np.percentile(kl, 99):.5f}  max {kl.max():.5f}  top-1 {100*t1.mean():.2f}%")
    print(f"  identity checks: {'ALL PASS' if ok else 'FAILED'}")


# --------------------------------------------------------------------------------------------


def self_test() -> int:
    """Build a tiny ref/test pair whose answers are known and check every axis reports them.

    The test dump is the reference's rows SHIFTED BY ONE, which is exactly the failure check 2
    exists to catch: the aligned pairing must then look bad and the `ref[i] vs test[i+1]` pairing
    must look perfect. A clamped column is planted in the test side so check 3 has something to
    count, and the vocab split is placed so that all the probability mass is on one side of it.
    """
    ok = True
    rng = np.random.default_rng(7)
    rows, vocab = 16, 64
    token_ids = [int(i % vocab) for i in range(rows + 1)]
    logits = rng.normal(0.0, 3.0, size=(rows + 1, vocab))
    # The last column is pushed far down so that clamping it on the test side (below) costs the KL
    # nothing -- check 3 must still COUNT the clamped entries, but the planted clamp must not be
    # what check 2's shifted-pairing assertion is reading.
    logits[:, -1] = -60.0
    lp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rd, td = tmp / "ref", tmp / "test"
        rd.mkdir(); td.mkdir()
        ref_rows = lp[:rows]
        # test[i+1] == ref[i]: the dump is shifted by exactly one row, the failure check 2 exists
        # to catch. test[0] is an unrelated row so the aligned pairing has nothing to latch onto.
        test_rows = np.vstack([lp[rows], lp[:rows - 1]]).copy()
        test_rows[:, -1] = LOGPROB_CLAMP      # a planted clamped column
        for directory, arr, src in ((rd, ref_rows, "reference"), (td, test_rows, "r4dx")):
            np.clip(arr, LOGPROB_CLAMP, None).astype("<f2").tofile(directory / "s.logprobs.f16")
            with open(directory / "s.meta.json", "w", encoding="utf-8") as f:
                json.dump({"T": rows + 1, "V": vocab, "dtype": "float16", "rows": rows,
                           "source": src,
                           "sha256_of_token_ids_json": token_ids_sha256(token_ids)}, f)
        seg = audit_segment("s", token_ids, rd, td, row_chunk=5, vocab_split=vocab - 1)

    a = seg["alignment"]
    print(f"[self-test] identity            ref={seg['identity']['ref_sha_matches']} "
          f"test={seg['identity']['test_sha_matches']} (expect True/True)")
    ok &= seg["identity"]["ref_sha_matches"] and seg["identity"]["test_sha_matches"]
    print(f"[self-test] aligned top-1       {a['top1_aligned_pct']:.2f}%  (expect ~0: the test dump "
          f"is shifted by one on purpose)")
    ok &= a["top1_aligned_pct"] < 25.0
    print(f"[self-test] ref[i]/test[i+1]    {a['top1_ref_i_vs_test_i_plus_1_pct']:.2f}%  (expect 100)")
    ok &= a["top1_ref_i_vs_test_i_plus_1_pct"] > 99.0
    print(f"[self-test] mean KL aligned     {a['mean_kl_aligned']:.4f}  vs shift+1 "
          f"{a['mean_kl_shifted_plus_1']:.4f}  (expect shift+1 much smaller)")
    ok &= a["mean_kl_shifted_plus_1"] < 0.01 * a["mean_kl_aligned"]
    c = seg["clamp"]
    print(f"[self-test] clamped entries     test {c['total_clamped_entries_test']} (expect {rows}), "
          f"ref {c['total_clamped_entries_ref']} (expect 0)")
    ok &= c["total_clamped_entries_test"] == rows and c["total_clamped_entries_ref"] == 0
    print(f"[self-test] mass on clamped     max {c['max_ref_mass_on_test_clamped']:.3e} (expect > 0 "
          f"but tiny: the planted column's reference mass is exp(-60)-ish by construction)")
    ok &= c["max_ref_mass_on_test_clamped"] > 0.0
    v = seg["vocab_region"]
    print(f"[self-test] vocab split         low mass {v['mass_lo']:.4f} + high mass {v['mass_hi']:.4f} "
          f"= {v['mass_lo'] + v['mass_hi']:.4f} (expect ~1)")
    ok &= abs(v["mass_lo"] + v["mass_hi"] - 1.0) < 1e-2
    ok &= abs(v["kl_lo"] + v["kl_hi"] - seg["mean_kl"]) < 1e-6 * max(1.0, abs(seg["mean_kl"]))
    print(f"[self-test] region KL splits    {v['kl_lo']:.5f} + {v['kl_hi']:.5f} = "
          f"{v['kl_lo'] + v['kl_hi']:.5f} vs mean KL {seg['mean_kl']:.5f}")
    ent = entropy_table([seg])
    total_binned = sum(b["n"] for b in ent["per_segment"]["s"])
    print(f"[self-test] entropy bins        {total_binned} rows binned of {rows}")
    ok &= total_binned == rows
    print(f"[self-test] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref-dir", type=Path)
    ap.add_argument("--test-dir", type=Path)
    ap.add_argument("--tokens", type=Path)
    ap.add_argument("--segment", action="append", default=None,
                    help="only this segment (repeatable, or comma-separated)")
    ap.add_argument("--row-chunk", type=int, default=32)
    ap.add_argument("--vocab-split", type=int, default=148000,
                    help="id where check 5 splits the vocabulary; 148000 is where this "
                         "checkpoint's extended/multilingual tail begins")
    ap.add_argument("--out", type=Path, default=None, help="write the audit as JSON here")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    missing = [n for n, v in (("--ref-dir", args.ref_dir), ("--test-dir", args.test_dir),
                              ("--tokens", args.tokens)) if v is None]
    if missing:
        raise SystemExit(f"[kl_audit] missing required argument(s): {', '.join(missing)} "
                         "(or pass --self-test)")

    with open(args.tokens, "r", encoding="utf-8") as f:
        doc = json.load(f)
    wanted = None
    if args.segment:
        wanted = [s for arg in args.segment for s in arg.split(",") if s]
    segments = [s for s in doc["segments"] if wanted is None or s["name"] in wanted]
    if not segments:
        raise SystemExit(f"no segments selected (available: {[s['name'] for s in doc['segments']]})")

    segs = [audit_segment(s["name"], s["token_ids"], args.ref_dir, args.test_dir,
                          args.row_chunk, args.vocab_split) for s in segments]
    ent = entropy_table(segs)
    report(segs, ent, args.ref_dir, args.test_dir)

    if args.out:
        payload = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "ref_dir": str(args.ref_dir), "test_dir": str(args.test_dir),
            "tokens_file": str(args.tokens), "logprob_clamp": LOGPROB_CLAMP,
            "vocab_split": args.vocab_split,
            "segments": [{k: v for k, v in s.items() if not k.startswith("_")} for s in segs],
            "entropy": ent,
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[kl_audit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
