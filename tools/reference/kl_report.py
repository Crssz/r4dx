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

Score-mask mode (`--use-score-mask`): segments of --tokens that carry a `score_mask` (chat_gemma.json) are
aggregated over the rows predicting model-turn tokens only.

Gemma M1 gate (docs/gemma4-plan.md 9.9), against the fp32 truth and relative to the noise of stock HF bf16 sdpa:

    <venv>\\Scripts\\python.exe tools\\reference\\kl_report.py --gate gemma-fp32 `
        --truth-dir D:\\models\\r4dx\\huihui-gemma\\kl\\fp32\\truth --noise-dir ...\\fp32\\bf16sdpa `
        --test-dir <r4dx dump dir> --tokens tools\\reference\\kl_corpus\\chat_gemma.json `
        --raw-tokens tools\\reference\\kl_corpus\\tokens_gemma.json --raw-max-tokens 512 --out gate.json
    (+ --base-dir <r4dx bf16-container dump> for a quantized container: judged by the KL increment <= 0.01)

Exit status 0 iff every primary (chat) group passes. Test: tools\\reference\\test_kl_report_gate.py.

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


def side_identity(directory: Path, meta: dict) -> dict:
    """What produced one side of a segment: the layer count (the sidecar's, else -- for dumps written
    before sidecars recorded it -- the directory's reference_run.json; None for the r4dx engine, which
    always runs the whole model), --debug-max-layers, and for a full_logits_golden.py --weights-gguf
    dump the GGUF's header sha256 and substituted tensor count."""
    run = {}
    run_path = directory / "reference_run.json"
    if "n_layers" not in meta and run_path.exists():
        try:
            with open(run_path, "r", encoding="utf-8") as f:
                run = json.load(f)
        except (OSError, ValueError):
            run = {}
    wg = meta.get("weights_gguf") or {}
    wo = meta.get("weights_override") or {}  # full_logits_golden.py --weights-override
    return {"source": meta.get("source"),
            "n_layers": meta.get("n_layers", run.get("n_layers")),
            "debug_max_layers": meta.get("debug_max_layers", run.get("debug_max_layers")),
            "gguf_header_sha256": wg.get("header_sha256"),
            "gguf_substituted_count": wg.get("substituted_count"),
            "gguf_path": wg.get("path"),
            "override_manifest_sha256": wo.get("manifest_sha256"),
            "override_path": wo.get("manifest")}


def check_sides(results: list[dict], strict: bool) -> list[str]:
    """Every segment of one side must come from the same producer: one layer count, one GGUF. A
    directory mixing two runs (another GGUF, a --debug-max-layers dump) would otherwise be scored
    segment by segment without complaint."""
    problems = []
    for side in ("ref", "test"):
        ids = {json.dumps({k: r[f"{side}_identity"].get(k) for k in (
                    "n_layers", "gguf_header_sha256", "gguf_substituted_count",
                    "override_manifest_sha256")}, sort_keys=True)
               for r in results}
        if len(ids) > 1:
            problems.append(f"the {side} directory mixes dumps of different runs across segments: "
                            + " vs ".join(sorted(ids)))
    if problems and strict:
        raise SystemExit("[kl_report] " + "; ".join(problems))
    for p in problems:
        print(f"[kl_report] WARNING {p}")
    return problems


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


def row_mask_from_score_mask(score_mask) -> np.ndarray:
    """Row i of a log-prob file predicts token i+1, so it is scored iff `score_mask[i+1]` is set
    (the chat corpus' score_mask marks the model-turn TOKENS). Returns a bool array of T-1 rows."""
    m = np.asarray(score_mask, dtype=bool)
    return m[1:]


def compare_segment(name: str, token_ids: list[int], ref_dir: Path, test_dir: Path,
                    row_chunk: int, strict: bool, row_mask: np.ndarray | None = None) -> dict:
    """`row_mask` (bool, T-1 rows; see row_mask_from_score_mask): when given only those rows are read and
    aggregated; "rows" is then the scored-row count, "rows_total" T-1, and `_pos` the original row index
    of every scored row."""
    total_len = len(token_ids)
    rows = total_len - 1
    if row_mask is not None:
        row_mask = np.asarray(row_mask, dtype=bool)
        if row_mask.shape != (rows,):
            raise ValueError(f"segment {name!r}: row_mask has shape {row_mask.shape}, expected ({rows},)")
        if not row_mask.any():
            raise ValueError(f"segment {name!r}: row_mask selects no rows")
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
    if ref_meta.get("weights_gguf"):  # full_logits_golden.py --weights-gguf: a test side, never a ref
        problems.append(f"ref side is a --weights-gguf dump (source {ref_meta.get('source')!r}), "
                        "not the bf16 reference")
    if ref_meta.get("weights_override"):  # --weights-override: likewise only ever a test side
        problems.append(f"ref side is a --weights-override dump (source {ref_meta.get('source')!r}), "
                        "not the bf16 reference")
    ref_id, test_id = side_identity(ref_dir, ref_meta), side_identity(test_dir, test_meta)
    if (ref_id["n_layers"] is not None and test_id["n_layers"] is not None
            and int(ref_id["n_layers"]) != int(test_id["n_layers"])):
        problems.append(f"layer count mismatch: ref ran {ref_id['n_layers']} layers, test "
                        f"{test_id['n_layers']} (a --debug-max-layers dump?)")
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

    kl = np.full(rows, np.nan, dtype=np.float64)
    nll_ref = np.full(rows, np.nan, dtype=np.float64)
    nll_test = np.full(rows, np.nan, dtype=np.float64)
    top1 = np.zeros(rows, dtype=bool)
    top5 = np.zeros(rows, dtype=bool)
    p_sum_dev = 0.0
    for start in range(0, rows, row_chunk):
        stop = min(start + row_chunk, rows)
        if row_mask is None:
            idx = np.arange(start, stop)
        else:
            idx = start + np.flatnonzero(row_mask[start:stop])
            if idx.size == 0:
                continue
        contiguous = idx.size == stop - start
        st = chunk_stats(ref[start:stop] if contiguous else ref[idx],
                         test[start:stop] if contiguous else test[idx], next_ids[idx])
        kl[idx] = st["kl"]
        nll_ref[idx] = st["nll_ref"]
        nll_test[idx] = st["nll_test"]
        top1[idx] = st["top1_agree"]
        top5[idx] = st["top5_contain"]
        p_sum_dev = max(p_sum_dev, float(np.abs(st["p_ref_sum"] - 1.0).max()))

    pos = np.arange(rows) if row_mask is None else np.flatnonzero(row_mask)
    kl, nll_ref, nll_test, top1, top5 = kl[pos], nll_ref[pos], nll_test[pos], top1[pos], top5[pos]
    rows_total = rows
    rows = int(pos.size)
    worst = int(kl.argmax())
    worst_row = int(pos[worst])
    return {
        "name": name,
        "T": total_len,
        "V": vocab,
        "rows": rows,
        "rows_total": rows_total,
        "scored_rows_only": row_mask is not None,
        "sha256_of_token_ids_json": sha,
        "mean_kl": float(kl.mean()),
        "median_kl": float(np.median(kl)),
        "p99_kl": float(np.percentile(kl, 99)),
        "max_kl": float(kl[worst]),
        "max_kl_position": worst_row,
        "max_kl_context_token": int(token_ids[worst_row]),
        "max_kl_next_token": int(token_ids[worst_row + 1]),
        "top1_agreement_pct": 100.0 * float(top1.mean()),
        "top5_containment_pct": 100.0 * float(top5.mean()),
        "nll_ref": float(nll_ref.mean()),
        "nll_test": float(nll_test.mean()),
        "ppl_ref": float(np.exp(nll_ref.mean())),
        "ppl_test": float(np.exp(nll_test.mean())),
        "positions_kl_gt_1": int((kl > KL_ALARM).sum()),
        "max_abs_p_ref_sum_minus_1": p_sum_dev,
        "problems": problems,
        "ref_identity": ref_id,
        "test_identity": test_id,
        "_pos": pos,
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
            worst_seg, worst_pos = s["name"], int(s["_pos"][worst - off])
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
    def describe(ident: dict) -> str:
        s = f"source {ident['source']!r}"
        if ident["n_layers"] is not None:
            s += f", {ident['n_layers']} layers"
        if ident["gguf_header_sha256"]:
            s += (f", GGUF weights `{Path(ident['gguf_path'] or '?').name}` (header sha256 "
                  f"{ident['gguf_header_sha256'][:16]}..., {ident['gguf_substituted_count']} tensors substituted)")
        return s

    lines = [
        f"# KL(reference || test) -- {len(segs)} segment(s)",
        "",
        f"- reference log-probs: `{ref_dir}` ({describe(segs[0]['ref_identity'])})",
        f"- test log-probs:      `{test_dir}` ({describe(segs[0]['test_identity'])})",
        f"- KL is in **nats**, summed over the full {segs[0]['V']}-way vocabulary, computed in fp64 "
        f"from fp16 inputs.",
        f"- `KL>1` counts positions whose KL exceeds {KL_ALARM} nat.",
    ]
    for side in ("ref", "test"):
        dbg = segs[0][f"{side}_identity"]["debug_max_layers"]
        if dbg:
            lines.append(f"- **WARNING: the {side} side is a --debug-max-layers {dbg} dump -- NOT a valid golden.**")
    lines += ["", head, sep]
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
# Gemma M1 gate against the fp32 truth (docs/gemma4-plan.md 9.9)
# --------------------------------------------------------------------------------------------
#
# The bf16 HF stack is not a stable yardstick for Gemma 4 (residual stream 100-300, bf16 rounding
# amplified), so the reference is an fp32-truth dump and the allowance is the NOISE of stock HF bf16 sdpa
# against that truth, measured on the same rows:
#
#     per group:  KL(truth||r4dx)  <= 1.5 * KL(truth||HF-bf16-sdpa) + 0.005          (mean over scored rows)
#                 top-1(truth,r4dx) >= top-1(truth,HF-bf16-sdpa) - 1 percentage point
#
# Quantized (trellis) containers are judged by the INCREMENT over the r4dx bf16 container instead
# (--base-dir): KL(truth||quant) - KL(truth||r4dx-bf16) <= 0.01 per group.
# Primary corpus: chat-templated sequences scored on model-turn tokens only (score_mask); groups are
# english / thai / code (python + cpp) and the pooled chat-ALL. Secondary: raw segments, REPORTED ONLY.

GATE_KL_FACTOR = 1.5
GATE_KL_SLACK = 0.005
GATE_TOP1_SLACK_PTS = 1.0
GATE_QUANT_INCREMENT = 0.01
CHAT_GATE_GROUPS = {"english": "english", "thai": "thai", "python": "code", "cpp": "code"}


def pool_stats(segs: list[dict]) -> dict:
    """Row-weighted statistics over the scored rows of several compare_segment() results."""
    kl = np.concatenate([s["_kl"] for s in segs])
    top1 = np.concatenate([s["_top1"] for s in segs])
    top5 = np.concatenate([s["_top5"] for s in segs])
    nll_ref = np.concatenate([s["_nll_ref"] for s in segs])
    nll_test = np.concatenate([s["_nll_test"] for s in segs])
    return {"rows": int(kl.size), "mean_kl": float(kl.mean()), "median_kl": float(np.median(kl)),
            "p99_kl": float(np.percentile(kl, 99)), "max_kl": float(kl.max()),
            "top1_pct": 100.0 * float(top1.mean()), "top5_pct": 100.0 * float(top5.mean()),
            "nll_ref": float(nll_ref.mean()), "nll_test": float(nll_test.mean())}


def gate_thresholds(noise: dict) -> dict:
    """The thresholds r4dx must meet, from the noise term (pool_stats of KL(truth||HF-bf16-sdpa))."""
    return {"kl_max": GATE_KL_FACTOR * noise["mean_kl"] + GATE_KL_SLACK,
            "top1_min_pct": noise["top1_pct"] - GATE_TOP1_SLACK_PTS}


def worst_rows(segs: list[dict], n: int = 5) -> list[dict]:
    rows = []
    for s in segs:
        k = np.argsort(s["_kl"])[::-1][:n]
        for j in k:
            p = int(s["_pos"][j])
            rows.append({"segment": s["name"], "row": p, "kl": float(s["_kl"][j]),
                         "context_token": int(s["_ids"][p]), "next_token": int(s["_ids"][p + 1])})
    rows.sort(key=lambda r: -r["kl"])
    return rows[:n]


def evaluate_group(test: list[dict], noise: list[dict], base: list[dict] | None, gated: bool) -> dict:
    t, nz = pool_stats(test), pool_stats(noise)
    thr = gate_thresholds(nz)
    out = {"rows": t["rows"], "test": t, "noise": nz, "thresholds": thr, "worst_rows": worst_rows(test),
           "kl_ok": t["mean_kl"] <= thr["kl_max"], "top1_ok": t["top1_pct"] >= thr["top1_min_pct"]}
    if base is not None:
        b = pool_stats(base)
        out["base"] = b
        out["increment_kl"] = t["mean_kl"] - b["mean_kl"]
        out["increment_top1_pts"] = t["top1_pct"] - b["top1_pct"]
        out["increment_ok"] = out["increment_kl"] <= GATE_QUANT_INCREMENT
        out["pass"] = out["increment_ok"] if gated else None
    else:
        out["pass"] = (out["kl_ok"] and out["top1_ok"]) if gated else None
    return out


def build_gate_groups(chat_items: list[dict], raw_items: list[dict]) -> list[tuple[str, bool, list[dict]]]:
    """[(group name, gated, items)]; an item is {"name","group","test","noise","base"}."""
    groups: dict[str, list[dict]] = {}
    for it in chat_items:
        groups.setdefault("chat-" + CHAT_GATE_GROUPS.get(it["group"], it["group"]), []).append(it)
    out = [(g, True, v) for g, v in sorted(groups.items())]
    if chat_items:
        out.append(("chat-ALL", True, list(chat_items)))
    for it in raw_items:
        out.append(("raw:" + it["name"], False, [it]))
    if raw_items:
        out.append(("raw-ALL", False, list(raw_items)))
    return out


def gate_gemma_fp32_markdown(res: dict) -> str:
    quant = res["mode"] == "quantized-increment"
    L = [f"# Gemma fp32-truth gate -- test `{res['test_dir']}`",
         f"- truth `{res['truth_dir']}`; noise (HF bf16 sdpa) `{res['noise_dir']}`"
         + (f"; base (r4dx bf16) `{res['base_dir']}`" if quant else ""),
         "- rule: " + ("KL(truth||quant) - KL(truth||r4dx-bf16) <= 0.01 per group (quantized: increment)" if quant else
                      f"mean KL <= {GATE_KL_FACTOR} x noise KL + {GATE_KL_SLACK}; top-1 >= noise top-1 - "
                      f"{GATE_TOP1_SLACK_PTS:g} pt"),
         "- chat groups are scored on model-turn tokens only; raw groups are REPORTED, no pass/fail", ""]
    head = ("| group | rows | test mean KL | median | p99 | noise mean KL | noise median | noise p99 | KL max (gate) | "
            "test top-1 | noise top-1 | top-1 min (gate) |" + (" incr KL | " if quant else " ") + "verdict |")
    L += [head, "|" + "---|" * (head.count("|") - 1)]
    for g in res["groups"]:
        t, n, th = g["test"], g["noise"], g["thresholds"]
        verdict = "info" if g["pass"] is None else ("PASS" if g["pass"] else "FAIL")
        if g["pass"] is not None and not quant:
            verdict += f" (KL {'ok' if g['kl_ok'] else 'X'}, top-1 {'ok' if g['top1_ok'] else 'X'})"
        L.append(f"| {g['name']} | {g['rows']} | {t['mean_kl']:.5f} | {t['median_kl']:.5f} | {t['p99_kl']:.5f} | "
                 f"{n['mean_kl']:.5f} | {n['median_kl']:.5f} | {n['p99_kl']:.5f} | {th['kl_max']:.5f} | "
                 f"{t['top1_pct']:.2f}% | {n['top1_pct']:.2f}% | {th['top1_min_pct']:.2f}% | "
                 + (f"{g['increment_kl']:+.5f} | " if quant else "") + f"{verdict} |")
    L += ["", "Worst rows per group (KL(truth||test), nats):"]
    for g in res["groups"]:
        w = ", ".join(f"{r['segment']}@{r['row']}={r['kl']:.3f}" for r in g["worst_rows"][:3])
        L.append(f"- {g['name']}: {w}")
    L += ["", f"**Overall (primary groups): {'PASS' if res['pass'] else 'FAIL'}**"]
    return "\n".join(L)


def gate_gemma_fp32(truth_dir: Path, noise_dir: Path, test_dir: Path, chat_tokens: Path,
                    raw_tokens: Path | None = None, raw_max_tokens: int = 512, raw_names: list[str] | None = None,
                    base_dir: Path | None = None, row_chunk: int = 32, strict: bool = True) -> dict:
    chat_doc = json.loads(Path(chat_tokens).read_text(encoding="utf-8"))
    chat_items, raw_items = [], []

    def run(name, ids, mask):
        r = {}
        for key, d in (("test", test_dir), ("noise", noise_dir), ("base", base_dir)):
            if d is None:
                r[key] = None
                continue
            r[key] = compare_segment(name, ids, truth_dir, d, row_chunk, strict, row_mask=mask)
            r[key]["_ids"] = ids
        return r

    for s in chat_doc["segments"]:
        if "score_mask" not in s:
            raise SystemExit(f"{chat_tokens}: segment {s['name']} has no score_mask (not a chat corpus)")
        assert len(s["score_mask"]) == len(s["token_ids"]), "score_mask length != token_ids length"
        r = run(s["name"], s["token_ids"], row_mask_from_score_mask(s["score_mask"]))
        chat_items.append({"name": s["name"], "group": s.get("group", "chat"), **r})
    if raw_tokens is not None:
        raw_doc = json.loads(Path(raw_tokens).read_text(encoding="utf-8"))
        for s in raw_doc["segments"]:
            if raw_names is not None and s["name"] not in raw_names:
                continue
            if not (Path(truth_dir) / f"{s['name']}.meta.json").exists():
                print(f"[kl_report] raw segment {s['name']}: no truth dump, skipped")
                continue
            ids = s["token_ids"][:raw_max_tokens] if raw_max_tokens else s["token_ids"]
            r = run(s["name"], ids, None)
            raw_items.append({"name": s["name"], "group": "raw", **r})
    groups = []
    for gname, gated, items in build_gate_groups(chat_items, raw_items):
        g = evaluate_group([i["test"] for i in items], [i["noise"] for i in items],
                           [i["base"] for i in items] if base_dir is not None else None, gated)
        g["name"], g["gated"], g["segments"] = gname, gated, [i["name"] for i in items]
        groups.append(g)
    primary = [g["pass"] for g in groups if g["gated"]]
    return {"gate": "gemma-fp32", "mode": "quantized-increment" if base_dir is not None else "bf16",
            "truth_dir": str(truth_dir), "noise_dir": str(noise_dir), "test_dir": str(test_dir),
            "base_dir": None if base_dir is None else str(base_dir), "chat_tokens": str(chat_tokens),
            "raw_tokens": None if raw_tokens is None else str(raw_tokens), "raw_max_tokens": raw_max_tokens,
            "rule": {"kl_factor": GATE_KL_FACTOR, "kl_slack": GATE_KL_SLACK, "top1_slack_pts": GATE_TOP1_SLACK_PTS,
                     "quant_increment": GATE_QUANT_INCREMENT},
            "groups": groups, "pass": bool(primary) and all(primary),
            "per_segment": [{"name": i["name"], "group": i["group"],
                             "test_mean_kl": float(i["test"]["mean_kl"]), "noise_mean_kl": float(i["noise"]["mean_kl"]),
                             "test_top1_pct": i["test"]["top1_agreement_pct"],
                             "noise_top1_pct": i["noise"]["top1_agreement_pct"], "rows": i["test"]["rows"]}
                            for i in chat_items + raw_items]}


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

        # A truncated (--debug-max-layers) dump must not be scored against a full reference: the test
        # sidecar says 16 layers, the reference's layer count comes from its reference_run.json.
        with open(ref_dir / "reference_run.json", "w", encoding="utf-8") as f:
            json.dump({"n_layers": 64}, f)
        meta_path = test_dir / "selftest.meta.json"
        good_meta = meta_path.read_text(encoding="utf-8")
        meta_path.write_text(json.dumps({**json.loads(good_meta), "n_layers": 16, "debug_max_layers": 16}),
                             encoding="utf-8")
        try:
            compare_segment("selftest", token_ids, ref_dir, test_dir, row_chunk=3, strict=True)
            print("[self-test] 3. layer-count mismatch NOT detected -- FAIL")
            ok = False
        except SystemExit as exc:
            print(f"[self-test] 3. layer-count mismatch detected: {str(exc)[:90]}")
        meta_path.write_text(good_meta, encoding="utf-8")
        # Segments of one side from two different runs (two GGUFs) must be refused too.
        seg_b = dict(seg, test_identity={**seg["test_identity"], "gguf_header_sha256": "ab" * 32})
        try:
            check_sides([seg, seg_b], strict=True)
            print("[self-test] 4. mixed test directory NOT detected -- FAIL")
            ok = False
        except SystemExit:
            print("[self-test] 4. mixed test directory detected")
        ok &= check_sides([seg, seg], strict=True) == []

        # A size/shape mismatch must be caught, not silently misread.
        (test_dir / "selftest.logprobs.f16").write_bytes(b"\x00" * 10)
        try:
            compare_segment("selftest", token_ids, ref_dir, test_dir, row_chunk=3, strict=True)
            print("[self-test] 5. truncated file NOT detected -- FAIL")
            ok = False
        except ValueError as exc:
            print(f"[self-test] 5. truncated file detected: {type(exc).__name__}")

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
    ap.add_argument("--use-score-mask", action="store_true",
                    help="score-mask aware mode for the plain report: segments of --tokens that carry a "
                         "`score_mask` are aggregated over the model-turn rows only")
    ap.add_argument("--gate", choices=["gemma-fp32"], default=None,
                    help="gemma-fp32: the M1 gate against the fp32 truth (see the section comment above "
                         "gate_gemma_fp32); needs --truth-dir --noise-dir --test-dir and --tokens=chat_gemma.json")
    ap.add_argument("--truth-dir", type=Path, help="gate: fp32-truth dump dir")
    ap.add_argument("--noise-dir", type=Path, help="gate: HF bf16 sdpa dump dir (the noise term)")
    ap.add_argument("--base-dir", type=Path, default=None,
                    help="gate: r4dx bf16-container dump; with it --test-dir is a quantized container judged by "
                         "the KL increment over this base (<= 0.01) instead of the noise-relative rule")
    ap.add_argument("--raw-tokens", type=Path, default=None, help="gate: raw segments (secondary, report only)")
    ap.add_argument("--raw-max-tokens", type=int, default=512)
    ap.add_argument("--raw-segment", default=None, help="gate: comma list of raw segment names")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if args.gate == "gemma-fp32":
        missing = [n for n, v in (("--truth-dir", args.truth_dir), ("--noise-dir", args.noise_dir),
                                  ("--test-dir", args.test_dir), ("--tokens", args.tokens)) if v is None]
        if missing:
            raise SystemExit(f"[kl_report] --gate gemma-fp32 needs {', '.join(missing)}")
        res = gate_gemma_fp32(args.truth_dir, args.noise_dir, args.test_dir, args.tokens, args.raw_tokens,
                              args.raw_max_tokens, None if args.raw_segment is None else args.raw_segment.split(","),
                              args.base_dir, args.row_chunk, strict=not args.allow_mismatch)
        print(gate_gemma_fp32_markdown(res))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps({"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), **res},
                                           indent=2), encoding="utf-8")
            print(f"\n[kl_report] wrote {args.out}")
        return 0 if res["pass"] else 1
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
        rmask = (row_mask_from_score_mask(seg["score_mask"])
                 if args.use_score_mask and "score_mask" in seg else None)
        results.append(compare_segment(seg["name"], seg["token_ids"], args.ref_dir, args.test_dir,
                                       args.row_chunk, strict=not args.allow_mismatch, row_mask=rmask))
    side_problems = check_sides(results, strict=not args.allow_mismatch)
    total = overall(results)
    print(markdown(results, total, args.ref_dir, args.test_dir))

    if args.out:
        sys.path.insert(0, str(Path(__file__).parent))
        from common import tokens_file_tokenizer_mode  # torch-free

        payload = {
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "ref_dir": str(args.ref_dir),
            "test_dir": str(args.test_dir),
            "tokens_file": str(args.tokens),
            "tokenizer": doc.get("tokenizer"),
            # docs/quant2.md 3.4: Thai KL is comparable only between runs of the same mode.
            "tokenizer_mode": tokens_file_tokenizer_mode(doc),
            "tokenizer_provenance": doc.get("tokenizer_provenance"),
            "kl_alarm_nats": KL_ALARM,
            "logprob_clamp": LOGPROB_CLAMP,
            "ref_identity": results[0]["ref_identity"],
            "test_identity": results[0]["test_identity"],
            "side_problems": side_problems,
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
