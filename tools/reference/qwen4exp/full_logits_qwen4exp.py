"""tools/reference/qwen4exp/full_logits_qwen4exp.py  (plan M0-5 / 4.10, the KL reference of the qwen4_exp gate)

Teacher-forced log-probabilities of Qwen3.8-Flash-Next from the layer-streaming HF reference (`ref.py`): the AWQ
int4 weights dequantized to bf16 one layer at a time, n-gram rows from the sidecar. For every segment of a tokens
file it runs the whole stack from position 0 (no cache) and writes

    <out-dir>/<segment>.logprobs.f16   rows 0..T-2 of log_softmax(logits), [T-1, 248320] float16
    <out-dir>/<segment>.meta.json      kl_report.py's shared format (T, V, rows, sha256_of_token_ids_json, n_layers, ...)
    <out-dir>/reference_run.json       the run record
    <out-dir>/tokens_used.json         the segments scored (after --segment / --max-tokens), kl_report.py's tokens format

exactly the format of tools/reference/full_logits_golden.py and gemma/full_logits_gemma.py (row i = log p(next |
tokens[0..i]), fp32 log_softmax, floored at -1e4, cast to fp16), so `kl_report.py` is shared unchanged.

Tokens: `kl_corpus/tokens_canon.json` (Qwen tokenizer, no BOS; the model's vocabulary is the 27B's, 248320).

QSA ties (plan Q23): `--tie-rule recent` (default) is the rule r4dx implements, so the gate compares like with like;
`--tie-rule hf` reproduces HF's own torch.topk choice. KL between a "hf" dump and a "recent" dump is the noise
floor of the rule: run the script twice into two directories and `kl_report.py --ref-dir A --test-dir B`.

GPU (hand to the user / take the gpu lock; device 1 only):

    $env:HIP_VISIBLE_DEVICES='1'; D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe `
        tools\\reference\\qwen4exp\\full_logits_qwen4exp.py --out-dir E:\\models\\r4dx\\qwen4exp\\kl\\ref-recent

`--tiny` runs the same code on the tiny-q4exp checkpoint on CPU (its own random segments, vocabulary 4096).
`--max-layers N` runs only the first N layers: NOT a valid golden (meta records debug_max_layers).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path

if "--tiny" in sys.argv:
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from qwen4exp import tiny  # noqa: E402
from qwen4exp.common_qwen4exp import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    DEFAULT_SIDECAR,
    KL_CORPUS_DIR,
    QWEN4EXP_OUT_DIR,
    TINY_DIR,
    resolve_device,
    sha256_file,
)
from qwen4exp.ref import (  # noqa: E402
    ATTN_IMPLS,
    DEFAULT_ATTN_QUERY_CHUNK,
    LOGITS_MODES,
    LOGPROB_CLAMP,
    TIE_RULES,
    Qwen4ExpStreamingReference,
    token_ids_sha256,
)

DEFAULT_OUT = QWEN4EXP_OUT_DIR / "kl" / "ref"


def load_tokens_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if "segments" not in doc:
        raise ValueError(f"{path} has no 'segments' key")
    return doc


def check_segment(name: str, ids: list[int], vocab: int) -> None:
    if len(ids) < 2:
        raise SystemExit(f"segment {name!r} has {len(ids)} tokens; need at least 2")
    if max(ids) >= vocab or min(ids) < 0:
        raise SystemExit(f"segment {name!r} has an id outside [0, {vocab})")


def tiny_segments(seed: int) -> list[dict]:
    g = torch.Generator().manual_seed(seed)
    segs = []
    for k, n in enumerate((20, 90)):
        ids = torch.randint(0, 4000, (n,), generator=g)
        if k == 1:
            ids[40] = tiny.TINY_EOS
        segs.append({"name": f"tiny_{k}", "token_ids": [int(v) for v in ids]})
    return segs


def run_segment(ref: Qwen4ExpStreamingReference, name: str, ids: list[int], out_dir: Path, args,
                tokens_sha: str | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.logprobs.f16"
    if ref.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with open(bin_path, "wb") as f:
        st = ref.logprobs_to_file(ids, f, mode=args.logits_mode, chunk=args.lm_head_chunk, row_block=args.row_block)
    rows = len(ids) - 1
    expect = rows * ref.vocab_size * 2
    if bin_path.stat().st_size != expect:
        raise RuntimeError(f"{bin_path} is {bin_path.stat().st_size} bytes, expected {expect}")
    last = [{"id": tid, "logit": logit, "prob": prob}
            for (logit, tid), prob in zip(st["last_top"], st["last_probs"])]
    peak = torch.cuda.max_memory_allocated() / 2**30 if ref.device.type == "cuda" else None
    meta = {
        "T": len(ids), "V": ref.vocab_size, "dtype": "float16", "rows": rows, "source": ref.source_tag,
        "torch_dtype": "bfloat16", "sha256_of_token_ids_json": token_ids_sha256(ids), "segment": name,
        "model_dir": str(ref.model_dir), "impl": f"qwen4exp-streaming-{ref.attn}-{ref.tie_rule}",
        "logits_mode": args.logits_mode, "attn": ref.attn, "attn_query_chunk": ref.attn_query_chunk,
        "lm_head_chunk": args.lm_head_chunk, "row_block": args.row_block, "logprob_clamp": LOGPROB_CLAMP,
        "row_semantics": "row i = log_softmax(logits at position i) = log p(next | token_ids[0..i]), fp32 then fp16",
        "max_abs_logsumexp_fp16": st["worst_lse"], "mean_nll_next_token": st["nll"],
        "perplexity": float(torch.exp(torch.tensor(st["nll"]))), "last_position_top_k": last,
        "seconds_stack": st["seconds_stack"], "seconds_lm_head": st["seconds_lm_head"], "peak_vram_gib": peak,
        "tokens_file_sha256": tokens_sha, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **ref.segment_meta(),
    }
    with open(out_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[full_logits_qwen4exp] {name}: T={len(ids)} stack={st['seconds_stack']:.1f}s "
          f"lm_head={st['seconds_lm_head']:.1f}s |lse|max={st['worst_lse']:.2e} ppl={meta['perplexity']:.3f} "
          f"-> {bin_path.name} ({bin_path.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--ngram-sidecar", type=Path, default=None,
                    help=f"n-gram sidecar (default {DEFAULT_SIDECAR} when it exists; else the checkpoint's own shards)")
    ap.add_argument("--tokens", type=Path, default=KL_CORPUS_DIR / "tokens_canon.json")
    ap.add_argument("--out-dir", type=Path, default=None, help=f"default {DEFAULT_OUT}")
    ap.add_argument("--segment", action="append", default=None, help="segment name (repeatable / comma list)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--tie-rule", default="recent", choices=TIE_RULES)
    ap.add_argument("--attn", default="sdpa", choices=ATTN_IMPLS)
    ap.add_argument("--attn-query-chunk", type=int, default=DEFAULT_ATTN_QUERY_CHUNK,
                    help="QSA layers process queries in blocks of this many tokens above this length")
    ap.add_argument("--logits-mode", default="fp32", choices=LOGITS_MODES)
    ap.add_argument("--lm-head-chunk", type=int, default=32768, help="vocab rows of lm_head multiplied at once")
    ap.add_argument("--row-block", type=int, default=128, help="sequence rows per log_softmax block")
    ap.add_argument("--max-tokens", type=int, default=None, help="truncate every segment (NOT a valid reference)")
    ap.add_argument("--max-layers", type=int, default=None, help="first N layers only: NOT a valid golden")
    ap.add_argument("--tiny", action="store_true", help="tiny-q4exp on CPU with its own random segments")
    ap.add_argument("--tiny-seed", type=int, default=3)
    args = ap.parse_args(argv)

    if args.tiny:
        args.model_dir = TINY_DIR / "hf"
        args.ngram_sidecar = TINY_DIR / "sidecar.r4dx-ngram"
        args.device = "cpu"
    elif args.ngram_sidecar is None and DEFAULT_SIDECAR.exists():
        args.ngram_sidecar = DEFAULT_SIDECAR
    out_dir = args.out_dir or (TINY_DIR / "kl" / "ref" if args.tiny else DEFAULT_OUT)
    device = resolve_device(args.device)  # the device rule: cuda only with HIP_VISIBLE_DEVICES='1'

    if args.tiny:
        doc, segs, tokens_sha = {"tokenizer": "tiny"}, tiny_segments(args.tiny_seed), None
    else:
        doc = load_tokens_file(args.tokens)
        segs = doc["segments"]
        tokens_sha = sha256_file(args.tokens)
    wanted = None if not args.segment else [s for a in args.segment for s in a.split(",") if s]
    segs = [s for s in segs if wanted is None or s["name"] in wanted]
    if not segs:
        raise SystemExit("no segments selected")
    if args.max_tokens:
        segs = [{"name": s["name"], "token_ids": s["token_ids"][: args.max_tokens]} for s in segs]

    t0 = time.perf_counter()
    ref = Qwen4ExpStreamingReference(args.model_dir, device, tie_rule=args.tie_rule, ngram_sidecar=args.ngram_sidecar,
                                     max_layers=args.max_layers, attn=args.attn, attn_query_chunk=args.attn_query_chunk,
                                     verbose=True)
    for s in segs:
        check_segment(s["name"], s["token_ids"], ref.vocab_size)
    print(f"[full_logits_qwen4exp] {ref.n_layers} layers, hidden={ref.arch.hidden}, V={ref.vocab_size}, "
          f"tie_rule={args.tie_rule}, attn={args.attn}, logits={args.logits_mode}, ready in {time.perf_counter() - t0:.1f}s",
          flush=True)
    if args.max_layers:
        print("[full_logits_qwen4exp] WARNING --max-layers: NOT a valid golden")

    run = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "model_dir": str(args.model_dir),
        "config_sha256": sha256_file(args.model_dir / "config.json"), "tokens_file": str(args.tokens),
        "tokens_file_sha256": tokens_sha, "tokenizer": doc.get("tokenizer"), "device": str(device),
        "source": ref.source_tag, "weights": "int4 AWQ dequantized to bf16" if ref.packed else "bf16 checkpoint",
        "torch_dtype": "bfloat16", "tie_rule": args.tie_rule, "attn": args.attn, "logits_mode": args.logits_mode,
        "lm_head_chunk": args.lm_head_chunk, "row_block": args.row_block, "n_layers": ref.n_layers,
        "debug_max_layers": args.max_layers, "vocab_size": ref.vocab_size, "arch": ref.arch.summary(),
        "ngram_sidecar": str(args.ngram_sidecar) if args.ngram_sidecar else None,
        "logprob_clamp": LOGPROB_CLAMP, "torch_version": torch.__version__, "segments": {},
    }
    for s in segs:
        run["segments"][s["name"]] = run_segment(ref, s["name"], s["token_ids"], out_dir, args, tokens_sha)
    with open(out_dir / "tokens_used.json", "w", encoding="utf-8") as f:
        json.dump({**{k: v for k, v in doc.items() if k != "segments"}, "segments": segs}, f)
    with open(out_dir / "reference_run.json", "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, default=str)
    print(f"[full_logits_qwen4exp] wrote {out_dir / 'reference_run.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
