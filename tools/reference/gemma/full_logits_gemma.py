"""tools/reference/gemma/full_logits_gemma.py  (M0-8, rung-4 / KL reference, "mode A")

The bf16 Huihui-Gemma-4-12B teacher-forced log-probability dump the KL gate compares r4dx against.
For every segment of a tokens file it runs the full text stack from a fresh context (position 0, causal
+ sliding mask, no cache) and writes

    <out-dir>/<segment>.logprobs.f16   rows 0..T-2 of log_softmax(logits), [T-1, 262144] float16
    <out-dir>/<segment>.meta.json      kl_report.py's shared format (T, V, rows, sha256_of_token_ids_json, ...)
    <out-dir>/reference_run.json       the run record (versions, modes, GemmGuard events, noise-floor pairing)

exactly the format of tools/reference/full_logits_golden.py (row i = log p(next | tokens[0..i]), fp32
log_softmax, floored at -1e4, cast to fp16), so `kl_report.py` is shared unchanged.

Tokens: `kl_corpus/tokens_gemma.json` (and `tokens_gemma_long.json`, 1600 tokens: the 1024-slot sliding
ring wraps) -- every segment starts with BOS=2, which is teacher-forced as an ordinary first token on
BOTH sides (HF adds none to raw text). Qwen ids are refused (`tokenizer_arch` must be gemma4).

Modes:
  mode A (default, --resident): the whole bf16 model on the device, ~24 GiB (23.9 GB checkpoint) plus the
      tied 1.9 GB embedding table; lm_head in vocab chunks of 32768 rows, row blocks of 128.
  mode B (--streaming): one decoder layer at a time from disk (slow, ~2 GiB VRAM) if VRAM is tight.
  --logits-mode fp32 (default): lm_head accumulated in fp32 and the softcap in fp32 (the plan's reference);
      hf-bf16: lm_head output rounded to bf16 and softcapped in bf16, which is what HF computes
      (docs/gemma4-semantics.md item 14). Run both once and compare to know what the choice is worth.

Noise floor (plan: "run twice"): `--noise-floor-out DIR` repeats every segment in the same process after
the first pass with a different reduction order (sdpa attention instead of eager, lm_head chunk 16384,
row block 64) into DIR; `kl_report.py --ref-dir <out-dir> --test-dir DIR` is then the bf16 self-KL noise
floor of this reference that the gate is measured against. Without the flag, run the script twice with
`--attn sdpa --lm-head-chunk 16384 --row-block 64 --out-dir <other>`.

GPU (hand this to the user; never run by an agent):

    $env:HIP_VISIBLE_DEVICES='1'; D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe `
        tools\\reference\\gemma\\full_logits_gemma.py --tokens tools\\reference\\kl_corpus\\tokens_gemma.json `
        --out-dir <models root>\\r4dx\\huihui-gemma\\kl\\ref --noise-floor-out <models root>\\r4dx\\huihui-gemma\\kl\\ref-noise

`--tiny` runs the same code on a tiny random checkpoint on CPU (the M0-6 smoke).
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import tempfile
import time
from pathlib import Path

if "--tiny" in sys.argv:
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from gemma.common_gemma import (  # noqa: E402
    BOS_ID,
    DEFAULT_MODEL_DIR,
    KL_CORPUS_DIR,
    load_tokenizer,
    resolve_device,
    sha256_file,
)
from gemma.ref import (  # noqa: E402
    ATTN_IMPLS,
    LOGITS_MODES,
    LOGPROB_CLAMP,
    GemmaReference,
    GemmGuardState,
    add_tiny_args,
    build_tiny_checkpoint,
    token_ids_sha256,
)

DEFAULT_OUT = KL_CORPUS_DIR.parent / "kl_out" / "gemma" / "ref"


def load_tokens_file(path: Path, allow_foreign: bool = False) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if "segments" not in doc:
        raise ValueError(f"{path} has no 'segments' key")
    if doc.get("tokenizer_arch") != "gemma4" and not allow_foreign:
        raise SystemExit(f"[full_logits_gemma] {path} is not a gemma4 tokens file (tokenizer_arch="
                         f"{doc.get('tokenizer_arch')!r}); Qwen ids must never be used for Gemma. Make one with "
                         "make_tokens_json.py --arch gemma4 (or pass --allow-foreign-tokens)")
    return doc


def check_segment(name: str, ids: list[int], vocab: int, require_bos: bool) -> None:
    if len(ids) < 2:
        raise SystemExit(f"segment {name!r} has {len(ids)} tokens; need at least 2")
    if max(ids) >= vocab or min(ids) < 0:
        raise SystemExit(f"segment {name!r} has an id outside [0, {vocab})")
    if require_bos and (ids[0] != BOS_ID or (len(ids) > 1 and ids[1] == BOS_ID)):
        raise SystemExit(f"segment {name!r} must start with exactly one BOS ({BOS_ID}) -- the KL convention "
                         "teacher-forces BOS first on both sides")


def run_segment(ref: GemmaReference, name: str, ids: list[int], out_dir: Path, args, tok, tag: str,
                tokens_sha: str | None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.logprobs.f16"
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    with open(bin_path, "wb") as f:
        st = ref.logprobs_to_file(ids, f, mode=args.logits_mode, chunk=args.lm_head_chunk, row_block=args.row_block)
    rows = len(ids) - 1
    expect = rows * ref.vocab_size * 2
    if bin_path.stat().st_size != expect:
        raise RuntimeError(f"{bin_path} is {bin_path.stat().st_size} bytes, expected {expect}")
    last = []
    for (logit, tid), prob in zip(st["last_top"], st["last_probs"]):
        e = {"id": tid, "logit": logit, "prob": prob}
        try:
            e["piece"] = tok.decode([tid])
        except Exception:
            pass
        last.append(e)
    peak = torch.cuda.max_memory_allocated() / 2**30 if args.device == "cuda" else None
    meta = {
        "T": len(ids), "V": ref.vocab_size, "dtype": "float16", "rows": rows, "source": ref.source_tag,
        "torch_dtype": "bfloat16", "sha256_of_token_ids_json": token_ids_sha256(ids), "segment": name,
        "model_dir": str(ref.model_dir), "impl": f"gemma-{'resident' if ref.resident else 'streaming'}-{ref.attn}",
        "logits_mode": args.logits_mode, "attn": ref.attn, "lm_head_chunk": args.lm_head_chunk,
        "row_block": args.row_block, "pass": tag, "bos_token_id": ids[0], "logprob_clamp": LOGPROB_CLAMP,
        "row_semantics": "row i = log_softmax(softcapped logits at position i) = log p(next | token_ids[0..i]), "
                         "fp32 then fp16; token_ids[0] is BOS",
        "max_abs_logsumexp_fp16": st["worst_lse"], "mean_nll_next_token": st["nll"],
        "perplexity": float(torch.exp(torch.tensor(st["nll"]))), "last_position_top_k": last,
        "seconds_stack": st["seconds_stack"], "seconds_lm_head": st["seconds_lm_head"], "peak_vram_gib": peak,
        "tokens_file_sha256": tokens_sha, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        **ref.segment_meta(),
    }
    with open(out_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"[full_logits_gemma] {tag} {name}: T={len(ids)} stack={st['seconds_stack']:.1f}s "
          f"lm_head={st['seconds_lm_head']:.1f}s |lse|max={st['worst_lse']:.2e} ppl={meta['perplexity']:.3f} "
          f"-> {bin_path.name} ({bin_path.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokens", type=Path, default=KL_CORPUS_DIR / "tokens_gemma.json")
    ap.add_argument("--out-dir", type=Path, default=None, help=f"default {DEFAULT_OUT}")
    ap.add_argument("--segment", action="append", default=None, help="segment name (repeatable / comma list)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--streaming", action="store_true", help="mode B: one layer at a time (default: mode A, resident)")
    ap.add_argument("--attn", default="eager", choices=ATTN_IMPLS)
    ap.add_argument("--logits-mode", default="fp32", choices=LOGITS_MODES)
    ap.add_argument("--lm-head-chunk", type=int, default=32768, help="vocab rows of lm_head resident at once")
    ap.add_argument("--row-block", type=int, default=128, help="sequence rows per log_softmax block")
    ap.add_argument("--max-tokens", type=int, default=None, help="truncate every segment (NOT a valid reference)")
    ap.add_argument("--debug-max-layers", type=int, default=None, help="first N layers only: NOT a valid golden")
    ap.add_argument("--noise-floor-out", type=Path, default=None,
                    help="after the first pass, repeat in-process with sdpa attention / chunk 16384 / row block 64 "
                         "into this dir (the bf16 self-KL noise floor pairing)")
    ap.add_argument("--no-gemm-guard", action="store_true",
                    help="do not recompute non-finite / >1e5 linear outputs (see ref.GemmGuardState)")
    ap.add_argument("--allow-foreign-tokens", action="store_true", help="accept a tokens file without tokenizer_arch gemma4")
    ap.add_argument("--no-require-bos", action="store_true", help="accept segments that do not start with BOS=2")
    add_tiny_args(ap)
    args = ap.parse_args(argv)

    tmp = None
    if args.tiny:
        tmp = tempfile.TemporaryDirectory()
        args.model_dir = build_tiny_checkpoint(Path(tmp.name) / "model", args.tiny_seed)
        args.device = "cpu"
        if args.out_dir is None:
            args.out_dir = Path(tmp.name) / "ref"
    out_dir = args.out_dir or DEFAULT_OUT
    if args.noise_floor_out is not None and args.noise_floor_out.resolve() == out_dir.resolve():
        raise SystemExit("--noise-floor-out must differ from --out-dir")
    device = resolve_device(args.device)  # the device rule (R4DX_REF_ALLOWED_DEVICES) for cuda

    if args.tiny:
        # a tiny random model has vocab 128: draw segments of its own, BOS first
        g = torch.Generator().manual_seed(args.tiny_seed + 3)
        segs = [{"name": f"tiny_{k}", "token_ids": [BOS_ID] + [int(v) for v in torch.randint(4, 128, (n,), generator=g)]}
                for k, n in enumerate((20, 31))]
        doc = {"tokenizer": "tiny", "tokenizer_arch": "gemma4"}
        tokens_sha = None
        tok = load_tokenizer(tiny_vocab=128)
    else:
        doc = load_tokens_file(args.tokens, args.allow_foreign_tokens)
        segs = doc["segments"]
        tokens_sha = sha256_file(args.tokens)
        try:
            tok = load_tokenizer(None)
        except Exception as exc:  # decoding pieces is a nicety
            print(f"[full_logits_gemma] tokenizer unavailable ({type(exc).__name__}); ids only")
            tok = load_tokenizer(tiny_vocab=8)
    wanted = None if not args.segment else [s for a in args.segment for s in a.split(",") if s]
    segs = [s for s in segs if wanted is None or s["name"] in wanted]
    if not segs:
        raise SystemExit("no segments selected")
    if args.max_tokens:
        segs = [{"name": s["name"], "token_ids": s["token_ids"][: args.max_tokens]} for s in segs]

    guard = None if args.no_gemm_guard else GemmGuardState()
    t0 = time.perf_counter()
    ref = GemmaReference(args.model_dir, device, max_layers=args.debug_max_layers, resident=not args.streaming,
                         attn=args.attn, gemm_guard=guard)
    for s in segs:
        check_segment(s["name"], s["token_ids"], ref.vocab_size, not args.no_require_bos)
    print(f"[full_logits_gemma] {ref.n_layers} layers, hidden={ref.arch.hidden}, V={ref.vocab_size}, "
          f"{'resident' if ref.resident else 'streaming'}, attn={args.attn}, logits={args.logits_mode}, "
          f"ready in {time.perf_counter() - t0:.1f}s", flush=True)
    if args.debug_max_layers:
        print("[full_logits_gemma] WARNING --debug-max-layers: NOT a valid golden")

    run = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "model_dir": str(args.model_dir),
        "config_sha256": sha256_file(args.model_dir / "config.json"), "tokens_file": str(args.tokens),
        "tokens_file_sha256": tokens_sha, "tokenizer": doc.get("tokenizer"), "tokenizer_arch": doc.get("tokenizer_arch"),
        "tokenizer_provenance": doc.get("tokenizer_provenance"), "bos_token_id": BOS_ID, "device": str(device),
        "source": ref.source_tag, "weights": "bf16 checkpoint", "torch_dtype": "bfloat16",
        "mode": "A-resident" if ref.resident else "B-streaming", "attn": args.attn, "logits_mode": args.logits_mode,
        "lm_head_chunk": args.lm_head_chunk, "row_block": args.row_block, "n_layers": ref.n_layers,
        "debug_max_layers": args.debug_max_layers, "vocab_size": ref.vocab_size, "arch": ref.arch.summary(),
        "logprob_clamp": LOGPROB_CLAMP, "torch_version": torch.__version__, "segments": {},
    }
    for s in segs:
        run["segments"][s["name"]] = run_segment(ref, s["name"], s["token_ids"], out_dir, args, tok, "primary", tokens_sha)

    if args.noise_floor_out is not None:
        # Same weights, different reduction order: sdpa attention (the text config is shared by every layer,
        # read at forward time), another lm_head chunking and row blocking.
        ref.text_config._attn_implementation = "sdpa"
        ref.attn = "sdpa"
        a2 = argparse.Namespace(**{**vars(args), "lm_head_chunk": 16384, "row_block": 64, "attn": "sdpa"})
        run["noise_floor"] = {"out_dir": str(args.noise_floor_out), "attn": "sdpa", "lm_head_chunk": 16384,
                              "row_block": 64, "segments": {}}
        for s in segs:
            run["noise_floor"]["segments"][s["name"]] = run_segment(
                ref, s["name"], s["token_ids"], args.noise_floor_out, a2, tok, "noise-floor", tokens_sha)
        with open(args.noise_floor_out / "reference_run.json", "w", encoding="utf-8") as f:
            json.dump({**run, "role": "noise-floor twin of " + str(out_dir)}, f, indent=2, default=str)
        print(f"[full_logits_gemma] noise floor: python tools\\reference\\kl_report.py --ref-dir {out_dir} "
              f"--test-dir {args.noise_floor_out} --tokens {args.tokens}")
    # the exact segments scored (after --segment / --max-tokens), in kl_report.py's tokens format
    with open(out_dir / "tokens_used.json", "w", encoding="utf-8") as f:
        json.dump({**{k: v for k, v in doc.items() if k != "segments"}, "segments": segs}, f)
    run["gemm_guard"] = guard.as_dict() if guard else None
    if guard:
        print(f"[full_logits_gemma] GemmGuard: {guard.recomputed} recomputed, {guard.confirmed_large} confirmed "
              f"large, {guard.failed} failed", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "reference_run.json", "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, default=str)
    print(f"[full_logits_gemma] wrote {out_dir / 'reference_run.json'}")
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
