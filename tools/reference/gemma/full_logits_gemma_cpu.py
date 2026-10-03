"""tools/reference/gemma/full_logits_gemma_cpu.py  (docs/gemma4-plan.md 9.9: the fp32-truth KL gate)

CPU-only teacher-forced log-prob dumps for the Gemma M1 gate, in kl_report.py's shared format
(`<seg>.logprobs.f16` rows 0..T-2 [T-1, V] fp16 + `<seg>.meta.json`), for any mix of tokens files
(chat_gemma.json with score masks, tokens_gemma.json raw segments truncated with --max-tokens).

Two variants, both streaming the 48 decoder layers from the bf16 checkpoint ONE LAYER AT A TIME FOR ALL
SEGMENTS IN LOCKSTEP (each layer is read from disk once per run, not once per segment):

  truth     exact fp32 upcast of the bf16 weights; fp32 activations, eager attention, fp32 final norm,
            fp32 lm_head + softcap.  Embedding = bf16 table * bf16(sqrt(3840)) = 62.0 exactly as HF, then
            widened.  This is the yardstick (the bf16 stack is not: residual stream reaches 100-300 and
            bf16 rounding is amplified, see tools/reference/gemma/investigate/).
  bf16sdpa  stock HF numerics: bf16 weights/activations, sdpa attention.  Two heads from the same hidden
            states:  `hf-bf16` lm_head (bf16 matmul, softcap in bf16: what HF computes; written to
            --out-dir) and `fp32` lm_head (written to --out-dir-fp32head if given).  KL(truth||bf16sdpa)
            per sequence group is the NOISE TERM the r4dx gate is relative to.

Resumable: the hidden states of all segments are checkpointed after every layer in <out-dir>/_ckpt.

NO GPU: the environment variables below are set before importing torch (an empty value does NOT hide a
GPU on Windows; "-1" does) and `torch.cuda.is_available()` must be False.

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe tools\\reference\\gemma\\full_logits_gemma_cpu.py `
        --variant truth --tokens tools\\reference\\kl_corpus\\chat_gemma.json `
        --tokens tools\\reference\\kl_corpus\\tokens_gemma.json --max-tokens 512 `
        --segment-raw cpp_source,thai_prose --out-dir D:\\models\\r4dx\\huihui-gemma\\kl\\fp32\\truth
"""

from __future__ import annotations

import os

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import argparse  # noqa: E402
import datetime as dt  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from collections import UserDict  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

assert torch.cuda.is_available() is False, "torch.cuda.is_available() is True: refusing to run (CPU only)"

from gemma.arch import FULL, SLIDING  # noqa: E402
from gemma.common_gemma import BOS_ID, DEFAULT_MODEL_DIR, sha256_file  # noqa: E402
from gemma.ref import (  # noqa: E402
    LOGPROB_CLAMP,
    GemmaReference,
    add_tiny_args,
    additive_mask,
    build_tiny_checkpoint,
    token_ids_sha256,
)

VARIANTS = ("truth", "bf16sdpa")


def load_segments(paths: list[Path], max_tokens: int | None, raw_names: list[str] | None,
                  only: list[str] | None) -> list[dict]:
    """Segments of every tokens file. A file with score masks (chat) is taken whole; a raw file contributes
    the segments named in `raw_names` (default all), truncated to `max_tokens`."""
    segs, seen = [], set()
    for p in paths:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        if doc.get("tokenizer_arch") != "gemma4":
            raise SystemExit(f"{p}: not a gemma4 tokens file")
        for s in doc["segments"]:
            chat = "score_mask" in s
            if not chat and raw_names is not None and s["name"] not in raw_names:
                continue
            if only and s["name"] not in only:
                continue
            ids = [int(t) for t in s["token_ids"]]
            if not chat and max_tokens:
                ids = ids[:max_tokens]
            if ids[0] != BOS_ID or ids[1:].count(BOS_ID) != 0 and not chat:
                raise SystemExit(f"segment {s['name']}: BOS rule violated")
            if s["name"] in seen:
                raise SystemExit(f"duplicate segment name {s['name']}")
            seen.add(s["name"])
            segs.append({"name": s["name"], "token_ids": ids, "chat": chat, "file": str(p)})
    if not segs:
        raise SystemExit("no segments selected")
    return segs


def log_softmax_rows(logits: torch.Tensor) -> torch.Tensor:
    return torch.log_softmax(logits, dim=-1).clamp_min(LOGPROB_CLAMP)


def write_head(ref: GemmaReference, hid: dict[str, torch.Tensor], segs: list[dict], head: str, out_dir: Path,
               meta_common: dict, chunk: int = 32768) -> None:
    """lm_head over the final hidden states of ALL segments at once, vocab chunk outermost (each table chunk
    is widened once), then per-segment log_softmax and fp16 dump. head: 'fp32' or 'hf-bf16'."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = ref.arch.softcap
    names = [s["name"] for s in segs]
    rows = {n: hid[n].shape[0] - 1 for n in names}                       # rows 0..T-2
    H = torch.cat([hid[n][:-1] for n in names], dim=0)                   # [N, hidden]
    N = H.shape[0]
    t0 = time.perf_counter()
    logits = torch.empty(N, ref.vocab_size, dtype=torch.float32)
    Hf = H.float()
    for s in range(0, ref.vocab_size, chunk):
        e = min(s + chunk, ref.vocab_size)
        w = ref.table[s:e]                                               # bf16
        if head == "fp32":
            blk = Hf @ w.float().T
            blk = torch.tanh(blk / cap) * cap
        else:
            blk = torch.nn.functional.linear(H.to(torch.bfloat16), w)   # bf16, as nn.Linear
            blk = (torch.tanh(blk / cap) * cap).float()                  # softcap in bf16 (as HF)
        logits[:, s:e] = blk
        print(f"[cpu-ref]   head {head}: vocab {e}/{ref.vocab_size} ({time.perf_counter() - t0:.0f}s)", flush=True)
    off = 0
    for sg in segs:
        n, ids = sg["name"], sg["token_ids"]
        r = rows[n]
        path = out_dir / f"{n}.logprobs.f16"
        nll, worst = 0.0, 0.0
        with open(path, "wb") as f:
            for s in range(0, r, 64):
                e = min(s + 64, r)
                lp = log_softmax_rows(logits[off + s:off + e])
                worst = max(worst, torch.logsumexp(lp, dim=-1).abs().max().item())
                tgt = torch.as_tensor(ids[s + 1:e + 1])
                nll -= lp[torch.arange(e - s), tgt].sum().item()
                f.write(np.ascontiguousarray(lp.to(torch.float16).numpy()).tobytes())
        off += r
        assert path.stat().st_size == r * ref.vocab_size * 2
        meta = {"T": len(ids), "V": ref.vocab_size, "dtype": "float16", "rows": r, "segment": n,
                "sha256_of_token_ids_json": token_ids_sha256(ids), "head": head,
                "mean_nll_next_token": nll / r, "perplexity": float(np.exp(nll / r)),
                "max_abs_logsumexp_fp16": worst,
                "row_semantics": "row i = log_softmax(softcapped logits at position i) = log p(next | ids[0..i])",
                "logprob_clamp": LOGPROB_CLAMP, **meta_common}
        (out_dir / f"{n}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[cpu-ref] {head} {n}: T={len(ids)} nll={nll / r:.4f} ppl={meta['perplexity']:.2f} -> {path.name}",
              flush=True)
    del logits


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokens", type=Path, action="append", default=None, help="repeatable")
    ap.add_argument("--max-tokens", type=int, default=512, help="truncation of RAW (mask-less) segments")
    ap.add_argument("--segment-raw", default=None, help="comma list of raw segment names to include (default all)")
    ap.add_argument("--segment", default=None, help="comma list: only these segment names (any file)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--out-dir-fp32head", type=Path, default=None,
                    help="bf16sdpa only: also write the fp32-lm_head log-probs of the same hidden states here")
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--debug-max-layers", type=int, default=None, help="NOT a valid reference")
    ap.add_argument("--no-resume", action="store_true")
    add_tiny_args(ap)
    args = ap.parse_args(argv)
    torch.set_num_threads(args.threads)

    tmp = None
    if args.tiny:
        tmp = tempfile.TemporaryDirectory()
        args.model_dir = build_tiny_checkpoint(Path(tmp.name) / "model", args.tiny_seed)
        g = torch.Generator().manual_seed(args.tiny_seed + 3)
        segs = [{"name": f"tiny_{k}", "chat": False, "file": "<tiny>",
                 "token_ids": [BOS_ID] + [int(v) for v in torch.randint(4, 128, (n,), generator=g)]}
                for k, n in enumerate((20, 31))]
        args.out_dir = args.out_dir or Path(tmp.name) / args.variant
        tokens_sha = None
    else:
        if not args.tokens:
            raise SystemExit("--tokens required")
        segs = load_segments(args.tokens, args.max_tokens,
                             None if args.segment_raw is None else args.segment_raw.split(","),
                             None if args.segment is None else args.segment.split(","))
        tokens_sha = {str(p): sha256_file(p) for p in args.tokens}
    if args.out_dir is None:
        raise SystemExit("--out-dir required")
    out_dir = args.out_dir
    truth = args.variant == "truth"
    work_dtype = torch.float32 if truth else torch.bfloat16
    attn = "eager" if truth else "sdpa"
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    ref = GemmaReference(args.model_dir, torch.device("cpu"), resident=False, attn=attn, gemm_guard=None)
    for s in segs:
        if max(s["token_ids"]) >= ref.vocab_size:
            raise SystemExit(f"segment {s['name']}: id outside vocab")
    n_layers = ref.n_layers if args.debug_max_layers is None else args.debug_max_layers
    cfg = ref.text_config
    cfg._attn_implementation = attn
    print(f"[cpu-ref] variant={args.variant} work_dtype={work_dtype} attn={attn} layers={n_layers} "
          f"segments={[(s['name'], len(s['token_ids'])) for s in segs]} threads={torch.get_num_threads()}", flush=True)

    ckpt = out_dir / "_ckpt" / "state.pt"
    start_layer, states = 0, {}
    if ckpt.exists() and not args.no_resume:
        c = torch.load(ckpt, weights_only=True)
        if c["names"] == [s["name"] for s in segs] and c["variant"] == args.variant:
            start_layer, states = c["next_layer"], c["states"]
            print(f"[cpu-ref] resuming at layer {start_layer}", flush=True)
    pos, pe, masks = {}, {}, {}
    with torch.no_grad():
        for s in segs:
            n, ids = s["name"], s["token_ids"]
            T = len(ids)
            h0 = ref.embed(ids).unsqueeze(0)                         # bf16 table * bf16 scale, as HF
            if n not in states:
                states[n] = h0.to(work_dtype)
            pos[n] = torch.arange(T).unsqueeze(0)
            pe[n] = {lt: ref.rotary(states[n], pos[n], lt) for lt in (FULL, SLIDING)}
            masks[n] = {FULL: additive_mask(T, T, 0, None, torch.device("cpu"), work_dtype),
                        SLIDING: additive_mask(T, T, 0, ref.arch.window, torch.device("cpu"), work_dtype)}
        for i in range(start_layer, n_layers):
            t0 = time.perf_counter()
            layer = ref.build_layer(i)
            if truth:
                layer = layer.float()
            lt = ref.layer_types[i]
            for s in segs:
                n = s["name"]
                states[n] = layer(states[n], shared_kv_states=UserDict(), position_embeddings=pe[n][lt],
                                  attention_mask=masks[n][lt], position_ids=pos[n], past_key_values=None)
            del layer
            hmax = max(float(states[n].abs().max()) for n in states)
            print(f"[cpu-ref] L{i:02d} {lt[:4]} {time.perf_counter() - t0:.1f}s max|h|={hmax:.1f} "
                  f"elapsed={(time.perf_counter() - t_start) / 60:.1f}min", flush=True)
            if not args.tiny:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"names": [s["name"] for s in segs], "variant": args.variant, "next_layer": i + 1,
                            "states": states}, str(ckpt) + ".tmp")
                os.replace(str(ckpt) + ".tmp", ckpt)
        hid = {}
        for s in segs:
            x = states[s["name"]][0]
            if truth:
                x = x * torch.pow(x.pow(2).mean(-1, keepdim=True) + ref.arch.eps, -0.5)
                hid[s["name"]] = x * ref.final_norm_w.float()
            else:
                hid[s["name"]] = ref.final_norm(x)
        meta_common = {
            "source": "reference-fp32-truth" if truth else "hf-bf16-sdpa", "variant": args.variant,
            "model_dir": str(args.model_dir), "attn": attn, "work_dtype": str(work_dtype),
            "n_layers": n_layers, "debug_max_layers": args.debug_max_layers, "torch_version": torch.__version__,
            "tokens_files": tokens_sha, "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "stack_seconds": time.perf_counter() - t_start,
        }
        heads = [("fp32", out_dir)] if truth else [("hf-bf16", out_dir)]
        if not truth and args.out_dir_fp32head is not None:
            heads.append(("fp32", args.out_dir_fp32head))
        for head, d in heads:
            write_head(ref, hid, segs, head, d, {**meta_common, "variant": args.variant + ("" if head != "fp32" or truth else "_fp32head")})
    for d in {out_dir, *([args.out_dir_fp32head] if args.out_dir_fp32head else [])}:
        (d / "run.json").write_text(json.dumps({
            "variant": args.variant, "segments": {s["name"]: len(s["token_ids"]) for s in segs},
            "total_seconds": time.perf_counter() - t_start, "n_layers": n_layers,
            "note": "see <seg>.meta.json"}, indent=2), encoding="utf-8")
    print(f"[cpu-ref] done in {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
