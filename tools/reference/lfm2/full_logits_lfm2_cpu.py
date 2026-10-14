"""tools/reference/lfm2/full_logits_lfm2_cpu.py  (docs/lfm2-plan.md L0-8 / 6.2: the fp32-truth KL gate)

CPU-only teacher-forced log-prob dumps for the LFM2.5-24B-A2B gate, in kl_report.py's shared format
(`<seg>.logprobs.f16` rows 0..T-2 [T-1, V=128000] fp16 + `<seg>.meta.json`), for any mix of tokens files
(chat_lfm2.json with score masks, tokens_lfm2.json / tokens_lfm2_long.json raw segments).

Two variants, both streaming the 40 decoder layers from the bf16 checkpoint ONE LAYER AT A TIME FOR ALL
SEGMENTS IN LOCKSTEP (each layer is read from disk once per run; the FFN/MoE of all segments is one batched
GEMM set, the mixers run per segment):

  truth     exact fp32 upcast of the bf16 weights; fp32 activations, eager attention, fp32 rope tables, fp32
            router / sigmoid / top-k / expert bias, fp32 expert math, fp32 final norm and lm_head.
  bf16sdpa  stock HF numerics: bf16 weights/activations, sdpa attention, bf16 router and rope tables (the F32
            expert_bias buffer stays F32).  `hf-bf16` head (bf16 matmul output, what HF computes) is written to
            --out-dir; the `fp32` head over the same hidden states to --out-dir-fp32head if given.
            With --truth-dir (the finished truth dump) it also writes <out-dir>/noise_term.json: per gate group
            the pooled KL(truth||bf16sdpa), top-1, the thresholds that candidate epsilons would imply
            (kl_report.lfm2_noise_term), and the router-id agreement truth vs bf16sdpa per MoE layer.

Logit columns >= --real-vocab (125017: the tokenizer's highest id is 125016; 128000 rows is padding) are MASKED:
the log-softmax runs over the real columns and the padding columns are written as the clamp value, so r4dx
(which masks them in its sampler) compares on the same footing.

Resumable: the hidden states of all segments (and the routed expert ids so far) are checkpointed after every
layer in <out-dir>/_ckpt/state.pt.  Per-layer progress and timing are printed.  Every layer's routed ids are also
saved to <out-dir>/router_ids.npz (`L{layer}__{segment}` [T, 4]).

NO GPU: the environment variables below are set before importing torch ("-1" hides the GPU on Windows) and
`torch.cuda.is_available()` must be False.  RAM peak: one fp32 layer (<= 3.5 GB for an MoE layer: 2.4 GB stacked
experts + mixer/router) + one bf16 shard-read transient + the fp32 embedding (1.05 GB, head stage only) + hidden
states (segments x T x 2048 x 4 B) + the head block.

FULL RUN (one big host job; main session; two variants one after the other):

    $py = 'D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe'
    $K = 'tools\\reference\\kl_corpus'; $O = 'E:\\models\\r4dx\\lfm2\\kl\\fp32'
    & $py -I tools\\reference\\lfm2\\full_logits_lfm2_cpu.py --variant truth --tokens $K\\chat_lfm2.json `
        --tokens $K\\tokens_lfm2.json --tokens $K\\tokens_lfm2_long.json --max-tokens 512 --long-max-tokens 6144 `
        --out-dir $O\\truth --threads 12
    & $py -I tools\\reference\\lfm2\\full_logits_lfm2_cpu.py --variant bf16sdpa --tokens $K\\chat_lfm2.json `
        --tokens $K\\tokens_lfm2.json --tokens $K\\tokens_lfm2_long.json --max-tokens 512 --long-max-tokens 6144 `
        --out-dir $O\\bf16sdpa --truth-dir $O\\truth --threads 12
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
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

assert torch.cuda.is_available() is False, "torch.cuda.is_available() is True: refusing to run (CPU only)"

from lfm2.common_lfm2 import (  # noqa: E402
    BOS_ID,
    DEFAULT_MODEL_DIR,
    LOGPROB_CLAMP,
    REAL_VOCAB,
    peak_rss_gb,
    sha256_file,
    token_ids_sha256,
)
from lfm2.ref import Lfm2Reference, add_tiny_args, build_tiny_checkpoint, causal_mask, masked_logprobs  # noqa: E402

VARIANTS = ("truth", "bf16sdpa")


def load_segments(paths: list[Path], max_tokens: int | None, long_max_tokens: int | None,
                  raw_names: list[str] | None, only: list[str] | None, bos_id: int = BOS_ID) -> list[dict]:
    """Segments of every tokens file. A file with score masks (chat) is taken whole; a raw file contributes the
    segments named in `raw_names` (default all), truncated to `max_tokens` (`long_max_tokens` for a segment whose
    name ends in `_long`)."""
    segs, seen = [], set()
    for p in paths:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        if doc.get("tokenizer_arch") != "lfm2":
            raise SystemExit(f"{p}: not an lfm2 tokens file (tokenizer_arch={doc.get('tokenizer_arch')!r})")
        for s in doc["segments"]:
            chat = "score_mask" in s
            if not chat and raw_names is not None and s["name"] not in raw_names:
                continue
            if only and s["name"] not in only:
                continue
            ids = [int(t) for t in s["token_ids"]]
            if not chat:
                cap = long_max_tokens if s["name"].endswith("_long") else max_tokens
                if cap:
                    ids = ids[:cap]
            if ids[0] != bos_id or ids[1:].count(bos_id) != 0:
                raise SystemExit(f"segment {s['name']}: BOS rule violated")
            if s["name"] in seen:
                raise SystemExit(f"duplicate segment name {s['name']}")
            seen.add(s["name"])
            segs.append({"name": s["name"], "token_ids": ids, "chat": chat, "file": str(p)})
    if not segs:
        raise SystemExit("no segments selected")
    return segs


def write_head(ref: Lfm2Reference, hid: dict[str, torch.Tensor], segs: list[dict], head: str, out_dir: Path,
               meta_common: dict, real_vocab: int, row_block: int = 128) -> None:
    """lm_head over the final hidden states, row blocks outermost, per-segment masked log_softmax and fp16 dump.
    head: 'fp32' (fp32 accumulate over the widened bf16 table, resident once) or 'hf-bf16' (bf16 matmul output)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    table32 = ref.table.float() if head == "fp32" else None
    for sg in segs:
        n, ids = sg["name"], sg["token_ids"]
        H = hid[n][:-1]
        r = H.shape[0]
        t0 = time.perf_counter()
        path = out_dir / f"{n}.logprobs.f16"
        nll, worst = 0.0, 0.0
        with open(path, "wb") as f:
            for s in range(0, r, row_block):
                e = min(s + row_block, r)
                if head == "fp32":
                    lg = H[s:e].float() @ table32.T
                else:
                    lg = torch.nn.functional.linear(H[s:e].to(torch.bfloat16), ref.table).float()
                lp = masked_logprobs(lg, real_vocab)
                worst = max(worst, torch.logsumexp(lp[:, :real_vocab], dim=-1).abs().max().item())
                tgt = torch.as_tensor(ids[s + 1:e + 1])
                nll -= lp[torch.arange(e - s), tgt].sum().item()
                f.write(np.ascontiguousarray(lp.to(torch.float16).numpy()).tobytes())
        assert path.stat().st_size == r * ref.vocab_size * 2
        meta = {"T": len(ids), "V": ref.vocab_size, "real_vocab": real_vocab, "dtype": "float16", "rows": r,
                "segment": n, "sha256_of_token_ids_json": token_ids_sha256(ids), "head": head,
                "mean_nll_next_token": nll / r, "perplexity": float(np.exp(nll / r)),
                "max_abs_logsumexp_fp16": worst,
                "row_semantics": "row i = log_softmax over columns < real_vocab of the logits at position i = "
                                 "log p(next | ids[0..i]); columns >= real_vocab hold the clamp",
                "logprob_clamp": LOGPROB_CLAMP, **meta_common}
        (out_dir / f"{n}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[cpu-ref] {head} {n}: T={len(ids)} nll={nll / r:.4f} ppl={meta['perplexity']:.2f} "
              f"({time.perf_counter() - t0:.0f}s) -> {path.name}", flush=True)
    del table32


def router_agreement(truth_npz: Path, noise_npz: Path) -> dict:
    """Per MoE layer: fraction of (token) rows whose SET of 4 selected experts is identical, and of (token, slot)
    pairs of the truth that also appear in the other side's top-4, truth vs the noise side."""
    a, b = np.load(truth_npz), np.load(noise_npz)
    per: dict[int, list[int]] = {}
    for k in a.files:
        if k not in b.files:
            continue
        layer = int(k.split("__")[0][1:])
        x, y = np.sort(a[k], axis=1), np.sort(b[k], axis=1)
        same_set = int((x == y).all(axis=1).sum())
        hit = int(sum(len(set(r1) & set(r2)) for r1, r2 in zip(a[k].tolist(), b[k].tolist())))
        c = per.setdefault(layer, [0, 0, 0])
        c[0] += same_set
        c[1] += hit
        c[2] += a[k].shape[0]
    rows = sum(c[2] for c in per.values())
    return {"per_layer": {str(l): {"rows": c[2], "same_expert_set_frac": c[0] / c[2],
                                   "slot_overlap_frac": c[1] / (c[2] * a[next(iter(a.files))].shape[1])}
                          for l, c in sorted(per.items())},
            "overall_same_expert_set_frac": sum(c[0] for c in per.values()) / max(rows, 1),
            "overall_slot_overlap_frac": sum(c[1] for c in per.values()) / max(rows * a[next(iter(a.files))].shape[1], 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=VARIANTS)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokens", type=Path, action="append", default=None, help="repeatable")
    ap.add_argument("--max-tokens", type=int, default=512, help="truncation of RAW (mask-less) segments")
    ap.add_argument("--long-max-tokens", type=int, default=6144, help="truncation of raw segments named *_long")
    ap.add_argument("--segment-raw", default=None, help="comma list of raw segment names to include (default all)")
    ap.add_argument("--segment", default=None, help="comma list: only these segment names (any file)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--out-dir-fp32head", type=Path, default=None,
                    help="bf16sdpa only: also write the fp32-lm_head log-probs of the same hidden states here")
    ap.add_argument("--truth-dir", type=Path, default=None,
                    help="bf16sdpa only: the finished truth dump; writes noise_term.json (+ router agreement)")
    ap.add_argument("--real-vocab", type=int, default=REAL_VOCAB)
    ap.add_argument("--bos-id", type=int, default=BOS_ID, help="BOS id the segments must start with (tests only)")
    ap.add_argument("--epsilon-candidates", default="0.0025,0.005,0.01")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--debug-max-layers", type=int, default=None, help="NOT a valid reference (timing probes)")
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
                 "token_ids": [1] + [int(v) for v in torch.randint(4, 150, (n,), generator=g)]}
                for k, n in enumerate((20, 31))]
        args.out_dir = args.out_dir or Path(tmp.name) / args.variant
        args.real_vocab = 150
        tokens_sha = None
    else:
        if not args.tokens:
            raise SystemExit("--tokens required")
        segs = load_segments(args.tokens, args.max_tokens, args.long_max_tokens,
                             None if args.segment_raw is None else args.segment_raw.split(","),
                             None if args.segment is None else args.segment.split(","), args.bos_id)
        tokens_sha = {str(p): sha256_file(p) for p in args.tokens}
    if args.out_dir is None:
        raise SystemExit("--out-dir required")
    out_dir = args.out_dir
    truth = args.variant == "truth"
    work_dtype = torch.float32 if truth else torch.bfloat16
    attn = "chunked" if truth else "sdpa"   # chunked = exact fp32 eager attention over query chunks (RAM)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    ref = Lfm2Reference(args.model_dir, torch.device("cpu"), dtype=work_dtype, attn=attn)
    for s in segs:
        if max(s["token_ids"]) >= ref.vocab_size:
            raise SystemExit(f"segment {s['name']}: id outside vocab")
    n_layers = ref.n_layers if args.debug_max_layers is None else args.debug_max_layers
    ref.n_layers = n_layers
    print(f"[cpu-ref] variant={args.variant} work_dtype={work_dtype} attn={attn} layers={n_layers} threads="
          f"{torch.get_num_threads()} real_vocab={args.real_vocab} segments="
          f"{[(s['name'], len(s['token_ids'])) for s in segs]}", flush=True)

    names = [s["name"] for s in segs]
    ckpt = out_dir / "_ckpt" / "state.pt"
    start_layer, states, router = 0, {}, {}
    if ckpt.exists() and not args.no_resume:
        c = torch.load(ckpt, weights_only=True)
        if c["names"] == names and c["variant"] == args.variant and c.get("debug") == args.debug_max_layers:
            start_layer, states, router = c["next_layer"], c["states"], c["router"]
            print(f"[cpu-ref] resuming at layer {start_layer}", flush=True)
    pos, pe, masks = {}, {}, {}
    with torch.no_grad():
        for s in segs:
            n, ids = s["name"], s["token_ids"]
            T = len(ids)
            if n not in states:
                states[n] = ref.embed(ids).to(work_dtype).unsqueeze(0)          # no scale; bf16 table widened
            pos[n] = ref.positions(T)
            pe[n] = ref.rope(states[n], pos[n])
            masks[n] = None   # chunked / sdpa attention build their own causal mask
        for i in range(start_layer, n_layers):
            t0 = time.perf_counter()
            layer = ref.build_layer(i)
            t_load = time.perf_counter() - t0
            out = ref.layer_step(layer, i, [states[n] for n in names], [pe[n] for n in names],
                                 [masks[n] for n in names], [pos[n] for n in names])
            for n, o in zip(names, out):
                states[n] = o
            if i in ref.last_router_ids:
                for n, ids_ in zip(names, ref.last_router_ids[i]):
                    router[f"L{i}__{n}"] = ids_.to(torch.int16)
            del layer, out
            hmax = max(float(states[n].abs().max()) for n in states)
            kind = "attn" if ref.arch.is_attn(i) else "conv"
            ffn = "dense" if ref.arch.is_dense(i) else "moe"
            print(f"[cpu-ref] L{i:02d} {kind}+{ffn} load {t_load:.1f}s total {time.perf_counter() - t0:.1f}s "
                  f"max|h|={hmax:.1f} elapsed={(time.perf_counter() - t_start) / 60:.1f}min peakRSS={peak_rss_gb() or 0:.2f}GiB", flush=True)
            if not args.tiny:
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"names": names, "variant": args.variant, "debug": args.debug_max_layers,
                            "next_layer": i + 1, "states": states, "router": router}, str(ckpt) + ".tmp")
                os.replace(str(ckpt) + ".tmp", ckpt)
        np.savez(out_dir / "router_ids.npz", **{k: v.numpy() for k, v in router.items()})
        hid = {n: ref.final_norm(states[n][0]) for n in names}                  # truth: fp32; bf16sdpa: HF bf16 norm
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
            write_head(ref, hid, segs, head, d, {**meta_common, "variant": args.variant + (
                "" if head != "fp32" or truth else "_fp32head")}, args.real_vocab)
    for d in {out_dir, *([args.out_dir_fp32head] if args.out_dir_fp32head else [])}:
        (d / "run.json").write_text(json.dumps({
            "variant": args.variant, "segments": {s["name"]: len(s["token_ids"]) for s in segs},
            "total_seconds": time.perf_counter() - t_start, "n_layers": n_layers, "real_vocab": args.real_vocab,
            "note": "see <seg>.meta.json"}, indent=2), encoding="utf-8")

    if not truth and args.truth_dir is not None and not args.tiny and args.debug_max_layers is None:
        import kl_report as K

        chat = [Path(p) for p in args.tokens if "score_mask" in json.loads(Path(p).read_text(encoding="utf-8"))["segments"][0]]
        raws = [Path(p) for p in args.tokens if Path(p) not in chat]
        if not chat:
            print("[cpu-ref] no chat tokens file among --tokens: noise_term.json skipped")
        else:
            cands = tuple(float(x) for x in args.epsilon_candidates.split(",") if x.strip())
            res = K.lfm2_noise_term(args.truth_dir, out_dir, chat[0], None, args.max_tokens, None, 32, True, cands)
            res["raw_noise"] = []
            for rp in raws:
                rr = K.lfm2_noise_term(args.truth_dir, out_dir, chat[0], rp, args.max_tokens, None, 32, True, cands)
                res["raw_noise"] += [g for g in rr["groups"] if g["name"].startswith("raw")]
            if (args.truth_dir / "router_ids.npz").exists():
                res["router_id_agreement_truth_vs_bf16sdpa"] = router_agreement(args.truth_dir / "router_ids.npz",
                                                                                out_dir / "router_ids.npz")
            res["note"] = ("epsilon of the lfm2-fp32 gate is set from this file by the user/main session: "
                           "kl_report.py --gate lfm2-fp32 --epsilon <value>")
            (out_dir / "noise_term.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(K.lfm2_noise_markdown(res))
            print(f"[cpu-ref] wrote {out_dir / 'noise_term.json'}", flush=True)
    print(f"[cpu-ref] done in {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

