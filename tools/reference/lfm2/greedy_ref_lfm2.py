"""tools/reference/lfm2/greedy_ref_lfm2.py  (docs/lfm2-plan.md L1-14: the greedy-smoke reference output)

Cache-less, layer-streamed CPU greedy decoding of LFM2.5-24B-A2B on 4 fixed chat prompts (3 plain + 1 tool-call
prompt), all prompts in lockstep: every step re-runs the whole (prompt + generated) sequence of every still-running
prompt through the 40 layers, reading each layer's bf16 weights from disk once per STEP for ALL prompts (the FFN/MoE of
all prompts is one batched GEMM set).  Recorded per generated token: the top-2 logits and their gap, so the gate
(plan 6.4: r4dx's greedy output shares a prefix of >= 16 tokens with this one, OR diverges at a near tie) can be
evaluated without re-running the reference.

Why cache-less: HF generate would need the whole 48 GB model resident (or accelerate offload); a cache-less loop with
layer streaming keeps host RAM at one layer (<= 1.3 GB bf16, 2.5 GB fp32) + the hidden states.  Numerics:
`--variant bf16sdpa` (default, "HF greedy": bf16 weights/activations, sdpa attention, bf16 router, bf16 head -- the
thing r4dx's engine is compared with) or `--variant truth` (fp32 activations / router / head: what the model "really"
says).  A cache-less bf16 forward differs from a KV-cached bf16 forward only by GEMM-shape rounding (a near-tie token can
flip), which is exactly what the recorded gaps are for.  Logit columns >= real vocab (125017) are masked.  The model
always thinks: every prompt ends in `<|im_start|>assistant\\n<think>`, so 16-24 new tokens are mostly the start of the
reasoning; the tool-call prompt is a normal tool-schema prompt (Pythonic call expected after the thought).

Cost: dominated by reading 48 GB per step (page cache helps): 24 steps = 24 x 48 GB.  Estimated 10-40 minutes
(UNVERIFIED; the first steps print timing and an ETA).  Resumable: `<out>.partial.json` is rewritten after every step.
RAM peak ~ 2 GB + one layer.  CPU only (HIP_VISIBLE_DEVICES=-1 asserted).

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -I tools\\reference\\lfm2\\greedy_ref_lfm2.py `
        --max-new-tokens 24 --threads 12 --out tools\\reference\\golden_out\\lfm2\\greedy_smoke.json

Compare an r4dx run (`{"<prompt name>": [generated ids...], ...}`):

    python tools\\reference\\lfm2\\greedy_ref_lfm2.py --compare greedy_smoke.json r4dx_greedy.json
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

NEAR_TIE_GAP = 0.5   # logit units (bf16 spacing near |x| = 16..32 is 0.125); recorded, applied at --compare time
REAL_VOCAB_DEFAULT = 125017

PROMPTS = [
    {"name": "haiku_gpu", "messages": [
        {"role": "user", "content": "Write a haiku about GPUs, then explain what a GPU is in two sentences."}]},
    {"name": "capital_th", "messages": [
        {"role": "user", "content": "What is the capital of Thailand, and what is it known for? Answer in three sentences."}]},
    {"name": "python_fib", "messages": [
        {"role": "system", "content": "You are a concise programming assistant."},
        {"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a one-line docstring."}]},
    {"name": "tool_weather", "tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "unit": {"type": "string",
                       "enum": ["celsius", "fahrenheit"]}}, "required": ["city"]}}}],
     "messages": [{"role": "user", "content": "What is the weather in Bangkok right now? Use celsius."}]},
]


def compare(ref_path: Path, test_path: Path, min_prefix: int) -> int:
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    test = json.loads(test_path.read_text(encoding="utf-8"))
    near = ref.get("near_tie_gap", NEAR_TIE_GAP)
    ok = True
    for p in ref["prompts"]:
        t = test.get(p["name"])
        if t is None:
            print(f"{p['name']:<16} MISSING in {test_path}")
            ok = False
            continue
        g = p["gen_ids"]
        n = min(len(g), len(t))
        k = next((i for i in range(n) if g[i] != t[i]), n)
        if k == n and len(g) == len(t):
            print(f"{p['name']:<16} IDENTICAL ({n} tokens)  PASS")
            continue
        if k == n:      # one side stopped early with an identical prefix: only OK when the shorter one ended on EOS
            passed = n >= min_prefix
            print(f"{p['name']:<16} common prefix {k} (lengths {len(g)} vs {len(t)})  {'PASS' if passed else 'FAIL'}")
            ok &= passed
            continue
        gap = p["steps"][k]["gap"] if k < len(p["steps"]) else None
        near_tie = gap is not None and gap < near
        passed = k >= min_prefix or near_tie
        ok &= passed
        print(f"{p['name']:<16} common prefix {k:>3}  first divergence at step {k}: ref id {g[k]} vs test id {t[k]}, "
              f"ref top-2 gap {'n/a' if gap is None else format(gap, '.3f')}{' (near tie)' if near_tie else ''}  "
              f"{'PASS' if passed else 'FAIL'}")
    print("RESULT", "PASS" if ok else "FAIL", f"(min prefix {min_prefix} or divergence at gap < {near})")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, type=Path, default=None, metavar=("REF.json", "TEST.json"))
    ap.add_argument("--min-prefix", type=int, default=16)
    ap.add_argument("--variant", default="bf16sdpa", choices=("bf16sdpa", "truth"))
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--repetition-penalty", type=float, default=1.0,
                    help="HF semantics over prompt + generated ids (default off: the gate compares plain greedy)")
    ap.add_argument("--near-tie-gap", type=float, default=NEAR_TIE_GAP)
    ap.add_argument("--real-vocab", type=int, default=REAL_VOCAB_DEFAULT)
    ap.add_argument("--prompts", default=None, help="comma list of prompt names (default all 4)")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke on a tiny random checkpoint (stub prompts)")
    ap.add_argument("--tiny-seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.compare:
        return compare(args.compare[0], args.compare[1], args.min_prefix)

    import torch

    assert torch.cuda.is_available() is False, "torch.cuda.is_available() is True: refusing to run (CPU only)"
    from lfm2.common_lfm2 import DEFAULT_MODEL_DIR, GOLDEN_DIR, Lfm2RefTokenizer, sha256_file
    from lfm2.ref import Lfm2Reference, build_tiny_checkpoint

    torch.set_num_threads(args.threads)
    tmp = None
    if args.tiny:
        tmp = tempfile.TemporaryDirectory()
        args.model_dir = build_tiny_checkpoint(Path(tmp.name) / "model", args.tiny_seed)
        args.real_vocab = 150
        args.out = args.out or Path(tmp.name) / "greedy_smoke.json"
        g = torch.Generator().manual_seed(args.tiny_seed + 9)
        prompts = [{"name": f"tiny_{k}", "prompt_ids": [1] + [int(v) for v in torch.randint(4, 150, (n,), generator=g)]}
                   for k, n in enumerate((9, 14, 6))]
        eos_ids = (2,)
        tok = None
    else:
        args.model_dir = args.model_dir or DEFAULT_MODEL_DIR
        args.out = args.out or (GOLDEN_DIR / "greedy_smoke.json")
        tok = Lfm2RefTokenizer(args.model_dir)
        wanted = None if args.prompts is None else set(args.prompts.split(","))
        prompts = []
        for p in PROMPTS:
            if wanted is not None and p["name"] not in wanted:
                continue
            kw = {"tools": p["tools"]} if p.get("tools") else {}
            ids = tok.encode_chat(p["messages"], add_generation_prompt=True, **kw)
            assert ids[0] == tok.bos_id and ids.count(tok.bos_id) == 1
            prompts.append({**p, "prompt_ids": ids})
        gen_cfg = json.loads((args.model_dir / "generation_config.json").read_text(encoding="utf-8"))
        eos = gen_cfg.get("eos_token_id", 124900)
        eos_ids = tuple(eos) if isinstance(eos, list) else (int(eos),)

    truth = args.variant == "truth"
    dtype = torch.float32 if truth else torch.bfloat16
    ref = Lfm2Reference(args.model_dir, torch.device("cpu"), dtype=dtype, attn="chunked" if truth else "sdpa")
    rv = min(args.real_vocab, ref.vocab_size)
    partial = args.out.with_suffix(".partial.json")
    state = {p["name"]: {"gen_ids": [], "steps": [], "done": False} for p in prompts}
    if partial.exists() and not args.no_resume:
        saved = json.loads(partial.read_text(encoding="utf-8"))
        if saved.get("variant") == args.variant and saved.get("prompt_ids") == {p["name"]: p["prompt_ids"] for p in prompts}:
            state = saved["state"]
            print(f"[greedy] resuming at step {max(len(s['gen_ids']) for s in state.values())}", flush=True)

    def step_logits(active: list[dict]):
        """Last-position logits [n_active, V] (fp32; head per variant) of the sequences prompt + generated."""
        with torch.no_grad():
            hs, pes, poss, masks, last = [], [], [], [], []
            for p in active:
                ids = p["prompt_ids"] + state[p["name"]]["gen_ids"]
                h = ref.embed(ids).to(dtype).unsqueeze(0)
                pos = ref.positions(len(ids))
                hs.append(h), poss.append(pos), pes.append(ref.rope(h, pos)), masks.append(None)
            for i in range(ref.n_layers):
                layer = ref.build_layer(i)
                hs = ref.layer_step(layer, i, hs, pes, masks, poss)
                del layer
            rows = torch.stack([ref.final_norm(h[0, -1:])[0] for h in hs])
            return ref.logits_rows(rows, "fp32" if truth else "hf-bf16", real_vocab=rv)

    t_start = time.perf_counter()
    for step in range(args.max_new_tokens):
        active = [p for p in prompts if not state[p["name"]]["done"] and len(state[p["name"]]["gen_ids"]) <= step]
        if not any(not s["done"] for s in state.values()):
            break
        if not active:
            continue
        t0 = time.perf_counter()
        logits = step_logits(active)
        for p, row in zip(active, logits):
            s = state[p["name"]]
            seen = set(p["prompt_ids"]) | set(s["gen_ids"])
            if args.repetition_penalty != 1.0:
                idx = torch.tensor(sorted(seen))
                v = row[idx]
                row[idx] = torch.where(v > 0, v / args.repetition_penalty, v * args.repetition_penalty)
            i1 = int(row.argmax())                   # first index among equal maxima (as HF's argmax)
            l1 = float(row[i1])
            rest = row.clone()
            rest[i1] = float("-inf")
            i2 = int(rest.argmax())
            l2 = float(rest[i2])
            s["gen_ids"].append(int(i1))
            s["steps"].append({"i": step, "id": int(i1), "logit": l1, "id2": int(i2), "logit2": l2, "gap": l1 - l2,
                               "prob": float(torch.softmax(row[:rv], -1)[i1])})
            if i1 in eos_ids:
                s["done"] = True
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps({"variant": args.variant, "prompt_ids": {p["name"]: p["prompt_ids"] for p in prompts},
                                       "state": state}), encoding="utf-8")
        dt_step = time.perf_counter() - t0
        done = step + 1
        print(f"[greedy] step {done}/{args.max_new_tokens} ({len(active)} active) {dt_step:.0f}s, ETA "
              f"{(args.max_new_tokens - done) * dt_step / 60:.1f} min", flush=True)

    results = []
    for p in prompts:
        s = state[p["name"]]
        gaps = [x["gap"] for x in s["steps"]]
        r = {"name": p["name"], "messages": p.get("messages"), "tools": bool(p.get("tools")), "prompt_ids": p["prompt_ids"],
             "gen_ids": s["gen_ids"], "text": tok.decode(s["gen_ids"]) if tok else None,
             "finish_reason": "stop" if s["gen_ids"] and s["gen_ids"][-1] in eos_ids else "length", "steps": s["steps"],
             "min_gap": min(gaps) if gaps else None,
             "near_tie_steps": [x["i"] for x in s["steps"] if x["gap"] < args.near_tie_gap]}
        results.append(r)
        print(f"[greedy] {r['name']}: prompt {len(r['prompt_ids'])} tok, generated {len(r['gen_ids'])} "
              f"({r['finish_reason']}), min top-2 gap {r['min_gap']:.3f}, near-tie steps {r['near_tie_steps'][:8]}",
              flush=True)
    import transformers

    doc = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "tool": "tools/reference/lfm2/greedy_ref_lfm2.py",
           "model_dir": str(args.model_dir), "config_sha256": sha256_file(args.model_dir / "config.json"),
           "variant": args.variant, "tokenizer": tok.describe() if tok else "stub", "dtype": str(dtype),
           "attn": "sdpa" if not truth else "chunked-eager-fp32", "max_new_tokens": args.max_new_tokens,
           "eos_ids": list(eos_ids), "repetition_penalty": args.repetition_penalty, "real_vocab": rv,
           "near_tie_gap": args.near_tie_gap, "gap_units": "output logits of the variant's head (no softcap)",
           "decoding": "greedy, cache-less layer-streamed full recompute each step, EOS stop, logit columns >= "
                       "real_vocab masked", "torch_version": torch.__version__,
           "transformers_version": transformers.__version__, "tiny": bool(args.tiny),
           "total_seconds": time.perf_counter() - t_start, "prompts": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"[greedy] wrote {args.out}")
    if args.tiny:
        # the exactness check lives in test_scripts_tiny.test_greedy_matches_hf_nocache
        pass
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
