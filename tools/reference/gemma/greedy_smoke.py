"""tools/reference/gemma/greedy_smoke.py  (M0-10: the greedy-smoke reference output)

HF greedy decoding (64 new tokens) of the bf16 Huihui-Gemma-4-12B on 3 fixed chat prompts plus one
`<|think|>` prompt, recording for every generated token the top-2 logits and their gap. The M1-30 gate
(docs/gemma4-plan.md): r4dx's greedy output must share a prefix of >= 16 tokens with this one, OR diverge
at a near tie, i.e. a step where this file's top-2 gap is small -- which is what the recorded gaps are for.

Output `greedy_smoke.json` (default tools/reference/golden_out/gemma/):

    prompts[]: name, enable_thinking, messages, prompt_ids (BOS first, from the chat template), gen_ids,
               text (decoded, special tokens kept), finish_reason, steps[{i, id, logit, id2, logit2, gap, prob}],
               min_gap, near_tie_steps (gap < near_tie_gap)
    near_tie_gap, max_new_tokens, model / config / tokenizer sha256, versions, attn, device

`gap` is in softcapped-logit units (the model's own output logits, bf16). Compare an r4dx run with

    python tools\\reference\\gemma\\greedy_smoke.py --compare greedy_smoke.json r4dx_greedy.json

where r4dx_greedy.json is `{"<prompt name>": [generated ids...], ...}` (greedy, same prompt ids); it prints,
per prompt, the common-prefix length, the first divergence, the reference gap there and PASS / FAIL against
`--min-prefix 16` (a divergence at a gap < near_tie_gap passes too).

GPU (hand this to the user; never run by an agent):

    $env:HIP_VISIBLE_DEVICES='1'; D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe `
        tools\\reference\\gemma\\greedy_smoke.py --out tools\\reference\\golden_out\\gemma\\greedy_smoke.json

`--tiny` runs it on a tiny random checkpoint on CPU (stub tokenizer). `--compare` needs no model.
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import os
import sys
import tempfile
from pathlib import Path

if "--tiny" in sys.argv or "--compare" in sys.argv:
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemma.common_gemma import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    EOS_IDS,
    GOLDEN_DIR,
    load_generation_config,
    load_tokenizer,
    resolve_device,
    sha256_file,
)
from gemma.ref import add_tiny_args, build_tiny_checkpoint  # noqa: E402

PROMPTS = [
    {"name": "haiku_gpu", "enable_thinking": False, "messages": [
        {"role": "user", "content": "Write a haiku about GPUs, then explain what a GPU is in two sentences."}]},
    {"name": "capital_th", "enable_thinking": False, "messages": [
        {"role": "user", "content": "What is the capital of Thailand, and what is it known for? Answer in three sentences."}]},
    {"name": "python_fib", "enable_thinking": False, "messages": [
        {"role": "system", "content": "You are a concise programming assistant."},
        {"role": "user", "content": "Write a Python function that returns the n-th Fibonacci number iteratively, with a one-line docstring."}]},
    {"name": "think_bat_ball", "enable_thinking": True, "messages": [
        {"role": "user", "content": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"}]},
]
NEAR_TIE_GAP = 0.5  # softcapped-logit units; bf16 spacing near |x|=30 is 0.125


def run_prompt(model, tok, p: dict, device, max_new: int, eos_ids, near_tie: float, pad_id: int, base_cfg) -> dict:
    import torch

    ids = tok.encode_chat(p["messages"], enable_thinking=p["enable_thinking"], add_generation_prompt=True)
    cfg = copy.deepcopy(base_cfg)
    cfg.update(do_sample=False, temperature=1.0, top_k=None, top_p=1.0, max_new_tokens=max_new,
               eos_token_id=list(eos_ids), pad_token_id=pad_id, use_cache=True,
               return_dict_in_generate=True, output_logits=True)
    inp = torch.tensor([ids], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(input_ids=inp, attention_mask=torch.ones_like(inp), generation_config=cfg)
    gen = [int(t) for t in out.sequences[0, len(ids):].tolist()]
    steps = []
    for i, lg in enumerate(out.logits):
        row = lg[0].float()
        top = torch.topk(row, 2)
        (l1, l2), (i1, i2) = top.values.tolist(), top.indices.tolist()
        note = None
        if i < len(gen) and gen[i] != i1:
            g = gen[i]
            if float(row[g]) == l1:       # exact tie in bf16 logits: the sampler's argmax picked another index
                (i1, i2), note = (g, i1), "tie"
            else:                          # a logits processor (suppress_tokens) changed the argmax
                (i1, l1, i2, l2), note = (g, float(row[g]), i1, l1), "processed"
        step = {"i": i, "id": i1, "logit": l1, "id2": i2, "logit2": l2, "gap": l1 - l2,
                "prob": float(torch.softmax(row, -1)[i1])}
        if note:
            step["note"] = note
        steps.append(step)
    gaps = [s["gap"] for s in steps]
    return {"name": p["name"], "enable_thinking": p["enable_thinking"], "messages": p["messages"],
            "prompt_ids": ids, "gen_ids": gen, "text": tok.decode(gen),
            "finish_reason": "stop" if gen and gen[-1] in eos_ids else "length", "steps": steps,
            "min_gap": min(gaps) if gaps else None, "near_tie_steps": [s["i"] for s in steps if s["gap"] < near_tie]}


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
        gap = p["steps"][k]["gap"] if k < len(p["steps"]) else None
        near_tie = gap is not None and gap < near
        passed = k >= min_prefix or near_tie
        ok &= passed
        print(f"{p['name']:<16} common prefix {k:>3}  first divergence at step {k}: ref id "
              f"{g[k] if k < len(g) else None} vs test id {t[k] if k < len(t) else None}, ref top-2 gap "
              f"{'n/a' if gap is None else format(gap, '.3f')}"
              f"{' (near tie)' if near_tie else ''}  {'PASS' if passed else 'FAIL'}")
    print("RESULT", "PASS" if ok else "FAIL", f"(min prefix {min_prefix} or divergence at gap < {near})")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokenizer-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=GOLDEN_DIR / "greedy_smoke.json")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--attn", default="eager", choices=["eager", "sdpa"])
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--near-tie-gap", type=float, default=NEAR_TIE_GAP)
    ap.add_argument("--compare", nargs=2, type=Path, default=None, metavar=("REF.json", "TEST.json"))
    ap.add_argument("--min-prefix", type=int, default=16)
    add_tiny_args(ap)
    args = ap.parse_args(argv)
    if args.compare:
        return compare(args.compare[0], args.compare[1], args.min_prefix)

    import torch
    from transformers import GenerationConfig
    from transformers.models.gemma4_unified.modeling_gemma4_unified import Gemma4UnifiedForConditionalGeneration

    tmp = None
    if args.tiny:
        tmp = tempfile.TemporaryDirectory()
        args.model_dir = build_tiny_checkpoint(Path(tmp.name) / "model", args.tiny_seed)
        args.device = "cpu"
        if args.out == ap.get_default("out"):
            args.out = Path(tmp.name) / "greedy_smoke.json"
    device = resolve_device(args.device)
    tok = load_tokenizer(args.tokenizer_dir, tiny_vocab=128 if args.tiny else None)
    gen_cfg = load_generation_config(args.model_dir)
    eos_ids = tuple(gen_cfg["eos_token_id"]) if isinstance(gen_cfg.get("eos_token_id"), list) else EOS_IDS
    kw = {"dtype": torch.bfloat16, "attn_implementation": args.attn}
    if device.type == "cuda":
        kw["device_map"] = {"": 0}
    model = Gemma4UnifiedForConditionalGeneration.from_pretrained(str(args.model_dir), **kw)
    if device.type != "cuda":
        model.to(device)
    model.eval()
    base_cfg = GenerationConfig.from_pretrained(str(args.model_dir))
    results = []
    for p in PROMPTS:
        r = run_prompt(model, tok, p, device, args.max_new_tokens, eos_ids, args.near_tie_gap, 0, base_cfg)
        results.append(r)
        print(f"[greedy_smoke] {r['name']}: prompt {len(r['prompt_ids'])} tok, generated {len(r['gen_ids'])} "
              f"({r['finish_reason']}), min top-2 gap {r['min_gap']:.3f}, near-tie steps {r['near_tie_steps'][:8]}",
              flush=True)
    import transformers

    doc = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "tool": "tools/reference/gemma/greedy_smoke.py",
        "model_dir": str(args.model_dir), "config_sha256": sha256_file(args.model_dir / "config.json"),
        "tokenizer": tok.describe(), "tokenizer_provenance": tok.provenance(), "attn": args.attn,
        "device": str(device), "dtype": "bfloat16", "max_new_tokens": args.max_new_tokens, "eos_ids": list(eos_ids),
        "near_tie_gap": args.near_tie_gap, "gap_units": "softcapped logits of the bf16 model (HF output logits)",
        "decoding": "greedy (do_sample=False), KV cache, EOS stop", "torch_version": torch.__version__,
        "transformers_version": transformers.__version__, "tiny": bool(args.tiny), "prompts": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"[greedy_smoke] wrote {args.out}")
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
