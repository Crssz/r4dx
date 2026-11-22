"""tools/reference/inkling2/greedy_smoke.py  (docs/inkling2-plan.md 7: the greedy-smoke reference output)

HF GREEDY decoding of Inkling-2-Mini on 4 fixed chat prompts (3 plain + 1 tool-call prompt), 64 new tokens each,
recording per generated token the top-2 logits and their gap (so the r4dx comparison "common prefix >= 16 tokens OR a
divergence at a near tie" can be evaluated without re-running the reference).

How it runs a 26 B model on a 64 GB CPU host: this is the REAL HF `Inkling2LoopForCausalLM.generate` (bf16, eager
attention, KV cache incl. the per-pass loop caches, stock experts implementation), but the expert weights are not held
dequantized (49 GB): the packed NVFP4/MXFP8 expert tensors of all 24 layers stay resident (~19 GB, as stored), and a
forward-pre-hook on every `experts` module dequantizes just the experts that layer call routes to (HF's own
`Inkling2LoopDequantizeExperts` + `Interleave`, sources cast to bf16 exactly as HF's loader does) into ONE shared
[128, ...] bf16 scratch that all 24 layers' `gate_up_proj` / `down_proj` point at.  HF's experts forward then runs
unmodified over the scratch rows it was routed to, so the numerics are stock HF.  Prefill dequantizes (almost) all 128
experts per visit (~1.5 s), a decode step at most 8 per visit.  Prompts run one at a time (batch 1, no padding).
`test_ref_tiny.py::test_lazy_generate_matches_hf` proves the scheme bit-equal to stock HF on a tiny checkpoint.

Cost (UNVERIFIED until the orchestrator's run; see the report): ~19 GB read + resident, then per prompt a prefill of ~1-2
minutes and per decode token ~48 visits x (<= 8 experts) of dequant: measured numbers are printed per token.
Resumable: `<out>.partial.json` is rewritten after every finished prompt.  CPU only (HIP_VISIBLE_DEVICES=-1 asserted).

    D:\\venvs\\r4dx-inkling-ref\\Scripts\\python.exe -I tools\\reference\\inkling2\\greedy_smoke.py --threads 16 `
        --max-new-tokens 64 --out tools\\reference\\golden_out\\inkling2\\greedy_smoke.json

Compare an r4dx run (`{"<prompt name>": [generated ids...], ...}`):

    python tools\\reference\\inkling2\\greedy_smoke.py --compare greedy_smoke.json r4dx_greedy.json
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
import types  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NEAR_TIE_GAP = 0.5   # logit units; recorded, applied at --compare time

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
        if k == n:
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


# --------------------------------------------------------------------------------------------
# The lazily-dequantizing HF model
# --------------------------------------------------------------------------------------------


class LazyExpertStore:
    """Packed expert tensors of every layer (as stored) + the shared bf16 scratch the HF experts modules point at."""

    PARTS = {"gate_up_proj": "w13", "down_proj": "w2"}

    def __init__(self, ref, log=print):
        import torch

        self.torch, self.ref, self.cfg = torch, ref, ref.config
        E, H, I = self.cfg.n_routed_experts, self.cfg.hidden_size, self.cfg.moe_intermediate_size
        self.scratch = {"gate_up_proj": torch.zeros(E, 2 * I, H, dtype=torch.bfloat16),
                        "down_proj": torch.zeros(E, H, I, dtype=torch.bfloat16)}
        self.raw: dict[int, dict] = {}
        t0 = time.perf_counter()
        from inkling2.ref import LAYER_PREFIX

        for L in range(self.cfg.num_hidden_layers):
            p = LAYER_PREFIX.format(L=L) + "mlp.experts."
            d = {}
            for part in ("w13", "w2"):
                for suf in ("_weight", "_weight_scale", "_weight_amax", "_weight_outer_scale_denominator"):
                    if ref.src.has(p + part + suf):
                        d[part + suf] = ref.raw(p + part + suf)
            self.raw[L] = d
            if L % 6 == 5:
                log(f"[greedy] packed experts read: layers 0..{L} ({time.perf_counter() - t0:.0f}s)", flush=True)
        self.stats = {"calls": 0, "experts_dequantized": 0, "seconds": 0.0}

    def _dequant(self, L: int, part: str, sl: slice):
        from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave

        torch = self.torch
        d = self.raw[L]
        keys = [f"{part}_weight", f"{part}_weight_scale", f"{part}_weight_amax", f"{part}_weight_outer_scale_denominator"]
        inp = {}
        for k in keys:
            if k in d:
                v = d[k]
                # per-expert tensors are sliced along dim 0; the outer denominator is one scalar for the layer
                inp[k] = (v if k.endswith("denominator") else v[sl]).to(torch.bfloat16)   # HF casts every source to bf16
        stub = types.SimpleNamespace(get_parameter=lambda name: types.SimpleNamespace(dtype=torch.bfloat16))
        out = Inkling2LoopDequantizeExperts().convert(inp, source_patterns=keys, target_patterns=["x"], model=stub,
                                                      full_layer_name="x")["x"]
        if part == "w13":
            out = Interleave(dim=1).convert({"x": out}, source_patterns=["x"], target_patterns=["y"])["y"]
        return out

    def fill(self, L: int, hit) -> None:
        """Dequantize the experts `hit` (iterable of ids) of layer L into the shared scratch."""
        torch = self.torch
        t0 = time.perf_counter()
        hit = sorted({int(e) for e in hit})
        if len(hit) > self.cfg.n_routed_experts // 4:                 # prefill: bulk, chunks of 8 inside the op
            for which, part in self.PARTS.items():
                self.scratch[which].copy_(self._dequant(L, part, slice(0, self.cfg.n_routed_experts)))
            n = self.cfg.n_routed_experts
        else:
            for e in hit:
                for which, part in self.PARTS.items():
                    self.scratch[which][e].copy_(self._dequant(L, part, slice(e, e + 1))[0])
            n = len(hit)
        self.stats["calls"] += 1
        self.stats["experts_dequantized"] += n
        self.stats["seconds"] += time.perf_counter() - t0


def build_lazy_model(model_dir, config=None, source=None, log=print):
    """`Inkling2LoopForCausalLM` in bf16 (stock HF numerics, eager attention, stock experts implementation) whose experts
    are filled on demand from the packed tensors. Returns (model, store)."""
    import torch

    from inkling2.ref import Inkling2Reference, layer_kind  # noqa: F401

    ref = Inkling2Reference(model_dir, dtype=torch.bfloat16, config=config, source=source)
    cfg = ref.config
    cfg._experts_implementation = ref.stock_experts_impl            # what from_pretrained would pick
    store = LazyExpertStore(ref, log=log)
    m = ref.m
    with torch.device("meta"):
        model = m.Inkling2LoopForCausalLM(cfg)
    sd = {}
    for L in range(cfg.num_hidden_layers):
        full = ref.layer_state(L, with_experts=False)
        for k, v in full.items():
            sd[f"model.layers.{L}.{k}"] = v
        sd[f"model.layers.{L}.mlp.experts.gate_up_proj"] = store.scratch["gate_up_proj"]
        sd[f"model.layers.{L}.mlp.experts.down_proj"] = store.scratch["down_proj"]
    sd["model.embed_tokens.weight"] = ref.embed_table
    sd["model.embed_tokens.embed_norm.weight"] = ref.embed_norm.weight
    sd["model.norm.weight"] = ref.final_norm.weight
    sd["model.final_input.query"] = ref.final_input.query
    sd["model.message_input.query"] = ref.message_input.query
    sd["lm_head.weight"] = ref.lm_head_w
    model.load_state_dict(sd, strict=True, assign=True)
    if any(t.is_meta for t in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError("parameters left on the meta device")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for L, layer in enumerate(model.model.layers):
        def pre(mod, args, L=L):
            store.fill(L, args[1].unique().tolist())
        layer.mlp.experts.register_forward_pre_hook(pre)
    return model, store


def greedy(model, ids: list[int], max_new: int, eos_ids: tuple, pad_id: int):
    """HF generate, greedy; returns (gen_ids, steps) with the top-2 logits of every step."""
    import torch
    from transformers import GenerationConfig

    gc = GenerationConfig(do_sample=False, max_new_tokens=max_new, eos_token_id=list(eos_ids), pad_token_id=pad_id,
                          temperature=None, top_k=None, top_p=None, use_cache=True, output_logits=True,
                          return_dict_in_generate=True)
    with torch.no_grad():
        out = model.generate(input_ids=torch.tensor([ids]), generation_config=gc)
    gen = [int(t) for t in out.sequences[0, len(ids):]]
    steps = []
    for i, lg in enumerate(out.logits):
        row = lg[0].float()
        v, ix = torch.topk(row, 2)
        steps.append({"i": i, "id": int(ix[0]), "logit": float(v[0]), "id2": int(ix[1]), "logit2": float(v[1]),
                      "gap": float(v[0] - v[1]), "prob": float(torch.softmax(row, -1)[ix[0]])})
        assert steps[-1]["id"] == gen[i], "greedy token != argmax of the recorded logits"
    return gen, steps


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs=2, type=Path, default=None, metavar=("REF.json", "TEST.json"))
    ap.add_argument("--min-prefix", type=int, default=16)
    ap.add_argument("--model-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--near-tie-gap", type=float, default=NEAR_TIE_GAP)
    ap.add_argument("--prompts", default=None, help="comma list of prompt names (default all 4)")
    ap.add_argument("--threads", type=int, default=16, help="torch threads (max 24)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke on a tiny random checkpoint (stub prompts)")
    ap.add_argument("--tiny-seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.compare:
        return compare(args.compare[0], args.compare[1], args.min_prefix)

    import torch

    from inkling2.common_inkling2 import (
        DEFAULT_MODEL_DIR,
        EOS_ID,
        GOLDEN_DIR,
        PAD_ID,
        Inkling2RefTokenizer,
        assert_cpu_only,
        peak_rss_gb,
        set_threads,
    )
    from inkling2.ref import build_tiny_checkpoint, tiny_config

    assert_cpu_only()
    set_threads(args.threads)
    tmp, config = None, None
    if args.tiny:
        tmp = tempfile.TemporaryDirectory()
        args.model_dir = build_tiny_checkpoint(Path(tmp.name) / "model", args.tiny_seed)
        config = tiny_config()
        args.out = args.out or Path(tmp.name) / "greedy_smoke.json"
        g = torch.Generator().manual_seed(args.tiny_seed + 9)
        prompts = [{"name": f"tiny_{k}", "prompt_ids": [int(v) for v in torch.randint(4, 150, (n,), generator=g)]}
                   for k, n in enumerate((9, 14, 6))]
        eos_ids, pad_id, tok = (2,), 0, None
    else:
        args.model_dir = args.model_dir or DEFAULT_MODEL_DIR
        args.out = args.out or (GOLDEN_DIR / "greedy_smoke.json")
        tok = Inkling2RefTokenizer(args.model_dir)
        wanted = None if args.prompts is None else set(args.prompts.split(","))
        prompts = []
        for p in PROMPTS:
            if wanted is not None and p["name"] not in wanted:
                continue
            kw = {"tools": p["tools"]} if p.get("tools") else {}
            ids = tok.encode_chat(p["messages"], add_generation_prompt=True, **kw)   # no generation prompt exists
            prompts.append({**p, "prompt_ids": ids})
        eos_ids, pad_id = (EOS_ID,), PAD_ID
    partial = args.out.with_suffix(".partial.json")
    state = {}
    if partial.exists() and not args.no_resume:
        saved = json.loads(partial.read_text(encoding="utf-8"))
        if saved.get("prompt_ids") == {p["name"]: p["prompt_ids"] for p in prompts} and \
                saved.get("max_new_tokens") == args.max_new_tokens:
            state = saved["state"]
            print(f"[greedy] resuming: {sorted(state)} done", flush=True)
    t_start = time.perf_counter()
    model, store = build_lazy_model(args.model_dir, config=config)
    print(f"[greedy] model ready in {time.perf_counter() - t_start:.0f}s, peakRSS={peak_rss_gb() or 0:.1f}GiB", flush=True)
    for p in prompts:
        if p["name"] in state:
            continue
        t0 = time.perf_counter()
        s0 = dict(store.stats)
        gen, steps = greedy(model, p["prompt_ids"], args.max_new_tokens, eos_ids, pad_id)
        state[p["name"]] = {"gen_ids": gen, "steps": steps, "seconds": time.perf_counter() - t0,
                            "experts_dequantized": store.stats["experts_dequantized"] - s0["experts_dequantized"]}
        partial.parent.mkdir(parents=True, exist_ok=True)
        partial.write_text(json.dumps({"prompt_ids": {q["name"]: q["prompt_ids"] for q in prompts},
                                       "max_new_tokens": args.max_new_tokens, "state": state}), encoding="utf-8")
        print(f"[greedy] {p['name']}: prompt {len(p['prompt_ids'])} tok -> {len(gen)} new in "
              f"{time.perf_counter() - t0:.0f}s ({(time.perf_counter() - t0) / max(len(gen), 1):.1f}s/token incl. prefill), "
              f"peakRSS={peak_rss_gb() or 0:.1f}GiB", flush=True)
    results = []
    for p in prompts:
        s = state[p["name"]]
        gaps = [x["gap"] for x in s["steps"]]
        r = {"name": p["name"], "messages": p.get("messages"), "tools": bool(p.get("tools")), "prompt_ids": p["prompt_ids"],
             "gen_ids": s["gen_ids"], "text": tok.decode(s["gen_ids"]) if tok else None,
             "finish_reason": "stop" if s["gen_ids"] and s["gen_ids"][-1] in eos_ids else "length", "steps": s["steps"],
             "seconds": s["seconds"], "min_gap": min(gaps) if gaps else None,
             "near_tie_steps": [x["i"] for x in s["steps"] if x["gap"] < args.near_tie_gap]}
        results.append(r)
        print(f"[greedy] {r['name']}: generated {len(r['gen_ids'])} ({r['finish_reason']}), min top-2 gap "
              f"{r['min_gap']:.3f}, near-tie steps {r['near_tie_steps'][:8]}", flush=True)
    import transformers

    doc = {"generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "tool": "tools/reference/inkling2/greedy_smoke.py",
           "model_dir": str(args.model_dir), "variant": "hf-bf16", "tokenizer": tok.describe() if tok else "stub",
           "dtype": "bfloat16", "attn": "eager", "experts_impl": model.config._experts_implementation,
           "max_new_tokens": args.max_new_tokens, "eos_ids": list(eos_ids), "near_tie_gap": args.near_tie_gap,
           "gap_units": "output logits after the final softcap (30)",
           "decoding": "HF generate, greedy, KV cache, batch 1, no repetition penalty, stop on "
                       "<|content_model_end_sampling|>; no generation prompt: the model emits its own "
                       "<|message_model|> header", "torch_version": torch.__version__,
           "transformers_version": transformers.__version__, "tiny": bool(args.tiny),
           "dequant_stats": store.stats, "total_seconds": time.perf_counter() - t_start, "prompts": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(f"[greedy] wrote {args.out}")
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
