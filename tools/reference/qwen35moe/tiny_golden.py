"""HF goldens of the tiny-q35moe (docs/qwen35moe-plan.md M2): per-layer activations for fixed prompts, CPU, bf16.

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\tiny_golden.py [--model E:\\models\\r4dx\\tiny-q35moe\\hf]
                                                                                       [--out E:\\models\\r4dx\\tiny-q35moe\\golden]

One file per prompt: `<out>/T{T}.safetensors`, T in PROMPT_LENGTHS (1, 7, 70, 300), plus `<out>/free_T{T}.json` with a
free-running greedy continuation (full forward per step, no cache) of 24 tokens and `<out>/meta.json`.
Prompts are seeded random ids < 4000; EOS (4095) sits inside the T=7 (position 3) and T=300 (positions 120, 201) prompts.
The run is `model(input_ids, use_cache=False)` with sdpa attention and the eager per-expert MoE loop, bf16, batch 1.

Tensor names (bf16 unless noted; T tokens; `{i}` in 0..3):
    input_ids                    i64 [T]
    embedding                    [T, 3072]
    layer{i}.residual_in         [T, 3072]
    layer{i}.attn_in             [T, 3072]    input_layernorm output (mixer input)
    layer{i}.attn_out            [T, 3072]    GDN / attention output before the residual add
    layer{i}.residual_attn       [T, 3072]
    layer{i}.mlp_in              [T, 3072]    post_attention_layernorm output (MoE input)
    layer{i}.router_logits       [T, 16]
    layer{i}.topk_ids            i64 [T, 8]   HF order (descending router probability)
    layer{i}.topk_weights        [T, 8]       renormalised top-8 probabilities
    layer{i}.shared_out          [T, 3072]    shared expert MLP output (before its sigmoid gate)
    layer{i}.shared_gate_logit   [T, 1]
    layer{i}.moe_out             [T, 3072]    routed experts + gated shared expert
    layer{i}.residual_out        [T, 3072]
  full-attention layer (3) only:
    layer3.attn_pre_o            [T, 4096]    attention output * sigmoid(gate) = o_proj input
    norm_out                     [T, 3072]    final RMSNorm (lm_head input)
    logits                       [T, 4096]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen35moe import tiny  # noqa: E402

PROMPT_LENGTHS = (1, 7, 70, 300)
PROMPT_SEED = 7
EOS_AT = {7: (3,), 300: (120, 201)}
FREE_STEPS = 24
DEFAULT_MODEL = tiny.DEFAULT_OUT / "hf"
DEFAULT_OUT = tiny.DEFAULT_OUT / "golden"


def golden_prompts() -> dict[int, list[int]]:
    gen = torch.Generator().manual_seed(PROMPT_SEED)
    out = {}
    for t in PROMPT_LENGTHS:
        ids = torch.randint(0, 4000, (t,), generator=gen)
        for pos in EOS_AT.get(t, ()):
            ids[pos] = tiny.TINY_EOS
        out[t] = ids.tolist()
    return out


def load_golden(golden_dir: str | Path, t: int) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(Path(golden_dir) / f"T{t}.safetensors"))


class Capture:
    """Forward hooks on the HF modules of one model, filling `self.t` with the tensor names above."""

    def __init__(self, model):
        self.t: dict[str, torch.Tensor] = {}
        self.handles = []
        text = model.model.language_model
        self.handles.append(text.embed_tokens.register_forward_hook(lambda m, a, o: self._put("embedding", o)))
        self.handles.append(text.norm.register_forward_hook(lambda m, a, o: self._put("norm_out", o)))
        for i, layer in enumerate(text.layers):
            p = f"layer{i}."
            h = self.handles.append
            h(layer.register_forward_pre_hook(lambda m, a, k, p=p: self._put(p + "residual_in", a[0] if a else k["hidden_states"]), with_kwargs=True))
            h(layer.register_forward_hook(lambda m, a, o, p=p: self._put(p + "residual_out", o if torch.is_tensor(o) else o[0])))
            h(layer.input_layernorm.register_forward_hook(lambda m, a, o, p=p: self._put(p + "attn_in", o)))
            mixer = layer.linear_attn if layer.block_type == "linear_attention" else layer.self_attn
            h(mixer.register_forward_hook(lambda m, a, o, p=p: self._put(p + "attn_out", o if torch.is_tensor(o) else o[0])))
            h(layer.post_attention_layernorm.register_forward_pre_hook(lambda m, a, p=p: self._put(p + "residual_attn", a[0])))
            h(layer.post_attention_layernorm.register_forward_hook(lambda m, a, o, p=p: self._put(p + "mlp_in", o)))
            moe = layer.mlp
            h(moe.register_forward_hook(lambda m, a, o, p=p: self._put(p + "moe_out", o if torch.is_tensor(o) else o[0])))
            h(moe.gate.register_forward_hook(lambda m, a, o, p=p: self._router(p, o)))
            h(moe.shared_expert.register_forward_hook(lambda m, a, o, p=p: self._put(p + "shared_out", o, False)))
            h(moe.shared_expert_gate.register_forward_hook(lambda m, a, o, p=p: self._put(p + "shared_gate_logit", o, False)))
            if layer.block_type != "linear_attention":
                h(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, p=p: self._put(p + "attn_pre_o", a[0])))

    def _put(self, name, x, batched=True):
        x = x.detach()
        self.t[name] = (x[0] if batched else x).clone()

    def _router(self, p, out):
        logits, scores, indices = out
        self.t[p + "router_logits"] = logits.detach().clone()
        self.t[p + "topk_weights"] = scores.detach().clone()
        self.t[p + "topk_ids"] = indices.detach().clone()

    def close(self):
        for hd in self.handles:
            hd.remove()


@torch.no_grad()
def run(model, ids: list[int]) -> dict[str, torch.Tensor]:
    cap = Capture(model)
    try:
        x = torch.tensor([ids], dtype=torch.long)
        out = model(input_ids=x, use_cache=False)
        cap.t["input_ids"] = x[0].clone()
        cap.t["logits"] = out.logits[0].detach().clone()
    finally:
        cap.close()
    return cap.t


@torch.no_grad()
def free_run(model, ids: list[int], steps: int) -> list[int]:
    cur = list(ids)
    new = []
    for _ in range(steps):
        out = model(input_ids=torch.tensor([cur], dtype=torch.long), use_cache=False)
        nxt = int(out.logits[0, -1].float().argmax())
        new.append(nxt)
        cur.append(nxt)
    return new


def main(argv=None) -> int:
    from safetensors.torch import save_file

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)
    torch.manual_seed(0)
    model = tiny.load_hf_model(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"prompts": {}, "free_steps": FREE_STEPS}
    for t, ids in golden_prompts().items():
        tensors = run(model, ids)
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(out / f"T{t}.safetensors"))
        free = free_run(model, ids, FREE_STEPS)
        (out / f"free_T{t}.json").write_text(json.dumps({"prompt": ids, "continuation": free}))
        meta["prompts"][str(t)] = {
            "eos_at": list(EOS_AT.get(t, ())),
            "shapes": {k: list(v.shape) for k, v in tensors.items() if k.startswith(("layer0.", "logits"))},
            "experts_hit_layer0": int(torch.unique(tensors["layer0.topk_ids"]).numel()),
        }
        print(f"[golden] T={t}: {len(tensors)} tensors, free-run {free[:6]}...")
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
