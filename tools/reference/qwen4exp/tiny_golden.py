"""HF goldens of the tiny-q4exp (plan 4.11 item 2): per-layer activations for fixed prompts, on CPU in bf16.

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen4exp\\tiny_golden.py [--model E:\\models\\r4dx\\tiny-q4exp\\hf]
                                                                                    [--out E:\\models\\r4dx\\tiny-q4exp\\golden]

One file per prompt and QSA tie rule: `<out>/T{T}.{rule}.safetensors`, with T in PROMPT_LENGTHS (1, 7, 70, 300).
`hf`     = HF's own `Qwen4ExpTextQSAIndexer.forward` (the per-query Python loop, torch.topk tie order).
`recent` = `qsa_indexer.indexer_forward(tie_rule="recent")` monkeypatched in (plan Q23): ties at the top-k cut
           keep the most recent blocks. r4dx is compared against `recent`.
Prompts are seeded random ids < 4000; EOS (4095) sits inside the T=7 (position 3) and T=300 (positions 120 and
201) prompts. The run is `model(input_ids, use_cache=False)` with sdpa attention and the per-expert eager MoE loop,
bf16 weights and activations, batch 1. `<out>/meta.json` lists prompts, tensor shapes and the tie statistics.

Tensor names (all bf16 unless noted; T tokens; `{i}` in 0..3; only layers listed are present):
    input_ids                    i64 [T]
    embedding                    [T, 2560]          embed_tokens output
    layer{i}.residual_in         [T, 10240]         decoder layer input (4 streams x 2560); layer 0 = embedding repeated 4x
    layer1.ple_embedding         [T, 2560]          PLE n-gram rows gathered and flattened (16 heads x 160)
    layer1.ple_ngram_ids         i64 [T, 16]        global row ids (ngram_ids.NGramHash, independent of HF)
    layer1.ple_out               [T, 10240]         PLE output (added to the residual)
    layer1.residual_ple          [T, 10240]         layer 1 residual after adding the PLE output
    layer{i}.attn_mix            [T, 2560]          attn_hyper_connection mixed input (block input)
    layer{i}.attn_inject         [T, 4]             attn_hyper_connection injection weights (2 * sigmoid(.))
    layer{i}.attn_out            [T, 2560]          GDN / QSA block output before the injection
    layer{i}.residual_attn       [T, 10240]         residual after the attention block (= mlp_hyper_connection input)
    layer{i}.mlp_mix             [T, 2560]          mlp_hyper_connection mixed input (MoE input)
    layer{i}.mlp_inject          [T, 4]
    layer{i}.router_logits       [T, 16]
    layer{i}.topk_ids            i64 [T, 10]        HF order (descending router prob)
    layer{i}.topk_weights        [T, 10]            renormalised top-10 probabilities
    layer{i}.shared_out          [T, 2560]          shared expert MLP output (before its sigmoid gate)
    layer{i}.shared_gate_logit   [T, 1]             shared_expert_gate(x)
    layer{i}.moe_out             [T, 2560]          routed experts + gated shared expert
    layer{i}.residual_out        [T, 10240]         decoder layer output
  QSA layer (3) only:
    layer3.index_q               [T, 4, 128]        indexer queries after q_layernorm + rope
    layer3.index_raw_keys        [T, 128]           indexer token keys (index_qk_proj, before pooling)
    layer3.index_block_keys      [T // 4, 128]      pooled, k_layernorm'd, roped block keys (block j = tokens 4j..4j+3)
    layer3.index_mask            u8 [T, T]          1 = token selected by the indexer for that query (before causal AND)
    layer3.attn_q                [T, 24, 256]       after q_norm + rope
    layer3.attn_k                [T, 2, 256]        after k_norm + rope
    layer3.attn_v                [T, 2, 256]
    layer3.attn_gate             [T, 6144]          output gate logits (sigmoid applied after attention)
    layer3.attn_pre_o            [T, 6144]          attention output * sigmoid(gate) = o_proj input
    mixer_out                    [T, 2560]          hyper_connection_mixer output (lm_head input)
    logits                       [T, 4096]

`load_golden(dir, T, rule)` returns the dict of tensors; `golden_prompts()` the seeded id lists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen4exp import qsa_indexer, tiny  # noqa: E402
from qwen4exp.ngram_ids import NGramHash  # noqa: E402

PROMPT_LENGTHS = (1, 7, 70, 300)
PROMPT_SEED = 7
TIE_RULES = ("hf", "recent")
DEFAULT_MODEL = tiny.DEFAULT_OUT / "hf"
DEFAULT_OUT = tiny.DEFAULT_OUT / "golden"
# (T, positions of EOS) -- two prompts carry EOS ids
EOS_AT = {7: (3,), 300: (120, 201)}


def golden_prompts() -> dict[int, list[int]]:
    """{T: token ids}: seeded random ids < 4000, EOS (4095) at EOS_AT[T]."""
    gen = torch.Generator().manual_seed(PROMPT_SEED)
    out = {}
    for t in PROMPT_LENGTHS:
        ids = torch.randint(0, 4000, (t,), generator=gen)
        for pos in EOS_AT.get(t, ()):
            ids[pos] = tiny.TINY_EOS
        out[t] = ids.tolist()
    return out


def golden_path(golden_dir: str | Path, t: int, rule: str) -> Path:
    return Path(golden_dir) / f"T{t}.{rule}.safetensors"


def load_golden(golden_dir: str | Path, t: int, rule: str) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file

    return load_file(str(golden_path(golden_dir, t, rule)))


class Capture:
    """Forward hooks on the HF modules of one model, filling `self.t` with the tensor names above."""

    def __init__(self, model, ngram: NGramHash):
        self.model = model
        self.ngram = ngram
        self.t: dict[str, torch.Tensor] = {}
        self.handles = []
        text = model.model.language_model
        self.text = text

        # embedding
        self.handles.append(text.embed_tokens.register_forward_hook(lambda m, a, o: self._put("embedding", o)))
        self.handles.append(text.hyper_connection_mixer.register_forward_hook(lambda m, a, o: self._put("mixer_out", o)))
        for i, layer in enumerate(text.layers):
            p = f"layer{i}."
            self.handles.append(layer.register_forward_pre_hook(lambda m, a, k, p=p: self._put(p + "residual_in", a[0]), with_kwargs=True))
            self.handles.append(layer.register_forward_hook(lambda m, a, o, p=p: self._put(p + "residual_out", o)))
            ah, mh = layer.attn_hyper_connection, layer.mlp_hyper_connection
            self.handles.append(ah.register_forward_hook(lambda m, a, o, p=p: self._hc(p + "attn_", o)))
            self.handles.append(mh.register_forward_hook(lambda m, a, o, p=p: self._hc(p + "mlp_", o)))
            self.handles.append(mh.register_forward_pre_hook(lambda m, a, p=p: self._put(p + "residual_attn", a[0])))
            blk = layer.linear_attn if layer.layer_type == "linear_attention" else layer.self_attn
            first = (lambda m, a, o, p=p: self._put(p + "attn_out", o if torch.is_tensor(o) else o[0]))
            self.handles.append(blk.register_forward_hook(first))
            moe = layer.mlp
            self.handles.append(moe.register_forward_hook(lambda m, a, o, p=p: self._put(p + "moe_out", o)))
            self.handles.append(moe.gate.register_forward_hook(lambda m, a, o, p=p: self._router(p, o)))
            self.handles.append(moe.shared_expert.register_forward_hook(lambda m, a, o, p=p: self._put(p + "shared_out", o, False)))
            self.handles.append(moe.shared_expert_gate.register_forward_hook(lambda m, a, o, p=p: self._put(p + "shared_gate_logit", o, False)))
            if layer.ple is not None:
                self.handles.append(layer.ple.register_forward_hook(lambda m, a, o, p=p: self._put(p + "ple_out", o)))
                self.handles.append(layer.ple.ple_embedding.register_forward_hook(lambda m, a, o, p=p: self._put(p + "ple_embedding", o)))
            if layer.layer_type != "linear_attention":
                self.handles.append(layer.self_attn.indexer.register_forward_hook(lambda m, a, o, p=p: self._mask(p, o)))
                self.handles.append(layer.self_attn.register_forward_pre_hook(lambda m, a, k, p=p: self._qsa_inputs(p, m, a, k), with_kwargs=True))
                self.handles.append(layer.self_attn.o_proj.register_forward_pre_hook(lambda m, a, p=p: self._put(p + "attn_pre_o", a[0])))

    def _put(self, name, x, batched=True):
        """Store a copy; `batched` tensors carry a leading batch dim of 1 that is dropped."""
        x = x.detach()
        self.t[name] = (x[0] if batched else x).clone()

    def _hc(self, prefix, out):
        mixed, _, inject = out
        self._put(prefix + "mix", mixed)
        self._put(prefix + "inject", inject)

    def _router(self, prefix, out):
        logits, scores, ids = out
        self._put(prefix + "router_logits", logits, False)
        self._put(prefix + "topk_weights", scores, False)
        self._put(prefix + "topk_ids", ids, False)

    def _mask(self, prefix, mask):
        m = mask[0, 0]
        sel = (m if m.dtype == torch.bool else m == 0).to(torch.uint8)
        self.t[prefix + "index_mask"] = sel.clone()

    def _qsa_inputs(self, prefix, attn, args, kwargs):
        """Recompute the QSA layer's intermediates with the module's own weights and HF's own ops."""
        from transformers.models.qwen4_exp.modeling_qwen4_exp import apply_rotary_pos_emb

        hidden, pos = args[0], args[1]
        cos_full, sin_full = pos
        seq = hidden.shape[1]
        ix = attn.indexer
        q, raw = qsa_indexer.project(ix, hidden, pos)
        self._put(prefix + "index_q", q)
        self._put(prefix + "index_raw_keys", raw)
        ratio = ix.compress_ratio
        nb = seq // ratio
        if nb > 0:
            starts = torch.arange(nb) * ratio
            bk = qsa_indexer._pooled_block_keys(raw[0, : nb * ratio], starts, cos_full[0], sin_full[0],
                                                ix.k_layernorm.weight, ix.k_layernorm.eps, ratio)
        else:
            bk = raw.new_zeros((0, raw.shape[-1]))
        self.t[prefix + "index_block_keys"] = bk.clone()
        # attention q/k/v exactly as Qwen4ExpTextAttention.forward
        cos, sin = (x[:, -seq:, :] for x in pos)
        shape = (*hidden.shape[:-1], -1, attn.head_dim)
        qg = attn.q_proj(hidden).view(*hidden.shape[:-1], -1, attn.head_dim * 2)
        qs, gate = torch.chunk(qg, 2, dim=-1)
        self._put(prefix + "attn_gate", gate.reshape(*hidden.shape[:-1], -1))
        qs = attn.q_norm(qs.view(shape)).transpose(1, 2)
        ks = attn.k_norm(attn.k_proj(hidden).view(shape)).transpose(1, 2)
        vs = attn.v_proj(hidden).view(shape).transpose(1, 2)
        qs, ks = apply_rotary_pos_emb(qs, ks, cos, sin)
        self._put(prefix + "attn_q", qs.transpose(1, 2))
        self._put(prefix + "attn_k", ks.transpose(1, 2))
        self._put(prefix + "attn_v", vs.transpose(1, 2))

    def close(self):
        for h in self.handles:
            h.remove()


class RecentIndexer:
    """Context manager: patch Qwen4ExpTextQSAIndexer.forward with the vectorised indexer, tie_rule given."""

    def __init__(self, tie_rule: str):
        self.tie_rule = tie_rule

    def __enter__(self):
        from transformers.models.qwen4_exp import modeling_qwen4_exp as m

        self.cls = m.Qwen4ExpTextQSAIndexer
        self.orig = self.cls.forward
        rule = self.tie_rule

        def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
            return qsa_indexer.indexer_forward(self, hidden_states, position_embeddings, attention_mask,
                                               past_key_values, tie_rule=rule)

        self.cls.forward = forward
        return self

    def __exit__(self, *exc):
        self.cls.forward = self.orig


def run_prompt(model, ngram: NGramHash, ids: list[int], rule: str) -> dict[str, torch.Tensor]:
    """All golden tensors of one prompt. rule "hf" runs HF's own indexer, "recent" the patched one."""
    cap = Capture(model, ngram)
    inp = torch.tensor([ids], dtype=torch.long)
    try:
        with torch.no_grad():
            if rule == "hf":
                out = model(input_ids=inp, use_cache=False)
            else:
                with RecentIndexer("recent"):
                    out = model(input_ids=inp, use_cache=False)
    finally:
        cap.close()
    t = cap.t
    t["input_ids"] = inp[0].clone()
    t["logits"] = out.logits[0].detach().clone()
    ngram_ids, _ = ngram.ngram_ids(np.array(ids, dtype=np.int64))
    t["layer1.residual_ple"] = t["layer1.residual_in"] + t["layer1.ple_out"]  # the add HF does in the decoder layer
    t["layer1.ple_ngram_ids"] = torch.from_numpy(ngram_ids.copy())
    # the independent hash must address exactly the rows HF gathered
    table = model.model.language_model.layers[1].ple.ple_embedding.ngram_embedding.weight
    gathered = table[t["layer1.ple_ngram_ids"]].flatten(-2)
    if not torch.equal(gathered, t["layer1.ple_embedding"]):
        raise RuntimeError("ngram_ids.NGramHash rows differ from the rows HF gathered")
    return t


def main(argv=None) -> int:
    from safetensors.torch import save_file

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    model = tiny.load_hf_model(args.model)
    ngram = NGramHash.from_config_file(Path(args.model) / "config.json")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = golden_prompts()
    meta = {"model": args.model, "prompts": {str(t): ids for t, ids in prompts.items()}, "tensors": {}, "mask_diff_hf_vs_recent": {}}
    for t, ids in prompts.items():
        results = {}
        for rule in TIE_RULES:
            tensors = run_prompt(model, ngram, ids, rule)
            results[rule] = tensors
            for name, x in tensors.items():
                if x.is_floating_point() and not bool(torch.isfinite(x).all()):
                    raise RuntimeError(f"T{t}.{rule}: {name} has non-finite values")
            save_file({k: v.contiguous() for k, v in tensors.items()}, str(golden_path(out, t, rule)),
                      metadata={"format": "pt", "tie_rule": rule, "T": str(t)})
            meta["tensors"].setdefault(str(t), {name: [str(x.dtype).replace("torch.", ""), list(x.shape)] for name, x in tensors.items()})
            print(f"T{t} {rule}: {len(tensors)} tensors, logits std {tensors['logits'].float().std():.4f}")
        diff = (results["hf"]["layer3.index_mask"] != results["recent"]["layer3.index_mask"]).any(dim=-1)
        meta["mask_diff_hf_vs_recent"][str(t)] = {"queries_with_different_selection": int(diff.sum()), "queries": t}
        print(f"T{t}: hf vs recent selection differs on {int(diff.sum())} of {t} queries; "
              f"max |logit diff| {(results['hf']['logits'].float() - results['recent']['logits'].float()).abs().max():.4g}")
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    total = sum(p.stat().st_size for p in out.glob("*.safetensors"))
    print(f"wrote {out} ({total / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
