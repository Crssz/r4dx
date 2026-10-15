"""LFM2.5-24B-A2B (lfm2_moe) architecture table: names, shapes and per-layer types, derived from a HF
config.json (no torch, no transformers).  Facts: docs/lfm2-semantics.md; on-disk names (7610 tensors):
tools/reference/lfm2/tensor_names.json (checked by `check_against_header`).

On disk the experts are PER-EXPERT 2-D tensors `feed_forward.experts.{e}.{w1,w2,w3}.weight`
(w1 = gate, w3 = up, w2 = down); the HF module holds them fused (`experts.gate_up_proj [E, 2I, H]`
= cat(w1, w3) along dim 1, `experts.down_proj [E, H, I]`); `ref.py` does the stacking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CONV = "conv"
ATTN = "full_attention"

EMBED_NAME = "model.embed_tokens.weight"
FINAL_NORM_NAME = "model.embedding_norm.weight"   # the FINAL norm (misleading name)


@dataclass(frozen=True)
class Lfm2Arch:
    hidden: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    vocab: int
    eps: float
    layer_types: tuple[str, ...]
    conv_kernel: int
    conv_bias: bool
    dense_layers: int
    intermediate: int
    moe_intermediate: int
    n_experts: int
    top_k: int
    norm_topk_prob: bool
    use_expert_bias: bool
    routed_scaling: float
    rope_theta: float
    max_position: int
    tie_embeddings: bool
    bos_id: int
    pad_id: int
    eos_ids: tuple[int, ...]

    @classmethod
    def from_config(cls, cfg: dict, generation_config: dict | None = None) -> "Lfm2Arch":
        n = int(cfg["num_hidden_layers"])
        types = tuple(cfg["layer_types"])
        if len(types) != n or any(t not in (CONV, ATTN) for t in types):
            raise ValueError(f"bad layer_types ({len(types)} entries for {n} layers)")
        eos = (generation_config or {}).get("eos_token_id", cfg.get("eos_token_id"))
        eos = tuple(int(x) for x in eos) if isinstance(eos, (list, tuple)) else (() if eos is None else (int(eos),))
        rp = cfg.get("rope_parameters") or {}
        if rp.get("rope_type", "default") != "default":
            raise ValueError(f"rope_type {rp.get('rope_type')!r}: only 'default' is supported")
        heads = int(cfg["num_attention_heads"])
        hidden = int(cfg["hidden_size"])
        return cls(
            hidden=hidden, n_layers=n, n_heads=heads, n_kv_heads=int(cfg["num_key_value_heads"]),
            head_dim=int(cfg.get("head_dim") or hidden // heads), vocab=int(cfg["vocab_size"]),
            eps=float(cfg["norm_eps"]), layer_types=types, conv_kernel=int(cfg["conv_L_cache"]),
            conv_bias=bool(cfg.get("conv_bias", False)), dense_layers=int(cfg["num_dense_layers"]),
            intermediate=int(cfg["intermediate_size"]), moe_intermediate=int(cfg["moe_intermediate_size"]),
            n_experts=int(cfg["num_experts"]), top_k=int(cfg["num_experts_per_tok"]),
            norm_topk_prob=bool(cfg.get("norm_topk_prob", True)),
            use_expert_bias=bool(cfg.get("use_expert_bias", True)),
            routed_scaling=float(cfg.get("routed_scaling_factor", 1.0)),
            rope_theta=float(rp.get("rope_theta", cfg.get("rope_theta", 10000.0))),
            max_position=int(cfg.get("max_position_embeddings", 0)),
            tie_embeddings=bool(cfg.get("tie_word_embeddings", True)),
            bos_id=int(cfg.get("bos_token_id", 1)), pad_id=int(cfg.get("pad_token_id", 0)), eos_ids=eos)

    @classmethod
    def from_model_dir(cls, model_dir: Path | str) -> "Lfm2Arch":
        d = Path(model_dir)
        cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
        gen = None
        if (d / "generation_config.json").is_file():
            gen = json.loads((d / "generation_config.json").read_text(encoding="utf-8"))
        return cls.from_config(cfg, gen)

    # -- per-layer ------------------------------------------------------------------------------

    def is_attn(self, i: int) -> bool:
        return self.layer_types[i] == ATTN

    def is_dense(self, i: int) -> bool:
        return i < self.dense_layers

    def attn_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == ATTN]

    def layer_prefix(self, i: int) -> str:
        return f"model.layers.{i}."

    def layer_shapes(self, i: int, with_experts: bool = True) -> dict[str, tuple[int, ...]]:
        """name relative to layer_prefix -> shape of every on-disk tensor of layer i."""
        H, hd = self.hidden, self.head_dim
        s: dict[str, tuple[int, ...]] = {"operator_norm.weight": (H,), "ffn_norm.weight": (H,)}
        if self.is_attn(i):
            s.update({"self_attn.q_proj.weight": (self.n_heads * hd, H),
                      "self_attn.k_proj.weight": (self.n_kv_heads * hd, H),
                      "self_attn.v_proj.weight": (self.n_kv_heads * hd, H),
                      "self_attn.out_proj.weight": (H, self.n_heads * hd),
                      "self_attn.q_layernorm.weight": (hd,), "self_attn.k_layernorm.weight": (hd,)})
        else:
            s.update({"conv.in_proj.weight": (3 * H, H), "conv.conv.weight": (H, 1, self.conv_kernel),
                      "conv.out_proj.weight": (H, H)})
        if self.is_dense(i):
            I = self.intermediate
            s.update({"feed_forward.w1.weight": (I, H), "feed_forward.w3.weight": (I, H),
                      "feed_forward.w2.weight": (H, I)})
        else:
            I = self.moe_intermediate
            s["feed_forward.gate.weight"] = (self.n_experts, H)
            if self.use_expert_bias:
                s["feed_forward.expert_bias"] = (self.n_experts,)
            if with_experts:
                for e in range(self.n_experts):
                    s[f"feed_forward.experts.{e}.w1.weight"] = (I, H)
                    s[f"feed_forward.experts.{e}.w3.weight"] = (I, H)
                    s[f"feed_forward.experts.{e}.w2.weight"] = (H, I)
        return s

    def all_names(self) -> dict[str, tuple[int, ...]]:
        out = {EMBED_NAME: (self.vocab, self.hidden), FINAL_NORM_NAME: (self.hidden,)}
        for i in range(self.n_layers):
            p = self.layer_prefix(i)
            out.update({p + k: v for k, v in self.layer_shapes(i).items()})
        return out

    def summary(self) -> dict:
        return {"hidden": self.hidden, "layers": self.n_layers, "heads": self.n_heads, "kv_heads": self.n_kv_heads,
                "head_dim": self.head_dim, "vocab": self.vocab, "eps": self.eps,
                "layer_types": "".join("A" if t == ATTN else "c" for t in self.layer_types),
                "conv_kernel": self.conv_kernel, "dense_layers": self.dense_layers, "intermediate": self.intermediate,
                "moe_intermediate": self.moe_intermediate, "experts": self.n_experts, "top_k": self.top_k,
                "rope_theta": self.rope_theta, "tie_embeddings": self.tie_embeddings, "bos_id": self.bos_id,
                "eos_ids": list(self.eos_ids)}


def check_against_header(arch: Lfm2Arch, tensors: dict) -> list[str]:
    """Problems between `arch` and a header (`name -> {"shape": [...], "dtype": ...}`); [] = agree."""
    want = arch.all_names()
    problems = [f"missing {n}" for n in sorted(set(want) - set(tensors))]
    problems += [f"unexpected {n}" for n in sorted(set(tensors) - set(want))]
    for n, shape in want.items():
        e = tensors.get(n)
        if e is not None and tuple(e["shape"]) != tuple(shape):
            problems.append(f"{n}: shape {tuple(e['shape'])} != {tuple(shape)}")
        if e is not None and n.endswith("expert_bias") and e["dtype"] != "F32":
            problems.append(f"{n}: dtype {e['dtype']} (expected F32)")
    return problems
