"""Gemma 4 architecture table: everything the reference, quantization and validation scripts need to
know about a gemma4_unified text stack, derived from a HF config.json (no torch, no transformers).

Facts are the VERIFIED ones of docs/gemma4-semantics.md. Checkpoint tensor names are the on-disk
names (tools/reference/gemma/tensor_names.json): `model.language_model.*`; the HF module path of the
same tensor is the same string minus the leading `model.` plus ... see `hf_param_name`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SLIDING = "sliding_attention"
FULL = "full_attention"

LM_PREFIX = "model.language_model."
EMBED_NAME = LM_PREFIX + "embed_tokens.weight"
FINAL_NORM_NAME = LM_PREFIX + "norm.weight"


@dataclass(frozen=True)
class GemmaArch:
    hidden: int
    n_layers: int
    n_heads: int
    vocab: int
    eps: float
    softcap: float | None
    window: int
    layer_types: tuple[str, ...]
    head_dim: int            # sliding layers
    global_head_dim: int     # full layers
    kv_heads: int            # sliding layers
    global_kv_heads: int     # full layers
    k_eq_v: bool             # full layers have no v_proj: V = raw k_proj output
    intermediate: int
    tie_embeddings: bool
    rope: dict[str, dict[str, Any]]
    max_position: int
    bos_id: int
    pad_id: int
    eos_ids: tuple[int, ...]
    activation: str

    # -- construction -------------------------------------------------------------------------

    @classmethod
    def from_config(cls, cfg: dict, generation_config: dict | None = None) -> "GemmaArch":
        t = cfg.get("text_config", cfg)
        n = int(t["num_hidden_layers"])
        types = tuple(t.get("layer_types") or
                      [FULL if (i + 1) % 6 == 0 else SLIDING for i in range(n)])
        if len(types) != n:
            raise ValueError(f"layer_types has {len(types)} entries for {n} layers")
        if types[-1] != FULL:
            types = types[:-1] + (FULL,)  # HF forces the last layer to full attention
        bad = [x for x in types if x not in (SLIDING, FULL)]
        if bad:
            raise ValueError(f"unknown layer types {sorted(set(bad))}")
        eos = (generation_config or {}).get("eos_token_id", cfg.get("eos_token_id", t.get("eos_token_id")))
        eos = tuple(eos) if isinstance(eos, (list, tuple)) else (() if eos is None else (int(eos),))
        return cls(
            hidden=int(t["hidden_size"]), n_layers=n, n_heads=int(t["num_attention_heads"]),
            vocab=int(t["vocab_size"]), eps=float(t.get("rms_norm_eps", 1e-6)),
            softcap=(None if t.get("final_logit_softcapping") is None
                     else float(t["final_logit_softcapping"])),
            window=int(t["sliding_window"]), layer_types=types,
            head_dim=int(t["head_dim"]), global_head_dim=int(t.get("global_head_dim") or t["head_dim"]),
            kv_heads=int(t["num_key_value_heads"]),
            global_kv_heads=int(t.get("num_global_key_value_heads") or t["num_key_value_heads"]),
            k_eq_v=bool(t.get("attention_k_eq_v", False)), intermediate=int(t["intermediate_size"]),
            tie_embeddings=bool(t.get("tie_word_embeddings", cfg.get("tie_word_embeddings", True))),
            rope={k: dict(v) for k, v in (t.get("rope_parameters") or {}).items()},
            max_position=int(t.get("max_position_embeddings", 0)),
            bos_id=int(t.get("bos_token_id", 2)), pad_id=int(t.get("pad_token_id", 0)),
            eos_ids=eos, activation=t.get("hidden_activation", "gelu_pytorch_tanh"))

    @classmethod
    def from_model_dir(cls, model_dir: Path | str) -> "GemmaArch":
        d = Path(model_dir)
        with open(d / "config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)
        gen = None
        if (d / "generation_config.json").is_file():
            with open(d / "generation_config.json", "r", encoding="utf-8") as f:
                gen = json.load(f)
        return cls.from_config(cfg, gen)

    # -- per-layer geometry -------------------------------------------------------------------

    def is_full(self, i: int) -> bool:
        return self.layer_types[i] == FULL

    def head_dim_of(self, i: int) -> int:
        return self.global_head_dim if self.is_full(i) else self.head_dim

    def kv_heads_of(self, i: int) -> int:
        return self.global_kv_heads if self.is_full(i) else self.kv_heads

    def has_v_proj(self, i: int) -> bool:
        return not (self.k_eq_v and self.is_full(i))

    def full_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == FULL]

    def sliding_layers(self) -> list[int]:
        return [i for i, t in enumerate(self.layer_types) if t == SLIDING]

    def rotary_dims(self, i: int) -> int:
        """Rotated dims of the head: all of head_dim for sliding (default rope), the first
        `partial_rotary_factor * head_dim` for full (proportional rope; the rest are NoPE)."""
        if not self.is_full(i):
            return self.head_dim
        return int(self.rope.get(FULL, {}).get("partial_rotary_factor", 1.0) * self.global_head_dim)

    def rope_theta(self, i: int) -> float:
        return float(self.rope.get(self.layer_types[i], {}).get("rope_theta", 10000.0))

    # -- tensors ------------------------------------------------------------------------------

    def layer_prefix(self, i: int) -> str:
        return f"{LM_PREFIX}layers.{i}."

    def linear_shapes(self, i: int) -> dict[str, tuple[int, int]]:
        """name (relative to layer_prefix) -> (out_features, in_features) of every Linear."""
        hd, nk = self.head_dim_of(i), self.kv_heads_of(i)
        s = {"self_attn.q_proj.weight": (self.n_heads * hd, self.hidden),
             "self_attn.k_proj.weight": (nk * hd, self.hidden),
             "self_attn.o_proj.weight": (self.hidden, self.n_heads * hd),
             "mlp.gate_proj.weight": (self.intermediate, self.hidden),
             "mlp.up_proj.weight": (self.intermediate, self.hidden),
             "mlp.down_proj.weight": (self.hidden, self.intermediate)}
        if self.has_v_proj(i):
            s["self_attn.v_proj.weight"] = (nk * hd, self.hidden)
        return s

    def norm_shapes(self, i: int) -> dict[str, tuple[int]]:
        hd = self.head_dim_of(i)
        return {"input_layernorm.weight": (self.hidden,), "post_attention_layernorm.weight": (self.hidden,),
                "pre_feedforward_layernorm.weight": (self.hidden,),
                "post_feedforward_layernorm.weight": (self.hidden,),
                "self_attn.q_norm.weight": (hd,), "self_attn.k_norm.weight": (hd,)}

    def layer_tensor_names(self, i: int) -> list[str]:
        """Every checkpoint tensor of layer i (full names), sorted. v_norm has no weight; layer_scalar is [1]."""
        p = self.layer_prefix(i)
        rel = list(self.linear_shapes(i)) + list(self.norm_shapes(i)) + ["layer_scalar"]
        return sorted(p + r for r in rel)

    def text_tensor_names(self) -> list[str]:
        out = [EMBED_NAME, FINAL_NORM_NAME]
        for i in range(self.n_layers):
            out += self.layer_tensor_names(i)
        return sorted(out)

    def quant_linears(self, i: int) -> dict[str, str]:
        """Quantization tap of each Linear (docs/gemma4-plan.md M1-25): the activation whose
        second moment is the Hessian input. `mlp_in` is pre_feedforward_layernorm's output."""
        taps = {"self_attn.q_proj.weight": "attn_in", "self_attn.k_proj.weight": "attn_in",
                "self_attn.o_proj.weight": "o_in", "mlp.gate_proj.weight": "mlp_in",
                "mlp.up_proj.weight": "mlp_in", "mlp.down_proj.weight": "down_in"}
        if self.has_v_proj(i):
            taps["self_attn.v_proj.weight"] = "attn_in"
        return taps

    # -- reporting ----------------------------------------------------------------------------

    def summary(self) -> dict:
        return {"hidden": self.hidden, "layers": self.n_layers, "heads": self.n_heads, "vocab": self.vocab,
                "eps": self.eps, "softcap": self.softcap, "window": self.window,
                "layer_types": "".join("F" if t == FULL else "s" for t in self.layer_types),
                "head_dim": self.head_dim, "global_head_dim": self.global_head_dim,
                "kv_heads": self.kv_heads, "global_kv_heads": self.global_kv_heads, "k_eq_v": self.k_eq_v,
                "intermediate": self.intermediate, "tie_embeddings": self.tie_embeddings,
                "bos_id": self.bos_id, "eos_ids": list(self.eos_ids)}


def check_against_header(arch: GemmaArch, tensors: dict[str, Any]) -> list[str]:
    """Problems between `arch` and a safetensors header (`name -> {"shape": [...], "dtype": ...}`,
    tensor_names.json's "tensors"); empty means the table agrees with the checkpoint for the text stack."""
    problems = []
    want = set(arch.text_tensor_names())
    have = {n for n in tensors if n.startswith(LM_PREFIX)}
    for n in sorted(want - have):
        problems.append(f"missing {n}")
    for n in sorted(have - want):
        problems.append(f"unexpected {n}")
    for i in range(arch.n_layers):
        p = arch.layer_prefix(i)
        for rel, shape in {**arch.linear_shapes(i), **arch.norm_shapes(i), "layer_scalar": (1,)}.items():
            e = tensors.get(p + rel)
            if e is not None and tuple(e["shape"]) != tuple(shape):
                problems.append(f"{p + rel}: shape {tuple(e['shape'])} != {tuple(shape)}")
    e = tensors.get(EMBED_NAME)
    if e is not None and tuple(e["shape"]) != (arch.vocab, arch.hidden):
        problems.append(f"{EMBED_NAME}: shape {tuple(e['shape'])} != {(arch.vocab, arch.hidden)}")
    return problems
