"""qwen4_exp architecture table: what the reference, quantization and validation scripts need to know about the
text stack, derived from a HF `config.json` (no torch, no transformers) -- the qwen4_exp counterpart of
tools/reference/gemma/arch.py.

Checkpoint tensor names are the on-disk names of Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ (`model.language_model.*`,
per-expert unfused linears, compressed-tensors `weight_packed/weight_scale/weight_zero_point/weight_shape` for the
int4 linears and plain `weight` for everything in the quantization `ignore` list). `tiny.py` writes the same names
with bf16 `weight` everywhere. `check_against_weight_map` verifies a checkpoint index against this table
(test_ref.py runs it on the tiny model and, when the repo is present, on the real one).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

GDN = "linear_attention"
QSA = "full_attention"
LM_PREFIX = "model.language_model."
EMBED_NAME = LM_PREFIX + "embed_tokens.weight"
LM_HEAD_NAME = "lm_head.weight"
MIXER_PREFIX = LM_PREFIX + "hyper_connection_mixer."
PACKED_PARTS = ("weight_packed", "weight_scale", "weight_zero_point", "weight_shape")
HC_PARTS = ("hc_norm.weight", "input_mix_weight_down.weight", "input_mix_weight_up.weight")


def text_config_of(config: dict) -> dict:
    return config.get("text_config", config)


@dataclass(frozen=True)
class Qwen4ExpArch:
    hidden: int
    n_layers: int
    vocab: int
    eps: float
    layer_types: tuple[str, ...]
    n_experts: int
    top_k: int
    moe_inter: int
    shared_inter: int
    hc_count: int
    hc_lowrank: int
    # GDN
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_head_dim: int
    gdn_conv: int
    # QSA
    q_heads: int
    kv_heads: int
    head_dim: int
    indexer_heads: int
    indexer_kv_heads: int
    indexer_head_dim: int
    indexer_budget: int
    indexer_ratio: int
    # PLE
    ple_layer_ids: tuple[int, ...]          # 1-indexed, as in the config
    ple_embed_dim: int
    ple_conv: int
    ngram_size: int
    heads_per_ngram: int
    ngram_parts: int
    eos_token_id: int

    @classmethod
    def from_config(cls, config: dict | str | Path) -> "Qwen4ExpArch":
        if not isinstance(config, dict):
            config = json.loads(Path(config).read_text(encoding="utf-8"))
        tc = text_config_of(config)
        eos = tc["eos_token_id"]
        eos = eos[0] if isinstance(eos, list) else eos
        n_layers = tc["num_hidden_layers"]
        types = tc.get("layer_types")
        if types is None:
            interval = tc.get("full_attention_interval", 4)
            types = [QSA if (i + 1) % interval == 0 else GDN for i in range(n_layers)]
        types = tuple(QSA if t in ("full_attention", "indexed_attention") else t for t in types)
        if len(types) != n_layers or set(types) - {GDN, QSA}:
            raise ValueError(f"layer_types {types} do not describe {n_layers} GDN/QSA layers")
        return cls(
            hidden=tc["hidden_size"], n_layers=n_layers, vocab=tc["vocab_size"], eps=tc.get("rms_norm_eps", 1e-6),
            layer_types=types, n_experts=tc["num_experts"], top_k=tc["num_experts_per_tok"],
            moe_inter=tc["moe_intermediate_size"], shared_inter=tc["shared_expert_intermediate_size"],
            hc_count=tc["hc_count"], hc_lowrank=tc["hc_lowrank"],
            gdn_key_heads=tc["linear_num_key_heads"], gdn_value_heads=tc["linear_num_value_heads"],
            gdn_head_dim=tc["linear_key_head_dim"], gdn_conv=tc["linear_conv_kernel_dim"],
            q_heads=tc["num_attention_heads"], kv_heads=tc["num_key_value_heads"], head_dim=tc["head_dim"],
            indexer_heads=tc["indexer_n_heads"], indexer_kv_heads=tc["indexer_kv_heads"],
            indexer_head_dim=tc["indexer_head_dim"], indexer_budget=tc["indexer_budget"],
            indexer_ratio=tc["indexer_compress_ratio"],
            ple_layer_ids=tuple(tc.get("ple_layer_ids") or ()), ple_embed_dim=tc.get("ple_embed_dim") or tc["hidden_size"],
            ple_conv=tc.get("ple_conv_kernel_size", 4), ngram_size=tc.get("ngram_size", 3),
            heads_per_ngram=tc.get("heads_per_ngram", 8), ngram_parts=tc.get("split_ngram_parts", 512), eos_token_id=eos,
        )

    @property
    def residual_width(self) -> int:
        return self.hc_count * self.hidden

    def ple_layers(self) -> tuple[int, ...]:
        """0-indexed decoder layers that run a PLE block."""
        return tuple(i - 1 for i in self.ple_layer_ids)

    def summary(self) -> dict[str, Any]:
        return {"hidden": self.hidden, "n_layers": self.n_layers, "vocab": self.vocab, "n_experts": self.n_experts,
                "top_k": self.top_k, "gdn_layers": [i for i, t in enumerate(self.layer_types) if t == GDN],
                "qsa_layers": [i for i, t in enumerate(self.layer_types) if t == QSA], "ple_layers": list(self.ple_layers()),
                "indexer_budget": self.indexer_budget, "indexer_ratio": self.indexer_ratio}

    # ---- checkpoint tensor names -------------------------------------------------------------------------

    @staticmethod
    def layer_prefix(i: int) -> str:
        return f"{LM_PREFIX}layers.{i}."

    def expert_base(self, i: int, e: int, proj: str) -> str:
        """Checkpoint name of one expert linear without the `.weight*` suffix (proj in gate_proj/up_proj/down_proj)."""
        return f"{self.layer_prefix(i)}mlp.experts.{e}.{proj}"

    def quantized_linears(self, i: int) -> dict[str, tuple[int, int]]:
        """{base name: (N, K)} of the linears the AWQ checkpoint stores int4 packed (everything in layer `i` that is a
        Linear and not in the quantization `ignore` list), experts included."""
        p = self.layer_prefix(i)
        h = self.hidden
        out: dict[str, tuple[int, int]] = {}
        if self.layer_types[i] == GDN:
            key_dim = self.gdn_key_heads * self.gdn_head_dim
            val_dim = self.gdn_value_heads * self.gdn_head_dim
            out[p + "linear_attn.in_proj_qkv"] = (2 * key_dim + val_dim, h)
            out[p + "linear_attn.in_proj_z"] = (val_dim, h)
            out[p + "linear_attn.out_proj"] = (h, val_dim)
        else:
            out[p + "self_attn.q_proj"] = (self.q_heads * self.head_dim * 2, h)
            out[p + "self_attn.k_proj"] = (self.kv_heads * self.head_dim, h)
            out[p + "self_attn.v_proj"] = (self.kv_heads * self.head_dim, h)
            out[p + "self_attn.o_proj"] = (h, self.q_heads * self.head_dim)
        for e in range(self.n_experts):
            out[self.expert_base(i, e, "gate_proj")] = (self.moe_inter, h)
            out[self.expert_base(i, e, "up_proj")] = (self.moe_inter, h)
            out[self.expert_base(i, e, "down_proj")] = (h, self.moe_inter)
        return out

    def dense_tensors(self, i: int) -> dict[str, tuple[int, ...]]:
        """{name: shape} of the layer's tensors that are never quantized (bf16 `weight` / buffers), without n-gram shards."""
        p = self.layer_prefix(i)
        h, w = self.hidden, self.residual_width
        out: dict[str, tuple[int, ...]] = {}
        for hc in ("attn_hyper_connection.", "mlp_hyper_connection."):
            out[p + hc + "hc_norm.weight"] = (w,)
            out[p + hc + "input_mix_weight_down.weight"] = (self.hc_lowrank, w)
            out[p + hc + "input_mix_weight_up.weight"] = (w, self.hc_lowrank)
            out[p + hc + "block_inject_weight.weight"] = (self.hc_count, w)
        out[p + "mlp.gate.weight"] = (self.n_experts, h)
        out[p + "mlp.shared_expert.gate_proj.weight"] = (self.shared_inter, h)
        out[p + "mlp.shared_expert.up_proj.weight"] = (self.shared_inter, h)
        out[p + "mlp.shared_expert.down_proj.weight"] = (h, self.shared_inter)
        out[p + "mlp.shared_expert_gate.weight"] = (1, h)
        if self.layer_types[i] == GDN:
            key_dim = self.gdn_key_heads * self.gdn_head_dim
            val_dim = self.gdn_value_heads * self.gdn_head_dim
            out[p + "linear_attn.A_log"] = (self.gdn_value_heads,)
            out[p + "linear_attn.dt_bias"] = (self.gdn_value_heads,)
            out[p + "linear_attn.conv1d.weight"] = (2 * key_dim + val_dim, 1, self.gdn_conv)
            out[p + "linear_attn.in_proj_a.weight"] = (self.gdn_value_heads, h)
            out[p + "linear_attn.in_proj_b.weight"] = (self.gdn_value_heads, h)
            out[p + "linear_attn.norm.weight"] = (self.gdn_head_dim,)
        else:
            out[p + "self_attn.q_norm.weight"] = (self.head_dim,)
            out[p + "self_attn.k_norm.weight"] = (self.head_dim,)
            ix = self.indexer_head_dim
            out[p + "self_attn.indexer.index_qk_proj.weight"] = ((self.indexer_heads + self.indexer_kv_heads) * ix, h)
            out[p + "self_attn.indexer.q_layernorm.weight"] = (ix,)
            out[p + "self_attn.indexer.k_layernorm.weight"] = (ix,)
        if i in self.ple_layers():
            heads = (self.ngram_size - 1) * self.heads_per_ngram
            out[p + "ple.key_proj.weight"] = (w, self.ple_embed_dim)
            out[p + "ple.value_proj.weight"] = (h, self.ple_embed_dim)
            for n in ("norm_key", "norm_query", "norm_conv"):
                out[p + f"ple.{n}.weight"] = (w,)
            out[p + "ple.conv1d.weight"] = (w, 1, self.ple_conv)
            out[p + "ple.ple_embedding.layer_multipliers"] = (self.ngram_size,)
            out[p + "ple.ple_embedding.ngram_heads_offsets"] = (heads,)
            out[p + "ple.ple_embedding.ngram_heads_vocab_sizes"] = (heads,)
        return out

    def ngram_shard_names(self, i: int) -> list[str]:
        return [f"{self.layer_prefix(i)}ple.ple_embedding.ngram_embedding.shard_{k}.weight" for k in range(self.ngram_parts)]

    def global_tensors(self) -> dict[str, tuple[int, ...]]:
        w = self.residual_width
        return {
            EMBED_NAME: (self.vocab, self.hidden),
            LM_HEAD_NAME: (self.vocab, self.hidden),
            MIXER_PREFIX + "hc_norm.weight": (w,),
            MIXER_PREFIX + "input_mix_weight_down.weight": (self.hc_lowrank, w),
            MIXER_PREFIX + "input_mix_weight_up.weight": (w, self.hc_lowrank),
        }

    def check_against_weight_map(self, weight_map: Iterable[str], layers: Iterable[int] | None = None) -> list[str]:
        """Problems between a checkpoint's tensor names and this table (empty list: they agree).

        `weight_map` is the key set of model.safetensors.index.json. Visual-tower and `mtp.*` tensors are ignored;
        every other name must be expected, and every expected name must exist (as plain `weight`, or as the four
        compressed-tensors parts for a quantized linear -- a linear is either all-plain or all-packed). `layers`
        limits the check to those layer indices."""
        names = set(weight_map)
        layers = list(range(self.n_layers)) if layers is None else list(layers)
        problems: list[str] = []
        seen: set[str] = set()

        def expect(name: str):
            seen.add(name)
            if name not in names:
                problems.append(f"missing {name}")

        for name in self.global_tensors():
            expect(name)
        for i in layers:
            for name in self.dense_tensors(i):
                expect(name)
            if i in self.ple_layers():
                for name in self.ngram_shard_names(i):
                    expect(name)
            for base in self.quantized_linears(i):
                plain = base + ".weight" in names
                packed = [base + "." + part in names for part in PACKED_PARTS]
                if plain and any(packed):
                    problems.append(f"{base}: both a plain weight and packed parts")
                elif not plain and not all(packed):
                    problems.append(f"missing {base} (neither .weight nor all of {', '.join(PACKED_PARTS)})")
                seen.update(base + "." + part for part in ("weight", *PACKED_PARTS))
        prefixes = tuple(self.layer_prefix(i) for i in layers)
        for name in names:
            if name in seen or name.startswith("model.visual.") or name.startswith("mtp."):
                continue
            if name.startswith(prefixes) or (not name.startswith(LM_PREFIX + "layers.") and name.startswith(("lm_head", LM_PREFIX))):
                problems.append(f"unexpected {name}")
        return sorted(problems)

    def layer_weight_names(self, i: int) -> list[str]:
        """Every plain-or-packed base/tensor name of layer `i` (dense tensors + quantized linear bases)."""
        return [*self.dense_tensors(i), *self.quantized_linears(i)]
