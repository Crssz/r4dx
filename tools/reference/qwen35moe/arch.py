"""qwen3_5_moe (Ornith-1.5-122B) architecture table: what the reference, quantization and validation scripts need to know about
the text stack, derived from a HF `config.json` (no torch, no transformers) -- the qwen3_5_moe counterpart of
tools/reference/qwen4exp/arch.py.

Checkpoint tensor names are those of the HF 5.x checkpoint (docs/qwen35moe-plan.md 1.1): `model.language_model.*`, plain bf16
`weight` everywhere, and the routed experts of each layer as TWO FUSED 3-D tensors

    <layer>.mlp.experts.gate_up_proj   [E, 2 * I, H]   expert e: rows [0, I) gate, rows [I, 2I) up
    <layer>.mlp.experts.down_proj      [E, H, I]

The quantization tooling (and r4dx-convert, through its FusedExpertsOverlay) addresses an expert linear by the VIRTUAL
per-expert base name `<layer>.mlp.experts.{e}.{gate_proj|up_proj|down_proj}` (what `expert_base` returns); `ref.CheckpointReader`
slices the fused tensors for them. `tiny.py` writes the same names.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

GDN = "linear_attention"
FULL = "full_attention"
LM_PREFIX = "model.language_model."
EMBED_NAME = LM_PREFIX + "embed_tokens.weight"
FINAL_NORM_NAME = LM_PREFIX + "norm.weight"
LM_HEAD_NAME = "lm_head.weight"


def text_config_of(config: dict) -> dict:
    return config.get("text_config", config)


@dataclass(frozen=True)
class Qwen35MoeArch:
    hidden: int
    n_layers: int
    vocab: int
    eps: float
    layer_types: tuple[str, ...]
    n_experts: int
    top_k: int
    moe_inter: int
    shared_inter: int
    # GDN
    gdn_key_heads: int
    gdn_value_heads: int
    gdn_key_dim: int
    gdn_value_dim: int
    gdn_conv: int
    # full attention
    q_heads: int
    kv_heads: int
    head_dim: int
    eos_token_id: int

    @classmethod
    def from_config(cls, config: dict | str | Path) -> "Qwen35MoeArch":
        if not isinstance(config, dict):
            config = json.loads(Path(config).read_text(encoding="utf-8"))
        tc = text_config_of(config)
        eos = tc["eos_token_id"]
        eos = eos[0] if isinstance(eos, list) else eos
        n_layers = tc["num_hidden_layers"]
        types = tc.get("layer_types")
        if types is None:
            interval = tc.get("full_attention_interval", 4)
            types = [FULL if (i + 1) % interval == 0 else GDN for i in range(n_layers)]
        types = tuple(types)
        if len(types) != n_layers or set(types) - {GDN, FULL}:
            raise ValueError(f"layer_types {types} do not describe {n_layers} GDN/full-attention layers")
        if tc.get("mlp_only_layers"):
            raise ValueError("mlp_only_layers is not empty: dense layers inside the MoE stack are not supported")
        return cls(
            hidden=tc["hidden_size"], n_layers=n_layers, vocab=tc["vocab_size"], eps=tc.get("rms_norm_eps", 1e-6),
            layer_types=types, n_experts=tc["num_experts"], top_k=tc["num_experts_per_tok"],
            moe_inter=tc["moe_intermediate_size"], shared_inter=tc["shared_expert_intermediate_size"],
            gdn_key_heads=tc["linear_num_key_heads"], gdn_value_heads=tc["linear_num_value_heads"],
            gdn_key_dim=tc["linear_key_head_dim"], gdn_value_dim=tc["linear_value_head_dim"], gdn_conv=tc["linear_conv_kernel_dim"],
            q_heads=tc["num_attention_heads"], kv_heads=tc["num_key_value_heads"], head_dim=tc["head_dim"], eos_token_id=eos,
        )

    # ---- derived sizes -----------------------------------------------------------------------------------

    @property
    def gdn_qkv_rows(self) -> int:
        return 2 * self.gdn_key_heads * self.gdn_key_dim + self.gdn_value_heads * self.gdn_value_dim

    @property
    def gdn_z_rows(self) -> int:
        return self.gdn_value_heads * self.gdn_value_dim

    def summary(self) -> dict[str, Any]:
        return {"hidden": self.hidden, "n_layers": self.n_layers, "vocab": self.vocab, "n_experts": self.n_experts,
                "top_k": self.top_k, "gdn_layers": [i for i, t in enumerate(self.layer_types) if t == GDN],
                "full_layers": [i for i, t in enumerate(self.layer_types) if t == FULL]}

    # ---- checkpoint tensor names -------------------------------------------------------------------------

    @staticmethod
    def layer_prefix(i: int) -> str:
        return f"{LM_PREFIX}layers.{i}."

    def fused_names(self, i: int) -> tuple[str, str]:
        p = self.layer_prefix(i)
        return p + "mlp.experts.gate_up_proj", p + "mlp.experts.down_proj"

    def expert_base(self, i: int, e: int, proj: str) -> str:
        """Virtual checkpoint name of one expert linear without the `.weight` suffix (proj in gate_proj/up_proj/down_proj)."""
        return f"{self.layer_prefix(i)}mlp.experts.{e}.{proj}"

    def quantized_linears(self, i: int) -> dict[str, tuple[int, int]]:
        """{base name: (N, K)} of the linears the converter turns into trellis (everything `LinearRole::kBody` in
        src/convert/qwen35moe_layout.cpp), experts included under their virtual per-expert names. The router, the shared expert,
        in_proj_a/b and the lm_head are not in it."""
        p = self.layer_prefix(i)
        h = self.hidden
        out: dict[str, tuple[int, int]] = {}
        if self.layer_types[i] == GDN:
            out[p + "linear_attn.in_proj_qkv"] = (self.gdn_qkv_rows, h)
            out[p + "linear_attn.in_proj_z"] = (self.gdn_z_rows, h)
            out[p + "linear_attn.out_proj"] = (h, self.gdn_z_rows)
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
        """{name: shape} of the layer's tensors that are never trellis-quantized (they stay bf16 / fp32 in the container)."""
        p = self.layer_prefix(i)
        h = self.hidden
        out: dict[str, tuple[int, ...]] = {
            p + "input_layernorm.weight": (h,),
            p + "post_attention_layernorm.weight": (h,),
            p + "mlp.gate.weight": (self.n_experts, h),
            p + "mlp.shared_expert.gate_proj.weight": (self.shared_inter, h),
            p + "mlp.shared_expert.up_proj.weight": (self.shared_inter, h),
            p + "mlp.shared_expert.down_proj.weight": (h, self.shared_inter),
            p + "mlp.shared_expert_gate.weight": (1, h),
        }
        if self.layer_types[i] == GDN:
            out[p + "linear_attn.A_log"] = (self.gdn_value_heads,)
            out[p + "linear_attn.dt_bias"] = (self.gdn_value_heads,)
            out[p + "linear_attn.conv1d.weight"] = (self.gdn_qkv_rows, 1, self.gdn_conv)
            out[p + "linear_attn.in_proj_a.weight"] = (self.gdn_value_heads, h)
            out[p + "linear_attn.in_proj_b.weight"] = (self.gdn_value_heads, h)
            out[p + "linear_attn.norm.weight"] = (self.gdn_value_dim,)
        else:
            out[p + "self_attn.q_norm.weight"] = (self.head_dim,)
            out[p + "self_attn.k_norm.weight"] = (self.head_dim,)
        return out

    def global_tensors(self) -> dict[str, tuple[int, ...]]:
        return {EMBED_NAME: (self.vocab, self.hidden), FINAL_NORM_NAME: (self.hidden,), LM_HEAD_NAME: (self.vocab, self.hidden)}

    def check_against_weight_map(self, weight_map: Iterable[str], layers: Iterable[int] | None = None) -> list[str]:
        """Problems between a checkpoint's tensor names and this table (empty list: they agree). `weight_map` is the key set of
        model.safetensors.index.json (fused experts); the visual tower and `mtp.*` are ignored; `layers` limits the check."""
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
            for name in self.fused_names(i):
                expect(name)
            p = self.layer_prefix(i)
            mixer = ("linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj") if self.layer_types[i] == GDN \
                else ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj")
            for m in mixer:
                expect(p + m + ".weight")
        prefixes = tuple(self.layer_prefix(i) for i in layers)
        for name in names:
            if name in seen or name.startswith("model.visual.") or name.startswith("mtp."):
                continue
            if name.startswith(prefixes) or (not name.startswith(LM_PREFIX + "layers.") and name.startswith(("lm_head", LM_PREFIX))):
                problems.append(f"unexpected {name}")
        return sorted(problems)
