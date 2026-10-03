"""tools/reference/arch_table.py -- the architecture table of the quantization tools (docs/gemma4-plan.md
4.5 / M1-25): everything `hessian_capture.py`, `imatrix_capture.py` and `trellis_quant.py` need to know about
WHICH linears a checkpoint family has, which activation feeds each, which norm precedes it and how the
container names it. One `ArchSpec` per `model_type`; the Qwen3.5 entry is the tables those scripts used to
hard-code (their module-level constants are now aliases of it, so a Qwen run is byte-for-byte what it was),
the Gemma 4 entry is new.

    spec = get_arch("gemma4_unified")         # or detect_arch(model_dir)
    spec.module_list("full_attention")        # HF modules of a full-attention layer (no v_proj: k_eq_v)
    spec.tap_of, spec.shared_with, ...        # the Hessian taps

No torch, no transformers: importing this is free (the dry-run paths and the tests use it on any python).

Gemma 4 taps (docs/gemma4-semantics.md; tensor names of tools/reference/gemma/tensor_names.json):
  attn_in  = input of q_proj (= input_layernorm's output; k_proj and, on sliding layers, v_proj read the same
             tensor; the 8 full layers have NO v_proj: V is the raw k_proj output)
  o_in     = input of o_proj
  mlp_in   = input of gate_proj = **pre_feedforward_layernorm**'s output (up_proj reads the same tensor); NOT
             post_attention_layernorm, which is Qwen's norm before the MLP and for Gemma is a post-norm of the
             attention output (sandwich norms)
  down_in  = input of down_proj
  lm_head  = the post-final-norm hidden (before the softcap), for the tied head
The file names keep the Qwen tap ids (`in`, `out`, `mlp_in`, `mlp_mid`: L{i:02d}.<tap>.hess) so a Gemma
hessian.json looks like a Qwen one to r4dx-convert's HessianStore; only the keys differ (attn.q instead of
attn.qg, no gdn.*, no mtp.*).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[2]
LAYER_PREFIX = "model.language_model.layers."
FULL = "full_attention"
SLIDING = "sliding_attention"
LINEAR = "linear_attention"


@dataclass(frozen=True)
class ArchSpec:
    name: str                          #: "qwen3_5" | "gemma4_unified"
    model_types: tuple                 #: config.json model_type values that select it
    converter_arch: str                #: imatrix_capture.audit_converter_source(arch=...)
    converter_path: str                #: repo-relative source whose add_linear set the audit re-derives
    norm_offset: float                 #: norm scale = offset + w: 1.0 Qwen (zero-centred), 0.0 Gemma (plain)
    lm_head_name: str                  #: HF tensor the container's lm_head is made from
    has_mtp: bool
    #: HF module (under a decoder layer) -> (container key suffix, tap) -- trellis_quant's LINEARS
    linears: dict = field(hash=False)
    #: HF module -> the qgroup that the allocator promotes as a unit (docs/trellis.md 9)
    qgroups: dict = field(hash=False)
    #: layer type -> the HF modules of such a layer, in the allocator's (EXL3's) module order
    modules_by_layer_type: dict = field(hash=False)
    #: hessian_capture: the norm that feeds each representative module (None = not norm-fed)
    norm_before: dict = field(hash=False)
    shared_with: dict = field(hash=False)      #: module whose input IS another module's input -> that module
    tap_of: dict = field(hash=False)           #: representative module -> tap id (file L{i}.<tap>.hess)
    mtp_tap_of: dict = field(hash=False)
    shared_gate: dict = field(hash=False)      #: layer type -> [(representative, [companions])]
    rms_norms: tuple = ()                      #: (norm module whose INPUT is hooked, tap id)
    rms_key_suffixes: dict = field(default_factory=dict, hash=False)
    attention_layer_types: tuple = (FULL,)

    # -- derived -----------------------------------------------------------------------------------

    def module_list(self, layer_type: str) -> list:
        try:
            return list(self.modules_by_layer_type[layer_type])
        except KeyError:
            raise ValueError(f"{self.name}: unknown layer type {layer_type!r} "
                             f"(have {sorted(self.modules_by_layer_type)})") from None

    def hf_name(self, layer: int, module: str) -> str:
        return f"{LAYER_PREFIX}{layer}.{module}.weight"

    def container_key(self, layer: int, module: str) -> str:
        return f"text.layers.{layer}.{self.linears[module][0]}"

    def tap_modules(self) -> set:
        """Every module a Hessian is hooked on (the representatives)."""
        return set(self.tap_of)

    @property
    def is_gemma(self) -> bool:
        return self.name == "gemma4_unified"

    def converter_source(self) -> Path:
        return REPO_ROOT / self.converter_path


# ---------------------------------------------------------------------------------------------------
# Qwen3.5 -- the tables hessian_capture.py / trellis_quant.py carried (unchanged values)
# ---------------------------------------------------------------------------------------------------

_QWEN_ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
_QWEN_GDN = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"]
_MLP = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

QWEN = ArchSpec(
    name="qwen3_5",
    model_types=("qwen3_5", "qwen3_5_text", "qwen3_5_moe"),
    converter_arch="qwen35",
    converter_path="src/convert/main.cpp",
    norm_offset=1.0,
    lm_head_name="lm_head.weight",
    has_mtp=True,
    linears={
        "self_attn.q_proj": ("attn.qg", "in"),
        "self_attn.k_proj": ("attn.k", "in"),
        "self_attn.v_proj": ("attn.v", "in"),
        "self_attn.o_proj": ("attn.o", "out"),
        "linear_attn.in_proj_qkv": ("gdn.in_proj_qkv", "in"),
        "linear_attn.in_proj_z": ("gdn.in_proj_z", "in"),
        "linear_attn.out_proj": ("gdn.out_proj", "out"),
        "mlp.gate_proj": ("mlp.gate_up", "mlp_in"),
        "mlp.up_proj": ("mlp.gate_up", "mlp_in"),
        "mlp.down_proj": ("mlp.down", "mlp_mid"),
    },
    qgroups={
        "self_attn.q_proj": "self_attn.qkv",
        "self_attn.k_proj": "self_attn.qkv",
        "self_attn.v_proj": "self_attn.qkv",
        "self_attn.o_proj": "self_attn.o",
        "linear_attn.in_proj_qkv": "linear_attn.qkvz",
        "linear_attn.in_proj_z": "linear_attn.qkvz",
        "linear_attn.out_proj": "linear_attn.o",
        "mlp.gate_proj": "mlp.gu",
        "mlp.up_proj": "mlp.gu",
        "mlp.down_proj": "mlp.d",
    },
    modules_by_layer_type={FULL: _QWEN_ATTN + _MLP, LINEAR: _QWEN_GDN + _MLP},
    norm_before={
        "linear_attn.in_proj_qkv": "input_layernorm",
        "self_attn.q_proj": "input_layernorm",
        "mlp.gate_proj": "post_attention_layernorm",
    },
    shared_with={
        "linear_attn.in_proj_z": "linear_attn.in_proj_qkv",
        "self_attn.k_proj": "self_attn.q_proj",
        "self_attn.v_proj": "self_attn.q_proj",
    },
    tap_of={
        "linear_attn.in_proj_qkv": "in",
        "self_attn.q_proj": "in",
        "linear_attn.out_proj": "out",
        "self_attn.o_proj": "out",
        "mlp.gate_proj": "mlp_in",
        "mlp.down_proj": "mlp_mid",
    },
    mtp_tap_of={
        "self_attn.q_proj": "in",
        "self_attn.o_proj": "out",
        "mlp.gate_proj": "mlp_in",
        "mlp.down_proj": "mlp_mid",
    },
    shared_gate={
        LINEAR: [("linear_attn.in_proj_qkv", ["linear_attn.in_proj_z"]),
                 ("mlp.gate_proj", ["mlp.up_proj"])],
        FULL: [("self_attn.q_proj", ["self_attn.k_proj", "self_attn.v_proj"]),
               ("mlp.gate_proj", ["mlp.up_proj"])],
    },
    rms_norms=(("input_layernorm", "in"), ("post_attention_layernorm", "mlp_in")),
    rms_key_suffixes={"in": ("gdn.in_proj_qkv", "gdn.in_proj_z", "attn.qg", "attn.k", "attn.v"),
                      "mlp_in": ("mlp.gate_up",)},
    attention_layer_types=(FULL,),
)

# ---------------------------------------------------------------------------------------------------
# Gemma 4 (gemma4_unified)
# ---------------------------------------------------------------------------------------------------

_GEMMA_ATTN_SLIDING = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
_GEMMA_ATTN_FULL = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.o_proj"]  # no v_proj: k_eq_v

GEMMA = ArchSpec(
    name="gemma4_unified",
    model_types=("gemma4_unified", "gemma4_unified_text"),
    converter_arch="gemma4_unified",
    converter_path="src/convert/gemma_layout.cpp",
    norm_offset=0.0,
    lm_head_name="model.language_model.embed_tokens.weight",  # tied: the head is the embedding table
    has_mtp=False,
    linears={
        "self_attn.q_proj": ("attn.q", "in"),
        "self_attn.k_proj": ("attn.k", "in"),
        "self_attn.v_proj": ("attn.v", "in"),
        "self_attn.o_proj": ("attn.o", "out"),
        "mlp.gate_proj": ("mlp.gate_up", "mlp_in"),
        "mlp.up_proj": ("mlp.gate_up", "mlp_in"),
        "mlp.down_proj": ("mlp.down", "mlp_mid"),
    },
    qgroups={
        "self_attn.q_proj": "self_attn.qkv",
        "self_attn.k_proj": "self_attn.qkv",
        "self_attn.v_proj": "self_attn.qkv",  # sliding layers only; a full layer's group is {q, k}
        "self_attn.o_proj": "self_attn.o",
        "mlp.gate_proj": "mlp.gu",
        "mlp.up_proj": "mlp.gu",
        "mlp.down_proj": "mlp.d",
    },
    modules_by_layer_type={SLIDING: _GEMMA_ATTN_SLIDING + _MLP, FULL: _GEMMA_ATTN_FULL + _MLP},
    norm_before={
        "self_attn.q_proj": "input_layernorm",
        "mlp.gate_proj": "pre_feedforward_layernorm",   # NOT post_attention_layernorm (sandwich norms)
    },
    shared_with={
        "self_attn.k_proj": "self_attn.q_proj",
        "self_attn.v_proj": "self_attn.q_proj",
    },
    tap_of={
        "self_attn.q_proj": "in",
        "self_attn.o_proj": "out",
        "mlp.gate_proj": "mlp_in",
        "mlp.down_proj": "mlp_mid",
    },
    mtp_tap_of={},
    shared_gate={
        SLIDING: [("self_attn.q_proj", ["self_attn.k_proj", "self_attn.v_proj"]),
                  ("mlp.gate_proj", ["mlp.up_proj"])],
        FULL: [("self_attn.q_proj", ["self_attn.k_proj"]),   # k_eq_v: no v_proj to compare
               ("mlp.gate_proj", ["mlp.up_proj"])],
    },
    rms_norms=(("input_layernorm", "in"), ("pre_feedforward_layernorm", "mlp_in")),
    rms_key_suffixes={"in": ("attn.q", "attn.k", "attn.v"), "mlp_in": ("mlp.gate_up",)},
    attention_layer_types=(SLIDING, FULL),
)

ARCHES = {QWEN.name: QWEN, GEMMA.name: GEMMA}
ARCH_NAMES = tuple(ARCHES)


def get_arch(name: str | None) -> ArchSpec:
    """The spec of `name` (None = qwen3_5, the historical default)."""
    if name is None:
        return QWEN
    for spec in ARCHES.values():
        if name == spec.name or name in spec.model_types:
            return spec
    raise ValueError(f"unknown arch {name!r} (have {', '.join(ARCH_NAMES)})")


def read_config(model_dir: Path | str) -> dict:
    with open(Path(model_dir) / "config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def detect_arch(model_dir: Path | str) -> ArchSpec:
    """The arch of a checkpoint directory, from config.json's model_type (top level, then
    text_config's); anything that is not Gemma 4 is the Qwen3.5 default, as before this table existed."""
    cfg = read_config(model_dir)
    for mt in (cfg.get("model_type"), (cfg.get("text_config") or {}).get("model_type")):
        if mt in GEMMA.model_types:
            return GEMMA
    return QWEN


def resolve_arch(requested: str | None, model_dir: Path | str | None) -> ArchSpec:
    """`--arch` of the tools: 'auto'/None = detect from model_dir (Qwen when there is none)."""
    if requested in (None, "auto"):
        return detect_arch(model_dir) if model_dir is not None and (Path(model_dir) / "config.json").is_file() else QWEN
    return get_arch(requested)


def text_config_view(arch: ArchSpec, model_dir: Path | str):
    """What enumerate_quantized_linears needs of a text config (`layer_types`, `num_hidden_layers`,
    `hidden_size`): the transformers config for Qwen (common.load_text_config), a plain namespace read
    from config.json for Gemma (no transformers, so the CPU dry-run needs only numpy)."""
    if not arch.is_gemma:
        from common import load_text_config

        return load_text_config(Path(model_dir))[1]
    cfg = read_config(model_dir)
    t = cfg.get("text_config", cfg)
    n = int(t["num_hidden_layers"])
    types = list(t.get("layer_types") or [FULL if (i + 1) % 6 == 0 else SLIDING for i in range(n)])
    if len(types) != n:
        raise ValueError(f"layer_types has {len(types)} entries for {n} layers")
    return SimpleNamespace(layer_types=types, num_hidden_layers=n, hidden_size=int(t["hidden_size"]),
                           intermediate_size=int(t["intermediate_size"]), vocab_size=int(t["vocab_size"]),
                           num_attention_heads=int(t["num_attention_heads"]),
                           head_dim=int(t["head_dim"]), global_head_dim=int(t.get("global_head_dim") or t["head_dim"]),
                           rms_norm_eps=float(t.get("rms_norm_eps", 1e-6)))


def norm_scale_name(arch: ArchSpec, layer: int, norm: str) -> str:
    return f"{LAYER_PREFIX}{layer}.{norm}.weight"


RMS_POST_FILE_RE = re.compile(r"L(\d{2,})\.(in|mlp_in)\.hess")
