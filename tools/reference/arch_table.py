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

import functools
import json
import os
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
    #: HF name prefix of a decoder layer (LFM2: `model.layers.`; everything else `model.language_model.layers.`)
    layer_prefix: str = LAYER_PREFIX
    #: LFM2 only: routed-expert count of the checkpoint this spec was built for (its per-expert module tables
    #: are sized by it); 0 for every other arch
    n_experts: int = 0

    # -- derived -----------------------------------------------------------------------------------

    def module_list(self, layer_type: str) -> list:
        try:
            return list(self.modules_by_layer_type[layer_type])
        except KeyError:
            raise ValueError(f"{self.name}: unknown layer type {layer_type!r} "
                             f"(have {sorted(self.modules_by_layer_type)})") from None

    def hf_name(self, layer: int, module: str) -> str:
        return f"{self.layer_prefix}{layer}.{module}.weight"

    def container_key(self, layer: int, module: str) -> str:
        return f"text.layers.{layer}.{self.linears[module][0]}"

    def tap_modules(self) -> set:
        """Every module a Hessian is hooked on (the representatives)."""
        return set(self.tap_of)

    @property
    def is_gemma(self) -> bool:
        return self.name == "gemma4_unified"

    @property
    def is_lfm2(self) -> bool:
        return self.name == "lfm2_moe"

    def layer_kinds(self, cfg: dict) -> list:
        """The per-layer type strings the tables below are keyed by. Qwen / Gemma: config's `layer_types`
        verbatim. LFM2: `<conv|full_attention>_<dense|moe>` (the FFN kind follows `num_dense_layers`)."""
        types = list(cfg["layer_types"])
        if not self.is_lfm2:
            return types
        nd = int(cfg.get("num_dense_layers", 2))
        return [f"{t}_{'dense' if i < nd else 'moe'}" for i, t in enumerate(types)]

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

# ---------------------------------------------------------------------------------------------------
# LFM2 (lfm2_moe; LiquidAI/LFM2.5-24B-A2B, docs/lfm2-plan.md 7.1)
# ---------------------------------------------------------------------------------------------------
#
# Layer kinds are `<conv|full_attention>_<dense|moe>` (ArchSpec.layer_kinds). HF names (the REAL checkpoint's,
# PER-EXPERT: experts.{e}.{w1 = gate, w3 = up, w2 = down}):
#   attention  self_attn.{q,k,v,out}_proj        container attn.{q,k,v,o}        taps in (q,k,v) / out
#   ShortConv  conv.in_proj / conv.out_proj      container conv.{in_proj,out_proj} taps in / out
#   dense FFN  feed_forward.{w1,w3} / w2         container mlp.gate_up / mlp.down taps mlp_in / mlp_mid
#   MoE FFN    feed_forward.experts.{e}.{w1,w3} / w2   container moe.experts.{e}.gate_up / .down
#              taps `E{e:02d}.mlp_in` / `E{e:02d}.mlp_mid`  (files L{i:02d}.E{e:02d}.mlp_in.hess ...):
#              each is the Hessian of the tokens ROUTED to expert e (gate/up share one input, as dense).
# `conv.conv.weight`, the router `feed_forward.gate.weight`, `expert_bias`, every norm and q/k layernorm stay
# bf16/fp32 passthrough (never quantized, never captured). Taps `in` / `mlp_in` are the outputs of
# `operator_norm` / `ffn_norm` (plain scale w, norm_offset 0).
# Quantization groups (the allocator's promotion units, plan 7.1): mixer in (q,k,v | conv.in_proj), mixer out,
# FFN gate_up of ALL experts of a layer (`moe.gu`), FFN down of ALL experts (`moe.d`).

LFM2_PREFIX = "model.layers."
#: The bf16 checkpoint (plan 6.2: R4DX_MODEL_DIR overrides, as common_lfm2.py will document)
LFM2_MODEL_DIR = Path(os.environ.get("R4DX_MODEL_DIR")
                      or Path(os.environ.get("R4DX_MODELS_ROOT", r"E:\models")) / "LiquidAI" / "LFM2.5-24B-A2B")
LFM2_CONV = "conv"
LFM2_KINDS = ("conv_dense", "conv_moe", "full_attention_dense", "full_attention_moe")
_LFM2_ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.out_proj"]
_LFM2_CONV = ["conv.in_proj", "conv.out_proj"]
_LFM2_DENSE = ["feed_forward.w1", "feed_forward.w3", "feed_forward.w2"]
LFM2_EXPERT_RE = re.compile(r"feed_forward\.experts\.(\d+)\.(w1|w3|w2)")


def lfm2_expert_modules(e: int) -> list:
    return [f"feed_forward.experts.{e}.w1", f"feed_forward.experts.{e}.w3", f"feed_forward.experts.{e}.w2"]


def lfm2_expert_of(module: str):
    """`feed_forward.experts.5.w3` -> (5, 'w3'); None for any other module."""
    m = LFM2_EXPERT_RE.fullmatch(module)
    return (int(m.group(1)), m.group(2)) if m else None


@functools.lru_cache(maxsize=None)
def lfm2_arch(n_experts: int = 64) -> ArchSpec:
    """The lfm2_moe spec for a checkpoint with `n_experts` routed experts (the real one has 64)."""
    experts = [m for e in range(n_experts) for m in lfm2_expert_modules(e)]
    linears = {
        "self_attn.q_proj": ("attn.q", "in"), "self_attn.k_proj": ("attn.k", "in"),
        "self_attn.v_proj": ("attn.v", "in"), "self_attn.out_proj": ("attn.o", "out"),
        "conv.in_proj": ("conv.in_proj", "in"), "conv.out_proj": ("conv.out_proj", "out"),
        "feed_forward.w1": ("mlp.gate_up", "mlp_in"), "feed_forward.w3": ("mlp.gate_up", "mlp_in"),
        "feed_forward.w2": ("mlp.down", "mlp_mid"),
    }
    qgroups = {
        "self_attn.q_proj": "self_attn.qkv", "self_attn.k_proj": "self_attn.qkv",
        "self_attn.v_proj": "self_attn.qkv", "self_attn.out_proj": "self_attn.o",
        "conv.in_proj": "conv.in", "conv.out_proj": "conv.o",
        "feed_forward.w1": "mlp.gu", "feed_forward.w3": "mlp.gu", "feed_forward.w2": "mlp.d",
    }
    norm_before = {"self_attn.q_proj": "operator_norm", "conv.in_proj": "operator_norm",
                   "feed_forward.w1": "ffn_norm"}
    shared_with = {"self_attn.k_proj": "self_attn.q_proj", "self_attn.v_proj": "self_attn.q_proj",
                   "feed_forward.w3": "feed_forward.w1"}
    tap_of = {"self_attn.q_proj": "in", "conv.in_proj": "in", "self_attn.out_proj": "out",
              "conv.out_proj": "out", "feed_forward.w1": "mlp_in", "feed_forward.w2": "mlp_mid"}
    for e in range(n_experts):
        w1, w3, w2 = lfm2_expert_modules(e)
        tag = f"E{e:02d}"
        linears[w1] = linears[w3] = (f"moe.experts.{e}.gate_up", f"{tag}.mlp_in")
        linears[w2] = (f"moe.experts.{e}.down", f"{tag}.mlp_mid")
        qgroups[w1] = qgroups[w3] = "moe.gu"
        qgroups[w2] = "moe.d"
        norm_before[w1] = "ffn_norm"
        shared_with[w3] = w1
        tap_of[w1], tap_of[w2] = f"{tag}.mlp_in", f"{tag}.mlp_mid"
    dense_gate = ("feed_forward.w1", ["feed_forward.w3"])
    qkv_gate = ("self_attn.q_proj", ["self_attn.k_proj", "self_attn.v_proj"])
    return ArchSpec(
        name="lfm2_moe", model_types=("lfm2_moe",), converter_arch="lfm2_moe",
        converter_path="src/convert/lfm2_layout.cpp", norm_offset=0.0,
        lm_head_name="model.embed_tokens.weight",   # tied: the head is the embedding table
        has_mtp=False, linears=linears, qgroups=qgroups,
        modules_by_layer_type={
            "conv_dense": _LFM2_CONV + _LFM2_DENSE, "conv_moe": _LFM2_CONV + experts,
            "full_attention_dense": _LFM2_ATTN + _LFM2_DENSE, "full_attention_moe": _LFM2_ATTN + experts},
        norm_before=norm_before, shared_with=shared_with, tap_of=tap_of, mtp_tap_of={},
        shared_gate={"conv_dense": [dense_gate], "conv_moe": [],
                     "full_attention_dense": [qkv_gate, dense_gate], "full_attention_moe": [qkv_gate]},
        rms_norms=(), rms_key_suffixes={},
        attention_layer_types=("full_attention_dense", "full_attention_moe"),
        layer_prefix=LFM2_PREFIX, n_experts=n_experts)


LFM2 = lfm2_arch(64)

ARCHES = {QWEN.name: QWEN, GEMMA.name: GEMMA, LFM2.name: LFM2}
ARCH_NAMES = tuple(ARCHES)


def get_arch(name: str | None) -> ArchSpec:
    """The spec of `name` (None = qwen3_5, the historical default). lfm2_moe: the 64-expert spec (resolve_arch
    with a model dir sizes it by the checkpoint's config)."""
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
    text_config's); LFM2 (sized by the config's num_experts); anything else that is not Gemma 4 is the Qwen3.5
    default, as before this table existed."""
    cfg = read_config(model_dir)
    for mt in (cfg.get("model_type"), (cfg.get("text_config") or {}).get("model_type")):
        if mt in GEMMA.model_types:
            return GEMMA
        if mt in LFM2.model_types:
            return lfm2_arch(int(cfg.get("num_experts", 64)))
    return QWEN


def resolve_arch(requested: str | None, model_dir: Path | str | None) -> ArchSpec:
    """`--arch` of the tools: 'auto'/None = detect from model_dir (Qwen when there is none)."""
    has_cfg = model_dir is not None and (Path(model_dir) / "config.json").is_file()
    if requested in (None, "auto"):
        return detect_arch(model_dir) if has_cfg else QWEN
    spec = get_arch(requested)
    if spec.is_lfm2 and has_cfg:   # the per-expert tables follow the checkpoint's expert count
        return lfm2_arch(int(read_config(model_dir).get("num_experts", 64)))
    return spec


def text_config_view(arch: ArchSpec, model_dir: Path | str):
    """What enumerate_quantized_linears needs of a text config (`layer_types`, `num_hidden_layers`,
    `hidden_size`): the transformers config for Qwen (common.load_text_config), a plain namespace read
    from config.json for Gemma and LFM2 (no transformers, so the CPU dry-run needs only numpy).
    LFM2: `layer_types` are the HF ones (conv / full_attention); `layer_kinds` the table keys."""
    if arch.is_lfm2:
        t = read_config(model_dir)
        n = int(t["num_hidden_layers"])
        if len(t["layer_types"]) != n:
            raise ValueError(f"layer_types has {len(t['layer_types'])} entries for {n} layers")
        return SimpleNamespace(
            layer_types=list(t["layer_types"]), layer_kinds=arch.layer_kinds(t), num_hidden_layers=n,
            hidden_size=int(t["hidden_size"]), intermediate_size=int(t["intermediate_size"]),
            moe_intermediate_size=int(t["moe_intermediate_size"]), vocab_size=int(t["vocab_size"]),
            num_attention_heads=int(t["num_attention_heads"]), num_key_value_heads=int(t["num_key_value_heads"]),
            head_dim=int(t.get("head_dim") or t["hidden_size"] // t["num_attention_heads"]),
            num_dense_layers=int(t.get("num_dense_layers", 2)), num_experts=int(t["num_experts"]),
            num_experts_per_tok=int(t["num_experts_per_tok"]), norm_eps=float(t.get("norm_eps", 1e-5)),
            conv_L_cache=int(t.get("conv_L_cache", 3)))
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
    return f"{arch.layer_prefix}{layer}.{norm}.weight"


RMS_POST_FILE_RE = re.compile(r"L(\d{2,})\.(in|mlp_in)\.hess")
