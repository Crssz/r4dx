"""Inkling2Reference: the text stack of an Inkling-2-Mini (`inkling2_loop`) checkpoint, walked LAYER-STREAMED in
visit order over the HF `modeling_inkling2_loop` modules (docs/inkling2-plan.md 3.3, 7), plus the dump CLI of the
fp32-truth / HF-bf16 KL gate.

    ref = Inkling2Reference(model_dir, dtype=torch.float32)        # weights are read one stored layer at a time
    hidden = ref.run({"seg": ids})["seg"]                           # [T, 2304] after the final AttnRes + norm
    logits = ref.head_rows(hidden)                                  # [T, 200058] fp32, softcapped

What it computes: exactly `Inkling2LoopForCausalLM.forward` (no cache, whole segment per layer visit), but with the
48 layer VISITS (layers 0-5, loop 6-17 x3 with cross-pass messages, 18-23) executed one at a time over all segments in
lockstep: the stored layer is read from the safetensors, its experts are dequantized with HF's own op
(`Inkling2LoopDequantizeExperts`, then the `Interleave(dim=1)` of the conversion mapping), the HF decoder-layer
module is built on the meta device and loaded with `assign=True`, run for every segment, and freed.  Per visit the
state of every segment (hidden, block states, the pre-loop state, the previous-pass messages' inputs) is
checkpointed for resumption.  `selftest_tiny` / `test_ref_tiny.py` prove it equal to HF's own forward on a tiny
random checkpoint in the REAL on-disk format (NVFP4 and MXFP8 experts, names exactly as in the real header).

Numerics (`dtype`):
  torch.float32   TRUTH: every weight upcast exactly (experts dequantized straight to fp32), fp32 activations, eager
                  attention, fp32 router / sigmoid / top-k, fp32 head.
  torch.bfloat16  stock HF: bf16 weights and activations (experts dequantized to bf16 as HF does at load), the
                  parameters HF keeps in fp32 (`_keep_in_fp32_modules_strict`: router weight/bias, short convs,
                  AttnRes queries, message-gate forget/gain, ...) stay fp32, eager attention (this model has no sdpa),
                  bf16 head matmul widened to fp32 before the softcap.

CLI (the dump of the gate; kl_report.py's shared format `<seg>.logprobs.f16` rows 0..T-2 [T-1, 200058] fp16 + `.meta.json`):
see `main` / the README block of docs.  torch/transformers are imported lazily where cheap.
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
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inkling2.common_inkling2 import (  # noqa: E402
    BEGIN_OF_TEXT_ID,
    DEFAULT_MODEL_DIR,
    LOGPROB_CLAMP,
    REAL_VOCAB,
    peak_rss_gb,
    sha256_file,
    token_ids_sha256,
)

# --------------------------------------------------------------------------------------------
# Checkpoint access
# --------------------------------------------------------------------------------------------


class CheckpointSource:
    """Original-format tensors (`model.llm.*`) of a safetensors directory (index.json or *.safetensors headers).
    Reads with plain seek+readinto, NEVER a safetensors mmap (common.raw_safetensors_read: the Windows commit-limit
    crash of whole-file maps)."""

    def __init__(self, model_dir):
        from common import raw_safetensors_header

        self.model_dir = Path(model_dir)
        self._hdr: dict[str, tuple] = {}
        self.where: dict[str, str] = {}
        files = sorted(p.name for p in self.model_dir.glob("*.safetensors"))
        if not files:
            raise RuntimeError(f"no *.safetensors under {self.model_dir}")
        for f in files:
            hdr = raw_safetensors_header(self.model_dir / f)
            self._hdr[f] = hdr
            for k in hdr[0]:
                self.where[k] = f

    def has(self, name: str) -> bool:
        return name in self.where

    def meta(self, name: str) -> dict:
        return self._hdr[self.where[name]][0][name]

    def names(self):
        return self.where.keys()

    def raw(self, name: str):
        from common import raw_safetensors_read

        f = self.where[name]
        return raw_safetensors_read(self.model_dir / f, name, self._hdr[f])


# --------------------------------------------------------------------------------------------
# The tensor table: checkpoint name <-> HF parameter name (the explicit, tested rename table)
# --------------------------------------------------------------------------------------------
#
# Derived from conversion_mapping.py (`inkling2_loop_text` + `inkling2_text`) and checked three ways:
#   * test_ref_tiny: HF's OWN loader (`from_pretrained` on a tiny checkpoint in these names) vs this table,
#   * `audit_against_header`: every name below exists in the real header with the shape the config implies and every
#     non-visual tensor of the real header is consumed,
#   * `Interleave(dim=1)` / `Inkling2LoopDequantizeExperts` / `Inkling2LoopMergeMessageQueries` are HF's own ops.

LAYER_PREFIX = "model.llm.layers.{L}."
GLOBAL_NAMES = ("model.llm.embed.weight", "model.llm.embed_norm.weight", "model.llm.norm.weight",
                "model.llm.unembed.weight", "model.llm.attn_res.output_query",
                "model.llm.middle_cycle_cross_pass_queries", "model.llm.middle_cycle_cross_pass_kv_queries")
EXPERT_PARTS = ("w13_weight", "w13_weight_scale", "w13_weight_amax", "w13_weight_outer_scale_denominator",
                "w2_weight", "w2_weight_scale", "w2_weight_amax", "w2_weight_outer_scale_denominator")


def layer_kind(cfg, L: int) -> tuple[bool, bool]:
    """(is_sliding, is_loop_layer)"""
    return (cfg.layer_types[L] == "hybrid_sliding",
            cfg.loop_layer_index_start <= L < cfg.loop_layer_index_end)


def layer_table(cfg, L: int) -> list[tuple[str, str, tuple]]:
    """[(checkpoint name, HF name relative to `model.layers.{L}.`, shape)] of every NON-expert tensor of stored layer
    L that lives under `model.llm.layers.{L}.` (the two AttnRes queries are in `query_table`)."""
    sliding, loop = layer_kind(cfg, L)
    H, K = cfg.hidden_size, cfg.conv_kernel_size
    nh = cfg.swa_num_attention_heads if sliding else cfg.num_attention_heads
    nkv = cfg.swa_num_key_value_heads if sliding else cfg.num_key_value_heads
    hd = cfg.swa_head_dim if sliding else cfg.head_dim
    rel_extent = cfg.sliding_window_size if sliding else cfg.rel_extent
    t = [
        ("attn.wq_du.weight", "self_attn.q_proj.linear.weight", (nh * hd, H)),
        ("attn.wk_dv.weight", "self_attn.k_proj.linear.weight", (nkv * hd, H)),
        ("attn.wv_dv.weight", "self_attn.v_proj.linear.weight", (nkv * hd, H)),
        ("attn.gate_attn.weight", "self_attn.o_proj.gate_proj.weight", (nh * hd, H)),
        ("attn.wo_ud.weight", "self_attn.o_proj.linear.weight", (H, nh * hd)),
        ("attn.wr_du.weight", "self_attn.relative_position_bias.linear.weight", (nh * cfg.d_rel, H)),
        ("attn.attn_sink_proj.weight", "self_attn.sink_proj.weight", (nh, H)),
        ("attn.q_l2_clip.scale", "self_attn.q_proj.norm.scale", (nh, hd)),
        ("attn.rel_l2_clip.scale", "self_attn.relative_position_bias.norm.scale", (nh, cfg.d_rel)),
        ("attn.rel_logits_proj.proj", "self_attn.relative_position_bias.proj", (cfg.d_rel, rel_extent)),
        ("mlp_norm.weight", "post_attention_layernorm.weight", (H,)),
        ("attn_sconv.weight", "post_attention_convolution.conv1d.weight", (H, 1, K)),
        ("mlp_sconv.weight", "post_mlp_convolution.conv1d.weight", (H, 1, K)),
        ("mlp.gate.weight", "mlp.gate.weight", (cfg.n_routed_experts, H)),
        ("mlp.gate.bias", "mlp.gate.e_score_correction_bias", (cfg.n_routed_experts,)),
    ]
    for kv in ("k", "v"):
        if sliding:
            t.append((f"attn.{kv}_sconv.weight", f"self_attn.{kv}_proj.sconv.conv1d.weight", (nkv * hd, 1, K)))
        else:
            t.append((f"attn.{kv}_sconv.static_w", f"self_attn.{kv}_proj.sconv.static_weight", (K, nkv * hd)))
            t.append((f"attn.{kv}_sconv.weight_proj", f"self_attn.{kv}_proj.sconv.weight_proj.weight", (nkv * K, H)))
    if sliding:
        t.append(("attn_norm.weight", "input_layernorm.weight", (H,)))
    else:
        for q in ("q", "k", "v"):
            t.append((f"{q}_input_norm.weight", f"{q}_input_layernorm.weight", (H,)))
    if loop:
        r = cfg.loop_gate_rank
        for ck, hf in (("attn", "q"), ("k", "k"), ("v", "v"), ("mlp", "mlp")):
            t.append((f"controllers.{ck}.down_weight", f"{hf}_message_gate.down_proj.weight", (r, 2 * H)))
            t.append((f"controllers.{ck}.forget_weight", f"{hf}_message_gate.forget_gate_proj.weight", (H, r)))
            t.append((f"controllers.{ck}.gain_weight", f"{hf}_message_gate.message_gate_proj.weight", (H, r)))
    return t


def query_table(cfg, L: int) -> list[tuple[str, str, tuple]]:
    """The AttnRes queries of layer L: `attn_queries.{L-1}` (L >= 1; [3, H] for a full-attention layer) and
    `mlp_queries.{L}`, both F32 on disk."""
    sliding, _ = layer_kind(cfg, L)
    H = cfg.hidden_size
    out = []
    if L >= 1:
        out.append((f"model.llm.attn_res.attn_queries.{L - 1}", "attention_input.query", (H,) if sliding else (3, H)))
    out.append((f"model.llm.attn_res.mlp_queries.{L}", "mlp_input.query", (H,)))
    return out


def expert_shapes(cfg) -> dict:
    """{part: (shape for NVFP4 layers, shape for MXFP8 layers)} of the expert tensors of one layer."""
    E, H, I = cfg.n_routed_experts, cfg.hidden_size, cfg.moe_intermediate_size
    return {
        "w13_weight": ((E, 2 * I, H // 2), (E, 2 * I, H)), "w13_weight_scale": ((E, 2 * I, H // 16), (E, 2 * I, H // 32)),
        "w13_weight_amax": ((E,), None), "w13_weight_outer_scale_denominator": ((1,), None),
        "w2_weight": ((E, H, I // 2), (E, H, I)), "w2_weight_scale": ((E, H, I // 16), (E, H, I // 32)),
        "w2_weight_amax": ((E,), None), "w2_weight_outer_scale_denominator": ((1,), None),
    }


def audit_against_header(cfg, names_shapes: dict) -> list[str]:
    """Problems of the table vs a header `{name: (dtype, shape)}`: a table name that is missing / has another shape,
    an expert format that is neither NVFP4 nor MXFP8, and a header name (other than `model.visual.*`) that no table
    entry consumes."""
    problems, consumed = [], set()
    shp = expert_shapes(cfg)
    for name in GLOBAL_NAMES:
        consumed.add(name)
        if name not in names_shapes:
            problems.append(f"missing {name}")
    for L in range(cfg.num_hidden_layers):
        p = LAYER_PREFIX.format(L=L)
        for ck, _, shape in layer_table(cfg, L):
            consumed.add(p + ck)
            got = names_shapes.get(p + ck)
            if got is None:
                problems.append(f"missing {p + ck}")
            elif tuple(got[1]) != tuple(shape):
                problems.append(f"{p + ck}: shape {tuple(got[1])} != {tuple(shape)}")
        for ck, _, shape in query_table(cfg, L):
            consumed.add(ck)
            got = names_shapes.get(ck)
            if got is None:
                problems.append(f"missing {ck}")
            elif tuple(got[1]) != tuple(shape):
                problems.append(f"{ck}: shape {tuple(got[1])} != {tuple(shape)}")
        w13 = names_shapes.get(p + "mlp.experts.w13_weight")
        if w13 is None:
            problems.append(f"missing {p}mlp.experts.w13_weight")
            continue
        nvfp4 = w13[0] == "U8"
        want_dtype = ("U8", "F8_E4M3") if nvfp4 else ("F8_E4M3", "U8")
        for part in EXPERT_PARTS:
            n = p + "mlp.experts." + part
            fmt = shp[part][0 if nvfp4 else 1]
            if fmt is None:
                if n in names_shapes:
                    problems.append(f"unexpected {n} in an MXFP8 layer")
                continue
            consumed.add(n)
            got = names_shapes.get(n)
            if got is None:
                problems.append(f"missing {n}")
            elif tuple(got[1]) != tuple(fmt):
                problems.append(f"{n}: shape {tuple(got[1])} != {tuple(fmt)}")
            if part.endswith("_weight") and got is not None and got[0] != want_dtype[0]:
                problems.append(f"{n}: dtype {got[0]} (layer is {'NVFP4' if nvfp4 else 'MXFP8'})")
            if part.endswith("_weight_scale") and got is not None and got[0] != want_dtype[1]:
                problems.append(f"{n}: dtype {got[0]}")
    for n in names_shapes:
        if n.startswith("model.visual.") or n in consumed:
            continue
        problems.append(f"unconsumed checkpoint tensor {n}")
    return problems


def param_dtype(full_name: str, dtype, keep_fp32) -> "object":
    """The dtype HF gives parameter `full_name` (a `model.*` path) under `from_pretrained(dtype=dtype)`."""
    import torch

    if dtype == torch.float32:
        return torch.float32
    return torch.float32 if any(k in full_name for k in keep_fp32) else dtype


# --------------------------------------------------------------------------------------------
# The walker
# --------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Visit:
    idx: int
    layer: int
    phase: str            # "pre" | "loop" | "post"
    pass_idx: int         # loop pass 0..P-1 (0 for pre/post)
    loop_j: int           # position inside the loop (-1 outside)


def visit_table(cfg) -> list[Visit]:
    a, b, P = cfg.loop_layer_index_start, cfg.loop_layer_index_end, cfg.loop_num_passes
    out: list[Visit] = []
    for L in range(a):
        out.append(Visit(len(out), L, "pre", 0, -1))
    for p in range(P):
        for j, L in enumerate(range(a, b)):
            out.append(Visit(len(out), L, "loop", p, j))
    for L in range(b, cfg.num_hidden_layers):
        out.append(Visit(len(out), L, "post", 0, -1))
    return out


class SegState:
    """Per-segment walker state (all tensors in the work dtype except nothing: everything the HF model keeps)."""

    TENSORS = ("h", "B", "h_pre", "B_pre", "prev_blocks", "prev_h")

    def __init__(self, ids, h):
        self.ids = ids
        self.h, self.B = h, h.new_zeros(*h.shape[:-1], 0, h.shape[-1])
        self.h_pre = self.B_pre = self.prev_blocks = self.prev_h = None
        self.n_pre = 0
        self.masks = self.rel = self.conv_mask = self.pos = None

    def dump(self) -> dict:
        d = {k: getattr(self, k) for k in self.TENSORS}
        d["n_pre"] = self.n_pre
        return d

    def load(self, d: dict) -> None:
        for k in self.TENSORS:
            setattr(self, k, d[k])
        self.n_pre = d["n_pre"]


class Inkling2Reference:
    source_tag = "reference"

    def __init__(self, model_dir=None, dtype=None, config=None, source: CheckpointSource | None = None):
        import torch
        import transformers.models.inkling2_loop.modeling_inkling2_loop as modeling
        from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

        self.m = modeling
        self.torch = torch
        self.model_dir = Path(model_dir) if model_dir is not None else DEFAULT_MODEL_DIR
        self.dtype = torch.bfloat16 if dtype is None else dtype
        self.truth = self.dtype == torch.float32
        self.src = source or CheckpointSource(self.model_dir)
        if config is None:
            raw = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
            tc = dict(raw.get("text_config", raw))
            tc.pop("model_type", None)
            config = Inkling2LoopTextConfig(**tc)
        self.config = config
        self.config._attn_implementation = "eager"
        # bf16 = stock HF: the default experts implementation `from_pretrained` resolves (grouped_mm when the CPU
        # build can dispatch it, else eager); truth = the per-expert eager loop (fp32 reference, no fused kernel)
        self.stock_experts_impl = self._resolve_experts_impl()
        self.experts_impl = "eager" if self.truth else self.stock_experts_impl
        self.config._experts_implementation = self.experts_impl
        self.keep_fp32 = tuple(modeling.Inkling2LoopPreTrainedModel._keep_in_fp32_modules_strict)
        self.visits = visit_table(self.config)
        self.n_layers = self.config.num_hidden_layers
        self.router_ids: dict[str, "torch.Tensor"] = {}
        self._cur_seg = None
        self._load_globals()

    # -- setup ------------------------------------------------------------------------------------

    def _resolve_experts_impl(self) -> str:
        """What `from_pretrained` would pick (None -> grouped_mm when the CPU build can dispatch it, else eager)."""
        import copy

        torch = self.torch
        cfg = copy.deepcopy(self.config)
        cfg._experts_implementation = None
        with torch.device("meta"):
            probe = self.m.Inkling2LoopForCausalLM(cfg)
        return str(probe.config._experts_implementation)

    def raw(self, name: str):
        return self.src.raw(name)

    def _cast(self, t, full_name: str):
        return t.to(param_dtype(full_name, self.dtype, self.keep_fp32))

    def _load_globals(self) -> None:
        torch, m, cfg = self.torch, self.m, self.config
        H = cfg.hidden_size
        self.embed_table = self.raw("model.llm.embed.weight").to(torch.bfloat16)      # [201024, H] bf16 as stored
        if tuple(self.embed_table.shape) != (cfg.vocab_size, H):
            raise RuntimeError(f"embed shape {tuple(self.embed_table.shape)} != ({cfg.vocab_size}, {H})")
        with torch.device("meta"):
            self.embed_norm = m.Inkling2LoopRMSNorm(H, eps=cfg.rms_norm_eps)
            self.final_norm = m.Inkling2LoopRMSNorm(H, eps=cfg.rms_norm_eps)
            self.final_input = m.Inkling2LoopBlockAttentionResidual(cfg)
            self.message_input = m.Inkling2LoopMessageAttentionResidual(cfg)
        self.embed_norm.load_state_dict(
            {"weight": self._cast(self.raw("model.llm.embed_norm.weight"), "model.embed_tokens.embed_norm.weight")},
            assign=True)
        self.final_norm.load_state_dict(
            {"weight": self._cast(self.raw("model.llm.norm.weight"), "model.norm.weight")}, assign=True)
        self.final_input.load_state_dict(
            {"query": self._cast(self.raw("model.llm.attn_res.output_query"), "model.final_input.query")}, assign=True)
        from transformers.core_model_loading import Inkling2LoopMergeMessageQueries

        q = Inkling2LoopMergeMessageQueries().convert(
            {"q": self.raw("model.llm.middle_cycle_cross_pass_queries"),
             "kv": self.raw("model.llm.middle_cycle_cross_pass_kv_queries")},
            source_patterns=["q", "kv"], target_patterns=["out"])["out"]
        self.message_input.load_state_dict({"query": self._cast(q, "model.message_input.query")}, assign=True)
        for mod in (self.embed_norm, self.final_norm, self.final_input, self.message_input):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad_(False)
        unembed = self.raw("model.llm.unembed.weight")[: cfg.unpadded_vocab_size].to(torch.bfloat16)
        self.lm_head_w = unembed.float() if self.truth else unembed                    # [200058, H]
        self.vocab = int(self.lm_head_w.shape[0])

    # -- weights ----------------------------------------------------------------------------------

    def experts(self, L: int, which: str):
        """Dequantized `gate_up_proj` ([E, 2I, H], gate rows then up rows) or `down_proj` ([E, H, I]) of layer L in
        the work dtype, with HF's own ops."""
        from transformers.core_model_loading import Inkling2LoopDequantizeExperts, Interleave

        p = LAYER_PREFIX.format(L=L) + "mlp.experts."
        part = "w13" if which == "gate_up_proj" else "w2"
        keys = [f"{part}_weight", f"{part}_weight_scale", f"{part}_weight_amax", f"{part}_weight_outer_scale_denominator"]
        inp = {k: self.raw(p + k) for k in keys if self.src.has(p + k)}
        if f"{part}_weight_scale" not in inp:
            raise RuntimeError(f"layer {L}: {part} is neither NVFP4 nor MXFP8 (no scale tensor)")
        dt = self.dtype
        if dt != self.torch.float32:
            # stock HF's loader casts EVERY source of a converter to the parameter dtype before the op runs: the codes
            # and e4m3/E8M0 scales survive exactly, but amax and the outer denominator (F32) are rounded to bf16, which
            # changes every NVFP4 expert weight by up to 2^-9 relative. The bf16 variant reproduces that.
            inp = {k: v.to(dt) for k, v in inp.items()}
        stub = types.SimpleNamespace(get_parameter=lambda name: types.SimpleNamespace(dtype=dt))
        out = Inkling2LoopDequantizeExperts().convert(inp, source_patterns=keys, target_patterns=[which],
                                                      model=stub, full_layer_name=which)[which]
        del inp
        if part == "w13":
            out = Interleave(dim=1).convert({"x": out}, source_patterns=["x"], target_patterns=["y"])["y"]
        return out

    def layer_state(self, L: int, with_experts: bool = True) -> dict:
        """HF state_dict (names relative to the layer module) of stored layer L in the work dtype rules
        (`with_experts=False`: without the two big dequantized expert tensors, greedy_smoke's lazy path)."""
        p = LAYER_PREFIX.format(L=L)
        base = f"model.layers.{L}."
        st = {}
        for ck, hf, shape in layer_table(self.config, L) + [(q, h, s) for q, h, s in query_table(self.config, L)]:
            t = self.raw(ck if ck.startswith("model.") else p + ck)
            if tuple(t.shape) != tuple(shape):
                raise RuntimeError(f"{ck}: shape {tuple(t.shape)} != {tuple(shape)}")
            st[hf] = self._cast(t, base + hf)
        if with_experts:
            st["mlp.experts.gate_up_proj"] = self.experts(L, "gate_up_proj")
            st["mlp.experts.down_proj"] = self.experts(L, "down_proj")
        return st

    def build_layer(self, L: int):
        torch, m, cfg = self.torch, self.m, self.config
        sliding, loop = layer_kind(cfg, L)
        cls = {(False, False): m.Inkling2LoopFullDecoderLayer, (True, False): m.Inkling2LoopSlidingDecoderLayer,
               (False, True): m.Inkling2LoopFullLoopDecoderLayer, (True, True): m.Inkling2LoopSlidingLoopDecoderLayer}[(sliding, loop)]
        with torch.device("meta"):
            layer = cls(cfg, L)
        state = self.layer_state(L)
        layer.load_state_dict(state, strict=True, assign=True)
        if any(t.is_meta for t in list(layer.parameters()) + list(layer.buffers())):
            raise RuntimeError(f"layer {L}: parameters left on the meta device")
        layer.eval()
        for prm in layer.parameters():
            prm.requires_grad_(False)
        layer.mlp.gate.register_forward_hook(self._router_hook(L))
        return layer

    def _router_hook(self, L: int):
        def hook(_mod, _inp, out):
            if self._cur_seg is not None:
                self.router_ids[f"V{self._visit_idx:02d}__{self._cur_seg}"] = out[2].detach().to(self.torch.int16).cpu()
        return hook

    # -- forward ----------------------------------------------------------------------------------

    def embed(self, ids):
        torch = self.torch
        rows = self.embed_table[torch.as_tensor([int(t) for t in ids], dtype=torch.long)]
        return self.embed_norm(rows.to(self.dtype)).unsqueeze(0)

    def _prepare(self, S: SegState) -> None:
        from transformers.masking_utils import (
            create_causal_mask,
            create_recurrent_attention_mask,
            create_sliding_window_causal_mask,
        )

        torch = self.torch
        T = S.h.shape[1]
        S.pos = (torch.arange(T) + 0).unsqueeze(0)
        kw = {"config": self.config, "inputs_embeds": S.h, "attention_mask": None, "cache": None, "position_ids": S.pos}
        S.masks = {"full_attention": create_causal_mask(**kw), "sliding_attention": create_sliding_window_causal_mask(**kw)}
        S.conv_mask = create_recurrent_attention_mask(**kw)
        S.rel = {"full_attention": self.m.get_relative_distance(S.h, None, is_sliding=False),
                 "sliding_attention": self.m.get_relative_distance(S.h, None, is_sliding=True)}

    def step(self, layer, v: Visit, S: SegState) -> None:
        """One layer visit for one segment: HF `Inkling2LoopTextModel.forward`'s loop body, verbatim."""
        torch = self.torch
        kw = dict(attention_mask=S.masks[layer.attention_type], relative_distance=S.rel[layer.attention_type],
                  conv_mask=S.conv_mask, cache=None, position_ids=S.pos)
        if v.phase == "loop":
            messages = None
            if v.pass_idx >= 1:
                if v.loop_j == 0:
                    # the pass restarts from the pre-loop state and receives the previous pass's last block + output
                    S.prev_blocks = S.B[..., S.n_pre + 1:, :]
                    S.prev_h = S.h
                    S.h, S.B = S.h_pre, S.B_pre
                messages = self.message_input(S.prev_blocks, S.prev_h, v.loop_j)
            new_h = layer(S.h, S.B, messages, **kw)
        else:
            new_h = layer(S.h, S.B, **kw)
        if layer.is_block_boundary:
            S.B = torch.cat([S.B, S.h.unsqueeze(-2)], dim=-2)
        S.h = new_h
        if v.phase == "pre" and v.layer == self.config.loop_layer_index_start - 1:
            S.h_pre, S.B_pre, S.n_pre = S.h, S.B, S.B.shape[-2]

    def final_hidden(self, S: SegState):
        """[T, H] after the final AttnRes and norm (the `last_hidden_state` of the HF text model)."""
        h = self.final_input(S.B, S.h)[0]
        return self.final_norm(h).to(self.dtype)[0]

    def run(self, seqs: dict, ckpt: Path | None = None, debug_max_visits: int | None = None, log=print,
            record=None) -> dict:
        """Teacher-forced walk of every segment through all 48 visits in lockstep. `seqs` {name: token ids}.
        Returns {name: final hidden [T, H]} (None entries when `debug_max_visits` stops early)."""
        torch = self.torch
        names = list(seqs)
        visits = self.visits if debug_max_visits is None else self.visits[:debug_max_visits]
        t_start = time.perf_counter()
        with torch.no_grad():
            states = {}
            for n in names:
                ids = seqs[n]
                states[n] = SegState(ids, self.embed(ids))
                self._prepare(states[n])
            start = 0
            if ckpt is not None and ckpt.exists():
                c = torch.load(ckpt, weights_only=True)
                if c["names"] == names and c["dtype"] == str(self.dtype) and c["lens"] == [len(seqs[n]) for n in names] \
                        and c["n_visits"] == len(self.visits):
                    start = c["next_visit"]
                    for n in names:
                        states[n].load(c["states"][n])
                    self.router_ids = c["router"]
                    log(f"[ref] resuming at visit {start}")
            for v in visits[start:]:
                t0 = time.perf_counter()
                layer = self.build_layer(v.layer)
                t_load = time.perf_counter() - t0
                self._visit_idx = v.idx
                for n in names:
                    self._cur_seg = n
                    self.step(layer, v, states[n])
                self._cur_seg = None
                del layer
                hmax = max(float(states[n].h.abs().max()) for n in names)
                log(f"[ref] visit {v.idx:02d}/{len(self.visits)} L{v.layer:02d} {v.phase}"
                    f"{'' if v.phase != 'loop' else f' pass{v.pass_idx}'} load {t_load:.1f}s total "
                    f"{time.perf_counter() - t0:.1f}s max|h|={hmax:.1f} elapsed={(time.perf_counter() - t_start) / 60:.1f}min "
                    f"peakRSS={peak_rss_gb() or 0:.2f}GiB")
                if record is not None:
                    record(v, states)
                if ckpt is not None:
                    ckpt.parent.mkdir(parents=True, exist_ok=True)
                    torch.save({"names": names, "dtype": str(self.dtype), "lens": [len(seqs[n]) for n in names],
                                "n_visits": len(self.visits), "next_visit": v.idx + 1,
                                "states": {n: states[n].dump() for n in names}, "router": self.router_ids},
                               str(ckpt) + ".tmp")
                    os.replace(str(ckpt) + ".tmp", ckpt)
            if len(visits) < len(self.visits):
                return {n: None for n in names}
            return {n: self.final_hidden(states[n]) for n in names}

    # -- head -------------------------------------------------------------------------------------

    def head_rows(self, hidden):
        """Softcapped fp32 logits [rows, 200058]: truth = fp32 matmul over the widened bf16 table; bf16 = the bf16
        matmul output of HF's lm_head widened to fp32; then `tanh(x / cap) * cap` in fp32 (HF `logit_softcap`)."""
        torch = self.torch
        cap = self.config.final_logit_softcapping
        if self.truth:
            lg = hidden.float() @ self.lm_head_w.T
        else:
            lg = torch.nn.functional.linear(hidden.to(torch.bfloat16), self.lm_head_w).float()
        return torch.tanh(lg / cap) * cap


def log_softmax_clamped(logits):
    import torch

    return torch.log_softmax(logits, dim=-1).clamp_min(LOGPROB_CLAMP)


# --------------------------------------------------------------------------------------------
# Tiny random-config checkpoint in the REAL on-disk format (the CPU smoke path of every script here)
# --------------------------------------------------------------------------------------------


def tiny_config(layers: int = 12, vocab: int = 160, hidden: int = 64, experts: int = 8, top_k: int = 2):
    """Loop 3..9 (6 loop layers, 3 passes, AttnRes block size 3), full-attention layers at (i + 1) % 3 == 0 (a full
    pre layer, two full loop layers, a full post layer), small clip thresholds / log-scaling floor / windows so every
    clip, the log scale and the sliding mask are active on ~20 tokens."""
    from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig

    assert layers % 3 == 0 and layers >= 12
    return Inkling2LoopTextConfig(
        vocab_size=vocab, unpadded_vocab_size=vocab - 10, hidden_size=hidden, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, swa_num_attention_heads=4,
        swa_num_key_value_heads=2, swa_head_dim=16, sliding_window_size=7, d_rel=4, rel_extent=9,
        log_scaling_n_floor=4, log_scaling_alpha=0.3,
        layer_types=["hybrid" if (i + 1) % 3 == 0 else "hybrid_sliding" for i in range(layers)],
        moe_intermediate_size=32, n_routed_experts=experts, num_experts_per_tok=top_k, route_scale=2.5,
        qk_l2_clip_threshold=2.0, rel_l2_clip_threshold=1.0, n_layers_per_attn_res_block=3,
        loop_layer_index_start=3, loop_layer_index_end=9, loop_num_passes=3, loop_gate_rank=8,
        mlp_gate_softcap=3.0, mlp_up_softcap=5.0, final_logit_softcapping=30.0, pad_token_id=0, eos_token_id=2,
        bos_token_id=1, max_position_embeddings=4096)


def build_tiny_checkpoint(out_dir, seed: int = 0, layers: int = 12, vocab: int = 160, cfg=None) -> Path:
    """Writes a tiny random inkling2_loop checkpoint in the real on-disk format: config.json (text config), TWO
    shards + index, names/dtypes/shapes of the real header (`layer_table`...), experts NVFP4 (U8 + F8_E4M3 scale +
    amax + outer denominator) in most layers and MXFP8 (F8_E4M3 + E8M0 U8) in layers 0, 5, 8, 11, padded
    embed/unembed (`vocab` rows, `unpadded_vocab_size` used)."""
    import torch
    from safetensors.torch import save_file

    cfg = cfg or tiny_config(layers, vocab)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    H, E, I = cfg.hidden_size, cfg.n_routed_experts, cfg.moe_intermediate_size

    def rn(*shape, std=0.1):
        return torch.randn(*shape, generator=g) * std

    def bf(t):
        return t.to(torch.bfloat16)

    disk: dict = {}
    disk["model.llm.embed.weight"] = bf(rn(vocab, H, std=0.5))
    disk["model.llm.embed_norm.weight"] = bf(1 + 0.05 * rn(H, std=1.0))
    disk["model.llm.norm.weight"] = bf(1 + 0.05 * rn(H, std=1.0))
    disk["model.llm.unembed.weight"] = bf(rn(vocab, H, std=0.3))
    disk["model.llm.attn_res.output_query"] = rn(H, std=0.8)
    disk["model.llm.middle_cycle_cross_pass_queries"] = rn(2 * (cfg.loop_layer_index_end - cfg.loop_layer_index_start), H, std=0.8)
    disk["model.llm.middle_cycle_cross_pass_kv_queries"] = rn(2 * (cfg.loop_layer_index_end - cfg.loop_layer_index_start), H, std=0.8)
    shp = expert_shapes(cfg)
    for L in range(cfg.num_hidden_layers):
        p = LAYER_PREFIX.format(L=L)
        for ck, hf, shape in layer_table(cfg, L):
            if ck.endswith("mlp.gate.bias"):
                t = (torch.rand(*shape, generator=g) - 0.5) * 0.6
            elif ck.endswith("mlp.gate.weight"):
                t = rn(*shape, std=0.5)
            elif "norm" in ck and ck.endswith("weight") and "sconv" not in ck and "controllers" not in ck:
                t = bf(1 + 0.05 * rn(*shape, std=1.0))
            elif ck.endswith("_l2_clip.scale"):
                t = bf(1 + 0.2 * rn(*shape, std=1.0))
            elif "sconv" in ck or ck.endswith("static_w"):
                t = bf(rn(*shape, std=0.4))
            elif ck.endswith("proj"):
                t = bf(rn(*shape, std=0.6))
            else:
                t = bf(rn(*shape, std=0.15))
            disk[p + ck] = t if (ck.endswith("mlp.gate.bias")) else (t if t.dtype == torch.bfloat16 else bf(t))
        for ck, _, shape in query_table(cfg, L):
            disk[ck] = rn(*shape, std=0.8)
        mx = L in (0, 5, 8, 11)
        ep = p + "mlp.experts."
        if mx:
            for part in ("w13", "w2"):
                w, s = shp[f"{part}_weight"][1], shp[f"{part}_weight_scale"][1]
                disk[ep + f"{part}_weight"] = rn(*w, std=0.1).to(torch.float8_e4m3fn)
                disk[ep + f"{part}_weight_scale"] = torch.randint(125, 130, s, generator=g, dtype=torch.uint8)
        else:
            for part in ("w13", "w2"):
                w, s = shp[f"{part}_weight"][0], shp[f"{part}_weight_scale"][0]
                disk[ep + f"{part}_weight"] = torch.randint(0, 256, w, generator=g, dtype=torch.uint8)
                disk[ep + f"{part}_weight_scale"] = (0.012 + 0.03 * torch.rand(*s, generator=g)).to(torch.float8_e4m3fn)
                disk[ep + f"{part}_weight_amax"] = 0.5 + torch.rand(E, generator=g)
                disk[ep + f"{part}_weight_outer_scale_denominator"] = torch.full((1,), 0.75)
    names = sorted(disk)
    half = len(names) // 2
    shards = {"model-00001-of-00002.safetensors": names[:half], "model-00002-of-00002.safetensors": names[half:]}
    wm = {}
    for fn, ns in shards.items():
        save_file({n: disk[n].contiguous() for n in ns}, str(out / fn), metadata={"format": "pt"})
        wm.update({n: fn for n in ns})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}), encoding="utf-8")
    cfg.save_pretrained(str(out))
    return out


def load_hf_model(model_dir, dtype, experts_implementation=None):
    """HF's own loader (conversion mapping + dequant op) on a text-config checkpoint directory."""
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM

    kw = {} if experts_implementation is None else {"experts_implementation": experts_implementation}
    m = Inkling2LoopForCausalLM.from_pretrained(str(model_dir), dtype=dtype, attn_implementation="eager", **kw)
    m.eval()
    return m


def selftest_tiny(tmp_dir, seed: int = 0, bf16: bool = True) -> dict:
    """CPU: the layer-streamed walker == `Inkling2LoopForCausalLM.forward` (HF loads the tiny real-format checkpoint
    with its own conversion mapping) in fp32 (max abs logits diff, expected ~1e-6) and in bf16; the walker's loaded
    layer states equal HF's parameters exactly. Returns the measured numbers; AssertionError on a mismatch."""
    import torch

    d = build_tiny_checkpoint(tmp_dir, seed)
    cfg = tiny_config()
    src = CheckpointSource(d)
    hdr = {n: (src.meta(n)["dtype"], src.meta(n)["shape"]) for n in src.names()}
    problems = audit_against_header(cfg, hdr)
    assert not problems, f"table vs tiny checkpoint: {problems[:5]}"
    g = torch.Generator().manual_seed(seed + 7)
    ids = [int(x) for x in torch.randint(3, cfg.unpadded_vocab_size, (23,), generator=g)]
    res = {"visits": len(visit_table(cfg)), "layer_types": cfg.layer_types}
    for dtype, tag in ((torch.float32, "fp32"),) + (((torch.bfloat16, "bf16"),) if bf16 else ()):
        # fp32 truth walks the per-expert eager loop: compared with HF forced to the same loop (and, as a number only,
        # with HF's default fused grouped_mm experts); bf16 is stock HF with its default experts implementation
        hf = load_hf_model(d, dtype, "eager" if dtype == torch.float32 else None)
        with torch.no_grad():
            want = hf(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float()
        ref = Inkling2Reference(d, dtype=dtype, config=tiny_config())
        res["experts_impl_stock"] = ref.stock_experts_impl
        if dtype != torch.float32:
            assert ref.stock_experts_impl == hf.config._experts_implementation, (ref.stock_experts_impl,
                                                                                  hf.config._experts_implementation)
        else:
            hf_def = load_hf_model(d, dtype)
            with torch.no_grad():
                res["fp32_max_abs_diff_vs_hf_default_experts"] = float(
                    (hf_def(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float() - want).abs().max())
            del hf_def
        # parameter equality (the rename table + dequant + interleave vs HF's loader)
        sd = hf.state_dict()
        for L in range(cfg.num_hidden_layers):
            for k, v in ref.layer_state(L).items():
                w = sd[f"model.layers.{L}.{k}"]
                assert v.dtype == w.dtype, (L, k, v.dtype, w.dtype)
                assert torch.equal(v, w), f"layer {L} {k}: loaded tensor differs from HF's"
        for k, mod in (("model.embed_tokens.embed_norm.weight", ref.embed_norm.weight), ("model.norm.weight", ref.final_norm.weight),
                       ("model.final_input.query", ref.final_input.query), ("model.message_input.query", ref.message_input.query)):
            assert torch.equal(mod, sd[k]) and mod.dtype == sd[k].dtype, k
        assert torch.equal(ref.embed_table, sd["model.embed_tokens.weight"].to(torch.bfloat16))
        assert torch.equal(ref.lm_head_w.to(sd["lm_head.weight"].dtype), sd["lm_head.weight"])
        h = ref.run({"a": ids})["a"]
        got = ref.head_rows(h)
        diff = float((got - want).abs().max())
        res[f"max_abs_diff_{tag}"] = diff
        res[f"max_abs_logit_{tag}"] = float(want.abs().max())
        res[f"{tag}_routing_rows"] = {k: tuple(v.shape) for k, v in list(ref.router_ids.items())[:2]}
        assert diff < (1e-5 if dtype == torch.float32 else 1e-3), f"{tag}: walker differs from HF forward by {diff}"
        del hf
    return res


# --------------------------------------------------------------------------------------------
# Dump driver (CLI)
# --------------------------------------------------------------------------------------------

VARIANTS = ("truth", "bf16")


def load_segments(paths, max_tokens, long_max_tokens, raw_names, only) -> list[dict]:
    """Segments of every tokens file. A chat file (score masks) is taken whole; a raw file contributes the segments in
    `raw_names` (default all), truncated to `max_tokens` (`long_max_tokens` for names ending in `_long`)."""
    segs, seen = [], set()
    for p in paths:
        doc = json.loads(Path(p).read_text(encoding="utf-8"))
        if doc.get("tokenizer_arch") != "inkling2":
            raise SystemExit(f"{p}: not an inkling2 tokens file (tokenizer_arch={doc.get('tokenizer_arch')!r})")
        for s in doc["segments"]:
            chat = "score_mask" in s
            if not chat and raw_names is not None and s["name"] not in raw_names:
                continue
            if only and s["name"] not in only:
                continue
            ids = [int(t) for t in s["token_ids"]]
            if not chat:
                cap = long_max_tokens if s["name"].endswith("_long") else max_tokens
                if cap:
                    ids = ids[:cap]
                if ids[0] != BEGIN_OF_TEXT_ID:
                    raise SystemExit(f"segment {s['name']}: a raw segment must start with <|begin_of_text|>")
            if s["name"] in seen:
                raise SystemExit(f"duplicate segment name {s['name']}")
            seen.add(s["name"])
            segs.append({"name": s["name"], "token_ids": ids, "chat": chat, "file": str(p)})
    if not segs:
        raise SystemExit("no segments selected")
    return segs


def write_dumps(ref: Inkling2Reference, hid: dict, segs: list[dict], out_dir: Path, meta_common: dict,
                row_block: int = 128) -> None:
    """Head over the final hidden states of every segment (rows 0..T-2), log_softmax over the 200058 columns, fp16."""
    import numpy as np
    import torch

    out_dir.mkdir(parents=True, exist_ok=True)
    for sg in segs:
        n, ids = sg["name"], sg["token_ids"]
        H = hid[n][:-1]
        r = H.shape[0]
        t0 = time.perf_counter()
        path = out_dir / f"{n}.logprobs.f16"
        nll, worst = 0.0, 0.0
        with open(path, "wb") as f:
            for s in range(0, r, row_block):
                e = min(s + row_block, r)
                lp = log_softmax_clamped(ref.head_rows(H[s:e]))
                worst = max(worst, torch.logsumexp(lp, dim=-1).abs().max().item())
                tgt = torch.as_tensor(ids[s + 1:e + 1])
                nll -= lp[torch.arange(e - s), tgt].sum().item()
                f.write(np.ascontiguousarray(lp.to(torch.float16).numpy()).tobytes())
        assert path.stat().st_size == r * ref.vocab * 2
        meta = {"T": len(ids), "V": ref.vocab, "real_vocab": ref.vocab, "dtype": "float16", "rows": r, "segment": n,
                "sha256_of_token_ids_json": token_ids_sha256(ids), "mean_nll_next_token": nll / r,
                "perplexity": float(np.exp(nll / r)), "max_abs_logsumexp_fp16": worst,
                "row_semantics": "row i = log_softmax over the 200058 lm_head columns (after the logit softcap 30) at "
                                 "position i = log p(next | ids[0..i])",
                "logprob_clamp": LOGPROB_CLAMP, **meta_common}
        (out_dir / f"{n}.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[ref] {n}: T={len(ids)} nll={nll / r:.4f} ppl={meta['perplexity']:.2f} "
              f"({time.perf_counter() - t0:.0f}s) -> {path.name}", flush=True)


def router_agreement(truth_npz: Path, noise_npz: Path) -> dict:
    """Per layer VISIT: fraction of token rows whose SET of top-k experts is identical, truth vs the noise side."""
    import numpy as np

    a, b = np.load(truth_npz), np.load(noise_npz)
    per: dict[int, list[int]] = {}
    k_top = None
    for k in a.files:
        if k not in b.files:
            continue
        visit = int(k.split("__")[0][1:])
        x, y = np.sort(a[k], axis=1), np.sort(b[k], axis=1)
        k_top = a[k].shape[1]
        c = per.setdefault(visit, [0, 0, 0])
        c[0] += int((x == y).all(axis=1).sum())
        c[1] += int(sum(len(set(r1) & set(r2)) for r1, r2 in zip(a[k].tolist(), b[k].tolist())))
        c[2] += a[k].shape[0]
    rows = sum(c[2] for c in per.values())
    return {"per_visit": {str(v): {"rows": c[2], "same_expert_set_frac": c[0] / c[2],
                                   "slot_overlap_frac": c[1] / (c[2] * k_top)} for v, c in sorted(per.items())},
            "overall_same_expert_set_frac": sum(c[0] for c in per.values()) / max(rows, 1),
            "overall_slot_overlap_frac": sum(c[1] for c in per.values()) / max(rows * (k_top or 1), 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, choices=VARIANTS,
                    help="truth: all fp32 (weights upcast, fp32 activations, eager attention); bf16: stock HF numerics")
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokens", type=Path, action="append", default=None, help="repeatable")
    ap.add_argument("--max-tokens", type=int, default=1024, help="truncation of RAW (mask-less) segments")
    ap.add_argument("--long-max-tokens", type=int, default=6144, help="truncation of raw segments named *_long")
    ap.add_argument("--segment-raw", default=None, help="comma list of raw segment names to include (default all)")
    ap.add_argument("--segment", default=None, help="comma list: only these segment names (any file)")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--truth-dir", type=Path, default=None,
                    help="bf16 only: the finished truth dump; writes noise_term.json (+ router agreement)")
    ap.add_argument("--epsilon-candidates", default="0.0025,0.005,0.01")
    ap.add_argument("--threads", type=int, default=16, help="torch threads (max 24)")
    ap.add_argument("--debug-max-visits", type=int, default=None, help="NOT a valid reference (timing probes)")
    ap.add_argument("--only-first-tokens", type=int, default=None,
                    help="timing probe: keep only the first N tokens of the first selected segment, run all visits, "
                         "write NO dump")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--tiny", action="store_true", help="CPU smoke on a tiny random checkpoint (never touches the real weights)")
    ap.add_argument("--tiny-seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch

    from inkling2.common_inkling2 import assert_cpu_only, set_threads

    assert_cpu_only()
    set_threads(args.threads)
    truth = args.variant == "truth"
    dtype = torch.float32 if truth else torch.bfloat16
    tmp, config = None, None
    if args.tiny:
        tmp = tempfile.TemporaryDirectory()
        args.model_dir = build_tiny_checkpoint(Path(tmp.name) / "model", args.tiny_seed)
        config = tiny_config()
        g = torch.Generator().manual_seed(args.tiny_seed + 3)
        segs = [{"name": f"tiny_{k}", "chat": False, "file": "<tiny>",
                 "token_ids": [BEGIN_OF_TEXT_ID % 160] + [int(v) for v in torch.randint(4, 150, (n,), generator=g)]}
                for k, n in enumerate((20, 31))]
        args.out_dir = args.out_dir or Path(tmp.name) / args.variant
        tokens_sha = None
    else:
        if not args.tokens:
            raise SystemExit("--tokens required")
        segs = load_segments(args.tokens, args.max_tokens, args.long_max_tokens,
                             None if args.segment_raw is None else args.segment_raw.split(","),
                             None if args.segment is None else args.segment.split(","))
        tokens_sha = {str(p): sha256_file(p) for p in args.tokens}
    if args.only_first_tokens:
        segs = [{**segs[0], "token_ids": segs[0]["token_ids"][:args.only_first_tokens]}]
    if args.out_dir is None and not args.only_first_tokens:
        raise SystemExit("--out-dir required")
    out_dir = args.out_dir
    t_start = time.perf_counter()
    ref = Inkling2Reference(args.model_dir, dtype=dtype, config=config)
    for s in segs:
        if max(s["token_ids"]) >= ref.config.vocab_size:
            raise SystemExit(f"segment {s['name']}: id outside vocab")
    print(f"[ref] variant={args.variant} dtype={dtype} attn=eager experts={ref.experts_impl} visits={len(ref.visits)} "
          f"threads={torch.get_num_threads()} segments={[(s['name'], len(s['token_ids'])) for s in segs]}", flush=True)
    names = [s["name"] for s in segs]
    ckpt = None if (out_dir is None or args.tiny) else out_dir / "_ckpt" / "state.pt"
    if ckpt is not None and args.no_resume and ckpt.exists():
        ckpt.unlink()
    hid = ref.run({s["name"]: s["token_ids"] for s in segs}, ckpt=ckpt, debug_max_visits=args.debug_max_visits)
    secs = time.perf_counter() - t_start
    print(f"[ref] stack done in {secs / 60:.1f} min, peakRSS={peak_rss_gb() or 0:.2f}GiB", flush=True)
    if args.only_first_tokens or args.debug_max_visits is not None:
        print("[ref] probe run: no dump written")
        return 0
    import numpy as np

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "router_ids.npz", **{k: v.numpy() for k, v in ref.router_ids.items()})
    meta_common = {"source": "reference-fp32-truth" if truth else "hf-bf16-eager", "variant": args.variant,
                   "model_dir": str(args.model_dir), "attn": "eager", "work_dtype": str(dtype),
                   "experts_impl": ref.experts_impl, "n_layers": ref.n_layers, "n_visits": len(ref.visits),
                   "debug_max_layers": None, "torch_version": torch.__version__, "tokens_files": tokens_sha,
                   "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(), "stack_seconds": secs}
    write_dumps(ref, hid, segs, out_dir, meta_common)
    (out_dir / "run.json").write_text(json.dumps({
        "variant": args.variant, "segments": {s["name"]: len(s["token_ids"]) for s in segs}, "total_seconds":
        time.perf_counter() - t_start, "n_visits": len(ref.visits), "peak_rss_gib": peak_rss_gb(),
        "note": "see <seg>.meta.json"}, indent=2), encoding="utf-8")
    if not truth and args.truth_dir is not None and not args.tiny:
        import kl_report as K

        chat = [Path(p) for p in args.tokens if "score_mask" in json.loads(Path(p).read_text(encoding="utf-8"))["segments"][0]]
        raws = [Path(p) for p in args.tokens if Path(p) not in chat]
        if not chat:
            print("[ref] no chat tokens file among --tokens: noise_term.json skipped")
        else:
            cands = tuple(float(x) for x in args.epsilon_candidates.split(",") if x.strip())
            res = K.lfm2_noise_term(args.truth_dir, out_dir, chat[0], None, args.max_tokens, None, 32, True, cands)
            res["raw_noise"] = []
            for rp in raws:
                rmax = args.long_max_tokens if "long" in rp.name else args.max_tokens
                rr = K.lfm2_noise_term(args.truth_dir, out_dir, chat[0], rp, rmax, None, 32, True, cands)
                res["raw_noise"] += [gg for gg in rr["groups"] if gg["name"].startswith("raw")]
            if (args.truth_dir / "router_ids.npz").exists():
                res["router_id_agreement_truth_vs_bf16"] = router_agreement(args.truth_dir / "router_ids.npz",
                                                                            out_dir / "router_ids.npz")
            res["note"] = ("the inkling2-fp32 gate rule is KL <= 1.5 x this noise + epsilon (default 0.005): "
                           "kl_report.py --gate inkling2-fp32 [--epsilon <value>]")
            (out_dir / "noise_term.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
            print(K.lfm2_noise_markdown(res, title="Inkling2", noise_label="HF-bf16-eager"))
            print(f"[ref] wrote {out_dir / 'noise_term.json'}", flush=True)
    print(f"[ref] done in {(time.perf_counter() - t_start) / 60:.1f} min", flush=True)
    if tmp is not None:
        tmp.cleanup()
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        import tempfile as _t

        with _t.TemporaryDirectory() as _d:
            print(json.dumps(selftest_tiny(_d), indent=2))
        raise SystemExit(0)
    raise SystemExit(main())
