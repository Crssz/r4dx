"""`Qwen4ExpStreamingReference`: the qwen4_exp text stack one decoder layer at a time (plan 4.10, Q6/Q11).

The reference runs HF's own modules (`Qwen4ExpTextDecoderLayer` and the sub-modules inside it, from
transformers.models.qwen4_exp) on weights read straight from a checkpoint directory, one layer resident at a time;
only the residual [1, T, 10240] lives between layers. What is NOT HF's code:

* the weights: a layer is built on the meta device, its parameters are filled from the checkpoint (int4
  compressed-tensors linears are dequantized on the fly with `awq_dequant`, per linear; plain `.weight` tensors,
  as in the tiny model, are read as they are), and dropped after the layer;
* the MoE experts: `StreamingExperts` runs HF's eager per-expert loop (`Qwen4ExpTextExperts.forward`, same ops in
  the same order, so same bits) but dequantizes only the experts a token routed to, one at a time;
* the n-gram rows of the PLE layer come from a `NGramRows` (sidecar memmap, in-RAM table or shard reads, see
  ngram_rows.py) instead of HF's `nn.Embedding`, which would be 95 GiB; the row ids are `ngram_ids.NGramHash`;
* QSA token selection is `qsa_indexer.indexer_forward` with `tie_rule` "hf" (HF's torch.topk choice among tied
  blocks, reproduced) or "recent" (plan Q23: ties keep the most recent blocks; the rule r4dx uses);
* above `attn_query_chunk` tokens a QSA layer processes its queries in blocks (sparse mask per block) instead of
  one [T, T] mask; up to that length the HF attention module runs unchanged.

dtype: `dtype=torch.bfloat16` (default) is the stock bf16 stack the plan's Q11 reference names. `dtype=torch.float32` runs the same
modules with fp32 weights and activations: the "fp32 truth" of the gate (as the Gemma lane): int4 weights dequantized in fp32 without
the bf16 rounding (`fp32_exact_weights=False`: rounded to bf16 then widened, i.e. the same weights as the bf16 stack), fp32 residual,
QSA selection scores, router and logits. `source_tag` is then "truth-fp32"; `forward_logits` needs `mode="fp32"`.

Equivalences checked by test_ref.py on the tiny model (CPU, bf16): with `tie_rule="hf"` the logits equal a plain HF
forward (`use_cache=False`, sdpa attention, eager experts) bit for bit, with "recent" they equal the golden.

Device: "cpu", or "cuda" which `common.resolve_device` only allows with HIP_VISIBLE_DEVICES='1' (R4DX_ALLOW_GPU0=1
admits '0'). Batch size is 1, no KV cache (teacher forcing of whole sequences, as the KL gate needs).

Hooks (`ReferenceHooks`, passed to the constructor or per call) are the callback API of the quantization tooling:

    layer_input(i, residual)            residual [1, T, 10240] entering layer i
    layer_output(i, residual)           residual leaving layer i
    moe_routing(i, topk_ids, counts)    topk_ids [T, 10] int64, counts [n_experts] int64 (sums to T * top_k)
    linear_input(i, name, x)            only when `capture_linear_inputs`: the input rows [n, K] of every linear of layer
                                        i, by checkpoint base name (`model.language_model.layers.3.self_attn.q_proj`,
                                        `...mlp.shared_expert.gate_proj`, `...mlp.gate` (router), per expert
                                        `...mlp.experts.{e}.gate_proj` / `.up_proj` (same rows) / `.down_proj`); `x` is
                                        the live tensor on the device: copy it if you keep it.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import raw_safetensors_header, raw_safetensors_read, resolve_device  # noqa: E402
from qwen4exp import qsa_indexer  # noqa: E402
from qwen4exp.arch import EMBED_NAME, GDN, LM_HEAD_NAME, MIXER_PREFIX, PACKED_PARTS, QSA, Qwen4ExpArch  # noqa: E402
from qwen4exp.awq_dequant import dequant_pack_quantized  # noqa: E402
from qwen4exp.awq_dequant_f32 import dequant_pack_quantized_f32  # noqa: E402
from qwen4exp.ngram_ids import NGramHash  # noqa: E402
from qwen4exp.ngram_rows import NGramRows, open_ngram_rows  # noqa: E402

DTYPE = torch.bfloat16
LOGPROB_CLAMP = -1e4
LOGITS_MODES = ("fp32", "hf-bf16")
TIE_RULES = ("hf", "recent")
ATTN_IMPLS = ("sdpa", "eager")
DEFAULT_ATTN_QUERY_CHUNK = 4096


# --------------------------------------------------------------------------------------------------------
# hooks
# --------------------------------------------------------------------------------------------------------


class ReferenceHooks:
    """No-op base of the callback API (module docstring). Subclass and override what you need."""

    capture_linear_inputs = False

    def layer_input(self, layer: int, residual: torch.Tensor) -> None:
        pass

    def layer_output(self, layer: int, residual: torch.Tensor) -> None:
        pass

    def moe_routing(self, layer: int, topk_ids: torch.Tensor, counts: torch.Tensor) -> None:
        pass

    def linear_input(self, layer: int, name: str, x: torch.Tensor) -> None:
        pass


class RoutingCounts(ReferenceHooks):
    """Per-layer, per-expert routed token counts: `counts[layer]` is an int64 [n_experts] tensor."""

    def __init__(self):
        self.counts: dict[int, torch.Tensor] = {}

    def moe_routing(self, layer, topk_ids, counts):
        self.counts[layer] = counts.detach().cpu().clone()


class LayerCapture(ReferenceHooks):
    """Keeps the residual entering/leaving the chosen layers (None: all) as CPU tensors [T, 10240]."""

    def __init__(self, layers=None):
        self.layers = None if layers is None else set(layers)
        self.inputs: dict[int, torch.Tensor] = {}
        self.outputs: dict[int, torch.Tensor] = {}

    def layer_input(self, layer, residual):
        if self.layers is None or layer in self.layers:
            self.inputs[layer] = residual[0].detach().cpu().clone()

    def layer_output(self, layer, residual):
        if self.layers is None or layer in self.layers:
            self.outputs[layer] = residual[0].detach().cpu().clone()


class CallbackHooks(ReferenceHooks):
    """Hooks from plain callables (any may be None); `linear_input` set turns linear-input capture on."""

    def __init__(self, layer_input=None, layer_output=None, moe_routing=None, linear_input=None):
        self._fns = {"layer_input": layer_input, "layer_output": layer_output, "moe_routing": moe_routing,
                     "linear_input": linear_input}
        self.capture_linear_inputs = linear_input is not None

    def layer_input(self, layer, residual):
        if self._fns["layer_input"]:
            self._fns["layer_input"](layer, residual)

    def layer_output(self, layer, residual):
        if self._fns["layer_output"]:
            self._fns["layer_output"](layer, residual)

    def moe_routing(self, layer, topk_ids, counts):
        if self._fns["moe_routing"]:
            self._fns["moe_routing"](layer, topk_ids, counts)

    def linear_input(self, layer, name, x):
        if self._fns["linear_input"]:
            self._fns["linear_input"](layer, name, x)


# --------------------------------------------------------------------------------------------------------
# checkpoint access
# --------------------------------------------------------------------------------------------------------


class CheckpointReader:
    """Tensor reads by name from a HF checkpoint dir (index or single file), plain file reads, no mmap."""

    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir)
        index = self.model_dir / "model.safetensors.index.json"
        if index.exists():
            self.weight_map: dict[str, str] = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        else:
            single = self.model_dir / "model.safetensors"
            if not single.exists():
                raise FileNotFoundError(f"{self.model_dir} has neither model.safetensors.index.json nor model.safetensors")
            self.weight_map = {k: "model.safetensors" for k in raw_safetensors_header(single)[0]}
        self._headers: dict[str, tuple[dict, int]] = {}
        self.weight_dtype = torch.bfloat16  # dtype of `linear_weight` (the reference sets float32 for dtype=float32)
        self.exact_dequant = False          # float32 only: keep the int4 dequant unrounded (else bf16-rounded, then widened)

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def is_packed(self, base: str) -> bool:
        return base + ".weight_packed" in self.weight_map

    def _file(self, name: str) -> tuple[Path, tuple[dict, int]]:
        if name not in self.weight_map:
            raise KeyError(f"{name} is not in the checkpoint index of {self.model_dir}")
        file = self.weight_map[name]
        if file not in self._headers:
            self._headers[file] = raw_safetensors_header(self.model_dir / file)
        return self.model_dir / file, self._headers[file]

    def get(self, name: str, rows: tuple[int, int] | None = None) -> torch.Tensor:
        path, hdr = self._file(name)
        return raw_safetensors_read(path, name, hdr, rows=rows)

    def get_rows(self, name: str, row_ids) -> torch.Tensor:
        """Rows `row_ids` (list of ints) of a 2D tensor, one open file, sorted reads."""
        path, (hdr, off) = self._file(name)
        ent = hdr[name]
        cols = ent["shape"][1]
        dt = {"BF16": torch.bfloat16, "F32": torch.float32, "F16": torch.float16}[ent["dtype"]]
        esz = torch.empty(0, dtype=dt).element_size()
        row_bytes = cols * esz
        base = off + ent["data_offsets"][0]
        out = torch.empty((len(row_ids), cols), dtype=dt)
        buf = out.view(torch.uint8).reshape(len(row_ids), row_bytes).numpy()
        with open(path, "rb", buffering=0) as f:
            for k in np.argsort(np.asarray(row_ids), kind="stable"):
                f.seek(base + int(row_ids[k]) * row_bytes)
                got = f.readinto(memoryview(buf[k]))
                if got != row_bytes:
                    raise EOFError(f"{path}: {name} row {row_ids[k]}: short read")
        return out

    def linear_weight(self, base: str, device: torch.device) -> torch.Tensor:
        """[N, K] weight on `device` of the linear `base`: plain `.weight`, or dequantized pack-quantized int4. bf16 (default):
        the dequant rounded to bf16 as compressed-tensors does. float32: widened (`exact_dequant` False) or the unrounded fp32 dequant."""
        if not self.is_packed(base):
            w = self.get(base + ".weight").to(device)
            return w if self.weight_dtype == torch.bfloat16 else w.to(self.weight_dtype)
        parts = {p: self.get(f"{base}.{p}") for p in PACKED_PARTS}
        shape = [int(v) for v in parts["weight_shape"].tolist()]
        with torch.device(device):  # awq_dequant builds its shift table with torch.arange
            args = (parts["weight_packed"].to(device), parts["weight_scale"].to(device),
                    parts["weight_zero_point"].to(device), shape)
            if self.weight_dtype == torch.bfloat16:
                return dequant_pack_quantized(*args)
            return dequant_pack_quantized_f32(*args, round_to_bf16=not self.exact_dequant)


# --------------------------------------------------------------------------------------------------------
# MoE experts, streamed
# --------------------------------------------------------------------------------------------------------


class StreamingExperts(nn.Module):
    """Stand-in for `Qwen4ExpTextExperts`: HF's eager per-expert loop with lazily dequantized expert weights."""

    def __init__(self, ref: "Qwen4ExpStreamingReference", layer: int, act_fn):
        super().__init__()
        self.ref = ref
        self.layer = layer
        self.act_fn = act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        ref, i = self.ref, self.layer
        num_experts = ref.arch.n_experts
        hooks = ref.hooks
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        counts = expert_mask[:num_experts].sum(dim=(-1, -2))
        hooks.moe_routing(i, top_k_index, counts)
        capture = hooks.capture_linear_inputs
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == num_experts:
                continue
            e = int(expert_idx)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate_w = ref.reader.linear_weight(ref.arch.expert_base(i, e, "gate_proj"), ref.device)
            up_w = ref.reader.linear_weight(ref.arch.expert_base(i, e, "up_proj"), ref.device)
            down_w = ref.reader.linear_weight(ref.arch.expert_base(i, e, "down_proj"), ref.device)
            if capture:
                hooks.linear_input(i, ref.arch.expert_base(i, e, "gate_proj"), current_state)
                hooks.linear_input(i, ref.arch.expert_base(i, e, "up_proj"), current_state)
            gate, up = nn.functional.linear(current_state, torch.cat([gate_w, up_w], dim=0)).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            if capture:
                hooks.linear_input(i, ref.arch.expert_base(i, e, "down_proj"), current_hidden_states)
            current_hidden_states = nn.functional.linear(current_hidden_states, down_w)
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


# --------------------------------------------------------------------------------------------------------
# the reference
# --------------------------------------------------------------------------------------------------------


def token_ids_sha256(token_ids) -> str:
    """Byte-identical to full_logits_golden.token_ids_sha256 / make_tokens_json.token_ids_sha256."""
    from common import sha256_bytes

    payload = json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


class Qwen4ExpStreamingReference:
    source_tag = "reference"

    def __init__(self, model_dir: str | Path, device: str | torch.device = "cpu", *, tie_rule: str = "hf",
                 ngram_sidecar: str | Path | None = None, max_layers: int | None = None, attn: str = "sdpa",
                 attn_query_chunk: int = DEFAULT_ATTN_QUERY_CHUNK, hooks: ReferenceHooks | None = None,
                 verbose: bool = False, dtype: torch.dtype = DTYPE, fp32_exact_weights: bool = True):
        from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpConfig

        if tie_rule not in TIE_RULES:
            raise ValueError(f"tie_rule must be one of {TIE_RULES}, got {tie_rule!r}")
        if attn not in ATTN_IMPLS:
            raise ValueError(f"attn must be one of {ATTN_IMPLS}, got {attn!r}")
        self.model_dir = Path(model_dir)
        self.device = device if isinstance(device, torch.device) else resolve_device(device)
        self.tie_rule = tie_rule
        self.attn = attn
        self.attn_query_chunk = attn_query_chunk
        self.hooks = hooks or ReferenceHooks()
        self.verbose = verbose
        self.config_dict = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.arch = Qwen4ExpArch.from_config(self.config_dict)
        if max_layers is not None and not 1 <= max_layers <= self.arch.n_layers:
            raise ValueError(f"max_layers must be in [1, {self.arch.n_layers}], got {max_layers}")
        self.max_layers = max_layers
        self.n_layers = self.arch.n_layers if max_layers is None else max_layers
        self.vocab_size = self.arch.vocab
        self.reader = CheckpointReader(self.model_dir)
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"dtype must be torch.bfloat16 or torch.float32, got {dtype}")
        self.dtype = dtype
        self.fp32_exact_weights = fp32_exact_weights
        self.reader.weight_dtype = dtype
        self.reader.exact_dequant = dtype == torch.float32 and fp32_exact_weights
        if dtype == torch.float32:
            self.source_tag = "truth-fp32"  # the whole stack in fp32 (plan: the gate's fp32 truth, as the Gemma lane)
        problems = self.arch.check_against_weight_map(self.reader.weight_map, layers=range(self.n_layers))
        if problems:
            raise ValueError(f"{self.model_dir} does not match the qwen4_exp tensor table: " + "; ".join(problems[:10])
                             + (f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""))
        self.config = Qwen4ExpConfig(**{k: v for k, v in self.config_dict.items() if k != "quantization_config"})
        self.text_config = self.config.text_config
        self.text_config._attn_implementation = attn
        self.ngram_hash = NGramHash.from_config(self.config_dict, ple_layer_index=0)
        self.ngram: NGramRows | None = None
        self._ngram_sidecar = ngram_sidecar
        self.last_ngram_ids: np.ndarray | None = None
        self._lm_head: torch.Tensor | None = None
        self._rotary = None
        self._mixer = None
        self.packed = any(self.reader.is_packed(b) for b in self.arch.quantized_linears(0))
        if verbose:
            print(f"[qwen4exp-ref] {self.model_dir.name}: {self.n_layers}/{self.arch.n_layers} layers, "
                  f"{'int4 AWQ' if self.packed else 'bf16'} linears, device {self.device}, tie_rule {tie_rule}, "
                  f"attn {attn}", flush=True)

    # ---- pieces ------------------------------------------------------------------------------------------

    def segment_meta(self) -> dict:
        return {"n_layers": self.n_layers, "debug_max_layers": self.max_layers, "tie_rule": self.tie_rule,
                "ngram": self.ngram.describe() if self.ngram else None, "compute_dtype": str(self.dtype).replace("torch.", ""),
                "fp32_exact_weights": self.fp32_exact_weights if self.dtype == torch.float32 else None}

    def _open_ngram(self) -> NGramRows:
        if self.ngram is None:
            if not self.arch.ple_layers() or self.arch.ple_layers()[0] >= self.n_layers:
                raise RuntimeError("no PLE layer in the layers that run")
            self.ngram = open_ngram_rows(self.model_dir, self.ngram_hash, self._ngram_sidecar)
            if self.verbose:
                print(f"[qwen4exp-ref] n-gram rows: {self.ngram.describe()}", flush=True)
        return self.ngram

    def embed(self, token_ids) -> torch.Tensor:
        ids = [int(t) for t in token_ids]
        if min(ids) < 0 or max(ids) >= self.vocab_size:
            raise ValueError(f"token id outside [0, {self.vocab_size})")
        uniq = sorted(set(ids))
        rows = self.reader.get_rows(EMBED_NAME, uniq)
        pos = {t: k for k, t in enumerate(uniq)}
        return rows[torch.tensor([pos[t] for t in ids])].to(self.device, self.dtype)

    def _rotary_emb(self):
        if self._rotary is None:
            from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRotaryEmbedding

            self._rotary = Qwen4ExpTextRotaryEmbedding(self.text_config).to(self.device)
        return self._rotary

    def _load_module(self, module: nn.Module, prefix: str, what: str) -> None:
        """Fill `module` (materialised, uninitialised) from the checkpoint; every parameter and buffer must be set."""
        for name, p in list(module.named_parameters()) + list(module.named_buffers()):
            full = prefix + name
            base = full[: -len(".weight")] if full.endswith(".weight") else None
            if base is not None and self.reader.is_packed(base):
                t = self.reader.linear_weight(base, self.device)
            else:
                t = self.reader.get(full)
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"{what}: {full} is {tuple(t.shape)} in the checkpoint, the module wants {tuple(p.shape)}")
            with torch.no_grad():
                p.copy_(t.to(device=self.device, dtype=p.dtype))
            p.requires_grad_(False)

    def _build_layer(self, i: int) -> nn.Module:
        from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

        old = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            with torch.device("meta"):
                layer = Qwen4ExpTextDecoderLayer(self.text_config, i)
        finally:
            torch.set_default_dtype(old)
        act_fn = layer.mlp.experts.act_fn
        layer.mlp.experts = StreamingExperts(self, i, act_fn)
        if layer.ple is not None:
            layer.ple.ple_embedding.ngram_embedding = None
            layer.ple.ple_embedding.forward = lambda input_ids, past_key_values=None: self._ngram_embedding(input_ids)
        layer.to_empty(device=self.device)
        self._load_module(layer, self.arch.layer_prefix(i), f"layer {i}")
        layer.eval()
        if self.arch.layer_types[i] == QSA:
            attn = layer.self_attn
            ix = attn.indexer
            rule = self.tie_rule
            ix.forward = lambda h, pe, m, pkv=None: qsa_indexer.indexer_forward(ix, h, pe, m, pkv, tie_rule=rule)
            attn.forward = lambda h, pe, attention_mask=None, past_key_values=None, **kw: self._qsa_forward(
                attn, h, pe, attention_mask, past_key_values, **kw)
        if self.hooks.capture_linear_inputs:
            self._register_linear_hooks(layer, i)
        return layer

    def _register_linear_hooks(self, layer: nn.Module, i: int) -> None:
        prefix = self.arch.layer_prefix(i)
        for name, mod in layer.named_modules():
            if isinstance(mod, nn.Linear) or name.endswith("mlp.gate"):
                full = prefix + name
                mod.register_forward_pre_hook(
                    lambda m, args, full=full: self.hooks.linear_input(i, full, args[0].reshape(-1, args[0].shape[-1])))

    def _ngram_embedding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Replacement for `Qwen4ExpTextNGramEmbedding.forward`: [1, T, heads * head_dim] bf16 on the device."""
        ids_np = input_ids.reshape(-1).cpu().numpy()
        ngram_ids, _ = self.ngram_hash.ngram_ids(ids_np)
        self.last_ngram_ids = ngram_ids
        rows = self._open_ngram().rows(ngram_ids)  # [T, heads, head_dim] bf16
        return rows.flatten(-2).unsqueeze(0).to(self.device, self.dtype)

    # ---- QSA layer ---------------------------------------------------------------------------------------

    def _qsa_forward(self, attn, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        """`Qwen4ExpTextAttention.forward` with the query axis processed in blocks of `attn_query_chunk`.

        Used instead of the HF module only when the sequence is longer than one block; otherwise the original
        method (bound to this instance's class) runs unchanged."""
        from transformers.models.qwen4_exp.modeling_qwen4_exp import (
            ALL_ATTENTION_FUNCTIONS,
            Qwen4ExpTextAttention,
            apply_rotary_pos_emb,
            eager_attention_forward,
        )

        seq = hidden_states.shape[1]
        if seq <= self.attn_query_chunk:
            return Qwen4ExpTextAttention.forward(attn, hidden_states, position_embeddings, attention_mask,
                                                 past_key_values, **kwargs)
        cos_full, sin_full = position_embeddings
        ix = attn.indexer
        q_idx, raw_keys = qsa_indexer.project(ix, hidden_states, position_embeddings)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, attn.head_dim)
        query_states, gate = torch.chunk(attn.q_proj(hidden_states).view(*input_shape, -1, attn.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)
        query_states = attn.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos_full, sin_full)
        interface = ALL_ATTENTION_FUNCTIONS.get_interface(attn.config._attn_implementation, eager_attention_forward)
        outs = []
        positions = torch.arange(seq, device=hidden_states.device)
        for c0 in range(0, seq, self.attn_query_chunk):
            c1 = min(seq, c0 + self.attn_query_chunk)
            visible = (positions[None, :c1] <= positions[c0:c1, None])[None, None]  # [1, 1, c, c1]
            selected = qsa_indexer.select_tokens(
                q_idx[:, c0:c1], raw_keys[:, :c1], cos_full[:, :c1], sin_full[:, :c1], visible,
                ix.k_layernorm.weight, ix.k_layernorm.eps, budget=ix.token_budget, compress_ratio=ix.compress_ratio,
                tie_rule=self.tie_rule)
            mask = selected if self.attn == "sdpa" else torch.where(selected, 0.0, torch.finfo(hidden_states.dtype).min).to(hidden_states.dtype)
            out, _ = interface(attn, query_states[:, :, c0:c1], key_states[:, :, :c1], value_states[:, :, :c1], mask,
                               dropout=0.0, scaling=attn.scaling)
            outs.append(out.reshape(1, c1 - c0, -1) * torch.sigmoid(gate[:, c0:c1]))
        return attn.o_proj(torch.cat(outs, dim=1).contiguous()), None

    # ---- forward -----------------------------------------------------------------------------------------

    @torch.no_grad()
    def run_layers(self, token_ids, hooks: ReferenceHooks | None = None) -> torch.Tensor:
        """Residual [1, T, 10240] after the first `n_layers` decoder layers."""
        from transformers.models.qwen4_exp.modeling_qwen4_exp import create_causal_mask, create_recurrent_attention_mask

        prev_hooks = self.hooks
        if hooks is not None:
            self.hooks = hooks
        try:
            ids = torch.tensor([[int(t) for t in token_ids]], dtype=torch.long)
            seq = ids.shape[1]
            embeds = self.embed(token_ids).unsqueeze(0)
            position_ids = (torch.arange(seq, device=self.device)).view(1, 1, -1).expand(4, 1, -1)
            text_position_ids, position_ids = position_ids[0], position_ids[1:]
            mask_kwargs = {"config": self.text_config, "inputs_embeds": embeds, "attention_mask": None,
                           "past_key_values": None, "position_ids": text_position_ids, "allow_is_causal_skip": False}
            qsa_dense = seq <= self.attn_query_chunk
            attention_mask = create_causal_mask(**mask_kwargs) if qsa_dense else None
            conv_mask = create_recurrent_attention_mask(**mask_kwargs)
            ple_ids = ids.to(self.device)
            if conv_mask is not None:
                ple_ids = torch.where(conv_mask.bool(), ple_ids, self.arch.eos_token_id)
            position_embeddings = self._rotary_emb()(embeds, position_ids)
            hidden = embeds.repeat(1, 1, self.arch.hc_count)
            for i in range(self.n_layers):
                t0 = time.perf_counter()
                self.hooks.layer_input(i, hidden)
                layer = self._build_layer(i)
                hidden = layer(hidden, position_embeddings=position_embeddings, attention_mask=attention_mask,
                               conv_mask=conv_mask, past_key_values=None, ple_input_ids=ple_ids)
                self.hooks.layer_output(i, hidden)
                del layer
                if self.verbose:
                    print(f"[qwen4exp-ref] layer {i} ({self.arch.layer_types[i]}) {time.perf_counter() - t0:.1f}s", flush=True)
            return hidden
        finally:
            self.hooks = prev_hooks

    @torch.no_grad()
    def mix(self, residual: torch.Tensor) -> torch.Tensor:
        """`hyper_connection_mixer`: [1, T, 10240] -> [1, T, 2560] (the lm_head input)."""
        if self._mixer is None:
            from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextGatedResidual

            old = torch.get_default_dtype()
            torch.set_default_dtype(self.dtype)
            try:
                with torch.device("meta"):
                    mixer = Qwen4ExpTextGatedResidual(self.text_config, use_combine=False)
            finally:
                torch.set_default_dtype(old)
            mixer.to_empty(device=self.device)
            self._load_module(mixer, MIXER_PREFIX, "hyper_connection_mixer")
            self._mixer = mixer.eval()
        return self._mixer(residual)

    def forward_hidden(self, token_ids, hooks: ReferenceHooks | None = None) -> torch.Tensor:
        """Mixer output [T, 2560] after `n_layers` layers (all of them unless max_layers was given)."""
        return self.mix(self.run_layers(token_ids, hooks))[0]

    def _lm_head_weight(self) -> torch.Tensor:
        if self._lm_head is None:
            self._lm_head = self.reader.get(LM_HEAD_NAME).to(self.device)
        return self._lm_head

    @torch.no_grad()
    def logits_rows(self, hidden_rows: torch.Tensor, mode: str = "fp32", chunk: int | None = None) -> torch.Tensor:
        """fp32 logits [..., V] of hidden rows [..., 2560]. "hf-bf16": lm_head in bf16 like HF (then widened);
        "fp32": accumulated in fp32. `chunk` limits the vocab rows multiplied at once."""
        if mode not in LOGITS_MODES:
            raise ValueError(f"mode must be one of {LOGITS_MODES}, got {mode!r}")
        if mode == "hf-bf16" and self.dtype != torch.bfloat16:
            raise ValueError("logits mode 'hf-bf16' needs the bf16 stack; use 'fp32' with dtype=float32")
        w = self._lm_head_weight()
        chunk = chunk or w.shape[0]
        parts = []
        for v0 in range(0, w.shape[0], chunk):
            wc = w[v0:v0 + chunk]
            if mode == "hf-bf16":
                parts.append(F.linear(hidden_rows, wc).float())
            else:
                parts.append(F.linear(hidden_rows.float(), wc.float()))
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)

    @torch.no_grad()
    def forward_logits(self, token_ids, mode: str = "hf-bf16", hooks: ReferenceHooks | None = None) -> torch.Tensor:
        """Logits [T, V] (fp32) of every position, lm_head applied to the whole sequence at once."""
        hidden = self.mix(self.run_layers(token_ids, hooks))
        return self.logits_rows(hidden, mode)[0]

    @torch.no_grad()
    def logprobs_to_file(self, token_ids, f, mode: str = "fp32", chunk: int = 32768, row_block: int = 128,
                         hooks: ReferenceHooks | None = None) -> dict:
        """Write rows 0..T-2 of log_softmax(logits) (fp32, clamped at LOGPROB_CLAMP, fp16) to the binary file `f`
        (format of full_logits_golden.py). Returns nll, worst |logsumexp| of the fp16 rows, top-5 of position T-1."""
        T = len(token_ids)
        t0 = time.perf_counter()
        hidden = self.forward_hidden(token_ids, hooks)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        t_stack = time.perf_counter() - t0
        t1 = time.perf_counter()
        nll_sum, worst = 0.0, 0.0
        for s in range(0, T - 1, row_block):
            e = min(s + row_block, T - 1)
            lg = self.logits_rows(hidden[s:e], mode, chunk)
            lp = torch.log_softmax(lg, dim=-1).clamp_min(LOGPROB_CLAMP).to(torch.float16)
            worst = max(worst, torch.logsumexp(lp.float(), dim=-1).abs().max().item())
            tgt = torch.as_tensor([int(t) for t in token_ids[s + 1:e + 1]], device=lp.device)
            nll_sum -= lp[torch.arange(e - s, device=lp.device), tgt].float().sum().item()
            f.write(np.ascontiguousarray(lp.cpu().numpy()).tobytes())
            del lg, lp
        last = self.logits_rows(hidden[-1:], mode, chunk)[0]
        top = torch.topk(last, min(5, last.numel()))
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        return {"nll": nll_sum / (T - 1), "worst_lse": worst, "seconds_stack": t_stack,
                "seconds_lm_head": time.perf_counter() - t1,
                "last_top": [(float(v), int(i)) for v, i in zip(top.values.tolist(), top.indices.tolist())],
                "last_probs": torch.softmax(last, dim=-1)[top.indices].tolist()}
