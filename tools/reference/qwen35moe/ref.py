"""`Qwen35MoeStreamingReference`: the qwen3_5_moe (Ornith-1.5-122B) text stack one decoder layer at a time.

The reference runs HF's own modules (`Qwen3_5MoeDecoderLayer` and everything inside it, transformers.models.qwen3_5_moe) on
weights read straight from the checkpoint shards, one layer resident at a time (a layer is ~5.3 GB bf16, 4.8 GB of it the two
fused expert tensors; the whole checkpoint is 250 GB and fits no RAM): only the residual [1, T, 3072] lives between layers.
What is NOT HF's code:

* the weights: a layer is built on the meta device, its parameters are filled from the checkpoint by name
  (`mlp.experts.gate_up_proj` [E, 2I, H] and `down_proj` [E, H, I] as stored) and dropped after the layer;
* the routed-experts forward: `HookedExperts` is `Qwen3_5MoeExperts.forward` line for line (same ops, same order, so same bits)
  with the `ReferenceHooks` callbacks added;
* the masks: batch 1, no padding, so no mask is built (`attention_mask=None`: sdpa runs causal, the GDN layers see no padding).

Equivalence checked by test_ref.py on the tiny model (CPU, bf16): the logits equal a plain HF forward (`use_cache=False`, sdpa
attention, eager experts) bit for bit, and the golden's.

dtype: bfloat16 (default) is the stock bf16 stack, the KL gate's reference; float32 runs the same modules in fp32 (the "truth").
Device: "cpu", or "cuda", which `common.resolve_device` only allows with HIP_VISIBLE_DEVICES='1' (R4DX_ALLOW_GPU0=1 admits '0').

Hooks (`ReferenceHooks`, passed to the constructor or per call), the callback API of the quantization tooling:

    layer_input(i, residual)            residual [1, T, 3072] entering layer i
    layer_output(i, residual)           residual leaving layer i
    moe_routing(i, topk_ids, counts)    topk_ids [T, 8] int64, counts [n_experts] int64 (sums to T * top_k)
    linear_input(i, name, x)            only when `capture_linear_inputs`: the input rows [n, K] of every linear of layer i, by
                                        checkpoint base name (`...layers.3.self_attn.q_proj`, `...mlp.shared_expert.gate_proj`,
                                        `...mlp.gate` (router), per expert `...mlp.experts.{e}.gate_proj` / `.up_proj` (same
                                        rows) / `.down_proj` (the silu(gate) * up activation)); `x` is the live tensor on the
                                        device: copy it if you keep it.
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import raw_safetensors_header, raw_safetensors_read, resolve_device  # noqa: E402
from qwen35moe.arch import EMBED_NAME, FINAL_NORM_NAME, FULL, LM_HEAD_NAME, Qwen35MoeArch  # noqa: E402

DTYPE = torch.bfloat16
LOGITS_MODES = ("fp32", "hf-bf16")
EXPERT_RE = re.compile(r"^(?P<fused>.*\.mlp\.experts\.)(?P<e>\d+)\.(?P<proj>gate_proj|up_proj|down_proj)$")


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
    """Keeps the residual entering/leaving the chosen layers (None: all) as CPU tensors [T, 3072]."""

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


# --------------------------------------------------------------------------------------------------------
# checkpoint access
# --------------------------------------------------------------------------------------------------------


class CheckpointReader:
    """Tensor reads by name from a HF checkpoint dir (index or single file), plain file reads, no mmap. Expert linears are
    addressed by their VIRTUAL per-expert names (`...mlp.experts.{e}.gate_proj`), sliced out of the fused 3-D tensors."""

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

    def has(self, name: str) -> bool:
        return name in self.weight_map

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

    def is_expert_base(self, base: str) -> bool:
        return EXPERT_RE.match(base) is not None

    def linear_weight(self, base: str, device: torch.device) -> torch.Tensor:
        """[N, K] weight on `device` of the linear `base` (bf16 as stored, or widened to float32): a plain `.weight`, or a
        virtual expert linear: gate_proj = rows [0, I) and up_proj = rows [I, 2I) of expert e of `gate_up_proj`, down_proj =
        expert e of `down_proj`."""
        m = EXPERT_RE.match(base)
        if m is None:
            w = self.get(base + ".weight")
        else:
            e, proj = int(m["e"]), m["proj"]
            if proj == "down_proj":
                w = self.get(m["fused"] + "down_proj", rows=(e, e + 1))[0]
            else:
                slab = self.get(m["fused"] + "gate_up_proj", rows=(e, e + 1))[0]      # [2I, H]
                half = slab.shape[0] // 2
                w = slab[:half] if proj == "gate_proj" else slab[half:]
            w = w.contiguous()
        w = w.to(device)
        return w if self.weight_dtype == torch.bfloat16 else w.to(self.weight_dtype)


# --------------------------------------------------------------------------------------------------------
# MoE experts with hooks
# --------------------------------------------------------------------------------------------------------


class HookedExperts(nn.Module):
    """`Qwen3_5MoeExperts` with the `ReferenceHooks` callbacks: the same eager per-expert loop over the fused parameters."""

    def __init__(self, ref: "Qwen35MoeStreamingReference", layer: int, src: nn.Module):
        super().__init__()
        self.ref = ref
        self.layer = layer
        self.num_experts = src.num_experts
        self.gate_up_proj = src.gate_up_proj
        self.down_proj = src.down_proj
        self.act_fn = src.act_fn

    def forward(self, hidden_states, top_k_index, top_k_weights):
        ref, i = self.ref, self.layer
        hooks = ref.hooks
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts + 1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        counts = expert_mask[:self.num_experts].sum(dim=(-1, -2))
        hooks.moe_routing(i, top_k_index, counts)
        capture = hooks.capture_linear_inputs
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            e = int(expert_idx)
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            if capture:
                hooks.linear_input(i, ref.arch.expert_base(i, e, "gate_proj"), current_state)
                hooks.linear_input(i, ref.arch.expert_base(i, e, "up_proj"), current_state)
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            if capture:
                hooks.linear_input(i, ref.arch.expert_base(i, e, "down_proj"), current_hidden_states)
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states


# --------------------------------------------------------------------------------------------------------
# the reference
# --------------------------------------------------------------------------------------------------------


def token_ids_sha256(token_ids) -> str:
    from common import sha256_bytes

    payload = json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode("utf-8")
    return sha256_bytes(payload)


class Qwen35MoeStreamingReference:
    source_tag = "reference"

    def __init__(self, model_dir: str | Path, device: str | torch.device = "cpu", *, max_layers: int | None = None,
                 attn: str = "sdpa", hooks: ReferenceHooks | None = None, verbose: bool = False, dtype: torch.dtype = DTYPE):
        from transformers import Qwen3_5MoeConfig

        self.model_dir = Path(model_dir)
        self.device = device if isinstance(device, torch.device) else resolve_device(device)
        self.attn = attn
        self.hooks = hooks or ReferenceHooks()
        self.verbose = verbose
        self.config_dict = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.arch = Qwen35MoeArch.from_config(self.config_dict)
        if max_layers is not None and not 1 <= max_layers <= self.arch.n_layers:
            raise ValueError(f"max_layers must be in [1, {self.arch.n_layers}], got {max_layers}")
        self.max_layers = max_layers
        self.n_layers = self.arch.n_layers if max_layers is None else max_layers
        self.vocab_size = self.arch.vocab
        self.reader = CheckpointReader(self.model_dir)
        if dtype not in (torch.bfloat16, torch.float32):
            raise ValueError(f"dtype must be torch.bfloat16 or torch.float32, got {dtype}")
        self.dtype = dtype
        self.reader.weight_dtype = dtype
        if dtype == torch.float32:
            self.source_tag = "truth-fp32"
        problems = self.arch.check_against_weight_map(self.reader.weight_map, layers=range(self.n_layers))
        if problems:
            raise ValueError(f"{self.model_dir} does not match the qwen3_5_moe tensor table: " + "; ".join(problems[:10])
                             + (f" (+{len(problems) - 10} more)" if len(problems) > 10 else ""))
        self.config = Qwen3_5MoeConfig.from_pretrained(str(self.model_dir))
        self.text_config = self.config.text_config
        self.text_config._attn_implementation = attn
        try:
            self.text_config.experts_implementation = "eager"
        except Exception:  # noqa: BLE001
            pass
        self._lm_head: torch.Tensor | None = None
        self._final_norm = None
        self._rotary = None
        if verbose:
            print(f"[qwen35moe-ref] {self.model_dir.name}: {self.n_layers}/{self.arch.n_layers} layers, device {self.device}, "
                  f"attn {attn}", flush=True)

    # ---- pieces ------------------------------------------------------------------------------------------

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
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTextRotaryEmbedding

            self._rotary = Qwen3_5MoeTextRotaryEmbedding(self.text_config).to(self.device)
        return self._rotary

    def position_embeddings(self, seq: int, like: torch.Tensor):
        """(cos, sin) of positions 0..seq-1 as `Qwen3_5MoeTextModel.forward` builds them (text positions on the 3 mrope axes)."""
        position_ids = torch.arange(seq, device=self.device).view(1, 1, -1).expand(4, 1, -1)[1:]
        return self._rotary_emb()(like, position_ids)

    def _load_module(self, module: nn.Module, prefix: str, what: str) -> None:
        """Fill `module` (materialised, uninitialised) from the checkpoint; every parameter and buffer must be set. Buffers that
        are not in the checkpoint (the rotary inv_freq lives outside the layers) are not expected here."""
        for name, p in list(module.named_parameters()) + list(module.named_buffers()):
            full = prefix + name
            t = self.reader.get(full)
            if tuple(t.shape) != tuple(p.shape):
                raise ValueError(f"{what}: {full} is {tuple(t.shape)} in the checkpoint, the module wants {tuple(p.shape)}")
            with torch.no_grad():
                p.copy_(t.to(device=self.device, dtype=p.dtype))
            p.requires_grad_(False)

    def _build_layer(self, i: int) -> nn.Module:
        from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeDecoderLayer

        old = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)
        try:
            with torch.device("meta"):
                layer = Qwen3_5MoeDecoderLayer(self.text_config, i)
        finally:
            torch.set_default_dtype(old)
        layer.to_empty(device=self.device)
        self._load_module(layer, self.arch.layer_prefix(i), f"layer {i}")
        layer.eval()
        layer.mlp.experts = HookedExperts(self, i, layer.mlp.experts)
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

    # ---- forward -----------------------------------------------------------------------------------------

    def layer_forward(self, layer_mod: nn.Module, hidden: torch.Tensor, position_embeddings) -> torch.Tensor:
        """One decoder layer exactly as `Qwen3_5MoeTextModel.forward` calls it (batch 1, no padding, no cache)."""
        return layer_mod(hidden, position_embeddings=position_embeddings, attention_mask=None, position_ids=None,
                         past_key_values=None)

    @torch.no_grad()
    def run_layers(self, token_ids, hooks: ReferenceHooks | None = None) -> torch.Tensor:
        """Residual [1, T, 3072] after the first `n_layers` decoder layers."""
        prev_hooks = self.hooks
        if hooks is not None:
            self.hooks = hooks
        try:
            seq = len(token_ids)
            hidden = self.embed(token_ids).unsqueeze(0)
            pe = self.position_embeddings(seq, hidden)
            for i in range(self.n_layers):
                t0 = time.perf_counter()
                self.hooks.layer_input(i, hidden)
                layer = self._build_layer(i)
                hidden = self.layer_forward(layer, hidden, pe)
                self.hooks.layer_output(i, hidden)
                del layer
                if self.verbose:
                    print(f"[qwen35moe-ref] layer {i} ({self.arch.layer_types[i]}) {time.perf_counter() - t0:.1f}s", flush=True)
            return hidden
        finally:
            self.hooks = prev_hooks

    @torch.no_grad()
    def final_norm(self, residual: torch.Tensor) -> torch.Tensor:
        """`model.language_model.norm`: [1, T, 3072] -> the lm_head input."""
        if self._final_norm is None:
            from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeRMSNorm

            old = torch.get_default_dtype()
            torch.set_default_dtype(self.dtype)
            try:
                with torch.device("meta"):
                    norm = Qwen3_5MoeRMSNorm(self.arch.hidden, eps=self.arch.eps)
            finally:
                torch.set_default_dtype(old)
            norm.to_empty(device=self.device)
            with torch.no_grad():
                norm.weight.copy_(self.reader.get(FINAL_NORM_NAME).to(self.device, norm.weight.dtype))
            self._final_norm = norm.eval()
        return self._final_norm(residual)

    def forward_hidden(self, token_ids, hooks: ReferenceHooks | None = None) -> torch.Tensor:
        """The lm_head input [T, 3072] after `n_layers` layers (all of them unless max_layers was given)."""
        return self.final_norm(self.run_layers(token_ids, hooks))[0]

    def _lm_head_weight(self) -> torch.Tensor:
        if self._lm_head is None:
            self._lm_head = self.reader.get(LM_HEAD_NAME).to(self.device)
        return self._lm_head

    @torch.no_grad()
    def logits_rows(self, hidden_rows: torch.Tensor, mode: str = "hf-bf16", chunk: int | None = None) -> torch.Tensor:
        """fp32 logits [..., V] of hidden rows [..., 3072]. "hf-bf16": lm_head in bf16 like HF (then widened); "fp32":
        accumulated in fp32. `chunk` limits the vocab rows multiplied at once."""
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
        """Logits [T, V] (fp32) of every position."""
        hidden = self.final_norm(self.run_layers(token_ids, hooks))
        return self.logits_rows(hidden, mode)[0]
