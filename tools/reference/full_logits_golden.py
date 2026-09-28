"""tools/reference/full_logits_golden.py

Runs the ORIGINAL bf16 Qwen3.8-27B checkpoint end to end -- all 64 decoder layers, the final norm
and the full `[248320, 5120]` lm_head -- over a fixed token sequence, and writes per-position
log-probabilities in the shared KL-comparison format (see "Shared file format" in
tools/reference/README.md). `kl_report.py` pairs these with the r4dx engine's own dump of the same
sequence to measure how far the quantized engine has drifted from the bf16 reference.

The 27B bf16 checkpoint is 51.7 GiB on disk -- it fits neither in this card's 32 GiB of VRAM nor
comfortably in 63 GiB of system RAM. So nothing is ever fully resident: the model skeleton is built
on the `meta` device (no storage at all), and `Qwen3_5TextModel.forward` -- transformers' own,
unmodified -- walks a lazy layer list that materializes each `Qwen3_5DecoderLayer`'s real weights
from the safetensors shards onto HIP device 1 just before it runs, and frees them immediately
after. Only the `[T, 5120]` hidden state survives from one layer to the next. The lm_head is
streamed the same way, a row block of the vocabulary at a time. Peak VRAM is ~2-4 GiB, not 52.

Two independent implementations of the *composition* around the layers are provided, and
`--cross-check` runs both and compares their top-5 (see "Validation" in the README):

  --impl model   (default) transformers' own `Qwen3_5TextModel.forward`, with `self.layers`
                 swapped for the lazy streamer. Real `create_causal_mask` /
                 `create_recurrent_attention_mask`, real `Qwen3_5TextRotaryEmbedding`, real layer
                 ordering, real final norm.
  --impl manual  a hand-rolled loop in this file: our own partial-rope cos/sin, our own additive
                 causal mask, our own zero-centered RMSNorm, our own layer ordering. Only the
                 `Qwen3_5DecoderLayer` modules themselves are shared with `--impl model`.

`--weights-gguf <file.gguf>` runs the same forward with a llama.cpp GGUF's quantized weights
dequantized and substituted in (`GGUFWeightsReference`, via gguf_dequant.py), so kl_report.py can
score that GGUF on our own tokens. Weights only: activations stay as in the bf16 reference. Its
output is tagged `"source": "reference-ggufweights"` with a `weights_gguf` record. It needs an explicit
`--out-dir`, and it will not write into a directory holding a bf16 reference (nor a bf16 run into a
GGUF-weights directory).

`--weights-override <dir>` does the same with the weights of an override directory (a
`weights_override.json` manifest written by trellis_quant.py: the EXL3 trellis oracle's packed
bitstreams + fp16 scales, reconstructed in the original basis on the device, or plain dense
tensors), via `OverrideWeightsReference`. Output tagged `"source": "reference-overrideweights"`
with a `weights_override` record (manifest sha256, measured bits/weight, per-class K); the same
explicit-`--out-dir` and no-mixing rules apply.

Usage (reference venv only -- read-only against the venv and the checkpoint):

    $env:HIP_VISIBLE_DEVICES = '1'
    <venv>\\Scripts\\python.exe tools\\reference\\full_logits_golden.py `
        --device cuda --tokens tools\\reference\\kl_corpus\\tokens.json `
        --out-dir tools\\reference\\kl_out\\ref

See tools/reference/README.md for the full option list, the validation runs and the expected
runtime.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    DEFAULT_MODEL_DIR,
    ShardIndex,
    load_ref_tokenizer,
    load_text_config,
    resolve_device,
    sha256_file,
    tokens_file_tokenizer_mode,
)

#: fp16's smallest finite magnitude is ~-65504, so a log-probability below that would become -inf
#: on the cast and poison every downstream sum. Clamp first. Only tokens whose probability is
#: effectively zero (exp(-1e4) is ~0 even in fp64) are affected, so no KL sum moves materially --
#: `kl_report.py` weights each term by p_ref, which is exactly 0 for those tokens.
LOGPROB_CLAMP = -1e4

#: HF weight-name prefixes for this checkpoint (`Qwen3_5ForConditionalGeneration`: the text stack
#: lives under `model.language_model.`, the head is a bare top-level `lm_head.weight`).
TEXT_PREFIX = "model.language_model."
LAYER_PREFIX = TEXT_PREFIX + "layers."
EMBED_NAME = TEXT_PREFIX + "embed_tokens.weight"
FINAL_NORM_NAME = TEXT_PREFIX + "norm.weight"
LM_HEAD_NAME = "lm_head.weight"


def _cuda_sync(device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _cuda_reset_peak(device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def _cuda_peak_gib(device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return torch.cuda.max_memory_allocated() / 2**30, torch.cuda.max_memory_reserved() / 2**30


def token_ids_sha256(token_ids) -> str:
    """The `sha256_of_token_ids_json` sidecar field, defined once here so both halves of the KL
    comparison compute it identically: sha256 of the compact JSON array of the segment's token ids
    (`json.dumps(ids, separators=(",", ":"))`, UTF-8), e.g. `[1,2,3]` -> sha256("[1,2,3]")."""
    payload = json.dumps([int(t) for t in token_ids], separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# --------------------------------------------------------------------------------------------
# Streaming weight access
# --------------------------------------------------------------------------------------------


def get_tensors_grouped(index: ShardIndex, names: list[str]) -> dict[str, torch.Tensor]:
    """`ShardIndex.get_tensor` for many names, opening each shard file once instead of once per
    tensor. Same `.clone()`-inside-the-`with` discipline as `common.get_tensor` (see the
    dangling-mmap comment there) -- the clone is what makes the result safe after the unmap."""
    from safetensors import safe_open

    by_shard: dict[str, list[str]] = {}
    for n in names:
        if n not in index.weight_map:
            raise KeyError(f"{n!r} is not in this checkpoint's model.safetensors.index.json")
        by_shard.setdefault(index.weight_map[n], []).append(n)
    out: dict[str, torch.Tensor] = {}
    for shard, shard_names in by_shard.items():
        with safe_open(str(index.model_dir / shard), framework="pt", device="cpu") as f:
            for n in shard_names:
                out[n] = f.get_tensor(n).clone()
    return out


def gather_embedding_rows(index: ShardIndex, token_ids: list[int], device, dtype) -> torch.Tensor:
    """`embed_tokens(input_ids)` without materializing the `[248320, 5120]` table (2.5 GiB bf16):
    reads just the rows the sequence actually uses, straight out of the shard's mmap."""
    from safetensors import safe_open

    shard = index.weight_map[EMBED_NAME]
    uniq = sorted(set(int(t) for t in token_ids))
    rows: dict[int, torch.Tensor] = {}
    with safe_open(str(index.model_dir / shard), framework="pt", device="cpu") as f:
        sl = f.get_slice(EMBED_NAME)
        shape = sl.get_shape()
        for tid in uniq:
            if not (0 <= tid < shape[0]):
                raise ValueError(f"token id {tid} out of range for embedding table {shape}")
            rows[tid] = sl[tid : tid + 1, :].clone()
    stacked = torch.cat([rows[int(t)] for t in token_ids], dim=0)
    return stacked.to(device=device, dtype=dtype).unsqueeze(0)  # [1, T, hidden]


class LazyDecoderLayers(nn.Module):
    """Stands in for `Qwen3_5TextModel.layers`. `Qwen3_5TextModel.forward` only ever does
    `enumerate(self.layers[: config.num_hidden_layers])`, so a slice that returns a generator of
    freshly-materialized layers is all it takes to stream a model that does not fit in memory."""

    def __init__(self, builder, n_layers: int):
        super().__init__()
        self._builder = builder
        self._n = n_layers
        self.load_seconds = 0.0
        self.layers_run = 0

    def __len__(self) -> int:
        return self._n

    def __getitem__(self, key):
        if isinstance(key, slice):
            return _LazyLayerIter(self, range(*key.indices(self._n)))
        return self._build(int(key))

    def __iter__(self):
        return iter(_LazyLayerIter(self, range(self._n)))

    def _build(self, i: int):
        t0 = time.perf_counter()
        layer = self._builder(i)
        self.load_seconds += time.perf_counter() - t0
        self.layers_run += 1
        return layer


class _LazyLayerIter:
    def __init__(self, owner: LazyDecoderLayers, rng):
        self._owner = owner
        self._rng = rng

    def __len__(self):
        return len(self._rng)

    def __iter__(self):
        for i in self._rng:
            layer = self._owner._build(i)
            yield layer
            del layer  # last strong reference: the layer's ~750 MiB of VRAM is freed here


# --------------------------------------------------------------------------------------------
# The streaming reference model
# --------------------------------------------------------------------------------------------


class StreamingReference:
    #: The `source` field of every `.meta.json` this reference writes (kl_report.py's shared format).
    source_tag = "reference"

    def __init__(self, model_dir: Path, device, dtype=torch.bfloat16, max_layers: int | None = None,
                 verbose: bool = True):
        import transformers.models.qwen3_5.modeling_qwen3_5 as modeling

        self.m = modeling
        self.model_dir = model_dir
        self.device = device
        self.dtype = dtype
        self.verbose = verbose
        self.index = ShardIndex.load(model_dir)
        if not self.index.weight_map:
            raise RuntimeError(f"no safetensors index/shards found under {model_dir}")
        self.config, self.text_config = load_text_config(model_dir)  # forces _attn_implementation=eager
        self.max_layers = max_layers
        self.n_layers = self.text_config.num_hidden_layers if max_layers is None else max_layers
        self.layer_types = list(self.text_config.layer_types)

        with torch.device("meta"):
            self.model = modeling.Qwen3_5TextModel(self.text_config)
        self.lazy = LazyDecoderLayers(self.build_layer, self.n_layers)
        del self.model.layers  # drop the meta ModuleList so nn.Module lets a non-ModuleList in
        self.model.layers = self.lazy
        # The final norm and the rope module are tiny -- keep them resident, real, for the whole run.
        norm_w = self.index.get_tensor(FINAL_NORM_NAME).to(device=device, dtype=dtype)
        self.model.norm.load_state_dict({"weight": norm_w}, assign=True)
        self.model.rotary_emb = modeling.Qwen3_5TextRotaryEmbedding(self.text_config).to(device=device)
        self.model.embed_tokens = None  # never called: we always pass inputs_embeds
        self.model.eval()
        # `num_hidden_layers` drives the forward's `self.layers[: n]` slice.
        self.model.config.num_hidden_layers = self.n_layers

        with safe_open_slice(self.index, LM_HEAD_NAME) as (shape, _sl):
            self.vocab_size, lm_hidden = int(shape[0]), int(shape[1])
        if lm_hidden != self.text_config.hidden_size:
            raise RuntimeError(f"lm_head hidden {lm_hidden} != config hidden {self.text_config.hidden_size}")
        if self.vocab_size != self.text_config.vocab_size and verbose:
            print(f"[full_logits] NOTE: lm_head rows {self.vocab_size} != config.vocab_size "
                  f"{self.text_config.vocab_size}; using the lm_head's own row count as V")

    # -- weights ------------------------------------------------------------------------------

    def build_layer(self, i: int):
        with torch.device("meta"):
            layer = self.m.Qwen3_5DecoderLayer(self.text_config, i)
        prefix = f"{LAYER_PREFIX}{i}."
        keys = list(layer.state_dict().keys())
        layer.load_state_dict(self.layer_state(prefix, keys), assign=True)
        layer.eval()
        for p in layer.parameters():
            p.requires_grad_(False)
        return layer

    def layer_state(self, prefix: str, keys: list[str]) -> dict[str, torch.Tensor]:
        """The decoder layer's state dict (`keys` under the checkpoint name `prefix`), on the device
        in `self.dtype`. `GGUFWeightsReference` overrides this to substitute GGUF weights."""
        raw = get_tensors_grouped(self.index, [prefix + k for k in keys])
        return {k: raw[prefix + k].to(device=self.device, dtype=self.dtype) for k in keys}

    def segment_meta(self) -> dict:
        """Extra fields for each segment's `.meta.json`. The layer count lets kl_report.py refuse to
        score a truncated (--debug-max-layers) dump against a full one."""
        return {"n_layers": self.n_layers, "debug_max_layers": self.max_layers}

    def embed(self, token_ids: list[int]) -> torch.Tensor:
        return gather_embedding_rows(self.index, token_ids, self.device, self.dtype)

    # -- forward ------------------------------------------------------------------------------

    def forward_hidden(self, token_ids: list[int], impl: str = "model") -> torch.Tensor:
        """Full text stack over `token_ids` from a fresh context (position 0, causal, no cache).
        Returns the post-final-norm hidden states `[T, hidden]` (bf16, on device)."""
        embeds = self.embed(token_ids)
        if impl == "model":
            with torch.no_grad():
                out = self.model(inputs_embeds=embeds, use_cache=False)
            return out.last_hidden_state[0]
        if impl == "manual":
            return self._forward_manual(embeds)
        raise ValueError(f"unknown --impl {impl!r}")

    # -- the independent composition (cross-check) --------------------------------------------

    def _manual_rope(self, total_len: int):
        """Partial rope cos/sin, derived here from the config rather than from
        `Qwen3_5TextRotaryEmbedding`. Text-only: the three mrope axes (t, h, w) all carry the same
        position index, and `recomposition_frequencies` interleaves h/w sections *from tensors that
        are elementwise identical to the t tensor*, so the recomposition is the identity and this
        reduces to plain rope over the first `head_dim * partial_rotary_factor` dims."""
        rp = self.text_config.rope_parameters
        theta = float(rp["rope_theta"])
        head_dim = self.text_config.head_dim
        rot_dim = int(head_dim * float(rp.get("partial_rotary_factor", 1.0)))
        inv_freq = 1.0 / (theta ** (torch.arange(0, rot_dim, 2, dtype=torch.float32, device=self.device) / rot_dim))
        pos = torch.arange(total_len, dtype=torch.float32, device=self.device)
        freqs = pos[:, None] * inv_freq[None, :]            # [T, rot_dim/2]
        cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)  # [T, rot_dim]
        sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
        return cos.to(self.dtype).unsqueeze(0), sin.to(self.dtype).unsqueeze(0)

    def _manual_mask(self, total_len: int) -> torch.Tensor:
        q = torch.arange(total_len, device=self.device).unsqueeze(1)
        k = torch.arange(total_len, device=self.device).unsqueeze(0)
        mask = torch.zeros(total_len, total_len, dtype=self.dtype, device=self.device)
        mask = mask.masked_fill(k > q, torch.finfo(self.dtype).min)
        return mask.view(1, 1, total_len, total_len)

    def _manual_final_norm(self, x: torch.Tensor) -> torch.Tensor:
        """Qwen3_5's zero-centered RMSNorm, written out here: fp32 normalize, scale by (1 + w)."""
        w = self.model.norm.weight.float()
        xf = x.float()
        normed = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.text_config.rms_norm_eps)
        return (normed * (1.0 + w)).to(x.dtype)

    def _forward_manual(self, embeds: torch.Tensor) -> torch.Tensor:
        total_len = embeds.shape[1]
        cos, sin = self._manual_rope(total_len)
        mask = self._manual_mask(total_len)
        h = embeds
        with torch.no_grad():
            for i in range(self.n_layers):
                layer = self.build_layer(i)
                attn_mask = None if self.layer_types[i] == "linear_attention" else mask
                h = layer(hidden_states=h, position_embeddings=(cos, sin), attention_mask=attn_mask,
                          position_ids=None, past_key_values=None)
                del layer
        return self._manual_final_norm(h)[0]

    # -- lm_head ------------------------------------------------------------------------------

    def logits_fp32(self, hidden: torch.Tensor, chunk: int = 32768) -> torch.Tensor:
        """`[R, hidden] -> [R, V]` fp32, streaming the lm_head `chunk` vocabulary rows at a time.
        The per-chunk matmul is done in bf16, exactly as `nn.Linear` does inside the real
        `Qwen3_5ForConditionalGeneration.forward`; only the accumulation buffer is fp32."""
        rows = hidden.shape[0]
        out = torch.empty(rows, self.vocab_size, device=self.device, dtype=torch.float32)
        for start, stop, w in self.lm_head_blocks(chunk):
            out[:, start:stop] = torch.nn.functional.linear(hidden, w).float()
            del w
        return out

    def lm_head_blocks(self, chunk: int):
        """Yields `(start, stop, lm_head[start:stop])` on the device in `self.dtype`. The block is
        never bound to a local here, so the caller's `del` frees it."""
        from safetensors import safe_open

        shard = self.index.weight_map[LM_HEAD_NAME]
        with safe_open(str(self.index.model_dir / shard), framework="pt", device="cpu") as f:
            sl = f.get_slice(LM_HEAD_NAME)
            for start in range(0, self.vocab_size, chunk):
                stop = min(start + chunk, self.vocab_size)
                yield start, stop, sl[start:stop, :].clone().to(device=self.device, dtype=self.dtype)


class safe_open_slice:
    """`with safe_open_slice(index, name) as (shape, sl):` -- shape lookup without a full read."""

    def __init__(self, index: ShardIndex, name: str):
        self.index, self.name = index, name

    def __enter__(self):
        from safetensors import safe_open

        self._f = safe_open(str(self.index.model_dir / self.index.weight_map[self.name]),
                            framework="pt", device="cpu")
        self._f.__enter__()
        sl = self._f.get_slice(self.name)
        return sl.get_shape(), sl

    def __exit__(self, *exc):
        return self._f.__exit__(*exc)


# --------------------------------------------------------------------------------------------
# A llama.cpp GGUF's weights in the reference forward (--weights-gguf)
# --------------------------------------------------------------------------------------------


def _err_rows(deq: torch.Tensor, w: torch.Tensor, step: int = 1 << 22):
    """Error sums of two CPU tensors of equal size (any float dtypes), seen as [shape[0], rest] (a
    1D tensor is one row). Returns ([sum (deq-w)^2, sum w^2, sum deq^2, sum deq*w], per-row
    sum (deq-w)^2, per-row sum w^2), the per-row ones as float64 numpy. fp32 products over row
    blocks of ~`step` elements, summed in fp64."""
    if deq.numel() != w.numel():
        raise ValueError(f"size mismatch {tuple(deq.shape)} vs {tuple(w.shape)}")
    a = deq.reshape(deq.shape[0], -1) if deq.dim() > 1 else deq.reshape(1, -1)
    b = w.reshape(a.shape)
    n = a.shape[0]
    rows = max(1, step // max(1, a.shape[1]))
    sums = [0.0, 0.0, 0.0, 0.0]
    row_sse, row_ssw = np.empty(n), np.empty(n)
    f64 = torch.float64
    for r in range(0, n, rows):
        x = a[r:r + rows].float()
        y = b[r:r + rows].float()
        d = x - y
        rs, rw = (d * d).sum(dim=1, dtype=f64), (y * y).sum(dim=1, dtype=f64)
        row_sse[r:r + rows], row_ssw[r:r + rows] = rs.numpy(), rw.numpy()
        sums[0] += float(rs.sum())
        sums[1] += float(rw.sum())
        sums[2] += float((x * x).sum(dtype=f64))
        sums[3] += float((x * y).sum(dtype=f64))
    return sums, row_sse, row_ssw


def _rel_cos(sums: list[float]) -> tuple[float, float]:
    sse, ssw, ssd, dot = sums
    rel = (sse / ssw) ** 0.5 if ssw > 0 else (0.0 if sse == 0 else float("inf"))
    den = (ssw * ssd) ** 0.5
    return rel, (dot / den if den > 0 else (1.0 if sse == 0 else 0.0))


class GGUFWeightsReference(StreamingReference):
    """`StreamingReference` with a llama.cpp GGUF's quantized weights substituted in (`--weights-gguf`).

    For every checkpoint tensor the forward reads (layer weights, the embedding rows, the final norm,
    the lm_head):

      * the GGUF stores it in a lossy type (anything but F32/BF16): replaced by the GGUF's value,
        dequantized in float32 by gguf_dequant.py, put back in the HF layout, then rounded to bf16 like
        every other weight of the reference. On first use it is compared with the bf16 value and the
        run aborts if rel error or cosine is outside gguf_dequant.WEIGHT_ERROR_BOUNDS. Each row is
        measured on its own too (for the embedding: each row the run uses), because the pooled
        figure hides one bad row, and rows above gguf_dequant.ROW_ERROR_FACTOR x the type bound
        are recorded. For the embedding and the lm_head, where one row is one token, such a used row
        aborts the run unless its f16 block scales are all coarse (gguf_dequant.F16_COARSE_SCALE:
        a row of ~1e-5 values, the file's own limit) or --gguf-allow-row-outliers is given.
        gguf_validate.py lists every such row of every tensor and re-decodes it with ggml's C.
      * the GGUF stores it losslessly (F32: the norms, A_log, dt_bias, conv1d, the GDN norm): the
        checkpoint's value, after checking it equals the GGUF's bit for bit (-exp(A_log): 1 ulp).
        Keeping it is then the same as substituting it; a mismatch means the GGUF is of another
        model, and the run aborts.
      * the GGUF lacks it: the checkpoint's value.

    Activations, the attention/GDN math, the norms and the lm_head matmul are the reference's own,
    unchanged, so the logits measure weight quantization only.
    """

    source_tag = "reference-ggufweights"

    def __init__(self, model_dir: Path, device, gguf_path: Path, dtype=torch.bfloat16,
                 max_layers: int | None = None, verbose: bool = True, threads: int = 8,
                 full_sha256: bool = False, allow_row_outliers: bool = False):
        import gguf_dequant as gd

        self.gd = gd
        self.gguf_path = Path(gguf_path)
        self.gguf = gd.GGUFFile(self.gguf_path)
        self.qmap = gd.Qwen35Map.from_gguf(self.gguf)
        self.threads = max(1, int(threads))
        self.allow_row_outliers = allow_row_outliers
        self.tensor_stats: dict[str, dict] = {}     # substituted: hf name -> gguf, type, rel, cos
        self.lossless_checks: dict[str, dict] = {}  # kept after an exact check: hf name -> ...
        self._shapes: dict[str, tuple[int, ...]] = {}
        self._embed_sums = [0.0, 0.0, 0.0, 0.0]
        self._embed_rows_checked: set[int] = set()
        self.dequant_seconds = 0.0
        super().__init__(model_dir, device, dtype=dtype, max_layers=max_layers, verbose=verbose)
        self._check_geometry()

        needed = [n for n in self.index.weight_map
                  if (li := self._layer_of(n)) is not None and li < self.n_layers]
        needed += [EMBED_NAME, FINAL_NORM_NAME, LM_HEAD_NAME]
        self.plan = {n: self._kind(n) for n in needed}
        self.gguf.check_decodable([self.qmap.gguf_of[n][0] for n, k in self.plan.items() if k != "absent"])
        for name in (EMBED_NAME, LM_HEAD_NAME):
            if self.plan[name] != "absent" and self.qmap.gguf_of[name][1] != "id":
                raise SystemExit(f"[full_logits] {name}: GGUF transform {self.qmap.gguf_of[name][1]!r} "
                                 "is not row-local; only 'id' is supported for the embedding / lm_head")
        norm_kind = self.plan[FINAL_NORM_NAME]
        if norm_kind != "absent":
            w = self.index.get_tensor(FINAL_NORM_NAME)
            if norm_kind == "lossless":
                self._verify_lossless(FINAL_NORM_NAME, w)
            else:
                self.model.norm.load_state_dict({"weight": self._substitute(FINAL_NORM_NAME, w)},
                                                assign=True)

        st = self.gguf_path.stat()
        g = self.gguf
        self.file_record = {
            "path": str(self.gguf_path.resolve()),
            "size_bytes": st.st_size,
            "mtime_utc": dt.datetime.fromtimestamp(st.st_mtime, tz=dt.timezone.utc).isoformat(),
            "header_sha256": g.header_sha256,
            "sha256": None,
            "gguf_version": g.version,
            "general.architecture": g.meta("general.architecture"),
            "general.name": g.meta("general.name"),
            "general.file_type": g.meta("general.file_type"),
            "general.quantized_by": g.meta("general.quantized_by"),
            "general.quantization_version": g.meta("general.quantization_version"),
            "quantize.imatrix.file": g.meta("quantize.imatrix.file"),
        }
        if full_sha256:
            t0 = time.perf_counter()
            self.file_record["sha256"] = g.sha256()
            if verbose:
                print(f"[full_logits] GGUF sha256 {self.file_record['sha256']} "
                      f"({time.perf_counter() - t0:.0f}s)", flush=True)
        if verbose:
            p = self.planned()
            print(f"[full_logits] --weights-gguf {self.gguf_path.name}: substituting "
                  f"{p['substitute']['count']} tensors {p['substitute']['by_type']}, keeping "
                  f"{p['lossless']['count']} lossless ones after an exact check, "
                  f"{len(p['absent'])} absent from the GGUF", flush=True)

    # -- plan -------------------------------------------------------------------------------------

    @staticmethod
    def _layer_of(name: str) -> int | None:
        if not name.startswith(LAYER_PREFIX):
            return None
        head = name[len(LAYER_PREFIX):].split(".", 1)[0]
        return int(head) if head.isdigit() else None

    def _kind(self, name: str) -> str:
        hit = self.qmap.gguf_of.get(name)
        if hit is None:
            return "absent"
        return "lossless" if self.gguf.tensor(hit[0]).type_name in self.gd.LOSSLESS_TYPES else "substitute"

    def _check_geometry(self) -> None:
        tc, q, g = self.text_config, self.qmap, self.gguf
        pairs = [  # (the base class has cut config.num_hidden_layers to --debug-max-layers by now)
            ("decoder layers", len(self.layer_types), q.n_trunk),
            ("hidden size", tc.hidden_size, g.meta("qwen35.embedding_length")),
            ("GDN key heads", tc.linear_num_key_heads, q.nk),
            ("GDN value heads", tc.linear_num_value_heads, q.nv),
            ("GDN key head dim", tc.linear_key_head_dim, q.dk),
            ("GDN value head dim", tc.linear_value_head_dim, q.dv),
            ("vocab (lm_head rows)", self.vocab_size, g.tensor("output.weight").shape[0]),
        ]
        bad = [f"{what}: checkpoint {a} vs GGUF {b}" for what, a, b in pairs if int(a) != int(b)]
        if bad:
            raise SystemExit("[full_logits] the GGUF does not match the checkpoint: " + "; ".join(bad))

    def planned(self) -> dict:
        sub = [n for n, k in self.plan.items() if k == "substitute"]
        by_type: dict[str, int] = {}
        for n in sub:
            t = self.gguf.tensor(self.qmap.gguf_of[n][0]).type_name
            by_type[t] = by_type.get(t, 0) + 1
        lossless = [n for n, k in self.plan.items() if k == "lossless"]
        by_tag: dict[str, int] = {}
        for n in lossless:
            tag = self.qmap.gguf_of[n][1]
            by_tag[tag] = by_tag.get(tag, 0) + 1
        return {"substitute": {"count": len(sub), "by_type": dict(sorted(by_type.items(), key=lambda kv: -kv[1]))},
                "lossless": {"count": len(lossless), "by_tag": by_tag},
                "absent": sorted(n for n, k in self.plan.items() if k == "absent")}

    # -- weights ----------------------------------------------------------------------------------

    def _record(self, name: str, gname: str, sums: list[float], row_sse: np.ndarray, row_ssw: np.ndarray,
                rows=None, **extra) -> None:
        """Check and record a substituted tensor: `sums` are the tensor-wide error sums so far,
        `row_sse` / `row_ssw` the per-row ones of the rows checked in THIS call (HF row ids `rows`,
        default 0..n-1). The embedding calls this again for each segment's new rows."""
        tname = self.gguf.tensor(gname).type_name
        rel, cos = _rel_cos(sums)
        bounds = self.gd.WEIGHT_ERROR_BOUNDS.get(tname)
        if bounds is None:
            raise SystemExit(f"[full_logits] no error bounds for GGUF type {tname} ({gname})")
        prev = self.tensor_stats.get(name, {})
        row_rel = self.gd.row_rel_errors(row_sse, row_ssw)
        ids = np.arange(row_rel.size) if rows is None else np.asarray(rows, dtype=np.int64)
        st = {"gguf": gname, "type": tname, "class": self.gd.tensor_class(name), "rel": rel, "cos": cos,
              "row_rel_max": prev.get("row_rel_max", -1.0), "row_rel_max_row": prev.get("row_rel_max_row"),
              "row_outliers": list(prev.get("row_outliers", [])), **extra}
        if row_rel.size and float(row_rel.max()) > st["row_rel_max"]:
            i = int(np.argmax(row_rel))
            st["row_rel_max"], st["row_rel_max_row"] = float(row_rel[i]), int(ids[i])
        new = self.gd.row_outliers(self.gguf, self.qmap, gname, self.qmap.gguf_of[name][1], row_rel,
                                   np.sqrt(row_ssw), ids)
        st["row_outliers"] += new
        self.tensor_stats[name] = st
        # The embedding is checked on the rows used so far: a sample, maybe of a handful of rows,
        # and single ordinary rows reach 0.106 against the Q4_K tensor bound of 0.100. So its pooled
        # rel is held to the per-row bound; its whole-tensor figure is gguf_validate.py's.
        max_rel, min_cos = bounds
        if name == EMBED_NAME:
            max_rel, min_cos = self.gd.row_error_bound(tname), -1.0
        if rel > max_rel or cos < min_cos:
            raise SystemExit(
                f"[full_logits] GGUF WEIGHT CHECK FAILED: {name} <- {gname} ({tname}) has rel error "
                f"{rel:.4f} (max {max_rel}) and cosine {cos:.6f} (min {min_cos}) against the bf16 "
                "checkpoint -- a mapping/layout error, or a GGUF of a different model")
        row_local = name in (EMBED_NAME, LM_HEAD_NAME)  # one row = one token's embedding / logit
        for o in new:
            what = (f"{name} <- {gname} ({tname}) row {o['row']} (GGUF row {o['gguf_row']}) has rel error "
                    f"{o['rel']:.4f} (per-row bound {self.gd.row_error_bound(tname):.3f}), ||w|| "
                    f"{o['norm']:.3g}, max |f16 scale d| {o['max_abs_scale']}")
            if row_local and not o["coarse_scale"] and not self.allow_row_outliers:
                raise SystemExit(
                    f"[full_logits] GGUF ROW CHECK FAILED: {what}, used by this run. Its scales are not "
                    "coarse, so this is a decode/mapping error until shown otherwise: run gguf_validate.py "
                    "(it re-decodes the row with ggml's C). If the row is the file's own, rerun with "
                    "--gguf-allow-row-outliers")
            if self.verbose and row_local:
                print(f"[full_logits] NOTE: {what}, used by this run: "
                      + ("its f16 block scales are all coarse (a row of ~1e-5 values), so this is the "
                         "file's own value" if o["coarse_scale"] else "allowed by --gguf-allow-row-outliers"),
                      flush=True)

    def _verify_lossless(self, name: str, w: torch.Tensor) -> None:
        if name in self.lossless_checks:
            return
        gname, tag = self.qmap.gguf_of[name]
        ok, ulp = self.gd.lossless_matches(self.qmap, tag, self.gguf.dequantize(gname), w.float().numpy())
        self.lossless_checks[name] = {"gguf": gname, "type": self.gguf.tensor(gname).type_name,
                                      "tag": tag, "max_ulp": ulp}
        if not ok:
            raise SystemExit(f"[full_logits] GGUF {gname} is stored losslessly but differs from the "
                             f"checkpoint's {name} by {ulp} ulp: the GGUF is not of this checkpoint")

    def _substitute(self, name: str, w: torch.Tensor | None) -> torch.Tensor:
        """The GGUF's value of checkpoint tensor `name`, bf16 on the device. `w` (the bf16 value, CPU)
        is needed on first use, for the shape and the error check."""
        gname, _tag = self.qmap.gguf_of[name]
        if w is not None:
            self._shapes[name] = tuple(w.shape)
        t0 = time.perf_counter()
        deq = torch.from_numpy(self.qmap.hf_tensor(self.gguf, name, self._shapes[name], threads=self.threads))
        self.dequant_seconds += time.perf_counter() - t0
        if name not in self.tensor_stats:
            if w is None:
                raise RuntimeError(f"{name}: first substitution needs the checkpoint value")
            self._record(name, gname, *_err_rows(deq, w))
        return deq.to(dtype=self.dtype).to(device=self.device)

    def layer_state(self, prefix: str, keys: list[str]) -> dict[str, torch.Tensor]:
        names = [prefix + k for k in keys]
        # The bf16 value of a substituted tensor is only read the first time (for its check).
        read = [n for n in names if self._kind(n) != "substitute" or n not in self.tensor_stats]
        raw = get_tensors_grouped(self.index, read)
        sd = {}
        for k, n in zip(keys, names):
            kind = self._kind(n)
            if kind == "substitute":
                sd[k] = self._substitute(n, raw.pop(n, None))
            else:
                if kind == "lossless":
                    self._verify_lossless(n, raw[n])
                sd[k] = raw.pop(n).to(device=self.device, dtype=self.dtype)
        return sd

    def _embedding_rows_bf16(self, ids: list[int]) -> torch.Tensor:
        from safetensors import safe_open

        with safe_open(str(self.index.model_dir / self.index.weight_map[EMBED_NAME]),
                       framework="pt", device="cpu") as f:
            sl = f.get_slice(EMBED_NAME)
            return torch.cat([sl[t:t + 1, :].clone() for t in ids], dim=0)

    def embed(self, token_ids: list[int]) -> torch.Tensor:
        kind = self.plan[EMBED_NAME]
        uniq = sorted(set(int(t) for t in token_ids))
        if kind != "substitute":
            if kind == "lossless":  # exact check on the rows this sequence uses
                gname = self.qmap.gguf_of[EMBED_NAME][0]
                got = self.gguf.dequantize_row_ids(gname, uniq)
                if not np.array_equal(got, self._embedding_rows_bf16(uniq).float().numpy()):
                    raise SystemExit(f"[full_logits] GGUF {gname} differs from the checkpoint's embedding")
            return super().embed(token_ids)
        gname = self.qmap.gguf_of[EMBED_NAME][0]
        t0 = time.perf_counter()
        deq = torch.from_numpy(self.gguf.dequantize_row_ids(gname, uniq))  # [n_uniq, hidden] fp32
        self.dequant_seconds += time.perf_counter() - t0
        new = [i for i, t in enumerate(uniq) if t not in self._embed_rows_checked]
        if new:  # the embedding's check covers every row used so far, pooled and one by one
            new_ids = [uniq[i] for i in new]
            sums, row_sse, row_ssw = _err_rows(deq[new], self._embedding_rows_bf16(new_ids))
            self._embed_sums = [a + b for a, b in zip(self._embed_sums, sums)]
            self._embed_rows_checked.update(new_ids)
            self._record(EMBED_NAME, gname, self._embed_sums, row_sse, row_ssw, rows=new_ids,
                         rows_checked=len(self._embed_rows_checked))
        pos = {t: i for i, t in enumerate(uniq)}
        rows = deq[torch.tensor([pos[int(t)] for t in token_ids], dtype=torch.long)]
        return rows.to(dtype=self.dtype).to(device=self.device).unsqueeze(0)

    def lm_head_blocks(self, chunk: int):
        import contextlib

        kind = self.plan[LM_HEAD_NAME]
        if kind != "substitute":
            gname = self.qmap.gguf_of[LM_HEAD_NAME][0] if kind == "lossless" else None
            for start, stop, w in super().lm_head_blocks(chunk):
                if gname is not None and LM_HEAD_NAME not in self.lossless_checks:
                    got = self.gguf.dequantize_rows(gname, start, stop, threads=self.threads)
                    if not np.array_equal(got, w.float().cpu().numpy()):
                        raise SystemExit(f"[full_logits] GGUF {gname} differs from the checkpoint's lm_head")
                yield start, stop, w
            if gname is not None:
                self.lossless_checks[LM_HEAD_NAME] = {"gguf": gname, "type": self.gguf.tensor(gname).type_name,
                                                      "tag": "id", "max_ulp": 0}
            return
        gname = self.qmap.gguf_of[LM_HEAD_NAME][0]
        check = LM_HEAD_NAME not in self.tensor_stats
        sums = [0.0, 0.0, 0.0, 0.0]
        row_sse, row_ssw = np.zeros(self.vocab_size), np.zeros(self.vocab_size)
        with contextlib.ExitStack() as stack:
            sl = None
            if check:
                sl = stack.enter_context(safe_open_slice(self.index, LM_HEAD_NAME))[1]
            for start in range(0, self.vocab_size, chunk):
                stop = min(start + chunk, self.vocab_size)
                t0 = time.perf_counter()
                deq = torch.from_numpy(self.gguf.dequantize_rows(gname, start, stop, threads=self.threads))
                self.dequant_seconds += time.perf_counter() - t0
                if check:
                    s, row_sse[start:stop], row_ssw[start:stop] = _err_rows(deq, sl[start:stop, :].clone())
                    sums = [a + b for a, b in zip(sums, s)]
                yield start, stop, deq.to(dtype=self.dtype).to(device=self.device)
                del deq
        if check:
            self._record(LM_HEAD_NAME, gname, sums, row_sse, row_ssw)

    # -- the record -------------------------------------------------------------------------------

    def weight_error_by_type(self) -> dict:
        by: dict[str, list[dict]] = {}
        for s in self.tensor_stats.values():
            by.setdefault(s["type"], []).append(s)
        out = {}
        for t, rs in sorted(by.items()):
            rels = sorted(r["rel"] for r in rs)
            out[t] = {"n": len(rs), "rel_median": rels[len(rels) // 2], "rel_max": rels[-1],
                      "cos_min": min(r["cos"] for r in rs),
                      "row_rel_max": max(r["row_rel_max"] for r in rs)}
        return out

    def row_outliers(self) -> dict:
        """hf name -> the rows above the per-row bound that the run used (see _record)."""
        return {n: s["row_outliers"] for n, s in self.tensor_stats.items() if s.get("row_outliers")}

    def segment_meta(self) -> dict:
        p = self.planned()
        return {**super().segment_meta(), "weights_gguf": {
            **{k: self.file_record[k] for k in ("path", "size_bytes", "mtime_utc", "header_sha256", "sha256")},
            "substituted_by_type": p["substitute"]["by_type"],
            "substituted_count": p["substitute"]["count"],
            "weights": "GGUF-dequantized for every tensor the GGUF quantized, bf16 checkpoint otherwise; "
                       "activations as in the bf16 reference (weight-only quantization)",
        }}

    def weights_record(self) -> dict:
        """The `weights_gguf` block of reference_run.json."""
        p = self.planned()
        by_class: dict[str, int] = {}
        for s in self.tensor_stats.values():
            by_class[s["class"]] = by_class.get(s["class"], 0) + 1
        ulps = [c["max_ulp"] for c in self.lossless_checks.values()]
        return {
            **self.file_record,
            "rule": ("every checkpoint tensor the forward reads is replaced by the GGUF's value when the "
                     "GGUF stores it in a lossy type (dequantized in fp32 by gguf_dequant.py, inverse "
                     "layout transform, rounded to bf16); tensors the GGUF stores as F32/BF16 are checked "
                     "equal to the checkpoint and kept; tensors the GGUF lacks are kept from the checkpoint"),
            "planned": p,
            "substituted": {"count": len(self.tensor_stats),
                            "by_type": dict(sorted(((t, v["n"]) for t, v in self.weight_error_by_type().items()),
                                                   key=lambda kv: -kv[1])),
                            "by_class": dict(sorted(by_class.items()))},
            "kept_lossless_verified": {"count": len(self.lossless_checks),
                                       "max_ulp": max(ulps) if ulps else None,
                                       "by_tag": {t: sum(1 for c in self.lossless_checks.values() if c["tag"] == t)
                                                  for t in sorted({c["tag"] for c in self.lossless_checks.values()})}},
            "kept_absent_from_gguf": p["absent"],
            "weight_error_bounds": self.gd.WEIGHT_ERROR_BOUNDS,
            "row_error_factor": self.gd.ROW_ERROR_FACTOR,
            "weight_error_by_type": self.weight_error_by_type(),
            "row_outliers_used": self.row_outliers(),
            "per_tensor": self.tensor_stats,
            "allow_row_outliers": self.allow_row_outliers,
            "dequant_threads": self.threads,
            "dequant_seconds": self.dequant_seconds,
            "note": ("weight-only: activations, attention/GDN math and the lm_head matmul are the bf16 "
                     "reference's. llama.cpp itself also quantizes activations inside its matmuls "
                     "(Q8_1/Q8_K) and keeps an f16 KV cache, so its own end-to-end KL is somewhat higher"),
        }


# --------------------------------------------------------------------------------------------
# An override directory's weights in the reference forward (--weights-override)
# --------------------------------------------------------------------------------------------


def override_rel_bound(K: float) -> float:
    """Loosest plausible relative weight error of a trellis tensor at K bits: 2^-(K-2) (0.25 at
    K = 4). LDLQ trades weight error for output error, so the measured values sit well inside it
    (docs/trellis.md; the manifest records each tensor's own); a layout or mapping error is ~1.4."""
    return 2.0 ** (-(float(K) - 2.0))


class OverrideWeightsReference(StreamingReference):
    """`StreamingReference` with the weights of an override directory substituted in
    (`--weights-override`): every checkpoint tensor its `weights_override.json` lists is replaced by
    the directory's value -- a `trellis-exl3` entry is reconstructed on the device from its stored
    bits exactly as trellis_quant.reconstruct does (diag(suh) P_k decode(words) P_n diag(svh),
    transposed to the HF [out, in] layout), a `dense` entry is read as is -- then rounded to bf16
    like every other weight. Everything else is the checkpoint's. Weight-only, like --weights-gguf.

    Checks, so a wrong directory cannot produce a plausible-looking KL: the manifest must be
    complete for the layers this run uses and name only decoder-layer weights of this checkpoint
    (same config.json sha256); on first use each tensor's shape must match and its relative error
    against the bf16 value must be under `override_rel_bound(K)` with cosine >= 0.9, AND within
    2% of the error trellis_quant.py recorded when it wrote the bits."""

    source_tag = "reference-overrideweights"

    def __init__(self, model_dir: Path, device, override_dir: Path, dtype=torch.bfloat16,
                 max_layers: int | None = None, verbose: bool = True):
        import trellis_quant as tq

        self.tq = tq
        self.manifest = tq.load_manifest(Path(override_dir))
        self.manifest_path = Path(self.manifest["_path"])
        self.manifest_sha256 = sha256_file(self.manifest_path)
        self.tensor_stats: dict[str, dict] = {}
        self.reconstruct_seconds = 0.0
        self._cb_cache: dict = {}
        super().__init__(model_dir, device, dtype=dtype, max_layers=max_layers, verbose=verbose)
        cfg_sha = sha256_file(Path(model_dir) / "config.json")
        if self.manifest.get("config_sha256") not in (None, cfg_sha):
            raise SystemExit(f"[full_logits] {self.manifest_path} was made for another checkpoint "
                             f"(config.json sha256 {self.manifest['config_sha256'][:16]}... vs {cfg_sha[:16]}...)")
        # A fine-tune (e.g. an abliterated model) shares the base's config.json byte for byte, so
        # the sha above cannot tell them apart: the manifest's own checkpoint path must match too.
        made_from = self.manifest.get("model_dir")
        if made_from is not None and (os.path.normcase(os.path.abspath(made_from))
                                      != os.path.normcase(os.path.abspath(str(model_dir)))):
            raise SystemExit(f"[full_logits] {self.manifest_path} was quantized from {made_from}, "
                             f"not --model-dir {model_dir}; pass the same --model-dir")
        names = list(self.manifest["tensors"])
        bad = [n for n in names if n not in self.index.weight_map or self._layer_of(n) is None]
        if bad:
            raise SystemExit(f"[full_logits] --weights-override: {len(bad)} manifest entries are not "
                             f"decoder-layer weights of this checkpoint (only those are supported): {bad[:5]}")
        self.plan = sorted(n for n in names if self._layer_of(n) < self.n_layers)
        missing_files = sorted({str(tq.tensor_file(self.manifest, self.manifest["tensors"][n]))
                                for n in self.plan
                                if not tq.tensor_file(self.manifest, self.manifest["tensors"][n]).exists()})
        if missing_files:
            raise SystemExit(f"[full_logits] --weights-override: missing files {missing_files[:5]}")
        if not self.manifest.get("complete", True):
            # An unfinished quantize-model run: usable only for layers it has fully written.
            listed = set(names)
            required = [tq.hf_name(i, m) for i in range(self.n_layers)
                        for m in tq.module_list(self.layer_types[i])]
            lacking = [n for n in required if n not in listed]
            if lacking:
                raise SystemExit(f"[full_logits] {self.manifest_path} is incomplete (quantize-model has not "
                                 f"finished) and lacks {len(lacking)} of the {len(required)} linears this "
                                 f"{self.n_layers}-layer run needs, e.g. {lacking[:3]}")
        if verbose:
            s = self.manifest.get("summary", {})
            print(f"[full_logits] --weights-override {self.manifest_path}: substituting {len(self.plan)} "
                  f"tensors ({self.manifest.get('encoding')}, {s.get('bpw') or float('nan'):.4f} bpw "
                  f"measured over the manifest, K per class "
                  f"{ {c: v['K'] for c, v in s.get('by_class', {}).items()} })", flush=True)

    @staticmethod
    def _layer_of(name: str) -> int | None:
        return GGUFWeightsReference._layer_of(name)

    def _check(self, name: str, w_hat: torch.Tensor, w_bf16: torch.Tensor) -> None:
        rec = self.manifest["tensors"][name]
        if tuple(w_hat.shape) != tuple(w_bf16.shape):
            raise SystemExit(f"[full_logits] OVERRIDE SHAPE CHECK FAILED: {name} is {tuple(w_hat.shape)} "
                             f"in the override, {tuple(w_bf16.shape)} in the checkpoint")
        w = w_bf16.to(device=w_hat.device, dtype=torch.float32)
        rel, cos = self.tq.rel_cos(w, w_hat)  # fp64 accumulation
        K = rec.get("K")
        bound = override_rel_bound(K) if K is not None else 0.5
        want = rec.get("rel_weight_err")
        st = {"K": K, "encoding": rec.get("encoding"), "rel": rel, "cos": cos, "rel_recorded": want,
              "rel_bound": bound, "bpw": (rec.get("bits") or {}).get("bpw"), "proxy": rec.get("proxy"),
              "file": str(self.tq.tensor_file(self.manifest, rec))}
        self.tensor_stats[name] = st
        if rel > bound or cos < 0.9:
            raise SystemExit(f"[full_logits] OVERRIDE WEIGHT CHECK FAILED: {name} (K={K}) has rel error "
                             f"{rel:.4f} (bound {bound:.3f}) and cosine {cos:.6f} against the bf16 "
                             "checkpoint -- a layout/mapping error or a directory of another model")
        if want is not None and abs(rel - want) > 0.02 * want + 1e-4:
            raise SystemExit(f"[full_logits] OVERRIDE WEIGHT CHECK FAILED: {name} reconstructs to rel error "
                             f"{rel:.5f} but trellis_quant.py recorded {want:.5f} when it wrote these "
                             "bits -- the file does not hold what the manifest says")

    def layer_state(self, prefix: str, keys: list[str]) -> dict[str, torch.Tensor]:
        names = [prefix + k for k in keys]
        sub = set(n for n in names if n in self.manifest["tensors"])
        read = [n for n in names if n not in sub or n not in self.tensor_stats]
        raw = get_tensors_grouped(self.index, read)
        sd = {}
        for k, n in zip(keys, names):
            if n in sub:
                t0 = time.perf_counter()
                w_hat = self.tq.load_override_tensor(self.manifest, n, self.device, self._cb_cache)
                self.reconstruct_seconds += time.perf_counter() - t0
                if n not in self.tensor_stats:
                    self._check(n, w_hat, raw.pop(n))
                sd[k] = w_hat.to(dtype=self.dtype)
                del w_hat
            else:
                sd[k] = raw.pop(n).to(device=self.device, dtype=self.dtype)
        return sd

    def segment_meta(self) -> dict:
        s = self.manifest.get("summary", {})
        return {**super().segment_meta(), "weights_override": {
            "manifest": str(self.manifest_path), "manifest_sha256": self.manifest_sha256,
            "encoding": self.manifest.get("encoding"), "bpw_target": self.manifest.get("bpw_target"),
            "bpw_measured": s.get("bpw"), "substituted_count": len(self.plan),
            "weights": "override directory for every tensor its manifest lists (reconstructed in the "
                       "original basis), bf16 checkpoint otherwise; activations as in the bf16 "
                       "reference (weight-only quantization)",
        }}

    def weights_record(self) -> dict:
        """The `weights_override` block of reference_run.json."""
        m = self.manifest
        s = m.get("summary", {})
        used = {n: m["tensors"][n] for n in self.plan}
        used_summary = self.tq.summarize_tensors(used) if used else {}
        rels = sorted(v["rel"] for v in self.tensor_stats.values())
        return {
            "manifest": str(self.manifest_path), "manifest_sha256": self.manifest_sha256,
            "encoding": m.get("encoding"), "bpw_target": m.get("bpw_target"),
            "K_uniform": m.get("K_uniform"), "allocation": m.get("allocation"),
            "hessian_dir": m.get("hessian_dir"), "hessian_manifest_sha256": m.get("hessian_manifest_sha256"),
            "hessian_basis": m.get("hessian_basis"), "recipe": m.get("recipe"),
            "provenance": m.get("provenance"),
            "summary_manifest": s,
            "summary_used": {k: used_summary.get(k) for k in ("n_tensors", "numel", "bits", "bpw",
                                                               "bpw_trellis_only", "decode_gib",
                                                               "proxy_mean", "proxy_max", "by_class")},
            "rule": ("every tensor the manifest lists is replaced by its reconstruction from the stored "
                     "form (trellis bits + fp16 suh/svh, or dense), rounded to bf16; everything else is "
                     "the bf16 checkpoint's"),
            "checked": {"count": len(self.tensor_stats), "rel_median": rels[len(rels) // 2] if rels else None,
                        "rel_max": rels[-1] if rels else None,
                        "cos_min": min((v["cos"] for v in self.tensor_stats.values()), default=None)},
            "per_tensor": self.tensor_stats,
            "reconstruct_seconds": self.reconstruct_seconds,
            "note": ("weight-only: activations, attention/GDN math, norms, embeddings and the lm_head are "
                     "the bf16 reference's; only the listed decoder linears are quantized"),
        }


def out_dir_dumps(out_dir: Path) -> list[dict]:
    """What `out_dir` already holds, one entry per reference_run.json / *.meta.json: `kind` 'bf16'
    (a bf16 reference), 'gguf' (a --weights-gguf dump) or 'override' (a --weights-override dump),
    `gguf` (the GGUF's header sha256, or the override manifest's sha256) and `n_layers` (None when
    unknown: segment sidecars written before they recorded it fall back to the directory's
    reference_run.json). Sidecars of other writers (the r4dx engine) are skipped."""
    if not out_dir.is_dir():
        return []
    docs = []
    run_path = out_dir / "reference_run.json"
    run_layers = None
    if run_path.exists():
        docs.append(("run", run_path))
    docs += [("meta", p) for p in sorted(out_dir.glob("*.meta.json"))]
    out = []
    for what, path in docs:
        try:
            with open(path, "r", encoding="utf-8") as f:
                doc = json.load(f)
        except (OSError, ValueError):
            continue
        wg = doc.get("weights_gguf")
        wo = doc.get("weights_override")
        if wg:
            kind, ident = "gguf", wg.get("header_sha256")
        elif wo:
            kind, ident = "override", wo.get("manifest_sha256")
        elif what == "run" or doc.get("source") == "reference":
            kind, ident = "bf16", None
        else:
            continue
        n_layers = doc.get("n_layers")
        if what == "run":
            run_layers = n_layers
        out.append({"file": path.name, "kind": kind, "gguf": ident, "n_layers": n_layers})
    for d in out:
        if d["n_layers"] is None:
            d["n_layers"] = run_layers
    return out


DUMP_KIND_TEXT = {"bf16": "bf16 reference", "gguf": "--weights-gguf dump",
                  "override": "--weights-override dump"}


def out_dir_conflicts(out_dir: Path, kind: str, gguf_sha: str | None, n_layers: int) -> list[str]:
    """Why a `kind` dump (of the GGUF with header sha256 `gguf_sha` / the override manifest with
    sha256 `gguf_sha`, over `n_layers` layers) must not be written into `out_dir`: kl_report.py
    pairs files by segment name, so a directory that mixes a bf16 reference with a substituted-
    weights dump, two different GGUFs or override manifests, or two layer counts would silently
    score the wrong thing. Empty when it may."""
    why = []
    for d in out_dir_dumps(out_dir):
        if d["kind"] != kind:
            why.append(f"{d['file']} is a {DUMP_KIND_TEXT[d['kind']]}")
        elif kind in ("gguf", "override") and d["gguf"] and gguf_sha and d["gguf"] != gguf_sha:
            what = "GGUF (header" if kind == "gguf" else "override manifest (manifest"
            why.append(f"{d['file']} is a dump of another {what} sha256 {d['gguf'][:16]}...)")
        elif d["n_layers"] is not None and int(d["n_layers"]) != int(n_layers):
            why.append(f"{d['file']} ran {d['n_layers']} layers, this run {n_layers}")
    return why


# --------------------------------------------------------------------------------------------
# Per-segment driver
# --------------------------------------------------------------------------------------------


def log_softmax_rows(logits: torch.Tensor, row_block: int = 128) -> tuple[torch.Tensor, float]:
    """In-place-ish fp32 log_softmax over the vocab, then the `LOGPROB_CLAMP` floor and the fp16
    cast. Returns the fp16 CPU tensor and the worst `|logsumexp(row)|` measured on the fp16 values
    actually written (validation (iii) in the README -- this is the end-to-end check, not a check
    of the fp32 intermediate)."""
    rows = logits.shape[0]
    out = torch.empty(rows, logits.shape[1], dtype=torch.float16, device="cpu")
    worst = 0.0
    for start in range(0, rows, row_block):
        stop = min(start + row_block, rows)
        blk = torch.log_softmax(logits[start:stop], dim=-1)
        blk = blk.clamp_min(LOGPROB_CLAMP).to(torch.float16)
        lse = torch.logsumexp(blk.float(), dim=-1).abs().max().item()
        worst = max(worst, lse)
        out[start:stop] = blk.cpu()
        del blk
    return out, worst


def topk_report(logits_row: torch.Tensor, k: int, tok=None) -> list[dict]:
    probs = torch.softmax(logits_row, dim=-1)
    top = torch.topk(logits_row, k)
    rep = []
    for logit, tid in zip(top.values.tolist(), top.indices.tolist()):
        entry = {"id": int(tid), "logit": float(logit), "prob": float(probs[tid].item())}
        if tok is not None:
            try:
                entry["piece"] = tok.decode([int(tid)])
            except Exception:
                pass
        rep.append(entry)
    return rep


def run_segment(ref: StreamingReference, name: str, token_ids: list[int], out_dir: Path,
                args, tok=None) -> dict:
    total_len = len(token_ids)
    if total_len < 2:
        raise ValueError(f"segment {name!r} has {total_len} tokens; need at least 2")
    _cuda_reset_peak(ref.device)
    t0 = time.perf_counter()
    hidden = ref.forward_hidden(token_ids, impl=args.impl)
    _cuda_sync(ref.device)
    t_stack = time.perf_counter() - t0

    t1 = time.perf_counter()
    logits = ref.logits_fp32(hidden, chunk=args.lm_head_chunk)
    _cuda_sync(ref.device)
    t_head = time.perf_counter() - t1

    last_top = topk_report(logits[-1], args.top_k, tok)

    # Rows 0..T-2 only: row i is log p(next token | tokens[0..i]). The last position's prediction
    # has no next token in this segment to be scored against, so it is reported but not written.
    lp16, worst_lse = log_softmax_rows(logits[:-1], row_block=args.row_block)
    del logits

    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = out_dir / f"{name}.logprobs.f16"
    arr = lp16.numpy()
    assert arr.dtype == np.float16 and arr.shape == (total_len - 1, ref.vocab_size)
    with open(bin_path, "wb") as f:
        arr.tofile(f)

    nll = float(-sum(float(arr[i, int(token_ids[i + 1])]) for i in range(total_len - 1)) / (total_len - 1))
    peak, reserved = _cuda_peak_gib(ref.device)

    meta = {
        "T": total_len,
        "V": ref.vocab_size,
        "dtype": "float16",
        "rows": total_len - 1,
        "source": getattr(ref, "source_tag", "reference"),
        "torch_dtype": "bfloat16",
        "sha256_of_token_ids_json": token_ids_sha256(token_ids),
        "segment": name,
        "model_dir": str(ref.model_dir),
        "impl": args.impl,
        "logprob_clamp": LOGPROB_CLAMP,
        "row_semantics": "row i = log_softmax(logits at position i) = log p(next | token_ids[0..i]), fp32 then fp16",
        "max_abs_logsumexp_fp16": worst_lse,
        "mean_nll_next_token": nll,
        "perplexity": float(np.exp(nll)),
        "last_position_top_k": last_top,
        "seconds_stack": t_stack,
        "seconds_lm_head": t_head,
        "peak_vram_gib": peak,
        "peak_vram_reserved_gib": reserved,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
    }
    if hasattr(ref, "segment_meta"):
        meta.update(ref.segment_meta())
    with open(out_dir / f"{name}.meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[full_logits] {name}: T={total_len} V={ref.vocab_size} stack={t_stack:.1f}s "
          f"lm_head={t_head:.1f}s peak={peak:.2f}GiB |logsumexp|max={worst_lse:.2e} "
          f"ppl={meta['perplexity']:.3f} -> {bin_path.name} "
          f"({bin_path.stat().st_size / 2**20:.1f} MiB)", flush=True)
    return meta


# --------------------------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------------------------


def last_position_logits(ref: StreamingReference, token_ids: list[int], impl: str, chunk: int) -> torch.Tensor:
    hidden = ref.forward_hidden(token_ids, impl=impl)
    return ref.logits_fp32(hidden[-1:], chunk=chunk)[0]


def do_cross_check(ref: StreamingReference, token_ids: list[int], args, tok) -> dict:
    """Validation (i): the same prompt through two independent compositions (see this file's
    docstring). Their last-position argmax and top-5 must agree."""
    print(f"[full_logits] cross-check: {len(token_ids)} tokens through --impl model and --impl manual",
          flush=True)
    res = {}
    for impl in ("model", "manual"):
        t0 = time.perf_counter()
        row = last_position_logits(ref, token_ids, impl, args.lm_head_chunk)
        res[impl] = {"top": topk_report(row, args.top_k, tok), "seconds": time.perf_counter() - t0,
                     "logits": row}
        print(f"  [{impl}] {res[impl]['seconds']:.1f}s top-{args.top_k}:")
        for rank, e in enumerate(res[impl]["top"]):
            print(f"    #{rank + 1}: id={e['id']:>7d} logit={e['logit']: .4f} prob={e['prob']: .4f} "
                  f"piece={e.get('piece', '')!r}")
    ids_model = [e["id"] for e in res["model"]["top"]]
    ids_manual = [e["id"] for e in res["manual"]["top"]]
    max_abs = (res["model"]["logits"] - res["manual"]["logits"]).abs().max().item()
    agree = ids_model == ids_manual
    print(f"  top-{args.top_k} ids agree: {agree}  (model={ids_model} manual={ids_manual})")
    print(f"  max |logit difference| over the whole {ref.vocab_size}-way vocabulary: {max_abs:.4e}")
    if not agree:
        raise SystemExit("[full_logits] CROSS-CHECK FAILED: the two compositions disagree on top-k")
    return {"prompt_tokens": token_ids, "top_k_model": ids_model, "top_k_manual": ids_manual,
            "max_abs_logit_diff": max_abs, "agree": agree,
            "top_model": res["model"]["top"], "top_manual": res["manual"]["top"]}


def do_greedy_selfcheck(ref: StreamingReference, prefix_ids: list[int], n_new: int, args, tok) -> dict:
    """Validation (ii): generate `n_new` tokens greedily (forward, argmax, append), then re-feed the
    whole sequence as a fixed context and require argmax at every generated position to reproduce
    the token that was generated there. Catches an off-by-one in the position/rope/mask wiring that
    a single forward pass cannot."""
    print(f"[full_logits] greedy self-consistency: {len(prefix_ids)}-token prefix + {n_new} tokens "
          f"({n_new + 1} full streaming passes, this is slow by design)", flush=True)
    ids = list(prefix_ids)
    generated = []
    for step in range(n_new):
        t0 = time.perf_counter()
        row = last_position_logits(ref, ids, args.impl, args.lm_head_chunk)
        nxt = int(row.argmax().item())
        ids.append(nxt)
        generated.append(nxt)
        piece = ""
        if tok is not None:
            try:
                piece = tok.decode([nxt])
            except Exception:
                pass
        print(f"  step {step + 1:>2}/{n_new}: id={nxt:>7d} {piece!r} ({time.perf_counter() - t0:.1f}s)",
              flush=True)

    t0 = time.perf_counter()
    hidden = ref.forward_hidden(ids, impl=args.impl)
    start = len(prefix_ids) - 1  # position that predicted generated[0]
    logits = ref.logits_fp32(hidden[start : len(ids) - 1], chunk=args.lm_head_chunk)
    argmax = logits.argmax(dim=-1).tolist()
    del logits
    mismatches = [(i, int(a), int(g)) for i, (a, g) in enumerate(zip(argmax, generated)) if a != g]
    print(f"  re-fed the {len(ids)}-token sequence in {time.perf_counter() - t0:.1f}s: "
          f"{len(generated) - len(mismatches)}/{len(generated)} positions reproduce their token")
    if mismatches:
        for i, a, g in mismatches[:10]:
            print(f"    MISMATCH at generated index {i}: refed argmax={a}, generated={g}")
        raise SystemExit("[full_logits] GREEDY SELF-CONSISTENCY FAILED")
    text = ""
    if tok is not None:
        try:
            text = tok.decode(generated)
        except Exception:
            pass
    print(f"  generated text: {text!r}")
    return {"prefix_len": len(prefix_ids), "n_new": n_new, "generated_ids": generated,
            "generated_text": text, "mismatches": mismatches, "passed": True}


# --------------------------------------------------------------------------------------------


#: Below this much free commit (Windows' commit limit, not free RAM) a run warns at start. Measured
#: on the CPU dry run, this process peaks at ~14-15 GiB of commit, bf16 or --weights-gguf alike.
#: Most of it is safetensors' shard mappings: while a shard is open, Windows charges its whole view
#: against commit (opening the 3.2 GiB lm_head shard takes 3.1 GiB of free commit, and reading one
#: layer from a 3.7 GiB shard raises the peak by ~7 GiB). The GGUF path adds up to ~1.5 GiB (a
#: 32768-row fp32 lm_head block is 640 MiB), and its dequant threads ~25 MiB each. A failed
#: allocation aborts the run halfway.
LOW_COMMIT_GIB = 20.0


def host_commit_gib() -> tuple[float | None, float | None]:
    """(commit this process can still allocate, this process's peak commit so far) in GiB, from
    GlobalMemoryStatusEx / K32GetProcessMemoryInfo. (None, None) off Windows or on any failure."""
    if sys.platform != "win32":
        return None, None
    import ctypes

    class MemoryStatusEx(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile",
                                              "ullAvailPageFile", "ullTotalVirtual", "ullAvailVirtual",
                                              "ullAvailExtendedVirtual")]

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
            (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                           "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                           "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

    try:
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
        ms = MemoryStatusEx()
        ms.dwLength = ctypes.sizeof(ms)
        pmc = ProcessMemoryCounters()
        pmc.cb = ctypes.sizeof(pmc)
        avail = ms.ullAvailPageFile / 2**30 if k32.GlobalMemoryStatusEx(ctypes.byref(ms)) else None
        peak = (pmc.PeakPagefileUsage / 2**30
                if k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb) else None)
        return avail, peak
    except (OSError, AttributeError):
        return None, None


def load_tokens_file(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    if "segments" not in doc:
        raise ValueError(f"{path} has no 'segments' key -- see the shared format in tools/reference/README.md")
    return doc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    ap.add_argument("--tokens", type=Path, default=Path(__file__).parent / "kl_corpus" / "tokens.json")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default tools/reference/kl_out/ref; required with --weights-gguf")
    ap.add_argument("--weights-gguf", type=Path, default=None, metavar="GGUF",
                    help="substitute this llama.cpp GGUF's quantized weights (dequantized) for the "
                         "checkpoint's; see GGUFWeightsReference. The output is NOT a bf16 reference")
    ap.add_argument("--gguf-threads", type=int, default=8, help="CPU threads for GGUF dequantization")
    ap.add_argument("--gguf-sha256", action="store_true",
                    help="record the whole GGUF's sha256 in the run record (reads the full file)")
    ap.add_argument("--gguf-allow-row-outliers", action="store_true",
                    help="use (and record) an embedding / lm_head row above the per-row error bound instead "
                         "of aborting; only after gguf_validate.py shows the row is the file's own")
    ap.add_argument("--weights-override", type=Path, default=None, metavar="DIR",
                    help="substitute the weights of this override directory (its weights_override.json, "
                         "e.g. trellis_quant.py quantize-model / mix output); see "
                         "OverrideWeightsReference. The output is NOT a bf16 reference")
    ap.add_argument("--segment", action="append", default=None,
                    help="segment name to run (repeatable, or comma-separated); default: all")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--impl", default="model", choices=["model", "manual"],
                    help="which composition runs the segments (see this file's docstring)")
    ap.add_argument("--lm-head-chunk", type=int, default=32768,
                    help="vocabulary rows of lm_head resident at once (32768 rows ~ 320 MiB bf16)")
    ap.add_argument("--row-block", type=int, default=128, help="sequence rows per log_softmax block")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=None, help="truncate every segment to this length")
    ap.add_argument("--debug-max-layers", type=int, default=None,
                    help="run only the first N decoder layers -- NOT a valid golden, timing only")
    ap.add_argument("--cross-check", type=int, default=0, metavar="N",
                    help="validation (i): run the first N tokens of the first selected segment "
                         "through both compositions and compare top-k (48 is the documented run)")
    ap.add_argument("--selfcheck-greedy", type=int, default=0, metavar="N",
                    help="validation (ii): greedily generate N tokens and re-feed them (slow: N+1 passes)")
    ap.add_argument("--selfcheck-prefix", type=int, default=48,
                    help="prefix length (tokens of the first selected segment) for --selfcheck-greedy")
    ap.add_argument("--skip-segments", action="store_true",
                    help="run only the requested validations, write no .logprobs.f16")
    args = ap.parse_args()

    # A --weights-gguf dump and a bf16 reference must never share a directory, nor two GGUFs or two
    # layer counts: kl_report.py pairs files by segment name, so a mixed directory would silently
    # score the wrong thing.
    if args.weights_gguf and args.weights_override:
        raise SystemExit("[full_logits] --weights-gguf and --weights-override are mutually exclusive")
    if args.out_dir is None:
        if args.weights_gguf or args.weights_override:
            raise SystemExit("[full_logits] --weights-gguf / --weights-override need an explicit --out-dir")
        args.out_dir = Path(__file__).parent / "kl_out" / "ref"
    gguf_sha = None
    dump_kind = "bf16"
    if args.weights_gguf:
        import gguf_dequant as gd

        dump_kind = "gguf"
        with gd.GGUFFile(args.weights_gguf) as g:
            gguf_sha = g.header_sha256
    elif args.weights_override:
        import trellis_quant as tq

        dump_kind = "override"
        gguf_sha = sha256_file(Path(tq.load_manifest(args.weights_override)["_path"]))
    want_layers = args.debug_max_layers or load_text_config(args.model_dir)[1].num_hidden_layers
    conflicts = out_dir_conflicts(args.out_dir, dump_kind, gguf_sha, want_layers)
    if conflicts:
        raise SystemExit(f"[full_logits] refusing to write a {DUMP_KIND_TEXT[dump_kind]} "
                         f"({want_layers} layers) into {args.out_dir}: " + "; ".join(conflicts[:5]))
    commit_avail, _ = host_commit_gib()
    if commit_avail is not None and commit_avail < LOW_COMMIT_GIB:
        print(f"[full_logits] WARNING: only {commit_avail:.1f} GiB of commit left on this machine (the "
              f"Windows commit limit, not free RAM). This run peaks at ~15 GiB of its own (see "
              f"LOW_COMMIT_GIB); want >= {LOW_COMMIT_GIB:.0f}. A failed host allocation (numpy "
              "ArrayMemoryError, or an access violation in native code) aborts the run halfway. Stop idle "
              "memory holders first (an r4dx-server, other python jobs). Lowering --lm-head-chunk / "
              "--gguf-threads saves well under 1 GiB.", flush=True)

    device = resolve_device(args.device)  # enforces $env:HIP_VISIBLE_DEVICES == '1' for cuda
    doc = load_tokens_file(args.tokens)
    wanted = None
    if args.segment:
        wanted = [s for arg in args.segment for s in arg.split(",") if s]
    segments = [s for s in doc["segments"] if wanted is None or s["name"] in wanted]
    if not segments:
        raise SystemExit(f"no segments selected (available: {[s['name'] for s in doc['segments']]})")
    if args.max_tokens:
        segments = [{"name": s["name"], "token_ids": s["token_ids"][: args.max_tokens]} for s in segments]

    # This script tokenizes nothing: the ids come from the tokens file, and the tokenizer mode that
    # made them (common.tokens_file_tokenizer_mode: make_tokens_json.py's record, the r4dx-cli dump
    # shape, ...; "unknown" when it cannot tell) is recorded below. The tokenizer only decodes
    # pieces for display, which no mode changes, so it is always the canonical one.
    tokens_mode = tokens_file_tokenizer_mode(doc)
    print(f"[full_logits] tokens file {args.tokens}: tokenizer mode {tokens_mode}")
    tok = None
    try:
        tok = load_ref_tokenizer(args.model_dir, "canonical")
    except Exception as exc:  # decoding pieces is a nicety, never a requirement
        print(f"[full_logits] tokenizer unavailable ({type(exc).__name__}: {exc}); ids only")

    print(f"[full_logits] building the streaming skeleton from {args.model_dir} on {device} ...", flush=True)
    t0 = time.perf_counter()
    if args.weights_gguf:
        ref = GGUFWeightsReference(args.model_dir, device, args.weights_gguf,
                                   max_layers=args.debug_max_layers, threads=args.gguf_threads,
                                   full_sha256=args.gguf_sha256,
                                   allow_row_outliers=args.gguf_allow_row_outliers)
    elif args.weights_override:
        ref = OverrideWeightsReference(args.model_dir, device, args.weights_override,
                                       max_layers=args.debug_max_layers)
    else:
        ref = StreamingReference(args.model_dir, device, max_layers=args.debug_max_layers)
    print(f"[full_logits] skeleton ready in {time.perf_counter() - t0:.1f}s: {ref.n_layers} layers, "
          f"hidden={ref.text_config.hidden_size}, V={ref.vocab_size}, impl={args.impl}", flush=True)
    if args.debug_max_layers:
        print("[full_logits] WARNING --debug-max-layers is set: these outputs are NOT a valid golden")

    run = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "model_dir": str(args.model_dir),
        "config_sha256": sha256_file(args.model_dir / "config.json"),
        "tokens_file": str(args.tokens),
        "tokenizer": doc.get("tokenizer"),
        "tokenizer_mode": tokens_mode,
        "tokenizer_provenance": doc.get("tokenizer_provenance"),
        "device": str(device),
        "source": ref.source_tag,
        "weights": ("GGUF-dequantized where the GGUF quantized, else bf16 checkpoint (see weights_gguf)"
                    if args.weights_gguf else
                    "override directory where its manifest lists a tensor, else bf16 checkpoint "
                    "(see weights_override)" if args.weights_override else "bf16 checkpoint"),
        "torch_dtype": "bfloat16",
        "impl": args.impl,
        "n_layers": ref.n_layers,
        "debug_max_layers": args.debug_max_layers,
        "vocab_size": ref.vocab_size,
        "logprob_clamp": LOGPROB_CLAMP,
        "torch_version": torch.__version__,
        "segments": {},
    }

    if args.cross_check:
        run["cross_check"] = do_cross_check(ref, segments[0]["token_ids"][: args.cross_check], args, tok)
    if args.selfcheck_greedy:
        run["greedy_selfcheck"] = do_greedy_selfcheck(
            ref, segments[0]["token_ids"][: args.selfcheck_prefix], args.selfcheck_greedy, args, tok)

    if not args.skip_segments:
        for seg in segments:
            run["segments"][seg["name"]] = run_segment(ref, seg["name"], seg["token_ids"], args.out_dir,
                                                       args, tok)
        run["total_seconds"] = sum(s["seconds_stack"] + s["seconds_lm_head"] for s in run["segments"].values())

    if args.weights_gguf:
        run["weights_gguf"] = ref.weights_record()
        w = run["weights_gguf"]
        print(f"[full_logits] GGUF weights: {w['substituted']['count']} tensors substituted "
              f"{w['substituted']['by_type']}, {w['kept_lossless_verified']['count']} lossless kept "
              f"(max {w['kept_lossless_verified']['max_ulp']} ulp), dequant {w['dequant_seconds']:.0f}s")
        for t, v in w["weight_error_by_type"].items():
            print(f"    {t:7s} n={v['n']:4d} rel median {v['rel_median']:.4f} max {v['rel_max']:.4f} "
                  f"cos min {v['cos_min']:.6f} worst row rel {v['row_rel_max']:.4f}")
        for name, rows in w["row_outliers_used"].items():
            print(f"    used rows above the per-row bound in {name}: "
                  + ", ".join(f"{o['row']} (rel {o['rel']:.3f}{', coarse scale' if o['coarse_scale'] else ''})"
                              for o in rows))
    if args.weights_override:
        run["weights_override"] = w = ref.weights_record()
        su, ck = w["summary_used"], w["checked"]
        if ck["count"] and su.get("bpw") is not None:
            print(f"[full_logits] override weights: {ck['count']} tensors substituted ({w['encoding']}), "
                  f"{su['bpw']:.4f} bpw measured ({su['decode_gib']:.3f} GiB), rel median "
                  f"{ck['rel_median']:.4f} max {ck['rel_max']:.4f}, cos min {ck['cos_min']:.6f}, "
                  f"reconstruct {w['reconstruct_seconds']:.0f}s")
            for cls, v in (su.get("by_class") or {}).items():
                print(f"    {cls:26s} n={v['n']:3d} K {v['K']} bpw {v['bpw']:.4f}")
        else:
            print("[full_logits] override weights: no tensor was substituted (no segment ran?)")
    avail_end, peak = host_commit_gib()
    run["host_commit_gib"] = {"available_at_start": commit_avail, "available_at_end": avail_end,
                              "process_peak": peak}
    if peak is not None and commit_avail is not None:
        print(f"[full_logits] host commit: process peak {peak:.1f} GiB, {commit_avail:.1f} GiB free "
              f"at start", flush=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "reference_run.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, default=str)
    print(f"[full_logits] wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
