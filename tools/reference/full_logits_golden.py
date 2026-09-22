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
    load_text_config,
    resolve_device,
    sha256_file,
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
        raw = get_tensors_grouped(self.index, [prefix + k for k in keys])
        sd = {k: raw[prefix + k].to(device=self.device, dtype=self.dtype) for k in keys}
        layer.load_state_dict(sd, assign=True)
        layer.eval()
        for p in layer.parameters():
            p.requires_grad_(False)
        return layer

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
        from safetensors import safe_open

        rows = hidden.shape[0]
        out = torch.empty(rows, self.vocab_size, device=self.device, dtype=torch.float32)
        shard = self.index.weight_map[LM_HEAD_NAME]
        with safe_open(str(self.index.model_dir / shard), framework="pt", device="cpu") as f:
            sl = f.get_slice(LM_HEAD_NAME)
            for start in range(0, self.vocab_size, chunk):
                stop = min(start + chunk, self.vocab_size)
                w = sl[start:stop, :].clone().to(device=self.device, dtype=self.dtype)
                out[:, start:stop] = torch.nn.functional.linear(hidden, w).float()
                del w
        return out


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
        "source": "reference",
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
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).parent / "kl_out" / "ref")
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

    tok = None
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(str(args.model_dir))
    except Exception as exc:  # decoding pieces is a nicety, never a requirement
        print(f"[full_logits] tokenizer unavailable ({type(exc).__name__}: {exc}); ids only")

    print(f"[full_logits] building the streaming skeleton from {args.model_dir} on {device} ...", flush=True)
    t0 = time.perf_counter()
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
        "device": str(device),
        "torch_dtype": "bfloat16",
        "impl": args.impl,
        "n_layers": ref.n_layers,
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

    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest = args.out_dir / "reference_run.json"
    with open(manifest, "w", encoding="utf-8") as f:
        json.dump(run, f, indent=2, default=str)
    print(f"[full_logits] wrote {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
