"""Exact Python/CPU/fp32 reference of the DFlash2 speculative drafter.

Ports the ROCmFPX reference implementation's `graph<false>` draft forward (target-feature
injection into the draft's own KV cache, the noise-block/block-diffusion draft pass, and the
DFlash2 selector lattice + greedy walk) line-for-line into plain numpy, reading the real Q8_0
draft weights through `tools/reference/gguf_min.py` (this repo's own from-scratch GGUF/Q8_0
parser -- no `gguf-py` import).

See `docs/dflash2.md` for every convention this file encodes (rope, norm, conv, selector,
lifecycle) and the exact reference-code line it was resolved against. This module does not touch
`gguf-py`, does not touch anything under `src/`, and never imports from the ROCmFPX checkout except
implicitly-by-derivation (i.e. this code, not that code, runs).

Embeddings and the LM head are the TARGET's, never the draft's own (the draft GGUF has no
`token_embd`/`output` tensors) -- see `TargetProvider` for the two supported sources:

    --synthetic              seeded random [vocab=4096, 5120] embedding table + [4096, 5120] lm
                              head (small, so golden fixtures stay a few MB); the selector
                              codebooks are indexed by token id, so this mode simply uses their
                              first `vocab` rows.
    --target-dir <path>      real Qwen3.8-27B checkpoint (safetensors): streams only the
                              embedding rows actually needed and the full lm_head via
                              `tools/reference/common.py`'s `ShardIndex` (never materializes any
                              other shard).

Run with the reference venv (`tools/reference/README.md`'s convention), CPU only:

    <reference venv>\\python.exe tools\\reference\\dflash2_ref.py --gen-fixtures all
    <reference venv>\\python.exe tools\\reference\\dflash2_ref.py --synthetic --anchor-id 7 \\
        --n-injected 40 --seed 0   # ad hoc single-round draft, prints the drafted tokens
    <reference venv>\\python.exe tools\\reference\\dflash2_ref.py --real captured.npz \\
        --target-dir C:\\AI\\models\\Qwen3.8-27B   # docs/dflash2.md "npz schema"
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_min import GGUFError, load_gguf  # noqa: E402

DEFAULT_DFLASH2_GGUF = Path(r"D:\models\Qwen3.8-27B-DFlash2\Qwen3.8-27B-DFlash2-Q8_0.gguf")
DEFAULT_TARGET_DIR = Path(r"C:\AI\models\Qwen3.8-27B")


# --------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------


@dataclass
class DFlash2Config:
    n_embd: int
    n_ff: int
    n_head: int
    n_head_kv: int
    head_dim: int
    n_layer: int
    rope_theta: float
    rms_eps: float
    block_size: int
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    target_layers: list[int]  # GGUF-recorded (already +1 vs the HF 0-based source layer)
    sliding_window: int
    mask_token_id: int
    vocab_size: int  # the TARGET's real vocab (248320); synthetic mode overrides at use time
    n_embd_inp_enc: int  # = len(target_layers) * target_hidden_size = 25600

    @classmethod
    def from_gguf(cls, g) -> "DFlash2Config":
        arch = g.require_meta("general.architecture")
        if arch != "dflash":
            raise GGUFError(f"expected general.architecture='dflash', got {arch!r}")

        def key(name: str):
            return g.require_meta(f"{arch}.{name}")

        n_head = int(key("attention.head_count"))
        n_head_kv = int(key("attention.head_count_kv"))
        head_dim = int(key("attention.key_length"))
        assert head_dim == int(key("attention.value_length")), "k/v head_dim must match"

        causal = bool(g.meta(f"{arch}.attention.causal", False))
        if causal:
            raise GGUFError("this reference only implements DFlash2's non-causal block attention")

        sections = list(g.meta(f"{arch}.rope.dimension_sections", [0, 0, 0, 0]))
        # No `{arch}.rope.dimension_count` key on this checkpoint -> llama-model.cpp's loader
        # (src/llama-model.cpp:1344-1346) defaults n_rot to n_embd_head_k_full = head_dim (full
        # rotation). See docs/dflash2.md "RoPE" for the full derivation, including why sections
        # [64,0,0,0] with n_rot=128 degenerates M-RoPE to plain NeoX rope over all 128 dims using
        # only the temporal position id.
        n_rot = int(g.meta(f"{arch}.rope.dimension_count", head_dim))
        if n_rot != head_dim:
            raise GGUFError(
                f"this reference assumes full-head rotation (n_rot==head_dim); got n_rot={n_rot}, "
                f"head_dim={head_dim} -- {arch}.rope.dimension_count is present and different, "
                "docs/dflash2.md's RoPE section needs re-deriving before this code is trusted"
            )
        if len(sections) != 4 or sections[1:] != [0, 0, 0] or sections[0] != n_rot // 2:
            raise GGUFError(
                f"expected degenerate temporal-only M-RoPE sections [{n_rot // 2},0,0,0], got "
                f"{sections} -- this reference's rope() only implements that specific case"
            )

        target_layers = [int(x) for x in key("target_layers")]

        sw_pattern = g.meta(f"{arch}.attention.sliding_window_pattern")
        if sw_pattern is not None and not all(bool(x) for x in sw_pattern):
            raise GGUFError("this reference assumes SWA applies to every DFlash2 layer")

        mask_token_id = int(g.require_meta("tokenizer.ggml.mask_token_id"))
        vocab_size = g.tensors["selector_predecessor.weight"].numpy_shape[0]

        return cls(
            n_embd=int(key("embedding_length")),
            n_ff=int(key("feed_forward_length")),
            n_head=n_head,
            n_head_kv=n_head_kv,
            head_dim=head_dim,
            n_layer=int(key("block_count")),
            rope_theta=float(key("rope.freq_base")),
            rms_eps=float(key("attention.layer_norm_rms_epsilon")),
            block_size=int(key("block_size")),
            conv_kernel_size=int(key("conv_kernel_size")),
            conv_group_size=int(key("conv_group_size")),
            selector_rank=int(key("selector_rank")),
            selector_top_k=int(key("selector_top_k")),
            target_layers=target_layers,
            sliding_window=int(key("attention.sliding_window")),
            mask_token_id=mask_token_id,
            vocab_size=vocab_size,
            n_embd_inp_enc=len(target_layers) * int(key("embedding_length")),
        )


# --------------------------------------------------------------------------------------------
# Weights
# --------------------------------------------------------------------------------------------


@dataclass
class LayerWeights:
    attn_norm: np.ndarray  # [5120]
    attn_q: np.ndarray  # [n_head*head_dim, 5120]
    attn_k: np.ndarray  # [n_head_kv*head_dim, 5120]
    attn_v: np.ndarray  # [n_head_kv*head_dim, 5120]
    attn_output: np.ndarray  # [5120, n_head*head_dim]
    attn_q_norm: np.ndarray  # [head_dim]
    attn_k_norm: np.ndarray  # [head_dim]
    attn_conv_base: np.ndarray  # [2(side), kernel, 5120]
    attn_conv_proj: np.ndarray  # [2*kernel*n_groups, 5120]
    ffn_norm: np.ndarray  # [5120]
    ffn_gate: np.ndarray  # [n_ff, 5120]
    ffn_up: np.ndarray  # [n_ff, 5120]
    ffn_down: np.ndarray  # [5120, n_ff]
    ffn_conv_base: np.ndarray  # [2(side), kernel, 5120]
    ffn_conv_proj: np.ndarray  # [2*kernel*n_groups, 5120]


@dataclass
class DFlash2Weights:
    cfg: DFlash2Config
    layers: list[LayerWeights]
    fc: np.ndarray  # [5120, n_embd_inp_enc]
    enc_output_norm: np.ndarray  # [5120]
    output_norm: np.ndarray  # [5120]
    selector_hidden: np.ndarray  # [rank, 5120]
    selector_predecessor: np.ndarray  # [vocab, rank]
    selector_successor: np.ndarray  # [vocab, rank]

    @classmethod
    def load(cls, gguf_path: str | Path) -> "DFlash2Weights":
        g = load_gguf(gguf_path)
        cfg = DFlash2Config.from_gguf(g)

        layers = []
        for il in range(cfg.n_layer):
            p = f"blk.{il}."
            layers.append(
                LayerWeights(
                    attn_norm=g.get_array(p + "attn_norm.weight"),
                    attn_q=g.get_array(p + "attn_q.weight"),
                    attn_k=g.get_array(p + "attn_k.weight"),
                    attn_v=g.get_array(p + "attn_v.weight"),
                    attn_output=g.get_array(p + "attn_output.weight"),
                    attn_q_norm=g.get_array(p + "attn_q_norm.weight"),
                    attn_k_norm=g.get_array(p + "attn_k_norm.weight"),
                    attn_conv_base=g.get_array(p + "attn_conv_base"),
                    attn_conv_proj=g.get_array(p + "attn_conv_proj.weight"),
                    ffn_norm=g.get_array(p + "ffn_norm.weight"),
                    ffn_gate=g.get_array(p + "ffn_gate.weight"),
                    ffn_up=g.get_array(p + "ffn_up.weight"),
                    ffn_down=g.get_array(p + "ffn_down.weight"),
                    ffn_conv_base=g.get_array(p + "ffn_conv_base"),
                    ffn_conv_proj=g.get_array(p + "ffn_conv_proj.weight"),
                )
            )

        return cls(
            cfg=cfg,
            layers=layers,
            fc=g.get_array("fc.weight"),
            enc_output_norm=g.get_array("enc.output_norm.weight"),
            output_norm=g.get_array("output_norm.weight"),
            selector_hidden=g.get_array("selector_hidden.weight"),
            selector_predecessor=g.get_array("selector_predecessor.weight"),
            selector_successor=g.get_array("selector_successor.weight"),
        )

    def bf16_rounded(self) -> "DFlash2Weights":
        """Return a copy with every linear weight (not norms/tables) rounded fp32->bf16->fp32.

        This is the precision the C++ port's bf16 GEMM path actually runs at (r4dx's own kernels
        are bf16 throughout, `docs/architecture.md`) -- fixtures also ship this variant's expected
        outputs so the C++ side has a tolerance-free target once it lands, per task item 3.
        """
        import copy

        def bf16(a: np.ndarray) -> np.ndarray:
            u = a.astype(np.float32).view(np.uint32)
            # round-to-nearest-even, matching third_party/libr4d/r4d_dflash_conv_body.h's r4d_f2bf
            rounded = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
            return rounded.view(np.float32).astype(np.float32)

        out = copy.deepcopy(self)
        for layer in out.layers:
            for field_name in (
                "attn_q", "attn_k", "attn_v", "attn_output", "attn_conv_proj",
                "ffn_gate", "ffn_up", "ffn_down", "ffn_conv_proj",
            ):
                setattr(layer, field_name, bf16(getattr(layer, field_name)))
        out.fc = bf16(out.fc)
        out.selector_hidden = bf16(out.selector_hidden)
        return out


# --------------------------------------------------------------------------------------------
# Target embedding / lm_head provider
# --------------------------------------------------------------------------------------------


class TargetProvider:
    """Abstracts the TARGET model's embed_tokens rows + lm_head (the draft has neither)."""

    def __init__(self, hidden: int):
        self.hidden = hidden

    def embed_rows(self, ids: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def lm_head(self) -> np.ndarray:
        """Full [vocab, hidden] lm_head matrix (small enough in both modes to hold in RAM)."""
        raise NotImplementedError

    @property
    def vocab_size(self) -> int:
        raise NotImplementedError


class SyntheticTarget(TargetProvider):
    def __init__(self, hidden: int, vocab: int, seed: int):
        super().__init__(hidden)
        self._vocab = vocab
        rng = np.random.RandomState(seed)
        # Typical transformer embedding-init scale; the absolute scale doesn't matter for a
        # self-consistent numerical fixture (only the draft's OWN weights are being golden-tested).
        self._embed = (rng.randn(vocab, hidden) * 0.02).astype(np.float32)
        self._head = (rng.randn(vocab, hidden) * 0.02).astype(np.float32)

    def embed_rows(self, ids: np.ndarray) -> np.ndarray:
        return self._embed[ids]

    def lm_head(self) -> np.ndarray:
        return self._head

    @property
    def vocab_size(self) -> int:
        return self._vocab


class RealTarget(TargetProvider):
    def __init__(self, target_dir: Path):
        from common import ShardIndex  # tools/reference/common.py, sibling module

        self.index = ShardIndex.load(target_dir)
        # Qwen3.8-27B: tied or untied embeddings -- container-format.md says untied
        # (tie_word_embeddings=false), so embed_tokens and lm_head are separate tensors.
        # The checkpoint is VLM-shaped (Qwen3_5ForConditionalGeneration): embed_tokens sits under
        # a `language_model` sub-module in the safetensors key, lm_head does not (verified against
        # this checkpoint's own model.safetensors.index.json, not assumed).
        embed_candidates = ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")
        self._embed_name = next((n for n in embed_candidates if self.index.available(n)), embed_candidates[0])
        self._head_name = "lm_head.weight"
        row0 = self.index.get_row_slice(self._embed_name, 0, 1)
        super().__init__(int(row0.shape[1]))
        self._head_cache: np.ndarray | None = None

    def embed_rows(self, ids: np.ndarray) -> np.ndarray:
        import torch

        out = np.empty((len(ids), self.hidden), dtype=np.float32)
        for i, tid in enumerate(ids):
            row = self.index.get_row_slice(self._embed_name, int(tid), int(tid) + 1)
            out[i] = row.to(dtype=torch.float32).numpy()[0]
        return out

    def lm_head(self) -> np.ndarray:
        import torch

        if self._head_cache is None:
            t = self.index.get_tensor(self._head_name)
            self._head_cache = t.to(dtype=torch.float32).numpy()
        return self._head_cache

    @property
    def vocab_size(self) -> int:
        return self.lm_head().shape[0]


# --------------------------------------------------------------------------------------------
# Core math
# --------------------------------------------------------------------------------------------


def linear(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """y = x @ w.T ; w is [out, in] (this reader's numpy row-major convention), no bias."""
    return x @ w.T


def rmsnorm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """Plain (non-zero-centered) RMSNorm: y = x * rsqrt(mean(x^2, axis=-1) + eps) * weight.

    NOT the `(1+w)` zero-centered convention `docs/architecture.md` documents for the TARGET
    model's own `Qwen3_5RMSNorm` -- DFlash2 is a llama.cpp-native GGUF checkpoint, and every norm
    site in `dflash.cpp` (attn_norm/ffn_norm/output_norm/output_norm_enc/q_norm/k_norm/the
    injection path's k_norm) goes through `build_norm(cur, w, NULL, LLM_NORM_RMS, il)`, which is
    ggml's plain `ggml_rms_norm` + `ggml_mul(cur, w)` -- no offset. See docs/dflash2.md "RMSNorm
    convention".
    """
    variance = np.mean(np.square(x.astype(np.float64)), axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(variance + eps)
    return (x.astype(np.float64) * inv).astype(np.float32) * weight


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def rope_neox(x: np.ndarray, pos: np.ndarray, theta: float, n_rot: int) -> np.ndarray:
    """GPT-NeoX-style split-half rope, pair (i, i+n_rot/2), single scalar position per token.

    x: [..., T, n_rot] (T on the second-to-last axis, matching every call site below: q/k are
    always shaped [T, n_head(_kv), head_dim] and pos is [T]). theta is the rope base (1e7).

    This is DFlash2's rope exactly because its M-RoPE sections are the degenerate [n_rot/2,0,0,0]
    (all frequency pairs assigned to the temporal section, none to height/width) -- see
    `DFlash2Config.from_gguf`'s assertion and docs/dflash2.md "RoPE" for the full derivation from
    `ggml_rope_multi`'s section semantics.
    """
    half = n_rot // 2
    inv_freq = theta ** (-(2.0 * np.arange(half, dtype=np.float64)) / n_rot)
    ang = np.outer(pos.astype(np.float64), inv_freq)  # [T, half]
    cos = np.cos(ang).astype(np.float32)
    sin = np.sin(ang).astype(np.float32)
    # broadcast [T,half] -> [T,1,half] against x's [T, H, half]
    while cos.ndim < x.ndim:
        cos = cos[:, None, :]
        sin = sin[:, None, :]
    x1 = x[..., :half]
    x2 = x[..., half:]
    out = np.empty_like(x)
    out[..., :half] = x1 * cos - x2 * sin
    out[..., half:] = x2 * cos + x1 * sin
    return out


def dflash2_conv(
    h: np.ndarray, dyn: np.ndarray, base: np.ndarray, side: int, group_size: int
) -> np.ndarray:
    """DFlash2's grouped dynamic depthwise conv (both attention and FFN use this, one call per
    side per sub-block). Formula, from `third_party/libr4d/r4d_dflash_conv_body.h`'s header
    comment (the fused-kernel spec) and matching `build_dflash2_conv` in dflash.cpp bit-for-bit:

        out[t,c] = (base[side,0,c] + dyn[t,side,0,g]) * h[t,c]
                 + (base[side,1,c] + dyn[t,side,1,g]) * h[t-1,c] * (t >= 1)     g = c // group_size

    generalized here to `kernel_size` taps (DFlash2 always uses 2) via `t >= tap`, matching
    `r4d_dflash_conv_body`'s `(t & blockmask) >= tap` for a single block of `T == block_size`
    tokens (blockmask degenerates to plain `t >= tap` when the whole array IS one block, t in
    [0, block_size)).

    h: [T, H]. dyn: [T, 2*kernel_size*n_groups] (column order: side-major, then tap, then group
    fastest -- `dflash.cpp`'s `ggml_reshape_4d(dynamic, n_groups, kernel_size, 2, n_tokens)`, group
    varying fastest). base: [2(side), kernel_size, H] (this reader's `numpy_shape` for the GGUF
    tensor `ne=[H, kernel, 2]`, i.e. side slowest, channel fastest in the raw file).
    """
    T, H = h.shape
    n_groups = H // group_size
    kernel_size = base.shape[1]
    out = np.zeros_like(h)
    for tap in range(kernel_size):
        col0 = side * (kernel_size * n_groups) + tap * n_groups
        d = dyn[:, col0 : col0 + n_groups]  # [T, n_groups]
        d_full = np.repeat(d, group_size, axis=1)  # [T, H]
        b_tap = base[side, tap, :]  # [H]
        if tap == 0:
            shifted = h
            valid = np.ones(T, dtype=np.float32)
        else:
            shifted = np.zeros_like(h)
            shifted[tap:] = h[:-tap]
            valid = (np.arange(T) >= tap).astype(np.float32)
        out = out + (b_tap[None, :] + d_full) * shifted * valid[:, None]
    return out


def swa_visible(query_pos: np.ndarray, key_pos: np.ndarray, sliding_window: int) -> np.ndarray:
    """`llama_hparams::is_masked_swa` (LLAMA_SWA_TYPE_STANDARD), inverted to "visible".

    Masked iff `query_pos - key_pos >= sliding_window`; DFlash2's `attention.causal=false` means
    there is no separate future-masking term (a key in the query's future is always visible unless
    the (negative) `query_pos - key_pos` happens to still trip the >= check, which it never does
    for a positive window). See docs/dflash2.md "SWA visibility rule".
    """
    diff = query_pos[:, None].astype(np.int64) - key_pos[None, :].astype(np.int64)
    return diff < sliding_window


def attention_gqa(
    q: np.ndarray, k: np.ndarray, v: np.ndarray, mask: np.ndarray, scale: float
) -> np.ndarray:
    """q: [Tq,Hq,D], k/v: [Tk,Hkv,D], mask: [Tq,Tk] bool (True=visible). Returns [Tq, Hq*D]."""
    Tq, Hq, D = q.shape
    Tk, Hkv, _ = k.shape
    group = Hq // Hkv
    k_rep = np.repeat(k, group, axis=1)  # [Tk,Hq,D], repeat_interleave semantics
    v_rep = np.repeat(v, group, axis=1)
    scores = np.einsum("qhd,khd->hqk", q.astype(np.float64), k_rep.astype(np.float64)) * scale
    neg_inf = np.finfo(np.float64).min
    scores = np.where(mask[None, :, :], scores, neg_inf)
    scores = scores - scores.max(axis=-1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=-1, keepdims=True)
    out = np.einsum("hqk,khd->qhd", w, v_rep.astype(np.float64))
    return out.reshape(Tq, Hq * D).astype(np.float32)


def top_k_desc(logits: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-row top-k (ids, values), sorted descending by value -- matches `ggml_top_k`."""
    idx_part = np.argpartition(-logits, kth=k - 1, axis=-1)[..., :k]
    part_vals = np.take_along_axis(logits, idx_part, axis=-1)
    order = np.argsort(-part_vals, axis=-1)
    ids = np.take_along_axis(idx_part, order, axis=-1)
    vals = np.take_along_axis(part_vals, order, axis=-1)
    return ids.astype(np.int64), vals.astype(np.float32)


# --------------------------------------------------------------------------------------------
# KV cache (draft's own, per-layer)
# --------------------------------------------------------------------------------------------


@dataclass
class LayerKvCache:
    k: np.ndarray  # [N, n_head_kv, head_dim], post-norm, post-rope
    v: np.ndarray  # [N, n_head_kv, head_dim], raw
    pos: np.ndarray  # [N] absolute positions


@dataclass
class DraftKvCache:
    layers: list[LayerKvCache]

    @classmethod
    def empty(cls, n_layer: int, n_head_kv: int, head_dim: int) -> "DraftKvCache":
        return cls(
            layers=[
                LayerKvCache(
                    k=np.zeros((0, n_head_kv, head_dim), dtype=np.float32),
                    v=np.zeros((0, n_head_kv, head_dim), dtype=np.float32),
                    pos=np.zeros((0,), dtype=np.int64),
                )
                for _ in range(n_layer)
            ]
        )

    def n_injected(self) -> int:
        return int(self.layers[0].pos.shape[0])

    def append(self, il: int, k: np.ndarray, v: np.ndarray, pos: np.ndarray) -> None:
        c = self.layers[il]
        c.k = np.concatenate([c.k, k], axis=0)
        c.v = np.concatenate([c.v, v], axis=0)
        c.pos = np.concatenate([c.pos, pos], axis=0)


# --------------------------------------------------------------------------------------------
# Encoder + injection
# --------------------------------------------------------------------------------------------


def encode_features(weights: DFlash2Weights, features: np.ndarray) -> np.ndarray:
    """g = rmsnorm_enc(fc(features)). features: [N, n_embd_inp_enc] -> g: [N, n_embd]."""
    g = linear(features, weights.fc)
    g = rmsnorm(g, weights.enc_output_norm, weights.cfg.rms_eps)
    return g


def inject(weights: DFlash2Weights, cache: DraftKvCache, g: np.ndarray, pos: np.ndarray) -> None:
    """Writes K (post-k_norm, post-rope) and V (raw) into every layer's own cache, in place."""
    cfg = weights.cfg
    for il, w in enumerate(weights.layers):
        K = linear(g, w.attn_k).reshape(-1, cfg.n_head_kv, cfg.head_dim)
        V = linear(g, w.attn_v).reshape(-1, cfg.n_head_kv, cfg.head_dim)
        K = rmsnorm(K, w.attn_k_norm, cfg.rms_eps)
        K = rope_neox(K, pos, cfg.rope_theta, cfg.head_dim)
        cache.append(il, K, V, pos)


# --------------------------------------------------------------------------------------------
# Draft block forward
# --------------------------------------------------------------------------------------------


@dataclass
class DraftRoundResult:
    tokens: list[int]
    intermediates: dict[str, np.ndarray]


def selector_walk(
    W_succ: np.ndarray,
    W_pred: np.ndarray,
    cand: np.ndarray,
    unary: np.ndarray,
    gate_vec: np.ndarray,
    anchor_id: int,
    block_size: int,
    p_min: float,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """docs/dflash2.md section 4.3's greedy lattice walk, lifted out of `draft_round` unchanged.

    Extracted (2026-09-20, stage S2) so `--real` mode can also run it on a MIXTURE of this
    reference's own tensors and a port's -- which is what attributes a chain divergence to the
    lm_head (`cand`/`unary`) versus the drafter itself (`gate`). `draft_round` calls it with its own
    three tensors, so every fixture it generates is bit-identical to before the extraction (gated by
    dflash2_selftest.py).
    """
    tokens: list[int] = []
    P = np.array([anchor_id], dtype=np.int64)
    pred_idx = 0
    score_matrices: dict[int, np.ndarray] = {}
    for t in range(1, block_size):
        cand_t = cand[t]
        unary_t = unary[t]
        gate_t = gate_vec[t]
        succ = W_succ[cand_t]  # [top_k, rank]
        pred = W_pred[P]  # [n_pred, rank]
        cond = pred * gate_t[None, :]  # [n_pred, rank]
        score = cond @ succ.T + unary_t[None, :]  # [n_pred, top_k] == score[a,b]
        score_matrices[t] = score.copy()

        scores_row = score[pred_idx]  # [top_k]
        b = int(np.argmax(scores_row))

        if p_min > 0.0:
            smax = scores_row.max()
            prob = 1.0 / np.sum(np.exp(scores_row - smax))
            if prob < p_min:
                break

        tokens.append(int(cand_t[b]))
        pred_idx = b
        P = cand_t
    return tokens, score_matrices


def draft_round(
    weights: DFlash2Weights,
    target: TargetProvider,
    cache: DraftKvCache,
    anchor_id: int,
    p_min: float = 0.0,
    capture_layer0: bool = False,
    mask_token_id: int | None = None,
) -> DraftRoundResult:
    cfg = weights.cfg
    block_size = cfg.block_size
    n_injected = cache.n_injected()

    # The real GGUF's mask_token_id (248070, the target's <|audio_start|> row) is out of range for
    # a --synthetic small-vocab target; callers using a synthetic target pass an in-range override
    # (see `_synthetic_mask_id`) since the synthetic embedding table only has `vocab` rows.
    mask_id = cfg.mask_token_id if mask_token_id is None else mask_token_id
    token_ids = np.array([anchor_id] + [mask_id] * (block_size - 1), dtype=np.int64)
    x = target.embed_rows(token_ids).astype(np.float32)  # [block_size, n_embd]
    pos = (n_injected + np.arange(block_size)).astype(np.int64)

    inter: dict[str, np.ndarray] = {"inp_noise_embd": x.copy(), "positions": pos.copy()}

    for il, w in enumerate(weights.layers):
        h = rmsnorm(x, w.attn_norm, cfg.rms_eps)
        dyn_attn = linear(h, w.attn_conv_proj)
        h_conv = dflash2_conv(h, dyn_attn, w.attn_conv_base, side=0, group_size=cfg.conv_group_size)

        q = linear(h_conv, w.attn_q).reshape(block_size, cfg.n_head, cfg.head_dim)
        k = linear(h_conv, w.attn_k).reshape(block_size, cfg.n_head_kv, cfg.head_dim)
        v = linear(h_conv, w.attn_v).reshape(block_size, cfg.n_head_kv, cfg.head_dim)

        q = rmsnorm(q, w.attn_q_norm, cfg.rms_eps)
        k = rmsnorm(k, w.attn_k_norm, cfg.rms_eps)
        if capture_layer0 and il == 0:
            # Pre-rope q/k (post q_norm/k_norm). Dumped so a port's standalone rope kernel has a
            # real (input, position, output) triple from this reference to check against -- without
            # it the only fixture-side rope anchor was a post-rope tensor whose own input was never
            # saved, forcing a port to inverse-rotate the output with its own convention first
            # (which can only ever prove the port is self-consistent). See docs/dflash2.md's
            # "Kernels" subsection, r4dx_rope_neox_bf16.
            inter["attn_q_prerope_l0"] = q.copy()
            inter["attn_k_prerope_l0"] = k.copy()
        q = rope_neox(q, pos, cfg.rope_theta, cfg.head_dim)
        k = rope_neox(k, pos, cfg.rope_theta, cfg.head_dim)

        kv_cache = cache.layers[il]
        full_k = np.concatenate([kv_cache.k, k], axis=0)
        full_v = np.concatenate([kv_cache.v, v], axis=0)
        full_pos = np.concatenate([kv_cache.pos, pos], axis=0)
        mask = swa_visible(pos, full_pos, cfg.sliding_window)

        scale = 1.0 / np.sqrt(cfg.head_dim)
        attn_out = attention_gqa(q, full_k, full_v, mask, scale)
        o = linear(attn_out, w.attn_output)
        if capture_layer0 and il == 0:
            inter["attn_o_preconv_l0"] = o.copy()
        o = dflash2_conv(o, dyn_attn, w.attn_conv_base, side=1, group_size=cfg.conv_group_size)

        if capture_layer0 and il == 0:
            # NOTE on the two historical names: `attn_conv_in_l0` is the side-0 conv's OUTPUT
            # (h_conv, i.e. the tensor going INTO attention) and `attn_conv_out_l0` is the side-1
            # conv's OUTPUT. Neither is a conv INPUT, so neither alone lets a port drive the fused
            # conv kernel. The four arrays added below close that gap: `attn_conv_x_l0` (side-0
            # input), `attn_o_preconv_l0` (side-1 input, captured above), `attn_dyn_l0` (the shared
            # dynamic projection both sides read) and `attn_conv_base_l0` (this layer's base
            # weight) -- with them, side 0 is (attn_conv_x_l0, dyn, base) -> attn_conv_in_l0 and
            # side 1 is (attn_o_preconv_l0, dyn, base) -> attn_conv_out_l0. The names of the two
            # pre-existing arrays are deliberately NOT changed (they are asserted by name in
            # dflash2_selftest.py and referenced by docs/dflash2.md section 9).
            inter["attn_conv_in_l0"] = h_conv.copy()
            inter["attn_conv_out_l0"] = o.copy()
            inter["attn_conv_x_l0"] = h.copy()
            inter["attn_dyn_l0"] = dyn_attn.copy()
            inter["attn_conv_base_l0"] = w.attn_conv_base.copy()
            # Post-rope block q/k and raw v, plus the pure attention output (pre-Wo). Together with
            # `injected_k_l0`/`injected_v_l0` (the 40-position store) and `n_injected`, this is a
            # complete, self-contained input/output pair for a standalone non-causal windowed GQA
            # attention kernel -- see docs/dflash2.md's "Kernels" subsection,
            # r4dx_dflash_attn_bf16.
            inter["attn_q_l0"] = q.copy()
            inter["attn_k_l0"] = k.copy()
            inter["attn_v_l0"] = v.copy()
            inter["attn_out_l0"] = attn_out.copy()

        x = x + o
        inter[f"x_post_attn_l{il}"] = x.copy()

        h2 = rmsnorm(x, w.ffn_norm, cfg.rms_eps)
        dyn_ffn = linear(h2, w.ffn_conv_proj)
        h2c = dflash2_conv(h2, dyn_ffn, w.ffn_conv_base, side=0, group_size=cfg.conv_group_size)
        gate = linear(h2c, w.ffn_gate)
        up = linear(h2c, w.ffn_up)
        f = linear(silu(gate) * up, w.ffn_down)
        f = dflash2_conv(f, dyn_ffn, w.ffn_conv_base, side=1, group_size=cfg.conv_group_size)
        x = x + f
        inter[f"x_post_ffn_l{il}"] = x.copy()

    x_final = rmsnorm(x, weights.output_norm, cfg.rms_eps)
    inter["x_final_normed"] = x_final.copy()

    lm_head = target.lm_head()
    vocab = lm_head.shape[0]
    logits = linear(x_final, lm_head)  # [block_size, vocab]
    inter["logits"] = logits.copy()

    top_k = cfg.selector_top_k
    cand, unary = top_k_desc(logits, top_k)  # [block_size, top_k] each
    gate_vec = linear(x_final, weights.selector_hidden)  # [block_size, rank]
    inter["cand"] = cand.copy()
    inter["unary"] = unary.copy()
    inter["gate"] = gate_vec.copy()

    W_succ = weights.selector_successor[:vocab]
    W_pred = weights.selector_predecessor[:vocab]

    tokens, score_matrices = selector_walk(
        W_succ, W_pred, cand, unary, gate_vec, anchor_id, block_size, p_min
    )

    inter["score_matrices"] = score_matrices
    inter["drafted_tokens"] = np.array(tokens, dtype=np.int64)

    return DraftRoundResult(tokens=tokens, intermediates=inter)


# --------------------------------------------------------------------------------------------
# Fixture generation
# --------------------------------------------------------------------------------------------


def _synthetic_mask_id(cfg: DFlash2Config, vocab: int) -> int:
    """The real mask_token_id (248070) is outside a small synthetic vocab; clamp deterministically."""
    return cfg.mask_token_id if cfg.mask_token_id < vocab else vocab - 1


def _synthetic_features(seed: int, n: int, n_embd_inp: int) -> np.ndarray:
    """~unit RMS per channel, like a real residual stream (task item 3)."""
    rng = np.random.RandomState(seed)
    return rng.randn(n, n_embd_inp).astype(np.float32)


def _save_npy(out_dir: Path, name: str, arr: np.ndarray) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.npy"
    np.save(path, arr)
    return {"file": path.name, "shape": list(arr.shape), "dtype": str(arr.dtype)}


def _score_matrices_to_dict(sm: dict[int, np.ndarray], out_dir: Path, prefix: str) -> dict:
    manifest = {}
    for t, mat in sm.items():
        manifest[str(t)] = _save_npy(out_dir, f"{prefix}_t{t}", mat)
    return manifest


def gen_fixture_a(weights: DFlash2Weights, out_root: Path, seed: int = 0) -> dict:
    """Synthetic vocab 4096, N=40 injected positions, full intermediate capture."""
    cfg = weights.cfg
    vocab = 4096
    n_injected = 40
    out_dir = out_root / "fixture_a"

    target = SyntheticTarget(cfg.n_embd, vocab, seed)
    features = _synthetic_features(seed, n_injected, cfg.n_embd_inp_enc)
    anchor_id = int(np.random.RandomState(seed + 1).randint(0, vocab))

    cache = DraftKvCache.empty(cfg.n_layer, cfg.n_head_kv, cfg.head_dim)
    g = encode_features(weights, features)
    pos = np.arange(n_injected, dtype=np.int64)
    inject(weights, cache, g, pos)

    mask_id = _synthetic_mask_id(cfg, vocab)
    result = draft_round(
        weights, target, cache, anchor_id, p_min=0.0, capture_layer0=True, mask_token_id=mask_id
    )

    manifest: dict = {
        "description": "Fixture A: synthetic vocab 4096, N=40 injected positions, real draft weights.",
        "seed": seed,
        "vocab": vocab,
        "n_injected": n_injected,
        "anchor_id": anchor_id,
        "mask_id": _synthetic_mask_id(cfg, vocab),
        "block_size": cfg.block_size,
        "target_provider": {
            "class": "SyntheticTarget",
            "note": (
                "The full [vocab=4096, hidden=5120] embedding table and lm_head are each ~80 MB "
                "of fp32 -- too large for this fixture directory's 60 MB budget -- and are exactly "
                "reproducible from (hidden, vocab, seed) via SyntheticTarget.__init__ "
                "(numpy.random.RandomState(seed).randn(vocab, hidden) * 0.02 for the embedding "
                "table, RandomState(seed).randn(vocab, hidden) * 0.02 drawn immediately after for "
                "the lm_head -- same RandomState instance, embedding drawn first). NOT dumped to "
                "disk; the selftest reconstructs both by re-running this exact class."
            ),
            "hidden": cfg.n_embd,
            "vocab": vocab,
            "seed": seed,
        },
        "arrays": {},
    }
    manifest["arrays"]["features"] = _save_npy(out_dir, "features", features)
    used_ids = np.array([anchor_id, _synthetic_mask_id(cfg, vocab)], dtype=np.int64)
    manifest["arrays"]["embedding_rows_used_ids"] = _save_npy(out_dir, "embedding_rows_used_ids", used_ids)
    manifest["arrays"]["embedding_rows_used"] = _save_npy(
        out_dir, "embedding_rows_used", target.embed_rows(used_ids)
    )
    manifest["arrays"]["g_encoded"] = _save_npy(out_dir, "g_encoded", g)
    for il in range(cfg.n_layer):
        manifest["arrays"][f"injected_k_l{il}"] = _save_npy(
            out_dir, f"injected_k_l{il}", cache.layers[il].k
        )
        manifest["arrays"][f"injected_v_l{il}"] = _save_npy(
            out_dir, f"injected_v_l{il}", cache.layers[il].v
        )
        manifest["arrays"][f"x_post_attn_l{il}"] = _save_npy(
            out_dir, f"x_post_attn_l{il}", result.intermediates[f"x_post_attn_l{il}"]
        )
        manifest["arrays"][f"x_post_ffn_l{il}"] = _save_npy(
            out_dir, f"x_post_ffn_l{il}", result.intermediates[f"x_post_ffn_l{il}"]
        )
    manifest["arrays"]["attn_conv_in_l0"] = _save_npy(
        out_dir, "attn_conv_in_l0", result.intermediates["attn_conv_in_l0"]
    )
    manifest["arrays"]["attn_conv_out_l0"] = _save_npy(
        out_dir, "attn_conv_out_l0", result.intermediates["attn_conv_out_l0"]
    )
    # Layer-0 device-kernel drive arrays (docs/dflash2.md "Kernels"): the conv's own two INPUTS +
    # dyn + base, the block's post-rope q/k/v and pre-rope q/k, the pure attention output, and the
    # final output_norm weight. ~1 MB total, well inside this directory's 60 MB budget; each one
    # exists so a standalone HIP kernel can be driven from this reference's real numbers instead of
    # only from a CPU re-implementation of the same formulas.
    for _name in (
        "attn_conv_x_l0",
        "attn_dyn_l0",
        "attn_conv_base_l0",
        "attn_o_preconv_l0",
        "attn_q_prerope_l0",
        "attn_k_prerope_l0",
        "attn_q_l0",
        "attn_k_l0",
        "attn_v_l0",
        "attn_out_l0",
    ):
        manifest["arrays"][_name] = _save_npy(out_dir, _name, result.intermediates[_name])
    manifest["arrays"]["output_norm_w"] = _save_npy(out_dir, "output_norm_w", weights.output_norm)
    manifest["x_final_normed_rms_eps"] = cfg.rms_eps
    manifest["x_final_normed_source"] = "x_post_ffn_l4"
    manifest["x_final_normed_weight"] = "output_norm_w"
    manifest["arrays"]["x_final_normed"] = _save_npy(out_dir, "x_final_normed", result.intermediates["x_final_normed"])
    manifest["arrays"]["logits"] = _save_npy(out_dir, "logits", result.intermediates["logits"])
    manifest["arrays"]["cand"] = _save_npy(out_dir, "cand", result.intermediates["cand"])
    manifest["arrays"]["unary"] = _save_npy(out_dir, "unary", result.intermediates["unary"])
    manifest["arrays"]["gate"] = _save_npy(out_dir, "gate", result.intermediates["gate"])
    manifest["score_matrices"] = _score_matrices_to_dict(
        result.intermediates["score_matrices"], out_dir, "score"
    )
    manifest["arrays"]["drafted_tokens"] = _save_npy(
        out_dir, "drafted_tokens", result.intermediates["drafted_tokens"]
    )
    manifest["drafted_tokens"] = result.tokens

    # bf16-rounded-weights expected output (task item 3's "tolerance-free target" for the C++ bf16 path)
    bf16_weights = weights.bf16_rounded()
    g_bf16 = encode_features(bf16_weights, features)
    cache_bf16 = DraftKvCache.empty(cfg.n_layer, cfg.n_head_kv, cfg.head_dim)
    inject(bf16_weights, cache_bf16, g_bf16, pos)
    result_bf16 = draft_round(bf16_weights, target, cache_bf16, anchor_id, p_min=0.0, mask_token_id=mask_id)
    manifest["arrays"]["logits_bf16_weights"] = _save_npy(
        out_dir, "logits_bf16_weights", result_bf16.intermediates["logits"]
    )
    manifest["arrays"]["drafted_tokens_bf16_weights"] = _save_npy(
        out_dir, "drafted_tokens_bf16_weights", result_bf16.intermediates["drafted_tokens"]
    )
    manifest["drafted_tokens_bf16_weights"] = result_bf16.tokens

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def gen_fixture_b(weights: DFlash2Weights, out_root: Path, seed: int = 0) -> dict:
    """Same as A but N=2100 injected positions (exercises the 2048 sliding window truncation)."""
    cfg = weights.cfg
    vocab = 4096
    n_injected = 2100
    out_dir = out_root / "fixture_b"

    target = SyntheticTarget(cfg.n_embd, vocab, seed)
    features = _synthetic_features(seed, n_injected, cfg.n_embd_inp_enc)
    anchor_id = int(np.random.RandomState(seed + 1).randint(0, vocab))

    cache = DraftKvCache.empty(cfg.n_layer, cfg.n_head_kv, cfg.head_dim)
    g = encode_features(weights, features)
    pos = np.arange(n_injected, dtype=np.int64)
    inject(weights, cache, g, pos)

    result = draft_round(
        weights, target, cache, anchor_id, p_min=0.0, mask_token_id=_synthetic_mask_id(cfg, vocab)
    )

    manifest: dict = {
        "description": "Fixture B: fixture A's setup but N=2100 injected positions (sliding_window=2048).",
        "seed": seed,
        "vocab": vocab,
        "n_injected": n_injected,
        "anchor_id": anchor_id,
        "block_size": cfg.block_size,
        "note": (
            "features/embedding/lm_head are not dumped (task item 3: fixture B saves only logits, "
            "lattice, drafted tokens) -- all are deterministically reproducible from `seed` via "
            "the same SyntheticTarget/_synthetic_features construction fixture A documents."
        ),
        "target_provider": {"class": "SyntheticTarget", "hidden": cfg.n_embd, "vocab": vocab, "seed": seed},
        "arrays": {},
    }
    manifest["arrays"]["logits"] = _save_npy(out_dir, "logits", result.intermediates["logits"])
    manifest["arrays"]["cand"] = _save_npy(out_dir, "cand", result.intermediates["cand"])
    manifest["arrays"]["unary"] = _save_npy(out_dir, "unary", result.intermediates["unary"])
    manifest["arrays"]["gate"] = _save_npy(out_dir, "gate", result.intermediates["gate"])
    manifest["score_matrices"] = _score_matrices_to_dict(
        result.intermediates["score_matrices"], out_dir, "score"
    )
    manifest["arrays"]["drafted_tokens"] = _save_npy(
        out_dir, "drafted_tokens", result.intermediates["drafted_tokens"]
    )
    manifest["drafted_tokens"] = result.tokens

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def gen_fixture_c(weights: DFlash2Weights, out_root: Path, seed: int = 0, p_min: float = 0.3) -> dict:
    """Fixture A's exact setup with p_min=0.3 to exercise the selector's early stop."""
    cfg = weights.cfg
    vocab = 4096
    n_injected = 40
    out_dir = out_root / "fixture_c"

    target = SyntheticTarget(cfg.n_embd, vocab, seed)
    features = _synthetic_features(seed, n_injected, cfg.n_embd_inp_enc)
    anchor_id = int(np.random.RandomState(seed + 1).randint(0, vocab))

    cache = DraftKvCache.empty(cfg.n_layer, cfg.n_head_kv, cfg.head_dim)
    g = encode_features(weights, features)
    pos = np.arange(n_injected, dtype=np.int64)
    inject(weights, cache, g, pos)

    result = draft_round(
        weights, target, cache, anchor_id, p_min=p_min, mask_token_id=_synthetic_mask_id(cfg, vocab)
    )

    manifest: dict = {
        "description": "Fixture C: fixture A's exact setup with p_min=0.3 (selector early stop).",
        "seed": seed,
        "vocab": vocab,
        "n_injected": n_injected,
        "anchor_id": anchor_id,
        "block_size": cfg.block_size,
        "p_min": p_min,
        "arrays": {},
    }
    manifest["arrays"]["logits"] = _save_npy(out_dir, "logits", result.intermediates["logits"])
    manifest["arrays"]["cand"] = _save_npy(out_dir, "cand", result.intermediates["cand"])
    manifest["arrays"]["unary"] = _save_npy(out_dir, "unary", result.intermediates["unary"])
    manifest["arrays"]["gate"] = _save_npy(out_dir, "gate", result.intermediates["gate"])
    manifest["score_matrices"] = _score_matrices_to_dict(
        result.intermediates["score_matrices"], out_dir, "score"
    )
    manifest["arrays"]["drafted_tokens"] = _save_npy(
        out_dir, "drafted_tokens", result.intermediates["drafted_tokens"]
    )
    manifest["drafted_tokens"] = result.tokens
    manifest["n_drafted_le_block_minus_1"] = len(result.tokens) <= cfg.block_size - 1

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


# --------------------------------------------------------------------------------------------
# --real mode: real captured target features
# --------------------------------------------------------------------------------------------


def _load_real_dump(path: Path) -> dict:
    """Accepts EITHER form of a real-capture dump.

    1. The `.npz` `docs/dflash2.md` section 8 documents (keys: features, anchor_id, optional
       n_injected/positions) -- unchanged, for a Python producer.
    2. A DIRECTORY (or the `manifest.json` inside one) written by
       `tests/model/tool_dflash_probe.cpp`: `features.bin`, raw little-endian float32 in C order,
       plus a `manifest.json` recording `features_shape`/`features_dtype`/`anchor_id`/`n_injected`
       (schema `dflash2_real_dump_v1`). A C++ tool has no npz writer and adding one there would be a
       zip implementation; a raw blob plus the shape is the whole difference.
    """
    if path.is_dir() or path.suffix == ".json":
        man_path = (path / "manifest.json") if path.is_dir() else path
        with open(man_path, "r", encoding="utf-8") as f:
            man = json.load(f)
        dtype = man.get("features_dtype", "float32")
        if dtype != "float32":
            raise ValueError(f"{man_path}: only float32 features are supported, got {dtype}")
        shape = tuple(int(d) for d in man["features_shape"])
        feats = np.fromfile(man_path.parent / man.get("features_file", "features.bin"),
                            dtype=np.float32)
        if feats.size != shape[0] * shape[1]:
            raise ValueError(
                f"{man_path}: features.bin holds {feats.size} floats, manifest says {shape}"
            )
        out = {"features": feats.reshape(shape), "anchor_id": int(man["anchor_id"])}
        if "n_injected" in man:
            out["n_injected"] = int(man["n_injected"])
        if "positions" in man:
            out["positions"] = np.asarray(man["positions"], dtype=np.int64)
        return out
    data = np.load(path)
    return {k: data[k] for k in data.files}


def run_real(weights: DFlash2Weights, npz_path: Path, target_dir: Path, p_min: float) -> None:
    """docs/dflash2.md "npz schema" documents the expected keys; see _load_real_dump for the
    equivalent raw-.bin + manifest.json directory form a C++ producer writes."""
    data = _load_real_dump(npz_path)
    features = np.asarray(data["features"]).astype(np.float32)
    anchor_id = int(data["anchor_id"])
    n_injected = int(data["n_injected"]) if "n_injected" in data else features.shape[0]
    positions = (
        np.asarray(data["positions"]).astype(np.int64)
        if "positions" in data
        else np.arange(n_injected, dtype=np.int64)
    )

    target = RealTarget(target_dir)
    cache = DraftKvCache.empty(weights.cfg.n_layer, weights.cfg.n_head_kv, weights.cfg.head_dim)
    g = encode_features(weights, features)
    inject(weights, cache, g, positions)

    result = draft_round(weights, target, cache, anchor_id, p_min=p_min)
    print(f"drafted {len(result.tokens)} tokens: {result.tokens}")
    cand = result.intermediates["cand"]
    for t in range(weights.cfg.block_size):
        print(f"  block pos {t}: top-{weights.cfg.selector_top_k} = {cand[t].tolist()}")

    if npz_path.is_dir():
        _compare_with_port(weights, npz_path, result, anchor_id, p_min)


def _rel_l2(got: np.ndarray, ref: np.ndarray) -> float:
    got = np.asarray(got, dtype=np.float64).ravel()
    ref = np.asarray(ref, dtype=np.float64).ravel()
    return float(np.linalg.norm(got - ref) / max(1e-30, np.linalg.norm(ref)))


def _compare_with_port(
    weights: DFlash2Weights, dump_dir: Path, result: DraftRoundResult, anchor_id: int, p_min: float
) -> None:
    """Compare this reference's round against the port's own round over the SAME real features.

    The port (`tests/model/tool_dflash_probe.cpp`) optionally writes its own `x_final_normed.bin`,
    `gate.bin`, `cand.bin`, `unary.bin` and `drafted_tokens` beside the features it dumped. When
    they are present this prints, in order: how far apart the two DRAFTERS are (`x_final_normed`,
    `gate` -- these involve no lm_head at all, so they isolate everything docs/dflash2.md specifies);
    how far apart the two TARGETS' lm_heads are (`cand`/`unary`, which the port computes through the
    container's 4-bit lm_head while this reference uses the checkpoint's fp32 one); and then two
    HYBRID walks that attribute any chain divergence to one side or the other.
    """
    with open(dump_dir / "manifest.json", "r", encoding="utf-8") as f:
        man = json.load(f)
    cfg = weights.cfg
    B, topk, rank = cfg.block_size, cfg.selector_top_k, cfg.selector_rank

    def load(name: str, dtype, shape):
        p = dump_dir / name
        if not p.exists():
            return None
        return np.fromfile(p, dtype=dtype).reshape(shape)

    port_xf = load("x_final_normed.bin", np.float32, (B, cfg.n_embd))
    port_gate = load("gate.bin", np.float32, (B, rank))
    port_cand = load("cand.bin", np.int32, (B, topk))
    port_unary = load("unary.bin", np.float32, (B, topk))
    port_tokens = man.get("drafted_tokens")
    if port_xf is None and port_gate is None and port_cand is None:
        print("(no port-side tensors in this dump -- nothing to compare)")
        return

    print("\n--- port vs reference, same real features ---")
    if port_xf is not None:
        print(f"  x_final_normed RelL2 = {_rel_l2(port_xf, result.intermediates['x_final_normed']):.3e}"
              "   [drafter only: no lm_head involved]")
    if port_gate is not None:
        print(f"  gate           RelL2 = {_rel_l2(port_gate, result.intermediates['gate']):.3e}"
              "   [drafter only: x_final @ selector_hidden]")
    if port_cand is not None:
        ref_cand = result.intermediates["cand"]
        exact = int((port_cand.astype(np.int64) == ref_cand).sum())
        overlap = sum(int(np.isin(port_cand[t], ref_cand[t]).sum()) for t in range(B))
        print(f"  cand  exact-position {exact}/{B * topk}, set-overlap {overlap}/{B * topk}"
              "   [lm_head: port uses the container's 4-bit head, this uses the checkpoint's fp32 one]")
    if port_unary is not None:
        print(f"  unary          RelL2 = {_rel_l2(port_unary, result.intermediates['unary']):.3e}")

    if port_tokens is not None:
        same = list(port_tokens) == list(result.tokens)
        print(f"  chain: port {list(port_tokens)}\n         ref  {list(result.tokens)}"
              f"   -> {'IDENTICAL' if same else 'DIFFERENT'}")
        if not same and port_cand is not None and port_gate is not None:
            vocab = result.intermediates["logits"].shape[1]
            W_succ = weights.selector_successor[:vocab]
            W_pred = weights.selector_predecessor[:vocab]
            hyb_lmhead, _ = selector_walk(
                W_succ, W_pred, port_cand.astype(np.int64), port_unary,
                result.intermediates["gate"], anchor_id, B, p_min
            )
            hyb_drafter, _ = selector_walk(
                W_succ, W_pred, result.intermediates["cand"], result.intermediates["unary"],
                port_gate, anchor_id, B, p_min
            )
            print(f"  attribution (which side moves the chain):")
            print(f"    port's cand/unary + REFERENCE gate  -> {hyb_lmhead}"
                  f"  {'== port' if hyb_lmhead == list(port_tokens) else ''}")
            print(f"    reference cand/unary + PORT gate    -> {hyb_drafter}"
                  f"  {'== reference' if hyb_drafter == list(result.tokens) else ''}")
            print("    (a hybrid that reproduces the PORT's chain from the port's cand alone means "
                  "the divergence is the lm_head's precision, not the drafter's.)")


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", type=Path, default=DEFAULT_DFLASH2_GGUF, help="DFlash2 Q8_0 GGUF path")
    ap.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "golden_out" / "dflash2")
    ap.add_argument("--gen-fixtures", choices=["A", "B", "C", "all"], default=None)
    ap.add_argument("--synthetic", action="store_true", help="ad hoc single-round draft, synthetic target")
    ap.add_argument("--vocab", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchor-id", type=int, default=1)
    ap.add_argument("--n-injected", type=int, default=40)
    ap.add_argument("--p-min", type=float, default=0.0)
    ap.add_argument("--real", type=Path, default=None,
                    help="real captured target features: either the .npz of docs/dflash2.md "
                         "section 8, or a directory holding features.bin + manifest.json as "
                         "tests/model/tool_dflash_probe.cpp writes")
    ap.add_argument("--target-dir", type=Path, default=DEFAULT_TARGET_DIR)
    args = ap.parse_args()

    weights = DFlash2Weights.load(args.gguf)

    if args.gen_fixtures:
        which = ["A", "B", "C"] if args.gen_fixtures == "all" else [args.gen_fixtures]
        for name in which:
            fn = {"A": gen_fixture_a, "B": gen_fixture_b, "C": gen_fixture_c}[name]
            print(f"generating fixture {name} ...")
            manifest = fn(weights, args.out_dir, seed=args.seed)
            print(f"  drafted_tokens = {manifest['drafted_tokens']}")
        return

    if args.real is not None:
        run_real(weights, args.real, args.target_dir, args.p_min)
        return

    if args.synthetic:
        target = SyntheticTarget(weights.cfg.n_embd, args.vocab, args.seed)
        features = _synthetic_features(args.seed, args.n_injected, weights.cfg.n_embd_inp_enc)
        cache = DraftKvCache.empty(weights.cfg.n_layer, weights.cfg.n_head_kv, weights.cfg.head_dim)
        g = encode_features(weights, features)
        pos = np.arange(args.n_injected, dtype=np.int64)
        inject(weights, cache, g, pos)
        mask_id = _synthetic_mask_id(weights.cfg, args.vocab)
        result = draft_round(
            weights, target, cache, args.anchor_id, p_min=args.p_min, mask_token_id=mask_id
        )
        print(f"drafted {len(result.tokens)} tokens: {result.tokens}")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
