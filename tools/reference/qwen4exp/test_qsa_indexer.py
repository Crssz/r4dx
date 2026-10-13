"""CPU tests of the vectorized QSA selection (plan M0-6) against HF's per-query loop
(`Qwen4ExpTextQSAIndexer.forward`, modular_qwen4_exp.py:368-474). No GPU, no model weights:

    D:\\venvs\\r4dx-gemma-ref\\Scripts\\python.exe -m pytest tools\\reference\\qwen4exp\\test_qsa_indexer.py

The indexer is a tiny random module from a reduced `Qwen4ExpTextConfig` (hidden 256, 4 heads x 128
index dims, 1 kv head, budget 2048 / ratio 4) with the checkpoint's rope settings (head_dim 256, partial
rotary 0.25, theta 1e7, interleaved mrope [11, 11, 10]), in bf16 like the checkpoint. Masks must be equal
bit for bit. Queries whose selection boundary is an exact score tie (frequent: blocks scoring exactly 0)
are counted and printed by test_report; a mismatch message says whether the bad queries were tied.
tie_rule="recent" (plan Q23) is checked against a per-query Python sorted() reference of the same rule and
must equal tie_rule="hf" on every untied query.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("HIP_VISIBLE_DEVICES", "")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import math  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402
from transformers import DynamicCache  # noqa: E402
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: E402
    Qwen4ExpTextQSAIndexer,
    Qwen4ExpTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

from qwen4exp import qsa_indexer as qsa  # noqa: E402

LAYER_IDX = 3  # first "indexed_attention" layer of the default 4-layer pattern
HIDDEN = 256
TIMINGS: dict[int, tuple[float, float]] = {}
TIES: dict[str, tuple[int, int]] = {}
RECENT_VS_HF: dict[str, tuple[int, int]] = {}


def make_config(budget: int = 2048, ratio: int = 4) -> Qwen4ExpTextConfig:
    return Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=budget,
        indexer_compress_ratio=ratio,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    )


def make_indexer(cfg: Qwen4ExpTextConfig, dtype: torch.dtype, seed: int = 0):
    torch.manual_seed(seed)
    indexer = Qwen4ExpTextQSAIndexer(cfg, LAYER_IDX)
    with torch.no_grad():
        indexer.q_layernorm.weight.normal_(0.0, 0.1)
        indexer.k_layernorm.weight.normal_(0.0, 0.1)
    rotary = Qwen4ExpTextRotaryEmbedding(cfg)
    return indexer.to(dtype).eval(), rotary


def position_embeddings(rotary, hidden: torch.Tensor, kv_length: int):
    pos = torch.arange(kv_length).view(1, 1, -1).expand(3, hidden.shape[0], -1)
    return rotary(hidden, pos)


def causal_mask(n_queries: int, kv_length: int, batch: int = 1) -> torch.Tensor:
    past = kv_length - n_queries
    q_pos = torch.arange(n_queries)[:, None] + past
    return (torch.arange(kv_length)[None, :] <= q_pos).view(1, 1, n_queries, kv_length).expand(batch, 1, -1, -1)


def assert_masks_equal(hf: torch.Tensor, mine: torch.Tensor, ties: torch.Tensor, label: str) -> None:
    assert hf.shape == mine.shape and hf.dtype == mine.dtype, (label, hf.shape, mine.shape, hf.dtype, mine.dtype)
    n_ties = int(ties.sum())
    TIES[label] = (n_ties, ties.numel())
    if n_ties:
        print(f"[{label}] {n_ties} queries have an exact score tie at the top-k boundary: "
              f"{ties.nonzero().tolist()[:20]}")
    if torch.equal(hf, mine):
        return
    diff_rows = (hf != mine).flatten(-1).any(-1).squeeze(1)  # [B, Q]
    bad = diff_rows.nonzero().tolist()
    tied_bad = (diff_rows & ties).nonzero().tolist()
    pytest.fail(
        f"[{label}] mask mismatch on {len(bad)} queries (first {bad[:10]}); of these {len(tied_bad)} have an "
        f"exact top-k boundary tie ({tied_bad[:10]}); total tie queries {n_ties}"
    )


def check_counts(mask: torch.Tensor, vis: torch.Tensor, budget: int, ratio: int) -> None:
    """Every query keeps all its visible tokens up to `budget` complete-block tokens, plus the tail."""
    lengths = vis.sum(-1)
    expect = torch.where(lengths // ratio > budget // ratio, budget + lengths % ratio, lengths)
    assert torch.equal(mask.sum(-1), expect)
    assert not bool((mask & ~vis).any())


def naive_recent_mask(indexer, q, raw_keys, cos, sin, vis: torch.Tensor) -> torch.Tensor:
    """Per-query Python reference of tie_rule "recent": HF's loop (HF's own norm module and rope), with the
    top-k replaced by sorted() on (score descending, block index descending). Returns bool [B, 1, Q, KV]."""
    ratio, k = indexer.compress_ratio, indexer.block_topk
    batch, _, n_queries, kv_length = vis.shape
    out = torch.zeros((batch, 1, n_queries, kv_length), dtype=torch.bool)
    for b in range(batch):
        keys = raw_keys[b]
        for i in range(n_queries):
            visible = torch.nonzero(vis[b, 0, i], as_tuple=False).flatten()
            n = visible.numel() // ratio
            if n > 0:
                blocks = visible[: n * ratio].view(n, ratio)
                pooled = keys.index_select(0, blocks.flatten()).view(n, ratio, -1).float().mean(dim=1)
                pooled = indexer.k_layernorm(pooled.to(keys.dtype))
                block_keys = apply_rotary_pos_emb(
                    pooled.unsqueeze(1), cos=cos[b].index_select(0, blocks[:, 0]),
                    sin=sin[b].index_select(0, blocks[:, 0]),
                ).squeeze(1)
                s = torch.matmul(q[b, i].float(), block_keys.float().transpose(-1, -2)).transpose(-1, -2)
                s = (torch.relu(s).sum(dim=-1) / math.sqrt(q.shape[-1])).tolist()
                order = sorted(range(n), key=lambda j: (-s[j], -j))[:k]
                out[b, 0, i, blocks[order].flatten()] = True
            out[b, 0, i, visible[n * ratio :]] = True
    return out


def check_recent_vs_hf(recent: torch.Tensor, hf: torch.Tensor, ties: torch.Tensor, label: str) -> torch.Tensor:
    """Untied queries must be equal under both rules; returns the [B, Q] queries where the rules differ."""
    diff = (recent != hf).any(dim=-1).squeeze(1)
    RECENT_VS_HF[label] = (int(diff.sum()), int(ties.sum()))
    untied_diff = (diff & ~ties).nonzero().tolist()
    assert not untied_diff, f"[{label}] 'recent' differs from 'hf' on untied queries {untied_diff[:10]}"
    return diff


@pytest.fixture(scope="module")
def tiny():
    cfg = make_config()
    indexer, rotary = make_indexer(cfg, torch.bfloat16)
    return cfg, indexer, rotary


@pytest.mark.parametrize("seq_len", [1, 3, 4, 2051, 2052, 3000, 4096])
def test_single_forward_matches_hf(tiny, seq_len, capsys):
    cfg, indexer, rotary = tiny
    torch.manual_seed(1000 + seq_len)
    hidden = torch.randn(1, seq_len, HIDDEN).to(torch.bfloat16)
    pe = position_embeddings(rotary, hidden, seq_len)
    mask = causal_mask(seq_len, seq_len)
    with torch.no_grad():
        t0 = time.perf_counter()
        hf = indexer(hidden, pe, mask, None)
        t1 = time.perf_counter()
        mine, ties = qsa.indexer_forward(indexer, hidden, pe, mask, None, return_ties=True)
        t2 = time.perf_counter()
    TIMINGS[seq_len] = (t1 - t0, t2 - t1)
    assert_masks_equal(hf, mine, ties, f"single {seq_len}")
    check_counts(mine, mask, cfg.indexer_budget, cfg.indexer_compress_ratio)
    if seq_len == 4096:
        with capsys.disabled():
            print(f"\n[timing] seq 4096 single forward: HF loop {t1 - t0:.3f} s, vectorized {t2 - t1:.3f} s")


@pytest.mark.parametrize("seq_len", [2052, 3000, 4096])
def test_recent_tie_rule_matches_naive(tiny, seq_len):
    """tie_rule="recent" equals the per-query sorted() reference; it differs from HF only on tied queries."""
    cfg, indexer, rotary = tiny
    torch.manual_seed(1000 + seq_len)  # same inputs as test_single_forward_matches_hf
    hidden = torch.randn(1, seq_len, HIDDEN).to(torch.bfloat16)
    pe = position_embeddings(rotary, hidden, seq_len)
    mask = causal_mask(seq_len, seq_len)
    with torch.no_grad():
        hf = indexer(hidden, pe, mask, None)
        recent, ties = qsa.indexer_forward(indexer, hidden, pe, mask, None, return_ties=True, tie_rule="recent")
        q, keys = qsa.project(indexer, hidden, pe)
        naive = naive_recent_mask(indexer, q, keys, pe[0], pe[1], mask)
    assert torch.equal(recent, naive)
    check_counts(recent, mask, cfg.indexer_budget, cfg.indexer_compress_ratio)
    diff = check_recent_vs_hf(recent, hf, ties, f"single {seq_len}")
    if seq_len == 3000:
        assert bool(diff.any()), "expected 'recent' to differ from 'hf' on at least one tied query at 3000"


def test_chunked_prefill_with_cache_matches_hf(tiny):
    """Prefill 2000 then 300 more tokens through a DynamicCache (HF) vs select_tokens on the full history."""
    cfg, indexer, rotary = tiny
    n0, n1 = 2000, 300
    total = n0 + n1
    torch.manual_seed(7)
    hidden = torch.randn(1, total, HIDDEN).to(torch.bfloat16)
    pe_all = position_embeddings(rotary, hidden, total)
    pe_0 = tuple(x[:, :n0] for x in pe_all)
    cache = DynamicCache(config=cfg)
    my_cache = DynamicCache(config=cfg)
    with torch.no_grad():
        hf0 = indexer(hidden[:, :n0], pe_0, causal_mask(n0, n0), cache)
        hf1 = indexer(hidden[:, n0:], pe_all, causal_mask(n1, total), cache)
        # select_tokens on the explicit full key history (keys projected per chunk, as the runtime does)
        _, k0 = qsa.project(indexer, hidden[:, :n0], pe_0)
        q1, k1 = qsa.project(indexer, hidden[:, n0:], pe_all)
        keys = torch.cat([k0, k1], dim=1)
        assert torch.equal(keys, cache.layers[LAYER_IDX].indexer_keys)
        mine1, ties1 = qsa.select_tokens(
            q1, keys, pe_all[0], pe_all[1], causal_mask(n1, total),
            indexer.k_layernorm.weight, indexer.k_layernorm.eps,
            budget=cfg.indexer_budget, compress_ratio=cfg.indexer_compress_ratio, return_ties=True,
        )
        # and the drop-in wrapper with its own cache
        w0, wt0 = qsa.indexer_forward(indexer, hidden[:, :n0], pe_0, causal_mask(n0, n0), my_cache, return_ties=True)
        w1, wt1 = qsa.indexer_forward(indexer, hidden[:, n0:], pe_all, causal_mask(n1, total), my_cache,
                                      return_ties=True)
        # tie_rule="recent": select_tokens on the full history, and the wrapper through its own cache
        recent1, rties1 = qsa.select_tokens(
            q1, keys, pe_all[0], pe_all[1], causal_mask(n1, total),
            indexer.k_layernorm.weight, indexer.k_layernorm.eps,
            budget=cfg.indexer_budget, compress_ratio=cfg.indexer_compress_ratio, return_ties=True,
            tie_rule="recent",
        )
        recent_cache = DynamicCache(config=cfg)
        r0 = qsa.indexer_forward(indexer, hidden[:, :n0], pe_0, causal_mask(n0, n0), recent_cache, tie_rule="recent")
        r1 = qsa.indexer_forward(indexer, hidden[:, n0:], pe_all, causal_mask(n1, total), recent_cache,
                                 tie_rule="recent")
        naive1 = naive_recent_mask(indexer, q1, keys, pe_all[0], pe_all[1], causal_mask(n1, total))
    assert_masks_equal(hf1, mine1, ties1, "chunk 2 select_tokens")
    assert_masks_equal(hf0, w0, wt0, "chunk 1 wrapper")
    assert_masks_equal(hf1, w1, wt1, "chunk 2 wrapper")
    check_counts(mine1, causal_mask(n1, total), cfg.indexer_budget, cfg.indexer_compress_ratio)
    assert torch.equal(recent1, naive1)
    assert torch.equal(r1, naive1)
    assert torch.equal(r0, hf0)  # 2000 tokens: at most 500 blocks, everything is selected under both rules
    check_recent_vs_hf(recent1, hf1, rties1, "chunk 2")


# Small budget (32 tokens = 8 blocks): exercises top-k on every query beyond 35 tokens, the left-padding
# fast path, the arbitrary-mask fallback and the eager (float) mask form, cheaply.
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mask_kind", ["causal", "left_pad", "random", "float_causal"])
def test_small_budget_masks_match_hf(dtype, mask_kind):
    cfg = make_config(budget=32, ratio=4)
    indexer, rotary = make_indexer(cfg, dtype, seed=3)
    batch, seq = 2, 301
    gen = torch.Generator().manual_seed(11)
    hidden = torch.randn(batch, seq, HIDDEN, generator=gen).to(dtype)
    pe = position_embeddings(rotary, hidden, seq)
    vis = causal_mask(seq, seq, batch).clone()
    if mask_kind == "left_pad":
        vis[1, :, :, :5] = False  # batch row 1 left-padded by 5 tokens
    elif mask_kind == "random":
        vis[1] &= torch.rand(1, seq, seq, generator=gen) < 0.7  # row 0 stays causal, row 1 is arbitrary
    mask = vis
    if mask_kind == "float_causal":
        mask = torch.zeros(vis.shape, dtype=dtype).masked_fill(~vis, torch.finfo(dtype).min)
    with torch.no_grad():
        hf = indexer(hidden, pe, mask, None)
        mine, ties = qsa.indexer_forward(indexer, hidden, pe, mask, None, query_chunk=64, return_ties=True)
        recent = qsa.indexer_forward(indexer, hidden, pe, mask, None, query_chunk=64, tie_rule="recent")
        q, keys = qsa.project(indexer, hidden, pe)
        naive = naive_recent_mask(indexer, q, keys, pe[0], pe[1], vis)
    assert_masks_equal(hf, mine, ties, f"small {mask_kind} {dtype}")
    if mask_kind == "float_causal":
        recent, hf = recent == 0, hf == 0
    else:
        check_counts(mine, vis, cfg.indexer_budget, cfg.indexer_compress_ratio)
    assert torch.equal(recent, naive)
    check_recent_vs_hf(recent, hf, ties, f"small {mask_kind} {dtype}")


def test_report(capsys):
    """Prints HF vs vectorized runtimes, the boundary-tie counts and the 'recent' vs 'hf' differences."""
    with capsys.disabled():
        print("\n[timing] seq: HF loop s / vectorized s")
        for seq_len, (t_hf, t_mine) in sorted(TIMINGS.items()):
            print(f"[timing] {seq_len:5d}: {t_hf:8.3f} / {t_mine:7.3f}")
        for label, (n_ties, n_queries) in TIES.items():
            print(f"[ties] {label}: {n_ties} of {n_queries} queries tied at the top-k boundary")
        for label, (n_diff, n_ties) in RECENT_VS_HF.items():
            print(f"[recent vs hf] {label}: {n_diff} queries differ, all within the {n_ties} tied queries")
