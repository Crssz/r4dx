"""Vectorized port of HF's QSA token selection for qwen4_exp (plan M0-6, section 4.10).

`transformers.models.qwen4_exp.Qwen4ExpTextQSAIndexer.forward` (modular_qwen4_exp.py:368-474) loops over
every (batch, query) pair in Python and re-pools, re-norms and re-ropes all visible key blocks per query.
That is fine at 4k and far too slow for the 8k/32k reference probes. `select_tokens` computes the same
selection mask after the indexer's projections:

1. visible tokens of a query are grouped in consecutive runs of `compress_ratio`; each complete block's raw
   keys are mean-pooled in fp32 and cast back to the keys' dtype,
2. `k_layernorm` ((1 + weight)-centred RMSNorm, `Qwen4ExpTextRMSNorm`) and rope at the block's first visible
   index (kv index into `cos_full` / `sin_full`),
3. score = sum over heads of relu(q_h . k_block) / sqrt(head_dim), in fp32,
4. keep the top min(budget // ratio, n_blocks) blocks plus the incomplete tail (at most ratio - 1 tokens).

Fast path: when, within a batch row, every query's visible set is a contiguous range [s, s + L_i) with the
same start s (causal masks, with or without left padding, chunked prefill and decode against a cache), the
blocks of every query are the same blocks s + ratio * j, so pooled block keys are computed once per row and
queries are scored in chunks of `query_chunk` with one matmul. Queries with n_blocks <= budget // ratio
select their whole visible range (HF's topk then returns every block), so they are not scored at all.
Any other visible mask falls back, per batch row, to a per-query loop with HF's exact op sequence.

Bit exactness: every op mirrors HF's (same dtypes, same op order, same reductions). Pooling, norm and
rope are row-independent; the score matmul is batched ([chunk * heads, d] x [d, n_blocks] instead of
HF's [heads, d] x [d, n_i]). On the CPU (torch 2.9.1, MKL) that gives bit-identical scores, checked by
test_qsa_indexer.py; other BLAS backends (GPU) are not guaranteed to. Exact score ties at the selection
boundary are common (a block whose heads all score <= 0 scores exactly 0) and make the selected set depend
on torch.topk's tie order for that exact input, which differs between a padded batched call and HF's
per-query call. With tie_rule "hf" (default) such queries are detected (k-th == (k+1)-th best score) and
rerun HF's own topk call on their unpadded scores, which reproduces HF's choice on the same device. With
tie_rule "recent" (plan Q23) the top-k is fully defined by (score descending, block index descending), so
the most recent tied blocks win, independent of topk internals. `return_ties=True` reports tied queries.
"""

from __future__ import annotations

import math

import torch


def rms_norm_centred(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """`Qwen4ExpTextRMSNorm.forward` without group_size: fp32 norm, times (1 + weight), cast back."""
    xf = x.float()
    out = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.type_as(x)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_partial_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, unsqueeze_dim: int) -> torch.Tensor:
    """HF `apply_rotary_pos_emb` for one tensor: rope on the first cos.shape[-1] dims, the rest unchanged."""
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    x_rope, x_nope = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rope = (x_rope * cos) + (_rotate_half(x_rope) * sin)
    return torch.cat([x_rope, x_nope], dim=-1)


def _block_scores(q_rows: torch.Tensor, block_keys_f32: torch.Tensor) -> torch.Tensor:
    """q_rows [..., heads, d] (any dtype), block keys [n, d] fp32 -> fp32 scores [..., n], as HF computes them."""
    scores = torch.matmul(q_rows.float(), block_keys_f32.transpose(-1, -2)).transpose(-1, -2)
    return torch.relu(scores).sum(dim=-1) / math.sqrt(q_rows.shape[-1])


def _recent_block_mask(scores: torch.Tensor, kth: torch.Tensor, k: int) -> torch.Tensor:
    """tie_rule "recent": top-k by (score descending, block index descending). kth [..., 1] is the k-th best
    score; its value is unique even though topk's index order on ties is not. Keeps every block above it plus
    the most recent blocks equal to it, k in total. Padding at -inf never equals a finite kth."""
    greater = scores > kth
    equal = scores == kth
    need = k - greater.sum(dim=-1, keepdim=True)
    equal_at_or_after = equal.flip(-1).cumsum(dim=-1).flip(-1)
    return greater | (equal & (equal_at_or_after <= need))


def _pooled_block_keys(
    keys: torch.Tensor,
    starts: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    ratio: int,
) -> torch.Tensor:
    """keys [n * ratio, d] (block-major), starts [n] kv indices -> pooled, normed, roped block keys [n, d]."""
    n, d = starts.shape[0], keys.shape[-1]
    pooled = keys.contiguous().view(n, ratio, d).float().mean(dim=1).to(keys.dtype)
    pooled = rms_norm_centred(pooled, k_weight, eps)
    return apply_partial_rope(
        pooled.unsqueeze(1), cos.index_select(0, starts), sin.index_select(0, starts), unsqueeze_dim=1
    ).squeeze(1)


def _common_start_ranges(vis: torch.Tensor) -> tuple[int, torch.Tensor] | None:
    """vis [Q, KV] bool. If every non-empty row is the range [s, s + L_i) for one s, return (s, L); else None."""
    lengths = vis.sum(dim=-1)
    nonempty = lengths > 0
    if not bool(nonempty.any()):
        return 0, lengths
    firsts = vis.int().argmax(dim=-1)[nonempty]
    start = int(firsts[0])
    if not bool((firsts == start).all()):
        return None
    pos = torch.arange(vis.shape[-1], device=vis.device)
    expected = (pos >= start) & (pos[None, :] < start + lengths[:, None])
    if not torch.equal(expected, vis):
        return None
    return start, lengths


def _select_row_ranges(
    out: torch.Tensor,
    ties: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    start: int,
    lengths: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    ratio: int,
    block_topk: int,
    query_chunk: int,
    return_ties: bool,
    tie_rule: str,
) -> None:
    """Fast path for one batch row. out [Q, KV] bool arrives equal to the visible mask and is narrowed in place."""
    n_blocks = lengths // ratio
    scored = torch.nonzero(n_blocks > block_topk, as_tuple=False).flatten()
    if scored.numel() == 0:
        return  # every query keeps all its visible blocks and its tail
    nb_max = int(n_blocks[scored].max())
    starts = start + ratio * torch.arange(nb_max, device=keys.device)
    block_keys = _pooled_block_keys(keys[start : start + nb_max * ratio], starts, cos, sin, k_weight, eps, ratio)
    block_keys_f32 = block_keys.float()
    for c0 in range(0, scored.numel(), query_chunk):
        rows = scored[c0 : c0 + query_chunk]
        nb_rows = n_blocks[rows]
        nbc = int(nb_rows.max())
        scores = _block_scores(q[rows], block_keys_f32[:nbc])  # [c, nbc]
        block_pos = torch.arange(nbc, device=scores.device)
        scores = scores.masked_fill(block_pos[None, :] >= nb_rows[:, None], float("-inf"))
        # Rows here have n_blocks > block_topk, so HF's topk(min(block_topk, n)) is topk(block_topk) and the
        # -inf padding (scores are >= 0) is never selected. topk(block_topk + 1) exposes boundary ties: without
        # one, the first block_topk sorted entries are the unique top-k set and both tie rules agree. With one
        # (common: blocks whose heads all score <= 0 tie at exactly 0), "hf" reruns HF's own call on the row's
        # unpadded 1D scores (the set depends on topk's tie order for that exact input) and "recent" keeps the
        # most recent tied blocks.
        top = scores.topk(block_topk + 1, dim=-1)
        block_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(1, top.indices[:, :block_topk], True)
        if block_topk > 0:
            tied = top.values[:, block_topk - 1] == top.values[:, block_topk]
            if tie_rule == "recent":
                block_mask = _recent_block_mask(scores, top.values[:, block_topk - 1 : block_topk], block_topk)
            else:
                for r in torch.nonzero(tied, as_tuple=False).flatten().tolist():
                    hf_selected = scores[r, : int(nb_rows[r])].contiguous().topk(block_topk, dim=0).indices
                    block_mask[r] = False
                    block_mask[r, hf_selected] = True
            if return_ties:
                ties[rows] = tied
        token_pos = torch.arange(nbc * ratio, device=scores.device)
        keep = block_mask.repeat_interleave(ratio, dim=1) | (token_pos[None, :] >= (nb_rows * ratio)[:, None])
        out[rows, start : start + nbc * ratio] &= keep


def _select_row_generic(
    out: torch.Tensor,
    ties: torch.Tensor,
    q: torch.Tensor,
    keys: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    vis: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    ratio: int,
    block_topk: int,
    return_ties: bool,
    tie_rule: str,
) -> None:
    """HF's per-query loop for one batch row with an arbitrary visible mask. out [Q, KV] bool arrives all False."""
    for qi in range(vis.shape[0]):
        visible = torch.nonzero(vis[qi], as_tuple=False).flatten()
        n = visible.shape[-1] // ratio
        if n > 0:
            block_tokens = visible[: n * ratio].view(n, ratio)
            block_keys = _pooled_block_keys(
                keys.index_select(0, block_tokens.flatten()), block_tokens[:, 0], cos, sin, k_weight, eps, ratio
            )
            scores = _block_scores(q[qi], block_keys.float())
            k = min(block_topk, n)
            top = scores.topk(min(k + 1, n), dim=0)
            if tie_rule == "recent" and 0 < k < n:
                selected = torch.nonzero(_recent_block_mask(scores, top.values[k - 1 : k], k), as_tuple=False)
                selected = selected.flatten()
            else:
                selected = scores.topk(k, dim=0).indices
            if return_ties and 0 < k < n:
                ties[qi] = top.values[k - 1] == top.values[k]
            out[qi, block_tokens.index_select(0, selected).flatten()] = True
        out[qi, visible[n * ratio :]] = True


@torch.no_grad()
def select_tokens(
    q: torch.Tensor,
    raw_keys: torch.Tensor,
    cos_full: torch.Tensor,
    sin_full: torch.Tensor,
    visible_mask: torch.Tensor,
    k_layernorm_weight: torch.Tensor,
    eps: float,
    budget: int = 2048,
    compress_ratio: int = 4,
    *,
    query_chunk: int = 256,
    return_ties: bool = False,
    tie_rule: str = "hf",
):
    """QSA selection mask, equal to `Qwen4ExpTextQSAIndexer.forward`'s return value.

    q            [B, Q, heads, d]  after q_layernorm and rope at the query positions (HF's `q`).
    raw_keys     [B, KV, d]        un-normed, un-roped token keys of the full history (HF's `raw_keys`
                                   after `update_indexer`).
    cos_full/sin_full [B, KV, rotary_dim]  rotary tables for every kv index (HF's full position_embeddings).
    visible_mask [B, 1, Q, KV]     bool (True = visible) or additive float (0 = visible), as HF accepts.

    Returns [B, 1, Q, KV]: bool, or for a float visible_mask the additive form (0 where selected, finfo.min
    elsewhere) in its dtype. With return_ties=True also returns a bool [B, Q] tensor marking queries whose
    k-th and (k+1)-th best block scores are exactly equal (their selected set depends on the tie rule).

    tie_rule "hf" (default) reproduces HF's torch.topk choice among tied blocks bit for bit; "recent" is a
    fully defined top-k by (score descending, block index descending): among equal scores the most recent
    blocks win. The incomplete tail is kept either way.
    """
    if tie_rule not in ("hf", "recent"):
        raise ValueError(f"tie_rule must be 'hf' or 'recent', got {tie_rule!r}")
    batch, n_queries = q.shape[0], q.shape[1]
    kv_length = visible_mask.shape[-1]
    if visible_mask.shape != (batch, 1, n_queries, kv_length):
        raise ValueError(f"visible_mask shape {tuple(visible_mask.shape)} != {(batch, 1, n_queries, kv_length)}")
    if raw_keys.shape[:2] != (batch, kv_length) or raw_keys.shape[-1] != q.shape[-1]:
        raise ValueError(f"raw_keys shape {tuple(raw_keys.shape)} does not match q {tuple(q.shape)} / kv {kv_length}")
    vis_all = visible_mask if visible_mask.dtype == torch.bool else visible_mask == 0
    block_topk = budget // compress_ratio

    out = torch.zeros((batch, n_queries, kv_length), dtype=torch.bool, device=visible_mask.device)
    ties = torch.zeros((batch, n_queries), dtype=torch.bool, device=visible_mask.device)
    for b in range(batch):
        vis = vis_all[b, 0]
        ranges = _common_start_ranges(vis)
        if ranges is not None:
            out[b] = vis
            start, lengths = ranges
            _select_row_ranges(
                out[b], ties[b], q[b], raw_keys[b], cos_full[b], sin_full[b], start, lengths,
                k_layernorm_weight, eps, compress_ratio, block_topk, query_chunk, return_ties, tie_rule,
            )
        else:
            _select_row_generic(
                out[b], ties[b], q[b], raw_keys[b], cos_full[b], sin_full[b], vis,
                k_layernorm_weight, eps, compress_ratio, block_topk, return_ties, tie_rule,
            )

    mask = out.unsqueeze(1)
    if visible_mask.is_floating_point():
        min_dtype = torch.finfo(visible_mask.dtype).min
        mask = torch.where(mask, visible_mask.new_zeros(()), min_dtype)
    return (mask, ties) if return_ties else mask


@torch.no_grad()
def project(indexer, hidden_states: torch.Tensor, position_embeddings) -> tuple[torch.Tensor, torch.Tensor]:
    """HF indexer projections with the module's own weights: (q after q_layernorm + rope, raw keys [B, S, d])."""
    batch, seq, _ = hidden_states.shape
    d = indexer.index_head_dim
    full_cos, full_sin = position_embeddings
    qk = indexer.index_qk_proj(hidden_states)
    q, token_k = torch.split(
        qk, [indexer.index_n_heads * d, indexer.index_kv_heads * d], dim=-1
    )
    q, raw_keys = q.reshape(batch, seq, -1, d), token_k.reshape(batch, seq, -1, d).squeeze(2)
    q = indexer.q_layernorm(q)
    q = apply_partial_rope(q, full_cos[:, -seq:, :], full_sin[:, -seq:, :], unsqueeze_dim=2)
    return q, raw_keys


@torch.no_grad()
def indexer_forward(
    indexer,
    hidden_states: torch.Tensor,
    position_embeddings,
    attention_mask: torch.Tensor,
    past_key_values=None,
    *,
    query_chunk: int = 256,
    return_ties: bool = False,
    tie_rule: str = "hf",
):
    """Drop-in for `Qwen4ExpTextQSAIndexer.forward(hidden_states, position_embeddings, attention_mask,
    past_key_values)` that reuses the module's weights and runs `select_tokens`."""
    q, raw_keys = project(indexer, hidden_states, position_embeddings)
    if past_key_values is not None:
        raw_keys = past_key_values.update_indexer(raw_keys, indexer.layer_idx)
    full_cos, full_sin = position_embeddings
    return select_tokens(
        q, raw_keys, full_cos, full_sin, attention_mask,
        indexer.k_layernorm.weight, indexer.k_layernorm.eps,
        budget=indexer.token_budget, compress_ratio=indexer.compress_ratio,
        query_chunk=query_chunk, return_ties=return_ties, tie_rule=tie_rule,
    )
