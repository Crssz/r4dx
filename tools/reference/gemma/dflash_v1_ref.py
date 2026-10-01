"""CPU numpy reference of one z-lab DFlash v1 draft block (Qwen3-style `DFlashDraftModel`, e.g.
z-lab/gemma4-12B-it-DFlash), written from the z-lab/dflash semantics (dflash/model.py), and the equivalence
test that is docs/gemma4-plan.md D-2 / D-3's CPU gate:

    the r4dx dflash2 forward (tools/reference/dflash2_ref.py::draft_round) run on the container that
    `r4dx-convert --dflash-hf` writes -- identity conv (base[side][tap0] = 1, [side][tap1] = 0, conv.proj = 0)
    and zero selector (hidden = 0, both codebooks = 0) -- produces EXACTLY the v1 per-position argmax.

v1 semantics encoded here (read from z-lab/dflash/model.py, recorded in src/convert/.../dflash2_hf.hpp):
  * context path:  g = hidden_norm(fc(concat(target features)))   -- k_proj / v_proj applied to g DIRECTLY, no
    input_layernorm; K = rope(k_norm(k_proj g)), V = v_proj g, at the feature's absolute position;
  * draft block:   x = embed(ids) * input_embedding_scale (default 1.0: the RAW target table rows);
    per layer: h = input_layernorm(x); q,k,v = proj(h); q_norm/k_norm (per head, plain RMS); rope(q at the block
    positions, k at [ctx positions ; block positions]); attention over [ctx ; block] keys: sliding layers CAUSAL within the block (z-lab is_causal), full layer
    unrestricted; sliding layers restricted to `q_pos - k_pos < sliding_window`, then o_proj, residual; post_attention_layernorm,
    SwiGLU MLP, residual;
  * output:        norm -> target lm_head -> optional `cap * tanh(l / cap)` (final_logit_softcapping);
    draft token i (block position i >= 1) = argmax of row i.
Everything is plain fp32/fp64 numpy on tiny random weights. No GPU, no torch (HIP/CUDA are hidden anyway).

Run:  <reference venv>\\python.exe tools\\reference\\gemma\\dflash_v1_ref.py        (self-test, prints PASS)
      <reference venv>\\python.exe -m pytest tools\\reference\\gemma\\dflash_v1_ref.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ["HIP_VISIBLE_DEVICES"] = "-1"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dflash2_ref as d2  # noqa: E402


def rms(x, w, eps):
    x64 = x.astype(np.float64)
    return ((x64 / np.sqrt(np.mean(x64 * x64, axis=-1, keepdims=True) + eps)).astype(np.float32)) * w


def make_v1_weights(rng, hidden, layers, ffn, heads, kv_heads, hd, n_feat_layers):
    """A random HF-named v1 state dict (fp32 values that are exactly representable in bf16)."""

    def bf16(a):
        u = a.astype(np.float32).view(np.uint32)
        return ((u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000).view(np.float32)

    def lin(n, k):
        return bf16(rng.randn(n, k).astype(np.float32) * 0.08)

    def nrm(n):
        return bf16(1.0 + 0.1 * rng.randn(n).astype(np.float32))

    sd = {"fc.weight": lin(hidden, n_feat_layers * hidden), "hidden_norm.weight": nrm(hidden), "norm.weight": nrm(hidden)}
    for i in range(layers):
        p = f"layers.{i}."
        sd[p + "input_layernorm.weight"] = nrm(hidden)
        sd[p + "post_attention_layernorm.weight"] = nrm(hidden)
        sd[p + "self_attn.q_proj.weight"] = lin(heads * hd, hidden)
        sd[p + "self_attn.k_proj.weight"] = lin(kv_heads * hd, hidden)
        sd[p + "self_attn.v_proj.weight"] = lin(kv_heads * hd, hidden)
        sd[p + "self_attn.o_proj.weight"] = lin(hidden, heads * hd)
        sd[p + "self_attn.q_norm.weight"] = nrm(hd)
        sd[p + "self_attn.k_norm.weight"] = nrm(hd)
        sd[p + "mlp.gate_proj.weight"] = lin(ffn, hidden)
        sd[p + "mlp.up_proj.weight"] = lin(ffn, hidden)
        sd[p + "mlp.down_proj.weight"] = lin(hidden, ffn)
    return sd


def v1_draft_tokens(sd, cfg, embed, lm_head, feats, ctx_pos, anchor_id, windows, softcap, embed_scale, causal=None):
    """The v1 block forward. Returns (drafted tokens for block positions 1.., final-normed [B, H], logits [B, V])."""
    H, nh, nkv, hd, B = cfg["hidden"], cfg["heads"], cfg["kv_heads"], cfg["hd"], cfg["block"]
    eps, theta, mask_id = cfg["eps"], cfg["theta"], cfg["mask_id"]
    n = len(ctx_pos)
    g = rms(feats @ sd["fc.weight"].T, sd["hidden_norm.weight"], eps)  # [n, H]
    ids = np.array([anchor_id] + [mask_id] * (B - 1))
    x = (embed[ids] * np.float32(embed_scale)).astype(np.float32)
    bpos = np.arange(B) + n
    for il in range(cfg["layers"]):
        p = f"layers.{il}."
        h = rms(x, sd[p + "input_layernorm.weight"], eps)
        q = (h @ sd[p + "self_attn.q_proj.weight"].T).reshape(B, nh, hd)
        k_noise = (h @ sd[p + "self_attn.k_proj.weight"].T).reshape(B, nkv, hd)
        v_noise = (h @ sd[p + "self_attn.v_proj.weight"].T).reshape(B, nkv, hd)
        k_ctx = (g @ sd[p + "self_attn.k_proj.weight"].T).reshape(n, nkv, hd)
        v_ctx = (g @ sd[p + "self_attn.v_proj.weight"].T).reshape(n, nkv, hd)
        qn, kw = sd[p + "self_attn.q_norm.weight"], sd[p + "self_attn.k_norm.weight"]
        q = d2.rope_neox(rms(q, qn, eps), bpos, theta, hd)
        k = np.concatenate([rms(k_ctx, kw, eps), rms(k_noise, kw, eps)], axis=0)
        kpos = np.concatenate([np.asarray(ctx_pos), bpos])
        k = d2.rope_neox(k, kpos, theta, hd)
        v = np.concatenate([v_ctx, v_noise], axis=0)
        win = windows[il]
        mask = np.ones((B, n + B), dtype=bool) if win is None else (bpos[:, None] - kpos[None, :] < win)
        if causal is not None and causal[il]:  # z-lab: is_causal = (layer_type == "sliding_attention")
            mask = mask & (kpos[None, :] <= bpos[:, None])
        a = d2.attention_gqa(q, k, v, mask, 1.0 / np.sqrt(hd))
        x = x + a @ sd[p + "self_attn.o_proj.weight"].T
        h2 = rms(x, sd[p + "post_attention_layernorm.weight"], eps)
        gt, up = h2 @ sd[p + "mlp.gate_proj.weight"].T, h2 @ sd[p + "mlp.up_proj.weight"].T
        x = x + ((gt / (1.0 + np.exp(-gt))) * up) @ sd[p + "mlp.down_proj.weight"].T
    xf = rms(x, sd["norm.weight"], eps)
    logits = xf @ lm_head.T
    if softcap > 0:
        logits = (softcap * np.tanh(logits.astype(np.float64) / softcap)).astype(np.float32)
    return [int(i) for i in np.argmax(logits[1:], axis=-1)], xf, logits


def v1_to_dflash2(sd, cfg, vocab):
    """The Python mirror of ConvertDflashHf's tensor mapping + synthesis (src/convert/.../dflash2_hf.hpp)."""
    H, B, kv = cfg["hidden"], cfg["block"], 2
    rank, group = 256, 16
    layers = []
    for il in range(cfg["layers"]):
        p = f"layers.{il}."
        base = np.zeros((2, kv, H), dtype=np.float32)
        base[:, 0, :] = 1.0  # identity: tap0 = 1, tap1 = 0, for both sides
        dyn_n = 2 * kv * (H // group)
        layers.append(
            d2.LayerWeights(
                attn_norm=sd[p + "input_layernorm.weight"],
                attn_q=sd[p + "self_attn.q_proj.weight"],
                attn_k=sd[p + "self_attn.k_proj.weight"],
                attn_v=sd[p + "self_attn.v_proj.weight"],
                attn_output=sd[p + "self_attn.o_proj.weight"],
                attn_q_norm=sd[p + "self_attn.q_norm.weight"],
                attn_k_norm=sd[p + "self_attn.k_norm.weight"],
                attn_conv_base=base.copy(),
                attn_conv_proj=np.zeros((dyn_n, H), dtype=np.float32),
                ffn_norm=sd[p + "post_attention_layernorm.weight"],
                ffn_gate=sd[p + "mlp.gate_proj.weight"],
                ffn_up=sd[p + "mlp.up_proj.weight"],
                ffn_down=sd[p + "mlp.down_proj.weight"],
                ffn_conv_base=base.copy(),
                ffn_conv_proj=np.zeros((dyn_n, H), dtype=np.float32),
            )
        )
    c = d2.DFlash2Config(
        n_embd=H, n_ff=cfg["ffn"], n_head=cfg["heads"], n_head_kv=cfg["kv_heads"], head_dim=cfg["hd"],
        n_layer=cfg["layers"], rope_theta=cfg["theta"], rms_eps=cfg["eps"], block_size=B, conv_kernel_size=2,
        conv_group_size=group, selector_rank=rank, selector_top_k=16, target_layers=cfg["target_layers"],
        sliding_window=cfg["window"], mask_token_id=cfg["mask_id"], vocab_size=vocab,
        n_embd_inp_enc=len(cfg["target_layers"]) * H,
    )
    return d2.DFlash2Weights(
        cfg=c, layers=layers, fc=sd["fc.weight"], enc_output_norm=sd["hidden_norm.weight"],
        output_norm=sd["norm.weight"], selector_hidden=np.zeros((rank, H), dtype=np.float32),
        selector_predecessor=np.zeros((vocab, rank), dtype=np.float32),
        selector_successor=np.zeros((vocab, rank), dtype=np.float32),
    )


class TiedTarget(d2.TargetProvider):
    """Target embed table == lm_head (tied, as Gemma 4)."""

    def __init__(self, table):
        super().__init__(table.shape[1])
        self.t = table

    def embed_rows(self, ids):
        return self.t[ids]

    def lm_head(self):
        return self.t

    @property
    def vocab_size(self):
        return self.t.shape[0]


def make_cfg(window):
    return dict(hidden=64, heads=4, kv_heads=2, hd=16, layers=3, ffn=128, block=16, eps=1e-6, theta=1e6,
                mask_id=4, window=window, target_layers=[2, 11], vocab=512)


def run_case(seed, n_ctx, window, softcap, embed_scale, full_layer_window="same"):
    rng = np.random.RandomState(seed)
    cfg = make_cfg(window)
    sd = make_v1_weights(rng, cfg["hidden"], cfg["layers"], cfg["ffn"], cfg["heads"], cfg["kv_heads"], cfg["hd"],
                         len(cfg["target_layers"]))
    embed = (rng.randn(cfg["vocab"], cfg["hidden"]) * 0.5).astype(np.float32)
    feats = (rng.randn(n_ctx, len(cfg["target_layers"]) * cfg["hidden"]) * 1.0).astype(np.float32)
    anchor = 17
    # layer_types [sliding, sliding, full]: the dflash2 runtime applies `window` to every layer (its finite cap);
    # `full_layer_window="same"` gives the v1 reference the same finite window so the two are comparable, None
    # models true full attention (documented difference once n_ctx + block > window).
    windows = [window, window, window if full_layer_window == "same" else None]
    causal = [True, True, False]  # z-lab: is_causal = (layer_type == sliding_attention); see dflash_hf_golden.py
    toks_v1, xf_v1, logits_v1 = v1_draft_tokens(sd, cfg, embed, embed, feats, np.arange(n_ctx), anchor, windows,
                                                softcap, embed_scale, causal=causal)

    w = v1_to_dflash2(sd, cfg, cfg["vocab"])
    cache = d2.DraftKvCache.empty(cfg["layers"], cfg["kv_heads"], cfg["hd"])
    d2.inject(w, cache, d2.encode_features(w, feats), np.arange(n_ctx))
    res = d2.draft_round(w, TiedTarget(embed), cache, anchor, p_min=0.0, capture_layer0=True, embed_scale=embed_scale,
                         logit_softcap=softcap, block_causal_layers=causal)
    return toks_v1, xf_v1, res


def check_case(**kw):
    toks_v1, xf_v1, res = run_case(**kw)
    tok_d2 = res.tokens
    assert len(tok_d2) == 15, f"the walk must emit block-1 = 15 tokens, got {len(tok_d2)}"
    assert toks_v1 == tok_d2, f"v1 argmax {toks_v1} != dflash2-identity walk {tok_d2} for {kw}"
    err = float(np.max(np.abs(res.intermediates["x_final_normed"] - xf_v1)))
    assert err < 2e-4, f"x_final differs by {err} for {kw}"
    # conv identity is exact: the side-0 conv output IS the normed input, the side-1 output IS the sublayer output
    assert np.array_equal(res.intermediates["attn_conv_in_l0"], res.intermediates["attn_conv_x_l0"])
    return toks_v1


def test_identity_equals_v1_argmax():
    check_case(seed=1, n_ctx=40, window=128, softcap=30.0, embed_scale=1.0)
    check_case(seed=2, n_ctx=8, window=128, softcap=30.0, embed_scale=1.0)
    check_case(seed=3, n_ctx=40, window=128, softcap=0.0, embed_scale=1.0)
    # sliding window cuts the context (n_ctx 90 > window 24): sliding layers AND the capped full layer agree
    check_case(seed=4, n_ctx=90, window=24, softcap=30.0, embed_scale=1.0)


def test_embed_scale_matters_and_is_consistent():
    check_case(seed=5, n_ctx=40, window=128, softcap=30.0, embed_scale=1.0)
    check_case(seed=5, n_ctx=40, window=128, softcap=30.0, embed_scale=62.0)  # Gemma-scaled rows
    # (a tiny tied random model can keep the same argmax tokens at both scales, so compare the hidden state)
    _, xa, _ = run_case(seed=5, n_ctx=40, window=128, softcap=30.0, embed_scale=1.0)
    _, xb, _ = run_case(seed=5, n_ctx=40, window=128, softcap=30.0, embed_scale=62.0)
    assert float(np.max(np.abs(xa - xb))) > 1e-2, "embed_scale must change the draft (otherwise the A/B knob is dead)"


def test_softcap_is_monotone_top1_unchanged():
    t0 = check_case(seed=6, n_ctx=40, window=128, softcap=0.0, embed_scale=1.0)
    t1 = check_case(seed=6, n_ctx=40, window=128, softcap=30.0, embed_scale=1.0)
    assert t0 == t1, "cap * tanh(l / cap) is monotone: the per-position argmax cannot move"


def test_known_gap_true_full_layer_vs_finite_window():
    """Documented limit (docs/dflash2.md Gemma section): the runtime caps the full layer's window at 2048 (LDS
    score row). With n_ctx + block > window the true-full v1 layer sees more context than the runtime."""
    toks_full, _, res = run_case(seed=7, n_ctx=90, window=24, softcap=30.0, embed_scale=1.0, full_layer_window=None)
    toks_cap, _, _ = run_case(seed=7, n_ctx=90, window=24, softcap=30.0, embed_scale=1.0, full_layer_window="same")
    # Not an assertion about equality: report only (the two may or may not differ on a tiny random model).
    print(f"[known gap] true-full vs capped-window tokens equal: {toks_full == toks_cap}")
    assert len(res.tokens) == 15


if __name__ == "__main__":
    test_identity_equals_v1_argmax()
    test_embed_scale_matters_and_is_consistent()
    test_softcap_is_monotone_top1_unchanged()
    test_known_gap_true_full_layer_vs_finite_window()
    print("PASS")

