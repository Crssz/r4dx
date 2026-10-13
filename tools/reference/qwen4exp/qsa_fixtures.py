"""Fixtures for the QSA attention core (src/model/qwen4exp: QsaLayerCore; test tests/model/qwen4exp/test_gpu_qsa.cpp).

Builds HF's `Qwen4ExpTextAttention` (layer index 3, REAL dims: hidden 2560, 24 q heads, 2 kv heads, head_dim 256,
partial rotary 0.25, theta 1e7, indexer 4 + 1 heads x 128, ratio 4) with random weights in bfloat16 on the CPU and
writes, for chunked runs of it, the stage inputs (the q/k/v/index projection outputs), HF's gated attention output
(the `o_proj` input) and the selected token sets.

Run (no GPU needed):
    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools/reference/qwen4exp/qsa_fixtures.py [--out DIR] [--only NAME ...]

Output dir (override with --out): E:\\models\\r4dx\\q4exp-fixtures\\QsaCore
    index.txt   one line per tensor: `<key> <bf16|f32|i32> <rank> <d0> .. <dn-1>`; data = raw little-endian `<key>.bin`
    params.txt  `<key> <integer or float>` lines

What is isolated. The kernels are compared with HF run on the SAME fp8 KV: K and V (after HF's own q/k norm and rope)
are quantized to e4m3 with the fixture's static per-kv-head descales exactly as r4dx writes them
(fp8(bf16 * (1 / descale)), saturating), and HF's attention runs on the dequantized values (bf16(fp8 * descale)).
The indexer's top-k loop is replaced by `qsa_indexer.indexer_forward(tie_rule="recent")` (plan Q23). Two expected
outputs per chunk: `out` (HF bf16 graph: bf16 scores, bf16 probabilities) and `out32` (the same selected sets, q as
HF computes it, K / V dequantized in fp32, scores / softmax / P.V in fp32, then HF's two bf16 roundings and the gate).

Cases (name: chunk lengths, indexer budget):
  attention, budget 64 (top 16 blocks; dense-exact up to 67 tokens), the prefill chunks first then decode steps:
    b64_p1 [1, 1x6]; b64_p5 [5, 1x6]; b64_p64 [64, 1x6]; b64_p70 [70, 1x8]; b64_p300 [256, 44, 1x8];
    b64_c3x100 [100, 100, 100, 1x8]; b64_mid [37, 61, 103, 1x5] (chunks start mid-block, blocks complete inside chunks)
  attention, budget 2048: b2048_p2052 [256x7, 255, 5, 1x6] (the first sparse query is position 2051);
    b2048_p2400 [256x9, 97, 1x3] (349 sparse rows)
  selection only, budget 2048, random hidden through the indexer projection: sel2051, sel2052, sel3000, sel8192

Per attention case `<case>`: params `<case>.budget`, `.nchunks`, `.chunk<i>.start`, `.chunk<i>.len`, `.nblocks`;
tensors `<case>.q_norm .k_norm .idx_q_norm .idx_k_norm` (bf16 raw weights), `.k_descale .v_descale` (f32 [2]),
`.inv_freq` (f32 [32], HF's table), `.blockkeys` (bf16 [nblocks, 128], HF pooled / normed / roped keys of every
complete block at the end), per chunk: `.c<i>.qg [T,12288] .k [T,512] .v [T,512] .idxqk [T,640]` (bf16),
`.c<i>.out [T,6144] .out32 [T,6144]` (bf16), `.c<i>.sel [T, budget+3]` (i32 token lists, -1 padded) and
`.c<i>.selcount [T]` (i32; -1 = dense row), `.c<i>.kq / .vq [T, 2, 256]` (i32 fp8 bytes written for the chunk).
Selection-only cases have `.idxqk`, `.idxq` (HF q after norm + rope), `.q_norm`, `.k_norm`, `.sel`, `.selcount`, `.blockkeys` and `.ties` (param).
"""

import argparse
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig  # noqa: E402
from transformers.models.qwen4_exp.modeling_qwen4_exp import (  # noqa: E402
    Qwen4ExpTextAttention,
    Qwen4ExpTextQSAIndexer,
    Qwen4ExpTextRotaryEmbedding,
    apply_rotary_pos_emb,
)

from qwen4exp import qsa_indexer as qsa  # noqa: E402

BF = torch.bfloat16
HIDDEN = 2560
HQ, HKV, HD = 24, 2, 256
LAYER_IDX = 3
RATIO = 4
FP8_MAX = 448.0

ATTN_CASES = {
    "b64_p1": (64, [1] + [1] * 6),
    "b64_p5": (64, [5] + [1] * 6),
    "b64_p64": (64, [64] + [1] * 6),
    "b64_p70": (64, [70] + [1] * 8),
    "b64_p300": (64, [256, 44] + [1] * 8),
    "b64_c3x100": (64, [100, 100, 100] + [1] * 8),
    "b64_mid": (64, [37, 61, 103] + [1] * 5),
    "b2048_p2052": (2048, [256] * 7 + [255, 5] + [1] * 6),
    "b2048_p2400": (2048, [256] * 9 + [97] + [1] * 3),
}
SEL_CASES = {"sel2051": 2051, "sel2052": 2052, "sel3000": 3000, "sel8192": 8192}


class Writer:
    """index.txt / params.txt are merged with an existing run (so `--only` refreshes single cases)."""

    def __init__(self, out):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.index = self._load("index.txt")
        self.params = self._load("params.txt")

    def _load(self, fname):
        path = os.path.join(self.out, fname)
        entries = {}
        if os.path.exists(path):
            with open(path) as f:
                for line in f.read().splitlines():
                    if line.strip():
                        entries[line.split(" ", 1)[0]] = line
        return entries

    def tensor(self, key, t):
        t = t.detach().contiguous()
        if t.dtype == torch.bfloat16:
            kind, arr = "bf16", t.view(torch.int16).numpy()
        elif t.dtype == torch.float32:
            kind, arr = "f32", t.numpy()
        elif t.dtype in (torch.int32, torch.int64, torch.uint8):
            kind, arr = "i32", t.to(torch.int32).numpy()
        else:
            raise ValueError(f"{key}: unsupported dtype {t.dtype}")
        arr.tofile(os.path.join(self.out, key + ".bin"))
        self.index[key] = f"{key} {kind} {len(t.shape)} " + " ".join(str(d) for d in t.shape)

    def param(self, key, value):
        self.params[key] = f"{key} {value}"

    def finish(self):
        with open(os.path.join(self.out, "index.txt"), "w") as f:
            f.write("\n".join(self.index.values()) + "\n")
        with open(os.path.join(self.out, "params.txt"), "w") as f:
            f.write("\n".join(self.params.values()) + "\n")


def make_config(budget: int) -> Qwen4ExpTextConfig:
    cfg = Qwen4ExpTextConfig(
        vocab_size=64,
        hidden_size=HIDDEN,
        num_hidden_layers=4,
        num_attention_heads=HQ,
        num_key_value_heads=HKV,
        head_dim=HD,
        max_position_embeddings=262144,
        rms_norm_eps=1e-6,
        indexer_n_heads=4,
        indexer_kv_heads=1,
        indexer_head_dim=128,
        indexer_budget=budget,
        indexer_compress_ratio=RATIO,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10_000_000,
            "partial_rotary_factor": 0.25,
            "mrope_section": [11, 11, 10],
            "mrope_interleaved": True,
        },
    )
    cfg._attn_implementation = "eager"
    return cfg


def rnd(gen, *shape, std=1.0, mean=0.0):
    return (torch.randn(*shape, generator=gen) * std + mean).to(BF)


def set_weight(param, value):
    with torch.no_grad():
        param.copy_(value)


def position_embeddings(rotary, hidden, n):
    pos = torch.arange(n).view(1, 1, -1).expand(3, hidden.shape[0], -1)
    return rotary(hidden, pos)


def causal_mask(n_q, n_kv):
    q_pos = torch.arange(n_q)[:, None] + (n_kv - n_q)
    return (torch.arange(n_kv)[None, :] <= q_pos).view(1, 1, n_q, n_kv)


def fp8_quant(x_bf16, descale):
    """fp8 e4m3 bytes (uint8) of bf16 x [1, heads, T, d] exactly as r4dx writes them: fp8(float(x) * (1 / descale))."""
    inv = 1.0 / descale.view(1, -1, 1, 1)
    return (x_bf16.float() * inv).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)


def fp8_dequant(q_u8, descale):
    return q_u8.view(torch.float8_e4m3fn).float() * descale.view(1, -1, 1, 1)


class FakeCache:
    """The two cache calls HF's attention makes. K / V are stored as fp8 bytes and handed back dequantized (bf16)."""

    def __init__(self, kd, vd):
        self.kd, self.vd = kd, vd
        self.kq = self.vq = self.raw = None
        self.chunk_kq = self.chunk_vq = None

    def update(self, k, v, layer_idx):
        self.chunk_kq, self.chunk_vq = fp8_quant(k, self.kd), fp8_quant(v, self.vd)
        self.kq = self.chunk_kq if self.kq is None else torch.cat([self.kq, self.chunk_kq], dim=2)
        self.vq = self.chunk_vq if self.vq is None else torch.cat([self.vq, self.chunk_vq], dim=2)
        return fp8_dequant(self.kq, self.kd).to(BF), fp8_dequant(self.vq, self.vd).to(BF)

    def update_indexer(self, raw_keys, layer_idx):
        self.raw = raw_keys if self.raw is None else torch.cat([self.raw, raw_keys], dim=1)
        return self.raw


def selection_lists(selected, vis, budget):
    """selected, vis [Q, KV] bool -> (tokens [Q, budget + 3] i32 padded with -1, count [Q] i32, -1 = dense row)."""
    n_q = selected.shape[0]
    toks = torch.full((n_q, budget + RATIO - 1), -1, dtype=torch.int32)
    cnt = torch.full((n_q,), -1, dtype=torch.int32)
    for i in range(n_q):
        visible = int(vis[i].sum())
        if visible // RATIO > budget // RATIO:
            idx = selected[i].nonzero().flatten().to(torch.int32)
            assert idx.numel() == budget + visible % RATIO, (idx.numel(), budget, visible)
            toks[i, : idx.numel()] = idx
            cnt[i] = idx.numel()
        else:
            assert torch.equal(selected[i], vis[i])
    return toks, cnt


def block_keys_hf(indexer, raw_keys, cos, sin, n_blocks):
    """HF's pooled / normed / roped block keys of blocks [0, n_blocks). raw_keys [1, N, 128]; cos / sin [1, N, 64]."""
    starts = torch.arange(n_blocks) * RATIO
    return qsa._pooled_block_keys(
        raw_keys[0, : n_blocks * RATIO], starts, cos[0], sin[0], indexer.k_layernorm.weight, indexer.k_layernorm.eps, RATIO
    )


def gen_attention_case(w, name, budget, chunks):
    t0 = time.time()
    n = sum(chunks)
    gen = torch.Generator().manual_seed(1000 + sum(ord(c) for c in name))
    cfg = make_config(budget)
    attn = Qwen4ExpTextAttention(cfg, LAYER_IDX).to(BF).eval()
    rotary = Qwen4ExpTextRotaryEmbedding(cfg)
    set_weight(attn.q_proj.weight, rnd(gen, HQ * HD * 2, HIDDEN, std=0.02))
    set_weight(attn.k_proj.weight, rnd(gen, HKV * HD, HIDDEN, std=0.02))
    set_weight(attn.v_proj.weight, rnd(gen, HKV * HD, HIDDEN, std=0.02))
    set_weight(attn.o_proj.weight, rnd(gen, HIDDEN, HQ * HD, std=0.02))
    # (1 + w) of 2.5 on q and k: logits with std ~6, i.e. peaked softmax rows (a flat softmax hides errors)
    set_weight(attn.q_norm.weight, rnd(gen, HD, std=0.1, mean=1.5))
    set_weight(attn.k_norm.weight, rnd(gen, HD, std=0.1, mean=1.5))
    ix = attn.indexer
    set_weight(ix.index_qk_proj.weight, rnd(gen, 640, HIDDEN, std=0.02))
    set_weight(ix.q_layernorm.weight, rnd(gen, 128, std=0.1))
    set_weight(ix.k_layernorm.weight, rnd(gen, 128, std=0.1))

    hidden = rnd(gen, 1, n, HIDDEN)
    cos, sin = position_embeddings(rotary, hidden, n)
    with torch.no_grad():
        qg_all = attn.q_proj(hidden)
        k_all = attn.k_proj(hidden)
        v_all = attn.v_proj(hidden)
        idxqk_all = ix.index_qk_proj(hidden)
        # static descales from the whole sequence's post-norm / post-rope K and V (per kv head)
        k_states = attn.k_norm(k_all.view(1, n, HKV, HD)).transpose(1, 2)
        k_rot = apply_rotary_pos_emb(k_states, cos=cos, sin=sin, unsqueeze_dim=1)
        v_states = v_all.view(1, n, HKV, HD).transpose(1, 2)
        kd = (k_rot.float().abs().amax(dim=(0, 2, 3)) / 224.0).float()
        vd = (v_states.float().abs().amax(dim=(0, 2, 3)) / 224.0).float()

    cache = FakeCache(kd, vd)
    captured = {}

    def indexer_fwd(hidden_states, position_embeddings_, attention_mask, past_key_values):
        m = qsa.indexer_forward(ix, hidden_states, position_embeddings_, attention_mask, past_key_values, tie_rule="recent")
        captured["mask"] = m
        return m

    ix.forward = indexer_fwd
    attn.o_proj.register_forward_pre_hook(lambda mod, args: captured.__setitem__("gated", args[0].detach().clone()))

    w.param(f"{name}.budget", budget)
    w.param(f"{name}.nchunks", len(chunks))
    w.tensor(f"{name}.q_norm", attn.q_norm.weight)
    w.tensor(f"{name}.k_norm", attn.k_norm.weight)
    w.tensor(f"{name}.idx_q_norm", ix.q_layernorm.weight)
    w.tensor(f"{name}.idx_k_norm", ix.k_layernorm.weight)
    w.tensor(f"{name}.k_descale", kd)
    w.tensor(f"{name}.v_descale", vd)
    w.tensor(f"{name}.inv_freq", rotary.inv_freq.float())

    start = 0
    n_sparse_rows = 0
    for ci, t_len in enumerate(chunks):
        end = start + t_len
        h_c = hidden[:, start:end]
        mask_bool = causal_mask(t_len, end)
        mask_f = torch.where(mask_bool, torch.zeros((), dtype=BF), torch.full((), torch.finfo(BF).min, dtype=BF))
        pe = (cos[:, :end], sin[:, :end])
        with torch.no_grad():
            attn(h_c, pe, mask_f, cache)
        gated = captured["gated"][0]  # [T, 6144] bf16, HF's `attn_output * sigmoid(gate)`
        selected = (captured["mask"][0, 0] == 0) & mask_bool[0, 0]
        toks, cnt = selection_lists(selected, mask_bool[0, 0], budget)
        n_sparse_rows += int((cnt >= 0).sum())

        # fp32 reference of the attention itself: same q, same selected sets, fp32 dequantized K / V
        with torch.no_grad():
            qg = attn.q_proj(h_c)
            q_raw, gate = torch.chunk(qg.view(1, t_len, -1, HD * 2), 2, dim=-1)
            gate = gate.reshape(1, t_len, -1)
            q = attn.q_norm(q_raw.reshape(1, t_len, HQ, HD)).transpose(1, 2)
            q = apply_rotary_pos_emb(q, cos=cos[:, start:end], sin=sin[:, start:end], unsqueeze_dim=1)
            kf = fp8_dequant(cache.kq, kd).repeat_interleave(HQ // HKV, dim=1)  # [1, 24, end, 256]
            vf = fp8_dequant(cache.vq, vd).repeat_interleave(HQ // HKV, dim=1)
            s = torch.matmul(q.float(), kf.transpose(2, 3)) * (HD**-0.5)
            s = s.masked_fill(~selected.view(1, 1, t_len, end), float("-inf"))
            o = torch.matmul(torch.softmax(s, dim=-1), vf)  # [1, 24, T, 256] fp32
            o = o.transpose(1, 2).reshape(1, t_len, HQ * HD).to(BF)
            out32 = (o * torch.sigmoid(gate))[0]

        pre = f"{name}.c{ci}"
        w.param(f"{pre}.start", start)
        w.param(f"{pre}.len", t_len)
        w.tensor(f"{pre}.qg", qg_all[0, start:end])
        w.tensor(f"{pre}.k", k_all[0, start:end])
        w.tensor(f"{pre}.v", v_all[0, start:end])
        w.tensor(f"{pre}.idxqk", idxqk_all[0, start:end])
        w.tensor(f"{pre}.out", gated)
        w.tensor(f"{pre}.out32", out32)
        w.tensor(f"{pre}.sel", toks)
        w.tensor(f"{pre}.selcount", cnt)
        w.tensor(f"{pre}.kq", cache.chunk_kq[0].transpose(0, 1))  # [T, 2, 256]
        w.tensor(f"{pre}.vq", cache.chunk_vq[0].transpose(0, 1))
        start = end

    n_blocks = n // RATIO
    w.param(f"{name}.nblocks", n_blocks)
    if n_blocks:
        with torch.no_grad():
            w.tensor(f"{name}.blockkeys", block_keys_hf(ix, cache.raw, cos, sin, n_blocks))
    print(f"[{name}] {len(chunks)} chunks, {n} tokens, budget {budget}, sparse rows {n_sparse_rows} ({time.time() - t0:.1f}s)", flush=True)


def gen_selection_case(w, name, n):
    t0 = time.time()
    gen = torch.Generator().manual_seed(7000 + n)
    cfg = make_config(2048)
    ix = Qwen4ExpTextQSAIndexer(cfg, LAYER_IDX).to(BF).eval()
    rotary = Qwen4ExpTextRotaryEmbedding(cfg)
    set_weight(ix.index_qk_proj.weight, rnd(gen, 640, HIDDEN, std=0.02))
    set_weight(ix.q_layernorm.weight, rnd(gen, 128, std=0.1))
    set_weight(ix.k_layernorm.weight, rnd(gen, 128, std=0.1))
    hidden = rnd(gen, 1, n, HIDDEN)
    cos, sin = position_embeddings(rotary, hidden, n)
    with torch.no_grad():
        idxqk = ix.index_qk_proj(hidden)
        mask, ties = qsa.indexer_forward(ix, hidden, (cos, sin), causal_mask(n, n), None, tie_rule="recent", return_ties=True)
        vis = causal_mask(n, n)[0, 0]
        toks, cnt = selection_lists(mask[0, 0], vis, 2048)
        raw = idxqk[..., 512:]
        n_blocks = n // RATIO
        bk = block_keys_hf(ix, raw, cos, sin, n_blocks)
    w.param(f"{name}.n", n)
    w.param(f"{name}.budget", 2048)
    w.param(f"{name}.ties", int(ties.sum()))
    w.param(f"{name}.nblocks", n_blocks)
    w.tensor(f"{name}.q_norm", ix.q_layernorm.weight)
    w.tensor(f"{name}.k_norm", ix.k_layernorm.weight)
    w.tensor(f"{name}.idxqk", idxqk[0])
    with torch.no_grad():
        hf_q, _ = qsa.project(ix, hidden, (cos, sin))
    w.tensor(f"{name}.idxq", hf_q[0])  # [n, 4, 128] after q_layernorm + rope, as HF computes it
    w.tensor(f"{name}.sel", toks)
    w.tensor(f"{name}.selcount", cnt)
    w.tensor(f"{name}.blockkeys", bk)
    w.tensor(f"{name}.inv_freq", rotary.inv_freq.float())
    print(f"[{name}] {n} tokens, sparse rows {int((cnt >= 0).sum())}, tied rows {int(ties.sum())} ({time.time() - t0:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"E:\models\r4dx\q4exp-fixtures\QsaCore")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    w = Writer(args.out)
    names = list(ATTN_CASES) + list(SEL_CASES)
    if args.only:
        names = [n for n in names if n in args.only]
    for name in names:
        if name in ATTN_CASES:
            budget, chunks = ATTN_CASES[name]
            gen_attention_case(w, name, budget, chunks)
        else:
            gen_selection_case(w, name, SEL_CASES[name])
    w.finish()
    print("wrote", args.out)


if __name__ == "__main__":
    main()
