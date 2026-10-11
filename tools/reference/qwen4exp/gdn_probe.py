#!/usr/bin/env python
"""GDN probe on REAL weights (docs/qwen4exp.md "Real-weights prefix"): the exact inputs and outputs of one qwen4_exp GDN layer's
scan on the real T70 activations, computed on the CPU with Hugging Face's own functions
(`torch_chunk_gated_delta_rule`, `torch_recurrent_gated_delta_rule`, `l2norm`, the sigmoid-gated RMSNorm), so that
`r4d_gdn_chunk_scan` (libr4d's chunked prefill scan) and the recurrent decode kernels can be compared against them.

What it reads
  --container  a prefix container written by `r4dx-convert --layers N --layouts bf16` (default D:\\q4exp\\real-prefix4-bf16.r4dx):
               the layer's gdn.* tensors (bf16 weights are stored raw; A_log / dt_bias fp32)
  --golden     a real layer golden (default E:\\models\\r4dx\\q4exp\\ref\\layer_goldens\\T70.recent.safetensors): the block input
               `layer{L}.attn_mix` [T, 2560] (the gated residual's mix, the GDN's x) and the block output `layer{L}.attn_out`

What it prints, per layer
  1. the probe's own agreement with the golden attn_out (HF chunk rule and HF recurrent rule: both must be bf16 noise; this is what
     makes the probe a reference)
  2. the gate statistics that decide whether libr4d's chunk scan is numerically in range: per-token g = -exp(A_log) softplus(a +
     dt_bias), and the per-64-row-chunk SPAN = -(cumsum of g over the chunk)[-1] for every value head. r4d_gdn_chunk_scan splits
     e^{g_i - g_j} as e^{g_i - c} e^{c - g_j} around the chunk midpoint c and clamps each half at e^80 (its source says so): it is exact
     only while span <= ~160 (fp32 tops out at e^88), attenuated beyond ~176.
  3. with --emulate: the arithmetic of the shipped scan ("split" form: midpoint reference, clamp at 80, bf16 staging of W / sA / V') and
     of the span-safe scan r4d_gdn_chunk_scan_stable ("stable" form: per-element e^{G_i - G_j}, K rows weighted by e^{G_last - G_t}), in
     fp32 on the CPU, each through the norm + out_proj against the golden attn_out. On the real T70 activations the split form reproduces
     the C++ shipped-scan error to four digits (layer 0: 5.4511 / 0.33969 against 5.4415 / 0.33966), which is the diagnosis; the stable
     form agrees with the golden at bf16 noise (0.077 / 0.0033).
With --dump <npz> the intermediates (x, mixed_qkv after the conv, q, k, v, g, beta, the HF outputs) are saved for a C++ harness.

Python: D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen4exp\\gdn_probe.py --layers 0 1 2
"""
import argparse
import json
import struct
import sys

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers.models.qwen4_exp import modeling_qwen4_exp as M

HEADS_K, HEADS_V, DIM, CONV_K = 16, 48, 128, 4
KEY_DIM, VALUE_DIM = HEADS_K * DIM, HEADS_V * DIM
CONV_DIM = 2 * KEY_DIM + VALUE_DIM
EPS = 1e-6
CHUNK = 64


class Container:
    """Read-only view of an r4dx container (safetensors framing, raw tensor bytes)."""

    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.base = 8 + n

    def raw(self, name):
        m = self.h[name]
        b, e = m["data_offsets"]
        self.f.seek(self.base + b)
        return self.f.read(e - b)

    def bf16(self, name, shape):
        a = np.frombuffer(self.raw(name), dtype=np.uint16).reshape(shape)
        return torch.from_numpy(a.astype(np.int16)).view(torch.bfloat16)

    def f32(self, name, shape):
        return torch.from_numpy(np.frombuffer(self.raw(name), dtype=np.float32).reshape(shape).copy())


def load_layer(c, layer):
    p = f"text.layers.{layer}.gdn."
    return {
        "qkv": c.bf16(p + "in_proj_qkv.bf16.w", (CONV_DIM, 2560)),
        "z": c.bf16(p + "in_proj_z.bf16.w", (VALUE_DIM, 2560)),
        "a": c.bf16(p + "in_proj_a", (HEADS_V, 2560)),
        "b": c.bf16(p + "in_proj_b", (HEADS_V, 2560)),
        "conv": c.bf16(p + "conv1d_weight", (CONV_DIM, 1, CONV_K)),
        "A_log": c.f32(p + "A_log", (HEADS_V,)),
        "dt_bias": c.f32(p + "dt_bias", (HEADS_V,)),
        "norm": c.bf16(p + "norm_weight", (DIM,)),
        "out": c.bf16(p + "out_proj.bf16.w", (2560, VALUE_DIM)),
    }


def front(x, w):
    """Everything before the scan, in HF's own order and dtypes (modeling_qwen4_exp.py:526-583)."""
    T = x.shape[0]
    mixed = F.linear(x, w["qkv"]).transpose(0, 1)[None]  # [1, conv_dim, T]
    z = F.linear(x, w["z"]).reshape(1, T, -1, DIM)
    b = F.linear(x, w["b"])[None]
    a = F.linear(x, w["a"])[None]
    mixed = M.causal_conv1d_fn(mixed, w["conv"].squeeze(1), None, activation="silu")
    mixed = mixed.transpose(1, 2)
    q, k, v = torch.split(mixed, [KEY_DIM, KEY_DIM, VALUE_DIM], dim=-1)
    q = q.reshape(1, T, -1, DIM)
    k = k.reshape(1, T, -1, DIM)
    v = v.reshape(1, T, -1, DIM)
    beta = b.sigmoid()
    g = -w["A_log"].float().exp() * F.softplus(a.float() + w["dt_bias"])
    return mixed, q, k, v, z, beta, g


def back(core, z, w):
    """gated RMSNorm (sigmoid gate, HF's Qwen4ExpTextRMSNormGated) then out_proj."""
    T = core.shape[1]
    norm = M.Qwen4ExpTextRMSNormGated(DIM, eps=EPS, activation="sigmoid")
    norm.weight.data = w["norm"].float().clone()
    y = norm(core.reshape(-1, DIM), z.reshape(-1, DIM)).reshape(T, -1)
    return F.linear(y, w["out"])


def rep(t):
    return t.repeat_interleave(HEADS_V // HEADS_K, dim=2)


def bfr(t):
    """round to bf16 and back (conv_prep stores q / k as bf16)"""
    return t.to(torch.bfloat16).float()


def stats(got, ref):
    got, ref = got.float(), ref.float()
    rms = ref.pow(2).mean().sqrt().item()
    d = (got - ref).abs()
    return d.max().item() / rms, d.mean().item() / rms, rms


def chunk_spans(g):
    """g [T, H] per-token gate (<= 0) -> [chunks, H] span = -(cumsum within the 64-row chunk)[-1]."""
    T, H = g.shape
    out = []
    for c0 in range(0, T, CHUNK):
        out.append(-g[c0:c0 + CHUNK].sum(0))
    return torch.stack(out)


def emulate_scan(q, k, v, g, beta, scale, form, clamp=80.0):
    """The arithmetic of a libr4d chunk scan (third_party/libr4d/r4d_gdn_chunk_scan_k128_v128_c64_bf16.hip), fp32 on the CPU, one value
    head at a time, zero initial state. q, k, v, g, beta: [T, D] / [T] for ONE head (q, k l2-normed and bf16-rounded, as conv_prep stores
    them; q NOT yet scaled). G = cumsum of g within the 64-row chunk. A (kkt_solve) always uses the exact per-element e^{G_i - G_j}.
    The state and the products read S rounded to bf16 (the kernel keeps S in fp32 accumulators and stages a bf16 copy for the WMMAs).
      form "split"  (the shipped kernel): rs[i] = scale e^{min(G_i - c, clamp)}, gv[j] = e^{min(c - G_j, clamp)}, c = (G_first+G_last)/2;
                    sA = rs * tril(P), V' = gv * (A W), S <- e^{G_last - c} (e^c S + V'^T K)
      form "stable" (the span-safe variant): sA[i][j] = scale e^{G_i - G_j} P[i][j] per element, V' = A W unweighted, and the state
                    update weights the K rows instead: S <- e^{G_last} S + (e^{G_last - G_t} K_t)^T V'  (every exponent <= 0)
    bf16 staging points (W, sA, V', the weighted K) are rounded."""
    T = q.shape[0]
    D = q.shape[1]
    S = torch.zeros(D, D)  # [k, v]
    o = torch.zeros(T, D)
    bf = lambda t: t.to(torch.bfloat16).float()
    for c0 in range(0, T, CHUNK):
        sl = slice(c0, min(c0 + CHUNK, T))
        n = sl.stop - sl.start
        G = g[sl].cumsum(0)
        kk, qq, vv, bb = k[sl], q[sl], v[sl], beta[sl]
        tri = torch.tril(torch.ones(n, n, dtype=torch.bool))
        # A = (I + strict_lower(diag(beta) K K^T e^{G_i - G_j}))^-1
        gram = kk @ kk.T
        dec = torch.exp((G[:, None] - G[None, :]).clamp(max=0))
        N = torch.tril(bb[:, None] * gram * dec, diagonal=-1)
        A = bf(torch.linalg.inv(torch.eye(n) + bf(N)))
        ge = torch.exp(G)
        Sb = bf(S)
        U = kk @ Sb  # [n, v]
        W = bf(bb[:, None] * (vv - ge[:, None] * U))
        P = qq @ kk.T
        if form == "split":
            c = 0.5 * (G[0] + G[-1])
            rs = scale * torch.exp(torch.clamp(G - c, max=clamp))
            gv = torch.exp(torch.clamp(c - G, max=clamp))
            sA = bf(torch.where(tri, P * rs[:, None], torch.zeros(())))
            Vp = bf(gv[:, None] * (A @ W))
            o[sl] = scale * ge[:, None] * (qq @ Sb) + sA @ Vp
            lam = torch.exp(G[-1] - c)
            S = lam * (torch.exp(c) * S + kk.T @ Vp)
        else:
            sA = bf(torch.where(tri, scale * P * dec, torch.zeros(())))
            Vp = bf(A @ W)
            o[sl] = scale * ge[:, None] * (qq @ Sb) + sA @ Vp
            kw = bf(torch.exp(G[-1] - G)[:, None] * kk)
            S = torch.exp(G[-1]) * S + kw.T @ Vp
    return o


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--container", default=r"D:\q4exp\real-prefix4-bf16.r4dx")
    ap.add_argument("--golden", default=r"E:\models\r4dx\q4exp\ref\layer_goldens\T70.recent.safetensors")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--emulate", action="store_true", help="emulate the shipped (split) and the span-safe (stable) chunk scan")
    ap.add_argument("--dump", help="npz path for the C++ harness (the first layer only)")
    a = ap.parse_args()
    torch.set_grad_enabled(False)
    c = Container(a.container)
    gold = safe_open(a.golden, "pt")
    for layer in a.layers:
        w = load_layer(c, layer)
        x = gold.get_tensor(f"layer{layer}.attn_mix")  # [T, 2560] bf16
        ref = gold.get_tensor(f"layer{layer}.attn_out")
        T = x.shape[0]
        mixed, q, k, v, z, beta, g = front(x, w)
        qr, kr = rep(q), rep(k)
        core_chunk, _ = M.torch_chunk_gated_delta_rule(qr, kr, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        core_rec, _ = M.torch_recurrent_gated_delta_rule(qr, kr, v, g=g, beta=beta, use_qk_l2norm_in_kernel=True)
        o_chunk = back(core_chunk, z, w)
        o_rec = back(core_rec, z, w)
        print(f"\n=== layer {layer} (T={T}) ===")
        for name, o in (("HF chunk rule", o_chunk), ("HF recurrent rule", o_rec)):
            mx, mean, rms = stats(o, ref)
            print(f"  {name:18s} vs golden attn_out: max/rms {mx:.4f}  mean/rms {mean:.5f}  (rms ref {rms:.4g})")
        mx, mean, _ = stats(o_chunk, o_rec)
        print(f"  HF chunk vs HF recurrent      : max/rms {mx:.4f}  mean/rms {mean:.5f}")

        gt = g[0]  # [T, H]
        spans = chunk_spans(gt)  # [chunks, H]
        print(f"  per-token g: min {gt.min():.1f}  p1 {gt.flatten().kthvalue(max(1, gt.numel() // 100)).values:.2f}  median {gt.median():.3f}"
              f"  max {gt.max():.4f};  tokens*heads with g < -20: {(gt < -20).float().mean() * 100:.1f}%  < -80: {(gt < -80).float().mean() * 100:.1f}%")
        print(f"  A = exp(A_log): min {w['A_log'].exp().min():.3f} max {w['A_log'].exp().max():.1f};  dt_bias: min {w['dt_bias'].min():.2f} max {w['dt_bias'].max():.2f}")
        for ci in range(spans.shape[0]):
            sp = spans[ci]
            rows = min(CHUNK, T - ci * CHUNK)
            print(f"  chunk {ci} ({rows} rows): span max {sp.max():8.1f}  heads with span > 88: {(sp > 88).sum().item():2d}/48  > 160: {(sp > 160).sum().item():2d}/48"
                  f"  > 176: {(sp > 176).sum().item():2d}/48")

        if a.emulate:
            ql = bfr(M.l2norm(rep(q).float(), dim=-1, eps=1e-6))[0]  # [T, H, D]
            kl = bfr(M.l2norm(rep(k).float(), dim=-1, eps=1e-6))[0]
            vf = v.float()[0]
            bf_ = beta.float()[0]
            scale = DIM ** -0.5
            exact = core_chunk.float()[0]
            for form in ("split", "stable"):
                emu = torch.zeros(T, HEADS_V, DIM)
                for h in range(HEADS_V):
                    emu[:, h] = emulate_scan(ql[:, h], kl[:, h], vf[:, h], gt[:, h], bf_[:, h], scale, form)
                mx, mean, rms = stats(emu, exact)
                o_emu = back(emu.to(torch.bfloat16)[None], z, w)
                mx2, mean2, _ = stats(o_emu, ref)
                print(f"  {form:6s}-form scan emulation: core vs HF chunk rule max/rms {mx:.4f} mean/rms {mean:.5f};  "
                      f"through norm + out_proj vs golden attn_out: max/rms {mx2:.4f} mean/rms {mean2:.5f}")

        if a.dump and layer == a.layers[0]:
            np.savez(a.dump, x=x.float().numpy(), mixed=mixed[0].float().numpy(), q=q.float().numpy(), k=k.float().numpy(),
                     v=v.float().numpy(), z=z.float().numpy(), beta=beta.float().numpy(), g=g.numpy(),
                     core_chunk=core_chunk.float().numpy(), core_rec=core_rec.float().numpy(), ref=ref.float().numpy())
            print(f"  dumped {a.dump}")


if __name__ == "__main__":
    sys.exit(main())
