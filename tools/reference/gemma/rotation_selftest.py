#!/usr/bin/env python
"""Gemma 4 residual-rotation selftest (docs/gemma4-plan.md 4.4, task M1-5).

Proves, in float64 numpy and with no checkpoint and no GPU, that "option A" is numerically the plain
Gemma 4 decoder layer:

  * the residual stream is carried rotated, r' = r Q  (Q = signs, blockwise Hadamard, block mixing R;
    hidden 3840 = 15 x 256 by default, the real geometry);
  * input_layernorm / pre_feedforward_layernorm fold into q/k/v and gate/up as W' = W diag(w) Q
    (PLAIN weight, norm_offset 0), and the runtime norm is the weightless rms(x'), because
    rms(x Q) = rms(x);
  * post_attention_layernorm / post_feedforward_layernorm do NOT fold (Q^T does not commute with a
    channelwise weight). o_proj and down_proj take only W Hb on the K side, so their output is in the
    ORIGINAL basis; the runtime fuses  r' <- (r' + Q(post_norm(out) * w)) * scale  into one kernel
    (post_rmsnorm_rotate_add; `post_rmsnorm_rotate_add_ref` below is its exact contract);
  * layer_scalar commutes with Q; the stack exits with x' Q^T before final_norm.

The layer under test is the real Gemma 4 one at reduced head/MLP counts (hidden 3840 stays real):
sliding layer (head_dim 256, GQA) and full layer (head_dim 512, one KV head, k == v, v_norm
weightless), qk-norm with weight, attention scale 1.0, causal, GeGLU (gelu tanh), sandwich norms,
layer_scalar. RoPE is omitted: it acts inside a head and commutes with everything here.

Also a negative control (folding Q^T into o/down under a post-norm must NOT match: this is why
option A exists) and a quantization-noise gate (the rotated path under bf16 weight rounding stays as
close to the plain path as the plain path's own bf16 rounding is to the exact one).

Exit code 0 = all gates pass. Needs only numpy (system / Python312 python is fine; read-only).
    python tools/reference/gemma/rotation_selftest.py [--hidden 3840] [--seed 1] [--quick]
"""
import argparse
import math
import sys

import numpy as np

EPS = 1e-6
FAILURES = []


def gate(ok, label):
    print(("OK   " if ok else "FAIL ") + label)
    if not ok:
        FAILURES.append(label)
    return ok


def rel(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-300))


# ---------------------------------------------------------------------------------------------------
# Rotation (same algebra as src/convert/include/r4dx_convert/rotation.hpp; the random draws differ,
# which does not matter: the identities hold for any signs and any orthogonal R)
# ---------------------------------------------------------------------------------------------------

def choose_rotation_block(hidden):
    """Largest power of two dividing hidden, capped at 1024 (ChooseRotationBlock)."""
    return min(hidden & -hidden, 1024)


def hadamard_matrix(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


class Rotation:
    def __init__(self, rng, hidden):
        self.hidden = hidden
        self.block = choose_rotation_block(hidden)
        self.nblk = hidden // self.block
        self.signs = rng.choice([-1.0, 1.0], size=hidden)
        g = rng.standard_normal((self.nblk, self.nblk))
        self.mix, _ = np.linalg.qr(g)  # R[c][b], orthogonal
        self.h = hadamard_matrix(self.block) / math.sqrt(self.block)

    def apply(self, x):
        """x [..., hidden] -> x Q."""
        lead = x.shape[:-1]
        y = (x * self.signs).reshape(*lead, self.nblk, self.block) @ self.h  # per-block Hadamard
        z = np.einsum("...ci,cb->...bi", y, self.mix)  # z[b,i] = sum_c y[c,i] R[c][b]
        return z.reshape(*lead, self.hidden)

    def apply_t(self, x):
        """x -> x Q^T (exact inverse)."""
        lead = x.shape[:-1]
        y = np.einsum("...bi,cb->...ci", x.reshape(*lead, self.nblk, self.block), self.mix)
        y = y @ self.h
        return y.reshape(*lead, self.hidden) * self.signs

    def dense(self):
        return self.apply(np.eye(self.hidden))


class BlockHadamard:
    """h Hb := (h * s) then FWHT / sqrt(B) per contiguous block of B."""

    def __init__(self, rng, k, block):
        assert k % block == 0
        self.k, self.block = k, block
        self.signs = rng.choice([-1.0, 1.0], size=k)
        self.h = hadamard_matrix(block) / math.sqrt(block)

    def apply(self, x):
        lead = x.shape[:-1]
        y = (x * self.signs).reshape(*lead, self.k // self.block, self.block) @ self.h
        return y.reshape(*lead, self.k)


# ---------------------------------------------------------------------------------------------------
# Gemma 4 layer pieces (HF semantics)
# ---------------------------------------------------------------------------------------------------

def rms(x):
    return x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)


def rmsnorm(x, w=None):
    """Gemma 4 RMSNorm: x * rsqrt(mean(x^2) + eps) * w (PLAIN weight, not 1 + w); w None = weightless."""
    y = rms(x)
    return y if w is None else y * w


def gelu_tanh(x):
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)))


def attention(q, k, v, scale=1.0):
    """q [T, Hq, D], k/v [T, Hkv, D], causal, GQA; returns [T, Hq*D]. Attention scaling is 1.0."""
    t, hq, d = q.shape
    hkv = k.shape[1]
    out = np.zeros((t, hq, d))
    mask = np.triu(np.full((t, t), -np.inf), 1)
    for h in range(hq):
        kv = h // (hq // hkv)
        s = (q[:, h] @ k[:, kv].T) * scale + mask
        s -= s.max(axis=-1, keepdims=True)
        p = np.exp(s)
        p /= p.sum(axis=-1, keepdims=True)
        out[:, h] = p @ v[:, kv]
    return out.reshape(t, hq * d)


class LayerWeights:
    """One decoder layer. `full`: head_dim 512, one KV head, k == v (no v_proj), else head_dim 256."""

    def __init__(self, rng, hidden, full, heads, kv_heads_sliding, inter):
        self.full = full
        self.hd = 512 if full else 256
        self.heads = heads
        self.kvh = 1 if full else kv_heads_sliding
        self.hidden, self.inter = hidden, inter
        s = 0.02

        def w(n, k):
            return rng.standard_normal((n, k)) * s

        def nw(n):  # plain norm weights: positive, around 1 (checkpoint-like)
            return 1.0 + 0.3 * rng.standard_normal(n)

        self.input_ln, self.post_attn_ln = nw(hidden), nw(hidden)
        self.pre_ff_ln, self.post_ff_ln = nw(hidden), nw(hidden)
        self.q_norm, self.k_norm = nw(self.hd), nw(self.hd)
        self.wq = w(self.heads * self.hd, hidden)
        self.wk = w(self.kvh * self.hd, hidden)
        self.wv = None if full else w(self.kvh * self.hd, hidden)
        self.wo = w(hidden, self.heads * self.hd)
        self.gate, self.up = w(inter, hidden), w(inter, hidden)
        self.down = w(hidden, inter)
        self.layer_scalar = float(0.5 + rng.random())


def qkv(lw, xn):
    t = xn.shape[0]
    q = (xn @ lw.wq.T).reshape(t, lw.heads, lw.hd)
    k = (xn @ lw.wk.T).reshape(t, lw.kvh, lw.hd)
    v = k.copy() if lw.full else (xn @ lw.wv.T).reshape(t, lw.kvh, lw.hd)
    return rmsnorm(q, lw.q_norm), rmsnorm(k, lw.k_norm), rmsnorm(v)  # v_norm: weightless


def plain_layer(lw, x):
    """HF Gemma 4 decoder layer, original basis."""
    xn = rmsnorm(x, lw.input_ln)
    q, k, v = qkv(lw, xn)
    a = attention(q, k, v)
    x = x + rmsnorm(a @ lw.wo.T, lw.post_attn_ln)
    xn = rmsnorm(x, lw.pre_ff_ln)
    h = gelu_tanh(xn @ lw.gate.T) * (xn @ lw.up.T)
    x = x + rmsnorm(h @ lw.down.T, lw.post_ff_ln)
    return x * lw.layer_scalar


# ---------------------------------------------------------------------------------------------------
# Option A
# ---------------------------------------------------------------------------------------------------

def post_rmsnorm_rotate_add_ref(r, y, w, rot, scale=1.0):
    """Contract of the fused kernel post_rmsnorm_rotate_add:
         r_out = (r + Q( rmsnorm_plain(y, w) )) * scale
    r: rotated residual [T, hidden]; y: sublayer output in the ORIGINAL basis [T, hidden];
    w: plain post-norm weight [hidden]; scale: layer_scalar (1.0 for the attention sublayer)."""
    return (r + rot.apply(rmsnorm(y, w))) * scale


def fold_in(w_lin, norm_w, rot):
    """W' = W diag(w) Q  (plain weight: norm offset 0)."""
    return rot.apply(w_lin * norm_w)  # row r -> (r * w) Q


def option_a_layer(lw, xr, rot, hb_o, hb_down, fold_qt_into_out=False):
    """The layer on the rotated residual xr = x Q, exactly as the runtime runs it.
    fold_qt_into_out=True is the REJECTED alternative (Q^T W on the output rows), a negative control."""
    wq, wk = fold_in(lw.wq, lw.input_ln[None, :], rot), fold_in(lw.wk, lw.input_ln[None, :], rot)
    wv = None if lw.full else fold_in(lw.wv, lw.input_ln[None, :], rot)
    xn = rms(xr)  # weightless: rms(x Q) == rms(x); the weight is folded into wq/wk/wv
    t = xr.shape[0]
    q = rmsnorm((xn @ wq.T).reshape(t, lw.heads, lw.hd), lw.q_norm)
    kraw = (xn @ wk.T).reshape(t, lw.kvh, lw.hd)
    vraw = kraw if lw.full else (xn @ wv.T).reshape(t, lw.kvh, lw.hd)  # full layers: v = raw k_proj
    k, v = rmsnorm(kraw, lw.k_norm), rmsnorm(vraw)
    a = attention(q, k, v)

    # o_proj: online h -> h Hb (one head per block), weight W Hb. Output stays in the original basis.
    wo = hb_o.apply(lw.wo)  # row -> row Hb
    if fold_qt_into_out:
        wo = rot.apply(wo.T).T  # Q^T W : each column as a row -> c Q
        out = hb_o.apply(a) @ wo.T
        xr = xr + rmsnorm(out, lw.post_attn_ln)  # post-norm applied in the rotated basis: wrong
    else:
        out = hb_o.apply(a) @ wo.T
        xr = post_rmsnorm_rotate_add_ref(xr, out, lw.post_attn_ln, rot)

    wg, wu = fold_in(lw.gate, lw.pre_ff_ln[None, :], rot), fold_in(lw.up, lw.pre_ff_ln[None, :], rot)
    xn = rms(xr)
    h = gelu_tanh(xn @ wg.T) * (xn @ wu.T)
    wd = hb_down.apply(lw.down)
    if fold_qt_into_out:
        wd = rot.apply(wd.T).T
        out = hb_down.apply(h) @ wd.T
        return (xr + rmsnorm(out, lw.post_ff_ln)) * lw.layer_scalar
    out = hb_down.apply(h) @ wd.T
    return post_rmsnorm_rotate_add_ref(xr, out, lw.post_ff_ln, rot, lw.layer_scalar)


# ---------------------------------------------------------------------------------------------------

def to_bf16(a):
    """Round float64 -> bf16 -> float64 (round to nearest even on the top 16 bits of the fp32)."""
    f = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32).astype(np.uint64)
    f = (f + 0x7FFF + ((f >> 16) & 1)) & 0xFFFF0000
    return f.astype(np.uint32).view(np.float32).astype(np.float64)


def run(args):
    rng = np.random.default_rng(args.seed)
    hidden = args.hidden
    rot = Rotation(rng, hidden)
    print(f"hidden {hidden}: block {rot.block} x nblk {rot.nblk}")

    # --- Q itself
    if hidden <= 1024 or args.dense:
        qd = rot.dense()
        gate(rel(qd @ qd.T, np.eye(hidden)) < 1e-12, "Q Q^T = I (dense)")
    x = rng.standard_normal((5, hidden))
    gate(rel(rot.apply_t(rot.apply(x)), x) < 1e-12, "x Q Q^T = x")
    gate(abs(np.linalg.norm(rot.apply(x)) / np.linalg.norm(x) - 1) < 1e-12, "||x Q|| = ||x||")
    gate(rel(np.sqrt(np.mean(rot.apply(x) ** 2, -1)), np.sqrt(np.mean(x**2, -1))) < 1e-12,
         "rms(x Q) = rms(x): the weightless norm commutes with Q")

    heads, kvh, inter = (4, 2, 1536) if not args.quick else (2, 2, 512)
    t = 6 if not args.quick else 3
    for full in (False, True):
        name = "full (hd 512, k==v)" if full else "sliding (hd 256, GQA)"
        lw = LayerWeights(rng, hidden, full, heads, kvh, inter)
        hb_o = BlockHadamard(rng, lw.heads * lw.hd, 256)  # o_swa K and o_full K, block 256
        hb_down = BlockHadamard(rng, inter, 512)
        x = rng.standard_normal((t, hidden)) * 3.0
        x[:, 7] = 40.0  # outlier channel, as real residuals have
        want = plain_layer(lw, x)
        got = rot.apply_t(option_a_layer(lw, rot.apply(x), rot, hb_o, hb_down))
        e = rel(got, want)
        gate(e < 1e-9, f"{name}: option A layer == plain layer, float64 (max rel err {e:.2e})")

        # two stacked layers keep the identity (the residual never leaves the rotated basis in between)
        lw2 = LayerWeights(rng, hidden, full, heads, kvh, inter)
        want2 = plain_layer(lw2, want)
        got2 = rot.apply_t(option_a_layer(lw2, option_a_layer(lw, rot.apply(x), rot, hb_o, hb_down),
                                          rot, BlockHadamard(rng, lw2.heads * lw2.hd, 256), hb_down))
        # (lw2 uses fresh Hadamard signs, which is fine: each Hb pairs with its own folded weight)
        gate(rel(got2, want2) < 1e-9, f"{name}: two stacked layers, float64 (max rel err {rel(got2, want2):.2e})")

        bad = rot.apply_t(option_a_layer(lw, rot.apply(x), rot, hb_o, hb_down, fold_qt_into_out=True))
        eb = rel(bad, want)
        gate(eb > 1e-2, f"{name}: folding Q^T into o/down under a post-norm is NOT equivalent "
                        f"({eb:.2e} off) -- why option A exists")

        # fused kernel contract vs its unfused pieces
        r = rng.standard_normal((t, hidden))
        y = rng.standard_normal((t, hidden)) * 2
        fused = post_rmsnorm_rotate_add_ref(rot.apply(r), y, lw.post_ff_ln, rot, lw.layer_scalar)
        unfused = (rot.apply(r) + rot.apply(rmsnorm(y, lw.post_ff_ln))) * lw.layer_scalar
        gate(rel(fused, unfused) < 1e-14, f"{name}: post_rmsnorm_rotate_add contract")
        gate(rel(rot.apply_t(fused), (r + rmsnorm(y, lw.post_ff_ln)) * lw.layer_scalar) < 1e-12,
             f"{name}: layer_scalar commutes with Q; fused add equals the original-basis add")

        # bf16 weight-rounding noise: rotated path stays as close to exact as the plain path does
        def rounded(l):
            import copy
            c = copy.copy(l)
            for n in ("wq", "wk", "wv", "wo", "gate", "up", "down"):
                if getattr(l, n) is not None:
                    setattr(c, n, to_bf16(getattr(l, n)))
            return c

        lwb = rounded(lw)
        e_plain = rel(plain_layer(lwb, x), want)
        e_rot = rel(rot.apply_t(option_a_layer(lwb, rot.apply(x), rot, hb_o, hb_down)), want)
        gate(e_rot < 4 * e_plain + 1e-6,
             f"{name}: bf16 weight rounding, rotated err {e_rot:.2e} vs plain err {e_plain:.2e}")

    return 0 if not FAILURES else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hidden", type=int, default=3840)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="fewer heads / smaller MLP / fewer tokens")
    ap.add_argument("--dense", action="store_true", help="also build the dense Q (3840^2 doubles) and check Q Q^T = I")
    args = ap.parse_args()
    rc = run(args)
    print("rotation_selftest: all gates passed" if rc == 0 else f"rotation_selftest: {len(FAILURES)} failure(s)")
    return rc


if __name__ == "__main__":
    sys.exit(main())
