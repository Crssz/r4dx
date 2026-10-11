"""tests/reference/test_trellis_kb3.py

CPU tests of trellis KB = 3 (3 bits per weight, 24 uint32 words per 16 x 16 tile; docs/qwen4exp-plan.md
4.3 / 4.11 item 6; docs/trellis-kernel.md "KB = 3") against tools/reference/trellis_quant.py, the oracle:

    (a) the Viterbi / packing path at K = 3: Gaussian tiles (closed rings, measured 3.000 bits per
        weight, pack/unpack round trip, MSE between the Gaussian D(R) bound and the scalar Lloyd-Max
        quantizer, strictly worse than K = 4 and K = 5);
    (b) the packed words: MSB-first layout against a slow bit-string packer, 8 K = 24 words per tile,
        state(p) = the 16 ring bits ending at bit 3 (p + 1), tail-biting;
    (c) quantize_group on a synthetic linear at K = 3, 4, 5: bits measured (K k n + scales), the
        stored-bits proxy error ordered K = 3 > 4 > 5, the relative-error trend, finite values;
    (d) the GPU kernel's KB = 3 lane map (libr4d r4d_trellis_dq.h r4d_trellis_k3_decode): per lane (32)
        of a tile pair, the four ring words at word (3c + 2h - 2 + i) mod 24, the 16h pre-alignment,
        the 16 states at big-endian offsets 19 + 3q of the 96 aligned bits, the fragment element
        order, emulated with 32-bit funnel shifts, equal to unpack_states + tensor_core_perm for every
        element of the tile (0 mismatches over 64 random tiles).

Plain script, no pytest (like test_trellis_quant.py):

    <venv>\\Scripts\\python.exe tests\\reference\\test_trellis_kb3.py

Exits 0 and prints "OK (<n> checks)" on success; 1 and every failed check otherwise. CPU only.
"""

from __future__ import annotations

import math
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "tools" / "reference"))

try:
    import numpy as np
    import torch
except ImportError as e:
    print(f"SKIP: {e} -- run this with the reference venv's python.exe (see docstring)")
    raise SystemExit(0)

import trellis_quant as tq  # noqa: E402

CHECKS = 0
FAILS: list[str] = []


def check(cond, what: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(what)
        print(f"FAIL: {what}")


# --------------------------------------------------------------------------------------------
# (a) Gaussian tiles at K = 3, 4, 5
# --------------------------------------------------------------------------------------------


def gaussian_mse(enc, x: torch.Tensor, K: int) -> dict:
    """Best-scale MSE of N(0,1) tiles x [T, 256] (the scale sweep of trellis_quant.cmd_gaussian)."""
    std = float(enc.cb.std())
    scales = [std * (0.8 + 0.05 * i) for i in range(13)]
    mses, sts = [], []
    for s in scales:
        q, st = enc.quantize(x * s, K)
        mses.append(float(((q / s - x) ** 2).mean()))
        sts.append(st)
    j = int(np.argmin(mses))
    return {"mse": mses[j], "states": sts[j], "scale": scales[j] / std}


def test_gaussian():
    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    rng = np.random.default_rng(31)
    x = torch.from_numpy(rng.standard_normal((96, 256)).astype(np.float32))
    res = {}
    for K in (3, 4, 5):
        r = gaussian_mse(enc, x, K)
        res[K] = r
        st = r["states"]
        D = tq.widths(K)
        check(D == [K] * 256, f"K={K} widths are 256 x {K}")
        check(tq.trellis_words(K) == 8 * K, f"K={K}: {8 * K} words per tile")
        check(bool(tq.ring_consistent(st.long(), D).all()), f"K={K} rings closed")
        w = tq.pack_states(st, K)
        check(w.shape == (96, 8 * K), f"K={K} packed shape {tuple(w.shape)}")
        check(torch.equal(tq.unpack_states(w, K), st.long()), f"K={K} pack/unpack round trip")
        bpw = w.numel() * 32 / (96 * 256)
        check(bpw == float(K), f"K={K} measured bits per weight {bpw}")
        lm = tq.lloyd_max_mse(K)
        floor = 2.0 ** (-2 * K)
        check(floor < r["mse"] < lm, f"K={K} gaussian MSE {r['mse']:.5f} in (D(R) {floor:.5f}, "
                                     f"Lloyd-Max {lm:.5f})")
    check(res[3]["mse"] > res[4]["mse"] > res[5]["mse"],
          f"gaussian MSE strictly decreases with K: {[round(res[k]['mse'], 5) for k in (3, 4, 5)]}")
    # the rate of one extra bit is ~6 dB for a good quantizer: 3 -> 4 and 4 -> 5 each cut the MSE 3-4.5x
    for a, b in ((3, 4), (4, 5)):
        ratio = res[a]["mse"] / res[b]["mse"]
        check(2.8 < ratio < 4.6, f"MSE ratio K={a}/K={b} = {ratio:.2f} (~4 for one bit)")
    print("  gaussian MSE: " + ", ".join(f"K={k}: {res[k]['mse']:.5f} (scale {res[k]['scale']:.2f} std)"
                                          for k in (3, 4, 5)))


# --------------------------------------------------------------------------------------------
# (b) packed word layout at K = 3
# --------------------------------------------------------------------------------------------


def slow_pack(states: list[int], K: int) -> list[int]:
    bits = []
    for s in states:
        bits += [(s >> (K - 1 - i)) & 1 for i in range(K)]
    words = []
    for w in range(len(bits) // 32):
        v = 0
        for b in bits[32 * w:32 * w + 32]:
            v = (v << 1) | b
        words.append(v)
    return words


def ring_bit(words: list[int], i: int) -> int:
    """Stream bit i of a ring (word w holds stream bits 32 w .. 32 w + 31, the earliest at bit 31)."""
    i %= 32 * len(words)
    return (words[i // 32] >> (31 - i % 32)) & 1


def test_layout():
    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    rng = np.random.default_rng(7)
    x = torch.from_numpy(rng.standard_normal((6, 256)).astype(np.float32))
    st, _ = enc.encode(x, 3)
    w = tq.pack_states(st, 3)
    for t in range(6):
        got = [int(v) & 0xFFFFFFFF for v in w[t].tolist()]
        check(got == slow_pack(st[t].tolist(), 3), f"tile {t}: MSB-first word layout == slow packer")
        # position p owns stream bits [3p, 3p + 3) = the low 3 bits of its state, MSB first
        own = all(((ring_bit(got, 3 * p) << 2) | (ring_bit(got, 3 * p + 1) << 1) | ring_bit(got, 3 * p + 2))
                  == (int(st[t][p]) & 7) for p in range(256))
        check(own, f"tile {t}: position p owns stream bits [3p, 3p+3) = state & 7")
        # state(p) = the 16 ring bits ending at 3 (p + 1), wrapping at 768 (tail-biting)
        ring_ok = True
        for p in range(256):
            v = 0
            for j in range(16):
                v = (v << 1) | ring_bit(got, 3 * (p + 1) - 16 + j)
            ring_ok &= v == int(st[t][p])
        check(ring_ok, f"tile {t}: state(p) = the 16 ring bits ending at 3(p+1), tail-biting")
    # any word pattern is a closed ring
    rw = torch.from_numpy(rng.integers(-2**31, 2**31, size=(8, 24), dtype=np.int64).astype(np.int32))
    us = tq.unpack_states(rw, 3)
    check(bool(tq.ring_consistent(us, tq.widths(3)).all()), "random 24-word tiles decode to closed rings")
    check(torch.equal(tq.pack_states(us, 3), rw), "pack(unpack(words)) == words at K = 3")
    # a tile is 768 bits = 24 words: bits per weight is exactly 3 (N K 3 / 8 bytes per linear)
    check(tq.trellis_words(3) * 32 == 256 * 3, "24 words = 768 bits per 256-weight tile")


# --------------------------------------------------------------------------------------------
# (c) quantize_group at K = 3, 4, 5
# --------------------------------------------------------------------------------------------


def random_spd(k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * k, k, generator=g) @ torch.diag(torch.linspace(0.2, 3.0, k))
    return (X.T @ X) / (4 * k)


def test_quantize_group():
    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    k, n = 256, 256
    H = random_spd(k, 41)
    g = torch.Generator().manual_seed(42)
    Whf = (torch.randn(n, k, generator=g) * 0.02).to(torch.bfloat16)
    W = Whf.float().T
    rec = {}
    for K in (3, 4, 5):
        res = tq.quantize_group({"w": Whf}, H, {"w": K}, enc, 1, {"w": 2}, verbose=False)["w"]
        r = res["record"]
        rec[K] = r
        check(res["words"].shape == (k // 16, n // 16, 8 * K), f"K={K} stored words shape {tuple(res['words'].shape)}")
        check(r["bits"]["trellis"] == k * n * K and r["bits"]["scales"] == 16 * (k + n),
              f"K={K} bits measured: {r['bits']['trellis']} trellis + {r['bits']['scales']} scales")
        check(all(math.isfinite(r[f]) for f in ("proxy", "proxy_damped_h", "rel_weight_err", "cos")),
              f"K={K} finite errors")
        W_st = tq.reconstruct(res["words"], res["suh"], res["svh"], K, enc.cb)
        check(abs(tq.proxy_error(W, W_st, H) - r["proxy"]) < 1e-6, f"K={K} record proxy == proxy of the stored bits")
    pr = {K: rec[K]["proxy"] for K in rec}
    rel = {K: rec[K]["rel_weight_err"] for K in rec}
    check(pr[3] > pr[4] > pr[5], f"proxy error decreases with K: {pr}")
    check(rel[3] > rel[4] > rel[5], f"relative weight error decreases with K: {rel}")
    check(pr[3] < 0.06 and rel[3] < 0.3, f"K=3 stays usable: proxy {pr[3]:.5f}, rel {rel[3]:.4f}")
    check(1.5 < pr[3] / pr[4] < 6.0, f"K=3 / K=4 proxy ratio {pr[3] / pr[4]:.2f} in the expected band (2-4)")
    print("  synthetic 256x256 (proxy, rel weight err, cos): " +
          "; ".join(f"K={K}: {rec[K]['proxy']:.5f}, {rec[K]['rel_weight_err']:.4f}, {rec[K]['cos']:.5f}"
                    for K in (3, 4, 5)))


# --------------------------------------------------------------------------------------------
# (d) the GPU lane map at KB = 3 (r4d_trellis_k3_decode), emulated with 32-bit funnel shifts
# --------------------------------------------------------------------------------------------

M32 = 0xFFFFFFFF


def alignbit(a: int, b: int, s: int) -> int:
    """v_alignbit_b32: the low 32 bits of (a:b) >> (s & 31)."""
    return ((((a & M32) << 32) | (b & M32)) >> (s & 31)) & M32


def lane_states_k3(ring: list[int], lane: int) -> list[int]:
    """The 16 states one lane of a wave32 B fragment reads from its tile's 24-word ring: the kernel's
    extraction, step for step. Words (3c + 2h - 2 + i) mod 24, i = 0..3; V_i = alignbit(W_i, W_{i+1},
    16h); state q is the 16 big-endian bits at offset 19 + 3q of V0:V1:V2."""
    c, h = lane & 7, lane >> 4
    W = [ring[(3 * c + 2 * h - 2 + i) % 24] for i in range(4)]
    V = [alignbit(W[i], W[i + 1], 16 * h) for i in range(3)]
    out = []
    for q in range(16):
        o = 19 + 3 * q
        i, r = o >> 5, o & 31
        if q == 15:
            s = V[2] >> 16                            # offset 64: V2's high half (op_sel, no extract)
        elif r <= 16:
            s = V[i] >> (16 - r)                      # inside V_i
        else:
            s = alignbit(V[i], V[i + 1], 48 - r)      # straddles V_i / V_i+1
        out.append(s & 0xFFFF)
    return out


def test_lane_map():
    rng = np.random.default_rng(99)
    cb = tq.codebook_np("mul1")
    inv = tq.tensor_core_perm_inv()
    F0Q = [0, 1, 8, 9, 2, 3, 10, 11]
    F1Q = [4, 5, 12, 13, 6, 7, 14, 15]
    bad = 0
    total = 0
    words = torch.from_numpy(rng.integers(-2**31, 2**31, size=(64, 2, 24), dtype=np.int64).astype(np.int32))
    for blk in range(64):
        tiles = []
        for t in range(2):
            states = tq.unpack_states(words[blk, t][None], 3)[0]
            vals = cb[states.numpy()][inv]                 # row-major [k * 16 + n] of the tile
            tiles.append(vals.reshape(16, 16))
        rings = [[int(v) & M32 for v in words[blk, t].tolist()] for t in range(2)]
        for lane in range(32):
            t, c, h = (lane >> 3) & 1, lane & 7, lane >> 4
            st = lane_states_k3(rings[t], lane)
            for fi, qs in enumerate((F0Q, F1Q)):
                for e in range(8):
                    k = 8 * (e >> 2) + 4 * h + (e & 3)
                    n = c + 8 * fi
                    total += 1
                    if cb[st[qs[e]]] != tiles[t][k, n]:
                        bad += 1
    check(bad == 0, f"KB=3 lane map: {bad} mismatches of {total} fragment elements")
    # the word loads of the 32 lanes cover every word of both tiles, and each lane's four words are
    # distinct (pair-grid block = tile t0's 24 words, then t1's)
    seen = set()
    for lane in range(32):
        t, c, h = (lane >> 3) & 1, lane & 7, lane >> 4
        ws = [(3 * c + 2 * h - 2 + i) % 24 for i in range(4)]
        check(len(set(ws)) == 4, f"lane {lane}: four distinct words")
        seen |= {t * 24 + w for w in ws}
    check(seen == set(range(48)), "the 32 lanes' loads cover all 48 words of a (pair, k-tile) block")
    print(f"  lane map: 64 random tile pairs x 32 lanes x 16 elements: {bad} mismatches of {total}")


def main() -> int:
    tests = [test_layout, test_lane_map, test_gaussian, test_quantize_group]
    for t in tests:
        print(f"{t.__name__} ...", flush=True)
        try:
            t()
        except Exception:  # noqa: BLE001
            FAILS.append(f"{t.__name__} raised")
            traceback.print_exc()
    if FAILS:
        print(f"FAILED {len(FAILS)} of {CHECKS} checks:")
        for f in FAILS:
            print(f"  {f}")
        return 1
    print(f"OK ({CHECKS} checks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
