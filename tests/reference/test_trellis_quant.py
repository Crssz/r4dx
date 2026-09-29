"""tests/reference/test_trellis_quant.py

Tests of tools/reference/trellis_quant.py (the EXL3 trellis oracle, spec docs/trellis.md), its
native encoders (tools/reference/trellis_viterbi.hip) and the --weights-override plumbing of
full_logits_golden.py. CPU always; the HIP encoder only when $env:HIP_VISIBLE_DEVICES is '1' (this
repo's GPU rule: device 1 only, and only when the caller asked for the GPU), SKIPped otherwise.

    (a) codebooks: mul1 / 3inst / mcg against the spec's reference values (docs/trellis.md 4.3) and
        an independent integer re-derivation of mul1; the tensor-core permutation; rate widths.
    (b) Viterbi optimality by brute force on tiny trellises (6-bit state, 6-8 positions, integer and
        half-integer widths): the pinned pass is the exact optimum for its boundary edge; the two-pass
        tail-biting result is a closed ring whose cost is exact and never below the true optimum.
    (c) the native CPU encoder (and HIP, when enabled) against the torch reference: identical states
        and costs, every rate the oracle uses, random data and a tie-heavy case.
    (d) the stored bitstream: pack/unpack round trips, the MSB-first word layout against a slow
        bit-string implementation, any word pattern decodes to a closed ring.
    (e) incoherence round trips (P_128, regularize -> unrotate), the global-scale sample order
        (EXL3's), block LDL factorization, LDLQ with H = I equals plain per-tile encoding, a full
        quantize_group on a synthetic linear with a dead input channel (finite, bits measured, refit
        never hurts, feedback helps), the int4 baseline, the allocator with EXL3's qgroups, .hess
        reading, Lloyd-Max values, the Gaussian MSE at K = 4.
    (f) the override directory: manifest + load_override_tensor, full_logits_golden's out-dir
        refusal between bf16 / gguf / override dumps, kl_report's identity; quantize-model's resume
        key (code hashes, per-tensor K); `mix --bpw 4.5` and qgroup-uniform allocations over the real
        checkpoint's 400 linears (SKIPped without the checkpoint).

Plain script, no pytest (like test_gguf_dequant.py):

    <venv>\\Scripts\\python.exe tests\\reference\\test_trellis_quant.py

Exits 0 and prints "OK (<n> checks)" on success; 1 and every failed check otherwise.
"""

from __future__ import annotations

import itertools
import json
import math
import os
import struct
import sys
import tempfile
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


def close(a, b, tol, what):
    check(abs(float(a) - float(b)) <= tol, f"{what}: {a} vs {b} (tol {tol})")


GPU = os.environ.get("HIP_VISIBLE_DEVICES") == "1" and torch.cuda.is_available()

# --------------------------------------------------------------------------------------------
# (a) codebooks, permutation, widths
# --------------------------------------------------------------------------------------------


def test_codebooks():
    cb = tq.codebook_np("mul1").astype(np.float64)
    close(cb.mean(), -0.00193, 5e-6, "mul1 mean")
    close(cb.std(), 1.00031, 5e-6, "mul1 std")
    close(cb.min(), -3.45312, 1e-5, "mul1 min")
    close(cb.max(), 3.34766, 1e-5, "mul1 max")
    check(len(np.unique(cb)) == 913, f"mul1 distinct values {len(np.unique(cb))} != 913")
    check(cb[0] == -3.453125, f"mul1 state 0 decodes to {cb[0]}, spec: the minimum -3.453125")
    check(tq.MUL1_K_INV == 0.00676727294921875 and tq.MUL1_K_BIAS == -10.3828125, "mul1 fp16 constants")
    # an independent derivation with Python integers, one state at a time
    rng = np.random.default_rng(3)
    for s in [0, 1, 2, 255, 256, 4095, 65535] + rng.integers(0, 65536, 200).tolist():
        x = (int(s) * 0x83DCD12D) % (1 << 32)
        ssum = sum((x >> (8 * i)) & 0xFF for i in range(4))
        v = float(np.float16((1024 + ssum) * 0.00676727294921875 - 10.3828125))
        if v != cb[s]:
            check(False, f"mul1 state {s}: {cb[s]} vs independent {v}")
            break
    else:
        check(True, "mul1 independent derivation")
    c3 = tq.codebook_np("3inst").astype(np.float64)
    close(c3.std(), 1.24371091, 5e-8, "3inst std (== EXL3 codebook_scale to 7 digits)")
    close(c3.std(), tq.CODEBOOK_SCALE, 1e-7, "codebook_scale")
    check(len(np.unique(c3)) == 10598, f"3inst distinct {len(np.unique(c3))} != 10598")
    cm = tq.codebook_np("mcg").astype(np.float64)
    close(cm.std(), 1.24411, 5e-6, "mcg std")
    check(len(np.unique(cm)) == 10746, f"mcg distinct {len(np.unique(cm))} != 10746")


def test_perm_widths():
    p = tq.tensor_core_perm()
    check(sorted(p.tolist()) == list(range(256)), "tensor_core_perm is a bijection of 0..255")
    check(p[:8].tolist() == [0, 16, 128, 144, 8, 24, 136, 152], f"perm[0:8] = {p[:8].tolist()}")
    # lane t = 5: r0 = 2, c0 = 1
    check(p[40:48].tolist() == [33, 49, 161, 177, 41, 57, 169, 185], f"perm[40:48] = {p[40:48].tolist()}")
    check(np.array_equal(p[tq.tensor_core_perm_inv()], np.arange(256)), "perm_inv inverts perm")
    check(tq.frac_k(4) == (4, 0) and tq.frac_k(3.5) == (3, 0xAAAA), "frac_k")
    d = tq.widths(3.5)
    check(d[:4] == [3, 4, 3, 4] and sum(d) == 896, "3.5 widths alternate 3/4 (odd positions 4), 896 bits")
    check(tq.trellis_words(4) == 32 and tq.trellis_words(3.5) == 28 and tq.trellis_words(4.5) == 36,
          "uint32 words per tile = 8K")
    try:
        tq.frac_k(2.25)
        check(False, "frac_k(2.25) must refuse")
    except ValueError:
        check(True, "frac_k refuses quarter rates")


# --------------------------------------------------------------------------------------------
# (b) Viterbi vs brute force on tiny trellises
# --------------------------------------------------------------------------------------------


def ring_states_from_bits(bits: list[int], D: list[int], L: int) -> list[int]:
    """state(p) = the L ring bits ending at S(p) (MSB first), ring = concatenated own fields."""
    R = len(bits)
    ends = list(itertools.accumulate(D))
    out = []
    for e in ends:
        v = 0
        for q in range(e - L, e):
            v = (v << 1) | bits[q % R]
        out.append(v)
    return out


def brute_force(x: np.ndarray, cb: np.ndarray, D: list[int], L: int):
    """Every ring: returns (states [N, n], costs [N], boundary in-edge of position 0 [N])."""
    R = sum(D)
    n = len(D)
    allst, costs, edges = [], [], []
    for word in range(1 << R):
        bits = [(word >> (R - 1 - i)) & 1 for i in range(R)]
        st = ring_states_from_bits(bits, D, L)
        allst.append(st)
        costs.append(sum((float(cb[s]) - float(x[p])) ** 2 for p, s in enumerate(st)))
        edges.append(st[0] >> D[0])
    return np.array(allst), np.array(costs), np.array(edges)


def test_viterbi_brute_force():
    rng = np.random.default_rng(11)
    cases = [(6, [2] * 6), (6, [1, 2] * 4), (6, [2, 1] * 4), (5, [1] * 9)]
    optimal = total = 0
    for L, D in cases:
        n = len(D)
        for trial in range(6):
            cb = rng.standard_normal(1 << L).astype(np.float32)
            x = rng.standard_normal(n).astype(np.float32)
            st_all, c_all, e_all = brute_force(x, cb, D, L)
            xt = torch.from_numpy(x)[None]
            cbt = torch.from_numpy(cb)
            # pinned pass: exact for every boundary edge
            for e0 in range(1 << (L - D[0])):
                mask = e_all == e0
                best = c_all[mask].min()
                st, cost = tq.viterbi_torch(xt, cbt, D, L=L, pinned=torch.tensor([e0]))
                rc = float(tq.ring_path_cost(st, xt, cbt)[0])
                if not (abs(float(cost[0]) - best) <= 1e-5 * max(1.0, best) and abs(rc - best) <= 1e-5 * max(1.0, best)
                        and bool(tq.ring_consistent(st, D, L)[0]) and int(st[0, 0]) >> D[0] == e0):
                    check(False, f"pinned Viterbi L={L} D={D} e0={e0}: cost {float(cost[0])} path {rc} "
                                 f"vs brute force {best}")
                    break
            else:
                check(True, f"pinned Viterbi exact L={L} D={D[:4]}.. trial {trial}")
            # two-pass tail-biting: a closed ring, exact cost, never below the optimum
            st, cost = tq.viterbi_torch(xt, cbt, D, L=L)
            rc = float(tq.ring_path_cost(st, xt, cbt)[0])
            gbest = c_all.min()
            check(bool(tq.ring_consistent(st, D, L)[0]), f"two-pass ring closed L={L} D={D[:4]}..")
            check(abs(rc - float(cost[0])) <= 1e-5 * max(1.0, rc), f"two-pass cost exact L={L}: {rc} vs {float(cost[0])}")
            check(rc >= gbest - 1e-6, f"two-pass cost {rc} below brute-force optimum {gbest}?")
            total += 1
            optimal += rc <= gbest + 1e-6
    # Rings this short hold only ~3 state-lengths, so pass 1's boundary guess is often off; no bound.
    print(f"  two-pass tail-biting hit the exact optimum in {optimal}/{total} tiny rings")


def test_tail_biting_near_optimal():
    """QTIP 3.2 / Table 2: when each half ring is many state-lengths long (EXL3: 128 positions vs a
    16-bit state of 4 positions at K = 4), the two-pass heuristic is within ~1e-4 of the exact
    tail-biting optimum (= the best pinned pass over every boundary edge). Here 128 positions, an
    8-bit state (4 positions at K = 2, 64 boundary edges), a Gaussian codebook and targets."""
    rng = np.random.default_rng(13)
    n, L, D = 128, 8, [2] * 128
    cb = torch.from_numpy(rng.standard_normal(1 << L).astype(np.float32))
    excess = []
    for _ in range(12):
        x = torch.from_numpy(rng.standard_normal((1, n)).astype(np.float32) * 0.8)
        _, c2 = tq.viterbi_torch(x, cb, D, L=L)
        edges = torch.arange(1 << (L - 2))
        _, cp = tq.viterbi_torch(x.expand(len(edges), n).contiguous(), cb, D, L=L, pinned=edges)
        opt = float(cp.min())
        check(float(c2[0]) >= opt - 1e-5, f"two-pass {float(c2[0])} below the exact optimum {opt}?")
        excess.append(float(c2[0]) / opt - 1.0)
    mean_ex = float(np.mean(excess))
    check(mean_ex < 1e-3, f"two-pass tail-biting mean excess over the exact optimum {mean_ex:.2e}")
    print(f"  two-pass tail-biting vs exact optimum ({n} positions, {L}-bit state): mean excess "
          f"{mean_ex:.2e}, max {max(excess):.2e}, exact in {sum(e < 1e-6 for e in excess)}/12")


# --------------------------------------------------------------------------------------------
# (c) native encoders vs the torch reference
# --------------------------------------------------------------------------------------------


def _compare_encoders(enc, ref, x, K, what, strict_states=True):
    st, c = enc.encode(x, K)
    st_r, c_r = ref.encode(x.to(ref.device), K)
    same = torch.equal(st.cpu(), st_r.cpu())
    dc = float((c.cpu() - c_r.cpu()).abs().max()) if c.numel() else 0.0
    check((same or not strict_states) and dc == 0.0, f"{what} K={K}: states identical={same}, max cost diff {dc}")
    if not same and not strict_states:
        print(f"  note: {what} K={K} picked another equal-cost path on a tie (costs identical)")
    return st


def test_native_cpu():
    try:
        enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    except Exception as exc:  # noqa: BLE001
        check(False, f"cpu encoder unavailable: {exc}")
        return
    ref = tq.TrellisEncoder("torch", "cpu", verbose=False)
    rng = np.random.default_rng(5)
    for K in (2, 3, 3.5, 4, 4.5, 5, 6):
        T = 6 if K < 3 else 12
        x = torch.from_numpy(rng.standard_normal((T, 256)).astype(np.float32) * 0.9)
        st = _compare_encoders(enc, ref, x, K, "cpu vs torch")
        check(bool(tq.ring_consistent(st.long(), tq.widths(K)).all()), f"cpu K={K} rings closed")
        pc = tq.ring_path_cost(st, tq.TrellisEncoder.round_targets(x), enc.cb)
        _, c = enc.encode(x, K)
        check(float((pc - c.double()).abs().max()) < 1e-3, f"cpu K={K} returned cost = path cost")
    # tie-heavy: all-zero targets (mul1 has only 913 values, so many paths tie)
    _compare_encoders(enc, ref, torch.zeros(4, 256), 4, "cpu vs torch, zero targets (ties)")
    _compare_encoders(enc, ref, torch.full((4, 256), 0.25), 3.5, "cpu vs torch, constant targets (ties)")
    enc3 = tq.TrellisEncoder("cpu", "cpu", codebook="3inst", verbose=False)
    ref3 = tq.TrellisEncoder("torch", "cpu", codebook="3inst", verbose=False)
    _compare_encoders(enc3, ref3, torch.from_numpy(rng.standard_normal((6, 256)).astype(np.float32)), 4,
                      "cpu vs torch, 3inst")


def test_native_hip():
    if not GPU:
        print("  SKIP hip encoder: set $env:HIP_VISIBLE_DEVICES='1' to include the GPU checks")
        return
    enc = tq.TrellisEncoder("hip", "cuda", verbose=False)
    ref = tq.TrellisEncoder("torch", "cuda", verbose=False)
    cpu = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    rng = np.random.default_rng(6)
    for K in (3, 3.5, 4, 4.5, 5, 5.5, 6):
        x = torch.from_numpy(rng.standard_normal((40, 256)).astype(np.float32) * 0.9)
        # torch's GPU argmin may break an exact tie differently: costs must still be identical
        _compare_encoders(enc, ref, x.cuda(), K, "hip vs torch(cuda)", strict_states=False)
        _compare_encoders(enc, cpu, x, K, "hip vs cpu")
    _compare_encoders(enc, cpu, torch.zeros(8, 256), 4, "hip vs cpu, zero targets (ties)")
    enc_t = tq.TrellisEncoder("hip", "cuda", codebook="3inst", verbose=False)
    cpu_t = tq.TrellisEncoder("cpu", "cpu", codebook="3inst", verbose=False)
    _compare_encoders(enc_t, cpu_t, torch.from_numpy(rng.standard_normal((20, 256)).astype(np.float32)), 4,
                      "hip (table codebook) vs cpu, 3inst")
    # more tiles than one launch holds: the launch loop must cover them all
    small = tq.TrellisEncoder("hip", "cuda", grid=7, tiles_per_launch=10, verbose=False)
    x = torch.from_numpy(rng.standard_normal((53, 256)).astype(np.float32))
    _compare_encoders(small, cpu, x, 4, "hip in 6 bounded launches vs cpu")


# --------------------------------------------------------------------------------------------
# (d) bitstream
# --------------------------------------------------------------------------------------------


def slow_pack(states: list[int], D: list[int]) -> list[int]:
    bits = []
    for s, d in zip(states, D):
        bits += [(s >> (d - 1 - i)) & 1 for i in range(d)]
    words = []
    for w in range(len(bits) // 32):
        v = 0
        for b in bits[32 * w:32 * w + 32]:
            v = (v << 1) | b
        words.append(v)
    return words


def test_bitstream():
    rng = np.random.default_rng(7)
    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    for K in (2, 3, 3.5, 4, 4.5, 5, 6):
        x = torch.from_numpy(rng.standard_normal((5, 256)).astype(np.float32))
        st, _ = enc.encode(x, K)
        w = tq.pack_states(st, K)
        check(w.shape == (5, tq.trellis_words(K)) and w.dtype == torch.int32, f"K={K} word shape {tuple(w.shape)}")
        check(torch.equal(tq.unpack_states(w, K), st.long()), f"K={K} unpack(pack(states)) == states")
        slow = slow_pack(st[0].tolist(), tq.widths(K))
        got = [int(v) & 0xFFFFFFFF for v in w[0].tolist()]
        check(got == slow, f"K={K} MSB-first word layout matches the slow bit-string packer")
        # any bit pattern is a valid ring
        rw = torch.from_numpy(rng.integers(-2**31, 2**31, size=(4, tq.trellis_words(K)), dtype=np.int64)
                              .astype(np.int32))
        us = tq.unpack_states(rw, K)
        check(bool(tq.ring_consistent(us, tq.widths(K)).all()), f"K={K} random words decode to closed rings")
        check(torch.equal(tq.pack_states(us, K), rw), f"K={K} pack(unpack(words)) == words")
    # measured bits: 256 K bits per tile exactly
    check(tq.trellis_words(3.5) * 32 == 256 * 3.5, "3.5 bpw tile = 896 bits")


# --------------------------------------------------------------------------------------------
# (e) incoherence, LDL, LDLQ, quantize_group, int4, allocation, .hess, Lloyd-Max, Gaussian MSE
# --------------------------------------------------------------------------------------------


def test_incoherence():
    P = tq.hadamard128("cpu")
    check(float((P @ P - torch.eye(128)).abs().max()) < 1e-5, "P_128 is orthogonal and symmetric (P P = I)")
    i, j = 5, 77
    close(float(P[i, j]) * math.sqrt(128), (-1) ** bin(i & j).count("1"), 1e-6, "P entries (-1)^popcount(i&j)/sqrt(128)")
    g = torch.Generator().manual_seed(0)
    W = torch.randn(256, 384, generator=g)
    check(float((tq.had_l(tq.had_l(W)) - W).abs().max()) < 1e-5, "had_l twice = identity")
    check(float((tq.had_r(tq.had_r(W)) - W).abs().max()) < 1e-5, "had_r twice = identity")
    W[:, 7] = 0.0  # a dead output channel
    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    su_s, sv_s = tq.random_signs(256, 1), tq.random_signs(384, 2)
    check(bool((su_s.abs() == 1).all()), "random signs are +-1")
    W_r, su, sv, rec = tq.regularize(W, su_s, sv_s, 4, enc)
    back = tq.unrotate(W_r, su, sv)
    rel = float((back - W).norm() / W.norm())
    check(rel < 1e-5, f"regularize -> unrotate round trip rel {rel}")
    check(float(back[:, 7].abs().max()) == 0.0 and rec["dead_out_channels"] == 1, "dead output channel -> sv 0")
    rms = float(torch.sqrt((W_r ** 2).mean()) / rec["g_scale"]["g"])
    close(rms, tq.CODEBOOK_SCALE, 0.02, "regularized rows have rms codebook_scale before g")
    check(0.3 < rec["g_scale"]["g"] < 1.5, f"g scale {rec['g_scale']['g']} plausible")


def test_sample_scale_tiles():
    """EXL3's exact sample order (quantize.py sample_scale_tiles): the width-3 wrapped diagonal
    i-major (repeat_interleave), then the num_x highest-mean-square tiles descending, then the num_x
    lowest ascending, each tile in tensor-core order. Stage 1 scores every third sample, so the
    order decides its subset."""
    g = torch.Generator().manual_seed(71)
    tk, tn = 5, 12
    W = torch.randn(16 * tk, 16 * tn, generator=g) * torch.linspace(0.5, 2.0, 16 * tn)[None, :]
    s = tq.sample_scale_tiles(W, 4)
    tiles = tq.matrix_to_tiles(W)  # [tk, tn, 256] row-major
    perm = torch.from_numpy(tq.tensor_core_perm().copy())
    diag_len = max(tk, tn)
    num_x = min(max(8, 3 * diag_len // 16), (tk * tn + 1) // 2)
    check(s.shape == (3 * diag_len + 2 * num_x, 256), f"sample count {tuple(s.shape)}")
    ok = all(torch.equal(s[j], tiles[(j // 3) % tk, (j // 3 + j % 3) % tn][perm]) for j in range(3 * diag_len))
    check(ok, "diagonal samples are i-major: (i mod tk, (i + w) mod tn), w innermost")
    ms = (tiles.reshape(-1, 256) ** 2).mean(dim=1)
    order = torch.argsort(ms, descending=True)
    hi = s[3 * diag_len:3 * diag_len + num_x]
    lo = s[3 * diag_len + num_x:]
    check(torch.equal(hi, tiles.reshape(-1, 256)[order[:num_x]][:, perm]), "hi tiles in descending mean square")
    check(torch.equal(lo, tiles.reshape(-1, 256)[order.flip(0)[:num_x]][:, perm]), "lo tiles in ascending mean square")
    close(float(tq.sample_scale_tiles(W, 3)[0, 0] / s[0, 0]), tq.LDLQ_DRIFT[3], 1e-6, "LDLQ drift at K = 3")


def random_spd(k: int, seed: int, dead: int | None = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(4 * k, k, generator=g) @ torch.diag(torch.linspace(0.2, 3.0, k))
    X[:, : k // 4] += X[:, k // 4: k // 2]  # correlated channels, so feedback matters
    if dead is not None:
        X[:, dead] = 0.0
    return (X.T @ X) / X.shape[0]


def test_block_ldl():
    H = random_spd(64, 3)
    L, retries = tq.block_ldl(H, 16)
    check(retries == 0, "block_ldl no retries on SPD H")
    C = torch.linalg.cholesky(H)
    Dm = torch.zeros_like(H)
    for b in range(4):
        s = slice(16 * b, 16 * b + 16)
        Dm[s, s] = C[s, s] @ C[s, s].T
        check(float(L[s, s].abs().max()) == 0.0, f"diagonal block {b} zeroed")
    Lu = L + torch.eye(64)
    err = float((Lu @ Dm @ Lu.T - H).abs().max() / H.abs().max())
    check(err < 1e-5, f"(L + I) D (L + I)^T == H, rel err {err}")
    check(float(torch.triu(L, 1).abs().max()) == 0.0, "L strictly block lower")
    I = torch.eye(256)
    L0, _ = tq.block_ldl(I)
    check(float(L0.abs().max()) == 0.0, "block_ldl(I) = 0")
    Lr, _ = tq.block_ldl(tq.rotate_hessian(tq.damp_hessian(I), tq.random_signs(256, 4)))
    check(float(Lr.abs().max()) < 1e-5, f"rotated identity keeps L ~ 0 ({float(Lr.abs().max())})")


def test_ldlq_identity():
    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    g = torch.Generator().manual_seed(9)
    W_r = torch.randn(256, 128, generator=g) * 0.9
    L, _ = tq.block_ldl(torch.eye(256))
    Q, states = tq.ldlq(W_r, L, 4, enc)
    perm = torch.from_numpy(tq.tensor_core_perm().copy())
    tiles = tq.matrix_to_tiles(W_r).reshape(-1, 256)[:, perm]
    vals, st = enc.quantize(tiles, 4)
    check(torch.equal(states.reshape(-1, 256), st), "LDLQ with H = I encodes every tile on its own")
    q_direct = tq.tiles_to_matrix(vals[:, torch.from_numpy(tq.tensor_core_perm_inv().copy())].reshape(16, 8, 256))
    check(torch.equal(Q, q_direct), "LDLQ with H = I: Q = plain per-tile quantization")
    words = tq.pack_states(states.reshape(-1, 256), 4).reshape(16, 8, -1)
    check(torch.equal(tq.decode_words(words, 4, enc.cb, chunk=5), Q), "decode_words(pack(states)) = Q (chunked)")
    # the concatenated pass (exl3 groups) equals separate passes: LDLQ never mixes columns
    H = random_spd(256, 8)
    L, _ = tq.block_ldl(tq.rotate_hessian(tq.damp_hessian(H), tq.random_signs(256, 3)))
    A, B = W_r[:, :128], torch.randn(256, 256, generator=g) * 0.9
    Qc, sc = tq.ldlq(torch.cat([A, B], dim=1), L, 3.5, enc)
    Qa, sa = tq.ldlq(A.contiguous(), L, 3.5, enc)
    Qb, sb = tq.ldlq(B, L, 3.5, enc)
    check(torch.equal(Qc, torch.cat([Qa, Qb], dim=1)) and torch.equal(sc, torch.cat([sa, sb], dim=1)),
          "LDLQ over concatenated tensors = separate passes")


def test_quantize_group():
    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    k, n = 256, 256
    H = random_spd(k, 21, dead=13)
    check(float(H[13].abs().max()) == 0.0, "synthetic dead input channel has a zero H row")
    g = torch.Generator().manual_seed(22)
    Whf = torch.randn(n, k, generator=g) * 0.02
    Whf[:, 5] *= 8.0  # an outlier input channel
    res = tq.quantize_group({"w": Whf.to(torch.bfloat16)}, H, {"w": 4}, enc, 1, {"w": 2}, verbose=False)["w"]
    rec = res["record"]
    check(all(math.isfinite(rec[k_]) for k_ in ("proxy", "proxy_damped_h", "rel_weight_err")),
          "quantize_group: finite errors with a dead channel")
    check(rec["bits"]["trellis"] == k * n * 4 and rec["bits"]["scales"] == 16 * (k + n), "bits measured")
    close(rec["bits"]["bpw"], 4 + (16 * (k + n) + 32) / (k * n), 1e-12, "bpw = K + scale overhead")
    check(res["words"].shape == (k // 16, n // 16, 32) and res["suh"].dtype == torch.float16, "stored shapes")
    W = Whf.to(torch.bfloat16).float().T
    W_st = tq.reconstruct(res["words"], res["suh"], res["svh"], 4, enc.cb)
    close(tq.proxy_error(W, W_st, H), rec["proxy"], 1e-6, "record proxy = proxy of the stored bits")
    Hd = tq.damp_hessian(H)
    # refit minimizes the damped-H proxy with the trellis fixed; fp16 storage of su/sv may add a hair
    check(rec["proxy_damped_h"] <= rec["proxy_before_refit_damped_h"] * (1 + 1e-3),
          f"refit does not hurt (damped H): {rec['proxy_before_refit_damped_h']} -> {rec['proxy_damped_h']}")
    close(tq.proxy_error(W, W_st, Hd), rec["proxy_damped_h"], 1e-6, "record damped proxy")
    check(rec["proxy"] < 0.02, f"K=4 proxy {rec['proxy']} small")
    # matched basis runs too
    res2 = tq.quantize_group({"w": Whf}, H, {"w": 4}, enc, 1, {"w": 2}, hessian_basis="matched",
                             verbose=False)["w"]["record"]
    check(math.isfinite(res2["proxy"]) and res2["proxy"] < 0.02, f"matched basis proxy {res2['proxy']}")
    # feedback helps on a correlated H: compare against no feedback (L = 0), same regularization
    W32 = Whf.float().T.contiguous()
    su_sign = tq.random_signs(k, 1)
    W_r, su, sv, _ = tq.regularize(W32, su_sign, tq.random_signs(n, 2), 4, enc)
    L, _ = tq.block_ldl(tq.rotate_hessian(Hd, su_sign))
    Qf, _ = tq.ldlq(W_r, L, 4, enc)
    Q0, _ = tq.ldlq(W_r, torch.zeros_like(L), 4, enc)
    pf = tq.proxy_error(W32, tq.unrotate(Qf, su, sv), H)
    p0 = tq.proxy_error(W32, tq.unrotate(Q0, su, sv), H)
    check(pf < p0, f"LDLQ feedback lowers the proxy: {pf:.5f} vs no feedback {p0:.5f}")
    print(f"  synthetic 256x256 K=4: proxy {rec['proxy']:.5f} (before refit {rec['proxy_before_refit']:.5f}),"
          f" matched basis {res2['proxy']:.5f}, no-feedback {p0:.5f}")


def test_refit():
    g = torch.Generator().manual_seed(31)
    H = tq.damp_hessian(random_spd(128, 30))
    W = torch.randn(128, 64, generator=g)
    Q = W + 0.1 * torch.randn(128, 64, generator=g)
    Q = Q * 1.3
    before = tq.proxy_error(W, Q, H)
    Q2, su, sv, _ = tq.refit_scales(W, Q, torch.ones(128), torch.ones(64), H)
    after = tq.proxy_error(W, Q2, H)
    check(after < before, f"refit_scales lowers the damped-H proxy {before:.4f} -> {after:.4f}")
    check(float((Q2 - Q * su[:, None] * sv[None, :]).abs().max()) < 1e-4, "refit only rescales rows/columns")


def test_int4():
    g = torch.Generator().manual_seed(41)
    W = torch.randn(32, 256, generator=g)
    dq = tq.int4_ldlq(W, torch.eye(256), group=64, damp=0.0)
    ok = True
    for gi in range(4):
        wg = W[:, 64 * gi:64 * gi + 64]
        sc, z = tq._int4_group_search(wg, torch.ones(64))
        q = torch.clamp(tq._round_half_away(wg / sc[:, None]) + z[:, None], 0, 15)
        ref = sc[:, None] * (q - z[:, None])
        ok &= bool(torch.allclose(dq[:, 64 * gi:64 * gi + 64], ref, atol=1e-6))
    check(ok, "int4 LDLQ with H = I is the per-group search without feedback")
    H = random_spd(256, 42)
    rtn = tq.int4_ldlq(W, torch.eye(256), group=64, damp=0.0)
    fb = tq.int4_ldlq(W, H, group=64, damp=0.01)
    check(tq.proxy_error(W.T, fb.T, H) < tq.proxy_error(W.T, rtn.T, H), "int4 LDLQ beats no feedback on H")


def test_allocate():
    ts = [{"name": f"L{i}.m{j}", "layer": i, "idx": j, "numel": 1000 * (1 + j)} for i in range(8) for j in range(3)]
    for bpw, want in ((4.0, {4.0}), (3.5, {3.5}), (5.0, {5.0})):
        r = tq.allocate(ts, bpw)
        check(set(r.values()) == want, f"allocate {bpw}: {set(r.values())}")
    r = tq.allocate(ts, 4.5)
    tot = sum(t["numel"] * r[t["name"]] for t in ts)
    check(tot <= int(4.5 * sum(t["numel"] for t in ts)), "4.5 mix within budget")
    # budget 24000 extra bits = four 6000-weight layers, taken from both ends of the stack inwards
    by_layer = {i: {r[f"L{i}.m{j}"] for j in range(3)} for i in range(8)}
    check(all(by_layer[i] == {5.0} for i in (0, 1, 6, 7)) and all(by_layer[i] == {4.0} for i in (2, 3, 4, 5)),
          f"4.5 mix promotes the ends first: {by_layer}")
    check(tq.rate_floor(4.5) == 4.0 and tq.rate_floor(3.7) == 3.5 and tq.rate_next(3.5) == 4.0
          and tq.rate_next(4.0) == 5.0, "rate_floor / rate_next")
    # EXL3's qgroups: a group is promoted only as a whole. 6 layers of (a: two 1000-weight tensors,
    # one qgroup) + (b: one 1000-weight tensor), 4.25 bpw: 4500 extra bits. Layer 0 goes up whole
    # (3000); at layer 5 group a (2000) no longer fits but b (1000) does. Per tensor, a0 of layer 5
    # would have been promoted instead of b.
    gs = [{"name": f"L{i}.{m}", "layer": i, "idx": j, "numel": 1000, "qgroup": q}
          for i in range(6) for j, (m, q) in enumerate((("a0", "a"), ("a1", "a"), ("b", "b")))]
    r = tq.allocate(gs, 4.25)
    want = {f"L0.{m}": 5.0 for m in ("a0", "a1", "b")} | {"L5.b": 5.0}
    check(all(r[n] == want.get(n, 4.0) for n in r), f"qgroups promote whole groups only: {r}")
    solo = tq.allocate([{k: v for k, v in t.items() if k != "qgroup"} for t in gs], 4.25)
    check(solo["L5.a0"] == 5.0 and solo["L5.b"] == 4.0, f"without qgroups every tensor is its own group: {solo}")


def test_hess_reader():
    with tempfile.TemporaryDirectory() as td:
        H = random_spd(48, 50).numpy().astype(np.float32)
        path = Path(td) / "t.hess"
        with open(path, "wb") as f:
            f.write(tq.HESS_HEADER.pack(b"R4DXHES1", 48, 1, 1234, float(np.diagonal(H).astype(np.float64).sum())))
            for i in range(48):
                f.write(H[i, i:].tobytes())
        got, rows = tq.read_hess(path)
        check(rows == 1234 and np.array_equal(got.numpy(), np.triu(H) + np.triu(H, 1).T), "read_hess round trip")


def test_lloyd_max_and_gaussian():
    for bits, want in ((1, 0.3634), (2, 0.1175), (3, 0.03454), (4, 0.009497)):
        close(tq.lloyd_max_mse(bits), want, 2e-4 * max(1, want * 10), f"Lloyd-Max {bits} bits")
    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    rng = np.random.default_rng(12)
    x = torch.from_numpy(rng.standard_normal((128, 256)).astype(np.float32))
    q, _ = enc.quantize(x * 0.9, 4)
    mse = float(((q / 0.9 - x) ** 2).mean())
    check(0.0040 < mse < 0.0050, f"Gaussian K=4 MSE {mse} (spec 0.00441; Lloyd-Max 0.0095)")


# --------------------------------------------------------------------------------------------
# (f) override directory plumbing
# --------------------------------------------------------------------------------------------


def test_override_plumbing():
    import full_logits_golden as flg
    import kl_report

    enc = tq.TrellisEncoder("cpu", "cpu", verbose=False)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        name = tq.hf_name(0, "mlp.down_proj")
        H = random_spd(256, 60)
        Whf = torch.randn(128, 256, generator=torch.Generator().manual_seed(61)) * 0.02
        r = tq.quantize_group({name: Whf}, H, {name: 3.5}, enc, 1, {name: 2}, verbose=False)[name]
        from safetensors.torch import save_file

        save_file({name + ".trellis": r["words"], name + ".suh": r["suh"], name + ".svh": r["svh"]},
                  str(td / "L00.safetensors"))
        rec = dict(r["record"], file="L00.safetensors")
        man = {"format": tq.OVERRIDE_FORMAT, "version": tq.OVERRIDE_VERSION, "encoding": tq.ENCODING_TRELLIS,
               "complete": True, "tensors": {name: rec}, "summary": tq.summarize_tensors({name: rec})}
        tq.write_json_atomic(td / tq.MANIFEST_NAME, man)
        m = tq.load_manifest(td)
        w = tq.load_override_tensor(m, name, "cpu")
        ref = tq.reconstruct(r["words"], r["suh"], r["svh"], 3.5, enc.cb).T
        check(torch.equal(w, ref), "load_override_tensor = reconstruct(stored), HF layout")
        rel = float((w - Whf).norm() / Whf.norm())
        close(rel, rec["rel_weight_err"], 1e-4, "reloaded rel error = recorded")
        check(rel < flg.override_rel_bound(3.5), f"rel {rel} under the loader's bound")
        close(man["summary"]["bpw"], 3.5 + (16 * (128 + 256) + 32) / (128 * 256), 1e-9, "summary bpw measured")
        # out-dir refusals: bf16 ref vs override vs gguf
        out = td / "dump"
        out.mkdir()
        (out / "a.meta.json").write_text(json.dumps({"source": "reference-overrideweights", "n_layers": 64,
                                                     "weights_override": {"manifest_sha256": "aa" * 32}}))
        check(flg.out_dir_conflicts(out, "override", "aa" * 32, 64) == [], "same override manifest: allowed")
        check(any("another override" in c for c in flg.out_dir_conflicts(out, "override", "bb" * 32, 64)),
              "another override manifest: refused")
        check(any("--weights-override dump" in c for c in flg.out_dir_conflicts(out, "bf16", None, 64)),
              "bf16 reference into an override dir: refused")
        check(any("--weights-override dump" in c for c in flg.out_dir_conflicts(out, "gguf", "cc" * 32, 64)),
              "gguf dump into an override dir: refused")
        ref_dir = td / "ref"
        ref_dir.mkdir()
        (ref_dir / "reference_run.json").write_text(json.dumps({"n_layers": 64}))
        check(any("bf16 reference" in c for c in flg.out_dir_conflicts(ref_dir, "override", "aa" * 32, 64)),
              "override dump into a bf16 reference dir: refused")
        ident = kl_report.side_identity(out, json.loads((out / "a.meta.json").read_text()))
        check(ident["override_manifest_sha256"] == "aa" * 32, "kl_report identity carries the manifest sha")


def test_manifest_resume_key():
    """quantize-model's resume key holds the code hashes, and write_manifest uses a layer record only
    when its job AND every tensor's K match this run (a reused directory never yields a complete
    manifest over layers of another rate or older code)."""
    import argparse

    code = tq.code_sha256()
    prov = tq.provenance()
    check(code == {"trellis_quant": prov["trellis_quant_sha256"], "trellis_viterbi": prov["trellis_viterbi_sha256"]},
          "code_sha256 = the provenance hashes")
    a, b = tq.hf_name(0, "mlp.down_proj"), tq.hf_name(1, "mlp.down_proj")
    rates = {a: 4.0, b: 4.0}
    job = {"model_dir": "m", "config_sha256": "c", "hessian_basis": "exl3", "code_sha256": code}

    def rec(K):
        bits = 256 * K
        return {"encoding": tq.ENCODING_TRELLIS, "K": K, "k": 16, "n": 16, "proxy": 0.001,
                "bits": {"trellis": bits, "scales": 512, "total": bits + 544, "bpw": (bits + 544) / 256}}

    def layer(td, i, name, K, j):
        tq.write_json_atomic(td / f"L{i:02d}.json", {"layer": i, "job": j, "tensors": {name: rec(K)}})

    args = argparse.Namespace(bpw=None, K=4.0)
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        layer(td, 0, a, 4.0, job)
        layer(td, 1, b, 5.0, job)  # a leftover of a K = 5 run in the same directory
        m = tq.write_manifest(td, job, rates, args)
        check(not m["complete"] and m["missing"] == [b] and m["stale_layers"] == ["L01.json"],
              f"a layer at another K is stale, not complete: {m['complete']} {m['missing']} {m['stale_layers']}")
        old = dict(job, code_sha256={"trellis_quant": "0" * 64, "trellis_viterbi": code["trellis_viterbi"]})
        layer(td, 1, b, 4.0, old)  # right K, older code
        m = tq.write_manifest(td, job, rates, args)
        check(not m["complete"] and m["stale_layers"] == ["L01.json"], "a layer of older code is stale")
        check(not tq.layer_record_usable(json.loads((td / "L01.json").read_text()), job, rates),
              "resume redoes a layer of older code")
        layer(td, 1, b, 4.0, job)
        m = tq.write_manifest(td, job, rates, args)
        check(m["complete"] and m["layers_done"] == [0, 1] and m["stale_layers"] == [], "complete once both match")


def test_mix_real_model():
    """`mix --bpw 4.5` over two fake uniform-rate manifests of the real checkpoint's 400 linears
    (reads only the checkpoint's config and safetensors headers). SKIPped without the checkpoint."""
    import argparse

    model_dir = Path(os.environ.get("R4DX_MODEL_DIR", r"D:\models\Huihui-Qwen3.8-27B-abliterated"))
    if not (model_dir / "model.safetensors.index.json").exists():
        print(f"  SKIP mix on the real model: no checkpoint at {model_dir}")
        return
    lin = tq.model_linears(model_dir)
    check(len(lin) == 400 and sum(t["numel"] for t in lin) == 24_326_963_200 == tq.MODEL_TILES * 256,
          "400 quantized decoder linears, 24.33 G weights")
    check(all(t["qgroup"] == tq.QGROUPS[t["module"]] for t in lin), "model_linears carries EXL3's qgroups")
    for bpw in (3.8, 4.25, 4.7):  # off the planned points, where grouping matters
        r = tq.allocate(lin, bpw)
        per_group: dict = {}
        for t in lin:
            per_group.setdefault((t["layer"], t["qgroup"]), set()).add(r[t["name"]])
        check(all(len(v) == 1 for v in per_group.values()), f"allocate {bpw}: every qgroup at one K")
        check(sum(t["numel"] * r[t["name"]] for t in lin) <= int(bpw * sum(t["numel"] for t in lin)),
              f"allocate {bpw}: within budget")
    job = {"model_dir": str(model_dir), "config_sha256": "x", "hessian_manifest_sha256": "y",
           "hessian_basis": "exl3", "codebook": "mul1"}
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for K in (4, 5):
            d = td / f"K{K}"
            d.mkdir()
            tens = {}
            for t in lin:
                bits = t["numel"] * K
                sc = 16 * (t["k"] + t["n"])
                tens[t["name"]] = {"encoding": tq.ENCODING_TRELLIS, "K": float(K), "codebook": "mul1",
                                   "k": t["k"], "n": t["n"], "file": f"L{t['layer']:02d}.safetensors",
                                   "bits": {"trellis": bits, "scales": sc, "total": bits + sc + 32,
                                            "bpw": (bits + sc + 32) / t["numel"]}}
            tq.write_json_atomic(d / tq.MANIFEST_NAME, {"format": tq.OVERRIDE_FORMAT, "version": tq.OVERRIDE_VERSION,
                                                        "complete": True, "tensors": tens, **job})
        tq.cmd_mix(argparse.Namespace(src=[str(td / "K4"), str(td / "K5")], bpw=4.5, out_dir=td / "mix"))
        m = tq.load_manifest(td / "mix")
        ks = {n: r["K"] for n, r in m["tensors"].items()}
        check(sum(1 for v in ks.values() if v == 5.0) == 200 and sum(1 for v in ks.values() if v == 4.0) == 200,
              "4.5 = 200 tensors at K=5, 200 at K=4")
        five = sorted({tq.parse_hf_name(n)[0] for n, v in ks.items() if v == 5.0})
        check(five == list(range(16)) + list(range(48, 64)), f"K=5 layers are 0-15 and 48-63: {five}")
        r5 = m["tensors"][tq.hf_name(0, "mlp.down_proj")]
        check(Path(r5["file"]).is_absolute() and Path(r5["file"]).parent == td / "K5", "K=5 entries point into K5")
        close(m["summary"]["bpw"], 4.5045, 2e-4, "mix measured bpw")
        close(m["summary"]["decode_gib"], 12.757, 2e-3, "mix GiB (docs/trellis.md 10)")


def main() -> int:
    tests = [test_codebooks, test_perm_widths, test_viterbi_brute_force, test_tail_biting_near_optimal,
             test_native_cpu, test_native_hip,
             test_bitstream, test_incoherence, test_sample_scale_tiles, test_block_ldl,
             test_ldlq_identity, test_quantize_group,
             test_refit, test_int4, test_allocate, test_hess_reader, test_lloyd_max_and_gaussian,
             test_override_plumbing, test_manifest_resume_key, test_mix_real_model]
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
