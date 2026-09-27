"""Correctness for the trellis GEMM (r4d_gemm_trellis_nt_m64 and its _raw entry) through r4d.pyd.

    HIP_VISIBLE_DEVICES=1 PYTHONPATH=. python test_trellis_gemm.py
    (Windows: build_windows.ps1 puts r4d.pyd in build-win, so PYTHONPATH=build-win.)

Random words are valid trellis weights at any rate (every 16-bit state hashes to a finite f16), so
no real weight is needed. The weight a call multiplies by is taken from trellis_reconstruct_f16,
the same tile decode the GEMM runs, whole-matrix (r4dx's test_trellis_decode holds that decode
bit-exact against the Python reference decode_words); everything after it is checked here on the
CPU in torch:

  raw    fp32 C = A @ Q against an fp64 A @ Q (relative Frobenius <= 1e-5), per A part: columns
         >= n_split read a1; one-hot A rows return rows of Q exactly (the GEMM and reconstruct
         agree on the lane map and the pair grid);
  full   bf16 C = bf16_rn((FWHT128(S) * svh) * out_scale) of the _raw sums S, BIT FOR BIT, with
         the FWHT in fp32 in r4d_fwht128.h's stage order (lg = 0..6, (a + b, a - b)) and torch's
         round-to-nearest-even bf16; and against the fp64 linear (relative Frobenius <= 4e-3);
         rows >= M untouched;
  rows   an M-row call equals the M = 1 call on each of its rows (row identity, M <= 16);
  tickets zero after every call that uses them; zero_tickets clears stale ones;
  rejects KB 3, NP * U > 8, a split group without ws / tickets, n_split off a 128 boundary.

Tunings cover the unsplit 128- and 256-wide blocks (the in-LDS finish) and split groups of 32-,
128- and 256-wide blocks at SKG 1-8 (the last block's finish), at KB 4 and 5, one and two A parts,
prescale 0 and 4, M in {1, 5, 16, 17, 64}.
"""
import math
import sys

import torch
import r4d

K, N = 1024, 512
MS = (1, 5, 16, 17, 64)
# (WV, SK, MT, NP, SKG, U, NT)
TUNINGS = [
    (4, 2, 1, 1, 1, 2, 1),   # Wc 128, unsplit
    (2, 8, 1, 4, 1, 2, 0),   # Wc 256, unsplit: two groups per block
    (1, 4, 1, 1, 2, 4, 1),   # Wc 32, SKG 2: 8 blocks per group
    (2, 4, 1, 2, 8, 1, 1),   # Wc 128, SKG 8
    (4, 2, 1, 2, 2, 2, 1),   # Wc 256, SKG 2: two groups per ticket
]
SENTINEL = 0x7FC1   # a bf16 NaN no finite linear produces


def stream():
    return torch.cuda.current_stream().cuda_stream


def split_group(t):
    WV, SK, MT, NP, SKG, U, NT = t
    return SKG > 1 or WV * NP * 32 < 128


def for_m(t, M):
    """A split group needs every row tile in one block: MT >= ceil(M / 16)."""
    if split_group(t):
        t = t[:2] + (max(t[2], (M + 15) // 16),) + t[3:]
    return t


def fwht128(x):
    """Unnormalized natural-order FWHT on each 128-block of the last dim, in x's dtype, stages
    lg = 0..6, each pair (i, i + 2^lg) -> (a + b, a - b): in fp32, r4d_fwht128.h's operations in its
    order, so bit-exact against the kernel."""
    shape = x.shape
    v = x.reshape(-1, 128).clone()
    for lg in range(7):
        h = 1 << lg
        w = v.view(-1, 128 // (2 * h), 2, h)
        a, b = w[:, :, 0, :], w[:, :, 1, :]
        v = torch.stack((a + b, a - b), dim=2).reshape(-1, 128)
    return v.reshape(shape)


class Case:
    def __init__(self, KB, parts, seed):
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.KB, self.parts = KB, parts
        self.n_split = N // 2 if parts == 2 else N
        words = torch.randint(-2**31, 2**31, (K * N * KB // 32,), generator=g, dtype=torch.int64)
        self.w = words.to(torch.int32).cuda()
        q = torch.empty(K, N, dtype=torch.float16, device="cuda")
        r4d.trellis_reconstruct_f16(self.w.data_ptr(), q.data_ptr(), K, N, KB, stream())
        torch.cuda.synchronize()
        self.q = q.cpu()
        sign = torch.randint(0, 2, (N,), generator=g).float() * 2 - 1
        mag = torch.exp(torch.empty(N).uniform_(math.log(0.6), math.log(1.9), generator=g))
        self.svh = (sign * mag).to(torch.float16).float()      # fp16 values, widened
        self.a = [(torch.randn(64, K, generator=g) * 0.03).to(torch.float16) for _ in range(2)]
        self.a_d = [x.cuda() for x in self.a]
        self.svh_d = self.svh.cuda()
        self.ws = torch.zeros(r4d.gemm_trellis_nt_m64_ws_bytes(64, N, 8) // 4, dtype=torch.float32,
                              device="cuda")
        self.tickets = torch.zeros(r4d.gemm_trellis_nt_m64_tickets_bytes(N) // 4, dtype=torch.int32,
                                   device="cuda")
        # fp64 A_p(n) @ Q
        qd = self.q.double()
        ref = self.a[0].double() @ qd
        if parts == 2:
            ref[:, self.n_split:] = (self.a[1].double() @ qd)[:, self.n_split:]
        self.ref = ref

    def a1_ptr(self):
        return self.a_d[1].data_ptr() if self.parts == 2 else 0

    def raw(self, t, M, a0=None):
        WV, SK, MT, NP, SKG, U, NT = t
        c = torch.zeros(64, N, dtype=torch.float32, device="cuda")
        r4d.gemm_trellis_nt_m64_raw((a0 if a0 is not None else self.a_d[0]).data_ptr(), self.a1_ptr(),
                                    self.n_split, self.w.data_ptr(), c.data_ptr(), self.ws.data_ptr(),
                                    self.tickets.data_ptr(), M, K, N, self.KB, WV, SK, MT, NP, SKG, U,
                                    NT, 0, stream())
        torch.cuda.synchronize()
        return c.cpu()

    def full(self, t, M, out_scale, a0=None, a1=None):
        WV, SK, MT, NP, SKG, U, NT = t
        c = torch.full((64, N), SENTINEL, dtype=torch.int16, device="cuda")
        a1p = (a1.data_ptr() if a1 is not None else self.a1_ptr())
        r4d.gemm_trellis_nt_m64((a0 if a0 is not None else self.a_d[0]).data_ptr(), a1p, self.n_split,
                                self.w.data_ptr(), self.svh_d.data_ptr(), c.data_ptr(), self.ws.data_ptr(),
                                self.tickets.data_ptr(), M, K, N, self.KB, WV, SK, MT, NP, SKG, U, NT,
                                out_scale, stream())
        torch.cuda.synchronize()
        return c.cpu()

    def tickets_zero(self):
        return int(self.tickets.abs().sum().item()) == 0


def bf16_bits(x):
    return x.to(torch.bfloat16).view(torch.int16)


def expect_throw(what, f):
    try:
        f()
    except Exception as e:                                          # noqa: BLE001
        print("%-58s rejected: %s" % (what, str(e)[:60]))
        return 0
    print("%-58s NOT REJECTED   <-- FAIL" % what)
    return 1


bad = 0
for kb, want in ((3, False), (4, True), (5, True), (6, False)):
    if bool(r4d.gemm_trellis_nt_m64_has_rate(kb)) != want:
        print("has_rate(%d) != %s   <-- FAIL" % (kb, want))
        bad += 1
if r4d.GEMM_TRELLIS_MAX_M != 64:
    print("GEMM_TRELLIS_MAX_M = %d, want 64   <-- FAIL" % r4d.GEMM_TRELLIS_MAX_M)
    bad += 1
if r4d.gemm_trellis_nt_m64_ws_bytes(16, 5120, 4) != 4 * 16 * 5120 * 4 or \
        r4d.gemm_trellis_nt_m64_tickets_bytes(5120) != 160:
    print("ws_bytes / tickets_bytes   <-- FAIL")
    bad += 1

print("%-4s %-3s %-2s %-26s %-3s %-10s %-10s %-10s %s" %
      ("KB", "P", "s", "WV/SK/MT/NP/SKG/U/NT", "M", "raw rel", "full rel", "bits off", ""))
seed = 0
calls = 0
for KB in (4, 5):
    for parts in (1, 2):
        seed += 1
        cs = Case(KB, parts, seed)
        if parts == 1:
            # One-hot rows: raw returns rows of Q exactly, at every tuning.
            rows = torch.randperm(K, generator=torch.Generator().manual_seed(seed))[:16]
            onehot = torch.zeros(64, K, dtype=torch.float16)
            onehot[torch.arange(16), rows] = 1.0
            for t in TUNINGS:
                oh = cs.raw(for_m(t, 16), 16, a0=onehot.cuda())
                if not torch.equal(oh[:16], cs.q[rows].float()):
                    print("KB %d %s one-hot rows of _raw != reconstruct   <-- FAIL" %
                          (KB, "/".join(map(str, t))))
                    bad += 1
        for s in ((0, 4) if parts == 2 else (0,)):
            out_scale = 2.0 ** -s / math.sqrt(128.0)
            osc = torch.tensor(out_scale, dtype=torch.float32)      # the float the kernel receives
            y64 = fwht128(cs.ref) * cs.svh.double() * out_scale
            for t0 in TUNINGS:
                for M in MS:
                    t = for_m(t0, M)
                    try:
                        raw = cs.raw(t, M)
                        full = cs.full(t, M, out_scale)
                    except RuntimeError:
                        continue                                   # not instantiated at this MT
                    calls += 1
                    ref = cs.ref[:M]
                    rel_raw = ((raw[:M].double() - ref).norm() / ref.norm()).item()
                    want_bits = bf16_bits((fwht128(raw[:M]) * cs.svh) * osc)
                    off = int((full[:M] != want_bits).sum().item())
                    got = full[:M].view(torch.bfloat16).double()
                    rel_full = ((got - y64[:M]).norm() / y64[:M].norm()).item()
                    untouched = bool((full[M:] == SENTINEL).all().item())
                    tz = cs.tickets_zero() if split_group(t) else True
                    fail = rel_raw > 1e-5 or off != 0 or rel_full > 4e-3 or not untouched or not tz
                    bad += fail
                    if fail or M in (1, 64):
                        print("%-4d %-3d %-2d %-26s %-3d %.3e  %.3e  %-10d %s" %
                              (KB, parts, s, "/".join(map(str, t)), M, rel_raw, rel_full, off,
                               ("<-- FAIL" + ("" if untouched else " rows>=M written") +
                                ("" if tz else " tickets")) if fail else ""))
        # Row identity: each row of an M-row call equals the M = 1 call on that row (one tuning for
        # every M <= 16; MT 1 already holds 16 rows).
        for t in TUNINGS:
            many = cs.full(t, 16, 1.0 / math.sqrt(128.0))
            for r in range(16):
                a0 = torch.zeros(64, K, dtype=torch.float16)
                a0[0] = cs.a[0][r]
                a1 = torch.zeros(64, K, dtype=torch.float16)
                a1[0] = cs.a[1][r]
                one = cs.full(t, 1, 1.0 / math.sqrt(128.0), a0=a0.cuda(),
                              a1=a1.cuda() if parts == 2 else None)
                if not torch.equal(one[0], many[r]):
                    print("KB %d P %d %s row %d: M=16 row != M=1 call   <-- FAIL" %
                          (KB, parts, "/".join(map(str, t)), r))
                    bad += 1
                    break

# Stale tickets: a split call with its counters left non-zero finishes nothing; after zero_tickets it
# is bit-exact again.
cs = Case(4, 1, 99)
t = (1, 4, 1, 1, 2, 4, 1)
good = cs.full(t, 8, 1.0 / math.sqrt(128.0))
cs.tickets.fill_(0x40404040)
stale = cs.full(t, 8, 1.0 / math.sqrt(128.0))
r4d.gemm_trellis_nt_m64_zero_tickets(cs.tickets.data_ptr(), cs.tickets.numel() * 4, stream())
again = cs.full(t, 8, 1.0 / math.sqrt(128.0))
if torch.equal(stale, good) or not torch.equal(again, good) or not cs.tickets_zero():
    print("stale tickets: result unchanged, or not bit-exact after zero_tickets   <-- FAIL")
    bad += 1
else:
    print("stale tickets: %d of %d outputs wrong; after zero_tickets bit-exact" %
          (int((stale != good).sum().item()), good.numel()))

cbuf = torch.zeros(64, N, dtype=torch.int16, device="cuda")
w, sv, c, ws, tk = cs.w.data_ptr(), cs.svh_d.data_ptr(), cbuf.data_ptr(), cs.ws.data_ptr(), cs.tickets.data_ptr()
a0 = cs.a_d[0].data_ptr()
bad += expect_throw("KB 3", lambda: r4d.gemm_trellis_nt_m64(
    a0, 0, N, w, sv, c, ws, tk, 1, K, N, 3, 4, 2, 1, 1, 1, 2, 1, 1.0, stream()))
bad += expect_throw("NP 4 * U 4 > 8", lambda: r4d.gemm_trellis_nt_m64(
    a0, 0, N, w, sv, c, ws, tk, 1, K, N, 4, 1, 2, 1, 4, 1, 4, 1, 1.0, stream()))
bad += expect_throw("split group (Wc 32) without ws / tickets", lambda: r4d.gemm_trellis_nt_m64(
    a0, 0, N, w, sv, c, 0, 0, 1, K, N, 4, 1, 2, 1, 1, 1, 2, 1, 1.0, stream()))
bad += expect_throw("n_split 192 (not a multiple of 128)", lambda: r4d.gemm_trellis_nt_m64(
    a0, a0, 192, w, sv, c, ws, tk, 1, K, N, 4, 4, 2, 1, 1, 1, 2, 1, 1.0, stream()))

print("calls:", calls)
if calls < 120:
    print("too few calls ran   <-- FAIL")
    bad += 1
print("FAILURES:", bad)
sys.exit(1 if bad else 0)
