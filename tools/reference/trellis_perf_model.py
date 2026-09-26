"""tools/reference/trellis_perf_model.py -- the per-SIMD cycle-budget model of docs/trellis-kernel.md
4.6 (CPU only, no GPU, no measurements of its own).

Byte rooflines only (docs/r9700.md rule 2). Today's w4a16 linears (q2ab_hv2_q3 at production groups,
docs/quant2.md:1190-1192) are taken as memory-bound at an effective bandwidth BW, which the trellis
kernel also gets when it is memory-bound. One 256-weight fragment on one SIMD (each lane owns 8 of
its weights) then costs the trellis kernel

    max(need, mem) + beta * min(need, mem)     cycles

  need  VALU decode (31 cycles at KB = 4, 3.875 op/w; 37.5 at KB = 5) + 8 * overhead op/w
        (+ 16 WMMA cycles when WMMA does not overlap VALU)
  mem   fragment bytes / (BW / 128 SIMDs) * SCLK
  beta  the exposed fraction of the shorter pipe (0 = perfect hiding)

Sections: (1) need vs memory cycles per fragment; (2) plain-decode change per (SCLK, overlap,
overhead, beta) -- 4.6's table; (3) the beta at which trellis meets A3 exactly (-0.5%); (4) trellis /
today byte ratios per shape family; (5) the M1/M2 budget: the weighted effective-bandwidth ratio M1's
raw GEMMs must reach for an M2 overhead budget X (4.6 point 3).

  & $py tools\\reference\\trellis_perf_model.py [--kb 4|5]
"""

from __future__ import annotations

import argparse

SIMDS = 128
TOK_S = 35.9                                   # plain decode today (q2ab_hv2_q3)
STEP_MS = 1000 / TOK_S
A3_MARGIN = 0.005                              # A3: >= today's median - 0.5%
WMMA_CYCLES = 16.0                             # one 16x16x16 f16 WMMA per fragment per lane (M <= 16)
DECODE_VALU = {4: 31.0, 5: 37.5}               # VALU cycles per fragment per lane (4.2, 4.4)
OVERHEADS = (0.25, 0.69)                       # op/w: scalar addressing / the probe's measured loop
SCLKS = (2.92e9, 2.35e9, 2.0e9)                # boost, sagged, low
BWS = (550e9, 604e9)                           # C8 live gate_up / C5
# (family, N, K, count per token, today's bpw): q2ab_hv2_q3's production groups
# (g32 = 5.0, g64 = 4.5, g128 = 4.25 bpw; bf16 attn.k/v in layers 32-63)
PROD = [("qkv", 10240, 5120, 48, 4.5),
        ("z", 6144, 5120, 24, 5.0), ("z", 6144, 5120, 24, 4.5),
        ("out", 5120, 6144, 24, 5.0), ("out", 5120, 6144, 24, 4.5),
        ("qg", 12288, 5120, 16, 4.5),
        ("k", 1024, 5120, 8, 4.5), ("k", 1024, 5120, 8, 16.0),
        ("v", 1024, 5120, 8, 4.5), ("v", 1024, 5120, 8, 16.0),
        ("o", 5120, 6144, 16, 5.0),
        ("gate_up", 34816, 5120, 32, 4.5), ("gate_up", 34816, 5120, 32, 4.25),
        ("down", 5120, 17408, 32, 4.25), ("down", 5120, 17408, 32, 5.0)]


def trellis_bpw(N: int, K: int, kb: int) -> float:
    """KB bits plus the fp16 suh [K] and svh [N]."""
    return kb + 16.0 * (N + K) / (N * K)


def need_cycles(kb: int, overlap: bool, ovh: float) -> float:
    need = DECODE_VALU[kb] + 8 * ovh
    return need if overlap else need + WMMA_CYCLES


def mem_cycles(frag_bytes: float, bw: float, sclk: float) -> float:
    return frag_bytes / (bw / SIMDS) * sclk


def run(bw: float, sclk: float, overlap: bool, ovh: float, beta: float, kb: int = 4) -> tuple[float, float, float]:
    """(today's linear ms, trellis linear ms, delta ms) per token."""
    need = need_cycles(kb, overlap, ovh)
    today = trel = 0.0
    for _, N, K, c, bpw in PROD:
        w = N * K * c
        today += w * bpw / 8 / bw
        tb = trellis_bpw(N, K, kb)
        mem = mem_cycles(256 * tb / 8, bw, sclk)
        f = (max(need, mem) + beta * min(need, mem)) / mem
        trel += w * tb / 8 / bw * f
    return today * 1e3, trel * 1e3, (trel - today) * 1e3


def plain_change_pct(delta_ms: float) -> float:
    return (STEP_MS / (STEP_MS + delta_ms) - 1) * 100


def breakeven_beta(bw: float, sclk: float, overlap: bool, ovh: float, kb: int) -> float:
    """The beta at which the step is exactly A3's margin slower (bisection)."""
    lim = A3_MARGIN * STEP_MS
    lo, hi = 0.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if run(bw, sclk, overlap, ovh, mid, kb)[2] < lim else (lo, mid)
    return lo


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb", type=int, default=4, choices=sorted(DECODE_VALU), help="uniform trellis rate")
    args = ap.parse_args()
    kb = args.kb
    lim = A3_MARGIN * STEP_MS
    print(f"KB = {kb}; step {STEP_MS:.2f} ms ({TOK_S} tok/s); A3 allows +{lim:.3f} ms per token")

    frag = 256 * kb / 8 + 0.1                  # the suh/svh share is about 0.1 B per fragment
    print(f"\n(1) cycles per 256-weight fragment per SIMD: decode {DECODE_VALU[kb]:.1f} VALU, "
          f"WMMA {WMMA_CYCLES:.0f}, memory at {frag:.1f} B")
    print("  BW GB/s  SCLK GHz  mem cyc  need/mem (no overlap, ovh 0.25)  (no overlap, ovh 0.69)")
    for bw in BWS:
        for sclk in SCLKS:
            m = mem_cycles(frag, bw, sclk)
            print(f"  {bw / 1e9:6.0f}  {sclk / 1e9:8.2f}  {m:7.1f}  {need_cycles(kb, False, 0.25) / m:31.2f}"
                  f"  {need_cycles(kb, False, 0.69) / m:22.2f}")

    print("\n(2) plain decode, trellis vs today (linears only; + = faster)")
    for bw in BWS:
        print(f"  BW_eff {bw / 1e9:.0f} GB/s")
        print("    sclk  overlap  ovh   beta  today_ms  trellis_ms  delta_ms  plain_tok/s_change")
        for sclk in SCLKS:
            for overlap in (True, False):
                for ovh in OVERHEADS:
                    for beta in (0.0, 0.2):
                        t, r, d = run(bw, sclk, overlap, ovh, beta, kb)
                        print(f"    {sclk / 1e9:4.2f}  {str(overlap):6s}  {ovh:4.2f}  {beta:3.1f}  {t:8.2f}  "
                              f"{r:10.2f}  {d:+8.2f}  {plain_change_pct(d):+6.1f}%")

    print("\n(3) break-even beta (trellis exactly A3's 0.5% slower)")
    for bw in BWS:
        for sclk in SCLKS:
            row = [f"overlap={int(o)} ovh={v}: {breakeven_beta(bw, sclk, o, v, kb):.3f}"
                   for o in (True, False) for v in OVERHEADS]
            print(f"  {bw / 1e9:.0f} GB/s {sclk / 1e9:.2f} GHz  " + "; ".join(row))

    print("\n(4) trellis / today bytes")
    fam: dict[str, list[float]] = {}
    for name, N, K, c, bpw in PROD:
        a = fam.setdefault(name, [0.0, 0.0])
        a[0] += N * K * c * bpw
        a[1] += N * K * c * trellis_bpw(N, K, kb)
        print(f"  {name:8s} today {bpw:5.2f} bpw x{c:2d}: {trellis_bpw(N, K, kb) / bpw:5.3f}")
    for name, (t, r) in fam.items():
        print(f"  {name:8s} family {r / t:5.3f}")
    T = sum(v[0] for v in fam.values())
    R = sum(v[1] for v in fam.values())
    print(f"  all      {R / T:5.3f}")

    print("\n(5) M1/M2 budget: A3 needs X - S <= "
          f"{lim:.2f} ms (S = M1's raw-GEMM saving, X = M2's added work)")
    for bw in BWS:
        t_ms = T / 8 / bw * 1e3
        for X in (0.0, 0.30, 0.60):
            r = (R / T) * t_ms / (t_ms + lim - X)
            print(f"  BW {bw / 1e9:.0f}: X <= {X:.2f} ms -> S >= {X - lim:+.2f} ms -> weighted effective-bandwidth "
                  f"ratio >= {r:.3f} of w4a16's")
        for r in (0.86, 0.85):
            d = (R / T) / r * t_ms - t_ms
            print(f"  BW {bw / 1e9:.0f}: ratio {r} -> raw delta {d:+.2f} ms = {plain_change_pct(d):+.1f}% before overheads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
