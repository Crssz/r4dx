"""CPU test of the alloc_q35.py class-p options (`--p-bf16`, `--p-w8`, `--loss-scale`) on the real model's config.json only (no oracle needed):
synthetic per-class losses stand in for the oracle statistics.

    python tools\\reference\\qwen35moe\\test_alloc_q35.py

* byte model: the bf16 twin of a class is 2 bytes per weight, the w8a16 g64 form k*n + 2*n*(k/64) (Inspect's counts);
* `--p-bf16 all`: class p of every layer is bf16 (sentinel K), the experts still fit both cards under the cap and start at the floor;
* without the options nothing changes (class p stays K4) and the same synthetic losses give the same allocation as before;
* `--loss-scale`: a scale that makes early layers valuable moves the K4 promotions to early layers, the reverse scale to late ones, and
  only the expert classes are scaled.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qwen35moe import alloc_q35 as A  # noqa: E402
from qwen35moe import tiny  # noqa: E402
from qwen35moe.arch import Qwen35MoeArch  # noqa: E402

RESULTS: list[tuple[bool, str]] = []


def check(cond: bool, what: str) -> None:
    RESULTS.append((bool(cond), what))
    print(("ok   " if cond else "FAIL ") + what, flush=True)


def synthetic_stats(arch: Qwen35MoeArch) -> dict:
    """K3 / K4 losses: every class loses 4x going from K3 to K4; layers differ by a smooth factor."""
    st = {}
    for K in (3.0, 4.0):
        st[K] = {(i, kd): {"loss": (1.0 + 0.01 * i) * (1.0 if K == 3.0 else 0.25) * (3.0 if kd == "down" else 2.0 if kd == "gu" else 0.5),
                           "names": []}
                 for i in range(arch.n_layers) for kd in A.KINDS}
    return st


def main() -> int:
    if not tiny.REAL_CONFIG.exists():
        print("SKIP: the real config.json is not on this machine")
        return 0
    arch = Qwen35MoeArch.from_config(tiny.REAL_CONFIG)
    model = A.Model(arch)
    cap = int(26 * A.GIB)
    floor = A.parse_floor(None)

    k_, n_ = arch.hidden, arch.quantized_linears(0)
    shapes = A.class_shapes(arch, 3, "p")
    check(A.class_bytes(arch, 3, "p", A.BF16_K) == sum(2 * k * n for k, n in shapes), "bf16 twin = 2 bytes per weight")
    check(A.class_bytes(arch, 3, "p", A.W8_K) == sum(k * n + 2 * n * (k // 64) for k, n in shapes), "w8a16 g64 = codes + f16 scale per 64")
    check(A.class_bytes(arch, 3, "p", A.BF16_K) > A.class_bytes(arch, 3, "p", A.W8_K) > A.class_bytes(arch, 3, "p", 4.0),
          "bf16 > w8a16 > trellis K4 bytes")

    stats = synthetic_stats(arch)
    ks0, _ = A.allocate(model, stats, cap, floor)
    check(all(ks0[(i, "p")] == 4.0 for i in range(arch.n_layers)), "default: class p stays K4")
    k0, d0, d1 = model.split(ks0, cap)
    check(d0 <= cap and d1 <= cap, f"default allocation fits both cards (k={k0})")

    bf16_all = A.bf16_layers(arch, "all")
    check(len(bf16_all) == arch.n_layers and len(A.bf16_layers(arch, "attn")) == 12 and len(A.bf16_layers(arch, "gdn")) == 36
          and not A.bf16_layers(arch, "none"), "bf16_layers: all / attn (12) / gdn (36) / none")
    st2 = dict(stats)
    st2[A.BF16_K] = {(i, "p"): {"loss": 0.0, "names": []} for i in bf16_all}
    ks1, _ = A.allocate(model, st2, cap, floor, bf16_p=bf16_all)
    check(all(ks1[(i, "p")] == A.BF16_K for i in range(arch.n_layers)), "--p-bf16 all: every class p is the bf16 sentinel")
    k1, e0, e1 = model.split(ks1, cap)
    check(e0 <= cap and e1 <= cap, f"p bf16 + experts fit both cards (k={k1}, {e0 / A.GIB:.2f} / {e1 / A.GIB:.2f} GiB)")
    n4_default = sum(1 for key, K in ks0.items() if key[1] != "p" and K == 4.0)
    n4_bf16 = sum(1 for key, K in ks1.items() if key[1] != "p" and K == 4.0)
    check(n4_bf16 < n4_default, f"bf16 p leaves fewer expert classes at K4 ({n4_bf16} vs {n4_default})")

    ksa, _ = A.allocate(model, {**stats, A.BF16_K: {(i, "p"): {"loss": 0.0, "names": []} for i in A.bf16_layers(arch, "attn")}}, cap, floor,
                        bf16_p=A.bf16_layers(arch, "attn"))
    check(all((ksa[(i, "p")] == A.BF16_K) == (arch.layer_types[i] == "full_attention") for i in range(arch.n_layers)),
          "--p-bf16 attn: only the full-attention layers' p is bf16")

    # --loss-scale: early layers valuable vs late layers valuable
    import copy

    def with_scale(scale):
        s = copy.deepcopy(stats)
        A.scale_losses(s, scale)
        return s

    early = [10.0 if i < 12 else 0.1 for i in range(arch.n_layers)]
    late = [0.1 if i < 36 else 10.0 for i in range(arch.n_layers)]
    s_early = with_scale(early)
    check(s_early[3.0][(5, "gu")]["loss"] == stats[3.0][(5, "gu")]["loss"] * 10.0 and s_early[3.0][(5, "p")]["loss"] == stats[3.0][(5, "p")]["loss"],
          "scale_losses multiplies gu / down only")
    ke, _ = A.allocate(model, s_early, cap, floor)
    kl, _ = A.allocate(model, with_scale(late), cap, floor)

    def k4_classes(ks, lo, hi):
        return sum(1 for (i, kd), K in ks.items() if kd != "p" and K == 4.0 and lo <= i < hi)

    check(k4_classes(ke, 0, 12) > k4_classes(kl, 0, 12), f"scale favouring early layers puts more K4 classes on layers 0-11 ({k4_classes(ke, 0, 12)} vs {k4_classes(kl, 0, 12)})")
    check(k4_classes(kl, 36, 48) > k4_classes(ke, 36, 48), f"scale favouring late layers puts more K4 classes on layers 36-47 ({k4_classes(kl, 36, 48)} vs {k4_classes(ke, 36, 48)})")

    bad = [w for ok, w in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
