"""tools/quant2/qwen35moe/disk_budget.py -- the disk and GPU-time budget of the qwen3_5_moe quantization chain (read-only; pipeline.ps1
-Stage budget, and the head of every plan).

    python tools\\quant2\\qwen35moe\\disk_budget.py [--root E:\\models\\r4dx\\q35moe] [--model-dir ...Ornith-1.5-122B] [--layers 0..47]
        [--ks 3,4] [--cap-gib 26] [--tokens 1000000 | --tokens-file tokens_calib.json]

Everything is derived from the config (shapes) and the byte model of alloc_q35.py; the GPU-time figures are ESTIMATES scaled from the
one measured layer (below) and say so: they move with the GPU (device 0 also drives the display) and with the lock holders.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference"))

from qwen35moe import alloc_q35 as A  # noqa: E402
from qwen35moe.arch import Qwen35MoeArch  # noqa: E402

GIB = 2**30
TILES_PER_S = 12_800.0       # END-TO-END rate, MEASURED 2026-10-21 on layer 0 of the real 4-layer slice (HIP device 0, the display GPU, cohort 32,
                             # KB 3): 13.98 M encoder tiles in 18.3 min with one 840 s slice restart (a 32-expert chunk: 1.70 M tiles in ~125 s;
                             # the encoder alone ran 19 k tiles/s). The qwen4_exp runs on device 1 had 23.7 k (KB 3) - 28 k (KB 4) tiles/s.
WORK_PER_TILE = 1.44         # encoder tiles (LDLQ pass + global-scale search) per raw tile: 53 k per expert vs 36.9 k raw
HESS_S_PER_MTOK = 380.0      # Hessian pass of one layer, MEASURED: 105,694 tokens in 40 s forward + 5 s finalize (device 0, peak VRAM 18.3 GiB, RSS 5.4 GiB)

def parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("..")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return sorted(set(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=r"E:\models\r4dx\q35moe")
    ap.add_argument("--model-dir", default=r"E:\models\r4dx\Ornith-1.5-122B")
    ap.add_argument("--layers", default="0..47")
    ap.add_argument("--ks", default="3,4")
    ap.add_argument("--cap-gib", type=float, default=26.0)
    ap.add_argument("--tokens", type=int, default=1_000_000)
    ap.add_argument("--tokens-file", default=None)
    a = ap.parse_args()

    arch = Qwen35MoeArch.from_config(Path(a.model_dir) / "config.json")
    layers = [l for l in parse_layers(a.layers) if l < arch.n_layers]
    ks = [int(k) for k in a.ks.split(",") if k]
    T = a.tokens
    src = "--tokens"
    if a.tokens_file and Path(a.tokens_file).exists():
        meta = Path(a.tokens_file).with_suffix(".meta.json")
        if meta.exists():
            T = int(json.loads(meta.read_text(encoding="utf-8"))["tokens"])
            src = f"{meta.name}"
    H, I, E = arch.hidden, arch.moe_inter, arch.n_experts
    boundary = T * H * 2
    state = E * (H * H + I * I) * 4
    hess_packed = E * (H * (H + 1) // 2 + I * (I + 1) // 2) * 4
    dense_taps = (H * (H + 1) // 2 + arch.gdn_z_rows * (arch.gdn_z_rows + 1) // 2) * 4
    probe = E * 256 * H * 2
    step = 2 * boundary + state + hess_packed + dense_taps + probe
    root = Path(a.root)
    probe_dir = root if root.exists() else next((p for p in root.parents if p.exists()), Path("."))
    free = shutil.disk_usage(probe_dir).free

    print(f"[budget] corpus {T:,} tokens ({src}); {len(layers)} layer(s) of {arch.n_layers}; root {root} ({free / GIB:.0f} GiB free on its drive)")
    print("[budget] layer step scratch (hessian stage, deleted after the layer):")
    for name, b in (("2 residual boundaries", 2 * boundary), ("accumulator state (gate/up + down)", state),
                    ("packed Hessians (.hess)", hess_packed + dense_taps), ("probe rows", probe)):
        print(f"           {name:<38} {b / GIB:7.2f} GiB")
    print(f"           {'peak per layer step':<38} {step / GIB:7.2f} GiB   (+ the oracle outputs below)")

    tot_oracle = 0
    tiles_raw = 0
    print("[budget] oracle outputs kept (words + sign vectors, all layers of -Layers):")
    mdl = A.Model(arch)
    for k in ks:
        b = sum(mdl.cbytes(i, kd, float(k)) for i in layers for kd in A.KINDS)
        tot_oracle += b
        print(f"           oracle-k{k:<30} {b / GIB:7.2f} GiB")
    for i in layers:
        for kd in A.KINDS:
            tiles_raw += sum(k_ * n_ // 256 for k_, n_ in A.class_shapes(arch, i, kd))
    embed = arch.vocab * H * 2
    out_container = 2 * a.cap_gib * GIB + embed
    need_end = tot_oracle + out_container
    print(f"[budget] container (upper bound: both cards at the cap + bf16 embedding {embed / GIB:.2f} GiB): {out_container / GIB:.1f} GiB")
    print(f"[budget] at the end of the chain: oracle dirs {tot_oracle / GIB:.1f} + container {out_container / GIB:.1f} = {need_end / GIB:.1f} GiB"
          f"; peak during layer {layers[-1] if layers else '-'}: {(tot_oracle + step) / GIB:.1f} GiB")
    ok = max(tot_oracle + step, need_end) <= free
    print(f"[budget] free {free / GIB:.0f} GiB: {'enough' if ok else 'NOT ENOUGH -- put -Root on a bigger drive (the Ornith checkpoint itself stays on E:)'}")

    secs = tiles_raw * WORK_PER_TILE / TILES_PER_S
    print(f"[budget] GPU time (ESTIMATE, one GPU, one process under the lock):")
    print(f"           oracle encoder: {tiles_raw / 1e6:.0f} M raw tiles per K x {WORK_PER_TILE} / {TILES_PER_S:.0f} tiles/s"
          f" = {secs / 3600:.1f} h per K, {len(ks) * secs / 3600:.1f} h for K {','.join(map(str, ks))}")
    hs = len(layers) * (T / 1e6 * HESS_S_PER_MTOK + 10)
    print(f"           hessian passes: {len(layers)} layers x ({T:,} tokens at {HESS_S_PER_MTOK:.0f} s per 1M tokens, measured on a 105k-token pilot)"
          f" = {hs / 3600:.1f} h; total ~{(hs + len(ks) * secs) / 3600:.0f} h")
    print("           slices of --BudgetSeconds (840 s) release the GPU lock in between, so other jobs interleave")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
