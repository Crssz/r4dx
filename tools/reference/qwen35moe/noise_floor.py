"""The bf16 noise floor of the tiny-q35moe goldens: HF's own bf16 run against its fp32 run (same fp8-rounded KV), per capture
point, as max|err| / rms(ref) and mean|err| / rms(ref) -- the numbers an r4dx comparison against the bf16 golden cannot beat.

    D:\\venvs\\r4dx-qwen4-ref\\Scripts\\python.exe tools\\reference\\qwen35moe\\noise_floor.py [--t 70]

Compares E:\\models\\r4dx\\tiny-q35moe\\golden_fp8\\T{t}.safetensors (bf16 run) with golden_f32_fp8 (fp32 run, rounded to bf16 on
store); both written by tiny_golden.py (--kv-fp8, and --fp32 --kv-fp8).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file

ROOT = Path(r"E:\models\r4dx\tiny-q35moe")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--t", type=int, default=70)
    args = ap.parse_args(argv)
    a = load_file(str(ROOT / "golden_fp8" / f"T{args.t}.safetensors"))
    b = load_file(str(ROOT / "golden_f32_fp8" / f"T{args.t}.safetensors"))
    print(f"T={args.t}: HF bf16 run vs HF fp32 run")
    print(f"  {'tensor':<26}{'max/rms':>9}{'mean/rms':>10}")
    for name in sorted(a):
        if not a[name].is_floating_point() or name not in b or name.endswith(("topk_weights",)):
            continue
        x, y = a[name].float(), b[name].float()
        rms = y.pow(2).mean().sqrt().item()
        if rms == 0:
            continue
        d = (x - y).abs()
        print(f"  {name:<26}{d.max().item() / rms:>9.4f}{d.mean().item() / rms:>10.5f}")
    for name in sorted(a):
        if name.endswith("topk_ids"):
            same = (a[name].sort(dim=-1).values == b[name].sort(dim=-1).values).all(dim=-1)
            print(f"  {name:<26} expert SET differs in {(~same).sum().item()} of {same.numel()} rows")
    return 0


if __name__ == "__main__":
    sys.exit(main())
