"""tools/quant2/qwen35moe/gpu_probe.py -- is the GPU responsive and does it have room? (pipeline.ps1 lane B runs it first.)

    set HIP_VISIBLE_DEVICES=0&& set R4DX_ALLOW_GPU0=1&& python gpu_probe.py [--min-free-gib 14] [--max-seconds 30]

Device 0 drives the user's display; sustained dual-GPU load has reset it before. The probe allocates a few MiB, runs 20 fp32 4096^2
matmuls, synchronizes, and checks the free VRAM. Exit 0 and one JSON line when fine; exit 1 and the reason otherwise. It never retries.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "reference"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--min-free-gib", type=float, default=14.0)
    ap.add_argument("--max-seconds", type=float, default=30.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    try:
        import torch
        from common import resolve_device

        dev = resolve_device(a.device)
        if dev.type != "cuda":
            raise RuntimeError("not a GPU device")
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info()
        x = torch.randn(4096, 4096, device=dev)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = x
        for _ in range(20):
            y = torch.tanh(y @ x * 1e-3)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ok_value = bool(torch.isfinite(y).all())
    except Exception as e:  # noqa: BLE001 -- any HIP error: report, never retry
        print(f"[gpu_probe] FAILED: {type(e).__name__}: {e}", flush=True)
        return 1
    res = {"device": name, "free_gib": round(free / 2**30, 2), "total_gib": round(total / 2**30, 2), "matmul_20x_seconds": round(dt, 3),
           "finite": ok_value}
    print("[gpu_probe] " + json.dumps(res), flush=True)
    if free / 2**30 < a.min_free_gib:
        print(f"[gpu_probe] FAILED: only {free / 2**30:.1f} GiB of VRAM free, the oracle needs {a.min_free_gib:g}", flush=True)
        return 1
    if dt > a.max_seconds or not ok_value:
        print(f"[gpu_probe] FAILED: the probe matmuls took {dt:.1f}s (limit {a.max_seconds:g}s) or produced non-finite values", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
