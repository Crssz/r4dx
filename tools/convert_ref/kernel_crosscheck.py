"""Task item 3's second gate: run the ACTUAL r4d kernels (through libr4d's r4d.pyd) on
r4dx-convert's packed layouts and compare against a torch matmul on the DEQUANTIZED weights
(rel err <= 2e-2). This validates kernel<->packer agreement -- does the kernel's own dequant math,
fed these bytes, reconstruct the value the packer intended -- not the quantizer's fidelity to the
original weights (that is tests/convert/test_quantize_roundtrip.cpp's job).

Weights are packed here with the SAME Python reference (w4_ref.py) that
tools/convert_ref/selftest_compare.py already proved byte-identical to r4dx-convert's C++ output,
so running the kernel against the Python-packed bytes is equivalent to running it against
r4dx-convert's own output.

GPU RULE: HIP device 1 only, one GPU process at a time (see r4dx/README.md). This script sets
HIP_VISIBLE_DEVICES=1 itself if unset, but run it alone, never alongside another GPU test.

Usage (reference venv, r4d.pyd on PYTHONPATH):
  $env:HIP_VISIBLE_DEVICES = '1'
  python kernel_crosscheck.py
"""
import os
import pathlib
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
# Derived from the environment; override with R4DX_LIBR4D_BUILD.
sys.path.insert(
    0, os.environ.get("R4DX_LIBR4D_BUILD", str(pathlib.Path.home() / "dev" / "libr4d" / "build-win"))
)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import r4d  # noqa: E402

import w4_ref  # noqa: E402


def make_weight(n: int, k: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * 0.3).to(torch.bfloat16).to(torch.float32)
    return w


def rel_err(got: torch.Tensor, ref: torch.Tensor) -> float:
    return ((got - ref).norm() / ref.norm()).item()


def check_w4a16(n: int, k: int, m: int) -> bool:
    w = make_weight(n, k, 1)
    q, scale, zero = w4_ref.quantize_asymmetric(w.numpy())
    wq = w4_ref.pack_nibbles(q, n, k)
    wsz = w4_ref.pack_w4a16_scales(scale, zero, n, k)

    gpr = k // w4_ref.GROUP
    dq = np.empty((n, k), dtype=np.float32)
    for g in range(gpr):
        sl = slice(g * w4_ref.GROUP, (g + 1) * w4_ref.GROUP)
        dq[:, sl] = scale[:, g:g + 1] * (q[:, sl].astype(np.float32) - zero[:, g:g + 1].astype(np.float32))
    ref_w = torch.from_numpy(dq)

    a = (torch.randn(m, k) * 0.3).to(torch.float16).cuda()
    wq_t = torch.from_numpy(wq).cuda()
    wsz_t = torch.from_numpy(wsz).cuda()
    c = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")

    WV, SK, MB, NPW, NT = 1, 2, 1, 1, 0
    r4d.gemm_w4a16_nt_m64(a.data_ptr(), wq_t.data_ptr(), wsz_t.data_ptr(), c.data_ptr(), m, k, n,
                           WV, SK, MB, NPW, NT, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    ref = a.float().cpu() @ ref_w.T
    ok = rel_err(c.float().cpu(), ref) < 2e-2
    print(f"w4a16  N={n} K={k} M={m}: rel_err={rel_err(c.float().cpu(), ref):.4e} -> {'PASS' if ok else 'FAIL'}")
    return ok



def main() -> int:
    n, k, m = 64, 256, 8
    ok = True
    ok &= check_w4a16(n, k, m)
    print("ALL LAYOUTS PASS" if ok else "SOME LAYOUTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
