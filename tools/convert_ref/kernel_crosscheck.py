"""Task item 3's second gate: run the ACTUAL r4d kernels (through libr4d's r4d.pyd) on
r4dx-convert's packed layouts and compare against a torch matmul on the DEQUANTIZED weights
(rel err <= 2e-2). This validates kernel<->packer agreement -- does the kernel's own dequant math,
fed these bytes, reconstruct the value the packer intended -- not the quantizer's fidelity to the
original weights (that is tests/convert/test_quantize_roundtrip.cpp's job).

Weights are packed here with the SAME Python references (w4_ref.py, mxfp4_ref.py) that
tools/convert_ref/selftest_compare.py already proved byte-identical to r4dx-convert's C++ output,
so running the kernel against the Python-packed bytes is equivalent to running it against
r4dx-convert's own output.

GPU RULE: HIP device 1 only, one GPU process at a time (see r4dx/README.md). This script sets
HIP_VISIBLE_DEVICES=1 itself if unset, but run it alone, never alongside another GPU test.

Usage (reference venv, r4d.pyd on PYTHONPATH):
  $env:HIP_VISIBLE_DEVICES = '1'
  C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe kernel_crosscheck.py
"""
import os
import pathlib
import sys

os.environ.setdefault("HIP_VISIBLE_DEVICES", "1")

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, r"C:\Users\user\dev\libr4d\build-win")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import r4d  # noqa: E402

import mxfp4_ref  # noqa: E402
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


def check_w4a8(n: int, k: int, m: int) -> bool:
    w = make_weight(n, k, 2)
    q, scale = w4_ref.quantize_symmetric_pinned8(w.numpy())
    wq = w4_ref.pack_nibbles(q, n, k)
    ws = w4_ref.pack_w4a8_scales(scale, n, k)

    gpr = k // w4_ref.GROUP
    dq = np.empty((n, k), dtype=np.float32)
    for g in range(gpr):
        sl = slice(g * w4_ref.GROUP, (g + 1) * w4_ref.GROUP)
        dq[:, sl] = scale[:, g:g + 1] * (q[:, sl].astype(np.float32) - 8.0)
    ref_w = torch.from_numpy(dq)

    a_bf16 = (torch.randn(m, k) * 0.3).to(torch.bfloat16).cuda()
    a_i8 = torch.empty(m, k, dtype=torch.int8, device="cuda")
    a_scale = torch.empty(m, dtype=torch.float32, device="cuda")
    r4d.quant_act_i8(a_bf16.data_ptr(), a_i8.data_ptr(), a_scale.data_ptr(), m, k,
                      torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    wq_t = torch.from_numpy(wq).cuda()
    ws_t = torch.from_numpy(ws).cuda()
    c = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
    WV, SK, MB, NPW, NT = 1, 2, 1, 1, 0
    r4d.gemm_w4a8_nt_m64(a_i8.data_ptr(), a_scale.data_ptr(), wq_t.data_ptr(), ws_t.data_ptr(),
                          c.data_ptr(), m, k, n, WV, SK, MB, NPW, NT,
                          torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    # NOTE: r4d_quant_act_i8's output buffer is NOT plain row-major -- r4d_gemm_w4a8_nt_m64.hip's
    # comment ("THE ACTIVATION IS PRE-SHUFFLED TOO") documents that it reorders each 16-wide k step
    # into [half 0's eight][half 1's eight] bytes, exactly the layout the gemm kernel's fragment
    # loader expects. Indexing a_i8[m, k] as if it were logical column k (as an earlier version of
    # this script did) silently reads the wrong element and produces a ~100% "relative error" that
    # has nothing to do with the weight path under test. The activation side of this test is r4d's
    # own (already-trusted) quant_act_i8 kernel, not something r4dx-convert produces, so the fair
    # reference for isolating the WEIGHT packing (this script's actual subject) is the original,
    # unquantized activation against the dequantized weight -- int8 activation quantization error is
    # a separate, much smaller effect (<1% typically) that would otherwise be folded into the same
    # number as any weight-packing bug.
    ref = a_bf16.float().cpu() @ ref_w.T
    got = c.float().cpu()
    ok = rel_err(got, ref) < 2e-2
    print(f"w4a8   N={n} K={k} M={m}: rel_err={rel_err(got, ref):.4e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_mxfp4(n: int, k: int, m: int) -> bool:
    w = make_weight(n, k, 3)
    packed, escale, wref = mxfp4_ref.quantize(w.numpy())
    wq = mxfp4_ref.permute_wq(packed, n, k)
    ws = mxfp4_ref.pack_ws(escale, n, k)

    gpr = k // mxfp4_ref.GROUP
    dq = np.empty((n, k), dtype=np.float32)
    for row in range(n):
        for g in range(gpr):
            sl = slice(g * mxfp4_ref.GROUP, (g + 1) * mxfp4_ref.GROUP)
            scale = 2.0 ** (int(escale[row, g]) - 127)
            for k2 in range(sl.start, sl.stop):
                byte = packed[row, k2 // 2]
                code = (byte & 0xF) if k2 % 2 == 0 else ((byte >> 4) & 0xF)
                mag = mxfp4_ref.MAGNITUDE[code & 0x7]
                sign = -1.0 if (code & 0x8) else 1.0
                dq[row, k2] = sign * mag * scale
    ref_w = torch.from_numpy(dq)

    a_f32 = (torch.randn(m, k) * 0.3)
    a_scale = a_f32.abs().amax(dim=-1).clamp_min(1e-12) / 448.0  # e4m3 max magnitude
    a_e4m3 = (a_f32 / a_scale[:, None]).to(torch.float8_e4m3fn).cuda()
    a_scale_cuda = a_scale.to(torch.float32).cuda()

    wq_t = torch.from_numpy(wq).cuda()
    ws_t = torch.from_numpy(ws).cuda()
    wref_t = torch.from_numpy(wref).cuda()
    c = torch.zeros(m, n, dtype=torch.bfloat16, device="cuda")
    WV, SK, MB, NPW = 1, 2, 1, 1
    r4d.gemm_mxfp4a8_nt_m64(a_e4m3.data_ptr(), a_scale_cuda.data_ptr(), wq_t.data_ptr(),
                             ws_t.data_ptr(), wref_t.data_ptr(), c.data_ptr(), m, k, n, WV, SK,
                             MB, NPW, torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    a_ref = a_e4m3.float().cpu() * a_scale[:, None]
    ref = a_ref @ ref_w.T
    got = c.float().cpu()
    ok = rel_err(got, ref) < 2e-2
    print(f"mxfp4  N={n} K={k} M={m}: rel_err={rel_err(got, ref):.4e} -> {'PASS' if ok else 'FAIL'}")
    return ok


def main() -> int:
    n, k, m = 64, 256, 8
    ok = True
    ok &= check_w4a16(n, k, m)
    ok &= check_w4a8(n, k, m)
    ok &= check_mxfp4(n, k, m)
    print("ALL LAYOUTS PASS" if ok else "SOME LAYOUTS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
