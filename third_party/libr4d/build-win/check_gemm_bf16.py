"""Ad-hoc numeric checks for the Windows build of r4d.pyd, against torch reference ops.

Run with the vLLM_for_AMD .venv-rocm10 python (torch + r4d both importable), e.g.:

    $env:HIP_VISIBLE_DEVICES = '1'
    $env:PYTHONPATH = "$env:USERPROFILE\\dev\\libr4d\\build-win"
    & "$env:USERPROFILE\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe" build-win\\check_gemm_bf16.py

Checks:
  1. gemm_bf16_nt_m64: C[M,N] = A[M,K] @ W[N,K]^T, bf16 in/out, fp32 accumulate, against
     torch.matmul.
  2. gdn_gated_rmsnorm_h128_bf16: out = rms(x) * w * act(z), against a direct torch reference
     built from the doc comment in r4d.h / r4d_gdn_gated_rmsnorm_h128_bf16.hip.
"""
import torch
import r4d


def check_gemm_bf16_nt_m64():
    torch.manual_seed(0)
    M, N, K = 8, 1024, 2048
    WV, SK, MB = 4, 4, 1
    assert K % (SK * 16) == 0
    assert WV * SK * 32 <= 1024

    A = (torch.randn(M, K, device="cuda", dtype=torch.bfloat16) * 0.2)
    W = (torch.randn(N, K, device="cuda", dtype=torch.bfloat16) * 0.2)
    C = torch.zeros(M, N, device="cuda", dtype=torch.bfloat16)

    r4d.gemm_bf16_nt_m64(A.data_ptr(), W.data_ptr(), C.data_ptr(), M, K, N, WV, SK, MB,
                          torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    ref = (A.float() @ W.float().T)
    got = C.float()
    abs_err = (got - ref).abs().max().item()
    rel_err = ((got - ref).norm() / ref.norm()).item()
    print(f"gemm_bf16_nt_m64  M={M} N={N} K={K} WV={WV} SK={SK} MB={MB}: "
          f"max_abs_err={abs_err:.4e} rel_err={rel_err:.4e}")
    return rel_err < 2e-2


def check_gdn_gated_rmsnorm_h128_bf16():
    torch.manual_seed(0)
    rows, width = 256, 128
    eps = 1e-6
    act = 0  # silu/swish

    x = (torch.randn(rows, width, device="cuda", dtype=torch.bfloat16) * 0.5)
    z = (torch.randn(rows, width, device="cuda", dtype=torch.bfloat16) * 0.5)
    w = (torch.rand(width, device="cuda", dtype=torch.float32) * 0.5 + 0.5)
    o = torch.zeros(rows, width, device="cuda", dtype=torch.bfloat16)

    r4d.gdn_gated_rmsnorm_h128_bf16(x.data_ptr(), z.data_ptr(), w.data_ptr(), o.data_ptr(),
                                     rows, width, width, width, width, eps, act,
                                     torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()

    xf = x.float()
    zf = z.float()
    rms = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    silu = zf * torch.sigmoid(zf)
    ref = xf * rms * w.float() * silu
    got = o.float()
    abs_err = (got - ref).abs().max().item()
    rel_err = ((got - ref).norm() / ref.norm()).item()
    print(f"gdn_gated_rmsnorm_h128_bf16  rows={rows} width={width}: "
          f"max_abs_err={abs_err:.4e} rel_err={rel_err:.4e}")
    return rel_err < 2e-2


if __name__ == "__main__":
    ok = True
    ok &= check_gemm_bf16_nt_m64()
    ok &= check_gdn_gated_rmsnorm_h128_bf16()
    print("ALL OK" if ok else "FAILURES PRESENT")
    raise SystemExit(0 if ok else 1)
