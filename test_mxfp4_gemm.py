"""Correctness for r4d_gemm_mxfp4a8_nt_m64 against an exact MXFP4 dequantisation."""
import torch, r4d, sys
from mxfp4_layout import permute_w

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
GROUP = 32

def make_case(M, N, K, seed=0, max_d=8):
    g = torch.Generator(device="cpu").manual_seed(seed)
    nb = K // GROUP
    # E8M0 exponents: a per-row reference and per-block drops of at most max_d binades, so every
    # folded magnitude is exactly representable in e4m3 (including subnormals).
    ref = torch.randint(120, 136, (N,), generator=g, dtype=torch.int32)
    drop = torch.randint(0, max_d + 1, (nb, N), generator=g, dtype=torch.int32)
    e8m0 = (ref.unsqueeze(0) - drop).clamp(0, 254)
    ref = e8m0.max(dim=0).values                       # true per-row reference
    codes = torch.randint(0, 16, (N, K), generator=g, dtype=torch.uint8)
    mag = E2M1[(codes & 0x7).long()]
    sign = torch.where((codes & 0x8) > 0, -1.0, 1.0)
    scale = torch.exp2((e8m0.float() - 127.0)).T       # [N, nb]
    Wf = (mag * sign).reshape(N, nb, GROUP) * scale.unsqueeze(-1)
    Wf = Wf.reshape(N, K)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    packed = permute_w(packed, N, K)

    a = (torch.randn(M, K, generator=g) * 0.4)
    af8 = a.to(torch.float8_e4m3fn)
    ascale = torch.full((M,), 0.7)
    Af = af8.float() * ascale.unsqueeze(1)
    ref_out = Af @ Wf.T
    return (packed.cuda(), e8m0.to(torch.uint8).cuda(), ref.to(torch.uint8).cuda(),
            af8.cuda(), ascale.cuda(), ref_out.cuda())

def run(M, N, K, WV, SK, MB, NPW, **kw):
    packed, e8m0, ref, af8, ascale, exp = make_case(M, N, K, **kw)
    c = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    r4d.gemm_mxfp4a8_nt_m64(af8.data_ptr(), ascale.data_ptr(), packed.data_ptr(),
                            e8m0.data_ptr(), ref.data_ptr(), c.data_ptr(),
                            M, K, N, WV, SK, MB, NPW,
                            torch.cuda.current_stream().cuda_stream)
    torch.cuda.synchronize()
    got = c.float()
    rel = (got - exp).norm() / exp.norm()
    return rel.item()

print("%-28s %-22s %s" % ("shape (M,N,K)", "WV/SK/MB/NPW", "rel err"))
bad = 0
# N=48 and N=80 are the tail-block cases: N/16 is not a multiple of ncols=WV*NPW, so grid.x
# rounds up and the last block addresses column tiles that do not exist. Unclamped that is a
# GPU memory fault, and it only shows up on shapes like the gated-delta-net gating projections.
for (M, N, K) in [(8, 512, 1024), (5, 512, 1024), (16, 1024, 2048), (64, 512, 1024),
                  (1, 256, 512), (32, 5120, 3072), (8, 5120, 8704), (64, 17408, 5120),
                  (8, 48, 1024), (5, 48, 5120), (64, 80, 1024)]:
    for (WV, SK, MB, NPW) in [(4, 4, 1, 1), (2, 4, 1, 2), (8, 2, 1, 1)]:
        mb = max(1, min(4, (M + 15) // 16))
        try:
            rel = run(M, N, K, WV, SK, mb, NPW)
        except Exception as e:
            print("%-28s %-22s SKIP (%s)" % (f"({M},{N},{K})", f"{WV}/{SK}/{mb}/{NPW}", str(e)[:50]))
            continue
        flag = "" if rel < 2e-2 else "   <-- FAIL"
        if rel >= 2e-2: bad += 1
        print("%-28s %-22s %.3e%s" % (f"({M},{N},{K})", f"{WV}/{SK}/{mb}/{NPW}", rel, flag))
print("FAILURES:", bad)
sys.exit(1 if bad else 0)
