"""Time r4d_gemm_mxfp4a8_nt_m64 on the Qwen3.8-27B TP2 decode shapes."""
import torch, r4d, time, itertools, os
from mxfp4_layout import permute_w

GROUP = 32
SHAPES = [("mlp.gate_up", 17408, 5120), ("mlp.down", 5120, 8704), ("gdn.in_qkv", 5120, 5120),
          ("gdn.in_z", 3072, 5120), ("gdn.out", 5120, 3072), ("attn.q", 6144, 5120),
          ("attn.o", 5120, 3072)]
M = int(os.environ.get("BENCH_M", "8"))
PEAK = 635.0

def alloc(N, K):
    g = torch.Generator(device="cpu").manual_seed(0)
    packed = permute_w(torch.randint(0, 256, (N, K // 2), generator=g,
                                     dtype=torch.uint8), N, K).cuda()
    e8m0 = torch.randint(118, 130, (K // GROUP, N), generator=g, dtype=torch.uint8).cuda()
    ref = e8m0.max(dim=0).values.contiguous().cuda()
    a = (torch.randn(M, K, generator=g) * 0.4).to(torch.float8_e4m3fn).cuda()
    asc = torch.full((M,), 0.7, device="cuda")
    c = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    return packed, e8m0, ref, a, asc, c

def bench(fn, n=60):
    for _ in range(10): fn()
    torch.cuda.synchronize(); best = 1e9
    for _ in range(n):
        t = time.perf_counter(); fn(); torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t)
    return best * 1e6

print(f"M={M}  (bytes = N*K/2 packed + N*K/32 scales; roofline {PEAK} GB/s)")
print("%-14s %7s %7s  %-16s %9s %8s" % ("shape", "N", "K", "best WV/SK/MB/NPW", "us", "%roof"))
grand = 0.0
for name, N, K in SHAPES:
    packed, e8m0, ref, a, asc, c = alloc(N, K)
    mb = max(1, min(4, (M + 15) // 16))
    best = None
    for WV, SK, NPW in itertools.product([1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4]):
        if WV * SK * 32 > 1024 or (WV * NPW) * SK * 256 * 4 > 64 * 1024: continue
        if K % (SK * GROUP) or (N // 16) % 1: continue
        def call():
            r4d.gemm_mxfp4a8_nt_m64(a.data_ptr(), asc.data_ptr(), packed.data_ptr(),
                                    e8m0.data_ptr(), ref.data_ptr(), c.data_ptr(),
                                    M, K, N, WV, SK, mb, NPW,
                                    torch.cuda.current_stream().cuda_stream)
        try:
            us = bench(call, 25)
        except Exception:
            continue
        if best is None or us < best[0]: best = (us, WV, SK, NPW)
    mb_bytes = (N * K / 2 + N * K / GROUP) / 1e6
    roof = mb_bytes / PEAK * 1e3
    us, WV, SK, NPW = best
    grand += us
    print("%-14s %7d %7d  %-16s %9.1f %7.0f%%" % (name, N, K, f"{WV}/{SK}/{mb}/{NPW}", us, 100*roof/us))
print(f"sum over the seven shapes: {grand:.1f} us")
