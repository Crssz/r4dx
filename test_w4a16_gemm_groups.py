"""Correctness for r4d_gemm_w4a16_nt_m64_g at every instantiated group, against a CPU dequant.

    HIP_VISIBLE_DEVICES=1 PYTHONPATH=. python test_w4a16_gemm_groups.py
    (Windows: build_windows.ps1 puts r4d.pyd in build-win, so PYTHONPATH=build-win.)

The packed weight does not depend on the group -- only the (scale, zero) stride does -- so every
case packs ONE set of 4-bit codes and runs it at groups 32, 64 and 128 with scales drawn per group.
The scales span four binades per group, so a kernel that reads group g's scale for group g+1 (the
failure a wrong Wsz stride, a wrong split offset or a wrong half-block split at G=32 produces) is
off by O(1), not by rounding. Also checked: the ungrouped entry point is bit-identical to _g at the
build default, has_group() answers for exactly the instantiated groups, and the entry rejects an
uninstantiated group and a K whose splits do not start on a 64-K packed block.
"""
import sys
import torch
import r4d

GROUPS = (32, 64, 128)


def nibble_pos(e):
    return 2 * e if e < 4 else 2 * (e - 4) + 1


def pack_wq(q, N, K):
    """[N, K] codes 0..15 -> the kernel's fragment-order dwords (r4d_gemm_w4a16_nt_m64.hip LAYOUT).

    Dword ((t*(K/64) + kb)*32 + lane)*4 + s holds, in nibble nibble_pos(e), the code of row
    16t + (lane&15) at k = (4kb + s)*16 + 8(e>>2) + 4(lane>>4) + (e&3), stored as a 4-bit two's
    complement value (code ^ 8), which the kernel's dequant XORs back.
    """
    nt, kb = N // 16, K // 64
    t = torch.arange(nt).view(nt, 1, 1, 1, 1)
    b = torch.arange(kb).view(1, kb, 1, 1, 1)
    lane = torch.arange(32).view(1, 1, 32, 1, 1)
    s = torch.arange(4).view(1, 1, 1, 4, 1)
    e = torch.arange(8).view(1, 1, 1, 1, 8)
    row = t * 16 + (lane & 15)
    k = (b * 4 + s) * 16 + 8 * (e >> 2) + 4 * (lane >> 4) + (e & 3)
    nib = (q.long()[row, k] ^ 8)                                   # [nt, kb, 32, 4, 8]
    shift = torch.tensor([4 * nibble_pos(i) for i in range(8)]).view(1, 1, 1, 1, 8)
    dw = (nib << shift).sum(-1)                                    # < 2^32
    dw = torch.where(dw >= 2**31, dw - 2**32, dw)
    return dw.to(torch.int32).reshape(-1).contiguous()


def pack_wsz(scale16, zero, N, K, group):
    """[N, K/group] f16 scale and integer zero -> dword (t*(K/group) + g)*16 + r: f16 scale in the
    low half, f16 of -(1024 + zero) in the high half."""
    ng = K // group
    lo = scale16.view(torch.int16).long() & 0xFFFF
    hi = (-(1024.0 + zero.float())).to(torch.float16).view(torch.int16).long() & 0xFFFF
    dw = lo | (hi << 16)                                           # [N, ng]
    dw = dw.view(N // 16, 16, ng).permute(0, 2, 1)                 # [nt, ng, 16]
    dw = torch.where(dw >= 2**31, dw - 2**32, dw)
    return dw.to(torch.int32).reshape(-1).contiguous()


def make_case(M, N, K, group, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    ng = K // group
    q = torch.randint(0, 16, (N, K), generator=g, dtype=torch.uint8)
    zero = torch.randint(0, 16, (N, ng), generator=g)
    mant = 0.5 + torch.rand(N, ng, generator=g)
    binade = torch.randint(-2, 2, (N, ng), generator=g).float()
    scale16 = (mant * torch.exp2(binade) * 0.02).to(torch.float16)
    W = (q.float() - zero.repeat_interleave(group, dim=1).float()) * \
        scale16.float().repeat_interleave(group, dim=1)
    a16 = (torch.randn(M, K, generator=g) * 0.5).to(torch.float16)
    ref = a16.float() @ W.T
    return (a16.cuda(), pack_wq(q, N, K).cuda(), pack_wsz(scale16, zero, N, K, group).cuda(),
            ref.cuda(), q)


def launch(fn, group, a16, wq, wsz, M, K, N, WV, SK, MB, NPW, NT):
    c = torch.zeros(M, N, dtype=torch.bfloat16, device="cuda")
    args = (a16.data_ptr(), wq.data_ptr(), wsz.data_ptr(), c.data_ptr(), M, K, N, WV, SK, MB, NPW,
            NT, torch.cuda.current_stream().cuda_stream)
    if group is None:
        fn(*args)
    else:
        fn(group, *args)
    torch.cuda.synchronize()
    return c


def expect_throw(what, f):
    try:
        f()
    except Exception as e:                                          # noqa: BLE001
        print("%-60s rejected: %s" % (what, str(e)[:70]))
        return 0
    print("%-60s NOT REJECTED   <-- FAIL" % what)
    return 1


bad = 0
print("GEMM_W4_GROUP (build default) =", r4d.GEMM_W4_GROUP)
for grp in (16, 32, 48, 64, 96, 128, 256):
    want = grp in GROUPS  # exactly 32/64/128, whatever the build default is (r4d.h)
    got = bool(r4d.gemm_w4a16_nt_m64_has_group(grp))
    if got != want:
        print("has_group(%d) = %s, want %s   <-- FAIL" % (grp, got, want))
        bad += 1

print("%-6s %-18s %-18s %s" % ("group", "shape (M,N,K)", "WV/SK/MB/NPW/NT", "rel err"))
# N=48 and N=80 are the tail-block cases (N/16 not a multiple of WV*NPW); K=5120 and 17408 are the
# model's own, and 17408 at SK=16 is 1088 per split: 17 packed blocks, so the G=32 path's
# split offset lands mid-row on an odd block.
SHAPES = [(1, 256, 512), (8, 512, 1024), (5, 48, 1024), (16, 1024, 2048), (64, 80, 1024),
          (24, 512, 5120), (8, 256, 17408), (64, 512, 3072)]
TUNINGS = [(4, 4, 1, 1, 1), (4, 4, 1, 1, 0), (2, 2, 0, 4, 0), (8, 1, 0, 1, 1), (1, 16, 0, 4, 1),
           (2, 8, 1, 4, 0)]
seed = 0
for group in GROUPS:
    kdiv = max(group, 64)
    for (M, N, K) in SHAPES:
        seed += 1
        a16, wq, wsz, ref, _ = make_case(M, N, K, group, seed)
        for (WV, SK, MB, NPW, NT) in TUNINGS:
            mb = MB if MB else max(1, min(4, (M + 15) // 16))
            tag = "%d/%d/%d/%d/%d" % (WV, SK, mb, NPW, NT)
            if K % (SK * kdiv):
                bad += expect_throw("g%d (%d,%d,%d) %s K %% (SK*%d) != 0" % (group, M, N, K, tag, kdiv),
                                    lambda: launch(r4d.gemm_w4a16_nt_m64_g, group, a16, wq, wsz,
                                                   M, K, N, WV, SK, mb, NPW, NT))
                continue
            got = launch(r4d.gemm_w4a16_nt_m64_g, group, a16, wq, wsz, M, K, N, WV, SK, mb, NPW,
                         NT).float()
            rel = ((got - ref).norm() / ref.norm()).item()
            flag = "" if rel < 1e-2 else "   <-- FAIL"
            bad += rel >= 1e-2
            print("%-6d %-18s %-18s %.3e%s" % (group, "(%d,%d,%d)" % (M, N, K), tag, rel, flag))
            if group == r4d.GEMM_W4_GROUP:
                old = launch(r4d.gemm_w4a16_nt_m64, None, a16, wq, wsz, M, K, N, WV, SK, mb, NPW,
                             NT).float()
                if not torch.equal(old, got):
                    print("       ungrouped entry differs from _g(%d)   <-- FAIL" % group)
                    bad += 1

# Rejections the table above does not reach: an uninstantiated group, and G=32 with K a multiple
# of SK*32 but not of SK*64 (the split would start mid-block).
a16, wq, wsz, _, _ = make_case(8, 256, 1024, 64, 99)
for grp in (16, 48, 96, 256):
    if not r4d.gemm_w4a16_nt_m64_has_group(grp):
        bad += expect_throw("group %d" % grp,
                            lambda: launch(r4d.gemm_w4a16_nt_m64_g, grp, a16, wq, wsz,
                                           8, 1024, 256, 4, 4, 1, 1, 0))
bad += expect_throw("g32 K=1024 SK=32 (1024 %% (32*64) != 0)",
                    lambda: launch(r4d.gemm_w4a16_nt_m64_g, 32, a16, wq, wsz,
                                   8, 1024, 256, 1, 32, 1, 1, 0))
bad += expect_throw("g32 K=96 SK=1 (96 %% 64 != 0)",
                    lambda: launch(r4d.gemm_w4a16_nt_m64_g, 32, a16, wq, wsz,
                                   8, 96, 256, 4, 1, 1, 1, 0))

print("FAILURES:", bad)
sys.exit(1 if bad else 0)
