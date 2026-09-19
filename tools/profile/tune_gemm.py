"""tools/profile/tune_gemm.py -- sweeps the legal (WV,SK,MB,NPW) grid for every r4d skinny GEMM
kernel (r4d_gemm_bf16_nt_m64 / r4d_gemm_w4a16_nt_m64 / r4d_gemm_w4a8_nt_m64 /
r4d_gemm_mxfp4a8_nt_m64) at this model's real (N,K) linear shapes (src/model/container.cpp's
LoadQuantLinear call sites) for M in {1,2,4,8,16,32,64}, and emits a C++ table
(src/model/gemm_tuning_table.inc) that src/model/linear.cpp's PickTuning consumes.

Route: the Python r4d.pyd binding (C:\\Users\\user\\dev\\libr4d\\build-win\\r4d.pyd) + the
read-only reference venv's torch, HIP device 1 -- same route third_party/libr4d/bench_mxfp4_gemm.py
already uses for its own (narrower) mxfp4 sweep; this tool generalizes that pattern to all four
GEMM families and every shape this model actually calls, keyed for linear.h's PickTuning(layout, N,
K, M) instead of printing a one-off report.

LEGAL-PARAMETER CONSTRAINTS (read from each kernel's own .hip source before writing this sweep,
per the task brief -- see the file:line references below, not re-derived from guessing):
  bf16   (third_party/libr4d/r4d_gemm_bf16_nt_m64.hip:153-158):
         M in 1..64; K % (SK*16) == 0; WV*SK*32 <= 1024; MB in 1..4. No N constraint, no NPW/NT.
  w4a16  (r4d_gemm_w4a16_nt_m64.hip:290-302):
         M in 1..64; N % 16 == 0; K % (SK*128) == 0; WV*SK*32 <= 1024; MB in 1..4;
         NPW in {1,4}; smem = WV*NPW*SK*1024 bytes <= 64*1024 i.e. WV*NPW*SK <= 64.
  w4a8   (r4d_gemm_w4a8_nt_m64.hip:265-280):
         same as w4a16 except K % (SK*128) == 0 (GEMM_W4A8_GROUP=128 too) and
         NPW in {1,2,4,8}.
  mxfp4  (r4d_gemm_mxfp4a8_nt_m64.hip:233-251):
         M in 1..64; N % 16 == 0; K % (SK*32) == 0 (GEMM_MXFP4_GROUP=32); WV*SK*32 <= 1024;
         MB in 1..4; NPW in {1,2,4,8}; WV*NPW*SK <= 64 (same 64KiB smem cap).
Every kernel entry point ALSO throws std::runtime_error itself on an out-of-range combo (r4d.h's
own doc comment: "the all-reduce and GEMM throw std::runtime_error... at the pybind call site"),
so this sweep's try/except around each call is a second, redundant safety net on top of the
constraints computed below -- not a substitute for reading them (a config this script never
generates because it fails the constraint check above is one r4d's own C++ validation would also
have rejected, confirmed by spot-checking a few rejected combos manually against the .hip source).

Usage (read-only reference venv, HIP device 1):
  C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe tools\\profile\\tune_gemm.py \
      [--quick] [--out src\\model\\gemm_tuning_table.inc]

--quick restricts M to {1, 8, 64} and halves the iteration count, for a fast sanity sweep; the full
run (default) covers M in {1,2,4,8,16,32,64} and is what produced docs/perf.md's table.
"""
import argparse
import itertools
import sys
import time

sys.path.insert(0, r"C:\Users\user\dev\libr4d\build-win")
import torch  # noqa: E402  (must come from the reference venv's interpreter)
import r4d  # noqa: E402

DEVICE = "cuda"  # ROCm torch build: torch.cuda.* addresses the HIP device HIP_VISIBLE_DEVICES picks

# ---- this model's real linear shapes (N, K), src/model/container.cpp's LoadQuantLinear call sites,
# values from C:\AI\models\Qwen3.8-27B\config.json (hidden_size=5120, intermediate_size=17408,
# vocab_size=248320) and docs/architecture.md (key_dim=2048, value_dim=6144, attn_out=6144) --------
SHAPES = [
    ("gdn.in_proj_qkv", 10240, 5120),   # 2*key_dim(2048) + value_dim(6144), hidden
    ("gdn.out_proj", 5120, 6144),       # hidden, value_dim
    ("attn.qg", 12288, 5120),           # 2*num_heads*head_dim (2*24*256), hidden
    ("attn.o", 5120, 6144),             # hidden, num_heads*head_dim
    ("mlp.gate_up", 34816, 5120),       # 2*intermediate_size, hidden
    ("mlp.down", 5120, 17408),          # hidden, intermediate_size
    ("lm_head", 248320, 5120),          # vocab_size, hidden (bf16-only on disk, timed here anyway
                                         # since ApplyLinear's dispatch table is layout-keyed, not
                                         # tensor-keyed -- see "Known limitation" note below)
]

M_BANDS_FULL = [1, 2, 4, 8, 16, 32, 64]
M_BANDS_QUICK = [1, 8, 64]

WV_CANDIDATES = [1, 2, 4, 8, 16, 32]
SK_CANDIDATES = [1, 2, 4, 8, 16, 32]
NPW_CANDIDATES = {"w4a16": [1, 4], "w4a8": [1, 2, 4, 8], "mxfp4": [1, 2, 4, 8]}
GROUP = {"w4a16": 128, "w4a8": 128, "mxfp4": 32}


def mb_for(m: int) -> int:
    return max(1, min(4, (m + 15) // 16))


def legal_bf16(k, wv, sk, mb):
    return k % (sk * 16) == 0 and wv * sk * 32 <= 1024 and 1 <= mb <= 4


def legal_quant(layout, n, k, wv, sk, mb, npw):
    group = GROUP[layout]
    if n % 16 != 0:
        return False
    if k % (sk * group) != 0:
        return False
    if wv * sk * 32 > 1024:
        return False
    if not (1 <= mb <= 4):
        return False
    if npw not in NPW_CANDIDATES[layout]:
        return False
    if wv * npw * sk > 64:  # WV*NPW*SK*1024B <= 64KiB
        return False
    return True


def alloc_bf16(m, n, k, gen):
    a = torch.randn(m, k, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    w = torch.randn(n, k, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    c = torch.zeros(m, n, dtype=torch.bfloat16, device=DEVICE)
    return a, w, c


def alloc_w4a16(m, n, k, gen):
    a = torch.randn(m, k, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    wq = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8).to(DEVICE)
    wsz = torch.randint(0, 2**31 - 1, (n * k // 128,), generator=gen, dtype=torch.int64) \
        .to(torch.uint32 if hasattr(torch, "uint32") else torch.int32).to(DEVICE)
    c = torch.zeros(m, n, dtype=torch.bfloat16, device=DEVICE)
    return a, wq, wsz, c


def alloc_w4a8(m, n, k, gen):
    a = torch.randint(-127, 127, (m, k), generator=gen, dtype=torch.int8).to(DEVICE)
    ascale = torch.full((m,), 0.05, device=DEVICE)
    wq = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8).to(DEVICE)
    ws = torch.randint(0, 2**31 - 1, (n * k // 128,), generator=gen, dtype=torch.int64) \
        .to(torch.uint32 if hasattr(torch, "uint32") else torch.int32).to(DEVICE)
    c = torch.zeros(m, n, dtype=torch.bfloat16, device=DEVICE)
    return a, ascale, wq, ws, c


def alloc_mxfp4(m, n, k, gen):
    a = (torch.randn(m, k, generator=gen) * 0.4).to(torch.float8_e4m3fn).to(DEVICE)
    ascale = torch.full((m,), 0.7, device=DEVICE)
    wq = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8).to(DEVICE)
    ws = torch.randint(118, 130, (k // 32, n), generator=gen, dtype=torch.uint8).to(DEVICE)
    wref = ws.to(torch.int64).amax(dim=0).to(torch.int8).to(DEVICE)
    c = torch.zeros(m, n, dtype=torch.bfloat16, device=DEVICE)
    return a, ascale, wq, ws, wref, c


def bench(fn, iters):
    for _ in range(min(5, iters)):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e6  # microseconds/call


def sweep_shape(layout, name, n, k, m_bands, iters):
    gen = torch.Generator(device="cpu").manual_seed(0)
    stream = torch.cuda.current_stream().cuda_stream
    results = {}
    for m in m_bands:
        mb = mb_for(m)
        best = None
        if layout == "bf16":
            a, w, c = alloc_bf16(m, n, k, gen)
            for wv, sk in itertools.product(WV_CANDIDATES, SK_CANDIDATES):
                if not legal_bf16(k, wv, sk, mb):
                    continue
                def call(wv=wv, sk=sk):
                    r4d.gemm_bf16_nt_m64(a.data_ptr(), w.data_ptr(), c.data_ptr(), m, k, n, wv, sk,
                                          mb, stream)
                try:
                    us = bench(call, iters)
                except Exception:
                    continue
                if best is None or us < best[0]:
                    best = (us, wv, sk, mb, 1, 1)
        elif layout == "w4a16":
            a, wq, wsz, c = alloc_w4a16(m, n, k, gen)
            for wv, sk, npw in itertools.product(WV_CANDIDATES, SK_CANDIDATES, NPW_CANDIDATES["w4a16"]):
                if not legal_quant("w4a16", n, k, wv, sk, mb, npw):
                    continue
                def call(wv=wv, sk=sk, npw=npw):
                    r4d.gemm_w4a16_nt_m64(a.data_ptr(), wq.data_ptr(), wsz.data_ptr(), c.data_ptr(),
                                           m, k, n, wv, sk, mb, npw, 1, stream)
                try:
                    us = bench(call, iters)
                except Exception:
                    continue
                if best is None or us < best[0]:
                    best = (us, wv, sk, mb, npw, 1)
        elif layout == "w4a8":
            a, ascale, wq, ws, c = alloc_w4a8(m, n, k, gen)
            for wv, sk, npw in itertools.product(WV_CANDIDATES, SK_CANDIDATES, NPW_CANDIDATES["w4a8"]):
                if not legal_quant("w4a8", n, k, wv, sk, mb, npw):
                    continue
                def call(wv=wv, sk=sk, npw=npw):
                    r4d.gemm_w4a8_nt_m64(a.data_ptr(), ascale.data_ptr(), wq.data_ptr(),
                                          ws.data_ptr(), c.data_ptr(), m, k, n, wv, sk, mb, npw, 1,
                                          stream)
                try:
                    us = bench(call, iters)
                except Exception:
                    continue
                if best is None or us < best[0]:
                    best = (us, wv, sk, mb, npw, 1)
        else:  # mxfp4
            a, ascale, wq, ws, wref, c = alloc_mxfp4(m, n, k, gen)
            for wv, sk, npw in itertools.product(WV_CANDIDATES, SK_CANDIDATES, NPW_CANDIDATES["mxfp4"]):
                if not legal_quant("mxfp4", n, k, wv, sk, mb, npw):
                    continue
                def call(wv=wv, sk=sk, npw=npw):
                    r4d.gemm_mxfp4a8_nt_m64(a.data_ptr(), ascale.data_ptr(), wq.data_ptr(),
                                             ws.data_ptr(), wref.data_ptr(), c.data_ptr(), m, k, n,
                                             wv, sk, mb, npw, stream)
                try:
                    us = bench(call, iters)
                except Exception:
                    continue
                if best is None or us < best[0]:
                    best = (us, wv, sk, mb, npw, 1)
        if best is not None:
            results[m] = best
    return results


LAYOUT_ENUM = {"bf16": "Layout::kBf16", "w4a16": "Layout::kW4a16", "w4a8": "Layout::kW4a8",
               "mxfp4": "Layout::kMxfp4"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out", default=r"src\model\gemm_tuning_table.inc")
    ap.add_argument("--layouts", default="bf16,w4a16,w4a8,mxfp4")
    args = ap.parse_args()

    m_bands = M_BANDS_QUICK if args.quick else M_BANDS_FULL
    iters = 15 if args.quick else 40
    layouts = args.layouts.split(",")

    all_rows = []  # (layout, name, N, K, M, WV, SK, MB, NPW, NT, us)
    print(f"{'layout':7s} {'shape':16s} {'N':>7s} {'K':>7s} {'M':>4s} {'WV/SK/MB/NPW':16s} {'us':>9s}")
    for layout in layouts:
        for name, n, k in SHAPES:
            res = sweep_shape(layout, name, n, k, m_bands, iters)
            for m, (us, wv, sk, mb, npw, nt) in sorted(res.items()):
                all_rows.append((layout, name, n, k, m, wv, sk, mb, npw, nt, us))
                print(f"{layout:7s} {name:16s} {n:7d} {k:7d} {m:4d} "
                      f"{f'{wv}/{sk}/{mb}/{npw}':16s} {us:9.2f}")

    with open(args.out, "w") as f:
        f.write("// AUTO-GENERATED by tools/profile/tune_gemm.py -- do not hand-edit.\n")
        f.write("// Measured on HIP device 1, R9700 (gfx1201), this model's real (N,K) shapes.\n")
        f.write("// Included by src/model/linear.cpp's PickTuning; see that file for the fallback\n")
        f.write("// path when (layout,N,K,M) misses this table (an untuned shape/M combo).\n")
        f.write("static const GemmTuningRow kGemmTuningTable[] = {\n")
        for layout, name, n, k, m, wv, sk, mb, npw, nt, us in all_rows:
            f.write(f"    {{{LAYOUT_ENUM[layout]}, {n}, {k}, {m}, "
                     f"{{{wv}, {sk}, {mb}, {npw}, {nt}}}}},  // {name} {us:.2f}us\n")
        f.write("};\n")
    print(f"\nwrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
