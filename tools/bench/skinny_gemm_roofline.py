"""tools/bench/skinny_gemm_roofline.py -- task 5 of the r9700-microbench brief: time
r4d_gemm_w4a16_nt_m64 and r4d_gemm_w4a8_nt_m64 at M=1, N=34816, K=5120 (this model's
mlp.gate_up shape, the single most expensive GEMM per docs/perf.md's per-op profile) through the
r4d.pyd binding, and compute achieved weight-streaming bytes/s to compare against the stream
bandwidth measured by bench_bandwidth.exe (task 1).

Route: same as tools/profile/tune_gemm.py / third_party/libr4d/bench_mxfp4_gemm.py -- the
reference venv's torch + r4d.pyd, HIP device 1. (N,K,M)=(34816,5120,1) and the WV/SK/MB/NPW/NT
tuning are the already-measured-optimal values from src/model/gemm_tuning_table.inc (mlp.gate_up
row, M=1 band) -- this script re-times them live rather than re-sweeping, since the sweep itself
was already done and checked in by the GEMM-tuning pass (docs/perf.md "GEMM tuning sweep").

Usage (read-only reference venv, HIP device 1):
  C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10\\Scripts\\python.exe tools\\bench\\skinny_gemm_roofline.py
"""
import os
import sys
import time
from pathlib import Path

# Derived from the environment; override with R4DX_LIBR4D_BUILD.
sys.path.insert(0, os.environ.get("R4DX_LIBR4D_BUILD", str(Path.home() / "dev" / "libr4d" / "build-win")))
import torch  # noqa: E402
import r4d  # noqa: E402

DEVICE = "cuda"
N, K, M = 34816, 5120, 1
GROUP = 128

# Tuning from src/model/gemm_tuning_table.inc, mlp.gate_up row, M=1 band (measured by
# tools/profile/tune_gemm.py's own sweep, this script just re-times the already-chosen winner):
#   w4a16: WV=2 SK=4 MB=1 NPW=1 NT=1 -> 147.76us
#   w4a8:  WV=2 SK=4 MB=1 NPW=1 NT=1 -> 146.04us
WV, SK, MB, NPW, NT = 2, 4, 1, 1, 1


def bench(fn, iters=60):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    best = 1e9
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best * 1e6  # us


def main():
    gen = torch.Generator(device="cpu").manual_seed(0)
    stream = torch.cuda.current_stream().cuda_stream

    # ---- w4a16 ----
    a16 = torch.randn(M, K, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    wq16 = torch.randint(0, 256, (N, K // 2), generator=gen, dtype=torch.uint8).to(DEVICE)
    wsz16 = torch.randint(0, 2**31 - 1, (N * K // GROUP,), generator=gen, dtype=torch.int64) \
        .to(torch.uint32 if hasattr(torch, "uint32") else torch.int32).to(DEVICE)
    c16 = torch.zeros(M, N, dtype=torch.bfloat16, device=DEVICE)

    def call_w4a16():
        r4d.gemm_w4a16_nt_m64(a16.data_ptr(), wq16.data_ptr(), wsz16.data_ptr(), c16.data_ptr(),
                               M, K, N, WV, SK, MB, NPW, NT, stream)

    us_w4a16_runs = [bench(call_w4a16) for _ in range(3)]

    # ---- w4a8 ----
    a8 = torch.randint(-127, 127, (M, K), generator=gen, dtype=torch.int8).to(DEVICE)
    ascale8 = torch.full((M,), 0.05, device=DEVICE)
    wq8 = torch.randint(0, 256, (N, K // 2), generator=gen, dtype=torch.uint8).to(DEVICE)
    ws8 = torch.randint(0, 2**31 - 1, (N * K // GROUP,), generator=gen, dtype=torch.int64) \
        .to(torch.uint32 if hasattr(torch, "uint32") else torch.int32).to(DEVICE)
    c8 = torch.zeros(M, N, dtype=torch.bfloat16, device=DEVICE)

    def call_w4a8():
        r4d.gemm_w4a8_nt_m64(a8.data_ptr(), ascale8.data_ptr(), wq8.data_ptr(), ws8.data_ptr(),
                              c8.data_ptr(), M, K, N, WV, SK, MB, NPW, NT, stream)

    us_w4a8_runs = [bench(call_w4a8) for _ in range(3)]

    # ---- bytes moved (weight read dominates at M=1; activation/output are <70KB, negligible) ----
    packed_bytes = N * K // 2       # 4-bit weights
    scale_bytes = (N * K // GROUP) * 4  # uint32 per-group scale
    weight_bytes = packed_bytes + scale_bytes

    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])

    us16 = median(us_w4a16_runs)
    us8 = median(us_w4a8_runs)
    gbps16 = weight_bytes / (us16 * 1e-6) / 1e9
    gbps8 = weight_bytes / (us8 * 1e-6) / 1e9

    print(f"shape mlp.gate_up N={N} K={K} M={M}  tuning WV={WV} SK={SK} MB={MB} NPW={NPW} NT={NT}")
    print(f"weight_bytes = packed({packed_bytes}) + scales({scale_bytes}) = {weight_bytes} "
          f"({weight_bytes/1024/1024:.2f} MiB)")
    print(f"w4a16: us_median={us16:.2f} raw_us={[round(x,2) for x in us_w4a16_runs]} "
          f"achieved_GBps={gbps16:.2f}")
    print(f"w4a8:  us_median={us8:.2f} raw_us={[round(x,2) for x in us_w4a8_runs]} "
          f"achieved_GBps={gbps8:.2f}")


if __name__ == "__main__":
    main()
