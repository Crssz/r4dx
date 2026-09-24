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
         M in 1..64; N % 16 == 0; K % (SK*group) == 0 (group = R4D_GEMM_W4_GROUP, see "W4A16
         GROUP" below); WV*SK*32 <= 1024; MB in 1..4;
         NPW in {1,4}; smem = WV*NPW*SK*1024 bytes <= 64*1024 i.e. WV*NPW*SK <= 64.
  w4a8   (r4d_gemm_w4a8_nt_m64.hip:265-280):
         same as w4a16 except K % (SK*128) == 0 (GEMM_W4A8_GROUP=128) and
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

Usage (HIP device 1; any CPython 3.12 with a ROCm build of torch -- r4d.pyd links python312 -- e.g.
the read-only reference venv C:\\Users\\user\\dev\\vLLM_for_AMD\\.venv-rocm10 where it exists):
  $env:HIP_VISIBLE_DEVICES = '1'
  $env:R4DX_LIBR4D_BUILD = "$HOME\\dev\\libr4d\\build-win\\g64"   # see "W4A16 GROUP" below
  <python.exe> tools\\profile\\tune_gemm.py [--quick] [--out src\\model\\gemm_tuning_table.inc]

--quick restricts M to {1, 8, 64} and halves the iteration count, for a fast sanity sweep; the full
run (default) covers M in {1,2,4,8,16,32,64} and is what produced docs/perf.md's table.

--m-bands "1,8,64" overrides the M band list entirely (e.g. "1" alone for a fast single-row re-run
of specific --shapes, docs/r9700.md Q5's own use case).

--replace rewrites, in place, only the rows this run sweeps and leaves every other row and the
file's header alone -- e.g. `--layouts w4a16 --replace` re-tunes w4a16 after a group change without
re-sweeping bf16/w4a8/mxfp4.

W4A16 GROUP. r4d_gemm_w4a16_nt_m64 is compiled at one group size (R4D_GEMM_W4_GROUP: K per
(scale, zero) dword), and the group decides both which SK are legal and how big the `wsz` buffer is
(N*K/group dwords). The r4dx build compiles the kernel at its R4DX_W4A16_GROUP option (default 64
since Milestone 11; third_party/CMakeLists.txt passes -DR4D_GEMM_W4_GROUP). r4d.pyd is built
separately, by libr4d's own build_windows.ps1, which passes no such flag -- so a stock
build-win\\r4d.pyd carries the kernel source's default, 128, and sweeping it tunes a kernel the
default r4dx build does not run. So main() prints the groups the loaded pyd was compiled with
(r4d.GEMM_W4_GROUP / GEMM_W4A8_GROUP / GEMM_MXFP4_GROUP), sizes every scale buffer from them (a
buffer sized for another group is either too short, and read past, or never fully read), and refuses
to sweep a layout whose pyd group differs from the r4dx build's: --w4a16-group (default 64) for
w4a16, 128 for w4a8, 32 for mxfp4.

Building a group-64 r4d.pyd without editing libr4d: build_windows.ps1 has no group parameter, but
clang's driver appends whatever CCC_OVERRIDE_OPTIONS names to every compile it runs, and
R4D_GEMM_W4_GROUP is read by r4d_gemm_w4a16_nt_m64.hip alone:

  $env:CCC_OVERRIDE_OPTIONS = '+-DR4D_GEMM_W4_GROUP=64'
  & $HOME\\dev\\libr4d\\build_windows.ps1 -OutDir build-win\\g64 -RocmRoot C:\\opt\\rocm `
      -Pybind11Include <any torch install>\\Lib\\site-packages\\torch\\include
  Remove-Item env:CCC_OVERRIDE_OPTIONS

-OutDir is relative to libr4d, and libr4d's .gitignore already covers build-win\\. -RocmRoot is the
HIP SDK the r4dx build itself compiles with (docs/build-windows.md). -Pybind11Include is only
needed when the reference venv, the script's default source of pybind11, is absent -- otherwise it
tries to install pybind11 into a fresh uv venv. build.log shows "### Adding argument
-DR4D_GEMM_W4_GROUP=64 at end" once per compile, and the group check above is the real
verification: the pyd must report w4a16=64, or this script will not sweep it.

Q5 fix (docs/r9700.md, Milestone 3 profiling pass, 2026-09-20): every bench() call used to read the
SAME weight buffer on every iteration of its timed loop, so a shape smaller than the 64 MiB
Infinity Cache (MALL) stayed resident across the whole timed loop -- exactly what docs/r9700.md's
rule 2 warns [TUNE]'s M=1 rows do (mxfp4 `mlp.down`/`gdn.in_proj_qkv` backing out ABOVE DRAM peak).
Each alloc_* function below now allocates a RING of independent weight-buffer copies whose combined
size exceeds 256 MiB (and is at least 4 buffers, even for shapes whose own footprint already exceeds
256 MiB alone) -- `RingCall` rotates through the ring once per timed call (not once per M-band), so
consecutive launches read *different* physical bytes and the 64 MiB MALL cannot make a cold-DRAM
shape look cache-resident. Only the WEIGHT buffer rotates (the activation `a` and output `c` buffers
stay single-allocated -- at M<=64 they are at most ~1.3 MiB, small enough that rotating them buys
nothing and would only slow the sweep down).
"""
import argparse
import itertools
import math
import os
import re
import sys
import time
from pathlib import Path

# Derived from the environment rather than hardcoded, so no local account name is baked into the
# repo. Override with R4DX_LIBR4D_BUILD if libr4d's Python extension lives elsewhere.
sys.path.insert(0, os.environ.get("R4DX_LIBR4D_BUILD", str(Path.home() / "dev" / "libr4d" / "build-win")))
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
    # R1 (docs/r9700.md): gdn.in_proj_z / attn.k / attn.v joined the quantized-linear family and
    # now route through ApplyLinear/PickTuning like every shape above -- previously gdn.in_proj_z
    # ran a hardcoded WV=4,SK=4,MB=1 (gdn_layer.cpp) and attn.k/v a validity-only heuristic
    # (attention/linear.hpp, now removed), neither ever swept here (R9700.md's R4). Re-sweep with
    # `--shapes gdn.in_proj_z,attn.k,attn.v --append` to add just these rows without re-running the
    # whole (much longer) table.
    ("gdn.in_proj_z", 6144, 5120),      # value_dim, hidden
    ("attn.k", 1024, 5120),             # kv_heads*head_dim (4*256), hidden
    ("attn.v", 1024, 5120),             # kv_heads*head_dim (4*256), hidden
    # Tensor parallel TP=2 per-rank shapes (docs/tp.md 2.7, P2a). These rows go ONLY into
    # src/model/gemm_tuning_table_tp2.inc (`--out src\model\gemm_tuning_table_tp2.inc --shapes
    # tp2.*`), never into gemm_tuning_table.inc: linear.cpp consults that table only on a thread that
    # loaded a TP rank, so TP=1 keeps its bytes and timing ((w4a16, 17408, 5120) is also the TP=1
    # DFlash drafter's gate_proj/up_proj, which must stay on FallbackTuning). main() enforces it: an
    # empty --shapes selects only the non-tp2 names, tp2.* runs only when named, and a run refuses
    # to write tp2.* rows to any other file or other rows to that one. attn.qg at TP=2 is
    # (6144, 5120) = gdn.in_proj_z above, served from the main table by the TP lookup's fallback.
    ("tp2.gdn.in_proj_qkv", 5120, 5120),  # 3 segments q|k|v of 1024/1024/3072, hidden
    ("tp2.gdn.in_proj_z", 3072, 5120),    # value_dim/2, hidden
    ("tp2.out_proj", 5120, 3072),         # hidden, value_dim/2 (gdn.out_proj) = heads/2*256 (attn.o)
    ("tp2.mlp.gate_up", 17408, 5120),     # 2 segments gate|up of 8704, hidden
    ("tp2.mlp.down", 5120, 8704),         # hidden, intermediate/2
    ("tp2.lm_head", 124160, 5120),        # vocab/2 (vocab-split lm_head), hidden
    ("tp2.attn.kv", 512, 5120),           # kv_heads/2*head_dim, hidden (attn.k and attn.v)
]

TP2_PREFIX = "tp2."
TP2_TABLE = "gemm_tuning_table_tp2.inc"

M_BANDS_FULL = [1, 2, 4, 8, 16, 32, 64]
M_BANDS_QUICK = [1, 8, 64]

WV_CANDIDATES = [1, 2, 4, 8, 16, 32]
SK_CANDIDATES = [1, 2, 4, 8, 16, 32]
NPW_CANDIDATES = {"w4a16": [1, 4], "w4a8": [1, 2, 4, 8], "mxfp4": [1, 2, 4, 8]}
# The groups the loaded r4d.pyd was compiled with ("W4A16 GROUP" above). Legality and every scale
# buffer's size come from these, so the sweep never hands a kernel a buffer sized for another group.
GROUP = {"w4a16": r4d.GEMM_W4_GROUP, "w4a8": r4d.GEMM_W4A8_GROUP, "mxfp4": r4d.GEMM_MXFP4_GROUP}
# The groups the r4dx build compiles w4a8/mxfp4 at (third_party/CMakeLists.txt); w4a16's is the
# R4DX_W4A16_GROUP build option, taken from --w4a16-group.
R4DX_GROUP = {"w4a8": 128, "mxfp4": 32}


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


RING_MIN_TOTAL_BYTES = 256 * 1024 * 1024
RING_MIN_COUNT = 4


def ring_count(bytes_per_copy: int) -> int:
    """Q5: how many independent copies of a `bytes_per_copy`-sized weight buffer are needed so the
    ring's combined footprint exceeds 256 MiB, with at least 4 copies regardless (so even a shape
    whose own weight already exceeds 256 MiB still rotates through >=4 distinct physical buffers,
    per the task's own ">=4 distinct weight buffers" wording, not just ">256 MiB total")."""
    return max(RING_MIN_COUNT, math.ceil(RING_MIN_TOTAL_BYTES / max(1, bytes_per_copy)))


class RingCall:
    """Wraps a zero-arg kernel-launch closure factory so each successive invocation rotates to the
    next buffer set in `bufs` (Q5, see module docstring) -- `fn(*bufs[i])` is called and `i` advances
    (mod len(bufs)) on every call, including bench()'s warmup iterations."""

    def __init__(self, fn, bufs):
        self.fn = fn
        self.bufs = bufs
        self.i = 0

    def __call__(self):
        buf = self.bufs[self.i % len(self.bufs)]
        self.i += 1
        self.fn(*buf)


def alloc_bf16(m, n, k, gen):
    a = torch.randn(m, k, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    c = torch.zeros(m, n, dtype=torch.bfloat16, device=DEVICE)
    count = ring_count(n * k * 2)
    ws = [torch.randn(n, k, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
          for _ in range(count)]
    return a, [(w,) for w in ws], c


def alloc_w4a16(m, n, k, gen):
    a = torch.randn(m, k, generator=gen, dtype=torch.float32).to(torch.bfloat16).to(DEVICE)
    c = torch.zeros(m, n, dtype=torch.bfloat16, device=DEVICE)
    nsz = n * k // GROUP["w4a16"]  # one (scale, zero) dword per (row, group)
    count = ring_count(n * (k // 2) + nsz * 4)
    bufs = []
    for _ in range(count):
        wq = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8).to(DEVICE)
        wsz = torch.randint(0, 2**31 - 1, (nsz,), generator=gen, dtype=torch.int64) \
            .to(torch.uint32 if hasattr(torch, "uint32") else torch.int32).to(DEVICE)
        bufs.append((wq, wsz))
    return a, bufs, c


def alloc_w4a8(m, n, k, gen):
    a = torch.randint(-127, 127, (m, k), generator=gen, dtype=torch.int8).to(DEVICE)
    ascale = torch.full((m,), 0.05, device=DEVICE)
    c = torch.zeros(m, n, dtype=torch.bfloat16, device=DEVICE)
    nws = n * k // GROUP["w4a8"]  # one scale dword per (row, group), f16 in the low half
    count = ring_count(n * (k // 2) + nws * 4)
    bufs = []
    for _ in range(count):
        wq = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8).to(DEVICE)
        ws = torch.randint(0, 2**31 - 1, (nws,), generator=gen, dtype=torch.int64) \
            .to(torch.uint32 if hasattr(torch, "uint32") else torch.int32).to(DEVICE)
        bufs.append((wq, ws))
    return a, ascale, bufs, c


def alloc_mxfp4(m, n, k, gen):
    a = (torch.randn(m, k, generator=gen) * 0.4).to(torch.float8_e4m3fn).to(DEVICE)
    ascale = torch.full((m,), 0.7, device=DEVICE)
    c = torch.zeros(m, n, dtype=torch.bfloat16, device=DEVICE)
    count = ring_count(n * (k // 2) + (k // 32) * n + n)
    bufs = []
    for _ in range(count):
        wq = torch.randint(0, 256, (n, k // 2), generator=gen, dtype=torch.uint8).to(DEVICE)
        ws = torch.randint(118, 130, (k // 32, n), generator=gen, dtype=torch.uint8).to(DEVICE)
        wref = ws.to(torch.int64).amax(dim=0).to(torch.int8).to(DEVICE)
        bufs.append((wq, ws, wref))
    return a, ascale, bufs, c


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
            a, w_bufs, c = alloc_bf16(m, n, k, gen)
            for wv, sk in itertools.product(WV_CANDIDATES, SK_CANDIDATES):
                if not legal_bf16(k, wv, sk, mb):
                    continue
                def fn(w, wv=wv, sk=sk):
                    r4d.gemm_bf16_nt_m64(a.data_ptr(), w.data_ptr(), c.data_ptr(), m, k, n, wv, sk,
                                          mb, stream)
                call = RingCall(fn, w_bufs)
                try:
                    us = bench(call, iters)
                except Exception:
                    continue
                if best is None or us < best[0]:
                    best = (us, wv, sk, mb, 1, 1)
        elif layout == "w4a16":
            a, w_bufs, c = alloc_w4a16(m, n, k, gen)
            for wv, sk, npw in itertools.product(WV_CANDIDATES, SK_CANDIDATES, NPW_CANDIDATES["w4a16"]):
                if not legal_quant("w4a16", n, k, wv, sk, mb, npw):
                    continue
                def fn(wq, wsz, wv=wv, sk=sk, npw=npw):
                    r4d.gemm_w4a16_nt_m64(a.data_ptr(), wq.data_ptr(), wsz.data_ptr(), c.data_ptr(),
                                           m, k, n, wv, sk, mb, npw, 1, stream)
                call = RingCall(fn, w_bufs)
                try:
                    us = bench(call, iters)
                except Exception:
                    continue
                if best is None or us < best[0]:
                    best = (us, wv, sk, mb, npw, 1)
        elif layout == "w4a8":
            a, ascale, w_bufs, c = alloc_w4a8(m, n, k, gen)
            for wv, sk, npw in itertools.product(WV_CANDIDATES, SK_CANDIDATES, NPW_CANDIDATES["w4a8"]):
                if not legal_quant("w4a8", n, k, wv, sk, mb, npw):
                    continue
                def fn(wq, ws, wv=wv, sk=sk, npw=npw):
                    r4d.gemm_w4a8_nt_m64(a.data_ptr(), ascale.data_ptr(), wq.data_ptr(),
                                          ws.data_ptr(), c.data_ptr(), m, k, n, wv, sk, mb, npw, 1,
                                          stream)
                call = RingCall(fn, w_bufs)
                try:
                    us = bench(call, iters)
                except Exception:
                    continue
                if best is None or us < best[0]:
                    best = (us, wv, sk, mb, npw, 1)
        else:  # mxfp4
            a, ascale, w_bufs, c = alloc_mxfp4(m, n, k, gen)
            for wv, sk, npw in itertools.product(WV_CANDIDATES, SK_CANDIDATES, NPW_CANDIDATES["mxfp4"]):
                if not legal_quant("mxfp4", n, k, wv, sk, mb, npw):
                    continue
                def fn(wq, ws, wref, wv=wv, sk=sk, npw=npw):
                    r4d.gemm_mxfp4a8_nt_m64(a.data_ptr(), ascale.data_ptr(), wq.data_ptr(),
                                             ws.data_ptr(), wref.data_ptr(), c.data_ptr(), m, k, n,
                                             wv, sk, mb, npw, stream)
                call = RingCall(fn, w_bufs)
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
# One emitted row: `    {Layout::kW4a16, 10240, 5120, 1, {1, 2, 1, 1, 1}},  // gdn.in_proj_qkv 45.76us`
# -> (layout enum, N, K, M, shape name), the key --replace matches on.
ROW_RE = re.compile(r"^\s*\{(Layout::\w+), (\d+), (\d+), (\d+), \{[^}]*\}\},\s*// (\S+) ")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--m-bands", default="",
                     help="comma-separated M values overriding --quick/the default full band list "
                          "(docs/r9700.md Q5: e.g. --m-bands 1 for a fast single-row re-run).")
    ap.add_argument("--out", default=r"src\model\gemm_tuning_table.inc")
    ap.add_argument("--layouts", default="bf16,w4a16,w4a8,mxfp4")
    ap.add_argument("--shapes", default="",
                     help="comma-separated subset of SHAPES' names to sweep (default: all but the "
                          "tp2.* ones, which run only when named and only into "
                          "gemm_tuning_table_tp2.inc). "
                          "R1 (docs/r9700.md R4): pass --shapes gdn.in_proj_z,attn.k,attn.v "
                          "--append to add just the three newly-quantized shapes' rows without "
                          "re-running the whole (much longer) table.")
    ap.add_argument("--append", action="store_true",
                     help="insert the new rows into the existing --out file (before its closing "
                          "'};') instead of overwriting it. Requires --out to already exist and "
                          "contain a well-formed kGemmTuningTable array (i.e. a prior non---append "
                          "run's output).")
    ap.add_argument("--replace", action="store_true",
                     help="rewrite, in place in the existing --out file, exactly the rows this run "
                          "sweeps -- matched on (layout, N, K, M, shape name) -- leaving every other "
                          "row and the header untouched. Every swept row must already exist there.")
    ap.add_argument("--w4a16-group", type=int, default=64,
                     help="the R4DX_W4A16_GROUP of the r4dx build this table is for (default 64, "
                          "that option's default). A w4a16 sweep refuses an r4d.pyd compiled at any "
                          "other group -- see the module docstring's \"W4A16 GROUP\".")
    args = ap.parse_args()
    if args.append and args.replace:
        raise SystemExit("--append and --replace are mutually exclusive")

    if args.m_bands:
        m_bands = [int(x) for x in args.m_bands.split(",") if x]
    else:
        m_bands = M_BANDS_QUICK if args.quick else M_BANDS_FULL
    iters = 15 if args.quick else 40
    layouts = args.layouts.split(",")
    shape_filter = set(s for s in args.shapes.split(",") if s)
    # The default set leaves out the TP=2 per-rank shapes: they are swept only when named, and only
    # into their own table (docs/tp.md 2.7; see SHAPES).
    shapes = [s for s in SHAPES
              if (s[0] in shape_filter if shape_filter else not s[0].startswith(TP2_PREFIX))]
    if shape_filter and len(shapes) != len(shape_filter):
        missing = shape_filter - {s[0] for s in shapes}
        raise SystemExit(f"--shapes named unknown shape(s): {sorted(missing)}")
    out_is_tp2 = Path(args.out).name == TP2_TABLE
    wrong_table = [s[0] for s in shapes if s[0].startswith(TP2_PREFIX) != out_is_tp2]
    if wrong_table:
        raise SystemExit(
            f"refusing to write {sorted(wrong_table)} to {args.out}: tp2.* rows go only to "
            f"{TP2_TABLE}, every other row only to another table (docs/tp.md 2.7)")

    want = dict(R4DX_GROUP, w4a16=args.w4a16_group)
    print(f"r4d.pyd: {r4d.__file__}")
    print("groups it was compiled with: " + ", ".join(f"{l}={g}" for l, g in GROUP.items()))
    wrong = [l for l in layouts if l in want and GROUP[l] != want[l]]
    if wrong:
        raise SystemExit(
            "refusing to sweep " + ", ".join(f"{l} (pyd group {GROUP[l]}, r4dx build group "
                                              f"{want[l]})" for l in wrong) +
            ": that would tune a kernel the r4dx build does not run. Build an r4d.pyd at the r4dx "
            "group and point R4DX_LIBR4D_BUILD at it (module docstring, \"W4A16 GROUP\"), or pass "
            "--w4a16-group to name the build this table is for.")

    all_rows = []  # (layout, name, N, K, M, WV, SK, MB, NPW, NT, us)
    print(f"{'layout':7s} {'shape':16s} {'N':>7s} {'K':>7s} {'M':>4s} {'WV/SK/MB/NPW':16s} {'us':>9s}")
    for layout in layouts:
        for name, n, k in shapes:
            res = sweep_shape(layout, name, n, k, m_bands, iters)
            for m, (us, wv, sk, mb, npw, nt) in sorted(res.items()):
                all_rows.append((layout, name, n, k, m, wv, sk, mb, npw, nt, us))
                print(f"{layout:7s} {name:16s} {n:7d} {k:7d} {m:4d} "
                      f"{f'{wv}/{sk}/{mb}/{npw}':16s} {us:9.2f}")

    row_lines = [
        f"    {{{LAYOUT_ENUM[layout]}, {n}, {k}, {m}, "
        f"{{{wv}, {sk}, {mb}, {npw}, {nt}}}}},  // {name} {us:.2f}us\n"
        for layout, name, n, k, m, wv, sk, mb, npw, nt, us in all_rows
    ]

    if args.append:
        with open(args.out) as f:
            existing = f.readlines()
        close_idx = next((i for i, line in enumerate(existing) if line.strip() == "};"), None)
        if close_idx is None:
            raise SystemExit(f"--append: {args.out} has no closing '}};' line to insert before -- "
                              "not a file this script previously wrote")
        merged = existing[:close_idx] + row_lines + existing[close_idx:]
        with open(args.out, "w") as f:
            f.writelines(merged)
        print(f"\nappended {len(all_rows)} rows to {args.out} (before line {close_idx + 1})")
    elif args.replace:
        with open(args.out) as f:
            existing = f.readlines()
        new = {(LAYOUT_ENUM[layout], n, k, m, name): line
               for (layout, name, n, k, m, *_), line in zip(all_rows, row_lines)}
        for i, line in enumerate(existing):
            hit = ROW_RE.match(line)
            if hit:
                key = (hit[1], int(hit[2]), int(hit[3]), int(hit[4]), hit[5])
                if key in new:
                    existing[i] = new.pop(key)
        if new:
            raise SystemExit(f"--replace: {args.out} has no row for {sorted(new)} -- nothing "
                              "written; use --append to add rows")
        with open(args.out, "w") as f:
            f.writelines(existing)
        print(f"\nreplaced {len(all_rows)} rows in {args.out}")
    else:
        with open(args.out, "w") as f:
            f.write("// AUTO-GENERATED by tools/profile/tune_gemm.py -- do not hand-edit.\n")
            f.write("// Measured on HIP device 1, R9700 (gfx1201), this model's real (N,K) shapes.\n")
            f.write("// Included by src/model/linear.cpp's PickTuning; see that file for the fallback\n")
            f.write("// path when (layout,N,K,M) misses this table (an untuned shape/M combo).\n")
            f.write("// Kernel groups swept: " +
                    ", ".join(f"{l}={GROUP[l]}" for l in layouts if l in GROUP) + ".\n")
            f.write("static const GemmTuningRow kGemmTuningTable[] = {\n")
            f.writelines(row_lines)
            f.write("};\n")
        print(f"\nwrote {len(all_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
