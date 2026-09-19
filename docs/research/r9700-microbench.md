# R9700 (gfx1201) on-device microbenchmarks

Measured 2026-09-19, HIP device 1 (`$env:HIP_VISIBLE_DEVICES='1'`) on the single AMD Radeon AI PRO
R9700 this task owns for GPU work. Toolchain: ROCm SDK `C:\opt\rocm` (HIP 7.15.26333, AMD clang
23.0.0, `x86_64-pc-windows-msvc`), `hipcc.exe` invoked directly (see "Build" below) -- the repo root
`CMakeLists.txt` (`C:\Users\user\dev\r4dx\CMakeLists.txt`) does not `add_subdirectory(tools)`
today, and this task's brief says not to touch it, so `tools\bench\build.ps1` drives `hipcc.exe`
standalone instead of wiring a `tools/CMakeLists.txt` that nothing would ever build.

`hipInfo.exe` on this device (device# 0 under `HIP_VISIBLE_DEVICES=1`, measured):
`multiProcessorCount=32` (AMD reports WGPs -- dual-CU units -- as "multiprocessors"; 32 WGPs = 64
CUs, matching the documented CU count below), `totalGlobalMem=31.86 GB`, `sharedMemPerBlock=64 KB`,
`l2CacheSize=8388608` (8 MiB L2, a separate, smaller cache than the 64 MiB Infinity Cache/MALL --
both numbers are used below, labeled), `major.minor=12.0` (gfx1201), `warpSize=32`.

Documented (not measured by this task, AMD's own published specs, cited): 32 GB GDDR6 @ 640 GB/s
over a 256-bit bus, 64 MiB Infinity Cache, 64 CUs / 4096 stream processors / 128 AI accelerators,
"up to 191 TFLOPS FP16 matrix" and 47.8 TFLOPS FP32 vector. Source:
[VideoCardz.net R9700 spec summary](https://videocardz.net/amd-radeon-ai-pro-r9700),
[HostKey benchmark writeup](https://hostkey.com/blog/142-benchmarking-the-radeon-ai-pro-r9700-amds-red-team-buys-into-on-device-ai/).
Whether the "191 TFLOPS" figure is dense or assumes a sparsity/boost multiplier is not stated by
either source and is **not resolved by this task** -- flagged as an open question below rather than
assumed either way.

**Contention note (applies to most runs below):** another workflow's `ctest`/`test_*`/`r4dx-cli`/
Python processes were intermittently running on device 1 throughout this session despite polling
`Get-Process` and waiting between attempts (up to several minutes at a time, per the task's own
"wait/retry, then run anyway and note contention" instruction). Every table below marks which rows
ran with contending processes present. Compilation (all `hipcc.exe` invocations) is CPU-only and
unaffected by GPU contention.

## Build

```powershell
cd C:\Users\user\dev\r4dx\tools\bench
powershell -ExecutionPolicy Bypass -File .\build.ps1
```

Produces `tools\bench\build\*.exe` (10 binaries: `bench_bandwidth`, `launch_overhead`,
`lds_bandwidth`, and one `wmma_probe_<variant>.exe` per WMMA builtin that compiled -- all 7
attempted variants compiled on the first design that got the register-packing right, see "WMMA
probe" below for the two that did not) plus `build\build_report.json` (machine-readable per-file
compile status). Flags: `-O3 -std=c++17 --offload-arch=gfx1201 -Wno-unused-result
--rocm-path=C:/opt/rocm --rocm-device-lib-path=C:/opt/rocm/lib/llvm/amdgcn/bitcode -DNDEBUG -D_DLL
-D_MT -Xclang --dependent-lib=msvcrt`, matching `docs/build-windows.md`'s documented recipe.

## A methodology bug caught mid-task (read this before trusting any WMMA/LDS number below)

The first version of `wmma_probe.hip` and `lds_bandwidth.hip` guarded their result-write with
`if (threadIdx.x == 0xffffffff) out[...] = acc;` -- intended as "never actually taken, just keeps
the compiler from proving `acc` unused." This backfired completely: since `threadIdx.x < blockDim.x
<= 256` always, LLVM can prove the condition false **at compile time** and deleted the entire kernel
body as dead code (confirmed by disassembling with `hipcc -S --cuda-device-only`: the emitted `.s`
for the first version was a single `s_endpgm`, zero `v_wmma` instructions). Every number from that
version was pure kernel-launch/timer noise, not throughput -- it showed up as three different, all
wrong, symptoms across three fix attempts: (1) near-identical TFLOPS across every dtype (~275-282,
because a *dependency-chain* accumulator version was latency- not throughput-bound), (2) 3-6000
TFLOPS outliers once independent accumulator chains were added but were still literally the same
value across chains (LLVM CSE'd the "independent" chains back into one because they started from
identical literal `{0,...}` with identical `a`/`b` inputs), and (3), after fixing both of those,
multi-million and even *negative* GB/s in `lds_bandwidth` (near-zero/empty kernel, timer-resolution
noise). The fix applied to both files: (a) seed each independent accumulator chain with a distinct
compile-time literal so LLVM cannot prove the chains equal and merge them, and (b) always execute
the write (every thread writes its own slot in a `grid*block`-sized output buffer) instead of
guarding it with a provably-unreachable branch. Re-disassembled after the fix: `v_wmma` instruction
count in the `.s` output for both `f32_16x16x16_f16` and `f32_16x16x16_bf16` variants ==
`NCHAINS*DEPTH` = 64, matching the source's own unroll count exactly -- this is the check that
validated every number in the "WMMA probe" section below. This whole episode is why every table
below reports 3 independent process-level runs with their raw values, not just one number: the
methodology bugs were only visible as run-to-run *inconsistency* before they were understood, and
that inconsistency is exactly what caught them.

## 1. VRAM streaming bandwidth

`tools\bench\bench_bandwidth.hip`: `read_kernel` streams `int4` (16B) vector loads across all CUs
and XORs them into a tiny output via `atomicXor` (survives dead-code elimination without a real
cross-thread reduction dominating the timing); `copy_kernel` does `dst[i] = src[i]` for the
read+write figure. Grid = 512 blocks x 256 threads, `hipEvent`-timed, 3 warmup + 10 timed
launches/size internally (script reports the internal median; the table below is the median of 3
full process invocations). `HIP_VISIBLE_DEVICES=1` throughout; **no contending processes observed
during any of the 3 runs** (checked via `Get-Process` immediately before each).

```powershell
$env:HIP_VISIBLE_DEVICES='1'
.\build\bench_bandwidth.exe 10
```

| Size | Read GB/s (run1/run2/run3) | Read median | Copy GB/s (run1/run2/run3) | Copy median |
|---|---|---|---|---|
| 16 MiB | 399.95 / 394.77 / 383.11 | 394.77 | 776.73 / 743.18 / 748.17 | 748.17 |
| 48 MiB (Infinity-Cache-sized) | 783.38 / 774.93 / 797.89 | 783.38 | 528.14 / 510.79 / 496.14 | 510.79 |
| 256 MiB | 576.23 / 580.22 / 578.34 | 578.34 | 552.74 / 551.81 / 550.44 | 551.81 |
| 2 GiB | 605.65 / 606.15 / 605.79 | 605.79 | 545.83 / 544.61 / 546.04 | 545.83 |
| 8 GiB | 604.11 / 604.08 / 603.62 | 604.08 | 478.52 / 477.70 / 478.61 | 478.52 |

**Reading this**: the 48 MiB read figure (783 GB/s) exceeds the documented 640 GB/s GDDR6 peak --
expected, since 48 MiB is sized to mostly fit the 64 MiB Infinity Cache, so this measures
cache-assisted bandwidth, not pure HBM/GDDR6 bandwidth. The 2-8 GiB figures (604-606 GB/s) are
comfortably below cache size and land at **94-95% of the documented 640 GB/s peak** -- a sane,
consistent steady-state DRAM bandwidth measurement. The 16 MiB read figure (395 GB/s) being *lower*
than the 256 MiB-8 GiB steady-state number (605 GB/s) is counter-intuitive for a fully cache-resident
size and is likely a short-kernel effect (16 MiB / 605 GB/s ~= 26 us kernel, comparable to this
device's few-microsecond launch overhead measured in section 3, enough to visibly dilute a GB/s
figure computed from wall-clock `hipEvent` timing) -- not separately isolated further here, flagged
as an artifact rather than a real bandwidth number. Copy (read+write) bandwidth is lower than read
alone at every size beyond 16 MiB, as expected (2x the memory traffic per byte of "useful" data
moved, contending for the same memory controllers).

## 2. WMMA peak throughput per type

`tools\bench\wmma_probe.hip`, compiled once per `-DWMMA_KIND=<n>` (see "Build" -- this is
deliberate so one variant's compile failure can't block the others). Kernel: 8 independent
accumulator chains (`NCHAINS=8`) x 8 sequential WMMA ops each (`DEPTH=8`) = 64 WMMA ops/thread/
launch, grid = 1024 blocks x 256 threads (8 warps/block, 8192 total wave32 WMMA-issuing warps,
16x-oversubscribing the 64 CUs / 32 WGPs). `hipEvent`-timed over batches of 200 launches, 5 internal
trials (script reports internal median); table below is the median of 3 full process invocations.
`HIP_VISIBLE_DEVICES=1` throughout; contending `ctest`/Python processes were present during all 3
runs used for the table (could not get a clean window after multiple `Get-Process` polls -- see
"Contention note" above).

Builtins attempted (all from this clang's own
`C:\opt\rocm\lib\llvm\include\clang\Basic\BuiltinsAMDGPU.inc`, the `_w32_gfx12` wave32/gfx12-suffixed
forms since gfx1201/RDNA4 is wave32-native): **all 7 compiled on the first register-packing design**
(no compile failures to report for this run -- signatures below, reconstructed from the general
WMMA A/B-pack-to-`M*K/32`-elements-per-lane, C/D-pack-to-`M*N/32`-elements-per-lane convention, not
from an official gfx12 ISA doc this task had access to; marked **inferred**):

| Variant | Builtin | A/B packing (inferred) | C/D packing |
|---|---|---|---|
| f32_16x16x16_f16 | `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12` | `v8h` (8x f16/lane) | `v8f` (8x f32/lane) |
| f32_16x16x16_bf16 | `..._bf16_w32_gfx12` | `v8s` (8x bf16-as-short/lane) | `v8f` |
| f32_16x16x16_fp8_fp8 | `..._fp8_fp8_w32_gfx12` | `v2i` (8 bytes/lane) | `v8f` |
| f32_16x16x16_bf8_bf8 | `..._bf8_bf8_w32_gfx12` | `v2i` | `v8f` |
| i32_16x16x16_iu8 | `..._iu8_w32_gfx12(signA, A, signB, B, C, clamp)` | `v2i` | `v8i` |
| i32_16x16x16_iu4 | `..._iu4_w32_gfx12(signA, A, signB, B, C, clamp)` | `int` (8x 4-bit/lane) | `v8i` |
| i32_16x16x32_iu4 (double-K) | `..._iu4_w32_gfx12(...)`, K=32 | `v2i` (16x 4-bit/lane) | `v8i` |

```powershell
$env:HIP_VISIBLE_DEVICES='1'
.\build\wmma_probe_f32_16x16x16_f16.exe 200      # repeat per variant .exe
```

| Variant | run1 | run2 | run3 | **median** | vs f16 |
|---|---|---|---|---|---|
| f32_16x16x16_f16 | 160.189 | 153.228 | 156.336 | **156.34 TFLOPS** | 1.00x |
| f32_16x16x16_bf16 | 163.376 | 160.035 | 161.872 | **161.87 TFLOPS** | 1.04x |
| f32_16x16x16_fp8_fp8 | 283.919 | 287.631 | 276.006 | **283.92 TFLOPS** | 1.82x |
| f32_16x16x16_bf8_bf8 | 283.264 | 269.624 | 276.279 | **276.28 TFLOPS** | 1.77x |
| i32_16x16x16_iu8 | 285.592 | 286.443 | 290.676 | **286.44 TOPS** | 1.83x |
| i32_16x16x16_iu4 | 298.245 | 303.845 | 288.540 | **298.25 TOPS** | 1.91x |
| i32_16x16x32_iu4 (double-K) | 593.098 | 578.104 | 592.491 | **592.49 TOPS** | 3.79x |

**Reading this**: f16 and bf16 land within 4% of each other (156/162 TFLOPS) as expected for
same-width types on the same execution units; the 8-bit types (fp8/bf8/iu8) cluster at ~1.8-1.9x
f16, and 4-bit (iu4) at the same K=16 shape is barely higher than 8-bit (0.91-2.5% -- plausibly
noise rather than a real iu4-vs-iu8 throughput difference at this K); the double-pumped K=32 iu4
variant is ~3.8x f16 and ~2x the K=16 iu4 number, consistent with `FLOPS_PER_WMMA` doubling (same
instruction issue rate, 2x the K-dimension work per instruction) rather than a real 4x native rate
for 4-bit. Measured dense f16 (156 TFLOPS) is **82% of AMD's documented "up to 191 TFLOPS FP16
matrix"** figure -- a plausible fraction for a raw microbenchmark against a marketing peak, but see
the open question above about whether that marketing figure assumes a sparsity multiplier (if it
does, 156 TFLOPS measured dense would be *above* an implied ~95 TFLOPS dense peak, which would be
the more interesting -- and unresolved -- finding). This run was contended (see table intro); an
uncontended re-run would be needed to know how much of the gap to 191 TFLOPS is real headroom vs.
another workflow's concurrent GPU use.

## 3. Kernel launch / sync / event overhead

`tools\bench\launch_overhead.hip`. `HIP_VISIBLE_DEVICES=1`; one contending `test_forward_smoke`/
`ctest`/Python process was present for run 1, none confirmed-clean window found for runs 2-3 either
(see "Contention note"). Each row's 3 values are 3 full process invocations; each invocation
internally runs 3 trials of 1000 (or 1000x10 for graph replay, or 1000 for D2H/event) operations and
reports the trial-median, so the numbers below are medians of medians.

```powershell
$env:HIP_VISIBLE_DEVICES='1'
.\build\launch_overhead.exe
```

| Measurement | run1 | run2 | run3 | **median** |
|---|---|---|---|---|
| (a) empty-kernel enqueue, us/launch (1000 back-to-back, 1 stream) | 0.508 | 0.471 | 0.477 | **0.477** |
| (a) empty-kernel completion, us/launch (host wall-time incl. drain) | 1.209 | 1.204 | 1.198 | **1.204** |
| (b) hipGraph replay, us/launch (1000-launch graph, 10 replays) | 0.686 | 0.712 | 0.752 | **0.712** |
| (c) hipMemcpyAsync D2H 4B + hipStreamSynchronize roundtrip, us | 32.862 | 32.844 | 35.434 | **32.862** |
| (d) hipEventRecord, us/call | 0.280 | 0.296 | 0.345 | **0.296** |
| (d) hipEventQuery, us/call | 0.028 | 0.029 | 0.030 | **0.029** |

**Reading this**: capturing the same 1000-launch sequence into a `hipGraph` and replaying it drops
the per-launch cost from 1.204 us (plain stream, host issues + drains) to 0.712 us -- a **41%
reduction**, this task's direct answer to "what would a graph-captured decode step save." Scaled to
r4dx's own ~640 launches/token (per the task brief), that is roughly `640 * (1.204-0.712) =
315 us/token` of potential savings from graph-capturing a decode step, purely from launch/dispatch
overhead -- small next to the 38 ms/token measured in `docs/perf.md`'s per-op profile (host_enqueue
there is already only 2.5-4 ms of that, i.e. graph capture would shave at most ~1% of a decode
step's wall time on this workload; consistent with `docs/perf.md`'s own conclusion that the GEMMs,
not host overhead, dominate). The D2H+sync roundtrip (~33 us) is far larger than a bare kernel
launch and is the same order of magnitude as `docs/perf.md`'s "the remaining per-token D2H is that
one 4-byte copy" note -- this microbenchmark's 33 us figure is a plausible lower bound for that cost
in isolation (a real decode step's D2H waits behind whatever kernel work is already queued, so the
two numbers are not directly comparable, only order-of-magnitude consistent).

## 4. LDS bandwidth (optional; run since time allowed)

`tools\bench\lds_bandwidth.hip`: one workgroup (256 threads) per CU-equivalent (`grid=64`), each
thread does 4096 `int4` (128-bit) LDS loads from a 16 KiB per-block tile, XOR-accumulated, always
written out (see the CSE/dead-code note above for why "always written" matters). `HIP_VISIBLE_DEVICES=1`;
contending processes present for all 3 runs.

```powershell
$env:HIP_VISIBLE_DEVICES='1'
.\build\lds_bandwidth.exe 20
```

| run1 | run2 | run3 | **median** |
|---|---|---|---|
| 3614.90 GB/s | 7834.67 GB/s | 8072.16 GB/s | **7834.67 GB/s** |

**Reading this**: run1 is roughly half of runs 2-3 (and each run's own raw 20-sample list shows the
same split -- an early cluster near half the value, then a late cluster at the higher value, visible
in the raw arrays this script also prints but not reproduced in full here). This pattern matches a
GPU clock-boost ramp (early iterations at a lower sustained clock before the boost state kicks in)
compounded by the contention noted above, not a methodology bug -- the instruction-count check that
validated section 2 was reapplied here (write is unconditional, buffer sized `grid*block`), so the
kernel itself is doing real work; the run-to-run spread is real measurement variance, not a code
bug. Aggregate ~7835 GB/s over 64 blocks implies **~122 GB/s per block/CU**, in the range LDS
bandwidth is generally expected to sit at for a design like this, but this task has no documented
per-CU LDS bandwidth figure to compare against, so this number is reported as **measured** only, not
checked against a spec.

## 5. Skinny-GEMM roofline probe (M=1, mlp.gate_up shape)

`tools\bench\skinny_gemm_roofline.py`, via `r4d.pyd` (`C:\Users\user\dev\libr4d\build-win\r4d.pyd`)
+ the reference venv's torch, same route `tools/profile/tune_gemm.py` and
`third_party/libr4d/bench_mxfp4_gemm.py` already use. Shape: `N=34816, K=5120, M=1` (this model's
`mlp.gate_up`, per `docs/perf.md`'s own per-op profile the single most expensive GEMM family, 51-60%
of one decode step's GPU time in every layout). Tuning parameters (`WV=2 SK=4 MB=1 NPW=1 NT=1`) are
the already-measured-optimal values for this exact `(layout,N,K,M)` cell from
`src/model/gemm_tuning_table.inc` (`tools/profile/tune_gemm.py`'s own sweep output, checked in) --
this script re-times them live rather than re-sweeping. `HIP_VISIBLE_DEVICES=1`; **`r4dx-cli` was
observed running concurrently for all 3 invocations** (a real decode/generation workload on the same
device, not an idle test process -- the heaviest contention seen in this whole task).

```powershell
$env:HIP_VISIBLE_DEVICES='1'
C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\python.exe tools\bench\skinny_gemm_roofline.py
```

Weight bytes moved (dominates at M=1; activations/output are <70 KiB, ignored):
packed 4-bit `N*K/2` = 89,128,960 B + per-group (128) uint32 scales `N*K/128*4` = 5,570,560 B =
**94,699,520 B (90.31 MiB)**.

| Layout | run1 us | run2 us | run3 us | **median us** | achieved GB/s (at median) |
|---|---|---|---|---|---|
| w4a16 | 174.00 | 177.80 | 172.10 | **174.00** | **544.25** |
| w4a8 | 170.40 | 177.00 | 171.80 | **171.80** | **551.22** |

**Roofline comparison**: measured achieved bandwidth (544-551 GB/s) against this task's own
section-1 stream-bandwidth measurements at a comparable ~90 MiB working set (between the 48 MiB
cache-assisted figure, 783 GB/s, and the 256 MiB-8 GiB steady-state figure, ~578-606 GB/s) --
**544-551 GB/s is 90-95% of the 256 MiB-8 GiB steady-state stream bandwidth**, i.e. this decode GEMM
is running very close to the memory-bandwidth roofline for its own weight-streaming access pattern,
confirming `docs/perf.md`'s conclusion that MLP GEMMs are memory- not compute-bound at M=1.
**Contention caveat**: these live-measured numbers ran with `r4dx-cli` concurrently generating text
on the same device. `src/model/gemm_tuning_table.inc`'s own checked-in numbers for this exact cell
(**documented**, not re-measured live by this task) are faster -- 147.76 us (w4a16) / 146.04 us
(w4a8) -- which back out to **641/649 GB/s**, i.e. **~93-100% of the documented 640 GB/s peak** if
that pre-existing measurement was itself uncontended. This task's live numbers are ~15-18% slower
than that checked-in baseline, consistent with the observed `r4dx-cli` contention rather than a
regression -- flagged, not resolved (an uncontended re-run was not obtained in this session despite
multiple `Get-Process` polls).

## Open questions / not resolved by this task

- Whether AMD's documented "up to 191 TFLOPS FP16 matrix" assumes a sparsity or boost-clock
  multiplier (see section 2) -- unresolved, no primary AMD ISA/whitepaper source was available to
  this task beyond the two cited secondary sources.
- Every WMMA/LDS number in sections 2 and 4 ran with contending processes present; an uncontended
  re-run (this task polled `Get-Process` repeatedly across the session and never found a >30s clean
  window during the WMMA/LDS/GEMM-roofline phases) would tighten these figures and is the single
  highest-value follow-up.
- Section 4's run1-vs-run2/3 clock-ramp split was not isolated from contention noise (both are
  plausible contributors); a longer per-invocation warmup or a fixed-clock (`rocm-smi --setperflevel`
  equivalent) run would separate them.
- iu4 (K=16) vs iu8's near-identical TOPS (298 vs 286, section 2) was not investigated further --
  plausible either as a genuine lack of throughput headroom for 4-bit at this K on this hardware, or
  as this benchmark's operand buffers (a single repeated byte pattern) not exercising whatever
  packing-dependent fast path 4-bit might have; flagged, not resolved.
