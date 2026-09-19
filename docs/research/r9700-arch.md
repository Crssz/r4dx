# R9700 (gfx1201, RDNA4) architecture + ISA research

Scope: architecture facts for the AMD Radeon AI PRO R9700 (gfx1201, Navi 48, RDNA4) and the gfx12
WMMA/SWMMAC/scalar-wait/scheduling ISA, gathered from (a) this machine's local ROCm 7.15 / AMD
clang 23 install at `C:\opt\rocm` (ground truth for "does builtin X exist for -mcpu=gfx1201", since
Clang's `BuiltinsAMDGPU.inc` gates every AMDGPU builtin behind an explicit required-target-feature
string per entry) and (b) public AMD/LLVM documentation. Every fact below is tagged
**measured** (read directly off this machine's SDK files or by compiling a probe), **documented**
(a primary published source -- AMD spec page, GPUOpen article, LLVM docs/source), or **inferred**
(reasoned from the above, not directly stated by a single source). No GPU kernels were run for this
document -- SDK header/builtin inspection only.

---

## 1. R9700 board

| Property | Value | Source | Confidence |
|---|---|---|---|
| Codename / process | Navi 48, RDNA4 (gfx1201), same die as RX 9070 XT | Sapphire/XFX product pages, Wikipedia RDNA4 | documented |
| Compute Units | 64 CU (32 WGP) | AMD Radeon AI PRO R9700 datasheet (odinsedge.ai mirror); matches `r4d.pyd`'s own target | documented |
| Stream processors | 4096 (64 CU x 64 ALU-equivalent, i.e. 2x SIMD32/CU) | product pages | documented |
| Ray accelerators | 64 | product pages | documented |
| 2nd-gen AI accelerators | 128 (2 per CU -- i.e. one per SIMD32) | AMD blog "AMD Radeon AI PRO R9700 to be available in workstation" | documented |
| Boost clock | up to 2920 MHz (base 1620 MHz, game clock 2350 MHz) | gpupoet.com / cputronic.com spec aggregation | documented |
| FP32 vector TFLOPS | 47.8 TFLOPS | AMD datasheet, corroborated by multiple spec aggregators | documented |
| FP16 vector TFLOPS | ~95.7 TFLOPS (2x FP32, standard RDNA rate-2 packed math) | spec aggregators | documented |
| FP16 matrix (WMMA, dense) | ~191 TFLOPS | spec aggregators, consistent with "2x throughput/CU vs RDNA3 for FP16" claim below | documented |
| VRAM | 32 GiB GDDR6, 256-bit bus, ~20 Gbps/pin -> 640 GB/s peak | AMD datasheet; `hipInfo` on this machine reports 31.86 GiB usable (`docs/perf.md`) | documented + measured (usable capacity) |
| Infinity Cache (L3) | 64 MiB | AMD datasheet, RDNA4 Infinity Cache press coverage (videocardz/tweaktown/igorslab, pre-launch rumors that matched the shipped 256-bit SKU) | documented |
| L2 cache | 8 MiB, GPU-wide, ~100 cycle latency | zolotukhin.ai AMD RDNA3/RDNA4 GPU reference (cross-checked against ROCm hardware tables); Navi 48 (RDNA4) doubling Navi 31's 6 MiB despite being a smaller die | documented |
| L1 (per shader-array) | 128 KiB | same source | documented |
| L0 vector cache (per CU) | 32 KiB, ~30 cycle latency | same source | documented |
| LDS | 64 KiB/CU in CU mode, 128 KiB/WGP in WGP mode (2 CUs share the pool) | AMD RDNA4 ISA Reference Guide (docs.amd.com), cross-checked against RDNA whitepaper's WGP description | documented |
| Wavefront model | native wave32 (RDNA), wave64 also encodable; **WMMA/SWMMAC on gfx12 require wave32** | AMD RDNA4 ISA Reference Guide; zolotukhin.ai inference reference | documented |
| PCIe | Gen5 x16 | AMD datasheet | documented |
| TBP | 300 W (750 W PSU recommended, 12V-2x6 connector) | AMD datasheet | documented |
| Form factor | dual-slot, blower | AMD datasheet | documented |

**AI TOPS (AMD-stated, dense vs. structured-sparse)**, from AMD's own R9700 marketing figures
(wccftech "4x more TOPS", spec aggregators cross-referencing the AMD datasheet):

| Format | Dense | 2:1 structured sparse |
|---|---|---|
| FP16/BF16 matrix | ~191 TFLOPS | ~383 TFLOPS |
| FP8 matrix | ~383 TFLOPS | ~765 TFLOPS |
| INT8 matrix | ~383 TOPS | ~765 TOPS |
| INT4 matrix | ~765 TOPS | **1531 TOPS** (AMD's headline number) |

Confidence: **documented**, but these are vendor marketing figures aggregated by third-party spec
sites, not from a single AMD whitepaper table with per-op breakdown -- treat the ratios (2x/CU FP16
vs RDNA3, 4x/CU INT8 vs RDNA3, sparsity giving up to 2x on top of dense) as the load-bearing claim,
sourced to AMD's own blog language ("RDNA 4 architecture offers up to 2x throughput per Compute
Unit for FP16 datatypes and 4x for INT8 datatypes... sparsity acceleration enables up to 4x FP16 and
8x INT8 throughput improvements via structured sparsity support" -- AMD Radeon AI PRO R9700
workstation blog post); the absolute TOPS numbers are **inferred** by multiplying those ratios onto
this specific 64-CU/2920 MHz SKU by the aggregator sites, not independently re-derived here.

Sources: [Sapphire R9700 product page](https://www.sapphiretech.com/en/commercial/radeon-ai-pro-r9700), [XFX R9700 product page](https://www.xfxforce.com/shop/xfx-amd-radeon-ai-pro-r9700-32gb-gddr6-4xdp-amd-rdna-tm-4), [AMD R9700 datasheet PDF](https://odinsedge.ai/wp-content/uploads/2025/08/AMD-Radeon-AI-PRO-R9700-Datasheet_v01.pdf), [AMD blog: R9700 workstation availability](https://www.amd.com/en/blogs/2025/amd-radeon-ai-pro-r9700-to-be-available-in-workstation.html), [wccftech TOPS comparison](https://wccftech.com/amd-radeon-ai-pro-r9700-gpu-4x-more-tops-2x-ai-performance-vs-radeon-pro-w7800/), [WareDB AI perf breakdown](https://waredb.com/processor/amd-radeon-ai-pro-r9700), [zolotukhin.ai RDNA3/RDNA4 GPU reference](https://zolotukhin.ai/zinc/docs/amd-gpu-reference/), [RDNA4 ISA Reference Guide (AMD docs.amd.com)](https://docs.amd.com/api/khub/documents/uQpkEvk3pv~kfAb2x~j4uw/content), [gpupoet.com R9700 specs](https://gpupoet.com/gpu/learn/card/amd-radeon-ai-pro-r9700), `docs/perf.md` (this repo, measured 31.86 GiB usable VRAM on HIP device 1).

Cache-hierarchy caveat: the L0/L1/L2/Infinity-Cache sizes above are corroborated by a
community-compiled reference (zolotukhin.ai) that states it cross-checked against AMD's own ROCm
hardware tables, not lifted directly from a numbered table in the AMD RDNA4 ISA Reference Guide
text this session could fetch -- flagged as **documented, secondary-sourced** rather than
primary-sourced, pending a direct read of the AMD PDF's cache section.

---

## 2. gfx12 (RDNA4) WMMA / SWMMAC ISA

**Method**: rather than trust the monolithic `llvm/IR/IntrinsicsAMDGPU.h` intrinsic enum (which
lists *every* AMDGPU generation's intrinsics in one file with no per-target filtering), the
authoritative per-target answer is `clang/Basic/BuiltinsAMDGPU.inc`, where every
`__builtin_amdgcn_*` entry carries an explicit required-target-feature string as its 4th field.
`grep`ping `C:\opt\rocm\lib\llvm\include\clang\Basic\BuiltinsAMDGPU.inc` (AMD clang 23.0.0git,
ROCm 7.15) gives a builtin-by-builtin ground truth for what `-mcpu=gfx1201` actually accepts, which
is what this section is built from. `--offload-arch=gfx1201 --offload-arch=gfx1250` probe compiles
(`clang -x hip --offload-arch=<x> -c ... -nogpuinc -nogpulib`) independently confirmed the front end
parses both targets' builtins identically at the Sema arg-count level (target-feature rejection
happens later, at Sema's required-feature check / CodeGen, not at the arg-count check) -- so the
`BuiltinsAMDGPU.inc` feature-string field, not a compile-and-see probe, is the reliable signal used
below.

### 2.1 Dense WMMA (`v_wmma_*`), gfx12/RDNA4-available (feature `wmma-128b-insts` or `swmmac-gfx1200-insts`)

The `_w32_gfx12` / `_w64_gfx12`-suffixed builtins are the RDNA4-native forms (AMD's GPUOpen guide:
*"There are a few WMMA intrinsics for AMD RDNA 4 architecture GPUs, which have a postfix `_gfx12`
that did not exist for WMMA intrinsics on RDNA 3 (gfx11)... In RDNA 3, some elements needed
duplication for A and B matrices, but this was removed in RDNA 4, simplifying the VGPR layout"*).
Plain (non-`_gfx12`) `_w32`/`_w64` builtins gated `wmma-256b-insts` are the gfx11 (RDNA3) legacy
encoding and are a **different, wider VGPR-fragment ISA** not used on gfx1201 (`amd_matrix_instruction_calculator`'s `--architecture rdna4` mode targets the `_gfx12` set specifically). All shapes below are `16x16x16` (K=16) or, for `iu4`, also `16x16x32` (K=32) -- gfx12 added a K=32 4-bit-input variant that gfx11 does not have (per LLVM/MLIR gfx11-vs-gfx12 WMMA dimension docs found via web search, corroborated by the local builtin list: `wmma_i32_16x16x32_iu4_w32_gfx12` exists, `wmma_i32_16x16x32_iu8_*_gfx12` does **not**).

| Instruction (builtin, gfx12 form) | Shape | A/B dtype | C/D (accum) dtype | Required feature (local SDK) |
|---|---|---|---|---|
| `wmma_f32_16x16x16_f16_w32_gfx12` | 16x16x16 | f16 | f32 | `wmma-128b-insts,wavefrontsize32` |
| `wmma_f32_16x16x16_bf16_w32_gfx12` | 16x16x16 | bf16 | f32 | `wmma-128b-insts,wavefrontsize32` |
| `wmma_f16_16x16x16_f16_w32_gfx12` | 16x16x16 | f16 | f16 | `wmma-128b-insts,wavefrontsize32` |
| `wmma_bf16_16x16x16_bf16_w32_gfx12` | 16x16x16 | bf16 | bf16 | `wmma-128b-insts,wavefrontsize32` |
| `wmma_f32_16x16x16_fp8_fp8_w32_gfx12` | 16x16x16 | fp8 (OCP e4m3) x fp8 | f32 | `wmma-128b-insts,wavefrontsize32` |
| `wmma_f32_16x16x16_fp8_bf8_w32_gfx12` | 16x16x16 | fp8 x bf8 (mixed) | f32 | `wmma-128b-insts,wavefrontsize32` |
| `wmma_f32_16x16x16_bf8_fp8_w32_gfx12` | 16x16x16 | bf8 x fp8 (mixed) | f32 | `wmma-128b-insts,wavefrontsize32` |
| `wmma_f32_16x16x16_bf8_bf8_w32_gfx12` | 16x16x16 | bf8 (OCP e5m2) x bf8 | f32 | `wmma-128b-insts,wavefrontsize32` |
| `wmma_i32_16x16x16_iu8_w32_gfx12` | 16x16x16 | iu8 (signed/unsigned int8) | i32 | `wmma-128b-insts,wavefrontsize32` |
| `wmma_i32_16x16x16_iu4_w32_gfx12` | 16x16x16 | iu4 | i32 | `wmma-128b-insts,wavefrontsize32` |
| `wmma_i32_16x16x32_iu4_w32_gfx12` | **16x16x32** | iu4 | i32 | `wmma-128b-insts,wavefrontsize32` -- gfx12-only K=32 4-bit variant |

Every one of the above has a matching `_w64_gfx12` sibling gated `wavefrontsize64` instead of
`wavefrontsize32` -- i.e. the instruction *encodes* for wave64 too, but AMD's own guidance (and the
zolotukhin.ai inference reference) is that **WMMA on gfx12 should be issued from wave32 kernels
only**; the `_w64_gfx12` forms exist for completeness/compatibility, not as the recommended path.
(This matches `libr4d`'s own note that its wave32 WMMA fragment layout assumption is load-bearing
throughout its kernels -- `third_party/libr4d/README.md`, "Notes for anyone reading the kernels".)

**Accumulator note**: gfx12 f16/bf16-input WMMA can accumulate into either the *same* narrow type
(`f16_16x16x16_f16` -> f16 out) or widen to f32 (`f32_16x16x16_f16` -> f32 out); fp8/bf8-input WMMA
on gfx12 only widens to f32 (there is no `f16_16x16x16_fp8_fp8` on gfx12 the way gfx1250 later adds
one -- see 2.3). iu8/iu4 always accumulate to i32.

### 2.2 SWMMAC (sparse WMMA, `v_swmmac_*`), gfx12/RDNA4-available (feature `swmmac-gfx1200-insts`)

SWMMAC takes a 2:1-structured-sparse A operand (half the K elements are physically stored, plus a
compression/index operand) and produces the same-shape result as a dense WMMA at **twice the K**
of the WMMA table above -- i.e. `16x16x32` for the f16/bf16/fp8/bf8 8-/16-bit-input families and
`16x16x64` for the iu4 family, at the throughput cost of one *dense* `16x16x16`/`16x16x32` MMA (this
is the "up to 2x from structured sparsity" AMD cites, see section 1).

| Instruction (builtin) | Shape | A(compressed)/B dtype | Accum | Required feature |
|---|---|---|---|---|
| `swmmac_f32_16x16x32_f16_w32` | 16x16x32 | f16 | f32 | `swmmac-gfx1200-insts,wavefrontsize32` |
| `swmmac_f32_16x16x32_bf16_w32` | 16x16x32 | bf16 | f32 | `swmmac-gfx1200-insts,wavefrontsize32` |
| `swmmac_f16_16x16x32_f16_w32` | 16x16x32 | f16 | f16 | `swmmac-gfx1200-insts,wavefrontsize32` |
| `swmmac_bf16_16x16x32_bf16_w32` | 16x16x32 | bf16 | bf16 | `swmmac-gfx1200-insts,wavefrontsize32` |
| `swmmac_f32_16x16x32_fp8_fp8_w32` | 16x16x32 | fp8 x fp8 | f32 | `swmmac-gfx1200-insts,wavefrontsize32` |
| `swmmac_f32_16x16x32_fp8_bf8_w32` | 16x16x32 | fp8 x bf8 | f32 | `swmmac-gfx1200-insts,wavefrontsize32` |
| `swmmac_f32_16x16x32_bf8_fp8_w32` | 16x16x32 | bf8 x fp8 | f32 | `swmmac-gfx1200-insts,wavefrontsize32` |
| `swmmac_f32_16x16x32_bf8_bf8_w32` | 16x16x32 | bf8 x bf8 | f32 | `swmmac-gfx1200-insts,wavefrontsize32` |
| `swmmac_i32_16x16x32_iu8_w32` | 16x16x32 | iu8 | i32 | `swmmac-gfx1200-insts,wavefrontsize32` |
| `swmmac_i32_16x16x32_iu4_w32` | 16x16x32 | iu4 | i32 | `swmmac-gfx1200-insts,wavefrontsize32` |
| `swmmac_i32_16x16x64_iu4_w32` | **16x16x64** | iu4 | i32 | `swmmac-gfx1200-insts,wavefrontsize32` |

All have `_w64` siblings, same wave32-preferred caveat as 2.1. Every one of these is at
`swmmac-gfx1200-insts` -- **not** `wmma-128b-insts` -- confirming SWMMAC is gated on a separate
target-feature flag from dense WMMA in this compiler, though both resolve true for `-mcpu=gfx1201`.

Source (methodology + terminology): [GPUOpen "Using the Matrix Cores of AMD RDNA 4 architecture GPUs"](https://gpuopen.com/learn/using_matrix_core_amd_rdna4/), [ROCm/amd_matrix_instruction_calculator](https://github.com/ROCm/amd_matrix_instruction_calculator), [rdna4-wmma-guide (community, cites 40.8 TFLOPS fused MXFP4 GEMM measured on an R9700)](https://github.com/JohnTDI-cpu/rdna4-wmma-guide). Shapes/dtypes/feature-gating table: **measured** directly from `C:\opt\rocm\lib\llvm\include\clang\Basic\BuiltinsAMDGPU.inc` (ROCm 7.15 / AMD clang 23.0.0git, this machine), grep'd for `BI__builtin_amdgcn_(wmma|swmmac)` and cross-referenced against each entry's `Builtin::Info{...}` required-feature field in the same file.

### 2.3 What is gfx1250-only (confirmed absent from gfx1201/RDNA4)

Every builtin below is present in the *compiler's* global builtin table (it will parse and
type-check for `-mcpu=gfx1201` the same as for `-mcpu=gfx1250`, since Sema's arg-count/type checks
run before target-feature gating) but carries `gfx1250-insts` (or another gfx1250-exclusive feature
string) as its required feature -- i.e. it is **rejected at CodeGen/ISel time for `-mcpu=gfx1201`**,
confirmed by direct inspection of `BuiltinsAMDGPU.inc`'s per-builtin required-feature field, not by
running a probe compile through to the backend (this machine has no `llc.exe` in
`C:\opt\rocm\lib\llvm\bin`; the feature-string method is authoritative without it since Clang's own
`SemaChecking` enforces this table before it ever reaches LLVM ISel):

- **Scale-aware MX-format WMMA** (the thing this doc was specifically asked to confirm/deny):
  `wmma_scale_f32_16x16x128_f8f6f4`, `wmma_scale16_f32_16x16x128_f8f6f4`,
  `wmma_scale_f32_32x16x128_f4`, `wmma_scale16_f32_32x16x128_f4`, and the unscaled-but-still-new
  `wmma_f32_16x16x128_f8f6f4` and `wmma_f32_32x16x128_f4` -- **all** gated `gfx1250-insts`. **There
  is no fp4 (e2m1), fp6 (e2m3/e3m2), or MX-scale-aware WMMA of any kind on gfx1201.** This directly
  answers the task's question: gfx12/RDNA4 has **zero** hardware WMMA support for fp4/fp6 or
  block-scaled (MX) operands, full stop -- MXFP4 inference on this card (`r4dx`'s `mxfp4` layout,
  `r4d_gemm_mxfp4a8_nt_m64`) necessarily **dequantizes to fp8/f16 in software before the WMMA**,
  it cannot issue a native 4-bit-with-scale matrix instruction the way gfx1250/CDNA4 can.
- **Wider-K WMMA/SWMMAC for 8-/16-bit types** (K=64 for f16/bf16, K=128 for fp8/bf8, on top of the
  gfx12 K=16/32 ceiling): `wmma_f16_16x16x32_f16`, `wmma_f16_16x16x64_{bf8,fp8}_{bf8,fp8}`,
  `wmma_f16_16x16x128_{bf8,fp8}_{bf8,fp8}`, `wmma_f32_16x16x32_{f16,bf16}`,
  `wmma_f32_16x16x64_{bf8,fp8}_{bf8,fp8}`, `wmma_f32_16x16x128_{bf8,fp8}_{bf8,fp8}`,
  `wmma_bf16_16x16x32_bf16`, `wmma_bf16f32_16x16x32_bf16`, `wmma_f32_16x16x4_f32` (a *plain f32*
  WMMA, new on gfx1250), `wmma_i32_16x16x64_iu8` -- all `gfx1250-insts`. The corresponding
  `swmmac_*_16x16x64` / `swmmac_*_16x16x128` sparse forms are likewise `gfx1250-insts`-only.
- **Scaled fp8/bf8/fp4/fp6/bf6 conversions** (`__builtin_amdgcn_cvt_scalef32_*`, ~80 builtins
  covering pk8/pk16/pk32-wide packed conversions plus stochastic-rounding `_sr_` variants): every
  single one is gated to one of `gfx1250-insts`, `bf8-cvt-scale-insts`, `fp8-cvt-scale-insts`,
  `fp4-cvt-scale-insts`, `fp6bf6-cvt-scale-insts`, or `f16bf16-to-fp6bf6-cvt-scale-insts` -- **none**
  of these feature strings are the plain `gfx12-insts`/`wmma-128b-insts` families gfx1201 satisfies.
  Since fp4/fp6/bf6 as *data types* don't exist in gfx1201 hardware at all (no register format, no
  WMMA operand type -- see above), this is expected, not a gap: there is nothing on gfx1201 for a
  scaled fp4/fp6 conversion to feed.
- **`ds_read_tr*`/`ds_load_tr8/16*`** (in-register LDS transpose reads): `ds_read_tr4/6/8/16_*` are
  all gated `gfx950-insts` (CDNA-only); `ds_load_tr8_b64`/`ds_load_tr16_b128_*` are `gfx1250-insts`.
  Only `ds_load_tr4_b64_v2i32` / `ds_load_tr6_b96_v3i32` are gated `transpose-load-f4f6-insts`
  (still not satisfied on gfx1201, since that feature is tied to the fp4/fp6 transpose-load path
  gfx1201 also lacks). **gfx1201 has no LDS transpose-read instruction of any width.**
- **Split async/tensor wait counters**: `s_wait_asynccnt`, `s_wait_tensorcnt` -- `gfx1250-insts`
  (gfx1250 adds `ASYNCcnt`/`TENSORcnt` for its new async-LDS and tensor-load instructions, per
  chipsandcheese's GFX1250 analysis; gfx1201 has neither the instructions nor the counters).

Source (local SDK, measured): `C:\opt\rocm\lib\llvm\include\clang\Basic\BuiltinsAMDGPU.inc` (grep'd
line ranges ~1470-1475 for the IR-intrinsic enum cross-reference in
`llvm/IR/IntrinsicsAMDGPU.h`, and the `Builtin::Info{...}` table around lines 2960-2990 (WMMA),
2844-2879 (SWMMAC), 2279-2355 (cvt_scalef32), 2402-2478 (transpose loads), 2781-2785 (split wait
counters) for the required-feature strings). Corroborating documented source for the
fp4/fp6/MX-scale conclusion specifically: [zolotukhin.ai "FP4 vs FP8 for RDNA4 Inference"](https://zolotukhin.ai/blog/2026-05-09-the-fp4-wave-breaks-at-rdna4-and-fp8-wmma-already-does-what-local-qwen3-needs/) (independently reaches the same "no native fp4 WMMA on RDNA4" conclusion from a different angle).

### 2.4 Dot-product instructions (`v_dot*`)

gfx12 keeps the same VOP3P dot-product set as gfx11 (both defined via the shared
`VOP3P_Real_gfx11_gfx12` tablegen class per LLVM's AMDGPU backend):

| Instruction | Operands | Accumulate |
|---|---|---|
| `v_dot2_f32_f16` | 2x f16 | f32 |
| `v_dot2_f32_bf16` | 2x bf16 | f32 |
| `v_dot4_i32_iu8` (alias `v_dot4_i32_i8`) | 4x (signed/unsigned) int8 | i32 |
| `v_dot4_u32_u8` | 4x uint8 | u32 |
| `v_dot4_f32_fp8_fp8` / `_fp8_bf8` / `_bf8_fp8` / `_bf8_bf8` | 4x 8-bit float, all 4 fp8/bf8 pairings | f32 |
| `v_dot8_i32_iu4` (alias `v_dot8_i32_i4`) | 8x (signed/unsigned) int4 | i32 |
| `v_dot8_u32_u4` | 8x uint4 | u32 |

These are scalar-lane VALU dot products (one dot product per SIMD lane per instruction, unlike
WMMA's whole-wave matrix tile) -- useful for irregular/small-K reductions the WMMA tile shapes don't
fit, not a competitor to WMMA for GEMM. Source: [LLVM "Syntax of GFX12 Instructions"](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX12.html), corroborated by [LLVM PR #118997 (v_dot4_i32_i8/v_dot8_i32_i4 GFX11+ aliases)](https://github.com/llvm/llvm-project/pull/118997). Confidence: documented.

### 2.5 FP8/BF8 conversion instructions (non-scaled, gfx1201-available)

| Builtin | Direction | Required feature (local SDK) | gfx1201? |
|---|---|---|---|
| `cvt_pk_f32_fp8` | packed fp8 -> 2x f32 | `fp8-conversion-insts` | yes |
| `cvt_pk_fp8_f32` | 2x f32 -> packed fp8 | `fp8-conversion-insts` | yes |
| `cvt_pk_bf8_f32` | 2x f32 -> packed bf8 | `fp8-conversion-insts` | yes |
| `cvt_pk_bf8_f16` | packed f16 -> packed bf8 | `gfx1250-insts` | **no** |
| `cvt_pk_fp8_f32_e5m3` | 2x f32 -> packed fp8 (e5m3 variant) | `fp8e5m3-insts` | **no** (e5m3 format is not gfx1201) |

The plain `fp8-conversion-insts` feature (not `gfx1250-insts`) is satisfied on gfx1201 -- **RDNA4
does have native HW `v_cvt_pk_f32_fp8`/`v_cvt_pk_fp8_f32`** for OCP e4m3 -- but with a known compiler
bug on this exact target: `llc` selects the `_e32` encoding for `word_sel=false` on gfx12, but
`llvm-mc` rejects assembling that `_e32` form for the same target (LLVM issue #224032, open as of
this research), forcing frameworks like Triton to work around it by shifting the packed word into
the high half and forcing the `_e64` encoding. **No scaled (`v_cvt_scalef32_*`) fp8 conversion
exists on gfx1201** -- those are gfx950(CDNA4)/gfx1250-only (see 2.3); Triton's AMD backend
explicitly falls back to a *software* OCP-fp8-to-bf16 path on the RDNA4 family for this reason.
Sources: [LLVM issue #224032 (fp8 e32/e64 encoding mismatch)](https://github.com/llvm/llvm-project/issues/224032), [Triton issue #11497 (RDNA4 fp8 upcast path, word_sel workaround)](https://github.com/triton-lang/triton/issues/11497). Feature-gating table: measured, same `BuiltinsAMDGPU.inc` file.

### 2.6 bf16 packing gap -- confirmed

`libr4d/README.md` states: *"gfx1201 has no `v_cvt_pk_bf16_f32`, no direct-to-LDS, and no
`ds_read_b64_tr_b16`. Packing bf16 is software, and the cheapest form is two adds plus a
`v_perm_b32`."* Independently confirmed against this machine's SDK: `grep`ping
`BuiltinsAMDGPU.inc` for `cvt_pk_bf16` finds **no** `__builtin_amdgcn_cvt_pk_bf16_f32` entry at all
(the builtin simply does not exist in this compiler's table for any target, not even gated off --
there is no hardware instruction to build a builtin around). Similarly `ds_read_b64_tr_b16` doesn't
exist as a named builtin; the closest family (`ds_read_tr16_b64_v4bf16` etc.) is gated
`gfx950-insts` per 2.3, confirming it is unavailable on gfx1201 regardless of exact mnemonic.
Confidence: **measured** (absence in the local builtin table), consistent with the libr4d README's
own claim.

### 2.7 Transpose loads: `global_load_tr` (gfx1201-available), `ds_read_tr`/`ds_load_tr` (gfx1201-unavailable)

Contrary to the `ds_read_tr*` LDS-side gap in 2.3, **global-memory transpose loads exist on
gfx1201**:

| Builtin | Shape | dtype | Required feature |
|---|---|---|---|
| `global_load_tr_b64_v2i32` | b64 (2x i32) | generic 32-bit | `gfx12-insts,wavefrontsize32` |
| `global_load_tr_b64_i32` | b64 (1x i32, wave64 form) | generic 32-bit | `gfx12-insts,wavefrontsize64` |
| `global_load_tr_b128_v8{bf16,f16,i16}` | b128 (8x 16-bit) | bf16/f16/i16 | `gfx12-insts,wavefrontsize32` |
| `global_load_tr_b128_v4{bf16,f16,i16}` | b128 (4x 16-bit, wave64 form) | bf16/f16/i16 | `gfx12-insts,wavefrontsize64` |

`global_load_tr4_b64`/`tr6_b96`/`tr8_b64`/`tr16_b128` (the fp4/fp6-oriented narrower transpose
widths) are gated `transpose-load-f4f6-insts` or `gfx1250-insts` respectively -- **not** available on
gfx1201, consistent with gfx1201 having no fp4/fp6 hardware path at all (2.3). So: gfx1201 can
transpose-load a 16-bit-element (bf16/f16/int16) tile straight from global memory into WMMA B-matrix
layout without a software transpose or an LDS round-trip, but only at the b64/b128, 16-bit-element
granularity -- not the narrower fp4/fp6/fp8 transpose widths gfx1250 adds. This lines up with a
(community, unverified-against-the-official-PDF) claim seen during research that RDNA4 "has ...
`global_load_tr` on gfx1200+" for exactly this purpose, now confirmed against the compiler's own
feature table rather than that secondary source. Source: measured, `BuiltinsAMDGPU.inc` (same file,
`global_load_tr*` entries, ~lines 2465-2478).

### 2.8 Lane-permute instructions

| Builtin | Required feature | gfx1201? |
|---|---|---|
| `permlane16` | `gfx10-insts` | yes (inherited since gfx10) |
| `permlanex16` | `gfx10-insts` | yes |
| `permlane64` | `gfx11-insts` | yes |
| `permlane16_var` | `gfx12-insts` | **yes, gfx12-new** |
| `permlanex16_var` | `gfx12-insts` | **yes, gfx12-new** |
| `permlane16_swap` | `permlane16-swap` (separate feature string; not confirmed satisfied on gfx1201 in this pass) | not confirmed |

`permlane16_var`/`permlanex16_var` (variable/per-lane-indexed cross-lane permute, vs. the
fixed-pattern `permlane16`/`permlanex16`) are new gfx12 instructions -- useful for the kind of
irregular cross-lane shuffles a WMMA-fragment repack or a paged-attention gather needs. Source:
measured, `BuiltinsAMDGPU.inc` lines ~2687-2698.

### 2.9 `s_wait_*` split counters (gfx12 ISA change from the unified `s_waitcnt`)

Starting at gfx12, AMD's ISA replaced the single `s_waitcnt` (VMcnt/EXPcnt/LGKMcnt) encoding with
separate per-resource wait instructions (`hasExtendedWaitCounts()` in LLVM, true for GFX12+):
`s_wait_loadcnt`, `s_wait_storecnt`, `s_wait_samplecnt`, `s_wait_bvhcnt`, `s_wait_dscnt`,
`s_wait_expcnt`, `s_wait_kmcnt`, plus fused forms `s_wait_loadcnt_dscnt` /
`s_wait_storecnt_dscnt`. These are real gfx12 ISA mnemonics (assembler-level, per LLVM's
`AMDGPUAsmGFX12` docs) rather than individual Clang builtins -- the generic
`__builtin_amdgcn_s_waitcnt` builtin (present, unversioned/target-feature-empty in
`BuiltinsAMDGPU.inc`) still exists and the compiler backend picks the correct split-counter
encoding automatically for gfx12+ targets; a kernel author writing intrinsics does not need to name
`s_wait_loadcnt` directly. `s_wait_asynccnt`/`s_wait_tensorcnt` are gfx1250-only additions (2.3) --
gfx1201 has no async-LDS or tensor-load pipeline to wait on. Source: [LLVM AMDGPUAsmGFX12 syntax docs](https://llvm.org/docs/AMDGPU/AMDGPUAsmGFX12.html), [chipsandcheese "Scrying the AMD GFX1250 LLVM Tea Leaves"](https://chipsandcheese.com/p/scrying-the-amd-gfx1250-llvm-tea) (contrasts gfx12 vs gfx1250 counter sets). Builtin-table cross-check: measured, `BuiltinsAMDGPU.inc` lines 2781-2785.

### 2.10 Scheduling builtins (`sched_group_barrier`, `iglp_opt`)

`__builtin_amdgcn_sched_group_barrier` and `__builtin_amdgcn_iglp_opt` both carry an **empty**
required-feature string in `BuiltinsAMDGPU.inc` (feature index `0`) -- i.e. they are
architecture-generic compiler pseudo-ops (they lower to LLVM scheduling metadata, not a real
instruction), available on gfx1201 same as everywhere else. This matches `libr4d/README.md`'s own
usage note: *"LLVM single-buffers LDS and drains before every WMMA unless you tell it not to.
`__builtin_amdgcn_sched_group_barrier` is the fix, and the granularity of the groups matters far
more than their contents -- coarser is better, up to the point where the pattern asks for more
outstanding loads than the hardware can hold."* Source: measured (`BuiltinsAMDGPU.inc` lines
2489, 2794) + `third_party/libr4d/README.md` (this repo).

---

## 3. Memory hierarchy behaviour relevant to LLM decode

- **Infinity Cache cannot help a weights-streamed-once-per-token workload.** `r4dx`'s own profile
  (`docs/perf.md`) puts ~51-60% of every decode step in the MLP GEMMs and ~85+% of total step time in
  MLP+GDN GEMMs combined -- these read a *different* ~14 GiB of 4-bit weight per token (every
  parameter touched exactly once per forward pass), so each byte is a cold miss into a 64 MiB
  Infinity Cache / 8 MiB L2 no matter the access order: there is no temporal reuse of weight bytes
  within a single token's forward pass for Infinity Cache to capture. Infinity Cache and L2 *can*
  help the pieces of state genuinely re-read within a step or across nearby steps -- the GDN
  recurrent state (`fp32[48,128,128]` per sequence, re-read and re-written every decode token,
  `docs/architecture.md` "GDN state"), the small per-layer control arrays r4dx already caches on
  device (`GdnControlCache`, `docs/perf.md` "Bugs found... item 3"), and KV cache blocks a
  paged-attention layer revisits within one query -- but none of that is the throughput-dominant
  cost this model pays per token. This is **inferred** from this repo's own measured per-op profile
  plus the basic streamed-weights-vs-cached-state distinction, not a directly cited AMD statement.
- **VRAM bandwidth, not Infinity Cache hit rate, is the decode-time ceiling for the streamed-weight
  path.** At ~14 GiB of weight read per token and the measured ~30-33 tok/s ceiling for the
  quantized layouts (`docs/perf.md`), the implied weight-read bandwidth is in the few-hundred-GB/s
  range, consistent with (well under) the board's 640 GB/s peak GDDR6 bandwidth once GEMM
  efficiency and kernel-launch/host overhead (2.5-9.6% of wall time per `docs/perf.md`'s per-op
  profile) are accounted for. **inferred**, arithmetic from this repo's own measured numbers, not an
  independently measured bandwidth-utilization percentage.
- **Fine-grained vs. coarse-grained memory on Windows**: ROCm distinguishes coherent/fine-grained
  memory (supports in-kernel atomics visible to host/peer GPUs, at the cost of leaving those
  allocations GPU-uncached) from non-coherent/coarse-grained memory (GPU-cacheable, only
  synchronized at command boundaries) via `hipHostMalloc`'s coherent/non-coherent flags and
  `hipExtMallocWithFlags`'s `hipDeviceMallocFinegrained`/`hipDeviceMallocUncached`/
  `hipDeviceMallocDefault` flags. `r4dx` does not currently request fine-grained memory anywhere in
  its own code (`src/core` buffer/arena plumbing uses plain `hipMalloc`) -- the coherent-memory
  tradeoff is a documented capability of the platform, not something this codebase exercises today.
  **documented** (HIP memory-management docs) + **measured** (absence of any
  `hipDeviceMallocFinegrained`/`Uncached` call in this repo, confirmed by the earlier `Grep` of
  `third_party/libr4d` and `src/core` during this research pass turning up no such call).

Sources: `docs/perf.md`, `docs/architecture.md` (this repo, measured), [HIP memory management docs (coherent vs. non-coherent)](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html).

---

## 4. HIP on Windows (WDDM) specifics

- **hipGraph**: public ROCm docs found during this research did not state an explicit
  Windows-vs-Linux support matrix entry for `hipGraph` specifically (the Windows/Linux parity table
  in AMD's HIP SDK for Windows release notes calls out MIOpen/MIGraphX/PyTorch/TensorFlow and
  `CMake HIP Language` as "Not Available"/"Unsupported" on Windows, but does not mention hipGraph by
  name either way). **open question**, not resolved by this pass -- flagged below.
- **Kernel launch latency**: this repo already has a direct, on-hardware measurement rather than a
  vendor number: `docs/perf.md`'s per-op profile reports `host_enqueue` (pure CPU launch-issue
  overhead for one full 64-layer decode step, ~140 kernel launches) at **2.48-4.25 ms** across the
  four layouts (6.5-9.6% of a 38-44 ms decode step for the quantized layouts, 0.9% of a 737 ms step
  for bf16) -- i.e. Windows WDDM kernel-launch overhead on this exact card/driver is on the order of
  **tens of microseconds per launch**, not milliseconds, and is not the decode-time bottleneck (the
  GEMMs are, per that same profile). **measured**, this repo, HIP device 1, 2026-09-19.
- **hipMemcpyAsync / stream ordering**: `docs/perf.md`'s "Bugs found and fixed" section documents
  two real bugs from this exact codebase where a plain (non-stream-argument) `hipMemcpy` against a
  buffer produced/consumed by a `hipStreamNonBlocking` stream was **not** implicitly ordered against
  that stream's in-flight work, causing an intermittent `HIP error 719` and a garbled-decode race
  respectively -- both fixed by routing through `hipMemcpyAsync(..., stream)` +
  `hipStreamSynchronize(stream)` at the right point. This is a directly measured confirmation (on
  this Windows/WDDM + ROCm 7.15 stack) that HIP's `hipStreamNonBlocking` semantics behave as
  documented (no implicit sync with the null/legacy stream) rather than silently falling back to
  Linux's occasionally-more-forgiving scheduling. **measured**, this repo.
- **hipHostMalloc / pinned memory**: documented to give faster H2D/D2H transfers and to be required
  for a *truly* async `hipMemcpyAsync` (an unpinned host pointer forces the copy to run
  synchronously regardless of the async call) -- generic HIP behavior, not Windows-specific in the
  docs found. `r4dx` does not currently use `hipHostMalloc` anywhere in `src/core`/`src/model`
  (confirmed absent during this research pass); its D2H copies (e.g. the single 4-byte argmax
  readback added in the "Host overhead" perf pass, `docs/perf.md` item 5) go through plain
  device-buffer `CopyToHost`, which is a candidate future optimization target the perf.md profile
  doesn't currently flag as bottleneck-relevant (finish_wait dominates wall time but is *blocked
  time*, not *extra* work, per that doc's own explanation).
- **`hipDeviceMallocUncached` / cooperative groups / grid sync**: `hipDeviceMallocUncached` exists
  as a flag to `hipExtMallocWithFlags` (generic HIP API surface, documented); cooperative-groups
  launch support is part of the HIP API surface generically, but this research pass found no
  Windows-specific statement about grid-wide cooperative-launch support on the Windows HIP runtime,
  and no evidence `r4dx` currently launches cooperative kernels. **open question / not exercised by
  this codebase**, flagged below rather than answered.
- **Timer resolution**: not directly documented for HIP-on-Windows event timers in sources found
  this pass; `r4dx`'s own profiling (`Model::DecodeStepProfiled`, `docs/perf.md`) uses paired
  `hipEvent`s with one final `hipEventSynchronize`, reporting sub-millisecond-resolution timings
  (individual op-family entries as low as 0.04-0.09 ms for `embed`) that are internally consistent
  step-to-step -- **inferred** the effective timer resolution is well under 100 microseconds from
  this repo's own measured data, since sub-100-microsecond entries are stable and non-zero, but no
  documented HIP-on-Windows timer-resolution spec was found to cite directly.

Sources: [AMD HIP SDK for Windows 7.2 release notes (Windows/Linux parity table)](https://rocm.docs.amd.com/projects/install-on-windows/en/develop/about/releasenotes.html), [HIP FAQ](https://rocm.docs.amd.com/projects/HIP/en/latest/faq.html), [HIP memory management docs](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hip_runtime_api/modules/memory_management.html), `docs/perf.md` (this repo, measured).

---

## 5. Known gfx12 compiler gotchas

From `third_party/libr4d/README.md` ("Notes for anyone reading the kernels"), all **measured** by
libr4d's own authors on this exact target and independently consistent with what this research pass
found in the compiler's builtin table:

1. **Wave32 WMMA fragment layout is load-bearing**: `idx = lane % 16`, `k = 8*(e>>2) + 4*(lane>>4) +
   (e&3)`; both A and B operands want K-contiguous rows, so swapping WMMA operands transposes the
   result for free -- a real optimization libr4d's GEMM kernels rely on, not just a correctness note.
2. **No `v_cvt_pk_bf16_f32`, no direct-to-LDS, no `ds_read_b64_tr_b16` on gfx1201** -- confirmed
   independently against the local builtin table in section 2.6 above. bf16 packing is done in
   software (two adds + `v_perm_b32`, per the README).
3. **LLVM single-buffers LDS and drains before every WMMA** unless told otherwise via
   `__builtin_amdgcn_sched_group_barrier` (confirmed generic/architecture-agnostic in section 2.10) --
   coarser scheduling groups are better, up to the point where a group asks for more outstanding
   loads than the hardware queue depth can hold. This is a compiler-default-is-conservative issue,
   not a hardware limitation: the hardware supports overlapping LDS traffic with WMMA issue, but the
   compiler's default scheduling doesn't exploit it without an explicit hint.
4. **`-mcumode` and per-translation-unit target features stay scoped**: libr4d compiles each `.hip`
   as a separate translation unit specifically so a flag like `-mcumode` (forcing CU mode -- see
   `__AMDGCN_CUMODE__` macro, confirmed `0`/WGP-mode-default for a plain `-mcpu=gfx1201` compile in
   this research pass, section "predefined macros" below) can apply to one kernel
   (`r4d_gdn_chunk_scan_k128_v128_c64_bf16.o`) without forcing it project-wide; `-ffp-contract=off`
   is the one flag applied globally, because the rotated 6-bit all-reduce kernel requires bit-exact
   (non-fused) multiply-add ordering across the two ranks or they diverge by ~1 ULP.
5. **Predefined macros for `-mcpu=gfx1201`** (measured, this pass, via
   `clang -x hip -E -dM ... --offload-arch=gfx1201 --cuda-device-only -nogpuinc -nogpulib` on an
   empty `.hip` file): `__GFX12__ 1`, `__gfx1201__ 1`, `__amdgcn_processor__ "gfx1201"`,
   `__amdgcn_target_id__ "gfx1201"`, `__AMDGCN_CUMODE__ 0` (i.e. **WGP mode is the compile default**;
   `-mcumode` flips this to `1` and switches the LDS-half-per-SIMD-pair allocation model described in
   section 1's LDS row).

---

## 6. Fact table (summary)

| Topic | Value | Source | Confidence |
|---|---|---|---|
| CU / WGP count | 64 CU / 32 WGP | AMD datasheet | documented |
| Boost clock | 2920 MHz | spec aggregators | documented |
| FP32 TFLOPS | 47.8 | AMD datasheet | documented |
| FP16 matrix (dense) TFLOPS | ~191 | spec aggregators | documented |
| INT4 sparse TOPS | 1531 (AMD headline) | AMD marketing, wccftech | documented |
| VRAM | 32 GiB GDDR6, 256-bit, 640 GB/s peak | AMD datasheet | documented |
| VRAM usable (this machine) | 31.86 GiB via hipInfo | `docs/perf.md` | measured |
| Infinity Cache | 64 MiB | AMD datasheet + press | documented |
| L2 | 8 MiB | zolotukhin.ai (secondary) | documented |
| L1 | 128 KiB/shader-array | zolotukhin.ai (secondary) | documented |
| L0 | 32 KiB/CU | zolotukhin.ai (secondary) | documented |
| LDS | 64 KiB/CU (CU mode) or 128 KiB/WGP (WGP mode) | AMD RDNA4 ISA ref | documented |
| PCIe | Gen5 x16 | AMD datasheet | documented |
| TBP | 300 W | AMD datasheet | documented |
| WMMA shapes on gfx1201 | 16x16x16 all dtypes; 16x16x32 iu4-only | `BuiltinsAMDGPU.inc` | measured |
| SWMMAC shapes on gfx1201 | 16x16x32 (8/16-bit types), 16x16x64 (iu4) | `BuiltinsAMDGPU.inc` | measured |
| fp4/fp6/MX-scale WMMA on gfx1201 | **absent** (gfx1250-only) | `BuiltinsAMDGPU.inc` (`gfx1250-insts` gate) | measured |
| Scaled (`cvt_scalef32_*`) fp8/fp4/fp6 conversions on gfx1201 | **absent** | `BuiltinsAMDGPU.inc` | measured |
| Unscaled `v_cvt_pk_{fp8,bf8}_f32` on gfx1201 | present (with a known e32/e64 assembler bug) | `BuiltinsAMDGPU.inc` + LLVM #224032 | measured + documented |
| `v_cvt_pk_bf16_f32` on gfx1201 | **absent** | `BuiltinsAMDGPU.inc` (no such builtin at all) + libr4d README | measured |
| `global_load_tr` (16-bit, b64/b128) on gfx1201 | present | `BuiltinsAMDGPU.inc` (`gfx12-insts`) | measured |
| `ds_read_tr*`/`ds_load_tr*` on gfx1201 | **absent** (gfx950- or gfx1250-only) | `BuiltinsAMDGPU.inc` | measured |
| `permlane16_var`/`permlanex16_var` on gfx1201 | present, gfx12-new | `BuiltinsAMDGPU.inc` (`gfx12-insts`) | measured |
| `s_wait_loadcnt`/`storecnt`/`dscnt`/`bvhcnt`/etc. | present (real gfx12 ISA, replaces unified `s_waitcnt`) | LLVM AMDGPUAsmGFX12 docs | documented |
| `s_wait_asynccnt`/`s_wait_tensorcnt` on gfx1201 | **absent** (gfx1250-only) | `BuiltinsAMDGPU.inc` | measured |
| `sched_group_barrier`/`iglp_opt` on gfx1201 | present, architecture-generic | `BuiltinsAMDGPU.inc` (empty feature gate) | measured |
| `__AMDGCN_CUMODE__` default for `-mcpu=gfx1201` | `0` (WGP mode) | `clang -E -dM` probe, this machine | measured |
| Decode host_enqueue overhead (this card/driver) | 2.48-4.25 ms per 64-layer step | `docs/perf.md` | measured |
| hipGraph Windows support status | not found stated explicitly either way | (search, this pass) | **open question** |
| Cooperative-launch/grid-sync on Windows HIP | not found stated explicitly | (search, this pass) | **open question** |

---

## Open questions (not resolved by this research pass)

- No direct read of AMD's RDNA4 ISA Reference Guide PDF's cache-hierarchy table was performed
  (fetched via web search summaries only); the L0/L1/L2 sizes in section 1 are corroborated by a
  secondary community reference, not the primary AMD PDF text.
- `hipGraph` support status specifically on the Windows HIP runtime (vs. Linux) was not found
  stated in any source checked this pass.
- Whether cooperative-groups grid-wide sync (`cudaLaunchCooperativeKernel`-equivalent) is supported
  on the Windows HIP driver was not found stated this pass.
- HIP-on-Windows event-timer resolution has no documented spec found; only inferred from this repo's
  own sub-100-microsecond-granularity profile data.
- The AMD-stated AI TOPS table in section 1 is an aggregation of third-party spec sites reading off
  the AMD datasheet/blog, not a transcription of a single official AMD table with FP8/INT8/INT4
  broken out by dense/sparse in one place -- worth re-verifying against the primary
  `odinsedge.ai` datasheet PDF directly (fetched via search snippet only in this pass, not opened in
  full) if these exact numbers become load-bearing for a decision.
