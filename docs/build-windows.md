# Building r4dx on Windows (HIP / gfx1201)

This is the recipe `build.ps1` runs and `tests/run_tests.ps1` tests against, verified working on
this machine on 2026-09-19 (`smoke_r4d` PASS on HIP device 1).

## Toolchain versions (exact, verified)

- **ROCm SDK**: `C:\opt\rocm` -- HIP 7.15.26333, AMD clang 23.0.0 (`8f497e09`), target
  `x86_64-pc-windows-msvc`. `hipcc.exe` = `C:\opt\rocm\bin\hipcc.exe`; `clang-cl.exe` /
  `lld-link.exe` under `C:\opt\rocm\lib\llvm\bin`; device bitcode
  `C:\opt\rocm\lib\llvm\amdgcn\bitcode`; runtime `C:\opt\rocm\bin\amdhip64_7.dll`; import lib
  `C:\opt\rocm\lib\amdhip64.lib`.
- **CMake**: 4.4.2, **Ninja**: 1.13.2 -- both from
  `C:\Users\user\dev\vLLM_for_AMD\.venv-rocm10\Scripts\{cmake,ninja,ctest}.exe`. The PATH cmake is
  3.31 and cannot drive this configuration; `build.ps1` / `tests/run_tests.ps1` always call the
  venv's binaries by full path, never bare `cmake`/`ninja`/`ctest`.
- **MSVC**: 2022 BuildTools 14.44 + Windows Kits 10.0.26100 -- not invoked directly; clang-cl
  auto-detects them (link/library search paths) without needing `vcvars64.bat`, because clang-cl
  probes the registry/VS installer metadata itself. No `vcvars` step was needed for this build.

## The build

```powershell
$env:HIP_VISIBLE_DEVICES = '1'     # harmless at configure/compile time, required at test time
.\build.ps1
```

`build.ps1` calls `cmake --preset win-hip` then `cmake --build --preset win-hip`, using the venv's
`cmake.exe`/`ninja.exe`. `CMakePresets.json`'s `win-hip` preset sets `CMAKE_CXX_COMPILER` to
ROCm's own `clang-cl.exe` (not MSVC `cl.exe`) so the plain-C++ sources (`src/*`, `tests/*`) share
one clang toolchain and one CRT selection with the HIP-compiled objects.

### How `r4d_core` gets built without CMake's HIP language

CMake's `HIP` language support fights `clang-cl` on this box (this is why `env_windows_rocm.cmd`
in the sibling `vLLM_for_AMD` project needs a full `vcvars64` + `CMAKE_HIP_COMPILER` dance for its
own build). r4dx sidesteps it entirely: the project is `CXX`-only, and
`third_party/CMakeLists.txt` drives `hipcc.exe` directly through 15 `add_custom_command(OUTPUT
<unit>.obj COMMAND ... hipcc.exe ...)` rules -- one per libr4d translation unit r4dx links -- then
hands the resulting `.obj` files to `add_library(r4d_core STATIC ...)` as pre-built "external
objects" (`set_source_files_properties(... EXTERNAL_OBJECT TRUE GENERATED TRUE)`), a standard CMake
pattern for objects that did not come from CMake's own compile rules. Ninja treats each hipcc
invocation as an ordinary custom-command edge, so it parallelizes them like any other build step
(15 units compiled in ~15s wall time with 32 janitor threads on this machine's Ninja default job
count).

### hipcc flags

Base flags (mirrors libr4d's own proven `build_windows.ps1`, minus the Python/pybind11-specific
bits that build needs and this one does not):

```
-O3 -std=c++17 --offload-arch=gfx1201 -Wno-unused-result -ffp-contract=off
--rocm-path=C:/opt/rocm --rocm-device-lib-path=C:/opt/rocm/lib/llvm/amdgcn/bitcode
-DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt
```

Per-unit extras: `r4d_gdn_chunk_scan_k128_v128_c64_bf16` gets `-mcumode`;
`r4d_gemm_w4a8_nt_m64` gets `-DR4D_GEMM_W4A8_GROUP=128` (both match libr4d's own
`build_windows.ps1` `$UNITS` table); `r4d_gemm_w4a16_nt_m64` gets
`-DR4D_GEMM_W4_GROUP=${R4DX_W4A16_GROUP}`, which is the one deliberate deviation from that table
-- see the next section.

### w4a16 group size (`R4DX_W4A16_GROUP`)

`R4DX_W4A16_GROUP` (root `CMakeLists.txt`, default **128**) is the w4a16 layout's group size: how
many contiguous `K` share one `(scale, zero)` pair. It is a single CMake cache variable because it
has to reach two places that must never disagree:

| reaches | as | used for |
|---|---|---|
| `r4d_gemm_w4a16_nt_m64` (`third_party/CMakeLists.txt`) | `-DR4D_GEMM_W4_GROUP=<g>` | the stride the GEMM reads `.w4a16.wsz` at |
| `r4dx_convert` (`src/convert/CMakeLists.txt`) | `-DR4DX_W4A16_GROUP=<g>` -> `kW4A16Group` | the stride the converter *writes* `.w4a16.wsz` at |

Three layers check that they agree, because a mismatch produces **wrong numbers and nothing else**
-- no crash, no NaN, no warning:

1. `r4dx-convert` asserts `kW4A16Group == r4d_gemm_w4a16_nt_m64_group()` at startup
   (`ValidateKernelGroupSizes`). w4a8 and mxfp4 are checked the same way, each against its own
   kernel export -- w4a16 and w4a8 have **separate** converter constants (`kW4A16Group`,
   `kW4A8Group`) precisely so this one can move on its own.
2. Every container records the group it was packed with in `__metadata__.quant.w4a16.group`, and
   `Container::Load` / `DflashDraftWeights::Open` refuse a container whose group differs from the
   kernel this binary was built with (`CheckW4a16Group`, `src/model/quant_linear.h`), naming both
   numbers. Pre-`quant`-block containers have no group recorded and are group 128 by construction,
   so they still load on a default build.
3. CMake rejects a group that is not a positive multiple of 64: the kernel packs
   `R4D_GEMM_W4_KPB = 64` contiguous K per weight block and derives `bpg = group / 64`, so **64 and
   128 are the only values libr4d accepts unmodified**.

Bits per weight is `4 + 32/group` -- 4.25 at 128, 4.5 at 64. What the extra quarter-bit buys is
measured in `docs/validation.md` "Milestone 11 / group size".

**Build a non-default group in its own build directory.** A container and the binaries that read it
ship as a matched pair, so `build/win-hip` stays group 128 and the `win-hip-g64` preset builds
group 64 into `build/win-hip-g64`:

```powershell
.\build.ps1 -Preset win-hip-g64
```

The `tests/convert` suite is group-agnostic: it runs every int4 quantizer, packer and search at
**both** 64 and 128 whatever `R4DX_W4A16_GROUP` this build is, against
`tests/convert/fixtures/*_g64.bin` (the group-128 fixtures keep their historical unsuffixed names).
`tools/convert_ref/selftest_compare.py` takes `--w4a16-group` / `--w4a8-group` and, by default,
reads the group back out of the container the exe under test just wrote.

The `tests/model` and `tests/model/attention` tests are different: they read fixed 4-layer
containers at hard-coded `D:\models\r4dx\` paths, packed at group 128, which a group-64 build
rightly refuses. Convert group-matched copies once and point the suite at them with
`R4DX_TEST_CONTAINER_DIR` (`tests/model/test_container_path.h`) -- same basename, new directory:

```powershell
$dir = 'D:\models\r4dx\g64-testctr'
# qwen38-27b-l4-bf16.r4dx:   --layers 4 --layouts bf16,mxfp4,w4a16,w4a8 --lm-head 4bit+bf16 --mtp off --vision off
# qwen38-27b-l4-mtp.r4dx:    --layers 4 --layouts bf16,w4a16            --lm-head 4bit+bf16 --mtp on  --vision off
# qwen38-27b-l4-allmtp.r4dx: --layers 4 --layouts bf16,w4a16,w4a8,mxfp4 --lm-head 4bit+bf16 --mtp on  --vision off
# the two DFlash2 drafters:  --dflash-gguf <Qwen3.8-27B-DFlash2-Q8_0.gguf> --layout {bf16,w4a16}
.\build\win-hip-g64\src\convert\r4dx-convert.exe --input C:\AI\models\Qwen3.8-27B --output "$dir\..." ...
$env:R4DX_TEST_CONTAINER_DIR = $dir
ctest --preset win-hip-g64
```

`hipcc.exe` needs its own `clang.exe`/`lld-link.exe`/device libs found via PATH even though
`--rocm-path` is passed; `third_party/CMakeLists.txt` prepends `C:\opt\rocm\bin` and
`C:\opt\rocm\lib\llvm\bin` to PATH for each invocation via `cmake -E env PATH=... hipcc.exe ...`
rather than mutating the whole build's PATH.

### CRT consistency (the one real gotcha)

hipcc's objects and clang-cl's objects have to agree on the MSVC CRT or the final link fails with
`LNK2038` (CRT version mismatch). The fix applied here: `CMakePresets.json`/root `CMakeLists.txt`
set `cmake_policy(SET CMP0091 NEW)` + `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL` (the dynamic,
`msvcrt.dll`-family CRT) for every CMake-compiled target, and `third_party/CMakeLists.txt` passes
the matching `-D_DLL -D_MT -Xclang --dependent-lib=msvcrt` to every hipcc invocation -- the same
flags libr4d's `build_windows.ps1` uses for its Python-extension build, for the same reason (there
it is matching `python312.dll`; here it is matching r4dx's own clang-cl objects). This combination
linked clean on the first try.

### git submodule over a local path

`git submodule add C:/Users/user/dev/libr4d third_party/libr4d` fails on current git
(`fatal: transport 'file' not allowed`) unless file-transport is explicitly allowed for that one
invocation:

```powershell
git -c protocol.file.allow=always submodule add C:/Users/user/dev/libr4d third_party/libr4d
cd third_party/libr4d
git -c protocol.file.allow=always fetch origin windows-llp64
git checkout windows-llp64
```

`.gitmodules` also pins `branch = windows-llp64` so `git submodule update --remote` (if ever run)
tracks the right branch. Checked-out commit: `7675605` ("Windows LLP64 sweep + Windows build
script for r4d.pyd"), matching the pinned commit in the task brief.

### r4d_registry links without the AR units

`r4d_registry.hip` was confirmed (by building and linking, then running `smoke_r4d`, which prints
all 25 registry rows including the 7 `ar_*` ones) to reference the all-reduce kernels **only as
data** -- string literals (`"ar_oneshot_2rank_exact"`, ...) and a compile-time `enum` constant
(`R4D_AR_TWOSHOT_WIDE_MAX_ELEMS`) in its constraint tables, never as a linked symbol. So `r4d_core`
links clean with the `r4d_ar_*.hip` units excluded and **no patch to the libr4d submodule was
needed** -- the "guard AR rows on a new `r4dx` branch" contingency in the task brief did not apply.

## The test

```powershell
.\tests\run_tests.ps1
```

Sets `HIP_VISIBLE_DEVICES=1` (this machine has two R9700s; only device 1 may ever be used -- see
README.md), builds `smoke_r4d`, then runs `ctest --preset win-hip --output-on-failure`.
`smoke_r4d` links `r4d_core` + `r4dx_hip_runtime`, prints `r4d_attn_dims`/`r4d_gdn_dims`/registry,
and runs `r4d_gemm_bf16_nt_m64` (M=8, N=1024, K=2048, WV=4, SK=4, MB=1 -- the same shape and
tuning as libr4d's `build-win/check_gemm_bf16.py`) against a CPU fp32 reference. Verified result on
this machine: `rel_err=1.6772e-03` (threshold `< 2e-2`), **PASS**.

## Full command reference

```powershell
# one-time
git -c protocol.file.allow=always submodule update --init third_party/libr4d

# build
$env:HIP_VISIBLE_DEVICES = '1'
.\build.ps1                 # or: .\build.ps1 -Clean

# test
.\tests\run_tests.ps1
```
