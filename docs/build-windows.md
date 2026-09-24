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

`R4DX_W4A16_GROUP` (root `CMakeLists.txt`, default **64** since Milestone 11) is the w4a16 layout's
group size: how many contiguous `K` share one `(scale, zero)` pair. It is a single CMake cache
variable because it has to reach two places that must never disagree:

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
   so nothing checks them -- they are the one case that can still be read at the wrong stride, and
   there are none left on this machine.

   The guard fires only when the load will actually **read** `.w4a16.wsz` bytes: `Container::Load`
   when one of `--layout` / the lm-head layout / the MTP-head layout is `w4a16`,
   `DflashDraftWeights::Open` when the drafter carries a `.w4a16.*` tensor at all. `r4dx-convert`
   writes the `quant` metadata block unconditionally, so a bf16 or mxfp4 container records a w4a16
   group for a layout it holds no tensor of, and `v5` holds valid group-independent `w4a8`/`mxfp4`
   layouts next to its group-128 `w4a16` -- `r4dx-cli --model ...v5.r4dx --layout mxfp4` is correct
   on a group-64 build and is allowed. Only `--layout w4a16` on `v5` is refused.
3. CMake rejects a group that is not a positive multiple of 64: the kernel packs
   `R4D_GEMM_W4_KPB = 64` contiguous K per weight block and derives `bpg = group / 64`, so **64 and
   128 are the only values libr4d accepts unmodified**.

A third place compiles this kernel outside CMake: libr4d's own `build_windows.ps1`, which builds the
`r4d.pyd` that `tools/profile/tune_gemm.py` times. It passes no group flag, so a stock pyd is group
128 and the tuner refuses to sweep w4a16 on it; `tune_gemm.py`'s module docstring ("W4A16 GROUP")
has the recipe for a group-64 pyd.

Bits per weight is `4 + 32/group` -- 4.25 at 128, 4.5 at 64. What the extra quarter-bit buys is
measured in `docs/validation.md` "Milestone 11 / group size"; why 64 became the default is
"Milestone 11 / recipe" in the same file.

**The default changed from 128 to 64 in Milestone 11, and that is a breaking change for existing
containers.** `qwen38-27b-v5.r4dx`, `v4`, `v3`, the 4-layer test containers and the original
DFlash2 drafters were all packed at group 128, and a default build now refuses every one of them:

```
r4dx::model: D:\models\r4dx\qwen38-27b-v5.r4dx was packed with w4a16 group=128 but this build's
r4d_gemm_w4a16_nt_m64 kernel reads group=64 -- the .w4a16.wsz scales would be read at the wrong
stride, producing wrong numbers with no other symptom. Reconfigure with -DR4DX_W4A16_GROUP=128 in
its own build directory, or re-convert the container with this build's r4dx-convert.
```

The two ways out are exactly the two the message names: re-convert (the production container is now
`qwen38-27b-v6.r4dx`, packed at 64 -- README's convert command), or build group 128 in its own build
directory with the `win-hip-g128` preset:

```powershell
.\build.ps1 -Preset win-hip-g128
```

**Note that an existing build directory keeps the group in its CMake cache.** Changing the default
in `CMakeLists.txt` does *not* move a `build/win-hip` that was configured before; `cmake --preset
win-hip` will happily re-report `w4a16 group = 128`. Pass `-DR4DX_W4A16_GROUP=64` once to move it
(the configure line prints the group it settled on), or delete the build directory.

**Build a non-default group in its own build directory.** A container and the binaries that read it
ship as a matched pair, so `build/win-hip` is group 64 and `build/win-hip-g128` is group 128; never
reconfigure one into the other in place.

The `tests/convert` suite is group-agnostic: it runs every int4 quantizer, packer and search at
**both** 64 and 128 whatever `R4DX_W4A16_GROUP` this build is, against
`tests/convert/fixtures/*_g64.bin` (the group-128 fixtures keep their historical unsuffixed names).
`tools/convert_ref/selftest_compare.py` takes `--w4a16-group` / `--w4a8-group` and, by default,
reads the group back out of the container the exe under test just wrote.

The `tests/model` and `tests/model/attention` tests are different: they open fixed 4-layer test
containers, and a container is only loadable by binaries built at its own group. Those containers
therefore exist once per group -- the historical **group-128** copies at `D:\models\r4dx\<name>`,
**group-64** copies under the same basenames in `D:\models\r4dx\g64\` -- and every test resolves
its path through `r4dx_test::ContainerPath` (`tests/model/test_container_path.h`) in this order:

1. `R4DX_TEST_CONTAINER_DIR`, if set: `<that dir>\<basename>` (an explicit override; it wins over
   the group, so pointing it at copies of the wrong group makes the affected tests **fail** on the
   loader's group guard, with the loader's message).
2. Otherwise the directory for the group the binary was built with, read at run time from
   `r4d_gemm_w4a16_nt_m64_group()` -- the same number the loader checks the container against:
   group 128 -> `D:\models\r4dx\<name>` (unchanged), group 64 -> `D:\models\r4dx\g64\<name>`.

The tests that need the **real 64-layer container** (`test_dflash_e2e`, `test_vision_tower`, and
the defaults of the `tool_*` diagnostics) use `ProductionTargetPath()` / `ProductionDrafterPath()`
from the same header, which pick the production pair packed at the build's group:
`qwen38-27b-v6.r4dx` + `qwen38-27b-dflash2-w4a16-g64.r4dx` at 64, `qwen38-27b-v3.r4dx` +
`qwen38-27b-dflash2-w4a16.r4dx` at 128, both in `D:\models\r4dx\`. `R4DX_TEST_CONTAINER_DIR` does
not apply to that pair (no 64-layer container is copied into `g64\`).

So **no environment variable is needed on either build** -- plain `ctest --preset win-hip` and
`ctest --preset win-hip-g128` (or `ctest --test-dir build\win-hip[-g128]` with
`HIP_VISIBLE_DEVICES=1`) each read the containers of their own group:

```powershell
.\build.ps1;                         ctest --preset win-hip        # group 64: g64\ + v6
.\build.ps1 -Preset win-hip-g128;    ctest --preset win-hip-g128   # group 128: D:\models\r4dx\ + v3
$env:R4DX_TEST_CONTAINER_DIR = 'E:\elsewhere'; ctest --preset win-hip   # optional override
```

A container that is **missing** makes its test SKIP (exit 77, `[SKIP] <path> not found`); one that
is **present but refused** makes it FAIL with the loader's message (`r4dx_test::RunGuardedMain`
catches what would otherwise escape `main()` and end the process as `0xc0000409`).

`tools/validate_dflash.ps1`, `validate_fusion.ps1`, `validate_spec_sampling.ps1` and
`tools/server/smoke.ps1` follow the same rules through `tools/r4dx_containers.ps1`: with no
`-Model`/`-Dflash` they read `R4DX_W4A16_GROUP` from the `CMakeCache.txt` of the build directory
whose executable they run (`build\win-hip`, or `build\<Preset>` for `smoke.ps1`) and pick the
matching production pair (the validators) or 4-layer test container (`smoke.ps1`). An explicit
`-Model` / `-Dflash` always wins.

The recipe for regenerating `g64\` (six containers, ~52 GiB, ~3 min of CPU) is:

```powershell
$dir = 'D:\models\r4dx\g64'
# qwen38-27b-l4-bf16.r4dx:   --layers 4 --layouts bf16,mxfp4,w4a16,w4a8 --lm-head 4bit+bf16 --mtp off --vision off
# qwen38-27b-l4-mtp.r4dx:    --layers 4 --layouts bf16,w4a16            --lm-head 4bit+bf16 --mtp on  --vision off
# qwen38-27b-l4-allmtp.r4dx: --layers 4 --layouts bf16,w4a16,w4a8,mxfp4 --lm-head 4bit+bf16 --mtp on  --vision off
# qwen38-27b-l4-mtp-draftvocab.r4dx: as -l4-mtp plus --draft-vocab-ids <ids.json> (see below)
# the two DFlash2 drafters:  --dflash-gguf <Qwen3.8-27B-DFlash2-Q8_0.gguf> --out ... --layout {bf16,w4a16}
.\build\win-hip\src\convert\r4dx-convert.exe --input C:\AI\models\Qwen3.8-27B --output "$dir\..." ...
```

`qwen38-27b-l4-mtp-draftvocab.r4dx` **is** in `g64\`, and it is the one container whose regeneration
is not a straight re-run: its `--draft-vocab-ids` JSON is a gitignored build artifact that was no
longer on disk, so it was rebuilt from a fresh arbitrary 4096-id subset (the 1416 distinct ids in
`tools/reference/kl_corpus/tokens.json`, padded from 0) -- the reduced-vocab draft head is about
correctness plumbing, not about which ids, so any 4096-id list does. Without it `test_mtp`'s
reduced-vocab cases drop to `[SKIP]` on a default build even though that path is on by default in
production. There is no 64-layer container in `g64\` (not worth a 42 GiB copy): the group-64
build's real-container tests use the production `qwen38-27b-v6.r4dx` instead, as above.

A bf16 or mxfp4 drafter does **not** need a group-64 copy (layer 2 above): only the `w4a16` one
does. `g64\qwen38-27b-dflash2-bf16.r4dx` was converted before that scope was narrowed and is
redundant; the original `D:\models\r4dx\qwen38-27b-dflash2-bf16.r4dx` loads on either build.

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
