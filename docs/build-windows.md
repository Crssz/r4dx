# Building r4dx on Windows (HIP / gfx1201)

This is the recipe `build.ps1` runs and `tests/run_tests.ps1` tests against, verified working on
this machine on 2026-09-19 (`smoke_r4d` PASS on HIP device 1).

## Toolchain versions (exact, verified)

- **ROCm SDK**: `C:\opt\rocm` -- HIP 7.15.26333, AMD clang 23.0.0 (`8f497e09`), target
  `x86_64-pc-windows-msvc`. `hipcc.exe` = `C:\opt\rocm\bin\hipcc.exe`; `clang-cl.exe` /
  `lld-link.exe` under `C:\opt\rocm\lib\llvm\bin`; device bitcode
  `C:\opt\rocm\lib\llvm\amdgcn\bitcode`; runtime `C:\opt\rocm\bin\amdhip64_7.dll`; import lib
  `C:\opt\rocm\lib\amdhip64.lib`.
- **CMake**: 4.4.2, **Ninja**: 1.13.2 -- originally both from a reference venv's
  `Scripts\{cmake,ninja,ctest}.exe`. `build.ps1` / `tests/run_tests.ps1` use
  `$env:R4DX_REFERENCE_VENV\Scripts` when that venv exists, else the cmake/ninja/ctest on PATH
  (CMake 3.31 + Ninja configure and build this preset too, verified 2026-09-22).
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
`third_party/CMakeLists.txt` drives `hipcc.exe` directly through 14 `add_custom_command(OUTPUT
<unit>.obj COMMAND ... hipcc.exe ...)` rules -- one per libr4d translation unit r4dx links -- then
hands the resulting `.obj` files to `add_library(r4d_core STATIC ...)` as pre-built "external
objects" (`set_source_files_properties(... EXTERNAL_OBJECT TRUE GENERATED TRUE)`), a standard CMake
pattern for objects that did not come from CMake's own compile rules. Ninja treats each hipcc
invocation as an ordinary custom-command edge, so it parallelizes them like any other build step
(the units compile in ~15s wall time with 32 janitor threads on this machine's Ninja default job
count). The 14 units are the attention (paged gqa6, paged gqa2 sliding-window, vit), 5 GDN, bf16 / w4a16 / trellis M<=64 / trellis
M=256 GEMM, DFlash conv and registry units.

### hipcc flags

Base flags:

```
-O3 -std=c++17 --offload-arch=gfx1201 -Wno-unused-result -ffp-contract=off
--rocm-path=C:/opt/rocm --rocm-device-lib-path=C:/opt/rocm/lib/llvm/amdgcn/bitcode
-DNDEBUG -D_DLL -D_MT -Xclang --dependent-lib=msvcrt
```

Per-unit extras: `r4d_gdn_chunk_scan_k128_v128_c64_bf16` gets `-mcumode`; no other unit takes one.

### w4a16 group size

The w4a16 layout's default group is **64**: how many contiguous `K` share one `(scale, zero)` pair.
It is a compile-time constant in two places that must never disagree:

| where | as | used for |
|---|---|---|
| `r4d_gemm_w4a16_nt_m64` (`third_party/libr4d/r4d_gemm_w4a16_nt_m64.hip`) | `R4D_GEMM_W4_GROUP` | the stride the GEMM reads `.w4a16.wsz` at |
| `r4dx_convert` (`src/convert/include/r4dx_convert/quant_int4.hpp`) | `kW4A16Group` | the stride the converter *writes* `.w4a16.wsz` at |

The kernel also serves groups 32 and 64 per call (`r4d_gemm_w4a16_nt_m64_g`): the LM head is packed at
32 and everything else at 64, and `r4dx-convert --w4a16-group-rule` can pick either per tensor. The
kernel packs `R4D_GEMM_W4_KPB = 64` contiguous K per weight block. Bits per weight is `4 + 32/group`,
so 4.5 at 64 and 5 at 32.

Two layers check that the constants agree, because a mismatch produces **wrong numbers and nothing
else** -- no crash, no NaN, no warning:

1. `r4dx-convert` asserts `kW4A16Group == r4d_gemm_w4a16_nt_m64_group()` at startup
   (`ValidateKernelGroupSizes`).
2. Every container records the group it was packed with in `__metadata__.quant.w4a16.group`, and
   `Container::Load` / `DflashDraftWeights::Open` refuse a container whose group differs from the
   kernel this binary was built with (`CheckW4a16Group`, `src/model/quant_linear.h`), naming both
   numbers. A container with no recorded group predates the `quant` block and is group 128 by
   construction, so the guard skips it; its scale tensors are refused by size instead
   (`CheckW4a16Shape`).

   The guard fires only when the load will actually **read** `.w4a16.wsz` bytes: `Container::Load`
   when one of `--layout` / the lm-head layout / the MTP-head layout is `w4a16`,
   `DflashDraftWeights::Open` when the drafter carries a `.w4a16.*` tensor at all. `r4dx-convert`
   writes the `quant` metadata block unconditionally, so a bf16 container records a w4a16 group for a
   layout it holds no tensor of.

A container packed at any other group (the old group-128 containers) is refused:

```
r4dx::model: D:\models\r4dx\qwen38-27b-v5.r4dx was packed with w4a16 group=128 but this build's
r4d_gemm_w4a16_nt_m64 kernel reads group=64 -- the .w4a16.wsz scales would be read at the wrong
stride, producing wrong numbers with no other symptom. Re-convert the container with this build's
r4dx-convert.
```

The way out is to re-convert (the production container is the Huihui trellis mix4.5m, whose w4a16
parts are packed at 32 and 64 -- README's convert commands).

The `tests/convert` suite runs the int4 quantizers, packers and search at 64 and 32 (the groups the
kernel serves), and the byte-exactness checks against the Python reference use
`tests/convert/fixtures/*_g64.bin`. `tools/convert_ref/selftest_compare.py` takes `--w4a16-group` and,
by default, reads the group back out of the container the exe under test just wrote.

The `tests/model` and `tests/model/attention` tests open fixed 4-layer test containers packed at group
64, in `D:\models\r4dx\g64\`, and every test resolves its path through `r4dx_test::ContainerPath`
(`tests/model/test_container_path.h`):

1. `R4DX_TEST_CONTAINER_DIR`, if set: `<that dir>\<basename>` (an explicit override; pointing it at
   copies of the wrong group makes the affected tests **fail** on the loader's group guard, with the
   loader's message).
2. Otherwise `D:\models\r4dx\g64\<basename>`.

The tests that need the **real 64-layer container** (`test_dflash_e2e`, `test_vision_tower`, the
real-container cases of `test_tp_emulation` / `test_tp_real_vs_emulation`, and the defaults of the
`tool_*` diagnostics) use `ProductionTargetPath()` / `ProductionDrafterPath()` /
`ProductionLayoutName()` from the same header: the Huihui abliterated trellis mix4.5m
`huihui-qwen38-27b-abl-trellis-mix45m.r4dx` (layout `trellis`) + `qwen38-27b-dflash2-w4a16-g64.r4dx`,
both in `D:\models\r4dx\`. The previous production container, the base `qwen38-27b-v6.r4dx`
(w4a16), was retired on 2026-09-29 with the base checkpoint; the tokenizer tests
(`R4DX_TOKENIZER_MODEL_DIR`) and `test_keep_bf16` (`R4DX_HF_CHECKPOINT`) read the Huihui checkpoint dir
`D:\models\Huihui-Qwen3.8-27B-abliterated` (byte-identical tokenizer files; layers 0-3 unchanged).
`R4DX_TEST_CONTAINER_DIR` does not apply to that pair (no 64-layer container is copied into `g64\`).

So **no environment variable is needed** -- plain `ctest --preset win-hip` (or `ctest --test-dir
build\win-hip` with `HIP_VISIBLE_DEVICES=1`) reads the containers above:

```powershell
.\build.ps1;  ctest --preset win-hip                                    # g64\ + the Huihui trellis container
$env:R4DX_TEST_CONTAINER_DIR = 'E:\elsewhere'; ctest --preset win-hip   # optional override
```

A container that is **missing** makes its test SKIP (exit 77, `[SKIP] <path> not found`); one that
is **present but refused** makes it FAIL with the loader's message (`r4dx_test::RunGuardedMain`
catches what would otherwise escape `main()` and end the process as `0xc0000409`).

`tools/validate_dflash.ps1`, `validate_fusion.ps1`, `validate_spec_sampling.ps1` and
`tools/server/smoke.ps1` follow the same rules through `tools/r4dx_containers.ps1`: with no
`-Model`/`-Dflash` they pick the production pair (the validators; the layout too,
`Get-R4dxProductionLayout`) or the 4-layer test container (`smoke.ps1`). An explicit `-Model` /
`-Dflash` always wins. The validators' `-Layouts` default is the production container's layout
(`trellis`), or `w4a16` when `-Model` is given explicitly.

The recipe for regenerating `g64\` (six containers, ~52 GiB, ~3 min of CPU) is:

```powershell
$dir = 'D:\models\r4dx\g64'
# (a convert writes only bf16 and w4a16 here)
# qwen38-27b-l4-bf16.r4dx:   --layers 4 --layouts bf16,w4a16 --lm-head 4bit+bf16 --mtp off --vision off
# qwen38-27b-l4-mtp.r4dx:    --layers 4 --layouts bf16,w4a16 --lm-head 4bit+bf16 --mtp on  --vision off
# qwen38-27b-l4-allmtp.r4dx: --layers 4 --layouts bf16,w4a16 --lm-head 4bit+bf16 --mtp on  --vision off
# qwen38-27b-l4-mtp-draftvocab.r4dx: as -l4-mtp plus --draft-vocab-ids <ids.json> (see below)
# the two DFlash2 drafters:  --dflash-gguf <Qwen3.8-27B-DFlash2-Q8_0.gguf> --out ... --layout {bf16,w4a16}
.\build\win-hip\src\convert\r4dx-convert.exe --input D:\models\Huihui-Qwen3.8-27B-abliterated --output "$dir\..." ...
```

`qwen38-27b-l4-mtp-draftvocab.r4dx` **is** in `g64\`, and it is the one container whose regeneration
is not a straight re-run: its `--draft-vocab-ids` JSON is a gitignored build artifact that was no
longer on disk, so it was rebuilt from a fresh arbitrary 4096-id subset (the 1416 distinct ids in
`tools/reference/kl_corpus/tokens.json`, padded from 0) -- the reduced-vocab draft head is about
correctness plumbing, not about which ids, so any 4096-id list does. Without it `test_mtp`'s
reduced-vocab cases drop to `[SKIP]` even though that path is on by default in production. There is no
64-layer container in `g64\` (not worth a 42 GiB copy): the real-container tests use the production
Huihui trellis container instead, as above.

A bf16 drafter does **not** need a group-64 copy: only the `w4a16` one does (the root group-128 w4a16
drafter container is refused by the group guard). `g64\qwen38-27b-dflash2-bf16.r4dx` was converted
before that scope was narrowed and is redundant; the original
`D:\models\r4dx\qwen38-27b-dflash2-bf16.r4dx` loads as is.

`hipcc.exe` needs its own `clang.exe`/`lld-link.exe`/device libs found via PATH even though
`--rocm-path` is passed; `third_party/CMakeLists.txt` prepends `C:\opt\rocm\bin` and
`C:\opt\rocm\lib\llvm\bin` to PATH for each invocation via `cmake -E env PATH=... hipcc.exe ...`
rather than mutating the whole build's PATH.

### CRT consistency (the one real gotcha)

hipcc's objects and clang-cl's objects have to agree on the MSVC CRT or the final link fails with
`LNK2038` (CRT version mismatch). The fix applied here: `CMakePresets.json`/root `CMakeLists.txt`
set `cmake_policy(SET CMP0091 NEW)` + `CMAKE_MSVC_RUNTIME_LIBRARY=MultiThreadedDLL` (the dynamic,
`msvcrt.dll`-family CRT) for every CMake-compiled target, and `third_party/CMakeLists.txt` passes
the matching `-D_DLL -D_MT -Xclang --dependent-lib=msvcrt` to every hipcc invocation, so the hipcc
objects match r4dx's own clang-cl objects. This combination linked clean on the first try.

### libr4d is vendored

The GPU kernels live as plain sources under `third_party/libr4d`, part of this repository: a plain
`git clone` builds, with nothing to fetch. Credit for the library is in the top-level `NOTICE`, and
`third_party/VERSIONS.md` records which upstream commit the tree came from.

### r4d_registry rows

`r4d_registry.hip` lists only the kernels that are in the vendored libr4d: `smoke_r4d` prints its 18
rows.

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
# build
$env:HIP_VISIBLE_DEVICES = '1'
.\build.ps1                 # or: .\build.ps1 -Clean

# test
.\tests\run_tests.ps1
```
