# r4dx

A from-scratch C++/HIP inference engine for Qwen/Qwen3.8-27B on a single AMD Radeon AI PRO R9700
(gfx1201) on Windows 11, built on top of the [libr4d](https://github.com) kernel library (a
plain-C-ABI, torch-free HIP kernel set). MIT licensed.

## What this is

r4dx is its own weight container, its own C++ model graph, and its own server -- no vLLM, no
ggml/GGUF. Third-party code is limited to header-only libraries vendored under `third_party/`
(nlohmann/json, cpp-httplib, minja, stb_image). GPU kernels come from `third_party/libr4d`
(a git submodule, branch `windows-llp64`), compiled into the static library target `r4d_core`.

## Decisions (summary)

- **Model**: Qwen/Qwen3.8-27B (`Qwen3_5ForConditionalGeneration`) -- 64 hybrid layers (48 Gated
  DeltaNet + 16 full attention), MTP head, vision tower carried in bf16.
- **Weights**: a custom converter round-trips the BF16 checkpoint into a safetensors-compatible
  container (see `docs/container-format.md`) in three quantized GEMM layouts so they can be A/B'd:
  MXFP4 (e2m1 + e8m0), INT4 g128 W4A16, and INT4 g128 W4A8. Weights are pre-permuted into each
  kernel's WMMA fragment order.
- **KV cache**: fp8 e4m3, paged in 16-token HND blocks, per-layer per-head static descales.
- **GPU rule**: only HIP device 1 (`$env:HIP_VISIBLE_DEVICES='1'`) is ever used on this machine.
- See `docs/architecture.md` for the forward-pass module map and `docs/container-format.md` for
  the weight container spec.

## Build (Windows, HIP/gfx1201)

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
.\build.ps1
```

This configures and builds with the `win-hip` CMake preset (Ninja + the vLLM_for_AMD venv's
CMake/Ninja binaries) against the ROCm SDK at `C:\opt\rocm`. See `docs/build-windows.md` for the
exact toolchain versions, flags, and gotchas.

## Test

```powershell
.\tests\run_tests.ps1
```

Sets `HIP_VISIBLE_DEVICES=1` and runs `ctest` against the `win-hip` build directory: 17 tests
covering `r4d_core` smoke, `src/core`/`src/kernels` device-buffer and kernel unit tests (rmsnorm,
residual add, silu_mul, rope, fp8/int8 activation quant, kv cache write, mxfp4 GEMM, attention
decode, GDN chunk scan, sampler), the converter's quantizer round-trip / byte-packer / kernel-decode
tests, the tokenizer's golden-case suite, and a CPU-only Python reference-manifest check
(`tests/reference/test_manifest.py`, run through the same `ctest` invocation). All 17 currently
pass. See `docs/status.md` for the full breakdown and known gaps, and `tools/convert_ref/` /
`tools/reference/` for the additional GPU-device-1 Python self-tests (kernel cross-checks and HF
`transformers` goldens) that run outside `ctest` -- see their READMEs for invocation.

## Layout

```
third_party/    r4d_core (libr4d submodule) + vendored header-only deps
src/core/       device/stream/buffer plumbing
src/kernels/    r4dx-owned HIP kernels (rmsnorm, rope, silu_mul, kv paging, sampling, ...)
src/model/      layer graph / forward pass
src/tokenizer/  tokenizer
src/convert/    weight converter (HF checkpoint -> r4dx container)
src/server/     OpenAI-compatible chat API
src/cli/        text-generation CLI
tests/          smoke + unit tests
docs/           architecture, container format, build notes, status
tools/          Python reference/validation tooling (read-only against the HF transformers venv)
```

## Status

Phase 0 (skeleton, container spec, `r4d_core`, converter, tokenizer, core kernels, reference
tooling) is complete and integrated -- see `docs/status.md` for what exists, what passes, known
gaps, and the next milestone (CLI text generation).
