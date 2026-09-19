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

Sets `HIP_VISIBLE_DEVICES=1` and runs `ctest` against the `win-hip` build directory: 24 tests
covering `r4d_core` smoke, `src/core`/`src/kernels` device-buffer and kernel unit tests (rmsnorm,
residual add, silu_mul, rope, fp8/int8 activation quant, kv cache write, mxfp4 GEMM, attention
decode, GDN chunk scan, sampler), the converter's quantizer round-trip / byte-packer / kernel-decode
/ KV-calibration / bf16-layout tests, the tokenizer's golden-case suite, `src/model`'s per-layer
tests (GDN layer, full-attention layer, final-norm+lm_head, assembled-`Model` forward-pass smoke
including a prefill/decode state-handoff equivalence check), `src/cli`'s argument-parsing tests, and
a CPU-only Python reference-manifest check (`tests/reference/test_manifest.py`, run through the same
`ctest` invocation). All 24 currently pass (~60s wall on HIP device 1). See `docs/status.md` for the
full breakdown and known gaps, and `tools/convert_ref/` / `tools/reference/` for the additional
GPU-device-1 Python self-tests (kernel cross-checks and HF `transformers` goldens) that run outside
`ctest` -- see their READMEs for invocation.

## Usage

### Convert a checkpoint to a container

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
.\build\win-hip\src\convert\r4dx-convert.exe `
    --input C:\AI\models\Qwen3.8-27B --output D:\models\r4dx\qwen38-27b.r4dx `
    --layouts mxfp4,w4a16,w4a8 --lm-head 4bit+bf16 --mtp on --vision on `
    --kv-calib D:\models\r4dx\qwen38-27b.kvcalib.json --threads 16
```

Produces a single container carrying every requested quantized GEMM layout (plus bf16 for
embeddings/attention-gate-output/vision/MTP tensors) side by side, so `r4dx-cli --layout` can A/B
them against the same file. `--layers N` converts only the first `N` transformer layers (useful for
a small smoke-test container); omit it to convert all 64. `--kv-calib` fills the fp8 KV cache's
per-head descales from a `tools/reference/kv_calibrate.py` JSON (falls back to a `1.0` placeholder
per-layer, with a stderr warning, if omitted or if a layer is missing from the JSON). See
`docs/container-format.md` for the on-disk layout and `src/convert/main.cpp`'s header comment for
the full flag list, including `--selftest` for the byte-exact packer self-check.

### Generate text

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
.\build\win-hip\src\cli\r4dx-cli.exe --model D:\models\r4dx\qwen38-27b.r4dx --layout mxfp4 `
    --prompt "Write a haiku about GPUs, then explain what a GPU is in two sentences." `
    --max-tokens 128 --temperature 0 --stats
```

`--layout` selects which quantized (or `bf16`) body-weight variant baked into the container to run
(`mxfp4` / `w4a16` / `w4a8` / `bf16`) -- attention's `qg`/`o` projections always run bf16 regardless
of this flag (see `docs/perf.md`'s "Known limitation"). `--prompt "..."` renders one turn through
the real chat template and generates once; `--chat` instead starts an interactive multi-turn REPL
(re-rendering the whole conversation each turn, feeding only the new tail tokens to the model).
`--tokenizer-dir` defaults to `C:\AI\models\Qwen3.8-27B` (where `tokenizer.json` /
`chat_template.jinja` / `generation_config.json` live); `--think {on|off}` toggles the chat
template's `enable_thinking`; `--temperature 0` selects greedy argmax decoding, otherwise
temperature/top-k/top-p/min-p sampling with `--seed` applies. `--max-ctx` bounds the KV cache and
GDN state allocation (default 131072; the bf16 layout needs a much smaller value -- see
`docs/perf.md`). `--stats` prints container-load time, prefill/decode tokens/s, and VRAM used. See
`src/cli/cli_args.h` for the full flag list and `docs/perf.md` for measured throughput per layout.

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

Milestone 1 (container loader, GDN + attention layers, model forward, `r4dx-cli` text generation)
is complete and integrated -- see `docs/status.md` for what exists, what passes, known gaps, and the
next milestone (OpenAI-compatible chat API + MTP self-speculation).
