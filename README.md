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

Sets `HIP_VISIBLE_DEVICES=1` and runs `ctest` against the `win-hip` build directory: 30 tests
covering `r4d_core` smoke, `src/core`/`src/kernels` device-buffer and kernel unit tests (rmsnorm,
residual add, silu_mul, rope, fp8/int8 activation quant, kv cache write, mxfp4 GEMM, attention
decode, GDN chunk scan, sampler), the converter's quantizer round-trip / byte-packer / kernel-decode
/ KV-calibration / bf16-layout tests, the tokenizer's golden-case suite, `src/model`'s per-layer
tests (GDN layer, full-attention layer, final-norm+lm_head, assembled-`Model` forward-pass smoke
including a prefill/decode state-handoff equivalence check, and MTP's verify/rejection-rewind
tests), `src/cli`'s argument-parsing tests, `src/server`'s CPU-only tests (CLI args, OpenAI request/
response JSON shapes, SSE framing, buffering/streaming sinks, the bounded request queue), and a
CPU-only Python reference-manifest check (`tests/reference/test_manifest.py`, run through the same
`ctest` invocation). All 30 currently pass (~90-120s wall on HIP device 1). See `docs/status.md` for
the full breakdown and known gaps, and `tools/convert_ref/` / `tools/reference/` for the additional
GPU-device-1 Python self-tests (kernel cross-checks and HF `transformers` goldens) that run outside
`ctest` -- see their READMEs for invocation. `tools/server/smoke.ps1` is a separate GPU integration
smoke test for `r4dx-server` (see "Run the OpenAI-compatible server" below) -- also not part of
`ctest`, since it needs a live HTTP server and a real container.

## Usage

### Convert a checkpoint to a container

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
.\build\win-hip\src\convert\r4dx-convert.exe `
    --input C:\AI\models\Qwen3.8-27B --output D:\models\r4dx\qwen38-27b-v3.r4dx `
    --layouts w4a8,w4a16,mxfp4 --lm-head 4bit --no-bf16 --mtp on --vision on `
    --kv-calib D:\models\r4dx\qwen38-27b.kvcalib.json
```

Produces a single container carrying every requested quantized GEMM layout (plus bf16 for
embeddings/vision/MTP tensors) side by side, so `r4dx-cli --layout` can A/B them against the same
file. `--layers N` converts only the first `N` transformer layers (useful for a small smoke-test
container); omit it to convert all 64. `--kv-calib` fills the fp8 KV cache's per-head descales from
a `tools/reference/kv_calibrate.py` JSON (falls back to a `1.0` placeholder per-layer, with a
stderr warning, if omitted or if a layer is missing from the JSON). `--no-bf16` (docs/r9700.md R1)
drops the full-model bf16 body layout and the bf16 `lm_head` variant entirely -- the current
recommended real-model container, `D:\models\r4dx\qwen38-27b-v3.r4dx`, was converted this way
(w4a8/w4a16/mxfp4 only, no bf16 anywhere except the small 4-layer test containers, which still pass
`--layouts bf16,...` since bf16 is the numerical-reference layout rungs 1-3 of `docs/validation.md`
need). The older `D:\models\r4dx\qwen38-27b.r4dx` (all four layouts including full bf16) is kept on
disk for comparison -- see `docs/perf.md`/`docs/r9700.md` for why bf16 is out of scope for
performance work. See `docs/container-format.md` for the on-disk layout and `src/convert/main.cpp`'s
header comment for the full flag list, including `--selftest` for the byte-exact packer self-check.

### Generate text

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
.\build\win-hip\src\cli\r4dx-cli.exe --model D:\models\r4dx\qwen38-27b-v3.r4dx --layout w4a16 `
    --prompt "Write a haiku about GPUs, then explain what a GPU is in two sentences." `
    --max-tokens 128 --temperature 0 --stats
```

`--layout` selects which quantized (or `bf16`) body-weight variant baked into the container to run
(`mxfp4` / `w4a16` / `w4a8` / `bf16`) -- as of the Milestone 2 performance pass this now includes
attention's `qg`/`o` projections too (they used to always run bf16 regardless of `--layout`; see
`docs/perf.md`'s "Known limitation", now resolved). `--prompt "..."` renders one turn through
the real chat template and generates once; `--chat` instead starts an interactive multi-turn REPL
(re-rendering the whole conversation each turn, feeding only the new tail tokens to the model).
`--tokenizer-dir` defaults to `C:\AI\models\Qwen3.8-27B` (where `tokenizer.json` /
`chat_template.jinja` / `generation_config.json` live); `--think {on|off}` toggles the chat
template's `enable_thinking`; `--temperature 0` selects greedy argmax decoding, otherwise
temperature/top-k/top-p/min-p sampling with `--seed` applies. `--max-ctx` bounds the KV cache and
GDN state allocation (default 131072; the bf16 layout needs a much smaller value -- see
`docs/perf.md`). `--stats` prints container-load time, prefill/decode tokens/s, and VRAM used, plus
(when `--mtp K>0`) an MTP acceptance-rate line. `--mtp K` (default 0) enables MTP self-speculative
decode: each decode round drafts up to `K` tokens via the checkpoint's own `mtp.*` weights, verifies
them against the real model in one batched call, and commits the accepted prefix (plus one corrected/
bonus token) -- greedy-only (`--mtp K` with `--temperature > 0` warns and forces `--mtp 0` for that
run). See `docs/mtp.md` for the full design, the container requirement (the container must carry
`mtp.*` weights -- `D:\models\r4dx\qwen38-27b.r4dx` already does), and measured acceptance/speedup
per layout (roughly +55-100% decode throughput at each layout's best `K`, `--mtp 3` a reasonable
default). See `src/cli/cli_args.h` for the full flag list and `docs/perf.md` for measured throughput
per layout.

## Run the OpenAI-compatible server

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
.\build\win-hip\src\server\r4dx-server.exe --model D:\models\r4dx\qwen38-27b-v3.r4dx --layout w4a16 `
    --host 127.0.0.1 --port 8080
```

Exposes `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (streaming via `"stream":
true` or non-streaming JSON), and `POST /v1/completions` (raw prompt, no chat template) -- a subset
of the OpenAI chat/completions API, single model, single GPU, one request processed at a time by a
dedicated worker thread (see `docs/server.md`'s "Concurrency model"). Multi-turn conversations reuse
the KV/GDN cache across requests the same way `r4dx-cli --chat` does, by matching each request's
full re-tokenized prompt against the tokens already committed to the model's state and re-prefilling
only the new tail (or reloading from scratch on a prefix mismatch). `--mtp` is not yet exposed as a
server flag (server-side MTP is future work); the server always runs plain decode. See `docs/
server.md` for the full endpoint/field reference, deferred features (tool-call parsing, vision), and
a captured real streamed answer, and `tools/server/smoke.ps1` for the GPU integration smoke test
(`.\tools\server\smoke.ps1` against the small 4-layer test container by default; pass
`-Model`/`-Layout`/`-Layers -1` to point it at a real container).

## Layout

```
third_party/    r4d_core (libr4d submodule) + vendored header-only deps
src/core/       device/stream/buffer plumbing
src/kernels/    r4dx-owned HIP kernels (rmsnorm, rope, silu_mul, kv paging, sampling, ...)
src/model/      layer graph / forward pass
src/tokenizer/  tokenizer
src/convert/    weight converter (HF checkpoint -> r4dx container)
src/server/     OpenAI-compatible chat API (r4dx-server)
src/cli/        text-generation CLI (r4dx-cli)
tests/          smoke + unit tests
docs/           architecture, container format, build notes, status
tools/          Python reference/validation tooling (read-only against the HF transformers venv)
```

## Status

Milestone 1 (container loader, GDN + attention layers, model forward, `r4dx-cli` text generation)
and Milestone 2 (`r4dx-server` OpenAI-compatible chat API, a decode/prefill performance pass, and
MTP self-speculative decode) are both complete and integrated -- see `docs/status.md` for what
exists, what passes, known gaps, and the next milestone (vision tower, then DFlash2 drafting).
