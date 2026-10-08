# r4dx

A from-scratch C++/HIP inference engine for **Qwen3.8-27B** on AMD RDNA4 (Radeon AI PRO R9700,
gfx1201) under Windows. It has its own weight container, model graph, tokenizer, CLI and an
OpenAI-compatible server. No PyTorch, vLLM or ggml at runtime.

> **Experimental.** This is a personal research project, developed and tested on one machine
> (Windows 11, two R9700 cards, ROCm SDK at `C:\opt\rocm`). Expect hard-coded paths, formats and
> flags that change without notice, and no support or stability guarantees. Quantized containers are
> tied to the binary that reads them. Sustained load on both GPUs has caused Windows display resets
> and, once, a suspected power-supply shutdown on this machine, so watch your PSU headroom.

## Features

- **Hybrid model support:** 64 layers = 48 Gated DeltaNet + 16 full-attention layers, MTP head and
  vision tower.
- **Trellis quantization (QTIP/EXL3-style):** about 4.5 bits per weight (mixed 4/5-bit), rounded with
  LDLQ against per-layer Hessians and a random Hadamard rotation, decoded inside a native RDNA4
  WMMA GEMM. The 4-bit group layout w4a16 (group 32 or 64) serves the LM head, the MTP head and the
  DFlash2 drafter.
- **KV cache:** fp8 e4m3, paged in 16-token blocks, up to the model's native 262144-token context.
- **Speculative decoding:** MTP and DFlash2 (block-diffusion drafter), lossless in distribution.
- **Prefill:** 256-row chunks (the M = 256 trellis GEMM; kill switch `R4DX_PREFILL_CHUNK=0`) with two
  lossy-but-validated speedups ON BY DEFAULT: an int8 x int8 GEMM for the full 256-row chunks, at TP = 1 and TP = 2, with
  one scale per activation row and per weight column (greedy text unchanged on the canon prompts and at 8k, 32k, 64k,
  KL(f16 || int8) 0.0013 on the canon corpus; kill switches `R4DX_PREFILL_INT8=0`, `R4DX_PREFILL_INT8_SCALES=blk128`
  (finer per-128 scales), `R4DX_PREFILL_INT8_TP2=0` (f16 at TP = 2) and `R4DX_PREFILL_INT8_FUSEDQ=0` (separate
  quantizer launch); docs/int8-prefill.md) and split-KV attention from 2048 tokens of
  context, exact-wide below (kill switch `R4DX_PREFILL_SPLITKV=exact`, which is the lossless exact-wide
  launch; `=0` the plain dense launch; `R4DX_PREFILL_SPLITKV_MIN` moves the depth; docs/prefill.md). Prefill is
  no longer bit-identical to the 64-row f16 path: set `R4DX_PREFILL_INT8=0 R4DX_PREFILL_SPLITKV=exact` to get
  the old bytes back. A 64-row Model, tails shorter than 256 rows, images and quant2 containers keep the f16 GEMM.
- **Tensor parallel across two GPUs (`--tp 2`):** all-reduce through pinned host memory, since the
  cards have no peer-to-peer path.
- **Pipeline-parallel prefill (`--pp 2`, off by default):** layers split across the two cards for long
  prompts, about 1.7-1.8x faster cold prefill and byte-identical to the single-card prefill; decode stays
  on the headless card (docs/pp-prefill.md).
- **Hybrid two-GPU mode (`--tp 2 --pp 2`, off by default):** the prompt goes through the `--pp 2` pipeline,
  its state is resharded once into the TP layout, and decode runs as `--tp 2`: `--pp 2`'s prefill speed with
  `--tp 2`'s decode speed (docs/pp-tp2-hybrid.md).
- **Server:** `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (streaming, tools,
  images) and `POST /v1/completions`. One request at a time, with prefix reuse across turns.

## Performance

Huihui Qwen3.8-27B, trellis mix4.5m container (about 17 GiB), AMD Radeon AI PRO R9700, greedy decoding,
ROCm 10.1.0. Measured 2026-10-08 on an idle machine unless noted.

### Modes

Time to first token (TTFT) is a cold prefill of the whole prompt.

| Mode | Flags | TTFT 8k | TTFT 32k | TTFT 128k | Decode |
|---|---|---|---|---|---|
| One GPU | (default) | 3.15 s | 14.0 s | 80.6 s | 37.4 tok/s |
| Pipeline prefill | `--pp 2` | 1.77 s | 7.77 s | 44.2 s | 36.3 tok/s |
| Tensor parallel | `--tp 2` | 2.89 s | 12.1 s | – | 59.8 tok/s |
| **Hybrid** | `--tp 2 --pp 2` | **1.77 s** | **7.78 s** | **44.0 s** | **59.8 tok/s** |

### A typical request: 8k-token prompt, 512-token answer

| Mode | Plain | With DFlash2 (`--dflash`, k = 7) |
|---|---|---|
| `--pp 2` | 15.9 s | 9.2 s |
| `--tp 2` | 11.4 s | 7.9 s |
| **Hybrid** | **10.4 s** | **7.2 s** |

Which mode to use:
- **Hybrid** for long prompts with real answers (chat, coding agents, document Q&A).
- **`--pp 2`** when VRAM is tight or the desktop card should stay free while it generates.
- **`--tp 2`** when prompts are short (under about 1024 tokens).

### One GPU in detail

| | |
|---|---|
| Accuracy vs bf16 (teacher-forced) | mean KL 0.00788, top-1 agreement 95.70 % |
| Decode, plain | 37.4 tok/s |
| Decode, DFlash2 `k=7` | 123 tok/s |
| Prefill, short prompts | about 1130 tok/s |
| TTFT 8k / 32k / 64k | 3.15 s / 14.0 s / 31.9 s |

### Prefill settings (one GPU, TTFT)

| Setting | 8k | 32k | 64k |
|---|---|---|---|
| Default: int8 GEMM, coarse scales, split-KV attention | 3.15 s | 14.0 s | 31.9 s |
| Per-128 int8 scales (`R4DX_PREFILL_INT8_SCALES=blk128`) | 3.63 s | 16.0 s | 36.0 s |
| Kill switches: f16 GEMM, exact attention (`R4DX_PREFILL_INT8=0 R4DX_PREFILL_SPLITKV=exact`) | 4.66 s | 22.7 s | 56.5 s |

### Notes

- **Exactness.**
  - `--pp 2` gives byte-identical output to one GPU.
  - `--tp 2` and the hybrid decode across both cards, so they are not bit-identical to one GPU.
  - KL against one GPU is 0.0014–0.0023 for `--tp 2` and 0.0005–0.0006 for the hybrid.
  - The int8 prefill is not bit-identical to the f16 path: mean KL 0.0013 on the canon corpus. Greedy text at 8k, 32k and 64k is unchanged ([docs/int8-prefill.md](docs/int8-prefill.md)).
- **DFlash2 speed.** The speedup depends on the prompt, from about 3 to 6 accepted tokens per round. `--tp 2` reached 162 tok/s on the four standard prompts (2026-09-30), and `--mtp 3` 116 tok/s. Over six mixed prompts the hybrid averaged 3.10 tokens per round against 3.25 for `--tp 2`, because the two generate different text.
- **VRAM.** The hybrid keeps two copies of each card's share of the model. At a 128k context, 11.7 GiB stays free on the desktop card and 9.0 GiB on the decode card. `--pp 2` uses 12.6 GiB on the desktop card.
- **Hybrid thresholds.** Prompts under 1024 tokens, and calls past the stage-KV cap, use the `--tp 2` prefill.
- **Details and raw logs:**
  - [docs/pp-tp2-hybrid.md](docs/pp-tp2-hybrid.md) (hybrid)
  - [docs/pp-prefill.md](docs/pp-prefill.md) (`--pp 2`)
  - [docs/tp.md](docs/tp.md) (`--tp 2`)
  - [docs/perf.md](docs/perf.md) and [docs/prefill.md](docs/prefill.md) (methodology)
  - [docs/huihui.md](docs/huihui.md) (accuracy)

## Requirements

- Windows 11, the AMD HIP SDK (default location `C:\opt\rocm`) and an RDNA4 GPU (gfx1201). It is
  tested on 32 GB cards; the 27B container alone takes about 17 GiB, plus KV cache that scales with
  `--max-ctx`.
- CMake, Ninja and clang-cl. The exact toolchain, flags and known gotchas are in
  [docs/build-windows.md](docs/build-windows.md).
- A Qwen3.8-27B-family checkpoint. Weights are not distributed here. The default container was made from
  the community `Huihui-Qwen3.8-27B-abliterated` variant, which has its refusal behaviour removed; the
  engine itself works with any checkpoint of the same architecture.

## Build

```powershell
git clone <repo-url> r4dx
cd r4dx
.\build.ps1
```

This configures and builds the `win-hip` preset. Run the tests with `.\tests\run_tests.ps1`
(GPU tests use the device set in `HIP_VISIBLE_DEVICES`; several need converted containers and skip
without them).

## Usage

Convert a checkpoint into a container (trellis recipe, see [docs/quant2.md](docs/quant2.md) and
`tools/quant2/trellis_convert.ps1`), then run it:

```powershell
$env:HIP_VISIBLE_DEVICES = '1'

# one-shot generation
.\build\win-hip\src\cli\r4dx-cli.exe --model <container.r4dx> --layout trellis `
    --prompt "Write a haiku about GPUs." --max-tokens 128 --temperature 0 --stats

# OpenAI-compatible server with DFlash2 speculative decoding
.\build\win-hip\src\server\r4dx-server.exe --model <container.r4dx> --layout trellis `
    --dflash <drafter.r4dx> --dflash-k 7 --host 127.0.0.1 --port 8080
```

Useful flags: `--think on|off`, `--mtp K`, `--max-ctx N`, `--tokenizer-dir <dir>` (needs
`tokenizer.json`, `chat_template.jinja` and `generation_config.json`), `--image <path>` and
`--tp 2` or `--pp 2` (unset `HIP_VISIBLE_DEVICES` so both cards are visible; `--pp-devices B,A` picks
the decode and the prefill card by HIP ordinal, default decode on physical device 1) and, for the server only,
`--request-log <path>` (off by default: one JSON line of token counts and timings per request,
docs/server.md "Request log"; add `--request-log-tokens` to also record prompt and completion token ids
for offline speculation studies, `tools/ngram/README.md`). `--help` lists everything. The full
reference for the converter, CLI, server and tensor parallelism is in [docs/usage.md](docs/usage.md)
and [docs/server.md](docs/server.md).

## Repository layout

```
src/core/       device, stream, buffer and all-reduce plumbing
src/kernels/    r4dx-owned HIP kernels (norms, rope, sampling, tensor-parallel, ...)
src/model/      layer graph and forward pass
src/tokenizer/  tokenizer and chat template
src/convert/    checkpoint -> container converter
src/vision/     vision tower
src/server/     OpenAI-compatible server (r4dx-server)
src/cli/        command-line generator (r4dx-cli)
third_party/    libr4d GPU kernels and vendored header-only libraries
tests/          unit, integration and GPU tests
tools/          quantization, validation, benchmarking and reference scripts
docs/           design notes and measurements
```

## Documentation

| Topic | Page |
|---|---|
| Architecture and module map | [docs/architecture.md](docs/architecture.md) |
| Container format | [docs/container-format.md](docs/container-format.md) |
| Building on Windows | [docs/build-windows.md](docs/build-windows.md) |
| Trellis quantization and kernel | [docs/quant2.md](docs/quant2.md), [docs/trellis-kernel.md](docs/trellis-kernel.md) |
| Default model and baselines | [docs/huihui.md](docs/huihui.md) |
| Speculative decoding | [docs/mtp.md](docs/mtp.md), [docs/dflash2.md](docs/dflash2.md), [docs/sampling.md](docs/sampling.md) |
| Server and vision | [docs/server.md](docs/server.md), [docs/vision.md](docs/vision.md) |
| Tensor parallelism | [docs/tp.md](docs/tp.md) |
| Performance and validation | [docs/perf.md](docs/perf.md), [docs/prefill.md](docs/prefill.md), [docs/validation.md](docs/validation.md) |
| Pipeline-parallel 2-GPU prefill (`--pp 2`, validated) | [docs/pp-prefill.md](docs/pp-prefill.md) |
| Hybrid 2-GPU mode (`--tp 2 --pp 2`: PP prefill + TP decode) | [docs/pp-tp2-hybrid.md](docs/pp-tp2-hybrid.md) |
| Hardware notes | [docs/r9700.md](docs/r9700.md) |
| Detailed usage and history | [docs/usage.md](docs/usage.md), [docs/status.md](docs/status.md) |

## License and credits

r4dx is released under the [MIT License](LICENSE).

The GPU kernels in `third_party/libr4d` (vendored as plain sources) come from
[libr4d](https://codeberg.org/StillDeadcode/libr4d) by StillDeadcode and contributors. That project does
not state a licence, so its code is used here for experimentation, credited to its authors and removed
on request; see [NOTICE](NOTICE). The trellis GEMM and other additions built on top of it are part of r4dx.
Vendored header-only libraries (nlohmann/json, cpp-httplib, minja, stb_image) keep their own licences;
versions are listed in `third_party/VERSIONS.md`.
