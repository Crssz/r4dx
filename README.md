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
- **Prefill:** 256-row chunks (the M = 256 trellis GEMM; bit-identical to 64-row chunks, kill switch
  `R4DX_PREFILL_CHUNK=0`) with a fused, exact-wide attention kernel; a faster split-KV variant is opt-in
  (`R4DX_PREFILL_SPLITKV=split`).
- **Tensor parallel across two GPUs (`--tp 2`):** all-reduce through pinned host memory, since the
  cards have no peer-to-peer path.
- **Server:** `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (streaming, tools,
  images) and `POST /v1/completions`. One request at a time, with prefix reuse across turns.

## Performance

Huihui trellis mix4.5m container, one R9700, greedy decoding:

| | |
|---|---|
| Accuracy vs bf16 (teacher-forced) | mean KL 0.00788, top-1 agreement 95.70% |
| Weights | about 17 GiB |
| Decode, plain | 36.7 tok/s |
| Decode, DFlash2 `k=7` | 108 tok/s |
| Prefill, short prompts | about 1130 tok/s |
| Cold prefill (time to first token) | 5.7 s at 8k, 26.7 s at 32k tokens (256-row chunks, HIP device 1; 7.7 s and 35.2 s with `R4DX_PREFILL_CHUNK=0`). 64-row chunks, measured earlier: 74 s at 64k, 194 s at 128k |

With `--tp 2` on two R9700s (same container, measured 2026-09-30 against a single-card baseline from the
same session), plain decode reaches 60.1 tok/s (1.70x one card), DFlash2 `k=7` 162 tok/s (1.58x) and
`--mtp 3` 116 tok/s (1.55x); cold prefill takes 3.9 s at 8k, 19.1 s at 32k and 49.0 s at 64k tokens
(1.33x to 1.44x faster than one card; 256-row chunks). Methodology and more numbers: [docs/perf.md](docs/perf.md), [docs/huihui.md](docs/huihui.md),
[docs/prefill.md](docs/prefill.md).

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
`--tp 2` (unset `HIP_VISIBLE_DEVICES` so both cards are visible). `--help` lists everything. The full
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
