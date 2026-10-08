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
  prompts, about 1.9x faster cold prefill and byte-identical to the single-card prefill; decode stays
  on the headless card (docs/pp-prefill.md).
- **Server:** `GET /health`, `GET /v1/models`, `POST /v1/chat/completions` (streaming, tools,
  images) and `POST /v1/completions`. One request at a time, with prefix reuse across turns.

## Performance

Huihui trellis mix4.5m container, one R9700, greedy decoding:

| | |
|---|---|
| Accuracy vs bf16 (teacher-forced) | mean KL 0.00788, top-1 agreement 95.70% |
| Weights | about 17 GiB (+15 MiB of int8 weight scale tables for the default int8 prefill; `R4DX_PREFILL_INT8_SCALES=blk128` makes them 0.7 GiB, `R4DX_PREFILL_INT8=0` skips them) |
| Decode, plain | 37.4 tok/s |
| Decode, DFlash2 `k=7` | 123 tok/s (write-once GDN state; 119 with `R4DX_GDN_WRITE_ONCE=0`; HIP device 1, 2026-10-08, `E:\models\r4dx\round2\bench_gdnwo`) |
| Prefill, short prompts | about 1130 tok/s |
| Cold prefill (time to first token), defaults (int8 prefill GEMM with coarse scales + split-KV attention) | 3.15 s at 8k, 14.0 s at 32k, 31.9 s at 64k tokens (HIP device 1, ROCm 10.1.0, 2026-10-08, 2 runs each; `E:\models\r4dx\int8v2\ttft_summary.txt`) |
| Cold prefill, two GPUs (`--pp 2`, pipeline-parallel prefill, decode unchanged on one card) | 1.70 s at 8k (1.85x), 7.39 s at 32k (1.90x), 16.8 s at 64k tokens (1.91x) against the single-card row above, same session, median of 2 runs each, greedy output identical (HIP ROCm 10.1.0, 2026-10-08; `E:\models\r4dx\pp2\ttft_summary.txt`; docs/pp-prefill.md). Decode with `--pp 2`: 0.997x tok/s, text identical. Off by default |
| Cold prefill, per-128 int8 scales (`R4DX_PREFILL_INT8_SCALES=blk128`, the 2026-10-07 default) | 3.63 s at 8k (2230 tok/s), 16.0 s at 32k (2050 tok/s), 36.0 s at 64k tokens (HIP device 1, ROCm 10.1.0, 2026-10-07, 2 runs each) |
| Cold prefill, kill switches (`R4DX_PREFILL_INT8=0 R4DX_PREFILL_SPLITKV=exact`, f16 GEMM + exact-wide attention) | 4.66 s at 8k, 22.7 s at 32k, 56.5 s at 64k tokens (same session, interleaved with the row above; 7.0 s and 32.3 s at 8k/32k with `R4DX_PREFILL_CHUNK=0`). 64-row chunks, measured earlier: 74 s at 64k, 194 s at 128k |

The defaults are about 32% (8k), 38% (32k) and 44% (64k) faster than the kill switches (whose figures are from the earlier session) and 13% /
12% / 11% faster than the per-128 scales. They are not bit-identical to the f16/exact path: mean KL 0.0013 on the canon corpus
(p99 0.0115, top-1 agreement 98.63%) and 0.0050 over the long-context set, with code_8k at 0.0034 and code_32k at 0.0199 (the
two segments to watch; docs/int8-prefill.md "Defaults on int8v2"); greedy text at 8k, 32k and 64k is unchanged, and the 8k/32k
long-context task set (7 tasks x 8 items x 2 lengths) scores 100.0 / 98.2, the same as with the per-128 scales (100 of 112
outputs identical, 0 score changes) against 100.0 / 97.5 with the kill switches. Decode is bit-identical to before (the
bit-exact decode cuts, docs/perf.md "decode-t1", measured +2.3% plain and +4.7% DFlash2 `k=7` on their branch).

With `--tp 2` on two R9700s (same container, measured 2026-09-30 against a single-card baseline from the
same session), plain decode reaches 60.1 tok/s (1.70x one card), DFlash2 `k=7` 162 tok/s (1.58x) and
`--mtp 3` 116 tok/s (1.55x); cold prefill took 3.9 s at 8k, 19.1 s at 32k and 49.0 s at 64k tokens
(1.33x to 1.44x faster than one card; 256-row chunks). With the int8 prefill GEMM now on at TP = 2 as well (coarse scales, 2026-10-08,
`E:\models\r4dx\int8v2\tp2\ttft_summary.txt`), cold prefill takes 2.88 s at 8k, 12.1 s at 32k and 25.5 s at 64k tokens
(-19%, -16% and -16% against TP = 2 f16 in the same session; `R4DX_PREFILL_INT8_TP2=0` restores f16; the canon KL p99 of 0.0146 is over
the 0.012 gate, see docs/int8-prefill.md). Methodology and more numbers: [docs/perf.md](docs/perf.md), [docs/huihui.md](docs/huihui.md),
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
