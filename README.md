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

Sets `HIP_VISIBLE_DEVICES=1` and runs `ctest` against the `win-hip` build directory: 62 tests
covering `r4d_core` smoke, `src/core`/`src/kernels` device-buffer and kernel unit tests (rmsnorm,
residual add, silu_mul, rope (including the 3-axis mrope partial-rotary kernel), fp8/int8
activation quant, kv cache write, mxfp4 GEMM, attention decode, GDN chunk scan, sampler, the P6
vectorized-kernel bandwidth golden, the P2 fused-quant-epilogue byte-diff harness, embedding-gather
in-range/OOB-clamp, the R9 reduced-vocab-draft-head gather-by-index kernel's in-range/OOB-clamp
golden, and the vision tower's six device primitives), the converter's quantizer
round-trip / byte-packer / kernel-decode / KV-calibration / bf16-layout tests, the tokenizer's
golden-case suite, `src/model`'s per-layer tests (GDN layer, full-attention layer,
final-norm+lm_head, the mrope-carrying attention layer, assembled-`Model` forward-pass smoke
including a prefill/decode state-handoff equivalence check and a `Model::Reset()` byte-identity
check, MTP's verify/rejection-rewind/mid-round-commit/K=16-wide-window/reduced-vocab-draft-head-
lossless tests, and the pure-CPU `mtp_round` commit-bookkeeping tests, including a K=16 wide-round
case), `src/cli`'s argument-parsing tests (including `--mtp-draft-head` and `--image`), `src/server`'s
CPU-only tests (CLI args including `--mtp-draft-head`, OpenAI request/response JSON shapes including
`tools`/`tool_choice`/`role: "tool"`/`"function"` parsing and image content parts, SSE framing,
buffering/streaming sinks including the tool-calls streaming chunk shape, the bounded request queue,
`PrefixState` including its image-aware key, and the tool-call surface-syntax parser --
`docs/server.md`'s "Tool calls"), the vision tower's tests (the CPU-only preprocessing / mrope /
index-math goldens, the whole tower against the real checkpoint's forward, and the shared
`ExpandImagePlaceholders` image-prompt-splicing tests -- `docs/vision.md`), and a
CPU-only Python reference-manifest check (`tests/reference/test_manifest.py`, run through the same
`ctest` invocation). 61 pass and 1 skips (`test_kernel_bandwidth`, whose golden is gitignored),
~636s wall on HIP device 1. See `docs/status.md` for
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
    --kv-calib D:\models\r4dx\qwen38-27b.kvcalib.json `
    --quant search --imatrix D:\models\r4dx\qwen38-27b.imatrix.npz
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

**`--quant` / `--imatrix` -- how the 4-bit values are chosen.** Neither flag changes a single byte of
the on-disk *layout* (docs/container-format.md, "How the quantized values are chosen"); they change
which `q` / `scale` / `zero` values land in those bytes, so any container is readable by any loader
either way, at exactly the same decode speed. `--quant search` replaces the historical min/max +
round-to-nearest grid with a per-`(row, 128-K group)` search over 21 candidate scales
(`0.85x .. 1.15x`) and three candidate integer zeros, plus a weighted least-squares refit of the
scale. `--quant rtn` is the **default** and is the historical behaviour byte for byte.

**Use them as a pair: `--quant search --imatrix <npz>`, or not at all.** `--imatrix <npz>` weights
the search's error term by each input channel's mean activation energy, from the importance matrix
`tools/reference/imatrix_capture.py` captures over the calibration corpus
(`D:\models\r4dx\qwen38-27b.imatrix.npz`, keyed by the converter's own container base names); it
requires `--quant search`. The weighting is not a refinement, it is the whole mechanism -- measured
on the real checkpoint against the bf16 reference (`docs/validation.md` "Milestone 10"):

| mode | w4a16 mean KL / top-1 | w4a8 | mxfp4 |
|---|---|---|---|
| `--quant rtn` (default) | 0.0724 / 88.4% | 0.1434 / 82.72% | 0.0912 / 86.14% |
| `--quant search` alone | 0.0713 / 87.71% | -- | -- |
| `--quant search --imatrix` | **0.0534 / 89.30%** | **0.1164 / 84.78%** | **0.0797 / 86.09%** |

The search alone lowers its own per-group objective on every single group and still buys nothing at
model level (top-1 is 0.7 points *worse* than `rtn`), which is why `rtn` stays the default and why
`--quant search` without `--imatrix` is not worth its ~2x conversion wall time (4-layer container:
18.9 s -> 41.0 s; the full 64-layer search+imatrix container takes 276 s). The "search is never
worse than rtn" property that `tests/convert/test_quant_search.cpp` gates is a statement about the
weighted squared reconstruction error of one `(row, group)`, not about KL or top-1.

The run logs how many linears it weighted and how many fell back to unweighted MSE; on this
checkpoint that is `341 weighted, 0 fell back`. Anything else means the `.npz` is stale for this
checkpoint and part of the model was quantized unweighted -- the coverage line says `WARNING` and
goes to stderr in that case.

### Generate text

```powershell
$env:HIP_VISIBLE_DEVICES = '1'
.\build\win-hip\src\cli\r4dx-cli.exe --model D:\models\r4dx\qwen38-27b-v3.r4dx --layout w4a16 `
    --prompt "Write a haiku about GPUs, then explain what a GPU is in two sentences." `
    --max-tokens 128 --temperature 0 --stats
```

Add `--image <path>` (repeatable, any container the vision tower loaded from) to ask about a
picture (docs/vision.md):

```powershell
.\build\win-hip\src\cli\r4dx-cli.exe --model D:\models\r4dx\qwen38-27b-v3.r4dx --layout w4a16 `
    --image photo.png --prompt "What is in this picture?" --max-tokens 128 --temperature 0 --stats
```

In `--chat`, attach an image to the NEXT turn with one or more leading `/image <path>` lines typed
into the REPL before the question itself.

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
GDN state allocation (default **262144**, matching the checkpoint's own `max_position_embeddings`
-- raised from a stale 131072 self-imposed cap, docs/r9700.md R13, 2026-09-20: real hardware
measurement shows the full 262144-token KV+GDN allocation still leaves 21-24% of the R9700's 32 GiB
free, and generation at 262144 real prefilled tokens is coherent and correct, see
`docs/perf.md`'s "Long-context validation"; the bf16 layout needs a much smaller value regardless,
since its 47.73 GiB of weights alone do not fit on this card -- see `docs/perf.md`). `--stats`
prints container-load time, prefill/decode tokens/s, and VRAM used, plus
(when `--mtp K>0`) an MTP acceptance-rate line. `--mtp K` (default 0) enables MTP self-speculative
decode: each decode round drafts up to `K` tokens via the checkpoint's own `mtp.*` weights, verifies
them against the real model in one batched call, and commits the accepted prefix (plus one corrected/
bonus token) -- **runs at any `--temperature` as of Milestone 6 stage S3**: `--temperature <= 0`
accepts a draft iff it equals the target's argmax (unchanged); `--temperature > 0` accepts it by
"sample-and-match" rejection sampling, which is lossless IN DISTRIBUTION (exactly one uniform draw
per emitted token, against the request's own post-filter distribution). For a fixed `--seed` the
emitted tokens match plain sampled decode's only up to a pre-existing, unrelated numeric mechanism:
a speculative round's logits come from one batched GEMM pass whose reduction order differs from
single-row decode's, which a CDF walk (unlike a greedy argmax) can be sensitive to on near-tie
candidates -- on the real 64-layer container this makes most sampled trajectories diverge from
plain decode's own text somewhere (measured 6/8 96-token trajectories in one sweep), though every
diverging token is still a legitimate canonical sample of the round that actually ran
(`docs/sampling.md` sections 9.3, 11-12, `tools/validate_spec_sampling.ps1`). See `docs/mtp.md` for
the full design, the container requirement (the container must carry
`mtp.*` weights -- `D:\models\r4dx\qwen38-27b.r4dx` already does), and measured acceptance/speedup
per layout (roughly +75-165% decode throughput over `--mtp 0` at each layout's best `K` -- w4a16
`K=3`, w4a8 `K=4`, mxfp4 `K=3`, see `docs/perf.md`'s consolidated table -- `--mtp 3` a reasonable
default). See `src/cli/cli_args.h` for the full flag list and `docs/perf.md` for measured throughput
per layout.

`--dflash <draft.r4dx>` (Milestone 5, docs/dflash2.md) enables DFlash2 block-diffusion
self-speculative decode instead of MTP -- mutually exclusive with `--mtp`, runs at any
`--temperature` (same sample-and-match acceptance, lossless in distribution, as `--mtp` above --
including the same batched-verify numeric caveat on a fixed-seed token-for-token match -- as of
Milestone 6 stage S3), needs a separate DFlash2 draft container
(`D:\models\r4dx\qwen38-27b-dflash2-{w4a16,w4a8,mxfp4,bf16}.r4dx`),
independent of the target's own `--layout`. `--dflash-k N` (1..7, default 7), `--dflash-p-min F`
and `--dflash-n-min N` tune the selector walk's early-stop/discard gates. Best measured so far
(docs/perf.md's Integrate-stage final confirmation sweep, each layout's own best `--dflash-k`
vs. best `--mtp K`, twice each): on a ~270-token code prompt, DFlash2 **beats MTP on all three
layouts** -- w4a16 **116.90/116.84 tok/s** (77.4% acceptance) vs MTP's 89.45/89.44 (+30.7%), w4a8
**109.22/109.29** vs 87.53/87.71 (+24.6%), mxfp4 **94.30/94.30** vs 75.50/75.68 (+24.9%). On the
standard haiku prompt DFlash2 still beats MTP on w4a16 (77.08/77.05 vs 68.72/68.62, +12.2%) and w4a8
(64.79/64.76 vs 57.69/57.66, +12.3%), but MTP still leads on mxfp4 (64.91/64.91 vs 58.57/58.63,
-9.8%) -- prompt-dependent, not a fixed ranking. Best single cell (w4a16/code) is within 3% of the
120 tok/s / 84% acceptance the reference ROCmFPX implementation reaches on this same card/draft. The
`p_min` sweep, the w4a8/mxfp4 DRAFT containers, and the mxfp4/standard-prompt acceptance gap are
still open (docs/dflash2.md section 7a, docs/perf.md's top section).

**Sampled (`--temperature > 0`) traffic, Milestone 6 stage S3.** The numbers above were all greedy
(`--temperature 0`) -- the setting real chat clients almost never use. As of this stage, `--mtp`/
`--dflash` speculation and the plain decode path's device-row-summary sampler
([sampling.md](docs/sampling.md)) both run at any temperature, lossless IN DISTRIBUTION -- a
fixed-seed sampled speculative request's text matches plain sampled decode's only up to the
pre-existing batched-verify numeric mechanism described above, which in practice moves most
trajectories on the real container. Measured, real 64-layer container: plain sampled decode's tax
over greedy fell from 6.2-7.5% to statistical parity (within about -0.8% to +0.7%, re-confirmed by
the Integrate stage), and w4a16's best sampled `--dflash k=7` cell reaches **154.28, 154.23 tok/s**
on a code prompt (4.28x plain sampled decode's pre-stage cost) -- see `docs/perf.md`'s top section
for the full matrix (all three sampling configs x both prompts x all three layouts, twice each) and
`docs/sampling.md` section 12.

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
only the new tail (via a cheap `Model::Reset()`, not a full reload, on any state-handling boundary
where the engine must catch up), or reloading from scratch only on a genuine prefix mismatch.
`--mtp N` (Milestone 3) enables server-side MTP self-speculative decode identically to `r4dx-cli
--mtp`: a request takes the MTP path iff the server was started with `--mtp N>0` against an
MTP-converted container AND that request is greedy (`temperature <= 0`); every accepted token still
streams as soon as it is committed. `--mtp-head-layout {bf16,layout}` is also a server-side flag now (mirrors the CLI), as is
`--mtp-draft-head {reduced,full}` (docs/r9700.md R9, "reduced-vocab draft head" -- see docs/mtp.md
for the design, measured coverage, and K-sweep). `--dflash <draft.r4dx>` (Milestone 5,
docs/dflash2.md) is the same server-side passthrough for DFlash2 as the CLI's own flag above,
mutually exclusive with `--mtp`; `tools/server/smoke.ps1 -Dflash <path>` verified it end to end
against the real container (streaming and tool-call mode both unaffected). Tool calls (OpenAI `tools`/`tool_choice`/
`message.tool_calls`/`role: "tool"` multi-turn round trips) are fully supported -- `tools` are
rendered into the prompt and a model-emitted `<tool_call>` is parsed back into a structured
`message.tool_calls` response (JSON-encoded `arguments` string, stable generated `id`s,
`finish_reason: "tool_calls"`), with malformed/unknown-tool output degrading to plain content
rather than erroring; see `docs/server.md`'s "Tool calls" section for the confirmed model surface
syntax, `tool_choice` coverage, and the streaming (buffer-whole) decision. Images (Milestone 8
stage 5, docs/server.md's "Images"): standard OpenAI `image_url`/`input_image` content parts, any
number of images in any position across a conversation, against a container whose vision tower is
loaded --

```powershell
$b64 = [Convert]::ToBase64String([IO.File]::ReadAllBytes("photo.png"))
curl.exe -s http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d @"
{"messages":[{"role":"user","content":[
  {"type":"image_url","image_url":{"url":"data:image/png;base64,$b64"}},
  {"type":"text","text":"What is in this picture?"}]}],
 "max_tokens":128,"temperature":0}
"@
```

A remote (`http://`/`https://`) `image_url` is never fetched (a clean `400`); an image against a
container with no vision tower loaded is also a clean `400` naming the reason. See `docs/server.md`
for the full endpoint/field reference and a captured real streamed answer, and
`tools/server/smoke.ps1` for the GPU integration smoke test (`.\tools\server\smoke.ps1` against the
small 4-layer test container by default; pass `-Model`/`-Layout`/`-Layers -1`/`-Mtp N` to point it
at a real container with MTP enabled, `-ToolRoundTrip` to exercise a real tool call/result/answer
round trip, and `-Vision` to exercise the full image suite -- description, OCR, multi-image,
image+tools, image+thinking, streaming, and multi-turn prefix reuse -- against a vision-capable
container).

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

**Rung 4 (teacher-forced KL vs bf16) measured and audited (2026-09-22).** `w4a16` overall mean KL
0.08794 nats / 87.00% top-1 agreement vs the original bf16 checkpoint on a held-out 4-segment
corpus. The number survived an adversarial audit (`tools/reference/kl_audit.py`,
`tools/reference/reference_selfcheck.py`): the bf16 reference matches a plain `transformers`
forward to one bf16 ulp per position, its own self-noise at 64 layers is 3.9e-04 nats, and the
`thai_prose` outlier is the extended-vocabulary tail of the 4-bit `lm_head`, not a measurement bug
-- see `docs/validation.md`'s "Rung 4 measurement: w4a16" and "Auditing this measurement".

**Milestone 8 done, integrated and measured (2026-09-22, stage 8).** Every gate green on a clean
`build.ps1 -Clean`: `ctest` 62 registered/61 passed/1 skipped, `tools/validate_dflash.ps1
-AllowBatchedVerifyDivergence` passed with the same 4-identical/5-known-divergence result as
before, and `tools/server/smoke.ps1` clean on the default 4-layer container, the real container with
`-Dflash -ToolRoundTrip`, and the real container with `-Vision -Dflash` (191 PASS, 0 FAIL). Headline
numbers, real 64-layer container, `w4a16`, `--dflash k=7` server config, greedy, twice each:

| | 448x448 | 1024x1024 | 1536x1536 (default cap) |
|---|---|---|---|
| Image tokens | 196 | 1024 | 1024 (downsized) |
| Encode ms | 33-34 | 159-160 | 160-161 |

Prefill with an image: ~995-1003 tok/s (same rate as text). Decode + acceptance on a 1024x1024
image prompt: plain 38.5-38.6 tok/s, `--mtp 3` 79.4-79.5 tok/s (58.5% acceptance), `--dflash k=7`
86.9 tok/s (33.1% acceptance). VRAM: `--vision off` 16.17 GiB vs `--vision auto` 17.03-17.04 GiB
(+0.86-0.87 GiB). Text-only decode with the tower resident: 38.75-38.81 tok/s -- no regression
against this project's own w4a16 baseline range. Full table: `docs/perf.md`'s "Milestone 8, stage
8" section. Known gaps (video input, remote/webp images, a real-photograph end-to-end check, the
non-ASCII prefix-cache round-trip gap): `docs/status.md`'s Milestone 8 stage 8 entry.

**User-facing image input shipped (2026-09-22, stage 5).** `--image <path>` (repeatable) on
`r4dx-cli`, `/image <path>` lines in `--chat`, and OpenAI-shaped `image_url`/`input_image` content
parts on `r4dx-server`'s `/v1/chat/completions` -- real-hardware verified: describing a synthetic
image, reading a rendered string back off an OCR image exactly, two images in one request, image +
tools, image + thinking, streaming, and an image-aware prefix cache (a turn that reuses an earlier
turn's own image does not re-encode it; a different image at the same conversation position never
reuses the wrong turn's KV state). A container with no vision tower now answers a clean `400`
naming the reason instead of the old blanket "not implemented". Full detail:
`docs/vision.md`'s "User-facing wiring: --image and image_url", `docs/server.md`'s "Images".

**The model answers questions about a picture (2026-09-22).** The vision tower's merged rows are
now spliced into the text embedding sequence at the image-placeholder positions, and 3-axis
`(t, h, w)` mrope position ids reach every rope call site in the decode stack -- prefill (chunked
included), plain decode, MTP verify and draft, DFlash2 injection and draft blocks. With an image in
the prompt a token's rope position and its KV slot index stop being the same number, permanently
for the rest of the conversation, so `Model::PrefillMultimodal` records the mrope delta and every
later step ropes at `sequence index + delta` while its slot stays the sequence index. Greedy, on
the real container: the synthetic golden image is described correctly (gradient, checkerboard,
circle); a three-bar chart's count, colours and ordering are all correct; five circles are counted
as `5`; `R4DX7391` is read exactly off a rendered text image. `--mtp 3` and `--dflash k=7` produce
byte-identical output to plain decode with acceptance *higher* than the same question asked
without a picture (2.91 vs 2.21 and 3.14 vs 2.52 tokens/round). Text-only generation is
byte-identical to a build of the last pre-vision commit on all three paths. ~~`--image` on the CLI
and `image_url` on the server are the next stage; `image_url` is still rejected with `400`.~~
**Done, 2026-09-22, stage 5 -- see the entry above.** Full
detail, including why the ring slot and the rope position deliberately part company for DFlash2:
`docs/vision.md`.

**Vision tower, on the GPU (2026-09-21).** `Qwen3_5VisionModel`'s forward now runs on device:
`src/vision` loads the container's 333 `vision.*` bf16 tensors (0.9154 GiB measured) and runs patch
embed -> the learned 48x48 position grid -> 27 encoder blocks over `r4d_attn_vit_h72_bf16` -> the
2x2 patch merger, validated tensor by tensor against the real checkpoint's own forward (56 tensors
across three golden cases, including every encoder block's output and a two-image batch). Encode
cost, real container, best of 3: **27.7 ms for 448x448, 149.5 ms for 1024x1024, 396.7 ms for
1536x1536** -- and 2048x2048 still fits next to the loaded 27B at the full 262144-token KV
allocation with 6.5 GiB free. `--vision {auto|on|off}` decides whether the 0.9 GiB is paid at all
(text-only output is byte-identical either way) and `--image-max-pixels N` (default 1024x1024)
downsizes a large attachment through the reference's own `smart_resize` rather than rejecting it.
Full detail, including why the deep-block numeric disagreement is the reference's own bf16
attention rather than an r4dx error: `docs/vision.md`.

Milestone 1 (container loader, GDN + attention layers, model forward, `r4dx-cli` text generation),
Milestone 2 (`r4dx-server` OpenAI-compatible chat API, a decode/prefill performance pass, and MTP
self-speculative decode), Milestone 3 (quantized `gdn.in_proj_z`/`attn.k`/`attn.v` + a 45 GiB real
container, fused residual+rmsnorm (R3), vectorized rmsnorm/residual_rmsnorm/silu_mul kernels (P6), a
configurable/measured MTP head layout, a device-resident embedding gather + MTP draft loop, and
server-side `Model::Reset()`/MTP/prefix-reuse hardening), and **Milestone 4** are all complete and
integrated. Milestone 4 (2026-09-20) root-caused and enabled the fused activation-quant epilogues
(R2/P2) for w4a8/mxfp4 (an `r4dx::core::Arena::Alloc` end-alignment gap; w4a16 stays unfused, a
separate measured wall-clock regression, not a correctness issue); re-swept
`src/model/gemm_tuning_table.inc` with a Q5-fixed methodology that eliminates cache-flattery
(mxfp4 decode improved, w4a16 flat, w4a8 `--mtp 3` regressed -- attributed to verify-band GEMM
retiling + numerical reduction-order drift, not a correctness bug); root-caused the MTP acceptance
gap to h_seed drift on one outlier residual dimension (a measured, expected quantization effect, not
a bug); built a reduced-vocab MTP draft head end to end (container format, loader, kernel, `K`
widened to 16) -- mechanism verified lossless on real hardware, but its ~2.5-3x economic projection
was not realized because this machine's only calibration corpus (WikiText-2) is too small; validated
long-context generation to the model's own native 262144-token ceiling and raised `--max-ctx`'s
default accordingly; documented the vision tower's full architecture against real `transformers`
source with real-hardware validation goldens (`docs/vision.md`) but did not implement its C++; and
shipped full OpenAI `tools`/`tool_choice`/`role:"tool"`/`"function"` support, hardened by a dedicated
review pass (8 findings, all fixed and regression-tested) -- see "Tool calls" below and
`docs/server.md`. See `docs/status.md` for the full Milestone 4 work-item table, what exists, what
passes, known gaps, and the proposed next milestone (the tiled WMMA prefill GEMM kernel (R10/P9),
then the vision tower's C++, then DFlash2 drafting).

Headline decode throughput on the real 64-layer container (each layout's own best `--mtp K`, HIP
device 1, post-Milestone-4 integration): **w4a16 68.73 tok/s (`K=3`, 46.3% acceptance), w4a8 57.86
tok/s (`K=4`, 31.0%), mxfp4 65.04 tok/s (`K=3`, 52.9%)** -- see `docs/perf.md`'s consolidated
Milestone 1 -> 2 -> 3 -> 4 table for the full progression.
