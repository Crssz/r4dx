# Status

Last updated: 2026-09-20 (Milestone 2 integration pass).

## Milestone 2: done

`r4dx-server` (OpenAI-compatible `/v1/chat/completions` + `/v1/completions` + `/health` +
`/v1/models`, streaming and non-streaming, single-worker-thread/single-GPU) and MTP
self-speculative decode (`--mtp K`, greedy-only, real per-sequence KV cache built in lockstep with
the backbone) are implemented, tested, and verified end-to-end on HIP device 1, on top of a decode/
prefill performance pass (quantized attention `qg`/`o` projections, prefill-chunk waste removal, a
measured `(N,K,M-band)` GEMM tuning table, device-side greedy argmax) that landed in the same
integration window. See `docs/server.md` for the server's full API/concurrency-model writeup,
`docs/mtp.md` for MTP's design/incident/measurement writeup, and `docs/perf.md` for the full
before/after performance table spanning Milestone 1 through Milestone 2. This integration pass
re-ran a clean `build.ps1 -Clean` rebuild, the full `ctest` suite, `tools/server/smoke.ps1`, and one
`r4dx-cli` generation per quantized layout at both `--mtp 0` and `--mtp 3` against the real 64-layer
container to confirm the milestone is reproducible end to end; see "What passes" below for the
numbers.

## Milestone 1: done

Container loader, GDN + full-attention layers, the assembled model forward pass, and `r4dx-cli`
text generation are implemented, tested, and verified end-to-end against the real 64-layer
container on HIP device 1 for all four body layouts (mxfp4/w4a16/w4a8/bf16). See the "Update"
sections below for the assembly + review-fix narrative and `docs/perf.md` for the full perf table
and verbatim generated text. This integration pass re-ran a clean `build.ps1 -Clean` rebuild, the
full `ctest` suite, and one `r4dx-cli` generation per layout against `D:\models\r4dx\qwen38-27b.r4dx`
to confirm the milestone is reproducible end to end; see "What passes" below for the numbers.

## What exists

- **Repo skeleton**: CMake project (`win-hip` preset, clang-cl + Ninja driven from the
  `vLLM_for_AMD` venv's CMake 4.4.2), `build.ps1` / `tests/run_tests.ps1`, vendored header-only
  third-party deps (nlohmann/json, cpp-httplib, minja, stb_image), `third_party/libr4d` submodule
  (branch `windows-llp64` @ `7675605`). `docs/container-format.md` and `docs/architecture.md` are
  the converter/loader and forward-pass contracts.
- **`r4d_core`** (`third_party/CMakeLists.txt`): the 15 required libr4d translation units built via
  `hipcc.exe` (paged/vit attention, GDN chunk-scan/conv/kkt-solve/recurrent-update/gated-rmsnorm,
  the four GEMM families, quant_act_i8, dflash_conv, registry). Confirmed `r4d_registry` links
  without any `r4d_ar_*` objects (AR kernels referenced only as data), so no libr4d submodule patch
  was needed.
- **`src/core`**: HIP device/stream/event/buffer/arena/tensor plumbing (`r4dx::core`), header-only.
- **`src/kernels`**: r4dx-owned HIP kernels -- rmsnorm, residual add, rope (partial + mrope), silu_mul,
  fp8 e4m3 / int8 row activation quant, paged KV cache write, embedding gather, sampler (argmax /
  temperature / top-k / min-p).
- **`src/tokenizer`**: BPE tokenizer + chat template (minja), vendored llama.cpp Unicode tables.
  Golden-tested against the real Qwen3.8-27B `tokenizer.json`.
- **`src/convert`**: `r4dx-convert` CLI -- HF safetensors -> r4dx container, producing all three
  quantized GEMM layouts (MXFP4, INT4 g128 W4A16, INT4 g128 W4A8) pre-permuted into WMMA fragment
  order, plus bf16 passthrough for embeddings/vision/MTP-fusion tensors. Exercised end-to-end
  against the real checkpoint at `C:\AI\models\Qwen3.8-27B`, including `--mtp on`.
- **`tools/reference`**, **`tools/convert_ref`**, **`tools/tok_ref`**: read-only Python validation
  tooling against the reference `transformers` 5.17.0 venv -- per-layer goldens (GDN layer, full
  attention layer, final-norm/lm_head, MTP), KV descale calibration, converter cross-checks against
  the real `r4d_core` kernels, tokenizer golden generation. None of this touches the C++ engine
  directly; it produces ground truth for `src/model`'s future tests.
- **`src/model`**, **`src/server`**, **`src/cli`** are now all implemented -- see the "Update"
  sections below (Milestone 1: model graph, forward pass, CLI text generation) and `docs/server.md` /
  `docs/mtp.md` (Milestone 2: OpenAI-compatible server, MTP self-speculative decode).

## What passes

Clean `-Clean` rebuild (`build.ps1 -Clean`, HIP device 1) plus full `ctest --preset win-hip` run
(`tests/run_tests.ps1`) on 2026-09-19:

```
100% tests passed out of 17
```

| Test | Covers |
|---|---|
| `smoke_r4d` | `r4d_core` link/geometry/registry + `r4d_gemm_bf16_nt_m64` vs CPU fp32 ref |
| `reference_manifest` | CPU-only Python check of `tools/reference/layer_golden.py`'s manifest contract |
| `convert_quantize_roundtrip` | quantizer rel-L2 error bounds (w4a16/w4a8/mxfp4) on random data |
| `convert_pack_bytes` | packer byte-exactness vs Python-reference-generated fixtures |
| `convert_kernel_decode` | packed bytes decoded via each kernel's own literal indexing (independent of the packer/Python references) |
| `test_core` | device/stream/buffer/arena/tensor plumbing |
| `test_rmsnorm`, `test_residual_add`, `test_silu_mul`, `test_rope` | elementwise/norm kernels vs CPU reference |
| `test_quant_act_fp8` | fp8 e4m3 row activation quant |
| `test_mxfp4_gemm` | real quant_act_fp8e4m3_row -> real `r4d_gemm_mxfp4a8_nt_m64` vs CPU fp32 ref |
| `test_kv_write` | paged fp8 KV cache write layout |
| `test_sampler` | argmax / temperature / top-k / min-p sampling paths |
| `test_attn_decode` | paged decode attention, incl. multi-sequence per-head descale broadcast |
| `test_gdn_chunk_scan` | GDN chunked-scan kernel vs CPU reference |
| `tokenizer_golden` | 103 encode + 10 chat-template + 11 HF-compat-mode + 4 header-field cases, 0 failures |

Additional GPU-device-1 Python self-tests, run manually (outside `ctest`, serialized on device 1)
during this integration pass, all green:

- `tools/convert_ref/selftest_compare.py` -- byte-exact packer check against the real
  `r4dx-convert` binary's own `--selftest` output.
- `tools/convert_ref/kernel_crosscheck.py` -- w4a16/w4a8/mxfp4 rel_err all under the 2e-2 gate
  against the real `r4d_core` GEMM kernels.
- `tools/reference/layer_golden.py --device cuda` -- all four components
  (`layer_000_gdn`, `layer_003_full_attention`, `final_norm_lm_head`, `mtp`) status=ok against real
  checkpoint weights.
- `tools/reference/kv_calibrate.py --device cuda --layer 3` -- produced k_amax/v_amax descale
  values.

## Update (2026-09-19, assembly + CLI milestone)

`src/model` (container loader, GDN layer, MLP, final norm + lm_head, the full-attention layer under
`src/model/attention/`, and the assembled `r4dx::model::Model` forward pass in `model.{h,cpp}`) and
`src/cli` (`r4dx-cli`, chat-template-driven prompt/--chat loop, streaming UTF-8-safe decode,
prefill/decode tokens/s + VRAM stats) are now implemented and exercised end-to-end against the real
64-layer container (`D:\models\r4dx\qwen38-27b.r4dx`) on HIP device 1, for all four body layouts
(mxfp4/w4a16/w4a8/bf16) -- coherent, on-topic generated text for all four; see `docs/perf.md` for
the full perf table, verbatim outputs, and two real stream-synchronization bugs found and fixed
during this pass (`src/model/gdn_layer.cpp`'s `UploadArray`, `src/model/model.cpp`'s
`Model::RunChunk` logits readback -- both missing an explicit wait against a `hipStreamNonBlocking`
stream). `src/server` remains an unimplemented placeholder. `tests/model/test_forward_smoke` and
`tests/cli/test_args` are new; full `ctest --preset win-hip` is 24/24 passing as of this update.

## What passes (Milestone 1 integration pass, 2026-09-19)

Clean `-Clean` rebuild (`build.ps1 -Clean`, HIP device 1, 84/84 build steps) plus full
`ctest --preset win-hip` run (`tests\run_tests.ps1`):

```
100% tests passed out of 24
Total Test time (real) = 59.50 sec
```

All 17 Phase-0 tests plus `convert_kv_calib`, `convert_bf16_layout`, `test_gdn_layer`,
`test_final_lm_head`, `test_forward_smoke`, `test_attn_layer`, `test_cli_args` (the seven tests
added across the conversion/assembly/review-fix stages) pass together in one run.

One `r4dx-cli` generation per body layout against the real, full 64-layer container
(`D:\models\r4dx\qwen38-27b.r4dx`, 87.79 GiB), same prompt/settings as `docs/perf.md`
(`--prompt "Write a haiku about GPUs, then explain what a GPU is in two sentences." --max-tokens 128
--temperature 0 --stats`, `--max-ctx 2048` except bf16's `--max-ctx 512`):

| Layout | Container load | Prefill | Decode | Generated tokens | VRAM |
|---|---|---|---|---|---|
| mxfp4 | 9.73 s | 400.21 tok/s | 24.90 tok/s | 82 | 17.79 GiB |
| w4a16 | 16.01 s | 419.44 tok/s | 29.08 tok/s | 88 | 17.79 GiB |
| w4a8  | 16.62 s | 410.75 tok/s | 27.88 tok/s | 87 | 17.79 GiB |
| bf16  | 60.77 s | 20.40 tok/s  | 1.38 tok/s  | 74 | 31.86 GiB |

All four generations were coherent, on-topic, and stopped on the model's own EOS token, matching
the assembly/review-fix stages' own runs within noise -- confirming the milestone is reproducible
from a clean rebuild. See `docs/perf.md` for the verbatim generated text and full narrative.

## What passes (Milestone 2 integration pass, 2026-09-20)

Clean `-Clean` rebuild (`build.ps1 -Clean`, HIP device 1, 107/107 build steps) plus full
`ctest --preset win-hip` run (`tests\run_tests.ps1`):

```
100% tests passed out of 30
Total Test time (real) = 93.50 sec
```

All 24 Milestone-1 tests plus `test_mtp` and the five `test_server_*` CPU-only tests (`test_server_args`,
`test_openai_types`, `test_sse`, `test_response_sink`, `test_request_queue`) pass together in one run.

`tools/server/smoke.ps1` (4-layer test container, `--layout w4a16 --layers 4`): all 20 shape/status
checks passed -- `/v1/models`, non-streaming and streaming `/v1/chat/completions` (SSE framing,
`[DONE]` terminator, per-event `chat.completion.chunk` shape), and a rejected-image-part `400`.

One `r4dx-cli` generation per quantized body layout, `--mtp 0` vs `--mtp 3`, against the real,
unmodified 64-layer container (`D:\models\r4dx\qwen38-27b.r4dx`), same prompt as `docs/perf.md`
(`--temperature 0 --max-ctx 2048 --stats`, `--max-tokens 128` except bf16's `--max-tokens 32
--max-ctx 512`):

| Layout | mtp=0 decode | mtp=3 decode | mtp=3 acceptance | speedup |
|---|---|---|---|---|
| mxfp4 | 27.08 tok/s | 47.46 tok/s | 41.9% | +75.3% |
| w4a16 | 32.83 tok/s | 66.42 tok/s | 54.3% | +102.3% |
| w4a8  | 30.95 tok/s | 47.72 tok/s | 32.5% | +54.2% |
| bf16  | 1.41 tok/s  | 2.21 tok/s  | 48.7% | +56.7% |

All eight runs were coherent, on-topic, and (except the two 32-token-capped bf16 runs, which hit
`--max-tokens` by design to keep the sweep's wall-clock bounded) stopped on the model's own EOS
token, matching the FIX pass's own numbers within run-to-run noise -- confirming Milestone 2 is
reproducible from a clean rebuild. See `docs/perf.md` and `docs/mtp.md` for the full per-layout
tables (all five `--mtp` values, not just 0 and 3) and verbatim generated text.

## Known gaps

- `r4d_gdn_conv_prep_w4_h128_bf16` / `conv_update` both live in the single
  `r4d_gdn_conv_w4_h128_bf16` translation unit per `r4d.h`; no gap, just worth remembering when
  wiring `src/model`.
- Vision tower weights are carried bf16-only for now (no quantized vision GEMM path yet); vision
  tower forward pass itself is the next milestone (see "Next milestone" below).
- fp8 KV descales are calibrated per-layer on demand via `kv_calibrate.py`
  (`tools/reference/kv_calibrate_out/kv_descale.json`, gitignored) but not yet wired into the
  converter -- `docs/container-format.md`'s descale table is still the placeholder `1.0` until
  `src/convert` consumes calibration output for all 16 full-attention layers.
- `tests/reference/test_manifest.py` is CPU-only and has no `pytest` dependency (the reference venv
  doesn't have `pytest` installed and is read-only) -- it's a plain script with bare asserts, run
  directly and also registered as the `reference_manifest` ctest test.
- No prefill kernel yet beyond the interim 64-row skinny-GEMM chunking path, now backed by a
  measured `(N,K,M-band)` GEMM tuning table (`docs/perf.md`'s "GEMM tuning sweep") rather than a
  single hardcoded tuple, but still not a dedicated WMMA prefill kernel; that remains future work.
- MTP acceptance (30-75%, best at low K) is well above the pre-fix 0-1.2% but still plausibly below
  what a purpose-trained self-speculative head could achieve -- not investigated further; see
  `docs/mtp.md`'s "Known gaps".
- `r4dx-server`'s tool-call parsing and vision content parts are both deferred (`docs/server.md`'s
  "Deferred / known gaps"); a mismatched-prefix conversation reset still pays a full container
  reload's latency (no lightweight `Model::Reset()` yet).
- `--chat` multi-turn only lightly exercised, now also true of `--chat` + MTP interaction together
  (`docs/mtp.md`'s "Known gaps").

## Next milestone

**Milestone 2** (`r4dx-server` OpenAI-compatible chat API, decode/prefill performance pass, MTP
self-speculative decode) is done -- see the "Milestone 2: done" section above, `docs/server.md`,
`docs/mtp.md`, and `docs/perf.md`. Next up: the vision tower forward pass, then DFlash2 drafting,
followed by prefix-caching, per the top-level project decisions.
