# Status

Last updated: 2026-09-19 (Milestone 1 integration pass).

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
- **Not yet implemented**: `src/model` (layer graph / forward pass), `src/server` (OpenAI chat API),
  `src/cli` (text-generation CLI) -- all three currently expose only a placeholder INTERFACE CMake
  target, per the Skeleton stage's design for later agents to fill in.

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

## Known gaps

- `src/server` is an unimplemented placeholder -- no OpenAI-compatible server yet.
- `r4d_gdn_conv_prep_w4_h128_bf16` / `conv_update` both live in the single
  `r4d_gdn_conv_w4_h128_bf16` translation unit per `r4d.h`; no gap, just worth remembering when
  wiring `src/model`.
- Vision tower weights are carried bf16-only for now (no quantized vision GEMM path yet); vision
  tower forward pass itself is a post-MTP-self-speculation milestone.
- fp8 KV descales are calibrated per-layer on demand via `kv_calibrate.py`
  (`tools/reference/kv_calibrate_out/kv_descale.json`, gitignored) but not yet wired into the
  converter -- `docs/container-format.md`'s descale table is still the placeholder `1.0` until
  `src/convert` consumes calibration output for all 16 full-attention layers.
- `tests/reference/test_manifest.py` is CPU-only and has no `pytest` dependency (the reference venv
  doesn't have `pytest` installed and is read-only) -- it's a plain script with bare asserts, run
  directly and also registered as the `reference_manifest` ctest test.
- No prefill kernel yet beyond the interim 64-row skinny-GEMM chunking path described in the top-level
  task decisions; a proper 4-bit WMMA prefill kernel is future work.

## Next milestone

**Milestone 1** (container loader, GDN + attention layers, model forward, `r4dx-cli` text
generation) is done -- see the "Milestone 1: done" section above, the "Update" sections, and
`docs/perf.md`. Next up: `src/server` (OpenAI-compatible streaming chat API) and MTP
self-speculation, followed by the vision tower, DFlash2, and prefix-caching milestones, per the
top-level project decisions.
