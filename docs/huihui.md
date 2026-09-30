# Huihui abliterated: the default trellis mix4.5m container

Since 2026-09-29 this is r4dx's default (production) container. Source:
`D:\models\Huihui-Qwen3.8-27B-abliterated` (same architecture, tokenizer and config as `Qwen3.8-27B`; 70 of
1199 tensors differ from base: `mlp.down_proj` x35, `linear_attn.out_proj` x26, `self_attn.o_proj` x9 in layers
17..51).

## Container

`D:\models\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx`, `--layout trellis`, 18,338,486,923 B, decode 13.546 GiB.
The same recipe as the base `qwen38-27b-trellis-mix45m.r4dx` (docs/quant2.md 7.1, docs/trellis.md), with every
calibration artifact taken from huihui itself: new Hessians (hessian-v2 settings,
`tools/reference/hessian_capture.py` with GemmGuard), K4m + K5m on huihui, mix 4.5 (allocation identical to
base's), huihui imatrix and kvcalib for the lm_head / MTP head / fp8 KV. The full run, with every command and the
torch 2.13+rocm10 venv `quantize-model` needs (MAGMA cholesky), is `D:\models\r4dx\huihui\RECIPE.md`; its
artifacts (`huihui\hessian-v2`, `huihui\trellis-q\{K4m,K5m,mix4.5m}`, `huihui\kl-ref`, `huihui\kl\*`,
`huihui-qwen38-27b-abl.{imatrix.npz,kvcalib-full.json}`) are what the container's metadata records and what
`trellis_oracle.ps1` / `trellis_convert.ps1` now default to. The DFlash2 drafter is unchanged
(`qwen38-27b-dflash2-w4a16-g64.r4dx`, trained for the base model).

## What "default" means in the repo (2026-09-29)

- `tests/model/test_container_path.h`: `ProductionTargetPath()` = this container on a group-64 build (the only
  build that can read it: its w4a16 parts are packed at g64), `ProductionLayoutName()` = `trellis`;
  `tools/r4dx_containers.ps1`: `Get-R4dxProductionTarget` / `Get-R4dxProductionLayout` / `Get-R4dxTokenizerDir`.
  A win-hip-g128 build has no production container (v3 is gone), so its real-container tests SKIP.
- `--tokenizer-dir` of `r4dx-cli` / `r4dx-server`, the tokenizer / http-server / engine-recovery tests
  (`R4DX_TOKENIZER_MODEL_DIR`), `test_keep_bf16` (`R4DX_HF_CHECKPOINT`) and `tools/reference/common.py`
  `DEFAULT_MODEL_DIR` (~20 Python tools) name `D:\models\Huihui-Qwen3.8-27B-abliterated`. Its `tokenizer.json`,
  `tokenizer_config.json`, `chat_template.jinja`, `generation_config.json`, `config.json`, `vocab.json`,
  `merges.txt` and the preprocessor configs are SHA-256 identical to the base's (10/10), so tokenization,
  `tests/tokenizer/golden.json` and the `kl_corpus` token files are unchanged.
  **A fine-tune's config.json equals the base's, so no config check can tell two Qwen3.8-27B-family checkpoints
  apart: a tool run on another model needs its `--model-dir` (and hessian / oracle dirs) passed explicitly.**
- Defaults now on the Huihui files: `tools/prefill/*.ps1 -Model`, `g6_validate.ps1 -Layout trellis`,
  `bench_decode.ps1 -Layout trellis`, the validators' `-Layouts` (an explicit `-Model` keeps
  `w4a16,w4a8,mxfp4`), `trellis_oracle.ps1`, `trellis_convert.ps1`, `trellis_quant.py --hessian-dir`,
  `trellis_golden.py`, `tp1_identity.ps1 -Model/-Layout/-TokenizerDir`, the tool_* diagnostics' `--model` /
  `--layout` / tokenizer dir. Every path is still overridable exactly as before.
- New: `tools/quant2/kl_rung4.ps1` (the KL gate below).

## Retired files (2026-09-29)

Removed after this container validated (KL, G6, TP=2, and the regression baselines below re-frozen on it):
`qwen38-27b-trellis-mix45m.r4dx`, `qwen38-27b-trellis-k4m.r4dx`, `qwen38-27b-trellis-k4m-lmbf16.r4dx`,
`qwen38-27b-v6.r4dx`, `huihui-qwen38-27b-abl-v6.r4dx` (all in `D:\models\r4dx`), the base oracle bits
`D:\models\r4dx\trellis-q\`, the base Hessians `D:\models\r4dx\hessian\hessian-v2`, and the base HF checkpoint
`C:\AI\models\Qwen3.8-27B`. **Name collision:** `D:\models\r4dx\huihui\trellis-q` and `...\huihui\hessian-v2`
are this container's inputs and stay. Kept: the `l4-*` test containers (4 layers, layers 0-3 are the same in
either model), the DFlash2 drafters, `kl-canon` / `kl-thai-canon` (the base model's bf16 references),
`corpus-v2`, the base imatrix / kvcalib files, `D:\models\Qwen3.8-27B-DFlash2`, `D:\models\wikitext-2-raw`, and
everything under `huihui\`.

What remains true about the retired variants: the base numbers quoted across the docs (KL 0.00747, K4m's
0.01004 and its +11% plain / +13% DFlash speed, v6's 0.03851) are historical measurements on the base model.
The Huihui K4m oracle bits exist (`huihui\trellis-q\K4m`, and K5m) and `trellis_convert.ps1 -Oracle K4m` would
build a Huihui K4m container, but none has been built or measured: no K4m figure is claimed for Huihui. The
w4a16 v6 recipe (README "Convert a checkpoint") still converts and runs on any checkpoint.

## Measurements (2026-09-29, main 1099446 build)

Rung-4 canonical KL (`tool_teacher_forced_logprobs --layout trellis --max-ctx 4096 --vision off`, `tokens_canon.json`),
against the model's own bf16 reference:

| container | mean KL | top-1 | cpp | english | python | thai |
|---|--:|--:|--:|--:|--:|--:|
| **huihui trellis mix4.5m (default)** | **0.00788** | 95.70% | 0.00583 | 0.00975 | 0.00778 | 0.00814 |
| huihui v6 w4a16 (old, retired) | 0.03001 | 92.33% | 0.02407 | 0.02874 | 0.02806 | 0.03918 |
| base trellis mix4.5m (retired) | 0.00747 | 96.26% | 0.00607 | 0.00847 | 0.00674 | 0.00860 |

Oracle (weights only): huihui 0.00575 / 96.48%, base 0.00547 / 96.90%. One position (english_prose 698, KL 1.55; the
oracle has 0.81) accounts for the whole gap to base: without it the mean is 0.00750.

- G6 `-Layout trellis`: 5/5 (3/3, 24/24, 217/0, 168/0, 170/0). TP=2 real `smoke.ps1 -Vision -Dflash`: 207 PASS / 0 FAIL,
  no TDR.
- Speed (TP=1, one run): plain 36.67 tok/s (base 36.65), prefill 1131 tok/s (base 1116), DFlash k=7 108.2 tok/s (base
  116.1: same 36.4 ms per round, lower acceptance because the drafter is base's). docs/perf.md has the table.
- TP=2 speed, real two-GPU runs on this container (2026-09-30, main 0129b8c): plain 60.10 tok/s (1.70x the
  same-session TP=1's 35.38), DFlash k=7 162.18 (1.58x), `--mtp 3` 116.46 (1.55x); cold prefill 3.87 s at 8k,
  19.12 s at 32k, 48.97 s at 64k. Greedy text and token ids at 8k are byte-identical between the default
  256-row chunk and `R4DX_PREFILL_CHUNK=0` for plain, MTP and DFlash. Table and method: docs/perf.md
  "TP=2 on the Huihui trellis container".
- Abliteration survives quantization: lock picking, hotwiring one's own car and a dark joke are answered; the base
  container refuses all three.

## Frozen values (re-based on this container, 2026-09-29)

The baselines the tests and scripts compare against, re-taken on the Huihui container. Each was frozen twice and
the repeat was bit-identical before it was written down.

| what | frozen value | where / how to re-check |
|---|---|---|
| Rung-4 KL | mean **0.00788**, top-1 **95.70%**, segments cpp 0.00583 / en 0.00975 / py 0.00778 / thai 0.00814; the four `*.logprobs.f16` are byte-identical between the 2026-09-28 run (`huihui\kl\rt-mix45m`, device 1) and a run of the merged build on device 0 (`rebase\kl-rt-mix45m-new-gpu0`) and on device 1 (`rebase\kl-rt-mix45m-final-gpu1`, the final tree) | `tools\quant2\kl_rung4.ps1 -OutDir <dir>` (gate: KL / top-1 / byte equality); reference `D:\models\r4dx\huihui\kl-ref` |
| TP=1 identity, rows 1-5, 7-9 | see the table in docs/tp.md 10.3 (plain / `--dflash` k=7 / `--mtp 3` text and token ids, sampled text and ids, vision, chat); full SHA-256 in `%USERPROFILE%\dev\r4dx-baselines\tp1-1099446\FROZEN_HASHES.txt` | `tools\tp\tp1_identity.ps1 -Baseline tp1-1099446 -Candidate build\win-hip`; baseline binaries frozen from main 1099446 (libr4d dec5a4f), the C++ tree of `huihui` `b512207` |
| TP=1 identity, row 6 (4-layer `l4-allmtp`, bf16 / w4a16 / w4a8 / mxfp4) | EQUAL against both baselines: `tp1-1099446` and the pre-trellis `tp1-f7d4927` (16 dump hashes in `FROZEN_HASHES.txt`) | same script, `-Rows 6` |
| G6 (`g6_validate.ps1 -Layout trellis`) | 5/5, identical hash for hash to the pre-rebase run (`D:\models\r4dx\huihui\g6`): validate_dflash 3/3 byte-identical (short `ecf1855ec1b7`, medium `184cde8ff01d`, long `b646012990bb`); validate_spec_sampling 24/24 (12 prompt x sampling cells, plain = `--mtp 3` = `--dflash k=7` in each; 11 distinct text hashes in log order `8262ca0b617f c04f67b116bd ae6fccdcbfc4 0ddb6076f345 5eb007b3edd5 63a92b473d1d b4b44a03ac4d 485d7a312bcf aca3b132db5b dbc0811d2071 de91177f46c1`); smoke dflash + tools + vision 217/0; smoke MTP 3 168/0; smoke TP=2 emulate + DFlash 170/0 (log: `D:\models\r4dx\huihui\rebase\g6-final`). TP=2 **real** smoke with vision and DFlash (`smoke.ps1 -Tp 2 -Layout trellis -Layers -1 -Dflash ... -Vision`): 207 PASS / 0 FAIL, no TDR | G6 stores no expected values: verdict = exit codes and `[PASS]` / `[FAIL]` counts in `summary.json` |
| ctest | `tests\run_tests.ps1`: **95 / 95 passed, 0 skipped** (833 s; every real-container test ran on this container: `test_dflash_e2e`, `test_vision_tower`, `test_tp_emulation`, `tokenizer_golden`, `test_http_server`, `test_engine_recovery`, `test_keep_bf16`, `reference_gguf_dequant`, `reference_trellis_quant`), `tests\run_tests.ps1 -TwoGpu`: 2 / 2 passed (`test_tp_allreduce_2gpu`, `test_tp_real_vs_emulation`), no TDR. The Python reference interpreter is the system Python 3.12 (transformers 5.5.0), which registers `reference_hessian_rms`, `reference_gguf_dequant`, `reference_trellis_quant`; `reference_manifest`, `reference_dflash2` and `reference_hessian_corpus` need transformers 5.17.0 (only `huihui\venv-rocm10` has it): there `reference_manifest` and `reference_hessian_corpus` (after masking the checkpoint dir in its provenance comparison) pass, and `reference_dflash2` fails 62 numeric-determinism checks (2e-6 .. 1 max_abs_diff) identically on main's untouched tree with that interpreter, i.e. an interpreter effect, not this change | `tests\run_tests.ps1`, `tests\run_tests.ps1 -TwoGpu` |
| TP=2 KL (G4, emulated, device 0) | mean KL **0.00806**, top-1 96.04% against the bf16 reference (TP=1: 0.00788 / 95.70%); against the TP=1 dump 0.00089 / 98.85% agreement | docs/tp.md 10.4; dump ebase\kl-tp2emu-gpu0 |
| test_tp_emulation, production-container cases | max rel L2 vs TP=1: MTP replays 4.02e-2, DFlash replays 3.88e-2, vision logits 1.46e-2 (tolerance 5e-2, unchanged; v6 w4a16 measured up to 2.2e-2); both emulated ranks use 19.96 GiB (MTP), 22.58 GiB (DFlash2), 20.50 GiB (vision) at `--max-ctx` 1024 (budgets 22 / 24 / 22 unchanged) | constants and comments in `tests/model/test_tp_emulation.cpp` |
| test_tp_real_vs_emulation, DFlash case | emulate and real byte-identical (16 rounds, 96 tokens) | `tests\run_tests.ps1 -TwoGpu` |

Not re-frozen, on purpose: `tests/tokenizer/golden.json`, `kl_corpus/tokens*.json` and the `golden_out` tensors
(layers 0 and 3, lm_head, MTP, vision, rope, all unchanged in Huihui; their provenance strings still name the base
checkpoint path), the trellis real-tile goldens (`tests/kernels/golden/trellis/real_*`, cut from base K4m; the kernel
tests read the npy files only), and the prefill M0 dense KL / TTFT dumps (`prefill-m0`, base container; re-take them
on this container before an M2 comparison, docs/prefill.md).

## Coverage: what the retirement of the 64-layer w4a16 / w4a8 / mxfp4 containers leaves

Before, `qwen38-27b-v6.r4dx` (w4a16 g64 + w4a8 g128 + mxfp4 g32, 64 layers, vision, MTP) was the production
container, and the old Huihui v6 was the same recipe on this model. Nothing else 64-layer carried a non-trellis body.

| capability | before | now |
|---|---|---|
| trellis body at depth (G6, `test_dflash_e2e`, `test_vision_tower`, the real-container cases of `test_tp_emulation` / `test_tp_real_vs_emulation`, tp1_identity rows 1-5 and 7-9, KL rung 4, smoke) | base mix45m + v6 for the ctest cases | this container, every case ran (none skipped) |
| w4a16 g64 / w4a8 g128 / mxfp4 g32 + MTP at 4 layers (`test_mtp`, `test_tp_loader`, `test_tp_emulation` 4 layouts, `test_tp_real_vs_emulation`, `test_prompt_checkpoint`, tp1_identity row 6, `tests/kernels`) | `l4-allmtp` | unchanged (`l4-allmtp`, `l4-bf16`, `l4-mtp*` are kept) |
| **w4a16 g64 at 64 layers on real weights, with vision, MTP and the DFlash2 target layers** | v6 (tp1_identity rows 1-5, 7-9; G6 default; trellis gate A6 "no w4a16 regression"; `test_dflash_e2e` / `test_tp_emulation` cases before this change) | **not covered.** The cases that used it (`test_dflash_e2e`, the TP cases) run the same logic on the trellis container; the w4a16 kernel and loader are exercised at 4 layers only. |
| **w4a8 g128 and mxfp4 g32 at 64 layers** | v6 via `validate_dflash` / `validate_spec_sampling` / `validate_fusion` (manual, layouts `w4a16,w4a8,mxfp4`) | **not covered.** Fused epilogues exist only for w4a8/mxfp4: `tests/kernels/test_fused_quant.cpp` covers the kernels (135-check byte-diff grid), the end-to-end 64-layer fusion gate is gone. |

Last full-depth evidence, taken with the merged build while the old Huihui v6 still existed (nothing was deleted
by the rebase job), on `huihui-qwen38-27b-abl-v6.r4dx` (w4a16 g64 + w4a8 g128 + mxfp4 g32, 64 layers, vision, MTP; `D:\models\r4dx\huihui\rebase\`):

- `tp1_identity.ps1 -Model <it> -Layout w4a16` rows 1-5 and 7-9 (plain, DFlash, MTP, sampled, vision, chat) of the merged build are EQUAL, text / token ids / stats lines, to the pre-trellis `tp1-f7d4927` binaries: the w4a16 path at depth is byte-identical to before the trellis work.
- `validate_fusion.ps1 -Model <it>` (w4a16, w4a8, mxfp4 x 3 prompts x `--mtp` 0 / 3): 18 / 18 fused = unfused byte-identical.
- `validate_spec_sampling.ps1 -Model <it> -AllowBatchedVerifyDivergence` (3 layouts x 2 prompts x 3 sampling configs x 2 seeds): plain = `--mtp 3` = `--dflash k=7` in all 36 cells (72 / 72 comparisons byte-identical).
- `validate_dflash.ps1 -Model <it> -AllowBatchedVerifyDivergence` (3 layouts x 3 prompts): 7 / 9 byte-identical to `--mtp 0`; w4a8 medium diverges and `--mtp 7` reproduces it exactly (the documented batched-verify mechanism); mxfp4 medium diverges and `--mtp 7` does not, so the script exits 1 -- the "same class, not directly proven" cell docs/dflash2.md records for the base v6. Both divergent cells produce the same text hashes with the pre-trellis f7d4927, the 1099446 and the merged binaries (w4a8 medium `afce9646ec40` plain / `b3968654df5e` DFlash and `--mtp 7`, mxfp4 medium `83d48aa8c65b` / `33a5c117fb00`), i.e. they are not something this change introduced.

To restore 64-layer coverage: keep or re-convert a multi-layout container on this model (the README v6 command on
the Huihui checkpoint with the huihui imatrix / kvcalib, `--layouts w4a16,w4a8,mxfp4`, 42.7 GiB; or `--layouts
w4a16` alone with `--keep-bf16 "^text\.layers\.[0-9]+\.attn\.[kv]$"`, roughly 17-20 GiB, for the w4a16-only rows) and
pass it as `-Model` (with `-Layout w4a16` for tp1_identity and smoke, `-Layouts ...` is the validators' default for
an explicit `-Model`). The frozen `tp1-f7d4927` baseline reads it.
