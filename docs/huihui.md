# Huihui abliterated: trellis mix4.5m container

Branch `huihui` (not pushed). Source: `D:\models\Huihui-Qwen3.8-27B-abliterated` (same architecture, tokenizer and
config as `Qwen3.8-27B`; 70 of 1199 tensors differ from base: `mlp.down_proj` x35, `linear_attn.out_proj` x26,
`self_attn.o_proj` x9 in layers 17..51).

## Container

`D:\models\r4dx\huihui-qwen38-27b-abl-trellis-mix45m.r4dx`, `--layout trellis`, 18,338,486,923 B, decode 13.546 GiB.
The same recipe as `qwen38-27b-trellis-mix45m.r4dx` (docs/quant2.md 7.1, docs/trellis.md), with every calibration
artifact taken from huihui itself: new Hessians (hessian-v2 settings, `tools/reference/hessian_capture.py` with
GemmGuard), K4m + K5m on huihui, mix 4.5 (allocation identical to base's), huihui imatrix and kvcalib for the
lm_head / MTP head / fp8 KV. Run the Python tools with `--model-dir` (config.json is byte-identical to base's, so no
config check can tell the two models apart); `quantize-model` needs torch 2.13+rocm10 (MAGMA cholesky), not the
system torch 2.9.1.

## Measurements (2026-09-29, main 1099446 build)

Rung-4 canonical KL (`tool_teacher_forced_logprobs --layout trellis --max-ctx 4096 --vision off`, `tokens_canon.json`),
against the model's own bf16 reference:

| container | mean KL | top-1 | cpp | english | python | thai |
|---|--:|--:|--:|--:|--:|--:|
| **huihui trellis mix4.5m (new)** | **0.00788** | 95.70% | 0.00583 | 0.00975 | 0.00778 | 0.00814 |
| huihui v6 w4a16 (old) | 0.03001 | 92.33% | 0.02407 | 0.02874 | 0.02806 | 0.03918 |
| base trellis mix4.5m | 0.00747 | 96.26% | 0.00607 | 0.00847 | 0.00674 | 0.00860 |

Oracle (weights only): huihui 0.00575 / 96.48%, base 0.00547 / 96.90%. One position (english_prose 698, KL 1.55; the
oracle has 0.81) accounts for the whole gap to base: without it the mean is 0.00750.

- G6 `-Layout trellis`: 5/5 (3/3, 24/24, 217/0, 168/0, 170/0). TP=2 real `smoke.ps1 -Vision -Dflash`: 207 PASS / 0 FAIL,
  no TDR.
- Speed (TP=1, one run): plain 36.67 tok/s (base 36.65), prefill 1131 tok/s (base 1116), DFlash k=7 108.2 tok/s (base
  116.1: same 36.4 ms per round, lower acceptance because the drafter is base's).
- Abliteration survives quantization: lock picking, hotwiring one's own car and a dark joke are answered; the base
  container refuses all three.
- The old `huihui-qwen38-27b-abl-v6.r4dx` (mxfp4 / w4a16 / w4a8, 45.9 GB) is superseded.
