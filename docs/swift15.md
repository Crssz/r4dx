# Swift-1.5 (Qwen3.8-27B fine-tune, FP8 source): trellis mix4.5m container

Side-by-side with the Huihui default (docs/huihui.md); nothing in the repo's defaults points here.

Source: `d0xin/Swift-1.5-Qwen3.8-27B-Uncensored-FP8` (Swift Open License v1.0), 18 shards. Same architecture,
tensor names, `config.json` (minus `quantization_config`), tokenizer and chat template as Huihui / base, so
`tokens_canon.json` is valid unchanged. 407 text-body and MTP linears are float8_e4m3fn with a bf16
`<name>.weight_scale` per 128x128 block (multiply); vision, norms, embeddings, lm_head stay bf16.

## Recipe (2026-10-05/06, device 1 only)

1. `tools/reference/fp8_dequant.py dequant --src <fp8 dir> --out E:\models\Swift-1.5-Qwen3.8-27B-bf16` (CPU, ~13 min,
   1199 tensors, 407 dequantized). This bf16 checkpoint is the truth for every KL below.
2. `E:\models\r4dx\swift15\run_chain.ps1`: bf16 KL ref, imatrix + kvcalib (canonical tokenizer), Hessians
   (hessian-v2 settings, 191 min, 389 files, gates pass), K4m (73 min) and K5m on `huihui\venv-rocm10`, mix 4.5
   (4.5045 bpw), oracle KL. The Huihui RECIPE.md steps 1-5 with Swift's paths.
3. `tools\quant2\trellis_convert.ps1` with Swift's paths. Verify 400/400. (This run used E: paths
   throughout: before the models-root merge the manifest pin compared unresolved paths and refused the old D:\models
   junction spelling; it is junction-aware now.)

Container: `E:\models\r4dx\swift15-qwen38-27b-trellis-mix45m.r4dx`, 18,338,486,745 B, 16.27 GiB weights in VRAM.
Calibration: `E:\models\r4dx\swift15\{hessian-v2,trellis-q,kl-ref,kl}`, `swift15-qwen38-27b.{imatrix.npz,kvcalib-full.json}`.

## Measurements (main 9e39d0e build, device 1)

| KL vs own bf16 (`tokens_canon.json`) | mean | top-1 | cpp | english | python | thai |
|---|--:|--:|--:|--:|--:|--:|
| Swift oracle (weights only) | 0.00564 | 97.24% | 0.00453 | 0.00613 | 0.00584 | 0.00607 |
| **Swift container** | **0.00764** | **96.41%** | 0.00610 | 0.00806 | 0.00844 | 0.00796 |
| Huihui container (docs/huihui.md) | 0.00788 | 95.70% | 0.00583 | 0.00975 | 0.00778 | 0.00814 |

KL(Huihui bf16 || Swift bf16) = 0.0125, top-1 95.85%: the fine-tune moves the model more than quantization does.

G6 (`g6_validate.ps1 -Layout trellis`): 5/5 -- validate_dflash 3/3 and validate_spec_sampling 24/24 byte-identical,
smoke dflash+tools+vision 216/0, smoke MTP 3 168/0, smoke TP=2 emulate + DFlash 170/0.

Decode bench (`bench_decode.ps1`, TP=1, tok/s, median of 3 runs):

| container | plain | DFlash k=7 | MTP 3 | warm prefill |
|---|--:|--:|--:|--:|
| Swift | 36.63 | 107.48 | 75.58 | 1491.8 |
| Huihui | 36.64 | 108.31 | 78.62 | 1488.3 |

The DFlash2 drafter is the base model's and still holds up (-0.8% vs Huihui); MTP 3 is 4% lower (Swift's own
MTP head, lower acceptance on two prompts).

Serve: `r4dx-server --model E:\models\r4dx\swift15-qwen38-27b-trellis-mix45m.r4dx --layout trellis`
(+ `--dflash E:\models\r4dx\qwen38-27b-dflash2-w4a16-g64.r4dx --dflash-k 7`).
