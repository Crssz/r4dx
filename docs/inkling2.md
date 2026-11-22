# Inkling-2-Mini on r4dx

`thinkingmachines/Inkling-2-Mini` and its abliterated variant `OpenUncensored/Inkling-2-Mini-Uncensored` (Apache-2.0;
same architecture, tensors and config) run as the `inkling2_loop` arch: a looped 128-expert MoE with 24 stored layers,
layers 6-17 run three times per token (48 layer visits), no RoPE (K/V short convolutions plus a learned relative bias),
per-token attention sinks, block attention residuals and cross-pass message gates. Design, semantics and every decision:
[inkling2-plan.md](inkling2-plan.md). Milestone 1 is **text only** (no vision encoder, no DSpark/MTP drafter).

## Convert

The checkpoint stores the routed experts pre-quantized (NVFP4 in layers 0-19, MXFP8 in layers 20-23; everything else
bf16). The converter writes the `mxfp8` layout (OCP e4m3fn codes + one E8M0 scale per 32 along K, run as W8A16): the
MXFP8 experts are copied bit-exact, the NVFP4 experts are dequantized in f32 and re-encoded (relative error at most
2^-4 per weight), and the bf16 linears and the lm_head are encoded the same way. The embedding stays bf16.

```powershell
build\win-hip\src\convert\r4dx-convert.exe --input E:\models\OpenUncensored\Inkling-2-Mini-Uncensored --output E:\models\r4dx\inkling2\inkling2-mini-uncensored-mxfp8.r4dx --layouts mxfp8 --no-bf16
```

78 s, 27,371,116,202 bytes (13078 tensors; 822 checkpoint tensors consumed, the 422 `model.visual.*` tensors skipped and
logged). `--keep-bf16 <regex>` keeps chosen linears bf16; `--layouts bf16` writes an all-bf16 container (52 GB, TP=2 only).
`--plan-only` audits the name mapping without reading tensor data.

## Run

```powershell
$env:HIP_VISIBLE_DEVICES='1'; build\win-hip\src\cli\r4dx-cli.exe --model E:\models\r4dx\inkling2\inkling2-mini-uncensored-mxfp8.r4dx --layout mxfp8 --tokenizer-dir E:\models\OpenUncensored\Inkling-2-Mini-Uncensored --dialect inkling2 --max-ctx 8192 --prompt "Explain why the sky is blue in two sentences."
```

```powershell
build\win-hip\src\server\r4dx-server.exe --model E:\models\r4dx\inkling2\inkling2-mini-uncensored-mxfp8.r4dx --layout mxfp8 --tp 2 --tokenizer-dir E:\models\OpenUncensored\Inkling-2-Mini-Uncensored --dialect inkling2 --port 8091
```

- **Memory.** TP=1 holds 27.3 GB of weights plus workspace: the CLI measured **31.45 GiB at `--max-ctx 8192`**, so on one
  32 GB card use a small context (the 131072 default does not fit at TP=1). `--tp 2` uses 15.66 GiB of buffers per card at
  131072 (22.9 GiB used per card including the runtime).
- **Context.** Default `--max-ctx` 131072, explicit values up to 1048576 (bf16 KV is 32 KiB/token: 8 full-attention
  visits; the 40 sliding visits keep 1536-position rings).
- **Dialect `inkling2`** (auto-detected): the model answers in typed parts after its own `<|message_model|>` header (the
  template has no generation prompt): `<|content_thinking|>` -> `reasoning_content`, `<|content_text|>` -> `content`,
  `<|content_invoke_tool_json|>{"name":..,"args":{..}}` -> `tool_calls`; stop on `<|content_model_end_sampling|>`.
  `reasoning_effort` accepts the template's names (none/minimal/low/medium/high/xhigh/max) or a number in [0, 0.99]
  (default 0.9); `enable_thinking: false` means effort 0. The server's `--think` flag is ignored for this dialect (thinking
  is on unless a request turns it off). Sampling default temperature 1.0 (generation_config).
- **Prefix cache.** The prompt checkpoint holds back the final `<|end_message|>` (one token), so both a following turn and
  an identical repeated prompt reuse the cache.
- **Do not overlap a model load with another bulk disk job** (for example the HF reference reading the checkpoint): a
  DirectStorage load under that load hard-crashed the machine on 2026-11-22; use `R4DX_DIRECT_STORAGE=0` if they must overlap.

## Validation (2026-11-22)

Teacher-forced KL against an **fp32 CPU truth** (lane D's layer-streamed walker over the HF `Inkling2Loop*` modules,
bit-equal to HF on a tiny checkpoint), with stock HF bf16 as the noise term; corpus `tools/reference/kl_corpus/chat_inkling2.json`
(12 conversations, 4756 scored assistant tokens). Rule (`kl_report.py --gate inkling2-fp32`, epsilon 0.005): per group
`KL <= 1.5 x KL(truth||HF-bf16) + 0.005` and `top-1 >= HF-bf16 top-1 - 1 pt`.

| group | HF-bf16 noise KL / top-1 | limit KL / top-1 | **r4dx TP=1** KL / top-1 | r4dx TP=2 KL / top-1 |
|---|---|---|---|---|
| code | 0.00205 / 99.02 % | 0.00807 / 98.02 % | 0.00392 / 98.10 % PASS | 0.00412 / 98.24 % PASS |
| english | 0.00291 / 97.35 % | 0.00937 / 96.35 % | 0.00633 / 97.07 % PASS | 0.00720 / 96.93 % PASS |
| thai | 0.00283 / 98.35 % | 0.00924 / 97.35 % | 0.00634 / 97.42 % PASS | 0.00621 / **96.77 % FAIL** |
| tool | 0.00213 / 99.21 % | 0.00820 / 98.21 % | 0.00740 / 98.22 % PASS | **0.00834 FAIL** / 98.22 % |
| **chat-ALL** | 0.00255 / 98.34 % | 0.00882 / 97.34 % | **0.00573 / 97.60 % PASS** | 0.00611 / 97.41 % PASS |

**TP=1 passes every group; TP=2 misses thai top-1 by 0.58 pt and tool KL by 0.00014.** The likely cause is the TP
all-reduce, which exists only in bf16 (`AllReduceSumBf16Rows`): each rank's partial sum is rounded to bf16 before the sum,
an extra rounding TP=1 does not have, and this model's routing is fragile late in the loop (even HF bf16 keeps the same
8 experts as the truth for only about 60 % of tokens at visits 30-39). An fp32 all-reduce is the follow-up.

Reported, not gated (raw text, TP=1): cpp 0.01056, english 0.00800, python 0.00580, thai 0.01035 (raw-ALL 0.00868 vs a
1.5 x noise + eps line of 0.01136); the 6144-token long segment 0.00972 (line 0.01313).

Other checks: GPU unit tests (`ctest -L inkling2_gpu`: mxfp8 GEMM bit-exactness incl. every e4m3 code and the TP shapes,
every Inkling kernel vs the CPU spec, attention with bias + sink + sliding ring), the tiny end-to-end test against HF at TP=1
and TP=2 (both GPUs, no TDR), CLI greedy at TP=1 and TP=2, and `tools\server\smoke.ps1 -Dialect inkling2 -Tp 2`: **65 PASS,
0 FAIL**. CPU tests: converter (mxfp8 format, expert codec vs HF dequant, layout + audit), tokenizer (301 encode goldens,
320k fuzz cases 0 mismatches), chat template (60 render + 15 raise cases byte-identical to HF), dialect/parser/engine,
config/visit table, and a CPU copy of the model graph equal to HF fp32 on the tiny checkpoint (mean KL 6.4e-12).

Artifacts: `E:\models\r4dx\inkling2\kl\{fp32\truth, fp32\bf16, r4dx-mxfp8, r4dx-mxfp8-tp2}` (gate.json in each test dir).

## Speed (unoptimized milestone 1)

| | TP=1 (device 1) | TP=2 |
|---|---|---|
| teacher-forced, chat corpus | 43.1 ms/row | 59.3 ms/row |
| decode (CLI, short prompt) | 23.6 tok/s | 20.0 tok/s |
| decode (server smoke) | | about 27 tok/s |
| prefill (25-token prompt) | 218 tok/s | 264 tok/s |

Decode is bound by the staged MoE (a host sync per layer visit, 48 per token) and the reference attention kernel; the
follow-ups are a grouped device-indexed MoE decode (LFM2/Ornith style) and a fused h128/GQA3 attention with bias + sink.

## Not done

- Vision (23-layer ViT) and the DSpark/MTP drafter (`mtp.safetensors`).
- fp32 TP all-reduce (the TP=2 gate miss), performance work above, fp8 KV.
- HF-vs-r4dx greedy token comparison: the HF reference is recorded (`E:\models\r4dx\inkling2\greedy\hf_greedy.json`, 4
  prompts x 64 tokens); the r4dx side and `greedy_smoke.py --compare` have not been run.
- **Positions past 8192 on the real model are untested.** That is where the full-attention layers' log scaling
  (`log_scaling_n_floor` 8192) starts. The code path is covered by the tiny end-to-end test against HF (floor 16 there), and
  the 6144-token set covers the sliding-ring wrap, but no real-weight run went beyond 6144 tokens. A 16k-token TP=2 NLL dump
  plus a 20k-token needle test was started on 2026-11-22 and stopped by the user before it finished (inputs were made by a
  scratch script from the repo's docs text; rerun it before relying on long contexts).
- Variable loop passes (`is_varloop`, the TML-d24-varloop-research checkpoint): skipped by decision; r4dx refuses such a
  config.
