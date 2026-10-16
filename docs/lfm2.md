# LFM2.5-24B-A2B in r4dx

Status 2026-10-16: **bf16 TP=2 works end to end** (KL gate PASS, server smoke by hand: chat with the reasoning split, Thai,
a tool call). **Trellis on one card works** as an 18.89 GB hybrid (K5 experts, bf16 non-expert and layers 2-5 experts;
KL gate PASS, about 61 tok/s decode; see "Trellis (one GPU)"). This page is the
user-facing summary; the design and task list are [lfm2-plan.md](lfm2-plan.md), the verified model facts (each with a
source line) are [lfm2-semantics.md](lfm2-semantics.md).

## Architecture

`LiquidAI/LFM2.5-24B-A2B` (`lfm2_moe`): 24.0 B parameters, 2.45 B active per token, 40 layers, hidden 2048, vocab 128000
(real ids end at 125016, the logit columns above are masked), 131072 positions (rope theta 1e7, no scaling).

- 30 gated short-convolution layers (`in_proj` 2048 -> 3x2048 as B, C, x; `h = B*x`; depthwise causal conv, kernel 3, no
  activation; `y = C*conv(h)`; `out_proj`), carrying a two-row state instead of a KV cache.
- 10 full-attention GQA layers (layers 2, 6, ..., 38): 32 q heads, 8 kv heads, head_dim 64, per-head QK RMSNorm before rope.
- FFN after every mixer: dense SwiGLU (I = 11776) in layers 0-1; in layers 2-39 a 64-expert top-4 MoE (I = 1536, no shared
  expert, sigmoid router with a selection-only `expert_bias`, weights normalized over the selected four).
- Tied embedding / lm_head, no logit softcap. KV cache is bf16 (10 attention layers only).

It is a third `TextModel` family (`Lfm2Model`, `Lfm2TpModel`) next to Qwen and Gemma; the other two paths are untouched.

## Container

```powershell
r4dx-convert --input E:\models\LiquidAI\LFM2.5-24B-A2B --output E:\models\r4dx\lfm2\lfm2-24b-a2b-bf16.r4dx `
    --layouts bf16 --lm-head bf16
```

(the container header carries `model_arch: lfm2_moe` and the verbatim `config.json`). Layout: per-expert linears
`text.layers.{i}.moe.experts.{e}.{gate_up,down}`, the router and the fp32 `expert_bias` stay un-quantized, the conv weight is
squeezed to `[2048, 3]`; see lfm2-plan.md section 4 for the full table. The bf16 container is **48.5 GB**; at TP=2 every
rank holds about **25.8 GiB** of VRAM (device 0, the desktop card, has about 2 GB free after that).

## Serving

```powershell
r4dx-server --model E:\models\r4dx\lfm2\lfm2-24b-a2b-bf16.r4dx --layout bf16 --tp 2 --port 8091
```

- `--dialect auto` picks `lfm2` from the container's `model_arch` (and the tokenizer's `<|startoftext|>` bos plus the
  `<|tool_call_start|>` token); `--dialect lfm2` forces it. The tokenizer directory defaults to
  `E:\models\LiquidAI\LFM2.5-24B-A2B` (`--tokenizer-dir` overrides).
- One bf16 container does not fit one card: `--tp 2` is required for it. `--pp 2` (pipeline prefill, hybrid) is not part of this family's validated modes.
- `--max-ctx` defaults to the checkpoint's 131072. `/v1/models` reports `meta.n_ctx_train` 131072 for LFM2 (read from the
  container's config; it used to show the Qwen constant 262144).
- Sampling defaults come from LFM's `generation_config.json` when the request leaves a field unset: temperature 0.1,
  top_k 50, **repetition_penalty 1.05** (applied on the host over a full-logits step, HF semantics; `repetition_penalty` is
  also a request field). Qwen and Gemma defaults are unchanged. A `--default-*` flag still wins over the dialect.
  Repetition penalty != 1 is rejected together with speculative decode (there is no drafter for this model anyway).
- Smoke test (needs the GPUs, see below): `tools\server\smoke.ps1 -Model E:\models\r4dx\lfm2\lfm2-24b-a2b-bf16.r4dx -Tp 2 -Dialect lfm2 -Port 8091`
  (`-Dialect lfm2` defaults `-Layout bf16` and `-Layers -1`; add `-TpFault` for the all-reduce fault-recovery steps). It is
  run on 2026-10-16: 75 checks pass on the trellis hybrid at `-Layout trellis -Tp 1`; on bf16 TP=2 all but three
  cosmetic checks passed before they were fixed (streamed vs non-streamed reasoning differ only by surrounding
  newlines; no checkpoint-VRAM log line for this model), and the bf16 run has not been repeated since the fix.
  `-TpFault` has not been run.

### Chat template, thinking, tool calls

- The chat template (`chat_template.jinja`, rendered by minja, polyfills off) ends the generation prompt with
  `<|im_start|>assistant\n<think>`: **the model always thinks**. `enable_thinking` / `reasoning_effort` are ignored;
  `/v1/models` reports `reasoning.default_enabled: true`. The server splits the generation at `</think>` into
  `reasoning_content` and `content` (streaming and not), and `usage.completion_tokens_details.reasoning_tokens` is always
  reported. Replayed assistant turns keep their thought only after the last user message.
- Tool calls are Pythonic: `<|tool_call_start|>[get_weather(location='Boston, MA'), ...]<|tool_call_end|>`. The server parses
  them into OpenAI `tool_calls` (arguments as a JSON string, `finish_reason: "tool_calls"`), streams them through the same
  gate as the other dialects, and converts `arguments` back to a mapping when it renders a replayed call (the template raises
  on a string). A malformed or truncated span degrades to plain content.
- Stop token `<|im_end|>` (124900).

## What is optimized and what is not

bf16 correctness comes first; the MoE path is the **staged** one from lfm2-plan.md 3.5:

- Decode (one row): the fp32 router kernel, then one host readback of the 4 expert ids per MoE layer (38 syncs per token), then
  per selected expert gate_up -> silu_mul -> down with the existing GEMMs, then a deterministic fp32 combine.
- Prefill (chunks of up to 256 rows): router, a deterministic counting sort of the (token, slot) pairs by expert, one count
  readback per layer, and one gate_up / silu_mul / down per non-empty expert.
- Not done yet (plan phase L4): device-indexed grouped decode kernels (no host sync), a grouped prefill GEMM over the sorted
  rows, larger MoE prefill chunks, any speculative decoding, the trellis kernels' indirect expert pointer. Only text exists: no
  vision, no audio.
- Both TP ranks run the same router on identical bits; every expert is split across the ranks (two all-reduces per layer).

## Validated numbers (bf16 TP=2)

KL gate (lfm2-plan.md 6.4), epsilon 0.005 set from LFM's own measured noise term; each cell is
`KL(fp32 truth || r4dx)` against `KL(fp32 truth || HF-bf16)`:

| segment group | r4dx bf16 TP=2 | HF-bf16 (noise) |
|---|--:|--:|
| ALL | 0.00993 | 0.01248 |
| Thai | 0.0165 | 0.0216 |
| code | 0.0076 | 0.0086 |
| english | 0.0077 | 0.0100 |
| tool | 0.0028 | 0.0026 |
| top-1 agreement | 96.83 % | 96.53 % |

The gate (`<= 1.5 x HF-bf16 + 0.005` per group, top-1 not worse than HF-bf16 minus 1 point) **passes**; r4dx is at or below
HF-bf16's own bf16 noise in every group except `tool` (0.0028 vs 0.0026, inside the allowance).

Speed (bf16, TP=2, one 2.45 B-active model on two R9700): teacher-forced scoring **25 ms per row**; server decode about
**60 tok/s**. No prefill, TTFT or long-context numbers have been taken yet, and none are claimed.

## Trellis (one GPU)

Done, 2026-10-16. The one-card format is a **hybrid**: K5 trellis for every expert except layers 2-5, bf16 for everything
that is not an expert and for the experts of layers 2-5. It passes the KL gate on every group; the 4.5-bit plan of
lfm2-plan.md 7 did not (details below). Unrotated, built with the Gemma tooling ported to `lfm2_moe`.

### Recipe

- **Calibration:** 1,170 model-generated samples (1.63 M tokens: 3 passes over 390 prompts, temperature 0.7, top_p 0.95,
  from the r4dx bf16 TP=2 server).
- **Hessians:** `hessian_capture.py --arch lfm2_moe`, per-expert routed Hessians plus a pooled one: 5,025 files, 31.3 GiB,
  70 minutes. Minimum routed rows per expert 1,195; 12 expert files were shrunk toward the pooled Hessian; numerically dead
  `mlp_mid` channels are exempt.
- **Oracles:** K4 / K5 / K6, `quantize-model --backend hip --hessian-basis matched` on device 1 (103 / 106 / 106 minutes).
  torch 2.9.1+rocm needs a CPU Cholesky and a chunked `had_r`, because ROCm matmul returns garbage above 2^20 folded rows.
  **K6 cannot be used**: the kernels and the converter / loader only instantiate KB 4 and 5.

### Results

Increment of `KL(fp32 truth || r4dx)` over r4dx bf16, per chat group (gate: increment <= 0.01 per group; the epsilon 0.005
version is also used for bf16):

| variant | size | code | english | thai | tool | ALL | gate |
|---|--:|--:|--:|--:|--:|--:|---|
| mix4.5 (4.5085 bpw) | 13.4 GB | +.0184 | +.0245 | +.0291 | +.0038 | +.0222 | FAIL |
| mix4.75 | -- | -- | -- | -- | -- | +.0175 | FAIL |
| all-K5 (5.02 bpw) | 14.8 GB | +.0121 | +.0188 | +.0197 | +.0022 | +.0156 | FAIL |
| K5 experts + bf16 non-expert | 15.8 GB | +.0080 | +.0107 | +.0137 | +.0012 | +.0100 | English / Thai FAIL |
| **hybrid** (K5 + non-expert bf16 + experts of layers 2-5 bf16) | **18.89 GB** | +.0053 | +.0080 | +.0046 | +.0009 | +.0055 | **PASS** all groups |

Layer sweep (K5 base + bf16 non-expert + bf16 experts of layers 2-N), ALL increment / size: 2-3 +.0077 / 17.3 GB; 2-4
+.0072 / 18.1; **2-5 +.0055 / 18.9**; 2-8 +.0046 / 21.2; 2-11 +.0036 / 23.5; 2-14 +.0029 / 25.8. Layers 15-27 only: +.0074;
layers 28-39 only: +.0090. A 4.75-bit base plus layers 2-5 fails (english +.0109); a 4.5-bit base fails.

Why: HF bf16 keeps the identical 4-expert set against the fp32 truth for only 87 % of tokens (3.3 % of expert slots differ;
85 % at layers about 20-30). Quantization damage is broad (75 % of the increase sits in the 50th-95th percentile of bf16
noise, not in outliers) and concentrated in the early layers' experts.

### Hybrid numbers

- Container 18.89 GB; **20.29 GiB** VRAM peak (teacher-forced, 8192 context).
- Top-1 **96.13 %** (bf16 TP=2 96.83 %, HF-bf16 96.53 %); all-chat test KL **.01545** (r4dx bf16 .00993, HF-bf16 noise .01248).
- Teacher-forced **17.0 ms/row** on one GPU (bf16 TP=2: 25 ms/row). Server decode about **61 tok/s** on one GPU with the
  untuned fallback GEMM tuning (bf16 TP=2: about 60-64 tok/s). Smoke suite: 75 checks pass.
- Not done yet: tuning-table rows for the LFM2 trellis shapes, M = 256 / int8 prefill, grouped MoE kernels, a bf16 attention
  kernel for head_dim 64, a long-context / needle test, prefill TTFT numbers.

### Reproduce

```powershell
trellis_quant.py mix --bpw 5.0 --src <q5 dir> --out-dir <mix5.0 dir>
r4dx-convert --input E:\models\LiquidAI\LFM2.5-24B-A2B --output E:\models\r4dx\lfm2\lfm2-trellis-hybrid.r4dx `
    --trellis-from <mix5.0 dir> --layouts bf16 --lm-head bf16 `
    --keep-bf16 '^(?!.*\.moe\.experts\.)|text\.layers\.(2|3|4|5)\.moe\.experts\.'
r4dx-server --model <container> --layout trellis --dialect lfm2 --tokenizer-dir <ckpt dir>
```

TP=1 only: trellis at TP=2 is refused.

## Pointers

- [lfm2-plan.md](lfm2-plan.md): design, decisions, risks, task list.
- [lfm2-semantics.md](lfm2-semantics.md): checkpoint layout, numerics, router statistics, template and tokenizer facts.
- [server.md](server.md): the server's request/response surface (dialect-independent parts).
- [tp.md](tp.md): the two-GPU tensor-parallel layer and its fault-recovery rules (`smoke.ps1 -TpFault`).
