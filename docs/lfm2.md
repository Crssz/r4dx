# LFM2.5-24B-A2B in r4dx

Status 2026-10-15: **bf16 TP=2 works end to end** (KL gate PASS, server smoke by hand: chat with the reasoning split, Thai,
a tool call). Trellis (the one-card shipping format) is **in progress**; nothing about it is validated yet. This page is the
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
  authored but not yet run: the first GPU run may need small adjustments to its model-dependent checks.

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

## Trellis (in progress)

The shipping plan is a trellis mix of about 4.5 bits per weight (about 14 GB) on one card, built with the Gemma tooling ported
to the `lfm2_moe` arch (Hessian collection, oracle, mix, import), unrotated. The tooling exists (Hessian / oracle / mix /
import commands accept `lfm2_moe`); no trellis container has been produced or measured, so there are no trellis accuracy or
speed numbers and the trellis kernels have not been wired to the expert linears. See lfm2-plan.md sections 7 and 9 (L3).

## Pointers

- [lfm2-plan.md](lfm2-plan.md): design, decisions, risks, task list.
- [lfm2-semantics.md](lfm2-semantics.md): checkpoint layout, numerics, router statistics, template and tokenizer facts.
- [server.md](server.md): the server's request/response surface (dialect-independent parts).
- [tp.md](tp.md): the two-GPU tensor-parallel layer and its fault-recovery rules (`smoke.ps1 -TpFault`).
