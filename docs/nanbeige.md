# Nanbeige4.3-33B in r4dx

Status 2026-10-26: **w8a16 at `--tp 2` works end to end**: CLI and server, thinking on and off, Chinese, XML tool calls,
the prefix cache (server smoke: every check passes). KL gate against an fp32 CPU truth: see "Accuracy". This page is the
user-facing summary; the design, task list and measured facts are [nanbeige-plan.md](nanbeige-plan.md). The trellis
(one-card) format was dropped from scope on 2026-10-26.

## Architecture

`Nanbeige/Nanbeige4.3-33B` is a plain dense Llama (`LlamaForCausalLM`, stock transformers, no remote code): 33.1 B
parameters, 52 layers, hidden 7168, every layer full GQA attention (56 q heads / 8 kv heads, head_dim 128, rope theta 1.2e8,
no scaling, 524288 positions) followed by a SwiGLU MLP (I = 22016), RMSNorm eps 1e-5, untied lm_head, vocab 166912 rows
(166107 tokenizer ids; nothing is masked, as in HF).

It is a sixth `TextModel` family (`Arch::kLlama`: `LlamaModel`, `LlamaTpModel`), a copy-adapt of the LFM2 engine without
ShortConv and MoE; the other families are untouched.

## Container

```powershell
r4dx-convert --input E:\models\Nanbeige\Nanbeige4.3-33B --output E:\models\r4dx\nanbeige\nanbeige43-33b-w8a16-g128.r4dx `
    --layouts w8a16 --no-bf16
```

34.8 GB (32.4 GiB), 732 tensors, 139 s on 24 CPU threads. Every linear (lm_head too) is w8a16 at group 128; norms and the
embedding stay bf16. The header carries `model_arch: llama`, `norm_kind: plain` and the verbatim `config.json`. bf16 (66 GB)
fits neither one card nor two, so there is no bf16 container to run.

## Tokenizer directory

The chat template ships inside `tokenizer_config.json`; r4dx wants `chat_template.jinja` next to `tokenizer.json`. The
dialect's default tokenizer directory is `<R4DX_MODELS_ROOT>\Nanbeige\Nanbeige4.3-33B-tok`, holding `tokenizer.json`,
`tokenizer_config.json`, `generation_config.json` and `chat_template.jinja` = the config's `chat_template` string verbatim
(SHA-256 `f2b1e2a1be25d277e8c903a2aa4d69ca5fd8a0fcb37d4792088e72426df5b91c`).

## Running

```powershell
# one-shot (both GPUs; unset HIP_VISIBLE_DEVICES)
r4dx-cli --model E:\models\r4dx\nanbeige\nanbeige43-33b-w8a16-g128.r4dx --layout w8a16 --tp 2 --max-ctx 8192 `
    --think on --prompt-file prompt.txt --max-tokens 512 --temperature 0 --stats

# OpenAI-compatible server
r4dx-server --model E:\models\r4dx\nanbeige\nanbeige43-33b-w8a16-g128.r4dx --layout w8a16 --tp 2 --max-ctx 32768 --port 8080
```

- `--tp 2` is required (17.3 GiB of weights per card). `--pp 2`, `--batch`, `--mtp`, `--dflash` and images are refused.
- `--max-ctx` defaults to 32768 (KV is bf16, 104 KiB per token per card); anything up to 524288 is accepted if it fits.
- `--dialect auto` picks `nanbeige` from the container (`model_arch: llama`) or the tokenizer (`LlamaTokenizer` with the
  `<|im_start|>` bos); `--dialect nanbeige` forces it.
- On Windows PowerShell 5.1 pass non-ASCII prompts with `--prompt-file` (the shell hands native programs the ANSI code page).

### Chat template, thinking, tool calls

Qwen's surface: the generation prompt ends `<|im_start|>assistant\n<think>\n` (thinking on) or `<think>\n\n</think>\n\n`
(off); the server splits `reasoning_content` from `content` at `</think>`. Thinking follows the request
(`chat_template_kwargs.enable_thinking`, or the server's `--think`); r4dx's default is off, as for Qwen, while the template's
own default is on. With no system message the template inserts its own Chinese system prompt. History reasoning is kept
unless `chat_template_kwargs.preserve_thinking` is false. Tool calls are the XML
`<tool_call><function=NAME><parameter=K>V</parameter></function></tool_call>` form (parsed into OpenAI `tool_calls`); pass
`chat_template_kwargs.tool_call_format = "json"` for the template's JSON variant. Tool results render as a user turn.

The tokenizer is the `legacy: false` SentencePiece shape (Metaspace prepend "first" + byte_fallback BPE + a Strip decoder);
r4dx matches `tokenizers` / `AutoTokenizer` on 176 golden cases (`tokenizer_golden_nanbeige`).

## Speed (2026-10-26, R9700 x2, w8a16, --tp 2, greedy)

| | |
|---|---|
| container load (DirectStorage) | about 21 s |
| decode, short context | 30.2-30.7 tok/s |
| prefill, 40-50 token prompts | 440-610 tok/s |
| VRAM per card at --max-ctx 8192 | 18.8 GiB |

## Accuracy

Pending the gate run (tools/reference/llama/kl_gate_llama.py; the plan's section 6.3 rule).

## Tests

| test | what |
|---|---|
| `test_llama_config`, `test_llama_container`, `convert_llama_layout` | CPU: config, loader table, TP=2 shard plans, converter |
| `tokenizer_golden_nanbeige` | CPU: tokenizer + template vs HF |
| `test_dialect` | CPU: the nanbeige dialect |
| `llama_tiny_e2e` | GPU: the stack on a tiny checkpoint vs HF, bf16 / w8a16, TP=1 / TP=2 |
| `test_attn_h128_gqa7` | GPU: the head_dim 128 / GQA 7 attention kernels vs the reference kernel |
| `tools\nanbeige\smoke_server.ps1` | both GPUs: the server end to end |

## Known limits

- bf16 KV only (fp8 KV is plan task N5-2), so long contexts cost 208 KiB per token.
- No speculative decoding (no drafter), no batched decode, no `--pp 2`.
- A DirectStorage load crashed once (an access violation inside `dstoragecore.dll`) while a long CPU job was reading the same
  drive; `R4DX_DIRECT_STORAGE=0` loads through the memory mapping instead.
