# Nanbeige4.3-33B in r4dx

Status 2026-10-26: **w8a16 at `--tp 2` works end to end**: CLI and server, thinking on and off, Chinese, XML tool calls,
the prefix cache (server smoke: every check passes). The KL gate against an fp32 CPU truth and the greedy smoke both pass (see "Accuracy"). This page is the
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
| decode, short context | 31.4 tok/s |
| prefill, 40-60 token prompts | 420-510 tok/s |
| VRAM per card at --max-ctx 8192 / 36864 | 18.4 / 21.6 GiB |

By prompt length (`r4dx-cli --stats`, 128 greedy tokens after the prompt, h128_gqa7 attention, idle machine):

| prompt | prefill | decode (128 tokens after it) |
|--:|--:|--:|
| 2092 | 624 tok/s (3.35 s) | 31.2 tok/s |
| 8346 | 616 tok/s (13.6 s) | 30.0 tok/s |
| 31682 | 581 tok/s (54.5 s) | 26.2 tok/s |

Prefill is GEMM-bound and flat in length: the w8a16 kernel (`r4dx_gemm_w8a16_nt_m64`) is decode-shaped, one launch per 64
rows with no large-M blocking, and reaches about 20 TFLOPS per card (neither the WMMA peak nor the weight bandwidth: 10
chunks/s x 17 GB is about 170 GB/s). A large-M w8a16 GEMM is the lever, not attention; the Nanbeige shapes also have no
tuning rows yet (`W8a16FallbackTuning`).

## Accuracy

**Gate: PASS** (2026-10-26, w8a16 `--tp 2`, h128_gqa7 attention). Truth is an fp32 CPU forward
(`tools/reference/llama/full_logits_llama_cpu.py --variant truth`, 212 min); the noise yardstick is the same forward in bf16
with bf16 SDPA and fp32 rope `inv_freq` (`--variant bf16sdpa`). Rule (plan 6.3): KL(truth||r4dx) <= 1.5 x KL(truth||noise)
+ 0.005 and top-1 >= noise - 1 point, per group, over the assistant tokens of 14 chat conversations
(`kl_corpus/build_chat_nanbeige.py`). `tools/reference/llama/kl_gate_llama.py`:

| group | rows | KL noise | KL r4dx | limit | top-1 noise | top-1 r4dx | p99 r4dx | |
|---|--:|--:|--:|--:|--:|--:|--:|---|
| english | 1426 | 0.00099 | 0.00145 | 0.00648 | 97.76 | 97.55 | 0.0085 | PASS |
| thai | 2531 | 0.00071 | 0.00125 | 0.00606 | 98.89 | 98.74 | 0.0110 | PASS |
| code | 1310 | 0.00073 | 0.00094 | 0.00610 | 99.47 | 98.93 | 0.0073 | PASS |
| tool | 510 | 0.00063 | 0.00078 | 0.00594 | 98.63 | 98.04 | 0.0066 | PASS |
| zh | 283 | 0.00054 | 0.00097 | 0.00582 | 98.59 | 98.94 | 0.0051 | PASS |
| **ALL** | 6060 | 0.00077 | 0.00118 | 0.00615 | 98.71 | 98.45 | 0.0093 | **PASS** |
| raw cpp_source | 511 | 0.00184 | 0.00194 | | 98.43 | 98.04 | 0.0210 | reported |
| raw english_prose | 511 | 0.00244 | 0.00504 | | 96.28 | 95.89 | 0.0408 | reported |
| raw python_source | 511 | 0.00192 | 0.00381 | | 95.89 | 94.32 | 0.0575 | reported |
| raw thai_prose | 511 | 0.00206 | 0.00315 | | 95.89 | 96.87 | 0.0172 | reported |
| raw mixed_long (6143 tokens) | 6143 | 0.01514 | 0.02003 | | 96.87 | 96.19 | 0.2476 | reported |

r4dx costs about 1.5x the bf16 reference's own rounding noise. The h128_gqa7 kernel and the reference attention kernel score
the same against the truth (unmasked rows: KL 0.0029 both, top-1 97.7 / 97.9). No fp32 residual is needed, although the
residual stream carries a massive-activation channel (max |h| about 32k from layer 2); norm weights are tame (largest 5.25,
the final norm; `tools/reference/llama/survey_small_tensors.py`).

**Greedy smoke: PASS** (plan 6.3; HF cannot generate with this model here). `r4dx-cli --think on --temperature 0
--max-tokens 128 --dump-token-ids` on four prompts (English, Thai, code, Chinese), the fp32 truth teacher-forces the same ids,
`tools/reference/llama/greedy_check_llama.py` checks each generated token: 495 of 505 are the truth's top-1, the other 10 are
near-ties (truth top-1 minus the chosen token's log-prob 0.014-0.243, limit 0.5), no misses.

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
