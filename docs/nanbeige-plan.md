# Nanbeige4.3-33B support for r4dx -- design and task plan

Written 2026-10-25 against `main` at `7fa31c9`. Proposed branch `nanbeige` (from `main`). **Design only**: no source file was
edited, no GPU workload or ctest was run, nothing was downloaded while writing it. Structure follows `docs/lfm2-plan.md`,
the precedent for a side model family behind `TextModel`. Where it says MUST, do exactly that. Anything tagged
**UNVERIFIED** is inferred, or read from a web summary rather than the file itself; the task named next to it settles it
before the code that depends on it is written. Section 11 lists every UNVERIFIED item, section 10 the open questions.

## Summary

`Nanbeige/Nanbeige4.3-33B` is a plain dense Llama: `LlamaForCausalLM`, 52 layers, every layer full-attention GQA
(56 q heads / 8 kv heads, head_dim 128) followed by a SwiGLU MLP, untied head, 33.12 B parameters, bf16 (66.3 GB on disk,
14 shards, no custom modeling code). It becomes a sixth `TextModel` family (`Arch::kLlama`, `LlamaModel` / `LlamaTpModel`),
built as a copy-adapt of the LFM2 engine with the ShortConv and MoE parts removed. The Qwen, Gemma, LFM2, qwen4_exp and
Ornith paths are not touched.

Two shipping formats:

| Format | Where | Container | Per card | Decode estimate (UNVERIFIED) |
|---|---|---|---|---|
| **w8a16 g128, `--tp 2`** | both R9700s | about 34.8 GB (32.4 GiB) | about 17.3 GiB weights | about 30-35 tok/s |
| **trellis about 4.5 bpw, TP=1** | one R9700 | about 20.4 GB (19.0 GiB) | 19.0 GiB weights | about 30 tok/s |

bf16 (66 GB) fits neither one card nor two, so **there is no r4dx bf16 run**: w8a16 TP=2 is the first engine and the
near-reference one, and both formats are gated against an fp32 CPU truth. First milestone is text only: forward pass,
tokenizer, chat template, thinking split, the existing Qwen XML tool parser, server smoke. No drafter, no vision. Every GPU
step is a command handed to the user (GPU runs need approval).

## 1. Decisions (proposed; the user confirms or changes them before N1)

1. **Side model.** Separate `LlamaModel` / `LlamaTpModel` behind `TextModel` (like LFM2). Copy-adapt, not a shared base:
   templating the LFM2 and Llama engines over one layer graph is a later dedupe, and copying keeps LFM2's byte identity
   free of risk.
2. **Weights.** w8a16 g128 at TP=2 first (the near-reference engine, also the calibration-sample generator), then trellis
   about 4.5 bpw at TP=1 as the one-card format, unrotated (the Gemma and LFM2 choice) unless the gate says otherwise.
3. **First milestone scope.** Text forward, tokenizer, chat template, thinking split, tool calls (existing
   `qwen3_coder`-style XML parser), server smoke. No drafter / MTP, no vision.
4. **Attention.** The stage-0 reference kernel first (exact, slow); a libr4d `h128_gqa7` instantiation of the gqa6 templates
   in N5. KV bf16 first; fp8 KV with calibration in N5.
5. **Gate.** fp32 CPU truth, layer-streamed; HF-bf16 as the noise term (section 6.3).

## 2. Model facts

### 2.1 Read from the Hugging Face repo (2026-10-25, web; re-read from the local files in N0-2)

| Item | Value |
|---|---|
| class / model_type | `LlamaForCausalLM` / `llama`; no `.py` files in the repo (stock transformers; the card's `trust_remote_code=True` is not needed, UNVERIFIED until N0-3 loads it without) |
| layers / hidden / FFN / vocab | 52 / 7168 / 22016 / 166912 |
| attention | 56 q heads, 8 kv heads (GQA 7), head_dim 128, no bias, no QK norm |
| rope | theta 1.2e8, no scaling, `max_position_embeddings` 524288; HF Llama `rotate_half` pairs `(i, i+64)` |
| norm | RMSNorm eps 1e-5 (HF Llama: fp32 compute, cast to input dtype, then `weight *`) |
| MLP | SiLU, `gate_proj`, `up_proj`, `down_proj`, no bias |
| embeddings | untied (`lm_head.weight` is its own tensor), no embedding scale, no logit softcap |
| ids | bos 166100 `<|im_start|>`, eos 166101 `<|im_end|>`, pad 0 (`<unk>`); `<|endoftext|>` 166102; `<think>` 166103, `</think>` 166104, `<tool_call>` 166105, `</tool_call>` 166106 (the last four are added tokens NOT marked special) |
| tokenizer | `LlamaTokenizer` (SentencePiece; `tokenizer.model` 2.78 MB and `tokenizer.json` 18.5 MB), `add_bos_token` true, `add_eos_token` false |
| chat template | embedded in `tokenizer_config.json` (no `chat_template.jinja`); kwargs `enable_thinking` (default on), `preserve_thinking`, `tool_call_format` (`xml` recommended, `json` supported) |
| serving hints (card) | SGLang `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`; sampling: agentic temperature 1.0 top_p 0.95, chat 0.6 / 0.95; no top_k or repetition penalty given |
| generation_config | bos 166100, eos 166101, pad 0, transformers 4.51.1 |
| license | Apache-2.0 |

### 2.2 Sizes (computed from the config)

Per layer: attention q 51.38 M + k 7.34 M + v 7.34 M + o 51.38 M = 117.44 M; MLP 3 x 157.81 M = 473.43 M; layer 590.87 M.
Totals: body linears 52 x 590.87 M = **30.725 B (92.8 %)**, embedding 1.196 B, lm_head 1.196 B, norms 0.75 M = **33.12 B**.
bf16 = 66.24 GB (the repo says 66.3 GB: consistent).

| Part | w8a16 g128 (1.0156 B/weight) | trellis TP=1 |
|---|--:|--:|
| body linears | 31.21 GB | 17.28 GB (4.5 bpw) |
| lm_head | 1.22 GB (w8a16) | 0.67 GB (w4a16 g32) |
| embedding (bf16, gathered) | 2.39 GB | 2.39 GB |
| **container** | **34.81 GB (32.42 GiB)** | **20.35 GB (18.95 GiB)** |
| per rank at TP=2 | 15.60 + 0.61 + 2.39 (replicated) = **18.60 GB (17.33 GiB)** | -- |

KV per token: 52 layers x 8 kv heads x 128 x (K, V). bf16 = **208 KiB/token** (104 KiB per rank at TP=2); fp8 = 104 KiB.

| Context | bf16 KV total | per rank, TP=2 | fp8 KV total |
|---|--:|--:|--:|
| 32768 | 6.5 GiB | 3.25 GiB | 3.25 GiB |
| 65536 | 13 GiB | 6.5 GiB | 6.5 GiB |
| 131072 | 26 GiB | 13 GiB | 13 GiB |

So the 512k context of the card is out of reach on this machine. TP=1 trellis leaves about 11.5 GiB for KV, arena and
logits: about 55k tokens at bf16 KV, about 110k at fp8. TP=2 w8a16 leaves about 13.5 GiB on device 1 and about 10 GiB on
device 0 (the desktop card): about 96k tokens at bf16 KV, about 190k at fp8, bounded by device 0. Proposal: default
`max_ctx` 32768 on both formats, explicit values up to what fits (open question 1).

## 3. Design: text model

### 3.1 Dispatch, config, TextModel fit

- `src/model/arch.h/.cpp`: `Arch::kLlama`, `ArchName` = `"llama"`; `DetectArchFromMetadata` accepts `model_arch == "llama"`
  and `model_config.model_type == "llama"` / `architectures` naming `LlamaForCausalLM`. `ResolveContainerMaxCtx` for kLlama:
  default 32768, an explicit value up to `max_position_embeddings` accepted (KV allocation then fails loudly if it does not
  fit). Existing arch tests extended; the other branches stay byte-identical.
- `src/model/llama_config.h` (header-only, HIP-free, strict `FromJson` like `Lfm2Config`): geometry above, `head_dim` from the
  config (present: 128) else `hidden / heads`; asserted: no `rope_scaling`, no attention / MLP bias, `tie_word_embeddings`
  false, `hidden_act` silu. `Shard(world, rank)`: q heads 56 -> 28, kv heads 8 -> 4 (GQA stays 7), `intermediate_size`
  22016 -> 11008, vocab slice 166912 -> 83456. `ToModelConfig()` fills the generic fields (GDN fields 0, `arch` = kLlama).
- `LoadTextModel` (`tp_model.cpp`) dispatches kLlama to `LoadLlamaTextModel` (new `llama_local_text_model.cpp`, a copy of
  `lfm2_local_text_model.cpp`): world 1 -> `LlamaLocalTextModel`, world 2 -> `LlamaTpModel`. TextModel map: Prefill /
  DecodeStep / Greedy / Sampled / Reset / Save+RestoreCheckpoint / `*Profiled`; `PrefillMultimodal` with spans throws;
  `HasVision` / `MtpEnabled` / `DflashEnabled` false.
- **State is positional only** (no recurrent state): `SaveCheckpoint` / `RestoreCheckpoint` only move `pos_` (KV positions are
  overwritten in place), so the server's prefix cache can rewind to any earlier position like a plain transformer. The
  `PrefixState` logic needs no change; the cheapest correct choice is to keep the GDN-style behaviour (reuse on strict
  extension, else restore the prompt checkpoint, else reset) and treat a true rewind as a later optimization.

### 3.2 Layer graph (`LlamaModel::RunChunk`, T <= 256 rows, one sequence)

```
embed gather (device table, bf16, no scale) -> h
for i in 0..51:
  x = input_layernorm(h)                         r4dx_rmsnorm_plain_bf16 (hidden 7168, eps 1e-5)
  q, k, v = proj(x); rope(q, k) theta 1.2e8; KV write; attention; m = o_proj(attn)
  [TP] all-reduce(m);  h += m                    r4dx_residual_add_bf16
  x = post_attention_layernorm(h)
  g = gate_up(x) [T, 2 * 22016]; a = silu_mul(g); f = down(a)
  [TP] all-reduce(f);  h += f
h = norm(h) -> lm_head (the rank's vocab slice under TP) -> fp32 logits
```

Reuse: `Lfm2DenseMlp`'s body (gate_up -> silu_mul -> down over `ApplyLinear`) is layout-generic already (bf16, w8a16,
trellis all go through `QuantLinear`); `LlamaDenseMlp` is that class with a `LlamaConfig`. Residual stream bf16 (no sandwich
norms, no massive-activation evidence); an fp32 residual is a contingency only if the N2 gate misses while the noise term
is stable (the Gemma lesson, docs/gemma4-plan.md 3.2.1). N0-5 records the norm-weight and activation ranges that decide it.

### 3.3 Attention (52 layers)

- New `LlamaAttnLayer` (`src/model/attention/include/r4dx/model/attention/llama_attention_layer.hpp`): `Lfm2AttnLayer` minus
  the per-head q/k norms. q / k / v separate linears first; a fused `qkv` (N 9216 at TP=1, 4608 at TP=2) is an N5 launch-count
  option, decided by a profile.
- Rope: `r4dx_rope_neox_bf16` at head_dim 128, theta 1.2e8, fp32 tables. HF computes `inv_freq = 1 / theta^(arange(0, 128,
  2) / 128)` in fp32 and the angle as fp32 `position x inv_freq`; the kernel MUST match that order (a golden at positions up to
  131071 pins it, N1-9), because at large positions the fp32 angle rounding is part of what the model was trained with.
- Kernel: **N1 uses the stage-0 reference kernel** (`attn_ref.h`: any head_dim in [8, 512], any GQA, exact). It re-reads K/V
  per q head (7x redundant at GQA 7), so it is the gate kernel, not the shipping one: about 0.9 GB of KV reads per layer per
  token at 32k context (56 heads x 16.8 MB).
- **N5-1: libr4d `r4d_attn_paged_h128_gqa7.hip`**, the gqa6 prefill / decode templates instantiated at head_dim 128, GQA 7,
  bf16 KV (`KVP 1`), the way `r4d_attn_paged_h64_gqa4.hip` did for LFM2. GQA 7 does not divide a prefill workgroup's 128 rows
  (`BLOCK_Q` = 18, 126 rows live); the gqa6 instantiation already lives with 128 % 6 != 0 (21 x 6 = 126), so the rows past
  `BLOCK_Q * GQA` must already be masked on store -- **UNVERIFIED** until N5-1 reads the store path. Decode holds
  `q_len x 7 <= 64` rows per workgroup (`q_len <= 9`). `O_KWIDE` (16 x 16-byte chunks per key) fits 128 bf16 dims exactly, so
  the full gqa6 prefill option set may apply (UNVERIFIED). `third_party/check_qwen_attn_isa.cmake` MUST still pass. fp8 KV
  needs the `KVP 0` instantiation too (N5-2).
- Cost note: all 52 layers are full attention (Qwen has 16), so long-prompt prefill is attention-heavy: at 32k tokens the
  attention FLOPs are about 40 % of the linear FLOPs, at 128k they dominate. Split-KV prefill from 2048 tokens (the LFM2 law)
  carries over.

### 3.4 TP=2 split

Everything is split inside a layer; two all-reduces per layer (104 per token) on the bf16 `[T, 7168]` sublayer output (a
decode row is 14 KiB: about 8 us per all-reduce through pinned host memory, about 0.9 ms per token in total, UNVERIFIED).

| Part | Split |
|---|---|
| attention | q heads 56 -> 28, kv heads 8 -> 4 per rank (GQA 7); q / k / v column-parallel (N 3584 / 512 / 512); `o_proj` row-parallel (K 3584) |
| MLP | `gate|up` column-parallel, two segments (N 11008 each); `down` row-parallel (K 11008) |
| embedding / head | table replicated (gather); `lm_head` rows `[r x 83456, (r+1) x 83456)`, the existing vocab-split merges (`tp_vocab.h`) |
| norms | replicated |

`tp::RuleFor` gains the `text.layers.{i}.attn.*` / `mlp.*` names (the LFM2 rules are the template). w8a16 shard legality:
every per-rank K is a multiple of 128 (3584 = 28 x 128, 11008 = 86 x 128), so `W8a16FallbackTuning` picks SK 2 for
`down` and SK 8 for the rest; nothing is refused.

## 4. Converter and container

Same safetensors-compatible shell as `docs/container-format.md`; `__metadata__` gains `"model_arch": "llama"`,
`"norm_kind": "plain"`; `model_config` is the verbatim config.json.

| Container tensor | HF source | Notes |
|---|---|---|
| `text.embed_tokens` | `model.embed_tokens.weight` | bf16 `[166912, 7168]` |
| `text.final_norm` | `model.norm.weight` | bf16 `[7168]` |
| `lm_head.{L}` | `lm_head.weight` | untied; w8a16 in the w8a16 container, w4a16 g32 (LDLQ) in the trellis one |
| `text.layers.{i}.input_norm`, `.post_attn_norm` | `input_layernorm`, `post_attention_layernorm` | bf16 `[7168]` |
| `text.layers.{i}.attn.{q,k,v,o}.{L}` | `self_attn.{q,k,v,o}_proj` | no bias |
| `text.layers.{i}.mlp.gate_up.{L}`, `.mlp.down.{L}` | `mlp.{gate;up}_proj`, `mlp.down_proj` | gate rows first (the LFM2 / Qwen convention) |

Code: `src/convert/llama_layout.{hpp,cpp}` (`AddLlamaTextStack`, the std::function-handle pattern of `lfm2_layout.*`);
`RunConvert` dispatches on `model_type == "llama"`; `--layouts w8a16 --no-bf16` and `--trellis-from` as for LFM2; `--mtp`,
vision, audio, `--kv-calib` rejected for this arch until N5-2 adds fp8 KV. Coverage audit: every checkpoint tensor consumed
or allow-listed (`rotary_emb.inv_freq`, if present, is allow-listed), else the run fails before an output file exists.
Tests: `tests/convert/test_llama_layout.cpp` on a synthetic tiny checkpoint (`tools/convert_ref/make_tiny_llama.py`: 3 layers,
hidden 1792, 14 q heads / 2 kv heads, head_dim 128 so GQA 7 and the N5 kernel geometry are exercised, FFN 512, vocab 4096,
untied head), and a layer-limited real container `nanbeige-l8` (layers 0-7 in bf16, about 9.5 GB of layers plus 4.8 GB of
embedding and head, fits one card) for single-device GPU goldens.

w8a16 conversion: LFM2's 48 GB took 108 s; this is about 66 GB of reads, a few minutes (one [CPU-heavy] job; the converter
streams tensors, it never holds the checkpoint).

## 5. Tokenizer, chat template, server

### 5.1 Tokenizer

`LlamaTokenizer` is SentencePiece BPE with byte fallback. r4dx's `kSpmByteFallback` backend (`bpe_tokenizer.cpp`, the
Gemma shape) accepts exactly: normalizer `Replace(" " -> U+2581)`, no pre-tokenizer or `Split(" ", MergedWithPrevious)`,
decoder `Sequence[Replace, ByteFallback, Fuse]`, no special tokens inserted by the post-processor, no added token with
`normalized`/`lstrip`/`rstrip`. A Llama-family `tokenizer.json` usually differs on three points (**UNVERIFIED** until N0-4
reads the head of the file):

1. normalizer `Sequence[Prepend(U+2581), Replace(" " -> U+2581)]` (the legacy SPM "dummy prefix"), or a `Metaspace`
   pre-tokenizer with `prepend_scheme` instead (the `legacy: false` shape);
2. decoder `Sequence[Replace, ByteFallback, Fuse, Strip(" ", 1, 0)]` (drops the dummy prefix's space on decode);
3. post-processor `TemplateProcessing` inserting BOS (here `<|im_start|>`, because `add_bos_token` is true).

Work (N3-1): extend `validate_spm` / `spm_encode_chunk` / `spm_decode` with the shape N0-4 finds, as a second SPM variant
(the Gemma shape stays byte-identical, pinned by its golden). The legacy dummy prefix interacts with special tokens (HF
`legacy=True` prepends U+2581 to the text after a special token too); the golden set MUST cover text right after
`<|im_start|>`, `<think>` and `<tool_call>`. Validation like LFM2: `tools/tok_ref/gen_golden_nanbeige.py` ->
`tests/tokenizer/golden_nanbeige.json` (English, Chinese, code, JSON, Thai, emoji, digit runs, whitespace runs, `\r\n`,
special tokens both ways, a 20 KB line, chat cases), raw `tokenizers` ids == `AutoTokenizer` ids asserted over the KL corpus.

BOS: chat prompts render `<|im_start|>` as text and are encoded with `parse_special = true` and no added BOS (HF
`apply_chat_template` tokenizes with `add_special_tokens=False`). Raw `/v1/completions` prompts: HF would prepend
`<|im_start|>`; the dialect's `bos_on_raw_prompt = true` matches HF (open question 4).

### 5.2 Chat template

The template lives in `tokenizer_config.json`; r4dx's `--tokenizer-dir` wants `chat_template.jinja`. N0-6 writes a tokenizer
directory `E:\models\Nanbeige\Nanbeige4.3-33B-tok\` (the Gemma `-tok` precedent): `tokenizer.json`, `tokenizer_config.json`,
`generation_config.json` and `chat_template.jinja` extracted verbatim (SHA-256 recorded). N0-6 also renders it in minja and
in HF for the KL-corpus conversations and a tool conversation, with `enable_thinking` on / off, `preserve_thinking` on / off
and `tool_call_format` xml / json; any minja gap gets a small patch in `third_party/minja/minja.hpp` noted in
`third_party/VERSIONS.md` (the Gemma M1-8 precedent). Whether polyfills stay on (Qwen) or off (Gemma, LFM2) is decided by
that render.

### 5.3 Dialect, thinking, tools

`DialectKind::kNanbeige`, `MakeNanbeige()` = `MakeQwen35()` with its own name, `eos_ids` from `generation_config.json`
(`[166101]`, plus `<|endoftext|>` 166102 if the template or the card ends turns with it, N0-6), and the prompt-side reasoning
opener the template's generation prompt actually emits (Qwen's is `"<think>\n"`; UNVERIFIED here). The card's serving flags
(`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`) say the surface is Qwen's: `<think>` / `</think>` reasoning and
the `<tool_call><function=NAME><parameter=...>` XML that `tool_call_parser.cpp` already parses. `enable_thinking` maps to the
template kwarg as for Qwen; `preserve_thinking` and `tool_call_format` become request extension fields passed to the template
(defaults: off, xml). `--dialect auto`: `tokenizer_config.json` naming `LlamaTokenizer` with bos `<|im_start|>`, cross-checked
against the container arch (`CheckDialectAgainstArch` gains `llama`). No repetition penalty is needed (the card gives none).

## 6. Reference tooling, goldens, KL gate

### 6.1 Environment (specify, do not create)

The existing reference venv (`D:\venvs\r4dx-gemma-ref`, transformers 5.18.0, torch 2.9.1+rocmsdk) has `LlamaForCausalLM`;
N0-3 only runs the import and a tiny random-config forward there. The checkpoint was saved with transformers 4.51.1;
Llama's numerics did not change between them for this config (no rope scaling), UNVERIFIED until the tiny forward matches.

### 6.2 Reference package `tools/reference/llama/`

Mirror of `tools/reference/lfm2/`, minus router and ShortConv: `arch.py`, `common_llama.py` (`DEFAULT_MODEL_DIR` =
`E:\models\Nanbeige\Nanbeige4.3-33B`, `R4DX_MODEL_DIR` override), `ref.py` (the HF `LlamaDecoderLayer` loop, streamed one
layer at a time, plus an fp32 twin), `layer_golden_llama.py`, `full_logits_llama_cpu.py` (`--variant truth|bf16sdpa`, the
shared `kl_report.py` dump format), `dump_header.py`, `survey_small_tensors.py`, `make_tiny_llama.py`, `test_scripts_tiny.py`,
`gen_samples_server.py` (after N3). `arch_table.py` gains a `llama` `ArchSpec` (section 7.1).

fp32 truth does not fit in RAM (33 B x 4 B = 132 GB, 63.5 GiB host), so it is **layer-streamed on CPU** like LFM2's: one
layer's bf16 tensors (1.18 GB) upcast to fp32 (2.36 GB), all segments through it in lockstep, per-layer hidden-state
checkpoints for resumption. Peak host memory about 6-10 GB. Truth = bf16 weights upcast, fp32 activations, eager attention,
fp32 rope, fp32 norm and head. `bf16sdpa` = stock HF numerics = the noise term. Both CPU only (`HIP_VISIBLE_DEVICES=-1`,
asserted). Cost: about 66 GB of weight reads plus about 2 x 33 B x 10k tokens = 0.66 PFLOP per variant, tens of minutes
(UNVERIFIED; N0-8 measures it).

### 6.3 Goldens and gate

- **Goldens (rung 3, CPU-produced, GPU-compared):** `layer_000` and `layer_026` and `layer_051` (fixed-seed inputs
  `[T = 64, 7168]` with perturbed norm weights, prefill then 4 decode steps; intermediates: norm out, q/k/v raw and roped,
  attention out, o_proj, mlp gate/up/act/down); unit goldens `rope_h128` (theta 1.2e8, positions to 131071), `final_norm`.
- **Corpus:** `tools/reference/kl_corpus/chat_nanbeige.json` (+ `build_chat_nanbeige.py`) from the same 10 conversations as
  `chat_lfm2.json`, rendered with this template (thinking on), scored on assistant tokens only via `score_mask`, plus 2 tool
  segments (XML format). Groups: english, code, thai, tool, chat-ALL, and a **zh** group (2 new Chinese conversations,
  author-written; Nanbeige is a Chinese-first model) -- open question 3.
- **w8a16 TP=2 gate (N2):** per group `KL(fp32||r4dx-w8a16) <= 1.5 x KL(fp32||HF-bf16) + 0.005`, and
  `top1(fp32, r4dx) >= top1(fp32, HF-bf16) - 1.0 pt`. This is the absolute rule LFM2 applied to its bf16 engine; LFM2's
  w8a16 sat +0.0033 above its bf16, so a pass is expected but not certain.
- **Trellis TP=1 gate (N4):** per group `KL(fp32||trellis) - KL(fp32||r4dx-w8a16) <= 0.01` (the Qwen / Gemma / LFM2 increment
  rule, with w8a16 standing in for bf16 because no r4dx bf16 run exists), plus the absolute rule above reported (open
  question 2).
- **Greedy smoke without an HF generator:** HF cannot generate with this model here (bf16 does not fit 2 x 32 GB, and a
  layer-streamed CPU decode reads 66 GB per token). Instead r4dx greedy-generates 128 tokens on 4 prompts, and the truth
  script teacher-forces those tokens: every generated token must be the truth's top-1, or within a recorded near-tie
  (truth top-1 minus the chosen token's logit < 0.5). Cheap, and it checks free generation, not only the corpus.
- **TP identity:** trellis TP=1 vs w8a16 TP=2 is not an identity check (different weights). Within a format: w8a16 TP=2 run
  twice must give the same bytes, and `nanbeige-l8` bf16 TP=1 vs TP=2 mean KL <= 3 x noise (the LFM2 rule).

## 7. Trellis pipeline (TP=1)

Reuse: `trellis_quant.py`, `hessian_capture.py`, `arch_table.py`, `lfm2_mix.py` (copied to `llama_mix.py` without the expert
groups), `trellis_import.hpp`, `tools/quant2/trellis_convert.ps1` patterns. A `tools/nanbeige/trellis_pipeline.ps1` with stage
names `hessian | oracle-k4 | oracle-k5 | mix | convert | gate`; a stage without `-Run` only prints its command and refuses
GPU work.

### 7.1 Arch entry

`ArchSpec` `llama`: taps `in` = `input_layernorm` output (feeds q, k, v), `out` = `o_proj` input, `mlp_in` =
`post_attention_layernorm` output (feeds gate, up), `mlp_mid` = `down_proj` input, `lm_head` = post-`norm` hidden. Linears
`attn.{q,k,v,o}`, `mlp.{gate_up,down}`. Every K and N is a multiple of 128 (TP=1: 1024, 7168, 22016, 44032, 166912), so the
quantizer core is unchanged. Unrotated, as decided; the norm-weight survey (N0-5) is the evidence for revisiting.

### 7.2 Calibration and Hessians

- **Samples:** the r4dx w8a16 TP=2 engine (N2/N3) drives `gen_samples_server.py` over `tools/quant2/corpus_v2_prompts.json`
  plus agentic / tool / Chinese / multilingual prompts, thinking kept; target about 1 M tokens, 60/40 model text vs local raw
  text, disjoint from the KL corpus (6-gram check). Dense model: every token feeds every Hessian, so no rare-expert problem.
- **Rows gate** (rows >= 4 x K): `down` K 22016 needs >= 88k rows: met many times over.
- **Size:** per layer `in`, `out`, `mlp_in` are 7168^2 x 4 B = 205 MB each and `mlp_mid` 22016^2 x 4 B = 1.94 GB: 2.56 GB per layer,
  **about 133 GB** for 52 layers as full fp32 matrices (about 67 GB as triangles; `--dry-run` reports the real figure), written
  to `D:\r4dx-scratch\nanbeige\hessian-v1` (D: 404 GB free). Capture streams one layer's bf16 weights (1.18 GB) on one GPU.

### 7.3 Oracle, mix, container

`quantize-model --arch llama --K 4` and `--K 5` (`--hessian-basis matched`, no rotation), `mix --bpw 4.5` (unit = one layer
role: attn in / attn out / mlp in / mlp down; attention linears are 20 % of the body), then `r4dx-convert --trellis-from <mix>
--trellis-verify full --no-bf16 --lm-head w4a16 --w4a16-group-rule "^lm_head$=32" --ldlq "^lm_head$" --hessian-dir <hessian-v1>`
into `E:\models\r4dx\nanbeige\nanbeige43-33b-trellis-mix45.r4dx`. Oracle cost about 1.2x Qwen 27B's (30.7 B vs about 25 B
quantized weights; UNVERIFIED). Lesson from LFM2: its 4.5 mix failed the gate and the shipped format was a K5 hybrid; the
dense Qwen 27B's 4.5 mix passed. If the mix misses, the next rungs are a 5.0 bpw mix (about 19.2 GB of body, still one card at
about 22.3 GB) and then an empirical per-layer sweep. Tuning rows (`gemm_tuning_table_trellis*.inc`) for N 7168 K 7168, N 1024
K 7168, N 44032 K 7168, N 7168 K 22016; `PlanTrellisM256` checked per shape (a failed plan silently falls back to 64-row
slices). `test_trellis_input` gains K 22016.

## 8. Budgets

| Resource | Budget |
|---|---|
| checkpoint | `E:\models\Nanbeige\Nanbeige4.3-33B`, 66.3 GB (14 shards). **Downloaded by the user** (nothing here downloads) |
| r4dx outputs | `E:\models\r4dx\nanbeige\`: w8a16 34.8 GB, trellis 20.4 GB, oracle K4 + K5 about 15 + 19 GB, `nanbeige-l8` about 14 GB, KL dumps about 10 GB: about 115 GB (E: 652 GB free) |
| scratch | Hessians about 133 GB at `D:\r4dx-scratch\nanbeige\hessian-v1` (D: 404 GB free); keep >= 30 GB free on every drive |
| host RAM 63.5 GiB | ONE big host-memory job at a time (truth CPU run, Hessian capture, oracle, sample merge). Layer streaming keeps peaks under 20 GB |
| VRAM, w8a16 TP=2 | 17.33 GiB weights per rank + KV (3.25 GiB per rank at 32k bf16) + arena about 1 GiB: about 21.6 GiB per rank at 32k. Device 0 drives the desktop (2-4.6 GB): about 6 GB margin at 32k, about 3 GB at 64k |
| VRAM, trellis TP=1 | 18.95 GiB weights + 6.5 GiB KV (32k bf16) + arena about 1.2 GiB = about 26.7 GiB on device 1 |

## 9. Task list

Legend: **[CPU]** authoring and tests need no GPU. **[GPU]** is a command handed to the user, run after approval (every
GPU workload, including every `tests/kernels/*` test). **[CPU-heavy]** is a long CPU/RAM job (one at a time). Build dir
`build\win-hip`.

### Lanes (each its own worktree off `nanbeige`; merge back before the dependent task)

| Lane | Branch | Tasks | Blocks |
|---|---|---|---|
| A refs | `nanbeige-ref` | N0-3..N0-8, N1-9 goldens | gate, goldens |
| B tokenizer + server | `nanbeige-tok` | N0-4, N0-6, N3-1..N3-5 | N3 smoke |
| C config + converter + container | `nanbeige-conv` | N1-1..N1-5 | N1-6 |
| D model + TP | `nanbeige-model` (after C) | N1-6..N1-11 | N2 |
| E quant tools | `nanbeige-quant` | N4-1, N4-2 (CPU authoring) | N4 runs |

### N0: setup and reference data

| Task | What | Kind |
|---|---|---|
| N0-1 | User downloads the checkpoint to `E:\models\Nanbeige\Nanbeige4.3-33B` (66.3 GB) | user |
| N0-2 | `dump_header.py`: every tensor name, shape, dtype; confirms section 4's source names, untied `lm_head`, no `inv_freq` surprises | [CPU] |
| N0-3 | Venv check: `LlamaForCausalLM` import, tiny random-config forward without `trust_remote_code` | [CPU] |
| N0-4 | Read the head of `tokenizer.json` (normalizer, pre-tokenizer, decoder, post-processor, `byte_fallback`, added tokens' flags); record in section 5.1 | [CPU] |
| N0-5 | `survey_small_tensors.py`: norm-weight min / max / mean per layer, embedding and head row norms | [CPU] |
| N0-6 | Write the `-tok` directory, extract `chat_template.jinja`, render in HF and minja for the cases of 5.2; settle the reasoning opener, eos list and polyfill choice | [CPU] |
| N0-7 | `build_chat_nanbeige.py` -> `chat_nanbeige.json`, `tokens_nanbeige.json` | [CPU] |
| N0-8 | fp32 truth and bf16sdpa dumps on the corpus; noise term per group; wall time recorded | [CPU-heavy] |

### N1: model (code CPU, tests GPU)

| Task | What | Kind |
|---|---|---|
| N1-1 | `Arch::kLlama`, detection, `ResolveContainerMaxCtx`; arch tests | [CPU] |
| N1-2 | `llama_config.h` + `Shard`; host test incl. TP=2 shapes | [CPU] |
| N1-3 | `make_tiny_llama.py` (convert_ref and reference copies), tiny checkpoint | [CPU] |
| N1-4 | `llama_layout.{hpp,cpp}`, `RunConvert` dispatch, coverage audit; `test_llama_layout` (bf16, w8a16 on the tiny checkpoint) | [CPU] |
| N1-5 | `LlamaContainer` (bf16, w8a16, trellis loaders; copy-adapt of `Lfm2Container`) | [CPU] |
| N1-6 | `LlamaAttnLayer` (reference kernel), `LlamaDenseMlp`, `LlamaModel` (RunChunk, decode, sampling, checkpoint) | [CPU] |
| N1-7 | `tp::RuleFor` names; `LlamaTpModel` (copy-adapt of `Lfm2TpModel`: rank threads, `TpGroup`, watchdog, warm-up) | [CPU] |
| N1-8 | `LoadLlamaTextModel`, CLI / server wiring, `--layout` checks for the arch | [CPU] |
| N1-9 | Goldens: `layer_golden_llama.py`, `rope_h128`; `tests/model/test_llama_layers.cpp` on `nanbeige-l8` | [CPU] / [GPU] |
| N1-10 | Convert `nanbeige-l8` (bf16, layers 0-7) and the tiny containers | [CPU-heavy] |
| N1-11 | Tiny e2e TP=1 and TP=2-emulate tests; `nanbeige-l8` golden run; TP=1 vs TP=2 KL on `nanbeige-l8` | [GPU] |

### N2: w8a16 TP=2 and its gate

| Task | What | Kind |
|---|---|---|
| N2-1 | Convert `nanbeige43-33b-w8a16-g128.r4dx` (`--layouts w8a16 --no-bf16`) | [CPU-heavy] |
| N2-2 | Load + 64-token greedy on 2 GPUs; VRAM per rank at 32k recorded | [GPU] |
| N2-3 | Teacher-forced logprob dumps (`tool_teacher_forced_logprobs.exe`, arch dispatch only); `kl_report.py --gate llama-fp32` | [GPU] + [CPU] |
| N2-4 | Greedy smoke by teacher-forced truth (6.3) | [GPU] + [CPU-heavy] |
| N2-5 | Decode / prefill speed at 2k, 8k, 32k (recorded, not gated) | [GPU] |

### N3: tokenizer, dialect, server

| Task | What | Kind |
|---|---|---|
| N3-1 | SPM variant for the N0-4 shape; `golden_nanbeige.json`; Gemma golden unchanged | [CPU] |
| N3-2 | `kNanbeige` dialect, auto-detection, arch cross-check, `preserve_thinking` / `tool_call_format` request fields | [CPU] |
| N3-3 | Template-render tests (`tests/server/test_nanbeige_template.cpp`, SHA-pinned) incl. tool history in XML | [CPU] |
| N3-4 | `tools/server/smoke.ps1 -Dialect nanbeige` (reasoning split, thinking off, Chinese, XML tool call round trip, streaming) | [GPU] |
| N3-5 | `docs/nanbeige.md` (run, numbers, limits), README "Other models" row | [CPU] |

### N4: trellis on one card

| Task | What | Kind |
|---|---|---|
| N4-1 | `arch_table.py` `llama` entry; `llama_mix.py`; `trellis_pipeline.ps1`; tiny trellis golden | [CPU] |
| N4-2 | `trellis_import.hpp` name map for the llama arch; refusal of a missing linear | [CPU] |
| N4-3 | Calibration samples (about 1 M tokens) through the w8a16 TP=2 server | [GPU] |
| N4-4 | Hessian capture (about 133 GB) | [GPU] + [CPU-heavy] |
| N4-5 | Oracle K4, K5; mix 4.5 | [GPU] + [CPU-heavy] |
| N4-6 | Convert trellis container; KL gate (increment over w8a16); greedy smoke; speed | [CPU-heavy] + [GPU] |
| N4-7 | Tuning rows for the four body shapes; `PlanTrellisM256` check | [GPU] |

### N5: performance and context (each measured, kept only if it pays)

| Task | What |
|---|---|
| N5-1 | libr4d `h128_gqa7` bf16-KV prefill / split-KV / decode (3.3); unit test vs the reference kernel; ISA guard |
| N5-2 | fp8 KV: the `KVP 0` instantiation, a `kv_calibrate` port for this arch, `--kv-calib` in the converter, KL A/B (doubles the context of both formats) |
| N5-3 | Fused `qkv` linear (launch count), only if a decode profile shows launch-bound layers |
| N5-4 | Embedding table in pinned host memory (gather over PCIe: 14 KB per decode token): frees 2.2 GiB per card |
| N5-5 | Trellis at TP=2 (about 10 GiB per rank; decode estimate about 50 tok/s); needs TP=2 trellis tuning rows |
| N5-6 | int8 prefill GEMM (the Qwen path's) for this arch, if prefill matters; KL A/B like docs/int8-prefill.md |

## 10. Open questions (for the user; none block N0)

1. **Default `max_ctx`.** 32768 proposed for both formats (KV is 208 KiB/token at bf16). 65536 fits TP=2 w8a16 with about 3 GB
   left on device 0, and does not fit trellis TP=1 with bf16 KV.
2. **Trellis gate reference.** The increment over r4dx w8a16 (proposed), or the absolute fp32-truth rule alone?
3. **A zh corpus group.** Proposed (Chinese-first model); two new author-written conversations.
4. **Raw-prompt BOS.** Match HF (`<|im_start|>` prepended to raw `/v1/completions` prompts, proposed) or send raw prompts
   unchanged?
5. **Which format first after N2** if time is short: trellis TP=1 (frees a card) or N5-1 / N5-2 (speed and context at TP=2)?
6. **Default sampling.** Card: temperature 0.6 / top_p 0.95 for chat, 1.0 / 0.95 for agents. Should the dialect supply the chat
   defaults when a request omits them, or keep the server's global defaults?

## 11. UNVERIFIED facts (settle by the named task before dependent code)

| Fact | Settled by |
|---|---|
| Safetensors names match section 4; `lm_head.weight` present; no extra tensors | N0-2 |
| Stock transformers loads it without `trust_remote_code`; 4.51.1 vs 5.18.0 numerics equal for this config | N0-3 |
| `tokenizer.json` normalizer / pre-tokenizer / decoder / post-processor shape and added-token flags | N0-4 |
| Norm weights and activations need no fp32 residual | N0-5, N2-3 |
| Template renders in minja; generation-prompt reasoning opener; eos list; polyfill choice | N0-6 |
| fp32 truth wall time (tens of minutes estimated) | N0-8 |
| The gqa6 prefill template masks rows past `BLOCK_Q x GQA` on store; `O_KWIDE` valid at head_dim 128 bf16 | N5-1 |
| All-reduce cost about 0.9 ms per token at TP=2 | N2-5 |
| Decode speeds (about 30-35 tok/s w8a16 TP=2, about 30 tok/s trellis TP=1) | N2-5, N4-6 |
| Oracle cost about 1.2x Qwen 27B's | N4-5 |
| The 4.5 bpw mix passes the trellis gate | N4-6 |
