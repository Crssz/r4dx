# LFM2.5-24B-A2B (`lfm2_moe`) semantics -- verified facts with sources

Written 2026-10-14 by lane A (docs/lfm2-plan.md L0-5). Every fact below was read from a local file; the source is given as
`file:line`. Abbreviations: `HF` = `D:\venvs\r4dx-gemma-ref\Lib\site-packages\transformers\models\lfm2_moe\modeling_lfm2_moe.py`
(transformers 5.18.0; the checkpoint was saved with 5.12.1), `cfg` = `E:\models\LiquidAI\LFM2.5-24B-A2B\config.json`,
`gen` = `generation_config.json`, `tpl` = `chat_template.jinja`, `hdr` = `tools/reference/lfm2/tensor_names.json`
(safetensors headers, `tools/reference/lfm2/dump_header.py`), `survey` = `tools/reference/lfm2/survey_small_tensors.json`
(`survey_small_tensors.py`), `gold` = `tools/reference/golden_out/lfm2/{manifest,router_manifest}.json` (git-ignored, regenerate with
`layer_golden_lfm2.py` / `router_golden.py`). The tokenizer section belongs to lane B (L0-4) and is appended below by that lane.

## 1. Architecture

| # | Fact | Source |
|---|---|---|
| 1 | model_type `lfm2_moe`, class `Lfm2MoeForCausalLM`; 40 layers, hidden 2048, vocab 128000, max positions 131072 | cfg:2-3, 10, 55, 64, 76 |
| 2 | `layer_types`: `full_attention` at layers 2, 6, 10, ..., 38 (10 layers); `conv` elsewhere (30 layers) | cfg:13-54 |
| 3 | attention: 32 q heads, 8 kv heads (GQA 4), NO `head_dim` in the config -> `hidden/heads` = 64; scale `head_dim^-0.5`; q/k/v/out projections without bias | cfg:60,65; HF:302-309 |
| 4 | per-head QK RMSNorm (`q_layernorm`, `k_layernorm`, size 64, eps `norm_eps`) is applied to the q/k projections BEFORE rope | HF:310-311, 324-329 |
| 5 | rope: `rotate_half` (pairs `(i, i+32)`), full rotary (64 dims), `rope_theta` 1e7, `rope_type` default (NO scaling); cos/sin are computed in fp32 and rounded to the activation dtype | cfg:67-70; HF:102-123, 224-254 |
| 6 | RMSNorm: `weight * (x * rsqrt(mean(x^2) + eps)).to(input_dtype)` (plain weight, not `1 + w`; fp32 compute, bf16 rounding BEFORE the weight multiply), `norm_eps` 1e-5 | HF:61-66; cfg:58 |
| 7 | decoder layer: `h = h + mixer(operator_norm(h))`; `h = h + ffn(ffn_norm(h))`; mixer = attention (layer type `full_attention`) or ShortConv (`conv`) | HF:503-523 |
| 8 | after the 40th layer `embedding_norm` (the FINAL norm, misleading name); lm_head = tied embedding table (no `lm_head` tensor), no logit softcap, no embedding scale | HF:564, 570, 629, 639, 648; cfg:72; hdr (`lm_head_present: false`) |
| 9 | ShortConv: `in_proj` H -> 3H (no bias, `conv_bias` false); chunk order **B, C, x**; `h = B * x`; depthwise causal `Conv1d` kernel `conv_L_cache` = **3**, groups = H, weight `[H, 1, 3]`, no bias, NO activation; `y = C * conv(h)`; `out_proj` H -> H | cfg:6-7; HF:421-431, 445-447, 463-474; hdr (`conv.conv.weight [2048,1,3]`) |
| 10 | the conv is `F.conv1d(h, weight, padding=K-1, groups=H)[:, :, :seq_len]` (taps oldest..newest), the cached decode path concatenates the last `K-1` = 2 `h` values (the conv state) | HF:385-405, 365-382 |
| 11 | dense FFN in layers 0-1 (`layer_idx < num_dense_layers`): `w2(silu(w1 x) * w3 x)`, I = 11776 | cfg:12, 61; HF:126-136, 486-490 |
| 12 | MoE FFN in layers 2-39: 64 experts, top-4, `moe_intermediate_size` 1536, NO shared expert; experts are `gate_up_proj [E, 2I, H]` (first half = gate = `w1`, second = up = `w3`) and `down_proj [E, H, I]` (= `w2`), `silu(gate) * up`, per-expert outputs scaled by the routing weight and accumulated with `index_add_` in the module dtype, in expert order | cfg:57,62-63; HF:139-176 |
| 13 | router: `logits = x @ W^T` (`gate.weight [64, 2048]`), `s = sigmoid(logits)`; selection = top-4 of `s + expert_bias` (`expert_bias` is an fp32 buffer `[64]`, `use_expert_bias` true); the weights are the UNBIASED `s` of the selected experts, divided by `sum + 1e-6` (`norm_topk_prob` true), times `routed_scaling_factor` (1.0) | cfg:59,71,75; HF:179-221 (bias buffer 213) |
| 14 | HF-bf16 router precision: bf16 matmul and bf16 sigmoid; `s + expert_bias` promotes to fp32 (bias is fp32) for the top-k; the weights are bf16 | HF:191-196 |
| 15 | ids: bos 124894 `<|startoftext|>`, eos 124900 `<|im_end|>` (`gen` eos list `[124900]`), pad 124893 | cfg:5,9,66; gen:3-5 |
| 16 | generation defaults: `do_sample` true, temperature 0.1, top_k 50, repetition_penalty 1.05 | gen:12-15 |
| 17 | dtype bf16 except the 38 `expert_bias` tensors (F32 `[64]`); `tie_word_embeddings` true | cfg:8,72; hdr (`dtypes {"BF16": 7572, "F32": 38}`) |

## 2. Checkpoint layout (header dump, `dump_header.py`; closes plan U4)

- 10 shards `model-0000N-of-00010.safetensors`, 47,944,143,456 bytes of files (tensor data + headers), 7610 tensors, index and headers agree.
- Top level: `model.embed_tokens.weight [128000, 2048]` and `model.embedding_norm.weight [2048]` only; no `lm_head`.
- Per layer `model.layers.N.`: `operator_norm.weight`, `ffn_norm.weight`; conv layers `conv.in_proj.weight [6144, 2048]`,
  `conv.conv.weight [2048, 1, 3]`, `conv.out_proj.weight [2048, 2048]`; attention layers `self_attn.{q_proj [2048,2048], k_proj [512,2048],
  v_proj [512,2048], out_proj [2048,2048]}.weight`, `self_attn.{q,k}_layernorm.weight [64]`; dense layers 0-1 `feed_forward.{w1,w3} [11776,2048]`,
  `feed_forward.w2 [2048,11776]`; MoE layers 2-39 `feed_forward.gate.weight [64,2048]`, `feed_forward.expert_bias [64]` (F32),
  `feed_forward.experts.{0..63}.{w1,w3} [1536,2048]` and `.w2 [2048,1536]` (**per-expert 2-D tensors, NOT fused**: 7296 expert tensors).
- `lfm2.arch.check_against_header` reports 0 problems (no missing, no unexpected, all shapes and the F32 dtype of `expert_bias` as expected).
- Consequences: the converter's fused-key branch is not needed for this checkpoint (the per-expert branch concatenates `w1` then `w3`);
  HF loads these keys into the fused modules itself (verified: `Lfm2MoeForCausalLM.from_pretrained` on a tiny checkpoint in this format
  is bit-equal to `lfm2.ref`, `ref.selftest_tiny`).
- Vocab padding: the embedding has 128000 rows; the tokenizer's highest id is 125016 (lane B, L0-4), so logit columns >= **125017** are
  padding. The reference dumps (`full_logits_lfm2_cpu.py`) log-softmax over the real columns and write the clamp (-1e4) in the padding
  columns; r4dx must mask them the same way (decision 2026-10-14).

## 3. Norm-weight survey (`survey_small_tensors.py`; closes plan U7, decides plan 7.1)

All 40 `operator_norm`, 40 `ffn_norm`, the final norm and the 10+10 QK norms (bf16 weights, plain scale):

| tensor | max abs over layers (range of per-layer max) | min abs |
|---|---|---|
| `operator_norm` | per-layer max 0.046 .. 0.797 (largest L37 0.797, L36 0.754, L34 0.672) | per-layer min 0.0056 .. 0.289 |
| `ffn_norm` | per-layer max 0.137 .. 1.148 (largest L38 1.148, L37 1.133, L36 1.031) | per-layer min 4.0e-5 (L0) .. 0.244 |
| `q_layernorm` / `k_layernorm` | per-layer max 1.875 .. 2.438 / 1.984 .. 2.422 | -- |
| final norm (`embedding_norm`) | max 3.297, mean 1.876, std 0.163 | min 0.053 |

Reading for the rotation question: **no norm scale is large** (the pre-mixer and pre-FFN weights are all below 1.15; Gemma's failure mode was
large pre-norm scales), so folding `diag(w_norm)` into the following linears would be benign. There is still no evidence that a rotation is
needed, so the decision stays **unrotated trellis** (plan 7.1, open question 11): revisit only if the trellis gate misses. Side note: a
near-zero channel scale (L0 `ffn_norm` min 4e-5) means those channels contribute almost nothing to the FFN input.

## 4. Router and expert_bias statistics

Static (all 38 MoE layers, `survey_small_tensors.py`):
- `expert_bias`: per layer mean exactly 0.0 (the aux-loss-free balancing keeps it zero-mean); per-layer |bias| max 0.071 .. 0.874, spread (max - min)
  0.093 .. 0.936; the largest at L37 (0.874), L29 (0.738), L25 (0.537); layer 2: |bias| max 0.172, spread 0.219.
- router `gate.weight`: values within +-0.71, mean row L2 norm 0.98 .. 3.30 across layers (layer 2: 2.84, rows 2.24 .. 3.61).

On real data (`router_golden.py`: the REAL `ffn_norm` outputs of the first 256 tokens of the first chat-corpus segment through layers 0..3, bf16 stack):

| | layer 2 | layer 3 |
|---|---|---|
| margin between the 4th and 5th BIASED score: min / p01 / p05 / median / p95 | 2.6e-5 / 1.9e-4 / 9.1e-4 / 1.07e-2 / 6.2e-2 | 1.3e-5 / 5.8e-5 / 6.8e-4 / 1.11e-2 / 8.1e-2 |
| rows with margin < 1e-3 / < 1e-2 (of 256) | 15 / 120 | 18 / 115 |
| rows where HF-bf16 routing selects a different expert set than fp32 routing | 4 | 6 |
| rows where the bias changes the selected set vs the unbiased top-4 | 186 | 162 |
| mean selected sigmoid / mean sigmoid over all experts | 0.280 / 0.098 | 0.299 / 0.080 |
| mean sum of the 4 selected sigmoids (before normalization) | 1.12 | 1.19 |
| experts hit by 256 tokens | 56 | 63 |

So about 1.5-2.5 % of rows route differently in bf16 than in fp32 (the router-flip noise source of plan R3) and about 5-7 % of rows have a
4th/5th margin below 1e-3; the expert bias matters for the majority of rows (selection is not the unbiased top-4). The one-pass layer goldens
(`gold`, 68 rows) show the same: 2-3 flipped rows per MoE layer and an `ffn_out` bf16-vs-fp32 relative error of 0.12-0.14 (dominated by the
flipped rows) against a `layer_out` relative error of 0.002.

## 5. Numerics notes for the references

- `lfm2.ref.Lfm2Reference` is the HF layer loop (bit-equal to `Lfm2MoeForCausalLM`, `ref.selftest_tiny`): **truth** = bf16 weights widened exactly, fp32
  activations / rope tables / router / sigmoid / top-k / expert math / final norm / head, attention in query chunks of 512 (exactly equal to HF eager,
  RAM-bounded); **bf16sdpa** = stock HF numerics (the `expert_bias` buffer stays fp32 as in HF).
- ShortConv kernel contract (`layer_golden_lfm2.py`, component `shortconv_state`): `h_t = bf16(B_t * x_t)`; `acc = w0*h_{t-2} + w1*h_{t-1} + w2*h_t`
  in fp32 over bf16 operands, rounded once; `y_t = bf16(C_t * conv_t)`; state = the last two `h` (zeros at start). Measured on real layer-0 weights:
  the formula equals HF's `F.conv1d` bit for bit (max abs diff 0.0), one pass == chunked (256 + 44) == chunked + 4 decode steps, final state equal.
- Per-layer bf16-vs-fp32 floors (68 rows, real weights): `layer_out` relative Frobenius error 0.0017 (layer 0), 0.0021 (layer 2), 0.0016 (layer 3);
  `q/k` after rope 0.004; `conv_y` 0.006; suggested C++ tolerance = 4x the per-tensor floor (`manifest.json`, `fp32_floor`).
- Router contract (`lfm2.ref.manual_router`): fp32 sigmoid of the bf16-operand logits, top-4 of `s + bias` with ties to the LOWER index, weights = unbiased `s`
  normalized by `sum + 1e-6`; equals the HF router run in fp32 on all goldens; ids stored in descending-score slot order.

## 6. Chat template (read from `tpl`; rendering is validated by lane B, L0-6)

- Starts with `bos_token` (`<|startoftext|>`) (tpl:1); `<|im_start|>role\n ... <|im_end|>\n` turns (tpl:76, 84, 114, 120).
- A leading system message is merged with `"List of tools: [" + tojson(tool), ... + "]"` into ONE system turn (tpl:55-77).
- Assistant turns keep `<think>{reasoning}</think>` only for assistant messages AFTER the last user message (or with `preserve_thinking`), taken from
  `message.thinking | reasoning | reasoning_content`; otherwise it is dropped, and a `</think>` found in the content is stripped to the text after it (tpl:83-101).
- Tool calls render as `<|tool_call_start|>[name(a='x', b=1), ...]<|tool_call_end|>` (strings single-quoted with `\\ \' \n \r` escaped; mappings and lists via
  `tojson`; other values via `string`); a JSON-STRING `arguments` value raises (tpl:4-12, 36-53). Role `tool` is a plain turn (tpl:116-121).
- `add_generation_prompt` appends `<|im_start|>assistant\n<think>` -- the model always thinks, there is no thinking switch (tpl:123-125).
- `{% generation %}` markers (tpl:86, 115) do not change the output; `CONTINUE_FINAL_MESSAGE_TAG ` handling (tpl:93-113).
- KL corpus consequence (`build_chat_lfm2.py`): every scored span is `{thought}</think>{answer}<|im_end|>` after the generation prompt; ids equal
  `AutoTokenizer.apply_chat_template(tokenize=True)` and the raw `tokenizers` encoding for all 12 segments.

## 7. Tokenizer (lane B, L0-4)

(Appended by lane B. Facts already settled in the shared lane notes: byte-level BPE with the Llama-3-style split regex, no normalizer, no automatic BOS,
base vocab 124893 + 124 added tokens, `<think>`/`</think>` single added tokens 124901/124902 (not `special`), `<|tool_call_start|>`/`<|tool_call_end|>`
124905/124906 (not `special`), max id 125016.)

## 8. Measured reference cost (CPU, this machine; shared with other lanes)

From a probe over layers 0-3 with all 12 chat segments (5022 tokens), 8 threads:
- fp32 truth: dense layer 11-22 s, MoE layer 8-22 s per layer per 5k tokens (about 60 GFLOP/s fp32; the spread is CPU contention from other jobs);
  bf16sdpa: 0.5-1.9 s per layer (AMX/bf16 GEMMs run at about 1.4 TFLOP/s). Peak RSS of the truth with an MoE layer: 3.6 GiB (raw non-mmap reads), 5.6 GiB with
  the 6144-token segment (activations + attention chunks).
- Full corpus (chat 5022 + raw 4 x 1024 + long 6144 = about 15.3k tokens): about 75 TFLOP for the truth -> 20-30 min quiet, up to an hour under contention;
  bf16sdpa 3-10 min; the fp32 head is about 8 TFLOP (2-3 min).
