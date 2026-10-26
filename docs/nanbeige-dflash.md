# Nanbeige4.3-33B: DFlash2 speculative decoding

Status 2026-10-27: **works at `--tp 2`** (CLI and server; the server smoke passes through DFlash rounds) with the
[ProCreations/Nanbeige4.3-33B-DFlash2](https://huggingface.co/ProCreations/Nanbeige4.3-33B-DFlash2) drafter. Decode is
1.0-2.0x faster depending on the text (math and code most, Chinese least; see "Measured"). Lossless: the target confirms
every token. With `R4DX_LLAMA_VERIFY_ATTN=rows` the output is byte-identical to plain decode (greedy and seeded sampling);
the faster default can flip a greedy token at a near-tie (one coherent word, as GEMM-shape changes do anywhere).

## The drafter

A native `DFlash2DraftModel` (Apache-2.0, ~1.0 B parameters, bf16): 3 non-causal layers, hidden 7168, MLP 8192, 28 q
heads / 4 kv heads (GQA 7), head_dim 128, rope theta 1.2e8, block 8 (one anchor + 7 drafts), mask token 0. It reads the
target residual after layers 1, 25 and 49 (HF `hidden_states[2, 26, 50]`, before the final norm), concatenated (21504
wide) through `fc` + `hidden_norm`, and shares the target's embedding and lm_head. Dynamic conv: 2 taps, groups of 16.
Candidate selector: rank 128, **top-32** (the Ornith / Qwen drafters use top-16).

## Convert

```powershell
hf download ProCreations/Nanbeige4.3-33B-DFlash2 --local-dir E:\models\ProCreations\Nanbeige4.3-33B-DFlash2
# w4a16 (recommended, 0.87 GB), the three mlp.down_proj kept bf16
r4dx-convert --dflash-hf E:\models\ProCreations\Nanbeige4.3-33B-DFlash2 `
    --out E:\models\r4dx\nanbeige\nanbeige43-dflash2-w4a16.r4dx --layout w4a16 --dflash-keep-bf16 down_proj
# bf16: 1.96 GB
r4dx-convert --dflash-hf E:\models\ProCreations\Nanbeige4.3-33B-DFlash2 `
    --out E:\models\r4dx\nanbeige\nanbeige43-dflash2-bf16.r4dx --layout bf16
```

The default target-layer offset (1) is right: the container stores layer-input indices [2, 26, 50].

**`--dflash-keep-bf16 down_proj` is required for a w4a16 Nanbeige drafter.** The w4a16 GEMM converts its activations to
f16; on ordinary prompts something in the drafter overflows it (most likely the SwiGLU products, `mlp.down_proj`'s input,
the one linear input that is not normed), the drafter's logits turn NaN and the round throws (`SelectorWalk: candidate id 2147483647`; 3 of 5
A/B prompts). With the flag the listed linears (HF names ending in one of the comma-separated suffixes) are stored bf16 and
the loader picks each linear's layout from the tensors it finds. Measured: `fc` alone (it reads the raw target residual,
|h| ~32k) does not fix it, `kernel_projection` neither; `down_proj` alone does. The bf16 drafter needs nothing.

## Run

```powershell
r4dx-server --model E:\models\r4dx\nanbeige\nanbeige43-33b-w8a16-g128.r4dx --layout w8a16 --tp 2 --max-ctx 32768 `
    --dflash E:\models\r4dx\nanbeige\nanbeige43-dflash2-w4a16.r4dx --dflash-k 4
r4dx-cli ... --tp 2 --dflash E:\models\r4dx\nanbeige\nanbeige43-dflash2-w4a16.r4dx --dflash-k 4 --stats
```

- `--dflash-k` is 1..7 (block 8; the TP verify window is at most 8 rows). **4 is the best measured setting** (k 7 drafts
  more but accepts the same, k 7 + `--dflash-p-min 0.5` is slower except on Chinese). The drafter is replicated on both
  cards.
- Any temperature: greedy rounds confirm drafts with the target's argmax; sampled rounds draw each verify row from the
  target's own distribution (one rng draw per emitted token) and accept a draft iff it equals the draw (sample-and-match).
  Both are lossless.
- `R4DX_LLAMA_VERIFY_ATTN=rows` runs a verify window's attention once per row at that row's own context length: every
  verify row then equals plain decode bit for bit, at k+1 KV reads per layer instead of one (fine at short context, slower
  than plain decode by 32k). The default `fused` is one launch per layer.

## How it is wired

- `LlamaModel` (src/model/llama_model.cpp) captures the residual entering layers 2, 26 and 50 for every row it runs,
  injects every committed row into the drafter's ring (`DflashDraft::InjectFeatures`), and runs a round as Gemma 4's
  does: draft (`DflashDraft::DraftRound` with the target's embedding and w8a16 lm_head), one verify window of k + 1 rows
  (final norm + lm_head on every row; per-row argmax, or per-row top-64 + logsumexp summaries when sampling), accept,
  inject the committed rows, advance. KV is positional, so rejected rows need no rollback.
- Exact verify: every linear is row-exact between a 1-row and an 8-row call (the w8a16 tuning is the same for M <= 16), but
  the h128_gqa7 decode attention picks its split-KV partition from `max_ctx`, so an 8-row launch partitions each row's keys
  differently from that row's own decode step. `rows` mode runs the decode kernel once per row at that row's context
  length (`LlamaAttnLayer::Forward`'s `seqused_rows`). With the reference attention kernel both arms were already
  identical, which is how the cause was found. Making the fused launch exact would need a ctx-independent split grid in
  libr4d's decode kernel (not done).
- `LlamaTpModel`: the drafter is loaded on both ranks (one shared host copy of the selector codebooks); each rank
  top-k's its own vocab shard and `tp::MergeTopK` merges them; verify argmaxes / summaries are merged across the shards.
  The rounds are compared across ranks like every other command.
- Drafter changes for this checkpoint: `selector_top_k` 32 (the first 32 of `r4dx_topk_lse_f32_ws`'s exact sorted top-64,
  same total order as the top-16 kernel), and `r4dx_dflash_attn_bf16` for GQA ratios above 4 (a kv head's q heads run in
  groups of 4 over `blockIdx.z`; ratio <= 4 is unchanged).
- The drafter's own attention is windowed at 2048 positions (the kernel's LDS row); the checkpoint was trained with full
  attention, so acceptance past ~2k tokens of context may be lower than the card's.

## Checks

| check | result |
|---|---|
| `nanbeige_dflash2_golden` (the HF `DFlash2DraftModel` on CPU fp32 vs r4dx, same random features / embedding) | bf16: final hidden RelL2 1.1e-2 (HF's own bf16-vs-fp32: 2.4e-2), top-32 overlap 219/224, top-1 7/7, greedy path 7/7. w4a16: RelL2 0.21, path 5/7 (reported) |
| `test_dflash_attn` [1c] (GQA 28/4 vs an fp64 reference, n_injected up to 5000) | norm_rel <= 1.7e-3, PASS |
| greedy A/B vs plain (`tools\nanbeige\dflash_ab.ps1`, 5 prompts x 512 tokens, thinking on), `rows` | byte-identical on every prompt, every k / p_min, both drafters |
| the same, `fused` | short prompts: diverges from plain on all 5, first at a near-tie word (e.g. "appended" -> "simply appended") and coherent after; 8k / 32k prompts: identical |
| sampled (temperature 0.6, top-p 0.95, seed 1234) DFlash vs plain, k 4, `rows` | byte-identical (code, English) |
| `tools\nanbeige\smoke_server.ps1 -Dflash ... -DflashK 4` | all 24 checks pass |

## Measured

2026-10-27, R9700 x2, `--tp 2`, w8a16 target, w4a16 drafter (`down_proj` bf16), greedy, thinking on, 512 tokens, idle
machine. Plain decode is 32.0 tok/s on these prompts. `rows` verify unless noted:

| prompt | k 7 | k 7, p_min 0.5 | **k 4** | k 7 fused | tokens / round (k 4) |
|---|--:|--:|--:|--:|--:|
| math (word problem) | 63.6 (1.98x) | 55.7 | **65.3 (2.03x)** | 57.4 | 2.49 |
| code (Python + tests) | 46.6 (1.45x) | 40.6 | **48.2 (1.50x)** | 51.2 | 1.83 |
| English explanation | 43.5 (1.36x) | 42.2 | **44.6 (1.39x)** | 47.4 | 1.70 |
| Thai | 35.2 (1.10x) | 33.8 | **36.6 (1.14x)** | 39.1 | 1.40 |
| Chinese | 32.3 (1.00x) | 34.5 | **33.6 (1.05x)** | 34.7 | 1.27 |

The bf16 drafter accepts the same (k 7: math 2.51, code 1.84, English 1.74, Thai 1.40, Chinese 1.28 tokens / round) and
is 5-10% slower (2 GB of weights per draft instead of 0.9). Sampled (temperature 0.6, k 4, `rows`): code 48.4 tok/s
(1.56x), English 43.2 (1.39x).

Long context (one prompt, 256 greedy tokens, k 4):

| context | plain | DFlash `rows` | DFlash `fused` (default) | tokens / round |
|--:|--:|--:|--:|--:|
| 8,346 | 30.6 | 33.6 (1.10x) | 36.5 (1.19x) | 1.42 |
| 31,682 | 26.7 | 25.9 (0.97x) | 32.7 (1.23x) | 1.45 |

The model card's own numbers (SGLang, RTX PRO 6000, BF16 target) are 1.76x greedy on average, 1.15x (Chinese) to 2.7x
(agentic); r4dx lands in the same range. A round costs more than a decode step (Chinese, k 7: 1.28 tokens per round at
32.3 tok/s = 40 ms per round against 31 ms per plain step: the draft plus the wider verify), so a prompt whose drafts are
rarely accepted gains nothing; at k 4 nothing measured was slower than plain except `rows` at 32k.
