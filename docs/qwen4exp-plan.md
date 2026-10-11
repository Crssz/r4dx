# Qwen3.8-Flash-Next (`qwen4_exp`) support for r4dx -- design and task plan

Written 2026-10-10 from a grilling session with the user. This is a **design only**: no source file was
edited, no GPU workload was run, nothing was downloaded and no package was installed while writing it.
Where it says MUST, do exactly that; anything not covered is a question for the user. Items tagged
**VERIFY** were read from the HF modeling code but not yet checked against a golden; the task that
settles each one is named.

## Summary

r4dx runs two architectures today, the Qwen3.5 hybrid (`Arch::kQwen35`) and Gemma 4 (`Arch::kGemma4`),
dispatched by `src/model/arch.h`. This plan adds a third, `qwen4_exp`, for
`ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ`: a 125B-parameter MoE (6B active per token) with
Gated DeltaNet + Qwen Sparse Attention (QSA), a 4-stream gated residual, a 95 GiB hashed n-gram embedding
(PLE) and 512 experts, top-10. The AWQ int4 weights are dequantized and requantized to trellis at about
3.3 bits/weight so the whole text model fits on the two R9700s, split by layer. The n-gram table stays
in BF16 on NVMe and is read row by row.

Milestone 1 is text-only chat through `r4dx-cli` and `r4dx-server`, with QSA, prefix reuse, thinking and
tool calls, gated on correctness only. MTP, vision and speed work are later milestones.

The largest new pieces are: the trellis KB = 3 decode path, the grouped MoE trellis GEMM, the QSA
indexer and sparse attention, the n-gram sidecar reader, and a decode that runs across both cards.
Every GPU step is a command handed to the user.

## 1. Agreed decisions

| # | Topic | Decision |
|---|---|---|
| Q1 | Milestone 1 scope | Text-only via `r4dx-cli` and `r4dx-server`: thinking on/off and `reasoning_effort`, tool calls, prefix reuse (Q19). MTP, vision and speed are later milestones. |
| Q2 | Integration | Third `Arch` (`kQwen4Exp`, on-disk name `qwen4_exp`) following `docs/gemma4-plan.md`: own config, loader and model behind `TextModel`; shared tokenizer, server, sampling, paged KV and trellis GEMM. |
| Q3 | Source weights | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ` at revision `598afc96128f76a67117caf878f538d34f6f1e02`. |
| Q3b | Fitting in VRAM | Dequantize AWQ int4 to BF16, then trellis-quantize to at most 3.5 bits/weight for the experts. |
| Q4, Q9 | Placement | All text weights resident on both R9700s, split by layer. The desktop card (device 0) holds half the model while it is loaded. |
| Q5, Q15 | N-gram table | Exact BF16, repacked into one r4dx sidecar file, read row by row with unbuffered I/O. |
| Q6, Q11 | Reference | HF `qwen4_exp`, one layer at a time on device 1, on the AWQ weights dequantized to BF16. |
| Q7 | Attention | QSA is implemented in milestone 1 (dense is exact only up to 2051 tokens, 2.4). |
| Q8 | Other models | Qwen3.8-27B and Gemma 4 paths stay byte-identical; their tests and KL gates do not change. |
| Q10, Q13 | Gate | Correctness only: mean KL <= 0.02 and top-1 agreement >= 93% against the reference, fp8 KV included, plus a server smoke. Speeds are recorded, not gated. |
| Q12, Q16 | Hessians | Captured from the reference over the hessian-v2 corpus plus a slice of `ukisai/Qwen3.8-27B-multi-turn-agent-sft`. An expert whose routed token count is below the linear's input dimension (2560 gate/up, 640 down) falls back to a damped diagonal Hessian; the count is logged per expert. |
| Q14 | Budget | At most 26 GiB of weights per card. Bits per tensor class chosen by a KL sweep with the existing mix-rule machinery. |
| Q17 | Tokenizer | Qwen3.5 split pattern (`\p{M}` absorbed), r4dx's canonical default (5.1). |
| Q18 | Disk | Everything under `E:\models\r4dx` (565 GiB free on 2026-10-10). |
| Q20 | Card roles | Bytes-balanced layer boundary. Device 0: embedding gather, n-gram reads, PLE, first layers. Device 1: last layers, final mixer, lm_head, sampling. |
| Q21 | Tracking | This document only; no GitHub issues. |
| Q22 | Validated context | KL gate on the canon corpus plus teacher-forced probes at 8k and 32k. Longer contexts run but are unvalidated in milestone 1. |
| Q23 | QSA ties | Ties at the top-k cut are frequent (a block whose 4 head dot products are all <= 0 scores exactly 0; M0-6 saw 125 tied queries of 3000), and CPU `torch.topk` picks among them in an implementation-defined order. r4dx and the gate reference both break ties by a fixed rule: prefer the most recent blocks. KL between HF's own torch choice and the rule is measured and reported next to the gate as a noise floor. |

## 2. Architecture facts

Sources: the checkpoint's `config.json`, safetensors headers of all 19 main shards and `model_mtp`
(read by HTTP range request on 2026-10-10), and `transformers/models/qwen4_exp/modular_qwen4_exp.py`
on `main` (line numbers below refer to that file).

### 2.1 Shapes

- hidden 2560, 48 layers, vocab 248320, `rms_norm_eps` 1e-6, `tie_word_embeddings` false, no final
  norm (the text model ends in `hyper_connection_mixer`, `:1027`).
- Layer pattern: every 4th layer (3, 7, ..., 47) is full attention, the other 36 are GDN.
- GDN: 16 key heads and 48 value heads of dim 128, conv kernel 4. `in_proj_qkv` 10240 x 2560,
  `in_proj_z` 6144 x 2560, `out_proj` 2560 x 6144, `in_proj_a` / `in_proj_b` 48 x 2560. The gated norm
  uses **sigmoid** (`output_gate_type: "sigmoid"`, `:322-327`), not the 27B's silu.
- Full attention: 24 query heads, 2 KV heads, head_dim 256; `q_proj` 12288 x 2560 (query + output gate,
  as Qwen3.5), `k_proj` / `v_proj` 512 x 2560, `o_proj` 2560 x 6144. Partial rotary 0.25, rope theta
  1e7, interleaved mrope sections [11, 11, 10].
- MoE: 512 experts, top-10, `moe_intermediate_size` 640, `norm_topk_prob` true, plus a shared expert of
  intermediate 640 with a sigmoid `shared_expert_gate`. In the AWQ checkpoint the experts are per-expert
  linears (`mlp.experts.{e}.{gate,up,down}_proj`), not the fused 3D tensors HF uses at runtime.
  **VERIFY** (M1-4): router softmax in fp32 then top-k then renormalize, as Qwen3-Next.
- Every norm is `(1 + weight)`-centred RMSNorm (`:864-866`).

### 2.2 Gated residual (hyper-connections)

The residual is 4 streams x 2560 = 10240 wide (`hidden_states.repeat(1, 1, 4)`, `:1014`). Each layer
has two `GatedResidual` blocks (`:531-559`), one around the mixer and one around the MoE:

```
n   = grouped_rmsnorm(h, group 2560) * (1 + hc_norm.w)          # h: [T, 10240]
m   = sigmoid(up(silu(down(n) / 4)))                             # down 10240->320, up 320->10240
x   = mean over 4 streams of (m * n)                             # [T, 2560], the block input
inj = 2 * sigmoid(block_inject(n) / 4)                           # [T, 4]
h   = h + broadcast(block_out) * inj  per stream                 # after the attention / MoE block
```

The final `hyper_connection_mixer` is the same block without `block_inject` and returns `x`.

### 2.3 PLE and the n-gram embedding

- Only one PLE layer: `ple_layer_ids` = [2] is 1-indexed, so it runs at layer index 1 (`:796`), before
  that layer's attention gated residual.
- 3-gram hashing over the token history with EOS segmentation: 2 n-gram orders (2 and 3) x 8 heads = 16
  heads, each of dim 2560 / 16 = 160 (`:608-708`). Each head has its own prime-sized vocabulary after
  20,000,000; head offsets concatenate them. Ids are `(t0*m0 xor t1*m1 [xor t2*m2]) mod prime_h`. The
  multipliers are bounded by `max_long // vocab`, so no product overflows int64 and every id is
  non-negative.
- Tokens before the most recent EOS (`eos_token_id` 248044) are replaced by EOS in the history
  (`_shift_right_ignore_eos`, `:643-657`); the history carries the last 2 ids across calls.
- Table: 128 shards of 2,500,012 rows x 160 BF16 = 320,001,536 rows, **95.37 GiB**.
  Per token: 16 rows of 320 bytes.
- PLE block (`:711-783`): `key_proj` 10240 x 2560, `value_proj` 2560 x 2560, three grouped norms, a
  depthwise conv1d over 10240 channels with kernel 4 and **dilation 3** (state 9 steps), silu. The output
  is added to the residual streams.

### 2.4 QSA (`:368-508`)

For each query, the indexer:
1. projects `index_qk_proj` (640 x 2560) into 4 query heads and 1 key head of dim 128; q gets
   `q_layernorm` and rope at the query position;
2. groups the visible keys into complete blocks of 4 consecutive tokens, mean-pools each block's raw
   keys in fp32, casts to bf16, applies `k_layernorm`, then rope at the block's first position;
3. scores each block as `sum over heads of relu(q_h . k_block) / sqrt(128)`;
4. keeps the top `2048 / 4 = 512` blocks plus the incomplete tail (at most 3 tokens);
5. the attention then runs over the selected tokens only (the indexer's mask is ANDed with causal).

Consequence: dense attention equals QSA only while a query sees at most 512 x 4 + 3 = **2051 tokens**.
The pooled key of a block never changes once the block is complete, so caching the pooled, normed,
roped block keys (192 bytes per token over 12 layers) is exact. The raw keys of the incomplete block are
buffered until it completes.

### 2.5 Checkpoint layout (AWQ)

- compressed-tensors `pack-quantized`: int4, group 128, asymmetric, `memoryless_minmax`. Per linear:
  `weight_packed` int32 [N, K/8], `weight_scale` bf16 [N, K/128], `weight_zero_point` int32
  [N/8, K/128], `weight_shape`. `W = (q - zp) * scale`.
- Kept in BF16 (the `ignore` list): router, shared experts and their gate, `in_proj_a/b`, every
  hyper-connection weight, the indexer, PLE projections, `lm_head`, `embed_tokens`, vision.
- On-disk text bytes: experts 58.4 GiB, hyper-connections 1.19, embed 1.18, lm_head 1.18, GDN 1.02,
  shared experts 0.44, attention 0.33, router 0.12. Separate: MTP 4.86 GiB, vision 0.84 GiB, n-gram
  table about 95 GiB (4 of its shards sit in the main files).

## 3. Budget and placement

### 3.1 Weight bytes on the GPUs (estimate; M1-14 measures the real container)

| Part | Format | GiB |
|---|---|--:|
| Experts, 120.8B params | trellis KB 3 gate/up, KB 4 down (3.33 bits/weight) + sign vectors | ~47.3 |
| GDN + attention linears, 2.67B params | trellis KB 4 | ~1.25 |
| Hyper-connections | bf16 | 1.19 |
| Shared experts | bf16 | 0.44 |
| lm_head | w4a16 group 32 | ~0.35 |
| Router, indexer, `in_proj_a/b`, PLE projections, norms | bf16 | ~0.23 |
| **Total** | | **~50.8** (26 GiB/card cap = 52) |

`embed_tokens` stays host-resident as for the 27B (`Container::EmbedTokensHost`). The 1.2 GiB of slack
goes to KB 4 promotions chosen by the sweep (M1-13). If the sweep cannot reach the gate inside 52 GiB,
stop and ask; do not raise the cap.

### 3.2 Runtime memory

- fp8 KV: 12 layers x 2 heads x 256 x 2 = 12 KiB/token, **3.0 GiB at 262144** (split 6/6 layers).
- QSA block keys: 192 B/token, 48 MiB at 262144.
- GDN state: 36 x 48 x 128 x 128 fp32 = 113 MB, plus conv states. PLE conv state 9 x 10240.
- Prefix-reuse snapshots hold the GDN, conv, PLE and n-gram states per kept prefix.

### 3.3 Layer split

The boundary is chosen at load so both cards hold the same weight bytes, with device 1 also carrying
lm_head and the final mixer. Per token, the hand-off is the 4-stream hidden state through pinned host
memory (no peer-to-peer path): 40 KB at decode, 10 MB per 256-row prefill chunk. Decode runs the two
halves in sequence; overlap belongs to the speed milestone.

## 4. Design

### 4.1 Arch, config, dispatch

- `Arch::kQwen4Exp`, `ArchName` = `"qwen4_exp"`. `DetectArchFromMetadata` maps `model_arch` =
  `qwen4_exp` (and `model_config.model_type` = `qwen4_exp`) to it. The `kQwen35` fallback for containers
  without `model_arch` is unchanged.
- `src/model/qwen4exp_config.h`: every key in 2.1-2.4 parsed from `text_config`, with asserts on the
  layer pattern, `ple_layer_ids`, `hc_count` 4, `indexer_kv_heads` 1 and `ngram_size` 3.
- `LoadTextModel` dispatches to a new `LoadQwen4ExpTextModel`. Nothing in the Qwen3.5 or Gemma branches
  changes.

### 4.2 Converter and container

- `r4dx-convert` gains an AWQ input path: dequantize pack-quantized int4 to bf16 per tensor on the CPU
  (2.5), then go through the existing body path (trellis import from a mix manifest, w4a16 lm_head,
  bf16 keeps). The dequant MUST match compressed-tensors bit for bit (M0-4).
- Container `__metadata__`: `"model_arch": "qwen4_exp"`, verbatim `model_config`, and an `ngram_sidecar`
  record (file name, row count, row bytes, head offsets, head primes, multipliers, sha256).
- Tensor names follow `text.layers.{i}.` as for Qwen; experts as `text.layers.{i}.moe.experts.{e}.{gate,up,down}`.
- **N-gram sidecar** (`*.r4dx-ngram`): a 4096-byte header (magic, version, rows, row bytes 320,
  head offsets and primes, multipliers, eos id, sha256 of the body), then rows in global id order. Built
  by streaming the 128 HF shards in order, so the shards can be deleted after the checksum verifies.

### 4.3 Trellis KB = 3

The trellis GEMM decodes KB 4 and 5 only (`docs/trellis-kernel.md` 4.4). Add KB 3 with the same
structure: fetch and extract change, the WMMA tail does not. `trellis_quant.py` already takes K as a
parameter (EXL3 heritage); confirm in M1-10. Tests mirror the KB 4/5 cases in `test_trellis_gemm.py`
and `test_trellis_linear`.

### 4.4 Quantization pipeline

Per layer, so the Hessians never all exist at once (a full set would be about 330 GB for gate/up alone):

1. The streaming reference runs the whole corpus through layer L and stores layer L's outputs
   (20 KB/token for the 4-stream state).
2. Accumulate Hessians for that layer's linears; expert Hessians count routed tokens only. gate and up
   share one Hessian per expert.
3. Apply the fallback rule (Q16) and log `tokens_routed` per expert.
4. Run the trellis oracle for layer L at KB 3 and KB 4, write its manifest entries, delete the Hessians.

Corpus target: about 1M tokens (hessian-v2 + agent SFT slice), so an average expert sees about 20k
tokens. M1-11 reports the share of experts that fell back. Allocation (M1-13) reuses
`trellis_quant.allocate` with the qgroups extended to expert gate+up and down, priced in nats/GiB.

### 4.5 Runtime: layer forward

- Gated residual: one fused kernel for grouped norm + mix-in (down GEMV, silu, up GEMV, sigmoid, mean)
  and one for inject + add. Residual streams are fp32 (the Gemma `fp32resid` lesson); **VERIFY** in
  M1-6 against the bf16 HF reference with a layer golden.
- GDN: reuse the 27B GDN layer with hidden 2560 and the sigmoid gated norm as a new epilogue option.
  The 27B path keeps its silu bytes.
- Full attention: reuse the Qwen3.5 attention pieces (fused q/gate split, q/k norm, mrope, fp8 KV write,
  output gate) with 24 / 2 heads, and replace the attention call with QSA (4.6).
- MoE (4.7) and the shared expert, router in fp32.
- PLE at layer 1 (4.8).

### 4.6 QSA kernels

- **Block-key cache**: when a 4-token block completes, pool, norm and rope it into the layer's
  block-key buffer (bf16 [blocks, 128]). Raw keys of the incomplete block live in a 3-row buffer.
- **Scoring + top-k**: per query, 4 heads x all complete blocks, relu-sum, then top-512 (a radix select
  over up to 65536 blocks at 262k). Selection is skipped when the block count is at most 512.
- **Sparse attention**: attention over a per-query token list gathered from the paged fp8 KV. Decode is
  one query; prefill is 256 queries, each with its own list.
- Ties (Q23): among equal scores, the most recent blocks win (higher block index first). The reference
  (`tools/reference/qwen4exp/qsa_indexer.py`) gets the same rule as an option next to its HF-exact mode.

### 4.7 MoE kernels

- Decode (M = 1): one launch for the 10 selected experts' gate+up (N 640, K 2560) with expert-id
  indirection, silu-mul, then one for down (N 2560, K 640) with the routing weights folded into the
  accumulate.
- Prefill (256 rows = 2560 routed pairs): sort pairs by expert, a grouped trellis GEMM over per-expert
  row ranges (M mostly under 16, WMMA M 16 tiles), scatter-add back with weights.
- Shared expert: the existing bf16 linear path, sigmoid gate fused into its epilogue.

### 4.8 N-gram reader and PLE

- On the host: compute the 16 ids per token (int64, 2.3), round each row to the 4 KiB sectors it spans
  (320-byte rows straddle sector boundaries) and issue overlapped unbuffered reads (`FILE_FLAG_NO_BUFFERING`,
  IOCP, queue depth 64). A 256-row chunk is 4096 rows. Reads are issued at the start of the chunk and
  awaited before layer 1, so layer 0 hides them.
- Rows are uploaded to device 0; the PLE kernels (two GEMMs, grouped norms, gate, dilated conv) run there.
- A failed or short read is a hard error for the request, never a silent zero row.

### 4.9 Tokenizer and server

- Tokenizer: no code change. The Swift repos' `tokenizer.json` carries the old Qwen2 split pattern
  (re-saved by transformers), but their `tokenizer_config.json` `pretokenize_regex` is the Qwen3.5
  pattern, which r4dx prefers. Vocab, merges (after normalizing pair vs string form) and added tokens equal
  the 27B's, so `tokens_canon.json` is reused unchanged.
- Chat template: byte-identical to the 27B's (`reasoning_effort` xhigh/medium/low,
  `preserve_thinking`, `<function=...><parameter=...>` tool calls). Add a `qwen4_exp` dialect equal to
  the Qwen one; EOS ids from the checkpoint's `generation_config.json` (**VERIFY** in M1-19).
- Prefix reuse: snapshots at prefix boundaries as for the 27B, with the PLE conv state, the 2-token
  n-gram history and the QSA partial block added to the snapshot.

### 4.10 Reference tooling and gate

- Venv: new `D:\venvs\r4dx-qwen4-ref` (Python312 base, same torch ROCm wheel as `r4dx-gemma-ref`, which
  already has transformers 5.18 with `qwen4_exp`) plus `compressed-tensors` for the M0-4 cross-check.
  Existing venvs are not modified.
- `tools/reference/arch_table.py` gets a `qwen4_exp` `ArchSpec`; a `Qwen4ExpStreamingReference` loads one
  layer at a time on device 1 from the dequantized AWQ weights and reads n-gram rows from the sidecar.
- The HF indexer loops over queries in Python (`:424-459`). The reference uses a vectorized port that MUST
  equal the HF loop bit for bit at contexts up to 4k (M0-6) before it is used for the 8k/32k probes.
- **Gate (M1-21)**: teacher-forced logprobs on `tokens_canon.json` with fp8 KV, `kl_report.py`: mean KL
  <= 0.02 and top-1 >= 93%. Probes at 8k and 32k are reported against the same thresholds. Plus the
  server smoke (M1-22).

### 4.11 Implementation contract (decided 2026-10-10 after mapping the code; overrides earlier text)

Code facts that forced these: `ModelConfig::FromJson` and `Container::Load` cannot parse qwen4_exp; libr4d has
no gqa-12 attention and no trellis KB 3; the converter reads only BF16/F32 sources; `--pp 2` keeps the full
model on one card.

1. **GPU policy (supersedes the "agents never run GPU work" lines above).** The user granted standing
   permission for GPU runs on both cards, one GPU process at a time, desktop card included. Every GPU process
   and every cmake build goes through `tools\with_lock.ps1 -Name gpu|build -Run "<cmd>"`. Nothing else is run in parallel on a GPU.
2. **Tiny model.** All kernel, converter and model work is validated first on a random-weight `tiny-q4exp`:
   the real geometry (hidden 2560, GDN 16/48x128, attention 24/2x256, 640-wide experts, hc 4, PLE at layer
   index 1) with 4 layers (GDN, GDN, GDN, QSA), 16 experts top-10, vocab 4096, `ngram_vocab_size_base` 2003,
   `split_ngram_parts` 4, `indexer_budget` 64 (ratio 4, so top-16 blocks; dense-exact up to 67 tokens),
   eos 4095. Built by `tools/reference/qwen4exp/tiny.py` in the *checkpoint format* (per-expert unfused
   bf16 linears, n-gram shards) at `E:\models\r4dx\tiny-q4exp\hf`, with HF goldens beside it.
3. **Container tensors** (`r4dx-convert`, new `src/convert/qwen4exp_layout.{hpp,cpp}`, `model_arch` "qwen4_exp";
   `{layout}` is `bf16|w4a16|trellis` as today; norm weights are stored raw, kernels apply `1 + w`):
   - globals: `text.embed_tokens` bf16 [vocab,2560]; `lm_head.{layout}` (w4a16 g32 recipe);
     `text.hc_final.{norm,down,up}` bf16 (norm [10240], down [320,10240], up [10240,320]).
   - every layer `text.layers.{i}.`: `hc.attn.{norm,down,up,inject}` and `hc.mlp.{norm,down,up,inject}` bf16
     (inject [4,10240]);
     `moe.router` bf16 [E,2560]; `moe.shared.{gate,up,down}` bf16 linears (`.bf16.w`); `moe.shared_gate` bf16 [1,2560];
     `moe.experts.{e}.gate_up.{layout}` (2 parts [640,640]) and `moe.experts.{e}.down.{layout}`.
   - GDN layers: `gdn.in_proj_qkv.{layout}`, `gdn.in_proj_z.{layout}`, `gdn.in_proj_a`, `gdn.in_proj_b` bf16,
     `gdn.conv1d_weight` bf16, `gdn.A_log`, `gdn.dt_bias` fp32, `gdn.norm_weight` bf16, `gdn.out_proj.{layout}`.
   - QSA layers: `attn.qg.{layout}` (per-head interleaved q|gate rows, as Qwen3.5), `attn.k.{layout}`,
     `attn.v.{layout}`, `attn.o.{layout}`, `attn.q_norm`, `attn.k_norm`, `attn.k_descale`, `attn.v_descale`,
     `attn.indexer.qk` bf16 [640,2560], `attn.indexer.q_norm`, `attn.indexer.k_norm` bf16 [128].
   - PLE layer (index 1): `ple.key_proj`, `ple.value_proj` bf16, `ple.norm_key/norm_query/norm_conv` [10240],
     `ple.conv1d` [10240,1,4].
   - metadata: `model_arch` "qwen4_exp", verbatim `model_config`, `ngram_sidecar` {path, sha256, rows},
     `quant.trellis` as today with `bits` in {3,4,5}. **One KB per (layer, expert class gate_up|down)**: all experts
     of a layer's class share it, so one launch config serves every expert; the allocator works at that granularity.
   - source weights are read through a **virtual AWQ source** in the converter: a pack-quantized `<b>.weight_packed`
     triple appears as a BF16 `<b>.weight` (C++ dequant bit-equal to `awq_dequant.py`); nothing is materialised.
4. **Runtime.** `Qwen4ExpContainer` + `Qwen4ExpModel` + `Qwen4ExpLocalTextModel` (+ `LoadQwen4ExpTextModel` at the
   `tp_model.cpp` hook), listed in `r4dx_model`; kernels and host helpers in `src/model/qwen4exp` (library
   `r4dx_model_qwen4exp_kernels`, glob-based, see its CMake). No `--pp`/`--tp`: the model owns two `Stage` objects
   (device 0: embedding gather, n-gram rows, PLE, layers [0,k); device 1: layers [k,48), final mixer, lm_head)
   driven in sequence on the calling thread; the hand-off is the residual in pinned host memory (D2H then H2D,
   synchronous). Residual stream dtype: bf16, to match the bf16 HF reference rounding points (constant in one
   header; revisit only if per-layer goldens drift).
5. **MoE in milestone 1 is host-routed, built on `ApplyLinear`**: router + top-10 on the GPU, ids to the host
   (one sync per layer), rows gathered per expert, `ApplyLinear` per (expert, class) on that expert's rows,
   weighted scatter-add. No grouped kernel yet (milestone 2). Per-expert `QuantLinear`s are views into one
   device buffer per layer and class; tickets are one slice each.
6. **KB 3**: libr4d `r4d_gemm_trellis_nt_m64` gets KB 3 and every acceptance point the code map lists
   (`trellis_meta.h`, `linear_layouts.hpp`, `quant_linear.h`, `linear.cpp` TrellisMaxMt, `trellis_import.hpp`,
   tuning fallback). The M=256 and int8 prefill GEMMs are NOT extended: the arch never opens those scopes.
7. **QSA**: no libr4d kernel fits (gqa 12, selected-token lists), so new kernels: block-key pooling/norm/rope,
   scoring + top-k with the `recent` tie rule (Q23), sparse attention over the paged fp8 KV.
8. **Checkpoint/prefix reuse** keeps the server's existing single-slot, prefill-boundary `SaveCheckpoint` /
   `RestoreCheckpoint`; the arch's state set is GDN + conv + PLE conv + n-gram history + QSA partial block and
   block-key buffers + position.


## 5. Task list

Legend: **[CPU]** needs no GPU. **[GPU]** is a command handed to the user; agents never run GPU work.
HF reference work runs on device 1 only (`$env:HIP_VISIBLE_DEVICES = '1'`).

### Status (branch `qwen4exp`, 2026-10-10)

| Task | State | Evidence |
|---|---|---|
| M0-1 | done | 69 files, 165.14 GiB, every size equal to the HF API at rev 598afc96 |
| M0-2 | done | `D:\venvs\r4dx-qwen4-ref`: torch 2.9.1+rocmsdk20260116, transformers 5.18.0, compressed-tensors 0.15.0.1 (newest that accepts torch 2.9.1; the checkpoint was written by 0.18.1a); `tools/reference/qwen4exp/requirements-qwen4-ref.txt` |
| M0-3 | done | `E:\models\r4dx\swift15-flash-next.r4dx-ngram`, 320,001,536 rows, sha256 `f90a8563...434d`; `verify` compared 1256 rows OK. HF shards kept (not deleted). |
| M0-4 | Python half done | `awq_dequant.py`: bit-equal to compressed-tensors on synthetic tensors and on 4 real tensors (expert gate/down, attention q, GDN qkv). C++ half moves to M1-2. |
| M0-5 | done | `Qwen4ExpStreamingReference` (+ layer-major resumable real runner `real_ref.py`): bit-equal to HF on tiny; real model 4 x 1k-token canon segments in 257 s on device 1. |
| M0-6 | done | `qsa_indexer.py`: masks equal HF's 1..4096 tokens, 16-24x faster; `tie_rule="recent"` (Q23) equals a naive reference. |
| M0-7 | done | Real references under `E:\models\r4dx\q4exp\ref\` (canon, 8k/32k probes, layer goldens, `kv_calib.json`, fp32 truth); sizes + sha256 in `tools/reference/qwen4exp/REAL_REF.md`. |
| M1-1 | done | `Arch::kQwen4Exp`, `qwen4exp_config.h`. |
| M1-2 | done | `r4dx-convert` qwen4_exp layout + virtual AWQ source (bit-equal to `awq_dequant.py` on 8 real tensors); real checkpoint `--plan-only` consumes all 73,884 packed linears, header 12 MB, plan 2.5 s. |
| M1-3 | done | `Qwen4ExpContainer` (loader test 453 checks on bf16/w4a16/synthetic trellis), real 4-layer prefix container `D:\q4exp\real-prefix4-bf16.r4dx`. |
| M1-4..9 | done | Kernels with GPU tests vs HF fixtures: hc, MoE (router ties exact), PLE, sigmoid gated GDN norm, QSA core (selected sets equal the `recent` reference on 100% of rows, 0.28 ms decode at 262k ctx), n-gram reader (4096 rows in ~23 ms cold, 16 rows 0.27 ms). |
| M1-5 | done | KB 3: 48 libr4d instantiations, bit-exact reconstruct and CPU-reference GEMM on the qwen4exp shapes; KB 4/5 device code identical to HEAD (ISA listing diff). |
| M1-16..18 | done on tiny + real prefix | `Qwen4ExpModel`/`LocalTextModel`; layer-isolated golden tests pass for both GDN prefill modes; prefill/decode/chunk-split/checkpoint bit-identical; two-device split bit-identical to single device (crash found and fixed: stream created before `hipSetDevice`). |
| M1-10..14 | done | Full 48-layer quantization completed in ~25.4 h GPU time. Container written to `E:\models\r4dx\swift15-flash-next-trellis-mix.r4dx` (56.47 GB, 51.83 GiB weights split k=26: 25.92 GiB on dev 0, 25.91 GiB on dev 1). |
| M1-19..20 | done | Dialect, CLI, server flags and docs in place. |
| M1-21 | done | KL gate evaluated: mean KL 0.0578 vs bf16 ref (top-1 89.44%) and 0.0590 vs fp32 truth (top-1 89.42%) on canon; 0.0500 / 91.71% on 8k probe. |
| M1-22 | done | Server smoke test passed (37/37 checks: health, models, chat, thinking low/med/xhigh, stream SSE, tools, multi-turn prefix reuse, ~20k needle at 459 tok/s, max-ctx 400, clean shutdown). |
| M1-23 | done | Non-regression passed: 184 pass, 25 skip, 0 fail; existing Qwen/Gemma code proven byte-identical. |
Findings that changed the plan during implementation:
- **libr4d `r4d_gdn_chunk_scan` is wrong for this checkpoint** (exact only while a 64-row chunk's gate span is under about 160 nats; here A reaches e^158, spans up to 6505). A span-safe additive unit `GdnChunkScanStable` is the default; the shipped kernel is unchanged (ISA identical to HEAD). Without it the real model would have produced garbage.
- **Gate.** A stock bf16 re-run of the HF stack differs from itself by mean KL ~0.011 / top-1 ~96% (append one token), so the original "KL <= 0.02 vs the bf16 reference" leaves only ~0.009 nats for trellis error. REAL_REF.md proposes the Gemma-style noise-relative gate against the fp32 truth: KL(truth||r4dx) <= 1.5 x N + 0.005 per group (canon ALL: 0.0261, top-1 >= 94.23%). **Both gates are evaluated at M1-21; which one is binding is the user's call.**
- The tie rule `recent` vs HF-exact costs 0.0007 mean KL at 8k in fp32 (0 on canon, which is shorter than the dense limit).
- Sidecar is 95.4 GiB (102,400,495,616 bytes), not 92.

### M0: setup and reference

- **M0-1 Download [network].** `hf download ukisai/Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ --revision
  598afc96128f76a67117caf878f538d34f6f1e02 --exclude "*.mp4" --local-dir E:\models\r4dx\Swift-1.5-Qwen3.8-Flash-Next-W4A16-AWQ`.
  Done: 165 GiB on disk, every file's size matches the HF API.
- **M0-2 Venv [CPU, network].** Create `D:\venvs\r4dx-qwen4-ref` (4.10). Done: `import
  transformers.models.qwen4_exp, compressed_tensors` succeeds; other venvs untouched.
- **M0-3 Sidecar repack [CPU].** Tool writes `E:\models\r4dx\swift15-flash-next.r4dx-ngram` from the
  shards (4.2). Done: header matches `ngram_heads_vocab_sizes` / `offsets` / `layer_multipliers` from the
  checkpoint; sha256 recorded; 1,000 random rows equal the shard bytes.
- **M0-4 AWQ dequant [CPU].** Python and C++ dequant of pack-quantized int4. Done: equal to
  compressed-tensors' decompression bit for bit on one tensor of each class.
- **M0-5 Streaming reference [CPU, then GPU].** `Qwen4ExpStreamingReference` + `ArchSpec`. Done: full
  logits for a 64-token prompt equal HF's own model (dequantized weights, no streaming) on a 2-layer
  truncated config; the full-depth run is a [GPU] command.
- **M0-6 Vectorized indexer [CPU].** Done: selections equal the HF loop at 2051, 2052, 3000 and 4096
  tokens on random keys and on one real layer.
- **M0-7 Goldens [GPU].** Per-layer goldens for layers 0, 1 (PLE), 3 (QSA), 47, plus the mixer and
  lm_head, and full-logit references for `tokens_canon.json`, 8k and 32k.

### M1: text, two GPUs, server

**Foundations [CPU]**
- **M1-1 Config + arch dispatch.** 4.1. Done: the real config parses; the arch detection tests cover all
  three arches; Qwen and Gemma tests unchanged.
- **M1-2 Converter AWQ path + container metadata.** 4.2. Done: a synthetic 2-layer AWQ checkpoint converts;
  metadata round-trips; refusal on a revision whose `quantization_config` differs.
- **M1-3 Loader.** Container + sidecar loading, layer-split weight placement (3.3). Done: tiny container
  loads at split points 1 and N-1.

**Kernels [CPU authoring, GPU tests]**
- **M1-4 Router + MoE decode/prefill.** 4.7. Done: equal to a CPU reference at M 1, 16, 256 for random
  routings including experts with zero rows; router order verified against the M0-7 golden.
- **M1-5 Trellis KB 3.** 4.3. Done: `test_trellis_gemm.py` and `test_trellis_linear` KB 3 cases pass; KB 4/5
  outputs byte-identical to before.
- **M1-6 Gated residual.** 4.5. Done: layer-0 golden within the existing per-layer tolerance.
- **M1-7 GDN sigmoid gate.** Done: layer-0 GDN golden; 27B GDN tests byte-identical.
- **M1-8 QSA.** 4.6. Done: selections equal M0-6 at 1k, 2051, 2052, 8k; attention output equal to a dense
  reference masked by the same selection.
- **M1-9 N-gram reader + PLE.** 4.8. Done: ids equal the HF reference including EOS segmentation and
  carry-over across chunks; layer-1 golden.

**Quantization [GPU commands]**
- **M1-10 Trellis tooling for qwen4_exp.** `trellis_quant.py` / `hessian_capture.py` via `ArchSpec`, KB 3
  confirmed, per-layer pipeline (4.4). Done: dry-run prints every stage's command.
- **M1-11 Calibration corpus + Hessians.** hessian-v2 + SFT slice to about 1M tokens. Done: per-expert
  token counts written; fallback share reported.
- **M1-12 Oracles KB 3 and KB 4.** Done: manifests complete for 48 layers.
- **M1-13 Allocation sweep.** Done: chosen mix <= 26 GiB per card with its KL.
- **M1-14 Convert.** Done: container written to `E:\models\r4dx\swift15-flash-next-trellis-<mix>.r4dx`;
  measured per-card bytes recorded here.
- **M1-15 fp8 KV calibration.** Existing `kv_calibrate_full.py` via `ArchSpec`. Done: descales in the
  container.

**Model and two-GPU execution**
- **M1-16 `Qwen4ExpModel` forward.** Prefill in 256-row chunks, decode, both halves. Done: CLI greedy
  output on the four standard prompts is coherent; per-layer goldens pass.
- **M1-17 Layer-split runner.** Reuse `pp_channel` / `pp_stage_buffers` for the host hand-off; decode
  crosses cards every token. Done: identical logits to a single-process two-device emulation on a tiny model.
- **M1-18 Prefix reuse.** 4.9. Done: a two-turn conversation reuses the prefix and its logits equal a
  cold prefill of the same tokens.

**Server**
- **M1-19 Dialect + EOS.** Done: `/v1/models` lists the model; template renders with `reasoning_effort`.
- **M1-20 Tools and thinking.** Done: the existing server tool and reasoning tests pass against this model.

**Gate**
- **M1-21 KL gate [GPU].** 4.10. Done: thresholds met on the canon corpus; 8k and 32k probes reported.
- **M1-22 Server smoke [GPU].** Existing smoke scripts with this container: tools, thinking on/off,
  multi-turn prefix reuse, a 20k-token prompt. Done: zero failures; decode and prefill speeds recorded.
- **M1-23 Non-regression [GPU].** 27B and Gemma ctest suites and KL gates. Done: byte-identical to
  the frozen baselines.

## 6. Later milestones (not designed here)

- **M2 speed**: overlap the two halves at decode, MoE kernel tuning, n-gram prefetch depth.
- **M3 MTP** from `model_mtp.safetensors` (bf16, its own MoE layer and QSA attention).
- **M4 vision** (Qwen3-VL tower, 27 blocks, bf16 in the checkpoint).
- Validation beyond 32k.

## 7. Risks

- **Expert coverage.** With top-10 of 512, rarely routed experts get diagonal Hessians. If the fallback
  share is high, KL may miss 0.02; the remedy is more corpus, not a bigger budget (Q14).
- **Double quantization.** The AWQ weights already carry int4 error; the gate measures only r4dx's error
  on top of it, by design (Q11).
- **Desktop card.** Device 0 holds about 26 GiB of weights plus KV; display resets under load have been
  seen on this machine (README). The smoke MUST run with the desktop in normal use.
- **NVMe latency.** 16 reads per token on the critical path at decode; if layer 0 does not hide them,
  milestone 2 adds prefetch.
- **QSA ties.** Frequent, not rare (Q23). If the HF-vs-rule noise floor is itself near 0.02, the gate
  thresholds need revisiting with the user.
