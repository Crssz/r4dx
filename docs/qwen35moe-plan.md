# Ornith-1.5-122B (`qwen3_5_moe`) support for r4dx -- design and task plan

Written 2026-10-20. Design only: nothing here has been built or run on a GPU yet. Items tagged **VERIFY** come
from the HF `config.json` / model card and must be checked against the downloaded safetensors index and a golden.

## 1. What the model is

Source: `ornith-ai/Ornith-1.5-122B` (MIT, 62 BF16 shards, ~250 GB), `model_type: qwen3_5_moe`,
`Qwen3_5MoeForConditionalGeneration`. It is the **Qwen3.5 hybrid** (the family `Arch::kQwen35` already runs for the
27B) with the dense MLP replaced by a sparse MoE. It is NOT `qwen4_exp`: no hyper-connections, no PLE n-gram table,
no QSA, plain residual stream with `input_layernorm` / `post_attention_layernorm`.

| Item | Value |
|---|---|
| hidden / layers / vocab | 3072 / 48 / 248320 (untied lm_head) |
| layer pattern | `full_attention_interval` 4: layers 3, 7, ..., 47 full attention (12), the other 36 GDN |
| full attention | 32 q heads, 2 kv heads, head_dim 256, output gate (`q_proj` = query + gate), partial rotary 0.25, theta 1e7, interleaved mrope [11, 11, 10] |
| GDN | 16 key heads, 64 value heads, dim 128, conv kernel 4 (27B: 16 / 48) |
| MoE | 256 experts, top-8, `moe_intermediate_size` 1024, shared expert 1024 (+ sigmoid `shared_expert_gate`), no dense layers (`mlp_only_layers` = []) |
| MTP | 1 layer (`mtp_num_hidden_layers`), `mtp_use_dedicated_embeddings` false |
| vision | 27-layer tower, hidden 1152, `out_hidden_size` 3072 (same family as the 27B's tower, different sizes) |
| context | 262144 native |
| tokens | EOS 248044 and 248046 (`<|im_end|>`), image 248056, video 248057 |

Parameter split (derived): routed experts 48 x 256 x 3 x 3072 x 1024 = 116.0 B; everything else ~9 B (shared
experts 0.45 B, attention/GDN, two 0.76 B embedding matrices, MTP, vision).

### 1.1 Checkpoint layout (from `model.safetensors.index.json`, 2026-10-20; total 250,172,994,016 bytes)

Prefix `model.language_model.` for the text tower, `model.visual.` for vision, `mtp.` for the MTP head, `lm_head.weight` at top level.

- GDN layer: `linear_attn.{in_proj_qkv,in_proj_z,in_proj_a,in_proj_b,out_proj}.weight`, `linear_attn.conv1d.weight`, `linear_attn.{dt_bias,A_log}`, `linear_attn.norm.weight`.
- Full-attention layer: `self_attn.{q,k,v,o}_proj.weight`, `self_attn.{q,k}_norm.weight`.
- Both: `input_layernorm.weight`, `post_attention_layernorm.weight`, `mlp.gate.weight` (router), `mlp.shared_expert.{gate,up,down}_proj.weight`, `mlp.shared_expert_gate.weight`.
- **Routed experts of the main model are FUSED 3-D tensors** (HF 5.x layout): `mlp.experts.gate_up_proj` and `mlp.experts.down_proj`, not per-expert linears. The converter must slice them per expert; gate-then-up order inside `gate_up_proj` is **VERIFY** against the golden (M2).
- **The MTP head ships with the checkpoint**: `mtp.layers.0.*` (a full-attention block with an MoE FFN whose experts are PER-EXPERT `experts.{E}.{gate,up,down}_proj.weight`, unlike the main model), `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.fc`, `mtp.norm`. MTP is therefore possible (M6). A DFlash 2 drafter for this model also exists (`incoai/Ornith-1.5-122B-DFlash2`, section 6), so DFlash2 is in scope too.
- Vision: `model.visual.patch_embed.proj`, `pos_embed`, `blocks.{0..26}.{norm1,norm2,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`, `merger.{norm,linear_fc1,linear_fc2}`.

## 2. Fit on two R9700 (32 GB each)

BF16 is 250 GB, so the weights must be trellis-quantized (as the 27B and qwen4_exp are). Experts at KB 3 are ~43.5 GB,
at KB 4 ~58 GB; the non-expert part at ~4.5 bits is ~5 GB. With the qwen4_exp budget of at most 26 GiB of weights
per card plus KV, the target is a **KB 3 / KB 4 mix by (layer, class)** of about 3.3-3.6 bits for experts, split by
layer over both cards. Expect ~50-52 GiB total. One card cannot hold it.

### 2.1 Quantization pipeline numbers and the host-RAM option (2026-10-21)

The pipeline (`tools/quant2/qwen35moe/`, README there) budgets weights at 26 GiB per card. Its byte model (matching
`Qwen35MoeContainer::Inspect`): all-KB3 experts + KB4 projections = 44.4 GiB (k = 24), all KB4 = 57.9 GiB, which does NOT fit
2 x 26 GiB; the allocator spends the ~7.6 GiB between them on the best loss-per-GiB promotions. Oracle time scaled from the one-layer pilot is ~14.6 h per K
(~29 h for K 3 + 4, ~34 h with the Hessians; README in that directory) on one GPU; kept data ~99 GiB (oracle dirs) + ~53 GiB (container).

Open option: a **host-RAM expert tier** (coldest experts in pinned RAM, CPU computes them at decode, PCIe stream at prefill; the
dropped Qwen3.8 port budgeted 11.5 GiB of a 64 GB machine). It would lift the budget to ~52 + ~10 GiB and make an all-KB4 mix fit. Quantization
side: an allocator `--host-gib` option and a placement list from the per-expert routed counts the Hessian pass already records. Runtime
side (CPU expert path, pinned buffers) belongs to the runtime owner. Not decided.

## 3. Integration

New `Arch::kQwen35Moe` (on-disk `model_arch` `"qwen3_5_moe"`), dispatched in `src/model/arch.{h,cpp}` BEFORE the
`kQwen35` fallback (today any unknown Qwen-like header falls through to `kQwen35`, which would try to read
`intermediate_size` and fail). `ModelConfig::FromJson` requires `intermediate_size`, which this config lacks, so the
MoE model gets its own `Qwen35MoeConfig` rather than a loosened shared parser (Q8: the 27B path stays byte-identical).

Reuse, do not copy:

| Piece | From | Change needed |
|---|---|---|
| GDN mixer | `Qwen4ExpGdn` (`qwen4exp_gdn.h`, silu gate selectable) or `GdnLayer` | config adapter; check libr4d GDN kernels accept 16 key / 64 value heads (ratio 4) **VERIFY** |
| full attention + paged fp8 KV | `Model` path in `model.cpp`, `attention/` | none expected; head_dim 256, 32/2 heads like the 27B's 24/4 -> check GQA 16 **VERIFY** |
| MoE block | `qwen4exp::MoeBlockBase` (host-routed, `MoeConfig` already parametric) | instantiate for hidden 3072, inter 1024, k 8, E 256 |
| grouped on-GPU decode MoE | `moe_trellis_decode.hip` | **done 2026-10-21 (section 7.1):** 3072 / 1024 / top-8 / any expert count <= 1024 / bits 3-4 instantiated, `Qwen35GroupedMoeDecode` (`src/model/qwen35moe_grouped_decode.*`) wired into `Qwen35MoeStage::RunMoe` for one-row steps; `R4DX_Q35_GROUPED=0` keeps the host-routed MoE |
| layer-split pipeline | `Qwen4ExpModel` / `Qwen4ExpStage` pattern, `ChooseSplit` | new `Qwen35MoeStage` without HC/PLE/QSA; no sidecar |
| container, trellis GEMM, tokenizer, server | shared | dialect `qwen3_5` (same Qwen3.5 template family) **VERIFY** the chat template hash against the 27B's |
| converter | `r4dx-convert` qwen4_exp branch | new branch for `model_type: qwen3_5_moe`; checkpoint has per-expert or fused 3D expert tensors **VERIFY** from `model.safetensors.index.json` |

## 4. Milestones

- **M0 (download running; index read, section 1.1)** download to `E:\models\r4dx\Ornith-1.5-122B`.
- **M1 config + arch (done 2026-10-20, CPU only)** `Qwen35MoeConfig` (`src/model/qwen35moe_config.h`), `Arch::kQwen35Moe`, `LoadTextModel` refuses it with a clear message until M3, `tests/model/test_qwen35moe_config.cpp` passes. The existing test that expected a Qwen3.5-MoE `model_config` to be `kQwen35` was updated. No dialect change needed: the arch name `qwen3_5_moe` already resolves to the Qwen3.5 dialect.
- **M2 tiny golden (done 2026-10-21)** `tools/reference/qwen35moe/tiny.py` builds the real geometry at 4 layers / 16 experts in the real (fused-expert) checkpoint format; `tiny_golden.py` records per-layer HF activations for T = 1/7/70/300 and 24-token free runs, in three flavours: bf16, `--kv-fp8` (K/V rounded to fp8 e4m3 like r4dx's cache) and `--fp32 --kv-fp8` (the near-exact reference); `noise_floor.py` reports HF's own bf16-vs-fp32 noise. **Pitfall found:** building the HF model on the meta device and `to_empty()` leaves the rotary `inv_freq` buffer uninitialised, i.e. RoPE silently off; `load_hf_model` now builds the model for real and asserts `inv_freq`.
- **M3 model (done on the tiny checkpoint, 2026-10-21)** `Qwen35MoeContainer` / `Qwen35MoeStage` / `Qwen35MoeModel` / `Qwen35MoeLocalTextModel` (two-device layer split, host-routed MoE, `R4DX_Q35_DEVICES` / `R4DX_Q35_SPLIT`), `r4dx-convert` branch, `LoadTextModel` dispatch, CLI/server trellis default. New libr4d unit `r4d_attn_paged_h256_gqa16.hip` (head_dim 256, 32 q / 2 kv heads, fp8 KV: prefill, split-KV prefill, decode), tested against the stage-0 reference attention (`tests/kernels/test_attn_h256_gqa16.cpp`, error ~2e-3 relative). `tests/model/qwen35moe/test_model_q35moe_tiny.cpp` PASSES against the HF golden (both GDN prefill paths): every layer in isolation from the golden's residual (GDN layers, attention layer, router, experts, shared expert), the whole model, and 24-token greedy free runs (24 of 24 agree at T = 70). **Real weights (2026-10-21):** the same test PASSES on a 4-layer prefix of the real checkpoint (`tools/reference/qwen35moe/real_prefix.py subset` + `r4dx-convert`, 21.6 GiB bf16, three GDN layers and the first attention layer; `R4DX_Q35MOE_TINY_DIR=E:\models\r4dx\q35moe-real4`), whole model and layer by layer, against HF bf16 and against the fp32 layer-isolated reference (`iso_golden.py`): GDN and attention outputs at mean/rms 0.001-0.004 of the fp32 value; the MoE output at 0.014-0.027, i.e. at parity with HF's own bf16 layer (0.008-0.034, same method); 24 of 24 greedy tokens agree at T = 70. `r4dx-cli` runs the prefix end to end (load, chat template, prefill, decode). Lessons: real activations carry outlier channels ~100x the rms, so the test judges the fraction of elements outside an absolute + relative band, not max/rms; HF's bf16 MoE is a noisy reference. Not done: the two-device split test (physical GPU 1 is faulted, see below), trellis/w4a16 containers (needs M4), long-context behaviour, speed (the MoE is host-routed: one sync per layer, one GEMM per expert).
- **M4 converter + quantization** Hessians per layer from an HF reference forward (layer-streaming, 250 GB does not fit RAM), trellis oracle at KB 3 and KB 4, allocation, `r4dx-convert` branch. This is the long, GPU-hours stage; reuse `tools/quant2/qwen4exp/` scripts with the Qwen3.5 layer builder.
- **M5 end to end** CLI/server chat on the real container, KL gate vs HF bf16 (mean KL <= 0.02, top-1 >= 93%, the qwen4_exp bar), thinking + tool calls through the Qwen3.5 template.
- **M6 parity features** vision tower (done 2026-10-23: `src/vision` reused unchanged, docs/qwen35moe-vision.md), `--tp 2` MoE sharding, MTP head, grouped GPU MoE decode, prefill int8 path, DFlash2 speculative decoding (section 6; after the quantized container and the grouped decode kernel), grouped MoE prefill speed (section 8). Performance estimates: section 7.

## 5. Out of scope / decisions

- **DFlash2**: NOT out of scope. An earlier version of this plan said no drafter exists; that was wrong, `incoai/Ornith-1.5-122B-DFlash2` exists (section 6). Original DFlash (v1) has no Ornith drafter and stays out.
- **MTP**: the weights ship (section 1.1). Deferred to M6, not dropped. DFlash2's acceptance length is far above a single MTP head's, so MTP may be dropped if DFlash2 works (decide after the DFlash2 measurement).
- **KV / long context**: fp8 paged KV as for the 27B; validated contexts are decided at M5.
- Every GPU job runs through `tools\with_lock.ps1 -Name gpu` and one at a time (dual-GPU sustained load has reset the display on this machine).


## 6. DFlash2 speculative decoding (built and measured 2026-10-23)

Source: `incoai/Ornith-1.5-122B-DFlash2` (Apache-2.0, 0.9 B, bf16, 1.86 GB; 6 layers, hidden 3072, 32 q / 8 kv heads x 128, window 2048, block 16 = anchor + 15 drafted tokens,
target taps at layers 1, 7, 14, 20, 26, 32, 39, 45, two-tap conv in groups of 16, selector rank 256 / top-k 16, mask token 248077). Model-card numbers (SGLang on two GB300,
bf16 target, 4096 new tokens, thinking on, NOT this machine): acceptance length 5.4 (MT-Bench) / 6.4 (MBPP) / 8.3 (GSM8K) / 8.6 (HumanEval) / 9.2 (MATH-500).
Everything below is r4dx on the two R9700, container `ornith-1.5-122b-trellis-mix-kv.r4dx`, greedy, thinking on, tool `tool_q35moe_dflash`. Design and conventions: `docs/dflash2.md` section 14.

### 6.1 Measured result

`r4dx-cli` / `r4dx-server`: `--dflash E:\models\r4dx\ornith-1.5-122b-dflash2-w4a16.r4dx --dflash-k 11 --dflash-p-min 0.8` (flags as for the other models; `--dflash-k` up to 15; the
global `--dflash-p-min` default stays 0, so pass it). Every DFlash2 stream below is **token-for-token identical to plain greedy decode** (section 6.3). Speed is the in-process tok/s of 399 decoded tokens
(`--max-ctx 8192`), best of 2, against plain decode run in the same process (plain: 60.6-61.9 tok/s, ~16.3 ms per token; it moves by +-3% between sessions, so only the ratios mean anything).

| prompt (class) | plain tok/s | k=4, p_min 0.35 | k=7, p_min 0.8 | **k=11, p_min 0.8** | tokens per round at k=11 / p 0.8 |
|---|---:|---:|---:|---:|---:|
| code1 CSV parser (code) | 61.4 | 1.55x | 1.81x | **1.82x** (111 tok/s) | 3.47 |
| code2 LRU cache (code) | 60.8 | 1.82x | 2.06x | **2.13x** (130 tok/s) | 4.16 |
| math1 trains + primes (math) | 61.0 | 1.85x | 2.17x | **2.27x** (140 tok/s) | 4.75 |
| math2 dice (math) | 61.3 | 1.78x | 2.18x | **2.33x** (142 tok/s) | 5.05 |
| chat1 Kyoto itinerary (chat) | 61.9 | 1.05x | 1.29x | **1.30x** (80 tok/s) | 2.04 |
| chat2 vaccines (chat) | 61.7 | 1.21x | 1.41x | **1.40x** (85 tok/s) | 2.40 |
| long1 4.4k-token prompt, summarise (chat) | 60.6 | 1.26x | 1.33x | **1.30x** (78 tok/s) | 2.83 |

**Plainly: with k=11 / p_min 0.8, chat about 1.3-1.4x, code about 1.8-2.1x, math about 2.3x, and a 4.4k-token prompt 1.3x.** That is at or above the 1.2x / 1.75x / 2.3x that section 7 expected, but only with a
confidence cut: with `p_min` 0 (the CLI default) chat is 0.96-1.17x at k=7 and large k loses to plain decode (see the grids in 6.5); k=15 with p_min 0 measured 0.72-0.80x on chat and long1 (an earlier run, not repeated).
Why: a verify round costs ~31-39 ms whatever the acceptance, plain decode produces a token every 16.3 ms, so a round must yield more than ~2.1-2.4 tokens to pay. Chat yields 2.0-2.5 per round even after the
cut; the cut (`p_min`, a drafter-confidence walk over the selector's candidates that stops drafting early) removes the long, low-probability tails of the window, which cost verify rows and are rejected anyway
(share of drafted tokens accepted on chat1 at k=7: 29% at p_min 0.35, 61% at 0.7). `p_min` 0.7-0.8 is a plateau (within 2-3% of each other on 7 of 7 prompts, 0.9 loses on code); k beyond 7 matters only on math and code.
The acceptance is below the model card's (code 3.5-4.7 tokens per round at k=11-15 against 6.4-8.6, math 4.8-6.5 against 8.3-9.2, chat 2.0-2.8 against 5.4) for two reasons we can name: the target is a 3.4-bit
trellis mix with KL 0.125 to bf16 (section 9), and the drafter was trained against the bf16 model; the card also averages over 4096-token generations at default sampling. Nothing here was tuned per prompt.

An earlier run on the same build before the drafter-attention and p_min work (`--max-ctx 4096`, 300 tokens, k=7, p_min 0.35) measured chat 1.21x, code 1.75x, math 2.28x; the 8192 / 399-token runs above are the ones to quote.

### 6.2 What was built (the work items of the original plan)

1. **Drafter container**: `r4dx-convert --dflash-hf` accepts the native `DFlash2DraftModel` checkpoint (real conv and selector copied, geometry from `dflash_config`, taps stored as layer inputs
   [2,8,15,21,27,33,40,46]); `E:\models\r4dx\ornith-1.5-122b-dflash2-{bf16,w4a16}.r4dx` (1.86 / 0.71 GB). The runtime's `DflashDraft` needed no change for the geometry. Checked against the vendored
   z-lab HF module on the real weights (`test_q35moe_dflash2_golden`: RelL2 5.1e-2 vs HF's own bf16 floor of 4.5e-2). **w4a16 vs bf16** (k=7, p_min 0.35, earlier run): 1.80x vs 1.76x (code2), 2.08x vs 2.02x (math1),
   1.12x vs 1.11x (chat2); the bf16 drafter does not accept more (5.47 vs 5.39 tokens per round on math1) and drafts ~1.8 ms/round slower, so w4a16 is the recommended drafter (and 1.15 GB smaller).
2. **Feature taps**: each stage copies the residual entering its tap layers (5 taps on device 0, 3 on device 1) into a device staging buffer and sends one 2D copy per chunk to pinned host memory; the model uploads the committed
   rows to the drafter (device 1) after every prefill chunk, decode step and accepted verify prefix. (Eight separate strided D2H copies per step cost ~3 ms; staging fixed it.)
3. **Rollback**: GDN layers use the 27B's write-once window state (state B + per-row log, `docs/gdn-write-once.md`; the kernels ran unchanged at 16 key / 64 value heads); rejected attention rows are simply overwritten.
4. **M-row grouped MoE** (`Qwen35GroupedMoeDecode::RunRows`): one launch per phase for 2-16 rows, each distinct expert streamed once for all the rows that chose it: 0.9 ms for 16 rows / 103 distinct experts on random routing (570-585 GB/s,
   the M=1 kernel gets 440), and its output is bit-identical to the M=1 kernel's per row. Real routing is far friendlier than random: a verify window's rows pick ~25-35 distinct experts per layer, MoE is ~25% of a verify step.
5. **Driver and flags**: `Qwen35MoeModel::{VerifyWindow, CommitVerifiedWindow, DecodeStepDflashGreedy}`, `Qwen35MoeLocalTextModel`, CLI/server flags; a temperature > 0 round is one plain sampled step (the server also switches injection off for sampled requests).
6. **Drafter attention** (`DflashAttnKernel`): the key pass runs 4 key groups x head_dim threads with a fixed-order group sum; at a 4.4k-token context the draft costs 9.1 ms per round at k=4, and long1 went from 0.93x to 1.24x at k=4 (same container, before/after this kernel and the later drafter work).
7. **Tests**: `test_model_q35moe_spec` (tiny trellis container: scripted windows with random rollbacks, plain-step and prefill transitions, DFlash rounds with random and scripted drafts; bit-identical to decode), `test_q35moe_grouped_moe_kernel` (M-row kernel vs host-routed and vs the M=1 kernel, plus the bench),
   `test_q35moe_dflash2_golden`, `convert_dflash_hf` (native fixture), `tool_q35moe_dflash` (real container: speed table and lossless audit), `tools/q35moe_server_dflash_smoke.ps1` (real container through `r4dx-server`: a plain, a streaming and a tool-call request,
   run with `--dflash` and without, greedy outputs byte-identical (tool call compared by function name and arguments; the random call id differs); 300-token chat end to end 87.9 tok/s with `--dflash` (k=7, p_min 0.35) vs 58.4 plain; the server log reports the round statistics).

### 6.3 Lossless: the verify window equals plain decode, bit for bit

The first working version was only noise-close: a teacher-forced audit (verify windows over the plain stream against the decode logits of the same positions) showed the top logit differing by 0.7-1.2 on average and up to 31, with 1.3-4.5% of
positions flipping their argmax, so DFlash2 streams left plain greedy after 20-120 tokens. (This 3.4-bit model is that sensitive to last-bit changes: swapping only the MoE implementation of plain decode, with 8 of 860k elements differing, moves the
same logits by up to 10 and flips 3 argmaxes in 399.) Two causes, both fixed: the M-row MoE used different split-group counts from the M=1 kernel (now 4/4 like it), and verify attention beyond 4 rows ran the prefill kernel (now the decode kernel, 4 rows per launch). Details in `docs/dflash2.md` section 14.3.
After the fix: 0.000 noise and 0 flips on every audited prompt and window (2-16 rows, up to a 4.4k-token context), and **every DFlash2 stream of every row of section 6.5 is identical to the plain stream**. The caveat that remains: a prompt is prefilled with different kernels
from decode, so "identical to plain greedy" means identical to prefill + plain decode of the same prompt.

### 6.4 Where the time goes, memory

Round at k=7 on math (8 rows), before the last tuning: ~39 ms = draft 5-7 + verify ~31 + inject 0.5 + sync; verify = GDN ~14 ms (36 layers), MoE ~11 (48), attention ~5 (12), head 1, the rest launch gaps (the verify step is ~1500 kernels; the GPU idles ~8 us between dependent small kernels, host enqueue is only ~1 ms of it).
Plain decode is ~16.3 ms per token for the same 48 layers, so a verify of 8 rows costs ~2x one decode step, which caps the gain to ~(tokens per round) / 2. The `p_min` cut works by shrinking the window to the few leading drafts the drafter is confident about, which is why it helps most where acceptance is low.
Memory: the drafter (w4a16 0.71 GB / bf16 1.86 GB), the write-once logs (0.66 MiB per GDN layer), the M-row MoE scratch (~170 MB per stage) and the staging buffers add roughly 1.3 GiB (w4a16) to device 1 (estimated from the allocations, not read back from the driver:
the server does not print a per-card total; the weights are 25.9 GiB on device 0 and 25.6 GiB on device 1 as before). `--max-ctx 16384` loads and serves with `--dflash` on the 32 GB cards.

### 6.5 Full grid (speed-up vs plain in the same run, tokens per round in brackets; k and p_min)

Three sessions of the same build (cells appearing in several runs show the last one; run-to-run spread about 3%, up to 8% on a single cell). All streams identical to plain greedy.

**k = 4** (speed-up vs plain; tokens per round in brackets)

| prompt | plain tok/s | p_min 0.00 | p_min 0.35 | p_min 0.50 | p_min 0.70 | p_min 0.80 | p_min 0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| code1 | 61.4 | 1.51 (3.41) | 1.55 (3.41) | 1.57 (3.35) | 1.60 (3.02) | - | - |
| code2 | 60.8 | 1.81 (3.84) | 1.82 (3.84) | 1.87 (3.80) | 1.72 (3.19) | - | - |
| math1 | 61.0 | 1.77 (3.95) | 1.85 (3.95) | 1.82 (3.91) | 1.86 (3.66) | - | - |
| math2 | 61.3 | 1.77 (3.91) | 1.78 (3.91) | 1.81 (3.91) | 1.84 (3.76) | - | - |
| chat1 | 61.9 | 0.97 (2.32) | 1.05 (2.28) | 1.10 (2.24) | 1.20 (2.09) | - | - |
| chat2 | 61.7 | 1.17 (2.64) | 1.21 (2.64) | 1.26 (2.57) | 1.32 (2.38) | - | - |
| long1 | 60.6 | 1.24 (2.98) | 1.26 (2.98) | 1.28 (2.96) | 1.29 (2.79) | - | - |

**k = 7** (speed-up vs plain; tokens per round in brackets)

| prompt | plain tok/s | p_min 0.00 | p_min 0.35 | p_min 0.50 | p_min 0.70 | p_min 0.80 | p_min 0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| code1 | 61.4 | 1.64 (4.16) | 1.69 (4.11) | 1.75 (4.03) | 1.81 (3.63) | 1.81 (3.30) | 1.73 (2.98) |
| code2 | 60.8 | 1.95 (4.69) | 1.94 (4.69) | 2.12 (4.64) | 2.02 (3.95) | 2.06 (3.84) | 1.91 (3.44) |
| math1 | 61.0 | 2.08 (5.39) | 2.17 (5.32) | 2.16 (5.05) | 2.14 (4.59) | 2.17 (4.38) | 2.11 (4.07) |
| math2 | 61.3 | 1.94 (4.81) | 1.95 (4.87) | 1.99 (4.75) | 2.15 (4.75) | 2.18 (4.48) | 2.16 (4.16) |
| chat1 | 61.9 | 0.96 (2.53) | 1.07 (2.48) | 1.14 (2.40) | 1.31 (2.18) | 1.29 (2.03) | 1.26 (1.89) |
| chat2 | 61.7 | 1.10 (2.77) | 1.15 (2.77) | 1.27 (2.70) | 1.38 (2.48) | 1.41 (2.38) | 1.39 (2.23) |
| long1 | 60.6 | 1.12 (3.22) | 1.18 (3.22) | 1.25 (3.09) | 1.30 (2.91) | 1.33 (2.83) | 1.33 (2.71) |

**k = 11** (speed-up vs plain; tokens per round in brackets)

| prompt | plain tok/s | p_min 0.00 | p_min 0.35 | p_min 0.50 | p_min 0.70 | p_min 0.80 | p_min 0.90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| code1 | 61.4 | 1.48 (4.48) | 1.51 (4.43) | 1.66 (4.29) | 1.79 (3.76) | 1.82 (3.47) | 1.72 (3.12) |
| code2 | 60.8 | 1.78 (5.18) | 1.85 (5.18) | 2.10 (5.12) | 2.10 (4.34) | 2.13 (4.16) | 1.98 (3.66) |
| math1 | 61.0 | 2.14 (6.54) | 2.18 (6.44) | 2.25 (6.14) | 2.25 (5.12) | 2.27 (4.75) | 2.21 (4.38) |
| math2 | 61.3 | 1.86 (5.62) | 1.93 (5.62) | 2.07 (5.62) | 2.22 (5.32) | 2.33 (5.05) | 2.32 (4.64) |
| chat1 | 61.9 | - | - | - | 1.31 (2.19) | 1.30 (2.04) | 1.27 (1.89) |
| chat2 | 61.7 | - | - | - | 1.38 (2.51) | 1.40 (2.40) | 1.38 (2.24) |
| long1 | 60.6 | - | - | - | 1.29 (2.93) | 1.30 (2.83) | 1.32 (2.71) |

### 6.6 What is not done / next

* A sampled (temperature > 0) round is not speculated (one plain sampled step; injection is switched off, the next greedy request refills the ring).
* `--tp 2` would shorten the verify step (plan section 7) and the drafter would then run replicated; not touched here (another session's work).
* Cheaper verify: fewer kernels per layer (graph capture or fused GDN projections) is the largest remaining lever; the MoE kernel is at ~560 GB/s and the GDN scan is latency- not bandwidth-bound.
* The defaults: `--dflash-p-min` is 0 for every model; for Ornith the table says 0.8 (and k 11) is better everywhere measured, so a per-model default would be a one-line follow-up (not done: the flag is shared with the 27B/Gemma paths, which were not re-measured).
* Not run: a thinking-off / sampled-temperature table, prompts longer than 4.4k tokens, task-accuracy (the streams are identical to plain greedy, so it is the plain model's accuracy).
* MTP is probably still not worth building next to this (plan section 9 decision stands).

## 7. Performance expectations (estimates, NOT measured on Ornith; written 2026-10-21)

Method: bytes read per token from the config (active ~9.0 B parameters, ~4.9 GB at ~3.5-bit experts / ~4.5-bit rest, shared experts assumed
quantized) at the qwen4exp-measured ~290 GB/s kernel bandwidth, plus ~4 ms fixed host-sync / card hand-off per pipeline step
(`docs/qwen4exp.md`, "Decode performance"); `--tp 2` factors from the 27B (`docs/tp.md`: 0.63-0.7x kernel time per rank, ~1.65x measured
on plain decode, all-reduce 7.5 us / 10 KiB); fusion and tuning +5-15% (the doc's measured fusion gains were small, 2.4% for HC + QSA).
P2P between the two R9700s is unavailable on Windows and would add only ~5%.

| Decode, single request (tok/s) | Range | Planning |
|---|---:|---:|
| today, host-routed MoE | 15-18 | 16 |
| pipeline split, grouped kernel + fusion (kernel measured: 0.12-0.135 ms per MoE layer, section 7.1) | 36-54 | 46 |
| + MTP (k = 1-2, accepted 70-85% assumed) | 50-75 | 62 |
| + DFlash2: chat / code / math / MATH-500-like | 40-75 / 47-89 / 61-115 / 68-128 | 58 / 70 / 88 / 98 |
| `--tp 2`, plain | 55-75 | 65 |
| `--tp 2` + MTP | 70-100 | 85 |
| `--tp 2` + DFlash2: chat / code / math / MATH-500-like | 56-100 / 66-120 / 85-150 / 95-170 | 85 / 100 / 125 / 140 |

Caveats: if the shared experts stay bf16 (as in qwen4exp) subtract ~8% from every row; run-to-run spread is ~10%; decode slows with context;
DFlash2 and MTP gains are workload dependent. **Measured** DFlash2 (section 6.1, k=11 / p_min 0.8, w4a16 drafter): chat 1.3-1.4x, code 1.8-2.1x, math 2.3x, 4.4k-token prompt 1.3x, i.e. about 80 / 111-130 / 140 / 78 tok/s on the 61 tok/s pipeline-split decode.

### 7.1 Grouped on-device decode MoE (measured 2026-10-21)

`Qwen35GroupedMoeDecode` runs the one-token MoE of a layer with no host sync: router GEMM, route, shared expert (5 bf16 kernels), sort,
gate_up input transform, grouped trellis GEMV, silu * up fused with the down input transform, grouped GEMV, combine -- 12 launches. It
applies to layers with hidden 3072, expert width 1024 (routed and shared), top_k <= 16, <= 1024 experts, trellis KB 3 or 4 on both expert
projections and a container prescale of 0; everything else, every prefill chunk and every captured run keeps the host-routed path.
`R4DX_Q35_GROUPED=0` (or `Qwen35MoeModelOptions::grouped_decode = 0`) switches it off.

Design choices that matter: the split-group counts are 4 (gate_up) and 4 (down), the counts the host-routed `ApplyLinear` picks at M = 1
for these shapes, the silu is rounded to bf16 before the multiply by up (as HF and the host path do), and the combine sums the experts in
router order -- so the bf16 outputs match the host-routed MoE almost bit for bit. The shared expert is NOT folded in as an extra expert:
its weights are bf16, the routed ones trellis, so it could not be bit-exact. A side stream for the shared expert was 1.5x slower (cross-stream
waits cost more than the overlap saves) and was removed.

Per layer-call at 256 experts, top-8 (synthetic weights, 6 layers cycled so the 64 MB Infinity cache cannot hold the working set; device 1;
`test_q35moe_grouped_moe_kernel`, 1000 calls, host wall):

| gate_up / down bits | grouped | host-routed | bytes per call | achieved |
|---|---:|---:|---:|---:|
| 4 / 4 | 0.131-0.135 ms | 0.43-0.51 ms | 58.6 MB (38.1 routed + 20.4 router / bf16 shared) | 435-446 GB/s |
| 3 / 4 | 0.124 ms | 0.44 ms | 52.3 MB | 423 GB/s |
| 3 / 3 | 0.120 ms | 0.43 ms | 49.2 MB | 410 GB/s |

About 3.3x over the host-routed layer. Phases (event bracketed, so they sum to more than the wall time): shared expert 0.045 ms, gate_up GEMV 0.06,
down GEMV 0.057, route 0.015-0.02, gate_up transform 0.025. The qwen4exp reference is 0.085 ms at 10 experts of 640 (less than half the bytes).
Split-group counts 2-12 (gate_up) and 1-8 (down) are within 3% of each other; 8 / 8 is 1-2% faster but rounds differently. The 20 MB of bf16
router + shared expert is a third of the bytes; quantizing the shared expert would cut ~0.03 ms per layer.

Equivalence (`test_model_q35moe_grouped`, trellis tiny container `q35moe-tiny\tiny-mix.r4dx`, prompts T = 1 / 7 / 70 / 300): the 48-step greedy
continuations are identical token for token (49 of 49) with zero teacher-forced argmax flips, logits identical bar one prompt at mean/rms 0.0002;
layer-level `moe_out` over 280 real single-row calls differs in 8 of 860k elements (largest 0.5% of the row rms). 256-expert synthetic layer:
grouped vs host-routed max|err|/rms 0.0005 (4-bit), 0.0000 (3-bit gate_up). Decode of the 4-layer tiny model: 2.5-3.9 ms -> 1.3-1.7 ms per token.
Not covered: no real-dimension trellis container exists yet (`q35moe-real4` is bf16, so the grouped path does not apply to it).
## 8. Prefill speed (planned, not started)

Expected today: ~150-300 tok/s (pipeline split), ~200-400 with `--tp 2`; an 8k prompt takes ~27-55 s to the first token, 32k ~2-2.5 min.
Scaled from the qwen4exp measurements (230-450 tok/s, ~5 TFLOPS useful, against the dense 27B's ~1000 tok/s / ~54 TFLOPS on the same cards).
Nothing was profiled; the causes are inferred:

1. Host-routed MoE: a router-id D2H copy and host sync per layer, then per-expert GEMMs, ~25,000 launches per 256-row chunk (~0.25-0.5 s of the
   observed ~0.57 s per chunk).
2. Skinny expert GEMMs: ~8 rows per expert in a 256-row chunk against 64-row tiles (`r4d_gemm_trellis_nt_m64`).
3. The trellis decode to fp16 is not amortised over few rows.
4. Nearly all 256 experts are read in every layer, so each chunk streams ~52 GB (floor ~0.11 s, ~2000 tok/s).
5. The pipeline split keeps one card idle at a time; the chunk cap is 256 rows.

Work items (after the quantized container; guess: 500-900 tok/s afterwards, practical ceiling ~1500):
a. Grouped prefill MoE GEMM: sort rows by expert, one launch per layer, dense tiles (probably the biggest gain).
b. Larger chunks (512-1024 rows) so each expert sees more rows and weights are streamed less often; check the activation VRAM.
c. fp8 / int8 MoE prefill path (already the M6 "prefill int8 path").
d. Overlap chunks across both cards (`docs/pp-prefill.md`, not ported to this model) and/or `--tp 2`.
e. Profile first: add an `R4DX_Q35_PROFILE` timing breakdown for prefill (per-layer MoE / GDN / attention) before changing kernels.

## 9. Status 2026-10-22: what shipped, measured numbers, decisions

**Usable model.** `E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx` (56.5 GB file; calibrated fp8 KV; same trellis bytes as `ornith-1.5-122b-trellis-mix.r4dx`).
KB 4 for attention / GDN projections, experts KB 3 / KB 4 mixed (allocation: `E:\models\r4dx\q35moe\mix\allocation_report.md`); 51.5 GiB of weights, layers 0-26 on
device 0 (25.93 GiB) and 27-47 on device 1 (25.57 GiB); about 26.9 GiB used per card with `--max-ctx 16384`.

```
set R4DX_ALLOW_GPU0=1
build\win-hip\src\cli\r4dx-cli.exe --model E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx --layout trellis --tokenizer-dir E:\models\r4dx\Ornith-1.5-122B --chat --max-ctx 16384 --stats
build\win-hip\src\server\r4dx-server.exe --model <same> --layout trellis --tokenizer-dir <same> --port 8080 --max-ctx 16384
```
(through `tools\with_lock.ps1 -Name gpu -Run "..."` whenever another GPU job may be running; the model needs both cards.)

**Measured (not estimated), 2026-10-22, one prompt each, so first readings.**

| item | result |
|---|---|
| load | 75-94 s |
| decode (grouped MoE kernel, default) | 56-70 tok/s (7 to 128 generated tokens; 60 tok/s at 7k context) |
| prefill | 117-230 tok/s on 24-43 token prompts; 7,125 tokens in 10.3 s (693 tok/s) on a quiet machine, 14.5 s (492) and 523 tok/s on later runs; plan on ~500 tok/s |
| chat / code / Thai / reasoning | correct on the CLI prompts; thinking on and off |
| server | plain, thinking (`reasoning_content`), tool call + follow-up turn, streaming: all passed |

**Quality (Rung 4 KL gate, `docs/qwen35moe-kl-gate.md`): NOT MET and accepted by the user.** Pooled over 18,454 held-out rows the calibrated container has KL 0.125 vs
bf16 (gate A <= 0.02) and 0.124 vs the fp32 truth (gate B <= 0.074), top-1 88.4% (gate >= 93%). Tool-use turns 0.034 / 94.4%, code 0.040 / 91.5%, English 0.053 / 90.3%,
Thai 0.071 / 88.1%, long text 0.184 / 86.8%. The loss is in the 3.4-bit weights (engine adds ~0.003, KV calibration changes nothing); it grows with context.
Class p (attention + GDN projections) causes about half of it. Evaluated and NOT built (user decision to stop): re-allocation only (predicted -0.005, measured +0.001 end to end),
bf16 class p (-0.006, slower), w8a16 class p (0.090 predicted, 150-250 lines, needs converter + loader), K5 class p (~4 h of Hessians).
No task-accuracy benchmark has been run: KL measures drift, not accuracy.

**Decisions.** Roadmap changes: the grouped decode kernel made decode fast enough (>= 35 tok/s bar met), so speculative decoding is lower priority than first planned;
MTP is probably dropped in favour of DFlash2 (drafter downloaded to `E:\models\r4dx\Ornith-1.5-122B-DFlash2`).

**Open items, in priority order (estimates):** P1 accuracy benchmark + 32k needle test (3-4 h); P2 prefill speed, grouped prefill MoE GEMM (6-12 h); P3 vision tower (done 2026-10-23, docs/qwen35moe-vision.md);
P4 w8a16 for class p (3-4 h, only if P1 shows a real loss); P5 DFlash2 (DONE 2026-10-23, section 6); P6 `--tp 2` (design done, `docs/qwen35moe-tp-design.md`: 70-95 h in seven phases, ~1.5x plain decode, trellis slices exactly, no re-quantization; IMPLEMENTED 2026-10-23 on branch `claude/q35moe-tp2-f5ebfb`, `docs/qwen35moe-tp.md`: decode 9.19 ms vs 14.67 ms per token = 1.60x, KL unchanged, DFlash2 under TP not yet); P7 MTP (0.5-1.5 days, dropped in favour of DFlash2 unless asked); P8 K5 / host-RAM expert tier;
P9 qwen4exp regression check with its real container (the shared kernel code was edited; only a synthetic test ran).

**Operational notes.** The 48-layer quantization stalled three times (singular refit solve, a zero-Hessian expert, pruned-channel blow-up in `regularize`) and once more on an unexplained
process kill; all fixes are in `tools/quant2/qwen35moe/`. `E:` is the tight resource (about 74 GiB free; `oracle-k3` + `oracle-k4` hold 99 GiB and are only needed for re-allocation).
Live progress table: `tools\q35_progress.ps1`.

**Update 2026-10-23 (DFlash2 merged, `d6507d5`).** DFlash2 works and is lossless (every stream equals plain greedy decode). Use:
`--dflash E:\models\r4dx\ornith-1.5-122b-dflash2-w4a16.r4dx --dflash-k 11 --dflash-p-min 0.8` (the CLI / server default p_min is 0 and must be passed; at p_min 0 chat is ~1.0x and k=15 can be slower than plain).
Measured against a same-process plain baseline of ~61 tok/s (399 tokens, both cards): code 1.8-2.1x, math 2.2-2.3x, chat 1.3-1.4x, a 4.4k-token prompt 1.3x. Acceptance is below the model card's
(math 4.8-6.5 vs 8.3-9.2 tokens per round), most likely because the target is a 3.4-bit mix. Extra VRAM on device 1 is an estimate of ~1.3 GiB (not measured). `--max-ctx 16384` loads and serves with `--dflash`.

**Update 2026-10-23 (`--tp 2` merged, `10a246d`; details in `docs/qwen35moe-tp.md`).** `--tp 2` works on Ornith (opt-in; the layer split stays the default and the two modes cannot share one process well).
Measured, same container and protocol, each mode in its own process: plain decode 9.19 ms/token = **108.7 tok/s** vs 14.67 ms = 68.2 tok/s on the layer split (**1.60x**); prefill of 512 tokens 1.10x; KL unchanged
(0.12364 vs 0.12361 against truth); fault recovery and a 5-minute soak (1.24M all-reduces, no TDR) passed; the 60-minute soak has not been run (needs the owner's go, display risk). DFlash2 under TP is lossless and
averages 116.7 tok/s over five corpus segments vs 85.4 for layer-split DFlash2 (1.37x), but only ~1.14-1.16x over plain TP, so with `--tp 2` the speculative gain is small (the `--dflash-k` cap of 7 in the shared parsers applies to `--tp 2`).
Open follow-ups: shared-expert quantization (bf16 now; ~20% of the decode bytes), the grouped prefill GEMM (TP barely helps first-token time), a known 2x slowdown / 0.8 GiB leak when the layer split and `--tp 2` are loaded in one process.
Commands: `r4dx-cli --model E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx --tokenizer-dir E:\models\r4dx\Ornith-1.5-122B --tp 2 --chat`, same flags for `r4dx-server`.
