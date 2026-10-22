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
- **M6 parity features** vision tower for the new sizes (reuse `src/vision`), `--tp 2` MoE sharding, MTP head, grouped GPU MoE decode, prefill int8 path, DFlash2 speculative decoding (section 6; after the quantized container and the grouped decode kernel), grouped MoE prefill speed (section 8). Performance estimates: section 7.

## 5. Out of scope / decisions

- **DFlash2**: NOT out of scope. An earlier version of this plan said no drafter exists; that was wrong, `incoai/Ornith-1.5-122B-DFlash2` exists (section 6). Original DFlash (v1) has no Ornith drafter and stays out.
- **MTP**: the weights ship (section 1.1). Deferred to M6, not dropped. DFlash2's acceptance length is far above a single MTP head's, so MTP may be dropped if DFlash2 works (decide after the DFlash2 measurement).
- **KV / long context**: fp8 paged KV as for the 27B; validated contexts are decided at M5.
- Every GPU job runs through `tools\with_lock.ps1 -Name gpu` and one at a time (dual-GPU sustained load has reset the display on this machine).


## 6. DFlash2 speculative decoding (planned, not started)

Source: `incoai/Ornith-1.5-122B-DFlash2` (Apache-2.0, `qwen3` architecture, 0.9 B parameters, BF16, ~1.8 GB). A block-diffusion drafter for
`ornith-ai/Ornith-1.5-122B`: block of 16 (15 drafted tokens per verification step), lossless (greedy output equals the target's). Downloaded 2026-10-21 to
`E:\models\r4dx\Ornith-1.5-122B-DFlash2` (6 layers, hidden 3072, 32 q / 8 kv heads x 128, sliding window 2048, target taps at layers
1, 7, 14, 20, 26, 32, 39, 45, two-tap conv group 16, selector rank 256 / top-k 16, mask token 248077; not yet checked against r4dx's
DFlash2 driver). Numbers below are from its model card (SGLang 0.5.21, two NVIDIA GB300, TP 2, Ornith default sampling, thinking on,
4096 new tokens), NOT measured on r4dx / R9700.

| Task | Acceptance length | Speedup at concurrency 1 |
|---|---:|---:|
| GSM8K | 8.33 | 2.27x |
| MATH-500 | 9.20 | 2.65x |
| HumanEval | 8.64 | 2.53x |
| MBPP | 6.37 | 1.98x |
| MT-Bench | 5.41 | 1.34x |

### 6.1 Expected value on r4dx (estimate, unmeasured)

A 16-row verify step in the MoE touches about 100 of 256 experts per layer (256 x (1 - (1 - 8/256)^16)), i.e. ~40% of the expert weights: roughly
20 GB read per step against ~5 GB for a single-token decode step. At the qwen4exp-measured efficiency that is ~70-80 ms per verify step against ~20 ms
for plain decode; with 5-9 accepted tokens per step the estimate is 70-120 tok/s, about 1.5-2.5x over a ~45 tok/s pipeline-split baseline. Low
confidence: it assumes the 16-row grouped expert GEMM runs at decent efficiency.

### 6.2 Work items (after M4 quantized container and the grouped M=1 decode kernel)

1. Download the drafter; convert to the r4dx DFlash2 container with the existing tooling (`docs/dflash2.md`); check its config against the driver
   (r4dx's current drafter is ~1.9 B and was trained for the Qwen3.8 27B: tap layers, hidden size, head layout, vocab).
2. Tap target hidden features from the Ornith layers the drafter was trained on (`Qwen35MoeStage` capture hook already exists).
3. Partial-accept rollback for the 36 GDN layers (the stage's GDN checkpoint exists but is untested for this use).
4. Grouped MoE kernel for M = 16 verify rows at 3072 / 1024 (new; the M=1 decode kernel does not cover it).
5. Driver wiring in `Qwen35MoeLocalTextModel`, server/CLI flags (`--dflash`), then a greedy-equality test against plain decode and an acceptance-length
   measurement on code / math / chat prompts.
6. Measure on r4dx; only then decide whether the MTP head (M6) is still worth building.

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
DFlash2 and MTP gains are workload dependent (chat ~1.3x, code and math 1.5-2.2x) and need the new verify-row kernels.

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

**Open items, in priority order (estimates):** P1 accuracy benchmark + 32k needle test (3-4 h); P2 prefill speed, grouped prefill MoE GEMM (6-12 h); P3 vision tower (4-8 h);
P4 w8a16 for class p (3-4 h, only if P1 shows a real loss); P5 DFlash2 (1-2 days); P6 `--tp 2` (2-4 days); P7 MTP (0.5-1.5 days); P8 K5 / host-RAM expert tier;
P9 qwen4exp regression check with its real container (the shared kernel code was edited; only a synthetic test ran).

**Operational notes.** The 48-layer quantization stalled three times (singular refit solve, a zero-Hessian expert, pruned-channel blow-up in `regularize`) and once more on an unexplained
process kill; all fixes are in `tools/quant2/qwen35moe/`. `E:` is the tight resource (about 74 GiB free; `oracle-k3` + `oracle-k4` hold 99 GiB and are only needed for re-allocation).
Live progress table: `tools\q35_progress.ps1`.
