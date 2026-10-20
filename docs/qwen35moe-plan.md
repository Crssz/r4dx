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
- **The MTP head ships with the checkpoint**: `mtp.layers.0.*` (a full-attention block with an MoE FFN whose experts are PER-EXPERT `experts.{E}.{gate,up,down}_proj.weight`, unlike the main model), `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`, `mtp.fc`, `mtp.norm`. MTP is therefore possible (M6). DFlash2 is possible too: a trained drafter has been released (section 5.1).
- Vision: `model.visual.patch_embed.proj`, `pos_embed`, `blocks.{0..26}.{norm1,norm2,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}`, `merger.{norm,linear_fc1,linear_fc2}`.

## 2. Fit on two R9700 (32 GB each)

BF16 is 250 GB, so the weights must be trellis-quantized (as the 27B and qwen4_exp are). Experts at KB 3 are ~43.5 GB,
at KB 4 ~58 GB; the non-expert part at ~4.5 bits is ~5 GB. With the qwen4_exp budget of at most 26 GiB of weights
per card plus KV, the target is a **KB 3 / KB 4 mix by (layer, class)** of about 3.3-3.6 bits for experts, split by
layer over both cards. Expect ~50-52 GiB total. One card cannot hold it.

### 2.1 Quantization pipeline numbers and the host-RAM option (2026-10-21)

The pipeline (`tools/quant2/qwen35moe/`, README there) budgets weights at 26 GiB per card. Its byte model (matching
`Qwen35MoeContainer::Inspect`): all-KB3 experts + KB4 projections = 44.4 GiB (k = 24), all KB4 = 57.9 GiB, which does NOT fit
2 x 26 GiB; the allocator spends the ~7.6 GiB between them on the best loss-per-GiB promotions. Oracle encoder estimate ~9 h per K on
one GPU; kept data ~99 GiB (oracle dirs) + ~53 GiB (container).

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
| grouped on-GPU decode MoE | `moe_trellis_decode.hip` | instantiated only for hidden 2560 / inter 640 / <=16 experts / bits 3-4: add 3072 / 1024 instantiation (speed work, milestone 2) |
| layer-split pipeline | `Qwen4ExpModel` / `Qwen4ExpStage` pattern, `ChooseSplit` | new `Qwen35MoeStage` without HC/PLE/QSA; no sidecar |
| container, trellis GEMM, tokenizer, server | shared | dialect `qwen3_5` (same Qwen3.5 template family) **VERIFY** the chat template hash against the 27B's |
| converter | `r4dx-convert` qwen4_exp branch | new branch for `model_type: qwen3_5_moe`; checkpoint has per-expert or fused 3D expert tensors **VERIFY** from `model.safetensors.index.json` |

## 4. Milestones

- **M0 (download running; index read, section 1.1)** download to `E:\models\r4dx\Ornith-1.5-122B`.
- **M1 config + arch (done 2026-10-20, CPU only)** `Qwen35MoeConfig` (`src/model/qwen35moe_config.h`), `Arch::kQwen35Moe`, `LoadTextModel` refuses it with a clear message until M3, `tests/model/test_qwen35moe_config.cpp` passes. The existing test that expected a Qwen3.5-MoE `model_config` to be `kQwen35` was updated. No dialect change needed: the arch name `qwen3_5_moe` already resolves to the Qwen3.5 dialect.
- **M2 tiny golden** a 2-layer, few-expert `qwen3_5_moe` built with HF transformers (>= 5.8.1, `D:\venvs\r4dx-qwen4-ref`) giving per-layer goldens, as `tools/reference/qwen4exp/tiny_golden.py` does.
- **M3 model** `Qwen35MoeStage` / `Qwen35MoeModel` on bf16 tiny container; layer-isolated tests against the goldens; then two-device split.
- **M4 converter + quantization** Hessians per layer from an HF reference forward (layer-streaming, 250 GB does not fit RAM), trellis oracle at KB 3 and KB 4, allocation, `r4dx-convert` branch. This is the long, GPU-hours stage; reuse `tools/quant2/qwen4exp/` scripts with the Qwen3.5 layer builder.
- **M5 end to end** CLI/server chat on the real container, KL gate vs HF bf16 (mean KL <= 0.02, top-1 >= 93%, the qwen4_exp bar), thinking + tool calls through the Qwen3.5 template.
- **M6 parity features** vision tower for the new sizes (reuse `src/vision`), `--tp 2` MoE sharding, MTP head, grouped GPU MoE decode, prefill int8 path.
- **M7 DFlash2 (added 2026-10-21)** download `incoai/Ornith-1.5-122B-DFlash2`, read its config, extend the runtime with target hidden taps and the drafter, verify-path acceptance gate on the KL corpus, speedup measured on 2 x R9700 (section 5.1). After M5; independent of the quantization pipeline except for the thinking-mode calibration slice.

## 5. Out of scope / decisions

- **DFlash2**: in plan since 2026-10-21, see 5.1 (M7). The earlier "no drafter exists" is obsolete.
- **MTP**: the weights ship (section 1.1). Deferred to M6, not dropped.
- **KV / long context**: fp8 paged KV as for the 27B; validated contexts are decided at M5.
- Every GPU job runs through `tools\with_lock.ps1 -Name gpu` and one at a time (dual-GPU sustained load has reset the display on this machine).

### 5.1 DFlash2 drafter (added 2026-10-21; milestone M7)

`incoai/Ornith-1.5-122B-DFlash2` (Apache-2.0, 0.9 B parameters, BF16 safetensors) is a trained DFlash 2 drafter for this exact
target: a block-diffusion drafter that proposes a block of 16 tokens (15 drafts per verify step) in one pass, keeps the top
candidates per position and lets a light selector trace one path; its backbone uses two-tap dynamic convolutions (the
`build_dflash2_conv` / `build_dflash2_selector` structure `docs/dflash2.md` already documents for the Huihui/Qwen3.8 drafters).
Lossless: greedy output equals the target's, sampling keeps its distribution.

Published numbers (SGLang 0.5.21, 2 x GB300 TP2, 128 samples/task, 4096 new tokens, T 0.6, top-p 0.95, top-k 20, thinking on),
acceptance length / speedup over autoregressive decoding at concurrency 1: GSM8K 8.33 / 2.27x, MATH-500 9.20 / 2.65x,
HumanEval 8.64 / 2.53x, MBPP 6.37 / 1.98x, MT-Bench 5.41 / 1.34x. These are datacenter-GPU, batch-1 figures; on two R9700 with
a trellis target and PCIe-only transfers the speedup is a measurement to make, not to quote.

What the model card does not say, and M7 must read from the downloaded `config.json` and tensor index (**VERIFY**): layer count and
width, which target layers it taps (`target_layer_ids`), `mask_token_id`, the tensor naming, and whether it reuses the target's
embedding / lm_head (0.9 B leaves no room for its own 248320 x 3072 head pair at 1.5 GB each, so expect it to share them).

Consequences for the plan:
- **Target hidden taps.** The drafter conditions on target hidden states from specific layers. The runtime must be able to export
  the residual at those layers during prefill and verify (a capture point in `Qwen35MoeStage`, alongside the existing
  `capture_active_` hook). With the layer-split over two cards the taps may sit on either card; the drafter lives on one.
- **Quantization is unaffected** except for calibration: the corpus should include thinking-mode traces, since the drafter is
  evaluated with thinking on and accepted-token statistics depend on the target's distribution there. The drafter itself ships
  in BF16 (1.8 GB) and is not quantized in M7 v1; it adds ~1.8 GB to the card that hosts it, to be budgeted against the 26 GiB cap.
- **Reuse.** `src/model` already has the Huihui DFlash2 path (drafter loader, `r4d_dflash_conv` kernel, `docs/dflash2.md`). M7 is a
  config adapter plus the tap points and a verify path for the MoE target (verify processes 16 tokens per step through all 256
  experts' routing, so the grouped on-GPU MoE decode matters more here than for AR decode; see M6).
- **Converter.** The existing GGUF-based draft container (`src/convert`) takes a GGUF; this drafter is a HF safetensors repo, so
  either convert it to the GGUF layout first (ROCmFPX `conversion/qwen.py` `DFlashModel`) or add a safetensors source. Decide at M7.
