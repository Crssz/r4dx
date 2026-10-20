# qwen3_5_moe (Ornith-1.5-122B) trellis quantization tooling (docs/qwen35moe-plan.md M4)

Takes the BF16 checkpoint `ornith-ai/Ornith-1.5-122B` (250 GB, 62 shards, fused 3-D expert tensors) to a trellis container that fits
two 26 GiB cards: KB 3 / KB 4 chosen per (layer, class). It is `tools/quant2/qwen4exp/` ported to qwen3_5_moe; everything is per
layer, resumable, and every GPU step is one bounded process under `tools\with_lock.ps1 -Name gpu`.

```
corpus_q35.py         tokens_calib.json       hessian-v2 pieces + SFT slice, segments <= 4096, canonical Qwen3.5 tokenizer
hessian_q35.py        hessian\L{ii}\          per layer: .hess files, hessian.json, routing_counts.json, capture.json, DONE.json
(tools\reference\qwen35moe\)                  + the next residual boundary work\b{ii+1}.bin
oracle_q35.py         oracle-k3\ oracle-k4\   per layer: L{ii}.{p,e00..}.safetensors + .json chunks, weights_override.json
alloc_q35.py          mix\                    mix manifest (one KB per (layer, class)), allocation.json, allocation_report.md
(tools\reference\qwen35moe\)
disk_budget.py        (stdout)                disk + GPU-time budget of the chain
pipeline.ps1          the driver              -Stage corpus|hessian|oracle|verify|alloc|convert|budget|all -Layers a..b [-Run]
tiny_e2e.py           the chain on tiny-q35moe
```

Reference side, `tools/reference/qwen35moe/`: `arch.py` (tensor table from `config.json`; experts under their VIRTUAL per-expert names,
`...mlp.experts.{e}.{gate,up,down}_proj`, which the converter's `FusedExpertsOverlay` also uses), `ref.py` (`Qwen35MoeStreamingReference`:
HF's own `Qwen3_5MoeDecoderLayer` built on the meta device and filled one layer at a time from the shards, a hook-carrying copy of the
eager experts loop, `ReferenceHooks` = the callback API the Hessian capture uses), `tiny.py` / `tiny_golden.py` (the random checkpoint
and goldens, not mine), `test_ref.py` (CPU test).

Paths are under `E:\models\r4dx\q35moe\` (`-Root`); the checkpoint is `E:\models\r4dx\Ornith-1.5-122B`. One interpreter for everything:
`D:\venvs\r4dx-qwen4-ref` (transformers 5.18, torch 2.9.1+rocm). The old oracle needed a MAGMA venv; `trellis_quant.block_ldl` now factors
on the CPU for GPU tensors, so it does not.

## Run it

```powershell
.\tools\quant2\qwen35moe\pipeline.ps1 -Stage all -Layers 0..47            # prints every command and the budget, runs nothing
.\tools\quant2\qwen35moe\pipeline.ps1 -Stage budget                       # just the disk / GPU-time budget
.\tools\quant2\qwen35moe\pipeline.ps1 -Stage all -Layers 0..47 -Run       # layer-major: hessian -> oracle K3 -> oracle K4 -> verify
.\tools\quant2\qwen35moe\pipeline.ps1 -Stage alloc -Run                   # needs both oracle dirs complete
.\tools\quant2\qwen35moe\pipeline.ps1 -Stage convert -Run                 # needs the mix (kvcalib.json optional, see below)
```

`-Device` is the HIP device of the GPU steps (default **0**: physical device 1 has been faulted since 2026-10-21; device 0 gets
`R4DX_ALLOW_GPU0=1`). A GPU step is `-BudgetSeconds` (default 840) of work, then the process saves its state and exits 75; the driver
releases the lock and takes it again for the next slice.

## Budget (from `disk_budget.py`, 1M-token corpus, 48 layers, K 3 + 4)

* **Scratch per layer step: 27 GiB** (two residual boundaries 2 x 5.7 GiB at [1M, 3072] bf16, accumulator state 10.0, packed Hessians 5.1,
  probe rows 0.4); the driver refuses to start a layer below `-MinFreeGiB` (40).
* **Kept:** oracle-k3 42.5 GiB + oracle-k4 56.5 GiB = 99 GiB; the container ~53 GiB (both cards at the cap + the 1.4 GiB bf16
  embedding table). Chain end: ~152 GiB. **`E:` had 148-193 GiB free (it fluctuates): put `-Root` on `D:` (349 GiB free) or free
  space first.** The checkpoint stays on `E:`.
* **GPU time (estimate, not measured at this size):** the oracle encoder is ~469 M raw tiles per K; at the qwen4_exp throughput
  (24k tiles/s, 1.7 work per raw tile) that is **~9 h per K, ~18.5 h for K 3 + 4**, one GPU, serialized under the lock. The Hessian
  passes (48 layers x 1M tokens, fp32 GEMMs per expert, GDN through the torch fallback) are not estimated; time layer 0 first.
* Byte model (`alloc_q35.py`): all-KB3 experts + KB4 projections = 44.4 GiB (k = 24, 22.0 / 22.4 GiB per card), all KB4 = 57.9 GiB
  (does not fit 2 x 26 GiB). So the mix has ~7.6 GiB of promotions to spend.

## The pieces (differences from qwen4_exp)

**Hessians** (`hessian_q35.py capture`). Layer L alone runs over the corpus: `Qwen35MoeStreamingReference._build_layer(L)` (the fused expert
tensors, 4.8 GB, live on the device for the layer), `ReferenceHooks.linear_input` on every linear. Taps, one Hessian per distinct input:
GDN `in` (qkv, z; K 3072; the equality of z's input is asserted), `out` (K 8192); attention `in` (q, k, v) and `o` (K 8192); per expert `gu`
(gate and up: ONE Hessian, K 3072) and `down` (the silu*up activation, K 1024). Not quantized, no Hessian: router, shared expert (+ its gate),
in_proj_a/b, norms, conv, A_log / dt_bias (the converter keeps them bf16/fp32). Fallback rule (Q16): an expert with fewer routed rows than
the input dimension (3072 / 1024) gets a diagonal Hessian from layer-wide second moments. Accumulators: 256 x (3072^2 + 1024^2) fp32 = 10 GiB on
the device; a seeded reservoir of 256 routed rows per expert is the probe sample for the output-referred error.

**Oracle** (`oracle_q35.py`). `trellis_quant.quantize_group` per shared-Hessian group, `matched` basis. Expert gate and up are one group (the
converter makes them one two-part container linear), down another; projections: GDN {qkv, z} + {out}, attention {q, k, v} + {o}. Weights are
read out of the fused tensors by `CheckpointReader.linear_weight` (virtual per-expert names). The manifest keys are the HF per-expert names
the converter's overlay presents, so `--trellis-from` needs nothing new.

**Allocation** (`alloc_q35.py`). Classes per layer: `p`, `gu`, `down`; ONE KB per class (the container refuses a class with mixed layouts or
KBs), start at the floor (p KB 4, experts KB 3), promote by output-referred loss reduction per GiB while both cards stay <= `-CapGib`.
Bytes = what `Qwen35MoeContainer::Inspect` counts: trellis words + fp32 sign vectors + tickets, norms, router, shared expert (bf16),
GDN/attention small tensors, final norm + w4a16 g32 lm_head on the last card; the embedding table is on the host.

**Convert.** `r4dx-convert --input <Ornith> --output ... --trellis-from <mix> --trellis-manifest-sha256 <sha> --trellis-verify full
--layouts w4a16 --no-bf16 --lm-head w4a16 --w4a16-group-rule "^lm_head$=32" [--kv-calib kvcalib.json]`. Without a KV calibration every
k/v descale is the placeholder 1.0 (the tiny container is built that way); calibrate before shipping. The vision tower and MTP head are
skipped by the converter (plan M6).

## Corpus caveat

On this machine (2026-10-21) `corpus_q35.py build` finds only calib.txt, kv_calib_corpus/ and the repo code windows: **no WikiText-2, no
corpus-v2 generated samples, no agent SFT parquet** (the old qwen4_exp `q4exp\` data is gone). Missing sources are skipped with a warning
and recorded in `tokens_calib.meta.json`; the resulting corpus is far below 1M tokens and has no agent / thinking traces. Before the real run:
fetch the SFT parquet (`hf download ukisai/Qwen3.8-27B-multi-turn-agent-sft --repo-type dataset --local-dir <Root>\sft`), WikiText-2 raw, and
ideally generate thinking-mode samples with Ornith itself (the DFlash2 drafter, plan 5.1, is judged on thinking traces).

## Option not built (a decision for the user)

* **Host-RAM expert tier.** The weight budget is per card (26 GiB x 2 = 52 GiB). If the coldest experts live in pinned host RAM (CPU
  computes them at decode, PCIe streams them at prefill; the abandoned Qwen3.8 port budgeted 11.5 GiB of the 64 GB), the budget grows by
  that tier and an all-KB4 mix (57.9 GiB) fits. The quantization side needs only an allocator option (`--host-gib`) and a placement list
  from `routing_counts.json` (per-expert routed counts of every layer are already recorded); the CPU-expert path is runtime work.

## Tests

`tools\reference\qwen35moe\test_ref.py` (CPU: the streaming reference equals a plain HF forward on the tiny model, bit for bit with the eager
experts); `tiny_e2e.py` runs the whole chain on tiny-q35moe (`--device cuda --gpu-device 0` for the oracle).
Note for the owner of `tiny_golden.py`: its docstring says "eager per-expert MoE loop", but HF 5.18's default `experts_implementation` is
`grouped_mm`, which is what the goldens ran (`load_hf_model` sets the wrong attribute). Eager and grouped_mm differ by ~1 bf16 ulp per
layer; the streaming reference uses eager.
