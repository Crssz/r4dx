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
* **GPU time, scaled from a pilot (layer 0 of the real 4-layer slice `E:\models\r4dx\q35moe-real4`, 105,694-token corpus, HIP device 0 =
  the display GPU, through `pipeline.ps1` under the lock):** the Hessian pass of a layer took 40 s forward + 5 s finalize for 105k tokens
  (peak VRAM 18.3 GiB, RSS 5.4 GiB) -> **~6.5 min per layer at 1M tokens, ~5 h for 48 layers**. The K3 oracle of the layer took **18.3 min**
  (8 chunks of 32 experts at ~125 s each, 1.70 M encoder tiles per chunk = 1.44 x the raw tile count, encoder 19 k tiles/s, 13.6 k end to
  end; one 840 s slice restart included) -> **~14.6 h per K, ~29 h for K 3 + 4**, serialized on one GPU under the lock, ~34 h with the
  Hessians. The qwen4_exp runs had 23.7-28 k tiles/s on device 1; device 1 is faulted and device 0 also drives the display, so a healthy
  device 1 should be faster. `pipeline.ps1` runs one GPU job at a time.* Byte model (`alloc_q35.py`): all-KB3 experts + KB4 projections = 44.4 GiB (k = 24, 22.0 / 22.4 GiB per card), all KB4 = 57.9 GiB
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
ideally generate thinking-mode samples with Ornith itself (the model is meant to be used with thinking on).

## Option not built (a decision for the user)

* **Host-RAM expert tier.** The weight budget is per card (26 GiB x 2 = 52 GiB). If the coldest experts live in pinned host RAM (CPU
  computes them at decode, PCIe streams them at prefill; the abandoned Qwen3.8 port budgeted 11.5 GiB of the 64 GB), the budget grows by
  that tier and an all-KB4 mix (57.9 GiB) fits. The quantization side needs only an allocator option (`--host-gib`) and a placement list
  from `routing_counts.json` (per-expert routed counts of every layer are already recorded); the CPU-expert path is runtime work.

## Validation done (2026-10-21)

* **tiny-q35moe, end to end (`tiny_e2e.py --device cuda --gpu-device 0`, ~12 min):** 4-layer Hessians + checks, oracle K3 / K4 (205 tensors
  each, complete), verify (sha256, shapes, reconstruction), K3 > K4 strictly ordered on proxy / proxy_abs / rel_weight_err for all 205
  tensors and on the probe output error for all 128 expert groups, the fallback rule on a 6k-token corpus (10 of 16 experts, flags in the
  records), allocation under a scaled cap (141 tensors K4 / 64 K3, both cards within the cap), and **`r4dx-convert --trellis-from <mix>
  --trellis-verify full` accepts the mix** (1.4 s, `E:\models\r4dx\q35moe-tiny\tiny-mix.r4dx`, 562 MB).
* **Byte model vs that container:** per layer the container is 0.34-0.37% SMALLER than `alloc_q35.py` predicts (a constant ~0.47 MB: the
  model counts gate and up sign vectors twice and the tickets, which are not tensors); the tail (w4a16 lm_head + final norm) matches to the
  byte. So the 26 GiB cap is applied conservatively.
* **Weight-only KL (`tools/reference/qwen35moe/kl_mix.py`), tiny random network:** uniform K4 mean KL 0.98 / top-1 29%, uniform K3 2.46 /
  10%, the mix 1.05 / 27%. A RANDOM-weight network has incompressible, near-Gaussian weights and chaotic depth (bf16 vs fp32 alone moves its
  layer-3 output by 7%), so these numbers only show the chain computes and orders K3 > mix > K4 sensibly; they say nothing about Ornith.
* **One real layer (pilot, NOT the 48-layer run):** layer 0 of `E:\models\r4dx\q35moe-real4` (real shapes, 256 experts), 105,694-token
  corpus, through `pipeline.ps1`: Hessian capture, `check`, oracle K3 (771 tensors), `verify` 0 problems. Findings:
  * **Dead channels.** Ornith's layer 0 has intermediate channels that never fire (gate / up weights down to ~1e-36): an expert's `down`
    Hessian has ~880 of 1024 zero diagonal entries, the GDN `out` Hessian 45 of 8192. This is the data, not a capture bug, and the first
    version of the capture refused it. Now recorded (`capture.json` `dead_channels`) and only a Hessian with no live channel is refused.
  * **The oracle steers error into them:** `down` records have `rel_weight_err` 1.5 (median, max 4.9) while their H-weighted `proxy` is
    0.002: the LDLQ feedback puts the quantization error where H is zero. That is what the objective asks for and the 0.025 mean(diag)
    damping bounds it, but it makes the unweighted weight error useless as a quality signal for these tensors (the converter's full
    reconstruction check compares against the recorded value, so it still passes) and it is only safe if the calibration really covers
    what fires. **Check with the KL gate on held-out text (including non-English and code) before trusting the mix.**
  * 118 of 256 experts fell back to the diagonal gate_up Hessian only because this pilot corpus has ~3,300 routed rows per expert on
    average (threshold 3,072); at 1M tokens the average is ~31,000.
## Tests

`tools\reference\qwen35moe\test_ref.py` (CPU: the streaming reference equals a plain HF forward on the tiny model, bit for bit with the eager
experts); `tiny_e2e.py` runs the whole chain on tiny-q35moe (`--device cuda --gpu-device 0` for the oracle).
Note for the owner of `tiny_golden.py`: its docstring says "eager per-expert MoE loop", but HF 5.18's default `experts_implementation` is
`grouped_mm`, which is what the goldens ran (`load_hf_model` sets the wrong attribute). Eager and grouped_mm differ by ~1 bf16 ulp per
layer; the streaming reference uses eager.
