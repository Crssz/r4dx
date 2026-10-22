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
## Singular refit (found in the 48-layer run, 2026-10-21; fixed in `refit_fallback.py`)

The K4 oracle of layer 2 died in `trellis_quant.refit_scales` (`torch.linalg.solve ... input matrix is singular`) after ~5,000 solves
had passed (layers 0-1 at K3 and K4, layer 2 at K3). Findings, all measured on the real layer-2 Hessians and K4 bits:

* **Cause: dead channels, but through conditioning, not a zero diagonal.** The row refit solves `A = (Q Q^T) o H_damped + 1e-6 mean(diag A) I`.
  H is damped (+0.025 mean(diag H) I), so no diagonal is zero; but an expert's `down_proj` input has up to 1014 of 1024 dead channels, so A is a
  tiny live block plus a damped diagonal and `mean(diag A)` (which sets the ridge) is dominated by dead rows. For the 256 `down` systems of
  layer 2: condition number 5e6 .. 8.5e8, median 4e7, **219 of 256 above 1/eps(fp32)** (1.7e7); the smallest eigenvalue of A before the ridge is
  ~1e-25 relative, numerically 0 in float64. Not singular in exact arithmetic (the ridge keeps it positive), but an fp32 pivoted LU can hit an
  exactly zero pivot: a knife edge.
* **It does not reproduce deterministically.** Re-running layer 2 / K4 in full (projections + 8 expert chunks, same inputs, device 1, cohort 32)
  raised nothing (`E:\models\r4dx\q35moe-repro`), the chunks both runs finished are bit-identical (all 288 + 9 tensors, all records), and 400 fp32
  solves of the worst systems on the GPU raised nothing either. So the raise depends on rounding that a re-run did not repeat; the cause is
  the ill-conditioning, the trigger is not pinned down further.
* **Fix.** `refit_fallback.install(tq)` (called by `oracle_q35.py quantize`) wraps `tq.refit_scales` so that `torch.linalg.solve` is
  `solve_with_fallback`: the first attempt is the original call (same bits when it works), only a raise goes down the ladder: float64 LU on the
  CPU -> the same system with a ridge of 1e-4 then 1e-2 x mean(diag A) -> minimum-norm `lstsq` (gelsd). Every use is in the record
  (`refit[round].fallback = {stage, error, device, k}`, `refit_fallbacks` = rounds, a WARNING line on stdout) and `alloc_q35.py`'s report
  counts them per class. `trellis_quant.py` is NOT edited: its sha256 (and trellis_viterbi.hip's) is part of the oracle's job key, so any
  edit would make every finished chunk stale.
* **Existing chunks stay valid and reproducible, verified:** re-running layer 2 / K4 chunks `p` and `e00` with the fixed oracle gives
  tensors and records identical to the original run's (288 / 288, 96 / 96), no fallback key. With every `down` solve forced to raise (the real
  threaded cohort path), the 32 forced groups are flagged, their trellis words are identical, proxies move by <= 1.3e-7 relative, sign vectors
  differ slightly (fp32 vs fp64 solve) and nothing else changes.
* **Line endings.** `code_sha256` hashes the working-tree BYTES: the main checkout has CRLF files, a worktree has LF (`.gitattributes` eol=lf),
  so the same source hashed differently and a chunk written from one checkout looked stale from the other. `oracle_q35.jobs_match` now
  accepts a stored hash equal to the as-is, LF or CRLF form of the current file (a changed source still mismatches).

Could it hit again? **Yes, anywhere the refit system is ill-conditioned; the fix covers every group the oracle quantizes.**
* K3 on later layers: yes (A depends on Q and H, not on K; K3 passed 5,000 solves on layers 0-2 by luck of rounding, and dead channels are
  common in the early layers).
* Expert `down_proj` groups: the worst class (cond up to 8.5e8 on layer 2). Expert `gate_up`: cond ~1e4 (one dead hidden channel), safe.
* GDN `out_proj` (k = 8192, 709 dead channels in layer 2): cond(A) = 2.3e8, also fragile. GDN `in` (qkv, z): 1.7e4, safe. Full-attention
  q/k/v/o (layers 3, 7, ...): not measured yet; `o_proj` (input = attention output x gate) can have dead channels like `out_proj`.
* Shared experts: not quantized (bf16 in the container), no group, no refit.
## Experts without calibration evidence, null groups (found at layer 3, 2026-10-21)

The layer-3 Hessian capture was refused: `non-finite Hessians, or Hessians without a live channel ... L03.e0072.down.hess, e0166, e0180, e0252`.
* **Which condition:** "no live channel", not non-finite. The four `down` Hessians are finite and EXACTLY zero (diagonal max 0.0, header
  trace 0.0); the `[0, 1, ..., 7]` in the message was just the first eight channel indices of the (truncated) list of non-positive channels,
  i.e. all 1024 of them.
* **Why they fall outside the Q16 fallback:** the fallback is for experts with FEWER routed rows than the input width. These four are the
  busiest experts of the layer: 63,208 / 66,225 / 55,076 / 59,893 routed rows (median 28,292), none "never routed".
* **Hypotheses falsified, on the real layer-3 data (probe rows + weights):** (a) never fires on this corpus: no, 55-66k rows each;
  (b) bf16 overflow in silu * up: no, no NaN/inf anywhere, `h` is exactly 0 in fp32 and in bf16; (c) accumulator bug with few rows: no, the
  same code gives live Hessians for the neighbours (expert 71: 28,984 rows, trace 0.30) and these experts' own gate_up Hessians are live
  (trace 741, the input `x` is normal, |x| max 14).
  **Cause: the checkpoint.** All three weight matrices of these experts are ~0: mean |w| 1.8e-35 (e72), 3.4e-34 (e166), 1.5e-26 (e180), 4.1e-31
  (e252); gate pre-activations ~1e-32, so silu(gate) * up underflows to exactly 0. They are pruned experts the router still sends tokens to
  (their output is below bf16 resolution of anything it is added to). Expert 71 shows the same pruning at channel level (511 of 1024
  gate rows ~0).
* **Policy (all logged):**
  1. `hessian_q35.py`: a NaN / inf Hessian FAILS the layer, naming the expert (`expert_hessian_usable`). An expert Hessian with no positive
     diagonal entry is "no evidence": the layer-wide diagonal of the Q16 fallback is written instead, the expert is listed in
     `routing_counts.json` / `capture.json` `no_evidence` (`gate_up`, `down`, and `dense` for a mixer tap, which gets a data-free identity)
     and in `fallback`. A Hessian with some live channels and some dead ones is untouched (dead channels are normal here).
  2. `oracle_q35.py`: a group whose weights are negligible (largest |w| < 1e-12; w^2 underflows fp32, so the regularize / scale search
     cannot run on them) is a NULL GROUP: zero words, su = 1, sv = 0, the stored linear reconstructs to exactly 0, `rel_weight_err` = 1.0
     (what r4dx-convert's double-precision check computes; `tq.rel_cos` would say 0.0 here), record key `null_group`, `proxy_abs` 0. It
     keeps the class's K and shapes, so the container stays uniform. A group with normal weights but no evidence would be quantized against
     the layer-wide diagonal (recorded `hessian.no_evidence`, `fallback`).
  3. `alloc_q35.py`'s report counts, per class and per layer, the tensors quantized without evidence and the null groups.
* **Checks:** layer 3 re-finalized from its saved accumulator state (scratch root `q35moe-fix3`): `no_evidence` down [72, 166, 180, 252],
  `hessian_q35.py check`: 0 problems; the oracle (K4, K3) of that layer, including the first full-attention q/k/v/o groups, ran through
  (numbers below); r4dx-convert's full reconstruction check accepts null groups (tiny mix with an expert replaced by null groups:
  `rel 1 (oracle 1) OK`).
* **Could the refusal hit again?** Layer-3 attention Hessians (`in`, `out`) have no dead channel at all. Later layers: a dense tap with
  an all-zero Hessian (a dead mixer) is now an identity-Hessian + `no_evidence.dense` entry instead of a stop; a NaN / inf still stops.

## Garbage groups: the `+ 1e-10` cancellation in regularize (found while checking layer 3, affects layers 0-2)

Looking at the layer-3 K4 records I found four expert `down_proj` groups with proxy 0.6 - 0.83 and `rel_weight_err` 1.2 - 1.7 (healthy: proxy
< 0.02, rel 0.04 - 0.4 where half the rows are dead), all with the global scale `g` at its floor 0.010. They were NOT calibration problems:
they reproduce with an identity Hessian, in the weights alone.
* **Cause.** `trellis_quant.regularize` divides every column of W^T by `c * sign + 1e-10` and every row by `r * sign / -1.2437 + 1e-10`. Ornith's
  pruned experts keep channels with weights 1e-12 .. 1e-30 (not exact zeros, which regularize handles). When such a channel's scale lands
  within a factor ~2 of 1e-10 and the random sign (seeded per tensor) is the wrong one, the divisor cancels to ~0 and the channel is blown up
  by up to 1e10 (layer 3 expert 108: row rms up to 179 against 1.24); the scale search then picks g = 0.01 and the whole tensor is quantized
  to noise. Which tensors are hit depends on the seed, so neighbours of the same expert are fine. Reproduced synthetically (tests): 4 of 8
  seeds fail on weights with log-uniform 1e-14.5 .. 1e-10 channels, none after the fix.
* **How many.** Finished oracle records of layers 0-2 (before the fix): proxy > 0.05 on K3: layer 0: 26 of 771 tensors, layer 1: 60, layer 2: 19;
  on K4: 12, 20, 10; `g <= 0.011` on 1-3 tensors per layer and K (the `down` signature), the others (`gu` groups with 92% pruned output rows,
  `g` 0.3, rel ~1.0) are the same failure on the column side. Together about 4% of the tensors of those layers.
* **Fix (oracle only, `trellis_quant.py` untouched).** `oracle_q35.clean_negligible_channels`: before quantizing, output rows / input columns
  of the HF weight whose largest |w| is below `--clean-below` (default 1e-8; live weights are ~1e-3 and bf16 resolves 4e-3 relative) are set
  to exactly 0, which takes regularize's dead-channel path. A tensor without such channels is returned unchanged. A group that is nothing
  but negligible channels becomes a null group. The record gets `cleaned_channels {out, in, below}`.
* **Verified on the real layer 3, K4 (same Hessians, same seeds):** tensors with proxy > 0.3: 4 -> 0 (max proxy 0.828 -> 0.007, the four
  experts 174 / 85 / 126 / 108: proxy 0.83 / 0.77 / 0.72 / 0.60 -> 0.0012 / 0.0022 / 0.0016 / 0.0011, `g` 0.010 -> 0.56 .. 1.21); medians per
  class unchanged (p 0.00056, gu 0.00031, down 0.00123 -> 0.00117); verify 0 problems.
* **Consequence: layers 0-2 must be redone.** The cleaning changes the numerics of any group with such channels (756 of 772 layer-3 records
  have some), so the job key carries `clean_channels_below` and every chunk written before is stale; `--clean-below 0` restores the old
  key (and the old garbage). Layers 0-2 had their Hessians deleted by `verify`: `pipeline.ps1` now notices (`.hess` files gone and
  `oracle_q35.py status` says the chunks are not usable) and captures them again -- from layer 0, because the boundaries between layers were
  deleted too. Cost: 3 x ~7 min of Hessians + 3 x (K3 ~14 + K4 ~13 min) of oracle.
* **So it cannot happen silently again:** the oracle prints `WARNING ... POOR group` (proxy > 0.3 or g <= 0.05) and `alloc_q35.py`'s report
  counts POOR groups in the chosen mix.
## Two lanes (`-Lanes 2`): K3 and K4 on two GPUs

```powershell
.\tools\quant2\qwen35moe\pipeline.ps1 -Stage all -Layers 0..47 -Lanes 2            # the plan: both lanes' commands, nothing runs
.\tools\quant2\qwen35moe\pipeline.ps1 -Stage all -Layers 0..47 -Lanes 2 -Run      # supervisor + lane A + lane B (detach the supervisor)
```

| | lane A | lane B |
|---|---|---|
| HIP device / lock | `-DeviceA 1` / `-LockA gpu` | `-DeviceB 0` (`R4DX_ALLOW_GPU0=1`) / `-LockB gpu-dev0` |
| per layer | wait for verify of layer i-`MaxInFlight` -> hessian -> check -> oracle K3 | wait for A's `hess-LNN.ready` -> oracle K4 |
| first thing | | `gpu_probe.py` (responsive, >= `-ProbeMinFreeGiB` 14 GiB free) |

The supervisor (the process you start) wipes `<Root>\lanes\` and `FAILED.txt`, starts the two lane processes hidden with their output in
`<Root>\logs\lane-A.out.log` / `lane-B.out.log`, prints a status line every 10 minutes, and runs alloc + convert when both lanes finished
every layer. Lane processes are ordinary `pipeline.ps1 -Stage all -Lane A|B -Run` with the supervisor's parameters forwarded; each step is
one 14-minute slice under its lock, as in single-lane mode (default `-Lanes 1` is unchanged).

**State.** Everything is re-derived from the oracle directories (`oracle_q35.py status`, `.hess` files, `DONE.json`), so any interruption
resumes by running the same command again; the marker files in `<Root>\lanes\` only coordinate the two live lanes:
`hess-LNN.ready` (A -> B), `kK-LNN.done`, `verify-LNN.claim` / `.done`, `STOP`, `lane-A.status` / `lane-B.status` (one line: time, what it
is doing). **Verify** (CPU, deletes the layer's Hessians) runs once per layer, by whichever lane finishes the layer second: both K done ->
create `verify-LNN.claim` with `FileMode.CreateNew` (atomic: exactly one lane gets it) -> run verify -> `verify-LNN.done`. Tested: 12 layers,
two jobs each finishing at random offsets, verify ran exactly 12 times. **Disk:** lane A starts the Hessians of layer i only when layer
i-`MaxInFlight` (2) is verified, so at most two layers of Hessians (2 x 5 GiB) plus one capture's scratch (27 GiB) exist; layers already
complete from an earlier run are passed by both lanes in seconds (`status`).

**Failure.** A failing step in either lane (non-zero exit; a prerequisite refusal; a display-driver reset event) writes
`<Root>\FAILED.txt` -- one line, `... FAILED lane=B step=oracle-k4-L17: exit 1 after 4.2 min: <last error line> [log ...]`, the FIRST failure of a
run stays -- and `lanes\STOP`. **Both lanes stop** (my choice: a failure is usually deterministic and also hits the other lane, and a stop
on the display GPU must not leave the other card running unattended): the other lane finishes its current slice (<= 14 min), notices STOP
between slices, exits with code 3 and leaves resumable state; nothing is killed. The supervisor then exits non-zero.

**Display safety (lane B is on the GPU that drives the display).** The probe runs first (lane B refuses to start if device 0 is
unresponsive or has < 14 GiB free). After every lane-B slice the driver looks in the System event log for display-driver reset events
(Display / amdkmdag / amdwddmg, ids 4101, 117, 141) since the lane started and stops both lanes if there is one. A HIP error in the oracle
process is a non-zero exit -> FAILED.txt. Nothing retries: the only loop is "slice returned 75 = more work", as before.

**Locks.** `tools\with_lock.ps1` accepts `-Name gpu-dev0` (and `gpu-test`). Lane B holds `gpu-dev0` only, lane A holds `gpu` only, so the
two never wait for each other; any OTHER job that uses device 0 should take `gpu-dev0` (a job on device 1 keeps taking `gpu`).

**Balance.** Per layer, lane A needs ~7 min of Hessians + ~14 min of K3 on device 1 (~21 min), lane B ~13-16 min of K4 on device 0, so lane A
is the bottleneck: ~21 min per layer, ~12 h for 34 layers instead of ~19 h. Putting the Hessians on lane B instead would balance them at
~14 vs ~20 min (no gain) and put the 18 GiB capture on the display GPU, so they stay on A.
**CPU contention, measured (2026-10-22, with the main single-lane run live on device 1 as "lane A").** A real layer-3 K4 oracle chunk of 32
experts on device 0 at the same time as the main run's K3 chunks on device 1 (the two-lane situation; `block_ldl` and the cohort threads
share the 16-core / 32-thread CPU): the lane-B chunk took 114.7 s (23.6 k encoder tiles/s) at `--cohort 32`, 129.2 s at `--cohort 16`, 115.5 s at
`--cohort 8`, against 92-98 s (28.4 k tiles/s) for the same K4 chunk running alone on device 1; the main run's K3 chunks went from ~110 s to
113-117 s (about 5% slower). CPU load during the overlap averaged 41% (peaks 91-94% while several groups factor on the CPU at once). So the
second oracle costs the first ~5% and itself ~15-20% against a lone run, and lowering the cohort does not help (8 = 32, 16 is worse):
`-CohortB` exists but the default (= `-Cohort`, 32) stays. Per layer that is lane A ~7 min Hessians + ~16 min K3 and lane B ~16 min K4 ->
~23 min per layer, ~12.5 h for the remaining ~33 layers. Lane B idles ~7 min per layer; splitting K3 chunks between the lanes (both work on a
layer's chunks from opposite ends) would use that and reach ~19 min per layer, but chunks of one K are written by one process at a time today,
so it is not built.
## Work queue (`-Workers 2`): both GPUs claim whole-layer oracle jobs (preferred over `-Lanes 2`)

```powershell
.\tools\quant2\qwen35moe\pipeline.ps1 -Stage all -Layers 0..47 -Workers 2            # the plan (what each worker runs; who gets which layer is decided at run time)
.\tools\quant2\qwen35moe\pipeline.ps1 -Stage all -Layers 0..47 -Workers 2 -Run      # supervisor + worker G1 + worker G0 (detach the supervisor)
```

| | worker G1 | worker G0 |
|---|---|---|
| HIP device / lock | `-DeviceA 1` / `-LockA gpu` | `-DeviceB 0` (`R4DX_ALLOW_GPU0=1`) / `-LockB gpu-dev0` |
| work | the Hessian chain FIRST (layer L needs the residual boundary of L-1), otherwise an oracle job | oracle jobs only; `gpu_probe.py` first |
| oracle slice | `-OracleSliceG1` 420 s (so the chain waits at most ~7 min) | `-BudgetSeconds` 840 s |

**The queue.** A job = one layer's oracle, K3 then K4, by one worker. A layer is claimable once the chain has marked its Hessians ready
(`hess-LNN.ready`); a worker takes the LOWEST ready, unclaimed, unverified layer by creating `oracle-LNN.claim` with `FileMode.CreateNew`
(atomic: tested, 12 layers, 3 concurrent claimers, every layer exactly once). A worker that holds a job still gives the chain priority: G1
checks, between two oracle slices, whether the next chain layer may start (at most `-MaxInFlight` = 3 layers of Hessians in flight: the
chain waits for verify of layer i-3), runs its Hessian capture to the end, then resumes its job (the oracle resumes at chunk granularity).
When the job is done the same worker refreshes the manifests and runs verify (CPU, deletes the layer's Hessians), then claims the next layer.

**Manifests.** Workers on different layers share one `oracle-kK` directory, whose `weights_override.json` is merged from every chunk file. The
oracle therefore runs with `--no-manifest` in this mode, and each finished layer refreshes both manifests (`oracle_q35.py manifest`, CPU,
~1 min) under a machine-wide mutex (`Global\r4dx-q35moe-manifest-<root>`); the manifest's temporary file is also private to the process now
(`atomic_manifest`). The supervisor refreshes them once more before alloc. Chunk files are per layer (`LNN.<chunk>.json`), so nothing else is shared.

**Everything else as in lane mode:** the supervisor wipes `<Root>\lanes\` and `FAILED.txt` and starts the two worker processes hidden (logs
`<Root>\logs\worker-G0.out.log` / `worker-G1.out.log`); a failure in either worker writes `FAILED.txt` with `worker=G0|G1` (the first
failure stays) and `STOP`, the other worker stops at its next slice boundary, nothing is killed; the display GPU is probed first and checked
for driver-reset events after every G0 slice; no blind retries; resumable by running the same command (state is re-derived from the oracle
directories; `-Layers` already complete are passed in seconds); default `-Workers 1` / `-Lanes 1` is the single-lane run, unchanged.

**Why this beats the K3 / K4 lane split.** Per layer the GPU time is ~7 min of Hessians + ~15-16 min of K3 + ~14-15 min of K4 (device 1; device 0
a little slower, and both ~5-15% slower while the other GPU runs an oracle: measured, "Two lanes" above). The lanes leave lane B idle 7 min of
every 23; the queue keeps both GPUs busy: ~38-40 GPU-minutes per layer over two GPUs = ~19-20 min per layer, ~10.5-11 h for 33 layers against
~12.5 h (the 16.5 min/layer of the ideal assumes no contention and a 5 min chain). What it costs: whole-layer jobs mean a layer's K3 and K4 finish
~30 min after it starts (Hessians for up to 3 layers wait on disk, 3 x 5 GiB), the chain can wait up to one oracle slice (7 min) on G1, the manifest
refresh adds ~1 min of CPU per layer, and the display GPU does a K3 job half of the time (it is ~15% slower at K3 than device 1).
## Failure marker

`pipeline.ps1 -Run` deletes `<Root>\FAILED.txt` at start and writes one line on any stage failure (a step's non-zero exit, a refused
prerequisite, any exception): `2026-10-21T10:12:12 FAILED step=hessian-L3: exit 1 after 4.2 min: RuntimeError: layer 3: ... [log <path>]`
(the step, the exit code, the last Python exception line of the step's log, the log path; <= 600 characters). A monitor polls for the file.
## Tests

`tests\reference\test_refit_fallback.py` (CPU, ~1 min: the refit fallback and the EOL-tolerant job match); `tests\reference\test_lanes.ps1 [-Gpu | -GpuWorkers]` (CPU: verify exactly once under a race, the work-queue claim, STOP / FAILED markers, the plans; `-Gpu` / `-GpuWorkers`: the tiny model through both lanes / both workers, ~12-17 min); `tests\reference\test_no_evidence.py` (CPU, ~1 min, 39 checks: the no-evidence policy through finalize_layer, null groups, the oracle record, the allocation report); `tools\reference\qwen35moe\test_ref.py` (CPU: the streaming reference equals a plain HF forward on the tiny model, bit for bit with the eager
experts); `tiny_e2e.py` runs the whole chain on tiny-q35moe (`--device cuda --gpu-device 0` for the oracle).
Note for the owner of `tiny_golden.py`: its docstring says "eager per-expert MoE loop", but HF 5.18's default `experts_implementation` is
`grouped_mm`, which is what the goldens ran (`load_hf_model` sets the wrong attribute). Eager and grouped_mm differ by ~1 bf16 ulp per
layer; the streaming reference uses eager.


## fp8 KV calibration and the Rung 4 KL gate (2026-10-22)

Everything runs through the streaming reference, layer-major over many segments (`tools/reference/qwen35moe/ref_run.py`: each layer's
5.3 GB is read once for all segments, residuals for all segments stay on the card, GPU slices of 14 minutes under the `gpu` lock via
`tools/reference/qwen35moe/run_slices.ps1`, resumable: exit 75 = run me again). CPU tests: `test_ref_run.py`, `test_gate_q35.py`.

* **KV calibration**: `ref_run.py kvcalib` records, for the 12 full-attention layers, the per-kv-head |max| of K (after q/k-norm and rope,
  what the paged fp8 cache stores) and V (raw v_proj) with HF's own sdpa path unchanged (a registered attention function). 51 evenly spread
  segments of `tokens_calib.json` (151,028 tokens: 27 hessian-v2 of 2048, 24 SFT of up to 4096, 21 of them 4096 long) ->
  `E:\models\r4dx\q35moe\kvcalib.json` (+ `.meta.json` with the per-segment maxima). amax: K 7.6-21.1, V 4.7-27.4 (descale 0.010-0.061, the
  placeholder is 1.0), spread over segments <= 1.4x of the median except one V head (2.0x), and the *held-out* corpus (19,575 tokens, up to
  10k long) reaches 0.58-1.03x of the calibrated amax in every head: no saturation (448 / amax >= 16x headroom).
  `r4dx-convert --kv-calib` -> `ornith-1.5-122b-trellis-mix-kv.r4dx` (37020/37020 trellis tensors verified, 308 s CPU);
  `check_kvcalib_q35.py <container> kvcalib.json`: 24 descale tensors = amax/448, 0 placeholders; the first container is untouched (all 1.0).
* **KL gate**: held-out corpus `tools/reference/kl_corpus/build_chat_q35.py` (Ornith tokenizer: 12 thinking / tool chat conversations scored on
  model-turn tokens, english / thai / python / cpp, 4 raw 1024-token texts, one 10,261-token mix of them), `gate_q35.py prep|eval|attrib`.
  HF bf16 and HF fp32 ("truth") dumps from `ref_run.py logprobs --dtype bf16|fp32`; r4dx dumps from `r4dx_dump.ps1` (both cards, both locks,
  `tool_teacher_forced_logprobs.exe --layout trellis`: Prefill(1) + DecodeStep, 19.0 ms/row). Gate A and the noise-relative gate B as in
  `tools/reference/qwen4exp/gate_eval.py`; noise N = KL(truth || stock HF bf16) on the same rows.
* **Weight-only diagnostic**: `ref_run.py logprobs --mix <mix dir> [--mix-layers A-B] [--mix-match REGEX]` runs the HF bf16 stack with the trellis
  linears (all, or a layer range / name pattern) replaced by their reconstructions (`kl_mix.MixReader`), lm_head / router / shared expert
  stock. It separates what the bits cost from what the engine adds. (Attribute against the fp32 truth, not the stock bf16 dump: any change of
  weights re-rolls the bf16 rounding noise, which is part of every KL against a bf16 dump.)
* A dump of one attribution run (layers 0-42 only) came out as unnormalised noise once (row log-sum-exp ~ 5e5, positive "log-probs"); the
  identical command repeated gave a valid dump. Cause not found; `gate_q35.py` does not check normalisation, so look at one row's
  log-sum-exp when a KL is negative or absurd (all dumps used below were checked: |logsumexp| <= 3e-3 = fp16 rounding).

Results are in `docs/qwen35moe-kl-gate.md`.
### Allocator options for class p and measured depth sensitivity (not used for the shipped mix)

`alloc_q35.py allocate` also takes `--p-bf16 none|attn|gdn|all` (class p kept as plain bf16: bytes as the container's bf16 twin, no manifest
tensors, convert with `--keep-bf16` over the same bases), `--p-w8 ...` (w8a16 g64 byte model, PREDICTION ONLY: the converter cannot combine w8a16 with
trellis) and `--loss-scale <json {"layers": [48 multipliers]}>` (per-layer multiplier of the expert classes' loss; the measured one is
`tools/reference/qwen35moe/loss_scale_depth_2026-10-22.json`). Example, attention bf16 with the depth-measured loss:

    python tools\reference\qwen35moe\alloc_q35.py allocate --src E:\models\r4dx\q35moe\oracle-k3 --src E:\models\r4dx\q35moe\oracle-k4
        --out-dir E:\models\r4dx\q35moe\cand\m_attn_bf16 --p-bf16 attn --loss-scale tools\reference\qwen35moe\loss_scale_depth_2026-10-22.json

CPU test: `tools/reference/qwen35moe/test_alloc_q35.py`. What the variants predicted and why none was built: docs/qwen35moe-kl-gate.md,
"Options evaluated, not built".