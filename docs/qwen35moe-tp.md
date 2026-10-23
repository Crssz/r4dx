# `--tp 2` for Ornith-1.5-122B (`qwen3_5_moe`): implementation and results

Branch `claude/q35moe-tp2-f5ebfb` (cut from `qwen35moe` after the DFlash2 merge). The design is [`qwen35moe-tp-design.md`](qwen35moe-tp-design.md); this file records what was
built, how to run it, and what was measured (2026-10-23, two Radeon AI PRO R9700, Windows, no P2P). The layer split stays the default; `--tp 2` is opt-in and the two do not
fit together (26 GiB of weights per card each), so the mode is chosen at load.

## Result in one table

| gate | what | measured |
|---|---|---|
| **G1** (P0) | every tensor of the real container classified; `decode(rank slice) == slice(decode(full))` bit for bit | 74,799 tensors classify; 25.997 GiB per rank; 20 real linears (GDN, attention, experts at KB 3 and 4) bit-exact on both ranks; row-parallel partials add up, column-parallel outputs bit-equal |
| P1 | tiny model, two ranks on one device (emulate) vs the single-device model, bf16 and trellis | rank 0 == rank 1 bit for bit; 24 of 24 greedy tokens; layer-isolated GDN 0.0013 / attention 0.0007 mean/rms; grouped Tp2 decode MoE == host-routed MoE bit for bit |
| **G3'** (P2) | median per-rank noop decode step <= 0.62 x the layer split's GPU time | 6.839 ms vs 13.164 ms = **0.520** |
| **G4'** (P3) | Rung 4 KL(truth\|\|TP2) <= KL(layer split) + 0.004, top-1 >= 88.0% | 0.12364 vs 0.12361, top-1 88.61% vs 88.20% (real mode; 18,454 rows) |
| **G6'** (P3) | real == emulation | the real container cannot be emulated (two 26 GiB ranks on one 32 GiB card): two real-mode dumps are SHA-256 equal (17 files); tiny model real == emulate bit for bit (bf16, trellis) |
| **G7'** (P3) | decode >= 1.35 x the layer split, same session | 9.19 ms vs 14.67 ms per token = **1.60x** (108.7 vs 68.2 tok/s); prefill of 512 tokens 1.10x |
| G8' (P5) | soak 5 min, then 60 min, 0 aborts / divergences / TDRs | 5 min PASS (45 iterations, 5 canaries equal, 1.24 M all-reduces); 60 min: see the end of this file |

## What was built

| file | role |
|---|---|
| `src/model/qwen35moe_config.h` | `Qwen35MoeConfig::Shard(world, rank)` and `tp_world` / `tp_rank` / `VocabShardSize()`; the alignment invariants of design 2.2 (128-blocks for trellis, K % 256 for row-parallel) |
| `src/model/model_config.h` | `shared_expert_intermediate_size` (set by `Qwen35MoeConfig::ToModelConfig`) |
| `src/model/tp/tp_shard.cpp` | `RuleFor`'s qwen3_5_moe table: GDN / attention from the dense table, `moe.router` / `moe.shared_gate` replicated, `moe.shared.*` and every `moe.experts.{e}.*` TP-split; `mlp.*`, `mtp.*`, `rotation.*`, `.rotated` refused |
| `src/model/qwen35moe_tp_shard.h` | HIP-free `PlanTrellisLinear` / `PlanExpertClass`: the byte runs of one rank's slice of a trellis linear (words, fp16 sign vectors) |
| `src/model/qwen35moe_container_tp.cpp` | `Qwen35MoeContainer::LoadShard`: every rank gathers its slice of the one container (mixers through `ShardLoader`, experts into the stacked buffers at the rank's size), `LoadEmbeddingHost` (the one pinned embedding both ranks share) |
| `src/model/qwen35moe_stage.{h,cpp}` | `tp_comm`: all-reduce after the GDN out_proj, inside `AttentionLayer` (`AttnConfig::comm`), after the MoE combine; vocab-slice head merged over the ranks (greedy pair, row summary, full row); `SubmitBounder` units for TP prefill chunks; `EnableStepProfile` (GPU event spans) |
| `src/model/qwen35moe_model.{h,cpp}` | `tp_world` / `tp_rank` / `tp_device` / `tp_comm` / `shared_embed_host` options (ONE stage, no hand-off), `TpWarmup` |
| `src/model/qwen4exp/kernels/moe_trellis_decode.hip`, `moe_trellis_grouped.h`, `qwen35moe_grouped_decode.*` | grouped decode MoE for expert width 512 (`GroupedTrellisGemvGateUpQ35Tp2`, `GroupedSiluMulTrellisInputDownQ35Tp2`, `GroupedTrellisGemvDownQ35Tp2`, `GroupedQ35Tp2SetSplits`); the TP=1 launchers are the same templates with `INTER = 1024` |
| `src/model/qwen35moe_tp_model.{h,cpp}` | `Qwen35MoeTpModel`: the `TextModel` facade (copy-adapt of `Lfm2TpModel`): rank threads, real / emulate / noop, error state machine, recovery, fault injection, stats |
| `src/model/qwen35moe_local_text_model.cpp`, `tp_model.cpp` | `LoadTextModel`: `--tp 2` dispatches to the facade; `--tp 2 --pp 2` stays refused for qwen3_5_moe |
| `src/core/direct_storage.cpp` | loader creation / destruction under a process-wide mutex (see "Findings") |

Tests (`tests/model/qwen35moe/`): `test_q35moe_tp_shard` (CPU; config, rules, plans, the real container), `test_q35moe_tp_container` (rank views == plan-gathered TP=1
bytes), `test_model_q35moe_tp_tiny` (two rank threads on one device vs the single-device model), `test_model_q35moe_tp_facade` (dispatch, emulate vs layer split, Reset /
checkpoint, fault recovery; `R4DX_TP2GPU=1` with both cards adds real == emulate). Tools: `tool_q35moe_tp_step` (P2 gate), `tool_q35moe_tp_e2e` (layer split vs `--tp 2`
through `TextModel`, run each mode in its own process), `tool_q35moe_tp_soak` + `tools\tp\q35moe_soak.ps1` (TDR-watched), `tools\tp\q35moe_tp_dump.ps1` (the Rung 4 dump
with `--tp 2`), `tools\q35moe_server_tp_smoke.ps1` (server: layer split vs `--tp 2`, optional fault recovery).

## How to run

```
r4dx-cli    --model E:\models\r4dx\ornith-1.5-122b-trellis-mix-kv.r4dx --tokenizer-dir E:\models\r4dx\Ornith-1.5-122B --tp 2 --prompt "..." --stats
r4dx-server --model <same> --tokenizer-dir <same> --tp 2 --max-ctx 8192
```

Real mode: rank 0 = HIP device 1 (headless), rank 1 = device 0 (the desktop card), `HIP_VISIBLE_DEVICES` unset. Load ~74 s; 27.2 GiB used per card (weights 26.0 + arenas),
4.7 GiB free at `--max-ctx 4096`-8192. Take the `gpu` and `gpu-dev0` locks (`tools\with_lock.ps1`) for any job. `--tp-mode noop` times one rank without a peer.

## Measurements

* **Per-rank step (P2)**, `tool_q35moe_tp_step`, 512-token prompt, hipEvent spans, 3 x 64 steps: layer split 13.164 ms (7.224 + 5.169 + head 0.776); noop rank 0 6.839 ms
  (layers 6.457 + head 0.387, grouped decode MoE in all 48 layers). Split-group sweep of the Tp2 GEMVs (gate_up / down): 4/4 6.81, 8/4 6.66, 2/4 6.83, 4/2 6.84, 4/8 6.85,
  6/4 6.96 ms; the default stays 4/4 (it matches the host-routed order, and the choice changes rounding). Prefill, last 256-row chunk: 275.9 ms per rank vs 351.6 ms.
* **Real mode**: per-rank layer GPU span 7.28 ms + head 0.39 ms, host wall 9.1-9.2 ms: the all-reduces cost ~0.8 ms and the facade round trip, merges and host enqueue ~1.5 ms.
* **Rung 4** (`gate_report_tp2.md`): pooled KL(truth||x) layer split 0.12361, TP2 0.12364; by group english 0.05195 / 0.05104, thai 0.07397 / 0.07649, code 0.03944 / 0.03969.
* **Soak, 5 minutes** (`tools\tp\q35moe_soak.ps1 --minutes 5 --max-ctx 8192`): 45 iterations, 62,381 tokens, prefill ~570 tok/s and decode ~100-105 tok/s inside the mixed workload,
  1,244,792 + 56,384 all-reduces, max exchange wait 64 ms, 980 submit units, aborts 0, buffer drift 0 MiB, no TDR.

## Findings and deviations from the design

1. **The per-rank grouped decode kernels came before P2**, not in P4: without them the host-routed MoE would have measured a pessimistic step (a host sync per layer). They are the same templated kernels
   at `INTER = 512`; the M-row (verify) variants are NOT instantiated (P6).
2. **G4' / G6' cannot use emulation on the real container** (two 26 GiB ranks on one 32 GiB card). G4' ran in real mode; G6' is two real dumps compared by SHA-256, plus real == emulate on the tiny model.
3. **DirectStorage lifecycle race**: two rank threads creating / destroying their `DirectLoadScope` loaders at the same moment crashed inside `dstoragecore.dll` (short tiny-model loads; the 52 GB
   loads never hit it). Creation and destruction of the loaders now take a process-wide mutex; reads stay concurrent. 3 of 3 real-mode facade tests pass with DirectStorage on afterwards.
4. **Layer split then `--tp 2` in one process** makes TP ~2x slower (17.8 ms/token; the lm_head span on rank 1 is 6.6 ms) and leaves 0.7-0.85 GiB allocated after the split model is destroyed. Cause not found;
   production loads one mode per process, and `tool_q35moe_tp_e2e` is run once per mode.
5. TP decode still has ~1.5 ms/token of host overhead outside the GPU spans (facade command round trip, head merge, enqueue). The GDN recurrent decode kernel does not shrink with TP (32 workgroups).
6. Prefill gains little (1.10x TTFT at 512 tokens): the host-routed MoE is launch-bound and TP keeps the launch count. The grouped prefill MoE GEMM (plan section 8 item a) is the fix, independent of TP.

## Not done

* P4 per-rank tuning rows for the prefill shapes (`gemm_tuning_table_trellis_tp2.inc`): the rank shapes run `FallbackTuning` (correct, untuned); the prefill is launch-bound, so the expected gain is small.
* P6 DFlash2 under TP (see below when started).
